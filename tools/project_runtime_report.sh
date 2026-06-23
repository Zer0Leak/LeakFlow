#!/usr/bin/env bash

# Runtime correctness and performance report for modern C++ projects.
#
# Fedora tools:
#   sudo dnf install valgrind kcachegrind perf
#
# Typical fast usage:
#   tools/project_runtime_report.sh . reports build
#
# Run only Valgrind Memcheck:
#   RUN_VALGRIND=1 TOOL_TIMEOUT=600 tools/project_runtime_report.sh . reports build
#
# Run explicit test binary under Valgrind:
#   TEST_COMMAND='./build/tests/core/leakflow_tests' RUN_VALGRIND=1 tools/project_runtime_report.sh . reports build
#
# Run benchmark/profiling:
#   BENCH_COMMAND='./build/benchmarks/leakflow_bench' RUN_BENCHMARK=1 RUN_CACHEGRIND=1 RUN_CALLGRIND=1 tools/project_runtime_report.sh . reports build
#
# Run sanitizer build folders:
#   RUN_SANITIZERS=1 ASAN_BUILD_DIR=build-asan UBSAN_BUILD_DIR=build-ubsan TSAN_BUILD_DIR=build-tsan tools/project_runtime_report.sh . reports build

set -euo pipefail

PROJECT_DIR="${1:-.}"
REPORT_ROOT="${2:-reports}"
BUILD_DIR="${3:-build}"

REPORT_FILE="$REPORT_ROOT/project_runtime_report.md"
REPORT_DIR="$REPORT_ROOT/.runtime_artifacts"

# Timeouts.
# TOOL_TIMEOUT applies to heavy tools: valgrind, perf, benchmarks, sanitizer commands.
# CTEST_TIMEOUT is passed to ctest.
TOOL_TIMEOUT="${TOOL_TIMEOUT:-300}"
CTEST_TIMEOUT="${CTEST_TIMEOUT:-60}"

# Main commands.
TEST_COMMAND="${TEST_COMMAND:-ctest --test-dir $BUILD_DIR --output-on-failure --timeout $CTEST_TIMEOUT}"
BENCH_COMMAND="${BENCH_COMMAND:-}"

# Runtime layers.
# Heavy tools are OFF by default to avoid "frozen" reports.
RUN_SANITIZERS="${RUN_SANITIZERS:-0}"
RUN_VALGRIND="${RUN_VALGRIND:-0}"
RUN_HELGRIND="${RUN_HELGRIND:-0}"
RUN_DRD="${RUN_DRD:-0}"
RUN_CACHEGRIND="${RUN_CACHEGRIND:-0}"
RUN_CALLGRIND="${RUN_CALLGRIND:-0}"
RUN_MASSIF="${RUN_MASSIF:-0}"
RUN_PERF_STAT="${RUN_PERF_STAT:-0}"
RUN_PERF_RECORD="${RUN_PERF_RECORD:-0}"
RUN_BENCHMARK="${RUN_BENCHMARK:-0}"

# Sanitizer build directories.
ASAN_BUILD_DIR="${ASAN_BUILD_DIR:-build-asan}"
UBSAN_BUILD_DIR="${UBSAN_BUILD_DIR:-build-ubsan}"
TSAN_BUILD_DIR="${TSAN_BUILD_DIR:-build-tsan}"
MSAN_BUILD_DIR="${MSAN_BUILD_DIR:-build-msan}"

# Sanitizer commands.
# If empty, the script uses ctest against the corresponding build dir.
ASAN_COMMAND="${ASAN_COMMAND:-}"
UBSAN_COMMAND="${UBSAN_COMMAND:-}"
TSAN_COMMAND="${TSAN_COMMAND:-}"
MSAN_COMMAND="${MSAN_COMMAND:-}"

# Quality gates.
MAX_VALGRIND_ERRORS="${MAX_VALGRIND_ERRORS:-0}"
MAX_HELGRIND_ERRORS="${MAX_HELGRIND_ERRORS:-0}"
MAX_DRD_ERRORS="${MAX_DRD_ERRORS:-0}"
MAX_SANITIZER_FAILURES="${MAX_SANITIZER_FAILURES:-0}"

# perf options.
PERF_RECORD_SECONDS="${PERF_RECORD_SECONDS:-}"
PERF_RECORD_FREQ="${PERF_RECORD_FREQ:-99}"

cd "$PROJECT_DIR"

mkdir -p "$REPORT_ROOT"
mkdir -p "$REPORT_DIR"

now="$(date '+%Y-%m-%d %H:%M:%S')"

have_cmd() {
    command -v "$1" >/dev/null 2>&1
}

run_with_timeout() {
    if have_cmd timeout; then
        timeout "$TOOL_TIMEOUT" "$@"
    else
        "$@"
    fi
}

section() {
    echo
    echo "## $1"
    echo
}

codeblock_begin() {
    echo '```text'
}

codeblock_end() {
    echo '```'
    echo
}

run_block() {
    local title="$1"
    shift

    section "$title"
    codeblock_begin
    "$@" || true
    codeblock_end
}

run_shell_command_logged() {
    local log_file="$1"
    local command_text="$2"

    echo "Command:"
    echo "$command_text"
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout bash -lc "$command_text" 2>&1 | tee "$log_file" || true
}

command_status_from_log() {
    local log="$1"

    if [[ ! -f "$log" ]]; then
        echo "unknown"
        return 0
    fi

    if grep -Eiq 'AddressSanitizer|UndefinedBehaviorSanitizer|ThreadSanitizer|MemorySanitizer|runtime error:|data race|heap-use-after-free|stack-use-after-return|stack-buffer-overflow|heap-buffer-overflow|SEGV|ABORTING' "$log"; then
        echo "fail"
    else
        echo "pass"
    fi
}

quality_gate_status() {
    local name="$1"
    local value="$2"
    local limit="$3"
    local direction="$4"

    if [[ "$value" == "unknown" || -z "$value" ]]; then
        printf "| %s | unknown | %s | ⚠️ not measured |\n" "$name" "$limit"
        return 0
    fi

    local value_int="${value%.*}"

    if [[ "$direction" == "max" ]]; then
        if (( value_int <= limit )); then
            printf "| %s | %s | <= %s | ✅ pass |\n" "$name" "$value" "$limit"
        else
            printf "| %s | %s | <= %s | ❌ fail |\n" "$name" "$value" "$limit"
        fi
    else
        if (( value_int >= limit )); then
            printf "| %s | %s | >= %s | ✅ pass |\n" "$name" "$value" "$limit"
        else
            printf "| %s | %s | >= %s | ❌ fail |\n" "$name" "$value" "$limit"
        fi
    fi
}

run_sanitizer_ctest_or_command() {
    local name="$1"
    local build_dir="$2"
    local explicit_command="$3"
    local log="$REPORT_DIR/${name,,}.txt"

    if [[ "$RUN_SANITIZERS" != "1" ]]; then
        echo "Skipped. Enable with RUN_SANITIZERS=1."
        return 0
    fi

    if [[ -n "$explicit_command" ]]; then
        run_shell_command_logged "$log" "$explicit_command"
        return 0
    fi

    if [[ ! -d "$build_dir" ]]; then
        echo "$name build directory not found: $build_dir"
        echo
        echo "Expected one of:"
        echo "- provide ${name}_COMMAND"
        echo "- or create build dir: $build_dir"
        return 0
    fi

    local command_text="ctest --test-dir '$build_dir' --output-on-failure --timeout '$CTEST_TIMEOUT'"
    run_shell_command_logged "$log" "$command_text"
}

run_asan() {
    run_sanitizer_ctest_or_command "ASAN" "$ASAN_BUILD_DIR" "$ASAN_COMMAND"
}

run_ubsan() {
    run_sanitizer_ctest_or_command "UBSAN" "$UBSAN_BUILD_DIR" "$UBSAN_COMMAND"
}

run_tsan() {
    run_sanitizer_ctest_or_command "TSAN" "$TSAN_BUILD_DIR" "$TSAN_COMMAND"
}

run_msan() {
    run_sanitizer_ctest_or_command "MSAN" "$MSAN_BUILD_DIR" "$MSAN_COMMAND"
}

sanitizer_failure_count() {
    local total=0

    for name in asan ubsan tsan msan; do
        local log="$REPORT_DIR/$name.txt"

        if [[ ! -f "$log" ]]; then
            continue
        fi

        local status
        status="$(command_status_from_log "$log")"

        if [[ "$status" == "fail" ]]; then
            total=$((total + 1))
        fi
    done

    echo "$total"
}

run_valgrind_memcheck() {
    if [[ "$RUN_VALGRIND" != "1" ]]; then
        echo "Skipped. Enable with RUN_VALGRIND=1."
        echo
        echo "Recommended:"
        echo "RUN_VALGRIND=1 TOOL_TIMEOUT=600 tools/project_runtime_report.sh . reports build"
        echo
        echo "Better with direct test binary:"
        echo "TEST_COMMAND='./build/tests/core/leakflow_tests' RUN_VALGRIND=1 tools/project_runtime_report.sh . reports build"
        return 0
    fi

    if ! have_cmd valgrind; then
        echo "valgrind not installed."
        echo "Install: sudo dnf install valgrind"
        return 0
    fi

    local log="$REPORT_DIR/valgrind-memcheck.txt"

    echo "Command:"
    echo "valgrind --tool=memcheck --trace-children=yes --leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=99 bash -lc \"$TEST_COMMAND\""
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout valgrind \
        --tool=memcheck \
        --trace-children=yes \
        --leak-check=full \
        --show-leak-kinds=all \
        --track-origins=yes \
        --error-exitcode=99 \
        bash -lc "$TEST_COMMAND" \
        2>&1 | tee "$log" || true
}

count_valgrind_errors() {
    local log="$REPORT_DIR/valgrind-memcheck.txt"

    if [[ ! -f "$log" ]]; then
        echo "unknown"
        return 0
    fi

    awk '
        /ERROR SUMMARY:/ {
            value=$4
            found=1
        }
        END {
            if (found) print value;
            else print "unknown";
        }
    ' "$log"
}

run_helgrind() {
    if [[ "$RUN_HELGRIND" != "1" ]]; then
        echo "Skipped. Enable with RUN_HELGRIND=1."
        return 0
    fi

    if ! have_cmd valgrind; then
        echo "valgrind not installed."
        return 0
    fi

    local log="$REPORT_DIR/valgrind-helgrind.txt"

    echo "Command:"
    echo "valgrind --tool=helgrind --trace-children=yes --error-exitcode=99 bash -lc \"$TEST_COMMAND\""
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout valgrind \
        --tool=helgrind \
        --trace-children=yes \
        --error-exitcode=99 \
        bash -lc "$TEST_COMMAND" \
        2>&1 | tee "$log" || true
}

count_helgrind_errors() {
    local log="$REPORT_DIR/valgrind-helgrind.txt"

    if [[ ! -f "$log" ]]; then
        echo "unknown"
        return 0
    fi

    awk '
        /ERROR SUMMARY:/ {
            value=$4
            found=1
        }
        END {
            if (found) print value;
            else print "unknown";
        }
    ' "$log"
}

run_drd() {
    if [[ "$RUN_DRD" != "1" ]]; then
        echo "Skipped. Enable with RUN_DRD=1."
        return 0
    fi

    if ! have_cmd valgrind; then
        echo "valgrind not installed."
        return 0
    fi

    local log="$REPORT_DIR/valgrind-drd.txt"

    echo "Command:"
    echo "valgrind --tool=drd --trace-children=yes --error-exitcode=99 bash -lc \"$TEST_COMMAND\""
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout valgrind \
        --tool=drd \
        --trace-children=yes \
        --error-exitcode=99 \
        bash -lc "$TEST_COMMAND" \
        2>&1 | tee "$log" || true
}

count_drd_errors() {
    local log="$REPORT_DIR/valgrind-drd.txt"

    if [[ ! -f "$log" ]]; then
        echo "unknown"
        return 0
    fi

    awk '
        /ERROR SUMMARY:/ {
            value=$4
            found=1
        }
        END {
            if (found) print value;
            else print "unknown";
        }
    ' "$log"
}

run_cachegrind() {
    if [[ "$RUN_CACHEGRIND" != "1" ]]; then
        echo "Skipped. Enable with RUN_CACHEGRIND=1."
        echo
        echo "Recommended:"
        echo "BENCH_COMMAND='./build/benchmarks/leakflow_bench' RUN_CACHEGRIND=1 tools/project_runtime_report.sh . reports build"
        return 0
    fi

    if ! have_cmd valgrind; then
        echo "valgrind not installed."
        return 0
    fi

    if [[ -z "$BENCH_COMMAND" ]]; then
        echo "BENCH_COMMAND is empty."
        echo "Set it to a benchmark or representative workload."
        return 0
    fi

    local out_pattern="$REPORT_DIR/cachegrind.out.%p"
    local summary="$REPORT_DIR/cachegrind-summary.txt"

    echo "Command:"
    echo "valgrind --tool=cachegrind --branch-sim=yes --trace-children=yes --cachegrind-out-file=$out_pattern bash -lc \"$BENCH_COMMAND\""
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout valgrind \
        --tool=cachegrind \
        --branch-sim=yes \
        --trace-children=yes \
        --cachegrind-out-file="$out_pattern" \
        bash -lc "$BENCH_COMMAND" \
        >/dev/null 2>&1 || true

    : > "$summary"

    if have_cmd cg_annotate; then
        for f in "$REPORT_DIR"/cachegrind.out.*; do
            [[ -f "$f" ]] || continue
            {
                echo "===== $f ====="
                cg_annotate "$f" 2>/dev/null | head -120
                echo
            } >> "$summary" || true
        done

        if [[ -s "$summary" ]]; then
            cat "$summary"
        else
            echo "No cachegrind output generated."
        fi
    else
        echo "cg_annotate not found."
        echo "Raw files:"
        ls -1 "$REPORT_DIR"/cachegrind.out.* 2>/dev/null || true
    fi
}

run_callgrind() {
    if [[ "$RUN_CALLGRIND" != "1" ]]; then
        echo "Skipped. Enable with RUN_CALLGRIND=1."
        echo
        echo "Recommended:"
        echo "BENCH_COMMAND='./build/benchmarks/leakflow_bench' RUN_CALLGRIND=1 tools/project_runtime_report.sh . reports build"
        return 0
    fi

    if ! have_cmd valgrind; then
        echo "valgrind not installed."
        return 0
    fi

    if [[ -z "$BENCH_COMMAND" ]]; then
        echo "BENCH_COMMAND is empty."
        echo "Set it to a benchmark or representative workload."
        return 0
    fi

    local out_pattern="$REPORT_DIR/callgrind.out.%p"
    local summary="$REPORT_DIR/callgrind-summary.txt"

    echo "Command:"
    echo "valgrind --tool=callgrind --trace-children=yes --callgrind-out-file=$out_pattern bash -lc \"$BENCH_COMMAND\""
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout valgrind \
        --tool=callgrind \
        --trace-children=yes \
        --callgrind-out-file="$out_pattern" \
        bash -lc "$BENCH_COMMAND" \
        >/dev/null 2>&1 || true

    : > "$summary"

    if have_cmd callgrind_annotate; then
        for f in "$REPORT_DIR"/callgrind.out.*; do
            [[ -f "$f" ]] || continue
            {
                echo "===== $f ====="
                callgrind_annotate "$f" 2>/dev/null | head -120
                echo
            } >> "$summary" || true
        done

        if [[ -s "$summary" ]]; then
            cat "$summary"
            echo
            echo "GUI inspection:"
            echo "kcachegrind $REPORT_DIR/callgrind.out.<pid>"
        else
            echo "No callgrind output generated."
        fi
    else
        echo "callgrind_annotate not found."
        echo "Raw files:"
        ls -1 "$REPORT_DIR"/callgrind.out.* 2>/dev/null || true
    fi
}

run_massif() {
    if [[ "$RUN_MASSIF" != "1" ]]; then
        echo "Skipped. Enable with RUN_MASSIF=1."
        echo
        echo "Recommended:"
        echo "BENCH_COMMAND='./build/benchmarks/leakflow_bench' RUN_MASSIF=1 tools/project_runtime_report.sh . reports build"
        return 0
    fi

    if ! have_cmd valgrind; then
        echo "valgrind not installed."
        return 0
    fi

    if [[ -z "$BENCH_COMMAND" ]]; then
        echo "BENCH_COMMAND is empty."
        echo "Set it to a benchmark or representative workload."
        return 0
    fi

    local out_pattern="$REPORT_DIR/massif.out.%p"
    local summary="$REPORT_DIR/massif-summary.txt"

    echo "Command:"
    echo "valgrind --tool=massif --trace-children=yes --massif-out-file=$out_pattern bash -lc \"$BENCH_COMMAND\""
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout valgrind \
        --tool=massif \
        --trace-children=yes \
        --massif-out-file="$out_pattern" \
        bash -lc "$BENCH_COMMAND" \
        >/dev/null 2>&1 || true

    : > "$summary"

    if have_cmd ms_print; then
        for f in "$REPORT_DIR"/massif.out.*; do
            [[ -f "$f" ]] || continue
            {
                echo "===== $f ====="
                ms_print "$f" 2>/dev/null | head -160
                echo
            } >> "$summary" || true
        done

        if [[ -s "$summary" ]]; then
            cat "$summary"
        else
            echo "No massif output generated."
        fi
    else
        echo "ms_print not found."
        echo "Raw files:"
        ls -1 "$REPORT_DIR"/massif.out.* 2>/dev/null || true
    fi
}

run_perf_stat() {
    if [[ "$RUN_PERF_STAT" != "1" ]]; then
        echo "Skipped. Enable with RUN_PERF_STAT=1."
        return 0
    fi

    if ! have_cmd perf; then
        echo "perf not installed."
        echo "Install: sudo dnf install perf"
        return 0
    fi

    if [[ -z "$BENCH_COMMAND" ]]; then
        echo "BENCH_COMMAND is empty."
        echo "Set it to a benchmark or representative workload."
        return 0
    fi

    local log="$REPORT_DIR/perf-stat.txt"

    echo "Command:"
    echo "perf stat -d bash -lc \"$BENCH_COMMAND\""
    echo
    echo "Timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout perf stat -d bash -lc "$BENCH_COMMAND" 2>&1 | tee "$log" || true
}

run_perf_record() {
    if [[ "$RUN_PERF_RECORD" != "1" ]]; then
        echo "Skipped. Enable with RUN_PERF_RECORD=1."
        return 0
    fi

    if ! have_cmd perf; then
        echo "perf not installed."
        echo "Install: sudo dnf install perf"
        return 0
    fi

    if [[ -z "$BENCH_COMMAND" ]]; then
        echo "BENCH_COMMAND is empty."
        echo "Set it to a benchmark or representative workload."
        return 0
    fi

    local data="$REPORT_DIR/perf.data"
    local report="$REPORT_DIR/perf-report.txt"

    echo "Command:"
    echo "perf record -F $PERF_RECORD_FREQ -g -o $data -- bash -lc \"$BENCH_COMMAND\""
    echo

    if [[ -n "$PERF_RECORD_SECONDS" ]]; then
        timeout "$PERF_RECORD_SECONDS" \
            perf record -F "$PERF_RECORD_FREQ" -g -o "$data" -- bash -lc "$BENCH_COMMAND" \
            >/dev/null 2>&1 || true
    else
        run_with_timeout perf record -F "$PERF_RECORD_FREQ" -g -o "$data" -- bash -lc "$BENCH_COMMAND" \
            >/dev/null 2>&1 || true
    fi

    if [[ -f "$data" ]]; then
        perf report --stdio -i "$data" > "$report" 2>/dev/null || true

        if [[ -s "$report" ]]; then
            head -160 "$report"
        else
            echo "perf.data generated, but perf report produced no text output."
            echo "Inspect manually:"
            echo "perf report -i $data"
        fi
    else
        echo "perf.data was not generated."
        echo "This may be blocked by kernel perf_event_paranoid settings."
    fi
}

run_benchmark() {
    if [[ "$RUN_BENCHMARK" != "1" ]]; then
        echo "Skipped. Enable with RUN_BENCHMARK=1."
        return 0
    fi

    if [[ -z "$BENCH_COMMAND" ]]; then
        echo "BENCH_COMMAND is empty."
        echo "Set it to your benchmark executable."
        return 0
    fi

    local log="$REPORT_DIR/benchmark.txt"

    run_shell_command_logged "$log" "$BENCH_COMMAND"
}

runtime_recommendations() {
    cat <<EOF
Recommended interpretation:

| Signal | Meaning |
|---|---|
| ASan failure | Memory safety bug: out-of-bounds, use-after-free, stack/heap issue. |
| UBSan failure | Undefined behavior: invalid shift, overflow, null/misaligned access, invalid cast, etc. |
| TSan failure | Data race or unsafe concurrent access. |
| MSan failure | Uninitialized memory read. |
| Valgrind Memcheck error | Invalid read/write, leak, uninitialized value, or lifetime issue. |
| Helgrind/DRD error | Threading synchronization problem. |
| Cachegrind hotspot | Function has high instruction/cache/branch cost. |
| Callgrind hotspot | Expensive call path or abstraction overhead. |
| Massif peak | Heap growth or excessive allocation pressure. |
| perf hotspot | Real CPU hotspot under native execution. |

For LeakFlow, prioritize runtime checks around:

- pipeline push/pull execution
- element lifecycle: create, configure, start, stop, destroy
- property parsing and type conversion
- buffer ownership and trace/tensor movement
- plugin loading/unloading
- NPY/trace loading
- CPU/GPU transfer boundaries
- concurrent queues and scheduler behavior
- error paths and early returns
EOF
}

build_instructions() {
    cat <<EOF
Suggested sanitizer builds:

ASan + UBSan:

cmake -S . -B build-asan \\
  -DCMAKE_BUILD_TYPE=Debug \\
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \\
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g" \\
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"

cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure --timeout $CTEST_TIMEOUT

TSan:

cmake -S . -B build-tsan \\
  -DCMAKE_BUILD_TYPE=Debug \\
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \\
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer -O1 -g" \\
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"

cmake --build build-tsan -j
ctest --test-dir build-tsan --output-on-failure --timeout $CTEST_TIMEOUT

Coverage build should remain separate from sanitizer builds.
Do not mix ASan and TSan in the same build.
EOF
}

{
    echo "# Modern C++ Runtime Quality Report"
    echo
    echo "- Generated: $now"
    echo "- Project: $(pwd)"
    echo "- Build directory: $BUILD_DIR"
    echo "- Report artifacts: $REPORT_DIR"
    echo

    section "Runtime Configuration"

    echo "| Variable | Value |"
    echo "|---|---|"
    echo "| TEST_COMMAND | \`$TEST_COMMAND\` |"
    echo "| BENCH_COMMAND | \`${BENCH_COMMAND:-not set}\` |"
    echo "| TOOL_TIMEOUT | $TOOL_TIMEOUT seconds |"
    echo "| CTEST_TIMEOUT | $CTEST_TIMEOUT seconds |"
    echo "| RUN_SANITIZERS | $RUN_SANITIZERS |"
    echo "| RUN_VALGRIND | $RUN_VALGRIND |"
    echo "| RUN_HELGRIND | $RUN_HELGRIND |"
    echo "| RUN_DRD | $RUN_DRD |"
    echo "| RUN_CACHEGRIND | $RUN_CACHEGRIND |"
    echo "| RUN_CALLGRIND | $RUN_CALLGRIND |"
    echo "| RUN_MASSIF | $RUN_MASSIF |"
    echo "| RUN_PERF_STAT | $RUN_PERF_STAT |"
    echo "| RUN_PERF_RECORD | $RUN_PERF_RECORD |"
    echo "| RUN_BENCHMARK | $RUN_BENCHMARK |"

    section "Runtime Gates"

    echo "| Gate | Threshold |"
    echo "|---|---:|"
    echo "| Max sanitizer failures | $MAX_SANITIZER_FAILURES |"
    echo "| Max Valgrind Memcheck errors | $MAX_VALGRIND_ERRORS |"
    echo "| Max Helgrind errors | $MAX_HELGRIND_ERRORS |"
    echo "| Max DRD errors | $MAX_DRD_ERRORS |"

    run_block "Level 4A - AddressSanitizer Run" run_asan
    run_block "Level 4B - UndefinedBehaviorSanitizer Run" run_ubsan
    run_block "Level 4C - ThreadSanitizer Run" run_tsan
    run_block "Level 4D - MemorySanitizer Run" run_msan

    run_block "Level 4E - Valgrind Memcheck" run_valgrind_memcheck
    run_block "Level 4F - Valgrind Helgrind" run_helgrind
    run_block "Level 4G - Valgrind DRD" run_drd

    run_block "Level 5A - Benchmark Run" run_benchmark
    run_block "Level 5B - Cachegrind" run_cachegrind
    run_block "Level 5C - Callgrind" run_callgrind
    run_block "Level 5D - Massif Heap Profiler" run_massif
    run_block "Level 5E - perf stat" run_perf_stat
    run_block "Level 5F - perf record" run_perf_record

    section "Runtime Quality Gate Results"

    sanitizer_failures="$(sanitizer_failure_count)"
    valgrind_errors="$(count_valgrind_errors)"
    helgrind_errors="$(count_helgrind_errors)"
    drd_errors="$(count_drd_errors)"

    echo "| Gate | Current | Target | Status |"
    echo "|---|---:|---:|---|"
    quality_gate_status "sanitizer failures" "$sanitizer_failures" "$MAX_SANITIZER_FAILURES" "max"
    quality_gate_status "Valgrind Memcheck errors" "$valgrind_errors" "$MAX_VALGRIND_ERRORS" "max"
    quality_gate_status "Helgrind errors" "$helgrind_errors" "$MAX_HELGRIND_ERRORS" "max"
    quality_gate_status "DRD errors" "$drd_errors" "$MAX_DRD_ERRORS" "max"

    section "Runtime Recommendations"
    runtime_recommendations

    section "Suggested Sanitizer Builds"
    codeblock_begin
    build_instructions
    codeblock_end

    section "Trust Notes"

    cat <<EOF
Runtime report trust levels:

| Tool | Trust | Notes |
|---|---|---|
| ASan | High | Very useful for memory safety bugs; requires sanitizer build. |
| UBSan | High | Very useful for undefined behavior; requires sanitizer build. |
| TSan | High | Strong data-race detector, but can be noisy and slow. |
| MSan | High but hard to use | Requires all code and libraries to be instrumented. |
| Valgrind Memcheck | High | Slow but deep memory/lifetime checking; does not require sanitizer build. |
| Helgrind/DRD | Medium-high | Useful thread checkers; TSan is usually better for modern Clang/GCC builds. |
| Cachegrind | Medium-high | Excellent for instruction/cache simulation; not real wall-clock performance. |
| Callgrind | Medium-high | Excellent for call-path cost attribution. |
| Massif | Medium-high | Good for heap growth and allocation pressure. |
| perf | High | Native profiler; depends on OS permissions and workload representativeness. |
| Benchmark output | High only if benchmarks are stable | Requires careful benchmark design. |

Use this report together with project_quality_report.md.
Static analysis finds potential defects.
Runtime analysis finds actual executed defects and performance behavior.
EOF

} > "$REPORT_FILE"

echo "Runtime report written to: $REPORT_FILE"
echo "Artifacts written to: $REPORT_DIR"
