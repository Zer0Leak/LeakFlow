#!/usr/bin/env bash

# Modern C++23 code quality report.
#
# Fedora:
#   sudo dnf install cloc cppcheck clang-tools-extra lcov
#   pip install lizard gcovr
#
# Basic usage:
#   tools/project_quality_report.sh
#
# Explicit usage:
#   tools/project_quality_report.sh . reports build
#
# Fast daily usage:
#   RUN_CPPCHECK=0 RUN_CLANG_TIDY=0 RUN_GCOVR=0 tools/project_quality_report.sh . reports build
#
# Practical serious usage:
#   RUN_CPPCHECK=1 RUN_CLANG_TIDY=1 RUN_GCOVR=0 MAX_TIDY_FILES=5 TOOL_TIMEOUT=120 tools/project_quality_report.sh . reports build
#
# Deep CI usage:
#   RUN_CPPCHECK=1 RUN_CLANG_TIDY=1 RUN_GCOVR=1 MAX_TIDY_FILES=0 TOOL_TIMEOUT=600 tools/project_quality_report.sh . reports build-coverage

set -euo pipefail

PROJECT_DIR="${1:-.}"
REPORT_ROOT="${2:-reports}"
BUILD_DIR="${3:-build}"

REPORT_FILE="$REPORT_ROOT/project_quality_report.md"
REPORT_DIR="$REPORT_ROOT/.quality_artifacts"

# Quality thresholds.
# Override from shell, e.g.:
#   MAX_CCN=12 MIN_LINE_COVERAGE=75 tools/project_quality_report.sh
MAX_CCN="${MAX_CCN:-10}"
MAX_FUNCTION_LENGTH="${MAX_FUNCTION_LENGTH:-80}"
MAX_FILE_LENGTH="${MAX_FILE_LENGTH:-600}"
MIN_LINE_COVERAGE="${MIN_LINE_COVERAGE:-70}"
MIN_BRANCH_COVERAGE="${MIN_BRANCH_COVERAGE:-50}"
MAX_CLANG_TIDY_WARNINGS="${MAX_CLANG_TIDY_WARNINGS:-0}"
MAX_CPPCHECK_ERRORS="${MAX_CPPCHECK_ERRORS:-0}"

# Heavy-tool switches.
# Defaults are chosen to avoid freezing on libtorch-heavy projects.
RUN_CPPCHECK="${RUN_CPPCHECK:-1}"
RUN_CLANG_TIDY="${RUN_CLANG_TIDY:-1}"
RUN_GCOVR="${RUN_GCOVR:-0}"

# 0 means no limit. For libtorch-heavy projects, keep this small for daily use.
MAX_TIDY_FILES="${MAX_TIDY_FILES:-10}"

# Per-tool timeout in seconds.
TOOL_TIMEOUT="${TOOL_TIMEOUT:-180}"

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

subsection() {
    echo
    echo "### $1"
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

count_files() {
    find . \
        -path './.git' -prune -o \
        -path './build' -prune -o \
        -path './build-*' -prune -o \
        -path './cmake-build-*' -prune -o \
        -path './.cache' -prune -o \
        -path './.vscode' -prune -o \
        -path './reports' -prune -o \
        -path './_deps' -prune -o \
        -path './third_party' -prune -o \
        -path './external' -prune -o \
        -type f -print
}

cpp_sources() {
    count_files | grep -E '\.(c|cc|cpp|cxx)$' || true
}

cpp_headers() {
    count_files | grep -E '\.(h|hh|hpp|hxx)$' || true
}

cpp_files() {
    count_files | grep -E '\.(c|cc|cpp|cxx|h|hh|hpp|hxx)$' || true
}

test_files() {
    count_files | grep -Ei '(^|/)(test|tests|spec|specs)(/|_)|(_test|_tests|Test|Tests)\.(c|cc|cpp|cxx|h|hpp|hxx)$' || true
}

compile_commands_path() {
    if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
        echo "$BUILD_DIR/compile_commands.json"
    elif [[ -f "compile_commands.json" ]]; then
        echo "compile_commands.json"
    else
        echo ""
    fi
}

compile_db_dir() {
    local ccdb
    ccdb="$(compile_commands_path)"

    if [[ -z "$ccdb" ]]; then
        echo ""
        return 0
    fi

    dirname "$ccdb"
}

filtered_compile_db_dir() {
    local ccdb
    ccdb="$(compile_commands_path)"

    if [[ -z "$ccdb" ]]; then
        echo ""
        return 0
    fi

    local filtered_dir="$REPORT_DIR/compile_commands_filtered"
    local filtered_db="$filtered_dir/compile_commands.json"

    mkdir -p "$filtered_dir"

    python3 - "$ccdb" "$filtered_db" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with src.open("r", encoding="utf-8") as f:
    data = json.load(f)

excluded_parts = [
    "/build/_deps/",
    "/_deps/",
    "/third_party/",
    "/external/",
]

filtered = []

for entry in data:
    file_path = entry.get("file", "")
    command = entry.get("command", "")
    directory = entry.get("directory", "")

    blob = f"{file_path}\n{command}\n{directory}"

    if any(part in blob for part in excluded_parts):
        continue

    filtered.append(entry)

with dst.open("w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(dst.parent)
PY
}

largest_code_files() {
    cpp_files \
        | while IFS= read -r f; do
            lines="$(wc -l < "$f" 2>/dev/null || echo 0)"
            if (( lines > MAX_FILE_LENGTH )); then
                printf "%8d  %s\n" "$lines" "$f"
            fi
        done \
        | sort -nr
}

all_code_files_by_size() {
    cpp_files \
        | while IFS= read -r f; do
            lines="$(wc -l < "$f" 2>/dev/null || echo 0)"
            printf "%8d  %s\n" "$lines" "$f"
        done \
        | sort -nr \
        | head -40
}

git_churn_summary() {
    echo "Last 30 days:"
    git log --since='30 days ago' --shortstat --oneline || true

    echo
    echo "Most changed files, last 90 days:"
    git log --since='90 days ago' --name-only --pretty=format: \
        | grep -E '\.(c|cc|cpp|cxx|h|hh|hpp|hxx)$' \
        | sort \
        | uniq -c \
        | sort -nr \
        | head -30 || true
}

run_cloc() {
    if have_cmd cloc; then
        cloc . \
            --exclude-dir=.git,build,build-asan,build-ubsan,build-tsan,build-msan,build-coverage,cmake-build-debug,cmake-build-release,.cache,.vscode,reports,_deps,third_party,external
    else
        echo "cloc not installed."
        echo "Install: sudo dnf install cloc"
    fi
}

run_lizard_full() {
    if have_cmd lizard; then
        lizard . \
            -x "./build/*" \
            -x "./build-*/*" \
            -x "./cmake-build-*/*" \
            -x "./.git/*" \
            -x "./.cache/*" \
            -x "./reports/*" \
            -x "./_deps/*" \
            -x "./third_party/*" \
            -x "./external/*"
    else
        echo "lizard not installed."
        echo "Install: pip install lizard"
    fi
}

run_lizard_refactor_candidates() {
    if have_cmd lizard; then
        lizard . \
            -C "$MAX_CCN" \
            -L "$MAX_FUNCTION_LENGTH" \
            -x "./build/*" \
            -x "./build-*/*" \
            -x "./cmake-build-*/*" \
            -x "./.git/*" \
            -x "./.cache/*" \
            -x "./reports/*" \
            -x "./_deps/*" \
            -x "./third_party/*" \
            -x "./external/*"
    else
        echo "lizard not installed."
        echo "Install: pip install lizard"
    fi
}

run_cppcheck() {
    if [[ "$RUN_CPPCHECK" != "1" ]]; then
        echo "Skipped. Enable with RUN_CPPCHECK=1."
        return 0
    fi

    if ! have_cmd cppcheck; then
        echo "cppcheck not installed."
        echo "Install: sudo dnf install cppcheck"
        return 0
    fi

    local cppcheck_xml="$REPORT_DIR/cppcheck.xml"
    local cppcheck_txt="$REPORT_DIR/cppcheck.txt"

    echo "Running cppcheck with timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout cppcheck \
        --enable=warning,style,performance,portability \
        --std=c++23 \
        --inline-suppr \
        --force \
        --suppress=missingIncludeSystem \
        -i build \
	-i build/_deps \
        -i build/_deps/fmt-src \
        -i build/_deps/cnpypp-src \
        -i build-asan \
        -i build-ubsan \
        -i build-tsan \
        -i build-msan \
        -i build-coverage \
        -i cmake-build-debug \
        -i cmake-build-release \
        -i reports \
        -i third_party \
        -i external \
        -i _deps \
        --xml \
        . \
        1>"$cppcheck_txt" \
        2>"$cppcheck_xml" || true

    run_with_timeout cppcheck \
        --enable=warning,style,performance,portability \
        --std=c++23 \
        --inline-suppr \
        --force \
        --suppress=missingIncludeSystem \
        -i build \
	-i build/_deps \
        -i build/_deps/fmt-src \
        -i build/_deps/cnpypp-src \
        -i build-asan \
        -i build-ubsan \
        -i build-tsan \
        -i build-msan \
        -i build-coverage \
        -i cmake-build-debug \
        -i cmake-build-release \
        -i reports \
        -i third_party \
        -i external \
        -i _deps \
        . \
        2>&1 || true
}

count_cppcheck_findings() {
    local xml="$REPORT_DIR/cppcheck.xml"

    if [[ ! -f "$xml" ]]; then
        echo "0"
        return 0
    fi

    grep -c '<error ' "$xml" || true
}

run_clang_tidy() {
    if [[ "$RUN_CLANG_TIDY" != "1" ]]; then
        echo "Skipped. Enable with RUN_CLANG_TIDY=1."
        return 0
    fi

    if ! have_cmd clang-tidy; then
        echo "clang-tidy not installed."
        echo "Install: sudo dnf install clang-tools-extra"
        return 0
    fi

    local db_dir
    db_dir="$(filtered_compile_db_dir)"

    if [[ -z "$db_dir" ]]; then
        echo "No compile_commands.json found."
        echo
        echo "For CMake, configure with:"
        echo "cmake -S . -B $BUILD_DIR -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
        return 0
    fi

    local out="$REPORT_DIR/clang-tidy.txt"
    : > "$out"

    local index=0

    echo "Using compile database directory: $db_dir"
    echo "MAX_TIDY_FILES=$MAX_TIDY_FILES"
    echo "Per-file timeout: ${TOOL_TIMEOUT}s"
    echo

    cpp_sources | while IFS= read -r file; do
        index=$((index + 1))

        if (( MAX_TIDY_FILES > 0 && index > MAX_TIDY_FILES )); then
            echo "Stopping after MAX_TIDY_FILES=$MAX_TIDY_FILES." | tee -a "$out"
            break
        fi

        echo "===== clang-tidy: $file =====" | tee -a "$out"

	run_with_timeout clang-tidy "$file" -p "$db_dir" \
	    --checks='bugprone-*,performance-*,portability-*,-portability-avoid-pragma-once,modernize-*,-modernize-use-trailing-return-type,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers,-cppcoreguidelines-pro-bounds-avoid-unchecked-container-access,-cppcoreguidelines-pro-type-reinterpret-cast,-cppcoreguidelines-owning-memory,readability-*,-readability-magic-numbers,-readability-identifier-length' \
            --quiet \
            2>&1 | tee -a "$out" || true

        echo | tee -a "$out"
    done

    cat "$out"
}

count_clang_tidy_warnings() {
    local out="$REPORT_DIR/clang-tidy.txt"

    if [[ ! -f "$out" ]]; then
        echo "0"
        return 0
    fi

    grep -Ec 'warning:|error:' "$out" || true
}

run_gcovr_coverage() {
    if [[ "$RUN_GCOVR" != "1" ]]; then
        echo "Skipped. Enable with RUN_GCOVR=1 on a coverage-instrumented build."
        echo
        echo "Your normal Debug build is not enough for useful gcovr coverage."
        return 0
    fi

    if ! have_cmd gcovr; then
        echo "gcovr not installed."
        echo "Install: pip install gcovr"
        return 0
    fi

    if [[ ! -d "$BUILD_DIR" ]]; then
        echo "Build directory '$BUILD_DIR' not found."
        echo "Coverage requires a coverage-instrumented build."
        return 0
    fi

    local txt="$REPORT_DIR/gcovr.txt"
    local xml="$REPORT_DIR/gcovr.xml"
    local html="$REPORT_DIR/gcovr.html"

    echo "Running gcovr with timeout: ${TOOL_TIMEOUT}s"
    echo

    run_with_timeout gcovr \
        --root . \
        --object-directory "$BUILD_DIR" \
        --exclude-directories "$BUILD_DIR" \
        --exclude '.*third_party.*' \
        --exclude '.*external.*' \
        --exclude '.*_deps.*' \
        --exclude '.*reports.*' \
        --txt \
        --xml-pretty \
        --xml "$xml" \
        --html-details "$html" \
        2>&1 | tee "$txt" || true
}

extract_gcovr_line_coverage() {
    local txt="$REPORT_DIR/gcovr.txt"

    if [[ ! -f "$txt" ]]; then
        echo "unknown"
        return 0
    fi

    awk '
        /TOTAL/ {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^[0-9]+(\.[0-9]+)?%$/) {
                    gsub("%", "", $i)
                    print $i
                    found=1
                    exit
                }
            }
        }
        END {
            if (!found) print "unknown"
        }
    ' "$txt"
}

coverage_recommendation() {
    local cov="$1"

    if [[ "$cov" == "unknown" || -z "$cov" || ! "$cov" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Coverage not measured. Use a separate coverage-instrumented build and run with RUN_GCOVR=1."
        return 0
    fi

    local cov_int="${cov%.*}"

    if (( cov_int < MIN_LINE_COVERAGE )); then
        echo "Coverage is below the target. Add tests around core pipeline components, plugin contracts, ownership boundaries, and error paths."
    else
        echo "Coverage is at or above the configured minimum."
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

codechecker_readiness() {
    local db
    db="$(compile_commands_path)"

    if [[ -n "$db" ]]; then
        echo "compile_commands.json found at: $db"
        echo
        echo "CodeChecker can use this project."
        echo
        echo "Example commands:"
        echo

	local filtered_db_dir
        filtered_db_dir="$(filtered_compile_db_dir)"

        echo "Filtered compile_commands.json generated at:"
        echo "$filtered_db_dir/compile_commands.json"
        echo
        echo "Recommended CodeChecker commands:"
        echo
        echo "CodeChecker analyze $filtered_db_dir/compile_commands.json -o $REPORT_DIR/codechecker-reports"
        echo "CodeChecker parse $REPORT_DIR/codechecker-reports -e html -o $REPORT_DIR/codechecker-html"
    else
        echo "compile_commands.json not found."
        echo
        echo "Generate it with CMake:"
        echo
        echo "cmake -S . -B $BUILD_DIR -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
    fi
}

ci_recommendation() {
    cat <<EOF
Recommended CI stages:

1. Configure:
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

2. Build:
   cmake --build build -j

3. Test:
   ctest --test-dir build --output-on-failure

4. Fast quality report:
   RUN_CPPCHECK=0 RUN_CLANG_TIDY=0 RUN_GCOVR=0 tools/project_quality_report.sh . reports build

5. Static analysis report:
   RUN_CPPCHECK=1 RUN_CLANG_TIDY=1 RUN_GCOVR=0 MAX_TIDY_FILES=10 TOOL_TIMEOUT=180 tools/project_quality_report.sh . reports build

6. Coverage:
   Use a separate coverage-instrumented build and run:
   RUN_GCOVR=1 tools/project_quality_report.sh . reports build-coverage

7. Quality gates:
   - clang-tidy warnings <= $MAX_CLANG_TIDY_WARNINGS
   - cppcheck findings <= $MAX_CPPCHECK_ERRORS
   - line coverage >= $MIN_LINE_COVERAGE%
   - cyclomatic complexity target <= $MAX_CCN
   - function length target <= $MAX_FUNCTION_LENGTH lines
EOF
}

{
    echo "# Modern C++23 Code Quality Report"
    echo
    echo "- Generated: $now"
    echo "- Project: $(pwd)"
    echo "- Build directory: $BUILD_DIR"
    echo "- Report artifacts: $REPORT_DIR"
    echo

    section "Executive Summary"

    source_count="$(cpp_sources | wc -l)"
    header_count="$(cpp_headers | wc -l)"
    test_count="$(test_files | wc -l)"
    compile_db="$(compile_commands_path)"

    echo "| Metric | Value |"
    echo "|---|---:|"
    echo "| Source files | $source_count |"
    echo "| Header files | $header_count |"
    echo "| Test files, approximate | $test_count |"

    if [[ -n "$compile_db" ]]; then
        echo "| compile_commands.json | found: $compile_db |"
    else
        echo "| compile_commands.json | not found |"
    fi

    section "Execution Settings"

    echo "| Setting | Value |"
    echo "|---|---:|"
    echo "| RUN_CPPCHECK | $RUN_CPPCHECK |"
    echo "| RUN_CLANG_TIDY | $RUN_CLANG_TIDY |"
    echo "| RUN_GCOVR | $RUN_GCOVR |"
    echo "| MAX_TIDY_FILES | $MAX_TIDY_FILES |"
    echo "| TOOL_TIMEOUT seconds | $TOOL_TIMEOUT |"
    echo "| Dependency directories excluded | build/_deps, _deps, third_party, external |"

    section "Quality Thresholds"

    echo "| Gate | Threshold |"
    echo "|---|---:|"
    echo "| Max cyclomatic complexity target | $MAX_CCN |"
    echo "| Max function length target | $MAX_FUNCTION_LENGTH lines |"
    echo "| Max file length target | $MAX_FILE_LENGTH lines |"
    echo "| Min line coverage target | $MIN_LINE_COVERAGE% |"
    echo "| Min branch coverage target | $MIN_BRANCH_COVERAGE% |"
    echo "| Max clang-tidy warnings | $MAX_CLANG_TIDY_WARNINGS |"
    echo "| Max cppcheck findings | $MAX_CPPCHECK_ERRORS |"

    run_block "Level 1A - LOC Summary - cloc" run_cloc

    run_block "Level 1B - Complexity Summary - lizard" run_lizard_full

    run_block "Level 1C - Refactoring Candidates - lizard" run_lizard_refactor_candidates

    run_block "Large Files Above Threshold" largest_code_files

    run_block "Largest Code Files" all_code_files_by_size

    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        run_block "Level 1D - Git Churn and Hotspots" git_churn_summary
    else
        section "Level 1D - Git Churn and Hotspots"
        echo "Not inside a Git repository."
    fi

    run_block "Level 2A - cppcheck Static Analysis" run_cppcheck

    run_block "Level 2B - clang-tidy Static Analysis" run_clang_tidy

    run_block "Level 3 - Coverage - gcovr" run_gcovr_coverage

    section "Level 4 - CodeChecker Readiness"
    codeblock_begin
    codechecker_readiness
    codeblock_end

    section "Level 5 - CI Quality Gates"

    cppcheck_count="$(count_cppcheck_findings)"
    clang_tidy_count="$(count_clang_tidy_warnings)"
    line_coverage="$(extract_gcovr_line_coverage)"

    echo "| Gate | Current | Target | Status |"
    echo "|---|---:|---:|---|"
    quality_gate_status "cppcheck findings" "$cppcheck_count" "$MAX_CPPCHECK_ERRORS" "max"
    quality_gate_status "clang-tidy warnings/errors" "$clang_tidy_count" "$MAX_CLANG_TIDY_WARNINGS" "max"
    quality_gate_status "line coverage" "$line_coverage" "$MIN_LINE_COVERAGE" "min"

    section "Coverage Recommendation"

    coverage_recommendation "$line_coverage"

    section "Refactoring Interpretation"

    cat <<EOF
Use this report like this:

| Signal | Interpretation |
|---|---|
| High CCN | Function has many decision paths; likely needs decomposition or stronger tests. |
| Long function | Function may mix responsibilities; consider extraction. |
| Large file | Possible module-boundary or abstraction problem. |
| High churn + high complexity | Strong refactoring candidate. |
| cppcheck finding | Possible bug, undefined behavior, portability issue, or maintainability issue. |
| clang-tidy finding | Modern C++ issue, style issue, bug-prone construct, or API misuse. |
| Low coverage | Tests do not exercise enough code; prioritize core abstractions and error paths. |
| Missing compile_commands.json | clang-tidy and CodeChecker cannot analyze the project properly. |
| Timed-out tool | Tool was too expensive for this run; increase TOOL_TIMEOUT or run a narrower/deeper job. |
EOF

    section "Recommended Modern C++23 Checks"

    cat <<EOF
For clang-tidy, consider using a .clang-tidy file with groups such as:

- bugprone-*
- cppcoreguidelines-*
- modernize-*
- performance-*
- portability-*
- readability-*

For a framework like LeakFlow, pay special attention to:

- ownership and lifetime boundaries
- plugin API stability
- virtual destructor correctness
- unnecessary copies in trace/data buffers
- exception safety
- const-correctness
- thread-safety assumptions
- enum/string property conversion safety
- test coverage of element/property negotiation
EOF

    section "CI Recommendation"

    codeblock_begin
    ci_recommendation
    codeblock_end

    section "Trust Notes"

    cat <<EOF
This report is stronger than a simple size report, but it still has layers of trust:

| Metric | Trust | Notes |
|---|---|---|
| cloc | High | Good for approximate code size. |
| lizard CCN | Medium-high | Good refactoring signal, not proof of bad design. |
| cppcheck | Medium-high | Useful bug/static-analysis signal; review false positives. |
| clang-tidy | High when compile_commands.json is correct | Best with a real CMake compile database, but can be slow with libtorch-heavy translation units. |
| gcovr coverage | High if build is properly instrumented | Coverage is only meaningful after running tests in a coverage build. |
| test file count | Medium | Filename heuristic only. |
| recommendations | Medium | Human review still required. |

This report should be treated as a serious engineering dashboard, not as an automatic architectural judge.
EOF

} > "$REPORT_FILE"

echo "Quality report written to: $REPORT_FILE"
echo "Artifacts written to: $REPORT_DIR"
