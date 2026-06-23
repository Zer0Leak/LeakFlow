#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="${1:-.}"
REPORT_ROOT="${2:-reports}"

REPORT_FILE="$REPORT_ROOT/project_size_report.md"

cd "$PROJECT_DIR"

mkdir -p "$REPORT_ROOT"

now="$(date '+%Y-%m-%d %H:%M:%S')"

have_cmd() {
    command -v "$1" >/dev/null 2>&1
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

count_files() {
    find . \
        -path './.git' -prune -o \
        -path './build' -prune -o \
        -path './build-*' -prune -o \
        -path './build-cuda' -prune -o \
        -path './cmake-build-*' -prune -o \
        -path './.cache' -prune -o \
        -path './.vscode' -prune -o \
        -path './reports' -prune -o \
        -path './papers' -prune -o \
        -path './traces' -prune -o \
        -path './_deps' -prune -o \
        -path './third_party' -prune -o \
        -path './external' -prune -o \
        -type f -print
}

cpp_files() {
    count_files | grep -E '\.(c|cc|cpp|cxx|h|hh|hpp|hxx)$' || true
}

source_files() {
    count_files | grep -E '\.(c|cc|cpp|cxx)$' || true
}

header_files() {
    count_files | grep -E '\.(h|hh|hpp|hxx)$' || true
}

md_files() {
    count_files | grep -E '\.md$' || true
}

test_files() {
    count_files | grep -Ei '(^|/)(test|tests|spec|specs)(/|_)|(_test|_tests|Test|Tests)\.(cpp|cc|cxx|h|hpp)$' || true
}

files_by_extension() {
    count_files | awk '
        function ext(path) {
            n = split(path, a, "/")
            name = a[n]

            if (name == "CMakeLists.txt") return "CMakeLists.txt"
            if (name !~ /\./) return "[no extension]"

            sub(/^.*\./, "", name)
            return name
        }

        {
            counts[ext($0)]++
        }

        END {
            for (e in counts)
                printf "%8d  %s\n", counts[e], e
        }
    ' | sort -nr | head -50
}

largest_files_by_lines() {
    count_files \
        | grep -E '\.(c|cc|cpp|cxx|h|hh|hpp|hxx|md|cmake|txt)$|CMakeLists\.txt$' \
        | while read -r f; do
            lines="$(wc -l < "$f" 2>/dev/null || echo 0)"
            printf "%8d  %s\n" "$lines" "$f"
        done \
        | sort -nr \
        | head -30
}

markdown_files_by_lines() {
    md_files \
        | while read -r f; do
            lines="$(wc -l < "$f" 2>/dev/null || echo 0)"
            printf "%8d  %s\n" "$lines" "$f"
        done \
        | sort -nr
}

detected_classes_structs() {
    local count=0
    local max=200

    while IFS= read -r file; do
        while IFS= read -r line; do
            printf '%s:%s\n' "$file" "$line"
            count=$((count + 1))

            if (( count >= max )); then
                return 0
            fi
        done < <(
            grep -nEh '^[[:space:]]*(class|struct)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$file" 2>/dev/null || true
        )
    done < <(cpp_files)
}

git_summary() {
    echo "Commits:"
    git rev-list --count HEAD

    echo
    echo "Contributors:"
    git shortlog -sn
}

git_churn_30_days() {
    git log --since='30 days ago' --shortstat --oneline
}

{
    echo "# Project Size Report"
    echo
    echo "- Generated: $now"
    echo "- Project: $(pwd)"
    echo

    section "Summary"

    total_files="$(count_files | wc -l)"
    cpp_count="$(cpp_files | wc -l)"
    source_count="$(source_files | wc -l)"
    header_count="$(header_files | wc -l)"
    md_count="$(md_files | wc -l)"
    test_count="$(test_files | wc -l)"

    full_disk_size="$(du -sh . | awk '{print $1}')"

    source_disk_size="$(
        count_files \
            | xargs -r du -ch 2>/dev/null \
            | awk '/total$/ {print $1}' \
            || true
    )"
    source_disk_size="${source_disk_size:-unknown}"
    
    excluded_disk_size="$(
        du -sch \
            build build-* build-cuda cmake-build-* \
            reports papers traces \
            _deps third_party external \
            2>/dev/null \
            | awk '/total$/ {print $1}' \
            || true
    )"
    excluded_disk_size="${excluded_disk_size:-0}"

    class_count="$(
        {
            while IFS= read -r file; do
                grep -Eh '^[[:space:]]*(class|struct)[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$file" 2>/dev/null || true
            done < <(cpp_files)
        } \
            | grep -vE '^[[:space:]]*//' \
            | wc -l \
            || true
    )"
    class_count="${class_count:-0}"

    md_words="$(
        md_files \
            | xargs -r cat \
            | wc -w
    )"

    echo "| Metric | Value |"
    echo "|---|---:|"
    echo "| Disk size, source files counted | $source_disk_size |"
    echo "| Disk size, excluded folders (traces, papers, build) | $excluded_disk_size |"
    echo "| Disk size, full directory | $full_disk_size |"
    echo "| Total files | $total_files |"
    echo "| C/C++ files | $cpp_count |"
    echo "| Source files | $source_count |"
    echo "| Header files | $header_count |"
    echo "| Markdown files | $md_count |"
    echo "| Markdown words | $md_words |"
    echo "| Test files | $test_count |"
    echo "| Classes / structs, approximate | $class_count |"

    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        commits="$(git rev-list --count HEAD 2>/dev/null || echo 0)"
        contributors="$(git shortlog -sn 2>/dev/null | wc -l || echo 0)"

        echo "| Git commits | $commits |"
        echo "| Git contributors | $contributors |"
    fi

    run_block "Files by Extension" files_by_extension

    section "C++ Files"

    echo "| Kind | Count |"
    echo "|---|---:|"
    echo "| Sources | $source_count |"
    echo "| Headers | $header_count |"
    echo "| Total C/C++ | $cpp_count |"

    run_block "Largest Source / Header / Markdown Files" largest_files_by_lines

    section "Documentation"

    echo "| Metric | Value |"
    echo "|---|---:|"
    echo "| Markdown files | $md_count |"
    echo "| Markdown words | $md_words |"

    run_block "Markdown Files by Line Count" markdown_files_by_lines

    section "Tests"

    echo "| Metric | Value |"
    echo "|---|---:|"
    echo "| Test files, approximate | $test_count |"

    run_block "Detected Test Files" test_files

    section "Classes and Structs"

    run_block "Detected Classes / Structs, Approximate" detected_classes_structs

    if have_cmd cloc; then
	run_block "Lines of Code - cloc" cloc . \
		--exclude-dir=.git,build,build-debug,build-release,build-cuda,build-asan,build-ubsan,build-tsan,build-msan,build-coverage,cmake-build-debug,cmake-build-release,.cache,.vscode,reports,papers,traces,_deps,third_party,external
    else
        section "Lines of Code - cloc"
        echo "Not available."
        echo
        echo "Install on Fedora:"
        echo
        echo '```bash'
        echo "sudo dnf install cloc"
        echo '```'
        echo
    fi

    if have_cmd lizard; then
        run_block "Complexity Summary - lizard" lizard . \
            -x "./build/*" \
            -x "./build-*/*" \
            -x "./cmake-build-*/*" \
            -x "./.git/*" \
            -x "./.cache/*" \
            -x "./reports/*" \
	    -x "./papers/*" \
            -x "./traces/*" \
            -x "./_deps/*" \
            -x "./third_party/*" \
            -x "./external/*"

        run_block "High Complexity Functions - CCN > 10" lizard -C 10 . \
            -x "./build/*" \
            -x "./build-*/*" \
            -x "./cmake-build-*/*" \
            -x "./.git/*" \
            -x "./.cache/*" \
            -x "./reports/*" \
	    -x "./papers/*" \
            -x "./traces/*" \
            -x "./_deps/*" \
            -x "./third_party/*" \
            -x "./external/*"
    else
        section "Complexity - lizard"
        echo "Not available."
        echo
        echo "Install:"
        echo
        echo '```bash'
        echo "pip install lizard"
        echo '```'
        echo
    fi

    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        run_block "Git Summary" git_summary
        run_block "Git Churn - Last 30 Days" git_churn_30_days
    else
        section "Git Summary"
        echo "This directory is not inside a Git repository."
        echo
    fi

    section "Recommended Interpretation"

    echo "| Signal | Meaning |"
    echo "|---|---|"
    echo "| High LOC + low tests | Project is growing faster than validation. |"
    echo "| Many headers + few tests | Public API may be under-tested. |"
    echo "| Many functions with CCN > 10 | Refactoring candidates. |"
    echo "| Very large files | Possible module boundary problem. |"
    echo "| High Markdown words | Good design/documentation investment. |"
    echo "| High Git churn | Architecture may still be unstable. |"

} > "$REPORT_FILE"

echo "Report written to: $REPORT_FILE"
