#!/usr/bin/env bash
#
# Gate stamps, and the push guard that reads them.
#
#     code_health/gate_stamp.sh record <gate>...        a gate measured the tree on the current base
#     code_health/gate_stamp.sh list                    the stamps and waivers of this worktree
#     code_health/gate_stamp.sh drop <gate>...          forget a stamp that does not bear on the push
#     code_health/gate_stamp.sh waive <gate> <reason>   push without a required gate, for a reason
#     code_health/gate_stamp.sh readers <file>          the gates that read a shared ledger file
#     code_health/gate_stamp.sh reach                   the gates the change of the worktree reaches
#     code_health/gate_stamp.sh censuses <file>         the prose censuses that read a file
#     code_health/gate_stamp.sh base [<ref>]            print, or store, the landing base
#     code_health/gate_stamp.sh check                   the pre-push guard of .pre-commit-config.yaml
#
# The landing base is the branch the worktree lands on (ADR 0240): origin/main for a branch of its
# own, or the shared branch, such as origin/dev. `base <ref>` stores it in the git directory
# of the worktree, and it defaults to origin/main.
#
# A gate measures a tree: the merge-base of the branch with the landing base, plus the branch. When
# a sibling commit lands on the base and the branch rebases onto it, the tree changes, and a gate
# measured before the rebase measured a program nobody will ship. `check` refuses the push when
# the base changed, after a gate measured the tree, a file the branch also changes, or a file in
# the same directory under src/ or ext/. Re-run the gates it names on the rebased tree, or `drop`
# the stamp of a gate the change cannot reach. Run `check` by hand after each rebase: it answers
# before the push does.
#
# These refinements keep the guard to the gates a move can reach:
#
#   - A commit this worktree pushed is its own, not a sibling's. A passed `check` of a push records
#     the pushed commits, and a later `check` leaves their files out.
#   - A shared ledger file, which nearly every ticket edits, stales only the gates that read it.
#     `readers` names them, and `test/test_76_gate_tooling.jl` holds the list against the tests
#     that name the file.
#   - A stamp goes stale only on a sibling's change that can alter what the gate measured. A page
#     of prose, a `.md` file, stales the prose gate, the test files, and the full doctest run when
#     it is a manual page under docs/src/. It stales no gate that measures the code. A sibling's
#     change to the Julia code under src/ or ext/ stales `jet`, `coverage` and a doctest stamp only
#     when its class, as `code_health/gate_reach.jl` names it, reaches that gate, as `reached`
#     reads the class for a push.
#   - A push that changes a `.jl` file under src/ or ext/ needs a stamp of each gate in
#     GATE_STAMP_REQUIRED, the gates of the `sweep-conform` skill, that the change reaches.
#     `code_health/gate_reach.jl` names the class of the change. A change to docstring prose
#     reaches neither `jet` nor `doctest`. A changed `jldoctest` block, a changed string literal
#     and a change to the code need the doctest stamp, and a changed literal does not need `jet`.
#     `waive` records the reason a gate does not apply.
#   - The doctest stamp is the scoped one, `doctest-files`, as JET and the other gates take a
#     scoped run. `docs/doctest_files.jl --against origin/main` runs the blocks of the changed
#     files, and the full run of `--all` covers it too. A change can alter what a block of another
#     file prints. The doctest job of CI runs every block and stays the gate for that case.
#   - A push that changes a text that a prose census reads needs a stamp of that census, whatever
#     the class of the change. `censuses` names the census files that read a path: the docstrings,
#     strings and names under src/ and ext/, the pages of the four page corpora, the standards
#     files, GLOSSARY.md and the prose baseline. Each census runs in seconds, and each one reached
#     `dev` red when a session changed prose and did not run it (`test_74` on `read-out`).
#     `test/run_files.jl` records the stamp of a test file only when the file is green, so a red
#     census does not let the push through. `waive test/<file> <reason>` records a known red.
#   - `coverage` is required of a `code` change alone. `test/run_files.jl --coverage=<dir>` records
#     it when the targeted files ran green and no line that the branch adds is a line that CI
#     reads as a miss (#1912, ADR 0082). A line that no test can reach carries a Coverage
#     Exemption, and `waive coverage <reason>` names it.
#
# `code_health/CodeHealth.jl` records every check and refresh of the gate scripts, and
# `test/run_files.jl` records every test file it runs. `docs/doctest_files.jl` records
# `doctest-files` for a scoped run and `doctest` for the full run of `--all`.
#
# Stamps live in the git directory of the worktree, so each worktree has its own and none is
# committed. The guard acts only in a worktree under `.claude/worktrees/`, where parallel sessions
# work. The checkout of the maintainer records and checks nothing.
#
# Environment:
#   GATE_STAMP_REF       the ref a gate is measured against. Default: the landing base, which
#                        is origin/main when `base` stored none.
#   GATE_STAMP_BRANCH    the globs, split on spaces, of the remote branches the guard protects,
#                        the branches of ADR 0240. Default: refs/heads/agents/* refs/heads/dev.
#   GATE_STAMP_REQUIRED  the gates a push that changes src/ or ext/ needs. Default:
#                        "jet size complexity perf doctest coverage". The tests set it empty.
#   GATE_STAMP_CENSUS    empty turns off the demand for the prose censuses. Default: on. The
#                        tests set it empty, except the test of that demand.
#   GATE_STAMP_FORCE     any value acts outside .claude/worktrees/ too. The tests use it.
#   PRE_COMMIT_TO_REF, PRE_COMMIT_REMOTE_BRANCH
#                        set by pre-commit for a pre-push hook: the pushed tip and the branch
#                        pushed to.

set -euo pipefail

usage() {
    echo "usage: gate_stamp.sh record <gate>... | list | drop <gate>... | waive <gate> <reason>" \
        "| readers <file> | reach | censuses <file> | base [<ref>] | check" >&2
    exit 2
}

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
top=$(git rev-parse --show-toplevel)
gitdir=$(git rev-parse --absolute-git-dir)
stamps="$gitdir/gate_stamps"
waivers="$gitdir/gate_waivers"
pushed="$gitdir/gate_pushed"
basefile="$gitdir/gate_base"
if [ -n "${GATE_STAMP_REF:-}" ]; then
    ref=$GATE_STAMP_REF
elif [ -s "$basefile" ]; then
    ref=$(cat "$basefile")
else
    ref=origin/main
fi
required=${GATE_STAMP_REQUIRED-jet size complexity perf doctest coverage}
census=${GATE_STAMP_CENSUS-on}

acts() {
    case "$top" in
    */.claude/worktrees/*) return 0 ;;
    esac
    [ -n "${GATE_STAMP_FORCE:-}" ]
}

# Print the lines of the file "$1", one `<gate> TAB ...` line per gate, without the gates named in
# the other arguments.
without() {
    local file=$1
    shift
    if [ -f "$file" ]; then
        awk -F '\t' 'BEGIN { for (i = 1; i < ARGC; i++) drop[ARGV[i]] = 1; ARGC = 1 }
                     !($1 in drop)' "$@" <"$file"
    fi
}

record() {
    [ $# -gt 0 ] || usage
    acts || return 0
    local base
    base=$(git merge-base "$ref" HEAD 2>/dev/null) || return 0
    local next
    next=$(without "$stamps" "$@")
    for gate in "$@"; do
        next+=$'\n'"$gate"$'\t'"$base"
    done
    printf '%s\n' "$next" | sed '/^$/d' >"$stamps"
}

drop() {
    [ $# -gt 0 ] || usage
    acts || return 0
    local next
    next=$(without "$stamps" "$@")
    printf '%s\n' "$next" | sed '/^$/d' >"$stamps"
}

waive() {
    [ $# -ge 2 ] || usage
    acts || return 0
    local gate=$1
    shift
    local next
    next=$(without "$waivers" "$gate")
    next+=$'\n'"$gate"$'\t'"$*"
    printf '%s\n' "$next" | sed '/^$/d' >"$waivers"
}

# Print the landing base, or store a new one. The landing base is the branch the worktree lands on
# (ADR 0240): `origin/main` for a branch of its own, `origin/dev` or `origin/agents/<map>-…`
# for a shared branch. Every stamp and every check reads it.
landing_base() {
    if [ $# -eq 0 ]; then
        echo "$ref"
        return 0
    fi
    [ $# -eq 1 ] || usage
    if ! git rev-parse --verify --quiet "$1^{commit}" >/dev/null; then
        echo "gate_stamp.sh: $1 names no commit. Fetch it first: git fetch origin <branch>." >&2
        return 2
    fi
    printf '%s\n' "$1" >"$basefile"
}

list() {
    if [ -f "$stamps" ]; then
        cat "$stamps"
    fi
    if [ -s "$waivers" ]; then
        sed 's/\t/\twaived: /' "$waivers"
    fi
}

# The gates that read a shared ledger file, or `*` for a file every gate can read. A ledger file is
# one that nearly every ticket edits, so a sibling's edit to it would stale every stamp.
readers() {
    case "$1" in
    code_health/sweep_manifest.toml)
        echo "test/test_26_docs.jl test/test_45_sweep_census.jl" \
            "test/test_47_alias_and_module_census.jl"
        ;;
    GLOSSARY.md)
        echo "prose test/test_26_docs.jl test/test_46_standards_citation_census.jl" \
            "test/test_72_prose_census.jl test/test_74_source_citation_census.jl"
        ;;
    code_health/size_baseline.toml)
        echo "size test/test_52_size_classification_census.jl"
        ;;
    code_health/perf_baseline.toml)
        echo "perf test/test_73_performance_trap_census.jl"
        ;;
    code_health/prose_baseline.toml)
        echo "prose test/test_72_prose_census.jl test/test_74_source_citation_census.jl"
        ;;
    code_health/*_baseline.toml)
        local name
        name=$(basename "$1")
        echo "${name%_baseline.toml}"
        ;;
    code_health/expansion_bound.toml)
        echo "expansion"
        ;;
    *)
        echo "*"
        ;;
    esac
}

# Whether the gate "$1" reads the file "$2". A page of prose, a `.md` file that is no ledger, is
# read by the prose gate and by a test file, which can read any page, and by the full doctest run
# when it is a manual page under docs/src/. No gate that measures the code reads one.
reads() {
    local r
    r=$(readers "$2")
    if [ "$r" = "*" ]; then
        case "$2" in
        *.md) reads_page "$1" "$2" ;;
        *) return 0 ;;
        esac
        return
    fi
    case " $r " in
    *" $1 "*) return 0 ;;
    esac
    return 1
}

# Whether the gate "$1" reads the page "$2", a `.md` file that is no ledger.
reads_page() {
    case "$1" in
    prose | test/*) return 0 ;;
    doctest)
        case "$2" in
        docs/src/*) return 0 ;;
        esac
        ;;
    esac
    return 1
}

# The prose censuses that read the path "$1", as the stamps that `test/run_files.jl` records, one
# to a line. Each case reads one corpus, and a path can be in more than one. The generated pages
# under docs/src/ are untracked, so no diff names them. The `CODING_STANDARDS.md` rows of the
# census files state each corpus.
censuses() {
    local t46=test/test_46_standards_citation_census.jl
    local t64=test/test_64_docs_page_metadata_census.jl
    local t71=test/test_71_process_citation_census.jl
    local t72=test/test_72_prose_census.jl
    local t74=test/test_74_source_citation_census.jl
    # The docstrings, the strings and the names of the library: `test_71` and `test_74` read
    # them, and `test_64` reads every published text for a link to another repository.
    case "$1" in
    src/*.jl | ext/*.jl) printf '%s\n' "$t64" "$t71" "$t74" ;;
    esac
    # The four page corpora: the Literate sources, the hand-written pages of docs/src/ outside
    # contribute/, the README and the Capability Catalogue.
    case "$1" in
    docs/src/contribute/*) ;;
    README.md | docs/capability_catalogue.jl | examples/*.jl | user_guide/*.jl | docs/src/*.md)
        printf '%s\n' "$t71" "$t72" "$t74"
        ;;
    esac
    # Every other published text of the docs tree, with its ADRs, papers and generators.
    case "$1" in
    README.md | examples/*.jl | user_guide/*.jl | docs/*.md | docs/*.jl | docs/*.typ)
        printf '%s\n' "$t64"
        ;;
    esac
    # The standards files, and the two ledgers of prose.
    case "$1" in
    CODING_STANDARDS.md | CLAUDE.md | */CLAUDE.md | .github/copilot-instructions.md | \
        .github/instructions/*.md | .github/prompts/*.md)
        printf '%s\n' "$t46"
        ;;
    GLOSSARY.md) printf '%s\n' "$t46" "$t72" "$t74" ;;
    code_health/prose_baseline.toml) printf '%s\n' "$t72" "$t74" ;;
    esac
}

# The prose censuses that a change to the files of the list "$1" needs, one path to a line. It
# prints each census once, and nothing when GATE_STAMP_CENSUS is empty.
needed_censuses() {
    [ -n "$census" ] || return 0
    local f
    printf '%s\n' "$1" | while read -r f; do
        [ -n "$f" ] || continue
        censuses "$f"
    done | sort -u
}

# Whether a change of the class "$2" reaches the gate "$1", as `reached` reads it. A stamp of the
# full doctest run and a stamp of the scoped run are both doctest stamps here.
class_reaches() {
    local gate=$1
    if [ "$gate" = doctest-files ]; then
        gate=doctest
    fi
    [ -n "$(reached "$gate" "$2")" ]
}

# Whether the file "$1" holds a line for the gate "$2".
names_gate() {
    [ -f "$1" ] && awk -F '\t' -v g="$2" '$1 == g { found = 1 } END { exit !found }' "$1"
}

# Whether a stamp or a waiver covers the gate "$1". The full doctest run covers the scoped one.
covered() {
    names_gate "$stamps" "$1" || names_gate "$waivers" "$1" ||
        { [ "$1" = doctest-files ] && covered doctest; }
}

# The class that code_health/gate_reach.jl names for the change from "$1" to "$2", a commit or
# WORKTREE, or `code` when the script cannot run.
reach_class() {
    local out
    if out=$(julia --startup-file=no "$here/gate_reach.jl" "$1" "$2" 2>/dev/null); then
        printf '%s\n' "$out" | awk '$1 == "reach" { print $2 }'
    else
        echo code
    fi
}

# The gates of the list "$1" that a change of the class "$2" reaches, one to a line.
reached() {
    local gate
    for gate in $1; do
        case "$2:$gate" in
        prose:jet | prose:doctest | block:jet | literal:jet) ;;
        prose:coverage | block:coverage | literal:coverage) ;;
        *:doctest) echo doctest-files ;;
        *) echo "$gate" ;;
        esac
    done
}

# Print the class of each changed file of the worktree against the ref, and the gates the change
# reaches.
reach() {
    local base out class need
    base=$(git merge-base "$ref" HEAD)
    out=$(julia --startup-file=no "$here/gate_reach.jl" "$base" WORKTREE)
    printf '%s\n' "$out"
    if [ "$(printf '%s\n' "$out" | wc -l)" -eq 1 ]; then
        echo "gates none: no .jl file under src/ or ext/ changes"
    else
        class=$(printf '%s\n' "$out" | awk '$1 == "reach" { print $2 }')
        echo "gates" $(reached "$required" "$class")
    fi
    need=$(needed_censuses "$({
        git diff --name-only --no-renames "$base"
        git ls-files --others --exclude-standard
    } | sort -u)")
    if [ -n "$need" ]; then
        echo "censuses" $need
    fi
}

# The files that the commits base..from change, without the commits this worktree pushed.
moved_files() {
    local base=$1 from=$2
    git log --no-renames --name-only --format='@%H' "$base..$from" 2>/dev/null |
        awk -v pushed="$pushed" '
            BEGIN { while ((getline line < pushed) > 0) own[line] = 1 }
            /^@/ { skip = (substr($0, 2) in own); next }
            NF && !skip { print }' |
        sort -u
}

# Refuse the push when a stamp's base and the tip of the ref differ in a file the push changes, or in
# a src/ or ext/ directory the push changes, or when a push that changes src/ or ext/ lacks a stamp
# of a required gate. A plain run, outside pre-commit, checks HEAD against the ref.
#
# The tip that counts is the tip of the landing base. On a branch of its own, the base is the branch
# the pull request merges into, and not the remote tip of the pushed branch, which its first push
# does not have. On a shared branch, the base is the shared branch itself.
check() {
    acts || return 0
    local protected=${GATE_STAMP_BRANCH:-refs/heads/agents/* refs/heads/dev}
    if [ -n "${PRE_COMMIT_REMOTE_BRANCH:-}" ]; then
        local glob globs hit_branch=0
        # `read -a` splits the list on spaces and expands no pathname. Each glob is unquoted in
        # the pattern, so `case` reads it as a glob.
        read -r -a globs <<<"$protected"
        # shellcheck disable=SC2254
        for glob in "${globs[@]}"; do
            case "$PRE_COMMIT_REMOTE_BRANCH" in
            $glob) hit_branch=1 ;;
            esac
        done
        [ "$hit_branch" = 1 ] || return 0
        # A push to the shared branch measures against it. A worktree with another landing base
        # measured a tree that this push does not land on.
        if [ "$PRE_COMMIT_REMOTE_BRANCH" = refs/heads/dev ] && [ "$ref" != origin/dev ]; then
            echo "This push lands on dev, and the landing base is $ref. Run"
            echo "  code_health/gate_stamp.sh base origin/dev"
            echo "then rebase onto origin/dev, and run the gates again."
            return 1
        fi
    fi
    local from to
    from=$(git rev-parse "$ref")
    to=${PRE_COMMIT_TO_REF:-$(git rev-parse HEAD)}
    git cat-file -e "$from^{commit}" 2>/dev/null || return 0

    # The branch's own files run from the merge-base, so a branch not yet rebased does not count
    # the sibling's changes as its own.
    local own dirs
    own=$(git diff --name-only --no-renames "$(git merge-base "$from" "$to")" "$to")
    # `grep` exits 1 when the branch changes nothing under src/ or ext/, and `pipefail` would make
    # that exit the script's.
    dirs=$(printf '%s\n' "$own" | { grep -E '^(src|ext)/.*\.jl$' || true; } |
        xargs -r -n 1 dirname | sort -u)

    local stale="" names="" class_base="" class_moved=""
    if [ -s "$stamps" ]; then
        while IFS=$'\t' read -r gate base; do
            [ -n "$gate" ] || continue
            [ "$base" != "$from" ] || continue
            local moved hit code_reaches=1
            moved=$(moved_files "$base" "$from") || moved=""
            # A gate that reads the class of a change goes stale on a sibling's change to the
            # Julia code under src/ or ext/ only when that class reaches it: a change to
            # docstring prose stales neither `jet`, `coverage` nor a doctest stamp. One call of
            # the classifier serves every gate measured on the same base.
            case "$gate" in
            jet | coverage | doctest | doctest-files)
                if [ "$class_base" != "$base" ]; then
                    class_moved=$(reach_class "$base" "$from")
                    class_base=$base
                fi
                class_reaches "$gate" "$class_moved" || code_reaches=0
                ;;
            esac
            hit=$(printf '%s\n' "$moved" | while read -r f; do
                [ -n "$f" ] || continue
                reads "$gate" "$f" || continue
                if [ "$code_reaches" = 0 ]; then
                    case "$f" in
                    src/*.jl | ext/*.jl) continue ;;
                    esac
                fi
                if grep -qxF -- "$f" <<<"$own" ||
                    grep -qxF -- "$(dirname "$f")" <<<"$dirs"; then
                    echo "$f"
                fi
            done)
            if [ -n "$hit" ]; then
                stale+="  $gate, measured on ${base:0:10}. Since then $ref changed:"$'\n'
                stale+=$(printf '%s\n' "$hit" | sed 's/^/      /')$'\n'
                names+=" $gate"
            fi
        done <"$stamps"
    fi

    local absent="" class=""
    if [ -n "$dirs" ] && [ -n "$required" ]; then
        class=$(reach_class "$(git merge-base "$from" "$to")" "$to")
        for gate in $(reached "$required" "$class"); do
            covered "$gate" || absent+=" $gate"
        done
    fi
    local unread="" runs=""
    for gate in $(needed_censuses "$own"); do
        if ! covered "$gate"; then
            unread+=" $gate"
            runs+=" $(basename "$gate")"
        fi
    done

    local status=0
    if [ -n "$stale" ]; then
        echo "The push target moved under these gates after they measured the tree:"
        printf '%s' "$stale"
        echo "Re-run each gate that your change can reach on the rebased tree. Forget the rest"
        echo "in one call:"
        echo "  code_health/gate_stamp.sh drop$names"
        status=1
    fi
    if [ -n "$absent" ]; then
        echo "This push changes src/ or ext/ ($class), and no stamp records these gates:$absent"
        echo "Run each one (the sweep-conform skill, § Land the branch), or record why it does not"
        echo "apply: code_health/gate_stamp.sh waive <gate> <reason>."
        status=1
    fi
    if [ -n "$unread" ]; then
        echo "This push changes a text that these prose censuses read, and no green run records"
        echo "them:$unread"
        echo "Run them, in seconds after the package load:"
        echo "  julia -t 1 --project=test test/run_files.jl --done=<path>$runs"
        echo "A census that is a known red takes: code_health/gate_stamp.sh waive test/<file> <issue>."
        status=1
    fi
    # Remember the commits of a push this guard let through, so a later check does not count
    # them as a sibling's. A plain run pushes nothing.
    if [ "$status" = 0 ] && [ -n "${PRE_COMMIT_TO_REF:-}" ]; then
        git rev-list "$from..$to" >>"$pushed" 2>/dev/null || true
    fi
    return "$status"
}

[ $# -gt 0 ] || usage
cmd=$1
shift
case "$cmd" in
record) record "$@" ;;
list) list ;;
drop) drop "$@" ;;
waive) waive "$@" ;;
readers)
    [ $# -eq 1 ] || usage
    readers "$1"
    ;;
reach) reach ;;
censuses)
    [ $# -eq 1 ] || usage
    censuses "$1"
    ;;
base) landing_base "$@" ;;
check) check ;;
*) usage ;;
esac
