#!/usr/bin/env bash
#
# Gate stamps, and the push guard that reads them.
#
#     code_health/gate_stamp.sh record <gate>...        a gate measured the tree on the current base
#     code_health/gate_stamp.sh list                    the stamps and waivers of this worktree
#     code_health/gate_stamp.sh drop <gate>...          forget a stamp that does not bear on the push
#     code_health/gate_stamp.sh waive <gate> <reason>   push without a required gate, for a reason
#     code_health/gate_stamp.sh readers <file>          the gates that read a shared ledger file
#     code_health/gate_stamp.sh check                   the pre-push guard of .pre-commit-config.yaml
#
# A gate measures a tree: the merge-base of the branch with origin/dev, plus the branch. When a
# sibling commit lands on origin/dev and the branch rebases onto it, the tree changes, and a gate
# measured before the rebase measured a program nobody will ship. `check` refuses the push when
# origin/dev changed, after a gate measured the tree, a file the branch also changes, or a file in
# the same directory under src/ or ext/. Re-run the gates it names on the rebased tree, or `drop`
# the stamp of a gate the change cannot reach. Run `check` by hand after each rebase: it answers
# before the push does.
#
# Three refinements keep the guard to the gates a move can reach:
#
#   - A commit this worktree pushed is its own, not a sibling's. A passed `check` of a push records
#     the pushed commits, and a later `check` leaves their files out.
#   - A shared ledger file, which nearly every ticket edits, stales only the gates that read it.
#     `readers` names them, and `test/test_76_gate_tooling.jl` holds the list against the tests
#     that name the file.
#   - A push that changes src/ or ext/ needs a stamp of each gate in GATE_STAMP_REQUIRED, the gates
#     of the `sweep-conform` skill. `waive` records the reason a gate does not apply.
#
# `code_health/CodeHealth.jl` records every check and refresh of the gate scripts, and
# `test/run_files.jl` records every test file it runs. `docs/doctest_files.jl` records
# `doctest-files` for a scoped run and `doctest` for the full run of `--all`.
#
# Stamps live in the git directory of the worktree, so each worktree has its own and none is
# committed. The guard acts only in a worktree under `.claude/worktrees/`, where parallel sessions
# work. The `dev` checkout of the maintainer records and checks nothing.
#
# Environment:
#   GATE_STAMP_REF       the ref a gate is measured against. Default: origin/dev.
#   GATE_STAMP_BRANCH    the remote branch the guard protects. Default: refs/heads/dev.
#   GATE_STAMP_REQUIRED  the gates a push that changes src/ or ext/ needs. Default:
#                        "jet size complexity perf doctest". The tests set it empty.
#   GATE_STAMP_FORCE     any value acts outside .claude/worktrees/ too. The tests use it.
#   PRE_COMMIT_FROM_REF, PRE_COMMIT_TO_REF, PRE_COMMIT_REMOTE_BRANCH
#                        set by pre-commit for a pre-push hook: the remote tip, the pushed tip,
#                        and the branch pushed to.

set -euo pipefail

usage() {
    echo "usage: gate_stamp.sh record <gate>... | list | drop <gate>... | waive <gate> <reason>" \
        "| readers <file> | check" >&2
    exit 2
}

top=$(git rev-parse --show-toplevel)
gitdir=$(git rev-parse --absolute-git-dir)
stamps="$gitdir/gate_stamps"
waivers="$gitdir/gate_waivers"
pushed="$gitdir/gate_pushed"
ref=${GATE_STAMP_REF:-origin/dev}
required=${GATE_STAMP_REQUIRED-jet size complexity perf doctest}

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

# Whether the gate "$1" reads the file "$2".
reads() {
    local r
    r=$(readers "$2")
    [ "$r" = "*" ] && return 0
    case " $r " in
    *" $1 "*) return 0 ;;
    esac
    return 1
}

# Whether the file "$1" holds a line for the gate "$2".
names_gate() {
    [ -f "$1" ] && awk -F '\t' -v g="$2" '$1 == g { found = 1 } END { exit !found }' "$1"
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

# Refuse the push when a stamp's base and the remote tip differ in a file the push changes, or in a
# src/ or ext/ directory the push changes, or when a push that changes src/ or ext/ lacks a stamp of
# a required gate. A plain run, outside pre-commit, checks HEAD against the ref.
check() {
    acts || return 0
    local protected=${GATE_STAMP_BRANCH:-refs/heads/dev}
    [ "${PRE_COMMIT_REMOTE_BRANCH:-$protected}" = "$protected" ] || return 0
    local from=${PRE_COMMIT_FROM_REF:-$(git rev-parse "$ref")}
    local to=${PRE_COMMIT_TO_REF:-$(git rev-parse HEAD)}
    # A new remote branch has no tip, and a tip this clone has not fetched makes the push a
    # non-fast-forward that git refuses by itself.
    case "$from" in
    0000000000000000000000000000000000000000) return 0 ;;
    esac
    git cat-file -e "$from^{commit}" 2>/dev/null || return 0

    # The branch's own files run from the merge-base, so a branch not yet rebased does not count
    # the sibling's changes as its own.
    local own dirs
    own=$(git diff --name-only --no-renames "$(git merge-base "$from" "$to")" "$to")
    # `grep` exits 1 when the branch changes nothing under src/ or ext/, and `pipefail` would make
    # that exit the script's.
    dirs=$(printf '%s\n' "$own" | { grep -E '^(src|ext)/' || true; } | xargs -r -n 1 dirname |
        sort -u)

    local stale="" names=""
    if [ -s "$stamps" ]; then
        while IFS=$'\t' read -r gate base; do
            [ -n "$gate" ] || continue
            [ "$base" != "$from" ] || continue
            local moved hit
            moved=$(moved_files "$base" "$from") || moved=""
            hit=$(printf '%s\n' "$moved" | while read -r f; do
                [ -n "$f" ] || continue
                reads "$gate" "$f" || continue
                if grep -qxF -- "$f" <<<"$own" ||
                    grep -qxF -- "$(dirname "$f")" <<<"$dirs"; then
                    echo "$f"
                fi
            done)
            if [ -n "$hit" ]; then
                stale+="  $gate, measured on ${base:0:10}. Since then origin/dev changed:"$'\n'
                stale+=$(printf '%s\n' "$hit" | sed 's/^/      /')$'\n'
                names+=" $gate"
            fi
        done <"$stamps"
    fi

    local absent=""
    if [ -n "$dirs" ]; then
        for gate in $required; do
            if ! names_gate "$stamps" "$gate" && ! names_gate "$waivers" "$gate"; then
                absent+=" $gate"
            fi
        done
    fi

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
        echo "This push changes src/ or ext/, and no stamp records these gates:$absent"
        echo "Run each one (the sweep-conform skill, § Land the branch), or record why it does not"
        echo "apply: code_health/gate_stamp.sh waive <gate> <reason>."
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
check) check ;;
*) usage ;;
esac
