#!/usr/bin/env bash
#
# Gate stamps, and the push guard that reads them.
#
#     code_health/gate_stamp.sh record <gate>...   a gate measured the tree on the current base
#     code_health/gate_stamp.sh list               the stamps of this worktree
#     code_health/gate_stamp.sh drop <gate>...     forget a stamp that does not bear on the push
#     code_health/gate_stamp.sh check              the pre-push guard of .pre-commit-config.yaml
#
# A gate measures a tree: the merge-base of the branch with origin/dev, plus the branch. When a
# sibling commit lands on origin/dev and the branch rebases onto it, the tree changes, and a gate
# measured before the rebase measured a program nobody will ship. `check` refuses the push when
# origin/dev changed, after a gate measured the tree, a file the branch also changes, or a file in
# the same directory under src/ or ext/. Re-run the gates it names on the rebased tree, or `drop`
# the stamp of a gate the change cannot reach.
#
# `code_health/CodeHealth.jl` records every check and refresh of the gate scripts, and
# `test/run_files.jl` records every test file it runs. A doctest run records itself with
# `record doctest`.
#
# Stamps live in the git directory of the worktree, so each worktree has its own and none is
# committed. The guard acts only in a worktree under `.claude/worktrees/`, where parallel sessions
# work. The `dev` checkout of the maintainer records and checks nothing.
#
# Environment:
#   GATE_STAMP_REF     the ref a gate is measured against. Default: origin/dev.
#   GATE_STAMP_BRANCH  the remote branch the guard protects. Default: refs/heads/dev.
#   GATE_STAMP_FORCE   any value acts outside .claude/worktrees/ too. The tests use it.
#   PRE_COMMIT_FROM_REF, PRE_COMMIT_TO_REF, PRE_COMMIT_REMOTE_BRANCH
#                      set by pre-commit for a pre-push hook: the remote tip, the pushed tip,
#                      and the branch pushed to.

set -euo pipefail

usage() {
    echo "usage: gate_stamp.sh record <gate>... | list | drop <gate>... | check" >&2
    exit 2
}

top=$(git rev-parse --show-toplevel)
stamps="$(git rev-parse --absolute-git-dir)/gate_stamps"
ref=${GATE_STAMP_REF:-origin/dev}

acts() {
    case "$top" in
    */.claude/worktrees/*) return 0 ;;
    esac
    [ -n "${GATE_STAMP_FORCE:-}" ]
}

# Write the stamps, one `<gate> TAB <base>` line per gate, without the gates named in "$@".
without() {
    if [ -f "$stamps" ]; then
        awk -F '\t' 'BEGIN { for (i = 1; i < ARGC; i++) drop[ARGV[i]] = 1; ARGC = 1 }
                     !($1 in drop)' "$@" <"$stamps"
    fi
}

record() {
    [ $# -gt 0 ] || usage
    acts || return 0
    local base
    base=$(git merge-base "$ref" HEAD 2>/dev/null) || return 0
    local next
    next=$(without "$@")
    for gate in "$@"; do
        next+=$'\n'"$gate"$'\t'"$base"
    done
    printf '%s\n' "$next" | sed '/^$/d' >"$stamps"
}

drop() {
    [ $# -gt 0 ] || usage
    acts || return 0
    local next
    next=$(without "$@")
    printf '%s\n' "$next" | sed '/^$/d' >"$stamps"
}

list() {
    if [ -f "$stamps" ]; then
        cat "$stamps"
    fi
}

# Refuse the push when a stamp's base and the remote tip differ in a file the push changes, or in a
# src/ or ext/ directory the push changes. A plain run, outside pre-commit, checks HEAD against
# the ref.
check() {
    acts || return 0
    [ -s "$stamps" ] || return 0
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

    local stale=""
    while IFS=$'\t' read -r gate base; do
        [ -n "$gate" ] || continue
        [ "$base" != "$from" ] || continue
        local moved hit
        moved=$(git diff --name-only --no-renames "$base" "$from" 2>/dev/null) || moved=""
        hit=$(printf '%s\n' "$moved" | while read -r f; do
            [ -n "$f" ] || continue
            if printf '%s\n' "$own" | grep -qxF -- "$f" ||
                printf '%s\n' "$dirs" | grep -qxF -- "$(dirname "$f")"; then
                echo "$f"
            fi
        done)
        if [ -n "$hit" ]; then
            stale+="  $gate, measured on ${base:0:10}. Since then origin/dev changed:"$'\n'
            stale+=$(printf '%s\n' "$hit" | sed 's/^/      /')$'\n'
        fi
    done <"$stamps"

    if [ -n "$stale" ]; then
        echo "The push target moved under these gates after they measured the tree:"
        printf '%s' "$stale"
        echo "Re-run each gate on the rebased tree, then push. A gate that the change cannot"
        echo "reach can be forgotten with \`code_health/gate_stamp.sh drop <gate>\`."
        return 1
    fi
}

[ $# -gt 0 ] || usage
cmd=$1
shift
case "$cmd" in
record) record "$@" ;;
list) list ;;
drop) drop "$@" ;;
check) check ;;
*) usage ;;
esac
