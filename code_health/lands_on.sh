#!/usr/bin/env bash
#
# The `**Lands on:**` line of ADR 0240: its grammar, and the checks of it.
#
#     code_health/lands_on.sh parse < body.md   print `<route> TAB <target>` of one issue body
#     code_health/lands_on.sh issue <number>    check the line of an issue against the repository
#     code_health/lands_on.sh pr <number>       check a pull request against the issues it closes
#
# The values of the line, and nothing after them on the line:
#
#     own branch                  a branch from origin/main, and a pull request of its own
#     own branch after <target>   the same, and the work starts when <target> is on main
#     dev                         the shared branch
#     dev after <target>          the same, and the work starts when dev holds <target>
#     same branch #<pr>           the head branch of an open pull request
#     stacked on <target>         a branch from the head of <target>, and a layer of its stack
#     map #<map>                  the branch of a map that lands as a whole
#     one branch dev-<feature>    the line of such a map itself
#     each ticket                 the line of a map whose tickets have a route each
#     no code                     a decision, a measurement or a research ticket that lands nothing
#
# <target> is `#<pr>`, or `dev-<feature>` while that branch has no pull request. The reason for the
# route goes below the line. Backticks around a value do not count.
#
# `.github/workflows/Route.yml` runs `issue` and `pr`, and a session runs them by hand. Exit codes:
# 0 the line holds, 1 it does not, 2 a wrong call, 3 the body has no line.
#
# Every `gh` call reads /dev/null, as `sweep_issues.sh` states.

set -euo pipefail

usage() {
    echo "usage: lands_on.sh parse < body | issue <number> | pr <number>" >&2
    exit 2
}

bad() {
    echo "$*" >&2
    return 1
}

readonly TARGET='(#[0-9]+|dev-[A-Za-z0-9][A-Za-z0-9._-]*)'

# Print `<route> TAB <target>` of the body on standard input. Return 3 when it has no line.
parse() {
    local lines value
    lines=$(tr -d '\r`' | grep -E '^\*\*Lands on:\*\*' || true)
    if [ -z "$lines" ]; then
        echo "The body has no **Lands on:** line." >&2
        return 3
    fi
    if [ "$(printf '%s\n' "$lines" | wc -l)" -gt 1 ]; then
        bad "The body has more than one **Lands on:** line."
        return
    fi
    value=$(printf '%s' "$lines" |
        sed -E 's/^\*\*Lands on:\*\*[[:space:]]*//; s/[[:space:]]+$//; s/[[:space:]]+/ /g')
    if [[ $value =~ ^(own\ branch|dev)$ ]]; then
        printf '%s\t\n' "$value"
    elif [[ $value =~ ^(own\ branch|dev)\ after\ $TARGET$ ]]; then
        printf '%s\t%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    elif [[ $value =~ ^stacked\ on\ $TARGET$ ]]; then
        printf 'stacked on\t%s\n' "${BASH_REMATCH[1]}"
    elif [[ $value =~ ^(same\ branch|map)\ (#[0-9]+)$ ]]; then
        printf '%s\t%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    elif [[ $value =~ ^one\ branch\ (dev-[A-Za-z0-9][A-Za-z0-9._-]*)$ ]]; then
        printf 'one branch\t%s\n' "${BASH_REMATCH[1]}"
    elif [[ $value =~ ^(each\ ticket|no\ code)$ ]]; then
        printf '%s\t\n' "$value"
    else
        bad "\`$value\` is not a route. CLAUDE.md § Parallel sessions lists the values."
    fi
}

# `#<n>` -> `open <head>`, `merged <head>`, `closed <head>`. A `dev-` branch -> `open <branch>`, or
# `gone <branch>` when origin does not have it.
target_state() {
    local target=$1
    if [[ $target == \#* ]]; then
        gh api "repos/{owner}/{repo}/pulls/${target#\#}" \
           --jq '(if .merged_at then "merged" else .state end) + " " + .head.ref' </dev/null ||
            bad "$target is not a pull request."
    elif [ -n "$(gh api "repos/{owner}/{repo}/git/matching-refs/heads/$target" \
                    --jq ".[] | select(.ref == \"refs/heads/$target\") | .ref" </dev/null)" ]; then
        echo "open $target"
    else
        echo "gone $target"
    fi
}

check_issue() {
    local n=$1 parsed route target title state path rc=0 known
    title=$(gh issue view "$n" --json title --jq .title </dev/null)
    parsed=$(gh issue view "$n" --json body --jq .body </dev/null | parse) || return $?
    IFS=$'\t' read -r route target <<<"$parsed"
    case "$route" in
    "stacked on" | "same branch" | "own branch" | dev)
        if [ -n "$target" ]; then
            state=$(target_state "$target") || return 1
            case "$route ${state%% *}" in
            *" open") ;;
            "own branch merged" | "own branch gone" | "dev merged" | "dev gone") ;;
            *) bad "#$n says \`$route $target\`, and $target is ${state%% *}. The route is \`own branch\` now." || rc=1 ;;
            esac
        fi
        ;;
    esac
    # A file that a title names, such as that of a sweep sub-issue, must be where `dev` can get it.
    if [ "$route" = dev ] && [ -z "$target" ]; then
        while read -r path; do
            [ -n "$path" ] || continue
            known=no
            git cat-file -e "origin/dev:$path" 2>/dev/null && known=yes
            git cat-file -e "origin/main:$path" 2>/dev/null && known=yes
            [ "$known" = yes ] ||
                bad "#$n lands on \`dev\`, and neither dev nor main has \`$path\`. Write \`dev after <target>\`." ||
                rc=1
        done < <(grep -oE '(src|ext)/[A-Za-z0-9_/.-]+\.jl' <<<"$title" || true)
    fi
    [ "$rc" = 0 ] && printf '%s\n' "$parsed"
    return $rc
}

# One issue of a pull request: `pr_agrees <pr> <head> <base> <route> <target>`.
pr_agrees() {
    local pr=$1 head=$2 base=$3 route=$4 target=$5 state
    case "$route" in
    "own branch")
        [ "$base" = main ] ||
            bad "the route is \`own branch\`, and the base is \`$base\`. Write \`stacked on\`, or change the base to main." ||
            return 1
        [ -n "$target" ] || return 0
        state=$(target_state "$target") || return 1
        [[ ${state%% *} != open ]] || bad "the work waits for $target, which is open."
        ;;
    "stacked on")
        state=$(target_state "$target") || return 1
        if [ "${state%% *}" = open ]; then
            [ "$base" = "${state#* }" ] ||
                bad "the route is \`stacked on $target\`, and the base is \`$base\`, not \`${state#* }\`."
        else
            [ "$base" = main ] || bad "$target is ${state%% *}, and the base is \`$base\`, not main."
        fi
        ;;
    "same branch")
        [ "#$pr" = "$target" ] || bad "the route is \`same branch $target\`, and this is #$pr."
        ;;
    dev) [ "$head" = dev ] || bad "the route is \`dev\`, and the head is \`$head\`." ;;
    *) return 0 ;;
    esac
}

check_pr() {
    local pr=$1 head base body n parsed route target rc=0
    head=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq .head.ref </dev/null)
    base=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq .base.ref </dev/null)
    body=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq '.body // ""' </dev/null)
    # The batch of `dev` holds many issues, and its own rule is the filing of each one.
    [ "$head" != dev ] || return 0
    # GitHub links a closing keyword only when the base is the default branch, so an upper layer of
    # a stack has no linked issue. The keyword in the body is read instead.
    for n in $(grep -oiE '\b(close[sd]?|fix(e[sd])?|resolve[sd]?):? #[0-9]+' <<<"$body" |
        grep -oE '[0-9]+$' | sort -un); do
        if ! parsed=$(gh issue view "$n" --json body --jq .body </dev/null | parse 2>&1); then
            echo "#$n: $parsed" >&2
            rc=1
            continue
        fi
        IFS=$'\t' read -r route target <<<"$parsed"
        if pr_agrees "$pr" "$head" "$base" "$route" "$target"; then
            echo "#$n: $route${target:+ $target}"
        else
            echo "  That is issue #$n." >&2
            rc=1
        fi
    done
    return $rc
}

main() {
    case "${1:-}" in
    parse) [ $# -eq 1 ] || usage; parse ;;
    issue) [ $# -eq 2 ] || usage; check_issue "$2" ;;
    pr) [ $# -eq 2 ] || usage; check_pr "$2" ;;
    *) usage ;;
    esac
}

# A test sources the file, so `main` runs only when the file is run.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
