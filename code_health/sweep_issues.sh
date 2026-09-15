#!/usr/bin/env bash
#
# The tracker's half of the sweep job of ADR 0084.
#
#     code_health/sweep_issues.sh dump  [--out <dir>]
#     code_health/sweep_issues.sh apply [--out <dir>] [--dry-run]
#
# `dump` asks the tracker what exists. `apply` reopens and opens what the plan names. Between the
# two runs `code_health/sweep_triage.jl`, which decides everything and writes to the tracker
# nothing.
#
# This file exists so that CI and a person run ONE copy of the tracker half. `.github/workflows/
# Sweep.yml` calls it, and so does a person who is filing an addition by hand. A second copy of
# `gh issue create` in a skill or in a runbook would drift from this one in silence, and the drift
# would only show as a wrongly parented issue somebody has to undo.
#
# Every `gh` call carries `</dev/null`. `gh` reads its standard input when a flag is absent, and a
# loop that shares its standard input with a `while read` swallows the rest of the plan.

set -euo pipefail

# A fork never files into anybody's tracker, and the repository etiquette of `CLAUDE.md` forbids
# posting outside `dcelisgarza`'s repositories. The workflow guards the whole job on the same
# name; this guard is what makes a local run safe, where no workflow condition stands between the
# person and the tracker.
readonly HOME_REPO="dcelisgarza/PortfolioOptimisers.jl"

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat >&2 <<'EOF'
usage: sweep_issues.sh dump  [--out <dir>]
       sweep_issues.sh apply [--out <dir>] [--dry-run]

  dump   write <dir>/maps.tsv and <dir>/existing.tsv from the tracker.
  apply  reopen <dir>/reopen.tsv, then open and parent <dir>/plan.tsv.

  --out <dir>  the plan directory. Default: code_health/_sweep.
  --dry-run    print what apply would do, and do none of it.
EOF
    exit 2
}

# The workflow already holds the name in `GITHUB_REPOSITORY`; a local run asks `gh`.
current_repo() {
    if [ -n "${GITHUB_REPOSITORY:-}" ]; then
        echo "$GITHUB_REPOSITORY"
    else
        gh repo view --json nameWithOwner --jq .nameWithOwner </dev/null
    fi
}

require_home_repo() {
    local repo
    repo="$(current_repo)"
    if [ "$repo" != "$HOME_REPO" ]; then
        echo "sweep_issues.sh refuses to touch the tracker of '$repo'." >&2
        echo "It files into '$HOME_REPO' alone." >&2
        exit 1
    fi
    echo "$repo"
}

do_dump() {
    local out="$1" repo
    repo="$(require_home_repo)"
    mkdir -p "$out"
    # The maps dump carries every `wayfinder:map` issue in the tracker. `sweep_triage.jl` picks the
    # child maps out of it by title and reads the umbrella by number, so no number is written down
    # here and none can go stale.
    gh issue list --repo "$repo" --label wayfinder:map --state all --limit 200 \
       --json number,state,title \
       --template '{{range .}}{{.number}}{{"\t"}}{{.state}}{{"\t"}}{{.title}}{{"\n"}}{{end}}' \
       </dev/null > "$out/maps.tsv"
    gh issue list --repo "$repo" --label sweep --state all --limit 500 \
       --json number,state,title \
       --template '{{range .}}{{.number}}{{"\t"}}{{.state}}{{"\t"}}{{.title}}{{"\n"}}{{end}}' \
       </dev/null > "$out/existing.tsv"
    echo "The tracker holds $(wc -l < "$out/maps.tsv") map(s)."
    echo "It holds $(wc -l < "$out/existing.tsv") sweep issue(s)."
}

do_apply() {
    local out="$1" dry="$2" repo n stem path parent url child count=0

    if [ ! -f "$out/plan.tsv" ] || [ ! -f "$out/reopen.tsv" ]; then
        echo "No plan in '$out'. Run sweep_triage.jl first." >&2
        exit 1
    fi

    if [ "$dry" = "yes" ]; then
        echo "Dry run. Nothing is opened and nothing is reopened."
        echo "--- the issues that would be reopened ---"
        cat "$out/reopen.tsv"
        echo "--- the sub-issues that would be opened ---"
        cat "$out/plan.tsv"
        return 0
    fi

    repo="$(require_home_repo)"

    # The map is reopened BEFORE its sub-issues are opened, which is the order #404's rule states.
    # `reopen.tsv` lists only issues the dump reported CLOSED, because `gh issue reopen` fails on an
    # issue that is already open.
    while read -r n; do
        [ -n "$n" ] || continue
        gh issue reopen --repo "$repo" "$n" </dev/null
        echo "Reopened #$n."
    done < "$out/reopen.tsv"

    while IFS=$'\t' read -r stem path parent; do
        [ -n "$stem" ] || continue
        url=$(gh issue create --repo "$repo" --label sweep \
                --title "$(cat "$out/$stem.title")" \
                --body-file "$out/$stem.body" </dev/null)
        echo "Opened $url for $path."
        # The sub-issues endpoint takes the child's DATABASE id, which is not the number in the
        # URL. `gh issue create` prints the URL alone, so the id is read back by number.
        child=$(gh api "repos/$repo/issues/${url##*/}" --jq .id </dev/null)
        gh api "repos/$repo/issues/$parent/sub_issues" \
           -F sub_issue_id="$child" --silent </dev/null
        echo "Attached it to child map #$parent."
        count=$((count + 1))
    done < "$out/plan.tsv"
    echo "Opened $count sub-issue(s)."
}

main() {
    [ $# -ge 1 ] || usage
    local cmd="$1" out="$ROOT/code_health/_sweep" dry="no"
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --out) [ $# -ge 2 ] || usage; out="$2"; shift 2 ;;
            --dry-run) dry="yes"; shift ;;
            *) usage ;;
        esac
    done
    case "$cmd" in
        dump) do_dump "$out" ;;
        apply) do_apply "$out" "$dry" ;;
        *) usage ;;
    esac
}

main "$@"
