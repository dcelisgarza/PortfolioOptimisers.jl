#!/usr/bin/env bash
#
# The record of a review: a GitHub review, and the commit status `Review` on the head (ADR 0240).
#
#     code_health/review_stamp.sh post <pr> <success|failure> <body file>
#     code_health/review_stamp.sh carry <pr>
#     code_health/review_stamp.sh patch <pr>
#
# GitHub does not let an author approve their own pull request, and one account does the work of
# this repository. So the record is a review of the kind COMMENT, and a status that a merge can
# wait for. `post` writes both. The last line of the review holds the patch id of the diff of the
# pull request against its base, so a stack layer records its own layer alone.
#
# A new head has no status. `carry` gives it the verdict of the last review when the patch id did
# not change, as after a merge of `main` or a rebase by GitHub. When the patch id changed, the head
# stays without a status, and a session reviews what changed. `.github/workflows/Route.yml` runs
# `carry` on each push. Only a review by the owner counts, because anybody can review a pull request
# of a public repository.
#
# `--verbatim` keeps whitespace in the patch id: an indentation is a change in Markdown and TOML.

set -euo pipefail

usage() {
    echo "usage: review_stamp.sh post <pr> <success|failure> <body file> | carry <pr> | patch <pr>" >&2
    exit 2
}

patch_id() {
    local pr=$1 base sha
    base=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq .base.ref </dev/null)
    sha=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq .head.sha </dev/null)
    git fetch -q origin "$base" "refs/pull/$pr/head"
    git rev-parse -q --verify "$sha^{commit}" >/dev/null
    git fetch -q origin "+refs/heads/$base:refs/remotes/origin/$base"
    git diff "origin/$base...$sha" | git patch-id --verbatim | cut -d' ' -f1
}

status() {
    local sha=$1 state=$2 patch=$3
    gh api "repos/{owner}/{repo}/statuses/$sha" -f state="$state" -f context=Review \
       -f description="The review of patch ${patch:0:12}" --silent </dev/null
}

post() {
    local pr=$1 state=$2 file=$3 sha patch body
    [[ $state == success || $state == failure ]] || usage
    [ -s "$file" ] || { echo "'$file' holds no review." >&2; exit 2; }
    sha=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq .head.sha </dev/null)
    patch=$(patch_id "$pr")
    [ -n "$patch" ] || { echo "#$pr has no diff." >&2; exit 1; }
    body=$(mktemp)
    { cat "$file"; printf '\n\n<!-- review-patch %s %s -->\n' "$patch" "$state"; } >"$body"
    gh pr review "$pr" --comment --body-file "$body" </dev/null
    rm -f "$body"
    status "$sha" "$state" "$patch"
    echo "#$pr at ${sha:0:10}: Review is $state for patch ${patch:0:12}."
}

carry() {
    local pr=$1 sha patch last
    sha=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq .head.sha </dev/null)
    if gh api "repos/{owner}/{repo}/commits/$sha/status" \
          --jq '.statuses[].context' </dev/null | grep -qx Review; then
        echo "#$pr at ${sha:0:10} has a Review status."
        return 0
    fi
    last=$(gh api --paginate "repos/{owner}/{repo}/pulls/$pr/reviews" \
              --jq '.[] | select(.author_association == "OWNER") | .body' </dev/null |
        grep -oE '<!-- review-patch [0-9a-f]+ (success|failure) -->' | tail -1 || true)
    if [ -z "$last" ]; then
        echo "#$pr has no review."
        return 0
    fi
    read -r _ _ was state _ <<<"$last"
    patch=$(patch_id "$pr")
    if [ "$patch" = "$was" ]; then
        status "$sha" "$state" "$patch"
        echo "#$pr at ${sha:0:10}: the patch did not change, so Review stays $state."
    else
        echo "#$pr at ${sha:0:10}: the patch changed after the review. Review what changed."
    fi
}

case "${1:-}" in
post) [ $# -eq 4 ] || usage; post "$2" "$3" "$4" ;;
carry) [ $# -eq 2 ] || usage; carry "$2" ;;
patch) [ $# -eq 2 ] || usage; patch_id "$2" ;;
*) usage ;;
esac
