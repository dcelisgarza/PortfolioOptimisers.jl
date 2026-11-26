#!/usr/bin/env bash
#
# The record of a review: a GitHub review, and the commit status `Review` on the head (ADR 0240).
#
#     code_health/review_stamp.sh post <pr> <sha> <success|failure> <body file>
#     code_health/review_stamp.sh carry <pr>
#     code_health/review_stamp.sh patch <pr>
#
# GitHub does not let an author approve their own pull request, and one account does the work of
# this repository. So the record is a review of the kind COMMENT, and a status that a merge can
# wait for. `post` writes both. The last line of the review holds the patch id of the diff of the
# pull request against its base, so a stack layer records its own layer alone. `<sha>` is the head
# that the reviewer read. `post` refuses when the head moved, so that no unread commit takes a verdict.
#
# A new head has no status. `carry` gives it the verdict of the last review when the patch id did
# not change, as after a merge of `main` or a rebase by GitHub. When the patch id changed, the head
# stays without a status, and a session reviews what changed. A change of the base changes the diff and
# keeps the head, so `carry` then sets the status to `pending`. `.github/workflows/Route.yml` runs
# `carry` on each push and on each edit. Only a review by the owner counts, because anybody can review a pull request
# of a public repository.
#
# `--verbatim` keeps whitespace in the patch id: an indentation is a change in Markdown and TOML. The
# diff comes from `diff-tree --full-index`, which reads no `diff.*` setting, so that a session and a
# runner get the same id. The id of a binary file is a hash of its index line.

set -euo pipefail

usage() {
    echo "usage: review_stamp.sh post <pr> <sha> <success|failure> <body file> | carry <pr> | patch <pr>" >&2
    exit 2
}

head_of() {
    gh api "repos/{owner}/{repo}/pulls/$1" --jq .head.sha </dev/null
}

# The patch id of `<pr> <sha>`. Bash clears `-e` in a command substitution, so each step returns.
patch_id() {
    local pr=$1 sha=$2 base from
    base=$(gh api "repos/{owner}/{repo}/pulls/$pr" --jq .base.ref </dev/null) || return
    git fetch -q origin "+refs/heads/$base:refs/remotes/origin/$base" "refs/pull/$pr/head" || return
    from=$(git merge-base "origin/$base" "$sha") || return
    git diff-tree -p --full-index "$from" "$sha" | git patch-id --verbatim | cut -d' ' -f1
}

describe() {
    echo "The review of patch ${1:0:12}"
}

status() {
    local sha=$1 state=$2 description=$3
    gh api "repos/{owner}/{repo}/statuses/$sha" -f state="$state" -f context=Review \
       -f description="$description" --silent </dev/null
}

post() {
    local pr=$1 read=$2 state=$3 file=$4 sha patch body
    [[ $state == success || $state == failure ]] || usage
    [[ $read =~ ^[0-9a-f]{7,40}$ ]] || usage
    [ -s "$file" ] || { echo "'$file' holds no review." >&2; exit 2; }
    sha=$(head_of "$pr")
    [[ $sha == "$read"* ]] ||
        { echo "The head of #$pr is ${sha:0:10}, not $read. Review what changed." >&2; exit 1; }
    patch=$(patch_id "$pr" "$sha")
    [ -n "$patch" ] || { echo "#$pr has no diff." >&2; exit 1; }
    body=$(mktemp)
    { cat "$file"; printf '\n\n<!-- review-patch %s %s -->\n' "$patch" "$state"; } >"$body"
    gh api "repos/{owner}/{repo}/pulls/$pr/reviews" -f event=COMMENT -f commit_id="$sha" \
       -F body=@"$body" --silent </dev/null
    rm -f "$body"
    status "$sha" "$state" "$(describe "$patch")"
    echo "#$pr at ${sha:0:10}: Review is $state for patch ${patch:0:12}."
}

carry() {
    local pr=$1 sha patch last was state seen
    sha=$(head_of "$pr")
    last=$(gh api --paginate "repos/{owner}/{repo}/pulls/$pr/reviews" \
              --jq '.[] | select(.author_association == "OWNER") | .body' </dev/null |
        grep -oE '<!-- review-patch [0-9a-f]+ (success|failure) -->' | tail -1 || true)
    if [ -z "$last" ]; then
        echo "#$pr has no review."
        return 0
    fi
    read -r _ _ was state _ <<<"$last"
    patch=$(patch_id "$pr" "$sha")
    seen=$(gh api "repos/{owner}/{repo}/commits/$sha/status" \
              --jq '.statuses[] | select(.context == "Review") | .state + " " + .description' </dev/null)
    if [ "$patch" = "$was" ]; then
        [ "$seen" = "$state $(describe "$patch")" ] || status "$sha" "$state" "$(describe "$patch")"
        echo "#$pr at ${sha:0:10}: the patch did not change, so Review stays $state."
    else
        # A head with no status stays so. One with a status has a base that changed under it.
        [ -z "$seen" ] || status "$sha" pending "The patch changed after the review"
        echo "#$pr at ${sha:0:10}: the patch changed after the review. Review what changed."
    fi
}

main() {
    case "${1:-}" in
    post) [ $# -eq 5 ] || usage; post "$2" "$3" "$4" "$5" ;;
    carry) [ $# -eq 2 ] || usage; carry "$2" ;;
    patch) [ $# -eq 2 ] || usage; patch_id "$2" "$(head_of "$2")" ;;
    *) usage ;;
    esac
}

# A test sources the file, so `main` runs only when the file is run.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
