#!/usr/bin/env python3
"""Delete from `gh-pages` each released version that is not the newest patch of a newest minor series.

`gh-pages` keeps one directory per released version (`v0.12`, `v0.30.1`, ...) and one alias link
per version. GitHub Pages expands each link into a full copy of its target, so every release adds
about twice the size of one build, and the site passed the 10 GiB limit of Pages once (#1722).

Documenter never deletes a version directory, and it makes the alias links for every directory that
is present, whatever `deploydocs(versions = ...)` lists. So the cap is a deletion, and this script
does it. It keeps the `PAGES_KEEP_MINORS` newest minor series (default 5), and of each kept series
only the directory of its newest patch. It deletes every other version directory. The project
releases once per pull request, so a minor series collects many patches, and each one costs a full
build. The version selector lists only the newest patch of each series (`v#.#`), so a superseded
patch is reachable by its URL alone. Then the script deletes each top-level link that no longer
resolves, and it removes the deleted versions from the selector in `versions.js`. Documenter writes
`versions.js` again at its next deploy, from the directories that are present, so the deleted
versions do not come back.

It refuses to delete a directory that a link with another name points at, such as `stable`.

    python3 .github/pages_prune.py <checkout of gh-pages>

It changes the files of the checkout only. The caller commits and pushes. It prints the name of
each deleted directory, one to a line, and nothing else to stdout.
"""

import os
import re
import shutil
import sys

KEEP = int(os.environ.get("PAGES_KEEP_MINORS", "5"))
VERSION = re.compile(r"^v(\d+)\.(\d+)(?:\.(\d+))?$")
ENTRY = re.compile(r'^\s*"([^"]+)",?\s*$')


def series(name):
    m = VERSION.match(name)
    return (int(m[1]), int(m[2]))


def version(name):
    # A directory of the old layout, such as `v0.12`, has no patch. It sorts below every patch of
    # its series.
    m = VERSION.match(name)
    return (int(m[1]), int(m[2]), -1 if m[3] is None else int(m[3]))


def main(site):
    if KEEP < 1:
        sys.exit(f"::error::PAGES_KEEP_MINORS is {KEEP}. It must be at least 1.")
    names = os.listdir(site)
    path = lambda name: os.path.join(site, name)
    dirs = [n for n in names if VERSION.match(n) and os.path.isdir(path(n)) and not os.path.islink(path(n))]
    kept = set(sorted({series(d) for d in dirs}, reverse=True)[:KEEP])
    newest = {s: max((d for d in dirs if series(d) == s), key=version) for s in kept}
    drop = sorted((d for d in dirs if newest.get(series(d)) != d), key=version)

    # A link such as `stable` names a version that a reader goes to without a number. Deleting its
    # target breaks the front door of the site, so stop instead.
    named = {n: os.path.realpath(path(n)) for n in names if os.path.islink(path(n)) and not VERSION.match(n)}
    for link, target in named.items():
        for d in drop:
            if target == os.path.realpath(path(d)):
                sys.exit(f"::error::`{link}` points at `{d}`, which is not the newest patch of the {KEEP} newest minor series.")

    for d in drop:
        shutil.rmtree(path(d))
        print(d)
    for n in names:
        if os.path.islink(path(n)) and not os.path.exists(path(n)):
            os.remove(path(n))

    js = path("versions.js")
    if os.path.exists(js):
        with open(js) as fh:
            lines = fh.readlines()
        keep = [l for l in lines if not (ENTRY.match(l) and not os.path.exists(path(ENTRY.match(l)[1])))]
        if keep != lines:
            with open(js, "w") as fh:
                fh.writelines(keep)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
