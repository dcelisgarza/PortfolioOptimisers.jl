#!/usr/bin/env python3
"""Compare the live ruleset of `main` with `.github/rulesets/protect-main.json` (ADR 0220).

    gh api repos/{owner}/{repo}/rulesets/<id> | python3 -I .github/ruleset_check.py <file>

It prints each value that differs and exits 1. A key that the file does not hold is not compared, so
`bypass_actors`, which the API shows to an admin alone, is out of scope. A list of rules or of checks
is compared by name, so its order does not count.
"""
import json
import sys


def named(items):
    """A list of objects with a name, as a mapping, or None."""
    for key in ("type", "context", "tool"):
        if items and all(isinstance(i, dict) and key in i for i in items):
            return {i[key]: i for i in items}
    return None


def differences(want, have, where=""):
    if isinstance(want, dict) and isinstance(have, dict):
        for key, value in want.items():
            at = f"{where}.{key}" if where else key
            if key not in have:
                yield f"{at}: the file holds {value!r}, and the ruleset has no such key"
            else:
                yield from differences(value, have[key], at)
        return
    if isinstance(want, list) and isinstance(have, list):
        w, h = named(want), named(have)
        if w is not None and (h is not None or not have):
            h = h or {}
            yield from differences(w, h, where)
            for key in h.keys() - w.keys():
                yield f"{where}.{key}: the ruleset holds it, and the file does not"
            return
        if sorted(map(repr, want)) == sorted(map(repr, have)):
            return
    if want != have:
        yield f"{where}: the file holds {want!r}, and the ruleset holds {have!r}"


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        want = json.load(f)
    found = list(differences(want, json.load(sys.stdin)))
    for line in found:
        print(line)
    if not found:
        print("The ruleset agrees with the file.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
