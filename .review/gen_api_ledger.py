#!/usr/bin/env python3
"""Regenerate .review/api_ledger.txt for PR #261.

Run from the repository root:  python3 .review/gen_api_ledger.py > .review/api_ledger.txt

Parses every `export` statement and every `struct` declaration at two refs and
diffs them. It reads the source rather than the diff, because a line-oriented
diff read cannot tell a multi-line `export` statement's unchanged continuation
lines from removals -- the fault that made the B5, B6, B7 and B10 API notes
wrong. It also reads a field declaration in every form this codebase uses:
bare, `::`-bounded, `<:`-bounded (@concrete), and prefixed by a propagation
macro (@fprop, @vprop, @pprop, @wprop, @cprop). Missing that last form is what
made the first ledger report four field removals that never happened.

It reports field changes twice, because the two readings answer different
questions. The SET reading says which names a type gained and lost. The
POSITIONAL reading says whether a stale positional call still matches: every
`@concrete` type has a positional constructor, so a field inserted in a
non-final position rebinds every argument from that slot onwards, while the set
reading shows only a plain addition.
"""
import re, subprocess, sys

def files(ref):
    out = subprocess.run(['git','ls-tree','-r','--name-only',ref,'src/'],
                         capture_output=True, text=True, check=True).stdout.split()
    return [f for f in out if f.endswith('.jl')]

def blob(ref, path):
    return subprocess.run(['git','show',f'{ref}:{path}'],
                          capture_output=True, text=True, check=True).stdout

def exports(ref):
    names=set()
    for f in files(ref):
        lines = blob(ref,f).splitlines()
        i=0
        while i < len(lines):
            m = re.match(r'^\s*export\s+(.*)$', lines[i])
            if m:
                buf = m.group(1)
                # consume continuations while the accumulated text ends with a comma
                while buf.rstrip().endswith(',') and i+1 < len(lines):
                    i += 1
                    buf += ' ' + lines[i].strip()
                buf = buf.split('#')[0]
                for n in buf.split(','):
                    n = n.strip()
                    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_!]*', n):
                        names.add(n)
            i += 1
    return names


DECL = re.compile(r'^(?:@\w+\s+)*(?:mutable\s+)?struct\s+([A-Za-z_]\w*)')
# A field may be prefixed by a propagation macro (@fprop, @vprop, @pprop, @wprop,
# @cprop) and may be bounded with `::` or `@concrete`'s `<:`.
PROP = r'@(?:fprop|vprop|pprop|wprop|cprop)\s+'
FIELD = re.compile(r'^    (?:' + PROP + r')?([A-Za-z_][A-Za-z0-9_]*)\s*(?:(?:::|<:).*)?$')
KW = {'end','return','else','elseif','begin','let','do','break','continue','true','false','nothing'}
CTOR  = re.compile(r'^    function\b')

def structs(ref):
    res = {}
    for f in files(ref):
        lines = blob(ref, f).split('\n')
        # drop triple-quoted docstring regions
        keep, indoc = [], False
        for ln in lines:
            n = ln.count('"""')
            if indoc:
                if n: indoc = False
                keep.append('')
                continue
            if n == 1:
                indoc = True; keep.append(''); continue
            keep.append(ln)
        lines = keep
        for i, ln in enumerate(lines):
            m = DECL.match(ln)
            if not m: continue
            # a one-line singleton (`struct Foo <: Bar end`) has no fields; without
            # this the scan runs on into the NEXT struct and steals its field list
            if re.search(r'\bend\s*$', ln):
                res.setdefault(m.group(1), [])
                continue
            fields, j = [], i + 1
            while j < len(lines) and lines[j] != 'end':
                if CTOR.match(lines[j]): break
                fm = FIELD.match(lines[j])
                if fm and fm.group(1) not in KW: fields.append(fm.group(1))
                j += 1
            res.setdefault(m.group(1), fields)
    return res



def report(base, head, label):
    be, he = exports(base), exports(head)
    bs, hs = structs(base), structs(head)
    print(f"=== {label}: {base} -> {head} ===")
    print(f"Exported names: {len(be)} -> {len(he)}")
    print(f"\nREMOVED EXPORTS ({len(be-he)}):")
    for n in sorted(be-he): print("   -", n)
    gone = sorted((set(bs)-set(hs)) & be)
    print(f"\nEXPORTED TYPES THAT NO LONGER EXIST ({len(gone)}): {gone}")
    print(f"\nADDED EXPORTS ({len(he-be)}):")
    print("   " + ", ".join(sorted(he-be)))
    rows = []
    for n in sorted(set(bs) & set(hs)):
        if n not in he: continue
        rm = [x for x in bs[n] if x not in hs[n]]
        ad = [x for x in hs[n] if x not in bs[n]]
        if rm or ad: rows.append((n, rm, ad))
    lose = [r for r in rows if r[1]]
    print(f"\nFIELD CHANGES ON EXPORTED TYPES ({len(rows)}; {len(lose)} lose a field):")
    for n, rm, ad in rows:
        mark = "  <-- LOSES A FIELD" if rm else ""
        print(f"   {n}: removed={rm or '-'}  added={ad or '-'}{mark}")
    order_report(bs, hs, he)


def classify(b, h):
    """Classify a field list change by POSITION, not by set membership.

    Set membership cannot see a reorder, and every `@concrete` type also has a
    positional constructor, so a field inserted in a non-final position reads as
    a plain addition while every positional argument from that slot onwards
    rebinds. Returns (kind, slot) where slot is the 1-based first divergent
    position, or None when the lists agree as far as the shorter one runs.
    """
    slot = None
    for i in range(min(len(b), len(h))):
        if b[i] != h[i]:
            slot = i + 1
            break
    if b == h:
        return ("IDENTICAL", None)
    if slot is None:
        return ("PURE APPEND", None) if len(h) > len(b) else ("TRUNCATION", None)
    if len(b) == len(h):
        return ("SAME ARITY REORDER", slot)
    return ("ARITY CHANGE", slot)


def order_report(bs, hs, he):
    """Report the POSITIONAL shape of every exported type present at both refs."""
    buckets = {"IDENTICAL": [], "PURE APPEND": [], "SAME ARITY REORDER": [],
               "ARITY CHANGE": [], "TRUNCATION": []}
    for n in sorted(set(bs) & set(hs)):
        if n not in he: continue
        b, h = bs[n], hs[n]
        kind, slot = classify(b, h)
        buckets[kind].append((n, b, h, slot))
    total = sum(len(v) for v in buckets.values())
    print(f"\nPOSITIONAL SHAPE OF EXPORTED TYPES ({total} present at both refs):")
    NOTE = {"IDENTICAL": "a positional call binds exactly as before",
            "PURE APPEND": "a positional call binds exactly as before",
            "SAME ARITY REORDER": "a stale positional call still MATCHES arity and rebinds",
            "ARITY CHANGE": "a stale positional call may match at the shorter arity",
            "TRUNCATION": "the type lost trailing fields"}
    for kind in ("IDENTICAL", "PURE APPEND", "SAME ARITY REORDER", "ARITY CHANGE",
                 "TRUNCATION"):
        rows = buckets[kind]
        print(f"   {kind} ({len(rows)}) -- {NOTE[kind]}:")
        if kind == "IDENTICAL":
            print("      (not listed)")
            continue
        for n, b, h, slot in rows:
            if slot is None and kind == "TRUNCATION":
                print(f"      {n}: {len(b)} -> {len(h)} fields, lost trailing {b[len(h):]}")
            elif slot is None:
                print(f"      {n}: {len(b)} -> {len(h)} fields, added {h[len(b):]}")
            else:
                print(f"      {n}: {len(b)} -> {len(h)} fields, first divergence at slot "
                      f"{slot}, {b[slot-1]} -> {h[slot-1]}")


if __name__ == "__main__":
    print("PUBLIC API LEDGER — PR #261")
    print("-" * 78)
    print("""Regenerate with:  python3 .review/gen_api_ledger.py > .review/api_ledger.txt

Two bases, because they answer different questions.

  * v0.28.0 -> HEAD is the RELEASE-NOTES base. v0.28.0 is the last released
    version, so this is what a user upgrading actually faces.
  * c975293745 -> HEAD is the REVIEW base, the branch point. c975293745 is an
    ancestor of the v0.28.0 tag, so part of this diff is already released and is
    NOT a 0.29.0 break -- `cle_pr` -> `x_src` is the clear case.

The generator parses every `export` statement and every `struct` declaration at
each ref. A multi-line export is read as one statement. A field declaration is
read whether it is bare, `::`-bounded, `<:`-bounded (@concrete), or carries a
propagation macro (@fprop, @vprop, @pprop, @wprop, @cprop). The FIRST ledger
missed the macro-prefixed form, which is why it reported four field removals
that never happened -- FactorBlackLittermanPrior `sets` and `w`,
FactorRiskBudgeting `sets`, BayesianBlackLittermanPrior `sets` -- and missed
that AugmentedBlackLittermanPrior collapses `a_sets`/`f_sets` into one `sets`.
""")
    report("v0.28.0", "HEAD", "RELEASE NOTES BASE")
    print()
    report("c975293745", "HEAD", "REVIEW BASE")
