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
