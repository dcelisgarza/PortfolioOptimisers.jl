---
status: accepted
---

# The baseline's row set is total, and a rename pairs by equal measurement

## Context

The gate is a ratchet against a committed baseline. A ratchet compares a measured number against a
recorded one, so it has no rule for a file it has never measured.

[ADR 0071](0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md) settled the file that
**disappears**: the row names nothing, and the gate turns red. It left the file that **appears**
unsettled. [ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) then ruled that every
file in scope gets a row, zeros included, which narrowed the question to one case. A file the
baseline does not name is either out of scope, or it is new.
[ADR 0072](0072-the-complexity-gate-measures-src-and-ext.md) already turns the gate red on a
tracked file that is neither measured nor a named Unmeasured Path. So a new file under `src/` falls
between the two rules: ADR 0072's coverage assertion passes, and ADR 0073's baseline has no row.

Three measurements taken against the working tree at `9d420aa7b6` decide the answer.

1. **A rename is not a corner case in this repository.** Over the last 200 commits, 28 commits add
   a `.jl` file under `src/` or `ext/` and **21 commits rename one**. The `NN_` numbering scheme
   makes a renumber a mass rename: one commit renamed **114** files and a second renamed 110.
2. **Over half the files in scope stand above a threshold today.** Measured per file as the maximum
   over definitions, with CodeComplexity 0.2.0 against the frozen thresholds:

   | metric | threshold | files above it, of 196 |
   | --- | --- | --- |
   | cyclomatic | 10 | 21 |
   | cognitive | 15 | 21 |
   | argument count | 5 | **101** |
   | any of the three | | **108** |

   Argument count alone accounts for 101 of the 108. A keyword-heavy estimator constructor is this
   library's house style.
3. **A rename breaks every Dismissal on the file.** A Report Fingerprint holds the attribution
   file, so `reviewed = raw − matched dismissals` rises to equal `raw` under the new path. The raw
   count survives the move. The reviewed count does not.

## Decision

### The rule is set equality, not a comparison

The gate derives the expected file set itself, from the tracked `.jl` files under the measured
roots. It compares two **sets**, and it never invents a number for a file it has not measured.

```text
expected = tracked .jl files under src/ and ext/
recorded = the keys of the generated file

recorded \ expected  ->  red, a row names no file
expected \ recorded  ->  red, a file has no row
```

Both differences are the same failure, so the deleted file and the added file share one rule. This
generalises ADR 0072's coverage assertion by one step: that rule says an in-scope file must exist,
and this one says it must also be named.

The two implicit-number rules were rejected. An implicit baseline of **zero** splits the two tools,
because every function has cyclomatic complexity of at least 1 while 128 of the 196 files measure
zero under JET, so an added file is always red under one tool and usually green under the other,
and it keeps no row either way. An implicit baseline of the **measured value** is a silent relax on
every added file: a contributor moves a complex definition into a new path and the gate says
nothing, which is the hole ADR 0073's `--accept-rise` rule exists to close.

### A rename pairs by equal measurement

The refresh pairs a dead row with an unnamed file when their measurements are **equal**, and it
carries the row to the new path with no flag. Leftovers on either side are a real deletion or a
real addition, and each faces its own rule.

The pairing is safe by arithmetic rather than by policy. When the numbers are equal, it does not
matter which dead row pairs with which unnamed file. The multiset of recorded numbers is the same
either way, and the ratchet's promise is that no number rises.

One hole follows and it is accepted. A contributor who deletes a file measuring 15 and adds an
unrelated file measuring 15 pairs the two and stays green. To launder a number you must delete the
same number, so the multiset still cannot rise.

Asking git for the renames was rejected. `git diff --find-renames` needs a base commit, which makes
the baseline's `commit` field binding, and ADR 0073 rules that field context only, because the
baseline is always older than the tree. Treating a rename as a deletion plus an addition was
rejected on measurement 1: the renumber commit would record 114 rows the ratchet did not authorise,
and a laundered number would hide among 228 changed lines.

### The JET baseline pairs on the raw count, and the refresh reports the Dismissal repairs

A JET row is `{raw, reviewed}`. Measurement 3 shows the two halves behave differently under a
rename, so the pairing key is the **raw** count alone.

The carried row keeps its **old** reviewed number. The next check therefore measures a higher
reviewed count than the row records, and the gate stays red until the Dismissals name the new path.
Recording the higher number instead needs `--accept-rise`, which already exists. No new rule
creates the red. The subtraction does, exactly as ADR 0071 arranged.

The refresh **prints** the Dismissal lines whose file field is now dead, naming the old path, the
new path and each affected class. It does not edit them.

```text
NOTE: 3 Dismissals still name a dead path.
  src/11_A.jl -> src/12_A.jl
    getfield/Union{} x2
    no matching method x1
Edit code_health/rulings.toml, then re-run.
```

A command that rewrote those paths was rejected. `rulings.toml` is the hand-written file of
ADR 0073's authorship split, and a generated edit landing in the file that holds human paragraphs
is the coupling that split exists to prevent. Printing the repair keeps the split and removes the
search.

### An added file must pass the scheduled job's candidacy test

An unpaired added file enters when [the scheduled job](0073-the-code-health-baseline-is-four-toml-files.md)
would not file it as an offender: cyclomatic at most 10, cognitive at most 15, argument count at
most 5, and JET reviewed equal to 0, with Exemptions dropped first.

```text
ERROR: src/25_New.jl enters at arg = 8, over the threshold of 5.
       Lower it, or add an Exemption naming (path, definition, arg).
```

This invents no flag, no metric split and no threshold. The two ways to satisfy it already exist:
lower the number, or declare an **Exemption**, which is keyed `(path, definition, metric)` and
carries a required reason.

Measurement 2 says the test will fire often, and that is the standard biting rather than the rule
failing. A file the test rejects is a file the scheduled job would file an issue against in the
same week. When the frequency becomes intolerable, the fix is to move the threshold, which is data
in `rulings.toml`.

Two softer rules were rejected. Testing cyclomatic and cognitive alone lets 175 of the 196 files
pass, but it admits a file that the map's own job then reports as an offender, and it needs a
written reason for treating one metric differently. Requiring an explicit act on every added row
carries no information: it would fire on 28 of the last 200 commits and tell a reviewer only that a
file was added, which the diff already shows.

The absolute thresholds this reuses are not the absolute thresholds the map rejected while
charting. That rejection was about **existing** files, which would need the whole codebase brought
under the line before the gate could go green. Nothing here asks an existing file to move.

### Every JET run is total over the whole scope

ADR 0073 keys the JET baseline by run and by file. Set equality needs one expected set per run, so
**each** run holds a row for every file in scope, zeros included. That is about 588 rows over the
three runs.

The case that decides it is a `src/` file that gains its first extension-run report. Under a total
run that is a rise from 0 to 1 and the ratchet catches it. Under a sparse run it is an added row,
indistinguishable from a new file, and the pairing rule hunts for a partner that cannot exist.

### The Expansion Bound's key set obeys the same rule

The Expansion Bound is keyed `(Declaration Macro, metric)` rather than by file, and ADR 0072 rules
that it ratchets exactly as the baseline does. Its expected key set is the macros declared in the
four declaring files, so set equality applies unchanged: a missing key turns the gate red and the
refresh writes it.

**No entry test applies to a new key.** The thresholds measure a definition, and this file records
an addition, which no threshold measures. The macro's own declared complexity is already gated,
because its declaration sits in a `src/` file that the complexity baseline measures. A fresh bound
is green on arrival, which is what ADR 0072 designed.

Holding an addition to the thresholds was rejected on ADR 0072's own numbers: `@forward_properties`
adds up to 31 cognitive and `@define_pretty_show` adds up to 43, so the rule would start red.

## Consequences

**A contributor who meets novel JET noise on a new file stays blocked.** The entry test needs
`reviewed = 0`, a Dismissal needs a Rationale, and ADR 0071 gives a new Rationale to the maintainer.
This decision confirms that consequence rather than softens it, because new code is the worst place
to let a suspected defect land. The block is narrow in practice: about 180 of the 315 reports are
one systematic class, which will carry an approved Rationale, and citing an approved Rationale is
bookkeeping that any contributor may do.

**Moving a file from `research/` into `src/` faces the entry test.** An Unmeasured Path holds no
row, so such a move has no dead row to pair with. That binds the prototypes-to-seams effort: a
prototype is measured on the day it becomes source.

**A deletion needs no flag.** It removes a recorded number, so a refresh that prunes an unpaired
dead row only lowers, and ADR 0073 already lets a refresh lower by default.

**The JET baseline triples to about 588 rows.** That is the price of one expected set per run, and
it is accepted for the same reason ADR 0073 accepted 128 rows of zeros.

**The vocabulary stays out of `CONTEXT.md`**, for the reason ADRs 0071, 0072 and 0073 all gave.
That file's preamble scopes it to the library's domain, and a baseline row set is repository
process.
