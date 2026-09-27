---
status: accepted
---

# A file over the size threshold that reads in order is cut into letter parts

## Context

[ADR 0101](0101-the-size-gate-counts-code-lines-and-binds-over-a-threshold.md) holds a file over
500 code lines at its recorded number, and names a split as the remedy, left to a person's
judgement. On 2026-09-24 fifteen files stood over the threshold. The first,
`src/01_Base/01_DocstringDictionaries.jl`, became a directory of files by subject
([ADR 0177](0177-the-docstring-dictionaries-are-a-directory-one-subject-to-a-file.md)).

The maintainer asked for the same logic on the other large files, with one condition: a file that
is already logical in its construction is cut as near the line limit as its definitions allow,
keeps its order, and names its parts with the letter suffixes `_a`, `_b`, and so on.

A survey of the other fourteen files found two kinds.

- **Eight hold one subject in a logical order**: an estimator family, an optimiser, or a family of
  views, from its types to its methods. A cut keeps the order, and no definition moves relative to
  another.
- **Six mix subjects**: `src/02_Tools.jl` is a grab bag, the plot methods of the Plots extension
  are spread over one verb per subject, and four base files interleave two or more subjects. These
  take a directory of files by subject, one commit each, like ADR 0177.

The rule of [ADR 0147](0147-a-numeric-prefix-is-unique-in-its-directory-and-the-include-list-is-the-prefix-order.md) that a number is unique in its
directory refuses two parts under one number, so the rule needed a case for parts.

## Decision

**A file over the threshold that holds one subject in order is cut into letter parts.**
`NN_Name.jl` becomes `NN_Name_a.jl`, `NN_Name_b.jl`, and so on, in the same directory. The parts
keep the number of the file and load in letter order, so the include list and the load order do
not change.

**The cut is mechanical.**

- A cut falls only before a top-level definition that carries a docstring, so a documented unit
  and the undocumented methods after it stay together.
- Each part takes as many definitions as fit under the threshold, counted by the classifier of the
  size gate. The lines between two definitions go with the part after them.
- The parts joined together are the old file, byte for byte, before the export lines move.
- Each part ends with the `export` and `public` lines of the names it defines, as
  `.github/instructions/julia-source-code.instructions.md` § *Code Organization* states for every
  source file. A name that no part defines, such as a method of a function another package owns,
  stays with the last part.

The first cut of `02_LinearConstraintGeneration.jl` filled its first part to exactly 500 code
lines, and its own `export` line then took it to 502. So a part leaves room for its export line:
that file was cut again at 495.

**The parts are one entry of their directory.** `test/test_47_alias_and_module_census.jl` reads a
name without its letter suffix when it checks that a number is unique, and it refuses a run that
does not start at `_a`, that leaves a letter out, or that stands beside an unsuffixed file of the
same name.

**Everything keyed on the old path moves to the part that holds the thing it names.**

- The API pages split one pair per part. An entry goes to the page of the part that declares its
  docstring, or that defines its method when the docstring it renders is a wider one declared
  elsewhere. An entry that no part declares or defines stays beside the entry before it.
- The manifest row splits by `documented_units` and `documented_bindings` per part. The units sum
  to the old row. The `algorithm` key of a part is its measured count, so the sum can stand above
  an old floor that was lower than the file.
- The size, complexity and perf rows are measured per part. Each definition over a threshold takes
  an Exemption citing `a-definition-a-file-split-moved-unchanged`, because it stood inside the old
  file's row and now drives its part's maximum.
- The JET rows are measured per part and written by hand, because `jet.jl refresh` refuses an added
  file that carries a reviewed report.
- The coverage rows split by the line of each miss, read from CI's `lcov.info`.
- An Exemption, a link target in an ADR, and a path in a test or a comment name the part. An ADR
  keeps its link text, because the text is history.

The eight files and their parts:

| File | Parts |
| ---- | ----- |
| `src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior.jl` | `_a`, `_b`, `_c` |
| `src/09_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `_a`, `_b` |
| `src/17_Optimisation/05_JuMP/03_JuMPOptimiser.jl` | `_a`, `_b` |
| `src/17_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl` | `_a`, `_b` |
| `src/05_Moments/30_RegimeAdjusted/02_RegimeAdjustedExpWeightedCovariance.jl` | `_a`, `_b` |
| `src/17_Optimisation/05_JuMP/06_NearOptimalCentering.jl` | `_a`, `_b` |
| `src/16_RiskMeasures/07_OWARiskMeasures.jl` | `_a`, `_b` |
| `src/05_Moments/05_Gerber/03_GerberIQCovariance.jl` | `_a`, `_b` |

## Consequences

- No file of `src/` stands over the threshold except the six that take a directory by subject.
- A first part filled near the threshold has little room to grow, so an addition to it can cross
  the threshold and trip the gate. The fix is the same cut: the definitions that no longer fit move
  to the start of the next part.
- A new definition joins the part where its subject already stands, as a new line of a file always
  did.
- Three entries on the NearOptimalCentering page render docstrings declared in
  `01_Base_Optimisation.jl` and `01_Base_JuMPOptimisation.jl`. The split keeps them where they
  stood; ADR 0150 would place them on the pages of those files.
