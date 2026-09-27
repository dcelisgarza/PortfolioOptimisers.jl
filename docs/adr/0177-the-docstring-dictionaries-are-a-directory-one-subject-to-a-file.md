---
status: accepted
---

# The docstring dictionaries are a directory, one subject to a file

## Context

[ADR 0100](0100-split-the-base-file-into-one-file-per-concept.md) gave the seven docstring
dictionaries one file, `src/01_Base/01_DocstringDictionaries.jl`. By 2026-09-24 the file held
1325 entries in 1650 lines, and 1341 of those lines were code: an entry is a string literal in a
call, so the size gate of [ADR 0101](0101-the-size-gate-counts-code-lines-and-binds-over-a-threshold.md)
counts it as code. The file was over the threshold of 500, so the gate held it at its recorded
number, and every session that added a key also had to raise the row in
`code_health/size_baseline.toml`. Sixty commits changed the file in the fourteen days before the
split.

`CLAUDE.md` names the file as one of three that every sweep ticket shares. Parallel sessions
added keys at the same places, mostly at the end of `arg_dict` and `math_dict`, so their edits
collided on rebase.

`math_dict` was a plain `Dict` call. The guard `unique_key_dict` built the other five tables, so
a key written twice in `math_dict` dropped the first description with no error.

The maintainer considered three layouts: one file per letter of the key, a number suffix per
part, and one file per subject in a directory. A letter split puts related keys in different
files and makes files of very different sizes: 89 keys of `arg_dict` start with `r` and one
starts with `j`. A number suffix does not tell a contributor which part a new key goes to. The
maintainer chose the directory.

## Decision

**`src/01_Base/01_DocstringDictionaries/` holds the tables, one subject to a file.**

| File | What it holds |
| ---- | ------------- |
| `01_Tables.jl` | `unique_key_dict!`, and the declaration and docstring of each of the seven tables, empty |
| `02_ArgDataAndMoments.jl` | `arg_dict`: data, moments, the matrices estimated from them, the regime-adjusted and exponentially weighted estimators, partial fit states |
| `03_ArgPriorsAndUncertaintySets.jl` | `arg_dict`: priors and their results, entropy pooling, Black-Litterman, opinion pooling, higher order priors, uncertainty sets |
| `04_ArgPhylogenyAndConstraints.jl` | `arg_dict`: phylogeny, clustering, universe sets, constraints and their generation, weight bounds, preselection |
| `05_ArgRiskMeasuresAndCosts.jl` | `arg_dict`: risk measures and their settings, return terms, tracking, turnover, fees, trading costs |
| `06_ArgOptimisation.jl` | `arg_dict`: optimisers, solvers, JuMP models, their results, cross-validation, prediction, allocation, the online step |
| `07_FieldDict.jl` | `field_dict`, derived from the whole of `arg_dict` |
| `08_ValidationsAndReturns.jl` | `err_name_dict`, `val_dict` and `ret_dict` |
| `09_MathDataAndMoments.jl` | `math_dict`: data, returns, moments, distances, feature matrices, denoising, norm errors, the Gerber family, calibration rules |
| `10_MathPriorsAndNetworks.jl` | `math_dict`: entropy pooling, phylogeny, centrality, clusters, preselection |
| `11_MathRiskAndOptimisation.jl` | `math_dict`: risk measures, their JuMP formulations, penalties, weight finalisation, meta-optimisers |
| `12_MathOnlineSelection.jl` | `math_dict`: online portfolio selection |
| `13_References.jl` | `ref_dict` |

A key goes to the file of its subject. The section comments of the old file gave the first
assignment. A section that held more than one subject, such as `# Other.` and `# Stats.`, was
split by line range, and each part took a comment that names its subject. The split moved every
entry unchanged: the multiset of `(table, key, description)` over the thirteen files equals the
one over the old file, 1325 entries, none lost and none added.

**Each table is filled in parts.** `01_Tables.jl` binds each table to an empty
`Dict{Symbol, String}`. Every later file makes one call `unique_key_dict!(table, :table, pairs...)`
per table it fills, and `07_FieldDict.jl` loads after the last `arg_dict` file, so it reads the
whole table.

**`unique_key_dict!` replaces `unique_key_dict`, and it guards every table, `math_dict`
included.** It throws when a key already holds a different description, from an earlier call or
from an earlier pair of the same call. A key that returns with the description it already holds
changes nothing. That is how a second evaluation of one file reads: Revise evaluates an edited
file again, and every key the file adds is then already in the table. A changed description of an
existing key still throws under Revise, and the fix is a restart. An identical copy of an entry
is the one repeat the guard lets through, and it loses no description.

**The API pages mirror the directory.** Each file has a page on each side, as
[ADR 0150](0150-a-family-of-files-is-a-directory-and-the-api-tree-is-one-page-per-file-at-the-same-path.md)
states. The pages of `01_Tables.jl` take the entries of the two old pages. A file that only
fills a table defines no name, so both of its pages say so, and the private page names the table
it fills.

## Consequences

- Each file is under 250 code lines, so the size gate binds on none of them, and a new key
  needs no edit of `code_health/size_baseline.toml`.
- Two sessions collide only when they add keys of one subject to one table.
- A repeated key in `math_dict` now fails at load time, as it does in the other tables.
- `field_dict` holds `String` values, where it held `SubString{String}` values.
- `test/test_26_docs.jl` reads `math_dict` from every `unique_key_dict!(math_dict, …)` call in
  the directory, and it excludes `13_References.jl` when it looks for the users of `ref_dict`.
