# PortfolioOptimisers.jl — Standards Map

This file is a **map, not a rule**. It carries no standards of its own: it tells you which file
governs the thing you are about to change, and which check fails if you get it wrong.

Read the row for your subject, open the file it names, and follow that file. If no row covers what
you are doing, the standard does not exist yet — say so rather than inventing one.

## Vocabulary

Four words are used precisely throughout this file, and by the `/improve-standards` skill that
audits it.

- **Rule** — one normative statement in a standards file.
- **Scope** — the set of files a Rule governs. A Scope is real only if you can run it and count.
- **Gate** — the automated check that fails when a Rule breaks. A Rule with no Gate holds only
  while every contributor remembers it.
- **Authority** — the one file that owns a Rule's text. Every other mention links here instead of
  restating it, because two copies drift.

## Precedence

When two files disagree, the higher entry wins. Report the disagreement rather than silently
picking a side — a contradiction between standards files is itself a defect.

 1. **`docs/adr/`** — a decision that reached `main` outranks every other file on the point it
    settles. An ADR describing superseded behaviour is correct history, not a bug. An ADR whose
    decision has **not** reached `main` does not yet hold this rank: it is a draft, it is rewritten
    in place rather than amended, and it loses to the files below it until the branch merges. This
    is not a corner case on a release branch — measured against `main` at `9adac7735b`, the current
    `dev` carries **25 ADRs that `main` has never seen**, and amends 15 more. All 40 change tier on
    merge.
 2. **`CONTEXT.md`** — the domain glossary. It fixes the words; nothing else may rename a concept.
 3. **`CLAUDE.md`** — the working agreements for this checkout, including the rules an agent must
    not break.
 4. **`.github/instructions/*.instructions.md`** — the per-scope coding standards.
 5. **`.github/copilot-instructions.md`** and **`.github/prompts/*.prompt.md`** — orientation and
    step-by-step workflows.
 6. **`docs/src/contribute/`** — the contributor and developer guides.

## What am I about to touch?

| If you are changing… | Authority | Gate |
| --- | --- | --- |
| Any file in `src/` | [`.github/instructions/julia-source-code.instructions.md`](.github/instructions/julia-source-code.instructions.md) | `pre-commit run -a`, `test/` |
| A docstring | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) | `test/test_26_docs.jl`, the doctest job |
| A return type annotation | [`.github/instructions/julia-return-types.instructions.md`](.github/instructions/julia-return-types.instructions.md) | none — unenforced |
| A test file | [`.github/instructions/julia-test-writing.instructions.md`](.github/instructions/julia-test-writing.instructions.md) | `test/runtests.jl` auto-discovery |
| Adding an estimator | [`.github/prompts/add-estimator.prompt.md`](.github/prompts/add-estimator.prompt.md) | `test/test_26_docs.jl` |
| Adding an algorithm | [`.github/prompts/add-algorithm.prompt.md`](.github/prompts/add-algorithm.prompt.md) | `test/test_26_docs.jl` |
| Adding a result type | [`.github/prompts/add-result.prompt.md`](.github/prompts/add-result.prompt.md) | `test/test_26_docs.jl` |
| A citation or a bibliography entry | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md), `ref_dict` in [`src/01_Base.jl`](src/01_Base.jl) | `test/test_26_docs.jl` |
| A new name for a concept | [`CONTEXT.md`](CONTEXT.md) | none — unenforced |
| An `export` line | [`CLAUDE.md`](CLAUDE.md) § Design rules | `test/test_43_exported_abstract_type_census.jl` |
| A constructor signature | [`.github/instructions/julia-source-code.instructions.md`](.github/instructions/julia-source-code.instructions.md) | `test/test_41_constructor_docstring_drift.jl` |
| An optimiser fallback shortcut | `test/test_40_fallback_shortcut_census.jl` — the census comment is the only written statement of this Rule | the same file |
| JuMP model state | ADR 0037, amending ADR 0004 | `test/test_28_seam_lock.jl` |
| A risk-measure ↔ optimiser pairing | ADR 0018 | `test/test_29_risk_measure_compatibility.jl` |
| A range risk measure | ADR 0057 | `test/test_44_range_tails_census.jl` |
| A combination weight on a meta-optimiser | ADR 0053 | `test/test_42_combination_weight_stacking.jl` |
| A capability the package offers | [`docs/capability_catalogue.jl`](docs/capability_catalogue.jl), ADR 0040 | `test/test_26_docs.jl` |
| A generated docs file | [`CLAUDE.md`](CLAUDE.md) § Editing | CI regenerates and overwrites |
| The paper's code listing | the recipe comment at the top of [`docs/paper/main.typ`](docs/paper/main.typ) | `.github/workflows/Paper.yml` |
| A decision worth recording | [`docs/adr/README.md`](docs/adr/README.md) | none — unenforced |
| A dependency | `Project.toml` | `.github/workflows/Aqua.yml` |
| Running Julia or the test suite | [`CLAUDE.md`](CLAUDE.md) § Running Julia | none — unenforced |
| Running doctests | the `run-doctests` skill | `.github/workflows/Docs.yml` (`doctest` job) |

## The standards files

| File | Owns | Scope |
| --- | --- | --- |
| `STANDARDS.md` | this map | the repository |
| `CONTEXT.md` | the domain glossary — terms and one or two sentences each | the repository |
| `CLAUDE.md` | working agreements for an agent in this checkout | the repository |
| `docs/adr/` | decisions and their reasoning, one file per decision | named per ADR |
| `.github/copilot-instructions.md` | architecture orientation and the before-you-commit checklist | the repository |
| `.github/instructions/julia-source-code.instructions.md` | type roles, constructors, aliases, validation, dispatch, exports | `src/**/*.jl` |
| `.github/instructions/julia-docstrings.instructions.md` | docstring sections, dictionaries, maths, `jldoctest` | `src/**/*.jl`, `docs/**/*.md` |
| `.github/instructions/julia-return-types.instructions.md` | when to annotate a return type | `src/**/*.jl` |
| `.github/instructions/julia-test-writing.instructions.md` | test file layout, `@safetestset`, validation tests | `test/` |
| `.github/prompts/*.prompt.md` | step-by-step workflows for adding a type or shipping a change | task-scoped |
| `docs/src/contribute/` | contributor and developer guides, release checklist | the repository |

`CONTEXT.md` is a glossary and nothing else. Reasoning and forensics belong in `docs/adr/`.

## The gates

Every Gate below is a real check that fails on a real breach.

| Gate | Enforces | How to run |
| --- | --- | --- |
| `pre-commit run -a` | formatting, explicit imports, markdown lint, YAML, TOML, JSON, line endings | `pre-commit run -a` |
| JuliaFormatter | 92-column margin, `yas` style, from `.JuliaFormatter.toml` | inside `pre-commit`, and `.github/workflows/FormatCheck.yml` |
| ExplicitImports | no implicit imports or non-public qualified accesses | inside `pre-commit` |
| markdownlint | markdown structure, from `.markdownlint.json` | inside `pre-commit` |
| `test/test_26_docs.jl` | every public and private name is documented; the Capability Catalogue is complete in both directions; every citation resolves, every `ref_dict` entry has a user, no reference prose is pasted inline, and an API page carries a bibliography block exactly when it cites | run the file |
| `test/test_41_constructor_docstring_drift.jl` | a `# Constructors` block matches the signature it copies | run the file |
| `test/test_43_exported_abstract_type_census.jl` | the exported abstract types are exactly the names on the allow-list in that file | run the file |
| `test/test_40_fallback_shortcut_census.jl` | a fallback shortcut's `Nothing` lands on `fb` | run the file |
| `test/test_44_range_tails_census.jl` | a range risk measure declares its tails, or is on the fused list | run the file |
| `test/test_42_combination_weight_stacking.jl` | a combination weight on a meta-optimiser reaches the model | run the file |
| `test/test_28_seam_lock.jl` | JuMP model state is reached only through its typed interface | run the file |
| `test/test_29_risk_measure_compatibility.jl` | a risk measure is paired only with an optimiser that supports it | run the file |
| `test/test_27_prefix_registration.jl` | a nested risk build namespaces its model-state keys | run the file |
| `.github/workflows/Docs.yml` (`doctest`) | every `jldoctest` block still produces its printed output | see the `run-doctests` skill |
| `.github/workflows/Paper.yml` | the paper's listing still runs against this checkout, and `docs/paper/main-jlyfish.json` is current | the workflow |
| `.github/workflows/Aqua.yml` | package-quality checks over the dependency graph | the workflow |
| `.github/workflows/LinkChecker.yml` | links in the built documentation resolve | the workflow |

A row of the previous table that reads **none — unenforced** names a Rule that no Gate checks. That
is a known state, not a hidden one: an unenforced Rule holds by review and by memory.

## Changing a standard

Amending a Rule is the maintainer's call. It is never a passing edit made while fixing something
else.

 1. Find the **Authority** for the Rule in the tables above. Change the text there, and nowhere
    else.
 2. If another file restates the Rule, replace the copy with a link. Two live copies drift.
 3. Update the row in this file if the Authority, the Scope, or the Gate changed.
 4. If the Rule is load-bearing and has no Gate, consider adding one in the existing census idiom —
   see `test/test_40_fallback_shortcut_census.jl` and `test/test_41_constructor_docstring_drift.jl`.
   Each opens with a comment saying what drifted, when, and why the check earns its runtime.
 5. If the change reverses a previous decision, record it in `docs/adr/` and add its row to
    `docs/adr/README.md`.

The `/improve-standards` skill audits this map: it checks the code against each Rule, and each Rule
against the code.

## Before you finish a change

From `.github/copilot-instructions.md`, which is the Authority for this list:

 1. `pre-commit run -a` passes.
 2. The tests for the area you changed pass.
 3. The doctests pass.
 4. New capabilities are in `docs/capability_catalogue.jl`.
 5. Docstrings and documentation reflect the change.
