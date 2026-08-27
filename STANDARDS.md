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
    in place rather than amended, and it loses to the files below it until the branch merges. A
    draft ADR changes tier on the day its branch merges. Check where an ADR stands before you lean
    on it.
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
| A docstring in `src/` or `ext/` | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) | `test/test_26_docs.jl`, the doctest job |
| The `# Algorithm` and `# JuMP formulation` sections of a SWEPT file | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md), and [`sweep/manifest.toml`](sweep/manifest.toml) for the `swept` flag that arms the demand | `test/test_26_docs.jl` |
| An alias docstring — acronym, factory, or dispatch | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *Section Structure for Aliases* | `test/test_26_docs.jl` — the sections a kind allows, `# Related` in a SWEPT file, and a library-wide ratchet; `test/test_47_alias_and_module_census.jl` — an acronym alias IS its target and its sentence names it, and a factory alias's sentence names every type it composes |
| A `# Details` section | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *`# Details` is abolished* | `test/test_26_docs.jl` — zero in a SWEPT file, and a library-wide count that may not rise |
| The `# Mathematical definition` section | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *What the section may not state* | none — unenforced |
| A mathematical symbol that two or more docstrings share | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *Notation is fixed by symbol and by family*, `math_dict` in [`src/01_Base.jl`](src/01_Base.jl) | `test/test_26_docs.jl` — a swept file copies no `math_dict` value, and a library-wide count that may not rise |
| The notation that siblings of one Family share | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *Notation is fixed by symbol and by family* | none — unenforced |
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
| Adding a file, a type or a function under `src/` or `ext/` | [`CLAUDE.md`](CLAUDE.md) § Functionality you add, and [`sweep/manifest.toml`](sweep/manifest.toml) for the row it names | `test/test_45_sweep_census.jl`, and `.github/workflows/Sweep.yml` when the child map has already closed; `test/test_47_alias_and_module_census.jl` for a new file, which `src/PortfolioOptimisers.jl` must `include` |
| A combination weight on a meta-optimiser | ADR 0053 | `test/test_42_combination_weight_stacking.jl` |
| A capability the package offers | [`docs/capability_catalogue.jl`](docs/capability_catalogue.jl), ADR 0040 | `test/test_26_docs.jl` |
| A standards file, or a name or a path one cites | [`STANDARDS.md`](STANDARDS.md) § *Changing a standard* | `test/test_46_standards_citation_census.jl` |
| A generated docs file | [`CLAUDE.md`](CLAUDE.md) § Editing | CI regenerates and overwrites |
| The paper's code listing | the recipe comment at the top of [`docs/paper/main.typ`](docs/paper/main.typ) | `.github/workflows/Paper.yml` |
| A decision worth recording | [`docs/adr/README.md`](docs/adr/README.md) | none — unenforced |
| A dependency | `Project.toml` | `.github/workflows/Aqua.yml` |
| Running Julia or the test suite | [`CLAUDE.md`](CLAUDE.md) § Running Julia | none — unenforced |
| Running doctests | the `run-doctests` skill | `.github/workflows/Docs.yml` (`doctest` job) |
| The line coverage of a file in `src/` or `ext/` | [ADR 0082](docs/adr/0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md) | `.github/workflows/ReusableTest.yml` (`coverage` job) |
| A Coverage Exemption | [ADR 0082](docs/adr/0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md), `code_health/rulings.toml` | `.github/workflows/ReusableTest.yml` (`coverage` job) for the count it stands for; `test/test_49_coverage_attribution_census.jl` for the definition it names |

## The standards files

| File | Owns | Scope |
| --- | --- | --- |
| `STANDARDS.md` | this map | the repository |
| `CONTEXT.md` | the domain glossary — terms and one or two sentences each | the repository |
| `CLAUDE.md` | working agreements for an agent in this checkout | the repository |
| `docs/adr/` | decisions and their reasoning, one file per decision | named per ADR |
| `.github/copilot-instructions.md` | architecture orientation and the before-you-commit checklist | the repository |
| `.github/instructions/julia-source-code.instructions.md` | type roles, constructors, aliases, validation, dispatch, exports | `src/**/*.jl` |
| `.github/instructions/julia-docstrings.instructions.md` | docstring sections, dictionaries, maths, the algorithm and JuMP formulation blocks, the sections an alias carries, `jldoctest`, and the pointers to the reference docstrings | `src/**/*.jl`, `ext/**/*.jl`, `docs/**/*.md` |
| `.github/instructions/julia-return-types.instructions.md` | when to annotate a return type | `src/**/*.jl` |
| `.github/instructions/julia-test-writing.instructions.md` | test file layout, `@safetestset`, validation tests | `test/` |
| `.github/prompts/*.prompt.md` | the order of the work for adding a type or shipping a change; every step links to the Authority that owns its rule, and a prompt states no rule of its own | task-scoped |
| `docs/src/contribute/` | contributor and developer guides, release checklist | the repository |

`CONTEXT.md` is a glossary and nothing else. Reasoning and forensics belong in `docs/adr/`.

## The gates

Every Gate below is a real check that fails on a real breach.

| Gate | Enforces | How to run |
| --- | --- | --- |
| `pre-commit run -a` | formatting, explicit imports, markdown lint, YAML, TOML, JSON, line endings | `pre-commit run -a` |
| JuliaFormatter | 92-column margin, `yas` style, from `.JuliaFormatter.toml` | inside `pre-commit`, and `.github/workflows/FormatCheck.yml` |
| ExplicitImports | no implicit imports or non-public qualified accesses | inside `pre-commit` |
| markdownlint | markdown structure, from `.markdownlint.json`, over the files `.markdownlintignore` leaves in scope | inside `pre-commit` |
| `test/test_26_docs.jl` | every public and private name is documented; the Capability Catalogue is complete in both directions; every citation in `src/` and `ext/` resolves, every `ref_dict` entry has a user, no reference prose is pasted inline, and an API page carries a bibliography block exactly when it cites; a name an extension declares itself carries a docstring once that extension's file is marked `swept` in [`sweep/manifest.toml`](sweep/manifest.toml); a swept file's docstring that documents a function building part of a `JuMP` model carries `# JuMP formulation` and each subsection the body's macros demand, a swept file's count of `# Algorithm` sections does not fall below its manifest row, a swept file carries no `# Details` section, and the library-wide count of `# Details` does not rise; an alias docstring carries no section outside the set its kind allows, a dispatch alias in a swept file carries `# Related`, and the library-wide count of dispatch aliases carrying none does not rise; a `Where:` bullet interpolates a `math_dict` value rather than copying it, with the same pair of a swept-file zero and a library-wide count that may not rise | run the file |
| `test/test_41_constructor_docstring_drift.jl` | a `# Constructors` block matches the signature it copies | run the file |
| `test/test_43_exported_abstract_type_census.jl` | the exported abstract types are exactly the names on the allow-list in that file | run the file |
| `test/test_40_fallback_shortcut_census.jl` | a fallback shortcut's `Nothing` lands on `fb` | run the file |
| `test/test_44_range_tails_census.jl` | a range risk measure declares its tails, or is on the fused list | run the file |
| `test/test_45_sweep_census.jl` | every file under `src/` and `ext/` has a sweep-manifest row naming its child map of #404, and the file's documented-unit count still matches that row | run the file |
| `test/test_47_alias_and_module_census.jl` | an acronym alias of `src/25_Aliases.jl` IS the binding its docstring names, a factory alias of that file EQUALS the long form its sentence names, and `src/PortfolioOptimisers.jl` `include`s every other file under `src/` exactly once | run the file |
| `test/test_46_standards_citation_census.jl` | every name and every path a standards file cites resolves against the repository, and no standards file states a count of the repository | run the file |
| `test/test_49_coverage_attribution_census.jl` | `code_health/coverage.jl` names a return-annotated definition by its function and a functor method by its receiver type, and every Coverage Exemption in `code_health/rulings.toml` names a definition its file holds | run the file |
| `test/test_42_combination_weight_stacking.jl` | a combination weight on a meta-optimiser reaches the model | run the file |
| `test/test_28_seam_lock.jl` | JuMP model state is reached only through its typed interface | run the file |
| `test/test_29_risk_measure_compatibility.jl` | a risk measure is paired only with an optimiser that supports it | run the file |
| `test/test_27_prefix_registration.jl` | a nested risk build namespaces its model-state keys | run the file |
| `.github/workflows/Docs.yml` (`doctest`) | every `jldoctest` block still produces its printed output | see the `run-doctests` skill |
| `.github/workflows/Paper.yml` | the paper's listing still runs against this checkout, and `docs/paper/main-jlyfish.json` is current | the workflow |
| `.github/workflows/ReusableTest.yml` (`coverage`) | a file's miss count has not risen above `code_health/coverage_baseline.toml`, an added file enters with every line covered or exempted, and every Coverage Exemption states the exact count it stands for | `julia --project=code_health code_health/coverage.jl check`, with `COVERAGE_LCOV` pointing at an `lcov.info` |
| `.github/workflows/Sweep.yml` | a file whose sweep-manifest row reads `swept = false` under a CLOSED child map of #404 reopens that map, reopens #404, and gets one `sweep` sub-issue | `julia --project=code_health code_health/sweep_triage.jl --maps <tsv>` for the plan; the workflow opens it |
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
