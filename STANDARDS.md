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
| A file's numeric prefix, the directory a family of files takes, the API page a file owns, or the `include` list of `src/PortfolioOptimisers.jl` | [`.github/instructions/julia-source-code.instructions.md`](.github/instructions/julia-source-code.instructions.md) § *Code Organization* | `test/test_47_alias_and_module_census.jl` |
| A numeric working type, an output element type, or a type coercion in `src/` | [`.github/instructions/julia-source-code.instructions.md`](.github/instructions/julia-source-code.instructions.md) § *Numeric types come from the data* | `test/test_55_numeric_coercion_census.jl` |
| A one-method unit whose body forwards to one expression, or an abstract type with one concrete subtype that nothing codes against | [`.github/instructions/julia-source-code.instructions.md`](.github/instructions/julia-source-code.instructions.md) § *A unit that offers nothing but indirection is inlined* | `test/test_70_trivial_unit_census.jl` |
| A `DomainError` raise, or a numeric-range guard in `src/` or `ext/` | [`.github/instructions/julia-source-code.instructions.md`](.github/instructions/julia-source-code.instructions.md) § *A `DomainError` carries the value and a message* | `test/test_63_domain_error_shape_census.jl` |
| A docstring in `src/` or `ext/` | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) | `test/test_26_docs.jl`, the doctest job |
| An `# Interfaces` section on an abstract type | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *What each section holds* | `test/test_61_interfaces_section_census.jl` — every verb heading and call-shaped bullet names a function and types the package defines, and every abstract type that parents a concrete type carries the section or is on the debt list in that file |
| The `# Algorithm` and `# JuMP formulation` sections of a SWEPT file | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md), and [`code_health/sweep_manifest.toml`](code_health/sweep_manifest.toml) for the `swept` flag that arms the demand | `test/test_26_docs.jl` |
| An alias docstring — acronym, factory, or dispatch | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *Section Structure for Aliases* | `test/test_26_docs.jl` — the sections a kind allows, `# Related` in a SWEPT file, and a library-wide ratchet; `test/test_47_alias_and_module_census.jl` — an acronym alias IS its target and its sentence names it, and a factory alias's sentence names every type it composes |
| A `# Details` section | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *`# Details` is abolished* | `test/test_26_docs.jl` — zero in a SWEPT file, and a library-wide count that may not rise |
| A reference to a GitHub issue, a pull request, an ADR, or an unpublished numerical experiment inside a docstring, an error message, the Capability Catalogue's `Prose` text, or the Literate sources of an example or a user-guide page, including inside an admonition | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *A docstring cites no process* | `test/test_71_process_citation_census.jl` — the docstrings of `src/` and `ext/`, the hand-written pages of `docs/src/` outside `docs/src/contribute/`, the catalogue's non-comment lines and the markdown lines of every Literate source name no `ADR nnnn`, `#nnn`, pull request or tracker URL, and neither does the error text of `src/` and `ext/`, which the file reads with `Meta.parseall`: a literal inside a `throw`, an `error`, an `…Error`/`…Exception` constructor, an `@argcheck` or an `@assert`, or a `const` named `*_remedy` or `*_message` |
| The prose of a docstring in `src/` or `ext/`, or of a dictionary value it interpolates | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *The prose passes `/unslop`* | none — unenforced; a file whose sweep-manifest row reads `swept = true` owes the pass, and its sweep ticket under #404 carries it |
| The prose of a Literate source under `examples/` or `user_guide/`: the `unslop` rules, the dash, the voice, the vocabulary, a narrated check, the paragraph, and the page's furniture | [`.github/instructions/julia-prose.instructions.md`](.github/instructions/julia-prose.instructions.md) | `test/test_72_prose_census.jl` — every counted rule of a text is at or below its row in [`code_health/prose_baseline.toml`](code_health/prose_baseline.toml), a text whose every count is zero carries no row, and a row that names no text fails; the reader is [`code_health/prose.jl`](code_health/prose.jl), which a rewrite session runs on one text with `scan`; rules 10, 11, 27, 28 and 32 and the self-audit hold by review |
| The prose of a hand-written page under `docs/src/` outside `docs/src/contribute/`, of `README.md`, or of `docs/capability_catalogue.jl`: the same rules, less `we` for the page's cells, the paragraph that says what the next cell does, and the page's furniture | [`.github/instructions/julia-prose.instructions.md`](.github/instructions/julia-prose.instructions.md) § *The page speaks to the reader* | `test/test_72_prose_census.jl` — the same per-text rows; a Markdown page is read as its own lines, and the catalogue as the body of every double-quoted string literal on a line that is neither a `#` comment nor part of a triple-quoted block |
| A mirror page's H1 under `docs/src/public_api/` or `docs/src/private_api/`, and its `Description` line | [`.github/instructions/julia-prose.instructions.md`](.github/instructions/julia-prose.instructions.md) § *What the rule does not read* | `test/test_64_docs_page_metadata_census.jl` — the H1 ends in `: public API` or `: private API` (ADR 0128) and the `Description` is the line `docs/page_metadata.jl` derives; the prose rule does not read either, so one line carries one gate |
| A glossary term of `CONTEXT.md`, a contributor's word for a mechanism, or the section sign `§`, inside the prose of a text a user reads as a page | [`.github/instructions/julia-prose.instructions.md`](.github/instructions/julia-prose.instructions.md) § *A page names a type by its identifier and a concept in plain words* | `test/test_72_prose_census.jl` — the `glossary` and `mechanism` counts of the text's row; the term list is read off [`CONTEXT.md`](CONTEXT.md) at every run, so it never goes stale |
| A string that a cell of a page prints (a table or plot title, a label, a column, a `println` sentence), or a comment in a code cell of a page | [`.github/instructions/julia-prose.instructions.md`](.github/instructions/julia-prose.instructions.md) § *The code of a page* | `test/test_72_prose_census.jl` — every plain string literal and every `#!` gotcha in the code of a text is read as prose, a `title =` string also as a heading, and the `comment` count of the text's row holds every other code comment at zero; whether a title claims what its cell does not show, and whether a gotcha is one, hold by review |
| A page that states the outcome of a comparison as proved instead of printing its number | [`.github/instructions/julia-prose.instructions.md`](.github/instructions/julia-prose.instructions.md) § *A check is a number the reader reads, never a verdict* | `test/test_72_prose_census.jl` — the `verdict` count of the text's row; `the identity matrix` is the matrix and is not counted |
| The `# Mathematical definition` section | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *What the section may not state* | none — unenforced |
| A mathematical symbol that two or more docstrings share | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *Notation is fixed by symbol and by family*, `math_dict` in the `*_Math*.jl` files of [`src/01_Base/01_DocstringDictionaries/`](src/01_Base/01_DocstringDictionaries/) | `test/test_26_docs.jl` — a swept file copies no `math_dict` value, a library-wide count that may not rise, and two keys of the table do not state one quantity |
| The notation that siblings of one Family share | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md) § *Notation is fixed by symbol and by family* | none — unenforced |
| A return type annotation | [`.github/instructions/julia-return-types.instructions.md`](.github/instructions/julia-return-types.instructions.md) | none — unenforced |
| A partial-fit state, or an estimator's `cache` field | [`docs/adr/0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md`](docs/adr/0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md) and [`docs/adr/0107-the-update-seam-has-two-verbs-and-a-view-slices-by-asset-and-drops-by-observation.md`](docs/adr/0107-the-update-seam-has-two-verbs-and-a-view-slices-by-asset-and-drops-by-observation.md) | `test/test_62_partial_fit_state_interface_census.jl` — every state owns `merge_states` and `copy` below the root, a view of every estimator that carries a `cache` slices or refuses it, and every state a view can reach owns `port_opt_view` or is paired in that file with a host no view reaches; `test/test_60_partial_fit_cache_narrowing_census.jl` — every estimator `partial_fit!` method narrows `cache` to its own state |
| A test file | [`.github/instructions/julia-test-writing.instructions.md`](.github/instructions/julia-test-writing.instructions.md) | `test/runtests.jl` auto-discovery |
| Adding an estimator | [`.github/prompts/add-estimator.prompt.md`](.github/prompts/add-estimator.prompt.md) | `test/test_26_docs.jl` |
| Adding an algorithm | [`.github/prompts/add-algorithm.prompt.md`](.github/prompts/add-algorithm.prompt.md) | `test/test_26_docs.jl` |
| Adding a result type | [`.github/prompts/add-result.prompt.md`](.github/prompts/add-result.prompt.md) | `test/test_26_docs.jl` |
| A citation or a bibliography entry | [`.github/instructions/julia-docstrings.instructions.md`](.github/instructions/julia-docstrings.instructions.md), `ref_dict` in [`src/01_Base/01_DocstringDictionaries/13_References.jl`](src/01_Base/01_DocstringDictionaries/13_References.jl) | `test/test_26_docs.jl` |
| A new name for a concept | [`CONTEXT.md`](CONTEXT.md) | none — unenforced |
| An `export` line | [`CLAUDE.md`](CLAUDE.md) § Design rules | `test/test_43_exported_abstract_type_census.jl` |
| A constructor signature | [`.github/instructions/julia-source-code.instructions.md`](.github/instructions/julia-source-code.instructions.md) | `test/test_41_constructor_docstring_drift.jl` |
| An optimiser fallback shortcut | `test/test_40_fallback_shortcut_census.jl` — the census comment is the only written statement of this Rule | the same file |
| JuMP model state | ADR 0037, amending ADR 0004 | `test/test_28_seam_lock.jl` |
| A risk-measure ↔ optimiser pairing | ADR 0018 | `test/test_29_risk_measure_compatibility.jl` |
| A range risk measure | ADR 0057 | `test/test_44_range_tails_census.jl` |
| A moment estimator that joins a Choice Surface | [ADR 0099](docs/adr/0099-choice-surface-membership-means-the-verbs-exist.md) — the leaf answers its family's verbs, and `test/moment_family_setup.jl` holds the split and the ownership predicate | `test/test_08l_moment_verb_census.jl`, and `test/test_08d_dims_guard.jl` for the `dims` guard on each pairing it admits |
| Adding a file, a type or a function under `src/` or `ext/` | [`CLAUDE.md`](CLAUDE.md) § Functionality you add, and [`code_health/sweep_manifest.toml`](code_health/sweep_manifest.toml) for the row it names | `test/test_45_sweep_census.jl`, and `.github/workflows/Sweep.yml` when the child map has already closed; `test/test_47_alias_and_module_census.jl` for a new file, which `src/PortfolioOptimisers.jl` must `include`. `julia --project=code_health code_health/sweep_check.jl --fetch` reports all of it before the commit |
| Filing the sweep sub-issue of an addition, by hand | [ADR 0084](docs/adr/0084-the-sweep-job-reopens-a-closed-child-map-and-files-the-addition.md), and the `sweep-file-issues` skill | `julia --project=code_health code_health/sweep_triage.jl --fetch --file <path>` for the plan, then `code_health/sweep_issues.sh apply` |
| A combination weight on a meta-optimiser | ADR 0053 | `test/test_42_combination_weight_stacking.jl` |
| A capability the package offers | [`docs/capability_catalogue.jl`](docs/capability_catalogue.jl), ADR 0040 | `test/test_26_docs.jl` |
| A standards file, or a name or a path one cites | [`STANDARDS.md`](STANDARDS.md) § *Changing a standard* | `test/test_46_standards_citation_census.jl` |
| A generated docs file | [`CLAUDE.md`](CLAUDE.md) § Editing, ADR 0151 | untracked; the Docs build writes it and checks its links |
| A link from a hand-written docs page to another page | ADR 0151 § *Amendment (2026-09-26)* | `test/test_64_docs_page_metadata_census.jl` — no relative `.md` link; the target H1 carries an `(@id label)` and the link reads `(@ref label)`, which the Docs build resolves |
| An `@docs` entry's placement in the mirrored public/private API trees ADR 0128 decides, or the classification a name earns | [ADR 0128](docs/adr/0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md), and [`docs/api_classification.jl`](docs/api_classification.jl) for the classifier | `test/test_65_docs_public_private_placement_census.jl` |
| A docs page's `<title>` or its description — the `Description` line of a hand-written page's `@meta` block, the derived line of a mirror page, the landing page's label in `docs/make.jl`, or the site-wide fallback | [ADR 0128](docs/adr/0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md) § *Amendment (2026-09-17)*, and [`docs/page_metadata.jl`](docs/page_metadata.jl) for the derivation | `test/test_64_docs_page_metadata_census.jl` |
| A private name's promotion to `public` | [ADR 0154](docs/adr/0154-a-private-abstract-type-earns-a-public-declaration-when-its-interfaces-section-names-the-seam.md) — an existing `# Interfaces` docstring section is the only route | `test/test_66_public_declaration_census.jl` |
| The paper's code listing | the recipe comment at the top of [`docs/paper/main.typ`](docs/paper/main.typ) | `.github/workflows/Paper.yml` |
| A decision worth recording | [`docs/adr/README.md`](docs/adr/README.md) | none — unenforced |
| A dependency | `Project.toml` | `.github/workflows/Aqua.yml` |
| Running Julia or the test suite | [`CLAUDE.md`](CLAUDE.md) § Running Julia | none — unenforced |
| Running doctests | the `run-doctests` skill | `.github/workflows/Docs.yml` (`doctest` job) |
| The line coverage of a file in `src/` or `ext/` | [ADR 0082](docs/adr/0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md) | `.github/workflows/ReusableTest.yml` (`coverage` job) |
| The size of a file in `src/` or `ext/` | [ADR 0101](docs/adr/0101-the-size-gate-counts-code-lines-and-binds-over-a-threshold.md) — code lines bind, a docstring line does not, and the ceiling is the greater of 500 and the recorded number | `.github/workflows/Complexity.yml` (`Size ratchet` step), or `julia --project=code_health code_health/size.jl check`; `test/test_52_size_classification_census.jl` for the classification itself |
| A performance trap in `src/` or `ext/`: a slice that copies where a view reads, a reduction or a search over a temporary, a broadcast that does not fuse, a linear-algebra chain that builds a matrix to read a scalar, one expensive call made twice, and an array a loop allocates on every iteration | [ADR 0175](docs/adr/0175-the-performance-gate-reads-traps-from-the-source-text-and-ratchets-the-count-per-file-and-per-rule.md), and a `[[perf_dismissal]]` in `code_health/rulings.toml` for a Finding that is not a trap, with no `code` to cover a rule over one definition; a swept file carries none, the fifth condition of #404 | `.github/workflows/Complexity.yml` (`Performance ratchet` step), or `julia --project=code_health code_health/perf.jl check`; `julia --project=code_health code_health/perf.jl scan <file>` lists the Findings with their replacements; `test/test_73_performance_trap_census.jl` for the rules themselves |
| A Coverage Exemption | [ADR 0082](docs/adr/0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md), `code_health/rulings.toml` | `.github/workflows/ReusableTest.yml` (`coverage` job) for the count it stands for; `test/test_49_coverage_attribution_census.jl` for the definition it names |
| A `COV_EXCL` marker in `src/` or `ext/` | [ADR 0082](docs/adr/0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md) — it is not admitted, and a Coverage Exemption is the one mechanism | `test/test_49_coverage_attribution_census.jl` |
| `test/test_52_size_classification_census.jl` | `code_health/size.jl` reads a docstring and a field docstring as prose and a value string as code, the four kinds partition every file in scope, and a file's code-line ceiling is the greater of the threshold and its recorded number | run the file |

## The standards files

| File | Owns | Scope |
| --- | --- | --- |
| `STANDARDS.md` | this map | the repository |
| `CONTEXT.md` | the domain glossary — terms and one or two sentences each | the repository |
| `CLAUDE.md` | working agreements for an agent in this checkout | the repository |
| `docs/adr/` | decisions and their reasoning, one file per decision | named per ADR |
| `.github/copilot-instructions.md` | architecture orientation and the before-you-commit checklist | the repository |
| `.github/instructions/julia-source-code.instructions.md` | type roles, constructors, aliases, validation, dispatch, exports | `src/**/*.jl` |
| `.github/instructions/julia-docstrings.instructions.md` | docstring sections, dictionaries, maths, the algorithm and JuMP formulation blocks, the sections an alias carries, `jldoctest`, the `/unslop` pass over a docstring and over a dictionary value, and the pointers to the reference docstrings | `src/**/*.jl`, `ext/**/*.jl`, `docs/**/*.md` |
| `.github/instructions/julia-prose.instructions.md` | the prose a user reads as a page: the `/unslop` pass, the dash, the voice, the vocabulary, a narrated check, the paragraph, the page's furniture, the printed strings and the code comments of a page, the derived text the rule does not read, and the census | `examples/**/*.jl`, `user_guide/*.jl`, `docs/src/**/*.md`, `README.md`, `docs/capability_catalogue.jl` |
| `.github/instructions/julia-return-types.instructions.md` | when to annotate a return type | `src/**/*.jl` |
| `.github/instructions/julia-test-writing.instructions.md` | test file layout, the per-file module and its `init_code` preamble, validation tests | `test/` |
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
| `test/test_26_docs.jl` | every public and private name is documented; the Capability Catalogue is complete in both directions; every citation in `src/` and `ext/` resolves, every `ref_dict` entry has a user, no reference prose is pasted inline, and an API page carries a bibliography block exactly when it cites; a name an extension declares itself carries a docstring once that extension's file is marked `swept` in [`code_health/sweep_manifest.toml`](code_health/sweep_manifest.toml); a swept file's docstring that documents a function building part of a `JuMP` model carries `# JuMP formulation` and each subsection the body's macros demand, a swept file's count of `# Algorithm` sections does not fall below its manifest row, a swept file carries no `# Details` section, and the library-wide count of `# Details` does not rise; an alias docstring carries no section outside the set its kind allows, a dispatch alias in a swept file carries `# Related`, and the library-wide count of dispatch aliases carrying none does not rise; a `Where:` bullet interpolates a `math_dict` value rather than copying it, with the same pair of a swept-file zero and a library-wide count that may not rise, and two `math_dict` keys do not open with one definition head, against the debt list of shared heads that testset records | run the file |
| `test/test_41_constructor_docstring_drift.jl` | a `# Constructors` block matches the signature it copies | run the file |
| `test/test_43_exported_abstract_type_census.jl` | the exported abstract types are exactly the names on the allow-list in that file | run the file |
| `test/test_64_docs_page_metadata_census.jl` | every hand-written docs page, Literate source and root-page generator states a `Description` that is not Documenter's default, is unique across the site and fits the band a search engine shows, with the American spelling on the landing line only; `docs/make.jl` labels the landing page with the ranked title and derives the site-wide fallback from it; and every mirror page's `Description` is the line `docs/page_metadata.jl` derives from its own H1 and `@docs` names, with the private-side H1 suffix; and no hand-written page links another page by a relative `.md` path | run the file |
| `test/test_65_docs_public_private_placement_census.jl` | every `@docs` entry on a page already migrated into `docs/src/public_api/` or `docs/src/private_api/` classifies, by `docs/api_classification.jl`, onto the side that holds it | run the file |
| `test/test_61_interfaces_section_census.jl` | every `## \`verb\`` heading and every call-shaped bullet of a `# Interfaces` section names a function and types the package defines, a bullet under a verb heading names that verb, and every abstract type that is the direct supertype of a concrete type carries a `# Interfaces` section or is on the debt list in that file, which may only shrink | run the file |
| `test/test_40_fallback_shortcut_census.jl` | a fallback shortcut's `Nothing` lands on `fb` | run the file |
| `test/test_44_range_tails_census.jl` | a range risk measure declares its tails, or is on the fused list | run the file |
| `test/test_08l_moment_verb_census.jl` | every concrete leaf of the three moment families answers its family's verbs with a method the library declares below the Choice Surface, and the one named exemption still fails | run the file |
| `test/test_45_sweep_census.jl` | every file under `src/` and `ext/` has a sweep-manifest row naming its child map of #404, the file's documented-unit count still matches that row, and a swept file's documented units are still the bindings its row records, one name per unit | run the file |
| `test/test_47_alias_and_module_census.jl` | an acronym alias of `src/23_Aliases.jl` IS the binding its docstring names, a factory alias of that file EQUALS the long form its sentence names, and `src/PortfolioOptimisers.jl` `include`s every other file under `src/` exactly once, in the order of numeric prefixes that are unique within each directory of `src/`, `ext/`, `docs/src/public_api/` and `docs/src/private_api/` | run the file |
| `test/test_46_standards_citation_census.jl` | every name and every path a standards file cites resolves against the repository, and no standards file states a count of the repository | run the file |
| `test/test_62_partial_fit_state_interface_census.jl` | every `AbstractPartialFitState` owns `merge_states` and `copy` below the root; the `port_opt_view` of every estimator carrying a `cache` slices the cache or refuses the view, or the estimator is on that file's identity list with its reason; every state a view can reach owns `port_opt_view`, and every state that does not is paired there with an identity-list host | run the file |
| `test/test_55_numeric_coercion_census.jl` | no file under `src/` or `ext/` coerces a numeric type with `float` outside `float_if_integer`, none reads a working type off a division of `one` or `zero`, and none rounds an index in one type and converts it in another | run the file |
| `test/test_70_trivial_unit_census.jl` | every one-method unit under `src/` or `ext/` whose body is one depth-one forward and whose name has at most one reference, and every abstract type with one concrete subtype that nothing codes against, is on the allow-list in that file, and every entry of that list is still flagged | run the file |
| `test/test_71_process_citation_census.jl` | no docstring or error text under `src/` or `ext/`, no hand-written page under `docs/src/` outside `docs/src/contribute/`, no non-comment line of the Capability Catalogue and no markdown line of an example or user-guide source names an ADR, an issue or pull-request number, the words `pull request`, or a tracker URL | run the file |
| `test/test_72_prose_census.jl` | every text a user reads as a page carries at or below its row of counted `unslop` rules in `code_health/prose_baseline.toml`, over the Literate sources of `examples/` and `user_guide/`, the hand-written pages of `docs/src/` outside `docs/src/contribute/`, `README.md` and `docs/capability_catalogue.jl`; a text whose every count is zero carries no row, and a mirror page's derived H1 and `Description` line are not read; the strings and the `#!` gotchas of a page's code are read, and every other code comment counts in `comment` | run the file |
| `test/test_63_domain_error_shape_census.jl` | no file under `src/` or `ext/` raises a `DomainError` from a sentence alone or from the bare type, so every raise carries the offending value in `.val` and a message in `.msg` | run the file |
| `test/test_49_coverage_attribution_census.jl` | `code_health/coverage.jl` names a return-annotated definition by its function and a functor method by its receiver type, every Coverage Exemption in `code_health/rulings.toml` names a definition its file holds, and no file under `src/` or `ext/` carries a `COV_EXCL` marker | run the file |
| `test/test_73_performance_trap_census.jl` | each rule of `code_health/perf.jl` flags the shape it names and stays silent on the near shapes its docstring excludes, a `[[perf_dismissal]]` subtracts from its rule's count and not from `raw`, one without `code` covers its rule over one definition, a rise over the recorded row fails `check`, and an added file enters at zero | run the file |
| `test/test_42_combination_weight_stacking.jl` | a combination weight on a meta-optimiser reaches the model | run the file |
| `test/test_28_seam_lock.jl` | JuMP model state is reached only through its typed interface | run the file |
| `test/test_29_risk_measure_compatibility.jl` | a risk measure is paired only with an optimiser that supports it | run the file |
| `test/test_27_prefix_registration.jl` | a nested risk build namespaces its model-state keys | run the file |
| `.github/workflows/Docs.yml` (`doctest`) | every `jldoctest` block still produces its printed output | see the `run-doctests` skill |
| `.github/workflows/Paper.yml` | the paper's listing still runs against this checkout, and `docs/paper/main-jlyfish.json` is current | the workflow |
| `.github/workflows/ReusableTest.yml` (`coverage`) | a file's miss count has not risen above `code_health/coverage_baseline.toml`, an added file enters with every line covered or exempted, and every Coverage Exemption states the exact count it stands for | `julia --project=code_health code_health/coverage.jl check`, with `COVERAGE_LCOV` pointing at an `lcov.info` |
| `.github/workflows/Sweep.yml` | a file whose sweep-manifest row reads `swept = false` under a CLOSED child map of #404 reopens that map, reopens #404, and gets one `sweep` sub-issue | `julia --project=code_health code_health/sweep_triage.jl --fetch` for the plan, then `code_health/sweep_issues.sh apply --dry-run`; the workflow calls the same script |
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
