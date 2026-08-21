---
status: accepted
---

# The code-health gate is two workflows, pinned by one committed Manifest

## Context

[ADR 0076](0076-the-code-health-pass-rule-is-a-ratchet.md) fixes the pass rule and
[ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) sites the files it reads. Neither
says what runs them. Three generated files cost two orders of magnitude apart — the complexity
measurement takes about 5 seconds, the Expansion Bound about 80 seconds, and the JET measurement
5 minutes 46 seconds at 2.4 GiB peak — and they answer to different triggers.

A baseline is a claim about an analyser. It is comparable only against a fixed one, so the versions
must be pinned. `Aqua.yml` is the precedent for a quality gate here: it `Pkg.add`s its tool in the
workflow rather than taking a test dependency, and it triggers on push to `main`, `dev*` and
`agents/*`, on pull request, and on `workflow_dispatch`.

Two facts constrain where the code lives. The pre-commit `julia-formatter` hook excludes
`^\.github/`, while `FormatCheck.yml` runs `format(".")` over the whole tree. So a `.jl` file under
`.github/` turns CI red on a commit that was clean locally. And `.gitignore` line 7 is a bare
`Manifest.toml`, so no Manifest is committed anywhere in this repository today.

## Decision

### Two workflows, split on the trigger and not on world age

`Complexity.yml` takes `Aqua.yml`'s events with **no `paths:` filter**.
[ADR 0072](0072-the-complexity-gate-measures-src-and-ext.md) makes the coverage assertion read the
whole tree, so a `src/**` filter would never fire on a new `.jl` file elsewhere — the very commit
that breaks the rule would be the commit that skips the check. It runs `complexity.jl check` and
then `expansion.jl check`, both unconditionally, for about 85 seconds.

`JET.yml` takes the same events **plus** a filter over `src/`, `ext/`, `code_health/`,
`Project.toml` and itself. JET's number cannot change unless one of those changed, and this
repository produces a great many docs-only commits, so the filter skips most six-minute runs and
misses no code change.

World age forced nothing. Two `julia` invocations are two fresh processes whether they sit in one
job or two. What forces two workflows is the asymmetry of the triggers.

Both take `FormatCheck.yml`'s concurrency idiom, which cancels a superseded pull-request run and
never cancels a push to a protected branch.

### `code_health/Manifest.toml` is the single source of truth for every version

The tools install from **`code_health/Project.toml` plus a committed `code_health/Manifest.toml`**,
instantiated with `Pkg.instantiate()`. The Manifest is what makes the pin total.

Equality `[compat]` bounds were rejected because they bind direct dependencies only. CodeComplexity
**is** a `JuliaSyntax` parser, so its definition count can shift when `JuliaSyntax` moves under it,
and no bound on CodeComplexity sees that. Ordinary caret compat was rejected because a JET patch
release would then turn CI red with no source change, and the failure would read as a regression
the author caused.

The committed Manifest needs an explicit `!code_health/Manifest.toml` negation in `.gitignore`. It
is this repository's first committed Manifest, and a visible exception is the right shape for a
deliberate one. CompatHelper never edits a Manifest, so the pin holds whichever subdirectories
`CompatHelper.main()` walks.

### The Julia version is read from the Manifest, never written in the YAML

The pin is **Julia 1.12.7**, written once as the Manifest's `julia_version` and read by both
workflows. This inverts `Aqua.yml`'s order: checkout comes **before** `setup-julia`, because the
version must be read out of the checkout.

```bash
v=$(grep '^julia_version' code_health/Manifest.toml | cut -d'"' -f2)
```

`setup-julia@latest` stays floating, per [ADR 0056](0056-the-julia-setup-action-floats-on-latest.md).

After this decision there is no version in the YAML that could float, so the item asking for a note
explaining why it must not float became arithmetic instead of prose. The real hazard is a casual
`Pkg.update()` inside `code_health/`: the Manifest advances, the pin advances with it, and the
baseline is compared against a different analyser. ADR 0073's provenance comparison catches that,
and the failure text names the fix — refresh the baseline in the **same** commit that moves
`code_health/Manifest.toml`.

### One entry script per generated file, over a shared module

```text
code_health/
  Project.toml
  Manifest.toml
  CodeHealth.jl           # shared: TOML read and write, provenance comparison, rendering
  complexity.jl           # -> complexity_baseline.toml     ~5 s
  expansion.jl            # -> expansion_bound.toml         ~80 s
  jet.jl                  # -> jet_baseline.toml            5:46
```

Each script takes one of three commands.

```bash
julia --project=code_health code_health/<tool>.jl check
julia --project=code_health code_health/<tool>.jl refresh
julia --project=code_health code_health/<tool>.jl refresh --accept-rise
```

The directory is `code_health/`, snake case after the existing `user_guide/`, and it matches the
`code-health` issue label. `.github/` was rejected on the mechanical fact above: a `.jl` file there
escapes the pre-commit hook and is then reformatted by `FormatCheck.yml`.

### `jet.jl` asserts its environment before it measures

JET 0.12.1 needs Julia 1.12 and degrades **silently**: on an unsupported version it loads
`JETEmpty.jl`, whose stubs warn on load and throw on call. So `jet.jl` asserts `JET.JET_AVAILABLE`,
and it asserts that each run's recorded load set is loaded, before it measures anything. A gate that
stops measuring without failing is worse than no gate.

### What a run prints

A trip prints **one `::error` annotation per offending file**, not only the first, plus a table in
`$GITHUB_STEP_SUMMARY`. The annotation is absent when a Declaration Macro in another file is the
cause, because the offending line is not in the file that changed. A green run publishes provenance:
the versions, the load sets, the files measured and the totals. A run that trips publishes the
Refresh Artifact, per [ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md).

## Consequences

- Every version the gate depends on moves in exactly one commit, and that commit must also refresh
  the baselines or the provenance comparison fails it.
- `JET.yml` carries a `paths:` filter, so it reports as skipped on a docs-only pull request. A
  skipped **required** check blocks a pull request. Nothing is a required check in this repository
  today, and whoever configures branch protection later must handle it. `Complexity.yml` carries no
  filter and is not exposed.
- The `code_health/` scripts are formatted by `FormatCheck.yml` and by the pre-commit hook, like
  every other `.jl` file in the repository.
