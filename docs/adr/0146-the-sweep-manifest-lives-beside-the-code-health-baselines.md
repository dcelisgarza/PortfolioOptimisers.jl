---
status: accepted
---

# The sweep manifest lives beside the code-health baselines

## Context

`sweep/manifest.toml` was the only file under `sweep/`, and `sweep/` was a top-level directory of
the repository. The manifest is one row per file under `src/` and `ext/`, and every reader of it
already lives in `code_health/` or loads `code_health/CodeHealth.jl` to read it:

| Reader | Where it lives |
| ------ | -------------- |
| `MANIFEST_PATH`, `manifest_row`, `expected_manifest` | `code_health/CodeHealth.jl` |
| `sweep_check.jl`, `sweep_triage.jl` | `code_health/` |
| `test_26_docs.jl`, `test_45_sweep_census.jl`, `test_47_alias_and_module_census.jl` | `test/`, through `CodeHealth.jl` or the same `joinpath` |
| `Sweep.yml` | `.github/workflows/`, through `sweep_triage.jl` |

The scratch directory the sweep job writes is already `code_health/_sweep/`, beside
`code_health/_triage/` and `code_health/_refresh/`. The four baselines the manifest is reconciled
against — `complexity_baseline.toml`, `coverage_baseline.toml`, `jet_baseline.toml`,
`size_baseline.toml` — and the `rulings.toml` that exempts rows from them are all flat TOML files
in `code_health/`. ADR 0073 made the baselines one directory; ADR 0074 made the manifest's row set
the same set the baselines carry. The manifest was the one file of that family that stood
elsewhere, and a top-level directory for one file said nothing a reader could use.

## Decision

**`sweep/manifest.toml` becomes `code_health/sweep_manifest.toml`, and `sweep/` is gone.**

The name follows its siblings: `<subject>_<kind>.toml`, flat in `code_health/`, so a listing of
the directory reads the manifest and the four baselines as one family.

**The move is structural.** No row of the manifest changes. Every reference under `.github/`,
`.claude/`, `code_health/`, `docs/adr/`, `test/`, `CLAUDE.md` and `STANDARDS.md` names the new
path. No ADR that names the old path has reached `main`, so each is rewritten in place rather than
amended.

## Consequences

The repository has one directory for the measurements that gate a commit and the manifest that
scopes them. A contributor who opens `code_health/` sees every file the sweep and the ratchets
read, and the row a new file owes to each.

A reader who knows the old path finds nothing at it. `CodeHealth.MANIFEST_PATH` is the one
constant that spells it in code, and the three tests that build the path themselves spell the
same `joinpath`.

## Alternatives considered

- **Keep `sweep/` and move the code-health scripts into it.** The scripts measure five things,
  and the sweep is one of them. The directory would be misnamed for four of its five files.
- **`code_health/sweep/manifest.toml`.** A subdirectory for one file, which is the shape this
  decision removes.
- **`code_health/manifest.toml`.** `code_health/Manifest.toml` already exists, and is the Pkg
  manifest that pins the gate (ADR 0077). Two files that differ by case in one directory is a
  trap on a case-insensitive filesystem and a misreading on every other.
