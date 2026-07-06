---
status: accepted
---

# Drop the Python dependency: native histogram binning and block bootstrap

## Context

Two features borrowed algorithms from Python packages through `PythonCall` and a `CondaPkg.toml`:

- **Histogram bin selection** ([10_Histogram.jl](../../src/08_Moments/10_Histogram.jl)) used
  [AstroPy](https://www.astropy.org/)'s `astropy.stats` for the Knuth, Freedman–Diaconis and
  Scott rules. The abstract type was literally named `AstroPyBins`, and every docstring pointed
  at the AstroPy library.
- **Block bootstrap uncertainty sets**
  ([04_BootstrapUncertaintySets.jl](../../src/14_UncertaintySets/04_BootstrapUncertaintySets.jl))
  used the [`arch`](https://pypi.org/project/arch/) package (`arch-py`) for the stationary,
  circular and moving block bootstraps.

That put a full CPython interpreter and two PyPI/conda packages behind two otherwise small
statistical routines. It made `PortfolioOptimisers.jl` non-installable as a pure-Julia package:
`Pkg.add` alone was not enough — a working Conda environment had to resolve `astropy` and
`arch-py`, which slows precompilation, complicates CI and reproducibility, exposes the package to
Python/Conda ABI drift, and blocks static/`PackageCompiler` deployment. The mathematics involved
(three closed-form bin-width rules; three index-resampling schemes) is modest and well
documented in primary sources.

## Decision

**Remove `PythonCall` and `CondaPkg.toml` entirely; reimplement both features natively in Julia.**

- `AstroPyBins` is renamed **`BinWidthBins`** — the honest name for what the family is (rules that
  pick a bin *count* by first computing an optimal bin *width*). `Knuth`, `FreedmanDiaconis`,
  `Scott` keep their names and gain native implementations; docstrings now cite the primary
  papers ([knuth2019], [freedman1981], [scott1979]) instead of AstroPy.
- The bootstrap generator is reimplemented as
  `bootstrap_indices(alg, rng, T, block_size)` — a pure index-resampling function — replacing the
  Python-object–returning `bootstrap_func`. `StationaryBootstrap` (geometric block lengths, wraps),
  `CircularBootstrap` (fixed length, wraps) and `MovingBootstrap` (fixed length, no wrap) are cited
  to their primary sources ([politis1994stationary], [politis1992circular], [kunsch1989]).
- `PythonCall` is dropped from `[deps]`/`[compat]` in `Project.toml` and from the module's `using`
  list; `CondaPkg.toml` is deleted.

## Considered options

- **Keep the Python packages.** Rejected: the dependency cost (interpreter + Conda resolution +
  ABI/reproducibility risk + no static deployment) is out of proportion to two small numeric
  routines, and it is felt by *every* user, not only those who use histograms or bootstraps.
- **Make Python an optional extension** (weakdep + `Requires`/package extension). Rejected: it
  keeps the reproducibility and deployment problems for anyone who does use the features, and adds
  conditional-loading complexity for algorithms we can just write out.

## Consequences

- The package is now pure Julia: `Pkg.add` installs it with no external toolchain, precompilation
  and CI are faster, and static/`PackageCompiler` deployment is unblocked.
- **Breaking rename** `AstroPyBins → BinWidthBins` (abstract type; the concrete `Knuth` /
  `FreedmanDiaconis` / `Scott` names are unchanged) and the internal `bootstrap_func →
  bootstrap_indices` signature change.
- Reference outputs shifted slightly — the native RNG path differs from `arch`'s, and the native
  bin rules differ from AstroPy at the last ULPs — so `BoxUncertaintySet`, `EllipsoidalUncertaintySet`
  and `covariance` test fixtures were regenerated. The numbers are the algorithms' own outputs now,
  not a foreign library's, so the doctest/CSV suite is the regression net going forward.
- We own the maths: any future bin rule or bootstrap variant is a Julia method, not a Python
  binding.
