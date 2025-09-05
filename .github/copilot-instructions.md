# Copilot Instructions for PortfolioOptimisers.jl

## Project Overview

PortfolioOptimisers.jl is a modular, extensible Julia package for advanced portfolio optimization, risk management, and financial analytics. The codebase is organized around composable estimators, algorithms, and result types, supporting a wide range of statistical, econometric, and optimization techniques.

## Architecture & Key Patterns

  - **Modular Design:** The main module (`src/PortfolioOptimisers.jl`) includes a large set of submodules, each focused on a specific aspect (moments, risk, priors, constraints, optimisation, etc.). Each submodule is further split into fine-grained files (e.g., different covariance estimators, risk measures, etc.).
  - **Abstract Types:** All estimators, algorithms, and results are built on a hierarchy of abstract types (e.g., `AbstractEstimator`, `AbstractAlgorithm`, `AbstractResult`). New functionality should extend these types for consistency and dispatch.
  - **Composability:** Estimators and algorithms are designed to be composed. For example, a covariance estimator can be wrapped with a matrix post-processing estimator, or a mean estimator can be combined with a shrinkage algorithm.
  - **Validation:** The codebase uses `@argcheck` (from ArgCheck.jl) extensively for input validation and defensive programming. Always validate arguments in new methods.
  - **Documentation:** All public types and methods are documented with docstrings, including usage examples and references to related types.

## Developer Workflows

  - **Contributing and development:**
    
      + Follow the guidelines in [`090-contributing.md`](../docs/src/090-contributing.md) and [`091-developer.md`](../docs/src/091-developer.md).

  - **Examples:**
    
      + Example scripts and notebooks are in `examples/`. Use these as references for end-to-end workflows.
  - **Testing:**
    
      + From Julia REPL: `] activate .` then `] test`.
      + Add new tests as `test-*.jl` in `test/`.
  - **Docs:**
    
      + Build locally: `julia --project=docs`, then `using LiveServer; servedocs()`.

## Project-Specific Conventions

  - **Naming:**
    
      + Abstract types are always prefixed with `Abstract` (e.g., `AbstractCovarianceEstimator`).
      + Concrete types are descriptive and often parameterized (e.g., `SimpleExpectedReturns{T1}`).
      + Algorithms are suffixed with `Algorithm` or named after the method (e.g., `SpectralDenoise`, `PCA`).

  - **File Naming:**
    
      + Source files are prefixed numerically to indicate load order and logical grouping.
  - **Extensibility:**
    
      + To add a new result, estimator or algorithm type, subtype the relevant abstract type and provide the necessary interfaces.
      + For new matrix processing or post-processing steps, implement the appropriate interface and document usage.
  - **Validation:**
    
      + Use `@argcheck` for all input validation, especially in constructors and public methods.
  - **Dispatch:**
    
      + Prefer multiple dispatch over conditionals for algorithm selection and extension.
  - **Commit/PR Practices:**
    
      + Use imperative, informative commit messages.
      + Keep commits atomic and rebase on `main` before PR.
  - **Releases:**
    
      + Use a `release-x.y.z` branch, update `Project.toml` and `CHANGELOG.md`, and follow the release checklist in `091-developer.md`.

## Integration & Dependencies

  - **Julia dependencies:**
    
      + Managed in `Project.toml` and loaded in the main module.

  - **Python dependencies:**
    
      + (for PythonCall) are managed in `CondaPkg.toml`.
  - **Package extensions:**
    
      + via extensions in `ext/`.
  - **Cross-Component Patterns:**
    
      + Estimators and algorithms are passed as arguments to higher-level routines (e.g., an optimizer receives a covariance estimator, which itself may wrap a denoiser or detoner).
      + Matrix processing and post-processing are always explicit and composable.

## Examples

  - To implement a new covariance estimator:
    
     1. Subtype `AbstractCovarianceEstimator`.
     2. Provide a constructor with argument validation.
     3. Add docstrings with usage and related types.
     4. Add a test file in `test/`.

  - To add a new algorithm used by an estimator:
    
     1. Subtype the appropriate subtype of `AbstractAlgorithm`.
     2. Implement the required interface.
     3. Document and test as above.

## Notable Patterns

  - All estimators, algorithms, and results are composable and validated.
  - Tests are isolated per file and auto-discovered.
  - Defensive programming is enforced via `@argcheck`.
  - Documentation is tightly integrated with code and examples.

## References

  - Main entry: `src/PortfolioOptimisers.jl`
  - Developer guide: `docs/src/91-developer.md`
  - Contributing: `docs/src/90-contributing.md`
  - Tests: `test/`
  - Docs: `docs/`

* * *

For further details, see the [README.md](../README.md), [91-developer.md](../docs/src/091-developer.md) and in-code documentation.
