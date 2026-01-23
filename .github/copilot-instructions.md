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
    
      + Follow the guidelines in [`1-contributing.md`](../docs/src/contribute/1-contributing.md) and [`2-developer.md`](../docs/src/contribute/2-developer.md).

  - **Examples:**
    
      + Example scripts and notebooks are in `examples/`. Use these as references for end-to-end workflows.

### Required Before Each Commit

  - **Linting and Formatting:**
    
      + **ALWAYS** run `pre-commit run -a` before committing changes.
      + The repository uses pre-commit hooks for linting and formatting.
      + Julia code is formatted using JuliaFormatter.jl.
      + **Commits will be rejected if pre-commit tests fail.**
      + Install pre-commit hooks once: `pre-commit install` (requires `pre-commit` via pipx).

  - **Testing:**
    
      + **ALWAYS** run tests after making code changes and before committing.
    
      + From Julia REPL:
        
        ```julia
        using Pkg
        Pkg.activate(".")
        Pkg.test()
        ```
      + Add new tests as `test-*.jl` in `test/`.
      + Tests must pass before creating a pull request.

### Build and Development Commands

  - **Formatting:**
    
      + Run linters and formatters: `pre-commit run -a`
      + Manual Julia formatting: Install JuliaFormatter.jl globally, then format as needed.

  - **Testing:**
    
      + Run all tests: From Julia REPL, activate environment (`] activate .`) then run tests (`] test`).
  - **Documentation:**
    
      + Build docs locally:
        
         1. `julia --project=docs`
         2. `] dev .` (first time only)
         3. `using LiveServer`
         4. `servedocs()`

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
    
      + Use a `release-x.y.z` branch, update `Project.toml` and `CHANGELOG.md`, and follow the release checklist in `2-developer.md`.

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

  - Main entry: [src/PortfolioOptimisers.jl](../src/PortfolioOptimisers.jl)
  - Developer guide: [developer](../docs/src/contribute/2-developer.md)
  - Contributing: [contributing](../docs/src/contribute/1-contributing.md)
  - Tests: [test](../test/)
  - Docs: [docs](../docs/)

## Code Review & Maintenance

  - Ensure tests cover new functionality and edge cases.
  - Keep documentation up to date with code changes.
  - Follow semantic versioning for releases.
  - Check spelling and grammar in code, docstrings and docs.
  - Ensure new code follows established patterns and conventions.
  - Ensure new documentation adheres to established patterns and conventions.

## Before Finalizing Work

Before completing any task or creating a pull request, **ALWAYS**:

 1. **Run pre-commit checks**: `pre-commit run -a` - All checks must pass.
 2. **Run tests**: Activate environment and run tests in Julia REPL - All tests must pass.
 3. **Verify changes**: Ensure all file changes align with the task requirements.
 4. **Update documentation**: If adding new features, update relevant docstrings and docs.
 5. **Check code quality**: Ensure code follows established patterns and conventions.

* * *

For further details, see the [README.md](../README.md), [`developer`](../docs/src/contribute/2-developer.md) and in-code documentation.
