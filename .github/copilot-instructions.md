# Copilot Instructions for PortfolioOptimisers.jl

## Project Overview

PortfolioOptimisers.jl is a Julia package for portfolio optimization and risk management. Its code composes three kinds of type: estimators, algorithms and results. The estimators cover statistical, econometric and optimization methods.

## Architecture & Key Patterns

- **Modular design.** The main module, `src/PortfolioOptimisers.jl`, includes one submodule for each subject, such as moments, risk, priors, constraints and optimisation. Each submodule is split into small files, for example one file for each covariance estimator or risk measure.
- **Abstract types.** Every estimator, algorithm and result descends from an abstract type, such as `AbstractEstimator`, `AbstractAlgorithm` or `AbstractResult`. New functionality should subtype these types, so that it dispatches like the rest of the library.
- **Composability.** An estimator or an algorithm can hold another one. For example, a matrix post-processing estimator can wrap a covariance estimator, and a mean estimator can hold a shrinkage algorithm.
- **Validation.** The code validates input with `@argcheck`, from ArgCheck.jl, and with shared assertion helpers. [`julia-source-code.instructions.md`](instructions/julia-source-code.instructions.md) § *Input Validation* owns the rule.
- **Documentation.** Every public type and method has a docstring, with usage examples and references to related types.

## Developer Workflows

- **Contributing and development.**

  - Follow the guidelines in [`1-contributing.md`](../docs/src/contribute/1-contributing.md) and [`2-developer.md`](../docs/src/contribute/2-developer.md).

- **Examples.**

  - The example scripts and notebooks are in `examples/`. Read them for end-to-end workflows.

### Required Before Each Commit

- **Linting and formatting.**

  - Run `pre-commit run -a` before you commit.
  - The pre-commit hooks lint and format the repository.
  - JuliaFormatter.jl formats the Julia code.
  - **A commit fails when a pre-commit check fails.**
  - Install the pre-commit hooks once with `pre-commit install`. It needs `pre-commit`, installed through pipx.

- **Testing.**

  - Run the tests for the area you changed before you commit. [`CLAUDE.md`](../CLAUDE.md) § *Running Julia* owns how to run them.

  - Add a new test as a `test_*.jl` file in `test/`.
  - The tests must pass before you open a pull request.

### Build and Development Commands

- **Formatting.**

  - Run the linters and formatters with `pre-commit run -a`.
  - To format Julia code by hand, install JuliaFormatter.jl globally and run it.

- **Testing.**

  - To run all tests, activate the environment in the Julia REPL with `] activate .`, then run `] test`.
- **Documentation.**

  - To build the docs locally, run these steps:

         1. `julia --project=docs`
         2. `using LiveServer`
         3. `servedocs()`

     `docs/Project.toml` points `PortfolioOptimisers` at this checkout through `[sources]`. Do not run `] dev .`, because `Pkg.develop` rewrites that entry.

## Project-Specific Conventions

- **Naming.**

  - The name of an abstract type always starts with `Abstract`, for example `AbstractCovarianceEstimator`.
  - A concrete type has a descriptive name, and it is often parameterized, for example `SimpleExpectedReturns{T1}`.
  - The name of an algorithm ends in `Algorithm` or names the method, for example `SpectralDenoise` or `PCA`.

- **File naming.**

  - The name of a source file starts with a number, which gives its load order and its group.
- **Extensibility.**

  - To add a new result, estimator or algorithm type, subtype the relevant abstract type and implement the interfaces that it needs.
  - For a new matrix processing or post-processing step, implement its interface and document how to use it.
- **Validation.**

  - Validate input as [`julia-source-code.instructions.md`](instructions/julia-source-code.instructions.md) § *Input Validation* states. Prefer a shared assertion helper over an inline `@argcheck`.
- **Dispatch.**

  - Prefer multiple dispatch over conditionals to select and extend an algorithm.
- **Commits and pull requests.**

  - Write each commit message in the imperative mood, and say what the commit changes.
  - Keep each commit atomic.
  - A session in this checkout rebases onto `dev` and pushes `dev`, as [`CLAUDE.md`](../CLAUDE.md) § *Parallel sessions* states.
  - An outside contributor who works from a fork rebases onto `upstream/main` before a PR, as [`docs/src/contribute/2-developer.md`](../docs/src/contribute/2-developer.md) states.
- **Releases.**

  - Use a `release-x.y.z` branch, update `Project.toml`, and follow the release checklist in `2-developer.md`.

## Integration & Dependencies

- **Julia dependencies.**

  - `Project.toml` declares them, and the main module loads them.

- **Package extensions.**

  - The package extensions are in `ext/`.
- **Cross-component patterns.**

  - A higher-level routine takes estimators and algorithms as arguments. For example, an optimizer receives a covariance estimator, and that estimator can wrap a denoiser or a detoner.
  - Matrix processing and post-processing are always explicit and composable.

## Examples

- To add an estimator, an algorithm or a result type, follow [`add-estimator.prompt.md`](prompts/add-estimator.prompt.md), [`add-algorithm.prompt.md`](prompts/add-algorithm.prompt.md) or [`add-result.prompt.md`](prompts/add-result.prompt.md). Each step links to the section that owns its rule.
- Before you design a change under `src/` or `ext/`, read [`julia-source-code.instructions.md`](instructions/julia-source-code.instructions.md) § *A design is chosen by the standing test*. [`src/CLAUDE.md`](../src/CLAUDE.md) lists the other design rules, each with a link to its owner.

## Notable Patterns

- Every estimator, algorithm and result is composable and validated.
- Each test file runs in isolation, and the test runner finds the files.
- The code checks its inputs with `@argcheck`.
- The docs build renders the docstrings and the examples into the documentation.

## References

- Main entry: [src/PortfolioOptimisers.jl](../src/PortfolioOptimisers.jl)
- Developer guide: [developer](../docs/src/contribute/2-developer.md)
- Contributing: [contributing](../docs/src/contribute/1-contributing.md)
- Tests: [test](../test/)
- Docs: [docs](../docs/)

## Code Review & Maintenance

- Follow semantic versioning for releases.

## Before Finalizing Work

Before you finish a task or open a pull request:

 1. **Run the pre-commit checks.** Run `pre-commit run -a`. Every check must pass.
 2. **Run the tests.** Run the tests for the area you changed, as `CLAUDE.md` § *Running Julia* states. All of them must pass.
 3. **Run the doctests.** `] test` does not run the doctests. The `run-doctests` skill owns the exact command.
 4. **Update the documentation.** When you add a feature, update the docstrings and the docs that it reaches.
 5. **Update the Capability Catalogue.** Give each new type and each new export its place, as [`julia-source-code.instructions.md`](instructions/julia-source-code.instructions.md) § *Capability Catalogue (required for every new public-facing addition)* states. `test/test_26_docs.jl` fails if you skip this step.
 6. **Conform an addition to the sweep.** A file added or changed under `src/` or `ext/` owes the four steps of `CLAUDE.md` § *Functionality you add*. Run `julia --project=code_health code_health/sweep_check.jl --fetch`. It reports each of the four steps, and it prints the manifest line to paste. See ADR 0084.

* * *

For more detail, read the [README.md](../README.md), the [`developer`](../docs/src/contribute/2-developer.md) guide and the documentation in the code.
