# Copilot Instructions for PortfolioOptimisers.jl

## Project Overview

PortfolioOptimisers.jl is a modular Julia package for portfolio optimization, supporting a wide range of models, constraints, and risk measures. The codebase is organized by functional domains (moments, constraints, priors, risk measures, optimization, etc.), with each major component in its own file or subdirectory under `src/`.

## Key Architectural Patterns

  - **Modular Structure:** Each subdomain (e.g., moments, priors, constraints) is implemented in its own file or subfolder. The main module (`src/PortfolioOptimisers.jl`) includes all components explicitly.
  - **Extension System:** Optional plotting extensions are in `ext/` and registered in `Project.toml` under `[extensions]`.
  - **Documentation:** Source docs are in `docs/src/`, with developer and contributor guides in `90-contributing.md` and `91-developer.md`.
  - **Testing:** Each test file in `test/` is auto-included if named `test-*.jl`. Do not add tests to `runtests.jl` directly.

## Developer Workflows

  - **Linting/Formatting:**
    
      + Use [EditorConfig](https://editorconfig.org) and [pre-commit](https://pre-commit.com) with JuliaFormatter.jl.
      + Run all hooks: `pre-commit run -a`.
  - **Testing:**
    
      + From Julia REPL: `] activate .` then `] test`.
      + Add new tests as `test-*.jl` in `test/`.
  - **Docs:**
    
      + Build locally: `julia --project=docs`, then `using LiveServer; servedocs()`.
  - **Branching/PRs:**
    
      + Branch from up-to-date `main` (see `91-developer.md` for naming conventions).
      + Ensure all tests and pre-commit hooks pass before PR.

## Project-Specific Conventions

  - **File Naming:**
    
      + Source files are prefixed numerically to indicate load order and logical grouping.
      + Test files must be named `test-*.jl` for auto-discovery.
  - **Commit/PR Practices:**
    
      + Use imperative, informative commit messages.
      + Keep commits atomic and rebase on `main` before PR.
  - **Releases:**
    
      + Use a `release-x.y.z` branch, update `Project.toml` and `CHANGELOG.md`, and follow the release checklist in `91-developer.md`.

## Integration Points & Dependencies

  - **Julia dependencies** are managed in `Project.toml` and loaded in the main module.
  - **Python dependencies** (for PythonCall) are managed in `CondaPkg.toml`.
  - **Optional plotting** via extensions in `ext/`.

## References

  - Main entry: `src/PortfolioOptimisers.jl`
  - Developer guide: `docs/src/91-developer.md`
  - Contributing: `docs/src/90-contributing.md`
  - Tests: `test/`
  - Docs: `docs/`

For further details, see the [README.md](../README.md), [91-developer.md](../docs/src/91-developer.md) and in-code documentation.
