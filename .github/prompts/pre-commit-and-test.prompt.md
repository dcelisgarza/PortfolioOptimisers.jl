---
agent: ask
description: Run pre-commit checks and tests before committing changes to PortfolioOptimisers.jl.
---

Run the pre-commit checks, the tests for the area you changed, and the doctests before you commit.

## Step 1. Run pre-commit checks

From the repository root:

```bash
pre-commit run -a
```

All hooks must pass, with exit code 0, before you go on. If a hook fails:

- **JuliaFormatter.** The hook applies its fixes itself. Run `pre-commit run -a` again to confirm that they pass.
- **Other hooks.** Read the output and fix the cause.

Do not go on to the tests until `pre-commit run -a` exits with no failures.

## Step 2. Run the tests for the area you changed

Run the targeted `test_*.jl` files for the area you changed. [`CLAUDE.md`](../../CLAUDE.md) § *Running Julia* owns how to run them, and why the full suite is the maintainer's to run.

All of them must pass. If any test fails:

1. Read the whole failure message to find the failing test and its file.
2. Find whether your changes cause the failure, or whether it failed before them.
3. Fix the failures that your changes cause.
4. Report a failure that came before your changes to the user, and ask whether to go on.

## Step 3. Run doctests

`] test` does **not** run the doctests. They are a separate step. The `run-doctests` skill owns the command, which mirrors the `doctest` job of `.github/workflows/Docs.yml`, and the fresh-process rule.

All doctests must pass. If a doctest fails:

- Check that the `jldoctest` block matches the output of `@define_pretty_show`.
- Check that the field names, types, and default values match the struct definition.
- Update the doctest to match the current output exactly, then run it again.

## Step 4. Confirm readiness

When all three steps pass with no failures, report the result to the user:

```text
pre-commit run -a  ✓  All N hooks passed.
targeted tests     ✓  All N tests passed.
doctest            ✓  All doctests passed.
```

The changes are ready to commit.
