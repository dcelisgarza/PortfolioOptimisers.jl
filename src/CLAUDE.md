# Design rules for library code

These rules bind every file under `src/` and `ext/`. The root `CLAUDE.md` holds the rules for the
whole repository. This file owns no rule. Each line names one rule and links the section that owns
its text, so read that section before you apply the rule. [`CODING_STANDARDS.md`](../CODING_STANDARDS.md)
routes every subject that this list does not name.

[sc]: ../.github/instructions/julia-source-code.instructions.md

- **The standing test picks a design, and correctness wins over convenience.** A break of an
  interface is not a cost, and a name of the break goes in the commit and in
  `docs/src/migration.md`. [`julia-source-code.instructions.md`][sc] § *A design is chosen by
  the standing test*.
- **An estimator never holds a Result**, except a Partial Fit State.
  [`julia-source-code.instructions.md`][sc] § *Estimator, Algorithm, and Result Roles*, and
  [ADR 0106](../docs/adr/0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md).
- **A numeric type is derived, never coerced.** [`julia-source-code.instructions.md`][sc]
  § *Numeric types come from the data*. `test/test_55_numeric_coercion_census.jl` gates it.
- **A container takes the element type of its values, and `Any` is the last resort.**
  [`julia-source-code.instructions.md`][sc] § *A container takes the element type of its
  values*. `test/test_87_any_container_census.jl` gates it.
- **`nothing` spells a skipped procedure or a trivial default, never a singleton. An abstract type,
  not a `Union` of its members. A covariance estimator takes no `mp` field.**
  [ADR 0222](../docs/adr/0222-a-type-is-kept-by-default-and-nothing-spells-a-skipped-procedure.md).
- **A wrapper is for a convenience of the user or for a dispatch problem, and a keyword wrapper is
  for a constructor or a user verb.** [`julia-source-code.instructions.md`][sc] § *A wrapper is
  for convenience or for a dispatch problem*, with the rulings of #2072.
  `test/test_70_trivial_unit_census.jl` and `code_health/forward_survey.jl` gate it.
- **A function returns a defined type or a plain tuple, not a named tuple.**
  [`julia-source-code.instructions.md`][sc] § *A function returns a defined type or a plain
  tuple, not a named tuple*, with the name of a new type. `code_health/forward_survey.jl` reports
  it.
- **The simple form wins, and a method for each type wins over a new dependency.**
  [`julia-source-code.instructions.md`][sc] § *The simple form wins*.
- **The text of a field or an argument comes from the docstring dictionaries.**
  [`julia-docstrings.instructions.md`](../.github/instructions/julia-docstrings.instructions.md)
  § *Documentation Dictionaries*.
- **A user-facing name is declared without approval: `public` when users extend it, exported when
  users use it.** [`julia-source-code.instructions.md`][sc] § *A user-facing name is declared
  without approval*. `test/test_43_exported_abstract_type_census.jl` and
  `test/test_66_public_declaration_census.jl` gate it.
