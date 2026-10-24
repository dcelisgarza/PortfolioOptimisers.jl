# Design rules for library code

These rules bind every file under `src/` and `ext/`. The root `CLAUDE.md` holds the rules for the
whole repository.

- **Estimators never hold Results internally.** Enforce it with the field's *type bound*, not a
  runtime check. Put precomputed structure on the Result type. The rule has one exception, and
  [ADR 0106](../docs/adr/0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md) states it.
  A partial-fit state is a Result that no consumer reads, so an estimator may hold one in a field
  bound to `Union{Nothing, <:AbstractPartialFitState}`. That bound is the enforcement, and it still
  refuses every other Result.
- **A numeric type is derived, never coerced.** Read it off the arguments with `eltype`, `typeof`,
  `real` and `promote_type`, or off the operation that widens it, and take an index with the
  in-function conversion `ceil(Int, x)` rather than `Int(ceil(x))`. Never wrap a derived type in
  `float`, and never read it off a division such as `typeof(one(T) / one(T))`. Both replace a
  `Rational`, and every other type that needed no repair. When integer data must hold a fraction or
  a `NaN`, pass the type of the data to `float_if_integer`. It converts an `Integer` to a float
  type and keeps every other type. The library must stay open to number types it has never seen,
  such as an AD dual, a unit-carrying quantity, a `Rational`, or a number type that another package
  defines. Every forced type closes the library to one of them. [`.github/instructions/julia-source-code.instructions.md`](../.github/instructions/julia-source-code.instructions.md)
  § *Numeric types come from the data* owns the rule, and
  `test/test_55_numeric_coercion_census.jl` gates it.
- **Prefer a per-type method over a new dependency** for reflection-style work. Derive the field
  list, write the constructor name once per type, and use the ordinary keyword constructor.
- Keep docstring field text in `field_dict` / `arg_dict` in
  `src/01_Base/01_DocstringDictionaries/`. Delete an entry that loses its last user.
  [`.github/instructions/julia-docstrings.instructions.md`](../.github/instructions/julia-docstrings.instructions.md)
  owns the rule for when a description must interpolate a key and when prose is permitted.
- **A user-facing name is declared without approval (#1489).** A name that users are meant to
  extend is `public`: an abstract type they subtype, and a verb that a subtype implements. A name
  that users are meant to use directly is exported: a type they construct, a function they call.
  When a name is unambiguously one of the two, declare it in the change that adds it. Ask the
  maintainer only when it is ambiguous whether users extend or use the name at all.
  - An `# Interfaces` section marks an extension point (ADR 0154), so its type and the verbs it
    names are `public`. A family whose concrete members are exported is an extension point.
  - A helper that the section tells an implementer to call is `public` too. Name a caller or a
    check that the library runs in plain code, with no link. ADR 0154 states the rule, and
    `test/test_66_public_declaration_census.jl` gates it.
  - Users extend an abstract type and do not use it, so it is `public`, not exported. Export one
    only when it is clear that users write it by name in their own code. Ask when it is not clear.
  - Internal machinery stays undeclared. A name that only the library calls is neither.

  `test/test_43_exported_abstract_type_census.jl` holds every exported and every `public` abstract
  type on an allow-list, so a change that declares an abstract type also edits that list.
  **Do not restate the count here or anywhere else.** The count changed four times, and each
  written copy became out of date.
