# Design rules for library code

These rules bind every file under `src/` and `ext/`. The root `CLAUDE.md` holds the rules for the
whole repository.

- **Estimators never hold Results internally.** Enforce it with the field's *type bound*, not a
  runtime check; precomputed structure belongs on the Result type. The rule has one exception, and
  [ADR 0106](../docs/adr/0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md) states it:
  a partial-fit state is a Result that no consumer reads, so an estimator may hold one in a field
  bound to `Union{Nothing, <:AbstractPartialFitState}`. That bound is the enforcement, and it still
  refuses every other Result.
- **A numeric type is derived, never coerced.** Read it off the arguments with `eltype`, `typeof`,
  `real` and `promote_type`, or off the operation that widens it, and take an index with the
  in-function conversion `ceil(Int, x)` rather than `Int(ceil(x))`. Never wrap a derived type in
  `float`, and never read it off a division such as `typeof(one(T) / one(T))`: both clobber a
  `Rational` and every type that needed no repair. When integer data must hold a fraction or a
  `NaN`, pass the type of the data to `float_if_integer`, which floats an `Integer` and keeps every
  other type. The library must stay open to number types it has never seen — an AD dual, a unit-carrying
  quantity, a `Rational`, a number type another package defines — and every forced type closes it to
  one of them. [`.github/instructions/julia-source-code.instructions.md`](../.github/instructions/julia-source-code.instructions.md)
  § *Numeric types come from the data* owns the rule, and
  `test/test_55_numeric_coercion_census.jl` gates it.
- **Prefer a per-type method over a new dependency** for reflection-style work. Derive the field
  list, write the constructor name once per type, and use the ordinary keyword constructor.
- Docstring field text is centralised in `field_dict` / `arg_dict` in
  `src/01_Base/01_DocstringDictionaries/`, and an entry
  that loses its last user is deleted. When a description must interpolate a key and when prose is
  permitted is stated by
  [`.github/instructions/julia-docstrings.instructions.md`](../.github/instructions/julia-docstrings.instructions.md),
  which owns that rule.
- **A user-facing name is declared without approval (#1489).** A name that users are meant to
  extend is `public`: an abstract type they subtype, and a verb that a subtype implements. A name
  that users are meant to use directly is exported: a type they construct, a function they call.
  When a name is unambiguously one of the two, declare it in the change that adds it. Ask the
  maintainer only when it is ambiguous whether users extend or use the name at all.
  - An `# Interfaces` section marks an extension point (ADR 0154), so its type and the verbs it
    names are `public`. A family whose concrete members are exported is an extension point.
  - A helper that the section tells an implementer to call is `public` too, and a caller or a
    check that the library runs is named in plain code, with no link. ADR 0154 states the rule,
    and `test/test_66_public_declaration_census.jl` gates it.
  - An abstract type is extended, not used, so it is `public`, not exported. Export one only when
    users plainly write it by name in their own code, and ask when that is not plain.
  - Internal machinery stays undeclared. A name that only the library calls is neither.

  `test/test_43_exported_abstract_type_census.jl` holds every exported and every `public` abstract
  type on an allow-list, so a declared abstract type is an edit to that list in the same change.
  **Do not restate the count here or anywhere else.** It has moved four times, and each written
  copy went stale where it stood.
