---
status: accepted
---

# A `nothing` field is hidden at the REPL, and shown in the docs

## Context

`@define_pretty_show` iterated `fieldnames(typeof(obj))` unconditionally. Every field printed,
and a field that held `nothing` printed as `name ┼ nothing`. Two measurements sized the cost,
taken on the day issue #702 was written:

- 1068 lines in `src/` printed a `nothing` field. Most of them said nothing to a reader.
- Adding one field to the eight moment estimators of issue #308 moved 237 show-render sites:
  `SimpleExpectedReturns` 112, `GeneralCovariance` 51, `Covariance` 49, `SimpleVariance` 19,
  `Coskewness` 3, `Cokurtosis` 3.

So a field that is `nothing` on every default path cost a doctest edit at every site that
rendered its owner, and bought the reader nothing. The `cache` field of issues #700 and #701 is
such a field, and it is the field that made the cost visible.

The library already had one display switch with four channels: `COMPACT_SHOW`, a `ScopedConfig`
with a global setter, a task-scoped `with_*` helper and a load-time preference key. The
documentation build already set it to `false` in two places, so that a rendered docstring keeps
its full detail.

## Decision

**A field that holds `nothing` is hidden by default, and the documentation build shows it.**

A reader at the REPL wants the fields that carry a value. A rendered docstring must show the
complete type, because it is the reference. So the shipped default of the new switch is `false`,
and the two doctest drivers, the docs build and the `run-doctests` skill call
`set_show_nothing_fields!(true)` beside their `set_compact_show!(false)` call. No jldoctest block
moves.

**The selection of fields has three arms, and the precedence is fixed.**

| Arm | Site | Beats |
| --- | --- | --- |
| A per-name entry | `SHOW_NOTHING_FIELDS[].by_type[name]` | the other two |
| The type's own choice | a `show_fields(obj::T)` overload | the global switch |
| The global switch | `SHOW_NOTHING_FIELDS[].default` | nothing |

`show_fields(obj)` returns every declared field by default. A type that always hides a field,
whatever the field holds, overloads it and returns the others. The overload holds under the
global `true` of the docs, so the `cache` field of #700 and #701 stays out of every rendered
docstring, which is the whole point.

A per-name `true` entry renders every declared field of that type, and so overrides the
overload. A per-name `false` entry hides the `nothing` fields of that type, whatever the global
switch says. This is what lets a caller override the per-type choice in either direction.

`pretty_show_fields(obj)` resolves the three arms in one place, and the macro reads it at both
sites that used to read `fieldnames`: the field list of the object, and the emptiness test that
decides whether a nested value prints on one line. So the `┴` marker lands on the last rendered
field, and a nested value whose every field is hidden takes `┴` as a fieldless one does.

**The per-name key is a `Symbol`, not a `Type`.** The load-time channel is TOML, which carries no
types, so the name is the only key it can carry. The name is the bare name of the type, so two
types of one name in two modules share one entry. A name that matches no type is accepted and
does nothing: a type outside the package can render through `@define_pretty_show` too, so an
unknown name is not an error the package can detect.

**A type whose every field is hidden prints as `T()`.** It reaches the branch a fieldless type
already took, so its name still prints and no empty body follows.

**Two preference keys.** `"show_nothing_fields"` is a boolean for the global switch, and
`"show_nothing_fields_by_type"` is a table of booleans, one per type name. A wrong value fails
closed at load, as every other preference value does.

## Consequences

- A `nothing` field disappears from the REPL rendering of every estimator, algorithm and result.
  A user who wants it back sets `set_show_nothing_fields!(true)`, or puts the key in
  `LocalPreferences.toml`.
- The docs render exactly the text they rendered before, because both doctest drivers and the
  docs build set the switch to `true`.
- Two expectations in `test/test_48_base_contracts.jl` moved: the `n ┴ nothing` line of the probe
  is hidden under the shipped default, and the `┴` marker moved to `dt`. Every other test file
  compares no rendered `nothing` line.
- A field that a type never wants shown goes through a `show_fields` overload, not through the
  configuration. Issues #700 and #701 use it for `cache`.
- `apply_preferences!` hands the two new keys to `apply_show_preferences!`, so its own branch
  count does not rise.

## Alternatives considered

- **Show every field, and pay the 237 sites.** Refused. Each site costs a doctest edit, and the
  reader gains a line that says `nothing`.
- **A per-type overload only, with no configuration.** Refused. The docs could not turn a hidden
  field back on, and a user could not hide the `nothing` fields of a type the library did not
  think of.
- **Key the per-name entry on a `Type`.** Refused. TOML carries no types, so the preference
  channel could not carry the entry.
- **Hide a `nothing` field in the docs too.** Refused. A rendered docstring is the reference, and
  a reader who looks up a type must see every field it has.
