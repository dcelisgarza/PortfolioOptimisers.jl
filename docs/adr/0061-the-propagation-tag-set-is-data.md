---
status: accepted
---

# The propagation tag set is data

## Context

`@propagatable` (ADR 0002) declares a struct and generates the `factory`, `port_opt_view` and
prior-selection methods that carry an estimator's fields through propagation. A field opts into
a channel with a tag macro: `@fprop`, `@vprop`, `@pprop`, `@cprop`, `@wprop`. The macro is deep
at the use site. 332 tag annotations across 145 `@propagatable` structs generate 231 methods.

The shallowness was in the recognition layer. The tag set was spelled out, positionally, at
seven sites:

1. five `is_*prop_macro` predicates, one per tag, 68 lines for 15 lines of logic;
2. `is_prop_tag_call`, a five-way `||` over those predicates;
3. `peel_prop_tags`, which tested four tags and let the fifth fall through an `else`;
4. `_record!`, a five-`Bool` positional signature inside `propagatable_parse_body`;
5. five error-only stub macros;
6. the factory emission, an `if`/`elseif` chain encoding the precedence `fprop → wprop`;
7. the prior emission, a second `if`/`elseif` chain encoding `pprop → cprop → wprop → fprop`.

Two hand-written precedence chains, and no comment linked them.

The `else` in `peel_prop_tags` is the hazard. A sixth tag added to the gate but not to the
`elseif` chain does not error. Every field carrying it silently becomes `@wprop`, so `factory`
substitutes observation weights into it. That is a wrong answer with no diagnostic.

The tax is observed, not hypothetical: `@wprop` was the last tag added (ADRs 0012 and 0013),
and it paid all seven.

## Decision

The tag set is data. `PROP_TAG_NAMES` is the table, and the recognition layer is derived
from it:

- `PROP_TAG_MACRO_NAMES` derives each macro name, so a row carries no second spelling.
- `prop_tag(x)` looks a macro reference up in the table and answers a tag or `nothing`. The
    five predicates are deleted.
- `is_prop_tag_call` and `peel_prop_tags` call `prop_tag`. A tag is looked up, never fallen
    through to; `peel_prop_tags` returns a `Set{Symbol}`.
- `propagatable_parse_body` returns one `Dict` entry per table row instead of five positional
    vectors.
- `prop_tag_expr(tag, fname, xf, mod, thread)` is the `name → field transform` half of the
    table: one branch per tag, shared by every channel.
- `PROP_TAG_CHANNELS` holds the precedence as data. Each channel declares a `gate` (the tags
    that make the method be emitted at all) and a `precedence` (the order in which a field's
    tags are consulted). `prop_channel_pairs` reads it, and the two `if`/`elseif` chains are
    deleted.

A new propagation channel is one table row, one `prop_tag_expr` branch, one `PROP_TAG_CHANNELS`
entry and one stub macro. `check_prop_tag_macros` runs at the end of the module and refuses to
precompile when a row lacks any of the three, so a row that parses and never propagates cannot
ship.

The five stub macros stay written out. They are public API and each carries hand-written
documentation that names its own channel; generating them would move that prose into a table
for no gain, and a missing stub is an `UndefVarError` at the use site rather than a wrong
answer. `check_prop_tag_macros` ties them to the table instead.

## Consequences

The generated code is unchanged. A differential test over 15 struct declarations, covering
every tag, both stacking orders, docstring-prefixed fields, an inner constructor and a
parametric head, compares the expansion before and after: 14 of the 15 are byte-identical.
The one difference is the view channel, whose keyword pairs now follow declaration order
instead of listing the `@vprop` fields first. The call is to a keyword constructor, so the
order carries no meaning.

The hazard is closed at both ends. A tag that reaches `peel_prop_tags` without a table row is
not a tag at all, so it is left in place and the field keeps its macro; and a table row without
a transform errors in `prop_tag_expr` rather than taking another tag's transform.

`is_fprop_macro`, `is_vprop_macro`, `is_pprop_macro`, `is_cprop_macro` and `is_wprop_macro` are
gone from the API page. They were private helpers.

ADRs 0002, 0010, 0012 and 0013 decide what the tags *mean*. This decides how the set is
spelled. Nothing in them is contradicted.

## Amendment (2026-08-21)

**The field transform is a function of the channel as well as the tag.** The decision above says
`prop_tag_expr(tag, fname, xf, mod, thread)` holds "one branch per tag, shared by every channel".
That is no longer true. The signature is now
`prop_tag_expr(channel, tag, fname, xf, mod, thread)`, and one tag may carry a different transform
in each channel that names it.

### What forced it

A fourth channel, `obs`, generates `obs_weights_view(x, i)` — the observation-axis counterpart of
`port_opt_view`. Its row is what the decision above promised a new channel would be:

```julia
obs = (gate = (:wprop,), precedence = (:wprop, :fprop))
```

It reads the tags a weights-carrying struct already has, and it needs both of them to mean
something **other** than what they mean to `factory`:

| Tag | `factory` transform | `obs` transform |
| --- | --- | --- |
| `@wprop` | `_wprop` — **replaces** the field with an incoming `ObsWeights` | `nothing_scalar_array_getindex` — **indexes** the value already there |
| `@fprop` | `factory_child` | `obs_weights_view` |

A transform keyed on the tag alone cannot express that. The alternative was a sixth tag, written
beside `@wprop` on every weights field — a second spelling of the same fact, which is the tax this
ADR exists to remove.

### Why the channel could not be one of the three that existed

`factory` writes **one** value into **every** `@wprop` field of a tree at once, so it cannot index a
field and leave a sibling `nothing`. A `SimpleVariance` holding a weighted mean and an unweighted
dispersion comes back from `factory` with both weighted, which is a different estimator.

`port_opt_view` threads one index into every `@vprop` field, and at its call sites — the
meta-optimisers and the cross-validation splitters — that index selects **assets**. An observation
weight is one value per row of the sample. The axis is a per-type convention rather than a property
of the channel, which is why the two verbs are told apart by name and not by argument.

### What the amendment changes

- `prop_tag_expr` takes the channel first. Its error names the channel as well as the tag.
- `check_prop_tag_macros` probes **every channel that names a tag**, not the tag once. A tag added
  to a second channel with no transform there refuses to precompile, exactly as a tag with no
  transform at all already did.
- `PROP_TAG_NAMES` is unchanged. No tag was added, and no struct in `src/` gained an annotation.

### What it does not change

The claim that a new propagation channel is a table row still holds, and this channel is the
evidence: one row, one gate, one precedence, plus the transforms the row needs. The hazard analysis
above is unchanged at both ends — a tag with no table row is not a tag, and a row with no transform
errors rather than taking another tag's. ADRs 0002, 0010, 0012 and 0013 still decide what the tags
mean; this still decides how the set is spelled. The set is now two-dimensional.
