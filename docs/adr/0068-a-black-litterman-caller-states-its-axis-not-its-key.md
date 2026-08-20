---
status: accepted
---

# A Black-Litterman caller states its axis, not its key

## Context

`bl_preroll` pre-computes the shared Black-Litterman inputs: the view matrix `P`, the view
returns `Q`, the blending parameter `tau`, and the scaled uncertainty matrix `omega`. Four
estimators call it — `BlackLittermanPrior`, `BayesianBlackLittermanPrior`,
`FactorBlackLittermanPrior`, and `AugmentedBlackLittermanPrior`, the last one twice.

Views resolve their names against one declared axis of [`UniverseSets`](@ref). Which axis is a
property of the estimator, not of the views: a `BlackLittermanPrior` view lands on the asset
distribution, and a view that updates the **factor** distribution lands on the factor axis.

The last argument was the *resolved key*, `key::Option{<:AbstractString}`. A caller therefore
had to read the key off its own `sets` before it could say which axis it meant. Only one shape
reaches these estimators with no `sets` at all: views supplied as a `BlackLittermanViews`
result carry their own `P`, resolve no names, and are deliberately permitted to supply none.
Reading `sets.fkey` to describe a universe that does not exist is the same error as reading the
universe itself, so every factor-axis caller wrote the same guard:

```julia
# Precomputed views ignore both the sets and the key, and are the one shape that reaches here
# with `pe.sets === nothing`, so the key is read only when there is a sets to read it from.
f_key = isnothing(pe.sets) ? nothing : pe.sets.fkey
```

Three copies, each under its own three-line comment explaining the same thing. The asset-space
caller passed nothing at all and needed no guard, so the argument was cheap for exactly one of
the four callers and cost the other three a nothing-guard each.

Raised as "`bl_preroll`'s axis" in the architecture review of 2026-08-17.

## Decision

The last argument names the **axis**, not the key on it:

```julia
function bl_preroll(views, sets, views_conf, prior_sigma, pe_tau, T, datatype, strict,
                    axis::Symbol = :xkey)
    @argcheck(axis in (:xkey, :fkey),
              DomainError(axis,
                          "axis must name a declared axis a view can land on, :xkey or :fkey"))
    key = isnothing(sets) ? nothing : getproperty(sets, axis)
    …
```

`axis` is a *field of* `UniverseSets` rather than a value read out of one: `:xkey` for the
asset axis, `:fkey` for the factor axis. A caller can name its axis whether or not it has a
`sets`, which is what removes the guard. Resolving the axis to a key is `bl_preroll`'s work,
and it happens only when there is a `sets` to read it from.

`:xkey` is the default because it is the axis the asset-space caller never has to supply. It
resolves identically to the old `nothing`: `get_black_litterman_views` already substituted
`sets.xkey` for a `nothing` key.

The `@argcheck` narrows the interface to the two axes a view can land on. Without it a wrong
symbol either raises a `FieldError` about a field of `UniverseSets` or, for `:zkey`, silently
resolves the feature axis.

## Consequences

The three nothing-guards and their three comments are gone. The two factor-axis callers and
the factor half of the augmented caller pass `:fkey` at the call site, and the asset-space
callers pass nothing.

Behaviour is unchanged. `:xkey` resolves to `sets.xkey`, which is what a `nothing` key already
resolved to, and `:fkey` resolves to `sets.fkey`, which is what the deleted guards computed. A
caller with `sets === nothing` still reaches `black_litterman_views` with `key === nothing`,
which the `BlackLittermanViews` method ignores along with the sets.

The width checks are untouched. The four callers keep their own universe-width pre-checks —
each names the estimator and the axis in its message, and each runs before the wrapped prior
is computed, so the caller fails fast and reads an error written in its own vocabulary.
`bl_preroll` keeps the post-check that a precomputed `P` is the width of the distribution it
updates, which is the only place that width is ever seen.

`bl_preroll` is unexported. Its docstring uses `TYPEDSIGNATURES`, so the rendered signature
follows the change; the prose and the new `arg_dict[:bl_axis]` entry describe the selector.

Nothing in ADR 0046 (`forward_prior`) is contradicted.
