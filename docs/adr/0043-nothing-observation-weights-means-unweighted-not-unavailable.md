---
status: accepted
---

# A `nothing` observation weight means *unweighted*, never *unavailable*

## Context

[`DynamicAbstractWeights`](../../src/01_Base/02_TypeRoots.jl) is an extension point: a user subtypes it and
implements `get_observation_weights` for the input shapes they care about — a `VecNum` arity and a
`MatNum` arity, both documented on the abstract type. The library ships **no concrete subtype and no
test** exercising one, so the contract lived entirely in prose.

The resolver's fallback was a single method covering both `nothing` and every dynamic type:

```julia
function get_observation_weights(::Option{<:DynamicAbstractWeights}, args...; kwargs...)
    return nothing
end
```

That one return value carried two incompatible meanings. **No weights were requested** (`w ===
nothing`) is what every consumer reads it as — each one branches `isnothing(w)` and computes the
unweighted result. **Weights could not be resolved** (a dynamic type with no method for the shape it
was handed) took the identical path: the estimator ran unweighted, returning a numerically plausible
number with no error, no warning, and no way for the caller to tell.

Issue [#177](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/177) proposed guarding the
21 call sites in `08_Moments` with an `@argcheck` on `isnothing(w) && !isnothing(ce.w)`, while
explicitly leaving the 25 sites in `19_RiskMeasures` and 24 in `20_Optimisation` alone, on the
grounds that the `nothing` fallback there was "load-bearing by design". **That reading was wrong**,
and the error mattered: it would have patched a third of the instances and left the other two thirds
silently broken.

Nothing in the library ever wants `nothing` back from a `DynamicAbstractWeights`:

- `19_RiskMeasures` dispatches on the weights type, resolves, then **rebuilds itself and
  re-dispatches** — e.g. `LowOrderMoment{<:Any, <:DynamicAbstractWeights, …}` in
  [`03_MomentRiskMeasures.jl`](../../src/19_RiskMeasures/03_MomentRiskMeasures.jl). A `nothing`
  resolution rebuilt the measure as `{…, Nothing, …}` and re-dispatched straight to the *unweighted*
  method. Same defect, one layer deeper.
- `20_Optimisation` resolves then branches, e.g. `wi = get_observation_weights(wi, pr.X)` followed by
  `if isnothing(wi)` in
  [`11_AverageDrawdownConstraints.jl`](../../src/20_Optimisation/20_RiskMeasureConstraints/11_AverageDrawdownConstraints.jl).
  That branch exists for the genuinely-unweighted case; an unresolvable dynamic type landed in it.

What is actually load-bearing is **resolve-before-dispatch** — the property the
[`average_drawdown`](../../src/19_RiskMeasures/11_AverageDrawdown.jl) docstring describes when it
says the aggregator "only ever sees a concrete weight vector or `nothing`". The `nothing` in that
sentence is the requested-unweighted case. #177 read it as sanctioning the unresolvable case too.

Because the defect is a *meaning* problem rather than a call-site problem, enumerating call sites was
also the wrong detection method. `cov(::GeneralCovariance, …)` in
[`03_Covariance.jl`](../../src/08_Moments/03_Covariance.jl) never called `get_observation_weights` at
all — it branched on `isnothing(ce.w)` and passed `ce.w` to `robust_cov` raw, so *any* dynamic type,
even a fully-implemented one, died with a `MethodError` inside `robust_cov` while its sibling `cor`
worked. No survey of `get_observation_weights` callers could have found it.

## Decision

**`get_observation_weights` returns `nothing` only for `w === nothing`, and that means "no weights
were requested". A `DynamicAbstractWeights` that cannot be resolved throws.** The single fallback
splits by type:

```julia
get_observation_weights(::Nothing, args...; kwargs...) = nothing
get_observation_weights(w::DynamicAbstractWeights, args...; kwargs...) =
    throw(ObservationWeightsError(...))   # names the type, the shape, and both signatures
get_observation_weights(w::VecNum, args...; kwargs...) = w
```

Dispatch coverage is unchanged, so **no call site needs editing**. All 70 pre-existing sites across
`08_Moments` (21), `19_RiskMeasures` (25) and `20_Optimisation` (24) become correct at once, and —
the point of locating the check here rather than at the callers — a newly written one cannot
reintroduce the bug.

Three corollaries follow:

1. **Strictness lives in the resolver, never at call sites.** A caller-side
   `isnothing(w) && !isnothing(ce.w)` check is redundant under this decision, and adding one is the
   specific mistake this ADR exists to prevent: it re-encodes the contract once per site, and the
   sites that forget are exactly the ones that silently break.
2. **Every consumer of `ObsWeights` must resolve before dispatching**, so the estimator downstream
   only ever sees a concrete vector or a deliberate `nothing`. `cov(::GeneralCovariance, …)` is
   fixed to do this, matching `cor`.
3. **The error carries the guidance**, since a user hitting it is by definition mid-way through
   implementing an extension point: [`ObservationWeightsError`](../../src/01_Base/07_Errors.jl) (a
   `PortfolioOptimisersError`, following `TimeDependentDefaultError`) names the offending type, the
   shape it was handed, and both method signatures to write.

The `DynamicAbstractWeights` doctest is corrected as part of this. It defined its methods on the
**abstract type** rather than on `MyWeights`, contradicting its own instructions three lines above
and — because those signatures are more specific in their second argument than the new throwing
fallback — capturing every other subtype and defeating the strictness for all 1-D and 2-D input
inside the doc build. It now dispatches on the concrete type and demonstrates the error with a
deliberately partial implementation.

## Consequences

- **User-visible behaviour change.** A partially-implemented `DynamicAbstractWeights` that
  previously returned a silently unweighted result now raises `ObservationWeightsError`. This is the
  intent of the change: the previous result was wrong, not merely undocumented. Nothing in the repo
  breaks — there was no concrete subtype and no test covering one.
- `cov(::GeneralCovariance, …)` with a *complete* dynamic weights type changes from `MethodError` to
  a correctly weighted covariance. This is a bug fix, not a break.
- `DynamicAbstractWeights` gains its first test coverage
  ([`test_08c_observation_weights.jl`](../../test/test_08c_observation_weights.jl)), pinning the
  resolver contract, the error message's content, the shared `moment_window_and_weights` helper, the
  `08_Moments` estimators, and the `19_RiskMeasures` resolve-then-rebuild path.
- Documenting the permissive `nothing` contract, as #177 asked for, is no longer needed: after this
  change there is no permissive contract to document. `nothing` has one meaning.
