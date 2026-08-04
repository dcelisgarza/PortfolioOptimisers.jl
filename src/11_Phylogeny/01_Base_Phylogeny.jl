"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny estimator types.

All concrete and/or abstract types implementing phylogeny-based estimation algorithms should be subtypes of `AbstractPhylogenyEstimator`.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny algorithm types.

All concrete and/or abstract types implementing specific phylogeny algorithms should be subtypes of `AbstractPhylogenyAlgorithm`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny result types.

All concrete and/or abstract types representing the result of a phylogeny estimation should be subtypes of `AbstractPhylogenyResult`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyResult <: AbstractResult end
"""
    const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}

Alias for a phylogeny estimator or result.

Matches either an [`AbstractPhylogenyEstimator`](@ref) or an [`AbstractPhylogenyResult`](@ref). Used internally for dispatch when either a phylogeny estimation configuration or pre-computed result is accepted.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the phylogeny estimator or result `pl` unchanged.

Identity pass-through used when a phylogeny estimator or pre-computed result is provided in a context that calls [`factory`](@ref).

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
  - [`factory`](@ref)
"""
function factory(pl::PlE_Pl, args...; kwargs...)::PlE_Pl
    return pl
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the phylogeny algorithm `alg` unchanged.

Identity pass-through used when a phylogeny algorithm is provided in a context that calls [`factory`](@ref).

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::AbstractPhylogenyAlgorithm, args...;
                 kwargs...)::AbstractPhylogenyAlgorithm
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all separation decay algorithms.

A separation decay turns a **separation** `d >= 0` — how far apart two assets sit in whatever structure the caller is reading — into a score, and is applied by [`separation_decay`](@ref). The family is open: a caller wanting a different fall-off defines a member and a [`separation_decay`](@ref) method for it, exactly as [`AbstractSimilarityMatrixAlgorithm`](@ref) is extended through [`distance_to_similarity`](@ref).

`d` is a **real** separation rather than an integer hop count, so one family serves an unweighted graph — where hop counts enter as integer-valued reals — and any structure whose separation is continuous.

# The contract

  - Defined for every `d >= 0`.
  - `f(0) > 0` and maximal. Self-inclusion is load-bearing rather than cosmetic: a decay that does not put an asset at the top of its own scale silently produces a *structural equivalence* matrix instead of a proximity one — see [`PhylogenyFeatures`](@ref)'s "Why the diagonal includes self".
  - Monotone non-increasing in `d`.
  - Never assumed to reach zero. **Truncation is a separate knob**: the consumer applies its own budget — `n` on [`NetworkEstimator`](@ref) — and the decay only shapes the fall-off inside it. An exponential never reaches zero, so budget and fall-off cannot be the same dial.
  - `f(d) >= 0` for `0 <= d <= dmax`. `0` is the unreachable sentinel, so a negative score *inside* the budget would place a **reachable** pair strictly below an **unreachable** one — an ordering inversion within the producer's own scale. It is not a claim that a signed score is wrong in general: the feature matrix is signed-tolerant by decision, and [`assert_metric_domain`](@ref) checks non-negativity per metric at the consumer rather than blanket. This clause is producer-local, and it is non-negativity rather than strict positivity because a decay that bottoms out at zero says *no relatedness*, which is the same claim an unreachable pair makes.

The clause is **scoped to the budget** because the sign outside it is unobservable — the consumer's `h[u] <= n` test short-circuits before the decay is ever evaluated there — and because the family's own default violates the wider statement: [`LinearDecay`](@ref) crosses zero at `d = dmax + 1` and is negative above it. Binding the clause on all `d >= 0` would need the `max(0, ⋅)` floor the budget knob exists to avoid, a second truncation biting before `n` does.

A zero in the resulting feature matrix therefore means **functionally unreachable**: either the graph is disconnected there, or the decay has fallen to nothing — the same claim about the pair, and nothing downstream can act on the difference. No shipped member emits zero anywhere inside the budget, so for what ships a zero is disconnection and nothing else.

# The budget is an argument, not a field

[`separation_decay`](@ref) takes the budget in scope as its third argument, `dmax`, and members may ignore it — only [`LinearDecay`](@ref) reads it, to set `f(0)`. Keeping it off the member is what makes the two knobs impossible to desync: `n` stays the single source of truth on [`NetworkEstimator`](@ref), rather than being mirrored on an algorithm that cannot see it at construction. [`ExponentialDecay`](@ref) provides the self-versus-neighbour contrast a free top-of-scale would have bought, without the hazard of a second truncation hiding inside the decay.

# Enforcement

The contract is enforced rather than merely documented, by a probing [`assert_separation_decay`](@ref) fallback on this type. The shipped members satisfy it by construction and override that fallback to a no-op, so the check is **opt-out**: an extension that says nothing about itself gets probed.

# Related

  - [`LinearDecay`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`assert_separation_decay`](@ref)
  - [`GradedNeighbourhood`](@ref)
  - [`PhylogenyFeatures`](@ref)
"""
abstract type AbstractSeparationDecayAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay falling off linearly to the edge of the budget.

# Mathematical definition

```math
\\begin{align}
f(d) &= d_{\\mathrm{max}} + 1 - d\\,,
\\end{align}
```

Where:

  - ``d``: Separation between two assets.
  - ``d_{\\mathrm{max}}``: Separation budget in scope.

The default, and the only member that reads the budget. It is the fall-off [`GradedNeighbourhood`](@ref) hardcoded before the family existed, so it reproduces those values exactly: a direct neighbour scores ``d_{\\mathrm{max}}``, the asset itself ``d_{\\mathrm{max}} + 1``.

Because truncation lives with the budget rather than in the decay, no `max(0, ⋅)` floor is needed — on the kept range `0 <= d <= dmax` the expression is strictly positive, bottoming out at `1`.

# Examples

```jldoctest
julia> separation_decay.(Ref(LinearDecay()), 0:3, 3)
4-element Vector{Int64}:
 4
 3
 2
 1
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`GradedNeighbourhood`](@ref)
"""
struct LinearDecay <: AbstractSeparationDecayAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay falling off exponentially.

# Mathematical definition

```math
\\begin{align}
f(d) &= e^{-\\lambda d}\\,,
\\end{align}
```

Where:

  - ``d``: Separation between two assets.
  - ``\\lambda``: Rate of the fall-off, `rate`.

Pins `f(0) = 1` and lets `rate` set the self-versus-neighbour contrast independently of the budget, which is what a caller wanting relatedness to drop *sharply* needs — the budget only says how far to look.

Parameterised by rate rather than by per-step retention. ``\\rho^d`` and ``e^{-\\lambda d}`` are the same function (``\\lambda = -\\log\\rho``), but "retention per step" is a statement about integers, and the family's argument is a real separation. The rate form also matches [`ExponentialSimilarity`](@ref) and [`ExpGerberIQDecay`](@ref), and needs only a one-sided bound to stay monotone.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExponentialDecay(;
        rate::Number = 1.0
    ) -> ExponentialDecay

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:sdrate])

# Examples

```jldoctest
julia> ExponentialDecay()
ExponentialDecay
  rate ┴ Float64: 1.0

julia> separation_decay.(Ref(ExponentialDecay()), 0:3, 3)
4-element Vector{Float64}:
 1.0
 0.36787944117144233
 0.1353352832366127
 0.049787068367863944
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`separation_decay`](@ref)
  - [`ExponentialSimilarity`](@ref)
"""
@concrete struct ExponentialDecay <: AbstractSeparationDecayAlgorithm
    """
    $(field_dict[:sdrate])
    """
    rate
    function ExponentialDecay(rate::Number)
        @argcheck(zero(rate) < rate,
                  DomainError(rate, "the rate of an ExponentialDecay must be > 0"))
        return new{typeof(rate)}(rate)
    end
end
function ExponentialDecay(; rate::Number = 1.0)::ExponentialDecay
    return ExponentialDecay(rate)
end
"""
$(DocStringExtensions.TYPEDEF)

Separation decay falling off as a power of the separation.

# Mathematical definition

```math
\\begin{align}
f(d) &= \\left(1 + d\\right)^{-p}\\,,
\\end{align}
```

Where:

  - ``d``: Separation between two assets.
  - ``p``: Exponent of the fall-off, `power`.

The middle ground between [`LinearDecay`](@ref) and [`ExponentialDecay`](@ref): heavier-tailed than the exponential, so distant assets keep a small but non-negligible score.

The `1 +` is what makes classical inverse-distance weighting finite at ``d = 0``; it also pins `f(0) = 1`, matching [`ExponentialDecay`](@ref)'s scale for free. The alternative spelling ``(1 + d^p)^{-1}`` is *not* used: it pins ``f(1) = 1/2`` for every `p` and is non-monotone in `p` at fixed `d`, so raising the exponent would score near neighbours *lower* and far ones higher.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReciprocalDecay(;
        power::Number = 1.0
    ) -> ReciprocalDecay

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:sdpower])

# Examples

```jldoctest
julia> ReciprocalDecay()
ReciprocalDecay
  power ┴ Float64: 1.0

julia> separation_decay.(Ref(ReciprocalDecay()), 0:3, 3)
4-element Vector{Float64}:
 1.0
 0.5
 0.3333333333333333
 0.25
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`separation_decay`](@ref)
"""
@concrete struct ReciprocalDecay <: AbstractSeparationDecayAlgorithm
    """
    $(field_dict[:sdpower])
    """
    power
    function ReciprocalDecay(power::Number)
        @argcheck(zero(power) < power,
                  DomainError(power, "the power of a ReciprocalDecay must be > 0"))
        return new{typeof(power)}(power)
    end
end
function ReciprocalDecay(; power::Number = 1.0)::ReciprocalDecay
    return ReciprocalDecay(power)
end
"""
    separation_decay(dk::LinearDecay, d::Number, dmax::Number)
    separation_decay(dk::ExponentialDecay, d::Number, dmax::Number)
    separation_decay(dk::ReciprocalDecay, d::Number, dmax::Number)

Score a separation under a decay algorithm.

The whole extension contract of [`AbstractSeparationDecayAlgorithm`](@ref): a new member is a struct and one method of this function.

# Arguments

  - `dk`: Separation decay algorithm.
  - `d`: Separation between two assets, `d >= 0`. Real rather than integral, so a weighted path length is as admissible as a hop count.
  - `dmax`: Separation budget in scope. **Inert for members that do not need it** — only [`LinearDecay`](@ref) reads it. Inert arguments have precedent here: [`phylogeny_features`](@ref) ignores its `alg` entirely for a partition source.

# Returns

  - `f::Number`: Score for the separation. Non-negative for `0 <= d <= dmax`; **above** the budget the sign is unconstrained and [`LinearDecay`](@ref) does go negative, which is harmless because the consumer's budget test short-circuits before the call.

# Validation

The contract is not checked here — this runs inside an `assets × assets` loop. Callers probe once up front with [`assert_separation_decay`](@ref) instead.

# Examples

```jldoctest
julia> separation_decay(LinearDecay(), 2, 3)
2

julia> separation_decay(ExponentialDecay(; rate = 2.0), 2, 3)
0.01831563888873418

julia> separation_decay(ReciprocalDecay(; power = 2.0), 2, 3)
0.1111111111111111
```

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`ExponentialDecay`](@ref)
  - [`ReciprocalDecay`](@ref)
  - [`assert_separation_decay`](@ref)
  - [`GradedNeighbourhood`](@ref)
"""
function separation_decay end
function separation_decay(::LinearDecay, d::Number, dmax::Number)::Number
    return dmax + one(dmax) - d
end
function separation_decay(dk::ExponentialDecay, d::Number, ::Number)::Number
    return exp(-dk.rate * d)
end
function separation_decay(dk::ReciprocalDecay, d::Number, ::Number)::Number
    return inv((one(d) + d)^dk.power)
end
"""
    assert_separation_decay(dk::AbstractSeparationDecayAlgorithm, ds, dmax::Number)
    assert_separation_decay(dk::Union{<:LinearDecay, <:ExponentialDecay, <:ReciprocalDecay},
                            ds, dmax::Number)

Check that a separation decay honours its contract over the separations it will be asked about.

The fallback **probes**: it evaluates [`separation_decay`](@ref) over `ds` and checks the result is finite, that `f(0)` is strictly positive and maximal, that the values are monotone non-increasing, and that none of them is negative. The three shipped members satisfy the contract by construction and override this to a no-op, so the probe costs nothing for what ships and is fail-safe for extensions.

Probing is cheap where it is used because `ds` is small and the loop it guards is not: [`GradedNeighbourhood`](@ref) passes `0:n`, which is *exhaustive* for a hop count — every separation the `assets × assets` loop can ever ask about, in `n + 1` evaluations.

Non-negativity gets **one extra evaluation at `d = dmax`**, whether or not `dmax` appears in `ds`, mirroring the out-of-loop evaluation of `f(0)`. That endpoint is what closes the clause over a *continuum*: monotonicity is already promised, so `f(dmax) >= 0` implies `f(d) >= 0` for every `d` in `[0, dmax]`, and a `ds` that can only ever be a sample — as it must be once separations are weighted path lengths — costs this clause nothing. Monotonicity itself gains nothing from the endpoint and remains genuinely sampled.

# Arguments

  - `dk`: Separation decay algorithm.
  - `ds`: Separations to probe. Need not be sorted. Precondition: `ds ⊆ [0, dmax]` — `ds` is what the guarded loop will ask about, and the loop never asks outside the budget.
  - `dmax`: Separation budget in scope, forwarded to [`separation_decay`](@ref) and probed as an endpoint in its own right.

# Returns

  - `nothing`.

# Validation

  - Every probed value is finite.
  - `f(0) > 0`.
  - `f(0) >= f(d)` for every probed `d`.
  - The probed values are monotone non-increasing in `d`.
  - `f(d) >= 0` for every probed `d`, and at `d = dmax` whether or not it was probed.

# Related

  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`separation_decay`](@ref)
  - [`GradedNeighbourhood`](@ref)
"""
function assert_separation_decay(dk::AbstractSeparationDecayAlgorithm, ds,
                                 dmax::Number)::Nothing
    f0 = separation_decay(dk, zero(dmax), dmax)
    @argcheck(isfinite(f0) && f0 > zero(f0),
              DomainError(f0,
                          "a separation decay must score d = 0 finite and strictly positive, got $(f0) from $(typeof(dk))"))
    ds = issorted(ds) ? ds : sort!(collect(ds))
    fp = f0
    for d in ds
        f = separation_decay(dk, d, dmax)
        @argcheck(isfinite(f) && f <= f0,
                  DomainError(f,
                              "a separation decay must be finite and maximal at d = 0, got $(f) at d = $(d) against $(f0) at d = 0 from $(typeof(dk))"))
        @argcheck(f <= fp,
                  DomainError(f,
                              "a separation decay must be monotone non-increasing, got $(f) at d = $(d) from $(typeof(dk))"))
        @argcheck(f >= zero(f),
                  DomainError(f,
                              "a separation decay must be non-negative inside the budget, got $(f) at d = $(d) against a budget of dmax = $(dmax) from $(typeof(dk))"))
        fp = f
    end
    # `ds` need not contain `dmax`, and a real separation makes an exhaustive `ds` impossible.
    # Monotonicity is already promised, so this one endpoint closes non-negativity over the
    # whole of `0 <= d <= dmax` -- the only clause whose domain closes from a single point.
    fmax = separation_decay(dk, dmax, dmax)
    @argcheck(fmax >= zero(fmax),
              DomainError(fmax,
                          "a separation decay must be non-negative inside the budget, got $(fmax) at d = $(dmax) against a budget of dmax = $(dmax) from $(typeof(dk))"))
    return nothing
end
# The shipped members satisfy the contract by construction, so the probe is turned off for
# them and left on for anything an extension defines -- opt-out, not opt-in.
function assert_separation_decay(::Union{<:LinearDecay, <:ExponentialDecay,
                                         <:ReciprocalDecay}, ::Any, ::Number)::Nothing
    return nothing
end

export AbstractSeparationDecayAlgorithm, LinearDecay, ExponentialDecay, ReciprocalDecay,
       separation_decay
