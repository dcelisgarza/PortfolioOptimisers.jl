"""
$(DocStringExtensions.TYPEDEF)

Buy and hold: the allocation drifts with the market and is never traded, the uniform buy-and-hold benchmark of every online selection paper (UBAH).

The update is the Price-Adjusted Allocation itself, `w .* x / ⟨w, x⟩`, projected onto the Allocation Set; on the simplex the projection is the identity, so buy-and-hold has zero turnover by construction. As the weighting of an [`ExpertMixture`](@ref) it is the wealth-weighted mixture `p ∝ p .* r` every paper writes as `BAH_W`, and the mixture's default.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BuyAndHold(; proj::EuclideanProjection = EuclideanProjection()) -> BuyAndHold

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BuyAndHold()
BuyAndHold
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`ExpertMixture`](@ref)
  - [`price_adjusted_allocation`](@ref)

# References

  - $(ref_dict[:lihoi2014])
"""
struct BuyAndHold{T1 <: EuclideanProjection} <: AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:proj])
    """
    proj::T1
    function BuyAndHold(proj::EuclideanProjection)
        return new{typeof(proj)}(proj)
    end
end
function BuyAndHold(; proj::EuclideanProjection = EuclideanProjection())::BuyAndHold
    return BuyAndHold(proj)
end
function online_update!(alg::BuyAndHold, st, w::AbstractVector, x::AbstractVector, ::Any,
                        set::AbstractAllocationSet)
    wh = price_adjusted_allocation(w, x)
    return st, project(alg.proj, set, wh, wh)
end
"""
$(DocStringExtensions.TYPEDEF)

The constant rebalanced portfolio: the same allocation `w` every period, rebalanced back to it after every price move (CRP; UCRP when `w` is uniform, the default).

It reads nothing of the market, so a given Start Allocation is held for one period and replaced by `w` at the first update. Sampled from the simplex, constant rebalanced portfolios are the experts of [`UniversalPortfolio`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConstantRebalancedPortfolio(;
        w::Option{<:AbstractVector} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ConstantRebalancedPortfolio

Keywords correspond to the struct's fields. A `nothing` `w` is the uniform portfolio over the pinned universe.

## Validation

  - `w`: non-empty, and every element is non-negative and finite, when it is given.
  - `sum(w) ≈ 1` when `w` is given, because every Allocation Set has a budget of one. Raises a `DomainError`.

## View parameters

When [`port_opt_view`](@ref) is called on this type, `w` is sliced to the selected indices and renormalised.

# Examples

```jldoctest
julia> ConstantRebalancedPortfolio()
ConstantRebalancedPortfolio
     w ┼ nothing
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`UniversalPortfolio`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)

# References

  - $(ref_dict[:cover1991])
"""
struct ConstantRebalancedPortfolio{T1 <: Option{<:AbstractVector},
                                   T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The allocation rebalanced to every period, over the pinned universe, or `nothing` for the uniform one.
    """
    w::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function ConstantRebalancedPortfolio(w::Option{<:AbstractVector},
                                         proj::EuclideanProjection)
        if !isnothing(w)
            assert_nonempty_nonneg_finite_val(w, :w)
            s = sum(w)
            @argcheck(isapprox(s, one(s)),
                      DomainError(s,
                                  "every Allocation Set has a budget of one, so the constant rebalanced portfolio's `w` must sum to one, got sum(w) = $s"))
        end
        return new{typeof(w), typeof(proj)}(w, proj)
    end
end
function ConstantRebalancedPortfolio(; w::Option{<:AbstractVector} = nothing,
                                     proj::EuclideanProjection = EuclideanProjection())::ConstantRebalancedPortfolio
    return ConstantRebalancedPortfolio(w, proj)
end
function port_opt_view(alg::ConstantRebalancedPortfolio, i, args...)
    return ConstantRebalancedPortfolio(; w = renormalised_view(alg.w, i), proj = alg.proj)
end
function online_update!(alg::ConstantRebalancedPortfolio, st, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    q = isnothing(alg.w) ? fill(one(eltype(w)) / length(w), length(w)) : alg.w
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

The carrier of [`NewtonStep`](@ref): the Gram matrix of the gradients and their weighted sum.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`NewtonStep`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct NewtonStepState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The Gram matrix ``A_t = I + \\sum_{s \\leq t} g_s g_s^\\intercal``, `assets × assets`, written in place.
    """
    A
    """
    The weighted gradient sum ``b_t = (1 + 1/\\beta) \\sum_{s \\leq t} g_s``, `assets × 1`, written in place.
    """
    b
end
function merge_states(::NewtonStepState, ::NewtonStepState)
    return throw(ArgumentError("a `NewtonStepState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::NewtonStepState)
    return NewtonStepState(x.n, copy(x.A), copy(x.b))
end
function port_opt_view(x::NewtonStepState, i, args...)
    return NewtonStepState(x.n, x.A[i, i], x.b[i])
end
"""
$(DocStringExtensions.TYPEDEF)

The online Newton step of Agarwal, Hazan, Kale and Schapire (2006): a second-order method on the exp-concave log-wealth objective, with logarithmic regret (ONS).

# Mathematical definition

With the gradient ``\\boldsymbol{g}_t = \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``,

```math
\\begin{align}
A_t &= I + \\sum_{s \\leq t} \\boldsymbol{g}_s \\boldsymbol{g}_s^\\intercal\\,,\\quad
\\boldsymbol{b}_t = \\left(1 + \\frac{1}{\\beta}\\right) \\sum_{s \\leq t} \\boldsymbol{g}_s\\,,\\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}\\left( \\delta A_t^{-1} \\boldsymbol{b}_t \\right)\\,,
\\end{align}
```

with, when `eta > 0`, the shrinkage ``(1 - \\eta) \\boldsymbol{q} + \\eta \\boldsymbol{1} / N`` of the raw Newton point towards the uniform portfolio **before** the projection, so that a bound of zero or a negative lower bound on the Allocation Set is honoured — mixed after the projection, the uniform mass would leave the set. The paper's projection is in the norm of ``A_t``, the [`GramProjection`](@ref), a programme on every set that carries its own solver; the default [`EuclideanProjection`](@ref) is the standard simplification that keeps the rule free of a solver, and on the simplex at `eta = 0` the two agree wherever the Newton point already lies in it. The rule binds its current ``A_t`` onto the geometry through [`gram_geometry`](@ref) at every step; the Start Allocation is projected before any gradient at ``A_0 = I``, in the Euclidean geometry. The regret is ``O(N \\log T)`` at ``O(N^2)`` a step under the paper's projection, which is why this rather than the universal portfolio is the practical second-order choice; the bound is stated for the [`GramProjection`](@ref) and is not claimed under the Euclidean default. As the weighting of an [`ExpertMixture`](@ref) it is the online Newton update over the expert-return vector, with a `K × K` Gram.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NewtonStep(;
        beta::Real = 1,
        delta::Real = 0.125,
        eta::Real = 0,
        proj::Union{<:EuclideanProjection, <:GramProjection} = EuclideanProjection()
    ) -> NewtonStep

Keywords correspond to the struct's fields. The `proj` slot is bound to the two geometries the rule's theorem and its standard simplification cover.

## Validation

  - `beta > 0`, `delta > 0`. A `DomainError` is thrown otherwise.
  - `0 <= eta < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> NewtonStep()
NewtonStep
   beta ┼ Int64: 1
  delta ┼ Float64: 0.125
    eta ┼ Int64: 0
   proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`NewtonStepState`](@ref)
  - [`GramProjection`](@ref)

# References

  - $(ref_dict[:agarwal2006])
"""
struct NewtonStep{T1 <: Real, T2 <: Real, T3 <: Real,
                  T4 <: Union{<:EuclideanProjection, <:GramProjection}} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The paper's ``\\beta``, which scales the gradient sum.
    """
    beta::T1
    """
    The paper's ``\\delta``, which scales the Newton direction.
    """
    delta::T2
    """
    Shrinkage of the raw Newton point towards the uniform portfolio before the projection, in `[0, 1)`; `0` is the paper's rule.
    """
    eta::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function NewtonStep(beta::T1, delta::T2, eta::T3,
                        proj::T4) where {T1 <: Real, T2 <: Real, T3 <: Real,
                                         T4 <:
                                         Union{<:EuclideanProjection, <:GramProjection}}
        @argcheck(beta > zero(beta), DomainError(beta, "beta must be positive"))
        @argcheck(delta > zero(delta), DomainError(delta, "delta must be positive"))
        @argcheck(zero(eta) <= eta < one(eta), DomainError(eta, "eta must be in [0, 1)"))
        return new{T1, T2, T3, T4}(beta, delta, eta, proj)
    end
end
function NewtonStep(; beta::Real = 1, delta::Real = 0.125, eta::Real = 0,
                    proj::Union{<:EuclideanProjection, <:GramProjection} = EuclideanProjection())::NewtonStep
    return NewtonStep(beta, delta, eta, proj)
end
function projection_geometry(::NewtonStep)
    # The Start Allocation is projected at `A_0 = I`, where the Gram norm is the Euclidean.
    return EuclideanProjection()
end
function rule_state_seed(::NewtonStep, w::AbstractVector)
    N = length(w)
    return NewtonStepState(0, Matrix{eltype(w)}(LinearAlgebra.I, N, N), zeros(eltype(w), N))
end
function online_update!(alg::NewtonStep, st::NewtonStepState, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    g = x ./ LinearAlgebra.dot(w, x)
    st.A .+= g * transpose(g)
    st.b .+= (1 + inv(alg.beta)) .* g
    q = alg.delta .* (LinearAlgebra.Symmetric(st.A) \ st.b)
    if !iszero(alg.eta)
        q .= (1 - alg.eta) .* q .+ alg.eta / length(q)
    end
    wn = project(gram_geometry(alg.proj, st.A), set, q, price_adjusted_allocation(w, x))
    return NewtonStepState(st.n + 1, st.A, st.b), wn
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the step-size rules of [`PassiveAggressiveMeanReversion`](@ref): the paper's three variants, which differ in one formula.

# Interfaces

In order to implement a new step-size rule, subtype `AbstractPassiveAggressiveSlack` and implement:

  - `passive_aggressive_step(slack::AbstractPassiveAggressiveSlack, loss::Real, denom::Real) -> Real`: The step length ``\\tau_t`` from the hinge loss ``\\max(0, \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle - \\epsilon)`` and the squared norm of the centred price relative.

## Arguments

  - `slack`: The rule.
  - `loss`: The hinge loss of the period.
  - `denom`: ``\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2``, positive.

## Returns

  - `tau::Real`: The step length.

# Related

  - [`NoSlack`](@ref)
  - [`LinearSlack`](@ref)
  - [`QuadraticSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)
"""
abstract type AbstractPassiveAggressiveSlack <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The step-size rule of PAMR: ``\\tau_t = \\ell_t / \\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2``, no slack.

# Examples

```jldoctest
julia> NoSlack()
NoSlack()
```

# Related

  - [`AbstractPassiveAggressiveSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)
"""
struct NoSlack <: AbstractPassiveAggressiveSlack end
"""
$(DocStringExtensions.TYPEDEF)

The step-size rule of PAMR-1: the step of [`NoSlack`](@ref) capped at the aggressiveness `C`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearSlack(; C::Real = 500) -> LinearSlack

## Validation

  - `C > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> LinearSlack()
LinearSlack
  C ┴ Int64: 500
```

# Related

  - [`AbstractPassiveAggressiveSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)
"""
struct LinearSlack{T1 <: Real} <: AbstractPassiveAggressiveSlack
    """
    Aggressiveness, the cap on the step.
    """
    C::T1
    function LinearSlack(C::Real)
        @argcheck(C > zero(C), DomainError(C, "C must be positive"))
        return new{typeof(C)}(C)
    end
end
function LinearSlack(; C::Real = 500)::LinearSlack
    return LinearSlack(C)
end
"""
$(DocStringExtensions.TYPEDEF)

The step-size rule of PAMR-2: ``\\tau_t = \\ell_t / (\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2 + 1 / (2C))``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    QuadraticSlack(; C::Real = 500) -> QuadraticSlack

## Validation

  - `C > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> QuadraticSlack()
QuadraticSlack
  C ┴ Int64: 500
```

# Related

  - [`AbstractPassiveAggressiveSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)
"""
struct QuadraticSlack{T1 <: Real} <: AbstractPassiveAggressiveSlack
    """
    Aggressiveness, which softens the denominator.
    """
    C::T1
    function QuadraticSlack(C::Real)
        @argcheck(C > zero(C), DomainError(C, "C must be positive"))
        return new{typeof(C)}(C)
    end
end
function QuadraticSlack(; C::Real = 500)::QuadraticSlack
    return QuadraticSlack(C)
end
"""
    passive_aggressive_step(::NoSlack, loss::Real, denom::Real)
    passive_aggressive_step(slack::LinearSlack, loss::Real, denom::Real)
    passive_aggressive_step(slack::QuadraticSlack, loss::Real, denom::Real)

The step length of a passive-aggressive update under each slack rule.

# Related

  - [`AbstractPassiveAggressiveSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)
"""
function passive_aggressive_step(::NoSlack, loss::Real, denom::Real)
    return loss / denom
end
function passive_aggressive_step(slack::LinearSlack, loss::Real, denom::Real)
    return min(slack.C, loss / denom)
end
function passive_aggressive_step(slack::QuadraticSlack, loss::Real, denom::Real)
    return loss / (denom + inv(2 * slack.C))
end
"""
$(DocStringExtensions.TYPEDEF)

The passive aggressive mean reversion of Li, Zhao, Hoi and Gopalkrishnan (2012): the closest allocation to the current one whose return on the last realised price relative is at most `eps` (PAMR).

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}\\left( \\boldsymbol{w}_t - \\tau_t \\left( \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\right) \\right)\\,,\\quad
\\ell_t = \\max\\left(0, \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle - \\epsilon\\right)\\,,
\\end{align}
```

with ``\\tau_t`` the step of the `slack` rule and ``\\bar{x}_t`` the mean of ``\\boldsymbol{x}_t``; the step is zero when every asset moved alike. The rule sells what just rose: it is a total bet on single-period mean reversion, and its constraint runs the other way from [`ForecastReversion`](@ref)'s, on the realised relative with the paper's ``\\epsilon`` in ``[0, 1]``; a threshold above one is admitted and makes the rule act only on a gain above it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PassiveAggressiveMeanReversion(;
        eps::Real = 0.5,
        slack::AbstractPassiveAggressiveSlack = NoSlack(),
        proj::EuclideanProjection = EuclideanProjection()
    ) -> PassiveAggressiveMeanReversion

Keywords correspond to the struct's fields. `NoSlack()`, `LinearSlack(; C)` and `QuadraticSlack(; C)` are the paper's PAMR, PAMR-1 and PAMR-2.

## Validation

  - `eps >= 0`. A `DomainError` is thrown otherwise. The paper states the threshold in `[0, 1]`.

# Examples

```jldoctest
julia> PassiveAggressiveMeanReversion()
PassiveAggressiveMeanReversion
    eps ┼ Float64: 0.5
  slack ┼ NoSlack()
   proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`AbstractPassiveAggressiveSlack`](@ref)
  - [`ForecastReversion`](@ref)

# References

  - $(ref_dict[:li2012pamr])
"""
struct PassiveAggressiveMeanReversion{T1 <: Real, T2 <: AbstractPassiveAggressiveSlack,
                                      T3 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Reversion threshold: the rule acts only when the last period's portfolio return exceeded it.
    """
    eps::T1
    """
    The step-size rule, one of the paper's three variants.
    """
    slack::T2
    """
    $(field_dict[:proj])
    """
    proj::T3
    function PassiveAggressiveMeanReversion(eps::Real,
                                            slack::AbstractPassiveAggressiveSlack,
                                            proj::EuclideanProjection)
        @argcheck(eps >= zero(eps), DomainError(eps, "eps must be non-negative"))
        return new{typeof(eps), typeof(slack), typeof(proj)}(eps, slack, proj)
    end
end
function PassiveAggressiveMeanReversion(; eps::Real = 0.5,
                                        slack::AbstractPassiveAggressiveSlack = NoSlack(),
                                        proj::EuclideanProjection = EuclideanProjection())::PassiveAggressiveMeanReversion
    return PassiveAggressiveMeanReversion(eps, slack, proj)
end
function online_update!(alg::PassiveAggressiveMeanReversion, st, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    dev = x .- Statistics.mean(x)
    denom = sum(abs2, dev)
    loss = max(zero(eltype(w)), LinearAlgebra.dot(w, x) - alg.eps)
    tau = iszero(denom) ? zero(loss) : passive_aggressive_step(alg.slack, loss, denom)
    q = w .- tau .* dev
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

The carrier of [`ExpertMixture`](@ref): every expert's Rule State, and the Rule State of the weighting over the experts.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ExpertMixture`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct ExpertMixtureState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    Each expert's carrier, one entry per expert, `nothing` where the expert carries nothing.
    """
    st
    """
    Each expert's allocation held during the current period, one vector per expert.
    """
    h
    """
    The weighting rule's carrier, or `nothing`.
    """
    pst
    """
    The weight vector over the experts held during the current period, `experts × 1`.
    """
    p
end
function merge_states(::ExpertMixtureState, ::ExpertMixtureState)
    return throw(ArgumentError("an `ExpertMixtureState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::ExpertMixtureState)
    return ExpertMixtureState(x.n, copy_column.(x.st), copy.(x.h), copy_column(x.pst),
                              copy(x.p))
end
function port_opt_view(x::ExpertMixtureState, i, args...)
    return ExpertMixtureState(x.n, [rule_state_view(s, i, args...) for s in x.st],
                              [renormalised_view(h, i) for h in x.h], copy_column(x.pst),
                              copy(x.p))
end
"""
$(DocStringExtensions.TYPEDEF)

The Gradient Point of an [`ExpertMixture`](@ref) under which every expert evaluates its gradient at its own iterate: the default, and the mixture as every paper before the dynamic-regret ones states it.

# Examples

```jldoctest
julia> OwnPoint()
OwnPoint()
```

# Related

  - [`ExpertMixture`](@ref)
  - [`BlendPoint`](@ref)
"""
struct OwnPoint <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The Gradient Point of an [`ExpertMixture`](@ref) under which every first-order expert evaluates its gradient at the mixture's played blend while stepping from its own iterate: the shared gradient of Zhang, Lu and Zhou (2018, Algorithm 4) and Zhao, Zhang, Zhang and Zhou (2020, Algorithm 2), on which their dynamic-regret bounds are stated, and the point [`Ader`](@ref) and [`Sword`](@ref) construct the mixture on.

The mixture hands the blend through the seven-argument [`online_update!`](@ref); a rule with no gradient takes its six-argument update and ignores the point. The first-order rules that read it are [`MirrorDescent`](@ref), [`OptimisticStep`](@ref), whose hint stays the predictor's, and [`AdaptiveSubgradient`](@ref).

# Examples

```jldoctest
julia> BlendPoint()
BlendPoint()
```

# Related

  - [`ExpertMixture`](@ref)
  - [`OwnPoint`](@ref)
  - [`MirrorDescent`](@ref)
  - [`OptimisticStep`](@ref)
  - [`AdaptiveSubgradient`](@ref)

# References

  - $(ref_dict[:zhang2018ader])
  - $(ref_dict[:zhao2020sword])
"""
struct BlendPoint <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The rule over rules: every expert takes its own Online Update, and a weighting that is itself an Online Selection Rule moves the weight vector over the experts on their period returns; the answer is the weighted mix of the experts' next allocations.

# Mathematical definition

With ``\\boldsymbol{h}_k(t)`` expert ``k``'s allocation held during period ``t`` and ``\\boldsymbol{p}_t`` the weight over the experts,

```math
\\begin{align}
r_{t, k} &= \\langle \\boldsymbol{h}_k(t), \\boldsymbol{x}_t \\rangle\\,,\\quad
\\boldsymbol{p}_{t+1} = \\mathrm{alg}(\\boldsymbol{p}_t, \\boldsymbol{r}_t)\\,,\\quad
\\boldsymbol{w}_{t+1} = \\sum_k p_{t+1, k}\\, \\boldsymbol{h}_k(t+1)\\,.
\\end{align}
```

Under [`BuyAndHold`](@ref), the default, ``\\boldsymbol{p}_{t+1} \\propto \\boldsymbol{p}_t \\odot \\boldsymbol{r}_t`` is the wealth-weighted mixture every paper writes as `BAH_W`, and the mixture over sampled constant rebalanced portfolios is Cover's universal portfolio ([`UniversalPortfolio`](@ref)). [`ExponentiatedGradient`](@ref) and [`NewtonStep`](@ref) on the slot are the online gradient and online Newton updates over the expert-return vector; the Newton weighting over `K` experts carries a `K × K` Gram, so two thousand sampled experts cost a `2000 × 2000` solve a row.

The weighting's step is projected onto the **Expert Set** on `eset`, the Allocation Set over the `K` experts, in the weighting's own Projection Geometry: `nothing`, the default, is the bare `K`-simplex, a no-op for the multiplicative weightings and the Euclidean scalar root for a Newton weighting; a given [`BoundedAllocationSet`](@ref) broadcasts a scalar bound over the experts and takes one entry per expert from a vector bound, so a cap on `eset` caps the trust in any one expert. The blend ``\\sum_k p_{t+1, k} \\boldsymbol{h}_k(t+1)`` is then projected onto the head's Allocation Set once more, in the mixture's own Euclidean geometry on `proj`, with the mixture's Price-Adjusted Allocation as the reference; on a [`BoundedAllocationSet`](@ref) a blend of bounded allocations is bounded and the projection would be the identity, so [`blend_projection`](@ref) skips it by dispatch and the default configuration solves nothing; on a [`ProgrammeAllocationSet`](@ref) it is the repair a turnover ceiling under a weighting other than buy-and-hold, or a MIP kind, needs, and the mixture pays `K + 1` programmes per period, its experts' and its own, beside the `K × K` Gram of a Newton weighting. The mixture reads nothing of a given Start Allocation: the head holds it for one period and it is replaced by the experts' mix. Each expert's own Rule State starts where its rule starts — a constant rebalanced portfolio at its own `w`, a rule that reads `w` at the Start Allocation — projected once onto the head's Allocation Set in the expert's geometry, as the head's `w0` is, so a sampled expert's wealth is Cover's ``S_t(\\boldsymbol{b})`` of its feasible allocation from the first row ([`project_start`](@ref)). The weight vector over the experts starts at `p0`, the mixture's own Start Allocation over them, uniform by default, and the start, given or uniform, is projected once onto the Expert Set in the weighting's geometry at the seed, as the head's `w0` is onto the Allocation Set, so a start outside the set — a given one, or the uniform one under a bound it violates — is made feasible and never refused.

`grad` is the **Gradient Point**: under [`OwnPoint`](@ref), the default, every expert reads its gradient at its own iterate; under [`BlendPoint`](@ref) every first-order expert reads it at the mixture's played blend ``\\boldsymbol{w}_t`` while stepping from its own iterate, the shared gradient ``\\nabla f_t(\\boldsymbol{w}_t)`` of Zhang, Lu and Zhou (2018) and Zhao, Zhang, Zhang and Zhou (2020), and a rule with no gradient ignores the point. Under the blend point and [`ExponentiatedGradient`](@ref) as the weighting the mixture's weight update is the exponentially weighted forecaster on the linearised loss ``\\langle \\nabla f_t(\\boldsymbol{w}_t), \\boldsymbol{h}_k(t) \\rangle`` exactly, wherever the second projection is the identity, because ``\\langle \\boldsymbol{p}_t, \\boldsymbol{r}_t \\rangle = \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` there: [`Ader`](@ref) and [`Sword`](@ref) construct that mixture over a geometric grid of first-order experts.

The mixture's regret against its best expert is exact for the shipped object wherever the second projection is the identity: ``\\log S_T(\\text{best expert}) - \\log S_T(\\text{mixture}) \\leq -\\log p_{1, k^\\star}`` for every sequence under the wealth weighting on the bare Expert Set, which is ``\\log K`` at the uniform start, because the mixture's wealth is the `p_1`-weighted average of the experts' wealths and puts at least ``p_{1, k^\\star}`` of its mass on the best expert ``k^\\star``. Where the Expert Set binds or the second projection repairs, the bound is not claimed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpertMixture(;
        experts::AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
        alg::AbstractOnlinePortfolioSelectionAlgorithm = BuyAndHold(),
        eset::Option{<:BoundedAllocationSet} = nothing,
        proj::EuclideanProjection = EuclideanProjection(),
        grad::Union{OwnPoint, BlendPoint} = OwnPoint(),
        p0::Option{<:AbstractVector} = nothing
    ) -> ExpertMixture

Keywords correspond to the struct's fields. A `nothing` `p0` starts the weighting uniform over the experts, and a `nothing` `eset` is the bare simplex over them.

## Validation

  - `experts` is non-empty. An `IsEmptyError` is thrown otherwise.
  - `rows_needed(alg) == 0`: the weighting is applied to the expert-return vector, for which no rows are held. An `ArgumentError` is thrown otherwise.
  - `p0`: non-empty, finite and of length `length(experts)`, when given; the start, given or uniform, is projected onto the Expert Set at the seed, so it need not lie in it.
  - `eset`, when given, holds a [`WeightBounds`](@ref) and no `sets`: an expert has no name for a [`WeightBoundsEstimator`](@ref) to resolve over. An `ArgumentError` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, every expert is viewed and the weighting, `eset`, `proj`, `grad` and `p0` are carried unchanged.

# Examples

```jldoctest
julia> ExpertMixture(; experts = [ExponentiatedGradient(), NewtonStep()])
ExpertMixture
  experts ┼ 2-element Vector{PortfolioOptimisers.AbstractOnlinePortfolioSelectionAlgorithm}
          │ MirrorDescent ⋯
          │ NewtonStep ⋯
      alg ┼ BuyAndHold
          │   proj ┴ EuclideanProjection()
     eset ┼ nothing
     proj ┼ EuclideanProjection()
     grad ┼ OwnPoint()
       p0 ┴ nothing
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`UniversalPortfolio`](@ref)
  - [`ExpertMixtureState`](@ref)
  - [`BuyAndHold`](@ref)
  - [`OwnPoint`](@ref)
  - [`BlendPoint`](@ref)
  - [`Ader`](@ref)
  - [`Sword`](@ref)

# References

  - $(ref_dict[:lihoi2014])
  - $(ref_dict[:zhang2018ader])
  - $(ref_dict[:zhao2020sword])
"""
struct ExpertMixture{T1 <: AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
                     T2 <: AbstractOnlinePortfolioSelectionAlgorithm,
                     T3 <: Option{<:BoundedAllocationSet}, T4 <: EuclideanProjection,
                     T5 <: Union{OwnPoint, BlendPoint}, T6 <: Option{<:AbstractVector}} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The expert rules, one Rule State each.
    """
    experts::T1
    """
    The weighting: a rule of the family applied to the expert-return vector in place of the price relative.
    """
    alg::T2
    """
    The Expert Set the weighting projects onto, over the experts, or `nothing` for the bare simplex over them.
    """
    eset::T3
    """
    The geometry the blend is projected onto the head's Allocation Set in, once more.
    """
    proj::T4
    """
    The Gradient Point: where every first-order expert evaluates its gradient.
    """
    grad::T5
    """
    The Start Allocation over the experts, projected onto the Expert Set at the seed, or `nothing` for uniform.
    """
    p0::T6
    function ExpertMixture(experts::AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
                           alg::AbstractOnlinePortfolioSelectionAlgorithm,
                           eset::Option{<:BoundedAllocationSet}, proj::EuclideanProjection,
                           grad::Union{OwnPoint, BlendPoint}, p0::Option{<:AbstractVector})
        @argcheck(!isempty(experts), IsEmptyError("experts cannot be empty"))
        @argcheck(rows_needed(alg) == 0,
                  ArgumentError("the weighting of an ExpertMixture is applied to the expert-return vector, for which no rows are held, so it must read none: `rows_needed(alg)` is $(rows_needed(alg)) for a `$(typeof(alg).name.name)`"))
        if !isnothing(p0)
            assert_nonempty(p0, :p0)
            assert_finite(p0, :p0)
            @argcheck(length(p0) == length(experts),
                      DimensionMismatch("p0 ($(length(p0))) must have one entry per expert ($(length(experts)))"))
        end
        if !isnothing(eset)
            @argcheck(isa(eset.wb, WeightBounds) && isnothing(eset.sets),
                      ArgumentError("the Expert Set is stated over the experts, which have no names: it holds a `WeightBounds` and no `sets`"))
        end
        return new{typeof(experts), typeof(alg), typeof(eset), typeof(proj), typeof(grad),
                   typeof(p0)}(experts, alg, eset, proj, grad, p0)
    end
end
function ExpertMixture(;
                       experts::AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
                       alg::AbstractOnlinePortfolioSelectionAlgorithm = BuyAndHold(),
                       eset::Option{<:BoundedAllocationSet} = nothing,
                       proj::EuclideanProjection = EuclideanProjection(),
                       grad::Union{OwnPoint, BlendPoint} = OwnPoint(),
                       p0::Option{<:AbstractVector} = nothing)::ExpertMixture
    return ExpertMixture(experts, alg, eset, proj, grad, p0)
end
function port_opt_view(alg::ExpertMixture, i, args...)
    return ExpertMixture(; experts = [port_opt_view(e, i, args...) for e in alg.experts],
                         alg = alg.alg, eset = alg.eset, proj = alg.proj, grad = alg.grad,
                         p0 = alg.p0)
end
function assert_rule_admits_set(alg::ExpertMixture, set::AbstractAllocationSet)::Nothing
    for e in alg.experts
        assert_rule_admits_set(e, set)
    end
    return nothing
end
function rows_needed(alg::ExpertMixture)
    need = rows_needed(alg.alg)
    for e in alg.experts
        need = rows_needed_max(need, rows_needed(e))
    end
    return need
end
function projection_geometry(alg::ExpertMixture)
    return alg.proj
end
function rule_state_seed(alg::ExpertMixture, w::AbstractVector,
                         set::Option{<:AbstractAllocationSet} = nothing)
    K = length(alg.experts)
    h = [project_start(projection_geometry(e), set, expert_start_allocation(e, w))
         for e in alg.experts]
    p = expert_start_weights(alg, K, eltype(w))
    return ExpertMixtureState(0, map(k -> rule_state_seed(alg.experts[k], h[k], set), 1:K),
                              h, rule_state_seed(alg.alg, p), p)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The weight vector over the `K` experts an [`ExpertMixture`](@ref) starts from: its `p0`, or the uniform vector when `p0` is `nothing`, projected once onto the Expert Set in the weighting's geometry, as the head's `w0` is onto the Allocation Set, so a start outside the set — a given one, or the uniform one under a bound it violates — is made feasible and never refused. On the bare simplex the uniform start lies in the set and the projection is skipped, so the default start is exactly uniform.

# Related

  - [`ExpertMixture`](@ref)
  - [`expert_allocation_set`](@ref)
  - [`project`](@ref)
"""
function expert_start_weights(alg::ExpertMixture, K::Integer, datatype::DataType)
    p0 = something(alg.p0, fill(one(datatype) / K, K))
    if isnothing(alg.p0) && isnothing(alg.eset)
        return p0
    end
    return project(projection_geometry(alg.alg),
                   expert_allocation_set(alg.eset, K, datatype), p0, p0)
end
"""
    expert_start_allocation(alg::AbstractOnlinePortfolioSelectionAlgorithm, w::AbstractVector)
    expert_start_allocation(alg::ConstantRebalancedPortfolio, w::AbstractVector)

The allocation an expert of an [`ExpertMixture`](@ref) holds during the first period: the Start Allocation for a rule that reads `w`, and the constant rebalanced portfolio's own `w`, so a sampled expert's wealth is Cover's ``S_t(\\boldsymbol{b}) = \\prod_{s \\leq t} \\langle \\boldsymbol{b}, \\boldsymbol{x}_s \\rangle`` from the first row. The head itself holds the Start Allocation for that period either way. The expert's `w` sums to one, as its constructor checks, and the seed of the mixture projects the answer onto the head's Allocation Set in the expert's geometry through [`project_start`](@ref), so a `w` that a bound of the set excludes, such as a one-hot allocation under a cap, is never held. The constant portfolio's `w` is answered in the numeric type the head's allocation and it promote to, so a target given as integers — the one-hot experts of a switching portfolio — seeds a carrier that holds the allocation a programme set projects it to.

# Related

  - [`ExpertMixture`](@ref)
  - [`ConstantRebalancedPortfolio`](@ref)
  - [`project_start`](@ref)
"""
function expert_start_allocation(::AbstractOnlinePortfolioSelectionAlgorithm,
                                 w::AbstractVector)
    return copy(w)
end
function expert_start_allocation(alg::ConstantRebalancedPortfolio, w::AbstractVector)
    if isnothing(alg.w)
        return fill(one(eltype(w)) / length(w), length(w))
    end
    # The carrier's numeric type is the head's, so a target given as integers, as the
    # one-hot experts of a switching portfolio are, holds the projected allocation.
    return promote_type(eltype(w), eltype(alg.w)).(alg.w)
end
function online_update!(alg::ExpertMixture, st::ExpertMixtureState, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet)
    r = [LinearAlgebra.dot(h, x) for h in st.h]
    point = gradient_point(alg.grad, w)
    for (k, e) in enumerate(alg.experts)
        st.st[k], st.h[k] = online_update!(e, st.st[k], st.h[k], x, rows, set, point)
    end
    # The weighting moves on the Expert Set, the bare simplex over the experts by default.
    pst, p = online_update!(alg.alg, st.pst, st.p, r, nothing,
                            expert_allocation_set(alg.eset, length(r), eltype(w)))
    q = zeros(eltype(w), length(w))
    for (k, h) in enumerate(st.h)
        q .+= p[k] .* h
    end
    # The blend meets the head's set once more, in the mixture's own geometry.
    wn = blend_projection(alg.proj, set, q, price_adjusted_allocation(w, x))
    return ExpertMixtureState(st.n + 1, st.st, st.h, pst, p), wn
end
"""
    gradient_point(grad::OwnPoint, w::AbstractVector)
    gradient_point(grad::BlendPoint, w::AbstractVector)

The Gradient Point an [`ExpertMixture`](@ref) hands its experts through the seven-argument [`online_update!`](@ref): `nothing` under [`OwnPoint`](@ref), so every expert takes its own six-argument update, and the mixture's played blend `w` under [`BlendPoint`](@ref).

# Related

  - [`ExpertMixture`](@ref)
  - [`online_update!`](@ref)
"""
function gradient_point(::OwnPoint, ::AbstractVector)
    return nothing
end
function gradient_point(::BlendPoint, w::AbstractVector)
    return w
end
"""
    expert_allocation_set(eset::Nothing, K::Integer, datatype::DataType)
    expert_allocation_set(eset::BoundedAllocationSet, K::Integer, datatype::DataType)

The Expert Set over `K` experts, resolved: the bare simplex when the mixture holds none, and the mixture's own bounds broadcast or matched over the experts otherwise.

# Related

  - [`ExpertMixture`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`resolve_allocation_set`](@ref)
"""
function expert_allocation_set(::Nothing, K::Integer, datatype::DataType)
    return resolve_allocation_set(BoundedAllocationSet(), K, false, datatype)
end
function expert_allocation_set(eset::BoundedAllocationSet, K::Integer, datatype::DataType)
    return resolve_allocation_set(eset, K, false, datatype)
end
"""
    UniversalPortfolio(; N::Integer, n_experts::Integer = 2000, alpha::Num_VecNum = 1, rng::Random.AbstractRNG = Random.default_rng(), seed::Option{<:Integer} = nothing, alg::AbstractOnlinePortfolioSelectionAlgorithm = BuyAndHold(), eset::Option{<:BoundedAllocationSet} = nothing, proj::EuclideanProjection = EuclideanProjection())

Cover's (1991) universal portfolio, approximated by a finite set of experts: the [`ExpertMixture`](@ref) under the wealth weighting over `n_experts` constant rebalanced portfolios sampled from a Dirichlet prior on the simplex of `N` assets (UP).

Cover defines the portfolio as the wealth-weighted average of **every** constant rebalanced portfolio, ``\\boldsymbol{w}_{t+1} = \\int_\\Delta \\boldsymbol{b}\\, S_t(\\boldsymbol{b})\\, d\\mu(\\boldsymbol{b}) / \\int_\\Delta S_t(\\boldsymbol{b})\\, d\\mu(\\boldsymbol{b})``, with ``S_t(\\boldsymbol{b}) = \\prod_{s \\leq t} \\langle \\boldsymbol{b}, \\boldsymbol{x}_s \\rangle``. The integral is intractable beyond a few assets, so the mixture replaces it with a Monte Carlo average over `n_experts` draws: `alpha = 1` is Cover's uniform prior, and `alpha = 1/2` is the Dirichlet(½) prior of Cover and Ordentlich (1996).

# Regret

Two bounds, each attributed to what it covers.

  - **Cover (1991), for the exact integral over the simplex.** For every price sequence, ``\\log S_T(\\text{BCRP}) - \\log S_T(\\text{UP}) \\leq (N - 1) \\log (T + 1)``, against the best constant rebalanced portfolio in hindsight. No estimator in the library carries a guarantee of that kind, because every one of them assumes something about the distribution.
  - **The mixture's own bound, exact for the shipped object.** For every sequence, ``\\log S_T(\\text{best sampled expert}) - \\log S_T(\\text{mixture}) \\leq \\log n_{\\text{experts}}``, because the mixture's wealth is the uniform average of the experts' wealths and an average is at least ``1 / n_{\\text{experts}}`` of its maximum.

The gap between the best sampled expert and the best constant rebalanced portfolio is sampling error: it shrinks with `n_experts`, has no closed form, and the count the simplex needs grows quickly with `N`. The cost is ``O(n_{\\text{experts}} \\cdot N)`` a row and the carrier is `n_experts × N`, which is why the [`NewtonStep`](@ref) rather than this is the practical second-order choice.

# Arguments

  - `N`: The number of assets of the pinned universe the experts are sampled over.
  - `n_experts`: The number of constant rebalanced experts.
  - `alpha`: The Dirichlet concentration, a scalar for the symmetric prior or one entry per asset.
  - `rng`, `seed`: The generator the experts are drawn from, resolved through [`resolve_rng`](@ref).
  - `alg`: The weighting over the experts.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry on every expert.

# Validation

  - `N >= 1`, `n_experts >= 1`. A `DomainError` is thrown otherwise.
  - `alpha`: non-empty, and every element is positive and finite; of length `N` when a vector. A `DimensionMismatch` is thrown otherwise.

# Examples

```jldoctest
julia> up = UniversalPortfolio(; N = 3, n_experts = 2, seed = 1);

julia> length(up.experts)
2
```

# Related

  - [`ExpertMixture`](@ref)
  - [`ConstantRebalancedPortfolio`](@ref)
  - [`BuyAndHold`](@ref)
  - [`RandomWeighted`](@ref)

# References

  - $(ref_dict[:cover1991])
  - $(ref_dict[:coverordentlich1996])
"""
function UniversalPortfolio(; N::Integer, n_experts::Integer = 2000, alpha::Num_VecNum = 1,
                            rng::Random.AbstractRNG = Random.default_rng(),
                            seed::Option{<:Integer} = nothing,
                            alg::AbstractOnlinePortfolioSelectionAlgorithm = BuyAndHold(),
                            eset::Option{<:BoundedAllocationSet} = nothing,
                            proj::EuclideanProjection = EuclideanProjection())::ExpertMixture
    @argcheck(N >= 1, DomainError(N, "N must be at least 1"))
    @argcheck(n_experts >= 1, DomainError(n_experts, "n_experts must be at least 1"))
    assert_nonempty_gt0_finite_val(alpha, :alpha)
    dist = if isa(alpha, Number)
        Distributions.Dirichlet(N, alpha)
    else
        @argcheck(length(alpha) == N,
                  DimensionMismatch("alpha ($(length(alpha))) must have one entry per asset ($N)"))
        Distributions.Dirichlet(alpha)
    end
    rng = resolve_rng(rng, seed)
    B = rand(rng, dist, n_experts)
    experts = [ConstantRebalancedPortfolio(; w = B[:, k], proj = proj) for k in 1:n_experts]
    return ExpertMixture(; experts = experts, alg = alg, eset = eset)
end
export BuyAndHold, ConstantRebalancedPortfolio, NewtonStep, NoSlack, LinearSlack,
       QuadraticSlack, PassiveAggressiveMeanReversion, OwnPoint, BlendPoint, ExpertMixture,
       UniversalPortfolio
public AbstractPassiveAggressiveSlack, passive_aggressive_step
