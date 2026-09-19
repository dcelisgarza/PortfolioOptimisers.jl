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

The exponentiated gradient of Helmbold, Schapire, Singer and Warmuth (1998): a multiplicative step up the last period's log return, under a relative-entropy penalty against the current allocation (EG).

# Mathematical definition

```math
\\begin{align}
w_{t+1, i} &\\propto w_{t, i} \\exp\\left( \\eta \\frac{x_{t, i}}{\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle} \\right)\\,,
\\end{align}
```

normalised to sum to one, which is the entropic projection onto the simplex. It is a momentum rule — it increases the weight of whatever just did well — and its regret is ``O(\\sqrt{T \\log N})`` with ``\\eta`` tuned to the horizon. As the weighting of an [`ExpertMixture`](@ref) it is the online gradient update over the expert-return vector.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExponentiatedGradient(; eta::Real = 0.05, proj::EntropicProjection = EntropicProjection()) -> ExponentiatedGradient

Keywords correspond to the struct's fields. The `proj` slot is bound to [`EntropicProjection`](@ref), the geometry of the theorem.

## Validation

  - `eta > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ExponentiatedGradient()
ExponentiatedGradient
   eta ┼ Float64: 0.05
  proj ┴ EntropicProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`EntropicProjection`](@ref)

# References

  - $(ref_dict[:helmbold1998])
"""
struct ExponentiatedGradient{T1 <: Real, T2 <: EntropicProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Learning rate. Larger reacts faster and is less stable.
    """
    eta::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function ExponentiatedGradient(eta::Real, proj::EntropicProjection)
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        return new{typeof(eta), typeof(proj)}(eta, proj)
    end
end
function ExponentiatedGradient(; eta::Real = 0.05,
                               proj::EntropicProjection = EntropicProjection())::ExponentiatedGradient
    return ExponentiatedGradient(eta, proj)
end
function online_update!(alg::ExponentiatedGradient, st, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    p = LinearAlgebra.dot(w, x)
    q = w .* exp.(alg.eta .* x ./ p)
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

with, when `eta > 0`, the shrinkage ``(1 - \\eta) \\boldsymbol{q} + \\eta \\boldsymbol{1} / N`` of the raw Newton point towards the uniform portfolio **before** the projection, so that a bound of zero or a negative lower bound on the Allocation Set is honoured — mixed after the projection, the uniform mass would leave the set. The paper's projection is in the norm of ``A_t``, the [`GramProjection`](@ref), a programme on every set that carries its own solver; the default [`EuclideanProjection`](@ref) is the standard simplification that keeps the rule free of a solver, and on the simplex at `eta = 0` the two agree wherever the Newton point already lies in it. The rule binds its current ``A_t`` onto the geometry through [`gram_geometry`](@ref) at every step; the Start Allocation is projected before any gradient at ``A_0 = I``, in the Euclidean geometry. The regret is ``O(N \\log T)`` at ``O(N^2)`` a step, which is why this rather than the universal portfolio is the practical second-order choice. As the weighting of an [`ExpertMixture`](@ref) it is the online Newton update over the expert-return vector, with a `K × K` Gram.

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

with ``\\tau_t`` the step of the `slack` rule and ``\\bar{x}_t`` the mean of ``\\boldsymbol{x}_t``; the step is zero when every asset moved alike. The rule sells what just rose: it is a total bet on single-period mean reversion, and its constraint runs the other way from [`ForecastReversion`](@ref)'s, on the realised relative with ``\\epsilon \\leq 1``.

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

  - `eps >= 0`. A `DomainError` is thrown otherwise.

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

The weighting's step is projected onto the **Expert Set** on `eset`, the Allocation Set over the `K` experts, in the weighting's own Projection Geometry: `nothing`, the default, is the bare `K`-simplex, a no-op for the multiplicative weightings and the Euclidean scalar root for a Newton weighting; a given [`BoundedAllocationSet`](@ref) broadcasts a scalar bound over the experts and takes one entry per expert from a vector bound, so a cap on `eset` caps the trust in any one expert. The blend ``\\sum_k p_{t+1, k} \\boldsymbol{h}_k(t+1)`` is then projected onto the head's Allocation Set once more, in the mixture's own Euclidean geometry on `proj`, with the mixture's Price-Adjusted Allocation as the reference; on a [`BoundedAllocationSet`](@ref) a blend of bounded allocations is bounded and the projection would be the identity, so [`blend_projection`](@ref) skips it by dispatch and the default configuration solves nothing; on a [`ProgrammeAllocationSet`](@ref) it is the repair a turnover ceiling under a weighting other than buy-and-hold, or a MIP kind, needs, and the mixture pays `K + 1` programmes per period, its experts' and its own, beside the `K × K` Gram of a Newton weighting. The mixture reads nothing of a given Start Allocation: the head holds it for one period and it is replaced by the experts' mix. Each expert's own Rule State starts where its rule starts — a constant rebalanced portfolio at its own `w`, a rule that reads `w` at the Start Allocation — so a sampled expert's wealth is Cover's ``S_t(\\boldsymbol{b})`` from the first row ([`expert_start_allocation`](@ref)).

The mixture's regret against its best expert is exact for the shipped object wherever the second projection is the identity: ``\\log S_T(\\text{best expert}) - \\log S_T(\\text{mixture}) \\leq \\log K`` for every sequence under the wealth weighting on the bare Expert Set, because the mixture's wealth is the `p_1`-weighted average of the experts' wealths. Where the Expert Set binds or the second projection repairs, the bound is not claimed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpertMixture(;
        experts::AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
        alg::AbstractOnlinePortfolioSelectionAlgorithm = BuyAndHold(),
        eset::Option{<:BoundedAllocationSet} = nothing,
        p::Option{<:AbstractVector} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ExpertMixture

Keywords correspond to the struct's fields. A `nothing` `p` starts the weighting uniform over the experts, and a `nothing` `eset` is the bare simplex over them.

## Validation

  - `experts` is non-empty. An `IsEmptyError` is thrown otherwise.
  - `rows_needed(alg) == 0`: the weighting is applied to the expert-return vector, for which no rows are held. An `ArgumentError` is thrown otherwise.
  - `p`: of length `length(experts)`, non-negative, finite and summing to one, when given.
  - `eset`, when given, holds a [`WeightBounds`](@ref) and no `sets`: an expert has no name for a [`WeightBoundsEstimator`](@ref) to resolve over. An `ArgumentError` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, every expert is viewed and the weighting, `eset`, `p` and `proj` are carried unchanged.

# Examples

```jldoctest
julia> ExpertMixture(; experts = [ExponentiatedGradient(), NewtonStep()])
ExpertMixture
  experts ┼ 2-element Vector{PortfolioOptimisers.AbstractOnlinePortfolioSelectionAlgorithm}
          │ ExponentiatedGradient ⋯
          │ NewtonStep ⋯
      alg ┼ BuyAndHold
          │   proj ┴ EuclideanProjection()
     eset ┼ nothing
        p ┼ nothing
     proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`UniversalPortfolio`](@ref)
  - [`ExpertMixtureState`](@ref)
  - [`BuyAndHold`](@ref)

# References

  - $(ref_dict[:lihoi2014])
"""
struct ExpertMixture{T1 <: AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
                     T2 <: AbstractOnlinePortfolioSelectionAlgorithm,
                     T3 <: Option{<:BoundedAllocationSet}, T4 <: Option{<:AbstractVector},
                     T5 <: EuclideanProjection} <: AbstractOnlinePortfolioSelectionAlgorithm
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
    The weight over the experts held during the first period, or `nothing` for uniform.
    """
    p::T4
    """
    The geometry the blend is projected onto the head's Allocation Set in, once more.
    """
    proj::T5
    function ExpertMixture(experts::AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
                           alg::AbstractOnlinePortfolioSelectionAlgorithm,
                           eset::Option{<:BoundedAllocationSet},
                           p::Option{<:AbstractVector}, proj::EuclideanProjection)
        @argcheck(!isempty(experts), IsEmptyError("experts cannot be empty"))
        @argcheck(rows_needed(alg) == 0,
                  ArgumentError("the weighting of an ExpertMixture is applied to the expert-return vector, for which no rows are held, so it must read none: `rows_needed(alg)` is $(rows_needed(alg)) for a `$(typeof(alg).name.name)`"))
        if !isnothing(p)
            assert_nonempty_nonneg_finite_val(p, :p)
            @argcheck(length(p) == length(experts),
                      DimensionMismatch("p ($(length(p))) must have one entry per expert ($(length(experts)))"))
            @argcheck(isapprox(sum(p), one(eltype(p))), DomainError(p, "p must sum to one"))
        end
        if !isnothing(eset)
            @argcheck(isa(eset.wb, WeightBounds) && isnothing(eset.sets),
                      ArgumentError("the Expert Set is stated over the experts, which have no names: it holds a `WeightBounds` and no `sets`"))
        end
        return new{typeof(experts), typeof(alg), typeof(eset), typeof(p), typeof(proj)}(experts,
                                                                                        alg,
                                                                                        eset,
                                                                                        p,
                                                                                        proj)
    end
end
function ExpertMixture(;
                       experts::AbstractVector{<:AbstractOnlinePortfolioSelectionAlgorithm},
                       alg::AbstractOnlinePortfolioSelectionAlgorithm = BuyAndHold(),
                       eset::Option{<:BoundedAllocationSet} = nothing,
                       p::Option{<:AbstractVector} = nothing,
                       proj::EuclideanProjection = EuclideanProjection())::ExpertMixture
    return ExpertMixture(experts, alg, eset, p, proj)
end
function port_opt_view(alg::ExpertMixture, i, args...)
    return ExpertMixture(; experts = [port_opt_view(e, i, args...) for e in alg.experts],
                         alg = alg.alg, eset = alg.eset, p = alg.p, proj = alg.proj)
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
function rule_state_seed(alg::ExpertMixture, w::AbstractVector)
    K = length(alg.experts)
    h = [expert_start_allocation(e, w) for e in alg.experts]
    p = isnothing(alg.p) ? fill(one(eltype(w)) / K, K) : copy(alg.p)
    return ExpertMixtureState(0, map(k -> rule_state_seed(alg.experts[k], h[k]), 1:K), h,
                              rule_state_seed(alg.alg, p), p)
end
"""
    expert_start_allocation(alg::AbstractOnlinePortfolioSelectionAlgorithm, w::AbstractVector)
    expert_start_allocation(alg::ConstantRebalancedPortfolio, w::AbstractVector)

The allocation an expert of an [`ExpertMixture`](@ref) holds during the first period: the Start Allocation for a rule that reads `w`, and the constant rebalanced portfolio's own `w`, so a sampled expert's wealth is Cover's ``S_t(\\boldsymbol{b}) = \\prod_{s \\leq t} \\langle \\boldsymbol{b}, \\boldsymbol{x}_s \\rangle`` from the first row. The head itself holds the Start Allocation for that period either way.

# Related

  - [`ExpertMixture`](@ref)
  - [`ConstantRebalancedPortfolio`](@ref)
"""
function expert_start_allocation(::AbstractOnlinePortfolioSelectionAlgorithm,
                                 w::AbstractVector)
    return copy(w)
end
function expert_start_allocation(alg::ConstantRebalancedPortfolio, w::AbstractVector)
    return isnothing(alg.w) ? fill(one(eltype(w)) / length(w), length(w)) : copy(alg.w)
end
function online_update!(alg::ExpertMixture, st::ExpertMixtureState, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet)
    r = [LinearAlgebra.dot(h, x) for h in st.h]
    for (k, e) in enumerate(alg.experts)
        st.st[k], st.h[k] = online_update!(e, st.st[k], st.h[k], x, rows, set)
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
export BuyAndHold, ConstantRebalancedPortfolio, ExponentiatedGradient, NewtonStep, NoSlack,
       LinearSlack, QuadraticSlack, PassiveAggressiveMeanReversion, ExpertMixture,
       UniversalPortfolio
public AbstractPassiveAggressiveSlack, passive_aggressive_step
