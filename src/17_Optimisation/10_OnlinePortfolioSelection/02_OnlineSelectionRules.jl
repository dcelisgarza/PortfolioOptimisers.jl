"""
$(DocStringExtensions.TYPEDEF)

Lets the allocation drift with the market and never trades it, the buy-and-hold benchmark of the online selection papers (BAH).

From the uniform Start Allocation it is the uniform buy-and-hold portfolio, which the papers use as the market index. The rule keeps no state and reads no rows.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\hat{\\boldsymbol{w}}_t \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_hat_t_padj])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:W_aset])
  - $(math_dict[:w_1_start])
  - $(math_dict[:t_period])
  - $(math_dict[:r_t_expert])
  - $(math_dict[:p_t_expert])

The entries of ``\\hat{\\boldsymbol{w}}_t`` are non-negative and sum to one. So on the simplex the projection returns ``\\hat{\\boldsymbol{w}}_t`` unchanged, and the rule never trades. The weight of asset ``i`` after period ``t`` is then ``w_{1,i} \\prod_{s \\leq t} x_{s,i}``, divided by the sum of that product over the assets.

As the weighting of an [`ExpertMixture`](@ref), the rule reads the expert-return vector ``\\boldsymbol{r}_t`` in place of ``\\boldsymbol{x}_t``. Its step is then ``\\boldsymbol{p}_{t+1} \\propto \\boldsymbol{p}_t \\odot \\boldsymbol{r}_t``, the wealth-weighted mixture that the papers write as `BAH_W`. It is the default weighting of the mixture.

# Algorithm

 1. Compute `wh`, the Price-Adjusted Allocation of `w` over `x`.
 2. Project `wh` onto the Allocation Set `set` with `alg.proj`, from `wh`. Return the carrier `st` unchanged, and the projection.

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
  - [`ExpertMixture`](@ref): the mixture whose default weighting this rule is.
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

Rebalances to the same allocation `w` after every price move, the constant rebalanced portfolio (CRP).

With the uniform `w`, the default, it is the uniform constant rebalanced portfolio (UCRP). The rule reads nothing of the market. The head holds a given Start Allocation for the first period, and the first update replaces it with `w`. The constant rebalanced portfolios sampled from the simplex are the experts of [`UniversalPortfolio`](@ref), and the single-asset ones are the experts of [`SwitchingPortfolio`](@ref).

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{b} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_crp])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:W_aset])
  - $(math_dict[:t_period])
  - $(math_dict[:S_t_crp])
  - $(math_dict[:x_t_rel])

On the simplex the projection returns ``\\boldsymbol{b}``, so the rule holds ``\\boldsymbol{b}`` in every period after the first. Over periods ``2`` to ``t`` its wealth is then ``S_t(\\boldsymbol{b}) / S_1(\\boldsymbol{b})``.

# Algorithm

 1. Set `q` to `alg.w`, or to the uniform allocation over the assets of `w` when `alg.w` is `nothing`.
 2. Project `q` onto `set` with `alg.proj`, from the Price-Adjusted Allocation of `w` over `x`. Return the carrier `st` unchanged, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConstantRebalancedPortfolio(;
        w::Option{<:AbstractVector} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ConstantRebalancedPortfolio

Keywords correspond to the struct's fields. A `nothing` `w` is the uniform allocation over the pinned universe.

## Validation

  - `w`, when given, is non-empty. An `IsEmptyError` is thrown otherwise.
  - Every element of `w` is finite and non-negative, when `w` is given. A `DomainError` is thrown otherwise.
  - `sum(w) ≈ 1` when `w` is given, because every Allocation Set has a budget of one. A `DomainError` is thrown otherwise.

## View parameters

`ConstantRebalancedPortfolio` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method slices `w` to the selected assets and divides it by the sum of the slice, so it sums to one again. A slice that sums to zero becomes the uniform allocation over the selected assets.
  - A `nothing` `w` stays `nothing`, so the view is uniform over the selected assets.

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
  - [`UniversalPortfolio`](@ref): the mixture over sampled constant rebalanced portfolios.
  - [`SwitchingPortfolio`](@ref): the mixture over the single-asset constant rebalanced portfolios.
  - [`BestConstantRebalancedPortfolio`](@ref): the best target in hindsight.
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cover1991])
"""
struct ConstantRebalancedPortfolio{T1 <: Option{<:AbstractVector},
                                   T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The target ``\\boldsymbol{b}``, the allocation that the rule rebalances to every period, over the pinned universe. A `nothing` value is the uniform allocation.
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

Carries the Gram matrix of the past gradients and their weighted sum, for the online Newton step.

[`NewtonStep`](@ref) seeds it with the identity matrix and a zero vector, and every update adds the gradient of the period to both in place. A view keeps the rows and the columns of the selected assets.

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
    The Gram matrix ``A_t = I + \\sum_{s \\leq t} \\boldsymbol{g}_s \\boldsymbol{g}_s^\\intercal``, `assets × assets`. The update writes it in place.
    """
    A
    """
    The weighted gradient sum ``\\boldsymbol{b}_t = (1 + 1/\\beta) \\sum_{s \\leq t} \\boldsymbol{g}_s``, `assets × 1`. The update writes it in place.
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

Steps to the Newton point of the log wealth of the past periods, the online Newton step (ONS) of Agarwal, Hazan, Kale and Schapire (2006).

The rule is a second-order method on the exp-concave log-wealth objective. Each step solves one dense ``N \\times N`` linear system, and the carrier holds an ``N \\times N`` matrix. Under a [`GramProjection`](@ref) each step also solves a programme. The head projects the Start Allocation before the first gradient, where ``A_0 = I`` and the Gram norm is the Euclidean norm, so it projects the start in the Euclidean geometry. As the weighting of an [`ExpertMixture`](@ref), the rule steps on the expert-return vector, and its Gram matrix is ``K \\times K`` for ``K`` experts.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{g}_s &= \\frac{\\boldsymbol{x}_s}{\\langle \\boldsymbol{w}_s, \\boldsymbol{x}_s \\rangle}\\,, \\\\
A_t &= I + \\sum_{s \\leq t} \\boldsymbol{g}_s \\boldsymbol{g}_s^\\intercal\\,, \\\\
\\boldsymbol{b}_t &= \\left(1 + \\frac{1}{\\beta}\\right) \\sum_{s \\leq t} \\boldsymbol{g}_s\\,, \\\\
\\boldsymbol{q} &= (1 - \\eta)\\, \\delta A_t^{-1} \\boldsymbol{b}_t + \\frac{\\eta}{N} \\boldsymbol{1}\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min}\\; (\\boldsymbol{w} - \\boldsymbol{q})^\\intercal M (\\boldsymbol{w} - \\boldsymbol{q})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{g}_s``: Gradient of the log wealth ``\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_s \\rangle`` of period ``s``, taken at the iterate.
  - ``A_t``: Gram matrix after period ``t``, ``N \\times N``.
  - ``I``: Identity matrix, ``N \\times N``.
  - ``\\boldsymbol{b}_t``: Weighted gradient sum after period ``t``.
  - ``\\beta``: Trade-off parameter, positive. It scales the gradient sum.
  - ``\\delta``: Scale of the Newton direction, positive.
  - ``\\eta``: Shrinkage towards the uniform allocation, in ``[0, 1)``.
  - $(math_dict[:q_raw])
  - ``M``: Matrix of the projection norm. It is ``A_t`` under a [`GramProjection`](@ref) and ``I`` under the [`EuclideanProjection`](@ref).
  - $(math_dict[:W_aset])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:N])
  - $(math_dict[:t_period])
  - $(math_dict[:T_regret])
  - ``\\alpha``: Market variability, the smallest entry of any price relative after each price relative is scaled so that its largest entry is one.

At ``M = A_t`` and ``\\eta = 0`` this is the update of the paper's Figure 1. The paper mixes the uniform allocation into the projected point, ``(1 - \\eta) \\boldsymbol{p}_t + \\eta \\boldsymbol{1} / N``. The rule mixes before the projection instead, so the answer stays in an Allocation Set whose bounds the uniform allocation breaks. On the simplex at ``\\eta = 0`` the two norms give one answer whenever ``\\delta A_t^{-1} \\boldsymbol{b}_t`` lies in the simplex.

The paper's Theorem 1 bounds the regret against the best constant rebalanced portfolio in hindsight. If the market variability is ``\\alpha > 0``, then ``\\eta = 0``, ``\\beta = \\alpha / (8 \\sqrt{N})`` and ``\\delta = 1`` give a regret of at most ``(10 N^{1.5} / \\alpha) \\log(N T / \\alpha^2)``. The rule at these values under the Gram norm is the paper's algorithm, so the bound holds for it. With no assumption on the market, the paper chooses ``\\eta`` and ``\\beta`` from ``T`` and bounds the regret by ``22 N^{1.25} \\sqrt{T \\log(N T)}``. That bound needs the paper's mix after the projection, so the rule does not carry it. The package claims neither bound under the Euclidean default.

# Algorithm

 1. Compute `g`, the gradient `x ./ dot(w, x)`.
 2. Add `g * transpose(g)` to `st.A` in place.
 3. Add `(1 + 1/beta) .* g` to `st.b` in place.
 4. Solve the symmetric system `st.A \\ st.b`, and scale the solution by `delta`, giving `q`.
 5. When `eta` is not zero, shrink `q` towards the uniform allocation in place.
 6. Bind `st.A` onto `alg.proj` with [`gram_geometry`](@ref). Project `q` onto `set` in that geometry, from the Price-Adjusted Allocation of `w` over `x`, giving `wn`.
 7. Return a carrier with the count raised by one, which holds the same `A` and `b`, and `wn`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NewtonStep(;
        beta::Real = 1,
        delta::Real = 0.125,
        eta::Real = 0,
        proj::Union{<:EuclideanProjection, <:GramProjection} = EuclideanProjection()
    ) -> NewtonStep

Keywords correspond to the struct's fields. The defaults `beta = 1`, `delta = 1/8` and `eta = 0` are the values of the paper's experiments. The `proj` slot takes the Gram geometry of the paper's theorem or the Euclidean geometry.

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
  - [`GramProjection`](@ref): the geometry of the paper's projection.
  - [`UniversalPortfolio`](@ref): the paper reports that the Newton step runs much faster than the sampled universal portfolio.

# References

  - $(ref_dict[:agarwal2006])
"""
struct NewtonStep{T1 <: Real, T2 <: Real, T3 <: Real,
                  T4 <: Union{<:EuclideanProjection, <:GramProjection}} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The trade-off parameter ``\\beta``, which scales the gradient sum.
    """
    beta::T1
    """
    The scale ``\\delta`` of the Newton direction.
    """
    delta::T2
    """
    The shrinkage ``\\eta`` of the Newton point towards the uniform allocation before the projection, in `[0, 1)`. At `0` the rule is the paper's algorithm.
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

Abstract supertype for the step-length rules of the passive aggressive mean reversion rule.

The paper's three variants, [`NoSlack`](@ref), [`LinearSlack`](@ref) and [`QuadraticSlack`](@ref), differ in one formula for the step length ``\\tau_t``. [`PassiveAggressiveMeanReversion`](@ref) holds one of them in its `slack` field.

# Interfaces

To implement a new step-length rule, subtype `AbstractPassiveAggressiveSlack` and implement the method below.

## `passive_aggressive_step`

  - `passive_aggressive_step(slack::AbstractPassiveAggressiveSlack, loss::Real, denom::Real) -> Real`: The step length ``\\tau_t`` of the period, non-negative.

### Arguments

  - `slack`: The rule.
  - `loss`: ``\\ell_t``, the hinge loss of the period, non-negative.
  - `denom`: ``\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2``, the squared norm of the centred price relative, positive.

### Returns

  - `tau::Real`: The step length.

# Examples

A rule that takes half the step of `NoSlack`.

```jldoctest
julia> struct HalfSlack <: PortfolioOptimisers.AbstractPassiveAggressiveSlack end

julia> function PortfolioOptimisers.passive_aggressive_step(::HalfSlack, loss::Real, denom::Real)
           return loss / (2 * denom)
       end

julia> PortfolioOptimisers.passive_aggressive_step(HalfSlack(), 0.2, 0.1)
1.0

julia> PassiveAggressiveMeanReversion(; slack = HalfSlack()).slack
HalfSlack()
```

# Related

  - [`NoSlack`](@ref)
  - [`LinearSlack`](@ref)
  - [`QuadraticSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)
  - [`passive_aggressive_step`](@ref)
"""
abstract type AbstractPassiveAggressiveSlack <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Takes the full step that brings the hinge loss of the period to zero, the step-length rule of PAMR.

# Mathematical definition

```math
\\begin{align}
\\tau_t &= \\frac{\\ell_t}{\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:tau_t_pa])
  - $(math_dict[:ell_t_pa])
  - $(math_dict[:xbar_t_pa])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eps_cw])
  - $(math_dict[:t_period])

It is the limit of the steps of [`LinearSlack`](@ref) and [`QuadraticSlack`](@ref) as ``C \\to \\infty``.

# Examples

```jldoctest
julia> NoSlack()
NoSlack()
```

# Related

  - [`AbstractPassiveAggressiveSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)

# References

  - $(ref_dict[:li2012pamr])
"""
struct NoSlack <: AbstractPassiveAggressiveSlack end
"""
$(DocStringExtensions.TYPEDEF)

Caps the step of `NoSlack` at the aggressiveness `C`, the step-length rule of PAMR-1.

# Mathematical definition

```math
\\begin{align}
\\tau_t &= \\min\\left( C, \\frac{\\ell_t}{\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:tau_t_pa])
  - $(math_dict[:C_pa])
  - $(math_dict[:ell_t_pa])
  - $(math_dict[:xbar_t_pa])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eps_cw])
  - $(math_dict[:t_period])

The paper derives this step from a linear penalty ``C \\xi`` on a slack ``\\xi \\geq 0`` of the loss constraint, without the non-negativity constraint on the allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearSlack(; C::Real = 500) -> LinearSlack

Keywords correspond to the struct's fields. The default is the value of the paper's experiments.

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
  - [`NoSlack`](@ref): the limit as ``C \\to \\infty``.

# References

  - $(ref_dict[:li2012pamr])
"""
struct LinearSlack{T1 <: Real} <: AbstractPassiveAggressiveSlack
    """
    The aggressiveness ``C``, the cap on the step.
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

Softens the step of `NoSlack` by adding `1 / (2C)` to its denominator, the step-length rule of PAMR-2.

# Mathematical definition

```math
\\begin{align}
\\tau_t &= \\frac{\\ell_t}{\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2 + \\frac{1}{2C}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:tau_t_pa])
  - $(math_dict[:C_pa])
  - $(math_dict[:ell_t_pa])
  - $(math_dict[:xbar_t_pa])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eps_cw])
  - $(math_dict[:t_period])

The paper derives this step from a quadratic penalty ``C \\xi^2`` on a slack ``\\xi`` of the loss constraint, without the non-negativity constraint on the allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    QuadraticSlack(; C::Real = 500) -> QuadraticSlack

Keywords correspond to the struct's fields. The default is the value of the paper's experiments.

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
  - [`NoSlack`](@ref): the limit as ``C \\to \\infty``.

# References

  - $(ref_dict[:li2012pamr])
"""
struct QuadraticSlack{T1 <: Real} <: AbstractPassiveAggressiveSlack
    """
    The aggressiveness ``C``, which softens the denominator.
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

Returns the step length of a passive-aggressive update under each step-length rule.

# Mathematical definition

```math
\\begin{align}
\\tau_t &= \\frac{\\ell_t}{\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2} && \\text{under } \\mathrm{NoSlack}\\,, \\\\
\\tau_t &= \\min\\left( C, \\frac{\\ell_t}{\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2} \\right) && \\text{under } \\mathrm{LinearSlack}\\,, \\\\
\\tau_t &= \\frac{\\ell_t}{\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2 + \\frac{1}{2C}} && \\text{under } \\mathrm{QuadraticSlack}\\,.
\\end{align}
```

Where:

  - $(math_dict[:tau_t_pa])
  - $(math_dict[:C_pa])
  - $(math_dict[:ell_t_pa])
  - $(math_dict[:xbar_t_pa])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:eps_cw])
  - $(math_dict[:t_period])

# Arguments

  - `slack`: The step-length rule.
  - `loss`: ``\\ell_t``, the hinge loss of the period, non-negative.
  - `denom`: ``\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert^2``, the squared norm of the centred price relative, positive.

# Returns

  - `tau::Real`: The step length ``\\tau_t``.

# Related

  - [`AbstractPassiveAggressiveSlack`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)

# References

  - $(ref_dict[:li2012pamr])
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

Moves the allocation the least distance that brings the last period's gross return down to `eps`, the passive aggressive mean reversion (PAMR) of Li, Zhao, Hoi and Gopalkrishnan (2012).

The rule sells what rose in the last period, a bet on mean reversion over one period. Its threshold bounds the return on the realised price relative from above. The threshold of [`ForecastReversion`](@ref) bounds the return on a forecast from below. The rule keeps no state and reads no rows.

# Mathematical definition

```math
\\begin{align}
\\ell_t &= \\max\\left(0, \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle - \\epsilon\\right)\\,, \\\\
\\boldsymbol{q} &= \\boldsymbol{w}_t - \\tau_t \\left( \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\right)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{q} \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:ell_t_pa])
  - $(math_dict[:tau_t_pa])
  - $(math_dict[:xbar_t_pa])
  - $(math_dict[:q_raw])
  - $(math_dict[:eps_cw])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:W_aset])
  - $(math_dict[:N])
  - $(math_dict[:t_period])

The step length ``\\tau_t`` is the step of the step-length rule. It is zero when every entry of ``\\boldsymbol{x}_t`` is equal, because then ``\\lVert \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\rVert = 0``. The paper sets it to zero in that case too.

The entries of ``\\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1}`` sum to zero, so ``\\boldsymbol{q}`` keeps the budget of one. Under [`NoSlack`](@ref) with ``\\ell_t > 0``, the gross return of ``\\boldsymbol{q}`` over period ``t`` is exactly ``\\epsilon``. So, by the paper's Proposition 1, ``\\boldsymbol{q}`` solves the paper's first problem without the non-negativity constraint. The projection onto the simplex is the normalisation step of the paper's Algorithm 1.

The paper takes ``\\epsilon \\geq 0``, and its experiments choose ``\\epsilon \\leq 1`` because the gross return of a period is near one. With a threshold above one, the rule moves only after a period whose gross return is above the threshold.

# Algorithm

 1. Compute `dev`, the price relative `x` minus its mean.
 2. Compute `denom`, the sum of the squares of `dev`.
 3. Compute `loss`, the hinge loss `max(0, dot(w, x) - eps)`.
 4. Set `tau` to zero when `denom` is zero. Otherwise, set `tau` to the step of `alg.slack` from `loss` and `denom`.
 5. Compute `q = w .- tau .* dev`.
 6. Project `q` onto `set` with `alg.proj`, from the Price-Adjusted Allocation of `w` over `x`. Return the carrier `st` unchanged, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PassiveAggressiveMeanReversion(;
        eps::Real = 0.5,
        slack::AbstractPassiveAggressiveSlack = NoSlack(),
        proj::EuclideanProjection = EuclideanProjection()
    ) -> PassiveAggressiveMeanReversion

Keywords correspond to the struct's fields. `NoSlack()`, `LinearSlack(; C)` and `QuadraticSlack(; C)` are the paper's PAMR, PAMR-1 and PAMR-2. The default `eps = 0.5` is the value of the paper's experiments.

## Validation

  - `eps >= 0`, the range that the paper states. A `DomainError` is thrown otherwise.

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
  - [`ForecastReversion`](@ref): the reversion rule whose threshold acts on a forecast.
  - [`ConfidenceWeightedMeanReversion`](@ref): the reversion rule that moves a belief over the allocation.

# References

  - $(ref_dict[:li2012pamr])
"""
struct PassiveAggressiveMeanReversion{T1 <: Real, T2 <: AbstractPassiveAggressiveSlack,
                                      T3 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The reversion threshold ``\\epsilon``. The rule moves only when the gross return of the last period is above it.
    """
    eps::T1
    """
    The step-length rule, one of the paper's three variants.
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

Carries the Rule State and the allocation of every expert, and the Rule State and the weights of the weighting, for an expert mixture.

[`ExpertMixture`](@ref) seeds it and replaces every entry once per period. A view slices the carrier and the allocation of every expert to the selected assets, and carries the weighting's carrier and the weights unchanged, because they are stated over the experts.

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
    The carrier of each expert, one entry per expert, `nothing` where the expert carries nothing.
    """
    st
    """
    The allocation that each expert holds during the current period, one vector per expert.
    """
    h
    """
    The carrier of the weighting, or `nothing`.
    """
    pst
    """
    The weight vector over the experts during the current period, `experts × 1`.
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

Makes every expert of a mixture read its gradient at its own iterate, the default Gradient Point.

Every expert then runs as it runs alone. [`ExpertMixture`](@ref) holds this point in its `grad` field by default.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{h}_k(t + 1) &= U_k\\left( \\boldsymbol{h}_k(t), \\nabla f_t(\\boldsymbol{h}_k(t)) \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:h_kt_expert])
  - ``U_k``: Online Update of expert ``k``, from its iterate, with the gradient that it reads.
  - $(math_dict[:f_t_online])
  - $(math_dict[:t_period])

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

Makes every first-order expert of a mixture read its gradient at the blend that the mixture played, while it steps from its own iterate.

This is the shared gradient of Zhang, Lu and Zhou (2018, Algorithm 4) and of Zhao, Zhang, Zhang and Zhou (2020, Algorithm 2). The papers state their dynamic-regret bounds on it, and [`Ader`](@ref) and [`Sword`](@ref) build the mixture on it. The mixture passes the blend through the seven-argument [`online_update!`](@ref). [`MirrorDescent`](@ref), [`OptimisticStep`](@ref) and [`AdaptiveSubgradient`](@ref) read it, and the hint of `OptimisticStep` stays the hint of its predictor. A rule with no gradient takes its six-argument update and ignores the point.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{h}_k(t + 1) &= U_k\\left( \\boldsymbol{h}_k(t), \\nabla f_t(\\boldsymbol{w}_t) \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:h_kt_expert])
  - ``U_k``: Online Update of expert ``k``, from its iterate, with the gradient that it reads.
  - $(math_dict[:f_t_online])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:t_period])

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

Mixes the allocations of several expert rules, with weights that a second rule moves on the experts' returns.

Every expert takes its own Online Update. The weighting is itself an Online Selection Rule, and it reads the expert-return vector in place of the price relative. Under [`BuyAndHold`](@ref), the default, the mixture is the wealth-weighted mixture that the papers write as `BAH_W`. Over sampled constant rebalanced portfolios it is Cover's universal portfolio, [`UniversalPortfolio`](@ref).

The weighting projects its step onto the Expert Set `eset`, in the weighting's own Projection Geometry. A `nothing` set is the bare simplex over the experts. A given [`BoundedAllocationSet`](@ref) broadcasts a scalar bound over the experts and reads one entry per expert from a vector bound, so a cap on `eset` caps the weight of any one expert.

The mixture then projects the blend onto the head's Allocation Set once more, in its own Euclidean geometry `proj`, from its Price-Adjusted Allocation. On a [`BoundedAllocationSet`](@ref) a blend of feasible allocations is feasible, so [`blend_projection`](@ref) skips the projection and the default configuration solves nothing. On a [`ProgrammeAllocationSet`](@ref) the projection is the repair that a MIP kind needs, or a turnover ceiling under a weighting other than buy-and-hold. Then the mixture solves ``K + 1`` programmes a period, one for each expert and one of its own. A Newton weighting holds a ``K \\times K`` Gram matrix, so two thousand experts cost a ``2000 \\times 2000`` solve a period.

The mixture reads nothing of a given Start Allocation. The head holds it for the first period, and the experts' mix replaces it. Each expert starts where its rule starts. A constant rebalanced portfolio starts at its own `w`, and any other rule starts at the Start Allocation. The seed projects each start onto the head's Allocation Set in the expert's geometry, as [`project_start`](@ref) projects the head's `w0`. So a sampled expert's wealth is Cover's ``S_t(\\boldsymbol{b})`` of its feasible allocation from the first period. The weights start at `p0`, uniform by default, and the seed projects them onto the Expert Set in the weighting's geometry. The seed makes a start outside a set feasible, and never refuses it.

`grad` is the Gradient Point. Under [`OwnPoint`](@ref), the default, every expert reads its gradient at its own iterate. Under [`BlendPoint`](@ref), every first-order expert reads its gradient at the blend ``\\boldsymbol{w}_t`` that the mixture played, and steps from its own iterate.

# Mathematical definition

```math
\\begin{align}
r_{t,k} &= \\langle \\boldsymbol{h}_k(t), \\boldsymbol{x}_t \\rangle\\,, \\\\
\\boldsymbol{h}_k(t + 1) &= U_k\\left( \\boldsymbol{h}_k(t), \\boldsymbol{x}_t \\right)\\,, \\\\
\\boldsymbol{p}_{t+1} &= V_{\\mathcal{E}}\\left( \\boldsymbol{p}_t, \\boldsymbol{r}_t \\right)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\sum_{k=1}^{K} p_{t+1,k}\\, \\boldsymbol{h}_k(t + 1) \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:r_t_expert])
  - $(math_dict[:h_kt_expert])
  - ``U_k``: Online Update of expert ``k``, from its iterate, on the price relative.
  - $(math_dict[:p_t_expert])
  - ``V_{\\mathcal{E}}``: Online Update of the weighting, on the expert-return vector, projected onto the Expert Set in the weighting's geometry.
  - $(math_dict[:E_eset])
  - $(math_dict[:K_expert])
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:W_aset])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:t_period])
  - ``S_{T,k}``: Wealth of expert ``k`` after period ``T``, from a wealth of one.
  - ``\\hat{S}_T``: Wealth of the mixture after period ``T``, from a wealth of one.
  - $(math_dict[:T_regret])
  - $(math_dict[:ell_t_surr])

Under [`BuyAndHold`](@ref) on the bare Expert Set, ``\\boldsymbol{p}_{t+1} \\propto \\boldsymbol{p}_t \\odot \\boldsymbol{r}_t``. Wherever the projection onto ``\\mathcal{W}`` returns the blend, the wealth of the mixture is then the ``\\boldsymbol{p}_1``-weighted average of the experts' wealths. So for every sequence of price relatives, ``\\max_k \\log S_{T,k} - \\log \\hat{S}_T \\leq -\\log p_{1,k^\\star}``, with ``k^\\star`` the best expert. At the uniform start the bound is ``\\log K``, the bound of Cover and Ordentlich (1996) for an equal split over expert strategies. The package does not claim the bound where the Expert Set binds or the second projection changes the blend.

Under [`BlendPoint`](@ref) with [`ExponentiatedGradient`](@ref) as the weighting, ``\\langle \\boldsymbol{p}_t, \\boldsymbol{r}_t \\rangle = \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` wherever the projection onto ``\\mathcal{W}`` returns the blend. The weight update is then exactly the exponentially weighted forecaster on the surrogate losses ``\\ell_t(\\boldsymbol{h}_k(t))``. [`Ader`](@ref) and [`Sword`](@ref) build that mixture over a geometric grid of first-order experts.

# Algorithm

 1. Compute `r`, the expert-return vector, from the held allocations `st.h`.
 2. Compute `point`, the Gradient Point, with [`gradient_point`](@ref).
 3. Take the Online Update of every expert from its own carrier and allocation, with `point`. Store the new carrier and allocation in `st.st` and `st.h`.
 4. Resolve the Expert Set with [`expert_allocation_set`](@ref). Take the Online Update of the weighting from `st.pst` and `st.p`, on `r`, over that set, giving `pst` and `p`.
 5. Compute the blend `q`, the sum of `p[k] .* h` over the experts.
 6. Project `q` onto `set` with [`blend_projection`](@ref) in `alg.proj`, from the Price-Adjusted Allocation of `w` over `x`, giving `wn`.
 7. Return a carrier with the count raised by one, which holds the new `pst` and `p`, and `wn`.

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
  - `rows_needed(alg) == 0`, because the weighting reads the expert-return vector, and the head holds no rows of it. An `ArgumentError` is thrown otherwise.
  - `p0`, when given, is non-empty. An `IsEmptyError` is thrown otherwise.
  - Every element of `p0` is finite, when `p0` is given. A `DomainError` is thrown otherwise.
  - `p0`, when given, has one entry per expert. A `DimensionMismatch` is thrown otherwise. The seed projects the start onto the Expert Set, so the start need not lie in it.
  - `eset`, when given, holds a [`WeightBounds`](@ref) and no `sets`, because an expert has no name for a [`WeightBoundsEstimator`](@ref) to resolve. An `ArgumentError` is thrown otherwise.

## View parameters

`ExpertMixture` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - Every expert recurses through [`port_opt_view`](@ref) with the same arguments.
  - The weighting, `eset`, `proj`, `grad` and `p0` pass unchanged, because they are stated over the experts, not over the assets.

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
  - [`UniversalPortfolio`](@ref): the mixture over sampled constant rebalanced portfolios.
  - [`SwitchingPortfolio`](@ref): the mixture over the single-asset constant rebalanced portfolios.
  - [`ExpertMixtureState`](@ref)
  - [`BuyAndHold`](@ref): the default weighting.
  - [`OwnPoint`](@ref)
  - [`BlendPoint`](@ref)
  - [`Ader`](@ref)
  - [`Sword`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:lihoi2014])
  - $(ref_dict[:coverordentlich1996])
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
    The weighting, a rule of the family that reads the expert-return vector in place of the price relative.
    """
    alg::T2
    """
    The Expert Set that the weighting projects onto, over the experts, or `nothing` for the bare simplex over them.
    """
    eset::T3
    """
    The geometry of the second projection, which projects the blend onto the head's Allocation Set.
    """
    proj::T4
    """
    The Gradient Point, where every first-order expert reads its gradient.
    """
    grad::T5
    """
    The Start Allocation over the experts, projected onto the Expert Set at the seed, or `nothing` for the uniform one.
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

Returns the weights over the `K` experts that an [`ExpertMixture`](@ref) starts from, projected onto the Expert Set.

The start is `p0`, or the uniform weights when `p0` is `nothing`. The projection is in the weighting's geometry, as the head projects `w0` onto the Allocation Set, so the projection makes a start outside the set feasible, and the function never refuses it. On the bare simplex the uniform start lies in the set, so the function skips the projection and the default start is exactly uniform.

# Algorithm

 1. Set `p0` to `alg.p0`, or to the uniform vector of length `K` in `datatype` when `alg.p0` is `nothing`.
 2. When `alg.p0` and `alg.eset` are both `nothing`, return `p0`.
 3. Otherwise, resolve the Expert Set with [`expert_allocation_set`](@ref). Project `p0` onto it in the geometry of `alg.alg`, from `p0`, and return the projection.

# Arguments

  - `alg`: The mixture.
  - `K`: The number of experts.
  - `datatype`: The numeric type of the weights.

# Returns

  - `p::AbstractVector`: The start weights, one entry per expert.

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

Returns the allocation that an expert of an [`ExpertMixture`](@ref) holds during the first period.

A rule that reads `w` holds the Start Allocation. A constant rebalanced portfolio holds its own target, or the uniform allocation when its `w` is `nothing`, so a sampled expert's wealth is Cover's ``S_t(\\boldsymbol{b})`` from the first period. The head itself holds the Start Allocation for that period in both cases. The seed of the mixture then projects the answer onto the head's Allocation Set in the expert's geometry, through [`project_start`](@ref). So the expert never holds a target that a bound of the set excludes, such as a one-hot allocation under a cap. The target comes back in the numeric type that the head's allocation and the target promote to. So an integer target, such as a one-hot expert of a switching portfolio, seeds a carrier that can hold the projected allocation.

# Arguments

  - `alg`: The expert rule.
  - `w`: The Start Allocation of the head.

# Returns

  - `h::AbstractVector`: The allocation that the expert holds during the first period, before the seed projects it.

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

Returns the Gradient Point that an [`ExpertMixture`](@ref) passes to its experts through the seven-argument [`online_update!`](@ref).

Under [`OwnPoint`](@ref) the point is `nothing`, so every expert takes its own six-argument update. Under [`BlendPoint`](@ref) it is the blend `w` that the mixture played.

# Arguments

  - `grad`: The Gradient Point of the mixture.
  - `w`: The blend that the mixture played during the period.

# Returns

  - `point::Option{<:AbstractVector}`: `nothing`, or `w`.

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

Resolves the Expert Set over `K` experts.

A mixture that holds no Expert Set gets the bare simplex over the experts. A mixture that holds one gets its bounds, a scalar bound broadcast over the experts and a vector bound read with one entry per expert.

# Arguments

  - `eset`: The Expert Set of the mixture, or `nothing`.
  - `K`: The number of experts.
  - `datatype`: The numeric type of the bounds.

# Returns

  - `set::BoundedAllocationSet`: The resolved Expert Set.

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
    UniversalPortfolio(;
        N::Integer,
        n_experts::Integer = 2000,
        alpha::Num_VecNum = 1,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        alg::AbstractOnlinePortfolioSelectionAlgorithm = BuyAndHold(),
        eset::Option{<:BoundedAllocationSet} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ExpertMixture

Builds Cover's (1991) universal portfolio as an expert mixture over `n_experts` constant rebalanced portfolios sampled from a Dirichlet prior on the simplex (UP).

Cover defines the portfolio as the wealth-weighted average of every constant rebalanced portfolio. The integral has no closed form beyond a few assets, so the mixture replaces it with an average over `n_experts` draws. `alpha = 1` is Cover's uniform prior, and `alpha = 1/2` is the Dirichlet(½) prior of Cover and Ordentlich (1996).

Each period costs ``O(K N)``, and the carrier holds ``K`` allocations of ``N`` entries. The gap between the best sampled expert and the best constant rebalanced portfolio is sampling error. It shrinks as `n_experts` grows, it has no closed form, and it needs more draws as ``N`` grows.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}_{t+1} &= \\frac{\\int_{\\Delta_N} \\boldsymbol{b}\\, S_t(\\boldsymbol{b})\\, d\\mu(\\boldsymbol{b})}{\\int_{\\Delta_N} S_t(\\boldsymbol{b})\\, d\\mu(\\boldsymbol{b})}
\\approx \\frac{\\sum_{k=1}^{K} \\boldsymbol{b}_k\\, S_t(\\boldsymbol{b}_k)}{\\sum_{k=1}^{K} S_t(\\boldsymbol{b}_k)}\\,, \\quad \\boldsymbol{b}_k \\sim \\mu = \\mathrm{Dirichlet}(\\boldsymbol{\\alpha})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_iter])
  - $(math_dict[:Delta_N_simplex])
  - $(math_dict[:b_crp])
  - $(math_dict[:S_t_crp])
  - ``\\mu``: Dirichlet prior on the simplex.
  - $(math_dict[:alpha_dirichlet_conc])
  - ``\\boldsymbol{b}_k``: Target of sampled expert ``k``.
  - $(math_dict[:K_expert])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:N])
  - $(math_dict[:t_period])
  - $(math_dict[:T_regret])
  - ``S^\\star_T``: Wealth of the best constant rebalanced portfolio in hindsight after period ``T``, from a wealth of one.
  - ``\\hat{S}_T``: Wealth of the universal portfolio after period ``T``, from a wealth of one.

Under the default weighting on the bare Expert Set, the mixture's allocation is the sampled ratio on the right, with no error beyond the sampling. Two bounds hold for every sequence of price relatives.

  - The exact integral, Theorems 1 and 2 of Cover and Ordentlich (1996). Under the uniform prior, ``\\log S^\\star_T - \\log \\hat{S}_T \\leq (N - 1) \\log(T + 1)``. Under the Dirichlet(½) prior, the bound is ``\\log 2 + \\frac{N - 1}{2} \\log(T + 1)``. Neither bound needs an assumption on the price relatives. Cover (1991) proves that the growth rates of the two portfolios agree as ``T \\to \\infty``, for every bounded sequence.
  - The sampled mixture, exact for the object that the package builds. The best sampled expert beats the mixture by at most ``\\log K`` in log wealth, because the wealth of the mixture is the average of the experts' wealths, and an average is at least ``1 / K`` of its largest term.

# Algorithm

 1. Check `N` and `n_experts`.
 2. Check `alpha`. Build the symmetric Dirichlet distribution of dimension `N` from a scalar `alpha`. From a vector `alpha`, check its length and build the Dirichlet distribution with it.
 3. Resolve the generator from `rng` and `seed` with [`resolve_rng`](@ref).
 4. Draw `B`, `N × n_experts`, one target per column.
 5. Build one [`ConstantRebalancedPortfolio`](@ref) for each column of `B`, each with `proj`.
 6. Return the [`ExpertMixture`](@ref) over them, with the weighting `alg` and the Expert Set `eset`.

# Arguments

  - `N`: The number of assets of the pinned universe that the experts are sampled over.
  - `n_experts`: The number ``K`` of constant rebalanced experts.
  - `alpha`: The Dirichlet concentration, a scalar for the symmetric prior or one entry per asset.
  - `rng`, `seed`: The generator that draws the experts, resolved through [`resolve_rng`](@ref).
  - `alg`: The weighting over the experts.
  - `eset`: The Expert Set that the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry of every expert.

# Validation

  - `N >= 1`, `n_experts >= 1`. A `DomainError` is thrown otherwise.
  - `alpha` is non-empty. An `IsEmptyError` is thrown otherwise.
  - Every element of `alpha` is positive and finite. A `DomainError` is thrown otherwise.
  - A vector `alpha` has one entry per asset. A `DimensionMismatch` is thrown otherwise.

# Examples

```jldoctest
julia> up = UniversalPortfolio(; N = 3, n_experts = 2, seed = 1);

julia> length(up.experts)
2
```

# Related

  - [`ExpertMixture`](@ref)
  - [`ConstantRebalancedPortfolio`](@ref)
  - [`BuyAndHold`](@ref): the default weighting.
  - [`NewtonStep`](@ref): its paper reports that it runs much faster than the sampled universal portfolio.
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
