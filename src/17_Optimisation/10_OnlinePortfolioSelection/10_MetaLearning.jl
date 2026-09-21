"""
$(DocStringExtensions.TYPEDEF)

The switching weighting of Singer (1997): the wealth step, then a fixed share that keeps each weight with probability ``1 - \\gamma`` and redistributes it uniformly over the other ``K - 1`` with probability ``\\gamma`` (SP).

# Mathematical definition

With ``\\hat{\\boldsymbol{w}}_t = \\boldsymbol{w}_t \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` the Price-Adjusted Allocation,

```math
\\begin{align}
w_{t+1, k} &= (1 - \\gamma)\\, \\hat{w}_{t, k} + \\frac{\\gamma}{K - 1} \\sum_{j \\neq k} \\hat{w}_{t, j}\\,,
\\end{align}
```

the paper's equations 4 to 6, whose update resembles the fixed share of Herbster and Warmuth, as the paper notes. The share is a stochastic-matrix mix of a vector on the simplex, so the raw step lies on the simplex and the rule projects nothing on the default Allocation Set; on a stated set the step is projected in the Euclidean geometry, the `proj` slot's bound. The rule carries nothing and reads no rows.

As the weighting of an [`ExpertMixture`](@ref) it is applied to the expert-return vector: the wealth step is the buy-and-hold weighting, and the share is the switch from one expert to another. Over the ``N`` single-asset constant rebalanced portfolios it is the paper's switching portfolio, [`SwitchingPortfolio`](@ref), and on the head over the assets it computes the same numbers, because the blend of the unit experts is the weight vector itself. The two ends of ``\\gamma`` are named rules: at ``\\gamma = 0`` the share is the identity and the rule is [`BuyAndHold`](@ref); at ``\\gamma = (K - 1) / K`` the share maps every vector to the uniform one, so over the unit experts the rule is the uniform [`ConstantRebalancedPortfolio`](@ref). The paper's varying rate, which follows the Krichevsky–Trofimov estimator, is not built.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SwitchingWeighting(; gamma::Real = 1/3, proj::EuclideanProjection = EuclideanProjection()) -> SwitchingWeighting

Keywords correspond to the struct's fields. The default rate is the paper's ``1/3``, at which the paper's two-stock case does well.

## Validation

  - `0 <= gamma <= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> SwitchingWeighting()
SwitchingWeighting
  gamma ┼ Float64: 0.3333333333333333
   proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ExpertMixture`](@ref)
  - [`SwitchingPortfolio`](@ref)
  - [`BuyAndHold`](@ref)

# References

  - $(ref_dict[:singer1997])
"""
struct SwitchingWeighting{T1 <: Real, T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The switching rate: the share of each weight redistributed over the others every period.
    """
    gamma::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function SwitchingWeighting(gamma::Real, proj::EuclideanProjection)
        @argcheck(zero(gamma) <= gamma <= one(gamma),
                  DomainError(gamma, "gamma must be in [0, 1]"))
        return new{typeof(gamma), typeof(proj)}(gamma, proj)
    end
end
function SwitchingWeighting(; gamma::Real = 1 / 3,
                            proj::EuclideanProjection = EuclideanProjection())::SwitchingWeighting
    return SwitchingWeighting(gamma, proj)
end
function online_update!(alg::SwitchingWeighting, st, w::AbstractVector, x::AbstractVector,
                        ::Any, set::AbstractAllocationSet)
    q = price_adjusted_allocation(w, x)
    K = length(q)
    # Over one expert there is nothing to switch to, and the share's denominator is zero.
    p = if K == 1
        q
    else
        (1 - alg.gamma) .* q .+ (alg.gamma / (K - 1)) .* (sum(q) .- q)
    end
    return st, project(alg.proj, set, p, q)
end
"""
    SwitchingPortfolio(; N::Integer, gamma::Real = 1/3, eset::Option{<:BoundedAllocationSet} = nothing, proj::EuclideanProjection = EuclideanProjection())

Singer's (1997) switching portfolio: the [`ExpertMixture`](@ref) under [`SwitchingWeighting`](@ref) over the `N` single-asset [`ConstantRebalancedPortfolio`](@ref)s of the pinned universe (SP).

The paper's portfolio is the Bayesian mix of a hidden process that holds one stock and switches to another with probability ``\\gamma`` each period; the mixture's weight over the unit experts is that posterior, and its blend is the weight vector itself. The degeneracies as rules: at `gamma = 0` the mixture is [`BuyAndHold`](@ref), and at `gamma = (N - 1) / N` it is the uniform constant rebalanced portfolio. A view of the head onto a subset of the assets drops no expert: the unit expert of an excluded asset has no mass on the kept assets, so its constant rebalanced portfolio becomes the uniform allocation over them, and the view runs with one uniform expert per excluded asset in place of the switching portfolio over the kept assets. Construct the portfolio over the assets it will run on.

# Arguments

  - `N`: The number of assets of the pinned universe, one unit expert each.
  - `gamma`: The switching rate.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry the blend is projected onto the head's Allocation Set in.

# Validation

  - `N >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> sp = SwitchingPortfolio(; N = 3);

julia> length(sp.experts), sp.alg.gamma
(3, 0.3333333333333333)
```

# Related

  - [`ExpertMixture`](@ref)
  - [`SwitchingWeighting`](@ref)
  - [`ConstantRebalancedPortfolio`](@ref)

# References

  - $(ref_dict[:singer1997])
"""
function SwitchingPortfolio(; N::Integer, gamma::Real = 1 / 3,
                            eset::Option{<:BoundedAllocationSet} = nothing,
                            proj::EuclideanProjection = EuclideanProjection())::ExpertMixture
    @argcheck(N >= 1, DomainError(N, "N must be at least 1"))
    experts = [ConstantRebalancedPortfolio(; w = [k == i ? 1 : 0 for i in 1:N],
                                           proj = proj) for k in 1:N]
    return ExpertMixture(; experts = experts, alg = SwitchingWeighting(; gamma = gamma),
                         eset = eset)
end
"""
    rate_grid_experts(eta_min::Real, K::Integer, obj::AbstractOnlineObjective)

The `K` [`GradientProjection`](@ref) experts of [`Ader`](@ref) and [`Sword`](@ref) at the geometric rate grid ``\\eta_i = 2^{i - 1} \\eta_{\\min}``, ``i = 1, \\ldots, K``, on the objective `obj`.

# Validation

  - `eta_min > 0`, `K >= 1`. A `DomainError` is thrown otherwise.

# Related

  - [`Ader`](@ref)
  - [`Sword`](@ref)
  - [`GradientProjection`](@ref)
"""
function rate_grid_experts(eta_min::Real, K::Integer, obj::AbstractOnlineObjective)
    @argcheck(eta_min > zero(eta_min), DomainError(eta_min, "eta_min must be positive"))
    @argcheck(K >= 1, DomainError(K, "K must be at least 1"))
    two = 2 * one(eta_min)
    return [GradientProjection(; eta = eta_min * two^(i - 1), obj = obj) for i in 1:K]
end
"""
    Ader(; eta_min::Real, K::Integer, eps::Real, obj::AbstractOnlineObjective = LogWealth(), eset::Option{<:BoundedAllocationSet} = nothing, proj::EuclideanProjection = EuclideanProjection())

The improved Ader of Zhang, Lu and Zhou (2018, Algorithms 3 and 4): the [`ExpertMixture`](@ref) under [`BlendPoint`](@ref) over `K` [`GradientProjection`](@ref) experts at the geometric rate grid ``\\eta_i = 2^{i - 1} \\eta_{\\min}``, weighted by [`ExponentiatedGradient`](@ref) at the rate `eps` from the start ``p_{0, i} \\propto 1 / (i (i + 1))``.

Every expert takes one Euclidean step from its own iterate on the gradient at the mixture's played blend, the surrogate loss ``\\ell_t(\\boldsymbol{w}) = \\langle \\nabla f_t(\\boldsymbol{w}_t), \\boldsymbol{w} - \\boldsymbol{w}_t \\rangle`` of the paper, and the weighting is the exponentially weighted forecaster on that loss, which over the expert-return vector is the exponentiated-gradient weighting exactly wherever the blend's second projection is the identity. The start over the experts is the paper's ``w_1^i = C / (i (i + 1))`` with ``C = 1 + 1/K``, which sums to one, so the projection at the seed is the identity; the smallest rate carries the largest prior weight.

The paper's grid and rates are stated for a horizon `T` on a set of diameter `D` under a gradient bound `G`: ``\\eta_{\\min} = (D / G) \\sqrt{7 / (2 T)}``, ``K = \\lceil \\tfrac{1}{2} \\log_2 (1 + 4 T / 7) \\rceil + 1`` and ``\\alpha = \\sqrt{2 / (T G^2 D^2)}`` on `eps`, at which the dynamic regret against any comparator sequence of path length ``P_T`` is ``O(\\sqrt{T (1 + P_T)})``. No online rule knows the horizon, so the three are the caller's; on the simplex ``D = \\sqrt{2}``, and the log-wealth gradient is bounded by ``1 / r`` when every price relative is at least ``r``.

# Arguments

  - `eta_min`: The smallest rate of the grid.
  - `K`: The number of experts.
  - `eps`: The weighting's rate, the paper's ``\\alpha``.
  - `obj`: The objective every expert steps on.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry the blend is projected onto the head's Allocation Set in.

# Validation

  - `eta_min > 0`, `K >= 1`, `eps > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> mix = Ader(; eta_min = 0.05, K = 3, eps = 0.1);

julia> [e.eta for e in mix.experts], mix.p0
([0.05, 0.1, 0.2], [0.6666666666666666, 0.2222222222222222, 0.1111111111111111])
```

# Related

  - [`ExpertMixture`](@ref)
  - [`BlendPoint`](@ref)
  - [`GradientProjection`](@ref)
  - [`ExponentiatedGradient`](@ref)
  - [`Sword`](@ref)

# References

  - $(ref_dict[:zhang2018ader])
"""
function Ader(; eta_min::Real, K::Integer, eps::Real,
              obj::AbstractOnlineObjective = LogWealth(),
              eset::Option{<:BoundedAllocationSet} = nothing,
              proj::EuclideanProjection = EuclideanProjection())::ExpertMixture
    experts = rate_grid_experts(eta_min, K, obj)
    p0 = [inv(i * (i + 1)) for i in 1:K]
    p0 ./= sum(p0)
    return ExpertMixture(; experts = experts, alg = ExponentiatedGradient(; eta = eps),
                         eset = eset, proj = proj, grad = BlendPoint(), p0 = p0)
end
"""
    Sword(; eta_min::Real, K::Integer, eps::Real, obj::AbstractOnlineObjective = LogWealth(), eset::Option{<:BoundedAllocationSet} = nothing, proj::EuclideanProjection = EuclideanProjection())

The small-loss Sword of Zhao, Zhang, Zhang and Zhou (2020, Theorem 5): the [`ExpertMixture`](@ref) under [`BlendPoint`](@ref) over `K` [`GradientProjection`](@ref) experts at the geometric rate grid ``\\eta_i = 2^{i - 1} \\eta_{\\min}``, weighted by [`ExponentiatedGradient`](@ref) at the rate `eps` from the uniform start.

The experts and the weighting are [`Ader`](@ref)'s — the paper's meta-algorithm is the exponentially weighted forecaster on the linearised loss with no optimism, which is the exponentiated-gradient weighting over the expert-return vector wherever the blend's second projection is the identity — and the two differ in the grid the paper states and in the start over the experts, uniform here. The paper's grid is for a horizon `T` on a set of diameter `D` under a gradient bound `G` and a smoothness constant `L`: ``\\eta_{\\min} = \\sqrt{D / (16 L G T)}`` and ``K = \\lceil \\tfrac{1}{2} \\log_2 (G T / (D L)) \\rceil + 1``, with ``\\varepsilon = \\sqrt{(2 + \\ln K) / (D^2 F_T)}`` on `eps` for ``F_T`` the cumulative loss, at which the dynamic regret is ``O(\\sqrt{(1 + P_T + F_T)(1 + P_T)})``; none of the four is known to an online rule, so the three are the caller's. The paper's gradient-variation form, whose experts take the extra-gradient step and whose weighting carries an optimistic hint, and its best-of-both-worlds form, which learns the hint in parallel, are not built.

# Arguments

  - `eta_min`: The smallest rate of the grid.
  - `K`: The number of experts.
  - `eps`: The weighting's rate, the paper's ``\\varepsilon``.
  - `obj`: The objective every expert steps on.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry the blend is projected onto the head's Allocation Set in.

# Validation

  - `eta_min > 0`, `K >= 1`, `eps > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> mix = Sword(; eta_min = 0.05, K = 3, eps = 0.1);

julia> [e.eta for e in mix.experts], mix.p0
([0.05, 0.1, 0.2], nothing)
```

# Related

  - [`ExpertMixture`](@ref)
  - [`BlendPoint`](@ref)
  - [`GradientProjection`](@ref)
  - [`ExponentiatedGradient`](@ref)
  - [`Ader`](@ref)

# References

  - $(ref_dict[:zhao2020sword])
"""
function Sword(; eta_min::Real, K::Integer, eps::Real,
               obj::AbstractOnlineObjective = LogWealth(),
               eset::Option{<:BoundedAllocationSet} = nothing,
               proj::EuclideanProjection = EuclideanProjection())::ExpertMixture
    experts = rate_grid_experts(eta_min, K, obj)
    return ExpertMixture(; experts = experts, alg = ExponentiatedGradient(; eta = eps),
                         eset = eset, proj = proj, grad = BlendPoint())
end
export SwitchingWeighting, SwitchingPortfolio, Ader, Sword
