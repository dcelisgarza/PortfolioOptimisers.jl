"""
$(DocStringExtensions.TYPEDEF)

Moves each weight to the other entries with probability ``\\gamma`` after the wealth step, the switching weighting of Singer (1997) (SP).

As the weighting of an [`ExpertMixture`](@ref) it acts on the expert-return vector. There the wealth step is the buy-and-hold weighting, and the share is the switch from one expert to another. Over the single-asset constant rebalanced portfolios it is the paper's switching portfolio, [`SwitchingPortfolio`](@ref). On the head over the assets it gives the same numbers, because the blend of the unit experts is the weight vector itself. The rule keeps no state and reads no rows. The package does not build the paper's second version, whose rate falls the longer the process holds one asset.

# Mathematical definition

```math
\\begin{align}
w_{t+1, k} &= (1 - \\gamma)\\, \\hat{w}_{t, k} + \\frac{\\gamma}{K - 1} \\sum_{j \\neq k} \\hat{w}_{t, j} \\\\
&= \\left(1 - \\frac{\\gamma K}{K - 1}\\right) \\hat{w}_{t, k} + \\frac{\\gamma}{K - 1}\\,.
\\end{align}
```

Where:

  - ``w_{t+1, k}``: Weight of entry ``k`` during period ``t + 1``.
  - $(math_dict[:w_hat_t_padj])
  - ``\\hat{w}_{t, k}``: Entry ``k`` of the Price-Adjusted Allocation.
  - ``\\gamma``: Switching rate, in ``[0, 1]``.
  - ``K``: Length of the weight vector. It is the asset count on the head and the expert count in a mixture.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:t_period])

The first line is the paper's equation 4 written in weights. The second line is its equation 6, because the entries of ``\\hat{\\boldsymbol{w}}_t`` sum to one. The paper notes that the update resembles the fixed share of Herbster and Warmuth. The share is a stochastic matrix, so ``\\boldsymbol{w}_{t+1}`` lies on the simplex, and the Euclidean projection onto the simplex returns it unchanged. At ``\\gamma = 0`` the share is the identity. At ``\\gamma = (K - 1) / K`` every entry of ``\\boldsymbol{w}_{t+1}`` is ``1 / K``.

# Algorithm

 1. Compute `q`, the Price-Adjusted Allocation of `w` over `x`.
 2. Set `K` to the length of `q`.
 3. When `K` is one, set `p` to `q`, because one entry has no other entry to switch to. Otherwise, set `p` to the fixed share of `q` at the rate `gamma`.
 4. Project `p` onto the Allocation Set with `alg.proj`, from `q`. Return the carrier `st` unchanged, and the projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SwitchingWeighting(; gamma::Real = 1/3, proj::EuclideanProjection = EuclideanProjection()) -> SwitchingWeighting

Keywords correspond to the struct's fields. The default rate is ``1/3``, the rate of the paper's experiments without transaction costs.

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
  - [`SwitchingPortfolio`](@ref): the mixture that this weighting builds over the unit experts.
  - [`BuyAndHold`](@ref): the weighting at ``\\gamma = 0``.
  - [`ConstantRebalancedPortfolio`](@ref): over the unit experts, the uniform one is the weighting at ``\\gamma = (K - 1) / K``.

# References

  - $(ref_dict[:singer1997])
"""
struct SwitchingWeighting{T1 <: Real, T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The switching rate, the share of each weight that moves to the other entries every period.
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

Builds Singer's (1997) switching portfolio, the [`ExpertMixture`](@ref) under [`SwitchingWeighting`](@ref) over the `N` single-asset [`ConstantRebalancedPortfolio`](@ref)s of the pinned universe (SP).

The paper's portfolio is the Bayesian mixture over the paths of a hidden process. The process holds one asset, and each period it switches to one of the others with probability ``\\gamma``. The weight of the mixture on a unit expert is the posterior probability that the process holds that asset in the next period.

A view of the head onto a subset of the assets drops no expert. The unit expert of an excluded asset has no mass on the kept assets, so its constant rebalanced portfolio becomes the uniform allocation over them. The view then runs with one uniform expert for each excluded asset, and it is not the switching portfolio over the kept assets. Construct the portfolio over the assets that it will run on.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{h}_k(t) &= \\boldsymbol{e}_k\\,, \\\\
\\boldsymbol{r}_t &= \\boldsymbol{x}_t\\,, \\\\
\\boldsymbol{w}_t &= \\sum_{k=1}^{N} p_{t, k}\\, \\boldsymbol{e}_k = \\boldsymbol{p}_t\\,.
\\end{align}
```

Where:

  - $(math_dict[:h_kt_expert])
  - ``\\boldsymbol{e}_k``: Unit vector of asset ``k``.
  - $(math_dict[:r_t_expert])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:p_t_expert])
  - $(math_dict[:N])
  - $(math_dict[:t_period])

So ``\\boldsymbol{p}_t`` takes the switching step over ``\\boldsymbol{x}_t``, and the portfolio holds the path that the switching step holds on the head over the assets.

# Arguments

  - `N`: The number of assets of the pinned universe, one unit expert each.
  - `gamma`: The switching rate.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry of the projection of the blend onto the head's Allocation Set.

# Validation

  - `N >= 1`. A `DomainError` is thrown otherwise.
  - `0 <= gamma <= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> sp = SwitchingPortfolio(; N = 3);

julia> length(sp.experts), sp.alg.gamma
(3, 0.3333333333333333)
```

# Related

  - [`ExpertMixture`](@ref)
  - [`SwitchingWeighting`](@ref): the weighting, which holds the same path on the head over the assets.
  - [`ConstantRebalancedPortfolio`](@ref): the uniform one is this portfolio at `gamma = (N - 1) / N`.
  - [`BuyAndHold`](@ref): this portfolio at `gamma = 0`.

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

Builds the `K` [`GradientProjection`](@ref) experts of [`Ader`](@ref) and [`Sword`](@ref) on a geometric grid of rates, each on the objective `obj`.

Every rate takes the numeric type of `eta_min`.

# Mathematical definition

```math
\\begin{align}
\\eta_k &= 2^{k - 1} \\eta_{\\min}\\,, \\quad k = 1, \\ldots, K\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_k_grid])
  - $(math_dict[:eta_min_grid])
  - $(math_dict[:K_expert])

The largest rate is ``2^{K - 1}`` times the smallest.

# Arguments

  - `eta_min`: The smallest rate of the grid.
  - `K`: The number of experts.
  - `obj`: The objective that every expert steps on.

# Validation

  - `eta_min > 0`, `K >= 1`. A `DomainError` is thrown otherwise.

# Returns

  - `experts::Vector{<:MirrorDescent}`: The `K` experts. Expert ``k`` takes the Euclidean gradient step at the rate ``\\eta_k``.

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

Builds the improved Ader of Zhang, Lu and Zhou (2018, Algorithms 3 and 4), an [`ExpertMixture`](@ref) of `K` [`GradientProjection`](@ref) experts on a geometric grid of rates.

Every expert takes one Euclidean step from its own iterate, on the gradient at the allocation that the mixture played, the [`BlendPoint`](@ref). The weighting is [`ExponentiatedGradient`](@ref) at the rate `eps`, from the paper's start over the experts. That start gives the largest weight to the smallest rate.

The weighting sets the weights of the experts from their log wealth under every objective `obj`. So the mixture is the paper's algorithm only under the default [`LogWealth`](@ref) objective. Under another objective, such as [`RiskLoss`](@ref), the experts step on that objective, but the weights follow the log wealth of the experts. The paper tunes the grid and the rate for a known horizon. An online rule does not know the horizon, so the three values are the caller's.

# Mathematical definition

```math
\\begin{align}
\\eta_k &= 2^{k - 1} \\eta_{\\min}\\,, \\\\
p_{1, k} &= \\frac{C}{k (k + 1)}\\,, \\\\
C &= 1 + \\frac{1}{K}\\,, \\\\
\\boldsymbol{w}_t &= \\sum_{k=1}^{K} p_{t, k}\\, \\boldsymbol{h}_k(t)\\,, \\\\
\\boldsymbol{h}_k(t + 1) &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{h}_k(t) - \\eta_k \\nabla f_t(\\boldsymbol{w}_t) \\right)\\,, \\\\
p_{t+1, k} &= \\frac{p_{t, k} \\exp\\left(-\\alpha\\, \\ell_t(\\boldsymbol{h}_k(t))\\right)}{\\sum_{j=1}^{K} p_{t, j} \\exp\\left(-\\alpha\\, \\ell_t(\\boldsymbol{h}_j(t))\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_k_grid])
  - $(math_dict[:eta_min_grid])
  - $(math_dict[:p_t_expert])
  - ``C``: Normaliser of the start over the experts.
  - $(math_dict[:K_expert])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:h_kt_expert])
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:f_t_online])
  - $(math_dict[:ell_t_surr])
  - ``\\alpha``: Learning rate of the weighting.
  - $(math_dict[:t_period])

The start sums to one, because ``\\sum_{k=1}^{K} 1 / (k (k + 1)) = K / (K + 1)``. Under the log-wealth loss ``f_t(\\boldsymbol{w}) = -\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle``, the blend gives ``\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle = \\langle \\boldsymbol{p}_t, \\boldsymbol{r}_t \\rangle``. The weight update is then the exponentiated-gradient step over the expert-return vector,

```math
\\begin{align}
p_{t+1, k} &= \\frac{p_{t, k} \\exp\\left(\\alpha\\, r_{t, k} / \\langle \\boldsymbol{p}_t, \\boldsymbol{r}_t \\rangle\\right)}{\\sum_{j=1}^{K} p_{t, j} \\exp\\left(\\alpha\\, r_{t, j} / \\langle \\boldsymbol{p}_t, \\boldsymbol{r}_t \\rangle\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:r_t_expert])
  - $(math_dict[:x_t_rel])

Theorem 4 of the paper sets

```math
\\begin{align}
\\eta_{\\min} &= \\frac{D}{G} \\sqrt{\\frac{7}{2 T}}\\,, \\\\
K &= \\left\\lceil \\tfrac{1}{2} \\log_2 \\left(1 + \\frac{4 T}{7}\\right) \\right\\rceil + 1\\,, \\\\
\\alpha &= \\sqrt{\\frac{2}{T G^2 D^2}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:D_diam])
  - $(math_dict[:G_gradbound])
  - $(math_dict[:T_regret])
  - $(math_dict[:P_T_path])

Under these values the dynamic regret against every comparator sequence is ``O(\\sqrt{T (1 + P_T)})``. On the simplex ``D = \\sqrt{2}``. Under the log-wealth loss the smallest ``G`` is ``\\max_t \\lVert \\boldsymbol{x}_t \\rVert_2 / \\min_i x_{t, i}``, which the vertex of the smallest price relative attains.

# Arguments

  - `eta_min`: The smallest rate of the grid.
  - `K`: The number of experts.
  - `eps`: The weighting's rate, the paper's ``\\alpha``.
  - `obj`: The objective every expert steps on.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry of the projection of the blend onto the head's Allocation Set.

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

Builds the small-loss Sword of Zhao, Zhang, Zhang and Zhou (2020, Theorem 5), an [`ExpertMixture`](@ref) of `K` [`GradientProjection`](@ref) experts on a geometric grid of rates, from the uniform start.

The experts, the [`BlendPoint`](@ref) and the [`ExponentiatedGradient`](@ref) weighting are those of [`Ader`](@ref). The two differ in the start over the experts, which is uniform here, and in the grid that the paper tunes. The paper's meta-algorithm is the exponentially weighted forecaster on the linearised loss, with no optimism.

As for [`Ader`](@ref), the weighting sets the weights of the experts from their log wealth under every objective `obj`. So the mixture is the paper's algorithm only under the default [`LogWealth`](@ref) objective. The paper tunes the grid and the rate for a known horizon and a known cumulative loss. An online rule knows neither, so the three values are the caller's.

The package builds neither of the paper's two other forms. In the gradient-variation form, the experts take the extra-gradient step and the weighting carries an optimistic hint. The best-of-both-worlds form runs a second meta-algorithm that learns the hint.

# Mathematical definition

```math
\\begin{align}
\\eta_k &= 2^{k - 1} \\eta_{\\min}\\,, \\\\
p_{1, k} &= \\frac{1}{K}\\,, \\\\
\\boldsymbol{w}_t &= \\sum_{k=1}^{K} p_{t, k}\\, \\boldsymbol{h}_k(t)\\,, \\\\
\\boldsymbol{h}_k(t + 1) &= \\mathrm{Proj}_{\\mathcal{W}}\\left( \\boldsymbol{h}_k(t) - \\eta_k \\nabla f_t(\\boldsymbol{w}_t) \\right)\\,, \\\\
p_{t+1, k} &= \\frac{p_{t, k} \\exp\\left(-\\varepsilon\\, \\ell_t(\\boldsymbol{h}_k(t))\\right)}{\\sum_{j=1}^{K} p_{t, j} \\exp\\left(-\\varepsilon\\, \\ell_t(\\boldsymbol{h}_j(t))\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_k_grid])
  - $(math_dict[:eta_min_grid])
  - $(math_dict[:p_t_expert])
  - $(math_dict[:K_expert])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:h_kt_expert])
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:f_t_online])
  - $(math_dict[:ell_t_surr])
  - ``\\varepsilon``: Learning rate of the weighting.
  - $(math_dict[:t_period])

The paper writes the forecaster on ``\\langle \\nabla f_t(\\boldsymbol{w}_t), \\boldsymbol{h}_k(t) \\rangle``. That term differs from ``\\ell_t(\\boldsymbol{h}_k(t))`` by a term that is the same for every expert, so the two give the same weights. Under the log-wealth loss ``f_t(\\boldsymbol{w}) = -\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle``, the weight update is the exponentiated-gradient step over the expert-return vector,

```math
\\begin{align}
p_{t+1, k} &= \\frac{p_{t, k} \\exp\\left(\\varepsilon\\, r_{t, k} / \\langle \\boldsymbol{p}_t, \\boldsymbol{r}_t \\rangle\\right)}{\\sum_{j=1}^{K} p_{t, j} \\exp\\left(\\varepsilon\\, r_{t, j} / \\langle \\boldsymbol{p}_t, \\boldsymbol{r}_t \\rangle\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:r_t_expert])
  - $(math_dict[:x_t_rel])

Theorem 5 of the paper sets

```math
\\begin{align}
\\eta_{\\min} &= \\sqrt{\\frac{D}{16 L G T}}\\,, \\\\
K &= \\left\\lceil \\tfrac{1}{2} \\log_2 \\frac{G T}{D L} \\right\\rceil + 1\\,, \\\\
\\varepsilon &= \\sqrt{\\frac{2 + \\ln K}{D^2 F_T^{\\boldsymbol{w}}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:D_diam])
  - $(math_dict[:G_gradbound])
  - ``L``: Smoothness constant of every online loss.
  - $(math_dict[:T_regret])
  - ``F_T^{\\boldsymbol{w}} = \\sum_{t=1}^{T} f_t(\\boldsymbol{w}_t)``: Cumulative loss of the played allocations.
  - ``F_T = \\sum_{t=1}^{T} f_t(\\boldsymbol{u}_t)``: Cumulative loss of the comparator sequence.
  - $(math_dict[:P_T_path])

Under these values the dynamic regret against every comparator sequence is ``O(\\sqrt{(1 + P_T + F_T)(1 + P_T)})``. The theorem needs online losses that are non-negative and ``L``-smooth on the whole space. The log-wealth loss has a value only where ``\\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle > 0``, so the bound does not apply to it.

# Arguments

  - `eta_min`: The smallest rate of the grid.
  - `K`: The number of experts.
  - `eps`: The weighting's rate, the paper's ``\\varepsilon``.
  - `obj`: The objective every expert steps on.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry of the projection of the blend onto the head's Allocation Set.

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
