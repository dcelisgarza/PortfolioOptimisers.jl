"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the two confidence formulations of [`ConfidenceWeightedMeanReversion`](@ref): the paper's two variants, which differ in how the constraint on the portfolio return reads the belief's spread.

# Interfaces

In order to implement a new formulation, subtype `AbstractConfidenceUpdate` and implement:

  - `confidence_step(formulation::AbstractConfidenceUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real, eps::Real) -> Real`: The Lagrange multiplier ``\\lambda_{t+1}`` of the period, the non-negative root of the formulation's quadratic, `0` when the constraint already holds.
  - `confidence_gain(formulation::AbstractConfidenceUpdate, lambda::Real, phi::Real, V::Real) -> Real`: The scalar the squared price relative is scaled by when it is added to the inverse belief covariance.

## Arguments

  - `formulation`: The formulation.
  - `V`: ``\\boldsymbol{x}_t^\\intercal \\Sigma_t \\boldsymbol{x}_t``, the belief's variance along the price relative.
  - `phi`: The paper's ``\\phi = \\Phi^{-1}(\\theta)``, the confidence quantile.
  - `xbar`: ``\\boldsymbol{1}^\\intercal \\Sigma_t \\boldsymbol{x}_t / \\boldsymbol{1}^\\intercal \\Sigma_t \\boldsymbol{1}``, the belief-weighted mean of the price relative.
  - `W`: ``\\boldsymbol{x}_t^\\intercal \\Sigma_t \\boldsymbol{1}``.
  - `M`: ``\\langle \\boldsymbol{\\mu}_t, \\boldsymbol{x}_t \\rangle``, the portfolio's return on the period.
  - `eps`: The reversion threshold.
  - `lambda`: The multiplier the step found.

## Returns

  - `lambda::Real`, or `gain::Real`.

# Related

  - [`VarianceUpdate`](@ref)
  - [`StandardDeviationUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
abstract type AbstractConfidenceUpdate <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The formulation of CWMR-Var: the constraint ``\\epsilon \\geq \\langle \\boldsymbol{\\mu}, \\boldsymbol{x}_t \\rangle + \\phi\\, \\boldsymbol{x}_t^\\intercal \\Sigma \\boldsymbol{x}_t`` reads the belief's **variance** along the price relative, and the inverse covariance gains ``2 \\lambda_{t+1} \\phi\\, \\mathrm{diag}(\\boldsymbol{x}_t)^2``.

# Examples

```jldoctest
julia> VarianceUpdate()
VarianceUpdate()
```

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`StandardDeviationUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
struct VarianceUpdate <: AbstractConfidenceUpdate end
"""
$(DocStringExtensions.TYPEDEF)

The formulation of CWMR-Stdev: the constraint ``\\epsilon \\geq \\langle \\boldsymbol{\\mu}, \\boldsymbol{x}_t \\rangle + \\phi \\sqrt{\\boldsymbol{x}_t^\\intercal \\Sigma \\boldsymbol{x}_t}`` reads the belief's **standard deviation** along the price relative, and the inverse covariance gains ``\\lambda_{t+1} \\phi\\, \\mathrm{diag}(\\boldsymbol{x}_t)^2 / \\sqrt{U_{t+1}}``, with ``\\sqrt{U_{t+1}} = \\left( -\\lambda_{t+1} \\phi V_t + \\sqrt{\\lambda_{t+1}^2 \\phi^2 V_t^2 + 4 V_t} \\right) / 2``.

# Examples

```jldoctest
julia> StandardDeviationUpdate()
StandardDeviationUpdate()
```

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`VarianceUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
struct StandardDeviationUpdate <: AbstractConfidenceUpdate end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The largest non-negative root of ``a \\lambda^2 + b \\lambda + c = 0``, and `0` when the quadratic has no real root or no non-negative one; a degenerate quadratic, `a == 0`, is solved as the line it is.

# Related

  - [`confidence_step`](@ref)
"""
function nonneg_quadratic_root(a::Real, b::Real, c::Real)
    z = zero(a / b)
    if iszero(a)
        return iszero(b) ? z : max(z, -c / b)
    end
    disc = b^2 - 4 * a * c
    if disc < zero(disc)
        return z
    end
    s = sqrt(disc)
    return max(z, (-b + s) / (2 * a), (-b - s) / (2 * a))
end
"""
    confidence_step(::VarianceUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real, eps::Real)
    confidence_step(::StandardDeviationUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real, eps::Real)

The Lagrange multiplier of a confidence-weighted step under each formulation: the non-negative root of the paper's quadratic in ``\\lambda``.

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`nonneg_quadratic_root`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
function confidence_step(::VarianceUpdate, V::Real, phi::Real, xbar::Real, W::Real, M::Real,
                         eps::Real)
    a = 2 * phi * V^2 - 2 * phi * xbar * V * W
    b = 2 * phi * eps * V - 2 * phi * V * M + V - xbar * W
    c = eps - M - phi * V
    return nonneg_quadratic_root(a, b, c)
end
function confidence_step(::StandardDeviationUpdate, V::Real, phi::Real, xbar::Real, W::Real,
                         M::Real, eps::Real)
    h = V - xbar * W + phi^2 * V / 2
    a = h^2 - phi^4 * V^2 / 4
    b = 2 * (eps - M) * h
    c = (eps - M)^2 - phi^2 * V
    return nonneg_quadratic_root(a, b, c)
end
"""
    confidence_gain(::VarianceUpdate, lambda::Real, phi::Real, V::Real)
    confidence_gain(::StandardDeviationUpdate, lambda::Real, phi::Real, V::Real)

The scalar the squared price relative is scaled by in the inverse-covariance update: ``2 \\lambda \\phi`` under the variance formulation, ``\\lambda \\phi / \\sqrt{U}`` under the standard-deviation one.

# Related

  - [`AbstractConfidenceUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversion`](@ref)
"""
function confidence_gain(::VarianceUpdate, lambda::Real, phi::Real, ::Real)
    return 2 * lambda * phi
end
function confidence_gain(::StandardDeviationUpdate, lambda::Real, phi::Real, V::Real)
    u = (-lambda * phi * V + sqrt(lambda^2 * phi^2 * V^2 + 4 * V)) / 2
    return lambda * phi / u
end
"""
$(DocStringExtensions.TYPEDEF)

The carrier of [`ConfidenceWeightedMeanReversion`](@ref): the diagonal of the belief covariance over the portfolio weights.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ConfidenceWeightedMeanReversion`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct ConfidenceWeightedMeanReversionState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The diagonal of ``\\Sigma_t``, the belief covariance over the weights, one entry per asset, written in place; seeded at ``1 / N^2`` per asset and rescaled to trace ``1 / N^2`` after every step, as the paper's algorithm states.
    """
    sigma
end
function merge_states(::ConfidenceWeightedMeanReversionState,
                      ::ConfidenceWeightedMeanReversionState)
    return throw(ArgumentError("a `ConfidenceWeightedMeanReversionState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::ConfidenceWeightedMeanReversionState)
    return ConfidenceWeightedMeanReversionState(x.n, copy(x.sigma))
end
function port_opt_view(x::ConfidenceWeightedMeanReversionState, i, args...)
    return ConfidenceWeightedMeanReversionState(x.n, x.sigma[i])
end
"""
$(DocStringExtensions.TYPEDEF)

The confidence weighted mean reversion of Li, Hoi, Zhao and Gopalkrishnan (2011, 2013): the portfolio is a Gaussian belief ``\\mathcal{N}(\\boldsymbol{\\mu}_t, \\Sigma_t)`` over the weights, moved the least in Gaussian relative entropy so that the next period's return is below `eps` with confidence ``\\theta`` (CWMR).

# Mathematical definition

With ``M_t = \\langle \\boldsymbol{\\mu}_t, \\boldsymbol{x}_t \\rangle``, ``V_t = \\boldsymbol{x}_t^\\intercal \\Sigma_t \\boldsymbol{x}_t``, ``W_t = \\boldsymbol{x}_t^\\intercal \\Sigma_t \\boldsymbol{1}`` and ``\\bar{x}_t = W_t / \\boldsymbol{1}^\\intercal \\Sigma_t \\boldsymbol{1}``,

```math
\\begin{align}
\\boldsymbol{\\mu}_{t+1} &= \\mathrm{Proj}\\left( \\boldsymbol{\\mu}_t - \\lambda_{t+1} \\Sigma_t \\left( \\boldsymbol{x}_t - \\bar{x}_t \\boldsymbol{1} \\right) \\right)\\,,\\\\
\\Sigma_{t+1}^{-1} &= \\Sigma_t^{-1} + \\gamma_{t+1}\\, \\mathrm{diag}(\\boldsymbol{x}_t)^2\\,,\\quad
\\Sigma_{t+1} \\leftarrow \\frac{\\Sigma_{t+1}}{N^2\\, \\mathrm{tr}(\\Sigma_{t+1})}\\,,
\\end{align}
```

where ``\\lambda_{t+1}`` is the non-negative root of the formulation's quadratic — ``0`` whenever the constraint already holds, so the rule is passive on a period whose return was below `eps` — and ``\\gamma_{t+1}`` is ``2 \\lambda_{t+1} \\phi`` under [`VarianceUpdate`](@ref) and ``\\lambda_{t+1} \\phi / \\sqrt{U_{t+1}}`` under [`StandardDeviationUpdate`](@ref). The belief covariance is diagonal throughout, which is all the paper keeps, so the carrier is one vector: the confidence in each weight. The mean of the belief is the allocation held, so the rule reads `w` and carries the covariance alone; the covariance is the rule's own and never a Prior's, because it is a belief over weights and not a moment of returns.

The rule's geometry is the Gaussian relative entropy over ``(\\boldsymbol{\\mu}, \\Sigma)``. The Allocation Set constrains the mean alone, and the paper's own projection of the mean onto the simplex is Euclidean, so the `proj` slot is bound to [`EuclideanProjection`](@ref); the covariance is rescaled to its trace, not projected. The rule sells what just rose: it is a total bet on single-period mean reversion, like [`PassiveAggressiveMeanReversion`](@ref), with a step that shrinks as the belief in a weight sharpens. The constraint is the 2013 text's, on the return ``\\langle \\boldsymbol{\\mu}, \\boldsymbol{x}_t \\rangle`` with ``\\epsilon = 0.5``; the 2011 text writes it on ``\\log \\langle \\boldsymbol{\\mu}, \\boldsymbol{x}_t \\rangle`` with ``\\epsilon = -0.5``, the same step with the price relative divided by ``M_t`` and the threshold compared with ``\\log M_t``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConfidenceWeightedMeanReversion(;
        eps::Real = 0.5,
        phi::Real = 2,
        formulation::AbstractConfidenceUpdate = VarianceUpdate(),
        proj::EuclideanProjection = EuclideanProjection()
    ) -> ConfidenceWeightedMeanReversion

Keywords correspond to the struct's fields, and the defaults are the paper's: `phi = 2` is the confidence parameter the paper sets without tuning and reports as not decisive. `VarianceUpdate()` and `StandardDeviationUpdate()` are the paper's CWMR-Var and CWMR-Stdev.

## Validation

  - `eps >= 0`, `phi >= 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ConfidenceWeightedMeanReversion()
ConfidenceWeightedMeanReversion
          eps ┼ Float64: 0.5
          phi ┼ Int64: 2
  formulation ┼ VarianceUpdate()
         proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`AbstractConfidenceUpdate`](@ref)
  - [`ConfidenceWeightedMeanReversionState`](@ref)
  - [`PassiveAggressiveMeanReversion`](@ref)

# References

  - $(ref_dict[:li2011cwmr])
  - $(ref_dict[:li2013cwmr])
"""
struct ConfidenceWeightedMeanReversion{T1 <: Real, T2 <: Real,
                                       T3 <: AbstractConfidenceUpdate,
                                       T4 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Reversion threshold: the belief is moved only when the last period's portfolio return, plus the confidence margin, exceeded it.
    """
    eps::T1
    """
    The confidence quantile ``\\phi = \\Phi^{-1}(\\theta)``, which scales the margin the belief's spread adds to the constraint.
    """
    phi::T2
    """
    The formulation, one of the paper's two variants.
    """
    formulation::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function ConfidenceWeightedMeanReversion(eps::Real, phi::Real,
                                             formulation::AbstractConfidenceUpdate,
                                             proj::EuclideanProjection)
        @argcheck(eps >= zero(eps), DomainError(eps, "eps must be non-negative"))
        @argcheck(phi >= zero(phi), DomainError(phi, "phi must be non-negative"))
        return new{typeof(eps), typeof(phi), typeof(formulation), typeof(proj)}(eps, phi,
                                                                                formulation,
                                                                                proj)
    end
end
function ConfidenceWeightedMeanReversion(; eps::Real = 0.5, phi::Real = 2,
                                         formulation::AbstractConfidenceUpdate = VarianceUpdate(),
                                         proj::EuclideanProjection = EuclideanProjection())::ConfidenceWeightedMeanReversion
    return ConfidenceWeightedMeanReversion(eps, phi, formulation, proj)
end
function rule_state_seed(::ConfidenceWeightedMeanReversion, w::AbstractVector)
    N = length(w)
    return ConfidenceWeightedMeanReversionState(0, fill(one(eltype(w)) / N^2, N))
end
function online_update!(alg::ConfidenceWeightedMeanReversion,
                        st::ConfidenceWeightedMeanReversionState, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    sigma = st.sigma
    M = LinearAlgebra.dot(w, x)
    V = LinearAlgebra.dot(sigma, x .^ 2)
    W = LinearAlgebra.dot(sigma, x)
    xbar = W / sum(sigma)
    lam = confidence_step(alg.formulation, V, alg.phi, xbar, W, M, alg.eps)
    q = w .- lam .* sigma .* (x .- xbar)
    gain = confidence_gain(alg.formulation, lam, alg.phi, V)
    sigma .= inv.(inv.(sigma) .+ gain .* x .^ 2)
    sigma ./= length(sigma)^2 * sum(sigma)
    wn = project(alg.proj, set, q, price_adjusted_allocation(w, x))
    return ConfidenceWeightedMeanReversionState(st.n + 1, sigma), wn
end
"""
$(DocStringExtensions.TYPEDEF)

The anti-correlation rule of Borodin, El-Yaniv and Gogan (2004): wealth is transferred from an asset to another whose recent growth lagged it and whose log price relatives over the latest window are positively correlated with the first's over the window before, so that the second is expected to emulate the first's past growth (Anticor).

# Mathematical definition

With ``L^{(1)}`` and ``L^{(2)}`` the log price relatives of the two consecutive windows of `window` rows ending at ``t``, ``\\boldsymbol{\\mu}_k`` and ``\\boldsymbol{\\sigma}_k`` their column means and sample standard deviations, and the lagged cross-window correlation

```math
\\begin{align}
M_{\\mathrm{cor}}(i, j) &= \\frac{\\frac{1}{w - 1} \\sum_k \\left( L^{(1)}_{k, i} - \\mu_{1, i} \\right) \\left( L^{(2)}_{k, j} - \\mu_{2, j} \\right)}{\\sigma_{1, i}\\, \\sigma_{2, j}}\\,,
\\end{align}
```

zero where either standard deviation is, asset ``i`` claims a transfer to ``j \\neq i`` when ``\\mu_{2, i} \\geq \\mu_{2, j}`` and ``M_{\\mathrm{cor}}(i, j) > 0``:

```math
\\begin{align}
\\mathrm{claim}_{i \\to j} &= M_{\\mathrm{cor}}(i, j) + \\max(0, -M_{\\mathrm{cor}}(i, i)) + \\max(0, -M_{\\mathrm{cor}}(j, j))\\,,\\quad
\\mathrm{transfer}_{i \\to j} = w_{t, i} \\frac{\\mathrm{claim}_{i \\to j}}{\\sum_j \\mathrm{claim}_{i \\to j}}\\,,\\\\
w_{t+1, i} &= w_{t, i} - \\sum_{j \\neq i} \\mathrm{transfer}_{i \\to j} + \\sum_{j \\neq i} \\mathrm{transfer}_{j \\to i}\\,.
\\end{align}
```

An asset's own negative autocorrelation raises every claim it takes part in. The transfers conserve the budget and move at most ``w_{t, i}`` out of asset ``i``, so the raw step lies in the simplex by construction and the rule projects nothing on the default Allocation Set; on a stated set the step is projected in the Euclidean geometry, the `proj` slot's bound, which is the identity wherever the step already lies in the set. Until the head holds ``2 w`` rows the rule continues from the allocation it is handed. The correlation is recomputed from the rows the head holds every period and nothing else is carried, so the rule's carrier is `nothing` and `rows_needed` is ``2 w``. The paper's headline is the uniform buy-and-hold mixture of this rule over `window` in `2:30`, the [`ExpertMixture`](@ref) over those experts.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AntiCorrelation(; window::Integer = 5, proj::EuclideanProjection = EuclideanProjection()) -> AntiCorrelation

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`: a window of one row has no sample standard deviation. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> AntiCorrelation()
AntiCorrelation
  window ┼ Int64: 5
    proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`ExpertMixture`](@ref)
  - [`lagged_window_correlation`](@ref)

# References

  - $(ref_dict[:borodin2004])
"""
struct AntiCorrelation{T1 <: Integer, T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The window ``w``: the rule reads the last ``2 w`` rows, two consecutive windows of ``w``.
    """
    window::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function AntiCorrelation(window::Integer, proj::EuclideanProjection)
        @argcheck(window >= 2, DomainError(window, "window must be at least 2"))
        return new{typeof(window), typeof(proj)}(window, proj)
    end
end
function AntiCorrelation(; window::Integer = 5,
                         proj::EuclideanProjection = EuclideanProjection())::AntiCorrelation
    return AntiCorrelation(window, proj)
end
function rows_needed(alg::AntiCorrelation)
    return 2 * alg.window
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The lagged cross-window correlation of two consecutive windows of log price relatives, `L1` and `L2`, both `window × assets`: entry `(i, j)` is the correlation of asset `i`'s column of the first window with asset `j`'s column of the second, with sample moments over `window - 1`, and zero where either column is constant. Also answers the column means of the second window, which rank the assets.

# Related

  - [`AntiCorrelation`](@ref)
  - [`anticorrelation_claims`](@ref)
"""
function lagged_window_correlation(L1::AbstractMatrix, L2::AbstractMatrix)
    mu1 = vec(Statistics.mean(L1; dims = 1))
    mu2 = vec(Statistics.mean(L2; dims = 1))
    s1 = vec(Statistics.std(L1; dims = 1))
    s2 = vec(Statistics.std(L2; dims = 1))
    cov = transpose(L1 .- transpose(mu1)) * (L2 .- transpose(mu2)) ./ (size(L1, 1) - 1)
    scale = s1 * transpose(s2)
    cor = cov ./ scale
    for k in eachindex(cor, scale)
        if iszero(scale[k])
            cor[k] = zero(eltype(cor))
        end
    end
    return cor, mu2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The claim matrix of an anti-correlation step: entry `(i, j)` is asset `i`'s claim to transfer wealth to asset `j`, positive only where `i` grew at least as much as `j` over the latest window and the lagged correlation is positive.

# Related

  - [`AntiCorrelation`](@ref)
  - [`lagged_window_correlation`](@ref)
  - [`wealth_transfer`](@ref)
"""
function anticorrelation_claims(cor::AbstractMatrix, mu2::AbstractVector)
    N = length(mu2)
    claim = zeros(eltype(cor), N, N)
    for i in 1:N, j in 1:N
        if i != j && mu2[i] >= mu2[j] && cor[i, j] > zero(eltype(cor))
            claim[i, j] = cor[i, j] +
                          max(zero(eltype(cor)), -cor[i, i]) +
                          max(zero(eltype(cor)), -cor[j, j])
        end
    end
    return claim
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The allocation after the transfers a claim matrix names: each asset moves its held weight out in proportion to its claims, and receives what the others move to it, so the budget is conserved.

# Related

  - [`AntiCorrelation`](@ref)
  - [`anticorrelation_claims`](@ref)
"""
function wealth_transfer(w::AbstractVector, claim::AbstractMatrix)
    q = copy(w)
    for i in eachindex(w)
        total = sum(view(claim, i, :))
        if total > zero(total)
            out = w[i] .* view(claim, i, :) ./ total
            q .+= out
            q[i] -= sum(out)
        end
    end
    return q
end
function online_update!(alg::AntiCorrelation, st, w::AbstractVector, x::AbstractVector,
                        rows, set::AbstractAllocationSet)
    wn = alg.window
    T = size(rows, 1)
    q = if T < 2 * wn
        w
    else
        L = log1p.(view(rows, (T - 2 * wn + 1):T, :))
        cor, mu2 = lagged_window_correlation(view(L, 1:wn, :),
                                             view(L, (wn + 1):(2 * wn), :))
        wealth_transfer(w, anticorrelation_claims(cor, mu2))
    end
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

The carrier of [`ExpectationMaximisation`](@ref): the allocation the recursion started from, which the online form of the update pulls towards whenever its rate falls.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ExpectationMaximisation`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct ExpectationMaximisationState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The Start Allocation ``\\boldsymbol{w}_1``, the prior of the online form.
    """
    w1
    """
    The Learning-Rate Schedule's statistic, or `nothing`.
    """
    s
end
function merge_states(::ExpectationMaximisationState, ::ExpectationMaximisationState)
    return throw(ArgumentError("an `ExpectationMaximisationState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::ExpectationMaximisationState)
    return ExpectationMaximisationState(x.n, copy(x.w1), copy_column(x.s))
end
function port_opt_view(x::ExpectationMaximisationState, i, args...)
    return ExpectationMaximisationState(x.n, renormalised_view(x.w1, i), copy_column(x.s))
end
"""
$(DocStringExtensions.TYPEDEF)

The expectation-maximisation update of Helmbold, Schapire, Singer and Warmuth (1997), which is Soft-Bayes (Orseau, Lattimore and Legg 2017) formula for formula: every weight is moved to the convex combination of holding and Cover's posterior, at the rate `eta` (EM, Soft-Bayes).

# Mathematical definition

With ``\\boldsymbol{g}_t = \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``,

```math
\\begin{align}
w_{t+1, i} &= w_{t, i} \\left( 1 - \\eta + \\eta\\, g_{t, i} \\right) = (1 - \\eta)\\, w_{t, i} + \\eta\\, \\frac{w_{t, i}\\, x_{t, i}}{\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle}\\,,
\\end{align}
```

which stays on the simplex without normalisation, because ``\\sum_i w_{t, i}\\, g_{t, i} = 1``, and lets no weight grow by more than the factor ``1 + \\eta`` in a period. It is the first-order approximation of [`ExponentiatedGradient`](@ref), derived from the chi-squared divergence, and it is not a mirror step: at ``\\eta = 1`` it is Cover's Bayesian mixture against the single assets, and below one it is that mixture slowed down. The update is written in the online form of Orseau and co-authors' Eq. 14,

```math
\\begin{align}
w_{t+1, i} &= w_{t, i} \\left( 1 - \\eta_t + \\eta_t\\, g_{t, i} \\right) \\frac{\\eta_{t+1}}{\\eta_t} + \\left( 1 - \\frac{\\eta_{t+1}}{\\eta_t} \\right) w_{1, i}\\,,
\\end{align}
```

a fixed-share pull towards the Start Allocation whenever the rate falls, with ``\\eta_t`` and ``\\eta_{t+1}`` read off the `eta` slot through [`learning_rate`](@ref) before and after the row; at a constant rate the pull vanishes and the update is the plain step exactly, and under a schedule that names a restart the update of that period answers the Start Allocation with the carrier back at its seed and the period count kept. The regret against every constant rebalanced portfolio is ``O(\\sqrt{T N \\log N})`` for every non-negative price sequence — no lower bound on the price relatives and no gradient bound, which the exponentiated gradient's bound needs — at ``\\bar{\\eta} = \\sqrt{\\log N / (T m)}``, ``\\eta = \\bar{\\eta} / (1 + \\bar{\\eta})``, with ``m \\leq N`` the number of assets that are ever the period's best. The raw step is positive wherever ``w_t`` is, so on the default Allocation Set the projection is the identity; the `proj` slot is bound to [`EuclideanProjection`](@ref), the geometry nearest the chi-squared one on a stated set. As the weighting of an [`ExpertMixture`](@ref) it is Soft-Bayes over the experts, the setting the 2017 paper is written for.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpectationMaximisation(; eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05, proj::EuclideanProjection = EuclideanProjection()) -> ExpectationMaximisation

Keywords correspond to the struct's fields.

## Validation

  - If `eta` is a number: `0 < eta < 1`: at one the step is Cover's mixture, and above one a weight can turn negative. A `DomainError` is thrown otherwise. A schedule is read at every period, and a scheduled rate outside the interval is the caller's.

# Examples

```jldoctest
julia> ExpectationMaximisation()
ExpectationMaximisation
   eta ┼ Float64: 0.05
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`ExpectationMaximisationState`](@ref)
  - [`ExponentiatedGradient`](@ref)
  - [`AbstractLearningRateSchedule`](@ref)
  - [`learning_rate`](@ref)

# References

  - $(ref_dict[:helmbold1997])
  - $(ref_dict[:orseau2017])
"""
struct ExpectationMaximisation{T1 <: Union{<:Real, <:AbstractLearningRateSchedule},
                               T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    Learning rate in `(0, 1)`, a number or a Learning-Rate Schedule: the share of every weight moved to Cover's posterior each period.
    """
    eta::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function ExpectationMaximisation(eta::Union{<:Real, <:AbstractLearningRateSchedule},
                                     proj::EuclideanProjection)
        if isa(eta, Real)
            @argcheck(zero(eta) < eta < one(eta), DomainError(eta, "eta must be in (0, 1)"))
        end
        return new{typeof(eta), typeof(proj)}(eta, proj)
    end
end
function ExpectationMaximisation(;
                                 eta::Union{<:Real, <:AbstractLearningRateSchedule} = 0.05,
                                 proj::EuclideanProjection = EuclideanProjection())::ExpectationMaximisation
    return ExpectationMaximisation(eta, proj)
end
function rule_state_seed(alg::ExpectationMaximisation, w::AbstractVector)
    return ExpectationMaximisationState(0, copy(w), schedule_state_seed(alg.eta, w))
end
function online_update!(alg::ExpectationMaximisation, st::ExpectationMaximisationState,
                        w::AbstractVector, x::AbstractVector, ::Any,
                        set::AbstractAllocationSet)
    t = st.n + 1
    if restart(alg.eta, t)
        # The period count survives the restart: the schedule's stages are cumulative.
        return ExpectationMaximisationState(t, st.w1, schedule_state_seed(alg.eta, st.w1)),
               copy(st.w1)
    end
    eta_t = learning_rate(alg.eta, t, st)
    g = x ./ LinearAlgebra.dot(w, x)
    q = w .* (one(eta_t) - eta_t .+ eta_t .* g)
    stn = ExpectationMaximisationState(t, st.w1, schedule_update!(alg.eta, st.s, w, x))
    ratio = learning_rate(alg.eta, t + 1, stn) / eta_t
    if !isone(ratio)
        q .= ratio .* q .+ (one(ratio) - ratio) .* st.w1
    end
    return stn, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

The aggregating algorithm of Vovk and Watkins (1998) over a finite set: every weight is multiplied by its period return raised to the rate `eta`, ``w_{t+1} \\propto w_t \\odot x_t^{\\eta}``, then normalised (AA).

At ``\\eta = 1`` it is the wealth weighting of [`BuyAndHold`](@ref) — over experts, the wealth-weighted mixture, and Cover's universal portfolio when the experts are sampled constant rebalanced portfolios — and below one it discounts every period's evidence alike. As the weighting of an [`ExpertMixture`](@ref) the mixture's blend is the linear average of the experts' allocations, which is the aggregating algorithm's own prediction under the logarithmic loss. The rate is fixed; the weighting with the shrinking rate ``1 / \\sqrt{t}`` is [`WeakAggregatingAlgorithm`](@ref). The raw step is a multiplicative update of a non-negative vector, so its geometry is the entropic one, and the `proj` slot is bound to [`EntropicProjection`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AggregatingAlgorithm(; eta::Real = 1, proj::EntropicProjection = EntropicProjection()) -> AggregatingAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - `eta > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> AggregatingAlgorithm()
AggregatingAlgorithm
   eta ┼ Int64: 1
  proj ┴ EntropicProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ExpertMixture`](@ref)
  - [`BuyAndHold`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)

# References

  - $(ref_dict[:vovkwatkins1998])
"""
struct AggregatingAlgorithm{T1 <: Real, T2 <: EntropicProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The rate the period return is raised to; `1` is the wealth weighting.
    """
    eta::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function AggregatingAlgorithm(eta::Real, proj::EntropicProjection)
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        return new{typeof(eta), typeof(proj)}(eta, proj)
    end
end
function AggregatingAlgorithm(; eta::Real = 1,
                              proj::EntropicProjection = EntropicProjection())::AggregatingAlgorithm
    return AggregatingAlgorithm(eta, proj)
end
function online_update!(alg::AggregatingAlgorithm, st, w::AbstractVector, x::AbstractVector,
                        ::Any, set::AbstractAllocationSet)
    q = w .* x .^ alg.eta
    return st, project(alg.proj, set, q, price_adjusted_allocation(w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

The carrier of [`TopK`](@ref) and [`WeakAggregatingAlgorithm`](@ref): the cumulative log wealth of every asset — every expert, on an [`ExpertMixture`](@ref)'s slot — and the allocation the recursion started from.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`TopK`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct CumulativeWealthState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The cumulative log wealth ``G_t = \\sum_{s \\leq t} \\log x_{s}`` of every asset, written in place.
    """
    G
    """
    The Start Allocation, the prior a weak aggregating step weights from.
    """
    p0
end
function merge_states(::CumulativeWealthState, ::CumulativeWealthState)
    return throw(ArgumentError("a `CumulativeWealthState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::CumulativeWealthState)
    return CumulativeWealthState(x.n, copy(x.G), copy(x.p0))
end
function port_opt_view(x::CumulativeWealthState, i, args...)
    return CumulativeWealthState(x.n, x.G[i], renormalised_view(x.p0, i))
end
"""
$(DocStringExtensions.TYPEDEF)

The top-``k`` selection of Li, Hoi and Gopalkrishnan (2011): the ``k`` assets — the ``k`` experts, on an [`ExpertMixture`](@ref)'s slot — of greatest cumulative wealth so far, held in equal weight, which is the CORN-K combination of that paper's correlation-driven experts (CORN-K).

The rule selects rather than projects: the raw step is ``1 / k`` on the ``k`` assets of greatest ``G_t = \\sum_{s \\leq t} \\log x_s`` and zero elsewhere, equal wealth breaking ties by index, and lies in the simplex by construction, so on the default Allocation Set the projection is the identity; on a stated set the step is projected in the Euclidean geometry, the `proj` slot's bound. At ``k = 1`` it follows the single best performer, the limit of [`WeakAggregatingAlgorithm`](@ref) as its rate vanishes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TopK(; k::Integer = 5, proj::EuclideanProjection = EuclideanProjection()) -> TopK

Keywords correspond to the struct's fields, and `k = 5` is the paper's.

## Validation

  - `k >= 1`. A `DomainError` is thrown otherwise.
  - `k` is at most the number of assets or experts, checked at the seed. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> TopK()
TopK
     k ┼ Int64: 5
  proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ExpertMixture`](@ref)
  - [`CumulativeWealthState`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)

# References

  - $(ref_dict[:li2011corn])
"""
struct TopK{T1 <: Integer, T2 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The number of assets, or experts, held.
    """
    k::T1
    """
    $(field_dict[:proj])
    """
    proj::T2
    function TopK(k::Integer, proj::EuclideanProjection)
        @argcheck(k >= 1, DomainError(k, "k must be at least 1"))
        return new{typeof(k), typeof(proj)}(k, proj)
    end
end
function TopK(; k::Integer = 5, proj::EuclideanProjection = EuclideanProjection())::TopK
    return TopK(k, proj)
end
function rule_state_seed(alg::TopK, w::AbstractVector)
    @argcheck(alg.k <= length(w),
              DomainError(alg.k,
                          "k must be at most the number of assets or experts, $(length(w))"))
    return CumulativeWealthState(0, zeros(eltype(w), length(w)), copy(w))
end
function online_update!(alg::TopK, st::CumulativeWealthState, w::AbstractVector,
                        x::AbstractVector, ::Any, set::AbstractAllocationSet)
    st.G .+= log.(x)
    order = sortperm(st.G; rev = true, alg = Base.Sort.DEFAULT_STABLE)
    q = zeros(eltype(w), length(w))
    q[view(order, 1:(alg.k))] .= one(eltype(w)) / alg.k
    wn = project(alg.proj, set, q, price_adjusted_allocation(w, x))
    return CumulativeWealthState(st.n + 1, st.G, st.p0), wn
end
"""
$(DocStringExtensions.TYPEDEF)

The weak aggregating algorithm of Kalnishkan and Vyugin (2008), in the form Yang, He, Lin and Zhang (2020) and Yang, He and Zhang (2022) apply to portfolios: every weight is the prior times the exponential of the cumulative log wealth at the shrinking rate ``1 / \\sqrt{t + 1}`` (WAA).

# Mathematical definition

With ``G_t = \\sum_{s \\leq t} \\log x_s`` the cumulative log wealth of every asset — every expert, on an [`ExpertMixture`](@ref)'s slot — and ``\\boldsymbol{p}_1`` the Start Allocation,

```math
\\begin{align}
w_{t+1, i} &\\propto p_{1, i} \\exp\\left( \\frac{G_{t, i}}{\\sqrt{t + 1}} \\right)\\,,
\\end{align}
```

normalised, which is the entropic projection onto the simplex; the `proj` slot is bound to [`EntropicProjection`](@ref). The rate is ``1 / \\sqrt{t}`` with no constant, as both applied papers fix it, and the weighting reads the cumulative wealth and not the previous weight, so the carrier holds the prior and the wealth. It differs from [`AggregatingAlgorithm`](@ref) by the shrinking rate, and from Kalnishkan and Vyugin's general form by the linear blend of the experts' allocations the mixture takes in place of a substitution function. At a rate of one it is the wealth weighting, and as the rate vanishes it tends to [`TopK`](@ref) at ``k = 1``. Over a set of experts the average log growth rate of the mixture is asymptotically that of the best expert.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WeakAggregatingAlgorithm(; proj::EntropicProjection = EntropicProjection()) -> WeakAggregatingAlgorithm

Keywords correspond to the struct's fields. The prior is the Start Allocation: on an [`ExpertMixture`](@ref), its `p`, uniform by default.

# Examples

```jldoctest
julia> WeakAggregatingAlgorithm()
WeakAggregatingAlgorithm
  proj ┴ EntropicProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`ExpertMixture`](@ref)
  - [`AggregatingExponentialGradient`](@ref)
  - [`CumulativeWealthState`](@ref)
  - [`AggregatingAlgorithm`](@ref)

# References

  - $(ref_dict[:kalnishkanvyugin2008])
  - $(ref_dict[:yang2020waeg])
  - $(ref_dict[:yang2022caeg])
"""
struct WeakAggregatingAlgorithm{T1 <: EntropicProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    $(field_dict[:proj])
    """
    proj::T1
    function WeakAggregatingAlgorithm(proj::EntropicProjection)
        return new{typeof(proj)}(proj)
    end
end
function WeakAggregatingAlgorithm(;
                                  proj::EntropicProjection = EntropicProjection())::WeakAggregatingAlgorithm
    return WeakAggregatingAlgorithm(proj)
end
function rule_state_seed(::WeakAggregatingAlgorithm, w::AbstractVector)
    return CumulativeWealthState(0, zeros(eltype(w), length(w)), copy(w))
end
function online_update!(alg::WeakAggregatingAlgorithm, st::CumulativeWealthState,
                        w::AbstractVector, x::AbstractVector, ::Any,
                        set::AbstractAllocationSet)
    st.G .+= log.(x)
    n = st.n + 1
    # The exponent is shifted by its maximum, which the normalisation undoes, so a long run
    # never overflows the exponential.
    q = st.p0 .* exp.((st.G .- maximum(st.G)) ./ sqrt((n + 1) * one(eltype(st.G))))
    wn = project(alg.proj, set, q, price_adjusted_allocation(w, x))
    return CumulativeWealthState(n, st.G, st.p0), wn
end
"""
    AggregatingExponentialGradient(; etas::AbstractVector{<:Real} = 0.01:0.01:0.2, eset::Option{<:BoundedAllocationSet} = nothing, proj::EuclideanProjection = EuclideanProjection())

The aggregation of exponentiated-gradient experts by the weak aggregating algorithm of Yang, He, Lin and Zhang (2020): the [`ExpertMixture`](@ref) under [`WeakAggregatingAlgorithm`](@ref) over one [`ExponentiatedGradient`](@ref) expert per rate in `etas`, uniform prior (WAEG).

Yang, He and Zhang (2022) index the experts by a continuum of rates under a measure and prove the mixture universal — its average log growth rate is asymptotically the best constant rebalanced portfolio's — but discretise the continuum to the identical grid `0.01:0.01:0.2` in their own experiments, so that form computes the same numbers as this constructor on that grid (CAEG). The 2020 paper reports near-flat wealth over the grid's size and its upper end, so the constructor's default is the papers' grid.

# Arguments

  - `etas`: The rates of the experts, one exponentiated-gradient expert each.
  - `eset`: The Expert Set the weighting projects onto, or `nothing` for the bare simplex.
  - `proj`: The geometry the blend is projected onto the head's Allocation Set in.

# Validation

  - `etas`: non-empty, and every element is positive and finite.

# Examples

```jldoctest
julia> mix = AggregatingExponentialGradient(; etas = [0.05, 0.1]);

julia> length(mix.experts), mix.alg
(2, WeakAggregatingAlgorithm
  proj ┴ EntropicProjection()
)
```

# Related

  - [`ExpertMixture`](@ref)
  - [`WeakAggregatingAlgorithm`](@ref)
  - [`ExponentiatedGradient`](@ref)

# References

  - $(ref_dict[:yang2020waeg])
  - $(ref_dict[:yang2022caeg])
"""
function AggregatingExponentialGradient(; etas::AbstractVector{<:Real} = 0.01:0.01:0.2,
                                        eset::Option{<:BoundedAllocationSet} = nothing,
                                        proj::EuclideanProjection = EuclideanProjection())::ExpertMixture
    assert_nonempty_gt0_finite_val(etas, :etas)
    experts = map(eta -> ExponentiatedGradient(; eta = eta), etas)
    return ExpertMixture(; experts = experts, alg = WeakAggregatingAlgorithm(), eset = eset,
                         proj = proj)
end
export ConfidenceWeightedMeanReversion, VarianceUpdate, StandardDeviationUpdate,
       AntiCorrelation, ExpectationMaximisation, AggregatingAlgorithm, TopK,
       WeakAggregatingAlgorithm, AggregatingExponentialGradient
public AbstractConfidenceUpdate, confidence_step, confidence_gain
