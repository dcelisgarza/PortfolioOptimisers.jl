"""
$(DocStringExtensions.TYPEDEF)

Forecasts the next price relative and its covariance by averaging many low-dimensional lagged regressions, weighted by their in-sample fit.

The prior of Xi, Li, Song and Ning (2023): the market is treated as a high-dimensional system whose next relatives are predicted from many random low-dimensional subsystems, each an ordinary least-squares regression of every asset's relative on the lagged relatives of `subsystem_size` assets over the window the rows span. Each subsystem's forecast is weighted by a Gaussian kernel of its in-sample mean squared error, and its predictive covariance — the sample covariance of the regressors pushed through the coefficients — is aggregated under the product of the two assets' weights. The mean and the covariance are read off the **same** regressions, which is what makes the pair consistent: the risk term of a programme that holds this prior is the uncertainty of the very forecast its return term reads.

# Mathematical definition

With ``\\boldsymbol{x}_\\tau = \\boldsymbol{1} + \\boldsymbol{r}_\\tau`` the price relatives of the ``T`` rows handed in, ``w = T - 1`` the regression window, and ``\\mathcal{A}_l = (l_1, \\ldots, l_s)`` the assets of the ``l``-th of ``L`` subsystems drawn uniformly without replacement,

```math
\\begin{align}
x_{\\tau, k} &= \\sum_{j = 1}^{s} \\beta^{(l)}_{k, j}\\, x_{\\tau - 1, l_j} + \\varepsilon^{(l)}_{\\tau, k}\\,, \\quad \\tau = T - w + 1, \\ldots, T\\,,\\\\
\\hat{x}^{(l)}_{T + 1, k} &= \\boldsymbol{x}_{T, \\mathcal{A}_l}^\\intercal \\hat{\\boldsymbol{\\beta}}^{(l)}_k\\,, \\quad
R^{(l)}_k = \\frac{1}{w} \\sum_{\\tau} \\left( x_{\\tau, k} - \\boldsymbol{x}_{\\tau - 1, \\mathcal{A}_l}^\\intercal \\hat{\\boldsymbol{\\beta}}^{(l)}_k \\right)^2\\,,\\\\
v^{(l)}_k &= \\frac{\\exp\\left(-R^{(l)}_k / \\sigma^2\\right)}{\\sum_{m = 1}^{L} \\exp\\left(-R^{(m)}_k / \\sigma^2\\right)}\\,, \\quad
\\hat{x}_{T + 1, k} = \\sum_{l = 1}^{L} v^{(l)}_k\\, \\hat{x}^{(l)}_{T + 1, k}\\,, \\quad
\\hat{\\boldsymbol{\\mu}} = \\hat{\\boldsymbol{x}}_{T + 1} - \\boldsymbol{1}\\,,\\\\
\\hat{\\Sigma}^{(l)} &= B^{(l)} S^{(l)} B^{(l)\\intercal}\\,, \\quad
\\hat{\\Sigma}_{k h} = \\frac{\\sum_{l = 1}^{L} v^{(l)}_k v^{(l)}_h\\, \\hat{\\Sigma}^{(l)}_{k h}}{\\sum_{l = 1}^{L} v^{(l)}_k v^{(l)}_h}\\,,
\\end{align}
```

where ``\\hat{\\boldsymbol{\\beta}}^{(l)}_k`` is the least-squares coefficient vector of asset ``k`` on subsystem ``l`` with no intercept, ``B^{(l)}`` the ``N \\times s`` matrix whose rows are those vectors, and ``S^{(l)}`` the ``s \\times s`` sample covariance of the subsystem's relatives over the last ``w`` rows, with the ``w - 1`` denominator. The paper's equations 4 to 10. The subsystems are the same for every target asset and are drawn afresh at every fit, so a rule that refits the prior each period re-samples them each period, as the paper does; a `seed` draws the same subsystems at every fit instead, for a reproducible run.

The kernel weights are computed after subtracting each asset's smallest error, which leaves every ratio unchanged and stops a window whose errors all exceed ``\\sigma^2`` by hundreds from underflowing every weight to zero. When two assets' weights nonetheless collapse onto different subsystems, so that no subsystem carries weight for both, their covariance entry has no support and is zero. A window of fewer rows than the subsystem has regressors makes the regression underdetermined, and the least-norm coefficients are taken. The aggregate covariance is a weighted average of positive semidefinite matrices under weights that differ entry by entry, so it need not be positive semidefinite itself; `pdm` repairs it to the nearest positive definite matrix and `nothing` leaves it as aggregated, for a consumer that takes it as a quadratic form.

The estimator reads returns, as every prior does, and forms the relatives at its door. It admits asset returns only; factor returns handed to it are ignored. A window of `w` regression pairs needs `w + 1` rows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowDimensionEnsemblePrior(; n_subsystems::Integer = 300, subsystem_size::Integer = 3, sigma::Real = 0.025, pdm::Option{<:Posdef} = Posdef(), rng::Random.AbstractRNG = Random.default_rng(), seed::Option{<:Integer} = nothing) -> LowDimensionEnsemblePrior

Keywords correspond to the struct's fields. The paper's defaults: `L = 300` subsystems, which its scan to two thousand converges by, `s = 3` assets each, and a kernel bandwidth of `0.025`.

## Validation

  - `n_subsystems >= 1`. A `DomainError` is thrown otherwise.
  - `subsystem_size >= 1`. A `DomainError` is thrown otherwise.
  - `sigma > 0` and finite. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> LowDimensionEnsemblePrior()
LowDimensionEnsemblePrior
    n_subsystems ┼ Int64: 300
  subsystem_size ┼ Int64: 3
           sigma ┼ Float64: 0.025
             pdm ┼ Posdef
                 │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
                 │   kwargs ┴ @NamedTuple{}: NamedTuple()
             rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
            seed ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`Posdef`](@ref)
  - [`LowDimensionEnsemblePortfolio`](@ref)
  - [`PriorExpectedReturns`](@ref)

# References

  - $(ref_dict[:xi2023oldem])
"""
struct LowDimensionEnsemblePrior{T1 <: Integer, T2 <: Integer, T3 <: Real,
                                 T4 <: Option{<:Posdef}, T5 <: Random.AbstractRNG,
                                 T6 <: Option{<:Integer}} <:
       AbstractLowOrderPriorEstimator_A
    """
    The number of low-dimensional subsystems drawn at every fit, the paper's `L`.
    """
    n_subsystems::T1
    """
    The number of assets in every subsystem, the paper's `s`; clipped to the number of assets in the rows when the universe is smaller.
    """
    subsystem_size::T2
    """
    The kernel bandwidth of the subsystem weights, the paper's `σ`: a subsystem's weight is the Gaussian kernel of its in-sample mean squared error at this bandwidth.
    """
    sigma::T3
    """
    The positive definite repair of the aggregated covariance, or `nothing` to leave it as aggregated.
    """
    pdm::T4
    """
    $(field_dict[:rng])
    """
    rng::T5
    """
    $(field_dict[:seed])
    """
    seed::T6
    function LowDimensionEnsemblePrior(n_subsystems::Integer, subsystem_size::Integer,
                                       sigma::Real, pdm::Option{<:Posdef},
                                       rng::Random.AbstractRNG, seed::Option{<:Integer})
        @argcheck(n_subsystems >= 1, DomainError(n_subsystems, "n_subsystems must be >= 1"))
        @argcheck(subsystem_size >= 1,
                  DomainError(subsystem_size, "subsystem_size must be >= 1"))
        @argcheck(zero(sigma) < sigma && isfinite(sigma),
                  DomainError(sigma, "sigma must be positive and finite"))
        return new{typeof(n_subsystems), typeof(subsystem_size), typeof(sigma), typeof(pdm),
                   typeof(rng), typeof(seed)}(n_subsystems, subsystem_size, sigma, pdm, rng,
                                              seed)
    end
end
function LowDimensionEnsemblePrior(; n_subsystems::Integer = 300,
                                   subsystem_size::Integer = 3, sigma::Real = 0.025,
                                   pdm::Option{<:Posdef} = Posdef(),
                                   rng::Random.AbstractRNG = Random.default_rng(),
                                   seed::Option{<:Integer} = nothing)::LowDimensionEnsemblePrior
    return LowDimensionEnsemblePrior(n_subsystems, subsystem_size, sigma, pdm, rng, seed)
end
function factor_residual_config(::LowDimensionEnsemblePrior)
    # The ensemble estimates the asset covariance directly; there is no factor lift and so no
    # residual block to remove (see [`factor_residual_config`](@ref)).
    return nothing
end
"""
    fit_min_rows(est)

The fewest observations an estimator tree fits on: the largest floor any estimator in the tree states, and one where none does.

An estimator whose fit is undefined below some number of rows states that number with a method of its own; every other estimator answers the largest floor among its estimator-valued fields, so a floor stated on a prior three slots deep reaches the rule that holds the head. [`LowDimensionEnsemblePrior`](@ref) states three: two regression pairs, the fewest whose regressor covariance exists.

# Related

  - [`LowDimensionEnsemblePrior`](@ref)
  - [`leader_min_rows`](@ref)
  - [`forecast_min_rows`](@ref)
"""
function fit_min_rows(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    return maximum(f -> fit_min_rows(getfield(est, f)), estimator_fields(est); init = 1)
end
function fit_min_rows(::Any)
    return 1
end
function fit_min_rows(::LowDimensionEnsemblePrior)
    return 3
end
"""
    prior(pe::LowDimensionEnsemblePrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)

The low-dimension ensemble prior of the rows: the kernel-weighted one-step forecast of the price relatives less one as `mu`, and the aggregated predictive covariance as `sigma`, over every row handed in. The fit reduces to the Coverage Universe of the window first, as every prior does — an asset with a non-finite return, or an inactive row of the panel, anywhere in the window is left out — and both moments are expanded back to the full universe with `NaN` outside it, so a consumer's Investable Mask reads the departure and the rows carried keep every column as given.

# Arguments

  - `pe`: The ensemble prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (ignored).
  - `pnl`: Asset Panel, whose active mask enters the Coverage Universe.
  - $(arg_dict[:dims])
  - `kwargs...`: Ignored.

# Validation

  - `size(X, 1) >= 3` after orientation: two regression pairs, the fewest whose regressor covariance exists. An `ArgumentError` is thrown otherwise.

# Returns

  - `pr::LowOrderPrior`: The rows as given, the forecast mean and the predictive covariance, `NaN` outside the Coverage Universe.

# Related

  - [`LowDimensionEnsemblePrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`coverage_reduction`](@ref)
  - [`expand_moment`](@ref)
"""
function prior(pe::LowDimensionEnsemblePrior, X::MatNum, ::Option{<:MatNum} = nothing,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
    Xf = dims_oriented(dims, X)
    cmsk, X = coverage_reduction(Xf, pnl)
    T, N = size(X)
    @argcheck(T >= 3,
              ArgumentError("the low-dimension ensemble needs at least three observations, got $T: two regression pairs are the fewest whose regressor covariance exists"))
    P = one(eltype(X)) .+ X
    # The regression pairs: every row but the first is a target, and the row before it the
    # regressor; the last row alone regresses the forecast.
    Y = P[2:end, :]
    Z = P[1:(end - 1), :]
    z = P[end, :]
    w = T - 1
    s = min(pe.subsystem_size, N)
    L = pe.n_subsystems
    rng = resolve_rng(pe.rng, pe.seed)
    idxs = map(_ -> StatsBase.sample(rng, 1:N, s; replace = false), 1:L)
    # One least-squares solve per subsystem serves every target asset: the coefficient
    # matrix is `s × N`, the paper's `B` transposed.
    Bs = map(idx -> Z[:, idx] \ Y, idxs)
    R = stack(map(l -> vec(sum(abs2, Y .- Z[:, idxs[l]] * Bs[l]; dims = 1)) ./ w, 1:L);
              dims = 1)
    F = stack(map(l -> transpose(Bs[l]) * z[idxs[l]], 1:L); dims = 1)
    # The kernel weights, shifted by each asset's smallest error: the ratios are unchanged
    # and a window of large errors no longer underflows every weight to zero.
    V = exp.(-(R .- minimum(R; dims = 1)) ./ abs2(pe.sigma))
    V ./= sum(V; dims = 1)
    mu = vec(sum(V .* F; dims = 1)) .- one(eltype(F))
    # The covariance of subsystem `l` is the regressors' sample covariance pushed through its
    # coefficients, weighted entry by entry by the product of the two assets' weights.
    num = sum(map(l -> (V[l, :] .* transpose(V[l, :])) .*
                       (transpose(Bs[l]) * Statistics.cov(Y[:, idxs[l]]) * Bs[l]), 1:L))
    den = transpose(V) * V
    # An entry whose weights vanish on every subsystem — the two assets' kernels collapsed
    # onto different subsystems — has no support, and the paper's ratio is undefined; it is
    # zero here.
    sigma = ifelse.(iszero.(den), zero(eltype(num)), num ./ den)
    # Symmetrised to the bit, so a consumer's Hermitian check holds; a matrix product need
    # not be.
    sigma = (sigma .+ transpose(sigma)) ./ 2
    return LowOrderPrior(; X = Xf, mu = expand_moment(mu, cmsk, 1),
                         sigma = expand_moment(posdef(pe.pdm, sigma), cmsk))
end

export LowDimensionEnsemblePrior
