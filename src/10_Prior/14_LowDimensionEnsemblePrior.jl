"""
$(DocStringExtensions.TYPEDEF)

Forecasts the next price relative and its covariance by averaging many low-dimensional lagged regressions, weighted by their in-sample fit.

This is the prior of Xi, Li, Song and Ning (2023). It treats the market as one high-dimensional system and draws many small subsystems of `subsystem_size` assets from it at random. In each subsystem, the price relative of every asset is regressed on the price relatives of the subsystem's assets one period earlier. The forecast of a subsystem counts in proportion to a Gaussian kernel of its in-sample mean squared error. Each subsystem also gives a predictive covariance, and the aggregate weighs it by the product of the kernel weights of the two assets.

The mean and the covariance come from the same regressions. So the risk term of a programme that reads this prior measures the uncertainty of the forecast that its return term reads. A window of ``w`` regression pairs needs ``w + 1`` rows.

# Mathematical definition

The rows are the periods ``t - w`` to ``t``. Subsystem ``l`` holds the assets ``\\mathcal{A}_l = (l_1, \\ldots, l_s)``. For every asset ``k``, every subsystem ``l`` and ``i = 0, \\ldots, w - 1``,

```math
\\begin{align}
x_{t-i,k} &= \\sum_{j = 1}^{s} \\beta^{(l)}_{k,j}\\, x_{t-i-1,l_j} + \\varepsilon^{(l)}_{t-i,k}\\,,\\\\
\\hat{\\boldsymbol{\\beta}}^{(l)}_k &= \\left(\\mathbf{X}^{(l)}\\right)^{+} \\boldsymbol{y}_k\\,,\\\\
\\hat{x}^{(l)}_{t+1,k} &= \\boldsymbol{x}_{t,\\mathcal{A}_l}^\\intercal \\hat{\\boldsymbol{\\beta}}^{(l)}_k\\,,\\\\
R^{(l)}_k &= \\frac{1}{w} \\sum_{i = 0}^{w - 1} \\left(x_{t-i,k} - \\boldsymbol{x}_{t-i-1,\\mathcal{A}_l}^\\intercal \\hat{\\boldsymbol{\\beta}}^{(l)}_k\\right)^2\\,,\\\\
v^{(l)}_k &= \\frac{\\exp\\left(-R^{(l)}_k / \\sigma^2\\right)}{\\sum_{m = 1}^{L} \\exp\\left(-R^{(m)}_k / \\sigma^2\\right)}\\,,\\\\
\\hat{x}_{t+1,k} &= \\sum_{l = 1}^{L} v^{(l)}_k\\, \\hat{x}^{(l)}_{t+1,k}\\,,\\\\
\\hat{\\boldsymbol{\\mu}} &= \\hat{\\boldsymbol{x}}_{t+1} - \\boldsymbol{1}\\,,\\\\
S^{(l)}_{pq} &= \\frac{1}{w - 1} \\sum_{i = 0}^{w - 1} \\left(x_{t-i,l_p} - \\bar{x}_{l_p}\\right) \\left(x_{t-i,l_q} - \\bar{x}_{l_q}\\right)\\,,\\\\
\\hat{\\mathbf{\\Sigma}}^{(l)} &= \\mathbf{B}^{(l)} \\mathbf{S}^{(l)} \\mathbf{B}^{(l)\\intercal}\\,,\\\\
\\left(\\hat{\\mathbf{\\Sigma}}_{t+1}\\right)_{kh} &= \\frac{\\sum_{l = 1}^{L} v^{(l)}_k v^{(l)}_h\\, \\hat{\\Sigma}^{(l)}_{kh}}{\\sum_{l = 1}^{L} v^{(l)}_k v^{(l)}_h}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_t_rel])
  - ``x_{t,k}``: Entry ``k`` of ``\\boldsymbol{x}_t``, and ``\\boldsymbol{x}_{t,\\mathcal{A}_l}`` the entries of the assets of ``\\mathcal{A}_l``.
  - $(math_dict[:t_period])
  - ``w``: Regression window, the number of regression pairs. The rows number ``w + 1``.
  - ``L``: Number of subsystems.
  - ``s``: Number of assets in each subsystem.
  - ``\\mathcal{A}_l``: Assets of subsystem ``l``, drawn uniformly without replacement. The same subsystems serve every asset ``k``.
  - ``\\beta^{(l)}_{k,j}``, ``\\varepsilon^{(l)}_{t-i,k}``: Coefficient and error term of the regression of asset ``k`` on subsystem ``l``, with no intercept.
  - ``\\mathbf{X}^{(l)}``: ``w \\times s`` matrix whose row ``i + 1`` is ``\\boldsymbol{x}_{t-i-1,\\mathcal{A}_l}^\\intercal``.
  - ``(\\cdot)^{+}``: Moore–Penrose pseudo-inverse.
  - ``\\boldsymbol{y}_k = (x_{t,k}, \\ldots, x_{t-w+1,k})^\\intercal``: Targets of the regression of asset ``k``.
  - ``\\hat{\\boldsymbol{\\beta}}^{(l)}_k``: Least-squares coefficient vector of asset ``k`` on subsystem ``l``.
  - ``\\hat{x}^{(l)}_{t+1,k}``: Forecast of subsystem ``l`` for asset ``k``.
  - ``R^{(l)}_k``: In-sample mean squared error of that regression.
  - ``\\sigma``: Kernel bandwidth.
  - ``v^{(l)}_k``: Kernel weight of subsystem ``l`` for asset ``k``. The weights of one asset sum to one.
  - $(math_dict[:xhat_fc])
  - ``\\hat{x}_{t+1,k}``: Entry ``k`` of ``\\hat{\\boldsymbol{x}}_{t+1}``.
  - ``\\hat{\\boldsymbol{\\mu}}``: Forecast return vector, the Price Relative Forecast less one.
  - ``\\mathbf{S}^{(l)}``: ``s \\times s`` sample covariance of the price relatives of the assets of ``\\mathcal{A}_l`` over the last ``w`` rows, with entries ``S^{(l)}_{pq}``.
  - ``\\bar{x}_{l_p} = \\frac{1}{w} \\sum_{i = 0}^{w - 1} x_{t-i,l_p}``: Mean of the price relatives of asset ``l_p`` over the last ``w`` rows.
  - ``\\mathbf{B}^{(l)}``: ``N \\times s`` matrix whose row ``k`` is ``\\hat{\\boldsymbol{\\beta}}^{(l)\\intercal}_k``.
  - ``\\hat{\\mathbf{\\Sigma}}^{(l)}``: Predictive covariance of subsystem ``l``, with entries ``\\hat{\\Sigma}^{(l)}_{kh}``.
  - $(math_dict[:Sigma_hat_pred])
  - $(math_dict[:N])

These are the equations 4 to 10 of the paper. When ``\\mathbf{X}^{(l)}`` has full column rank, the pseudo-inverse gives the paper's ``(\\mathbf{X}^{(l)\\intercal} \\mathbf{X}^{(l)})^{-1} \\mathbf{X}^{(l)\\intercal} \\boldsymbol{y}_k``. When the window has fewer pairs than the subsystem has assets, it gives the least-squares coefficients of least norm.

Every ``\\hat{\\mathbf{\\Sigma}}^{(l)}`` is positive semidefinite. But ``\\hat{\\mathbf{\\Sigma}}_{t+1}`` averages them under weights that differ from entry to entry, so it need not be positive semidefinite itself.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowDimensionEnsemblePrior(; n_subsystems::Integer = 300, subsystem_size::Integer = 3, sigma::Real = 0.025, pdm::Option{<:Posdef} = Posdef(), rng::Random.AbstractRNG = Random.default_rng(), seed::Option{<:Integer} = nothing) -> LowDimensionEnsemblePrior

Keywords correspond to the struct's fields. The defaults are the paper's. The paper scans ``L`` from 100 to 2000 and finds that the mean squared prediction error converges once ``L`` exceeds 300. It sets three assets in each subsystem and a kernel bandwidth of `0.025`.

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

  - $(ref_dict[:xi2023oldem]) Equations 4 to 10 and Section 5.3.
"""
struct LowDimensionEnsemblePrior{T1 <: Integer, T2 <: Integer, T3 <: Real,
                                 T4 <: Option{<:Posdef}, T5 <: Random.AbstractRNG,
                                 T6 <: Option{<:Integer}} <:
       AbstractLowOrderPriorEstimator_A
    """
    The number of subsystems that every fit draws, the paper's ``L``.
    """
    n_subsystems::T1
    """
    The number of assets in every subsystem, the paper's ``s``. When the rows hold fewer assets, the fit uses all of them.
    """
    subsystem_size::T2
    """
    The kernel bandwidth of the subsystem weights, the paper's ``\\sigma``.
    """
    sigma::T3
    """
    The positive definite repair of the aggregated covariance, or `nothing` to leave the aggregate as it is. On short windows the aggregate is usually indefinite.
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

Returns the fewest observations that an estimator tree can fit on.

An estimator whose fit is undefined below some number of rows states that number with a method of its own. [`LowDimensionEnsemblePrior`](@ref) states three, because two regression pairs are the fewest whose regressor covariance exists. Every other estimator returns the largest floor in its fields, so a floor that a prior states three levels down reaches the rule that holds the head.

# Mathematical definition

```math
\\begin{align}
n(e) &= \\begin{cases}
n_e & \\text{if } e \\text{ states a floor}\\,,\\\\
\\max\\left(1,\\, \\max_{f \\in \\mathcal{F}(e)} n(f)\\right) & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``n(e)``: Fewest rows that estimator ``e`` can fit on.
  - ``n_e``: Floor that the type of ``e`` states.
  - ``\\mathcal{F}(e)``: Estimators that the fields of ``e`` hold, one for each estimator-valued field and one for each entry of a field that holds a vector of estimators. A value that is not an estimator has ``n = 1``.

# Arguments

  - `est`: The estimator tree, or any value.

# Returns

  - `n::Integer`: The fewest rows.

# Related

  - [`LowDimensionEnsemblePrior`](@ref)
  - [`leader_min_rows`](@ref)
  - [`forecast_min_rows`](@ref)
"""
function fit_min_rows(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    return maximum(f -> fit_min_rows(getfield(est, f)), fieldnames(typeof(est)); init = 1)
end
function fit_min_rows(v::AbstractVector{<:Union{<:AbstractEstimator,
                                                <:StatsBase.CovarianceEstimator}})
    return maximum(fit_min_rows, v; init = 1)
end
function fit_min_rows(::Any)
    return 1
end
function fit_min_rows(::LowDimensionEnsemblePrior)
    return 3
end
"""
    prior(pe::LowDimensionEnsemblePrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)

Computes the low-dimension ensemble prior of the rows, with the forecast return as `mu` and the aggregated predictive covariance as `sigma`.

The fit reads the Coverage Universe of the rows, as every prior does. An asset with a non-finite return, or with an inactive row of the panel, at any observation is left out. Both moments are expanded back to the full universe with `NaN` outside it, so the Investable Mask of a consumer reads the departure. The returned rows keep every column as given.

# Algorithm

 1. Orient `X` to observations by assets, giving `Xf`.
 2. Reduce `Xf` to its Coverage Universe with [`coverage_reduction`](@ref), giving the mask `cmsk` and the reduced returns `X` of `T` rows and `N` assets.
 3. Check that `T >= 3`.
 4. Form the price relatives `P = 1 + X`. Split them into the targets `Y`, every row but the first, the regressors `Z`, every row but the last, and the last row `z`.
 5. Clip the subsystem size to `N`, giving `s`.
 6. Resolve the random number generator from `rng` and `seed`, and draw `L` subsystems `idxs` of `s` assets each, without replacement. One draw serves every target asset. With a `seed`, every call draws the same subsystems. Without one, every call draws new ones, so a rule that refits the prior at every period draws new subsystems at every period, as the paper does.
 7. Solve one least-squares problem for each subsystem, for all the targets at once, giving the `s × N` coefficient matrices `Bs`. Julia's `\\` gives the coefficients of least norm when the window has fewer pairs than `s`.
 8. Compute the `L × N` in-sample mean squared errors `R` and the subsystem forecasts `F`.
 9. Subtract the smallest error of each asset from its column of `R`, giving `E`. This leaves every kernel ratio unchanged. It also stops the kernel from underflowing to zero for every subsystem when the errors are large against ``\\sigma^2``.
10. Compute the kernel weights `V` from `E`, and scale each column to sum to one.
11. Compute `mu` as the weighted forecasts less one.
12. For each pair of assets, find the smallest sum `m` of their two shifted errors over the subsystems. The covariance weight of subsystem `l` for the pair is the kernel of its sum less `m`. The two shifts cancel in the ratio, and the largest weight of each pair is one, so no denominator underflows to zero.
13. Accumulate the weighted subsystem covariances in `num` and the weights in `den`, and divide, giving `sigma`.
14. Symmetrise `sigma`, because a matrix product need not be symmetric to the bit.
15. Repair `sigma` with [`posdef`](@ref) under `pdm`, and expand `mu` and `sigma` to the full universe with [`expand_moment`](@ref).

# Arguments

  - `pe`: The ensemble prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix, ignored.
  - `pnl`: Asset Panel, whose active mask enters the Coverage Universe.
  - $(arg_dict[:dims])
  - `kwargs...`: Ignored.

# Validation

  - `size(X, 1) >= 3` after orientation, because two regression pairs are the fewest whose regressor covariance exists. An `ArgumentError` is thrown otherwise.

# Returns

  - `pr::LowOrderPrior`: The rows as given, the forecast mean and the predictive covariance, with `NaN` outside the Coverage Universe.

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
    # Each asset's errors above its smallest: every kernel ratio is unchanged, and a window
    # of large errors no longer underflows every weight to zero.
    s2 = abs2(pe.sigma)
    E = R .- minimum(R; dims = 1)
    V = exp.(.-E ./ s2)
    V ./= sum(V; dims = 1)
    mu = vec(sum(V .* F; dims = 1)) .- one(eltype(F))
    # The covariance weight of a pair is the product of the two kernels. Shifted by its
    # largest value over the subsystems, it is exact in the ratio and never underflows to
    # zero for every subsystem, so every entry has a denominator of at least one.
    m = view(E, 1, :) .+ transpose(view(E, 1, :))
    for l in 2:L
        m .= min.(m, view(E, l, :) .+ transpose(view(E, l, :)))
    end
    u = similar(m)
    num = zero(m)
    den = zero(m)
    for l in 1:L
        u .= exp.((m .- view(E, l, :) .- transpose(view(E, l, :))) ./ s2)
        sl = transpose(Bs[l]) * Statistics.cov(view(Y, :, idxs[l])) * Bs[l]
        num .+= u .* sl
        den .+= u
    end
    sigma = num ./ den
    # Symmetrised to the bit, so a consumer's Hermitian check holds; a matrix product need
    # not be.
    sigma = (sigma .+ transpose(sigma)) ./ 2
    return LowOrderPrior(; X = Xf, mu = expand_moment(mu, cmsk, 1),
                         sigma = expand_moment(posdef(pe.pdm, sigma), cmsk))
end

export LowDimensionEnsemblePrior
