"""
$(DocStringExtensions.TYPEDEF)

Empirical prior estimator for asset returns.

`EmpiricalPrior` is a low order prior estimator that computes the mean and covariance of asset returns using empirical (sample-based) statistics. It supports custom expected returns and covariance estimators, as well as an optional investment horizon for log-normalisation and scaling.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EmpiricalPrior(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        horizon::Option{<:Number} = nothing,
        fill_limit::Option{<:Real} = nothing,
        max_scenarios::Option{<:Integer} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> EmpiricalPrior

Keywords correspond to the struct's fields.

## Validation

  - If `horizon` is not `nothing`, `horizon > 0`.
  - If `fill_limit` is not `nothing`, `0 < fill_limit <= 1`. Zero is not a value, because `nothing` already means that no fill passes in silence. The fit refuses a value looser than `1 - min_coverage` over the arms that state a coverage floor, because such a value could never fire.
  - If `max_scenarios` is not `nothing`, `max_scenarios > 0`.

## The incremental fit

The empirical prior **folds and carries**. [`partial_fit!`](@ref) forwards each observation to `me` and to `ce`, both of which fold exactly, and appends the row to a [`PriorCarryState`](@ref) of its own; the one-argument [`prior`](@ref) reads `mu` and `sigma` off the two folded arms and `X` off the carried rows. The step is therefore quadratic in the number of assets and independent of the number of observations folded, and the buffer is memory rather than arithmetic: it exists because [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures, not because anything needs refitting. ADR 0136 records the decision.

## Two caps, two names

`max_scenarios` and [`Online`](@ref)'s `max_history` are different knobs, and both are wanted.

  - `EmpiricalPrior(; max_scenarios = w)` fits `mu` and `sigma` over **every** observation and carries the last `w` rows as `X`. It is a property of the result, so it reads the same in batch and online, and it has no batch equal to assert: the moments and the scenarios come from different windows by construction.
  - `Online(EmpiricalPrior(); max_history = w)` windows the **whole** fit: the estimator refits from the buffer, so it equals `prior(pe, X[end - w + 1:end, :])` exactly.

Both set, they nest: the moments come from the last `max_history` observations and `X` from the last `max_scenarios` of those. A `max_scenarios` at or above the number of observations carried is a no-op.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `me`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

## Observation view parameters

When [`obs_weights_view`](@ref) is called on this type:

  - `cache`: Dropped via [`obs_weights_view`](@ref), because no slice of a state exists on the observation axis.

# Examples

```jldoctest
julia> EmpiricalPrior()
EmpiricalPrior
          ce ┼ PortfolioOptimisersCovariance
             │   ce ┼ Covariance
             │      │    me ┼ SimpleExpectedReturns
             │      │       │   w ┴ nothing
             │      │    ce ┼ GeneralCovariance
             │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │      │       │    w ┴ nothing
             │      │   alg ┼ FullMoment()
             │      │     w ┴ nothing
             │   mp ┼ MatrixProcessing
             │      │     pdm ┼ Posdef
             │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      │      dn ┼ nothing
             │      │      dt ┼ nothing
             │      │     alg ┼ nothing
             │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
          me ┼ SimpleExpectedReturns
             │   w ┴ nothing
     horizon ┼ nothing
  fill_limit ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`prior`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
  - [`PriorCarryState`](@ref)
  - [`Online`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.1.
  - $(ref_dict[:meucci2005]) Chapter 3.
"""
@propagatable @concrete struct EmpiricalPrior <: AbstractLowOrderPriorEstimator_A
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:horizon])
    """
    horizon
    """
    $(field_dict[:fill_limit])
    """
    fill_limit
    """
    $(field_dict[:max_scenarios])
    """
    max_scenarios
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function EmpiricalPrior(ce::StatsBase.CovarianceEstimator,
                            me::AbstractExpectedReturnsEstimator, horizon::Option{<:Number},
                            fill_limit::Option{<:Real}, max_scenarios::Option{<:Integer},
                            cache::Option{<:AbstractPartialFitState})
        if !isnothing(horizon)
            @argcheck(horizon > 0, DomainError(horizon, "horizon must be > 0"))
        end
        if !isnothing(fill_limit)
            @argcheck(0 < fill_limit <= 1,
                      DomainError(fill_limit,
                                  "fill_limit is a share of an investable column's own observations, so it must lie in (0, 1]. Pass `nothing`, the default, to derive it from the arms' coverage floor, or to be told about every fill where no arm states one; `1` to be told about none."))
        end
        if !isnothing(max_scenarios)
            @argcheck(max_scenarios > zero(max_scenarios),
                      DomainError(max_scenarios,
                                  "max_scenarios is the number of observations the result carries as scenarios, so it must be positive. Pass `nothing`, the default, to carry every observation the fit read."))
        end
        return new{typeof(ce), typeof(me), typeof(horizon), typeof(fill_limit),
                   typeof(max_scenarios), typeof(cache)}(ce, me, horizon, fill_limit,
                                                         max_scenarios, cache)
    end
end
function EmpiricalPrior(;
                        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        horizon::Option{<:Number} = nothing,
                        fill_limit::Option{<:Real} = nothing,
                        max_scenarios::Option{<:Integer} = nothing,
                        cache::Option{<:AbstractPartialFitState} = nothing)::EmpiricalPrior
    return EmpiricalPrior(ce, me, horizon, fill_limit, max_scenarios, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`EmpiricalPrior`](@ref) except `cache`, and `max_scenarios` only where it is set.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. `max_scenarios` is a cap most callers never set, and a `nothing` row for it would move every rendering of every host that carries a prior; it appears exactly where a caller chose one. Set `set_show_nothing_fields!(:EmpiricalPrior, true)` to render both. ADR 0105 records the decision.

# Arguments

  - `pe`: Empirical prior estimator.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ce, :me, :horizon, :fill_limit)` with no scenario cap and `(:ce, :me, :horizon, :fill_limit, :max_scenarios)` with one.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(pe::EmpiricalPrior)
    return if isnothing(pe.max_scenarios)
        (:ce, :me, :horizon, :fill_limit)
    else
        (:ce, :me, :horizon, :fill_limit, :max_scenarios)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`EmpiricalPrior`](@ref) method of [`coverage_floor`](@ref), the binding floor of its two arms.

`pe.me` and `pe.ce` may each carry a [`CoveragePolicy`](@ref), and either bounds every investable column on its own, because the Investable Mask is the conjunction of the two admissions and both read the same per-asset observation count. The binding floor is therefore the **maximum** of the floors the arms state, and a mixed configuration needs no rule of its own (ADR 0118).

# Arguments

  - `pe`: The empirical prior estimator.

# Returns

  - `floor::Option{<:Real}`: The binding coverage floor, or `nothing` when neither arm states one.

# Related

  - [`coverage_floor`](@ref)
  - [`resolve_fill_limit`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
function coverage_floor(pe::EmpiricalPrior)
    return coverage_floor(coverage_floor(pe.me), coverage_floor(pe.ce))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Scales a pair of log-return moments to an investment horizon and converts them to arithmetic returns, in place.

The horizon algebra of [`EmpiricalPrior`](@ref), written once. The batch method reaches it after fitting `pe.me` and `pe.ce` on `log1p.(X)`, and the read-out of a folded prior reaches it after reading the same two moments off their states, so the two answer identically by construction rather than by a test over two copies of the arithmetic.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{\\mu}} &= h \\cdot \\hat{\\boldsymbol{\\mu}}_{\\log}\\,, &
\\tilde{\\mathbf{\\Sigma}} &= h \\cdot \\hat{\\mathbf{\\Sigma}}_{\\log}\\,, \\\\
\\hat{\\mu}_i &= \\exp\\!\\left(\\tilde{\\mu}_i + \\tfrac{1}{2}\\tilde{\\sigma}_{ii}\\right) - 1\\,, &
\\hat{\\sigma}_{ij} &= (\\hat{\\mu}_i + 1)(\\hat{\\mu}_j + 1)\\left(\\exp(\\tilde{\\sigma}_{ij}) - 1\\right)\\,.
\\end{align}
```

Where:

  - ``h``: Investment horizon.
  - ``\\hat{\\boldsymbol{\\mu}}_{\\log}``, ``\\hat{\\mathbf{\\Sigma}}_{\\log}``: Mean and covariance of the log-returns.
  - ``\\hat{\\mu}_i``, ``\\hat{\\sigma}_{ij}``: Arithmetic mean return and covariance.

# Algorithm

The order of steps 3 and 4 is **not free**. Step 4 reads the `mu` step 3 left, which is ``\\hat{\\mu}_i + 1`` and not ``\\hat{\\mu}_i``, because step 5 has not yet subtracted the one, and the second closed form asks for exactly that factor. Moving step 5 in front of step 4 replaces each ``\\hat{\\mu}_i + 1`` by ``\\hat{\\mu}_i``, and on a daily return series that collapses the covariance to a small fraction of its value.

 1. Scale `mu` by `horizon`.
 2. Scale `sigma` by `horizon`.
 3. Overwrite `mu` with the exponential of the first closed form, which is the arithmetic mean **plus one**.
 4. Overwrite `sigma` with the second closed form, whose ``\\hat{\\mu}_i + 1`` factors are the `mu` of step 3.
 5. Subtract one from `mu`.

# Arguments

  - `mu`: The log-return mean, overwritten with the arithmetic mean.
  - `sigma`: The log-return covariance, overwritten with the arithmetic covariance.
  - `horizon`: The investment horizon.

# Returns

  - `nothing`. Both arguments are modified in place.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`prior`](@ref)
"""
function horizon_moments!(mu::VecNum, sigma::MatNum, horizon::Number)
    mu .*= horizon
    sigma .*= horizon
    mu .= exp.(mu + 0.5 * LinearAlgebra.diag(sigma))
    sigma .= (mu ⊗ mu) ⊙ (exp.(sigma) .- one(eltype(sigma)))
    mu .-= one(eltype(mu))
    return nothing
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any}, X::MatNum,
          F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute empirical prior moments for asset returns (no horizon adjustment).

`prior` estimates the mean and covariance of asset returns using the specified empirical prior estimator, without log-normalisation or scaling for investment horizon. The mean and covariance are computed using the estimators stored in `pe`, and returned in a [`LowOrderPrior`](@ref) result.

# Mathematical definition

`pe.me` computes the mean and `pe.ce` computes the covariance, so both moments are whatever those estimators return. Under the default pair — [`SimpleExpectedReturns`](@ref) and [`PortfolioOptimisersCovariance`](@ref) with no observation weights — they reduce to the sample moments:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\frac{1}{T} \\sum_{t=1}^{T} \\boldsymbol{x}_t\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\frac{1}{T-1} \\sum_{t=1}^{T} (\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})(\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` mean vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` covariance matrix.
  - ``\\boldsymbol{x}_t``: ``N \\times 1`` vector of asset returns at time ``t``.
  - $(math_dict[:T])

Every choice inside `pe.me` and `pe.ce` reaches the result. A shrunk mean and a denoised covariance move both away from the display above rather than refining it.

This method takes the **arithmetic** moments of `X` directly. It applies no log transform, so it is not the ``h = 1`` case of the horizon method: that one still passes through ``\\log(1 + x_t)`` and back, and the round trip is an identity only in the limit of small returns.

# The scenario cap

`pe.max_scenarios` cuts the returns matrix the result carries down to its last `max_scenarios` rows, through [`scenario_window`](@ref), and leaves `mu` and `sigma` fitted over every observation. The cut is taken **before** the fill, so the share the fill measures is the share of the window a consumer actually reads. A `max_scenarios` of `nothing`, the default, carries every row, and a cap at or above the number of observations is a `view` that copies nothing.

# The scenario fill

Under a **mask-aware** `pe.me` and `pe.ce` — the exponentially weighted family, and the plain family wherever a [`CoveragePolicy`](@ref) is set — an asset that lists inside the window is answered from the observations it has, so it is investable and its column of `X` still carries a `NaN` at every earlier row. [`scenario_fill`](@ref) writes `0` at those entries once, so that every consumer of the result reads a finite investable column; a non-investable asset keeps its whole `NaN` column, and `mu` and `sigma` are untouched. The fill costs accuracy at four consumers, and the message names all four: a scenario-based measure understates that asset's risk over the filled rows, a hierarchical optimiser may branch the asset alone and overweight it, an entropy pooling view on it is calibrated on the filled column, and a meta-optimiser carries the fill into the outer problem. It is silent while the **worst investable column's own** filled share stays at or below the limit [`resolve_fill_limit`](@ref) derives, warns above it, and refuses any fill under `strict`. `pe.fill_limit` defaults to `nothing`, which derives `1 - min_coverage` from the arms' coverage floor and never fires, and which names every fill where no arm states a floor. Under a **plain** estimator with no policy the fill never fires, because an asset the estimator could not cover leaves the Coverage Universe and is not investable.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (ignored).
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `strict`: Whether a zero-filled scenario raises rather than warns. Any fill raises under `strict`; otherwise a column filled above the limit [`resolve_fill_limit`](@ref) derives warns, and every fill warns while that limit is `nothing`.
  - `kwargs...`: Additional keyword arguments passed to mean and covariance estimators.

# Validation

  - `dims in (1, 2)`.
  - At least one asset must be in the Coverage Universe.
  - `pe.fill_limit` is at most `1 - min_coverage` over the arms that state a coverage floor, else a `DomainError` is thrown at the fit, because a looser limit is dead by construction.
  - The zero-filled share of the worst investable column of `X` is at or below the limit [`resolve_fill_limit`](@ref) derives, else a warning naming the assets is emitted, or an `ArgumentError` naming them is raised under `strict`, which any fill raises. A limit of `nothing` — no arm states a floor and `pe.fill_limit` is `nothing` — is at or below no share, so any fill is named.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`coverage_mask`](@ref)
  - [`scenario_fill`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing, <:Any}, X::MatNum,
               ::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    X = dims_oriented(dims, X)
    # The limit resolves against the arms' coverage floor at the fit, not at construction,
    # because a floor is a field of the arm and the estimator cannot see it from its own
    # constructor. A limit looser than admission refuses here (see
    # [`resolve_fill_limit`](@ref)).
    fill_limit = resolve_fill_limit(pe.fill_limit, coverage_floor(pe))
    mu = vec(Statistics.mean(pe.me, X, pnl; dims = 1, kwargs...))
    sigma = Statistics.cov(pe.ce, X, pnl; dims = 1, kwargs...)
    # The Scenario Cap cuts the rows the result carries and leaves the moments above it
    # alone, so the fill runs over the matrix a consumer actually reads and its share is
    # measured against that window (see [`scenario_window`](@ref)).
    Xs = scenario_window(pe.max_scenarios, X)
    # A mask-aware estimator answers a young asset from the rows it has, so the asset is
    # investable and its column still carries the gap. The fill is paid once, here, because
    # every consumer of the result reads that column (see [`scenario_fill`](@ref)).
    return LowOrderPrior(; X = scenario_fill(Xs, mu, sigma, strict, fill_limit), mu = mu,
                         sigma = sigma)
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any}, X::MatNum,
          F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute empirical prior moments for asset returns with investment horizon adjustment.

`prior` estimates the mean and covariance of asset returns using the specified empirical prior estimator, applying log-normalisation and scaling for the investment horizon. The asset returns are log-transformed, moments are computed using the estimators stored in `pe`, and then rescaled according to the investment horizon. The final mean and covariance are transformed back to arithmetic returns and returned in a [`LowOrderPrior`](@ref) result.

# Mathematical definition

`pe.me` and `pe.ce` are applied to the **log-returns** ``\\log(1 + x_t)`` rather than to `X` itself. The two log-moments are scaled by the investment horizon ``h``, then converted back to arithmetic returns:

```math
\\begin{align}
\\tilde{\\boldsymbol{\\mu}} &= h \\cdot \\hat{\\boldsymbol{\\mu}}_{\\log}\\,, \\\\
\\tilde{\\mathbf{\\Sigma}} &= h \\cdot \\hat{\\mathbf{\\Sigma}}_{\\log}\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mu}_i &= \\exp\\!\\left(\\tilde{\\mu}_i + \\tfrac{1}{2}\\tilde{\\sigma}_{ii}\\right) - 1\\,, \\\\
\\hat{\\sigma}_{ij} &= (\\hat{\\mu}_i + 1)(\\hat{\\mu}_j + 1)\\left(\\exp(\\tilde{\\sigma}_{ij}) - 1\\right)\\,.
\\end{align}
```

Where:

  - ``\\tilde{\\boldsymbol{\\mu}}``, ``\\tilde{\\mathbf{\\Sigma}}``: Horizon-scaled log-return mean and covariance.
  - ``h``: Investment horizon.
  - ``\\hat{\\boldsymbol{\\mu}}_{\\log}``, ``\\hat{\\mathbf{\\Sigma}}_{\\log}``: Mean and covariance of the log-returns ``\\log(1 + x_t)``, computed by `pe.me` and `pe.ce`.
  - ``\\hat{\\mu}_i``: Arithmetic mean return for asset ``i``.
  - ``\\hat{\\sigma}_{ij}``: Arithmetic covariance between assets ``i`` and ``j``.

`X` in the returned [`LowOrderPrior`](@ref) is the arithmetic returns matrix the caller supplied. Only the moments are computed in log space.

# Algorithm

The order of steps 5 to 7 is **not free**. Step 6 reads the `mu` that step 5 left, which is ``\\hat{\\mu}_i + 1`` and not ``\\hat{\\mu}_i``, because step 7 has not yet subtracted the one. The second closed form asks for exactly that factor, so the body meets it by ordering rather than by recomputing. Moving step 7 in front of step 6 replaces each ``\\hat{\\mu}_i + 1`` by ``\\hat{\\mu}_i``, and on a daily return series that collapses the covariance to a small fraction of its value.

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Take the log-returns `X_log`, as `log1p.(X)`.
 3. Compute the log-return mean `mu` on `X_log`, under `pe.me`, and the log-return covariance `sigma` on `X_log`, under `pe.ce`.
 4. Scale both by `pe.horizon`, giving ``\\tilde{\\boldsymbol{\\mu}}`` and ``\\tilde{\\mathbf{\\Sigma}}``.
 5. Overwrite `mu` with the exponential of the first closed form. This is the arithmetic mean **plus one**, because the subtraction is still to come.
 6. Overwrite `sigma` with the second closed form, whose ``\\hat{\\mu}_i + 1`` factors are the `mu` of step 5.
 7. Subtract one from `mu`, giving ``\\hat{\\boldsymbol{\\mu}}``.
 8. Return a [`LowOrderPrior`](@ref) carrying the arithmetic `X` of step 1 under [`scenario_fill`](@ref), `mu` and `sigma`.

# The scenario cap

`pe.max_scenarios` cuts the returns matrix the result carries down to its last `max_scenarios` rows, through [`scenario_window`](@ref), and leaves `mu` and `sigma` fitted over every observation. The cut is taken **before** the fill, so the share the fill measures is the share of the window a consumer actually reads. A `max_scenarios` of `nothing`, the default, carries every row, and a cap at or above the number of observations is a `view` that copies nothing.

# The scenario fill

Step 8 takes the same fill the no-horizon method takes, on the **arithmetic** `X` the caller handed in and against the arithmetic moments the result carries. Under a mask-aware `pe.me` and `pe.ce` an asset that lists inside the window is investable and its column still carries a `NaN` at every earlier row; [`scenario_fill`](@ref) writes `0` there once, silently while the worst investable column's own filled share stays at or below the limit [`resolve_fill_limit`](@ref) derives, with a warning above it, and refuses any fill under `strict`. `pe.fill_limit` defaults to `nothing`, which derives `1 - min_coverage` from the arms' coverage floor and names every fill where no arm states one. A scenario-based measure then understates that asset's risk over the filled rows, and `mu` and `sigma` are untouched.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (ignored).
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `strict`: Whether a zero-filled scenario raises rather than warns. Any fill raises under `strict`; otherwise a column filled above the limit [`resolve_fill_limit`](@ref) derives warns, and every fill warns while that limit is `nothing`.
  - `kwargs...`: Additional keyword arguments passed to mean and covariance estimators.

# Validation

  - `dims in (1, 2)`.
  - At least one asset must be in the Coverage Universe.
  - `pe.fill_limit` is at most `1 - min_coverage` over the arms that state a coverage floor, else a `DomainError` is thrown at the fit, because a looser limit is dead by construction.
  - The zero-filled share of the worst investable column of `X` is at or below the limit [`resolve_fill_limit`](@ref) derives, else a warning naming the assets is emitted, or an `ArgumentError` naming them is raised under `strict`, which any fill raises. A limit of `nothing` — no arm states a floor and `pe.fill_limit` is `nothing` — is at or below no share, so any fill is named.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`coverage_mask`](@ref)
  - [`scenario_fill`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any}, X::MatNum,
               ::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    X = dims_oriented(dims, X)
    fill_limit = resolve_fill_limit(pe.fill_limit, coverage_floor(pe))
    X_log = log1p.(X)
    mu = vec(Statistics.mean(pe.me, X_log, pnl; dims = 1, kwargs...))
    sigma = Statistics.cov(pe.ce, X_log, pnl; dims = 1, kwargs...)
    horizon_moments!(mu, sigma, pe.horizon)
    Xs = scenario_window(pe.max_scenarios, X)
    # The fill is on the arithmetic `X` the caller handed in, and it is taken after step 7,
    # because the Investable Mask is read off the arithmetic moments the result carries.
    return LowOrderPrior(; X = scenario_fill(Xs, mu, sigma, strict, fill_limit), mu = mu,
                         sigma = sigma)
end

function factor_residual_config(::EmpiricalPrior)
    # An empirical prior estimates the asset covariance directly. There is no factor lift,
    # so there is no residual block to remove (see [`factor_residual_config`](@ref)).
    return nothing
end

export EmpiricalPrior
