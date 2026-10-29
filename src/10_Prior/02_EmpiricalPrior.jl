"""
$(DocStringExtensions.TYPEDEF)

Empirical prior estimator for asset returns.

`EmpiricalPrior` is a low order prior estimator. It fits the mean of the asset returns with `me` and their covariance with `ce`. With a `horizon`, it fits both moments on the log-returns, scales them to the horizon, and converts them back to arithmetic returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EmpiricalPrior(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        horizon::Option{<:Number} = nothing,
        fill_limit::Option{<:Real} = nothing,
        max_scenarios::Option{<:Integer} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> EmpiricalPrior

Keywords correspond to the struct's fields.

## Validation

  - If `horizon` is not `nothing`, `horizon > 0`.
  - If `fill_limit` is not `nothing`, `0 < fill_limit <= 1`. The constructor refuses zero. Where no arm states a coverage floor, `nothing` already names every fill. The fit refuses a value above `1 - min_coverage` of the binding floor of the arms, because such a value can never fire.
  - If `max_scenarios` is not `nothing`, `max_scenarios > 0`.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).

## The incremental fit

The empirical prior folds its moments and carries its rows. [`partial_fit!`](@ref) folds each observation into each member of `me` and `ce` that folds, and appends the row to a [`PriorCarryState`](@ref) of its own. The one-argument [`prior`](@ref) reads `mu` and `sigma` off the state of a folded member, and `X` off the carried rows. A member that does not fold runs its batch verb over the carried rows, see [`read_member`](@ref). When both members fold, the cost of a step grows with the square of the number of assets, and not with the number of observations folded. The prior carries the rows because [`LowOrderPrior`](@ref) holds `X` for the scenario risk measures.

## Two caps, two names

`max_scenarios` and the `max_history` of [`Online`](@ref) are two different caps.

  - `EmpiricalPrior(; max_scenarios = w)` fits `mu` and `sigma` over every observation, and carries the last `w` rows as `X`. The cap is a property of the result, so it cuts the same rows in batch and online. No batch fit gives the same result, because the moments and the scenarios come from different windows.
  - `Online(EmpiricalPrior(); max_history = w)` cuts the whole fit to the last `w` observations. The estimator refits from the buffer, so the result equals `prior(pe, X[end - w + 1:end, :])`.

When both caps are set, the moments come from the last `max_history` observations, and `X` comes from the last `max_scenarios` of those. A `max_scenarios` at or above the number of carried observations changes nothing.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `me`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ce`: Recursively viewed through [`port_opt_view`](@ref).
  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

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
  fill_limit ┼ nothing
       x_src ┴ nothing
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

  - $(ref_dict[:meucci2005]) Section 2.6.5, Equations 2.219 and 2.220, Section 3.2, Equation 3.64, and Section 3.3.1, Equation 3.80.
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
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function EmpiricalPrior(ce::StatsBase.CovarianceEstimator,
                            me::AbstractExpectedReturnsEstimator, horizon::Option{<:Number},
                            fill_limit::Option{<:Real}, max_scenarios::Option{<:Integer},
                            cache::Option{<:AbstractPartialFitState},
                            x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :EmpiricalPrior)
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
                   typeof(max_scenarios), typeof(cache), typeof(x_src)}(ce, me, horizon,
                                                                        fill_limit,
                                                                        max_scenarios,
                                                                        cache, x_src)
    end
end
function EmpiricalPrior(;
                        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        horizon::Option{<:Number} = nothing,
                        fill_limit::Option{<:Real} = nothing,
                        max_scenarios::Option{<:Integer} = nothing,
                        cache::Option{<:AbstractPartialFitState} = nothing,
                        x_src::Option{<:AbstractReturnsSource} = nothing)::EmpiricalPrior
    return EmpiricalPrior(ce, me, horizon, fill_limit, max_scenarios, cache, x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`EmpiricalPrior`](@ref) except `cache`, and `max_scenarios` only where it is set.

The state in `cache` belongs to an incremental fit and is not part of the configuration, and it would print under the estimator at every site that renders one. Most callers leave `max_scenarios` at `nothing`, and a `nothing` row would change the rendering of every estimator that holds a prior. The row appears only when a caller sets a cap. Set `set_show_nothing_fields!(:EmpiricalPrior, true)` to render both.

# Arguments

  - `pe`: Empirical prior estimator.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ce, :me, :horizon, :fill_limit, :x_src)` with no scenario cap and `(:ce, :me, :horizon, :fill_limit, :max_scenarios, :x_src)` with one.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(pe::EmpiricalPrior)
    return if isnothing(pe.max_scenarios)
        (:ce, :me, :horizon, :fill_limit, :x_src)
    else
        (:ce, :me, :horizon, :fill_limit, :max_scenarios, :x_src)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`EmpiricalPrior`](@ref) method of [`coverage_floor`](@ref), the binding floor of its two arms.

`pe.me` and `pe.ce` can each carry a [`CoveragePolicy`](@ref). The Investable Mask is the conjunction of the two admissions, and both admissions read the same count of observations per asset. So each floor bounds every investable column, and the binding floor is the maximum of the floors that the arms state. A configuration where only one arm states a floor needs no rule of its own.

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

This is the one copy of the horizon conversion of [`EmpiricalPrior`](@ref). The batch method calls it after it fits `pe.me` and `pe.ce` on `log1p.(X)`. `prior(pe)` with no data calls it after it reads the same two moments off their states. The two routes then run the same arithmetic.

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

Steps 3 to 5 must run in this order. Step 4 reads the `mu` that step 3 left, which is ``\\hat{\\mu}_i + 1``, because step 5 has not yet subtracted the one. The second closed form needs that factor. If step 5 runs before step 4, each ``\\hat{\\mu}_i + 1`` becomes ``\\hat{\\mu}_i``. Entry ``(i,\\, j)`` of the covariance then falls by the factor ``\\hat{\\mu}_i \\hat{\\mu}_j / ((\\hat{\\mu}_i + 1)(\\hat{\\mu}_j + 1))``, which is about six orders of magnitude on a daily return series.

 1. Scale `mu` by `horizon`.
 2. Scale `sigma` by `horizon`.
 3. Overwrite `mu` with the exponential term of the first closed form. This is the arithmetic mean plus one.
 4. Overwrite `sigma` with the second closed form, whose ``\\hat{\\mu}_i + 1`` factors are the `mu` of step 3.
 5. Subtract one from `mu`.

# Arguments

  - `mu`: The log-return mean, overwritten with the arithmetic mean.
  - `sigma`: The log-return covariance, overwritten with the arithmetic covariance.
  - `horizon`: The investment horizon.

# Returns

  - `nothing`. The function changes both arguments in place.

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

Compute the empirical prior moments of asset returns with no horizon conversion.

`pe.me` fits the mean and `pe.ce` fits the covariance on `X` itself, with no log transform and no horizon scale. The method returns them in a [`LowOrderPrior`](@ref).

# Mathematical definition

The two moments are what `pe.me` and `pe.ce` return. The default pair, [`SimpleExpectedReturns`](@ref) and [`PortfolioOptimisersCovariance`](@ref) with no observation weights, gives the sample moments:

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

Any other configuration of `pe.me` or `pe.ce` changes the result. For example, a shrunk mean or a denoised covariance differs from the display above.

This method takes the arithmetic moments of `X`. It applies no log transform, so it is not the ``h = 1`` case of the horizon method. The horizon method converts the returns to ``\\log(1 + x_t)`` and back, and that round trip leaves the moments unchanged only in the limit of small returns.

# The scenario cap

`pe.max_scenarios` cuts the returns matrix of the result to its last `max_scenarios` rows, through [`scenario_window`](@ref). `mu` and `sigma` stay fitted over every observation. The cut runs before the fill, so the fill measures its share over the rows that a consumer reads. The default, `nothing`, carries every row. A cap at or above the number of observations returns `X` itself, and a smaller cap returns a `view`, so the cut copies nothing.

# The counts of the moments

The result states in `eoc` the effective number of observations of `mu` and of `sigma`, through [`scenario_moment_counts`](@ref). `pe.me` and `pe.ce` each state the count of their own moment with [`effective_observation_count`](@ref). An unweighted estimator is worth `size(X, 1)` observations, and a weighted one, such as an exponentially weighted covariance, is worth the count of its weights. When the cap cuts, the count of an unweighted moment is the number of observations that the moments read. A consumer that prices a sample size reads these counts and not the number of carried rows. An entry equal to the number of carried rows stays `nothing`.

# The scenario fill

A mask-aware `pe.me` and `pe.ce` fit an asset that lists inside the window from the observations it has. The exponentially weighted family is mask-aware, and so is the plain family wherever a [`CoveragePolicy`](@ref) is set. Such an asset is investable, and its column of `X` still carries a `NaN` at every earlier row. [`scenario_fill`](@ref) writes `0` at those entries once, so that every consumer of the result reads a finite investable column. A non-investable asset keeps its whole `NaN` column, and the fill does not change `mu` or `sigma`.

The fill costs accuracy at four consumers, and the message names all four:

  - A scenario-based risk measure understates the risk of the asset over the filled rows.
  - A hierarchical optimiser can put the asset in a branch of its own and overweight it.
  - An entropy pooling view on the asset is calibrated on the filled column.
  - A meta-optimiser carries the fill into the outer problem.

The fill is silent while the filled share of the worst investable column stays at or below the limit that [`resolve_fill_limit`](@ref) derives. Above the limit it warns, and under `strict` it refuses any fill. `pe.fill_limit` defaults to `nothing`. Where an arm states a coverage floor, `nothing` derives `1 - min_coverage`, which never fires. Where no arm states a floor, `nothing` names every fill. Under a plain estimator with no policy the fill never fires, because an asset that the estimator cannot cover leaves the Coverage Universe and is not investable.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (ignored).
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `strict`: Whether a zero-filled scenario raises an error instead of a warning. Under `strict`, any fill raises. Otherwise a column filled above the limit that [`resolve_fill_limit`](@ref) derives warns, and every fill warns while that limit is `nothing`. `strict` also reaches the covariance, where a peel of a [`CoveragePolicy`](@ref) raises under `strict` and warns otherwise.
  - `kwargs...`: Additional keyword arguments, passed to the mean estimator and the covariance estimator.

# Validation

  - `dims in (1, 2)`.
  - At least one asset must be in the Coverage Universe.
  - `pe.fill_limit` is at most `1 - min_coverage` of the binding floor of the arms, see [`coverage_floor`](@ref). Otherwise the fit raises a `DomainError`, because a looser limit can never fire.
  - The zero-filled share of the worst investable column of `X` is at or below the limit that [`resolve_fill_limit`](@ref) derives. Otherwise the fit warns and names the assets. Under `strict`, any fill raises an `ArgumentError` that names them. The limit is `nothing` when no arm states a floor and `pe.fill_limit` is `nothing`, and no share passes a limit of `nothing`, so the fit names every fill.

# Returns

  - `pr::LowOrderPrior`: The prior result, which holds the asset returns, the mean vector and the covariance matrix.

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
    sigma = Statistics.cov(library_covariance_estimator(pe.ce), X, pnl; dims = 1,
                           strict = strict, kwargs...)
    # The Scenario Cap cuts the rows the result carries and leaves the moments above it
    # alone, so the fill runs over the matrix a consumer actually reads and its share is
    # measured against that window (see [`scenario_window`](@ref)).
    Xs = scenario_window(pe.max_scenarios, X)
    # A mask-aware estimator answers a young asset from the rows it has, so the asset is
    # investable and its column still carries the gap. The fill is paid once, here, because
    # every consumer of the result reads that column (see [`scenario_fill`](@ref)).
    return LowOrderPrior(; X = scenario_fill(Xs, mu, sigma, strict, fill_limit), mu = mu,
                         sigma = sigma, eoc = scenario_moment_counts(pe, X, size(Xs, 1)))
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number, <:Any}, X::MatNum,
          F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute the empirical prior moments of asset returns at an investment horizon.

`pe.me` and `pe.ce` fit the mean and the covariance of the log-returns. [`horizon_moments!`](@ref) scales both moments to the horizon and converts them back to arithmetic returns. The method returns them in a [`LowOrderPrior`](@ref).

# Mathematical definition

`pe.me` and `pe.ce` fit the log-returns ``\\log(1 + x_t)``, not `X`. The method scales the two log-moments by the investment horizon ``h``, and then converts them back to arithmetic returns:

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

The `X` of the returned [`LowOrderPrior`](@ref) is the arithmetic returns matrix that the caller gave. Only the moments come from the log-returns.

# Algorithm

Steps 5 to 7 must run in this order. Step 6 reads the `mu` that step 5 left, which is ``\\hat{\\mu}_i + 1``, because step 7 has not yet subtracted the one. The second closed form needs that factor, so the order supplies it with no second computation. If step 7 runs before step 6, each ``\\hat{\\mu}_i + 1`` becomes ``\\hat{\\mu}_i``. Entry ``(i,\\, j)`` of the covariance then falls by the factor ``\\hat{\\mu}_i \\hat{\\mu}_j / ((\\hat{\\mu}_i + 1)(\\hat{\\mu}_j + 1))``, which is about six orders of magnitude on a daily return series. Steps 4 to 7 are [`horizon_moments!`](@ref).

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Take the log-returns `X_log`, as `log1p.(X)`.
 3. Fit the log-return mean `mu` on `X_log` with `pe.me`, and the log-return covariance `sigma` on `X_log` with `pe.ce`.
 4. Scale both by `pe.horizon` to get ``\\tilde{\\boldsymbol{\\mu}}`` and ``\\tilde{\\mathbf{\\Sigma}}``.
 5. Overwrite `mu` with the exponential term of the first closed form. This is the arithmetic mean plus one, because step 7 has not run yet.
 6. Overwrite `sigma` with the second closed form, whose ``\\hat{\\mu}_i + 1`` factors are the `mu` of step 5.
 7. Subtract one from `mu` to get ``\\hat{\\boldsymbol{\\mu}}``.
 8. Cut the arithmetic `X` of step 1 with [`scenario_window`](@ref), and fill it with [`scenario_fill`](@ref). Return a [`LowOrderPrior`](@ref) that holds this `X`, `mu`, `sigma` and the `eoc` of [`scenario_moment_counts`](@ref) over `X_log`.

# The scenario cap

`pe.max_scenarios` cuts the returns matrix of the result to its last `max_scenarios` rows, through [`scenario_window`](@ref). `mu` and `sigma` stay fitted over every observation. The cut runs before the fill, so the fill measures its share over the rows that a consumer reads. The default, `nothing`, carries every row. A cap at or above the number of observations returns `X` itself, and a smaller cap returns a `view`, so the cut copies nothing.

# The counts of the moments

The result states in `eoc` the effective number of observations of `mu` and of `sigma`, through [`scenario_moment_counts`](@ref). `pe.me` and `pe.ce` each state the count of their own moment with [`effective_observation_count`](@ref). An unweighted estimator is worth `size(X, 1)` observations, and a weighted one, such as an exponentially weighted covariance, is worth the count of its weights. When the cap cuts, the count of an unweighted moment is the number of observations that the moments read. A consumer that prices a sample size reads these counts and not the number of carried rows. An entry equal to the number of carried rows stays `nothing`.

# The scenario fill

Step 8 applies the fill of the method with no horizon. It fills the arithmetic `X` that the caller gave, and it reads the Investable Mask off the arithmetic moments that the result carries. Under a mask-aware `pe.me` and `pe.ce`, an asset that lists inside the window is investable, and its column still carries a `NaN` at every earlier row. [`scenario_fill`](@ref) writes `0` at those entries once. The fill is silent while the filled share of the worst investable column stays at or below the limit that [`resolve_fill_limit`](@ref) derives. Above the limit it warns, and under `strict` it refuses any fill.

`pe.fill_limit` defaults to `nothing`. Where an arm states a coverage floor, `nothing` derives `1 - min_coverage`. Where no arm states a floor, `nothing` names every fill. A scenario-based risk measure understates the risk of a filled asset over the filled rows. The fill does not change `mu` or `sigma`.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor returns matrix (ignored).
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `strict`: Whether a zero-filled scenario raises an error instead of a warning. Under `strict`, any fill raises. Otherwise a column filled above the limit that [`resolve_fill_limit`](@ref) derives warns, and every fill warns while that limit is `nothing`. `strict` also reaches the covariance, where a peel of a [`CoveragePolicy`](@ref) raises under `strict` and warns otherwise.
  - `kwargs...`: Additional keyword arguments, passed to the mean estimator and the covariance estimator.

# Validation

  - `dims in (1, 2)`.
  - At least one asset must be in the Coverage Universe.
  - `pe.fill_limit` is at most `1 - min_coverage` of the binding floor of the arms, see [`coverage_floor`](@ref). Otherwise the fit raises a `DomainError`, because a looser limit can never fire.
  - The zero-filled share of the worst investable column of `X` is at or below the limit that [`resolve_fill_limit`](@ref) derives. Otherwise the fit warns and names the assets. Under `strict`, any fill raises an `ArgumentError` that names them. The limit is `nothing` when no arm states a floor and `pe.fill_limit` is `nothing`, and no share passes a limit of `nothing`, so the fit names every fill.

# Returns

  - `pr::LowOrderPrior`: The prior result, which holds the asset returns, the mean vector and the covariance matrix.

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
    sigma = Statistics.cov(library_covariance_estimator(pe.ce), X_log, pnl; dims = 1,
                           strict = strict, kwargs...)
    horizon_moments!(mu, sigma, pe.horizon)
    Xs = scenario_window(pe.max_scenarios, X)
    # The fill is on the arithmetic `X` the caller handed in, and it is taken after step 7,
    # because the Investable Mask is read off the arithmetic moments the result carries.
    return LowOrderPrior(; X = scenario_fill(Xs, mu, sigma, strict, fill_limit), mu = mu,
                         sigma = sigma,
                         eoc = scenario_moment_counts(pe, X_log, size(Xs, 1)))
end

function factor_residual_config(::EmpiricalPrior)
    # An empirical prior estimates the asset covariance directly. There is no factor lift,
    # so there is no residual block to remove (see [`factor_residual_config`](@ref)).
    return nothing
end

export EmpiricalPrior
