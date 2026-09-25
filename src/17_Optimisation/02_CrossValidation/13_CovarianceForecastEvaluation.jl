"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the realised quantities that the evaluation judges a covariance forecast against.

The evaluation forms a covariance forecast ``\\hat{\\mathbf{\\Sigma}}_t`` at a step and judges it on the ``h`` returns that follow the step. A member of this family chooses the statistic of those returns that stands in for the covariance they were drawn from. The two members the library defines take the realised covariance of the returns and the outer product of their sum. When the returns are serially uncorrelated, both statistics have the expectation ``h\\, \\mathbf{\\Sigma}_t``, and at ``h = 1`` they are equal. So the member changes the variance of the per-step diagnostic and the question that it answers, but not its target.

All concrete subtypes should subtype `AbstractRealisedTarget`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype
`AbstractRealisedTarget` and implement the following methods:

## Required method name

  - `realised_target(target::AbstractRealisedTarget, Zc::MatNum) -> Tuple{MatNum, AbstractMatrix{<:Integer}}`: Return the realised matrix of the centred test rows, and the count of rows over which each of its cells was formed.
  - `target_dof(target::AbstractRealisedTarget, n::Integer, h::Integer) -> Integer`: Return the degrees of freedom of the step's Mahalanobis statistic under a Gaussian null, for `n` active assets and a horizon of `h`.
  - `target_step_dof(target::AbstractRealisedTarget, h::Integer) -> Integer`: Return the degrees of freedom of one asset's, or one portfolio's, step ratio under a Gaussian null.

### Arguments

  - `target`: The concrete subtype instance.
  - `Zc`: The centred test rows, `observations × assets`. A cell that an asset did not trade is non-finite.
  - `n`: The number of active assets at the step.
  - `h`: The horizon of the step.

### Returns

  - `S::MatNum`: The realised matrix, `assets × assets`.
  - `H::AbstractMatrix{<:Integer}`: The count of rows at which both assets of each cell are finite.
  - `dof::Integer`: The degrees of freedom.

# Examples

```jldoctest
julia> RealisedCovariance()
RealisedCovariance()
```

# Related

  - [`RealisedCovariance`](@ref)
  - [`HorizonReturn`](@ref)
  - [`realised_target`](@ref)
  - [`covariance_forecast_step`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
"""
abstract type AbstractRealisedTarget <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Judges a covariance forecast against the realised covariance of the returns that follow it.

This is the default target. It sums the outer products of the centred returns of the step, one per row. Under a Gaussian null the Mahalanobis ratio of the step then has variance ``2 / (N h)``, smaller by a factor of ``h`` than under [`HorizonReturn`](@ref). The two targets cost the same to compute, so the default is the one with the smaller variance. It asks whether the forecast describes the dispersion of each observation.

# Mathematical definition

```math
\\begin{align}
\\mathbf{S}_t &= \\sum_{s=1}^{h} \\boldsymbol{z}_{t+s} \\boldsymbol{z}_{t+s}^\\intercal\\,, \\\\
\\mathbb{E}\\left[\\mathbf{S}_t \\mid \\mathcal{F}_t\\right] &= h\\, \\mathbf{\\Sigma}_t\\,.
\\end{align}
```

Where:

  - $(math_dict[:S_t_realised])
  - $(math_dict[:h_step])
  - ``\\boldsymbol{z}_{t+s}``: Centred return of observation ``t + s``, the return less the location the forecast is about.
  - ``\\mathbf{\\Sigma}_t``: True conditional covariance of the returns that follow step ``t``.
  - ``\\mathcal{F}_t``: Information available at step ``t``.

The expectation holds when the ``h`` returns are serially uncorrelated. It makes ``\\mathbf{S}_t / h`` a conditionally unbiased proxy of ``\\mathbf{\\Sigma}_t`` for every distribution of the returns. At ``h = 1`` the proxy has rank one and equals the proxy of [`HorizonReturn`](@ref). A cell whose two assets share no finite row is zero, and its count is zero.

# Examples

```jldoctest
julia> RealisedCovariance()
RealisedCovariance()
```

# Related

  - [`AbstractRealisedTarget`](@ref)
  - [`HorizonReturn`](@ref)
  - [`realised_target`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
"""
struct RealisedCovariance <: AbstractRealisedTarget end
"""
$(DocStringExtensions.TYPEDEF)

Judges a covariance forecast against the outer product of the return earned over the horizon.

It asks the question of a holder over the ``h`` observations of the step. The return summed over the step is what a book held for the whole horizon earns, and its outer product is a rank-one proxy of ``h\\, \\mathbf{\\Sigma}_t``. Under a Gaussian null the Mahalanobis ratio of the step has variance ``2 / N`` at every ``h``, so this target is the noisier of the two at every horizon above one.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{R}_t &= \\sum_{s=1}^{h} \\boldsymbol{z}_{t+s}\\,, \\\\
\\mathbf{S}_t &= \\boldsymbol{R}_t \\boldsymbol{R}_t^\\intercal\\,, \\\\
\\mathbb{E}\\left[\\mathbf{S}_t \\mid \\mathcal{F}_t\\right] &= h\\, \\mathbf{\\Sigma}_t\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{R}_t``: Return earned over the horizon of step ``t``, summed over the finite observations of each asset.
  - $(math_dict[:S_t_realised])
  - $(math_dict[:h_step])
  - ``\\boldsymbol{z}_{t+s}``: Centred return of observation ``t + s``, the return less the location the forecast is about.
  - ``\\mathbf{\\Sigma}_t``: True conditional covariance of the returns that follow step ``t``.
  - ``\\mathcal{F}_t``: Information available at step ``t``.

The expectation holds when the ``h`` returns are serially uncorrelated. At ``h = 1`` the proxy equals the proxy of [`RealisedCovariance`](@ref). A non-finite cell adds nothing to the sum, and the count of a cell is the number of rows at which both of its assets are finite.

# Examples

```jldoctest
julia> HorizonReturn()
HorizonReturn()
```

# Related

  - [`AbstractRealisedTarget`](@ref)
  - [`RealisedCovariance`](@ref)
  - [`realised_target`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
"""
struct HorizonReturn <: AbstractRealisedTarget end
"""
    realised_target(target::RealisedCovariance, Zc::MatNum)
    realised_target(target::HorizonReturn, Zc::MatNum)

Form the realised matrix of a step and the count of rows behind each of its cells.

These are the two methods of the [`AbstractRealisedTarget`](@ref) interface that the library defines. A non-finite cell of `Zc` is an observation that the asset did not trade. It adds nothing to the sum, and the count leaves it out. The count `H` holds, for each pair of assets, the number of rows at which both are finite. The kernel scales the forecast by it cell by cell, `H ⊙ Σ̂`, so a gap in the test window changes the target and not the reading.

# Algorithm

 1. Mark the finite cells of `Zc`, giving `fin`.
 2. Replace each non-finite cell of `Zc` with zero, giving `Zf`.
 3. Under [`RealisedCovariance`](@ref), form the realised matrix `S = Zf' Zf`. Under [`HorizonReturn`](@ref), sum each column of `Zf`, giving `R`, and form `S = R R'`.
 4. Count the rows at which both assets of each pair are finite, giving `H = fin' fin`.

# Arguments

  - `target`: The realised quantity.
  - `Zc`: The centred test rows, `observations × assets`.

# Returns

  - `S::MatNum`: The realised matrix, `assets × assets`.
  - `H::AbstractMatrix{<:Integer}`: The pairwise count of finite rows.

# Related

  - [`AbstractRealisedTarget`](@ref)
  - [`RealisedCovariance`](@ref)
  - [`HorizonReturn`](@ref)
  - [`covariance_forecast_step`](@ref)
"""
function realised_target(::RealisedCovariance, Zc::MatNum)
    fin = isfinite.(Zc)
    Zf = ifelse.(fin, Zc, zero(eltype(Zc)))
    return transpose(Zf) * Zf, transpose(fin) * fin
end
function realised_target(::HorizonReturn, Zc::MatNum)
    fin = isfinite.(Zc)
    Zf = ifelse.(fin, Zc, zero(eltype(Zc)))
    R = vec(sum(Zf; dims = 1))
    return R * transpose(R), transpose(fin) * fin
end
"""
    target_dof(target::RealisedCovariance, n::Integer, h::Integer)
    target_dof(target::HorizonReturn, n::Integer, h::Integer)

Degrees of freedom of a step's Mahalanobis statistic under a Gaussian null, by realised quantity.

The statistic is the trace of the whitened realised matrix. Under [`RealisedCovariance`](@ref) it has `n * h` degrees of freedom, because each of the `h` returns adds `n` whitened coordinates. Under [`HorizonReturn`](@ref) it has `n`, because the horizon return is one draw at every `h`. The summary reads the count to weight the steps of a run and to set the width of the band on the mean ratio. The exceedance rate reads it for its chi-squared threshold. Under a Gaussian null the statistic is chi-squared on this count when the test window has no gap, and only then.

# Arguments

  - `target`: The realised quantity.
  - `n`: The number of active assets at the step.
  - `h`: The horizon of the step.

# Returns

  - `dof::Integer`: The degrees of freedom.

# Related

  - [`AbstractRealisedTarget`](@ref)
  - [`target_step_dof`](@ref)
  - [`covariance_forecast_summary`](@ref)
"""
function target_dof(::RealisedCovariance, n::Integer, h::Integer)
    return n * h
end
function target_dof(::HorizonReturn, n::Integer, ::Integer)
    return n
end
"""
    target_step_dof(target::RealisedCovariance, h::Integer)
    target_step_dof(target::HorizonReturn, h::Integer)

Degrees of freedom of one asset's, or one portfolio's, step ratio under a Gaussian null, by realised quantity.

The count is `h` under [`RealisedCovariance`](@ref), one squared whitened coordinate per return. It is `1` under [`HorizonReturn`](@ref), one draw at every `h`. The summary reads it to weight the steps of the diagonal ratio and to set the width of its band.

# Arguments

  - `target`: The realised quantity.
  - `h`: The horizon of the step.

# Returns

  - `dof::Integer`: The degrees of freedom.

# Related

  - [`AbstractRealisedTarget`](@ref)
  - [`target_dof`](@ref)
  - [`covariance_forecast_summary`](@ref)
"""
function target_step_dof(::RealisedCovariance, h::Integer)
    return h
end
function target_step_dof(::HorizonReturn, ::Integer)
    return one(Int)
end
"""
    forecast_location(ce::Covariance, X::MatNum; dims::Int = 1, active_mask = nothing, kwargs...)
    forecast_location(ce::GeneralCovariance, X::MatNum; dims::Int = 1, kwargs...)
    forecast_location(ce::Union{<:ExpWeightedCovariance, <:RegimeAdjustedExpWeightedCovariance}, X::MatNum; dims::Int = 1, active_mask = nothing, kwargs...)
    forecast_location(ce::Union{<:PortfolioOptimisersCovariance, <:CorrelationCovariance}, X::MatNum; kwargs...)
    forecast_location(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    forecast_location(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    forecast_location(ce::Union{<:ExpWeightedCovariance, <:RegimeAdjustedExpWeightedCovariance}, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    forecast_location(ce::PortfolioOptimisersCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    forecast_location(ce::Covariance)
    forecast_location(ce::GeneralCovariance)
    forecast_location(ce::Union{<:ExpWeightedCovariance, <:RegimeAdjustedExpWeightedCovariance})
    forecast_location(ce::PortfolioOptimisersCovariance)
    forecast_location(ce::CorrelationCovariance)
    forecast_location(ce::AbstractCovarianceEstimator)
    forecast_location(pe::AbstractPriorEstimator, rd::ReturnsResult; strict::Bool = false, kwargs...)
    forecast_location(pe::AbstractPriorEstimator; strict::Bool = false, kwargs...)

Read the location that a covariance forecast is about, off the estimator that formed it.

A second moment is always taken about some centre, ``\\mathbb{E}[(\\boldsymbol{z} - \\boldsymbol{c})(\\boldsymbol{z} - \\boldsymbol{c})^\\intercal] = \\hat{\\mathbf{\\Sigma}}``, and the centre belongs to the estimator. It is the mean that its centring estimator fitted, the weighted sample mean that it subtracted, the exponentially weighted location that its state carries, or zero for an estimator that assumes a zero mean. An evaluation against the raw outer product ``\\boldsymbol{z}\\boldsymbol{z}^\\intercal`` would judge the forecast against a proxy of ``\\mathbf{\\Sigma} + (\\boldsymbol{\\mu} - \\boldsymbol{c})(\\boldsymbol{\\mu} - \\boldsymbol{c})^\\intercal``. It would then read a bias of ``\\boldsymbol{c}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{c} / N`` into a forecast that has none. So the evaluation centres the test rows on this location. No flag turns the centring off, because the location belongs to the estimator and not to the caller.

Each family has two arities. The data form answers for a batch fit over `X`. The data-less form answers for the state that the estimator carries after [`partial_fit!`](@ref), so a refit fold and a stepped fold centre on the same quantity. For a family state, the data-less form returns the vector of the state itself, which the next fold writes in place.

  - [`Covariance`](@ref) answers the centre that [`weighted_centre`](@ref) resolves from `ce.me` and `ce.w`, or `state.mu`. Under a [`CoveragePolicy`](@ref) the batch fit and the fold are one computation, so the data form folds the window and reads the same `state.mu`. That vector is the diagonal of the per-pair centre, the available-case mean of each asset. The forecast centres each off-diagonal product on the mean of the observations that the pair shares, and one vector cannot carry that centre. The per-asset ratios read the diagonal only.
  - [`GeneralCovariance`](@ref) answers the sample mean, weighted by `ce.w` when it carries observation weights, or `state.mu`.
  - [`ExpWeightedCovariance`](@ref) and [`RegimeAdjustedExpWeightedCovariance`](@ref) answer `state.location`, or zero for every asset when `centred = true`, because the estimator then assumes a zero mean. The data form runs over the window the same pass that the batch fit runs, and reads the location at the end of the pass.
  - [`PortfolioOptimisersCovariance`](@ref) and [`CorrelationCovariance`](@ref) answer what the estimator that they hold answers, because a matrix transform moves no centre. A `StatsBase.CovarianceEstimator` that they hold is read as a [`GeneralCovariance`](@ref), as `cov` reads it.
  - Any other [`AbstractCovarianceEstimator`](@ref) answers the sample mean of the window, over the finite rows of each column. Its data-less form reads the state through [`forecast_state_location`](@ref), which refuses a state of a shape that it does not know.
  - Under [`Online`](@ref), every family answers its data form over the rows and masks of the buffer, because a buffer means the batch verb over its rows at every read-out.
  - An [`AbstractPriorEstimator`](@ref) answers the centre of its `sigma` through [`prior_forecast_location`](@ref). That is the `mu` it publishes for a family that forms both moments about one mean. An [`EmpiricalPrior`](@ref) whose `me` shrinks the mean publishes the shrunk `mu`, but its `ce` centres the covariance on its own estimate of the mean, so it answers the location of its `ce`. A factor prior and a high-order prior answer the centre of the prior that they lift or wrap. The keyword `strict` reaches [`prior`](@ref) alone.

The Asset Panel form follows the moment seam of the estimator's own `cov`. Under no policy it reduces the window to its Coverage Universe and frames the answer with `NaN` outside it. Under a policy, and for the two mask-aware exponentially weighted families, it hands the active mask of the panel to the estimator. A [`PortfolioOptimisersCovariance`](@ref) hands the panel to the estimator that it holds, as its `cov` does.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:pe])
  - $(arg_dict[:X])
  - $(arg_dict[:rd])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the window, `observations × assets`, or `nothing`.
  - `kwargs...`: Additional keyword arguments passed to the centring estimator.

# Validation

  - The data-less form requires a state. An `ArgumentError` is thrown otherwise.
  - The data-less form requires a state of a shape that [`forecast_state_location`](@ref) reads: a [`CovarianceState`](@ref), an exponentially weighted state or a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `c::VecNum`: The location, `assets × 1`.

# Related

  - [`covariance_forecast_step`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`weighted_centre`](@ref)
  - [`partial_fit!`](@ref)
  - [`prior_forecast_location`](@ref)
"""
function forecast_location(ce::Covariance, X::MatNum; dims::Int = 1,
                           active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           kwargs...)
    return forecast_location(ce, ce.cvg, X; dims = dims, active_mask = active_mask,
                             kwargs...)
end
function forecast_location(ce::Covariance, ::Nothing, X::MatNum; dims::Int = 1,
                           active_mask = nothing, estimation_mask = nothing, kwargs...)
    return vec(weighted_centre(X, ce.me, ce.w; dims = dims, kwargs...))
end
function forecast_location(ce::Covariance, cvg::CoveragePolicy, X::MatNum; dims::Int = 1,
                           active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           estimation_mask = nothing, kwargs...)
    ce = partial_fit!(Covariance(; me = ce.me, ce = ce.ce, alg = ce.alg, w = ce.w,
                                 cvg = cvg), X; dims = dims, active_mask = active_mask)
    return forecast_location(ce)
end
function forecast_location(ce::GeneralCovariance, X::MatNum; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    w = get_observation_weights(ce.w, X; dims = 1, kwargs...)
    return if isnothing(w)
        vec(Statistics.mean(X; dims = 1))
    else
        vec(Statistics.mean(X, w; dims = 1))
    end
end
function forecast_location(ce::Union{<:ExpWeightedCovariance,
                                     <:RegimeAdjustedExpWeightedCovariance}, X::MatNum;
                           dims::Int = 1,
                           active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           kwargs...)
    ce = rebuild_estimator(ce, (; cache = nothing))
    return forecast_location(partial_fit!(ce, X; dims = dims, active_mask = active_mask,
                                          kwargs...))
end
function forecast_location(ce::Union{<:PortfolioOptimisersCovariance,
                                     <:CorrelationCovariance}, X::MatNum; kwargs...)
    return forecast_location(library_covariance_estimator(ce.ce), X; kwargs...)
end
function forecast_location(::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                           kwargs...)
    return finite_column_mean(dims_oriented(dims, X))
end
function forecast_location(ce::AbstractCovarianceEstimator, X::MatNum,
                           pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    return coverage_panel_moment(forecast_location, ce, forecast_coverage_policy(ce), X,
                                 pnl, (m, cmsk) -> expand_moment(m, cmsk, 1); dims = dims,
                                 kwargs...)
end
function forecast_location(ce::Union{<:ExpWeightedCovariance,
                                     <:RegimeAdjustedExpWeightedCovariance}, X::MatNum,
                           pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
    return forecast_location(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
function forecast_location(ce::PortfolioOptimisersCovariance, X::MatNum,
                           pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    return forecast_location(library_covariance_estimator(ce.ce), X, pnl; dims = dims,
                             kwargs...)
end
function forecast_location(ce::Covariance)
    return forecast_state_location(ce, ce.cache)
end
function forecast_location(ce::GeneralCovariance)
    return forecast_state_location(ce, ce.cache)
end
function forecast_location(ce::Union{<:ExpWeightedCovariance,
                                     <:RegimeAdjustedExpWeightedCovariance})
    c = forecast_state_location(ce, ce.cache)
    # Under `centred = true` the estimator never writes its location, so the state keeps the
    # seed, which is `NaN` for an asset that was inactive at the first observation; the
    # location such an estimator is about is zero for every asset.
    return ce.centred ? zero(c) : c
end
function forecast_location(ce::PortfolioOptimisersCovariance)
    return forecast_state_location(ce, ce.cache)
end
function forecast_location(ce::CorrelationCovariance)
    return forecast_location(ce.ce)
end
function forecast_location(ce::AbstractCovarianceEstimator)
    return forecast_state_location(ce, partial_fit_cache(ce))
end
function forecast_location(pe::AbstractPriorEstimator, rd::ReturnsResult;
                           strict::Bool = false, kwargs...)
    return prior_forecast_location(pe, prior(pe, rd; strict = strict, kwargs...), rd;
                                   kwargs...)
end
function forecast_location(pe::AbstractPriorEstimator; strict::Bool = false, kwargs...)
    return prior_forecast_location(pe, prior(pe; strict = strict, kwargs...); kwargs...)
end
"""
    prior_forecast_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult, rd::ReturnsResult; kwargs...)
    prior_forecast_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult, X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    prior_forecast_location(pe::EmpiricalPrior, pr::AbstractPriorResult, X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    prior_forecast_location(pe::FactorPrior, pr::AbstractPriorResult, X::MatNum, F::MatNum, pnl::Option{<:AssetPanel}; kwargs...)
    prior_forecast_location(pe::Union{<:HighOrderPriorEstimator, <:HighOrderFactorPriorEstimator}, pr::HighOrderPrior, X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    prior_forecast_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult; kwargs...)
    prior_forecast_location(pe::HighOrderPriorEstimator, pr::HighOrderPrior; kwargs...)

Read the location that the `sigma` of a prior is about, from the estimator and the result `pr` that it fitted.

The rule is the same for every family. The location is the `mu` that the prior would publish if its mean estimator were the centring estimator of its covariance estimator. Only a family that takes its two moments from two independent estimators, or that lifts the moments of such a prior, answers something other than `pr.mu`:

  - [`EmpiricalPrior`](@ref) answers [`forecast_location`](@ref) of its `ce` over the rows that `ce` saw. The horizon arm fits `ce` on the log-returns, so it maps that log location through the map that `mu` takes, ``c_i = \\exp(h c_{\\log, i} + \\tfrac{1}{2} h \\sigma_{\\log, ii}) - 1``. It reads ``h \\sigma_{\\log, ii} = \\log(1 + \\sigma_{ii} / (\\mu_i + 1)^2)`` back off the published pair.
  - [`FactorPrior`](@ref) answers the lift of the location of its factor prior, ``\\mathbf{M} \\boldsymbol{c}_f + \\boldsymbol{b}``, which it writes as ``\\boldsymbol{\\mu} + \\mathbf{M} (\\boldsymbol{c}_f - \\boldsymbol{\\mu}_f)``.
  - [`HighOrderPriorEstimator`](@ref) and [`HighOrderFactorPriorEstimator`](@ref) answer the location of the prior that they wrap, because they publish its `mu` and `sigma` unchanged.
  - Every other family answers `pr.mu`. The Black–Litterman and entropy pooling families and [`OpinionPoolingPrior`](@ref) form `sigma` about their posterior or pooled `mu`. [`CrossSectionalFactorPrior`](@ref) and [`LowDimensionEnsemblePrior`](@ref) publish a forecast of the next return as `mu`.

For the default [`EmpiricalPrior`](@ref), the mean estimator is the centring estimator of `ce`, so every family answers `pr.mu`. The data-less form reads the state as [`prior`](@ref) does, through [`prior_state_location`](@ref).

# Arguments

  - $(arg_dict[:pe])
  - $(arg_dict[:pr]) It is the result that `pe` fitted over the same rows.
  - $(arg_dict[:rd])
  - `X`: Asset returns matrix, `observations × assets`.
  - `F`: Factor returns matrix, `observations × factors`, or `nothing`.
  - $(arg_dict[:pnl_moment])
  - `kwargs...`: Additional keyword arguments, passed to the centring estimator as [`prior`](@ref) passes them to `ce`.

# Returns

  - `c::VecNum`: The location, `assets × 1`.

# Related

  - [`forecast_location`](@ref)
  - [`forecast_moments`](@ref)
  - [`prior_state_location`](@ref)
  - [`prior`](@ref)
"""
function prior_forecast_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult,
                                 rd::ReturnsResult; kwargs...)
    return prior_forecast_location(pe, pr, rd.X, rd.F, rd.pnl; iv = rd.iv, ivpa = rd.ivpa,
                                   kwargs...)
end
function prior_forecast_location(::AbstractPriorEstimator, pr::AbstractPriorResult,
                                 ::MatNum, ::Option{<:MatNum}, ::Option{<:AssetPanel};
                                 kwargs...)
    return pr.mu
end
function prior_forecast_location(pe::EmpiricalPrior{<:Any, <:Any, Nothing},
                                 ::AbstractPriorResult, X::MatNum, ::Option{<:MatNum},
                                 pnl::Option{<:AssetPanel}; kwargs...)
    return forecast_location(library_covariance_estimator(pe.ce), X, pnl; dims = 1,
                             kwargs...)
end
function prior_forecast_location(pe::EmpiricalPrior{<:Any, <:Any, <:Number},
                                 pr::AbstractPriorResult, X::MatNum, ::Option{<:MatNum},
                                 pnl::Option{<:AssetPanel}; kwargs...)
    c = forecast_location(library_covariance_estimator(pe.ce), log1p.(X), pnl; dims = 1,
                          kwargs...)
    return horizon_location(c, pr, pe.horizon)
end
function prior_forecast_location(pe::FactorPrior, pr::AbstractPriorResult, ::MatNum,
                                 F::MatNum, ::Option{<:AssetPanel}; kwargs...)
    # The factor prior is fitted on `F` alone, so its location is read over `F` alone.
    fpr = pr.fpr
    c = prior_forecast_location(pe.pe, fpr, F, nothing, nothing)
    return c === fpr.mu ? pr.mu : pr.mu + pr.rr.M * (c - fpr.mu)
end
function prior_forecast_location(pe::Union{<:HighOrderPriorEstimator,
                                           <:HighOrderFactorPriorEstimator},
                                 pr::HighOrderPrior, X::MatNum, F::Option{<:MatNum},
                                 pnl::Option{<:AssetPanel}; kwargs...)
    return prior_forecast_location(pe.pe, pr.pr, X, F, pnl; kwargs...)
end
function prior_forecast_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult;
                                 kwargs...)
    cache = hasfield(typeof(pe), :cache) ? getfield(pe, :cache) : nothing
    return prior_state_location(pe, pr, cache; kwargs...)
end
function prior_forecast_location(pe::HighOrderPriorEstimator, pr::HighOrderPrior; kwargs...)
    # The host owns no buffer, and `prior(pe)` reads the prior that it wraps.
    return prior_forecast_location(pe.pe, pr.pr; kwargs...)
end
"""
    prior_state_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult, state; kwargs...)
    prior_state_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult, state::SampleBufferState; kwargs...)
    prior_state_location(pe::EmpiricalPrior, pr::AbstractPriorResult, state::PriorCarryState; kwargs...)

Read the location of a prior out of the state that its estimator carries, by the type of the state.

This is the data-less arm of [`prior_forecast_location`](@ref), and it takes the routes that the data-less [`prior`](@ref) takes:

  - A [`SampleBufferState`](@ref) is a refit, so the method answers the data form over the rows, the factor rows and the masks of the buffer.
  - The [`PriorCarryState`](@ref) of an [`EmpiricalPrior`](@ref) answers the location of `ce`. A `ce` that folds answers [`forecast_location`](@ref) of its state. A `ce` that does not fold answers the data form over the carried rows, as [`read_member`](@ref) refits it. The horizon arm reads the log-returns of the rows, and maps the location as the data form does.
  - Every other state, and `nothing`, answers `pr.mu`.

# Arguments

  - $(arg_dict[:pe])
  - $(arg_dict[:pr])
  - `state`: The state that `pe` carries, or `nothing`.
  - `kwargs...`: Additional keyword arguments, passed to the data form or to the refit.

# Returns

  - `c::VecNum`: The location, `assets × 1`.

# Related

  - [`prior_forecast_location`](@ref)
  - [`prior`](@ref)
  - [`read_member`](@ref)
"""
function prior_state_location(::AbstractPriorEstimator, pr::AbstractPriorResult, ::Any;
                              kwargs...)
    return pr.mu
end
function prior_state_location(pe::AbstractPriorEstimator, pr::AbstractPriorResult,
                              state::SampleBufferState; kwargs...)
    return prior_forecast_location(pe, pr, sample_buffer(state), factor_buffer(state),
                                   nothing; sample_buffer_kwargs(state)..., kwargs...)
end
function prior_state_location(pe::EmpiricalPrior{<:Any, <:Any, Nothing},
                              ::AbstractPriorResult, state::PriorCarryState; kwargs...)
    return carried_location(pe.ce, sample_buffer(state); kwargs...)
end
function prior_state_location(pe::EmpiricalPrior{<:Any, <:Any, <:Number},
                              pr::AbstractPriorResult, state::PriorCarryState; kwargs...)
    c = carried_location(pe.ce, log1p.(sample_buffer(state)); kwargs...)
    return horizon_location(c, pr, pe.horizon)
end
"""
    carried_location(ce, X::MatNum; kwargs...)

Read the location of a covariance estimator that a fold-and-carry prior holds, out of its state or over the rows that the prior carries.

This is [`read_member`](@ref) for [`forecast_location`](@ref). A `StatsBase.CovarianceEstimator` does not fold, and it is read as a [`GeneralCovariance`](@ref) over the rows, as `cov` reads it.

# Arguments

  - $(arg_dict[:ce])
  - `X`: The rows that the prior carries, `observations × assets`.
  - `kwargs...`: Additional keyword arguments, passed to the refit alone.

# Returns

  - `c::VecNum`: The location, `assets × 1`.

# Related

  - [`read_member`](@ref)
  - [`prior_state_location`](@ref)
"""
function carried_location(ce, X::MatNum; kwargs...)
    return if supports_partial_fit(ce)
        forecast_location(ce)
    else
        forecast_location(library_covariance_estimator(ce), X; dims = 1, kwargs...)
    end
end
"""
    horizon_location(c::VecNum, pr::AbstractPriorResult, horizon::Number)

Map the log location of a covariance estimator to the arithmetic location at an investment horizon, as [`horizon_moments!`](@ref) maps the log mean.

# Mathematical definition

```math
\\begin{align}
h \\sigma_{\\log, ii} &= \\log\\!\\left(1 + \\frac{\\sigma_{ii}}{(\\mu_i + 1)^2}\\right)\\,, \\\\
c_i &= \\exp\\!\\left(h c_{\\log, i} + \\tfrac{1}{2} h \\sigma_{\\log, ii}\\right) - 1\\,.
\\end{align}
```

Where:

  - ``h``: Investment horizon.
  - ``\\mu_i``, ``\\sigma_{ii}``: The arithmetic mean and variance that the prior publishes.
  - ``\\sigma_{\\log, ii}``: The log-return variance that the prior fitted. The first line inverts the second closed form of [`horizon_moments!`](@ref) on the diagonal.
  - ``c_{\\log, i}``: The log location of the covariance estimator.

When `c` is the log mean of the prior, ``c_i = \\mu_i``.

# Arguments

  - `c`: The log location, `assets × 1`.
  - $(arg_dict[:pr])
  - `horizon`: The investment horizon.

# Returns

  - `c::VecNum`: The arithmetic location, `assets × 1`.

# Related

  - [`horizon_moments!`](@ref)
  - [`prior_forecast_location`](@ref)
"""
function horizon_location(c::VecNum, pr::AbstractPriorResult, horizon::Number)
    hs = log1p.(LinearAlgebra.diag(pr.sigma) ./ abs2.(pr.mu .+ one(eltype(pr.mu))))
    return expm1.(horizon * c + hs / 2)
end
"""
    forecast_state_location(ce, state)

Read the location out of the state that a covariance estimator carries, by the type of the state.

This is the data-less arm of [`forecast_location`](@ref), with one method per shape of state:

  - A [`CovarianceState`](@ref) carries `mu`.
  - The two exponentially weighted states carry `location`.
  - A [`SampleBufferState`](@ref) carries the rows and the masks that it folded. The method answers the data form of the estimator over them, because a buffer means the batch verb over its rows at every read-out.
  - A [`PortfolioOptimisersCovariance`](@ref) whose own `cache` is `nothing` forwards to the estimator that it holds, which carries the state.

The first two methods return the vector of the state itself, and the next fold writes that vector in place. A caller that keeps the location across a fold must copy it, as [`covariance_forecast_evaluation`](@ref) does when it stores the forecasts.

# Validation

  - `state` is not `nothing`. An `ArgumentError` is thrown otherwise, because there is nothing to read.
  - `state` has one of the shapes above. An `ArgumentError` is thrown otherwise.

# Related

  - [`forecast_location`](@ref)
  - [`partial_fit_cache`](@ref)
"""
function forecast_state_location(::Any, state::CovarianceState)
    return state.mu
end
function forecast_state_location(::Any,
                                 state::Union{<:ExpWeightedCovarianceState,
                                              <:RegimeAdjustedCovarianceState})
    return state.location
end
function forecast_state_location(ce, state::SampleBufferState)
    return forecast_location(ce, sample_buffer(state); sample_buffer_kwargs(state)...)
end
function forecast_state_location(ce::PortfolioOptimisersCovariance, ::Nothing)
    return forecast_location(ce.ce)
end
function forecast_state_location(ce, ::Nothing)
    return throw(ArgumentError("`$(typeof(ce).name.name)` holds no partial-fit state, so there is no location to read. Call `partial_fit!(ce, X)` first, or `forecast_location(ce, X)` for a fit over a whole sample."))
end
function forecast_state_location(ce, state::AbstractPartialFitState)
    return throw(ArgumentError("`$(typeof(ce).name.name)` carries a `$(typeof(state).name.name)`, and `forecast_location` has no method that reads a location out of it. Add `forecast_location(ce::$(typeof(ce).name.name))`, or wrap the estimator in `Online` so its state is a sample buffer whose column mean is the location."))
end
"""
    forecast_coverage_policy(ce)

Read the [`CoveragePolicy`](@ref) that decides the Asset Panel arm of [`forecast_location`](@ref), out of an estimator that may have no such field.

[`coverage_policy`](@ref) answers for the two families that can carry a policy. Every other covariance estimator answers `nothing`, so the panel form reduces the window to the Coverage Universe, as the moment seam does for those estimators. A [`PortfolioOptimisersCovariance`](@ref) does not reach this verb, because its panel form forwards to the estimator that it holds.

# Related

  - [`forecast_location`](@ref)
  - [`coverage_policy`](@ref)
"""
function forecast_coverage_policy(ce::Union{<:GeneralCovariance, <:Covariance})
    return coverage_policy(ce)
end
function forecast_coverage_policy(::Any)
    return nothing
end
"""
    finite_column_mean(X::MatNum)

Mean of each column of `X` over its finite entries, and `NaN` for a column that has none.

# Related

  - [`forecast_location`](@ref)
"""
function finite_column_mean(X::MatNum)
    fin = isfinite.(X)
    Xf = ifelse.(fin, X, zero(eltype(X)))
    return vec(sum(Xf; dims = 1)) ./ vec(count(fin; dims = 1))
end
"""
    covariance_forecast_step(sigma::MatNum, Z::MatNum, c::VecNum, w::Option{<:VecNum_VecVecNum}, target::AbstractRealisedTarget)

Score one covariance forecast against the returns realised after it.

This is the per-step kernel of [`covariance_forecast_evaluation`](@ref). It takes bare arrays, so a caller who holds forecasts of their own can build the same Result by hand. The kernel reads the forecast on the assets that are active at the step, and centres the test rows on the location of the forecast. It then forms the realised matrix through the target and returns the diagnostics of the step. Four diagnostics read the forecast without a portfolio, and two more come once per test portfolio.

# Mathematical definition

```math
\\begin{align}
\\mathbf{H}_{ij} &= \\left\\lvert \\{ s : z_{t+s,i},\\, z_{t+s,j} \\text{ finite} \\} \\right\\rvert\\,, \\\\
m_t &= \\frac{\\operatorname{tr}\\left(\\left(\\mathbf{H} \\odot \\hat{\\mathbf{\\Sigma}}_t\\right)^{-1} \\mathbf{S}_t\\right)}{N}\\,, \\\\
d_{t,i} &= \\frac{(\\mathbf{S}_t)_{ii}}{\\mathbf{H}_{ii}\\, (\\hat{\\mathbf{\\Sigma}}_t)_{ii}}\\,, \\\\
L^{\\mathrm{QLIKE}}_t &= h \\log\\left|\\hat{\\mathbf{\\Sigma}}_t\\right| + N h\\, m_t\\,, \\\\
L^{\\mathrm{F}}_t &= \\sum_{i,j : \\mathbf{H}_{ij} > 0} \\left(\\frac{(\\mathbf{S}_t)_{ij}}{\\mathbf{H}_{ij}} - (\\hat{\\mathbf{\\Sigma}}_t)_{ij}\\right)^2\\,, \\\\
b_t &= \\frac{\\boldsymbol{w}^\\intercal \\boldsymbol{R}_t}{\\sqrt{\\boldsymbol{w}^\\intercal \\left(\\mathbf{H} \\odot \\hat{\\mathbf{\\Sigma}}_t\\right) \\boldsymbol{w}}}\\,, \\\\
L^{\\mathrm{P}}_t &= \\log\\left(\\boldsymbol{w}^\\intercal \\left(\\mathbf{H} \\odot \\hat{\\mathbf{\\Sigma}}_t\\right) \\boldsymbol{w}\\right) + \\frac{\\sum_{s=1}^{h} (\\boldsymbol{w}^\\intercal \\boldsymbol{z}_{t+s})^2}{\\boldsymbol{w}^\\intercal \\left(\\mathbf{H} \\odot \\hat{\\mathbf{\\Sigma}}_t\\right) \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{H}``: Pairwise count of the finite test observations of each pair of assets.
  - $(math_dict[:Sigma_hat_t])
  - $(math_dict[:S_t_realised])
  - $(math_dict[:h_step])
  - $(math_dict[:N])
  - ``m_t``: Mahalanobis ratio of step ``t``.
  - ``d_{t,i}``: Diagonal ratio of asset ``i`` at step ``t``.
  - ``L^{\\mathrm{QLIKE}}_t``: QLIKE loss of step ``t``.
  - ``L^{\\mathrm{F}}_t``: Frobenius loss of step ``t``.
  - ``b_t``: Standardised return of the test portfolio at step ``t``.
  - ``L^{\\mathrm{P}}_t``: QLIKE loss of the test portfolio's variance at step ``t``.
  - ``\\boldsymbol{R}_t``: Return earned over the horizon of step ``t``, summed over the finite observations of each asset.
  - ``\\boldsymbol{z}_{t+s}``: Centred return of observation ``t + s``, zero at a cell the asset did not trade.
  - $(math_dict[:w_port])

With no gap in the test window ``\\mathbf{H} = h \\mathbf{1}\\mathbf{1}^\\intercal``, and the forms reduce to ``m_t = \\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{S}_t) / (N h)``, ``d_{t,i} = (\\mathbf{S}_t)_{ii} / (h (\\hat{\\mathbf{\\Sigma}}_t)_{ii})``, ``L^{\\mathrm{QLIKE}}_t = h \\log|\\hat{\\mathbf{\\Sigma}}_t| + \\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{S}_t)``, ``L^{\\mathrm{F}}_t = \\lVert \\mathbf{S}_t / h - \\hat{\\mathbf{\\Sigma}}_t \\rVert_F^2`` and ``b_t = \\boldsymbol{w}^\\intercal \\boldsymbol{R}_t / \\sqrt{h\\, \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}}_t \\boldsymbol{w}}``.

When the forecast is the true covariance, ``\\mathbb{E}[m_t] = \\mathbb{E}[d_{t,i}] = 1``, and ``b_t`` has mean zero and variance one, for every distribution of the returns. These moments hold under a gap too, because the count scales each cell of the forecast. With no gap and under Gaussian returns, ``m_t`` times its degrees of freedom is chi-squared on them ([`target_dof`](@ref)). A ratio above one means that the forecast under-predicts the dispersion, in the metric of the forecast, and a ratio below one means that it over-predicts it.

Under [`RealisedCovariance`](@ref), ``L^{\\mathrm{QLIKE}}_t`` is ``-2`` times the Gaussian log-likelihood of the returns of the step under the forecast, up to a constant. Under [`HorizonReturn`](@ref) at ``h > 1`` it is not a log-likelihood, but it has the same expectation. ``L^{\\mathrm{F}}_t`` is the squared distance of the forecast from the per-cell proxy. With no gap, the true covariance minimises the expected value of each loss under any conditionally unbiased proxy. So either loss ranks two forecasts on the proxy as it would rank them on the truth. Under a gap this holds for ``L^{\\mathrm{F}}_t`` only, and the expected ``L^{\\mathrm{QLIKE}}_t`` can be lower at a forecast that is not the truth. The level of neither loss is a calibration reading on its own.

``L^{\\mathrm{P}}_t`` is the univariate QLIKE of the variance of the portfolio. It reads the per-row portfolio returns, whichever target formed ``\\mathbf{S}_t``.

# Algorithm

 1. Find the active subset `a`, the assets whose forecast variance and location are finite and which have at least one finite test return. Refuse an empty subset by name.
 2. Take the forecast on the active subset, giving `sa = sigma[a, a]`.
 3. Centre the test rows of `a` on `c[a]`, giving `Zc`. A non-finite cell stays non-finite.
 4. Form `(S, H)` through [`realised_target`](@ref), and the effective forecast `seff = H .* sa`.
 5. Factor `sa` by Cholesky, giving `ch`.
 6. When every entry of `H` is equal, compute `trace`, the trace of `seff⁻¹ S`, through `ch`. Otherwise factor `seff` by Cholesky and compute `trace` through that factor.
 7. Read the Mahalanobis ratio `m` and the QLIKE loss `qlike` from `ch` and `trace`, the Frobenius loss `frob` over the cells with a positive count, and the diagonal ratio `d` of each active asset. `d` is `NaN` outside `a`.
 8. Replace each non-finite cell of `Zc` with zero, giving `Zf`, and sum each column of `Zf`, giving `R`.
 9. Resolve the test portfolios over `a` through [`resolve_forecast_weights`](@ref), giving `ws`.
10. For each portfolio, read the standardised return `b` from `R`, and the portfolio QLIKE `pq` from the per-row portfolio returns `Zf * wk`.

# Arguments

  - `sigma`: The forecast, `assets × assets`. A `NaN` diagonal marks an inactive asset.
  - `Z`: The test rows, `observations × assets`. A non-finite cell is an observation that the asset did not trade.
  - `c`: The location the forecast is about, `assets × 1`.
  - `w`: The test portfolios on the full universe: `nothing` for inverse volatility over the active subset, a vector for one portfolio, a vector of vectors for several. Each is renormalised over the active subset.
  - `target`: The realised quantity.

# Validation

  - `sigma` is square and `Z`, `c` and every portfolio of `w` have its width. A `DimensionMismatch` is thrown otherwise.
  - At least one asset is active: a finite forecast variance, a finite location, and a finite test return. An `ArgumentError` is thrown otherwise.
  - The forecast on the active subset is positive definite. A `LinearAlgebra.PosDefException` is thrown otherwise.

# Returns

  - `step::NamedTuple`: The diagnostics of the step. `n_valid` is the count of active assets. `mahalanobis_ratio`, `qlike` and `frobenius` are scalars, `diagonal_ratio` is `assets × 1`, and `standardised_return` and `portfolio_qlike` hold one entry per portfolio.

# Related

  - [`covariance_forecast_evaluation`](@ref)
  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`realised_target`](@ref)
  - [`resolve_forecast_weights`](@ref)
  - [`forecast_location`](@ref)
"""
function covariance_forecast_step(sigma::MatNum, Z::MatNum, c::VecNum,
                                  w::Option{<:VecNum_VecVecNum},
                                  target::AbstractRealisedTarget)
    N = size(sigma, 1)
    @argcheck(size(sigma, 2) == N && size(Z, 2) == N && length(c) == N,
              DimensionMismatch("the forecast is $(size(sigma, 1)) × $(size(sigma, 2)), the test rows have $(size(Z, 2)) columns and the location $(length(c)) entries; all three must share one asset axis."))
    a = findall(isfinite.(LinearAlgebra.diag(sigma)) .& isfinite.(c) .&
                vec(any(isfinite, Z; dims = 1)))
    @argcheck(!isempty(a),
              ArgumentError("no asset is active at this step: every asset has a non-finite forecast variance or location, or no finite test return, so there is nothing to score. Drop the step, or hand the kernel a forecast and test rows that share at least one active asset."))
    h = size(Z, 1)
    na = length(a)
    sa = sigma[a, a]
    Zc = Z[:, a] .- transpose(c[a])
    S, H = realised_target(target, Zc)
    seff = H .* sa
    ch = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(sa))
    h11 = H[1, 1]
    trace = if all(==(h11), H)
        LinearAlgebra.tr(ch \ S) / h11
    else
        LinearAlgebra.tr(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(seff)) \ S)
    end
    m = trace / na
    qlike = h * LinearAlgebra.logdet(ch) + h * trace
    frob = zero(m)
    for j in axes(S, 2), i in axes(S, 1)
        if H[i, j] > 0
            frob += abs2(S[i, j] / H[i, j] - sa[i, j])
        end
    end
    d = fill(convert(typeof(m), NaN), N)
    d[a] .= LinearAlgebra.diag(S) ./ (LinearAlgebra.diag(H) .* LinearAlgebra.diag(sa))
    fin = isfinite.(Zc)
    Zf = ifelse.(fin, Zc, zero(eltype(Zc)))
    R = vec(sum(Zf; dims = 1))
    ws = resolve_forecast_weights(w, a, sa, N)
    b = Vector{typeof(m)}(undef, length(ws))
    pq = Vector{typeof(m)}(undef, length(ws))
    for (k, wk) in enumerate(ws)
        v = LinearAlgebra.dot(wk, seff, wk)
        b[k] = LinearAlgebra.dot(wk, R) / sqrt(v)
        pq[k] = log(v) + sum(abs2, Zf * wk) / v
    end
    return (; n_valid = na, mahalanobis_ratio = m, diagonal_ratio = d, qlike = qlike,
            frobenius = frob, standardised_return = b, portfolio_qlike = pq)
end
"""
    resolve_forecast_weights(w::Nothing, a::VecInt, sa::MatNum, N::Integer)
    resolve_forecast_weights(w::VecNum, a::VecInt, sa::MatNum, N::Integer)
    resolve_forecast_weights(w::VecVecNum, a::VecInt, sa::MatNum, N::Integer)

Resolve the test portfolios of a step over its active subset.

`nothing` means inverse volatility, recomputed from the forecast of the step over the active assets, ``w_i \\propto 1 / \\sqrt{(\\hat{\\mathbf{\\Sigma}}_t)_{ii}}``. It keeps the most volatile asset from dominating the reading. A vector is one static portfolio, and a vector of vectors is several. The method cuts each portfolio to the active subset and rescales it to sum to one there. So the weight of a delisted asset moves to the remaining assets, in proportion to their weights. The method does not check a subset whose weights sum to zero. Such a portfolio has no scale, and its diagnostics are not finite.

# Arguments

  - `w`: The test portfolios on the full universe, or `nothing`.
  - `a`: The active subset.
  - `sa`: The forecast on the active subset.
  - `N`: The width of the universe.

# Validation

  - Every portfolio has `N` entries. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `ws::Vector{<:VecNum}`: One weights vector per portfolio, each over `a`.

# Related

  - [`covariance_forecast_step`](@ref)
"""
function resolve_forecast_weights(::Nothing, ::VecInt, sa::MatNum, ::Integer)
    iv = inv.(sqrt.(LinearAlgebra.diag(sa)))
    return [iv ./ sum(iv)]
end
function resolve_forecast_weights(w::VecNum, a::VecInt, ::MatNum, N::Integer)
    @argcheck(length(w) == N,
              DimensionMismatch("the test portfolio has $(length(w)) weights and the universe $(N) assets; a portfolio is given on the full universe and cut to the active subset per step."))
    wa = w[a]
    return [wa ./ sum(wa)]
end
function resolve_forecast_weights(w::VecVecNum, a::VecInt, sa::MatNum, N::Integer)
    return [only(resolve_forecast_weights(wk, a, sa, N)) for wk in w]
end
"""
    partial_fit!(ce::AbstractCovarianceEstimator, rd::ReturnsResult)
    partial_fit!(pe::AbstractPriorEstimator, rd::ReturnsResult)

Fold the observations of a carrier into a covariance estimator, or into a prior.

This is the carrier arity that [`online_folds`](@ref), the online arm of the fold loop, calls for the two kinds of estimator that [`covariance_forecast_evaluation`](@ref) threads through it. A prior folds through [`fold_prior`](@ref), the same forward that the step of an optimiser makes. A covariance estimator folds `rd.X` in its own arity. When the Asset Panel varies in time, the method reads its active mask through [`step_active_mask`](@ref) and passes it as the `active_mask` keyword. That verb also refuses a mask that the step cannot carry.

# Algorithm

 1. For a prior, return [`fold_prior`](@ref)`(pe, rd)`. The steps below are for a covariance estimator.
 2. Check that `rd.X` is not `nothing`.
 3. Read the active mask of the step through [`step_active_mask`](@ref), giving `amsk`.
 4. Fold `rd.X` into `ce` through `partial_fit!(ce, rd.X)`, with `active_mask = amsk` when `amsk` is not `nothing`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:pe])
  - `rd`: The carrier of the observations to fold, `observations × assets`.

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - Everything [`step_active_mask`](@ref) refuses.

# Returns

  - `est`: The estimator, with the observations folded into its state.

# Related

  - [`fold_prior`](@ref)
  - [`step_active_mask`](@ref)
  - [`online_folds`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
"""
function partial_fit!(ce::AbstractCovarianceEstimator, rd::ReturnsResult)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    amsk = step_active_mask(rd)
    return if isnothing(amsk)
        partial_fit!(ce, rd.X)
    else
        partial_fit!(ce, rd.X; active_mask = amsk)
    end
end
function partial_fit!(pe::AbstractPriorEstimator, rd::ReturnsResult)
    return fold_prior(pe, rd)
end
"""
    is_time_dependent(est::Union{<:AbstractCovarianceEstimator, <:AbstractPriorEstimator, <:Online})
    needs_previous_weights(est::Union{<:AbstractCovarianceEstimator, <:AbstractPriorEstimator, <:Online})

Answer the two per-type predicates that [`fold_loop`](@ref) reads, for an estimator that is not an optimiser.

No covariance estimator and no prior holds a [`TimeDependent`](@ref) schedule, and neither reads the weights of a previous fold. The schedule type is declared after both families, so no field of theirs can name it. An [`Online`](@ref) wrapper at the root answers the same, because the loop resolves it before the first fold. Both predicates answer `false`. The loop then neither resolves a context nor threads weights, and a batch run of the evaluation takes the parallel arm under the sequential executor that it receives.

# Related

  - [`fold_loop`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
"""
function is_time_dependent(::Union{<:AbstractCovarianceEstimator, <:AbstractPriorEstimator,
                                   <:Online})
    return false
end
function needs_previous_weights(::Union{<:AbstractCovarianceEstimator,
                                        <:AbstractPriorEstimator, <:Online})
    return false
end
"""
    online_entry_state(o::Online)

Name the state that a wrapped estimator carries at the entry of the online arm of the fold loop, or answer `nothing`.

This is the [`Online`](@ref) method of [`online_entry_state`](@ref), for a wrapper at the root of [`covariance_forecast_evaluation`](@ref). A wrapper holds no `cache` of its own, so the walk enters the estimator that it wraps and puts `est.` before the path that it finds there. For a covariance estimator that the caller folded before the call, the path is `est.cache`. The generic walk through [`estimator_fields`](@ref) would give the same answer. This method gives it without a request for a field that the wrapper does not have.

# Related

  - [`online_entry_state`](@ref)
  - [`assert_online_entry`](@ref)
  - [`Online`](@ref)
"""
function online_entry_state(o::Online)
    path = online_entry_state(o.est)
    return isnothing(path) ? nothing : string("est.", path)
end
"""
    advance_previous_fold(pws, prev, step::NamedTuple)

Hand the next fold what the last fold held, when the result of the fold is a step record and not a prediction.

The step record of the kernel carries no weights, so the method threads nothing and returns `prev` unchanged. [`online_folds`](@ref) reaches this method under [`covariance_forecast_evaluation`](@ref).

# Related

  - [`threads_weights`](@ref)
  - [`online_folds`](@ref)
"""
function advance_previous_fold(::Any, prev, ::NamedTuple)
    return prev
end
"""
    forecast_moments(ce::AbstractCovarianceEstimator, rd::ReturnsResult, train_idx::VecInt)
    forecast_moments(ce::AbstractCovarianceEstimator, rd::ReturnsResult, ::Nothing)
    forecast_moments(pe::AbstractPriorEstimator, rd::ReturnsResult, train_idx::VecInt)
    forecast_moments(pe::AbstractPriorEstimator, rd::ReturnsResult, ::Nothing)

Read the forecast of a fold and the location that it is about, by whether the fold carries a training window.

These are the two arms of the callback of [`covariance_forecast_evaluation`](@ref). Dispatch on the `train` of the fold chooses the arm, as it does for the optimiser in [`fit_fold_result`](@ref). A training window means a refit, which every batch fold runs. `nothing` means that the estimator holds its window, so the fold reads the moments out of the state. The method reads a prior once for both moments.

# Algorithm

 1. With a training window, take the view of the carrier on its rows, giving `rdt`. Fit a covariance estimator through the Asset Panel seam of the moment verbs, `cov(ce, rdt.X, rdt.pnl)` and [`forecast_location`](@ref)`(ce, rdt.X, rdt.pnl)`. Fit a prior once over the view, giving `pr`.
 2. With `nothing`, read `cov(ce)` and `forecast_location(ce)` out of the state of a covariance estimator, or `pr = prior(pe)` out of the state of a prior.
 3. Return the forecast and the location. For a prior they are `pr.sigma` and [`prior_forecast_location`](@ref) of `pr`, which reads the centre of `pr.sigma` without a second fit.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:pe])
  - `rd`: The carrier.
  - `train_idx`: The fold's training window, or `nothing` for a stepped estimator.

# Returns

  - `sigma::MatNum`: The forecast, `assets × assets`.
  - `c::VecNum`: The location the forecast is about, `assets × 1`.

# Related

  - [`covariance_forecast_evaluation`](@ref)
  - [`forecast_location`](@ref)
  - [`fit_fold_result`](@ref)
  - [`Fold`](@ref)
"""
function forecast_moments(ce::AbstractCovarianceEstimator, rd::ReturnsResult,
                          train_idx::VecInt)
    rdt = port_opt_view(rd, train_idx, :)
    return Statistics.cov(ce, rdt.X, rdt.pnl; dims = 1),
           forecast_location(ce, rdt.X, rdt.pnl; dims = 1)
end
function forecast_moments(ce::AbstractCovarianceEstimator, ::ReturnsResult, ::Nothing)
    return Statistics.cov(ce), forecast_location(ce)
end
function forecast_moments(pe::AbstractPriorEstimator, rd::ReturnsResult, train_idx::VecInt)
    rdt = port_opt_view(rd, train_idx, :)
    pr = prior(pe, rdt)
    return pr.sigma, prior_forecast_location(pe, pr, rdt)
end
function forecast_moments(pe::AbstractPriorEstimator, ::ReturnsResult, ::Nothing)
    pr = prior(pe)
    return pr.sigma, prior_forecast_location(pe, pr)
end
"""
$(DocStringExtensions.TYPEDEF)

The per-step diagnostics of a covariance forecast over a walk-forward, and its forecasts on request.

`CovarianceForecastEvaluationResult` is what [`covariance_forecast_evaluation`](@ref) returns. It keeps every diagnostic per step, four that read the forecast on its own and two per test portfolio. The summary, the comparison and the plots are verbs over it. It keeps the forecasts themselves only on request. A run of `M` steps over `N` assets holds `N² × M` numbers of forecasts against `N × M` diagonal ratios, and the online form is meant for a step at every observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceForecastEvaluationResult(
        dates, test_idx, horizon, target, n_valid, mahalanobis_ratio, diagonal_ratio,
        qlike, frobenius, standardised_return, portfolio_qlike, w, sigma, location
    ) -> CovarianceForecastEvaluationResult

The arguments correspond to the fields of the struct, in the order of their declaration. The type is a Result, which [`covariance_forecast_evaluation`](@ref) builds and a caller reads. It has no keyword constructor, and it validates nothing of its own.

# Related

  - [`covariance_forecast_evaluation`](@ref)
  - [`covariance_forecast_step`](@ref)
  - [`covariance_forecast_summary`](@ref)
  - [`covariance_forecast_compare`](@ref)
  - [`covariance_forecast_portfolio`](@ref)
"""
@concrete struct CovarianceForecastEvaluationResult <: AbstractResult
    """
    Label of each step, one entry per step. It is the timestamp of the first test row of the step when the carrier holds timestamps, and the index of that row otherwise.
    """
    dates
    """
    Test rows of each step, one window per step, in the order of the split.
    """
    test_idx
    """
    Horizon of each step, the number of test rows that the evaluation judged the forecast on, one entry per step. Every entry is the `test_size` of the scheme under an index walk-forward, and the length of the calendar period under a date walk-forward.
    """
    horizon
    """
    Realised quantity that the evaluation judged the forecasts against.
    """
    target
    """
    Number of active assets at each step, one entry per step. An active asset has a finite forecast variance, a finite location and at least one finite test return.
    """
    n_valid
    """
    Mahalanobis ratio of each step, one entry per step. Its target value is one.
    """
    mahalanobis_ratio
    """
    Diagonal ratio of each asset at each step, `steps × assets`, and `NaN` where the asset was not active. Its target value is one.
    """
    diagonal_ratio
    """
    QLIKE loss of each step, one entry per step. Lower is better, and only the difference between two forecasts is a reading.
    """
    qlike
    """
    Frobenius loss of each step, one entry per step. Lower is better, and only the difference between two forecasts is a reading.
    """
    frobenius
    """
    Standardised return of each test portfolio at each step, `steps × portfolios`. Mean zero and standard deviation one under a calibrated forecast.
    """
    standardised_return
    """
    QLIKE loss of the variance of each test portfolio at each step, `steps × portfolios`. Lower is better.
    """
    portfolio_qlike
    """
    Test portfolios as the caller gave them to the evaluation. `nothing` means inverse volatility per step, a vector is one portfolio, and a vector of vectors is several.
    """
    w
    """
    Forecast of each step, one `assets × assets` matrix per step, or `nothing` unless `store_forecasts = true`.
    """
    sigma
    """
    Location that the evaluation centred each forecast on, one `assets × 1` vector per step, or `nothing` unless `store_forecasts = true`.
    """
    location
end
"""
    covariance_forecast_evaluation(est::Union{<:AbstractCovarianceEstimator, <:AbstractPriorEstimator, <:Online},
                                   rd::ReturnsResult, cv::WFCVER; w::Option{<:VecNum_VecVecNum} = nothing,
                                   target::AbstractRealisedTarget = RealisedCovariance(),
                                   store_forecasts::Bool = false) -> CovarianceForecastEvaluationResult

Evaluate a covariance forecast out of sample over a walk-forward, in batch or online.

The verb runs through the one fold loop of the library. At each fold the callback reads the forecast and its location off the estimator of the fold, and scores them on the test rows through [`covariance_forecast_step`](@ref). That estimator is a refit over the training window when the fold carries one, and the threaded state under an Online Scheme. The four forms of the evaluation are compositions that the walk-forward and the [`Online`](@ref) wrapper already express:

  - `expand_train = true` gives the batch expanding form.
  - `expand_train = false` gives the batch rolling form.
  - `OnlineIndexWalkForward(…)` gives the online expanding form.
  - `OnlineIndexWalkForward(…)` with `Online(est; max_history = w)` gives the online rolling form.

The verb reads none of these settings. So the online run gives the rows of the batch expanding run fold for fold, to the tolerance of the estimator's own fold. The date form, the purge, and a listing or a delisting need no code of their own.

The loop receives the estimator as a configuration only. Under an Online Scheme it warms up one estimator on the first training window and folds the new rows of each fold into it. So the loop refuses, by name, an estimator that enters with a state or that cannot fold. An estimator that cannot fold but has a `cache` field can be wrapped in [`Online`](@ref), which folds it from a buffer. Each read-out is then a batch fit over the rows of the buffer. An estimator with no `cache` field has nowhere to carry a buffer, and `Online` refuses it.

# Algorithm

 1. Check `rd.X`, and check that an [`Online`](@ref) at the root comes with an Online Scheme.
 2. Split the carrier by `cv`, giving `train_idx` and `test_idx`, and check that the folds are not shuffled.
 3. Run [`fold_loop`](@ref) over the `n` folds, giving `steps`. At each fold, read `(sigma, c)` through [`forecast_moments`](@ref) and score the test rows through [`covariance_forecast_step`](@ref). When `store_forecasts = true`, add copies of `sigma` and `c` to the step record, because an online state writes its location in place.
 4. Stack the diagonal ratios, the standardised returns and the portfolio QLIKE losses of the steps into the matrices `d`, `b` and `pq`, one row per step.
 5. Label each step with the timestamp of its first test row, or with the index of that row when `rd.ts` is `nothing`, giving `dates`.
 6. Build the Result from the columns, the horizons `length.(test_idx)`, `target` and `w`.

# Arguments

  - `est`: The covariance estimator, or the prior estimator whose `sigma` is the forecast. Under an Online Scheme either can be wrapped in [`Online`](@ref), to fold from a buffer and, with `max_history`, over a rolling window.
  - $(arg_dict[:rd])
  - `cv`: The walk-forward, in its index or date form, or its split.
  - `w`: The test portfolios on the full universe. `nothing` means inverse volatility recomputed from the forecast of each step, a vector is one static portfolio, and a vector of vectors is several.
  - `target`: The realised quantity that the evaluation judges the forecasts against.
  - `store_forecasts`: Whether to keep the forecast and the location of every step in the Result, for [`covariance_forecast_portfolio`](@ref).

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - The folds of the scheme are not shuffled, through [`assert_unshuffled_folds`](@ref). An `ArgumentError` is thrown otherwise.
  - An [`Online`](@ref) at the root comes with an Online Scheme. An `ArgumentError` is thrown otherwise, because a batch fold would never seed the buffer.
  - Everything that [`covariance_forecast_step`](@ref) refuses at a step, and everything that the online arm of the fold loop refuses at entry under an Online Scheme.

# Returns

  - `cfer::CovarianceForecastEvaluationResult`: The per-step diagnostics, and the forecasts when asked.

# Related

  - [`CovarianceForecastEvaluationResult`](@ref)
  - [`covariance_forecast_step`](@ref)
  - [`covariance_forecast_summary`](@ref)
  - [`covariance_forecast_compare`](@ref)
  - [`covariance_forecast_portfolio`](@ref)
  - [`forecast_location`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
  - [`OnlineIndexWalkForward`](@ref)
  - [`OnlineDateWalkForward`](@ref)
  - [`Online`](@ref)
"""
function covariance_forecast_evaluation(est::Union{<:AbstractCovarianceEstimator,
                                                   <:AbstractPriorEstimator, <:Online},
                                        rd::ReturnsResult, cv::WFCVER;
                                        w::Option{<:VecNum_VecVecNum} = nothing,
                                        target::AbstractRealisedTarget = RealisedCovariance(),
                                        store_forecasts::Bool = false)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    @argcheck(!isa(est, Online) || folds_are_stepped(cv),
              ArgumentError("`Online($(typeof(est.est).name.name))` declares a refit from a sample buffer, which only the fold loop's online arm seeds and threads, and this scheme is not an Online Scheme, so every fold would refit the estimator from its training window and the buffer would never exist. Run the evaluation under `OnlineIndexWalkForward` or `OnlineDateWalkForward`, or hand it the estimator unwrapped."))
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    n = length(train_idx)
    steps, _ = fold_loop(est, n, FLoops.SequentialEx(), NamedTuple; rd = rd,
                         train_idx = train_idx, test_idx = test_idx, cv = cv) do fold
        sigma, c = forecast_moments(fold.est, fold.rd, fold.train)
        step = covariance_forecast_step(sigma, view(fold.rd.X, fold.test, :), c, w, target)
        # An online state updates its location in place, so a stored step keeps a copy.
        return store_forecasts ? (; step..., sigma = copy(sigma), location = copy(c)) : step
    end
    X = rd.X
    N = size(X, 2)
    P = length(steps[1].standardised_return)
    d = Matrix{eltype(steps[1].diagonal_ratio)}(undef, n, N)
    b = Matrix{eltype(steps[1].standardised_return)}(undef, n, P)
    pq = Matrix{eltype(steps[1].portfolio_qlike)}(undef, n, P)
    for (i, s) in enumerate(steps)
        d[i, :] .= s.diagonal_ratio
        b[i, :] .= s.standardised_return
        pq[i, :] .= s.portfolio_qlike
    end
    firsts = [first(t) for t in test_idx]
    dates = isnothing(rd.ts) ? firsts : rd.ts[firsts]
    return CovarianceForecastEvaluationResult(dates, test_idx,
                                              [length(t) for t in test_idx], target,
                                              [s.n_valid for s in steps],
                                              [s.mahalanobis_ratio for s in steps], d,
                                              [s.qlike for s in steps],
                                              [s.frobenius for s in steps], b, pq, w,
                                              if store_forecasts
                                                  [s.sigma for s in steps]
                                              else
                                                  nothing
                                              end, if store_forecasts
                                                  [s.location for s in steps]
                                              else
                                                  nothing
                                              end)
end

export RealisedCovariance, HorizonReturn, covariance_forecast_step,
       covariance_forecast_evaluation, CovarianceForecastEvaluationResult
# The `# Interfaces`-marked type and verbs of #1138 (ADR 0154).
public AbstractRealisedTarget, realised_target, target_dof, target_step_dof
