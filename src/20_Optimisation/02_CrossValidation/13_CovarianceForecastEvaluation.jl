"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the realised quantities a covariance forecast is judged against.

A covariance forecast ``\\hat{\\mathbf{\\Sigma}}_t`` is formed at a step and judged on the ``h`` returns that follow it. The member of this family says **which** statistic of those returns stands in for the covariance they were drawn from: their realised covariance, or the outer product of their sum. Both have the same expectation, ``h\\, \\mathbf{\\Sigma}_t`` when the returns are serially uncorrelated, and both coincide at ``h = 1``, so the member decides the variance of the per-step diagnostic and the question it answers, never its target.

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
  - `Zc`: The centred test rows, `observations × assets`; a cell an asset did not trade is non-finite.
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

This is the default target. It sums the outer product of every centred return of the step, so it is a per-row statistic: under a Gaussian null the step's Mahalanobis ratio has variance ``2 / (N h)``, the smaller of the two members' by a factor of ``h``, which is why it is the default when the better statistic costs nothing. It asks whether the forecast describes each day's dispersion.

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

The expectation holds when the ``h`` returns are serially uncorrelated, and it makes ``\\mathbf{S}_t / h`` a conditionally unbiased proxy of ``\\mathbf{\\Sigma}_t`` whatever the distribution of the returns. At ``h = 1`` the proxy has rank one and equals the [`HorizonReturn`](@ref) member's. A cell whose two assets share no finite row is zero, and its count is zero.

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

This is the reference implementation's member, and the ``h``-day holder's question: the return summed over the step is what a book held for the whole horizon earns, and its outer product is a rank-one proxy of ``h\\, \\mathbf{\\Sigma}_t``. Under a Gaussian null the step's Mahalanobis ratio has variance ``2 / N``, whatever ``h``, so it is the noisier member at every horizon above one.

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

The expectation holds when the ``h`` returns are serially uncorrelated. At ``h = 1`` the proxy equals the [`RealisedCovariance`](@ref) member's. A non-finite cell contributes nothing to the sum, and the count of a cell is the number of rows at which both of its assets are finite.

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

The two arms of the [`AbstractRealisedTarget`](@ref) interface the library ships. A non-finite cell of `Zc` is an observation the asset did not trade, so it contributes nothing to the sum and is not counted; the count `H` is the pairwise number of finite rows, and it is what scales the forecast cell by cell, `H ⊙ Σ̂`, so that a gap in the test window moves the target and not the reading.

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

The statistic is the trace of the whitened realised matrix: `n * h` under [`RealisedCovariance`](@ref), because every one of the `h` returns contributes `n` whitened coordinates, and `n` under [`HorizonReturn`](@ref), because the horizon return is one draw whatever `h`. The summary reads it to weight the steps of a run and to width the band on the mean ratio, and the exceedance rate reads it for its chi-squared threshold.

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

`h` under [`RealisedCovariance`](@ref), one squared whitened coordinate per return, and `1` under [`HorizonReturn`](@ref), one draw whatever `h`. The summary reads it to weight the steps of the diagonal ratio and to width its band.

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
    forecast_location(ce::Covariance)
    forecast_location(ce::GeneralCovariance)
    forecast_location(ce::Union{<:ExpWeightedCovariance, <:RegimeAdjustedExpWeightedCovariance})
    forecast_location(ce::PortfolioOptimisersCovariance)
    forecast_location(ce::CorrelationCovariance)
    forecast_location(ce::AbstractCovarianceEstimator)
    forecast_location(pe::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    forecast_location(pe::AbstractPriorEstimator; kwargs...)

Read the location a covariance forecast is about, off the estimator that formed it.

A second moment is always of returns about some centre, ``\\mathbb{E}[(\\boldsymbol{z} - \\boldsymbol{c})(\\boldsymbol{z} - \\boldsymbol{c})^\\intercal] = \\hat{\\mathbf{\\Sigma}}``, and the centre is the estimator's own: the mean its centring estimator fitted, the weighted sample mean it subtracted, the exponentially weighted location its state carries, or zero for an estimator that assumed it. An evaluation that judged the forecast against the raw outer product ``\\boldsymbol{z}\\boldsymbol{z}^\\intercal`` would be judging it against a proxy of ``\\mathbf{\\Sigma} + (\\boldsymbol{\\mu} - \\boldsymbol{c})(\\boldsymbol{\\mu} - \\boldsymbol{c})^\\intercal``, and read a bias of ``\\boldsymbol{c}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{c} / N`` into a forecast that has none. So the evaluation centres the test rows here, and there is no flag: the location is the estimator's, not the caller's.

Two arities per family. The data form answers for a batch fit over `X`, and the data-less form answers for the state the estimator carries after [`partial_fit!`](@ref), so a refit fold and a stepped fold centre on the same quantity.

  - [`Covariance`](@ref): the centre [`weighted_centre`](@ref) resolves from `ce.me` and `ce.w`, or `state.mu`. Under a [`CoveragePolicy`](@ref) the batch fit *is* the fold, so the data form folds the window and reads the same `state.mu`: the diagonal of the per-pair centre, each asset's own available-case mean. The off-diagonal products of the forecast are centred on the mean of the observations the pair shares, which is not a location the vector can carry; the diagonal is the one the per-asset ratios read.
  - [`GeneralCovariance`](@ref): the sample mean, weighted by `ce.w` when it carries observation weights, or `state.mu`.
  - [`ExpWeightedCovariance`](@ref) and [`RegimeAdjustedExpWeightedCovariance`](@ref): `state.location`, or zero for every asset when `centred = true`, because the estimator then assumed it. The data form runs the same pass over the window that the batch fit runs, and reads the location the pass ends on.
  - [`PortfolioOptimisersCovariance`](@ref) and [`CorrelationCovariance`](@ref): the answer of the estimator they hold, because a matrix transform moves no centre.
  - Any other [`AbstractCovarianceEstimator`](@ref): the window's own sample mean over the finite rows of each column. An estimator that carries a state of its own and no method is refused by name, because there is no window to read.
  - Under [`Online`](@ref), whatever the family: the family's data form over the buffer's rows and masks, because a buffer means the batch verb over the buffer's rows for every read-out.
  - An [`AbstractPriorEstimator`](@ref): the `mu` the prior publishes.

The Asset Panel form follows the moment seam of the estimator's own `cov`: under no policy it reduces the window to its Coverage Universe and frames the answer with `NaN` outside it, and under a policy, or for the two mask-aware exponentially weighted families, it hands the estimator the panel's active mask.

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
  - The data-less fallback requires the state to be a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `c::VecNum`: The location, `assets × 1`.

# Related

  - [`covariance_forecast_step`](@ref)
  - [`covariance_forecast_evaluation`](@ref)
  - [`weighted_centre`](@ref)
  - [`partial_fit!`](@ref)
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
    return forecast_location(ce.ce, X; kwargs...)
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
    amsk, _ = panel_moment_masks(pnl)
    return forecast_location(ce, X; dims = dims, active_mask = amsk, kwargs...)
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
function forecast_location(pe::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    return prior(pe, rd; kwargs...).mu
end
function forecast_location(pe::AbstractPriorEstimator; kwargs...)
    return prior(pe; kwargs...).mu
end
"""
    forecast_state_location(ce, state)

Read the location out of the state a covariance estimator carries, by the state's type.

The data-less arm of [`forecast_location`](@ref), one method per state shape: a [`CovarianceState`](@ref) carries `mu`, the two exponentially weighted states carry `location`, a [`SampleBufferState`](@ref) carries the rows and the masks they were folded with, and answers the estimator's own data form over them, because a buffer means the batch verb over the buffer's rows for every read-out; and a composite whose own `cache` is `nothing` forwards to the estimator it holds, which carries the state. `nothing` is refused by name, because there is nothing to read, and so is a state of a shape the verb does not know.

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

[`coverage_policy`](@ref) answers for the two families that can carry one; every other covariance estimator answers `nothing`, so the panel form reduces to the Coverage Universe as the moment seam does for them.

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

Mean of each column of `X` over its finite entries, `NaN` for a column with none.

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

The per-step kernel of [`covariance_forecast_evaluation`](@ref), on bare arrays, so a caller holding forecasts of their own builds the same Result by hand. It reads the forecast on the assets that are active at the step, centres the test rows on the location the forecast is about, forms the realised matrix through the target, and answers the diagnostics of the step: three that read the forecast without a portfolio, and two per test portfolio.

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

With no gap in the test window ``\\mathbf{H} = h \\mathbf{1}\\mathbf{1}^\\intercal``, and the forms reduce to ``m_t = \\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{S}_t) / (N h)``, ``d_{t,i} = (\\mathbf{S}_t)_{ii} / (h (\\hat{\\mathbf{\\Sigma}}_t)_{ii})``, ``L^{\\mathrm{QLIKE}}_t = h \\log|\\hat{\\mathbf{\\Sigma}}_t| + \\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{S}_t)``, ``L^{\\mathrm{F}}_t = \\lVert \\mathbf{S}_t / h - \\hat{\\mathbf{\\Sigma}}_t \\rVert_F^2`` and ``b_t = \\boldsymbol{w}^\\intercal \\boldsymbol{R}_t / \\sqrt{h\\, \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}}_t \\boldsymbol{w}}``. When the forecast is the true covariance, ``\\mathbb{E}[m_t] = \\mathbb{E}[d_{t,i}] = 1`` and ``b_t`` has mean zero and variance one, whatever the distribution of the returns; under Gaussian returns ``m_t`` times its degrees of freedom is chi-squared on them ([`target_dof`](@ref)). A ratio above one is an under-prediction of the dispersion in the metric of the forecast, and one below it an over-prediction. ``L^{\\mathrm{QLIKE}}_t`` is ``-2`` times the Gaussian log-likelihood of the step's returns under the forecast, up to a constant, and ``L^{\\mathrm{F}}_t`` the squared distance of the forecast from the per-cell proxy; the expected value of each is minimised by the true covariance under any conditionally unbiased proxy, so either ranks two forecasts on the proxy as it would on the truth, and neither's level is a calibration reading on its own. ``L^{\\mathrm{P}}_t`` is the univariate QLIKE of the portfolio's variance, and it reads the per-row portfolio returns whichever target formed ``\\mathbf{S}_t``.

# Algorithm

 1. Take the active subset `a`: the assets whose forecast variance and location are finite and which have at least one finite test return. Refuse an empty subset by name.
 2. Centre the test rows of `a` on `c[a]`, leaving a non-finite cell non-finite, giving `Zc`.
 3. Form `(S, H)` through [`realised_target`](@ref), and the effective forecast `H .* sigma[a, a]`.
 4. Factor `sigma[a, a]` once by Cholesky, for its log-determinant and, when `H` is uniform, for the whitening; a `H` that varies by cell factors the effective forecast a second time.
 5. Read the Mahalanobis ratio, the diagonal ratio of each active asset (`NaN` outside `a`), the QLIKE loss and the Frobenius loss.
 6. Resolve the test portfolios over `a` through [`resolve_forecast_weights`](@ref), and read each one's standardised return and portfolio QLIKE from the column sums and the per-row portfolio returns of `Zc`, a non-finite cell counting zero.

# Arguments

  - `sigma`: The forecast, `assets × assets`; a `NaN` diagonal marks an inactive asset.
  - `Z`: The test rows, `observations × assets`; a non-finite cell is an observation the asset did not trade.
  - `c`: The location the forecast is about, `assets × 1`.
  - `w`: The test portfolios on the full universe: `nothing` for inverse volatility over the active subset, a vector for one portfolio, a vector of vectors for several. Each is renormalised over the active subset.
  - `target`: The realised quantity.

# Validation

  - `sigma` is square and `Z`, `c` and every portfolio of `w` have its width. A `DimensionMismatch` is thrown otherwise.
  - At least one asset is active: a finite forecast variance, a finite location, and a finite test return. An `ArgumentError` is thrown otherwise.

# Returns

  - `step::NamedTuple`: `n_valid`, the active count; `mahalanobis_ratio`; `diagonal_ratio`, `assets × 1`; `qlike`; `frobenius`; `standardised_return`, one per portfolio; and `portfolio_qlike`, one per portfolio.

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

`nothing` is inverse volatility recomputed from the step's forecast, ``w_i \\propto 1 / \\sqrt{(\\hat{\\mathbf{\\Sigma}}_t)_{ii}}`` over the active assets, the reference implementation's default and the one that stops the most volatile asset dominating the reading. A vector is one static portfolio and a vector of vectors several; each is cut to the active subset and renormalised to sum to one there, as the reference does, so a delisted asset's weight is spread over the assets that remain. No guard is placed on a subset whose weights sum to zero: the portfolio is then not a portfolio, and its ratio is not a number.

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

The carrier arity the fold loop's online arm calls, [`online_folds`](@ref), for the two kinds of estimator [`covariance_forecast_evaluation`](@ref) threads through it. A prior takes [`fold_prior`](@ref), the one forward the optimiser's step makes. A covariance estimator takes `rd.X` in its own arity, with the active mask of a time-varying Asset Panel as the keyword its step reads, through [`step_active_mask`](@ref), which also refuses what the step cannot carry.

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

Answer the two per-type predicates [`fold_loop`](@ref) reads, for an estimator that is not an optimiser.

No covariance estimator and no prior holds a [`TimeDependent`](@ref) schedule — the schedule type is declared after both families, so no field of theirs can name it — and neither reads a previous fold's weights. An [`Online`](@ref) wrapper at the root answers the same, because the loop resolves it before the first fold. Both answer `false`, so the loop neither resolves a context nor threads weights, and a batch run of the evaluation takes the parallel arm under the sequential executor it is handed.

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

Name the state a wrapped estimator carries at the entry of the fold loop's online arm, or answer `nothing`.

The [`Online`](@ref) arm of [`online_entry_state`](@ref), for the wrapper at the root of [`covariance_forecast_evaluation`](@ref): a wrapper holds no `cache` of its own, so the walk descends into the estimator it wraps and prefixes `est` to what it finds there, `est.cache` for a covariance estimator handed over already folded. The generic walk would reach the same answer through [`estimator_fields`](@ref); this arm states it without asking the wrapper for a field it does not have.

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

Hand the next fold what the last one held, when the fold's result is a step record and not a prediction.

The kernel's step record carries no weights, so there is nothing to thread and `prev` stays where it was. This is the arm [`online_folds`](@ref) reaches under [`covariance_forecast_evaluation`](@ref).

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

Read a fold's forecast and the location it is about, by whether the fold carries a training window.

The two arms of [`covariance_forecast_evaluation`](@ref)'s callback, chosen by dispatch on the fold's `train` exactly as [`fit_fold_result`](@ref) chooses the optimiser's. A window fits the estimator over it through the Asset Panel seam of the moment verbs, `cov(ce, X, pnl)` and [`forecast_location`](@ref)`(ce, X, pnl)`, or `prior(pe, rd)`, the refit every batch fold runs. `nothing` says *the estimator holds its window*, so the fold reads the state out, `cov(ce)` and `forecast_location(ce)`, or `prior(pe)`. A prior is read once for both moments.

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
    pr = prior(pe, port_opt_view(rd, train_idx, :))
    return pr.sigma, pr.mu
end
function forecast_moments(pe::AbstractPriorEstimator, ::ReturnsResult, ::Nothing)
    pr = prior(pe)
    return pr.sigma, pr.mu
end
"""
$(DocStringExtensions.TYPEDEF)

The per-step diagnostics of a covariance forecast over a walk-forward, and its forecasts on request.

`CovarianceForecastEvaluationResult` is what [`covariance_forecast_evaluation`](@ref) returns. It keeps every diagnostic per step — three that read the forecast on its own, two per test portfolio — so the summary, the comparison and the plots are verbs over it, and it keeps the forecasts themselves only when asked, because a run of `M` steps over `N` assets pairs `N² × M` numbers with `N × M`, and the run the online form exists for is a step at every observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceForecastEvaluationResult(
        dates, test_idx, horizon, target, n_valid, mahalanobis_ratio, diagonal_ratio,
        qlike, frobenius, standardised_return, portfolio_qlike, w, sigma, location
    ) -> CovarianceForecastEvaluationResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a Result, so [`covariance_forecast_evaluation`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`covariance_forecast_evaluation`](@ref)
  - [`covariance_forecast_step`](@ref)
  - [`covariance_forecast_summary`](@ref)
  - [`covariance_forecast_compare`](@ref)
  - [`covariance_forecast_portfolio`](@ref)
"""
@concrete struct CovarianceForecastEvaluationResult <: AbstractResult
    """
    Label of each step: the timestamp of its first test row when the carrier holds one, else the row's index. One entry per step.
    """
    dates
    """
    Test rows of each step, in split order. One window per step.
    """
    test_idx
    """
    Horizon of each step, the number of test rows the forecast was judged on. One entry per step; every entry is the scheme's `test_size` under an index walk-forward, and a calendar period's own length under a date walk-forward.
    """
    horizon
    """
    Realised quantity the forecasts were judged against.
    """
    target
    """
    Number of active assets at each step: a finite forecast variance and at least one finite test return. One entry per step.
    """
    n_valid
    """
    Mahalanobis ratio of each step. One entry per step; the target is one.
    """
    mahalanobis_ratio
    """
    Diagonal ratio of each asset at each step, `steps × assets`; `NaN` where the asset was not active. The target is one.
    """
    diagonal_ratio
    """
    QLIKE loss of each step. One entry per step; lower is better, and only a difference between two forecasts is a reading.
    """
    qlike
    """
    Frobenius loss of each step. One entry per step; lower is better, and only a difference between two forecasts is a reading.
    """
    frobenius
    """
    Standardised return of each test portfolio at each step, `steps × portfolios`. Mean zero and standard deviation one under a calibrated forecast.
    """
    standardised_return
    """
    QLIKE loss of each test portfolio's variance at each step, `steps × portfolios`; lower is better.
    """
    portfolio_qlike
    """
    Test portfolios as handed to the evaluation: `nothing` for inverse volatility per step, a vector for one portfolio, a vector of vectors for several.
    """
    w
    """
    Forecast of each step, one `assets × assets` matrix per step, or `nothing` unless `store_forecasts = true`.
    """
    sigma
    """
    Location each forecast was centred on, one `assets × 1` vector per step, or `nothing` unless `store_forecasts = true`.
    """
    location
end
"""
    covariance_forecast_evaluation(est::Union{<:AbstractCovarianceEstimator, <:AbstractPriorEstimator, <:Online},
                                   rd::ReturnsResult, cv::WFCVER; w::Option{<:VecNum_VecVecNum} = nothing,
                                   target::AbstractRealisedTarget = RealisedCovariance(),
                                   store_forecasts::Bool = false) -> CovarianceForecastEvaluationResult

Evaluate a covariance forecast out of sample over a walk-forward, in batch or online.

One verb through the one fold loop. Per fold the callback reads the forecast and its location off the fold's estimator — a refit over the training window when the fold carries one, the threaded state when the scheme declares a Fold Fit of [`OnlineStep`](@ref) — and scores it on the test rows through [`covariance_forecast_step`](@ref). The batch expanding, batch rolling, online expanding and online rolling forms are therefore the four compositions the walk-forward and the [`Online`](@ref) wrapper already express: `expand_train = true`, `expand_train = false`, `ff = OnlineStep()`, and `ff = OnlineStep()` with `Online(est; max_history = w)`. The verb reads none of them, so the online run reaches the batch expanding run's rows fold for fold, to the tolerance of the estimator's own fold, and the date form, the purge and a listing or delisting come for free.

The estimator is threaded as the configuration alone. Under `OnlineStep()` the loop warms one estimator up on the first training window and folds each fold's new rows into it, so an estimator entering with a state, or one that cannot fold, is refused at the door by name; wrap such an estimator in [`Online`](@ref) to fold it from a buffer, and the read-out is then a batch fit over the buffer's rows.

# Arguments

  - `est`: The covariance estimator, or the prior estimator whose `sigma` is the forecast; either may be wrapped in [`Online`](@ref) under `ff = OnlineStep()`, to fold from a buffer and, with `max_history`, over a rolling window.
  - $(arg_dict[:rd])
  - `cv`: The walk-forward, index or date form, or its split.
  - `w`: The test portfolios on the full universe: `nothing` for inverse volatility recomputed from each step's forecast, a vector for one static portfolio, a vector of vectors for several.
  - `target`: The realised quantity the forecasts are judged against.
  - `store_forecasts`: Whether to keep every step's forecast and location in the Result, for [`covariance_forecast_portfolio`](@ref).

# Validation

  - `rd.X` is not `nothing`. An `IsNothingError` is thrown otherwise.
  - The scheme's folds are not shuffled, through [`assert_unshuffled_folds`](@ref).
  - An [`Online`](@ref) at the root is handed a scheme that declares a Fold Fit. An `ArgumentError` is thrown otherwise: a batch fold would never seed the buffer.
  - Everything [`covariance_forecast_step`](@ref) refuses at a step, and everything the fold loop's online arm refuses at entry under `OnlineStep()`.

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
  - [`OnlineStep`](@ref)
  - [`Online`](@ref)
"""
function covariance_forecast_evaluation(est::Union{<:AbstractCovarianceEstimator,
                                                   <:AbstractPriorEstimator, <:Online},
                                        rd::ReturnsResult, cv::WFCVER;
                                        w::Option{<:VecNum_VecVecNum} = nothing,
                                        target::AbstractRealisedTarget = RealisedCovariance(),
                                        store_forecasts::Bool = false)
    @argcheck(!isnothing(rd.X), IsNothingError("rd.X cannot be nothing"))
    @argcheck(!isa(est, Online) || !isnothing(fold_fit(cv)),
              ArgumentError("`Online($(typeof(est.est).name.name))` declares a refit from a sample buffer, which only the fold loop's online arm seeds and threads, and this scheme declares no Fold Fit, so every fold would refit the estimator from its training window and the buffer would never exist. Declare `ff = OnlineStep()` on the scheme, or hand the evaluation the estimator unwrapped."))
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    n = length(train_idx)
    steps, _ = fold_loop(est, n, FLoops.SequentialEx(), NamedTuple; rd = rd,
                         train_idx = train_idx, test_idx = test_idx, cv = cv) do fold
        sigma, c = forecast_moments(fold.est, fold.rd, fold.train)
        step = covariance_forecast_step(sigma, view(fold.rd.X, fold.test, :), c, w, target)
        return store_forecasts ? (; step..., sigma = sigma, location = c) : step
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
