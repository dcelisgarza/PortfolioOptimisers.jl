"""
    apply_fitted_step(fitted, data) -> data′

Replay one fitted pipeline step on a data window during prediction.

Preprocessing steps transform the window at the data level they apply to: price-level fitted objects ([`AbstractPricesPreprocessingResult`](@ref), [`AbstractPricesPreprocessingEstimator`](@ref)) transform price-level windows, returns-level ones transform returns-level windows, and [`PricesToReturns`](@ref) converts the window from prices to returns. A fitted object whose data level does not match the current window passes it through unchanged — mirroring fit, where such a step cannot affect the data that reaches the optimiser. Non-preprocessing fitted results (priors, phylogeny, uncertainty, constraints, optimisation) pass the window through untouched, and a nested [`PipelineResult`](@ref) replays its own steps recursively.

# Arguments

  - `fitted`: A fitted per-step result from a [`PipelineResult`](@ref).
  - `data`: The current data window ([`AbstractPricesResult`](@ref) or [`AbstractReturnsResult`](@ref)).

# Returns

  - `data′`: The transformed (or untouched) data window.

# Related

  - [`apply_fitted_steps`](@ref)
  - [`apply_preprocessing`](@ref)
  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
"""
function apply_fitted_step(::Any,
                           data::Union{<:AbstractPricesResult, <:AbstractReturnsResult})
    return data
end
function apply_fitted_step(f::PricesToReturns, pr::AbstractPricesResult)
    return apply_preprocessing(f, pr)
end
apply_fitted_step(::PricesToReturns, rd::AbstractReturnsResult) = rd
function apply_fitted_step(f::Union{<:AbstractPricesPreprocessingResult,
                                    <:AbstractPricesPreprocessingEstimator},
                           pr::AbstractPricesResult)
    return apply_preprocessing(f, pr)
end
function apply_fitted_step(::Union{<:AbstractPricesPreprocessingResult,
                                   <:AbstractPricesPreprocessingEstimator},
                           rd::AbstractReturnsResult)
    return rd
end
function apply_fitted_step(f::Union{<:AbstractReturnsPreprocessingResult,
                                    <:AbstractReturnsPreprocessingEstimator},
                           rd::AbstractReturnsResult)
    return apply_preprocessing(f, rd)
end
function apply_fitted_step(::Union{<:AbstractReturnsPreprocessingResult,
                                   <:AbstractReturnsPreprocessingEstimator},
                           pr::AbstractPricesResult)
    return pr
end
function apply_fitted_step(f::PipelineResult,
                           data::Union{<:AbstractPricesResult, <:AbstractReturnsResult})
    return apply_fitted_steps(f.results, data)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replay the fitted preprocessing steps of a pipeline on a data window, in step order.

# Arguments

  - `results`: The fitted per-step results of a [`PipelineResult`](@ref).
  - `data`: The data window to transform.

# Returns

  - `data′`: The transformed data window (returns-level when the steps include a [`PricesToReturns`](@ref) conversion).

# Related

  - [`apply_fitted_step`](@ref)
  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
"""
function apply_fitted_steps(results::Tuple,
                            data::Union{<:AbstractPricesResult, <:AbstractReturnsResult})
    for f in results
        data = apply_fitted_step(f, data)
    end
    return data
end
"""
    predict(res::PipelineResult, data::AbstractPricesResult, window = :) -> PredictionResult
    predict(res::PipelineResult, data::AbstractReturnsResult, window = :) -> PredictionResult

Apply a fitted pipeline to an unseen data window and produce the same [`PredictionResult`](@ref) the weights-level machinery consumes.

The `window` selects observation rows of `data` (integer indices, timestamps, or `:` for all rows). The window is transformed by replaying the fitted preprocessing steps in step order — the *training* universe subset, the *training* imputation parameters, then the returns conversion — so no statistics of the test window leak into the transformation. The result is then handed to the existing weights-level `predict`, so scorers and risk measures carry over untouched.

Price-level data requires the pipeline to contain a [`PricesToReturns`](@ref) step; a pipeline that produced no optimisation result cannot predict.

# Arguments

  - `res`: The fitted [`PipelineResult`](@ref).
  - `data`: Price- or returns-level data containing the window ([`PricesResult`](@ref) or [`ReturnsResult`](@ref)).
  - `window`: Observation window into the rows of `data`. Integer indices, timestamps, or `:` (all rows).

# Returns

  - `pred::PredictionResult`: The weights-level prediction on the transformed window.

# Related

  - [`PipelineResult`](@ref)
  - [`apply_fitted_steps`](@ref)
  - [`prices_view`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
"""
function StatsAPI.predict(res::PipelineResult, data::AbstractPricesResult, window = Colon())
    opt = res.ctx.opt
    @argcheck(!isnothing(opt),
              IsNothingError("the pipeline produced no optimisation result; add a terminal optimisation step before predicting"))
    pr = prices_view(data, window)
    rd = apply_fitted_steps(res.results, pr)
    @argcheck(isa(rd, AbstractReturnsResult),
              ArgumentError("the pipeline's fitted steps do not convert price-level data to returns; predicting on a $(typeof(data)) requires a PricesToReturns step"))
    assert_universe_aligned(res, rd)
    return StatsAPI.predict(opt, rd)
end
function StatsAPI.predict(res::PipelineResult, data::AbstractReturnsResult,
                          window = Colon())
    opt = res.ctx.opt
    @argcheck(!isnothing(opt),
              IsNothingError("the pipeline produced no optimisation result; add a terminal optimisation step before predicting"))
    rd = isa(window, Colon) ? data : returns_result_view(data, window, :)
    rd = apply_fitted_steps(res.results, rd)
    assert_universe_aligned(res, rd)
    return StatsAPI.predict(opt, rd)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that replaying a pipeline's fitted steps on a test window reproduces the training asset universe.

The terminal weights are indexed by the *training* universe, so a test window whose transformed returns carry a different asset set (or a different asset order) would silently misalign weights and returns. This is the failure the fit/apply contract exists to prevent, so it is reported as an error naming both universes rather than surfacing as a dimension mismatch inside the risk calculation.

The usual cause is relying on [`PricesToReturns`](@ref) alone to define the universe: it is stateless, and the underlying [`prices_to_returns`](@ref) drops assets that are entirely missing in the window being converted, which differs between train and test. Pin the universe with a [`MissingDataFilter`](@ref) step, and fill the remaining gaps with an [`Imputer`](@ref) step, before converting.

# Arguments

  - `res`: The fitted [`PipelineResult`](@ref).
  - `rd`: The transformed test-window returns.

# Returns

  - `nothing`.

# Related

  - [`predict(res::PipelineResult, data::AbstractPricesResult, window)`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`Imputer`](@ref)
"""
function assert_universe_aligned(res::PipelineResult, rd::AbstractReturnsResult)::Nothing
    train = res.ctx.returns
    if isnothing(train)
        return nothing
    end
    @argcheck(rd.nx == train.nx,
              ArgumentError("the pipeline's fitted steps produced a test-window universe $(rd.nx) that differs from the training universe $(train.nx), so the weights and the test returns would not be aligned. PricesToReturns is stateless and drops assets that are entirely missing in the window it converts; pin the universe with a MissingDataFilter step (and an Imputer step to fill the remaining gaps) before converting to returns."))
    return nothing
end
"""
    Base.split(ccv::CombinatorialCrossValidation, pr::AbstractPricesResult)

Deliberately unsupported: combinatorial cross-validation stays returns-level.

Combinatorial folds recombine non-contiguous test groups, which is incompatible with the contiguous-window semantics price-level (pipeline) splitting relies on for stateful preprocessing. Throws an `ArgumentError`; split returns-level data instead, or use [`KFold`](@ref) / a [`WalkForwardEstimator`](@ref) at the prices level.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`Base.split(kf::KFold, rd::Rd_Pr)`](@ref)
"""
function Base.split(::CombinatorialCrossValidation, ::AbstractPricesResult)
    return throw(ArgumentError("CombinatorialCrossValidation is returns-level only; price-level (pipeline) splitting requires contiguous windows — use KFold or a walk-forward scheme instead"))
end
"""
    Base.split(mrcv::MultipleRandomised, pr::AbstractPricesResult)

Deliberately unsupported: multiple-randomised cross-validation stays returns-level.

Its resampled asset subsets and randomised paths are defined over the returns matrix, not over price observations, so it cannot drive the contiguous input-row windows price-level (pipeline) splitting requires. Throws an `ArgumentError`; split returns-level data instead.

# Related

  - [`MultipleRandomised`](@ref)
  - [`Base.split(kf::KFold, rd::Rd_Pr)`](@ref)
"""
function Base.split(::MultipleRandomised, ::AbstractPricesResult)
    return throw(ArgumentError("MultipleRandomised is returns-level only; price-level (pipeline) splitting requires contiguous input-row windows — use KFold or a walk-forward scheme instead"))
end
"""
    port_opt_view(pipe::Pipeline, i, args...; kwargs...)

Deliberately unsupported: a [`Pipeline`](@ref) cannot be sub-selected by asset view.

Meta-optimisers (`NestedClustered`, `Stacking`, `SubsetResampling`) build asset sub-portfolios by taking a `port_opt_view` of their inner estimator. A pipeline's asset universe is *fitted state* — the missing-data filter decides it from the training window — so an asset view taken before fitting is not well defined. Wrapping a `Pipeline` in a meta-optimiser is therefore unsupported in v1 (ADR 0028, "Future expansion"); a meta-optimiser may still be the *optimisation step of* a pipeline.

# Related

  - [`Pipeline`](@ref)
  - [`optimise(::Pipeline)`](@ref)
"""
function port_opt_view(::Pipeline, args...; kwargs...)
    return throw(ArgumentError("a Pipeline cannot be sub-selected with port_opt_view: its asset universe is fitted state, so wrapping a Pipeline inside a meta-optimiser is unsupported (ADR 0028). A meta-optimiser may be used as the optimisation step of a Pipeline instead."))
end
