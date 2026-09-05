"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation estimators.

# Related

  - [`CrossValidationResult`](@ref)
  - [`OptimisationCrossValidationEstimator`](@ref)
  - [`NonOptimisationCrossValidationEstimator`](@ref)
"""
abstract type CrossValidationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation result types.

# Related

  - [`CrossValidationEstimator`](@ref)
  - [`OptimisationCrossValidationResult`](@ref)
  - [`NonOptimisationCrossValidationResult`](@ref)
"""
abstract type CrossValidationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation algorithm types.

# Related

  - [`CrossValidationEstimator`](@ref)
  - [`CrossValidationResult`](@ref)
"""
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Identity split for [`CrossValidationResult`](@ref). Returns the result unchanged, used as a no-op fallback when splitting is not applicable.
"""
function Base.split(res::CrossValidationResult, args...)
    return res
end
"""
    CVER = Union{<:CrossValidationEstimator, <:CrossValidationResult}

Union of all cross-validation estimators and result types.
"""
const CVER = Union{<:CrossValidationEstimator, <:CrossValidationResult}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for cross-validation estimators used in portfolio optimisation.
Subtypes implement different splitting strategies (sequential or non-sequential) for
out-of-sample testing of optimisation pipelines.

# Related

  - [`CrossValidationEstimator`](@ref)
  - [`SequentialCrossValidationEstimator`](@ref)
  - [`NonSequentialCrossValidationEstimator`](@ref)
"""
abstract type OptimisationCrossValidationEstimator <: CrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for sequential optimisation cross-validation estimators. Sequential
schemes produce time-ordered, non-overlapping folds (e.g. walk-forward).

# Related

  - [`OptimisationCrossValidationEstimator`](@ref)
  - [`SequentialCrossValidationResult`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
"""
abstract type SequentialCrossValidationEstimator <: OptimisationCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential optimisation cross-validation estimators. Non-
sequential schemes may produce randomly sampled or combinatorial folds.

# Related

  - [`OptimisationCrossValidationEstimator`](@ref)
  - [`NonSequentialCrossValidationResult`](@ref)
  - [`KFold`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
abstract type NonSequentialCrossValidationEstimator <: OptimisationCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all optimisation cross-validation result types.

# Related

  - [`CrossValidationResult`](@ref)
  - [`SequentialCrossValidationResult`](@ref)
  - [`NonSequentialCrossValidationResult`](@ref)
"""
abstract type OptimisationCrossValidationResult <: CrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for sequential optimisation cross-validation results.

# Related

  - [`OptimisationCrossValidationResult`](@ref)
  - [`SequentialCrossValidationEstimator`](@ref)
  - [`WalkForwardResult`](@ref)
"""
abstract type SequentialCrossValidationResult <: OptimisationCrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential optimisation cross-validation results.

# Related

  - [`OptimisationCrossValidationResult`](@ref)
  - [`NonSequentialCrossValidationEstimator`](@ref)
  - [`KFoldResult`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
"""
abstract type NonSequentialCrossValidationResult <: OptimisationCrossValidationResult end
# The split has already happened, so the count is the length of the enumeration the result
# carries, and the data argument the estimator methods take is not read. This is the
# `n_splits(cv)` and `n_splits(cv, rd)` pair that the `n_splits` docstring promises for a
# result type; every concrete `OptimisationCrossValidationResult` carries `test_idx`.
function n_splits(res::OptimisationCrossValidationResult)
    return length(res.test_idx)
end
function n_splits(res::OptimisationCrossValidationResult, ::Prices_RR)
    return n_splits(res)
end
"""
    OptCVER

Union of all optimisation cross-validation estimators and results.
"""
const OptCVER = Union{<:OptimisationCrossValidationEstimator,
                      <:OptimisationCrossValidationResult}

"""
    NonSeqCVER

Union of all non-sequential cross-validation estimators and results.
"""
const NonSeqCVER = Union{<:NonSequentialCrossValidationEstimator,
                         <:NonSequentialCrossValidationResult}
"""
    SeqCVER

Union of all sequential cross-validation estimators and results.
"""
const SeqCVER = Union{<:SequentialCrossValidationEstimator,
                      <:SequentialCrossValidationResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the number of observations (rows) cross-validation folds index into.

# Arguments

  - `data`: Returns-level or price-level data ([`Prices_RR`](@ref)).

# Returns

  - `T::Integer`: The number of observation rows.

# Related

  - [`cv_timestamps`](@ref)
  - [`Base.split`](@ref)
"""
cv_nobs(rd::AbstractReturnsResult) = size(rd.X, 1)
cv_nobs(pr::AbstractPricesResult) = size(TimeSeries.values(pr.X), 1)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the timestamp vector aligned with the observation rows of `data`, or `nothing` when it has none.

# Arguments

  - `data`: Returns-level or price-level data ([`Prices_RR`](@ref)).

# Returns

  - `ts`: Timestamp vector, or `nothing`.

# Related

  - [`cv_nobs`](@ref)
  - [`Base.split`](@ref)
"""
cv_timestamps(rd::AbstractReturnsResult) = rd.ts
cv_timestamps(pr::AbstractPricesResult) = TimeSeries.timestamp(pr.X)
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for cross-validation estimators used in non-optimisation contexts
(e.g. resampling for hierarchical clustering or phylogeny methods).

# Related

  - [`CrossValidationEstimator`](@ref)
"""
abstract type NonOptimisationCrossValidationEstimator <: CrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for sequential non-optimisation cross-validation estimators. Sequential
schemes produce time-ordered, non-overlapping folds.

# Related

  - [`NonOptimisationCrossValidationEstimator`](@ref)
  - [`NonOptimisationSequentialCrossValidationResult`](@ref)
"""
abstract type NonOptimisationSequentialCrossValidationEstimator <:
              NonOptimisationCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential non-optimisation cross-validation estimators. Non-
sequential schemes may produce randomly sampled or combinatorial folds.

# Related

  - [`NonOptimisationCrossValidationEstimator`](@ref)
  - [`NonOptimisationNonSequentialCrossValidationResult`](@ref)
"""
abstract type NonOptimisationNonSequentialCrossValidationEstimator <:
              NonOptimisationCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for result types produced by non-optimisation cross-validation
routines.

# Related

  - [`CrossValidationResult`](@ref)
  - [`NonOptimisationCrossValidationEstimator`](@ref)
  - [`NonOptimisationSequentialCrossValidationResult`](@ref)
  - [`NonOptimisationNonSequentialCrossValidationResult`](@ref)
"""
abstract type NonOptimisationCrossValidationResult <: CrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for sequential non-optimisation cross-validation result types.

# Related

  - [`NonOptimisationCrossValidationResult`](@ref)
  - [`NonOptimisationSequentialCrossValidationEstimator`](@ref)
"""
abstract type NonOptimisationSequentialCrossValidationResult <:
              NonOptimisationCrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential non-optimisation cross-validation result types.

# Related

  - [`NonOptimisationCrossValidationResult`](@ref)
  - [`NonOptimisationNonSequentialCrossValidationEstimator`](@ref)
"""
abstract type NonOptimisationNonSequentialCrossValidationResult <:
              NonOptimisationCrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Stores the portfolio returns data associated with a cross-validation prediction. Packages
asset returns, factor returns, benchmark returns, timestamps, implied volatilities, and
the implied volatility risk premium adjustment for use in prediction result types.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PredictionReturnsResult(;
        nx::Option{<:VecStr} = nothing,
        X::Option{<:VecNum_VecVecNum} = nothing,
        nf::Option{<:VecStr} = nothing,
        F::Option{<:MatNum} = nothing,
        nb::Option{<:VecStr} = nothing,
        B::Option{<:VecNum_VecVecNum} = nothing,
        ts::Option{<:VecDate} = nothing,
        iv::Option{<:VecNum_VecVecNum} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing
    ) -> PredictionReturnsResult

Keywords correspond to the struct's fields.

## No feature matrix

This carrier used to transport a per-fold collapsed feature matrix, so that [`rebuild_returns_result`](@ref) could stack the folds into the outer problem's. It no longer does: the outer collapse is recomputed at the assembly seam from the original, unsliced `rd.Z` and the fold's weights, which is the *same* call the non-cross-validated path makes (see [`rebuild_returns_result`](@ref)). Nothing is lost — the fold's weights are on [`PredictionResult`](@ref)'s `res`, and `ts` here is the fold's slice of the original clock, which is everything the seam needs to reconstruct any fold's view.

## Validation

  - `nf` and `F` must be consistent (both nothing, or `F` has `length(nf)` columns).
  - If `X` and `F` provided: row count of `F` matches length of each `X` vector.
  - If `B` and `X` provided: same type (`VecNum`/`VecVecNum`) and matching lengths.
  - If `ts` provided: `!isempty(ts)`; at least one of `X`, `F` is not `nothing`; lengths of `ts` match `X`, `F`, and `B` where applicable.
  - If `iv` is a `VecNum`: `ivpa` is scalar or nothing; `iv` is non-empty, non-negative, and finite; `length(iv) == length(X)`.
  - If `iv` is a `VecVecNum`: `ivpa` is `VecNum` or nothing; `length(iv) == length(X) == length(ivpa)`; each sub-vector non-empty, non-negative, finite, and same length as corresponding `X`.

# Related

  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`rebuild_returns_result`](@ref)
"""
@concrete struct PredictionReturnsResult <: AbstractReturnsResult
    """
    $(field_dict[:pred_nx])
    """
    nx
    """
    $(field_dict[:X])
    """
    X
    """
    $(field_dict[:pred_nf])
    """
    nf
    """
    $(field_dict[:F])
    """
    F
    """
    $(field_dict[:pred_nb])
    """
    nb
    """
    $(field_dict[:pred_B])
    """
    B
    """
    $(field_dict[:ts])
    """
    ts
    """
    $(field_dict[:iv_ret])
    """
    iv
    """
    $(field_dict[:ivpa])
    """
    ivpa
    function PredictionReturnsResult(nx::Option{<:VecStr}, X::Option{<:VecNum_VecVecNum},
                                     nf::Option{<:VecStr}, F::Option{<:MatNum},
                                     nb::Option{<:VecStr}, B::Option{<:VecNum_VecVecNum},
                                     ts::Option{<:VecDate}, iv::Option{<:VecNum_VecVecNum},
                                     ivpa::Option{<:Num_VecNum})
        check_names_and_returns_matrix(nf, F, :nf, :F)
        if !isnothing(X) && !isnothing(F)
            if isa(X, VecNum)
                @argcheck(length(X) == size(F, 1), DimensionMismatch)
            else
                @argcheck(all(x -> length(x) == size(F, 1), X),
                          DimensionMismatch("each element of X must have the same length as the number of rows in F"))
            end
        end
        if !isnothing(B) && !isnothing(X)
            if isa(B, VecNum) && isa(X, VecNum)
                @argcheck(length(B) == length(X), DimensionMismatch)
            elseif isa(B, VecVecNum) && isa(X, VecVecNum)
                @argcheck(length(B) == length(X), DimensionMismatch)
                for (x, b) in zip(X, B)
                    @argcheck(length(x) == length(b), DimensionMismatch)
                end
            else
                throw(ArgumentError("If B is a vector of scalars, X must also be a vector of scalars, and if B is a vector of vectors, X must be a vector of vectors, got typeof(X) = $(typeof(X)), typeof(B) = $(typeof(B))"))
            end
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)), IsNothingError)
            if isa(X, VecNum)
                @argcheck(length(ts) == length(X), DimensionMismatch)
            elseif isa(X, VecVecNum)
                @argcheck(all(x -> length(x) == length(ts), X),
                          DimensionMismatch("each element of X must have length $(length(ts))"))
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1), DimensionMismatch)
            end
            if isa(B, VecNum)
                @argcheck(length(ts) == length(B), DimensionMismatch)
            elseif isa(B, VecVecNum)
                @argcheck(all(x -> length(x) == length(ts), B),
                          DimensionMismatch("each element of B must have length $(length(ts))"))
            end
        end
        if isa(iv, VecNum)
            @argcheck(isa(ivpa, Option{<:Number}),
                      ArgumentError("ivpa must be a scalar (or nothing) when iv is a vector of numbers, got typeof(ivpa) = $(typeof(ivpa))"))
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(length(iv) == length(X), DimensionMismatch)
        elseif isa(iv, VecVecNum)
            @argcheck(isa(ivpa, Option{<:VecNum}),
                      ArgumentError("ivpa must be a vector of numbers (or nothing) when iv is a vector of vectors of numbers, got typeof(ivpa) = $(typeof(ivpa))"))
            @argcheck(length(iv) == length(X), DimensionMismatch)
            @argcheck(length(ivpa) == length(X), DimensionMismatch)
            for (ivi, ivpai, Xi) in zip(iv, ivpa, X)
                assert_nonempty_nonneg_finite_val(ivi, :iv)
                assert_nonempty_gt0_finite_val(ivpai, :ivpa)
                @argcheck(length(ivi) == length(Xi), DimensionMismatch)
            end
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(nb), typeof(B),
                   typeof(ts), typeof(iv), typeof(ivpa)}(nx, X, nf, F, nb, B, ts, iv, ivpa)
    end
end
function PredictionReturnsResult(; nx::Option{<:VecStr} = nothing,
                                 X::Option{<:VecNum_VecVecNum} = nothing,
                                 nf::Option{<:VecStr} = nothing,
                                 F::Option{<:MatNum} = nothing,
                                 nb::Option{<:VecStr} = nothing,
                                 B::Option{<:VecNum_VecVecNum} = nothing,
                                 ts::Option{<:VecDate} = nothing,
                                 iv::Option{<:VecNum_VecVecNum} = nothing,
                                 ivpa::Option{<:Num_VecNum} = nothing)::PredictionReturnsResult
    return PredictionReturnsResult(nx, X, nf, F, nb, B, ts, iv, ivpa)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prediction result types.

All concrete prediction result types from cross-validation should subtype `AbstractPredictionResult`.

# Related

  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
abstract type AbstractPredictionResult <: AbstractResult end
"""
    ruined_retcodes(retcode::VecOptRetCode, ruined::VecInt)

Set the return code of every ruined member of a population to an [`OptimisationFailure`](@ref).

The failure payload names the member and the reason, so a reader of `res.retcode` finds why the member left the run. A member that is not named keeps the code its own optimisation gave it.

# Algorithm

 1. Walk the codes with their positions, and replace the code of a named member with a failure that states the reason.

# Arguments

  - `retcode`: Return codes of the population, one per member.
  - `ruined`: Indices of the members whose drifted wealth is not positive.

# Returns

  - `VecOptRetCode`: The codes, with the ruined members failed.

# Related

  - [`mark_ruined_members`](@ref)
  - [`OptimisationFailure`](@ref)
  - [`held_weights_result`](@ref)
"""
function ruined_retcodes(retcode::VecOptRetCode, ruined::VecInt)
    return OptimisationReturnCode[if i in ruined
                                      OptimisationFailure(;
                                                          res = "the drifted wealth of member $(i) is not positive, so the fold dropped it and its series and held weights are `NaN`")
                                  else
                                      rc
                                  end
                                  for (i, rc) in pairs(retcode)]
end
"""
    mark_ruined_members(res::NonFiniteAllocationOptimisationResult, ruined::Nothing)
    mark_ruined_members(res::NonFiniteAllocationOptimisationResult, ruined::VecInt)

Rebuild a fold's optimisation result so that its ruined members carry a failure code.

The drop needs no machinery of its own. The library already folds a vector of return codes with `any(x -> isa(x, OptimisationFailure), …)`, and the cross-validation path already filters a path on `isa(y.res.retcode, OptimisationSuccess)`, so a failed member takes the path it is on out of the run.

# Algorithm

 1. With no ruined member, and with an empty set of them, give `res` unchanged. Nothing is rebuilt on the ordinary path.
 2. Otherwise rebuild `res` through [`set_retcode`](@ref), with the codes [`ruined_retcodes`](@ref) makes.

# Arguments

  - `res`: Optimisation result of the fold.
  - `ruined`: Indices of the members whose drifted wealth is not positive, or `nothing`.

# Returns

  - `NonFiniteAllocationOptimisationResult`: The result, rebuilt only when a member was ruined.

# Related

  - [`ruined_retcodes`](@ref)
  - [`set_retcode`](@ref)
  - [`held_weights_result`](@ref)
"""
function mark_ruined_members(res::NonFiniteAllocationOptimisationResult, ::Nothing)
    return res
end
function mark_ruined_members(res::NonFiniteAllocationOptimisationResult, ruined::VecInt)
    return if isempty(ruined)
        res
    else
        set_retcode(res, ruined_retcodes(res.retcode, ruined))
    end
end
"""
    warn_ruined_members(wd::AbstractWeightDrift, args...)
    warn_ruined_members(wd::Nothing, ruined::Nothing, n::Integer)
    warn_ruined_members(wd::Nothing, ruined::VecInt, n::Integer)

Warn once when a drift dropped members of a population, and the return series did not.

A fold whose series is drifted is warned about by [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref), which runs the same drift over the same window. A fold that drifts only its held weights has no such site, so the warning is raised here instead. Either way the fold warns once.

# Algorithm

 1. With a drifted series, say nothing. The series already warned.
 2. With no ruined member, say nothing.
 3. Otherwise warn, and name the count and the members.

# Arguments

  - `wd`: Weight drift of the scheme, or `nothing`.
  - `ruined`: Indices of the ruined members, or `nothing`.
  - `n`: Number of members of the population.

# Returns

  - `nothing`.

# Related

  - [`held_weights_result`](@ref)
  - [`mark_ruined_members`](@ref)
"""
function warn_ruined_members(::AbstractWeightDrift, args...)::Nothing
    return nothing
end
function warn_ruined_members(::Nothing, ::Nothing, ::Integer)::Nothing
    return nothing
end
function warn_ruined_members(::Nothing, ruined::VecInt, n::Integer)::Nothing
    if !isempty(ruined)
        @warn "the drifted wealth of $(length(ruined)) of $(n) population member(s) is not positive, so their held weights are `NaN` and their members are dropped: $(ruined)"
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Stores the result of a single cross-validation fold prediction. Pairs an optimisation
result with the returns data from the test period.

## Fold provenance

The fold's rows are not stored here. They do not need to be: `rd.ts` is the fold's slice of
the original clock — [`port_opt_view`](@ref) slices it with the very `test_idx` the fold was
built from — so [`feature_row_indices`](@ref) recovers them by matching timestamps whenever
a consumer needs absolute rows. [`rebuild_returns_result`](@ref) is the one that does, and
recovering rather than storing is what keeps that recovery correct on the combinatorial
path, where a path's folds are assembled in split order rather than chronologically.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PredictionResult(;
        res::NonFiniteAllocationOptimisationResult,
        rd::PredictionReturnsResult,
        hw::Option{<:HeldWeightsResult} = nothing
    ) -> PredictionResult

Keywords correspond to the struct's fields. `res` and `rd` are required, because a fold prediction is meaningless without either half. `hw` defaults to `nothing`, which is the fold that held its target weights on every observation.

## The held-weights record

`hw` is present only when a Weight Drift ran over the fold, so a reader dispatches on its absence rather than testing for it. It carries the asset returns of the fold, the weights held after the last observation and the form that made them, and [`weight_path`](@ref) rebuilds the weight path from it.

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`fit_predict`](@ref)
  - [`PredictionReturnsResult`](@ref)
  - [`rebuild_returns_result`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`weight_path`](@ref)
"""
@concrete struct PredictionResult <: AbstractPredictionResult
    """
    $(field_dict[:pred_res])
    """
    res
    """
    $(field_dict[:rd])
    """
    rd
    """
    $(field_dict[:hw])
    """
    hw
    function PredictionResult(res::NonFiniteAllocationOptimisationResult,
                              rd::PredictionReturnsResult, hw::Option{<:HeldWeightsResult})
        return new{typeof(res), typeof(rd), typeof(hw)}(res, rd, hw)
    end
end
function PredictionResult(; res::NonFiniteAllocationOptimisationResult,
                          rd::PredictionReturnsResult,
                          hw::Option{<:HeldWeightsResult} = nothing)::PredictionResult
    return PredictionResult(res, rd, hw)
end
"""
    previous_weights(pws::Any, prev::Nothing)
    previous_weights(pws::Nothing, prev::PredictionResult)
    previous_weights(pws::AbstractPreviousWeightsSource, prev::PredictionResult)

Read the weights a fold threads into the fold that follows it.

This is the one seam of the Previous-Weights Source. The first fold of a run has no fold behind it, so it threads nothing whatever the source is. A later fold threads the target weights of the previous fold by default, and the weights that fold **held** after its last observation when the source asks for them.

# Algorithm

 1. With no previous fold, give `nothing`.
 2. With no source, give the target weights of the previous fold, `prev.res.w`.
 3. With a source, give the held weights of the previous fold, `prev.hw.w`.

# Arguments

  - `pws`: Previous-weights source of the scheme, or `nothing`.
  - `prev`: Prediction result of the previous fold, or `nothing`.

# Returns

  - `Option{<:VecNum_VecVecNum}`: The weights the next fold reads through [`factory`](@ref).

# Related

  - [`AbstractPreviousWeightsSource`](@ref)
  - [`DriftedWeights`](@ref)
  - [`fold_loop`](@ref)
  - [`HeldWeightsResult`](@ref)
"""
function previous_weights(::Any, ::Nothing)
    return nothing
end
function previous_weights(::Nothing, prev::PredictionResult)
    return prev.res.w
end
function previous_weights(::AbstractPreviousWeightsSource, prev::PredictionResult)
    return prev.hw.w
end
"""
    VecPredRes = AbstractVector{<:PredictionResult}

Alias for a vector of single-fold prediction results.

Represents a collection of [`PredictionResult`](@ref) objects from cross-validation folds.

# Related

  - [`PredictionResult`](@ref)
  - [`VecVecPredRes`](@ref)
"""
const VecPredRes = AbstractVector{<:PredictionResult}
"""
    VecVecPredRes = AbstractVector{<:VecPredRes}

Alias for a vector of vectors of prediction results.

Represents the outer collection of cross-validation paths, where each inner vector contains prediction results from a single path.

# Related

  - [`VecPredRes`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
const VecVecPredRes = AbstractVector{<:VecPredRes}
function expected_risk(r::BaseRM_VecBaseRM, pred::PredictionResult; kwargs...)
    return expected_risk_from_returns(r, pred.rd.X; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Roll a risk measure over the return series a single fold formed.

The realised-history reading of [`rolling_window_measure`](@ref) on a fold: `pred.rd.X` is the series [`predict`](@ref) stored, so no weights are read and none are needed. Under a weight drift that series is the drifted one, and a window of it is a sub-series of the fold's own drift.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `pred::PredictionResult`: Single-fold prediction result.
  - `window::Integer`: Size of the rolling window (number of periods).

# Returns

  - `risks::VecNum`: Expected risk values for each rolling window.

# Related

  - [`rolling_window_measure`](@ref)
  - [`expected_risk`](@ref)
  - [`PredictionResult`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, pred::PredictionResult,
                                window::Integer; kwargs...)
    return rolling_window_measure(r, pred.rd.X, window; kwargs...)
end
"""
    calc_net_asset_returns(pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult}, fees = nothing)
    calc_net_asset_returns(pred::PredictionResult{<:Any, <:Any, Nothing}, args...)

Split a fold's net return series over the assets that produced it.

The fold-taking method of [`calc_net_asset_returns`](@ref), and the mirror of [`calc_net_returns(res::OptimisationResult, X, fees)`](@ref): it resolves the fold's asset returns, its weight path and its fee, so a caller holding a fold reaches the split in one call. The fee is settled against the fold's own length exactly as [`predict`](@ref) settled it, so the rows of the result sum to the series the fold stored, to rounding.

A fold that carries no Held Weights record raises. `pred.rd.X` is the **portfolio** series, not the asset returns, so a fold whose scheme ran neither switch keeps no matrix to split.

# Arguments

  - `pred`: Single-fold prediction result.
  - `fees`: Fees that take precedence over the result's own.
  - `args...`: Additional arguments (ignored by the refusing method).

# Validation

  - The fold carries a [`HeldWeightsResult`](@ref), else an `ArgumentError` naming the two switches.

# Returns

  - `ret::MatNum`: Per asset net returns of the fold, one row per observation.

# Related

  - [`calc_net_asset_returns`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`weight_path`](@ref)
  - [`amortise_fees`](@ref)
  - [`PredictionResult`](@ref)
"""
function calc_net_asset_returns(pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult},
                                fees::Option{<:Fees} = nothing)
    hw = pred.hw
    return calc_net_asset_returns(weight_path(hw, pred.res.w), hw.X,
                                  amortise_fees(extract_fees(pred.res, fees),
                                                size(hw.X, 1)))
end
function calc_net_asset_returns(::PredictionResult{<:Any, <:Any, Nothing}, args...)
    return throw(ArgumentError("`calc_net_asset_returns(pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`.\nSet one of them so the fold records its asset returns, or call `calc_net_asset_returns(w, X, fees)` with the returns you fitted on."))
end
"""
    risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult}, fees = nothing; kwargs...)
    risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, Nothing}, args...; kwargs...)

Decompose a fold's risk over its assets.

The fold-taking method of [`risk_contribution`](@ref). It resolves the fold's **target** weights, its asset returns and its fee, and hands them to the free function unchanged, so the figures are the free function's own.

Under a Weight Drift the figures are exact to **first order in the drift** only, for the reason the free function states: the drifted series is not linear in the target weights, so the contributions sum to the fold's realised risk approximately rather than exactly. The target weights are still what a contribution is reported against, because they are the decision the finite difference perturbs.

A fold that carries no Held Weights record raises, because `pred.rd.X` is the portfolio series and no asset matrix survives on the fold.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to differentiate, or a vector of them.
  - `pred`: Single-fold prediction result.
  - `fees`: Fees that take precedence over the result's own.
  - `args...`: Additional arguments (ignored by the refusing method).

# Validation

  - The fold carries a [`HeldWeightsResult`](@ref), else an `ArgumentError` naming the two switches.

# Returns

  - `Vector`: Risk contributions (or marginal risks) for each asset.

# Related

  - [`risk_contribution`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`PredictionResult`](@ref)
"""
function risk_contribution(r::BaseRM_VecBaseRM,
                           pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult},
                           fees::Option{<:Fees} = nothing; kwargs...)
    hw = pred.hw
    return risk_contribution(r, pred.res.w, hw.X,
                             amortise_fees(extract_fees(pred.res, fees), size(hw.X, 1));
                             kwargs...)
end
function risk_contribution(::BaseRM_VecBaseRM, ::PredictionResult{<:Any, <:Any, Nothing},
                           args...; kwargs...)
    return throw(ArgumentError("`risk_contribution(r, pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`.\nSet one of them so the fold records its asset returns, or call `risk_contribution(r, pred.res.w, rd.X, fees)` with the returns you fitted on."))
end
"""
    factor_risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult}, fees = nothing; rd, kwargs...)
    factor_risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, Nothing}, args...; kwargs...)

Decompose a fold's risk over its factors.

The fold-taking method of [`factor_risk_contribution`](@ref), and the twin of the [`risk_contribution`](@ref) method above. It resolves the fold's target weights, its asset returns and its fee the same way, and it builds the `rd` the loadings are fitted from out of the fold itself: the fold's asset returns beside the factor block [`reconstruct_rd`](@ref) carried through. A caller who wants other loadings passes its own `rd`, or a precomputed [`Regression`](@ref) as `re`.

The first-order caveat of the [`risk_contribution`](@ref) method above holds here unchanged, and a fold that carries no Held Weights record raises for the same reason.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to decompose, or a vector of them.
  - `pred`: Single-fold prediction result.
  - `fees`: Fees that take precedence over the result's own.
  - `args...`: Additional arguments (ignored by the refusing method).

# Keyword Arguments

  - `rd::ReturnsResult`: Returns result the loadings are fitted from. Defaults to the fold's own asset returns and factor block.

# Validation

  - The fold carries a [`HeldWeightsResult`](@ref), else an `ArgumentError` naming the two switches.

# Returns

  - `Vector`: Risk contributions for each factor, with the last element being the idiosyncratic (off-factor) contribution.

# Related

  - [`factor_risk_contribution`](@ref)
  - [`risk_contribution`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`PredictionResult`](@ref)
"""
function factor_risk_contribution(r::BaseRM_VecBaseRM,
                                  pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult},
                                  fees::Option{<:Fees} = nothing;
                                  rd::ReturnsResult = ReturnsResult(; nx = pred.rd.nx,
                                                                    X = pred.hw.X,
                                                                    nf = pred.rd.nf,
                                                                    F = pred.rd.F),
                                  kwargs...)
    hw = pred.hw
    return factor_risk_contribution(r, pred.res.w, hw.X,
                                    amortise_fees(extract_fees(pred.res, fees),
                                                  size(hw.X, 1)); rd = rd, kwargs...)
end
function factor_risk_contribution(::BaseRM_VecBaseRM,
                                  ::PredictionResult{<:Any, <:Any, Nothing}, args...;
                                  kwargs...)
    return throw(ArgumentError("`factor_risk_contribution(r, pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`.\nSet one of them so the fold records its asset returns, or call `factor_risk_contribution(r, pred.res.w, rd.X, fees; rd = rd)` with the returns you fitted on."))
end
"""
    mapreduce_RetMtx(rd, sym = :X)

Concatenate return matrices from a vector of `PredictionReturnsResult` objects.

Internal helper that vertically concatenates the field `sym` across all elements of `rd`. Handles both single-asset (vector) and multi-asset (vector of vectors) return data.

# Arguments

  - `rd`: Vector of [`PredictionReturnsResult`](@ref) objects.
  - `sym`: Symbol of the field to extract (default `:X`).

# Returns

  - Concatenated return matrix or vector of vectors.
"""
function mapreduce_RetMtx(rd::AbstractVector{<:PredictionReturnsResult{<:Any, <:VecNum}},
                          sym = :X)
    return mapreduce(x -> getproperty(x, sym), vcat, rd)
end
function mapreduce_RetMtx(rd::AbstractVector{<:PredictionReturnsResult{<:Any, <:VecVecNum}},
                          sym = :X)
    N = length(getproperty(rd[1], sym))
    X = [eltype(getproperty(rd[1], sym)[1])[] for _ in 1:N]
    for i in 1:N
        X[i] = mapreduce(x -> getproperty(x, sym)[i], vcat, rd)
    end
    return X
end
"""
$(DocStringExtensions.TYPEDEF)

Stores predictions from multiple cross-validation folds as a single combined result.
Concatenates the test-period returns from all folds into an aggregated
[`PredictionReturnsResult`](@ref).

Per-observation quantities (`X`, `F`, `B`, `ts`, `iv`) stack across folds. `ivpa` is per-asset, not per-observation, and each fold's [`reconstruct_rd`](@ref) has already collapsed it to one value per synthetic asset using *that fold's* weights, so it cannot stack — it is **reduced to the last fold's value**. This matches [`predict_realised_vols`](@ref), which reads the last row of the stacked `iv`: the premium divisor is paired with the implied volatility it divides.

A feature matrix is **not** among them. It is not carried through the folds at all — [`rebuild_returns_result`](@ref) recomputes the outer collapse from the original `rd.Z`, reaching each fold through `pred[f].res.w` and `pred[f].rd.ts`, so `pred` is what this result has to retain for it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MultiPeriodPredictionResult(;
        pred::VecPredRes,
        id::Any = nothing
    ) -> MultiPeriodPredictionResult

Keywords correspond to the struct's fields. `pred` is required: the constructor stacks the folds' returns data into `mrd`, and the stack of no folds has no columns, no names, and no clock.

## Validation

  - `!isempty(pred)`.

# Related

  - [`PredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`sort_by_measure`](@ref)
  - [`PredictionReturnsResult`](@ref)
"""
@concrete struct MultiPeriodPredictionResult <: AbstractPredictionResult
    """
    $(field_dict[:pred])
    """
    pred
    """
    $(field_dict[:mrd])
    """
    mrd
    """
    $(field_dict[:id_pred])
    """
    id
    function MultiPeriodPredictionResult(pred::VecPredRes, id::Any)
        @argcheck(!isempty(pred), IsEmptyError("pred cannot be empty"))
        rd = getfield.(pred, :rd)
        nx = rd[1].nx
        X = mapreduce_RetMtx(rd)
        nf = rd[1].nf
        F = isnothing(rd[1].F) ? nothing : mapreduce(x -> getproperty(x, :F), vcat, rd)
        nb = rd[1].nb
        B = isnothing(rd[1].B) ? nothing : mapreduce_RetMtx(rd, :B)
        ts = isnothing(rd[1].ts) ? nothing : mapreduce(x -> getproperty(x, :ts), vcat, rd)
        iv = isnothing(rd[1].iv) ? nothing : mapreduce(x -> getproperty(x, :iv), vcat, rd)
        # Per-asset, so it cannot stack; reduced to the last fold to pair with the last
        # row of `iv`, which is what `predict_realised_vols` divides by it.
        ivpa = rd[end].ivpa
        mrd = PredictionReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nb, B = B,
                                      ts = ts, iv = iv, ivpa = ivpa)
        return new{typeof(pred), typeof(mrd), typeof(id)}(pred, mrd, id)
    end
end
function MultiPeriodPredictionResult(; pred::VecPredRes,
                                     id::Any = nothing)::MultiPeriodPredictionResult
    return MultiPeriodPredictionResult(pred, id)
end
"""
    VecMPredRes = AbstractVector{<:MultiPeriodPredictionResult}

Alias for a vector of multi-period prediction results.

Represents a collection of [`MultiPeriodPredictionResult`](@ref) objects.

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`PredRes_MultiPredRes`](@ref)
"""
const VecMPredRes = AbstractVector{<:MultiPeriodPredictionResult}
# Virtual properties `:res` and `:rd` broadcast over the inner `pred` vector, collecting
# per-fold results and relative drawdowns (see [`@forward_properties`](@ref)).
@forward_properties MultiPeriodPredictionResult begin
    compute(res, pred.res; broadcast)
    compute(rd, pred.rd; broadcast)
end
function expected_risk(r::BaseRM_VecBaseRM, mpred::MultiPeriodPredictionResult; kwargs...)
    X = mpred.mrd.X
    return expected_risk_from_returns(r, X; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Roll a risk measure over the return series a whole path formed.

`mpred.mrd.X` concatenates the folds of the path into one series, so a window can straddle a rebalance and read observations from two folds. That is the realised history: the fund held one set of weights before the rebalance and another after it, and the window sees both.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `mpred::MultiPeriodPredictionResult`: Multi-period prediction result.
  - `window::Integer`: Size of the rolling window (number of periods).

# Returns

  - `risks::VecNum`: Expected risk values for each rolling window.

# Related

  - [`rolling_window_measure`](@ref)
  - [`expected_risk`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, mpred::MultiPeriodPredictionResult,
                                window::Integer; kwargs...)
    return rolling_window_measure(r, mpred.mrd.X, window; kwargs...)
end
"""
    PredRes_MultiPredRes = Union{<:PredictionResult, <:MultiPeriodPredictionResult}

Alias for a single-fold or multi-period prediction result.

Matches either a [`PredictionResult`](@ref) or a [`MultiPeriodPredictionResult`](@ref).

# Related

  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`VecPredRes_MultiPredRes`](@ref)
"""
const PredRes_MultiPredRes = Union{<:PredictionResult, <:MultiPeriodPredictionResult}
"""
    VecPredRes_MultiPredRes = AbstractVector{<:PredRes_MultiPredRes}

Alias for a vector of single-fold or multi-period prediction results.

Represents a collection of [`PredRes_MultiPredRes`](@ref) elements.

# Related

  - [`PredRes_MultiPredRes`](@ref)
"""
const VecPredRes_MultiPredRes = AbstractVector{<:PredRes_MultiPredRes}
"""
$(DocStringExtensions.TYPEDEF)

Stores a collection of multi-period prediction results produced by a population-based
cross-validation scheme (e.g. [`MultipleRandomised`](@ref)). Each element of `pred`
represents one random asset-subset path.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PopulationPredictionResult(;
        pred::VecPredRes_MultiPredRes = Vector{PredRes_MultiPredRes}(undef, 0)
    ) -> PopulationPredictionResult

Keywords correspond to the struct's fields. An empty `pred` is admitted: a population from which every path was dropped is a valid, if empty, answer.

# Related

  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`sort_by_measure`](@ref)
  - [`MultipleRandomised`](@ref)
"""
@concrete struct PopulationPredictionResult <: AbstractPredictionResult
    """
    $(field_dict[:pred])
    """
    pred
    function PopulationPredictionResult(pred::VecPredRes_MultiPredRes)
        return new{typeof(pred)}(pred)
    end
end
function PopulationPredictionResult(;
                                    pred::VecPredRes_MultiPredRes = Vector{PredRes_MultiPredRes}(undef,
                                                                                                 0))::PopulationPredictionResult
    return PopulationPredictionResult(pred)
end
function expected_risk(r::BaseRM_VecBaseRM, preds::VecMPredRes; kwargs...)
    return [expected_risk(r, pred; kwargs...) for pred in preds]
end
function expected_risk(r::BaseRM_VecBaseRM, ppred::PopulationPredictionResult; kwargs...)
    return expected_risk(r, ppred.pred; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Roll a risk measure over each path of a vector of multi-period prediction results.

Maps the multi-period method over `preds`, so each path is rolled on its own series. The paths of a combinatorial scheme cover the same calendar, so their windows are comparable across the vector.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `preds::VecMPredRes`: Vector of multi-period prediction results.
  - `window::Integer`: Size of the rolling window (number of periods).

# Returns

  - `risks::Vector{<:VecNum}`: Rolling risk values, one vector per path.

# Related

  - [`rolling_window_measure`](@ref)
  - [`expected_risk`](@ref)
  - [`VecMPredRes`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, preds::VecMPredRes, window::Integer;
                                kwargs...)
    return [rolling_window_measure(r, pred, window; kwargs...) for pred in preds]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Roll a risk measure over every path of a population prediction result.

Delegates to the vector method on `ppred.pred`, which is the route [`expected_risk`](@ref) takes on the same type.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to evaluate, or a vector of them.
  - `ppred::PopulationPredictionResult`: Population prediction result.
  - `window::Integer`: Size of the rolling window (number of periods).

# Returns

  - `risks::Vector{<:VecNum}`: Rolling risk values, one vector per path.

# Related

  - [`rolling_window_measure`](@ref)
  - [`expected_risk`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function rolling_window_measure(r::BaseRM_VecBaseRM, ppred::PopulationPredictionResult,
                                window::Integer; kwargs...)
    return rolling_window_measure(r, ppred.pred, window; kwargs...)
end
"""
    sort_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM; kwargs...)

Sort the successful paths in a [`PopulationPredictionResult`](@ref) by their expected
risk under `r`. Paths where any fold returned a non-success retcode are excluded.

# Arguments

  - `ppred::PopulationPredictionResult`: Population prediction to sort.
  - `r::BaseRM_VecBaseRM`: Risk measure used for ranking, or a vector of them. A vector is scalarised by `kwargs.sca`, defaulting to [`SumScalariser`](@ref).
  - `kwargs...`: Keyword arguments forwarded to `expected_risk`.

# Returns

  - `Vector{MultiPeriodPredictionResult}`: Sorted vector of successful path predictions.

## A mixed vector is refused here, and accepted by its sibling

The ranking direction comes from [`bigger_is_better`](@ref), which **throws** on a vector whose elements disagree on polarity, because the flag decides which tail of the ranking is best and neither answer would be right. [`quantile_by_measure`](@ref) takes an explicit `sign` instead, so it admits a mixed vector.

# Related

  - [`quantile_by_measure`](@ref)
  - [`bigger_is_better`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`expected_risk`](@ref)
"""
function sort_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM; kwargs...)
    pred = filter(x -> all(y -> isa(y.res.retcode, OptimisationSuccess), x.pred),
                  ppred.pred)
    return sort(pred; by = x -> expected_risk(r, x; kwargs...), rev = bigger_is_better(r))
end
"""
    quantile_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM, q::Real;
                        r_kwargs::NamedTuple = (;), q_kwargs::NamedTuple = (;),
                        sign::Integer = 1)

Select the successful path in `ppred` whose expected risk under `r` is closest to the `q`-th quantile of the risk distribution across all successful paths.

# Arguments

  - `ppred::PopulationPredictionResult`: Population prediction result.
  - `r::BaseRM_VecBaseRM`: Risk measure for computing path risks, or a vector of them. A vector is scalarised by `r_kwargs.sca`, defaulting to [`SumScalariser`](@ref).
  - `q::Real`: Quantile level in `[0, 1]`.
  - `r_kwargs::NamedTuple = (;)`: Keyword arguments forwarded to `expected_risk`.
  - `q_kwargs::NamedTuple = (;)`: Keyword arguments forwarded to `Statistics.quantile`.
  - `sign::Integer = 1`: Orientation of the risk scale. Use `1` when a larger risk is worse, `-1` when it is better. This is what lets a mixed vector through, see below.

# Returns

  - [`MultiPeriodPredictionResult`](@ref): The path closest to the `q`-th quantile.

## A mixed vector is accepted here, and refused by its sibling

[`sort_by_measure`](@ref) calls [`bigger_is_better`](@ref) for its `rev` flag, so it **throws** on a vector whose elements disagree on polarity. This function takes an explicit `sign` instead, so the caller has already supplied the one thing `bigger_is_better` cannot infer, and a mixed vector is admitted.

# Related

  - [`sort_by_measure`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`expected_risk`](@ref)
"""
function quantile_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM,
                             q::Real; r_kwargs::NamedTuple = (;),
                             q_kwargs::NamedTuple = (;), sign::Integer = 1)
    pred = filter(x -> all(y -> isa(y.res.retcode, OptimisationSuccess), x.pred),
                  ppred.pred)
    rks = [sign*expected_risk(r, p; r_kwargs...) for p in pred]
    rkq = Statistics.quantile(rks, q; q_kwargs...)
    rk_min = typemax(eltype(rks))
    idx = 1
    for (i, rk) in enumerate(rks)
        rkd = abs(rk - rkq)
        if rkd < rk_min
            rk_min = rkd
            idx = i
        end
    end
    return pred[idx]
    # sorted_predictions = sort_by_measure(ppred, r; kwargs...)
    # idx = max(1, round(Int, Statistics.quantile(1:length(sorted_predictions), q)))
    # return sorted_predictions[idx]
end
"""
    collapse_benchmark(B::Nothing, w::VecNum_VecVecNum, hw)
    collapse_benchmark(B::VecNum, w::VecNum, hw)
    collapse_benchmark(B::VecNum, w::VecVecNum, hw)
    collapse_benchmark(B::MatNum, w::VecNum, hw::Nothing)
    collapse_benchmark(B::MatNum, w::VecVecNum, hw::Nothing)
    collapse_benchmark(B::MatNum, w::VecNum, hw::HeldWeightsResult)
    collapse_benchmark(B::MatNum, w::VecVecNum, hw::HeldWeightsResult)

Collapse a fold's benchmark asset returns into a benchmark return series.

A benchmark that is already a series passes through. A benchmark matrix is contracted with the fold's own weights, and the method is chosen by the pair `(B, w)`, so nothing is tested at run time.

The fold's Held Weights record picks the reading. Without one the matrix collapses against the target weights, which is the library's original behaviour and what a fold that ran no drift keeps. With one it collapses row by row against the weight path, so the benchmark follows the same convention the portfolio series follows and a caller comparing the two — a tracking error, for instance — compares two series scored the same way.

# Algorithm

 1. On `nothing`, give `nothing`.
 2. On a benchmark series, give it back, repeated once per member under a population.
 3. On a matrix with no record, give `B * w`, once per member under a population.
 4. On a matrix with a record, give `vec(sum(B ⊙ U; dims = 2))` for the fold's weight path `U`, once per member under a population.

# Arguments

  - `B`: Benchmark returns of the fold: `nothing`, a series, or an observations × assets matrix.
  - `w`: Target weights of the fold, or a population of them.
  - `hw`: Held Weights record of the fold, or `nothing`.

# Returns

  - The benchmark return series, or a vector of them under a population, or `nothing`.

# Related

  - [`reconstruct_rd`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`weight_path`](@ref)
  - [`PredictionReturnsResult`](@ref)
"""
function collapse_benchmark(::Nothing, ::VecNum_VecVecNum, ::Any)
    return nothing
end
function collapse_benchmark(B::VecNum, ::VecNum, ::Any)
    return B
end
function collapse_benchmark(B::VecNum, w::VecVecNum, ::Any)
    return fill(B, length(w))
end
function collapse_benchmark(B::MatNum, w::VecNum, ::Nothing)
    return B * w
end
function collapse_benchmark(B::MatNum, w::VecVecNum, ::Nothing)
    return [B * wi for wi in w]
end
function collapse_benchmark(B::MatNum, w::VecNum, hw::HeldWeightsResult)
    return vec(sum(B ⊙ weight_path(hw, w); dims = 2))
end
function collapse_benchmark(B::MatNum, w::VecVecNum, hw::HeldWeightsResult)
    return [vec(sum(B ⊙ U; dims = 2)) for U in weight_path(hw, w)]
end
"""
    reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult, X, hw = nothing)

Reconstruct a `PredictionReturnsResult` from an optimisation result and returns data.

Computes the benchmark returns, the implied volatilities and the implied volatility risk premium adjustment from the optimisation result weights and the original returns data.

The benchmark collapse follows the fold's weight-drift setting whenever `rd.B` is a matrix. A fold that carries a [`HeldWeightsResult`](@ref) collapses the matrix row by row against its weight path, the same convention its portfolio series is scored under; a fold that carries none collapses it against the target weights, as before. [`collapse_benchmark`](@ref) is the verb, and it reads the pair by dispatch.

## No feature matrix

The fold does not collapse `rd.Z`. Only one weight vector is in scope here, which is not enough to collapse a *square* feature matrix — the second contraction needs every synthetic asset's weights at once — so a fold-side collapse could only serve one of the two shapes, and the two paths into the outer problem would disagree on what a square carrier means. [`rebuild_returns_result`](@ref) instead recomputes the collapse for the whole synthetic universe at once, from the original `rd.Z` and the fold weights it reaches through `pred[f].res.w`.

# Arguments

  - `res::NonFiniteAllocationOptimisationResult`: Fitted optimisation result.
  - `rd::ReturnsResult`: Original returns data.
  - `X`: Portfolio returns (vector or vector of vectors).
  - `hw`: Held Weights record of the fold, or `nothing`.

# Returns

  - [`PredictionReturnsResult`](@ref) with updated benchmark returns, implied volatilities and implied volatility risk premium adjustment.

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`PredictionReturnsResult`](@ref)
  - [`rebuild_returns_result`](@ref)
  - [`collapse_benchmark`](@ref)
  - [`HeldWeightsResult`](@ref)
"""
function reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        X::VecNum, hw::Option{<:HeldWeightsResult} = nothing)
    B = collapse_benchmark(rd.B, res.w, hw)
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        # `iv` and `ivpa` are intensive, so they collapse as convex combinations.
        w = synthetic_asset_weights(res.w)
        if iv_flag
            iv = iv * w
        end
        if ivpa_flag
            ivpa = LinearAlgebra.dot(rd.ivpa, w)
        end
    end
    return PredictionReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, nb = rd.nb,
                                   B = B, ts = rd.ts, iv = iv, ivpa = ivpa)
end
function reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        X::VecVecNum, hw::Option{<:HeldWeightsResult} = nothing)
    nb = rd.nb
    B = collapse_benchmark(rd.B, res.w, hw)
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        # `iv` and `ivpa` are intensive, so they collapse as convex combinations.
        w = [synthetic_asset_weights(wi) for wi in res.w]
        if iv_flag
            iv = [iv * w for w in w]
        end
        if ivpa_flag
            ivpa = [LinearAlgebra.dot(ivpa, wi) for wi in w]
        end
    end
    if isa(ivpa, Number)
        ivpa = range(; start = ivpa, stop = ivpa, length = length(res.w))
    end
    return PredictionReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, nb = nb,
                                   B = B, ts = rd.ts, iv = iv, ivpa = ivpa)
end
"""
    predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)
    predict(res, rd, test_idx, cols = :)
    predict(res, rd, test_idxs::VecVecInt, cols = :)

Apply an optimisation result `res` to returns data `rd` to produce a
[`PredictionResult`](@ref) or a vector of prediction results.

When `test_idx` is provided, only the rows (observations) indexed by `test_idx` (and
optionally columns `cols`) of `rd` are used for the prediction.

The `test_idx` method settles the fee's amortisation horizon against the fold's own length before
charging it: `amortise_fees(extract_fees(res, nothing), size(rdi.X, 1))`. A stated `horizon` on the
fee's `fa` overrides this and reaches unchanged. The whole-sample method, `predict(res, rd)`, gets
no such rebuild, so it charges the whole one-off cost unless the fee already carries a stated
`horizon`.

# Arguments

  - `res::NonFiniteAllocationOptimisationResult`: Fitted optimisation result.
  - `rd::ReturnsResult`: Returns data for the prediction period.
  - `test_idx`: Observation index or vector of observation indices for the test fold.
  - `cols`: Column selector. Defaults to `:` (all assets).

# Returns

  - [`PredictionResult`](@ref) or vector of [`PredictionResult`](@ref).

# Related

  - [`fit_predict`](@ref)
  - [`fit_and_predict`](@ref)
  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`amortise_fees`](@ref)
  - [`extract_fees`](@ref)
"""
function StatsAPI.predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult;
                          wd::Option{<:AbstractWeightDrift} = nothing,
                          hwd::Option{<:AbstractWeightDrift} = wd,
                          store_weight_path::Bool = false)
    X = calc_net_returns(res, rd.X, nothing, wd, rd.ts)
    (hw, ruined) = held_weights_result(hwd, res.w, rd.X, store_weight_path, rd.ts)
    warn_ruined_members(wd, ruined, length(res.w))
    res = mark_ruined_members(res, ruined)
    rd = reconstruct_rd(res, rd, X, hw)
    return PredictionResult(; res = res, rd = rd, hw = hw)
end
"""
    fit_predict(opt::OptE_Opt, rd::ReturnsResult)

Fit optimisation estimator `opt` on returns data `rd` and immediately produce a
[`PredictionResult`](@ref) for the same data.

# Arguments

  - `opt`: Optimisation estimator or result.
  - `rd::ReturnsResult`: Returns data.

# Returns

  - [`PredictionResult`](@ref).

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`PredictionResult`](@ref)
  - [`fit_and_predict`](@ref)
"""
function fit_predict(opt::OptE_Opt, rd::ReturnsResult)
    res = optimise(opt, rd)
    return StatsAPI.predict(res, rd)
end
function StatsAPI.predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                          test_idx::VecInt, cols = :;
                          wd::Option{<:AbstractWeightDrift} = nothing,
                          hwd::Option{<:AbstractWeightDrift} = wd,
                          store_weight_path::Bool = false)
    rdi = port_opt_view(rd, test_idx, cols)
    fees = amortise_fees(extract_fees(res, nothing), size(rdi.X, 1))
    obs = drift_observations(rdi.ts, test_idx)
    X = calc_net_returns(res, rdi.X, fees, wd, obs)
    (hw, ruined) = held_weights_result(hwd, res.w, rdi.X, store_weight_path, obs)
    warn_ruined_members(wd, ruined, length(res.w))
    res = mark_ruined_members(res, ruined)
    rdi = reconstruct_rd(res, rdi, X, hw)
    return PredictionResult(; res = res, rd = rdi, hw = hw)
end
function StatsAPI.predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                          test_idxs::VecVecInt, cols = :; kwargs...)
    return [StatsAPI.predict(res, rd, test_idx, cols; kwargs...) for test_idx in test_idxs]
end
"""
    fit_and_predict(opt, rd::ReturnsResult, cv::NonSeqCVER; cols, ex, id) -> MultiPeriodPredictionResult
    fit_and_predict(opt, rd::ReturnsResult, cv::CombCVER; cols, ex) -> PopulationPredictionResult
    fit_and_predict(opt, rd::ReturnsResult; train_idx, test_idx, cols) -> PredictionResult
    fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult; test_idx, cols) -> PredictionResult

Fit an optimisation estimator on training data and predict on test data using cross-validation.

The three-argument method (`opt`, `rd`, `cv`) performs full cross-validated prediction over all folds of `cv`.
The two-argument methods operate on a single pre-defined train/test split or on a pre-existing result.

# Arguments

  - `opt`: Optimisation estimator or an existing optimisation result.
  - `rd::ReturnsResult`: Returns data.
  - `cv::NonSeqCVER`: Non-sequential cross-validation estimator (e.g. [`KFold`](@ref) or [`CombinatorialCrossValidation`](@ref)).
  - `cv::CombCVER`: Combinatorial cross-validation estimator or result ([`CombinatorialCrossValidation`](@ref)).
  - `train_idx::VecInt`: Training indices.
  - `test_idx`: Test indices (vector or vector of vectors).
  - `cols`: Column selector (default `:` for all assets).

# Returns

  - [`MultiPeriodPredictionResult`](@ref), [`PopulationPredictionResult`](@ref), or [`PredictionResult`](@ref).

# Details

  - A combinatorial `cv` takes its own method, because its folds recombine into several paths rather than one. That method regroups the fold predictions by path and returns one [`MultiPeriodPredictionResult`](@ref) per path, wrapped in a [`PopulationPredictionResult`](@ref).

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`optimise`](@ref)
  - [`KFold`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult;
                         test_idx::VecInt_VecVecInt, cols = :,
                         wd::Option{<:AbstractWeightDrift} = nothing,
                         hwd::Option{<:AbstractWeightDrift} = wd,
                         store_weight_path::Bool = false, kwargs...)
    return StatsAPI.predict(res, rd, test_idx, cols; wd = wd, hwd = hwd,
                            store_weight_path = store_weight_path)
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult;
                         train_idx::VecInt, test_idx::VecInt_VecVecInt, cols = :,
                         wd::Option{<:AbstractWeightDrift} = nothing,
                         hwd::Option{<:AbstractWeightDrift} = wd,
                         store_weight_path::Bool = false)
    rd_train = port_opt_view(rd, train_idx, cols)
    if !isa(cols, Colon)
        opt = port_opt_view(opt, cols, rd.X)
    end
    #! Add ability to do callbacks
    res = optimise(opt, rd_train)
    return StatsAPI.predict(res, rd, test_idx, cols; wd = wd, hwd = hwd,
                            store_weight_path = store_weight_path)
end
"""
    sort_predictions!(res::VecVecInt, predictions::VecPredRes) -> VecPredRes
    sort_predictions!(res::CrossValidationResult, predictions::VecPredRes) -> VecPredRes

Sort prediction results to match the order of test indices.

Reorders `predictions` so that they align with the original time ordering of `test_idx`. The key is the first observation of each fold, so the folds come back in the order the timeline visits them.

# Arguments

  - `res`:

      + `::VecVecInt`: Vector of test index vectors.
      + `::CrossValidationResult`: Cross validation result object, uses the test indices stored in `res.test_idx`.

  - `predictions`: Vector of prediction results, one per fold, in split order.

# Validation

  - Every element of `test_idx` holds unique indices.

# Returns

  - Sorted predictions vector.

# Details

  - [`CombinatorialCrossValidationResult`](@ref) has its own method with a different shape and a different job. Its folds do not form one timeline, so that method takes a vector of per-split vectors and regroups them by path into [`MultiPeriodPredictionResult`](@ref)s rather than sorting one timeline.

# Related

  - [`fit_and_predict`](@ref)
  - [`path_fit_and_predict`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
"""
function sort_predictions!(test_idx::VecVecInt, predictions::VecPredRes)
    @argcheck(all(x -> allunique(x), test_idx), "Test indices must be unique.")
    idx = sortperm(test_idx; by = x -> x[1])
    return predictions[idx]
end
function sort_predictions!(res::CrossValidationResult, predictions::VecPredRes)
    return sort_predictions!(res.test_idx, predictions)
end
"""
    cv_sequential_info()

Build the informational message emitted when a cross-validation run runs its folds
sequentially. [`run_folds`](@ref) is the only site that emits it, and it is the sequential
loop, so the message states the two facts that sent the run there rather than quoting a
value back.

The two facts are the conjunction [`fold_loop`](@ref) computes. The fold enumeration of the
scheme is a timeline ([`folds_are_time_ordered`](@ref)), and the estimator needs the
previous fold's weights ([`needs_previous_weights`](@ref)). Either one alone leaves the
folds independent. Time dependence is neither of them: a [`TimeDependent`](@ref) schedule is
known for every fold before the loop starts, so [`fold_loop`](@ref) resolves it in parallel.

# Returns

  - `String`: The message.

# Related

  - [`run_folds`](@ref)
  - [`fold_loop`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function cv_sequential_info()
    return "Running cross-validation sequentially because the folds of the cross-validation scheme are a timeline (folds_are_time_ordered(cv) == true) and the optimiser must use the previous optimisation's weights (needs_previous_weights(opt) == true). The second fact is because somewhere within the optimisation estimator is contained at least one of the following:\n\t- Turnover and/or TurnoverEstimator,\n\t- WeightsTracking,\n\t- TurnoverRiskMeasure,\n\t- custom constraints which use asset weights,\n\t- custom objective penalties which use asset weights,\n\t- a time-dependent constraint whose entries need previous weights (e.g. a PreviousWeightsFunction).\nTo enable parallel processing please either mark the weights as fixed or remove the offending component(s). Time-dependent constraints alone do not force sequential processing."
end
"""
    parallel_folds(fit_fold, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult)

Run `n` cross-validation folds in parallel, filling `predictions[i] = fit_fold(i)` for `i in 1:n`
over executor `ex`. `ElT` is the per-fold result element type (a single [`PredictionResult`](@ref)
for time-ordered schemes, a `Vector{PredictionResult}` for the multi-path combinatorial scheme).

This is the sibling of [`run_folds`](@ref), and the two divide the work by name. A fold here
takes no previous fold, so `fit_fold` takes the fold index alone. [`fold_loop`](@ref)
decides which of the two runs, and neither one re-decides.

`ElT` is a *positional* `::Type{ElT}` argument, not a keyword, so a method always
specialises on it and `Vector{ElT}(undef, n)` stays a compile-time construction. As a
keyword its value only survives constant propagation, which one forwarding hop is enough
to lose — see the amendment of ADR 0067.

# Related

  - [`run_folds`](@ref)
  - [`fold_loop`](@ref)
  - [`fit_and_predict`](@ref)
"""
function parallel_folds(fit_fold, n::Integer, ex::FLoops.Transducers.Executor,
                        ::Type{ElT} = PredictionResult) where {ElT}
    predictions = Vector{ElT}(undef, n)
    FLoops.@floop ex for i in 1:n
        predictions[i] = fit_fold(i)
    end
    return predictions
end
"""
    run_folds(fit_fold, n::Integer, ::Type{ElT} = PredictionResult)

Run `n` cross-validation folds in order, filling
`predictions[i] = fit_fold(i, predictions[i - 1])` for `i in 1:n`, and emit
[`cv_sequential_info`](@ref). Fold 1 takes `nothing`, because it has no fold behind it. The
caller uses the previous fold to thread its weights into fold `i`. `ElT` is the per-fold
result element type.

This is the sequential loop, and it does that one job. [`fold_loop`](@ref) is the only site
that calls it, and it calls it only when the folds are a timeline *and* the estimator needs
the previous fold's weights. The loop therefore neither re-decides nor takes an executor:
its sibling [`parallel_folds`](@ref) owns the other case.

`ElT` is a *positional* `::Type{ElT}` argument for the reason given in
[`parallel_folds`](@ref).

# Related

  - [`parallel_folds`](@ref)
  - [`fold_loop`](@ref)
  - [`cv_sequential_info`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`fit_and_predict`](@ref)
"""
function run_folds(fit_fold, n::Integer, ::Type{ElT} = PredictionResult) where {ElT}
    @info(cv_sequential_info())
    predictions = Vector{ElT}(undef, n)
    for i in 1:n
        predictions[i] = fit_fold(i, i > 1 ? predictions[i - 1] : nothing)
    end
    return predictions
end
"""
    assert_unshuffled_folds(cv, train_idx)

Assert that the cross-validation scheme `cv` enumerates unshuffled folds.

Two checks, applied to every scheme:

 1. `cv` must not declare a set `shuffle` field. The check is by `hasfield`, so it holds
    for a user-defined scheme too — no scheme in this package has such a field.
 2. Every fold's training indices must increase strictly. A scheme may leave gaps (purging,
    embargoing and the combinatorial splits all do), but it must never reorder rows.

A shuffled fold breaks the timeline that the fold loop, the [`TimeDependentContext`](@ref)
schedules and the rolling transforms all read the fold's rows in.

# Related

  - [`fold_loop`](@ref)
  - [`fit_and_predict`](@ref)
  - [`cross_val_predict`](@ref)
"""
function assert_unshuffled_folds(cv, train_idx)
    @argcheck(!(hasfield(typeof(cv), :shuffle) && cv.shuffle),
              "Cross validation estimator must not be shuffled.")
    @argcheck(all(x -> all(>(zero(eltype(x))), diff(x)), train_idx),
              "Cross validation estimator must not be shuffled.")
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

One fold of a cross-validation scheme, as [`fold_loop`](@ref) hands it to its callback.

The record is the fold loop's whole hand-off. `est` and `rd` are already resolved: the
asset view is taken, every [`TimeDependent`](@ref) schedule is swapped for its fold-`i`
value, and the previous fold's weights are threaded in. `train` and `test` are this fold's
own windows, so a callback never indexes `train_idx`/`test_idx` itself.

The type is immutable and every field is concretely typed at the construction site, so the
record costs nothing at run time. [`fold_loop`](@ref) is the only site that builds one,
which is why there is no keyword constructor.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`fold_loop`](@ref)
  - [`TimeDependentContext`](@ref)
  - [`fit_and_predict`](@ref)
"""
struct Fold{T1, T2, T3, T4, T5, T6}
    """
    Index of the fold within the scheme's `split` enumeration (1-based).
    """
    i::T1
    """
    Number of folds in the enumeration.
    """
    n::T2
    """
    The fold's resolved estimator: asset-viewed, schedule-swapped, weights-threaded.
    """
    est::T3
    """
    The fold's (possibly asset-viewed) input data.
    """
    rd::T4
    """
    The fold's training indices.
    """
    train::T5
    """
    The fold's test indices.
    """
    test::T6
end
"""
    folds_are_time_ordered(cv)

Return `true` if the fold enumeration of a cross-validation scheme is a timeline.

This is one half of the conjunction [`fold_loop`](@ref) computes, so a scheme states for
itself whether its folds carry history. A walk-forward, a multiple-randomised path, and any
scheme with no more specific method enumerate their folds in time order. Fold `i` has fold
`i - 1` behind it, so the loop may run the folds in order and thread the previous fold's
weights. [`needs_previous_weights`](@ref) is the other half, and it decides whether the loop
does so.

A [`NonSeqCVER`](@ref) scheme answers `false`. A k-fold and a combinatorial enumeration are
not timelines: each fold is independent of the others, so no fold has a previous fold, and
the loop runs them in parallel. A `KFold` training window holds rows that follow its test
window, so a quantity measured against another fold's weights is not a backtest reading.

The method is per type and takes the scheme itself, so inference reads the answer from the
type of `cv` and never needs the value. `folds_are_time_ordered(::Any)` answers `nothing`
too, which is what [`fold_loop`](@ref) receives from a call site that holds no scheme.

# Related

  - [`fold_loop`](@ref)
  - [`NonSeqCVER`](@ref)
  - [`needs_previous_weights`](@ref)
  - [`cross_val_predict`](@ref)
"""
folds_are_time_ordered(::Any) = true
folds_are_time_ordered(::NonSeqCVER) = false

"""
    fold_evaluation(cv)

Read the evaluation switches of a cross-validation scheme, in one named triple.

Every scheme entry point reads the same three settings before it runs its folds, and each scheme states them for itself through a method of its own. A scheme that carries none of them, and a call site that holds a split result rather than the scheme that made it, reach the fallback and get today's behaviour: no drift, no drifted previous weights, and no stored weight path.

The method is per type and takes the scheme itself, so inference reads the answer from the type of `cv`, exactly as [`folds_are_time_ordered`](@ref) does.

# Returns

  - `(; wd, pws, store_weight_path)`: The Weight Drift, the Previous-Weights Source, and the flag that stores a fold's weight path.

# Related

  - [`folds_are_time_ordered`](@ref)
  - [`held_weights_drift`](@ref)
  - [`AbstractWeightDrift`](@ref)
  - [`AbstractPreviousWeightsSource`](@ref)
  - [`fold_loop`](@ref)
"""
function fold_evaluation(::Any)
    return (; wd = nothing, pws = nothing, store_weight_path = false)
end
"""
    fold_loop(fit_fold, est, n::Integer, ex::FLoops.Transducers.Executor,
              ::Type{ElT} = PredictionResult; rd, train_idx, test_idx,
              path_id = nothing, cv = nothing, fold_view = nothing)

Run the `n` folds of a cross-validation scheme over `est`, and resolve each fold's estimator
before the callback sees it.

This is the fold loop of the package. Every cross-validation entry point goes through
it: the optimiser-level schemes and the [`Pipeline`](@ref) ones alike. For fold `i` it
does four steps.

 1. It takes the fold's view of `(est, rd)` through `fold_view`. An asset-resampling scheme
    gives one; the other schemes do not.
 2. It swaps every [`TimeDependent`](@ref) schedule for its fold-`i` value against a
    [`TimeDependentContext`](@ref), if `est` [`is_time_dependent`](@ref). The swap runs
    first, so a freshly swapped-in per-fold entry also gets the weights of step 3.
 3. It threads the previous fold's weights in through [`factory`](@ref), if `est`
    [`needs_previous_weights`](@ref).
 4. It calls `fit_fold(fold)`, with `fold` a [`Fold`](@ref).

The callback takes the one [`Fold`](@ref) record, so a call site names what it reads
(`fold.est`, `fold.train`) instead of relying on the position of an argument.

[`assert_time_dependent_fold_count`](@ref) runs once, before the loop.

This is also the one site that decides how the folds run. A run is sequential only when two
facts hold at once: the fold enumeration of `cv` is a timeline
([`folds_are_time_ordered`](@ref)), *and* `est` needs the previous fold's weights
([`needs_previous_weights`](@ref)). The conjunction routes through [`run_folds`](@ref).
Every other case routes through [`parallel_folds`](@ref), because a fold with no fold behind
it, or a fold whose estimator reads no previous weights, is independent of the other folds.
Neither loop re-decides.

`cv` is the scheme, and the loop reads its two per-type predicates rather than a
value a call site computes. Both are decided by the *types* of `cv` and `est`, so inference
folds the conjunction and eliminates the arm that cannot run. A `Bool` keyword cannot do
this: its value survives only by constant propagation, which one call hop loses, and the
sequential arm is then inferred even where it can never run — see the amendments of ADR
0067. The two path-level sites enumerate an inner walk-forward and hold no scheme, so they
omit `cv`; `folds_are_time_ordered(nothing)` answers `true`.

`ElT` is the per-fold result element type: a single
[`PredictionResult`](@ref) for a time-ordered scheme, a `Vector{PredictionResult}` for the
multi-path combinatorial scheme. It is positional for the reason given in
[`parallel_folds`](@ref).

# Related

  - [`Fold`](@ref)
  - [`run_folds`](@ref)
  - [`parallel_folds`](@ref)
  - [`assert_unshuffled_folds`](@ref)
  - [`folds_are_time_ordered`](@ref)
  - [`fit_and_predict`](@ref)
  - [`cross_val_predict`](@ref)
"""
function fold_loop(fit_fold, est, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult; rd, train_idx, test_idx,
                   path_id = nothing, cv = nothing, fold_view = nothing,
                   pws = nothing) where {ElT}
    td_flag = is_time_dependent(est)
    if td_flag
        assert_time_dependent_fold_count(est, n)
    end
    prev_w_flag = needs_previous_weights(est)
    function fold(i, prev)
        w_prev = previous_weights(pws, prev)
        (esti, rdi) = isnothing(fold_view) ? (est, rd) : fold_view(i)
        # Resolve time-dependent entries first, so a freshly swapped-in per-fold entry also
        # receives the previous weights from the factory pass below.
        if td_flag
            ctx = TimeDependentContext(; i = i, n = n, rd = rdi, train_idx = train_idx,
                                       test_idx = test_idx, w_prev = w_prev,
                                       path_id = path_id)
            esti = update_time_dependent_estimator(esti, ctx)
        end
        if !isnothing(w_prev) && prev_w_flag
            esti = factory(esti, w_prev)
        end
        return fit_fold(Fold(i, n, esti, rdi, train_idx[i], test_idx[i]))
    end
    # Both halves are per-type methods over the concretely-typed `cv` and `est`, so
    # inference decides the conjunction from types alone and eliminates the arm that
    # cannot run. A `Bool` keyword would leave the `run_folds` arm inferred, and its
    # abstractly-typed `predictions[i - 1]` is a runtime dispatch. See the ADR 0067
    # amendments.
    return if folds_are_time_ordered(cv) && prev_w_flag
        run_folds(fold, n, ElT)
    else
        parallel_folds(i -> fold(i, nothing), n, ex, ElT)
    end
end
function fit_and_predict(opt::OptE_Opt_TD, rd::ReturnsResult, cv::NonSeqCVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    (; wd, pws, store_weight_path) = fold_evaluation(cv)
    hwd = held_weights_drift(wd, pws)
    predictions = fold_loop(opt, length(train_idx), ex; rd = rd, train_idx = train_idx,
                            test_idx = test_idx, cv = cv, pws = pws) do fold
        return fit_and_predict(fold.est, fold.rd; train_idx = fold.train,
                               test_idx = fold.test, cols = cols, wd = wd, hwd = hwd,
                               store_weight_path = store_weight_path)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end

export PredictionResult, MultiPeriodPredictionResult, PopulationPredictionResult,
       PredictionReturnsResult, fit, predict, fit_predict, sort_by_measure
