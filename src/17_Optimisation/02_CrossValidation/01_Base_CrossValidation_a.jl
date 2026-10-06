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

Give a split result back unchanged.

A [`CrossValidationResult`](@ref) holds folds that are already split, so an entry point that calls `split(cv, rd)` on a scheme reads the same folds when it holds a result in place of the scheme. The data argument is not read.

# Arguments

  - `res`: The split result.
  - `args...`: The data the scheme would split, which is ignored.

# Returns

  - `res`, the same object.
"""
function Base.split(res::CrossValidationResult, args...)
    return res
end
"""
    CVER = Union{<:CVE_Onl, <:CrossValidationResult}

Union of all cross-validation schemes, plain or online, and result types.

# Related

  - [`CVE_Onl`](@ref)
  - [`CrossValidationResult`](@ref)
"""
const CVER = Union{<:CVE_Onl, <:CrossValidationResult}
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

Abstract supertype for non-sequential optimisation cross-validation estimators. A
non-sequential scheme makes folds that are not a timeline, such as the folds of a k-fold or a
combinatorial split.

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

# Related

  - [`OptimisationCrossValidationEstimator`](@ref)
  - [`OptimisationCrossValidationResult`](@ref)
  - [`NonSeqCVER`](@ref)
  - [`SeqCVER`](@ref)
"""
const OptCVER = Union{<:OptimisationCrossValidationEstimator,
                      <:OptimisationCrossValidationResult}

"""
    NonSeqCVER

Union of all non-sequential cross-validation estimators and results.

[`folds_are_time_ordered`](@ref) answers `false` for every member, so the fold loop runs their folds in parallel.

# Related

  - [`NonSequentialCrossValidationEstimator`](@ref)
  - [`NonSequentialCrossValidationResult`](@ref)
  - [`SeqCVER`](@ref)
  - [`folds_are_time_ordered`](@ref)
"""
const NonSeqCVER = Union{<:NonSequentialCrossValidationEstimator,
                         <:NonSequentialCrossValidationResult}
"""
    SeqCVER

Union of all sequential cross-validation estimators and results.

# Related

  - [`SequentialCrossValidationEstimator`](@ref)
  - [`SequentialCrossValidationResult`](@ref)
  - [`NonSeqCVER`](@ref)
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

Return the positions of the assets in the Coverage Universe of a cross-validation window.

This is the sibling of [`cv_nobs`](@ref) for the asset axis. A fold that draws a subset of assets draws it from these positions, so it never draws a column that could not have been traded.

The two data-level methods differ only in how they reach the numeric matrix. Returns data holds it directly, and price data holds a `TimeArray`. The two mask-level methods take the `nothing` sentinel of an all-covered window and the mask of a window with gaps.

# Algorithm

 1. Derive the Coverage Universe of the window's asset matrix and its panel with [`coverage_mask`](@ref).
 2. Return every position of the asset axis on the `nothing` sentinel.
 3. Return `findall(cmsk)` otherwise.

# Arguments

  - `data`: Returns-level or price-level data ([`Prices_RR`](@ref)).
  - `cmsk`: The Coverage Universe, or `nothing`.
  - `N`: The number of assets of the window.

# Validation

  - At least one asset must be in the Coverage Universe of the window. [`coverage_mask`](@ref) throws an `IsEmptyError` on a window in which every asset is dead.

# Returns

  - `live::Vector{Int}`: The positions of the covered assets, in increasing order.

# Related

  - [`cv_nobs`](@ref)
  - [`coverage_mask`](@ref)
  - [`MultipleRandomised`](@ref)
"""
function cv_live_assets(rd::AbstractReturnsResult)
    return cv_live_assets(coverage_mask(rd.X, rd.pnl; dims = 1), size(rd.X, 2))
end
function cv_live_assets(pr::AbstractPricesResult)
    X = TimeSeries.values(pr.X)
    return cv_live_assets(coverage_mask(X, pr.pnl; dims = 1), size(X, 2))
end
cv_live_assets(::Nothing, N::Integer) = collect(one(N):N)
cv_live_assets(cmsk::BitVector, ::Integer) = findall(cmsk)
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

Abstract supertype for non-sequential non-optimisation cross-validation estimators. A
non-sequential scheme makes folds that are not a timeline, such as randomly sampled or
combinatorial folds.

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

Stores the portfolio return series of a cross-validation prediction, with the data that is
aligned to it.

`X` is the portfolio series, not the asset returns: one vector for a single portfolio, or one
vector per member of a population. Beside it the `PredictionReturnsResult` holds the factor returns, the
benchmark series, the Exogenous Series, the timestamps, the implied volatilities and the implied volatility risk
premium adjustment of the same observations.

The `PredictionReturnsResult` holds no feature matrix. [`rebuild_returns_result`](@ref) computes the collapse of
the outer problem from the original `rd.pnl` and the weights of each fold, which is the call the
path without cross-validation makes. The weights of a fold are on the `res` of its
[`PredictionResult`](@ref), and `ts` is the slice of the original clock that the fold covers,
so each fold can be rebuilt from what the result holds.

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
        ne::Option{<:VecStr} = nothing,
        E::Option{<:MatNum} = nothing,
        ts::Option{<:VecDate} = nothing,
        iv::Option{<:VecNum_VecVecNum} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing
    ) -> PredictionReturnsResult

Keywords correspond to the struct's fields.

## Validation

  - `nf` and `F`, and `ne` and `E`, pass [`assert_prediction_series`](@ref): each pair is consistent, and the matrix has one row per observation of each series in `X` and of `ts`.
  - If `B` and `X` are given, they have the same shape (`VecNum` or `VecVecNum`) and matching lengths. A mixed pair raises an `ArgumentError`.
  - If `ts` is given, it is not empty, at least one of `X` and `F` is not `nothing`, and its length matches `X`, `F`, `B` and `E` where they are given.
  - If `iv` is given, `X` is given too, else an `IsNothingError` is raised.
  - If `iv` is a `VecNum`, `ivpa` is a scalar or `nothing`, `iv` is non-empty, non-negative and finite, `ivpa` is positive and finite, and `length(iv) == length(X)`.
  - If `iv` is a `VecVecNum`, `ivpa` is a `VecNum` or `nothing`, `length(iv) == length(X)`, and `length(ivpa) == length(X)` when `ivpa` is given. Each entry of `ivpa` is positive and finite, and each vector of `iv` is non-empty, non-negative, finite, and as long as its series in `X`.

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
    $(field_dict[:pred_ne])
    """
    ne
    """
    $(field_dict[:pred_E])
    """
    E
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
                                     ne::Option{<:VecStr}, E::Option{<:MatNum},
                                     ts::Option{<:VecDate}, iv::Option{<:VecNum_VecVecNum},
                                     ivpa::Option{<:Num_VecNum})
        assert_prediction_series(nf, F, X, ts, :nf, :F)
        assert_prediction_series(ne, E, X, ts, :ne, :E)
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
            @argcheck(!isnothing(X),
                      IsNothingError("X cannot be nothing if iv is not `nothing`"))
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(length(iv) == length(X), DimensionMismatch)
        elseif isa(iv, VecVecNum)
            @argcheck(isa(ivpa, Option{<:VecNum}),
                      ArgumentError("ivpa must be a vector of numbers (or nothing) when iv is a vector of vectors of numbers, got typeof(ivpa) = $(typeof(ivpa))"))
            @argcheck(!isnothing(X),
                      IsNothingError("X cannot be nothing if iv is not `nothing`"))
            @argcheck(length(iv) == length(X), DimensionMismatch)
            # A `nothing` premium compares `X` with itself, so only a given one is checked.
            @argcheck(length(something(ivpa, X)) == length(X), DimensionMismatch)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            for (ivi, Xi) in zip(iv, X)
                assert_nonempty_nonneg_finite_val(ivi, :iv)
                @argcheck(length(ivi) == length(Xi), DimensionMismatch)
            end
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(nb), typeof(B),
                   typeof(ne), typeof(E), typeof(ts), typeof(iv), typeof(ivpa)}(nx, X, nf,
                                                                                F, nb, B,
                                                                                ne, E, ts,
                                                                                iv, ivpa)
    end
end
function PredictionReturnsResult(; nx::Option{<:VecStr} = nothing,
                                 X::Option{<:VecNum_VecVecNum} = nothing,
                                 nf::Option{<:VecStr} = nothing,
                                 F::Option{<:MatNum} = nothing,
                                 nb::Option{<:VecStr} = nothing,
                                 B::Option{<:VecNum_VecVecNum} = nothing,
                                 ne::Option{<:VecStr} = nothing,
                                 E::Option{<:MatNum} = nothing,
                                 ts::Option{<:VecDate} = nothing,
                                 iv::Option{<:VecNum_VecVecNum} = nothing,
                                 ivpa::Option{<:Num_VecNum} = nothing)::PredictionReturnsResult
    return PredictionReturnsResult(nx, X, nf, F, nb, B, ne, E, ts, iv, ivpa)
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

A failure code is enough to drop the member. The library reads a vector of return codes with `any(x -> isa(x, OptimisationFailure), …)`, and the cross-validation path keeps a path only when `isa(y.res.retcode, OptimisationSuccess)` holds, so a failed member takes its path out of the run.

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

When the series of a fold is drifted, [`calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs)`](@ref) runs the same drift over the same window and warns. A fold that drifts only its held weights runs no such call, so this function warns instead. Either way the fold warns once.

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

Stores the result of a single cross-validation fold prediction. It pairs an optimisation
result with the portfolio return series of the test period.

A fold of a cross-validation records its rows in two keys. `idx` holds the `test_idx` of the
fold, the positions of its rows in the returns data that the cross-validation split. `rd.ts`
is the slice of the original clock that the fold covers, because [`port_opt_view`](@ref)
slices it with the same `test_idx`, so [`feature_row_indices`](@ref) finds the absolute rows
by their timestamps. [`rebuild_returns_result`](@ref) reads the timestamps. A realised
[`factor_attribution`](@ref) reads the timestamps, or the positions when the data carries no
timestamps. Both keys stay correct on the combinatorial path, where the folds of a path come
in split order and not in time order.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PredictionResult(;
        res::NonFiniteAllocationOptimisationResult,
        rd::PredictionReturnsResult,
        hw::Option{<:HeldWeightsResult} = nothing,
        idx::Option{<:VecInt} = nothing
    ) -> PredictionResult

Keywords correspond to the struct's fields. `res` and `rd` are required, because a fold prediction needs both. `hw` defaults to `nothing`, which is a fold that held its target weights on every observation. `idx` defaults to `nothing`, which is a prediction made over returns data that no fold index selected.

`hw` is present only when a Weight Drift or a Previous-Weights Source ran over the fold, and the fold-taking consumers dispatch on its type. It carries the asset returns of the fold, the weights the drift started from, the weights held after the last observation and the drift that made them. [`weight_path`](@ref) rebuilds the weight path from it.

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
    """
    Position of each observation of the fold in the returns data that the cross-validation split, the `test_idx` of the fold, or `nothing` when no fold index selected the observations.
    """
    idx
    function PredictionResult(res::NonFiniteAllocationOptimisationResult,
                              rd::PredictionReturnsResult, hw::Option{<:HeldWeightsResult},
                              idx::Option{<:VecInt})
        return new{typeof(res), typeof(rd), typeof(hw), typeof(idx)}(res, rd, hw, idx)
    end
end
function PredictionResult(; res::NonFiniteAllocationOptimisationResult,
                          rd::PredictionReturnsResult,
                          hw::Option{<:HeldWeightsResult} = nothing,
                          idx::Option{<:VecInt} = nothing)::PredictionResult
    return PredictionResult(res, rd, hw, idx)
end
"""
    previous_weights(pws::Any, prev::Nothing)
    previous_weights(pws::Nothing, prev::PredictionResult)
    previous_weights(pws::AbstractPreviousWeightsSource, prev::PredictionResult)

Read the weights a fold threads into the fold that follows it.

This is the one function that reads the Previous-Weights Source. The first fold of a run has no fold behind it, so it threads nothing whatever the source is. A later fold threads the target weights of the previous fold by default, and the weights that fold **held** after its last observation when the scheme sets a source.

`prev` is the last fold whose weights can be threaded, which is not always the fold before. The sequential loops advance it only when [`threads_weights`](@ref) holds of a fold, so a failed fold is skipped and the fold before it is read. The weights this function gives are therefore finite whenever it gives any.

# Algorithm

 1. With no previous fold, give `nothing`.
 2. With no source, give the target weights of the previous fold, `prev.res.w`.
 3. With a source, give the held weights of the previous fold, `prev.hw.w`.

# Arguments

  - `pws`: Previous-weights source of the scheme, or `nothing`.
  - `prev`: Prediction result of the last threadable fold, or `nothing`.

# Returns

  - `Option{<:VecNum_VecVecNum}`: The weights the next fold reads through [`factory`](@ref).

# Related

  - [`AbstractPreviousWeightsSource`](@ref)
  - [`DriftedWeights`](@ref)
  - [`threads_weights`](@ref)
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
    threads_weights(pws::Nothing, pred::PredictionResult)
    threads_weights(pws::AbstractPreviousWeightsSource, pred::PredictionResult)

Say whether a fold's prediction carries weights the next fold can be handed.

[`previous_weights`](@ref) reads the weights, and this function says whether the fold has any to read. The sequential loops, [`run_folds`](@ref) and [`online_folds`](@ref), hand a fold on only when this holds, so a failed fold is never read and the last threadable fold is read instead.

The weights that are read decide the test. The target weights are finite exactly when every return code is an [`OptimisationSuccess`](@ref). The held weights are finite when the drift ran, and [`held_start_weights`](@ref) lets it run on a failed fold that was handed previous weights. So under a source, a failed fold that held its book is threaded, and only a fold with nothing to hold is skipped.

# Algorithm

 1. With no source, hold when the fold's return code is a success, every member's under a population.
 2. With a source, hold when the fold's held weights are all finite, every member's under a population.

# Arguments

  - `pws`: Previous-weights source of the scheme, or `nothing`.
  - `pred`: Prediction result of the fold.

# Returns

  - `Bool`: Whether the fold's weights can be threaded.

# Related

  - [`previous_weights`](@ref)
  - [`held_start_weights`](@ref)
  - [`run_folds`](@ref)
  - [`online_folds`](@ref)
  - [`PreviousWeights`](@ref): The fallback that turns a failed fold into a threadable one.
"""
function threads_weights(::Nothing, pred::PredictionResult)
    return fold_solved(pred.res.retcode)
end
function threads_weights(::AbstractPreviousWeightsSource, pred::PredictionResult)
    return all(w -> all(isfinite, w), held_weight_members(pred.hw.w))
end
"""
    fold_solved(retcode::OptimisationReturnCode)
    fold_solved(retcode::VecOptRetCode)

Say whether a fold's return code, or every member's under a population, is an [`OptimisationSuccess`](@ref).

# Related

  - [`threads_weights`](@ref)
  - [`OptimisationSuccess`](@ref)
"""
function fold_solved(retcode::OptimisationReturnCode)
    return isa(retcode, OptimisationSuccess)
end
function fold_solved(retcode::VecOptRetCode)
    return all(fold_solved, retcode)
end
"""
    held_weight_members(w::VecNum)
    held_weight_members(w::VecVecNum)

Iterate the held weights of a fold one member at a time: a single vector is a population of one.

# Related

  - [`threads_weights`](@ref)
"""
function held_weight_members(w::VecNum)
    return (w,)
end
function held_weight_members(w::VecVecNum)
    return w
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

This is the method of [`rolling_window_measure`](@ref) over the realised history of a fold. `pred.rd.X` is the series that [`predict`](@ref) stored, so the method reads no weights. Under a Weight Drift that series is the drifted one, and each window is a part of the drift of the fold.

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

This is the fold-taking method of [`calc_net_asset_returns`](@ref), and the mirror of [`calc_net_returns(res::OptimisationResult, X, fees)`](@ref). It finds the asset returns, the weight path and the fee of the fold, so a caller that holds a fold reaches the split in one call. The fee is spread over the length of the fold as [`predict`](@ref) spread it, so the rows of the result sum to the series the fold stored, to rounding.

A fold that carries no Held Weights record raises. `pred.rd.X` is the **portfolio** series, not the asset returns, so a fold whose scheme set neither `wd` nor `pws` keeps no matrix to split.

# Algorithm

 1. Rebuild the weight path of the fold from its record with [`weight_path`](@ref), giving `U`.
 2. Resolve the fee with [`fold_fees`](@ref): the fee of the result, or the caller's fee viewed at the Investable Mask.
 3. Split the net return of each observation over the assets with the free [`calc_net_asset_returns`](@ref), from `U`, the asset returns `hw.X`, the fee and the mask.

# Arguments

  - `pred`: Single-fold prediction result.
  - `fees`: A caller's [`Fees`](@ref) on the caller's universe, which takes precedence over the result's own and is viewed at the result's Investable Mask through [`fold_fees`](@ref).
  - `args...`: Additional arguments (ignored by the refusing method).

# Validation

  - The fold carries a [`HeldWeightsResult`](@ref), else an `ArgumentError` naming the two switches.

# Returns

  - `ret::MatNum`: Per asset net returns of the fold, one row per observation.

# Related

  - [`calc_net_asset_returns`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`weight_path`](@ref)
  - [`PredictionResult`](@ref)
  - [`fold_fees`](@ref)
"""
function calc_net_asset_returns(pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult},
                                fees::Option{<:Fees} = nothing)
    hw = pred.hw
    # The record was expanded back to the caller's universe on the way out of `predict`, so
    # its matrix and its weight path span every asset while the result's fee spans the two
    # reduced axes. The mask is what reunites them: it says which columns the five per asset
    # fields were priced on and which columns the two liquidation charges were priced on.
    return calc_net_asset_returns(weight_path(hw, pred.res.w), hw.X,
                                  fold_fees(pred.res, fees, hw.X),
                                  result_investable_mask(pred.res))
end
function calc_net_asset_returns(::PredictionResult{<:Any, <:Any, Nothing}, args...)
    return throw(ArgumentError("`calc_net_asset_returns(pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`.\nSet one of them so the fold records its asset returns, or call `calc_net_asset_returns(w, X, fees)` with the returns you fitted on."))
end
"""
    risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult}, fees = nothing; kwargs...)
    risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, Nothing}, args...; kwargs...)

Decompose a fold's risk over its assets.

This is the fold-taking method of [`risk_contribution`](@ref). It finds the **target** weights, the asset returns and the fee of the fold, and hands them to the free function, so the figures are the ones the free function gives.

The three are not on one universe. The Held Weights record and the target weights come back on the caller's universe through [`expand_held_weights`](@ref), and the fee of the result stays on the universe the fit solved on. The method therefore views the weights and the asset returns at the Investable Mask of the result before the finite difference, as [`investable_reduction`](@ref) does for the value-level method, and expands the answer back with a zero at every non-investable asset. A result whose mask is `nothing` views nothing.

Under a Weight Drift the figures are exact to **first order in the drift** only. The drifted series is not linear in the target weights, so the contributions sum to the realised risk of the fold approximately. The contributions are still stated against the target weights, because the finite difference perturbs them.

A fold that carries no Held Weights record raises, because `pred.rd.X` is the portfolio series and the fold keeps no asset matrix.

# Algorithm

 1. Read the Investable Mask of the result with [`result_investable_mask`](@ref), giving `imsk`.
 2. View the target weights and the asset returns `hw.X` at `imsk` with [`investable_weights_view`](@ref).
 3. Resolve the fee with [`fold_fees`](@ref).
 4. Compute the contributions with the free [`risk_contribution`](@ref), giving `rc`.
 5. Expand `rc` to the caller's universe with [`expand_investable_weights`](@ref), with a zero at every non-investable asset.

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to differentiate, or a vector of them.
  - `pred`: Single-fold prediction result.
  - `fees`: A caller's [`Fees`](@ref) on the caller's universe, which takes precedence over the result's own and is viewed at the result's Investable Mask through [`fold_fees`](@ref).
  - `args...`: Additional arguments (ignored by the refusing method).

# Validation

  - The fold carries a [`HeldWeightsResult`](@ref), else an `ArgumentError` naming the two switches.

# Returns

  - `Vector`: Risk contributions (or marginal risks) for each asset of the caller's universe, exactly `0` at a non-investable one.

# Related

  - [`risk_contribution`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`PredictionResult`](@ref)
  - [`expand_held_weights`](@ref)
  - [`investable_weights_view`](@ref)
  - [`fold_fees`](@ref)
"""
function risk_contribution(r::BaseRM_VecBaseRM,
                           pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult},
                           fees::Option{<:Fees} = nothing; kwargs...)
    # The record and the target weights are on the caller's universe, and the result's
    # fee is on the investable one, so the fold views the first two at the mask before
    # the finite difference and expands the answer back, as the value-level method does.
    imsk = result_investable_mask(pred.res)
    rc = risk_contribution(r, investable_weights_view(imsk, pred.res.w),
                           investable_weights_view(imsk, pred.hw.X),
                           fold_fees(pred.res, fees, pred.hw.X); kwargs...)
    return expand_investable_weights(imsk, rc)
end
function risk_contribution(::BaseRM_VecBaseRM, ::PredictionResult{<:Any, <:Any, Nothing},
                           args...; kwargs...)
    return throw(ArgumentError("`risk_contribution(r, pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`.\nSet one of them so the fold records its asset returns, or call `risk_contribution(r, pred.res.w, rd.X, fees)` with the returns you fitted on."))
end
"""
    factor_risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult}, fees = nothing; rd, kwargs...)
    factor_risk_contribution(r::BaseRM_VecBaseRM, pred::PredictionResult{<:Any, <:Any, Nothing}, args...; kwargs...)

Decompose a fold's risk over its factors.

This is the fold-taking method of [`factor_risk_contribution`](@ref), and the twin of the fold-taking [`risk_contribution`](@ref) method. It finds the target weights, the asset returns and the fee of the fold, and views them at the Investable Mask of the result, as that method does. It also builds the `rd` that the loadings are fitted from, out of the fold itself: the asset returns of the fold beside the factor block that [`reconstruct_rd`](@ref) kept. A caller who wants other loadings passes its own `rd`, or a precomputed [`Regression`](@ref) as `re`.

The first-order caveat of the fold-taking [`risk_contribution`](@ref) method holds here unchanged, and a fold that carries no Held Weights record raises for the same reason.

# Algorithm

 1. Read the Investable Mask of the result, giving `imsk`.
 2. View the target weights and the asset returns `hw.X` at `imsk` with [`investable_weights_view`](@ref).
 3. Resolve the fee with [`fold_fees`](@ref).
 4. Resolve the returns result of the loadings on the live assets with [`fold_factor_returns`](@ref).
 5. Compute the factor contributions with the free [`factor_risk_contribution`](@ref).

# Arguments

  - `r::BaseRM_VecBaseRM`: Risk measure to decompose, or a vector of them.
  - `pred`: Single-fold prediction result.
  - `fees`: A caller's [`Fees`](@ref) on the caller's universe, which takes precedence over the result's own and is viewed at the result's Investable Mask through [`fold_fees`](@ref).
  - `args...`: Additional arguments (ignored by the refusing method).

# Keyword Arguments

  - `rd::Option{<:ReturnsResult} = nothing`: Returns result the loadings are fitted from, on the caller's universe, or `nothing` for the fold's own asset returns and factor block.

# Validation

  - The fold carries a [`HeldWeightsResult`](@ref), else an `ArgumentError` naming the two switches.

# Returns

  - `Vector`: Risk contributions for each factor, with the last element being the idiosyncratic (off-factor) contribution.

# Related

  - [`factor_risk_contribution`](@ref)
  - [`risk_contribution`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`PredictionResult`](@ref)
  - [`fold_factor_returns`](@ref)
  - [`fold_fees`](@ref)
"""
function factor_risk_contribution(r::BaseRM_VecBaseRM,
                                  pred::PredictionResult{<:Any, <:Any, <:HeldWeightsResult},
                                  fees::Option{<:Fees} = nothing;
                                  rd::Option{<:ReturnsResult} = nothing, kwargs...)
    # The same view at the mask as the `risk_contribution` method above, and `rd` takes
    # it too, so the loadings are fitted over the live assets the weights are scored on.
    imsk = result_investable_mask(pred.res)
    return factor_risk_contribution(r, investable_weights_view(imsk, pred.res.w),
                                    investable_weights_view(imsk, pred.hw.X),
                                    fold_fees(pred.res, fees, pred.hw.X);
                                    rd = fold_factor_returns(imsk, rd, pred), kwargs...)
end
"""
    fold_factor_returns(imsk, rd::Nothing, pred::PredictionResult)
    fold_factor_returns(imsk, rd::ReturnsResult, pred::PredictionResult)

Resolve the returns result a fold's factor loadings are fitted from, on the live assets of the fold.

The fold-taking [`factor_risk_contribution`](@ref) reads `rd` by dispatch. A caller's `rd` is on the caller's universe, as it is for the value-level method, so it is viewed at the Investable Mask of the result. With `nothing`, the function builds a returns result from the fold itself. The `nx` of the fold is on the reduced axis already.

# Algorithm

 1. On a caller's `rd`, view it at `imsk` with [`investable_returns_view`](@ref).
 2. On `nothing`, view the asset returns `hw.X` of the fold at `imsk`, and build a [`ReturnsResult`](@ref) from them, the `nx` of the fold and the factor block `nf`, `F` that [`reconstruct_rd`](@ref) kept.

# Arguments

  - `imsk`: The result's Investable Mask, or `nothing`.
  - `rd`: Returns result on the caller's universe, or `nothing`.
  - `pred`: Single-fold prediction result carrying a [`HeldWeightsResult`](@ref).

# Returns

  - `rd::ReturnsResult`: The returns result on the live assets.

# Related

  - [`factor_risk_contribution`](@ref)
  - [`investable_returns_view`](@ref)
  - [`investable_weights_view`](@ref)
  - [`HeldWeightsResult`](@ref)
"""
function fold_factor_returns(imsk, ::Nothing, pred::PredictionResult)
    X = investable_weights_view(imsk, pred.hw.X)
    return ReturnsResult(; nx = pred.rd.nx, X = X, nf = pred.rd.nf, F = pred.rd.F,
                         ne = pred.rd.ne, E = pred.rd.E)
end
function fold_factor_returns(imsk, rd::ReturnsResult, ::PredictionResult)
    return investable_returns_view(imsk, rd)
end
function factor_risk_contribution(::BaseRM_VecBaseRM,
                                  ::PredictionResult{<:Any, <:Any, Nothing}, args...;
                                  kwargs...)
    return throw(ArgumentError("`factor_risk_contribution(r, pred::PredictionResult)` needs the fold's asset returns, and this fold kept none: `pred.rd.X` is the portfolio return series, and `pred.hw` is absent because the fold's scheme set neither `wd` nor `pws`.\nSet one of them so the fold records its asset returns, or call `factor_risk_contribution(r, pred.res.w, rd.X, fees; rd = rd)` with the returns you fitted on."))
end
"""
    mapreduce_RetMtx(rd, sym = :X)

Concatenate the series of a field over the folds of a path.

A field holds one series for a single portfolio, or one series per member for a population. The folds are concatenated in the order of `rd`.

# Algorithm

 1. For one series per fold, concatenate the series of every fold with `vcat`.
 2. For one series per member, concatenate the series of member `i` over every fold, for each member `i` of the first fold.

# Arguments

  - `rd`: Vector of [`PredictionReturnsResult`](@ref) objects, one per fold.
  - `sym`: Name of the field to concatenate, `:X` by default.

# Returns

  - The concatenated series, or one concatenated series per member.
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

Stores the predictions of the folds of one path, with their return data concatenated into one
[`PredictionReturnsResult`](@ref).

The quantities with one value per observation (`X`, `F`, `B`, `ts`, `iv`) are concatenated
over the folds. `ivpa` has one value per synthetic asset, and the [`reconstruct_rd`](@ref) of
each fold has already collapsed it with the weights of that fold, so it cannot be
concatenated. The constructor keeps the value of the **last fold**. This matches
[`predict_realised_vols`](@ref), which reads the last row of the concatenated `iv`, so the
premium divisor pairs with the implied volatility it divides.

A feature matrix is **not** among the concatenated quantities, because the folds do not carry
one. [`rebuild_returns_result`](@ref) computes the outer collapse from the original `rd.pnl`,
and reaches each fold through `pred[f].res.w` and `pred[f].rd.ts`. That is why the result keeps
`pred`.

The Result of an online run also carries, in `opt`, the estimator that the fold loop threaded,
folded through the last training end `last(train_idx[end])`. A batch run writes `nothing`.
[`Resume`](@ref) hands the Result back to the loop over a longer history, and the loop continues
from the fold after the last one held. A hand step in the value form, `partial_fit(res.opt, rows)`, leaves the Result resumable. The bang form, `partial_fit!(res.opt, rows)`, writes the
arrays of the state in place, the held timestamps among them. The Result then names a row at
which no fold ends, and `Resume` refuses it. [`partial_fit!`](@ref) states this contract for a
kept estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MultiPeriodPredictionResult(;
        pred::VecPredRes,
        id::Any = nothing,
        opt::Option{<:AbstractEstimator} = nothing
    ) -> MultiPeriodPredictionResult

Keywords correspond to the struct's fields. `pred` is required. The constructor concatenates the return data of the folds into `mrd`, and a concatenation of no folds has no names and no clock.

## Validation

  - `!isempty(pred)`.

# Related

  - [`PredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`sort_by_measure`](@ref)
  - [`PredictionReturnsResult`](@ref)
  - [`Resume`](@ref)
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
    """
    $(field_dict[:opt_pred])
    """
    opt
    function MultiPeriodPredictionResult(pred::VecPredRes, id::Any,
                                         opt::Option{<:AbstractEstimator})
        @argcheck(!isempty(pred), IsEmptyError("pred cannot be empty"))
        rd = getfield.(pred, :rd)
        nx = rd[1].nx
        X = mapreduce_RetMtx(rd)
        nf = rd[1].nf
        F = isnothing(rd[1].F) ? nothing : mapreduce(x -> getproperty(x, :F), vcat, rd)
        nb = rd[1].nb
        B = isnothing(rd[1].B) ? nothing : mapreduce_RetMtx(rd, :B)
        ne = rd[1].ne
        E = isnothing(rd[1].E) ? nothing : mapreduce(x -> getproperty(x, :E), vcat, rd)
        ts = isnothing(rd[1].ts) ? nothing : mapreduce(x -> getproperty(x, :ts), vcat, rd)
        iv = isnothing(rd[1].iv) ? nothing : mapreduce(x -> getproperty(x, :iv), vcat, rd)
        # Per-asset, so it cannot stack; reduced to the last fold to pair with the last
        # row of `iv`, which is what `predict_realised_vols` divides by it.
        ivpa = rd[end].ivpa
        mrd = PredictionReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nb, B = B,
                                      ne = ne, E = E, ts = ts, iv = iv, ivpa = ivpa)
        return new{typeof(pred), typeof(mrd), typeof(id), typeof(opt)}(pred, mrd, id, opt)
    end
end
function MultiPeriodPredictionResult(; pred::VecPredRes, id::Any = nothing,
                                     opt::Option{<:AbstractEstimator} = nothing)::MultiPeriodPredictionResult
    return MultiPeriodPredictionResult(pred, id, opt)
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
# per-fold results and per-fold prediction returns (see [`@forward_properties`](@ref)).
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

Stores the paths of a cross-validation scheme that makes more than one path. Each element of
`pred` is one path: a random asset subset of [`MultipleRandomised`](@ref), or a path of
[`CombinatorialCrossValidation`](@ref). A member is usually a
[`MultiPeriodPredictionResult`](@ref), and a population built by hand can hold a single-fold
[`PredictionResult`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PopulationPredictionResult(;
        pred::VecPredRes_MultiPredRes = Vector{PredRes_MultiPredRes}(undef, 0)
    ) -> PopulationPredictionResult

Keywords correspond to the struct's fields. An empty `pred` is admitted: a population from which every path was dropped is a valid, if empty, answer.

## Every multi-period member carries an `id`

A [`MultiPeriodPredictionResult`](@ref) member whose `id` is `nothing` takes its position in `pred` as its `id`, through [`population_ids`](@ref). The schemes that make a population already number their paths that way, so the rule changes nothing for them. It gives an identifier to a population built by hand from [`cross_val_predict`](@ref) streams, so the path that [`NearestQuantilePrediction`](@ref) or [`sort_by_measure`](@ref) selects names its place in the population. A member that already carries an `id` keeps it. A single-fold [`PredictionResult`](@ref) has no `id` field, and is kept as it is.

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
        pred = population_ids(pred)
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
    return [expected_risk(r, p; kwargs...) for p in ppred.pred]
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

Rolls `r` over each member of `ppred.pred`, as [`expected_risk`](@ref) does on the same type, so a member can be a single fold or a multi-period result.

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
    return [rolling_window_measure(r, p, window; kwargs...) for p in ppred.pred]
end
"""
    sort_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM; kwargs...)

Sort the successful paths in a [`PopulationPredictionResult`](@ref) by their expected
risk under `r`. A path in which any fold returned a failure code is left out.

A path whose measure is not finite goes **after** every finite one, under both directions of
the ranking, because a non-finite number is not a best and must not be the answer of a
`first`. The finite members are sorted among themselves, and the non-finite ones keep their own
order at the end.

The direction of the ranking comes from [`bigger_is_better`](@ref), which **throws** on a vector
of measures that disagree on polarity. [`quantile_by_measure`](@ref) takes an explicit `sign`
instead, so it admits a mixed vector.

# Algorithm

 1. Keep the successful members of `ppred` with [`successful_members`](@ref).
 2. Compute the expected risk of each member under `r`, giving `rks`.
 3. Sort the members whose `rks` is finite by `rks`, in descending order when [`bigger_is_better`](@ref) holds of `r` and in ascending order otherwise.
 4. Append the members whose `rks` is not finite, in their own order.

# Arguments

  - `ppred::PopulationPredictionResult`: Population prediction to sort.
  - `r::BaseRM_VecBaseRM`: Risk measure used for ranking, or a vector of them. A vector is scalarised by `kwargs.sca`, defaulting to [`SumScalariser`](@ref).
  - `kwargs...`: Keyword arguments forwarded to `expected_risk`.

# Validation

  - A vector `r` whose measures disagree on polarity raises in [`bigger_is_better`](@ref).

# Returns

  - `Vector`: The successful members, sorted. Each is a [`MultiPeriodPredictionResult`](@ref), or a [`PredictionResult`](@ref) when the population holds single folds.

# Related

  - [`quantile_by_measure`](@ref)
  - [`bigger_is_better`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`expected_risk`](@ref)
"""
function sort_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM; kwargs...)
    pred = successful_members(ppred)
    rks = [expected_risk(r, x; kwargs...) for x in pred]
    fin = isfinite.(rks)
    idx = findall(fin)
    return [pred[idx[sortperm(rks[idx]; rev = bigger_is_better(r))]]; pred[findall(!, fin)]]
end
"""
    quantile_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM, q::Real;
                        r_kwargs::NamedTuple = (;), q_kwargs::NamedTuple = (;),
                        sign::Integer = 1)

Select the successful path in `ppred` whose expected risk under `r` is closest to the `q`-th quantile of the risk distribution across all successful paths.

A path whose measure is not finite takes no part. It is dropped before the quantile, so one failed path cannot make the quantile non-finite and cannot be the answer.

[`sort_by_measure`](@ref) calls [`bigger_is_better`](@ref) for its direction, so it **throws** on a vector of measures that disagree on polarity. This function takes an explicit `sign` instead. The caller gives the one fact that `bigger_is_better` cannot infer, so a mixed vector is admitted.

# Algorithm

 1. Keep the successful members of `ppred` with [`successful_members`](@ref).
 2. Compute `sign` times the expected risk of each member under `r`, giving `rks`.
 3. Drop the members whose `rks` is not finite.
 4. Compute the `q`-th quantile of `rks` with `Statistics.quantile`, giving `rkq`.
 5. Return the first member whose `abs(rks[i] - rkq)` is the smallest.

# Arguments

  - `ppred::PopulationPredictionResult`: Population prediction result.
  - `r::BaseRM_VecBaseRM`: Risk measure for computing path risks, or a vector of them. A vector is scalarised by `r_kwargs.sca`, defaulting to [`SumScalariser`](@ref).
  - `q::Real`: Quantile level in `[0, 1]`.
  - `r_kwargs::NamedTuple = (;)`: Keyword arguments forwarded to `expected_risk`.
  - `q_kwargs::NamedTuple = (;)`: Keyword arguments forwarded to `Statistics.quantile`.
  - `sign::Integer = 1`: Orientation of the risk scale. Use `1` when a larger risk is worse, `-1` when it is better. This is what lets a mixed vector through, see below.

# Validation

  - At least one successful member must have a finite measure. With none, `Statistics.quantile` receives an empty vector and throws.

# Returns

  - The member closest to the `q`-th quantile: a [`MultiPeriodPredictionResult`](@ref), or a [`PredictionResult`](@ref) when the population holds single folds.

# Related

  - [`sort_by_measure`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`expected_risk`](@ref)
"""
function quantile_by_measure(ppred::PopulationPredictionResult, r::BaseRM_VecBaseRM,
                             q::Real; r_kwargs::NamedTuple = (;),
                             q_kwargs::NamedTuple = (;), sign::Integer = 1)
    pred = successful_members(ppred)
    rks = [sign*expected_risk(r, p; r_kwargs...) for p in pred]
    fin = findall(isfinite, rks)
    rks = rks[fin]
    pred = pred[fin]
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

A benchmark that is already a series passes through. A benchmark matrix is contracted with the weights of the fold, and dispatch on the pair `(B, w)` chooses the method.

The Held Weights record of the fold chooses the weights. Without one, the matrix collapses against the target weights, which is what a fold that ran no drift held. With one, it collapses row by row against the weight path, as the portfolio series does. A caller that compares the two series, for a tracking error for instance, then compares two series formed the same way.

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
    investable_fold_view(imsk::Nothing, w, rd::ReturnsResult, fees::Option{<:Fees})
    investable_fold_view(imsk::BitVector, w, rd::ReturnsResult, fees::Option{<:Fees})

View a fold's weights and test window at the Investable Mask, and pass its fees through.

An optimisation reduces to the Investable Mask at its entry and expands the solved weights back to the caller's universe, so the weight of an asset that the fit found non-investable is `0`. The fold views the weights and the window at the mask together, before anything reads the window. A dead column is therefore never read, and [`filter_held_gaps`](@ref) runs over the investable columns alone.

The fees are **not** viewed. A result carries the objects of the universe it solved on, so `res.fees` is on the investable universe already, and a second view would index its per-asset rates by positions of the full universe. The function returns the fees unchanged so that the fold receives the three values it scores with from one call.

# Algorithm

 1. With no mask, give `(w, rd, fees)` unchanged.
 2. With a mask, give the weights viewed at the mask with [`investable_weights_view`](@ref), `rd` viewed at the columns `findall(imsk)` with [`port_opt_view`](@ref), and `fees` unchanged.

# Arguments

  - `imsk`: The Investable Mask, or `nothing`.
  - `w`: The fold's target weights, or a population of them.
  - $(arg_dict[:rd])
  - `fees`: [`Fees`](@ref) the fold is charged, on the investable universe, or `nothing`.

# Returns

  - `(w, rd, fees)`: The weights and the window reduced to the investable assets, or unchanged, and the fees as given.

# Related

  - [`result_investable_mask`](@ref)
  - [`investable_weights_view`](@ref)
  - [`filter_held_gaps`](@ref)
  - [`expand_held_weights`](@ref)
  - [`port_opt_view`](@ref)
"""
function investable_fold_view(::Nothing, w::VecNum_VecVecNum, rd::ReturnsResult,
                              fees::Option{<:Fees})
    return w, rd, fees
end
function investable_fold_view(imsk::BitVector, w::VecNum_VecVecNum, rd::ReturnsResult,
                              fees::Option{<:Fees})
    return investable_weights_view(imsk, w), port_opt_view(rd, findall(imsk)), fees
end
"""
    fold_fees(res::OptimisationResult, fees::Nothing, X::MatNum)
    fold_fees(res::OptimisationResult, fees::Fees, X::MatNum)

Resolve the fee a fold-taking consumer charges: the result's own, or a caller's viewed at the result's Investable Mask.

The three fold-taking consumers, [`calc_net_asset_returns`](@ref), [`risk_contribution`](@ref) and [`factor_risk_contribution`](@ref) on a [`PredictionResult`](@ref), take an optional `fees`, so a caller can score a stored fold under a fee of their own. The two fees are on different universes. The fit reduced the fee of the result, so its five per-asset fields are on the investable axis and its two liquidation charges, `lq` and `flq`, are on the complement. A caller's fee is on the caller's universe, as a caller's `rd` is, so it takes the same view that a fee takes at the fit.

Without the view, a per-asset rate on the full universe met the reduced weights with a `BoundsError`, and a liquidation charge on the full universe charged every position as though it had been liquidated.

# Algorithm

 1. With no caller fee, give the fee of the result through [`extract_fees`](@ref).
 2. With a caller fee, view it with [`investable_fees_view`](@ref). The per-asset fields are sliced to the mask and the two liquidation charges to its complement, which is found from the width of `X`. The two liquidation charges are removed when the mask is `nothing`, because no asset left.

# Arguments

  - `res::OptimisationResult`: The fold's optimisation result, carrying the mask and its own fee.
  - `fees`: A caller's [`Fees`](@ref) on the caller's universe, or `nothing` to charge the result's own.
  - `X`: The fold's expanded record, `observations × assets` on the caller's universe. Only its width is read.

# Returns

  - `fees::Option{<:Fees}`: The fee on the two axes the result's mask leaves, or `nothing`.

# Related

  - [`extract_fees`](@ref)
  - [`investable_fees_view`](@ref)
  - [`investable_fold_view`](@ref)
  - [`result_investable_mask`](@ref)
"""
function fold_fees(res::OptimisationResult, ::Nothing, ::MatNum)
    return extract_fees(res, nothing)
end
function fold_fees(res::OptimisationResult, fees::Fees, X::MatNum)
    return investable_fees_view(fees, result_investable_mask(res), X)
end
"""
    reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult, X, hw = nothing, w = res.w)

Reconstruct a `PredictionReturnsResult` from an optimisation result and returns data.

It collapses the benchmark returns, the implied volatilities and the implied volatility risk premium adjustment of `rd` with the weights of the fold, and pairs them with the portfolio series `X`.

A benchmark matrix collapses as [`collapse_benchmark`](@ref) states: against the weight path when the fold carries a [`HeldWeightsResult`](@ref), and against the target weights when it does not. `iv` and `ivpa` are rates, so they collapse as convex combinations, against the weights of [`synthetic_asset_weights`](@ref).

The fold does not collapse the panel of `rd`. A *square* feature matrix needs the weights of every synthetic asset at once for its second contraction, and only one weight vector is known here. [`rebuild_returns_result`](@ref) instead computes the collapse for the whole synthetic universe from the original `rd.pnl` and the fold weights `pred[f].res.w`.

# Algorithm

 1. Collapse `rd.B` with [`collapse_benchmark`](@ref), from `w` and `hw`.
 2. When `rd.iv` is given or `rd.ivpa` is a vector, compute the convex weights `cw` of `w` with [`synthetic_asset_weights`](@ref), one vector per member under a population.
 3. Collapse `rd.iv` to `rd.iv * cw`, and a vector `rd.ivpa` to `dot(rd.ivpa, cw)`. A scalar `ivpa` stays as it is.
 4. Under a population, repeat a scalar `ivpa` once per member.
 5. Build the [`PredictionReturnsResult`](@ref) from `nx`, `X`, the factor block, `nb`, the collapsed values and `ts`.

# Arguments

  - `res::NonFiniteAllocationOptimisationResult`: Fitted optimisation result.
  - `rd::ReturnsResult`: Original returns data.
  - `X`: Portfolio returns (vector or vector of vectors).
  - `hw`: Held Weights record of the fold, or `nothing`.
  - `w`: The weights the fold's series was formed from. It defaults to `res.w`, and a fold that viewed its window at the Investable Mask passes the view instead, so the collapse reads the same asset axis `rd` carries.

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
                        X::VecNum, hw::Option{<:HeldWeightsResult} = nothing,
                        w::VecNum_VecVecNum = res.w)
    B = collapse_benchmark(rd.B, w, hw)
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        # `iv` and `ivpa` are intensive, so they collapse as convex combinations. They are
        # collapsed against the same weights the series was formed from, which is the view
        # at the Investable Mask when the fold took one.
        cw = synthetic_asset_weights(w)
        if iv_flag
            iv = iv * cw
        end
        if ivpa_flag
            ivpa = LinearAlgebra.dot(rd.ivpa, cw)
        end
    end
    return PredictionReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, nb = rd.nb,
                                   B = B, ne = rd.ne, E = rd.E, ts = rd.ts, iv = iv,
                                   ivpa = ivpa)
end
function reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        X::VecVecNum, hw::Option{<:HeldWeightsResult} = nothing,
                        w::VecNum_VecVecNum = res.w)
    nb = rd.nb
    B = collapse_benchmark(rd.B, w, hw)
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        # `iv` and `ivpa` are intensive, so they collapse as convex combinations, against
        # the same weights the series was formed from — see the singular twin above.
        cw = [synthetic_asset_weights(wi) for wi in w]
        if iv_flag
            iv = [iv * wi for wi in cw]
        end
        if ivpa_flag
            ivpa = [LinearAlgebra.dot(ivpa, wi) for wi in cw]
        end
    end
    if isa(ivpa, Number)
        ivpa = range(; start = ivpa, stop = ivpa, length = length(res.w))
    end
    return PredictionReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, nb = nb,
                                   B = B, ne = rd.ne, E = rd.E, ts = rd.ts, iv = iv,
                                   ivpa = ivpa)
end
"""
    held_start_weights(retcode::OptimisationSuccess, w::VecNum, w_prev)
    held_start_weights(retcode::OptimisationFailure, w::VecNum, w_prev::Nothing)
    held_start_weights(retcode::OptimisationFailure, w::VecNum, w_prev::VecNum)
    held_start_weights(retcode::OptimisationReturnCode, w::VecVecNum, w_prev)
    held_start_weights(retcode::VecOptRetCode, w::VecVecNum, w_prev::Nothing)
    held_start_weights(retcode::VecOptRetCode, w::VecVecNum, w_prev::VecNum)
    held_start_weights(retcode::VecOptRetCode, w::VecVecNum, w_prev::VecVecNum)

Name the weights a fold's drift starts from: its own on a solved fold, the previous weights on a failed one.

A fold that could not rebalance holds what it held. So under a Weight Drift or a Previous-Weights Source, a failed fold drifts the weights it was handed and not its `NaN` target. The return code decides, read per member under a population, and the previous weights are one vector for every member or one vector per member. A population solved under one return code, such as the frontier of one [`JuMPOptimisationResult`](@ref), reads that code for every member. A failed fold or member with no previous weights keeps its `NaN` weights, and [`held_weights_result`](@ref) records `NaN` for it without a drift.

# Algorithm

 1. On an [`OptimisationSuccess`](@ref), give `w`.
 2. On an [`OptimisationFailure`](@ref) over one vector, give `w_prev`, or `w` when there is none.
 3. Over a population with one return code, repeat that code once per member.
 4. Over a population with no `w_prev`, give `w`.
 5. Over a population with one `w_prev` vector, repeat it once per member.
 6. Over a population with one `w_prev` per member, give each member its own vector on a success and its entry of `w_prev` on a failure.

# Arguments

  - `retcode`: Return code of the fold, or one per member of the population.
  - `w`: Target weights of the fold, on the universe the fold is scored on.
  - `w_prev`: Previous weights the fold was handed, on the same universe, or `nothing`.

# Validation

  - A per-member `w_prev` has one entry per member of `w`, else a `DimensionMismatch` is raised.

# Returns

  - `VecNum_VecVecNum`: The start weights, `w0` of the fold's [`HeldWeightsResult`](@ref).

# Related

  - [`held_weights_result`](@ref)
  - [`HeldWeightsResult`](@ref)
  - [`previous_weights`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
"""
function held_start_weights(::OptimisationSuccess, w::VecNum, ::Any)
    return w
end
function held_start_weights(::OptimisationFailure, w::VecNum, ::Nothing)
    return w
end
function held_start_weights(::OptimisationFailure, ::VecNum, w_prev::VecNum)
    return w_prev
end
function held_start_weights(retcode::OptimisationReturnCode, w::VecVecNum, w_prev)
    return held_start_weights(fill(retcode, length(w)), w, w_prev)
end
function held_start_weights(::VecOptRetCode, w::VecVecNum, ::Nothing)
    return w
end
function held_start_weights(retcode::VecOptRetCode, w::VecVecNum, w_prev::VecNum)
    return held_start_weights(retcode, w, fill(w_prev, length(w)))
end
function held_start_weights(retcode::VecOptRetCode, w::VecVecNum, w_prev::VecVecNum)
    @argcheck(length(w_prev) == length(w),
              DimensionMismatch("`length(w_prev) == length(w)` must hold.\nlength(w_prev) => $(length(w_prev))\nlength(w) => $(length(w))"))
    return [isa(rc, OptimisationSuccess) ? wi : wp
            for (rc, wi, wp) in zip(retcode, w, w_prev)]
end

export PredictionResult, MultiPeriodPredictionResult, PopulationPredictionResult,
       PredictionReturnsResult, predict, sort_by_measure
public previous_weights
