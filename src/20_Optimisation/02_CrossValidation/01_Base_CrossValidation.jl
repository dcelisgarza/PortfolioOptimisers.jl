"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation estimators in `PortfolioOptimisers.jl`.

# Related

  - [`CrossValidationResult`](@ref)
  - [`OptimisationCrossValidationEstimator`](@ref)
  - [`NonOptimisationCrossValidationEstimator`](@ref)
"""
abstract type CrossValidationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation result types in `PortfolioOptimisers.jl`.

# Related

  - [`CrossValidationEstimator`](@ref)
  - [`OptimisationCrossValidationResult`](@ref)
"""
abstract type CrossValidationResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-validation algorithm types.
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
    const CVER = Union{<:CrossValidationEstimator, <:CrossValidationResult}

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
"""
abstract type SequentialCrossValidationEstimator <: OptimisationCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential optimisation cross-validation estimators. Non-
sequential schemes may produce randomly sampled or combinatorial folds.

# Related

  - [`OptimisationCrossValidationEstimator`](@ref)
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
"""
abstract type SequentialCrossValidationResult <: OptimisationCrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential optimisation cross-validation results.

# Related

  - [`OptimisationCrossValidationResult`](@ref)
"""
abstract type NonSequentialCrossValidationResult <: OptimisationCrossValidationResult end
"""
    const OptCVER

Union of all optimisation cross-validation estimators and results.
"""
const OptCVER = Union{<:OptimisationCrossValidationEstimator,
                      <:OptimisationCrossValidationResult}

"""
    const NonSeqCVER

Union of all non-sequential cross-validation estimators and results.
"""
const NonSeqCVER = Union{<:NonSequentialCrossValidationEstimator,
                         <:NonSequentialCrossValidationResult}
"""
    const SeqCVER

Union of all sequential cross-validation estimators and results.
"""
const SeqCVER = Union{<:SequentialCrossValidationEstimator,
                      <:SequentialCrossValidationResult}
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

Abstract supertype for sequential non-optimisation cross-validation estimators.
"""
abstract type NonOptimisationSequentialCrossValidationEstimator <:
              NonOptimisationCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential non-optimisation cross-validation estimators.
"""
abstract type NonOptimisationNonSequentialCrossValidationEstimator <:
              NonOptimisationCrossValidationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for result types produced by non-optimisation cross-validation
routines.
"""
abstract type NonOptimisationCrossValidationResult <: CrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for sequential non-optimisation cross-validation result types.
"""
abstract type NonOptimisationSequentialCrossValidationResult <:
              NonOptimisationCrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for non-sequential non-optimisation cross-validation result types.
"""
abstract type NonOptimisationNonSequentialCrossValidationResult <:
              NonOptimisationCrossValidationResult end
"""
$(DocStringExtensions.TYPEDEF)

Stores the portfolio returns data associated with a cross-validation prediction. Packages
asset returns, factor returns, benchmark returns, timestamps, and investment vehicle
information for use in prediction result types.

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
        _check_names_and_returns_matrix(nf, F, :nf, :F)
        if !isnothing(X) && !isnothing(F)
            if isa(X, VecNum)
                @argcheck(length(X) == size(F, 1), DimensionMismatch)
            else
                @argcheck(all(x -> length(x) == size(F, 1), X))
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
                throw(ArgumentError("If B is VecNum, X must be `VecNum`, and if B is `VecVecNum`, X must be `VecVecNum`"))
            end
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)), IsNothingError)
            if isa(X, VecNum)
                @argcheck(length(ts) == length(X), DimensionMismatch)
            elseif isa(X, VecVecNum)
                @argcheck(all(x -> length(x) == length(ts), X))
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1), DimensionMismatch)
            end
            if isa(B, VecNum)
                @argcheck(length(ts) == length(B), DimensionMismatch)
            elseif isa(B, VecVecNum)
                @argcheck(all(x -> length(x) == length(ts), B))
            end
        end
        if isa(iv, VecNum)
            @argcheck(isa(ivpa, Option{<:Number}))
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(length(iv) == length(X), DimensionMismatch)
        elseif isa(iv, VecVecNum)
            @argcheck(isa(ivpa, Option{<:VecNum}))
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

Abstract supertype for all prediction result types in `PortfolioOptimisers.jl`.

All concrete prediction result types from cross-validation should subtype `AbstractPredictionResult`.

# Related

  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
abstract type AbstractPredictionResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Stores the result of a single cross-validation fold prediction. Pairs an optimisation
result with the returns data from the test period.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`fit_predict`](@ref)
  - [`PredictionReturnsResult`](@ref)
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
    function PredictionResult(res::NonFiniteAllocationOptimisationResult,
                              rd::PredictionReturnsResult)
        return new{typeof(res), typeof(rd)}(res, rd)
    end
end
function PredictionResult(; res::NonFiniteAllocationOptimisationResult,
                          rd::PredictionReturnsResult)::PredictionResult
    return PredictionResult(res, rd)
end
"""
    const VecPredRes = AbstractVector{<:PredictionResult}

Alias for a vector of single-fold prediction results.

Represents a collection of [`PredictionResult`](@ref) objects from cross-validation folds.

# Related

  - [`PredictionResult`](@ref)
  - [`VecVecPredRes`](@ref)
"""
const VecPredRes = AbstractVector{<:PredictionResult}
"""
    const VecVecPredRes = AbstractVector{<:VecPredRes}

Alias for a vector of vectors of prediction results.

Represents the outer collection of cross-validation paths, where each inner vector contains prediction results from a single path.

# Related

  - [`VecPredRes`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
const VecVecPredRes = AbstractVector{<:VecPredRes}
"""
    _prediction_expected_risk(r, X; kwargs...)

Compute the expected risk for a prediction result's returns.

Internal helper that dispatches on whether `X` is a plain vector or a vector of vectors.

# Arguments

  - `r`: Risk measure.
  - `X`: Returns vector or vector of vectors.
  - `kwargs...`: Additional keyword arguments passed to [`expected_risk`](@ref).

# Returns

  - Expected risk value(s).
"""
function _prediction_expected_risk(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
    return r(X)
end
function _prediction_expected_risk(r::AbstractBaseRiskMeasure, X::VecVecNum; kwargs...)
    return [r(Xi) for Xi in X]
end
function expected_risk(r::AbstractBaseRiskMeasure, pred::PredictionResult; kwargs...)
    return _prediction_expected_risk(r, pred.rd.X, kwargs...)
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

# Fields

$(DocStringExtensions.FIELDS)

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
        rd = getfield.(pred, :rd)
        nx = rd[1].nx
        X = mapreduce_RetMtx(rd)
        nf = rd[1].nf
        F = isnothing(rd[1].F) ? nothing : mapreduce(x -> getproperty(x, :F), vcat, rd)
        nb = rd[1].nb
        B = isnothing(rd[1].B) ? nothing : mapreduce_RetMtx(rd, :B)
        ts = isnothing(rd[1].ts) ? nothing : mapreduce(x -> getproperty(x, :ts), vcat, rd)
        iv = isnothing(rd[1].iv) ? nothing : mapreduce(x -> getproperty(x, :iv), vcat, rd)
        ivpa = rd[1].ivpa
        mrd = PredictionReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nb, B = B,
                                      ts = ts, iv = iv, ivpa = ivpa)
        return new{typeof(pred), typeof(mrd), typeof(id)}(pred, mrd, id)
    end
end
function MultiPeriodPredictionResult(;
                                     pred::VecPredRes = Vector{PredictionResult}(undef, 0),
                                     id::Any = nothing)::MultiPeriodPredictionResult
    return MultiPeriodPredictionResult(pred, id)
end
"""
    const VecMPredRes = AbstractVector{<:MultiPeriodPredictionResult}

Alias for a vector of multi-period prediction results.

Represents a collection of [`MultiPeriodPredictionResult`](@ref) objects.

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`PredRes_MultiPredRes`](@ref)
"""
const VecMPredRes = AbstractVector{<:MultiPeriodPredictionResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`MultiPeriodPredictionResult`](@ref). Virtual properties `:res` and `:rd` broadcast over the inner `pred` vector, collecting per-fold results and relative drawdowns.
"""
function Base.getproperty(mpred::MultiPeriodPredictionResult, sym::Symbol)
    return if sym == :res
        getfield.(getfield(mpred, :pred), :res)
    elseif sym === :rd
        getfield.(getfield(mpred, :pred), :rd)
    else
        getfield(mpred, sym)
    end
end
function expected_risk(r::AbstractBaseRiskMeasure, mpred::MultiPeriodPredictionResult;
                       kwargs...)
    X = mpred.mrd.X
    return _prediction_expected_risk(r, X; kwargs...)
end
"""
    const PredRes_MultiPredRes = Union{<:PredictionResult, <:MultiPeriodPredictionResult}

Alias for a single-fold or multi-period prediction result.

Matches either a [`PredictionResult`](@ref) or a [`MultiPeriodPredictionResult`](@ref).

# Related

  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`VecPredRes_MultiPredRes`](@ref)
"""
const PredRes_MultiPredRes = Union{<:PredictionResult, <:MultiPeriodPredictionResult}
"""
    const VecPredRes_MultiPredRes = AbstractVector{<:PredRes_MultiPredRes}

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
                                    pred::VecPredRes_MultiPredRes = Vector{<:PredRes_MultiPredRes}(undef,
                                                                                                   0))::PopulationPredictionResult
    return PopulationPredictionResult(pred)
end
function expected_risk(r::AbstractBaseRiskMeasure, preds::VecMPredRes; kwargs...)
    return [expected_risk(r, pred; kwargs...) for pred in preds]
end
function expected_risk(r::AbstractBaseRiskMeasure, ppred::PopulationPredictionResult;
                       kwargs...)
    return expected_risk(r, ppred.pred; kwargs...)
end
"""
    sort_by_measure(ppred::PopulationPredictionResult, r::AbstractBaseRiskMeasure; kwargs...)

Sort the successful paths in a [`PopulationPredictionResult`](@ref) by their expected
risk under `r`. Paths where any fold returned a non-success retcode are excluded.

# Arguments

  - `ppred::PopulationPredictionResult`: Population prediction to sort.
  - `r::AbstractBaseRiskMeasure`: Risk measure used for ranking.

# Returns

  - `Vector{MultiPeriodPredictionResult}`: Sorted vector of successful path predictions.

# Related

  - [`PopulationPredictionResult`](@ref)
  - [`expected_risk`](@ref)
"""
function sort_by_measure(ppred::PopulationPredictionResult, r::AbstractBaseRiskMeasure;
                         kwargs...)
    pred = filter(x -> all(y -> isa(y.res.retcode, OptimisationSuccess), x.pred),
                  ppred.pred)
    return sort(pred; by = x -> expected_risk(r, x; kwargs...), rev = bigger_is_better(r))
end
"""
    quantile_by_measure(ppred::PopulationPredictionResult, r::AbstractBaseRiskMeasure, q::Real;
                        r_kwargs::NamedTuple = (;), q_kwargs::NamedTuple = (;))

Select the successful path in `ppred` whose expected risk under `r` is closest to the `q`-th quantile of the risk distribution across all successful paths.

# Arguments

  - `ppred::PopulationPredictionResult`: Population prediction result.
  - `r::AbstractBaseRiskMeasure`: Risk measure for computing path risks.
  - `q::Real`: Quantile level in `[0, 1]`.
  - `r_kwargs::NamedTuple = (;)`: Keyword arguments forwarded to `expected_risk`.
  - `q_kwargs::NamedTuple = (;)`: Keyword arguments forwarded to `Statistics.quantile`.

# Returns

  - [`MultiPeriodPredictionResult`](@ref): The path closest to the `q`-th quantile.

# Related

  - [`sort_by_measure`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`expected_risk`](@ref)
"""
function quantile_by_measure(ppred::PopulationPredictionResult, r::AbstractBaseRiskMeasure,
                             q::Real; r_kwargs::NamedTuple = (;),
                             q_kwargs::NamedTuple = (;))
    pred = filter(x -> all(y -> isa(y.res.retcode, OptimisationSuccess), x.pred),
                  ppred.pred)
    rks = [expected_risk(r, p; r_kwargs...) for p in pred]
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
    reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult, X)

Reconstruct a `PredictionReturnsResult` from an optimisation result and returns data.

Computes benchmark, investment vehicle, and per-asset allocation data from the optimisation result weights and the original returns data.

# Arguments

  - `res::NonFiniteAllocationOptimisationResult`: Fitted optimisation result.
  - `rd::ReturnsResult`: Original returns data.
  - `X`: Portfolio returns (vector or vector of vectors).

# Returns

  - [`PredictionReturnsResult`](@ref) with updated benchmark and investment vehicle data.

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`PredictionReturnsResult`](@ref)
"""
function reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                        X::VecNum)
    nb = rd.nb
    B = !isa(rd.B, MatNum) ? rd.B : rd.B * res.w
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        w = abs.(res.w)
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
                        X::VecVecNum)
    nb = rd.nb
    B = if isnothing(rd.B)
        nothing
    elseif isa(rd.B, VecNum)
        fill(rd.B, length(res.w))
    else
        [rd.B * w for w in res.w]
    end
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        w = [abs.(wi) for wi in res.w]
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
"""
function StatsAPI.predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)
    X = calc_net_returns(res, rd.X)
    rd = reconstruct_rd(res, rd, X)
    return PredictionResult(; res = res, rd = rd)
end
"""
    fit_predict(opt, rd::ReturnsResult)

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
                          test_idx::VecInt, cols = :)
    rdi = returns_result_view(rd, test_idx, cols)
    X = calc_net_returns(res, rdi.X)
    rdi = reconstruct_rd(res, rdi, X)
    return PredictionResult(; res = res, rd = rdi)
end
function StatsAPI.predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                          test_idxs::VecVecInt, cols = :)
    return [StatsAPI.predict(res, rd, test_idx, cols) for test_idx in test_idxs]
end
"""
    fit_and_predict(opt, rd::ReturnsResult, cv::NonSeqCVER; cols, ex, id) -> MultiPeriodPredictionResult
    fit_and_predict(opt, rd::ReturnsResult; train_idx, test_idx, cols) -> PredictionResult
    fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult; test_idx, cols) -> PredictionResult

Fit an optimisation estimator on training data and predict on test data using cross-validation.

The three-argument method (`opt`, `rd`, `cv`) performs full cross-validated prediction over all folds of `cv`.
The two-argument methods operate on a single pre-defined train/test split or on a pre-existing result.

# Arguments

  - `opt`: Optimisation estimator or an existing optimisation result.
  - `rd::ReturnsResult`: Full returns data.
  - `cv::NonSeqCVER`: Non-sequential cross-validation estimator (e.g. [`KFold`](@ref) or [`CombinatorialCrossValidation`](@ref)).
  - `train_idx::VecInt`: Training indices.
  - `test_idx`: Test indices (vector or vector of vectors).
  - `cols`: Column selector (default `:` for all assets).

# Returns

  - [`MultiPeriodPredictionResult`](@ref) or [`PredictionResult`](@ref).

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`optimise`](@ref)
  - [`KFold`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult;
                         test_idx::VecInt_VecVecInt, cols = :, kwargs...)
    return predict(res, rd, test_idx, cols)
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult;
                         train_idx::VecInt, test_idx::VecInt_VecVecInt, cols = :)
    rd_train = returns_result_view(rd, train_idx, cols)
    if !isa(cols, Colon)
        opt = opt_view(opt, cols, rd.X)
    end
    #! Add ability to do callbacks
    res = optimise(opt, rd_train)
    return predict(res, rd, test_idx, cols)
end
"""
    sort_predictions!(res::Union{test_idx, CrossValidationResult}, pred::VecPredRes) -> VecPredRes

Sort prediction results to match the order of test indices.

Reorders `predictions` so that they align with the original time ordering of `test_idx`.

# Arguments

  - `res`:

      + `::VecVecInt`: Vector of test index vectors.
      + `::CrossValidationResult`: Cross validation result object, uses the test indices stored in `res.test_idx`.

  - `pred`: Vector of prediction results.

# Returns

  - Sorted predictions vector.

# Related

  - [`fit_and_predict`](@ref)
  - [`path_fit_and_predict`](@ref)
"""
function sort_predictions!(test_idx::VecVecInt, predictions::VecPredRes)
    @argcheck(all(x -> allunique(x), test_idx), "Test indices must be unique.")
    idx = sortperm(test_idx; by = x -> x[1])
    return predictions[idx]
end
function sort_predictions!(res::CrossValidationResult, predictions::VecPredRes)
    return sort_predictions!(res.test_idx, predictions)
end
function fit_and_predict(opt::OptE_Opt, rd::ReturnsResult, cv::NonSeqCVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         id = nothing)
    @argcheck(!(hasproperty(cv, :shuffle) && cv.shuffle),
              "Cross validation estimator must not be shuffled.")
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    @argcheck(all(map(x -> x > zero(x), map(x -> diff(x), train_idx))),
              "Cross validation estimator must not be shuffled.")
    predictions = Vector{PredictionResult}(undef, length(train_idx))
    FLoops.@floop ex for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
        predictions[i] = fit_and_predict(opt, rd; train_idx = train, test_idx = test,
                                         cols = cols)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end

export PredictionResult, MultiPeriodPredictionResult, PopulationPredictionResult,
       PredictionReturnsResult, predict, fit_predict, sort_by_measure
