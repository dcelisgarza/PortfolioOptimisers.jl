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

# Related

  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`fit_predict`](@ref)
  - [`PredictionReturnsResult`](@ref)
  - [`rebuild_returns_result`](@ref)
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
        # Per-asset, so it cannot stack; reduced to the last fold to pair with the last
        # row of `iv`, which is what `predict_realised_vols` divides by it.
        ivpa = rd[end].ivpa
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
function expected_risk(r::BaseRM_VecBaseRM, preds::VecMPredRes; kwargs...)
    return [expected_risk(r, pred; kwargs...) for pred in preds]
end
function expected_risk(r::BaseRM_VecBaseRM, ppred::PopulationPredictionResult; kwargs...)
    return expected_risk(r, ppred.pred; kwargs...)
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
    reconstruct_rd(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult, X)

Reconstruct a `PredictionReturnsResult` from an optimisation result and returns data.

Computes benchmark, investment vehicle and per-asset allocation data from the optimisation result weights and the original returns data.

## No feature matrix

The fold does not collapse `rd.Z`. Only one weight vector is in scope here, which is not enough to collapse a *square* feature matrix — the second contraction needs every synthetic asset's weights at once — so a fold-side collapse could only serve one of the two shapes, and the two paths into the outer problem would disagree on what a square carrier means. [`rebuild_returns_result`](@ref) instead recomputes the collapse for the whole synthetic universe at once, from the original `rd.Z` and the fold weights it reaches through `pred[f].res.w`.

# Arguments

  - `res::NonFiniteAllocationOptimisationResult`: Fitted optimisation result.
  - `rd::ReturnsResult`: Original returns data.
  - `X`: Portfolio returns (vector or vector of vectors).

# Returns

  - [`PredictionReturnsResult`](@ref) with updated benchmark and investment vehicle data.

# Related

  - [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref)
  - [`PredictionReturnsResult`](@ref)
  - [`rebuild_returns_result`](@ref)
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
                          test_idx::VecInt, cols = :)
    rdi = port_opt_view(rd, test_idx, cols)
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
  - `rd::ReturnsResult`: FullMoment returns data.
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
    return StatsAPI.predict(res, rd, test_idx, cols)
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult;
                         train_idx::VecInt, test_idx::VecInt_VecVecInt, cols = :)
    rd_train = port_opt_view(rd, train_idx, cols)
    if !isa(cols, Colon)
        opt = port_opt_view(opt, cols, rd.X)
    end
    #! Add ability to do callbacks
    res = optimise(opt, rd_train)
    return StatsAPI.predict(res, rd, test_idx, cols)
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
"""
    cv_sequential_info(prev_w_flag::Bool, time_dep_flag::Bool)

Build the informational message emitted when a cross-validation run falls back to sequential
execution because the optimiser needs the previous fold's weights and/or is time dependent.
Shared by the [`run_folds`](@ref) walk fallback used in walk-forward and multiple-randomised
cross-validation.
"""
function cv_sequential_info(prev_w_flag::Bool)
    return "Running cross-validation sequentially because the optimiser must use the previous optimisation's weights (needs_previous_weights(opt) == $prev_w_flag). This is because somewhere within the optimisation estimator is contained at least one of the following:\n\t- Turnover and/or TurnoverEstimator,\n\t- WeightsTracking,\n\t- TurnoverRiskMeasure,\n\t- custom constraints which use asset weights,\n\t- custom objective penalties which use asset weights,\n\t- a time-dependent constraint whose entries need previous weights (e.g. a PreviousWeightsFunction).\nTo enable parallel processing please either mark the weights as fixed or remove the offending component(s). Time-dependent constraints alone do not force sequential processing."
end
"""
    parallel_folds(fit_fold, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult)

Run `n` cross-validation folds in parallel, filling `predictions[i] = fit_fold(i)` for `i in 1:n`
over executor `ex`. `ElT` is the per-fold result element type (a single [`PredictionResult`](@ref)
for time-ordered schemes, a `Vector{PredictionResult}` for the multi-path combinatorial scheme).

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
    run_folds(fit_fold, opt, n::Integer, ex::FLoops.Transducers.Executor,
              ::Type{ElT} = PredictionResult,
              ::Val{PW} = Val(needs_previous_weights(opt)))

Run `n` cross-validation folds, either in parallel or — when `opt` needs the previous fold's
weights (`needs_previous_weights`) — sequentially, emitting [`cv_sequential_info`](@ref).
`fit_fold(i, prev)` returns the prediction for fold `i`, where `prev` is `nothing` in parallel
mode or the previous fold's prediction in sequential mode; the caller uses `prev` to thread
previous weights into fold `i`. Time-dependent constraints alone do not force sequential
processing — their per-fold values are known upfront.

`ElT` is a *positional* `::Type{ElT}` argument for the reason given in
[`parallel_folds`](@ref). The previous-weights flag is a `Val` for the same reason applied
to a branch: as a run-time `Bool` the sequential branch is *inferred* even when it can
never run, and it reads an abstractly-typed `predictions[i - 1]`, which is a runtime
dispatch. As a type parameter the branch is eliminated.

# Related

  - [`parallel_folds`](@ref)
  - [`fold_loop`](@ref)
  - [`cv_sequential_info`](@ref)
  - [`fit_and_predict`](@ref)
"""
function run_folds(fit_fold, opt, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult,
                   ::Val{PW} = Val(needs_previous_weights(opt))) where {ElT, PW}
    if PW
        @info(cv_sequential_info(PW))
        predictions = Vector{ElT}(undef, n)
        for i in 1:n
            predictions[i] = fit_fold(i, i > 1 ? predictions[i - 1] : nothing)
        end
        return predictions
    end
    return parallel_folds(i -> fit_fold(i, nothing), n, ex, ElT)
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
    fold_loop(fit_fold, est, n::Integer, ex::FLoops.Transducers.Executor,
              ::Type{ElT} = PredictionResult; rd, train_idx, test_idx,
              path_id = nothing, time_ordered::Bool = true, fold_view = nothing)

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

`time_ordered` states whether the fold enumeration of the scheme is a timeline. `true`
routes through [`run_folds`](@ref), so an optimiser that needs the previous weights runs
sequentially. `false` routes through [`parallel_folds`](@ref), because a scheme whose
folds are not a timeline has no previous fold to thread. `ElT` is the per-fold result
element type: a single
[`PredictionResult`](@ref) for a time-ordered scheme, a `Vector{PredictionResult}` for the
multi-path combinatorial scheme. It is positional for the reason given in
[`parallel_folds`](@ref).

# Related

  - [`Fold`](@ref)
  - [`run_folds`](@ref)
  - [`parallel_folds`](@ref)
  - [`assert_unshuffled_folds`](@ref)
  - [`fit_and_predict`](@ref)
  - [`cross_val_predict`](@ref)
"""
function fold_loop(fit_fold, est, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult; rd, train_idx, test_idx,
                   path_id = nothing, time_ordered::Bool = true,
                   fold_view = nothing) where {ElT}
    td_flag = is_time_dependent(est)
    if td_flag
        assert_time_dependent_fold_count(est, n)
    end
    prev_w_flag = needs_previous_weights(est)
    function fold(i, prev)
        w_prev = isnothing(prev) ? nothing : prev.res.w
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
    # The previous-weights flag crosses into `run_folds` as a `Val`, not a `Bool`. As a
    # value it is a run-time argument, so the sequential branch is inferred even for an
    # optimiser that never takes it, and that branch reads an abstractly-typed
    # `predictions[i - 1]`, which is a runtime dispatch. As a type parameter the branch is
    # eliminated instead. See the ADR 0067 amendment.
    return if time_ordered
        run_folds(fold, est, n, ex, ElT, Val(prev_w_flag))
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
    predictions = fold_loop(opt, length(train_idx), ex; rd = rd, train_idx = train_idx,
                            test_idx = test_idx, time_ordered = false) do fold
        return fit_and_predict(fold.est, fold.rd; train_idx = fold.train,
                               test_idx = fold.test, cols = cols)
    end
    return MultiPeriodPredictionResult(; pred = predictions, id = id)
end

export PredictionResult, MultiPeriodPredictionResult, PopulationPredictionResult,
       PredictionReturnsResult, fit, predict, fit_predict, sort_by_measure
