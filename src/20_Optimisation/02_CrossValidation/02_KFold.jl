"""
$(DocStringExtensions.TYPEDEF)

Implements non-sequential k-fold cross-validation with optional purging and embargoing of training samples.

The observations are cut into `n` consecutive folds. Each fold is the test set of one split, and the remaining folds form that split's training set. `purged_size` drops the last `purged_size` rows of the training fold that precedes the test fold, which removes the rows whose labels overlap the test period. `embargo_size` widens the gap on the other side: the first `purged_size + embargo_size` rows of the training fold that follows the test fold are dropped, which removes the rows whose features are contaminated by the test period. Both default to `0`, which gives the plain k-fold scheme.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KFold(;
        n::Integer = 5,
        purged_size::Integer = 0,
        embargo_size::Integer = 0,
        wd::Option{<:AbstractWeightDrift} = nothing,
        store_weight_path::Bool = false,
    ) -> KFold

Keyword arguments correspond to the struct's fields.

## Weight drift

`wd` is the Weight Drift of the scheme, and `nothing` is the library's original behaviour: a fold's return series is `X * w` net of fees, read at the target weights of that fold. A [`SelfFinancingDrift`](@ref) reads the series as the wealth ratio of the drifted holdings instead, and the fold carries a [`HeldWeightsResult`](@ref). `store_weight_path` makes the fold store the weight path it computed, which a reader otherwise rebuilds on demand.

A k-fold enumeration is not a timeline, so this scheme carries no Previous-Weights Source. Each of its folds is independent of the others, and no fold has a fold behind it to inherit weights from.

## Validation

  - `n` must be non-empty, greater than zero, and finite.
  - `purged_size` and `embargo_size` must be non-empty and finite.
  - [`Base.split`](@ref) additionally checks `purged_size + embargo_size < div(T, n)`, because a gap as wide as the smallest fold would empty a training fold.

# Examples

```jldoctest
julia> KFold(; n = 5, purged_size = 7, embargo_size = 11)
KFold
                  n ┼ Int64: 5
        purged_size ┼ Int64: 7
       embargo_size ┼ Int64: 11
                 wd ┼ nothing
  store_weight_path ┴ Bool: false
```

# Related

  - [`cross_val_predict`](@ref)
  - [`search_cross_validation`](@ref)
  - [`NonSequentialCrossValidationEstimator`](@ref)
  - [`KFoldResult`](@ref)
  - [`n_splits`](@ref)

# References

  - $(ref_dict[:lopezdeprado2018]) Chapter 7.
  - $(ref_dict[:cajas2025]) Section 15.2.
"""
@concrete struct KFold <: NonSequentialCrossValidationEstimator
    """
    $(field_dict[:n_folds])
    """
    n
    """
    $(field_dict[:purged_size])
    """
    purged_size
    """
    $(field_dict[:embargo_size])
    """
    embargo_size
    """
    $(field_dict[:wd])
    """
    wd
    """
    $(field_dict[:store_weight_path])
    """
    store_weight_path
    function KFold(n::Integer, purged_size::Integer, embargo_size::Integer,
                   wd::Option{<:AbstractWeightDrift}, store_weight_path::Bool)
        assert_nonempty_gt0_finite_val(n, :n)
        assert_nonempty_finite_val(purged_size, :purged_size)
        assert_nonempty_finite_val(embargo_size, :embargo_size)
        return new{typeof(n), typeof(purged_size), typeof(embargo_size), typeof(wd),
                   typeof(store_weight_path)}(n, purged_size, embargo_size, wd,
                                              store_weight_path)
    end
end
function KFold(; n::Integer = 5, purged_size::Integer = 0, embargo_size::Integer = 0,
               wd::Option{<:AbstractWeightDrift} = nothing,
               store_weight_path::Bool = false)::KFold
    return KFold(n, purged_size, embargo_size, wd, store_weight_path)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`KFold`](@ref) after splitting data into training and testing folds.

Stores the train and test index vectors for each fold.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KFoldResult(; train_idx::VecVecInt, test_idx::VecVecInt) -> KFoldResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(train_idx)`.
  - `!isempty(test_idx)`.
  - `length(train_idx) == length(test_idx)`.

# Related

  - [`KFold`](@ref)
  - [`NonSequentialCrossValidationResult`](@ref)
  - [`n_splits`](@ref)
"""
@concrete struct KFoldResult <: NonSequentialCrossValidationResult
    """
    $(field_dict[:train_idx])
    """
    train_idx
    """
    $(field_dict[:test_idx])
    """
    test_idx
    function KFoldResult(train_idx::VecVecInt, test_idx::VecVecInt)
        @argcheck(!isempty(train_idx), IsEmptyError("train_idx cannot be empty"))
        @argcheck(!isempty(test_idx), IsEmptyError("test_idx cannot be empty"))
        @argcheck(length(train_idx) == length(test_idx),
                  DimensionMismatch("train_idx ($(length(train_idx))) must match test_idx ($(length(test_idx)))"))
        return new{typeof(train_idx), typeof(test_idx)}(train_idx, test_idx)
    end
end
function KFoldResult(; train_idx::VecVecInt, test_idx::VecVecInt)::KFoldResult
    return KFoldResult(train_idx, test_idx)
end
"""
    Base.split(kf::KFold, rd::Prices_RR) -> KFoldResult

Split the data `rd` into `n` non-overlapping folds using k-fold cross-validation
with optional purging and embargoing.

# Arguments

  - `kf::KFold`: K-fold cross-validation estimator.
  - `rd`: Returns-level or price-level data to split ([`Prices_RR`](@ref)).

# Validation

  - `purged_size + embargo_size < div(T, n)`, where `T` is the number of observations. A gap as wide as the smallest fold would empty a training fold.

# Returns

  - `KFoldResult`: Result containing train and test indices for each fold.

# Details

  - The folds are consecutive and cover every observation exactly once. `mod(T, n)` extra rows are given one each to the first `mod(T, n)` folds, so the fold sizes differ by at most one row.
  - For the split whose test set is fold `i`, the training folds are every fold other than `i`, concatenated in ascending order. Fold `i - 1` loses its last `purged_size` rows and fold `i + 1` loses its first `purged_size + embargo_size` rows.
  - The training indices of every split increase strictly, which is what [`assert_unshuffled_folds`](@ref) needs.

# Related

  - [`KFold`](@ref)
  - [`KFoldResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(kf::KFold, rd::Prices_RR)
    T = cv_nobs(rd)
    (; n, purged_size, embargo_size) = kf
    idx = 1:T
    min_fold_size = div(T, n)
    @argcheck(purged_size + embargo_size < min_fold_size,
              DomainError(purged_size + embargo_size,
                          "purged_size + embargo_size ($(purged_size + embargo_size)) must be less than the minimum fold size ($min_fold_size)"))
    fold_sizes = fill(min_fold_size, n)
    fold_sizes[1:(mod(T, n))] .+= one(eltype(fold_sizes))
    test_idx = Vector{typeof(idx)}(undef, 0)
    current = one(eltype(fold_sizes))
    for fold_size in fold_sizes
        start, stop = current, current + fold_size
        push!(test_idx, idx[start:(stop - 1)])
        current = stop
    end
    train_idx = Vector{Vector{typeof(T)}}(undef, 0)
    idx = 1:n
    for i in idx
        tidx = setdiff(idx, i)
        tmp_test_idx = Vector{typeof(idx)}(undef, 0)
        for j in tidx
            if j == i - 1
                push!(tmp_test_idx, test_idx[j][1:(end - purged_size)])
            elseif j == i + 1
                push!(tmp_test_idx, test_idx[j][(1 + purged_size + embargo_size):end])
            else
                push!(tmp_test_idx, test_idx[j])
            end
        end
        push!(train_idx, reduce(vcat, tmp_test_idx))
    end
    return KFoldResult(; train_idx = train_idx, test_idx = test_idx)
end
function n_splits(kf::KFold, rd::Prices_RR)
    return min(kf.n, cv_nobs(rd))
end

"""
    fold_evaluation(cv::KFold)

Read the evaluation switches of a [`KFold`](@ref).

The folds of this scheme are not a timeline, so it carries no Previous-Weights Source and the triple names `nothing` for it. There is no previous fold whose weights a fold of this scheme could inherit.

# Returns

  - `(; wd, pws, store_weight_path)`: The Weight Drift, the Previous-Weights Source, and the flag that stores a fold's weight path.

# Related

  - [`fold_evaluation`](@ref)
  - [`KFold`](@ref)
  - [`held_weights_drift`](@ref)
"""
function fold_evaluation(cv::KFold)
    return (; wd = cv.wd, pws = nothing, store_weight_path = cv.store_weight_path)
end
export KFold, KFoldResult
