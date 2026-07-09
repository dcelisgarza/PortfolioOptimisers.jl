"""
$(DocStringExtensions.TYPEDEF)

Implements non-sequential k-fold cross-validation with optional purging and embargoing of training samples.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KFold(;
        n::Integer = 5,
        purged_size::Integer = 0,
        embargo_size::Integer = 0,
    ) -> KFold

Keyword arguments correspond to the struct's fields.

## Validation

  - `n` must be non-empty, greater than zero, and finite.
  - `purged_size` and `embargo_size` must be non-empty and finite.

# Examples

```jldoctest
julia> KFold(; n = 5, purged_size = 7, embargo_size = 11)
KFold
             n ┼ Int64: 5
   purged_size ┼ Int64: 7
  embargo_size ┴ Int64: 11
```

# Related

  - [`NonSequentialCrossValidationEstimator`](@ref)
  - [`KFoldResult`](@ref)
  - [`n_splits`](@ref)
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
    function KFold(n::Integer, purged_size::Integer, embargo_size::Integer)
        assert_nonempty_gt0_finite_val(n, :n)
        assert_nonempty_finite_val(purged_size, :purged_size)
        assert_nonempty_finite_val(embargo_size, :embargo_size)
        return new{typeof(n), typeof(purged_size), typeof(embargo_size)}(n, purged_size,
                                                                         embargo_size)
    end
end
function KFold(; n::Integer = 5, purged_size::Integer = 0, embargo_size::Integer = 0)::KFold
    return KFold(n, purged_size, embargo_size)
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
    Base.split(kf::KFold, rd::Rd_Pr) -> KFoldResult

Split the data `rd` into `n` non-overlapping folds using k-fold cross-validation
with optional purging and embargoing.

# Arguments

  - `kf::KFold`: K-fold cross-validation estimator.
  - `rd`: Returns-level or price-level data to split ([`Rd_Pr`](@ref)).

# Returns

  - `KFoldResult`: Result containing train and test indices for each fold.

# Related

  - [`KFold`](@ref)
  - [`KFoldResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(kf::KFold, rd::Rd_Pr)
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
function n_splits(kf::KFold, rd::Rd_Pr)
    return min(kf.n, cv_nobs(rd))
end

export KFold, KFoldResult
