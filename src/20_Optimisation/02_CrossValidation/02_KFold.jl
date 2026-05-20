"""
$(DocStringExtensions.TYPEDEF)

Implements non-sequential k-fold cross-validation with optional purging and embargoing of training samples.

# Fields

  - `n`: Number of folds to split the data into.
  - `purged_size`: Number of observations to exclude from the start/end of each train set adjacent to a test set.
  - `embargo_size`: Number of observations to exclude from the start of each train set after a test set.

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
    n
    purged_size
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

  - `train_idx`: Vector of training index ranges for each fold.
  - `test_idx`: Vector of testing index ranges for each fold.

# Related

  - [`KFold`](@ref)
  - [`NonSequentialCrossValidationResult`](@ref)
  - [`n_splits`](@ref)
"""
@concrete struct KFoldResult <: NonSequentialCrossValidationResult
    train_idx
    test_idx
    function KFoldResult(train_idx::VecVecInt, test_idx::VecVecInt)
        @argcheck(!isempty(train_idx))
        @argcheck(!isempty(test_idx))
        @argcheck(length(train_idx) == length(test_idx))
        return new{typeof(train_idx), typeof(test_idx)}(train_idx, test_idx)
    end
end
function KFoldResult(; train_idx::VecVecInt, test_idx::VecVecInt)::KFoldResult
    return KFoldResult(train_idx, test_idx)
end
"""
    Base.split(kf::KFold, rd::ReturnsResult) -> KFoldResult

Split the returns data `rd` into `n` non-overlapping folds using k-fold cross-validation
with optional purging and embargoing.

# Arguments

  - `kf::KFold`: K-fold cross-validation estimator.
  - `rd::ReturnsResult`: Returns data to split.

# Returns

  - `KFoldResult`: Result containing train and test indices for each fold.

# Related

  - [`KFold`](@ref)
  - [`KFoldResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(kf::KFold, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n, purged_size, embargo_size) = kf
    idx = 1:T
    min_fold_size = div(T, n)
    @argcheck(purged_size + embargo_size < min_fold_size)
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
function n_splits(kf::KFold, rd::ReturnsResult)
    return min(kf.n, size(rd.X, 1))
end

export KFold, KFoldResult
