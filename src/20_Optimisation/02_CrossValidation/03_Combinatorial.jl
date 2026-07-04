"""
$(DocStringExtensions.TYPEDEF)

Implements combinatorial non-sequential cross-validation with purging and embargoing, allowing for all possible combinations of test folds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CombinatorialCrossValidation(;
        n_folds::Integer = 10,
        n_test_folds::Integer = 8,
        purged_size::Integer = 0,
        embargo_size::Integer = 0,
        warn_comb::Integer = 100_000,
    ) -> CombinatorialCrossValidation

Keyword arguments correspond to the struct's fields.

## Validation

  - `n_folds` must be non-empty, greater than zero, and finite.
  - `n_test_folds` must be non-empty, greater than zero, and finite.
  - `purged_size` and `embargo_size` must be non-empty and finite.
  - Warns if the number of combinations exceeds `warn_comb`.

# Examples

```jldoctest
julia> CombinatorialCrossValidation(; n_folds = 10, n_test_folds = 8, purged_size = 2,
                                    embargo_size = 1)
CombinatorialCrossValidation
       n_folds ┼ Int64: 10
  n_test_folds ┼ Int64: 8
   purged_size ┼ Int64: 2
  embargo_size ┴ Int64: 1
```

# Related

  - [`NonSequentialCrossValidationEstimator`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
  - [`n_splits`](@ref)
"""
@concrete struct CombinatorialCrossValidation <: NonSequentialCrossValidationEstimator
    """
    $(field_dict[:n_folds])
    """
    n_folds
    """
    $(field_dict[:n_test_folds])
    """
    n_test_folds
    """
    $(field_dict[:purged_size])
    """
    purged_size
    """
    $(field_dict[:embargo_size])
    """
    embargo_size
    function CombinatorialCrossValidation(n_folds::Integer, n_test_folds::Integer,
                                          purged_size::Integer, embargo_size::Integer,
                                          warn_comb::Integer = 100_000)
        assert_nonempty_gt0_finite_val(n_folds, :n_folds)
        assert_nonempty_gt0_finite_val(n_test_folds, :n_test_folds)
        assert_nonempty_finite_val(purged_size, :purged_size)
        assert_nonempty_finite_val(embargo_size, :embargo_size)
        if binomial(n_folds, n_test_folds) > warn_comb
            @warn("The number of splits for `n_folds = $n_folds` and `n_test_folds = $n_test_folds` is `$(binomial(n_folds, n_test_folds))`, which may be computationally expensive. The number of combinations should typically be between 10^1 to 10^4 for statistical power. Such a large number of combinations may lead to long computation times and memory issues. Consider reducing `n_folds` or shifting `n_test_folds` further away from being equal to `div(n_folds, 2) = $(div(n_folds, 2))`.")
        end
        return new{typeof(n_folds), typeof(n_test_folds), typeof(purged_size),
                   typeof(embargo_size)}(n_folds, n_test_folds, purged_size, embargo_size)
    end
end
function CombinatorialCrossValidation(; n_folds::Integer = 10, n_test_folds::Integer = 8,
                                      purged_size::Integer = 0, embargo_size::Integer = 0,
                                      warn_comb::Integer = 100_000)::CombinatorialCrossValidation
    return CombinatorialCrossValidation(n_folds, n_test_folds, purged_size, embargo_size,
                                        warn_comb)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`CombinatorialCrossValidation`](@ref) after splitting data into combinatorial training and testing folds.

Stores the train index vectors, nested test index vectors (one per path), and a matrix of path IDs mapping folds to paths.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CombinatorialCrossValidationResult(;
        train_idx::VecVecInt,
        test_idx::VecVecVecInt,
        path_ids::AbstractMatrix{<:Integer}
    ) -> CombinatorialCrossValidationResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(train_idx)`.
  - `!isempty(test_idx)`.
  - `!isempty(path_ids)`.
  - `length(train_idx) == length(test_idx) == size(path_ids, 2)`.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`NonSequentialCrossValidationResult`](@ref)
  - [`n_splits`](@ref)
"""
@concrete struct CombinatorialCrossValidationResult <: NonSequentialCrossValidationResult
    """
    $(field_dict[:train_idx])
    """
    train_idx
    """
    $(field_dict[:test_idx])
    """
    test_idx
    """
    $(field_dict[:path_ids])
    """
    path_ids
    function CombinatorialCrossValidationResult(train_idx::VecVecInt,
                                                test_idx::VecVecVecInt,
                                                path_ids::AbstractMatrix{<:Integer})
        @argcheck(!isempty(train_idx), IsEmptyError("train_idx cannot be empty"))
        @argcheck(!isempty(test_idx), IsEmptyError("test_idx cannot be empty"))
        @argcheck(!isempty(path_ids), IsEmptyError("path_ids cannot be empty"))
        @argcheck(length(train_idx) == length(test_idx) == size(path_ids, 2),
                  DimensionMismatch("train_idx ($(length(train_idx))), test_idx ($(length(test_idx))), and path_ids columns ($(size(path_ids, 2))) must all match"))
        return new{typeof(train_idx), typeof(test_idx), typeof(path_ids)}(train_idx,
                                                                          test_idx,
                                                                          path_ids)
    end
end
function CombinatorialCrossValidationResult(; train_idx::VecVecInt, test_idx::VecVecVecInt,
                                            path_ids::AbstractMatrix{<:Integer})::CombinatorialCrossValidationResult
    return CombinatorialCrossValidationResult(train_idx, test_idx, path_ids)
end
"""
    const CombCVER = Union{<:CombinatorialCrossValidation,
                           <:CombinatorialCrossValidationResult}

Alias for a combinatorial cross-validation estimator or result.

Matches either a [`CombinatorialCrossValidation`](@ref) estimator or a [`CombinatorialCrossValidationResult`](@ref).

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
"""
const CombCVER = Union{<:CombinatorialCrossValidation, <:CombinatorialCrossValidationResult}
function n_splits(n_folds::Integer, n_test_folds::Integer)
    return binomial(n_folds, n_test_folds)
end
function n_splits(ccv::CombinatorialCrossValidation)
    return n_splits(ccv.n_folds, ccv.n_test_folds)
end
"""
    n_test_paths(n_folds, n_test_folds)

Compute the number of test paths in combinatorial cross-validation.

Returns the number of unique recombined test paths from `n_folds` folds choosing `n_test_folds` test folds. Also accepts a `CombinatorialCrossValidation` object directly.

# Arguments

  - `n_folds`: Total number of folds.
  - `n_test_folds`: Number of test folds per combination.

# Returns

  - Integer number of test paths.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`recombined_paths`](@ref)
"""
function n_test_paths(n_folds::Integer, n_test_folds::Integer)
    return div(n_splits(n_folds, n_test_folds) * n_test_folds, n_folds)
end
function n_test_paths(ccv::CombinatorialCrossValidation)
    return div(n_splits(ccv) * ccv.n_test_folds, ccv.n_folds)
end
"""
    average_train_size(T, n_folds, n_test_folds)

Compute the average training set size for combinatorial cross-validation.

# Arguments

  - `T`: Total number of observations.
  - `n_folds`: Total number of folds.
  - `n_test_folds`: Number of test folds per combination.

# Returns

  - Average number of training observations per fold.

# Related

  - [`CombinatorialCrossValidation`](@ref)
"""
function average_train_size(T::Integer, n_folds::Integer, n_test_folds::Integer)
    return T / n_folds * (n_folds - n_test_folds)
end
function average_train_size(ccv::CombinatorialCrossValidation, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n_folds, n_test_folds) = ccv
    return average_train_size(T, n_folds, n_test_folds)
end
"""
    test_set_index(ccv)

Generate all test set index combinations for combinatorial cross-validation.

Returns a vector of test fold index combinations for `ccv`.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - Vector of test index combinations.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`binary_train_test_sets`](@ref)
"""
function test_set_index(ccv::CombinatorialCrossValidation)
    return collect(Combinatorics.combinations(1:(ccv.n_folds), ccv.n_test_folds))
end
"""
    binary_train_test_sets(ccv)

Generate binary train/test set assignment matrices for combinatorial cross-validation.

Returns a matrix indicating which samples are in train (0) and test (1) sets for each combination.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - Binary train/test assignment matrix.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`test_set_index`](@ref)
"""
function binary_train_test_sets(ccv::CombinatorialCrossValidation)
    n_folds = ccv.n_folds
    num_splits = n_splits(ccv)
    folds_train_test = falses(n_folds, num_splits)
    test_set_idx = test_set_index(ccv)
    for (i, idx) in enumerate(test_set_idx)
        folds_train_test[idx, i] .= true
    end
    return folds_train_test
end
"""
    recombined_paths(ccv)

Generate the recombined test paths for combinatorial cross-validation.

Returns a vector of vectors representing the recombined test paths — sequences of test fold indices that together cover the entire dataset.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - Vector of recombined path index vectors.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`n_test_paths`](@ref)
"""
function recombined_paths(ccv::CombinatorialCrossValidation)
    bidx = binary_train_test_sets(ccv)
    out = zeros(Int, size(bidx, 1), n_test_paths(ccv))
    for i in axes(bidx, 1)
        tmp = view(bidx, i, :)
        out[i, :] .= findall(tmp)
    end
    return out
end
"""
    get_path_ids(ccv)

Get path identifiers for each test combination in combinatorial cross-validation.

Returns the path assignment for each test combination, mapping combinations to their recombined paths.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - Vector of path IDs.

# Related

  - [`recombined_paths`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function get_path_ids(ccv::CombinatorialCrossValidation)
    rcp = recombined_paths(ccv)
    num_splits = n_splits(ccv)
    ids = zeros(Int, ccv.n_test_folds, num_splits)
    for j in axes(ids, 2)
        inds = findall(x -> x == j, rcp)
        for i in axes(ids, 1)
            ids[end - i + 1, j] = inds[i][2]
        end
    end
    return ids
end
"""
    Base.split(ccv::CombinatorialCrossValidation, rd::ReturnsResult) -> CombinatorialCrossValidationResult

Split the returns data `rd` into all possible combinations of training and test folds using
combinatorial cross-validation with optional purging and embargoing.

# Arguments

  - `ccv::CombinatorialCrossValidation`: Combinatorial cross-validation estimator.
  - `rd::ReturnsResult`: Returns data to split.

# Returns

  - `CombinatorialCrossValidationResult`: Result containing train indices, nested test index
    vectors (one per path), and a matrix of path IDs mapping folds to paths.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(ccv::CombinatorialCrossValidation, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n_folds, n_test_folds, purged_size, embargo_size) = ccv
    min_fold_size = div(T, n_folds)
    @argcheck(purged_size + embargo_size < min_fold_size,
              DomainError(purged_size + embargo_size,
                          "purged_size + embargo_size ($(purged_size + embargo_size)) must be less than the minimum fold size ($min_fold_size)"))
    fold_idx_num = div.(0:(T - 1), min_fold_size)
    fold_idx_num[fold_idx_num .== n_folds] .= n_folds - 1
    fold_idx_num .+= 1
    num_splits = n_splits(ccv)
    test_set_idx = test_set_index(ccv)
    rcp = recombined_paths(ccv)
    train_test_idx = zeros(typeof(T), T, num_splits)
    for i in 1:num_splits
        train_test_idx[reduce(vcat, [findall(x -> x == j, fold_idx_num) for j in test_set_idx[i]]), i] .= one(num_splits)
    end
    dif = diff(train_test_idx; dims = 1)
    before_idx = findall(x -> x == 1, dif)
    before_idx_1 = getindex.(getindex.(before_idx, 1))
    before_idx_2 = getindex.(getindex.(before_idx, 2))
    for i in 0:(purged_size - 1)
        j = map(x -> max(one(x), x - i), before_idx_1)
        for (j, k) in zip(j, before_idx_2)
            train_test_idx[j, k] = -one(num_splits)
        end
    end
    after_idx = findall(x -> x == -1, dif)
    after_idx_1 = getindex.(getindex.(after_idx, 1))
    after_idx_2 = getindex.(getindex.(after_idx, 2))
    for i in 1:(purged_size + embargo_size)
        j = map(x -> min(T, x + i), after_idx_1)
        for (j, k) in zip(j, after_idx_2)
            train_test_idx[j, k] = -one(num_splits)
        end
    end
    fold_index = [findall(x -> x == i, fold_idx_num) for i in 1:n_folds]
    train_idx = Vector{Vector{typeof(T)}}(undef, num_splits)
    test_idx_list = Vector{Vector{Vector{typeof(T)}}}(undef, num_splits)
    for i in 1:num_splits
        train_idx[i] = findall(x -> x == zero(x), view(train_test_idx, :, i))
        test_idx_list[i] = sort!([fold_index[j[1]] for j in findall(x -> x == i, rcp)];
                                 by = x -> x[1])
    end
    path_ids = zeros(Int, n_test_folds, num_splits)
    for j in axes(path_ids, 2)
        inds = findall(x -> x == j, rcp)
        for i in axes(path_ids, 1)
            path_ids[end - i + 1, j] = inds[i][2]
        end
    end
    return CombinatorialCrossValidationResult(; train_idx = train_idx,
                                              test_idx = test_idx_list, path_ids = path_ids)
end
"""
    optimal_number_folds(T::Integer, target_train_size::Integer,
                         target_n_test_paths::Integer; train_size_w::Number = 1,
                         n_test_paths_w::Number = 1, maxval::Number = 1e5) -> Tuple{Int, Int}

Find the optimal `(n_folds, n_test_folds)` pair for combinatorial cross-validation by minimising a weighted cost that balances the average training size against the number of test paths.

# Mathematical definition

The cost function for a candidate `(n_folds, n_test_folds)` pair is:

```math
\\begin{align}
\\text{cost} &= w_{\\text{ntp}} \\frac{|P(n,k) - P^*|}{P^*} + w_{\\text{tr}} \\frac{|\\bar{T}(n,k) - T^*|}{T^*}\\,.
\\end{align}
```

Where:

  - ``\\text{cost}``: Weighted cost for the candidate fold configuration.
  - ``w_{\\text{ntp}}``: Weight on the test-paths component.
  - ``w_{\\text{tr}}``: Weight on the training-size component.
  - ``P(n,k)``: Number of test paths for ``n`` folds and ``k`` test folds.
  - ``\\bar{T}(n,k)``: Average training size for ``n`` folds and ``k`` test folds.
  - ``P^*``: Target number of test paths (`target_n_test_paths`).
  - ``T^*``: Target training size (`target_train_size`).

# Arguments

  - `T`: Total number of observations in the dataset.
  - `target_train_size`: Desired average number of observations in each training set.
  - `target_n_test_paths`: Desired number of recombined test paths.
  - `train_size_w`: Weight applied to the training-size component of the cost (default `1`).
  - `n_test_paths_w`: Weight applied to the test-paths component of the cost (default `1`).
  - `maxval`: Early-exit threshold; a fold configuration whose cost exceeds `maxval` prunes subsequent higher `n_test_folds` values (default `1e5`).

# Returns

  - `Tuple{Int, Int}`: The optimal `(n_folds, n_test_folds)` pair minimising the weighted cost. Returns `(0, 0)` when no valid configuration is found.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`n_test_paths`](@ref)
  - [`average_train_size`](@ref)
"""
function optimal_number_folds(T::Integer, target_train_size::Integer,
                              target_n_test_paths::Integer; train_size_w::Number = 1,
                              n_test_paths_w::Number = 1, maxval::Number = 1e5)
    function _cost(x::Integer, y::Integer)
        return n_test_paths_w * abs(n_test_paths(x, y) - target_n_test_paths) /
               target_n_test_paths +
               train_size_w * abs(average_train_size(T, x, y) - target_train_size) /
               target_train_size
    end
    min_cost = typemax(promote_type(typeof(train_size_w), typeof(n_test_paths_w),
                                    typeof(maxval)))
    n_folds_opt = 0
    n_test_folds_opt = 0
    for n_folds in 3:(T + 1)
        i = nothing
        for n_test_folds in 2:n_folds
            if !(isnothing(i) || n_folds - n_test_folds <= i)
                continue
            end
            cost = _cost(n_folds, n_test_folds)
            if cost < min_cost
                min_cost = cost
                n_folds_opt = n_folds
                n_test_folds_opt = n_test_folds
            end
            if isnothing(i) && cost > maxval
                i = n_test_folds
            end
        end
    end
    return n_folds_opt, n_test_folds_opt
end
function sort_predictions!(res::CombinatorialCrossValidationResult,
                           predictions::VecVecPredRes)
    path_ids = res.path_ids
    sorted_preds = [sizehint!(Vector{PredictionResult}(undef, 0),
                              count(x -> x == i, path_ids)) for i in 1:maximum(path_ids)]
    for (j, prediction) in enumerate(predictions)
        for (i, pred) in enumerate(prediction)
            push!(sorted_preds[path_ids[i, j]], pred)
        end
    end
    return [MultiPeriodPredictionResult(; pred = pred, id = i)
            for (i, pred) in enumerate(sorted_preds)]
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         cv::CombCVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    predictions = parallel_folds(length(train_idx), ex; ElT = Vector{PredictionResult}) do i
        return fit_and_predict(opt, rd; train_idx = train_idx[i], test_idx = test_idx[i],
                               cols = cols)
    end
    return PopulationPredictionResult(; pred = sort_predictions!(cv_res, predictions))
end
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                         cv::CombCVER;
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    cv_res = split(cv, rd)
    test_idx = cv_res.test_idx
    predictions = parallel_folds(length(test_idx), ex; ElT = Vector{PredictionResult}) do i
        return StatsAPI.predict(res, rd, test_idx[i])
    end
    return PopulationPredictionResult(; pred = sort_predictions!(cv_res, predictions))
end

export CombinatorialCrossValidation, CombinatorialCrossValidationResult,
       optimal_number_folds
