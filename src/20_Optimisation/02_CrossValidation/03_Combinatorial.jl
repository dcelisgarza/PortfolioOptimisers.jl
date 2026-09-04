"""
$(DocStringExtensions.TYPEDEF)

Implements combinatorial non-sequential cross-validation with purging and embargoing, allowing for all possible combinations of test folds.

The observations are cut into `n_folds` consecutive folds, and every combination of `n_test_folds` of them is one split's test set. The remaining folds form that split's training set, less the purged and embargoed rows on each side of every test block. Each fold is a test fold in the same number of splits, so the test blocks recombine into `n_test_paths` full backtest paths, each covering every observation exactly once.

# Mathematical definition

```math
\\begin{align}
S(n,\\, k) &= \\binom{n}{k} = \\frac{n!}{k!\\,(n-k)!}\\,, \\\\
\\varphi(n,\\, k) &= \\frac{k}{n} \\binom{n}{k} = \\frac{(n-1)!}{(k-1)!\\,(n-k)!}\\,, \\\\
\\bar{T}(n,\\, k) &= \\frac{T}{n} (n - k)\\,.
\\end{align}
```

Where:

  - ``S(n,\\, k)``: Number of splits, one per combination of test folds.
  - ``\\varphi(n,\\, k)``: Number of recombined test paths.
  - ``\\bar{T}(n,\\, k)``: Average number of training observations per split, before purging and embargoing.
  - ``n``: Number of folds, `n_folds`.
  - ``k``: Number of test folds per split, `n_test_folds`.
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CombinatorialCrossValidation(;
        n_folds::Integer = 10,
        n_test_folds::Integer = 2,
        purged_size::Integer = 0,
        embargo_size::Integer = 0,
        max_comb::Integer = 100_000,
    ) -> CombinatorialCrossValidation

Keyword arguments correspond to the struct's fields.

The default holds out `2` of `10` folds for testing. This gives `binomial(10, 2) = 45` splits, `9` recombined test paths, and an average training set of `80%` of the observations. Because `binomial(n, k) == binomial(n, n - k)`, a transposed pair such as `n_test_folds = 8` produces the same number of splits on a training set of only `20%` of the observations. Choose `n_test_folds` well below `div(n_folds, 2)`, or let [`optimal_number_folds`](@ref) choose the pair.

## Validation

  - `n_folds` must be non-empty, greater than zero, and finite.
  - `n_test_folds` must be non-empty, greater than zero, and finite.
  - `purged_size` and `embargo_size` must be non-empty and finite.
  - `binomial(n_folds, n_test_folds) <= max_comb`. `max_comb` is a constructor argument, not a field, so it bounds the split count at construction and is not carried on the estimator.
  - [`Base.split`](@ref) additionally checks `purged_size + embargo_size < div(T, n_folds)`, because a gap as wide as the smallest fold would empty a training fold.

# Examples

```jldoctest
julia> CombinatorialCrossValidation(; n_folds = 10, n_test_folds = 2, purged_size = 2,
                                    embargo_size = 1)
CombinatorialCrossValidation
       n_folds ┼ Int64: 10
  n_test_folds ┼ Int64: 2
   purged_size ┼ Int64: 2
  embargo_size ┴ Int64: 1
```

# Related

  - [`cross_val_predict`](@ref)
  - [`search_cross_validation`](@ref)
  - [`NonSequentialCrossValidationEstimator`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
  - [`n_splits`](@ref)
  - [`n_test_paths`](@ref)
  - [`average_train_size`](@ref)
  - [`optimal_number_folds`](@ref)

# References

  - $(ref_dict[:lopezdeprado2018]) Chapter 12.
  - $(ref_dict[:cajas2025]) Section 15.3, Equations 15.1 and 15.2.
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
                                          max_comb::Integer = 100_000)
        assert_nonempty_gt0_finite_val(n_folds, :n_folds)
        assert_nonempty_gt0_finite_val(n_test_folds, :n_test_folds)
        assert_nonempty_finite_val(purged_size, :purged_size)
        assert_nonempty_finite_val(embargo_size, :embargo_size)
        @argcheck(binomial(n_folds, n_test_folds) <= max_comb,
                  ArgumentError("The number of splits for `n_folds = $n_folds` and `n_test_folds = $n_test_folds` is `$(binomial(n_folds, n_test_folds))`, which is greater than the maximum allowed `$max_comb`. The number of combinations should typically be between 10^1 to 10^4 for statistical power. Such a large number of combinations may lead to long computation times and memory issues. Consider reducing `n_folds` or shifting `n_test_folds` further away from being equal to `div(n_folds, 2) = $(div(n_folds, 2))`."))

        return new{typeof(n_folds), typeof(n_test_folds), typeof(purged_size),
                   typeof(embargo_size)}(n_folds, n_test_folds, purged_size, embargo_size)
    end
end
function CombinatorialCrossValidation(; n_folds::Integer = 10, n_test_folds::Integer = 2,
                                      purged_size::Integer = 0, embargo_size::Integer = 0,
                                      max_comb::Integer = 100_000)::CombinatorialCrossValidation
    return CombinatorialCrossValidation(n_folds, n_test_folds, purged_size, embargo_size,
                                        max_comb)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`CombinatorialCrossValidation`](@ref) after splitting data into combinatorial training and testing folds.

Stores the train index vectors, the nested test index vectors, and a matrix of path identifiers.

Every field is indexed by split. `train_idx[j]` holds the training rows of split `j`, purged and embargoed. `test_idx[j]` holds that split's `n_test_folds` test blocks, sorted by first observation. Column `j` of `path_ids` names the path each of those blocks belongs to, in the same order, which is what recombines the blocks of all splits into full backtest paths.

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
  - [`get_path_ids`](@ref)
  - [`sort_predictions!`](@ref)
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
function n_splits(ccv::CombinatorialCrossValidation, ::Prices_RR)
    return n_splits(ccv)
end
"""
    n_test_paths(n_folds, n_test_folds)
    n_test_paths(ccv::CombinatorialCrossValidation)

Compute the number of test paths in combinatorial cross-validation.

Returns the number of unique recombined test paths from `n_folds` folds choosing `n_test_folds` test folds. Also accepts a `CombinatorialCrossValidation` object directly.

The count is `div(binomial(n_folds, n_test_folds) * n_test_folds, n_folds)`, which is the ``\\varphi(n,\\, k)`` of [`CombinatorialCrossValidation`](@ref). Each fold is a test fold in the same number of splits, so the test blocks recombine into exactly this many paths, and each path covers every observation once.

# Arguments

  - `n_folds`: Total number of folds.
  - `n_test_folds`: Number of test folds per combination.
  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration, which supplies both counts.

# Returns

  - Integer number of test paths.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`recombined_paths`](@ref)
  - [`average_train_size`](@ref)
  - [`optimal_number_folds`](@ref)
"""
function n_test_paths(n_folds::Integer, n_test_folds::Integer)
    return div(n_splits(n_folds, n_test_folds) * n_test_folds, n_folds)
end
function n_test_paths(ccv::CombinatorialCrossValidation)
    return div(n_splits(ccv) * ccv.n_test_folds, ccv.n_folds)
end
"""
    average_train_size(T, n_folds, n_test_folds)
    average_train_size(ccv::CombinatorialCrossValidation, rd::Prices_RR)

Compute the average training set size for combinatorial cross-validation.

The count is `T / n_folds * (n_folds - n_test_folds)`, taken before purging and embargoing, so the training set of a split with a non-zero `purged_size` or `embargo_size` is smaller than this.

# Arguments

  - `T`: Total number of observations.
  - `n_folds`: Total number of folds.
  - `n_test_folds`: Number of test folds per combination.
  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration, which supplies `n_folds` and `n_test_folds`.
  - `rd`: Returns-level or price-level data, which supplies `T` through [`cv_nobs`](@ref) ([`Prices_RR`](@ref)).

# Returns

  - Average number of training observations per split.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`n_test_paths`](@ref)
  - [`optimal_number_folds`](@ref)
  - [`cv_nobs`](@ref)
"""
function average_train_size(T::Integer, n_folds::Integer, n_test_folds::Integer)
    return T / n_folds * (n_folds - n_test_folds)
end
function average_train_size(ccv::CombinatorialCrossValidation, rd::Prices_RR)
    T = cv_nobs(rd)
    (; n_folds, n_test_folds) = ccv
    return average_train_size(T, n_folds, n_test_folds)
end
"""
    test_set_index(ccv)

Generate all test set index combinations for combinatorial cross-validation.

Each element is one split's test set, given as the `n_test_folds` fold indices it holds. The elements are fold indices, not observation indices, and the splits come in the order `Combinatorics.combinations` enumerates them.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - `Vector{Vector{Int}}` of length `n_splits`.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`binary_train_test_sets`](@ref)
"""
function test_set_index(ccv::CombinatorialCrossValidation)
    return collect(Combinatorics.combinations(1:(ccv.n_folds), ccv.n_test_folds))
end
"""
    binary_train_test_sets(ccv)

Generate the binary train/test fold assignment matrix for combinatorial cross-validation.

The rows are folds and the columns are splits. An entry is `true` when that fold is a test fold of that split, and `false` when it is a training fold. The assignment is over folds, not over observations.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - `BitMatrix` of size `(n_folds, n_splits)`.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`test_set_index`](@ref)
  - [`recombined_paths`](@ref)
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

The rows are folds and the columns are paths. Entry `(f, p)` is the index of the split that supplies fold `f` to path `p`. Reading one column therefore names one split per fold, and those blocks together cover every observation exactly once.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - `Matrix{Int}` of size `(n_folds, n_test_paths)`.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`n_test_paths`](@ref)
  - [`binary_train_test_sets`](@ref)
  - [`get_path_ids`](@ref)
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

Get path identifiers for each test fold of each split in combinatorial cross-validation.

This is the transpose of the view [`recombined_paths`](@ref) gives. The rows are the test folds of a split and the columns are splits. Entry `(m, j)` is the path that the `m`-th test block of split `j` belongs to, where the blocks of a split are counted in ascending observation order. This is the `path_ids` field of [`CombinatorialCrossValidationResult`](@ref), and it is what pairs each block of `test_idx[j]` with its path.

# Arguments

  - `ccv`: [`CombinatorialCrossValidation`](@ref) configuration.

# Returns

  - `Matrix{Int}` of size `(n_test_folds, n_splits)`.

# Related

  - [`recombined_paths`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
  - [`sort_predictions!`](@ref)
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
    Base.split(ccv::CombinatorialCrossValidation, rd::Prices_RR) -> CombinatorialCrossValidationResult

Split the data `rd` into all possible combinations of training and test folds using
combinatorial cross-validation with optional purging and embargoing.

# Arguments

  - `ccv::CombinatorialCrossValidation`: Combinatorial cross-validation estimator.
  - `rd`: Returns-level or price-level data to split ([`Prices_RR`](@ref)).

# Validation

  - `purged_size + embargo_size < div(T, n_folds)`, where `T` is the number of observations. A gap as wide as the smallest fold would empty a training fold.

# Returns

  - `CombinatorialCrossValidationResult`: Result containing train indices, nested test index
    vectors (one per path), and a matrix of path IDs mapping folds to paths.

# Details

  - The folds are consecutive and cover every observation exactly once. `mod(T, n_folds)` extra rows go to the last fold, so the fold sizes differ by at most `n_folds - 1` rows.
  - Every training row within `purged_size` rows before a test block, or within `purged_size + embargo_size` rows after one, is dropped from that split's training set.
  - Column `j` of `path_ids` names the path of each test block of split `j`, in the same order as `test_idx[j]`, which is sorted by first observation. Collecting the blocks of one path over all splits therefore reproduces the whole timeline in order.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`CombinatorialCrossValidationResult`](@ref)
  - [`n_splits`](@ref)
  - [`get_path_ids`](@ref)
  - [`cv_nobs`](@ref)
"""
function Base.split(ccv::CombinatorialCrossValidation, rd::Prices_RR)
    T = cv_nobs(rd)
    (; n_folds, purged_size, embargo_size) = ccv
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
    path_ids = get_path_ids(ccv)
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
  - `maxval`: Pruning threshold (default `1e5`). Within one `n_folds`, let `m` be the first `n_test_folds` whose cost exceeds `maxval`. The search then skips every `n_test_folds` in `(m, n_folds - m)` and resumes at the mirror value `n_folds - m`, because `binomial(n, k) == binomial(n, n - k)` makes the two ends of the range the cheap ones and the middle the expensive one.

# Returns

  - `Tuple{Int, Int}`: The optimal `(n_folds, n_test_folds)` pair minimising the weighted cost. Returns `(0, 0)` when no valid configuration is found.

# Details

  - The search walks `n_folds` from `3` to `T + 1` and `n_test_folds` from `2` to `n_folds`.
  - A candidate whose split count overflows `Int` is treated as infinitely expensive rather than raising. Such a configuration is unusable: [`CombinatorialCrossValidation`](@ref) refuses it through `max_comb` long before the count reaches that size. Raising `maxval` widens the search into that region, so the guard is what keeps a large `maxval` from failing.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`n_test_paths`](@ref)
  - [`average_train_size`](@ref)
"""
function optimal_number_folds(T::Integer, target_train_size::Integer,
                              target_n_test_paths::Integer; train_size_w::Number = 1,
                              n_test_paths_w::Number = 1, maxval::Number = 1e5)
    function _cost(x::Integer, y::Integer)
        # `binomial` raises rather than saturating, and a configuration whose split count
        # cannot be represented is unusable anyway, so it costs infinity instead of
        # failing the search. Only a `maxval` large enough to defeat the mirror pruning
        # reaches this branch.
        ntp = try
            n_test_paths(x, y)
        catch err
            if !(isa(err, OverflowError))
                rethrow()
            end
            nothing
        end
        if isnothing(ntp)
            return Inf
        end
        return n_test_paths_w * abs(ntp - target_n_test_paths) / target_n_test_paths +
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
function fit_and_predict(opt::OptE_TD, rd::ReturnsResult, cv::CombCVER; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    cv_res = split(cv, rd)
    (; train_idx, test_idx) = cv_res
    assert_unshuffled_folds(cv, train_idx)
    # A fold is a train/test split and `i` is its position in the split enumeration —
    # no ordering is imposed on time-dependent entries; the user keys them off the
    # fold's indices (ctx.train_idx[ctx.i] / ctx.test_idx[ctx.i]).
    predictions = fold_loop(opt, length(train_idx), ex, Vector{PredictionResult}; rd = rd,
                            train_idx = train_idx, test_idx = test_idx, cv = cv) do fold
        return fit_and_predict(fold.est, fold.rd; train_idx = fold.train,
                               test_idx = fold.test, cols = cols)
    end
    return PopulationPredictionResult(; pred = sort_predictions!(cv_res, predictions))
end
function fit_and_predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult,
                         cv::CombCVER;
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    cv_res = split(cv, rd)
    test_idx = cv_res.test_idx
    assert_unshuffled_folds(cv, cv_res.train_idx)
    predictions = parallel_folds(length(test_idx), ex, Vector{PredictionResult}) do i
        return StatsAPI.predict(res, rd, test_idx[i])
    end
    return PopulationPredictionResult(; pred = sort_predictions!(cv_res, predictions))
end

export CombinatorialCrossValidation, CombinatorialCrossValidationResult,
       optimal_number_folds
