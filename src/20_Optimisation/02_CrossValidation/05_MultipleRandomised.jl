"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for estimators that determine the size of each asset subset.

# Related

  - [`MultipleRandomised`](@ref)
"""
abstract type SubsetSizeEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for estimators that determine the number of random subsets to draw.

# Related

  - [`MultipleRandomised`](@ref)
"""
abstract type NumberSubsetsEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for estimators that determine the rolling window size.

# Related

  - [`MultipleRandomised`](@ref)
"""
abstract type WindowSizeEstimator <: AbstractEstimator end
"""
    const SubsetSizeEC

Union of [`SubsetSizeEstimator`](@ref) subtypes and plain functions that compute a
subset size from a returns dataset.
"""
const SubsetSizeEC = Union{<:SubsetSizeEstimator, <:Function}
"""
    const NumberSubsetsEC

Union of [`NumberSubsetsEstimator`](@ref) subtypes and plain functions that compute the
number of subsets from a returns dataset.
"""
const NumberSubsetsEC = Union{<:NumberSubsetsEstimator, <:Function}
"""
    const WindowSizeEC

Union of [`WindowSizeEstimator`](@ref) subtypes and plain functions that compute a
window size from a returns dataset.
"""
const WindowSizeEC = Union{<:WindowSizeEstimator, <:Function}
"""
    const SubsetSizeE

Union of a concrete subset-size value or an estimator/function for it.
"""
const SubsetSizeE = Union{<:Number, <:SubsetSizeEC}
"""
    const NumberSubsetsE

Union of a concrete number-of-subsets value or an estimator/function for it.
"""
const NumberSubsetsE = Union{<:Integer, <:NumberSubsetsEC}
"""
    const WindowSizeE

Union of a concrete window-size value or an estimator/function for it.
"""
const WindowSizeE = Union{<:Number, <:WindowSizeEC}
"""
$(DocStringExtensions.TYPEDEF)

Cross-validation scheme that draws multiple random asset subsets and applies a
walk-forward estimator to each. Each combination of a random asset subset and a set of
walk-forward folds forms one *path*.

# Fields

  - `cv::WalkForwardEstimator`: Walk-forward estimator applied within each asset subset.
  - `subset_size::SubsetSizeE`: Size of each asset subset (integer count, fraction of
    total, or callable).
  - `n_subsets::NumberSubsetsE`: Number of random subsets to draw.
  - `max_comb::Integer`: Maximum number of combinations to enumerate exactly. When the
    total number of combinations exceeds this limit an approximate sampling approach is
    used.
  - `window_size::Option{<:WindowSizeE}`: Optional rolling observation window. When set,
    each subset uses a randomly chosen contiguous window of this length.
  - `rng::Random.AbstractRNG`: Random number generator.
  - `seed::Option{<:Integer}`: Optional random seed.

# Constructors

    MultipleRandomised(
        cv::WalkForwardEstimator;
        subset_size::SubsetSizeE = 1,
        n_subsets::NumberSubsetsE = 2,
        max_comb::Integer = 1_000_000_000,
        window_size::Option{<:WindowSizeE} = nothing,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing
    ) -> MultipleRandomised

# Related

  - [`MultipleRandomisedResult`](@ref)
  - [`MultipleRandomisedResult`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
"""
@concrete struct MultipleRandomised <: NonOptimisationSequentialCrossValidationEstimator
    cv
    subset_size
    n_subsets
    max_comb
    window_size
    rng
    seed
    function MultipleRandomised(cv::WalkForwardEstimator, subset_size::SubsetSizeE,
                                n_subsets::NumberSubsetsE, max_comb::Integer,
                                window_size::Option{<:WindowSizeE}, rng::Random.AbstractRNG,
                                seed::Option{<:Integer})
        if isa(subset_size, Integer)
            assert_nonempty_nonneg_finite_val(subset_size - 1, "subset_size - 1")
        elseif isa(subset_size, AbstractFloat)
            @argcheck(0 < subset_size < 1)
        end
        if isa(n_subsets, Integer)
            assert_nonempty_nonneg_finite_val(n_subsets - 2, "n_subsets - 2")
        end
        assert_nonempty_gt0_finite_val(max_comb, :max_comb)
        if isa(window_size, Integer)
            assert_nonempty_nonneg_finite_val(window_size - 2, "window_size - 2")
        elseif isa(window_size, AbstractFloat)
            @argcheck(0 < window_size < 1)
        end
        return new{typeof(cv), typeof(subset_size), typeof(n_subsets), typeof(max_comb),
                   typeof(window_size), typeof(rng), typeof(seed)}(cv, subset_size,
                                                                   n_subsets, max_comb,
                                                                   window_size, rng, seed)
    end
end
function MultipleRandomised(cv::WalkForwardEstimator; subset_size::SubsetSizeE = 1,
                            n_subsets::NumberSubsetsE = 2,
                            max_comb::Integer = 1_000_000_000,
                            window_size::Option{<:WindowSizeE} = nothing,
                            rng::Random.AbstractRNG = Random.default_rng(),
                            seed::Option{<:Integer} = nothing)
    return MultipleRandomised(cv, subset_size, n_subsets, max_comb, window_size, rng, seed)
end
"""
$(DocStringExtensions.TYPEDEF)

Stores the split result produced by [`MultipleRandomised`](@ref). Contains the training,
test, and asset index sets for every fold across all random paths, along with a path
identifier for each fold.

# Fields

  - `train_idx::VecVecInt`: Training observation indices per fold.
  - `test_idx::VecVecInt`: Test observation indices per fold.
  - `asset_idx::VecVecInt`: Asset column indices per fold.
  - `path_ids::VecInt`: Path identifier for each fold.

# Related

  - [`MultipleRandomised`](@ref)
  - [`MultipleRandomised`](@ref)
"""
@concrete struct MultipleRandomisedResult <: NonOptimisationSequentialCrossValidationResult
    train_idx
    test_idx
    asset_idx
    path_ids
    function MultipleRandomisedResult(train_idx::VecVecInt, test_idx::VecVecInt,
                                      asset_idx::VecVecInt, path_ids::VecInt)
        @argcheck(!isempty(train_idx))
        @argcheck(!isempty(test_idx))
        @argcheck(!isempty(asset_idx))
        @argcheck(!isempty(path_ids))
        @argcheck(length(train_idx) ==
                  length(test_idx) ==
                  length(asset_idx) ==
                  length(path_ids))
        return new{typeof(train_idx), typeof(test_idx), typeof(asset_idx),
                   typeof(path_ids)}(train_idx, test_idx, asset_idx, path_ids)
    end
end
function MultipleRandomisedResult(; train_idx::VecVecInt, test_idx::VecVecInt,
                                  asset_idx::VecVecInt, path_ids::VecInt)
    return MultipleRandomisedResult(train_idx, test_idx, asset_idx, path_ids)
end
function n_splits(mre::MultipleRandomised, rd::ReturnsResult)
    if !isnothing(mre.window_size) && isa(mre.cv, DateWalkForward)
        throw(ArgumentError("when using a `DateWalkForward` with `window_size`, the number of splits cannot be determined before calling [`split`](@ref)."))
    end
    if !isnothing(mre.window_size)
        rd = returns_result_view(rd, 1:(mre.window_size), :)
    end
    return mre.n_subsets * n_splits(mre.cv, rd)
end
function n_splits(mrr::MultipleRandomisedResult)
    return length(mrr.path_ids)
end
"""
    const MRCVR = Union{<:MultipleRandomised, <:MultipleRandomisedResult}

Alias for a multiple-randomised cross-validation estimator or result.

Matches either a [`MultipleRandomised`](@ref) estimator or a [`MultipleRandomisedResult`](@ref).

# Related

  - [`MultipleRandomised`](@ref)
  - [`MultipleRandomisedResult`](@ref)
"""
const MRCVR = Union{<:MultipleRandomised, <:MultipleRandomisedResult}
"""
    combination_by_index(idx, N, k)

Return the `idx`-th combination of `k` items from `N` total items.

Internal helper for combinatorial path generation. Converts a lexicographic combination index to the actual combination elements.

# Arguments

  - `idx`: Combination index (1-based).
  - `N`: Total number of items.
  - `k`: Number of items in each combination.

# Returns

  - Vector of `k` item indices.

# Related

  - [`CombinatorialCrossValidation`](@ref)
  - [`sample_unique_assets`](@ref)
"""
function combination_by_index(idx::Integer, N::Integer, k::Integer)
    n_comb = binomial(N, k)
    @argcheck(0 < idx <= n_comb)
    combination = Vector{typeof(N)}(undef, k)
    remaining_rank = idx
    next_element = 1
    for i in 1:k
        remaining_slots = k - i
        x = next_element
        block_size = binomial(N - x, remaining_slots)
        while block_size < remaining_rank
            remaining_rank -= block_size
            x += 1
            block_size = binomial(N - x, remaining_slots)
        end
        combination[i] = x
        next_element = x + 1
    end
    return combination
end
"""
    sample_unique_assets(N, k, n_subsets; kwargs...)

Sample `n_subsets` unique asset subsets of size `k` from `N` assets.

Internal function used in multiple-randomised cross-validation to generate diverse asset subsets for resampling.

# Arguments

  - `N`: Total number of assets.
  - `k`: Subset size.
  - `n_subsets`: Number of unique subsets to sample.
  - `kwargs...`: Additional keyword arguments (e.g., random seed).

# Returns

  - Matrix of size `(k, n_subsets)` with asset indices.

# Related

  - [`MultipleRandomised`](@ref)
  - [`combination_by_index`](@ref)
"""
function sample_unique_assets(N::Integer, k::Integer, n_subsets::Integer;
                              max_comb::Integer = 1_000_000_000,
                              rng::Random.AbstractRNG = Random.default_rng(),
                              seed::Option{<:Integer} = nothing)
    assert_nonempty_nonneg_finite_val(N, :N)
    assert_nonempty_nonneg_finite_val(k, :k)
    @argcheck(k <= N)
    assert_nonempty_finite_val(n_subsets, :n_subsets)
    n_comb = binomial(N, k)
    @argcheck(n_subsets <= n_comb,
              "n_subsets = $n_subsets must not be greater than `binomial(assets, subset_size) = n_comb => binomial($N, $k) = $n_comb`.")
    if !isnothing(seed)
        Random.seed!(rng, seed)
    end
    subsets = Matrix{typeof(N)}(undef, k, n_subsets)
    if n_comb <= max_comb
        ranks = StatsBase.sample(rng, 1:n_comb, n_subsets; replace = false)
        @inbounds for (i, rank) in enumerate(ranks)
            subsets[:, i] .= combination_by_index(rank, N, k)
        end
    else
        @warn("The number of combinations for `subset_size = $k` and `N = $N` is `binomial(assets, subset_size) = n_comb => binomial($N, $k) = $n_comb`, which may be computationally expensive. We will use an approximate alternate approach. If you want the exact approach consider increasing `max_comb` or moving `subset_size` closer to `div(assets, 2) = $(div(N, 2))`.")
        for i in axes(subsets, 2)
            subsets[:, i] .= sort!(StatsBase.sample(rng, 1:N, k; replace = false))
        end
    end
    return subsets
end
"""
    get_subset_size(subset_size, rd, args...)

Get the actual asset subset size for multiple-randomised cross-validation.

Resolves the subset size from either an integer (direct count) or a fraction of the total assets.

# Arguments

  - `subset_size`: Integer or float subset size specification.
  - `rd`: Returns result or prior.
  - `args...`: Additional arguments.

# Returns

  - Integer subset size.

# Related

  - [`MultipleRandomised`](@ref)
  - [`sample_unique_assets`](@ref)
"""
function get_subset_size(subset_size::Integer, rd::Pr_RR, args...)
    @argcheck(subset_size <= size(rd.X, 2),
              "subset_size must not be greater than the number of assets")
    return subset_size
end
function get_subset_size(subset_size::AbstractFloat, rd::Pr_RR, args...)
    subset_size = max(round(Int, subset_size * size(rd.X, 2)), 1)
    return subset_size
end
function get_subset_size(subset_size::SubsetSizeEC, rd::Pr_RR)
    res = subset_size(rd)
    assert_nonempty_nonneg_finite_val(res - 1, "subset_size - 1")
    @argcheck(res <= size(rd.X, 2),
              "subset_size must not be greater than the number of assets")
    return res
end
"""
    get_window_size(window_size, rd, args...)

Get the actual rolling window size for multiple-randomised cross-validation.

Resolves the window size from `nothing` (no windowing), an integer (direct count), a float (fraction of observations), or a callable.

# Arguments

  - `window_size`: Window size specification (`nothing`, integer, float, or callable).
  - `rd`: Returns result or prior.
  - `args...`: Additional arguments.

# Returns

  - Integer window size or `nothing`.

# Related

  - [`MultipleRandomised`](@ref)
  - [`get_subset_size`](@ref)
"""
function get_window_size(::Nothing, args...)
    return nothing
end
function get_window_size(window_size::Integer, rd::Pr_RR, args...)
    @argcheck(window_size <= size(rd.X, 1),
              "window_size must not be greater than the number of observations")
    return window_size
end
function get_window_size(window_size::AbstractFloat, rd::Pr_RR, args...)
    window_size = max(round(Int, window_size * size(rd.X, 1)), 2)
    @argcheck(window_size <= size(rd.X, 1),
              "window_size must not be greater than the number of observations")
    return window_size
end
function get_window_size(window_size::WindowSizeEC, rd::Pr_RR)
    res = window_size(rd)
    assert_nonempty_nonneg_finite_val(res - 2, "window_size - 2")
    @argcheck(res <= size(rd.X, 1),
              "window_size must not be greater than the number of observations")
    return res
end
"""
    get_n_subsets(n_subsets, args...)

Get the number of asset subsets for multiple-randomised cross-validation.

Resolves the number of subsets from either an integer (direct count) or a callable that computes it from the returns data.

# Arguments

  - `n_subsets`: Integer or callable number-of-subsets specification.
  - `args...`: Additional arguments (returns result or prior).

# Returns

  - Integer number of subsets.

# Related

  - [`MultipleRandomised`](@ref)
  - [`sample_unique_assets`](@ref)
"""
function get_n_subsets(n_subsets::Integer, args...)
    return n_subsets
end
function get_n_subsets(n_subsets::NumberSubsetsEC, rd::Pr_RR)
    res = n_subsets(rd)
    assert_nonempty_nonneg_finite_val(res - 2, "n_subsets - 2")
    return res
end
"""
    Base.split(mrcv::MultipleRandomised, rd::ReturnsResult) -> MultipleRandomisedResult

Split the returns data `rd` by drawing multiple random asset subsets and applying the
internal walk-forward estimator to each subset. Each combination of a random asset subset
and a set of walk-forward folds forms one path.

# Arguments

  - `mrcv::MultipleRandomised`: Multiple randomised cross-validation estimator.
  - `rd::ReturnsResult`: Returns data to split.

# Returns

  - `MultipleRandomisedResult`: Result containing training, test, and asset indices for
    every fold across all random paths, together with a path identifier for each fold.

# Related

  - [`MultipleRandomised`](@ref)
  - [`MultipleRandomisedResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(mrcv::MultipleRandomised, rd::ReturnsResult)
    T, N = size(rd.X)
    (; cv, subset_size, n_subsets, max_comb, window_size, rng, seed) = mrcv
    subset_size = get_subset_size(subset_size, rd)
    n_subsets = get_n_subsets(n_subsets, rd)
    window_size = get_window_size(window_size, rd)
    n_comb = binomial(N, subset_size)
    @argcheck(n_subsets <= n_comb,
              "n_subsets = $n_subsets must not be greater than `binomial(assets, subset_size) = n_comb => binomial($N, $subset_size) = $n_comb`.")
    asset_idx = sample_unique_assets(N, subset_size, n_subsets; max_comb = max_comb,
                                     rng = rng, seed = seed)
    path_ids = Vector{typeof(n_subsets)}(undef, 0)
    train_indices = Vector{UnitRange{typeof(T)}}(undef, 0)
    test_indices = Vector{UnitRange{typeof(T)}}(undef, 0)
    asset_indices = Vector{Vector{typeof(T)}}(undef, 0)
    for i in 1:n_subsets
        if isnothing(window_size)
            start_obs = 1
            rdi = rd
        else
            start_obs = rand(rng, 1:(T - window_size))
            idx = start_obs:(start_obs + window_size)
            rdi = returns_result_view(rd, idx, :)
        end
        start_obs -= 1
        (; train_idx, test_idx) = try
            split(cv, rdi)
        catch err
            if isa(err, IsEmptyError)
                throw(IsEmptyError(err.msg *
                                   "\nthe local rd does not contain enough observations for the window size:\n$rdi"))
            else
                rethrow(err)
            end
        end
        num_splits = length(train_idx)
        append!(path_ids, fill(i, num_splits))
        append!(train_indices, [t .+ start_obs for t in train_idx])
        append!(test_indices, [t .+ start_obs for t in test_idx])
        append!(asset_indices, Iterators.repeated(view(asset_idx, :, i), num_splits))
    end
    return MultipleRandomisedResult(; train_idx = train_indices, test_idx = test_indices,
                                    asset_idx = asset_indices, path_ids = path_ids)
end
"""
    path_fit_and_predict(opt, rd, train_idx, test_idx, cols; ex, id)

Fit and predict along a sequence of (train, test, asset) triples, respecting sequential constraints.

Handles sequential and parallel execution. If the optimiser requires previous weights or is time-dependent, runs sequentially and passes weights/state between periods. Otherwise, runs in parallel using the provided executor.

# Arguments

  - `opt::NonFiniteAllocationOptimisationEstimator`: Portfolio optimisation estimator.
  - `rd::ReturnsResult`: Full returns data.
  - `train_idx`: Sequence of training index vectors.
  - `test_idx`: Sequence of test index vectors.
  - `cols`: Sequence of asset column indices for each fold.
  - `ex::FLoops.Transducers.Executor`: Executor for parallel processing.
  - `id`: Optional path identifier.

# Returns

  - [`MultiPeriodPredictionResult`](@ref) with predictions sorted by test index.

# Related

  - [`fit_and_predict`](@ref)
  - [`needs_previous_weights`](@ref)
  - [`is_time_dependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
"""
function path_fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator,
                              rd::ReturnsResult, train_idx, test_idx, cols;
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                              id = nothing)
    predictions = Vector{PredictionResult}(undef, length(train_idx))
    prev_w_flag = needs_previous_weights(opt)
    time_dep_flag = is_time_dependent(opt)
    if prev_w_flag || time_dep_flag
        @info("Running walk forward sequentially because the optimiser must either use the previous optimisation's weights (needs_previous_weights(opt) == $prev_w_flag), and/or is time dependent (is_time_dependent(opt) == $time_dep_flag). This is because somewhere within the optimisation estimator is contained at least one of the following:\n\t- Turnover and/or TurnoverEstimator,\n\t- WeightsTracking,\n\t- TurnoverRiskMeasure,\n\t- custom constraints which use asset weights,\n\t- custom objective penalties which use asset weights.-\n\t Or there is a time dependent constraint or objective penalty.\nTo enable parallel processing please either mark the weights as fixed or remove the offending component(s).")
        for (i, (train, test, col)) in enumerate(zip(train_idx, test_idx, cols))
            rdi = returns_result_view(rd, col)
            opti = opt_view(opt, col, rdi.X)
            if i > 1
                if prev_w_flag
                    opti = factory(opti, predictions[i - 1].res.w)
                end
                if time_dep_flag
                    opti = update_time_dependent_estimator(opti, i, rdi, train_idx,
                                                           test_idx)
                end
            end
            predictions[i] = fit_and_predict(opti, rdi; train_idx = train, test_idx = test)
        end
    else
        FLoops.@floop ex for (i, (train, test, col)) in
                             enumerate(zip(train_idx, test_idx, cols))
            predictions[i] = fit_and_predict(opt, rd; train_idx = train, test_idx = test,
                                             cols = col)
        end
    end
    return MultiPeriodPredictionResult(; pred = sort_predictions!(test_idx, predictions),
                                       id = id)
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         cv::MRCVR; ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         kwargs...)
    cv_res = split(cv, rd)
    (; train_idx, test_idx, asset_idx, path_ids) = cv_res
    unique_ids = unique(path_ids)
    dict = [Vector{Tuple{eltype(train_idx), eltype(test_idx), eltype(asset_idx)}}(undef, 0)
            for _ in unique_ids]
    for (train, test, asset, path_id) in zip(train_idx, test_idx, asset_idx, path_ids)
        push!(dict[path_id], (train, test, asset))
    end
    predictions = Vector{MultiPeriodPredictionResult}(undef, length(unique_ids))
    FLoops.@floop ex for (i, vals) in enumerate(dict)
        train = map(x -> x[1], vals)
        test = map(x -> x[2], vals)
        asset = map(x -> x[3], vals)
        predictions[i] = path_fit_and_predict(opt, rd, train, test, asset; ex = ex, id = i)
    end
    return PopulationPredictionResult(; pred = predictions)
end

export MultipleRandomised, MultipleRandomisedResult
