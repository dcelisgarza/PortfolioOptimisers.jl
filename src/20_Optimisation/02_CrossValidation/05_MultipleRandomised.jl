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

$(DocStringExtensions.FIELDS)

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

`cv` is positional; remaining arguments are keyword and correspond to the struct's fields.

## Validation

  - If `subset_size` is an `Integer`: `subset_size >= 1`.
  - If `subset_size` is a float: `0 < subset_size < 1`.
  - If `n_subsets` is an `Integer`: `n_subsets >= 2`.
  - `max_comb > 0` and finite.
  - If `window_size` is an `Integer`: `window_size >= 2`.
  - If `window_size` is a float: `0 < window_size < 1`.

# Related

  - [`MultipleRandomisedResult`](@ref)
  - [`WalkForwardEstimator`](@ref)
  - [`IndexWalkForward`](@ref)
  - [`DateWalkForward`](@ref)
"""
@concrete struct MultipleRandomised <: NonOptimisationSequentialCrossValidationEstimator
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:subset_size])
    """
    subset_size
    """
    $(field_dict[:n_subsets])
    """
    n_subsets
    """
    $(field_dict[:max_comb])
    """
    max_comb
    """
    $(field_dict[:window_size])
    """
    window_size
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    function MultipleRandomised(cv::WalkForwardEstimator, subset_size::SubsetSizeE,
                                n_subsets::NumberSubsetsE, max_comb::Integer,
                                window_size::Option{<:WindowSizeE}, rng::Random.AbstractRNG,
                                seed::Option{<:Integer})
        if isa(subset_size, Integer)
            assert_nonempty_nonneg_finite_val(subset_size - 1, "subset_size - 1")
        elseif isa(subset_size, AbstractFloat)
            assert_unit_interval(subset_size, :subset_size)
        end
        if isa(n_subsets, Integer)
            assert_nonempty_nonneg_finite_val(n_subsets - 2, "n_subsets - 2")
        end
        assert_nonempty_gt0_finite_val(max_comb, :max_comb)
        if isa(window_size, Integer)
            assert_nonempty_nonneg_finite_val(window_size - 2, "window_size - 2")
        elseif isa(window_size, AbstractFloat)
            assert_unit_interval(window_size, :window_size)
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

$(DocStringExtensions.FIELDS)

# Constructors

    MultipleRandomisedResult(;
        train_idx::VecVecInt,
        test_idx::VecVecInt,
        asset_idx::VecVecInt,
        path_ids::VecInt
    ) -> MultipleRandomisedResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(train_idx)`.
  - `!isempty(test_idx)`.
  - `!isempty(asset_idx)`.
  - `!isempty(path_ids)`.
  - `length(train_idx) == length(test_idx) == length(asset_idx) == length(path_ids)`.

# Related

  - [`MultipleRandomised`](@ref)
"""
@concrete struct MultipleRandomisedResult <: NonOptimisationSequentialCrossValidationResult
    """
    $(field_dict[:train_idx])
    """
    train_idx
    """
    $(field_dict[:test_idx])
    """
    test_idx
    """
    $(field_dict[:asset_idx])
    """
    asset_idx
    """
    $(field_dict[:path_ids])
    """
    path_ids
    function MultipleRandomisedResult(train_idx::VecVecInt, test_idx::VecVecInt,
                                      asset_idx::VecVecInt, path_ids::VecInt)
        @argcheck(!isempty(train_idx), IsEmptyError("train_idx cannot be empty"))
        @argcheck(!isempty(test_idx), IsEmptyError("test_idx cannot be empty"))
        @argcheck(!isempty(asset_idx), IsEmptyError("asset_idx cannot be empty"))
        @argcheck(!isempty(path_ids), IsEmptyError("path_ids cannot be empty"))
        @argcheck(length(train_idx) ==
                  length(test_idx) ==
                  length(asset_idx) ==
                  length(path_ids),
                  DimensionMismatch("train_idx ($(length(train_idx))), test_idx ($(length(test_idx))), asset_idx ($(length(asset_idx))), and path_ids ($(length(path_ids))) must all match"))
        return new{typeof(train_idx), typeof(test_idx), typeof(asset_idx),
                   typeof(path_ids)}(train_idx, test_idx, asset_idx, path_ids)
    end
end
function MultipleRandomisedResult(; train_idx::VecVecInt, test_idx::VecVecInt,
                                  asset_idx::VecVecInt,
                                  path_ids::VecInt)::MultipleRandomisedResult
    return MultipleRandomisedResult(train_idx, test_idx, asset_idx, path_ids)
end
function n_splits(mre::MultipleRandomised, rd::Prices_RR)
    if !isnothing(mre.window_size) && isa(mre.cv, DateWalkForward)
        throw(ArgumentError("when using a `DateWalkForward` with `window_size`, the number of splits cannot be determined before calling [`split`](@ref)."))
    end
    if !isnothing(mre.window_size)
        rd = port_opt_view(rd, 1:(mre.window_size), :)
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
    combination_by_index(idx::Integer, N::Integer, k::Integer) -> VecInt

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
    @argcheck(0 < idx <= n_comb, DomainError(idx, "idx must be in (0, $n_comb]"))
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
    @argcheck(k <= N, DomainError("k ($k) must be less than or equal to N ($N)"))
    assert_nonempty_finite_val(n_subsets, :n_subsets)
    n_comb = binomial(N, k)
    @argcheck(n_subsets <= n_comb,
              "n_subsets = $n_subsets must not be greater than `binomial(assets, subset_size) = n_comb => binomial($N, $k) = $n_comb`.")
    rng = resolve_rng(rng, seed)
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
function get_subset_size(subset_size::Integer, rd::Union{<:Pr_RR, <:AbstractPricesResult},
                         args...)
    @argcheck(subset_size <= size(rd.X, 2),
              "subset_size must not be greater than the number of assets")
    return subset_size
end
function get_subset_size(subset_size::AbstractFloat,
                         rd::Union{<:Pr_RR, <:AbstractPricesResult}, args...)
    subset_size = max(round(Int, subset_size * size(rd.X, 2)), 1)
    return subset_size
end
function get_subset_size(subset_size::SubsetSizeEC,
                         rd::Union{<:Pr_RR, <:AbstractPricesResult})
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
function get_window_size(window_size::Integer, rd::Union{<:Pr_RR, <:AbstractPricesResult},
                         args...)
    @argcheck(window_size <= size(rd.X, 1),
              "window_size must not be greater than the number of observations")
    return window_size
end
function get_window_size(window_size::AbstractFloat,
                         rd::Union{<:Pr_RR, <:AbstractPricesResult}, args...)
    window_size = max(round(Int, window_size * size(rd.X, 1)), 2)
    @argcheck(window_size <= size(rd.X, 1),
              "window_size must not be greater than the number of observations")
    return window_size
end
function get_window_size(window_size::WindowSizeEC,
                         rd::Union{<:Pr_RR, <:AbstractPricesResult})
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

This is the single point at which the (possibly [`TimeDependent`](@ref) or callable) subset count becomes a concrete integer, so it is where the [`RESOURCE_LIMITS`](@ref) `max_n_subsets` ceiling is enforced — a constructor check could not see the value a schedule or callable ultimately produces. Every subset runs a full inner optimisation, so an absurd count is a compute-exhaustion sink rather than a merely large allocation.

# Arguments

  - `n_subsets`: Integer or callable number-of-subsets specification.
  - `args...`: Additional arguments (returns result or prior).

# Returns

  - Integer number of subsets.

# Related

  - [`MultipleRandomised`](@ref)
  - [`sample_unique_assets`](@ref)
  - [`assert_resource_cap`](@ref)
"""
function get_n_subsets(n_subsets::Integer, args...)
    assert_resource_cap(n_subsets, RESOURCE_LIMITS[].max_n_subsets, :n_subsets,
                        :max_n_subsets)
    return n_subsets
end
function get_n_subsets(n_subsets::NumberSubsetsEC,
                       rd::Union{<:Pr_RR, <:AbstractPricesResult})
    res = n_subsets(rd)
    assert_nonempty_nonneg_finite_val(res - 2, "n_subsets - 2")
    assert_resource_cap(res, RESOURCE_LIMITS[].max_n_subsets, :n_subsets, :max_n_subsets)
    return res
end
"""
    Base.split(mrcv::MultipleRandomised, rd::Prices_RR) -> MultipleRandomisedResult

Split the price- or returns-level data `rd` by drawing multiple random asset subsets and
applying the internal walk-forward estimator to each subset. Each combination of a random
asset subset and a set of walk-forward folds forms one path.

Unlike combinatorial cross-validation, multiple-randomised resampling draws over **assets**
(columns) while every observation window comes from an inner walk-forward, so the rows of
each fold stay contiguous. That is why it is admissible at the price level for a
price-starting pipeline — the rolling-window rule that blocks combinatorial does not apply.

# Arguments

  - `mrcv::MultipleRandomised`: Multiple randomised cross-validation estimator.
  - `rd::Prices_RR`: Price- or returns-level data to split.

# Returns

  - `MultipleRandomisedResult`: Result containing training, test, and asset indices for
    every fold across all random paths, together with a path identifier for each fold.

# Related

  - [`MultipleRandomised`](@ref)
  - [`MultipleRandomisedResult`](@ref)
  - [`n_splits`](@ref)
"""
function Base.split(mrcv::MultipleRandomised, rd::Prices_RR)
    T, N = size(rd.X)
    (; cv, subset_size, n_subsets, max_comb, window_size, rng, seed) = mrcv
    subset_size = get_subset_size(subset_size, rd)
    n_subsets = get_n_subsets(n_subsets, rd)
    window_size = get_window_size(window_size, rd)
    n_comb = binomial(N, subset_size)
    @argcheck(n_subsets <= n_comb,
              "n_subsets = $n_subsets must not be greater than `binomial(assets, subset_size) = n_comb => binomial($N, $subset_size) = $n_comb`.")
    # Resolve once: this single stream must serve BOTH the asset-subset sampling below and the
    # per-path window offsets (`rand(rng, ...)` in the loop), so `seed` governs the whole split.
    # `sample_unique_assets` is therefore handed the already-resolved rng and no seed.
    rng = resolve_rng(rng, seed)
    asset_idx = sample_unique_assets(N, subset_size, n_subsets; max_comb = max_comb,
                                     rng = rng)
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
            rdi = port_opt_view(rd, idx, :)
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
  - `rd::ReturnsResult`: FullMoment returns data.
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
function path_fit_and_predict(opt::OptE_TD, rd::ReturnsResult, train_idx, test_idx, cols;
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                              id = nothing)
    n = length(train_idx)
    td_flag = is_time_dependent(opt)
    if td_flag
        assert_time_dependent_fold_count(opt, n)
    end
    # `i` is the fold's position in the path's split enumeration — no ordering is imposed
    # on time-dependent entries (predictions are sorted for reporting only, after the
    # loop); the user keys entries off ctx.train_idx[ctx.i] / ctx.test_idx[ctx.i].
    predictions = run_folds(opt, n, ex) do i, prev
        col = cols[i]
        rdi = port_opt_view(rd, col)
        opti = port_opt_view(opt, col, rdi.X)
        # Resolve time-dependent constraints first so freshly swapped-in per-fold
        # constraints also receive the previous weights from the factory pass below.
        if td_flag
            ctx = TimeDependentContext(; i = i, n = n, rd = rdi, train_idx = train_idx,
                                       test_idx = test_idx,
                                       w_prev = isnothing(prev) ? nothing : prev.res.w,
                                       path_id = id)
            opti = update_time_dependent_estimator(opti, ctx)
        end
        if !isnothing(prev) && needs_previous_weights(opt)
            opti = factory(opti, prev.res.w)
        end
        return fit_and_predict(opti, rdi; train_idx = train_idx[i], test_idx = test_idx[i])
    end
    return MultiPeriodPredictionResult(; pred = sort_predictions!(test_idx, predictions),
                                       id = id)
end
function fit_and_predict(opt::OptE_TD, rd::ReturnsResult, cv::MRCVR;
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(), kwargs...)
    cv_res = split(cv, rd)
    (; train_idx, test_idx, asset_idx, path_ids) = cv_res
    unique_ids = unique(path_ids)
    dict = [Vector{Tuple{eltype(train_idx), eltype(test_idx), eltype(asset_idx)}}(undef, 0)
            for _ in unique_ids]
    for (train, test, asset, path_id) in zip(train_idx, test_idx, asset_idx, path_ids)
        push!(dict[path_id], (train, test, asset))
    end
    predictions = parallel_folds(length(unique_ids), ex; ElT = MultiPeriodPredictionResult
                                 ) do i
        vals = dict[i]
        train = map(x -> x[1], vals)
        test = map(x -> x[2], vals)
        asset = map(x -> x[3], vals)
        return path_fit_and_predict(opt, rd, train, test, asset; ex = ex, id = i)
    end
    return PopulationPredictionResult(; pred = predictions)
end

export MultipleRandomised, MultipleRandomisedResult
