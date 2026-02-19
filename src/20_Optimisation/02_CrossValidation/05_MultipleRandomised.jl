struct MultipleRandomised{T1, T2, T3, T4, T5, T6, T7} <:
       NonOptimisationSequentialCrossValidationEstimator
    cv::T1
    subset_size::T2
    n_subsets::T3
    max_comb::T4
    window_size::T5
    rng::T6
    seed::T7
    function MultipleRandomised(cv::WalkForwardEstimator, subset_size::Integer,
                                n_subsets::Integer, max_comb::Integer,
                                window_size::Option{<:Integer}, rng::Random.AbstractRNG,
                                seed::Option{<:Integer})
        assert_nonempty_nonneg_finite_val(subset_size - 1, "subset_size - 1")
        assert_nonempty_nonneg_finite_val(n_subsets - 2, "n_subsets - 2")
        assert_nonempty_gt0_finite_val(max_comb, :max_comb)
        if !isnothing(window_size)
            assert_nonempty_nonneg_finite_val(window_size - 2, "window_size - 2")
        end
        return new{typeof(cv), typeof(subset_size), typeof(n_subsets), typeof(max_comb),
                   typeof(window_size), typeof(rng), typeof(seed)}(cv, subset_size,
                                                                   n_subsets, max_comb,
                                                                   window_size, rng, seed)
    end
end
function MultipleRandomised(cv::WalkForwardEstimator; subset_size::Integer = 1,
                            n_subsets::Integer = 2, max_comb::Integer = 1_000_000_000,
                            window_size::Option{<:Integer} = nothing,
                            rng::Random.AbstractRNG = Random.default_rng(),
                            seed::Option{<:Integer} = nothing)
    return MultipleRandomised(cv, subset_size, n_subsets, max_comb, window_size, rng, seed)
end
struct MultipleRandomisedResult{T1, T2, T3, T4} <:
       NonOptimisationSequentialCrossValidationResult
    train_idx::T1
    test_idx::T2
    asset_idx::T3
    path_ids::T4
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
function Base.split(mrcv::MultipleRandomised, rd::ReturnsResult)
    T, N = size(rd.X)
    (; cv, subset_size, n_subsets, max_comb, window_size, rng, seed) = mrcv
    @argcheck(subset_size <= N, "subset_size must not be greater than the number of assets")
    if !isnothing(window_size)
        @argcheck(window_size <= T,
                  "window_size must not be greater than the number of observations")
    end
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
        (; train_idx, test_idx) = split(cv, rdi)
        num_splits = length(train_idx)
        append!(path_ids, fill(i, num_splits))
        append!(train_indices, [t .+ start_obs for t in train_idx])
        append!(test_indices, [t .+ start_obs for t in test_idx])
        append!(asset_indices, Iterators.repeated(view(asset_idx, :, i), num_splits))
    end

    return MultipleRandomisedResult(; train_idx = train_indices, test_idx = test_indices,
                                    asset_idx = asset_indices, path_ids = path_ids)
end
function path_fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator,
                              rd::ReturnsResult, train_idx, test_idx, cols;
                              ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    predictions = Vector{PredictionResult}(undef, length(train_idx))
    if needs_previous_weights(opt)
        @info("Running walk forward sequentially because the optimiser must use the previous optimisation's weights. This is because somewhere within the optimisation estimator is contained at least one of the following:\n\t- Turnover and/or TurnoverEstimator,\n\t- WeightsTracking,\n\t- TurnoverRiskMeasure,\n\t- custom constraints which use asset weights,\n\t- custom objective penalties which use asset weights.\nTo enable parallel processing please either mark the weights as fixed or remove the component(s) which use(s) them.")
        for (i, (train, test, col)) in enumerate(zip(train_idx, test_idx, cols))
            rdi = returns_result_view(rd, col)
            opti = opt_view(opt, col, rdi.X)
            if i > 1
                opti = factory(opti, predictions[i - 1].res.w)
            end
            predictions[i] = fit_and_predict(opti, rdi; train_idx = train, test_idx = test)
        end
    else
        let opt = opt
            FLoops.@floop ex for (i, (train, test, col)) in
                                 enumerate(zip(train_idx, test_idx, cols))
                predictions[i] = fit_and_predict(opt, rd; train_idx = train,
                                                 test_idx = test, cols = col)
            end
        end
    end
    return MultiPeriodPredictionResult(; pred = sort_predictions!(test_idx, predictions))
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         cv::MultipleRandomisedResult;
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(), kwargs...)
    (; train_idx, test_idx, asset_idx, path_ids) = cv
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
        predictions[i] = path_fit_and_predict(opt, rd, train, test, asset)
    end
    return PopulationPredictionResult(; pred = predictions)
end

export MultipleRandomised, MultipleRandomisedResult
