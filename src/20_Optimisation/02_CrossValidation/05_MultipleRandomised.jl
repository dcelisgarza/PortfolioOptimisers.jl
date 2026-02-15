struct MultipleRandomised{T1, T2, T3, T4, T5, T6, T7} <: SequentialCrossValidationEstimator
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
        train_idx, test_idx = split(cv, rdi)
        num_splits = length(train_idx)
        append!(path_ids, fill(i, num_splits))
        append!(train_indices, [t .+ start_obs for t in train_idx])
        append!(test_indices, [t .+ start_obs for t in test_idx])
        append!(asset_indices, Iterators.repeated(view(asset_idx, :, i), num_splits))
    end

    return train_indices, test_indices, asset_indices, path_ids
end

export MultipleRandomised
