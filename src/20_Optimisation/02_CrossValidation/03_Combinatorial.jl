struct CombinatorialCrossValidation{T1, T2, T3, T4} <: NonSequentialCrossValidationEstimator
    n_folds::T1
    n_test_folds::T2
    purged_size::T3
    embargo_size::T4
    function CombinatorialCrossValidation(n_folds::Integer, n_test_folds::Integer,
                                          purged_size::Integer, embargo_size::Integer)
        assert_nonempty_gt0_finite_val(n_folds, :n_folds)
        assert_nonempty_gt0_finite_val(n_test_folds, :n_test_folds)
        assert_nonempty_finite_val(purged_size, :purged_size)
        assert_nonempty_finite_val(embargo_size, :embargo_size)
        if binomial(n_folds, n_test_folds) > 100_000
            @warn("The number of splits for `n_folds = $n_folds` and `n_test_folds = $n_test_folds` is `$(binomial(n_folds, n_test_folds))`, which may be computationally expensive. The number of combinations should typically be between 10^1 to 10^4 for statistical power. Such a large number of combinations may lead to long computation times and memory issues. Consider reducing `n_folds` or shifting `n_test_folds` further away from being equal to `div(n_folds, 2) = $(div(n_folds, 2))`.")
        end
        return new{typeof(n_folds), typeof(n_test_folds), typeof(purged_size),
                   typeof(embargo_size)}(n_folds, n_test_folds, purged_size, embargo_size)
    end
end
function CombinatorialCrossValidation(; n_folds::Integer = 10, n_test_folds::Integer = 8,
                                      purged_size::Integer = 0, embargo_size::Integer = 0)
    return CombinatorialCrossValidation(n_folds, n_test_folds, purged_size, embargo_size)
end
struct CombinatorialCrossValidationResult{T1, T2, T3} <: NonSequentialCrossValidationResult
    train_idx::T1
    test_idx::T2
    path_ids::T3
    function CombinatorialCrossValidationResult(train_idx::VecVecInt,
                                                test_idx::VecVecVecInt,
                                                path_ids::AbstractMatrix{<:Integer})
        @argcheck(!isempty(train_idx))
        @argcheck(!isempty(test_idx))
        @argcheck(!isempty(path_ids))
        @argcheck(length(train_idx) == length(test_idx) == size(path_ids, 2))
        return new{typeof(train_idx), typeof(test_idx), typeof(path_ids)}(train_idx,
                                                                          test_idx,
                                                                          path_ids)
    end
end
function CombinatorialCrossValidationResult(; train_idx::VecVecInt, test_idx::VecVecVecInt,
                                            path_ids::AbstractMatrix{<:Integer})
    return CombinatorialCrossValidationResult(train_idx, test_idx, path_ids)
end
function n_splits(n_folds::Integer, n_test_folds::Integer)
    return binomial(n_folds, n_test_folds)
end
function n_splits(ccv::CombinatorialCrossValidation)
    return n_splits(ccv.n_folds, ccv.n_test_folds)
end
function n_test_paths(n_folds::Integer, n_test_folds::Integer)
    return div(n_splits(n_folds, n_test_folds) * n_test_folds, n_folds)
end
function n_test_paths(ccv::CombinatorialCrossValidation)
    return div(n_splits(ccv) * ccv.n_test_folds, ccv.n_folds)
end
function average_train_size(T::Integer, n_folds::Integer, n_test_folds::Integer)
    return T / n_folds * (n_folds - n_test_folds)
end
function average_train_size(ccv::CombinatorialCrossValidation, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n_folds, n_test_folds) = ccv
    return average_train_size(T, n_folds, n_test_folds)
end
function test_set_index(ccv::CombinatorialCrossValidation)
    return collect(Combinatorics.combinations(1:(ccv.n_folds), ccv.n_test_folds))
end
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
function recombined_paths(ccv::CombinatorialCrossValidation)
    bidx = binary_train_test_sets(ccv)
    out = zeros(Int, size(bidx, 1), n_test_paths(ccv))
    for i in axes(bidx, 1)
        tmp = view(bidx, i, :)
        out[i, :] .= findall(tmp)
    end
    return out
end
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
function Base.split(ccv::CombinatorialCrossValidation, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n_folds, n_test_folds, purged_size, embargo_size) = ccv
    min_fold_size = div(T, n_folds)
    @argcheck(purged_size + embargo_size < min_fold_size)
    fold_idx_num = div.(0:(T - 1), min_fold_size)
    fold_idx_num[fold_idx_num .== n_folds] .= n_folds - 1
    fold_idx_num .+= 1
    num_splits = n_splits(ccv)
    test_set_idx = test_set_index(ccv)
    rcp = recombined_paths(ccv)
    train_test_idx = zeros(typeof(T), T, num_splits)
    for i in 1:num_splits
        train_test_idx[vcat([findall(x -> x == j, fold_idx_num) for j in test_set_idx[i]]...), i] .= one(num_splits)
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
                           predictions::AbstractVector{<:AbstractVector{<:PredictionResult}})
    path_ids = res.path_ids
    sorted_preds = [PredictionResult[] for _ in 1:maximum(path_ids)]
    for (j, prediction) in enumerate(predictions)
        for (i, pred) in enumerate(prediction)
            push!(sorted_preds[path_ids[i, j]], pred)
        end
    end
    return sorted_preds
end
function fit_and_predict(opt::NonFiniteAllocationOptimisationEstimator, rd::ReturnsResult,
                         cv::CombinatorialCrossValidationResult; cols = :,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    (; train_idx, test_idx) = cv
    predictions = Vector{Vector{PredictionResult}}(undef, length(train_idx))
    FLoops.@floop ex for (i, (train, test)) in enumerate(zip(train_idx, test_idx))
        predictions[i] = fit_and_predict(opt, rd; train_idx = train, test_idx = test,
                                         cols = cols)
    end
    return sort_predictions!(cv, predictions)
end

export CombinatorialCrossValidation, CombinatorialCrossValidationResult,
       optimal_number_folds
