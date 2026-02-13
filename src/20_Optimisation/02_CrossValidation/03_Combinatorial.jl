struct CombinatorialCrossValidation{T1, T2, T3, T4} <: CrossValidationEstimator
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
function n_splits(ccv::CombinatorialCrossValidation)
    return binomial(ccv.n_folds, ccv.n_test_folds)
end
function n_test_paths(ccv::CombinatorialCrossValidation)
    return div(n_splits(ccv) * ccv.n_test_folds, ccv.n_folds)
end
function average_train_size(ccv::CombinatorialCrossValidation, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n_folds, n_test_folds) = ccv
    return T / n_folds * (n_folds - n_test_folds)
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
    ids = zeros(Int, num_splits, ccv.n_test_folds)
    for i in axes(ids, 1)
        inds = findall(x -> x == i, rcp)
        for j in axes(ids, 2)
            ids[i, end - j + 1] = inds[j][2]
        end
    end
    return ids
end
function Base.split(ccv::CombinatorialCrossValidation, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n_folds, n_test_folds, purged_size, embargo_size) = ccv
    min_fold_size = div(T, n_folds)
    pes = purged_size + embargo_size
    @argcheck(pes < min_fold_size)
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
    for i in 1:pes
        j = map(x -> min(T, x + i), after_idx_1)
        for (j, k) in zip(j, after_idx_2)
            train_test_idx[j, k] = -one(num_splits)
        end
    end
    fold_index = Dict(i => findall(fold_idx_num .== i) for i in 1:n_folds)
    #! allocate train and test induces
    for i in 1:num_splits
        train_idx = findall(x -> x == 0, view(train_test_idx, :, i))
        test_idx_list = [fold_index[j[1]] for j in findall(x -> x == i, rcp)]
        return train_idx, test_idx_list
    end
    return fold_index
end

export CombinatorialCrossValidation, n_test_paths, average_train_size, test_set_index,
       binary_train_test_sets, recombined_paths, get_path_ids
