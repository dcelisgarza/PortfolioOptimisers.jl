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
        return new{typeof(n_folds), typeof(n_test_folds), typeof(purged_size),
                   typeof(embargo_size)}(n_folds, n_test_folds, purged_size, embargo_size)
    end
end
function CombinatorialCrossValidation(; n_folds::Integer = 10, n_test_folds::Integer = 8,
                                      purged_size::Integer = 0, embargo_size::Integer = 0)
    return CombinatorialCrossValidation(n_folds, n_test_folds, purged_size, embargo_size)
end
function Base.split(ccv::CombinatorialCrossValidation, rd::ReturnsResult)
    #=
    T = size(rd.X, 1)
    (; n_folds, n_test_folds, purged_size, embargo_size) = ccv
    idx = 1:T
    min_fold_size = div(T, n_folds)
    @argcheck(purged_size + embargo_size < min_fold_size)
    fold_sizes = fill(min_fold_size, n_folds)
    fold_sizes[1:(mod(T, n_folds))] .+= one(eltype(fold_sizes))
    fold_indices = Vector{typeof(idx)}(undef, 0)
    current = one(eltype(fold_sizes))
    for fold_size in fold_sizes
        start, stop = current, current + fold_size
        push!(fold_indices, idx[start:(stop - 1)])
        current = stop
    end
    test_indices = Vector{typeof(idx)}(undef, 0)
    for test_fold in combinations(1:n_folds, n_test_folds)
        push!(test_indices, vcat(fold_indices[test_fold]...))
    end
    train_indices = Vector{Vector{eltype(T)}}(undef, 0)
    for test_fold in combinations(1:n_folds, n_test_folds)
        tmp_test_idx = Vector{typeof(idx)}(undef, 0)
        for j in test_fold
            if j == minimum(test_fold) - 1
                push!(tmp_test_idx, fold_indices[j][1:(end - purged_size)])
            elseif j == maximum(test_fold) + 1
                push!(tmp_test_idx,
                      fold_indices[j][(1 + purged_size + embargo_size):end])
            else
                push!(tmp_test_idx, fold_indices[j])
            end
        end
        push!(train_indices,
              setdiff(idx, vcat(tmp_test_idx..., fold_indices[test_fold]...)))
    end
    return train_indices, test_indices
    =#
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
    type = promote_type(typeof(num_splits), typeof(n_folds))
    folds_train_test = zeros(type, n_folds, num_splits)
    test_set_idx = test_set_index(ccv)
    for (i, idx) in enumerate(test_set_idx)
        folds_train_test[idx, i] .= one(type)
    end
    return folds_train_test
end

export CombinatorialCrossValidation, n_test_paths, average_train_size, test_set_index,
       binary_train_test_sets
