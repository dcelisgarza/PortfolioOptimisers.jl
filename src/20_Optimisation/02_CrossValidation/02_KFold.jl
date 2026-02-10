struct KFold{T1, T2, T3} <: NonSequentialCrossValidationEstimator
    n::T1
    purged_size::T2
    embargo_size::T3
    function KFold(n::Integer, purged_size::Integer, embargo_size::Integer)
        assert_nonempty_gt0_finite_val(n, :n)
        assert_nonempty_finite_val(purged_size, :purged_size)
        assert_nonempty_finite_val(embargo_size, :embargo_size)
        return new{typeof(n), typeof(purged_size), typeof(embargo_size)}(n, purged_size,
                                                                         embargo_size)
    end
end
function KFold(; n::Integer = 5, purged_size::Integer = 0, embargo_size::Integer = 0)
    return KFold(n, purged_size, embargo_size)
end
function Base.split(kf::KFold, rd::ReturnsResult)
    T = size(rd.X, 1)
    (; n, purged_size, embargo_size) = kf
    idx = 1:T
    min_fold_size = div(T, n)
    @argcheck(purged_size + embargo_size < min_fold_size)
    fold_sizes = fill(min_fold_size, n)
    fold_sizes[1:(mod(T, n))] .+= one(eltype(fold_sizes))
    test_indices = Vector{typeof(idx)}(undef, 0)
    current = one(eltype(fold_sizes))
    for fold_size in fold_sizes
        start, stop = current, current + fold_size
        push!(test_indices, idx[start:(stop - 1)])
        current = stop
    end
    train_indices = Vector{Vector{eltype(T)}}(undef, 0)
    idx = 1:n
    for i in idx
        tidx = setdiff(idx, i)
        tmp_test_idx = Vector{typeof(idx)}(undef, 0)
        for j in tidx
            if j == i - 1
                push!(tmp_test_idx, test_indices[j][1:(end - purged_size)])
            elseif j == i + 1
                push!(tmp_test_idx, test_indices[j][(1 + purged_size + embargo_size):end])
            else
                push!(tmp_test_idx, test_indices[j])
            end
        end
        push!(train_indices, vcat(tmp_test_idx...))
    end
    return train_indices, test_indices
end
function n_splits(kf::KFold, rd::ReturnsResult)
    return min(kf.n, size(rd.X, 1))
end

export KFold
