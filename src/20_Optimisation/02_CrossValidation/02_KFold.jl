struct KFold{T1} <: NonSequentialCrossValidationEstimator
    n::T1
    function KFold(n::Integer)
        assert_nonempty_gt0_finite_val(n, :n)
        return new{typeof(n)}(n)
    end
end
function KFold(; n::Integer = 5)
    return KFold(n)
end
function Base.split(kf::KFold, rd::ReturnsResult)
    T = size(rd.X, 1)
    idx = 1:T
    n = kf.n
    fold_sizes = fill(div(T, n), n)
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
        push!(train_indices, vcat(test_indices[setdiff(idx, i)]...))
    end
    return train_indices, test_indices
end
function n_splits(kf::KFold, rd::ReturnsResult)
    return min(kf.n, size(rd.X, 1))
end

export KFold
