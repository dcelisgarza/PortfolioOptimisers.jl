abstract type CokurtosisEstimator end
function cokurtosis end
function _cokurosis(X, mp)
    T, N = size(X)
    o = transpose(range(; start = one(eltype(X)), stop = one(eltype(X)), length = N))
    z = kron(o, X) .* kron(X, o)
    ckurt = transpose(z) * z / T
    mtx_process!(mp, ckurt, X)
    return ckurt
end

export cokurtosis
