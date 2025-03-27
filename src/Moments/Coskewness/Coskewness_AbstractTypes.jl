abstract type CoskewnessEstimator end
function coskewness end
function __coskewness(cskew, X, mp)
    N = size(cskew, 1)
    V = zeros(eltype(cskew), N, N)
    for i ∈ 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew, :, j:k)
        mtx_process!(mp, coskew_jk, X)
        vals, vecs = eigen(coskew_jk)
        vals .= clamp.(real.(vals), typemin(eltype(cskew)), 0) +
                clamp.(imag.(vals), typemin(eltype(cskew)), 0)im
        V .-= real(vecs * Diagonal(vals) * transpose(vecs))
    end
    return V
end
function _coskewness(y, X, mp)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)),
                        length = size(X, 2)))
    z = kron(o, y) .* kron(y, o)
    cskew = transpose(X) * z / size(X, 1)
    V = __coskewness(cskew, X, mp)
    return cskew, V
end

export coskewness
