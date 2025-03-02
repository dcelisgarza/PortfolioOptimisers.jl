abstract type CoskewnessEstimator end
function coskewness end
function _coskewness(y, X, mp)
    T, N = size(y)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)), length = N))
    z = kron(o, y) .* kron(y, o)
    cskew = transpose(X) * z / T
    V = zeros(eltype(y), N, N)
    for i ∈ 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew, :, j:k)
        mtx_process!(mp, coskew_jk, X)
        vals, vecs = eigen(coskew_jk)
        vals = clamp.(real.(vals), -Inf, 0) .+ clamp.(imag.(vals), -Inf, 0)im
        V .-= real(vecs * Diagonal(vals) * transpose(vecs))
    end
    return cskew, V
end

export coskewness
