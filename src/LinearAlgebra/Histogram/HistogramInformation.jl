function calc_hist_data(xj::AbstractVector, xi::AbstractVector, bins::Integer)
    bp1 = bins + one(bins)
    xjl = minimum(xj) - eps(eltype(xj))
    xjh = maximum(xj) + eps(eltype(xj))
    xil = minimum(xi) - eps(eltype(xi))
    xih = maximum(xi) + eps(eltype(xi))
    hx = fit(Histogram, xj, range(xjl; stop = xjh, length = bp1)).weights
    hx /= sum(hx)
    hy = fit(Histogram, xi, range(xil; stop = xih, length = bp1)).weights
    hy /= sum(hy)
    ex = entropy(hx)
    ey = entropy(hy)
    hxy = fit(Histogram, (xj, xi),
              (range(xjl; stop = xjh, length = bp1), range(xil; stop = xih, length = bp1))).weights
    return ex, ey, hxy
end
function intrinsic_mutual_info(X::AbstractMatrix)
    p_i = vec(sum(X; dims = 2))
    p_j = vec(sum(X; dims = 1))
    if length(p_i) == 1 || length(p_j) == 1
        return zero(eltype(p_j))
    end
    mask = findall(.!iszero.(X))
    nz = vec(X[mask])
    nz_sum = sum(nz)
    log_nz = log.(nz)
    nz_nm = nz / nz_sum
    outer = p_i[getindex.(mask, 1)] .* p_j[getindex.(mask, 2)]
    log_outer = -log.(outer) .+ log(sum(p_i)) .+ log(sum(p_j))
    mi = (nz_nm .* (log_nz .- log(nz_sum)) .+ nz_nm .* log_outer)
    mi[abs.(mi) .< eps(eltype(mi))] .= zero(eltype(X))
    return sum(mi)
end
function variation_info(X::AbstractMatrix,
                        bins::Union{<:AbstractBins, <:Integer} = B_HacineGharbiRavier(),
                        normalise::Bool = true)
    T, N = size(X)
    var_mtx = Matrix{eltype(X)}(undef, N, N)
    bin_width_func = get_bin_width_func(bins)
    for j ∈ axes(X, 2)
        xj = view(X, :, j)
        for i ∈ 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)
            mut_ixy = intrinsic_mutual_info(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normalise
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
            end
            var_ixy = clamp(var_ixy, zero(eltype(X)), Inf)
            var_mtx[j, i] = var_mtx[i, j] = var_ixy
        end
    end
    return var_mtx
end
#=
function mutual_variation_info(X::AbstractMatrix,
                               bins::Union{<:AbstractBins, <:Integer} = Knuth(),
                               normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)
    var_mtx = Matrix{eltype(X)}(undef, N, N)

    bin_width_func = get_bin_width_func(bins)

    for j ∈ axes(X, 2)
        xj = view(X, :, j)
        for i ∈ 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)

            mut_ixy = intrinsic_mutual_info(hxy)
            var_ixy = ex + ey - 2 * mut_ixy
            if normalise
                vxy = ex + ey - mut_ixy
                var_ixy = var_ixy / vxy
                mut_ixy /= min(ex, ey)
            end

            # if abs(mut_ixy) < eps(typeof(mut_ixy)) || mut_ixy < zero(eltype(X))
            #     mut_ixy = zero(eltype(X))
            # end
            # if abs(var_ixy) < eps(typeof(var_ixy)) || var_ixy < zero(eltype(X))
            #     var_ixy = zero(eltype(X))
            # end

            mut_ixy = clamp(mut_ixy, zero(eltype(X)), Inf)
            var_ixy = clamp(var_ixy, zero(eltype(X)), Inf)

            mut_mtx[j, i] = mut_mtx[i, j] = mut_ixy
            var_mtx[j, i] = var_mtx[i, j] = var_ixy
        end
    end

    return mut_mtx, var_mtx
end
=#
function mutual_info(X::AbstractMatrix,
                     bins::Union{<:AbstractBins, <:Integer} = B_HacineGharbiRavier(),
                     normalise::Bool = true)
    T, N = size(X)
    mut_mtx = Matrix{eltype(X)}(undef, N, N)
    bin_width_func = get_bin_width_func(bins)
    for j ∈ axes(X, 2)
        xj = view(X, :, j)
        for i ∈ 1:j
            xi = view(X, :, i)
            nbins = calc_num_bins(bins, xj, xi, j, i, bin_width_func, T)
            ex, ey, hxy = calc_hist_data(xj, xi, nbins)
            mut_ixy = intrinsic_mutual_info(hxy)
            if normalise
                mut_ixy /= min(ex, ey)
            end
            mut_ixy = clamp(mut_ixy, zero(eltype(X)), Inf)
            mut_mtx[j, i] = mut_mtx[i, j] = mut_ixy
        end
    end
    return mut_mtx
end
