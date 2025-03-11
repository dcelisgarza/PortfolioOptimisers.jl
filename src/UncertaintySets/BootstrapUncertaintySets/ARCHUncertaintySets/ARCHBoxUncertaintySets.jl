function uncertainty_set(ue::ARCHUncertaintySetEstimator{<:Any, <:BoxUncertaintySetClass,
                                                         <:Any, <:Any, <:Any, <:Any, <:Any},
                         X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    N = size(pm.X, 2)
    mus, sigmas = bootstrap_generator(ue, pm.X)
    q = ue.q * 0.5
    mu_l = Vector{eltype(pm.X)}(undef, N)
    mu_u = Vector{eltype(pm.X)}(undef, N)
    sigma_l = Matrix{eltype(pm.X)}(undef, N, N)
    sigma_u = Matrix{eltype(pm.X)}(undef, N, N)
    for j ∈ 1:N
        mu_j = mus[j, :]
        mu_l[j] = quantile(mu_j, q)
        mu_u[j] = quantile(mu_j, one(q) - q)
        for i ∈ j:N
            sigma_ij = sigmas[i, j, :]
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    return BoxUncertaintySet(; lo = mu_l, hi = mu_u),
           BoxUncertaintySet(; lo = sigma_l, hi = sigma_u)
end
function mu_uncertainty_set(ue::ARCHUncertaintySetEstimator{<:Any, <:BoxUncertaintySetClass,
                                                            <:Any, <:Any, <:Any, <:Any,
                                                            <:Any}, X::AbstractMatrix,
                            args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    N = size(pm.X, 2)
    mus = mu_bootstrap_generator(ue, pm.X)
    q = ue.q * 0.5
    mu_l = Vector{eltype(pm.X)}(undef, N)
    mu_u = Vector{eltype(pm.X)}(undef, N)
    for j ∈ 1:N
        mu_j = mus[j, :]
        mu_l[j] = quantile(mu_j, q)
        mu_u[j] = quantile(mu_j, one(q) - q)
    end
    return BoxUncertaintySet(; lo = mu_l, hi = mu_u)
end
function sigma_uncertainty_set(ue::ARCHUncertaintySetEstimator{<:Any,
                                                               <:BoxUncertaintySetClass,
                                                               <:Any, <:Any, <:Any, <:Any,
                                                               <:Any}, X::AbstractMatrix,
                               args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    N = size(pm.X, 2)
    sigmas = sigma_bootstrap_generator(ue, pm.X)
    q = ue.q * 0.5
    sigma_l = Matrix{eltype(pm.X)}(undef, N, N)
    sigma_u = Matrix{eltype(pm.X)}(undef, N, N)
    for j ∈ 1:N
        for i ∈ j:N
            sigma_ij = sigmas[i, j, :]
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    return BoxUncertaintySet(; lo = sigma_l, hi = sigma_u)
end
