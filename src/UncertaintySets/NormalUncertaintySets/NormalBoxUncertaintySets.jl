function uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any, <:BoxUncertaintySetClass,
                                                           <:Any, <:Any, <:Any},
                         X::AbstractMatrix, args...; dims::Int = 1)
    T, N = size(X)
    pm = prior(ue.pe, X, args...; dims = dims)
    sigma = pm.sigma
    q = ue.q
    sigma_mu = sigma / T
    Random.seed!(ue.rng, ue.seed)
    sigmas = rand(Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j ∈ 1:N
        for i ∈ j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    mu_u = range(; start = zero(eltype(sigma)), stop = zero(eltype(sigma)), length = N)
    mu_l = cquantile(Normal(), q * 0.5) * sqrt.(diag(sigma))
    return BoxUncertaintySet(; lo = mu_l, hi = mu_u),
           BoxUncertaintySet(; lo = sigma_l, hi = sigma_u)
end
function mu_uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any,
                                                              <:BoxUncertaintySetClass,
                                                              <:Any, <:Any, <:Any},
                            X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    sigma = pm.sigma
    q = ue.q
    mu_u = range(; start = zero(eltype(sigma)), stop = zero(eltype(sigma)),
                 length = size(X, 1))
    mu_l = cquantile(Normal(), q * 0.5) * sqrt.(diag(sigma))
    return BoxUncertaintySet(; lo = mu_l, hi = mu_u)
end
function sigma_uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any,
                                                                 <:BoxUncertaintySetClass,
                                                                 <:Any, <:Any, <:Any},
                               X::AbstractMatrix, args...; dims::Int = 1)
    T, N = size(X)
    pm = prior(ue.pe, X, args...; dims = dims)
    sigma = pm.sigma
    q = ue.q
    sigma_mu = sigma / T
    Random.seed!(ue.rng, ue.seed)
    sigmas = rand(Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j ∈ 1:N
        for i ∈ j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    return BoxUncertaintySet(; lo = sigma_l, hi = sigma_u)
end
