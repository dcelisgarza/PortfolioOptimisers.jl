function ucs(ue::NormalUncertaintySetEstimator{<:Any, <:BoxUncertaintySetClass, <:Any,
                                               <:Any, <:Any}, X::AbstractMatrix, args...;
             dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    T, N = size(pm.X)
    sigma = pm.sigma
    q = ue.q * 0.5
    sigma_mu = sigma / T
    Random.seed!(ue.rng, ue.seed)
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j ∈ 1:N
        for i ∈ j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    fix_non_positive_definite_matrix!(ue.pe.ce.mp.fnpdm, sigma_l)
    fix_non_positive_definite_matrix!(ue.pe.ce.mp.fnpdm, sigma_u)
    mu_u = cquantile(Normal(), q) * sqrt.(diag(sigma_mu)) * 2
    mu_l = range(; start = zero(eltype(sigma)), stop = zero(eltype(sigma)), length = N)
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u),
           BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
function mu_uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any,
                                                              <:BoxUncertaintySetClass,
                                                              <:Any, <:Any, <:Any},
                            X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    sigma = pm.sigma
    q = ue.q * 0.5
    mu_u = cquantile(Normal(), q) * sqrt.(diag(sigma / size(pm.X, 1))) * 2
    mu_l = range(; start = zero(eltype(sigma)), stop = zero(eltype(sigma)),
                 length = size(pm.X, 2))
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u)
end
function sigma_uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any,
                                                                 <:BoxUncertaintySetClass,
                                                                 <:Any, <:Any, <:Any},
                               X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    T, N = size(pm.X)
    sigma = pm.sigma
    q = ue.q * 0.5
    sigma_mu = sigma / T
    Random.seed!(ue.rng, ue.seed)
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j ∈ 1:N
        for i ∈ j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    fix_non_positive_definite_matrix!(ue.pe.ce.mp.fnpdm, sigma_l)
    fix_non_positive_definite_matrix!(ue.pe.ce.mp.fnpdm, sigma_u)
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
