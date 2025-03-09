function uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any,
                                                           <:EllipseUncertaintySetClass{<:Any,
                                                                                        <:NormalKUncertaintyMethod},
                                                           <:Any, <:Any, <:Any},
                         X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, mu, sigma) = pm
    T, N = size(X)
    sigma_mu = sigma / T
    Random.seed!(ue.rng, ue.seed)
    X_mu = transpose(rand(MvNormal(mu, sigma), ue.n_sim))
    sigmas = rand(Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Matrix{eltype(sigma)}(N^2, ue.n_sim)
    for i ∈ axes(X_sigma, 2)
        X_sigma[:, i] .= vec(sigmas[i] - sigma)
    end
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    if ue.class.diagonal
        sigma_sigma .= Diagonal(sigma_sigma)
        sigma_mu .= Diagonal(sigma_mu)
    end
    k_mu = k_uncertainty_set(ue.class.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_uncertainty_set(ue.class.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
    return nothing
end
function uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any,
                                                           <:EllipseUncertaintySetClass{<:Any,
                                                                                        <:ChiSqKUncertaintyMethod},
                                                           <:Any, <:Any, <:Any},
                         X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, mu, sigma) = pm
    T, N = size(X)
    sigma_mu = sigma / T
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    if ue.class.diagonal
        sigma_sigma .= Diagonal(sigma_sigma)
        sigma_mu .= Diagonal(sigma_mu)
    end
    k_mu = k_uncertainty_set(ue.class.method, ue.q, 1:(ue.n_sim), sigma_mu)
    k_sigma = k_uncertainty_set(ue.class.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
    return nothing
end
function uncertainty_set(ue::NormalUncertaintySetEstimator{<:Any,
                                                           <:EllipseUncertaintySetClass{<:Any,
                                                                                        <:Any},
                                                           <:Any, <:Any, <:Any},
                         X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, mu, sigma) = pm
    T, N = size(X)
    sigma_mu = sigma / T
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    if ue.class.diagonal
        sigma_sigma .= Diagonal(sigma_sigma)
        sigma_mu .= Diagonal(sigma_mu)
    end
    k_mu = k_uncertainty_set(ue.class.method, ue.q)
    k_sigma = k_uncertainty_set(ue.class.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
    return nothing
end
