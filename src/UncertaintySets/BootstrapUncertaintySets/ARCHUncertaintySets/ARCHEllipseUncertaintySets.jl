function uncertainty_set(ue::ARCHUncertaintySetEstimator{<:Any, <:EllipseUncertaintySet,
                                                         <:Any, <:Any, <:Any, <:Any, <:Any},
                         X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, mu, sigma) = pm
    N = size(X, 2)
    mus, sigmas = bootstrap_generator(ue, X)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    for i ∈ 1:(ue.n_sim)
        X_sigma[:, i] = vec(sigmas[:, :, i] - sigma)
        X_mu[:, i] = vec(mus[:, i] - mu)
    end
    sigma_sigma = cov(ue.pe.ce, X_sigma)
    sigma_mu = cov(ue.pe.ce, X_mu)
    if ue.class.diagonal
        sigma_sigma .= Diagonal(sigma_sigma)
        sigma_mu .= Diagonal(sigma_mu)
    end
    k_sigma = k_uncertainty_set(ue.class.method, ue.q, X_sigma, sigma_sigma)
    k_mu = k_uncertainty_set(ue.class.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
function mu_uncertainty_set(ue::ARCHUncertaintySetEstimator{<:Any, <:EllipseUncertaintySet,
                                                            <:Any, <:Any, <:Any, <:Any,
                                                            <:Any}, X::AbstractMatrix,
                            args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, mu) = pm
    N = size(X, 2)
    mus = mu_bootstrap_generator(ue, X)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    for i ∈ 1:(ue.n_sim)
        X_mu[:, i] = vec(mus[:, i] - mu)
    end
    sigma_mu = cov(ue.pe.ce, X_mu)
    if ue.class.diagonal
        sigma_mu .= Diagonal(sigma_mu)
    end
    k_mu = k_uncertainty_set(ue.class.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu)
end
function sigma_uncertainty_set(ue::ARCHUncertaintySetEstimator{<:Any,
                                                               <:EllipseUncertaintySet,
                                                               <:Any, <:Any, <:Any, <:Any,
                                                               <:Any}, X::AbstractMatrix,
                               args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    N = size(X, 2)
    sigmas = sigma_bootstrap_generator(ue, X)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i ∈ 1:(ue.n_sim)
        X_sigma[:, i] = vec(sigmas[:, :, i] - sigma)
    end
    sigma_sigma = cov(ue.pe.ce, X_sigma)
    if ue.class.diagonal
        sigma_sigma .= Diagonal(sigma_sigma)
    end
    k_sigma = k_uncertainty_set(ue.class.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end