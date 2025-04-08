function ucs(ue::ARCHUncertaintySetEstimator{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                             <:Any, <:Any, <:Any, <:Any}, X::AbstractMatrix,
             args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    N = size(pm.X, 2)
    mus, sigmas = bootstrap_generator(ue, pm.X)
    X_mu = Matrix{eltype(pm.X)}(undef, N, ue.n_sim)
    X_sigma = Matrix{eltype(pm.X)}(undef, N^2, ue.n_sim)
    for i ∈ axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pm.mu)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pm.sigma)
    end
    X_mu = transpose(X_mu)
    X_sigma = transpose(X_sigma)
    sigma_mu, sigma_sigma = if ue.class.diagonal
        Diagonal(cov(ue.pe.ce, X_mu)), Diagonal(cov(ue.pe.ce, X_sigma))
    else
        cov(ue.pe.ce, X_mu), cov(ue.pe.ce, X_sigma)
    end
    k_mu = k_ucs(ue.class.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_ucs(ue.class.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
function mu_ucs(ue::ARCHUncertaintySetEstimator{<:Any, <:EllipseUncertaintySetAlgorithm,
                                                <:Any, <:Any, <:Any, <:Any, <:Any},
                X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    N = size(pm.X, 2)
    mus = mu_bootstrap_generator(ue, pm.X)
    X_mu = Matrix{eltype(pm.X)}(undef, N, ue.n_sim)
    for i ∈ axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pm.mu)
    end
    X_mu = transpose(X_mu)
    sigma_mu = if ue.class.diagonal
        Diagonal(cov(ue.pe.ce, X_mu))
    else
        cov(ue.pe.ce, X_mu)
    end
    k_mu = k_ucs(ue.class.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu)
end
function sigma_ucs(ue::ARCHUncertaintySetEstimator{<:Any, <:EllipseUncertaintySetAlgorithm,
                                                   <:Any, <:Any, <:Any, <:Any, <:Any},
                   X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    N = size(pm.X, 2)
    sigmas = sigma_bootstrap_generator(ue, pm.X)
    X_sigma = Matrix{eltype(pm.X)}(undef, N^2, ue.n_sim)
    for i ∈ axes(X_sigma, 2)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pm.sigma)
    end
    X_sigma = transpose(X_sigma)
    sigma_sigma = if ue.class.diagonal
        Diagonal(cov(ue.pe.ce, X_sigma))
    else
        cov(ue.pe.ce, X_sigma)
    end
    k_sigma = k_ucs(ue.class.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
