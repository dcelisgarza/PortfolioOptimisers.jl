function ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                               <:EllipseUncertaintySetClass{<:Any,
                                                                            <:NormalKUncertaintyMethod},
                                               <:Any, <:Any, <:Any}, X::AbstractMatrix,
             args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, mu, sigma) = pm
    T, N = size(X)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    Random.seed!(ue.rng, ue.seed)
    X_mu = transpose(rand(ue.rng, MvNormal(mu, sigma), ue.n_sim))
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i ∈ axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    X_sigma = transpose(reshape(X_sigma, N^2, :))
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.class.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.class.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_ucs(ue.class.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
function ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                               <:EllipseUncertaintySetClass{<:Any,
                                                                            <:ChiSqKUncertaintyMethod},
                                               <:Any, <:Any, <:Any}, X::AbstractMatrix,
             args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    T = size(X, 1)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.class.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.class.method, ue.q, 1:(ue.n_sim), sigma_mu)
    k_sigma = k_ucs(ue.class.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
function ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                               <:EllipseUncertaintySetClass{<:Any, <:Any},
                                               <:Any, <:Any, <:Any}, X::AbstractMatrix,
             args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    T = size(X, 1)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.class.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.class.method, ue.q)
    k_sigma = k_ucs(ue.class.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
function mu_ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                                  <:EllipseUncertaintySetClass{<:Any,
                                                                               <:NormalKUncertaintyMethod},
                                                  <:Any, <:Any, <:Any}, X::AbstractMatrix,
                args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, mu, sigma) = pm
    T = size(X, 1)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    Random.seed!(ue.rng, ue.seed)
    X_mu = transpose(rand(ue.rng, MvNormal(mu, sigma), ue.n_sim))
    if ue.class.diagonal
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.class.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu)
end
function mu_ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                                  <:EllipseUncertaintySetClass{<:Any,
                                                                               <:ChiSqKUncertaintyMethod},
                                                  <:Any, <:Any, <:Any}, X::AbstractMatrix,
                args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    T = size(X, 1)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    if ue.class.diagonal
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.class.method, ue.q, 1:(ue.n_sim), sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu)
end
function mu_ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                                  <:EllipseUncertaintySetClass{<:Any,
                                                                               <:Any},
                                                  <:Any, <:Any, <:Any}, X::AbstractMatrix,
                args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    T = size(X, 1)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    if ue.class.diagonal
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.class.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu)
end
function sigma_ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                                     <:EllipseUncertaintySetClass{<:Any,
                                                                                  <:NormalKUncertaintyMethod},
                                                     <:Any, <:Any, <:Any},
                   X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    T, N = size(X)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    Random.seed!(ue.rng, ue.seed)
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i ∈ axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    X_sigma = transpose(reshape(X_sigma, N^2, :))
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.class.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.class.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
function sigma_ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                                     <:EllipseUncertaintySetClass{<:Any,
                                                                                  <:ChiSqKUncertaintyMethod},
                                                     <:Any, <:Any, <:Any},
                   X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    T = size(X, 1)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.class.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.class.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
function sigma_ucs(ue::NormalUncertaintySetEstimator{<:Any,
                                                     <:EllipseUncertaintySetClass{<:Any,
                                                                                  <:Any},
                                                     <:Any, <:Any, <:Any},
                   X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; X, sigma) = pm
    T = size(X, 1)
    sigma_mu = sigma / T
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    fit_estimator!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.class.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.class.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma)
end
