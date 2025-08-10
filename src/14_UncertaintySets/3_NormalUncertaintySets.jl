struct NormalUncertaintySet{T1, T2, T3, T4, T5, T6} <: AbstractUncertaintySetEstimator
    pe::T1
    alg::T2
    n_sim::T3
    q::T4
    rng::T5
    seed::T6
end
function NormalUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                              alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                              n_sim::Integer = 3_000, q::Real = 0.05,
                              rng::AbstractRNG = Random.default_rng(),
                              seed::Union{Nothing, <:Integer} = nothing)
    @argcheck(n_sim > zero(n_sim))
    @argcheck(zero(q) < q < one(q))
    return NormalUncertaintySet(pe, alg, n_sim, q, rng, seed)
end
function commutation_matrix(x::AbstractMatrix)
    m, n = size(x)
    mn = m * n
    row = 1:mn
    col = vec(transpose(reshape(row, m, n)))
    data = range(; start = 1, stop = 1, length = mn)
    return sparse(row, col, data, mn, mn)
end
function ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                      <:Any}, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    T, N = size(pr.X)
    sigma = pr.sigma
    q = ue.q * 0.5
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    posdef!(ue.pe.ce.mp.pdm, sigma_l)
    posdef!(ue.pe.ce.mp.pdm, sigma_u)
    mu_u = cquantile(Normal(), q) * sqrt.(diag(sigma_mu)) * 2
    mu_l = range(; start = zero(eltype(sigma)), stop = zero(eltype(sigma)), length = N)
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u),
           BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
function mu_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                         <:Any}, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    q = ue.q * 0.5
    mu_u = cquantile(Normal(), q) * sqrt.(diag(sigma / size(pr.X, 1))) * 2
    mu_l = range(; start = zero(eltype(sigma)), stop = zero(eltype(sigma)),
                 length = size(pr.X, 2))
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u)
end
function sigma_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any,
                                            <:Any, <:Any}, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    T, N = size(pr.X)
    sigma = pr.sigma
    q = ue.q * 0.5
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    posdef!(ue.pe.ce.mp.pdm, sigma_l)
    posdef!(ue.pe.ce.mp.pdm, sigma_u)
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                       <:Any}, <:Any, <:Any,
                                      <:Any}, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, mu, sigma) = pr
    T, N = size(X)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    X_mu = transpose(rand(ue.rng, MvNormal(mu, sigma), ue.n_sim))
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i in axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    X_sigma = transpose(reshape(X_sigma, N^2, :))
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                       <:Any}, <:Any, <:Any,
                                      <:Any}, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
function ucs(ue::NormalUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                      <:Any, <:Any, <:Any}, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q)
    k_sigma = k_ucs(ue.alg.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, mu, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    X_mu = transpose(rand(ue.rng, MvNormal(mu, sigma), ue.n_sim))
    if ue.alg.diagonal
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if ue.alg.diagonal
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                         <:Any, <:Any, <:Any}, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if ue.alg.diagonal
        sigma_mu = Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T, N = size(X)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i in axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    X_sigma = transpose(reshape(X_sigma, N^2, :))
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                            <:Any, <:Any, <:Any}, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end

export NormalUncertaintySet
