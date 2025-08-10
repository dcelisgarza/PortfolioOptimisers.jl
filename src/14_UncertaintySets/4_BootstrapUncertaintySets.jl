abstract type BootsrapUncertaintySetEstimator <: AbstractUncertaintySetEstimator end
abstract type ARCHBootstrapSet <: AbstractAlgorithm end
struct StationaryBootstrap <: ARCHBootstrapSet end
struct CircularBootstrap <: ARCHBootstrapSet end
struct MovingBootstrap <: ARCHBootstrapSet end
function bootstrap_func(::StationaryBootstrap, block_size, X, seed)
    return pyimport("arch.bootstrap").StationaryBootstrap(block_size, X; seed = seed)
end
function bootstrap_func(::CircularBootstrap, block_size, X, seed)
    return pyimport("arch.bootstrap").CircularBlockBootstrap(block_size, X; seed = seed)
end
function bootstrap_func(::MovingBootstrap, block_size, X, seed)
    return pyimport("arch.bootstrap").MovingBlockBootstrap(block_size, X; seed = seed)
end
struct ARCHUncertaintySet{T1, T2, T3, T4, T5, T6, T7} <: BootsrapUncertaintySetEstimator
    pe::T1
    alg::T2
    n_sim::T3
    block_size::T4
    q::T5
    seed::T6
    bootstrap::T7
end
function ARCHUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                            alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                            n_sim::Integer = 3_000, block_size::Integer = 3, q::Real = 0.05,
                            seed::Union{Nothing, <:Integer} = nothing,
                            bootstrap::ARCHBootstrapSet = StationaryBootstrap())
    @argcheck(n_sim > zero(n_sim))
    @argcheck(block_size > zero(block_size))
    @argcheck(zero(q) < q < one(q))
    return ARCHUncertaintySet(pe, alg, n_sim, block_size, q, seed, bootstrap)
end
function bootstrap_generator(ue::ARCHUncertaintySet, X::AbstractMatrix; kwargs...)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, Py(X).to_numpy(), ue.seed)
    for (i, data) in enumerate(gen.bootstrap(ue.n_sim))
        X = pyconvert(Array, data)[1][1]
        mu = mean(ue.pe.me, X; dims = 1, kwargs...)
        mus[:, i] = vec(mu)
        sigmas[:, :, i] = cov(ue.pe.ce, X; dims = 1, mean = mu, kwargs...)
    end
    return mus, sigmas
end
function mu_bootstrap_generator(ue::ARCHUncertaintySet, X::AbstractMatrix; kwargs...)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, Py(X).to_numpy(), ue.seed)
    for (i, data) in enumerate(gen.bootstrap(ue.n_sim))
        X = pyconvert(Array, data)[1][1]
        mu = mean(ue.pe.me, X; dims = 1, kwargs...)
        mus[:, i] .= vec(mu)
    end
    return mus
end
function sigma_bootstrap_generator(ue::ARCHUncertaintySet, X::AbstractMatrix; kwargs...)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, Py(X).to_numpy(), ue.seed)
    for (i, data) in enumerate(gen.bootstrap(ue.n_sim))
        X = pyconvert(Array, data)[1][1]
        mu = mean(ue.pe.me, X; dims = 1, kwargs...)
        sigmas[:, :, i] .= cov(ue.pe.ce, X; dims = 1, mean = mu, kwargs...)
    end
    return sigmas
end
function ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    mus, sigmas = bootstrap_generator(ue, pr.X; kwargs...)
    q = ue.q * 0.5
    mu_l = Vector{eltype(pr.X)}(undef, N)
    mu_u = Vector{eltype(pr.X)}(undef, N)
    sigma_l = Matrix{eltype(pr.X)}(undef, N, N)
    sigma_u = Matrix{eltype(pr.X)}(undef, N, N)
    for j in 1:N
        mu_j = mus[j, :]
        mu_l[j] = quantile(mu_j, q)
        mu_u[j] = quantile(mu_j, one(q) - q)
        for i in j:N
            sigma_ij = sigmas[i, j, :]
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u),
           BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
function mu_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    mus = mu_bootstrap_generator(ue, pr.X; kwargs...)
    q = ue.q * 0.5
    mu_l = Vector{eltype(pr.X)}(undef, N)
    mu_u = Vector{eltype(pr.X)}(undef, N)
    for j in 1:N
        mu_j = mus[j, :]
        mu_l[j] = quantile(mu_j, q)
        mu_u[j] = quantile(mu_j, one(q) - q)
    end
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u)
end
function sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                          <:Any, <:Any, <:Any}, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    sigmas = sigma_bootstrap_generator(ue, pr.X; kwargs...)
    q = ue.q * 0.5
    sigma_l = Matrix{eltype(pr.X)}(undef, N, N)
    sigma_u = Matrix{eltype(pr.X)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = sigmas[i, j, :]
            sigma_l[j, i] = sigma_l[i, j] = quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = quantile(sigma_ij, one(q) - q)
        end
    end
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
function ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    mus, sigmas = bootstrap_generator(ue, pr.X; kwargs...)
    X_mu = Matrix{eltype(pr.X)}(undef, N, ue.n_sim)
    X_sigma = Matrix{eltype(pr.X)}(undef, N^2, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pr.mu)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pr.sigma)
    end
    X_mu = transpose(X_mu)
    X_sigma = transpose(X_sigma)
    sigma_mu, sigma_sigma = if ue.alg.diagonal
        Diagonal(cov(ue.pe.ce, X_mu)), Diagonal(cov(ue.pe.ce, X_sigma))
    else
        cov(ue.pe.ce, X_mu), cov(ue.pe.ce, X_sigma)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
function mu_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                       <:Any, <:Any, <:Any, <:Any}, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    mus = mu_bootstrap_generator(ue, pr.X; kwargs...)
    X_mu = Matrix{eltype(pr.X)}(undef, N, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pr.mu)
    end
    X_mu = transpose(X_mu)
    sigma_mu = if ue.alg.diagonal
        Diagonal(cov(ue.pe.ce, X_mu))
    else
        cov(ue.pe.ce, X_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
function sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                          <:Any, <:Any, <:Any, <:Any}, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    sigmas = sigma_bootstrap_generator(ue, pr.X; kwargs...)
    X_sigma = Matrix{eltype(pr.X)}(undef, N^2, ue.n_sim)
    for i in axes(X_sigma, 2)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pr.sigma)
    end
    X_sigma = transpose(X_sigma)
    sigma_sigma = if ue.alg.diagonal
        Diagonal(cov(ue.pe.ce, X_sigma))
    else
        cov(ue.pe.ce, X_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end

export StationaryBootstrap, CircularBootstrap, MovingBootstrap, ARCHUncertaintySet
