abstract type ARCHBootstrapSet end
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
struct ARCHUncertaintySetEstimator{T1 <: AbstractPriorEstimator, T2 <: UncertaintySetClass,
                                   T3 <: ARCHBootstrapSet, T4 <: Integer, T5 <: Integer,
                                   T6 <: Real, T7 <: Union{Nothing, <:Integer}} <:
       BootsrapUncertaintySetEstimator
    pe::T1
    class::T2
    bootstrap::T3
    n_sim::T4
    block_size::T5
    q::T6
    seed::T7
end
function ARCHUncertaintySetEstimator(;
                                     pe::AbstractPriorEstimator = EmpiricalPriorEstimator(),
                                     class::UncertaintySetClass = BoxUncertaintySetClass(),
                                     bootstrap::ARCHBootstrapSet = StationaryBootstrap(),
                                     n_sim::Integer = 3_000, block_size::Integer = 3,
                                     q::Real = 0.05,
                                     seed::Union{<:Integer, Nothing} = nothing)
    @smart_assert(n_sim > zero(n_sim))
    @smart_assert(block_size > zero(block_size))
    @smart_assert(zero(q) < q < one(q))
    return ARCHUncertaintySetEstimator{typeof(pe), typeof(class), typeof(bootstrap),
                                       typeof(n_sim), typeof(block_size), typeof(q),
                                       typeof(seed)}(pe, class, bootstrap, n_sim,
                                                     block_size, q, seed)
end
function bootstrap_generator(ue::ARCHUncertaintySetEstimator, X::AbstractMatrix)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, Py(X).to_numpy(), ue.seed)
    for (i, data) ∈ enumerate(gen.bootstrap(ue.n_sim))
        X = pyconvert(Array, data)[1][1]
        mu = mean(ue.pe.me, X; dims = 1)
        mus[:, i] = vec(mu)
        sigmas[:, :, i] = cov(ue.pe.ce, X; dims = 1, mean = mu)
    end
    return mus, sigmas
end
function mu_bootstrap_generator(ue::ARCHUncertaintySetEstimator, X::AbstractMatrix)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, Py(X).to_numpy(), ue.seed)
    for (i, data) ∈ enumerate(gen.bootstrap(ue.n_sim))
        X = pyconvert(Array, data)[1][1]
        mu = mean(ue.pe.me, X; dims = 1)
        mus[:, i] .= vec(mu)
    end
    return mus
end
function sigma_bootstrap_generator(ue::ARCHUncertaintySetEstimator, X::AbstractMatrix)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, Py(X).to_numpy(), ue.seed)
    for (i, data) ∈ enumerate(gen.bootstrap(ue.n_sim))
        X = pyconvert(Array, data)[1][1]
        mu = mean(ue.pe.me, X; dims = 1)
        sigmas[:, :, i] .= cov(ue.pe.ce, X; dims = 1, mean = mu)
    end
    return sigmas
end

export StationaryBootstrap, CircularBootstrap, MovingBootstrap, ARCHUncertaintySetEstimator
