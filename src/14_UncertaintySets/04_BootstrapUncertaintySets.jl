"""
    abstract type BootstrapUncertaintySetEstimator <: AbstractUncertaintySetEstimator end

Abstract type for estimators that construct uncertainty sets using bootstrap methods in portfolio optimisation.

Subtypes implement specific bootstrap algorithms (e.g., stationary, circular, moving block) to estimate uncertainty sets for risk or prior statistics.

# Related Types

  - [`ARCHUncertaintySet`](@ref)
"""
abstract type BootstrapUncertaintySetEstimator <: AbstractUncertaintySetEstimator end
"""
    abstract type ARCHBootstrapSet <: AbstractAlgorithm end

Abstract type for bootstrap algorithms used in constructing uncertainty sets for time series data in portfolio optimisation.

Subtypes implement specific bootstrap methods using [`archpy`](https://pypi.org/project/arch/).

# Related Types

  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
"""
abstract type ARCHBootstrapSet <: AbstractAlgorithm end
"""
    struct StationaryBootstrap <: ARCHBootstrapSet end

Bootstrap algorithm for constructing uncertainty sets using a [stationary bootstrap](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.StationaryBootstrap.html#arch.bootstrap.StationaryBootstrap) in time series data.

# Related Types

  - [`ARCHBootstrapSet`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
"""
struct StationaryBootstrap <: ARCHBootstrapSet end
"""
    struct CircularBootstrap <: ARCHBootstrapSet end

Bootstrap algorithm for constructing uncertainty sets using a [circular bootstrap](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html#arch.bootstrap.CircularBlockBootstrap) in time series data.

# Related Types

  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
"""
struct CircularBootstrap <: ARCHBootstrapSet end
"""
    struct MovingBootstrap <: ARCHBootstrapSet end

Bootstrap algorithm for constructing uncertainty sets using a [moving bootstrap](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.MovingBlockBootstrap.html#arch.bootstrap.MovingBlockBootstrap) in time series data.

# Related Types

  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
"""
struct MovingBootstrap <: ARCHBootstrapSet end
"""
    bootstrap_func(alg::ARCHBootstrapSet, block_size, X, seed)

Creates a bootstrap generator for time series data using the specified bootstrap algorithm.

# Arguments

  - `alg`: Bootstrap algorithm type.
  - `block_size`: Size of blocks for resampling.
  - `X`: Data matrix to be resampled.
  - `seed`: Random seed for reproducibility.

# Returns

  - Returns a Python bootstrap generator object from the `arch.bootstrap` package.

# Details

  - Dispatches to the appropriate bootstrap algorithm based on `alg`.
  - Uses Python's `arch.bootstrap` via `pyimport` for stationary, circular, or moving block bootstraps.
  - The generator can be used to produce resampled datasets for uncertainty set estimation.

# Related

  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
  - [`ARCHBootstrapSet`](@ref)
"""
function bootstrap_func(::StationaryBootstrap, block_size, X, seed)
    return PythonCall.pyimport("arch.bootstrap").StationaryBootstrap(block_size, X;
                                                                     seed = seed)
end
function bootstrap_func(::CircularBootstrap, block_size, X, seed)
    return PythonCall.pyimport("arch.bootstrap").CircularBlockBootstrap(block_size, X;
                                                                        seed = seed)
end
function bootstrap_func(::MovingBootstrap, block_size, X, seed)
    return PythonCall.pyimport("arch.bootstrap").MovingBlockBootstrap(block_size, X;
                                                                      seed = seed)
end
"""
    struct ARCHUncertaintySet{T1, T2, T3, T4, T5, T6, T7} <: BootstrapUncertaintySetEstimator
        pe::T1
        alg::T2
        n_sim::T3
        block_size::T4
        q::T5
        seed::T6
        bootstrap::T7
    end

Estimator for box or ellipse uncertainty sets using bootstrap methods for time series data in portfolio optimisation.

# Fields

  - `pe`: Prior estimator used to compute expected returns and covariance statistics.
  - `alg`: Uncertainty set algorithm.
  - `n_sim`: Number of bootstrap simulations for uncertainty set estimation.
  - `block_size`: Size of blocks for bootstrap resampling.
  - `q`: Quantile or confidence level for uncertainty set bounds.
  - `seed`: Optional random seed for reproducibility.
  - `bootstrap`: Bootstrap algorithm type.

# Constructors

    ARCHUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                       alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                       n_sim::Integer = 3_000, block_size::Integer = 3, q::Number = 0.05,
                       seed::Option{<:Integer} = nothing,
                       bootstrap::ARCHBootstrapSet = StationaryBootstrap())

Keyword arguments correspond to the fields above.

## Validation

  - `n_sim > 0`.
  - `block_size > 0`.
  - `0 < q < 1`.

# Examples

```jldoctest
julia> ARCHUncertaintySet()
ARCHUncertaintySet
          pe ┼ EmpiricalPrior
             │        ce ┼ PortfolioOptimisersCovariance
             │           │   ce ┼ Covariance
             │           │      │    me ┼ SimpleExpectedReturns
             │           │      │       │   w ┴ nothing
             │           │      │    ce ┼ GeneralCovariance
             │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │           │      │       │    w ┴ nothing
             │           │      │   alg ┴ Full()
             │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
             │           │      │       pdm ┼ Posdef
             │           │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton       
             │           │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │           │      │   denoise ┼ nothing
             │           │      │    detone ┼ nothing
             │           │      │       alg ┼ nothing
             │           │      │     order ┴ DenoiseDetoneAlg()
             │        me ┼ SimpleExpectedReturns
             │           │   w ┴ nothing
             │   horizon ┴ nothing
         alg ┼ BoxUncertaintySetAlgorithm()
       n_sim ┼ Int64: 3000
  block_size ┼ Int64: 3
           q ┼ Float64: 0.05
        seed ┼ nothing
   bootstrap ┴ StationaryBootstrap()
```

# Related

  - [`BootstrapUncertaintySetEstimator`](@ref)
  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
"""
struct ARCHUncertaintySet{T1, T2, T3, T4, T5, T6, T7} <: BootstrapUncertaintySetEstimator
    pe::T1
    alg::T2
    n_sim::T3
    block_size::T4
    q::T5
    seed::T6
    bootstrap::T7
    function ARCHUncertaintySet(pe::AbstractPriorEstimator,
                                alg::AbstractUncertaintySetAlgorithm, n_sim::Integer,
                                block_size::Integer, q::Number, seed::Option{<:Integer},
                                bootstrap::ARCHBootstrapSet)
        @argcheck(n_sim > zero(n_sim))
        @argcheck(block_size > zero(block_size))
        @argcheck(zero(q) < q < one(q))
        return new{typeof(pe), typeof(alg), typeof(n_sim), typeof(block_size), typeof(q),
                   typeof(seed), typeof(bootstrap)}(pe, alg, n_sim, block_size, q, seed,
                                                    bootstrap)
    end
end
function ARCHUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                            alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                            n_sim::Integer = 3_000, block_size::Integer = 3,
                            q::Number = 0.05, seed::Option{<:Integer} = nothing,
                            bootstrap::ARCHBootstrapSet = StationaryBootstrap())
    return ARCHUncertaintySet(pe, alg, n_sim, block_size, q, seed, bootstrap)
end
"""
    bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)

Generates bootstrapped samples of expected returns and covariance statistics for time series data using the specified bootstrap algorithm.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mus::Matrix{<:Number}`: Matrix of bootstrapped expected return vectors (`size(X, 2) × ue.n_sim`).
  - `sigmas::Array{<:Number, 3})`: Array of bootstrapped covariance matrices (`size(X, 2) × size(X, 2) × ue.n_sim`).

# Details

  - Uses the bootstrap algorithm specified in `ue.bootstrap` to generate resampled datasets.
  - For each bootstrap sample, computes the expected return and covariance using the prior estimator in `ue.pe`.
  - Stores the bootstrapped expected returns and covariances for uncertainty set estimation.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`bootstrap_func`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, PythonCall.Py(X).to_numpy(), ue.seed)
    for (i, data) in enumerate(gen.bootstrap(ue.n_sim))
        X = PythonCall.pyconvert(Array, data)[1][1]
        mu = Statistics.mean(ue.pe.me, X; dims = 1, kwargs...)
        mus[:, i] = vec(mu)
        sigmas[:, :, i] = Statistics.cov(ue.pe.ce, X; dims = 1, kwargs...)
    end
    return mus, sigmas
end
"""
    mu_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)

Generates bootstrap samples of expected return vectors for returns data using the specified bootstrap algorithm.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mus::Matrix{<:Number}`: Matrix of bootstrapped expected return vectors (`size(X, 2) × ue.n_sim`).

# Details

  - Uses the bootstrap algorithm specified in `ue.bootstrap` to generate resampled datasets.
  - For each bootstrap sample, computes the expected return using the prior estimator in `ue.pe`.
  - Stores the bootstrapped expected returns for uncertainty set estimation.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`bootstrap_func`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, PythonCall.Py(X).to_numpy(), ue.seed)
    for (i, data) in enumerate(gen.bootstrap(ue.n_sim))
        X = PythonCall.pyconvert(Array, data)[1][1]
        mu = Statistics.mean(ue.pe.me, X; dims = 1, kwargs...)
        mus[:, i] = vec(mu)
    end
    return mus
end
"""
    sigma_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)

Generates bootstrap samples of covariance matrices for time series data using the specified bootstrap algorithm.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

    - `sigmas::Array{<:Number, 3}`: Array of bootstrapped covariance matrices (`size(X, 2) × size(X, 2) × ue.n_sim`).

# Details

  - Uses the bootstrap algorithm specified in `ue.bootstrap` to generate resampled datasets.
  - For each bootstrap sample, computes the covariance using the prior estimator in `ue.pe`.
  - Stores the bootstrapped covariances for uncertainty set estimation.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`bootstrap_func`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
"""
function sigma_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    gen = bootstrap_func(ue.bootstrap, ue.block_size, PythonCall.Py(X).to_numpy(), ue.seed)
    for (i, data) in enumerate(gen.bootstrap(ue.n_sim))
        X = PythonCall.pyconvert(Array, data)[1][1]
        sigmas[:, :, i] = Statistics.cov(ue.pe.ce, X; dims = 1, kwargs...)
    end
    return sigmas
end
"""
    ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                               <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for expected returns and covariance statistics using bootstrap resampling for time series data.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Generates bootstrap samples of expected returns and covariance using the specified bootstrap algorithm.
  - Computes lower and upper bounds for expected returns and covariance using quantiles of bootstrapped samples.
  - Returns both sets as a tuple.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
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
        mu_l[j] = Statistics.quantile(mu_j, q)
        mu_u[j] = Statistics.quantile(mu_j, one(q) - q)
        for i in j:N
            sigma_ij = sigmas[i, j, :]
            sigma_l[j, i] = sigma_l[i, j] = Statistics.quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = Statistics.quantile(sigma_ij, one(q) - q)
        end
    end
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u),
           BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
"""
    mu_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                  <:Any, <:Any, <:Any}, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for expected returns using bootstrap resampling for time series data.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Generates bootstrap samples of expected returns using the specified bootstrap algorithm.
  - Computes lower and upper bounds for expected returns using quantiles of bootstrapped samples.
  - Returns the expected returns box uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    mus = mu_bootstrap_generator(ue, pr.X; kwargs...)
    q = ue.q * 0.5
    mu_l = Vector{eltype(pr.X)}(undef, N)
    mu_u = Vector{eltype(pr.X)}(undef, N)
    for j in 1:N
        mu_j = mus[j, :]
        mu_l[j] = Statistics.quantile(mu_j, q)
        mu_u[j] = Statistics.quantile(mu_j, one(q) - q)
    end
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u)
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                     <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for covariance using bootstrap resampling for time series data.

# Arguments

  - `ue`: ARCH uncertainty set estimator. Contains prior estimator, box algorithm, simulation parameters, block size, quantile, seed, and bootstrap type.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Generates bootstrap samples of covariance using the specified bootstrap algorithm.
  - Computes lower and upper bounds for covariance using quantiles of bootstrapped samples.
  - Returns the covariance box uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                          <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    sigmas = sigma_bootstrap_generator(ue, pr.X; kwargs...)
    q = ue.q * 0.5
    sigma_l = Matrix{eltype(pr.X)}(undef, N, N)
    sigma_u = Matrix{eltype(pr.X)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = sigmas[i, j, :]
            sigma_l[j, i] = sigma_l[i, j] = Statistics.quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = Statistics.quantile(sigma_ij, one(q) - q)
        end
    end
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
"""
    ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any, <:Any,
                               <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipse uncertainty sets for expected returns and covariance statistics using bootstrap resampling for time series data.

# Arguments

  - `ue`: ARCH uncertainty set estimator. Contains prior estimator, ellipse algorithm, simulation parameters, block size, quantile, seed, and bootstrap type.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Ellipse uncertainty set for expected returns.
  - `sigma_ucs::EllipseUncertaintySet`: Ellipse uncertainty set for covariance.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Generates bootstrap samples of expected returns and covariance using the specified bootstrap algorithm.
  - Computes deviations from prior statistics for each bootstrap sample.
  - Computes covariance matrices of deviations and constructs ellipse uncertainty sets using the specified method and quantile.
  - Returns both sets as a tuple.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
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
    sigma_mu = Statistics.cov(ue.pe.ce, X_mu)
    sigma_sigma = Statistics.cov(ue.pe.ce, X_sigma)
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
"""
    mu_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any, <:Any,
                                  <:Any, <:Any, <:Any}, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for expected returns using bootstrap resampling for time series data.

# Arguments

  - `ue`: ARCH uncertainty set estimator. Contains prior estimator, ellipse algorithm, simulation parameters, block size, quantile, seed, and bootstrap type.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Ellipse uncertainty set for expected returns.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Generates bootstrap samples of expected returns using the specified bootstrap algorithm.
  - Computes deviations from prior expected returns for each bootstrap sample.
  - Computes the covariance matrix of deviations and constructs an ellipse uncertainty set using the specified method and quantile.
  - Returns the expected returns ellipse uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                       <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    mus = mu_bootstrap_generator(ue, pr.X; kwargs...)
    X_mu = Matrix{eltype(pr.X)}(undef, N, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pr.mu)
    end
    X_mu = transpose(X_mu)
    sigma_mu = Statistics.cov(ue.pe.ce, X_mu)
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any, <:Any,
                                     <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for covariance using bootstrap resampling for time series data.

# Arguments

  - `ue`: ARCH uncertainty set estimator. Contains prior estimator, ellipse algorithm, simulation parameters, block size, quantile, seed, and bootstrap type.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipseUncertaintySet`: Ellipse uncertainty set for covariance.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Generates bootstrap samples of covariance using the specified bootstrap algorithm.
  - Computes deviations from prior covariance for each bootstrap sample.
  - Computes the covariance matrix of deviations and constructs an ellipse uncertainty set using the specified method and quantile.
  - Returns the covariance ellipse uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                          <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    N = size(pr.X, 2)
    sigmas = sigma_bootstrap_generator(ue, pr.X; kwargs...)
    X_sigma = Matrix{eltype(pr.X)}(undef, N^2, ue.n_sim)
    for i in axes(X_sigma, 2)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pr.sigma)
    end
    X_sigma = transpose(X_sigma)
    sigma_sigma = Statistics.cov(ue.pe.ce, X_sigma)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end

export StationaryBootstrap, CircularBootstrap, MovingBootstrap, ARCHUncertaintySet
