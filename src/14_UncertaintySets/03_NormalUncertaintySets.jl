"""
    struct NormalUncertaintySet{T1, T2, T3, T4, T5, T6} <: AbstractUncertaintySetEstimator
        pe::T1
        alg::T2
        n_sim::T3
        q::T4
        rng::T5
        seed::T6
    end

Estimator for box or ellipse uncertainty sets under the assumption of normally distributed returns in portfolio optimisation.

# Fields

  - `pe`: Prior estimator used to compute mean and covariance statistics.
  - `alg`: Uncertainty set algorithm (box or ellipse).
  - `n_sim`: Number of simulations for uncertainty set estimation.
  - `q`: Quantile or confidence level for uncertainty set bounds.
  - `rng`: Random number generator for simulation.
  - `seed`: Optional random seed for reproducibility.

# Constructors

    NormalUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                         alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                         n_sim::Integer = 3_000, q::Number = 0.05,
                         rng::Random.AbstractRNG = Random.default_rng(),
                         seed::Option{<:Integer} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - `n_sim > 0`.
  - `0 < q < 1`.

# Examples

```jldoctest
julia> NormalUncertaintySet()
NormalUncertaintySet
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
      q ┼ Float64: 0.05
    rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
   seed ┴ nothing
```

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
"""
struct NormalUncertaintySet{T1, T2, T3, T4, T5, T6} <: AbstractUncertaintySetEstimator
    pe::T1
    alg::T2
    n_sim::T3
    q::T4
    rng::T5
    seed::T6
    function NormalUncertaintySet(pe::AbstractPriorEstimator,
                                  alg::AbstractUncertaintySetAlgorithm, n_sim::Integer,
                                  q::Number, rng::Random.AbstractRNG,
                                  seed::Option{<:Integer})
        @argcheck(zero(n_sim) < n_sim)
        @argcheck(zero(q) < q < one(q))
        return new{typeof(pe), typeof(alg), typeof(n_sim), typeof(q), typeof(rng),
                   typeof(seed)}(pe, alg, n_sim, q, rng, seed)
    end
end
function NormalUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                              alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                              n_sim::Integer = 3_000, q::Number = 0.05,
                              rng::Random.AbstractRNG = Random.default_rng(),
                              seed::Option{<:Integer} = nothing)
    return NormalUncertaintySet(pe, alg, n_sim, q, rng, seed)
end
"""
    commutation_matrix(X::MatNum)

Constructs the commutation matrix for a given matrix.

# Arguments

  - `X`: Input matrix. Used to determine the dimensions of the commutation matrix.

# Returns

  - `cX::SparseArrays.SparseMatrixCSC{<:Number, <:Integer}`: Returns a sparse commutation matrix of size `(m*n, m*n)` where `m, n = size(X)`.

# Details

  - The commutation matrix is used to vectorize and permute matrix elements.
  - It is constructed by reshaping and transposing the indices of the input matrix.
  - The output is a sparse matrix for efficiency.

# Examples

```jldoctest
julia> PortfolioOptimisers.commutation_matrix(rand(2, 2))
4×4 SparseArrays.SparseMatrixCSC{Float64, Int64} with 4 stored entries:
 1.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0   ⋅ 
  ⋅   1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0

julia> PortfolioOptimisers.commutation_matrix(rand(2, 3))
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 6 stored entries:
 1.0   ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅   1.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅ 
  ⋅   1.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0   ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0

julia> PortfolioOptimisers.commutation_matrix(rand(3, 2))
6×6 SparseArrays.SparseMatrixCSC{Float64, Int64} with 6 stored entries:
 1.0   ⋅    ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅   1.0   ⋅    ⋅ 
  ⋅   1.0   ⋅    ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅   1.0   ⋅ 
  ⋅    ⋅   1.0   ⋅    ⋅    ⋅ 
  ⋅    ⋅    ⋅    ⋅    ⋅   1.0
```
"""
function commutation_matrix(X::MatNum)
    m, n = size(X)
    mn = m * n
    row = 1:mn
    col = vec(transpose(reshape(row, m, n)))
    data = range(1, 1; length = mn)
    return SparseArrays.sparse(row, col, data, mn, mn)
end
"""
    ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for mean and covariance statistics under the assumption of normally distributed returns.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates covariance matrices using the Wishart distribution.
  - Computes lower and upper bounds for covariance using quantiles of simulated values.
  - Computes upper bound for mean using the normal quantile and simulated covariance.
  - Returns both sets as a tuple.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                      <:Any}, X::MatNum, F::Option{<:MatNum} = nothing;
             dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    T, N = size(pr.X)
    sigma = pr.sigma
    q = ue.q * 0.5
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = Statistics.quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = Statistics.quantile(sigma_ij, one(q) - q)
        end
    end
    posdef!(ue.pe.ce.mp.pdm, sigma_l)
    posdef!(ue.pe.ce.mp.pdm, sigma_u)
    mu_u = Distributions.cquantile(Distributions.Normal(), q) *
           sqrt.(LinearAlgebra.diag(sigma_mu)) *
           2
    mu_l = range(zero(eltype(sigma)), zero(eltype(sigma)); length = N)
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u),
           BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for expected returns under the assumption of normally distributed returns.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Computes the upper bound for mean using the normal quantile and simulated covariance.
  - Sets the lower bound for mean to zero.
  - Ignores additional arguments and keyword arguments except those passed to the prior estimator.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                         <:Any}, X::MatNum, F::Option{<:MatNum} = nothing;
                dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    q = ue.q * 0.5
    mu_u = Distributions.cquantile(Distributions.Normal(), q) *
           sqrt.(LinearAlgebra.diag(sigma / size(pr.X, 1))) *
           2
    mu_l = range(zero(eltype(sigma)), zero(eltype(sigma)); length = size(pr.X, 2))
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
              X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for covariance under the assumption of normally distributed returns.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates covariance matrices using the Wishart distribution.
  - Computes lower and upper bounds for covariance using quantiles of simulated values.
  - Ensures positive definiteness of the bounds.
  - Ignores additional arguments and keyword arguments except those passed to the prior estimator.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any,
                                            <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    T, N = size(pr.X)
    sigma = pr.sigma
    q = ue.q * 0.5
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = getindex.(sigmas[:], i, j)
            sigma_l[j, i] = sigma_l[i, j] = Statistics.quantile(sigma_ij, q)
            sigma_u[j, i] = sigma_u[i, j] = Statistics.quantile(sigma_ij, one(q) - q)
        end
    end
    posdef!(ue.pe.ce.mp.pdm, sigma_l)
    posdef!(ue.pe.ce.mp.pdm, sigma_u)
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
"""
    ucs(ue::NormalUncertaintySet{<:Any,
                                 <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipse uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipseUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates mean and covariance using multivariate normal and Wishart distributions.
  - Constructs commutation matrix for covariance vectorization.
  - Computes scaling parameters for mean and covariance using the specified ellipse algorithm.
  - Returns both sets as a tuple.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                       <:Any}, <:Any, <:Any,
                                      <:Any}, X::MatNum, F::Option{<:MatNum} = nothing;
             dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, mu, sigma) = pr
    T, N = size(X)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    X_mu = transpose(rand(ue.rng, Distributions.MvNormal(mu, sigma), ue.n_sim))
    sigmas = rand(ue.rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i in axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    X_sigma = transpose(reshape(X_sigma, N^2, :))
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
"""
    ucs(ue::NormalUncertaintySet{<:Any,
                                 <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipse uncertainty sets for expected returns and covariance statistics using the chi-squared scaling algorithm under the assumption of normally distributed returns.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipseUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Constructs commutation matrix for covariance vectorization.
  - Computes scaling parameters for mean and covariance using the chi-squared scaling algorithm.
  - Applies diagonal processing if specified in the algorithm.
  - Returns both sets as a tuple.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                       <:Any}, <:Any, <:Any,
                                      <:Any}, X::MatNum, F::Option{<:MatNum} = nothing;
             dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
"""
    ucs(ue::NormalUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                 <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipse uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns, using a generic ellipse algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipseUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Constructs commutation matrix for covariance vectorization.
  - Computes scaling parameters for mean and covariance using the provided ellipse algorithm.
  - Applies diagonal processing if specified in the algorithm.
  - Returns both sets as a tuple.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                      <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q)
    k_sigma = k_ucs(ue.alg.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet()),
           EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for expected returns under the assumption of normally distributed returns, using a normal scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates mean vectors using the multivariate normal distribution.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Computes the scaling parameter for the ellipse using the normal scaling algorithm.
  - Returns an ellipse uncertainty set for the mean.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, mu, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    X_mu = transpose(rand(ue.rng, Distributions.MvNormal(mu, sigma), ue.n_sim))
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for expected returns under the assumption of normally distributed returns, using a chi-squared scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Computes the scaling parameter for the ellipse using the chi-squared scaling algorithm.
  - Returns an ellipse uncertainty set for the mean.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                          <:Any}, <:Any,
                                         <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_mu)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for expected returns under the assumption of normally distributed returns, using a chi-squared scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipseUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Computes the scaling parameter for the ellipse using the chi-squared scaling algorithm.
  - Returns an ellipse uncertainty set for the mean.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                         <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                 class = MuEllipseUncertaintySet())
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                       <:Any, <:Any, <:Any},
              X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for covariance under the assumption of normally distributed returns, using a normal scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipseUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates covariance matrices using the Wishart distribution.
  - Constructs commutation matrix for covariance vectorization.
  - Computes the scaling parameter for the ellipse using the normal scaling algorithm.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Returns an ellipse uncertainty set for covariance.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T, N = size(X)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i in axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    X_sigma = transpose(reshape(X_sigma, N^2, :))
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                       <:Any, <:Any, <:Any},
              X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for covariance under the assumption of normally distributed returns, using a chi-squared scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipseUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Constructs commutation matrix for covariance vectorization.
  - Computes the scaling parameter for the ellipse using the chi-squared scaling algorithm.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Returns an ellipse uncertainty set for covariance.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                             <:Any}, <:Any,
                                            <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                       <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipse uncertainty set for covariance under the assumption of normally distributed returns, using a generic ellipse algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - `dims`: Dimension along which to compute statistics.
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipseUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Constructs commutation matrix for covariance vectorization.
  - Computes the scaling parameter for the ellipse using the provided ellipse algorithm.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Returns an ellipse uncertainty set for covariance.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipseUncertaintySetAlgorithm`](@ref)
  - [`EllipseUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipseUncertaintySetAlgorithm{<:Any, <:Any},
                                            <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; X, sigma) = pr
    T = size(X, 1)
    sigma_mu = sigma / T
    posdef!(ue.pe.ce.mp.pdm, sigma_mu)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(ue.pe.ce.mp.pdm, sigma_sigma)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q)
    return EllipseUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                 class = SigmaEllipseUncertaintySet())
end

export NormalUncertaintySet
