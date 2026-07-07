"""
$(DocStringExtensions.TYPEDEF)

Estimator for box or ellipsoidal uncertainty sets under the assumption of normally distributed returns in portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormalUncertaintySet(;
        pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
        alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
        n_sim::Integer = 3_000,
        q::Number = 0.05,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        ens::Option{<:Number} = nothing,
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        kwargs::NamedTuple = (;),
    ) -> NormalUncertaintySet

Keywords correspond to the struct's fields.

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
         │           │      │   alg ┴ FullMoment()
         │           │   mp ┼ MatrixProcessing
         │           │      │     pdm ┼ Posdef
         │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │           │      │      dn ┼ nothing
         │           │      │      dt ┼ nothing
         │           │      │     alg ┼ nothing
         │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │        me ┼ SimpleExpectedReturns
         │           │   w ┴ nothing
         │   horizon ┴ nothing
     alg ┼ BoxUncertaintySetAlgorithm()
   n_sim ┼ Int64: 3000
       q ┼ Float64: 0.05
     rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
    seed ┼ nothing
     ens ┼ nothing
     pdm ┼ Posdef
         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │   kwargs ┴ @NamedTuple{}: NamedTuple()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
"""
@concrete struct NormalUncertaintySet <: AbstractUncertaintySetEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:ucsa])
    """
    alg
    """
    $(field_dict[:n_sim])
    """
    n_sim
    """
    $(field_dict[:q_bs])
    """
    q
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    """
    $(field_dict[:ens])
    """
    ens
    """
    $(field_dict[:pdm])
    """
    pdm
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function NormalUncertaintySet(pe::AbstractLowOrderPriorEstimator,
                                  alg::AbstractUncertaintySetAlgorithm, n_sim::Integer,
                                  q::Number, rng::Random.AbstractRNG,
                                  seed::Option{<:Integer}, ens::Option{<:Number},
                                  pdm::Option{<:AbstractPosdefEstimator},
                                  kwargs::NamedTuple)
        @argcheck(zero(n_sim) < n_sim, DomainError(n_sim, "n_sim must be > 0"))
        assert_unit_interval(q, :q)
        return new{typeof(pe), typeof(alg), typeof(n_sim), typeof(q), typeof(rng),
                   typeof(seed), typeof(ens), typeof(pdm), typeof(kwargs)}(pe, alg, n_sim,
                                                                           q, rng, seed,
                                                                           ens, pdm, kwargs)
    end
end
function NormalUncertaintySet(; pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
                              alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                              n_sim::Integer = 3_000, q::Number = 0.05,
                              rng::Random.AbstractRNG = Random.default_rng(),
                              seed::Option{<:Integer} = nothing,
                              ens::Option{<:Number} = nothing,
                              pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                              kwargs::NamedTuple = (;))::NormalUncertaintySet
    return NormalUncertaintySet(pe, alg, n_sim, q, rng, seed, ens, pdm, kwargs)
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
    choose_scaling_parameter(ue, pr)

Select the scaling parameter for a normal uncertainty set from the prior result.

Internal helper that extracts the appropriate scaling parameter from `ue` given the prior result `pr`. Dispatches on the uncertainty set and prior types.

# Arguments

  - `ue`: [`NormalUncertaintySet`](@ref) estimator.
  - `pr`: Prior result.

# Returns

  - Scaling parameter value.

# Related

  - [`NormalUncertaintySet`](@ref)
"""
function choose_scaling_parameter(ue::NormalUncertaintySet, pr::LowOrderPrior)
    return if !isnothing(ue.ens)
        ue.ens
    elseif !isnothing(pr.ens)
        pr.ens
    else
        size(pr.X, 1)
    end
end
"""
    mu_asymptotic_cov(pdm, sigma, T)

Asymptotic covariance of the mean estimator, ``\\hat{\\mathbf{\\Sigma}} / T``, projected to the
nearest positive definite matrix with `pdm`. Shared by the box and ellipsoidal
[`NormalUncertaintySet`](@ref) constructions.

# Related

  - [`sigma_asymptotic_cov`](@ref)
  - [`NormalUncertaintySet`](@ref)
"""
function mu_asymptotic_cov(pdm::Option{<:AbstractPosdefEstimator}, sigma::MatNum, T::Number)
    sigma_mu = sigma / T
    posdef!(pdm, sigma_mu)
    return sigma_mu
end
"""
    sigma_asymptotic_cov(pdm, sigma_mu, sigma, T)

Asymptotic (vectorised) covariance of the covariance estimator,
``T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\mu} \\otimes \\mathbf{\\Sigma}_{\\mu})``,
projected to the nearest positive definite matrix with `pdm`. `sigma_mu` is the (raw,
pre-diagonalisation) mean asymptotic covariance from [`mu_asymptotic_cov`](@ref); `K` is the
[`commutation_matrix`](@ref) of `sigma`.

# Related

  - [`mu_asymptotic_cov`](@ref)
  - [`NormalUncertaintySet`](@ref)
"""
function sigma_asymptotic_cov(pdm::Option{<:AbstractPosdefEstimator}, sigma_mu::MatNum,
                              sigma::MatNum, T::Number)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(pdm, sigma_sigma)
    return sigma_sigma
end
"""
    ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for mean and covariance statistics under the assumption of normally distributed returns.

# Mathematical definition

Simulates ``M`` covariance matrices ``\\hat{\\mathbf{\\Sigma}}^{(m)} \\sim \\mathrm{Wishart}(T,\\, \\hat{\\mathbf{\\Sigma}}/T)`` and computes element-wise quantile bounds:

```math
\\begin{align}
(\\mathbf{\\Sigma}_{lb})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{ub})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Mean bounds use the normal complementary quantile ``z_{q/2} = \\Phi^{-1}(1-q/2)``:

```math
\\begin{align}
\\boldsymbol{\\mu}_{lb} &= \\boldsymbol{0}\\,, \\\\
\\mu_{ub,i} &= 2 z_{q/2} \\sqrt{(\\hat{\\mathbf{\\Sigma}}/T)_{ii}}\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{lb})_{ij}``, ``(\\mathbf{\\Sigma}_{ub})_{ij}``: Element-wise lower/upper covariance bounds.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Lower and upper quantile functions at level ``q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: ``(i,j)``-element of the ``m``-th simulated Wishart covariance draw.
  - ``\\boldsymbol{\\mu}_{lb}``, ``\\mu_{ub,i}``: Lower and upper bounds for expected returns.
  - ``z_{q/2} = \\Phi^{-1}(1 - q/2)``: Normal complementary quantile.
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - $(math_dict[:T])
  - ``q``: Significance level.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
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
    N = size(pr.X, 2)
    T = choose_scaling_parameter(ue, pr)
    sigma = pr.sigma
    q = ue.q * 0.5
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = getindex.(sigmas, i, j)
            sigma_l[j, i] = sigma_l[i, j] = Statistics.quantile(sigma_ij, q; ue.kwargs...)
            sigma_u[j, i] = sigma_u[i, j] = Statistics.quantile(sigma_ij, one(q) - q;
                                                                ue.kwargs...)
        end
    end
    posdef!(ue.pdm, sigma_l)
    posdef!(ue.pdm, sigma_u)
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

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}_{lb} &= \\boldsymbol{0}\\,, \\\\
\\mu_{ub,i} &= 2 \\Phi^{-1}\\!\\left(1 - \\tfrac{q}{2}\\right) \\sqrt{(\\hat{\\mathbf{\\Sigma}}/T)_{ii}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{lb}``, ``\\mu_{ub,i}``: Lower and upper bounds for expected returns.
  - ``\\Phi^{-1}(1 - q/2)``: Normal complementary quantile.
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - $(math_dict[:T])
  - ``q``: Significance level.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
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
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, pr.sigma, T)
    q = ue.q * 0.5
    mu_u = Distributions.cquantile(Distributions.Normal(), q) *
           sqrt.(LinearAlgebra.diag(sigma_mu)) *
           2
    mu_l = range(zero(eltype(sigma_mu)), zero(eltype(sigma_mu)); length = size(pr.X, 2))
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
              X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for covariance under the assumption of normally distributed returns.

# Mathematical definition

Simulates ``M`` matrices ``\\hat{\\mathbf{\\Sigma}}^{(m)} \\sim \\mathrm{Wishart}(T,\\,\\hat{\\mathbf{\\Sigma}}/T)`` and takes element-wise quantiles:

```math
\\begin{align}
(\\mathbf{\\Sigma}_{lb})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{ub})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{lb})_{ij}``, ``(\\mathbf{\\Sigma}_{ub})_{ij}``: Element-wise lower/upper covariance bounds.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Lower and upper quantile functions at level ``q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: ``(i,j)``-element of the ``m``-th Wishart draw.
  - $(math_dict[:T])
  - ``q``: Significance level.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
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
    N = size(pr.X, 2)
    T = choose_scaling_parameter(ue, pr)
    sigma = pr.sigma
    q = ue.q * 0.5
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    sigma_l = Matrix{eltype(sigma)}(undef, N, N)
    sigma_u = Matrix{eltype(sigma)}(undef, N, N)
    for j in 1:N
        for i in j:N
            sigma_ij = getindex.(sigmas, i, j)
            sigma_l[j, i] = sigma_l[i, j] = Statistics.quantile(sigma_ij, q; ue.kwargs...)
            sigma_u[j, i] = sigma_u[i, j] = Statistics.quantile(sigma_ij, one(q) - q;
                                                                ue.kwargs...)
        end
    end
    posdef!(ue.pdm, sigma_l)
    posdef!(ue.pdm, sigma_u)
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u)
end
"""
    ucs(ue::NormalUncertaintySet{<:Any,
                                 <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipsoidal uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns.

# Mathematical definition

Ellipsoidal sets centred at the prior estimates with asymptotic covariances:

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\mu} &= \\hat{\\mathbf{\\Sigma}} / T\\,, \\\\
\\mathbf{\\Sigma}_{\\Sigma} &= T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\mu} \\otimes \\mathbf{\\Sigma}_{\\mu})\\,.
\\end{align}
```

The scaling ``k`` is fitted empirically from simulated samples ``\\hat{\\boldsymbol{\\mu}}^{(m)} \\sim \\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}})`` and ``\\hat{\\mathbf{\\Sigma}}^{(m)} \\sim \\mathrm{Wishart}(T, \\mathbf{\\Sigma}_{\\mu})``:

```math
\\begin{align}
\\mathcal{E}_{\\mu} &= \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\mu}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\mu}^2 \\right\\}\\,.
\\end{align}
```

```math
\\begin{align}
\\mathcal{E}_{\\Sigma} &= \\left\\{\\mathbf{\\Sigma} : \\left\\| \\mathbf{\\Sigma}_{\\Sigma}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}}) \\right\\|_2 \\leq k_{\\Sigma} \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\mu}``: Asymptotic covariance of the mean estimator.
  - ``\\mathbf{\\Sigma}_{\\Sigma}``: Asymptotic covariance of the covariance estimator (vectorised).
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - $(math_dict[:T])
  - ``\\mathbf{I}``: Identity matrix.
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``\\mathcal{E}_{\\mu}``: Ellipsoidal uncertainty set for expected returns.
  - ``\\mathcal{E}_{\\Sigma}``: Ellipsoidal uncertainty set for covariance.
  - ``k_{\\mu}``, ``k_{\\Sigma}``: Empirically fitted scaling parameters.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates mean and covariance using multivariate normal and Wishart distributions.
  - Constructs commutation matrix for covariance vectorization.
  - Computes scaling parameters for mean and covariance using the specified ellipsoidal algorithm.
  - Returns both sets as a tuple.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                           <:Any}, <:Any,
                                      <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; mu, sigma) = pr
    N = size(pr.X, 2)
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
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
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipsoidalUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                     class = MuEllipsoidalUncertaintySet()),
           EllipsoidalUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                     class = SigmaEllipsoidalUncertaintySet())
end
"""
    ucs(ue::NormalUncertaintySet{<:Any,
                                 <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipsoidal uncertainty sets for expected returns and covariance statistics using the chi-squared scaling algorithm under the assumption of normally distributed returns.

# Mathematical definition

Asymptotic covariances are the same as the normal-``k`` variant:

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\mu} &= \\hat{\\mathbf{\\Sigma}} / T\\,, \\\\
\\mathbf{\\Sigma}_{\\Sigma} &= T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\mu} \\otimes \\mathbf{\\Sigma}_{\\mu})\\,.
\\end{align}
```

The scaling ``k`` is the chi-squared quantile:

```math
\\begin{align}
k_{\\mu} &= \\sqrt{\\chi^2_{N,\\,1-q}}\\,, \\\\
k_{\\Sigma} &= \\sqrt{\\chi^2_{N^2,\\,1-q}}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\mu}``: Asymptotic covariance of the mean estimator.
  - ``\\mathbf{\\Sigma}_{\\Sigma}``: Asymptotic covariance of the covariance estimator (vectorised).
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - $(math_dict[:T])
  - ``\\mathbf{I}``: Identity matrix.
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``k_{\\mu}``, ``k_{\\Sigma}``: Chi-squared scaling parameters for mean and covariance ellipsoids.
  - ``\\chi^2_{N,\\,1-q}``: ``(1-q)``-quantile of the chi-squared distribution with ``N`` degrees of freedom.
  - ``N``: Number of assets.
  - ``q``: Significance level.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Constructs commutation matrix for covariance vectorization.
  - Computes scaling parameters for mean and covariance using the chi-squared scaling algorithm.
  - Applies diagonal processing if specified in the algorithm.
  - Returns both sets as a tuple.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                           <:Any}, <:Any,
                                      <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_mu)
    k_sigma = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipsoidalUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                     class = MuEllipsoidalUncertaintySet()),
           EllipsoidalUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                     class = SigmaEllipsoidalUncertaintySet())
end
"""
    ucs(ue::NormalUncertaintySet{<:Any, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                 <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipsoidal uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns, using a generic ellipsoidal algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty sets.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Constructs commutation matrix for covariance vectorization.
  - Computes scaling parameters for mean and covariance using the provided ellipsoidal algorithm.
  - Applies diagonal processing if specified in the algorithm.
  - Returns both sets as a tuple.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                      <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q)
    k_sigma = k_ucs(ue.alg.method, ue.q)
    return EllipsoidalUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                     class = MuEllipsoidalUncertaintySet()),
           EllipsoidalUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                     class = SigmaEllipsoidalUncertaintySet())
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for expected returns under the assumption of normally distributed returns, using a normal scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates mean vectors using the multivariate normal distribution.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Computes the scaling parameter for the ellipsoidal using the normal scaling algorithm.
  - Returns an ellipsoidal uncertainty set for the mean.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                              <:Any}, <:Any,
                                         <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; mu, sigma) = pr
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    X_mu = transpose(rand(ue.rng, Distributions.MvNormal(mu, sigma), ue.n_sim))
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, X_mu, sigma_mu)
    return EllipsoidalUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                     class = MuEllipsoidalUncertaintySet())
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for expected returns under the assumption of normally distributed returns, using a chi-squared scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Computes the scaling parameter for the ellipsoidal using the chi-squared scaling algorithm.
  - Returns an ellipsoidal uncertainty set for the mean.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                              <:Any}, <:Any,
                                         <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_mu)
    return EllipsoidalUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                     class = MuEllipsoidalUncertaintySet())
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for expected returns under the assumption of normally distributed returns, using a chi-squared scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Computes the scaling parameter for the ellipsoidal using the chi-squared scaling algorithm.
  - Returns an ellipsoidal uncertainty set for the mean.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                         <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    if ue.alg.diagonal
        sigma_mu = LinearAlgebra.Diagonal(sigma_mu)
    end
    k_mu = k_ucs(ue.alg.method, ue.q)
    return EllipsoidalUncertaintySet(; sigma = sigma_mu, k = k_mu,
                                     class = MuEllipsoidalUncertaintySet())
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                       <:Any, <:Any, <:Any},
              X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for covariance under the assumption of normally distributed returns, using a normal scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Simulates covariance matrices using the Wishart distribution.
  - Constructs commutation matrix for covariance vectorization.
  - Computes the scaling parameter for the ellipsoidal using the normal scaling algorithm.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Returns an ellipsoidal uncertainty set for covariance.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                                 <:Any},
                                            <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    N = size(pr.X, 2)
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    if !isnothing(ue.seed)
        Random.seed!(ue.rng, ue.seed)
    end
    sigmas = rand(ue.rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i in axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    X_sigma = transpose(reshape(X_sigma, N^2, :))
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, X_sigma, sigma_sigma)
    return EllipsoidalUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                     class = SigmaEllipsoidalUncertaintySet())
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any},
                                       <:Any, <:Any, <:Any},
              X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for covariance under the assumption of normally distributed returns, using a chi-squared scaling algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Constructs commutation matrix for covariance vectorization.
  - Computes the scaling parameter for the ellipsoidal using the chi-squared scaling algorithm.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Returns an ellipsoidal uncertainty set for covariance.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`ChiSqKUncertaintyAlgorithm`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm,
                                                                                 <:Any},
                                            <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q, 1:(ue.n_sim), sigma_sigma)
    return EllipsoidalUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                     class = SigmaEllipsoidalUncertaintySet())
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                       <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for covariance under the assumption of normally distributed returns, using a generic ellipsoidal algorithm.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

# Details

  - Computes prior statistics using the provided prior estimator.
  - Processes the covariance matrix for positive definiteness.
  - Constructs commutation matrix for covariance vectorization.
  - Computes the scaling parameter for the ellipsoidal using the provided ellipsoidal algorithm.
  - Applies diagonal processing to the covariance if specified in the algorithm.
  - Returns an ellipsoidal uncertainty set for covariance.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:Any,
                                                                                 <:Any},
                                            <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    if ue.alg.diagonal
        sigma_sigma = LinearAlgebra.Diagonal(sigma_sigma)
    end
    k_sigma = k_ucs(ue.alg.method, ue.q)
    return EllipsoidalUncertaintySet(; sigma = sigma_sigma, k = k_sigma,
                                     class = SigmaEllipsoidalUncertaintySet())
end

export NormalUncertaintySet
