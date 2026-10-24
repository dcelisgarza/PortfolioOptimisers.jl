"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the rules that build the diagonal covariance shape of a normal uncertainty set.

The rules are for a [`NormalUncertaintySet`](@ref). The shape of the covariance set is the asymptotic covariance of ``\\operatorname{vec}(\\hat{\\mathbf{\\Sigma}})``, of size ``N^{2} \\times N^{2}``. An algorithm with `diagonal = true` keeps one variance for each entry of ``\\hat{\\mathbf{\\Sigma}}``, and each subtype states which variance. The mean axis does not read the rule, because its diagonal shape is the diagonal of ``\\hat{\\mathbf{\\Sigma}} / T`` under every subtype.

# Interfaces

## `diagonal_sigma_shape`

  - `diagonal_sigma_shape(dc::AbstractDiagonalConstruction, pdm::Option{<:AbstractPosdefEstimator}, sigma_mu::MatNum, sigma::MatNum, T::Number) -> LinearAlgebra.Diagonal`: Returns the diagonal covariance shape, ``N^{2} \\times N^{2}``, with non-negative entries.

# Related

  - [`DiagonalOfShape`](@ref)
  - [`ShapeOfDiagonal`](@ref)
  - [`diagonal_sigma_shape`](@ref)
  - [`NormalUncertaintySet`](@ref)
"""
abstract type AbstractDiagonalConstruction <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Takes the diagonal of the full covariance shape.

This is the default rule. Each entry is the exact variance of the matching entry of ``\\hat{\\mathbf{\\Sigma}}``, at the covariance matrix at the centre of the set. So the diagonal set and the full set, with `diagonal = false`, use the same sampling law.

# Mathematical definition

```math
\\begin{align}
\\left(\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}\\right)_{(i,j),(i,j)} &= \\dfrac{\\hat{\\mathbf{\\Sigma}}_{ii} \\hat{\\mathbf{\\Sigma}}_{jj} + \\hat{\\mathbf{\\Sigma}}_{ij}^{2}}{T}\\,.
\\end{align}
```

Where:

  - ``\\left(\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}\\right)_{(i,j),(i,j)}``: Diagonal entry of the covariance shape at the ``(i, j)`` entry of ``\\hat{\\mathbf{\\Sigma}}``.
  - $(math_dict[:Sigma_hat_ii])
  - $(math_dict[:Sigma_hat_ij])
  - $(math_dict[:T])

It is the variance of the ``(i, j)`` entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw.

# Algorithm

The method of [`diagonal_sigma_shape`](@ref) for this type does these steps.

 1. Build the full shape with [`sigma_asymptotic_cov`](@ref), which repairs it under `pdm`.
 2. Return its diagonal as a `LinearAlgebra.Diagonal`.

# Examples

```jldoctest
julia> DiagonalOfShape()
DiagonalOfShape()
```

# Related

  - [`AbstractDiagonalConstruction`](@ref)
  - [`ShapeOfDiagonal`](@ref)
  - [`diagonal_sigma_shape`](@ref)
"""
struct DiagonalOfShape <: AbstractDiagonalConstruction end
"""
$(DocStringExtensions.TYPEDEF)

Sets the covariances of ``\\hat{\\mathbf{\\Sigma}}`` to zero, then builds the covariance shape.

Each entry is the variance of the matching entry of ``\\hat{\\mathbf{\\Sigma}}`` if ``\\hat{\\mathbf{\\Sigma}}_{ij} = 0`` for ``i \\neq j``. The centre of the set stays ``\\hat{\\mathbf{\\Sigma}}``, with its covariances. The rule drops the term ``\\hat{\\mathbf{\\Sigma}}_{ij}^{2}`` from each off-diagonal entry, and [`DiagonalOfShape`](@ref) keeps that term. So this rule is less exact, and its diagonal set can differ from the full set, with `diagonal = false`.

# Mathematical definition

```math
\\begin{align}
\\left(\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}\\right)_{(i,j),(i,j)} &= \\dfrac{\\left(1 + \\delta_{ij}\\right) \\hat{\\mathbf{\\Sigma}}_{ii} \\hat{\\mathbf{\\Sigma}}_{jj}}{T}\\,.
\\end{align}
```

Where:

  - ``\\left(\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}\\right)_{(i,j),(i,j)}``: Diagonal entry of the covariance shape at the ``(i, j)`` entry of ``\\hat{\\mathbf{\\Sigma}}``.
  - ``\\delta_{ij}``: Kronecker delta, `1` when ``i = j`` and `0` otherwise.
  - $(math_dict[:Sigma_hat_ii])
  - $(math_dict[:T])

It is the diagonal of ``T (\\mathbf{I} + \\mathbf{K})(\\mathbf{D} \\otimes \\mathbf{D})``, where ``\\mathbf{D}`` is the diagonal of ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``. The two rules give the same value at each entry ``(i, i)``.

# Algorithm

The method of [`diagonal_sigma_shape`](@ref) for this type does these steps.

 1. Read `d`, the diagonal of `sigma_mu`.
 2. Form the vector `v = vec(T .* d .* transpose(d))`, whose entry ``(i, j)`` in column-major order is ``T d_{i} d_{j}``.
 3. Double the ``N`` entries ``(i, i)``, at the stride `N + 1` of `v`.
 4. Return `LinearAlgebra.Diagonal(v)`. The method makes no ``N^{2} \\times N^{2}`` matrix, and it does not read `pdm`. A diagonal matrix with non-negative entries is positive semi-definite, and `posdef!` does not change such a matrix.

# Examples

```jldoctest
julia> ShapeOfDiagonal()
ShapeOfDiagonal()
```

# Related

  - [`AbstractDiagonalConstruction`](@ref)
  - [`DiagonalOfShape`](@ref)
  - [`diagonal_sigma_shape`](@ref)
"""
struct ShapeOfDiagonal <: AbstractDiagonalConstruction end
"""
$(DocStringExtensions.TYPEDEF)

Fits a box, an ellipsoidal or a norm-ball uncertainty set from the sampling laws of normal returns.

Under normal returns, the sample mean has a normal law, and the sample covariance has a Wishart law. Equation 11.16 of [cajas2025](@cite) states the two laws. The Wishart law is the distribution of Equation 9 of [wishart1928](@cite). Corollary 4.2 of [magnus1979](@cite) gives the covariance of that law, which is the shape of the ellipsoid on the covariance axis. Equation 11.17 of [cajas2025](@cite) takes the box from the quantiles of the two laws.

`alg` selects the shape of the set. `n_sim` sets the number of draws on each route that simulates. [`DeltaUncertaintySet`](@ref) fits a box with no sample, and [`ARCHUncertaintySet`](@ref) fits a set with no assumed law.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormalUncertaintySet(;
        pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
        alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
        n_sim::Integer = 3_000,
        q::Number = 0.05,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        ens::Option{<:Number} = nothing,
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        dc::AbstractDiagonalConstruction = DiagonalOfShape(),
        kwargs::NamedTuple = (;),
    ) -> NormalUncertaintySet

The keywords are the fields of the struct.

## Validation

  - `n_sim > 0`.
  - `0 < q < 1`.

# Examples

```jldoctest
julia> NormalUncertaintySet()
NormalUncertaintySet
      pe ┼ EmpiricalPrior
         │           ce ┼ PortfolioOptimisersCovariance
         │              │   ce ┼ Covariance
         │              │      │    me ┼ SimpleExpectedReturns
         │              │      │       │   w ┴ nothing
         │              │      │    ce ┼ GeneralCovariance
         │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │              │      │       │    w ┴ nothing
         │              │      │   alg ┼ FullMoment()
         │              │      │     w ┴ nothing
         │              │   mp ┼ MatrixProcessing
         │              │      │     pdm ┼ Posdef
         │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │              │      │      dn ┼ nothing
         │              │      │      dt ┼ nothing
         │              │      │     alg ┼ nothing
         │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │           me ┼ SimpleExpectedReturns
         │              │   w ┴ nothing
         │      horizon ┼ nothing
         │   fill_limit ┼ nothing
         │        x_src ┴ nothing
     alg ┼ BoxUncertaintySetAlgorithm()
   n_sim ┼ Int64: 3000
       q ┼ Float64: 0.05
     rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
    seed ┼ nothing
     ens ┼ nothing
     pdm ┼ Posdef
         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      dc ┼ DiagonalOfShape()
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`AbstractDiagonalConstruction`](@ref)

# References

  - $(ref_dict[:wishart1928]) Section 3, Equation 9.
  - $(ref_dict[:magnus1979]) Section 4, Corollary 4.2.
  - $(ref_dict[:cajas2025]) Equations 11.16 and 11.17. $(ref_dict[:no_original_source])
"""
@concrete struct NormalUncertaintySet <: AbstractUncertaintySetEstimator
    """
    $(field_dict[:pe_ucs])
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
    $(field_dict[:dc_ucs])
    """
    dc
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function NormalUncertaintySet(pe::Option{<:AbstractLowOrderPriorEstimator},
                                  alg::AbstractUncertaintySetAlgorithm, n_sim::Integer,
                                  q::Number, rng::Random.AbstractRNG,
                                  seed::Option{<:Integer}, ens::Option{<:Number},
                                  pdm::Option{<:AbstractPosdefEstimator},
                                  dc::AbstractDiagonalConstruction, kwargs::NamedTuple)
        @argcheck(zero(n_sim) < n_sim, DomainError(n_sim, "n_sim must be > 0"))
        assert_resource_cap(n_sim, RESOURCE_LIMITS[].max_n_sim, :n_sim, :max_n_sim)
        assert_unit_interval(q, :q)
        return new{typeof(pe), typeof(alg), typeof(n_sim), typeof(q), typeof(rng),
                   typeof(seed), typeof(ens), typeof(pdm), typeof(dc), typeof(kwargs)}(pe,
                                                                                       alg,
                                                                                       n_sim,
                                                                                       q,
                                                                                       rng,
                                                                                       seed,
                                                                                       ens,
                                                                                       pdm,
                                                                                       dc,
                                                                                       kwargs)
    end
end
function NormalUncertaintySet(;
                              pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
                              alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                              n_sim::Integer = 3_000, q::Number = 0.05,
                              rng::Random.AbstractRNG = Random.default_rng(),
                              seed::Option{<:Integer} = nothing,
                              ens::Option{<:Number} = nothing,
                              pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                              dc::AbstractDiagonalConstruction = DiagonalOfShape(),
                              kwargs::NamedTuple = (;))::NormalUncertaintySet
    return NormalUncertaintySet(pe, alg, n_sim, q, rng, seed, ens, pdm, dc, kwargs)
end
"""
    commutation_matrix(X::MatNum)

Builds the commutation matrix for the size of `X`.

The commutation matrix is the permutation that maps the column-major vectorisation of a matrix to the vectorisation of its transpose. For a square matrix, it is its own inverse. It holds one non-zero entry in each row, so the function returns a sparse matrix. The function reads the size of `X`, and it does not read the entries.

# Mathematical definition

```math
\\mathbf{K}_{m,n} \\operatorname{vec}(\\mathbf{A}) = \\operatorname{vec}\\!\\left(\\mathbf{A}^{\\intercal}\\right)\\,, \\qquad \\mathbf{A} \\in \\mathbb{R}^{m \\times n}\\,.
```

Where:

  - ``\\mathbf{K}_{m,n}``: Commutation matrix of an ``m \\times n`` matrix, of size ``mn \\times mn``.
  - ``\\mathbf{A}``: Any ``m \\times n`` matrix.
  - ``\\operatorname{vec}``: Column-major vectorisation.

For a symmetric ``\\mathbf{A}``, ``\\mathbf{K}_{n,n} \\operatorname{vec}(\\mathbf{A}) = \\operatorname{vec}(\\mathbf{A})``. So a symmetric matrix cannot show the difference between ``\\mathbf{K}`` and the identity.

# Algorithm

 1. Read `m, n = size(X)`, and set `mn = m * n`, the side of the result.
 2. Number the entries of an `m` by `n` matrix in column-major order. This gives `row = 1:mn`.
 3. Find the position of each entry in the vectorised transpose with `col = vec(transpose(reshape(row, m, n)))`.
 4. Write a one at each `(row, col)` pair. The result is the sparse `mn` by `mn` matrix `cX`.

# Arguments

  - `X`: Matrix whose size sets the size of the commutation matrix.

# Returns

  - `cX::SparseArrays.SparseMatrixCSC{<:Number, <:Integer}`: Sparse commutation matrix of size `(m*n, m*n)`, where `m, n = size(X)`.

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

# Related

  - [`sigma_asymptotic_cov`](@ref)
  - [`NormalUncertaintySet`](@ref)

# References

  - $(ref_dict[:magnus1979]) Section 3, Definition 3.1 and Theorem 3.1 (vii). $(ref_dict[:no_original_source])
  - $(ref_dict[:tracy1969])
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

Returns ``T``, the effective number of observations. The covariance divided by ``T`` is the asymptotic covariance of the mean.

The function reads three sources in order, and it returns the first that is not `nothing`. The sources are the `ens` of the estimator, the `ens` of the prior, and the row count of the returns matrix of the prior. A weighted or shrunk prior has fewer effective observations than rows. A prior under a Scenario Cap has fewer rows than the observations of its moments, and it states the larger count in `ens`. The first two sources are for these two cases.

Each set of this estimator scales with ``T``. The width of a set is proportional to ``T^{-1/2}``, so a prior with fewer effective scenarios than rows gives a wider set. A division of ``T`` by four doubles the width of the mean box.

# Algorithm

 1. If `ue.ens` is not `nothing`, return it. It is the effective count that the estimator states.
 2. Otherwise, if `pr.ens` is not `nothing`, return it. It is the effective count that the prior holds.
 3. Otherwise, return `size(pr.X, 1)`, the row count of the returns matrix of the prior.

# Arguments

  - `ue`: [`NormalUncertaintySet`](@ref) estimator.
  - `pr`: Prior Result.

# Returns

  - `T::Number`: Effective number of observations.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`mu_asymptotic_cov`](@ref)
"""
function choose_scaling_parameter(ue::NormalUncertaintySet, pr::AbstractPriorResult)
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

Returns the asymptotic covariance of the mean estimator, projected to the nearest positive definite matrix with `pdm`.

Every route of [`NormalUncertaintySet`](@ref) calls this function, so all routes use the same mean shape. A division by a positive scalar keeps the definiteness of `sigma`. So `posdef!` changes nothing when `sigma` is positive definite.

# Mathematical definition

```math
\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} = \\dfrac{\\hat{\\mathbf{\\Sigma}}}{T}\\,.
```

Where:

  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

# Algorithm

 1. Divide `sigma` by `T` to get `sigma_mu`.
 2. Repair `sigma_mu` in place with `posdef!` under `pdm`.
 3. Return `sigma_mu`.

# Arguments

  - `pdm`: Positive definite matrix estimator. `nothing` skips the repair.
  - `sigma`: Point estimate of the covariance matrix.
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref).

# Returns

  - `sigma_mu::MatNum`: Asymptotic covariance of the mean estimator.

# Related

  - [`sigma_asymptotic_cov`](@ref)
  - [`choose_scaling_parameter`](@ref)
  - [`NormalUncertaintySet`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equations 11.16 and 11.24. $(ref_dict[:no_original_source])
"""
function mu_asymptotic_cov(pdm::Option{<:AbstractPosdefEstimator}, sigma::MatNum, T::Number)
    sigma_mu = sigma / T
    posdef!(pdm, sigma_mu)
    return sigma_mu
end
"""
    sigma_asymptotic_cov(pdm, sigma_mu, sigma, T)

Returns the asymptotic covariance of the vectorised covariance estimator, projected to the nearest positive definite matrix with `pdm`.

`sigma_mu` is the full mean asymptotic covariance from [`mu_asymptotic_cov`](@ref), before any diagonal step. ``\\mathbf{K}`` is the [`commutation_matrix`](@ref) of `sigma`. The two matrix arguments depend on each other. The form below is the sampling law of a normal sample only if `sigma_mu` comes from the same `sigma` and the same `T`.

# Mathematical definition

```math
\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} = T \\left(\\mathbf{I} + \\mathbf{K}\\right) \\left(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}\\right) = \\dfrac{\\left(\\mathbf{I} + \\mathbf{K}\\right) \\left(\\hat{\\mathbf{\\Sigma}} \\otimes \\hat{\\mathbf{\\Sigma}}\\right)}{T}\\,.
```

Where:

  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the vectorised covariance estimator.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - $(math_dict[:I_identity])
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

The left-hand form is Corollary 4.2 of [magnus1979](@cite), ``\\operatorname{Var}(\\operatorname{vec} \\mathbf{S}) = k (\\mathbf{I} + \\mathbf{K})(\\mathbf{V} \\otimes \\mathbf{V})`` for ``\\mathbf{S} \\sim \\mathrm{Wishart}(k, \\mathbf{V})``, at ``k = T`` and ``\\mathbf{V} = \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``. The right-hand form is the asymptotic covariance of ``\\operatorname{vec}(\\hat{\\mathbf{\\Sigma}})`` for a normal sample. Its diagonal entry at ``(i,j)`` is ``\\left(\\hat{\\Sigma}_{ii} \\hat{\\Sigma}_{jj} + \\hat{\\Sigma}_{ij}^{2}\\right) / T``, the variance of the ``(i,j)`` entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw. So the sample that [`ucs`](@ref) draws and the shape matrix that measures the sample have the same scale.

A set with `diagonal = true` keeps this diagonal under [`DiagonalOfShape`](@ref), the default. [`ShapeOfDiagonal`](@ref) builds the diagonal from the variances alone. [`diagonal_sigma_shape`](@ref) states both rules.

# Algorithm

 1. Build `K = commutation_matrix(sigma)`. `sigma` is square, so `K` is ``N^{2} \\times N^{2}``.
 2. Form `sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)`.
 3. Repair `sigma_sigma` in place with `posdef!` under `pdm`.
 4. Return `sigma_sigma`.

# Arguments

  - `pdm`: Positive definite matrix estimator. `nothing` skips the repair.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref).
  - `sigma`: Point estimate of the covariance matrix. The function reads only its size, through [`commutation_matrix`](@ref).
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref).

# Returns

  - `sigma_sigma::MatNum`: Asymptotic covariance of the vectorised covariance estimator, ``N^{2} \\times N^{2}``.

# Related

  - [`mu_asymptotic_cov`](@ref)
  - [`commutation_matrix`](@ref)
  - [`diagonal_sigma_shape`](@ref)
  - [`NormalUncertaintySet`](@ref)

# References

  - $(ref_dict[:magnus1979]) Section 4, Corollary 4.2.
"""
function sigma_asymptotic_cov(pdm::Option{<:AbstractPosdefEstimator}, sigma_mu::MatNum,
                              sigma::MatNum, T::Number)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(pdm, sigma_sigma)
    return sigma_sigma
end
"""
    diagonal_sigma_shape(::DiagonalOfShape, pdm::Option{<:AbstractPosdefEstimator},
                         sigma_mu::MatNum, sigma::MatNum, T::Number)
    diagonal_sigma_shape(::ShapeOfDiagonal, ::Option{<:AbstractPosdefEstimator},
                         sigma_mu::MatNum, ::MatNum, T::Number)

Returns the diagonal covariance shape of a [`NormalUncertaintySet`](@ref) under the rule that `dc` names.

# Algorithm

 1. On [`DiagonalOfShape`](@ref), run the steps that its docstring lists.
 2. On [`ShapeOfDiagonal`](@ref), run the steps that its docstring lists.

# Arguments

  - `dc`: Construction of the diagonal shape.
  - `pdm`: Positive definite matrix estimator of the full shape. [`ShapeOfDiagonal`](@ref) does not read it.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref).
  - `sigma`: Point estimate of the covariance matrix. Only [`DiagonalOfShape`](@ref) reads it, and it reads only the size.
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref).

# Returns

  - `sigma_sigma::LinearAlgebra.Diagonal`: Diagonal covariance shape, ``N^{2} \\times N^{2}``.

# Related

  - [`AbstractDiagonalConstruction`](@ref)
  - [`DiagonalOfShape`](@ref)
  - [`ShapeOfDiagonal`](@ref)
  - [`sigma_asymptotic_cov`](@ref)
  - [`normal_sigma_shape`](@ref)
"""
function diagonal_sigma_shape(::DiagonalOfShape, pdm::Option{<:AbstractPosdefEstimator},
                              sigma_mu::MatNum, sigma::MatNum, T::Number)
    return LinearAlgebra.Diagonal(sigma_asymptotic_cov(pdm, sigma_mu, sigma, T))
end
function diagonal_sigma_shape(::ShapeOfDiagonal, ::Option{<:AbstractPosdefEstimator},
                              sigma_mu::MatNum, ::MatNum, T::Number)
    d = LinearAlgebra.diag(sigma_mu)
    N = length(d)
    v = vec(T .* d .* transpose(d))
    v[1:(N + 1):end] .*= 2
    return LinearAlgebra.Diagonal(v)
end
"""
    normal_sigma_shape(ue::NormalUncertaintySet, sigma_mu::MatNum, sigma::MatNum, T::Number)

Returns the covariance shape of an ellipsoidal or norm-ball [`NormalUncertaintySet`](@ref).

Each ellipsoidal and norm-ball method of [`ucs`](@ref) and [`sigma_ucs`](@ref) builds its shape with this function. So the rule of `ue.dc` applies on every route that reads a diagonal shape. The box methods read no shape.

# Algorithm

 1. If `ue.alg.diagonal` is `true`, return [`diagonal_sigma_shape`](@ref) under `ue.dc`.
 2. Otherwise, return the full shape of [`sigma_asymptotic_cov`](@ref). The function does not read `ue.dc` in this case.

# Arguments

  - `ue`: Normal uncertainty set estimator, whose `alg` is an [`EllipsoidalUncertaintySetAlgorithm`](@ref) or a [`NormBallUncertaintySetAlgorithm`](@ref).
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref).
  - `sigma`: Point estimate of the covariance matrix.
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref).

# Returns

  - `sigma_sigma::MatNum`: Covariance shape, ``N^{2} \\times N^{2}``.

# Related

  - [`diagonal_sigma_shape`](@ref)
  - [`sigma_asymptotic_cov`](@ref)
  - [`NormalUncertaintySet`](@ref)
"""
function normal_sigma_shape(ue::NormalUncertaintySet, sigma_mu::MatNum, sigma::MatNum,
                            T::Number)
    return if ue.alg.diagonal
        diagonal_sigma_shape(ue.dc, ue.pdm, sigma_mu, sigma, T)
    else
        sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    end
end
"""
    mu_normal_box_set(mu::VecNum, sigma_mu::MatNum, q::Number)

Builds the box of the mean axis under normal returns, with a lower bound of zero and the full width in the upper bound.

`q` is the significance level after the halving. [`set_ucs_return_constraints!`](@ref) reads only the half-width ``(\\boldsymbol{\\mu}_{u} - \\boldsymbol{\\mu}_{l})/2`` of the pair, and it centres that width on `val`. So the half-width is ``z_{q} \\sqrt{\\operatorname{diag}(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})}``. Neither bound is a bound on the mean. The box methods of [`ucs`](@ref) and [`mu_ucs`](@ref) for [`NormalUncertaintySet`](@ref) call this function.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}_{l} &= \\boldsymbol{0}\\,, \\\\
\\mu_{u,i} &= 2 z_{q} \\sqrt{\\left(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}\\right)_{ii}}\\,, \\qquad z_{q} = \\Phi^{-1}(1 - q)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{l}``, ``\\mu_{u,i}``: Lower bound, and ``i``-th entry of the upper bound.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``z_{q}``: Normal complementary quantile at the halved significance level.
  - ``q``: Half significance level.

Each interval covers ``1 - 2q`` of the sampling law of its entry, because each of the two tails holds ``q``. Each interval applies to one entry, so the box is not a joint region for the full vector.

The half-width is the ``\\delta_{\\boldsymbol{\\mu},i}`` of Equation 11.17 of [cajas2025](@cite). The text of that equation calls ``\\sigma_i`` the standard deviation of the asset. But its Listing 11.7 and this method use the standard error of the mean, ``\\sqrt{(\\hat{\\mathbf{\\Sigma}} / T)_{ii}}``.

# Algorithm

 1. Get the ``1 - q`` standard normal quantile with `Distributions.cquantile`. This is the multiplier ``z_{q}``.
 2. Multiply it by the square roots of `LinearAlgebra.diag(sigma_mu)` and by two. The result `mu_u` is the width of the box on each asset.
 3. Build `mu_l`, a range of zeros with the same length and element type.
 4. Return a [`BoxUncertaintySet`](@ref) that holds `mu_l`, `mu_u` and `mu`. `mu` is the centre of the width.

# Arguments

  - `mu`: Point estimate of the expected returns vector. The set holds it as its centre.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref).
  - `q`: Half significance level, which [`normal_box_preamble`](@ref) halved.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`mu_asymptotic_cov`](@ref)
  - [`normal_box_preamble`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`set_ucs_return_constraints!`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.17 and Listing 11.7. $(ref_dict[:no_original_source])
"""
function mu_normal_box_set(mu::VecNum, sigma_mu::MatNum, q::Number)
    mu_u = Distributions.cquantile(Distributions.Normal(), q) *
           sqrt.(LinearAlgebra.diag(sigma_mu)) *
           2
    mu_l = range(zero(eltype(sigma_mu)), zero(eltype(sigma_mu)); length = size(sigma_mu, 1))
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u, val = mu)
end
"""
    sigma_normal_box_set(ue::NormalUncertaintySet, pr::LowOrderPrior, T::Number,
                         sigma_mu::MatNum, q::Number)

Builds the box of the covariance axis of a [`NormalUncertaintySet`](@ref).

The function draws `ue.n_sim` Wishart matrices with `T` degrees of freedom and scale `sigma_mu`. The element-wise `q` and `1 - q` quantiles of the draws are the bounds. The function then projects each bound to the nearest positive definite matrix. The box methods of [`ucs`](@ref) and [`sigma_ucs`](@ref) both call it, so the two give the same box under one seed.

The mean of a ``\\mathrm{Wishart}(T, \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})`` draw is ``T \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``, which is ``\\hat{\\mathbf{\\Sigma}}``. So the draws are centred on the point estimate, and the two bounds are on the two sides of it.

# Algorithm

 1. Read the point estimate `sigma = pr.sigma`, the centre of the bounds.
 2. Resolve the random number generator from `ue.rng` and `ue.seed` with [`resolve_rng`](@ref).
 3. Draw `ue.n_sim` matrices `sigmas` from `Distributions.Wishart(T, sigma_mu)`.
 4. Take the element-wise `q` and `1 - q` quantiles of `sigmas` with [`box_quantile_bounds`](@ref) to get `sigma_l` and `sigma_u`.
 5. Repair `sigma_l` and `sigma_u` in place with `posdef!`. The quantiles are entry by entry, so neither bound is a Wishart draw, and neither bound is always positive definite. `posdef!` does not change a bound that is already positive definite.
 6. Return a [`BoxUncertaintySet`](@ref) that holds `sigma_l`, `sigma_u` and `sigma`.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `pr`: Prior Result. The function reads the point estimate `pr.sigma` and the asset count `size(pr.X, 2)`.
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref), the degrees of freedom of the Wishart law.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref), the scale of the Wishart law.
  - `q`: Half significance level.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`normal_box_preamble`](@ref)
  - [`mu_normal_box_set`](@ref)
  - [`box_quantile_bounds`](@ref)
"""
function sigma_normal_box_set(ue::NormalUncertaintySet, pr::AbstractPriorResult, T::Number,
                              sigma_mu::MatNum, q::Number)
    sigma = pr.sigma
    rng = resolve_rng(ue.rng, ue.seed)
    sigmas = rand(rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    sigma_l, sigma_u = box_quantile_bounds(eltype(sigma), (i, j) -> getindex.(sigmas, i, j),
                                           size(pr.X, 2), q, ue.kwargs)
    posdef!(ue.pdm, sigma_l)
    posdef!(ue.pdm, sigma_u)
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u, val = sigma)
end
"""
    normal_box_preamble(ue::NormalUncertaintySet, pr::AbstractPriorResult)

Does the first steps that the three box methods of a [`NormalUncertaintySet`](@ref) have in common.

The function finds the scaling parameter from the Prior Result, and then it makes the two values that the two boxes use. The box methods of [`ucs`](@ref), [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) all call it first.

The returned `q` is half of `ue.q`, and the ellipsoidal route does not halve the level. A box bounds each entry on two sides, so each tail gets half of the significance level. An ellipsoid cuts only the upper tail of a non-negative distance. So one cut at the ``1 - q`` quantile covers ``1 - q``.

# Algorithm

 1. Resolve the scaling parameter with [`choose_scaling_parameter`](@ref) to get `T`.
 2. Build the mean asymptotic covariance with [`mu_asymptotic_cov`](@ref) to get `sigma_mu`.
 3. Halve `ue.q` to get `q`, the significance level of one tail.
 4. Return `T`, `sigma_mu` and `q` as a tuple.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `pr`: Prior Result that the set fits on.

# Returns

  - `(T, sigma_mu, q)`: Scaling parameter, mean asymptotic covariance, and half significance level.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`choose_scaling_parameter`](@ref)
  - [`mu_asymptotic_cov`](@ref)
  - [`mu_normal_box_set`](@ref)
  - [`sigma_normal_box_set`](@ref)
"""
function normal_box_preamble(ue::NormalUncertaintySet, pr::AbstractPriorResult)
    T = choose_scaling_parameter(ue, pr)
    return T, mu_asymptotic_cov(ue.pdm, pr.sigma, T), ue.q * 0.5
end
"""
    normal_mu_error_sample(ue::NormalUncertaintySet, rng::Random.AbstractRNG, mu::VecNum,
                           sigma_mu::MatNum)

Draws `ue.n_sim` estimation errors of the mean from the sampling law of normal returns.

Each row of the sample is an estimation error, a draw minus `mu`. The draws come from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}}/T)``, so the sample and its shape matrix have the same scale. A draw from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}})`` multiplies every deviation by ``\\sqrt{T}``, so it also multiplies the radius by ``\\sqrt{T}``. The caller resolves the generator and gives it to the function. So a caller that needs both axes draws them from one stream.

# Algorithm

 1. Draw `ue.n_sim` vectors from `Distributions.MvNormal(mu, sigma_mu)`. The result has one column for each draw.
 2. Subtract `mu` from each column. Transpose the result, so that each row holds one estimation error.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `rng`: Random number generator, which the caller resolves with [`resolve_rng`](@ref).
  - `mu`: Point estimate of the expected returns vector, the centre of the draws.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref).

# Returns

  - `X_mu::MatNum`: Sampled estimation errors, `ue.n_sim` by `N`.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`normal_sigma_error_sample`](@ref)
  - [`mu_asymptotic_cov`](@ref)
  - [`k_ucs`](@ref)
  - [`k_norm_ball`](@ref)
"""
function normal_mu_error_sample(ue::NormalUncertaintySet, rng::Random.AbstractRNG,
                                mu::VecNum, sigma_mu::MatNum)
    return transpose(rand(rng, Distributions.MvNormal(mu, sigma_mu), ue.n_sim) .- mu)
end
"""
    normal_sigma_error_sample(ue::NormalUncertaintySet, rng::Random.AbstractRNG,
                              sigma::MatNum, sigma_mu::MatNum, T::Number, N::Integer)

Draws `ue.n_sim` estimation errors of the vectorised covariance from the Wishart law of normal returns.

Each row of the sample is an estimation error, a draw minus `sigma`. The variance of an entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw is the matching diagonal entry of ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``, the asymptotic covariance of the covariance estimator. So the sample and its shape matrix have the same scale. The caller gives `N`, and the function does not read it from `sigma`. So every caller reads the asset count from one source, `size(pr.X, 2)`.

# Algorithm

 1. Draw `ue.n_sim` matrices from `Distributions.Wishart(T, sigma_mu)` to get `sigmas`.
 2. Subtract `sigma` from each draw into an `N` by `N` by `ue.n_sim` array.
 3. Reshape the array to `N^2` rows, with one column for each draw. Transpose it, so that each row holds one vectorised estimation error.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `rng`: Random number generator, which the caller resolves with [`resolve_rng`](@ref).
  - `sigma`: Point estimate of the covariance matrix, the centre of the draws.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref), the scale of the Wishart law.
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref), the degrees of freedom of the Wishart law.
  - `N`: Asset count. The caller reads it from `size(pr.X, 2)`.

# Returns

  - `X_sigma::MatNum`: Sampled estimation errors, `ue.n_sim` by `N^2`.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`normal_mu_error_sample`](@ref)
  - [`sigma_asymptotic_cov`](@ref)
  - [`k_ucs`](@ref)
  - [`k_norm_ball`](@ref)
"""
function normal_sigma_error_sample(ue::NormalUncertaintySet, rng::Random.AbstractRNG,
                                   sigma::MatNum, sigma_mu::MatNum, T::Number, N::Integer)
    sigmas = rand(rng, Distributions.Wishart(T, sigma_mu), ue.n_sim)
    X_sigma = Array{eltype(sigma)}(undef, N, N, ue.n_sim)
    for i in axes(sigmas, 1)
        X_sigma[:, :, i] = sigmas[i] - sigma
    end
    return transpose(reshape(X_sigma, N^2, :))
end
# A NormalUncertaintySet with no prior of its own reads the prior result it is handed
# (see [`reads_prior_result`](@ref)).
function reads_prior_result(::NormalUncertaintySet{Nothing})::Bool
    return true
end
"""
    ucs(ue::NormalUncertaintySet, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    mu_ucs(ue::NormalUncertaintySet, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    sigma_ucs(ue::NormalUncertaintySet, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Fits the sets of a [`NormalUncertaintySet`](@ref) from returns data, through the prior of the set.

These are the returns-data methods of the three verbs. Each verb has one such method for all shapes, because the next call selects the shape. Each method fits `ue.pe` once through [`ucs_prior`](@ref). It then gives the result to the prior-result method of the same verb, on the same set with `pe = nothing`. That method dispatches on the box, the ellipsoid or the norm ball.

So a set with a prior of its own fits on that prior alone, and the prior fits on the returns that the caller gives. The two routes use the same body for each shape. Let `ue′` be `ue` with `pe = nothing`. Then `ucs(ue, X)` and `ucs(ue′, prior(ue.pe, X))` give identical sets.

# Algorithm

 1. Fit the prior with [`ucs_prior`](@ref) on `ue.pe`, `X` and `F` to get `pr`.
 2. Call the prior-result method of the verb on `pr` and on the set with `pe = nothing`. That method builds the set of the shape that `ue.alg` names.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Matrix of asset returns.
  - `F`: Optional matrix of factor returns, which a factor prior needs.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method gives to the prior estimator.

# Validation

  - `ue.pe` is not `nothing`. If it is, [`ucs_prior`](@ref) raises an `ArgumentError` that names the prior-result calls.

# Returns

  - `mu_ucs::AbstractUncertaintySetResult`: Expected returns uncertainty set, from `mu_ucs`.
  - `sigma_ucs::AbstractUncertaintySetResult`: Covariance uncertainty set, from `sigma_ucs`.
  - `(mu_ucs, sigma_ucs)`: Both, from `ucs`.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`ucs_prior`](@ref)
  - [`reads_prior_result`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
"""
function ucs(ue::NormalUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
             dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return ucs(rebuild_estimator(ue, (; pe = nothing)), pr)
end
function mu_ucs(ue::NormalUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return mu_ucs(rebuild_estimator(ue, (; pe = nothing)), pr)
end
function sigma_ucs(ue::NormalUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                   dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return sigma_ucs(rebuild_estimator(ue, (; pe = nothing)), pr)
end
"""
    ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
        pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the box sets of the mean and the covariance under normal returns.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

# Mathematical definition

The method draws ``M`` covariance matrices ``\\hat{\\mathbf{\\Sigma}}^{(m)} \\sim \\mathrm{Wishart}(T,\\, \\hat{\\mathbf{\\Sigma}}/T)``, and the element-wise quantiles of the draws are the bounds:

```math
\\begin{align}
(\\mathbf{\\Sigma}_{l})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{u})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

The bounds of the mean use the normal complementary quantile ``z_{q/2} = \\Phi^{-1}(1-q/2)``:

```math
\\begin{align}
\\boldsymbol{\\mu}_{l} &= \\boldsymbol{0}\\,, \\\\
\\mu_{u,i} &= 2 z_{q/2} \\sqrt{(\\hat{\\mathbf{\\Sigma}}/T)_{ii}}\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{l})_{ij}``, ``(\\mathbf{\\Sigma}_{u})_{ij}``: Element-wise lower and upper bounds of the covariance.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at the levels ``q/2`` and ``1 - q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: Entry ``(i,j)`` of the ``m``-th Wishart draw.
  - ``M``: Number of draws, `ue.n_sim`.
  - ``\\boldsymbol{\\mu}_{l}``, ``\\mu_{u,i}``: Lower bound, and ``i``-th entry of the upper bound, that the builder writes for the mean axis.
  - ``z_{q/2} = \\Phi^{-1}(1 - q/2)``: Normal complementary quantile.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - ``q``: Significance level.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Run [`normal_box_preamble`](@ref) on `prr` to get `T`, `sigma_mu` and the halved `q`.
 3. Build the mean set with [`mu_normal_box_set`](@ref) from `prr.mu`, `sigma_mu` and `q`.
 4. Build the covariance set with [`sigma_normal_box_set`](@ref) from `ue`, `prr`, `T`, `sigma_mu` and `q`.
 5. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref). A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.
 6. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                      <:Any}, pr::AbstractPriorResult; rd = nothing,
             kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    T, sigma_mu, q = normal_box_preamble(ue, prr)
    mu_set, sigma_set = mu_normal_box_set(prr.mu, sigma_mu, q),
                        sigma_normal_box_set(ue, prr, T, sigma_mu, q)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    mu_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
           pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the box set of the mean under normal returns.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}_{l} &= \\boldsymbol{0}\\,, \\\\
\\mu_{u,i} &= 2 \\Phi^{-1}\\!\\left(1 - \\tfrac{q}{2}\\right) \\sqrt{(\\hat{\\mathbf{\\Sigma}}/T)_{ii}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{l}``, ``\\mu_{u,i}``: Lower bound, and ``i``-th entry of the upper bound, that the builder writes for the mean axis.
  - ``\\Phi^{-1}(1 - q/2)``: Normal complementary quantile.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - ``q``: Significance level.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Run [`normal_box_preamble`](@ref) on `prr` to get `sigma_mu` and the halved `q`. The method discards `T`, because only the covariance set reads it.
 3. Build the mean set with [`mu_normal_box_set`](@ref) from `prr.mu`, `sigma_mu` and `q`.
 4. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any,
                                         <:Any, <:Any}, pr::AbstractPriorResult;
                rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    _, sigma_mu, q = normal_box_preamble(ue, prr)
    set = mu_normal_box_set(prr.mu, sigma_mu, q)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
              pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the box set of the covariance under normal returns.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

# Mathematical definition

The method draws ``M`` matrices ``\\hat{\\mathbf{\\Sigma}}^{(m)} \\sim \\mathrm{Wishart}(T,\\,\\hat{\\mathbf{\\Sigma}}/T)``, and the element-wise quantiles of the draws are the bounds:

```math
\\begin{align}
(\\mathbf{\\Sigma}_{l})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{u})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{l})_{ij}``, ``(\\mathbf{\\Sigma}_{u})_{ij}``: Element-wise lower and upper bounds of the covariance.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at the levels ``q/2`` and ``1 - q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: Entry ``(i,j)`` of the ``m``-th Wishart draw.
  - ``M``: Number of draws, `ue.n_sim`.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - ``q``: Significance level.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Run [`normal_box_preamble`](@ref) on `prr` to get `T`, `sigma_mu` and the halved `q`.
 3. Build the covariance set with [`sigma_normal_box_set`](@ref) from `ue`, `prr`, `T`, `sigma_mu` and `q`.
 4. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any,
                                            <:Any, <:Any}, pr::AbstractPriorResult;
                   rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    T, sigma_mu, q = normal_box_preamble(ue, prr)
    set = sigma_normal_box_set(ue, prr, T, sigma_mu, q)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    ucs(ue::NormalUncertaintySet{Nothing,
                                 <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal sets of the mean and the covariance under normal returns, with radii from simulated estimation errors.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

Each row of the two samples is an estimation error. [`k_ucs`](@ref) measures a Mahalanobis distance against the shape matrix. So step 6 draws from the sampling law of the estimator, ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}}/T)``, and subtracts ``\\hat{\\boldsymbol{\\mu}}`` from each draw. A draw from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}})`` multiplies every deviation by ``\\sqrt{T}``, so it also multiplies the radius by ``\\sqrt{T}``. The sample of step 7 has the scale of its shape for the same reason. The variance of an entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw is the matching diagonal entry of ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``.

The two draws use one generator, so the covariance radius of this method differs from that of [`sigma_ucs`](@ref) under one seed. Step 5 resolves the generator once, and step 6 advances it before step 7 draws. [`sigma_ucs`](@ref) draws its Wishart matrices from the start of the stream. The two mean radii are equal, because the mean sample is the first draw in both methods. The box route does not have this difference. [`sigma_normal_box_set`](@ref) resolves its own generator, so the two box methods give the same covariance box.

# Mathematical definition

The two ellipsoids have the prior estimates as centres, and the two asymptotic covariances as shapes:

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} &= \\hat{\\mathbf{\\Sigma}} / T\\,, \\\\
\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} &= T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})\\,.
\\end{align}
```

[`k_ucs`](@ref) gives the radius of each ellipsoid for [`NormalKUncertaintyAlgorithm`](@ref), and the docstring of that algorithm states the form. The method gives the significance level to [`k_ucs`](@ref) with no division, because an ellipsoid cuts only the upper tail of a non-negative distance.

```math
\\begin{align}
\\mathcal{E}_{\\boldsymbol{\\mu}} &= \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\boldsymbol{\\mu}}^2 \\right\\}\\,.
\\end{align}
```

```math
\\begin{align}
\\mathcal{E}_{\\mathbf{\\Sigma}} &= \\left\\{\\mathbf{\\Sigma} : \\left\\lVert  \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}})  \\right\\rVert_2 \\leq k_{\\mathbf{\\Sigma}} \\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the covariance estimator, vectorised.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:I_identity])
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``\\mathcal{E}_{\\boldsymbol{\\mu}}``: Ellipsoidal uncertainty set for expected returns.
  - ``\\mathcal{E}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for covariance.
  - ``k_{\\boldsymbol{\\mu}}``, ``k_{\\mathbf{\\Sigma}}``: Radii of the two ellipsoids, fitted on the sampled estimation errors.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `mu`, `sigma` and `N = size(prr.X, 2)` from `prr`. The asset count comes from the returns matrix of the prior, so the method follows a prior that changes the asset count.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 4. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 5. Resolve the random number generator from `ue.rng` and `ue.seed` with [`resolve_rng`](@ref).
 6. Draw the mean sample `X_mu` with [`normal_mu_error_sample`](@ref). Each row holds one estimation error of the mean.
 7. Draw the covariance sample `X_sigma` with [`normal_sigma_error_sample`](@ref). Each row holds one vectorised estimation error of the covariance.
 8. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 9. Fit the mean ellipsoid with [`ellipsoidal_set`](@ref) on `X_mu` and `sigma_mu`, with `prr.mu` as the centre.
10. Fit the covariance ellipsoid the same way on `X_sigma` and `sigma_sigma`, with `prr.sigma` as the centre.
11. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref). A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.
12. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{Nothing,
                                      <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                           <:Any}, <:Any,
                                      <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing,
             kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    (; mu, sigma) = prr
    N = size(prr.X, 2)
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    mu_set, sigma_set = ellipsoidal_set(ue.alg, ue.q, X_mu, sigma_mu,
                                        MuUncertaintySetClass(), prr.mu),
                        ellipsoidal_set(ue.alg, ue.q, X_sigma, sigma_sigma,
                                        SigmaUncertaintySetClass(), prr.sigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                 <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal sets of the mean and the covariance under normal returns, with radii that read no sample.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

This method draws no sample. It is the method for each radius algorithm that reads no sample, which is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref) or a plain number. The method for [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs. The two routes build the same shapes, so they differ only in the radius.

# Mathematical definition

The two ellipsoids have the prior estimates as centres, and the two asymptotic covariances as shapes:

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} &= \\hat{\\mathbf{\\Sigma}} / T\\,, \\\\
\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} &= T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the covariance estimator, vectorised.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:I_identity])
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.

[`k_ucs`](@ref) gives the radius of each ellipsoid for `ue.alg.method`, and the docstring of that algorithm states the closed form. The method gives the significance level to [`k_ucs`](@ref) with no division, because an ellipsoid cuts only the upper tail of a non-negative distance.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 4. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 5. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 6. Fit the mean ellipsoid with [`ellipsoidal_set`](@ref) on `sigma_mu`, with `nothing` in place of a sample and `prr.mu` as the centre.
 7. Fit the covariance ellipsoid the same way on `sigma_sigma`, with `prr.sigma` as the centre.
 8. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref). A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.
 9. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function ucs(ue::NormalUncertaintySet{Nothing,
                                      <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                      <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
             rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    mu_set, sigma_set = ellipsoidal_set(ue.alg, ue.q, nothing, sigma_mu,
                                        MuUncertaintySetClass(), prr.mu),
                        ellipsoidal_set(ue.alg, ue.q, nothing, sigma_sigma,
                                        SigmaUncertaintySetClass(), prr.sigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    mu_ucs(ue::NormalUncertaintySet{Nothing,
                                    <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal set of the mean under normal returns, with a radius from simulated estimation errors.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

Each row of the sample is an estimation error. [`k_ucs`](@ref) measures a Mahalanobis distance against the shape matrix. So step 5 draws from the sampling law of the estimator, ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}}/T)``, and subtracts ``\\hat{\\boldsymbol{\\mu}}`` from each draw. A draw from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}})`` multiplies every deviation by ``\\sqrt{T}``, so it also multiplies the radius by ``\\sqrt{T}``.

# Mathematical definition

```math
\\mathcal{E}_{\\boldsymbol{\\mu}} = \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\boldsymbol{\\mu}}^2 \\right\\}\\,, \\qquad \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} = \\hat{\\mathbf{\\Sigma}} / T\\,.
```

Where:

  - ``\\mathcal{E}_{\\boldsymbol{\\mu}}``: Ellipsoidal uncertainty set for expected returns.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated expected returns vector.
  - ``k_{\\boldsymbol{\\mu}}``: Radius fitted on the sample, the ``1 - q`` quantile of the sampled Mahalanobis distances.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `mu` and `sigma` from `prr`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 4. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 5. Resolve the random number generator with [`resolve_rng`](@ref). Draw the sample `X_mu` with [`normal_mu_error_sample`](@ref). Each row holds one estimation error.
 6. Fit the ellipsoid with [`ellipsoidal_set`](@ref) on `X_mu` and `sigma_mu`, with `prr.mu` as the centre.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{Nothing,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                              <:Any}, <:Any,
                                         <:Any, <:Any}, pr::AbstractPriorResult;
                rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    (; mu, sigma) = prr
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    set = ellipsoidal_set(ue.alg, ue.q, X_mu, sigma_mu, MuUncertaintySetClass(), prr.mu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    mu_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                    <:Any, <:Any, <:Any},
           pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal set of the mean under normal returns, with a radius that reads no sample.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

The shape matrix is ``\\hat{\\mathbf{\\Sigma}} / T``, the asymptotic covariance of the mean that [`mu_asymptotic_cov`](@ref) builds. This method draws no sample. It is the method for each radius algorithm that reads no sample, which is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref) or a plain number. The method for [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs, and it builds the same shape.

# Mathematical definition

```math
\\mathcal{E}_{\\boldsymbol{\\mu}} = \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\boldsymbol{\\mu}}^2 \\right\\}\\,, \\qquad \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} = \\hat{\\mathbf{\\Sigma}} / T\\,.
```

Where:

  - ``\\mathcal{E}_{\\boldsymbol{\\mu}}``: Ellipsoidal uncertainty set for expected returns.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated expected returns vector.
  - ``k_{\\boldsymbol{\\mu}}``: Radius, the one [`k_ucs`](@ref) returns for `ue.alg.method`.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

The method gives the significance level to [`k_ucs`](@ref) with no division, because an ellipsoid cuts only the upper tail of a non-negative distance.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 4. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 5. Fit the ellipsoid with [`ellipsoidal_set`](@ref) on `sigma_mu`, with `nothing` in place of a sample and `prr.mu` as the centre.
 6. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(ue::NormalUncertaintySet{Nothing,
                                         <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                         <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    set = ellipsoidal_set(ue.alg, ue.q, nothing, sigma_mu, MuUncertaintySetClass(), prr.mu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                       <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                       <:Any, <:Any, <:Any},
              pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal set of the covariance under normal returns, with a radius from simulated estimation errors.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

Each row of the sample is an estimation error. Step 5 subtracts ``\\hat{\\mathbf{\\Sigma}}`` from each Wishart draw. The variance of an entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw is the matching diagonal entry of ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``. So the sample and its shape matrix have the same scale. The method reads `N` from `size(prr.X, 2)`, and [`ucs`](@ref) reads it from the same source. So a prior that changes the asset count changes `N` in both methods, and the two shape matrices are equal.

Under one seed, this radius differs from the covariance radius of [`ucs`](@ref). This method draws its Wishart matrices from the start of the stream. [`ucs`](@ref) resolves one generator and draws its mean sample first, so its Wishart draws differ from these. Both radii are valid fits of the same quantity. The two methods give the same covariance set only on the box route.

# Mathematical definition

```math
\\mathcal{E}_{\\mathbf{\\Sigma}} = \\left\\{\\mathbf{\\Sigma} : \\left\\lVert  \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}})  \\right\\rVert_2 \\leq k_{\\mathbf{\\Sigma}} \\right\\}\\,, \\qquad \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} = T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})\\,.
```

Where:

  - ``\\mathcal{E}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for the covariance matrix.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the covariance estimator, vectorised.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - $(math_dict[:I_identity])
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``k_{\\mathbf{\\Sigma}}``: Radius fitted on the sample, the ``1 - q`` quantile of the sampled Mahalanobis distances.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma` and `N = size(prr.X, 2)`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 4. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 5. Resolve the random number generator with [`resolve_rng`](@ref). Draw the sample `X_sigma` with [`normal_sigma_error_sample`](@ref). Each row holds one vectorised estimation error.
 6. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 7. Fit the ellipsoid with [`ellipsoidal_set`](@ref) on `X_sigma` and `sigma_sigma`, with `prr.sigma` as the centre.
 8. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                                 <:Any},
                                            <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                   rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    N = size(prr.X, 2)
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    set = ellipsoidal_set(ue.alg, ue.q, X_sigma, sigma_sigma, SigmaUncertaintySetClass(),
                          prr.sigma)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                       <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                       <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal set of the covariance under normal returns, with a radius that reads no sample.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

This method draws no sample. It is the method for each radius algorithm that reads no sample, which is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref) or a plain number. The method for [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs. The two routes build the same shapes, so they differ only in the radius.

# Mathematical definition

```math
\\mathcal{E}_{\\mathbf{\\Sigma}} = \\left\\{\\mathbf{\\Sigma} : \\left\\lVert  \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}})  \\right\\rVert_2 \\leq k_{\\mathbf{\\Sigma}} \\right\\}\\,, \\qquad \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} = T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})\\,.
```

Where:

  - ``\\mathcal{E}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for the covariance matrix.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the covariance estimator, vectorised.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator, ``\\hat{\\mathbf{\\Sigma}} / T``.
  - $(math_dict[:I_identity])
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``k_{\\mathbf{\\Sigma}}``: Radius, the one [`k_ucs`](@ref) returns for `ue.alg.method`.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

The method gives the significance level to [`k_ucs`](@ref) with no division, because an ellipsoid cuts only the upper tail of a non-negative distance.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 4. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 5. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 6. Fit the ellipsoid with [`ellipsoidal_set`](@ref) on `sigma_sigma`, with `nothing` in place of a sample and `prr.sigma` as the centre.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`k_ucs`](@ref)
  - [`mu_ucs`](@ref)
"""
function sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                            <:EllipsoidalUncertaintySetAlgorithm{<:Any,
                                                                                 <:Any},
                                            <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                   rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    set = ellipsoidal_set(ue.alg, ue.q, nothing, sigma_sigma, SigmaUncertaintySetClass(),
                          prr.sigma)
    return expand_investable_ucs(set, imsk, pr)
end

"""
    ucs(ue::NormalUncertaintySet{Nothing,
                                 <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                 <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball sets of the mean and the covariance under normal returns, with radii from simulated estimation errors.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

The two sets are the two ellipsoids of the ellipsoidal route, with each shape matrix in factored form. So they are the same regions, and they give the same weights. A [`NormBallUncertaintySet`](@ref) holds the factor, so the consumer factorises nothing when it builds the model. This method draws the two samples from one generator, so the Wishart draws come after the normal draws in the stream.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `mu`, `sigma` and `N = size(prr.X, 2)` from `prr`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref). Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Resolve one generator with [`resolve_rng`](@ref). Draw the two samples with [`normal_mu_error_sample`](@ref) and [`normal_sigma_error_sample`](@ref), mean first.
 5. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 6. Assemble the two sets with [`norm_ball_set`](@ref), on the mean shape and on the covariance shape.
 7. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref). A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.
 8. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::NormBallUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_set`](@ref)
  - [`k_norm_ball`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:magnus1979]) Section 4, Corollary 4.2.
  - $(ref_dict[:cajas2025]) Section 11.3.2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function ucs(ue::NormalUncertaintySet{Nothing,
                                      <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                        <:Any, <:Any},
                                      <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
             rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    (; mu, sigma) = prr
    N = size(prr.X, 2)
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    mu_set, sigma_set = norm_ball_set(ue.alg, ue.q, X_mu, sigma_mu, MuUncertaintySetClass(),
                                      prr.mu),
                        norm_ball_set(ue.alg, ue.q, X_sigma, sigma_sigma,
                                      SigmaUncertaintySetClass(), prr.sigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                 <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball sets of the mean and the covariance under normal returns, with radii that read no sample.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

This method draws no sample. It is the method for each radius algorithm that reads no sample, which is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref) or a plain number. The method for [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs. The two routes build the same maps, so they differ only in the radius.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref). Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 5. Assemble the two sets with [`norm_ball_set`](@ref), with `nothing` in place of a sample.
 6. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref). A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.
 7. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::NormBallUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_set`](@ref)
  - [`k_norm_ball`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:magnus1979]) Section 4, Corollary 4.2.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equation 50.
  - $(ref_dict[:elghaoui2003]) Theorem 1, Equation 8.
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function ucs(ue::NormalUncertaintySet{Nothing,
                                      <:NormBallUncertaintySetAlgorithm{<:Any, <:Any,
                                                                        <:Any}, <:Any,
                                      <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing,
             kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    mu_set, sigma_set = norm_ball_set(ue.alg, ue.q, nothing, sigma_mu,
                                      MuUncertaintySetClass(), prr.mu),
                        norm_ball_set(ue.alg, ue.q, nothing, sigma_sigma,
                                      SigmaUncertaintySetClass(), prr.sigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    mu_ucs(ue::NormalUncertaintySet{Nothing,
                                    <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                    <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball set of the mean under normal returns, with a radius from simulated estimation errors.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

The map is a factor of ``\\hat{\\mathbf{\\Sigma}} / T``. So the set is the mean ellipsoid of the ellipsoidal route, with its shape in factored form. This method draws its normal sample from the start of the stream. [`ucs`](@ref) also draws its mean sample first, so under one seed the two radii are equal.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `mu` and `sigma` from `prr`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref). Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Resolve the generator with [`resolve_rng`](@ref). Draw the sample with [`normal_mu_error_sample`](@ref).
 5. Assemble the set with [`norm_ball_set`](@ref), with `prr.mu` as the centre.
 6. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_set`](@ref)
  - [`normal_mu_error_sample`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function mu_ucs(ue::NormalUncertaintySet{Nothing,
                                         <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                           <:Any, <:Any},
                                         <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    (; mu, sigma) = prr
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    set = norm_ball_set(ue.alg, ue.q, X_mu, sigma_mu, MuUncertaintySetClass(), prr.mu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    mu_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                    <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball set of the mean under normal returns, with a radius that reads no sample.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

This method draws no sample. It is the method for each radius algorithm that reads no sample, which is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref) or a plain number. The method for [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs, and builds the same map.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref). Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Assemble the set with [`norm_ball_set`](@ref), with `nothing` in place of a sample and `prr.mu` as the centre.
 5. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_set`](@ref)
  - [`k_norm_ball`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equation 50.
  - $(ref_dict[:elghaoui2003]) Theorem 1, Equation 8.
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function mu_ucs(ue::NormalUncertaintySet{Nothing,
                                         <:NormBallUncertaintySetAlgorithm{<:Any, <:Any,
                                                                           <:Any}, <:Any,
                                         <:Any, <:Any}, pr::AbstractPriorResult;
                rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    set = norm_ball_set(ue.alg, ue.q, nothing, sigma_mu, MuUncertaintySetClass(), prr.mu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                       <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                       <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball set of the covariance under normal returns, with a radius from simulated estimation errors.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

The map is a factor of the asymptotic covariance of the vectorised covariance. So the set is the covariance ellipsoid of the ellipsoidal route, with its shape in factored form. Under one seed, this radius differs from the covariance radius of [`ucs`](@ref). This method draws its Wishart matrices from the start of the stream, and [`ucs`](@ref) draws its mean sample first. Both radii are valid fits of the same quantity.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma` and `N = size(prr.X, 2)`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref). Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Resolve the generator with [`resolve_rng`](@ref). Draw the sample with [`normal_sigma_error_sample`](@ref).
 5. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 6. Assemble the set with [`norm_ball_set`](@ref), with `prr.sigma` as the centre.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `sigma_ucs::NormBallUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_set`](@ref)
  - [`normal_sigma_error_sample`](@ref)
  - [`mu_ucs`](@ref)

# References

  - $(ref_dict[:magnus1979]) Section 4, Corollary 4.2.
  - $(ref_dict[:cajas2025]) Section 11.3.2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                            <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                              <:Any, <:Any},
                                            <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                   rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    N = size(prr.X, 2)
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    set = norm_ball_set(ue.alg, ue.q, X_sigma, sigma_sigma, SigmaUncertaintySetClass(),
                        prr.sigma)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                       <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball set of the covariance under normal returns, with a radius that reads no sample.

This is the prior-result method of the verb, and only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the set and the objective have the same centre, and the set folds with that prior under the online step. Outside an optimiser, the caller gives `prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref), and then it calls this method. So the two routes use the same body.

This method draws no sample. It is the method for each radius algorithm that reads no sample, which is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref) or a plain number. The method for [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs, and builds the same map.

# Algorithm

 1. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Read `sigma = prr.sigma`.
 3. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref). Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Build the covariance shape `sigma_sigma` with [`normal_sigma_shape`](@ref).
 5. Assemble the set with [`norm_ball_set`](@ref), with `nothing` in place of a sample and `prr.sigma` as the centre.
 6. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: Normal uncertainty set estimator with no prior of its own.
  - `pr`: Prior Result that the set fits on.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments. The method ignores them.

# Returns

  - `sigma_ucs::NormBallUncertaintySet`: Covariance uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_set`](@ref)
  - [`k_norm_ball`](@ref)
  - [`mu_ucs`](@ref)

# References

  - $(ref_dict[:magnus1979]) Section 4, Corollary 4.2.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equation 50.
  - $(ref_dict[:elghaoui2003]) Theorem 1, Equation 8.
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function sigma_ucs(ue::NormalUncertaintySet{Nothing,
                                            <:NormBallUncertaintySetAlgorithm{<:Any, <:Any,
                                                                              <:Any}, <:Any,
                                            <:Any, <:Any}, pr::AbstractPriorResult;
                   rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    sigma = prr.sigma
    T = choose_scaling_parameter(ue, prr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = normal_sigma_shape(ue, sigma_mu, sigma, T)
    set = norm_ball_set(ue.alg, ue.q, nothing, sigma_sigma, SigmaUncertaintySetClass(),
                        prr.sigma)
    return expand_investable_ucs(set, imsk, pr)
end

export NormalUncertaintySet, DiagonalOfShape, ShapeOfDiagonal
public AbstractDiagonalConstruction, diagonal_sigma_shape
