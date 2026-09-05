"""
$(DocStringExtensions.TYPEDEF)

Fits a box or an ellipsoidal uncertainty set from the sampling laws that normal returns imply: the mean is normal and the covariance is Wishart.

The two laws are Equation 11.16 of the source. `alg` picks the shape, and `n_sim` sets the number of Wishart draws the covariance bounds are read from. Its sampling-free counterpart is [`DeltaUncertaintySet`](@ref), and its distribution-free counterpart is [`ARCHUncertaintySet`](@ref).

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
         │           │      │   alg ┼ FullMoment()
         │           │      │     w ┴ nothing
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

# References

  - $(ref_dict[:cajas2025]) Equations 11.16, 11.17 and 11.24.
  - $(ref_dict[:fabozzi2007])
  - $(ref_dict[:meucci2005])
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
        assert_resource_cap(n_sim, RESOURCE_LIMITS[].max_n_sim, :n_sim, :max_n_sim)
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

The commutation matrix is the permutation that carries the column-major vectorisation of a matrix into the vectorisation of its transpose. It is its own inverse, and it holds one entry per row, so it is built as a sparse matrix. Only the shape of `X` is read, never its entries.

# Mathematical definition

```math
\\mathbf{K}_{m,n} \\operatorname{vec}(\\mathbf{A}) = \\operatorname{vec}\\!\\left(\\mathbf{A}^{\\intercal}\\right)\\,, \\qquad \\mathbf{A} \\in \\mathbb{R}^{m \\times n}\\,.
```

Where:

  - ``\\mathbf{K}_{m,n}``: Commutation matrix of an ``m \\times n`` matrix, of size ``mn \\times mn``.
  - ``\\mathbf{A}``: Any ``m \\times n`` matrix.
  - ``\\operatorname{vec}``: Column-major vectorisation.

A symmetric ``\\mathbf{A}`` satisfies ``\\mathbf{K}_{n,n} \\operatorname{vec}(\\mathbf{A}) = \\operatorname{vec}(\\mathbf{A})``, so it cannot separate ``\\mathbf{K}`` from the identity.

# Algorithm

 1. Read `m, n = size(X)` and form `mn = m * n`, the side of the result.
 2. Number the entries of an `m` by `n` matrix in column-major order, giving `row = 1:mn`.
 3. Send each number through the transpose with `col = vec(transpose(reshape(row, m, n)))`, the position that entry takes in the vectorised transpose.
 4. Write a one at every `(row, col)` pair, giving the sparse `mn` by `mn` matrix `cX`.

# Arguments

  - `X`: Input matrix. Used to determine the dimensions of the commutation matrix.

# Returns

  - `cX::SparseArrays.SparseMatrixCSC{<:Number, <:Integer}`: Returns a sparse commutation matrix of size `(m*n, m*n)` where `m, n = size(X)`.

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

  - $(ref_dict[:cajas2025]) Equation 11.17.
  - $(ref_dict[:meucci2005])
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

Return ``T``, the effective number of observations that divides the covariance to give the asymptotic covariance of the mean.

Three sources are read in order, and the first that is not `nothing` wins: the estimator's own `ens`, the prior's `ens`, and the row count of the prior's returns matrix. The first two exist because a weighted or a shrunk prior carries fewer effective observations than it has rows. Every set the file builds is scaled by ``T``, and the width of a set scales as ``T^{-1/2}``, so a prior that reports fewer effective scenarios than it has rows widens the set. Quartering ``T`` doubles the width of the mean box.

# Algorithm

 1. Return `ue.ens` when it is not `nothing`, the effective count the estimator states.
 2. Otherwise return `pr.ens` when it is not `nothing`, the effective count the prior carries.
 3. Otherwise return `size(pr.X, 1)`, the row count of the prior's returns matrix.

# Arguments

  - `ue`: [`NormalUncertaintySet`](@ref) estimator.
  - `pr`: Prior result.

# Returns

  - `T::Number`: Effective number of observations.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`mu_asymptotic_cov`](@ref)
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

Asymptotic covariance of the mean estimator, projected to the nearest positive definite matrix with `pdm`.

Shared by the box and the ellipsoidal [`NormalUncertaintySet`](@ref) constructions, so the two routes cannot drift apart. The division is by a positive scalar, so it preserves definiteness: `posdef!` moves nothing when `sigma` is already positive definite.

# Mathematical definition

```math
\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} = \\dfrac{\\hat{\\mathbf{\\Sigma}}}{T}\\,.
```

Where:

  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

# Algorithm

 1. Divide `sigma` by `T`, giving `sigma_mu`.
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

  - $(ref_dict[:cajas2025]) Equation 11.16.
  - $(ref_dict[:meucci2005])
"""
function mu_asymptotic_cov(pdm::Option{<:AbstractPosdefEstimator}, sigma::MatNum, T::Number)
    sigma_mu = sigma / T
    posdef!(pdm, sigma_mu)
    return sigma_mu
end
"""
    sigma_asymptotic_cov(pdm, sigma_mu, sigma, T)

Asymptotic covariance of the vectorised covariance estimator, projected to the nearest positive definite matrix with `pdm`.

`sigma_mu` is the raw, pre-diagonalisation mean asymptotic covariance from [`mu_asymptotic_cov`](@ref), and ``\\mathbf{K}`` is the [`commutation_matrix`](@ref) of `sigma`. The two matrix arguments are not independent: the form below reduces to the sampling law of a normal sample only when `sigma_mu` was built from the same `sigma` and the same `T`.

# Mathematical definition

```math
\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} = T \\left(\\mathbf{I} + \\mathbf{K}\\right) \\left(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}\\right) = \\dfrac{\\left(\\mathbf{I} + \\mathbf{K}\\right) \\left(\\hat{\\mathbf{\\Sigma}} \\otimes \\hat{\\mathbf{\\Sigma}}\\right)}{T}\\,.
```

Where:

  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the vectorised covariance estimator.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``\\mathbf{I}``: Identity matrix.
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

The right-hand form is the asymptotic covariance of ``\\operatorname{vec}(\\hat{\\mathbf{\\Sigma}})`` for a normal sample. Its ``(i,j)`` diagonal entry is ``\\left(\\hat{\\Sigma}_{ii} \\hat{\\Sigma}_{jj} + \\hat{\\Sigma}_{ij}^{2}\\right) / T``, which is the variance of the ``(i,j)`` entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw. The sample that [`ucs`](@ref) builds from those draws and the shape matrix it measures that sample against are therefore on one scale.

# Algorithm

 1. Build `K = commutation_matrix(sigma)`. `sigma` is square, so `K` is ``N^{2} \\times N^{2}``.
 2. Form `sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)`.
 3. Repair `sigma_sigma` in place with `posdef!` under `pdm`.
 4. Return `sigma_sigma`.

# Arguments

  - `pdm`: Positive definite matrix estimator. `nothing` skips the repair.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref).
  - `sigma`: Point estimate of the covariance matrix. Only its shape is read, through [`commutation_matrix`](@ref).
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref).

# Returns

  - `sigma_sigma::MatNum`: Asymptotic covariance of the vectorised covariance estimator, ``N^{2} \\times N^{2}``.

# Related

  - [`mu_asymptotic_cov`](@ref)
  - [`commutation_matrix`](@ref)
  - [`NormalUncertaintySet`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equations 11.17 and 11.24.
  - $(ref_dict[:meucci2005])
"""
function sigma_asymptotic_cov(pdm::Option{<:AbstractPosdefEstimator}, sigma_mu::MatNum,
                              sigma::MatNum, T::Number)
    K = commutation_matrix(sigma)
    sigma_sigma = T * (LinearAlgebra.I + K) * kron(sigma_mu, sigma_mu)
    posdef!(pdm, sigma_sigma)
    return sigma_sigma
end
"""
    mu_normal_box_set(mu::VecNum, sigma_mu::MatNum, q::Number)

Box uncertainty set for expected returns under normality: a zero lower bound, and an upper bound that carries the whole width of the box.

`q` is the already-halved significance level. [`set_ucs_return_constraints!`](@ref) reads the pair only through its half-width ``(\\boldsymbol{\\mu}_{u} - \\boldsymbol{\\mu}_{l})/2``, which is therefore ``z_{q} \\sqrt{\\operatorname{diag}(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})}``, and it centres that width on the vector the set carries. Neither bound is a bound on the mean on its own. Shared by the box [`ucs`](@ref) and [`mu_ucs`](@ref) constructions for [`NormalUncertaintySet`](@ref).

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

Each entry of the resulting interval covers ``1 - 2q`` of the sampling law of that entry, because the two tails carry ``q`` each. The interval is one entry at a time, so it is not a joint region for the whole vector.

# Algorithm

 1. Take the ``1 - q`` standard normal quantile with `Distributions.cquantile`, giving the multiplier ``z_{q}``.
 2. Multiply it by the square roots of `LinearAlgebra.diag(sigma_mu)` and by two, giving `mu_u`, the width of the box on each asset.
 3. Build `mu_l`, a vector of zeros of the same length and element type.
 4. Return a [`BoxUncertaintySet`](@ref) carrying `mu_l`, `mu_u`, and `mu`, the vector the width is centred on.

# Arguments

  - `mu`: Point estimate of the expected returns vector, carried by the set as its centre.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref).
  - `q`: Half significance level, already halved by [`normal_box_preamble`](@ref).

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`mu_asymptotic_cov`](@ref)
  - [`normal_box_preamble`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`set_ucs_return_constraints!`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equations 11.14 and 11.19.
  - $(ref_dict[:fabozzi2007])
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

Covariance box uncertainty set of a [`NormalUncertaintySet`](@ref).

Draws `ue.n_sim` Wishart matrices with `T` degrees of freedom and scale `sigma_mu`, takes the element-wise `q` and `1 - q` quantiles as the bounds, then projects each bound to the nearest positive definite matrix. This is the covariance half of [`ucs`](@ref) and the whole of [`sigma_ucs`](@ref), so the simulation is written once and the two entry points cannot drift apart. The mean of a ``\\mathrm{Wishart}(T, \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})`` draw is ``T \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``, which is ``\\hat{\\mathbf{\\Sigma}}``, so the draws centre on the point estimate and the bounds bracket it.

# Algorithm

 1. Read the point estimate `sigma = pr.sigma`, the centre the bounds bracket.
 2. Resolve the random number generator from `ue.rng` and `ue.seed` with [`resolve_rng`](@ref).
 3. Draw `ue.n_sim` matrices `sigmas` from `Distributions.Wishart(T, sigma_mu)`.
 4. Take the element-wise `q` and `1 - q` quantiles of `sigmas` with [`box_quantile_bounds`](@ref), giving `sigma_l` and `sigma_u`.
 5. Repair `sigma_l` and `sigma_u` in place with `posdef!`. A quantile is taken entry by entry, so neither bound is a Wishart draw and neither is positive definite by construction. A bound that is already positive definite is not moved.
 6. Return a [`BoxUncertaintySet`](@ref) carrying `sigma_l`, `sigma_u` and `sigma`.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `pr`: Prior result carrying the point estimate `pr.sigma`.
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref), used as the Wishart degrees of freedom.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref), used as the Wishart scale.
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
function sigma_normal_box_set(ue::NormalUncertaintySet, pr::LowOrderPrior, T::Number,
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
    normal_box_preamble(ue::NormalUncertaintySet, X::MatNum,
                        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Shared preamble of the three box entry points of a [`NormalUncertaintySet`](@ref).

Fits the prior, resolves the scaling parameter, and derives the two quantities both box sets are built from. [`ucs`](@ref), [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) all start here, so the prior call and the scaling choice are written once instead of three times. **The returned `q` is halved, and the ellipsoidal route halves nothing.** A box bounds each entry on both sides, so half of the significance level goes into each tail. An ellipsoid cuts only the upper tail of a distance that cannot be negative, so one cut at the ``1 - q`` quantile already covers ``1 - q``.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`.
 2. Resolve the scaling parameter with [`choose_scaling_parameter`](@ref), giving `T`.
 3. Build the mean asymptotic covariance with [`mu_asymptotic_cov`](@ref), giving `sigma_mu`.
 4. Halve `ue.q`, giving `q`, the significance level of one tail.
 5. Return `pr`, `T`, `sigma_mu` and `q` as a tuple.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `(pr, T, sigma_mu, q)`: Prior result, scaling parameter, mean asymptotic covariance, and half significance level.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`choose_scaling_parameter`](@ref)
  - [`mu_asymptotic_cov`](@ref)
  - [`mu_normal_box_set`](@ref)
  - [`sigma_normal_box_set`](@ref)
"""
function normal_box_preamble(ue::NormalUncertaintySet, X::MatNum,
                             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    T = choose_scaling_parameter(ue, pr)
    return pr, T, mu_asymptotic_cov(ue.pdm, pr.sigma, T), ue.q * 0.5
end
"""
    normal_mu_error_sample(ue::NormalUncertaintySet, rng::Random.AbstractRNG, mu::VecNum,
                           sigma_mu::MatNum)

Draw `ue.n_sim` estimation errors of the mean from the sampling law that normal returns imply.

**The sample is the estimation error, not the level.** The draws come from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}}/T)`` and `mu` is subtracted from each, so the sample and the shape it is measured against are on one scale. A draw from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}})`` in its place multiplies every deviation, and therefore the radius, by ``\\sqrt{T}``. The generator is passed in rather than resolved here, so a caller that needs both axes draws them off one stream.

# Algorithm

 1. Draw `ue.n_sim` vectors from `Distributions.MvNormal(mu, sigma_mu)`, giving one column per simulation.
 2. Subtract `mu` from every column, and transpose, giving one estimation error per row.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `rng`: Random number generator, resolved by the caller with [`resolve_rng`](@ref).
  - `mu`: Point estimate of the expected returns vector, the centre the draws are taken around.
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

Draw `ue.n_sim` estimation errors of the vectorised covariance from the Wishart law that normal returns imply.

**The sample is the estimation error, not the level.** Each draw has `sigma` subtracted from it, and the variance of an entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw is the matching diagonal entry of the covariance asymptotic covariance, so the sample and the shape it is measured against are on one scale. `N` is passed in rather than read from `sigma`, so every caller reads the asset count from one source, `size(pr.X, 2)`.

# Algorithm

 1. Draw `ue.n_sim` matrices from `Distributions.Wishart(T, sigma_mu)`, giving `sigmas`.
 2. Subtract `sigma` from each draw into an `N` by `N` by `ue.n_sim` array.
 3. Reshape to `N^2` columns and transpose, giving one vectorised estimation error per row.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `rng`: Random number generator, resolved by the caller with [`resolve_rng`](@ref).
  - `sigma`: Point estimate of the covariance matrix, the centre the draws are taken around.
  - `sigma_mu`: Mean asymptotic covariance from [`mu_asymptotic_cov`](@ref), the Wishart scale.
  - `T`: Scaling parameter from [`choose_scaling_parameter`](@ref), the Wishart degrees of freedom.
  - `N`: Asset count, read by the caller from `size(pr.X, 2)`.

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
"""
    ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for mean and covariance statistics under the assumption of normally distributed returns.

# Mathematical definition

Simulates ``M`` covariance matrices ``\\hat{\\mathbf{\\Sigma}}^{(m)} \\sim \\mathrm{Wishart}(T,\\, \\hat{\\mathbf{\\Sigma}}/T)`` and computes element-wise quantile bounds:

```math
\\begin{align}
(\\mathbf{\\Sigma}_{l})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{u})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Mean bounds use the normal complementary quantile ``z_{q/2} = \\Phi^{-1}(1-q/2)``:

```math
\\begin{align}
\\boldsymbol{\\mu}_{l} &= \\boldsymbol{0}\\,, \\\\
\\mu_{u,i} &= 2 z_{q/2} \\sqrt{(\\hat{\\mathbf{\\Sigma}}/T)_{ii}}\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{l})_{ij}``, ``(\\mathbf{\\Sigma}_{u})_{ij}``: Element-wise lower/upper covariance bounds.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Lower and upper quantile functions at level ``q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: ``(i,j)``-element of the ``m``-th simulated Wishart covariance draw.
  - ``\\boldsymbol{\\mu}_{l}``, ``\\mu_{u,i}``: Lower and upper bounds for expected returns.
  - ``z_{q/2} = \\Phi^{-1}(1 - q/2)``: Normal complementary quantile.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - ``q``: Significance level.

# Algorithm

 1. Run [`normal_box_preamble`](@ref), giving `pr`, `T`, `sigma_mu` and the halved `q`.
 2. Build the mean set with [`mu_normal_box_set`](@ref) from `pr.mu`, `sigma_mu` and `q`.
 3. Build the covariance set with [`sigma_normal_box_set`](@ref) from `ue`, `pr`, `T`, `sigma_mu` and `q`.
 4. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty sets.

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
    pr, T, sigma_mu, q = normal_box_preamble(ue, X, F; dims = dims, kwargs...)
    return mu_normal_box_set(pr.mu, sigma_mu, q),
           sigma_normal_box_set(ue, pr, T, sigma_mu, q)
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for expected returns under the assumption of normally distributed returns.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\mu}_{l} &= \\boldsymbol{0}\\,, \\\\
\\mu_{u,i} &= 2 \\Phi^{-1}\\!\\left(1 - \\tfrac{q}{2}\\right) \\sqrt{(\\hat{\\mathbf{\\Sigma}}/T)_{ii}}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\mu}_{l}``, ``\\mu_{u,i}``: Lower and upper bounds for expected returns.
  - ``\\Phi^{-1}(1 - q/2)``: Normal complementary quantile.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - ``q``: Significance level.

# Algorithm

 1. Run [`normal_box_preamble`](@ref), giving `pr`, `sigma_mu` and the halved `q`. The scaling parameter is dropped, because only the covariance set reads it.
 2. Build and return the mean set with [`mu_normal_box_set`](@ref) from `pr.mu`, `sigma_mu` and `q`.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

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
    pr, _, sigma_mu, q = normal_box_preamble(ue, X, F; dims = dims, kwargs...)
    return mu_normal_box_set(pr.mu, sigma_mu, q)
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
(\\mathbf{\\Sigma}_{l})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{u})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{l})_{ij}``, ``(\\mathbf{\\Sigma}_{u})_{ij}``: Element-wise lower/upper covariance bounds.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Lower and upper quantile functions at level ``q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: ``(i,j)``-element of the ``m``-th Wishart draw.
  - $(math_dict[:T])
  - ``q``: Significance level.

# Algorithm

 1. Run [`normal_box_preamble`](@ref), giving `pr`, `T`, `sigma_mu` and the halved `q`.
 2. Build and return the covariance set with [`sigma_normal_box_set`](@ref) from `ue`, `pr`, `T`, `sigma_mu` and `q`.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

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
    pr, T, sigma_mu, q = normal_box_preamble(ue, X, F; dims = dims, kwargs...)
    return sigma_normal_box_set(ue, pr, T, sigma_mu, q)
end
"""
    ucs(ue::NormalUncertaintySet{<:Any,
                                 <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                 <:Any, <:Any, <:Any},
        X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipsoidal uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns.

**The two samples are estimation errors, not levels.** [`k_ucs`](@ref) measures a Mahalanobis distance against the shape matrix, so step 5 draws from the sampling law of the estimator, ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}}/T)``, and centres the draws on ``\\hat{\\boldsymbol{\\mu}}``. A draw from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}})`` in its place multiplies every deviation, and therefore the radius, by ``\\sqrt{T}``. Step 6 is on the matching scale for the same reason: the variance of an entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw is the matching diagonal entry of ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``.

**One generator serves both draws, so this method's covariance radius is not [`sigma_ucs`](@ref)'s.** Issue #590 holds the decision on whether to change that. Step 4 resolves the generator once, and step 5 consumes it before step 6 reaches it. [`sigma_ucs`](@ref) draws its Wishart matrices off a generator that nothing has advanced, so under one seed the two covariance radii differ. The mean radius agrees, because the mean sample is the first draw in both. The box route has no such split: [`sigma_normal_box_set`](@ref) resolves its own generator, so there the two entry points agree entry for entry.

# Mathematical definition

Ellipsoidal sets centred at the prior estimates with asymptotic covariances:

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} &= \\hat{\\mathbf{\\Sigma}} / T\\,, \\\\
\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} &= T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})\\,.
\\end{align}
```

The radius of each ellipsoid is the one [`k_ucs`](@ref) returns for [`NormalKUncertaintyAlgorithm`](@ref), which states its own form. The significance level reaches it undivided, because an ellipsoid cuts only the upper tail of a distance that cannot be negative.

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
  - ``\\mathbf{I}``: Identity matrix.
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``\\mathcal{E}_{\\boldsymbol{\\mu}}``: Ellipsoidal uncertainty set for expected returns.
  - ``\\mathcal{E}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for covariance.
  - ``k_{\\boldsymbol{\\mu}}``, ``k_{\\mathbf{\\Sigma}}``: Radii of the two ellipsoids, fitted on the sampled estimation errors.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `mu`, `sigma` and `N = size(pr.X, 2)` from it. The asset count comes from the prior's own returns matrix, so a prior that changes the asset count is followed.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 3. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Resolve the random number generator from `ue.rng` and `ue.seed` with [`resolve_rng`](@ref).
 5. Draw the mean sample with [`normal_mu_error_sample`](@ref), giving `X_mu`, one mean estimation error per row.
 6. Draw the covariance sample with [`normal_sigma_error_sample`](@ref), giving `X_sigma`, one vectorised covariance estimation error per row.
 7. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 8. Fit the mean ellipsoid with [`ellipsoidal_set`](@ref) on `X_mu` and `sigma_mu`, with `pr.mu` as the centre.
 9. Fit the covariance ellipsoid the same way on `X_sigma` and `sigma_sigma`, with `pr.sigma` as the centre.
10. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty sets.

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
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_mu, sigma_mu,
                           MuUncertaintySetClass(), pr.mu),
           ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_sigma, sigma_sigma,
                           SigmaUncertaintySetClass(), pr.sigma)
end
"""
    ucs(ue::NormalUncertaintySet{<:Any, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                 <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipsoidal uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns, using a generic ellipsoidal algorithm.

**This route runs no simulation, so it serves every radius algorithm that reads no sample.** That is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref), and a plain number. Its sibling on [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs. The two routes build the same shapes, so they differ only in the radius.

# Mathematical definition

Ellipsoidal sets centred at the prior estimates, whose shapes are the two asymptotic covariances:

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
  - ``\\mathbf{I}``: Identity matrix.
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.

The radius of each ellipsoid is the one [`k_ucs`](@ref) returns for `ue.alg.method`, and that algorithm states its own closed form. The significance level reaches it undivided, because an ellipsoid cuts only the upper tail of a distance that cannot be negative.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 3. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 5. Fit the mean ellipsoid with [`ellipsoidal_set`](@ref) on `sigma_mu`, passing `nothing` in place of a sample and `pr.mu` as the centre.
 6. Fit the covariance ellipsoid the same way on `sigma_sigma`, with `pr.sigma` as the centre.
 7. Return the two sets as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty sets.

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
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, nothing, sigma_mu,
                           MuUncertaintySetClass(), pr.mu),
           ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, nothing, sigma_sigma,
                           SigmaUncertaintySetClass(), pr.sigma)
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for expected returns under the assumption of normally distributed returns, using a normal scaling algorithm.

**The sample is the estimation error, not the level.** [`k_ucs`](@ref) measures a Mahalanobis distance against the shape matrix, so step 4 draws from the sampling law of the estimator, ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}}/T)``, and centres the draws on ``\\hat{\\boldsymbol{\\mu}}``. A draw from ``\\mathcal{N}(\\hat{\\boldsymbol{\\mu}}, \\hat{\\mathbf{\\Sigma}})`` in its place multiplies every deviation, and therefore the radius, by ``\\sqrt{T}``.

# Mathematical definition

```math
\\mathcal{E}_{\\boldsymbol{\\mu}} = \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\boldsymbol{\\mu}}^2 \\right\\}\\,, \\qquad \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} = \\hat{\\mathbf{\\Sigma}} / T\\,.
```

Where:

  - ``\\mathcal{E}_{\\boldsymbol{\\mu}}``: Ellipsoidal uncertainty set for expected returns.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated expected returns vector.
  - ``k_{\\boldsymbol{\\mu}}``: Empirically fitted radius, the ``1 - q`` quantile of the sampled Mahalanobis distances.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `mu` and `sigma` from it.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 3. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Resolve the random number generator with [`resolve_rng`](@ref), and draw the sample with [`normal_mu_error_sample`](@ref), giving `X_mu`, one estimation error per row.
 5. Fit and return the ellipsoid with [`ellipsoidal_set`](@ref) on `X_mu` and `sigma_mu`, with `pr.mu` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.

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
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_mu, sigma_mu,
                           MuUncertaintySetClass(), pr.mu)
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                    <:Any, <:Any, <:Any},
           X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for expected returns under the assumption of normally distributed returns, using a generic ellipsoidal algorithm.

The shape matrix is ``\\hat{\\mathbf{\\Sigma}} / T`` of Equation 11.24. This method runs no simulation, so it serves every radius algorithm that reads none, which is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref) and a plain number. Its sibling on [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs, and builds the same shape.

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

The significance level reaches [`k_ucs`](@ref) undivided, because an ellipsoid cuts only the upper tail of a distance that cannot be negative.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 3. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Fit and return the ellipsoid with [`ellipsoidal_set`](@ref) on `sigma_mu`, passing `nothing` in place of a sample and `pr.mu` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`NormalUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
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
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, nothing, sigma_mu,
                           MuUncertaintySetClass(), pr.mu)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any},
                                       <:Any, <:Any, <:Any},
              X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for covariance under the assumption of normally distributed returns, using a normal scaling algorithm.

**The sample is the estimation error, not the level.** The draws of step 4 are subtracted from ``\\hat{\\mathbf{\\Sigma}}``, and the variance of an entry of a ``\\mathrm{Wishart}(T, \\hat{\\mathbf{\\Sigma}}/T)`` draw is the matching diagonal entry of ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``, so the sample and the shape it is measured against are on one scale. `N` is read from `size(pr.X, 2)`, the same source [`ucs`](@ref) reads it from, so a prior that changes the asset count moves both, and the two shape matrices are equal.

**The radius is not the one [`ucs`](@ref) fits, under the same seed.** Issue #590 holds the decision on whether to change that. This method draws its Wishart matrices off a generator that nothing has advanced. [`ucs`](@ref) resolves one generator and draws its mean sample off it first, so its Wishart draws differ from these. Both radii are valid fits of the same quantity, and only the box route makes the two entry points agree entry for entry.

# Mathematical definition

```math
\\mathcal{E}_{\\mathbf{\\Sigma}} = \\left\\{\\mathbf{\\Sigma} : \\left\\lVert  \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}})  \\right\\rVert_2 \\leq k_{\\mathbf{\\Sigma}} \\right\\}\\,, \\qquad \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} = T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})\\,.
```

Where:

  - ``\\mathcal{E}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for the covariance matrix.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the covariance estimator, vectorised.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator.
  - ``\\mathbf{I}``: Identity matrix.
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``k_{\\mathbf{\\Sigma}}``: Empirically fitted radius, the ``1 - q`` quantile of the sampled Mahalanobis distances.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma` and `N = size(pr.X, 2)`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 3. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Resolve the random number generator with [`resolve_rng`](@ref), and draw the sample with [`normal_sigma_error_sample`](@ref), giving `X_sigma`, one vectorised estimation error per row.
 5. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 6. Fit and return the ellipsoid with [`ellipsoidal_set`](@ref) on `X_sigma` and `sigma_sigma`, with `pr.sigma` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

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
    rng = resolve_rng(ue.rng, ue.seed)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_sigma, sigma_sigma,
                           SigmaUncertaintySetClass(), pr.sigma)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any},
                                       <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for covariance under the assumption of normally distributed returns, using a generic ellipsoidal algorithm.

**This route runs no simulation, so it serves every radius algorithm that reads no sample.** That is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref), and a plain number. Its sibling on [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs. The two routes build the same shapes, so they differ only in the radius.

# Mathematical definition

```math
\\mathcal{E}_{\\mathbf{\\Sigma}} = \\left\\{\\mathbf{\\Sigma} : \\left\\lVert  \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}})  \\right\\rVert_2 \\leq k_{\\mathbf{\\Sigma}} \\right\\}\\,, \\qquad \\mathbf{\\Sigma}_{\\mathbf{\\Sigma}} = T (\\mathbf{I} + \\mathbf{K})(\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\otimes \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}})\\,.
```

Where:

  - ``\\mathcal{E}_{\\mathbf{\\Sigma}}``: Ellipsoidal uncertainty set for the covariance matrix.
  - ``\\mathbf{\\Sigma}_{\\mathbf{\\Sigma}}``: Asymptotic covariance of the covariance estimator, vectorised.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Asymptotic covariance of the mean estimator, ``\\hat{\\mathbf{\\Sigma}} / T``.
  - ``\\mathbf{I}``: Identity matrix.
  - ``\\mathbf{K}``: Commutation matrix.
  - ``\\otimes``: Kronecker product.
  - ``k_{\\mathbf{\\Sigma}}``: Radius, the one [`k_ucs`](@ref) returns for `ue.alg.method`.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])

The significance level reaches [`k_ucs`](@ref) undivided, because an ellipsoid cuts only the upper tail of a distance that cannot be negative.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref).
 3. Build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 4. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 5. Fit and return the ellipsoid with [`ellipsoidal_set`](@ref) on `sigma_sigma`, passing `nothing` in place of a sample and `pr.sigma` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Covariance uncertainty set.

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
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, nothing, sigma_sigma,
                           SigmaUncertaintySetClass(), pr.sigma)
end

"""
    ucs(ue::NormalUncertaintySet{<:Any,
                                 <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                 <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs norm-ball uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns, using a normal scaling algorithm.

The two sets are the two ellipsoids of the sibling route with their shape matrices factorised, so they name the same region and reach the same weights. The gain is on the consumer's side: a [`NormBallUncertaintySet`](@ref) carries the factor, so neither builder factorises anything at solve time. This route draws the sample the empirical radius reads, off one generator, so its Wishart draws follow its normal draws.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `mu`, `sigma` and `N = size(pr.X, 2)`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref), and build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 3. Resolve one generator with [`resolve_rng`](@ref), and draw the two samples with [`normal_mu_error_sample`](@ref) and [`normal_sigma_error_sample`](@ref), mean first.
 4. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 5. Assemble the two sets with [`norm_ball_set`](@ref), on the mean shape and on the covariance shape, and return them as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

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

  - $(ref_dict[:cajas2025]) Equations 11.16, 11.17 and 11.24.
  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
"""
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                        <:Any, <:Any},
                                      <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; mu, sigma) = pr
    N = size(pr.X, 2)
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    return norm_ball_set(ue.alg, ue.q, X_mu, sigma_mu, MuUncertaintySetClass(), pr.mu),
           norm_ball_set(ue.alg, ue.q, X_sigma, sigma_sigma, SigmaUncertaintySetClass(),
                         pr.sigma)
end
"""
    ucs(ue::NormalUncertaintySet{<:Any, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                 <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs norm-ball uncertainty sets for expected returns and covariance statistics under the assumption of normally distributed returns, using a generic radius algorithm.

**This route runs no simulation, so it serves every radius algorithm that reads no sample.** That is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref), and a plain number. Its sibling on [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs. The two routes build the same maps, so they differ only in the radius.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref), and build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 3. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 4. Assemble the two sets with [`norm_ball_set`](@ref), passing `nothing` in place of a sample, and return them as a tuple, mean first.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

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

  - $(ref_dict[:cajas2025]) Equations 11.16, 11.17 and 11.24.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
"""
function ucs(ue::NormalUncertaintySet{<:Any,
                                      <:NormBallUncertaintySetAlgorithm{<:Any, <:Any,
                                                                        <:Any}, <:Any,
                                      <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    return norm_ball_set(ue.alg, ue.q, nothing, sigma_mu, MuUncertaintySetClass(), pr.mu),
           norm_ball_set(ue.alg, ue.q, nothing, sigma_sigma, SigmaUncertaintySetClass(),
                         pr.sigma)
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any,
                                    <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                    <:Any, <:Any, <:Any}, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a norm-ball uncertainty set for expected returns under the assumption of normally distributed returns, using a normal scaling algorithm.

The map is the factor of ``\\hat{\\mathbf{\\Sigma}} / T``, so the set is the mean ellipsoid of the sibling route with its shape factorised. This method draws its normal sample off a generator that nothing has advanced, which is the same stream position [`ucs`](@ref) draws its own mean sample from, so the two radii agree under one seed.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `mu` and `sigma`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref), and build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 3. Resolve the generator with [`resolve_rng`](@ref), and draw the sample with [`normal_mu_error_sample`](@ref).
 4. Assemble and return the set with [`norm_ball_set`](@ref), with `pr.mu` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

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

  - $(ref_dict[:cajas2025]) Equation 11.24.
  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                           <:Any, <:Any},
                                         <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    (; mu, sigma) = pr
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_mu = normal_mu_error_sample(ue, rng, mu, sigma_mu)
    return norm_ball_set(ue.alg, ue.q, X_mu, sigma_mu, MuUncertaintySetClass(), pr.mu)
end
"""
    mu_ucs(ue::NormalUncertaintySet{<:Any, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                    <:Any, <:Any, <:Any}, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a norm-ball uncertainty set for expected returns under the assumption of normally distributed returns, using a generic radius algorithm.

**This route runs no simulation, so it serves every radius algorithm that reads no sample.** That is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref), and a plain number. Its sibling on [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs, and builds the same map.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref), and build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 3. Assemble and return the set with [`norm_ball_set`](@ref), passing `nothing` in place of a sample and `pr.mu` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

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

  - $(ref_dict[:cajas2025]) Equation 11.24.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
"""
function mu_ucs(ue::NormalUncertaintySet{<:Any,
                                         <:NormBallUncertaintySetAlgorithm{<:Any, <:Any,
                                                                           <:Any}, <:Any,
                                         <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    return norm_ball_set(ue.alg, ue.q, nothing, sigma_mu, MuUncertaintySetClass(), pr.mu)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                       <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any},
                                       <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a norm-ball uncertainty set for covariance under the assumption of normally distributed returns, using a normal scaling algorithm.

The map is the factor of the vectorised covariance's asymptotic covariance, so the set is the covariance ellipsoid of the sibling route with its shape factorised. **The radius is not the one [`ucs`](@ref) fits, under the same seed**, because this method draws its Wishart matrices off a generator that nothing has advanced while [`ucs`](@ref) draws its mean sample first. Both radii are valid fits of the same quantity.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma` and `N = size(pr.X, 2)`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref), and build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 3. Resolve the generator with [`resolve_rng`](@ref), and draw the sample with [`normal_sigma_error_sample`](@ref).
 4. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 5. Assemble and return the set with [`norm_ball_set`](@ref), with `pr.sigma` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

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

  - $(ref_dict[:cajas2025]) Equations 11.17 and 11.24.
  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm,
                                                                              <:Any, <:Any},
                                            <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    N = size(pr.X, 2)
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    rng = resolve_rng(ue.rng, ue.seed)
    X_sigma = normal_sigma_error_sample(ue, rng, sigma, sigma_mu, T, N)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    return norm_ball_set(ue.alg, ue.q, X_sigma, sigma_sigma, SigmaUncertaintySetClass(),
                         pr.sigma)
end
"""
    sigma_ucs(ue::NormalUncertaintySet{<:Any, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any},
                                       <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a norm-ball uncertainty set for covariance under the assumption of normally distributed returns, using a generic radius algorithm.

**This route runs no simulation, so it serves every radius algorithm that reads no sample.** That is [`ChiSqKUncertaintyAlgorithm`](@ref), [`GeneralKUncertaintyAlgorithm`](@ref), and a plain number. Its sibling on [`NormalKUncertaintyAlgorithm`](@ref) draws the sample that the empirical radius needs, and builds the same map.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `sigma = pr.sigma`.
 2. Resolve the scaling parameter `T` with [`choose_scaling_parameter`](@ref), and build the mean shape `sigma_mu` with [`mu_asymptotic_cov`](@ref).
 3. Build the covariance shape `sigma_sigma` with [`sigma_asymptotic_cov`](@ref).
 4. Assemble and return the set with [`norm_ball_set`](@ref), passing `nothing` in place of a sample and `pr.sigma` as the centre.

# Arguments

  - `ue`: Normal uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

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

  - $(ref_dict[:cajas2025]) Equations 11.17 and 11.24.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
"""
function sigma_ucs(ue::NormalUncertaintySet{<:Any,
                                            <:NormBallUncertaintySetAlgorithm{<:Any, <:Any,
                                                                              <:Any}, <:Any,
                                            <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    sigma = pr.sigma
    T = choose_scaling_parameter(ue, pr)
    sigma_mu = mu_asymptotic_cov(ue.pdm, sigma, T)
    sigma_sigma = sigma_asymptotic_cov(ue.pdm, sigma_mu, sigma, T)
    return norm_ball_set(ue.alg, ue.q, nothing, sigma_sigma, SigmaUncertaintySetClass(),
                         pr.sigma)
end

export NormalUncertaintySet
