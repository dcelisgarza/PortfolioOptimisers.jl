"""
    abstract type BaseSmythBrobyCovariance <: BaseGerberCovariance end

Abstract supertype for all Smyth-Broby covariance estimators in PortfolioOptimisers.jl.

All concrete types implementing Smyth-Broby covariance estimation algorithms should subtype `BaseSmythBrobyCovariance`. This enables a consistent interface for Smyth-Broby-based covariance estimators throughout the package.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
"""
abstract type BaseSmythBrobyCovariance <: BaseGerberCovariance end
"""
    abstract type SmythBrobyCovarianceAlgorithm <: AbstractMomentAlgorithm end

Abstract supertype for all Smyth-Broby covariance algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific Smyth-Broby covariance algorithms should subtype `SmythBrobyCovarianceAlgorithm`. This enables flexible extension and dispatch of Smyth-Broby covariance routines.

These types are used to specify the algorithm when constructing a [`SmythBrobyCovariance`](@ref) estimator.

# Related

  - [`BaseSmythBrobyCovariance`](@ref)
  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
"""
abstract type SmythBrobyCovarianceAlgorithm <: AbstractMomentAlgorithm end
"""
    abstract type UnstandardisedSmythBrobyCovarianceAlgorithm <: SmythBrobyCovarianceAlgorithm end

Abstract supertype for all unstandardised Smyth-Broby covariance algorithm types.

Concrete types implementing unstandardised Smyth-Broby covariance algorithms should subtype `UnstandardisedSmythBrobyCovarianceAlgorithm`.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`SmythBrobyCovariance`](@ref)
"""
abstract type UnstandardisedSmythBrobyCovarianceAlgorithm <: SmythBrobyCovarianceAlgorithm end
"""
    abstract type StandardisedSmythBrobyCovarianceAlgorithm <: SmythBrobyCovarianceAlgorithm end

Abstract supertype for all standardised Smyth-Broby covariance algorithm types. These Z-transform the data before applying the Smyth-Broby covariance algorithm.

Concrete types implementing standardised Smyth-Broby covariance algorithms should subtype `StandardisedSmythBrobyCovarianceAlgorithm`.

# Related

  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`StandardisedSmythBroby0`](@ref)
  - [`StandardisedSmythBroby1`](@ref)
  - [`StandardisedSmythBroby2`](@ref)
  - [`StandardisedSmythBrobyGerber0`](@ref)
  - [`StandardisedSmythBrobyGerber1`](@ref)
  - [`StandardisedSmythBrobyGerber2`](@ref)
  - [`SmythBrobyCovariance`](@ref)
"""
abstract type StandardisedSmythBrobyCovarianceAlgorithm <: SmythBrobyCovarianceAlgorithm end
"""
    struct SmythBroby0 <: UnstandardisedSmythBrobyCovarianceAlgorithm end

Implements the original Smyth-Broby covariance algorithm (unstandardised variant).

# Related

  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
"""
struct SmythBroby0 <: UnstandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct SmythBroby1 <: UnstandardisedSmythBrobyCovarianceAlgorithm end

Implements the first variant of the Smyth-Broby covariance algorithm (unstandardised).

# Related

  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby2`](@ref)
"""
struct SmythBroby1 <: UnstandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct SmythBroby2 <: UnstandardisedSmythBrobyCovarianceAlgorithm end

Implements the second variant of the Smyth-Broby covariance algorithm (unstandardised).

# Related

  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
"""
struct SmythBroby2 <: UnstandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct SmythBrobyGerber0 <: UnstandardisedSmythBrobyCovarianceAlgorithm end

Implements the original Gerber-style variant of the Smyth-Broby covariance algorithm (unstandardised).

# Related

  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
"""
struct SmythBrobyGerber0 <: UnstandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct SmythBrobyGerber1 <: UnstandardisedSmythBrobyCovarianceAlgorithm end

Implements the first Gerber-style variant of the Smyth-Broby covariance algorithm (unstandardised).

# Related

  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber2`](@ref)
"""
struct SmythBrobyGerber1 <: UnstandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct SmythBrobyGerber2 <: UnstandardisedSmythBrobyCovarianceAlgorithm end

Implements the second Gerber-style variant of the Smyth-Broby covariance algorithm (unstandardised).

# Related

  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
"""
struct SmythBrobyGerber2 <: UnstandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct StandardisedSmythBroby0 <: StandardisedSmythBrobyCovarianceAlgorithm end

Implements the original Smyth-Broby covariance algorithm on Z-transformed data (standardised variant).

# Related

  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBroby1`](@ref)
  - [`StandardisedSmythBroby2`](@ref)
"""
struct StandardisedSmythBroby0 <: StandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct StandardisedSmythBroby1 <: StandardisedSmythBrobyCovarianceAlgorithm end

Implements the first variant of the Smyth-Broby covariance algorithm on Z-transformed data (standardised).

# Related

  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBroby0`](@ref)
  - [`StandardisedSmythBroby2`](@ref)
"""
struct StandardisedSmythBroby1 <: StandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct StandardisedSmythBroby2 <: StandardisedSmythBrobyCovarianceAlgorithm end

Implements the second variant of the Smyth-Broby covariance algorithm on Z-transformed data (standardised).

# Related

  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBroby0`](@ref)
  - [`StandardisedSmythBroby1`](@ref)
"""
struct StandardisedSmythBroby2 <: StandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct StandardisedSmythBrobyGerber0 <: StandardisedSmythBrobyCovarianceAlgorithm end

Implements the original Gerber-style variant of the Smyth-Broby covariance algorithm on Z-transformed data (standardised).

# Related

  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBrobyGerber1`](@ref)
  - [`StandardisedSmythBrobyGerber2`](@ref)
"""
struct StandardisedSmythBrobyGerber0 <: StandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct StandardisedSmythBrobyGerber1 <: StandardisedSmythBrobyCovarianceAlgorithm end

Implements the first Gerber-style variant of the Smyth-Broby covariance algorithm on Z-transformed data (standardised).

# Related

  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBrobyGerber0`](@ref)
  - [`StandardisedSmythBrobyGerber2`](@ref)
"""
struct StandardisedSmythBrobyGerber1 <: StandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct StandardisedSmythBrobyGerber2 <: StandardisedSmythBrobyCovarianceAlgorithm end

Implements the second Gerber-style variant of the Smyth-Broby covariance algorithm on Z-transformed data (standardised).

# Related

  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBrobyGerber0`](@ref)
  - [`StandardisedSmythBrobyGerber1`](@ref)
"""
struct StandardisedSmythBrobyGerber2 <: StandardisedSmythBrobyCovarianceAlgorithm end
"""
    struct SmythBrobyCovariance{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
           BaseSmythBrobyCovariance
        me::T1
        ve::T2
        pdm::T3
        threshold::T4
        c1::T5
        c2::T6
        c3::T7
        n::T8
        alg::T9
        threads::T10
    end

A flexible container type for configuring and applying Smyth-Broby covariance estimators in PortfolioOptimisers.jl.

`SmythBrobyCovariance` encapsulates all components required for Smyth-Broby-based covariance or correlation estimation, including the expected returns estimator, variance estimator, positive definite matrix estimator, algorithm parameters, and the specific Smyth-Broby algorithm variant. This enables modular and extensible workflows for robust covariance estimation using Smyth-Broby statistics.

# Fields

  - `me`: Expected returns estimator.
  - `ve`: Variance estimator.
  - `pdm`: Positive definite matrix estimator.
  - `threshold`: Threshold parameter for Smyth-Broby covariance computation.
  - `c1`: Zone of confusion parameter.
  - `c2`: Zone of indecision lower bound.
  - `c3`: Zone of indecision upper bound.
  - `n`: Exponent parameter for the Smyth-Broby kernel.
  - `alg`: Smyth-Broby covariance algorithm variant.
  - `threads`: Parallel execution strategy.

# Constructor

    SmythBrobyCovariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                         ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                         pdm::Union{Nothing, <:Posdef} = Posdef(), threshold::Real = 0.5,
                         c1::Real = 0.5, c2::Real = 0.5, c3::Real = 4, n::Real = 2,
                         alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1(),
                         threads::FLoops.Transducers.Executor = ThreadedEx())

Keyword arguments correspond to the fields above.

## Validation

  - `0 < threshold < 1`.
  - `0 < c1 <= 1`.
  - `0 < c2 <= 1`.
  - `c3 > c2`.

# Examples

```jldoctest
julia> SmythBrobyCovariance()
SmythBrobyCovariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
         ve ┼ SimpleVariance
            │          me ┼ SimpleExpectedReturns
            │             │   w ┴ nothing
            │           w ┼ nothing
            │   corrected ┴ Bool: true
        pdm ┼ Posdef
            │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
  threshold ┼ Float64: 0.5
         c1 ┼ Float64: 0.5
         c2 ┼ Float64: 0.5
         c3 ┼ Int64: 4
          n ┼ Int64: 2
        alg ┼ SmythBrobyGerber1()
    threads ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`BaseSmythBrobyCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`SimpleVariance`](@ref)
  - [`Posdef`](@ref)
  - [`SmythBrobyCovarianceAlgorithm`](@ref)
  - [`SmythBroby0`](@ref)
  - [`SmythBroby1`](@ref)
  - [`SmythBroby2`](@ref)
  - [`StandardisedSmythBroby0`](@ref)
  - [`StandardisedSmythBroby1`](@ref)
  - [`StandardisedSmythBroby2`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`StandardisedSmythBrobyGerber0`](@ref)
  - [`StandardisedSmythBrobyGerber1`](@ref)
  - [`StandardisedSmythBrobyGerber2`](@ref)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-executor)
"""
struct SmythBrobyCovariance{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       BaseSmythBrobyCovariance
    me::T1
    ve::T2
    pdm::T3
    threshold::T4
    c1::T5
    c2::T6
    c3::T7
    n::T8
    alg::T9
    threads::T10
    function SmythBrobyCovariance(me::AbstractExpectedReturnsEstimator,
                                  ve::StatsBase.CovarianceEstimator,
                                  pdm::Union{Nothing, <:Posdef}, threshold::Real, c1::Real,
                                  c2::Real, c3::Real, n::Real,
                                  alg::SmythBrobyCovarianceAlgorithm,
                                  threads::FLoops.Transducers.Executor)
        @argcheck(zero(threshold) < threshold < one(threshold),
                  DomainError("0 < threshold < 1 must hold. Got\nthreshold => $threshold"))
        @argcheck(zero(c1) < c1 <= one(c1),
                  DomainError("0 < c1 <= 1 must hold. Got\nc1 => $c1"))
        @argcheck(zero(c2) < c2 <= one(c2),
                  DomainError("0 < c2 <= 1 must hold. Got\nc2 => $c2"))
        @argcheck(c2 < c3, DomainError)
        return new{typeof(me), typeof(ve), typeof(pdm), typeof(threshold), typeof(c1),
                   typeof(c2), typeof(c3), typeof(n), typeof(alg), typeof(threads)}(me, ve,
                                                                                    pdm,
                                                                                    threshold,
                                                                                    c1, c2,
                                                                                    c3, n,
                                                                                    alg,
                                                                                    threads)
    end
end
function SmythBrobyCovariance(;
                              me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                              ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                              pdm::Union{Nothing, <:Posdef} = Posdef(),
                              threshold::Real = 0.5, c1::Real = 0.5, c2::Real = 0.5,
                              c3::Real = 4, n::Real = 2,
                              alg::SmythBrobyCovarianceAlgorithm = SmythBrobyGerber1(),
                              threads::FLoops.Transducers.Executor = ThreadedEx())
    return SmythBrobyCovariance(me, ve, pdm, threshold, c1, c2, c3, n, alg, threads)
end
function factory(ce::SmythBrobyCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SmythBrobyCovariance(; me = factory(ce.me, w), ve = factory(ce.ve, w),
                                pdm = ce.pdm, threshold = ce.threshold, c1 = ce.c1,
                                c2 = ce.c2, c3 = ce.c3, n = ce.n, alg = ce.alg,
                                threads = ce.threads)
end
"""
    sb_delta(xi::Real, xj::Real, mui::Real, muj::Real, sigmai::Real, sigmaj::Real, c1::Real,
             c2::Real, c3::Real, n::Real)

Smyth-Broby kernel function for covariance and correlation computation.

This function computes the kernel value for a pair of asset returns, applying the Smyth-Broby logic for zones of confusion and indecision. It is used to aggregate positive and negative co-movements in Smyth-Broby covariance algorithms.

# Arguments

  - `xi`: Return for asset `i`.
  - `xj`: Return for asset `j`.
  - `mui`: Mean for asset `i`.
  - `muj`: Mean for asset `j`.
  - `sigmai`: Standard deviation for asset `i`.
  - `sigmaj`: Standard deviation for asset `j`.
  - `c1`: Zone of confusion parameter.
  - `c2`: Zone of indecision lower bound.
  - `c3`: Zone of indecision upper bound.
  - `n`: Exponent parameter for the kernel.

# Returns

  - `score::Real`: The computed score for the pair `(xi, xj)`.

# Details

 1. If both returns are within the zone of confusion (`abs(xi) < sigmai * c1` and `abs(xj) < sigmaj * c1`), returns zero.
 2. Computes centered and scaled returns `ri`, `rj`.
 3. If both are within the zone of indecision (`ri < c2 && rj < c2`) or both are above the upper bound (`ri > c3 && rj > c3`), returns zero.
 4. Otherwise, returns `sqrt((1 + ri) * (1 + rj)) / (1 + abs(ri - rj)^n)`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`smythbroby`](@ref)
"""
function sb_delta(xi::Real, xj::Real, mui::Real, muj::Real, sigmai::Real, sigmaj::Real,
                  c1::Real, c2::Real, c3::Real, n::Real)
    # Zone of confusion.
    # If the return is not a significant proportion of the standard deviation, we classify it as noise.
    if abs(xi) < sigmai * c1 && abs(xj) < sigmaj * c1
        return zero(eltype(xi))
    end

    # Zone of indecision.
    # Center returns at mu = 0 and sigma = 1.
    ri = abs((xi - mui) / sigmai)
    rj = abs((xj - muj) / sigmaj)
    # If the return is less than c2 standard deviations, or greater than c3 standard deviations, we can't make a call since it may be noise, or overall market forces.
    if ri < c2 && rj < c2 || ri > c3 && rj > c3
        return zero(eltype(xi))
    end

    kappa = sqrt((one(ri) + ri) * (one(rj) + rj))
    gamma = abs(ri - rj)

    return kappa / (one(gamma) + gamma^n)
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SmythBroby0, <:Any}, X::AbstractMatrix,
               mean_vec::AbstractArray, std_vec::AbstractArray)

Implements the original Smyth-Broby covariance/correlation algorithm (unstandardised variant).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the original `SmythBroby0` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBroby0` algorithm.
  - `X`: Data matrix (observations × assets).
  - `mean_vec`: Vector of means for each asset, used for centering.
  - `std_vec`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions.
 5. The correlation is computed as `(pos - neg) / (pos + neg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby0, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:StandardisedSmythBroby0, <:Any}, X::AbstractMatrix)

Implements the original Smyth-Broby covariance/correlation algorithm on Z-transformed data (standardised variant).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the original `StandardisedSmythBroby0` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `StandardisedSmythBroby0` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel (with mean 0 and standard deviation 1) to accumulate positive (`pos`) and negative (`neg`) contributions.
 5. The correlation is computed as `(pos - neg) / (pos + neg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBroby0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBroby0,
                                             <:Any}, X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                end
            end
            den = (pos + neg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:StandardisedSmythBroby0, <:Any}, X::AbstractMatrix)

Implements the original Smyth-Broby covariance/correlation algorithm on Z-transformed data (standardised variant).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the original `StandardisedSmythBroby0` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `StandardisedSmythBroby0` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel (with mean 0 and standard deviation 1) to accumulate positive (`pos`) and negative (`neg`) contributions.
 5. The correlation is computed as `(pos - neg) / (pos + neg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBroby0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby1, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:StandardisedSmythBroby1, <:Any}, X::AbstractMatrix)

Implements the first variant of the Smyth-Broby covariance/correlation algorithm on Z-transformed data (standardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `StandardisedSmythBroby1` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive, negative, and neutral (non-exceedance) co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `StandardisedSmythBroby1` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive, negative, and neutral co-movements.
 4. Use the `sb_delta` kernel (with mean 0 and standard deviation 1) to accumulate positive (`pos`), negative (`neg`), and neutral (`nn`) contributions.
 5. The correlation is computed as `(pos - neg) / (pos + neg + nn)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBroby1`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBroby1,
                                             <:Any}, X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)), one(eltype(X)),
                                   one(eltype(X)), c1, c2, c3, n)
                end
            end
            den = (pos + neg + nn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (pos - neg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SmythBroby2, <:Any}, X::AbstractMatrix,
               mean_vec::AbstractArray, std_vec::AbstractArray)

Implements the second variant of the Smyth-Broby covariance/correlation algorithm (unstandardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBroby2` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements. The resulting matrix is then standardised by the geometric mean of its diagonal elements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBroby2` algorithm.
  - `X`: Data matrix (observations × assets).
  - `mean_vec`: Vector of means for each asset, used for centering.
  - `std_vec`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, standardised and projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions.
 5. The raw correlation is computed as `pos - neg`.
 6. The resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.
 7. The matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBroby2`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBroby2, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                end
            end
            rho[j, i] = rho[i, j] = pos - neg
        end
    end
    h = sqrt.(diag(rho))
    rho .= Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:StandardisedSmythBroby2, <:Any}, X::AbstractMatrix)

Implements the second variant of the Smyth-Broby covariance/correlation algorithm on Z-transformed data (standardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `StandardisedSmythBroby2` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements. The resulting matrix is then standardised by the geometric mean of its diagonal elements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `StandardisedSmythBroby2` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, standardised and projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel (with mean 0 and standard deviation 1) to accumulate positive (`pos`) and negative (`neg`) contributions.
 5. The raw correlation is computed as `pos - neg`.
 6. The resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.
 7. The matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBroby2`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBroby2,
                                             <:Any}, X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                end
            end
            rho[j, i] = rho[i, j] = pos - neg
        end
    end
    h = sqrt.(diag(rho))
    rho .= Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SmythBrobyGerber0, <:Any}, X::AbstractMatrix,
               mean_vec::AbstractArray, std_vec::AbstractArray)

Implements the original Gerber-style variant of the Smyth-Broby covariance/correlation algorithm (unstandardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber0` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber0` algorithm.
  - `X`: Data matrix (observations × assets).
  - `mean_vec`: Vector of means for each asset, used for centering.
  - `std_vec`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber0, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:StandardisedSmythBrobyGerber0, <:Any},
               X::AbstractMatrix)

Implements the original Gerber-style variant of the Smyth-Broby covariance/correlation algorithm on Z-transformed data (standardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `StandardisedSmythBrobyGerber0` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `StandardisedSmythBrobyGerber0` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel (with mean 0 and standard deviation 1) to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBrobyGerber0`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBrobyGerber0,
                                             <:Any}, X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            den = (tpos + tneg)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SmythBrobyGerber1, <:Any}, X::AbstractMatrix,
               mean_vec::AbstractArray, std_vec::AbstractArray)

Implements the first Gerber-style variant of the Smyth-Broby covariance/correlation algorithm (unstandardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber1` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive, negative, and neutral co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber1` algorithm.
  - `X`: Data matrix (observations × assets).
  - `mean_vec`: Vector of means for each asset, used for centering.
  - `std_vec`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive, negative, and neutral co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`), negative (`neg`), and neutral (`nn`) contributions, and count the number of positive (`cpos`), negative (`cneg`), and neutral (`cnn`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg + nn * cnn)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber1`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber1, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:StandardisedSmythBrobyGerber1, <:Any},
               X::AbstractMatrix)

Implements the first Gerber-style variant of the Smyth-Broby covariance/correlation algorithm on Z-transformed data (standardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `StandardisedSmythBrobyGerber1` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive, negative, and neutral co-movements, with additional weighting by the count of co-movements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `StandardisedSmythBrobyGerber1` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive, negative, and neutral co-movements.
 4. Use the `sb_delta` kernel (with mean 0 and standard deviation 1) to accumulate positive (`pos`), negative (`neg`), and neutral (`nn`) contributions, and count the number of positive (`cpos`), negative (`cneg`), and neutral (`cnn`) co-movements.
 5. The correlation is computed as `(pos * cpos - neg * cneg) / (pos * cpos + neg * cneg + nn * cnn)` if the denominator is nonzero, otherwise zero.
 6. The resulting matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBrobyGerber1`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBrobyGerber1,
                                             <:Any}, X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            cneg = 0
            cpos = 0
            cnn = 0
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                elseif abs(xi) < ti && abs(xj) < tj
                    nn += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)), one(eltype(X)),
                                   one(eltype(X)), c1, c2, c3, n)
                    cnn += 1
                end
            end
            tpos = pos * cpos
            tneg = neg * cneg
            tnn = nn * cnn
            den = (tpos + tneg + tnn)
            rho[j, i] = rho[i, j] = if !iszero(den)
                (tpos - tneg) / den
            else
                zero(eltype(X))
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:SmythBrobyGerber2, <:Any}, X::AbstractMatrix,
               mean_vec::AbstractArray, std_vec::AbstractArray)

Implements the second Gerber-style variant of the Smyth-Broby covariance/correlation algorithm (unstandardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `SmythBrobyGerber2` algorithm. The computation is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements. The resulting matrix is then standardised by the geometric mean of its diagonal elements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `SmythBrobyGerber2` algorithm.
  - `X`: Data matrix (observations × assets).
  - `mean_vec`: Vector of means for each asset, used for centering.
  - `std_vec`: Vector of standard deviations for each asset, used for scaling and thresholding.

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, standardised and projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, compute the centered and scaled returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The raw correlation is computed as `pos * cpos - neg * cneg`.
 6. The resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.
 7. The matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`SmythBrobyGerber2`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:SmythBrobyGerber2, <:Any},
                    X::AbstractMatrix, mean_vec::AbstractArray, std_vec::AbstractArray)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        muj = mean_vec[j]
        sigmaj = std_vec[j]
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            mui = mean_vec[i]
            sigmai = std_vec[i]
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold * sigmai
                tj = threshold * sigmaj
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, mui, muj, sigmai, sigmaj, c1, c2, c3, n)
                    cneg += 1
                end
            end
            rho[j, i] = rho[i, j] = pos * cpos - neg * cneg
        end
    end
    h = sqrt.(diag(rho))
    rho .= Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                        <:StandardisedSmythBrobyGerber2, <:Any},
               X::AbstractMatrix)

Implements the second Gerber-style variant of the Smyth-Broby covariance/correlation algorithm on Z-transformed data (standardised).

This method computes the Smyth-Broby correlation or covariance matrix for the input data matrix `X` using the `StandardisedSmythBrobyGerber2` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding the data, applying the Smyth-Broby kernel, and aggregating positive and negative co-movements, with additional weighting by the count of co-movements. The resulting matrix is then standardised by the geometric mean of its diagonal elements.

# Arguments

  - `ce`: Smyth-Broby covariance estimator configured with the `StandardisedSmythBrobyGerber2` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix, standardised and projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each pair of assets `(i, j)`, iterate over all observations.
 2. For each observation, use the Z-transformed returns for assets `i` and `j`.
 3. Apply the threshold to classify joint positive and negative co-movements.
 4. Use the `sb_delta` kernel (with mean 0 and standard deviation 1) to accumulate positive (`pos`) and negative (`neg`) contributions, and count the number of positive (`cpos`) and negative (`cneg`) co-movements.
 5. The raw correlation is computed as `pos * cpos - neg * cneg`.
 6. The resulting matrix is standardised by dividing each element by the geometric mean of the corresponding diagonal elements.
 7. The matrix is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`StandardisedSmythBrobyGerber2`](@ref)
  - [`sb_delta`](@ref)
  - [`posdef!`](@ref)
"""
function smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:StandardisedSmythBrobyGerber2,
                                             <:Any}, X::AbstractMatrix)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    threshold = ce.threshold
    c1 = ce.c1
    c2 = ce.c2
    c3 = ce.c3
    n = ce.n
    @floop ce.threads for j in axes(X, 2)
        for i in 1:j
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            cneg = 0
            cpos = 0
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                ti = threshold
                tj = threshold
                if xi >= ti && xj >= tj || xi <= -ti && xj <= -tj
                    pos += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cpos += 1
                elseif xi >= ti && xj <= -tj || xi <= -ti && xj >= tj
                    neg += sb_delta(xi, xj, zero(eltype(X)), zero(eltype(X)),
                                    one(eltype(X)), one(eltype(X)), c1, c2, c3, n)
                    cneg += 1
                end
            end
            rho[j, i] = rho[i, j] = pos * cpos - neg * cneg
        end
    end
    h = sqrt.(diag(rho))
    rho .= Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    cor(ce::SmythBrobyCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the Smyth-Broby correlation matrix.

This method computes the Smyth-Broby correlation matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Smyth-Broby correlation is then computed via [`smythbroby`](@ref).

# Arguments

  - `ce`: Smyth-Broby covariance estimator.

      + `ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:UnstandardisedSmythBrobyCovarianceAlgorithm, <:Any}`: Compute the unstandardised Smyth-Broby correlation matrix.
      + `ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyCovarianceAlgorithm, <:Any}`: Compute the standardised Smyth-Broby correlation matrix.

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `mean`: Optional mean vector for centering. If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `rho::Matrix{<:Real}`: The Smyth-Broby correlation matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBroby0, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBroby1, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBroby2, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBrobyGerber0, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBrobyGerber1, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBrobyGerber2, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBroby0, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBroby1, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBroby2, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyGerber0, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyGerber1, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyGerber2, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cov(ce::SmythBrobyCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:UnstandardisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? Statistics.mean(ce.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    return smythbroby(ce, X, mean_vec, std_vec)
end
function Statistics.cor(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:StandardisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? Statistics.mean(ce.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return smythbroby(ce, X)
end
"""
    cov(ce::SmythBrobyCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the Smyth-Broby covariance matrix.

This method computes the Smyth-Broby covariance matrix for the input data matrix `X`. The mean and standard deviation vectors are computed using the estimator's expected returns and variance estimators. The Smyth-Broby covariance is then computed via [`smythbroby`](@ref).

# Arguments

  - `ce`: Smyth-Broby covariance estimator.

      + `ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:UnstandardisedSmythBrobyCovarianceAlgorithm, <:Any}`: Compute the unstandardised Smyth-Broby covariance matrix.
      + `ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyCovarianceAlgorithm, <:Any}`: Compute the standardised Smyth-Broby covariance matrix.

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `mean`: Optional mean vector for centering. If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the mean and standard deviation estimators.

# Returns

  - `sigma::Matrix{<:Real}`: The Smyth-Broby covariance matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`SmythBrobyCovariance`](@ref)
  - [`UnstandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`StandardisedSmythBrobyCovarianceAlgorithm`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBroby0, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBroby1, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBroby2, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBrobyGerber0, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBrobyGerber1, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SmythBrobyGerber2, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBroby0, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBroby1, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBroby2, <:Any}, X::AbstractMatrix)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyGerber0, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyGerber1, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`smythbroby(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:StandardisedSmythBrobyGerber2, <:Any}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cov(ce::SmythBrobyCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:UnstandardisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? Statistics.mean(ce.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    return smythbroby(ce, X, mean_vec, std_vec) ⊙ (std_vec ⊗ std_vec)
end
function Statistics.cov(ce::SmythBrobyCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Any, <:Any,
                                                 <:StandardisedSmythBrobyCovarianceAlgorithm,
                                                 <:Any}, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? Statistics.mean(ce.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return smythbroby(ce, X) ⊙ (std_vec ⊗ std_vec)
end

export SmythBroby0, SmythBroby1, SmythBroby2, SmythBrobyGerber0, SmythBrobyGerber1,
       SmythBrobyGerber2, StandardisedSmythBroby0, StandardisedSmythBroby1,
       StandardisedSmythBroby2, StandardisedSmythBrobyGerber0,
       StandardisedSmythBrobyGerber1, StandardisedSmythBrobyGerber2, SmythBrobyCovariance
