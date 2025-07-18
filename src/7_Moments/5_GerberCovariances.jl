"""
    abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end

Abstract supertype for all Gerber covariance estimators in PortfolioOptimisers.jl.

All concrete types implementing Gerber covariance estimation algorithms should subtype `BaseGerberCovariance`. This enables a consistent interface for Gerber-based covariance estimators throughout the package.

# Related

  - [`GerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
"""
abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end

"""
    abstract type GerberCovarianceAlgorithm <: AbstractMomentAlgorithm end

Abstract supertype for all Gerber covariance algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific Gerber covariance algorithms should subtype `GerberCovarianceAlgorithm`. This enables flexible extension and dispatch of Gerber covariance routines.

These types are used to specify the algorithm when constructing a [`GerberCovariance`](@ref) estimator.

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`UnNormalisedGerberCovarianceAlgorithm`](@ref)
  - [`NormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
"""
abstract type GerberCovarianceAlgorithm <: AbstractMomentAlgorithm end

"""
    abstract type UnNormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end

Abstract supertype for all unnormalised Gerber covariance algorithm types.

Concrete types implementing unnormalised Gerber covariance algorithms should subtype `UnNormalisedGerberCovarianceAlgorithm`.

# Related

  - [`GerberCovarianceAlgorithm`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`GerberCovariance`](@ref)
"""
abstract type UnNormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end

"""
    abstract type NormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end

Abstract supertype for all normalised Gerber covariance algorithm types. These Z-transform the data before applying the Gerber covariance algorithm.

Concrete types implementing normalised Gerber covariance algorithms should subtype `NormalisedGerberCovarianceAlgorithm`.

# Related

  - [`GerberCovarianceAlgorithm`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber1`](@ref)
  - [`NormalisedGerber2`](@ref)
  - [`GerberCovariance`](@ref)
"""
abstract type NormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end

"""
    struct Gerber0 <: UnNormalisedGerberCovarianceAlgorithm end

Implements the original Gerber covariance algorithm.

# Related

  - [`UnNormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
"""
struct Gerber0 <: UnNormalisedGerberCovarianceAlgorithm end

"""
    struct Gerber1 <: UnNormalisedGerberCovarianceAlgorithm end

Implements the first variant of the Gerber covariance algorithm.

# Related

  - [`UnNormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber2`](@ref)
"""
struct Gerber1 <: UnNormalisedGerberCovarianceAlgorithm end

"""
    struct Gerber2 <: UnNormalisedGerberCovarianceAlgorithm end

Implements the second variant of the Gerber covariance algorithm.

# Related

  - [`UnNormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
"""
struct Gerber2 <: UnNormalisedGerberCovarianceAlgorithm end

"""
    struct NormalisedGerber0{T1 <: AbstractExpectedReturnsEstimator} <: NormalisedGerberCovarianceAlgorithm
        me::T1
    end

Implements the original Gerber covariance algorithm on Z-transformed data.

# Fields

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used for mean-centering prior to normalisation.

# Constructor

  - `NormalisedGerber0(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())`: Construct the original normalised Gerber covariance algorithm.

# Related

  - [`NormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`NormalisedGerber1`](@ref)
  - [`NormalisedGerber2`](@ref)
"""
struct NormalisedGerber0{T1 <: AbstractExpectedReturnsEstimator} <:
       NormalisedGerberCovarianceAlgorithm
    me::T1
end
"""
    NormalisedGerber0(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())

Creates a [`NormalisedGerber0`](@ref) object using the specified expected returns estimator for mean-centering prior to normalisation.

# Arguments

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used for mean-centering before normalisation.

# Returns

  - `NormalisedGerber0`: An instance of the original normalised Gerber covariance algorithm.

# Examples

```jldoctest
julia> ng0 = NormalisedGerber0()
NormalisedGerber0
  me | SimpleExpectedReturns
     |   w | nothing
```

# Related

  - [`NormalisedGerber0`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
"""
function NormalisedGerber0(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber0{typeof(me)}(me)
end

"""
    struct NormalisedGerber1{T1 <: AbstractExpectedReturnsEstimator} <: NormalisedGerberCovarianceAlgorithm
        me::T1
    end

Implements the first variant of the Gerber covariance algorithm on Z-transformed data.

# Fields

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used for mean-centering prior to normalisation.

# Constructor

  - `NormalisedGerber1(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())`: Construct the first variant of the normalised Gerber covariance algorithm.

# Related

  - [`NormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber2`](@ref)
"""
struct NormalisedGerber1{T1 <: AbstractExpectedReturnsEstimator} <:
       NormalisedGerberCovarianceAlgorithm
    me::T1
end
"""
    NormalisedGerber1(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())

Creates a [`NormalisedGerber1`](@ref) object using the specified expected returns estimator for mean-centering prior to normalisation.

# Arguments

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used for mean-centering before normalisation.

# Returns

  - `NormalisedGerber1`: An instance of the original normalised Gerber covariance algorithm.

# Examples

```jldoctest
julia> ng0 = NormalisedGerber1()
NormalisedGerber1
  me | SimpleExpectedReturns
     |   w | nothing
```

# Related

  - [`NormalisedGerber1`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
"""
function NormalisedGerber1(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber1{typeof(me)}(me)
end

"""
    struct NormalisedGerber2{T1 <: AbstractExpectedReturnsEstimator} <: NormalisedGerberCovarianceAlgorithm
        me::T1
    end

Implements the second variant of the Gerber covariance algorithm on Z-transformed data.

# Fields

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used for mean-centering prior to normalisation.

# Constructors

  - `NormalisedGerber2(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())`: Construct the second variant of the normalised Gerber covariance algorithm.

These types are used to specify the algorithm when constructing a [`GerberCovariance`](@ref) estimator with normalisation.

# Related

  - [`NormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber1`](@ref)
"""
struct NormalisedGerber2{T1 <: AbstractExpectedReturnsEstimator} <:
       NormalisedGerberCovarianceAlgorithm
    me::T1
end
"""
    NormalisedGerber2(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())

Creates a [`NormalisedGerber2`](@ref) object using the specified expected returns estimator for mean-centering prior to normalisation.

# Arguments

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used for mean-centering before normalisation.

# Returns

  - `NormalisedGerber2`: An instance of the original normalised Gerber covariance algorithm.

# Examples

```jldoctest
julia> ng0 = NormalisedGerber2()
NormalisedGerber2
  me | SimpleExpectedReturns
     |   w | nothing
```

# Related

  - [`NormalisedGerber2`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
"""
function NormalisedGerber2(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber2{typeof(me)}(me)
end
for alg in (Gerber0, Gerber1, Gerber2)
    eval(quote
             function factory(alg::$(alg), ::Any)
                 return alg
             end
         end)
end
for alg in (NormalisedGerber0, NormalisedGerber1, NormalisedGerber2)
    eval(quote
             function factory(alg::$(alg), w::Union{Nothing, <:AbstractWeights})
                 return $(alg)(; me = factory(alg.me, w))
             end
         end)
end

"""
    struct GerberCovariance{T1 <: StatsBase.CovarianceEstimator,
                            T2 <: PosdefEstimator,
                            T3 <: Real,
                            T4 <: GerberCovarianceAlgorithm} <: BaseGerberCovariance
        ve::T1
        pdm::T2
        threshold::T3
        alg::T4
    end

A flexible container type for configuring and applying Gerber covariance estimators in PortfolioOptimisers.jl.

`GerberCovariance` encapsulates all components required for Gerber-based covariance or correlation estimation, including the variance estimator, positive definite matrix estimator, threshold parameter, and the specific Gerber algorithm variant. This enables modular and extensible workflows for robust covariance estimation using Gerber statistics.

# Fields

  - `ve::StatsBase.CovarianceEstimator`: Variance estimator.
  - `pdm::PosdefEstimator`: Positive definite matrix estimator (see [`PosdefEstimator`](@ref)).
  - `threshold::Real`: Threshold parameter for Gerber covariance computation (typically in (0, 1)).
  - `alg::GerberCovarianceAlgorithm`: Gerber covariance algorithm variant.

# Constructor

    GerberCovariance(; alg::GerberCovarianceAlgorithm = Gerber1(),
                      ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                      pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator(),
                      threshold::Real = 0.5)

Construct a `GerberCovariance` estimator with the specified algorithm, variance estimator, positive definite estimator, and threshold.

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`SimpleVariance`](@ref)
  - [`PosdefEstimator`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber1`](@ref)
  - [`NormalisedGerber2`](@ref)
"""
struct GerberCovariance{T1 <: StatsBase.CovarianceEstimator, T2 <: PosdefEstimator,
                        T3 <: Real, T4 <: GerberCovarianceAlgorithm} <: BaseGerberCovariance
    ve::T1
    pdm::T2
    threshold::T3
    alg::T4
end
"""
    GerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                     pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator(),
                     threshold::Real = 0.5,
                     alg::GerberCovarianceAlgorithm = Gerber1())

Construct a [`GerberCovariance`](@ref) estimator for robust Gerber-based covariance or correlation estimation.

This constructor creates a `GerberCovariance` object using the specified Gerber algorithm, variance estimator, positive definite matrix estimator, and threshold parameter. The estimator is highly modular, allowing users to select from different Gerber algorithm variants, and positive definite projections.

# Arguments

  - `ve::StatsBase.CovarianceEstimator`: Variance estimator.
  - `pdm::Union{Nothing, <:PosdefEstimator}`: Positive definite matrix estimator.
  - `threshold::Real`: Threshold parameter for Gerber covariance computation.
  - `alg::GerberCovarianceAlgorithm`: Gerber covariance algorithm variant.

# Returns

  - `GerberCovariance`: A configured Gerber covariance estimator.

# Validation

  - Asserts that `threshold` is strictly in `(0, 1)`.

# Examples

```jldoctest
julia> gc = GerberCovariance()
GerberCovariance
         ve | SimpleVariance
            |          me | SimpleExpectedReturns
            |             |   w | nothing
            |           w | nothing
            |   corrected | Bool: true
        pdm | PosdefEstimator
            |   alg | UnionAll: NearestCorrelationMatrix.Newton
  threshold | Float64: 0.5
        alg | Gerber1()
```

# Related

  - [`GerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`SimpleVariance`](@ref)
  - [`PosdefEstimator`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber1`](@ref)
  - [`NormalisedGerber2`](@ref)
"""
function GerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                          pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator(),
                          threshold::Real = 0.5, alg::GerberCovarianceAlgorithm = Gerber1())
    @smart_assert(zero(threshold) < threshold < one(threshold))
    return GerberCovariance{typeof(ve), typeof(pdm), typeof(threshold), typeof(alg)}(ve,
                                                                                     pdm,
                                                                                     threshold,
                                                                                     alg)
end

"""
    gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix, std_vec::AbstractArray)

Implements the original Gerber correlation algorithm.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the original Gerber0 algorithm. The computation is based on thresholding the standardized data and counting co-occurrences of threshold exceedances.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}`: Gerber correlation estimator configured with the `Gerber0` algorithm.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `std_vec::AbstractArray`: Vector of standard deviations for each asset, used to scale the threshold.

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each entry in `X`, compute two Boolean matrices:

      + `U`: Entries where `X` exceeds `threshold * std_vec`.
      + `D`: Entries where `X` is less than `-threshold * std_vec`.

 2. Compute `UmD = U - D` and `UpD = U + D`.
 3. The Gerber correlation is given by `(UmD' * UmD) ⊘ (UpD' * UpD)`.
 4. The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`posdef!`](@ref)
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix,
                std_vec::AbstractArray)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    std_vec = std_vec * ce.threshold
    U .= X .>= std_vec
    D .= X .<= -std_vec
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    UpD = U + D
    rho = (transpose(UmD) * UmD) ⊘ (transpose(UpD) * UpD)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0}, X::AbstractMatrix)

Implements the original Gerber correlation algorithm on Z-transformed data.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the original `NormalisedGerber0` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding and counting co-occurrences of threshold exceedances.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0}`: Gerber correlation estimator configured with the `NormalisedGerber0` algorithm.
  - `X::AbstractMatrix`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each entry in `X`, compute two Boolean matrices:

      + `U`: Entries where `X` exceeds `ce.threshold`.
      + `D`: Entries where `X` is less than `-ce.threshold`.

 2. Compute `UmD = U - D` and `UpD = U + D`.
 3. The Gerber correlation is given by `(UmD' * UmD) ⊘ (UpD' * UpD)`.
 4. The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`posdef!`](@ref)
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0},
                X::AbstractMatrix)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    U .= X .>= ce.threshold
    D .= X .<= -ce.threshold
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    UpD = U + D
    rho = (transpose(UmD) * UmD) ⊘ (transpose(UpD) * UpD)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix, std_vec::AbstractArray)

Implements the first variant of the Gerber correlation algorithm.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the Gerber1 algorithm. The computation is based on thresholding the standardized data, counting co-occurrences of threshold exceedances, and adjusting for non-exceedance events.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}`: Gerber correlation estimator configured with the `Gerber1` algorithm.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `std_vec::AbstractArray`: Vector of standard deviations for each asset, used to scale the threshold.

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each entry in `X`, compute three Boolean matrices:

      + `U`: Entries where `X` exceeds `threshold * std_vec`.
      + `D`: Entries where `X` is less than `-threshold * std_vec`.
      + `N`: Entries where `X` is within `[-threshold * std_vec, threshold * std_vec]` (i.e., neither up nor down).

 2. Compute `UmD = U - D`.
 3. The Gerber1 correlation is given by `(UmD' * UmD) ⊘ (T .- (N' * N))`, where `T` is the number of observations.
 4. The result is projected to the nearest positive definite matrix using `posdef!`.
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix,
                std_vec::AbstractArray)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    N = Matrix{Bool}(undef, T, N)
    std_vec = std_vec * ce.threshold
    U .= X .>= std_vec
    D .= X .<= -std_vec
    N .= .!U .& .!D
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    rho = transpose(UmD) * (UmD) ⊘ (T .- transpose(N) * N)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1}, X::AbstractMatrix)

Implements the first variant of the Gerber correlation algorithm on Z-transformed data.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the `NormalisedGerber1` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding, counting co-occurrences of threshold exceedances, and adjusting for non-exceedance events.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1}`: Gerber correlation estimator configured with the `NormalisedGerber1` algorithm.
  - `X::AbstractMatrix`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each entry in `X`, compute three Boolean matrices:

      + `U`: Entries where `X` exceeds `ce.threshold`.
      + `D`: Entries where `X` is less than `-ce.threshold`.
      + `N`: Entries where `X` is within `[-ce.threshold, ce.threshold]` (i.e., neither up nor down).

 2. Compute `UmD = U - D`.
 3. The Gerber1 correlation is given by `(UmD' * UmD) ⊘ (T .- (N' * N))`, where `T` is the number of observations.
 4. The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`NormalisedGerber1`](@ref)
  - [`posdef!`](@ref)
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1},
                X::AbstractMatrix)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    N = Matrix{Bool}(undef, T, N)
    U .= X .>= ce.threshold
    D .= X .<= -ce.threshold
    N .= .!U .& .!D
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    rho = transpose(UmD) * (UmD) ⊘ (T .- transpose(N) * N)
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix, std_vec::AbstractArray)

Implements the second variant of the Gerber correlation algorithm.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the Gerber2 algorithm. The computation is based on thresholding the standardized data, constructing a signed indicator matrix, and normalizing by the geometric mean of diagonal elements.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}`: Gerber correlation estimator configured with the `Gerber2` algorithm.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `std_vec::AbstractArray`: Vector of standard deviations for each asset, used to scale the threshold.

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation or correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each entry in `X`, compute two Boolean matrices:

      + `U`: Entries where `X` exceeds `threshold * std_vec`.
      + `D`: Entries where `X` is less than `-threshold * std_vec`.

 2. Compute the signed indicator matrix `UmD = U - D`.
 3. Compute the raw Gerber2 matrix `H = UmD' * UmD`.
 4. Normalize: `rho = H ⊘ (h * h')`, where `h = sqrt.(diag(H))`.
 5. The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`Gerber2`](@ref)
  - [`posdef!`](@ref)
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix,
                std_vec::AbstractArray)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    std_vec = std_vec * ce.threshold
    U .= X .>= std_vec
    D .= X .<= -std_vec
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    H = transpose(UmD) * (UmD)
    h = sqrt.(diag(H))
    rho = H ⊘ (h * transpose(h))
    posdef!(ce.pdm, rho)
    return rho
end
"""
    gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2}, X::AbstractMatrix)

Implements the second variant of the Gerber correlation algorithm on Z-transformed data.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the `NormalisedGerber2` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding, constructing a signed indicator matrix, and normalizing by the geometric mean of diagonal elements.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2}`: Gerber correlation estimator configured with the `NormalisedGerber2` algorithm.
  - `X::AbstractMatrix`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

# Details

The algorithm proceeds as follows:

 1. For each entry in `X`, compute two Boolean matrices:

      + `U`: Entries where `X` exceeds `ce.threshold`.
      + `D`: Entries where `X` is less than `-ce.threshold`.

 2. Compute the signed indicator matrix `UmD = U - D`.
 3. Compute the raw Gerber2 matrix `H = UmD' * UmD`.
 4. Normalize: `rho = H ⊘ (h * h')`, where `h = sqrt.(diag(H))`.
 5. The result is projected to the nearest positive definite matrix using `posdef!`.

# Related

  - [`GerberCovariance`](@ref)
  - [`NormalisedGerber2`](@ref)
  - [`posdef!`](@ref)
"""
function gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2},
                X::AbstractMatrix)
    T, N = size(X)
    U = Matrix{Bool}(undef, T, N)
    D = Matrix{Bool}(undef, T, N)
    U .= X .>= ce.threshold
    D .= X .<= -ce.threshold
    # nconc = transpose(U) * U + transpose(D) * D
    # ndisc = transpose(U) * D + transpose(D) * U
    # H = nconc - ndisc
    UmD = U - D
    H = transpose(UmD) * (UmD)
    h = sqrt.(diag(H))
    rho = H ⊘ (h * transpose(h))
    posdef!(ce.pdm, rho)
    return rho
end

"""
    cor(ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm},
        X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Gerber correlation matrix using an unnormalised Gerber covariance estimator.

This method computes the Gerber correlation matrix for the input data matrix `X` using the specified unnormalised Gerber covariance estimator. The standard deviation vector is computed using the estimator's variance estimator. The Gerber correlation is then computed via [`gerber`](@ref).

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm}`: Gerber covariance estimator.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation matrix.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`cov(ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:UnNormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return gerber(ce, X, std_vec)
end

"""
    cov(ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm},
        X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Gerber covariance matrix using an unnormalised Gerber covariance estimator.

This method computes the Gerber covariance matrix for the input data matrix `X` using the specified unnormalised Gerber covariance estimator. The standard deviation vector is computed using the estimator's variance estimator. The Gerber correlation is computed via [`gerber`](@ref), and the result is rescaled to a covariance matrix using the standard deviation vector.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm}`: Gerber covariance estimator.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - `sigma::Matrix{Float64}`: The Gerber covariance matrix.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`cor(ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:UnNormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return gerber(ce, X, std_vec) ⊙ (std_vec ⊗ std_vec)
end

"""
    cor(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm},
        X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Gerber correlation matrix using a normalised Gerber covariance estimator.

This method computes the Gerber correlation matrix for the input data matrix `X` using the specified normalised Gerber covariance estimator. The standard deviation vector is computed using the estimator's variance estimator. The Gerber correlation is then computed via [`gerber`](@ref).

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm}`: Gerber covariance estimator.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - `rho::Matrix{Float64}`: The Gerber correlation matrix.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2}, X::AbstractMatrix)`](@ref)
  - [`cov(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:NormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? Statistics.mean(ce.alg.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return gerber(ce, X)
end

"""
    cov(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm},
        X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Gerber covariance matrix using a normalised Gerber covariance estimator.

This method computes the Gerber covariance matrix for the input data matrix `X` using the specified normalised Gerber covariance estimator. The standard deviation vector is computed using the estimator's variance estimator. The Gerber correlation is computed via [`gerber`](@ref), and the result is rescaled to a covariance matrix using the standard deviation vector.

# Arguments

  - `ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm}`: Gerber covariance estimator.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - `sigma::Matrix{Float64}`: The Gerber covariance matrix.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2}, X::AbstractMatrix)`](@ref)
  - [`cor(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm}, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:NormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mean_vec = isnothing(mean) ? Statistics.mean(ce.alg.me, X; dims = 1, kwargs...) : mean
    std_vec = std(ce.ve, X; dims = 1, mean = mean_vec, kwargs...)
    idx = iszero.(std_vec)
    std_vec[idx] .= eps(eltype(X))
    X = (X .- mean_vec) ⊘ std_vec
    return gerber(ce, X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::GerberCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return GerberCovariance(; alg = factory(ce.alg, w), ve = factory(ce.ve, w),
                            pdm = ce.pdm, threshold = ce.threshold)
end

export GerberCovariance, Gerber0, Gerber1, Gerber2, NormalisedGerber0, NormalisedGerber1,
       NormalisedGerber2
