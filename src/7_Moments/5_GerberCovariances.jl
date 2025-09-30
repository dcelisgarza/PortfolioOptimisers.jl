"""
```julia
abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end
```

Abstract supertype for all Gerber covariance estimators in PortfolioOptimisers.jl.

All concrete types implementing Gerber covariance estimation algorithms should subtype `BaseGerberCovariance`. This enables a consistent interface for Gerber-based covariance estimators throughout the package.

# Related

  - [`GerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
"""
abstract type BaseGerberCovariance <: AbstractCovarianceEstimator end

"""
```julia
abstract type GerberCovarianceAlgorithm <: AbstractMomentAlgorithm end
```

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
```julia
abstract type UnNormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end
```

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
```julia
abstract type NormalisedGerberCovarianceAlgorithm <: GerberCovarianceAlgorithm end
```

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
```julia
struct Gerber0 <: UnNormalisedGerberCovarianceAlgorithm end
```

Implements the original Gerber covariance algorithm.

# Related

  - [`UnNormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
"""
struct Gerber0 <: UnNormalisedGerberCovarianceAlgorithm end

"""
```julia
struct Gerber1 <: UnNormalisedGerberCovarianceAlgorithm end
```

Implements the first variant of the Gerber covariance algorithm.

# Related

  - [`UnNormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber2`](@ref)
"""
struct Gerber1 <: UnNormalisedGerberCovarianceAlgorithm end

"""
```julia
struct Gerber2 <: UnNormalisedGerberCovarianceAlgorithm end
```

Implements the second variant of the Gerber covariance algorithm.

# Related

  - [`UnNormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
"""
struct Gerber2 <: UnNormalisedGerberCovarianceAlgorithm end

"""
```julia
struct NormalisedGerber0{T1} <: NormalisedGerberCovarianceAlgorithm
    me::T1
end
```

Implements the original Gerber covariance algorithm on Z-transformed data.

# Fields

  - `me`: Expected returns estimator used for mean-centering prior to normalisation.

# Constructor

```julia
NormalisedGerber0(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> ng0 = NormalisedGerber0()
NormalisedGerber0
  me | SimpleExpectedReturns
     |   w | nothing
```

# Related

  - [`NormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`NormalisedGerber1`](@ref)
  - [`NormalisedGerber2`](@ref)
"""
struct NormalisedGerber0{T1} <: NormalisedGerberCovarianceAlgorithm
    me::T1
    function NormalisedGerber0(me::AbstractExpectedReturnsEstimator)
        return new{typeof(me)}(me)
    end
end
function NormalisedGerber0(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber0(me)
end

"""
```julia
struct NormalisedGerber1{T1} <: NormalisedGerberCovarianceAlgorithm
    me::T1
end
```

Implements the first variant of the Gerber covariance algorithm on Z-transformed data.

# Fields

  - `me`: Expected returns estimator used for mean-centering prior to normalisation.

# Constructor

```julia
NormalisedGerber1(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> ng0 = NormalisedGerber1()
NormalisedGerber1
  me | SimpleExpectedReturns
     |   w | nothing
```

# Related

  - [`NormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber2`](@ref)
"""
struct NormalisedGerber1{T1} <: NormalisedGerberCovarianceAlgorithm
    me::T1
    function NormalisedGerber1(me::AbstractExpectedReturnsEstimator)
        return new{typeof(me)}(me)
    end
end
function NormalisedGerber1(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber1(me)
end

"""
```julia
struct NormalisedGerber2{T1} <: NormalisedGerberCovarianceAlgorithm
    me::T1
end
```

Implements the second variant of the Gerber covariance algorithm on Z-transformed data.

# Fields

  - `me`: Expected returns estimator used for mean-centering prior to normalisation.

# Constructor

```julia
NormalisedGerber2(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> ng0 = NormalisedGerber2()
NormalisedGerber2
  me | SimpleExpectedReturns
     |   w | nothing
```

# Related

  - [`NormalisedGerberCovarianceAlgorithm`](@ref)
  - [`GerberCovariance`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber1`](@ref)
"""
struct NormalisedGerber2{T1} <: NormalisedGerberCovarianceAlgorithm
    me::T1
    function NormalisedGerber2(me::AbstractExpectedReturnsEstimator)
        return new{typeof(me)}(me)
    end
end
function NormalisedGerber2(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns())
    return NormalisedGerber2(me)
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
```julia
struct GerberCovariance{T1, T2, T3, T4} <: BaseGerberCovariance
    ve::T1
    pdm::T2
    threshold::T3
    alg::T4
end
```

A flexible container type for configuring and applying Gerber covariance estimators in PortfolioOptimisers.jl.

`GerberCovariance` encapsulates all components required for Gerber-based covariance or correlation estimation, including the variance estimator, positive definite matrix estimator, threshold parameter, and the specific Gerber algorithm variant. This enables modular and extensible workflows for robust covariance estimation using Gerber statistics.

# Fields

  - `ve`: Variance estimator.
  - `pdm`: Positive definite matrix estimator (see [`Posdef`](@ref)).
  - `threshold`: Threshold parameter for Gerber covariance computation.
  - `alg`: Gerber covariance algorithm variant.

# Constructor

```julia
GerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                 pdm::Union{Nothing, <:Posdef} = Posdef(), threshold::Real = 0.5,
                 alg::GerberCovarianceAlgorithm = Gerber1())
```

Keyword arguments correspond to the fields above.

## Validation

  - `0 < threshold < 1`.

# Related

  - [`BaseGerberCovariance`](@ref)
  - [`GerberCovarianceAlgorithm`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`SimpleVariance`](@ref)
  - [`Posdef`](@ref)
  - [`Gerber0`](@ref)
  - [`Gerber1`](@ref)
  - [`Gerber2`](@ref)
  - [`NormalisedGerber0`](@ref)
  - [`NormalisedGerber1`](@ref)
  - [`NormalisedGerber2`](@ref)
"""
struct GerberCovariance{T1, T2, T3, T4} <: BaseGerberCovariance
    ve::T1
    pdm::T2
    threshold::T3
    alg::T4
    function GerberCovariance(ve::StatsBase.CovarianceEstimator,
                              pdm::Union{Nothing, <:Posdef}, threshold::Real,
                              alg::GerberCovarianceAlgorithm)
        @argcheck(zero(threshold) < threshold < one(threshold))
        return new{typeof(ve), typeof(pdm), typeof(threshold), typeof(alg)}(ve, pdm,
                                                                            threshold, alg)
    end
end
function GerberCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                          pdm::Union{Nothing, <:Posdef} = Posdef(), threshold::Real = 0.5,
                          alg::GerberCovarianceAlgorithm = Gerber1())
    return GerberCovariance(ve, pdm, threshold, alg)
end

"""
```julia
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix,
       std_vec::AbstractArray)
```

Implements the original Gerber correlation algorithm.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the original Gerber0 algorithm. The computation is based on thresholding the standardized data and counting co-occurrences of threshold exceedances.

# Arguments

  - `ce`: Gerber correlation estimator configured with the `Gerber0` algorithm.
  - `X`: Data matrix (observations × assets).
  - `std_vec`: Vector of standard deviations for each asset, used to scale the threshold.

# Returns

  - `rho::Matrix{<:Real}`: The Gerber correlation, projected to be positive definite using the estimator's `pdm` field.

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
```julia
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0}, X::AbstractMatrix)
```

Implements the original Gerber correlation algorithm on Z-transformed data.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the original `NormalisedGerber0` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding and counting co-occurrences of threshold exceedances.

# Arguments

  - `ce`: Gerber correlation estimator configured with the `NormalisedGerber0` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

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
```julia
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix,
       std_vec::AbstractArray)
```

Implements the first variant of the Gerber correlation algorithm.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the Gerber1 algorithm. The computation is based on thresholding the standardized data, counting co-occurrences of threshold exceedances, and adjusting for non-exceedance events.

# Arguments

  - `ce`: Gerber correlation estimator configured with the `Gerber1` algorithm.
  - `X`: Data matrix (observations × assets).
  - `std_vec`: Vector of standard deviations for each asset, used to scale the threshold.

# Returns

  - `rho::Matrix{<:Real}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

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
```julia
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1}, X::AbstractMatrix)
```

Implements the first variant of the Gerber correlation algorithm on Z-transformed data.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the `NormalisedGerber1` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding, counting co-occurrences of threshold exceedances, and adjusting for non-exceedance events.

# Arguments

  - `ce`: Gerber correlation estimator configured with the `NormalisedGerber1` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

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
```julia
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix,
       std_vec::AbstractArray)
```

Implements the second variant of the Gerber correlation algorithm.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the Gerber2 algorithm. The computation is based on thresholding the standardized data, constructing a signed indicator matrix, and normalizing by the geometric mean of diagonal elements.

# Arguments

  - `ce`: Gerber correlation estimator configured with the `Gerber2` algorithm.
  - `X`: Data matrix (observations × assets).
  - `std_vec`: Vector of standard deviations for each asset, used to scale the threshold.

# Returns

  - `rho::Matrix{<:Real}`: The Gerber correlation or correlation matrix, projected to be positive definite using the estimator's `pdm` field.

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
```julia
gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2}, X::AbstractMatrix)
```

Implements the second variant of the Gerber correlation algorithm on Z-transformed data.

This method computes the Gerber correlation or correlation matrix for the input data matrix `X` using the `NormalisedGerber2` algorithm. The computation is performed on data that has already been Z-transformed (mean-centered and standardised), and is based on thresholding, constructing a signed indicator matrix, and normalizing by the geometric mean of diagonal elements.

# Arguments

  - `ce`: Gerber correlation estimator configured with the `NormalisedGerber2` algorithm.
  - `X`: Z-transformed data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Real}`: The Gerber correlation matrix, projected to be positive definite using the estimator's `pdm` field.

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
```julia
cor(ce::GerberCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the Gerber correlation matrix using an unnormalised Gerber covariance estimator.

This method computes the Gerber correlation matrix for the input data matrix `X` using the specified unnormalised Gerber covariance estimator. The standard deviation vector is computed using the estimator's variance estimator. The Gerber correlation is then computed via [`gerber`](@ref).

# Arguments

  - `ce::GerberCovariance`: Gerber covariance estimator.

      + `ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm}`: Compute the unnormalised Gerber correlation matrix.
      + `ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm}`: Compute the normalised Gerber correlation matrix.

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - `rho::Matrix{<:Real}`: The Gerber correlation matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2}, X::AbstractMatrix)`](@ref)
  - [`cov(ce::GerberCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:UnNormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return gerber(ce, X, std_vec)
end
function Statistics.cor(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:NormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
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
```julia
cov(ce::GerberCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the Gerber covariance matrix using an unnormalised Gerber covariance estimator.

This method computes the Gerber covariance matrix for the input data matrix `X` using the specified unnormalised Gerber covariance estimator. The standard deviation vector is computed using the estimator's variance estimator. The Gerber correlation is computed via [`gerber`](@ref), and the result is rescaled to a covariance matrix using the standard deviation vector.

# Arguments

  - `ce::GerberCovariance`: Gerber covariance estimator.

      + `ce::GerberCovariance{<:Any, <:Any, <:Any, <:UnNormalisedGerberCovarianceAlgorithm}`: Compute the unnormalised Gerber covariance matrix.
      + `ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerberCovarianceAlgorithm}`: Compute the normalised Gerber covariance matrix.

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the standard deviation estimator.

# Returns

  - `sigma::Matrix{<:Real}`: The Gerber covariance matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`GerberCovariance`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber0}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber1}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:Gerber2}, X::AbstractMatrix, std_vec::AbstractArray)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber0}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber1}, X::AbstractMatrix)`](@ref)
  - [`gerber(ce::GerberCovariance{<:Any, <:Any, <:Any, <:NormalisedGerber2}, X::AbstractMatrix)`](@ref)
  - [`cor(ce::GerberCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:UnNormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return gerber(ce, X, std_vec) ⊙ (std_vec ⊗ std_vec)
end
function Statistics.cov(ce::GerberCovariance{<:Any, <:Any, <:Any,
                                             <:NormalisedGerberCovarianceAlgorithm},
                        X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
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
