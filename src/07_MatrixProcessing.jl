"""
    abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end

Abstract supertype for all matrix processing estimator types in `PortfolioOptimisers.jl`.

All concrete types that implement matrix processing routines—such as covariance matrix cleaning, denoising, or detoning—should subtype `AbstractMatrixProcessingEstimator`.

# Interfaces

In order to implement a new matrix processing estimator which will work seamlessly with the library, subtype `AbstractMatrixProcessingEstimator` including all necessary parameters as part of the struct, and implement the following methods:

  - `matrix_processing!`(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...; kwargs...): In-place processing of a covariance or correlation matrix.
  - `matrix_processing`(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...; kwargs...): Optional out-of-place processing of a covariance or correlation matrix.

For example, we can create a dummy matrix processing estimator as follows:

```jldoctest
julia> struct MyMatrixProcessingEstimator <: PortfolioOptimisers.AbstractMatrixProcessingEstimator end

julia> function PortfolioOptimisers.matrix_processing!(est::MyMatrixProcessingEstimator,
                                                       sigma::PortfolioOptimisers.MatNum,
                                                       X::PortfolioOptimisers.MatNum)
           # Implement your in-place matrix processing logic here.
           println("Processing matrix in-place...")
           return sigma
       end

julia> function PortfolioOptimisers.matrix_processing(est::MyMatrixProcessingEstimator,
                                                      sigma::PortfolioOptimisers.MatNum,
                                                      X::PortfolioOptimisers.MatNum)
           sigma = copy(sigma)
           matrix_processing!(est, sigma, X)
           return sigma
       end

julia> matrix_processing!(MyMatrixProcessingEstimator(), [1.0 2.0; 2.0 1.0], rand(10, 2))
Processing matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0

julia> matrix_processing(MyMatrixProcessingEstimator(), [1.0 2.0; 2.0 1.0], rand(10, 2))
Processing matrix in-place...
2×2 Matrix{Float64}:
 1.0  2.0
 2.0  1.0
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
"""
    abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end

Abstract supertype for all matrix processing algorithm types in `PortfolioOptimisers.jl`.

All concrete types that implement a specific matrix processing algorithm should subtype `AbstractMatrixProcessingAlgorithm`.

# Interfaces

In order to implement a new matrix processing algorithm that works with the current matrix processing estimator, subtype `AbstractMatrixProcessingAlgorithm`, including all necessary parameters as part of the struct, and implement the following methods:

  - [`matrix_processing_algorithm!`](@ref): In-place application of a custom matrix processing algorithm.
  - [`matrix_processing_algorithm`](@ref): Optional out-of-place application of a custom matrix processing algorithm.

For example, we can create a dummy matrix processing algorithm as follows:

```jldoctest
julia> struct MyMatrixProcessingAlgorithm <: PortfolioOptimisers.AbstractMatrixProcessingAlgorithm end

julia> function PortfolioOptimisers.matrix_processing_algorithm!(alg::MyMatrixProcessingAlgorithm,
                                                                 sigma::PortfolioOptimisers.MatNum,
                                                                 X::PortfolioOptimisers.MatNum;
                                                                 kwargs...)
           # Implement your in-place matrix processing algorithm logic here.
           println("Applying custom matrix processing algorithm in-place...")
           return sigma
       end

julia> function PortfolioOptimisers.matrix_processing_algorithm(alg::MyMatrixProcessingAlgorithm,
                                                                sigma::PortfolioOptimisers.MatNum,
                                                                X::PortfolioOptimisers.MatNum;
                                                                kwargs...)
           sigma = copy(sigma)
           matrix_processing_algorithm!(alg, sigma, X; kwargs...)
           return sigma
       end

julia> matrix_processing!(DenoiseDetoneAlgMatrixProcessing(; alg = MyMatrixProcessingAlgorithm()),
                          [1.0 2.0; 2.0 1.0], rand(10, 2))
Applying custom matrix processing algorithm in-place...

julia> matrix_processing(DenoiseDetoneAlgMatrixProcessing(; alg = MyMatrixProcessingAlgorithm()),
                         [1.0 2.0; 2.0 1.0], rand(10, 2))
Applying custom matrix processing algorithm in-place...
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end
"""
    abstract type AbstractMatrixProcessingOrder <: AbstractAlgorithm end

Abstract supertype for matrix processing order types in `PortfolioOptimisers.jl`.

All concrete types that specify the order of matrix processing steps—such as denoising, detoning, and algorithm application—should subtype `AbstractMatrixProcessingOrder`. This enables flexible configuration of the sequence in which matrix processing operations are applied within the matrix processing pipeline.

# Related Types

  - [`DenoiseDetoneAlg`](@ref)
  - [`DenoiseAlgDetone`](@ref)
  - [`DetoneDenoiseAlg`](@ref)
  - [`DetoneAlgDenoise`](@ref)
  - [`AlgDenoiseDetone`](@ref)
  - [`AlgDetoneDenoise`](@ref)
"""
abstract type AbstractMatrixProcessingOrder <: AbstractAlgorithm end
"""
    struct DenoiseDetoneAlg <: AbstractMatrixProcessingOrder end

Matrix processing order: Denoising → Detoning → Custom Algorithm.

`DenoiseDetoneAlg` specifies that matrix processing should be performed in the order of denoising, then detoning, followed by any custom algorithm. This type is used to configure the sequence of operations in matrix processing pipelines.

# Related

  - [`AbstractMatrixProcessingOrder`](@ref)
  - [`DenoiseAlgDetone`](@ref)
  - [`DetoneDenoiseAlg`](@ref)
  - [`DetoneAlgDenoise`](@ref)
  - [`AlgDenoiseDetone`](@ref)
  - [`AlgDetoneDenoise`](@ref)
"""
struct DenoiseDetoneAlg <: AbstractMatrixProcessingOrder end
"""
    struct DenoiseAlgDetone <: AbstractMatrixProcessingOrder end

Matrix processing order: Denoising → Custom Algorithm → Detoning.

`DenoiseAlgDetone` specifies that matrix processing should be performed in the order of denoising, then applying a custom algorithm, followed by detoning. This type is used to configure the sequence of operations in matrix processing pipelines.

# Related

  - [`AbstractMatrixProcessingOrder`](@ref)
  - [`DenoiseDetoneAlg`](@ref)
  - [`DetoneDenoiseAlg`](@ref)
  - [`DetoneAlgDenoise`](@ref)
  - [`AlgDenoiseDetone`](@ref)
  - [`AlgDetoneDenoise`](@ref)
"""
struct DenoiseAlgDetone <: AbstractMatrixProcessingOrder end
"""
    struct DetoneDenoiseAlg <: AbstractMatrixProcessingOrder end

Matrix processing order: Detoning → Denoising → Custom Algorithm.

`DetoneDenoiseAlg` specifies that matrix processing should be performed in the order of detoning, then denoising, followed by any custom algorithm. This type is used to configure the sequence of operations in matrix processing pipelines.

# Related

  - [`AbstractMatrixProcessingOrder`](@ref)
  - [`DenoiseDetoneAlg`](@ref)
  - [`DenoiseAlgDetone`](@ref)
  - [`DetoneAlgDenoise`](@ref)
  - [`AlgDenoiseDetone`](@ref)
  - [`AlgDetoneDenoise`](@ref)
"""
struct DetoneDenoiseAlg <: AbstractMatrixProcessingOrder end
"""
    struct DetoneAlgDenoise <: AbstractMatrixProcessingOrder end

Matrix processing order: Detoning → Custom Algorithm → Denoising.

`DetoneAlgDenoise` specifies that matrix processing should be performed in the order of detoning, then applying a custom algorithm, followed by denoising. This type is used to configure the sequence of operations in matrix processing pipelines.

# Related

  - [`AbstractMatrixProcessingOrder`](@ref)
  - [`DenoiseDetoneAlg`](@ref)
  - [`DenoiseAlgDetone`](@ref)
  - [`DetoneDenoiseAlg`](@ref)
  - [`AlgDenoiseDetone`](@ref)
  - [`AlgDetoneDenoise`](@ref)
"""
struct DetoneAlgDenoise <: AbstractMatrixProcessingOrder end
"""
    struct AlgDenoiseDetone <: AbstractMatrixProcessingOrder end

Matrix processing order: Custom Algorithm → Denoising → Detoning.

`AlgDenoiseDetone` specifies that matrix processing should be performed in the order of applying a custom algorithm, then denoising, followed by detoning. This type is used to configure the sequence of operations in matrix processing pipelines.

# Related

  - [`AbstractMatrixProcessingOrder`](@ref)
  - [`DenoiseDetoneAlg`](@ref)
  - [`DenoiseAlgDetone`](@ref)
  - [`DetoneDenoiseAlg`](@ref)
  - [`DetoneAlgDenoise`](@ref)
  - [`AlgDetoneDenoise`](@ref)
"""
struct AlgDenoiseDetone <: AbstractMatrixProcessingOrder end
"""
    struct AlgDetoneDenoise <: AbstractMatrixProcessingOrder end

Matrix processing order: Custom Algorithm → Detoning → Denoising.

`AlgDetoneDenoise` specifies that matrix processing should be performed in the order of applying a custom algorithm, then detoning, followed by denoising. This type is used to configure the sequence of operations in matrix processing pipelines.

# Related

  - [`AbstractMatrixProcessingOrder`](@ref)
  - [`DenoiseDetoneAlg`](@ref)
  - [`DenoiseAlgDetone`](@ref)
  - [`DetoneDenoiseAlg`](@ref)
  - [`DetoneAlgDenoise`](@ref)
  - [`AlgDenoiseDetone`](@ref)
"""
struct AlgDetoneDenoise <: AbstractMatrixProcessingOrder end
"""
    matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)

No-op fallback for matrix processing algorithm routines.

These methods are called internally when no matrix processing algorithm is specified (i.e., when the algorithm argument is `nothing`). They perform no operation and return `nothing`, ensuring that the matrix processing pipeline can safely skip optional algorithmic steps.

# Arguments

  - `::Nothing`: Indicates that no algorithm is not `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is returned unchanged.

# Related

  - [`matrix_processing_algorithm`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
function matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
"""
    matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)

Same as [`matrix_processing_algorithm!`](@ref), but meant for returning a new matrix instead of modifying it in-place.

# Related

  - [`matrix_processing_algorithm!`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
function matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
"""
    struct DenoiseDetoneAlgMatrixProcessing{T1, T2, T3, T4, T5} <: AbstractMatrixProcessingEstimator
        pdm::T1
        dn::T2
        dt::T3
        alg::T4
        order::T5
    end

A flexible container type for configuring and applying matrix processing routines in `PortfolioOptimisers.jl`.

`DenoiseDetoneAlgMatrixProcessing` encapsulates all steps required for processing covariance or correlation matrices, including positive definiteness enforcement, denoising, detoning, and optional custom matrix processing algorithms via [`matrix_processing!`](@ref) and [`matrix_processing`](@ref). This estimator allows users to build complex matrix processing pipelines tailored to their specific needs.

# Fields

  - $(glossary[:opdm])
  - $(glossary[:odn])
  - $(glossary[:odt])
  - `alg`: Optional custom matrix processing algorithm.
  - `order`: Specifies the order in which denoising, detoning, and custom algorithm steps are applied.

# Constructor

    DenoiseDetoneAlgMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                     dn::Option{<:Denoise} = nothing,
                                     dt::Option{<:Detone} = nothing,
                                     alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                     order::AbstractMatrixProcessingOrder = DenoiseDetoneAlg())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> DenoiseDetoneAlgMatrixProcessing()
DenoiseDetoneAlgMatrixProcessing
    pdm ┼ Posdef
        │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
     dn ┼ nothing
     dt ┼ nothing
    alg ┼ nothing
  order ┴ DenoiseDetoneAlg()

julia> DenoiseDetoneAlgMatrixProcessing(; dn = Denoise(), dt = Detone(; n = 2))
DenoiseDetoneAlgMatrixProcessing
    pdm ┼ Posdef
        │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
     dn ┼ Denoise
        │      alg ┼ ShrunkDenoise
        │          │   alpha ┴ Float64: 0.0
        │     args ┼ Tuple{}: ()
        │   kwargs ┼ @NamedTuple{}: NamedTuple()
        │   kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
        │        m ┼ Int64: 10
        │        n ┼ Int64: 1000
        │      pdm ┼ Posdef
        │          │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
     dt ┼ Detone
        │     n ┼ Int64: 2
        │   pdm ┼ Posdef
        │       │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
    alg ┼ nothing
  order ┴ DenoiseDetoneAlg()
```

# Related

  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
  - [`Option`](@ref)
  - [`Posdef`](@ref)
  - [`Denoise`](@ref)
  - [`Detone`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
struct DenoiseDetoneAlgMatrixProcessing{T1, T2, T3, T4, T5} <:
       AbstractMatrixProcessingEstimator
    pdm::T1
    dn::T2
    dt::T3
    alg::T4
    order::T5
    function DenoiseDetoneAlgMatrixProcessing(pdm::Option{<:Posdef}, dn::Option{<:Denoise},
                                              dt::Option{<:Detone},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                              order::AbstractMatrixProcessingOrder = DenoiseDetoneAlg())
        return new{typeof(pdm), typeof(dn), typeof(dt), typeof(alg), typeof(order)}(pdm, dn,
                                                                                    dt, alg,
                                                                                    order)
    end
end
function DenoiseDetoneAlgMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          dn::Option{<:Denoise} = nothing,
                                          dt::Option{<:Detone} = nothing,
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                          order::AbstractMatrixProcessingOrder = DenoiseDetoneAlg())
    return DenoiseDetoneAlgMatrixProcessing(pdm, dn, dt, alg, order)
end
"""
    matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...;
                       kwargs...)
    matrix_processing!(::Nothing, sigma::MatNum, args...; kwargs...)

No-op fallback for in-place processing of a covariance or correlation matrix.

# Arguments

  - $(glossary[:omp])
  - $(glossary[:sigrho])
  - $(glossary[:X])
  - `args...`: Additional positional arguments passed to custom algorithms.
  - `kwargs...`: Additional keyword arguments passed to custom algorithms.

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is modified in-place.

# Related

  - [`matrix_processing`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
  - [`posdef!`](@ref)
  - [`denoise!`](@ref)
  - [`detone!`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`MatNum`](@ref)
"""
function matrix_processing!(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
"""
    matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing, sigma::MatNum, X::MatNum, args...;
                       kwargs...)

In-place matrix processing pipeline using the `DenoiseDetoneAlg` order.

This method applies a sequence of matrix processing steps to the input covariance or correlation matrix `sigma`, modifying it in-place. The steps are performed in the following order: positive definiteness enforcement, denoising, detoning, and an optional custom algorithm. The order is determined by the `DenoiseDetoneAlg` type.

# Arguments

  - $(glossary[:omp])
  - $(glossary[:sigrho])
  - $(glossary[:X])
  - `args...`: Additional positional arguments passed to custom algorithms.
  - `kwargs...`: Additional keyword arguments passed to custom algorithms.

# Returns

  - `sigma::MatNum`: The input matrix `sigma` is modified in-place.

# Details

  - Applies positive definiteness enforcement using `mp.pdm`.
  - Applies denoising using `mp.dn` and the ratio `T / N` from `X`.
  - Applies detoning using `mp.dt`.
  - Applies an optional custom matrix processing algorithm using `mp.alg`.
  - The order of operations depends on the specific type of `mp.order`.

# Examples

```jldoctest
julia> using StableRNGs, Statistics

julia> rng = StableRNG(123456789);

julia> X = rand(rng, 10, 5);

julia> sigma = cov(X)
5×5 Matrix{Float64}:
  0.132026     0.0022567   0.0198243    0.00359832  -0.00743829
  0.0022567    0.0514194  -0.0131242    0.004123     0.0312379
  0.0198243   -0.0131242   0.0843837   -0.0325342   -0.00609624
  0.00359832   0.004123   -0.0325342    0.0424332    0.0152574
 -0.00743829   0.0312379  -0.00609624   0.0152574    0.0926441

julia> matrix_processing!(DenoiseDetoneAlgMatrixProcessing(; dn = Denoise()), sigma, X)
5×5 Matrix{Float64}:
 0.132026  0.0        0.0        0.0        0.0
 0.0       0.0514194  0.0        0.0        0.0
 0.0       0.0        0.0843837  0.0        0.0
 0.0       0.0        0.0        0.0424332  0.0
 0.0       0.0        0.0        0.0        0.0926441

julia> sigma = cov(X)
5×5 Matrix{Float64}:
  0.132026     0.0022567   0.0198243    0.00359832  -0.00743829
  0.0022567    0.0514194  -0.0131242    0.004123     0.0312379
  0.0198243   -0.0131242   0.0843837   -0.0325342   -0.00609624
  0.00359832   0.004123   -0.0325342    0.0424332    0.0152574
 -0.00743829   0.0312379  -0.00609624   0.0152574    0.0926441

julia> matrix_processing!(DenoiseDetoneAlgMatrixProcessing(; dt = Detone()), sigma, X)
5×5 Matrix{Float64}:
 0.132026    0.0124802   0.0117303    0.0176194    0.0042142
 0.0124802   0.0514194   0.0273105   -0.0290864    0.0088165
 0.0117303   0.0273105   0.0843837   -0.00279296   0.0619156
 0.0176194  -0.0290864  -0.00279296   0.0424332   -0.0242252
 0.0042142   0.0088165   0.0619156   -0.0242252    0.0926441
```

# Related

  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
  - [`AbstractMatrixProcessingOrder`](@ref)
  - [`matrix_processing`](@ref)
  - [`posdef!`](@ref)
  - [`denoise!`](@ref)
  - [`detone!`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
  - [`MatNum`](@ref)

# References

  - [mlp1](@cite) M. M. De Prado. *Machine learning for asset managers* (Cambridge University Press, 2020). Chapter 2.
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices*. Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DenoiseDetoneAlg},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.dn, sigma, T / N)
    detone!(mp.dt, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    return sigma
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DenoiseAlgDetone},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.dn, sigma, T / N)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    detone!(mp.dt, sigma)
    return sigma
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DetoneDenoiseAlg},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.dt, sigma)
    denoise!(mp.dn, sigma, T / N)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    return sigma
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DetoneAlgDenoise},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.dt, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    denoise!(mp.dn, sigma, T / N)
    return sigma
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:AlgDenoiseDetone},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    denoise!(mp.dn, sigma, T / N)
    detone!(mp.dt, sigma)
    return sigma
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:AlgDetoneDenoise},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    detone!(mp.dt, sigma)
    denoise!(mp.dn, sigma, T / N)
    return sigma
end
"""
    matrix_processing(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...;
                      kwargs...)
    matrix_processing(::Nothing, args...; kwargs...)

Out-of-place version of [`matrix_processing!`](@ref).

# Related

  - [`matrix_processing!`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
  - [`posdef!`](@ref)
  - [`denoise!`](@ref)
  - [`detone!`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`MatNum`](@ref)
"""
function matrix_processing(::Nothing, sigma::MatNum, args...; kwargs...)
    return sigma
end
function matrix_processing(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum,
                           args...; kwargs...)
    sigma = copy(sigma)
    matrix_processing!(mp, sigma, X, args...; kwargs...)
    return sigma
end

export DenoiseDetoneAlgMatrixProcessing, DenoiseDetoneAlg, DenoiseAlgDetone,
       DetoneDenoiseAlg, DetoneAlgDenoise, AlgDenoiseDetone, AlgDetoneDenoise,
       matrix_processing, matrix_processing!
