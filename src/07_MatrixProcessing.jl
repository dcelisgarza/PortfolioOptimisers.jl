"""
    abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end

Abstract supertype for all matrix processing estimator types in `PortfolioOptimisers.jl`.

All concrete types that implement matrix processing routines—such as covariance matrix cleaning, denoising, or detoning—should subtype `AbstractMatrixProcessingEstimator`. This enables a consistent interface for matrix processing estimators throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
"""
    abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end

Abstract supertype for all matrix processing algorithm types in `PortfolioOptimisers.jl`.

All concrete types that implement a specific matrix processing algorithm (e.g., custom cleaning or transformation routines) should subtype `AbstractMatrixProcessingAlgorithm`. This enables flexible extension and dispatch of matrix processing routines.

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

Matrix processing order: Denoising → Detoning → Algorithm.

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

Matrix processing order: Denoising → Algorithm → Detoning.

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

Matrix processing order: Detoning → Denoising → Algorithm.

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

Matrix processing order: Detoning → Algorithm → Denoising.

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

Matrix processing order: Algorithm → Denoising → Detoning.

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

Matrix processing order: Algorithm → Detoning → Denoising.

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
    matrix_processing_algorithm!(::Nothing, args...; kwargs...)

No-op fallback for matrix processing algorithm routines.

These methods are called internally when no matrix processing algorithm is specified (i.e., when the algorithm argument is `nothing`). They perform no operation and return `nothing`, ensuring that the matrix processing pipeline can safely skip optional algorithmic steps.

# Arguments

  - `::Nothing`: Indicates that no algorithm is provided.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`matrix_processing_algorithm`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
function matrix_processing_algorithm!(::Nothing, args...; kwargs...)
    return nothing
end
"""
    matrix_processing_algorithm(::Nothing, args...; kwargs...)

Same as [`matrix_processing_algorithm!`](@ref), but meant for returning a new matrix instead of modifying it in-place.

# Related

  - [`matrix_processing_algorithm!`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
function matrix_processing_algorithm(::Nothing, args...; kwargs...)
    return nothing
end
"""
    struct DenoiseDetoneAlgMatrixProcessing{T1, T2, T3, T4, T5} <: AbstractMatrixProcessingEstimator
        pdm::T1
        denoise::T2
        detone::T3
        alg::T4
        order::T5
    end

A flexible container type for configuring and applying matrix processing routines in `PortfolioOptimisers.jl`.

`DenoiseDetoneAlgMatrixProcessing` encapsulates all steps required for processing covariance or correlation matrices, including positive definiteness enforcement, denoising, detoning, and optional custom matrix processing algorithms. It is the standard estimator type for matrix processing pipelines and supports a variety of estimator and algorithm types.

# Fields

  - `pdm`: Positive definite matrix estimator (see [`Posdef`](@ref)), or `nothing` to skip.
  - `denoise`: Denoising estimator (see [`Denoise`](@ref)), or `nothing` to skip.
  - `detone`: Detoning estimator (see [`Detone`](@ref)), or `nothing` to skip.
  - `alg`: Optional custom matrix processing algorithm, or `nothing` to skip.
  - `order`: Specifies the order in which denoising, detoning, and algorithmic steps are applied.

# Constructor

    DenoiseDetoneAlgMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                            denoise::Option{<:Denoise} = nothing,
                            detone::Option{<:Detone} = nothing,
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
  denoise ┼ nothing
   detone ┼ nothing
      alg ┼ nothing
    order ┴ DenoiseDetoneAlg()

julia> DenoiseDetoneAlgMatrixProcessing(; denoise = Denoise(), detone = Detone(; n = 2))
DenoiseDetoneAlgMatrixProcessing
      pdm ┼ Posdef
          │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │   kwargs ┴ @NamedTuple{}: NamedTuple()
  denoise ┼ Denoise
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
   detone ┼ Detone
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
  - [mpdist](@cite) V. A. Marčenko and L. A. Pastur. *Distribution of eigenvalues for some sets of random matrices.* Mathematics of the USSR-Sbornik 1, 457 (1967).
"""
struct DenoiseDetoneAlgMatrixProcessing{T1, T2, T3, T4, T5} <:
       AbstractMatrixProcessingEstimator
    pdm::T1
    denoise::T2
    detone::T3
    alg::T4
    order::T5
    function DenoiseDetoneAlgMatrixProcessing(pdm::Option{<:Posdef},
                                              denoise::Option{<:Denoise},
                                              detone::Option{<:Detone},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                              order::AbstractMatrixProcessingOrder = DenoiseDetoneAlg())
        return new{typeof(pdm), typeof(denoise), typeof(detone), typeof(alg),
                   typeof(order)}(pdm, denoise, detone, alg, order)
    end
end
function DenoiseDetoneAlgMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          denoise::Option{<:Denoise} = nothing,
                                          detone::Option{<:Detone} = nothing,
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                          order::AbstractMatrixProcessingOrder = DenoiseDetoneAlg())
    return DenoiseDetoneAlgMatrixProcessing(pdm, denoise, detone, alg, order)
end
"""
    matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...;
                       kwargs...)
    matrix_processing!(::Nothing, args...; kwargs...)

In-place processing of a covariance or correlation matrix.

# Arguments

  - `mp::AbstractMatrixProcessingEstimator`: Matrix processing estimator specifying the pipeline.
  - `sigma`: Covariance or correlation matrix to be processed (modified in-place).
  - `X`: Data matrix (observations × assets) used for denoising and detoning.
  - `args...`: Additional positional arguments passed to custom algorithms.
  - `kwargs...`: Additional keyword arguments passed to custom algorithms.

# Returns

  - `nothing`. The input matrix `sigma` is modified in-place.

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

julia> matrix_processing!(DenoiseDetoneAlgMatrixProcessing(; denoise = Denoise()), sigma, X)

julia> sigma
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

julia> matrix_processing!(DenoiseDetoneAlgMatrixProcessing(; detone = Detone()), sigma, X)

julia> sigma
5×5 Matrix{Float64}:
 0.132026    0.0124802   0.0117303    0.0176194    0.0042142
 0.0124802   0.0514194   0.0273105   -0.0290864    0.0088165
 0.0117303   0.0273105   0.0843837   -0.00279296   0.0619156
 0.0176194  -0.0290864  -0.00279296   0.0424332   -0.0242252
 0.0042142   0.0088165   0.0619156   -0.0242252    0.0926441
```

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
function matrix_processing!(::Nothing, args...; kwargs...)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DenoiseDetoneAlg},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.denoise, sigma, T / N)
    detone!(mp.detone, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DenoiseAlgDetone},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.denoise, sigma, T / N)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    detone!(mp.detone, sigma)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DetoneDenoiseAlg},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.detone, sigma)
    denoise!(mp.denoise, sigma, T / N)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DetoneAlgDenoise},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.detone, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    denoise!(mp.denoise, sigma, T / N)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:AlgDenoiseDetone},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    denoise!(mp.denoise, sigma, T / N)
    detone!(mp.detone, sigma)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:AlgDetoneDenoise},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, sigma, X; kwargs...)
    detone!(mp.detone, sigma)
    denoise!(mp.denoise, sigma, T / N)
    return nothing
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
function matrix_processing(::Nothing, args...; kwargs...)
    return nothing
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
