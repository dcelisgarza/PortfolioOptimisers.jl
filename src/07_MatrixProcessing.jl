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
abstract type AbstractMatrixProcessingOrder <: AbstractAlgorithm end
"""
Possible values include:

        + `DenoiseDetoneAlg`: Denoising → Detoning → Algorithm
        + `DenoiseAlgDetone`: Denoising → Algorithm → Detoning
        + `DetoneDenoiseAlg`: Detoning → Denoising → Algorithm
        + `DetoneAlgDenoise`: Detoning → Algorithm → Denoising
        + `AlgDenoiseDetone`: Algorithm → Denoising → Detoning
        + `AlgDetoneDenoise`: Algorithm → Detoning → Denoising
"""
struct DenoiseDetoneAlg <: AbstractMatrixProcessingOrder end
struct DenoiseAlgDetone <: AbstractMatrixProcessingOrder end
struct DetoneDenoiseAlg <: AbstractMatrixProcessingOrder end
struct DetoneAlgDenoise <: AbstractMatrixProcessingOrder end
struct AlgDenoiseDetone <: AbstractMatrixProcessingOrder end
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
          │        n ┴ Int64: 1000
   detone ┼ Detone
          │   n ┴ Int64: 2
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

      + `mp::DenoiseDetoneAlgMatrixProcessing`: The specified matrix processing steps are applied to `sigma` using the provided data matrix `X`.
      + `mp::Nothing`: No-op.

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
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    detone!(mp.detone, sigma, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DenoiseAlgDetone},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    detone!(mp.detone, sigma, mp.pdm)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DetoneDenoiseAlg},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.detone, sigma, mp.pdm)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:DetoneAlgDenoise},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.detone, sigma, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:AlgDenoiseDetone},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    detone!(mp.detone, sigma, mp.pdm)
    return nothing
end
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing{<:Any, <:Any, <:Any, <:Any,
                                                                 <:AlgDetoneDenoise},
                            sigma::MatNum, X::MatNum, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    detone!(mp.detone, sigma, mp.pdm)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
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
