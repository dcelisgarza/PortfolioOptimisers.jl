"""
    abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end

Abstract supertype for all matrix processing estimator types in PortfolioOptimisers.jl.

All concrete types that implement matrix processing routines—such as covariance matrix cleaning, denoising, or detoning—should subtype `AbstractMatrixProcessingEstimator`. This enables a consistent interface for matrix processing estimators throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
"""
    abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end

Abstract supertype for all matrix processing algorithm types in PortfolioOptimisers.jl.

All concrete types that implement a specific matrix processing algorithm (e.g., custom cleaning or transformation routines) should subtype `AbstractMatrixProcessingAlgorithm`. This enables flexible extension and dispatch of matrix processing routines.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`DenoiseDetoneAlgMatrixProcessing`](@ref)
"""
abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end
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
    struct DenoiseDetoneAlgMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
        pdm::T1
        denoise::T2
        detone::T3
        alg::T4
    end

A flexible container type for configuring and applying matrix processing routines in PortfolioOptimisers.jl.

`DenoiseDetoneAlgMatrixProcessing` encapsulates all steps required for processing covariance or correlation matrices, including positive definiteness enforcement, denoising, detoning, and optional custom matrix processing algorithms. It is the standard estimator type for matrix processing pipelines and supports a variety of estimator and algorithm types.

# Fields

  - `pdm`: Positive definite matrix estimator (see [`Posdef`](@ref)), or `nothing` to skip.
  - `denoise`: Denoising estimator (see [`Denoise`](@ref)), or `nothing` to skip.
  - `detone`: Detoning estimator (see [`Detone`](@ref)), or `nothing` to skip.
  - `alg`: Optional custom matrix processing algorithm, or `nothing` to skip.

# Constructor

    DenoiseDetoneAlgMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                            denoise::Option{<:Denoise} = nothing,
                            detone::Option{<:Detone} = nothing,
                            alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing)

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
      alg ┴ nothing

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
      alg ┴ nothing
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
struct DenoiseDetoneAlgMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
    pdm::T1
    denoise::T2
    detone::T3
    alg::T4
    function DenoiseDetoneAlgMatrixProcessing(pdm::Option{<:Posdef},
                                              denoise::Option{<:Denoise},
                                              detone::Option{<:Detone},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm})
        return new{typeof(pdm), typeof(denoise), typeof(detone), typeof(alg)}(pdm, denoise,
                                                                              detone, alg)
    end
end
function DenoiseDetoneAlgMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          denoise::Option{<:Denoise} = nothing,
                                          detone::Option{<:Detone} = nothing,
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing)
    return DenoiseDetoneAlgMatrixProcessing(pdm, denoise, detone, alg)
end
"""
    matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::MatNum, X::MatNum, args...;
                       kwargs...)
    matrix_processing!(::Nothing, args...; kwargs...)

In-place processing of a covariance or correlation matrix.

The processing pipeline consists of:

 1. Positive definiteness enforcement via [`posdef!`](@ref).
 2. Denoising via [`denoise!`](@ref).
 3. Detoning via [`detone!`](@ref).
 4. Optional custom matrix processing algorithm via [`matrix_processing_algorithm!`](@ref).

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
function matrix_processing!(mp::DenoiseDetoneAlgMatrixProcessing, sigma::MatNum, X::MatNum,
                            args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    detone!(mp.detone, sigma, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
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
function matrix_processing(mp::DenoiseDetoneAlgMatrixProcessing, sigma::MatNum, X::MatNum,
                           args...; kwargs...)
    sigma = copy(sigma)
    matrix_processing!(mp, sigma, X, args...; kwargs...)
    return sigma
end

#########
struct DenoiseAlgDetoneMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
    pdm::T1
    denoise::T2
    alg::T3
    detone::T4
    function DenoiseAlgDetoneMatrixProcessing(pdm::Option{<:Posdef},
                                              denoise::Option{<:Denoise},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                              detone::Option{<:Detone})
        return new{typeof(pdm), typeof(denoise), typeof(alg), typeof(detone)}(pdm, denoise,
                                                                              alg, detone)
    end
end
function DenoiseAlgDetoneMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          denoise::Option{<:Denoise} = nothing,
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                          detone::Option{<:Detone} = nothing)
    return DenoiseAlgDetoneMatrixProcessing(pdm, denoise, alg, detone)
end
function matrix_processing!(mp::DenoiseAlgDetoneMatrixProcessing, sigma::MatNum, X::MatNum,
                            args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    detone!(mp.detone, sigma, mp.pdm)
    return nothing
end
struct DetoneDenoiseAlgMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
    pdm::T1
    detone::T2
    denoise::T3
    alg::T4
    function DetoneDenoiseAlgMatrixProcessing(pdm::Option{<:Posdef},
                                              detone::Option{<:Detone},
                                              denoise::Option{<:Denoise},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm})
        return new{typeof(pdm), typeof(detone), typeof(denoise), typeof(alg)}(pdm, detone,
                                                                              denoise, alg)
    end
end
function DetoneDenoiseAlgMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          detone::Option{<:Detone} = nothing,
                                          denoise::Option{<:Denoise} = nothing,
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing)
    return DetoneDenoiseAlgMatrixProcessing(pdm, detone, denoise, alg)
end
function matrix_processing!(mp::DetoneDenoiseAlgMatrixProcessing, sigma::MatNum, X::MatNum,
                            args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.detone, sigma, mp.pdm)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    return nothing
end
struct DetoneAlgDenoiseMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
    pdm::T1
    detone::T2
    alg::T3
    denoise::T4
    function DetoneAlgDenoiseMatrixProcessing(pdm::Option{<:Posdef},
                                              detone::Option{<:Detone},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                              denoise::Option{<:Denoise})
        return new{typeof(pdm), typeof(detone), typeof(alg), typeof(denoise)}(pdm, detone,
                                                                              alg, denoise)
    end
end
function DetoneAlgDenoiseMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          detone::Option{<:Detone} = nothing,
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                          denoise::Option{<:Denoise} = nothing)
    return DetoneAlgDenoiseMatrixProcessing(pdm, detone, alg, denoise)
end
function matrix_processing!(mp::DetoneAlgDenoiseMatrixProcessing, sigma::MatNum, X::MatNum,
                            args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    detone!(mp.detone, sigma, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    return nothing
end
struct AlgDenoiseDetoneMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
    pdm::T1
    alg::T2
    denoise::T3
    detone::T4
    function AlgDenoiseDetoneMatrixProcessing(pdm::Option{<:Posdef},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                              denoise::Option{<:Denoise},
                                              detone::Option{<:Detone})
        return new{typeof(pdm), typeof(alg), typeof(denoise), typeof(detone)}(pdm, alg,
                                                                              denoise,
                                                                              detone)
    end
end
function AlgDenoiseDetoneMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                          denoise::Option{<:Denoise} = nothing,
                                          detone::Option{<:Detone} = nothing,)
    return AlgDenoiseDetoneMatrixProcessing(pdm, alg, denoise, detone)
end
function matrix_processing!(mp::AlgDenoiseDetoneMatrixProcessing, sigma::MatNum, X::MatNum,
                            args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    detone!(mp.detone, sigma, mp.pdm)
    return nothing
end
struct AlgDetoneDenoiseMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
    pdm::T1
    alg::T2
    detone::T3
    denoise::T4
    function AlgDetoneDenoiseMatrixProcessing(pdm::Option{<:Posdef},
                                              alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                              detone::Option{<:Detone},
                                              denoise::Option{<:Denoise})
        return new{typeof(pdm), typeof(alg), typeof(detone), typeof(denoise)}(pdm, alg,
                                                                              detone,
                                                                              denoise)
    end
end
function AlgDetoneDenoiseMatrixProcessing(; pdm::Option{<:Posdef} = Posdef(),
                                          alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                                          detone::Option{<:Detone} = nothing,
                                          denoise::Option{<:Denoise} = nothing)
    return AlgDetoneDenoiseMatrixProcessing(pdm, alg, detone, denoise)
end
function matrix_processing!(mp::AlgDetoneDenoiseMatrixProcessing, sigma::MatNum, X::MatNum,
                            args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    detone!(mp.detone, sigma, mp.pdm)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    return nothing
end
#########

export DenoiseDetoneAlgMatrixProcessing, matrix_processing, matrix_processing!
