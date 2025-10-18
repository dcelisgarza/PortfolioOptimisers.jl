"""
    abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end

Abstract supertype for all matrix processing estimator types in PortfolioOptimisers.jl.

All concrete types that implement matrix processing routines—such as covariance matrix cleaning, denoising, or detoning—should subtype `AbstractMatrixProcessingEstimator`. This enables a consistent interface for matrix processing estimators throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
"""
abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
"""
    abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end

Abstract supertype for all matrix processing algorithm types in PortfolioOptimisers.jl.

All concrete types that implement a specific matrix processing algorithm (e.g., custom cleaning or transformation routines) should subtype `AbstractMatrixProcessingAlgorithm`. This enables flexible extension and dispatch of matrix processing routines.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
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
  - [`DefaultMatrixProcessing`](@ref)
"""
function matrix_processing_algorithm!(::Nothing, args...; kwargs...)
    return nothing
end
"""
    matrix_processing_algorithm(::Nothing, args...; kwargs...)

Same as [`matrix_processing_algorithm!`](@ref), but meant for returning a new matrix instead of modifying it in-place.

# Related

  - [`matrix_processing_algorithm!`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
"""
function matrix_processing_algorithm(::Nothing, args...; kwargs...)
    return nothing
end
"""
    struct DefaultMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
        pdm::T1
        denoise::T2
        detone::T3
        alg::T4
    end

A flexible container type for configuring and applying matrix processing routines in PortfolioOptimisers.jl.

`DefaultMatrixProcessing` encapsulates all steps required for processing covariance or correlation matrices, including positive definiteness enforcement, denoising, detoning, and optional custom matrix processing algorithms. It is the standard estimator type for matrix processing pipelines and supports a variety of estimator and algorithm types.

# Fields

  - `pdm`: Positive definite matrix estimator (see [`Posdef`](@ref)), or `nothing` to skip.
  - `denoise`: Denoising estimator (see [`Denoise`](@ref)), or `nothing` to skip.
  - `detone`: Detoning estimator (see [`Detone`](@ref)), or `nothing` to skip.
  - `alg`: Optional custom matrix processing algorithm, or `nothing` to skip.

# Constructor

    DefaultMatrixProcessing(; pdm::Union{Nothing, <:Posdef} = Posdef(),
                            denoise::Union{Nothing, <:Denoise} = nothing,
                            detone::Union{Nothing, <:Detone} = nothing,
                            alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm} = nothing)

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> mp = DefaultMatrixProcessing()
DefaultMatrixProcessing
      pdm | Posdef
          |   alg | UnionAll: NearestCorrelationMatrix.Newton
  denoise | nothing
   detone | nothing
      alg | nothing

julia> mp = DefaultMatrixProcessing(; denoise = Denoise(), detone = Detone(; n = 2))
DefaultMatrixProcessing
      pdm | Posdef
          |   alg | UnionAll: NearestCorrelationMatrix.Newton
  denoise | Denoise
          |      alg | ShrunkDenoise
          |          |   alpha | Float64: 0.0
          |     args | Tuple{}: ()
          |   kwargs | @NamedTuple{}: NamedTuple()
          |   kernel | typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
          |        m | Int64: 10
          |        n | Int64: 1000
   detone | Detone
          |   n | Int64: 2
      alg | nothing
```

# Related

  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
"""
struct DefaultMatrixProcessing{T1, T2, T3, T4} <: AbstractMatrixProcessingEstimator
    pdm::T1
    denoise::T2
    detone::T3
    alg::T4
    function DefaultMatrixProcessing(pdm::Union{Nothing, <:Posdef},
                                     denoise::Union{Nothing, <:Denoise},
                                     detone::Union{Nothing, <:Detone},
                                     alg::Union{Nothing,
                                                <:AbstractMatrixProcessingAlgorithm})
        return new{typeof(pdm), typeof(denoise), typeof(detone), typeof(alg)}(pdm, denoise,
                                                                              detone, alg)
    end
end
function DefaultMatrixProcessing(; pdm::Union{Nothing, <:Posdef} = Posdef(),
                                 denoise::Union{Nothing, <:Denoise} = nothing,
                                 detone::Union{Nothing, <:Detone} = nothing,
                                 alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm} = nothing)
    return DefaultMatrixProcessing(pdm, denoise, detone, alg)
end
"""
    matrix_processing!(mp::AbstractMatrixProcessingEstimator, sigma::AbstractMatrix,
                       X::AbstractMatrix, args...; kwargs...)
    matrix_processing!(::Nothing, args...; kwargs...)

In-place processing of a covariance or correlation matrix.

The processing pipeline consists of:

 1. Positive definiteness enforcement via [`posdef!`](@ref).
 2. Denoising via [`denoise!`](@ref).
 3. Detoning via [`detone!`](@ref).
 4. Optional custom matrix processing algorithm via [`matrix_processing_algorithm!`](@ref).

# Arguments

  - `mp::AbstractMatrixProcessingEstimator`: Matrix processing estimator specifying the pipeline.

      + `mp::DefaultMatrixProcessing`: The specified matrix processing steps are applied to `sigma` using the provided data matrix `X`.
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

julia> matrix_processing!(DefaultMatrixProcessing(; denoise = Denoise()), sigma, X)

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

julia> matrix_processing!(DefaultMatrixProcessing(; detone = Detone()), sigma, X)

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
  - [`DefaultMatrixProcessing`](@ref)
  - [`posdef!`](@ref)
  - [`denoise!`](@ref)
  - [`detone!`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
"""
function matrix_processing!(::Nothing, args...; kwargs...)
    return nothing
end
function matrix_processing!(mp::DefaultMatrixProcessing, sigma::AbstractMatrix,
                            X::AbstractMatrix, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.denoise, sigma, T / N, mp.pdm)
    detone!(mp.detone, sigma, mp.pdm)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    return nothing
end
"""
    matrix_processing(mp::AbstractMatrixProcessingEstimator, sigma::AbstractMatrix,
                      X::AbstractMatrix, args...; kwargs...)
    matrix_processing(::Nothing, args...; kwargs...)

Out-of-place version of [`matrix_processing!`](@ref).

# Related

  - [`matrix_processing!`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
  - [`posdef!`](@ref)
  - [`denoise!`](@ref)
  - [`detone!`](@ref)
  - [`matrix_processing_algorithm!`](@ref)
"""
function matrix_processing(::Nothing, args...; kwargs...)
    return nothing
end
function matrix_processing(mp::DefaultMatrixProcessing, sigma::AbstractMatrix,
                           X::AbstractMatrix, args...; kwargs...)
    sigma = copy(sigma)
    matrix_processing!(mp, sigma, X, args...; kwargs...)
    return sigma
end

export DefaultMatrixProcessing, matrix_processing, matrix_processing!
