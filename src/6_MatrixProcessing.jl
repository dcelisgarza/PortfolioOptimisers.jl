"""
    AbstractMatrixProcessingEstimator <: AbstractEstimator

Abstract supertype for all matrix processing estimator types in PortfolioOptimisers.jl.

All concrete types that implement matrix processing routines—such as covariance matrix cleaning, denoising, or detoning—should subtype `AbstractMatrixProcessingEstimator`. This enables a consistent interface for matrix processing estimators throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
  - [`NonPositiveDefiniteMatrixProcessing`](@ref)
"""
abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end

"""
    AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm

Abstract supertype for all matrix processing algorithm types in PortfolioOptimisers.jl.

All concrete types that implement a specific matrix processing algorithm (e.g., custom cleaning or transformation routines) should subtype `AbstractMatrixProcessingAlgorithm`. This enables flexible extension and dispatch of matrix processing routines.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
  - [`NonPositiveDefiniteMatrixProcessing`](@ref)
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
  - [`NonPositiveDefiniteMatrixProcessing`](@ref)
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
  - [`NonPositiveDefiniteMatrixProcessing`](@ref)
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

  - `pdm::Union{Nothing, <:PosdefEstimator}`: Positive definite matrix estimator (see [`PosdefEstimator`](@ref)), or `nothing` to skip.
  - `denoise::Union{Nothing, <:Denoise}`: Denoising estimator (see [`Denoise`](@ref)), or `nothing` to skip.
  - `detone::Union{Nothing, <:Detone}`: Detoning estimator (see [`Detone`](@ref)), or `nothing` to skip.
  - `alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm}`: Optional custom matrix processing algorithm, or `nothing` to skip.

# Constructor

    DefaultMatrixProcessing(; pdm = PosdefEstimator(), denoise = nothing, detone = nothing, alg = nothing)

Keyword arguments correspond to the fields above. The constructor infers types and sets defaults for robust matrix processing.

# Related

  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
  - [`NonPositiveDefiniteMatrixProcessing`](@ref)
"""
struct DefaultMatrixProcessing{T1 <: Union{Nothing, <:PosdefEstimator},
                               T2 <: Union{Nothing, <:Denoise},
                               T3 <: Union{Nothing, <:Detone},
                               T4 <: Union{Nothing, <:AbstractMatrixProcessingAlgorithm}} <:
       AbstractMatrixProcessingEstimator
    pdm::T1
    denoise::T2
    detone::T3
    alg::T4
end
"""
    DefaultMatrixProcessing(; pdm = PosdefEstimator(), denoise = nothing, detone = nothing, alg = nothing)

Construct a [`DefaultMatrixProcessing`](@ref) object, configuring all steps for matrix processing in PortfolioOptimisers.jl.

# Arguments

  - `pdm::Union{Nothing, <:PosdefEstimator}`: Positive definite matrix estimator.
  - `denoise::Union{Nothing, <:Denoise}`: Denoising estimator.
  - `detone::Union{Nothing, <:Detone}`: Detoning estimator.
  - `alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm}`: Optional custom matrix processing algorithm.

# Returns

  - `DefaultMatrixProcessing`: A configured matrix processing estimator.

# Examples

```jldoctest
julia> mp = DefaultMatrixProcessing()
DefaultMatrixProcessing
      pdm | PosdefEstimator
          |   alg | UnionAll: NearestCorrelationMatrix.Newton
  denoise | nothing
   detone | nothing
      alg | nothing

julia> mp = DefaultMatrixProcessing(; denoise = Denoise(), detone = Detone(; n = 2))
DefaultMatrixProcessing
      pdm | PosdefEstimator
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

  - [`DefaultMatrixProcessing`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
"""
function DefaultMatrixProcessing(;
                                 pdm::Union{Nothing, <:PosdefEstimator} = PosdefEstimator(),
                                 denoise::Union{Nothing, <:Denoise} = nothing,
                                 detone::Union{Nothing, <:Detone} = nothing,
                                 alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm} = nothing)
    return DefaultMatrixProcessing{typeof(pdm), typeof(denoise), typeof(detone),
                                   typeof(alg)}(pdm, denoise, detone, alg)
end

"""
    struct NonPositiveDefiniteMatrixProcessing{T1, T2, T3} <: AbstractMatrixProcessingEstimator
        denoise::T1
        detone::T2
        alg::T3
    end

A container type for matrix processing pipelines that do **not** enforce positive definiteness in PortfolioOptimisers.jl.

`NonPositiveDefiniteMatrixProcessing` is intended for workflows where positive definiteness is not required or is handled externally. It supports denoising, detoning, and optional custom matrix processing algorithms, but skips positive definite projection.

# Fields

  - `denoise::Union{Nothing, <:Denoise}`: Denoising estimator (see [`Denoise`](@ref)), or `nothing` to skip.
  - `detone::Union{Nothing, <:Detone}`: Detoning estimator (see [`Detone`](@ref)), or `nothing` to skip.
  - `alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm}`: Optional custom matrix processing algorithm, or `nothing` to skip.

# Constructor

    NonPositiveDefiniteMatrixProcessing(; denoise = nothing, detone = nothing, alg = nothing)

Keyword arguments correspond to the fields above. The constructor infers types and sets defaults for robust matrix processing without positive definite enforcement.

# Related

  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
"""
struct NonPositiveDefiniteMatrixProcessing{T1 <: Union{Nothing, <:Denoise},
                                           T2 <: Union{Nothing, <:Detone},
                                           T3 <: Union{Nothing,
                                                       <:AbstractMatrixProcessingAlgorithm}} <:
       AbstractMatrixProcessingEstimator
    denoise::T1
    detone::T2
    alg::T3
end
"""
    NonPositiveDefiniteMatrixProcessing(; denoise = nothing, detone = nothing, alg = nothing)

Construct a [`NonPositiveDefiniteMatrixProcessing`](@ref) object, configuring matrix processing steps without positive definite enforcement.

# Arguments

  - `denoise::Union{Nothing, <:Denoise} = nothing`: Denoising estimator.
  - `detone::Union{Nothing, <:Detone} = nothing`: Detoning estimator.
  - `alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm} = nothing`: Optional custom matrix processing algorithm.

# Returns

  - `NonPositiveDefiniteMatrixProcessing`: A configured matrix processing estimator.

# Examples

```jldoctest
julia> mp = NonPositiveDefiniteMatrixProcessing()
NonPositiveDefiniteMatrixProcessing
  denoise | nothing
   detone | nothing
      alg | nothing

julia> mp = NonPositiveDefiniteMatrixProcessing(; denoise = Denoise(), detone = Detone(; n = 2))
NonPositiveDefiniteMatrixProcessing
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

  - [`NonPositiveDefiniteMatrixProcessing`](@ref)
  - [`matrix_processing!`](@ref)
  - [`matrix_processing`](@ref)
"""
function NonPositiveDefiniteMatrixProcessing(; denoise::Union{Nothing, <:Denoise} = nothing,
                                             detone::Union{Nothing, <:Detone} = nothing,
                                             alg::Union{Nothing,
                                                        <:AbstractMatrixProcessingAlgorithm} = nothing)
    return NonPositiveDefiniteMatrixProcessing{typeof(denoise), typeof(detone),
                                               typeof(alg)}(denoise, detone, alg)
end

"""
    matrix_processing!(mp::DefaultMatrixProcessing, sigma::AbstractMatrix, X::AbstractMatrix, args...; kwargs...)
    matrix_processing!(mp::NonPosdefMatrixProcessing, sigma::AbstractMatrix, X::AbstractMatrix, args...; kwargs...)
    matrix_processing!(::Nothing, args...; kwargs...)

In-place processing of a covariance or correlation matrix.

  - If `mp` is `nothing`, this is a no-op and returns `nothing`.
  - If `mp` is a [`DefaultMatrixProcessing`](@ref) object, the specified matrix processing steps are applied to `sigma` in-place, using the provided data matrix `X`.
  - If `mp` is a [`NonPositiveDefiniteMatrixProcessing`](@ref) object, the specified matrix processing steps **without enforcing positive definiteness** are applied to `sigma` in-place, using the provided data matrix `X`.

The processing pipeline consists of:

 1. Positive definiteness enforcement via [`posdef!`](@ref) (if `mp.pdm` is [`DefaultMatrixProcessing`](@ref)).
 2. Denoising via [`denoise!`](@ref) (if `mp.denoise` is not `nothing`).
 3. Detoning via [`detone!`](@ref) (if `mp.detone` is not `nothing`).
 4. Optional custom matrix processing algorithm via [`matrix_processing_algorithm!`](@ref) (if `mp.alg` is not `nothing`).

# Arguments

  - `mp::Union{Nothing, <:AbstractMatrixProcessingEstimator}`: Matrix processing estimator specifying the pipeline.
  - `sigma::AbstractMatrix`: Covariance or correlation matrix to be processed (modified in-place).
  - `X::AbstractMatrix`: Data matrix (observations × assets) used for denoising and detoning.
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
function matrix_processing!(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
                            X::AbstractMatrix, args...; kwargs...)
    T, N = size(X)
    posdef!(nothing, sigma)
    denoise!(mp.denoise, sigma, T / N, nothing)
    detone!(mp.detone, sigma, nothing)
    matrix_processing_algorithm!(mp.alg, nothing, sigma, X; kwargs...)
    return nothing
end
"""
    matrix_processing(mp::DefaultMatrixProcessing, sigma::AbstractMatrix, X::AbstractMatrix, args...; kwargs...)
    matrix_processing(mp::NonPosdefMatrixProcessing, sigma::AbstractMatrix, X::AbstractMatrix, args...; kwargs...)
    matrix_processing(::Nothing, args...; kwargs...)

Same as [`matrix_processing!`](@ref), but returns a new matrix instead of modifying `sigma` in-place.

  - If `mp` is `nothing`, this is a no-op and returns `nothing`.

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

julia> sigma_ds = matrix_processing(DefaultMatrixProcessing(; denoise = Denoise()), sigma, X)
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

julia> sigma_dt = matrix_processing(DefaultMatrixProcessing(; detone = Detone()), sigma, X)
5×5 Matrix{Float64}:
 0.132026    0.0124802   0.0117303    0.0176194    0.0042142
 0.0124802   0.0514194   0.0273105   -0.0290864    0.0088165
 0.0117303   0.0273105   0.0843837   -0.00279296   0.0619156
 0.0176194  -0.0290864  -0.00279296   0.0424332   -0.0242252
 0.0042142   0.0088165   0.0619156   -0.0242252    0.0926441
```

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
function matrix_processing(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
                           X::AbstractMatrix, args...; kwargs...)
    sigma = copy(sigma)
    matrix_processing!(mp, sigma, X, args...; kwargs...)
    return sigma
end

export DefaultMatrixProcessing, NonPositiveDefiniteMatrixProcessing, matrix_processing,
       matrix_processing!
