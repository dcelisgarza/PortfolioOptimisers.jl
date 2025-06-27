abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end
function matrix_processing_algorithm!(::Nothing, args...; kwargs...)
    return nothing
end
function matrix_processing_algorithm(::Nothing, args...; kwargs...)
    return nothing
end
function matrix_processing!(::Nothing, args...; kwargs...)
    return nothing
end
function matrix_processing(::Nothing, args...; kwargs...)
    return nothing
end
struct DefaultMatrixProcessing{T1 <: Union{Nothing, <:PosDefEstimator},
                               T2 <: Union{Nothing, <:Denoise},
                               T3 <: Union{Nothing, <:Detone},
                               T4 <: Union{Nothing, <:AbstractMatrixProcessingAlgorithm}} <:
       AbstractMatrixProcessingEstimator
    pdm::T1
    denoise::T2
    detone::T3
    alg::T4
end
function DefaultMatrixProcessing(;
                                 pdm::Union{Nothing, <:PosDefEstimator} = PosDefEstimator(),
                                 denoise::Union{Nothing, <:Denoise} = nothing,
                                 detone::Union{Nothing, <:Detone} = nothing,
                                 alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm} = nothing)
    return DefaultMatrixProcessing{typeof(pdm), typeof(denoise), typeof(detone),
                                   typeof(alg)}(pdm, denoise, detone, alg)
end
function matrix_processing!(mp::DefaultMatrixProcessing, sigma::AbstractMatrix,
                            X::AbstractMatrix, args...; kwargs...)
    T, N = size(X)
    posdef!(mp.pdm, sigma)
    denoise!(mp.denoise, mp.pdm, sigma, T / N)
    detone!(mp.detone, mp.pdm, sigma)
    matrix_processing_algorithm!(mp.alg, mp.pdm, sigma, X; kwargs...)
    return nothing
end
function matrix_processing(mp::DefaultMatrixProcessing, sigma::AbstractMatrix,
                           X::AbstractMatrix, args...; kwargs...)
    sigma = copy(sigma)
    matrix_processing!(mp, sigma, X, args...; kwargs...)
    return sigma
end
struct NonPositiveDefiniteMatrixProcessing{T1 <: Union{Nothing, <:Denoise},
                                           T2 <: Union{Nothing, <:Detone},
                                           T3 <: Union{Nothing,
                                                       <:AbstractMatrixProcessingAlgorithm}} <:
       AbstractMatrixProcessingEstimator
    denoise::T1
    detone::T2
    alg::T3
end
function NonPositiveDefiniteMatrixProcessing(; denoise::Union{Nothing, <:Denoise} = nothing,
                                             detone::Union{Nothing, <:Detone} = nothing,
                                             alg::Union{Nothing,
                                                        <:AbstractMatrixProcessingAlgorithm} = nothing)
    return NonPositiveDefiniteMatrixProcessing{typeof(denoise), typeof(detone),
                                               typeof(alg)}(denoise, detone, alg)
end
function matrix_processing!(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
                            X::AbstractMatrix, args...; kwargs...)
    T, N = size(X)
    posdef!(nothing, sigma)
    denoise!(mp.denoise, nothing, sigma, T / N)
    detone!(mp.detone, nothing, sigma)
    matrix_processing_algorithm!(mp.alg, nothing, sigma, X; kwargs...)
    return nothing
end
function matrix_processing(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
                           X::AbstractMatrix, args...; kwargs...)
    sigma = copy(sigma)
    matrix_processing!(mp, sigma, X, args...; kwargs...)
    return sigma
end

export DefaultMatrixProcessing, NonPositiveDefiniteMatrixProcessing, matrix_processing,
       matrix_processing!
