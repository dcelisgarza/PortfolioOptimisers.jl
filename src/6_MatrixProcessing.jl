abstract type AbstractMatrixProcessingEstimator <: AbstractEstimator end
abstract type AbstractMatrixProcessingAlgorithm <: AbstractAlgorithm end
struct DefaultMatrixProcessing{T1 <: Union{Nothing, <:PosDefEstimator},
                               T2 <: Union{Nothing, <:DenoiseEstimator},
                               T3 <: Union{Nothing, <:DetoneEstimator},
                               T4 <: Union{Nothing, <:AbstractMatrixProcessingAlgorithm}} <:
       AbstractMatrixProcessingEstimator
    pdm::T1
    denoise::T2
    detone::T3
    alg::T4
end
function DefaultMatrixProcessing(;
                                 pdm::Union{Nothing, <:PosDefEstimator} = PosDefEstimator(),
                                 denoise::Union{Nothing, <:DenoiseEstimator} = nothing,
                                 detone::Union{Nothing, <:DetoneEstimator} = nothing,
                                 alg::Union{Nothing, <:AbstractMatrixProcessingAlgorithm} = nothing)
    return DefaultMatrixProcessing{typeof(pdm), typeof(denoise), typeof(detone),
                                   typeof(alg)}(pdm, denoise, detone, alg)
end
function fit!(mp::DefaultMatrixProcessing, sigma::AbstractMatrix, X::AbstractMatrix,
              args...; kwargs...)
    T, N = size(X)
    fit!(mp.pdm, sigma)
    fit!(mp.denoise, mp.pdm, sigma, T / N)
    fit!(mp.detone, mp.pdm, sigma)
    fit!(mp.alg, mp.pdm, sigma, X)
    return nothing
end
function fit(mp::DefaultMatrixProcessing, sigma::AbstractMatrix, X::AbstractMatrix, args...;
             kwargs...)
    X = copy(X)
    fit!(mp, sigma, X, args...; kwargs...)
    return X
end
struct NonPositiveDefiniteMatrixProcessing{T1 <: Union{Nothing, <:DenoiseEstimator},
                                           T2 <: Union{Nothing, <:DetoneEstimator},
                                           T3 <: Union{Nothing,
                                                       <:AbstractMatrixProcessingAlgorithm}} <:
       AbstractMatrixProcessingEstimator
    denoise::T1
    detone::T2
    alg::T3
end
function NonPositiveDefiniteMatrixProcessing(;
                                             denoise::Union{Nothing, <:DenoiseEstimator} = nothing,
                                             detone::Union{Nothing, <:DetoneEstimator} = nothing,
                                             alg::Union{Nothing,
                                                        <:AbstractMatrixProcessingAlgorithm} = nothing)
    return NonPositiveDefiniteMatrixProcessing{typeof(denoise), typeof(detone),
                                               typeof(alg)}(denoise, detone, alg)
end
function fit!(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
              X::AbstractMatrix, args...; kwargs...)
    T, N = size(X)
    fit!(nothing, sigma)
    fit!(mp.denoise, nothing, sigma, T / N)
    fit!(mp.detone, nothing, sigma)
    fit!(mp.alg, nothing, sigma, X)
    return nothing
end
function fit(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
             X::AbstractMatrix, args...; kwargs...)
    X = copy(X)
    fit!(mp, sigma, X, args...; kwargs...)
    return X
end

export DefaultMatrixProcessing, NonPositiveDefiniteMatrixProcessing
