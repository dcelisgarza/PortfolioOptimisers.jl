struct DefaultMatrixProcessing{T1 <: Union{Nothing, <:FixNonPositiveDefiniteMatrix},
                               T2 <: Union{Nothing, <:DenoiseAlgorithm},
                               T3 <: Union{Nothing, <:AbstractDetone},
                               T4 <: Union{Nothing, <:LoGo}} <: MatrixProcessing
    fnpdm::T1
    denoise::T2
    detone::T3
    logo::T4
end
function DefaultMatrixProcessing(;
                                 fnpdm::Union{Nothing, <:FixNonPositiveDefiniteMatrix} = FNPDM_NearestCorrelationMatrix(),
                                 denoise::Union{Nothing, <:DenoiseAlgorithm} = nothing,
                                 detone::Union{Nothing, <:AbstractDetone} = nothing,
                                 logo::Union{Nothing, <:LoGo} = nothing)
    return DefaultMatrixProcessing{typeof(fnpdm), typeof(denoise), typeof(detone),
                                   typeof(logo)}(fnpdm, denoise, detone, logo)
end
function mtx_process!(mp::DefaultMatrixProcessing, sigma::AbstractMatrix, X::AbstractMatrix,
                      args...; kwargs...)
    T, N = size(X)
    fix_non_positive_definite_matrix!(mp.fnpdm, sigma)
    denoise!(mp.denoise, mp.fnpdm, sigma, T / N)
    detone!(mp.detone, mp.fnpdm, sigma)
    LoGo!(mp.logo, mp.fnpdm, sigma, X)
    return nothing
end
struct NonPositiveDefiniteMatrixProcessing{T1 <: Union{Nothing, <:DenoiseAlgorithm},
                                           T2 <: Union{Nothing, <:AbstractDetone},
                                           T3 <: Union{Nothing, <:LoGo}} <: MatrixProcessing
    denoise::T1
    detone::T2
    logo::T3
end
function NonPositiveDefiniteMatrixProcessing(;
                                             denoise::Union{Nothing, <:DenoiseAlgorithm} = nothing,
                                             detone::Union{Nothing, <:AbstractDetone} = nothing,
                                             logo::Union{Nothing, <:LoGo} = nothing)
    return NonPositiveDefiniteMatrixProcessing{typeof(denoise), typeof(detone),
                                               typeof(logo)}(denoise, detone, logo)
end
function mtx_process!(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
                      X::AbstractMatrix, args...; kwargs...)
    T, N = size(X)
    fix_non_positive_definite_matrix!(nothing, sigma)
    denoise!(mp.denoise, nothing, sigma, T / N)
    detone!(mp.detone, nothing, sigma)
    LoGo!(mp.logo, nothing, sigma, X)
    return nothing
end

export DefaultMatrixProcessing, NonPositiveDefiniteMatrixProcessing
