struct DefaultMatrixProcessing{T1 <: FixNonPositiveDefiniteMatrix, T2 <: DenoiseAlgorithm,
                               T3 <: AbstractDetone, T4 <: Union{Nothing, <:LoGo}} <:
       MatrixProcessing
    fnpdm::T1
    denoise::T2
    detone::T3
    logo::T4
end
function DefaultMatrixProcessing(;
                                 fnpdm::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                                 denoise::DenoiseAlgorithm = NoDenoise(),
                                 detone::NoDetone = NoDetone(),
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
struct NonPositiveDefiniteMatrixProcessing{T1 <: DenoiseAlgorithm, T2 <: AbstractDetone,
                                           T3 <: Union{Nothing, <:LoGo}} <: MatrixProcessing
    denoise::T1
    detone::T2
    logo::T3
end
function NonPositiveDefiniteMatrixProcessing(; denoise::DenoiseAlgorithm = NoDenoise(),
                                             detone::NoDetone = NoDetone(),
                                             logo::Union{Nothing, <:LoGo} = nothing)
    return NonPositiveDefiniteMatrixProcessing{typeof(denoise), typeof(detone),
                                               typeof(logo)}(denoise, detone, logo)
end
function mtx_process!(mp::NonPositiveDefiniteMatrixProcessing, sigma::AbstractMatrix,
                      X::AbstractMatrix, args...; kwargs...)
    T, N = size(X)
    fix_non_positive_definite_matrix!(FNPDM_NoFix(), sigma)
    denoise!(mp.denoise, FNPDM_NoFix(), sigma, T / N)
    detone!(mp.detone, FNPDM_NoFix(), sigma)
    LoGo!(mp.logo, FNPDM_NoFix(), sigma, X)
    return nothing
end

export DefaultMatrixProcessing, NonPositiveDefiniteMatrixProcessing
