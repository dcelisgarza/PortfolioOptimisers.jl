struct DefaultMatrixProcessing{T1 <: FixNonPositiveDefiniteMatrix, T2 <: DenoiseAlgorithm,
                               T3 <: AbstractDetone, T4 <: AbstractLoGo} <: MatrixProcessing
    fnpdm::T1
    denoise::T2
    detone::T3
    logo::T4
end
function DefaultMatrixProcessing(;
                                 fnpdm::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                                 denoise::DenoiseAlgorithm = NoDenoise(),
                                 detone::NoDetone = NoDetone(),
                                 logo::AbstractLoGo = NoLoGo())
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

export DefaultMatrixProcessing
