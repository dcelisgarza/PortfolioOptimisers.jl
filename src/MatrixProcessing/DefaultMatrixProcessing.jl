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
function mtx_process!(mp::DefaultMatrixProcessing, X::AbstractMatrix, T, N, args...;
                      kwargs...)
    fix_non_positive_definite_matrix!(mp.fnpdm, X)
    denoise!(mp.denoise, mp.fnpdm, X, T / N)
    detone!(mp.detone, mp.fnpdm, X)
    logo!(mp.logo, mp.fnpdm, X)
    return nothing
end

export DefaultMatrixProcessing
