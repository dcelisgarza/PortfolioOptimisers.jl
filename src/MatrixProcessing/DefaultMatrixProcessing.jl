struct DefaultMatrixProcessing{T1 <: FixNonPositiveDefiniteMatrix, T2 <: DenoiseAlgorithm,
                               T3 <: AbstractDetone, T4 <: AbstractLoGo} <: MatrixProcessing
    fnpd::T1
    denoise::T2
    detone::T3
    logo::T4
end
function DefaultMatrixProcessing(;
                                 fnpd::FixNonPositiveDefiniteMatrix = FNPD_NearestCorrelationMatrix(),
                                 denoise::DenoiseAlgorithm = NoDenoise(),
                                 detone::NoDetone = NoDetone(),
                                 logo::AbstractLoGo = NoLoGo())
    return DefaultMatrixProcessing{typeof(fnpd), typeof(denoise), typeof(detone),
                                   typeof(logo)}(fnpd, denoise, detone, logo)
end
function mtx_process!(mp::DefaultMatrixProcessing, X::AbstractMatrix, T, N, args...;
                      kwargs...)
    fix_non_positive_definite_matrix!(mp.fnpd, X)
    denoise!(mp.denoise, mp.fnpd, X, T / N)
    detone!(mp.detone, mp.fnpd, X)
    logo!(mp.logo, mp.fnpd, X)
    return nothing
end

export DefaultMatrixProcessing
