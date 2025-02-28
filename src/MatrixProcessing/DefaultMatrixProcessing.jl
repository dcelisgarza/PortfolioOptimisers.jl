struct DefaultMatrixProcessing{T1 <: FixNonPositiveDefiniteMatrix, T2 <: DenoiseAlgorithm,
                               T3 <: AbstractDetone, T4 <: AbstractLoGo} <: MatrixProcessing
    fix_non_pos_def::T1
    denoise::T2
    detone::T3
    logo::T4
end
function DefaultMatrixProcessing(;
                                 fix_non_pos_def::FixNonPositiveDefiniteMatrix = FNPDM_NearestCorrelationMatrix(),
                                 denoise::DenoiseAlgorithm = NoDenoise(),
                                 detone::NoDetone = NoDetone(),
                                 logo::AbstractLoGo = NoLoGo())
    return DefaultMatrixProcessing{typeof(fix_non_pos_def), typeof(denoise), typeof(detone),
                                   typeof(logo)}(fix_non_pos_def, denoise, detone, logo)
end
function mtx_process!(mp::DefaultMatrixProcessing, X::AbstractMatrix, T, N, args...;
                      kwargs...)
    fix_non_positive_definite_matrix!(mp.fix_non_pos_def, X)
    denoise!(mp.denoise, mp.fix_non_pos_def, X, T / N)
    detone!(mp.detone, mp.fix_non_pos_def, X)
    logo!(mp.logo, mp.fix_non_pos_def, X)
    return nothing
end

export DefaultMatrixProcessing
