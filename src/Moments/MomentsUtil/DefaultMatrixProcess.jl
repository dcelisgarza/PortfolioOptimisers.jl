struct DefaultMatrixProcess <: MatrixProcess end

function mtx_process!(::DefaultMatrixProcess, ce, X::AbstractMatrix, T, N, args...;
                      kwargs...)
    # posdef_fix!(ce.posdef, X)
    # denoise!(ce.denoise, ce.posdef, X, T / N)
    # detone!(ce.detone, ce.posdef, X)
    # logo!(ce.logo, ce.posdef, X)
    return nothing
end

export DefaultMatrixProcess
