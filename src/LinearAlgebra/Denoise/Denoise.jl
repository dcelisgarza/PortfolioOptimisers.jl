function errPDF(x, vals; kernel = AverageShiftedHistograms.Kernels.gaussian, m = 10,
                n = 1000, q = 1000)
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    rg = range(e_min, e_max; length = n)
    pdf1 = q ./ (2 * pi * x * rg) .*
           sqrt.(clamp.((e_max .- rg) .* (rg .- e_min), zero(x), typemax(x)))
    e_min, e_max = x * (1 - sqrt(1.0 / q))^2, x * (1 + sqrt(1.0 / q))^2
    res = ash(vals; rng = range(e_min, e_max; length = n), kernel = kernel, m = m)
    pdf2 = [AverageShiftedHistograms.pdf(res, i) for i ∈ pdf1]
    pdf2[.!isfinite.(pdf2)] .= 0.0
    sse = sum((pdf2 - pdf1) .^ 2)
    return sse
end
function find_max_eval(vals, q; kernel = AverageShiftedHistograms.Kernels.gaussian,
                       m::Integer = 10, n::Integer = 1000, args = (), kwargs = (;))
    res = Optim.optimize(x -> errPDF(x, vals; kernel = kernel, m = m, n = n, q = q), 0.0,
                         1.0, args...; kwargs...)
    x = Optim.converged(res) ? Optim.minimizer(res) : 1.0
    e_max = x * (1.0 + sqrt(1.0 / q))^2
    return e_max, x
end
function denoise!(de::DenoiseAlgorithm,
                  fnpdm::Union{Nothing, <:FixNonPositiveDefiniteMatrix}, X::AbstractMatrix,
                  q::Real)
    issquare(X)
    s = diag(X)
    iscov = any(.!isone.(s))
    if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor!(X, s)
    end
    vals, vecs = eigen(X)
    max_val = find_max_eval(vals, q; kernel = de.kernel, m = de.m, n = de.n, args = de.args,
                            kwargs = de.kwargs)[1]
    num_factors = findlast(vals .< max_val)
    denoise!(de, X, vals, vecs, num_factors)
    fix_non_positive_definite_matrix!(fnpdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
