abstract type AbstractDenoiseEstimator <: AbstractEstimator end
struct BaseDenoise{T1 <: Integer, T2 <: Integer, T3, T4 <: Tuple, T5 <: NamedTuple} <:
       AbstractDenoiseEstimator
    m::T1
    n::T2
    kernel::T3
    args::T4
    kwargs::T5
end
function BaseDenoise(; m::Integer = 10, n::Integer = 1000,
                     kernel = AverageShiftedHistograms.Kernels.gaussian, args::Tuple = (),
                     kwargs::NamedTuple = (;))
    return BaseDenoise{typeof(m), typeof(n), typeof(kernel), typeof(args), typeof(kwargs)}(m,
                                                                                           n,
                                                                                           kernel,
                                                                                           args,
                                                                                           kwargs)
end
struct SpectralDenoise{T1 <: BaseDenoise} <: AbstractDenoiseEstimator
    de::T1
end
function SpectralDenoise(; de::BaseDenoise = BaseDenoise())
    return SpectralDenoise{typeof(de)}(de)
end
struct FixedDenoise{T1 <: BaseDenoise} <: AbstractDenoiseEstimator
    de::T1
end
function FixedDenoise(; de::BaseDenoise = BaseDenoise())
    return FixedDenoise{typeof(de)}(de)
end
struct ShrunkDenoise{T1 <: BaseDenoise, T2 <: Real} <: AbstractDenoiseEstimator
    de::T1
    alpha::T2
end
function ShrunkDenoise(; de::BaseDenoise = BaseDenoise(), alpha::Real = 0.0)
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return ShrunkDenoise{typeof(de), typeof(alpha)}(de, alpha)
end
function Base.getproperty(d::Union{<:SpectralDenoise, <:FixedDenoise, <:ShrunkDenoise},
                          sym::Symbol)
    if sym ∈ (:m, :n, :kernel, :args, :kwargs)
        return getfield(d.de, sym)
    else
        return getfield(d, sym)
    end
end
function fit!(::SpectralDenoise, X::AbstractMatrix, vals::AbstractVector,
              vecs::AbstractMatrix, num_factors::Integer)
    _vals = copy(vals)
    _vals[1:num_factors] .= zero(eltype(X))
    X .= cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
    return nothing
end
function fit!(::FixedDenoise, X::AbstractMatrix, vals::AbstractVector, vecs::AbstractMatrix,
              num_factors::Integer)
    _vals = copy(vals)
    _vals[1:num_factors] .= sum(_vals[1:num_factors]) / num_factors
    X .= cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
    return nothing
end
function fit!(de::ShrunkDenoise, X::AbstractMatrix, vals::AbstractVector,
              vecs::AbstractMatrix, num_factors::Integer)
    # Small
    vals_l = vals[1:num_factors]
    vecs_l = vecs[:, 1:num_factors]

    # Large
    vals_r = vals[(num_factors + 1):end]
    vecs_r = vecs[:, (num_factors + 1):end]

    corr0 = vecs_r * Diagonal(vals_r) * transpose(vecs_r)
    corr1 = vecs_l * Diagonal(vals_l) * transpose(vecs_l)

    X .= corr0 + de.alpha * corr1 + (one(de.alpha) - de.alpha) * Diagonal(corr1)
    return nothing
end
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
function fit!(de::AbstractDenoiseEstimator, pdm::Union{Nothing, <:PosDefMatrixEstimator},
              X::AbstractMatrix, q::Real)
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
    fit!(de, X, vals, vecs, num_factors)
    fit!(pdm, X)
    if iscov
        StatsBase.cor2cov!(X, s)
    end
    return nothing
end
function fit(de::AbstractDenoiseEstimator, pdm::Union{Nothing, <:PosDefMatrixEstimator},
             X::AbstractMatrix, q::Real)
    X = copy(X)
    fit!(de, pdm, X, q)
    return X
end

export BaseDenoise, SpectralDenoise, FixedDenoise, ShrunkDenoise