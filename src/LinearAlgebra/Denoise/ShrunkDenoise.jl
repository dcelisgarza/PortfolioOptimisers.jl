struct ShrunkDenoise{T1 <: Real, T2 <: Integer, T3 <: Integer, T4, T5 <: Tuple,
                     T6 <: NamedTuple} <: DenoiseAlgorithm
    alpha::T1
    m::T2
    n::T3
    kernel::T4
    args::T5
    kwargs::T6
end
function ShrunkDenoise(; alpha::Real = 0.0, m::Integer = 10, n::Integer = 1000,
                       kernel = AverageShiftedHistograms.Kernels.gaussian, args::Tuple = (),
                       kwargs::NamedTuple = (;))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return ShrunkDenoise{typeof(alpha), typeof(m), typeof(n), typeof(kernel), typeof(args),
                         typeof(kwargs)}(alpha, m, n, kernel, args, kwargs)
end
function denoise!(de::ShrunkDenoise, X::AbstractMatrix, vals::AbstractVector,
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

export ShrunkDenoise
