struct FixedDenoise{T1 <: Real, T2 <: Integer, T3 <: Integer, T4, T5 <: Tuple,
                    T6 <: NamedTuple} <: DenoiseAlgorithm
    alpha::T1
    m::T2
    n::T3
    kernel::T4
    args::T5
    kwargs::T6
end
function FixedDenoise(; alpha::Real = 0.0, m::Integer = 10, n::Integer = 1000,
                      kernel = AverageShiftedHistograms.Kernels.gaussian, args::Tuple = (),
                      kwargs::NamedTuple = (;))
    @smart_assert(zero(alpha) <= alpha <= one(alpha))
    return FixedDenoise{typeof(alpha), typeof(m), typeof(n), typeof(kernel), typeof(args),
                        typeof(kwargs)}(alpha, m, n, kernel, args, kwargs)
end
function denoise!(::FixedDenoise, X::AbstractMatrix, vals::AbstractVector,
                  vecs::AbstractMatrix, num_factors::Integer)
    _vals = copy(vals)
    _vals[1:num_factors] .= sum(_vals[1:num_factors]) / num_factors
    X .= cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
    return nothing
end

export FixedDenoise
