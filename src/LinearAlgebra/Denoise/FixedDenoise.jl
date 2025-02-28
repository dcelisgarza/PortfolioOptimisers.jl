struct FixedDenoise{T1 <: Integer, T2 <: Integer, T3, T4 <: Tuple, T5 <: NamedTuple} <:
       DenoiseAlgorithm
    m::T1
    n::T2
    kernel::T3
    args::T4
    kwargs::T5
end
function FixedDenoise(; m::Integer = 10, n::Integer = 1000,
                      kernel = AverageShiftedHistograms.Kernels.gaussian, args::Tuple = (),
                      kwargs::NamedTuple = (;))
    return FixedDenoise{typeof(m), typeof(n), typeof(kernel), typeof(args), typeof(kwargs)}(m,
                                                                                            n,
                                                                                            kernel,
                                                                                            args,
                                                                                            kwargs)
end
function denoise!(::FixedDenoise, X::AbstractMatrix, vals::AbstractVector,
                  vecs::AbstractMatrix, num_factors::Integer)
    _vals = copy(vals)
    _vals[1:num_factors] .= sum(_vals[1:num_factors]) / num_factors
    X .= cov2cor(vecs * Diagonal(_vals) * transpose(vecs))
    return nothing
end

export FixedDenoise
