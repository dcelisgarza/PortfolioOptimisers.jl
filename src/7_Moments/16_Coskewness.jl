abstract type CoskewnessEstimator <: AbstractEstimator end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end
struct Coskewness{T1 <: AbstractExpectedReturnsEstimator,
                  T2 <: AbstractMatrixProcessingEstimator, T3 <: AbstractMomentAlgorithm} <:
       CoskewnessEstimator
    me::T1
    mp::T2
    alg::T3
end
function Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full())
    return Coskewness{typeof(me), typeof(mp), typeof(alg)}(me, mp, alg)
end
#=
function Base.show(io::IO, csk::Coskewness)
    println(io, "Coskewness")
    for field in fieldnames(typeof(csk))
        val = getfield(csk, field)
        print(io, "  ", lpad(string(field), 3), " ")
        if isa(val, AbstractExpectedReturnsEstimator) ||
           isa(val, AbstractMatrixProcessingEstimator) ||
           isa(val, AbstractMomentAlgorithm)
            ioalg = IOBuffer()
            show(ioalg, val)
            algstr = String(take!(ioalg))
            alglines = split(algstr, '\n')
            println(io, "| ", alglines[1])
            for l in alglines[2:end]
                println(io, "      | ", l)
            end
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
end
=#
function factory(ce::Coskewness, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Coskewness(; me = factory(ce.me, w), mp = ce.mp, alg = ce.alg)
end
function __coskewness(cskew, X, mp)
    N = size(cskew, 1)
    V = zeros(eltype(cskew), N, N)
    for i in 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew, :, j:k)
        matrix_processing!(mp, coskew_jk, X)
        vals, vecs = eigen(coskew_jk)
        vals .= clamp.(real.(vals), typemin(eltype(cskew)), zero(eltype(cskew))) +
                clamp.(imag.(vals), typemin(eltype(cskew)), zero(eltype(cskew)))im
        V .-= real(vecs * Diagonal(vals) * transpose(vecs))
    end
    return V
end
function _coskewness(y, X, mp)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)),
                        length = size(X, 2)))
    z = kron(o, y) ⊙ kron(y, o)
    cskew = transpose(X) * z / size(X, 1)
    V = __coskewness(cskew, X, mp)
    return cskew, V
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    y = X .- mu
    return _coskewness(y, X, ske.mp)
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(y, X, ske.mp)
end

export Coskewness, coskewness
