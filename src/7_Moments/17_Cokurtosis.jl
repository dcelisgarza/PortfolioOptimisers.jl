abstract type CokurtosisEstimator <: AbstractEstimator end
function cokurtosis(::Nothing, args...; kwargs...)
    return nothing
end
struct Cokurtosis{T1 <: AbstractExpectedReturnsEstimator,
                  T2 <: AbstractMatrixProcessingEstimator, T3 <: AbstractMomentAlgorithm} <:
       CokurtosisEstimator
    me::T1
    mp::T2
    alg::T3
end
function Cokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full())
    return Cokurtosis{typeof(me), typeof(mp), typeof(alg)}(me, mp, alg)
end
#=
function Base.show(io::IO, ck::Cokurtosis)
    println(io, "Cokurtosis")
    for field in fieldnames(typeof(ck))
        val = getfield(ck, field)
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
function factory(ce::Cokurtosis, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Cokurtosis(; me = factory(ce.me, w), mp = ce.mp, alg = ce.alg)
end
function _cokurosis(X, mp)
    T, N = size(X)
    o = transpose(range(; start = one(eltype(X)), stop = one(eltype(X)), length = N))
    z = kron(o, X) ⊙ kron(X, o)
    ckurt = transpose(z) * z / T
    matrix_processing!(mp, ckurt, X)
    return ckurt
end
function cokurtosis(ke::Cokurtosis{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ke.me, X; kwargs...) : mean
    X = X .- mu
    return _cokurosis(X, ke.mp)
end
function cokurtosis(ke::Cokurtosis{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ke.me, X; kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return _cokurosis(X, ke.mp)
end

export cokurtosis, Cokurtosis
