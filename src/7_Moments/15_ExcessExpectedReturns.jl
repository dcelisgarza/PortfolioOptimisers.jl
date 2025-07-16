struct ExcessExpectedReturns{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real} <:
       AbstractShrunkExpectedReturnsEstimator
    me::T1
    rf::T2
end
function ExcessExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               rf::Real = 0.0)
    return ExcessExpectedReturns{typeof(me), typeof(rf)}(me, rf)
end
#=
function Base.show(io::IO, eer::ExcessExpectedReturns)
    println(io, "ExcessExpectedReturns")
    for field in fieldnames(typeof(eer))
        val = getfield(eer, field)
        print(io, "  ", string(field), " ")
        if isa(val, AbstractExpectedReturnsEstimator)
            ioalg = IOBuffer()
            show(ioalg, val)
            algstr = String(take!(ioalg))
            alglines = split(algstr, '\n')
            println(io, "| ", alglines[1])
            for l in alglines[2:end]
                println(io, "     | ", l)
            end
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
end
=#
function Statistics.mean(me::ExcessExpectedReturns, X::AbstractArray; dims::Int = 1,
                         kwargs...)
    return mean(me.me, X; dims = dims, kwargs...) .- me.rf
end
function factory(me::ExcessExpectedReturns, w::Union{Nothing, <:AbstractWeights} = nothing)
    return ExcessExpectedReturns(; me = factory(me.me, w), rf = me.rf)
end

export ExcessExpectedReturns
