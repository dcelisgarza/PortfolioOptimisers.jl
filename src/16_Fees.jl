struct Fees{T1 <: Union{Nothing, <:Turnover},
            T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
            T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
            T4 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
            T5 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}, T6 <: NamedTuple} <:
       AbstractEstimator
    tn::T1
    l::T2
    s::T3
    fl::T4
    fs::T5
    kwargs::T6
end
function Fees(; tn::Union{Nothing, <:Turnover} = nothing,
              l::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              s::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fl::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fs::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              kwargs::NamedTuple = (; atol = 1e-8))
    if isa(l, AbstractVector)
        @smart_assert(!isempty(l))
    end
    if isa(s, AbstractVector)
        @smart_assert(!isempty(s))
    end
    if isa(fl, AbstractVector)
        @smart_assert(!isempty(fl))
    end
    if isa(fs, AbstractVector)
        @smart_assert(!isempty(fs))
    end
    if !isnothing(l)
        @smart_assert(all(x -> x > zero(x), l))
    end
    if !isnothing(s)
        @smart_assert(all(x -> x > zero(x), s))
    end
    if !isnothing(fl)
        @smart_assert(all(x -> x > zero(x), fl))
    end
    if !isnothing(fs)
        @smart_assert(all(x -> x > zero(x), fs))
    end
    return Fees{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs), typeof(kwargs)}(tn,
                                                                                          l,
                                                                                          s,
                                                                                          fl,
                                                                                          fs,
                                                                                          kwargs)
end
Base.length(::Fees) = 1
Base.iterate(::Fees, i = 1) = i <= 1 ? (i, nothing) : nothing
function Base.show(io::IO, fees::Fees)
    println(io, "Fees")
    for field in fieldnames(typeof(fees))
        val = getfield(fees, field)
        print(io, "  ", lpad(string(field), 6), " ")
        if isnothing(val)
            println(io, "| nothing")
        elseif isa(val, Turnover)
            io_tn = IOBuffer()
            show(io_tn, val)
            tnstr = String(take!(io_tn))
            tnlines = split(tnstr, '\n')
            println(io, "| ", tnlines[1])
            for l in tnlines[2:end]
                println(io, "         | ", l)
            end
        elseif isa(val, AbstractVector) && length(val) ≤ 6
            println(io, "| $(typeof(val)): ", repr(val))
        elseif isa(val, AbstractVector)
            println(io, "| $(length(val))-element $(typeof(val))")
        elseif isa(val, NamedTuple)
            println(io, "| $(typeof(val)): ", repr(val))
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
end
function fees_view(::Nothing, ::Any)
    return nothing
end
function fees_view(fees::Fees, i::AbstractVector)
    tn = turnover_view(fees.tn, i)
    l = nothing_scalar_array_view(fees.l, i)
    s = nothing_scalar_array_view(fees.s, i)
    fl = nothing_scalar_array_view(fees.fl, i)
    fs = nothing_scalar_array_view(fees.fs, i)
    return Fees(; tn = tn, l = l, s = s, fl = fl, fs = fs, kwargs = fees.kwargs)
end
function factory(fees::Fees, w::AbstractVector)
    return Fees(; tn = factory(fees.tn, w), l = fees.l, s = fees.s, fl = fees.fl,
                fs = fees.fs, kwargs = fees.kwargs)
end
function calc_fees(w::AbstractVector, p::AbstractVector, ::Nothing, ::Function)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::Real, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot_scalar(fees * w[idx], p[idx])
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::AbstractVector{<:Real},
                   op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot(fees[idx], w[idx] .* p[idx])
end
function calc_fees(w::AbstractVector, p::AbstractVector, ::Nothing)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::AbstractVector, p::AbstractVector, tn::Turnover{<:Any, <:Real})
    return dot_scalar(tn.val * abs.(w - tn.w), p)
end
function calc_fees(w::AbstractVector, p::AbstractVector,
                   tn::Turnover{<:Any, <:AbstractVector})
    return dot(tn.val, abs.(w - tn.w) .* p)
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::Fees)
    fees_long = calc_fees(w, p, fees.l, .>=)
    fees_short = -calc_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_fees(w::AbstractVector, ::Nothing, ::Function)
    return zero(eltype(w))
end
function calc_fees(w::AbstractVector, fees::Real, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return sum(fees * w[idx])
end
function calc_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return dot(fees[idx], w[idx])
end
function calc_fees(w::AbstractVector, tn::Turnover{<:Any, <:Real})
    return sum(tn.val * abs.(w - tn.w))
end
function calc_fees(w::AbstractVector, tn::Turnover{<:Any, <:AbstractVector})
    return dot(tn.val, abs.(w - tn.w))
end
function calc_fees(w::AbstractVector, ::Nothing)
    return zero(eltype(w))
end
function calc_fixed_fees(w::AbstractVector, ::Nothing, kwargs::NamedTuple, op::Function)
    return zero(eltype(w))
end
function calc_fixed_fees(w::AbstractVector, fees::Real, kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return fees * sum(idx2)
end
function calc_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                         kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    return sum(fees[idx1][idx2])
end
function calc_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_fees(w, fees.l, .>=)
    fees_short = -calc_fees(w, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, ::Nothing, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::AbstractVector, fees::Real, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] .= fees * w[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] .= fees[idx] ⊙ w[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, ::Nothing)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::AbstractVector, tn::Turnover{<:Any, <:Real})
    return tn.val * abs.(w - tn.w)
end
function calc_asset_fees(w::AbstractVector, tn::Turnover{<:Any, <:AbstractVector})
    return tn.val ⊙ abs.(w - tn.w)
end
function calc_asset_fixed_fees(w::AbstractVector, ::Nothing, ::NamedTuple, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fixed_fees(w::AbstractVector, fees::Real, kwargs::NamedTuple,
                               op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] .= fees * idx2
    return fees_w
end
function calc_asset_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                               kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); kwargs...)
    fees_w[idx1] .= fees[idx1][idx2]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_asset_fees(w, fees.l, .>=)
    fees_short = -calc_asset_fees(w, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, ::Nothing, ::Function)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::Real, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] .= fees * w[idx] ⊙ p[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::AbstractVector{<:Real},
                         op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] .= fees[idx] ⊙ w[idx] ⊙ p[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, tn::Turnover{<:Any, <:Real})
    return tn.val * abs.(w - tn.w) ⊙ p
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, ::Nothing)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector,
                         tn::Turnover{<:Any, <:AbstractVector})
    return tn.val ⊙ abs.(w - tn.w) ⊙ p
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::Fees)
    fees_long = calc_asset_fees(w, p, fees.l, .>=)
    fees_short = -calc_asset_fees(w, p, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.kwargs, .<)
    fees_turnover = calc_asset_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_net_returns(w::AbstractVector, X::AbstractMatrix, args...)
    return X * w
end
function calc_net_returns(w::AbstractVector, X::AbstractMatrix, fees::Fees)
    return X * w .- calc_fees(w, fees)
end
function calc_net_asset_returns(w::AbstractVector, X::AbstractMatrix, args...)
    return X ⊙ transpose(w)
end
function calc_net_asset_returns(w::AbstractVector, X::AbstractMatrix, fees::Fees)
    return X ⊙ transpose(w) .- transpose(calc_asset_fees(w, fees))
end
function cumulative_returns(X::AbstractArray; compound::Bool = false, dims::Int = 1)
    return if compound
        cumprod(one(eltype(X)) .+ X; dims = dims)
    else
        cumsum(X; dims = dims)
    end
end
function drawdowns(X::AbstractArray; cX::Bool = false, compound::Bool = false,
                   dims::Int = 1)
    cX = !cX ? cumulative_returns(X; compound = compound, dims = dims) : X
    if compound
        return cX ./ accumulate(max, cX; dims = dims) .- one(eltype(X))
    else
        return cX - accumulate(max, cX; dims = dims)
    end
    return nothing
end

export Fees, calc_fees, calc_asset_fees, calc_net_returns, calc_net_asset_returns,
       cumulative_returns, drawdowns
