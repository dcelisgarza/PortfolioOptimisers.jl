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
    tol_kwargs::T6
end
function Fees(; tn::Union{Nothing, <:Turnover} = nothing,
              l::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              s::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fl::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fs::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              tol_kwargs::NamedTuple = (; atol = 1e-8))
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
        @smart_assert(all(l .> Ref(zero(l))))
    end
    if !isnothing(s)
        @smart_assert(all(s .> Ref(zero(s))))
    end
    if !isnothing(fl)
        @smart_assert(all(fl .> Ref(zero(fl))))
    end
    if !isnothing(fs)
        @smart_assert(all(fs .> Ref(zero(fs))))
    end
    return Fees{typeof(tn), typeof(l), typeof(s), typeof(fl), typeof(fs),
                typeof(tol_kwargs)}(tn, l, s, fl, fs, tol_kwargs)
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
    return Fees(; tn = tn, l = l, s = s, fl = fl, fs = fs, tol_kwargs = fees.tol_kwargs)
end
function factory(fees::Fees, w::AbstractVector)
    return Fees(; tn = factory(fees.tn, w), l = fees.l, s = fees.s, fl = fees.fl,
                fs = fees.fs, tol_kwargs = fees.tol_kwargs)
end
function calc_fees(w::AbstractVector, p::AbstractVector, ::Nothing, ::Function)
    return zero(promote_type(eltype(w), eltype(p)))
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::Real, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot_scalar(fees * view(w, idx), view(p, idx))
end
function calc_fees(w::AbstractVector, p::AbstractVector, fees::AbstractVector{<:Real},
                   op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    return dot(view(fees, idx), view(w, idx) .* view(p, idx))
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
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.tol_kwargs, .<)
    fees_turnover = calc_fees(w, p, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_fees(w::AbstractVector, ::Nothing, ::Function)
    return zero(eltype(w))
end
function calc_fees(w::AbstractVector, fees::Real, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return sum(fees * view(w, idx))
end
function calc_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    return dot(view(fees, idx), view(w, idx))
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
function calc_fixed_fees(w::AbstractVector, ::Nothing, tol_kwargs::NamedTuple, op::Function)
    return zero(eltype(w))
end
function calc_fixed_fees(w::AbstractVector, fees::Real, tol_kwargs::NamedTuple,
                         op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(view(w, idx1), Ref(zero(promote_type(eltype(w), eltype(fees))));
                       tol_kwargs...)
    return fees * sum(idx2)
end
function calc_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                         tol_kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(view(w, idx1), Ref(zero(promote_type(eltype(w), eltype(fees))));
                       tol_kwargs...)
    return sum(view(view(fees, idx1), idx2))
end
function calc_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_fees(w, fees.l, .>=)
    fees_short = -calc_fees(w, fees.s, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fl, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fs, fees.tol_kwargs, .<)
    fees_turnover = calc_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, ::Nothing, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::AbstractVector, fees::Real, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] .= fees * view(w, idx)
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
    fees_w[idx] .= view(fees, idx) ⊙ view(w, idx)
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
function calc_asset_fixed_fees(w::AbstractVector, fees::Real, tol_kwargs::NamedTuple,
                               op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(view(w, idx1), zero(promote_type(eltype(w), eltype(fees)));
                       tol_kwargs...)
    fees_w[idx1] .= fees * idx2
    return fees_w
end
function calc_asset_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                               tol_kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(view(w, idx1), zero(promote_type(eltype(w), eltype(fees)));
                       tol_kwargs...)
    fees_w[idx1] .= view(view(fees, idx1), idx2)
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_asset_fees(w, fees.l, .>=)
    fees_short = -calc_asset_fees(w, fees.s, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.tol_kwargs, .<)
    fees_turnover = calc_asset_fees(w, fees.tn)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, ::Nothing, ::Function)
    return zeros(promote_type(eltype(w), eltype(p)), length(w))
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::Real, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] .= fees * view(w, idx) ⊙ view(p, idx)
    return fees_w
end
function calc_asset_fees(w::AbstractVector, p::AbstractVector, fees::AbstractVector{<:Real},
                         op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(p), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(p), eltype(fees))))
    fees_w[idx] .= view(fees, idx) ⊙ view(w, idx) ⊙ view(p, idx)
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
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fl, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fs, fees.tol_kwargs, .<)
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
        cumprod(Ref(one(eltype(X))) .+ X; dims = dims)
    else
        cumsum(X; dims = dims)
    end
end
function drawdowns(X::AbstractArray; compound::Bool = false, dims::Int = 1)
    cX = cumulative_returns(X; compound = compound, dims = dims)
    if compound
        return cX ./ accumulate(max, cX; dims = dims) .- Ref(one(eltype(X)))
    else
        return cX - accumulate(max, cX; dims = dims)
    end
    return nothing
end

export Fees, calc_fees, calc_asset_fees, calc_net_returns, calc_net_asset_returns,
       cumulative_returns, drawdowns
