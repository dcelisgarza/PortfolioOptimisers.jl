struct FeesEstimator{T1, T2, T3, T4, T5, T6} <: AbstractEstimator
    tn::T1
    l::T2
    s::T3
    fl::T4
    fs::T5
    kwargs::T6
end
function FeesEstimator(; tn::Union{Nothing, <:TurnoverEstimator, <:Turnover} = nothing,
                       l::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       s::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       fl::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                 <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       fs::Union{Nothing, <:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                 <:AbstractVector{<:Pair{<:AbstractString, <:Real}}} = nothing,
                       kwargs::NamedTuple = (; atol = 1e-8))
    if isa(l, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(l), IsEmptyError(non_empty_msg("`l`") * "."))
    end
    if isa(s, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(s), IsEmptyError(non_empty_msg("`s`") * "."))
    end
    if isa(fl, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(fl), IsEmptyError(non_empty_msg("`fl`") * "."))
    end
    if isa(fs, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(fs), IsEmptyError(non_empty_msg("`fs`") * "."))
    end
    return FeesEstimator(tn, l, s, fl, fs, kwargs)
end
function fees_view(fees::FeesEstimator, i::AbstractVector)
    tn = turnover_view(fees.tn, i)
    return FeesEstimator(; tn = tn, l = fees.l, s = fees.s, fl = fees.fl, fs = fees.fs,
                         kwargs = fees.kwargs)
end
function fees_constraints(fees::FeesEstimator, sets::AssetSets;
                          datatype::DataType = Float64, strict::Bool = false)
    return Fees(; tn = turnover_constraints(fees.tn, sets; strict = strict),
                l = estimator_to_val(fees.l, sets, zero(datatype); strict = strict),
                s = estimator_to_val(fees.s, sets, zero(datatype); strict = strict),
                fl = estimator_to_val(fees.fl, sets, zero(datatype); strict = strict),
                fs = estimator_to_val(fees.fs, sets, zero(datatype); strict = strict))
end
struct Fees{T1, T2, T3, T4, T5, T6} <: AbstractResult
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
    if isa(l, Real)
        @argcheck(l >= zero(l), DomainError("`l` must be non-negative:\nl => $l"))
    elseif isa(l, AbstractVector)
        @argcheck(!isempty(l) && all(x -> x >= zero(x), l),
                  AssertionError("`l` must be non-empty and all must be non-negative:\nall(x -> x >= zero(x), l) => $(all(x -> x >= zero(x), l))"))
    end
    if isa(s, Real)
        @argcheck(s >= zero(s), DomainError("`s` must be non-negative:\ns => $s"))
    elseif isa(s, AbstractVector)
        @argcheck(!isempty(s) && all(x -> x >= zero(x), s),
                  AssertionError("`s` must be non-empty and all must be non-negative:\nall(x -> x >= zero(x), s) => $(all(x -> x >= zero(x), s))"))
    end
    if isa(fl, Real)
        @argcheck(fl >= zero(fl), DomainError("`fl` must be non-negative:\nfl => $fl"))
    elseif isa(fl, AbstractVector)
        @argcheck(!isempty(fl) && all(x -> x >= zero(x), fl),
                  AssertionError("`fl` must be non-empty and all must be non-negative:\nall(x -> x >= zero(x), fl) => $(all(x -> x >= zero(x), fl))"))
    end
    if isa(fs, Real)
        @argcheck(fs >= zero(fs), DomainError("`fs` must be non-negative:\nfs => $fs"))
    elseif isa(fs, AbstractVector)
        @argcheck(!isempty(fs) && all(x -> x >= zero(x), fs),
                  AssertionError("`fs` must be non-empty and all must be non-negative:\nall(x -> x >= zero(x), fs) => $(all(x -> x >= zero(x), fs))"))
    end
    return Fees(tn, l, s, fl, fs, kwargs)
end
function fees_constraints(fees::Union{Nothing, <:Fees}, args...; kwargs...)
    return fees
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

export FeesEstimator, Fees, fees_constraints, calc_fees, calc_asset_fees, calc_net_returns,
       calc_net_asset_returns, cumulative_returns, drawdowns
