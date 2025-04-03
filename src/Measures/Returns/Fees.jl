struct Fees{T1 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
            T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
            T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
            T4 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
            T5 <: Union{Nothing, <:Turnover}, T6 <: NamedTuple}
    long::T1
    short::T2
    fixed_long::T3
    fixed_short::T4
    turnover::T5
    tol_kwargs::T6
end
function Fees(; long::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              short::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fixed_long::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              fixed_short::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
              turnover::Union{Nothing, <:Turnover} = nothing,
              tol_kwargs::NamedTuple = (; atol = 1e-8))
    if isa(long, AbstractVector)
        @smart_assert(!isempty(long))
    end
    if isa(short, AbstractVector)
        @smart_assert(!isempty(short))
    end
    if isa(fixed_long, AbstractVector)
        @smart_assert(!isempty(fixed_long))
    end
    if isa(fixed_short, AbstractVector)
        @smart_assert(!isempty(fixed_short))
    end
    if !isnothing(long)
        @smart_assert(all(long .> zero(long)))
    end
    if !isnothing(short)
        @smart_assert(all(short .> zero(short)))
    end
    if !isnothing(fixed_long)
        @smart_assert(all(fixed_long .> zero(fixed_long)))
    end
    if !isnothing(fixed_short)
        @smart_assert(all(fixed_short .> zero(fixed_short)))
    end
    return Fees{typeof(long), typeof(short), typeof(fixed_long), typeof(fixed_short),
                typeof(turnover), typeof(tol_kwargs)}(long, short, fixed_long, fixed_short,
                                                      turnover, tol_kwargs)
end
function cluster_fees_factory(::Nothing, cluster::AbstractVector; kwargs...)
    return nothing
end
function cluster_fees_factory(fees::Fees, cluster::AbstractVector; kwargs...)
    long = cluster_real_or_vector_factory(fees.long, cluster)
    fixed_long = cluster_real_or_vector_factory(fees.fixed_long, cluster)
    turnover = cluster_turnover_factory(fees.turnover, cluster)
    return Fees(; long = long, fixed_long = fixed_long, turnover = turnover,
                tol_kwargs = fees.tol_kwargs)
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, ::Nothing, ::Function)
    return zero(promote_type(eltype(w), eltype(latest_prices)))
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Real,
                   op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
    return sum(fees * w[idx] .* latest_prices[idx])
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector,
                   fees::AbstractVector{<:Real}, op::Function)
    idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
    return dot(fees[idx], w[idx] .* latest_prices[idx])
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, ::Nothing)
    return zero(promote_type(eltype(w), eltype(latest_prices)))
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector,
                   turnover::Turnover{<:Real, <:Any})
    return sum(turnover.val * abs.(w - turnover.w) .* latest_prices)
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector,
                   turnover::Turnover{<:AbstractVector, <:Any})
    return dot(turnover.val, abs.(w - turnover.w) .* latest_prices)
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Fees)
    fees_long = calc_fees(w, latest_prices, fees.long, .>=)
    fees_short = -calc_fees(w, latest_prices, fees.short, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_fees(w, latest_prices, fees.turnover)
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
function calc_fees(w::AbstractVector, turnover::Turnover{<:Real, <:Any})
    return sum(turnover.val * abs.(w - turnover.w))
end
function calc_fees(w::AbstractVector, turnover::Turnover{<:AbstractVector, <:Any})
    return dot(turnover.val, abs.(w - turnover.w))
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
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); tol_kwargs...)
    return fees * sum(idx2)
end
function calc_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                         tol_kwargs::NamedTuple, op::Function)
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); tol_kwargs...)
    return sum(fees[idx1][idx2])
end
function calc_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_fees(w, fees.long, .>=)
    fees_short = -calc_fees(w, fees.short, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_fees(w, fees.turnover)
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
    fees_w[idx] .= fees[idx] .* w[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, ::Nothing)
    return zeros(eltype(w), length(w))
end
function calc_asset_fees(w::AbstractVector, turnover::Turnover{<:Real, <:Any})
    return turnover.val * abs.(w - turnover.w)
end
function calc_asset_fees(w::AbstractVector,
                         turnover::Turnover{<:AbstractVector{<:Real}, <:Any})
    return turnover.val .* abs.(w - turnover.w)
end
function calc_asset_fixed_fees(w::AbstractVector, ::Nothing, ::NamedTuple, ::Function)
    return zeros(eltype(w), length(w))
end
function calc_asset_fixed_fees(w::AbstractVector, fees::Real, tol_kwargs::NamedTuple,
                               op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); tol_kwargs...)
    fees_w[idx1] .= fees * idx2
    return fees_w
end
function calc_asset_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                               tol_kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
    idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees))); tol_kwargs...)
    fees_w[idx1] .= fees[idx1][idx2]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::Fees)
    fees_long = calc_asset_fees(w, fees.long, .>=)
    fees_short = -calc_asset_fees(w, fees.short, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_asset_fees(w, fees.turnover)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector, ::Nothing,
                         ::Function)
    return zeros(promote_type(eltype(w), eltype(latest_prices)), length(w))
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Real,
                         op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(latest_prices), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
    fees_w[idx] .= fees * w[idx] .* latest_prices[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         fees::AbstractVector{<:Real}, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(latest_prices), eltype(fees)), length(w))
    idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
    fees_w[idx] .= fees[idx] .* w[idx] .* latest_prices[idx]
    return fees_w
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         turnover::Turnover{<:Real, <:Any})
    return turnover.val * abs.(w - turnover.w) .* latest_prices
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector, ::Nothing)
    return zeros(promote_type(eltype(w), eltype(latest_prices)), length(w))
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         turnover::Turnover{<:AbstractVector{<:Real}, <:Any})
    return turnover.val .* abs.(w - turnover.w) .* latest_prices
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Fees)
    fees_long = calc_asset_fees(w, latest_prices, fees.long, .>=)
    fees_short = -calc_asset_fees(w, latest_prices, fees.short, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_asset_fees(w, latest_prices, fees.turnover)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_net_returns(w::AbstractVector, X::AbstractMatrix, args...)
    return X * w
end
function calc_net_returns(w::AbstractVector, X::AbstractMatrix, fees::Fees)
    return X * w .- calc_fees(w, fees)
end
function calc_net_asset_returns(w::AbstractVector, X::AbstractMatrix, args...)
    return X .* transpose(w)
end
function calc_net_asset_returns(w::AbstractVector, X::AbstractMatrix, fees::Fees)
    return X .* transpose(w) .- transpose(calc_asset_fees(w, fees))
end

export Fees, calc_fees, calc_asset_fees, calc_net_returns, calc_net_asset_returns
