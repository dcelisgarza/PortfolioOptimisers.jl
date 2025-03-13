struct Fees{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
            T2 <: Union{<:Real, <:AbstractVector{<:Real}},
            T3 <: Union{<:Real, <:AbstractVector{<:Real}},
            T4 <: Union{<:Real, <:AbstractVector{<:Real}}, T5 <: AbstractTurnover,
            T6 <: NamedTuple}
    long::T1
    short::T2
    fixed_long::T3
    fixed_short::T4
    turnover::T5
    tol_kwargs::T6
end
function Fees(; long::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              short::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              fixed_long::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              fixed_short::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              turnover::AbstractTurnover = NoTurnover(),
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
    @smart_assert(all(long .>= zero(long)))
    @smart_assert(all(short .>= zero(short)))
    @smart_assert(all(fixed_long .>= zero(fixed_long)))
    @smart_assert(all(fixed_short .>= zero(fixed_short)))
    return Fees{typeof(long), typeof(short), typeof(fixed_long), typeof(fixed_short),
                typeof(turnover), typeof(tol_kwargs)}(long, short, fixed_long, fixed_short,
                                                      turnover, tol_kwargs)
end
function cluster_turnover_fees_factory(turnover::NoTurnover, ::AbstractVector)
    return turnover
end
function cluster_turnover_fees_factory(turnover::Turnover, cluster::AbstractVector)
    val = cluster_real_or_vector_factory(turnover.val, cluster)
    w = view(turnover.w, cluster)
    return Turnover(; val = val, w = w)
end
function cluster_fees_factory(fees::Fees; cluster::AbstractVector, kwargs...)
    long = cluster_real_or_vector_factory(fees.long, cluster)
    fixed_long = cluster_real_or_vector_factory(fees.fixed_long, cluster)
    turnover = cluster_turnover_fees_factory(fees.turnover, cluster)
    return Fees(; long = long, fixed_long = fixed_long, turnover = turnover,
                tol_kwargs = fees.tol_kwargs)
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Real,
                   op::Function)
    return if !iszero(fees)
        idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
        sum(fees * w[idx] .* latest_prices[idx])
    else
        zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees)))
    end
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector,
                   fees::AbstractVector{<:Real}, op::Function)
    return if !all(iszero.(fees))
        idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
        dot(fees[idx], w[idx] .* latest_prices[idx])
    else
        zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees)))
    end
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector,
                   turnover::Turnover{<:Real, <:Any})
    return if !iszero(turnover.val)
        sum(turnover.val * abs.(turnover.w .- w) .* latest_prices)
    else
        zero(promote_type(eltype(w), eltype(latest_prices), eltype(turnover.val),
                          eltype(turnover.w)))
    end
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector,
                   turnover::Turnover{<:AbstractVector, <:Any})
    if !all(iszero.(turnover.val))
        dot(turnover.val, abs.(turnover.w .- w) .* latest_prices)
    else
        zero(promote_type(eltype(w), eltype(latest_prices), eltype(turnover.val),
                          eltype(turnover.w)))
    end
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, ::NoTurnover)
    return zero(promote_type(eltype(w), eltype(latest_prices)))
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Fees = Fees())
    fees_long = calc_fees(w, latest_prices, fees.long, .>=)
    fees_short = calc_fees(w, latest_prices, -fees.short, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_fees(w, latest_prices, fees.turnover)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_fees(w::AbstractVector, fees::Real, op::Function)
    return if !iszero(fees)
        idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
        sum(fees * w[idx])
    else
        zero(promote_type(eltype(w), eltype(fees)))
    end
end
function calc_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    return if !all(iszero.(fees))
        idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
        dot(fees[idx], w[idx])
    else
        zero(promote_type(eltype(w), eltype(fees)))
    end
end
function calc_fees(w::AbstractVector, turnover::Turnover{<:Real, <:Any})
    return if !iszero(turnover.val)
        sum(turnover.val * abs.(turnover.w .- w))
    else
        zero(promote_type(eltype(w), eltype(turnover.val), eltype(turnover.w)))
    end
end
function calc_fees(w::AbstractVector, turnover::Turnover{<:AbstractVector, <:Any})
    return if !all(iszero.(turnover.val))
        dot(turnover.val, abs.(turnover.w .- w))
    else
        zero(promote_type(eltype(w), eltype(turnover.val), eltype(turnover.w)))
    end
end
function calc_fees(w::AbstractVector, ::NoTurnover)
    return zero(eltype(w))
end
function calc_fixed_fees(w::AbstractVector, fees::Real, tol_kwargs::NamedTuple,
                         op::Function)
    return if !iszero(fees)
        idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
        idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees)));
                           tol_kwargs...)
        fees * sum(idx2)
    else
        zero(promote_type(eltype(w), eltype(fees)))
    end
end
function calc_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                         tol_kwargs::NamedTuple, op::Function)
    return if !all(iszero.(fees))
        idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
        idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees)));
                           tol_kwargs...)
        sum(fees[idx1][idx2])
    else
        zero(promote_type(eltype(w), eltype(fees)))
    end
end
function calc_fees(w::AbstractVector, fees::Fees = Fees())
    fees_long = calc_fees(w, fees.long, .>=)
    fees_short = calc_fees(w, -fees.short, .<)
    fees_fixed_long = calc_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_fees(w, fees.turnover)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, fees::Real, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    if !iszero(fees)
        idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
        fees_w[idx] .= fees * w[idx]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::AbstractVector{<:Real}, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    if !all(iszero.(fees))
        idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
        fees_w[idx] .= fees[idx] .* w[idx]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, turnover::Turnover{<:Real, <:Any})
    fees_w = zeros(promote_type(eltype(w), eltype(turnover.val), eltype(turnover.w)),
                   length(w))
    if !iszero(turnover.val)
        fees_w .= turnover.val * abs.(turnover.w .- w)
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector,
                         turnover::Turnover{<:AbstractVector{<:Real}, <:Any})
    fees_w = zeros(promote_type(eltype(w), eltype(turnover.val), eltype(turnover.w)),
                   length(w))
    if !all(iszero.(turnover.val))
        fees_w .= turnover.val .* abs.(turnover.w .- w)
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, ::NoTurnover)
    return zeros(eltype(w), length(w))
end
function calc_asset_fixed_fees(w::AbstractVector, fees::Real, tol_kwargs::NamedTuple,
                               op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    if !iszero(fees)
        idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
        idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees)));
                           tol_kwargs...)
        fees_w[idx1] .= fees * idx2
    end
    return fees_w
end
function calc_asset_fixed_fees(w::AbstractVector, fees::AbstractVector{<:Real},
                               tol_kwargs::NamedTuple, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(fees)), length(w))
    if !all(iszero.(fees))
        idx1 = op(w, zero(promote_type(eltype(w), eltype(fees))))
        idx2 = .!isapprox.(w[idx1], zero(promote_type(eltype(w), eltype(fees)));
                           tol_kwargs...)
        fees_w[idx1] .= fees[idx1][idx2]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, fees::Fees = Fees())
    fees_long = calc_asset_fees(w, fees.long, .>=)
    fees_short = calc_asset_fees(w, -fees.short, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_asset_fees(w, fees.turnover)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector, fees::Real,
                         op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(latest_prices), eltype(fees)), length(w))
    if !iszero(fees)
        idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
        fees_w[idx] .= fees * w[idx] .* latest_prices[idx]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         fees::AbstractVector{<:Real}, op::Function)
    fees_w = zeros(promote_type(eltype(w), eltype(latest_prices), eltype(fees)), length(w))
    if !all(iszero.(fees))
        idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
        fees_w[idx] .= fees[idx] .* w[idx] .* latest_prices[idx]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         turnover::Turnover{<:Real, <:Any})
    fees_w = zeros(promote_type(eltype(w), eltype(latest_prices), eltype(turnover.val),
                                eltype(turnover.w)), length(w))
    if !iszero(turnover.val)
        fees_w .= turnover.val * abs.(turnover.w .- w) .* latest_prices
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         turnover::Turnover{<:AbstractVector{<:Real}, <:Any})
    fees_w = zeros(promote_type(eltype(w), eltype(latest_prices), eltype(turnover.val),
                                eltype(turnover.w)), length(w))
    if !all(iszero.(turnover.val))
        fees_w .= turnover.val .* abs.(turnover.w .- w) .* latest_prices
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector, ::NoTurnover)
    return zeros(promote_type(eltype(w), eltype(latest_prices)), length(w))
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         fees::Fees = Fees())
    fees_long = calc_asset_fees(w, latest_prices, fees.long, .>=)
    fees_short = calc_asset_fees(w, latest_prices, -fees.short, .<)
    fees_fixed_long = calc_asset_fixed_fees(w, fees.fixed_long, fees.tol_kwargs, .>=)
    fees_fixed_short = calc_asset_fixed_fees(w, fees.fixed_short, fees.tol_kwargs, .<)
    fees_turnover = calc_asset_fees(w, latest_prices, fees.turnover)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_turnover
end
function calc_net_returns(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    return X * w .- calc_fees(w, fees)
end
function calc_net_asset_returns(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    return X .* transpose(w) .- transpose(calc_asset_fees(w, fees))
end

export Fees, calc_fees, calc_asset_fees, calc_net_returns, calc_net_asset_returns
