struct Fees{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
            T2 <: Union{<:Real, <:AbstractVector{<:Real}},
            T3 <: Union{<:Real, <:AbstractVector{<:Real}},
            T4 <: Union{<:Real, <:AbstractVector{<:Real}}, T5 <: AbstractTurnover,
            T6 <: NamedTuple}
    long::T1
    short::T2
    fixed_long::T3
    fixed_short::T4
    rebalance::T5
    tol_kwargs::T6
end
function Fees(; long::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              short::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              fixed_long::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              fixed_short::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
              rebalance::AbstractTurnover = NoTurnover(),
              tol_kwargs::NamedTuple = (; atol = 1e-8))
    @smart_assert(all(long .>= zero(long)))
    @smart_assert(all(short .>= zero(short)))
    @smart_assert(all(fixed_long .>= zero(fixed_long)))
    @smart_assert(all(fixed_short .>= zero(fixed_short)))
    return Fees{typeof(long), typeof(short), typeof(fixed_long), typeof(fixed_short),
                typeof(rebalance), typeof(tol_kwargs)}(long, short, fixed_long, fixed_short,
                                                       rebalance, tol_kwargs)
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
    return if !(isempty(fees) || all(iszero.(fees)))
        idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
        dot(fees[idx], w[idx] .* latest_prices[idx])
    else
        zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees)))
    end
end
function calc_fees(w::AbstractVector, latest_prices::AbstractVector, rebalance::Turnover)
    fees_rebal = rebalance.val
    benchmark = rebalance.w
    return if isa(fees_rebal, Real)
        sum(fees_rebal * abs.(benchmark .- w) .* latest_prices)
    elseif isa(fees_rebal, AbstractVector) &&
           !(isempty(fees_rebal) || all(iszero.(fees_rebal)))
        dot(fees_rebal, abs.(benchmark .- w) .* latest_prices)
    else
        zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees_rebal),
                          eltype(benchmark)))
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
    fees_rebal = calc_fees(w, latest_prices, fees.rebalance)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_rebal
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
    return if !(isempty(fees) || all(iszero.(fees)))
        idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
        dot(fees[idx], w[idx])
    else
        zero(promote_type(eltype(w), eltype(fees)))
    end
end
function calc_fees(w::AbstractVector, rebalance::Turnover)
    fees_rebal = rebalance.val
    benchmark = rebalance.w
    return if isa(fees_rebal, Real)
        sum(fees_rebal * abs.(benchmark .- w))
    elseif isa(fees_rebal, AbstractVector) &&
           !(isempty(fees_rebal) || all(iszero.(fees_rebal)))
        dot(fees_rebal, abs.(benchmark .- w))
    else
        zero(promote_type(eltype(w), eltype(fees_rebal), eltype(benchmark)))
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
    return if !(isempty(fees) || all(iszero.(fees)))
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
    fees_rebal = calc_fees(w, fees.rebalance)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_rebal
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
    if !(isempty(fees) || all(iszero.(fees)))
        idx = op(w, zero(promote_type(eltype(w), eltype(fees))))
        fees_w[idx] .= fees[idx] .* w[idx]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, rebalance::Turnover)
    fees_rebal = rebalance.val
    benchmark = rebalance.w
    fees_w = zeros(promote_type(eltype(w), eltype(fees_rebal), eltype(benchmark)),
                   length(w))
    if isa(fees_rebal, Real)
        fees_w .= fees_rebal * abs.(benchmark .- w)
    elseif isa(fees_rebal, AbstractVector) &&
           !(isempty(fees_rebal) || all(iszero.(fees_rebal)))
        fees_w .= fees_rebal .* abs.(benchmark .- w)
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
    if !(isempty(fees) || all(iszero.(fees)))
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
    fees_rebal = calc_asset_fees(w, fees.rebalance)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_rebal
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
    if !(isempty(fees) || all(iszero.(fees)))
        idx = op(w, zero(promote_type(eltype(w), eltype(latest_prices), eltype(fees))))
        fees_w[idx] .= fees[idx] .* w[idx] .* latest_prices[idx]
    end
    return fees_w
end
function calc_asset_fees(w::AbstractVector, latest_prices::AbstractVector,
                         rebalance::Turnover)
    fees_rebal = rebalance.val
    benchmark = rebalance.w
    fees_w = zeros(promote_type(eltype(w), eltype(latest_prices), eltype(fees_rebal),
                                eltype(benchmark)), length(w))
    if isa(fees_rebal, Real)
        fees_w .= fees_rebal * abs.(benchmark .- w) .* latest_prices
    elseif isa(fees_rebal, AbstractVector) &&
           !(isempty(fees_rebal) || all(iszero.(fees_rebal)))
        fees_w .= fees_rebal .* abs.(benchmark .- w) .* latest_prices
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
    fees_rebal = calc_asset_fees(w, latest_prices, fees.rebalance)
    return fees_long + fees_short + fees_fixed_long + fees_fixed_short + fees_rebal
end
function calc_net_returns(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    return X * w .- calc_fees(w, fees)
end
function calc_net_asset_returns(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    return X .* transpose(w) .- transpose(calc_asset_fees(w, fees))
end

export Fees, calc_fees, calc_asset_fees, calc_net_returns, calc_net_asset_returns
