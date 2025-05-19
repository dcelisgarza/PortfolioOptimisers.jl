struct SquareRootKurtosis{T1 <: RiskMeasureSettings,
                          T2 <: Union{Nothing, <:AbstractWeights},
                          T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                          T4 <: Union{Nothing, <:AbstractMatrix},
                          T5 <: Union{Nothing, <:Integer}, T6 <: AbstractMomentAlgorithm} <:
       SquareRootKurtosisRiskMeasure
    settings::T1
    w::T2
    mu::T3
    kt::T4
    N::T5
    alg::T6
end
function SquareRootKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                            kt::Union{Nothing, <:AbstractMatrix} = nothing,
                            N::Union{Nothing, <:Integer} = nothing,
                            alg::AbstractMomentAlgorithm = Full())
    mu_flag = isa(mu, AbstractVector)
    kt_flag = isa(kt, AbstractMatrix)
    if mu_flag
        @smart_assert(!isempty(mu))
    end
    if kt_flag
        @smart_assert(!isempty(kt))
        issquare(kt)
    end
    if mu_flag && kt_flag
        @smart_assert(length(mu)^2 == size(kt, 2))
    end
    if !isnothing(N)
        @smart_assert(N > zero(N))
    end
    return SquareRootKurtosis{typeof(settings), typeof(w), typeof(mu), typeof(kt),
                              typeof(N), typeof(alg)}(settings, w, mu, kt, N, alg)
end
function calc_moment_target(::SquareRootKurtosis{<:Any, Nothing, Nothing, <:Any, <:Any,
                                                 <:Any}, ::Any, x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::SquareRootKurtosis{<:Any, <:AbstractWeights, Nothing, <:Any,
                                                  <:Any, <:Any}, ::Any, x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(r::SquareRootKurtosis{<:Any, <:Any, <:AbstractVector, <:Any,
                                                  <:Any, <:Any}, w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::SquareRootKurtosis{<:Any, <:Any, <:Real, <:Any, <:Any,
                                                  <:Any}, ::Any, ::Any)
    return r.mu
end
function calc_moment_val(r::SquareRootKurtosis, w::AbstractVector, X::AbstractMatrix,
                         fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x ⊖ target
end
function (r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Full})(w::AbstractVector,
                                                                            X::AbstractMatrix,
                                                                            fees::Union{Nothing,
                                                                                        <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return sqrt(isnothing(r.w) ? mean(val .^ 4) : mean(val .^ 4, r.w))
end
function (r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Semi})(w::AbstractVector,
                                                                            X::AbstractMatrix,
                                                                            fees::Union{Nothing,
                                                                                        <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return sqrt(isnothing(r.w) ? mean(val .^ 4) : mean(val .^ 4, r.w))
end
function factory(r::SquareRootKurtosis,
                 pr::HighOrderPriorResult{<:LowOrderPriorResult, <:Any, <:Any, <:Any,
                                          <:Any}, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, pr.w)
    mu = nothing_scalar_array_factory(r.mu, pr.mu)
    kt = nothing_scalar_array_factory(r.kt, pr.kt)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = w, mu = mu, kt = kt,
                              N = r.N)
end
function factory(r::SquareRootKurtosis, pr::LowOrderPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, pr.w)
    mu = nothing_scalar_array_factory(r.mu, pr.mu)
    kt = nothing_scalar_array_factory(r.kt, nothing)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = w, mu = mu, kt = kt,
                              N = r.N)
end
function risk_measure_view(r::SquareRootKurtosis, i::AbstractVector, args...)
    mu = r.mu
    kt = r.kt
    j = if isa(mu, AbstractVector)
        length(mu)
    elseif isa(kt, AbstractMatrix)
        round(Int, sqrt(size(kt, 1)))
    else
        nothing
    end
    if !isnothing(j) && !isnothing(kt)
        idx = fourth_moment_index_factory(j, i)
        kt = nothing_scalar_array_view(kt, idx)
    end
    mu = nothing_scalar_array_view(mu, i)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = r.w, mu = mu,
                              kt = kt, N = r.N)
end

export SquareRootKurtosis
