struct SquareRootKurtosis{T1 <: RiskMeasureSettings, T2 <: AbstractMomentAlgorithm,
                          T3 <: Union{Nothing, <:AbstractWeights},
                          T4 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                          T5 <: Union{Nothing, <:AbstractMatrix}} <:
       SquareRootKurtosisRiskMeasure
    settings::T1
    alg::T2
    w::T3
    mu::T4
    kt::T5
end
function SquareRootKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            alg::AbstractMomentAlgorithm = Full(),
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                            kt::Union{Nothing, <:AbstractMatrix} = nothing)
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
    return SquareRootKurtosis{typeof(settings), typeof(alg), typeof(w), typeof(mu),
                              typeof(kt)}(settings, alg, w, mu, kt)
end
function calc_moment_target(::SquareRootKurtosis{<:Any, <:Any, Nothing, Nothing, <:Any},
                            ::Any, x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::SquareRootKurtosis{<:Any, <:Any, <:AbstractWeights, Nothing,
                                                  <:Any}, ::Any, x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:AbstractVector,
                                                  <:Any}, w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Real, <:Any},
                            ::Any, ::Any)
    return r.mu
end
function calc_moment_val(r::SquareRootKurtosis, w::AbstractVector, X::AbstractMatrix,
                         fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::SquareRootKurtosis{<:Any, <:Full, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                     X::AbstractMatrix,
                                                                     fees::Union{Nothing,
                                                                                 <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return sqrt(sum(val .^ 4) / size(X, 1))
end
function (r::SquareRootKurtosis{<:Any, <:Semi, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                     X::AbstractMatrix,
                                                                     fees::Union{Nothing,
                                                                                 <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return sqrt(sum(val[val .<= zero(eltype(val))] .^ 4) / size(X, 1))
end
function risk_measure_factory(r::SquareRootKurtosis, prior::HighOrderPriorResult, args...;
                              kwargs...)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    kt = risk_measure_nothing_real_array_factory(r.kt, prior.kt)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = r.w, mu = mu,
                              kt = kt)
end
function risk_measure_factory(r::SquareRootKurtosis,
                              prior::HighOrderPriorResult{<:EntropyPoolingPriorResult,
                                                          <:Any, <:Any, <:Any, <:Any},
                              args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    kt = risk_measure_nothing_real_array_factory(r.kt, prior.kt)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = w, mu = mu, kt = kt)
end
function risk_measure_factory(r::SquareRootKurtosis, prior::AbstractLowOrderPriorResult,
                              args...; kwargs...)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    kt = risk_measure_nothing_real_array_factory(r.kt, nothing)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = r.w, mu = mu,
                              kt = kt)
end
function risk_measure_factory(r::SquareRootKurtosis, prior::EntropyPoolingPriorResult,
                              args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_real_array_factory(r.mu, prior.mu)
    kt = risk_measure_nothing_real_array_factory(r.kt, nothing)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = w, mu = mu, kt = kt)
end
function risk_measure_view(r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any,
                                                 <:AbstractMatrix},
                           prior::AbstractPriorResult, i::AbstractVector, args...;
                           kwargs...)
    mu = risk_measure_nothing_real_array_view(r.mu, prior.mu, i)
    idx = fourth_moment_index_factory(size(prior.X, 2), i)
    kt = risk_measure_nothing_real_array_view(r.kt, nothing, idx)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = r.w, mu = mu,
                              kt = kt)
end
function risk_measure_view(r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any,
                                                 <:AbstractMatrix},
                           prior::EntropyPoolingPriorResult, i::AbstractVector, args...;
                           kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_real_array_view(r.mu, prior.mu, i)
    idx = fourth_moment_index_factory(size(prior.X, 2), i)
    kt = risk_measure_nothing_real_array_view(r.kt, nothing, idx)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = w, mu = mu, kt = kt)
end
function risk_measure_view(r::SquareRootKurtosis, prior::HighOrderPriorResult,
                           i::AbstractVector, args...; kwargs...)
    mu = risk_measure_nothing_real_array_view(r.mu, prior.mu, i)
    idx = fourth_moment_index_factory(size(prior.X, 2), i)
    kt = risk_measure_nothing_real_array_view(r.kt, prior.kt, idx)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = r.w, mu = mu,
                              kt = kt)
end
function risk_measure_view(r::SquareRootKurtosis,
                           prior::HighOrderPriorResult{<:EntropyPoolingPriorResult, <:Any,
                                                       <:Any, <:Any, <:Any},
                           i::AbstractVector, args...; kwargs...)
    w = risk_measure_nothing_real_array_factory(r.w, prior.pm.w)
    mu = risk_measure_nothing_real_array_view(r.mu, prior.mu, i)
    idx = fourth_moment_index_factory(size(prior.X, 2), i)
    kt = risk_measure_nothing_real_array_view(r.kt, prior.kt, idx)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = w, mu = mu, kt = kt)
end

export SquareRootKurtosis
