struct SquareRootKurtosis{T1 <: RiskMeasureSettings, T2 <: AbstractMomentAlgorithm,
                          T3 <: Union{Nothing, <:AbstractWeights},
                          T4 <: Union{Nothing, <:AbstractVector{<:Real}},
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
                            mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
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
function (r::SquareRootKurtosis{<:Any, <:Full, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                     X::AbstractMatrix,
                                                                     fees::Union{Nothing,
                                                                                 <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_ret_mu(x, w, r)
    val = x .- mu
    return sqrt(sum(val .^ 4) / length(x))
end
function (r::SquareRootKurtosis{<:Any, <:Semi, <:Any, <:Any, <:Any})(w::AbstractVector,
                                                                     X::AbstractMatrix,
                                                                     fees::Union{Nothing,
                                                                                 <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_ret_mu(x, w, r)
    val = x .- mu
    return sqrt(sum(val[val .<= zero(eltype(val))] .^ 4) / length(x))
end

export SquareRootKurtosis