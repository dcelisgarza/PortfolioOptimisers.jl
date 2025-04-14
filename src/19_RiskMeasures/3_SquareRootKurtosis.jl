struct SquareRootKurtosis{T1 <: RiskMeasureSettings,
                          T2 <: Union{Nothing, <:AbstractWeights},
                          T3 <: Union{Nothing, <:AbstractVector{<:Real}},
                          T4 <: Union{Nothing, <:AbstractMatrix},
                          T5 <: AbstractMomentAlgorithm} <: SquareRootKurtosisRiskMeasure
    settings::T1
    w::T2
    mu::T3
    kt::T4
    alg::T5
end
function SquareRootKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                            kt::Union{Nothing, <:AbstractMatrix} = nothing,
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
    return SquareRootKurtosis{typeof(settings), typeof(w), typeof(mu), typeof(kt),
                              typeof(alg)}(settings, w, mu, kt, alg)
end
function (r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Full})(w::AbstractVector,
                                                                     X::AbstractMatrix,
                                                                     fees::Union{Nothing,
                                                                                 <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_ret_mu(x, w, r)
    val = x .- mu
    return sqrt(sum(val .^ 4) / length(x))
end
function (r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Semi})(w::AbstractVector,
                                                                     X::AbstractMatrix,
                                                                     fees::Union{Nothing,
                                                                                 <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    mu = calc_ret_mu(x, w, r)
    val = x .- mu
    return sqrt(sum(val[val .<= zero(eltype(val))] .^ 4) / length(x))
end

export SquareRootKurtosis