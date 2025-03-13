struct SquareRootKurtosis{T1 <: RiskMeasureSettings,
                          T2 <: Union{Nothing, <:AbstractWeights},
                          T3 <: Union{Nothing, <:AbstractVector{<:Real}},
                          T4 <: Union{Nothing, <:AbstractMatrix}} <: MuRiskMeasure
    settings::T1
    w::T2
    mu::T3
    kt::T4
end
function SquareRootKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                            kt::Union{Nothing, <:AbstractMatrix, Nothing} = nothing)
    if !isnothing(kt) && !isempty(kt)
        @smart_assert(size(kt, 1) == size(kt, 2))
    end
    return SquareRootKurtosis{typeof(settings), typeof(w), typeof(mu), typeof(kt)}(settings,
                                                                                   w, mu,
                                                                                   kt)
end
function (r::SquareRootKurtosis)(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees();
                                 scale::Bool = false)
    x = calc_net_returns(X, w, fees)
    mu = calc_ret_mu(x, w, r)
    val = x .- mu
    return sqrt(sum(val .^ 4) / length(x))
end

export SquareRootKurtosis
