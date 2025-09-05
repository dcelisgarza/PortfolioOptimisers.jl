struct SquareRootKurtosis{T1, T2, T3, T4, T5, T6} <: SquareRootKurtosisRiskMeasure
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
        @argcheck(!isempty(mu) && all(isfinite, mu))
    elseif isa(mu, Real)
        @argcheck(isfinite(mu))
    end
    if isa(w, AbstractWeights)
        @argcheck(!isempty(w))
    end
    if kt_flag
        @argcheck(!isempty(kt))
        assert_matrix_issquare(kt)
    end
    if mu_flag && kt_flag
        @argcheck(length(mu)^2 == size(kt, 2))
    end
    if !isnothing(N)
        @argcheck(N > zero(N))
    end
    return SquareRootKurtosis(settings, w, mu, kt, N, alg)
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
    return x .- target
end
function (r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Full})(w::AbstractVector,
                                                                            X::AbstractMatrix,
                                                                            fees::Union{Nothing,
                                                                                        <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? mean(val) : mean(val, r.w))
end
function (r::SquareRootKurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Semi})(w::AbstractVector,
                                                                            X::AbstractMatrix,
                                                                            fees::Union{Nothing,
                                                                                        <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? mean(val) : mean(val, r.w))
end
function factory(r::SquareRootKurtosis,
                 pr::HighOrderPrior{<:LowOrderPrior, <:Any, <:Any, <:Any, <:Any}, args...;
                 kwargs...)
    w = nothing_scalar_array_factory(r.w, pr.w)
    mu = nothing_scalar_array_factory(r.mu, pr.mu)
    kt = nothing_scalar_array_factory(r.kt, pr.kt)
    return SquareRootKurtosis(; settings = r.settings, alg = r.alg, w = w, mu = mu, kt = kt,
                              N = r.N)
end
function factory(r::SquareRootKurtosis, pr::LowOrderPrior, args...; kwargs...)
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
        isqrt(size(kt, 1))
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
