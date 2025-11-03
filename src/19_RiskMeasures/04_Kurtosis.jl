"""
    struct Kurtosis{T1, T2, T3, T4, T5, T6, T7} <: RiskMeasure
        settings::T1
        w::T2
        mu::T3
        kt::T4
        N::T5
        alg1::T6
        alg2::T7
    end

Represents the square root kurtosis risk measure in PortfolioOptimisers.jl.

Computes portfolio risk as the square root of the fourth central moment (kurtosis) of the return distribution, optionally using custom weights, expected returns, and a kurtosis (fourth moment) matrix. This risk measure can be evaluated using either the full or semi (downside) deviations, depending on the algorithm provided.

# Fields

  - `settings`: Risk measure configuration.
  - `w`: Optional vector of observation weights.
  - `mu`: Optional expected returns value, vector, or `VecScalar` for the moment target, via [`calc_moment_target`](@ref). If `nothing` it is computed from the returns series using the optional weights in `w`.
  - `kt`: Optional cokurtosis (fourth moment) matrix that overrides the prior `kt` when provided.
  - `N`: Optional integer specifying the number of eigenvalues per asset to use from the cokurtosis matrix in an approximate formulation. If `nothing`, the exact formulation is used.
  - `alg1`: Moment algorithm specifying whether to use all or only downside deviations.
  - `alg2`: Specifies the `JuMP` formulation used to encode the risk measure.

# Constructors

    Kurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                       w::Union{Nothing, <:AbstractWeights} = nothing,
                       mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar} = nothing,
                       kt::Union{Nothing, <:AbstractMatrix} = nothing,
                       N::Union{Nothing, <:Integer} = nothing,
                       alg1::AbstractMomentAlgorithm = Full(),
                       alg2::VarianceFormulation = SOCRiskExpr())

Keyword arguments correspond to the fields above.

## Validation

  - If `mu` is not `nothing`:

      + `::Real`: `isfinite(mu)`.
      + `::AbstractVector`: `!isempty(mu)` and `all(isfinite, mu)`.

  - If `w` is not `nothing`, `!isempty(w)`.
  - If `kt` is not `nothing`:

      + `!isempty(kt)` and `size(kt, 1) == size(kt, 2)`.

      + If `mu` is not `nothing`:

          * `::AbstractVector`: `length(mu)^2 == size(kt, 1)`.
          * `::VecScalar`: `length(mu.v)^2 == size(kt, 1)`.
  - If `N` is provided: must be positive.

# `JuMP` Formulations

## Exact

This formulation is used when `N` is `nothing`.

## Approximate

This formulation is used when `N` is an integer, the larger the value of `N`, the more accurate and expensive it becomes.

# Examples

```jldoctest
julia> Kurtosis()
Kurtosis
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ nothing
        mu ┼ nothing
        kt ┼ nothing
         N ┼ nothing
      alg1 ┼ Full()
      alg2 ┴ SOCRiskExpr()
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
struct Kurtosis{T1, T2, T3, T4, T5, T6, T7} <: RiskMeasure
    settings::T1
    w::T2
    mu::T3
    kt::T4
    N::T5
    alg1::T6
    alg2::T7
    function Kurtosis(settings::RiskMeasureSettings, w::Union{Nothing, <:AbstractWeights},
                      mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar},
                      kt::Union{Nothing, <:AbstractMatrix}, N::Union{Nothing, <:Integer},
                      alg1::AbstractMomentAlgorithm, alg2::SecondMomentFormulation)
        mu_flag = isa(mu, AbstractVector)
        kt_flag = isa(kt, AbstractMatrix)
        if mu_flag
            @argcheck(!isempty(mu))
            @argcheck(all(isfinite, mu))
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
            @argcheck(length(mu)^2 == size(kt, 1))
        elseif isa(mu, VecScalar) && kt_flag
            @argcheck(length(mu.v)^2 == size(kt, 1))
        end
        if !isnothing(N)
            @argcheck(N > zero(N))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(kt), typeof(N),
                   typeof(alg1), typeof(alg2)}(settings, w, mu, kt, N, alg1, alg2)
    end
end
function Kurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  w::Union{Nothing, <:AbstractWeights} = nothing,
                  mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar} = nothing,
                  kt::Union{Nothing, <:AbstractMatrix} = nothing,
                  N::Union{Nothing, <:Integer} = nothing,
                  alg1::AbstractMomentAlgorithm = Full(),
                  alg2::SecondMomentFormulation = SOCRiskExpr())
    return Kurtosis(settings, w, mu, kt, N, alg1, alg2)
end
function calc_moment_target(::Kurtosis{<:Any, Nothing, Nothing, <:Any, <:Any, <:Any, <:Any},
                            ::Any, x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::Kurtosis{<:Any, <:AbstractWeights, Nothing, <:Any, <:Any,
                                        <:Any, <:Any}, ::Any, x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:AbstractVector, <:Any, <:Any, <:Any,
                                        <:Any}, w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecScalar, <:Any, <:Any, <:Any,
                                        <:Any}, w::AbstractVector, ::Any)
    return dot(w, r.mu.v) + r.mu.s
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:Real, <:Any, <:Any, <:Any, <:Any},
                            ::Any, ::Any)
    return r.mu
end
function calc_deviations_vec(r::Kurtosis, w::AbstractVector, X::AbstractMatrix,
                             fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Full, <:SOCRiskExpr})(w::AbstractVector,
                                                                                 X::AbstractMatrix,
                                                                                 fees::Union{Nothing,
                                                                                             <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? mean(val) : mean(val, r.w))
end
function (r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Semi, <:SOCRiskExpr})(w::AbstractVector,
                                                                                 X::AbstractMatrix,
                                                                                 fees::Union{Nothing,
                                                                                             <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? mean(val) : mean(val, r.w))
end
function (r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Full,
                      <:QuadSecondMomentFormulations})(w::AbstractVector, X::AbstractMatrix,
                                                       fees::Union{Nothing, <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Semi,
                      <:QuadSecondMomentFormulations})(w::AbstractVector, X::AbstractMatrix,
                                                       fees::Union{Nothing, <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function factory(r::Kurtosis, pr::HighOrderPrior, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, pr.w)
    mu = nothing_scalar_array_factory(r.mu, pr.mu)
    kt = nothing_scalar_array_factory(r.kt, pr.kt)
    return Kurtosis(; settings = r.settings, w = w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2)
end
function factory(r::Kurtosis, pr::LowOrderPrior, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, pr.w)
    mu = nothing_scalar_array_factory(r.mu, pr.mu)
    kt = nothing_scalar_array_factory(r.kt, nothing)
    return Kurtosis(; settings = r.settings, w = w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2)
end
function risk_measure_view(r::Kurtosis, i::AbstractVector, args...)
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
    return Kurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2)
end

export Kurtosis
