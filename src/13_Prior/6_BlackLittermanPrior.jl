"""
```julia
struct BlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7} <: AbstractLowOrderPriorEstimator_AF
    pe::T1
    mp::T2
    views::T3
    sets::T4
    views_conf::T5
    rf::T6
    tau::T7
end
```

Black-Litterman prior estimator for asset returns.

`BlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using the Black-Litterman model. It combines a prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and a blending parameter `tau`. The estimator supports both direct and constraint-based views, and allows for flexible confidence specification and matrix processing.

# Fields

  - `pe`: Prior estimator.
  - `mp`: Matrix post-processing estimator.
  - `views`: Views estimator or views object.
  - `sets`: Asset sets.
  - `views_conf`: View confidence(s).
  - `rf`: Risk-free rate.
  - `tau`: Blending parameter.

# Constructor

```julia
BlackLittermanPrior(;
                    pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(;
                                                                               me = EquilibriumExpectedReturns()),
                    mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                    views::Union{<:LinearConstraintEstimator, <:BlackLittermanViews},
                    sets::Union{Nothing, <:AssetSets} = nothing,
                    views_conf::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                    rf::Real = 0.0, tau::Union{Nothing, <:Real} = nothing)
```

Keyword arguments correspond to the fields above.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`AssetSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
struct BlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7} <: AbstractLowOrderPriorEstimator_AF
    pe::T1
    mp::T2
    views::T3
    sets::T4
    views_conf::T5
    rf::T6
    tau::T7
    function BlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mp::AbstractMatrixProcessingEstimator,
                                 views::Union{<:LinearConstraintEstimator,
                                              <:BlackLittermanViews},
                                 sets::Union{Nothing, <:AssetSets},
                                 views_conf::Union{Nothing, <:Real,
                                                   <:AbstractVector{<:Real}}, rf::Real,
                                 tau::Union{Nothing, <:Real})
        if isa(views, LinearConstraintEstimator)
            @argcheck(!isnothing(sets))
        end
        assert_bl_views_conf(views_conf, views)
        if !isnothing(tau)
            @argcheck(tau > zero(tau))
        end
        return new{typeof(pe), typeof(mp), typeof(views), typeof(sets), typeof(views_conf),
                   typeof(rf), typeof(tau)}(pe, mp, views, sets, views_conf, rf, tau)
    end
end
function BlackLittermanPrior(;
                             pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(;
                                                                                        me = EquilibriumExpectedReturns()),
                             mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                             views::Union{<:LinearConstraintEstimator,
                                          <:BlackLittermanViews},
                             sets::Union{Nothing, <:AssetSets} = nothing,
                             views_conf::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                             rf::Real = 0.0, tau::Union{Nothing, <:Real} = nothing)
    return BlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau)
end
function Base.getproperty(obj::BlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function factory(pe::BlackLittermanPrior, w::Union{Nothing, <:AbstractWeights} = nothing)
    return BlackLittermanPrior(; pe = factory(pe.pe, w), mp = pe.mp, views = pe.views,
                               sets = pe.sets, views_conf = pe.views_conf, rf = pe.rf,
                               tau = pe.tau)
end
"""
```julia
calc_omega(views_conf::Union{Nothing, <:Real, <:AbstractVector{<:Real}}, P::AbstractMatrix,
           sigma::AbstractMatrix)
```

Compute the Black-Litterman view uncertainty matrix `Ω`.

This method constructs the view uncertainty matrix `Ω` for the Black-Litterman model when no explicit view confidences are provided (`views_conf = nothing`). The uncertainty for each view is set to the variance of the projected prior covariance, i.e., `Ω = diag(P * Σ * P')`, where `P` is the view matrix and `Σ` is the prior covariance matrix.

# Arguments

  - `views_conf`:

      + `::Nothing`: Indicates no view confidence is specified, `Diagonal(P * sigma * transpose(P))`.
      + `::Real`: Scalar confidence level applied uniformly to all views, `(1/v - 1) * Diagonal(P * sigma * transpose(P))`, where `v` is the view confidence level.
      + `::AbstractVector{<:Real}`: Vector of confidence levels for each view, `(1 ./ v - 1) * Diag(P * Σ * P')`.

  - `P::AbstractMatrix`: The view matrix (views × assets).
  - `sigma::AbstractMatrix`: The prior covariance matrix (assets × assets).

# Returns

  - `omega::Diagonal`: Diagonal matrix of view uncertainties.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`calc_omega`](@ref)
"""
function calc_omega(::Nothing, P::AbstractMatrix, sigma::AbstractMatrix)
    return Diagonal(P * sigma * transpose(P))
end
function calc_omega(views_conf::Real, P::AbstractMatrix, sigma::AbstractMatrix)
    alphas = inv(views_conf) - one(eltype(views_conf))
    return Diagonal(alphas .* P * sigma * transpose(P))
end
function calc_omega(views_conf::AbstractVector, P::AbstractMatrix, sigma::AbstractMatrix)
    alphas = inv.(views_conf) .- one(eltype(views_conf))
    return Diagonal(alphas .* P * sigma * transpose(P))
end
function vanilla_posteriors(tau::Real, rf::Real, prior_mu::AbstractVector,
                            prior_sigma::AbstractMatrix, omega::AbstractMatrix,
                            P::AbstractMatrix, Q::AbstractVector)
    v1 = tau * prior_sigma * transpose(P)
    v2 = P * v1 + omega
    v3 = Q - P * prior_mu
    posterior_mu = (prior_mu + v1 * (v2 \ v3)) .+ rf
    posterior_sigma = prior_sigma + tau * prior_sigma - v1 * (v2 \ transpose(v1))
    return posterior_mu, posterior_sigma
end
function prior(pe::BlackLittermanPrior, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    @argcheck(length(pe.sets.dict[pe.sets.key]) == size(X, 2))
    prior_model = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_mu, prior_sigma = prior_model.X, prior_model.mu, prior_model.sigma
    (; P, Q) = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
                                     strict = strict)
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    omega = tau * calc_omega(pe.views_conf, P, prior_sigma)
    posterior_mu, posterior_sigma = vanilla_posteriors(tau, pe.rf, prior_mu, prior_sigma,
                                                       omega, P, Q)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma)
end

export BlackLittermanPrior
