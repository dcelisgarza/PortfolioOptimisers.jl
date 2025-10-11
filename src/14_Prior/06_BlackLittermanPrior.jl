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
  - `tau`: Blending parameter. When computing the prior if `nothing`, defaults to `1/T` where `T` is the number of observations.

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

# Examples

```jldoctest
julia> BlackLittermanPrior(; sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"])),
                           views = LinearConstraintEstimator(;
                                                             val = ["A == 0.03", "B + C == 0.04"]))
BlackLittermanPrior
          pe | EmpiricalPrior
             |        ce | PortfolioOptimisersCovariance
             |           |   ce | Covariance
             |           |      |    me | SimpleExpectedReturns
             |           |      |       |   w | nothing
             |           |      |    ce | GeneralCovariance
             |           |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             |           |      |       |    w | nothing
             |           |      |   alg | Full()
             |           |   mp | DefaultMatrixProcessing
             |           |      |       pdm | Posdef
             |           |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
             |           |      |   denoise | nothing
             |           |      |    detone | nothing
             |           |      |       alg | nothing
             |        me | EquilibriumExpectedReturns
             |           |   ce | PortfolioOptimisersCovariance
             |           |      |   ce | Covariance
             |           |      |      |    me | SimpleExpectedReturns
             |           |      |      |       |   w | nothing
             |           |      |      |    ce | GeneralCovariance
             |           |      |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             |           |      |      |       |    w | nothing
             |           |      |      |   alg | Full()
             |           |      |   mp | DefaultMatrixProcessing
             |           |      |      |       pdm | Posdef
             |           |      |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
             |           |      |      |   denoise | nothing
             |           |      |      |    detone | nothing
             |           |      |      |       alg | nothing
             |           |    w | nothing
             |           |    l | Int64: 1
             |   horizon | nothing
          mp | DefaultMatrixProcessing
             |       pdm | Posdef
             |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
             |   denoise | nothing
             |    detone | nothing
             |       alg | nothing
       views | LinearConstraintEstimator
             |   val | Vector{String}: ["A == 0.03", "B + C == 0.04"]
        sets | AssetSets
             |    key | String: "nx"
             |   dict | Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
  views_conf | nothing
          rf | Float64: 0.0
         tau | nothing
```

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

  - `P`: The view matrix (views × assets).
  - `sigma`: The prior covariance matrix (assets × assets).

# Returns

  - `omega::Diagonal`: Diagonal matrix of view uncertainties.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`vanilla_posteriors`](@ref)
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
"""
```julia
vanilla_posteriors(tau::Real, rf::Real, prior_mu::AbstractVector,
                   prior_sigma::AbstractMatrix, omega::AbstractMatrix, P::AbstractMatrix,
                   Q::AbstractVector)
```

Compute the Black-Litterman posterior mean and covariance for asset returns.

`vanilla_posteriors` implements the standard Black-Litterman update equations, combining the prior mean and covariance with user or algorithmic views. The function returns the posterior mean and covariance matrix, incorporating the blending parameter `tau`, risk-free rate `rf`, view uncertainty matrix `omega`, view matrix `P`, and view returns vector `Q`.

# Arguments

  - `tau`: Scalar blending parameter for prior and views.
  - `rf`: Risk-free rate to add to the posterior mean.
  - `prior_mu`: Prior mean vector of asset returns.
  - `prior_sigma`: Prior covariance matrix of asset returns.
  - `omega`: View uncertainty matrix.
  - `P`: View matrix (views × assets).
  - `Q`: Vector of view returns (views).

# Returns

  - `posterior_mu::Vector{<:Real}`: Posterior mean vector of asset returns.
  - `posterior_sigma::Matrix{<:Real}`: Posterior covariance matrix of asset returns.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`calc_omega`](@ref)
"""
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
"""
```julia
prior(pe::BlackLittermanPrior, X::AbstractMatrix;
      F::Union{Nothing, <:AbstractMatrix} = nothing, dims::Int = 1, strict::Bool = false,
      kwargs...)
```

Compute the Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the Black-Litterman model, combining a prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and blending parameter `tau`. The method supports both direct and constraint-based views, flexible confidence specification, and matrix processing.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F{Nothing, <:AbstractMatrix}`: Optional factor matrix (default: `nothing`).
  - `dims`: Dimension along which to compute moments (`1` = columns/assets, `2` = rows). Default is `1`.
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, and posterior covariance matrix.

# Validation

  - `dims in (1, 2)`.
  - `length(pe.sets.dict[pe.sets.key]) == size(X, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets are in columns.
  - The prior model is computed using the embedded prior estimator `pe.pe`.
  - Views are extracted using [`black_litterman_views`](@ref), which returns the view matrix `P` and view returns vector `Q`.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of observations.
  - The view uncertainty matrix `omega` is computed using [`calc_omega`](@ref).
  - The posterior mean and covariance are computed using [`vanilla_posteriors`](@ref).
  - Matrix processing is applied to the posterior covariance and asset returns using the embedded matrix processing estimator `pe.mp`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
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
