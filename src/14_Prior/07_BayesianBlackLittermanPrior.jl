"""
    struct BayesianBlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7} <:
           AbstractLowOrderPriorEstimator_F
        pe::T1
        mp::T2
        views::T3
        sets::T4
        views_conf::T5
        rf::T6
        tau::T7
    end

Bayesian Black-Litterman prior estimator for asset returns.

`BayesianBlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using a Bayesian Black-Litterman model. It combines a factor prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and a blending parameter `tau`. This estimator supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates Bayesian updating for posterior inference.

# Fields

  - `pe`: Factor prior estimator.
  - `mp`: Matrix post-processing estimator.
  - `views`: Views estimator or views object.
  - `sets`: Asset sets.
  - `views_conf`: View confidence(s).
  - `rf`: Risk-free rate.
  - `tau`: Blending parameter. When computing the prior, if `nothing`, defaults to `1/T` where `T` is the number of factor observations.

# Constructor

    BayesianBlackLittermanPrior(;
                                pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(;
                                                                                      pe = EmpiricalPrior(;
                                                                                                          me = EquilibriumExpectedReturns())),
                                mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                views::Union{<:LinearConstraintEstimator,
                                             <:BlackLittermanViews},
                                sets::Union{Nothing, <:AssetSets} = nothing,
                                views_conf::Union{Nothing, <:Number, <:NumVec} = nothing,
                                rf::Number = 0.0, tau::Union{Nothing, <:Number} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - If `views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(sets)`.
  - If `views_conf` is not `nothing`, `views_conf` is validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

# Examples

```jldoctest
julia> BayesianBlackLittermanPrior(;
                                   sets = AssetSets(; key = "nx",
                                                    dict = Dict("nx" => ["A", "B", "C"])),
                                   views = LinearConstraintEstimator(;
                                                                     val = ["A == 0.03",
                                                                            "B + C == 0.04"]))
BayesianBlackLittermanPrior
          pe ┼ FactorPrior
             │    pe ┼ EmpiricalPrior
             │       │        ce ┼ PortfolioOptimisersCovariance
             │       │           │   ce ┼ Covariance
             │       │           │      │    me ┼ SimpleExpectedReturns
             │       │           │      │       │   w ┴ nothing
             │       │           │      │    ce ┼ GeneralCovariance
             │       │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │       │           │      │       │    w ┴ nothing
             │       │           │      │   alg ┴ Full()
             │       │           │   mp ┼ DefaultMatrixProcessing
             │       │           │      │       pdm ┼ Posdef
             │       │           │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
             │       │           │      │   denoise ┼ nothing
             │       │           │      │    detone ┼ nothing
             │       │           │      │       alg ┴ nothing
             │       │        me ┼ EquilibriumExpectedReturns
             │       │           │   ce ┼ PortfolioOptimisersCovariance
             │       │           │      │   ce ┼ Covariance
             │       │           │      │      │    me ┼ SimpleExpectedReturns
             │       │           │      │      │       │   w ┴ nothing
             │       │           │      │      │    ce ┼ GeneralCovariance
             │       │           │      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │       │           │      │      │       │    w ┴ nothing
             │       │           │      │      │   alg ┴ Full()
             │       │           │      │   mp ┼ DefaultMatrixProcessing
             │       │           │      │      │       pdm ┼ Posdef
             │       │           │      │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
             │       │           │      │      │   denoise ┼ nothing
             │       │           │      │      │    detone ┼ nothing
             │       │           │      │      │       alg ┴ nothing
             │       │           │    w ┼ nothing
             │       │           │    l ┴ Int64: 1
             │       │   horizon ┴ nothing
             │    mp ┼ DefaultMatrixProcessing
             │       │       pdm ┼ Posdef
             │       │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
             │       │   denoise ┼ nothing
             │       │    detone ┼ nothing
             │       │       alg ┴ nothing
             │    re ┼ StepwiseRegression
             │       │     crit ┼ PValue
             │       │          │   threshold ┴ Float64: 0.05
             │       │      alg ┼ Forward()
             │       │   target ┼ LinearModel
             │       │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │    ve ┼ SimpleVariance
             │       │          me ┼ SimpleExpectedReturns
             │       │             │   w ┴ nothing
             │       │           w ┼ nothing
             │       │   corrected ┴ Bool: true
             │   rsd ┴ Bool: true
          mp ┼ DefaultMatrixProcessing
             │       pdm ┼ Posdef
             │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
             │   denoise ┼ nothing
             │    detone ┼ nothing
             │       alg ┴ nothing
       views ┼ LinearConstraintEstimator
             │   val ┴ Vector{String}: ["A == 0.03", "B + C == 0.04"]
        sets ┼ AssetSets
             │    key ┼ String: "nx"
             │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
  views_conf ┼ nothing
          rf ┼ Float64: 0.0
         tau ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`FactorPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`AssetSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
struct BayesianBlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7} <:
       AbstractLowOrderPriorEstimator_F
    pe::T1
    mp::T2
    views::T3
    sets::T4
    views_conf::T5
    rf::T6
    tau::T7
    function BayesianBlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_F_AF,
                                         mp::AbstractMatrixProcessingEstimator,
                                         views::Union{<:LinearConstraintEstimator,
                                                      <:BlackLittermanViews},
                                         sets::Union{Nothing, <:AssetSets},
                                         views_conf::Union{Nothing, <:Number, <:NumVec},
                                         rf::Number, tau::Union{Nothing, <:Number})
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
function BayesianBlackLittermanPrior(;
                                     pe::AbstractLowOrderPriorEstimator_F_AF = FactorPrior(;
                                                                                           pe = EmpiricalPrior(;
                                                                                                               me = EquilibriumExpectedReturns())),
                                     mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                     views::Union{<:LinearConstraintEstimator,
                                                  <:BlackLittermanViews},
                                     sets::Union{Nothing, <:AssetSets} = nothing,
                                     views_conf::Union{Nothing, <:Number, <:NumVec} = nothing,
                                     rf::Number = 0.0,
                                     tau::Union{Nothing, <:Number} = nothing)
    return BayesianBlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau)
end
function factory(pe::BayesianBlackLittermanPrior, w::Option{<:AbstractWeights} = nothing)
    return BayesianBlackLittermanPrior(; pe = factory(pe.pe, w), mp = pe.mp,
                                       views = pe.views, sets = pe.sets,
                                       views_conf = pe.views_conf, rf = pe.rf, tau = pe.tau)
end
function Base.getproperty(obj::BayesianBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
"""
    prior(pe::BayesianBlackLittermanPrior, X::NumMat, F::NumMat; dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute Bayesian Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the Bayesian Black-Litterman model, combining a factor prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and blending parameter `tau`. This method supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates Bayesian updating for posterior inference.

# Arguments

  - `pe`: Bayesian Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - `dims`: Dimension along which to compute moments (`1` = columns/assets, `2` = rows). Default is `1`.
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, and factor prior details.

# Validation

  - `dims in (1, 2)`.
  - `length(pe.sets.dict[pe.sets.key]) == size(F, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - The factor prior is computed using the embedded prior estimator `pe.pe`.
  - Views are extracted using [`black_litterman_views`](@ref), which returns the view matrix `P` and view returns vector `Q`.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of factor observations.
  - The view uncertainty matrix `f_omega` is computed using [`calc_omega`](@ref).
  - Bayesian posterior mean and covariance are computed via the model's update equations.
  - Matrix processing is applied to the posterior covariance and asset returns using the embedded matrix processing estimator `pe.mp`.
  - The result includes factor prior mean and covariance, and regression details.

# Related

  - [`BayesianBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
"""
function prior(pe::BayesianBlackLittermanPrior, X::NumMat, F::NumMat; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.sets.dict[pe.sets.key]) == size(F, 2))
    prior_result = prior(pe.pe, X, F; strict = strict, kwargs...)
    posterior_X, prior_sigma, f_mu, f_sigma, rr = prior_result.X, prior_result.sigma,
                                                  prior_result.f_mu, prior_result.f_sigma,
                                                  prior_result.rr
    (; P, Q) = black_litterman_views(pe.views, pe.sets; datatype = eltype(posterior_X),
                                     strict = strict)
    tau = isnothing(pe.tau) ? inv(size(F, 1)) : pe.tau
    f_omega = tau * calc_omega(pe.views_conf, P, f_sigma)
    (; b, M) = rr
    sigma_hat = f_sigma \ I + transpose(P) * (f_omega \ P)
    mu_hat = sigma_hat \ (f_sigma \ f_mu + transpose(P) * (f_omega \ Q))
    v1 = prior_sigma \ M
    v2 = sigma_hat + transpose(M) * v1
    v3 = prior_sigma \ I
    posterior_sigma = (v3 - v1 * (v2 \ transpose(M)) * v3) \ I
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    posterior_mu = (posterior_sigma * v1 * (v2 \ sigma_hat) * mu_hat + b) .+ pe.rf
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         rr = rr, f_mu = f_mu, f_sigma = f_sigma)
end

export BayesianBlackLittermanPrior
