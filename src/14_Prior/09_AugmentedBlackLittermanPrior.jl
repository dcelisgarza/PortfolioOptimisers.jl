"""
    struct AugmentedBlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
                                        T14, T15} <: AbstractLowOrderPriorEstimator_F
        a_pe::T1
        f_pe::T2
        mp::T3
        re::T4
        ve::T5
        a_views::T6
        f_views::T7
        a_sets::T8
        f_sets::T9
        a_views_conf::T10
        f_views_conf::T11
        w::T12
        rf::T13
        l::T14
        tau::T15
    end

Augmented Black-Litterman prior estimator for asset returns.

`AugmentedBlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using an augmented Black-Litterman model. It combines asset and factor prior estimators, matrix post-processing, regression and variance estimators, asset and factor views, asset and factor sets, view confidences, weights, risk-free rate, leverage, and a blending parameter `tau`. This estimator supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates joint asset-factor Bayesian updating for posterior inference.

# Fields

  - `a_pe`: Asset prior estimator.
  - `f_pe`: Factor prior estimator.
  - `mp`: Matrix post-processing estimator.
  - `re`: Regression estimator for factor loadings.
  - `ve`: Variance estimator for residuals.
  - `a_views`: Asset views estimator or views object.
  - `f_views`: Factor views estimator or views object.
  - `a_sets`: Asset sets.
  - `f_sets`: Factor sets.
  - `a_views_conf`: Asset view confidence(s).
  - `f_views_conf`: Factor view confidence(s).
  - `w`: Optional weights for assets.
  - `rf`: Risk-free rate.
  - `l`: Optional leverage parameter.
  - `tau`: Blending parameter. When computing the prior, if `nothing`, defaults to `1/T` where `T` is the number of observations.

# Constructor

    AugmentedBlackLittermanPrior(; a_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                 f_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                 mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                 re::AbstractRegressionEstimator = StepwiseRegression(),
                                 ve::AbstractVarianceEstimator = SimpleVariance(),
                                 a_views::Union{<:LinearConstraintEstimator,
                                                <:BlackLittermanViews},
                                 f_views::Union{<:LinearConstraintEstimator,
                                                <:BlackLittermanViews},
                                 a_sets::Union{Nothing, <:AssetSets} = nothing,
                                 f_sets::Union{Nothing, <:AssetSets} = nothing,
                                 a_views_conf::Union{Nothing, <:Number, <:NumVec} = nothing,
                                 f_views_conf::Union{Nothing, <:Number, <:NumVec} = nothing,
                                 w::Union{Nothing, <:NumVec} = nothing, rf::Number = 0.0,
                                 l::Union{Nothing, <:Number} = nothing,
                                 tau::Union{Nothing, <:Number} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is provided, `!isempty(w)`.
  - If `a_views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(a_sets)`.
  - If `f_views` is a [`LinearConstraintEstimator`](@ref), `!isnothing(f_sets)`.
  - If `a_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `f_views_conf` is not `nothing`, validated with [`assert_bl_views_conf`](@ref).
  - If `tau` is not `nothing`, `tau > 0`.

# Examples

```jldoctest
julia> AugmentedBlackLittermanPrior(;
                                    a_sets = AssetSets(; key = "nx",
                                                       dict = Dict("nx" => ["A", "B", "C"])),
                                    f_sets = AssetSets(; key = "fx",
                                                       dict = Dict("fx" => ["F1", "F2"])),
                                    a_views = LinearConstraintEstimator(;
                                                                        val = ["A == 0.03",
                                                                               "B + C == 0.04"]),
                                    f_views = LinearConstraintEstimator(;
                                                                        val = ["F1 == 0.01",
                                                                               "F2 == 0.02"]))
AugmentedBlackLittermanPrior
          a_pe ┼ EmpiricalPrior
               │        ce ┼ PortfolioOptimisersCovariance
               │           │   ce ┼ Covariance
               │           │      │    me ┼ SimpleExpectedReturns
               │           │      │       │   w ┴ nothing
               │           │      │    ce ┼ GeneralCovariance
               │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
               │           │      │       │    w ┴ nothing
               │           │      │   alg ┴ Full()
               │           │   mp ┼ DefaultMatrixProcessing
               │           │      │       pdm ┼ Posdef
               │           │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
               │           │      │   denoise ┼ nothing
               │           │      │    detone ┼ nothing
               │           │      │       alg ┴ nothing
               │        me ┼ SimpleExpectedReturns
               │           │   w ┴ nothing
               │   horizon ┴ nothing
          f_pe ┼ EmpiricalPrior
               │        ce ┼ PortfolioOptimisersCovariance
               │           │   ce ┼ Covariance
               │           │      │    me ┼ SimpleExpectedReturns
               │           │      │       │   w ┴ nothing
               │           │      │    ce ┼ GeneralCovariance
               │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
               │           │      │       │    w ┴ nothing
               │           │      │   alg ┴ Full()
               │           │   mp ┼ DefaultMatrixProcessing
               │           │      │       pdm ┼ Posdef
               │           │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
               │           │      │   denoise ┼ nothing
               │           │      │    detone ┼ nothing
               │           │      │       alg ┴ nothing
               │        me ┼ SimpleExpectedReturns
               │           │   w ┴ nothing
               │   horizon ┴ nothing
            mp ┼ DefaultMatrixProcessing
               │       pdm ┼ Posdef
               │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
               │   denoise ┼ nothing
               │    detone ┼ nothing
               │       alg ┴ nothing
            re ┼ StepwiseRegression
               │     crit ┼ PValue
               │          │   threshold ┴ Float64: 0.05
               │      alg ┼ Forward()
               │   target ┼ LinearModel
               │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
            ve ┼ SimpleVariance
               │          me ┼ SimpleExpectedReturns
               │             │   w ┴ nothing
               │           w ┼ nothing
               │   corrected ┴ Bool: true
       a_views ┼ LinearConstraintEstimator
               │   val ┴ Vector{String}: ["A == 0.03", "B + C == 0.04"]
       f_views ┼ LinearConstraintEstimator
               │   val ┴ Vector{String}: ["F1 == 0.01", "F2 == 0.02"]
        a_sets ┼ AssetSets
               │    key ┼ String: "nx"
               │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
        f_sets ┼ AssetSets
               │    key ┼ String: "fx"
               │   dict ┴ Dict{String, Vector{String}}: Dict("fx" => ["F1", "F2"])
  a_views_conf ┼ nothing
  f_views_conf ┼ nothing
             w ┼ nothing
            rf ┼ Float64: 0.0
             l ┼ nothing
           tau ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`AssetSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
struct AugmentedBlackLittermanPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
                                    T14, T15} <: AbstractLowOrderPriorEstimator_F
    a_pe::T1
    f_pe::T2
    mp::T3
    re::T4
    ve::T5
    a_views::T6
    f_views::T7
    a_sets::T8
    f_sets::T9
    a_views_conf::T10
    f_views_conf::T11
    w::T12
    rf::T13
    l::T14
    tau::T15
    function AugmentedBlackLittermanPrior(a_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          f_pe::AbstractLowOrderPriorEstimator_A_AF,
                                          mp::AbstractMatrixProcessingEstimator,
                                          re::AbstractRegressionEstimator,
                                          ve::AbstractVarianceEstimator,
                                          a_views::Union{<:LinearConstraintEstimator,
                                                         <:BlackLittermanViews},
                                          f_views::Union{<:LinearConstraintEstimator,
                                                         <:BlackLittermanViews},
                                          a_sets::Union{Nothing, <:AssetSets},
                                          f_sets::Union{Nothing, <:AssetSets},
                                          a_views_conf::Union{Nothing, <:Number, <:NumVec},
                                          f_views_conf::Union{Nothing, <:Number, <:NumVec},
                                          w::Union{Nothing, <:NumVec}, rf::Number,
                                          l::Union{Nothing, <:Number},
                                          tau::Union{Nothing, <:Number})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        if isa(a_views, LinearConstraintEstimator)
            @argcheck(!isnothing(a_sets))
        end
        if isa(f_views, LinearConstraintEstimator)
            @argcheck(!isnothing(f_sets))
        end
        assert_bl_views_conf(a_views_conf, a_views)
        assert_bl_views_conf(f_views_conf, f_views)
        if !isnothing(tau)
            @argcheck(tau > zero(tau))
        end
        return new{typeof(a_pe), typeof(f_pe), typeof(mp), typeof(re), typeof(ve),
                   typeof(a_views), typeof(f_views), typeof(a_sets), typeof(f_sets),
                   typeof(a_views_conf), typeof(f_views_conf), typeof(w), typeof(rf),
                   typeof(l), typeof(tau)}(a_pe, f_pe, mp, re, ve, a_views, f_views, a_sets,
                                           f_sets, a_views_conf, f_views_conf, w, rf, l,
                                           tau)
    end
end
function AugmentedBlackLittermanPrior(;
                                      a_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                      f_pe::AbstractLowOrderPriorEstimator_A_AF = EmpiricalPrior(),
                                      mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                                      re::AbstractRegressionEstimator = StepwiseRegression(),
                                      ve::AbstractVarianceEstimator = SimpleVariance(),
                                      a_views::Union{<:LinearConstraintEstimator,
                                                     <:BlackLittermanViews},
                                      f_views::Union{<:LinearConstraintEstimator,
                                                     <:BlackLittermanViews},
                                      a_sets::Union{Nothing, <:AssetSets} = nothing,
                                      f_sets::Union{Nothing, <:AssetSets} = nothing,
                                      a_views_conf::Union{Nothing, <:Number, <:NumVec} = nothing,
                                      f_views_conf::Union{Nothing, <:Number, <:NumVec} = nothing,
                                      w::Union{Nothing, <:NumVec} = nothing,
                                      rf::Number = 0.0,
                                      l::Union{Nothing, <:Number} = nothing,
                                      tau::Union{Nothing, <:Number} = nothing)
    return AugmentedBlackLittermanPrior(a_pe, f_pe, mp, re, ve, a_views, f_views, a_sets,
                                        f_sets, a_views_conf, f_views_conf, w, rf, l, tau)
end
function factory(pe::AugmentedBlackLittermanPrior, w::Union{Nothing, <:NumVec} = nothing)
    return AugmentedBlackLittermanPrior(; a_pe = factory(pe.a_pe, w),
                                        f_pe = factory(pe.f_pe, w), mp = pe.mp,
                                        re = factory(pe.re, w), ve = factory(pe.ve, w),
                                        a_views = pe.a_views, f_views = pe.f_views,
                                        a_sets = pe.a_sets, f_sets = pe.f_sets,
                                        a_views_conf = pe.a_views_conf,
                                        f_views_conf = pe.f_views_conf, w = pe.w,
                                        rf = pe.rf, l = pe.l, tau = pe.tau)
end
function Base.getproperty(obj::AugmentedBlackLittermanPrior, sym::Symbol)
    return if sym == :me
        obj.a_pe.me
    elseif sym == :ce
        obj.a_pe.ce
    elseif sym == :f_me
        obj.f_pe.me
    elseif sym == :f_ce
        obj.f_pe.ce
    else
        getfield(obj, sym)
    end
end
"""
    prior(pe::AugmentedBlackLittermanPrior, X::NumMat, F::NumMat; dims::Int = 1,
          strict::Bool = false, kwargs...)

Compute augmented Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the augmented Black-Litterman model, combining asset and factor prior estimators, matrix post-processing, regression and variance estimators, asset and factor views, asset and factor sets, view confidences, weights, risk-free rate, leverage, and a blending parameter `tau`. This method supports both direct and constraint-based views, flexible confidence specification, and matrix processing, and incorporates joint asset-factor Bayesian updating for posterior inference.

# Arguments

  - `pe`: Augmented Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Factor matrix (observations × factors).
  - `dims`: Dimension along which to compute moments (`1` = columns/assets, `2` = rows). Default is `1`.
  - `strict`: If `true`, enforce strict validation of views and sets. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and matrix processing.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, regression result, and factor prior details.

# Validation

  - `dims in (1, 2)`.
  - `length(pe.a_sets.dict[pe.a_sets.key]) == size(X, 2)`.
  - `length(pe.f_sets.dict[pe.f_sets.key]) == size(F, 2)`.
  - If `pe.w` is provided, `length(pe.w) == size(X, 2)`.

# Details

  - If `dims == 2`, `X` and `F` are transposed to ensure assets/factors are in columns.
  - Asset and factor priors are computed using the embedded prior estimators `pe.a_pe` and `pe.f_pe`.
  - Factor regression is performed using the regression estimator `pe.re`.
  - Asset and factor views are extracted using `black_litterman_views`, which returns the view matrices and view returns vectors.
  - `tau` defaults to `1/T` if not specified, where `T` is the number of observations.
  - View uncertainty matrices for assets and factors are computed using `calc_omega`.
  - The augmented prior mean and covariance are constructed by combining asset and factor priors and regression loadings.
  - The augmented Black-Litterman posterior mean and covariance are computed using `vanilla_posteriors`.
  - Matrix processing is applied to the augmented posterior covariance.
  - The final asset posterior mean and covariance are extracted from the augmented results and adjusted for regression intercepts and risk-free rate.
  - The result includes asset returns, posterior mean, posterior covariance, regression result, and factor prior details.

# Related

  - [`AugmentedBlackLittermanPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
  - [`calc_omega`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function prior(pe::AugmentedBlackLittermanPrior, X::NumMat, F::NumMat; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        F = transpose(F)
    end
    @argcheck(length(pe.a_sets.dict[pe.a_sets.key]) == size(X, 2))
    @argcheck(length(pe.f_sets.dict[pe.f_sets.key]) == size(F, 2))
    # Asset prior.
    a_prior = prior(pe.a_pe, X; strict = strict, kwargs...)
    a_prior_mu, a_prior_sigma = a_prior.mu, a_prior.sigma
    # Factor prior.
    f_prior = prior(pe.f_pe, F; strict = strict)
    f_prior_mu, f_prior_sigma = f_prior.mu, f_prior.sigma
    # Black litterman on the factors.
    rr = regression(pe.re, X, F)
    (; b, M) = rr
    posterior_X = F * transpose(M) .+ transpose(b)
    (; P, Q) = black_litterman_views(pe.a_views, pe.a_sets; datatype = eltype(posterior_X),
                                     strict = strict)
    f_views = black_litterman_views(pe.f_views, pe.f_sets; datatype = eltype(posterior_X),
                                    strict = strict)
    f_P, f_Q = f_views.P, f_views.Q
    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    a_omega = tau * calc_omega(pe.a_views_conf, P, a_prior_sigma)
    f_omega = tau * calc_omega(pe.f_views_conf, f_P, f_prior_sigma)
    aug_prior_sigma = hcat(vcat(a_prior_sigma, f_prior_sigma * transpose(M)),
                           vcat(M * f_prior_sigma, f_prior_sigma))
    aug_P = hcat(vcat(P, zeros(size(f_P, 1), size(P, 2))),
                 vcat(zeros(size(P, 1), size(f_P, 2)), f_P))
    aug_Q = vcat(Q, f_Q)
    aug_omega = hcat(vcat(a_omega, zeros(size(f_omega, 1), size(a_omega, 1))),
                     vcat(zeros(size(a_omega, 1), size(f_omega, 1)), f_omega))
    aug_prior_mu = if !isnothing(pe.l)
        w = if !isnothing(pe.w)
            @argcheck(length(pe.w) == size(X, 2))
            pe.w
        else
            iN = inv(size(X, 2))
            range(iN, iN; length = size(X, 2))
        end
        pe.l * (vcat(a_prior_sigma, f_prior_sigma * transpose(M))) * w
    else
        vcat(a_prior_mu, f_prior_mu) .- pe.rf
    end
    aug_posterior_mu, aug_posterior_sigma = vanilla_posteriors(tau, pe.rf, aug_prior_mu,
                                                               aug_prior_sigma, aug_omega,
                                                               aug_P, aug_Q)
    matrix_processing!(pe.mp, aug_posterior_sigma, hcat(posterior_X, F))
    posterior_mu = (aug_posterior_mu[1:size(X, 2)] + b) .+ pe.rf
    posterior_sigma = aug_posterior_sigma[1:size(X, 2), 1:size(X, 2)]
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma,
                         rr = rr, f_mu = f_prior_mu, f_sigma = f_prior_sigma,
                         f_w = f_prior.w)
end

export AugmentedBlackLittermanPrior
