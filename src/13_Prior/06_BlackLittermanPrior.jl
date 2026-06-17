"""
$(DocStringExtensions.TYPEDEF)

Black-Litterman prior estimator for asset returns.

`BlackLittermanPrior` is a low order prior estimator that computes the mean and covariance of asset returns using the Black-Litterman model. It combines a prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and a blending parameter `tau`. The estimator supports both direct and constraint-based views, and allows for flexible confidence specification and matrix processing.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BlackLittermanPrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(;
            me = EquilibriumExpectedReturns()
        ),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
        views::Lc_BLV,
        sets::Option{<:AssetSets} = nothing,
        views_conf::Option{<:Num_VecNum} = nothing,
        rf::Number = 0.0,
        tau::Option{<:Number} = nothing
    ) -> BlackLittermanPrior

Keywords correspond to the struct's fields.

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
          pe ┼ EmpiricalPrior
             │        ce ┼ PortfolioOptimisersCovariance
             │           │   ce ┼ Covariance
             │           │      │    me ┼ SimpleExpectedReturns
             │           │      │       │   w ┴ nothing
             │           │      │    ce ┼ GeneralCovariance
             │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │           │      │       │    w ┴ nothing
             │           │      │   alg ┴ Full()
             │           │   mp ┼ MatrixProcessing
             │           │      │     pdm ┼ Posdef
             │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │           │      │      dn ┼ nothing
             │           │      │      dt ┼ nothing
             │           │      │     alg ┼ nothing
             │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │        me ┼ EquilibriumExpectedReturns
             │           │   ce ┼ PortfolioOptimisersCovariance
             │           │      │   ce ┼ Covariance
             │           │      │      │    me ┼ SimpleExpectedReturns
             │           │      │      │       │   w ┴ nothing
             │           │      │      │    ce ┼ GeneralCovariance
             │           │      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │           │      │      │       │    w ┴ nothing
             │           │      │      │   alg ┴ Full()
             │           │      │   mp ┼ MatrixProcessing
             │           │      │      │     pdm ┼ Posdef
             │           │      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │           │      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │           │      │      │      dn ┼ nothing
             │           │      │      │      dt ┼ nothing
             │           │      │      │     alg ┼ nothing
             │           │      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │           │    w ┼ nothing
             │           │    l ┴ Int64: 1
             │   horizon ┴ nothing
          mp ┼ MatrixProcessing
             │     pdm ┼ Posdef
             │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      dn ┼ nothing
             │      dt ┼ nothing
             │     alg ┼ nothing
             │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
       views ┼ LinearConstraintEstimator
             │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
             │   key ┴ nothing
        sets ┼ AssetSets
             │    key ┼ String: "nx"
             │   ukey ┼ String: "ux"
             │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
  views_conf ┼ nothing
          rf ┼ Float64: 0.0
         tau ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`EmpiricalPrior`](@ref)
  - [`BlackLittermanViews`](@ref)
  - [`AssetSets`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
@propagatable @concrete struct BlackLittermanPrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:mp])
    """
    mp
    """
    $(field_dict[:views])
    """
    views
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:views_conf])
    """
    views_conf
    """
    $(field_dict[:rf])
    """
    rf
    """
    $(field_dict[:tau])
    """
    tau
    function BlackLittermanPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mp::AbstractMatrixProcessingEstimator, views::Lc_BLV,
                                 sets::Option{<:AssetSets},
                                 views_conf::Option{<:Num_VecNum}, rf::Number,
                                 tau::Option{<:Number})
        assert_bl(views, sets, views_conf, tau)
        return new{typeof(pe), typeof(mp), typeof(views), typeof(sets), typeof(views_conf),
                   typeof(rf), typeof(tau)}(pe, mp, views, sets, views_conf, rf, tau)
    end
end
function BlackLittermanPrior(;
                             pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(;
                                                                                        me = EquilibriumExpectedReturns()),
                             mp::AbstractMatrixProcessingEstimator = MatrixProcessing(),
                             views::Lc_BLV, sets::Option{<:AssetSets} = nothing,
                             views_conf::Option{<:Num_VecNum} = nothing, rf::Number = 0.0,
                             tau::Option{<:Number} = nothing)::BlackLittermanPrior
    return BlackLittermanPrior(pe, mp, views, sets, views_conf, rf, tau)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties BlackLittermanPrior begin
    forward(pe, me, ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that the Black-Litterman prior's views, sets, view confidences, and blending parameter are valid.
"""
function assert_bl(views::Lc_BLV, sets::Option{<:AssetSets},
                   views_conf::Option{<:Num_VecNum}, tau::Option{<:Number})
    if isa(views, LinearConstraintEstimator)
        @argcheck(!isnothing(sets))
    end
    assert_bl_views_conf(views_conf, views)
    if !isnothing(tau)
        @argcheck(tau > zero(tau))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pre-compute shared Black-Litterman inputs from views, prior covariance, and blending parameters.

Extracts the view matrix `P`, view returns vector `Q`, and excluded indices from `views` and `sets` via [`black_litterman_views`](@ref), resolves `tau`, filters excluded rows from `views_conf` via [`remove_excl_views`](@ref), and computes the scaled uncertainty matrix `omega = tau * Ω` via [`calc_omega`](@ref).

# Arguments

  - $(arg_dict[:views])
  - $(arg_dict[:sets])
  - $(arg_dict[:views_conf])
  - `prior_sigma::MatNum`: Prior covariance matrix of asset returns `assets × assets`.
  - `pe_tau::Option{<:Number}`: Optional user-specified blending parameter. If `nothing`, defaults to `1/T`.
  - `T::Integer`: Number of observations used to compute the default `tau = 1/T`.
  - $(arg_dict[:datatype])
  - $(arg_dict[:strict])

# Returns

  - `(; P, Q, tau, omega)`: Named tuple where:

      + `P::MatNum`: View matrix `views × assets`.
      + `Q::VecNum`: View returns vector `views × 1`.
      + `tau::Number`: Resolved blending parameter.
      + `omega::LinearAlgebra.Diagonal`: Scaled view uncertainty matrix `tau * Ω`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`black_litterman_views`](@ref)
  - [`calc_omega`](@ref)
  - [`remove_excl_views`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function bl_preroll(views, sets, views_conf, prior_sigma, pe_tau, T, datatype, strict)
    (; P, Q, excl) = black_litterman_views(views, sets; datatype = datatype,
                                           strict = strict)
    tau = isnothing(pe_tau) ? inv(T) : pe_tau
    views_conf = remove_excl_views(views_conf, excl)
    omega = tau * calc_omega(views_conf, P, prior_sigma)
    return (; P, Q, tau, omega)
end
"""
    calc_omega(views_conf::Option{<:Num_VecNum}, P::MatNum,
               sigma::MatNum) -> LinearAlgebra.Diagonal

Compute the Black-Litterman view uncertainty matrix `Ω`.

# Mathematical definition

Let ``\\mathbf{P}`` be the ``K \\times N`` view matrix and ``\\mathbf{\\Sigma}`` the ``N \\times N`` prior covariance matrix. The view uncertainty matrix ``\\mathbf{\\Omega}`` for each `views_conf` variant is:

```math
\\begin{align}
\\mathbf{\\Omega} &= \\mathrm{Diag}(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^\\intercal) \\quad (\\text{no confidence})\\,.
\\end{align}
```

```math
\\begin{align}
\\mathbf{\\Omega} &= \\left(\\frac{1}{v} - 1\\right) \\mathrm{Diag}(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^\\intercal) \\quad (\\text{scalar confidence } v)\\,.
\\end{align}
```

```math
\\begin{align}
\\mathbf{\\Omega} &= \\mathrm{Diag}\\!\\left(\\left(\\frac{1}{\\boldsymbol{v}} - \\boldsymbol{1}\\right) \\odot \\mathrm{diag}(\\mathbf{P} \\mathbf{\\Sigma} \\mathbf{P}^\\intercal)\\right) \\quad (\\text{vector confidence } \\boldsymbol{v})\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Omega}``: ``K \\times K`` diagonal view uncertainty matrix.
  - ``\\mathbf{P}``: ``K \\times N`` views matrix.
  - ``\\mathbf{\\Sigma}``: ``N \\times N`` prior covariance matrix.
  - ``v``: Scalar view confidence level.
  - ``\\boldsymbol{v}``: ``K \\times 1`` vector of view confidence levels.
  - ``\\odot``: Element-wise multiplication.

# Arguments

  - `views_conf`:

      + `::Nothing`: No confidence specified; `Ω = Diag(P * sigma * P')`.
      + `::Number`: Scalar confidence `v`; `Ω = (1/v - 1) * Diag(P * sigma * P')`.
      + `::VecNum`: Per-view confidences `v`; `Ω = Diag((1 ./ v .- 1) .* diag(P * sigma * P'))`.

  - $(arg_dict[:P])

  - $(arg_dict[:sigma])

# Returns

  - `omega::LinearAlgebra.Diagonal`: Diagonal view uncertainty matrix `views × views`.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`bl_preroll`](@ref)
  - [`vanilla_posteriors`](@ref)
"""
function calc_omega(::Nothing, P::MatNum, sigma::MatNum)
    return LinearAlgebra.Diagonal(P * sigma * transpose(P))
end
function calc_omega(views_conf::Number, P::MatNum, sigma::MatNum)
    alphas = inv(views_conf) - one(eltype(views_conf))
    return LinearAlgebra.Diagonal(alphas .* P * sigma * transpose(P))
end
function calc_omega(views_conf::VecNum, P::MatNum, sigma::MatNum)
    alphas = inv.(views_conf) .- one(eltype(views_conf))
    return LinearAlgebra.Diagonal(alphas .* P * sigma * transpose(P))
end
"""
    vanilla_posteriors(tau::Number, rf::Number, prior_mu::VecNum,
                       prior_sigma::MatNum, omega::MatNum, P::MatNum,
                       Q::VecNum)

Compute the Black-Litterman posterior mean and covariance for asset returns.

`vanilla_posteriors` implements the standard Black-Litterman update equations, combining the prior mean and covariance with user or algorithmic views. The function returns the posterior mean and covariance matrix, incorporating the blending parameter `tau`, risk-free rate `rf`, view uncertainty matrix `omega`, view matrix `P`, and view returns vector `Q`.

# Mathematical definition

Let ``\\boldsymbol{\\Pi}`` be the prior mean, ``\\mathbf{\\Sigma}`` the prior covariance, ``\\tau`` the scaling parameter, ``\\mathbf{P}`` the view matrix, ``\\boldsymbol{q}`` the view vector, and ``\\mathbf{\\Omega}`` the view uncertainty matrix:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{BL} &= \\boldsymbol{\\Pi} + \\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal \\left(\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}\\right)^{-1} (\\boldsymbol{q} - \\mathbf{P}\\boldsymbol{\\Pi}) + r_f\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{BL} &= \\mathbf{\\Sigma} + \\tau\\mathbf{\\Sigma} - \\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal \\left(\\mathbf{P}\\tau\\mathbf{\\Sigma}\\mathbf{P}^\\intercal + \\mathbf{\\Omega}\\right)^{-1} \\mathbf{P}\\tau\\mathbf{\\Sigma}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{BL}``: Black-Litterman posterior mean vector.
  - ``\\hat{\\mathbf{\\Sigma}}_{BL}``: Black-Litterman posterior covariance matrix.
  - ``\\boldsymbol{\\Pi}``: ``N \\times 1`` prior (equilibrium) expected returns.
  - ``\\mathbf{\\Sigma}``: ``N \\times N`` prior covariance matrix.
  - ``\\tau``: Scaling parameter for the uncertainty in the prior.
  - ``\\mathbf{P}``: ``K \\times N`` views matrix.
  - ``\\boldsymbol{q}``: ``K \\times 1`` views vector.
  - ``\\mathbf{\\Omega}``: ``K \\times K`` view uncertainty matrix.
  - ``r_f``: Risk-free rate.

# Arguments

  - `tau`: Scalar blending parameter for prior and views.
  - `rf`: Risk-free rate to add to the posterior mean.
  - `prior_mu`: Prior mean vector of asset returns.
  - `prior_sigma`: Prior covariance matrix of asset returns.
  - `omega`: View uncertainty matrix.
  - `P`: View matrix (views × assets).
  - `Q`: Vector of view returns (views).

# Returns

  - `posterior_mu::VecNum`: Posterior mean vector of asset returns.
  - `posterior_sigma::Matrix{<:Number}`: Posterior covariance matrix of asset returns.

# Related

  - [`BlackLittermanPrior`](@ref)
  - [`calc_omega`](@ref)
"""
function vanilla_posteriors(tau::Number, rf::Number, prior_mu::VecNum, prior_sigma::MatNum,
                            omega::MatNum, P::MatNum, Q::VecNum)
    v1 = tau * prior_sigma * transpose(P)
    v2 = P * v1 + omega
    v3 = Q - P * prior_mu
    posterior_mu = (prior_mu + v1 * (v2 \ v3)) .+ rf
    posterior_sigma = prior_sigma + tau * prior_sigma - v1 * (v2 \ transpose(v1))
    return posterior_mu, posterior_sigma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Remove excluded views from `views_conf`.

Returns `views_conf` unchanged when `excl` is `nothing` or `views_conf` is scalar. Filters `views_conf` by removing indices in `excl` when both are vectors.

# Related

  - [`BlackLittermanPrior`](@ref)
"""
function remove_excl_views(views_conf::Option{<:Number}, args...)
    return views_conf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Remove excluded views from `views_conf`.

Returns `views_conf` unchanged (no exclusions).

# Related

  - [`BlackLittermanPrior`](@ref)
"""
function remove_excl_views(views_conf::VecNum, ::Nothing)
    return views_conf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Remove excluded views from `views_conf`.

Returns a view of `views_conf` with indices in `excl` removed.

# Related

  - [`BlackLittermanPrior`](@ref)
"""
function remove_excl_views(views_conf::VecNum, excl::VecInt)
    return nothing_scalar_array_view(views_conf, setdiff(1:length(views_conf), excl))
end
"""
    prior(pe::BlackLittermanPrior, X::MatNum;
          F::Option{<:MatNum} = nothing, dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute the Black-Litterman prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns using the Black-Litterman model, combining a prior estimator, matrix post-processing, user or algorithmic views, asset sets, view confidences, risk-free rate, and blending parameter `tau`. The method supports both direct and constraint-based views, flexible confidence specification, and matrix processing.

# Mathematical definition

The Black-Litterman posterior distribution combines the prior ``(\\boldsymbol{\\Pi}, \\tau \\mathbf{\\Sigma})`` with investor views ``(\\mathbf{P}, \\boldsymbol{q}, \\mathbf{\\Omega})``:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{BL} &= \\left[(\\tau\\mathbf{\\Sigma})^{-1} + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\mathbf{P}\\right]^{-1} \\left[(\\tau\\mathbf{\\Sigma})^{-1} \\boldsymbol{\\Pi} + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\boldsymbol{q}\\right]\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{BL} &= \\mathbf{\\Sigma} + \\left[(\\tau\\mathbf{\\Sigma})^{-1} + \\mathbf{P}^\\intercal \\mathbf{\\Omega}^{-1} \\mathbf{P}\\right]^{-1}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\Pi}``: `N × 1` prior (equilibrium) expected returns.
  - ``\\mathbf{\\Sigma}``: `N × N` prior covariance matrix.
  - ``\\tau``: Scaling parameter for the uncertainty in the prior.
  - ``\\mathbf{P}``: `K × N` views matrix (each row is one view).
  - ``\\boldsymbol{q}``: `K × 1` views vector.
  - ``\\mathbf{\\Omega}``: `K × K` views uncertainty matrix.

# Arguments

  - `pe`: Black-Litterman prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F{Nothing, <:MatNum}`: Optional factor matrix (default: `nothing`).
  - $(arg_dict[:dims])
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
function prior(pe::BlackLittermanPrior, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
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
    (; P, Q, tau, omega) = bl_preroll(pe.views, pe.sets, pe.views_conf, prior_sigma, pe.tau,
                                      size(X, 1), eltype(posterior_X), strict)
    posterior_mu, posterior_sigma = vanilla_posteriors(tau, pe.rf, prior_mu, prior_sigma,
                                                       omega, P, Q)
    matrix_processing!(pe.mp, posterior_sigma, posterior_X; kwargs...)
    return LowOrderPrior(; X = posterior_X, mu = posterior_mu, sigma = posterior_sigma)
end

export BlackLittermanPrior
