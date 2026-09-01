"""
$(DocStringExtensions.TYPEDEF)

Adapts any `StatsBase.CovarianceEstimator` to the library's calling convention, carrying its observation weights alongside it.

The estimator and the weights travel together in one object, so a caller passes one value where the `StatsBase` API takes a separate estimator and weight vector at every call site. `ce` accepts any subtype of `StatsBase.CovarianceEstimator`, so an estimator from a package such as [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl) reaches the library unchanged.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeneralCovariance(;
        ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
            corrected = true),
        w::Option{<:ObsWeights} = nothing
    ) -> GeneralCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `ce`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> GeneralCovariance()
GeneralCovariance
  ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w ┴ nothing

julia> GeneralCovariance(; w = StatsBase.Weights([0.1, 0.2, 0.7]))
GeneralCovariance
  ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Option`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct GeneralCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:oow])
    """
    @wprop w
    function GeneralCovariance(ce::StatsBase.CovarianceEstimator, w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(ce), typeof(w)}(ce, w)
    end
end
function GeneralCovariance(;
                           ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                          corrected = true),
                           w::Option{<:ObsWeights} = nothing)::GeneralCovariance
    return GeneralCovariance(ce, w)
end
"""
    Statistics.cov(
        ce::GeneralCovariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the covariance matrix using a [`GeneralCovariance`](@ref) estimator.

This method dispatches to the appropriate [`robust_cov`](@ref) depending on `ce.w`, which computes the covariance matrix using `ce.ce`.

# Algorithm

 1. Resolve the observation weights from `ce.w` against `X`, giving `w`.
 2. When `w` is `nothing`, call [`robust_cov`](@ref) with `ce.ce` and `X` alone.
 3. Otherwise call [`robust_cov`](@ref) with `ce.ce`, `X` and `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to [`robust_cov`](@ref).

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cov(GeneralCovariance(), X)
2×2 Matrix{Float64}:
 0.0001  0.0001
 0.0001  0.0001
```

# Related

  - [`MatNum`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`robust_cov`](@ref)
  - [`cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    w = get_observation_weights(ce.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        robust_cov(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cov(ce.ce, X, w; dims = dims, mean = mean, kwargs...)
    end
end
"""
    Statistics.cor(
        ce::GeneralCovariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the correlation matrix using a [`GeneralCovariance`](@ref) estimator.

This method dispatches to the appropriate [`robust_cor`](@ref) depending on `ce.w`, which computes the correlation matrix using `ce.ce`.

# Algorithm

 1. Resolve the observation weights from `ce.w` against `X`, giving `w`.
 2. When `w` is `nothing`, call [`robust_cor`](@ref) with `ce.ce` and `X` alone.
 3. Otherwise call [`robust_cor`](@ref) with `ce.ce`, `X` and `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to [`robust_cor`](@ref).

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cor(GeneralCovariance(), X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`MatNum`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`robust_cor`](@ref)
  - [`cov(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::GeneralCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    w = get_observation_weights(ce.w, X; dims = dims, kwargs...)
    if isnothing(w)
        robust_cor(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cor(ce.ce, X, w; dims = dims, mean = mean, kwargs...)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the covariance matrix of asset returns from a centring estimator, a covariance estimator, and a moment algorithm.

`Covariance` encapsulates all components required for estimating the covariance matrix of asset returns, including the expected returns estimator for centering the data, the covariance estimator, and the moment algorithm.

`w` weights the whole estimate, so it reaches the centre as well as the deviations. The four methods send `me` and `ce` through [`factory`](@ref), which replaces the weights of each with `w`, so `w` wins over the weights that `me` and `ce` carry. Pass `mean` for a centre that `w` does not describe. ADR 0088 records the decision.

`ce` admits any `StatsBase.CovarianceEstimator`, and no verb of this library reads the weights of one that the library does not own. A `ce` from a package such as [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl) therefore keeps its own configuration, and `w` is the field that weights a `Covariance`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Covariance(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
        alg::AbstractMomentAlgorithm = FullMoment(),
        w::Option{<:ObsWeights} = nothing
    ) -> Covariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `ce`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `me`: Recursively indexed via [`obs_weights_view`](@ref).
  - `ce`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> Covariance()
Covariance
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ GeneralCovariance
      │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │    w ┴ nothing
  alg ┼ FullMoment()
    w ┴ nothing

julia> Covariance(; w = StatsBase.AnalyticWeights([0.2, 0.3, 0.5]))
Covariance
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ GeneralCovariance
      │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │    w ┴ nothing
  alg ┼ FullMoment()
    w ┴ StatsBase.AnalyticWeights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`Option`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`covariance_centre_and_estimator`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct Covariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:malg])
    """
    alg
    """
    $(field_dict[:oow])
    """
    @wprop w
    function Covariance(me::AbstractExpectedReturnsEstimator,
                        ce::StatsBase.CovarianceEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(ce), typeof(alg), typeof(w)}(me, ce, alg, w)
    end
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
                    alg::AbstractMomentAlgorithm = FullMoment(),
                    w::Option{<:ObsWeights} = nothing)::Covariance
    return Covariance(me, ce, alg, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the centring vector and the inner covariance estimator that a [`Covariance`](@ref) method computes with.

The four methods of `Statistics.cov` and `Statistics.cor` that take a [`Covariance`](@ref) reach one centre and one inner estimator by this verb, so `ce.w` reaches the centre and the deviations by one rule. ADR 0088 records the decision.

# Algorithm

 1. Resolve the centre `mu` from `ce.me` and `ce.w` with [`weighted_centre`](@ref), which reads `mean` when the caller gave one. `ce.w` reaches `ce.me` through [`factory`](@ref), so the centre carries the weights of the deviations.
 2. `ce.w` is `nothing`: return `ce.ce` unchanged.
 3. `ce.w` is not `nothing`: send `ce.ce` through [`factory_child`](@ref) with `ce.w`. An estimator of the library takes the weights; a `StatsBase.CovarianceEstimator` that is not one of them passes through unchanged, because no verb of this library reads its weights.

Step 2 is a performance guard and not a second contract. `ce.w` is a field, so its type decides the branch, and the guard keeps a windowed loop from rebuilding the estimator tree of `ce` once per window. A `ce.ce` that holds weights of its own therefore keeps them when `ce.w` is `nothing`, and loses them to `ce.w` when it is not. That is what [`factory`](@ref) does on every other path.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - `mu::Union{<:Number, <:ArrNum}`: Centring vector.
  - `cel::StatsBase.CovarianceEstimator`: Inner covariance estimator, weighted by `ce.w` when it is not `nothing`.

# Related

  - [`Covariance`](@ref)
  - [`weighted_centre`](@ref)
  - [`factory`](@ref)
  - [`factory_child`](@ref)
"""
function covariance_centre_and_estimator(ce::Covariance, X::MatNum; dims::Int = 1,
                                         mean = nothing, kwargs...)
    mu = weighted_centre(X, ce.me, ce.w; dims = dims, mean = mean, kwargs...)
    return mu, isnothing(ce.w) ? ce.ce : factory_child(ce.ce, ce.w)
end
"""
    Statistics.cov(
        ce::Covariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the covariance matrix using a [`Covariance`](@ref) estimator.

# Mathematical definition

FullMoment covariance:

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{ij} &= \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{ti} - \\hat{\\mu}_i)(r_{tj} - \\hat{\\mu}_j)\\,.
\\end{align}
```

SemiMoment (downside) covariance, from the de-meaned returns clipped at zero:

```math
\\begin{align}
\\tilde{r}_{tj} &= \\min(r_{tj} - \\hat{\\mu}_j,\\, 0)\\,,\\\\
\\hat{\\mathbf{\\Sigma}}^{\\text{semi}}_{ij} &= \\frac{1}{T-1} \\sum_{t=1}^{T} \\tilde{r}_{ti} \\, \\tilde{r}_{tj}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat_ij])
  - ``\\hat{\\mathbf{\\Sigma}}^{\\text{semi}}_{ij}``: Estimated semi-covariance between assets ``i`` and ``j``.
  - $(math_dict[:r_tj])
  - ``r_{ti}``: Return of asset ``i`` at time ``t``.
  - $(math_dict[:mu_hat_j])
  - ``\\hat{\\mu}_i``: Estimated mean of asset ``i``.
  - ``\\tilde{r}_{ti}``, ``\\tilde{r}_{tj}``: De-meaned returns of assets ``i`` and ``j``, clipped at zero.
  - $(math_dict[:T])

The semi-covariance keeps the ``T-1`` divisor of the full moment, so it is not the covariance of the clipped returns about their own mean.

# Algorithm

 1. Resolve the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref). When `mean` is `nothing`, `mu` comes from `ce.me`; otherwise it comes from `mean`. When `ce.w` is not `nothing`, `ce.w` reaches `ce.me` and `ce.ce` through [`factory`](@ref) first.
 2. Delegate to `Statistics.cov(cel, X; dims = dims, mean = mu, kwargs...)`.

# Arguments

  - $(arg_dict[:ce])
      + `ce::Covariance{<:Any, <:Any, <:FullMoment}`: Covariance estimator with [`FullMoment`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:SemiMoment}`: Covariance estimator with [`SemiMoment`](@ref) moment algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cov(Covariance(), X)
2×2 Matrix{Float64}:
 0.0001  0.0001
 0.0001  0.0001

julia> cov(Covariance(; alg = SemiMoment()), X)
2×2 Matrix{Float64}:
 5.0e-5  5.0e-5
 5.0e-5  5.0e-5
```

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`cor(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu, cel = covariance_centre_and_estimator(ce, X; dims = dims, mean = mean, kwargs...)
    return Statistics.cov(cel, X; dims = dims, mean = mu, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SemiMoment`](@ref) variant of [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Clips de-meaned returns to zero before computing the covariance matrix, capturing only downside co-movements.

# Algorithm

 1. Resolve the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref).
 2. Replace `X` with `min.(X .- mu, 0)`, the de-meaned returns clipped at zero.
 3. Delegate to `Statistics.cov(cel, X; dims = dims, mean = 0, kwargs...)`. The zero mean is what stops the clipped returns being centred a second time.
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu, cel = covariance_centre_and_estimator(ce, X; dims = dims, mean = mean, kwargs...)
    X = min.(X .- mu, zero(eltype(X)))
    return Statistics.cov(cel, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end
"""
    Statistics.cor(
        ce::Covariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the correlation matrix using a [`Covariance`](@ref) estimator.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{P}}_{ij} &= \\frac{\\hat{\\mathbf{\\Sigma}}_{ij}}{\\hat{\\sigma}_i \\hat{\\sigma}_j}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{P}}_{ij}``: Estimated correlation between assets ``i`` and ``j``.
  - $(math_dict[:Sigma_hat_ij])
  - $(math_dict[:sigma_hat_i])
  - ``\\hat{\\sigma}_j``: Estimated standard deviation of asset ``j``.

The `alg` field of `ce` reaches ``\\hat{\\mathbf{\\Sigma}}``: [`SemiMoment`](@ref) standardises the semi-covariance, so the diagonal is one and an off-diagonal entry is a downside correlation.

# Algorithm

 1. Resolve the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref). When `mean` is `nothing`, `mu` comes from `ce.me`; otherwise it comes from `mean`. When `ce.w` is not `nothing`, `ce.w` reaches `ce.me` and `ce.ce` through [`factory`](@ref) first.
 2. Delegate to `Statistics.cor(cel, X; dims = dims, mean = mu, kwargs...)`.

# Arguments

  - $(arg_dict[:ce])

      + `ce::Covariance{<:Any, <:Any, <:FullMoment}`: Covariance estimator with [`FullMoment`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:SemiMoment}`: Covariance estimator with [`SemiMoment`](@ref) moment algorithm.

  - $(arg_dict[:X])

  - $(arg_dict[:dims])

  - $(arg_dict[:omean]) If not provided, computed using `ce.me`.

  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cor(Covariance(), X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu, cel = covariance_centre_and_estimator(ce, X; dims = dims, mean = mean, kwargs...)
    return Statistics.cor(cel, X; dims = dims, mean = mu, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SemiMoment`](@ref) variant of [`cor(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Clips de-meaned returns to zero before computing the correlation matrix, capturing only downside co-movements.

# Algorithm

 1. Resolve the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref).
 2. Replace `X` with `min.(X .- mu, 0)`, the de-meaned returns clipped at zero.
 3. Delegate to `Statistics.cor(cel, X; dims = dims, mean = 0, kwargs...)`. The zero mean is what stops the clipped returns being centred a second time.
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu, cel = covariance_centre_and_estimator(ce, X; dims = dims, mean = mean, kwargs...)
    X = min.(X .- mu, zero(eltype(X)))
    return Statistics.cor(cel, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

export GeneralCovariance, Covariance, cov, cor
