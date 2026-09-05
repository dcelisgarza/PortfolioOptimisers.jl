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
        w::Option{<:ObsWeights} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> GeneralCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `ce`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).
  - `cache`: Dropped via [`obs_weights_view`](@ref), because no slice of a state exists on the observation axis.

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
  - [`CovarianceState`](@ref)
  - [`partial_fit!`](@ref)
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
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function GeneralCovariance(ce::StatsBase.CovarianceEstimator, w::Option{<:ObsWeights},
                               cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(ce), typeof(w), typeof(cache)}(ce, w, cache)
    end
end
function GeneralCovariance(;
                           ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                          corrected = true),
                           w::Option{<:ObsWeights} = nothing,
                           cache::Option{<:AbstractPartialFitState} = nothing)::GeneralCovariance
    return GeneralCovariance(ce, w, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`GeneralCovariance`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:GeneralCovariance, true)` to render it. ADR 0105 records the decision.

# Arguments

  - `::GeneralCovariance`: Covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ce, :w)`.

# Related

  - [`GeneralCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::GeneralCovariance) = (:ce, :w)
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
        w::Option{<:ObsWeights} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> Covariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `ce`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `me`: Recursively indexed via [`obs_weights_view`](@ref).
  - `ce`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).
  - `cache`: Dropped via [`obs_weights_view`](@ref), because no slice of a state exists on the observation axis.

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
  - [`CovarianceState`](@ref)
  - [`partial_fit!`](@ref)
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
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function Covariance(me::AbstractExpectedReturnsEstimator,
                        ce::StatsBase.CovarianceEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights}, cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(ce), typeof(alg), typeof(w), typeof(cache)}(me, ce,
                                                                                  alg, w,
                                                                                  cache)
    end
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
                    alg::AbstractMomentAlgorithm = FullMoment(),
                    w::Option{<:ObsWeights} = nothing,
                    cache::Option{<:AbstractPartialFitState} = nothing)::Covariance
    return Covariance(me, ce, alg, w, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`Covariance`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:Covariance, true)` to render it. ADR 0105 records the decision.

# Arguments

  - `::Covariance`: Covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:me, :ce, :alg, :w)`.

# Related

  - [`Covariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::Covariance) = (:me, :ce, :alg, :w)
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
"""
$(DocStringExtensions.TYPEDEF)

Carries the running observation count, mean and co-moment accumulator of an incremental covariance fit.

The state of [`GeneralCovariance`](@ref) and of `Covariance{<:Any, <:Any, <:FullMoment}` under [`partial_fit!`](@ref). One struct serves both, because the two estimators run the same recursion over the same three quantities. `M` is the accumulator ``\\sum_t (\\boldsymbol{r}_t - \\hat{\\boldsymbol{\\mu}}) (\\boldsymbol{r}_t - \\hat{\\boldsymbol{\\mu}})^{\\intercal}`` and not the covariance, so [`cov(ce::GeneralCovariance, state::CovarianceState)`](@ref) divides it by the count, or by the count less one when the inner `StatsBase.SimpleCovariance` is corrected.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceState(;
        n::Integer = 0,
        mu::VecNum,
        M::MatNum = zeros(eltype(mu), length(mu), length(mu))
    ) -> CovarianceState

Keywords correspond to the struct's fields. A state seeded for `N` assets is `CovarianceState(; mu = zeros(N))`, which [`partial_fit!`](@ref) builds when the `cache` field of the estimator holds `nothing`.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `!isempty(mu)`. An `IsEmptyError` is thrown otherwise.
  - Every entry of `mu` and of `M` is finite. An `IsNonFiniteError` is thrown otherwise.
  - `size(M) == (length(mu), length(mu))`. A `DimensionMismatch` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `mu`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `M`: Sliced to the selected indices on both axes via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.CovarianceState(; mu = [0.0, 0.0])
PortfolioOptimisers.CovarianceState
   n ┼ Int64: 0
  mu ┼ Vector{Float64}: [0.0, 0.0]
   M ┴ 2×2 Matrix{Float64}
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`Covariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
@concrete struct CovarianceState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    $(field_dict[:pf_mu])
    """
    mu
    """
    $(field_dict[:pf_M])
    """
    M
end
function CovarianceState(; n::Integer = 0, mu::VecNum,
                         M::MatNum = zeros(eltype(mu), length(mu), length(mu)))::CovarianceState
    assert_partial_fit_state(n, mu, M)
    return CovarianceState(n, mu, M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`CovarianceState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref).
 2. Fold the counts, the means and the accumulators with [`chan_merge`](@ref), whose outer-product method reads a co-moment accumulator.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::CovarianceState`: The state the two blocks give when they are fitted as one block.

# Related

  - [`CovarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::CovarianceState, b::CovarianceState)
    assert_mergeable_states(a, b)
    n, mu, M = chan_merge(a.n, a.mu, a.M, b.n, b.mu, b.M)
    return CovarianceState(n, mu, M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`CovarianceState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. The count is a scalar and passes through, and the running mean and the co-moment accumulator are copied.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::CovarianceState`: A fresh state, equal to `x`, whose `mu` and `M` are fresh arrays.

# Related

  - [`CovarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::CovarianceState)
    return CovarianceState(x.n, copy(x.mu), copy(x.M))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`CovarianceState`](@ref) to the selected assets.

The Welford accumulator of one pair of assets reads those two assets' observations alone, and reads no third asset. So the slice of the state is the state of the sliced universe, entry for entry, and the count is shared by every asset and passes through. The slice copies by index and does not `view`: a later [`partial_fit!`](@ref) on the viewed estimator would otherwise write through into the arrays of the estimator the view was taken from.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `state::CovarianceState`: The state of the same sample over the selected assets.

# Related

  - [`CovarianceState`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::CovarianceState, i, args...)
    return CovarianceState(x.n, x.mu[i], x.M[i, i])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves the bias correction of the covariance estimator an incremental fit reproduces, and refuses every other estimator.

A [`CovarianceState`](@ref) accumulates the full-moment sum of outer products about the running mean, which is what `StatsBase.SimpleCovariance` divides by the count or by the count less one. Every other covariance estimator reads the sample in a way the accumulator cannot answer, so it is refused rather than answered wrongly. The verb walks the wrappers, so it reads the flag through a [`GeneralCovariance`](@ref) and through a `Covariance` whose algorithm is [`FullMoment`](@ref).

# Arguments

  - $(arg_dict[:ce])

# Validation

  - `ce` carries no observation weights, at every level it is walked through. An `ArgumentError` is thrown otherwise.
  - The innermost estimator is a `StatsBase.SimpleCovariance`. An `ArgumentError` is thrown otherwise.

# Returns

  - `corrected::Bool`: Bias correction of the innermost `StatsBase.SimpleCovariance`.

# Related

  - [`CovarianceState`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`Covariance`](@ref)
  - [`partial_fit!`](@ref)
"""
partial_fit_corrected(ce::StatsBase.SimpleCovariance) = ce.corrected
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`GeneralCovariance`](@ref) method of [`partial_fit_corrected`](@ref). Refuses the observation weights of `ce` with [`assert_partial_fittable`](@ref), then reads the flag off `ce.ce`.
"""
function partial_fit_corrected(ce::GeneralCovariance)
    assert_partial_fittable(nothing, ce.w, "GeneralCovariance")
    return partial_fit_corrected(ce.ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Covariance{<:Any, <:Any, <:FullMoment}` method of [`partial_fit_corrected`](@ref). Refuses the observation weights and the centring estimator of `ce` with [`assert_partial_fittable`](@ref), then reads the flag off `ce.ce`.
"""
function partial_fit_corrected(ce::Covariance{<:Any, <:Any, <:FullMoment})
    assert_partial_fittable(ce.me, ce.w, "Covariance")
    return partial_fit_corrected(ce.ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fallback method of [`partial_fit_corrected`](@ref). Refuses every covariance estimator an incremental fit does not reproduce, naming the type that was handed over.
"""
function partial_fit_corrected(ce::StatsBase.CovarianceEstimator)
    return throw(ArgumentError("an incremental covariance fit folds one observation into a Welford accumulator, which reproduces the full-moment sample covariance of `StatsBase.SimpleCovariance` and no other estimator. `$(typeof(ce))` is not one, so use the batch method."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the [`CovarianceState`](@ref) an incremental covariance fit folds into, seeding one of zeros when the estimator carries none.

Both covariance estimators of the seam seed the same state from the same observation, so the seed is written once here rather than at each [`partial_fit!`](@ref) method.

# Arguments

  - `cache`: The state the estimator carries, or `nothing`.
  - `x`: One observation, `assets × 1`, read for its length and its element type.

# Returns

  - `state::CovarianceState`: The state `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`CovarianceState`](@ref)
  - [`partial_fit!`](@ref)
"""
function covariance_state_seed(cache::Option{<:CovarianceState}, x::VecNum)
    return if isnothing(cache)
        CovarianceState(0, zeros(eltype(x), length(x)),
                        zeros(eltype(x), length(x), length(x)))
    else
        cache
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`CovarianceState`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the running count, mean and co-moment accumulator.

# Mathematical definition

```math
\\begin{align}
n &\\leftarrow n + 1\\\\
\\boldsymbol{d} &= \\boldsymbol{x} - \\boldsymbol{\\mu}\\\\
\\boldsymbol{\\mu} &\\leftarrow \\boldsymbol{\\mu} + \\frac{\\boldsymbol{d}}{n}\\\\
\\boldsymbol{M} &\\leftarrow \\boldsymbol{M} + \\boldsymbol{d} (\\boldsymbol{x} - \\boldsymbol{\\mu})^{\\intercal}\\, .
\\end{align}
```

Where:

  - ``n``: observation count.
  - ``\\boldsymbol{x}``: the observation.
  - ``\\boldsymbol{\\mu}``: the running mean.
  - ``\\boldsymbol{d}``: deviation of the observation from the mean **before** the fold.
  - ``\\boldsymbol{M}``: the running co-moment accumulator.

The last line reads ``\\boldsymbol{\\mu}`` **after** the third line moved it, where ``\\boldsymbol{d}`` read it before. That asymmetry is Welford's, and it is what keeps the accumulator positive semi-definite.

# Algorithm

 1. Refuse an observation whose length is not the number of assets the state describes.
 2. Add one to the count.
 3. Take the deviation of the observation from the mean before the fold, giving `d`.
 4. Move `mu` in place along `d`, by the reciprocal of the new count.
 5. Add the outer product of `d` and the deviation from the mean **after** the fold to `M`, in place.
 6. Rebind the count with `Accessors.@reset`, and return the state.
"""
function partial_fit!(state::CovarianceState, x::VecNum)
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    n = state.n + 1
    d = x .- state.mu
    state.mu .+= d ./ n
    state.M .+= d .* transpose(x .- state.mu)
    return Accessors.@reset state.n = n
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into the partial-fit state of a [`GeneralCovariance`](@ref) estimator.

The block arm of the [`partial_fit!`](@ref) interface. Welford's update reads one observation at a time, so the block is folded row by row and the answer is the answer of the same rows handed over one at a time.

# Algorithm

 1. Orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Fold each row in turn with the single-observation arm of [`partial_fit!`](@ref), rebinding the estimator each time.

# Arguments

  - `ce`: Covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])

# Returns

  - `ce::GeneralCovariance`: The estimator carrying the state after the last row.

# Related

  - [`GeneralCovariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ce::GeneralCovariance, X::MatNum; dims::Int = 1)
    X = dims_oriented(dims, X)
    for i in axes(X, 1)
        ce = partial_fit!(ce, view(X, i, :))
    end
    return ce
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`GeneralCovariance`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the state the `cache` field carries, seeding it on the first call.

# Algorithm

 1. Refuse an estimator an incremental fit does not reproduce, with [`partial_fit_corrected`](@ref).
 2. Seed a [`CovarianceState`](@ref) of zeros over `length(x)` assets when `ce.cache` holds `nothing`, with [`covariance_state_seed`](@ref).
 3. Fold `x` into the state.
 4. Rebind `ce.cache` with `Accessors.@reset`, and return the estimator.
"""
function partial_fit!(ce::GeneralCovariance, x::VecNum)
    partial_fit_corrected(ce)
    return Accessors.@reset ce.cache = partial_fit!(covariance_state_seed(ce.cache, x), x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into the partial-fit state of a [`Covariance`](@ref) estimator under [`FullMoment`](@ref).

The block arm of the [`partial_fit!`](@ref) interface. Welford's update reads one observation at a time, so the block is folded row by row and the answer is the answer of the same rows handed over one at a time.

# Algorithm

 1. Orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Fold each row in turn with the single-observation arm of [`partial_fit!`](@ref), rebinding the estimator each time.

# Arguments

  - `ce`: Covariance estimator with a [`FullMoment`](@ref) moment algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])

# Returns

  - `ce::Covariance`: The estimator carrying the state after the last row.

# Related

  - [`Covariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1)
    X = dims_oriented(dims, X)
    for i in axes(X, 1)
        ce = partial_fit!(ce, view(X, i, :))
    end
    return ce
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Covariance{<:Any, <:Any, <:FullMoment}` method of [`partial_fit!`](@ref). Folds one observation into the state the `cache` field carries, seeding it on the first call.

# Algorithm

 1. Refuse an estimator an incremental fit does not reproduce, with [`partial_fit_corrected`](@ref).
 2. Seed a [`CovarianceState`](@ref) of zeros over `length(x)` assets when `ce.cache` holds `nothing`, with [`covariance_state_seed`](@ref).
 3. Fold `x` into the state.
 4. Rebind `ce.cache` with `Accessors.@reset`, and return the estimator.
"""
function partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment}, x::VecNum)
    partial_fit_corrected(ce)
    return Accessors.@reset ce.cache = partial_fit!(covariance_state_seed(ce.cache, x), x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fallback [`Covariance`](@ref) method of [`partial_fit!`](@ref). Refuses every moment algorithm but [`FullMoment`](@ref), naming the reason.

[`SemiMoment`](@ref) clamps the de-meaned returns at zero **before** the covariance, and the centre it de-means by is a statistic of the whole sample. So a centre that moves moves every past clamp, and a past observation's membership of the downside flips. An incremental fit never reads a past observation again, so it cannot re-clamp one, and the batch method is the one that answers.
"""
function partial_fit!(ce::Covariance, ::VecNum_MatNum; kwargs...)
    return throw(ArgumentError("an incremental covariance fit folds one observation into a Welford accumulator, which reproduces the `FullMoment` sample covariance alone. `$(typeof(ce.alg))` reads the whole sample at every observation, so use the batch method."))
end
"""
    Statistics.cov(
        ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}},
        state::CovarianceState
    ) -> MatNum
    Statistics.cov(
        ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}}
    ) -> MatNum

Read the covariance matrix of an incremental fit out of a [`CovarianceState`](@ref).

The two-argument method reads a state the caller holds, and the one-argument method reads the state the `cache` field of `ce` carries. The bias correction comes from the innermost `StatsBase.SimpleCovariance`, which [`partial_fit_corrected`](@ref) resolves.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\frac{M}{n - c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``M``: Running co-moment accumulator.
  - ``n``: Observation count.
  - ``c``: One when the innermost `StatsBase.SimpleCovariance` is corrected, and zero otherwise.

# Algorithm

 1. Resolve the bias correction with [`partial_fit_corrected`](@ref), which refuses every estimator an incremental fit does not reproduce.
 2. Take the divisor `n - c`, and return a matrix of `NaN` when it is below one, in the way `min_obs` reads an asset with too few observations.
 3. Otherwise divide the accumulator by the divisor.

# Arguments

  - $(arg_dict[:ce])
  - `state`: The state to read.

# Validation

  - `ce` passes [`partial_fit_corrected`](@ref). An `ArgumentError` is thrown otherwise.
  - `ce.cache` is not `nothing`, for the one-argument method. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:sigma]) `NaN` where the state holds too few observations.

# Examples

```jldoctest
julia> ce = foldl(partial_fit!, eachrow([0.01 0.02; 0.03 0.04; 0.02 0.03]); init = Covariance());

julia> cov(ce)
2×2 Matrix{Float64}:
 0.0001  0.0001
 0.0001  0.0001
```

# Related

  - [`GeneralCovariance`](@ref)
  - [`Covariance`](@ref)
  - [`CovarianceState`](@ref)
  - [`partial_fit!`](@ref)
  - [`partial_fit_corrected`](@ref)
  - [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::Union{<:GeneralCovariance,
                                  <:Covariance{<:Any, <:Any, <:FullMoment}},
                        state::CovarianceState)
    k = state.n - partial_fit_corrected(ce)
    return k >= one(k) ? state.M ./ k : fill(convert(eltype(state.M), NaN), size(state.M))
end
function Statistics.cov(ce::Union{<:GeneralCovariance,
                                  <:Covariance{<:Any, <:Any, <:FullMoment}})
    return Statistics.cov(ce, partial_fit_cache(ce))
end

export GeneralCovariance, Covariance, cov, cor
