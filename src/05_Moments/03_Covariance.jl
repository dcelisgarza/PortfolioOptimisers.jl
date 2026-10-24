"""
$(DocStringExtensions.TYPEDEF)

Adapts a `StatsBase.CovarianceEstimator` to the calling convention of the library, and holds its observation weights with it.

The estimator and the weights are in one object, so a caller passes one value. The `StatsBase` API takes an estimator and a separate weight vector at each call site. `ce` accepts all subtypes of `StatsBase.CovarianceEstimator`, so an estimator from another package, such as CovarianceEstimation.jl, gets to the library without change.

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

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ce`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `ce`: Recursively indexed through [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).
  - `cache`: Removed by [`obs_weights_view`](@ref), because a state has no slice on the observation axis.

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

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:GeneralCovariance, true)` to render it.

# Arguments

  - `::GeneralCovariance`: The covariance estimator. The method reads only its type.

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

Compute the covariance matrix with a [`GeneralCovariance`](@ref) estimator.

The method calls [`robust_cov`](@ref) with or without observation weights, as `ce.w` sets. [`robust_cov`](@ref) computes the covariance matrix with `ce.ce`.

# Algorithm

 1. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 2. Get the observation weights `w` from `ce.w` and `X`.
 3. When `w` is `nothing`, call [`robust_cov`](@ref) with `ce.ce` and `X` only.
 4. Otherwise call [`robust_cov`](@ref) with `ce.ce`, `X` and `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for [`robust_cov`](@ref) and for the observation weights.

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
    assert_finite_sample(X)
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

Compute the correlation matrix with a [`GeneralCovariance`](@ref) estimator.

The method calls [`robust_cor`](@ref) with or without observation weights, as `ce.w` sets. [`robust_cor`](@ref) computes the correlation matrix with `ce.ce`.

# Algorithm

 1. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 2. Get the observation weights `w` from `ce.w` and `X`.
 3. When `w` is `nothing`, call [`robust_cor`](@ref) with `ce.ce` and `X` only.
 4. Otherwise call [`robust_cor`](@ref) with `ce.ce`, `X` and `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for [`robust_cor`](@ref) and for the observation weights.

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
    assert_finite_sample(X)
    w = get_observation_weights(ce.w, X; dims = dims, kwargs...)
    if isnothing(w)
        robust_cor(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cor(ce.ce, X, w; dims = dims, mean = mean, kwargs...)
    end
end
"""
    library_covariance_estimator(ce::AbstractCovarianceEstimator) -> AbstractCovarianceEstimator
    library_covariance_estimator(ce::StatsBase.CovarianceEstimator) -> GeneralCovariance

Returns a nested covariance estimator in a form that accepts all the calls that the library makes to a nested estimator.

A field bound to `StatsBase.CovarianceEstimator` accepts an estimator that the library does not own, such as `StatsBase.SimpleCovariance()`. The library calls a nested estimator in these three ways, and `StatsBase` has no method for any of them:

  - With its own keywords, such as `iv`, `ivpa` and `active_mask`.
  - With an Asset Panel argument.
  - Through `Statistics.var` and `Statistics.std`.

Thus a verb that passes a call to a nested estimator calls it through this function. The two methods select one branch each:

  - The function returns an [`AbstractCovarianceEstimator`](@ref) without change, so it still gets all the keywords.
  - The function wraps all other `StatsBase.CovarianceEstimator` types in a [`GeneralCovariance`](@ref) without weights. The wrapper is an [`AbstractCovarianceEstimator`](@ref), so the Asset Panel methods and the diagonal methods of `var` and `std` accept it. Its `cov` and `cor` go through [`robust_cov`](@ref) and [`robust_cor`](@ref). These call the wrapped estimator with only `dims` and `mean` when its method does not take the other keywords. The wrapper returns what the bare estimator returns for the same `dims` and `mean`.

# Arguments

  - $(arg_dict[:ce])

# Returns

  - `ce`: `ce` without change, or `GeneralCovariance(; ce = ce)`.

# Related

  - [`GeneralCovariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`robust_cov`](@ref)
  - [`robust_cor`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
  - [`std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function library_covariance_estimator(ce::AbstractCovarianceEstimator)
    return ce
end
function library_covariance_estimator(ce::StatsBase.CovarianceEstimator)
    return GeneralCovariance(; ce = ce)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the covariance matrix of asset returns from a centring estimator, a covariance estimator, and a moment algorithm.

`me` centres the data, `ce` computes the covariance matrix, and `alg` selects the full moment or the semi-moment.

`w` weights the whole estimate, so it weights the centre and the deviations. The four methods send `me` and `ce` through [`factory`](@ref), which replaces the weights that each of them holds with `w`. Pass `mean` for a centre that `w` does not describe.

`ce` accepts all `StatsBase.CovarianceEstimator` types. No verb of the library reads the weights of an estimator that the library does not own. A `ce` from another package, such as CovarianceEstimation.jl, keeps its own configuration, and `w` is the field that weights a `Covariance`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Covariance(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
        alg::AbstractMomentAlgorithm = FullMoment(),
        w::Option{<:ObsWeights} = nothing,
        cvg::Option{<:CoveragePolicy} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> Covariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated through [`factory`](@ref).
  - `ce`: Recursively updated through [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `ce`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `me`: Recursively indexed through [`obs_weights_view`](@ref).
  - `ce`: Recursively indexed through [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).
  - `cache`: Removed by [`obs_weights_view`](@ref), because a state has no slice on the observation axis.

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
    $(field_dict[:cvg])
    """
    cvg
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function Covariance(me::AbstractExpectedReturnsEstimator,
                        ce::StatsBase.CovarianceEstimator, alg::AbstractMomentAlgorithm,
                        w::Option{<:ObsWeights}, cvg::Option{<:CoveragePolicy},
                        cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(ce), typeof(alg), typeof(w), typeof(cvg),
                   typeof(cache)}(me, ce, alg, w, cvg, cache)
    end
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralCovariance(),
                    alg::AbstractMomentAlgorithm = FullMoment(),
                    w::Option{<:ObsWeights} = nothing,
                    cvg::Option{<:CoveragePolicy} = nothing,
                    cache::Option{<:AbstractPartialFitState} = nothing)::Covariance
    return Covariance(me, ce, alg, w, cvg, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`Covariance`](@ref) except `cache`, and `cvg` only where a policy is set.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:Covariance, true)` to render it. The method reads `cvg` from the instance and not from the type, as the method for [`SimpleExpectedReturns`](@ref) does. That method states the reason.

# Arguments

  - `ce`: The covariance estimator. The method reads only its `cvg` field.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:me, :ce, :alg, :w)` with no policy and `(:me, :ce, :alg, :w, :cvg)` with one.

# Related

  - [`Covariance`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(ce::Covariance)
    return isnothing(ce.cvg) ? (:me, :ce, :alg, :w) : (:me, :ce, :alg, :w, :cvg)
end
"""
    gap_fill_value(ce::Covariance{<:Any, <:Any, <:Any, <:Any, <:CoveragePolicy}) -> Float64

Return `NaN`, so that a gapped sample gets to the available-case arm of a [`Covariance`](@ref) with its gaps.

A [`Covariance`](@ref) with a [`CoveragePolicy`](@ref) fits each pair on the observations at which its two assets are finite and active. [`coverage_covariance`](@ref) states this. A finite fill value makes a gapped cell into an observation, and the pair then reads a return that the asset did not have. The consumer passes the sample without change, together with the active mask that explains each gap. A [`Covariance`](@ref) with no policy refuses a gap itself, so it keeps the fallback of zero.

A wrapper that passes the sample on, such as [`PortfolioOptimisersCovariance`](@ref), returns the same value as this method, so the wrapper reads a gapped sample when the `Covariance` in it holds a policy.

# Arguments

  - `ce`: The covariance estimator, with a [`CoveragePolicy`](@ref) in its `cvg` field.

# Returns

  - `fv::Float64`: `NaN`.

# Related

  - [`Covariance`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`gap_fill_value`](@ref)
  - [`coverage_covariance`](@ref)
"""
function gap_fill_value(::Covariance{<:Any, <:Any, <:Any, <:Any, <:CoveragePolicy})
    return NaN
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Get the centring vector and the inner covariance estimator for a [`Covariance`](@ref) method.

The four methods of `Statistics.cov` and `Statistics.cor` that take a [`Covariance`](@ref) get one centre and one inner estimator from this function, so one rule applies `ce.w` to the centre and to the deviations.

# Algorithm

 1. Get the centre `mu` from `ce.me` and `ce.w` with [`weighted_centre`](@ref). It uses `mean` if the caller gave one. `ce.w` gets to `ce.me` through [`factory`](@ref), so the centre has the weights of the deviations.
 2. If `ce.w` is `nothing`, take `ce.ce` without change.
 3. If `ce.w` is not `nothing`, send `ce.ce` through [`factory_child`](@ref) with `ce.w`. An estimator of the library takes the weights. A `StatsBase.CovarianceEstimator` that the library does not own does not change, because no verb of the library reads its weights.
 4. Send the estimator of step 2 or 3 through [`library_covariance_estimator`](@ref). It wraps an estimator that the library does not own in a [`GeneralCovariance`](@ref), so that the keywords of the caller, such as `iv` and `ivpa`, do not get to a `StatsBase` method that refuses them.

Step 2 is for performance and is not a second contract. `ce.w` is a field, so its type decides the branch. The step stops a windowed loop from a rebuild of the estimator tree of `ce` in each window. A `ce.ce` with its own weights keeps them when `ce.w` is `nothing`, and `ce.w` replaces them when it is not. [`factory`](@ref) does the same on all other paths.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for the mean estimator.

# Returns

  - `mu::Union{<:Number, <:ArrNum}`: The centring vector.
  - `cel::AbstractCovarianceEstimator`: The inner covariance estimator. It has the weights `ce.w` when `ce.w` is not `nothing`. It is in a [`GeneralCovariance`](@ref) wrapper when the library does not own it.

# Related

  - [`Covariance`](@ref)
  - [`weighted_centre`](@ref)
  - [`factory`](@ref)
  - [`factory_child`](@ref)
"""
function covariance_centre_and_estimator(ce::Covariance, X::MatNum; dims::Int = 1,
                                         mean = nothing, kwargs...)
    mu = weighted_centre(X, ce.me, ce.w; dims = dims, mean = mean, kwargs...)
    return mu, library_covariance_estimator(if isnothing(ce.w)
                                                ce.ce
                                            else
                                                factory_child(ce.ce, ce.w)
                                            end)
end
"""
    Statistics.cov(
        ce::Covariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the covariance matrix with a [`Covariance`](@ref) estimator.

# Mathematical definition

FullMoment covariance:

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{ij} &= \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{ti} - \\hat{\\mu}_i)(r_{tj} - \\hat{\\mu}_j)\\,.
\\end{align}
```

SemiMoment covariance, the downside covariance of the de-meaned returns clipped at zero:

```math
\\begin{align}
\\tilde{r}_{tj} &= \\min(r_{tj} - \\hat{\\mu}_j,\\, 0)\\,,\\\\
\\hat{\\mathbf{\\Sigma}}^{\\text{semi}}_{ij} &= \\frac{1}{T-1} \\sum_{t=1}^{T} \\tilde{r}_{ti} \\, \\tilde{r}_{tj}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat_ij])
  - ``\\hat{\\mathbf{\\Sigma}}^{\\text{semi}}_{ij}``: The estimated semi-covariance between assets ``i`` and ``j``.
  - $(math_dict[:r_tj])
  - ``r_{ti}``: The return of asset ``i`` at time ``t``.
  - $(math_dict[:mu_hat_j])
  - ``\\hat{\\mu}_i``: The estimated mean of asset ``i``.
  - ``\\tilde{r}_{ti}``, ``\\tilde{r}_{tj}``: The de-meaned returns of assets ``i`` and ``j``, clipped at zero.
  - $(math_dict[:T])

The semi-covariance keeps the ``T-1`` divisor of the full moment, so it is not the covariance of the clipped returns about their own mean.

# Algorithm

 1. Get the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref). When `mean` is `nothing`, `mu` comes from `ce.me`. Otherwise it comes from `mean`. When `ce.w` is not `nothing`, `ce.w` first gets to `ce.me` and `ce.ce` through [`factory`](@ref).
 2. Call `Statistics.cov(cel, X; dims = dims, mean = mu, kwargs...)`.

# Arguments

  - $(arg_dict[:ce])
      + `ce::Covariance{<:Any, <:Any, <:FullMoment}`: A covariance estimator with the [`FullMoment`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:SemiMoment}`: A covariance estimator with the [`SemiMoment`](@ref) moment algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) If it is `nothing`, the method computes it with `ce.me`.
  - `kwargs...`: More keyword arguments for the mean estimator and for the inner covariance estimator.

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
                        dims::Int = 1, mean = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return coverage_covariance(Statistics.cov, ce, ce.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
"""
    coverage_covariance(f, ce, cvg, X; dims::Int = 1, mean = nothing,
                        active_mask = nothing, kwargs...) -> MatNum

Routes a covariance or correlation fit to the Coverage Universe arm or to the available-case arm.

The `cvg` field of the estimator is the third argument, so dispatch on the policy selects the arm, not a branch on its value. [`coverage_mean`](@ref) does the same for a sample mean. `f` is `Statistics.cov` or `Statistics.cor`. The moment algorithm of `ce` selects the family of arms. The [`FullMoment`](@ref) arms fold. The [`SemiMoment`](@ref) arms have no recursion, and they make two passes over the block.

# Arguments

  - `f`: `Statistics.cov` or `Statistics.cor`.
  - $(arg_dict[:ce])
  - `cvg`: The policy that the estimator holds. It selects the arm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: A precomputed centre, or `nothing`.
  - `active_mask`: The active mask of the Asset Panel, `observations × assets`, or `nothing`. The Coverage Universe arms refuse an inactive cell, as they refuse a return that is not finite.
  - `kwargs...`: More keyword arguments for the centring estimator and for the inner covariance estimator.

# Returns

  - `sigma::MatNum`: The covariance or correlation matrix.

# Related

  - [`CoveragePolicy`](@ref)
  - [`Covariance`](@ref)
  - [`coverage_mean`](@ref)
  - [`coverage_valid_block`](@ref)
"""
function coverage_covariance end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The `Nothing` method of the [`FullMoment`](@ref) arm of [`coverage_covariance`](@ref) is the Coverage Universe arm. It gets the centre and the inner estimator with [`covariance_centre_and_estimator`](@ref), and then calls the inner estimator. This is the body that the verb had before the `cvg` field existed. It refuses a gap and an inactive cell of `active_mask` with [`assert_finite_sample`](@ref), because a plain estimator reads an inactive cell as a return that is not finite.

# Related

  - [`coverage_covariance`](@ref)
  - [`covariance_centre_and_estimator`](@ref)
"""
function coverage_covariance(f::F, ce::Covariance{<:Any, <:Any, <:FullMoment}, ::Nothing,
                             X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...) where {F}
    assert_finite_sample(X, active_mask)
    mu, cel = covariance_centre_and_estimator(ce, X; dims = dims, mean = mean, kwargs...)
    return f(cel, X; dims = dims, mean = mu, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`CoveragePolicy`](@ref) method of the [`FullMoment`](@ref) arm of [`coverage_covariance`](@ref) is the available-case arm. It fits each pair on the observations at which its two assets are finite and active. The batch result is the incremental result, because the arm folds the block through [`partial_fit!`](@ref) and then reads the state.

The centre of a pair is the mean of that pair, which a pairwise Welford recursion holds, so the diagonal is the ordinary available-case variance of each asset. An off-diagonal entry is centred on the observations that the pair shares. The arm refuses a `mean` from the caller. [`partial_fit_corrected`](@ref) refuses a `ce.ce`, a `ce.me` or a `ce.w` that an incremental fit cannot reproduce.

# Algorithm

 1. Refuse a `mean` from the caller.
 2. Read the Bessel correction, and refuse a configuration that no incremental fit can reproduce, with [`partial_fit_corrected`](@ref).
 3. Fold each row of `X` into a new state with [`partial_fit!`](@ref), together with its row of the active mask.
 4. Read the state with [`cov(ce::Covariance, state::CovarianceState)`](@ref). Rescale the result to a unit diagonal when the verb is `Statistics.cor`.

# Related

  - [`coverage_covariance`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_correlation`](@ref)
  - [`partial_fit!`](@ref)
"""
function coverage_covariance(f::F, ce::Covariance{<:Any, <:Any, <:FullMoment},
                             cvg::CoveragePolicy, X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             strict::Bool = false, kwargs...) where {F}
    @argcheck(isnothing(mean),
              ArgumentError("an available-case covariance centres each pair on that pair's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    partial_fit_corrected(ce)
    ce = partial_fit!(Covariance(; me = ce.me, ce = ce.ce, alg = ce.alg, w = ce.w,
                                 cvg = cvg), X; dims = dims, active_mask = active_mask)
    return coverage_correlation(f, Statistics.cov(ce; strict = strict))
end
"""
    coverage_correlation(f::typeof(Statistics.cov), sigma::MatNum) -> MatNum
    coverage_correlation(f::typeof(Statistics.cor), sigma::MatNum) -> MatNum

Changes an available-case covariance to the result that the verb of the caller asks for.

An available-case fit makes a covariance for both verbs, because the state holds only a co-moment accumulator. The correlation is that covariance, rescaled by the square roots of its diagonal. The method dispatches on the verb and does not compare it, so the compiler makes the choice.

# Arguments

  - `f`: `Statistics.cov` or `Statistics.cor`.
  - `sigma`: The available-case covariance.

# Returns

  - `val::MatNum`: `sigma` without change, or `sigma` rescaled to a unit diagonal.

# Related

  - [`coverage_covariance`](@ref)
  - [`Covariance`](@ref)
"""
function coverage_correlation(::typeof(Statistics.cov), sigma::MatNum)
    return sigma
end
function coverage_correlation(::typeof(Statistics.cor), sigma::MatNum)
    s = sqrt.(LinearAlgebra.diag(sigma))
    return sigma ./ (s * transpose(s))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`SemiMoment`](@ref) method of [`cov(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). It clips the de-meaned returns at zero before it computes the covariance matrix, so the matrix holds only the downside co-movements.

# Algorithm

 1. Get the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref).
 2. Replace `X` with `min.(X .- mu, 0)`, the de-meaned returns clipped at zero.
 3. Call `Statistics.cov(cel, X; dims = dims, mean = 0, kwargs...)`. The zero mean stops a second centring of the clipped returns.
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return coverage_covariance(Statistics.cov, ce, ce.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The `Nothing` method of the [`SemiMoment`](@ref) arm of [`coverage_covariance`](@ref) is the Coverage Universe arm. It clips the de-meaned returns at zero, and then calls the inner estimator with a zero centre. This is the body that the verb had before the `cvg` field existed. It refuses a gap and an inactive cell of `active_mask` with [`assert_finite_sample`](@ref), because a plain estimator reads an inactive cell as a return that is not finite.

# Related

  - [`coverage_covariance`](@ref)
  - [`SemiMoment`](@ref)
"""
function coverage_covariance(f::F, ce::Covariance{<:Any, <:Any, <:SemiMoment}, ::Nothing,
                             X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             kwargs...) where {F}
    assert_finite_sample(X, active_mask)
    mu, cel = covariance_centre_and_estimator(ce, X; dims = dims, mean = mean, kwargs...)
    X = min.(X .- mu, zero(eltype(X)))
    return f(cel, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`CoveragePolicy`](@ref) method of the [`SemiMoment`](@ref) arm of [`coverage_covariance`](@ref) is the available-case arm. It makes two passes over the block and does not fold, because a semi-covariance has no incremental recursion. The clip is about a centre that the whole window sets, so a new observation changes all the past terms.

The centre is the available-case mean of each asset, over only the finite and active observations of that asset. It is not the mean of the pair, which the [`FullMoment`](@ref) arm uses. The plain path has the same asymmetry. It takes the clip about a centre, and the co-moment about zero.

# Algorithm

 1. Check `dims`. Refuse a `ce.me` or a `ce.w` that holds observation weights, with [`assert_partial_fittable`](@ref). Refuse a `mean` from the caller.
 2. Read the valid entries of the block with [`coverage_valid_block`](@ref).
 3. Centre each asset on its available-case mean. Clip the deviations at zero, and set the invalid entries to zero.
 4. Take the numerator as the cross product of the clipped block, and the denominator as the cross product of the valid mask, which counts the observations of each pair.
 5. Admit the assets with [`coverage_admission`](@ref). Then peel the assets of the undetermined pairs with [`coverage_peel`](@ref), which gives a warning, or refuses them under `strict`.
 6. Divide and put the result in a frame with [`coverage_divide`](@ref). Rescale the result to a unit diagonal when the verb is `Statistics.cor`.

# Related

  - [`coverage_covariance`](@ref)
  - [`coverage_valid_block`](@ref)
  - [`coverage_divide`](@ref)
"""
function coverage_covariance(f::F, ce::Covariance{<:Any, <:Any, <:SemiMoment},
                             cvg::CoveragePolicy, X::MatNum; dims::Int = 1, mean = nothing,
                             active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                             strict::Bool = false, kwargs...) where {F}
    assert_dims(dims)
    assert_partial_fittable(ce.me, ce.w, "Covariance")
    @argcheck(isnothing(mean),
              ArgumentError("an available-case semi-covariance centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    Xo, msk, mu, active, stale = coverage_valid_block(X, active_mask, cvg.alg; dims = dims)
    Y = min.(Xo .- transpose(mu), zero(eltype(mu)))
    Y[.!msk] .= zero(eltype(Y))
    mski = Int.(msk)
    nu = transpose(mski) * mski
    corrected = partial_fit_corrected(ce.ce)
    cmsk = coverage_peel(cvg,
                         coverage_admission(cvg, CoverageCounts(nu, nothing, active, stale),
                                            size(Xo, 1)), nu, corrected, strict)
    sigma = coverage_divide(transpose(Y) * Y, nu, corrected, cmsk)
    return coverage_correlation(f, sigma)
end
"""
    Statistics.cor(
        ce::Covariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the correlation matrix with a [`Covariance`](@ref) estimator.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{P}}_{ij} &= \\frac{\\hat{\\mathbf{\\Sigma}}_{ij}}{\\hat{\\sigma}_i \\hat{\\sigma}_j}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{P}}_{ij}``: The estimated correlation between assets ``i`` and ``j``.
  - $(math_dict[:Sigma_hat_ij])
  - $(math_dict[:sigma_hat_i])
  - ``\\hat{\\sigma}_j``: The estimated standard deviation of asset ``j``.

The `alg` field of `ce` changes ``\\hat{\\mathbf{\\Sigma}}``. [`SemiMoment`](@ref) standardises the semi-covariance, so the diagonal is one and an off-diagonal entry is a downside correlation.

# Algorithm

 1. Get the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref). When `mean` is `nothing`, `mu` comes from `ce.me`. Otherwise it comes from `mean`. When `ce.w` is not `nothing`, `ce.w` first gets to `ce.me` and `ce.ce` through [`factory`](@ref).
 2. Call `Statistics.cor(cel, X; dims = dims, mean = mu, kwargs...)`.

# Arguments

  - $(arg_dict[:ce])

      + `ce::Covariance{<:Any, <:Any, <:FullMoment}`: A covariance estimator with the [`FullMoment`](@ref) moment algorithm.
      + `ce::Covariance{<:Any, <:Any, <:SemiMoment}`: A covariance estimator with the [`SemiMoment`](@ref) moment algorithm.

  - $(arg_dict[:X])

  - $(arg_dict[:dims])

  - $(arg_dict[:omean]) If it is `nothing`, the method computes it with `ce.me`.

  - `kwargs...`: More keyword arguments for the mean estimator and for the inner correlation estimator.

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
                        dims::Int = 1, mean = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return coverage_covariance(Statistics.cor, ce, ce.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`SemiMoment`](@ref) method of [`cor(ce::Covariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). It clips the de-meaned returns at zero before it computes the correlation matrix, so the matrix holds only the downside co-movements.

# Algorithm

 1. Get the centring vector `mu` and the inner estimator `cel` with [`covariance_centre_and_estimator`](@ref).
 2. Replace `X` with `min.(X .- mu, 0)`, the de-meaned returns clipped at zero.
 3. Call `Statistics.cor(cel, X; dims = dims, mean = 0, kwargs...)`. The zero mean stops a second centring of the clipped returns.
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:SemiMoment}, X::MatNum;
                        dims::Int = 1, mean = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return coverage_covariance(Statistics.cor, ce, ce.cvg, X; dims = dims, mean = mean,
                               active_mask = active_mask, kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the observation count, the running mean and the co-moment accumulator of an incremental covariance fit.

It is the state of [`GeneralCovariance`](@ref) and of `Covariance{<:Any, <:Any, <:FullMoment}` under [`partial_fit!`](@ref). One struct serves both, because the two estimators run the same recursion over the same three quantities. `M` is the accumulator ``\\sum_t (\\boldsymbol{r}_t - \\hat{\\boldsymbol{\\mu}}) (\\boldsymbol{r}_t - \\hat{\\boldsymbol{\\mu}})^{\\intercal}`` and not the covariance, so [`cov(ce::GeneralCovariance, state::CovarianceState)`](@ref) divides it by the count, or by the count less one when the inner `StatsBase.SimpleCovariance` is corrected.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CovarianceState(;
        n::Integer = 0,
        mu::VecNum,
        M::MatNum = zeros(eltype(mu), length(mu), length(mu)),
        cvg::Option{<:CoverageCounts} = nothing
    ) -> CovarianceState

Keywords correspond to the struct's fields. A state seeded for `N` assets is `CovarianceState(; mu = zeros(N))`, which [`partial_fit!`](@ref) builds when the `cache` field of the estimator holds `nothing`.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `!isempty(mu)`. An `IsEmptyError` is thrown otherwise.
  - Every entry of `mu` and of `M` is finite. An `IsNonFiniteError` is thrown otherwise.
  - `size(M) == (length(mu), length(mu))`. A `DimensionMismatch` is thrown otherwise.

## View parameters

[`port_opt_view`](@ref) takes the view of its fields at the selected assets:

  - `mu`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `M`: Sliced to the selected indices on both axes through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.CovarianceState(; mu = [0.0, 0.0])
PortfolioOptimisers.CovarianceState
    n ┼ Int64: 0
   mu ┼ Vector{Float64}: [0.0, 0.0]
    M ┼ 2×2 Matrix{Float64}
  cvg ┴ nothing
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
    """
    $(field_dict[:pf_cvg])
    """
    cvg
end
function CovarianceState(; n::Integer = 0, mu::VecNum,
                         M::MatNum = zeros(eltype(mu), length(mu), length(mu)),
                         cvg::Option{<:CoverageCounts} = nothing)::CovarianceState
    assert_partial_fit_state(n, mu, M)
    return CovarianceState(n, mu, M, cvg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`CovarianceState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref).
 2. If one of the two states holds no coverage counts, fold the counts, the means and the accumulators with [`chan_merge`](@ref). Its outer-product method reads a co-moment accumulator.
 3. Otherwise, fold the centre and the accumulator of each pair with the pairwise counts `nu` of the two states. A pair with no shared observation in the two blocks keeps a centre and an accumulator of zero. Read the mean of each asset from the diagonal of the centre, and merge the coverage counts too.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::CovarianceState`: The state of the two blocks, as if one fit read them as one block.

# Related

  - [`CovarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::CovarianceState, b::CovarianceState)
    assert_mergeable_states(a, b)
    ca, cb = a.cvg, b.cvg
    if isnothing(ca) || isnothing(cb)
        n, mu, M = chan_merge(a.n, a.mu, a.M, b.n, b.mu, b.M)
        return CovarianceState(n, mu, M, nothing)
    end
    nu = ca.nu .+ cb.nu
    centre = similar(ca.centre)
    M = similar(a.M)
    for j in axes(nu, 2), i in axes(nu, 1)
        if iszero(nu[i, j])
            centre[i, j] = zero(eltype(centre))
            M[i, j] = zero(eltype(M))
        else
            di = cb.centre[i, j] - ca.centre[i, j]
            dj = cb.centre[j, i] - ca.centre[j, i]
            centre[i, j] = ca.centre[i, j] + di * (cb.nu[i, j] / nu[i, j])
            M[i, j] = a.M[i, j] +
                      b.M[i, j] +
                      di * dj * (ca.nu[i, j] * cb.nu[i, j] / nu[i, j])
        end
    end
    mu = [centre[i, i] for i in axes(centre, 1)]
    return CovarianceState(a.n + b.n, mu, M,
                           CoverageCounts(nu, centre, copy(cb.active),
                                          coverage_merge_stale(ca, cb, b.n)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`CovarianceState`](@ref), so the copy shares no array with the original.

This method is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. The count is a scalar, and the method uses it without change. The method copies the running mean, the co-moment accumulator and the coverage counts.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::CovarianceState`: A new state, equal to `x`, whose `mu` and `M` are new arrays.

# Related

  - [`CovarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::CovarianceState)
    return CovarianceState(x.n, copy(x.mu), copy(x.M),
                           isnothing(x.cvg) ? nothing : copy(x.cvg))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`CovarianceState`](@ref) to the selected assets.

The Welford accumulator of one pair of assets reads only the observations of those two assets, and it reads no third asset. The slice of the state is the state of the sliced universe, entry for entry. All the assets share the count, so the method uses it without change. The slice copies by index and does not use a `view`. With a `view`, a later [`partial_fit!`](@ref) on the viewed estimator writes into the arrays of the original estimator.

# Arguments

  - `x`: The state to slice.
  - `i`: The index or the indices of the assets to keep.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `state::CovarianceState`: The state of the same sample over the selected assets.

# Related

  - [`CovarianceState`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::CovarianceState, i, args...)
    return CovarianceState(x.n, x.mu[i], x.M[i, i], coverage_counts_view(x.cvg, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the bias correction of a covariance estimator that an incremental fit can reproduce, and refuses all other estimators.

A [`CovarianceState`](@ref) accumulates the full-moment sum of outer products about the running mean. `StatsBase.SimpleCovariance` divides that sum by the count, or by the count less one. All other covariance estimators read the sample in a way that the accumulator cannot reproduce, so the function refuses them and does not give a wrong result. The function goes through the wrappers, so it reads the flag through a [`GeneralCovariance`](@ref) and through a `Covariance` whose algorithm is [`FullMoment`](@ref).

# Arguments

  - $(arg_dict[:ce])

# Validation

  - `ce` holds no observation weights at each level that the function goes through. An `ArgumentError` is thrown otherwise.
  - The innermost estimator is a `StatsBase.SimpleCovariance`. An `ArgumentError` is thrown otherwise.

# Returns

  - `corrected::Bool`: The bias correction of the innermost `StatsBase.SimpleCovariance`.

# Related

  - [`CovarianceState`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`Covariance`](@ref)
  - [`partial_fit!`](@ref)
"""
partial_fit_corrected(ce::StatsBase.SimpleCovariance) = ce.corrected
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`GeneralCovariance`](@ref) method of [`partial_fit_corrected`](@ref) refuses the observation weights of `ce` with [`assert_partial_fittable`](@ref). Then it reads the flag from `ce.ce`.
"""
function partial_fit_corrected(ce::GeneralCovariance)
    assert_partial_fittable(nothing, ce.w, "GeneralCovariance")
    return partial_fit_corrected(ce.ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The `Covariance{<:Any, <:Any, <:FullMoment}` method of [`partial_fit_corrected`](@ref) refuses the observation weights and the centring estimator of `ce` with [`assert_partial_fittable`](@ref). Then it reads the flag from `ce.ce`.
"""
function partial_fit_corrected(ce::Covariance{<:Any, <:Any, <:FullMoment})
    assert_partial_fittable(ce.me, ce.w, "Covariance")
    return partial_fit_corrected(ce.ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the fallback method of [`partial_fit_corrected`](@ref). It refuses each covariance estimator that an incremental fit cannot reproduce, and its error names the type of the estimator.
"""
function partial_fit_corrected(ce::StatsBase.CovarianceEstimator)
    return throw(ArgumentError("an incremental covariance fit folds one observation into a Welford accumulator, which reproduces the full-moment sample covariance of `StatsBase.SimpleCovariance` and no other estimator. `$(typeof(ce))` is not one, so use the batch method."))
end
"""
    covariance_state_seed(cache::Nothing, x::VecNum,
                          cvg::Option{<:CoveragePolicy} = nothing) -> CovarianceState
    covariance_state_seed(cache::CovarianceState, x::VecNum,
                          cvg::Option{<:CoveragePolicy} = nothing) -> CovarianceState

Returns the [`CovarianceState`](@ref) that an incremental covariance fit folds into. If the estimator holds no state, the function seeds a state of zeros.

The two covariance estimators, [`GeneralCovariance`](@ref) and [`Covariance`](@ref), seed the same state from the same observation, so the seed is in this one function and not in each [`partial_fit!`](@ref) method.

# Arguments

  - `cache`: The state that the estimator holds, or `nothing`.
  - `x`: One observation, `assets × 1`. The function reads only its length and its element type.
  - `cvg`: The policy that the estimator holds. It decides if the seed holds per-pair counts.

# Returns

  - `state::CovarianceState`: The state that `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`CovarianceState`](@ref)
  - [`partial_fit!`](@ref)
"""
function covariance_state_seed(::Nothing, x::VecNum,
                               cvg::Option{<:CoveragePolicy} = nothing)
    N = length(x)
    Tf = float_if_integer(eltype(x))
    return CovarianceState(0, zeros(Tf, N), zeros(Tf, N, N),
                           coverage_counts_seed(cvg, nothing, N, Tf, true))
end
function covariance_state_seed(cache::CovarianceState, x::VecNum,
                               cvg::Option{<:CoveragePolicy} = nothing)
    counts = coverage_counts_seed(cvg, cache.cvg, length(x), float_if_integer(eltype(x)),
                                  true)
    return if counts === cache.cvg
        cache
    else
        CovarianceState(cache.n, cache.mu, cache.M, counts)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`CovarianceState`](@ref) method of [`partial_fit!`](@ref) folds one observation into the count, the running mean and the co-moment accumulator.

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

  - ``n``: The observation count.
  - ``\\boldsymbol{x}``: The observation.
  - ``\\boldsymbol{\\mu}``: The running mean.
  - ``\\boldsymbol{d}``: The deviation of the observation from the mean before the fold.
  - ``\\boldsymbol{M}``: The running co-moment accumulator.

The last line reads ``\\boldsymbol{\\mu}`` after the third line changed it, but ``\\boldsymbol{d}`` reads it before the change. This asymmetry is the one of Welford, and it keeps the accumulator positive semi-definite.

# Algorithm

 1. Refuse an observation whose length is not the number of assets that the state describes.
 2. Add one to the count.
 3. Take the deviation `d` of the observation from the mean before the fold.
 4. Add `d`, divided by the new count, to `mu` in place.
 5. Add the outer product of `d` and the deviation from the mean after the fold to `M`, in place.
 6. Build the state with the new count through its positional constructor, and return it.
"""
function partial_fit!(state::CovarianceState, x::VecNum)
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    n = state.n + 1
    d = x .- state.mu
    state.mu .+= d ./ n
    state.M .+= d .* transpose(x .- state.mu)
    return CovarianceState(n, state.mu, state.M, state.cvg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `Nothing` method of the coverage arm of [`partial_fit!`](@ref) for a [`CovarianceState`](@ref). An estimator that holds no [`CoveragePolicy`](@ref) folds through [`partial_fit!(state::CovarianceState, x::VecNum)`](@ref). A plain state holds one count for all the assets, so it cannot skip an inactive cell. [`inactive_as_nonfinite`](@ref) writes `NaN` in that cell, and the row and the column of the asset are then `NaN`. The Asset Panel method gives the same `NaN` for an asset outside the Coverage Universe of the same rows.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`CovarianceState`](@ref)
"""
function partial_fit!(state::CovarianceState, x::VecNum, ::Nothing,
                      active_mask::Option{<:AbstractVector{<:Bool}})
    return partial_fit!(state, inactive_as_nonfinite(x, active_mask))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CoveragePolicy`](@ref) method of the coverage arm of [`partial_fit!`](@ref) for a [`CovarianceState`](@ref). It folds one observation into the count, the centre and the co-moment accumulator of each pair. Each pair reads only its own observations.

# Mathematical definition

```math
\\begin{align}
\\nu_{jk} &\\leftarrow \\nu_{jk} + 1\\\\
d_{jk} &= r_{tj} - c_{jk}\\\\
c_{jk} &\\leftarrow c_{jk} + \\frac{d_{jk}}{\\nu_{jk}}\\\\
d_{kj} &= r_{tk} - c_{kj}\\\\
c_{kj} &\\leftarrow c_{kj} + \\frac{d_{kj}}{\\nu_{jk}}\\\\
M_{jk} &\\leftarrow M_{jk} + d_{jk} (r_{tk} - c_{kj})\\, ,
\\end{align}
```

for each pair ``(j, k)`` whose two assets are finite and active at observation ``t``. The method applies no line to a pair that is not. Where:

  - ``\\nu_{jk}``: The number of observations at which the two assets of the pair are finite and active.
  - $(math_dict[:r_tj])
  - ``c_{jk}``: The running mean of asset ``j`` over the observations of the pair ``(j, k)``, and ``c_{kj}`` is the running mean of asset ``k`` over the same observations.
  - ``M_{jk}``: The running co-moment accumulator of the pair.

This is the Welford recursion for each pair. A covariance that folds one observation at a time is the covariance of the same rows in one block fit. The diagonal is the ordinary available-case variance of each asset. The pair is the unit of the centre and of the count, and this makes the recursion exact. An entry centred on the mean of the whole window needs a correction of all past terms when the window grows.

# Algorithm

 1. Refuse an observation whose length is not the number of assets that the state describes.
 2. Read the valid assets, and the assets that became inactive at this observation, with [`coverage_valid`](@ref).
 3. Apply the fold-time rule of the coverage algorithm with [`fold_inactive!`](@ref).
 4. Fold the observation into each pair of valid assets. Compute the upper triangle and copy it to the lower triangle, so that the accumulator stays exactly symmetric.
 5. Copy the diagonal of the centre to `mu`. The diagonal is the available-case mean of each asset.
 6. Update the per-asset records with [`coverage_step!`](@ref). Add one to the observation count, and return the state.

# Arguments

  - `state`: The state to fold into. The method changes it in place.
  - `x`: One observation, with one entry for each asset.
  - `cvg`: The policy that the estimator holds.
  - `active_mask`: The active mask of the Asset Panel at this observation, or `nothing`.

# Validation

  - `length(x)` is the number of assets that the state describes. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::CovarianceState`: The state after the observation.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_valid`](@ref)
  - [`fold_inactive!`](@ref)
  - [`coverage_step!`](@ref)
"""
function partial_fit!(state::CovarianceState, x::VecNum, cvg::CoveragePolicy,
                      active_mask::Option{<:AbstractVector{<:Bool}})
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    valid, ni = coverage_valid(x, active_mask, state.cvg)
    state = fold_inactive!(cvg.alg, state, ni)
    counts = state.cvg
    nu, centre, M = counts.nu, counts.centre, state.M
    idx = findall(valid)
    for a in eachindex(idx)
        i = idx[a]
        n = nu[i, i] + 1
        d = x[i] - centre[i, i]
        centre[i, i] += d / n
        M[i, i] += d * (x[i] - centre[i, i])
        nu[i, i] = n
        state.mu[i] = centre[i, i]
        for b in (a + 1):length(idx)
            j = idx[b]
            m = nu[i, j] + 1
            di = x[i] - centre[i, j]
            centre[i, j] += di / m
            dj = x[j] - centre[j, i]
            centre[j, i] += dj / m
            M[i, j] += di * (x[j] - centre[j, i])
            M[j, i] = M[i, j]
            nu[i, j] = m
            nu[j, i] = m
        end
    end
    coverage_step!(counts, valid, active_mask)
    return CovarianceState(state.n + 1, state.mu, M, counts)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CovarianceState`](@ref) method of [`fold_inactive!`](@ref) under [`ResetCoverage`](@ref). It sets to zero the count, the centre and the accumulator of each row and column of an asset that became inactive at this observation. Each pair of a relisted asset then starts cold.

# Related

  - [`fold_inactive!`](@ref)
  - [`ResetCoverage`](@ref)
  - [`CovarianceState`](@ref)
"""
function fold_inactive!(::ResetCoverage, state::CovarianceState, ni::AbstractVector{<:Bool})
    coverage_reset!(state.cvg.nu, ni)
    coverage_reset!(state.cvg.centre, ni)
    coverage_reset!(state.mu, ni)
    coverage_reset!(state.M, ni)
    return state
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into the partial-fit state of a [`GeneralCovariance`](@ref) estimator.

This method is the block arm of the [`partial_fit!`](@ref) interface. The Welford update reads one observation at a time, so the method folds the block one row at a time. The result equals the result of the same rows given one at a time.

# Algorithm

 1. Orient `X` to `observations × assets`. Transpose it when `dims == 2`.
 2. Fold each row in order with the single-observation arm of [`partial_fit!`](@ref). Rebind the estimator after each row.

# Arguments

  - `ce`: The covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])

# Returns

  - `ce::GeneralCovariance`: The estimator that holds the state after the last row.

# Related

  - [`GeneralCovariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ce::GeneralCovariance{<:Any, <:Any, <:Option{<:CovarianceState}},
                      X::MatNum; dims::Int = 1)
    X = dims_oriented(dims, X)
    for i in axes(X, 1)
        ce = partial_fit!(ce, view(X, i, :))
    end
    return ce
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`GeneralCovariance`](@ref) method of [`partial_fit!`](@ref) folds one observation into the state of the `cache` field. On the first call, it seeds that state.

# Algorithm

 1. Refuse an estimator that an incremental fit cannot reproduce, with [`partial_fit_corrected`](@ref).
 2. Seed a [`CovarianceState`](@ref) of zeros over `length(x)` assets when `ce.cache` holds `nothing`, with [`covariance_state_seed`](@ref).
 3. Fold `x` into the state.
 4. Rebuild the estimator with the new state in `cache` through [`rebuild_estimator`](@ref), and return it.
"""
function partial_fit!(ce::GeneralCovariance{<:Any, <:Any, <:Option{<:CovarianceState}},
                      x::VecNum)
    partial_fit_corrected(ce)
    return rebuild_estimator(ce,
                             (;
                              cache = partial_fit!(covariance_state_seed(ce.cache, x), x)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into the partial-fit state of a [`Covariance`](@ref) estimator under [`FullMoment`](@ref).

This method is the block arm of the [`partial_fit!`](@ref) interface. The Welford update reads one observation at a time, so the method folds the block one row at a time. The result equals the result of the same rows given one at a time.

# Algorithm

 1. Orient `X` and `active_mask` to `observations × assets`. Transpose them when `dims == 2`.
 2. Fold each row in order with the single-observation arm of [`partial_fit!`](@ref), together with its row of the active mask. Rebind the estimator after each row.

# Arguments

  - `ce`: The covariance estimator with a [`FullMoment`](@ref) moment algorithm.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, with the same size as `X`, or `nothing`.

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, its size is the size of `X` after the orientation. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `ce::Covariance`: The estimator that holds the state after the last row.

# Related

  - [`Covariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment, <:Any, <:Any,
                                     <:Option{<:CovarianceState}}, X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing)
    X = dims_oriented(dims, X)
    amsk = isnothing(active_mask) ? nothing : dims_oriented(dims, active_mask)
    if !isnothing(amsk)
        @argcheck(size(amsk) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(amsk)))"))
    end
    for i in axes(X, 1)
        ce = partial_fit!(ce, view(X, i, :);
                          active_mask = isnothing(amsk) ? nothing : view(amsk, i, :))
    end
    return ce
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The `Covariance{<:Any, <:Any, <:FullMoment}` method of [`partial_fit!`](@ref) folds one observation into the state of the `cache` field. On the first call, it seeds that state.

# Algorithm

 1. Refuse an estimator that an incremental fit cannot reproduce, with [`partial_fit_corrected`](@ref).
 2. Seed a [`CovarianceState`](@ref) of zeros over `length(x)` assets when `ce.cache` holds `nothing`, with [`covariance_state_seed`](@ref).
 3. Fold `x` into the state, with the policy `ce.cvg` and the active mask.
 4. Rebuild the estimator with the new state in `cache` through [`rebuild_estimator`](@ref), and return it.
"""
function partial_fit!(ce::Covariance{<:Any, <:Any, <:FullMoment, <:Any, <:Any,
                                     <:Option{<:CovarianceState}}, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    partial_fit_corrected(ce)
    state = covariance_state_seed(ce.cache, x, ce.cvg)
    return rebuild_estimator(ce, (; cache = partial_fit!(state, x, ce.cvg, active_mask)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the fallback [`Covariance`](@ref) method of [`partial_fit!`](@ref). It refuses all moment algorithms except [`FullMoment`](@ref), and its error states the reason.

[`SemiMoment`](@ref) clips the de-meaned returns at zero before the covariance, and its centre is a statistic of the whole sample. A change of the centre changes all past clips, and a past observation can go into or out of the downside. An incremental fit does not read a past observation again, so it cannot clip it again. Only the batch method gives the correct result.
"""
function partial_fit!(ce::Covariance{<:Any, <:Any, <:Any, <:Any, <:Any,
                                     <:Option{<:CovarianceState}}, ::VecNum_MatNum;
                      kwargs...)
    return throw(ArgumentError("an incremental covariance fit folds one observation into a Welford accumulator, which reproduces the `FullMoment` sample covariance alone. `$(typeof(ce.alg))` reads the whole sample at every observation, so use the batch method."))
end
"""
    Statistics.cov(
        ce::Union{<:GeneralCovariance, <:Covariance{<:Any, <:Any, <:FullMoment}},
        state::CovarianceState;
        strict::Bool = false
    ) -> MatNum
    Statistics.cov(ce::Union{<:GeneralCovariance, <:Covariance}; strict::Bool = false) -> MatNum

Read the covariance matrix of an incremental fit from a [`CovarianceState`](@ref).

The two-argument method reads a state that the caller holds. The one-argument method reads the state in the `cache` field of `ce`. The bias correction comes from the innermost `StatsBase.SimpleCovariance`, and [`partial_fit_corrected`](@ref) reads it.

The one-argument method accepts all moment algorithms, not only [`FullMoment`](@ref). An estimator wrapped in [`Online`](@ref) holds a [`SampleBufferState`](@ref) and not a [`CovarianceState`](@ref). To read that state, the method runs the batch verb over the rows of the buffer, and all algorithms accept that. The method dispatches on what `cache` holds, so a [`SemiMoment`](@ref) estimator returns a result when a wrapper gave it a buffer. When it holds no state, [`partial_fit_cache`](@ref) refuses it with an error that names the cause.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}} &= \\frac{M}{n - c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Sigma_hat])
  - ``M``: The running co-moment accumulator.
  - ``n``: The observation count.
  - ``c``: One when the innermost `StatsBase.SimpleCovariance` is corrected, and zero otherwise.

# Algorithm

 1. Get the bias correction with [`partial_fit_corrected`](@ref), which refuses each estimator that an incremental fit cannot reproduce.
 2. Take the divisor `n - c`. Return a matrix of `NaN` when the divisor is less than one. `min_obs` reads an asset with too few observations in the same way.
 3. Otherwise divide the accumulator by the divisor.

# Arguments

  - $(arg_dict[:ce])
  - `state`: The state to read.
  - `strict`: If `true`, a peel of the [`CoveragePolicy`](@ref) of `ce` throws an error. If `false`, the peel gives a warning. See [`coverage_peel`](@ref).

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
                        state::CovarianceState; strict::Bool = false)
    return coverage_covariance(ce, coverage_policy(ce), state; strict = strict)
end
"""
    coverage_policy(ce::GeneralCovariance) -> Nothing
    coverage_policy(ce::Covariance) -> Option{<:CoveragePolicy}

Reads the [`CoveragePolicy`](@ref) of a covariance estimator. The estimator can have no `cvg` field.

[`GeneralCovariance`](@ref) and [`Covariance`](@ref) share one state, and one method that makes the estimate from it, `cov(ce)` with no data. Only [`Covariance`](@ref) has a `cvg` field, so that method gets the policy through this function and not from the field. Dispatch still selects the arm, and the estimator with no field returns `nothing`.

# Arguments

  - `ce`: The covariance estimator.

# Returns

  - `cvg::Option{<:CoveragePolicy}`: The policy that the estimator holds, or `nothing`.

# Related

  - [`CoveragePolicy`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`Covariance`](@ref)
  - [`coverage_covariance`](@ref)
"""
function coverage_policy(::GeneralCovariance)
    return nothing
end
function coverage_policy(ce::Covariance)
    return ce.cvg
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `Nothing` method of [`coverage_covariance`](@ref) for a state. `cov(ce)` with no data calls it. All the pairs share one count, so the result is all `NaN` until the count is more than the Bessel correction.

[`GeneralCovariance`](@ref) has no `cvg` field and reads the same state, so it always gets to this arm.

# Related

  - [`coverage_covariance`](@ref)
  - [`CovarianceState`](@ref)
"""
function coverage_covariance(ce::Union{<:GeneralCovariance,
                                       <:Covariance{<:Any, <:Any, <:FullMoment}}, ::Nothing,
                             state::CovarianceState; kwargs...)
    k = state.n - partial_fit_corrected(ce)
    return k >= one(k) ? state.M ./ k : fill(convert(eltype(state.M), NaN), size(state.M))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CoveragePolicy`](@ref) method of [`coverage_covariance`](@ref) for a state. `cov(ce)` with no data calls it. The method divides the accumulator of each pair by the count of that pair less the Bessel correction. An asset that the policy refuses is `NaN` in all of its row and column.

A pair of two admitted assets that share too few observations has no covariance, because a covariance of no observations is not a number. [`coverage_peel`](@ref) removes the assets of such pairs from the admitted set with the `peel` rule of the policy. It reports the peel as `strict` sets. Under [`NoPeel`](@ref) the cell stays `NaN`, and the matrix repair refuses it.

# Related

  - [`coverage_covariance`](@ref)
  - [`coverage_admission`](@ref)
  - [`coverage_peel`](@ref)
  - [`coverage_divide`](@ref)
"""
function coverage_covariance(ce::Covariance{<:Any, <:Any, <:FullMoment},
                             cvg::CoveragePolicy, state::CovarianceState;
                             strict::Bool = false)
    counts = state.cvg
    corrected = partial_fit_corrected(ce)
    cmsk = coverage_peel(cvg, coverage_admission(cvg, counts, state.n), counts.nu,
                         corrected, strict)
    return coverage_divide(state.M, counts.nu, corrected, cmsk)
end
function Statistics.cov(ce::Union{<:GeneralCovariance, <:Covariance}; strict::Bool = false)
    return Statistics.cov(ce, partial_fit_cache(ce); strict = strict)
end
"""
    Statistics.cor(ce::Union{<:GeneralCovariance,
                             <:Covariance{<:Any, <:Any, <:FullMoment}},
                   state::CovarianceState)
    Statistics.cor(ce::Union{<:GeneralCovariance, <:Covariance})

Reads a correlation matrix from a folded covariance estimator.

This method is the correlation form of `cov(ce)` with no data. It takes the same two steps as the batch method. First it computes the folded covariance. Then it applies [`coverage_correlation`](@ref), which is the one place that holds the conversion. A folded estimator returns `cor` for the same configurations as `cov`, and the composite [`PortfolioOptimisersCovariance`](@ref) can give both.

# Arguments

  - $(arg_dict[:ce])
  - `state`: The state that the estimator holds.
  - `strict`: If `true`, a peel of the [`CoveragePolicy`](@ref) of `ce` throws an error. If `false`, the peel gives a warning. See [`coverage_peel`](@ref).

# Validation

  - `ce.cache` is not `nothing`, for the one-argument form. An `ArgumentError` is thrown otherwise.

# Returns

  - `rho::MatNum`: The correlation matrix of the observations that the state holds.

# Related

  - [`CovarianceState`](@ref)
  - [`coverage_correlation`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.cor(ce::Union{<:GeneralCovariance,
                                  <:Covariance{<:Any, <:Any, <:FullMoment}},
                        state::CovarianceState; strict::Bool = false)
    return coverage_correlation(Statistics.cor,
                                coverage_covariance(ce, coverage_policy(ce), state;
                                                    strict = strict))
end
function Statistics.cor(ce::Union{<:GeneralCovariance, <:Covariance}; strict::Bool = false)
    return Statistics.cor(ce, partial_fit_cache(ce); strict = strict)
end

# `GeneralCovariance` folds in every configuration. `Covariance` folds under `FullMoment`
# alone: the `SemiMoment` clip is taken against the sample mean, so a past observation's
# membership of the clipped set flips when the mean moves, and that arm falls back to the
# default, which buffers wherever a wrapper gave it a buffer (see
# [`supports_partial_fit`](@ref)).
function supports_partial_fit(::GeneralCovariance)
    return true
end
function supports_partial_fit(::Covariance{<:Any, <:Any, <:FullMoment})
    return true
end
export GeneralCovariance, Covariance, cov, cor
