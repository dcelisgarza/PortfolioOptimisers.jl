"""
$(DocStringExtensions.TYPEDEF)

Computes the marginal variance and standard deviation, with optional observation weights and an optional bias correction.

`me` centres the data when the caller gives no `mean`. `w` weights the observations, and `corrected` selects the bias correction. Only the matrix methods use `me`. The vector methods let `Statistics` centre the data.

`w` weights the whole estimate, so it weights the centre and the deviations. The matrix methods send `me` through [`factory`](@ref), which replaces the weights of `me` with `w`. `Statistics` centres a weighted vector on its weighted mean. The two paths give the same number for the same data, and `w` replaces the weights that `me` holds. Pass `mean` for a centre that `w` does not describe.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleVariance(;
        me::Option{<:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
        w::Option{<:ObsWeights} = nothing,
        corrected::Bool = true,
        cvg::Option{<:CoveragePolicy} = nothing,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> SimpleVariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - `corrected = true` needs a weight type that has a bias correction. The constructor does not check it, and `StatsBase` throws an `ArgumentError` at the fit. See the bias-correction bullet of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref).

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated through [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `me`: Recursively indexed through [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).
  - `cache`: Removed by [`obs_weights_view`](@ref), because a state has no slice on the observation axis.

# Examples

```jldoctest
julia> SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`SimpleVarianceState`](@ref)
  - [`partial_fit!`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct SimpleVariance <: AbstractVarianceEstimator
    """
    $(field_dict[:ome])
    """
    @fprop @vprop me
    """
    $(field_dict[:ow])
    """
    @wprop w
    """
    $(field_dict[:corrected])
    """
    corrected
    """
    $(field_dict[:cvg])
    """
    cvg
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function SimpleVariance(me::Option{<:AbstractExpectedReturnsEstimator},
                            w::Option{<:ObsWeights}, corrected::Bool,
                            cvg::Option{<:CoveragePolicy},
                            cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(w), typeof(corrected), typeof(cvg), typeof(cache)}(me,
                                                                                         w,
                                                                                         corrected,
                                                                                         cvg,
                                                                                         cache)
    end
end
function SimpleVariance(;
                        me::Option{<:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                        w::Option{<:ObsWeights} = nothing, corrected::Bool = true,
                        cvg::Option{<:CoveragePolicy} = nothing,
                        cache::Option{<:AbstractPartialFitState} = nothing)::SimpleVariance
    return SimpleVariance(me, w, corrected, cvg, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`SimpleVariance`](@ref) except `cache`, and `cvg` only where a policy is set.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:SimpleVariance, true)` to render it. The method reads `cvg` from the instance and not from the type, as the method for [`SimpleExpectedReturns`](@ref) does. That method states the reason.

# Arguments

  - `ve`: The variance estimator. The method reads only its `cvg` field.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:me, :w, :corrected)` with no policy and `(:me, :w, :corrected, :cvg)` with one.

# Related

  - [`SimpleVariance`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(ve::SimpleVariance)
    return isnothing(ve.cvg) ? (:me, :w, :corrected) : (:me, :w, :corrected, :cvg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the dispersion for the [`SimpleVariance`](@ref) methods of `Statistics.std` and `Statistics.var`.

# Algorithm

The matrix method:

 1. Check that `dims` is `1` or `2`.
 2. Get the centring vector `mu` from `me` and `ve.w` with [`weighted_centre`](@ref). It uses `mean` if the caller gave one.
 3. Get the observation weights `w` from `ve.w` and `X` with [`get_observation_weights`](@ref).
 4. When `w` is `nothing`, call `f(X; dims = dims, corrected = ve.corrected, mean = mu)`.
 5. Otherwise call `f(X, w, dims; corrected = ve.corrected, mean = mu)`.

The vector method:

 1. Get the observation weights `w` from `ve.w` and `X`.
 2. When `w` is `nothing`, call `f(X; corrected = ve.corrected, mean = mean)`.
 3. Otherwise call `f(X, w; corrected = ve.corrected, mean = mean)`.

The two methods get one centre by two routes. The matrix method gets a centre before it calls `f`. [`weighted_centre`](@ref) takes that centre from `me` after [`factory`](@ref) writes `ve.w` into `me`. The vector method passes `mean` to `f` without change, so a `mean` of `nothing` lets `f` centre on the weighted mean of `X`.

One `SimpleVariance` gives the same number for a one-column matrix and for the matching vector. `ve.w` replaces the weights that `me` holds, as [`factory`](@ref) does on all other paths. Pass `mean` for any other centre.

[`weighted_centre`](@ref) calls [`factory`](@ref) only when `ve.w` is not `nothing`. That test is for performance and is not a second contract. `ve.w` is a field, so its type decides the branch. The test stops a windowed loop from a rebuild of the estimator tree of `me` in each window.

# Arguments

  - `f`: The dispersion function, `Statistics.std` or `Statistics.var`.
  - `ve::SimpleVariance`: The variance estimator. It gives the observation weights and the `corrected` flag.
  - `me::AbstractExpectedReturnsEstimator`: The expected returns estimator for the centre when the caller gives no `mean`. Only the matrix method takes it.
  - `X::VecNum_MatNum`: The data matrix or vector.
  - `dims::Int = 1`: The dimension along which the computation runs. Only the matrix method takes it.
  - `mean = nothing`: The precomputed mean.
  - `kwargs...`: More keyword arguments for the centre and for the observation weights. Only the matrix method takes them.

# Validation

  - $(val_dict[:dims]) Only the matrix method checks it.

# Returns

  - `sigma::Union{<:Number, <:ArrNum}`: The dispersion of `X` that `f` computes.

# Related

  - [`SimpleVariance`](@ref)
  - [`weighted_centre`](@ref)
  - [`get_observation_weights`](@ref)
"""
function simple_variance_kernel(f::F, ve::SimpleVariance,
                                me::AbstractExpectedReturnsEstimator, X::MatNum;
                                dims::Int = 1, mean = nothing,
                                active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                                kwargs...) where {F}
    return coverage_variance(f, ve, ve.cvg, me, X; dims = dims, mean = mean,
                             active_mask = active_mask, kwargs...)
end
"""
    coverage_variance(f, ve, cvg, me, X; dims::Int = 1, mean = nothing,
                      active_mask = nothing, kwargs...) -> ArrNum
    coverage_variance(f, ve, cvg, X::VecNum; mean = nothing) -> Number

Routes a dispersion fit to the Coverage Universe arm or to the available-case arm.

The `cvg` field of the estimator is the third argument, so dispatch on the policy selects the arm, not a branch on its value. [`coverage_mean`](@ref) does the same for a sample mean. `f` is `Statistics.var` or `Statistics.std`. The available-case arm computes a variance, and [`coverage_moment_map`](@ref) changes it to the result that `f` asks for.

# Arguments

  - `f`: `Statistics.var` or `Statistics.std`.
  - `ve`: The variance estimator.
  - `cvg`: The policy that the estimator holds. It selects the arm.
  - `me`: The estimator that centres the fit.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: A precomputed centre, or `nothing`.
  - `active_mask`: The active mask of the Asset Panel, `observations × assets`, or `nothing`. The Coverage Universe arm refuses an inactive cell, as it refuses a return that is not finite.
  - `kwargs...`: More keyword arguments for the centring estimator and for the observation weights.

# Returns

  - `val`: The dispersion, with the shape that `f` gives.

# Related

  - [`CoveragePolicy`](@ref)
  - [`SimpleVariance`](@ref)
  - [`coverage_mean`](@ref)
  - [`coverage_moment_map`](@ref)
"""
function coverage_variance end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The `Nothing` method of [`coverage_variance`](@ref) is the Coverage Universe arm. It is the body that [`simple_variance_kernel`](@ref) had before the `cvg` field existed. It refuses a gapped sample with [`assert_finite_sample`](@ref). A cell is a gap when its return is not finite or when `active_mask` is `false` at that cell, so the arm refuses an inactive cell too.

# Related

  - [`coverage_variance`](@ref)
  - [`simple_variance_kernel`](@ref)
"""
function coverage_variance(f::F, ve::SimpleVariance, ::Nothing,
                           me::AbstractExpectedReturnsEstimator, X::MatNum; dims::Int = 1,
                           mean = nothing,
                           active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           kwargs...) where {F}
    assert_dims(dims)
    assert_finite_sample(X, active_mask)
    mu = weighted_centre(X, me, ve.w; dims = dims, mean = mean, kwargs...)
    w = get_observation_weights(ve.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        f(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        f(X, w, dims; corrected = ve.corrected, mean = mu)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`CoveragePolicy`](@ref) method of [`coverage_variance`](@ref) is the available-case arm. It fits the dispersion of each asset on the finite and active observations of that asset. The batch result is the incremental result, because the arm folds the block through [`partial_fit!`](@ref) and then reads the state.

The centre is the running mean of the fold, so the arm refuses a `mean` from the caller, and it does not ignore that `mean` without a message. An available-case fit centres each asset on the observations of that asset, and a centre fitted over the whole window is different.

# Algorithm

 1. Fold each row of `X` into a new state with [`partial_fit!`](@ref), together with its row of the active mask.
 2. Read the state with [`var(ve::SimpleVariance, state::SimpleVarianceState)`](@ref).
 3. Change the variance to the result that `f` asks for with [`coverage_moment_map`](@ref). Orient the result as the `dims` of the caller sets.

# Related

  - [`coverage_variance`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_moment_map`](@ref)
  - [`partial_fit!`](@ref)
"""
function coverage_variance(f::F, ve::SimpleVariance, cvg::CoveragePolicy,
                           ::AbstractExpectedReturnsEstimator, X::MatNum; dims::Int = 1,
                           mean = nothing,
                           active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           kwargs...) where {F}
    assert_dims(dims)
    @argcheck(isnothing(mean),
              ArgumentError("an available-case variance centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    ve = partial_fit!(SimpleVariance(; w = ve.w, corrected = ve.corrected, cvg = cvg), X;
                      dims = dims, active_mask = active_mask)
    val = coverage_moment_map(f, Statistics.var(ve))
    return isone(dims) ? permutedims(val) : reshape(val, :, 1)
end
"""
    coverage_moment_map(f::typeof(Statistics.var), v::VecNum) -> VecNum
    coverage_moment_map(f::typeof(Statistics.std), v::VecNum) -> VecNum

Changes an available-case variance to the result that the verb of the caller asks for.

[`coverage_variance`](@ref) fits a variance for both verbs, because the state holds only a second-moment accumulator. The standard deviation is the square root of that variance. The method dispatches on the verb and does not compare it, so the compiler makes the choice.

# Arguments

  - `f`: `Statistics.var` or `Statistics.std`.
  - `v`: The available-case variance.

# Returns

  - `val::VecNum`: `v` without change, or its element-wise square root.

# Related

  - [`coverage_variance`](@ref)
  - [`SimpleVariance`](@ref)
"""
function coverage_moment_map(::typeof(Statistics.var), v::VecNum)
    return v
end
function coverage_moment_map(::typeof(Statistics.std), v::VecNum)
    return sqrt.(v)
end
function simple_variance_kernel(f::F, ve::SimpleVariance, X::VecNum;
                                mean = nothing) where {F}
    return coverage_variance(f, ve, ve.cvg, X; mean = mean)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The `Nothing` method of the vector arm of [`coverage_variance`](@ref) is the Coverage Universe arm over the series of one asset. It refuses a gapped series with [`assert_finite_sample`](@ref).

# Related

  - [`coverage_variance`](@ref)
  - [`simple_variance_kernel`](@ref)
"""
function coverage_variance(f::F, ve::SimpleVariance, ::Nothing, X::VecNum;
                           mean = nothing) where {F}
    assert_finite_sample(X)
    w = get_observation_weights(ve.w, X)
    return if isnothing(w)
        f(X; corrected = ve.corrected, mean = mean)
    else
        f(X, w; corrected = ve.corrected, mean = mean)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CoveragePolicy`](@ref) method of the vector arm of [`coverage_variance`](@ref). One asset has no pair, so the available-case fit over its series is the ordinary fit over its finite entries. The method compares the coverage floor with the fraction of finite entries in the series. It returns `NaN` if too few finite entries remain, or if the policy does not admit the asset. It refuses a `mean` from the caller, as the matrix method does.

# Related

  - [`coverage_variance`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`admits`](@ref)
"""
function coverage_variance(f::F, ve::SimpleVariance, cvg::CoveragePolicy, X::VecNum;
                           mean = nothing) where {F}
    @argcheck(isnothing(mean),
              ArgumentError("an available-case variance centres each asset on that asset's own observations, so it cannot take a centre fitted over the whole window. Pass `mean = nothing`, or clear `cvg`."))
    Xf = filter(isfinite, X)
    Tf = float_if_integer(eltype(X))
    return if length(Xf) - ve.corrected < 1 ||
              !admits(cvg.alg, length(Xf) / max(length(X), 1), true, 0, cvg.min_coverage)
        Tf(NaN)
    else
        f(Xf; corrected = ve.corrected, mean = nothing)
    end
end
"""
    Statistics.std(
        ve::SimpleVariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...,
    ) -> ArrNum

Compute the standard deviation of a matrix with a [`SimpleVariance`](@ref) estimator.

The method computes the standard deviation of each asset in `X`, as `ve` configures it.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_j &= \\sqrt{\\hat{\\sigma}^2_j}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_j``: The estimated standard deviation of asset ``j``.
  - $(math_dict[:sigma2_hat_j])

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines ``\\hat{\\sigma}^2_j`` in each of the four cases that `ve.w` and `ve.corrected` select.

# Algorithm

 1. Check that `dims` is `1` or `2`.
 2. When `mean` is `nothing`, compute the centring vector `mu` with `ve.me`, after [`factory`](@ref) writes `ve.w` into it. Otherwise take `mu` from `mean`.
 3. Get the observation weights `w` from `ve.w` and `X`.
 4. When `w` is `nothing`, take the unweighted standard deviation of `X` along `dims`, centred on `mu`.
 5. Otherwise take the standard deviation of `X` weighted by `w` along `dims`, centred on `mu`.

``\\hat{\\mu}_j`` comes from `ve.me`, and `ve.w` gets to `ve.me` through [`factory`](@ref). A `SimpleVariance` with a `w` applies the same weights to the centre and to the squared deviations. A vector and its one-column matrix give the same number. Pass `mean` for any other centre.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for the mean estimator `ve.me` and for [`get_observation_weights`](@ref).

# Validation

  - $(val_dict[:dims])
  - `corrected = true` needs a weight type that has a bias correction. A plain `StatsBase.Weights` has none, and `StatsBase` throws an `ArgumentError`.

# Returns

  - $(ret_dict[:stdarr])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> std(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 1.41421  1.41421
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    return simple_variance_kernel(Statistics.std, ve, ve.me, X; dims = dims, mean = mean,
                                  kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `SimpleVariance{Nothing}` method of [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). When the caller gives no `mean`, it computes the mean with a [`SimpleExpectedReturns`](@ref), because the `me` field is `nothing`.
"""
function Statistics.std(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return simple_variance_kernel(Statistics.std, ve, SimpleExpectedReturns(), X;
                                  dims = dims, mean = mean, kwargs...)
end
"""
    Statistics.std(
        ve::SimpleVariance,
        X::VecNum;
        mean = nothing
    ) -> Number

Compute the standard deviation of a vector with a [`SimpleVariance`](@ref) estimator.

The method computes the standard deviation of the series `X`, as `ve` configures it.

# Mathematical definition

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines the variance in each of the four cases that `ve.w` and `ve.corrected` select. This method returns its square root.

# Algorithm

 1. Get the observation weights `w` from `ve.w` and `X`.
 2. When `w` is `nothing`, take the unweighted standard deviation of `X`, centred on `mean`.
 3. Otherwise take the standard deviation of `X` weighted by `w`, centred on `mean`.

The vector methods ignore `ve.me`. A `mean` of `nothing` goes to `Statistics.std`, which centres on the mean of `X`. That mean is the weighted mean when `w` is not `nothing`. The matrix methods get the centre from `ve.me` with the same `ve.w`, so the two paths give the same number for the same data.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Validation

  - `corrected = true` needs a weight type that has a bias correction. A plain `StatsBase.Weights` has none, and `StatsBase` throws an `ArgumentError`.

# Returns

  - $(ret_dict[:stdnum])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> std(sv, X)
1.0

julia> svw = SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false

julia> std(svw, X)
0.7810249675906654
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::VecNum; mean = nothing)
    return simple_variance_kernel(Statistics.std, ve, X; mean = mean)
end
"""
    Statistics.var(
        ve::SimpleVariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> ArrNum

Compute the variance of a matrix with a [`SimpleVariance`](@ref) estimator.

The method computes the variance of each asset in `X`, as `ve` configures it.

# Mathematical definition

Unweighted, `corrected = true`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

Unweighted, `corrected = false`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

Weighted:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{\\sum_{t=1}^{T} w_t (r_{tj} - \\hat{\\mu}_j)^2}{\\sum_{t=1}^{T} w_t - c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma2_hat_j])
  - $(math_dict[:r_tj])
  - $(math_dict[:mu_hat_j])
  - $(math_dict[:T])
  - $(math_dict[:w_t_obs])
  - $(math_dict[:c_weight_bias])

# Algorithm

 1. Check that `dims` is `1` or `2`.
 2. When `mean` is `nothing`, compute the centring vector `mu` with `ve.me`, after [`factory`](@ref) writes `ve.w` into it. Otherwise take `mu` from `mean`.
 3. Get the observation weights `w` from `ve.w` and `X`.
 4. When `w` is `nothing`, take the unweighted variance of `X` along `dims`, centred on `mu`.
 5. Otherwise take the variance of `X` weighted by `w` along `dims`, centred on `mu`.

``\\hat{\\mu}_j`` comes from `ve.me`, and `ve.w` gets to `ve.me` through [`factory`](@ref). A `SimpleVariance` with a `w` applies the same weights to the centre and to the squared deviations. A vector and its one-column matrix give the same number. Pass `mean` for any other centre.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: More keyword arguments for the mean estimator `ve.me` and for [`get_observation_weights`](@ref).

# Validation

  - $(val_dict[:dims])
  - `corrected = true` needs a weight type that has a bias correction. A plain `StatsBase.Weights` has none, and `StatsBase` throws an `ArgumentError`.

# Returns

  - $(ret_dict[:vararr])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> var(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 2.0  2.0
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    return simple_variance_kernel(Statistics.var, ve, ve.me, X; dims = dims, mean = mean,
                                  kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `SimpleVariance{Nothing}` method of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). When the caller gives no `mean`, it computes the mean with a [`SimpleExpectedReturns`](@ref), because the `me` field is `nothing`.
"""
function Statistics.var(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return simple_variance_kernel(Statistics.var, ve, SimpleExpectedReturns(), X;
                                  dims = dims, mean = mean, kwargs...)
end
"""
    Statistics.var(
        ve::SimpleVariance,
        X::VecNum;
        mean = nothing
    ) -> Number

Compute the variance of a vector with a [`SimpleVariance`](@ref) estimator.

The method computes the variance of the series `X`, as `ve` configures it.

# Mathematical definition

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines the variance in each of the four cases that `ve.w` and `ve.corrected` select. This method returns it for one series.

# Algorithm

 1. Get the observation weights `w` from `ve.w` and `X`.
 2. When `w` is `nothing`, take the unweighted variance of `X`, centred on `mean`.
 3. Otherwise take the variance of `X` weighted by `w`, centred on `mean`.

The vector methods ignore `ve.me`. A `mean` of `nothing` goes to `Statistics.var`, which centres on the mean of `X`. That mean is the weighted mean when `w` is not `nothing`. The matrix methods get the centre from `ve.me` with the same `ve.w`, so the two paths give the same number for the same data.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Validation

  - `corrected = true` needs a weight type that has a bias correction. A plain `StatsBase.Weights` has none, and `StatsBase` throws an `ArgumentError`.

# Returns

  - $(ret_dict[:varnum])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> var(sv, X)
1.0

julia> svw = SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false

julia> var(svw, X)
0.61
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::VecNum; mean = nothing)
    return simple_variance_kernel(Statistics.var, ve, X; mean = mean)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the observation count, the running mean and the per-asset second-moment accumulator of an incremental variance fit.

It is the state of [`SimpleVariance`](@ref) under [`partial_fit!`](@ref). `M` is the accumulator ``\\sum_t (r_{tj} - \\hat{\\mu}_j)^2`` and not the variance, so [`var(ve::SimpleVariance, state::SimpleVarianceState)`](@ref) divides it by the count, or by the count less one when `ve.corrected` is `true`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleVarianceState(;
        n::Integer = 0,
        mu::VecNum,
        M::VecNum = zeros(eltype(mu), length(mu)),
        cvg::Option{<:CoverageCounts} = nothing
    ) -> SimpleVarianceState

Keywords correspond to the struct's fields. A state seeded for `N` assets is `SimpleVarianceState(; mu = zeros(N))`, which [`partial_fit!`](@ref) builds when the `cache` field of the estimator holds `nothing`.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `!isempty(mu)`. An `IsEmptyError` is thrown otherwise.
  - Every entry of `mu` and of `M` is finite. An `IsNonFiniteError` is thrown otherwise.
  - `length(M) == length(mu)`. A `DimensionMismatch` is thrown otherwise.

## View parameters

[`port_opt_view`](@ref) takes the view of its fields at the selected assets:

  - `mu`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `M`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.SimpleVarianceState(; mu = [0.0, 0.0])
PortfolioOptimisers.SimpleVarianceState
    n ┼ Int64: 0
   mu ┼ Vector{Float64}: [0.0, 0.0]
    M ┼ Vector{Float64}: [0.0, 0.0]
  cvg ┴ nothing
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`SimpleVariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
@concrete struct SimpleVarianceState <: AbstractPartialFitState
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
function SimpleVarianceState(; n::Integer = 0, mu::VecNum,
                             M::VecNum = zeros(eltype(mu), length(mu)),
                             cvg::Option{<:CoverageCounts} = nothing)::SimpleVarianceState
    assert_partial_fit_state(n, mu, M)
    return SimpleVarianceState(n, mu, M, cvg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`SimpleVarianceState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref).
 2. If one of the two states holds no coverage counts, fold the counts, the means and the accumulators with [`chan_merge`](@ref). Its element-wise method reads a per-asset accumulator.
 3. Otherwise, fold the mean and the accumulator of each asset with the per-asset counts `nu` of the two states. An asset with no observation in the two blocks keeps a mean and an accumulator of zero. Merge the coverage counts too.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::SimpleVarianceState`: The state of the two blocks, as if one fit read them as one block.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::SimpleVarianceState, b::SimpleVarianceState)
    assert_mergeable_states(a, b)
    ca, cb = a.cvg, b.cvg
    if isnothing(ca) || isnothing(cb)
        n, mu, M = chan_merge(a.n, a.mu, a.M, b.n, b.mu, b.M)
        return SimpleVarianceState(n, mu, M, nothing)
    end
    nu = ca.nu .+ cb.nu
    mu = similar(a.mu)
    M = similar(a.M)
    for i in eachindex(mu, nu)
        if iszero(nu[i])
            mu[i] = zero(eltype(mu))
            M[i] = zero(eltype(M))
        else
            d = b.mu[i] - a.mu[i]
            mu[i] = a.mu[i] + d * (cb.nu[i] / nu[i])
            M[i] = a.M[i] + b.M[i] + d^2 * (ca.nu[i] * cb.nu[i] / nu[i])
        end
    end
    return SimpleVarianceState(a.n + b.n, mu, M,
                               CoverageCounts(nu, nothing, copy(cb.active),
                                              coverage_merge_stale(ca, cb, b.n)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`SimpleVarianceState`](@ref), so the copy shares no array with the original.

This method is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. The count is a scalar, and the method uses it without change. The method copies the running mean, the per-asset accumulator and the coverage counts.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::SimpleVarianceState`: A new state, equal to `x`, whose `mu` and `M` are new vectors.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::SimpleVarianceState)
    return SimpleVarianceState(x.n, copy(x.mu), copy(x.M),
                               isnothing(x.cvg) ? nothing : copy(x.cvg))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`SimpleVarianceState`](@ref) to the selected assets.

The Welford accumulator of one asset reads only the observations of that asset, so the slice of the state is the state of the sliced universe, entry for entry. All the assets share the count, so the method uses it without change. The slice copies by index and does not use a `view`. With a `view`, a later [`partial_fit!`](@ref) on the viewed estimator writes into the arrays of the original estimator.

# Arguments

  - `x`: The state to slice.
  - `i`: The index or the indices of the assets to keep.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `state::SimpleVarianceState`: The state of the same sample over the selected assets.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::SimpleVarianceState, i, args...)
    return SimpleVarianceState(x.n, x.mu[i], x.M[i], coverage_counts_view(x.cvg, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`SimpleVarianceState`](@ref) method of [`partial_fit!`](@ref) folds one observation into the count, the running mean and the per-asset accumulator.

# Mathematical definition

```math
\\begin{align}
n &\\leftarrow n + 1\\\\
\\boldsymbol{d} &= \\boldsymbol{x} - \\boldsymbol{\\mu}\\\\
\\boldsymbol{\\mu} &\\leftarrow \\boldsymbol{\\mu} + \\frac{\\boldsymbol{d}}{n}\\\\
\\boldsymbol{M} &\\leftarrow \\boldsymbol{M} + \\boldsymbol{d} \\odot (\\boldsymbol{x} - \\boldsymbol{\\mu})\\, .
\\end{align}
```

Where:

  - ``n``: The observation count.
  - ``\\boldsymbol{x}``: The observation.
  - ``\\boldsymbol{\\mu}``: The running mean.
  - ``\\boldsymbol{d}``: The deviation of the observation from the mean before the fold.
  - ``\\boldsymbol{M}``: The running per-asset accumulator.

The last line reads ``\\boldsymbol{\\mu}`` after the third line changed it, but ``\\boldsymbol{d}`` reads it before the change. This asymmetry is the one of Welford, and it keeps the accumulator non-negative.

# Algorithm

 1. Refuse an observation whose length is not the number of assets that the state describes.
 2. Add one to the count.
 3. Take the deviation `d` of the observation from the mean before the fold.
 4. Add `d`, divided by the new count, to `mu` in place.
 5. Add `d` times the deviation from the mean after the fold to `M`, in place.
 6. Build the state with the new count through its positional constructor, and return it.
"""
function partial_fit!(state::SimpleVarianceState, x::VecNum)
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    n = state.n + 1
    d = x .- state.mu
    state.mu .+= d ./ n
    state.M .+= d .* (x .- state.mu)
    return SimpleVarianceState(n, state.mu, state.M, state.cvg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `Nothing` method of the coverage arm of [`partial_fit!`](@ref) for a [`SimpleVarianceState`](@ref). An estimator that holds no [`CoveragePolicy`](@ref) folds through [`partial_fit!(state::SimpleVarianceState, x::VecNum)`](@ref). A plain state holds one count for all the assets, so it cannot skip an inactive cell. [`inactive_as_nonfinite`](@ref) writes `NaN` in that cell, and the variance of the asset is then `NaN`. The Asset Panel method gives the same `NaN` for an asset outside the Coverage Universe of the same rows.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`SimpleVarianceState`](@ref)
"""
function partial_fit!(state::SimpleVarianceState, x::VecNum, ::Nothing,
                      active_mask::Option{<:AbstractVector{<:Bool}})
    return partial_fit!(state, inactive_as_nonfinite(x, active_mask))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CoveragePolicy`](@ref) method of the coverage arm of [`partial_fit!`](@ref) for a [`SimpleVarianceState`](@ref). It folds one observation into the count, the running mean and the accumulator of each asset. Each asset reads only its own observations.

# Mathematical definition

```math
\\begin{align}
\\nu_j &\\leftarrow \\nu_j + 1\\\\
d_j &= r_{tj} - \\mu_j\\\\
\\mu_j &\\leftarrow \\mu_j + \\frac{d_j}{\\nu_j}\\\\
M_j &\\leftarrow M_j + d_j (r_{tj} - \\mu_j)\\, ,
\\end{align}
```

for each asset ``j`` that is finite and active at observation ``t``. The method applies no line to an asset that is not finite and active. Where:

  - ``\\nu_j``: The number of observations at which asset ``j`` is finite and active.
  - $(math_dict[:r_tj])
  - ``\\mu_j``: The running mean of asset ``j``.
  - ``d_j``: The deviation of asset ``j`` from its mean before the fold.
  - ``M_j``: The running second-moment accumulator of asset ``j``.

The last line reads ``\\mu_j`` after the third line changed it, which is the asymmetry of Welford. The accumulator is exact for each asset, and a skipped observation has no effect on the asset.

# Algorithm

 1. Refuse an observation whose length is not the number of assets that the state describes.
 2. Read the valid assets, and the assets that became inactive at this observation, with [`coverage_valid`](@ref).
 3. Apply the fold-time rule of the coverage algorithm with [`fold_inactive!`](@ref).
 4. Fold the return of each valid asset into the count, the mean and the accumulator of that asset.
 5. Update the per-asset records with [`coverage_step!`](@ref). Add one to the observation count, and return the state.

# Arguments

  - `state`: The state to fold into. The method changes it in place.
  - `x`: One observation, with one entry for each asset.
  - `cvg`: The policy that the estimator holds.
  - `active_mask`: The active mask of the Asset Panel at this observation, or `nothing`.

# Validation

  - `length(x)` is the number of assets that the state describes. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::SimpleVarianceState`: The state after the observation.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`coverage_valid`](@ref)
  - [`fold_inactive!`](@ref)
  - [`coverage_step!`](@ref)
"""
function partial_fit!(state::SimpleVarianceState, x::VecNum, cvg::CoveragePolicy,
                      active_mask::Option{<:AbstractVector{<:Bool}})
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    valid, ni = coverage_valid(x, active_mask, state.cvg)
    state = fold_inactive!(cvg.alg, state, ni)
    counts = state.cvg
    for i in eachindex(x, valid)
        if valid[i]
            nu = counts.nu[i] + 1
            counts.nu[i] = nu
            d = x[i] - state.mu[i]
            state.mu[i] += d / nu
            state.M[i] += d * (x[i] - state.mu[i])
        end
    end
    coverage_step!(counts, valid, active_mask)
    return SimpleVarianceState(state.n + 1, state.mu, state.M, counts)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`SimpleVarianceState`](@ref) method of [`fold_inactive!`](@ref) under [`ResetCoverage`](@ref). It sets to zero the count, the centre, the running mean and the accumulator of each asset that became inactive at this observation. A relisted asset then starts cold. The centre of a per-asset state is `nothing`, because its `mu` is already the centre of the cell. [`coverage_reset!`](@ref) does nothing to a `nothing`.

# Related

  - [`fold_inactive!`](@ref)
  - [`ResetCoverage`](@ref)
  - [`SimpleVarianceState`](@ref)
"""
function fold_inactive!(::ResetCoverage, state::SimpleVarianceState,
                        ni::AbstractVector{<:Bool})
    coverage_reset!(state.cvg.nu, ni)
    coverage_reset!(state.cvg.centre, ni)
    coverage_reset!(state.mu, ni)
    coverage_reset!(state.M, ni)
    return state
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into the partial-fit state of a [`SimpleVariance`](@ref) estimator.

This method is the block arm of the [`partial_fit!`](@ref) interface. The Welford update reads one observation at a time, so the method folds the block one row at a time. The result equals the result of the same rows given one at a time.

# Algorithm

 1. Orient `X` and `active_mask` to `observations × assets`. Transpose them when `dims == 2`.
 2. Fold each row in order with the single-observation arm of [`partial_fit!`](@ref), together with its row of the active mask. Rebind the estimator after each row.

# Arguments

  - `ve`: The variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, with the same size as `X`, or `nothing`.

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, its size is the size of `X` after the orientation. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `ve::SimpleVariance`: The estimator that holds the state after the last row.

# Related

  - [`SimpleVariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ve::SimpleVariance{<:Any, <:Any, <:Any, <:Any,
                                         <:Option{<:SimpleVarianceState}}, X::MatNum;
                      dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing)
    X = dims_oriented(dims, X)
    amsk = isnothing(active_mask) ? nothing : dims_oriented(dims, active_mask)
    if !isnothing(amsk)
        @argcheck(size(amsk) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(amsk)))"))
    end
    for i in axes(X, 1)
        ve = partial_fit!(ve, view(X, i, :);
                          active_mask = isnothing(amsk) ? nothing : view(amsk, i, :))
    end
    return ve
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The [`SimpleVariance`](@ref) method of [`partial_fit!`](@ref) folds one observation into the state of the `cache` field. On the first call, it seeds that state.

# Algorithm

 1. Refuse a configuration that no incremental fit can reproduce, with [`assert_partial_fittable`](@ref).
 2. Seed a [`SimpleVarianceState`](@ref) of zeros over `length(x)` assets when `ve.cache` holds `nothing`, with [`variance_state_seed`](@ref).
 3. Fold `x` into the state, with the policy `ve.cvg` and the active mask.
 4. Rebuild the estimator with the new state in `cache` through [`rebuild_estimator`](@ref), and return it.
"""
function partial_fit!(ve::SimpleVariance{<:Any, <:Any, <:Any, <:Any,
                                         <:Option{<:SimpleVarianceState}}, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    assert_partial_fittable(ve.me, ve.w, "SimpleVariance")
    state = variance_state_seed(ve.cache, x, ve.cvg)
    return rebuild_estimator(ve, (; cache = partial_fit!(state, x, ve.cvg, active_mask)))
end
"""
    variance_state_seed(cache::Nothing, x::VecNum,
                        cvg::Option{<:CoveragePolicy} = nothing) -> SimpleVarianceState
    variance_state_seed(cache::SimpleVarianceState, x::VecNum,
                        cvg::Option{<:CoveragePolicy} = nothing) -> SimpleVarianceState

Returns the [`SimpleVarianceState`](@ref) that an incremental variance fit folds into. If the estimator holds no state, the function seeds a state of zeros.

The seed is in this function and not in [`partial_fit!`](@ref), so the fold is one line, and the dispatch on the type of the `cache` field is in one place.

# Arguments

  - `cache`: The state that the estimator holds, or `nothing`.
  - `x`: One observation, `assets × 1`. The function reads only its length and its element type.
  - `cvg`: The policy that the estimator holds. It decides if the seed holds per-asset counts.

# Returns

  - `state::SimpleVarianceState`: The state that `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`partial_fit!`](@ref)
"""
function variance_state_seed(::Nothing, x::VecNum, cvg::Option{<:CoveragePolicy} = nothing)
    N = length(x)
    Tf = float_if_integer(eltype(x))
    return SimpleVarianceState(0, zeros(Tf, N), zeros(Tf, N),
                               coverage_counts_seed(cvg, nothing, N, Tf, false))
end
function variance_state_seed(cache::SimpleVarianceState, x::VecNum,
                             cvg::Option{<:CoveragePolicy} = nothing)
    counts = coverage_counts_seed(cvg, cache.cvg, length(x), float_if_integer(eltype(x)),
                                  false)
    return if counts === cache.cvg
        cache
    else
        SimpleVarianceState(cache.n, cache.mu, cache.M, counts)
    end
end
"""
    Statistics.var(
        ve::SimpleVariance,
        state::SimpleVarianceState
    ) -> VecNum
    Statistics.var(
        ve::SimpleVariance
    ) -> VecNum

Read the variance of an incremental fit from a [`SimpleVarianceState`](@ref).

The two-argument method reads a state that the caller holds. The one-argument method reads the state in the `cache` field of `ve`. Both methods return the per-asset variance as a vector, `assets × 1`. The batch method over a matrix returns a row when `dims = 1`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{M_j}{n - c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma2_hat_j])
  - ``M_j``: The running accumulator of asset ``j``.
  - ``n``: The observation count.
  - ``c``: One when `ve.corrected` is `true`, and zero otherwise.

# Algorithm

 1. Refuse a configuration that no incremental fit can reproduce, with [`assert_partial_fittable`](@ref).
 2. Take the divisor `n - c`. Return a vector of `NaN` when the divisor is less than one. `min_obs` reads an asset with too few observations in the same way.
 3. Otherwise divide the accumulator by the divisor.

# Arguments

  - $(arg_dict[:ve])
  - `state`: The state to read.

# Validation

  - `ve` holds no observation weights. An `ArgumentError` is thrown otherwise.
  - `ve.me` is `nothing`, or a [`SimpleExpectedReturns`](@ref) that holds no observation weights. An `ArgumentError` is thrown otherwise.
  - `ve.cache` is not `nothing`, for the one-argument method. An `ArgumentError` is thrown otherwise.

# Returns

  - `vr::VecNum`: The per-asset variance of the fit, `assets × 1`, with `NaN` where the state holds too few observations.

# Examples

```jldoctest
julia> ve = foldl(partial_fit!, eachrow([1.0 2.0; 3.0 4.0]); init = SimpleVariance());

julia> var(ve)
2-element Vector{Float64}:
 2.0
 2.0
```

# Related

  - [`SimpleVariance`](@ref)
  - [`SimpleVarianceState`](@ref)
  - [`partial_fit!`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, state::SimpleVarianceState)
    assert_partial_fittable(ve.me, ve.w, "SimpleVariance")
    return coverage_variance(ve, ve.cvg, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the `Nothing` method of [`coverage_variance`](@ref) for a state. `var(ve)` with no data calls it. All the assets share one count, so the result is all `NaN` until the count is more than the Bessel correction.

# Related

  - [`coverage_variance`](@ref)
  - [`SimpleVarianceState`](@ref)
"""
function coverage_variance(ve::SimpleVariance, ::Nothing, state::SimpleVarianceState)
    k = state.n - ve.corrected
    return k >= one(k) ? state.M ./ k : fill(convert(eltype(state.M), NaN), length(state.M))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

This is the [`CoveragePolicy`](@ref) method of [`coverage_variance`](@ref) for a state. `var(ve)` with no data calls it. The method divides the accumulator of each asset by the count of that asset less the Bessel correction. An asset that the policy refuses is `NaN`.

# Related

  - [`coverage_variance`](@ref)
  - [`coverage_admission`](@ref)
  - [`coverage_divide`](@ref)
"""
function coverage_variance(ve::SimpleVariance, cvg::CoveragePolicy,
                           state::SimpleVarianceState)
    counts = state.cvg
    return coverage_divide(state.M, counts.nu, ve.corrected,
                           coverage_admission(cvg, counts, state.n))
end
function Statistics.var(ve::SimpleVariance)
    return Statistics.var(ve, partial_fit_cache(ve))
end
"""
    Statistics.std(ve::SimpleVariance, state::SimpleVarianceState) -> VecNum

Read the standard deviation of an incremental fit from a [`SimpleVarianceState`](@ref).

`std(ve)` with no data reads the state in the `cache` field of `ve` through this method.

# Arguments

  - $(arg_dict[:ve])
  - `state`: The state to read.

# Returns

  - `sd::VecNum`: The per-asset standard deviation of the fit, `assets × 1`, with `NaN` where the state holds too few observations.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`Statistics.var(ve::SimpleVariance, state::SimpleVarianceState)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, state::SimpleVarianceState)
    return sqrt.(Statistics.var(ve, state))
end
# Every configuration of this family folds, for the reason [`SimpleExpectedReturns`](@ref)
# does (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(::SimpleVariance)
    return true
end
"""
    simple_variance_count(w::Nothing, corrected::Bool, f::AbstractVector{Bool})
    simple_variance_count(w::StatsBase.FrequencyWeights, corrected::Bool,
                          f::AbstractVector{Bool})
    simple_variance_count(w::StatsBase.AbstractWeights, corrected::Bool,
                          f::AbstractVector{Bool})

Return the effective count and the divisor of one column of a [`SimpleVariance`](@ref), over the rows that `f` marks.

The function dispatches on the weight type. Frequency weights count replicated observations, so their count is their sum. All other weight types count with the Kish formula. A column with no gap reads the weights without change, so the usual case copies nothing. The divisor is the divisor of `StatsBase`, stated in the count. It follows the bias correction of each weight type, and this function holds no copy of that rule.

# Mathematical definition

```math
\\begin{align}
n &= \\begin{cases}
\\left\\lvert \\mathcal{F} \\right\\rvert & \\textrm{with no weights}\\\\
\\sum_{t \\in \\mathcal{F}} w_{t} & \\textrm{with frequency weights}\\\\
\\dfrac{\\left(\\sum_{t \\in \\mathcal{F}} w_{t}\\right)^{2}}{\\sum_{t \\in \\mathcal{F}} w_{t}^{2}} & \\textrm{otherwise}
\\end{cases}\\,, \\\\
m &= \\dfrac{n \\, D}{\\sum_{t \\in \\mathcal{F}} w_{t}}\\,.
\\end{align}
```

Where:

  - ``n``: The effective count of the observations.
  - ``m``: The divisor of the variance for that count. With no weights it is ``n - 1`` when `corrected` is `true`, and ``n`` otherwise.
  - ``\\mathcal{F}``: The rows that `f` marks.
  - $(math_dict[:w_t_obs])
  - ``D``: The divisor of the weighted variance of `StatsBase`. It is the inverse of `StatsBase.varcorrection(w, corrected)`.

# Arguments

  - `w`: The observation weights, or `nothing`.
  - `corrected`: The `corrected` field of the estimator.
  - `f`: The rows of the column that the estimate reads.

# Returns

  - `(; n, m)::NamedTuple`: The effective count and the divisor.

# Related

  - [`variance_count`](@ref)
  - [`SimpleVariance`](@ref)
"""
function simple_variance_count(::Nothing, corrected::Bool, f::AbstractVector{Bool})
    n = count(f)
    return (; n = n, m = n - corrected)
end
function simple_variance_count(w::StatsBase.FrequencyWeights, corrected::Bool,
                               f::AbstractVector{Bool})
    wf = all(f) ? w : w[f]
    n = sum(wf)
    return (; n = n, m = inv(StatsBase.varcorrection(wf, corrected)))
end
function simple_variance_count(w::StatsBase.AbstractWeights, corrected::Bool,
                               f::AbstractVector{Bool})
    wf = all(f) ? w : w[f]
    s = sum(wf)
    n = s^2 / sum(abs2, wf)
    return (; n = n, m = n / (s * StatsBase.varcorrection(wf, corrected)))
end
function variance_count(ve::SimpleVariance, X::MatNum)
    w = get_observation_weights(ve.w, X; dims = 1)
    cnt = [simple_variance_count(w, ve.corrected, isfinite.(view(X, :, i)))
           for i in axes(X, 2)]
    return (; n = getindex.(cnt, :n), m = getindex.(cnt, :m))
end
export SimpleVariance, var, std
