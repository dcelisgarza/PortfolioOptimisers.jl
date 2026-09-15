"""
$(DocStringExtensions.TYPEDEF)

Computes the marginal variance and standard deviation, optionally weighted and optionally bias-corrected.

`me` centres the data when no `mean` is supplied, `w` weights the observations, and `corrected` selects the bias correction. `me` reaches the matrix methods only: the vector methods leave the centring to `Statistics`.

`w` weights the whole estimate, so it reaches the centre as well as the deviations. The matrix methods send `me` through [`factory`](@ref), which replaces the weights of `me` with `w`, and `Statistics` centres a weighted vector on its weighted mean. Both paths therefore answer the same number over the same data, and `w` wins over the weights that `me` carries. Pass `mean` for a centre that `w` does not describe. ADR 0088 records the decision.

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
  - `corrected = true` needs a weight type that carries a bias correction. See the bias-correction bullet of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `me`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).
  - `cache`: Dropped via [`obs_weights_view`](@ref), because no slice of a state exists on the observation axis.

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

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:SimpleVariance, true)` to render it. ADR 0105 records the decision. `cvg` is read from the instance rather than from the type, as it is for [`SimpleExpectedReturns`](@ref), which states the reason.

# Arguments

  - `ve`: Variance estimator, read for its `cvg` field.

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

Dispersion kernel shared by the [`SimpleVariance`](@ref) methods of `Statistics.std` and `Statistics.var`.

# Algorithm

The matrix method:

 1. Check that `dims` is `1` or `2`.
 2. Resolve the centring vector `mu` from `me` and `ve.w` with [`weighted_centre`](@ref), which reads `mean` when the caller gave one.
 3. Resolve the observation weights from `ve.w` against `X` with [`get_observation_weights`](@ref), giving `w`.
 4. When `w` is `nothing`, call `f(X; dims = dims, corrected = ve.corrected, mean = mu)`.
 5. Otherwise call `f(X, w, dims; corrected = ve.corrected, mean = mu)`.

The vector method:

 1. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 2. When `w` is `nothing`, call `f(X; corrected = ve.corrected, mean = mean)`.
 3. Otherwise call `f(X, w; corrected = ve.corrected, mean = mean)`.

The two methods reach one centre by two routes. The matrix method resolves a centre before it calls `f`, and [`weighted_centre`](@ref) takes that centre from `me` after [`factory`](@ref) writes `ve.w` into it. The vector method passes `mean` through, so a `mean` of `nothing` leaves `f` to centre on the **weighted** mean of `X`. One `SimpleVariance` therefore answers a one-column matrix and the matching vector with one number. `ve.w` wins over the weights that `me` carries, which is what [`factory`](@ref) does on every other path. ADR 0088 records the decision, and `mean` takes any other centre.

[`weighted_centre`](@ref) calls [`factory`](@ref) only when `ve.w` is not `nothing`. That test is a performance guard and not a second contract: `ve.w` is a field, so its type decides the branch, and the guard keeps a windowed loop from rebuilding the estimator tree of `me` once per window.

# Arguments

  - `f`: Dispersion function to apply, either `Statistics.std` or `Statistics.var`.
  - `ve::SimpleVariance`: Variance estimator. Supplies the observation weights and the `corrected` flag.
  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used when no `mean` is provided. Matrix methods only.
  - `X::VecNum_MatNum`: Data matrix or vector.
  - `dims::Int = 1`: Dimension along which to operate. Matrix methods only.
  - `mean = nothing`: Precomputed mean.
  - `kwargs...`: Forwarded to the mean and weight resolution. Matrix methods only.

# Validation

  - $(val_dict[:dims]) Matrix methods only.

# Returns

  - `sigma::Union{<:Number, <:ArrNum}`: Dispersion of `X` computed by `f`.

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

The `cvg` field of the estimator is passed as the third argument, so the arm is chosen by **dispatch on the policy** rather than by a branch on its value, exactly as [`coverage_mean`](@ref) does for a sample mean. `f` is `Statistics.var` or `Statistics.std`, and the available-case arm maps one onto the other through [`coverage_moment_map`](@ref).

# Arguments

  - `f`: `Statistics.var` or `Statistics.std`.
  - `ve`: Variance estimator.
  - `cvg`: The policy the estimator carries, which selects the arm.
  - `me`: The estimator that centres the fit.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `mean`: A precomputed centre, or `nothing`.
  - `active_mask`: The active mask of the Asset Panel, `observations × assets`, or `nothing`. The Coverage Universe arm ignores it.
  - `kwargs...`: Additional keyword arguments passed to the centring estimator and the weights.

# Returns

  - `val`: The dispersion, of the shape `f` gives.

# Related

  - [`CoveragePolicy`](@ref)
  - [`SimpleVariance`](@ref)
  - [`coverage_mean`](@ref)
  - [`coverage_moment_map`](@ref)
"""
function coverage_variance end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Nothing` method of [`coverage_variance`](@ref). The Coverage Universe arm, which is the body [`simple_variance_kernel`](@ref) has always had: it refuses a gapped sample with [`assert_finite_sample`](@ref) and ignores the active mask.

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
    assert_finite_sample(X)
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

[`CoveragePolicy`](@ref) method of [`coverage_variance`](@ref). The available-case arm: each asset's dispersion is fitted on that asset's own finite and active observations, and the batch answer **is** the incremental one, because the arm folds the block through [`partial_fit!`](@ref) and reads the state out.

The centre is the fold's own running mean, so a `mean` a caller passes is refused rather than silently dropped: an available-case fit centres each asset on that asset's own observations, and a centre fitted over the whole window is not that.

# Algorithm

 1. Fold every row of `X` into a fresh state with [`partial_fit!`](@ref), carrying the active mask.
 2. Read the state out with [`var(ve::SimpleVariance, state::SimpleVarianceState)`](@ref).
 3. Map the variance onto the answer `f` asks for with [`coverage_moment_map`](@ref), and orient it as the caller's `dims` asks.

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

Maps an available-case variance onto the answer the caller's verb asks for.

[`coverage_variance`](@ref) fits a variance whichever verb called it, because the state carries a second-moment accumulator and nothing else, so the standard deviation is its square root. Dispatching on the verb rather than comparing it keeps the choice at compile time.

# Arguments

  - `f`: `Statistics.var` or `Statistics.std`.
  - `v`: The available-case variance.

# Returns

  - `val::VecNum`: `v` itself, or its entrywise square root.

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

`Nothing` method of the vector arm of [`coverage_variance`](@ref). The Coverage Universe arm over one asset's series, which refuses a gapped series with [`assert_finite_sample`](@ref).

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

[`CoveragePolicy`](@ref) method of the vector arm of [`coverage_variance`](@ref). One asset has no pair, so available-case estimation over its series is the ordinary fit over the finite entries of it, and the coverage floor is read against the length of the series.

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
    Tf = typeof(zero(eltype(X)) / one(Int))
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

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a matrix.

This method computes the standard deviation of the input matrix `X` using the configuration specified in `ve`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_j &= \\sqrt{\\hat{\\sigma}^2_j}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_j``: Estimated standard deviation of asset ``j``.
  - $(math_dict[:sigma2_hat_j])

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines ``\\hat{\\sigma}^2_j`` in each of the four cases that `ve.w` and `ve.corrected` select.

# Algorithm

 1. Check that `dims` is `1` or `2`.
 2. When `mean` is `nothing`, compute the centring vector `mu` with `ve.me`, after [`factory`](@ref) writes `ve.w` into it; otherwise take `mu` from `mean`.
 3. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 4. When `w` is `nothing`, take the unweighted standard deviation of `X` along `dims`, centred on `mu`.
 5. Otherwise take the standard deviation of `X` weighted by `w` along `dims`, centred on `mu`.

``\\hat{\\mu}_j`` comes from `ve.me`, and `ve.w` reaches `ve.me` through [`factory`](@ref). A `SimpleVariance` whose `w` is set therefore weights the centre and the squared deviations alike, so a vector and its one-column matrix answer the same number. Pass `mean` for any other centre. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - $(val_dict[:dims])
  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

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

`SimpleVariance{Nothing}` overload of [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
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

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the standard deviation of the input vector `X` using the configuration specified in `ve`.

# Mathematical definition

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines the variance in each of the four cases that `ve.w` and `ve.corrected` select, and this method returns its square root.

# Algorithm

 1. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 2. When `w` is `nothing`, take the unweighted standard deviation of `X`, centred on `mean`.
 3. Otherwise take the standard deviation of `X` weighted by `w`, centred on `mean`.

The vector methods ignore `ve.me`: a `mean` of `nothing` reaches `Statistics.std`, which centres on the mean of `X` — the **weighted** mean when `w` is not `nothing`. The matrix methods resolve the centre from `ve.me` under the same `ve.w`, so the two paths answer the same number for the same data. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Validation

  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

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

Compute the variance using a [`SimpleVariance`](@ref) estimator for a matrix.

This method computes the variance of the input matrix `X` using the configuration specified in `ve`.

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
 2. When `mean` is `nothing`, compute the centring vector `mu` with `ve.me`, after [`factory`](@ref) writes `ve.w` into it; otherwise take `mu` from `mean`.
 3. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 4. When `w` is `nothing`, take the unweighted variance of `X` along `dims`, centred on `mu`.
 5. Otherwise take the variance of `X` weighted by `w` along `dims`, centred on `mu`.

``\\hat{\\mu}_j`` comes from `ve.me`, and `ve.w` reaches `ve.me` through [`factory`](@ref). A `SimpleVariance` whose `w` is set therefore weights the centre and the squared deviations alike, so a vector and its one-column matrix answer the same number. Pass `mean` for any other centre. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - $(val_dict[:dims])
  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

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

`SimpleVariance{Nothing}` overload of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
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

Compute the variance using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the variance of the input vector `X` using the configuration specified in `ve`.

# Mathematical definition

[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref) defines the variance in each of the four cases that `ve.w` and `ve.corrected` select, and this method returns it for a single series.

# Algorithm

 1. Resolve the observation weights from `ve.w` against `X`, giving `w`.
 2. When `w` is `nothing`, take the unweighted variance of `X`, centred on `mean`.
 3. Otherwise take the variance of `X` weighted by `w`, centred on `mean`.

The vector methods ignore `ve.me`: a `mean` of `nothing` reaches `Statistics.var`, which centres on the mean of `X` — the **weighted** mean when `w` is not `nothing`. The matrix methods resolve the centre from `ve.me` under the same `ve.w`, so the two paths answer the same number for the same data. ADR 0088 records the decision.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Validation

  - `corrected = true` needs a weight type that carries a bias correction. A plain `StatsBase.Weights` carries none, and `StatsBase` raises an `ArgumentError`.

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

Carries the running observation count, mean and per-asset second-moment accumulator of an incremental variance fit.

The state of [`SimpleVariance`](@ref) under [`partial_fit!`](@ref). `M` is the accumulator ``\\sum_t (r_{tj} - \\hat{\\mu}_j)^2`` and not the variance, so [`var(ve::SimpleVariance, state::SimpleVarianceState)`](@ref) divides it by the count, or by the count less one when `ve.corrected` holds.

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

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `mu`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `M`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
 2. Fold the counts, the means and the accumulators with [`chan_merge`](@ref), whose elementwise method reads a per-asset accumulator.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::SimpleVarianceState`: The state the two blocks give when they are fitted as one block.

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

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. The count is a scalar and passes through, and the running mean and the per-asset accumulator are copied.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::SimpleVarianceState`: A fresh state, equal to `x`, whose `mu` and `M` are fresh vectors.

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

The Welford accumulator of one asset reads that asset's observations alone, and reads no other asset. So the slice of the state is the state of the sliced universe, entry for entry, and the count is shared by every asset and passes through. The slice copies by index and does not `view`: a later [`partial_fit!`](@ref) on the viewed estimator would otherwise write through into the arrays of the estimator the view was taken from.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments (ignored).

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

[`SimpleVarianceState`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the running count, mean and per-asset accumulator.

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

  - ``n``: observation count.
  - ``\\boldsymbol{x}``: the observation.
  - ``\\boldsymbol{\\mu}``: the running mean.
  - ``\\boldsymbol{d}``: deviation of the observation from the mean **before** the fold.
  - ``\\boldsymbol{M}``: the running per-asset accumulator.

The last line reads ``\\boldsymbol{\\mu}`` **after** the third line moved it, where ``\\boldsymbol{d}`` read it before. That asymmetry is Welford's, and it is what keeps the accumulator non-negative.

# Algorithm

 1. Refuse an observation whose length is not the number of assets the state describes.
 2. Add one to the count.
 3. Take the deviation of the observation from the mean before the fold, giving `d`.
 4. Move `mu` in place along `d`, by the reciprocal of the new count.
 5. Add `d` times the deviation from the mean **after** the fold to `M`, in place.
 6. Rebind the count with `Accessors.@reset`, and return the state.
"""
function partial_fit!(state::SimpleVarianceState, x::VecNum)
    @argcheck(length(x) == length(state.mu),
              DimensionMismatch("the observation must have one entry per asset, but the state describes $(length(state.mu)) assets and `x` has $(length(x)) entries."))
    n = state.n + 1
    d = x .- state.mu
    state.mu .+= d ./ n
    state.M .+= d .* (x .- state.mu)
    return Accessors.@reset state.n = n
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Nothing` method of the coverage arm of [`partial_fit!`](@ref) for a [`SimpleVarianceState`](@ref). An estimator that carries no [`CoveragePolicy`](@ref) folds through [`partial_fit!(state::SimpleVarianceState, x::VecNum)`](@ref), and the active mask is ignored.

# Related

  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`SimpleVarianceState`](@ref)
"""
function partial_fit!(state::SimpleVarianceState, x::VecNum, ::Nothing,
                      ::Option{<:AbstractVector{<:Bool}})
    return partial_fit!(state, x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`CoveragePolicy`](@ref) method of the coverage arm of [`partial_fit!`](@ref) for a [`SimpleVarianceState`](@ref). Folds one observation into the running per-asset count, mean and accumulator, reading each asset's own observations alone.

# Mathematical definition

```math
\\begin{align}
\\nu_j &\\leftarrow \\nu_j + 1\\\\
d_j &= r_{tj} - \\mu_j\\\\
\\mu_j &\\leftarrow \\mu_j + \\frac{d_j}{\\nu_j}\\\\
M_j &\\leftarrow M_j + d_j (r_{tj} - \\mu_j)\\, ,
\\end{align}
```

for every asset ``j`` that is finite and active at observation ``t``, and no line at all for an asset that is not. Where:

  - ``\\nu_j``: the number of observations at which asset ``j`` was finite and active.
  - $(math_dict[:r_tj])
  - ``\\mu_j``: the running mean of asset ``j``.
  - ``d_j``: the deviation of asset ``j`` from its mean before the fold.
  - ``M_j``: the running second-moment accumulator of asset ``j``.

The last line reads ``\\mu_j`` after the third line moved it, which is Welford's asymmetry, so the accumulator is exact per asset and a skipped observation costs the asset nothing.

# Algorithm

 1. Refuse an observation whose length is not the number of assets the state describes.
 2. Read the valid assets and the newly inactive ones with [`coverage_valid`](@ref).
 3. Apply the algorithm's fold-time rule with [`fold_inactive!`](@ref).
 4. Fold each valid asset's return into its own count, mean and accumulator.
 5. Move the per-asset bookkeeping on with [`coverage_step!`](@ref), add one to the observation count, and return the state.

# Arguments

  - `state`: The state to fold into, mutated in place.
  - `x`: One observation, one entry per asset.
  - `cvg`: The policy the estimator carries.
  - `active_mask`: The active mask of the Asset Panel at this observation, or `nothing`.

# Validation

  - `length(x)` is the number of assets the state describes. A `DimensionMismatch` is thrown otherwise.

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
    return Accessors.@reset state.n = state.n + 1
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SimpleVarianceState`](@ref) method of [`fold_inactive!`](@ref) under [`ResetCoverage`](@ref). Zeroes the count, the centre, the running mean and the accumulator of every asset that has just gone inactive, so that a relisting starts the asset cold. The centre of a per-asset state is `nothing`, because its `mu` is already the cell's centre, and [`coverage_reset!`](@ref) passes that through.

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

The block arm of the [`partial_fit!`](@ref) interface. Welford's update reads one observation at a time, so the block is folded row by row and the answer is the answer of the same rows handed over one at a time.

# Algorithm

 1. Orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Fold each row in turn with the single-observation arm of [`partial_fit!`](@ref), rebinding the estimator each time.

# Arguments

  - `ve`: Variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])

# Returns

  - `ve::SimpleVariance`: The estimator carrying the state after the last row.

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

[`SimpleVariance`](@ref) method of [`partial_fit!`](@ref). Folds one observation into the state the `cache` field carries, seeding it on the first call.

# Algorithm

 1. Refuse a configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Seed a [`SimpleVarianceState`](@ref) of zeros over `length(x)` assets when `ve.cache` holds `nothing`, with [`variance_state_seed`](@ref).
 3. Fold `x` into the state.
 4. Rebind `ve.cache` with `Accessors.@reset`, and return the estimator.
"""
function partial_fit!(ve::SimpleVariance{<:Any, <:Any, <:Any, <:Any,
                                         <:Option{<:SimpleVarianceState}}, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    assert_partial_fittable(ve.me, ve.w, "SimpleVariance")
    state = variance_state_seed(ve.cache, x, ve.cvg)
    return Accessors.@reset ve.cache = partial_fit!(state, x, ve.cvg, active_mask)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the [`SimpleVarianceState`](@ref) an incremental variance fit folds into, seeding one of zeros when the estimator carries none.

The seed is written here rather than inside [`partial_fit!`](@ref), so the fold reads as one line and the branch that reads the `cache` field has one home.

# Arguments

  - `cache`: The state the estimator carries, or `nothing`.
  - `x`: One observation, `assets × 1`, read for its length and its element type.

# Returns

  - `state::SimpleVarianceState`: The state `cache` holds, or a state of zeros over `length(x)` assets.

# Related

  - [`SimpleVarianceState`](@ref)
  - [`partial_fit!`](@ref)
"""
function variance_state_seed(cache::Option{<:SimpleVarianceState}, x::VecNum,
                             cvg::Option{<:CoveragePolicy} = nothing)
    N = length(x)
    Tf = typeof(zero(eltype(x)) / one(Int))
    return if isnothing(cache)
        SimpleVarianceState(0, zeros(Tf, N), zeros(Tf, N),
                            coverage_counts_seed(cvg, nothing, N, Tf, false))
    else
        Accessors.@reset cache.cvg = coverage_counts_seed(cvg, cache.cvg, N, Tf, false)
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

Read the variance of an incremental fit out of a [`SimpleVarianceState`](@ref).

The two-argument method reads a state the caller holds, and the one-argument method reads the state the `cache` field of `ve` carries. Both return the per-asset variance as a vector, `assets × 1`, where the batch method over a matrix returns a row when `dims = 1`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{M_j}{n - c}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma2_hat_j])
  - ``M_j``: running accumulator of asset ``j``.
  - ``n``: observation count.
  - ``c``: one when `ve.corrected` holds, and zero otherwise.

# Algorithm

 1. Refuse a configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Take the divisor `n - c`, and return a vector of `NaN` when it is below one, in the way `min_obs` reads an asset with too few observations.
 3. Otherwise divide the accumulator by the divisor.

# Arguments

  - $(arg_dict[:ve])
  - `state`: The state to read.

# Validation

  - `ve` carries no observation weights. An `ArgumentError` is thrown otherwise.
  - `ve.me` is a [`SimpleExpectedReturns`](@ref) carrying no observation weights, or `nothing`. An `ArgumentError` is thrown otherwise.
  - `ve.cache` is not `nothing`, for the one-argument method. An `ArgumentError` is thrown otherwise.

# Returns

  - `vr::VecNum`: Per-asset variance of the fit, `assets × 1`, or `NaN` where the state holds too few observations.

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

`Nothing` method of the read-out arm of [`coverage_variance`](@ref). Every asset shares one count, so the whole answer is `NaN` until the count passes the Bessel correction.

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

[`CoveragePolicy`](@ref) method of the read-out arm of [`coverage_variance`](@ref). Each asset's accumulator is divided by that asset's own count less the Bessel correction, and an asset the policy refuses is `NaN`.

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
# Every configuration of this family folds, for the reason [`SimpleExpectedReturns`](@ref)
# does (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(::SimpleVariance)
    return true
end
export SimpleVariance, var, std
