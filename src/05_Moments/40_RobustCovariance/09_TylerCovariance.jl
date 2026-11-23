"""
$(DocStringExtensions.TYPEDEF)

Estimates the covariance of the returns from the shape matrix of Tyler's M-estimator, which resists heavy tails.

The shape ``\\mathbf{V}`` of [tyler1987](@cite) solves a fixed-point equation in which each observation has the weight ``p / d^2``, where ``d^2`` is its squared distance from the centre in the metric of ``\\mathbf{V}``. So an observation far from the centre has a small weight, and its direction alone enters the shape. The estimator is distribution-free over the elliptical distributions, and it exists when the data are in general position with ``n > p``. It does not exist when a subspace of dimension ``r`` through the centre holds a share of the observations of ``r/p`` or more, for example when half of the returns of one of two assets are zero. The fixed point then does not converge, and the fit warns. A cluster of outliers at one place can hold such a share when ``p`` is large, so this estimator and its shrunk form resist heavy tails and scattered outliers, and [`MCDCovariance`](@ref) resists a cluster. The equation fixes the shape and not the scale.

No copy of [tyler1987](@cite) or of [hettmansperger2002](@cite) was found, so the two equations below follow the code of the R package `ICSNP`, version 1.2-0.

  - `me = nothing` gives the joint location of [hettmansperger2002](@cite): the spatial median in the coordinates that the shape makes spherical, solved with the shape. A given `me`, or a `mean` keyword, fixes the centre. That is the classical estimator of [tyler1987](@cite).
  - `ve = nothing` gives the scale ``\\mathbf{\\Sigma} = \\mathbf{V} \\, \\mathrm{med}(d^2) / \\chi^2_{p, 1/2}``, which is robust and consistent at the normal model. A given `ve` keeps the correlation of ``\\mathbf{V}`` and takes the standard deviations from `std(ve, X)`, as [`GerberCovariance`](@ref) does.
  - `shr` shrinks the shape to a target, see [`TylerShrinkage`](@ref). The estimate is then the shrunk Tyler shape of [chen2011](@cite) or of [ollila2014](@cite), and it exists with fewer observations than assets. With `ve = nothing`, the scale makes the trace of the covariance the sum of the robust variances of the assets, because the median of the distances is far below the one of the ``\\chi^2_p`` distribution when ``p > n``. The cut-off of the outliers does not assume that distribution, see [`robust_outliers`](@ref).
  - `w` gives the weighted fixed point ``\\sum_t w_t \\boldsymbol{x}_t \\boldsymbol{x}_t^\\intercal / d_t^2 \\, / \\sum_t w_t``, a weighted spatial median and a weighted median for the scale. A `StatsBase.FrequencyWeights` gives the fit of the data with each observation repeated by its weight. Another weight type gives the weighted median of `StatsBase` for that type, which interpolates between the observations.

`StatsAPI.fit` returns the location, the covariance, and the squared distance, the weight and the outlier flag of each observation in a [`RobustCovarianceResult`](@ref). The weight is ``p / d^2`` in the metric of the covariance, and it does not include the observation weight. An observation at the centre has no direction, so it leaves the fixed point and has the weight zero. With a given `ve`, the covariance has other distances than the shape, so the weight is not the weight of the fixed point. `cov` returns the covariance of that fit.

!!! warning

    A fixed centre that is not robust moves with the outliers, and a variance estimator that is not robust gives the outliers a large weight in the scale. `SimpleExpectedReturns` and `SimpleVariance` are such estimators. Use `me = nothing` or a [`MedianExpectedReturns`](@ref) or a [`RobustExpectedReturns`](@ref), and `ve = nothing`.

The estimator has no exact update, so an [`Online`](@ref) wrapper refits it over the rows of its Sample Buffer, see [`SampleBufferState`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathbf{V} &= \\frac{p}{\\sum_t w_t} \\sum_{t=1}^{T} w_t \\frac{(\\boldsymbol{x}_t - \\boldsymbol{\\mu})(\\boldsymbol{x}_t - \\boldsymbol{\\mu})^\\intercal}{d_t^2}\\,, \\qquad d_t^2 = (\\boldsymbol{x}_t - \\boldsymbol{\\mu})^\\intercal \\mathbf{V}^{-1} (\\boldsymbol{x}_t - \\boldsymbol{\\mu})\\,, \\\\
\\boldsymbol{0} &= \\sum_{t=1}^{T} w_t \\frac{\\boldsymbol{x}_t - \\boldsymbol{\\mu}}{d_t}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{V}``: The shape matrix, normalised to the trace ``p``.
  - ``\\boldsymbol{\\mu}``: The location. The second equation holds for the joint location alone.
  - ``w_t``: The observation weight, one when `w` is `nothing`.
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TylerCovariance(;
        me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
        ve::Option{<:StatsBase.CovarianceEstimator} = nothing,
        shr::Option{<:TylerShrinkage} = nothing,
        w::Option{<:ObsWeights} = nothing,
        q::Number = 0.975,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> TylerCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - `0 < q < 1`.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated through [`factory`](@ref).
  - `ve`: Recursively updated through [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `ve`: Recursively viewed through [`port_opt_view`](@ref).
  - `shr`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> TylerCovariance()
TylerCovariance
   me ┼ nothing
   ve ┼ nothing
  shr ┼ nothing
    w ┼ nothing
    q ┴ Float64: 0.975
```

# Related

  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`MCDCovariance`](@ref)
  - [`TylerShrinkage`](@ref)
  - [`RobustExpectedReturns`](@ref)

# References

  - $(ref_dict[:tyler1987])
  - $(ref_dict[:hettmansperger2002])
  - $(ref_dict[:chen2011]) Section III.
  - $(ref_dict[:ollila2014]) Section V.
"""
@propagatable @concrete struct TylerCovariance <: AbstractRobustCovarianceEstimator
    """
    $(field_dict[:rob_me])
    """
    @fprop @vprop me
    """
    $(field_dict[:tyler_ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:tyler_shr])
    """
    @vprop shr
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:rob_q])
    """
    q
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function TylerCovariance(me::Option{<:AbstractExpectedReturnsEstimator},
                             ve::Option{<:StatsBase.CovarianceEstimator},
                             shr::Option{<:TylerShrinkage}, w::Option{<:ObsWeights},
                             q::Number, cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        assert_open_probability(q, :q)
        return new{typeof(me), typeof(ve), typeof(shr), typeof(w), typeof(q),
                   typeof(cache)}(me, ve, shr, w, q, cache)
    end
end
function TylerCovariance(; me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
                         ve::Option{<:StatsBase.CovarianceEstimator} = nothing,
                         shr::Option{<:TylerShrinkage} = nothing,
                         w::Option{<:ObsWeights} = nothing, q::Number = 0.975,
                         cache::Option{<:AbstractPartialFitState} = nothing)::TylerCovariance
    return TylerCovariance(me, ve, shr, w, q, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`TylerCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Set `set_show_nothing_fields!(:TylerCovariance, true)` to render it.

# Arguments

  - `::TylerCovariance`: The estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:me, :ve, :shr, :w, :q)`.

# Related

  - [`TylerCovariance`](@ref)
  - [`show_fields`](@ref)
"""
show_fields(::TylerCovariance) = (:me, :ve, :shr, :w, :q)
"""
    TYLER_MAX_ITER

The largest number of fixed-point iterations of [`tyler_fixed_point`](@ref), 1000.

# Related

  - [`tyler_fixed_point`](@ref)
"""
const TYLER_MAX_ITER = 1000
"""
    StatsAPI.fit(ce::TylerCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
        -> RobustCovarianceResult

Fits the location and the covariance of [`TylerCovariance`](@ref) to the data.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), and refuse a gapped sample with [`assert_finite_sample`](@ref).
 2. Resolve the observation weights with [`get_observation_weights`](@ref), and refuse a vector of another length.
 3. Take the fixed centre with [`robust_centre`](@ref), or `nothing` for the joint location.
 4. Find the location and the covariance with [`tyler_fit`](@ref).
 5. Compute the squared distance of each observation in the metric of the covariance with [`squared_distances`](@ref). The weight of an observation is `p / d2`, and zero for an observation that [`away_from_centre`](@ref) excludes.
 6. Flag the outliers with [`robust_outliers`](@ref).

# Arguments

  - `ce`: The Tyler covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) A given centre fixes the location of the fit.
  - `kwargs...`: More keyword arguments for `ce.me` and `ce.ve`.

# Validation

  - $(val_dict[:dims])
  - `all(isfinite, X)`.
  - The observation weights have one entry for each observation.
  - With `shr = nothing`, more than `p` observations of positive weight. Otherwise, the method raises a `DomainError` that names `n`, `p` and the limit.
  - With a `shr`, the intensity is above the one at which the shape exists, see [`assert_tyler_intensity`](@ref).

# Returns

  - `res::RobustCovarianceResult`: The fit.

# Related

  - [`TylerCovariance`](@ref)
  - [`tyler_fit`](@ref)
"""
function StatsAPI.fit(ce::TylerCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                      kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    n, p = size(X)
    w = get_observation_weights(ce.w, X)
    @argcheck(isnothing(w) || length(w) == n,
              DimensionMismatch("the observation weights must have one entry for each of the $n observations. Got $(length(w))"))
    centre = robust_centre(ce.me, X; mean = mean, kwargs...)
    mu, sigma, rho = tyler_fit(ce.shr, ce.ve, X, w, centre; kwargs...)
    d2 = squared_distances(X, mu, sigma)
    return RobustCovarianceResult(mu, sigma, d2,
                                  ifelse.(away_from_centre(d2), p ./ d2, zero(eltype(d2))),
                                  robust_outliers(rho, d2, p, ce.q), rho)
end
function effective_observation_count(ce::TylerCovariance, X::MatNum; kwargs...)
    return observation_weight_count(get_observation_weights(ce.w, X), size(X, 1))
end
"""
    away_from_centre(d2::VecNum) -> BitVector

Marks the observations that have a direction from the centre in a fit of [`TylerCovariance`](@ref).

An observation at the centre has the squared distance zero, so its term in the equations of the fixed point is not defined. The function reads a distance of at most `eps` times the median of the distances as zero.

# Arguments

  - `d2`: The squared distance of each observation.

# Returns

  - `pos::BitVector`: The observations away from the centre.

# Related

  - [`tyler_fixed_point`](@ref)
"""
function away_from_centre(d2::VecNum)
    return d2 .> eps(eltype(d2)) * Statistics.median(d2)
end
"""
    tyler_fixed_point(X::MatNum, w::Option{<:StatsBase.AbstractWeights},
                      centre::Option{<:VecNum}, reg::Option{<:ShapeShrinkage})
        -> Tuple{VecNum, MatNum}

Solves the fixed-point equations of [`TylerCovariance`](@ref) for the shape and, with no fixed centre, for the location.

# Algorithm

 1. Start the location at the coordinatewise weighted median, or at `centre`. Start the shape at the diagonal matrix of the squares of the robust scales of the centred columns, from [`robust_column_scale`](@ref), scaled to the trace `p`. The start does not move the fixed point, and an outlier of any size does not make it singular.
 2. Repeat at most [`TYLER_MAX_ITER`](@ref) times:
     1. Compute the squared distances `d2` in the metric of the shape with [`squared_distances`](@ref). An observation that [`away_from_centre`](@ref) does not mark leaves both sums.
     2. Update the shape with [`tyler_shape_step`](@ref), and scale it to the trace `p`, or shrink it with [`shrink_shape!`](@ref).
     3. With no fixed centre, update the location with [`spatial_median_step`](@ref).
     4. Stop when the relative change of the shape and the change of the location in the metric of the shape are both below `sqrt(eps)`.
 3. Warn when the iterations do not converge, and return the last iterate.

# Arguments

  - `X`: The data matrix, `observations × assets`.
  - `w`: The observation weights, or `nothing`.
  - `centre`: The fixed centre, or `nothing`.
  - `reg`: The shrinkage to the identity matrix, or `nothing`.

# Returns

  - `(mu, V)`: The location and the shape. The shape has the trace `p`, unless the form of `reg` is [`PenalisedShrinkage`](@ref).

# Related

  - [`TylerCovariance`](@ref)
  - [`tyler_shape_step`](@ref)
  - [`spatial_median_step`](@ref)

# References

  - $(ref_dict[:tyler1987])
  - $(ref_dict[:hettmansperger2002])
"""
function tyler_fixed_point(X::MatNum, w::Option{<:StatsBase.AbstractWeights},
                           centre::Option{<:VecNum}, reg::Option{<:ShapeShrinkage})
    wt = observation_weight_vector(X, w)
    mu = if isnothing(centre)
        [observation_median(view(X, :, j), w) for j in axes(X, 2)]
    else
        centre
    end
    D = X .- transpose(mu)
    s = map(robust_column_scale, eachcol(D))
    V = trace_normalise!(Matrix(LinearAlgebra.Diagonal(abs2.(s))))
    tol = sqrt(eps(real(eltype(V))))
    for _ in 1:TYLER_MAX_ITER
        d2 = squared_distances(X, mu, V)
        pos = away_from_centre(d2)
        Vs = tyler_shape_step(D, d2, wt, pos)
        Vn = isnothing(reg) ? trace_normalise!(Vs) : shrink_shape!(reg.alg, Vs, reg.rho)
        mun = isnothing(centre) ? spatial_median_step(X, mu, V, d2, wt, pos) : mu
        dmu = squared_distances(transpose(mun), mu, V)[1]
        done = LinearAlgebra.norm(Vn - V) <= tol * LinearAlgebra.norm(V) && dmu <= tol^2
        mu, V = mun, Vn
        D = X .- transpose(mu)
        if done
            return mu, V
        end
    end
    @warn("The fixed point of Tyler's M-estimator did not converge in $(TYLER_MAX_ITER) iterations. The last iterate is returned.",
          maxlog = log_limit(:warn), _id = @log_id())
    return mu, V
end
"""
    observation_median(x::VecNum, w::Option{<:StatsBase.AbstractWeights}) -> Number

Returns the median of `x`, weighted by the observation weights `w` when they are not `nothing`.

The weighted median is the one of `StatsBase`. Under `StatsBase.FrequencyWeights` it is the median of the data with each value repeated by its weight.

# Arguments

  - `x`: The values.
  - `w`: The observation weights, or `nothing`.

# Returns

  - `m::Number`: The median.

# Related

  - [`tyler_fixed_point`](@ref)
  - [`tyler_scale`](@ref)
"""
function observation_median(x::VecNum, ::Nothing)
    return Statistics.median(x)
end
function observation_median(x::VecNum, w::StatsBase.AbstractWeights)
    return Statistics.median(x, w)
end
"""
    trace_normalise!(V::MatNum) -> MatNum

Scales the shape matrix `V` in place to the trace `p`, its number of rows.

# Arguments

  - `V`: The shape matrix.

# Returns

  - `V::MatNum`: The input matrix, scaled.

# Related

  - [`tyler_fixed_point`](@ref)
"""
function trace_normalise!(V::MatNum)
    V .*= size(V, 1) / LinearAlgebra.tr(V)
    return V
end
"""
    tyler_shape_step(D::MatNum, d2::VecNum, wt::VecNum, pos::AbstractVector{Bool}) -> Matrix

Returns one step of the shape equation of [`TylerCovariance`](@ref): ``p \\sum_t w_t \\boldsymbol{x}_t \\boldsymbol{x}_t^\\intercal / d_t^2 \\, / \\sum_t w_t`` over the observations of `pos`.

The step multiplies each row by the square root of its weight and takes one product of the result with itself, so the matrix is exactly symmetric.

# Arguments

  - `D`: The data minus the location, `observations × assets`.
  - `d2`: The squared distance of each observation.
  - `wt`: The weight of each observation.
  - `pos`: The observations away from the centre.

# Returns

  - `V::Matrix`: The new shape, before the normalisation.

# Related

  - [`tyler_fixed_point`](@ref)
"""
function tyler_shape_step(D::MatNum, d2::VecNum, wt::VecNum, pos::AbstractVector{Bool})
    A = ifelse.(pos, sqrt.(wt ./ d2), zero(eltype(d2))) .* D
    return size(D, 2) * (transpose(A) * A) / sum(wt[pos])
end
"""
    spatial_median_step(X::MatNum, mu::VecNum, V::MatNum, d2::VecNum, wt::VecNum,
                        pos::AbstractVector{Bool}) -> Vector

Returns one step of the weighted spatial median in the metric of the shape `V`, the location equation of [`TylerCovariance`](@ref).

The step is the Weiszfeld step ``\\tilde{\\boldsymbol{\\mu}} = \\sum_t (w_t/d_t) \\boldsymbol{x}_t / \\sum_t (w_t/d_t)`` over the observations away from the centre. When some observations are at the centre, with the total weight ``\\eta``, the step is the modification of Vardi and Zhang: ``\\max(0, 1 - \\eta/r) \\tilde{\\boldsymbol{\\mu}} + \\min(1, \\eta/r) \\boldsymbol{\\mu}``, where ``r`` is the length of ``\\sum_t w_t (\\boldsymbol{x}_t - \\boldsymbol{\\mu})/d_t`` in the metric of `V`. The R package `ICSNP`, version 1.2-0, takes the same step in its spatial median.

# Arguments

  - `X`: The data matrix, `observations × assets`.
  - `mu`: The location.
  - `V`: The shape.
  - `d2`: The squared distance of each observation.
  - `wt`: The weight of each observation.
  - `pos`: The observations away from the centre.

# Returns

  - `mu::Vector`: The new location.

# Related

  - [`tyler_fixed_point`](@ref)

# References

  - $(ref_dict[:hettmansperger2002])
"""
function spatial_median_step(X::MatNum, mu::VecNum, V::MatNum, d2::VecNum, wt::VecNum,
                             pos::AbstractVector{Bool})
    a = ifelse.(pos, wt ./ sqrt.(d2), zero(eltype(d2)))
    mt = vec(transpose(X) * a) / sum(a)
    eta = sum(wt[.!pos])
    if iszero(eta)
        return mt
    end
    R = vec(transpose(X .- transpose(mu)) * a)
    r = sqrt(LinearAlgebra.dot(R, V \ R))
    return max(zero(r), 1 - eta / r) * mt + min(one(r), eta / r) * mu
end
"""
    tyler_scale(ve::Nothing, X::MatNum, mu::VecNum, V::MatNum, w::Option{<:StatsBase.AbstractWeights},
                v::Nothing; kwargs...) -> Matrix
    tyler_scale(ve::Nothing, X::MatNum, mu::VecNum, V::MatNum, w::Option{<:StatsBase.AbstractWeights},
                v::VecNum; kwargs...) -> Matrix
    tyler_scale(ve::StatsBase.CovarianceEstimator, X::MatNum, mu::VecNum, V::MatNum,
                w::Option{<:StatsBase.AbstractWeights}, v::Option{<:VecNum}; kwargs...) -> Matrix

Scales the shape of [`TylerCovariance`](@ref) to a covariance matrix.

The first `nothing` method multiplies the shape by the weighted median of the squared distances in its metric, divided by the median of the ``\\chi^2_p`` distribution. At the normal model, half of the squared distances in the metric of the covariance are below that median, so the scale is consistent. The second `nothing` method is the one of a shrunk shape. It scales the trace to the sum of the robust variances `v`. The estimator method keeps the correlation of the shape and takes the standard deviations from `std(ve, X)`, through [`library_covariance_estimator`](@ref) for an estimator of another package.

# Arguments

  - `ve`: The variance estimator, or `nothing`.
  - `X`: The data matrix, `observations × assets`.
  - `mu`: The location.
  - `V`: The shape.
  - `w`: The observation weights, or `nothing`.
  - `v`: The robust variance of each asset for a shrunk shape, or `nothing`.
  - `kwargs...`: More keyword arguments for `std(ve, X)`.

# Returns

  - `sigma::Matrix`: The covariance matrix.

# Related

  - [`TylerCovariance`](@ref)
"""
function tyler_scale(::Nothing, X::MatNum, mu::VecNum, V::MatNum,
                     w::Option{<:StatsBase.AbstractWeights}, ::Nothing; kwargs...)
    md = observation_median(squared_distances(X, mu, V), w)
    med = Distributions.quantile(Distributions.Chisq(size(X, 2)), 0.5)
    return V .* (md / convert(typeof(md), med))
end
function tyler_scale(::Nothing, ::MatNum, ::VecNum, V::MatNum,
                     ::Option{<:StatsBase.AbstractWeights}, v::VecNum; kwargs...)
    return V .* (sum(v) / LinearAlgebra.tr(V))
end
function tyler_scale(ve::StatsBase.CovarianceEstimator, X::MatNum, ::VecNum, V::MatNum,
                     ::Option{<:StatsBase.AbstractWeights}, ::Option{<:VecNum}; kwargs...)
    sd = vec(Statistics.std(library_covariance_estimator(ve), X; dims = 1, kwargs...))
    return StatsBase.cor2cov!(StatsBase.cov2cor(V), sd)
end

export TylerCovariance
