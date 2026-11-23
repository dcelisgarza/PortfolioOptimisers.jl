"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the reweighting steps that follow a raw robust fit.

The raw MCD reads only `h` of the observations, so its efficiency is low. A reweighting step fits the location and the scatter again over the observations that the raw fit does not flag. [`MCDReweighting`](@ref) gives the hard weights of [rousseeuw1999](@cite).

# Interfaces

To implement a new reweighting step, subtype `AbstractRobustReweighting`, keep each parameter in a field, and implement one method:

  - `PortfolioOptimisers.mcd_reweight(rw, ce::MCDCovariance, X::MatNum, mu::VecNum, sigma::MatNum, idx::AbstractVector{<:Integer}, centre::Option{<:VecNum}) -> Tuple`: Returns the location, the scatter and the weight of each observation in the scatter, from the raw fit `(mu, sigma)` of the subset `idx`.

# Related

  - [`MCDReweighting`](@ref)
  - [`MCDCovariance`](@ref)
"""
abstract type AbstractRobustReweighting <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Fits the location and the scatter again over the observations whose raw squared distance is below the quantile `q` of the ``\\chi^2_p`` distribution.

Step 7 of the algorithm of [rousseeuw1999](@cite) gives the weight one to an observation inside the cut-off and zero to the rest. The new location is the mean of the kept observations, and the new scatter is their covariance with the divisor ``k - 1``, where ``k`` is their number. The scatter then gets the consistency factor of [`mcd_consistency`](@ref) at `q`, and the small-sample factor of the reweighted MCD when [`MCDCovariance`](@ref) has one. When every observation is kept, the scatter gets no factor, as in the R package `robustbase`, version 0.99-7.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MCDReweighting(;
        q::Number = 0.975
    ) -> MCDReweighting

Keywords correspond to the struct's fields.

## Validation

  - `0 < q < 1`.

# Examples

```jldoctest
julia> MCDReweighting()
MCDReweighting
  q ┴ Float64: 0.975
```

# Related

  - [`AbstractRobustReweighting`](@ref)
  - [`MCDCovariance`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 4, step 7.
"""
@concrete struct MCDReweighting <: AbstractRobustReweighting
    """
    $(field_dict[:rw_q])
    """
    q
    function MCDReweighting(q::Number)
        assert_open_probability(q, :q)
        return new{typeof(q)}(q)
    end
end
function MCDReweighting(; q::Number = 0.975)::MCDReweighting
    return MCDReweighting(q)
end
"""
    assert_open_probability(q::Number, name::Symbol)

Refuses a probability outside the open interval from zero to one.

# Arguments

  - `q`: The probability.
  - `name`: The name of the field, for the message.

# Validation

  - `0 < q < 1`. Otherwise, the function raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`MCDCovariance`](@ref)
  - [`TylerCovariance`](@ref)
"""
function assert_open_probability(q::Number, name::Symbol)
    @argcheck(zero(q) < q < one(q), DomainError(q, "`$(name)` must be in (0, 1), got $q"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the location and the covariance of the returns from the subset of `h` observations whose covariance matrix has the smallest determinant.

The minimum covariance determinant of [rousseeuw1999](@cite) reads only the `h` observations that are nearest to each other, so up to ``n - h`` bad observations cannot move it. At the default subset fraction of one half, ``h = \\lfloor (n + p + 1)/2 \\rfloor`` and the breakdown point is about one half. The estimate goes through these steps:

 1. The search `alg` finds the subset, see [`DetMCD`](@ref) and [`FastMCD`](@ref).
 2. The raw location is the mean of the subset, and the raw scatter is its covariance. The scatter gets the consistency factor of [croux1999](@cite), see [`mcd_consistency`](@ref), and the small-sample factor `ssc` of [pison2002](@cite), see [`MCDSmallSampleCorrection`](@ref).
 3. The reweighting step `rw` fits the location and the scatter again over the observations that the raw fit does not flag, see [`MCDReweighting`](@ref). `rw = nothing` gives the raw MCD.

`StatsAPI.fit` returns the location, the scatter, and the squared distance, the weight and the outlier flag of each observation in a [`RobustCovarianceResult`](@ref). `cov` returns the scatter of that fit.

`me = nothing` gives the joint location of the MCD. A given `me`, or a `mean` keyword, fixes the centre, and the search then reads the scatter of each subset about that centre with the divisor ``h``.

!!! warning

    A fixed centre that is not robust moves with the outliers. The search then measures the distances from a wrong centre. `SimpleExpectedReturns` is such an estimator. Use `me = nothing`, a [`MedianExpectedReturns`](@ref), or a [`RobustExpectedReturns`](@ref).

The estimator has no field of observation weights. [`factory`](@ref) passes the weights to `me`, and the scatter ignores them, because no standard weighted MCD exists.

The estimator has no exact update, so an [`Online`](@ref) wrapper refits it over the rows of its Sample Buffer, see [`SampleBufferState`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MCDCovariance(;
        me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
        alg::AbstractMCDAlgorithm = DetMCD(),
        alpha::Number = 0.5,
        ssc::Option{<:MCDSmallSampleCorrection} = MCDSmallSampleCorrection(),
        rw::Option{<:AbstractRobustReweighting} = MCDReweighting(),
        q::Number = 0.975,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> MCDCovariance

Keywords correspond to the struct's fields.

## Validation

  - `1/2 <= alpha <= 1`.
  - `0 < q < 1`.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> MCDCovariance()
MCDCovariance
     me ┼ nothing
    alg ┼ DetMCD()
  alpha ┼ Float64: 0.5
    ssc ┼ MCDSmallSampleCorrection()
     rw ┼ MCDReweighting
        │   q ┴ Float64: 0.975
      q ┴ Float64: 0.975
```

# Related

  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`DetMCD`](@ref)
  - [`FastMCD`](@ref)
  - [`MCDReweighting`](@ref)
  - [`MCDSmallSampleCorrection`](@ref)
  - [`TylerCovariance`](@ref)
  - [`RobustExpectedReturns`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Sections 2 and 4.
  - $(ref_dict[:hubert2012]) Section 3.
  - $(ref_dict[:croux1999]) Not read. See [`mcd_consistency`](@ref).
  - $(ref_dict[:pison2002]) Sections 3 and 4.
"""
@propagatable @concrete struct MCDCovariance <: AbstractRobustCovarianceEstimator
    """
    $(field_dict[:rob_me])
    """
    @fprop @vprop me
    """
    $(field_dict[:mcd_alg])
    """
    alg
    """
    $(field_dict[:mcd_alpha])
    """
    alpha
    """
    $(field_dict[:mcd_ssc])
    """
    ssc
    """
    $(field_dict[:mcd_rw])
    """
    rw
    """
    $(field_dict[:rob_q])
    """
    q
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function MCDCovariance(me::Option{<:AbstractExpectedReturnsEstimator},
                           alg::AbstractMCDAlgorithm, alpha::Number,
                           ssc::Option{<:MCDSmallSampleCorrection},
                           rw::Option{<:AbstractRobustReweighting}, q::Number,
                           cache::Option{<:AbstractPartialFitState})
        @argcheck(one(alpha) / 2 <= alpha <= one(alpha),
                  DomainError(alpha, "`alpha` must be in [1/2, 1], got $alpha"))
        assert_open_probability(q, :q)
        return new{typeof(me), typeof(alg), typeof(alpha), typeof(ssc), typeof(rw),
                   typeof(q), typeof(cache)}(me, alg, alpha, ssc, rw, q, cache)
    end
end
function MCDCovariance(; me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
                       alg::AbstractMCDAlgorithm = DetMCD(), alpha::Number = 0.5,
                       ssc::Option{<:MCDSmallSampleCorrection} = MCDSmallSampleCorrection(),
                       rw::Option{<:AbstractRobustReweighting} = MCDReweighting(),
                       q::Number = 0.975,
                       cache::Option{<:AbstractPartialFitState} = nothing)::MCDCovariance
    return MCDCovariance(me, alg, alpha, ssc, rw, q, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`MCDCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Set `set_show_nothing_fields!(:MCDCovariance, true)` to render it.

# Arguments

  - `::MCDCovariance`: The estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:me, :alg, :alpha, :ssc, :rw, :q)`.

# Related

  - [`MCDCovariance`](@ref)
  - [`show_fields`](@ref)
"""
show_fields(::MCDCovariance) = (:me, :alg, :alpha, :ssc, :rw, :q)
"""
    StatsAPI.fit(ce::MCDCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
        -> RobustCovarianceResult

Fits the minimum covariance determinant of [`MCDCovariance`](@ref) to the data.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), and refuse a gapped sample with [`assert_finite_sample`](@ref).
 2. Compute the subset size `h` with [`mcd_subset_size`](@ref), and refuse `h <= p` with [`assert_robust_rows`](@ref).
 3. Take the fixed centre with [`robust_centre`](@ref): the `mean` keyword, the mean of `ce.me`, or `nothing` for a joint location.
 4. Standardise the data with [`mcd_standardise`](@ref), and find the subset with [`mcd_search`](@ref).
 5. Fit the raw location and scatter of the subset in the units of the data with [`mcd_subset_fit`](@ref). Multiply the scatter by [`mcd_consistency`](@ref) at `h / n` and by the raw factor of [`mcd_correction`](@ref).
 6. Reweight with [`mcd_reweight`](@ref).
 7. Compute the squared distance of each observation with [`squared_distances`](@ref), and build the result with [`robust_result`](@ref).

# Arguments

  - `ce`: The MCD covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) A given centre fixes the location of the fit.
  - `kwargs...`: More keyword arguments for `ce.me`.

# Validation

  - $(val_dict[:dims])
  - `all(isfinite, X)`.
  - `h > p`. Otherwise, the method raises a `DomainError` that names `h`, `p` and the limit.
  - No column of `X` is constant, see [`mcd_standardise`](@ref).
  - The factor of `ce.ssc` is positive, see [`mcd_correction`](@ref).
  - No subset of `h` observations lies on a hyperplane. Otherwise, the method raises the `DomainError` of [`squared_distances`](@ref).

# Returns

  - `res::RobustCovarianceResult`: The fit. The weight of an observation comes from [`mcd_reweight`](@ref). With `rw = nothing` or [`MCDReweighting`](@ref), it is one when the observation is in the last subset that defines `sigma`, and zero otherwise.

# Related

  - [`MCDCovariance`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`mcd_search`](@ref)
"""
function StatsAPI.fit(ce::MCDCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                      kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    n, p = size(X)
    h = mcd_subset_size(ce.alpha, n, p)
    assert_robust_rows(h, p, "the size h of the MCD subset")
    centre = robust_centre(ce.me, X; mean = mean, kwargs...)
    Z = mcd_standardise(X, centre)
    idx = mcd_search(ce.alg, Z, h, !isnothing(centre))
    mu, sigma = mcd_fit_rows(X, idx, centre)
    sigma .*= mcd_consistency(p, h / n) * mcd_correction(ce.ssc, p, n, ce.alpha, :raw)
    mu, sigma, w = mcd_reweight(ce.rw, ce, X, mu, sigma, idx, centre)
    return robust_result(mu, sigma, squared_distances(X, mu, sigma), w, ce.q)
end
"""
    mcd_fit_rows(X::MatNum, idx::AbstractVector{<:Integer}, centre::Option{<:VecNum})
        -> Tuple{VecNum, MatNum}

Returns the location and the scatter of the observations `idx` of `X`, in the units of the data.

With no fixed centre, these are the mean and the covariance with the divisor ``k - 1`` of the ``k`` observations. With a fixed centre, the location is the centre and the scatter has the divisor ``k``, see [`mcd_subset_fit`](@ref).

# Arguments

  - `X`: The data matrix, `observations × assets`.
  - `idx`: The positions of the observations.
  - `centre`: The fixed centre, or `nothing`.

# Returns

  - `(mu, sigma)`: The location and the scatter.

# Related

  - [`mcd_subset_fit`](@ref)
  - [`StatsAPI.fit(ce::MCDCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function mcd_fit_rows(X::MatNum, idx::AbstractVector{<:Integer}, centre::Option{<:VecNum})
    if isnothing(centre)
        return mcd_subset_fit(X, idx, false)
    end
    return centre, last(mcd_subset_fit(X .- transpose(centre), idx, true))
end
"""
    mcd_reweight(rw::Nothing, ce::MCDCovariance, X::MatNum, mu::VecNum, sigma::MatNum,
                 idx::AbstractVector{<:Integer}, centre::Option{<:VecNum})
    mcd_reweight(rw::MCDReweighting, ce::MCDCovariance, X::MatNum, mu::VecNum,
                 sigma::MatNum, idx::AbstractVector{<:Integer}, centre::Option{<:VecNum})

Applies the reweighting step of an [`MCDCovariance`](@ref) to the raw fit, and returns the location, the scatter and the weight of each observation in the scatter.

The `nothing` method returns the raw fit and its subset. The [`MCDReweighting`](@ref) method keeps the observations whose raw squared distance is below the quantile `rw.q` of the ``\\chi^2_p`` distribution, and fits them with [`mcd_fit_rows`](@ref). When it does not keep every observation, it multiplies the scatter by [`mcd_consistency`](@ref) at `rw.q` and by the reweighted factor of [`mcd_correction`](@ref).

# Arguments

  - `rw`: The reweighting step, or `nothing`.
  - `ce`: The estimator, for its subset fraction and its small-sample correction.
  - `X`: The data matrix, `observations × assets`.
  - `mu`, `sigma`: The raw location and scatter.
  - `idx`: The positions of the raw subset.
  - `centre`: The fixed centre, or `nothing`.

# Returns

  - `(mu, sigma, w)`: The location, the scatter, and the weight of each observation in the scatter, one or zero for both methods.

# Related

  - [`MCDReweighting`](@ref)
  - [`mcd_fit_rows`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 4, step 7.
"""
function mcd_reweight(::Nothing, ::MCDCovariance, X::MatNum, mu::VecNum, sigma::MatNum,
                      idx::AbstractVector{<:Integer}, ::Option{<:VecNum})
    w = zeros(eltype(sigma), size(X, 1))
    w[idx] .= one(eltype(w))
    return mu, sigma, w
end
function mcd_reweight(rw::MCDReweighting, ce::MCDCovariance, X::MatNum, mu::VecNum,
                      sigma::MatNum, ::AbstractVector{<:Integer}, centre::Option{<:VecNum})
    n, p = size(X)
    # The factor depends on the shape of the sample alone, so a sample that is too small for it
    # raises whether or not every observation is kept.
    f = mcd_consistency(p, rw.q) * mcd_correction(ce.ssc, p, n, ce.alpha, :rew)
    cut = Distributions.quantile(Distributions.Chisq(p), rw.q)
    keep = squared_distances(X, mu, sigma) .< cut
    mu, sigma = mcd_fit_rows(X, findall(keep), centre)
    if !all(keep)
        sigma .*= f
    end
    return mu, sigma, convert(Vector{eltype(sigma)}, keep)
end

export MCDReweighting, MCDCovariance
public AbstractRobustReweighting, mcd_reweight
