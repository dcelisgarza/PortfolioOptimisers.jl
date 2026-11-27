"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the reweighting steps that follow a raw robust fit.

The raw MCD reads only `h` of the observations, so its efficiency is low. A reweighting step fits the location and the scatter again from the raw fit. [`MCDReweighting`](@ref) gives the hard weights of [rousseeuw1999](@cite) to the observations that the raw fit does not flag. [`BisquareReweighting`](@ref) gives the soft weights of a bisquare M-step, which is also the second part of [`MMCovariance`](@ref).

# Interfaces

To implement a new reweighting step, subtype `AbstractRobustReweighting`, keep each parameter in a field, and implement one method:

  - `PortfolioOptimisers.mcd_reweight(rw, ce::MCDCovariance, smp::MCDSample, raw::RobustCovarianceResult) -> RobustCovarianceResult`: Returns the fit of the step from the raw fit.

# Related

  - [`MCDReweighting`](@ref)
  - [`BisquareReweighting`](@ref)
  - [`MCDCovariance`](@ref)
"""
abstract type AbstractRobustReweighting <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Fits the location and the scatter again over the observations that the raw fit does not flag as outliers.

Step 7 of the algorithm of [rousseeuw1999](@cite) gives the weight one to an observation whose raw squared distance is inside the quantile `q` of the ``\\chi^2_p`` distribution, and zero to the rest. With a shrinkage, the cut-off is the one of [`robust_outliers`](@ref). The new location is the mean of the kept observations, and the new scatter is their covariance with the divisor ``k - 1``, where ``k`` is their number. The scatter then gets the consistency factor of [`mcd_consistency`](@ref) at `q`, and the small-sample factor of the reweighted MCD when [`MCDCovariance`](@ref) has one. When every observation is kept, the scatter gets no factor, as in the R package `robustbase`, version 0.99-7.

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
 3. The reweighting step `rw` fits the location and the scatter again over the observations that the raw fit does not flag, see [`MCDReweighting`](@ref). [`BisquareReweighting`](@ref) gives soft weights in place of the hard ones. `rw = nothing` gives the raw MCD.

`StatsAPI.fit` returns the location, the scatter, and the squared distance, the weight and the outlier flag of each observation in a [`RobustCovarianceResult`](@ref). `cov` returns the scatter of that fit.

`me = nothing` gives the joint location of the MCD. A given `me`, or a `mean` keyword, fixes the centre, and the search then reads the scatter of each subset about that centre with the divisor ``h``.

!!! warning

    A fixed centre that is not robust moves with the outliers. The search then measures the distances from a wrong centre. `SimpleExpectedReturns` is such an estimator. Use `me = nothing`, a [`MedianExpectedReturns`](@ref), or a [`RobustExpectedReturns`](@ref).

`shr` shrinks the scatter of each subset to a target, see [`MCDShrinkage`](@ref). The estimate is then the minimum regularised covariance determinant of [boudt2020](@cite), and it exists with fewer observations than assets. These rules change:

  - The subset holds ``\\lceil \\alpha n \\rceil`` observations, see [`mcd_subset_size`](@ref).
  - `ssc` is `nothing`, because the small-sample factors hold for a scatter with no shrinkage.
  - The squared distance of an observation of the subset is its distance from the fit that leaves it out, and the cut-off of the outliers does not assume the ``\\chi^2_p`` distribution, see [`leave_one_out!`](@ref) and [`robust_outliers`](@ref).
  - `rw = nothing` gives the estimate of [boudt2020](@cite). The default reweights it, which is an addition of this library. The last paragraph of that work states that a better cut-off would improve the estimate by reweighting. On normal samples with one factor and the subset fraction one half, the reweighting makes the error of the covariance 46 % smaller with 100 observations of 100 assets, and 24 % smaller with 250 observations of 500 assets.

The estimator has no field of observation weights. [`factory`](@ref) passes the weights to `me`, and the scatter ignores them, because no standard weighted MCD exists.

The estimator has no exact update, so an [`Online`](@ref) wrapper refits it over the rows of its Sample Buffer, see [`SampleBufferState`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MCDCovariance(;
        me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
        alg::AbstractMCDAlgorithm = DetMCD(),
        alpha::Number = 0.5,
        shr::Option{<:MCDShrinkage} = nothing,
        ssc::Option{<:MCDSmallSampleCorrection} = isnothing(shr) ?
                                                  MCDSmallSampleCorrection() : nothing,
        rw::Option{<:AbstractRobustReweighting} = MCDReweighting(),
        q::Number = 0.975,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> MCDCovariance

Keywords correspond to the struct's fields.

## Validation

  - `1/2 <= alpha <= 1`.
  - `0 < q < 1`.
  - `shr` or `ssc` is `nothing`.
  - `shr` is `nothing` when `rw` is a [`BisquareReweighting`](@ref). The M-step needs more observations than assets, and it does not shrink.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed through [`port_opt_view`](@ref).
  - `shr`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> MCDCovariance()
MCDCovariance
     me ┼ nothing
    alg ┼ DetMCD()
  alpha ┼ Float64: 0.5
    shr ┼ nothing
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
  - [`MCDShrinkage`](@ref)
  - [`TylerCovariance`](@ref)
  - [`RobustExpectedReturns`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Sections 2 and 4.
  - $(ref_dict[:hubert2012]) Section 3.
  - $(ref_dict[:croux1999]) Not read. See [`mcd_consistency`](@ref).
  - $(ref_dict[:pison2002]) Sections 3 and 4.
  - $(ref_dict[:boudt2020]) Sections 2 and 3.
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
    $(field_dict[:mcd_shr])
    """
    @vprop shr
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
                           shr::Option{<:MCDShrinkage},
                           ssc::Option{<:MCDSmallSampleCorrection},
                           rw::Option{<:AbstractRobustReweighting}, q::Number,
                           cache::Option{<:AbstractPartialFitState})
        @argcheck(one(alpha) / 2 <= alpha <= one(alpha),
                  DomainError(alpha, "`alpha` must be in [1/2, 1], got $alpha"))
        assert_open_probability(q, :q)
        @argcheck(isnothing(shr) || isnothing(ssc),
                  ArgumentError("the small-sample correction holds for an MCD with no shrinkage. Set `ssc = nothing` with a `shr`"))
        @argcheck(isnothing(shr) || !isa(rw, BisquareReweighting),
                  ArgumentError("the bisquare M-step needs more observations than assets, and it does not shrink. Use `rw = MCDReweighting()` or `rw = nothing` with a `shr`"))
        return new{typeof(me), typeof(alg), typeof(alpha), typeof(shr), typeof(ssc),
                   typeof(rw), typeof(q), typeof(cache)}(me, alg, alpha, shr, ssc, rw, q,
                                                         cache)
    end
end
function MCDCovariance(; me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
                       alg::AbstractMCDAlgorithm = DetMCD(), alpha::Number = 0.5,
                       shr::Option{<:MCDShrinkage} = nothing,
                       ssc::Option{<:MCDSmallSampleCorrection} = if isnothing(shr)
                           MCDSmallSampleCorrection()
                       else
                           nothing
                       end, rw::Option{<:AbstractRobustReweighting} = MCDReweighting(),
                       q::Number = 0.975,
                       cache::Option{<:AbstractPartialFitState} = nothing)::MCDCovariance
    return MCDCovariance(me, alg, alpha, shr, ssc, rw, q, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`MCDCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Set `set_show_nothing_fields!(:MCDCovariance, true)` to render it.

# Arguments

  - `::MCDCovariance`: The estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:me, :alg, :alpha, :shr, :ssc, :rw, :q)`.

# Related

  - [`MCDCovariance`](@ref)
  - [`show_fields`](@ref)
"""
show_fields(::MCDCovariance) = (:me, :alg, :alpha, :shr, :ssc, :rw, :q)
"""
    StatsAPI.fit(ce::MCDCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
        -> RobustCovarianceResult

Fits the minimum covariance determinant of [`MCDCovariance`](@ref) to the data.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), and refuse a gapped sample with [`assert_finite_sample`](@ref).
 2. Compute the subset size `h` with [`mcd_subset_size`](@ref), which refuses a sample that is too small.
 3. Take the fixed centre with [`robust_centre`](@ref): the `mean` keyword, the mean of `ce.me`, or `nothing` for a joint location.
 4. Standardise the data with [`mcd_sample`](@ref), and find the subset and the shrinkage intensity with [`mcd_search`](@ref).
 5. Fit the raw location and scatter of the subset in the units of the data with [`mcd_fit`](@ref), with the product of [`mcd_consistency`](@ref) at `h / n` and of the raw factor of [`mcd_correction`](@ref).
 6. Reweight with [`mcd_reweight`](@ref).

# Arguments

  - `ce`: The MCD covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) A given centre fixes the location of the fit.
  - `kwargs...`: More keyword arguments for `ce.me`.

# Validation

  - $(val_dict[:dims])
  - `all(isfinite, X)`.
  - With `shr = nothing`, `h > p`. Otherwise, the method raises a `DomainError` that names `h`, `p` and the limit.
  - No column of `X` is constant, see [`mcd_standardise`](@ref).
  - The factor of `ce.ssc` is positive, see [`mcd_correction`](@ref).
  - The scatter of each subset is positive definite. With `shr = nothing` this means that no subset of `h` observations lies on a hyperplane. Otherwise, the method raises the `DomainError` of [`squared_distances`](@ref).

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
    h = mcd_subset_size(ce.shr, ce.alpha, n, p)
    smp = mcd_sample(ce.shr, X, robust_centre(ce.me, X; mean = mean, kwargs...))
    c = mcd_consistency(p, h / n)
    idx, rho = mcd_search(ce.alg, smp, h, c)
    raw = mcd_fit(ce, smp, idx, c * mcd_correction(ce.ssc, p, n, ce.alpha, :raw), rho)
    return mcd_reweight(ce.rw, ce, smp, raw)
end
"""
    mcd_sample(shr::Nothing, X::MatNum, centre::Option{<:VecNum}) -> MCDSample
    mcd_sample(shr::MCDShrinkage, X::MatNum, centre::Option{<:VecNum}) -> MCDSample

Builds the data that the fit of an [`MCDCovariance`](@ref) reads.

The `nothing` method standardises the data with [`mcd_standardise`](@ref). The [`MCDShrinkage`](@ref) method also builds the target with [`robust_target`](@ref) from the pilot of [`mcd_pilot`](@ref), and multiplies the standardised data by the inverse of the symmetric root of the target. The target is then the identity matrix.

[boudt2020](@cite) takes the eigenvectors of the target in place of the symmetric root. The search then reads rotated data, and the rotation is not unique when two eigenvalues are equal, as in an equicorrelation target. The two give one target, and differ in the starts of the search alone.

# Arguments

  - `shr`: The shrinkage, or `nothing`.
  - `X`: The data matrix, `observations × assets`.
  - `centre`: The fixed centre, or `nothing`.

# Returns

  - `smp::MCDSample`: The data of the fit.

# Related

  - [`MCDSample`](@ref)
  - [`mcd_search`](@ref)

# References

  - $(ref_dict[:boudt2020]) Section 2.2, Equations 6 to 8.
"""
function mcd_sample(::Nothing, X::MatNum, centre::Option{<:VecNum})
    return MCDSample(X, first(mcd_standardise(X, centre)), nothing, nothing, centre)
end
function mcd_sample(shr::MCDShrinkage, X::MatNum, centre::Option{<:VecNum})
    Z, s = mcd_standardise(X, centre)
    # A column with no robust scale has the scale one in `s`, so its variance in the pilot is zero.
    ps = ifelse.(iszero.(quantile_scale.(eachcol(Z), nothing)), zero(eltype(s)), s)
    T = robust_target(shr.tgt, mcd_pilot(shr.tgt, X, ps))
    D = LinearAlgebra.Diagonal(inv.(s))
    return MCDSample(X, Z / symmetric_root(D * T * D), T, shr.intensity, centre)
end
"""
    mcd_pilot(tgt::Union{IdentityTarget, ScaledIdentityTarget, DiagonalTarget, MatNum},
              X::MatNum, s::VecNum) -> LinearAlgebra.Diagonal
    mcd_pilot(tgt::CovarianceShrinkageTarget, X::MatNum, s::VecNum) -> Matrix

Returns the robust pilot covariance that the target rule of an [`MCDShrinkage`](@ref) reads.

A rule that reads the variances alone gets the diagonal matrix of the robust variances. Another rule gets the sine transform of the Kendall correlations, ``\\sin(\\pi \\tau / 2)``, scaled by the robust scales. [boudt2020](@cite) takes the mean Kendall correlation for its equicorrelation target. The sine transform is an addition of this library. It makes the correlation consistent at an elliptical model. The R package `rrcov`, version 1.7-7, applies the same transform to the Spearman correlations, for which it is not the consistent one, so the two equicorrelation targets differ.

# Arguments

  - `tgt`: The target.
  - `X`: The data matrix, `observations × assets`.
  - `s`: The robust scale of each column, zero for a column with no robust scale.

# Returns

  - `pilot::AbstractMatrix`: The pilot, in the units of the data.

# Related

  - [`robust_target`](@ref)
  - [`MCDShrinkage`](@ref)

# References

  - $(ref_dict[:boudt2020]) Section 2.4, Equation 15, and Section 4.
"""
function mcd_pilot(::Union{IdentityTarget, ScaledIdentityTarget, DiagonalTarget, MatNum},
                   ::MatNum, s::VecNum)
    return LinearAlgebra.Diagonal(abs2.(s))
end
function mcd_pilot(::CovarianceShrinkageTarget, X::MatNum, s::VecNum)
    return s .* sinpi.(StatsBase.corkendall(X) ./ 2) .* transpose(s)
end
"""
    subset_shrinkage(rho::Nothing, c::Number, p::Integer) -> Nothing
    subset_shrinkage(rho::Number, c::Number, p::Integer) -> SubsetShrinkage

Returns the [`SubsetShrinkage`](@ref) of a search, whose target is the identity matrix.

# Arguments

  - `rho`: The intensity, or `nothing`.
  - `c`: The consistency factor.
  - `p`: The number of assets.

# Returns

  - `reg::Option{<:SubsetShrinkage}`: The shrinkage of the search, or `nothing`.

# Related

  - [`mcd_search`](@ref)
"""
function subset_shrinkage(::Nothing, ::Number, ::Integer)
    return nothing
end
function subset_shrinkage(rho::Number, c::Number, p::Integer)
    return SubsetShrinkage(rho, c, LinearAlgebra.Diagonal(fill(one(rho), p)))
end
"""
    mcd_intensity(intensity::Nothing, W::MatNum, idx::AbstractVector{<:Integer}, c::Number, fixed::Bool)
    mcd_intensity(rho::Number, W::MatNum, idx::AbstractVector{<:Integer}, c::Number, fixed::Bool)
    mcd_intensity(rule::ConditionNumberIntensity, W::MatNum, idx::AbstractVector{<:Integer},
                  c::Number, fixed::Bool)
    mcd_intensity(rule::SphericityIntensity, W::MatNum, idx::AbstractVector{<:Integer},
                  c::Number, fixed::Bool)

Returns the shrinkage intensity of one subset of an [`MCDCovariance`](@ref).

A number is the intensity, and `nothing` gives `nothing`. Each rule states its formula under `# Mathematical definition`.

# Arguments

  - `intensity`: The rule, the intensity, or `nothing`.
  - `W`: The data, in the coordinates that turn the target into the identity matrix.
  - `idx`: The positions of the subset.
  - `c`: The factor of the scatter of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `rho::Option{<:Number}`: The intensity.

# Related

  - [`AbstractSubsetShrinkageIntensity`](@ref)
  - [`tyler_intensity`](@ref)
  - [`pooled_intensity`](@ref)
"""
function mcd_intensity(::Nothing, ::MatNum, ::AbstractVector{<:Integer}, ::Number, ::Bool)
    return nothing
end
function mcd_intensity(rho::Number, ::MatNum, ::AbstractVector{<:Integer}, ::Number, ::Bool)
    return rho
end
function mcd_intensity(rule::ConditionNumberIntensity, W::MatNum,
                       idx::AbstractVector{<:Integer}, c::Number, fixed::Bool)
    S = last(mcd_subset_fit(W, idx, fixed))
    vals = LinearAlgebra.eigvals!(LinearAlgebra.Symmetric(shrink_scatter!(S, c)))
    a = last(vals) - rule.max_cond * max(first(vals), zero(eltype(vals)))
    # A scatter inside the bound has a negative `a`, which can also make the denominator negative.
    a = max(a, zero(a))
    return a / (rule.max_cond - 1 + a)
end
function mcd_intensity(::SphericityIntensity, W::MatNum, idx::AbstractVector{<:Integer},
                       ::Number, fixed::Bool)
    n, p = size(W)
    D = fixed ? W : W .- Statistics.median(W; dims = 1)
    g = sphericity(spatial_sign_scatter(D, observation_weight_vector(W, nothing)), n)
    b = (g + p) / (length(idx) - !fixed)
    return b / (g - 1 + b)
end
"""
    pooled_intensity(intensity::Nothing, W::MatNum, starts, c::Number, fixed::Bool)
    pooled_intensity(intensity::Union{Number, AbstractSubsetShrinkageIntensity}, W::MatNum,
                     starts, c::Number, fixed::Bool)

Returns one shrinkage intensity for every start of an MCD search, and the starts that it permits.

Every subset must have the same intensity, so that the determinants compare. Steps 3.3 to 3.5 of the algorithm of [boudt2020](@cite) take the intensity of each start. A start that holds outliers can make its own intensity too large, so a start whose intensity is above the largest of 0.1 and of the median leaves the search. The pooled intensity is the largest one of the starts that stay, the smallest value at which each of them meets the rule. The R package `rrcov`, version 1.7-7, does the same. The text of [boudt2020](@cite) takes the cut-off itself when an intensity is above 0.1.

# Arguments

  - `intensity`: The rule, the intensity, or `nothing`.
  - `W`: The data, in the coordinates that turn the target into the identity matrix.
  - `starts`: The start subsets.
  - `c`: The consistency factor.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `(rho, starts)`: The intensity, or `nothing` with no shrinkage, and the starts.

# Related

  - [`mcd_intensity`](@ref)
  - [`mcd_search`](@ref)

# References

  - $(ref_dict[:boudt2020]) Section 3, steps 3.3 to 3.5 of the algorithm.
"""
function pooled_intensity(::Nothing, ::MatNum, starts, ::Number, ::Bool)
    return nothing, starts
end
function pooled_intensity(intensity::Union{Number, AbstractSubsetShrinkageIntensity},
                          W::MatNum, starts, c::Number, fixed::Bool)
    rhos = [mcd_intensity(intensity, W, idx, c, fixed) for idx in starts]
    lim = one(eltype(rhos)) / 10
    keep = rhos .<= max(lim, Statistics.median(rhos))
    return maximum(rhos[keep]), starts[keep]
end
"""
    mcd_fit(ce::MCDCovariance, smp::MCDSample, idx::AbstractVector{<:Integer}, c::Number,
            rho::Option{<:Number}) -> RobustCovarianceResult

Fits the location and the scatter of the observations `idx`, in the units of the data.

# Algorithm

 1. Compute the location and the scatter with [`mcd_fit_rows`](@ref).
 2. Multiply the scatter by `c`, and blend it with the target at the intensity `rho`, with [`shrink_scatter!`](@ref).
 3. Compute the squared distance of each observation with [`squared_distances`](@ref). With a shrinkage, replace the distance of each observation of `idx` with [`leave_one_out!`](@ref).
 4. Flag the outliers at `ce.q` with [`robust_outliers`](@ref).

# Arguments

  - `ce`: The estimator, for its `q`.
  - `smp`: The data of the fit.
  - `idx`: The positions of the observations.
  - `c`: The factor of the scatter.
  - `rho`: The shrinkage intensity, or `nothing`.

# Returns

  - `fit::RobustCovarianceResult`: The fit. The weight of an observation is one when it is in `idx`, and zero otherwise.

# Related

  - [`MCDSample`](@ref)
  - [`mcd_reweight`](@ref)
"""
function mcd_fit(ce::MCDCovariance, smp::MCDSample, idx::AbstractVector{<:Integer},
                 c::Number, rho::Option{<:Number})
    mu, sigma = mcd_fit_rows(smp.X, idx, smp.centre)
    reg = isnothing(rho) ? c : SubsetShrinkage(rho, c, smp.tgt)
    shrink_scatter!(sigma, reg)
    d2 = squared_distances(smp.X, mu, sigma)
    leave_one_out!(d2, idx, reg, !isnothing(smp.centre))
    w = zeros(eltype(sigma), size(smp.X, 1))
    w[idx] .= one(eltype(w))
    return RobustCovarianceResult(mu, sigma, d2, w,
                                  robust_outliers(rho, d2, length(mu), ce.q), rho)
end
"""
    leave_one_out!(d2::VecNum, idx::AbstractVector{<:Integer}, c::Number, fixed::Bool) -> VecNum
    leave_one_out!(d2::VecNum, idx::AbstractVector{<:Integer}, reg::SubsetShrinkage, fixed::Bool) -> VecNum

Replaces the squared distance of each observation of the subset with its distance from the fit that leaves the observation out.

A scatter of few observations follows each observation that it holds. So the distance of such an observation is too small, and the distance of every other observation looks large beside it. With fewer observations than assets, the two sets do not overlap, and half of the clean observations look like outliers. The distance from the fit that leaves the observation out compares with the distance of an observation outside the subset. The method of a number, the fit with no shrinkage, changes nothing.

# Mathematical definition

```math
\\begin{align}
\\tilde{d}_t^2 &= \\frac{g^2 d_t^2}{1 - g b \\, d_t^2}\\,, & g &= \\frac{k}{k - 1}\\,, & b &= \\frac{(1 - \\rho) c}{k - 1}\\,.
\\end{align}
```

Where:

  - ``d_t^2``: The squared distance of an observation of the subset.
  - ``k``: The number of observations of the subset.
  - ``\\rho``: The shrinkage intensity.
  - ``c``: The factor of the scatter.

The formula keeps the divisor of the scatter. With a fixed centre, ``g = 1`` and the divisor is ``k``.

# Arguments

  - `d2`: The squared distance of each observation.
  - `idx`: The positions of the subset.
  - `c`, `reg`: The factor, or the shrinkage.
  - `fixed`: Whether the centre is fixed.

# Returns

  - `d2::VecNum`: The input vector.

# Related

  - [`mcd_fit`](@ref)
  - [`robust_outliers`](@ref)
"""
function leave_one_out!(d2::VecNum, ::AbstractVector{<:Integer}, ::Number, ::Bool)
    return d2
end
function leave_one_out!(d2::VecNum, idx::AbstractVector{<:Integer}, reg::SubsetShrinkage,
                        fixed::Bool)
    k = length(idx)
    g = fixed ? one(eltype(d2)) : oftype(one(eltype(d2)), k / (k - 1))
    b = (1 - reg.rho) * reg.c / (k - !fixed)
    for i in idx
        d2[i] = abs2(g) * d2[i] / max(1 - g * b * d2[i], eps(eltype(d2)))
    end
    return d2
end
"""
    mcd_fit_rows(X::MatNum, idx::AbstractVector{<:Integer}, centre::Nothing) -> Tuple{VecNum, MatNum}
    mcd_fit_rows(X::MatNum, idx::AbstractVector{<:Integer}, centre::VecNum) -> Tuple{VecNum, MatNum}

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
  - [`mcd_fit`](@ref)
"""
function mcd_fit_rows(X::MatNum, idx::AbstractVector{<:Integer}, ::Nothing)
    return mcd_subset_fit(X, idx, false)
end
function mcd_fit_rows(X::MatNum, idx::AbstractVector{<:Integer}, centre::VecNum)
    return centre, last(mcd_subset_fit(X .- transpose(centre), idx, true))
end
"""
    mcd_reweight(rw::Nothing, ce::MCDCovariance, smp::MCDSample, raw::RobustCovarianceResult) -> RobustCovarianceResult
    mcd_reweight(rw::MCDReweighting, ce::MCDCovariance, smp::MCDSample, raw::RobustCovarianceResult) -> RobustCovarianceResult

Applies the reweighting step of an [`MCDCovariance`](@ref) to the raw fit.

The `nothing` method returns the raw fit. The [`MCDReweighting`](@ref) method keeps the observations that [`robust_outliers`](@ref) does not flag at `rw.q`, and fits them with [`mcd_fit`](@ref). When it does not keep every observation, it multiplies the scatter by [`mcd_consistency`](@ref) at `rw.q` and by the reweighted factor of [`mcd_correction`](@ref). With a shrinkage, it finds the intensity of the kept observations with [`mcd_intensity`](@ref).

# Arguments

  - `rw`: The reweighting step, or `nothing`.
  - `ce`: The estimator, for its subset fraction and its small-sample correction.
  - `smp`: The data of the fit.
  - `raw`: The raw fit.

# Returns

  - `fit::RobustCovarianceResult`: The fit.

# Related

  - [`MCDReweighting`](@ref)
  - [`mcd_fit`](@ref)

# References

  - $(ref_dict[:rousseeuw1999]) Section 4, step 7.
"""
function mcd_reweight(::Nothing, ::MCDCovariance, ::MCDSample, raw::RobustCovarianceResult)
    return raw
end
function mcd_reweight(rw::MCDReweighting, ce::MCDCovariance, smp::MCDSample,
                      raw::RobustCovarianceResult)
    n, p = size(smp.X)
    # The factor depends on the shape of the sample alone, so a sample that is too small for it
    # raises whether or not every observation is kept.
    c = mcd_consistency(p, rw.q) * mcd_correction(ce.ssc, p, n, ce.alpha, :rew)
    out = robust_outliers(raw.intensity, raw.d2, p, rw.q)
    c = any(out) ? c : one(c)
    idx = findall(.!out)
    rho = mcd_intensity(smp.intensity, smp.Z, idx, c, !isnothing(smp.centre))
    return mcd_fit(ce, smp, idx, c, rho)
end

export MCDReweighting, MCDCovariance
public AbstractRobustReweighting, MCDSample, mcd_fit, mcd_intensity, mcd_reweight
