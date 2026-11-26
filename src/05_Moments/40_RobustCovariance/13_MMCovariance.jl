"""
$(DocStringExtensions.TYPEDEF)

Fits the location and the shape again with the soft weights of the bisquare function, with the scale of the raw fit fixed.

The M-step of [tatsuoka2000](@cite) and [salibianbarrera2006](@cite) keeps the scale ``s = \\lvert \\mathbf{\\Sigma}_0 \\rvert^{1/p}`` of the raw covariance ``\\mathbf{\\Sigma}_0``, and solves the fixed point of a bisquare M-estimator of the location and of the shape of determinant one from the raw fit. Its constant ``c^2`` is larger than the one of the raw fit, so the step gives a high efficiency at the normal model, see [`bisquare_efficiency`](@ref). The raw fit gives the breakdown point, and the step keeps it. The weight of an observation falls from one at the centre to zero at ``c^2``, see [`BisquareRho`](@ref).

The step is the second part of [`MMCovariance`](@ref). It also follows an [`MCDCovariance`](@ref) with `rw = BisquareReweighting()`, a soft reweighting in place of the hard weights of [`MCDReweighting`](@ref). The scale of the MCD is consistent at the normal model, so the efficiency holds there too.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BisquareReweighting(;
        eff::Number = 0.95
    ) -> BisquareReweighting

Keywords correspond to the struct's fields.

## Validation

  - `0 < eff < 1`.

# Examples

```jldoctest
julia> BisquareReweighting()
BisquareReweighting
  eff ┴ Float64: 0.95
```

# Related

  - [`AbstractRobustReweighting`](@ref)
  - [`MMCovariance`](@ref)
  - [`MCDCovariance`](@ref)
  - [`BisquareRho`](@ref)

# References

  - $(ref_dict[:tatsuoka2000]) Not read. See [`bisquare_mstep`](@ref).
  - $(ref_dict[:salibianbarrera2006]) Not read. See [`bisquare_efficiency`](@ref).
"""
@concrete struct BisquareReweighting <: AbstractRobustReweighting
    """
    $(field_dict[:mm_eff])
    """
    eff
    function BisquareReweighting(eff::Number)
        assert_open_probability(eff, :eff)
        return new{typeof(eff)}(eff)
    end
end
function BisquareReweighting(; eff::Number = 0.95)::BisquareReweighting
    return BisquareReweighting(eff)
end
"""
    mm_constant(rw::BisquareReweighting, p::Integer) -> Number

Returns the tuning constant ``c^2`` of the bisquare function at which the shape of the M-step has the efficiency `rw.eff` at the normal model.

The efficiency of [`bisquare_efficiency`](@ref) rises with ``c^2``, so a bisection finds the constant. The interval starts at ``[p/2, p]``. Its lower bound halves and its upper bound doubles until it holds the constant.

# Arguments

  - `rw`: The reweighting step.
  - `p`: The number of assets.

# Returns

  - `c2::Number`: The tuning constant.

# Related

  - [`BisquareReweighting`](@ref)
  - [`bisquare_efficiency`](@ref)
"""
function mm_constant(rw::BisquareReweighting, p::Integer)
    excess(c2) = bisquare_efficiency(p, c2) - rw.eff
    hi = p * one(rw.eff)
    lo = hi / 2
    while excess(lo) > 0
        lo /= 2
    end
    while excess(hi) < 0
        hi *= 2
    end
    return Roots.find_zero(excess, (lo, hi), Roots.Bisection())
end
"""
    bisquare_mstep(rw::BisquareReweighting, X::MatNum, mu::VecNum, sigma::MatNum,
                   fixed::Bool, q::Number, maxsteps::Integer) -> RobustCovarianceResult

Does the M-step of [`BisquareReweighting`](@ref) from the location `mu` and the covariance `sigma` of a raw fit.

# Algorithm

 1. Take the constant ``c^2`` of [`mm_constant`](@ref), the shape ``\\mathbf{\\Sigma}_0 / s`` of determinant one with [`unit_determinant!`](@ref), and the scale ``s = \\lvert \\mathbf{\\Sigma}_0 \\rvert^{1/p}`` as the ratio of a diagonal entry of the covariance and of the shape.
 2. Repeat at most `maxsteps` times:
     1. Take the weight of each observation, [`srho_weight`](@ref) of [`BisquareRho`](@ref) at its squared distance in the metric of the shape, divided by `s`.
     2. Take the weighted mean and the weighted scatter of determinant one with [`weighted_shape`](@ref).
     3. Stop when the relative change of the shape and the change of the location in the metric of the shape are both below `sqrt(eps)`, see [`shape_converged`](@ref).
 3. Warn when the steps do not converge, and keep the last iterate.
 4. Build the fit of the covariance `s` times the shape with [`srho_fit`](@ref), with the weights at ``c^2``.

# Arguments

  - `rw`: The reweighting step.
  - `X`: The data matrix, `observations × assets`.
  - `mu`: The location of the raw fit.
  - `sigma`: The covariance of the raw fit.
  - `fixed`: Whether the centre is fixed at `mu`.
  - `q`: The probability of the cut-off of the outliers.
  - `maxsteps`: The largest number of steps, [`S_MAX_ITER`](@ref) in a fit.

# Validation

  - `sigma` is positive definite, and so is the weighted scatter of each step. Otherwise, the method raises the `DomainError` of [`unit_determinant!`](@ref).

# Returns

  - `res::RobustCovarianceResult`: The fit.

# Related

  - [`BisquareReweighting`](@ref)
  - [`MMCovariance`](@ref)
  - [`weighted_shape`](@ref)

# References

  - $(ref_dict[:tatsuoka2000]) Not read. The fixed point agrees with the R package `rrcov`, version 1.7-7.
"""
function bisquare_mstep(rw::BisquareReweighting, X::MatNum, mu::VecNum, sigma::MatNum,
                        fixed::Bool, q::Number, maxsteps::Integer)
    p = size(X, 2)
    c2 = mm_constant(rw, p)
    V = unit_determinant!(copy(sigma))
    s = sigma[1, 1] / V[1, 1]
    tol = sqrt(eps(real(eltype(V))))
    rho = BisquareRho()
    done = false
    for _ in 1:maxsteps
        w = srho_weight.(Ref(rho), squared_distances(X, mu, V) ./ s, c2)
        mun, Vn = weighted_shape(X, w, mu, fixed)
        done = shape_converged(mu, V, mun, Vn, tol)
        mu, V = mun, Vn
        if done
            break
        end
    end
    if !done
        @warn("The M-step of the bisquare reweighting did not converge in $(maxsteps) steps. The last iterate is returned.",
              maxlog = log_limit(:warn), _id = @log_id())
    end
    return srho_fit(rho, X, mu, s * V, c2, q)
end
"""
    mcd_reweight(rw::BisquareReweighting, ce::MCDCovariance, smp::MCDSample,
                 raw::RobustCovarianceResult) -> RobustCovarianceResult

Applies the M-step of [`BisquareReweighting`](@ref) to the raw fit of an [`MCDCovariance`](@ref), with [`bisquare_mstep`](@ref).

The scale of the step is the one of the raw covariance, which has the consistency factor and the small-sample factor of the MCD. The fit flags the outliers at `ce.q`.

# Arguments

  - `rw`: The reweighting step.
  - `ce`: The estimator, for its `q`.
  - `smp`: The data of the fit.
  - `raw`: The raw fit.

# Returns

  - `fit::RobustCovarianceResult`: The fit.

# Related

  - [`BisquareReweighting`](@ref)
  - [`MCDCovariance`](@ref)
"""
function mcd_reweight(rw::BisquareReweighting, ce::MCDCovariance, smp::MCDSample,
                      raw::RobustCovarianceResult)
    return bisquare_mstep(rw, smp.X, raw.mu, raw.sigma, !isnothing(smp.centre), ce.q,
                          S_MAX_ITER)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the location and the covariance of the returns with an MM estimator, which has a high breakdown point and a high efficiency at the normal model.

The MM estimator of [tatsuoka2000](@cite) and [salibianbarrera2006](@cite) has two parts:

 1. The S estimate `s`, see [`SCovariance`](@ref). It gives the breakdown point and the scale.
 2. The M-step `rw` from the S estimate, with the scale fixed, see [`BisquareReweighting`](@ref). It gives the efficiency.

The default gives the breakdown point one half and the efficiency 95 % of the shape. The estimate agrees with `CovMMest` of the R package `rrcov`, version 1.7-7, with the start `method = "sdet"`.

`StatsAPI.fit` returns the location, the covariance, and the squared distance, the weight and the outlier flag of each observation in a [`RobustCovarianceResult`](@ref). The weight of an observation is its bisquare weight in the M-step. `cov` returns the covariance of that fit.

The `me` of `s` fixes the centre of both parts, and a `mean` keyword does the same. The `q` of `s` and its `cache` are not read: the fit flags the outliers at the `q` of this estimator, and an [`Online`](@ref) wrapper reads the `cache` of this estimator.

The estimator has no exact update, so an [`Online`](@ref) wrapper refits it over the rows of its Sample Buffer, see [`SampleBufferState`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MMCovariance(;
        s::SCovariance = SCovariance(),
        rw::BisquareReweighting = BisquareReweighting(),
        q::Number = 0.975,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> MMCovariance

Keywords correspond to the struct's fields.

## Validation

  - `0 < q < 1`.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `s`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `s`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> MMCovariance()
MMCovariance
   s ┼ SCovariance
     │    me ┼ nothing
     │   rho ┼ BisquareRho()
     │   bdp ┼ Float64: 0.5
     │   alg ┼ DetS()
     │     q ┴ Float64: 0.975
  rw ┼ BisquareReweighting
     │   eff ┴ Float64: 0.95
   q ┴ Float64: 0.975
```

# Related

  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`SCovariance`](@ref)
  - [`BisquareReweighting`](@ref)
  - [`MCDCovariance`](@ref)

# References

  - $(ref_dict[:tatsuoka2000]) Not read. See [`bisquare_mstep`](@ref).
  - $(ref_dict[:salibianbarrera2006]) Not read. See [`bisquare_efficiency`](@ref).
  - $(ref_dict[:maronna2019]) Not read.
"""
@propagatable @concrete struct MMCovariance <: AbstractRobustCovarianceEstimator
    """
    $(field_dict[:mm_s])
    """
    @fprop @vprop s
    """
    $(field_dict[:mm_rw])
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
    function MMCovariance(s::SCovariance, rw::BisquareReweighting, q::Number,
                          cache::Option{<:AbstractPartialFitState})
        assert_open_probability(q, :q)
        return new{typeof(s), typeof(rw), typeof(q), typeof(cache)}(s, rw, q, cache)
    end
end
function MMCovariance(; s::SCovariance = SCovariance(),
                      rw::BisquareReweighting = BisquareReweighting(), q::Number = 0.975,
                      cache::Option{<:AbstractPartialFitState} = nothing)::MMCovariance
    return MMCovariance(s, rw, q, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`MMCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Set `set_show_nothing_fields!(:MMCovariance, true)` to render it.

# Arguments

  - `::MMCovariance`: The estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:s, :rw, :q)`.

# Related

  - [`MMCovariance`](@ref)
  - [`show_fields`](@ref)
"""
show_fields(::MMCovariance) = (:s, :rw, :q)
"""
    StatsAPI.fit(ce::MMCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
        -> RobustCovarianceResult

Fits the MM estimate of [`MMCovariance`](@ref) to the data.

# Algorithm

 1. Fit the S estimate `ce.s` with its `StatsAPI.fit`, with `dims`, `mean` and `kwargs`.
 2. Orient `X` with [`dims_oriented`](@ref).
 3. Do the M-step of `ce.rw` from the S estimate with [`bisquare_mstep`](@ref). The centre is fixed when the `mean` keyword or the `me` of `ce.s` is not `nothing`.

# Arguments

  - `ce`: The MM covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) A given centre fixes the location of the fit.
  - `kwargs...`: More keyword arguments for the `me` of `ce.s`.

# Validation

  - The validation of `StatsAPI.fit` of [`SCovariance`](@ref).
  - The weighted scatter of each M-step is positive definite. Otherwise, the method raises the `DomainError` of [`unit_determinant!`](@ref).

# Returns

  - `res::RobustCovarianceResult`: The fit. The weight of an observation is its bisquare weight in the M-step.

# Related

  - [`MMCovariance`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`bisquare_mstep`](@ref)
"""
function StatsAPI.fit(ce::MMCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
    raw = StatsAPI.fit(ce.s, X; dims = dims, mean = mean, kwargs...)
    fixed = !(isnothing(mean) && isnothing(ce.s.me))
    return bisquare_mstep(ce.rw, dims_oriented(dims, X), raw.mu, raw.sigma, fixed, ce.q,
                          S_MAX_ITER)
end

export BisquareReweighting, MMCovariance
