"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the search algorithms of an S estimator of location and scatter.

The S estimate of [`SCovariance`](@ref) is the location and the scatter of determinant one whose distances have the smallest M-scale. No closed form gives it, and the objective has local minima. A search starts from a set of estimates and improves each one by I-steps, see [`s_isteps`](@ref). [`s_search`](@ref) returns the estimate with the smallest scale.

# Interfaces

To implement a new search, subtype `AbstractSAlgorithm`, keep each parameter in a field, and implement one method:

  - `PortfolioOptimisers.s_starts(alg, X::MatNum, centre::Option{<:VecNum}, h::Integer) -> Vector{Tuple{VecNum, MatNum}}`: Returns the start locations and scatters of the data `X`. `centre` is the fixed centre, or `nothing`. `h` is the size of a start subset.

The fit does I-steps from each start until the estimate does not change. A search with many starts can keep a part of them first, with one more method:

  - `PortfolioOptimisers.s_refine(alg, rho::AbstractSRho, X::MatNum, starts, fixed::Bool, k::Number, b::Number) -> Vector{Tuple{VecNum, MatNum}}`: Returns the starts from which the fit does I-steps until the estimate does not change. The default returns `starts`.

# Related

  - [`DetS`](@ref)
  - [`SCovariance`](@ref)
  - [`s_isteps`](@ref)
"""
abstract type AbstractSAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Searches the S estimate from the six deterministic starts of [`DetMCD`](@ref).

The search of [hubert2015](@cite) takes the six start subsets of [hubert2012](@cite), see [`detmcd_initset`](@ref). Each subset of ``h = \\lfloor (n + p + 1)/2 \\rfloor`` observations gives a start: the mean and the covariance of its observations. The search does I-steps from each start until the estimate does not change, and keeps the estimate with the smallest M-scale. Two calls on the same data give the same estimate, so `cov(ce, X) == fit(ce, X).sigma` holds.

The estimate agrees with the R package `rrcov`, version 1.7-7, with `method = "sdet"`, and also with its random search `method = "sfast"`, when both converge to the same minimum.

# Examples

```jldoctest
julia> DetS()
DetS()
```

# Related

  - [`AbstractSAlgorithm`](@ref)
  - [`SCovariance`](@ref)
  - [`DetMCD`](@ref)

# References

  - $(ref_dict[:hubert2015]) Not read. See [`SCovariance`](@ref).
  - $(ref_dict[:hubert2012]) Sections 3.1 and 3.2.
"""
struct DetS <: AbstractSAlgorithm end
"""
    S_MAX_ITER

The largest number of iterations of one fixed point of an S or MM fit, 1000.

The I-steps of [`s_isteps`](@ref), the steps of the M-scale of [`m_scale`](@ref), and the steps of the M-step of [`bisquare_mstep`](@ref) each stop at this bound. A fit of real data converges in a few tens of steps.

# Related

  - [`s_isteps`](@ref)
  - [`m_scale`](@ref)
  - [`bisquare_mstep`](@ref)
"""
const S_MAX_ITER = 1000
"""
$(DocStringExtensions.TYPEDEF)

Estimates the location and the covariance of the returns with an S estimator, which has a high breakdown point.

The S estimator of [davies1987](@cite) takes the location ``\\boldsymbol{\\mu}`` and the shape ``\\mathbf{V}`` of determinant one that make the M-scale ``s`` of the squared distances smallest. The covariance is ``s \\mathbf{V}``. The rho function `rho` is bounded, so a share of outliers below the breakdown point `bdp` cannot move the estimate without bound. Each observation has the continuous weight of the rho function, see [`srho_weight`](@ref), so an outlier has a weight near zero and the estimate needs no hard cut-off. The estimate goes through these steps:

 1. The search `alg` finds the starts, see [`DetS`](@ref).
 2. The I-steps of [`s_isteps`](@ref) improve each start until it does not change.
 3. The fit keeps the estimate with the smallest M-scale.

At `bdp = 0.5`, the efficiency of the S estimate at the normal model is low for few assets. [`MMCovariance`](@ref) takes an M-step from it to a high efficiency, with the same breakdown point.

`StatsAPI.fit` returns the location, the covariance, and the squared distance, the weight and the outlier flag of each observation in a [`RobustCovarianceResult`](@ref). `cov` returns the covariance of that fit.

`me = nothing` gives the joint location of the S estimate. A given `me`, or a `mean` keyword, fixes the centre, and the fit then estimates the shape and the scale alone.

!!! warning

    A fixed centre that is not robust moves with the outliers. The fit then measures the distances from a wrong centre. `SimpleExpectedReturns` is such an estimator. Use `me = nothing`, a [`MedianExpectedReturns`](@ref), or a [`RobustExpectedReturns`](@ref).

The estimator has no field of observation weights. [`factory`](@ref) passes the weights to `me`, and the scatter ignores them.

The estimator has no exact update, so an [`Online`](@ref) wrapper refits it over the rows of its Sample Buffer, see [`SampleBufferState`](@ref).

# Mathematical definition

```math
\\begin{align}
(\\boldsymbol{\\mu}, \\mathbf{V}) &= \\underset{\\boldsymbol{m}, \\, \\lvert \\mathbf{A} \\rvert = 1}{\\arg\\min} \\; s(\\boldsymbol{m}, \\mathbf{A})\\,, & \\frac{1}{T} \\sum_{t=1}^{T} \\rho\\left(\\frac{(\\boldsymbol{x}_t - \\boldsymbol{m})^\\intercal \\mathbf{A}^{-1} (\\boldsymbol{x}_t - \\boldsymbol{m})}{s(\\boldsymbol{m}, \\mathbf{A})}\\right) &= b\\,.
\\end{align}
```

Where:

  - ``s``: The M-scale of the squared distances, see [`m_scale`](@ref).
  - ``\\rho``: The rho function `rho`.
  - ``b``: The right side of the equation of the scale, see [`s_tuning`](@ref).
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SCovariance(;
        me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
        rho::AbstractSRho = BisquareRho(),
        bdp::Number = 0.5,
        alg::AbstractSAlgorithm = DetS(),
        q::Number = 0.975,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> SCovariance

Keywords correspond to the struct's fields.

## Validation

  - `0 < bdp <= 1/2`.
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
julia> SCovariance()
SCovariance
   me ┼ nothing
  rho ┼ BisquareRho()
  bdp ┼ Float64: 0.5
  alg ┼ DetS()
    q ┴ Float64: 0.975
```

# Related

  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`MMCovariance`](@ref)
  - [`BisquareRho`](@ref)
  - [`DetS`](@ref)
  - [`MCDCovariance`](@ref)

# References

  - $(ref_dict[:davies1987]) Not read. The estimate agrees with the R package `rrcov`, version 1.7-7.
  - $(ref_dict[:lopuhaa1989]) Not read. See [`s_isteps`](@ref).
  - $(ref_dict[:hubert2015]) Not read. See [`DetS`](@ref).
"""
@propagatable @concrete struct SCovariance <: AbstractRobustCovarianceEstimator
    """
    $(field_dict[:rob_me])
    """
    @fprop @vprop me
    """
    $(field_dict[:s_rho])
    """
    rho
    """
    $(field_dict[:s_bdp])
    """
    bdp
    """
    $(field_dict[:s_alg])
    """
    alg
    """
    $(field_dict[:rob_q])
    """
    q
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function SCovariance(me::Option{<:AbstractExpectedReturnsEstimator}, rho::AbstractSRho,
                         bdp::Number, alg::AbstractSAlgorithm, q::Number,
                         cache::Option{<:AbstractPartialFitState})
        @argcheck(zero(bdp) < bdp <= one(bdp) / 2,
                  DomainError(bdp, "`bdp` must be in (0, 1/2], got $bdp"))
        assert_open_probability(q, :q)
        return new{typeof(me), typeof(rho), typeof(bdp), typeof(alg), typeof(q),
                   typeof(cache)}(me, rho, bdp, alg, q, cache)
    end
end
function SCovariance(; me::Option{<:AbstractExpectedReturnsEstimator} = nothing,
                     rho::AbstractSRho = BisquareRho(), bdp::Number = 0.5,
                     alg::AbstractSAlgorithm = DetS(), q::Number = 0.975,
                     cache::Option{<:AbstractPartialFitState} = nothing)::SCovariance
    return SCovariance(me, rho, bdp, alg, q, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`SCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit, not a part of the configuration. Set `set_show_nothing_fields!(:SCovariance, true)` to render it.

# Arguments

  - `::SCovariance`: The estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:me, :rho, :bdp, :alg, :q)`.

# Related

  - [`SCovariance`](@ref)
  - [`show_fields`](@ref)
"""
show_fields(::SCovariance) = (:me, :rho, :bdp, :alg, :q)
"""
    StatsAPI.fit(ce::SCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
        -> RobustCovarianceResult

Fits the S estimate of [`SCovariance`](@ref) to the data.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), and refuse a gapped sample with [`assert_finite_sample`](@ref).
 2. Refuse a sample whose start subsets of ``h = \\lfloor (n + p + 1)/2 \\rfloor`` observations are too small, with [`assert_robust_rows`](@ref).
 3. Take the fixed centre with [`robust_centre`](@ref): the `mean` keyword, the mean of `ce.me`, or `nothing` for a joint location.
 4. Take the tuning constant and the right side of the M-scale with [`s_tuning`](@ref).
 5. Find the location, the shape and the scale with [`s_search`](@ref).
 6. Build the fit with [`srho_fit`](@ref).

# Arguments

  - `ce`: The S covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean]) A given centre fixes the location of the fit.
  - `kwargs...`: More keyword arguments for `ce.me`.

# Validation

  - $(val_dict[:dims])
  - `all(isfinite, X)`.
  - `h > p`. Otherwise, the method raises a `DomainError` that names `h`, `p` and the limit.
  - No column of `X` is constant, see [`mcd_standardise`](@ref).
  - The scatter of each step is positive definite. Otherwise, the method raises the `DomainError` of [`squared_distances`](@ref) or of [`unit_determinant!`](@ref).

# Returns

  - `res::RobustCovarianceResult`: The fit. The weight of an observation is its weight of the rho function, see [`srho_weight`](@ref).

# Related

  - [`SCovariance`](@ref)
  - [`RobustCovarianceResult`](@ref)
  - [`s_search`](@ref)
"""
function StatsAPI.fit(ce::SCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    n, p = size(X)
    h = div(n + p + 1, 2)
    assert_robust_rows(h, p, "the size h of a start subset")
    centre = robust_centre(ce.me, X; mean = mean, kwargs...)
    k, b = s_tuning(ce.rho, ce.bdp, n, p)
    mu, V, s = s_search(ce.alg, ce.rho, X, centre, h, k, b, S_MAX_ITER)
    return srho_fit(ce.rho, X, mu, s * V, k, ce.q)
end
"""
    srho_fit(rho::AbstractSRho, X::MatNum, mu::VecNum, sigma::MatNum, k::Number, q::Number)
        -> RobustCovarianceResult

Builds the fit of an S or MM estimate from its location and its covariance.

The squared distance of an observation is its distance in the metric of `sigma`, from [`squared_distances`](@ref). Its weight is [`srho_weight`](@ref) at that distance, and its flag is the one of [`robust_outliers`](@ref) at `q`.

# Arguments

  - `rho`: The rho function of the weights.
  - `X`: The data matrix, `observations × assets`.
  - `mu`: The location.
  - `sigma`: The covariance, the scale times the shape.
  - `k`: The tuning constant of the rho function.
  - `q`: The probability of the cut-off of the outliers.

# Returns

  - `res::RobustCovarianceResult`: The fit.

# Related

  - [`SCovariance`](@ref)
  - [`MMCovariance`](@ref)
"""
function srho_fit(rho::AbstractSRho, X::MatNum, mu::VecNum, sigma::MatNum, k::Number,
                  q::Number)
    d2 = squared_distances(X, mu, sigma)
    return RobustCovarianceResult(mu, sigma, d2, srho_weight.(Ref(rho), d2, k),
                                  robust_outliers(nothing, d2, length(mu), q), nothing)
end
"""
    s_search(alg::AbstractSAlgorithm, rho::AbstractSRho, X::MatNum, centre::Option{<:VecNum},
             h::Integer, k::Number, b::Number, maxsteps::Integer) -> Tuple{VecNum, MatNum, Number}

Returns the location, the shape of determinant one and the M-scale of the best estimate that the search `alg` finds.

# Algorithm

 1. Take the starts with [`s_starts`](@ref), and keep the best of a search with many starts with [`s_refine`](@ref).
 2. Do at most `maxsteps` I-steps from each start with [`s_isteps`](@ref), until the estimate does not change.
 3. Return the estimate with the smallest M-scale. The first start wins a tie. Warn when that estimate did not converge.

# Arguments

  - `alg`: The search algorithm.
  - `rho`: The rho function.
  - `X`: The data matrix, `observations × assets`.
  - `centre`: The fixed centre, or `nothing`.
  - `h`: The size of a start subset.
  - `k`: The tuning constant of the rho function.
  - `b`: The right side of the equation of the M-scale.
  - `maxsteps`: The largest number of I-steps from one start, [`S_MAX_ITER`](@ref) in a fit.

# Returns

  - `(mu, V, s)`: The location, the shape and the scale.

# Related

  - [`AbstractSAlgorithm`](@ref)
  - [`s_isteps`](@ref)
"""
function s_search(alg::AbstractSAlgorithm, rho::AbstractSRho, X::MatNum,
                  centre::Option{<:VecNum}, h::Integer, k::Number, b::Number,
                  maxsteps::Integer)
    fixed = !isnothing(centre)
    starts = s_refine(alg, rho, X, s_starts(alg, X, centre, h), fixed, k, b)
    fits = [s_isteps(rho, X, mu, S, fixed, k, b, maxsteps) for (mu, S) in starts]
    mu, V, s, done = fits[argmin(map(f -> f[3], fits))]
    if !done
        @warn("The I-steps of the S estimate did not converge in $(maxsteps) steps. The last iterate is returned.",
              maxlog = log_limit(:warn), _id = @log_id())
    end
    return mu, V, s
end
"""
    s_starts(alg::DetS, X::MatNum, centre::Option{<:VecNum}, h::Integer) -> Vector{Tuple{VecNum, MatNum}}

Returns the start locations and scatters of the search `alg`.

The [`DetS`](@ref) method standardises the data with [`mcd_standardise`](@ref), takes the six start subsets of [`detmcd_initset`](@ref) over the directions of [`detmcd_directions`](@ref), and fits each subset in the units of the data with [`mcd_fit_rows`](@ref). With a fixed centre, the location of each start is the centre.

# Arguments

  - `alg`: The search algorithm.
  - `X`: The data matrix, `observations × assets`.
  - `centre`: The fixed centre, or `nothing`.
  - `h`: The size of a start subset.

# Returns

  - `starts::Vector{Tuple{VecNum, MatNum}}`: The start locations and scatters.

# Related

  - [`s_search`](@ref)
  - [`detmcd_initset`](@ref)

# References

  - $(ref_dict[:hubert2015]) Not read. See [`DetS`](@ref).
"""
function s_starts(::DetS, X::MatNum, centre::Option{<:VecNum}, h::Integer)
    Z = first(mcd_standardise(X, centre))
    fixed = !isnothing(centre)
    return [mcd_fit_rows(X, detmcd_initset(Z, P, h, fixed), centre)
            for P in detmcd_directions(Z)]
end
"""
    s_refine(alg::AbstractSAlgorithm, rho::AbstractSRho, X::MatNum, starts, fixed::Bool,
             k::Number, b::Number) -> Vector{Tuple{VecNum, MatNum}}

Keeps the best starts of a search. The default method keeps every start.

# Arguments

  - `alg`: The search algorithm.
  - `rho`: The rho function.
  - `X`: The data matrix, `observations × assets`.
  - `starts`: The start locations and scatters.
  - `fixed`: Whether the centre is fixed.
  - `k`: The tuning constant of the rho function.
  - `b`: The right side of the equation of the M-scale.

# Returns

  - `starts::Vector{Tuple{VecNum, MatNum}}`: The starts from which the fit does I-steps until the estimate does not change.

# Related

  - [`s_starts`](@ref)
  - [`s_search`](@ref)
"""
function s_refine(::AbstractSAlgorithm, ::AbstractSRho, ::MatNum, starts, ::Bool, ::Number,
                  ::Number)
    return starts
end
"""
    s_isteps(rho::AbstractSRho, X::MatNum, mu::VecNum, S::MatNum, fixed::Bool, k::Number,
             b::Number, maxsteps::Integer) -> Tuple{VecNum, MatNum, Number, Bool}

Does I-steps of an S estimate from the location `mu` and the scatter `S`.

An I-step is a step of the fixed-point equations of [lopuhaa1989](@cite), which state that an S estimate is an M-estimate with the weights of its own rho function. The step does not increase the M-scale. It stops when the relative change of the shape and the change of the location in the metric of the shape are both below `sqrt(eps)`, or after `maxsteps` steps.

# Algorithm

 1. Scale `S` to the determinant one with [`unit_determinant!`](@ref). Compute the squared distances `t` with [`squared_distances`](@ref), and their M-scale `s` with [`m_scale`](@ref), from the start ``\\mathrm{med}(t)``.
 2. Repeat at most `maxsteps` times:
     1. Take the weight of each observation, [`srho_weight`](@ref) at `t / s`.
     2. Take the weighted mean and the weighted scatter of determinant one with [`weighted_shape`](@ref).
     3. Compute the new distances, and their M-scale from the start `s`.

# Arguments

  - `rho`: The rho function.
  - `X`: The data matrix, `observations × assets`.
  - `mu`: The start location.
  - `S`: The start scatter. The method does not change it.
  - `fixed`: Whether the centre is fixed at `mu`.
  - `k`: The tuning constant of the rho function.
  - `b`: The right side of the equation of the M-scale.
  - `maxsteps`: The largest number of steps.

# Returns

  - `(mu, V, s, done)`: The location, the shape of determinant one, the M-scale, and whether the steps converged.

# Related

  - [`s_search`](@ref)
  - [`weighted_shape`](@ref)
  - [`m_scale`](@ref)

# References

  - $(ref_dict[:lopuhaa1989]) Not read. The fixed point agrees with the R package `rrcov`, version 1.7-7.
"""
function s_isteps(rho::AbstractSRho, X::MatNum, mu::VecNum, S::MatNum, fixed::Bool,
                  k::Number, b::Number, maxsteps::Integer)
    V = unit_determinant!(copy(S))
    t = squared_distances(X, mu, V)
    s = m_scale(rho, t, k, b, Statistics.median(t))
    tol = sqrt(eps(real(eltype(V))))
    for _ in 1:maxsteps
        mun, Vn = weighted_shape(X, srho_weight.(Ref(rho), t ./ s, k), mu, fixed)
        done = shape_converged(mu, V, mun, Vn, tol)
        mu, V = mun, Vn
        t = squared_distances(X, mu, V)
        s = m_scale(rho, t, k, b, s)
        if done
            return mu, V, s, true
        end
    end
    return mu, V, s, false
end
"""
    shape_converged(mu::VecNum, V::MatNum, mun::VecNum, Vn::MatNum, tol::Number) -> Bool

Returns whether a step of a fixed point moved the location and the shape by less than `tol`.

The relative change of the shape is in the Frobenius norm, and the change of the location is its distance in the metric of the old shape.

# Arguments

  - `mu`, `V`: The old location and shape.
  - `mun`, `Vn`: The new location and shape.
  - `tol`: The tolerance.

# Returns

  - `done::Bool`: Whether both changes are below `tol`.

# Related

  - [`s_isteps`](@ref)
  - [`bisquare_mstep`](@ref)
"""
function shape_converged(mu::VecNum, V::MatNum, mun::VecNum, Vn::MatNum, tol::Number)
    return LinearAlgebra.norm(Vn - V) <= tol * LinearAlgebra.norm(V) &&
           only(squared_distances(transpose(mun), mu, V)) <= abs2(tol)
end
"""
    weighted_shape(X::MatNum, w::VecNum, mu::VecNum, fixed::Bool) -> Tuple{VecNum, MatNum}

Returns the weighted mean of the observations and their weighted scatter about it, scaled to the determinant one.

With a fixed centre, the location stays at `mu`, and the scatter is about `mu`. A common factor of the weights does not change the result.

# Arguments

  - `X`: The data matrix, `observations × assets`.
  - `w`: The weight of each observation.
  - `mu`: The location, used when `fixed` is `true`.
  - `fixed`: Whether the centre is fixed at `mu`.

# Returns

  - `(mu, V)`: The location and the shape.

# Related

  - [`s_isteps`](@ref)
  - [`unit_determinant!`](@ref)
"""
function weighted_shape(X::MatNum, w::VecNum, mu::VecNum, fixed::Bool)
    mun = fixed ? mu : vec(transpose(X) * w) / sum(w)
    D = X .- transpose(mun)
    A = sqrt.(w) .* D
    return mun, unit_determinant!(transpose(A) * A)
end
"""
    unit_determinant!(S::MatNum) -> MatNum

Scales a scatter matrix in place to the determinant one.

# Arguments

  - `S`: The scatter matrix.

# Validation

  - `S` is positive definite, as [`scatter_cholesky`](@ref) reads it. Otherwise, the function raises a `DomainError` that states that the weighted observations lie on a hyperplane.

# Returns

  - `S::MatNum`: The input matrix, scaled.

# Related

  - [`weighted_shape`](@ref)
"""
function unit_determinant!(S::MatNum)
    C = scatter_cholesky(S)
    @argcheck(!isnothing(C),
              DomainError(size(S),
                          "the weighted scatter matrix is singular, so the observations of positive weight lie on a hyperplane. A robust covariance needs data in general position"))
    S ./= exp(LinearAlgebra.logdet(C) / size(S, 1))
    return S
end

public AbstractSAlgorithm, s_starts, s_refine
export DetS, SCovariance
