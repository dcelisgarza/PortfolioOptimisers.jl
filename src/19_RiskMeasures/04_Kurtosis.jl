"""
$(DocStringExtensions.TYPEDEF)

Represents the square root kurtosis risk measure.

Computes portfolio risk as the square root of the fourth central moment (kurtosis) of the return distribution, optionally using custom weights, expected returns, and a kurtosis (fourth moment) matrix. This risk measure can be evaluated using either the full or semi (downside) deviations, depending on the algorithm provided.

# Mathematical definition

Let ``\\boldsymbol{x} = \\mathbf{X} \\boldsymbol{w}`` be the ``T \\times 1`` vector of portfolio returns, and let ``\\mu`` be the chosen centre (mean, weighted mean, or user-supplied value). Define the centred deviations:

```math
\\begin{align}
\\delta_t &= x_t - \\mu\\,.
\\end{align}
```

The square-root kurtosis (full moment) is:

```math
\\begin{align}
\\mathrm{Kurt}(\\boldsymbol{w}) &= \\sqrt{\\frac{1}{T} \\sum_{t=1}^{T} \\delta_t^4}\\,.
\\end{align}
```

Equivalently, using the ``N^2 \\times N^2`` cokurtosis matrix ``\\hat{\\mathbf{K}}`` and the Kronecker product ``\\otimes``:

```math
\\begin{align}
\\mathrm{Kurt}(\\boldsymbol{w}) &= \\sqrt[4]{(\\boldsymbol{w}^\\intercal \\otimes \\boldsymbol{w}^\\intercal)\\, \\hat{\\mathbf{K}}\\, (\\boldsymbol{w} \\otimes \\boldsymbol{w})}\\,.
\\end{align}
```

For the semi (downside) variant, only non-positive deviations contribute:

```math
\\begin{align}
\\mathrm{SKurt}(\\boldsymbol{w}) &= \\sqrt{\\frac{1}{T} \\sum_{t=1}^{T} \\min(\\delta_t, 0)^4}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: ``N \\times 1`` asset weights vector.
  - ``\\mathbf{X}``: ``T \\times N`` asset returns matrix.
  - ``\\hat{\\mathbf{K}}``: ``N^2 \\times N^2`` cokurtosis matrix.
  - ``\\otimes``: Kronecker product.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Kurtosis(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:MuSlot} = nothing,
        kt::Option{<:KtSlot} = nothing,
        N::Option{<:Integer} = nothing,
        alg1::AbstractMomentAlgorithm = FullMoment(),
        alg2::VarianceFormulation = SOCRiskExpr(),
        pe::Option{<:AbstractPriorEstimator} = nothing,
    ) -> Kurtosis

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is not `nothing`:

      + `::Number`: `isfinite(mu)`.
      + `::VecNum`: `!isempty(mu)` and `all(isfinite, mu)`.

  - If `w` is not `nothing`, `!isempty(w)`.

  - If `kt` is not `nothing`:

      + `!isempty(kt)` and `size(kt, 1) == size(kt, 2)`.

      + If `mu` is not `nothing`:

          * `::VecNum`: `length(mu)^2 == size(kt, 1)`.
          * `::VecScalar`: `length(mu.v)^2 == size(kt, 1)`.

  - If `N` is not `nothing`: must be positive.

!!! warning

    `mu` and `kt` are stated independently, so nothing makes them agree with each other. A caller who wants one consistent set names `pe` alone and lets it fill both from a single fit. A caller who states them by hand must make sure that they agree.

!!! info

    `kt` also admits a [`CokurtosisEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref), and `mu` an [`AbstractExpectedReturnsEstimator`](@ref) or an [`AbstractPriorEstimator`](@ref). Either is resolved against the optimisation's own prior — see [`resolve_deferred_quantities`](@ref). A cokurtosis estimator in `kt` also supplies `mu` from its own `me`, so that the tensor and the centre it was taken about come out of **one** object. A deferred slot wins over `pe`.

# `JuMP` Formulations

## Exact

This formulation is used when `N` is `nothing`.

## Approximate

This formulation is used when `N` is an integer, the larger the value of `N`, the more accurate and expensive it becomes.

# Examples

```jldoctest
julia> Kurtosis()
Kurtosis
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ nothing
        mu ┼ nothing
        kt ┼ nothing
         N ┼ nothing
      alg1 ┼ FullMoment()
      alg2 ┼ SOCRiskExpr()
        pe ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`FullMoment`](@ref)
  - [`SemiMoment`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
@concrete struct Kurtosis <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w_rm])
    """
    w
    """
    $(field_dict[:mu_rm])
    """
    mu
    """
    $(field_dict[:kt])
    """
    kt
    """
    $(field_dict[:N_kt])
    """
    N
    """
    $(field_dict[:alg1])
    """
    alg1
    """
    $(field_dict[:alg2])
    """
    alg2
    """
    $(field_dict[:pe_rm])
    """
    pe
    function Kurtosis(settings::RiskMeasureSettings, w::Option{<:ObsWeights},
                      mu::Option{<:MuSlot}, kt::Option{<:KtSlot}, N::Option{<:Integer},
                      alg1::AbstractMomentAlgorithm, alg2::SecondMomentFormulation,
                      pe::Option{<:AbstractPriorEstimator})
        mu_flag = isa(mu, VecNum)
        kt_flag = isa(kt, MatNum)
        if mu_flag
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu), IsNonFiniteError("mu must be finite, got $mu"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        assert_nonempty_nonneg_finite_val(w, :w)
        if kt_flag
            @argcheck(!isempty(kt), IsEmptyError("kt cannot be empty"))
            assert_matrix_issquare(kt, :kt)
        end
        if mu_flag && kt_flag
            @argcheck(length(mu)^2 == size(kt, 1),
                      DimensionMismatch("length(mu)^2 ($(length(mu)^2)) must match size(kt, 1) ($(size(kt, 1)))"))
        elseif isa(mu, VecScalar) && kt_flag
            @argcheck(length(mu.v)^2 == size(kt, 1),
                      DimensionMismatch("length(mu.v)^2 ($(length(mu.v)^2)) must match size(kt, 1) ($(size(kt, 1)))"))
        end
        if !isnothing(N)
            @argcheck(N > zero(N), DomainError(N, "N must be positive"))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(kt), typeof(N),
                   typeof(alg1), typeof(alg2), typeof(pe)}(settings, w, mu, kt, N, alg1,
                                                           alg2, pe)
    end
end
function Kurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  w::Option{<:ObsWeights} = nothing, mu::Option{<:MuSlot} = nothing,
                  kt::Option{<:KtSlot} = nothing, N::Option{<:Integer} = nothing,
                  alg1::AbstractMomentAlgorithm = FullMoment(),
                  alg2::SecondMomentFormulation = SOCRiskExpr(),
                  pe::Option{<:AbstractPriorEstimator} = nothing)::Kurtosis
    return Kurtosis(settings, w, mu, kt, N, alg1, alg2, pe)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by [`Kurtosis`](@ref) `r` against prior result `pr`.

Three passes, in order.

 1. A deferred `mu` resolves on its own.
 2. A deferred `kt` resolves next, and it carries the centre with it. `kt` is a moment **about** a centre, so the two are one pair of quantities out of one object: when `mu` is still unstated, [`deferred_centre`](@ref) reads it off the cokurtosis estimator's own `me`, threads it into the fit as `mean =`, and it becomes the resolved `mu`. A stated `mu` wins and is threaded in its place. An [`AbstractPriorEstimator`](@ref) centres itself, so the centre is read back off the prior result it produced.
 3. `pe` fans out into whatever both passes left `nothing`.

A deferred slot therefore **wins over `pe`**, which is the map's precedence rule one level down. `kt` lives on a [`HighOrderPrior`](@ref), so a prior estimator named in `kt` or `pe` must compute one.

# Related

  - [`Kurtosis`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_centre`](@ref)
  - [`fan_out_slot`](@ref)
  - [`fit_deferred_quantity`](@ref)
"""
function resolve_deferred_quantities(r::Kurtosis, pr::AbstractPriorResult)::Kurtosis
    if isnothing(r.pe) && !isa(r.mu, DeferredQuantity) && !isa(r.kt, DeferredQuantity)
        return r
    end
    mu = resolve_slot(r.mu, :mu, pr)
    kt = r.kt
    if isa(kt, DeferredQuantity)
        centre = isnothing(mu) ? deferred_centre(kt, pr) : mu
        fitted = fit_deferred_moment(kt, pr, centre)
        kt = deferred_quantity(fitted, :kt)
        mu = nothing_scalar_array_selector(centre, deferred_derived_quantity(fitted, :mu))
    end
    if isnothing(r.pe)
        return Kurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt, N = r.N,
                        alg1 = r.alg1, alg2 = r.alg2, pe = nothing)
    end
    fitted = fit_deferred_quantity(r.pe, pr)
    return Kurtosis(; settings = r.settings, w = r.w, mu = fan_out_slot(fitted, mu, :mu),
                    kt = fan_out_slot(fitted, kt, :kt), N = r.N, alg1 = r.alg1,
                    alg2 = r.alg2, pe = nothing)
end
# Deferrable slots — see `deferred_slots`.
deferred_slots(r::Kurtosis) = (; mu = r.mu, kt = r.kt, pe = r.pe)
"""
    calc_moment_target(::Kurtosis{<:Any, Nothing, Nothing, ...}, ::Any, x::VecNum)
    calc_moment_target(r::Kurtosis{<:Any, <:ObsWeights, Nothing, ...}, ::Any, x::VecNum)
    calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecNum, ...}, w::VecNum, ::Any)
    calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecScalar, ...}, w::VecNum, ::Any)
    calc_moment_target(r::Kurtosis{<:Any, <:Any, <:Number, ...}, ::Any, ::Any)

Compute the target value for kurtosis moment calculations.

Dispatches on the type of `r.w` and `r.mu` to select the appropriate centring target. Follows the same rules as [`calc_moment_target`](@ref).

# Related

  - [`Kurtosis`](@ref)
  - [`calc_moment_target`](@ref)
  - [`calc_deviations_vec`](@ref)
"""
function calc_moment_target(::Kurtosis{<:Any, Nothing, Nothing, <:Any, <:Any, <:Any, <:Any},
                            ::Any, x::VecNum)
    return Statistics.mean(x)
end
function calc_moment_target(r::Kurtosis{<:Any, <:ObsWeights, Nothing, <:Any, <:Any, <:Any,
                                        <:Any}, ::Any, x::VecNum)
    return Statistics.mean(x, r.w)
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecNum, <:Any, <:Any, <:Any, <:Any},
                            w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu)
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:VecScalar, <:Any, <:Any, <:Any,
                                        <:Any}, w::VecNum, ::Any)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s
end
function calc_moment_target(r::Kurtosis{<:Any, <:Any, <:Number, <:Any, <:Any, <:Any, <:Any},
                            ::Any, ::Any)
    return r.mu
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the vector of deviations from the target value for [`Kurtosis`](@ref) risk measures.

See [`calc_deviations_vec`](@ref) for details.

# Related

  - [`Kurtosis`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::Kurtosis, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x)
    return x .- tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the vector of deviations from the target value for a precomputed returns series for [`Kurtosis`](@ref).

Single-argument form used by the precomputed-returns functor `r(x::VecNum)` (ADR 0007).

# Related

  - [`Kurtosis`](@ref)
  - [`calc_deviations_vec`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_deviations_vec(r::Kurtosis, x::VecNum)
    return x .- calc_moment_target(r, nothing, x)
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:FullMoment, <:SOCRiskExpr}, val::VecNum)
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w))
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:SemiMoment, <:SOCRiskExpr}, val::VecNum)
    val = min.(val, zero(eltype(val)))
    val .= val .^ 4
    return sqrt(isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w))
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:FullMoment, <:QuadSecondMomentFormulations},
                     val::VecNum)
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function moment_risk(r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any,
                                 <:Any, <:SemiMoment, <:QuadSecondMomentFormulations},
                     val::VecNum)
    val = min.(val, zero(eltype(val)))
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function (r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any, <:Any,
                      <:Any, <:Any})(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    return moment_risk(r, calc_deviations_vec(r, w, X, fees))
end
function (r::Kurtosis{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any, <:Any, <:Any,
                      <:Any, <:Any})(x::VecNum)
    return moment_risk(r, calc_deviations_vec(r, x))
end
function (r::Kurtosis{<:Any, <:DynamicAbstractWeights, <:Any, <:Any, <:Any, <:SemiMoment,
                      <:Any})(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    return Kurtosis(; settings = r.settings, w = get_observation_weights(r.w, X), mu = r.mu,
                    kt = r.kt, N = r.N, alg1 = r.alg1, alg2 = r.alg2, pe = r.pe)(w, X, fees)
end
function (r::Kurtosis{<:Any, <:DynamicAbstractWeights, <:Any, <:Any, <:Any, <:SemiMoment,
                      <:Any})(x::VecNum)
    return Kurtosis(; settings = r.settings, w = get_observation_weights(r.w, x), mu = r.mu,
                    kt = r.kt, N = r.N, alg1 = r.alg1, alg2 = r.alg2, pe = r.pe)(x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Kurtosis`](@ref) by selecting the cokurtosis matrix, expected returns, and weights from the risk-measure instance or falling back to a [`HighOrderPrior`](@ref) result.

# Related

  - [`Kurtosis`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Kurtosis, pr::HighOrderPrior, args...; kwargs...)::Kurtosis
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    kt = nothing_scalar_array_selector(r.kt, pr.kt)
    return Kurtosis(; settings = r.settings, w = w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2, pe = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`Kurtosis`](@ref) from a [`LowOrderPrior`](@ref) result (cokurtosis matrix is not used).

# Related

  - [`Kurtosis`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`factory`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
"""
function factory(r::Kurtosis, pr::LowOrderPrior, args...; kwargs...)::Kurtosis
    r = resolve_deferred_quantities(r, pr)
    w = nothing_scalar_array_selector(r.w, pr.w)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    kt = nothing_scalar_array_selector(r.kt, nothing)
    return Kurtosis(; settings = r.settings, w = w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2, pe = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`Kurtosis`](@ref) `r` sliced to asset indices `i`.

Slices both the cokurtosis matrix `kt` and the expected returns `mu` for cluster-based optimisation.

# Related

  - [`Kurtosis`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
  - [`fourth_moment_index_generator`](@ref)
"""
function port_opt_view(r::Kurtosis, i, args...)::Kurtosis
    mu = r.mu
    kt = r.kt
    j = nothing
    j = if isa(mu, VecNum)
        length(mu)
    elseif isa(kt, MatNum)
        isqrt(size(kt, 1))
    end
    if !isnothing(j) && !isnothing(kt)
        idx = fourth_moment_index_generator(j, i)
        kt = nothing_scalar_array_view(kt, idx)
    end
    mu = nothing_scalar_array_view(mu, i)
    return Kurtosis(; settings = r.settings, w = r.w, mu = mu, kt = kt, N = r.N,
                    alg1 = r.alg1, alg2 = r.alg2, pe = r.pe)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::Kurtosis) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether [`Kurtosis`](@ref) `r` supports precomputed-return evaluation.

Delegates to [`weight_independent_target`](@ref) on `r.mu`: `true` iff the target is
`Nothing`, a `Number`, or a [`MedianCenteringFunction`](@ref); `false` for per-asset targets.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
  - [`Kurtosis`](@ref)
"""
supports_precomputed_returns(r::Kurtosis) = weight_independent_target(r.mu)

export Kurtosis
