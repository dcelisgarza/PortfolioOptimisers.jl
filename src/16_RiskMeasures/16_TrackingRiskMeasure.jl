# https://portfoliooptimizationbook.com/slides/slides-index-tracking.pdf
"""
$(DocStringExtensions.TYPEDEF)

Bounds how far the risk of a portfolio can move from the risk of a benchmark portfolio.

[`RiskTrackingRiskMeasure`](@ref) measures the same quantity in the same two modes, and this type bounds it. Put it in the `tr` slot of a `JuMPOptimiser`. The distance is a difference of risks, not a norm of the difference of the return series, and that is how it differs from [`TrackingError`](@ref).

In an optimisation the independent bound is exact, and the dependent bound holds from above only. The model states the portfolio risk through an upper bound that the solver can raise. So a portfolio whose risk is below the benchmark risk by more than `err` also satisfies the dependent bound. This is true of every measure whose model is an upper bound, among them [`ConditionalValueatRisk`](@ref), [`StandardDeviation`](@ref) and a [`Variance`](@ref) in the semidefinite form. A [`Variance`](@ref) outside the semidefinite form does not solve in the dependent mode.

The model charges the fee of the portfolio on the returns that the tracked measure reads, in both modes, and it never reads `tr.fees`. In the independent mode, the functor of [`RiskTrackingRiskMeasure`](@ref) charges the fee of the weight difference instead. So with a fee, a returns-based measure reads back a value that is different from the value that the model bounds.

!!! warning

    The default `err = 0.0` bounds the tracked quantity at zero. In the independent mode with a positive-definite measure, the bound pins the portfolio weights to the benchmark weights. Give an `err` unless you want that result.

# Mathematical definition

The `alg` field selects which of two quantities `err` bounds. The independent mode, [`IndependentVariableTracking`](@ref), bounds the risk of the weight difference. The dependent mode, [`DependentVariableTracking`](@ref), bounds the absolute difference of the two risks.

```math
\\begin{align}
\\rho(\\boldsymbol{w} - \\boldsymbol{w}_b) &\\leq \\varepsilon\\,, \\\\
\\lvert \\rho(\\boldsymbol{w}) - \\rho(\\boldsymbol{w}_b) \\rvert &\\leq \\varepsilon\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:w_b_track])
  - $(math_dict[:rho_track])
  - ``\\varepsilon``: Tolerance, the `err` field.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskTrackingError(;
        tr::WeightsTracking,
        r::AbstractBaseRiskMeasure = StandardDeviation(),
        err::Number = 0.0,
        alg::VariableTracking = IndependentVariableTracking()
    ) -> RiskTrackingError

Keywords correspond to the struct's fields.

The constructor passes `r` through [`no_bounds_no_risk_expr_risk_measure`](@ref), which drops the `settings.ub` and the `settings.rke` of the tracked measure. The tracked measure only measures the distance. It does not bound the portfolio and it adds no term to the objective, so `err` is the only bound of the constraint.

## Validation

  - `err`, through [`assert_nonempty_nonneg_finite_val`](@ref): `err` is finite and not negative.

## View parameters

`RiskTrackingError` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `tr` recurses through [`port_opt_view`](@ref) with the asset indices alone, which slices the benchmark weights.
  - `r` recurses through [`port_opt_view`](@ref) with the asset indices and the returns matrix, because a measure can slice a moment that it holds.
  - `err` and `alg` pass through unchanged.

# Examples

```jldoctest
julia> RiskTrackingError(; tr = WeightsTracking(; w = [0.5, 0.5]), err = 0.05)
RiskTrackingError
   tr ┼ WeightsTracking
      │    fees ┼ nothing
      │       w ┼ Vector{Float64}: [0.5, 0.5]
      │   fixed ┴ Bool: false
    r ┼ StandardDeviation
      │   settings ┼ RiskMeasureSettings
      │            │   scale ┼ Int64: 1
      │            │      ub ┼ nothing
      │            │     rke ┴ Bool: false
      │      sigma ┼ nothing
      │       chol ┴ nothing
  err ┼ Float64: 0.05
  alg ┴ IndependentVariableTracking()
```

# Related

  - [`set_tracking_error_constraints!`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)

# References

  - $(ref_dict[:palomar2025])
  - $(ref_dict[:cajas2025]) Section 9.2.
"""
@concrete struct RiskTrackingError <: AbstractTracking
    """
    $(field_dict[:tr_spec])
    """
    tr
    """
    $(field_dict[:r_risk])
    """
    r
    """
    $(field_dict[:err])
    """
    err
    """
    $(field_dict[:tralg])
    """
    alg
    function RiskTrackingError(tr::WeightsTracking, r::AbstractBaseRiskMeasure, err::Number,
                               alg::VariableTracking)
        assert_nonempty_nonneg_finite_val(err, :err)
        r = no_bounds_no_risk_expr_risk_measure(r)
        return new{typeof(tr), typeof(r), typeof(err), typeof(alg)}(tr, r, err, alg)
    end
end
function RiskTrackingError(; tr::WeightsTracking,
                           r::AbstractBaseRiskMeasure = StandardDeviation(),
                           err::Number = 0.0,
                           alg::VariableTracking = IndependentVariableTracking())::RiskTrackingError
    return RiskTrackingError(tr, r, err, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`RiskTrackingError`](@ref) restricted to the assets at the indices `i`.

A hierarchical or clustering optimiser calls it to build the constraint of one cluster. The benchmark weights of the result are the entries `i` of `tr.tr.w`, so they no longer sum to the budget of the whole benchmark.

# Algorithm

 1. Slice the benchmark with `port_opt_view(tr.tr, i)`.
 2. Slice the tracked measure with `port_opt_view(tr.r, i, X)`.
 3. Build a new [`RiskTrackingError`](@ref) from the two slices, `tr.err` and `tr.alg`.

# Related

  - [`RiskTrackingError`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(tr::RiskTrackingError, i, X::MatNum, args...)
    return RiskTrackingError(; tr = port_opt_view(tr.tr, i), r = port_opt_view(tr.r, i, X),
                             err = tr.err, alg = tr.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`RiskTrackingError`](@ref) whose tracked measure is resolved against a prior result, and whose benchmark takes the weights `w`.

# Algorithm

 1. Advance the benchmark with `factory(tr.tr, w)`. The benchmark stays as it is when `w` is `nothing`, or when its `fixed` flag is `true`.
 2. Resolve the tracked measure with `factory(tr.r, pr, slv, ucs, w, args...; kwargs...)`, which fills the moments that it reads from `pr`.
 3. Build a new [`RiskTrackingError`](@ref) from the two results, `tr.err` and `tr.alg`.

# Related

  - [`RiskTrackingError`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
"""
function factory(tr::RiskTrackingError, pr::AbstractPriorResult, slv::Any, ucs::Any,
                 w::Option{<:VecNum} = nothing, args...; kwargs...)::RiskTrackingError
    return RiskTrackingError(; tr = factory(tr.tr, w),
                             r = factory(tr.r, pr, slv, ucs, w, args...; kwargs...),
                             err = tr.err, alg = tr.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the benchmark or the tracked measure of [`RiskTrackingError`](@ref) `tr` reads the previous portfolio weights.

A [`WeightsTracking`](@ref) benchmark reads them unless its `fixed` flag is `true`.

# Related

  - [`RiskTrackingError`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(tr::RiskTrackingError)
    return (needs_previous_weights(tr.tr) || needs_previous_weights(tr.r))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`RiskTrackingError`](@ref) whose benchmark and tracked measure take the portfolio weights `w`.

# Algorithm

 1. Advance the benchmark with `factory(tr.tr, w)`. A [`WeightsTracking`](@ref) whose `fixed` flag is `true` stays as it is.
 2. Advance the tracked measure with `factory(tr.r, w)`.
 3. Build a new [`RiskTrackingError`](@ref) from the two results, `tr.err` and `tr.alg`.

# Related

  - [`RiskTrackingError`](@ref)
  - [`factory`](@ref)
"""
function factory(tr::RiskTrackingError, w::VecNum)::RiskTrackingError
    return RiskTrackingError(; tr = factory(tr.tr, w), r = factory(tr.r, w), err = tr.err,
                             alg = tr.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the tracking error, a scaled norm of the gap between the net portfolio returns and a benchmark return series.

The benchmark is a return series in a [`ReturnsTracking`](@ref), and a weight vector in a [`WeightsTracking`](@ref), which builds the series from the returns matrix. [`TrackingError`](@ref) bounds the same quantity in an optimisation, and [`RiskTrackingRiskMeasure`](@ref) measures a difference of risks in its place.

# Mathematical definition

The `alg` field selects the norm and the divisor that scales it.

```math
\\begin{align}
\\boldsymbol{x} &= \\mathbf{X}\\boldsymbol{w} - F(\\boldsymbol{w})\\,, \\\\
\\mathrm{TE}_{L_2}(\\boldsymbol{w}) &= \\frac{\\lVert \\boldsymbol{x} - \\boldsymbol{b} \\rVert_2}{\\sqrt{T - d}}\\,, \\\\
\\mathrm{TE}_{L_2^2}(\\boldsymbol{w}) &= \\frac{\\lVert \\boldsymbol{x} - \\boldsymbol{b} \\rVert_2^2}{T - d}\\,, \\\\
\\mathrm{TE}_{L_1}(\\boldsymbol{w}) &= \\frac{\\lVert \\boldsymbol{x} - \\boldsymbol{b} \\rVert_1}{T - d}\\,, \\\\
\\mathrm{TE}_{L_p}(\\boldsymbol{w}) &= \\frac{\\lVert \\boldsymbol{x} - \\boldsymbol{b} \\rVert_p}{(T - d)^{1/p}}\\,, \\\\
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{w}) &= \\lVert \\boldsymbol{x} - \\boldsymbol{b} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}``: Net portfolio return series ``T \\times 1``.
  - ``\\mathbf{X}``: Returns matrix ``T \\times N``.
  - $(math_dict[:w_port])
  - ``F(\\boldsymbol{w})``: Fee series ``T \\times 1`` of the portfolio, zero when `fees` is `nothing`. See [`calc_net_returns`](@ref).
  - ``\\boldsymbol{b}``: Benchmark return series ``T \\times 1``, from [`tracking_benchmark`](@ref) on `tr`. A [`WeightsTracking`](@ref) gives ``\\boldsymbol{b} = \\mathbf{X}\\boldsymbol{w}_b - F_b(\\boldsymbol{w}_b)``, with the fee ``F_b`` of its own `fees` field.
  - $(math_dict[:w_b_track])
  - ``\\mathrm{TE}_{L_2}``, ``\\mathrm{TE}_{L_2^2}``, ``\\mathrm{TE}_{L_1}``, ``\\mathrm{TE}_{L_p}``, ``\\mathrm{TE}_{L_\\infty}``: Tracking error under [`L2Norm`](@ref), [`SquaredL2Norm`](@ref), [`L1Norm`](@ref), [`LpNorm`](@ref) and [`LInfNorm`](@ref).
  - $(math_dict[:T])
  - $(math_dict[:d_ddof])
  - $(math_dict[:p_norm_order])

With ``d = 0``, the ``L_2`` and ``L_1`` forms are Equations 9.16 and 9.17 of Cajas's book. With ``d = 0``, the squared ``L_2`` form is the empirical tracking error of Benidis, Feng and Palomar.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrackingRiskMeasure(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        tr::AbstractTrackingAlgorithm,
        alg::NormError = L2Norm()
    ) -> TrackingRiskMeasure

Keywords correspond to the struct's fields.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `tr`: Recursively viewed via [`port_opt_view`](@ref).

# Functor

    (r::TrackingRiskMeasure)(w::VecNum, X::MatNum, fees = nothing)

Computes the tracking error of the portfolio weights `w` on the returns matrix `X`.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee of the portfolio.

[`calc_net_returns`](@ref) charges `fees` on the portfolio series, and [`tracking_benchmark`](@ref) charges `r.tr.fees` on the benchmark series. Both charge over the same `X`, and neither divides by a fold. A bare [`AmortisedFees`](@ref) on either fee charges its fixed amounts in full. Give a `horizon` to both fees to spread both of them.

## Precomputed portfolio returns

    (r::TrackingRiskMeasure{<:Any, <:ReturnsTracking})(x::VecNum)

Computes the tracking error of a portfolio return series `x` (``T \\times 1``) that the caller already holds.
Only a [`ReturnsTracking`](@ref) measure supports this, because its benchmark is a return
series. A [`WeightsTracking`](@ref) measure builds its benchmark from the asset returns, so it
needs the portfolio weights. On a series it throws an `ArgumentError`. See
[`supports_precomputed_returns`](@ref).

# Examples

```jldoctest
julia> TrackingRiskMeasure(; tr = ReturnsTracking(; w = [0.1, -0.2, 0.3]))
TrackingRiskMeasure
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        tr ┼ ReturnsTracking
           │   w ┴ Vector{Float64}: [0.1, -0.2, 0.3]
       alg ┼ L2Norm
           │   ddof ┴ Int64: 1
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`TurnoverRiskMeasure`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
  - [`AbstractTrackingAlgorithm`](@ref)
  - [`NormError`](@ref)
  - [`norm_error`](@ref)
  - [`port_opt_view`](@ref)
  - [`expected_risk`](@ref)

# References

  - $(ref_dict[:palomar2025])
  - $(ref_dict[:cajas2025]) Section 9.2, Equations 9.16 and 9.17.
  - $(ref_dict[:benidis2018])
"""
@propagatable @concrete struct TrackingRiskMeasure <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:tr_spec])
    """
    @vprop tr
    """
    $(field_dict[:tralg])
    """
    alg
    function TrackingRiskMeasure(settings::RiskMeasureSettings,
                                 tr::AbstractTrackingAlgorithm, alg::NormError)
        return new{typeof(settings), typeof(tr), typeof(alg)}(settings, tr, alg)
    end
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tr::AbstractTrackingAlgorithm,
                             alg::NormError = L2Norm())::TrackingRiskMeasure
    return TrackingRiskMeasure(settings, tr, alg)
end
function (r::TrackingRiskMeasure)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    benchmark = tracking_benchmark(r.tr, X)
    return norm_error(r.alg, calc_net_returns(w, X, fees), benchmark, size(X, 1))
end
function (r::TrackingRiskMeasure{<:Any, <:ReturnsTracking})(x::VecNum)
    benchmark = tracking_benchmark(r.tr, x)
    return norm_error(r.alg, x, benchmark, length(x))
end
function (r::TrackingRiskMeasure{<:Any, <:WeightsTracking})(::VecNum)
    return throw(ArgumentError("`TrackingRiskMeasure` with a `WeightsTracking` algorithm cannot be computed from a precomputed portfolio return series, because the benchmark is rebuilt from the asset returns and needs the portfolio weights. Call `r(w, X, fees)` with the weights and the asset returns matrix, or use `ReturnsTracking` to track a return series directly."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the benchmark of [`TrackingRiskMeasure`](@ref) `r` reads the previous portfolio weights.

A [`WeightsTracking`](@ref) benchmark reads them unless its `fixed` flag is `true`. A [`ReturnsTracking`](@ref) benchmark never reads them.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::TrackingRiskMeasure)
    return needs_previous_weights(r.tr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`TrackingRiskMeasure`](@ref) whose benchmark takes the portfolio weights `w`.

`factory(r.tr, w)` advances the benchmark. A [`WeightsTracking`](@ref) whose `fixed` flag is `false` takes `w` as its new benchmark weights, and any other benchmark stays as it is. `r.settings` and `r.alg` pass through unchanged.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TrackingRiskMeasure, w::VecNum)
    return TrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w), alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `factory(r, w)`, the [`TrackingRiskMeasure`](@ref) whose benchmark takes the portfolio weights `w`.

The measure reads no moment, so the prior result, the solver and the uncertainty set in the second to fourth positions are not read.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TrackingRiskMeasure, ::Any, ::Any, ::Any, w::VecNum, args...; kwargs...)
    return factory(r, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures how far the risk of a portfolio is from the risk of a benchmark portfolio, under any risk measure.

The `alg` field selects the mode. The independent mode, [`IndependentVariableTracking`](@ref), takes the risk of the weight difference. The dependent mode, [`DependentVariableTracking`](@ref), takes the absolute difference of the two risks. [`RiskTrackingError`](@ref) bounds the same quantity in an optimisation, and [`TrackingRiskMeasure`](@ref) measures a norm of the gap between two return series in its place.

In an optimisation the independent mode is exact, and the dependent mode holds from one side only. The model states the portfolio risk through an upper bound that the solver can raise. So the model penalises a portfolio that is riskier than the benchmark, and it can report zero for a portfolio that is less risky. This is true of every measure whose model is an upper bound, among them [`ConditionalValueatRisk`](@ref), [`StandardDeviation`](@ref) and a [`Variance`](@ref) in the semidefinite form. The constructor warns for a measure whose model is a quadratic expression, and a [`Variance`](@ref) outside the semidefinite form does not solve in the dependent mode.

The functor never reads `tr.fees`. It passes the `fees` of the portfolio to the tracked measure. In the independent mode the tracked measure charges the fee of ``\\boldsymbol{w} - \\boldsymbol{w}_b``, but an optimisation charges the fee of ``\\boldsymbol{w}``. So with a fee, a returns-based measure reports a value that is different from the value that the model minimises. In the dependent mode the functor and the model both charge the fee of each weight vector on its own risk.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RT}_{\\mathrm{ind}}(\\boldsymbol{w}) &= \\rho(\\boldsymbol{w} - \\boldsymbol{w}_b)\\,, \\\\
\\mathrm{RT}_{\\mathrm{dep}}(\\boldsymbol{w}) &= \\lvert \\rho(\\boldsymbol{w}) - \\rho(\\boldsymbol{w}_b) \\rvert\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RT}_{\\mathrm{ind}}(\\boldsymbol{w})``: Risk tracking in the independent mode, the risk of the weight difference.
  - ``\\mathrm{RT}_{\\mathrm{dep}}(\\boldsymbol{w})``: Risk tracking in the dependent mode, the absolute difference of the two risks.
  - $(math_dict[:w_port])
  - $(math_dict[:w_b_track])
  - $(math_dict[:rho_track])
  - ``\\mathbf{\\Sigma}``: Covariance matrix of a [`StandardDeviation`](@ref) tracked measure.

With a [`StandardDeviation`](@ref), the independent mode is ``\\sqrt{(\\boldsymbol{w} - \\boldsymbol{w}_b)^\\intercal \\mathbf{\\Sigma} (\\boldsymbol{w} - \\boldsymbol{w}_b)}``, Equation 9.18 of Cajas's book.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskTrackingRiskMeasure(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        tr::WeightsTracking,
        r::AbstractBaseRiskMeasure = Variance(),
        alg::VariableTracking = IndependentVariableTracking()
    ) -> RiskTrackingRiskMeasure

Keywords correspond to the struct's fields.

The constructor passes `r` through [`no_bounds_no_risk_expr_risk_measure`](@ref), which drops the `settings.ub` and the `settings.rke` of the tracked measure. The tracked measure only measures the distance, and `settings` of this measure holds its bound and its objective flag. In the dependent mode, the constructor logs a warning when `r` is a measure whose model is a quadratic expression, such as a [`Variance`](@ref).

## View parameters

`RiskTrackingRiskMeasure` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `tr` recurses through [`port_opt_view`](@ref) with the asset indices alone, which slices the benchmark weights.
  - `r` recurses through [`port_opt_view`](@ref) with the asset indices and the returns matrix, because a measure can slice a moment that it holds.
  - `settings` and `alg` pass through unchanged.

# Functor

    (r::RiskTrackingRiskMeasure)(w::VecNum, X::MatNum, fees = nothing)

Computes the risk tracking of the portfolio weights `w` on the returns matrix `X`, in the mode that `r.alg` selects.

## Arguments

  - `w::VecNum`: Portfolio weights vector.
  - `X::MatNum`: Asset returns matrix (``T \\times N``).
  - `fees`: Optional fee of the portfolio. The tracked measure receives it with each weight vector that it reads.

# Examples

```jldoctest
julia> RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = [0.5, 0.5]))
RiskTrackingRiskMeasure
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        tr ┼ WeightsTracking
           │    fees ┼ nothing
           │       w ┼ Vector{Float64}: [0.5, 0.5]
           │   fixed ┴ Bool: false
         r ┼ Variance
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Int64: 1
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │      sigma ┼ nothing
           │       chol ┼ nothing
           │         rc ┼ nothing
           │        alg ┴ SquaredSOCRiskExpr()
       alg ┴ IndependentVariableTracking()
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`TurnoverRiskMeasure`](@ref)
  - [`WeightsTracking`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
  - [`expected_risk`](@ref)

# References

  - $(ref_dict[:palomar2025])
  - $(ref_dict[:cajas2025]) Section 9.2, Equation 9.18.
"""
@concrete struct RiskTrackingRiskMeasure <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:tr_spec])
    """
    tr
    """
    $(field_dict[:r_risk])
    """
    r
    """
    $(field_dict[:tralg])
    """
    alg
    function RiskTrackingRiskMeasure(settings::RiskMeasureSettings, tr::WeightsTracking,
                                     r::AbstractBaseRiskMeasure, alg::VariableTracking)
        if isa(alg, DependentVariableTracking) && isa(r, QuadExpressionRiskMeasures)
            @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.")
        end
        r = no_bounds_no_risk_expr_risk_measure(r)
        return new{typeof(settings), typeof(tr), typeof(r), typeof(alg)}(settings, tr, r,
                                                                         alg)
    end
end
function RiskTrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 tr::WeightsTracking,
                                 r::AbstractBaseRiskMeasure = Variance(),
                                 alg::VariableTracking = IndependentVariableTracking())
    return RiskTrackingRiskMeasure(settings, tr, r, alg)
end
# Deferrable slots — see `deferred_slots`. The tracked measure carries its own, so both the
# check and the derived recursion in `resolve_deferred_quantities` reach them through `r`.
# `tr` holds the benchmark weights and `alg` the tracking variable, neither of which defers.
deferred_slots(r::RiskTrackingRiskMeasure) = (; r = r.r)
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:IndependentVariableTracking})(w::VecNum, X::MatNum,
                                                                     fees::Option{<:Fees} = nothing)
    wb = r.tr.w
    wd = w - wb
    return expected_risk(r.r, wd, X, fees)
end
function (r::RiskTrackingRiskMeasure{<:Any, <:Any, <:AbstractBaseRiskMeasure,
                                     <:DependentVariableTracking})(w::VecNum, X::MatNum,
                                                                   fees::Option{<:Fees} = nothing)
    wb = r.tr.w
    r1 = expected_risk(r.r, w, X, fees)
    r2 = expected_risk(r.r, wb, X, fees)
    return abs(r1 - r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`RiskTrackingRiskMeasure`](@ref) restricted to the assets at the indices `i`.

A hierarchical or clustering optimiser calls it to build the measure of one cluster. The benchmark weights of the result are the entries `i` of `r.tr.w`, so they no longer sum to the budget of the whole benchmark.

# Algorithm

 1. Slice the benchmark with `port_opt_view(r.tr, i)`.
 2. Slice the tracked measure with `port_opt_view(r.r, i, X)`.
 3. Build a new [`RiskTrackingRiskMeasure`](@ref) from the two slices, `r.settings` and `r.alg`.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(r::RiskTrackingRiskMeasure, i, X::MatNum, args...)
    tr = port_opt_view(r.tr, i)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = tr,
                                   r = port_opt_view(r.r, i, X), alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`RiskTrackingRiskMeasure`](@ref) whose tracked measure is resolved against the prior result `pr`.

`factory(r.r, pr, args...; kwargs...)` fills the moments that the tracked measure reads from `pr`. The benchmark stays as it is, even when `args` carries portfolio weights. The method `factory(r, w)` advances it.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskTrackingRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = r.tr,
                                   r = factory(r.r, pr, args...; kwargs...), alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the benchmark or the tracked measure of [`RiskTrackingRiskMeasure`](@ref) `r` reads the previous portfolio weights.

A [`WeightsTracking`](@ref) benchmark reads them unless its `fixed` flag is `true`.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::RiskTrackingRiskMeasure)
    return (needs_previous_weights(r.tr) || needs_previous_weights(r.r))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`RiskTrackingRiskMeasure`](@ref) whose benchmark and tracked measure take the portfolio weights `w`.

# Algorithm

 1. Advance the benchmark with `factory(r.tr, w)`. A [`WeightsTracking`](@ref) whose `fixed` flag is `true` stays as it is.
 2. Advance the tracked measure with `factory(r.r, w)`.
 3. Build a new [`RiskTrackingRiskMeasure`](@ref) from the two results, `r.settings` and `r.alg`.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskTrackingRiskMeasure, w::VecNum)
    return RiskTrackingRiskMeasure(; settings = r.settings, tr = factory(r.tr, w),
                                   r = factory(r.r, w), alg = r.alg)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::TrackingRiskMeasure) = WeightsReturnsFeesInput()
risk_input_kind(::RiskTrackingRiskMeasure) = WeightsReturnsFeesInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false` for a [`TrackingRiskMeasure`](@ref) with a [`WeightsTracking`](@ref) benchmark.

The measure builds the benchmark series from the returns matrix, so it needs the portfolio weights and the returns matrix, not a return series.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`WeightsTracking`](@ref)
"""
supports_precomputed_returns(::TrackingRiskMeasure{<:Any, <:WeightsTracking})::Bool = false
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` for a [`TrackingRiskMeasure`](@ref) with a [`ReturnsTracking`](@ref) benchmark.

The benchmark is a return series, so the tracking error is a function of the net portfolio return series alone.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`ReturnsTracking`](@ref)
"""
supports_precomputed_returns(::TrackingRiskMeasure{<:Any, <:ReturnsTracking})::Bool = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false` for every [`RiskTrackingRiskMeasure`](@ref), because the measure needs the portfolio weights.

Its `tr` field is a [`WeightsTracking`](@ref), and both functors read the benchmark weights from it. The independent mode takes the tracked risk of `w - r.tr.w`, and the dependent mode takes the difference of the tracked risk at `w` and at `r.tr.w`. A net return series carries no weights, so neither difference exists for it.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
  - [`WeightsTracking`](@ref)
  - [`expected_risk_from_returns`](@ref): the contract entry this predicate gates.
"""
supports_precomputed_returns(::RiskTrackingRiskMeasure)::Bool = false

export TrackingRiskMeasure, RiskTrackingRiskMeasure, RiskTrackingError
