"""
$(DocStringExtensions.TYPEDEF)

Measures the turnover of a portfolio, the ``L^1`` distance of its weights from a vector of reference weights.

The measure reads no returns. It reads the portfolio weights and the reference weights `w`, and nothing else. As a risk term it penalises the trades of a rebalance, and with `settings.ub` it bounds their sum. [`Turnover`](@ref) bounds the change of each asset, and this measure bounds or penalises the sum of the changes.

The measure is the full sum of the absolute changes, the two-way turnover. A purchase and the sale that pays for it both count. So a move of a weight of 0.25 from one asset to another has a turnover of 0.5. The one-way convention halves this sum, and the measure does not.

`fixed` sets the source of the reference weights. With `fixed = false`, [`factory`](@ref) replaces `w` with the weights that it receives, so in a sequence of rebalances `w` holds the weights of the last rebalance. [`needs_previous_weights`](@ref) then returns `true`. With `fixed = true`, `w` is a fixed target, and `factory` keeps it. The constructor has no default for `w`, so a measure always holds reference weights. A `factory` call that receives no weights keeps them.

In an optimisation the model is exact. The risk that the model reports, divided by the budget variable ``k`` of a ratio problem, is equal to the functor at the optimal weights.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Turnover}(\\boldsymbol{w}) &= \\lVert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\rVert_1 = \\sum_{i=1}^{N} \\lvert w_i - w_{0,i} \\rvert\\,.
\\end{align}
```

Where:

  - $(math_dict[:Tn_turnover])
  - $(math_dict[:w_port])
  - $(math_dict[:w_0_turnover])
  - $(math_dict[:w_i_asset])
  - ``w_{0,i}``: Reference weight of asset ``i``.
  - $(math_dict[:N])

Equation 9.10 of [cajas2025](@cite) bounds the absolute change of each asset, ``\\lvert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\rvert``. This measure is the sum of that vector.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TurnoverRiskMeasure(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::VecNum,
        fixed::Bool = false
    ) -> TurnoverRiskMeasure

Keywords correspond to the struct's fields.

## Validation

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`. Every entry must be finite, so the constructor refuses `[0.5, NaN]`.

# Functor

    (r::TurnoverRiskMeasure)(w::VecNum)

Computes the turnover of the portfolio weights `w` from the reference weights `r.w`.

## Arguments

  - `w::VecNum`: Portfolio weights vector. It has the length of `r.w`. A vector of a different length raises a `DimensionMismatch`.

The measure holds no [`Fees`](@ref), so fee amortisation cannot reach it. Each fold evaluates it against the reference weights that [`factory`](@ref) gives.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `w`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> TurnoverRiskMeasure(; w = [0.5, 0.5])
TurnoverRiskMeasure
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ Vector{Float64}: [0.5, 0.5]
     fixed ┴ Bool: false

julia> TurnoverRiskMeasure(; w = [0.5, 0.25, 0.25])([0.25, 0.5, 0.25])
0.5
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`Turnover`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`needs_previous_weights`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.10. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct TurnoverRiskMeasure <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w_ref])
    """
    @vprop w
    """
    $(field_dict[:fixed])
    """
    fixed
    function TurnoverRiskMeasure(settings::RiskMeasureSettings, w::VecNum, fixed::Bool)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(settings), typeof(w), typeof(fixed)}(settings, w, fixed)
    end
end
function TurnoverRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::VecNum, fixed::Bool = false)::TurnoverRiskMeasure
    return TurnoverRiskMeasure(settings, w, fixed)
end
function (r::TurnoverRiskMeasure)(w::VecNum)
    return LinearAlgebra.norm(r.w - w, 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when [`TurnoverRiskMeasure`](@ref) `r` reads the previous portfolio weights.

The measure reads them when `r.fixed` is `false`, because [`factory`](@ref) then replaces the reference weights with the weights that it receives. When `r.fixed` is `true`, the reference weights are a fixed target, and the function returns `false`.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(r::TurnoverRiskMeasure)
    return !r.fixed
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a [`TurnoverRiskMeasure`](@ref) whose reference weights are `w`.

When `r.fixed` is `true`, the function returns `r` and does not read `w`. Otherwise it builds a new measure with `w` as the reference weights, and `r.settings` and `r.fixed` pass through unchanged.

# Validation

  - When `r.fixed` is `false`, the constructor checks `w` through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TurnoverRiskMeasure, w::VecNum)::TurnoverRiskMeasure
    return if r.fixed
        r
    else
        TurnoverRiskMeasure(; settings = r.settings, w = w, fixed = r.fixed)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `factory(r, w)`, the [`TurnoverRiskMeasure`](@ref) whose reference weights are `w`.

The measure reads no moment, so the prior result, the solver and the uncertainty set in the second to fourth positions are not read. When `w` is `nothing`, the function returns `r` unchanged.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::TurnoverRiskMeasure, ::Any, ::Any, ::Any, w::Option{<:VecNum} = nothing,
                 args...; kwargs...)::TurnoverRiskMeasure
    return factory(r, w)
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::TurnoverRiskMeasure) = WeightsInput()

export TurnoverRiskMeasure
