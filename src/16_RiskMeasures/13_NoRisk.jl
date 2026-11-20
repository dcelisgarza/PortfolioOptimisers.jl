"""
$(DocStringExtensions.TYPEDEF)

Measures no risk: its value is zero for every portfolio.

The JuMP formulation adds a zero risk expression and no row, so a linear programme stays a linear programme. [`MeanRisk`](@ref) needs a risk measure, and without `NoRisk` an objective that ignores the risk still builds the default [`Variance`](@ref) term. That term adds second-order cone rows, so the model needs a conic solver. Problems 9, 14 and 21 of [quintile](@cite) are linear programmes of this kind. They give the robust ``1/N`` and quintile portfolios, the dollar-neutral quintile portfolio and the inverse-volatility quintile portfolio. The global maximum return portfolio of Table I of [quintile](@cite) is another one.

`NoRisk` is valid only under an objective that does not read the risk.

  - [`MeanRisk`](@ref) accepts it under [`MaximumReturn`](@ref) and under [`MaximumUtility`](@ref). Under [`MaximumUtility`](@ref) the risk term is zero, so the two objectives give the same weights.
  - [`MeanRisk`](@ref) refuses it under [`MinimumRisk`](@ref) and under [`MaximumRatio`](@ref), through [`assert_no_risk_objective_compatibility`](@ref).
  - [`RiskBudgeting`](@ref), [`FactorRiskContribution`](@ref), [`NearOptimalCentering`](@ref), [`HierarchicalRiskParity`](@ref) and [`HierarchicalEqualRiskContribution`](@ref) refuse it, through [`assert_risk_measure_required`](@ref).

A refusal is an `ArgumentError` from the constructor of the optimiser, and [`zero_risk_expression_cause`](@ref) names the cause in its message. A bound `settings.ub` has no effect, because [`RiskMeasureSettings`](@ref) refuses a negative bound and the risk is always zero.

# Mathematical definition

```math
\\begin{align}
\\mathrm{NoRisk}(\\boldsymbol{x}) &= 0\\,.
\\end{align}
```

Where:

  - ``\\mathrm{NoRisk}(\\boldsymbol{x})``: Risk of the portfolio, which is zero.
  - $(math_dict[:xret])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NoRisk(; settings::RiskMeasureSettings = RiskMeasureSettings()) -> NoRisk

Keywords correspond to the struct's fields.

# Functor

    (r::NoRisk)(x::VecNum)

Returns zero, with the element type of the portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> r = NoRisk()
NoRisk
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true

julia> r([0.1, -0.2, 0.05])
0.0

julia> r(Float32[0.1, -0.2])
0.0f0
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MaximumReturn`](@ref)
  - [`CharacteristicUncertaintySet`](@ref): The uncertainty set of the robust portfolios of [quintile](@cite).
  - [`norisk_flag`](@ref)
  - [`zero_risk_expression_flag`](@ref)

# References

  - $(ref_dict[:quintile]) Problems 9, 14 and 21, and Table I.
"""
@concrete struct NoRisk <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    function NoRisk(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function NoRisk(; settings::RiskMeasureSettings = RiskMeasureSettings())::NoRisk
    return NoRisk(settings)
end
function (::NoRisk)(x::VecNum)
    return zero(eltype(x))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::NoRisk) = NetReturnsInput()
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `r` is a [`NoRisk`](@ref), or a vector that holds at least one.

The test is `any`, not `all`, because of the hierarchical optimisers. [`HierarchicalRiskParity`](@ref) and [`HierarchicalEqualRiskContribution`](@ref) allocate by the inverse of a risk. The risk of [`NoRisk`](@ref) is zero, so one `NoRisk` can make a divisor zero even beside a real measure. HRP forms `alpha = 1 - lrisk / (lrisk + rrisk)`, and HERC forms `inv.(risk) ./ sum(inv.(risk))` for each measure.

The table gives the result for `r = [NoRisk(), Variance()]` over a 200×4 returns matrix, when no check refuses the vector.

| Scalariser            | HRP weights   | HERC weights |
|:--------------------- |:------------- |:------------ |
| `SumScalariser`       | finite        | `NaN`        |
| `MinScalariser`       | `NaN`         | `NaN`        |
| `MaxScalariser`       | finite        | finite       |
| `LogSumExpScalariser` | an even split | `NaN`        |

[`MaxScalariser`](@ref) is the only scalariser that gives finite weights for both, because it can discard the zero term. HERC computes `inv(0) = Inf` and then `Inf / Inf = NaN` for each measure, before the scalariser combines the measures, so every other scalariser keeps the `NaN`. HRP fails under [`MinScalariser`](@ref) alone, where the combined risk is zero and `alpha` is `0 / 0`. Under [`LogSumExpScalariser`](@ref), `alpha` is near `0.5`, so the split carries no information.

The hierarchical optimisers do not read `settings.rke`, so the type of a measure is the only signal there. A test with `all` accepts every `NaN` row of the table. The JuMP optimisers ask a different question, whether the `:risk` expression is zero, and [`zero_risk_expression_flag`](@ref) answers it.

# Arguments

  - $(arg_dict[:r])

# Returns

  - `Bool`: `true` when `r` is or holds a [`NoRisk`](@ref).

# Related

  - [`NoRisk`](@ref)
  - [`assert_no_risk_objective_compatibility`](@ref)
  - [`assert_risk_measure_required`](@ref)
"""
function norisk_flag(r)::Bool
    return isa(r, NoRisk) || (isa(r, AbstractVector) && any(x -> isa(x, NoRisk), r))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the `:risk` expression of the model is zero.

The function tests the state of the expression, and not the type of a measure. Two cases give a zero expression, and the function returns `true` when either holds.

 1. A [`NoRisk`](@ref) measure is present, which [`norisk_flag`](@ref) tests with `any`. Here
    `any` is wider than the state: `[NoRisk(), Variance()]` gives a `:risk` that is not zero,
    and the function still returns `true`. So [`MinimumRisk`](@ref) and [`MaximumRatio`](@ref)
    refuse the pair, with a message that overstates the cause. A `NoRisk` beside a real
    measure has no use, so the refusal stays.
 2. Every measure sets `settings.rke = false`, so [`set_risk_expression!`](@ref) adds nothing to
    the expression. The test is `all`, because one measure that enters keeps the expression
    away from zero.

The two cases combine only because case 1 already covers the mixed vector. The return side
tests `all(isa NoRisk || !rke)`, which agrees with this function except at
`[NoRisk(), Variance()]`, where the return form is the accurate one. The return side has no
case like case 1, so it must use the combined form. See [`zero_return_expression_flag`](@ref).

The `all` of case 2 lets a measure bound the risk and stay out of the objective. This is the
risk side of what [`set_return_bounds!`](@ref) does for the return:

```julia
r = [Variance(), Variance(; settings = RiskMeasureSettings(; rke = false, ub = u))]
```

The second measure bounds the risk by `ub` and does not enter the objective, and the function
returns `false`.

# Arguments

  - $(arg_dict[:r])

# Returns

  - `Bool`: `true` when the risk expression of `r` is zero.

# Related

  - [`norisk_flag`](@ref)
  - [`zero_return_expression_flag`](@ref)
  - [`assert_risk_measure_required`](@ref)
  - [`assert_no_risk_objective_compatibility`](@ref)
"""
function zero_risk_expression_flag(r)::Bool
    return norisk_flag(r) || (if isa(r, AbstractVector)
                                  !isempty(r) && all(x -> !x.settings.rke, r)
                              else
                                  !r.settings.rke
                              end)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that a zero risk expression comes with an objective that does not read the risk.

Under [`MinimumRisk`](@ref) a zero `:risk` makes the objective zero, so every feasible portfolio is optimal and the solver returns any one of them. Under [`MaximumRatio`](@ref) the row that fixes the risk of the ratio has no effect, and the model is unbounded. [`zero_risk_expression_flag`](@ref) decides whether the expression is zero, so the check covers a [`NoRisk`](@ref) measure and a vector whose every measure sets `settings.rke = false`.

The constructor of [`MeanRisk`](@ref) calls the function. It skips a [`TimeDependent`](@ref) schedule, because [`assert_time_dependent_substitution`](@ref) runs the constructor again on each entry of the schedule.

# Arguments

  - $(arg_dict[:r])
  - `obj`: Objective function of the optimiser.

# Validation

  - When [`zero_risk_expression_flag`](@ref) is `true`: `!isa(obj, MinimumRisk)` and `!isa(obj, MaximumRatio)`. A failure throws an `ArgumentError` whose message names the cause from [`zero_risk_expression_cause`](@ref).

# Returns

  - `nothing`.

# Related

  - [`NoRisk`](@ref)
  - [`MeanRisk`](@ref)
  - [`zero_risk_expression_flag`](@ref)
  - [`assert_risk_measure_required`](@ref)
"""
function assert_no_risk_objective_compatibility(r, obj)::Nothing
    if isa(r, TimeDependent) || isa(obj, TimeDependent) || !zero_risk_expression_flag(r)
        return nothing
    end
    cause = zero_risk_expression_cause(r)
    @argcheck(!isa(obj, MinimumRisk),
              ArgumentError("A zero risk expression is incompatible with MinimumRisk: $cause, so the objective would be identically zero, every feasible portfolio is optimal, and the solver would return an arbitrary one. Use obj = MaximumReturn(), or give a risk measure that enters the risk expression."))
    @argcheck(!isa(obj, MaximumRatio),
              ArgumentError("A zero risk expression is incompatible with MaximumRatio: $cause, so the ratio's normalising constraint is vacuous and the model unbounded. Use obj = MaximumReturn(), or give a risk measure that enters the risk expression."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the cause of a zero risk expression, as a phrase for an error message.

The function names a [`NoRisk`](@ref) measure when [`norisk_flag`](@ref) is `true`. Otherwise it names the `settings.rke = false` of every measure. A caller calls it only after [`zero_risk_expression_flag`](@ref) returns `true`.

# Arguments

  - $(arg_dict[:r])

# Returns

  - `String`: The cause, as a phrase that follows "but" or a colon in a sentence.

# Related

  - [`zero_risk_expression_flag`](@ref)
  - [`assert_no_risk_objective_compatibility`](@ref)
  - [`assert_risk_measure_required`](@ref)
"""
function zero_risk_expression_cause(r)::String
    return if norisk_flag(r)
        "a `NoRisk` measure is present and it measures nothing"
    else
        "every risk measure carries `settings.rke = false`, so none of them enters the risk expression"
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that `r` gives a risk expression that is not zero, for an optimiser that is built on its risk.

Only [`MeanRisk`](@ref) can use a zero risk expression, under an objective that does not read it. Every other optimiser that takes a risk measure depends on the risk. A risk budget has nothing to share, a risk contribution is zero, and a hierarchical optimiser divides by a zero risk. So these optimisers refuse the measure and do not return a degenerate result.

`flag` selects the test, because the two kinds of caller do not see the same cases.

  - [`RiskBudgeting`](@ref), [`FactorRiskContribution`](@ref) and [`NearOptimalCentering`](@ref) pass [`zero_risk_expression_flag`](@ref). It covers a [`NoRisk`](@ref) measure and a vector whose every measure sets `settings.rke = false`.
  - [`HierarchicalRiskParity`](@ref) and [`HierarchicalEqualRiskContribution`](@ref) keep the default [`norisk_flag`](@ref). They do not build a JuMP risk expression, so they do not read `rke`, and the wider test refuses a measure that they use correctly.

The function skips a [`TimeDependent`](@ref) schedule, because [`assert_time_dependent_substitution`](@ref) checks each entry of the schedule.

# Arguments

  - $(arg_dict[:r])
  - `T::Symbol`: Name of the calling optimiser, for the error message.
  - `flag`: Test of a zero risk expression, [`norisk_flag`](@ref) or [`zero_risk_expression_flag`](@ref).

# Validation

  - `!flag(r)`. A failure throws an `ArgumentError` that names `T` and the cause from [`zero_risk_expression_cause`](@ref).

# Returns

  - `nothing`.

# Related

  - [`NoRisk`](@ref)
  - [`zero_risk_expression_flag`](@ref)
  - [`assert_no_risk_objective_compatibility`](@ref)
"""
function assert_risk_measure_required(r, T::Symbol; flag = norisk_flag)::Nothing
    if isa(r, TimeDependent) || !flag(r)
        return nothing
    end
    cause = zero_risk_expression_cause(r)
    return throw(ArgumentError("$T needs a non-zero risk expression, but $cause. $T's formulation is built around a risk measure, so the result would be degenerate rather than risk-free. Only MeanRisk can express a problem with no risk term at all, under an objective that never reads it (e.g. obj = MaximumReturn()). Give a risk measure that enters the risk expression."))
end

export NoRisk
