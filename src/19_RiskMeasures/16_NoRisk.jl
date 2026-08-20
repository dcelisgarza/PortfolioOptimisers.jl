"""
$(DocStringExtensions.TYPEDEF)

Risk measure that contributes no risk.

`NoRisk` measures nothing: its functor returns zero and its optimisation formulation adds a zero risk expression. It exists so that a risk-taking optimiser can express a problem that genuinely has no risk term, without a vestigial one changing the model class.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NoRisk(; settings::RiskMeasureSettings = RiskMeasureSettings()) -> NoRisk

Keywords correspond to the struct's fields.

# Details

[`MeanRisk`](@ref) requires a risk measure, so an objective that ignores risk — [`MaximumReturn`](@ref) — would otherwise still build the default [`Variance`](@ref) term. That term is discarded by the objective, but it drags second-order cone constraints into the model, forcing a conic solver on a problem that is a linear program. `NoRisk` keeps such problems linear.

Two families need this:

  - The robust best-characteristic portfolios of [quintile](@cite) — the ``1/N``, quintile and inverse-volatility portfolios — which are linear programs (see [`CharacteristicUncertaintySet`](@ref)).
  - The global maximum return portfolio (GMRP), `maximize μᵀw subject to w ≥ 0, 1ᵀw = 1`.

# Notes

`NoRisk` is only coherent under an objective that does not consult risk. [`MeanRisk`](@ref) rejects it with [`MinimumRisk`](@ref) (whose objective would be identically zero, so *any* feasible portfolio would be optimal) and with [`MaximumRatio`](@ref) (whose risk-normalisation constraint would become vacuous, leaving the model unbounded). Use it with [`MaximumReturn`](@ref), or with [`MaximumUtility`](@ref) where the risk-aversion term drops out.

Setting `settings.ub` is legal but pointless: the bound applies to a quantity that is always zero.

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MaximumReturn`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
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

Return `true` when `r` is a [`NoRisk`](@ref), or a collection holding **one**.

The quantifier is `any`, not `all`, and it is earned on the hierarchical path. [`HierarchicalRiskParity`](@ref) and [`HierarchicalEqualRiskContribution`](@ref) allocate by *inverse* risk, so a single [`NoRisk`](@ref) can drive a divisor to zero even with a real measure beside it. `NoRisk`'s functor returns zero, so its unitary risks are all zero; HRP then forms `alpha = 1 - lrisk / (lrisk + rrisk)` and HERC forms `inv.(risk) ./ sum(inv.(risk))`.

Measured on `r = [NoRisk(), Variance()]` over a 200×4 return matrix:

| Scalariser            | HRP `alpha`      | HERC inner weights |
|:--------------------- |:---------------- |:------------------ |
| `SumScalariser`       | finite           | `NaN`              |
| `MinScalariser`       | `NaN`            | `NaN`              |
| `MaxScalariser`       | finite           | finite             |
| `LogSumExpScalariser` | finite, near 0.5 | `NaN`              |

Only [`MaxScalariser`](@ref) escapes on both, being the one reduction that can discard the zero term. HERC fails under three of the four: `inv(0)` is `Inf` and `Inf / Inf` is `NaN`, and that `NaN` is produced *inside* the per-measure closure, so every scalariser that combines the terms propagates it. HRP fails only under [`MinScalariser`](@ref), where the aggregate is itself zero and `alpha` becomes `0 / 0`; under [`LogSumExpScalariser`](@ref) it stays finite but collapses towards `0.5`, an uninformative split rather than an error.

`settings.rke` is inert on the hierarchical path, so a measure's type is the only signal available there. An `all` quantifier would admit every `NaN` row above.

This is **not** the right quantifier for the JuMP axis, where the question is whether the `:risk` expression is identically zero. See [`zero_risk_expression_flag`](@ref).

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

Return `true` when the model's `:risk` expression is identically zero.

The degeneracy guard tests the **state** of the expression, not the **type** of the term (ADR
0054). Two routes reach that state, and this predicate is the disjunction of both:

 1. A [`NoRisk`](@ref) measure is present. This is [`norisk_flag`](@ref), whose quantifier is
    `any` — earned on the *hierarchical* path, where one `NoRisk` zeroes a divisor under three
    of the four scalarisers, and documented there. On **this** axis `any` is wider than the
    state it names: `[NoRisk(), Variance()]` leaves `:risk` non-zero, yet the predicate returns
    `true`, so the pair is refused under [`MinimumRisk`](@ref) and [`MaximumRatio`](@ref) with a
    cause string that overstates the case. The refusal is kept because the configuration is
    vestigial — a `NoRisk` beside a real measure buys nothing, which is the opposite of what
    [`NoRisk`](@ref) exists for — but it is a type test surviving inside a state test.
 2. Every measure carries `settings.rke = false`, so [`set_risk_expression!`](@ref) pushes
    nothing. The quantifier here is `all`, because one included measure leaves the expression
    non-zero.

The halves compose only because route 1's `any` subsumes the mixed case that a composed test
would otherwise miss. The exact mirror of the return axis is `all(isa NoRisk || !rke)`, which
agrees with this predicate everywhere except `[NoRisk(), Variance()]`, where the fused form is
the accurate one. The return axis must fuse because it has no such subsuming half; see
[`zero_return_expression_flag`](@ref).

`all` on the state route keeps a **constraint-only** measure expressible, which is the risk
side of the term [`set_return_bounds!`](@ref) supports on the return side:

```julia
r = [Variance(), Variance(; settings = RiskMeasureSettings(; rke = false, ub = u))]
```

The second measure binds a `ub` without entering the objective, and this predicate does not
refuse it.

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

Assert that a zero risk expression is paired with an objective that ignores risk.

Rejects a zero `:risk` under [`MinimumRisk`](@ref) — whose objective would be identically zero, so the solver could return *any* feasible portfolio, silently — and under [`MaximumRatio`](@ref), whose risk-normalisation constraint would go vacuous and leave the model unbounded.

The criterion is [`zero_risk_expression_flag`](@ref), so the guard covers both routes to a zero expression: a [`NoRisk`](@ref) measure, and every measure carrying `settings.rke = false` (ADR 0054). The second route shipped unguarded since the inclusion flag was introduced.

Called from [`MeanRisk`](@ref)'s constructor. [`TimeDependent`](@ref) schedules are skipped here and reached instead through [`assert_time_dependent_substitution`](@ref), which re-runs the constructor on each scheduled entry.

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

Name the route by which the risk expression became identically zero, for an error message.

# Related

  - [`zero_risk_expression_flag`](@ref)
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

Assert that `r` gives a non-zero risk expression, for optimisers built around one.

A zero risk expression is only coherent in [`MeanRisk`](@ref), under an objective that never consults it. Every other risk-taking optimiser *is* its risk measure — a risk budget with nothing to budget, a risk contribution that is always zero, a clustering optimiser dividing by a zero risk — so they reject it rather than return a degenerate answer.

`flag` selects the predicate, because the two families of caller do not see the same routes (ADR 0054):

  - The JuMP optimisers pass [`zero_risk_expression_flag`](@ref), which covers a [`NoRisk`](@ref) measure **and** every measure carrying `settings.rke = false`.
  - [`HierarchicalRiskParity`](@ref) and [`HierarchicalEqualRiskContribution`](@ref) keep the default [`norisk_flag`](@ref). They never reach the JuMP risk builders, so `rke` is inert for them, and widening the predicate would refuse a configuration that solves correctly today.

`T` names the calling optimiser, for the error message. [`TimeDependent`](@ref) schedules are skipped here and reached instead through [`assert_time_dependent_substitution`](@ref).

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
