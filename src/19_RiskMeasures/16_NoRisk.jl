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

Return `true` when `r` is a [`NoRisk`](@ref), or a collection holding one.

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

Assert that a [`NoRisk`](@ref) measure is paired with an objective that ignores risk.

Rejects [`NoRisk`](@ref) under [`MinimumRisk`](@ref) — whose objective would be identically zero, so the solver could return *any* feasible portfolio, silently — and under [`MaximumRatio`](@ref), whose risk-normalisation constraint would go vacuous and leave the model unbounded.

Called from [`MeanRisk`](@ref)'s constructor. [`TimeDependent`](@ref) schedules are skipped here and reached instead through [`assert_time_dependent_substitution`](@ref), which re-runs the constructor on each scheduled entry.

# Related

  - [`NoRisk`](@ref)
  - [`MeanRisk`](@ref)
  - [`assert_risk_measure_required`](@ref)
"""
function assert_no_risk_objective_compatibility(r, obj)::Nothing
    if isa(r, TimeDependent) || isa(obj, TimeDependent) || !norisk_flag(r)
        return nothing
    end
    @argcheck(!isa(obj, MinimumRisk),
              ArgumentError("NoRisk is incompatible with MinimumRisk: the objective would be identically zero, so every feasible portfolio is optimal and the solver would return an arbitrary one. NoRisk exists so that an objective which ignores risk can avoid a vestigial risk term — use obj = MaximumReturn(), or give a real risk measure."))
    @argcheck(!isa(obj, MaximumRatio),
              ArgumentError("NoRisk is incompatible with MaximumRatio: the ratio is normalised by the risk, so a zero risk makes the normalising constraint vacuous and the model unbounded. Use obj = MaximumReturn(), or give a real risk measure."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `r` is a real risk measure, for optimisers whose formulation is built around one.

[`NoRisk`](@ref) is only coherent in [`MeanRisk`](@ref), under an objective that never consults the risk expression. Every other risk-taking optimiser *is* its risk measure — a risk budget with nothing to budget, a risk contribution that is always zero, a clustering optimiser dividing by a zero risk — so they reject it rather than return a degenerate answer.

`T` names the calling optimiser, for the error message. [`TimeDependent`](@ref) schedules are skipped here and reached instead through [`assert_time_dependent_substitution`](@ref).

# Related

  - [`NoRisk`](@ref)
  - [`assert_no_risk_objective_compatibility`](@ref)
"""
function assert_risk_measure_required(r, T::Symbol)::Nothing
    if isa(r, TimeDependent) || !norisk_flag(r)
        return nothing
    end
    return throw(ArgumentError("NoRisk cannot be used with $T: it measures nothing, and $T's formulation is built around a risk measure, so the result would be degenerate rather than risk-free. NoRisk exists only so MeanRisk can express a problem with no risk term at all (e.g. obj = MaximumReturn()). Give a real risk measure."))
end

export NoRisk
