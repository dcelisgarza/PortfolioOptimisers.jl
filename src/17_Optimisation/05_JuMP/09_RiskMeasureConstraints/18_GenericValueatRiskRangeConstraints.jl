"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add `GenericValueatRiskRange` constraints to `model` by delegating to the loss- and
gain-side sub-constraints and summing the resulting expressions.

Calls [`set_range_risk_constraints!`](@ref), which reads the two tails from
[`range_tails`](@ref) — here `r.loss` and `r.gain` as given — and builds each through
[`set_risk_constraints!`](@ref). This is the same path every other range measure takes.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::GenericValueatRiskRange`: The generic Value-at-Risk range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `genvar_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`GenericValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::GenericValueatRiskRange,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :genvar_range_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add `GenericValueatRiskRange` constraints to `model` when both tails are a [`ValueatRisk`](@ref) with [`DistributionValueatRisk`](@ref).

When the two tails read the same expected returns and the same factor of the covariance matrix, they read one standard deviation variable and one cone, as [`ValueatRiskRange`](@ref) does. Otherwise each tail builds a cone of its own, as in the method for any `GenericValueatRiskRange`.

The cone bounds ``g`` from below only, so a model is bounded when the coefficient of ``g`` is not negative. With a cone for each tail, that is ``z_{\\mathrm{loss}} \\geq 0`` and ``z_{\\mathrm{gain}} \\geq 0``. A `Normal` with a mean ``m`` far from zero makes one of the two negative, and the minimisation then has no finite solution. With one cone, the condition is ``z_{\\mathrm{loss}} + z_{\\mathrm{gain}} \\geq 0``, the condition of [`ValueatRiskRange`](@ref), which does not depend on ``m``.

The tails keep their expressions, so the `ub` of each tail bounds its own value. A bound on a tail whose coefficient is negative lets ``g`` rise above the standard deviation, so the model holds a relaxation of that bound.

# Mathematical definition

```math
\\begin{align}
\\mathrm{GenVaRRange}(\\boldsymbol{w}) &= \\left(-\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} + z_{\\mathrm{loss}}\\, g\\right) + \\left(\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} + z_{\\mathrm{gain}}\\, g\\right)\\,, \\quad g \\geq \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,.
\\end{align}
```

Where:

  - ``\\mathrm{GenVaRRange}(\\boldsymbol{w})``: Generic Value-at-Risk range.
  - ``\\boldsymbol{\\mu}``: Expected returns vector.
  - $(math_dict[:w_port])
  - ``z_{\\mathrm{loss}}``, ``z_{\\mathrm{gain}}``: The z-scores of the loss tail and of the gain tail.
  - ``g``: Standard deviation variable.
  - ``\\mathbf{G}``: Cholesky factor of the covariance matrix.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::GenericValueatRiskRange`: The generic Value-at-Risk range risk measure, with two parametric tails.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `genvar_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`GenericValueatRiskRange`](@ref)
  - [`ValueatRiskRange`](@ref)
  - [`set_distribution_var_cone!`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::GenericValueatRiskRange{<:Any,
                                                          <:ValueatRisk{<:Any, <:Any, <:Any,
                                                                        <:DistributionValueatRisk},
                                                          <:ValueatRisk{<:Any, <:Any, <:Any,
                                                                        <:DistributionValueatRisk}},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    loss, gain = r.loss.alg, r.gain.alg
    G = chol_sigma_selector(model, pr, loss)
    # Tails that read different moments have different standard deviations, so each keeps a
    # cone of its own.
    shared = (nothing_scalar_array_selector(loss.mu, pr.mu) ==
              nothing_scalar_array_selector(gain.mu, pr.mu)) &
             (G == chol_sigma_selector(model, pr, gain))
    g_var = if shared
        set_distribution_var_cone!(model, i, G, get_w(model, prefix), :g_var_range_,
                                   :cvar_range_soc_; prefix = prefix)
    else
        nothing
    end
    return set_range_risk_constraints!(model, i, r, :genvar_range_, opt, pr, args...;
                                       prefix = prefix, g_var = g_var, kwargs...)
end
