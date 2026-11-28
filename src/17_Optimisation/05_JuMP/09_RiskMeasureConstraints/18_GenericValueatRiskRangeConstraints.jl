function range_risk_key(::GenericValueatRiskRange)
    return :genvar_range_risk_
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add `GenericValueatRiskRange` constraints to `model` when both tails are a [`ValueatRisk`](@ref) with [`DistributionValueatRisk`](@ref).

When the two tails read the same factor of the covariance matrix, they read one standard deviation variable and one cone, as [`ValueatRiskRange`](@ref) does. Otherwise each tail builds a cone of its own, as in the method of `set_risk_constraints!` for a [`RangeRiskMeasure`](@ref). The cone bounds the standard deviation ``\\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2``, which does not read the expected returns, so tails with different expected returns share it too. Each tail reads its own expected returns in its expression.

The cone bounds ``g`` from below only, so a model is bounded when the coefficient of ``g`` is not negative. With a cone for each tail, that is ``z_{\\mathrm{loss}} \\geq 0`` and ``z_{\\mathrm{gain}} \\geq 0``. A `Normal` with a mean ``m`` far from zero makes one of the two negative, and the minimisation then has no finite solution. With one cone, the condition is ``z_{\\mathrm{loss}} + z_{\\mathrm{gain}} \\geq 0``, the condition of [`ValueatRiskRange`](@ref), which does not depend on ``m``.

The tails keep their expressions, so the `ub` of each tail bounds its own value, and a [`Frontier`](@ref) in that `ub` spans the values of its own tail. A bound on a tail whose coefficient is negative lets ``g`` rise above the standard deviation, so the model holds a relaxation of that bound.

# Mathematical definition

```math
\\begin{align}
\\mathrm{GenVaRRange}(\\boldsymbol{w}) &= \\left(-\\boldsymbol{\\mu}_{\\mathrm{loss}}^\\intercal \\boldsymbol{w} + z_{\\mathrm{loss}}\\, g\\right) + \\left(\\boldsymbol{\\mu}_{\\mathrm{gain}}^\\intercal \\boldsymbol{w} + z_{\\mathrm{gain}}\\, g\\right)\\,, \\quad g \\geq \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,.
\\end{align}
```

Where:

  - ``\\mathrm{GenVaRRange}(\\boldsymbol{w})``: Generic Value-at-Risk range.
  - ``\\boldsymbol{\\mu}_{\\mathrm{loss}}``, ``\\boldsymbol{\\mu}_{\\mathrm{gain}}``: The expected returns vectors of the loss tail and of the gain tail.
  - $(math_dict[:w_port])
  - ``z_{\\mathrm{loss}}``, ``z_{\\mathrm{gain}}``: The z-scores of the loss tail and of the gain tail.
  - ``g``: Standard deviation variable.
  - ``\\mathbf{G}``: Cholesky factor of the covariance matrix.

# Algorithm

 1. Read the tails `loss` and `gain` with [`range_tails`](@ref), and the key `name` with [`range_risk_key`](@ref). Read the Cholesky factor of each tail with [`chol_sigma_selector`](@ref). When the two factors are equal, build the cone of the shared standard deviation `g_var` with [`set_distribution_var_cone!`](@ref). Otherwise `g_var` is `nothing`, and each tail builds a cone of its own.
 2. Build the two tails, register their sum and bound it, as steps 2 to 5 of the method of `set_risk_constraints!` for a [`RangeRiskMeasure`](@ref) do. Each tail reads `g_var`. This copy of the steps agrees with that method.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::GenericValueatRiskRange`: The generic Value-at-Risk range risk measure, with two parametric tails.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `range_risk`: The combined `loss + gain` risk expression added to the model, registered under the key `genvar_range_risk_`.

# Related

  - [`GenericValueatRiskRange`](@ref)
  - [`ValueatRiskRange`](@ref)
  - [`RangeRiskMeasure`](@ref)
  - [`range_risk_key`](@ref)
  - [`set_risk_frontier_owner!`](@ref)
  - [`GainTail`](@ref)
  - [`set_distribution_var_cone!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::GenericValueatRiskRange{<:Any,
                                                          <:ValueatRisk{<:Any, <:Any, <:Any,
                                                                        <:DistributionValueatRisk},
                                                          <:ValueatRisk{<:Any, <:Any, <:Any,
                                                                        <:DistributionValueatRisk}},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    (; loss, gain) = range_tails(r)
    name = range_risk_key(r)
    G = chol_sigma_selector(model, pr, loss.alg)
    # The cone bounds the standard deviation, which reads `G` and not `mu`. Tails that read
    # different covariance matrices have different standard deviations, so each keeps a cone of
    # its own.
    g_var = if G == chol_sigma_selector(model, pr, gain.alg)
        set_distribution_var_cone!(model, i, G, get_w(model, prefix), :g_var_range_,
                                   :cvar_range_soc_; prefix = prefix)
    else
        nothing
    end
    # A sweep on a tail takes its span from that tail, so each tail records the measure of its
    # own side as the owner of its entry before the whole range claims the rest.
    first = risk_frontier_length(model)
    loss_risk = set_risk_constraints!(model, nested_index(:loss_, i), loss, opt, pr,
                                      args...; loss = true, prefix = prefix, g_var = g_var,
                                      kwargs...)
    set_risk_frontier_owner!(model, first, loss)
    first = risk_frontier_length(model)
    gain_risk = set_risk_constraints!(model, nested_index(:gain_, i), gain, opt, pr,
                                      args...; loss = false, prefix = prefix, g_var = g_var,
                                      kwargs...)
    set_risk_frontier_owner!(model, first, GainTail(gain))
    range_risk = state_set!(model, prefix, name, i,
                            JuMP.@expression(model, loss_risk + gain_risk))
    set_risk_bounds_and_expression!(model, opt, range_risk, r.settings, name, i;
                                    prefix = prefix)
    return range_risk
end
