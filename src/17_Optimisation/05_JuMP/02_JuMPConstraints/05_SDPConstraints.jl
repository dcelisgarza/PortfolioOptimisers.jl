"""
    weights_prefix(model::JuMP.Model, prefix::Symbol)

Returns the Model State namespace that owns the weights registered under `prefix`.

One lifted matrix ``\\mathbf{W}`` belongs to one weight vector, and so do the marks that describe the measures built on it, `variance_flag` and `rc_variance`. A build that registers weights it did not make also registers `w_owner`, the namespace that owns them. A programme Allocation Set in a leader's model registers the model's own `w` under `:aset_` with the owner `Symbol("")`. A [`DependentVariableTracking`](@ref) build registers the weights of its enclosing build under its tracking prefix, with the owner of those weights. The set's ceiling, the tracking build's inner measures and the head's measures then read one ``\\mathbf{W}`` and one set of marks, as the measures of one head do. A prefix without `w_owner` owns itself, as an [`IndependentVariableTracking`](@ref) prefix does, because the benchmark shifts its weights.

# Arguments

  - $(arg_dict[:model])
  - `prefix`: The Model State namespace the weights are registered under.

# Returns

  - `owner::Symbol`: The `w_owner` entry under `prefix` when there is one, and `prefix` otherwise.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`get_w`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
"""
function weights_prefix(model::JuMP.Model, prefix::Symbol)
    return state_has(model, prefix, :w_owner) ? state_get(model, prefix, :w_owner) : prefix
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Mark `variance_flag` on the weights under `prefix` when the variance is a positive term of the objective's risk.

The PSD cone ``[\\mathbf{W}\\ \\boldsymbol{w};\\ \\boldsymbol{w}^\\intercal\\ k] \\succeq 0`` forces only ``\\mathbf{W} \\succeq \\boldsymbol{w}\\boldsymbol{w}^\\intercal``, and ``\\mathbf{W} + t\\,\\boldsymbol{e}_i\\boldsymbol{e}_i^\\intercal`` stays feasible for every ``t \\geq 0``, also under the phylogeny rows ``\\mathbf{A} \\odot \\mathbf{W} = \\mathbf{0}``, because ``A_{ii} = 0``. That step adds ``s\\,t\\,\\Sigma_{ii}`` to a variance term ``s\\,\\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W})``. A variance that the objective minimises with ``s > 0`` therefore prices the growth of ``\\mathbf{W}``, as the phylogeny's ``p\\,\\mathrm{tr}(\\mathbf{W})`` penalty does, and [`set_sdp_phylogeny_constraints!`](@ref) can omit the penalty.

A variance that is only a bound puts no price on the growth, and neither does the inner variance of a [`RiskTrackingRiskMeasure`](@ref). The constructor of the tracking measure clears the inner `rke`. The dependent term ``|\\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W}) - r_b k|`` rewards a larger ``\\mathbf{W}`` when the portfolio's variance is below the benchmark's. So neither marks the flag. The rule reads each variance alone, and the objective's role is read by [`mark_risk_minimised!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `prefix`: The Model State namespace of the build. The mark goes to its owner, [`weights_prefix`](@ref).
  - `settings`: The variance's settings. The mark needs `settings.rke` and `settings.scale > 0`.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`mark_risk_minimised!`](@ref)
  - [`Variance`](@ref)
"""
function mark_objective_variance!(model::JuMP.Model, prefix::Symbol,
                                  settings::RiskMeasureSettings)
    if settings.rke && settings.scale > zero(settings.scale)
        mark_state!(model, weights_prefix(model, prefix), :variance_flag)
    end
    return nothing
end
"""
    mark_risk_minimised!(model::JuMP.Model, obj::ObjectiveFunction)

Mark `risk_minimised` when the objective `obj` minimises the risk term.

[`assemble_jump_model!`](@ref) calls it after the return constraints, so a [`MaximumRatio`](@ref) has chosen its form. A marked variance holds the lifted matrix down only when the objective minimises it, [`mark_objective_variance!`](@ref).

  - [`MinimumRisk`](@ref) minimises the risk.
  - [`MaximumUtility`](@ref) minimises it when `l > 0`.
  - [`MaximumRatio`](@ref) minimises it in the return form. In the risk form, which registers `sr_risk`, the risk is a bound.
  - Every other objective, such as [`MaximumReturn`](@ref), leaves the risk out of the objective.

# Arguments

  - $(arg_dict[:model])
  - `obj`: The objective that the model builds.

# Returns

  - `nothing`.

# Related

  - [`mark_objective_variance!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
"""
function mark_risk_minimised!(::JuMP.Model, ::ObjectiveFunction)
    return nothing
end
function mark_risk_minimised!(model::JuMP.Model, ::MinimumRisk)
    shared_set!(model, :risk_minimised, true)
    return nothing
end
function mark_risk_minimised!(model::JuMP.Model, obj::MaximumUtility)
    if obj.l > zero(obj.l)
        shared_set!(model, :risk_minimised, true)
    end
    return nothing
end
function mark_risk_minimised!(model::JuMP.Model, ::MaximumRatio)
    if !shared_has(model, :sr_risk)
        shared_set!(model, :risk_minimised, true)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a positive semidefinite (PSD) constraint to the JuMP optimisation model for the portfolio weights.

Creates a symmetric matrix variable `W` and enforces that the bordered matrix `[W w; wᵀ k]` lies in the PSD cone. Returns immediately if `W` already exists in `model`. The matrix is registered under the namespace that owns the weights under `prefix`, [`weights_prefix`](@ref), so weights registered twice have one `W`.

# Mathematical definition

```math
\\begin{align}
\\mathbf{M} &= \\begin{bmatrix} \\mathbf{W} & \\boldsymbol{w} \\\\ \\boldsymbol{w}^\\intercal & k \\end{bmatrix} \\succeq 0 \\\\
&\\quad\\Leftrightarrow\\quad \\mathbf{W} \\succeq \\frac{\\boldsymbol{w}\\boldsymbol{w}^\\intercal}{k}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}``: Bordered positive semidefinite matrix.
  - ``\\mathbf{W}``: Symmetric ``N \\times N`` matrix variable.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

# Arguments

  - $(arg_dict[:model])

# Returns

  - `W`: Symmetric JuMP variable matrix of size `N × N`.

# Related

  - [`set_sdp_frc_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_constraints!(model::JuMP.Model; prefix::Symbol = Symbol(""))
    prefix = weights_prefix(model, prefix)
    return state_build!(model, prefix, :W) do
        w = get_w(model, prefix)
        k = effective_k(model)
        sc = get_constraint_scale(model)
        N = length(w)
        W = JuMP.@variable(model, [1:N, 1:N], Symmetric)
        M = state_set!(model, prefix, :M,
                       JuMP.@expression(model, hcat(vcat(W, transpose(w)), vcat(w, k))))
        state_set!(model, prefix, :M_PSD, JuMP.@constraint(model, sc * M in JuMP.PSDCone()))
        return W
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a positive semidefinite (PSD) constraint for factor risk contribution to the JuMP optimisation model.

Creates a symmetric matrix variable `frc_W` and enforces that the bordered matrix `[frc_W w1; w1ᵀ k]` lies in the PSD cone. Returns immediately if `frc_W` already exists in `model`.

# Arguments

  - $(arg_dict[:model])

# Returns

  - `frc_W`: Symmetric JuMP variable matrix of size `Nf × Nf`.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_frc_constraints!(model::JuMP.Model)
    if shared_has(model, :frc_W)
        return shared_get(model, :frc_W)
    end
    w1 = shared_get(model, :w1)
    sc = get_constraint_scale(model)
    k = get_k(model)
    Nf = length(w1)
    JuMP.@variable(model, frc_W[1:Nf, 1:Nf], Symmetric)
    JuMP.@expression(model, frc_M, hcat(vcat(frc_W, transpose(w1)), vcat(w1, k)))
    JuMP.@constraint(model, frc_M_PSD, sc * frc_M in JuMP.PSDCone())
    return frc_W
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add semidefinite phylogeny constraints to the JuMP optimisation model.

Iterates over `plgs` and, for each [`SemiDefinitePhylogeny`](@ref) entry, enforces `A ⊙ W = 0` and adds `p * tr(W)` to the objective penalty. The penalty holds ``\\mathbf{W}`` down to ``\\boldsymbol{w}\\boldsymbol{w}^\\intercal``. It is omitted when a variance on the same weights marked `variance_flag`, [`mark_objective_variance!`](@ref), and the objective minimises the risk, [`mark_risk_minimised!`](@ref), because that variance then does the same work. A variance that is only a bound, or the inner variance of a tracking measure, keeps the penalty. Does nothing when `plgs` contains no [`SemiDefinitePhylogeny`](@ref) instances.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Phylogeny constraint(s). Accepts `nothing`, a single phylogeny, or a vector.
  - `prefix::Symbol`: The Model State namespace the rows and the penalty are registered under, `Symbol("")` for a head's own. The lifted `W` and the `variance_flag` it reads belong to the weights, [`weights_prefix`](@ref), so a programme Allocation Set's phylogeny in a leader's model reuses the leader's `W` and prefixes only its rows.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_phylogeny_constraints!(model::JuMP.Model, plgs::Option{<:PlC_VecPlC};
                                        prefix::Symbol = Symbol(""))
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    owner = weights_prefix(model, prefix)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, SemiDefinitePhylogeny)
            continue
        end
        A = pl.A
        state_set!(model, prefix, :sdp_plg_, i, JuMP.@constraint(model, sc * A ⊙ W == 0))
        if !(shared_has(model, :risk_minimised) && state_has(model, owner, :variance_flag))
            p = pl.p
            plp = state_set!(model, prefix, :sdp_plg_p_, i,
                             JuMP.@expression(model, p * LinearAlgebra.tr(W)))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add semidefinite phylogeny constraints for factor risk contribution to the JuMP optimisation model.

Iterates over `plgs` and, for each [`SemiDefinitePhylogeny`](@ref) entry, enforces `A ⊙ frc_W = 0` and adds `p * tr(frc_W)` to the objective penalty, by the rule of [`set_sdp_phylogeny_constraints!`](@ref). Does nothing when `plgs` contains no [`SemiDefinitePhylogeny`](@ref) instances.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: Phylogeny constraint(s). Accepts `nothing`, a single phylogeny, or a vector.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_frc_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
"""
function set_sdp_frc_phylogeny_constraints!(model::JuMP.Model,
                                            plgs::Option{<:PlCE_PlC_VecPlCE_PlC})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = get_constraint_scale(model)
    W = set_sdp_frc_constraints!(model)
    for (i, pl) in enumerate(plgs)
        if !isa(pl, SemiDefinitePhylogeny)
            continue
        end
        A = pl.A
        state_set!(model, Symbol(""), :frc_sdp_plg_, i,
                   JuMP.@constraint(model, sc * A ⊙ W == 0))
        if !(shared_has(model, :risk_minimised) && shared_has(model, :variance_flag))
            p = pl.p
            plp = state_set!(model, Symbol(""), :frc_sdp_plg_p_, i,
                             JuMP.@expression(model, p * LinearAlgebra.tr(W)))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
