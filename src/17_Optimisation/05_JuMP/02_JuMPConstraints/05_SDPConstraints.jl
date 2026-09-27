"""
    weights_prefix(model::JuMP.Model, prefix::Symbol)

Returns the Model State namespace that owns the weights registered under `prefix`.

One lifted matrix ``\\mathbf{W}`` belongs to one weight vector, and so do the marks that describe the measures built on it, `variance_flag` and `rc_variance`. A build that registers weights it did not make also registers `w_owner`, the namespace that owns them. A programme Allocation Set in a leader's model registers the model's own `w` under `:aset_` with the owner `Symbol("")`. A build under [`DependentVariableTracking`](@ref) registers the weights of its enclosing build under its tracking prefix, with the owner of those weights. So the set's ceiling, the inner measures of the tracking build and the head's measures read one ``\\mathbf{W}`` and one set of marks. A prefix without `w_owner` owns itself. An [`IndependentVariableTracking`](@ref) prefix is one of these, because the benchmark shifts its weights.

# Arguments

  - $(arg_dict[:model])
  - `prefix`: The Model State namespace that the weights are registered under.

# Returns

  - `owner::Symbol`: The `w_owner` entry under `prefix` when there is one, and `prefix` otherwise.

# Related

  - [`set_sdp_constraints!`](@ref)
  - [`get_w`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`RiskTrackingRiskMeasure`](@ref)
  - [`RiskTrackingError`](@ref)
"""
function weights_prefix(model::JuMP.Model, prefix::Symbol)
    return state_has(model, prefix, :w_owner) ? state_get(model, prefix, :w_owner) : prefix
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Mark `variance_flag` on the weights under `prefix` when the variance is a positive term of the objective's risk.

The mark goes to the namespace that owns the weights, [`weights_prefix`](@ref). A variance that is only a bound marks nothing. The inner variance of a [`RiskTrackingRiskMeasure`](@ref) or a [`RiskTrackingError`](@ref) marks nothing either, because the constructors of both clear the inner `rke`. The function reads one variance at a time. [`mark_risk_minimised!`](@ref) reads the role of the objective, and [`set_sdp_phylogeny_constraints!`](@ref) omits its penalty only when both marks are present.

# Mathematical definition

The PSD cone of [`set_sdp_constraints!`](@ref) bounds ``\\mathbf{W}`` from below only, and the phylogeny rows leave its diagonal free:

```math
\\begin{align}
\\mathbf{W} \\succeq \\frac{\\boldsymbol{w}\\boldsymbol{w}^\\intercal}{k}\\,, \\qquad \\mathbf{A} \\odot \\mathbf{W} = \\mathbf{0}\\,, \\qquad A_{ii} = 0\\,.
\\end{align}
```

So the step ``\\mathbf{W} + t\\,\\boldsymbol{e}_i\\boldsymbol{e}_i^\\intercal`` with ``t \\geq 0`` satisfies both, and it changes a variance term by

```math
\\begin{align}
s\\,\\mathrm{tr}\\left(\\boldsymbol{\\Sigma}\\left(\\mathbf{W} + t\\,\\boldsymbol{e}_i\\boldsymbol{e}_i^\\intercal\\right)\\right) - s\\,\\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W}) = s\\,t\\,\\Sigma_{ii}\\,.
\\end{align}
```

A variance that the objective minimises with ``s > 0`` therefore puts a price on the growth of ``\\mathbf{W}``, as the phylogeny's penalty ``p\\,\\mathrm{tr}(\\mathbf{W})`` does. A bound puts no price on it. The inner variance of a tracking measure enters the model as ``|\\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W}) - r_b k|``, which falls as ``\\mathbf{W}`` grows while the portfolio's variance is below the benchmark's.

Where:

  - ``\\mathbf{W}``: The symmetric ``N \\times N`` lifted matrix of the weights.
  - ``\\mathbf{A}``: The relatedness matrix of a semidefinite phylogeny, symmetric with a zero diagonal.
  - ``\\boldsymbol{e}_i``: The ``i``-th unit vector.
  - ``t``: The size of the step.
  - ``s``: The scale of the variance in the objective, `settings.scale`.
  - ``\\boldsymbol{\\Sigma}``: The covariance matrix of the variance.
  - ``p``: The penalty of the semidefinite phylogeny.
  - ``r_b``: The variance of the benchmark.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

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

[`assemble_jump_model!`](@ref) calls it after the return constraints, when a [`MaximumRatio`](@ref) has chosen its form. A variance with the mark of [`mark_objective_variance!`](@ref) holds the lifted matrix down only when the objective minimises it.

  - [`MinimumRisk`](@ref) minimises the risk.
  - [`MaximumUtility`](@ref) minimises it when `l > 0`.
  - [`MaximumRatio`](@ref) minimises it in the return form. The risk form registers `sr_risk`, and the risk is then a bound.
  - Every other objective, such as [`MaximumReturn`](@ref), leaves the risk out of the objective.

# Arguments

  - $(arg_dict[:model])
  - `obj`: The objective that the model builds.

# Returns

  - `nothing`.

# Related

  - [`mark_objective_variance!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`set_sdp_frc_phylogeny_constraints!`](@ref)
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

Add the positive semidefinite (PSD) cone that lifts the portfolio weights to a matrix variable.

The matrix belongs to the namespace that owns the weights under `prefix`, [`weights_prefix`](@ref), so weights registered twice have one ``\\mathbf{W}``. A second call for the same owner returns the first call's matrix and adds nothing.

# Mathematical definition

```math
\\begin{align}
\\mathbf{M} &= \\begin{bmatrix} \\mathbf{W} & \\boldsymbol{w} \\\\ \\boldsymbol{w}^\\intercal & k \\end{bmatrix} \\succeq 0 \\\\
&\\quad\\Leftrightarrow\\quad \\mathbf{W} \\succeq \\frac{\\boldsymbol{w}\\boldsymbol{w}^\\intercal}{k} \\quad \\text{for } k > 0\\,.
\\end{align}
```

The equivalence is the Schur complement of ``k``. At ``k = 0`` the cone forces ``\\boldsymbol{w} = \\mathbf{0}``.

Where:

  - ``\\mathbf{M}``: The bordered ``(N + 1) \\times (N + 1)`` matrix.
  - ``\\mathbf{W}``: The symmetric ``N \\times N`` lifted matrix of the weights.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget]) It is ``1`` under a unit budget, [`effective_k`](@ref).

# JuMP formulation

## Variables

  - `w`: read, the weights under the owner's namespace.
  - `k`: read through [`effective_k`](@ref).
  - `W`: created, a symmetric ``N \\times N`` matrix, registered as `W` under the owner's namespace.

## Expressions

  - `M`: ``\\mathbf{M}``, the bordered matrix above.

## Constraints

  - `M_PSD`: ``s_c \\mathbf{M} \\in \\mathcal{S}_{+}^{N + 1}``.

Where:

  - $(math_dict[:sc_scale])
  - ``\\mathcal{S}_{+}^{N + 1}``: The cone of positive semidefinite ``(N + 1) \\times (N + 1)`` matrices.
  - Each name carries the owner's namespace as a prefix. It is empty in a head.

# Arguments

  - $(arg_dict[:model])
  - `prefix`: The Model State namespace of the build. The entries go to its owner, [`weights_prefix`](@ref).

# Returns

  - `W`: The lifted matrix, a symmetric ``N \\times N`` matrix of JuMP variables.

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

Add the positive semidefinite (PSD) cone that lifts the decision vector of a [`FactorRiskContribution`](@ref) to a matrix variable.

It is [`set_sdp_constraints!`](@ref) on the decision vector ``\\boldsymbol{z}``. That vector is the factor weights ``\\boldsymbol{w}_1`` when `flag = false`, and the factor weights followed by the off-factor weights ``\\boldsymbol{w}_2`` when `flag = true`. A second call returns the first call's matrix and adds nothing.

The lift covers the whole decision vector because the asset weights are ``\\mathbf{P} \\boldsymbol{z}``, with ``\\mathbf{P}`` the basis of [`set_factor_risk_contribution_constraints!`](@ref). So ``\\mathrm{tr}(\\mathbf{P}^\\intercal \\mathbf{\\Sigma} \\mathbf{P} \\mathbf{W}_z)`` is the variance of the asset weights at rank one. A lift of ``\\boldsymbol{w}_1`` alone gives the variance of ``\\mathbf{B}_1 \\boldsymbol{w}_1`` only, and omits the off-factor weights. The leading ``N_f \\times N_f`` block of ``\\mathbf{W}_z`` is a lift of ``\\boldsymbol{w}_1`` by itself, because a principal submatrix of a PSD matrix is PSD. The factor phylogeny reads that block.

# Mathematical definition

```math
\\begin{align}
\\mathbf{M}_z &= \\begin{bmatrix} \\mathbf{W}_z & \\boldsymbol{z} \\\\ \\boldsymbol{z}^\\intercal & k \\end{bmatrix} \\succeq 0 \\\\
&\\quad\\Leftrightarrow\\quad \\mathbf{W}_z \\succeq \\frac{\\boldsymbol{z}\\boldsymbol{z}^\\intercal}{k} \\quad \\text{for } k > 0\\,, \\\\
\\boldsymbol{z} &= \\begin{cases} \\boldsymbol{w}_1 & \\text{if } \\texttt{flag} = \\texttt{false}\\,, \\\\ \\begin{bmatrix} \\boldsymbol{w}_1 \\\\ \\boldsymbol{w}_2 \\end{bmatrix} & \\text{if } \\texttt{flag} = \\texttt{true}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``\\mathbf{M}_z``: The bordered ``(N_z + 1) \\times (N_z + 1)`` matrix.
  - ``\\mathbf{W}_z``: The symmetric ``N_z \\times N_z`` lifted matrix of the decision vector.
  - ``\\boldsymbol{z}``: The decision vector, of length ``N_z``. ``N_z = N_f`` when `flag = false`, and ``N_z = N`` when `flag = true`.
  - $(math_dict[:w_1_factor])
  - $(math_dict[:w_2_off_factor])
  - $(math_dict[:k_budget])

# JuMP formulation

## Variables

  - `w1`: read, the factor weights.
  - `w2`: read when present, the off-factor weights.
  - `k`: read through [`get_k`](@ref).
  - `frc_W`: created, a symmetric ``N_z \\times N_z`` matrix.

## Expressions

  - `frc_M`: ``\\mathbf{M}_z``, the bordered matrix above.

## Constraints

  - `frc_M_PSD`: ``s_c \\mathbf{M}_z \\in \\mathcal{S}_{+}^{N_z + 1}``.

Where:

  - $(math_dict[:sc_scale])
  - ``\\mathcal{S}_{+}^{N_z + 1}``: The cone of positive semidefinite ``(N_z + 1) \\times (N_z + 1)`` matrices.

# Arguments

  - $(arg_dict[:model])

# Returns

  - `frc_W`: The lifted matrix, a symmetric ``N_z \\times N_z`` matrix of JuMP variables.

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
    z = shared_has(model, :w2) ? vcat(w1, shared_get(model, :w2)) : w1
    sc = get_constraint_scale(model)
    k = get_k(model)
    Nz = length(z)
    JuMP.@variable(model, frc_W[1:Nz, 1:Nz], Symmetric)
    JuMP.@expression(model, frc_M, hcat(vcat(frc_W, transpose(z)), vcat(z, k)))
    JuMP.@constraint(model, frc_M_PSD, sc * frc_M in JuMP.PSDCone())
    return frc_W
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the rows of each [`SemiDefinitePhylogeny`](@ref) in `plgs` on the lifted matrix of the weights.

A semidefinite phylogeny asks that no two linked assets both carry weight. The rows state this on ``\\mathbf{W}``, and the penalty ``p\\,\\mathrm{tr}(\\mathbf{W})`` holds ``\\mathbf{W}`` down to ``\\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``. The function omits the penalty when a variance on the same weights carries the mark of [`mark_objective_variance!`](@ref) and the objective carries the mark of [`mark_risk_minimised!`](@ref), because that variance then puts the same price on the growth of ``\\mathbf{W}``. A variance that is only a bound, or the inner variance of a tracking measure, keeps the penalty. The function does nothing when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).

# Algorithm

 1. Return when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).
 2. Get the lifted matrix `W` from [`set_sdp_constraints!`](@ref), which builds it once for the owner of the weights.
 3. Find `owner`, the namespace that owns the weights, with [`weights_prefix`](@ref).
 4. For each semidefinite entry at position `i` of `plgs`, register the row `sdp_plg_<i>`. Skip an entry of another kind. A single entry has position 1.
 5. When the model does not carry both `risk_minimised` and the `variance_flag` of `owner`, register the penalty `sdp_plg_p_<i>` and add it to the objective penalty with [`add_to_objective_penalty!`](@ref).

# JuMP formulation

## Variables

  - `W`: read, the lifted matrix of [`set_sdp_constraints!`](@ref) under the owner of the weights.

## Expressions

  - `sdp_plg_p_<i>`: ``p\\,\\mathrm{tr}(\\mathbf{W})``, added to the objective penalty. It is absent when both marks are present.

## Constraints

  - `sdp_plg_<i>`: ``s_c\\,\\mathbf{A} \\odot \\mathbf{W} = \\mathbf{0}``.

Where:

  - ``\\mathbf{W}``: The symmetric ``N \\times N`` lifted matrix of the weights, ``\\mathbf{W} \\succeq \\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``.
  - ``\\mathbf{A}``: The relatedness matrix of the entry, symmetric with a zero diagonal.
  - ``p``: The penalty of the entry.
  - $(math_dict[:i_plg]) Each name carries `prefix`.
  - $(math_dict[:sc_scale])
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

## Relaxation

$(val_dict[:relax])

  - The exact rows are ``A_{ij}\\,w_i w_j = 0``, so at most one asset of a linked pair carries weight. The rows `sdp_plg_<i>` bind ``\\mathbf{W}`` in place of ``\\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``. So the weights that they admit include every weight vector that the exact rows admit, and can hold both assets of a linked pair.
  - The rows are tight when ``\\mathbf{W} = \\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``, a matrix of rank one. The rows do not force this. The penalty `sdp_plg_p_<i>`, or a variance that the objective minimises, puts a price on the growth of ``\\mathbf{W}``. With ``p = 0`` and no such variance, nothing holds ``\\mathbf{W}`` down.
  - The penalty is at least ``p \\lVert \\boldsymbol{w} \\rVert_2^2 / k``, so it also spreads the weights. A large ``p`` can therefore leave ``\\mathbf{W}`` above rank one and put weight on both assets of a linked pair.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: The phylogeny constraints, `nothing`, one result, or a vector of results.
  - `prefix::Symbol`: The Model State namespace of the rows and the penalty, `Symbol("")` for a head's own. The lifted matrix and the `variance_flag` that the function reads belong to the weights, [`weights_prefix`](@ref). So a programme Allocation Set's phylogeny in a leader's model reads the leader's ``\\mathbf{W}``, and only its rows carry the prefix.

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

Add the rows of each [`SemiDefinitePhylogeny`](@ref) in `plgs` on the lifted matrix of the factor weights of a [`FactorRiskContribution`](@ref).

It is [`set_sdp_phylogeny_constraints!`](@ref) on the factor weights, with the same rule for the penalty. The head calls it after [`assemble_jump_model!`](@ref), so the marks of the variance and of the objective are present when the function reads them. It reads them in the head's own namespace. The function does nothing when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).

# Algorithm

 1. Return when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).
 2. Get the lifted matrix `frc_W` from [`set_sdp_frc_constraints!`](@ref), and take its leading block over the factor weights.
 3. For each semidefinite entry at position `i` of `plgs`, register the row `frc_sdp_plg_<i>`. Skip an entry of another kind.
 4. When the model does not carry both `risk_minimised` and `variance_flag`, register the penalty `frc_sdp_plg_p_<i>` and add it to the objective penalty with [`add_to_objective_penalty!`](@ref).

# JuMP formulation

## Variables

  - `frc_W`: read, the lifted matrix of [`set_sdp_frc_constraints!`](@ref).

## Expressions

  - `frc_sdp_plg_p_<i>`: ``p\\,\\mathrm{tr}(\\mathbf{W}_f)``, added to the objective penalty. It is absent when both marks are present.

## Constraints

  - `frc_sdp_plg_<i>`: ``s_c\\,\\mathbf{A} \\odot \\mathbf{W}_f = \\mathbf{0}``.

Where:

  - ``\\mathbf{W}_f``: The leading ``N_f \\times N_f`` block of `frc_W`, the lifted matrix of the factor weights, ``\\mathbf{W}_f \\succeq \\boldsymbol{w}_1\\boldsymbol{w}_1^\\intercal / k``. It is the whole of `frc_W` when `flag = false`.
  - $(math_dict[:w_1_factor])
  - ``\\mathbf{A}``: The relatedness matrix of the entry over the factors, symmetric with a zero diagonal.
  - ``p``: The penalty of the entry.
  - $(math_dict[:i_plg])
  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])

## Relaxation

$(val_dict[:relax])

  - The rows `frc_sdp_plg_<i>` bind ``\\mathbf{W}_f`` in place of ``\\boldsymbol{w}_1\\boldsymbol{w}_1^\\intercal / k``, as the rows of [`set_sdp_phylogeny_constraints!`](@ref) bind ``\\mathbf{W}``. So the factor weights that they admit can hold both factors of a linked pair.
  - The rows are tight when ``\\mathbf{W}_f = \\boldsymbol{w}_1\\boldsymbol{w}_1^\\intercal / k``. The penalty `frc_sdp_plg_p_<i>`, or a variance that the objective minimises, holds ``\\mathbf{W}_f`` down, and it spreads the factor weights as the asset penalty spreads the weights.

# Arguments

  - $(arg_dict[:model])
  - `plgs`: The phylogeny constraints on the factors, `nothing`, one result, or a vector of results.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_frc_constraints!`](@ref)
  - [`set_sdp_phylogeny_constraints!`](@ref)
  - [`SemiDefinitePhylogeny`](@ref)
  - [`FactorRiskContribution`](@ref)
"""
function set_sdp_frc_phylogeny_constraints!(model::JuMP.Model,
                                            plgs::Option{<:PlCE_PlC_VecPlCE_PlC})
    if !(isa(plgs, SemiDefinitePhylogeny) ||
         isa(plgs, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), plgs))
        return nothing
    end
    sc = get_constraint_scale(model)
    # The lift covers the off-factor weights under `flag = true`. The rows and the penalty
    # read its factor block alone.
    Nf = length(shared_get(model, :w1))
    W = set_sdp_frc_constraints!(model)[1:Nf, 1:Nf]
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
