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

The mark goes to the namespace that owns the weights, [`weights_prefix`](@ref). A variance that is only a bound marks nothing. Under [`MaximumRatio`](@ref) the same condition makes a lifted variance state its term on the lift of [`set_sdp_objective_lift!`](@ref), [`sdp_objective_lift_flag`](@ref), and the phylogeny then reads that lift. The inner variance of a [`RiskTrackingRiskMeasure`](@ref) or a [`RiskTrackingError`](@ref) marks nothing either, because the constructors of both clear the inner `rke`. The function reads one variance at a time. [`mark_risk_minimised!`](@ref) reads the role of the objective, and [`set_sdp_phylogeny_constraints!`](@ref) omits its penalty only when both marks are present.

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
    set_sdp_lift_corner!(model::JuMP.Model, obj::ObjectiveFunction, pr::AbstractPriorResult)

Register `sdp_corner`, the corner that a lifted variance takes under [`MaximumRatio`](@ref) when its `corner` is `nothing`.

The head calls it after [`set_maximum_ratio_factor_variables!`](@ref) and before the risk builders. Every other objective registers nothing, and a variance then states its objective term on the lift with the corner ``k``, because ``k`` is the constant ``1``. So the entry also tells a builder that the objective is a ratio. [`set_sdp_objective_lift!`](@ref) reads it.

The corner sets the scale of the lifted matrix ``\\mathbf{W}_c``, which is ``k^{2} / c`` times a matrix of rank one, and not the weights. The best corner is the ``k`` of the solution. The return form fixes ``k = \\mathrm{ohf} / (\\mathrm{ret} - r_f)``, so the function divides `ohf` by the size of a return. The default `ohf` is that size, so the corner is ``1`` when `obj.ohf` is `nothing`.

# Mathematical definition

```math
\\begin{align}
c &= \\begin{cases} 1 & \\text{if } \\texttt{obj.ohf} = \\texttt{nothing}\\,, \\\\ \\dfrac{\\mathrm{ohf}}{\\min\\left(10^{3},\\, \\max\\left(10^{-3},\\, \\frac{1}{N} \\sum_{j=1}^{N} \\lvert \\mu_j \\rvert\\right)\\right)} & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``c``: The derived corner.
  - $(math_dict[:ohf_ratio]) Here it is the stated `obj.ohf`.
  - ``\\mu_j``: Entry ``j`` of the expected returns of the prior, `pr.mu`. The normalisation of [`set_maximum_ratio_normalisation!`](@ref) reads the characteristic of the return terms, which the model resolves after the risk builders. The prior's vector is its value when a single term states no vector of its own.
  - $(math_dict[:N])

# Arguments

  - $(arg_dict[:model])
  - `obj`: The objective. Only a [`MaximumRatio`](@ref) registers the corner.
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`set_sdp_objective_lift!`](@ref)
  - [`SquareRootRatio`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
"""
function set_sdp_lift_corner!(::JuMP.Model, ::ObjectiveFunction, ::AbstractPriorResult)
    return nothing
end
function set_sdp_lift_corner!(model::JuMP.Model, obj::MaximumRatio, pr::AbstractPriorResult)
    c = if isnothing(obj.ohf)
        one(eltype(pr.mu))
    else
        obj.ohf / min(1e3, max(1e-3, Statistics.mean(abs, pr.mu)))
    end
    shared_set!(model, :sdp_corner, c)
    return nothing
end
"""
    sdp_objective_lift_flag(model::JuMP.Model, ratio_alg::RatioFormulation,
                            settings::RiskMeasureSettings)

Return `true` when a lifted variance states its objective term on the lift with a constant corner, [`set_sdp_objective_lift!`](@ref).

The term needs that lift only when it is a positive term of the objective of a [`MaximumRatio`](@ref): `ratio_alg` is [`SquareRootRatio`](@ref), the head registered `sdp_corner` with [`set_sdp_lift_corner!`](@ref), `settings.rke` is `true` and `settings.scale > 0`. Under every other objective ``k`` is the constant ``1``, and the two lifts are one matrix. A bound needs degree one, so it reads the lift with the corner ``k``. A term with a negative scale is maximised, and only a bound holds a lifted matrix from above. [`LinearRatio`](@ref) gives `false`.

# Arguments

  - $(arg_dict[:model])
  - `ratio_alg`: The `ratio_alg` of the measure.
  - `settings`: The settings of the measure.

# Returns

  - `flag::Bool`.

# Related

  - [`set_sdp_objective_lift!`](@ref)
  - [`mark_objective_variance!`](@ref)
"""
function sdp_objective_lift_flag(::JuMP.Model, ::LinearRatio, ::RiskMeasureSettings)
    return false
end
function sdp_objective_lift_flag(model::JuMP.Model, ::SquareRootRatio,
                                 settings::RiskMeasureSettings)
    return shared_has(model, :sdp_corner) &&
           settings.rke &&
           settings.scale > zero(settings.scale)
end
"""
    assert_one_lift_corner(c_lift::Number, c::Number)

Check that a measure asks the lift for the corner that the lift already holds.

The lift of the objective belongs to one weight vector, as ``\\mathbf{W}`` does, so every measure on those weights reads one matrix and one corner. The corner does not change the weights, so the measures need not agree on it for the answer. The function refuses a disagreement, because the model can honour only one corner and must not drop the other in silence.

# Validation

  - `c_lift == c`. Otherwise the function raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`set_sdp_objective_lift!`](@ref)
  - [`set_sdp_frc_objective_lift!`](@ref)
"""
function assert_one_lift_corner(c_lift::Number, c::Number)
    @argcheck(c_lift == c,
              ArgumentError("two lifted variances on one weight vector ask for different corners of the lift of the objective, $c_lift and $c. The measures share the lift, and the lift has one corner. The corner changes the conditioning of the solve, not the weights, so state one `corner` on every measure, or leave each `nothing` to take the corner that the model derives."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the lift with a constant corner that the objective term of a lifted variance reads under [`MaximumRatio`](@ref), and return its lifted matrix.

[`set_sdp_constraints!`](@ref) states ``\\mathbf{W} \\succeq \\boldsymbol{y}\\boldsymbol{y}^\\intercal / k``. A risk term that is linear in ``\\mathbf{W}`` then has degree one in ``(\\boldsymbol{y}, k)``, so [`MaximumRatio`](@ref) divides the excess return by the risk itself, not by its square root. This lift puts the constant ``c`` in the corner, so a term that is linear in ``\\mathbf{W}_c`` has degree two and gives the ratio of the square root, as the second-order cone variance does. The bound of the measure stays on ``\\mathbf{W}``, because the row ``R \\leq u\\,k`` needs degree one.

The lift belongs to the namespace that owns the weights under `prefix`, [`weights_prefix`](@ref), as ``\\mathbf{W}`` does. A second call for the same owner returns the first call's matrix and adds nothing. A semidefinite phylogeny on the same weights reads ``\\mathbf{W}_c`` when the objective minimises the risk, [`set_sdp_phylogeny_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathbf{M}_c &= \\begin{bmatrix} \\mathbf{W}_c & \\boldsymbol{y} \\\\ \\boldsymbol{y}^\\intercal & c \\end{bmatrix} \\succeq 0 \\quad\\Leftrightarrow\\quad c\\,\\mathbf{W}_c \\succeq \\boldsymbol{y}\\boldsymbol{y}^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}_c``: The bordered ``(N + 1) \\times (N + 1)`` matrix.
  - ``\\mathbf{W}_c``: The symmetric ``N \\times N`` lifted matrix of the objective.
  - ``c``: The corner, `corner` when it is a number and the `sdp_corner` of [`set_sdp_lift_corner!`](@ref) when it is `nothing`.
  - $(math_dict[:y_homog])

A term of the objective reads ``\\mathbf{W}_c``, which is ``\\boldsymbol{y}\\boldsymbol{y}^\\intercal / c`` at rank one. So the term is ``1 / c`` times the quadratic form in ``\\boldsymbol{y}``. A constant factor does not move the maximiser of the ratio, and a corner near ``k`` keeps the term near the scale of the risk at ``k = 1``. A term that is ``\\boldsymbol{y}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{y}`` itself is of the order of ``k^{2}``, and a small `ohf` puts it below the tolerance of the solver.

# JuMP formulation

## Variables

  - `w`: read, the weights under the owner's namespace.
  - `lift_W`: created, ``\\mathbf{W}_c``, a symmetric ``N \\times N`` matrix.

## Expressions

  - `lift_M`: ``\\mathbf{M}_c``.

## Constraints

  - `lift_M_PSD`: ``s_c \\mathbf{M}_c \\in \\mathcal{S}_{+}^{N + 1}``.

Where:

  - $(math_dict[:sc_scale])
  - ``\\mathcal{S}_{+}^{N + 1}``: The cone of positive semidefinite ``(N + 1) \\times (N + 1)`` matrices.
  - Each name carries the owner's namespace as a prefix. It is empty in a head. `lift_corner` records ``c``.

# Arguments

  - $(arg_dict[:model])
  - `corner`: The `corner` of the measure, a number or `nothing`.

# Keyword arguments

  - `prefix::Symbol`: The Model State namespace of the build. The entries go to its owner, [`weights_prefix`](@ref).

# Validation

  - Every call for one owner resolves to one corner, [`assert_one_lift_corner`](@ref).

# Returns

  - `W`: ``\\mathbf{W}_c``, a symmetric ``N \\times N`` matrix of JuMP variables.

# Related

  - [`SquareRootRatio`](@ref)
  - [`sdp_objective_lift_flag`](@ref)
  - [`set_sdp_lift_corner!`](@ref)
  - [`set_sdp_constraints!`](@ref)
  - [`set_sdp_frc_objective_lift!`](@ref)
"""
function set_sdp_objective_lift!(model::JuMP.Model, corner::Option{<:Number};
                                 prefix::Symbol = Symbol(""))
    owner = weights_prefix(model, prefix)
    lp = nested_prefix(owner, :lift_)
    c = isnothing(corner) ? shared_get(model, :sdp_corner) : corner
    W = state_build!(model, lp, :W) do
        w = get_w(model, owner)
        sc = get_constraint_scale(model)
        N = length(w)
        W = JuMP.@variable(model, [1:N, 1:N], Symmetric)
        M = state_set!(model, lp, :M,
                       JuMP.@expression(model, hcat(vcat(W, transpose(w)), vcat(w, c))))
        state_set!(model, lp, :M_PSD, JuMP.@constraint(model, sc * M in JuMP.PSDCone()))
        state_set!(model, lp, :corner, c)
        return W
    end
    assert_one_lift_corner(state_get(model, lp, :corner), c)
    return W
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the lift with a constant corner of the decision vector of a [`FactorRiskContribution`](@ref), and return its lifted matrix.

It is [`set_sdp_objective_lift!`](@ref) on the decision vector ``\\boldsymbol{z}`` of [`set_sdp_frc_constraints!`](@ref). A second call returns the first call's matrix and adds nothing. The factor phylogeny reads the leading ``N_f \\times N_f`` block of ``\\mathbf{W}_c`` when the objective minimises the risk, [`set_sdp_frc_phylogeny_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathbf{M}_c &= \\begin{bmatrix} \\mathbf{W}_c & \\boldsymbol{z} \\\\ \\boldsymbol{z}^\\intercal & c \\end{bmatrix} \\succeq 0 \\quad\\Leftrightarrow\\quad c\\,\\mathbf{W}_c \\succeq \\boldsymbol{z}\\boldsymbol{z}^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}_c``: The bordered ``(N_z + 1) \\times (N_z + 1)`` matrix.
  - ``\\mathbf{W}_c``: The symmetric ``N_z \\times N_z`` lifted matrix of the objective.
  - ``\\boldsymbol{z}``: The decision vector, the factor weights, followed by the off-factor weights when `flag = true`.
  - ``c``: The corner, `corner` when it is a number and the `sdp_corner` of [`set_sdp_lift_corner!`](@ref) when it is `nothing`.

# JuMP formulation

## Variables

  - `w1`: read, the factor weights. `w2`: read when present, the off-factor weights.
  - `lift_frc_W`: created, ``\\mathbf{W}_c``, a symmetric ``N_z \\times N_z`` matrix.

## Expressions

  - `lift_frc_M`: ``\\mathbf{M}_c``.

## Constraints

  - `lift_frc_M_PSD`: ``s_c \\mathbf{M}_c \\in \\mathcal{S}_{+}^{N_z + 1}``.

Where:

  - $(math_dict[:sc_scale])
  - ``\\mathcal{S}_{+}^{N_z + 1}``: The cone of positive semidefinite ``(N_z + 1) \\times (N_z + 1)`` matrices.
  - `lift_frc_corner` records ``c``.

# Arguments

  - $(arg_dict[:model])
  - `corner`: The `corner` of the measure, a number or `nothing`.

# Validation

  - Every call resolves to one corner, [`assert_one_lift_corner`](@ref).

# Returns

  - `W`: ``\\mathbf{W}_c``, a symmetric ``N_z \\times N_z`` matrix of JuMP variables.

# Related

  - [`set_sdp_objective_lift!`](@ref)
  - [`set_sdp_frc_constraints!`](@ref)
  - [`SquareRootRatio`](@ref)
"""
function set_sdp_frc_objective_lift!(model::JuMP.Model, corner::Option{<:Number})
    lp = nested_prefix(Symbol(""), :lift_)
    c = isnothing(corner) ? shared_get(model, :sdp_corner) : corner
    W = state_build!(model, lp, :frc_W) do
        w1 = shared_get(model, :w1)
        z = shared_has(model, :w2) ? vcat(w1, shared_get(model, :w2)) : w1
        sc = get_constraint_scale(model)
        Nz = length(z)
        W = JuMP.@variable(model, [1:Nz, 1:Nz], Symmetric)
        M = state_set!(model, lp, :frc_M,
                       JuMP.@expression(model, hcat(vcat(W, transpose(z)), vcat(z, c))))
        state_set!(model, lp, :frc_M_PSD, JuMP.@constraint(model, sc * M in JuMP.PSDCone()))
        state_set!(model, lp, :frc_corner, c)
        return W
    end
    assert_one_lift_corner(state_get(model, lp, :frc_corner), c)
    return W
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the rows of each [`SemiDefinitePhylogeny`](@ref) in `plgs` on the lifted matrix of the weights.

A semidefinite phylogeny asks that no two linked assets both carry weight. The rows state this on ``\\mathbf{W}``, and the penalty ``p\\,\\mathrm{tr}(\\mathbf{W})`` holds ``\\mathbf{W}`` down to ``\\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``. The function omits the penalty when a variance on the same weights carries the mark of [`mark_objective_variance!`](@ref) and the objective carries the mark of [`mark_risk_minimised!`](@ref), because that variance then puts the same price on the growth of ``\\mathbf{W}``. A variance that is only a bound, or the inner variance of a tracking measure, keeps the penalty. The function does nothing when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).

Under [`MaximumRatio`](@ref) a variance in the objective can state its term on the lift of [`set_sdp_objective_lift!`](@ref), ``\\mathbf{W}_c``, and not on ``\\mathbf{W}``. Nothing then holds ``\\mathbf{W}`` down, and rows on it bind nothing. So when the objective minimises the risk and that lift exists, the rows read ``\\mathbf{W}_c`` and the function adds no penalty. The lift exists only for a term that also marks `variance_flag`. The rows are homogeneous in the matrix, so they hold on either lift.

# Algorithm

 1. Return when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).
 2. Find `owner`, the namespace that owns the weights, with [`weights_prefix`](@ref).
 3. When the model carries `risk_minimised` and the lift `lift_W` of `owner`, take that lift as `W`, and mark it held down.
 4. Otherwise get the lifted matrix `W` from [`set_sdp_constraints!`](@ref), which builds it once for the owner of the weights. It is held down when the model carries both `risk_minimised` and the `variance_flag` of `owner`.
 5. For each semidefinite entry at position `i` of `plgs`, register the row `sdp_plg_<i>`. Skip an entry of another kind. A single entry has position 1.
 6. When `W` is not held down, register the penalty `sdp_plg_p_<i>` and add it to the objective penalty with [`add_to_objective_penalty!`](@ref).

# JuMP formulation

## Variables

  - `W`: read, the lifted matrix of [`set_sdp_constraints!`](@ref) under the owner of the weights, or the lift `lift_W` of [`set_sdp_objective_lift!`](@ref) as step 3 states.

## Expressions

  - `sdp_plg_p_<i>`: ``p\\,\\mathrm{tr}(\\mathbf{W})``, added to the objective penalty. It is absent when `W` is held down.

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
    owner = weights_prefix(model, prefix)
    lp = nested_prefix(owner, :lift_)
    minimised = shared_has(model, :risk_minimised)
    # The lift of the objective exists only for a positive term of a `MaximumRatio`, and
    # that term marks `variance_flag`. It holds the lift down when the objective minimises
    # the risk, so the rows read it and need no penalty (#1676).
    held, W = if minimised && state_has(model, lp, :W)
        true, state_get(model, lp, :W)
    else
        minimised && state_has(model, owner, :variance_flag),
        set_sdp_constraints!(model; prefix = prefix)
    end
    for (i, pl) in enumerate(plgs)
        if !isa(pl, SemiDefinitePhylogeny)
            continue
        end
        A = pl.A
        state_set!(model, prefix, :sdp_plg_, i, JuMP.@constraint(model, sc * A ⊙ W == 0))
        if !held
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

It is [`set_sdp_phylogeny_constraints!`](@ref) on the factor weights, with the same rule for the penalty and for the lift of the objective, `lift_frc_W` of [`set_sdp_frc_objective_lift!`](@ref). The head calls it after [`assemble_jump_model!`](@ref), so the marks of the variance and of the objective are present when the function reads them. It reads them in the head's own namespace. The function does nothing when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).

# Algorithm

 1. Return when `plgs` holds no [`SemiDefinitePhylogeny`](@ref).
 2. When the model carries `risk_minimised` and the lift `lift_frc_W`, take the leading block of that lift over the factor weights, and mark it held down.
 3. Otherwise get the lifted matrix `frc_W` from [`set_sdp_frc_constraints!`](@ref), and take its leading block over the factor weights. It is held down when the model carries both `risk_minimised` and `variance_flag`.
 4. For each semidefinite entry at position `i` of `plgs`, register the row `frc_sdp_plg_<i>`. Skip an entry of another kind.
 5. When the block is not held down, register the penalty `frc_sdp_plg_p_<i>` and add it to the objective penalty with [`add_to_objective_penalty!`](@ref).

# JuMP formulation

## Variables

  - `frc_W`: read, the lifted matrix of [`set_sdp_frc_constraints!`](@ref), or `lift_frc_W` of [`set_sdp_frc_objective_lift!`](@ref) as step 2 states.

## Expressions

  - `frc_sdp_plg_p_<i>`: ``p\\,\\mathrm{tr}(\\mathbf{W}_f)``, added to the objective penalty. It is absent when the block is held down.

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
    lp = nested_prefix(Symbol(""), :lift_)
    minimised = shared_has(model, :risk_minimised)
    held, W = if minimised && state_has(model, lp, :frc_W)
        true, state_get(model, lp, :frc_W)[1:Nf, 1:Nf]
    else
        minimised && shared_has(model, :variance_flag),
        set_sdp_frc_constraints!(model)[1:Nf, 1:Nf]
    end
    for (i, pl) in enumerate(plgs)
        if !isa(pl, SemiDefinitePhylogeny)
            continue
        end
        A = pl.A
        state_set!(model, Symbol(""), :frc_sdp_plg_, i,
                   JuMP.@constraint(model, sc * A ⊙ W == 0))
        if !held
            p = pl.p
            plp = state_set!(model, Symbol(""), :frc_sdp_plg_p_, i,
                             JuMP.@expression(model, p * LinearAlgebra.tr(W)))
            add_to_objective_penalty!(model, plp)
        end
    end
    return nothing
end
