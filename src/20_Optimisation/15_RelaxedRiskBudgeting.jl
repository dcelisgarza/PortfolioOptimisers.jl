"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for relaxed risk budgeting algorithm variants.

# Related Types

  - [`BasicRelaxedRiskBudgeting`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
  - [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)
"""
abstract type RelaxedRiskBudgetingAlgorithm <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Basic Relaxed Risk Budgeting formulation.

Uses the basic Second Order Cone (SOC) relaxation of the risk budgeting problem without additional regularisation.

# Related Types

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
"""
struct BasicRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Regularised Relaxed Risk Budgeting formulation.

Extends the basic SOC formulation with a regularisation term to improve numerical stability.

# Related Types

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`BasicRelaxedRiskBudgeting`](@ref)
  - [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)
"""
struct RegularisedRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Regularised and penalised Relaxed Risk Budgeting formulation.

Extends the regularised formulation with a penalty on deviations from target risk budgets, controlled by parameter `p`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegularisedPenalisedRelaxedRiskBudgeting(;
        p::Number = 1.0
    ) -> RegularisedPenalisedRelaxedRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(p)` and `p > 0`.

# Related Types

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
"""
@concrete struct RegularisedPenalisedRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm
    """
    $(field_dict[:p_rm])
    """
    p
    function RegularisedPenalisedRelaxedRiskBudgeting(p::Number)
        @argcheck(isfinite(p) && p > zero(p), DomainError(p, "p must be finite and > 0"))
        return new{typeof(p)}(p)
    end
end
function RegularisedPenalisedRelaxedRiskBudgeting(;
                                                  p::Number = 1.0)::RegularisedPenalisedRelaxedRiskBudgeting
    return RegularisedPenalisedRelaxedRiskBudgeting(p)
end
"""
$(DocStringExtensions.TYPEDEF)

Relaxed Risk Budgeting (RRB) portfolio optimiser.

`RelaxedRiskBudgeting` implements a relaxed formulation of the risk budgeting problem using a Second Order Cone constraint on the portfolio variance. Unlike [`RiskBudgeting`](@ref), it does not require a logarithmic or mixed-integer formulation, making it computationally more tractable.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelaxedRiskBudgeting(;
        opt::JuMPOptimiser,
        rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
        wi::Option{<:VecNum} = nothing,
        alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
        fb::Option{<:OptE_Opt} = nothing
    ) -> RelaxedRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - If `wi` is provided: `!isempty(wi)`.

# Mathematical definition

The Relaxed Risk Budgeting (RRB) formulation replaces the non-convex risk-parity constraint with a second-order cone (SOC) relaxation. Let ``\\mathbf{G}`` be the Cholesky factor of ``\\mathbf{\\Sigma}`` (so ``\\mathbf{G}^\\intercal\\mathbf{G} = \\mathbf{\\Sigma}``). Introduce auxiliary variables ``\\boldsymbol{\\zeta} = \\mathbf{\\Sigma}\\boldsymbol{w}``, ``\\psi \\geq 0``, ``\\gamma \\geq 0``:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\psi,\\gamma,\\boldsymbol{\\zeta}}{\\min} \\quad & \\psi - \\gamma\\,, \\\\
\\text{s.t.} \\quad & \\boldsymbol{\\zeta} = \\mathbf{\\Sigma}\\boldsymbol{w}\\,, \\\\
& \\begin{pmatrix} w_i + \\zeta_i \\\\ 2\\gamma\\sqrt{b_i} \\\\ w_i - \\zeta_i \\end{pmatrix} \\in \\mathcal{K}_{\\mathrm{SOC}}\\,, \\quad \\forall i\\,.
\\end{align}
```

The risk cone constraint (basic variant): ``(\\psi,\\, \\mathbf{G}\\boldsymbol{w}) \\in \\mathcal{K}_{\\mathrm{SOC}}``, i.e. ``\\psi \\geq \\|\\mathbf{G}\\boldsymbol{w}\\|_2 = \\sqrt{\\boldsymbol{w}^\\intercal\\mathbf{\\Sigma}\\boldsymbol{w}}``.

Where:

  - ``\\boldsymbol{w}``: Portfolio weight vector.
  - ``\\psi``, ``\\gamma``: Scalar auxiliary variables.
  - ``\\boldsymbol{\\zeta}``: Auxiliary vector equal to ``\\mathbf{\\Sigma}\\boldsymbol{w}``.
  - ``b_i``: Risk budget for asset ``i``.
  - ``\\mathbf{G}``: Cholesky factor of ``\\mathbf{\\Sigma}`` (so ``\\mathbf{G}^\\intercal\\mathbf{G} = \\mathbf{\\Sigma}``).
  - ``\\mathbf{\\Sigma}``: Covariance matrix.
  - ``\\mathcal{K}_{\\mathrm{SOC}}``: Second-order cone.

# Notes

Because this is a *relaxation* of the risk budgeting problem, the realised risk contributions will not adhere to the target risk budget as tightly as the exact logarithmic-barrier or mixed-integer formulations in [`RiskBudgeting`](@ref). In well-behaved problems the deviation is negligible, but in pathological cases (e.g. ill-conditioned covariance matrices or extreme budget allocations) it can be noticeable. The trade-off is that the SOC formulation is convex and composes cleanly with additional constraints, making it the friendlier choice when the risk budget is one of several objectives rather than a hard requirement. Use [`RiskBudgeting`](@ref) when strict adherence to the risk budget is essential.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
"""
@propagatable @concrete struct RelaxedRiskBudgeting <: JuMPOptimisationEstimator
    """
    $(field_dict[:opt_jmp])
    """
    @fprop opt
    """
    $(field_dict[:rba])
    """
    rba
    """
    $(field_dict[:wi])
    """
    wi
    """
    Relaxed risk budgeting algorithm variant.
    """
    alg
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function RelaxedRiskBudgeting(opt::JuMPOptimiser, rba::RiskBudgetingAlgorithm,
                                  wi::Option{<:VecNum}, alg::RelaxedRiskBudgetingAlgorithm,
                                  fb::Option{<:OptE_Opt})
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        return new{typeof(opt), typeof(rba), typeof(wi), typeof(alg), typeof(fb)}(opt, rba,
                                                                                  wi, alg,
                                                                                  fb)
    end
end
function RelaxedRiskBudgeting(; opt::JuMPOptimiser,
                              rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
                              wi::Option{<:VecNum} = nothing,
                              alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
                              fb::Option{<:OptE_Opt} = nothing)::RelaxedRiskBudgeting
    return RelaxedRiskBudgeting(opt, rba, wi, alg, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the JuMP optimiser or fallback requires previous portfolio weights.
"""
function needs_previous_weights(opt::RelaxedRiskBudgeting)
    return (needs_previous_weights(opt.opt) || needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`RelaxedRiskBudgeting`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(rrb::RelaxedRiskBudgeting, i, X::MatNum,
                       args...)::RelaxedRiskBudgeting
    X = isa(rrb.opt.pe, AbstractPriorResult) ? rrb.opt.pe.X : X
    opt = port_opt_view(rrb.opt, i, X)
    rba = port_opt_view(rrb.rba, i)
    wi = nothing_scalar_array_view(rrb.wi, i)
    return RelaxedRiskBudgeting(; opt = opt, rba = rba, wi = wi, alg = rrb.alg, fb = rrb.fb)
end
"""
    set_relaxed_risk_budgeting_alg_constraints!(alg, model, w, sigma, chol)

Add algorithm-specific second-order cone constraints for Relaxed Risk Budgeting.

Dispatches based on the RRB algorithm variant. Adds second-order cone constraints implementing the basic, regularised, or regularised-penalised RRB formulation.

# Arguments

  - `alg`: RRB algorithm ([`BasicRelaxedRiskBudgeting`](@ref), [`RegularisedRelaxedRiskBudgeting`](@ref), or [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)).
  - `model::JuMP.Model`: JuMP optimisation model.
  - `w::VecJuMPScalar`: Portfolio weight variables.
  - `sigma::MatNum`: Covariance matrix.
  - `chol::Option{<:MatNum}`: Optional pre-computed Cholesky factor.

# Returns

  - `nothing`.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`set_relaxed_risk_budgeting_constraints!`](@ref)
"""
function set_relaxed_risk_budgeting_alg_constraints!(::BasicRelaxedRiskBudgeting,
                                                     model::JuMP.Model, w::VecJuMPScalar,
                                                     sigma::MatNum,
                                                     chol::Option{<:MatNum} = nothing)
    sc = get_constraint_scale(model)
    psi = model[:psi]
    G = isnothing(chol) ? LinearAlgebra.cholesky(sigma).U : chol
    JuMP.@constraint(model, cbasic_rrp, [sc * psi; sc * G * w] in JuMP.SecondOrderCone())
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(::RegularisedRelaxedRiskBudgeting,
                                                     model::JuMP.Model, w::VecJuMPScalar,
                                                     sigma::MatNum,
                                                     chol::Option{<:MatNum} = nothing)
    sc = get_constraint_scale(model)
    psi = model[:psi]
    G = isnothing(chol) ? LinearAlgebra.cholesky(sigma).U : chol
    JuMP.@variable(model, rho >= 0)
    JuMP.@constraints(model,
                      begin
                          creg_rrp_soc_1,
                          [sc * 2 * psi;
                           sc * 2 * G * w;
                           sc * -2 * rho] in JuMP.SecondOrderCone()
                          creg_rrp_soc_2, [sc * rho; sc * G * w] in JuMP.SecondOrderCone()
                      end)
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(alg::RegularisedPenalisedRelaxedRiskBudgeting,
                                                     model::JuMP.Model, w::VecJuMPScalar,
                                                     sigma::MatNum,
                                                     chol::Option{<:MatNum} = nothing)
    sc = get_constraint_scale(model)
    psi = model[:psi]
    G = isnothing(chol) ? LinearAlgebra.cholesky(sigma).U : chol
    theta = LinearAlgebra.Diagonal(sqrt.(LinearAlgebra.diag(sigma)))
    p = alg.p
    JuMP.@variable(model, rho >= 0)
    JuMP.@constraints(model,
                      begin
                          creg_pen_rrp_soc_1,
                          [sc * 2 * psi;
                           sc * 2 * G * w;
                           sc * -2 * rho] in JuMP.SecondOrderCone()
                          creg_pen_rrp_soc_2,
                          [sc * rho;
                           sc * sqrt(p) * theta * w] in JuMP.SecondOrderCone()
                      end)
    return nothing
end
"""
    _set_relaxed_risk_budgeting_constraints!(model, ...)

Internal function to set relaxed risk budgeting constraints in the JuMP model.

Configures inequality constraints for the relaxed risk budgeting formulation, allowing small deviations from exact budget targets.

# Arguments

  - `model`: JuMP model.
  - Additional relaxed risk budgeting parameters.

# Returns

  - `nothing`.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`_set_risk_budgeting_constraints!`](@ref)
"""
function _set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                  rrb::RelaxedRiskBudgeting,
                                                  w::VecJuMPScalar, sigma::MatNum,
                                                  chol::Option{<:MatNum} = nothing)
    N = length(w)
    rkb = risk_budget_constraints(rrb.rba.rkb, rrb.rba.sets; N = N, strict = rrb.opt.strict)
    rb = rkb.val
    @argcheck(length(rb) == N, DimensionMismatch("rb ($(length(rb))) must match N ($N)"))
    sc = get_constraint_scale(model)
    JuMP.@variables(model, begin
                        psi >= 0
                        gamma >= 0
                        zeta[1:N] >= 0
                    end)
    JuMP.@expression(model, risk, psi - gamma)
    # RRB constraints.
    JuMP.@constraints(model,
                      begin
                          crrp, sc * (zeta - sigma * w) == 0
                          crrp_soc[i = 1:N],
                          [sc * (w[i] + zeta[i])
                           sc * (2 * gamma * sqrt(rb[i]))
                           sc * (w[i] - zeta[i])] in JuMP.SecondOrderCone()
                      end)
    set_relaxed_risk_budgeting_alg_constraints!(rrb.alg, model, w, sigma, chol)
    return rkb
end
"""
    set_relaxed_risk_budgeting_constraints!(model, rrb, pr, wb, args...)

Add Relaxed Risk Budgeting (RRB) constraints and weight variables to the JuMP model.

Dispatches based on the risk budgeting algorithm type. Configures weight variables, budget constraints, second-order cone constraints, and weight bounds.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `rrb::RelaxedRiskBudgeting`: RRB estimator configuration.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `wb::WeightBounds`: Weight bounds configuration.
  - `args...`: Additional arguments (e.g. returns data for factor risk budgeting).

# Returns

  - Processed risk budgeting attributes.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`set_relaxed_risk_budgeting_alg_constraints!`](@ref)
"""
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting{<:Any,
                                                                           <:FactorRiskBudgeting,
                                                                           <:Any, <:Any},
                                                 pr::AbstractPriorResult, wb::WeightBounds,
                                                 rd::ReturnsResult)
    b1, rr = set_factor_risk_contribution_constraints!(model, rrb.rba.re, rd, rrb.rba.flag,
                                                       rrb.wi)
    rkb = _set_relaxed_risk_budgeting_constraints!(model, rrb, model[:w1],
                                                   Matrix(LinearAlgebra.Symmetric(rr.L \
                                                                                  pr.sigma *
                                                                                  b1)))
    set_weight_constraints!(model, wb, rrb.opt.bgt, rrb.opt.sbgt)
    return ProcessedFactorRiskBudgetingAttributes(; rkb = rkb, b1 = b1, rr = rr)
end
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting{<:Any,
                                                                           <:AssetRiskBudgeting,
                                                                           <:Any, <:Any},
                                                 pr::AbstractPriorResult, wb::WeightBounds,
                                                 args...)
    set_w!(model, pr.X, rrb.wi)
    set_weight_constraints!(model, wb, rrb.opt.bgt, nothing, true)
    rkb = _set_relaxed_risk_budgeting_constraints!(model, rrb, get_w(model), pr.sigma,
                                                   pr.chol)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
function _optimise(rrb::RelaxedRiskBudgeting, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    attrs = processed_jump_optimiser_attributes(rrb.opt, rd; dims = dims, kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rrb.opt.sc, rrb.opt.so)
    JuMP.@expression(model, k, 1)
    prb = set_relaxed_risk_budgeting_constraints!(model, rrb, attrs.pr, attrs.wb, rd)
    assemble_jump_model!(model, rrb, rrb.opt, attrs, rd)
    set_portfolio_objective_function!(model, MinimumRisk(), attrs.ret, rrb.opt.cobj, rrb,
                                      attrs.pr)
    retcode, sol = optimise_JuMP_model!(model, rrb, eltype(attrs.pr.X))
    return RiskBudgetingResult(;
                               jr = JuMPOptimisationResult(; oe = typeof(rrb), pa = attrs,
                                                           retcode = retcode, sol = sol,
                                                           model = ifelse(save, model,
                                                                          nothing)),
                               prb = prb, fb = nothing)
end
"""
    optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> RiskBudgetingResult

Run the Relaxed Risk Budgeting portfolio optimisation.

# Arguments

  - `rrb`: The relaxed risk budgeting optimiser to use.
  - $(arg_dict[:rd]) If `isa(rrb.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`RiskBudgetingResult`](@ref)
"""
function optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(rrb, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export BasicRelaxedRiskBudgeting, RegularisedRelaxedRiskBudgeting,
       RegularisedPenalisedRelaxedRiskBudgeting, RelaxedRiskBudgeting
