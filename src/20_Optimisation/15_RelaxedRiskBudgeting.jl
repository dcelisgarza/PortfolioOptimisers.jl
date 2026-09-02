"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for relaxed risk budgeting algorithm variants.

# Related

  - [`BasicRelaxedRiskBudgeting`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
  - [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)

# References

  - $(ref_dict[:richardroncalli2019])
"""
abstract type RelaxedRiskBudgetingAlgorithm <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Bounds the risk variable by the portfolio standard deviation alone, which is the relaxation with no extra term.

# Related

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)

# References

  - $(ref_dict[:gambetakwon2020])
  - $(ref_dict[:richardroncalli2019])
"""
struct BasicRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Adds a second cone on an auxiliary scalar, which lifts the floor on the risk variable and improves numerical stability.

# Related

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`BasicRelaxedRiskBudgeting`](@ref)
  - [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)

# References

  - $(ref_dict[:richardroncalli2019])
"""
struct RegularisedRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Bounds the auxiliary scalar by the individual standard deviations rather than the portfolio one, weighted by `p`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegularisedPenalisedRelaxedRiskBudgeting(;
        p::Number = 1.0
    ) -> RegularisedPenalisedRelaxedRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(p)` and `p > 0`.

# Related

  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`RegularisedRelaxedRiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)

# References

  - $(ref_dict[:richardroncalli2019])
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
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`RelaxedRiskBudgeting`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function relaxed_risk_budgeting_td_defaults()::NamedTuple
    return (; rba = AssetRiskBudgeting())
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
        rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
        wi::TD_Option{<:VecNum} = nothing,
        alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> RelaxedRiskBudgeting

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the budgeting algorithm (and with it the risk budget), warm start and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `wi` and `fb`). The relaxation variant `alg` is formulation control and stays static.

## Validation

  - If `wi` is provided: `!isempty(wi)`.
  - `fb` schedules: `bind !== :nearest`.

# Mathematical definition

The Relaxed Risk Budgeting (RRB) formulation replaces the non-convex risk-parity constraint with a second-order cone (SOC) relaxation. Let ``\\mathbf{G}`` be the Cholesky factor of ``\\mathbf{\\Sigma}`` (so ``\\mathbf{G}^\\intercal\\mathbf{G} = \\mathbf{\\Sigma}``). Introduce auxiliary variables ``\\boldsymbol{\\zeta} = \\mathbf{\\Sigma}\\boldsymbol{w}``, ``\\psi \\geq 0``, ``\\gamma \\geq 0``:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\psi,\\gamma,\\boldsymbol{\\zeta}}{\\min} \\quad & \\psi - \\gamma\\,, \\\\
\\text{s.t.} \\quad & \\boldsymbol{\\zeta} = \\mathbf{\\Sigma}\\boldsymbol{w}\\,, \\\\
& \\begin{pmatrix} w_i + \\zeta_i \\\\ 2\\gamma\\sqrt{b_i} \\\\ w_i - \\zeta_i \\end{pmatrix} \\in \\mathcal{K}_{\\mathrm{SOC}}\\,, \\quad \\forall i\\,.
\\end{align}
```

The variant in `alg` decides the cone that bounds ``\\psi``. The three variants are versions A, B and C of the constrained risk budgeting model of Richard and Roncalli.

[`BasicRelaxedRiskBudgeting`](@ref) bounds it by the portfolio standard deviation alone:

```math
\\begin{align}
\\psi &\\geq \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2 = \\sqrt{\\boldsymbol{w}^\\intercal\\mathbf{\\Sigma}\\boldsymbol{w}}\\,.
\\end{align}
```

[`RegularisedRelaxedRiskBudgeting`](@ref) adds a scalar ``\\rho \\geq 0`` and a second cone:

```math
\\begin{align}
\\psi &\\geq \\sqrt{\\boldsymbol{w}^\\intercal\\mathbf{\\Sigma}\\boldsymbol{w} + \\rho^{2}}\\,, \\\\
\\rho &\\geq \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,.
\\end{align}
```

[`RegularisedPenalisedRelaxedRiskBudgeting`](@ref) keeps the first cone and replaces the second one, so that ``\\rho`` is bounded by the weighted individual standard deviations rather than by the portfolio one. `p` weights that term:

```math
\\begin{align}
\\rho &\\geq \\sqrt{p} \\, \\lVert \\mathbf{\\Theta}\\boldsymbol{w} \\rVert_2\\,, \\quad \\mathbf{\\Theta} = \\mathrm{diag}\\left(\\sqrt{\\mathrm{diag}(\\mathbf{\\Sigma})}\\right)\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: Portfolio weight vector.
  - ``\\psi``: Average risk of the portfolio.
  - ``\\gamma``: Lower bound of the risk contribution of every asset.
  - ``\\boldsymbol{\\zeta}``: Auxiliary vector equal to ``\\mathbf{\\Sigma}\\boldsymbol{w}``.
  - ``b_i``: Risk budget for asset ``i``.
  - ``\\mathbf{G}``: Cholesky factor of ``\\mathbf{\\Sigma}`` (so ``\\mathbf{G}^\\intercal\\mathbf{G} = \\mathbf{\\Sigma}``).
  - ``\\mathbf{\\Sigma}``: Covariance matrix.
  - ``\\mathbf{\\Theta}``: Diagonal matrix of the individual standard deviations.
  - ``\\rho``: Scalar auxiliary variable of the two regularised variants.
  - ``p``: Penalty weight of [`RegularisedPenalisedRelaxedRiskBudgeting`](@ref).
  - ``\\mathcal{K}_{\\mathrm{SOC}}``: Second-order cone.

# Details

  - The hyperbolic constraint is the substitution that makes the least squares risk parity
    problem disciplined convex, and the same device carries it here.
  - The hyperbolic constraint reads ``\\gamma \\leq \\sqrt{w_{i} \\zeta_{i} / b_{i}}`` for every ``i``, and ``w_{i} \\zeta_{i}`` is the risk contribution of asset ``i`` under the variance. So maximising ``\\gamma`` drives the contributions towards the stated proportions, while minimising ``\\psi`` drives the total risk down. The single objective ``\\psi - \\gamma`` does both.
  - The relaxation reads the covariance alone. This head resolves no risk measure, which is why its result carries no `r`.

# Notes

Because this is a *relaxation* of the risk budgeting problem, the realised risk contributions will not adhere to the target risk budget as tightly as the exact logarithmic-barrier or mixed-integer formulations in [`RiskBudgeting`](@ref). In well-behaved problems the deviation is negligible, but in pathological cases (e.g. ill-conditioned covariance matrices or extreme budget allocations) it can be noticeable. The trade-off is that the SOC formulation is convex and composes cleanly with additional constraints, making it the friendlier choice when the risk budget is one of several objectives rather than a hard requirement. Use [`RiskBudgeting`](@ref) when strict adherence to the risk budget is essential.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`RelaxedRiskBudgeting` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` already holds a prior **result**, the method replaces `X` with `opt.pe.X`, so the children are viewed against the prior's own observations rather than the caller's matrix.
  - `opt` recurses through [`port_opt_view`](@ref) with that matrix. `rba` recurses with the index alone.
  - `wi` is sliced to the selected assets.
  - `alg` and `fb` are carried through unchanged.

# Related

  - [`optimise`](@ref)
  - [`RelaxedRiskBudgetingResult`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgetingAlgorithm`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:gambetakwon2020])
  - $(ref_dict[:richardroncalli2019])
  - $(ref_dict[:mausserromanko2014])
  - $(ref_dict[:cajas2025]) Section 10.1.2, Equations 10.5-10.6.
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
    function RelaxedRiskBudgeting(opt::JuMPOptimiser, rba::TD{<:RiskBudgetingAlgorithm},
                                  wi::TD_Option{<:VecNum},
                                  alg::RelaxedRiskBudgetingAlgorithm,
                                  fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :RelaxedRiskBudgeting)
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        assert_time_dependent_substitution(RelaxedRiskBudgeting, (; opt, rba, wi, alg, fb),
                                           relaxed_risk_budgeting_td_defaults())
        return new{typeof(opt), typeof(rba), typeof(wi), typeof(alg), typeof(fb)}(opt, rba,
                                                                                  wi, alg,
                                                                                  fb)
    end
end
function RelaxedRiskBudgeting(; opt::JuMPOptimiser,
                              rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
                              wi::TD_Option{<:VecNum} = nothing,
                              alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
                              fb::TDO_Option{<:OptE_Opt} = nothing)::RelaxedRiskBudgeting
    return RelaxedRiskBudgeting(opt, rba, wi, alg, fb)
end
function time_dependent_field_defaults(::RelaxedRiskBudgeting)::NamedTuple
    return relaxed_risk_budgeting_td_defaults()
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for Relaxed Risk Budgeting portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

It carries **no** `r`. A [`RelaxedRiskBudgeting`](@ref) run builds its constraints straight from `pr.sigma` and never resolves a risk measure, so it sits on the [`NonRiskJuMPOptimisationResult`](@ref) half of the split and not beside [`RiskBudgetingResult`](@ref), whose `r` is mandatory.

Property access delegates to the embedded [`JuMPOptimisationResult`](@ref); unknown properties forward to `prb` first, then through `jr` (including the virtual `:w` and the `pa` fall-through).

# Constructors

    RelaxedRiskBudgetingResult(;
        jr::JuMPOptimisationResult,
        prb::Union{ProcessedAssetRiskBudgetingAttributes,
                   ProcessedFactorRiskBudgetingAttributes},
        fb::Option{<:OptE_Opt}
    ) -> RelaxedRiskBudgetingResult

Keywords correspond to the struct's fields.

# Related

  - [`RelaxedRiskBudgeting`](@ref)
  - [`RiskBudgetingResult`](@ref)
  - [`NonRiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
@concrete struct RelaxedRiskBudgetingResult <: NonRiskJuMPOptimisationResult
    """
    Shared JuMP result core, see [`JuMPOptimisationResult`](@ref).
    """
    jr
    """
    $(field_dict[:prb])
    """
    prb
    """
    $(field_dict[:fb])
    """
    fb
    function RelaxedRiskBudgetingResult(jr::JuMPOptimisationResult,
                                        prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                                   ProcessedFactorRiskBudgetingAttributes},
                                        fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(prb), typeof(fb)}(jr, prb, fb)
    end
end
function RelaxedRiskBudgetingResult(; jr::JuMPOptimisationResult,
                                    prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                               ProcessedFactorRiskBudgetingAttributes},
                                    fb::Option{<:OptE_Opt})::RelaxedRiskBudgetingResult
    return RelaxedRiskBudgetingResult(jr, prb, fb)
end
# Unique field `prb` resolves directly; unknown properties forward into `prb` first, then
# into the embedded [`JuMPOptimisationResult`](@ref) `jr` (the virtual `:w` and `pa` fall-through).
@forward_properties RelaxedRiskBudgetingResult begin
    forward(prb)
    forward(jr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the JuMP optimiser or fallback requires previous portfolio weights.
"""
function needs_previous_weights(opt::RelaxedRiskBudgeting)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.fb))
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
    psi = shared_get(model, :psi)
    G = isnothing(chol) ? LinearAlgebra.cholesky(sigma).U : chol
    JuMP.@constraint(model, cbasic_rrp, [sc * psi; sc * G * w] in JuMP.SecondOrderCone())
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(::RegularisedRelaxedRiskBudgeting,
                                                     model::JuMP.Model, w::VecJuMPScalar,
                                                     sigma::MatNum,
                                                     chol::Option{<:MatNum} = nothing)
    sc = get_constraint_scale(model)
    psi = shared_get(model, :psi)
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
    psi = shared_get(model, :psi)
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
    rkb = risk_budget_constraints(rrb.rba.rkb, rrb.rba.sets,
                                  risk_budget_universe_key(rrb.rba, N); N = N,
                                  strict = rrb.opt.strict)
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
    b1, rr = set_factor_risk_contribution_constraints!(model, rrb.rba.re, rd, pr,
                                                       rrb.rba.flag, rrb.wi)
    rkb = _set_relaxed_risk_budgeting_constraints!(model, rrb, shared_get(model, :w1),
                                                   Matrix(LinearAlgebra.Symmetric(rr.L \
                                                                                  pr.sigma *
                                                                                  b1)))
    set_weight_constraints!(model, wb, rrb.opt)
    return ProcessedFactorRiskBudgetingAttributes(; rkb = rkb, b1 = b1, rr = rr)
end
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting{<:Any,
                                                                           <:AssetRiskBudgeting,
                                                                           <:Any, <:Any},
                                                 pr::AbstractPriorResult, wb::WeightBounds,
                                                 args...)
    set_w!(model, pr.X, rrb.wi)
    set_weight_constraints!(model, wb, rrb.opt, true)
    rkb = _set_relaxed_risk_budgeting_constraints!(model, rrb, get_w(model), pr.sigma,
                                                   pr.chol)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
function _optimise(rrb::RelaxedRiskBudgeting, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    rrb = reset_time_dependent_estimator(rrb)
    attrs = processed_jump_optimiser_attributes(rrb.opt, rd; dims = dims, kwargs...)
    # The bundle reduced what it carries. The head carries the rest — an initial weight
    # vector, a risk measure holding per-asset data, tracking, a custom term — and hands
    # them to `assemble_jump_model!` itself, so it takes the same view of itself and of
    # `rd`. Both are unchanged when every asset is investable.
    rrb, rd = investable_view(rrb, rd, attrs.pr, attrs.imsk)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rrb.opt.sc, rrb.opt.so)
    set_maximum_ratio_factor_variables!(model, MinimumRisk())
    prb = set_relaxed_risk_budgeting_constraints!(model, rrb, attrs.pr, attrs.wb, rd)
    assemble_jump_model!(model, rrb, rrb.opt, attrs, rd)
    set_portfolio_objective_function!(model, MinimumRisk(), rrb, attrs)
    retcode, sol = optimise_JuMP_model!(model, rrb, eltype(attrs.pr.X))
    jr = JuMPOptimisationResult(; pa = attrs, retcode = retcode, sol = sol,
                                model = ifelse(save, model, nothing))
    return RelaxedRiskBudgetingResult(; jr = jr, prb = prb, fb = nothing)
end
"""
    optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> RelaxedRiskBudgetingResult

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
  - [`RelaxedRiskBudgetingResult`](@ref)
"""
function optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(rrb, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

@pipe_delegates RelaxedRiskBudgeting opt
@pipe_route_rkb RelaxedRiskBudgeting
export BasicRelaxedRiskBudgeting, RegularisedRelaxedRiskBudgeting,
       RegularisedPenalisedRelaxedRiskBudgeting, RelaxedRiskBudgeting,
       RelaxedRiskBudgetingResult
