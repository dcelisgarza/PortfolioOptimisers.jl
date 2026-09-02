"""
$(DocStringExtensions.TYPEDEF)

Result type for Factor Risk Contribution portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

Property access delegates to the embedded [`JuMPOptimisationResult`](@ref); unknown properties forward into `rr` first, then through `jr` (including the virtual `:w` and the `pa` fall-through).

# Constructors

    FactorRiskContributionResult(;
        jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM, rr::AbstractTimeSeriesRegressionResult,
        frc_plr::Option{<:AbstractPhylogenyConstraintResult}, fb::Option{<:OptE_Opt}
    ) -> FactorRiskContributionResult

Keywords correspond to the struct's fields.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
@concrete struct FactorRiskContributionResult <: RiskJuMPOptimisationResult
    """
    Shared JuMP result core, see [`JuMPOptimisationResult`](@ref).
    """
    jr
    """
    $(field_dict[:r_res])
    """
    r
    """
    $(field_dict[:reg_rr])
    """
    rr
    """
    Factor risk contribution placeholder result.
    """
    frc_plr
    """
    $(field_dict[:fb])
    """
    fb
    function FactorRiskContributionResult(jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                          rr::AbstractTimeSeriesRegressionResult,
                                          frc_plr::Option{<:AbstractPhylogenyConstraintResult},
                                          fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(r), typeof(rr), typeof(frc_plr), typeof(fb)}(jr, r,
                                                                                   rr,
                                                                                   frc_plr,
                                                                                   fb)
    end
end
function FactorRiskContributionResult(; jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                      rr::AbstractTimeSeriesRegressionResult,
                                      frc_plr::Option{<:AbstractPhylogenyConstraintResult},
                                      fb::Option{<:OptE_Opt})::FactorRiskContributionResult
    return FactorRiskContributionResult(jr, r, rr, frc_plr, fb)
end
# Unique fields resolve directly; unknown properties forward into `rr` first, then into the
# embedded [`JuMPOptimisationResult`](@ref) `jr` (the virtual `:w` and `pa` fall-through).
@forward_properties FactorRiskContributionResult begin
    forward(rr)
    forward(jr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`FactorRiskContribution`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function factor_risk_contribution_td_defaults()::NamedTuple
    return (; re = StepwiseRegression(), r = Variance(), obj = MinimumRisk())
end
"""
$(DocStringExtensions.TYPEDEF)

Solves a mean-risk problem whose decision variable is the vector of factor exposures rather than the vector of asset weights.

The asset weights are recovered from the exposures through the factor loadings, so a constraint written on the decision variable is a constraint on a factor. This is the change of basis alone: `FactorRiskContribution` sets **no** risk budget of its own. A target contribution per factor is stated through the risk measure's own `rc` constraints, exactly as it is for assets.

# Mathematical definition

The factor model of the loadings, fitted by `re` or carried by the prior:

```math
\\begin{align}
\\mathbf{R} &= \\mathbf{F} \\mathbf{B}^\\intercal + \\mathbf{E}\\,.
\\end{align}
```

The portfolio's factor exposures are ``\\boldsymbol{y}_{f} = \\mathbf{B}^\\intercal \\boldsymbol{w}``, and the weights are recovered from them by the Moore-Penrose pseudoinverse:

```math
\\begin{align}
\\boldsymbol{w} &= (\\mathbf{B}^\\intercal)^{+} \\boldsymbol{y}_{f}\\,, \\\\
\\boldsymbol{w} &= (\\mathbf{B}^\\intercal)^{+} \\boldsymbol{y}_{f} + (\\tilde{\\mathbf{B}}^\\intercal)^{+} \\tilde{\\boldsymbol{y}}_{af}\\,, \\quad \\tilde{\\mathbf{B}} = \\ker(\\mathbf{B}^\\intercal)\\,.
\\end{align}
```

The first form is the model built when `flag = false`, and the second when `flag = true`. The second adds the ``N - N_{f}`` directions the loadings do not span, so the weight vector is no longer confined to the factor subspace.

The problem is then the mean-risk problem of [`MeanRisk`](@ref) over ``\\boldsymbol{y}_{f}``:

```math
\\begin{align}
\\underset{\\boldsymbol{y}_{f}}{\\min} \\; f(\\boldsymbol{w}(\\boldsymbol{y}_{f})) \\quad \\text{s.t.} \\quad \\boldsymbol{w}(\\boldsymbol{y}_{f}) \\in \\mathcal{W}\\,.
\\end{align}
```

The risk contribution of factor ``j``, which [`factor_risk_contribution`](@ref) reports, follows from the Euler decomposition in that basis:

```math
\\begin{align}
RC_j(\\boldsymbol{y}_{f}) &= \\left[ \\frac{\\partial \\mathcal{R}(\\boldsymbol{w})}{\\partial \\boldsymbol{w}} (\\mathbf{B}^\\intercal)^{+} \\right]_{j} \\left[ \\mathbf{B}^\\intercal \\boldsymbol{w} \\right]_{j}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{R}``: Asset returns matrix.
  - ``\\mathbf{F}``: Factor returns matrix.
  - ``\\mathbf{B}``: Loading matrix, of size ``N \\times N_{f}``.
  - ``\\mathbf{E}``: Residual matrix.
  - ``\\boldsymbol{w}``: Portfolio weight vector.
  - ``\\boldsymbol{y}_{f}``: Factor exposure vector, the decision variable.
  - ``\\tilde{\\boldsymbol{y}}_{af}``: Exposures to the additional directions, which carry no economic interpretation.
  - ``(\\cdot)^{+}``: Moore-Penrose pseudoinverse.
  - ``f(\\boldsymbol{w})``: Objective function (depends on [`ObjectiveFunction`](@ref)).
  - ``\\mathcal{W}``: Feasible weight set defined by portfolio constraints.
  - ``\\mathcal{R}(\\boldsymbol{w})``: Portfolio risk measure.
  - ``RC_j``: Risk contribution of factor ``j``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorRiskContribution(;
        opt::JuMPOptimiser,
        re::TD{<:RegE_Reg} = StepwiseRegression(),
        r::TD{<:RM_VecRM} = Variance(),
        obj::TD{<:ObjectiveFunction} = MinimumRisk(),
        frc_ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wi::TD_Option{<:VecNum} = nothing,
        flag::Bool = false,
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> FactorRiskContribution

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the factor model, risk measure, objective, placeholder constraints, asset sets, warm start and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `flag` is execution control and stays static.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is a vector: `!isempty(wi)`.
  - `fb` schedules: `bind !== :nearest`.
  - A risk expression that is identically zero is refused. The problem is stated in the
    factor basis, so a model with no risk term bounds nothing. A [`NoRisk`](@ref) measure and
    `settings.rke = false` on every measure are the two routes to it.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`FactorRiskContribution` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` already holds a prior **result**, the method replaces `X` with `opt.pe.X`, so the children are viewed against the prior's own observations rather than the caller's matrix.
  - `opt` and `r` recurse through [`port_opt_view`](@ref) with that matrix. `re` recurses with the index alone, which slices its loadings to the selected assets and leaves the factor axis whole.
  - `wi` is carried through unchanged, because it holds initial **factor** weights. The optimisation re-bases the weight variable onto the factor axis, so the asset selection does not index `wi`.
  - `obj`, `frc_ple`, `sets`, `flag` and `fb` are carried through unchanged.

# Related

  - [`optimise`](@ref)
  - [`FactorRiskContributionResult`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 10.2.1.
  - $(ref_dict[:roncalliweisang2012])
  - $(ref_dict[:meucci2007])
"""
@propagatable @concrete struct FactorRiskContribution <: RiskJuMPOptimisationEstimator
    """
    $(field_dict[:opt_jmp])
    """
    @fprop opt
    """
    $(field_dict[:re])
    """
    re
    """
    $(field_dict[:r_opt])
    """
    @fprop r
    """
    $(field_dict[:obj])
    """
    obj
    """
    Factor risk contribution placeholder constraints.
    """
    frc_ple
    """
    $(field_dict[:sets])
    """
    sets
    """
    $(field_dict[:wi])
    """
    wi
    """
    $(field_dict[:flag])
    """
    flag
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function FactorRiskContribution(opt::JuMPOptimiser, re::TD{<:RegE_Reg},
                                    r::TD{<:RM_VecRM}, obj::TD{<:ObjectiveFunction},
                                    frc_ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC},
                                    sets::TD_Option{<:UniverseSets},
                                    wi::TD_Option{<:VecNum}, flag::Bool,
                                    fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :FactorRiskContribution)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        assert_risk_measure_required(r, :FactorRiskContribution;
                                     flag = zero_risk_expression_flag)
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        assert_time_dependent_substitution(FactorRiskContribution,
                                           (; opt, re, r, obj, frc_ple, sets, wi, flag, fb),
                                           factor_risk_contribution_td_defaults())
        return new{typeof(opt), typeof(re), typeof(r), typeof(obj), typeof(frc_ple),
                   typeof(sets), typeof(wi), typeof(flag), typeof(fb)}(opt, re, r, obj,
                                                                       frc_ple, sets, wi,
                                                                       flag, fb)
    end
end
function FactorRiskContribution(; opt::JuMPOptimiser,
                                re::TD{<:RegE_Reg} = StepwiseRegression(),
                                r::TD{<:RM_VecRM} = Variance(),
                                obj::TD{<:ObjectiveFunction} = MinimumRisk(),
                                frc_ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
                                sets::TD_Option{<:UniverseSets} = nothing,
                                wi::TD_Option{<:VecNum} = nothing, flag::Bool = false,
                                fb::TDO_Option{<:OptE_Opt} = nothing)::FactorRiskContribution
    return FactorRiskContribution(opt, re, r, obj, frc_ple, sets, wi, flag, fb)
end
function time_dependent_field_defaults(::FactorRiskContribution)::NamedTuple
    return factor_risk_contribution_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (JuMP optimiser, risk measure, or fallback).
"""
function needs_previous_weights(opt::FactorRiskContribution)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`FactorRiskContribution`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(frc::FactorRiskContribution, i, X::MatNum,
                       args...)::FactorRiskContribution
    X = isa(frc.opt.pe, AbstractPriorResult) ? frc.opt.pe.X : X
    opt = port_opt_view(frc.opt, i, X)
    re = port_opt_view(frc.re, i)
    r = port_opt_view(frc.r, i, X)
    return FactorRiskContribution(; opt = opt, re = re, r = r, obj = frc.obj,
                                  frc_ple = frc.frc_ple, sets = frc.sets, wi = frc.wi,
                                  flag = frc.flag, fb = frc.fb)
end
"""
    set_factor_risk_contribution_constraints!(model, re, rd, pr, flag, wi)

Add factor risk contribution constraints to the JuMP model.

Re-bases the weight variable onto the factor axis, `w = b1 * w1` (or `w = b1 * w1 + b2 * w2` when `flag` is `true`), using the factor loadings to specify the basis.

The loadings come from [`resolve_factor_regression`](@ref), which is the same precedence the value-level [`factor_risk_contribution`](@ref) uses: a precomputed [`Regression`](@ref) in `re` wins, then the prior's own `rr`, then a refit from `rd`. The prior outranks the refit so that the decision basis is the one the moments were projected through.

!!! warning

    A stated regression **estimator** loses to a prior that carries loadings. To override a factor prior, pass the loadings as a precomputed [`Regression`](@ref) in `re`.

# Arguments

  - `model`: JuMP model.
  - `re`: Regression result or estimator ([`RegE_Reg`](@ref)).
  - `rd`: Returns result carrying `X` and `F`, used only when the loadings must be refitted.
  - `pr`: Prior result, read for its factor block.
  - `flag`: Whether to add the off-factor weight block.
  - `wi`: Optional initial factor weights.

# Returns

  - `b1, rr`: The factor basis and the loadings it was built from.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`resolve_factor_regression`](@ref)
  - [`RegE_Reg`](@ref)
"""
function set_factor_risk_contribution_constraints!(model::JuMP.Model, re::RegE_Reg,
                                                   rd::ReturnsResult,
                                                   pr::Option{<:AbstractPriorResult},
                                                   flag::Bool, wi::Option{<:VecNum})
    rr = resolve_factor_regression(re, rd, pr)
    Bt = transpose(rr.L)
    b1 = LinearAlgebra.pinv(Bt)
    Nf = size(b1, 2)
    if flag
        b2 = LinearAlgebra.pinv(transpose(LinearAlgebra.nullspace(Bt)))
        N = size(rr.M, 1)
        JuMP.@variables(model, begin
                            w1[1:Nf]
                            w2[1:(N - Nf)]
                        end)
        JuMP.@expression(model, w, b1 * w1 + b2 * w2)
    else
        JuMP.@variable(model, w1[1:Nf])
        JuMP.@expression(model, w, b1 * w1)
    end
    set_initial_w!(w1, wi)
    return b1, rr
end
function _optimise(frc::FactorRiskContribution, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    frc = reset_time_dependent_estimator(frc)
    attrs = processed_jump_optimiser_attributes(frc.opt, rd; dims = dims, kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, frc.opt.sc, frc.opt.so)
    set_maximum_ratio_factor_variables!(model, frc.obj)
    b1, rr = set_factor_risk_contribution_constraints!(model, frc.re, rd, attrs.pr,
                                                       frc.flag, frc.wi)
    set_weight_constraints!(model, attrs.wb, frc.opt)
    frc_plr = phylogeny_constraints(frc.frc_ple, rd.F, kwargs...)
    set_sdp_frc_phylogeny_constraints!(model, frc_plr)
    assemble_jump_model!(model, frc, frc.opt, attrs, rd, frc.r, frc.obj, b1, false)
    set_portfolio_objective_function!(model, frc.obj, frc, attrs)
    retcode, sol = optimise_JuMP_model!(model, frc, eltype(attrs.pr.X))
    return FactorRiskContributionResult(;
                                        jr = JuMPOptimisationResult(; pa = attrs,
                                                                    retcode = retcode,
                                                                    sol = sol,
                                                                    model = ifelse(save,
                                                                                   model,
                                                                                   nothing)),
                                        r = factory(frc.r, attrs.pr, frc.opt.slv), rr = rr,
                                        frc_plr = frc_plr, fb = nothing)
end
"""
    optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                      <:Any, <:Any, Nothing
                  },
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> FactorRiskContributionResult

Run the Factor Risk Contribution portfolio optimisation.

# Arguments

  - `frc`: The factor risk contribution optimiser to use.
  - $(arg_dict[:rd]) If `isa(frc.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`FactorRiskContributionResult`](@ref)
"""
function optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(frc, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

@pipe_delegates FactorRiskContribution opt
@pipe_route_sigma_ucs FactorRiskContribution
export FactorRiskContribution, FactorRiskContributionResult
