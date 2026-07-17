"""
$(DocStringExtensions.TYPEDEF)

Result type for Factor Risk Contribution portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

Property access delegates to the embedded [`JuMPOptimisationResult`](@ref); unknown properties forward into `rr` first, then through `jr` (including the virtual `:w` and the `pa` fall-through).

# Constructors

    FactorRiskContributionResult(;
        jr::JuMPOptimisationResult, rr::AbstractRegressionResult,
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
    function FactorRiskContributionResult(jr::JuMPOptimisationResult,
                                          rr::AbstractRegressionResult,
                                          frc_plr::Option{<:AbstractPhylogenyConstraintResult},
                                          fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(rr), typeof(frc_plr), typeof(fb)}(jr, rr, frc_plr, fb)
    end
end
function FactorRiskContributionResult(; jr::JuMPOptimisationResult,
                                      rr::AbstractRegressionResult,
                                      frc_plr::Option{<:AbstractPhylogenyConstraintResult},
                                      fb::Option{<:OptE_Opt})::FactorRiskContributionResult
    return FactorRiskContributionResult(jr, rr, frc_plr, fb)
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

Factor Risk Contribution (FRC) portfolio optimiser.

`FactorRiskContribution` allocates portfolio weights so that each factor (and the idiosyncratic component) contributes a target proportion to the total portfolio risk. It combines factor regression with a JuMP-based risk budgeting optimisation.

# Mathematical definition

Factor model:

```math
\\begin{align}
\\boldsymbol{r}_i &= \\alpha_i + \\mathbf{F} \\boldsymbol{\\beta}_i + \\boldsymbol{\\varepsilon}_i\\,.
\\end{align}
```

Factor risk contribution for factor ``k``:

```math
\\begin{align}
RC_k &= \\beta_{k,\\boldsymbol{w}} \\cdot \\frac{\\partial \\mathcal{R}(\\boldsymbol{w})}{\\partial \\beta_{k,\\boldsymbol{w}}}\\,, \\\\
\\beta_{k,\\boldsymbol{w}} &= \\boldsymbol{w}^\\intercal \\boldsymbol{\\beta}_k\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{r}_i``: Return vector of asset ``i``.
  - ``\\alpha_i``: Intercept (idiosyncratic return) for asset ``i``.
  - ``\\mathbf{F}``: Factor returns matrix.
  - ``\\boldsymbol{\\beta}_i``: Factor loading vector for asset ``i``.
  - ``\\boldsymbol{\\varepsilon}_i``: Idiosyncratic residual for asset ``i``.
  - ``RC_k``: Risk contribution of factor ``k``.
  - ``\\beta_{k,\\boldsymbol{w}}``: Portfolio-level exposure to factor ``k``.
  - ``\\mathcal{R}(\\boldsymbol{w})``: Portfolio risk measure.
  - ``\\boldsymbol{w}``: Portfolio weight vector.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorRiskContribution(;
        opt::JuMPOptimiser,
        re::TD{<:RegE_Reg} = StepwiseRegression(),
        r::TD{<:RM_VecRM} = Variance(),
        obj::TD{<:ObjectiveFunction} = MinimumRisk(),
        frc_ple::TD_Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
        sets::TD_Option{<:AssetSets} = nothing,
        wi::TD_Option{<:VecNum} = nothing,
        flag::Bool = false,
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> FactorRiskContribution

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the factor model, risk measure, objective, placeholder constraints, asset sets, warm start and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `flag` is execution control and stays static.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is a vector: `!isempty(wi)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Related

  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`factory`](@ref)
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
                                    frc_ple::TD_Option{<:PlCE_PhC_VecPlCE_PlC},
                                    sets::TD_Option{<:AssetSets}, wi::TD_Option{<:VecNum},
                                    flag::Bool, fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :FactorRiskContribution)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        assert_risk_measure_required(r, :FactorRiskContribution)
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
                                frc_ple::TD_Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
                                sets::TD_Option{<:AssetSets} = nothing,
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
    set_factor_risk_contribution_constraints!(model, re, ...)

Add factor risk contribution constraints to the JuMP model.

Sets up the factor-level risk budgeting constraints in the optimisation model, using the regression result or estimator `re` to specify factor loadings.

# Arguments

  - `model`: JuMP model.
  - `re`: Regression result or estimator ([`RegE_Reg`](@ref)).
  - Additional risk and budget parameters.

# Returns

  - `nothing`.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`RegE_Reg`](@ref)
"""
function set_factor_risk_contribution_constraints!(model::JuMP.Model, re::RegE_Reg,
                                                   rd::ReturnsResult, flag::Bool,
                                                   wi::Option{<:VecNum})
    if isa(re, AbstractRegressionEstimator)
        @argcheck(!isnothing(rd.X) && !isnothing(rd.F),
                  IsNothingError("Factor risk budgeting/contribution with a regression estimator (`re::$(typeof(re))`) must fit the factor model, which needs the returns data: `rd.X` and `rd.F` must not be `nothing`.\nEither pass the `ReturnsResult` to `optimise` (e.g. `optimise(est, rd)`), or supply a precomputed `Regression` result as `re`, which needs no data.\nGot\nisnothing(rd.X) => $(isnothing(rd.X))\nisnothing(rd.F) => $(isnothing(rd.F))"))
    end
    rr = regression(re, rd)
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
    set_maximum_ratio_factor_variables!(model, attrs.pr.mu, frc.obj)
    b1, rr = set_factor_risk_contribution_constraints!(model, frc.re, rd, frc.flag, frc.wi)
    set_weight_constraints!(model, attrs.wb, frc.opt.bgt, frc.opt.sbgt)
    frc_plr = phylogeny_constraints(frc.frc_ple, rd.F, kwargs...)
    set_sdp_frc_phylogeny_constraints!(model, frc_plr)
    assemble_jump_model!(model, frc, frc.opt, attrs, rd, frc.r, frc.obj, b1, false)
    set_portfolio_objective_function!(model, frc.obj, attrs.ret, frc.opt.cobj, frc,
                                      attrs.pr, attrs)
    retcode, sol = optimise_JuMP_model!(model, frc, eltype(attrs.pr.X))
    return FactorRiskContributionResult(;
                                        jr = JuMPOptimisationResult(; pa = attrs,
                                                                    retcode = retcode,
                                                                    sol = sol,
                                                                    model = ifelse(save,
                                                                                   model,
                                                                                   nothing)),
                                        rr = rr, frc_plr = frc_plr, fb = nothing)
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

export FactorRiskContribution, FactorRiskContributionResult
