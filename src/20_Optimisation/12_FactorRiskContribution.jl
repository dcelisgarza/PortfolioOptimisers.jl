"""
$(DocStringExtensions.TYPEDEF)

Result type for Factor Risk Contribution portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

The `w` property is forwarded from `sol.w`.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct FactorRiskContributionResult <: NonFiniteAllocationOptimisationResult
    """
    $(field_dict[:oe])
    """
    oe
    """
    $(field_dict[:pa])
    """
    pa
    """
    $(field_dict[:reg_rr])
    """
    rr
    """
    Factor risk contribution placeholder result.
    """
    frc_plr
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:sol])
    """
    sol
    """
    $(field_dict[:model])
    """
    model
    """
    $(field_dict[:fb])
    """
    fb
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a [`FactorRiskContributionResult`](@ref) with an updated fallback optimiser `fb`.
"""
function factory(res::FactorRiskContributionResult, fb::Option{<:OptE_Opt})
    return FactorRiskContributionResult(res.oe, res.pa, res.rr, res.frc_plr, res.retcode,
                                        res.sol, res.model, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`FactorRiskContributionResult`](@ref). Virtual property `:w` extracts portfolio weights from `sol`; other unknown properties forward to `r.rr` then `r.pa`.
"""
function Base.getproperty(r::FactorRiskContributionResult, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? getfield(r.sol, :w) : getfield.(r.sol, :w)
    elseif sym in propertynames(r)
        getfield(r, sym)
    elseif sym in propertynames(r.rr)
        getproperty(r.rr, sym)
    elseif sym in propertynames(r.pa)
        getproperty(r.pa, sym)
    else
        getfield(r, sym)
    end
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
        opt::JuMPOptimiser = JuMPOptimiser(),
        re::RegE_Reg = StepwiseRegression(),
        r::RM_VecRM = Variance(),
        obj::ObjectiveFunction = MinimumRisk(),
        frc_ple::Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
        sets::Option{<:AssetSets} = nothing,
        wi::Option{<:VecNum} = nothing,
        flag::Bool = false,
        fb::Option{<:OptE_Opt} = nothing
    ) -> FactorRiskContribution

Keywords correspond to the struct's fields.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is provided: `!isempty(wi)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@prop`-tagged fields are automatically propagated:

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
    @prop opt
    """
    $(field_dict[:re])
    """
    re
    """
    $(field_dict[:r_opt])
    """
    @prop r
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
    @prop fb
    function FactorRiskContribution(opt::JuMPOptimiser, re::RegE_Reg, r::RM_VecRM,
                                    obj::ObjectiveFunction,
                                    frc_ple::Option{<:PlCE_PhC_VecPlCE_PlC},
                                    sets::Option{<:AssetSets}, wi::Option{<:VecNum},
                                    flag::Bool, fb::Option{<:OptE_Opt})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
        end
        if isa(wi, VecNum)
            @argcheck(!isempty(wi))
        end
        return new{typeof(opt), typeof(re), typeof(r), typeof(obj), typeof(frc_ple),
                   typeof(sets), typeof(wi), typeof(flag), typeof(fb)}(opt, re, r, obj,
                                                                       frc_ple, sets, wi,
                                                                       flag, fb)
    end
end
#= Old factory function:
function factory(frc::FactorRiskContribution, w::AbstractVector)::FactorRiskContribution
    opt = factory(frc.opt, w)
    r = factory(frc.r, w)
    fb = factory(frc.fb, w)
    return FactorRiskContribution(; opt = opt, re = frc.re, r = r, obj = frc.obj,
                                  frc_ple = frc.frc_ple, sets = frc.sets, wi = frc.wi,
                                  flag = frc.flag, fb = fb)
end
=#
function FactorRiskContribution(; opt::JuMPOptimiser = JuMPOptimiser(),
                                re::RegE_Reg = StepwiseRegression(),
                                r::RM_VecRM = Variance(),
                                obj::ObjectiveFunction = MinimumRisk(),
                                frc_ple::Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
                                sets::Option{<:AssetSets} = nothing,
                                wi::Option{<:VecNum} = nothing, flag::Bool = false,
                                fb::Option{<:OptE_Opt} = nothing)::FactorRiskContribution
    return FactorRiskContribution(opt, re, r, obj, frc_ple, sets, wi, flag, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (JuMP optimiser, risk measure, or fallback).
"""
function needs_previous_weights(opt::FactorRiskContribution)
    return (needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`FactorRiskContribution`](@ref) for asset index set `i` and returns matrix `X`.
"""
function opt_view(frc::FactorRiskContribution, i, X::MatNum)::FactorRiskContribution
    X = isa(frc.opt.pe, AbstractPriorResult) ? frc.opt.pe.X : X
    opt = opt_view(frc.opt, i, X)
    re = regression_view(frc.re, i)
    r = risk_measure_view(frc.r, i, X)
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
    attrs = processed_jump_optimiser_attributes(frc.opt, rd; dims = dims, kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, frc.opt.sc, frc.opt.so)
    set_maximum_ratio_factor_variables!(model, attrs.pr.mu, frc.obj)
    b1, rr = set_factor_risk_contribution_constraints!(model, frc.re, rd, frc.flag, frc.wi)
    set_weight_constraints!(model, attrs.wb, frc.opt.bgt, frc.opt.sbgt)
    _assemble_jump_model!(model, frc, frc.opt, attrs, rd; r = frc.r, b1 = b1, obj = frc.obj,
                          sdp_phylogeny = false)
    frc_plr = phylogeny_constraints(frc.frc_ple, rd.F, kwargs...)
    set_sdp_frc_phylogeny_constraints!(model, frc_plr)
    set_portfolio_objective_function!(model, frc.obj, attrs.ret, frc.opt.cobj, frc,
                                      attrs.pr)
    retcode, sol = optimise_JuMP_model!(model, frc, eltype(attrs.pr.X))
    return FactorRiskContributionResult(typeof(frc), attrs, rr, frc_plr, retcode, sol,
                                        ifelse(save, model, nothing), nothing)
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
