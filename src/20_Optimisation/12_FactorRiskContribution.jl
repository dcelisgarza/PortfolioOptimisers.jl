"""
$(DocStringExtensions.TYPEDEF)

Result type for Factor Risk Contribution portfolio optimisation.

# Fields

  - `oe`: Type of the optimisation estimator that produced this result.
  - `pa`: Processed optimisation attributes.
  - `rr`: Regression result used for factor decomposition.
  - `frc_plr`: Factor risk contribution placeholder result.
  - `retcode`: Optimisation return code.
  - `sol`: JuMP model solution.
  - `model`: The JuMP model.
  - `fb`: Fallback result.

The `w` property is forwarded from `sol.w`.

# Related

  - [`FactorRiskContribution`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct FactorRiskContributionResult <: NonFiniteAllocationOptimisationResult
    oe
    pa
    rr
    frc_plr
    retcode
    sol
    model
    fb
end
function factory(res::FactorRiskContributionResult, fb::Option{<:OptE_Opt})
    return FactorRiskContributionResult(res.oe, res.pa, res.rr, res.frc_plr, res.retcode,
                                        res.sol, res.model, fb)
end
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

# Fields

  - `opt`: JuMP optimiser configuration.
  - `re`: Regression estimator for computing factor loadings.
  - `r`: Risk measure or vector of risk measures.
  - `obj`: Portfolio objective function.
  - `frc_ple`: Factor risk contribution placeholder constraints.
  - `sets`: Asset sets.
  - `wi`: Initial weights for warm-starting.
  - `flag`: If `true`, uses the full factor regression decomposition; if `false`, uses a simplified approach.
  - `fb`: Fallback optimiser.

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

# Related

  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`factor_risk_contribution`](@ref)
"""
@concrete struct FactorRiskContribution <: RiskJuMPOptimisationEstimator
    opt
    re
    r
    obj
    frc_ple
    sets
    wi
    flag
    fb
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
function needs_previous_weights(opt::FactorRiskContribution)
    return (needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
function factory(frc::FactorRiskContribution, w::AbstractVector)::FactorRiskContribution
    opt = factory(frc.opt, w)
    r = factory(frc.r, w)
    fb = factory(frc.fb, w)
    return FactorRiskContribution(; opt = opt, re = frc.re, r = r, obj = frc.obj,
                                  frc_ple = frc.frc_ple, sets = frc.sets, wi = frc.wi,
                                  flag = frc.flag, fb = fb)
end
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
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, slt, sst, sgmtx, sglt, sgst, plr, tn, fees, ret) = processed_jump_optimiser_attributes(frc.opt,
                                                                                                                                                rd;
                                                                                                                                                dims = dims,
                                                                                                                                                kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, frc.opt.sc, frc.opt.so)
    set_maximum_ratio_factor_variables!(model, pr.mu, frc.obj)
    b1, rr = set_factor_risk_contribution_constraints!(model, frc.re, rd, frc.flag, frc.wi)
    set_weight_constraints!(model, wb, frc.opt.bgt, frc.opt.sbgt)
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, frc.opt.card, gcardr, plr, lt, st, fees, frc.opt.ss)
    set_smip_constraints!(model, wb, frc.opt.scard, sgcardr, smtx, sgmtx, slt, sst, sglt,
                          sgst, frc.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, frc.opt.tr, frc, plr, fees, b1; rd = rd)
    set_number_effective_assets!(model, frc.opt.nea)
    set_l1_regularisation!(model, frc.opt.l1)
    set_l2_regularisation!(model, frc.opt.l2)
    set_linf_regularisation!(model, frc.opt.linf)
    set_lp_regularisation!(model, frc.opt.lp)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, frc.r, frc, pr, plr, fees, b1; rd = rd)
    scalarise_risk_expression!(model, frc.opt.sca)
    set_return_constraints!(model, ret, frc.obj, pr; rd = rd)
    frc_plr = phylogeny_constraints(frc.frc_ple, rd.F, kwargs...)
    set_sdp_frc_phylogeny_constraints!(model, frc_plr)
    add_custom_constraint!(model, frc.opt.ccnt, frc, pr)
    set_portfolio_objective_function!(model, frc.obj, ret, frc.opt.cobj, frc, pr)
    retcode, sol = optimise_JuMP_model!(model, frc, eltype(pr.X))
    return FactorRiskContributionResult(typeof(frc),
                                        ProcessedJuMPOptimiserAttributes(pr, wb, lt, st,
                                                                         lcsr, ctr, gcardr,
                                                                         sgcardr, smtx,
                                                                         sgmtx, slt, sst,
                                                                         sglt, sgst, tn,
                                                                         fees, plr, ret),
                                        rr, frc_plr, retcode, sol,
                                        ifelse(save, model, nothing), nothing)
end
"""
    optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                      <:Any, <:Any, Nothing
                  },
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> FactorRiskContributionResult

# Arguments

  - `frc`: The factor risk contribution optimiser to use.
  - $(arg_dict[:rd]) If `isa(hec.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(frc, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export FactorRiskContribution, FactorRiskContributionResult
