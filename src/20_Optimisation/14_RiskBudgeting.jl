"""
$(DocStringExtensions.TYPEDEF)

Result type for Risk Budgeting portfolio optimisation.

$(DocStringExtensions.FIELDS)

The `w` property is forwarded from `sol.w`.

# Related

  - [`RiskBudgeting`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct RiskBudgetingResult <: NonFiniteAllocationOptimisationResult
    "$(field_dict[:oe])"
    oe
    "$(field_dict[:pa])"
    pa
    "$(field_dict[:prb])"
    prb
    "$(field_dict[:retcode])"
    retcode
    "$(field_dict[:sol])"
    sol
    "$(field_dict[:model])"
    model
    "$(field_dict[:fb])"
    fb
end
function factory(res::RiskBudgetingResult, fb::Option{<:OptE_Opt})
    return RiskBudgetingResult(res.oe, res.pa, res.prb, res.retcode, res.sol, res.model, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Access properties of [`RiskBudgetingResult`](@ref). Virtual property `:w` extracts portfolio weights from `sol`; other unknown properties forward to `r.prb` then `r.pa`.
"""
function Base.getproperty(r::RiskBudgetingResult, sym::Symbol)
    return if sym == :w
        r.sol.w
    elseif sym in propertynames(r)
        getfield(r, sym)
    elseif sym in propertynames(r.prb)
        getproperty(r.prb, sym)
    elseif sym in propertynames(r.pa)
        getproperty(r.pa, sym)
    else
        getfield(r, sym)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Processed factor risk budgeting attributes for intermediate computations.

# Related

  - [`RiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
"""
@concrete struct ProcessedFactorRiskBudgetingAttributes <: AbstractResult
    rkb
    b1
    rr
end
"""
$(DocStringExtensions.TYPEDEF)

Processed asset risk budgeting attributes for intermediate computations.

# Related

  - [`RiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
"""
@concrete struct ProcessedAssetRiskBudgetingAttributes <: AbstractResult
    rkb
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk budgeting optimisation formulations.

# Related Types

  - [`LogRiskBudgeting`](@ref)
  - [`MixedIntegerRiskBudgeting`](@ref)
"""
abstract type RiskBudgetingFormulation <: OptimisationAlgorithm end
"""
    risk_budgeting_formulation_view(::RiskBudgetingFormulation, args...) -> nothing

Default fallback for risk budgeting formulation view. Returns `nothing` for formulations that do not require view slicing.
"""
function risk_budgeting_formulation_view(::RiskBudgetingFormulation, args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Log-barrier formulation for Risk Budgeting.

Uses a logarithmic objective to enforce the risk budget constraints. Can provide an optional orthant vector to allow for negative weights in specific assets.

# Arguments

  - `z::Option{<:VecInt}`: Optional orthant vector defining which asset can have negative weights. If nothing all assets will have positive weights.

# Related Types

  - [`RiskBudgetingFormulation`](@ref)
  - [`MixedIntegerRiskBudgeting`](@ref)
"""
@concrete struct LogRiskBudgeting{T} <: RiskBudgetingFormulation
    z::T
    function LogRiskBudgeting(z::Option{<:VecInt})
        if !isnothing(z)
            @argcheck(!isempty(z))
            @argcheck(all(x->abs(x) == 1, z))
        end
        return new{typeof(z)}(z)
    end
end
function LogRiskBudgeting(; z::Option{<:VecInt} = nothing)::LogRiskBudgeting
    return LogRiskBudgeting(z)
end
function risk_budgeting_formulation_view(alg::LogRiskBudgeting{Nothing}, i)
    return alg
end
function risk_budgeting_formulation_view(alg::LogRiskBudgeting{<:VecInt}, i)
    return LogRiskBudgeting(; z = view(alg.z, i))
end
"""
$(DocStringExtensions.TYPEDEF)

Mixed-integer formulation for Risk Budgeting.

Uses binary variables and big-M constraints to enforce the risk budget constraints. This can find the minimal risk portfolio which meets the risk budgeting constraints by exploring all possible sign combinations of weights. This can be very expensive for large universes.

# Related Types

  - [`RiskBudgetingFormulation`](@ref)
  - [`LogRiskBudgeting`](@ref)
"""
struct MixedIntegerRiskBudgeting <: RiskBudgetingFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk budgeting algorithm specifications.

# Related Types

  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
"""
abstract type RiskBudgetingAlgorithm <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Asset-level Risk Budgeting algorithm.

`AssetRiskBudgeting` specifies the risk budget as a vector of asset-level risk targets, optionally grouped by asset sets.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetRiskBudgeting(;
        rkb::Option{<:RkbE_Rkb} = nothing,
        sets::Option{<:AssetSets} = nothing,
        alg::RiskBudgetingFormulation = LogRiskBudgeting()
    ) -> AssetRiskBudgeting

Keywords correspond to the struct's fields.

# Related

  - [`RiskBudgetingAlgorithm`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
"""
@concrete struct AssetRiskBudgeting <: RiskBudgetingAlgorithm
    "$(field_dict[:rkb])"
    rkb
    "$(field_dict[:sets])"
    sets
    "$(field_dict[:rba])"
    alg
    function AssetRiskBudgeting(rkb::Option{<:RkbE_Rkb}, sets::Option{<:AssetSets},
                                alg::RiskBudgetingFormulation)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(rkb), typeof(sets), typeof(alg)}(rkb, sets, alg)
    end
end
function AssetRiskBudgeting(; rkb::Option{<:RkbE_Rkb} = nothing,
                            sets::Option{<:AssetSets} = nothing,
                            alg::RiskBudgetingFormulation = LogRiskBudgeting())::AssetRiskBudgeting
    return AssetRiskBudgeting(rkb, sets, alg)
end
"""
    risk_budgeting_algorithm_view(r, i)

Return a view or subset of a risk budgeting algorithm for cluster index `i`.

Used in hierarchical optimisation to slice risk budget and asset set configurations for each cluster.

# Arguments

  - `r`: Risk budgeting algorithm ([`AssetRiskBudgeting`](@ref) or [`FactorRiskBudgeting`](@ref)).
  - `i`: Cluster or asset index.

# Returns

  - Sliced risk budgeting algorithm.

# Related

  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
"""
function risk_budgeting_algorithm_view(r::AssetRiskBudgeting, i)
    rkb = risk_budget_view(r.rkb, i)
    sets = asset_sets_view(r.sets, i)
    alg = risk_budgeting_formulation_view(r.alg, i)
    return AssetRiskBudgeting(; rkb = rkb, sets = sets, alg = alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Factor-level Risk Budgeting algorithm.

`FactorRiskBudgeting` specifies the risk budget at the factor level, using a factor model regression to decompose risk across factors and an idiosyncratic component.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorRiskBudgeting(;
        re::RegE_Reg = StepwiseRegression(),
        rkb::Option{<:RkbE_Rkb} = nothing,
        sets::Option{<:AssetSets} = nothing,
        flag::Bool = true
    ) -> FactorRiskBudgeting

Keywords correspond to the struct's fields.

# Related

  - [`RiskBudgetingAlgorithm`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
"""
@concrete struct FactorRiskBudgeting <: RiskBudgetingAlgorithm
    "$(field_dict[:re])"
    re
    "$(field_dict[:rkb])"
    rkb
    "$(field_dict[:sets])"
    sets
    "$(field_dict[:flag])"
    flag
    function FactorRiskBudgeting(re::RegE_Reg, rkb::Option{<:RkbE_Rkb},
                                 sets::Option{<:AssetSets}, flag::Bool)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(re), typeof(rkb), typeof(sets), typeof(flag)}(re, rkb, sets, flag)
    end
end
function FactorRiskBudgeting(; re::RegE_Reg = StepwiseRegression(),
                             rkb::Option{<:RkbE_Rkb} = nothing,
                             sets::Option{<:AssetSets} = nothing,
                             flag::Bool = true)::FactorRiskBudgeting
    return FactorRiskBudgeting(re, rkb, sets, flag)
end
"""
    risk_budgeting_algorithm_view(r::FactorRiskBudgeting, i)

Return a view of a `FactorRiskBudgeting` algorithm for cluster index `i`.

Slices the regression estimator for the given cluster while keeping the risk budget, asset sets, and idiosyncratic flag unchanged.

# Arguments

  - `r::FactorRiskBudgeting`: Factor-level risk budgeting algorithm.
  - `i`: Cluster or asset index.

# Returns

  - `FactorRiskBudgeting` with the regression estimator sliced to cluster `i`.

# Related

  - [`risk_budgeting_algorithm_view`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
"""
function risk_budgeting_algorithm_view(r::FactorRiskBudgeting, i)
    re = regression_view(r.re, i)
    return FactorRiskBudgeting(; re = re, rkb = r.rkb, sets = r.sets, flag = r.flag)
end
"""
$(DocStringExtensions.TYPEDEF)

Risk Budgeting (RB) portfolio optimiser.

`RiskBudgeting` allocates portfolio weights so that each asset (or factor) contributes a specified fraction of the total portfolio risk. It uses a logarithmic or mixed-integer formulation and can be combined with any risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudgeting(;
        opt::JuMPOptimiser = JuMPOptimiser(),
        r::RM_VecRM = Variance(),
        rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
        wi::Option{<:VecNum} = nothing,
        fb::Option{<:OptE_Opt} = nothing
    ) -> RiskBudgeting

Keywords correspond to the struct's fields.

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
"""
@concrete struct RiskBudgeting <: RiskJuMPOptimisationEstimator
    "$(field_dict[:opt_jmp])"
    opt
    "$(field_dict[:r_opt])"
    r
    "$(field_dict[:rba])"
    rba
    "$(field_dict[:wi])"
    wi
    "$(field_dict[:fb])"
    fb
    function RiskBudgeting(opt::JuMPOptimiser, r::RM_VecRM, rba::RiskBudgetingAlgorithm,
                           wi::Option{<:VecNum}, fb::Option{<:OptE_Opt})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
        end
        if isa(wi, VecNum)
            @argcheck(!isempty(wi))
        end
        return new{typeof(opt), typeof(r), typeof(rba), typeof(wi), typeof(fb)}(opt, r, rba,
                                                                                wi, fb)
    end
end
function RiskBudgeting(; opt::JuMPOptimiser = JuMPOptimiser(), r::RM_VecRM = Variance(),
                       rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
                       wi::Option{<:VecNum} = nothing,
                       fb::Option{<:OptE_Opt} = nothing)::RiskBudgeting
    return RiskBudgeting(opt, r, rba, wi, fb)
end
function needs_previous_weights(opt::RiskBudgeting)
    return (needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
function factory(rb::RiskBudgeting, w::AbstractVector)::RiskBudgeting
    opt = factory(rb.opt, w)
    r = factory(rb.r, w)
    fb = factory(rb.fb, w)
    return RiskBudgeting(; opt = opt, r = r, rba = rb.rba, wi = rb.wi, fb = fb)
end
function opt_view(rb::RiskBudgeting, i, X::MatNum)::RiskBudgeting
    X = isa(rb.opt.pe, AbstractPriorResult) ? rb.opt.pe.X : X
    opt = opt_view(rb.opt, i, X)
    r = risk_measure_view(rb.r, i, X)
    rba = risk_budgeting_algorithm_view(rb.rba, i)
    wi = nothing_scalar_array_view(rb.wi, i)
    return RiskBudgeting(; opt = opt, r = r, rba = rba, wi = wi, fb = rb.fb)
end
"""
    _set_risk_budgeting_constraints!(model, rb, ...)

Internal function to set risk budgeting constraints in the JuMP model.

Configures the equality constraints ensuring each asset's marginal risk contribution equals its budget target.

# Arguments

  - `model`: JuMP model.
  - `rb`: [`RiskBudgeting`](@ref) optimiser configuration.
  - Additional risk and budget parameters.

# Returns

  - `nothing`.

# Related

  - [`RiskBudgeting`](@ref)
"""
function _set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting,
                                          w::VecJuMPScalar; strict::Bool = false)
    N = length(w)
    rkb = risk_budget_constraints(rb.rba.rkb, rb.rba.sets; N = N, strict = strict)
    rb = rkb.val
    @argcheck(length(rb) == N)
    sc = model[:sc]
    JuMP.@variables(model, begin
                        k
                        log_w[1:N]
                    end)
    JuMP.@constraints(model,
                      begin
                          clog_w[i = 1:N],
                          [sc * log_w[i], sc, sc * w[i]] in JuMP.MOI.ExponentialCone()
                          crkb, sc * LinearAlgebra.dot(rb, log_w) >= 0
                      end)
    return rkb
end
"""
    set_risk_budgeting_constraints!(model, rb, pr, wb, args...)

Add risk budgeting constraints and weight variables to the JuMP model.

Dispatches based on the risk budgeting algorithm and formulation. Sets up weight variables, logarithmic risk budget constraints, and weight bounds for the specified formulation (log, MIP, or factor-based).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `rb::RiskBudgeting`: Risk budgeting estimator configuration.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `wb::WeightBounds`: Weight bounds configuration.
  - `args...`: Additional arguments (e.g. returns data for factor risk budgeting).

# Returns

  - Processed risk budgeting attributes.

# Related

  - [`RiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
"""
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:LogRiskBudgeting{Nothing}},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
    set_w!(model, pr.X, rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w]; strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, nothing, true)
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:LogRiskBudgeting{<:VecInt}},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
    set_w!(model, pr.X, rb.wi)
    z = rb.rba.alg.z
    @argcheck(length(z) == length(model[:w]))
    w = z .* model[:w]
    rkb = _set_risk_budgeting_constraints!(model, rb, w; strict = rb.opt.strict)
    sc = model[:sc]
    k = model[:k]
    JuMP.@constraints(model, begin
                          mipcrkb, sc * (sum(w) - k) >= 0
                          orthcrkb, sc * w >= 0
                      end)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:FactorRiskBudgeting, <:Any},
                                         ::Any, wb::WeightBounds, rd::ReturnsResult)
    b1, rr = set_factor_risk_contribution_constraints!(model, rb.rba.re, rd, rb.rba.flag,
                                                       rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w1]; strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedFactorRiskBudgetingAttributes(rkb, b1, rr)
end
"""
    set_rb_mip_w!(model::JuMP.Model, X::MatNum)

Create long and short weight variables for MIP risk budgeting in the JuMP model.

Registers long `lw`, short `sw` weight variables and the derived expressions `w = lw - sw` and `w_obj = lw + sw`.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix (used to determine number of assets).

# Returns

  - `nothing`.

# Related

  - [`set_risk_budgeting_constraints!`](@ref)
  - [`RiskBudgeting`](@ref)
"""
function set_rb_mip_w!(model::JuMP.Model, X::MatNum)
    N = size(X, 2)
    JuMP.@variables(model, begin
                        lw[1:N] >= 0
                        sw[1:N] >= 0
                    end)
    JuMP.@expressions(model, begin
                          w, lw - sw
                          w_obj, lw + sw
                      end)
    return nothing
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:MixedIntegerRiskBudgeting},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
    set_rb_mip_w!(model, pr.X)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w_obj]; strict = rb.opt.strict)
    w = model[:w]
    sc = model[:sc]
    k = model[:k]
    JuMP.@constraint(model, mipcrkb, sc * (sum(w) - k) >= 0)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
function _optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, slt, sst, sgmtx, sglt, sgst, plr, tn, fees, ret) = processed_jump_optimiser_attributes(rb.opt,
                                                                                                                                                rd;
                                                                                                                                                dims = dims,
                                                                                                                                                kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    prb = set_risk_budgeting_constraints!(model, rb, pr, wb, rd)
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, rb.opt.card, gcardr, plr, lt, st, fees, rb.opt.ss,
                         isa(rb.rba,
                             AssetRiskBudgeting{<:Any, <:Any, <:MixedIntegerRiskBudgeting}))
    set_smip_constraints!(model, wb, rb.opt.scard, sgcardr, smtx, sgmtx, slt, sst, sglt,
                          sgst, rb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rb.opt.tr, rb, plr, fees; rd = rd)
    set_number_effective_assets!(model, rb.opt.nea)
    set_l1_regularisation!(model, rb.opt.l1)
    set_l2_regularisation!(model, rb.opt.l2)
    set_linf_regularisation!(model, rb.opt.linf)
    set_lp_regularisation!(model, rb.opt.lp)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, rb.r, rb, pr, plr, fees; rd = rd)
    scalarise_risk_expression!(model, rb.opt.sca)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, plr)
    add_custom_constraint!(model, rb.opt.ccnt, rb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rb.opt.cobj, rb, pr)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(pr.X))
    return RiskBudgetingResult(typeof(rb),
                               ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr,
                                                                gcardr, sgcardr, smtx,
                                                                sgmtx, slt, sst, sglt, sgst,
                                                                tn, fees, plr, ret), prb,
                               retcode, sol, ifelse(save, model, nothing), nothing)
end
"""
    optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> RiskBudgetingResult

# Arguments

  - `rb`: The risk budgeting optimiser to use.
  - $(arg_dict[:rd]) If `isa(hec.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.
"""
function optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(rb, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export AssetRiskBudgeting, FactorRiskBudgeting, RiskBudgeting, RiskBudgetingResult,
       LogRiskBudgeting, MixedIntegerRiskBudgeting
