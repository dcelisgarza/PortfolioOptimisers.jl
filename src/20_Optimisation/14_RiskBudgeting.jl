"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for processed risk budgeting attributes. Every collection of processed risk budgeting attributes should subtype this.

# Related

  - [`ProcessedAssetRiskBudgetingAttributes`](@ref)
  - [`ProcessedFactorRiskBudgetingAttributes`](@ref)
"""
abstract type ProcessedRiskBudgetingAttributes <: ProcessedAttributes end
"""
$(DocStringExtensions.TYPEDEF)

Processed factor risk budgeting attributes for intermediate computations.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`RiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
"""
@concrete struct ProcessedFactorRiskBudgetingAttributes <: ProcessedRiskBudgetingAttributes
    """
    Processed risk budget constraints vector.
    """
    rkb
    """
    Factor-level risk budget vector.
    """
    b1
    """
    Regression result used for factor loading estimation.
    """
    rr
    function ProcessedFactorRiskBudgetingAttributes(rkb::RiskBudget, b1::MatNum,
                                                    rr::AbstractRegressionResult)
        return new{typeof(rkb), typeof(b1), typeof(rr)}(rkb, b1, rr)
    end
end
function ProcessedFactorRiskBudgetingAttributes(; rkb::RiskBudget, b1::MatNum,
                                                rr::AbstractRegressionResult)::ProcessedFactorRiskBudgetingAttributes
    return ProcessedFactorRiskBudgetingAttributes(rkb, b1, rr)
end
"""
$(DocStringExtensions.TYPEDEF)

Processed asset risk budgeting attributes for intermediate computations.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`RiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
"""
@concrete struct ProcessedAssetRiskBudgetingAttributes <: ProcessedRiskBudgetingAttributes
    """
    Processed asset risk budget constraints vector.
    """
    rkb
    function ProcessedAssetRiskBudgetingAttributes(rkb::RiskBudget)
        return new{typeof(rkb)}(rkb)
    end
end
function ProcessedAssetRiskBudgetingAttributes(;
                                               rkb::RiskBudget)::ProcessedAssetRiskBudgetingAttributes
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for Risk Budgeting portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

Property access delegates to the embedded [`JuMPOptimisationResult`](@ref); unknown properties forward to `prb` first, then through `jr` (including the virtual `:w` and the `pa` fall-through).

# Constructors

    RiskBudgetingResult(;
        jr::JuMPOptimisationResult,
        prb::Union{ProcessedAssetRiskBudgetingAttributes,
                   ProcessedFactorRiskBudgetingAttributes},
        fb::Option{<:OptE_Opt}
    ) -> RiskBudgetingResult

Keywords correspond to the struct's fields.

# Related

  - [`RiskBudgeting`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
@concrete struct RiskBudgetingResult <: RiskJuMPOptimisationResult
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
    function RiskBudgetingResult(jr::JuMPOptimisationResult,
                                 prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                            ProcessedFactorRiskBudgetingAttributes},
                                 fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(prb), typeof(fb)}(jr, prb, fb)
    end
end
function RiskBudgetingResult(; jr::JuMPOptimisationResult,
                             prb::Union{ProcessedAssetRiskBudgetingAttributes,
                                        ProcessedFactorRiskBudgetingAttributes},
                             fb::Option{<:OptE_Opt})::RiskBudgetingResult
    return RiskBudgetingResult(jr, prb, fb)
end
# Unique field `prb` resolves directly; unknown properties forward into `prb` first, then
# into the embedded [`JuMPOptimisationResult`](@ref) `jr` (the virtual `:w` and `pa` fall-through).
@forward_properties RiskBudgetingResult begin
    forward(prb)
    forward(jr)
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
    port_opt_view(::RiskBudgetingFormulation, args...) -> nothing

Default fallback for risk budgeting formulation view. Returns `nothing` for formulations that do not require view slicing.
"""
function port_opt_view(::RiskBudgetingFormulation, ::Any, args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Log-barrier formulation for Risk Budgeting.

Uses a logarithmic objective to enforce the risk budget constraints. Can provide an optional orthant vector to allow for negative weights in specific assets.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogRiskBudgeting(;
        z::Option{<:VecInt} = nothing
    ) -> LogRiskBudgeting

Keywords correspond to the struct's fields.

## Validation

  - If `z` is provided: `!isempty(z)` and `all(x -> abs(x) == 1, z)`.

# Related

  - [`RiskBudgetingFormulation`](@ref)
  - [`MixedIntegerRiskBudgeting`](@ref)
"""
@concrete struct LogRiskBudgeting{T} <: RiskBudgetingFormulation
    """
    Optional orthant vector of ±1 defining which assets can have negative weights (`-1`) or must be positive (`+1`). If `nothing`, all assets have positive weights.
    """
    z::T
    function LogRiskBudgeting(z::Option{<:VecInt})
        if !isnothing(z)
            @argcheck(!isempty(z), IsEmptyError("z cannot be empty"))
            @argcheck(all(x -> abs(x) == 1, z),
                      ArgumentError("all elements of z must be ±1"))
        end
        return new{typeof(z)}(z)
    end
end
function LogRiskBudgeting(; z::Option{<:VecInt} = nothing)::LogRiskBudgeting
    return LogRiskBudgeting(z)
end
function port_opt_view(alg::LogRiskBudgeting{Nothing}, i, args...)
    return alg
end
function port_opt_view(alg::LogRiskBudgeting{<:VecInt}, i, args...)
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

## Validation

  - If `rkb` is a `RiskBudgetEstimator`: `!isnothing(sets)`.

# Related

  - [`RiskBudgetingAlgorithm`](@ref)
  - [`FactorRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
"""
@propagatable @concrete struct AssetRiskBudgeting <: RiskBudgetingAlgorithm
    """
    $(field_dict[:rkb])
    """
    @vprop rkb
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:rba])
    """
    @vprop alg
    function AssetRiskBudgeting(rkb::Option{<:RkbE_Rkb}, sets::Option{<:AssetSets},
                                alg::RiskBudgetingFormulation)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
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

## Validation

  - If `rkb` is a `RiskBudgetEstimator`: `!isnothing(sets)`.

# Related

  - [`RiskBudgetingAlgorithm`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`RiskBudgeting`](@ref)
"""
@propagatable @concrete struct FactorRiskBudgeting <: RiskBudgetingAlgorithm
    """
    $(field_dict[:re])
    """
    @vprop re
    """
    $(field_dict[:rkb])
    """
    rkb
    """
    $(field_dict[:sets])
    """
    sets
    """
    $(field_dict[:flag])
    """
    flag
    function FactorRiskBudgeting(re::RegE_Reg, rkb::Option{<:RkbE_Rkb},
                                 sets::Option{<:AssetSets}, flag::Bool)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
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
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`RiskBudgeting`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`RiskBudgeting`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function risk_budgeting_td_defaults()::NamedTuple
    return (; r = Variance(), rba = AssetRiskBudgeting())
end
"""
$(DocStringExtensions.TYPEDEF)

Risk Budgeting (RB) portfolio optimiser.

`RiskBudgeting` allocates portfolio weights so that each asset (or factor) contributes a specified fraction of the total portfolio risk. It uses a logarithmic or mixed-integer formulation and can be combined with any risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskBudgeting(;
        opt::JuMPOptimiser,
        r::TD{<:RM_VecRM} = Variance(),
        rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
        wi::TD_Option{<:VecNum} = nothing,
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> RiskBudgeting

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the risk measure, budgeting algorithm (and with it the risk budget), warm start and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `wi` and `fb`).

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is provided: `!isempty(wi)`.
  - `fb` schedules: `bind !== :nearest`.

# Mathematical definition

Risk budgeting allocates weights so that each asset ``i`` contributes a target fraction ``b_i`` of total portfolio risk:

```math
\\begin{align}
w_i \\frac{\\partial \\rho(\\boldsymbol{w})}{\\partial w_i} &= b_i \\, \\rho(\\boldsymbol{w}), \\quad \\sum_{i=1}^{N} b_i = 1, \\quad b_i \\geq 0\\,.
\\end{align}
```

The logarithmic formulation (`LogRiskBudgeting`) solves the equivalent convex problem:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} \\; \\rho(\\boldsymbol{w}) - \\sum_{i=1}^{N} b_i \\ln w_i \\quad \\text{s.t.} \\quad \\boldsymbol{w} > \\boldsymbol{0}\\,.
\\end{align}
```

Where:

  - ``w_i``: Portfolio weight of asset ``i``.
  - ``\\rho(\\boldsymbol{w})``: Portfolio risk measure.
  - ``b_i``: Risk budget (target risk fraction) for asset ``i``.
  - ``N``: Number of assets.
  - ``\\boldsymbol{w}``: Portfolio weight vector.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`AssetRiskBudgeting`](@ref)
  - [`FactorRiskBudgeting`](@ref)
"""
@propagatable @concrete struct RiskBudgeting <: RiskJuMPOptimisationEstimator
    """
    $(field_dict[:opt_jmp])
    """
    @fprop opt
    """
    $(field_dict[:r_opt])
    """
    @fprop r
    """
    $(field_dict[:rba])
    """
    rba
    """
    $(field_dict[:wi])
    """
    wi
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function RiskBudgeting(opt::JuMPOptimiser, r::TD{<:RM_VecRM},
                           rba::TD{<:RiskBudgetingAlgorithm}, wi::TD_Option{<:VecNum},
                           fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :RiskBudgeting)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        assert_risk_measure_required(r, :RiskBudgeting)
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        assert_time_dependent_substitution(RiskBudgeting, (; opt, r, rba, wi, fb),
                                           risk_budgeting_td_defaults())
        return new{typeof(opt), typeof(r), typeof(rba), typeof(wi), typeof(fb)}(opt, r, rba,
                                                                                wi, fb)
    end
end
function RiskBudgeting(; opt::JuMPOptimiser, r::TD{<:RM_VecRM} = Variance(),
                       rba::TD{<:RiskBudgetingAlgorithm} = AssetRiskBudgeting(),
                       wi::TD_Option{<:VecNum} = nothing,
                       fb::TDO_Option{<:OptE_Opt} = nothing)::RiskBudgeting
    return RiskBudgeting(opt, r, rba, wi, fb)
end
function time_dependent_field_defaults(::RiskBudgeting)::NamedTuple
    return risk_budgeting_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (JuMP optimiser, risk measure, or fallback).
"""
function needs_previous_weights(opt::RiskBudgeting)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`RiskBudgeting`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(rb::RiskBudgeting, i, X::MatNum, args...)::RiskBudgeting
    X = isa(rb.opt.pe, AbstractPriorResult) ? rb.opt.pe.X : X
    opt = port_opt_view(rb.opt, i, X)
    r = port_opt_view(rb.r, i, X)
    rba = port_opt_view(rb.rba, i)
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
    @argcheck(length(rb) == N, DimensionMismatch("rb ($(length(rb))) must match N ($N)"))
    sc = get_constraint_scale(model)
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
    # The log-barrier normalisation above pins the scale, so downstream builders may use 1
    # in place of the free variable `k`.
    set_unit_budget!(model)
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
    rkb = _set_risk_budgeting_constraints!(model, rb, get_w(model); strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, nothing, true)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
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
    @argcheck(length(z) == length(get_w(model)),
              DimensionMismatch("z ($(length(z))) must match w ($(length(get_w(model)))))"))
    w = z .* get_w(model)
    rkb = _set_risk_budgeting_constraints!(model, rb, w; strict = rb.opt.strict)
    sc = get_constraint_scale(model)
    k = get_k(model)
    JuMP.@constraints(model, begin
                          mipcrkb, sc * (sum(w) - k) >= 0
                          orthcrkb, sc * w >= 0
                      end)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:FactorRiskBudgeting, <:Any},
                                         ::Any, wb::WeightBounds, rd::ReturnsResult)
    b1, rr = set_factor_risk_contribution_constraints!(model, rb.rba.re, rd, rb.rba.flag,
                                                       rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w1]; strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedFactorRiskBudgetingAttributes(; rkb = rkb, b1 = b1, rr = rr)
end
"""
    set_rb_mip_w!(model::JuMP.Model, X::MatNum)

Create long and short weight variables for MIP risk budgeting in the JuMP model.

Registers long `lw`, short `sw` weight variables and the derived expressions `w = lw - sw` and `w_obj = lw + sw`.

Because `w` is *derived* from the parts, this declares a [`WeightsFromParts`](@ref) decomposition contract for builders that pin the decomposition.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix (used to determine number of assets).

# Returns

  - `nothing`.

# Related

  - [`set_risk_budgeting_constraints!`](@ref)
  - [`set_decomposition_contract!`](@ref)
  - [`WeightsFromParts`](@ref)
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
    set_decomposition_contract!(model, WeightsFromParts())
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
    w = get_w(model)
    sc = get_constraint_scale(model)
    k = get_k(model)
    JuMP.@constraint(model, mipcrkb, sc * (sum(w) - k) >= 0)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedAssetRiskBudgetingAttributes(; rkb = rkb)
end
function _optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    rb = reset_time_dependent_estimator(rb)
    attrs = processed_jump_optimiser_attributes(rb.opt, rd; dims = dims, kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    prb = set_risk_budgeting_constraints!(model, rb, attrs.pr, attrs.wb, rd)
    assemble_jump_model!(model, rb, rb.opt, attrs, rd, rb.r, MinimumRisk())
    set_portfolio_objective_function!(model, MinimumRisk(), attrs.ret, rb.opt.cobj, rb,
                                      attrs.pr, attrs)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(attrs.pr.X))
    return RiskBudgetingResult(;
                               jr = JuMPOptimisationResult(; pa = attrs, retcode = retcode,
                                                           sol = sol,
                                                           model = ifelse(save, model,
                                                                          nothing)),
                               prb = prb, fb = nothing)
end
"""
    optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> RiskBudgetingResult

Run the Risk Budgeting portfolio optimisation.

# Arguments

  - `rb`: The risk budgeting optimiser to use.
  - $(arg_dict[:rd]) If `isa(rb.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`RiskBudgeting`](@ref)
  - [`RiskBudgetingResult`](@ref)
"""
function optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(rb, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export AssetRiskBudgeting, FactorRiskBudgeting, RiskBudgeting, RiskBudgetingResult,
       LogRiskBudgeting, MixedIntegerRiskBudgeting
