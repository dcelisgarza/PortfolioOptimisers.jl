"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the processed attributes of an optimiser.

A subtype holds the results that one optimiser family makes once for each `optimise` call. The
family gives them to the code that builds its model. Every bundle of processed optimiser
attributes subtypes `ProcessedAttributes`.

# Related

  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`ProcessedRiskBudgetingAttributes`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
abstract type ProcessedAttributes <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Holds the prior result and every processed constraint result that a JuMP model reads.

[`processed_jump_optimiser_attributes`](@ref) makes the bundle once for each `optimise` call,
and [`assemble_jump_model!`](@ref) reads it. So each builder of the model reads a result, and no
builder processes an estimator again.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProcessedJuMPOptimiserAttributes(;
        pr::AbstractPriorResult, wb::Option{<:WeightBounds}, lt::Option{<:Threshold},
        st::Option{<:Threshold}, lcsr::Option{<:Lc_VecLc}, ctr::Option{<:Lc_CC_VecCC},
        gcardr::Option{<:LinearConstraint}, sgcardr::Option{<:Lc_VecLc},
        smtx::Option{<:MatNum_VecMatNum}, sgmtx::Option{<:MatNum_VecMatNum},
        slt::Option{<:Bt_VecOptBt}, sst::Option{<:Bt_VecOptBt}, sglt::Option{<:Bt_VecOptBt},
        sgst::Option{<:Bt_VecOptBt}, tn::Option{<:Tn_VecTn}, fees::Option{<:Fees},
        plr::Option{<:Union{AbstractPhylogenyConstraintResult,
                            AbstractVector{<:AbstractPhylogenyConstraintResult}}},
        ret::JRE_VecJRE, sca::Scalariser, imsk::Option{<:BitVector} = nothing
    ) -> ProcessedJuMPOptimiserAttributes

Keywords correspond to the struct's fields. Each field holds the result of the slot of the
same name in [`JuMPOptimiser`](@ref), and not the estimator. `ret` is the exception, because a
returns estimator has no result form. Make the bundle with
[`processed_jump_optimiser_attributes`](@ref). The constructor checks the types of the fields
and no value, because the processing checked the values.

# Examples

```jldoctest
julia> pr = prior(EmpiricalPrior(),
                  ReturnsResult(; nx = [\"a\", \"b\"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1]));

julia> ProcessedJuMPOptimiserAttributes(; pr = pr, wb = nothing, lt = nothing, st = nothing,
                                        lcsr = nothing, ctr = nothing, gcardr = nothing,
                                        sgcardr = nothing, smtx = nothing, sgmtx = nothing,
                                        slt = nothing, sst = nothing, sglt = nothing,
                                        sgst = nothing, tn = nothing, fees = nothing,
                                        plr = nothing, ret = ArithmeticReturn(),
                                        sca = SumScalariser()) isa ProcessedJuMPOptimiserAttributes
true
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`MeanRisk`](@ref)
"""
@concrete struct ProcessedJuMPOptimiserAttributes <: ProcessedAttributes
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:lt])
    """
    lt
    """
    $(field_dict[:st])
    """
    st
    """
    $(field_dict[:lcsr])
    """
    lcsr
    """
    $(field_dict[:ctr])
    """
    ctr
    """
    $(field_dict[:gcardr])
    """
    gcardr
    """
    $(field_dict[:sgcardr])
    """
    sgcardr
    """
    $(field_dict[:smtx])
    """
    smtx
    """
    $(field_dict[:sgmtx])
    """
    sgmtx
    """
    $(field_dict[:slt])
    """
    slt
    """
    $(field_dict[:sst])
    """
    sst
    """
    $(field_dict[:sglt])
    """
    sglt
    """
    $(field_dict[:sgst])
    """
    sgst
    """
    $(field_dict[:tnr])
    """
    tn
    """
    $(field_dict[:feesr])
    """
    fees
    """
    $(field_dict[:plr])
    """
    plr
    """
    $(field_dict[:ret_jmp])
    """
    ret
    """
    $(field_dict[:sca_res])
    """
    sca
    """
    $(field_dict[:imsk])
    """
    imsk
    function ProcessedJuMPOptimiserAttributes(pr::AbstractPriorResult,
                                              wb::Option{<:WeightBounds},
                                              lt::Option{<:Threshold},
                                              st::Option{<:Threshold},
                                              lcsr::Option{<:Lc_VecLc},
                                              ctr::Option{<:Lc_CC_VecCC},
                                              gcardr::Option{<:LinearConstraint},
                                              sgcardr::Option{<:Lc_VecLc},
                                              smtx::Option{<:MatNum_VecMatNum},
                                              sgmtx::Option{<:MatNum_VecMatNum},
                                              slt::Option{<:Bt_VecOptBt},
                                              sst::Option{<:Bt_VecOptBt},
                                              sglt::Option{<:Bt_VecOptBt},
                                              sgst::Option{<:Bt_VecOptBt},
                                              tn::Option{<:Tn_VecTn}, fees::Option{<:Fees},
                                              plr::Option{<:Union{AbstractPhylogenyConstraintResult,
                                                                  AbstractVector{<:AbstractPhylogenyConstraintResult}}},
                                              ret::JRE_VecJRE, sca::Scalariser,
                                              imsk::Option{<:BitVector})
        return new{typeof(pr), typeof(wb), typeof(lt), typeof(st), typeof(lcsr),
                   typeof(ctr), typeof(gcardr), typeof(sgcardr), typeof(smtx),
                   typeof(sgmtx), typeof(slt), typeof(sst), typeof(sglt), typeof(sgst),
                   typeof(tn), typeof(fees), typeof(plr), typeof(ret), typeof(sca),
                   typeof(imsk)}(pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx,
                                 slt, sst, sglt, sgst, tn, fees, plr, ret, sca, imsk)
    end
end
function ProcessedJuMPOptimiserAttributes(; pr::AbstractPriorResult,
                                          wb::Option{<:WeightBounds},
                                          lt::Option{<:Threshold}, st::Option{<:Threshold},
                                          lcsr::Option{<:Lc_VecLc},
                                          ctr::Option{<:Lc_CC_VecCC},
                                          gcardr::Option{<:LinearConstraint},
                                          sgcardr::Option{<:Lc_VecLc},
                                          smtx::Option{<:MatNum_VecMatNum},
                                          sgmtx::Option{<:MatNum_VecMatNum},
                                          slt::Option{<:Bt_VecOptBt},
                                          sst::Option{<:Bt_VecOptBt},
                                          sglt::Option{<:Bt_VecOptBt},
                                          sgst::Option{<:Bt_VecOptBt},
                                          tn::Option{<:Tn_VecTn}, fees::Option{<:Fees},
                                          plr::Option{<:Union{AbstractPhylogenyConstraintResult,
                                                              AbstractVector{<:AbstractPhylogenyConstraintResult}}},
                                          ret::JRE_VecJRE, sca::Scalariser,
                                          imsk::Option{<:BitVector} = nothing)::ProcessedJuMPOptimiserAttributes
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr,
                                            smtx, sgmtx, slt, sst, sglt, sgst, tn, fees,
                                            plr, ret, sca, imsk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Restricts an optimiser head and its returns data to the assets that an Investable Mask keeps.

[`processed_jump_optimiser_attributes`](@ref) restricts the prior result and every constraint result of the bundle. It cannot restrict the values that the head holds outside the bundle, such as an initial weight vector, a risk measure with data for each asset, a tracking estimator or a custom constraint. These go from the head to [`assemble_jump_model!`](@ref) directly. So each head calls this function on itself and on `rd` before it builds a model.

The function does not report the reduction, because [`investable_reduction`](@ref) reports it once for the bundle.

# Algorithm

 1. When `imsk` is `nothing`, every asset is investable. Return `optimiser` and `rd` unchanged, with no allocation.
 2. Take the returns matrix `X` from `rd.X`, or from `pr.X` when `rd` holds no matrix. When the head holds a fitted prior, [`port_opt_view`](@ref) reads the matrix of that prior and ignores `X`, so the two choices give the same view.
 3. Find the positions `idx = findall(imsk)` of the investable assets. Every other caller of [`port_opt_view`](@ref) passes an index vector, so the function passes one too.
 4. View `optimiser` at `idx` with [`port_opt_view`](@ref). Then declare the names that the mask leaves out on the Non-Investable Axis of the sets that the head holds, with [`non_investable_universe`](@ref). The generic method returns a head with no such sets unchanged. A risk budget keyed by name resolves after this step.
 5. View `rd` at `idx`, and return the two views.

# Arguments

  - `optimiser::JuMPOptimisationEstimator`: The optimiser head to view.
  - $(arg_dict[:rd])
  - $(arg_dict[:pr])
  - $(arg_dict[:imsk])

# Returns

  - `(optimiser, rd)`: Both restricted to the investable assets, or both unchanged.

# Related

  - [`investable_mask`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`port_opt_view`](@ref)
"""
function investable_view(optimiser::JuMPOptimisationEstimator, rd::ReturnsResult,
                         ::AbstractPriorResult, ::Nothing)
    return optimiser, rd
end
function investable_view(optimiser::JuMPOptimisationEstimator, rd::ReturnsResult,
                         pr::AbstractPriorResult, imsk::BitVector)
    X = isnothing(rd.X) ? pr.X : rd.X
    idx = findall(imsk)
    # The head takes the view that `investable_reduction` took of the optimiser for the
    # bundle, so it declares the Non-Investable Axis on the sets that it holds. It reports
    # nothing, because `investable_reduction` reported the departure.
    return non_investable_universe(port_opt_view(optimiser, idx, X),
                                   non_investable_names(rd.nx, imsk)),
           port_opt_view(rd, idx)
end
function expand_investable_weights(::Nothing, sol::JuMPOptSol_VecJuMPOptSol)
    return sol
end
function expand_investable_weights(imsk::BitVector, sol::JuMPOptimisationSolution)
    return JuMPOptimisationSolution(; w = expand_investable_weights(imsk, sol.w))
end
function expand_investable_weights(imsk::BitVector, sol::VecJuMPOptSol)
    return [expand_investable_weights(imsk, s) for s in sol]
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the fields that every JuMP optimisation result shares.

Each concrete JuMP result holds it as its first field, `jr`, as each JuMP optimiser holds a [`JuMPOptimiser`](@ref) as `opt`. The concrete result adds its own fields and the last field `fb`.

The result holds the weights on the full asset universe of the caller. The solver solves over the investable assets alone, and `model` keeps that reduced problem when the head saves the model. The property `w` reads the weights of `sol`, and a vector of solutions gives a vector of weight vectors. A property name that is neither `w` nor a field reads the field of `pa` of the same name.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimisationResult(;
        pa::ProcessedJuMPOptimiserAttributes,
        retcode::OptRetCode_VecOptRetCode,
        sol::JuMPOptSol_VecJuMPOptSol,
        model::Option{<:JuMP.Model}
    ) -> JuMPOptimisationResult

Keywords correspond to the struct's fields. The keyword constructor expands the weights of `sol` from the investable assets to the full asset universe with [`expand_investable_weights`](@ref), at the mask `pa.imsk`. An asset outside the mask gets the weight zero. When `pa.imsk` is `nothing`, `sol` is kept unchanged.

## Validation

  - When `pa.imsk` is a mask, each weight vector of `sol` holds one weight for each `true` entry of the mask. Otherwise the constructor raises a `DimensionMismatch`.

# Related

  - [`BaseJuMPOptimisationResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`set_retcode`](@ref): Rebuilds the result with a new return code, and does not expand the weights again.
"""
@concrete struct JuMPOptimisationResult <: BaseJuMPOptimisationResult
    """
    $(field_dict[:pa])
    """
    pa
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
    # The type is declared here and not beside its supertype `BaseJuMPOptimisationResult`, so
    # that this constructor can bind `ProcessedJuMPOptimiserAttributes` and the solution types.
    function JuMPOptimisationResult(pa::ProcessedJuMPOptimiserAttributes,
                                    retcode::OptRetCode_VecOptRetCode,
                                    sol::JuMPOptSol_VecJuMPOptSol,
                                    model::Option{<:JuMP.Model})
        return new{typeof(pa), typeof(retcode), typeof(sol), typeof(model)}(pa, retcode,
                                                                            sol, model)
    end
end
function JuMPOptimisationResult(; pa::ProcessedJuMPOptimiserAttributes,
                                retcode::OptRetCode_VecOptRetCode,
                                sol::JuMPOptSol_VecJuMPOptSol,
                                model::Option{<:JuMP.Model})::JuMPOptimisationResult
    # Every JuMP family builds its result through this constructor, so the weight expansion
    # is here. `MeanRisk`, `RiskBudgeting`, `RelaxedRiskBudgeting`, `FactorRiskContribution`
    # and `NearOptimalCentering` all call it. The expansion is not in the inner constructor,
    # because a new value of `sol` there makes the inferred field types wider.
    return JuMPOptimisationResult(pa, retcode, expand_investable_weights(pa.imsk, sol),
                                  model)
end
# The JuMP families carry the mask on the processed attribute bundle, and every concrete JuMP
# result embeds the shared core as `jr`, so two methods cover the whole side.
function result_investable_mask(res::JuMPOptimisationResult)
    return res.pa.imsk
end
function result_investable_mask(res::Union{RiskJuMPOptimisationResult,
                                           NonRiskJuMPOptimisationResult})
    return result_investable_mask(res.jr)
end
"""
    set_retcode(res::JuMPOptimisationResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`JuMPOptimisationResult`](@ref) with a different return code.

The rebuild calls the positional constructor, not the keyword constructor. The keyword constructor expands the weights from the investable assets to the full asset universe. The weights of `res.sol` are expanded already, and a second expansion raises a `DimensionMismatch` when the mask leaves out an asset.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Returns

  - [`JuMPOptimisationResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
function set_retcode(res::JuMPOptimisationResult, retcode::OptRetCode_VecOptRetCode)
    return JuMPOptimisationResult(res.pa, retcode, res.sol, res.model)
end
# Virtual property `:w` extracts portfolio weights from `sol` (a single solution or a vector
# of them, hence the broadcast); unknown properties forward to `pa` (see [`@forward_properties`](@ref)).
@forward_properties JuMPOptimisationResult begin
    compute(w, sol.w; broadcast)
    forward(pa)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the static defaults of the [`JuMPOptimiser`](@ref) fields that can hold a [`TimeDependent`](@ref).

The constructor reads them when it checks each entry of a schedule, and [`time_dependent_field_defaults`](@ref) returns them. So the value of a field outside every fold loop is declared once. A field whose static default is `nothing` is not in the tuple.

# Returns

  - `NamedTuple`: The defaults of `pe`, `wb`, `bgt`, `ret` and `sca`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function jump_optimiser_td_defaults()::NamedTuple
    return (; pe = EmpiricalPrior(), wb = WeightBounds(), bgt = 1.0,
            ret = ArithmeticReturn(), sca = SumScalariser())
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the problem definition and the solver settings that every JuMP optimiser shares.

It holds the prior estimator, the solvers, the budgets, the bounds, the constraints, the fees, the tracking and the regularisation. Give it to a JuMP optimiser such as [`MeanRisk`](@ref) or [`RiskBudgeting`](@ref), which adds the objective and the risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimiser(;
        pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
        slv::Slv_VecSlv,
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        bgt::TD_Option{<:Num_BgtCE} = 1.0,
        sbgt::TD_Option{<:Num_BgtRg} = nothing,
        gbgt::TD_Option{<:Num_BgtRg} = nothing,
        lt::TD_Option{<:BtE_Bt} = nothing,
        st::TD_Option{<:BtE_Bt} = nothing,
        lcse::TD_Option{<:EcE_LcE_Lc_VecEcE_LcE_Lc} = nothing,
        cte::TD_Option{<:Lc_CC_VecCC} = nothing,
        gcarde::TD_Option{<:LcE_Lc} = nothing,
        sgcarde::TD_Option{<:LcE_Lc_VecLcE_Lc} = nothing,
        smtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
        sgmtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
        slt::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sst::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sglt::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sgst::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        tn::TD_Option{<:TnE_Tn_VecTnE_Tn} = nothing,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        tr::TD_Option{<:Tr_VecTr} = nothing,
        ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
        ret::TD{<:JRE_VecJRE} = ArithmeticReturn(),
        sca::TD{<:NonHierarchicalScalariser} = SumScalariser(),
        ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr} = nothing,
        cobj::TD_Option{<:JuMPObj_VecJuMPObj} = nothing,
        sc::Number = 1,
        so::Number = 1,
        ss::TD_Option{<:Number} = nothing,
        card::TD_Option{<:Integer} = nothing,
        scard::TD_Option{<:Int_VecInt} = nothing,
        l2c::TD_Option{<:Num_NormCeilCal} = nothing,
        lpc::TD_Option{<:LpReg_VecLpReg} = nothing,
        linfc::TD_Option{<:Num_NormCeilCal} = nothing,
        l1::TD_Option{<:Num_AmbRadCal} = nothing,
        l2::TD_Option{<:L2Reg_VecL2Reg} = nothing,
        lp::TD_Option{<:LpReg_VecLpReg} = nothing,
        linf::TD_Option{<:Num_AmbRadCal} = nothing,
        xbgt::Bool = false,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing,
    ) -> JuMPOptimiser

Keywords correspond to the struct's fields. A field typed [`TD_Option`](@ref) or [`TD`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value. A cross-validation fold loop resolves the schedule at each fold, and an `optimise` call outside a fold loop reads the static default of the field. So the prior estimator, the returns model, the scalariser, the asset sets and the constraints can change from fold to fold. The execution settings `slv`, `sc`, `so` and `strict` are static.

## Validation

  - If `slv` is a vector: `!isempty(slv)`.
  - If `bgt` is a number: `isfinite(bgt)`.
  - If `bgt` is a [`BudgetCostEstimator`](@ref): `isnothing(sbgt)`. Otherwise the constructor raises a `ConflictingArgumentError`.
  - If `sbgt` or `gbgt` is a number: it is finite and `>= 0`.
  - If `gbgt` is not `nothing`: `bgt` and `sbgt` are not both numbers, see [`assert_gross_budget_admissible`](@ref). Two numbers fix the gross budget of the model at `bgt + 2 * sbgt`, which is the row that `gbgt` writes. The realised gross exposure `norm(w, 1)` equals this budget when `xbgt` is `true`, and it can be smaller when `xbgt` is `false`.
  - If `gbgt` is not `nothing` and `wb` is a [`WeightBounds`](@ref): `wb` admits a negative weight. The check of a [`WeightBoundsEstimator`](@ref) waits for the solve.
  - If `cte`, `tn`, `tr`, `l2`, `lp` or `lpc` is a vector: `!isempty`. An empty `l2`, `lp` or `lpc` builds no term, which is what `nothing` states.
  - If `card`, `l2c`, `linfc`, `l1` or `linf` is a number: it is finite and `> 0`. `l1` and `linf` also take an ambiguity-radius rule, and `l2c` and `linfc` a norm-ceiling rule. A rule states no number at construction, so [`assemble_jump_model!`](@ref) checks the number that the rule returns.
  - The rule in each [`LpRegularisation`](@ref) agrees with the field that holds it. `lp` is a penalty, so it refuses a norm-ceiling rule, and `lpc` is a constraint, so it refuses an ambiguity-radius rule. The term has one field for both readings, so the constructor of the optimiser is the first point that knows the reading. The constructors of [`L2Regularisation`](@ref) and [`LpRegularisation`](@ref) check the other values of `l2`, `lp` and `lpc`.
  - The shapes of `scard`, `smtx`, `slt` and `sst` agree, see [`assert_subgroup_mip_fields`](@ref). The shapes of `sgcarde`, `sgmtx`, `sglt` and `sgst` agree, see [`assert_subgrouped_mip_fields`](@ref).
  - If a slot holds an estimator keyed by name ([`name_keyed`](@ref)), or `fees` is a [`FeesEstimator`](@ref): `!isnothing(sets)`.
  - If `tr` holds a [`DataBenchmarkTracking`](@ref): `pe` does not state `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).
  - If a field holds a [`TimeDependent`](@ref): the constructor runs again once for each entry of the schedule, with the other schedules at their static defaults, see [`assert_time_dependent_substitution`](@ref). So a type error or a conflict between two fields shows at construction. A check that couples a schedule to a static field runs again when each fold rebuilds the optimiser.

A value outside its bound raises an `IsEmptyError`, an `IsNonFiniteError`, a `DomainError`, an `IsNothingError`, an `ArgumentError` or a `DimensionMismatch`, as the helper that checks it states.

# Related

  - [`BaseJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
"""
@concrete struct JuMPOptimiser <: BaseJuMPOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:wb_jmp])
    """
    wb
    """
    $(field_dict[:bgt])
    """
    bgt
    """
    $(field_dict[:sbgt])
    """
    sbgt
    """
    $(field_dict[:gbgt])
    """
    gbgt
    """
    $(field_dict[:lt])
    """
    lt
    """
    $(field_dict[:st])
    """
    st
    """
    $(field_dict[:lcse])
    """
    lcse
    """
    $(field_dict[:cte_jmp])
    """
    cte
    """
    $(field_dict[:gcarde])
    """
    gcarde
    """
    $(field_dict[:sgcarde])
    """
    sgcarde
    """
    $(field_dict[:smtx])
    """
    smtx
    """
    $(field_dict[:sgmtx])
    """
    sgmtx
    """
    $(field_dict[:slt])
    """
    slt
    """
    $(field_dict[:sst])
    """
    sst
    """
    $(field_dict[:sglt])
    """
    sglt
    """
    $(field_dict[:sgst])
    """
    sgst
    """
    $(field_dict[:tn_jmp])
    """
    tn
    """
    $(field_dict[:fees_jmp])
    """
    fees
    """
    $(field_dict[:sets])
    """
    sets
    """
    $(field_dict[:tr_jmp])
    """
    tr
    """
    $(field_dict[:ple_jmp])
    """
    ple
    """
    $(field_dict[:ret_jmp])
    """
    ret
    """
    $(field_dict[:sca])
    """
    sca
    """
    $(field_dict[:ccnt])
    """
    ccnt
    """
    $(field_dict[:cobj])
    """
    cobj
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:ss])
    """
    ss
    """
    $(field_dict[:card])
    """
    card
    """
    $(field_dict[:scard])
    """
    scard
    """
    $(field_dict[:l2c])
    """
    l2c
    """
    $(field_dict[:lpc])
    """
    lpc
    """
    $(field_dict[:linfc])
    """
    linfc
    """
    $(field_dict[:l1])
    """
    l1
    """
    $(field_dict[:l2])
    """
    l2
    """
    $(field_dict[:lp])
    """
    lp
    """
    $(field_dict[:linf])
    """
    linf
    """
    $(field_dict[:xbgt])
    """
    xbgt
    """
    $(field_dict[:strict_opt]) Under `true` the fit also refuses an asset with a gap in the rows that a plain graph reader reads, which it otherwise drops with a warning, see [`graph_reader_mask`](@ref).
    """
    strict
    """
    $(field_dict[:cache_opt])
    """
    cache
    function JuMPOptimiser(pe::Onl{<:TD{<:PrE_Pr}}, slv::Slv_VecSlv,
                           wb::TD_Option{<:WbE_Wb}, bgt::TD_Option{<:Num_BgtCE},
                           sbgt::TD_Option{<:Num_BgtRg}, gbgt::TD_Option{<:Num_BgtRg},
                           lt::TD_Option{<:BtE_Bt}, st::TD_Option{<:BtE_Bt},
                           lcse::TD_Option{<:EcE_LcE_Lc_VecEcE_LcE_Lc},
                           cte::TD_Option{<:Lc_CC_VecCC}, gcarde::TD_Option{<:LcE_Lc},
                           sgcarde::TD_Option{<:LcE_Lc_VecLcE_Lc},
                           smtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           sgmtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           slt::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           sst::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           sglt::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           sgst::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           tn::TD_Option{<:TnE_Tn_VecTnE_Tn}, fees::TD_Option{<:FeesE_Fees},
                           sets::TD_Option{<:UniverseSets}, tr::TD_Option{<:Tr_VecTr},
                           ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC}, ret::TD{<:JRE_VecJRE},
                           sca::TD{<:NonHierarchicalScalariser},
                           ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr},
                           cobj::TD_Option{<:JuMPObj_VecJuMPObj}, sc::Number, so::Number,
                           ss::TD_Option{<:Number}, card::TD_Option{<:Integer},
                           scard::TD_Option{<:Int_VecInt},
                           l2c::TD_Option{<:Num_NormCeilCal},
                           lpc::TD_Option{<:LpReg_VecLpReg},
                           linfc::TD_Option{<:Num_NormCeilCal},
                           l1::TD_Option{<:Num_AmbRadCal}, l2::TD_Option{<:L2Reg_VecL2Reg},
                           lp::TD_Option{<:LpReg_VecLpReg},
                           linf::TD_Option{<:Num_AmbRadCal}, xbgt::Bool, strict::Bool,
                           cache::Option{<:ReturnsBufferState})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        if isa(bgt, Number)
            @argcheck(isfinite(bgt), IsNonFiniteError("bgt must be finite, got $bgt"))
        elseif isa(bgt, BudgetCostEstimator)
            @argcheck(isnothing(sbgt),
                      ConflictingArgumentError("sbgt must be nothing when bgt is a BudgetCostEstimator, got sbgt = $sbgt"))
        end
        if isa(sbgt, Number)
            assert_nonempty_nonneg_finite_val(sbgt, :sbgt)
        end
        if isa(gbgt, Number)
            assert_nonempty_nonneg_finite_val(gbgt, :gbgt)
        end
        assert_gross_budget_admissible(bgt, sbgt, gbgt, wb)
        # `map` compiles one call for each slot, so the slots of different types are not boxed.
        map(((cte, :cte), (tn, :tn), (tr, :tr), (l2, :l2), (lp, :lp), (lpc, :lpc))) do (x,
                                                                                        name)
            if isa(x, AbstractVector)
                @argcheck(!isempty(x), IsEmptyError("$name cannot be empty"))
            end
            return nothing
        end
        assert_data_benchmark_prior(tr, pe, :tr, :JuMPOptimiser)
        # The method for a value other than a number checks nothing, so `nothing`, a schedule
        # and a calibration rule pass these calls.
        assert_nonempty_gt0_finite_val(card, :card)
        assert_nonempty_gt0_finite_val(l2c, :l2c)
        assert_nonempty_gt0_finite_val(linfc, :linfc)
        assert_nonempty_gt0_finite_val(l1, :l1)
        assert_nonempty_gt0_finite_val(linf, :linf)
        # `LpRegularisation.val` is read as a coefficient in `lp` and as a ceiling in
        # `lpc`, and one field cannot carry two bounds. This is the first point at which
        # the reading is known, so it is where the wrong role is refused. A `TimeDependent`
        # needs no guard: it is not a term, so it meets the permissive fallback, and the
        # constructor test-substitutes each of its entries through this same check.
        assert_penalty_coefficient_role(lp)
        assert_norm_ceiling_role(lpc)
        assert_subgroup_mip_fields(scard, smtx, slt, sst)
        assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)
        if isa(fees, FeesEstimator) || any(name_keyed,
                                           (wb, lt, st, lcse, gcarde, sgcarde, smtx, sgmtx, slt, sst, sglt, sgst, tn))
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when estimator-type fields are provided"))
        end
        assert_time_dependent_substitution(JuMPOptimiser,
                                           (; pe, slv, wb, bgt, sbgt, gbgt, lt, st, lcse,
                                            cte, gcarde, sgcarde, smtx, sgmtx, slt, sst,
                                            sglt, sgst, tn, fees, sets, tr, ple, ret, sca,
                                            ccnt, cobj, sc, so, ss, card, scard, l2c, lpc,
                                            linfc, l1, l2, lp, linf, xbgt, strict),
                                           jump_optimiser_td_defaults())
        return new{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                   typeof(gbgt), typeof(lt), typeof(st), typeof(lcse), typeof(cte),
                   typeof(gcarde), typeof(sgcarde), typeof(smtx), typeof(sgmtx),
                   typeof(slt), typeof(sst), typeof(sglt), typeof(sgst), typeof(tn),
                   typeof(fees), typeof(sets), typeof(tr), typeof(ple), typeof(ret),
                   typeof(sca), typeof(ccnt), typeof(cobj), typeof(sc), typeof(so),
                   typeof(ss), typeof(card), typeof(scard), typeof(l2c), typeof(lpc),
                   typeof(linfc), typeof(l1), typeof(l2), typeof(lp), typeof(linf),
                   typeof(xbgt), typeof(strict), typeof(cache)}(pe, slv, wb, bgt, sbgt,
                                                                gbgt, lt, st, lcse, cte,
                                                                gcarde, sgcarde, smtx,
                                                                sgmtx, slt, sst, sglt, sgst,
                                                                tn, fees, sets, tr, ple,
                                                                ret, sca, ccnt, cobj, sc,
                                                                so, ss, card, scard, l2c,
                                                                lpc, linfc, l1, l2, lp,
                                                                linf, xbgt, strict, cache)
    end
end
function JuMPOptimiser(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(), slv::Slv_VecSlv,
                       wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                       bgt::TD_Option{<:Num_BgtCE} = 1.0,
                       sbgt::TD_Option{<:Num_BgtRg} = nothing,
                       gbgt::TD_Option{<:Num_BgtRg} = nothing,
                       lt::TD_Option{<:BtE_Bt} = nothing, st::TD_Option{<:BtE_Bt} = nothing,
                       lcse::TD_Option{<:EcE_LcE_Lc_VecEcE_LcE_Lc} = nothing,
                       cte::TD_Option{<:Lc_CC_VecCC} = nothing,
                       gcarde::TD_Option{<:LcE_Lc} = nothing,
                       sgcarde::TD_Option{<:LcE_Lc_VecLcE_Lc} = nothing,
                       smtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                       sgmtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                       slt::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sst::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sglt::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sgst::TD_Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       tn::TD_Option{<:TnE_Tn_VecTnE_Tn} = nothing,
                       fees::TD_Option{<:FeesE_Fees} = nothing,
                       sets::TD_Option{<:UniverseSets} = nothing,
                       tr::TD_Option{<:Tr_VecTr} = nothing,
                       ple::TD_Option{<:PlCE_PlC_VecPlCE_PlC} = nothing,
                       ret::TD{<:JRE_VecJRE} = ArithmeticReturn(),
                       sca::TD{<:NonHierarchicalScalariser} = SumScalariser(),
                       ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr} = nothing,
                       cobj::TD_Option{<:JuMPObj_VecJuMPObj} = nothing, sc::Number = 1,
                       so::Number = 1, ss::TD_Option{<:Number} = nothing,
                       card::TD_Option{<:Integer} = nothing,
                       scard::TD_Option{<:Int_VecInt} = nothing,
                       l2c::TD_Option{<:Num_NormCeilCal} = nothing,
                       lpc::TD_Option{<:LpReg_VecLpReg} = nothing,
                       linfc::TD_Option{<:Num_NormCeilCal} = nothing,
                       l1::TD_Option{<:Num_AmbRadCal} = nothing,
                       l2::TD_Option{<:L2Reg_VecL2Reg} = nothing,
                       lp::TD_Option{<:LpReg_VecLpReg} = nothing,
                       linf::TD_Option{<:Num_AmbRadCal} = nothing, xbgt::Bool = false,
                       strict::Bool = false,
                       cache::Option{<:ReturnsBufferState} = nothing)::JuMPOptimiser
    return JuMPOptimiser(pe, slv, wb, bgt, sbgt, gbgt, lt, st, lcse, cte, gcarde, sgcarde,
                         smtx, sgmtx, slt, sst, sglt, sgst, tn, fees, sets, tr, ple, ret,
                         sca, ccnt, cobj, sc, so, ss, card, scard, l2c, lpc, linfc, l1, l2,
                         lp, linf, xbgt, strict, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns `true` if a field of `opt` reads the portfolio weights of the previous fold.

The function asks `tn`, `fees`, `tr`, `ccnt` and `cobj`. It also asks every field that holds a [`TimeDependent`](@ref) that the fold loop resolves, see [`time_dependent_fields`](@ref). A schedule reads the previous weights when it holds a [`PreviousWeightsFunction`](@ref) or an entry that reads them.

# Arguments

  - `opt::JuMPOptimiser`: The JuMP optimiser.

# Returns

  - `Bool`: `true` if one of these fields reads the previous weights, and `false` otherwise.

# Examples

```jldoctest
julia> PortfolioOptimisers.needs_previous_weights(JuMPOptimiser(;
                                                                slv = Solver(; solver = nothing)))
false
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(opt::JuMPOptimiser)
    return (needs_previous_weights(opt.tn) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.tr) ||
            needs_previous_weights(opt.ccnt) ||
            needs_previous_weights(opt.cobj) ||
            any(f -> needs_previous_weights(getfield(opt, f)), time_dependent_fields(opt)))
end
function time_dependent_field_defaults(::JuMPOptimiser)::NamedTuple
    return jump_optimiser_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `opt` whose fields that read previous weights hold the weights `w`.

The function calls `factory(x, w)` on `tn`, `fees`, `tr`, `ccnt` and `cobj`, and it keeps every other field. A [`TimeDependent`](@ref) in one of these fields applies `factory` to each of its entries. The keyword constructor of [`JuMPOptimiser`](@ref) checks the new fields.

# Arguments

  - `opt::JuMPOptimiser`: The JuMP optimiser.
  - `w::AbstractVector`: The new portfolio weights.

# Returns

  - `JuMPOptimiser`: The optimiser with the five fields rebuilt for `w`.

# Examples

```jldoctest
julia> opt = JuMPOptimiser(; slv = Solver(; solver = nothing));

julia> PortfolioOptimisers.factory(opt, fill(0.1, 10)) isa JuMPOptimiser
true
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`factory`](@ref)
"""
function factory(opt::JuMPOptimiser, w::AbstractVector)::JuMPOptimiser
    return rebuild_estimator(opt,
                             (; tn = factory(opt.tn, w), fees = factory(opt.fees, w),
                              tr = factory(opt.tr, w), ccnt = factory(opt.ccnt, w),
                              cobj = factory(opt.cobj, w)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a copy of `opt` restricted to the assets at the indices `i`.

A hierarchical optimiser calls it to build the optimiser of one cluster, and the entry of a fit calls it to restrict the optimiser to the Investable Mask.

# Algorithm

 1. When `opt.pe` is a prior result, replace `X` with `opt.pe.X`.
 2. View at `i` the fields that can hold a value for each asset: `pe`, `wb`, `bgt`, `lt`, `st`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn`, `lcse`, `sets`, `ret`, `ccnt`, `cobj` and `cache`. When `smtx` and `sgmtx` hold the same object, view it once, so that the two views are one object too. The same holds for `slt` and `sglt`, and for `sst` and `sgst`.
 3. View `fees` and `tr` at `i` with `X`. A fee finds the assets outside `i`, which its liquidation charges read, from the width of `X`. A tracking measure gives `X` to the risk measure that it tracks.
 4. Keep every other field. An estimator in `cte`, `gcarde` or `sgcarde` resolves its names against the viewed `sets`. A precomputed constraint in one of them keeps its width, because no view can narrow its matrix, see `port_opt_view(::LinearConstraint, i)`. `ple` reads the viewed prior. The other fields hold no value for each asset.
 5. Rebuild the optimiser with its keyword constructor, which checks the viewed fields again.

# Arguments

  - `opt::JuMPOptimiser`: The JuMP optimiser.
  - `i`: The indices of the assets that the view keeps.
  - `X::MatNum`: The returns matrix of the full universe, with one column for each asset. A prior result in `opt.pe` gives its own `X` in its place.

# Returns

  - `JuMPOptimiser`: The optimiser restricted to the assets at `i`.

# Examples

```jldoctest
julia> opt = JuMPOptimiser(; slv = Solver(; solver = nothing));

julia> X = rand(50, 5);

julia> PortfolioOptimisers.port_opt_view(opt, 1:3, X) isa JuMPOptimiser
true
```

# Related

  - [`JuMPOptimiser`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(opt::JuMPOptimiser, i, X::MatNum, args...)::JuMPOptimiser
    X = isa(opt.pe, AbstractPriorResult) ? opt.pe.X : X
    pe = port_opt_view(opt.pe, i)
    wb = port_opt_view(opt.wb, i)
    bgt = port_opt_view(opt.bgt, i)
    lt = port_opt_view(opt.lt, i)
    st = port_opt_view(opt.st, i)
    if opt.smtx === opt.sgmtx
        smtx = sgmtx = port_opt_view(opt.smtx, i)
    else
        smtx = port_opt_view(opt.smtx, i)
        sgmtx = port_opt_view(opt.sgmtx, i)
    end
    if opt.slt === opt.sglt
        slt = sglt = port_opt_view(opt.slt, i)
    else
        slt = port_opt_view(opt.slt, i)
        sglt = port_opt_view(opt.sglt, i)
    end
    if opt.sst === opt.sgst
        sst = sgst = port_opt_view(opt.sst, i)
    else
        sst = port_opt_view(opt.sst, i)
        sgst = port_opt_view(opt.sgst, i)
    end
    tn = port_opt_view(opt.tn, i)
    lcse = port_opt_view(opt.lcse, i)
    sets = port_opt_view(opt.sets, i)
    # A fee has two axes. Its five fields for each asset are on the kept assets, and its two
    # liquidation charges are on the other assets. Its view finds the other assets from the
    # width of the unreduced `X`, so it is the one constraint that needs the matrix.
    fees = port_opt_view(opt.fees, i, X)
    tr = port_opt_view(opt.tr, i, X)
    ret = port_opt_view(opt.ret, i)
    ccnt = port_opt_view(opt.ccnt, i)
    cobj = port_opt_view(opt.cobj, i)
    cache = port_opt_view(opt.cache, i)
    return rebuild_estimator(opt,
                             (; pe, wb, bgt, lt, st, lcse, smtx, sgmtx, slt, sst, sglt,
                              sgst, tn, fees, sets, tr, ret, ccnt, cobj, cache))
end
function non_investable_universe(opt::JuMPOptimiser, ni::VecStr)::JuMPOptimiser
    return rebuild_estimator(opt, (; sets = non_investable_sets(opt.sets, ni)))
end

export ProcessedJuMPOptimiserAttributes, JuMPOptimiser
