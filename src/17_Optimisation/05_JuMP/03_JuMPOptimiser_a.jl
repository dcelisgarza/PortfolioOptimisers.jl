"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for processed optimiser attributes.

A subtype is the flat bundle of results one optimiser family produces once per `optimise`
call and hands to its model-assembly pipeline. Every collection of processed optimiser
attributes subtypes `ProcessedAttributes`.

# Related

  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`ProcessedRiskBudgetingAttributes`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
abstract type ProcessedAttributes <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Flat bundle of all processed constraint and prior results consumed by
[`assemble_jump_model!`](@ref).

Produced once per `optimise` call by [`processed_jump_optimiser_attributes`](@ref) and
passed directly to the model-assembly pipeline, so every builder reads already-resolved
results rather than re-processing estimators.

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
        plr::Option{<:Union{<:AbstractPhylogenyConstraintResult,
                            <:AbstractVector{<:AbstractPhylogenyConstraintResult}}},
        ret::JRE_VecJRE, sca::Scalariser, imsk::Option{<:BitVector} = nothing
    ) -> ProcessedJuMPOptimiserAttributes

Keywords correspond to the struct's fields. The field types are the *result* side of the
matching [`JuMPOptimiser`](@ref) estimator slots: this bundle holds the constraint and
prior results produced by [`processed_jump_optimiser_attributes`](@ref), not the raw
estimators (`ret` aside — a returns estimator has no separate result form). In practice,
construct via [`processed_jump_optimiser_attributes`](@ref) rather than directly.

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
    # Field types are the *result* side of each matching `JuMPOptimiser` estimator slot:
    # this bundle holds the constraint/prior results produced by
    # `processed_jump_optimiser_attributes` — never the raw estimators. `ret` is the sole
    # exception: a returns estimator has no separate result form. Processing has already
    # validated the contents, so this bundle only type-gates.
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
                                              plr::Option{<:Union{<:AbstractPhylogenyConstraintResult,
                                                                  <:AbstractVector{<:AbstractPhylogenyConstraintResult}}},
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
                                          plr::Option{<:Union{<:AbstractPhylogenyConstraintResult,
                                                              <:AbstractVector{<:AbstractPhylogenyConstraintResult}}},
                                          ret::JRE_VecJRE, sca::Scalariser,
                                          imsk::Option{<:BitVector} = nothing)::ProcessedJuMPOptimiserAttributes
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr,
                                            smtx, sgmtx, slt, sst, sglt, sgst, tn, fees,
                                            plr, ret, sca, imsk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reduce an optimisation estimator and its returns data to the assets an Investable Mask keeps.

[`processed_jump_optimiser_attributes`](@ref) reduces what the bundle carries — the prior result and every constraint result. It cannot reduce what the *head* carries: an initial weight vector, a risk measure holding per-asset data, a tracking estimator, a custom constraint. Those travel from the head into [`assemble_jump_model!`](@ref) unmediated by the bundle, so each head takes this view of itself and of `rd` before it assembles a model.

The `nothing` method is the whole all-investable path: it returns both arguments unchanged, so a universe with nothing to exclude allocates nothing and takes the route it took before the mask existed.

The returns matrix the view slices tracking against is `rd.X`, and `pr.X` when the caller stated a fitted prior instead of data. [`port_opt_view`](@ref) reads the prior's own matrix in that case and ignores what it is given, so either is correct and only one of them always exists.

The mask is handed to [`port_opt_view`](@ref) as the index vector `findall(imsk)` rather than as the mask itself. Every other caller of that verb passes an integer index, and a view specialised on one index type is a view whose inference is already exercised.

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
    # The head takes the same view of itself that the bundle's door took of the optimiser,
    # so it declares the Non-Investable Axis on whatever sets it carries — a risk budget
    # keyed by name is resolved from here, after this view. It stays quiet: the door has
    # already announced the departure, and one event is reported once.
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

Shared field core for JuMP-based optimisation results.

Holds the fields common to every JuMP optimisation result. Embedded as the first field (`jr`) of each concrete JuMP result, analogous to how [`JuMPOptimiser`](@ref) is embedded as `opt` in each JuMP optimiser. The concrete result keeps only its unique fields plus the trailing `fb`.

Defined here (rather than in `01_Base_JuMPOptimisation.jl`, where its [`BaseJuMPOptimisationResult`](@ref) supertype lives) so its typed constructor can bind `pa::ProcessedJuMPOptimiserAttributes` and `sol::JuMPOptimisationSolution`, both in scope at this point in load order.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimisationResult(;
        pa::ProcessedJuMPOptimiserAttributes,
        retcode::OptRetCode_VecOptRetCode,
        sol::JuMPOptSol_VecJuMPOptSol,
        model::Option{<:JuMP.Model}
    ) -> JuMPOptimisationResult

Keywords correspond to the struct's fields.

# Related

  - [`BaseJuMPOptimisationResult`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimiser`](@ref)
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
    # The one door every JuMP family's result comes through, and so the one place the weight
    # expansion belongs: `MeanRisk`, `RiskBudgeting`, `RelaxedRiskBudgeting`,
    # `FactorRiskContribution` and `NearOptimalCentering` all build their result here. `sol`
    # arrives holding the reduced vector the solver returned, and the result carries that
    # vector on the caller's own universe; the reduced problem survives in `model` when the
    # head was asked to save it. It sits here rather than in the inner constructor because
    # the inner one states the field types of `new`, and a reassignment there widens what
    # inference knows about `sol` and `model` both.
    return JuMPOptimisationResult(pa, retcode, expand_investable_weights(pa.imsk, sol),
                                  model)
end
# The JuMP families carry the mask on the processed attribute bundle, and every concrete JuMP
# result embeds the shared core as `jr`, so two methods cover the whole side.
function result_investable_mask(res::JuMPOptimisationResult)
    return res.pa.imsk
end
function result_investable_mask(res::Union{<:RiskJuMPOptimisationResult,
                                           <:NonRiskJuMPOptimisationResult})
    return result_investable_mask(res.jr)
end
"""
    set_retcode(res::JuMPOptimisationResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`JuMPOptimisationResult`](@ref) with a different return code.

The rebuild reaches the inner constructor rather than the keyword one, because the keyword constructor expands the solver's reduced weight vector onto the caller's universe. `sol` is already expanded here, so a second pass through that door would expand it twice.

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

Assert that `val` is finite and positive; throw an `ArgCheck` error otherwise.

Accepts a scalar `Number` or a `VecNum`. The scalar overload requires `isfinite(val)` and
`val > 0`. The vector overload is weaker: it requires at least one finite element, at least
one strictly positive element, and no negative element, so a vector carrying zeros passes
where the scalar `0` does not.

# Arguments

  - `val`: Scalar or vector to validate.

# Returns

  - `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.assert_finite_nonnegative_real_or_vec(1.0)

julia> PortfolioOptimisers.assert_finite_nonnegative_real_or_vec([0.5, 1.0])

```

# Related

  - [`JuMPOptimiser`](@ref)
"""
function assert_finite_nonnegative_real_or_vec(val::Number)::Nothing
    @argcheck(isfinite(val), IsNonFiniteError("val must be finite, got $val"))
    @argcheck(val > zero(val), DomainError(val, "val must be > 0"))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::VecNum)::Nothing
    @argcheck(any(isfinite, val),
              IsNonFiniteError("val must contain at least one finite element, got $val"))
    @argcheck(any(x -> x > zero(x), val),
              DomainError(val, "val must contain at least one positive element"))
    @argcheck(all(x -> zero(x) <= x, val),
              DomainError(val, "all elements of val must be >= 0"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`JuMPOptimiser`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

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

Main JuMP-based portfolio optimiser configuration.

`JuMPOptimiser` collects all the inputs needed to formulate and solve a JuMP-based portfolio optimisation problem: prior estimator, solver, constraints, bounds, fees, tracking, regularisation, and more. It is intended to be passed to a higher-level optimiser such as [`MeanRisk`](@ref) or [`RiskBudgeting`](@ref).

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
        xbgt::Bool = false,
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
        brt::Bool = false,
        x_src::Symbol = :prior,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing,
    ) -> JuMPOptimiser

Keywords correspond to the struct's fields. Fields typed [`TD_Option`](@ref) or [`TD`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value; a cross-validation fold loop resolves it per fold, and a fold-less `optimise` runs with the field at its static default. The problem definition — the prior estimator, returns model, scalariser and asset sets as much as the constraints — may therefore vary over folds; execution control (`slv`, `sc`, `so`, `brt`, `x_src`, `strict`) stays static.

## Validation

  - `x_src in (:prior, :data)`.
  - If `slv` is a vector: `!isempty(slv)`.
  - If `bgt` is a number: `isfinite(bgt)`.
  - If `bgt` is a `BudgetCostEstimator`: `isnothing(sbgt)`.
  - If `sbgt` is a number: `isfinite(sbgt)` and `sbgt >= 0`.
  - If `gbgt` is a number: `isfinite(gbgt)` and `gbgt >= 0`.
  - If `gbgt` is provided: `wb` must admit short positions, and `bgt` and `sbgt` must not *both* be pinned numbers (which already determine the gross exposure as `bgt + 2 * sbgt`).
  - If `cte` is a vector: `!isempty(cte)`.
  - If `card` is provided: `card > 0` and finite.
  - If `tn` or `tr` is a vector: each must be non-empty.
  - If `l2c`, `linfc`, `l1`, or `linf` is provided as a number: each must be `> 0` and finite. `l1` and `linf` also take an ambiguity-radius rule, and `l2c` and `linfc` a norm-ceiling rule. A rule states no number here, so the check runs on the number the rule returns, in [`assemble_jump_model!`](@ref).
  - The rule in each [`LpRegularisation`](@ref) is checked against the field that holds it: `lp` is a penalty, so it refuses a norm-ceiling rule, and `lpc` is a constraint, so it refuses an ambiguity-radius rule. The term itself carries one bound for both readings, so this is the point at which the reading is known.
  - If `l2`, `lp` or `lpc` is a vector: each must be non-empty. An empty vector builds no term, which is what `nothing` already spells.
  - `l2`, `lp` and `lpc` are validated by their own estimator constructors ([`L2Regularisation`](@ref), [`LpRegularisation`](@ref)).
  - If `scard` is provided: compatible `smtx`, `slt`, `sst` sizes required.
  - If `sgcarde` is provided: compatible `sgmtx`, `sglt`, `sgst` sizes required.
  - If any estimator-type field (`wb`, `lt`, `fees`, etc.) is provided: `!isnothing(sets)`.
  - If any field holds a [`TimeDependent`](@ref): every vector entry is test-substituted through this constructor (with the other time-dependent fields at their static defaults) so type and cross-field compatibility errors surface immediately. Validation coupling a time-dependent field to static fields is deferred to the per-fold rebuild.

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
    $(field_dict[:xbgt])
    """
    xbgt
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
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:x_src])
    """
    x_src
    """
    $(field_dict[:strict_opt])
    """
    strict
    """
    $(field_dict[:cache_opt])
    """
    cache
    function JuMPOptimiser(pe::Onl{<:TD{<:PrE_Pr}}, slv::Slv_VecSlv,
                           wb::TD_Option{<:WbE_Wb}, bgt::TD_Option{<:Num_BgtCE},
                           sbgt::TD_Option{<:Num_BgtRg}, gbgt::TD_Option{<:Num_BgtRg},
                           xbgt::Bool, lt::TD_Option{<:BtE_Bt}, st::TD_Option{<:BtE_Bt},
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
                           linf::TD_Option{<:Num_AmbRadCal}, brt::Bool, x_src::Symbol,
                           strict::Bool, cache::Option{<:ReturnsBufferState})
        assert_source_selector(x_src, :x_src)
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
        if isa(cte, AbstractVector)
            @argcheck(!isempty(cte), IsEmptyError("cte cannot be empty"))
        end
        if !isnothing(card) && !isa(card, TimeDependent)
            assert_nonempty_gt0_finite_val(card, :card)
        end
        if isa(tn, AbstractVector)
            @argcheck(!isempty(tn), IsEmptyError("tn cannot be empty"))
        end
        if isa(tr, AbstractVector)
            @argcheck(!isempty(tr), IsEmptyError("tr cannot be empty"))
        end
        if !isnothing(l2c) && !isa(l2c, TimeDependent)
            assert_nonempty_gt0_finite_val(l2c, :l2c)
        end
        if !isnothing(linfc) && !isa(linfc, TimeDependent)
            assert_nonempty_gt0_finite_val(linfc, :linfc)
        end
        if !isnothing(l1) && !isa(l1, TimeDependent)
            assert_nonempty_gt0_finite_val(l1, :l1)
        end
        if isa(l2, AbstractVector)
            @argcheck(!isempty(l2), IsEmptyError("l2 cannot be empty"))
        end
        if isa(lp, AbstractVector)
            @argcheck(!isempty(lp), IsEmptyError("lp cannot be empty"))
        end
        if isa(lpc, AbstractVector)
            @argcheck(!isempty(lpc), IsEmptyError("lpc cannot be empty"))
        end
        # `LpRegularisation.val` is read as a coefficient in `lp` and as a ceiling in
        # `lpc`, and one field cannot carry two bounds. This is the first point at which
        # the reading is known, so it is where the wrong role is refused. A `TimeDependent`
        # needs no guard: it is not a term, so it meets the permissive fallback, and the
        # constructor test-substitutes each of its entries through this same check.
        assert_penalty_coefficient_role(lp)
        assert_norm_ceiling_role(lpc)
        if !isnothing(linf) && !isa(linf, TimeDependent)
            assert_nonempty_gt0_finite_val(linf, :linf)
        end
        assert_subgroup_mip_fields(scard, smtx, slt, sst)
        assert_subgrouped_mip_fields(sgcarde, sgmtx, sglt, sgst)
        if isa(wb, WeightBoundsEstimator) ||
           isa(lt, ThresholdEstimator) ||
           isa(st, ThresholdEstimator) ||
           isa(slt, ThresholdEstimator) ||
           isa(sst, ThresholdEstimator) ||
           isa(sglt, ThresholdEstimator) ||
           isa(sgst, ThresholdEstimator) ||
           isa(lcse, LinearConstraintEstimator) ||
           isa(lcse, ExposureConstraintEstimator) ||
           isa(cte, LinearConstraintEstimator) ||
           isa(gcarde, LinearConstraintEstimator) ||
           isa(sgcarde, LinearConstraintEstimator) ||
           isa(smtx, AssetSetsMatrixEstimator) ||
           isa(sgmtx, AssetSetsMatrixEstimator) ||
           isa(fees, FeesEstimator) ||
           isa(tn, TurnoverEstimator) ||
           isa(slt, AbstractVector) && any(x -> isa(x, ThresholdEstimator), slt) ||
           isa(sst, AbstractVector) && any(x -> isa(x, ThresholdEstimator), sst) ||
           isa(sglt, AbstractVector) && any(x -> isa(x, ThresholdEstimator), sglt) ||
           isa(sgst, AbstractVector) && any(x -> isa(x, ThresholdEstimator), sgst) ||
           isa(lcse, AbstractVector) &&
           any(x -> isa(x, LinearConstraintEstimator) || isa(x, ExposureConstraintEstimator),
               lcse) ||
           isa(cte, AbstractVector) && any(x -> isa(x, LinearConstraintEstimator), cte) ||
           isa(gcarde, AbstractVector) &&
           any(x -> isa(x, LinearConstraintEstimator), gcarde) ||
           isa(sgcarde, AbstractVector) &&
           any(x -> isa(x, LinearConstraintEstimator), sgcarde) ||
           isa(smtx, AbstractVector) && any(x -> isa(x, AssetSetsMatrixEstimator), smtx) ||
           isa(sgmtx, AbstractVector) &&
           any(x -> isa(x, AssetSetsMatrixEstimator), sgmtx) ||
           isa(tn, AbstractVector) && any(x -> isa(x, TurnoverEstimator), tn)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when estimator-type fields are provided"))
        end
        assert_time_dependent_substitution(JuMPOptimiser,
                                           (; pe, slv, wb, bgt, sbgt, gbgt, xbgt, lt, st,
                                            lcse, cte, gcarde, sgcarde, smtx, sgmtx, slt,
                                            sst, sglt, sgst, tn, fees, sets, tr, ple, ret,
                                            sca, ccnt, cobj, sc, so, ss, card, scard, l2c,
                                            lpc, linfc, l1, l2, lp, linf, brt, x_src,
                                            strict), jump_optimiser_td_defaults())
        return new{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                   typeof(gbgt), typeof(xbgt), typeof(lt), typeof(st), typeof(lcse),
                   typeof(cte), typeof(gcarde), typeof(sgcarde), typeof(smtx),
                   typeof(sgmtx), typeof(slt), typeof(sst), typeof(sglt), typeof(sgst),
                   typeof(tn), typeof(fees), typeof(sets), typeof(tr), typeof(ple),
                   typeof(ret), typeof(sca), typeof(ccnt), typeof(cobj), typeof(sc),
                   typeof(so), typeof(ss), typeof(card), typeof(scard), typeof(l2c),
                   typeof(lpc), typeof(linfc), typeof(l1), typeof(l2), typeof(lp),
                   typeof(linf), typeof(brt), typeof(x_src), typeof(strict), typeof(cache)}(pe,
                                                                                            slv,
                                                                                            wb,
                                                                                            bgt,
                                                                                            sbgt,
                                                                                            gbgt,
                                                                                            xbgt,
                                                                                            lt,
                                                                                            st,
                                                                                            lcse,
                                                                                            cte,
                                                                                            gcarde,
                                                                                            sgcarde,
                                                                                            smtx,
                                                                                            sgmtx,
                                                                                            slt,
                                                                                            sst,
                                                                                            sglt,
                                                                                            sgst,
                                                                                            tn,
                                                                                            fees,
                                                                                            sets,
                                                                                            tr,
                                                                                            ple,
                                                                                            ret,
                                                                                            sca,
                                                                                            ccnt,
                                                                                            cobj,
                                                                                            sc,
                                                                                            so,
                                                                                            ss,
                                                                                            card,
                                                                                            scard,
                                                                                            l2c,
                                                                                            lpc,
                                                                                            linfc,
                                                                                            l1,
                                                                                            l2,
                                                                                            lp,
                                                                                            linf,
                                                                                            brt,
                                                                                            x_src,
                                                                                            strict,
                                                                                            cache)
    end
end
function JuMPOptimiser(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(), slv::Slv_VecSlv,
                       wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                       bgt::TD_Option{<:Num_BgtCE} = 1.0,
                       sbgt::TD_Option{<:Num_BgtRg} = nothing,
                       gbgt::TD_Option{<:Num_BgtRg} = nothing, xbgt::Bool = false,
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
                       linf::TD_Option{<:Num_AmbRadCal} = nothing, brt::Bool = false,
                       x_src::Symbol = :prior, strict::Bool = false,
                       cache::Option{<:ReturnsBufferState} = nothing)::JuMPOptimiser
    return JuMPOptimiser(pe, slv, wb, bgt, sbgt, gbgt, xbgt, lt, st, lcse, cte, gcarde,
                         sgcarde, smtx, sgmtx, slt, sst, sglt, sgst, tn, fees, sets, tr,
                         ple, ret, sca, ccnt, cobj, sc, so, ss, card, scard, l2c, lpc,
                         linfc, l1, l2, lp, linf, brt, x_src, strict, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights.

Checks turnover, fees, tracking error, custom constraint, and custom objective fields.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.

# Returns

  - `Bool`: `true` if any sub-estimator needs previous weights; `false` otherwise.

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

Return a copy of `opt` with all weight-tracking estimator fields updated via `factory` for the new weights `w`.

Updates turnover, fees, tracking error, custom constraint, and custom objective fields;
all other fields are carried through unchanged.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - `w::AbstractVector`: New portfolio weights.

# Returns

  - `JuMPOptimiser`: Updated optimiser with weight-tracking fields refreshed.

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
    tn = factory(opt.tn, w)
    fees = factory(opt.fees, w)
    tr = factory(opt.tr, w)
    ccnt = factory(opt.ccnt, w)
    cobj = factory(opt.cobj, w)
    return JuMPOptimiser(; pe = opt.pe, slv = opt.slv, wb = opt.wb, bgt = opt.bgt,
                         sbgt = opt.sbgt, gbgt = opt.gbgt, xbgt = opt.xbgt, lt = opt.lt,
                         st = opt.st, lcse = opt.lcse, cte = opt.cte, gcarde = opt.gcarde,
                         sgcarde = opt.sgcarde, smtx = opt.smtx, sgmtx = opt.sgmtx,
                         slt = opt.slt, sst = opt.sst, sglt = opt.sglt, sgst = opt.sgst,
                         tn = tn, fees = fees, sets = opt.sets, tr = tr, ple = opt.ple,
                         ret = opt.ret, sca = opt.sca, ccnt = ccnt, cobj = cobj,
                         sc = opt.sc, so = opt.so, ss = opt.ss, card = opt.card,
                         scard = opt.scard, l2c = opt.l2c, lpc = opt.lpc, linfc = opt.linfc,
                         l1 = opt.l1, l2 = opt.l2, lp = opt.lp, linf = opt.linf,
                         brt = opt.brt, x_src = opt.x_src, strict = opt.strict,
                         cache = opt.cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of `opt` restricted to asset indices `i`.

Slices all per-asset estimator fields (prior, weight bounds, thresholds, turnover, fees,
tracking, custom constraint/objective) to the cluster; solver and scalar parameters are
carried through unchanged.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - `i`: Asset index or index set for the cluster.
  - `X::MatNum`: Asset returns matrix (observations x assets) used to slice tracking estimators. A precomputed prior in `opt.pe` supplies its own `X` instead.

# Returns

  - `JuMPOptimiser`: Cluster-restricted optimiser.

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
    # A fee spans two axes: the five per-asset fields on the investable assets, and the two
    # liquidation carriers on the complement. Its view derives that complement from the
    # width of the unreduced `X`, so this is the one constraint that must be handed the
    # matrix rather than the index alone.
    fees = port_opt_view(opt.fees, i, X)
    tr = port_opt_view(opt.tr, i, X)
    ret = port_opt_view(opt.ret, i)
    ccnt = port_opt_view(opt.ccnt, i)
    cobj = port_opt_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         gbgt = opt.gbgt, xbgt = opt.xbgt, lt = lt, st = st, lcse = lcse,
                         cte = opt.cte, gcarde = opt.gcarde, sgcarde = opt.sgcarde,
                         smtx = smtx, sgmtx = sgmtx, slt = slt, sst = sst, sglt = sglt,
                         sgst = sgst, tn = tn, fees = fees, sets = sets, tr = tr,
                         ple = opt.ple, ret = ret, sca = opt.sca, ccnt = ccnt, cobj = cobj,
                         sc = opt.sc, so = opt.so, ss = opt.ss, card = opt.card,
                         scard = opt.scard, l2c = opt.l2c, lpc = opt.lpc, linfc = opt.linfc,
                         l1 = opt.l1, l2 = opt.l2, lp = opt.lp, linf = opt.linf,
                         brt = opt.brt, x_src = opt.x_src, strict = opt.strict,
                         cache = port_opt_view(opt.cache, i))
end
function non_investable_universe(opt::JuMPOptimiser, ni::VecStr)::JuMPOptimiser
    return rebuild_estimator(opt, (; sets = non_investable_sets(opt.sets, ni)))
end

export ProcessedJuMPOptimiserAttributes, JuMPOptimiser
