"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for processed optimiser attributes. Every collection of processed optimiser attributes should subtype

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
results rather than re-processing estimators. See ADR 0008 (`0008-jump-model-assembly.md`).

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
        plr::Option{<:AbstractPhylogenyConstraintResult}, ret::JuMPReturnsEstimator
    ) -> ProcessedJuMPOptimiserAttributes

Keywords correspond to the struct's fields. The field types are the *result* side of the
matching [`JuMPOptimiser`](@ref) estimator slots: this bundle holds the constraint and
prior results produced by [`processed_jump_optimiser_attributes`](@ref), not the raw
estimators (`ret` aside — a returns estimator has no separate result form). In practice,
construct via [`processed_jump_optimiser_attributes`](@ref) rather than directly.

# Examples

```jldoctest
julia> pr = prior(EmpiricalPrior(),
                  ReturnsResult(; nx = ["a", "b"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1]));

julia> ProcessedJuMPOptimiserAttributes(; pr = pr, wb = nothing, lt = nothing, st = nothing,
                                        lcsr = nothing, ctr = nothing, gcardr = nothing,
                                        sgcardr = nothing, smtx = nothing, sgmtx = nothing,
                                        slt = nothing, sst = nothing, sglt = nothing,
                                        sgst = nothing, tn = nothing, fees = nothing,
                                        plr = nothing, ret = ArithmeticReturn()) isa
       ProcessedJuMPOptimiserAttributes
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
                                              ret::JuMPReturnsEstimator)
        return new{typeof(pr), typeof(wb), typeof(lt), typeof(st), typeof(lcsr),
                   typeof(ctr), typeof(gcardr), typeof(sgcardr), typeof(smtx),
                   typeof(sgmtx), typeof(slt), typeof(sst), typeof(sglt), typeof(sgst),
                   typeof(tn), typeof(fees), typeof(plr), typeof(ret)}(pr, wb, lt, st, lcsr,
                                                                       ctr, gcardr, sgcardr,
                                                                       smtx, sgmtx, slt,
                                                                       sst, sglt, sgst, tn,
                                                                       fees, plr, ret)
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
                                          ret::JuMPReturnsEstimator)::ProcessedJuMPOptimiserAttributes
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr,
                                            smtx, sgmtx, slt, sst, sglt, sgst, tn, fees,
                                            plr, ret)
end
"""
$(DocStringExtensions.TYPEDEF)

Shared field core for JuMP-based optimisation results.

Holds the fields common to every JuMP optimisation result. Embedded as the first field (`jr`) of each concrete JuMP result, analogous to how [`JuMPOptimiser`](@ref) is embedded as `opt` in each JuMP optimiser. The concrete result keeps only its unique fields plus the trailing `fb`.

Defined here (rather than in `08_Base_JuMPOptimisation.jl`, where its [`BaseJuMPOptimisationResult`](@ref) supertype lives) so its typed constructor can bind `pa::ProcessedJuMPOptimiserAttributes` and `sol::JuMPOptimisationSolution`, both in scope at this point in load order.

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
    return JuMPOptimisationResult(pa, retcode, sol, model)
end
# Virtual property `:w` extracts portfolio weights from `sol` (a single solution or a vector
# of them, hence the broadcast); unknown properties forward to `pa` (see [`@forward_properties`](@ref)).
@forward_properties JuMPOptimisationResult begin
    compute(w, sol.w; broadcast)
    forward(pa)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `val` is finite and non-negative; throw an `ArgCheck` error otherwise.

Accepts a scalar `Number` or a `VecNum`; the vector overload requires at least one finite
and non-negative element and no negative elements.

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
        pe::TD{<:PrE_Pr} = EmpiricalPrior(),
        slv::Slv_VecSlv,
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        bgt::TD_Option{<:Num_BgtCE} = 1.0,
        sbgt::TD_Option{<:Num_BgtRg} = nothing,
        gbgt::TD_Option{<:Num_BgtRg} = nothing,
        xbgt::Bool = false,
        lt::TD_Option{<:BtE_Bt} = nothing,
        st::TD_Option{<:BtE_Bt} = nothing,
        lcse::TD_Option{<:LcE_Lc_VecLcE_Lc} = nothing,
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
        sets::TD_Option{<:AssetSets} = nothing,
        tr::TD_Option{<:Tr_VecTr} = nothing,
        ple::TD_Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
        ret::TD{<:JuMPReturnsEstimator} = ArithmeticReturn(),
        sca::TD{<:NonHierarchicalScalariser} = SumScalariser(),
        ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr} = nothing,
        cobj::TD_Option{<:JuMPObj_VecJuMPObj} = nothing,
        sc::Number = 1,
        so::Number = 1,
        ss::TD_Option{<:Number} = nothing,
        card::TD_Option{<:Integer} = nothing,
        scard::TD_Option{<:Int_VecInt} = nothing,
        l2c::TD_Option{<:Number} = nothing,
        lpc::TD_Option{<:LpReg_VecLpReg} = nothing,
        linfc::TD_Option{<:Number} = nothing,
        l1::TD_Option{<:Number} = nothing,
        l2::TD_Option{<:L2Reg_VecL2Reg} = nothing,
        linf::TD_Option{<:Number} = nothing,
        lp::TD_Option{<:LpReg_VecLpReg} = nothing,
        brt::Bool = false,
        cle_pr::Bool = true,
        strict::Bool = false,
    ) -> JuMPOptimiser

Keywords correspond to the struct's fields. Fields typed [`TD_Option`](@ref) or [`TD`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value; a cross-validation fold loop resolves it per fold, and a fold-less `optimise` runs with the field at its static default. The problem definition — the prior estimator, returns model, scalariser and asset sets as much as the constraints — may therefore vary over folds; execution control (`slv`, `sc`, `so`, `brt`, `cle_pr`, `strict`) stays static.

## Validation

  - If `slv` is a vector: `!isempty(slv)`.
  - If `bgt` is a number: `isfinite(bgt)`.
  - If `bgt` is a `BudgetCostEstimator`: `isnothing(sbgt)`.
  - If `sbgt` is a number: `isfinite(sbgt)` and `sbgt >= 0`.
  - If `gbgt` is a number: `isfinite(gbgt)` and `gbgt >= 0`.
  - If `gbgt` is provided: `wb` must admit short positions, and `bgt` and `sbgt` must not *both* be pinned numbers (which already determine the gross exposure as `bgt + 2 * sbgt`).
  - If `cte` is a vector: `!isempty(cte)`.
  - If `card` is provided: `card > 0` and finite.
  - If `tn` or `tr` is a vector: each must be non-empty.
  - If `l2c`, `linfc`, `l1`, or `linf` is provided: each must be `> 0` and finite.
  - If `lp` is a vector: `!isempty(lp)`.
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
    Centring constraint estimator(s) or constraint(s).
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
    $(field_dict[:ple])
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
    $(field_dict[:linf])
    """
    linf
    """
    $(field_dict[:lp])
    """
    lp
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:cle_pr])
    """
    cle_pr
    """
    $(field_dict[:strict_opt])
    """
    strict
    function JuMPOptimiser(pe::TD{<:PrE_Pr}, slv::Slv_VecSlv, wb::TD_Option{<:WbE_Wb},
                           bgt::TD_Option{<:Num_BgtCE}, sbgt::TD_Option{<:Num_BgtRg},
                           gbgt::TD_Option{<:Num_BgtRg}, xbgt::Bool,
                           lt::TD_Option{<:BtE_Bt}, st::TD_Option{<:BtE_Bt},
                           lcse::TD_Option{<:LcE_Lc_VecLcE_Lc},
                           cte::TD_Option{<:Lc_CC_VecCC}, gcarde::TD_Option{<:LcE_Lc},
                           sgcarde::TD_Option{<:LcE_Lc_VecLcE_Lc},
                           smtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           sgmtx::TD_Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           slt::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           sst::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           sglt::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           sgst::TD_Option{<:BtE_Bt_VecOptBtE_Bt},
                           tn::TD_Option{<:TnE_Tn_VecTnE_Tn}, fees::TD_Option{<:FeesE_Fees},
                           sets::TD_Option{<:AssetSets}, tr::TD_Option{<:Tr_VecTr},
                           ple::TD_Option{<:PlCE_PhC_VecPlCE_PlC},
                           ret::TD{<:JuMPReturnsEstimator},
                           sca::TD{<:NonHierarchicalScalariser},
                           ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr},
                           cobj::TD_Option{<:JuMPObj_VecJuMPObj}, sc::Number, so::Number,
                           ss::TD_Option{<:Number}, card::TD_Option{<:Integer},
                           scard::TD_Option{<:Int_VecInt}, l2c::TD_Option{<:Number},
                           lpc::TD_Option{<:LpReg_VecLpReg}, linfc::TD_Option{<:Number},
                           l1::TD_Option{<:Number}, l2::TD_Option{<:L2Reg_VecL2Reg},
                           linf::TD_Option{<:Number}, lp::TD_Option{<:LpReg_VecLpReg},
                           brt::Bool, cle_pr::Bool, strict::Bool)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        if isa(bgt, Number)
            @argcheck(isfinite(bgt), IsNonFiniteError("bgt must be finite, got $bgt"))
        elseif isa(bgt, BudgetCostEstimator)
            @argcheck(isnothing(sbgt),
                      IsNothingError("sbgt must be nothing when bgt is a BudgetCostEstimator, got sbgt = $sbgt"))
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
        if !isnothing(linf) && !isa(linf, TimeDependent)
            assert_nonempty_gt0_finite_val(linf, :linf)
        end
        if isa(lp, AbstractVector)
            @argcheck(!isempty(lp), IsEmptyError("lp cannot be empty"))
        end
        scard_td = any(x -> isa(x, TimeDependent), (scard, smtx, slt, sst))
        if !scard_td && isa(scard, Integer)
            assert_nonempty_gt0_finite_val(scard, :scard)
            @argcheck(isa(smtx, MatNum_ASetMatE),
                      ArgumentError("smtx must be a MatNum_ASetMatE when scard is an Integer, got $(typeof(smtx))"))
            @argcheck(isa(slt, Option{<:BtE_Bt}),
                      ArgumentError("slt must be a scalar BtE_Bt or nothing when scard is an Integer, got $(typeof(slt))"))
            @argcheck(isa(sst, Option{<:BtE_Bt}),
                      ArgumentError("sst must be a scalar BtE_Bt or nothing when scard is an Integer, got $(typeof(sst))"))
        elseif !scard_td && isa(scard, VecInt)
            assert_nonempty_gt0_finite_val(scard, :scard)
            @argcheck(isa(smtx, AbstractVector),
                      ArgumentError("smtx must be an AbstractVector when scard is a VecInt, got $(typeof(smtx))"))
            @argcheck(length(scard) == length(smtx),
                      DimensionMismatch("scard ($(length(scard))) must match smtx ($(length(smtx)))"))
            if isa(slt, AbstractVector)
                @argcheck(!isempty(slt), IsEmptyError("slt cannot be empty"))
                @argcheck(length(scard) == length(slt),
                          DimensionMismatch("scard ($(length(scard))) must match slt ($(length(slt)))"))
            end
            if isa(sst, AbstractVector)
                @argcheck(!isempty(sst), IsEmptyError("sst cannot be empty"))
                @argcheck(length(scard) == length(sst),
                          DimensionMismatch("scard ($(length(scard))) must match sst ($(length(sst)))"))
            end
        elseif !scard_td && isnothing(scard) && (isa(slt, BtE_Bt) || isa(sst, BtE_Bt))
            @argcheck(isa(smtx, MatNum_ASetMatE),
                      ArgumentError("smtx must be a MatNum_ASetMatE when slt or sst is a scalar BtE_Bt, got $(typeof(smtx))"))
        elseif !scard_td &&
               isnothing(scard) &&
               (isa(slt, AbstractVector) || isa(sst, AbstractVector))
            @argcheck(isa(smtx, AbstractVector),
                      ArgumentError("smtx must be an AbstractVector when slt or sst is a vector, got $(typeof(smtx))"))
            @argcheck(!isempty(smtx), IsEmptyError("smtx cannot be empty"))
            if isa(slt, AbstractVector)
                @argcheck(!isempty(slt), IsEmptyError("slt cannot be empty"))
                @argcheck(length(slt) == length(smtx),
                          DimensionMismatch("slt ($(length(slt))) must match smtx ($(length(smtx)))"))
            end
            if isa(sst, AbstractVector)
                @argcheck(!isempty(sst), IsEmptyError("sst cannot be empty"))
                @argcheck(length(sst) == length(smtx),
                          DimensionMismatch("sst ($(length(sst))) must match smtx ($(length(smtx)))"))
            end
        end
        sgcard_td = any(x -> isa(x, TimeDependent), (sgcarde, sgmtx, sglt, sgst))
        if !sgcard_td && isa(sgcarde, LcE_Lc)
            @argcheck(isa(sgmtx, MatNum_ASetMatE),
                      ArgumentError("sgmtx must be a MatNum_ASetMatE when sgcarde is a scalar LcE_Lc, got $(typeof(sgmtx))"))
            @argcheck(isa(sglt, Option{<:BtE_Bt}),
                      ArgumentError("sglt must be a scalar BtE_Bt or nothing when sgcarde is a scalar LcE_Lc, got $(typeof(sglt))"))
            @argcheck(isa(sgst, Option{<:BtE_Bt}),
                      ArgumentError("sgst must be a scalar BtE_Bt or nothing when sgcarde is a scalar LcE_Lc, got $(typeof(sgst))"))
            if isa(sgcarde, LinearConstraint) && isa(smtx, MatNum)
                N = size(smtx, 1)
                N_ineq = !isnothing(sgcarde.ineq) ? length(sgcarde.B_ineq) : 0
                N_eq = !isnothing(sgcarde.eq) ? length(sgcarde.B_eq) : 0
                @argcheck(N == N_ineq + N_eq,
                          DimensionMismatch("smtx rows ($N) must equal N_ineq + N_eq ($(N_ineq + N_eq))"))
            end
        elseif !sgcard_td && isa(sgcarde, AbstractVector)
            @argcheck(!isempty(sgcarde), IsEmptyError("sgcarde cannot be empty"))
            @argcheck(isa(sgmtx, AbstractVector),
                      ArgumentError("sgmtx must be an AbstractVector when sgcarde is a vector, got $(typeof(sgmtx))"))
            @argcheck(!isempty(sgmtx), IsEmptyError("sgmtx cannot be empty"))
            @argcheck(length(sgcarde) == length(sgmtx),
                      DimensionMismatch("sgcarde ($(length(sgcarde))) must match sgmtx ($(length(sgmtx)))"))
            if isa(sglt, AbstractVector)
                @argcheck(!isempty(sglt), IsEmptyError("sglt cannot be empty"))
                @argcheck(length(sgcarde) == length(sglt),
                          DimensionMismatch("sgcarde ($(length(sgcarde))) must match sglt ($(length(sglt)))"))
            end
            if isa(sgst, AbstractVector)
                @argcheck(!isempty(sgst), IsEmptyError("sgst cannot be empty"))
                @argcheck(length(sgcarde) == length(sgst),
                          DimensionMismatch("sgcarde ($(length(sgcarde))) must match sgst ($(length(sgst)))"))
            end
            for (sgc, smt) in zip(sgcarde, sgmtx)
                if isa(sgc, LinearConstraint) && isa(smt, MatNum)
                    N = size(smt, 1)
                    N_ineq = !isnothing(sgc.ineq) ? length(sgc.B_ineq) : 0
                    N_eq = !isnothing(sgc.eq) ? length(sgc.B_eq) : 0
                    @argcheck(N == N_ineq + N_eq,
                              DimensionMismatch("smt rows ($N) must equal N_ineq + N_eq ($(N_ineq + N_eq))"))
                end
            end
        elseif !sgcard_td && isnothing(sgcarde) && (isa(sglt, BtE_Bt) || isa(sgst, BtE_Bt))
            @argcheck(isa(sgmtx, MatNum_ASetMatE),
                      ArgumentError("sgmtx must be a MatNum_ASetMatE when sglt or sgst is a scalar BtE_Bt, got $(typeof(sgmtx))"))
        elseif !sgcard_td &&
               isnothing(sgcarde) &&
               (isa(sglt, AbstractVector) || isa(sgst, AbstractVector))
            @argcheck(isa(sgmtx, AbstractVector),
                      ArgumentError("sgmtx must be an AbstractVector when sglt or sgst is a vector, got $(typeof(sgmtx))"))
            @argcheck(!isempty(sgmtx), IsEmptyError("sgmtx cannot be empty"))
            if isa(sglt, AbstractVector)
                @argcheck(!isempty(sglt), IsEmptyError("sglt cannot be empty"))
                @argcheck(length(sglt) == length(sgmtx),
                          DimensionMismatch("sglt ($(length(sglt))) must match sgmtx ($(length(sgmtx)))"))
            end
            if isa(sgst, AbstractVector)
                @argcheck(!isempty(sgst), IsEmptyError("sgst cannot be empty"))
                @argcheck(length(sgst) == length(sgmtx),
                          DimensionMismatch("sgst ($(length(sgst))) must match sgmtx ($(length(sgmtx)))"))
            end
        end
        if isa(wb, WeightBoundsEstimator) ||
           isa(lt, ThresholdEstimator) ||
           isa(st, ThresholdEstimator) ||
           isa(slt, ThresholdEstimator) ||
           isa(sst, ThresholdEstimator) ||
           isa(sglt, ThresholdEstimator) ||
           isa(sgst, ThresholdEstimator) ||
           isa(lcse, LinearConstraintEstimator) ||
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
           isa(lcse, AbstractVector) && any(x -> isa(x, LinearConstraintEstimator), lcse) ||
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
                                            lpc, linfc, l1, l2, linf, lp, brt, cle_pr,
                                            strict), jump_optimiser_td_defaults())
        return new{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                   typeof(gbgt), typeof(xbgt), typeof(lt), typeof(st), typeof(lcse),
                   typeof(cte), typeof(gcarde), typeof(sgcarde), typeof(smtx),
                   typeof(sgmtx), typeof(slt), typeof(sst), typeof(sglt), typeof(sgst),
                   typeof(tn), typeof(fees), typeof(sets), typeof(tr), typeof(ple),
                   typeof(ret), typeof(sca), typeof(ccnt), typeof(cobj), typeof(sc),
                   typeof(so), typeof(ss), typeof(card), typeof(scard), typeof(l2c),
                   typeof(lpc), typeof(linfc), typeof(l1), typeof(l2), typeof(linf),
                   typeof(lp), typeof(brt), typeof(cle_pr), typeof(strict)}(pe, slv, wb,
                                                                            bgt, sbgt, gbgt,
                                                                            xbgt, lt, st,
                                                                            lcse, cte,
                                                                            gcarde, sgcarde,
                                                                            smtx, sgmtx,
                                                                            slt, sst, sglt,
                                                                            sgst, tn, fees,
                                                                            sets, tr, ple,
                                                                            ret, sca, ccnt,
                                                                            cobj, sc, so,
                                                                            ss, card, scard,
                                                                            l2c, lpc, linfc,
                                                                            l1, l2, linf,
                                                                            lp, brt, cle_pr,
                                                                            strict)
    end
end
function JuMPOptimiser(; pe::TD{<:PrE_Pr} = EmpiricalPrior(), slv::Slv_VecSlv,
                       wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                       bgt::TD_Option{<:Num_BgtCE} = 1.0,
                       sbgt::TD_Option{<:Num_BgtRg} = nothing,
                       gbgt::TD_Option{<:Num_BgtRg} = nothing, xbgt::Bool = false,
                       lt::TD_Option{<:BtE_Bt} = nothing, st::TD_Option{<:BtE_Bt} = nothing,
                       lcse::TD_Option{<:LcE_Lc_VecLcE_Lc} = nothing,
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
                       sets::TD_Option{<:AssetSets} = nothing,
                       tr::TD_Option{<:Tr_VecTr} = nothing,
                       ple::TD_Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
                       ret::TD{<:JuMPReturnsEstimator} = ArithmeticReturn(),
                       sca::TD{<:NonHierarchicalScalariser} = SumScalariser(),
                       ccnt::TD_Option{<:JuMPConstr_VecJuMPConstr} = nothing,
                       cobj::TD_Option{<:JuMPObj_VecJuMPObj} = nothing, sc::Number = 1,
                       so::Number = 1, ss::TD_Option{<:Number} = nothing,
                       card::TD_Option{<:Integer} = nothing,
                       scard::TD_Option{<:Int_VecInt} = nothing,
                       l2c::TD_Option{<:Number} = nothing,
                       lpc::TD_Option{<:LpReg_VecLpReg} = nothing,
                       linfc::TD_Option{<:Number} = nothing,
                       l1::TD_Option{<:Number} = nothing,
                       l2::TD_Option{<:L2Reg_VecL2Reg} = nothing,
                       linf::TD_Option{<:Number} = nothing,
                       lp::TD_Option{<:LpReg_VecLpReg} = nothing, brt::Bool = false,
                       cle_pr::Bool = true, strict::Bool = false)::JuMPOptimiser
    return JuMPOptimiser(pe, slv, wb, bgt, sbgt, gbgt, xbgt, lt, st, lcse, cte, gcarde,
                         sgcarde, smtx, sgmtx, slt, sst, sglt, sgst, tn, fees, sets, tr,
                         ple, ret, sca, ccnt, cobj, sc, so, ss, card, scard, l2c, lpc,
                         linfc, l1, l2, linf, lp, brt, cle_pr, strict)
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
                         l1 = opt.l1, l2 = opt.l2, linf = opt.linf, lp = opt.lp,
                         brt = opt.brt, cle_pr = opt.cle_pr, strict = opt.strict)
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
  - `X::MatNum`: FullMoment returns matrix used to slice tracking estimators.

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
    sets = port_opt_view(opt.sets, i)
    fees = port_opt_view(opt.fees, i)
    tr = port_opt_view(opt.tr, i, X)
    ret = port_opt_view(opt.ret, i)
    ccnt = port_opt_view(opt.ccnt, i)
    cobj = port_opt_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         gbgt = opt.gbgt, xbgt = opt.xbgt, lt = lt, st = st,
                         lcse = opt.lcse, cte = opt.cte, gcarde = opt.gcarde,
                         sgcarde = opt.sgcarde, smtx = smtx, sgmtx = sgmtx, slt = slt,
                         sst = sst, sglt = sglt, sgst = sgst, tn = tn, fees = fees,
                         sets = sets, tr = tr, ple = opt.ple, ret = ret, sca = opt.sca,
                         ccnt = ccnt, cobj = cobj, sc = opt.sc, so = opt.so, ss = opt.ss,
                         card = opt.card, scard = opt.scard, l2c = opt.l2c, lpc = opt.lpc,
                         linfc = opt.linfc, l1 = opt.l1, l2 = opt.l2, linf = opt.linf,
                         lp = opt.lp, brt = opt.brt, cle_pr = opt.cle_pr,
                         strict = opt.strict)
end
"""
    processed_jump_optimiser_attributes(
        opt::JuMPOptimiser,
        rd::ReturnsResult;
        dims::Int = 1
    ) -> ProcessedJuMPOptimiserAttributes

Compute all constraint and prior results needed for model assembly.

Resolves every estimator field of `opt` against `rd` — running priors, weight bounds,
thresholds, linear constraints, centrality, cardinality, turnover, fees, and phylogeny
— and returns the fully processed bundle as a
[`ProcessedJuMPOptimiserAttributes`](@ref). The result is consumed directly by
[`assemble_jump_model!`](@ref) and also stored in the per-optimiser Result struct, so
processing happens exactly once per `optimise` call.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - $(arg_dict[:rd])
  - `dims::Int = 1`: Observation dimension passed to the prior estimator.

# Returns

  - [`ProcessedJuMPOptimiserAttributes`](@ref): Fully resolved constraint and prior bundle.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`processed_jump_optimiser`](@ref)
"""
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1, kwargs...)
    rd = returns_result_picker(rd, opt.brt)
    pr = prior(opt.pe, rd; dims = dims)
    X = pr.X
    datatype = eltype(X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(X, 2), strict = opt.strict,
                                   datatype = datatype)
    lt = threshold_constraints(opt.lt, opt.sets; datatype = datatype, strict = opt.strict)
    st = threshold_constraints(opt.st, opt.sets; datatype = datatype, strict = opt.strict)
    lcsr = linear_constraints(opt.lcse, opt.sets; datatype = datatype, strict = opt.strict)
    ctr = centrality_constraints(opt.cte, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd,
                                 cle_pr = opt.cle_pr, kwargs...)
    gcardr = linear_constraints(opt.gcarde, opt.sets; datatype = Int, strict = opt.strict)
    sgcardr = linear_constraints(opt.sgcarde, opt.sets; datatype = Int, strict = opt.strict)
    if opt.smtx === opt.sgmtx
        smtx = sgmtx = asset_sets_matrix(opt.smtx, opt.sets)
    else
        smtx = asset_sets_matrix(opt.smtx, opt.sets)
        sgmtx = asset_sets_matrix(opt.sgmtx, opt.sets)
    end
    if opt.slt === opt.sglt
        slt = sglt = threshold_constraints(opt.slt, opt.sets; datatype = datatype,
                                           strict = opt.strict)
    else
        slt = threshold_constraints(opt.slt, opt.sets; datatype = datatype,
                                    strict = opt.strict)
        sglt = threshold_constraints(opt.sglt, opt.sets; datatype = datatype,
                                     strict = opt.strict)
    end
    if opt.sst === opt.sgst
        sst = sgst = threshold_constraints(opt.sst, opt.sets; datatype = datatype,
                                           strict = opt.strict)
    else
        sst = threshold_constraints(opt.sst, opt.sets; datatype = datatype,
                                    strict = opt.strict)
        sgst = threshold_constraints(opt.sgst, opt.sets; datatype = datatype,
                                     strict = opt.strict)
    end
    tn = turnover_constraints(opt.tn, opt.sets; datatype = datatype, strict = opt.strict)
    fees = fees_constraints(opt.fees, opt.sets; datatype = datatype, strict = opt.strict)
    plr = phylogeny_constraints(opt.ple, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd,
                                cle_pr = opt.cle_pr, kwargs...)
    ret = factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(; pr = pr, wb = wb, lt = lt, st = st,
                                            lcsr = lcsr, ctr = ctr, gcardr = gcardr,
                                            sgcardr = sgcardr, smtx = smtx, sgmtx = sgmtx,
                                            slt = slt, sst = sst, sglt = sglt, sgst = sgst,
                                            tn = tn, fees = fees, plr = plr, ret = ret)
end
"""
    no_bounds_optimiser(opt::JuMPOptimiser, args...) -> JuMPOptimiser

Return a copy of `opt` with the returns estimator replaced by its unbounded variant.

Used internally in risk-frontier sub-problems where weight and risk bounds must be
removed so the solver can range freely.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - `args...`: Forwarded to [`no_bounds_returns_estimator`](@ref).

# Returns

  - `JuMPOptimiser`: Optimiser with an unbounded returns estimator.

# Examples

```jldoctest
julia> opt = JuMPOptimiser(; slv = Solver(; solver = nothing));

julia> PortfolioOptimisers.no_bounds_optimiser(opt) isa JuMPOptimiser
true
```

# Related

  - [`no_bounds_returns_estimator`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(fieldnames(typeof(opt)), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getfield.(opt, pnames))...)
end
"""
    jump_optimiser_from_attributes(
        opt::JuMPOptimiser,
        attrs::ProcessedJuMPOptimiserAttributes
    ) -> JuMPOptimiser

Repackage a [`ProcessedJuMPOptimiserAttributes`](@ref) into a [`JuMPOptimiser`](@ref).

Maps result-named fields onto the optimiser's estimator-named slots (`lcsr` → `lcse`,
`plr` → `ple`, `pr` → `pe`, …) and carries all remaining settings through from `opt`.
Used where a processed optimiser object is needed for inner sub-problems (e.g.
[`near_optimal_centering_setup`](@ref)) while the same `attrs` is also passed directly to
[`assemble_jump_model!`](@ref) — so processing happens once and is never round-tripped.

# Arguments

  - `opt::JuMPOptimiser`: Source optimiser supplying solver, scalar settings, and any
    fields not present in `attrs`.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed constraint and prior bundle.

# Returns

  - `JuMPOptimiser`: Optimiser with all estimator slots populated from `attrs`.

# Related

  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`processed_jump_optimiser`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function jump_optimiser_from_attributes(opt::JuMPOptimiser,
                                        attrs::ProcessedJuMPOptimiserAttributes)
    # The remap is derived by reflection so it can never silently drop a field (a hand-written
    # pull-through once omitted `scard`, disabling sub-group cardinality in constrained-NOC;
    # fixed in 9ff28d6). Every `opt` field flows through by name (`base`); every `attrs` result
    # field overrides its optimiser slot. Six result fields carry a different name than their
    # slot — listed in `rename` — the other twelve match by name and map automatically.
    rename = (; pr = :pe, lcsr = :lcse, ctr = :cte, gcardr = :gcarde, sgcardr = :sgcarde,
              plr = :ple)
    of = fieldnames(JuMPOptimiser)
    af = fieldnames(ProcessedJuMPOptimiserAttributes)
    base = NamedTuple{of}(getfield.(opt, of))
    overrides = NamedTuple{map(f -> get(rename, f, f), af)}(getfield.(attrs, af))
    return JuMPOptimiser(; merge(base, overrides)...)
end
"""
    processed_jump_optimiser(
        opt::JuMPOptimiser,
        rd::ReturnsResult;
        dims::Int = 1
    ) -> JuMPOptimiser

Build a fully-processed [`JuMPOptimiser`](@ref) from raw configuration and returns data.

Calls [`processed_jump_optimiser_attributes`](@ref) then repackages the result via
[`jump_optimiser_from_attributes`](@ref). Used where a processed optimiser is needed for
inner sub-problems (e.g. [`NearOptimalCentering`](@ref)) while the `attrs` bundle is
reused directly by [`assemble_jump_model!`](@ref).

# Arguments

  - `opt::JuMPOptimiser`: Raw optimiser configuration.
  - $(arg_dict[:rd])
  - `dims::Int = 1`: Observation dimension passed to the prior estimator.

# Returns

  - `JuMPOptimiser`: Optimiser with all estimator slots populated from processed results.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`jump_optimiser_from_attributes`](@ref)
"""
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1,
                                  kwargs...)
    attrs = processed_jump_optimiser_attributes(opt, rd; dims = dims, kwargs...)
    return jump_optimiser_from_attributes(opt, attrs)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add risk-measure constraints and scalarise the combined risk expression in `model`.

One step of [`assemble_jump_model!`](@ref), dispatched on `r`: when `r` is `nothing`
(e.g. [`RelaxedRiskBudgeting`](@ref), whose risk lives in its head) this is a no-op.
`extra` is the optional trailing-argument tuple — empty for most optimisers, or `(b1,)`
for [`FactorRiskContribution`](@ref) — splatted into `set_risk_constraints!` so the
non-factor call signature is reproduced exactly.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model (mutated in place).
  - `r`: Risk measure(s), or `nothing` to skip this step entirely.
  - `optimiser`: Dispatch object for [`set_risk_constraints!`](@ref).
  - `opt::JuMPOptimiser`: Supplies the scalariser `opt.sca`.
  - `pr`: Prior result passed to risk builders.
  - `pl`: Phylogeny result passed to risk builders.
  - `fees`: Fees result passed to risk builders.
  - `extra`: Trailing-argument tuple splatted into risk builders (`()` or `(b1,)`).
  - `rd`: Returns result forwarded as a keyword argument to risk builders.

# Returns

  - `nothing`.

# Related

  - [`assemble_jump_model!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`scalarise_risk_expression!`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`FactorRiskContribution`](@ref)
"""
function set_risk_and_scalarise!(::JuMP.Model, ::Nothing, args...; kwargs...)
    return nothing
end
function set_risk_and_scalarise!(model::JuMP.Model, r, optimiser, opt, pr, pl, fees,
                                 args...; rd)
    set_risk_constraints!(model, r, optimiser, pr, pl, fees, args...; rd = rd)
    scalarise_risk_expression!(model, opt.sca)
    return nothing
end

"""
    assemble_jump_model!(
        model::JuMP.Model,
        optimiser::JuMPOptimisationEstimator,
        opt::JuMPOptimiser,
        attrs::ProcessedJuMPOptimiserAttributes,
        rd::ReturnsResult,
        r::Option{<:RM_VecRM} = nothing,
        obj::ObjectiveFunction = MinimumRisk(),
        b1::Option{<:MatNum} = nothing,
        sdp_asset_phylogeny::Bool = true
    ) -> Nothing

Run the invariant model-assembly sequence shared by all single-JuMP-model optimisers.

Executes the constraint-builder pipeline — from `set_linear_weight_constraints!` through
`add_custom_constraint!` — that sits between the per-optimiser *head* (weight variables)
and *tail* (objective + solve). The head must have populated Model State (`w`/`k` variables)
before calling this function. See `Model Assembly` in `CONTEXT.md` and
`0008-jump-model-assembly.md`.

# Arguments

  - $(arg_dict[:model])
  - `optimiser::JuMPOptimisationEstimator`: Dispatch object for risk, tracking, and custom
    constraint builders.
  - `opt::JuMPOptimiser`: Supplies scalar settings (`l2c`, `linfc`, `l1`, `l2`, `linf`, `lp`,
    `card`, `scard`, `tr`, `ccnt`, `sca`, `ss`).
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed constraint and prior bundle
    produced by [`processed_jump_optimiser_attributes`](@ref).
  - $(arg_dict[:rd])
  - `r::Option{<:RiskMeasure} = nothing`: Risk measure(s), or `nothing` to skip risk
    constraints and scalarisation (the [`RelaxedRiskBudgeting`](@ref) path).
  - `obj::ObjectiveFunction = MinimumRisk()`: Objective used by the return constraints.
  - `b1::Option{<:MatNum} = nothing`: Factor loading matrix for
    [`FactorRiskContribution`](@ref); `nothing` for all other optimisers.
  - `sdp_asset_phylogeny::Bool = true`: Whether to apply the standard asset-space SDP phylogeny
    constraints. [`FactorRiskContribution`](@ref) passes `false` and applies its own
    factor-space variant in its tail instead.

# Returns

  - `nothing`. Mutates `model` in place.

# Examples

```jldoctest
julia> using PortfolioOptimisers

julia> pr = prior(EmpiricalPrior(),
                  ReturnsResult(; nx = ["a", "b"], X = [0.1 -0.2; -0.1 0.2; 0.05 0.1]));

julia> ProcessedJuMPOptimiserAttributes(; pr = pr, wb = nothing, lt = nothing, st = nothing,
                                        lcsr = nothing, ctr = nothing, gcardr = nothing,
                                        sgcardr = nothing, smtx = nothing, sgmtx = nothing,
                                        slt = nothing, sst = nothing, sglt = nothing,
                                        sgst = nothing, tn = nothing, fees = nothing,
                                        plr = nothing, ret = ArithmeticReturn()) isa
       ProcessedJuMPOptimiserAttributes
true
```

# Related

  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`set_risk_and_scalarise!`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`FactorRiskContribution`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function assemble_jump_model!(model::JuMP.Model, optimiser::JuMPOptimisationEstimator,
                              opt::JuMPOptimiser, attrs::ProcessedJuMPOptimiserAttributes,
                              rd::ReturnsResult, r::Option{<:RM_VecRM} = nothing,
                              obj::ObjectiveFunction = MinimumRisk(),
                              b1::Option{<:MatNum} = nothing,
                              sdp_asset_phylogeny::Bool = true)
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst, tn, fees, plr, ret) = attrs
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, opt.card, gcardr, plr, lt, st, fees, opt.ss, opt.xbgt)
    set_smip_constraints!(model, wb, opt.scard, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst,
                          opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, opt.tr, optimiser, plr, fees, b1; rd = rd)
    set_weight_norm_2_constraints!(model, opt.l2c)
    set_weight_norm_p_constraints!(model, opt.lpc)
    set_weight_norm_inf_constraints!(model, opt.linfc)
    set_l1_regularisation!(model, opt.l1)
    set_l2_regularisation!(model, opt.l2)
    set_linf_regularisation!(model, opt.linf)
    set_lp_regularisation!(model, opt.lp)
    set_non_fixed_fees!(model, fees)
    set_risk_and_scalarise!(model, r, optimiser, opt, pr, plr, fees, b1; rd = rd)
    set_return_constraints!(model, ret, obj, pr; rd = rd)
    set_iplg_constraints!(model, plr)
    if sdp_asset_phylogeny
        set_sdp_phylogeny_constraints!(model, plr)
    end
    add_custom_constraint!(model, opt.ccnt, optimiser, attrs)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Route a mean uncertainty set into a [`JuMPOptimiser`](@ref)'s return estimator.

One of the two [Routing Targets](@ref PIPELINE_ROUTING_TARGETS) that names no plain field: the set lands in `ret.ucs`, and only an [`ArithmeticReturn`](@ref) can bound expected returns, so any other return estimator is an error rather than a silent drop.

Internal machinery — not part of the user-facing API.

# Related

  - [`pipe_route`](@ref)
  - [`route_sigma_ucs`](@ref)
"""
function pipe_route(cfg::JuMPOptimiser, ::Val{:mu_ucs}, v)
    @argcheck(isa(cfg.ret, ArithmeticReturn),
              ArgumentError("cannot route a mean uncertainty set into a $(Base.typename(typeof(cfg.ret)).wrapper); expected returns uncertainty requires an ArithmeticReturn return estimator"))
    return Accessors.set(cfg, Accessors.PropertyLens{:ret}(),
                         Accessors.set(cfg.ret, Accessors.PropertyLens{:ucs}(), v))
end
pipe_accepts(::JuMPOptimiser, ::Val{:mu_ucs})::Bool = true

export ProcessedJuMPOptimiserAttributes, JuMPOptimiser
