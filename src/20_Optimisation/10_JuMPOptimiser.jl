"""
$(DocStringExtensions.TYPEDEF)

Intermediate result type storing processed optimisation attributes for `JuMPOptimiser`.

Used internally to pass processed constraints, bounds, and other data between optimisation stages.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`JuMPOptimiser`](@ref)
  - [`MeanRisk`](@ref)
"""
@concrete struct ProcessedJuMPOptimiserAttributes <: AbstractResult
    """
    Prior estimation result.
    """
    pr
    """
    Processed weight bounds constraints.
    """
    wb
    """
    Processed long threshold constraints.
    """
    lt
    """
    Processed short threshold constraints.
    """
    st
    """
    Processed linear constraints result.
    """
    lcsr
    """
    Processed centrality constraints result.
    """
    ctr
    """
    Processed group cardinality constraints result.
    """
    gcardr
    """
    Processed sub-group cardinality constraints result.
    """
    sgcardr
    """
    Asset sets matrix for group selection.
    """
    smtx
    """
    Asset sets matrix for sub-group selection.
    """
    sgmtx
    """
    Processed long threshold constraints for groups.
    """
    slt
    """
    Processed short threshold constraints for groups.
    """
    sst
    """
    Processed long threshold constraints for sub-groups.
    """
    sglt
    """
    Processed short threshold constraints for sub-groups.
    """
    sgst
    """
    Processed turnover constraints.
    """
    tn
    """
    Processed fees constraints.
    """
    fees
    """
    Processed phylogeny constraints result.
    """
    plr
    """
    Processed JuMP returns estimator.
    """
    ret
end
"""
    assert_finite_nonnegative_real_or_vec(val)

Assert that a value is a finite, non-negative real number or vector of such.

Throws an `ArgCheck` error if `val` contains non-finite or negative elements.

# Arguments

  - `val`: Scalar or vector to validate.

# Returns

  - `nothing` on success.

# Related

  - [`JuMPOptimiser`](@ref)
"""
function assert_finite_nonnegative_real_or_vec(val::Number)::Nothing
    @argcheck(isfinite(val))
    @argcheck(val > zero(val))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::VecNum)::Nothing
    @argcheck(any(isfinite, val))
    @argcheck(any(x -> x > zero(x), val))
    @argcheck(all(x -> zero(x) <= x, val))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Main JuMP-based portfolio optimiser configuration.

`JuMPOptimiser` collects all the inputs needed to formulate and solve a JuMP-based portfolio optimisation problem: prior estimator, solver, constraints, bounds, fees, tracking, regularisation, and more. It is intended to be passed to a higher-level optimiser such as [`MeanRisk`](@ref) or [`RiskBudgeting`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimiser(;
        pe::PrE_Pr = EmpiricalPrior(),
        slv::Slv_VecSlv,
        wb::Option{<:WbE_Wb} = WeightBounds(),
        bgt::Option{<:Num_BgtCE} = 1.0,
        sbgt::Option{<:Num_BgtRg} = nothing,
        lt::Option{<:BtE_Bt} = nothing,
        st::Option{<:BtE_Bt} = nothing,
        lcse::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
        cte::Option{<:Lc_CC_VecCC} = nothing,
        gcarde::Option{<:LcE_Lc} = nothing,
        sgcarde::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
        smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
        sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
        slt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sglt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        sgst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
        tn::Option{<:TnE_Tn_VecTnE_Tn} = nothing,
        fees::Option{<:FeesE_Fees} = nothing,
        sets::Option{<:AssetSets} = nothing,
        tr::Option{<:Tr_VecTr} = nothing,
        ple::Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
        ret::JuMPReturnsEstimator = ArithmeticReturn(),
        sca::NonHierarchicalScalariser = SumScalariser(),
        ccnt::Option{<:CustomJuMPConstraint} = nothing,
        cobj::Option{<:CustomJuMPObjective} = nothing,
        sc::Number = 1,
        so::Number = 1,
        ss::Option{<:Number} = nothing,
        card::Option{<:Integer} = nothing,
        scard::Option{<:Int_VecInt} = nothing,
        nea::Option{<:Number} = nothing,
        l1::Option{<:Number} = nothing,
        l2::Option{<:Number} = nothing,
        linf::Option{<:Number} = nothing,
        lp::Option{LpReg_VecLpReg} = nothing,
        brt::Bool = false,
        cle_pr::Bool = true,
        strict::Bool = false
    ) -> JuMPOptimiser

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a vector: `!isempty(slv)`.
  - If `bgt` is a number: `isfinite(bgt)`.
  - If `bgt` is a `BudgetCostEstimator`: `isnothing(sbgt)`.
  - If `sbgt` is a number: `isfinite(sbgt)` and `sbgt >= 0`.
  - If `cte` is a vector: `!isempty(cte)`.
  - If `card` is provided: `card > 0` and finite.
  - If `tn` or `tr` is a vector: each must be non-empty.
  - If `nea`, `l1`, `l2`, or `linf` is provided: each must be `> 0` and finite.
  - If `scard` is provided: compatible `smtx`, `slt`, `sst` sizes required.
  - If `sgcarde` is provided: compatible `sgmtx`, `sglt`, `sgst` sizes required.
  - If any estimator-type field (`wb`, `lt`, `fees`, etc.) is provided: `!isnothing(sets)`.

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
    Centring constraint estimator or constraint(s).
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
    $(field_dict[:nea])
    """
    nea
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
    function JuMPOptimiser(pe::PrE_Pr, slv::Slv_VecSlv, wb::Option{<:WbE_Wb},
                           bgt::Option{<:Num_BgtCE}, sbgt::Option{<:Num_BgtRg},
                           lt::Option{<:BtE_Bt}, st::Option{<:BtE_Bt},
                           lcse::Option{<:LcE_Lc_VecLcE_Lc}, cte::Option{<:Lc_CC_VecCC},
                           gcarde::Option{<:LcE_Lc}, sgcarde::Option{<:LcE_Lc_VecLcE_Lc},
                           smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           slt::Option{<:BtE_Bt_VecOptBtE_Bt},
                           sst::Option{<:BtE_Bt_VecOptBtE_Bt},
                           sglt::Option{<:BtE_Bt_VecOptBtE_Bt},
                           sgst::Option{<:BtE_Bt_VecOptBtE_Bt},
                           tn::Option{<:TnE_Tn_VecTnE_Tn}, fees::Option{<:FeesE_Fees},
                           sets::Option{<:AssetSets}, tr::Option{<:Tr_VecTr},
                           ple::Option{<:PlCE_PhC_VecPlCE_PlC}, ret::JuMPReturnsEstimator,
                           sca::NonHierarchicalScalariser,
                           ccnt::Option{<:CustomJuMPConstraint},
                           cobj::Option{<:CustomJuMPObjective}, sc::Number, so::Number,
                           ss::Option{<:Number}, card::Option{<:Integer},
                           scard::Option{<:Int_VecInt}, nea::Option{<:Number},
                           l1::Option{<:Number}, l2::Option{<:Number},
                           linf::Option{<:Number}, lp::Option{LpReg_VecLpReg}, brt::Bool,
                           cle_pr::Bool, strict::Bool)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        if isa(bgt, Number)
            @argcheck(isfinite(bgt))
        elseif isa(bgt, BudgetCostEstimator)
            @argcheck(isnothing(sbgt))
        end
        if isa(sbgt, Number)
            assert_nonempty_nonneg_finite_val(sbgt, :sbgt)
        end
        if isa(cte, AbstractVector)
            @argcheck(!isempty(cte))
        end
        if !isnothing(card)
            assert_nonempty_gt0_finite_val(card, :card)
        end
        if isa(tn, AbstractVector)
            @argcheck(!isempty(tn))
        end
        if isa(tr, AbstractVector)
            @argcheck(!isempty(tr))
        end
        if !isnothing(nea)
            assert_nonempty_gt0_finite_val(nea, :nea)
        end
        if !isnothing(l1)
            assert_nonempty_gt0_finite_val(l1, :l1)
        end
        if !isnothing(l2)
            assert_nonempty_gt0_finite_val(l2, :l2)
        end
        if !isnothing(linf)
            assert_nonempty_gt0_finite_val(linf, :linf)
        end
        if isa(lp, AbstractVector)
            @argcheck(!isempty(lp))
        end
        if isa(scard, Integer)
            assert_nonempty_gt0_finite_val(scard, :scard)
            @argcheck(isa(smtx, MatNum_ASetMatE))
            @argcheck(isa(slt, Option{<:BtE_Bt}))
            @argcheck(isa(sst, Option{<:BtE_Bt}))
        elseif isa(scard, VecInt)
            assert_nonempty_gt0_finite_val(scard, :scard)
            @argcheck(isa(smtx, AbstractVector))
            @argcheck(length(scard) == length(smtx))
            if isa(slt, AbstractVector)
                @argcheck(!isempty(slt))
                @argcheck(length(scard) == length(slt))
            end
            if isa(sst, AbstractVector)
                @argcheck(!isempty(sst))
                @argcheck(length(scard) == length(sst))
            end
        elseif isnothing(scard) && (isa(slt, BtE_Bt) || isa(sst, BtE_Bt))
            @argcheck(isa(smtx, MatNum_ASetMatE))
        elseif isnothing(scard) && (isa(slt, AbstractVector) || isa(sst, AbstractVector))
            @argcheck(isa(smtx, AbstractVector))
            @argcheck(!isempty(smtx))
            if isa(slt, AbstractVector)
                @argcheck(!isempty(slt))
                @argcheck(length(slt) == length(smtx))
            end
            if isa(sst, AbstractVector)
                @argcheck(!isempty(sst))
                @argcheck(length(sst) == length(smtx))
            end
        end
        if isa(sgcarde, LcE_Lc)
            @argcheck(isa(sgmtx, MatNum_ASetMatE))
            @argcheck(isa(sglt, Option{<:BtE_Bt}))
            @argcheck(isa(sgst, Option{<:BtE_Bt}))
            if isa(sgcarde, LinearConstraint) && isa(smtx, MatNum)
                N = size(smtx, 1)
                N_ineq = !isnothing(sgcarde.ineq) ? length(sgcarde.B_ineq) : 0
                N_eq = !isnothing(sgcarde.eq) ? length(sgcarde.B_eq) : 0
                @argcheck(N == N_ineq + N_eq)
            end
        elseif isa(sgcarde, AbstractVector)
            @argcheck(!isempty(sgcarde))
            @argcheck(isa(sgmtx, AbstractVector))
            @argcheck(!isempty(sgmtx))
            @argcheck(length(sgcarde) == length(sgmtx))
            if isa(sglt, AbstractVector)
                @argcheck(!isempty(sglt))
                @argcheck(length(sgcarde) == length(sglt))
            end
            if isa(sgst, AbstractVector)
                @argcheck(!isempty(sgst))
                @argcheck(length(sgcarde) == length(sgst))
            end
            for (sgc, smt) in zip(sgcarde, sgmtx)
                if isa(sgc, LinearConstraint) && isa(smt, MatNum)
                    N = size(smt, 1)
                    N_ineq = !isnothing(sgc.ineq) ? length(sgc.B_ineq) : 0
                    N_eq = !isnothing(sgc.eq) ? length(sgc.B_eq) : 0
                    @argcheck(N == N_ineq + N_eq)
                end
            end
        elseif isnothing(sgcarde) && (isa(sglt, BtE_Bt) || isa(sgst, BtE_Bt))
            @argcheck(isa(sgmtx, MatNum_ASetMatE))
        elseif isnothing(sgcarde) &&
               (isa(sglt, AbstractVector) || isa(sgst, AbstractVector))
            @argcheck(isa(sgmtx, AbstractVector))
            @argcheck(!isempty(sgmtx))
            if isa(sglt, AbstractVector)
                @argcheck(!isempty(sglt))
                @argcheck(length(sglt) == length(sgmtx))
            end
            if isa(sgst, AbstractVector)
                @argcheck(!isempty(sgst))
                @argcheck(length(sgst) == length(sgmtx))
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
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                   typeof(lt), typeof(st), typeof(lcse), typeof(cte), typeof(gcarde),
                   typeof(sgcarde), typeof(smtx), typeof(sgmtx), typeof(slt), typeof(sst),
                   typeof(sglt), typeof(sgst), typeof(tn), typeof(fees), typeof(sets),
                   typeof(tr), typeof(ple), typeof(ret), typeof(sca), typeof(ccnt),
                   typeof(cobj), typeof(sc), typeof(so), typeof(ss), typeof(card),
                   typeof(scard), typeof(nea), typeof(l1), typeof(l2), typeof(linf),
                   typeof(lp), typeof(brt), typeof(cle_pr), typeof(strict)}(pe, slv, wb,
                                                                            bgt, sbgt, lt,
                                                                            st, lcse, cte,
                                                                            gcarde, sgcarde,
                                                                            smtx, sgmtx,
                                                                            slt, sst, sglt,
                                                                            sgst, tn, fees,
                                                                            sets, tr, ple,
                                                                            ret, sca, ccnt,
                                                                            cobj, sc, so,
                                                                            ss, card, scard,
                                                                            nea, l1, l2,
                                                                            linf, lp, brt,
                                                                            cle_pr, strict)
    end
end
function JuMPOptimiser(; pe::PrE_Pr = EmpiricalPrior(), slv::Slv_VecSlv,
                       wb::Option{<:WbE_Wb} = WeightBounds(),
                       bgt::Option{<:Num_BgtCE} = 1.0, sbgt::Option{<:Num_BgtRg} = nothing,
                       lt::Option{<:BtE_Bt} = nothing, st::Option{<:BtE_Bt} = nothing,
                       lcse::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
                       cte::Option{<:Lc_CC_VecCC} = nothing,
                       gcarde::Option{<:LcE_Lc} = nothing,
                       sgcarde::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
                       smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                       sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                       slt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sglt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sgst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       tn::Option{<:TnE_Tn_VecTnE_Tn} = nothing,
                       fees::Option{<:FeesE_Fees} = nothing,
                       sets::Option{<:AssetSets} = nothing,
                       tr::Option{<:Tr_VecTr} = nothing,
                       ple::Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       sca::NonHierarchicalScalariser = SumScalariser(),
                       ccnt::Option{<:CustomJuMPConstraint} = nothing,
                       cobj::Option{<:CustomJuMPObjective} = nothing, sc::Number = 1,
                       so::Number = 1, ss::Option{<:Number} = nothing,
                       card::Option{<:Integer} = nothing,
                       scard::Option{<:Int_VecInt} = nothing,
                       nea::Option{<:Number} = nothing, l1::Option{<:Number} = nothing,
                       l2::Option{<:Number} = nothing, linf::Option{<:Number} = nothing,
                       lp::Option{<:LpReg_VecLpReg} = nothing, brt::Bool = false,
                       cle_pr::Bool = true, strict::Bool = false)::JuMPOptimiser
    return JuMPOptimiser(pe, slv, wb, bgt, sbgt, lt, st, lcse, cte, gcarde, sgcarde, smtx,
                         sgmtx, slt, sst, sglt, sgst, tn, fees, sets, tr, ple, ret, sca,
                         ccnt, cobj, sc, so, ss, card, scard, nea, l1, l2, linf, lp, brt,
                         cle_pr, strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (turnover, fees, tracking, custom constraint, or custom objective).
"""
function needs_previous_weights(opt::JuMPOptimiser)
    return (needs_previous_weights(opt.tn) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.tr) ||
            needs_previous_weights(opt.ccnt) ||
            needs_previous_weights(opt.cobj))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build an updated [`JuMPOptimiser`](@ref) with all estimator fields that track previous weights updated via `factory` using `w`.
"""
function factory(opt::JuMPOptimiser, w::AbstractVector)::JuMPOptimiser
    tn = factory(opt.tn, w)
    fees = factory(opt.fees, w)
    tr = factory(opt.tr, w)
    ccnt = factory(opt.ccnt, w)
    cobj = factory(opt.cobj, w)
    return JuMPOptimiser(; pe = opt.pe, slv = opt.slv, wb = opt.wb, bgt = opt.bgt,
                         sbgt = opt.sbgt, lt = opt.lt, st = opt.st, lcse = opt.lcse,
                         cte = opt.cte, gcarde = opt.gcarde, sgcarde = opt.sgcarde,
                         smtx = opt.smtx, sgmtx = opt.sgmtx, slt = opt.slt, sst = opt.sst,
                         sglt = opt.sglt, sgst = opt.sgst, tn = tn, fees = fees,
                         sets = opt.sets, tr = tr, ple = opt.ple, ret = opt.ret,
                         sca = opt.sca, ccnt = ccnt, cobj = cobj, sc = opt.sc, so = opt.so,
                         ss = opt.ss, card = opt.card, scard = opt.scard, nea = opt.nea,
                         l1 = opt.l1, l2 = opt.l2, linf = opt.linf, lp = opt.lp,
                         brt = opt.brt, cle_pr = opt.cle_pr, strict = opt.strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`JuMPOptimiser`](@ref) for asset index set `i` and returns matrix `X`.

Slices all per-asset estimator fields (prior, bounds, thresholds, turnover, fees, tracking, custom constraint/objective) to the cluster `i`, leaving solver and scalar parameters unchanged.
"""
function opt_view(opt::JuMPOptimiser, i, X::MatNum)::JuMPOptimiser
    X = isa(opt.pe, AbstractPriorResult) ? opt.pe.X : X
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    bgt = budget_view(opt.bgt, i)
    lt = threshold_view(opt.lt, i)
    st = threshold_view(opt.st, i)
    if opt.smtx === opt.sgmtx
        smtx = sgmtx = asset_sets_matrix_view(opt.smtx, i)
    else
        smtx = asset_sets_matrix_view(opt.smtx, i)
        sgmtx = asset_sets_matrix_view(opt.sgmtx, i)
    end
    if opt.slt === opt.sglt
        slt = sglt = threshold_view(opt.slt, i)
    else
        slt = threshold_view(opt.slt, i)
        sglt = threshold_view(opt.sglt, i)
    end
    if opt.sst === opt.sgst
        sst = sgst = threshold_view(opt.sst, i)
    else
        sst = threshold_view(opt.sst, i)
        sgst = threshold_view(opt.sgst, i)
    end
    tn = turnover_view(opt.tn, i)
    sets = asset_sets_view(opt.sets, i)
    fees = fees_view(opt.fees, i)
    tr = tracking_view(opt.tr, i, X)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcse = opt.lcse, cte = opt.cte,
                         gcarde = opt.gcarde, sgcarde = opt.sgcarde, smtx = smtx,
                         sgmtx = sgmtx, slt = slt, sst = sst, sglt = sglt, sgst = sgst,
                         tn = tn, fees = fees, sets = sets, tr = tr, ple = opt.ple,
                         ret = ret, sca = opt.sca, ccnt = ccnt, cobj = cobj, sc = opt.sc,
                         so = opt.so, ss = opt.ss, card = opt.card, scard = opt.scard,
                         nea = opt.nea, l1 = opt.l1, l2 = opt.l2, linf = opt.linf,
                         lp = opt.lp, brt = opt.brt, cle_pr = opt.cle_pr,
                         strict = opt.strict)
end
"""
    processed_jump_optimiser_attributes(opt, rd; kwargs...)

Compute and process all optimiser attributes from a JuMP optimiser and returns data.

Internal function that applies the `factory` transform and resolves view slices for all estimator fields of `opt` given the returns data `rd`.

# Arguments

  - `opt`: [`JuMPOptimiser`](@ref) configuration.
  - `rd`: [`ReturnsResult`](@ref) data.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Named tuple of processed attributes.

# Related

  - [`JuMPOptimiser`](@ref)
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
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr,
                                            smtx, sgmtx, slt, sst, sglt, sgst, tn, fees,
                                            plr, ret)
end
"""
    no_bounds_optimiser(opt, args...)

Create a version of the JuMP optimiser with bounds removed for unbounded sub-problems.

Internal helper used in risk frontier construction sub-problems. Strips weight and risk bounds from `opt` so the sub-problem is unconstrained.

# Arguments

  - `opt`: [`JuMPOptimiser`](@ref) configuration.
  - `args...`: Additional arguments.

# Returns

  - [`JuMPOptimiser`](@ref) without bounds.

# Related

  - [`no_bounds_returns_estimator`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(propertynames(opt), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getproperty.(opt, pnames))...)
end
"""
    jump_optimiser_from_attributes(opt, attrs)

Repackage a [`ProcessedJuMPOptimiserAttributes`](@ref) back into a [`JuMPOptimiser`](@ref),
mapping the result-named fields onto the optimiser's estimator-named slots (`lcsr` → `lcse`,
`plr` → `ple`, `pr` → `pe`, …) and carrying the remaining settings through from `opt`.

Used where a fully-processed optimiser object is needed (e.g. the inner sub-problems of
[`near_optimal_centering_setup`](@ref)) while the same `attrs` is reused directly by
[`_assemble_jump_model!`](@ref) — so the processing is done once and never round-tripped.

# Related

  - [`processed_jump_optimiser`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
"""
function jump_optimiser_from_attributes(opt::JuMPOptimiser,
                                        attrs::ProcessedJuMPOptimiserAttributes)
    return JuMPOptimiser(; pe = attrs.pr, slv = opt.slv, wb = attrs.wb, bgt = opt.bgt,
                         sbgt = opt.sbgt, lt = attrs.lt, st = attrs.st, lcse = attrs.lcsr,
                         cte = attrs.ctr, gcarde = attrs.gcardr, sgcarde = attrs.sgcardr,
                         smtx = attrs.smtx, sgmtx = attrs.sgmtx, slt = attrs.slt,
                         sst = attrs.sst, sglt = attrs.sglt, sgst = attrs.sgst,
                         tn = attrs.tn, fees = attrs.fees, sets = opt.sets, tr = opt.tr,
                         ple = attrs.plr, ret = attrs.ret, sca = opt.sca, ccnt = opt.ccnt,
                         cobj = opt.cobj, sc = opt.sc, so = opt.so, ss = opt.ss,
                         card = opt.card, nea = opt.nea, l1 = opt.l1, l2 = opt.l2,
                         linf = opt.linf, lp = opt.lp, brt = opt.brt, cle_pr = opt.cle_pr,
                         strict = opt.strict)
end
"""
    processed_jump_optimiser(opt, rd; dims = 1)

Build a fully processed `JuMPOptimiser` from raw configuration and returns data: computes the
[`ProcessedJuMPOptimiserAttributes`](@ref) and repackages them via
[`jump_optimiser_from_attributes`](@ref).

# Related

  - [`JuMPOptimiser`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`jump_optimiser_from_attributes`](@ref)
"""
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1,
                                  kwargs...)
    attrs = processed_jump_optimiser_attributes(opt, rd; dims = dims, kwargs...)
    return jump_optimiser_from_attributes(opt, attrs)
end

"""
    _set_risk_and_scalarise!(model, r, optimiser, opt, pr, pl, fees, extra; rd)

Add the risk-measure constraints and scalarise the combined risk expression, as one step of
[`_assemble_jump_model!`](@ref). Dispatched on the risk measure: when `r === nothing` (e.g.
[`RelaxedRiskBudgeting`](@ref), whose risk lives in its head) this is a no-op. `extra` is the
optional trailing-argument tuple (empty, or `(b1,)` for [`FactorRiskContribution`](@ref)),
splatted into the risk builder so the non-factor call is reproduced exactly.
"""
function _set_risk_and_scalarise!(::JuMP.Model, ::Nothing, args...; kwargs...)
    return nothing
end
function _set_risk_and_scalarise!(model::JuMP.Model, r, optimiser, opt, pr, pl, fees, extra;
                                  rd)
    set_risk_constraints!(model, r, optimiser, pr, pl, fees, extra...; rd = rd)
    scalarise_risk_expression!(model, opt.sca)
    return nothing
end

"""
    _assemble_jump_model!(model, optimiser, opt, attrs, rd;
                          r = nothing, b1 = nothing, obj = MinimumRisk(),
                          miprb_flag = false, sdp_phylogeny = true)

Run the invariant model-assembly sequence shared by every single-JuMP-model Optimisation
Estimator — the steps between shaping the weight variables (the per-optimiser *head*) and
setting the objective/solving (the per-optimiser *tail*). See `Model Assembly` in
`CONTEXT.md` and [ADR 0008](../../docs/adr/0008-jump-model-assembly.md).

Constraint *results* are read from `attrs` (a [`ProcessedJuMPOptimiserAttributes`](@ref));
scalar *settings* from `opt` (the [`JuMPOptimiser`](@ref)); `optimiser` is the dispatch
object for the risk, tracking, and custom-constraint builders. Per-optimiser context that
varies inside the middle rides in as optional keyword arguments:

  - `r`: the risk measure(s), or `nothing` when the optimiser carries none.
  - `b1`: the factor loading matrix threaded into tracking and risk by
    [`FactorRiskContribution`](@ref); `nothing` otherwise (then no trailing argument is
    passed, reproducing the non-factor calls exactly).
  - `obj`: the objective used by the return constraints.
  - `miprb_flag`: the mixed-integer risk-budgeting flag consumed by
    [`set_mip_constraints!`](@ref).
  - `sdp_phylogeny`: whether to apply the standard (asset-space) SDP phylogeny constraints.
    [`FactorRiskContribution`](@ref) passes `false` because it applies its own factor-space
    phylogeny constraints in its tail instead.

The head must have populated the Model State (the `w`/`k` variables) before this is called.
Returns `nothing`; mutates `model`.
"""
function _assemble_jump_model!(model::JuMP.Model, optimiser::JuMPOptimisationEstimator,
                               opt::JuMPOptimiser, attrs::ProcessedJuMPOptimiserAttributes,
                               rd::ReturnsResult; r = nothing, b1 = nothing,
                               obj::ObjectiveFunction = MinimumRisk(),
                               miprb_flag::Bool = false, sdp_phylogeny::Bool = true)
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst, tn, fees, plr, ret) = attrs
    extra = isnothing(b1) ? () : (b1,)
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, opt.card, gcardr, plr, lt, st, fees, opt.ss, miprb_flag)
    set_smip_constraints!(model, wb, opt.scard, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst,
                          opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, opt.tr, optimiser, plr, fees, extra...;
                                    rd = rd)
    set_number_effective_assets!(model, opt.nea)
    set_l1_regularisation!(model, opt.l1)
    set_l2_regularisation!(model, opt.l2)
    set_linf_regularisation!(model, opt.linf)
    set_lp_regularisation!(model, opt.lp)
    set_non_fixed_fees!(model, fees)
    _set_risk_and_scalarise!(model, r, optimiser, opt, pr, plr, fees, extra; rd = rd)
    set_return_constraints!(model, ret, obj, pr; rd = rd)
    if sdp_phylogeny
        set_sdp_phylogeny_constraints!(model, plr)
    end
    add_custom_constraint!(model, opt.ccnt, optimiser, pr)
    return nothing
end

export ProcessedJuMPOptimiserAttributes, JuMPOptimiser
