@concrete struct ProcessedJuMPOptimiserAttributes <: AbstractResult
    pr
    wb
    lt
    st
    lcsr
    ctr
    gcardr
    sgcardr
    smtx
    sgmtx
    slt
    sst
    sglt
    sgst
    tn
    fees
    plr
    ret
end
function assert_finite_nonnegative_real_or_vec(val::Number)
    @argcheck(isfinite(val))
    @argcheck(val > zero(val))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::VecNum)
    @argcheck(any(isfinite, val))
    @argcheck(any(x -> x > zero(x), val))
    @argcheck(all(x -> zero(x) <= x, val))
    return nothing
end
@concrete struct JuMPOptimiser <: BaseJuMPOptimisationEstimator
    pe
    slv
    wb
    bgt
    sbgt
    lt
    st
    lcse
    cte
    gcarde
    sgcarde
    smtx
    sgmtx
    slt
    sst
    sglt
    sgst
    tn
    fees
    sets
    tr
    ple
    ret
    sca
    ccnt
    cobj
    sc
    so
    ss
    card
    scard
    nea
    l1
    l2
    linf
    lp
    brt
    cle_pr
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
                       cle_pr::Bool = true, strict::Bool = false)
    return JuMPOptimiser(pe, slv, wb, bgt, sbgt, lt, st, lcse, cte, gcarde, sgcarde, smtx,
                         sgmtx, slt, sst, sglt, sgst, tn, fees, sets, tr, ple, ret, sca,
                         ccnt, cobj, sc, so, ss, card, scard, nea, l1, l2, linf, lp, brt,
                         cle_pr, strict)
end
function needs_previous_weights(opt::JuMPOptimiser)
    return (needs_previous_weights(opt.tn) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.tr) ||
            needs_previous_weights(opt.ccnt) ||
            needs_previous_weights(opt.cobj))
end
function factory(opt::JuMPOptimiser, w::AbstractVector)
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
function opt_view(opt::JuMPOptimiser, i, X::MatNum)
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
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1)
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
                                 cle_pr = opt.cle_pr)
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
                                cle_pr = opt.cle_pr)
    ret = factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr,
                                            smtx, sgmtx, slt, sst, sglt, sgst, tn, fees,
                                            plr, ret)
end
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(propertynames(opt), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getproperty.(opt, pnames))...)
end
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1)
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst, tn, fees, plr, ret) = processed_jump_optimiser_attributes(opt,
                                                                                                                                                rd;
                                                                                                                                                dims = dims)
    return JuMPOptimiser(; pe = pr, slv = opt.slv, wb = wb, bgt = opt.bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcse = lcsr, cte = ctr, gcarde = gcardr,
                         sgcarde = sgcardr, smtx = smtx, sgmtx = sgmtx, slt = slt,
                         sst = sst, sglt = sglt, sgst = sgst, tn = tn, fees = fees,
                         sets = opt.sets, tr = opt.tr, ple = plr, ret = ret, sca = opt.sca,
                         ccnt = opt.ccnt, cobj = opt.cobj, sc = opt.sc, so = opt.so,
                         ss = opt.ss, card = opt.card, nea = opt.nea, l1 = opt.l1,
                         l2 = opt.l2, linf = opt.linf, lp = opt.lp, brt = opt.brt,
                         cle_pr = opt.cle_pr, strict = opt.strict)
end

export ProcessedJuMPOptimiserAttributes, JuMPOptimiser
