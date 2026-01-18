struct ProcessedJuMPOptimiserAttributes{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                                        T13, T14, T15, T16, T17, T18} <: AbstractResult
    pr::T1
    wb::T2
    lt::T3
    st::T4
    lcs::T5
    ct::T6
    gcard::T7
    sgcard::T8
    smtx::T9
    sgmtx::T10
    slt::T11
    sst::T12
    sglt::T13
    sgst::T14
    pl::T15
    tn::T16
    fees::T17
    ret::T18
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
struct JuMPOptimiser{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16,
                     T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
                     T31, T32, T33, T34, T35} <: BaseJuMPOptimisationEstimator
    pr::T1 # PriorEstimator
    slv::T2
    wb::T3 # WeightBounds
    bgt::T4 # BudgetRange
    sbgt::T5 # LongShortSum
    lt::T6 # l threshold
    st::T7
    lcs::T8
    ct::T9
    gcard::T10
    sgcard::T11
    smtx::T12
    sgmtx::T13
    slt::T14
    sst::T15
    sglt::T16
    sgst::T17
    sets::T18
    pl::T19
    tn::T20 # Turnover
    te::T21 # TrackingError
    fees::T22
    ret::T23
    sca::T24
    ccnt::T25
    cobj::T26
    sc::T27
    so::T28
    ss::T29
    card::T30
    scard::T31
    nea::T32
    l1::T33
    l2::T34
    strict::T35
    function JuMPOptimiser(pr::PrE_Pr, slv::Slv_VecSlv, wb::Option{<:WbE_Wb},
                           bgt::Option{<:Num_BgtCE}, sbgt::Option{<:Num_BgtRg},
                           lt::Option{<:BtE_Bt}, st::Option{<:BtE_Bt},
                           lcs::Option{<:LcE_Lc_VecLcE_Lc}, ct::Option{<:Lc_CC_VecCC},
                           gcard::Option{<:LcE_Lc}, sgcard::Option{<:LcE_Lc_VecLcE_Lc},
                           smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE},
                           slt::Option{<:BtE_Bt_VecOptBtE_Bt},
                           sst::Option{<:BtE_Bt_VecOptBtE_Bt},
                           sglt::Option{<:BtE_Bt_VecOptBtE_Bt},
                           sgst::Option{<:BtE_Bt_VecOptBtE_Bt}, sets::Option{<:AssetSets},
                           pl::Option{<:PlCE_PhC_VecPlCE_PlC},
                           tn::Option{<:TnE_Tn_VecTnE_Tn}, te::Option{<:Tr_VecTr},
                           fees::Option{<:FeesE_Fees}, ret::JuMPReturnsEstimator,
                           sca::NonHierarchicalScalariser,
                           ccnt::Option{<:CustomJuMPConstraint},
                           cobj::Option{<:CustomJuMPObjective}, sc::Number, so::Number,
                           ss::Option{<:Number}, card::Option{<:Integer},
                           scard::Option{<:Int_VecInt}, nea::Option{<:Number},
                           l1::Option{<:Number}, l2::Option{<:Number}, strict::Bool)
        if isa(bgt, Number)
            @argcheck(isfinite(bgt))
        elseif isa(bgt, BudgetCostEstimator)
            @argcheck(isnothing(sbgt))
        end
        if isa(sbgt, Number)
            @argcheck(isfinite(sbgt))
            @argcheck(sbgt >= 0)
        end
        if isa(ct, AbstractVector)
            @argcheck(!isempty(ct))
        end
        if !isnothing(card)
            @argcheck(isfinite(card))
            @argcheck(card > 0)
        end
        if isa(scard, Integer)
            @argcheck(isfinite(scard))
            @argcheck(scard > 0)
            @argcheck(isa(smtx, MatNum_ASetMatE))
            @argcheck(isa(slt, Option{<:BtE_Bt}))
            @argcheck(isa(sst, Option{<:BtE_Bt}))
        elseif isa(scard, VecInt)
            @argcheck(!isempty(scard))
            @argcheck(all(isfinite, scard))
            @argcheck(all(x -> x > 0, scard))
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
        if isa(sgcard, LcE_Lc)
            @argcheck(isa(sgmtx, MatNum_ASetMatE))
            @argcheck(isa(sglt, Option{<:BtE_Bt}))
            @argcheck(isa(sgst, Option{<:BtE_Bt}))
            if isa(sgcard, LinearConstraint) && isa(smtx, MatNum)
                N = size(smtx, 1)
                N_ineq = !isnothing(sgcard.ineq) ? length(sgcard.B_ineq) : 0
                N_eq = !isnothing(sgcard.eq) ? length(sgcard.B_eq) : 0
                @argcheck(N == N_ineq + N_eq)
            end
        elseif isa(sgcard, AbstractVector)
            @argcheck(!isempty(sgcard))
            @argcheck(isa(sgmtx, AbstractVector))
            @argcheck(!isempty(sgmtx))
            @argcheck(length(sgcard) == length(sgmtx))
            if isa(sglt, AbstractVector)
                @argcheck(!isempty(sglt))
                @argcheck(length(sgcard) == length(sglt))
            end
            if isa(sgst, AbstractVector)
                @argcheck(length(sgcard) == length(sgst))
            end
            for (sgc, smt) in zip(sgcard, sgmtx)
                if isa(sgc, LinearConstraint) && isa(smt, MatNum)
                    N = size(smt, 1)
                    N_ineq = !isnothing(sgc.ineq) ? length(sgc.B_ineq) : 0
                    N_eq = !isnothing(sgc.eq) ? length(sgc.B_eq) : 0
                    @argcheck(N == N_ineq + N_eq)
                end
            end
        elseif isnothing(sgcard) && (isa(sglt, BtE_Bt) || isa(sgst, BtE_Bt))
            @argcheck(isa(sgmtx, MatNum_ASetMatE))
        elseif isnothing(sgcard) && (isa(sglt, AbstractVector) || isa(sgst, AbstractVector))
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
           isa(lt, BuyInThresholdEstimator) ||
           isa(st, BuyInThresholdEstimator) ||
           isa(lcs, LinearConstraintEstimator) ||
           isa(ct, LinearConstraintEstimator) ||
           isa(gcard, LinearConstraintEstimator) ||
           !isa(sgcard, Option{<:Lc_VecLc}) ||
           !isnothing(scard) ||
           isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets))
        end
        if isa(tn, AbstractVector)
            @argcheck(!isempty(tn))
        end
        if isa(te, AbstractVector)
            @argcheck(!isempty(te))
        end
        if !isnothing(nea)
            @argcheck(nea > zero(nea))
        end
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        return new{typeof(pr), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                   typeof(lt), typeof(st), typeof(lcs), typeof(ct), typeof(gcard),
                   typeof(sgcard), typeof(smtx), typeof(sgmtx), typeof(slt), typeof(sst),
                   typeof(sglt), typeof(sgst), typeof(sets), typeof(pl), typeof(tn),
                   typeof(te), typeof(fees), typeof(ret), typeof(sca), typeof(ccnt),
                   typeof(cobj), typeof(sc), typeof(so), typeof(ss), typeof(card),
                   typeof(scard), typeof(nea), typeof(l1), typeof(l2), typeof(strict)}(pr,
                                                                                       slv,
                                                                                       wb,
                                                                                       bgt,
                                                                                       sbgt,
                                                                                       lt,
                                                                                       st,
                                                                                       lcs,
                                                                                       ct,
                                                                                       gcard,
                                                                                       sgcard,
                                                                                       smtx,
                                                                                       sgmtx,
                                                                                       slt,
                                                                                       sst,
                                                                                       sglt,
                                                                                       sgst,
                                                                                       sets,
                                                                                       pl,
                                                                                       tn,
                                                                                       te,
                                                                                       fees,
                                                                                       ret,
                                                                                       sca,
                                                                                       ccnt,
                                                                                       cobj,
                                                                                       sc,
                                                                                       so,
                                                                                       ss,
                                                                                       card,
                                                                                       scard,
                                                                                       nea,
                                                                                       l1,
                                                                                       l2,
                                                                                       strict)
    end
end
function JuMPOptimiser(; pr::PrE_Pr = EmpiricalPrior(), slv::Slv_VecSlv,
                       wb::Option{<:WbE_Wb} = WeightBounds(),
                       bgt::Option{<:Num_BgtCE} = 1.0, sbgt::Option{<:Num_BgtRg} = nothing,
                       lt::Option{<:BtE_Bt} = nothing, st::Option{<:BtE_Bt} = nothing,
                       lcs::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
                       ct::Option{<:Lc_CC_VecCC} = nothing,
                       gcard::Option{<:LcE_Lc} = nothing,
                       sgcard::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
                       smtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                       sgmtx::Option{<:MatNum_ASetMatE_VecMatNum_ASetMatE} = nothing,
                       slt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sglt::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sgst::Option{<:BtE_Bt_VecOptBtE_Bt} = nothing,
                       sets::Option{<:AssetSets} = nothing,
                       pl::Option{<:PlCE_PhC_VecPlCE_PlC} = nothing,
                       tn::Option{<:TnE_Tn_VecTnE_Tn} = nothing,
                       te::Option{<:Tr_VecTr} = nothing,
                       fees::Option{<:FeesE_Fees} = nothing,
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       sca::NonHierarchicalScalariser = SumScalariser(),
                       ccnt::Option{<:CustomJuMPConstraint} = nothing,
                       cobj::Option{<:CustomJuMPObjective} = nothing, sc::Number = 1,
                       so::Number = 1, ss::Option{<:Number} = nothing,
                       card::Option{<:Integer} = nothing,
                       scard::Option{<:Int_VecInt} = nothing,
                       nea::Option{<:Number} = nothing, l1::Option{<:Number} = nothing,
                       l2::Option{<:Number} = nothing, strict::Bool = false)
    return JuMPOptimiser(pr, slv, wb, bgt, sbgt, lt, st, lcs, ct, gcard, sgcard, smtx,
                         sgmtx, slt, sst, sglt, sgst, sets, pl, tn, te, fees, ret, sca,
                         ccnt, cobj, sc, so, ss, card, scard, nea, l1, l2, strict)
end
function opt_view(opt::JuMPOptimiser, i, X::MatNum)
    X = isa(opt.pr, AbstractPriorResult) ? opt.pr.X : X
    pr = prior_view(opt.pr, i)
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
    sets = nothing_asset_sets_view(opt.sets, i)
    tn = turnover_view(opt.tn, i)
    te = tracking_view(opt.te, i, X)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(; pr = pr, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = opt.lcs, ct = opt.ct, gcard = opt.gcard,
                         sgcard = opt.sgcard, smtx = smtx, sgmtx = sgmtx, slt = slt,
                         sst = sst, sglt = sglt, sgst = sgst, sets = sets, pl = opt.pl,
                         tn = tn, te = te, fees = fees, ret = ret, sca = opt.sca,
                         ccnt = ccnt, cobj = cobj, sc = opt.sc, so = opt.so, ss = opt.ss,
                         card = opt.card, scard = opt.scard, nea = opt.nea, l1 = opt.l1,
                         l2 = opt.l2, strict = opt.strict)
end
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1)
    pr = prior(opt.pr, rd; dims = dims)
    datatype = eltype(pr.X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(pr.X, 2), strict = opt.strict,
                                   datatype = datatype)
    lt = threshold_constraints(opt.lt, opt.sets; datatype = datatype, strict = opt.strict)
    st = threshold_constraints(opt.st, opt.sets; datatype = datatype, strict = opt.strict)
    lcs = linear_constraints(opt.lcs, opt.sets; datatype = datatype, strict = opt.strict)
    ct = centrality_constraints(opt.ct, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    gcard = linear_constraints(opt.gcard, opt.sets; datatype = Int, strict = opt.strict)
    sgcard = linear_constraints(opt.sgcard, opt.sets; datatype = Int, strict = opt.strict)
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
    pl = phylogeny_constraints(opt.pl, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    tn = turnover_constraints(opt.tn, opt.sets; datatype = datatype, strict = opt.strict)
    fees = fees_constraints(opt.fees, opt.sets; datatype = datatype, strict = opt.strict)
    ret = factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs, ct, gcard, sgcard, smtx,
                                            sgmtx, slt, sst, sglt, sgst, pl, tn, fees, ret)
end
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(propertynames(opt), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getproperty.(opt, pnames))...)
end
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1)
    (; pr, wb, lt, st, lcs, ct, gcard, sgcard, smtx, sgmtx, slt, sst, sglt, sgst, pl, tn, fees, ret) = processed_jump_optimiser_attributes(opt,
                                                                                                                                           rd;
                                                                                                                                           dims = dims)
    return JuMPOptimiser(; pr = pr, slv = opt.slv, wb = wb, bgt = opt.bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = lcs, ct = ct, gcard = gcard,
                         sgcard = sgcard, smtx = smtx, sgmtx = sgmtx, slt = slt, sst = sst,
                         sglt = sglt, sgst = sgst, sets = opt.sets, pl = pl, tn = tn,
                         te = opt.te, fees = fees, ret = ret, sca = opt.sca,
                         ccnt = opt.ccnt, cobj = opt.cobj, sc = opt.sc, so = opt.so,
                         ss = opt.ss, card = opt.card, nea = opt.nea, l1 = opt.l1,
                         l2 = opt.l2, strict = opt.strict)
end

export ProcessedJuMPOptimiserAttributes, JuMPOptimiser
