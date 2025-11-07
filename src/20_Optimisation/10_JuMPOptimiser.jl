struct ProcessedJuMPOptimiserAttributes{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                                        T13, T14, T15, T16, T17, T18} <: AbstractResult
    pr::T1
    wb::T2
    lt::T3
    st::T4
    lcs::T5
    cent::T6
    gcard::T7
    sgcard::T8
    smtx::T9
    sgmtx::T10
    slt::T11
    sst::T12
    sglt::T13
    sgst::T14
    plg::T15
    tn::T16
    fees::T17
    ret::T18
end
struct ProcessedFactorRiskBudgetingAttributes{T1, T2, T3} <: AbstractResult
    rkb::T1
    b1::T2
    rr::T3
end
struct ProcessedAssetRiskBudgetingAttributes{T1} <: AbstractResult
    rkb::T1
end
struct JuMPOptimisation{T1, T2, T3, T4, T5, T6} <: OptimisationResult
    oe::T1
    pa::T2
    retcode::T3
    sol::T4
    model::T5
    fb::T6
end
function opt_attempt_factory(res::JuMPOptimisation, fb)
    return JuMPOptimisation(res.oe, res.pa, res.retcode, res.sol, res.model, fb)
end
struct JuMPOptimisationFactorRiskContribution{T1, T2, T3, T4, T5, T6, T7, T8} <:
       OptimisationResult
    oe::T1
    pa::T2
    rr::T3
    frc_plg::T4
    retcode::T5
    sol::T6
    model::T7
    fb::T8
end
function opt_attempt_factory(res::JuMPOptimisationFactorRiskContribution, fb)
    return JuMPOptimisationFactorRiskContribution(res.oe, res.pa, res.rr, res.frc_plg,
                                                  res.retcode, res.sol, res.model, fb)
end
struct JuMPOptimisationRiskBudgeting{T1, T2, T3, T4, T5, T6, T7} <: OptimisationResult
    oe::T1
    pa::T2
    prb::T3
    retcode::T4
    sol::T5
    model::T6
    fb::T7
end
function opt_attempt_factory(res::JuMPOptimisationRiskBudgeting, fb)
    return JuMPOptimisationRiskBudgeting(res.oe, res.pa, res.prb, res.retcode, res.sol,
                                         res.model, fb)
end
function Base.getproperty(r::JuMPOptimisation, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? getfield(r.sol, :w) : getfield.(r.sol, :w)
    elseif sym in (:oe, :pa, :retcode, :sol, :model, :fb)
        getfield(r, sym)
    else
        getfield(r.pa, sym)
    end
end
function Base.getproperty(r::JuMPOptimisationFactorRiskContribution, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? getfield(r.sol, :w) : getfield.(r.sol, :w)
    elseif sym in (:oe, :pa, :rr, :frc_plg, :retcode, :sol, :model, :fb)
        getfield(r, sym)
    else
        getfield(r.pa, sym)
    end
end
function Base.getproperty(r::JuMPOptimisationRiskBudgeting, sym::Symbol)
    return if sym == :w
        r.sol.w
    elseif sym in (:oe, :pa, :prb, :retcode, :sol, :model, :fb)
        getfield(r, sym)
    else
        getfield(r.pa, sym)
    end
end
function assert_finite_nonnegative_real_or_vec(val::Number)
    @argcheck(isfinite(val))
    @argcheck(val > zero(val))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::NumVec)
    @argcheck(any(isfinite, val))
    @argcheck(any(x -> x > zero(x), val))
    @argcheck(all(x -> zero(x) <= x, val))
    return nothing
end
#=
opt: JuMPOptimiser
    pe: prior estimator.
        ce: covariance estimator.
            ce: covariance estimator.
                me: mean estimator.
                    w: weights vector for mean estimation.
                ce: covariance estimator.
                    w: weights vector for covariance estimation.
                alg: full or downside (semi) covariance estimator.
            mp: matrix processing.
                pdm: nearest correlation matrix projection.
                denoise: covariance denoising.
                detone: covariance detoning.
                alg: matrix processing algorithm.
        me: mean estimator.
            w: weights vector for mean estimation.
        horizon: investment horizon.
    slv: solver (explained above).
    wb: weight bounds.
        lb: lower bounds for all assets.
        ub: upper bounds for all assets.
    bgt: budget constraint (sum of weights).
    sbgt: short budget constraint (absolute value sum of negative weights).
    lt: long weight buy in threshold(s) result(s) or estimators.
    st: absolute value of the short weight buy in threshold.
    lcs: linear constraint estimator(s) and/or result(s).
    cent: centrality constraint result(s) and/or estimator(s).
    gcard: group cardinality constraint result(s) and/or estimator(s).
    sgcard: set group cardinality constraint result(s) and/or estimator(s).
    smtx: set matrix estimator(s) and/or result(s).
    sgmtx: group set matrix estimator(s) and/or result(s).
    slt: set long buy in threshold.
    sst: set short buy in threshold.
    sglt: set group long buy in threshold.
    sgst: set group short buy in threshold.
    sets: asset sets.
    plg: phylogeny constraint estimator(s) and/or result(s).
    tn:
    te:
    fees:
=#
struct JuMPOptimiser{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16,
                     T17, T18, T19, T20, T21, T22, T23, T24, T25, T26, T27, T28, T29, T30,
                     T31, T32, T33, T34, T35} <: BaseJuMPOptimisationEstimator
    pe::T1 # PriorEstimator
    slv::T2
    wb::T3 # WeightBounds
    bgt::T4 # BudgetRange
    sbgt::T5 # LongShortSum
    lt::T6 # l threshold
    st::T7
    lcs::T8
    cent::T9
    gcard::T10
    sgcard::T11
    smtx::T12
    sgmtx::T13
    slt::T14
    sst::T15
    sglt::T16
    sgst::T17
    sets::T18
    plg::T19
    tn::T20 # Turnover
    te::T21 # TrackingError
    fees::T22
    ret::T23
    sce::T24
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
    function JuMPOptimiser(pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                           slv::Union{<:Solver, <:VecSolver},
                           wb::Union{Nothing, <:WeightBoundsEstimator, <:WeightBounds},
                           bgt::Union{Nothing, <:Number, <:BudgetConstraintEstimator},
                           sbgt::Union{Nothing, <:Number, <:BudgetRange},
                           lt::Union{Nothing, <:BuyInThresholdEstimator, <:BuyInThreshold},
                           st::Union{Nothing, <:BuyInThresholdEstimator, <:BuyInThreshold},
                           lcs::Union{Nothing, <:LinearConstraintEstimator,
                                      <:LinearConstraint,
                                      <:AbstractVector{<:Union{<:LinearConstraintEstimator,
                                                               <:LinearConstraint}}},
                           cent::Union{Nothing, <:CentralityConstraint,
                                       <:AbstractVector{<:CentralityConstraint},
                                       <:LinearConstraint},
                           gcard::Union{Nothing, <:LinearConstraintEstimator,
                                        <:LinearConstraint},
                           sgcard::Union{Nothing, <:LinearConstraintEstimator,
                                         <:LinearConstraint,
                                         <:AbstractVector{<:Union{<:LinearConstraintEstimator,
                                                                  <:LinearConstraint}}},
                           smtx::Union{Nothing, <:AssetSetsMatrixEstimator, <:NumMat,
                                       <:AbstractVector{<:Union{<:AssetSetsMatrixEstimator,
                                                                <:NumMat}}},
                           sgmtx::Union{Nothing, <:AssetSetsMatrixEstimator, <:NumMat,
                                        <:AbstractVector{<:Union{<:AssetSetsMatrixEstimator,
                                                                 <:NumMat}}},
                           slt::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                      <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                               <:BuyInThresholdEstimator}}},
                           sst::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                      <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                               <:BuyInThresholdEstimator}}},
                           sglt::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                       <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                                <:BuyInThresholdEstimator}}},
                           sgst::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                       <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                                <:BuyInThresholdEstimator}}},
                           sets::Union{Nothing, <:AssetSets},
                           plg::Union{Nothing, <:AbstractPhylogenyConstraintEstimator,
                                      <:AbstractPhylogenyConstraintResult,
                                      <:AbstractVector{<:Union{<:AbstractPhylogenyConstraintEstimator,
                                                               <:AbstractPhylogenyConstraintResult}}},
                           tn::Union{Nothing, <:TurnoverEstimator, <:Turnover,
                                     <:AbstractVector{<:Union{<:TurnoverEstimator,
                                                              <:Turnover}}},
                           te::Union{Nothing, <:AbstractTracking,
                                     <:AbstractVector{<:AbstractTracking}},
                           fees::Union{Nothing, <:FeesEstimator, <:Fees},
                           ret::JuMPReturnsEstimator, sce::Scalariser,
                           ccnt::Union{Nothing, <:CustomJuMPConstraint},
                           cobj::Union{Nothing, <:CustomJuMPObjective}, sc::Number,
                           so::Number, ss::Option{<:Number},
                           card::Union{Nothing, <:Integer},
                           scard::Union{Nothing, <:Integer, <:IntVec},
                           nea::Option{<:Number}, l1::Option{<:Number},
                           l2::Option{<:Number}, strict::Bool)
        if isa(bgt, Number)
            @argcheck(isfinite(bgt))
        elseif isa(bgt, BudgetCostEstimator)
            @argcheck(isnothing(sbgt))
        end
        if isa(sbgt, Number)
            @argcheck(isfinite(sbgt))
            @argcheck(sbgt >= 0)
        end
        if isa(cent, AbstractVector)
            @argcheck(!isempty(cent))
        end
        if !isnothing(card)
            @argcheck(isfinite(card))
            @argcheck(card > 0)
        end
        if isa(scard, Integer)
            @argcheck(isfinite(scard))
            @argcheck(scard > 0)
            @argcheck(isa(smtx, Union{<:AssetSetsMatrixEstimator, <:NumMat}))
            @argcheck(isa(slt, Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator}))
            @argcheck(isa(sst, Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator}))
        elseif isa(scard, IntVec)
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
        elseif isnothing(scard) &&
               (isa(slt, Union{<:BuyInThreshold, <:BuyInThresholdEstimator}) ||
                isa(sst, Union{<:BuyInThreshold, <:BuyInThresholdEstimator}))
            @argcheck(isa(smtx, Union{<:AssetSetsMatrixEstimator, <:NumMat}))
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
        if isa(sgcard, Union{<:LinearConstraintEstimator, <:LinearConstraint})
            @argcheck(isa(sgmtx, Union{<:AssetSetsMatrixEstimator, <:NumMat}))
            @argcheck(isa(sglt,
                          Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator}))
            @argcheck(isa(sgst,
                          Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator}))
            if isa(sgcard, LinearConstraint) && isa(smtx, NumMat)
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
                if isa(sgc, LinearConstraint) && isa(smt, NumMat)
                    N = size(smt, 1)
                    N_ineq = !isnothing(sgc.ineq) ? length(sgc.B_ineq) : 0
                    N_eq = !isnothing(sgc.eq) ? length(sgc.B_eq) : 0
                    @argcheck(N == N_ineq + N_eq)
                end
            end
        elseif isnothing(sgcard) &&
               (isa(sglt, Union{<:BuyInThreshold, <:BuyInThresholdEstimator}) ||
                isa(sgst, Union{<:BuyInThreshold, <:BuyInThresholdEstimator}))
            @argcheck(isa(sgmtx, Union{<:AssetSetsMatrixEstimator, <:NumMat}))
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
           isa(cent, LinearConstraintEstimator) ||
           isa(gcard, LinearConstraintEstimator) ||
           !isa(sgcard,
                Union{Nothing, <:LinearConstraint, <:AbstractVector{<:LinearConstraint}}) ||
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
        if isa(slv, VecSolver)
            @argcheck(!isempty(slv))
        end
        return new{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                   typeof(lt), typeof(st), typeof(lcs), typeof(cent), typeof(gcard),
                   typeof(sgcard), typeof(smtx), typeof(sgmtx), typeof(slt), typeof(sst),
                   typeof(sglt), typeof(sgst), typeof(sets), typeof(plg), typeof(tn),
                   typeof(te), typeof(fees), typeof(ret), typeof(sce), typeof(ccnt),
                   typeof(cobj), typeof(sc), typeof(so), typeof(ss), typeof(card),
                   typeof(scard), typeof(nea), typeof(l1), typeof(l2), typeof(strict)}(pe,
                                                                                       slv,
                                                                                       wb,
                                                                                       bgt,
                                                                                       sbgt,
                                                                                       lt,
                                                                                       st,
                                                                                       lcs,
                                                                                       cent,
                                                                                       gcard,
                                                                                       sgcard,
                                                                                       smtx,
                                                                                       sgmtx,
                                                                                       slt,
                                                                                       sst,
                                                                                       sglt,
                                                                                       sgst,
                                                                                       sets,
                                                                                       plg,
                                                                                       tn,
                                                                                       te,
                                                                                       fees,
                                                                                       ret,
                                                                                       sce,
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
function JuMPOptimiser(;
                       pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                       slv::Union{<:Solver, <:VecSolver},
                       wb::Union{Nothing, <:WeightBoundsEstimator, <:WeightBounds} = WeightBounds(),
                       bgt::Union{Nothing, <:Number, <:BudgetConstraintEstimator} = 1.0,
                       sbgt::Union{Nothing, <:Number, <:BudgetRange} = nothing,
                       lt::Union{Nothing, <:BuyInThresholdEstimator, <:BuyInThreshold} = nothing,
                       st::Union{Nothing, <:BuyInThresholdEstimator, <:BuyInThreshold} = nothing,
                       lcs::Union{Nothing, <:LinearConstraintEstimator, <:LinearConstraint,
                                  <:AbstractVector{<:Union{<:LinearConstraintEstimator,
                                                           <:LinearConstraint}}} = nothing,
                       cent::Union{Nothing, <:CentralityConstraint,
                                   <:AbstractVector{<:CentralityConstraint},
                                   <:LinearConstraint} = nothing,
                       gcard::Union{Nothing, <:LinearConstraintEstimator,
                                    <:LinearConstraint} = nothing,
                       sgcard::Union{Nothing, <:LinearConstraintEstimator,
                                     <:LinearConstraint,
                                     <:AbstractVector{<:Union{<:LinearConstraintEstimator,
                                                              <:LinearConstraint}}} = nothing,
                       smtx::Union{Nothing, <:AssetSetsMatrixEstimator, <:NumMat,
                                   <:AbstractVector{<:Union{<:AssetSetsMatrixEstimator,
                                                            <:NumMat}}} = nothing,
                       sgmtx::Union{Nothing, <:AssetSetsMatrixEstimator, <:NumMat,
                                    <:AbstractVector{<:Union{<:AssetSetsMatrixEstimator,
                                                             <:NumMat}}} = nothing,
                       slt::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                  <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                           <:BuyInThresholdEstimator}}} = nothing,
                       sst::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                  <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                           <:BuyInThresholdEstimator}}} = nothing,
                       sglt::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                   <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                            <:BuyInThresholdEstimator}}} = nothing,
                       sgst::Union{Nothing, <:BuyInThreshold, <:BuyInThresholdEstimator,
                                   <:AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                            <:BuyInThresholdEstimator}}} = nothing,
                       sets::Union{Nothing, <:AssetSets} = nothing,
                       plg::Union{Nothing, <:AbstractPhylogenyConstraintEstimator,
                                  <:AbstractPhylogenyConstraintResult,
                                  <:AbstractVector{<:Union{<:AbstractPhylogenyConstraintEstimator,
                                                           <:AbstractPhylogenyConstraintResult}}} = nothing,
                       tn::Union{Nothing, <:TurnoverEstimator, <:Turnover,
                                 <:AbstractVector{<:Union{<:TurnoverEstimator, <:Turnover}}} = nothing,
                       te::Union{Nothing, <:AbstractTracking,
                                 <:AbstractVector{<:AbstractTracking}} = nothing,
                       fees::Union{Nothing, <:FeesEstimator, <:Fees} = nothing,
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       sce::Scalariser = SumScalariser(),
                       ccnt::Union{Nothing, <:CustomJuMPConstraint} = nothing,
                       cobj::Union{Nothing, <:CustomJuMPObjective} = nothing,
                       sc::Number = 1, so::Number = 1, ss::Option{<:Number} = nothing,
                       card::Union{Nothing, <:Integer} = nothing,
                       scard::Union{Nothing, <:Integer, <:IntVec} = nothing,
                       nea::Option{<:Number} = nothing, l1::Option{<:Number} = nothing,
                       l2::Option{<:Number} = nothing, strict::Bool = false)
    return JuMPOptimiser(pe, slv, wb, bgt, sbgt, lt, st, lcs, cent, gcard, sgcard, smtx,
                         sgmtx, slt, sst, sglt, sgst, sets, plg, tn, te, fees, ret, sce,
                         ccnt, cobj, sc, so, ss, card, scard, nea, l1, l2, strict)
end
function opt_view(opt::JuMPOptimiser, i, X::NumMat)
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
    sets = nothing_asset_sets_view(opt.sets, i)
    tn = turnover_view(opt.tn, i)
    te = tracking_view(opt.te, i, X)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = opt.lcs, cent = opt.cent,
                         gcard = opt.gcard, sgcard = opt.sgcard, smtx = smtx, sgmtx = sgmtx,
                         slt = slt, sst = sst, sglt = sglt, sgst = sgst, sets = sets,
                         plg = opt.plg, tn = tn, te = te, fees = fees, ret = ret,
                         sce = opt.sce, ccnt = ccnt, cobj = cobj, sc = opt.sc, so = opt.so,
                         ss = opt.ss, card = opt.card, scard = opt.scard, nea = opt.nea,
                         l1 = opt.l1, l2 = opt.l2, strict = opt.strict)
end
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1)
    pr = prior(opt.pe, rd; dims = dims)
    datatype = eltype(pr.X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(pr.X, 2), strict = opt.strict,
                                   datatype = datatype)
    lt = threshold_constraints(opt.lt, opt.sets; datatype = datatype, strict = opt.strict)
    st = threshold_constraints(opt.st, opt.sets; datatype = datatype, strict = opt.strict)
    lcs = linear_constraints(opt.lcs, opt.sets; datatype = datatype, strict = opt.strict)
    cent = centrality_constraints(opt.cent, pr.X; iv = rd.iv, ivpa = rd.ivpa)
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
    plg = phylogeny_constraints(opt.plg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    tn = turnover_constraints(opt.tn, opt.sets; datatype = datatype, strict = opt.strict)
    fees = fees_constraints(opt.fees, opt.sets; datatype = datatype, strict = opt.strict)
    ret = jump_returns_factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx,
                                            sgmtx, slt, sst, sglt, sgst, plg, tn, fees, ret)
end
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(propertynames(opt), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getproperty.(opt, pnames))...)
end
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, sgmtx, slt, sst, sglt, sgst, plg, tn, fees, ret) = processed_jump_optimiser_attributes(opt,
                                                                                                                                              rd;
                                                                                                                                              dims = dims)
    return JuMPOptimiser(; pe = pr, slv = opt.slv, wb = wb, bgt = opt.bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = lcs, cent = cent, gcard = gcard,
                         sgcard = sgcard, smtx = smtx, sgmtx = sgmtx, slt = slt, sst = sst,
                         sglt = sglt, sgst = sgst, sets = opt.sets, plg = plg, tn = tn,
                         te = opt.te, fees = fees, ret = ret, sce = opt.sce,
                         ccnt = opt.ccnt, cobj = opt.cobj, sc = opt.sc, so = opt.so,
                         ss = opt.ss, card = opt.card, nea = opt.nea, l1 = opt.l1,
                         l2 = opt.l2, strict = opt.strict)
end

export ProcessedJuMPOptimiserAttributes, JuMPOptimisation, JuMPOptimisationRiskBudgeting,
       JuMPOptimisationFactorRiskContribution, JuMPOptimiser
