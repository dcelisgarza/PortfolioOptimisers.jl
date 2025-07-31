struct ProcessedJuMPOptimiserAttributes{T1 <: AbstractPriorResult,
                                        T2 <: Union{Nothing, <:WeightBoundsResult},
                                        T3 <: Union{Nothing, <:Real, <:AbstractVector},
                                        T4 <: Union{Nothing, <:Real, <:AbstractVector},
                                        T5 <: Union{Nothing, <:LinearConstraintResult},
                                        T6 <: Union{Nothing, <:LinearConstraintResult},
                                        T7 <: Union{Nothing, <:LinearConstraintResult},
                                        T8 <: Union{Nothing, <:LinearConstraintResult,
                                                    <:AbstractVector{<:LinearConstraintResult}},
                                        T9 <: Union{Nothing, <:AbstractMatrix,
                                                    <:AbstractVector{<:AbstractMatrix}},
                                        T10 <: Union{Nothing, <:AbstractMatrix,
                                                     <:AbstractVector{<:AbstractMatrix}},
                                        T11 <: Union{Nothing, <:PhilogenyConstraintResult},
                                        T12 <: Union{Nothing, <:PhilogenyConstraintResult},
                                        T13 <: Union{Nothing, <:Turnover,
                                                     <:AbstractVector{<:Turnover}},
                                        T14 <: Union{Nothing, <:Fees},
                                        T15 <: JuMPReturnsEstimator} <: AbstractResult
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
    nplg::T11
    cplg::T12
    tn::T13
    fees::T14
    ret::T15
end
struct JuMPOptimisationResult{T1 <: Type, T2 <: ProcessedJuMPOptimiserAttributes,
                              T3 <: Union{<:OptimisationReturnCode,
                                          <:AbstractVector{<:OptimisationReturnCode}},
                              T4 <: Union{<:JuMPOptimisationSolution,
                                          <:AbstractVector{<:JuMPOptimisationSolution}},
                              T5 <: Union{Nothing, JuMP.Model}} <: OptimisationResult
    oe::T1
    pa::T2
    retcode::T3
    sol::T4
    model::T5
end
struct JuMPOptimisationFactorRiskContributionResult{T1 <: Type,
                                                    T2 <: ProcessedJuMPOptimiserAttributes,
                                                    T11 <: Union{Nothing,
                                                                 <:SemiDefinitePhilogenyResult},
                                                    T12 <: Union{Nothing,
                                                                 <:SemiDefinitePhilogenyResult},
                                                    T13 <: OptimisationReturnCode,
                                                    T14 <: JuMPOptimisationSolution,
                                                    T15 <: Union{Nothing, JuMP.Model}} <:
       OptimisationResult
    oe::T1
    pa::T2
    frc_nplg::T11
    frc_cplg::T12
    retcode::T13
    sol::T14
    model::T15
end
function Base.getproperty(r::Union{<:JuMPOptimisationResult,
                                   <:JuMPOptimisationFactorRiskContributionResult},
                          sym::Symbol)
    return if sym == :pr
        r.pa.pr
    elseif sym == :wb
        r.pa.wb
    elseif sym == :lt
        r.pa.lt
    elseif sym == :st
        r.pa.st
    elseif sym == :lcs
        r.pa.lcs
    elseif sym == :cent
        r.pa.cent
    elseif sym == :gcard
        r.pa.gcard
    elseif sym == :sgcard
        r.pa.sgcard
    elseif sym == :smtx
        r.pa.smtx
    elseif sym == :sgmtx
        r.pa.sgmtx
    elseif sym == :nplg
        r.pa.nplg
    elseif sym == :cplg
        r.pa.cplg
    elseif sym == :tn
        r.pa.tn
    elseif sym == :fees
        r.pa.fees
    elseif sym == :ret
        r.pa.ret
    elseif sym == :w
        !isa(r.sol, AbstractVector) ? r.sol.w : getproperty.(r.sol, :w)
    else
        getfield(r, sym)
    end
end
struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                     T2 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                     T3 <: Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                     T4 <: Union{Nothing, <:Real, <:BudgetRange, <:BudgetCosts},
                     T5 <: Union{Nothing, <:Real, <:BudgetRange},
                     T6 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:AbstractDict,
                                 <:AbstractVector{<:Pair{<:Any, <:Real}}},
                     T7 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:AbstractDict,
                                 <:AbstractVector{<:Pair{<:Any, <:Real}}},
                     T8 <: Union{Nothing, <:AbstractString, Expr,
                                 <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                 <:AbstractVector{<:Union{<:AbstractString, Expr}},
                       #! Start: to delete
                                 <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                       #! End: to delete
                                 <:LinearConstraintResult},
                     T9 <: Union{Nothing, <:LinearConstraintResult},
                     T10 <: Union{Nothing, <:CentralityConstraintEstimator,
                                  <:AbstractVector{<:CentralityConstraintEstimator},
                                  <:LinearConstraintResult},
                     T11 <: Union{Nothing, <:AbstractString, Expr,
                                  <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                  <:AbstractVector{<:Union{<:AbstractString, Expr}},
                       #! Start: to delete
                                  <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                       #! End: to delete
                                  <:LinearConstraintResult},
                     T12 <: Union{Nothing, <:AbstractString, Expr,
                                  <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                  <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                  <:AbstractVector{<:AbstractVector{<:AbstractString}},
                                  <:AbstractVector{<:AbstractVector{Expr}},
                                  <:AbstractVector{<:AbstractVector{<:Union{<:AbstractString,
                                                                            Expr}}},
                       #! Start: to delete
                                  <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                       #! End: to delete
                                  <:LinearConstraintResult,
                                  <:AbstractVector{<:LinearConstraintResult}},
                     T13 <: Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                  <:AbstractVector{Symbol}, <:AbstractVector{<:AbstractString},
                                  <:AbstractVector{<:AbstractMatrix}},
                     T14 <: Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                  <:AbstractVector{Symbol}, <:AbstractVector{<:AbstractString},
                                  <:AbstractVector{<:AbstractMatrix}},
                     T15 <: Union{Nothing, <:AssetSets,
                       #! Start: to delete
                                  <:DataFrame
                       #! End: to delete
                       },
                     T16 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T17 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T18 <: Union{Nothing, <:TurnoverEstimator,
                                  <:AbstractVector{<:TurnoverEstimator}, <:Turnover,
                                  <:AbstractVector{<:Turnover},
                                  <:AbstractVector{<:Union{<:TurnoverEstimator, <:Turnover}}},
                     T19 <: Union{Nothing, <:AbstractTracking,
                                  <:AbstractVector{<:AbstractTracking}},
                     T20 <: Union{Nothing, <:FeesEstimator, <:Fees},
                     T21 <: JuMPReturnsEstimator, T22 <: Scalariser,
                     T23 <: Union{Nothing, <:CustomConstraint},
                     T24 <: Union{Nothing, <:CustomObjective}, T25 <: Real, T26 <: Real,
                     T27 <: Union{Nothing, <:Integer},
                     T28 <: Union{Nothing, <:Integer, <:AbstractVector{<:Integer}},
                     T29 <: Union{Nothing, <:Real}, T30 <: Union{Nothing, <:Real},
                     T31 <: Union{Nothing, <:Real}, T32 <: Union{Nothing, <:Real},
                     T33 <: Bool} <: BaseJuMPOptimisationEstimator
    pe::T1 # PriorEstimator
    slv::T2
    wb::T3 # WeightBoundsResult
    bgt::T4 # BudgetRange
    sbgt::T5 # LongShortSum
    lt::T6 # l threshold
    st::T7
    lcs::T8
    lcm::T9
    cent::T10
    gcard::T11
    sgcard::T12
    smtx::T13
    sgmtx::T14
    sets::T15
    nplg::T16
    cplg::T17
    tn::T18 # Turnover
    te::T19 # TrackingError
    fees::T20
    ret::T21
    sce::T22
    ccnt::T23
    cobj::T24
    sc::T25
    so::T26
    card::T27
    scard::T28
    nea::T29
    l1::T30
    l2::T31
    ss::T32
    strict::T33
end
function assert_finite_nonnegative_real_or_vec(val::Real)
    @smart_assert(isfinite(val) && val > zero(val))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::AbstractVector{<:Real})
    @smart_assert(any(isfinite, val) &&
                  any(x -> x > zero(x), val) &&
                  all(x -> x >= zero(x), val))
    return nothing
end
function JuMPOptimiser(;
                       pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                       slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                       wb::Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint} = WeightBoundsResult(),
                       bgt::Union{Nothing, <:Real, <:BudgetConstraintEstimator} = 1.0,
                       sbgt::Union{Nothing, <:Real, <:BudgetRange} = nothing,
                       lt::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:AbstractDict,
                                 <:AbstractVector{<:Pair{<:Any, <:Real}}} = nothing,
                       st::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:AbstractDict,
                                 <:AbstractVector{<:Pair{<:Any, <:Real}}} = nothing,
                       lcs::Union{Nothing, <:AbstractString, Expr,
                                  <:AbstractVector{<:AbstractString},
                                  <:AbstractVector{Expr},
                                  <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                  #! Start: to delete
                                  <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                                  #! End: to delete
                                  <:LinearConstraintResult} = nothing,
                       lcm::Union{Nothing, <:LinearConstraintResult} = nothing,
                       cent::Union{Nothing, <:CentralityConstraintEstimator,
                                   <:AbstractVector{<:CentralityConstraintEstimator},
                                   <:LinearConstraintResult} = nothing,
                       gcard::Union{Nothing, <:AbstractString, Expr,
                                    <:AbstractVector{<:AbstractString},
                                    <:AbstractVector{Expr},
                                    <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                    #! Start: to delete
                                    <:LinearConstraint,
                                    <:AbstractVector{<:LinearConstraint},
                                    #! End: to delete
                                    <:LinearConstraintResult} = nothing,
                       sgcard::Union{Nothing, <:AbstractString, Expr,
                                     <:AbstractVector{<:AbstractString},
                                     <:AbstractVector{Expr},
                                     <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                     <:AbstractVector{<:AbstractVector{<:AbstractString}},
                                     <:AbstractVector{<:AbstractVector{Expr}},
                                     <:AbstractVector{<:AbstractVector{<:Union{<:AbstractString,
                                                                               Expr}}},
                                     #! Start: to delete
                                     <:LinearConstraint,
                                     <:AbstractVector{<:LinearConstraint},
                                     #! End: to delete
                                     <:LinearConstraintResult,
                                     <:AbstractVector{<:LinearConstraintResult}} = nothing,
                       smtx::Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                   <:AbstractVector{Symbol},
                                   <:AbstractVector{<:AbstractString},
                                   <:AbstractVector{<:AbstractMatrix}} = nothing,
                       sgmtx::Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix,
                                    <:AbstractVector{Symbol},
                                    <:AbstractVector{<:AbstractString},
                                    <:AbstractVector{<:AbstractMatrix}} = nothing,
                       sets::Union{Nothing, <:AssetSets,
                                   #! Start: to delete
                                   <:DataFrame
                                   #! End: to delete
                                   } = nothing,
                       nplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       cplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       tn::Union{Nothing, <:TurnoverEstimator,
                                 <:AbstractVector{<:TurnoverEstimator}, <:Turnover,
                                 <:AbstractVector{<:Turnover},
                                 <:AbstractVector{<:Union{<:TurnoverEstimator, <:Turnover}}} = nothing,
                       te::Union{Nothing, <:AbstractTracking,
                                 <:AbstractVector{<:AbstractTracking}} = nothing,
                       fees::Union{Nothing, <:FeesEstimator, <:Fees} = nothing,
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       sce::Scalariser = SumScalariser(),
                       ccnt::Union{Nothing, <:CustomConstraint} = nothing,
                       cobj::Union{Nothing, <:CustomObjective} = nothing, sc::Real = 1,
                       so::Real = 1, card::Union{Nothing, <:Integer} = nothing,
                       scard::Union{Nothing, <:Integer, <:AbstractVector{<:Integer}} = nothing,
                       nea::Union{Nothing, <:Real} = nothing,
                       l1::Union{Nothing, <:Real} = nothing,
                       l2::Union{Nothing, <:Real} = nothing,
                       ss::Union{Nothing, <:Real} = nothing, strict::Bool = false)
    if isa(bgt, Real)
        @smart_assert(isfinite(bgt))
    elseif isa(bgt, BudgetCostEstimator)
        @smart_assert(isnothing(sbgt))
    end
    if isa(sbgt, Real)
        @smart_assert(isfinite(sbgt) && sbgt >= 0)
    elseif isa(sbgt, BudgetRange)
        lb = sbgt.lb
        ub = sbgt.ub
        lb_flag = isnothing(lb)
        ub_flag = isnothing(ub)
        @smart_assert(lb_flag ⊼ ub_flag)
        if !lb_flag
            @smart_assert(lb >= zero(lb))
        end
        if !ub_flag
            @smart_assert(ub >= zero(ub))
        end
        if !lb_flag && !ub_flag
            @smart_assert(lb <= ub)
        end
    end
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    if isa(cent, AbstractVector)
        @smart_assert(!isempty(cent))
    end
    if !isnothing(card)
        @smart_assert(isfinite(card) && card > 0)
    end
    if !isnothing(scard)
        if isa(scard, Integer)
            @smart_assert(isfinite(scard) && scard > 0)
            @smart_assert(isa(smtx, Union{Symbol, <:AbstractString, <:AbstractMatrix}))
        elseif isa(scard, AbstractVector)
            @smart_assert(!isempty(scard))
            @smart_assert(all(isfinite, scard) && all(x -> x > 0, scard))
            @smart_assert(isa(smtx,
                              Union{<:AbstractVector{Symbol},
                                    <:AbstractVector{<:AbstractString},
                                    <:AbstractVector{<:AbstractMatrix}}))
            @smart_assert(length(scard) == length(smtx))
        end
    end
    if isa(gcard, AbstractVector)
        @smart_assert(!isempty(gcard))
    end
    if isa(sgcard, AbstractVector)
        @smart_assert(!isempty(sgcard))
    end
    if isa(sgcard,
           Union{<:AbstractString, Expr, <:AbstractVector{<:AbstractString},
                 <:AbstractVector{Expr}, <:AbstractVector{<:Union{<:AbstractString, Expr}},
                 <:LinearConstraintResult})
        @smart_assert(isa(sgmtx, Union{Symbol, <:AbstractString, <:AbstractMatrix}))
    elseif isa(sgcard, AbstractVector{<:AbstractVector})
        @smart_assert(isa(sgmtx,
                          Union{<:AbstractVector{Symbol},
                                <:AbstractVector{<:AbstractString},
                                <:AbstractVector{<:AbstractMatrix}}))
        @smart_assert(length(scard) == length(sgmtx))
    end
    if isa(wb, WeightBoundsConstraint) ||
       !isa(lt, Union{Nothing, <:Real, <:AbstractVector{<:Real}}) ||
       !isa(st, Union{Nothing, <:Real, <:AbstractVector{<:Real}}) ||
       !isa(lcs, Union{Nothing, <:LinearConstraintResult}) ||
       !isa(cent, Union{Nothing, <:LinearConstraintResult}) ||
       !isa(gcard, Union{Nothing, <:LinearConstraintResult}) ||
       !isa(sgcard,
            Union{Nothing, <:LinearConstraintResult,
                  <:AbstractVector{<:LinearConstraintResult}}) ||
       !isnothing(scard)
        @smart_assert(!isnothing(sets))
    end
    if isa(sgcard, LinearConstraintResult) && isa(smtx, AbstractMatrix)
        N = size(smtx, 1)
        N_ineq = !isnothing(sgcard.ineq) ? length(sgcard.B_ineq) : 0
        N_eq = !isnothing(sgcard.eq) ? length(sgcard.B_eq) : 0
        @smart_assert(N == N_ineq + N_eq)
    end
    if isa(lt, Real) || isa(lt, AbstractVector{<:Real})
        assert_finite_nonnegative_real_or_vec(lt)
    end
    if isa(st, Real) || isa(st, AbstractVector{<:Real})
        assert_finite_nonnegative_real_or_vec(st)
    end
    if isa(tn, AbstractVector)
        @smart_assert(!isempty(tn))
    end
    if isa(te, AbstractVector)
        @smart_assert(!isempty(te))
    end
    if !isnothing(nea)
        @smart_assert(nea > zero(nea))
    end
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return JuMPOptimiser{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                         typeof(lt), typeof(st), typeof(lcs), typeof(lcm), typeof(cent),
                         typeof(gcard), typeof(sgcard), typeof(smtx), typeof(sgmtx),
                         typeof(sets), typeof(nplg), typeof(cplg), typeof(tn), typeof(te),
                         typeof(fees), typeof(ret), typeof(sce), typeof(ccnt), typeof(cobj),
                         typeof(sc), typeof(so), typeof(card), typeof(scard), typeof(nea),
                         typeof(l1), typeof(l2), typeof(ss), typeof(strict)}(pe, slv, wb,
                                                                             bgt, sbgt, lt,
                                                                             st, lcs, lcm,
                                                                             cent, gcard,
                                                                             sgcard, smtx,
                                                                             sgmtx, sets,
                                                                             nplg, cplg, tn,
                                                                             te, fees, ret,
                                                                             sce, ccnt,
                                                                             cobj, sc, so,
                                                                             card, scard,
                                                                             nea, l1, l2,
                                                                             ss, strict)
end
function opt_view(opt::JuMPOptimiser, i::AbstractVector, X::AbstractMatrix)
    X = isa(opt.pe, AbstractPriorResult) ? opt.pe.X : X
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    bgt = budget_view(opt.bgt, i)
    lt = threshold_view(opt.lt, i)
    st = threshold_view(opt.st, i)
    # lcs = linear_constraint_view(opt.lcs, i)
    # gcard = linear_constraint_view(opt.gcard, i)
    # sgcard = linear_constraint_view(opt.sgcard, i)
    smtx = asset_sets_matrix_view(opt.smtx, i)
    sgmtx = if opt.smtx === opt.sgmtx
        smtx
    else
        asset_sets_matrix_view(opt.sgmtx, i)
    end
    sets = nothing_asset_sets_view(opt.sets, i)
    tn = turnover_view(opt.tn, i)
    te = tracking_view(opt.te, i, X)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = opt.lcs, lcm = opt.lcm, cent = opt.cent,
                         gcard = opt.gcard, sgcard = opt.sgcard, smtx = smtx, sgmtx = sgmtx,
                         sets = sets, nplg = opt.nplg, cplg = opt.cplg, tn = tn, te = te,
                         fees = fees, ret = ret, sce = opt.sce, ccnt = ccnt, cobj = cobj,
                         sc = opt.sc, so = opt.so, card = opt.card, scard = opt.scard,
                         nea = opt.nea, l1 = opt.l1, l2 = opt.l2, ss = opt.ss,
                         strict = opt.strict)
end
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1)
    pr = prior(opt.pe, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    datatype = eltype(pr.X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(pr.X, 2), strict = opt.strict,
                                   datatype = datatype)
    lt = threshold_constraints(opt.lt, opt.sets; datatype = datatype, strict = opt.strict)
    st = threshold_constraints(opt.st, opt.sets; datatype = datatype, strict = opt.strict)
    lcs = linear_constraints(opt.lcs, opt.sets; datatype = datatype, strict = opt.strict)
    cent = centrality_constraints(opt.cent, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    gcard = linear_constraints(opt.gcard, opt.sets; datatype = Int, strict = opt.strict)
    sgcard = linear_constraints(opt.sgcard, opt.sets; datatype = Int, strict = opt.strict)
    smtx = asset_sets_matrix(opt.smtx, opt.sets)
    sgmtx = if opt.smtx === opt.sgmtx
        smtx
    else
        asset_sets_matrix(opt.sgmtx, opt.sets)
    end
    nplg = philogeny_constraints(opt.nplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    cplg = philogeny_constraints(opt.cplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    tn = turnover_constraints(opt.tn, opt.sets; strict = opt.strict, datatype = datatype)
    fees = fees_constraints(opt.fees, opt.sets; strict = opt.strict, datatype = datatype)
    ret = jump_returns_factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx,
                                            sgmtx, nplg, cplg, tn, fees, ret)
end
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(propertynames(opt), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getproperty.(Ref(opt), pnames))...)
end
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, sgmtx, nplg, cplg, tn, fees, ret) = processed_jump_optimiser_attributes(opt,
                                                                                                                               rd;
                                                                                                                               dims = dims)
    return JuMPOptimiser(; pe = pr, slv = opt.slv, wb = wb, bgt = opt.bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = lcs, lcm = opt.lcm, cent = cent,
                         gcard = gcard, sgcard = sgcard, smtx = smtx, sgmtx = sgmtx,
                         sets = opt.sets, nplg = nplg, cplg = cplg, tn = tn, te = opt.te,
                         fees = fees, ret = ret, sce = opt.sce, ccnt = opt.ccnt,
                         cobj = opt.cobj, sc = opt.sc, so = opt.so, card = opt.card,
                         nea = opt.nea, l1 = opt.l1, l2 = opt.l2, ss = opt.ss,
                         strict = opt.strict)
end

export JuMPOptimisationResult, JuMPOptimiser
