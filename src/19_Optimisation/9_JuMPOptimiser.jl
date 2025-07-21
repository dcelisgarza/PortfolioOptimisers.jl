struct JuMPOptimisationResult{T1 <: Type, T2 <: AbstractPriorResult,
                              T3 <: Union{Nothing, <:WeightBoundsResult},
                              T4 <: Union{Nothing, <:LinearConstraintResult},
                              T5 <: Union{Nothing, <:LinearConstraintResult},
                              T6 <: Union{Nothing, <:LinearConstraintResult},
                              T7 <: Union{Nothing, <:LinearConstraintResult},
                              T8 <:
                              Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix},
                              T9 <: Union{Nothing, <:PhilogenyConstraintResult},
                              T10 <: Union{Nothing, <:PhilogenyConstraintResult},
                              T11 <: JuMPReturnsEstimator,
                              T12 <: Union{<:OptimisationReturnCode,
                                           <:AbstractVector{<:OptimisationReturnCode}},
                              T13 <: Union{<:JuMPOptimisationSolution,
                                           <:AbstractVector{<:JuMPOptimisationSolution}},
                              T14 <: Union{Nothing, JuMP.Model}} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    lcs::T4
    cent::T5
    gcard::T6
    sgcard::T7
    smtx::T8
    nplg::T9
    cplg::T10
    ret::T11
    retcode::T12
    sol::T13
    model::T14
end
function Base.getproperty(r::JuMPOptimisationResult, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? r.sol.w : getproperty.(r.sol, :w)
    else
        getfield(r, sym)
    end
end
struct JuMPOptimisationFactorRiskContributionResult{T1 <: Type, T2 <: AbstractPriorResult,
                                                    T3 <:
                                                    Union{Nothing, <:WeightBoundsResult},
                                                    T4 <: Union{Nothing,
                                                                <:LinearConstraintResult},
                                                    T5 <: Union{Nothing,
                                                                <:LinearConstraintResult},
                                                    T6 <: Union{Nothing,
                                                                <:LinearConstraintResult},
                                                    T7 <: Union{Nothing,
                                                                <:LinearConstraintResult},
                                                    T8 <:
                                                    Union{Nothing, Symbol, <:AbstractString,
                                                          <:AbstractMatrix},
                                                    T9 <: Union{Nothing,
                                                                <:IntegerPhilogenyResult},
                                                    T10 <: Union{Nothing,
                                                                 <:IntegerPhilogenyResult},
                                                    T11 <: Union{Nothing,
                                                                 <:SemiDefinitePhilogenyResult},
                                                    T12 <: Union{Nothing,
                                                                 <:SemiDefinitePhilogenyResult},
                                                    T13 <: OptimisationReturnCode,
                                                    T14 <: JuMPOptimisationSolution,
                                                    T15 <: Union{Nothing, JuMP.Model}} <:
       OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    lcs::T4
    cent::T5
    gcard::T6
    sgcard::T7
    smtx::T8
    nplg::T9
    cplg::T10
    frc_nplg::T11
    frc_cplg::T12
    retcode::T13
    sol::T14
    model::T15
end
function Base.getproperty(r::JuMPOptimisationFactorRiskContributionResult, sym::Symbol)
    return if sym == :w
        r.sol.w
    else
        getfield(r, sym)
    end
end
struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                     T2 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                     T3 <: Union{Nothing, <:WeightBoundsResult, <:AbstractString, Expr,
                                 <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                 <:AbstractVector{<:Union{<:AbstractString, Expr}},
                       #! Start: to delete
                                 <:WeightBoundsConstraint
                       #! End: to delete
                       }, T4 <: Union{Nothing, <:Real, <:BudgetRange, <:BudgetCosts},
                     T5 <: Union{Nothing, <:Real, <:BudgetRange},
                     T6 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                     T7 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
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
                       #! Start: to delete
                                  <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                       #! End: to delete
                                  <:LinearConstraintResult},
                     T13 <: Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix},
                     T14 <: Union{Nothing, <:AssetSets,
                       #! Start: to delete
                                  <:DataFrame
                       #! End: to delete
                       },
                     T15 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T16 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T17 <: Union{Nothing, <:Turnover},
                     T18 <:
                     Union{Nothing, <:AbstractTracking, <:AbstractVector{AbstractTracking}},
                     T21 <: Union{Nothing, <:Fees}, T22 <: JuMPReturnsEstimator,
                     T23 <: Scalariser, T24 <: Union{Nothing, <:CustomConstraint},
                     T25 <: Union{Nothing, <:CustomObjective}, T26 <: Real, T27 <: Real,
                     T28 <: Union{Nothing, <:Integer}, T29 <: Union{Nothing, <:Integer},
                     T30 <: Union{Nothing, <:Real}, T31 <: Union{Nothing, <:Real},
                     T32 <: Union{Nothing, <:Real}, T33 <: Union{Nothing, <:Real},
                     T34 <: Bool} <: BaseJuMPOptimisationEstimator
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
    sets::T14
    nplg::T15
    cplg::T16
    tn::T17 # Turnover
    te::T18 # TrackingError
    fees::T21
    ret::T22
    sce::T23
    ccnt::T24
    cobj::T25
    sc::T26
    so::T27
    card::T28
    scard::T29
    nea::T30
    l1::T31
    l2::T32
    ss::T33
    strict::T34
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
                       wb::Union{Nothing, <:WeightBoundsResult, <:AbstractString, Expr,
                                 <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                                 <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                 #! Start: to delete
                                 <:WeightBoundsConstraint
                                 #! End: to delete
                                 } = WeightBoundsResult(),
                       bgt::Union{Nothing, <:Real, <:BudgetConstraintEstimator} = 1.0,
                       sbgt::Union{Nothing, <:Real, <:BudgetRange} = nothing,
                       lt::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                       st::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
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
                                     #! Start: to delete
                                     <:LinearConstraint,
                                     <:AbstractVector{<:LinearConstraint},
                                     #! End: to delete
                                     <:LinearConstraintResult} = nothing,
                       smtx::Union{Nothing, Symbol, <:AbstractString, <:AbstractMatrix} = nothing,
                       sets::Union{Nothing, <:AssetSets,
                                   #! Start: to delete
                                   <:DataFrame
                                   #! End: to delete
                                   } = nothing,
                       nplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       cplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       tn::Union{Nothing, <:Turnover, <:AbstractVector{<:Turnover}} = nothing,
                       te::Union{Nothing, <:AbstractTracking,
                                 <:AbstractVector{<:AbstractTracking}} = nothing,
                       fees::Union{Nothing, <:Fees} = nothing,
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       sce::Scalariser = SumScalariser(),
                       ccnt::Union{Nothing, <:CustomConstraint} = nothing,
                       cobj::Union{Nothing, <:CustomObjective} = nothing, sc::Real = 1,
                       so::Real = 1, card::Union{Nothing, <:Integer} = nothing,
                       scard::Union{Nothing, <:Integer} = nothing,
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
    if !isnothing(scard) || !isnothing(sgcard)
        @smart_assert(!isnothing(smtx))
        if !isnothing(scard)
            @smart_assert(isfinite(scard) && scard > 0)
        end
    end
    if isa(gcard, AbstractVector)
        @smart_assert(!isempty(gcard))
    end
    if isa(sgcard, AbstractVector)
        @smart_assert(!isempty(sgcard))
    end
    if isa(wb, WeightBoundsConstraint) ||
       isa(lcs,
           Union{<:AbstractString, Expr, <:AbstractVector{<:AbstractString},
                 <:AbstractVector{Expr}, <:AbstractVector{<:Union{<:AbstractString, Expr}},
                 #! Start: to delete
                 <:LinearConstraint, <:AbstractVector{<:LinearConstraint}
                 #! End: to delete
                 }) ||
       isa(cent,
           Union{<:AbstractString, Expr, <:AbstractVector{<:AbstractString},
                 <:AbstractVector{Expr}, <:AbstractVector{<:Union{<:AbstractString, Expr}},
                 #! Start: to delete
                 <:CentralityConstraintEstimator,
                 <:AbstractVector{<:CentralityConstraintEstimator}
                 #! End: to delete
                 }) ||
       isa(gcard,
           Union{<:AbstractString, Expr, <:AbstractVector{<:AbstractString},
                 <:AbstractVector{Expr}, <:AbstractVector{<:Union{<:AbstractString, Expr}},
                 #! Start: to delete
                 <:LinearConstraint, <:AbstractVector{<:LinearConstraint}
                 #! End: to delete

                 }) ||
       isa(sgcard,
           Union{<:AbstractString, Expr, <:AbstractVector{<:AbstractString},
                 <:AbstractVector{Expr}, <:AbstractVector{<:Union{<:AbstractString, Expr}},
                 #! Start: to delete
                 <:LinearConstraint, <:AbstractVector{<:LinearConstraint}
                 #! End: to delete

                 })
        @smart_assert(!isnothing(sets))
    end
    if isa(sgcard, LinearConstraintResult) && isa(smtx, AbstractMatrix)
        N = size(smtx, 1)
        N_ineq = !isnothing(sgcard.ineq) ? length(sgcard.B_ineq) : 0
        N_eq = !isnothing(sgcard.eq) ? length(sgcard.B_eq) : 0
        @smart_assert(N == N_ineq + N_eq)
    end
    if !isnothing(lt)
        assert_finite_nonnegative_real_or_vec(lt)
    end
    if !isnothing(st)
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
                         typeof(gcard), typeof(sgcard), typeof(smtx), typeof(sets),
                         typeof(nplg), typeof(cplg), typeof(tn), typeof(te), typeof(fees),
                         typeof(ret), typeof(sce), typeof(ccnt), typeof(cobj), typeof(sc),
                         typeof(so), typeof(card), typeof(scard), typeof(nea), typeof(l1),
                         typeof(l2), typeof(ss), typeof(strict)}(pe, slv, wb, bgt, sbgt, lt,
                                                                 st, lcs, lcm, cent, gcard,
                                                                 sgcard, smtx, sets, nplg,
                                                                 cplg, tn, te, fees, ret,
                                                                 sce, ccnt, cobj, sc, so,
                                                                 card, scard, nea, l1, l2,
                                                                 ss, strict)
end
function opt_view(opt::JuMPOptimiser, i::AbstractVector, X::AbstractMatrix)
    X = isa(opt.pe, AbstractPriorResult) ? opt.pe.X : X
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    bgt = budget_view(opt.bgt, i)
    lt = nothing_scalar_array_view(opt.lt, i)
    st = nothing_scalar_array_view(opt.lt, i)
    # lcs = linear_constraint_view(opt.lcs, i)
    # gcard = linear_constraint_view(opt.gcard, i)
    # sgcard = linear_constraint_view(opt.sgcard, i)
    smtx = asset_sets_matrix_view(opt.smtx, i)
    sets = nothing_asset_sets_view(opt.sets, i)
    tn = turnover_view(opt.tn, i)
    te = tracking_view(opt.te, i, X)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = opt.lcs, lcm = opt.lcm, cent = opt.cent,
                         gcard = opt.gcard, sgcard = opt.sgcard, smtx = smtx, sets = sets,
                         nplg = opt.nplg, cplg = opt.cplg, tn = tn, te = te, fees = fees,
                         ret = ret, sce = opt.sce, ccnt = ccnt, cobj = cobj, sc = opt.sc,
                         so = opt.so, card = opt.card, scard = opt.scard, nea = opt.nea,
                         l1 = opt.l1, l2 = opt.l2, ss = opt.ss, strict = opt.strict)
end
struct ProcessedJuMPOptimiserAttributes{T1 <: AbstractPriorResult,
                                        T2 <: Union{Nothing, <:WeightBoundsResult},
                                        T3 <: Union{Nothing, <:LinearConstraintResult},
                                        T4 <: Union{Nothing, <:LinearConstraintResult},
                                        T5 <: Union{Nothing, <:LinearConstraintResult},
                                        T6 <: Union{Nothing, <:LinearConstraintResult},
                                        T7 <: Union{Nothing, Symbol, <:AbstractString,
                                                    <:AbstractMatrix},
                                        T8 <: Union{Nothing, <:PhilogenyConstraintResult},
                                        T9 <: Union{Nothing, <:PhilogenyConstraintResult},
                                        T10 <: JuMPReturnsEstimator} <: AbstractResult
    pr::T1
    wb::T2
    lcs::T3
    cent::T4
    gcard::T5
    sgcard::T6
    smtx::T7
    nplg::T8
    cplg::T9
    ret::T10
end
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1)
    pr = prior(opt.pe, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    datatype = eltype(pr.X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(pr.X, 2), strict = opt.strict)
    lcs = linear_constraints(opt.lcs, opt.sets; datatype = datatype, strict = opt.strict)
    cent = centrality_constraints(opt.cent, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    gcard = linear_constraints(opt.gcard, opt.sets; datatype = Int, strict = opt.strict)
    sgcard = linear_constraints(opt.sgcard, opt.sets; datatype = datatype,
                                strict = opt.strict)
    smtx = asset_sets_matrix(opt.smtx, opt.sets)
    nplg = philogeny_constraints(opt.nplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    cplg = philogeny_constraints(opt.cplg, pr.X; iv = rd.iv, ivpa = rd.ivpa)
    ret = jump_returns_factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(pr, wb, lcs, cent, gcard, sgcard, smtx, nplg,
                                            cplg, ret)
end
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = propertynames(opt)
    idx = findfirst(x -> x == :ret, pnames)
    return JuMPOptimiser((getproperty(opt, pnames[i]) for i in 1:(idx - 1))...,
                         no_bounds_returns_estimator(opt.ret, args...),
                         (getproperty(opt, pnames[i]) for i in (idx + 1):length(pnames))...)
end
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1)
    (; pr, wb, lcs, cent, gcard, sgcard, smtx, nplg, cplg, ret) = processed_jump_optimiser_attributes(opt,
                                                                                                      rd;
                                                                                                      dims = dims)
    return JuMPOptimiser(; pe = pr, slv = opt.slv, wb = wb, bgt = opt.bgt, sbgt = opt.sbgt,
                         lt = opt.lt, st = opt.st, lcs = lcs, lcm = opt.lcm, cent = cent,
                         gcard = gcard, sgcard = sgcard, smtx = smtx, sets = opt.sets,
                         nplg = nplg, cplg = cplg, tn = opt.tn, te = opt.te,
                         fees = opt.fees, ret = ret, sce = opt.sce, ccnt = opt.ccnt,
                         cobj = opt.cobj, sc = opt.sc, so = opt.so, card = opt.card,
                         nea = opt.nea, l1 = opt.l1, l2 = opt.l2, ss = opt.ss,
                         strict = opt.strict)
end

export JuMPOptimisationResult, JuMPOptimiser
