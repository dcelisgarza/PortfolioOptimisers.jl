struct JuMPOptimisationResult{T1 <: Type, T2 <: AbstractPriorResult,
                              T3 <: Union{Nothing, WeightBoundsResult},
                              T4 <: Union{Nothing, <:LinearConstraintResult},
                              T5 <: Union{Nothing, <:LinearConstraintResult},
                              T6 <: Union{Nothing, <:LinearConstraintResult},
                              T7 <: Union{Nothing, <:PhilogenyConstraintResult},
                              T8 <: Union{Nothing, <:PhilogenyConstraintResult},
                              T9 <: OptimisationReturnCode, T10 <: JuMPOptimisationSolution,
                              T11 <: Union{Nothing, JuMP.Model}} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    lcs::T4
    cent::T5
    gcard::T6
    nplg::T7
    cplg::T8
    retcode::T9
    sol::T10
    model::T11
end
function Base.getproperty(r::JuMPOptimisationResult, sym::Symbol)
    return if sym == :w
        r.sol.w
    else
        getfield(r, sym)
    end
end
struct JuMPOptimisationFactorRiskContributionResult{T1 <: Type, T2 <: AbstractPriorResult,
                                                    T3 <:
                                                    Union{Nothing, WeightBoundsResult},
                                                    T4 <: Union{Nothing,
                                                                <:LinearConstraintResult},
                                                    T5 <: Union{Nothing,
                                                                <:LinearConstraintResult},
                                                    T6 <: Union{Nothing,
                                                                <:LinearConstraintResult},
                                                    T7 <: Union{Nothing,
                                                                <:IntegerPhilogenyResult},
                                                    T8 <: Union{Nothing,
                                                                <:IntegerPhilogenyResult},
                                                    T9 <: Union{Nothing,
                                                                <:SemiDefinitePhilogenyResult},
                                                    T10 <: Union{Nothing,
                                                                 <:SemiDefinitePhilogenyResult},
                                                    T11 <: OptimisationReturnCode,
                                                    T12 <: JuMPOptimisationSolution,
                                                    T13 <: Union{Nothing, JuMP.Model}} <:
       OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    lcs::T4
    cent::T5
    gcard::T6
    nplg::T7
    cplg::T8
    frc_nplg::T9
    frc_cplg::T10
    retcode::T11
    sol::T12
    model::T13
end
function Base.getproperty(r::JuMPOptimisationFactorRiskContributionResult, sym::Symbol)
    return if sym == :w
        r.sol.w
    else
        getfield(r, sym)
    end
end
struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                     T2 <: Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                     T3 <: Union{Nothing, <:Real, <:BudgetRange},
                     T4 <: Union{Nothing, <:Real, <:BudgetRange},
                     T5 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintResult},
                     T6 <: Union{Nothing, <:LinearConstraintResult},
                     T7 <: Union{Nothing, <:CentralityConstraintEstimator,
                                 <:AbstractVector{<:CentralityConstraintEstimator},
                                 <:LinearConstraintResult}, T8 <: Union{Nothing, <:Integer},
                     T9 <: Union{Nothing, <:CardinalityConstraint,
                                 <:AbstractVector{<:CardinalityConstraint},
                                 <:LinearConstraintResult},
                     T10 <: Union{Nothing, <:DataFrame},
                     T11 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T12 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T13 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                     T14 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                     T15 <: Union{Nothing, <:Turnover},
                     T16 <: Union{Nothing, <:TrackingError}, T17 <: Union{Nothing, <:Real},
                     T18 <: Union{Nothing, <:Real}, T19 <: Union{Nothing, <:Real},
                     T20 <: Union{Nothing, <:Fees}, T21 <: Scalariser,
                     T22 <: JuMPReturnsEstimator, T23 <: Union{Nothing, <:CustomConstraint},
                     T24 <: Union{Nothing, <:CustomObjective}, T25 <: Real, T26 <: Real,
                     T27 <: Union{Nothing, <:Real},
                     T28 <: Union{<:Solver, <:AbstractVector{<:Solver}}, T29 <: Bool} <:
       JuMPOptimisationEstimator
    pe::T1 # PriorEstimator
    wb::T2 # WeightBoundsResult
    bgt::T3 # BudgetRange
    sbgt::T4 # LongShortSum
    lcs::T5
    lcm::T6
    cent::T7
    card::T8
    gcard::T9
    sets::T10
    nplg::T11
    cplg::T12
    lt::T13 # long threshold
    st::T14
    tn::T15 # Turnover
    te::T16 # TrackingError
    nea::T17
    l1::T18
    l2::T19
    fees::T20
    sce::T21
    ret::T22
    ccnt::T23
    cobj::T24
    sc::T25
    so::T26
    ss::T27
    slv::T28
    strict::T29
end
function assert_finite_nonnegative_real_or_vec(val::Real)
    @smart_assert(isfinite(val) && val > zero(val))
    return nothing
end
function assert_finite_nonnegative_real_or_vec(val::AbstractVector{<:Real})
    @smart_assert(any(isfinite, val) && any(val .> zero(val)) && all(val .>= zero(val)))
    return nothing
end
function JuMPOptimiser(;
                       pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                       wb::Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint} = WeightBoundsResult(),
                       bgt::Union{Nothing, <:Real, <:BudgetRange} = 1.0,
                       sbgt::Union{Nothing, <:Real, <:BudgetRange} = nothing,
                       lcs::Union{Nothing, <:LinearConstraint,
                                  <:AbstractVector{<:LinearConstraint},
                                  <:LinearConstraintResult} = nothing,
                       lcm::Union{Nothing, <:LinearConstraintResult} = nothing,
                       cent::Union{Nothing, <:CentralityConstraintEstimator,
                                   <:AbstractVector{<:CentralityConstraintEstimator},
                                   <:LinearConstraintResult} = nothing,
                       card::Union{Nothing, <:Integer} = nothing,
                       gcard::Union{Nothing, <:CardinalityConstraint,
                                    <:AbstractVector{<:CardinalityConstraint},
                                    <:LinearConstraintResult} = nothing,
                       sets::Union{Nothing, <:DataFrame} = nothing,
                       nplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       cplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       lt::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                       st::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                       tn::Union{Nothing, <:Turnover} = nothing,
                       te::Union{Nothing, <:TrackingError} = nothing,
                       nea::Union{Nothing, <:Real} = nothing,
                       l1::Union{Nothing, <:Real} = nothing,
                       l2::Union{Nothing, <:Real} = nothing,
                       fees::Union{Nothing, <:Fees} = nothing,
                       sce::Scalariser = SumScalariser(),
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       ccnt::Union{Nothing, <:CustomConstraint} = nothing,
                       cobj::Union{Nothing, <:CustomObjective} = nothing, sc::Real = 1,
                       so::Real = 1, ss::Union{Nothing, <:Real} = nothing,
                       slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                       strict::Bool = false)
    if isa(bgt, Real)
        @smart_assert(isfinite(bgt) && bgt >= 0)
    end
    if isa(sbgt, Real)
        @smart_assert(isfinite(sbgt) && sbgt >= 0)
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
    if isa(gcard, AbstractVector)
        @smart_assert(!isempty(gcard))
    end
    if isa(wb, WeightBoundsConstraint) ||
       isa(lcs, LinearConstraint) ||
       isa(lcs, AbstractVector{<:LinearConstraint}) ||
       isa(cent, CentralityConstraintEstimator) ||
       isa(cent, AbstractVector{<:CentralityConstraintEstimator}) ||
       isa(gcard, CardinalityConstraint) ||
       isa(gcard, AbstractVector{<:CardinalityConstraint})
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    if !isnothing(lt)
        assert_finite_nonnegative_real_or_vec(lt)
    end
    if !isnothing(st)
        assert_finite_nonnegative_real_or_vec(st)
    end
    if !isnothing(nea)
        @smart_assert(nea > zero(nea))
    end
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return JuMPOptimiser{typeof(pe), typeof(wb), typeof(bgt), typeof(sbgt), typeof(lcs),
                         typeof(lcm), typeof(cent), typeof(card), typeof(gcard),
                         typeof(sets), typeof(nplg), typeof(cplg), typeof(lt), typeof(st),
                         typeof(tn), typeof(te), typeof(nea), typeof(l1), typeof(l2),
                         typeof(fees), typeof(sce), typeof(ret), typeof(ccnt), typeof(cobj),
                         typeof(sc), typeof(so), typeof(ss), typeof(slv), typeof(strict)}(pe,
                                                                                          wb,
                                                                                          bgt,
                                                                                          sbgt,
                                                                                          lcs,
                                                                                          lcm,
                                                                                          cent,
                                                                                          card,
                                                                                          gcard,
                                                                                          sets,
                                                                                          nplg,
                                                                                          cplg,
                                                                                          lt,
                                                                                          st,
                                                                                          tn,
                                                                                          te,
                                                                                          nea,
                                                                                          l1,
                                                                                          l2,
                                                                                          fees,
                                                                                          sce,
                                                                                          ret,
                                                                                          ccnt,
                                                                                          cobj,
                                                                                          sc,
                                                                                          so,
                                                                                          ss,
                                                                                          slv,
                                                                                          strict)
end
function opt_view(opt::JuMPOptimiser, i::AbstractVector)
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    lcs = linear_constraint_view(opt.lcs, i)
    gcard = cardinality_constraint_view(opt.gcard, i)
    sets = nothing_dataframe_view(opt.sets, i)
    lt = nothing_scalar_array_view(opt.lt, i)
    st = nothing_scalar_array_view(opt.lt, i)
    tn = turnover_view(opt.tn, i)
    te = tracking_view(opt.te, i)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(pe, wb, opt.bgt, opt.sbgt, lcs, opt.lcm, opt.cent, opt.card, gcard,
                         sets, opt.nplg, opt.cplg, lt, st, tn, te, opt.nea, opt.l1, opt.l2,
                         fees, opt.sce, ret, ccnt, cobj, opt.sc, opt.so, opt.ss, opt.slv,
                         opt.strict)
end
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1)
    pr = prior(opt.pe, rd.X, rd.F; dims = dims)
    datatype = eltype(pr.X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(pr.X, 2), strict = opt.strict)
    lcs = linear_constraints(opt.lcs, opt.sets; datatype = datatype, strict = opt.strict)
    cent = centrality_constraints(opt.cent, pr.X)
    gcard = cardinality_constraints(opt.gcard, opt.sets; datatype = datatype,
                                    strict = opt.strict)
    nplg = philogeny_constraints(opt.nplg, pr.X)
    cplg = philogeny_constraints(opt.cplg, pr.X)
    return pr, wb, lcs, cent, gcard, nplg, cplg
end

export JuMPOptimisationResult, JuMPOptimiser
