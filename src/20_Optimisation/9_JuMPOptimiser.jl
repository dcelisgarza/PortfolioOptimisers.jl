struct JuMPOptimisationResult{T1 <: Type, T2 <: AbstractPriorResult,
                              T3 <: Union{Nothing, <:WeightBoundsResult},
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
                                                    Union{Nothing, <:WeightBoundsResult},
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
                     T2 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                     T3 <: Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                     T4 <: Union{Nothing, <:Real, <:BudgetRange},
                     T5 <: Union{Nothing, <:Real, <:BudgetRange},
                     T6 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                     T7 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                     T8 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintResult},
                     T9 <: Union{Nothing, <:LinearConstraintResult},
                     T10 <: Union{Nothing, <:CentralityConstraintEstimator,
                                  <:AbstractVector{<:CentralityConstraintEstimator},
                                  <:LinearConstraintResult},
                     T11 <: Union{Nothing, <:CardinalityConstraint,
                                  <:AbstractVector{<:CardinalityConstraint},
                                  <:LinearConstraintResult},
                     T12 <: Union{Nothing, <:DataFrame},
                     T13 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T14 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T15 <: Union{Nothing, <:Turnover},
                     T16 <: Union{Nothing, <:TrackingError},
                     T17 <: Union{Nothing, <:TrackingError},
                     T18 <: Union{Nothing, <:VolTrackingError},
                     T19 <: Union{Nothing, <:Fees}, T20 <: JuMPReturnsEstimator,
                     T21 <: Scalariser, T22 <: Union{Nothing, <:CustomConstraint},
                     T23 <: Union{Nothing, <:CustomObjective}, T24 <: Real, T25 <: Real,
                     T26 <: Union{Nothing, <:Integer}, T27 <: Union{Nothing, <:Real},
                     T28 <: Union{Nothing, <:Real}, T29 <: Union{Nothing, <:Real},
                     T30 <: Union{Nothing, <:Real}, T31 <: Bool} <:
       BaseJuMPOptimisationEstimator
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
    sets::T12
    nplg::T13
    cplg::T14
    tn::T15 # Turnover
    te1::T16 # TrackingError
    te2::T17 # TrackingError
    tev::T18
    fees::T19
    ret::T20
    sce::T21
    ccnt::T22
    cobj::T23
    sc::T24
    so::T25
    card::T26
    nea::T27
    l1::T28
    l2::T29
    ss::T30
    strict::T31
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
                       slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                       wb::Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint} = WeightBoundsResult(),
                       bgt::Union{Nothing, <:Real, <:BudgetRange} = 1.0,
                       sbgt::Union{Nothing, <:Real, <:BudgetRange} = nothing,
                       lt::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                       st::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                       lcs::Union{Nothing, <:LinearConstraint,
                                  <:AbstractVector{<:LinearConstraint},
                                  <:LinearConstraintResult} = nothing,
                       lcm::Union{Nothing, <:LinearConstraintResult} = nothing,
                       cent::Union{Nothing, <:CentralityConstraintEstimator,
                                   <:AbstractVector{<:CentralityConstraintEstimator},
                                   <:LinearConstraintResult} = nothing,
                       gcard::Union{Nothing, <:CardinalityConstraint,
                                    <:AbstractVector{<:CardinalityConstraint},
                                    <:LinearConstraintResult} = nothing,
                       sets::Union{Nothing, <:DataFrame} = nothing,
                       nplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       cplg::Union{Nothing, <:PhilogenyConstraintEstimator,
                                   <:PhilogenyConstraintResult} = nothing,
                       tn::Union{Nothing, <:Turnover} = nothing,
                       te1::Union{Nothing, <:TrackingError} = nothing,
                       te2::Union{Nothing, <:TrackingError} = nothing,
                       tev::Union{Nothing, <:VolTrackingError} = nothing,
                       fees::Union{Nothing, <:Fees} = nothing,
                       ret::JuMPReturnsEstimator = ArithmeticReturn(),
                       sce::Scalariser = SumScalariser(),
                       ccnt::Union{Nothing, <:CustomConstraint} = nothing,
                       cobj::Union{Nothing, <:CustomObjective} = nothing, sc::Real = 1,
                       so::Real = 1, card::Union{Nothing, <:Integer} = nothing,
                       nea::Union{Nothing, <:Real} = nothing,
                       l1::Union{Nothing, <:Real} = nothing,
                       l2::Union{Nothing, <:Real} = nothing,
                       ss::Union{Nothing, <:Real} = nothing, strict::Bool = false)
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
    return JuMPOptimiser{typeof(pe), typeof(slv), typeof(wb), typeof(bgt), typeof(sbgt),
                         typeof(lt), typeof(st), typeof(lcs), typeof(lcm), typeof(cent),
                         typeof(gcard), typeof(sets), typeof(nplg), typeof(cplg),
                         typeof(tn), typeof(te1), typeof(te2), typeof(tev), typeof(fees),
                         typeof(ret), typeof(sce), typeof(ccnt), typeof(cobj), typeof(sc),
                         typeof(so), typeof(card), typeof(nea), typeof(l1), typeof(l2),
                         typeof(ss), typeof(strict)}(pe, slv, wb, bgt, sbgt, lt, st, lcs,
                                                     lcm, cent, gcard, sets, nplg, cplg, tn,
                                                     te1, te2, tev, fees, ret, sce, ccnt,
                                                     cobj, sc, so, card, nea, l1, l2, ss,
                                                     strict)
end
function opt_view(opt::JuMPOptimiser, i::AbstractVector)
    pe = prior_view(opt.pe, i)
    wb = weight_bounds_view(opt.wb, i)
    lt = nothing_scalar_array_view(opt.lt, i)
    st = nothing_scalar_array_view(opt.lt, i)
    lcs = linear_constraint_view(opt.lcs, i)
    gcard = cardinality_constraint_view(opt.gcard, i)
    sets = nothing_dataframe_view(opt.sets, i)
    tn = turnover_view(opt.tn, i)
    te1 = tracking_view(opt.te1, i)
    te2 = tracking_view(opt.te2, i)
    tev = tracking_view(opt.tev, i)
    fees = fees_view(opt.fees, i)
    ret = jump_returns_view(opt.ret, i)
    ccnt = custom_constraint_view(opt.ccnt, i)
    cobj = custom_objective_view(opt.cobj, i)
    return JuMPOptimiser(; pe = pe, slv = opt.slv, wb = wb, bgt = opt.bgt, sbgt = opt.sbgt,
                         lt = lt, st = st, lcs = lcs, lcm = opt.lcm, cent = opt.cent,
                         gcard = gcard, sets = sets, nplg = opt.nplg, cplg = opt.cplg,
                         tn = tn, te1 = te1, te2 = te2, tev = tev, fees = fees, ret = ret,
                         sce = opt.sce, ccnt = ccnt, cobj = cobj, sc = opt.sc, so = opt.so,
                         card = opt.card, nea = opt.nea, l1 = opt.l1, l2 = opt.l2,
                         ss = opt.ss, strict = opt.strict)
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
