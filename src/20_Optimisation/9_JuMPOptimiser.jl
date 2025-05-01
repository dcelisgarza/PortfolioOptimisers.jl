struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                     T2 <: Union{Nothing, <:AbstractVector{<:Real}},
                     T3 <: Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                     T4 <: Union{Nothing, <:Real, <:BudgetRange},
                     T5 <: Union{Nothing, <:Real, <:BudgetRange},
                     T6 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintResult},
                     T7 <: Union{Nothing, <:LinearConstraintResult},
                     T8 <: Union{Nothing, <:CentralityConstraintEstimator,
                                 <:AbstractVector{<:CentralityConstraintEstimator},
                                 <:LinearConstraintResult}, T9 <: Union{Nothing, <:Integer},
                     T10 <: Union{Nothing, <:CardinalityConstraint,
                                  <:AbstractVector{<:CardinalityConstraint},
                                  <:LinearConstraintResult},
                     T11 <: Union{Nothing, <:DataFrame},
                     T12 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T13 <: Union{Nothing, <:PhilogenyConstraintEstimator,
                                  <:PhilogenyConstraintResult},
                     T14 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                     T15 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                     T16 <: Union{Nothing, <:Turnover},
                     T17 <: Union{Nothing, <:TrackingError}, T18 <: Union{Nothing, <:Real},
                     T19 <: Union{Nothing, <:Real}, T20 <: Union{Nothing, <:Real},
                     T21 <: Union{Nothing, <:Fees}, T22 <: Scalariser,
                     T23 <: JuMPReturnsEstimator, T24 <: Union{Nothing, <:CustomConstraint},
                     T25 <: Union{Nothing, <:CustomObjective}, T26 <: Real, T27 <: Real,
                     T28 <: Union{Nothing, <:Real},
                     T29 <: Union{<:Solver, <:AbstractVector{<:Solver}}, T30 <: Bool,
                     T31 <: Bool, T32 <: Bool} <: JuMPOptimisationEstimator
    pe::T1 # PriorEstimator
    wi::T2
    wb::T3 # WeightBoundsResult
    bgt::T4 # BudgetRange
    sbgt::T5 # LongShortSum
    lcs::T6
    lcm::T7
    cent::T8
    card::T9
    gcard::T10
    sets::T11
    nplg::T12
    cplg::T13
    lt::T14 # long threshold
    st::T15
    tn::T16 # Turnover
    te::T17 # TrackingError
    nea::T18
    l1::T19
    l2::T20
    fees::T21
    sce::T22
    ret::T23
    ccnt::T24
    cobj::T25
    sc::T26
    so::T27
    ss::T28
    slv::T29
    str_names::T30
    save::T31
    strict::T32
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
                       wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
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
                       str_names::Bool = false, save::Bool = false, strict::Bool = false)
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
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
    return JuMPOptimiser{typeof(pe), typeof(wi), typeof(wb), typeof(bgt), typeof(sbgt),
                         typeof(lcs), typeof(lcm), typeof(cent), typeof(card),
                         typeof(gcard), typeof(sets), typeof(nplg), typeof(cplg),
                         typeof(lt), typeof(st), typeof(tn), typeof(te), typeof(nea),
                         typeof(l1), typeof(l2), typeof(fees), typeof(sce), typeof(ret),
                         typeof(ccnt), typeof(cobj), typeof(sc), typeof(so), typeof(ss),
                         typeof(slv), typeof(str_names), typeof(save), typeof(strict)}(pe,
                                                                                       wi,
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
                                                                                       str_names,
                                                                                       save,
                                                                                       strict)
end

export JuMPOptimiser
