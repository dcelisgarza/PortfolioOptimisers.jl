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
                     T14 <: Union{Nothing, <:BuyInThreshold},
                     T15 <: Union{Nothing, <:Turnover},
                     T16 <: Union{Nothing, <:TrackingError}, T17 <: Union{Nothing, <:Real},
                     T18 <: Union{Nothing, <:Real}, T19 <: Union{Nothing, <:Real},
                     T20 <: Union{Nothing, <:Fees}, T21 <: Scalariser,
                     T22 <: JuMPReturnsEstimator, T23 <: Union{Nothing, <:CustomConstraint},
                     T24 <: Union{Nothing, <:CustomObjective}, T25 <: Real, T26 <: Real,
                     T27 <: Union{Nothing, <:Real},
                     T28 <: Union{<:Solver, <:AbstractVector{<:Solver}}, T29 <: Bool,
                     T30 <: Bool, T31 <: Bool} <: JuMPOptimisationEstimator
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
    bit::T14 # BuyInThreshold
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
    str_names::T29
    save::T30
    strict::T31
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
                       bit::Union{Nothing, <:BuyInThreshold} = nothing,
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
        @smart_assert(isfinite(bgt))
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
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    if !isnothing(nea)
        @smart_assert(nea > zero(nea))
    end
    return JuMPOptimiser{typeof(pe), typeof(wi), typeof(wb), typeof(bgt), typeof(sbgt),
                         typeof(lcs), typeof(lcm), typeof(cent), typeof(card),
                         typeof(gcard), typeof(sets), typeof(nplg), typeof(cplg),
                         typeof(bit), typeof(tn), typeof(te), typeof(nea), typeof(l1),
                         typeof(l2), typeof(fees), typeof(sce), typeof(ret), typeof(ccnt),
                         typeof(cobj), typeof(sc), typeof(so), typeof(ss), typeof(slv),
                         typeof(str_names), typeof(save), typeof(strict)}(pe, wi, wb, bgt,
                                                                          sbgt, lcs, lcm,
                                                                          cent, card, gcard,
                                                                          sets, nplg, cplg,
                                                                          bit, tn, te, nea,
                                                                          l1, l2, fees, sce,
                                                                          ret, ccnt, cobj,
                                                                          sc, so, ss, slv,
                                                                          str_names, save,
                                                                          strict)
end

export JuMPOptimiser
