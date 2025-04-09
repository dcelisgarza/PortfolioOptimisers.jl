struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                     T2 <: Union{Nothing, <:AbstractVector{<:Real}},
                     T3 <: Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints},
                     T4 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T5 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T6 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintResult},
                     T7 <: Union{Nothing, <:LinearConstraintResult},
                     T8 <: Union{Nothing, <:CentralityConstraint,
                                 <:AbstractVector{<:CentralityConstraint},
                                 <:LinearConstraintResult}, T9::Union{Nothing, <:Integer},
                     T10 <: Union{Nothing, <:CardinalityConstraint,
                                  <:AbstractVector{<:CardinalityConstraint},
                                  <:LinearConstraintResult},
                     T11 <: Union{Nothing, DataFrame},
                     T12 <:
                     Union{Nothing, <:PhilogenyConstraint, <:PhilogenyConstraintModel},
                     T13 <:
                     Union{Nothing, <:PhilogenyConstraint, <:PhilogenyConstraintModel},
                     T14 <: Union{Nothing, <:BuyInThreshold},
                     T15 <: Union{Nothing, <:Turnover},
                     T16 <: Union{Nothing, <:TrackingError}, T17 <: Union{Nothing, <:Real},
                     T18 <: Union{Nothing, <:Real}, T19 <: Union{Nothing, Fees},
                     T20 <: Scalariser, T21 <: PortfolioReturnType,
                     T22 <: Union{Nothing, <:CustomConstraint},
                     T23 <: Union{Nothing, <:CustomObjective}, T24 <: Real, T25 <: Real,
                     T26 <: Real, T27 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                     T28 <: Bool, T29 <: Bool, T30 <: Bool} <: JuMPOptimisationType
    pe::T1 # PriorEstimator
    wi::T2
    wb::T3 # WeightBounds
    bgt::T4 # BudgetConstraint
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
    l1::T17
    l2::T18
    fees::T19
    sce::T20
    ret::T21
    ccnt::T22
    cobj::T23
    sc::T24
    so::T25
    ss::T26
    slv::T27
    str_names::T28
    save::T29
    strict::T30
end
function JuMPOptimiser(;
                       pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                       wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                       wb::Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints} = WeightBounds(),
                       bgt::Union{Nothing, <:Real, <:BudgetConstraint} = 1.0,
                       sbgt::Union{Nothing, <:Real, <:BudgetConstraint} = nothing,
                       lcs::Union{Nothing, <:LinearConstraint,
                                  <:AbstractVector{<:LinearConstraint},
                                  <:LinearConstraintResult} = nothing,
                       lcm::Union{Nothing, <:LinearConstraintResult} = nothing,
                       cent::Union{Nothing, <:CentralityConstraint,
                                   <:AbstractVector{<:CentralityConstraint},
                                   <:LinearConstraintResult} = nothing,
                       card::Union{Nothing, <:Integer} = nothing,
                       gcard::Union{Nothing, <:CardinalityConstraint,
                                    <:AbstractVector{<:CardinalityConstraint},
                                    <:LinearConstraintResult} = nothing,
                       sets::Union{Nothing, DataFrame} = nothing,
                       nplg::Union{Nothing, <:PhilogenyEstimator,
                                   <:PhilogenyConstraintModel} = nothing,
                       cplg::Union{Nothing, <:PhilogenyEstimator,
                                   <:PhilogenyConstraintModel} = nothing,
                       bit::Union{Nothing, <:BuyInThreshold} = nothing,
                       tn::Union{Nothing, <:Turnover} = nothing,
                       te::Union{Nothing, <:TrackingError} = nothing,
                       l1::Union{Nothing, <:Real} = nothing,
                       l2::Union{Nothing, <:Real} = nothing,
                       fees::Union{Nothing, Fees} = nothing,
                       sce::Scalariser = SumScalariser(),
                       ret::PortfolioReturnType = ArithmeticReturn(),
                       ccnt::Union{Nothing, <:CustomConstraint} = nothing,
                       cobj::Union{Nothing, <:CustomObjective} = nothing, sc::Real = 1,
                       so::Real = 1, ss::Real = 100_000.0,
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
    if isa(lcs, LinearConstraint) ||
       isa(lcs, AbstractVector{<:LinearConstraint}) ||
       isa(cent, CentralityConstraint) ||
       isa(cent, AbstractVector{<:CentralityConstraint}) ||
       isa(gcard, CardinalityConstraint) ||
       isa(gcard, AbstractVector{<:CardinalityConstraint})
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return JuMPOptimiser{typeof(pe), typeof(wi), typeof(wb), typeof(bgt), typeof(sbgt),
                         typeof(lcs), typeof(lcm), typeof(cent), typeof(card),
                         typeof(gcard), typeof(sets), typeof(nplg), typeof(cplg),
                         typeof(bit), typeof(tn), typeof(te), typeof(l1), typeof(l2),
                         typeof(fees), typeof(sce), typeof(ret), typeof(ccnt), typeof(cobj),
                         typeof(sc), typeof(so), typeof(ss), typeof(slv), typeof(str_names),
                         typeof(save), typeof(strict)}(pe, wi, wb, bgt, sbgt, lcs, lcm,
                                                       cent, card, gcard, sets, nplg, cplg,
                                                       bit, tn, te, l1, l2, fees, sce, ret,
                                                       ccnt, cobj, sc, so, ss, slv,
                                                       str_names, save, strict)
end

export JuMPOptimiser
