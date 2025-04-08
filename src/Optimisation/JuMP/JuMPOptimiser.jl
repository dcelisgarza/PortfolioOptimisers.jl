struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                     T2 <: Union{Nothing, <:AbstractVector{<:Real}},
                     T3 <: Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints},
                     T4 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T5 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T6 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintModel},
                     T7 <: Union{Nothing, <:LinearConstraintModel},
                     T8 <: Union{Nothing, <:CentralityConstraint,
                                 <:AbstractVector{<:CentralityConstraint},
                                 <:LinearConstraintModel},
                     T9 <: Union{Nothing, <:CardinalityConstraint,
                                 <:AbstractVector{<:CardinalityConstraint},
                                 <:LinearConstraintModel}, T10 <: Union{Nothing, DataFrame},
                     T11 <:
                     Union{Nothing, <:PhilogenyConstraint, <:PhilogenyConstraintModel},
                     T12 <:
                     Union{Nothing, <:PhilogenyConstraint, <:PhilogenyConstraintModel},
                     T13 <: Union{Nothing, <:BuyInThreshold},
                     T14 <: Union{Nothing, <:Turnover},
                     T15 <: Union{Nothing, <:TrackingError}, T16 <: Union{Nothing, <:Real},
                     T17 <: Union{Nothing, <:Real}, T18 <: Union{Nothing, Fees},
                     T19 <: Scalariser, T20 <: PortfolioReturnType,
                     T21 <: Union{Nothing, <:CustomConstraint},
                     T22 <: Union{Nothing, <:CustomObjective}, T23 <: Real, T24 <: Real,
                     T25 <: Real, T26 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                     T27 <: Bool, T28 <: Bool, T29 <: Bool} <: JuMPOptimisationType
    pe::T1 # PriorEstimator
    wi::T2
    wb::T3 # WeightBounds
    bgt::T4 # BudgetConstraint
    sbgt::T5 # LongShortSum
    lcs::T6
    lcm::T7
    cent::T8
    card::T9
    sets::T10
    nplg::T11
    cplg::T12
    bit::T13 # BuyInThreshold
    tn::T14 # Turnover
    te::T15 # TrackingError
    l1::T16
    l2::T17
    fees::T18
    sce::T19
    ret::T20
    ccnt::T21
    cobj::T22
    sc::T23
    so::T24
    ss::T25
    slv::T26
    str_names::T27
    save::T28
    strict::T29
end
function JuMPOptimiser(;
                       pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                       wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                       wb::Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints} = WeightBounds(),
                       bgt::Union{Nothing, <:Real, <:BudgetConstraint} = 1.0,
                       sbgt::Union{Nothing, <:Real, <:BudgetConstraint} = nothing,
                       lcs::Union{Nothing, <:LinearConstraint,
                                  <:AbstractVector{<:LinearConstraint},
                                  <:LinearConstraintModel} = nothing,
                       lcm::Union{Nothing, <:LinearConstraintModel} = nothing,
                       cent::Union{Nothing, <:CentralityConstraint,
                                   <:AbstractVector{<:CentralityConstraint},
                                   <:LinearConstraintModel} = nothing,
                       card::Union{Nothing, <:CardinalityConstraint,
                                   <:AbstractVector{<:CardinalityConstraint},
                                   <:LinearConstraintModel} = nothing,
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
    if isa(card, AbstractVector)
        @smart_assert(!isempty(card))
    end
    if isa(lcs, LinearConstraint) ||
       isa(lcs, AbstractVector{<:LinearConstraint}) ||
       isa(cent, CentralityConstraint) ||
       isa(cent, AbstractVector{<:CentralityConstraint}) ||
       isa(card, CardinalityConstraint) ||
       isa(card, AbstractVector{<:CardinalityConstraint})
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return JuMPOptimiser{typeof(pe), typeof(wi), typeof(wb), typeof(bgt), typeof(sbgt),
                         typeof(lcs), typeof(lcm), typeof(cent), typeof(card), typeof(sets),
                         typeof(nplg), typeof(cplg), typeof(bit), typeof(tn), typeof(te),
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
                                                                                       sets,
                                                                                       nplg,
                                                                                       cplg,
                                                                                       bit,
                                                                                       tn,
                                                                                       te,
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
