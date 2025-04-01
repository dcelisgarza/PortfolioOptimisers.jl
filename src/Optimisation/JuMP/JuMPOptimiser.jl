struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
                     T2 <: Union{Nothing, <:AbstractVector{<:Real}},
                     T3 <: Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints},
                     T4 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T5 <: Union{Nothing, <:LongShortSum},
                     T6 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintModel},
                     T7 <: Union{Nothing, <:LinearConstraintModel},
                     T8 <: Union{Nothing, <:CardinalityConstraint,
                                 <:AbstractVector{<:CardinalityConstraint},
                                 <:LinearConstraintModel},
                     T9 <: Union{Nothing, <:CentralityConstraint,
                                 <:AbstractVector{<:CentralityConstraint},
                                 <:LinearConstraintModel}, T10 <: Union{Nothing, DataFrame},
                     T11 <:
                     Union{Nothing, <:PhilogenyEstimator, <:PhilogenyConstraintModel},
                     T12 <:
                     Union{Nothing, <:PhilogenyEstimator, <:PhilogenyConstraintModel},
                     T13 <: Union{Nothing, <:BuyInThreshold},
                     T14 <: Union{Nothing, <:Turnover},
                     T15 <: Union{Nothing, <:TrackingError}, T16 <: Union{Nothing, <:Real},
                     T17 <: Union{Nothing, <:Real}, T18 <: Union{Nothing, Fees},
                     T19 <: Scalariser, T20 <: PortfolioReturnType,
                     T21 <: Union{Nothing, <:CustomConstraint},
                     T22 <: Union{Nothing, <:CustomObjective}, T23 <: Real, T24 <: Real,
                     T25 <: Union{<:Solver, <:AbstractVector{<:Solver}}, T26 <: Bool,
                     T27 <: Bool, T28 <: Bool} <: JuMPOptimisationType
    pe::T1 # PriorEstimator
    wi::T2
    wb::T3 # WeightBounds
    bgt::T4 # BudgetConstraint
    lss::T5 # LongShortSum
    lcs::T6
    lcm::T7
    card::T7
    cent::T8
    sets::T10
    nadj::T11
    cadj::T12
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
    slv::T25
    str_names::T26
    save::T27
    strict::T28
end
function JuMPOptimiser(;
                       pe::Union{<:AbstractPriorEstimator, <:AbstractPriorModel} = EmpiricalPriorEstimator(),
                       wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                       wb::Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints} = WeightBounds(),
                       bgt::Union{Nothing, <:Real, <:BudgetConstraint} = 1.0,
                       lss::Union{Nothing, <:LongShortSum} = nothing,
                       lcs::Union{Nothing, <:LinearConstraint,
                                  <:AbstractVector{<:LinearConstraint},
                                  <:LinearConstraintModel} = nothing,
                       lcm::Union{Nothing, <:LinearConstraintModel} = nothing,
                       card::Union{Nothing, <:CardinalityConstraint,
                                   <:AbstractVector{<:CardinalityConstraint},
                                   <:LinearConstraintModel} = nothing,
                       cent::Union{Nothing, <:CentralityConstraint,
                                   <:AbstractVector{<:CentralityConstraint},
                                   <:LinearConstraintModel} = nothing,
                       sets::Union{Nothing, DataFrame} = nothing,
                       nadj::Union{Nothing, <:PhilogenyEstimator,
                                   <:PhilogenyConstraintModel} = nothing,
                       cadj::Union{Nothing, <:PhilogenyEstimator,
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
                       so::Real = 1, slv::Union{<:Solver, <:AbstractVector{<:Solver}},
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
    if isa(card, AbstractVector)
        @smart_assert(!isempty(card))
    end
    if isa(cent, AbstractVector)
        @smart_assert(!isempty(cent))
    end
    if isa(lcs, LinearConstraint) ||
       isa(lcs, AbstractVector{<:LinearConstraint}) ||
       isa(card, CardinalityConstraint) ||
       isa(card, AbstractVector{<:CardinalityConstraint}) ||
       isa(cent, CentralityConstraint) ||
       isa(cent, AbstractVector{<:CentralityConstraint})
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return JuMPOptimiser{typeof(pe), typeof(wi), typeof(wb), typeof(bgt), typeof(lss),
                         typeof(lcs), typeof(lcm), typeof(card), typeof(cent), typeof(sets),
                         typeof(nadj), typeof(cadj), typeof(bit), typeof(tn), typeof(te),
                         typeof(l1), typeof(l2), typeof(fees), typeof(sce), typeof(ret),
                         typeof(ccnt), typeof(cobj), typeof(sc), typeof(so), typeof(slv),
                         typeof(str_names), typeof(save), typeof(strict)}(pe, wi, wb, bgt,
                                                                          lss, lcs, lcm,
                                                                          card, cent, sets,
                                                                          nadj, cadj, bit,
                                                                          tn, te, l1, l2,
                                                                          fees, sce, ret,
                                                                          ccnt, cobj, sc,
                                                                          so, slv,
                                                                          str_names, save,
                                                                          strict)
end

export JuMPOptimiser
