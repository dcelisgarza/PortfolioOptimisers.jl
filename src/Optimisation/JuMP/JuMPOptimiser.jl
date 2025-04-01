struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
                     T2 <: Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints},
                     T3 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T4 <: Union{Nothing, <:LongShortSum},
                     T5 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintModel},
                     T6 <: Union{Nothing, <:LinearConstraintModel},
                     T7 <: Union{Nothing, <:CardinalityConstraint,
                                 <:AbstractVector{<:CardinalityConstraint},
                                 <:LinearConstraintModel},
                     T8 <: Union{Nothing, <:CentralityConstraint,
                                 <:AbstractVector{<:CentralityConstraint},
                                 <:LinearConstraintModel}, T9 <: Union{Nothing, DataFrame},
                     T10 <:
                     Union{Nothing, <:PhilogenyEstimator, <:PhilogenyConstraintModel},
                     T11 <:
                     Union{Nothing, <:PhilogenyEstimator, <:PhilogenyConstraintModel},
                     T12 <: Union{Nothing, <:BuyInThreshold},
                     T13 <: Union{Nothing, <:Turnover},
                     T14 <: Union{Nothing, <:TrackingError}, T15 <: Union{Nothing, <:Real},
                     T16 <: Union{Nothing, <:Real}, T17 <: Union{Nothing, Fees},
                     T18 <: Scalariser, T19 <: PortfolioReturnType,
                     T20 <: Union{Nothing, <:CustomConstraint},
                     T21 <: Union{Nothing, <:CustomObjective}, T22 <: Real, T23 <: Real,
                     T24 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}}
    pe::T1 # PriorEstimator
    wb::T2 # WeightBounds
    bgt::T3 # BudgetConstraint
    lss::T4 # LongShortSum
    lcs::T5
    lcm::T6
    card::T7
    cent::T8
    sets::T9
    nadj::T10
    cadj::T11
    bit::T12 # BuyInThreshold
    tn::T13 # Turnover
    te::T14 # TrackingError
    l1::T15
    l2::T16
    fees::T17
    sce::T18
    ret::T19
    ccnt::T20
    cobj::T21
    sc::T22
    so::T23
    slv::T24
end
function JuMPOptimiser(; bgt::Union{Nothing, <:Real, <:BudgetConstraint} = 1.0,
                       lcs::Union{Nothing, <:LinearConstraint,
                                  <:AbstractVector{<:LinearConstraint},
                                  <:LinearConstraintModel} = nothing,
                       card::Union{Nothing, <:CardinalityConstraint,
                                   <:AbstractVector{<:CardinalityConstraint},
                                   <:LinearConstraintModel} = nothing,
                       cent::Union{Nothing, <:LinearConstraintModel} = nothing,
                       sets::Union{Nothing, DataFrame} = nothing)
    if isa(bgt, Real)
        @smart_assert(isfinite(bgt))
    end
    if isa(lcs, LinearConstraint) ||
       isa(lcs, AbstractVector{<:LinearConstraint}) ||
       isa(card, CardinalityConstraint) ||
       isa(card, AbstractVector{<:CardinalityConstraint}) ||
       isa(cent, CentralityConstraint) ||
       isa(cent, AbstractVector{<:CentralityConstraint})
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
end
