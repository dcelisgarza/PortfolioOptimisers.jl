struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
                     T2 <: Union{Nothing, <:WeightBounds, <:WeightBoundsConstraints},
                     T3 <: Union{Nothing, <:Real, <:BudgetConstraint},
                     T4 <: Union{Nothing, <:LongShortSum},
                     T5 <: Union{Nothing, <:LinearConstraint,
                                 <:AbstractVector{<:LinearConstraint}, <:LinearConstraintModel},
                     T6 <: Union{Nothing, <:CardinalityConstraint,
                                 <:AbstractVector{<:CardinalityConstraint},
                                 <:LinearConstraintModel},
                     T7 <: Union{Nothing, <:CentralityConstraint,
                                 <:AbstractVector{<:CentralityConstraint},
                                 <:LinearConstraintModel}, T8 <: Union{Nothing, DataFrame},
                     T9 <: Union{Nothing, <:PhilogenyEstimator, <:PhilogenyConstraintModel},
                     T10 <:
                     Union{Nothing, <:PhilogenyEstimator, <:PhilogenyConstraintModel},
                     T11 <: Union{Nothing, <:BuyInThreshold}, T12 <: Union{Nothing, Fees},
                     T13 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                     T14 <: Scalariser}
    pe::T1 # PriorEstimator
    wb::T2 # WeightBounds
    bgt::T3 # BudgetConstraint
    lss::T4 # LongShortSum
    lcs::T5
    card::T6
    cent::T7
    sets::T8
    nadj::T9
    cadj::T10
    bit::T11 # BuyInThreshold
    fees::T12
    slv::T13
    sce::T14
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
