# struct TraditionalOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
#                             T2 <: Union{<:WeightBounds, WeightBoundsConstraints},
#                             T3 <: Union{<:Real, <:BudgetConstraint}, T4 <: LongShortBounds,
#                             T5 <: Integer, T6 <: Union{Nothing, <:LinearConstraintModel},
#                             T3 <: Union{Nothing, Fees}, T4 <: Scalariser,
#                             T7 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}}
#     pe::T1
#     wb::T2
#     bgt::T3
#     slb::T4
#     card::T5
#     gcard::T6
#     fees::T3
#     sce::T4
#     slv::T7
# end

# export TraditionalOptimiser