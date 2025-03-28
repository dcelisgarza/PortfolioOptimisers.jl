# struct JuMPOptimiser{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorModel},
#                      T2 <: Union{<:ClusteringEstimator,
#                                  <:AbstractPortfolioOptimisersClusteringResult},
#                      T3 <: Union{Nothing, Fees}, T4 <: Scalariser,
#                      T5 <: Union{<:WeightBounds, WeightBoundsConstraints},
#                      T7 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}}
#     pe::T1
#     cle::T2
#     fees::T3
#     sce::T4
#     wb::T5
#     slv::T7
# end
