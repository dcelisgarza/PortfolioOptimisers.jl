abstract type AbstractPriorEstimator <: AbstractEstimator end
abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_1_0 <: AbstractLowOrderPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_2_1 <: AbstractLowOrderPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_2_2 <: AbstractLowOrderPriorEstimator end
abstract type AbstractLowOrderPriorEstimator_1o2_1o2 <: AbstractLowOrderPriorEstimator end
const AbstractLowOrderPriorEstimatorMap_2_1 = Union{<:AbstractLowOrderPriorEstimator_1_0,
                                                    <:AbstractLowOrderPriorEstimator_1o2_1o2}
const AbstractLowOrderPriorEstimatorMap_2_2 = Union{<:AbstractLowOrderPriorEstimator_2_1,
                                                    <:AbstractLowOrderPriorEstimator_2_2,
                                                    <:AbstractLowOrderPriorEstimator_1o2_1o2}
const AbstractLowOrderPriorEstimatorMap_1o2_1o2 = Union{<:AbstractLowOrderPriorEstimator_1_0,
                                                        <:AbstractLowOrderPriorEstimator_2_1,
                                                        <:AbstractLowOrderPriorEstimator_2_2,
                                                        <:AbstractLowOrderPriorEstimator_1o2_1o2}
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end

abstract type AbstractPriorResult <: AbstractResult end
abstract type AbstractLowOrderPriorResult <: AbstractPriorResult end
# 0 = no asset views, 1 = asset views 
# 0 = no factor, 1 = factor + chol, 2 = factor
# 0 = no factor views, 1 = factor views
# Asset
abstract type AbstractPriorResult_A <: AbstractLowOrderPriorResult end
# Asset + factor + chol
abstract type AbstractPriorResult_AFC <: AbstractLowOrderPriorResult end
# # Asset + asset views
# abstract type AbstractPriorResult_AV <: AbstractLowOrderPriorResult end
# # Asset + factor + factor views + chol
# abstract type AbstractPriorResult_AFVC <: AbstractLowOrderPriorResult end
# Partial factor
abstract type AbstractPriorResult_PF <: AbstractLowOrderPriorResult end
# Asset + partial factor
abstract type AbstractPriorResult_APF <: AbstractLowOrderPriorResult end
# Asset + asset views + factor + factor views 
# abstract type AbstractPriorResult_AVFV <: AbstractLowOrderPriorResult end
abstract type AbstractEntropyPoolingPriorResult <: AbstractLowOrderPriorResult end
abstract type AbstractHighOrderPriorResult <: AbstractPriorResult end

function prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    return prior(pr, rd.X, rd.F; kwargs...)
end
function prior_view(pr::AbstractPriorEstimator, args...; kwargs...)
    return pr
end
function prior(pr::AbstractPriorResult, args...; kwargs...)
    return pr
end
function factory(::Nothing, args...; kwargs...)
    return nothing
end

export prior
