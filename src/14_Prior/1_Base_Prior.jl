abstract type AbstractPriorResult <: AbstractResult end
abstract type AbstractPriorEstimator <: AbstractEstimator end
# 0 = no asset views, 1 = asset views 
# 0 = no factor, 1 = factor + chol, 2 = factor
# 0 = no factor views, 1 = factor views
abstract type AbstractLowOrderPriorResult <: AbstractPriorResult end
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
abstract type AbstractHighOrderPriorResult <: AbstractPriorResult end
abstract type AbstractEntropyPoolingPriorResult <: AbstractPriorResult end
abstract type AbstractPriorEstimator_1_0 <: AbstractPriorEstimator end
abstract type AbstractPriorEstimator_2_1 <: AbstractPriorEstimator end
abstract type AbstractPriorEstimator_2_2 <: AbstractPriorEstimator end
abstract type AbstractPriorEstimator_1o2_1o2 <: AbstractPriorEstimator end

const AbstractPriorEstimatorMap_2_1 = Union{<:AbstractPriorEstimator_1_0,
                                            <:AbstractPriorEstimator_1o2_1o2}
const AbstractPriorEstimatorMap_2_2 = Union{<:AbstractPriorEstimator_2_1,
                                            <:AbstractPriorEstimator_2_2,
                                            <:AbstractPriorEstimator_1o2_1o2}
const AbstractPriorEstimatorMap_1o2_1o2 = Union{<:AbstractPriorEstimator_1_0,
                                                <:AbstractPriorEstimator_2_1,
                                                <:AbstractPriorEstimator_2_2,
                                                <:AbstractPriorEstimator_1o2_1o2}
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
