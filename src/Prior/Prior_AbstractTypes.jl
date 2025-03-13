abstract type AbstractPriorModel end
abstract type AbstractPriorEstimator end

# 0 = no asset views, 1 = asset views 
# 0 = no factor, 1 = factor + chol, 2 = factor
# 0 = no factor views, 1 = factor views

abstract type LowOrderAbstractPriorModel <: AbstractPriorModel end
# Asset
abstract type AbstractPriorModel_A <: LowOrderAbstractPriorModel end
# Asset + factor + chol
abstract type AbstractPriorModel_AFC <: LowOrderAbstractPriorModel end
# Asset + asset views
abstract type AbstractPriorModel_AV <: LowOrderAbstractPriorModel end
# Asset + factor + factor views + chol
abstract type AbstractPriorModel_AFVC <: LowOrderAbstractPriorModel end
# Asset + factor + factor views
abstract type AbstractPriorModel_AFV <: LowOrderAbstractPriorModel end
# Asset + asset views + factor + factor views 
abstract type AbstractPriorModel_AVFV <: LowOrderAbstractPriorModel end

abstract type AbstractHighOrderPriorModel <: AbstractPriorModel end

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

function prior end

export prior
