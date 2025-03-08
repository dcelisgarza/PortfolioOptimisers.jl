abstract type AbstractPriorEstimator end
abstract type AbstractPriorModel end
abstract type AbstractBlackLittermanPriorEstimator <: AbstractPriorEstimator end
abstract type AbstractBlackLittermanPriorModel <: AbstractPriorModel end
function prior end

export prior
