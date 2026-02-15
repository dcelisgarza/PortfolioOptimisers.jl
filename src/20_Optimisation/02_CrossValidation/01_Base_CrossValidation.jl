abstract type CrossValidationEstimator <: AbstractEstimator end
abstract type CrossValidationResult <: AbstractResult end
abstract type CrossValidationAlgorithm <: AbstractAlgorithm end
abstract type SequentialCrossValidationEstimator <: CrossValidationEstimator end
abstract type NonSequentialCrossValidationEstimator <: CrossValidationEstimator end
abstract type SequentialCrossValidationResult <: CrossValidationResult end
abstract type NonSequentialCrossValidationResult <: CrossValidationResult end
