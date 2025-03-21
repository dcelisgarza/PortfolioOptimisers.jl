abstract type UncertaintySetEstimator end
abstract type UncertaintySetClass end
abstract type UncertaintySet end
struct NoUncertaintySetEstimator <: UncertaintySetEstimator end
struct NoUncertaintySet <: UncertaintySet end
abstract type UncertaintyKMethod end
function uncertainty_set end
function mu_uncertainty_set end
function sigma_uncertainty_set end
uncertainty_set(::NoUncertaintySetEstimator, args...; kwargs...) = NoUncertaintySet()
mu_uncertainty_set(::NoUncertaintySetEstimator, args...; kwargs...) = NoUncertaintySet()
sigma_uncertainty_set(::NoUncertaintySetEstimator, args...; kwargs...) = NoUncertaintySet()

export uncertainty_set, mu_uncertainty_set, sigma_uncertainty_set,
       NoUncertaintySetEstimator, NoUncertaintySet
