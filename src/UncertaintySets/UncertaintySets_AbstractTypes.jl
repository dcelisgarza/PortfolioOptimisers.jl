abstract type UncertaintySetEstimator end
abstract type UncertaintySetClass end
abstract type UncertaintySet end
struct NoUncertaintySet <: UncertaintySet end
abstract type UncertaintyKMethod end

function uncertainty_set end
function mu_uncertainty_set end
function sigma_uncertainty_set end

export uncertainty_set, mu_uncertainty_set, sigma_uncertainty_set, NoUncertaintySet
