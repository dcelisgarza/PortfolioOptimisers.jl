abstract type UncertaintySetEstimator end
abstract type UncertaintySetClass end
abstract type UncertaintySet end
abstract type UncertaintyKMethod end
function uncertainty_set end
function mu_uncertainty_set end
function sigma_uncertainty_set end
uncertainty_set(::Nothing, args...; kwargs...) = nothing
mu_uncertainty_set(::Nothing, args...; kwargs...) = nothing
sigma_uncertainty_set(::Nothing, args...; kwargs...) = nothing

export uncertainty_set, mu_uncertainty_set, sigma_uncertainty_set
