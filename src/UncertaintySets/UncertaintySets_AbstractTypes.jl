abstract type UncertaintySetEstimator end
abstract type UncertaintySetClass end
abstract type UncertaintySet end
abstract type UncertaintyKMethod end
function ucs end
function mu_uncertainty_set end
function sigma_uncertainty_set end
ucs(::Nothing, args...; kwargs...) = nothing
mu_uncertainty_set(::Nothing, args...; kwargs...) = nothing
sigma_uncertainty_set(::Nothing, args...; kwargs...) = nothing

export ucs, mu_uncertainty_set, sigma_uncertainty_set
