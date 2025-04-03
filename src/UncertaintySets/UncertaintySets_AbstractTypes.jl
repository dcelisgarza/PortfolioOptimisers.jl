abstract type UncertaintySetEstimator end
abstract type UncertaintySetClass end
abstract type UncertaintySet end
abstract type UncertaintyKMethod end
function ucs end
function mu_ucs end
function sigma_ucs end
ucs(::Nothing, args...; kwargs...) = nothing
mu_ucs(::Nothing, args...; kwargs...) = nothing
sigma_ucs(::Nothing, args...; kwargs...) = nothing

ucs(uc::UncertaintySet, args...; kwargs...) = uc
mu_ucs(uc::UncertaintySet, args...; kwargs...) = uc
sigma_ucs(uc::UncertaintySet, args...; kwargs...) = uc

export ucs, mu_ucs, sigma_ucs
