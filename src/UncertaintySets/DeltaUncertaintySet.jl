struct DeltaUncertaintySetEstimator{T1 <: AbstractPriorEstimator, T2 <: Real, T3 <: Real} <:
       UncertaintySetEstimator
    pe::T1
    delta_mu::T2
    delta_sigma::T3
end
function DeltaUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                      delta_mu::Real = 0.1, delta_sigma::Real = 0.1)
    return DeltaUncertaintySetEstimator{typeof(pe), typeof(delta_mu), typeof(delta_sigma)}(pe,
                                                                                           delta_mu,
                                                                                           delta_sigma)
end
function uncertainty_set(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...;
                         dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    (; mu, sigma) = pm
    d_sigma = ue.delta_sigma * abs.(sigma)
    return BoxUncertaintySet(; lo = range(; start = 0, stop = 0, length = length(mu)),
                             hi = ue.delta_mu * abs.(mu) * 2),
           BoxUncertaintySet(; lo = sigma - d_sigma, hi = sigma + d_sigma)
end
function mu_uncertainty_set(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...;
                            dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    mu = pm.mu
    return BoxUncertaintySet(; lo = range(; start = 0, stop = 0, length = length(mu)),
                             hi = ue.delta_mu * abs.(mu) * 2)
end
function sigma_uncertainty_set(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...;
                               dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    sigma = pm.sigma
    d_sigma = ue.delta_sigma * abs.(sigma)
    return BoxUncertaintySet(; lo = sigma - d_sigma, hi = sigma + d_sigma)
end

export DeltaUncertaintySetEstimator