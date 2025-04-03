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
function ucs(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    d_sigma = ue.delta_sigma * abs.(pm.sigma)
    return BoxUncertaintySet(; lb = range(; start = 0, stop = 0, length = length(pm.mu)),
                             ub = ue.delta_mu * abs.(pm.mu) * 2),
           BoxUncertaintySet(; lb = pm.sigma - d_sigma, ub = pm.sigma + d_sigma)
end
function mu_ucs(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    return BoxUncertaintySet(; lb = range(; start = 0, stop = 0, length = length(pm.mu)),
                             ub = ue.delta_mu * abs.(pm.mu) * 2)
end
function sigma_ucs(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...;
                   dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    d_sigma = ue.delta_sigma * abs.(pm.sigma)
    return BoxUncertaintySet(; lb = pm.sigma - d_sigma, ub = pm.sigma + d_sigma)
end

export DeltaUncertaintySetEstimator
