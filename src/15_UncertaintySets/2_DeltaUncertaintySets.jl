struct DeltaUncertaintySetEstimator{T1 <: AbstractPriorEstimator, T2 <: Real, T3 <: Real} <:
       AbstractUncertaintySetEstimator
    pe::T1
    dmu::T2
    dsigma::T3
end
function DeltaUncertaintySetEstimator(;
                                      pe::AbstractPriorEstimator = EmpiricalPriorEstimator(),
                                      dmu::Real = 0.1, dsigma::Real = 0.1)
    return DeltaUncertaintySetEstimator{typeof(pe), typeof(dmu), typeof(dsigma)}(pe, dmu,
                                                                                 dsigma)
end
function ucs(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    d_sigma = ue.dsigma * abs.(pm.sigma)
    return BoxUncertaintySetResult(;
                                   lb = range(; start = 0, stop = 0,
                                              length = length(pm.mu)),
                                   ub = ue.dmu * abs.(pm.mu) * 2),
           BoxUncertaintySetResult(; lb = pm.sigma - d_sigma, ub = pm.sigma + d_sigma)
end
function mu_ucs(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...; dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    return BoxUncertaintySetResult(;
                                   lb = range(; start = 0, stop = 0,
                                              length = length(pm.mu)),
                                   ub = ue.dmu * abs.(pm.mu) * 2)
end
function sigma_ucs(ue::DeltaUncertaintySetEstimator, X::AbstractMatrix, args...;
                   dims::Int = 1)
    pm = prior(ue.pe, X, args...; dims = dims)
    d_sigma = ue.dsigma * abs.(pm.sigma)
    return BoxUncertaintySetResult(; lb = pm.sigma - d_sigma, ub = pm.sigma + d_sigma)
end

export DeltaUncertaintySetEstimator
