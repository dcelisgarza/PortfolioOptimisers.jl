struct DeltaUncertaintySet{T1 <: AbstractPriorEstimator, T2 <: Real, T3 <: Real} <:
       AbstractUncertaintySetEstimator
    pe::T1
    dmu::T2
    dsigma::T3
end
function DeltaUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                             dmu::Real = 0.1, dsigma::Real = 0.1)
    return DeltaUncertaintySet{typeof(pe), typeof(dmu), typeof(dsigma)}(pe, dmu, dsigma)
end
function ucs(ue::DeltaUncertaintySet, X::AbstractMatrix, args...; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, args...; dims = dims, kwargs...)
    d_sigma = ue.dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = range(; start = 0, stop = 0, length = length(pr.mu)),
                             ub = ue.dmu * abs.(pr.mu) * 2),
           BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma)
end
function mu_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix, args...; dims::Int = 1,
                kwargs...)
    pr = prior(ue.pe, X, args...; dims = dims, kwargs...)
    return BoxUncertaintySet(; lb = range(; start = 0, stop = 0, length = length(pr.mu)),
                             ub = ue.dmu * abs.(pr.mu) * 2)
end
function sigma_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix, args...; dims::Int = 1,
                   kwargs...)
    pr = prior(ue.pe, X, args...; dims = dims, kwargs...)
    d_sigma = ue.dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma)
end

export DeltaUncertaintySet
