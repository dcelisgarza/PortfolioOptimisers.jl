struct DeltaUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetEstimator
    pe::T1
    dmu::T2
    dsigma::T3
    function DeltaUncertaintySet(pe::AbstractPriorEstimator, dmu::Real, dsigma::Real)
        @argcheck(dmu >= 0.0)
        @argcheck(dsigma >= 0.0)
        return new{typeof(pe), typeof(dmu), typeof(dsigma)}(pe, dmu, dsigma)
    end
end
function DeltaUncertaintySet(; pe::AbstractPriorEstimator = EmpiricalPrior(),
                             dmu::Real = 0.1, dsigma::Real = 0.1)
    return DeltaUncertaintySet(pe, dmu, dsigma)
end
function ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    d_sigma = ue.dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = range(; start = 0, stop = 0, length = length(pr.mu)),
                             ub = ue.dmu * abs.(pr.mu) * 2),
           BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma)
end
function mu_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    return BoxUncertaintySet(; lb = range(; start = 0, stop = 0, length = length(pr.mu)),
                             ub = ue.dmu * abs.(pr.mu) * 2)
end
function sigma_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    d_sigma = ue.dsigma * abs.(pr.sigma)
    return BoxUncertaintySet(; lb = pr.sigma - d_sigma, ub = pr.sigma + d_sigma)
end

export DeltaUncertaintySet
