struct BlackLittermanPriorModel{T1 <: AbstractMatrix, T2 <: AbstractVector,
                                T3 <: AbstractMatrix, T4 <: AbstractMatrix,
                                T5 <: AbstractVector} <: AbstractPriorModel_AV
    X::T1
    mu::T2
    sigma::T3
    P::T4
    Q::T5
end
function BlackLittermanPriorModel(; X::AbstractMatrix, mu::AbstractVector,
                                  sigma::AbstractMatrix, P::AbstractMatrix,
                                  Q::AbstractVector)
    if isa(X, AbstractMatrix)
        @smart_assert(!isempty(X))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
    end
    if isa(P, AbstractMatrix)
        @smart_assert(!isempty(P))
    end
    if isa(Q, AbstractVector)
        @smart_assert(!isempty(Q))
    end
    @smart_assert(size(X, 2) ==
                  length(mu) ==
                  size(sigma, 1) ==
                  size(sigma, 2) ==
                  size(P, 2))
    @smart_assert(length(Q) == size(P, 1))
    return BlackLittermanPriorModel{typeof(X), typeof(mu), typeof(sigma), typeof(P),
                                    typeof(Q)}(X, mu, sigma, P, Q)
end
struct BlackLittermanPriorEstimator{T1 <: AbstractPriorEstimatorMap_1o2_1o2,
                                    T2 <: MatrixProcessing,
                                    T3 <: Union{<:LinearConstraintAtom,
                                                <:AbstractVector{<:LinearConstraintAtom}},
                                    T4 <: DataFrame, T5 <: Real,
                                    T6 <: Union{Nothing, <:AbstractVector},
                                    T7 <: Union{Nothing, <:Real}} <:
       AbstractPriorEstimator_1o2_1o2
    pe::T1
    mp::T2
    views::T3
    sets::T4
    rf::T5
    views_conf::T6
    tau::T7
end
function Base.getproperty(obj::BlackLittermanPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function BlackLittermanPriorEstimator(;
                                      pe::AbstractPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(;
                                                                                                      me = EquilibriumExpectedReturns()),
                                      mp::MatrixProcessing = DefaultMatrixProcessing(),
                                      views::Union{<:LinearConstraintAtom,
                                                   <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                      sets::DataFrame = DataFrame(), rf::Real = 0.0,
                                      views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                      tau::Union{Nothing, <:Real} = nothing)
    if !isnothing(views_conf)
        @smart_assert(length(views) == length(views_conf))
        @smart_assert(all(zero(eltype(views_conf)) .< views_conf .< one(eltype(views_conf))))
    end
    if !isnothing(tau)
        @smart_assert(tau > zero(tau))
    end
    return BlackLittermanPriorEstimator{typeof(pe), typeof(mp), typeof(views), typeof(sets),
                                        typeof(rf), typeof(views_conf), typeof(tau)}(pe, mp,
                                                                                     views,
                                                                                     sets,
                                                                                     rf,
                                                                                     views_conf,
                                                                                     tau)
end
function prior(pe::BlackLittermanPriorEstimator, X::AbstractMatrix, args...; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    @smart_assert(nrow(pe.sets) == size(X, 2))

    prior_model = prior(pe.pe, X, args...)
    posterior_X, prior_mu, prior_sigma = prior_model.X, prior_model.mu, prior_model.sigma

    P, Q = views_constraints(pe.views, pe.sets; datatype = eltype(posterior_X),
                             strict = strict)
    @smart_assert(!isempty(P))

    tau = isnothing(pe.tau) ? inv(size(X, 1)) : pe.tau
    views_conf = pe.views_conf
    omega = tau * Diagonal(if isnothing(views_conf)
                               P * prior_sigma * transpose(P)
                           else
                               idx = iszero.(views_conf)
                               views_conf[idx] .= eps(eltype(views_conf))
                               alphas = inv.(views_conf) .- 1
                               alphas .* P * prior_sigma * transpose(P)
                           end)

    v1 = tau * prior_sigma * transpose(P)
    v2 = P * v1 + omega
    v3 = Q .- P * prior_mu
    posterior_mu = prior_mu + v1 * (v2 \ v3) .+ pe.rf
    posterior_sigma = prior_sigma + tau * prior_sigma - v1 * (v2 \ transpose(v1))
    mtx_process!(pe.mp, posterior_sigma, posterior_X)
    return BlackLittermanPriorModel(; X = posterior_X, mu = posterior_mu,
                                    sigma = posterior_sigma, P = P, Q = Q)
end

export BlackLittermanPriorModel, BlackLittermanPriorEstimator
