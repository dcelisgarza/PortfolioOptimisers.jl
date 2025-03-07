struct BlackLittermanPriorEstimator{T1 <: PriorEstimator, T2 <: MatrixProcessing,
                                    T3 <: Union{<:LinearConstraintAtom,
                                                <:AbstractVector{<:LinearConstraintAtom}},
                                    T4 <: DataFrame, T5 <: Real,
                                    T6 <: Union{Nothing, <:AbstractVector},
                                    T7 <: Union{Nothing, <:Real}} <: PriorEstimator
    pe::T1
    mp::T2
    asset_views::T3
    asset_sets::T4
    rf::T5
    asset_views_conf::T6
    tau::T7
end
function BlackLittermanPriorEstimator(;
                                      pe::PriorEstimator = EmpiricalPriorEstimator(;
                                                                                   me = EquilibriumExpectedReturns()),
                                      mp::MatrixProcessing = DefaultMatrixProcessing(),
                                      asset_views::Union{<:LinearConstraintAtom,
                                                         <:AbstractVector{<:LinearConstraintAtom}} = LinearConstraintAtom(),
                                      asset_sets::DataFrame = DataFrame(), rf::Real = 0.0,
                                      asset_views_conf::Union{Nothing, <:AbstractVector} = nothing,
                                      tau::Union{Nothing, <:Real} = nothing)
    if !isnothing(asset_views_conf)
        @smart_assert(length(asset_views) == length(asset_views_conf))
        @smart_assert(all(zero(eltype(asset_views_conf)) .<
                          asset_views_conf .<=
                          one(eltype(asset_views_conf))))
    end
    return BlackLittermanPriorEstimator{typeof(pe), typeof(mp), typeof(asset_views),
                                        typeof(asset_sets), typeof(rf),
                                        typeof(asset_views_conf), typeof(tau)}(pe, mp,
                                                                               asset_views,
                                                                               asset_sets,
                                                                               rf,
                                                                               asset_views_conf,
                                                                               tau)
end
function prior(pe::BlackLittermanPriorEstimator, X::AbstractMatrix, args...; dims::Int = 1,
               strict::Bool = false)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    T, N = size(X)
    @smart_assert(nrow(pe.asset_sets) == N)

    prior_model = prior(pe.pe, X, args...)
    prior_X, prior_mu, prior_sigma = prior_model.X, prior_model.mu, prior_model.sigma

    P, Q = views_constraints(pe.asset_views, pe.asset_sets; datatype = eltype(prior_X),
                             strict = strict)
    @smart_assert(!isempty(P))

    asset_views_conf = pe.asset_views_conf
    tau = isnothing(pe.tau) ? inv(T) : pe.tau
    omega = tau * Diagonal(if isnothing(asset_views_conf)
                               P * prior_sigma * transpose(P)
                           else
                               idx = iszero.(asset_views_conf)
                               asset_views_conf[idx] .= eps(eltype(asset_views_conf))
                               alphas = inv.(asset_views_conf) .- 1
                               alphas .* P * prior_sigma * transpose(P)
                           end)

    v = tau * prior_sigma * transpose(P)
    a = P * v + omega
    b = Q .- P * prior_mu
    posterior_mu = prior_mu + v * (a \ b) .+ pe.rf
    posterior_sigma = prior_sigma + tau * prior_sigma - v * (a \ transpose(v))

    # a = (tau * prior_sigma) \ I
    # b = omega \ I
    # c = transpose(P) * b
    # M = (a + c * P) \ I
    # PI = M * (a * prior_mu + c * Q)
    # posterior_mu = PI .+ pe.rf
    # posterior_sigma = prior_sigma + M

    mtx_process!(pe.mp, posterior_sigma, prior_X)
    return Prior(; X = prior_X, mu = posterior_mu, sigma = posterior_sigma)
end

export BlackLittermanPriorEstimator
