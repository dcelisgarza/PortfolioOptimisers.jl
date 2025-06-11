struct SchurHierarchicalOptimisationResult{T1 <: Type, T2 <: AbstractPriorResult,
                                           T3 <: Union{Nothing, <:WeightBoundsResult},
                                           T4 <: AbstractClusteringResult, T5 <: Real,
                                           T6 <: OptimisationReturnCode,
                                           T7 <: AbstractVector} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    clr::T4
    gamma::T5
    retcode::T6
    w::T7
end
struct SchurParams{T1 <: Union{<:StandardDeviation, <:Variance}, T2 <: Real,
                   T3 <: Union{Nothing, <:Real}, T4 <: Union{Nothing, <:Real},
                   T5 <: Bool} <: AbstractAlgorithm
    r::T1
    gamma::T2
    N::T3
    tol::T4
    flag::T5
end
function SchurParams(; r::Union{<:StandardDeviation, <:Variance} = Variance(),
                     gamma::Real = 0.5, N::Union{Nothing, <:Real} = nothing,
                     tol::Union{Nothing, <:Real} = nothing, flag::Bool = true)
    @smart_assert(gamma >= zero(gamma))
    return SchurParams{typeof(r), typeof(gamma), typeof(N), typeof(tol), typeof(flag)}(r,
                                                                                       gamma,
                                                                                       N,
                                                                                       tol,
                                                                                       flag)
end
function schur_params_view(sp::SchurParams, i::AbstractVector, X::AbstractMatrix)
    r = risk_measure_view(sp.r, i, X)
    return SchurParams(; r = r, gamma = sp.gamma, N = sp.N, tol = sp.tol, flag = sp.flag)
end
struct SchurHierarchicalRiskParity{T1 <: HierarchicalOptimiser,
                                   T2 <:
                                   Union{<:SchurParams, <:AbstractVector{<:SchurParams}}} <:
       ClusteringOptimisationEstimator
    opt::T1
    params::T2
end
function SchurHierarchicalRiskParity(; opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                     params::Union{<:SchurParams,
                                                   <:AbstractVector{<:SchurParams}} = SchurParams())
    if isa(params, AbstractVector)
        @smart_assert(!isempty(params))
    end
    return SchurHierarchicalRiskParity{typeof(opt), typeof(params)}(opt, params)
end
function opt_view(shc::SchurHierarchicalRiskParity, i::AbstractVector, X::AbstractMatrix)
    opt = opt_view(shc.opt, i)
    params = schur_params_view(shc.params, i, X)
    return SchurHierarchicalRiskParity(; opt = opt, params = params)
end
function symmetric_step_up_matrix(n1::Integer, n2::Integer)
    @smart_assert(abs(n1 - n2) <= 1)
    if n1 == n2
        return I(n1)
    elseif n1 < n2
        return transpose(symmetric_step_up_matrix(n2, n1)) * n1 / n2
    end

    m = zeros(n1, n2)
    e = vcat(ones(1, n2) / n2, I(n2))
    m .+= e
    for i ∈ 1:(n1 - 1)
        e[i, :] .= view(e, i + 1, :)
        e[i + 1, :] .= view(e, i, :)
        m .+= e
    end
    return m / n1
end
function schur_augmentation(A::AbstractMatrix, B::AbstractMatrix, D::AbstractMatrix,
                            gamma::Real)
    Na = size(A, 1)
    Nd = size(D, 1)
    if iszero(gamma) || isone(Na) || isone(Nd)
        return A
    end

    A_aug = A - gamma * B * (D \ transpose(B))
    m = symmetric_step_up_matrix(Na, Nd)
    r = I - gamma * (transpose(B) \ transpose(D)) * transpose(m)
    A_aug = r \ A_aug
    A_aug = (A_aug + transpose(A_aug)) / 2

    return A_aug
end
function naive_portfolio_risk(::Variance, sigma::AbstractMatrix)
    w = inv.(diag(sigma))
    w ./= sum(w)
    return dot(w, sigma, w)
end
function naive_portfolio_risk(::StandardDeviation, sigma::AbstractMatrix)
    w = inv.(diag(sigma))
    w ./= sum(w)
    return sqrt(dot(w, sigma, w))
end
function schur_weights(pr::AbstractPriorResult, r::Union{<:Variance, <:StandardDeviation},
                       items::AbstractVector, wb::WeightBoundsResult, params::SchurParams,
                       flag::Bool = true, gamma::Union{Nothing, <:Real} = nothing)
    sigma = copy(r.sigma)
    gamma = isnothing(gamma) ? params.gamma : gamma
    w = ones(eltype(pr.X), size(pr.X, 2))
    @inbounds while length(items) > 0
        items = [i[j:k] for i ∈ items
                 for (j, k) ∈ ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i ∈ 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            A = view(sigma, lc, lc)
            C = view(sigma, rc, rc)
            A_aug, C_aug = if length(lc) <= 1
                A, C
            else
                B = view(sigma, lc, rc)
                schur_augmentation(A, B, C, gamma),
                schur_augmentation(C, transpose(B), A, gamma)
            end
            if flag
                if !isposdef(A_aug) || !isposdef(C_aug)
                    return nothing
                end
            end
            sigma[lc, lc] .= A_aug
            sigma[rc, rc] .= C_aug
            lrisk = naive_portfolio_risk(r, A_aug)
            rrisk = naive_portfolio_risk(r, C_aug)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    return w
end
function schur_binary_search(objective::Function, lgamma::Real, hgamma::Real, lrisk::Real,
                             tol::Real = 1e-4)
    iter = ceil(Int, log2((hgamma - lgamma) / tol) * 2 + 1)
    for _ ∈ 1:iter
        mgamma = (lgamma + hgamma) * 0.5
        w, risk = objective(mgamma)
        hrisk = objective(mgamma - tol)[2]
        if risk <= lrisk && risk <= hrisk
            # If risk at midpoint is lower than at the lower bound and lower than the risk just below the midpoint, we can update the lower bound to the midpoint.
            lgamma = mgamma
            lrisk = risk
            if (hgamma - lgamma) <= tol
                # Return if the difference between upper and lower bounds is within the tolerance.
                return w, lgamma
            end
        else
            # Else we update the upper bound to the midpoint.
            hgamma = mgamma
        end
    end
    throw(ArgumentError("Binary search did not converge within the specified tolerance."))
end
#! Refactor as shur_monotonic by creating a variable inside. monotonic_params = SchurParams(r=r, gamma=gamma, alg=SchurNonMonotonic()). Keep tol and N in SchurMonotonic()
function monotonic_schur_weights(pr::AbstractPriorResult,
                                 r::Union{<:Variance, <:StandardDeviation},
                                 items::AbstractVector, wb::WeightBoundsResult,
                                 params::SchurParams)
    if iszero(params.gamma)
        return schur_weights(pr, r, items, wb, params, false), params.gamma
    end
    function objective(x::Real)
        w = schur_weights(pr, r, items, wb, params, true, x)
        risk = isnothing(w) ? typemax(eltype(pr.X)) : dot(w, r.sigma, w)
        return w, risk
    end
    tol = isnothing(params.tol) ? 1e-4 : params.tol
    max_gamma = params.gamma
    gammas = range(; start = zero(max_gamma), stop = max_gamma,
                   length = isnothing(params.N) ? 10 : params.N)
    risks = fill(typemax(eltype(pr.X)), size(gammas))
    w, risk = objective(gammas[1])
    risks[1] = risk
    # First binary search, finds the point at which the risk starts to increase with gamma, if it exists.
    for i ∈ 2:length(gammas)
        w, risk = objective(gammas[i])
        risks[i] = risk
        if risk >= risks[i - 1]
            # Turning point is strictly between [gammas[i-2], gammas[i]].
            lidx = max(1, i - 2)
            return schur_binary_search(objective, gammas[lidx], gammas[i], risks[lidx], tol)
        end
    end
    # If there's no turning point in the range of gammas, check the derivative at the last gamma.
    if risk <= objective(max_gamma - tol)[2]
        return w, max_gamma
    end
    # If the turning point exists and was not found within the range, or the last gamma, it is between the last two gammas.
    return schur_binary_search(objective, gammas[end - 1], gammas[end], risks[end - 1], tol)
end
function optimise!(shc::SchurHierarchicalRiskParity, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, kwargs...)
    pr = prior(shc.opt.pe, rd.X, rd.F; dims = dims)
    clr = clusterise(shc.opt.cle, pr.X; dims = dims)
    items = [clr.clustering.order]
    wb = weight_bounds_constraints(shc.opt.wb, shc.opt.sets; N = size(pr.X, 2),
                                   strict = shc.opt.strict)
    r = factory(params.r, pr)
    w, gamma = if shc.params.flag
        monotonic_schur_weights(pr, r, items, wb, params)
    else
        schur_weights(pr, r, items, wb, params)
    end
    retcode, w = clustering_optimisation_result(shc.opt.cwf, wb, w)
    return SchurHierarchicalOptimisationResult(typeof(SchurHierarchicalRiskParity), pr, wb,
                                               clr, gamma, retcode, w)
end

export SchurHierarchicalOptimisationResult, SchurParams, SchurHierarchicalRiskParity