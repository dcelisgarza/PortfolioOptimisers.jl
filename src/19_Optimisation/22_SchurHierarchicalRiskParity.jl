struct SchurHierarchicalRiskParityOptimisation{T1, T2, T3, T4, T5, T6, T7} <:
       OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    clr::T4
    gamma::T5
    retcode::T6
    w::T7
end
abstract type SchurAlgorithm <: AbstractAlgorithm end
struct NonMonotonicSchur <: SchurAlgorithm end
struct MonotonicSchur{T1, T2, T3, T4} <: SchurAlgorithm
    N::T1
    tol::T2
    iter::T3
    strict::T4
end
function MonotonicSchur(; N::Integer = 10, tol::Real = 1e-4,
                        iter::Union{Nothing, Integer} = nothing, strict::Bool = false)
    @argcheck(N > 0)
    @argcheck(tol > 0)
    if !isnothing(iter)
        @argcheck(iter > 0)
    end
    return MonotonicSchur(N, tol, iter, strict)
end
struct SchurParams{T1, T2, T3, T4, T5} <: AbstractAlgorithm
    r::T1
    gamma::T2
    pdm::T3
    alg::T4
    flag::T5
end
function SchurParams(; r::Union{<:StandardDeviation, <:Variance} = Variance(),
                     gamma::Real = 0.5, pdm::Union{Nothing, <:Posdef} = Posdef(),
                     alg::SchurAlgorithm = MonotonicSchur(), flag::Bool = true)
    @argcheck(one(gamma) >= gamma >= zero(gamma))
    return SchurParams(r, gamma, pdm, alg, flag)
end
function schur_params_view(sp::SchurParams, i::AbstractVector, X::AbstractMatrix)
    r = risk_measure_view(sp.r, i, X)
    return SchurParams(; r = r, gamma = sp.gamma, pdm = sp.pdm, alg = sp.alg,
                       flag = sp.flag)
end
struct SchurHierarchicalRiskParity{T1, T2} <: ClusteringOptimisationEstimator
    opt::T1
    params::T2
end
function SchurHierarchicalRiskParity(; opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                     params::Union{<:SchurParams,
                                                   <:AbstractVector{<:SchurParams}} = SchurParams())
    if isa(params, AbstractVector)
        @argcheck(!isempty(params))
    end
    return SchurHierarchicalRiskParity(opt, params)
end
function opt_view(sh::SchurHierarchicalRiskParity, i::AbstractVector, X::AbstractMatrix)
    X = isa(sh.opt.pe, AbstractPriorResult) ? sh.opt.pe.X : X
    opt = opt_view(sh.opt, i)
    params = schur_params_view(sh.params, i, X)
    return SchurHierarchicalRiskParity(; opt = opt, params = params)
end
function symmetric_step_up_matrix(n1::Integer, n2::Integer)
    @argcheck(abs(n1 - n2) <= 1)
    if n1 == n2
        return I(n1)
    elseif n1 < n2
        return transpose(symmetric_step_up_matrix(n2, n1)) * n1 / n2
    end

    m = zeros(n1, n2)
    e = vcat(ones(1, n2) / n2, I(n2))
    m .+= e
    for i in 1:(n1 - 1)
        e[i, :] .= view(e, i + 1, :)
        e[i + 1, :] .= view(e, i, :)
        m .+= e
    end
    return m / n1
end
function schur_augmentation(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix,
                            gamma::Real)
    Na = size(A, 1)
    Nd = size(C, 1)
    if iszero(gamma) || isone(Na) || isone(Nd)
        return A
    end

    A_aug = A - gamma * B * (C \ transpose(B))
    m = symmetric_step_up_matrix(Na, Nd)
    r = I - gamma * (transpose(B) \ transpose(C)) * transpose(m)
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
function schur_weights(pr::AbstractPriorResult, items::AbstractVector, wb::WeightBounds,
                       params::SchurParams{<:Any, <:Any, <:Any, <:NonMonotonicSchur, <:Any},
                       gamma::Union{Nothing, <:Real} = nothing)
    r = factory(params.r, pr)
    sigma = ismutable(r.sigma) ? copy(r.sigma) : Matrix(r.sigma)
    gamma = isnothing(gamma) ? params.gamma : gamma
    w = ones(eltype(pr.X), size(pr.X, 2))
    pdm = params.pdm
    flag = params.flag
    @inbounds while length(items) > 0
        items = [i[j:k] for i in items
                 for (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i in 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            A = view(sigma, lc, lc)
            C = view(sigma, rc, rc)
            if length(lc) <= 1
                A_aug = A
                C_aug = C
            else
                B = view(sigma, lc, rc)
                sigma[lc, lc] = schur_augmentation(A, B, C, gamma)
                sigma[rc, rc] = schur_augmentation(C, transpose(B), A, gamma)
                A_aug = view(sigma, lc, lc)
                C_aug = view(sigma, rc, rc)
            end
            if flag
                try
                    posdef!(pdm, A_aug)
                    posdef!(pdm, C_aug)
                catch e
                    throw(ArgumentError("Augmented matrix could not be made positive definite. Use `MonotonicSchur()` or reduce gamma: $gamma."))
                end
            else
                if !isposdef(A_aug) || !isposdef(C_aug)
                    return nothing, nothing
                end
            end
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
    return w, gamma
end
function schur_binary_search(objective::Function, lgamma::Real, hgamma::Real, lrisk::Real,
                             tol::Real = 1e-4, iter::Union{Nothing, <:Integer} = nothing,
                             strict::Bool = false)
    w = nothing
    if isnothing(iter)
        iter = ceil(Int, log2((hgamma - lgamma) / tol) * 4 + 10)
    end
    for _ in 1:iter
        mgamma = (lgamma + hgamma) * 0.5
        w, risk, hrisk = objective(mgamma)..., objective(mgamma - tol)[2]
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
    msg = "Binary search did not converge within the specified tolerance: tol => $tol"
    if strict
        throw(ArgumentError(msg))
    else
        @warn(msg)
        return w, lgamma
    end
end
function schur_weights(pr::AbstractPriorResult, items::AbstractVector, wb::WeightBounds,
                       params::SchurParams{<:Any, <:Any, <:Any, <:MonotonicSchur, <:Any})
    max_gamma = params.gamma
    r = factory(params.r, pr)
    if iszero(max_gamma)
        nm_params = SchurParams(; r = r, gamma = max_gamma, pdm = params.pdm,
                                alg = NonMonotonicSchur(), flag = params.flag)
        return schur_weights(pr, items, wb, nm_params)
    end
    nm_params = SchurParams(; r = r, gamma = max_gamma, pdm = params.pdm,
                            alg = NonMonotonicSchur(), flag = false)
    function objective(x::Real)
        w = schur_weights(pr, items, wb, nm_params, x)[1]
        risk = isnothing(w) ? typemax(eltype(pr.X)) : dot(w, r.sigma, w)
        return w, risk
    end
    gammas = range(; start = zero(max_gamma), stop = max_gamma, length = params.alg.N)
    risks = fill(typemax(eltype(pr.X)), size(gammas))
    w, risk = objective(gammas[1])
    risks[1] = risk
    # First binary search, finds the point at which the risk starts to increase with gamma, if it exists.
    for i in 2:length(gammas)
        w, risk = objective(gammas[i])
        risks[i] = risk
        if risk >= risks[i - 1]
            # Turning point is strictly between [gammas[i-2], gammas[i]].
            lidx = max(1, i - 2)
            return schur_binary_search(objective, gammas[lidx], gammas[i], risks[lidx],
                                       params.alg.tol, params.alg.iter, params.alg.strict)
        end
    end
    # If there's no turning point in the range of gammas, check the derivative at the last gamma.
    if risk <= objective(max_gamma - params.alg.tol)[2]
        return w, max_gamma
    end
    # If the turning point exists and was not found within the range, or the last gamma, it is between the last two gammas.
    return schur_binary_search(objective, gammas[end - 1], gammas[end], risks[end - 1],
                               params.alg.tol, params.alg.iter, params.alg.strict)
end
function optimise!(sh::SchurHierarchicalRiskParity{<:Any, <:Any},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    pr = prior(sh.opt.pe, rd; dims = dims)
    clr = clusterise(sh.opt.cle, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    items = [clr.clustering.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(pr.X, 2),
                                   strict = sh.opt.strict, datatype = eltype(pr.X))
    w, gamma = schur_weights(pr, items, wb, sh.params)
    retcode, w = clustering_optimisation_result(sh.opt.cwf, wb, w)
    return SchurHierarchicalRiskParityOptimisation(typeof(sh), pr, wb, clr, gamma, retcode,
                                                   w)
end
function optimise!(sh::SchurHierarchicalRiskParity{<:Any, <:AbstractVector},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    pr = prior(sh.opt.pe, rd; dims = dims)
    clr = clusterise(sh.opt.cle, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    items = [clr.clustering.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(pr.X, 2),
                                   strict = sh.opt.strict, datatype = eltype(pr.X))
    params = sh.params
    gammas = Vector{eltype(pr.X)}(undef, length(params))
    w = zeros(eltype(pr.X), size(pr.X, 2))
    for (i, ps) in enumerate(params)
        wi, gamma = schur_weights(pr, items, wb, ps)
        w .+= ps.r.settings.scale * wi
        gammas[i] = gamma
    end
    retcode, w = clustering_optimisation_result(sh.opt.cwf, wb, w / sum(w))
    return SchurHierarchicalRiskParityOptimisation(typeof(sh), pr, wb, clr, gammas, retcode,
                                                   w)
end

export SchurHierarchicalRiskParityOptimisation, SchurParams, SchurHierarchicalRiskParity,
       NonMonotonicSchur, MonotonicSchur
