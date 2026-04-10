const Sd_Var = Union{<:StandardDeviation, <:Variance}
"""
$(DocStringExtensions.TYPEDEF)

Result type returned by [`SchurComplementHierarchicalRiskParity`](@ref) optimisation.

Stores the optimisation estimator, prior result, weight bounds, clustering result, Schur complement scaling parameter, return code, optimised weights, and optional fallback estimator.

# Fields

  - `oe`: Optimisation estimator used.
  - `pr`: Prior result containing asset moments.
  - `wb`: Weight bounds estimator.
  - `clr`: Clustering result.
  - `gamma`: Schur complement scaling parameter.
  - `retcode`: Optimisation return code.
  - `w`: Optimised asset weights.
  - `fb`: Optional fallback optimisation estimator.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct SchurComplementHierarchicalRiskParityResult <:
                 NonFiniteAllocationOptimisationResult
    oe
    pr
    wb
    clr
    gamma
    retcode
    w
    fb
end
function factory(res::SchurComplementHierarchicalRiskParityResult, fb::Option{<:OptE_Opt})
    return SchurComplementHierarchicalRiskParityResult(res.oe, res.pr, res.wb, res.clr,
                                                       res.gamma, res.retcode, res.w, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for Schur Complement algorithm variants.

# Related Types

  - [`NonMonotonicSchurComplement`](@ref)
  - [`MonotonicSchurComplement`](@ref)
"""
abstract type SchurComplementAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Non-monotonic Schur Complement algorithm variant for SCHRP.

Uses the raw Schur complement formula without monotonicity correction.

# Related Types

  - [`SchurComplementAlgorithm`](@ref)
  - [`MonotonicSchurComplement`](@ref)
"""
struct NonMonotonicSchurComplement <: SchurComplementAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Monotonic Schur Complement algorithm variant for SCHRP.

Applies a bisection-based correction to ensure the Schur complement allocation factor ``\\gamma`` is monotonically increasing with cluster risk, controlled by convergence tolerance `tol` and maximum iterations `iter`.

# Fields

  - `N`: Number of bisection steps.
  - `tol`: Convergence tolerance.
  - `iter`: Maximum iterations (or `nothing` for no limit).
  - `strict`: If `true`, raises an error if convergence is not achieved.

# Constructors

    MonotonicSchurComplement(;
        N::Integer = 10,
        tol::Number = 1e-4,
        iter::Option{<:Integer} = nothing,
        strict::Bool = false
    ) -> MonotonicSchurComplement

Keywords correspond to the struct's fields.

# Related Types

  - [`SchurComplementAlgorithm`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
"""
@concrete struct MonotonicSchurComplement <: SchurComplementAlgorithm
    N
    tol
    iter
    strict
    function MonotonicSchurComplement(N::Integer, tol::Number, iter::Option{<:Integer},
                                      strict::Bool)
        @argcheck(N > 0)
        @argcheck(tol > 0)
        if !isnothing(iter)
            @argcheck(iter > 0)
        end
        return new{typeof(N), typeof(tol), typeof(iter), typeof(strict)}(N, tol, iter,
                                                                         strict)
    end
end
function MonotonicSchurComplement(; N::Integer = 10, tol::Number = 1e-4,
                                  iter::Option{<:Integer} = nothing, strict::Bool = false)
    return MonotonicSchurComplement(N, tol, iter, strict)
end
"""
$(DocStringExtensions.TYPEDEF)

Parameters for the Schur Complement step of SCHRP.

`SchurComplementParams` collects the risk measure, initial allocation factor ``\\gamma``, positive-definite matrix correction, and monotonicity algorithm used in the Schur Complement Hierarchical Risk Parity optimisation.

# Fields

  - `r`: Risk measure (`Variance` or `StandardDeviation`) used for intra-cluster risk.
  - `gamma`: Initial allocation factor (``0 \\leq \\gamma \\leq 1``).
  - `pdm`: Positive definite matrix correction method.
  - `alg`: Schur Complement algorithm variant.
  - `flag`: If `true`, uses the Schur Complement inversion; if `false`, uses the direct Schur inverse.

# Constructors

    SchurComplementParams(;
        r::Sd_Var = Variance(),
        gamma::Number = 0.5,
        pdm::Option{<:Posdef} = Posdef(),
        alg::SchurComplementAlgorithm = MonotonicSchurComplement(),
        flag::Bool = true
    ) -> SchurComplementParams

Keywords correspond to the struct's fields.

## Validation

  - `0 <= gamma <= 1`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementAlgorithm`](@ref)
"""
@concrete struct SchurComplementParams <: AbstractAlgorithm
    r
    gamma
    pdm
    alg
    flag
    function SchurComplementParams(r::Sd_Var, gamma::Number, pdm::Option{<:Posdef},
                                   alg::SchurComplementAlgorithm, flag::Bool)
        @argcheck(one(gamma) >= gamma >= zero(gamma))
        return new{typeof(r), typeof(gamma), typeof(pdm), typeof(alg), typeof(flag)}(r,
                                                                                     gamma,
                                                                                     pdm,
                                                                                     alg,
                                                                                     flag)
    end
end
function SchurComplementParams(; r::Sd_Var = Variance(), gamma::Number = 0.5,
                               pdm::Option{<:Posdef} = Posdef(),
                               alg::SchurComplementAlgorithm = MonotonicSchurComplement(),
                               flag::Bool = true)
    return SchurComplementParams(r, gamma, pdm, alg, flag)
end
const VecScP = AbstractVector{<:SchurComplementParams}
const ScP_VecScP = Union{<:SchurComplementParams, <:VecScP}
function schur_complement_params_view(sp::SchurComplementParams, i, X::MatNum)
    r = risk_measure_view(sp.r, i, X)
    return SchurComplementParams(; r = r, gamma = sp.gamma, pdm = sp.pdm, alg = sp.alg,
                                 flag = sp.flag)
end
"""
$(DocStringExtensions.TYPEDEF)

Schur Complement Hierarchical Risk Parity (SCHRP) portfolio optimiser.

`SchurComplementHierarchicalRiskParity` extends HRP by using the Schur complement of the covariance matrix to more accurately decompose inter-cluster risk when allocating portfolio weights across the dendrogram.

# Fields

  - `opt`: Base hierarchical optimiser configuration.
  - `params`: Schur Complement parameters (single or vector).
  - `fb`: Fallback optimiser.

# Constructors

    SchurComplementHierarchicalRiskParity(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        params::ScP_VecScP = SchurComplementParams(),
        fb::Option{<:OptE_Opt} = nothing
    ) -> SchurComplementHierarchicalRiskParity

Keywords correspond to the struct's fields.

## Validation

  - If `params` is a vector: `!isempty(params)`.

# Examples

```jldoctest
julia> SchurComplementHierarchicalRiskParity()
SchurComplementHierarchicalRiskParity
  opt ┼ HierarchicalOptimiser
  params ┼ SchurComplementParams
  fb ┴ nothing
```

# Related

  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementParams`](@ref)
"""
@concrete struct SchurComplementHierarchicalRiskParity <: ClusteringOptimisationEstimator
    opt
    params
    fb
    function SchurComplementHierarchicalRiskParity(opt::HierarchicalOptimiser,
                                                   params::ScP_VecScP,
                                                   fb::Option{<:OptE_Opt})
        if isa(params, AbstractVector)
            @argcheck(!isempty(params))
        end
        return new{typeof(opt), typeof(params), typeof(fb)}(opt, params, fb)
    end
end
function SchurComplementHierarchicalRiskParity(;
                                               opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                               params::ScP_VecScP = SchurComplementParams(),
                                               fb::Option{<:OptE_Opt} = nothing)
    return SchurComplementHierarchicalRiskParity(opt, params, fb)
end
function needs_previous_weights(opt::SchurComplementHierarchicalRiskParity)
    return (needs_previous_weights(opt.opt) || needs_previous_weights(opt.fb))
end
function factory(sh::SchurComplementHierarchicalRiskParity, w::AbstractVector)
    opt = factory(sh.opt, w)
    fb = factory(sh.fb, w)
    return SchurComplementHierarchicalRiskParity(; opt = opt, params = sh.params, fb = fb)
end
function opt_view(sh::SchurComplementHierarchicalRiskParity, i, X::MatNum)
    X = isa(sh.opt.pe, AbstractPriorResult) ? sh.opt.pe.X : X
    opt = opt_view(sh.opt, i)
    params = schur_complement_params_view(sh.params, i, X)
    return SchurComplementHierarchicalRiskParity(; opt = opt, params = params, fb = sh.fb)
end
function symmetric_step_up_matrix(n1::Integer, n2::Integer)
    @argcheck(abs(n1 - n2) <= 1)

    if n1 == n2
        return LinearAlgebra.I(n1)
    elseif n1 < n2
        return transpose(symmetric_step_up_matrix(n2, n1)) * n1 / n2
    end

    m = zeros(n1, n2)
    row = fill(inv(n2), n2)
    e = LinearAlgebra.I(n2)
    for i in axes(m, 1)
        mj = vcat(e[1:(i - 1), :], row', e[i:end, :])
        m .+= mj / n1
    end
    return m
end
function schur_augmentation(A::MatNum, B::MatNum, C::MatNum, gamma::Number)
    Na = size(A, 1)
    Nc = size(C, 1)
    if iszero(gamma) || isone(Na) || isone(Nc)
        return A
    end
    A_aug = A - gamma * B * (C \ transpose(B))
    m = symmetric_step_up_matrix(Na, Nc)
    r = LinearAlgebra.I - gamma * transpose(transpose(C) \ transpose(B)) * transpose(m)
    A_aug = r \ A_aug
    return (A_aug + transpose(A_aug)) / 2
end
function naive_portfolio_risk(::Variance, sigma::MatNum)
    w = inv.(LinearAlgebra.diag(sigma))
    w ./= sum(w)
    return LinearAlgebra.dot(w, sigma, w)
end
function naive_portfolio_risk(::StandardDeviation, sigma::MatNum)
    w = inv.(LinearAlgebra.diag(sigma))
    w ./= sum(w)
    return sqrt(LinearAlgebra.dot(w, sigma, w))
end
function schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt,
                                  wb::WeightBounds,
                                  params::SchurComplementParams{<:Any, <:Any, <:Any,
                                                                <:NonMonotonicSchurComplement,
                                                                <:Any},
                                  gamma::Option{<:Number} = nothing)
    r = factory(params.r, pr)
    sigma = ismutable(r.sigma) ? copy(r.sigma) : Matrix(r.sigma)
    gamma = isnothing(gamma) ? params.gamma : gamma
    X = pr.X
    w = ones(eltype(X), size(X, 2))
    pdm = params.pdm
    flag = params.flag
    while length(items) > 0
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
                A_aug = schur_augmentation(A, B, C, gamma)
                C_aug = schur_augmentation(C, transpose(B), A, gamma)
                sigma[lc, lc] = A_aug
                sigma[rc, rc] = C_aug
            end
            if flag
                try
                    posdef!(pdm, A_aug)
                    posdef!(pdm, C_aug)
                catch e
                    throw(ArgumentError("Augmented matrix could not be made positive definite. Use `MonotonicSchurComplement()` or reduce gamma: $gamma."))
                end
            else
                if !LinearAlgebra.isposdef(A_aug) || !LinearAlgebra.isposdef(C_aug)
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
function schur_complement_binary_search(objective::Function, lgamma::Number, hgamma::Number,
                                        lrisk::Number, tol::Number = 1e-4,
                                        iter::Option{<:Integer} = nothing,
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
    strict ? throw(ArgumentError(msg)) : @warn(msg)
    return w, lgamma
end
function schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt,
                                  wb::WeightBounds,
                                  params::SchurComplementParams{<:Any, <:Any, <:Any,
                                                                <:MonotonicSchurComplement,
                                                                <:Any})
    max_gamma = params.gamma
    r = factory(params.r, pr)
    X = pr.X
    if iszero(max_gamma)
        nm_params = SchurComplementParams(; r = r, gamma = max_gamma, pdm = params.pdm,
                                          alg = NonMonotonicSchurComplement(),
                                          flag = params.flag)
        return schur_complement_weights(pr, items, wb, nm_params)
    end
    nm_params = SchurComplementParams(; r = r, gamma = max_gamma, pdm = params.pdm,
                                      alg = NonMonotonicSchurComplement(), flag = false)
    function objective(x::Number)
        w = schur_complement_weights(pr, items, wb, nm_params, x)[1]
        risk = isnothing(w) ? typemax(eltype(X)) : LinearAlgebra.dot(w, r.sigma, w)
        return w, risk
    end
    gammas = range(zero(max_gamma), max_gamma; length = params.alg.N)
    risks = fill(typemax(eltype(X)), size(gammas))
    w, risk = objective(gammas[1])
    risks[1] = risk
    # First binary search, finds the point at which the risk starts to increase with gamma, if it exists.
    for i in 2:length(gammas)
        w, risk = objective(gammas[i])
        risks[i] = risk
        if risk >= risks[i - 1]
            # Turning point is strictly between [gammas[i-2], gammas[i]].
            lidx = max(1, i - 2)
            return schur_complement_binary_search(objective, gammas[lidx], gammas[i],
                                                  risks[lidx], params.alg.tol,
                                                  params.alg.iter, params.alg.strict)
        end
    end
    # If there's no turning point in the range of gammas, check the derivative at the last gamma.
    if risk <= objective(max_gamma - params.alg.tol)[2]
        return w, max_gamma
    end
    # If the turning point exists and was not found within the range, or the last gamma, it is between the last two gammas.
    return schur_complement_binary_search(objective, gammas[end - 1], gammas[end],
                                          risks[end - 1], params.alg.tol, params.alg.iter,
                                          params.alg.strict)
end
function _optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    rd = returns_result_picker(rd, sh.opt.brt)
    pr = prior(sh.opt.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(sh.opt.cle, pr; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     cle_pr = sh.opt.cle_pr)
    items = [clr.res.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(X, 2),
                                   strict = sh.opt.strict, datatype = eltype(X))
    w, gamma = schur_complement_weights(pr, items, wb, sh.params)
    retcode, w = finalise_weight_bounds(sh.opt.wf, wb, w)
    return SchurComplementHierarchicalRiskParityResult(typeof(sh), pr, wb, clr, gamma,
                                                       retcode, w, nothing)
end
function _optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:AbstractVector},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    rd = returns_result_picker(rd, sh.opt.brt)
    pr = prior(sh.opt.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(sh.opt.cle, pr; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     cle_pr = sh.opt.cle_pr)
    items = [clr.res.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(X, 2),
                                   strict = sh.opt.strict, datatype = eltype(X))
    params = sh.params
    gammas = Vector{eltype(X)}(undef, length(params))
    w = zeros(eltype(X), size(X, 2))
    for (i, ps) in enumerate(params)
        wi, gamma = schur_complement_weights(pr, items, wb, ps)
        w .+= ps.r.settings.scale * wi
        gammas[i] = gamma
    end
    retcode, w = finalise_weight_bounds(sh.opt.wf, wb, w / sum(w))
    return SchurComplementHierarchicalRiskParityResult(typeof(sh), pr, wb, clr, gammas,
                                                       retcode, w, nothing)
end
function optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(sh, rd; dims = dims, kwargs...)
end

export SchurComplementHierarchicalRiskParityResult, SchurComplementParams,
       SchurComplementHierarchicalRiskParity, NonMonotonicSchurComplement,
       MonotonicSchurComplement
