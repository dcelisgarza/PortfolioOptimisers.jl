"""
    const Sd_Var = Union{<:StandardDeviation, <:Variance}

Alias for a standard deviation or variance risk measure.

Used in the Schur Complement HRP to accept either risk measure type for computing naive portfolio risk.

# Related

  - [`StandardDeviation`](@ref)
  - [`Variance`](@ref)
  - [`SchurComplementParams`](@ref)
"""
const Sd_Var = Union{<:StandardDeviation, <:Variance}
"""
$(DocStringExtensions.TYPEDEF)

Result type returned by [`SchurComplementHierarchicalRiskParity`](@ref) optimisation.

Stores the optimisation estimator, prior result, weight bounds, clustering result, Schur complement scaling parameter, return code, optimised weights, and optional fallback estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct SchurComplementHierarchicalRiskParityResult <: NonJuMPOptimisationResult
    """
    $(field_dict[:oe])
    """
    oe
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:clr])
    """
    clr
    """
    $(field_dict[:gamma_schur])
    """
    gamma
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:pw])
    """
    w
    """
    $(field_dict[:fb])
    """
    fb
    function SchurComplementHierarchicalRiskParityResult(oe::Type{<:OptimisationEstimator},
                                                         pr::Option{<:AbstractPriorResult},
                                                         wb::Option{<:WeightBounds},
                                                         clr::Option{<:AbstractClusteringResult},
                                                         gamma::Union{<:Number, <:VecNum},
                                                         retcode::OptimisationReturnCode,
                                                         w::Option{<:VecNum},
                                                         fb::Option{<:OptE_Opt})
        return new{typeof(oe), typeof(pr), typeof(wb), typeof(clr), typeof(gamma),
                   typeof(retcode), typeof(w), typeof(fb)}(oe, pr, wb, clr, gamma, retcode,
                                                           w, fb)
    end
end
function SchurComplementHierarchicalRiskParityResult(; oe::Type{<:OptimisationEstimator},
                                                     pr::Option{<:AbstractPriorResult},
                                                     wb::Option{<:WeightBounds},
                                                     clr::Option{<:AbstractClusteringResult},
                                                     gamma::Union{<:Number, <:VecNum},
                                                     retcode::OptimisationReturnCode,
                                                     w::Option{<:VecNum},
                                                     fb::Option{<:OptE_Opt})::SchurComplementHierarchicalRiskParityResult
    return SchurComplementHierarchicalRiskParityResult(oe, pr, wb, clr, gamma, retcode, w,
                                                       fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for Schur Complement algorithm variants.

# Related

  - [`NonMonotonicSchurComplement`](@ref)
  - [`MonotonicSchurComplement`](@ref)
"""
abstract type SchurComplementAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Non-monotonic Schur Complement algorithm variant for SCHRP.

Uses the raw Schur complement formula without monotonicity correction.

# Related

  - [`SchurComplementAlgorithm`](@ref)
  - [`MonotonicSchurComplement`](@ref)
"""
struct NonMonotonicSchurComplement <: SchurComplementAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Monotonic Schur Complement algorithm variant for SCHRP.

Applies a bisection-based correction to ensure the Schur complement allocation factor ``\\gamma`` is monotonically increasing with cluster risk, controlled by convergence tolerance `tol` and maximum iterations `iter`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MonotonicSchurComplement(;
        N::Integer = 10,
        tol::Number = 1e-4,
        iter::Option{<:Integer} = nothing,
        strict::Bool = false
    ) -> MonotonicSchurComplement

Keywords correspond to the struct's fields.

# Related

  - [`SchurComplementAlgorithm`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
"""
@concrete struct MonotonicSchurComplement <: SchurComplementAlgorithm
    """
    $(field_dict[:N_msc])
    """
    N
    """
    $(field_dict[:tol])
    """
    tol
    """
    $(field_dict[:iter])
    """
    iter
    """
    $(field_dict[:strict_conv])
    """
    strict
    function MonotonicSchurComplement(N::Integer, tol::Number, iter::Option{<:Integer},
                                      strict::Bool)
        @argcheck(N > 0, DomainError(N, "N must be > 0"))
        @argcheck(tol > 0, DomainError(tol, "tol must be > 0"))
        if !isnothing(iter)
            @argcheck(iter > 0, DomainError(iter, "iter must be > 0"))
        end
        return new{typeof(N), typeof(tol), typeof(iter), typeof(strict)}(N, tol, iter,
                                                                         strict)
    end
end
function MonotonicSchurComplement(; N::Integer = 10, tol::Number = 1e-4,
                                  iter::Option{<:Integer} = nothing,
                                  strict::Bool = false)::MonotonicSchurComplement
    return MonotonicSchurComplement(N, tol, iter, strict)
end
"""
$(DocStringExtensions.TYPEDEF)

Parameters for the Schur Complement step of SCHRP.

`SchurComplementParams` collects the risk measure, initial allocation factor ``\\gamma``, positive-definite matrix correction, and monotonicity algorithm used in the Schur Complement Hierarchical Risk Parity optimisation.

# Fields

$(DocStringExtensions.FIELDS)

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
    """
    $(field_dict[:r])
    """
    r
    """
    $(field_dict[:gamma_schur])
    """
    gamma
    """
    $(field_dict[:pdm])
    """
    pdm
    """
    $(field_dict[:schalg])
    """
    alg
    """
    $(field_dict[:flag])
    """
    flag
    function SchurComplementParams(r::Sd_Var, gamma::Number, pdm::Option{<:Posdef},
                                   alg::SchurComplementAlgorithm, flag::Bool)
        @argcheck(one(gamma) >= gamma >= zero(gamma),
                  DomainError(gamma, "gamma must be in [0, 1]"))
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
                               flag::Bool = true)::SchurComplementParams
    return SchurComplementParams(r, gamma, pdm, alg, flag)
end
"""
    const VecScP = AbstractVector{<:SchurComplementParams}

Alias for a vector of Schur complement parameters.

Represents a collection of [`SchurComplementParams`](@ref) objects, used when different cluster levels have different Schur complement configurations.

# Related

  - [`SchurComplementParams`](@ref)
  - [`ScP_VecScP`](@ref)
"""
const VecScP = AbstractVector{<:SchurComplementParams}
"""
    const ScP_VecScP = Union{<:SchurComplementParams, <:VecScP}

Alias for a single or vector of Schur complement parameters.

Matches either a single [`SchurComplementParams`](@ref) or a vector of them ([`VecScP`](@ref)).

# Related

  - [`SchurComplementParams`](@ref)
  - [`VecScP`](@ref)
"""
const ScP_VecScP = Union{<:SchurComplementParams, <:VecScP}
"""
    port_opt_view(sp, i, X)

Get a view or subset of Schur complement parameters for cluster index `i`.

Returns a [`SchurComplementParams`](@ref) with the risk measure sliced for the given cluster index. Used internally when iterating over cluster levels.

# Arguments

  - `sp`: [`SchurComplementParams`](@ref) or vector thereof.
  - `i`: Cluster index or range.
  - `X`: Data matrix (used for slicing risk measures).

# Returns

  - Sliced [`SchurComplementParams`](@ref).

# Related

  - [`SchurComplementParams`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function port_opt_view(sp::SchurComplementParams, i, X::MatNum, args...)
    r = port_opt_view(sp.r, i, X)
    return SchurComplementParams(; r = r, gamma = sp.gamma, pdm = sp.pdm, alg = sp.alg,
                                 flag = sp.flag)
end
"""
$(DocStringExtensions.TYPEDEF)

Schur Complement Hierarchical Risk Parity (SCHRP) portfolio optimiser.

`SchurComplementHierarchicalRiskParity` extends HRP by using the Schur complement of the covariance matrix to more accurately decompose inter-cluster risk when allocating portfolio weights across the dendrogram.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SchurComplementHierarchicalRiskParity(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        params::ScP_VecScP = SchurComplementParams(),
        fb::Option{<:OptE_Opt} = nothing
    ) -> SchurComplementHierarchicalRiskParity

Keywords correspond to the struct's fields.

## Validation

  - If `params` is a vector: `!isempty(params)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> SchurComplementHierarchicalRiskParity()
SchurComplementHierarchicalRiskParity
     opt ┼ HierarchicalOptimiser
         │       pe ┼ EmpiricalPrior
         │          │        ce ┼ PortfolioOptimisersCovariance
         │          │           │   ce ┼ Covariance
         │          │           │      │    me ┼ SimpleExpectedReturns
         │          │           │      │       │   w ┴ nothing
         │          │           │      │    ce ┼ GeneralCovariance
         │          │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │          │           │      │       │    w ┴ nothing
         │          │           │      │   alg ┴ FullMoment()
         │          │           │   mp ┼ MatrixProcessing
         │          │           │      │     pdm ┼ Posdef
         │          │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │          │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │          │           │      │      dn ┼ nothing
         │          │           │      │      dt ┼ nothing
         │          │           │      │     alg ┼ nothing
         │          │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │          │        me ┼ SimpleExpectedReturns
         │          │           │   w ┴ nothing
         │          │   horizon ┴ nothing
         │      cle ┼ ClustersEstimator
         │          │    ce ┼ PortfolioOptimisersCovariance
         │          │       │   ce ┼ Covariance
         │          │       │      │    me ┼ SimpleExpectedReturns
         │          │       │      │       │   w ┴ nothing
         │          │       │      │    ce ┼ GeneralCovariance
         │          │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │          │       │      │       │    w ┴ nothing
         │          │       │      │   alg ┴ FullMoment()
         │          │       │   mp ┼ MatrixProcessing
         │          │       │      │     pdm ┼ Posdef
         │          │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │          │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │          │       │      │      dn ┼ nothing
         │          │       │      │      dt ┼ nothing
         │          │       │      │     alg ┼ nothing
         │          │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │          │    de ┼ Distance
         │          │       │   power ┼ nothing
         │          │       │     alg ┴ CanonicalDistance()
         │          │   alg ┼ HClustAlgorithm
         │          │       │   linkage ┴ Symbol: :ward
         │          │   onc ┼ OptimalNumberClusters
         │          │       │   max_k ┼ nothing
         │          │       │     alg ┼ SecondOrderDifference
         │          │       │         │   alg ┼ StandardisedValue
         │          │       │         │       │   mv ┼ MeanValue
         │          │       │         │       │      │   w ┴ nothing
         │          │       │         │       │   sv ┼ StdValue
         │          │       │         │       │      │           w ┼ nothing
         │          │       │         │       │      │   corrected ┴ Bool: true
         │      slv ┼ nothing
         │       wb ┼ WeightBounds
         │          │   lb ┼ Float64: 0.0
         │          │   ub ┴ Float64: 1.0
         │     fees ┼ nothing
         │     sets ┼ nothing
         │       wf ┼ IterativeWeightFinaliser
         │          │   iter ┴ Int64: 100
         │      brt ┼ Bool: false
         │   cle_pr ┼ Bool: true
         │   strict ┴ Bool: false
  params ┼ SchurComplementParams
         │       r ┼ Variance
         │         │   settings ┼ RiskMeasureSettings
         │         │            │   scale ┼ Float64: 1.0
         │         │            │      ub ┼ nothing
         │         │            │     rke ┴ Bool: true
         │         │      sigma ┼ nothing
         │         │       chol ┼ nothing
         │         │         rc ┼ nothing
         │         │        alg ┴ SquaredSOCRiskExpr()
         │   gamma ┼ Float64: 0.5
         │     pdm ┼ Posdef
         │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │     alg ┼ MonotonicSchurComplement
         │         │        N ┼ Int64: 10
         │         │      tol ┼ Float64: 0.0001
         │         │     iter ┼ nothing
         │         │   strict ┴ Bool: false
         │    flag ┴ Bool: true
      fb ┴ nothing
```

# Mathematical definition

When splitting cluster ``C`` with sub-clusters ``C_1`` and ``C_2``, the Schur complement of the covariance partitioned as ``\\mathbf{\\Sigma}_{C} = \\begin{pmatrix} \\mathbf{\\Sigma}_{11} & \\mathbf{\\Sigma}_{12} \\\\ \\mathbf{\\Sigma}_{21} & \\mathbf{\\Sigma}_{22} \\end{pmatrix}`` is:

```math
\\begin{align}
\\mathbf{S}(\\mathbf{\\Sigma}_{11}) &= \\mathbf{\\Sigma}_{22} - \\mathbf{\\Sigma}_{21} \\mathbf{\\Sigma}_{11}^{-1} \\mathbf{\\Sigma}_{12}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{S}(\\mathbf{\\Sigma}_{11})``: Schur complement of the covariance block ``\\mathbf{\\Sigma}_{11}``.
  - ``\\mathbf{\\Sigma}_{11}``, ``\\mathbf{\\Sigma}_{12}``, ``\\mathbf{\\Sigma}_{21}``, ``\\mathbf{\\Sigma}_{22}``: Covariance sub-blocks corresponding to the partition of cluster ``C`` into ``C_1`` and ``C_2``.

The bisection weight ``\\alpha`` is then computed from the Schur-complement-corrected inter-cluster risks of ``C_1`` and ``C_2``, yielding a more accurate decomposition than vanilla HRP.

# Related

  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct SchurComplementHierarchicalRiskParity <:
                               ClusteringOptimisationEstimator
    """
    $(field_dict[:opt_hier])
    """
    @fprop opt
    """
    $(field_dict[:params])
    """
    params
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function SchurComplementHierarchicalRiskParity(opt::HierarchicalOptimiser,
                                                   params::ScP_VecScP,
                                                   fb::Option{<:OptE_Opt})
        if isa(params, AbstractVector)
            @argcheck(!isempty(params), IsEmptyError("params cannot be empty"))
        end
        return new{typeof(opt), typeof(params), typeof(fb)}(opt, params, fb)
    end
end
function SchurComplementHierarchicalRiskParity(;
                                               opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                               params::ScP_VecScP = SchurComplementParams(),
                                               fb::Option{<:OptE_Opt} = nothing)::SchurComplementHierarchicalRiskParity
    return SchurComplementHierarchicalRiskParity(opt, params, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`SchurComplementHierarchicalRiskParity`](@ref) requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function needs_previous_weights(opt::SchurComplementHierarchicalRiskParity)
    return (needs_previous_weights(opt.opt) || needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`SchurComplementHierarchicalRiskParity`](@ref) `sh` sliced to asset indices `i`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(sh::SchurComplementHierarchicalRiskParity, i, X::MatNum,
                       args...)::SchurComplementHierarchicalRiskParity
    X = isa(sh.opt.pe, AbstractPriorResult) ? sh.opt.pe.X : X
    opt = port_opt_view(sh.opt, i)
    params = port_opt_view(sh.params, i, X)
    return SchurComplementHierarchicalRiskParity(; opt = opt, params = params, fb = sh.fb)
end
"""
    symmetric_step_up_matrix(n1::Integer, n2::Integer)

Construct a symmetric step-up matrix for Schur complement augmentation.

Builds a matrix that maps between cluster levels of sizes `n1` and `n2`. Requires `|n1 - n2| <= 1`.

# Arguments

  - `n1`: Number of rows (one cluster level size).
  - `n2`: Number of columns (adjacent cluster level size).

# Returns

  - A numeric matrix of size `n1 × n2`.

# Related

  - [`schur_augmentation`](@ref)
"""
function symmetric_step_up_matrix(n1::Integer, n2::Integer)
    @argcheck(abs(n1 - n2) <= 1,
              DomainError("n1 ($n1) and n2 ($n2) must differ by at most 1"))

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
"""
    schur_augmentation(A, B, C, gamma)

Apply the Schur complement augmentation to a risk sub-matrix.

Computes the augmented matrix `A - gamma * B * inv(C) * B'` and applies a step-up matrix correction. Used internally in the SCHRP algorithm to decompose inter-cluster risk.

# Arguments

  - `A`: Intra-cluster covariance sub-matrix.
  - `B`: Off-diagonal cross-cluster covariance sub-matrix.
  - `C`: Inter-cluster covariance sub-matrix.
  - `gamma`: Schur complement scaling parameter.

# Returns

  - Augmented matrix.

# Related

  - [`SchurComplementParams`](@ref)
  - [`symmetric_step_up_matrix`](@ref)
"""
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
"""
    naive_portfolio_risk(r, sigma)

Compute the naive (inverse-volatility) portfolio risk for a given risk measure.

Returns the portfolio risk when weights are set to the inverse-variance or inverse-volatility allocation (i.e., `w = 1/diag(sigma)`, normalised to sum to one). Dispatches on the risk measure type.

# Arguments

  - `r`: Risk measure ([`Variance`](@ref) or [`StandardDeviation`](@ref)).
  - `sigma`: Covariance matrix.

# Returns

  - Scalar portfolio risk.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`schur_complement_weights`](@ref)
"""
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
"""
    schur_complement_weights(pr, items, ...)

Compute HRP/HERC weights using the Schur complement method.

Allocates weights across cluster levels using the Schur complement of the covariance matrix, providing more accurate inter-cluster risk decomposition than naive HRP.

# Arguments

  - `pr`: Prior result containing asset moments.
  - `items`: Vector of vectors of asset indices per cluster level.
  - Additional parameters from `SchurComplementParams`.

# Returns

  - Portfolio weight vector.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`schur_augmentation`](@ref)
"""
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
            # Copies, not views: the write-backs into `sigma` below (and `posdef!`) must
            # not alias the blocks being augmented.
            A = sigma[lc, lc]
            C = sigma[rc, rc]
            if length(lc) <= 1
                A_aug = A
                C_aug = C
            else
                B = sigma[lc, rc]
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
                    throw(ArgumentError("Augmented matrix could not be made positive definite. Use `MonotonicSchurComplement()` or reduce gamma: $gamma. Original error: $(sprint(showerror, e))"))
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
"""
    schur_complement_binary_search(objective, lgamma, hgamma, ...)

Binary search for the optimal Schur complement scaling parameter γ.

Performs binary search in the interval `[lgamma, hgamma]` to find the γ value that satisfies the monotonicity condition for the Schur complement HRP.

# Arguments

  - `objective`: Function to evaluate monotonicity.
  - `lgamma`: Lower bound for γ search.
  - `hgamma`: Upper bound for γ search.
  - Additional tolerance and iteration parameters.

# Returns

  - Optimal γ value.

# Related

  - [`MonotonicSchurComplement`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute SCHRP weights using the monotonic Schur complement method.

Uses binary search to find the γ value that maximises risk reduction while maintaining monotonicity, then delegates to the non-monotonic overload.

# Related

  - [`schur_complement_weights`](@ref)
  - [`MonotonicSchurComplement`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the Schur Complement HRP optimisation for a single set of parameters.

Internal dispatch called by [`optimise`](@ref). Computes the prior and clustering, then applies `schur_complement_weights` to allocate portfolio weights.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
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
    return SchurComplementHierarchicalRiskParityResult(; oe = typeof(sh), pr = pr, wb = wb,
                                                       clr = clr, gamma = gamma,
                                                       retcode = retcode, w = w,
                                                       fb = nothing)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the Schur Complement HRP optimisation for a vector of parameter sets.

Internal dispatch called by [`optimise`](@ref). Combines risk-scaled weights from multiple `SchurComplementParams` configurations.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
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
    return SchurComplementHierarchicalRiskParityResult(; oe = typeof(sh), pr = pr, wb = wb,
                                                       clr = clr, gamma = gammas,
                                                       retcode = retcode, w = w,
                                                       fb = nothing)
end
"""
    optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any, Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...) -> SchurComplementHierarchicalRiskParityResult

Run the Schur Complement Hierarchical Risk Parity portfolio optimisation.

# Arguments

  - `sh`: The Schur complement hierarchical risk parity optimiser to use.
  - $(arg_dict[:rd]) If `isa(sh.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementHierarchicalRiskParityResult`](@ref)
"""
function optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(sh, rd; dims = dims, kwargs...)
end

export SchurComplementHierarchicalRiskParityResult, SchurComplementParams,
       SchurComplementHierarchicalRiskParity, NonMonotonicSchurComplement,
       MonotonicSchurComplement
