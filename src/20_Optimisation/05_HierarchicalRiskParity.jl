"""
$(DocStringExtensions.TYPEDEF)

Hierarchical Risk Parity (HRP) portfolio optimiser.

`HierarchicalRiskParity` implements the Hierarchical Risk Parity algorithm of López de Prado (2016). It clusters assets using hierarchical clustering, then allocates weights by recursively bisecting the dendrogram and applying inverse-risk weighting within each cluster.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalRiskParity(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        r::OptRM_VecOptRM = Variance(),
        sca::Scalariser = SumScalariser(),
        fb::Option{<:OptE_Opt} = nothing
    ) -> HierarchicalRiskParity

Keywords correspond to the struct's fields.

## Validation

  - If `r` is a vector: `!isempty(r)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> HierarchicalRiskParity()
HierarchicalRiskParity
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
    r ┼ Variance
      │   settings ┼ RiskMeasureSettings
      │            │   scale ┼ Float64: 1.0
      │            │      ub ┼ nothing
      │            │     rke ┴ Bool: true
      │      sigma ┼ nothing
      │       chol ┼ nothing
      │         rc ┼ nothing
      │        alg ┴ SquaredSOCRiskExpr()
  sca ┼ SumScalariser()
   fb ┴ nothing
```

# Mathematical definition

At each bisection step, the algorithm splits cluster ``C`` into sub-clusters ``C_1`` and ``C_2`` and allocates weights proportional to inverse portfolio risk:

```math
\\begin{align}
\\alpha &= \\frac{\\tilde{\\rho}(C_2)}{\\tilde{\\rho}(C_1) + \\tilde{\\rho}(C_2)}\\,, \\\\
\\boldsymbol{w}_{C_1} \\leftarrow \\alpha \\, \\boldsymbol{w}_C, \\quad \\boldsymbol{w}_{C_2} \\leftarrow (1 - \\alpha) \\, \\boldsymbol{w}_C\\,.
\\end{align}
```

Where:

  - ``\\alpha``: Bisection weight allocating fraction of cluster weight ``\\boldsymbol{w}_C`` to sub-cluster ``C_1``.
  - ``\\tilde{\\rho}(C)``: Risk of the quasi-diagonal sub-portfolio restricted to cluster ``C``.
  - ``C_1``, ``C_2``: Sub-clusters of the bisected cluster ``C``.
  - ``\\boldsymbol{w}_C``: Weight vector assigned to cluster ``C`` before bisection.

# Related

  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalOptimiser`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct HierarchicalRiskParity <: ClusteringOptimisationEstimator
    """
    $(field_dict[:opt_hier])
    """
    @fprop opt
    """
    $(field_dict[:r])
    """
    @fprop r
    """
    $(field_dict[:sca])
    """
    sca
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function HierarchicalRiskParity(opt::HierarchicalOptimiser, r::OptRM_VecOptRM,
                                    sca::Scalariser, fb::Option{<:OptE_Opt})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        return new{typeof(opt), typeof(r), typeof(sca), typeof(fb)}(opt, r, sca, fb)
    end
end
function HierarchicalRiskParity(; opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                r::OptRM_VecOptRM = Variance(),
                                sca::Scalariser = SumScalariser(),
                                fb::Option{<:OptE_Opt} = nothing)::HierarchicalRiskParity
    return HierarchicalRiskParity(opt, r, sca, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`HierarchicalRiskParity`](@ref) requires previous portfolio weights.

Returns `true` if any of the base optimiser, risk measure, or fallback estimator require previous weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`HierarchicalRiskParity`](@ref)
"""
function needs_previous_weights(opt::HierarchicalRiskParity)
    return (needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`HierarchicalRiskParity`](@ref) `hrp` sliced to asset indices `i`.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum,
                       args...)::HierarchicalRiskParity
    X = isa(hrp.opt.pe, AbstractPriorResult) ? hrp.opt.pe.X : X
    r = port_opt_view(hrp.r, i, X)
    opt = port_opt_view(hrp.opt, i)
    return HierarchicalRiskParity(; r = r, opt = opt, sca = hrp.sca, fb = hrp.fb)
end
"""
    split_factor_weight_constraints(alpha, wb, w, ...)

Split and scale factor weight constraints for hierarchical risk parity.

Distributes the weight constraints across clusters based on the hierarchical factor `alpha` and the current weight allocation `w`.

# Arguments

  - `alpha`: Hierarchical scaling factor.
  - `wb`: Weight bounds.
  - `w`: Current portfolio weights.
  - Additional parameters.

# Returns

  - Tuple of updated weight bounds for each cluster.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`WeightBounds`](@ref)
"""
function split_factor_weight_constraints(alpha::Number, wb::WeightBounds, w::VecNum,
                                         lc::VecNum, rc::VecNum)
    lb = wb.lb
    ub = wb.ub
    wlc = w[lc[1]]
    wrc = w[rc[1]]
    if iszero(wlc)
        wlc = sqrt(eps(typeof(wlc)))
    end
    if iszero(wrc)
        wrc = sqrt(eps(typeof(wrc)))
    end
    alpha = min(sum(view(ub, lc)) / wlc, max(sum(view(lb, lc)) / wlc, alpha))
    return one(alpha) -
           min(sum(view(ub, rc)) / wrc, max(sum(view(lb, rc)) / wrc, one(alpha) - alpha))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the Hierarchical Risk Parity optimisation for a single risk measure.

Internal dispatch called by [`optimise`](@ref). Computes the prior, clusters assets, applies inverse-risk bisection allocation, and finalises weight bounds.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(hrp::HierarchicalRiskParity{<:Any, <:OptimisationRiskMeasure},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    hrp = reset_time_dependent_estimator(hrp)
    rd = returns_result_picker(rd, hrp.opt.brt)
    pr = prior(hrp.opt.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(hrp.opt.cle, pr; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     cle_pr = hrp.opt.cle_pr)
    r = factory(hrp.r, pr, hrp.opt.slv)
    wu = Matrix{eltype(X)}(undef, size(X, 2), 2)
    fees = fees_constraints(hrp.opt.fees, hrp.opt.sets; strict = hrp.opt.strict,
                            datatype = eltype(X))
    rku = unitary_expected_risks(r, X, fees)
    wb = weight_bounds_constraints(hrp.opt.wb, hrp.opt.sets; N = size(X, 2),
                                   strict = hrp.opt.strict, datatype = eltype(X))
    w = ones(eltype(X), size(X, 2))
    items = [clr.res.order]
    while length(items) > 0
        items = [i[j:k] for i in items
                 for (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i in 1:2:length(items)
            fill!(wu, zero(eltype(X)))
            lc = items[i]
            rc = items[i + 1]
            wu[lc, 1] .= inv.(view(rku, lc))
            wu[lc, 1] ./= sum(view(wu, lc, 1))
            wu[rc, 2] .= inv.(view(rku, rc))
            wu[rc, 2] ./= sum(view(wu, rc, 2))
            lrisk = expected_risk(r, view(wu, :, 1), X, fees)
            rrisk = expected_risk(r, view(wu, :, 2), X, fees)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    retcode, w = finalise_weight_bounds(hrp.opt.wf, wb, w / sum(w))
    return HierarchicalResult(; oe = typeof(hrp), pr = pr, clr = clr, wb = wb, fees = fees,
                              retcode = retcode, w = w, fb = nothing)
end
"""
    hrp_scalarised_risk(scalariser, wu, wk, rku, lc, rc, rs, X, fees)

Compute the scalarised HRP left/right cluster risk for weight allocation.

Aggregates risk measures across clusters using a scalariser (sum, max, min, or log-sum-exp), returning the left and right cluster risks used to allocate weights in HRP.

# Arguments

  - `scalariser`: Scalarisation strategy ([`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref), or [`LogSumExpScalariser`](@ref)).
  - `wu`: Unitary weight matrix (pre-allocated buffer).
  - `wk`: Cluster weight vector.
  - `rku`: Unitary risk vector.
  - `lc`: Left cluster asset indices.
  - `rc`: Right cluster asset indices.
  - `rs`: Vector of risk measures.
  - `X`: Return matrix.
  - `fees`: Optional fees.

# Returns

  - `(lrisk, rrisk)`: Left and right cluster risk scalars.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`herc_scalarised_risk_i!`](@ref)
"""
function hrp_scalarised_risk(sca::Scalariser, wu::MatNum, wk::VecNum, rku::VecNum,
                             lc::VecNum, rc::VecNum, rs::VecOptRM, X::MatNum,
                             fees::Option{<:Fees})
    return scalarise(sca, rs; by = sum) do r
        fill!(wu, zero(eltype(X)))
        unitary_expected_risks!(wk, rku, r, X, fees)
        wu[lc, 1] .= inv.(view(rku, lc))
        wu[lc, 1] ./= sum(view(wu, lc, 1))
        wu[rc, 2] .= inv.(view(rku, rc))
        wu[rc, 2] ./= sum(view(wu, rc, 2))
        scale = r.settings.scale
        return (expected_risk(r, view(wu, :, 1), X, fees) * scale,
                expected_risk(r, view(wu, :, 2), X, fees) * scale)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the Hierarchical Risk Parity optimisation for a vector of risk measures, scalarising each cluster's risk.

Internal dispatch called by [`optimise`](@ref). Uses [`hrp_scalarised_risk`](@ref) to aggregate multiple risk measures.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`hrp_scalarised_risk`](@ref)
  - [`optimise`](@ref)
  - [`_optimise`](@ref)
"""
function _optimise(hrp::HierarchicalRiskParity{<:Any, <:VecOptRM},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    hrp = reset_time_dependent_estimator(hrp)
    rd = returns_result_picker(rd, hrp.opt.brt)
    pr = prior(hrp.opt.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(hrp.opt.cle, pr; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     cle_pr = hrp.opt.cle_pr)
    r = factory(hrp.r, pr, hrp.opt.slv)
    wu = Matrix{eltype(X)}(undef, size(X, 2), 2)
    wk = zeros(eltype(X), size(X, 2))
    rku = Vector{eltype(X)}(undef, size(X, 2))
    fees = fees_constraints(hrp.opt.fees, hrp.opt.sets; strict = hrp.opt.strict,
                            datatype = eltype(X))
    wb = weight_bounds_constraints(hrp.opt.wb, hrp.opt.sets; N = size(X, 2),
                                   strict = hrp.opt.strict, datatype = eltype(X))
    w = ones(eltype(X), size(X, 2))
    items = [clr.res.order]
    while length(items) > 0
        items = [i[j:k] for i in items
                 for (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i in 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            lrisk, rrisk = hrp_scalarised_risk(hrp.sca, wu, wk, rku, lc, rc, r, X, fees)
            # Allocate weight to clusters.
            alpha = one(lrisk) - lrisk / (lrisk + rrisk)
            alpha = split_factor_weight_constraints(alpha, wb, w, lc, rc)
            # Weight constraints.
            w[lc] .*= alpha
            w[rc] .*= one(alpha) - alpha
        end
    end
    retcode, w = finalise_weight_bounds(hrp.opt.wf, wb, w / sum(w))
    return HierarchicalResult(; oe = typeof(hrp), pr = pr, clr = clr, wb = wb, fees = fees,
                              retcode = retcode, w = w, fb = nothing)
end
"""
    optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...) -> HierarchicalResult

Run the Hierarchical Risk Parity portfolio optimisation.

# Arguments

  - `hrp`: The hierarchical risk parity optimiser to use.
  - $(arg_dict[:rd]) If `isa(hrp.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalResult`](@ref)
"""
function optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(hrp, rd; dims = dims, kwargs...)
end

export HierarchicalRiskParity
