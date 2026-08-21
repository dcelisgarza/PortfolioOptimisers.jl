"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`HierarchicalRiskParity`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function hierarchical_risk_parity_td_defaults()::NamedTuple
    return (; r = Variance(), sca = SumScalariser())
end
"""
$(DocStringExtensions.TYPEDEF)

Allocates weights by recursively bisecting the dendrogram's leaf order and splitting each part's weight in inverse proportion to the risk of its two halves.

This is the Hierarchical Risk Parity algorithm. It clusters the assets, orders the leaves, then walks that order down to the individual assets. It solves no optimisation problem, so it accepts a risk measure that is not convex.

# Mathematical definition

Every asset starts at ``w_i = 1``. The algorithm splits the dendrogram's leaf order in half, then splits each half in half, until every part holds one asset. At each split it divides the part's weight between the two halves ``C_1`` and ``C_2``:

```math
\\begin{align}
\\tilde{w}_i(C) &= \\frac{\\rho(\\{i\\})^{-1}}{\\sum_{j \\in C} \\rho(\\{j\\})^{-1}} \\quad \\forall\\, i \\in C\\,,\\\\
\\tilde{\\rho}(C) &= \\rho\\left(\\tilde{\\boldsymbol{w}}(C)\\right)\\,,\\\\
\\alpha &= \\frac{\\tilde{\\rho}(C_2)}{\\tilde{\\rho}(C_1) + \\tilde{\\rho}(C_2)}\\,,\\\\
\\boldsymbol{w}_{C_1} &\\leftarrow \\alpha \\, \\boldsymbol{w}_{C_1}\\,,\\\\
\\boldsymbol{w}_{C_2} &\\leftarrow (1 - \\alpha) \\, \\boldsymbol{w}_{C_2}\\,.
\\end{align}
```

Where:

  - ``\\rho``: Risk measure `r`, resolved by [`factory`](@ref) against the prior.
  - ``\\rho(\\{i\\})``: Risk of asset ``i`` held alone, from [`unitary_expected_risks`](@ref).
  - ``\\tilde{\\boldsymbol{w}}(C)``: Naive risk parity weights inside part ``C``, zero outside it.
  - ``\\tilde{\\rho}(C)``: Risk of that naive risk parity sub-portfolio.
  - ``\\alpha``: Fraction of the part's weight that goes to ``C_1``.
  - ``C_1``, ``C_2``: The two halves of the part being split, in leaf order.

Two steps follow the recursion. The split factor ``\\alpha`` is first clamped so that neither half can leave the resolved weight bounds, see [`split_factor_weight_constraints`](@ref). The final vector is then normalised to sum to one and passed to the weight finaliser `wf`. When `r` is a vector, [`hrp_scalarised_risk`](@ref) combines the measures with `sca` before the split factor is formed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalRiskParity(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        r::TD{<:OptRM_VecOptRM} = Variance(),
        sca::TD{<:Scalariser} = SumScalariser(),
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> HierarchicalRiskParity

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the risk measure, scalariser and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `fb`).

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`HierarchicalRiskParity` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` already holds a prior **result**, the method replaces `X` with `opt.pe.X`, so the children are viewed against the prior's own observations rather than the caller's matrix.
  - `r` recurses through [`port_opt_view`](@ref) with that matrix. `opt` recurses with the index alone, because a [`HierarchicalOptimiser`](@ref) holds no risk measure and therefore needs no returns.
  - `sca` and `fb` are carried through unchanged.

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
      │    x_src ┼ Symbol: :prior
      │    z_src ┼ Symbol: :data
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

# Related

  - [`optimise`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalOptimiser`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`unitary_expected_risks`](@ref)
  - [`split_factor_weight_constraints`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:lopezdeprado2016])
  - $(ref_dict[:cajas2025]) Section 12.1.
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
    function HierarchicalRiskParity(opt::HierarchicalOptimiser, r::TD{<:OptRM_VecOptRM},
                                    sca::TD{<:Scalariser}, fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :HierarchicalRiskParity)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        assert_risk_measure_required(r, :HierarchicalRiskParity)
        assert_time_dependent_substitution(HierarchicalRiskParity, (; opt, r, sca, fb),
                                           hierarchical_risk_parity_td_defaults())
        return new{typeof(opt), typeof(r), typeof(sca), typeof(fb)}(opt, r, sca, fb)
    end
end
function HierarchicalRiskParity(; opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                r::TD{<:OptRM_VecOptRM} = Variance(),
                                sca::TD{<:Scalariser} = SumScalariser(),
                                fb::TDO_Option{<:OptE_Opt} = nothing)::HierarchicalRiskParity
    return HierarchicalRiskParity(opt, r, sca, fb)
end
function time_dependent_field_defaults(::HierarchicalRiskParity)::NamedTuple
    return hierarchical_risk_parity_td_defaults()
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
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
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
    split_factor_weight_constraints(alpha::Number, wb::WeightBounds, w::VecNum,
                                    lc::VecNum, rc::VecNum) -> Number

Clamp a bisection split factor so that neither half of the split leaves its weight bounds.

The recursion scales the left half by `alpha` and the right half by `1 - alpha`. Both halves carry one common weight before the split, so the bounds on that half's total translate into bounds on the factor itself. This method clamps the factor for the left half, then clamps its complement for the right half, and returns what survives both.

# Arguments

  - `alpha`: The unclamped split factor, the left half's share of the part's weight.
  - `wb`: Resolved weight bounds. `wb.lb` and `wb.ub` must be **vectors** over the whole universe, which is what [`weight_bounds_constraints`](@ref) returns.
  - `w`: The current weight vector, before this split is applied.
  - `lc`: Asset indices of the left half.
  - `rc`: Asset indices of the right half.

# Returns

  - `alpha::Number`: The clamped split factor. It is a **scalar**, not a bounds object.

# Details

  - Each half's common weight is read from its first member, `w[lc[1]]` and `w[rc[1]]`, because the recursion has scaled every member of a half by the same factors.
  - A zero common weight is replaced by `sqrt(eps(...))`, so the division that turns a weight bound into a factor bound cannot divide by zero.
  - The bound on a half is the **sum** of its members' bounds, so a half whose bounds cannot hold the weight it is given leaves the factor at the nearest reachable value rather than failing.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
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
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(hrp.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     x_src = hrp.opt.x_src, z_src = hrp.opt.z_src)
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
    return HierarchicalRiskParityResult(;
                                        hr = HierarchicalResult(; pr = pr, clr = clr,
                                                                wb = wb, fees = fees,
                                                                retcode = retcode, w = w),
                                        r = r, sca = hrp.sca, fb = nothing)
end
"""
    hrp_scalarised_risk(sca::Scalariser, wu::MatNum, wk::VecNum, rku::VecNum,
                        lc::VecNum, rc::VecNum, rs::VecOptRM, X::MatNum,
                        fees::Option{<:Fees}) -> Tuple

Combine several risk measures into the one left and one right risk that a bisection step needs.

Each measure builds its own naive risk parity sub-portfolio for the two halves, because the weights follow that measure's own unitary risks. The scalariser then combines the two risk vectors, pair by pair, into a single pair.

# Arguments

  - `sca`: Scalarisation strategy ([`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref), or [`LogSumExpScalariser`](@ref)).
  - `wu`: Scratch weight matrix, of size `size(X, 2)` by two. It is refilled per measure.
  - `wk`: Scratch weight vector for [`unitary_expected_risks!`](@ref), of length `size(X, 2)`.
  - `rku`: Scratch unitary risk vector, of length `size(X, 2)`. It is overwritten per measure.
  - `lc`: Asset indices of the left half.
  - `rc`: Asset indices of the right half.
  - `rs`: Vector of risk measures, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees, or `nothing`.

# Returns

  - `(lrisk, rrisk)::Tuple`: The combined left and right risk.

# Details

  - Each measure's contribution is multiplied by its own `settings.scale` before the scalariser sees it.
  - The scalariser reduces over the pair with `by = sum`, so a [`MaxScalariser`](@ref) selects the measure with the largest **total** of the two risks, and returns that measure's own pair.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`unitary_expected_risks!`](@ref)
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
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(hrp.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     x_src = hrp.opt.x_src, z_src = hrp.opt.z_src)
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
    return HierarchicalRiskParityResult(;
                                        hr = HierarchicalResult(; pr = pr, clr = clr,
                                                                wb = wb, fees = fees,
                                                                retcode = retcode, w = w),
                                        r = r, sca = hrp.sca, fb = nothing)
end
"""
    optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...) -> HierarchicalRiskParityResult

Run the Hierarchical Risk Parity portfolio optimisation.

# Arguments

  - `hrp`: The hierarchical risk parity optimiser to use.
  - $(arg_dict[:rd]) If `isa(hrp.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Details

Unlike [`HierarchicalEqualRiskContribution`](@ref) and [`NestedClustered`](@ref), this optimiser accepts no `branchorder` keyword. Recursive bisection allocates by splitting the dendrogram's leaf permutation, so that permutation is the algorithm's input rather than a presentation detail, and the clusterisation always runs with the optimal ordering. A `branchorder` passed here is absorbed by `kwargs` and ignored. See ADR 0055.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
"""
function optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
    return _optimise(hrp, rd; dims = dims, kwargs...)
end

@pipe_delegates HierarchicalRiskParity opt
@pipe_route_sigma_ucs HierarchicalRiskParity
export HierarchicalRiskParity
