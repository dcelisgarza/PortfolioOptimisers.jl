"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`HierarchicalRiskParity`](@ref) fields that may hold a [`TimeDependent`](@ref).

The constructor's substitution check and [`time_dependent_field_defaults`](@ref) both read it, so the file states the fold-less value of each field once. A field whose static default is `nothing` has no entry.

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

This is the Hierarchical Risk Parity algorithm. It clusters the assets, orders the leaves, then splits that order until every part holds one asset. It solves no optimisation problem, so it accepts a risk measure that is not convex. The resolved weight bounds clamp the split factor at every split, and the weight finaliser of `opt` acts on the result. [`_optimise`](@ref) states the steps.

# Mathematical definition

The leaf order of the dendrogram is the root part. A part ``C`` of more than one asset splits into its two halves ``C_1`` and ``C_2``. The split repeats on each half until every part holds one asset. The part's weight ``W(C)`` divides between the two halves:

```math
\\begin{align}
\\tilde{w}_i(C) &= \\frac{\\rho(\\{i\\})^{-1}}{\\sum_{j \\in C} \\rho(\\{j\\})^{-1}} \\quad \\forall\\, i \\in C\\,,\\\\
\\tilde{\\rho}(C) &= \\rho\\left(\\tilde{\\boldsymbol{w}}(C)\\right)\\,,\\\\
\\alpha &= \\frac{\\tilde{\\rho}(C_2)}{\\tilde{\\rho}(C_1) + \\tilde{\\rho}(C_2)}\\,,\\\\
W(C_1) &= \\hat{\\alpha} \\, W(C)\\,,\\\\
W(C_2) &= (1 - \\hat{\\alpha}) \\, W(C)\\,,\\\\
w_i &= W(\\{i\\})\\,.
\\end{align}
```

Where:

  - ``\\rho``: Risk measure `r`, resolved by [`factory`](@ref) against the prior. When `r` is a vector, [`hrp_scalarised_risk`](@ref) states the pair ``\\left(\\tilde{\\rho}(C_1), \\tilde{\\rho}(C_2)\\right)`` that the scalariser `sca` makes.
  - ``\\rho(\\{i\\})``: Risk of asset ``i`` held alone, from [`unitary_expected_risks`](@ref).
  - ``\\tilde{\\boldsymbol{w}}(C)``: Naive risk parity weights inside part ``C``, zero outside it.
  - ``\\tilde{\\rho}(C)``: Risk of that naive risk parity sub-portfolio.
  - $(math_dict[:alpha_split_hier])
  - ``\\hat{\\alpha}``: The split factor ``\\alpha`` clamped to the weight bounds of the two halves, see [`split_factor_weight_constraints`](@ref). Under the default bounds, zero and one for every asset, ``\\hat{\\alpha} = \\alpha``.
  - $(math_dict[:C_halves_hier])
  - ``W(C)``: Weight of part ``C``, with ``W = 1`` for the root part.
  - $(math_dict[:w_i_asset])

The weights of the two halves sum to the weight of their part, so ``\\sum_i w_i = 1``. When the weight bounds admit a portfolio, every asset holds its bounds, see [`split_factor_weight_constraints`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalRiskParity(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        r::TD{<:OptRM_VecOptRM} = Variance(),
        sca::TD{<:Scalariser} = SumScalariser(),
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> HierarchicalRiskParity

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value. The risk measure, the scalariser and the fallback are problem definition, so a cross-validation fold loop resolves them per fold. A fold-less `optimise` runs with each at its static default, which is `nothing` for `fb`.

## Validation

  - If `r` is a vector: `!isempty(r)`. The constructor throws an `IsEmptyError` otherwise.
  - `r` holds no [`NoRisk`](@ref), alone or in a vector. The constructor throws an `ArgumentError` otherwise, through [`assert_risk_measure_required`](@ref), because a zero risk leaves the split factor undefined.
  - `fb` schedules: `bind !== :nearest`. The constructor throws an `ArgumentError` otherwise.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`HierarchicalRiskParity` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` holds a prior result, the method replaces `X` with `opt.pe.X`, so it views the children against the observations of that prior, not against the caller's matrix.
  - `r` and `opt` recurse through [`port_opt_view`](@ref) with that matrix.
  - `fb` recurses through [`view_child`](@ref) with that matrix.
  - `sca` stays unchanged.

# Examples

```jldoctest
julia> HierarchicalRiskParity()
HierarchicalRiskParity
  opt ┼ HierarchicalOptimiser
      │       pe ┼ EmpiricalPrior
      │          │           ce ┼ PortfolioOptimisersCovariance
      │          │              │   ce ┼ Covariance
      │          │              │      │    me ┼ SimpleExpectedReturns
      │          │              │      │       │   w ┴ nothing
      │          │              │      │    ce ┼ GeneralCovariance
      │          │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │          │              │      │       │    w ┴ nothing
      │          │              │      │   alg ┼ FullMoment()
      │          │              │      │     w ┴ nothing
      │          │              │   mp ┼ MatrixProcessing
      │          │              │      │     pdm ┼ Posdef
      │          │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │          │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │          │              │      │      dn ┼ nothing
      │          │              │      │      dt ┼ nothing
      │          │              │      │     alg ┼ nothing
      │          │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │          │           me ┼ SimpleExpectedReturns
      │          │              │   w ┴ nothing
      │          │      horizon ┼ nothing
      │          │   fill_limit ┴ nothing
      │      cle ┼ ClustersEstimator
      │          │    ce ┼ PortfolioOptimisersCovariance
      │          │       │   ce ┼ Covariance
      │          │       │      │    me ┼ SimpleExpectedReturns
      │          │       │      │       │   w ┴ nothing
      │          │       │      │    ce ┼ GeneralCovariance
      │          │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │          │       │      │       │    w ┴ nothing
      │          │       │      │   alg ┼ FullMoment()
      │          │       │      │     w ┴ nothing
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

Return whether the [`HierarchicalRiskParity`](@ref) needs previous portfolio weights.

Returns `true` when `opt.opt`, `opt.r` or `opt.fb` needs previous weights, or when a [`TimeDependent`](@ref) schedule in a field holds a value that does. A [`TurnoverRiskMeasure`](@ref) and a turnover fee both need them.

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

When `hrp.opt.pe` holds a prior result, the view reads that prior's `X` in place of the `X` it receives. The struct's `## View parameters` states what each field does.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum,
                       args...)::HierarchicalRiskParity
    X = isa(hrp.opt.pe, AbstractPriorResult) ? hrp.opt.pe.X : X
    r = port_opt_view(hrp.r, i, X)
    opt = port_opt_view(hrp.opt, i, X)
    return HierarchicalRiskParity(; r = r, opt = opt, sca = hrp.sca,
                                  fb = view_child(hrp.fb, i, X))
end
function non_investable_universe(hrp::HierarchicalRiskParity,
                                 ni::VecStr)::HierarchicalRiskParity
    return rebuild_estimator(hrp, (; opt = non_investable_universe(hrp.opt, ni)))
end
"""
    split_factor_weight_constraints(alpha::Number, wb::WeightBounds, w::VecNum,
                                    lc::VecNum, rc::VecNum) -> Number

Clamp a bisection split factor so that neither half of the split leaves its weight bounds.

The recursion scales the left half by `alpha` and the right half by `1 - alpha`. Both halves carry the weight of their part before the split, so the bounds on each half's total are bounds on the factor. [`HierarchicalRiskParity`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) both call it.

# Mathematical definition

```math
\\begin{align}
L(C) &= \\sum_{i \\in C} l_i\\,,\\\\
U(C) &= \\sum_{i \\in C} u_i\\,,\\\\
\\alpha_1 &= \\min\\left(\\frac{U(C_1)}{W(C)}, \\max\\left(\\frac{L(C_1)}{W(C)}, \\alpha\\right)\\right)\\,,\\\\
\\hat{\\alpha} &= 1 - \\min\\left(\\frac{U(C_2)}{W(C)}, \\max\\left(\\frac{L(C_2)}{W(C)}, 1 - \\alpha_1\\right)\\right)\\,.
\\end{align}
```

Where:

  - ``l_i``, ``u_i``: Lower and upper weight bounds of asset ``i``, with ``l_i \\leq u_i``.
  - ``L(C)``, ``U(C)``: Lower and upper bounds on the total weight of part ``C``.
  - $(math_dict[:C_halves_hier])
  - ``W(C)``: Weight of the part ``C`` before the split, with ``W(C) > 0``.
  - $(math_dict[:alpha_split_hier])
  - ``\\alpha_1``: The factor clamped to the bounds of ``C_1``.
  - ``\\hat{\\alpha}``: The factor clamped to the bounds of ``C_1``, then to the bounds of ``C_2``.

The clamp to the bounds of ``C_2`` comes last. When the bounds of the two halves cannot both hold, those of ``C_2`` hold and those of ``C_1`` do not. When ``L(C) \\leq W(C) \\leq U(C)``, the two intervals of the factor meet, and ``\\hat{\\alpha}`` lies in both. The root part has ``W = 1``, so when ``\\sum_i l_i \\leq 1 \\leq \\sum_i u_i`` every part keeps its total inside its bounds, and every asset holds its own bounds at the last split.

# Algorithm

 1. Read the weight of the part, `c`, from the first member of the left half. The recursion scales every member of a part by the same factors, so each member carries the part's weight.
 2. When `c` is zero, return `alpha` unchanged. The weights of the two halves are zero whatever the factor, and the bounds divided by `c` are not defined.
 3. Clamp `alpha` to the bounds of the left half, ``\\alpha_1``.
 4. Clamp `1 - alpha` to the bounds of the right half, and return its complement, ``\\hat{\\alpha}``.

# Arguments

  - `alpha`: The unclamped split factor, the left half's share of the part's weight.
  - `wb`: Resolved weight bounds. `wb.lb` and `wb.ub` must be vectors over the whole universe, which is what [`weight_bounds_constraints`](@ref) returns.
  - `w`: The current weight vector, before this split.
  - `lc`: Asset indices of the left half.
  - `rc`: Asset indices of the right half.

# Returns

  - `alpha::Number`: The clamped split factor, a scalar.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`WeightBounds`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function split_factor_weight_constraints(alpha::Number, wb::WeightBounds, w::VecNum,
                                         lc::VecNum, rc::VecNum)
    c = w[lc[1]]
    if iszero(c)
        return alpha
    end
    lb = wb.lb
    ub = wb.ub
    alpha = min(sum(view(ub, lc)) / c, max(sum(view(lb, lc)) / c, alpha))
    return one(alpha) -
           min(sum(view(ub, rc)) / c, max(sum(view(lb, rc)) / c, one(alpha) - alpha))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the Hierarchical Risk Parity optimisation for a single risk measure.

[`optimise`](@ref) calls this method. The struct [`HierarchicalRiskParity`](@ref) states the mathematics.

# Algorithm

 1. Resolve every [`TimeDependent`](@ref) field of `hrp` to its static default, with [`reset_time_dependent_estimator`](@ref).
 2. Pick the returns `rd` that `opt.brt` selects, with [`returns_result_picker`](@ref).
 3. Fit the prior `pr` with `opt.pe`.
 4. Find the Investable Mask `imsk` of `pr`. Resolve the fees `fees` on the full universe, and place them on the assets of `imsk`.
 5. Remove the liquidation carriers from `fees`, giving `cfees`, the fees that the risks read.
 6. Reduce `pr`, `hrp` and `rd` to the assets of `imsk`, with [`investable_reduction`](@ref). `X` is the returns matrix of the reduced `pr`.
 7. Cluster the assets with `opt.cle`, giving `clr`. The call passes no `branchorder`, so the leaf order `clr.res.order` keeps the default optimal order, because a different order changes the allocation.
 8. Resolve the risk measure `r` against `pr` and `opt.slv`, with [`factory`](@ref).
 9. Compute the risk of each asset held alone, `rku`, with [`unitary_expected_risks`](@ref).
10. Resolve the weight bounds `wb` from `opt.wb` and `opt.sets`.
11. Set every entry of `w` to one, and set `items` to the one part `clr.res.order`.
12. Replace each part in `items` that holds more than one asset by its two halves. Stop when `items` is empty.
13. For each pair of halves `lc` and `rc`, write their naive risk parity weights into the two columns of `wu`, and compute their risks `lrisk` and `rrisk` with [`expected_risk`](@ref).
14. Form the split factor `alpha`, and clamp it with [`split_factor_weight_constraints`](@ref).
15. Multiply `w[lc]` by `alpha` and `w[rc]` by `1 - alpha`, then go back to step 12.
16. Divide `w` by its sum, and apply the weight finaliser `opt.wf` with [`finalise_weight_bounds`](@ref), giving `retcode` and the final `w`.
17. Return a [`HierarchicalRiskParityResult`](@ref). Its [`HierarchicalResult`](@ref) expands `w` back onto the full asset universe through `imsk`.

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
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset, and a carrier keyed by name cannot resolve at all once its `w` sits on the
    # complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    # A weight is a quotient of two risks, so an integer sample takes a float weight type.
    T = typeof(one(eltype(pr.X)) / one(eltype(pr.X)))
    imsk = investable_mask(pr)
    fees = investable_fees_view(fees_constraints(hrp.opt.fees, hrp.opt.sets;
                                                 strict = hrp.opt.strict, datatype = T),
                                imsk, pr.X)
    # A forced exit is charged once, against the full-universe weight vector the fit
    # rebuilds, so it rides on the result alone. No sub-problem below holds that vector —
    # the exiting asset is in no cluster, its column being `NaN` — so none prices an exit.
    cfees = strip_liquidation_carriers(fees, nothing)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here:
    # the distance the clustering is built from never sees a `NaN`, and the cluster count
    # is chosen on the investable universe. The weights are expanded back in
    # `HierarchicalResult`.
    _, pr, hrp, rd = investable_reduction(imsk, pr, hrp, rd)
    X = pr.X
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(hrp.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     x_src = hrp.opt.x_src)
    assert_clustering_universe(clr, size(X, 2))
    r = factory(hrp.r, pr, hrp.opt.slv)
    wu = Matrix{T}(undef, size(X, 2), 2)
    rku = unitary_expected_risks(r, X, cfees)
    wb = weight_bounds_constraints(hrp.opt.wb, hrp.opt.sets; N = size(X, 2),
                                   strict = hrp.opt.strict, datatype = T)
    w = ones(T, size(X, 2))
    items = [clr.res.order]
    while length(items) > 0
        items = [i[j:k] for i in items
                 for (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i in 1:2:length(items)
            fill!(wu, zero(T))
            lc = items[i]
            rc = items[i + 1]
            wu[lc, 1] .= inv.(view(rku, lc))
            wu[lc, 1] ./= sum(view(wu, lc, 1))
            wu[rc, 2] .= inv.(view(rku, rc))
            wu[rc, 2] ./= sum(view(wu, rc, 2))
            lrisk = expected_risk(r, view(wu, :, 1), X, cfees)
            rrisk = expected_risk(r, view(wu, :, 2), X, cfees)
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
                                                                retcode = retcode, w = w,
                                                                imsk = imsk), r = r,
                                        sca = hrp.sca, fb = nothing)
end
"""
    hrp_scalarised_risk(sca::Scalariser, wu::MatNum, rkus::VecVecNum, lc::VecNum,
                        rc::VecNum, rs::VecOptRM, X::MatNum,
                        fees::Option{<:Fees}) -> Tuple

Combine several risk measures into the one left and one right risk that a bisection step needs.

Each measure builds its own naive risk parity sub-portfolio for the two halves, because the weights follow that measure's own unitary risks. The scalariser then combines the pairs of the measures into a single pair.

# Mathematical definition

```math
\\begin{align}
\\tilde{w}^{(k)}_i(C) &= \\frac{\\rho_k(\\{i\\})^{-1}}{\\sum_{j \\in C} \\rho_k(\\{j\\})^{-1}} \\quad \\forall\\, i \\in C\\,,\\\\
\\boldsymbol{p}_k &= s_k \\left(\\rho_k\\left(\\tilde{\\boldsymbol{w}}^{(k)}(C_1)\\right), \\rho_k\\left(\\tilde{\\boldsymbol{w}}^{(k)}(C_2)\\right)\\right)\\,,\\\\
\\left(\\tilde{\\rho}(C_1), \\tilde{\\rho}(C_2)\\right) &= \\begin{cases}
\\sum_{k} \\boldsymbol{p}_k & \\text{sum}\\,,\\\\
\\boldsymbol{p}_{k^\\star},\\; k^\\star = \\underset{k}{\\arg\\max}\\, \\boldsymbol{1}^\\intercal \\boldsymbol{p}_k & \\text{max}\\,,\\\\
\\boldsymbol{p}_{k^\\star},\\; k^\\star = \\underset{k}{\\arg\\min}\\, \\boldsymbol{1}^\\intercal \\boldsymbol{p}_k & \\text{min}\\,,\\\\
\\frac{1}{\\gamma} \\log \\sum_{k} \\exp\\left(\\gamma \\, \\boldsymbol{p}_k\\right) & \\text{log-sum-exp}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\rho_k``: Risk measure ``k`` of `rs`.
  - ``\\rho_k(\\{i\\})``: Risk of asset ``i`` held alone under ``\\rho_k``, the entry ``i`` of `rkus[k]`.
  - ``\\tilde{\\boldsymbol{w}}^{(k)}(C)``: Naive risk parity weights of ``\\rho_k`` inside part ``C``, zero outside it.
  - ``s_k``: The scale `settings.scale` of ``\\rho_k``.
  - ``\\boldsymbol{p}_k``: The scaled pair of risks of the two halves under ``\\rho_k``.
  - $(math_dict[:C_halves_hier])
  - ``\\gamma``: The parameter `gamma` of a [`LogSumExpScalariser`](@ref). The logarithm and the exponential act on each entry of the pair.

The four cases are [`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref) and [`LogSumExpScalariser`](@ref). A maximum or a minimum selects one measure by the total of its pair and returns that measure's own pair. It does not take the maximum or the minimum of each entry.

# Arguments

  - `sca`: Scalarisation strategy.
  - `wu`: Scratch weight matrix, of size `size(X, 2)` by two. It is refilled per measure.
  - `rkus`: The unitary risks of each measure, from [`unitary_expected_risks`](@ref). `rkus[k]` belongs to `rs[k]`, and has length `size(X, 2)`.
  - `lc`: Asset indices of the left half.
  - `rc`: Asset indices of the right half.
  - `rs`: Vector of risk measures, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees, or `nothing`.

# Returns

  - `(lrisk, rrisk)::Tuple`: The combined left and right risk.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`scalarise`](@ref)
  - [`unitary_expected_risks`](@ref)
  - [`herc_scalarised_risk_i!`](@ref)
"""
function hrp_scalarised_risk(sca::Scalariser, wu::MatNum, rkus::VecVecNum, lc::VecNum,
                             rc::VecNum, rs::VecOptRM, X::MatNum, fees::Option{<:Fees})
    return scalarise(sca, zip(rs, rkus); by = sum) do (r, rku)
        fill!(wu, zero(eltype(wu)))
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

[`optimise`](@ref) calls this method. It runs the steps of the single-measure method, with two changes.

# Algorithm

 1. Run steps 1 to 8 of the single-measure method. `r` is now a vector of resolved measures.
 2. Compute the unitary risks of each measure once, `rkus`, with [`unitary_expected_risks`](@ref). They do not depend on the split.
 3. Run steps 10 to 12 of the single-measure method.
 4. For each pair of halves `lc` and `rc`, combine the measures into one pair `lrisk` and `rrisk`, with [`hrp_scalarised_risk`](@ref) and `sca`.
 5. Run steps 14 to 17 of the single-measure method.

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
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset, and a carrier keyed by name cannot resolve at all once its `w` sits on the
    # complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    # A weight is a quotient of two risks, so an integer sample takes a float weight type.
    T = typeof(one(eltype(pr.X)) / one(eltype(pr.X)))
    imsk = investable_mask(pr)
    fees = investable_fees_view(fees_constraints(hrp.opt.fees, hrp.opt.sets;
                                                 strict = hrp.opt.strict, datatype = T),
                                imsk, pr.X)
    # A forced exit is charged once, against the full-universe weight vector the fit
    # rebuilds, so it rides on the result alone. No sub-problem below holds that vector —
    # the exiting asset is in no cluster, its column being `NaN` — so none prices an exit.
    cfees = strip_liquidation_carriers(fees, nothing)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here:
    # the distance the clustering is built from never sees a `NaN`, and the cluster count
    # is chosen on the investable universe. The weights are expanded back in
    # `HierarchicalResult`.
    _, pr, hrp, rd = investable_reduction(imsk, pr, hrp, rd)
    X = pr.X
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(hrp.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     x_src = hrp.opt.x_src)
    assert_clustering_universe(clr, size(X, 2))
    r = factory(hrp.r, pr, hrp.opt.slv)
    wu = Matrix{T}(undef, size(X, 2), 2)
    rkus = [unitary_expected_risks(ri, X, cfees) for ri in r]
    wb = weight_bounds_constraints(hrp.opt.wb, hrp.opt.sets; N = size(X, 2),
                                   strict = hrp.opt.strict, datatype = T)
    w = ones(T, size(X, 2))
    items = [clr.res.order]
    while length(items) > 0
        items = [i[j:k] for i in items
                 for (j, k) in ((1, div(length(i), 2)), (1 + div(length(i), 2), length(i)))
                 if length(i) > 1]
        for i in 1:2:length(items)
            lc = items[i]
            rc = items[i + 1]
            lrisk, rrisk = hrp_scalarised_risk(hrp.sca, wu, rkus, lc, rc, r, X, cfees)
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
                                                                retcode = retcode, w = w,
                                                                imsk = imsk), r = r,
                                        sca = hrp.sca, fb = nothing)
end
"""
    optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
             rd::ReturnsResult; dims::Int = 1, kwargs...) -> HierarchicalRiskParityResult

Run the Hierarchical Risk Parity portfolio optimisation.

Unlike [`HierarchicalEqualRiskContribution`](@ref) and [`NestedClustered`](@ref), this optimiser takes no `branchorder` keyword. Recursive bisection splits the leaf order of the dendrogram, so that order is an input of the allocation, and the clusterisation always keeps the optimal order.

# Arguments

  - `hrp`: The hierarchical risk parity optimiser to use.
  - $(arg_dict[:rd]) When `hrp.opt.pe` holds a prior result, the allocation reads no returns from `rd`, but a fallback or the clusterisation can.
  - `dims`: The dimension along which observations advance in time.
  - `kwargs`: Passed to [`_optimise`](@ref), which ignores them. A `branchorder` passed here has no effect.

# Validation

  - No field in the tree of `hrp` holds an [`Online`](@ref). The method throws an `ArgumentError` that names the field otherwise, through [`assert_batch_entry`](@ref). A plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the online arm of the fold loop.

# Related

  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
"""
function optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
    assert_batch_entry(hrp, "`optimise`")
    return _optimise(hrp, rd; dims = dims, kwargs...)
end

@pipe_delegates HierarchicalRiskParity opt
@pipe_route_sigma_ucs HierarchicalRiskParity
export HierarchicalRiskParity
