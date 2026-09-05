"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`HierarchicalEqualRiskContribution`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function herc_td_defaults()::NamedTuple
    return (; ri = Variance(), ro = Variance(), scai = SumScalariser(),
            scao = SumScalariser())
end
"""
$(DocStringExtensions.TYPEDEF)

Splits weight down the dendrogram between clusters by their outer risk `ro`, then splits each cluster's share between its assets by their inner risk `ri`.

This is the Hierarchical Equal Risk Contribution algorithm. It differs from [`HierarchicalRiskParity`](@ref) in two ways: it stops at the optimal number of clusters rather than at the individual assets, and it follows the dendrogram's own branch structure rather than halving a leaf order. The name is the source's; the resulting portfolio is **not** a risk parity portfolio, and the clusters do not contribute equally to risk.

# Mathematical definition

Cut the dendrogram at `opt.cle`'s optimal number of clusters ``K``. Every asset starts at ``w_i = 1``. The algorithm then runs two independent allocations.

**Inside a cluster**, each asset takes a share in inverse proportion to its own inner risk:

```math
\\begin{align}
\\rho_i(\\{j\\}) &= \\textrm{risk of asset } j \\textrm{ held alone under } \\rho_i\\,,\\\\
w_j &= \\frac{\\rho_i(\\{j\\})^{-1}}{\\sum_{l \\in C_k} \\rho_i(\\{l\\})^{-1}} \\quad \\forall\\, j \\in C_k\\,.
\\end{align}
```

**Between clusters**, the algorithm walks the ``K - 1`` internal nodes of the dendrogram in order of falling height. At each node it splits the weight of everything below that node between its two branches:

```math
\\begin{align}
\\tilde{w}_j(C_k) &= \\frac{\\rho_o(\\{j\\})^{-1}}{\\sum_{l \\in C_k} \\rho_o(\\{l\\})^{-1}} \\quad \\forall\\, j \\in C_k\\,,\\\\
\\tilde{\\rho}_o(C_k) &= \\rho_o\\left(\\tilde{\\boldsymbol{w}}(C_k)\\right)\\,,\\\\
\\alpha &= \\frac{\\sum_{C_k \\subseteq B_2} \\tilde{\\rho}_o(C_k)}{\\sum_{C_k \\subseteq B_1} \\tilde{\\rho}_o(C_k) + \\sum_{C_k \\subseteq B_2} \\tilde{\\rho}_o(C_k)}\\,,\\\\
\\boldsymbol{w}_{B_1} &\\leftarrow \\alpha \\, \\boldsymbol{w}_{B_1}\\,,\\\\
\\boldsymbol{w}_{B_2} &\\leftarrow (1 - \\alpha) \\, \\boldsymbol{w}_{B_2}\\,.
\\end{align}
```

Where:

  - ``\\rho_i``, ``\\rho_o``: The inner measure `ri` and the outer measure `ro`, resolved by [`factory`](@ref).
  - ``C_k``: The ``k``-th cluster. ``K`` of them partition the universe.
  - ``B_1``, ``B_2``: The left and right branches of the dendrogram node being split. Each is a union of whole clusters.
  - ``\\tilde{\\boldsymbol{w}}(C_k)``: Naive risk parity weights inside cluster ``C_k`` under ``\\rho_o``, zero outside it.
  - ``\\tilde{\\rho}_o(C_k)``: Risk of that sub-portfolio, the cluster's contribution to a branch's risk.
  - ``\\alpha``: Fraction of the node's weight that goes to ``B_1``.
  - ``w_j``: Weight of asset ``j``. The final vector is normalised to sum to one.

The two allocations multiply: an asset's final weight is its share inside its cluster, times every split factor on the path from the root to that cluster. **This is not a flat inverse-risk allocation over the ``K`` clusters.** A recursion of binary splits agrees with one only when ``K = 2``. On a twelve-asset sample that clusters into three, the recursion gives cluster weights ``[0.7024, 0.1603, 0.1373]`` where a flat allocation would give ``[0.3697, 0.3394, 0.2908]``.

When `ri` or `ro` is a vector, [`herc_scalarised_risk_i!`](@ref) and [`herc_scalarised_risk_o!`](@ref) combine the measures with `scai` and `scao` before the weights are formed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalEqualRiskContribution(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        ri::TD{<:OptRM_VecOptRM} = Variance(),
        ro::TD{<:OptRM_VecOptRM} = ri,
        scai::TD{<:Scalariser} = SumScalariser(),
        scao::TD{<:Scalariser} = scai,
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> HierarchicalEqualRiskContribution

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the inner/outer risk measures, their scalarisers and the fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `fb`). The executor `ex` is execution control and stays static.

## Validation

  - If `ri` or `ro` is a vector: `!isempty(ri)` / `!isempty(ro)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `ri`: Recursively updated via [`factory`](@ref).
  - `ro`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`HierarchicalEqualRiskContribution` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` already holds a prior **result**, the method replaces `X` with `opt.pe.X`, so the children are viewed against the prior's own observations rather than the caller's matrix.
  - `ri` and `ro` recurse through [`port_opt_view`](@ref) with that matrix. `opt` recurses with the index alone.
  - When `ri` and `ro` are the **same object**, the method views it once and shares the result, so the two fields stay aliased on the subset as they were on the whole universe.
  - `scai`, `scao`, `ex` and `fb` are carried through unchanged.

# Examples

```jldoctest
julia> HierarchicalEqualRiskContribution()
HierarchicalEqualRiskContribution
   opt ┼ HierarchicalOptimiser
       │       pe ┼ EmpiricalPrior
       │          │        ce ┼ PortfolioOptimisersCovariance
       │          │           │   ce ┼ Covariance
       │          │           │      │    me ┼ SimpleExpectedReturns
       │          │           │      │       │   w ┴ nothing
       │          │           │      │    ce ┼ GeneralCovariance
       │          │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
       │          │           │      │       │    w ┴ nothing
       │          │           │      │   alg ┼ FullMoment()
       │          │           │      │     w ┴ nothing
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
       │    z_src ┼ Symbol: :data
       │   strict ┴ Bool: false
    ri ┼ Variance
       │   settings ┼ RiskMeasureSettings
       │            │   scale ┼ Float64: 1.0
       │            │      ub ┼ nothing
       │            │     rke ┴ Bool: true
       │      sigma ┼ nothing
       │       chol ┼ nothing
       │         rc ┼ nothing
       │        alg ┴ SquaredSOCRiskExpr()
    ro ┼ Variance
       │   settings ┼ RiskMeasureSettings
       │            │   scale ┼ Float64: 1.0
       │            │      ub ┼ nothing
       │            │     rke ┴ Bool: true
       │      sigma ┼ nothing
       │       chol ┼ nothing
       │         rc ┼ nothing
       │        alg ┴ SquaredSOCRiskExpr()
  scai ┼ SumScalariser()
  scao ┼ SumScalariser()
    ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
    fb ┴ nothing
```

# Related

  - [`optimise`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`HierarchicalOptimiser`](@ref)
  - [`unitary_expected_risks`](@ref)
  - [`herc_risk`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:raffinot2017])
  - $(ref_dict[:raffinot2018])
  - $(ref_dict[:cajas2025]) Section 12.2.
"""
@propagatable @concrete struct HierarchicalEqualRiskContribution <:
                               ClusteringOptimisationEstimator
    """
    $(field_dict[:opt_hier])
    """
    @fprop opt
    """
    $(field_dict[:ri])
    """
    @fprop ri
    """
    $(field_dict[:ro])
    """
    @fprop ro
    """
    $(field_dict[:scai])
    """
    scai
    """
    $(field_dict[:scao])
    """
    scao
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function HierarchicalEqualRiskContribution(opt::HierarchicalOptimiser,
                                               ri::TD{<:OptRM_VecOptRM},
                                               ro::TD{<:OptRM_VecOptRM},
                                               scai::TD{<:Scalariser},
                                               scao::TD{<:Scalariser},
                                               ex::FLoops.Transducers.Executor,
                                               fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb,
                                                  :HierarchicalEqualRiskContribution)
        if isa(ri, AbstractVector)
            @argcheck(!isempty(ri), IsEmptyError("ri cannot be empty"))
        end
        if isa(ro, AbstractVector)
            @argcheck(!isempty(ro), IsEmptyError("ro cannot be empty"))
        end
        assert_risk_measure_required(ri, :HierarchicalEqualRiskContribution)
        assert_risk_measure_required(ro, :HierarchicalEqualRiskContribution)
        assert_time_dependent_substitution(HierarchicalEqualRiskContribution,
                                           (; opt, ri, ro, scai, scao, ex, fb),
                                           herc_td_defaults())
        return new{typeof(opt), typeof(ri), typeof(ro), typeof(scai), typeof(scao),
                   typeof(ex), typeof(fb)}(opt, ri, ro, scai, scao, ex, fb)
    end
end
function HierarchicalEqualRiskContribution(;
                                           opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                           ri::TD{<:OptRM_VecOptRM} = Variance(),
                                           ro::TD{<:OptRM_VecOptRM} = ri,
                                           scai::TD{<:Scalariser} = SumScalariser(),
                                           scao::TD{<:Scalariser} = scai,
                                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                           fb::TDO_Option{<:OptE_Opt} = nothing)::HierarchicalEqualRiskContribution
    return HierarchicalEqualRiskContribution(opt, ri, ro, scai, scao, ex, fb)
end
function time_dependent_field_defaults(::HierarchicalEqualRiskContribution)::NamedTuple
    return herc_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`HierarchicalEqualRiskContribution`](@ref) requires previous portfolio weights.

Returns `true` if any of the base optimiser, inner/outer risk measures, or fallback require previous weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
function needs_previous_weights(opt::HierarchicalEqualRiskContribution)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.ri) ||
            needs_previous_weights(opt.ro) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of [`HierarchicalEqualRiskContribution`](@ref) `hec` sliced to asset indices `i`.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(hec::HierarchicalEqualRiskContribution, i, X::MatNum,
                       args...)::HierarchicalEqualRiskContribution
    X = isa(hec.opt.pe, AbstractPriorResult) ? hec.opt.pe.X : X
    ri = hec.ri
    ro = hec.ro
    if ri === ro
        ri = ro = port_opt_view(ri, i, X)
    else
        ri = port_opt_view(ri, i, X)
        ro = port_opt_view(ro, i, X)
    end
    opt = port_opt_view(hec.opt, i)
    return HierarchicalEqualRiskContribution(; ri = ri, ro = ro, opt = opt, scai = hec.scai,
                                             scao = hec.scao, ex = hec.ex, fb = hec.fb)
end
"""
    herc_scalarised_risk_o!(sca::Scalariser, wk::VecNum, roku::VecNum_MatNum,
                            rkbo::VecNum, cl::VecInt, ros::VecOptRM, X::MatNum,
                            fees::Option{<:Fees}) -> Number

Combine several outer measures into the one risk that cluster `cl` contributes to its branch.

Each measure builds its own naive risk parity sub-portfolio over `cl`, because the weights follow that measure's own unitary risks. The scalariser then combines the per-measure risks into one number.

# Arguments

  - `sca`: Scalarisation strategy ([`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref), or [`LogSumExpScalariser`](@ref)).
  - `wk`: Scratch weight vector for [`unitary_expected_risks!`](@ref), of length `size(X, 2)`.
  - `roku`: Unitary outer risk buffer. A vector is overwritten per measure; a matrix already caches column `i` per measure, see [`herc_unitary_risks_o!`](@ref).
  - `rkbo`: Weight buffer, written in place over the entries of `cl`. The caller zeroes those entries again after the split.
  - `cl`: Asset indices of the cluster.
  - `ros`: Vector of outer risk measures, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees, or `nothing`.

# Returns

  - `risk::Number`: The combined outer risk of the cluster.

# Details

  - Each measure's contribution is multiplied by its own `settings.scale` before the scalariser sees it.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_scalarised_risk_i!`](@ref)
  - [`herc_unitary_risks_o!`](@ref)
  - [`unitary_expected_risks!`](@ref)
"""
function herc_scalarised_risk_o!(sca::Scalariser, wk::VecNum, roku::VecNum_MatNum,
                                 rkbo::VecNum, cl::VecInt, ros::VecOptRM, X::MatNum,
                                 fees::Option{<:Fees})
    return scalarise(sca, pairs(ros)) do (i, ro)
        rokui = herc_unitary_risks_o!(wk, roku, i, ro, X, fees)
        rkbo[cl] .= inv.(view(rokui, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        return ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
end
"""
    herc_unitary_risks_i!(wk, rku, i, r, X, fees)

Return the unitary risk vector for inner measure `i` of a HERC scalarised risk computation.

When `rku` is a vector it is a shared buffer and the unitary risks for measure `r` are recomputed into it. When `rku` is a matrix, the unitary risks are computed into column `i`, caching them per measure for reuse by [`herc_scalarised_risk_o!`](@ref) when the inner and outer measures are identical.

# Related

  - [`herc_scalarised_risk_i!`](@ref)
  - [`herc_unitary_risks_o!`](@ref)
  - [`unitary_expected_risks!`](@ref)
"""
function herc_unitary_risks_i!(wk::VecNum, rku::VecNum, ::Any, r::OptimisationRiskMeasure,
                               X::MatNum, fees::Option{<:Fees})
    unitary_expected_risks!(wk, rku, r, X, fees)
    return rku
end
function herc_unitary_risks_i!(wk::VecNum, rku::MatNum, i::Any, r::OptimisationRiskMeasure,
                               X::MatNum, fees::Option{<:Fees})
    rkui = view(rku, :, i)
    unitary_expected_risks!(wk, rkui, r, X, fees)
    return rkui
end
"""
    herc_unitary_risks_o!(wk, rku, i, r, X, fees)

Return the unitary risk vector for outer measure `i` of a HERC scalarised risk computation.

When `rku` is a vector it is a shared buffer and the unitary risks for measure `r` are recomputed into it. When `rku` is a matrix, column `i` already caches measure `i`'s unitary risks (filled by [`herc_scalarised_risk_i!`](@ref) when the inner and outer measures are identical) and is returned as-is.

# Related

  - [`herc_scalarised_risk_o!`](@ref)
  - [`herc_unitary_risks_i!`](@ref)
  - [`unitary_expected_risks!`](@ref)
"""
function herc_unitary_risks_o!(wk::VecNum, rku::VecNum, ::Any, r::OptimisationRiskMeasure,
                               X::MatNum, fees::Option{<:Fees})
    unitary_expected_risks!(wk, rku, r, X, fees)
    return rku
end
function herc_unitary_risks_o!(::VecNum, rku::MatNum, i::Any, ::OptimisationRiskMeasure,
                               ::MatNum, ::Option{<:Fees})
    return view(rku, :, i)
end
"""
    herc_scalarised_risk_i!(sca::Scalariser, wk::VecNum, riku::VecNum_MatNum,
                            cl::VecInt, ris::VecOptRM, X::MatNum,
                            fees::Option{<:Fees}) -> VecNum

Combine several inner measures into the one weight vector that shares a cluster's weight between its assets.

# Arguments

  - `sca`: Scalarisation strategy ([`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref), or [`LogSumExpScalariser`](@ref)).
  - `wk`: Scratch weight vector for [`unitary_expected_risks!`](@ref), of length `size(X, 2)`.
  - `riku`: Unitary inner risk buffer. A vector is overwritten per measure; a matrix caches column `i` per measure, see [`herc_unitary_risks_i!`](@ref).
  - `cl`: Asset indices of the cluster.
  - `ris`: Vector of inner risk measures, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees, or `nothing`.

# Returns

  - `w::VecNum`: The intra-cluster weights, of length `length(cl)`. They sum to one under [`MaxScalariser`](@ref) and [`MinScalariser`](@ref), and to the total of the measures' `settings.scale` under [`SumScalariser`](@ref). [`LogSumExpScalariser`](@ref) gives a total that varies with the cluster, but only in the far decimals: two measures at unit scale gave **3.772618**, **3.772634** and **3.772624** on the three clusters of a twelve-asset sample. `_optimise` normalises the whole weight vector at the end, so a total that is the same for every cluster cancels, and the three scalarisers put the same weight on each cluster to **1e-6**.

# Details

  - The two methods **normalise at different points**, and the difference is deliberate. [`SumScalariser`](@ref) and [`LogSumExpScalariser`](@ref) normalise each measure's inverse-risk vector first and combine the resulting weight vectors. [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) select one measure — the one whose total scaled risk over `cl` is largest or smallest — and normalise that measure's vector alone, because selecting between already-normalised vectors would compare quantities that all sum to one.
  - Each measure's contribution is multiplied by its own `settings.scale`.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_scalarised_risk_o!`](@ref)
  - [`herc_unitary_risks_i!`](@ref)
  - [`unitary_expected_risks!`](@ref)
"""
function herc_scalarised_risk_i!(sca::Union{SumScalariser, LogSumExpScalariser}, wk::VecNum,
                                 riku::VecNum_MatNum, cl::VecInt, ris::VecOptRM, X::MatNum,
                                 fees::Option{<:Fees})
    return scalarise(sca, pairs(ris)) do (i, ri)
        rikui = herc_unitary_risks_i!(wk, riku, i, ri, X, fees)
        risk = inv.(view(rikui, cl))
        risk ./= sum(risk)
        return ri.settings.scale * risk
    end
end
function herc_scalarised_risk_i!(sca::Union{MaxScalariser, MinScalariser}, wk::VecNum,
                                 riku::VecNum_MatNum, cl::VecInt, ris::VecOptRM, X::MatNum,
                                 fees::Option{<:Fees})
    risk = scalarise(sca, pairs(ris); by = first) do (i, ri)
        rikui = herc_unitary_risks_i!(wk, riku, i, ri, X, fees)
        rk = ri.settings.scale * view(rikui, cl)
        return (sum(rk), rk)
    end[2]
    risk = inv.(risk)
    risk ./= sum(risk)
    return risk
end
"""
    herc_risk(hec::HierarchicalEqualRiskContribution, pr::AbstractPriorResult,
              cls::VecVecInt) -> Tuple

Compute the intra-cluster weights and the per-cluster outer risks that HERC allocates with.

# Arguments

  - `hec`: The optimiser. Its `ri`/`ro` arity and its executor `ex` select the method.
  - `pr`: Prior result. Its `X` is the return matrix and its moments resolve the measures.
  - `cls`: Asset indices of each cluster, one entry per cluster.

# Returns

  - `(w, rkcl, fees, ri, ro)::Tuple`: The intra-cluster weights over the whole universe, each cluster's outer risk, the resolved fees, and the two resolved risk measures.

# Details

  - Eight methods cover the four arity pairs of `ri` and `ro` against a sequential or a parallel executor. A parallel executor gets one buffer column per cluster, so no two iterations write the same entry.
  - When `hec.ri === hec.ro` the measure is resolved once and the unitary risks are computed once, because the two allocations then read the same vector.
  - `w` covers the whole universe. Each cluster's entries sum to one, so a cluster's share still has to be applied on top.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_scalarised_risk_i!`](@ref)
  - [`herc_scalarised_risk_o!`](@ref)
  - [`unitary_expected_risks`](@ref)
"""
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any, <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    riku = unitary_expected_risks(ri, X, fees)
    if hec.ri === hec.ro
        ro = ri
        roku = riku
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
        roku = unitary_expected_risks(ro, X, fees)
    end
    rkbo = zeros(eltype(X), size(X, 2))
    rkcl = Vector{eltype(X)}(undef, length(cls))
    w = Vector{eltype(X)}(undef, size(X, 2))
    let
        roku_i, ro_i = roku, ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] .= inv.(view(riku, cl))
            w[cl] ./= sum(view(w, cl))
            rkbo[cl] .= inv.(view(roku_i, cl))
            rkbo[cl] ./= sum(view(rkbo, cl))
            rkcl[i] = expected_risk(ro_i, rkbo, X, fees)
            rkbo[cl] .= zero(eltype(X))
        end
    end
    return w, rkcl, fees, ri, ro
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    riku = unitary_expected_risks(ri, X, fees)
    if hec.ri === hec.ro
        ro = ri
        roku = riku
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
        roku = unitary_expected_risks(ro, X, fees)
    end
    Nc = length(cls)
    rkbo = zeros(eltype(X), size(X, 2), Nc)
    rkcl = Vector{eltype(X)}(undef, Nc)
    w = Vector{eltype(X)}(undef, size(X, 2))
    let
        roku_i, ro_i = roku, ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] .= inv.(view(riku, cl))
            w[cl] ./= sum(view(w, cl))
            rkbo[cl, i] .= inv.(view(roku_i, cl))
            rkbo[cl, i] ./= sum(view(rkbo, cl, i))
            rkcl[i] = expected_risk(ro_i, view(rkbo, :, i), X, fees)
        end
    end
    return w, rkcl, fees, ri, ro
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM, <:VecOptRM,
                                                          <:Any, <:Any,
                                                          <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    if hec.ri === hec.ro
        ro = ri
        rku = zeros(eltype(X), size(X, 2), length(ri))
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
        rku = Vector{eltype(X)}(undef, size(X, 2))
    end
    rkcl = Vector{eltype(X)}(undef, length(cls))
    w = Vector{eltype(X)}(undef, size(X, 2))
    wk = zeros(eltype(X), size(X, 2))
    rkbo = zeros(eltype(X), size(X, 2))
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    let
        rku_i, ro_i = rku, ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] = herc_scalarised_risk_i!(hec.scai, wk, rku_i, cl, ri, X, fees)
            rkcl[i] = herc_scalarised_risk_o!(hec.scao, wk, rku_i, rkbo, cl, ro_i, X, fees)
            rkbo[cl] .= zero(eltype(X))
        end
    end
    return w, rkcl, fees, ri, ro
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM, <:VecOptRM,
                                                          <:Any, <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    if hec.ri === hec.ro
        ro = ri
    else
        ro = factory(hec.ro, pr, hec.opt.slv)
    end
    Nc = length(cls)
    rkcl = Vector{eltype(X)}(undef, Nc)
    w = Vector{eltype(X)}(undef, size(X, 2))
    rku = Matrix{eltype(X)}(undef, size(X, 2), Nc)
    wk = zeros(eltype(X), size(X, 2), Nc)
    rkbo = zeros(eltype(X), size(X, 2), Nc)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    let
        ro_i = ro
        FLoops.@floop hec.ex for (i, cl) in pairs(cls)
            w[cl] = herc_scalarised_risk_i!(hec.scai, view(wk, :, i), view(rku, :, i), cl,
                                            ri, X, fees)
            rkcl[i] = herc_scalarised_risk_o!(hec.scao, view(wk, :, i), view(rku, :, i),
                                              view(rkbo, :, i), cl, ro_i, X, fees)
        end
    end
    return w, rkcl, fees, ri, ro
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:VecOptRM, <:Any, <:Any,
                                                          <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    riku = unitary_expected_risks(ri, X, fees)
    ro = factory(hec.ro, pr, hec.opt.slv)
    rkcl = Vector{eltype(X)}(undef, length(cls))
    w = Vector{eltype(X)}(undef, size(X, 2))
    roku = Vector{eltype(X)}(undef, size(X, 2))
    wk = zeros(eltype(X), size(X, 2))
    rkbo = zeros(eltype(X), size(X, 2))
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkcl[i] = herc_scalarised_risk_o!(hec.scao, wk, roku, rkbo, cl, ro, X, fees)
        rkbo[cl] .= zero(eltype(X))
    end
    return w, rkcl, fees, ri, ro
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:OptimisationRiskMeasure,
                                                          <:VecOptRM, <:Any, <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    riku = unitary_expected_risks(ri, X, fees)
    ro = factory(hec.ro, pr, hec.opt.slv)
    Nc = length(cls)
    rkcl = Vector{eltype(X)}(undef, Nc)
    w = Vector{eltype(X)}(undef, size(X, 2))
    roku = Matrix{eltype(X)}(undef, size(X, 2), Nc)
    wk = zeros(eltype(X), size(X, 2), Nc)
    rkbo = zeros(eltype(X), size(X, 2), Nc)
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] .= inv.(view(riku, cl))
        w[cl] ./= sum(view(w, cl))
        rkcl[i] = herc_scalarised_risk_o!(hec.scao, view(wk, :, i), view(roku, :, i),
                                          view(rkbo, :, i), cl, ro, X, fees)
    end
    return w, rkcl, fees, ri, ro
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any, <:FLoops.SequentialEx},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    ro = factory(hec.ro, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    roku = unitary_expected_risks(ro, X, fees)
    rkcl = Vector{eltype(X)}(undef, length(cls))
    w = Vector{eltype(X)}(undef, size(X, 2))
    wk = zeros(eltype(X), size(X, 2))
    riku = Vector{eltype(X)}(undef, size(X, 2))
    rkbo = zeros(eltype(X), size(X, 2))
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] = herc_scalarised_risk_i!(hec.scai, wk, riku, cl, ri, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        rkcl[i] = expected_risk(ro, rkbo, X, fees)
        rkbo[cl] .= zero(eltype(X))
    end
    return w, rkcl, fees, ri, ro
end
function herc_risk(hec::HierarchicalEqualRiskContribution{<:Any, <:VecOptRM,
                                                          <:OptimisationRiskMeasure, <:Any,
                                                          <:Any,
                                                          <:FLoops.Transducers.Executor},
                   pr::AbstractPriorResult, cls::VecVecInt)
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    ro = factory(hec.ro, pr, hec.opt.slv)
    fees = fees_constraints(hec.opt.fees, hec.opt.sets; strict = hec.opt.strict,
                            datatype = eltype(X))
    roku = unitary_expected_risks(ro, X, fees)
    Nc = length(cls)
    rkcl = Vector{eltype(X)}(undef, Nc)
    w = Vector{eltype(X)}(undef, size(X, 2))
    wk = zeros(eltype(X), size(X, 2), Nc)
    riku = Matrix{eltype(X)}(undef, size(X, 2), Nc)
    rkbo = zeros(eltype(X), size(X, 2), Nc)
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] = herc_scalarised_risk_i!(hec.scai, view(wk, :, i), view(riku, :, i), cl, ri,
                                        X, fees)
        rkbo[cl, i] .= inv.(view(roku, cl))
        rkbo[cl, i] ./= sum(view(rkbo, cl, i))
        rkcl[i] = expected_risk(ro, view(rkbo, :, i), X, fees)
    end
    return w, rkcl, fees, ri, ro
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Internal implementation of Hierarchical Equal Risk Contribution optimisation.

Clusters assets, computes intra- and inter-cluster risk contributions, and allocates weights via the bisection method along the dendrogram.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`optimise`](@ref)
"""
function _optimise(hec::HierarchicalEqualRiskContribution,
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   branchorder::Symbol = :optimal, kwargs...)
    hec = reset_time_dependent_estimator(hec)
    rd = returns_result_picker(rd, hec.opt.brt)
    pr = prior(hec.opt.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(hec.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     branchorder = branchorder, x_src = hec.opt.x_src,
                     z_src = hec.opt.z_src)
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    w, rkcl, fees, ri, ro = herc_risk(hec, pr, cls)
    nd = to_tree(clr.res)[2]
    hs = [i.height for i in nd]
    nd = nd[sortperm(hs; rev = true)]
    # Treat each cluster as its own portfolio and optimise each one individually.
    # Calculate the weight of each cluster relative to the other clusters.
    lc = Vector{Int}(undef, 0)
    rc = Vector{Int}(undef, 0)
    for i in nd[1:(clr.k - 1)]
        if is_leaf(i)
            continue
        end
        # Do this recursively accounting for the dendrogram structure.
        ln = pre_order(i.left)
        rn = pre_order(i.right)
        for (i, cl) in pairs(cls)
            if issubset(cl, ln)
                push!(lc, i)
            elseif issubset(cl, rn)
                push!(rc, i)
            end
        end
        # Cluster risk contributions.
        lrisk = sum(view(rkcl, lc))
        rrisk = sum(view(rkcl, rc))
        risk = lrisk + rrisk
        alpha = one(lrisk) - lrisk / risk
        # Asset risk contribution times cluster risk contribution.
        w[ln] *= alpha
        w[rn] *= one(alpha) - alpha
        empty!(lc)
        empty!(rc)
    end
    wb = weight_bounds_constraints(hec.opt.wb, hec.opt.sets; N = length(w),
                                   strict = hec.opt.strict, datatype = eltype(X))
    retcode, w = finalise_weight_bounds(hec.opt.wf, wb, w / sum(w))
    return HierarchicalEqualRiskContributionResult(;
                                                   hr = HierarchicalResult(; pr = pr,
                                                                           clr = clr,
                                                                           wb = wb,
                                                                           fees = fees,
                                                                           retcode = retcode,
                                                                           w = w), ri = ri,
                                                   ro = ro, scai = hec.scai,
                                                   scao = hec.scao, fb = nothing)
end
"""
    optimise(hec::HierarchicalEqualRiskContribution{
                     <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing
                 },
            rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
            branchorder::Symbol = :optimal, kwargs...) -> HierarchicalEqualRiskContributionResult

Run the Hierarchical Equal Risk Contribution portfolio optimisation.

# Arguments

  - `hec`: The hierarchical equal risk contribution optimiser to use.
  - $(arg_dict[:rd]) If `isa(hec.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `branchorder`: The branch order to use for the clusterisation, this optimisation can use non-optimal branch orders, which make the clustering faster but the dendrogram won't be as nice.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
"""
function optimise(hec::HierarchicalEqualRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, kwargs...)
    return _optimise(hec, rd; dims = dims, branchorder = branchorder, kwargs...)
end

export HierarchicalEqualRiskContribution
