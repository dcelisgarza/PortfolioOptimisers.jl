"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`HierarchicalEqualRiskContribution`](@ref) fields that may hold a [`TimeDependent`](@ref).

The constructor's substitution check and [`time_dependent_field_defaults`](@ref) both read it, so the file states the fold-less value of each field once. A field whose static default is `nothing` has no entry.

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

This is the Hierarchical Equal Risk Contribution algorithm. It differs from [`HierarchicalRiskParity`](@ref) in two ways. It stops at the optimal number of clusters, not at the individual assets, and it splits at the branches of the dendrogram, not at the halves of a leaf order. The name is the source's. The portfolio is not a risk parity portfolio, and the clusters do not contribute equally to risk. It solves no optimisation problem, so it accepts a risk measure that is not convex. The weight finaliser of `opt` acts on the result, and [`_optimise`](@ref) states the steps.

# Mathematical definition

Cut the dendrogram into ``K`` clusters. The ``K - 1`` nodes of the dendrogram above the cut each split their weight between their two branches, and every cluster lies wholly in one branch of each node above it:

```math
\\begin{align}
\\tilde{w}_i(C_k) &= \\frac{\\rho_o(\\{i\\})^{-1}}{\\sum_{l \\in C_k} \\rho_o(\\{l\\})^{-1}} \\quad \\forall\\, i \\in C_k\\,,\\\\
\\tilde{\\rho}_o(C_k) &= \\rho_o\\left(\\tilde{\\boldsymbol{w}}(C_k)\\right)\\,,\\\\
\\alpha_n &= \\frac{\\sum_{C_k \\subseteq B_2(n)} \\tilde{\\rho}_o(C_k)}{\\sum_{C_k \\subseteq B_1(n)} \\tilde{\\rho}_o(C_k) + \\sum_{C_k \\subseteq B_2(n)} \\tilde{\\rho}_o(C_k)}\\,,\\\\
W(C_k) &= \\prod_{n \\in \\mathcal{N},\\, C_k \\subseteq B_1(n)} \\alpha_n \\prod_{n \\in \\mathcal{N},\\, C_k \\subseteq B_2(n)} \\left(1 - \\alpha_n\\right)\\,,\\\\
v_i &= \\frac{\\rho_i(\\{i\\})^{-1}}{\\sum_{l \\in C_k} \\rho_i(\\{l\\})^{-1}} \\quad \\forall\\, i \\in C_k\\,,\\\\
w_i &= W(C_k) \\, v_i \\quad \\forall\\, i \\in C_k\\,.
\\end{align}
```

Where:

  - ``\\rho_i``, ``\\rho_o``: The inner measure `ri` and the outer measure `ro`, resolved by [`factory`](@ref) against the prior.
  - ``\\rho(\\{i\\})``: Risk of asset ``i`` held alone under the measure ``\\rho``, from [`unitary_expected_risks`](@ref).
  - ``C_k``: The ``k``-th cluster. The ``K`` clusters partition the universe.
  - ``\\mathcal{N}``: The ``K - 1`` nodes of the dendrogram above the cut.
  - ``B_1(n)``, ``B_2(n)``: The left and right branches of node ``n``. Each is a union of whole clusters.
  - ``\\tilde{\\boldsymbol{w}}(C_k)``: Naive risk parity weights inside cluster ``C_k`` under ``\\rho_o``, zero outside it.
  - ``\\tilde{\\rho}_o(C_k)``: Risk of that naive risk parity sub-portfolio.
  - ``\\alpha_n``: Split factor of node ``n``, the fraction of its weight that goes to ``B_1(n)``.
  - ``W(C_k)``: Weight of cluster ``C_k``, the product of the split factors on the path from the root to ``C_k``.
  - ``v_i``: Share of asset ``i`` in the weight of its cluster.
  - $(math_dict[:w_i_asset])

When `ri` is a vector, [`herc_intra_cluster_weights`](@ref) states the shares ``v_i`` that the scalariser `scai` makes. When `ro` is a vector, [`herc_cluster_risk`](@ref) states the risk ``\\tilde{\\rho}_o(C_k)`` that the scalariser `scao` makes.

The weights of the two branches of a node sum to the weight of the node, and the shares inside a cluster sum to one, so ``\\sum_k W(C_k) = 1`` and ``\\sum_i w_i = 1``. When ``K = 2`` the one split gives each cluster a weight in inverse proportion to its risk. When ``K > 2`` a cluster's weight is a product of split factors, and it is in general not the weight of a flat inverse-risk allocation over the ``K`` clusters. The weights depend on the merge tree and on the cut. They do not depend on the order of the leaves.

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

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value. The inner and outer risk measures, their scalarisers and the fallback are problem definition, so a cross-validation fold loop resolves them per fold. A fold-less `optimise` runs with each at its static default, which is `nothing` for `fb`. The executor `ex` is execution control and stays static.

## Validation

  - If `ri` is a vector: `!isempty(ri)`. If `ro` is a vector: `!isempty(ro)`. The constructor throws an `IsEmptyError` otherwise.
  - `ri` and `ro` hold no [`NoRisk`](@ref), alone or in a vector. The constructor throws an `ArgumentError` otherwise, through [`assert_risk_measure_required`](@ref), because a zero risk leaves the split factor undefined.
  - `fb` schedules: `bind !== :nearest`. The constructor throws an `ArgumentError` otherwise.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `ri`: Recursively updated via [`factory`](@ref).
  - `ro`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`HierarchicalEqualRiskContribution` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` holds a prior result, the method replaces `X` with `opt.pe.X`, so it views the children against the observations of that prior, not against the caller's matrix.
  - `ri`, `ro` and `opt` recurse through [`port_opt_view`](@ref) with that matrix.
  - When `ri` and `ro` are the same object, the method views it once and gives the view to both fields, so the two fields stay one object on the subset.
  - `fb` recurses through [`view_child`](@ref) with that matrix.
  - `scai`, `scao` and `ex` stay unchanged.

# Examples

```jldoctest
julia> HierarchicalEqualRiskContribution()
HierarchicalEqualRiskContribution
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

Return whether the [`HierarchicalEqualRiskContribution`](@ref) needs previous portfolio weights.

Returns `true` when `opt.opt`, `opt.ri`, `opt.ro` or `opt.fb` needs previous weights, or when a [`TimeDependent`](@ref) schedule in a field holds a value that does. A [`TurnoverRiskMeasure`](@ref) and a turnover fee both need them.

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

When `hec.opt.pe` holds a prior result, the view reads that prior's `X` in place of the `X` it receives. The struct's `## View parameters` states what each field does.

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
    opt = port_opt_view(hec.opt, i, X)
    return HierarchicalEqualRiskContribution(; ri = ri, ro = ro, opt = opt, scai = hec.scai,
                                             scao = hec.scao, ex = hec.ex,
                                             fb = view_child(hec.fb, i, X))
end
function non_investable_universe(hec::HierarchicalEqualRiskContribution, ni::VecStr)
    return rebuild_estimator(hec, (; opt = non_investable_universe(hec.opt, ni)))
end
"""
    herc_unitary_risks(r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees}) -> VecNum
    herc_unitary_risks(rs::VecOptRM, X::MatNum, fees::Option{<:Fees}) -> VecVecNum

Return the risk of each asset held alone, once per measure.

The unitary risks do not depend on the cluster, so [`herc_risk`](@ref) computes them once for the whole universe, and every cluster reads its own entries.

# Arguments

  - `r`, `rs`: A risk measure, or a vector of them, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees, or `nothing`.

# Returns

  - `rku::VecNum`: For one measure, the unitary risks from [`unitary_expected_risks`](@ref).
  - `rkus::VecVecNum`: For a vector of measures, one vector of unitary risks per measure, in the order of `rs`.

# Related

  - [`unitary_expected_risks`](@ref)
  - [`herc_intra_cluster_weights`](@ref)
  - [`herc_cluster_risk`](@ref)
"""
function herc_unitary_risks(r::OptimisationRiskMeasure, X::MatNum, fees::Option{<:Fees})
    return unitary_expected_risks(r, X, fees)
end
function herc_unitary_risks(rs::VecOptRM, X::MatNum, fees::Option{<:Fees})
    return [unitary_expected_risks(r, X, fees) for r in rs]
end
"""
    herc_intra_cluster_weights(sca::Scalariser, r::OptimisationRiskMeasure, rku::VecNum,
                               cl::VecInt) -> VecNum
    herc_intra_cluster_weights(sca::Scalariser, rs::VecOptRM, rkus::VecVecNum,
                               cl::VecInt) -> VecNum

Return the shares that divide the weight of cluster `cl` between its assets.

One measure gives the naive risk parity weights inside the cluster, and ignores `sca`. A vector of measures gives the naive risk parity weights of each measure, and the scalariser combines them into one vector that sums to one.

# Mathematical definition

```math
\\begin{align}
v^{(k)}_i &= \\frac{\\rho_k(\\{i\\})^{-1}}{\\sum_{l \\in C} \\rho_k(\\{l\\})^{-1}} \\quad \\forall\\, i \\in C\\,,\\\\
u_i &= \\begin{cases}
\\sum_{k} s_k \\, v^{(k)}_i & \\text{sum}\\,,\\\\
v^{(k^\\star)}_i,\\; k^\\star = \\underset{k}{\\arg\\max} \\sum_{l \\in C} s_k \\, \\rho_k(\\{l\\}) & \\text{max}\\,,\\\\
v^{(k^\\star)}_i,\\; k^\\star = \\underset{k}{\\arg\\min} \\sum_{l \\in C} s_k \\, \\rho_k(\\{l\\}) & \\text{min}\\,,\\\\
\\frac{1}{\\gamma} \\log \\sum_{k} \\exp\\left(\\gamma \\, s_k \\, v^{(k)}_i\\right) & \\text{log-sum-exp}\\,,
\\end{cases}\\\\
v_i &= \\frac{u_i}{\\sum_{l \\in C} u_l} \\quad \\forall\\, i \\in C\\,.
\\end{align}
```

Where:

  - ``C``: The cluster `cl`.
  - ``\\rho_k``: Risk measure ``k``. One measure is the case of one ``k`` with ``s_k = 1``.
  - ``\\rho_k(\\{i\\})``: Risk of asset ``i`` held alone under ``\\rho_k``.
  - ``v^{(k)}_i``: Naive risk parity weight of asset ``i`` inside ``C`` under ``\\rho_k``.
  - $(math_dict[:s_k_scale])
  - $(math_dict[:gamma_lse])
  - ``u_i``: The combined weight of asset ``i`` before the normalisation.
  - ``v_i``: Share of asset ``i`` in the weight of ``C``.

The four cases are [`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref) and [`LogSumExpScalariser`](@ref). Under a sum, ``\\boldsymbol{v}`` is the average of the ``\\boldsymbol{v}^{(k)}`` weighted by ``s_k``. A maximum or a minimum selects one measure by its total scaled unitary risk over ``C``, and ``\\boldsymbol{v}`` is that measure's ``\\boldsymbol{v}^{(k)}``. The shares sum to one under every scalariser, so the cluster keeps the weight that the split factors give it.

# Arguments

  - `sca`: Scalarisation strategy.
  - `r`, `rs`: A risk measure, or a vector of them, already resolved by [`factory`](@ref).
  - `rku`, `rkus`: The unitary risks of `r`, or of each measure of `rs`, from [`herc_unitary_risks`](@ref).
  - `cl`: Asset indices of the cluster.

# Returns

  - `v::VecNum`: The shares, of length `length(cl)`, in the order of `cl`.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_cluster_risk`](@ref)
  - [`herc_unitary_risks`](@ref)
  - [`scalarise`](@ref)
"""
function herc_intra_cluster_weights(::Scalariser, ::OptimisationRiskMeasure, rku::VecNum,
                                    cl::VecInt)
    v = inv.(view(rku, cl))
    return v ./= sum(v)
end
function herc_intra_cluster_weights(sca::Union{SumScalariser, LogSumExpScalariser},
                                    rs::VecOptRM, rkus::VecVecNum, cl::VecInt)
    u = scalarise(sca, pairs(rs)) do (k, r)
        return r.settings.scale * herc_intra_cluster_weights(sca, r, rkus[k], cl)
    end
    return u ./= sum(u)
end
function herc_intra_cluster_weights(sca::Union{MaxScalariser, MinScalariser}, rs::VecOptRM,
                                    rkus::VecVecNum, cl::VecInt)
    k = scalarise(sca, pairs(rs); by = first) do (k, r)
        return (r.settings.scale * sum(view(rkus[k], cl)), k)
    end[2]
    return herc_intra_cluster_weights(sca, rs[k], rkus[k], cl)
end
"""
    herc_cluster_risk(sca::Scalariser, r::OptimisationRiskMeasure, rku::VecNum,
                      cl::VecInt, X::MatNum, fees::Option{<:Fees}) -> Number
    herc_cluster_risk(sca::Scalariser, rs::VecOptRM, rkus::VecVecNum, cl::VecInt,
                      X::MatNum, fees::Option{<:Fees}) -> Number

Return the risk of the naive risk parity sub-portfolio of cluster `cl`, the risk from which HERC forms its split factors.

One measure ignores `sca`. A vector of measures builds one sub-portfolio per measure, because the weights follow that measure's own unitary risks, and the scalariser combines their scaled risks into one number.

# Mathematical definition

```math
\\begin{align}
\\tilde{w}^{(k)}_i(C) &= \\frac{\\rho_k(\\{i\\})^{-1}}{\\sum_{l \\in C} \\rho_k(\\{l\\})^{-1}} \\quad \\forall\\, i \\in C\\,,\\\\
p_k &= s_k \\, \\rho_k\\left(\\tilde{\\boldsymbol{w}}^{(k)}(C)\\right)\\,,\\\\
\\tilde{\\rho}(C) &= \\begin{cases}
\\sum_{k} p_k & \\text{sum}\\,,\\\\
\\max_{k} p_k & \\text{max}\\,,\\\\
\\min_{k} p_k & \\text{min}\\,,\\\\
\\frac{1}{\\gamma} \\log \\sum_{k} \\exp\\left(\\gamma \\, p_k\\right) & \\text{log-sum-exp}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``C``: The cluster `cl`.
  - ``\\rho_k``: Risk measure ``k``. One measure is the case of one ``k``, and its risk is not scaled, so ``s_k = 1``.
  - ``\\rho_k(\\{i\\})``: Risk of asset ``i`` held alone under ``\\rho_k``.
  - ``\\tilde{\\boldsymbol{w}}^{(k)}(C)``: Naive risk parity weights of ``\\rho_k`` inside ``C``, zero outside it.
  - $(math_dict[:s_k_scale])
  - ``p_k``: The scaled risk of the sub-portfolio of ``\\rho_k``.
  - $(math_dict[:gamma_lse])
  - ``\\tilde{\\rho}(C)``: The combined risk of the cluster.

The risk of one measure carries no scale. Every cluster would carry the same scale, and a split factor is a ratio of cluster risks, so the scale would cancel.

# Arguments

  - `sca`: Scalarisation strategy.
  - `r`, `rs`: A risk measure, or a vector of them, already resolved by [`factory`](@ref).
  - `rku`, `rkus`: The unitary risks of `r`, or of each measure of `rs`, from [`herc_unitary_risks`](@ref).
  - `cl`: Asset indices of the cluster.
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees, or `nothing`.

# Returns

  - `risk::Number`: The combined risk of the cluster.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_intra_cluster_weights`](@ref)
  - [`herc_unitary_risks`](@ref)
  - [`scalarise`](@ref)
"""
function herc_cluster_risk(sca::Scalariser, r::OptimisationRiskMeasure, rku::VecNum,
                           cl::VecInt, X::MatNum, fees::Option{<:Fees})
    w = zeros(float_if_integer(eltype(X)), size(X, 2))
    w[cl] = herc_intra_cluster_weights(sca, r, rku, cl)
    return expected_risk(r, w, X, fees)
end
function herc_cluster_risk(sca::Scalariser, rs::VecOptRM, rkus::VecVecNum, cl::VecInt,
                           X::MatNum, fees::Option{<:Fees})
    return scalarise(sca, pairs(rs)) do (k, r)
        return r.settings.scale * herc_cluster_risk(sca, r, rkus[k], cl, X, fees)
    end
end
"""
    herc_risk(hec::HierarchicalEqualRiskContribution, pr::AbstractPriorResult,
              cls::VecVecInt, fees::Option{<:Fees}) -> Tuple

Compute the shares inside each cluster and the risk of each cluster that HERC allocates with.

# Algorithm

 1. Resolve the inner measure `ri` against `pr` and `hec.opt.slv`, with [`factory`](@ref).
 2. Compute its unitary risks `riku`, with [`herc_unitary_risks`](@ref).
 3. When `hec.ro` is the same object as `hec.ri`, take `ro = ri` and `roku = riku`. Otherwise resolve `ro` and compute `roku` the same way.
 4. For each cluster `cl`, under the executor `hec.ex`, write the shares of [`herc_intra_cluster_weights`](@ref) into `w[cl]`, and the risk of [`herc_cluster_risk`](@ref) into the cluster's entry of `rkcl`.

# Arguments

  - `hec`: The optimiser. It gives the measures, the scalarisers `scai` and `scao`, and the executor `ex`.
  - `pr`: Prior result. Its `X` is the return matrix, and its moments resolve the measures.
  - `cls`: Asset indices of each cluster, one entry per cluster.
  - `fees`: Resolved fees, or `nothing`. The caller resolves them on its own universe and strips both liquidation carriers first, because a cluster-level risk prices no forced exit. The exiting asset is in no cluster, because its column is `NaN`.

# Returns

  - `(w, rkcl, ri, ro)::Tuple`: The shares over the whole universe, the risk of each cluster, and the two resolved measures. The shares of each cluster sum to one, so the split factors still have to scale them.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_intra_cluster_weights`](@ref)
  - [`herc_cluster_risk`](@ref)
  - [`herc_unitary_risks`](@ref)
"""
function herc_risk(hec::HierarchicalEqualRiskContribution, pr::AbstractPriorResult,
                   cls::VecVecInt, fees::Option{<:Fees})
    X = pr.X
    ri = factory(hec.ri, pr, hec.opt.slv)
    riku = herc_unitary_risks(ri, X, fees)
    ro = hec.ri === hec.ro ? ri : factory(hec.ro, pr, hec.opt.slv)
    roku = hec.ri === hec.ro ? riku : herc_unitary_risks(ro, X, fees)
    # A weight is a quotient of two risks, so an integer sample takes a float weight type,
    # and every other sample keeps its own type.
    T = float_if_integer(eltype(X))
    w = Vector{T}(undef, size(X, 2))
    rkcl = Vector{T}(undef, length(cls))
    FLoops.@floop hec.ex for (i, cl) in pairs(cls)
        w[cl] = herc_intra_cluster_weights(hec.scai, ri, riku, cl)
        rkcl[i] = herc_cluster_risk(hec.scao, ro, roku, cl, X, fees)
    end
    return w, rkcl, ri, ro
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the Hierarchical Equal Risk Contribution optimisation.

[`optimise`](@ref) calls this method. The struct [`HierarchicalEqualRiskContribution`](@ref) states the mathematics.

# Algorithm

 1. Resolve every [`TimeDependent`](@ref) field of `hec` to its static default, with [`reset_time_dependent_estimator`](@ref).
 2. Pick the returns `rd` that `opt.brt` selects, with [`returns_result_picker`](@ref).
 3. Fit the prior `pr` with `opt.pe`.
 4. Find the Investable Mask `imsk` of `pr`. Resolve the fees `fees` on the full universe, and place them on the assets of `imsk`.
 5. Remove the liquidation carriers from `fees`, giving `cfees`, the fees that the risks read.
 6. Reduce `pr`, `hec` and `rd` to the assets of `imsk`, with [`investable_reduction`](@ref). `X` is the returns matrix of the reduced `pr`, and `N` its number of assets.
 7. Cluster the assets with `opt.cle`, giving `clr`, and cut its dendrogram into `clr.k` clusters, giving `cls`, the asset indices of each cluster.
 8. Compute the shares `w` and the cluster risks `rkcl` with [`herc_risk`](@ref).
 9. Build the nodes `nd` of the dendrogram with [`to_tree`](@ref). The cut does the first `N - clr.k` merges in the order of their heights, as `Clustering.cutree` does, so the other `clr.k - 1` merges are the nodes above the cut.
10. For each node above the cut, read the assets `ln` and `rn` of its two branches, and the clusters `lc` and `rc` inside them.
11. Form the split factor `alpha` from the risks of `lc` and `rc`. Multiply `w[ln]` by `alpha` and `w[rn]` by `1 - alpha`.
12. Resolve the weight bounds `wb` from `opt.wb` and `opt.sets`. Divide `w` by its sum, and apply the weight finaliser `opt.wf` with [`finalise_weight_bounds`](@ref), giving `retcode` and the final `w`.
13. Return a [`HierarchicalEqualRiskContributionResult`](@ref). Its [`HierarchicalResult`](@ref) expands `w` back onto the full asset universe through `imsk`.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_risk`](@ref)
  - [`optimise`](@ref)
"""
function _optimise(hec::HierarchicalEqualRiskContribution,
                   rd::ReturnsResult = ReturnsResult(); branchorder::Symbol = :optimal,
                   kwargs...)
    hec = reset_time_dependent_estimator(hec)
    rd = returns_result_picker(rd, hec.opt.brt)
    pr = prior(hec.opt.pe, rd)
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset, and a carrier keyed by name cannot resolve at all once its `w` sits on the
    # complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    # A weight is a quotient of two risks, so an integer sample takes a float weight type,
    # and every other sample keeps its own type.
    T = float_if_integer(eltype(pr.X))
    imsk = investable_mask(pr)
    fees = investable_fees_view(fees_constraints(hec.opt.fees, hec.opt.sets;
                                                 strict = hec.opt.strict, datatype = T),
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
    _, pr, hec, rd = investable_reduction(imsk, pr, hec, rd)
    X = pr.X
    N = size(X, 2)
    clr = clusterise(hec.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa,
                     branchorder = branchorder, x_src = hec.opt.x_src)
    assert_clustering_universe(clr, N)
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    w, rkcl, ri, ro = herc_risk(hec, pr, cls, cfees)
    nd = to_tree(clr.res)[2]
    # `Clustering.cutree` does the first `N - k` merges in the order of `sortperm` of the
    # heights, so the last `k - 1` merges of that order are the nodes above the cut. A sort
    # of the nodes by height alone picks a node inside a cluster when two heights tie at
    # the cut, and the split factor of that node is `0 / 0`.
    lc = Vector{Int}(undef, 0)
    rc = Vector{Int}(undef, 0)
    for j in view(sortperm(clr.res.heights), (N - clr.k + 1):(N - 1))
        node = nd[N + j]
        ln = pre_order(node.left)
        rn = pre_order(node.right)
        for (i, cl) in pairs(cls)
            if issubset(cl, ln)
                push!(lc, i)
            elseif issubset(cl, rn)
                push!(rc, i)
            end
        end
        lrisk = sum(view(rkcl, lc))
        rrisk = sum(view(rkcl, rc))
        alpha = one(lrisk) - lrisk / (lrisk + rrisk)
        w[ln] .*= alpha
        w[rn] .*= one(alpha) - alpha
        empty!(lc)
        empty!(rc)
    end
    wb = weight_bounds_constraints(hec.opt.wb, hec.opt.sets; N = N, strict = hec.opt.strict,
                                   datatype = T)
    retcode, w = finalise_weight_bounds(hec.opt.wf, wb, w / sum(w))
    return HierarchicalEqualRiskContributionResult(;
                                                   hr = HierarchicalResult(; pr = pr,
                                                                           clr = clr,
                                                                           wb = wb,
                                                                           fees = fees,
                                                                           retcode = retcode,
                                                                           w = w,
                                                                           imsk = imsk),
                                                   ri = ri, ro = ro, scai = hec.scai,
                                                   scao = hec.scao, fb = nothing)
end
"""
    optimise(hec::HierarchicalEqualRiskContribution{
                     <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing
                 },
            rd::ReturnsResult; branchorder::Symbol = :optimal, kwargs...) -> HierarchicalEqualRiskContributionResult

Run the Hierarchical Equal Risk Contribution portfolio optimisation.

# Arguments

  - `hec`: The hierarchical equal risk contribution optimiser to use.
  - $(arg_dict[:rd]) If `isa(hec.opt.pe, AbstractPriorResult)`, a standalone optimisation does not need `rd`, but a fallback or the clustering can need it.
  - `branchorder`: The branch order of the clustering. The weights do not depend on the order of the leaves, so a faster order that is not optimal gives the same weights. When `opt.cle` holds a clustering result, the order has no effect.
  - `kwargs`: Keyword arguments that the method accepts and does not read.

# Validation

  - No field in the tree of `hec` holds an [`Online`](@ref). The method throws an `ArgumentError` that names the field otherwise, through [`assert_batch_entry`](@ref). A plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the fold loop's online arm.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
  - [`_optimise`](@ref)
"""
function optimise(hec::HierarchicalEqualRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, Nothing}, rd::ReturnsResult;
                  branchorder::Symbol = :optimal, kwargs...)
    assert_batch_entry(hec, "`optimise`")
    return _optimise(hec, rd; branchorder = branchorder, kwargs...)
end

export HierarchicalEqualRiskContribution
