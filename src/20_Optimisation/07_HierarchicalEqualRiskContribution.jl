"""
$(DocStringExtensions.TYPEDEF)

Hierarchical Equal Risk Contribution (HERC) portfolio optimiser.

`HierarchicalEqualRiskContribution` implements the Hierarchical Equal Risk Contribution algorithm. It clusters assets, then allocates weights so that each cluster contributes equally to total portfolio risk (using `ro`), and within each cluster, assets are weighted by inverse intra-cluster risk (using `ri`).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalEqualRiskContribution(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        ri::OptRM_VecOptRM = Variance(),
        ro::OptRM_VecOptRM = ri,
        scai::Scalariser = SumScalariser(),
        scao::Scalariser = scai,
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::Option{<:OptE_Opt} = nothing
    ) -> HierarchicalEqualRiskContribution

Keywords correspond to the struct's fields.

## Validation

  - If `ri` or `ro` is a vector: `!isempty(ri)` / `!isempty(ro)`.

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

# Mathematical definition

Let ``K`` be the number of clusters. The inter-cluster (outer) step assigns equal risk contribution across all clusters using risk measure ``\\rho_o``:

```math
\\begin{align}
w_{C_k} &= \\frac{\\tilde{\\rho}_o(C_k)^{-1}}{\\sum_{j=1}^{K} \\tilde{\\rho}_o(C_j)^{-1}}\\,.
\\end{align}
```

Within each cluster ``C_k``, the intra-cluster (inner) step assigns weights proportional to inverse intra-cluster risk ``\\rho_i``:

```math
\\begin{align}
w_i &\\propto \\tilde{\\rho}_i(\\{i\\})^{-1}, \\quad i \\in C_k\\,, \\\\
\\sum_{i \\in C_k} w_i &= w_{C_k}\\,.
\\end{align}
```

Where:

  - ``w_{C_k}``: Weight allocated to cluster ``C_k``.
  - ``\\tilde{\\rho}_o(C_k)``: Outer (inter-cluster) risk of cluster ``C_k``.
  - ``\\tilde{\\rho}_i(\\{i\\})``: Inner (intra-cluster) risk of asset ``i``.
  - ``K``: Number of clusters.
  - ``w_i``: Final weight of asset ``i``.
  - ``\\tilde{\\rho}``: Quasi-diagonal cluster portfolio risk.

# Related

  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`HierarchicalOptimiser`](@ref)
"""
@concrete struct HierarchicalEqualRiskContribution <: ClusteringOptimisationEstimator
    """
    $(field_dict[:opt_hier])
    """
    opt
    """
    $(field_dict[:ri])
    """
    ri
    """
    $(field_dict[:ro])
    """
    ro
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
    fb
    function HierarchicalEqualRiskContribution(opt::HierarchicalOptimiser,
                                               ri::OptRM_VecOptRM, ro::OptRM_VecOptRM,
                                               scai::Scalariser, scao::Scalariser,
                                               ex::FLoops.Transducers.Executor,
                                               fb::Option{<:OptE_Opt})
        if isa(ri, AbstractVector)
            @argcheck(!isempty(ri), IsEmptyError("ri cannot be empty"))
        end
        if isa(ro, AbstractVector)
            @argcheck(!isempty(ro), IsEmptyError("ro cannot be empty"))
        end
        return new{typeof(opt), typeof(ri), typeof(ro), typeof(scai), typeof(scao),
                   typeof(ex), typeof(fb)}(opt, ri, ro, scai, scao, ex, fb)
    end
end
function HierarchicalEqualRiskContribution(;
                                           opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                           ri::OptRM_VecOptRM = Variance(),
                                           ro::OptRM_VecOptRM = ri,
                                           scai::Scalariser = SumScalariser(),
                                           scao::Scalariser = scai,
                                           ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                           fb::Option{<:OptE_Opt} = nothing)::HierarchicalEqualRiskContribution
    return HierarchicalEqualRiskContribution(opt, ri, ro, scai, scao, ex, fb)
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
    return (needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.ri) ||
            needs_previous_weights(opt.ro) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create a [`HierarchicalEqualRiskContribution`](@ref) updating the base optimiser, risk measures, and fallback with weights `w`.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`factory`](@ref)
"""
function factory(hec::HierarchicalEqualRiskContribution,
                 w::AbstractVector)::HierarchicalEqualRiskContribution
    opt = factory(hec.opt, w)
    ri = factory(hec.ri, w)
    ro = factory(hec.ro, w)
    fb = factory(hec.fb, w)
    return HierarchicalEqualRiskContribution(; opt = opt, ri = ri, ro = ro, scai = hec.scai,
                                             scao = hec.scao, ex = hec.ex, fb = fb)
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
    herc_scalarised_risk_o!(scalariser, wk, roku, rkbo, cl, ros, X, fees)

Compute and accumulate the scalarised outer (inter-cluster) HERC risk in-place.

Updates `rkbo` with inverse-risk weights for cluster `cl` and accumulates the scaled risk contribution using the given scalariser strategy.

# Arguments

  - `scalariser`: Scalarisation strategy ([`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref), or [`LogSumExpScalariser`](@ref)).
  - `wk`: Cluster weight vector.
  - `roku`: Unitary outer risk vector or matrix.
  - `rkbo`: Outer risk buffer vector (modified in-place).
  - `cl`: Cluster asset indices.
  - `ros`: Vector of outer risk measures.
  - `X`: Return matrix.
  - `fees`: Optional fees.

# Returns

  - Scalarised outer cluster risk scalar.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_scalarised_risk_i!`](@ref)
"""
function herc_scalarised_risk_o!(::SumScalariser, wk::VecNum, roku::VecNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = zero(eltype(X))
    for ro in ros
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
    return crisk
end
function herc_scalarised_risk_o!(::SumScalariser, wk::VecNum, roku::MatNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = zero(eltype(X))
    for (i, ro) in pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk += ro.settings.scale * expected_risk(ro, rkbo, X, fees)
    end
    return crisk
end
function herc_scalarised_risk_o!(::MaxScalariser, wk::VecNum, roku::VecNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = typemin(eltype(X))
    for ro in ros
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i > crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(::MaxScalariser, wk::VecNum, roku::MatNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = typemin(eltype(X))
    for (i, ro) in pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i > crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(::MinScalariser, wk::VecNum, roku::VecNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = typemax(eltype(X))
    for ro in ros
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i < crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(::MinScalariser, wk::VecNum, roku::MatNum, rkbo::VecNum,
                                 cl::VecInt, ros::VecOptRM, X::MatNum, fees::Option{<:Fees})
    crisk = typemax(eltype(X))
    for (i, ro) in pairs(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk_i = ro.settings.scale * expected_risk(ro, rkbo, X, fees)
        if crisk_i < crisk
            crisk = crisk_i
        end
    end
    return crisk
end
function herc_scalarised_risk_o!(sca::LogSumExpScalariser, wk::VecNum, roku::VecNum,
                                 rkbo::VecNum, cl::VecInt, ros::VecOptRM, X::MatNum,
                                 fees::Option{<:Fees})
    crisk = Vector{eltype(X)}(undef, length(ros))
    for (i, ro) in enumerate(ros)
        unitary_expected_risks!(wk, roku, ro, X, fees)
        rkbo[cl] .= inv.(view(roku, cl))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk[i] = ro.settings.scale * sca.gamma * expected_risk(ro, rkbo, X, fees)
    end
    return LogExpFunctions.logsumexp(crisk) / sca.gamma
end
function herc_scalarised_risk_o!(sca::LogSumExpScalariser, ::VecNum, roku::MatNum,
                                 rkbo::VecNum, cl::VecInt, ros::VecOptRM, X::MatNum,
                                 fees::Option{<:Fees})
    crisk = Vector{eltype(X)}(undef, length(ros))
    for (i, ro) in enumerate(ros)
        rkbo[cl] .= inv.(view(roku, cl, i))
        rkbo[cl] ./= sum(view(rkbo, cl))
        crisk[i] = ro.settings.scale * sca.gamma * expected_risk(ro, rkbo, X, fees)
    end
    return LogExpFunctions.logsumexp(crisk) / sca.gamma
end
"""
    herc_scalarised_risk_i!(scalariser, wk, riku, cl, ris, X, fees)

Compute the scalarised inner (intra-cluster) HERC risk for cluster `cl`.

Aggregates inner risk measures across the assets in cluster `cl` using the given scalariser, returning the per-asset risk vector used for intra-cluster weight allocation.

# Arguments

  - `scalariser`: Scalarisation strategy ([`SumScalariser`](@ref), [`MaxScalariser`](@ref), [`MinScalariser`](@ref), or [`LogSumExpScalariser`](@ref)).
  - `wk`: Cluster weight vector.
  - `riku`: Unitary inner risk vector or matrix.
  - `cl`: Cluster asset indices.
  - `ris`: Vector of inner risk measures.
  - `X`: Return matrix.
  - `fees`: Optional fees.

# Returns

  - Per-asset inner risk vector for assets in `cl`.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_scalarised_risk_o!`](@ref)
"""
function herc_scalarised_risk_i!(::SumScalariser, wk::VecNum, riku::VecNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for ri in ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] += ri.settings.scale * view(risk, :, 1)
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::SumScalariser, wk::VecNum, riku::MatNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] .= inv.(view(view(riku, :, i), cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 2] += ri.settings.scale * view(risk, :, 1)
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MaxScalariser, wk::VecNum, riku::VecNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk_t = typemin(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for ri in ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] = ri.settings.scale * view(riku, cl)
        risk_i = sum(view(risk, :, 1))
        if risk_i > risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] = view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MaxScalariser, wk::VecNum, riku::MatNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk_t = typemin(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] = ri.settings.scale * view(riku, cl, i)
        risk_i = sum(view(risk, :, 1))
        if risk_i > risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] = view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MinScalariser, wk::VecNum, riku::VecNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk_t = typemax(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for ri in ris
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] = ri.settings.scale * view(riku, cl)
        risk_i = sum(view(risk, :, 1))
        if risk_i < risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] = view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(::MinScalariser, wk::VecNum, riku::MatNum, cl::VecInt,
                                 ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    risk_t = typemax(eltype(X))
    risk = zeros(eltype(X), length(cl), 2)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] = ri.settings.scale * view(riku, cl, i)
        risk_i = sum(view(risk, :, 1))
        if risk_i < risk_t
            risk_t = risk_i
            risk[:, 2] .= inv.(view(risk, :, 1))
            risk[:, 2] = view(risk, :, 2) / sum(view(risk, :, 2))
        end
    end
    return view(risk, :, 2)
end
function herc_scalarised_risk_i!(sca::LogSumExpScalariser, wk::VecNum, riku::VecNum,
                                 cl::VecInt, ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    N = length(ris)
    risk = zeros(eltype(X), length(cl), 1 + N)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, riku, ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 1 + i] = ri.settings.scale * sca.gamma * view(risk, :, 1)
    end
    return LogExpFunctions.logsumexp(view(risk, :, 2:N); dims = 2) / sca.gamma
end
function herc_scalarised_risk_i!(sca::LogSumExpScalariser, wk::VecNum, riku::MatNum,
                                 cl::VecInt, ris::VecOptRM, X::MatNum, fees::Option{<:Fees})
    N = length(ris)
    risk = zeros(eltype(X), length(cl), 1 + N)
    for (i, ri) in pairs(ris)
        unitary_expected_risks!(wk, view(riku, :, i), ri, X, fees)
        risk[:, 1] .= inv.(view(riku, cl, i))
        risk[:, 1] ./= sum(view(risk, :, 1))
        risk[:, 1 + i] += ri.settings.scale * sca.gamma * view(risk, :, 1)
    end
    return LogExpFunctions.logsumexp(view(risk, :, 2:N); dims = 2) / sca.gamma
end
"""
    herc_risk(hec, pr, cls)

Compute per-cluster risk contributions for HERC weight allocation.

Evaluates the inner and outer risk measures for all clusters in `cls`, returning the risk arrays needed to allocate intra- and inter-cluster weights.

# Arguments

  - `hec`: [`HierarchicalEqualRiskContribution`](@ref) optimiser instance.
  - `pr`: Prior result containing asset moments and return data.
  - `cls`: Vector of vectors of asset indices per cluster.

# Returns

  - `(riku, roku)`: Inner and outer per-asset risk arrays.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`herc_scalarised_risk_i!`](@ref)
  - [`herc_scalarised_risk_o!`](@ref)
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
    return w, rkcl, fees
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
    return w, rkcl, fees
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
    return w, rkcl, fees
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
    return w, rkcl, fees
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
    return w, rkcl, fees
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
    return w, rkcl, fees
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
    return w, rkcl, fees
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
    return w, rkcl, fees
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
    rd = returns_result_picker(rd, hec.opt.brt)
    pr = prior(hec.opt.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(hec.opt.cle, pr; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     branchorder = branchorder, cle_pr = hec.opt.cle_pr)
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    w, rkcl, fees = herc_risk(hec, pr, cls)
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
    return HierarchicalResult(; oe = typeof(hec), pr = pr, clr = clr, wb = wb, fees = fees,
                              retcode = retcode, w = w, fb = nothing)
end
"""
    optimise(hec::HierarchicalEqualRiskContribution{
                     <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing
                 },
            rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
            branchorder::Symbol = :optimal, kwargs...) -> HierarchicalResult

Run the Hierarchical Equal Risk Contribution portfolio optimisation.

# Arguments

  - `hec`: The hierarchical equal risk contribution optimiser to use.
  - $(arg_dict[:rd]) If `isa(hec.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `branchorder`: The branch order to use for the clusterisation, this optimisation can use non-optimal branch orders, which make the clustering faster but the dendrogram won't be as nice.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`HierarchicalResult`](@ref)
"""
function optimise(hec::HierarchicalEqualRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, kwargs...)
    return _optimise(hec, rd; dims = dims, branchorder = branchorder, kwargs...)
end

export HierarchicalEqualRiskContribution
