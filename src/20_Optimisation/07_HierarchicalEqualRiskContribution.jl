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

Hierarchical Equal Risk Contribution (HERC) portfolio optimiser.

`HierarchicalEqualRiskContribution` implements the Hierarchical Equal Risk Contribution algorithm. It clusters assets, then allocates weights so that each cluster contributes equally to total portfolio risk (using `ro`), and within each cluster, assets are weighted by inverse intra-cluster risk (using `ri`).

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
    hec = reset_time_dependent_estimator(hec)
    rd = returns_result_picker(rd, hec.opt.brt)
    pr = prior(hec.opt.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(hec.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
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
