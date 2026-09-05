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

Holds the prior result, the resolved weight bounds, the clustering result, the resolved risk measure, the Schur complement parameter the allocation ran at, the return code, the optimised weights, and the optional fallback estimator.

# Fields

$(DocStringExtensions.FIELDS)

## The measure is on the result, and the result is flat

Schur joins [`HierarchicalOptimisationResult`](@ref) — it embeds a [`HierarchicalOptimiser`](@ref), which is the family's membership rule — but it keeps its own flat field block rather than embedding [`HierarchicalResult`](@ref). Its field set genuinely differs: it carries `gamma`, and it has **no** `fees` field.

It carries **no scalariser**, because it carries no vector of measures to combine. `SchurComplementParams.r` is bounded [`Sd_Var`](@ref), so Schur takes one standard deviation or one variance.

!!! warning

    On the **multi-bundle** path `r` is a vector, and the blend is over **portfolios**, not risks: the loop accumulates `w .+= ps.r.settings.scale * wi`. So `expected_risk(res.r, res.w, res.pr)` reports the measure-scalarised figure on the blended weights, which is **not** the number Schur computed. On the single-bundle path, which is the default, the same call is exactly right.

# Constructors

    SchurComplementHierarchicalRiskParityResult(;
        pr::Option{<:AbstractPriorResult},
        wb::Option{<:WeightBounds},
        clr::Option{<:AbstractClusteringResult},
        r::Union{<:Sd_Var, <:VecBaseRM},
        gamma::Union{<:Number, <:VecNum},
        retcode::OptimisationReturnCode,
        w::Option{<:VecNum},
        fb::Option{<:OptE_Opt}
    ) -> SchurComplementHierarchicalRiskParityResult

Keywords correspond to the struct's fields.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`HierarchicalOptimisationResult`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct SchurComplementHierarchicalRiskParityResult <:
                 HierarchicalOptimisationResult
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
    $(field_dict[:r_res_schur])
    """
    r
    """
    $(field_dict[:gamma_schur_res])
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
    function SchurComplementHierarchicalRiskParityResult(pr::Option{<:AbstractPriorResult},
                                                         wb::Option{<:WeightBounds},
                                                         clr::Option{<:AbstractClusteringResult},
                                                         r::Union{<:Sd_Var, <:VecBaseRM},
                                                         gamma::Union{<:Number, <:VecNum},
                                                         retcode::OptimisationReturnCode,
                                                         w::Option{<:VecNum},
                                                         fb::Option{<:OptE_Opt})
        return new{typeof(pr), typeof(wb), typeof(clr), typeof(r), typeof(gamma),
                   typeof(retcode), typeof(w), typeof(fb)}(pr, wb, clr, r, gamma, retcode,
                                                           w, fb)
    end
end
function SchurComplementHierarchicalRiskParityResult(; pr::Option{<:AbstractPriorResult},
                                                     wb::Option{<:WeightBounds},
                                                     clr::Option{<:AbstractClusteringResult},
                                                     r::Union{<:Sd_Var, <:VecBaseRM},
                                                     gamma::Union{<:Number, <:VecNum},
                                                     retcode::OptimisationReturnCode,
                                                     w::Option{<:VecNum},
                                                     fb::Option{<:OptE_Opt})::SchurComplementHierarchicalRiskParityResult
    return SchurComplementHierarchicalRiskParityResult(pr, wb, clr, r, gamma, retcode, w,
                                                       fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that choose the Schur complement parameter ``\\gamma``.

A subtype decides what [`SchurComplementParams`](@ref)`.gamma` means: the value to use, or the upper end of a range to search.

# Related

  - [`SchurComplementParams`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
  - [`MonotonicSchurComplement`](@ref)
"""
abstract type SchurComplementAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Runs the allocation at the ``\\gamma`` the caller gave, with no search.

The augmentation uses [`SchurComplementParams`](@ref)`.gamma` exactly. Portfolio variance is not monotonic in ``\\gamma``, so a larger value does not always give a lower-variance portfolio; [`MonotonicSchurComplement`](@ref) searches for the value that does.

# Related

  - [`SchurComplementAlgorithm`](@ref)
  - [`MonotonicSchurComplement`](@ref)
  - [`SchurComplementParams`](@ref)
"""
struct NonMonotonicSchurComplement <: SchurComplementAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Searches ``[0, \\gamma]`` for the value that gives the lowest portfolio variance.

Portfolio variance is not monotonic in the Schur complement parameter: it falls, then rises again. This algorithm scans `N` values across the range, stops at the first one whose variance is no lower than its predecessor, and bisects the bracket around that turning point to `tol`. The allocation then runs at the value it found, which is at most the [`SchurComplementParams`](@ref)`.gamma` the caller asked for.

The objective is the variance ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}`` in every case, including when the measure is a [`StandardDeviation`](@ref). The search also runs with the positive-definite repair **off**, so a ``\\gamma`` whose augmented block is not positive definite scores an infinite variance and is passed over rather than raising.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MonotonicSchurComplement(;
        N::Integer = 10,
        tol::Number = 1e-4,
        iter::Option{<:Integer} = nothing,
        strict::Bool = false
    ) -> MonotonicSchurComplement

Keywords correspond to the struct's fields. `iter` defaults to `nothing`, which means the bisection derives its own budget from the bracket and `tol`, as `ceil(Int, log2((hgamma - lgamma) / tol) * 4 + 10)`.

## Validation

  - `N > 0`.
  - `tol > 0`.
  - If `iter` is given: `iter > 0`.

# Related

  - [`SchurComplementAlgorithm`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`schur_complement_binary_search`](@ref)
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

Collects the risk measure, the interpolation parameter ``\\gamma``, and the two algorithms that one Schur complement bundle runs with.

[`SchurComplementHierarchicalRiskParity`](@ref) holds one of these, or a vector of them. A vector runs one allocation per bundle and blends the resulting **portfolios** by each bundle's `r.settings.scale`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SchurComplementParams(;
        r::Sd_Var = Variance(),
        gamma::Number = 0.5,
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        alg::SchurComplementAlgorithm = MonotonicSchurComplement(),
        flag::Bool = true
    ) -> SchurComplementParams

Keywords correspond to the struct's fields. `r` is bounded to [`Sd_Var`](@ref) because the allocation needs a risk it can read straight off an augmented covariance block.

## Validation

  - `0 <= gamma <= 1`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementAlgorithm`](@ref)
  - [`MonotonicSchurComplement`](@ref)
  - [`NonMonotonicSchurComplement`](@ref)
  - [`Sd_Var`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cotton2024])
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
    $(field_dict[:flag_schur])
    """
    flag
    function SchurComplementParams(r::Sd_Var, gamma::Number,
                                   pdm::Option{<:AbstractPosdefEstimator},
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
                               pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
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
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`SchurComplementHierarchicalRiskParity`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function schur_complement_hrp_td_defaults()::NamedTuple
    return (; params = SchurComplementParams())
end
"""
$(DocStringExtensions.TYPEDEF)

Runs the hierarchical risk parity recursion on covariance blocks that a Schur complement has augmented with the information in the cross-cluster block.

The parameter ``\\gamma`` interpolates: at `gamma = 0` the allocation is exactly [`HierarchicalRiskParity`](@ref), and a larger value moves it towards the minimum variance portfolio.

# Mathematical definition

The recursion is that of [`HierarchicalRiskParity`](@ref): split the dendrogram's leaf order in half, and divide the part's weight between the two halves in inverse proportion to their risks. Schur changes only how each half's covariance block is read. Partition the part's covariance over its two halves ``C_1`` and ``C_2``:

```math
\\begin{align}
\\mathbf{\\Sigma} &= \\begin{pmatrix} \\mathbf{\\Sigma}_{11} & \\mathbf{\\Sigma}_{12} \\\\ \\mathbf{\\Sigma}_{21} & \\mathbf{\\Sigma}_{22} \\end{pmatrix}\\,,\\\\
\\mathbf{A} &= \\mathbf{\\Sigma}_{11} - \\gamma \\, \\mathbf{\\Sigma}_{12} \\mathbf{\\Sigma}_{22}^{-1} \\mathbf{\\Sigma}_{21}\\,,\\\\
\\mathbf{R} &= \\mathbf{I} - \\gamma \\, \\mathbf{\\Sigma}_{12} \\mathbf{\\Sigma}_{22}^{-1} \\mathbf{M}^\\intercal\\,,\\\\
\\hat{\\mathbf{\\Sigma}}_{11} &= \\frac{1}{2}\\left(\\mathbf{R}^{-1}\\mathbf{A} + \\left(\\mathbf{R}^{-1}\\mathbf{A}\\right)^\\intercal\\right)\\,.
\\end{align}
```

``\\hat{\\mathbf{\\Sigma}}_{22}`` follows by exchanging the two halves. The risk of a half is then read off its augmented block with the naive risk parity weights that block implies:

```math
\\begin{align}
\\tilde{w}_i &= \\frac{\\left(\\hat{\\mathbf{\\Sigma}}_{11}\\right)_{ii}^{-1}}{\\sum_{j} \\left(\\hat{\\mathbf{\\Sigma}}_{11}\\right)_{jj}^{-1}}\\,,\\\\
\\tilde{\\rho}(C_1) &= \\tilde{\\boldsymbol{w}}^\\intercal \\hat{\\mathbf{\\Sigma}}_{11} \\tilde{\\boldsymbol{w}}\\,,\\\\
\\alpha &= \\frac{\\tilde{\\rho}(C_2)}{\\tilde{\\rho}(C_1) + \\tilde{\\rho}(C_2)}\\,.
\\end{align}
```

Where:

  - ``\\gamma``: The interpolation parameter, `params.gamma`, in ``[0, 1]``.
  - ``\\mathbf{\\Sigma}_{11}``, ``\\mathbf{\\Sigma}_{12}``, ``\\mathbf{\\Sigma}_{21}``, ``\\mathbf{\\Sigma}_{22}``: Covariance blocks of the partition of the part into ``C_1`` and ``C_2``.
  - ``\\mathbf{A}``: The Schur complement of ``\\mathbf{\\Sigma}_{22}``, scaled by ``\\gamma``. At ``\\gamma = 1`` it is that Schur complement exactly.
  - ``\\mathbf{M}``: The symmetric step-up matrix of size ``|C_1|`` by ``|C_2|``, see [`symmetric_step_up_matrix`](@ref).
  - ``\\mathbf{R}``: The step-up correction that carries the augmentation back to the size of ``C_1``.
  - ``\\hat{\\mathbf{\\Sigma}}_{11}``: The augmented block of ``C_1``, symmetrised.
  - ``\\tilde{\\boldsymbol{w}}``: Naive risk parity weights over ``C_1``, read from the **augmented** diagonal.
  - ``\\tilde{\\rho}``: Risk of that sub-portfolio. A [`StandardDeviation`](@ref) measure takes the square root of the quadratic form; a [`Variance`](@ref) measure does not.
  - ``\\alpha``: Fraction of the part's weight that goes to ``C_1``.

Three details bound the recursion. ``\\gamma = 0`` leaves ``\\hat{\\mathbf{\\Sigma}}_{11} = \\mathbf{\\Sigma}_{11}``, so the allocation is then exactly [`HierarchicalRiskParity`](@ref) under the same measure. A half holding one asset is left unaugmented, because ``\\mathbf{M}`` needs two halves that differ in size by at most one. And ``\\alpha`` is clamped against the resolved weight bounds by [`split_factor_weight_constraints`](@ref) before it is applied, exactly as in HRP.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SchurComplementHierarchicalRiskParity(;
        opt::HierarchicalOptimiser = HierarchicalOptimiser(),
        params::TD{<:ScP_VecScP} = SchurComplementParams(),
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> SchurComplementHierarchicalRiskParity

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the Schur parameters (risk measure and `gamma`) and the fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `fb`).

## Validation

  - If `params` is a vector: `!isempty(params)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

## View parameters

`SchurComplementHierarchicalRiskParity` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` already holds a prior **result**, the method replaces `X` with `opt.pe.X`, so the children are viewed against the prior's own observations rather than the caller's matrix.
  - `params` recurses through [`port_opt_view`](@ref) with that matrix. `opt` recurses with the index alone.
  - `fb` is carried through unchanged.

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

# Related

  - [`optimise`](@ref)
  - [`SchurComplementHierarchicalRiskParityResult`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`symmetric_step_up_matrix`](@ref)
  - [`schur_augmentation`](@ref)
  - [`split_factor_weight_constraints`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cotton2024])
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
                                                   params::TD{<:ScP_VecScP},
                                                   fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb,
                                                  :SchurComplementHierarchicalRiskParity)
        if isa(params, AbstractVector)
            @argcheck(!isempty(params), IsEmptyError("params cannot be empty"))
        end
        assert_time_dependent_substitution(SchurComplementHierarchicalRiskParity,
                                           (; opt, params, fb),
                                           schur_complement_hrp_td_defaults())
        return new{typeof(opt), typeof(params), typeof(fb)}(opt, params, fb)
    end
end
function SchurComplementHierarchicalRiskParity(;
                                               opt::HierarchicalOptimiser = HierarchicalOptimiser(),
                                               params::TD{<:ScP_VecScP} = SchurComplementParams(),
                                               fb::TDO_Option{<:OptE_Opt} = nothing)::SchurComplementHierarchicalRiskParity
    return SchurComplementHierarchicalRiskParity(opt, params, fb)
end
function time_dependent_field_defaults(::SchurComplementHierarchicalRiskParity)::NamedTuple
    return schur_complement_hrp_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`SchurComplementHierarchicalRiskParity`](@ref) requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
function needs_previous_weights(opt::SchurComplementHierarchicalRiskParity)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.fb))
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
    symmetric_step_up_matrix(n1::Integer, n2::Integer) -> AbstractMatrix

Build the matrix that carries a Schur complement between two halves of nearly equal size.

The augmentation subtracts a term shaped by the **other** half, so the result has to come back to the size of the half being augmented. This matrix is that map. It is the identity when the halves are equal, and it averages over every position an extra uniform row can take when one half is longer by one.

# Arguments

  - `n1`: Size of the half being augmented, and the number of rows.
  - `n2`: Size of the other half, and the number of columns.

# Validation

  - `abs(n1 - n2) <= 1`. A bisection produces halves that differ by at most one, so no other shape can reach this method.

# Returns

  - `m::AbstractMatrix`: An `n1` by `n2` matrix. It is `LinearAlgebra.I(n1)` when `n1 == n2`.

# Details

  - The `n1 < n2` case is the transpose of the `n1 > n2` case, scaled by `n1 / n2`.
  - Every row of the `n1 > n2` matrix sums to one.

# Related

  - [`schur_augmentation`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
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
    schur_augmentation(A::MatNum, B::MatNum, C::MatNum, gamma::Number) -> MatNum

Augment one half's covariance block with the information in the cross block.

This is the ``\\hat{\\mathbf{\\Sigma}}_{11}`` of [`SchurComplementHierarchicalRiskParity`](@ref)'s mathematical definition. It scales the Schur complement of `C` by `gamma`, carries the result back to the size of `A` with a [`symmetric_step_up_matrix`](@ref), and symmetrises.

# Arguments

  - `A`: Covariance block of the half being augmented.
  - `B`: Cross-covariance between the two halves, with `A`'s assets along the rows.
  - `C`: Covariance block of the **other** half.
  - `gamma`: Interpolation parameter in `[0, 1]`.

# Returns

  - `A_aug::MatNum`: The augmented block, of the same size as `A`, and symmetric.

# Details

  - `A` is returned untouched when `gamma` is zero, or when either half holds one asset. A one-asset half leaves the step-up matrix nothing to average over.
  - The result is **not** guaranteed positive definite. [`schur_complement_weights`](@ref) decides what to do about that, from [`SchurComplementParams`](@ref)`.flag`.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`symmetric_step_up_matrix`](@ref)
  - [`naive_portfolio_risk`](@ref)
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
    naive_portfolio_risk(r::Sd_Var, sigma::MatNum) -> Number

Compute the risk of the naive risk parity portfolio a covariance matrix implies.

The weights are the **inverse diagonal** of `sigma`, normalised to sum to one, for both risk measures. The measure changes only what is done with the quadratic form. `sigma` here is an augmented block, so its diagonal is not a plain asset variance.

# Arguments

  - `r`: Risk measure. [`Variance`](@ref) returns the quadratic form; [`StandardDeviation`](@ref) returns its square root.
  - `sigma`: Covariance matrix, usually the output of [`schur_augmentation`](@ref).

# Returns

  - `risk::Number`: The portfolio risk under `r`.

# Details

  - The weights are inverse **variance**, not inverse volatility, under both measures. This matches the naive risk parity allocation of [`HierarchicalRiskParity`](@ref) with a [`Variance`](@ref) measure, and is why `gamma = 0` reproduces that optimiser exactly.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`Sd_Var`](@ref)
  - [`schur_augmentation`](@ref)
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
    schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt,
                             wb::WeightBounds, params::SchurComplementParams,
                             gamma::Option{<:Number} = nothing) -> Tuple

Run the Schur complement recursion at one value of ``\\gamma``.

Splits `items` in half repeatedly, augments each half's covariance block with [`schur_augmentation`](@ref), and divides the weight between the halves in inverse proportion to their augmented risks. This method takes a [`NonMonotonicSchurComplement`](@ref); the [`MonotonicSchurComplement`](@ref) method searches over ``\\gamma`` and delegates here.

# Arguments

  - `pr`: Prior result. Its covariance seeds the recursion and its `X` sizes the weight vector.
  - `items`: The leaf orders to split. The recursion starts from one entry, the whole dendrogram order.
  - `wb`: Resolved weight bounds, which clamp each split factor.
  - `params`: The bundle's parameters. `gamma`, `pdm`, `flag` and `r` are all read here.
  - `gamma`: A value that overrides `params.gamma`, or `nothing` to use the field. The monotonic search passes each candidate this way.

# Returns

  - `(w, gamma, r)::Tuple`: The unnormalised weight vector, the ``\\gamma`` the recursion ran at, and the risk measure resolved by [`factory`](@ref). `w` is `nothing` when `params.flag` is `false` and an augmented block was not positive definite.

# Details

  - The running covariance matrix is **updated in place**: each augmented block is written back, so a later split sees the earlier augmentation. A half holding one asset is left unaugmented.
  - With `params.flag` true, `params.pdm` repairs an augmented block that is not positive definite, and a failed repair raises an `ArgumentError` naming `gamma`.
  - With `params.flag` false, no repair happens and the whole allocation is abandoned. The monotonic search wants that; [`assert_schur_weights`](@ref) turns it into a message for anyone else.

# Related

  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`SchurComplementParams`](@ref)
  - [`schur_augmentation`](@ref)
  - [`naive_portfolio_risk`](@ref)
  - [`split_factor_weight_constraints`](@ref)
  - [`assert_schur_weights`](@ref)
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
                    # Three values, like every other return of this method, and the real
                    # `gamma` so a caller can name the value that failed. The monotonic
                    # search reads `[1]` and treats `nothing` as an infinite risk; a caller
                    # that destructures three names must not get a `BoundsError` instead.
                    return nothing, gamma, r
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
    # The resolved measure is threaded out beside `gamma`. It is computed here and was
    # discarded before, so the result had no way to name what it optimised.
    return w, gamma, r
end
"""
    schur_complement_binary_search(objective::Function, lgamma::Number, hgamma::Number,
                                   lrisk::Number, tol::Number = 1e-4,
                                   iter::Option{<:Integer} = nothing,
                                   strict::Bool = false) -> Tuple

Bisect a bracket that holds the variance-minimising ``\\gamma``.

[`MonotonicSchurComplement`](@ref)'s coarse scan hands over a bracket in which the portfolio variance stops falling. This method halves that bracket until it is narrower than `tol`, keeping the midpoint only when its variance beats both the current lower end and the point one `tol` below it.

# Arguments

  - `objective`: Takes a ``\\gamma`` and returns `(w, risk)`. `risk` is `typemax` when the allocation fails.
  - `lgamma`: Lower end of the bracket, and the incumbent.
  - `hgamma`: Upper end of the bracket.
  - `lrisk`: The variance already measured at `lgamma`. It is the value a midpoint must beat.
  - `tol`: Width at which the bracket is narrow enough, and the step used for the one-sided derivative test.
  - `iter`: Iteration budget. `nothing` derives one from the bracket and `tol`.
  - `strict`: Whether a bracket that never narrows to `tol` raises rather than warns.

# Returns

  - `(w, gamma)::Tuple`: The weight vector of the best ``\\gamma`` seen, and that ``\\gamma``.

# Details

  - The returned `w` is the weight vector of the **last** midpoint evaluated, which is the incumbent's whenever the loop ends by narrowing the bracket.
  - Failing to narrow within `iter` is reported through [`strict_diagnostic`](@ref), so the search returns its incumbent rather than failing.

# Related

  - [`MonotonicSchurComplement`](@ref)
  - [`schur_complement_weights`](@ref)
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
    strict_diagnostic(msg, strict)
    return w, lgamma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Search ``[0, \\gamma]`` for the variance-minimising Schur complement parameter, then allocate at it.

Scans `params.alg.N` values across the range and stops at the first one whose portfolio variance is no lower than its predecessor. It then bisects the bracket around that turning point with [`schur_complement_binary_search`](@ref). When the variance is still falling at the top of the range, that top value is used. Every evaluation delegates to the [`NonMonotonicSchurComplement`](@ref) method with the positive-definite repair off, so a ``\\gamma`` that fails scores an infinite variance.

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
        wi, gi, _ = schur_complement_weights(pr, items, wb, nm_params)
        return wi, gi, r
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
            wi, gi = schur_complement_binary_search(objective, gammas[lidx], gammas[i],
                                                    risks[lidx], params.alg.tol,
                                                    params.alg.iter, params.alg.strict)
            return wi, gi, r
        end
    end
    # If there's no turning point in the range of gammas, check the derivative at the last gamma.
    if risk <= objective(max_gamma - params.alg.tol)[2]
        return w, max_gamma, r
    end
    # If the turning point exists and was not found within the range, or the last gamma, it is between the last two gammas.
    wi, gi = schur_complement_binary_search(objective, gammas[end - 1], gammas[end],
                                            risks[end - 1], params.alg.tol, params.alg.iter,
                                            params.alg.strict)
    return wi, gi, r
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that [`schur_complement_weights`](@ref) produced a weight vector.

A [`SchurComplementParams`](@ref) with `flag = false` does not repair an augmented block that is not positive definite. It abandons the allocation and returns `nothing` instead, which is what the monotonic search needs. A caller that keeps the weights needs the reason in words.

# Arguments

  - `w`: Weight vector returned by [`schur_complement_weights`](@ref), or `nothing`.
  - `gamma`: The value of `gamma` the allocation ran with.

# Validation

  - `!isnothing(w)`.

# Returns

  - `nothing`.

# Related

  - [`schur_complement_weights`](@ref)
  - [`SchurComplementParams`](@ref)
"""
function assert_schur_weights(w::Option{<:VecNum}, gamma::Number)::Nothing
    @argcheck(!isnothing(w),
              ArgumentError("Augmented matrix is not positive definite at gamma = $gamma, and `flag = false` disables the positive-definite repair. Set `flag = true`, use `MonotonicSchurComplement()`, or reduce gamma."))
    return nothing
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
    sh = reset_time_dependent_estimator(sh)
    rd = returns_result_picker(rd, sh.opt.brt)
    pr = prior(sh.opt.pe, rd; dims = dims)
    X = pr.X
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(sh.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     x_src = sh.opt.x_src, z_src = sh.opt.z_src)
    items = [clr.res.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(X, 2),
                                   strict = sh.opt.strict, datatype = eltype(X))
    w, gamma, r = schur_complement_weights(pr, items, wb, sh.params)
    assert_schur_weights(w, gamma)
    retcode, w = finalise_weight_bounds(sh.opt.wf, wb, w)
    return SchurComplementHierarchicalRiskParityResult(; pr = pr, wb = wb, clr = clr, r = r,
                                                       gamma = gamma, retcode = retcode,
                                                       w = w, fb = nothing)
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
    sh = reset_time_dependent_estimator(sh)
    rd = returns_result_picker(rd, sh.opt.brt)
    pr = prior(sh.opt.pe, rd; dims = dims)
    X = pr.X
    # No `branchorder`: recursive bisection splits `clr.res.order`, so the leaf
    # permutation is the algorithm's input and must stay `:optimal` (ADR 0055).
    clr = clusterise(sh.opt.cle, pr; rd = rd, iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     x_src = sh.opt.x_src, z_src = sh.opt.z_src)
    items = [clr.res.order]
    wb = weight_bounds_constraints(sh.opt.wb, sh.opt.sets; N = size(X, 2),
                                   strict = sh.opt.strict, datatype = eltype(X))
    params = sh.params
    gammas = Vector{eltype(X)}(undef, length(params))
    rs = Vector{Any}(undef, length(params))
    w = zeros(eltype(X), size(X, 2))
    for (i, ps) in enumerate(params)
        wi, gamma, ri = schur_complement_weights(pr, items, wb, ps)
        assert_schur_weights(wi, gamma)
        w .+= ps.r.settings.scale * wi
        gammas[i] = gamma
        rs[i] = ri
    end
    retcode, w = finalise_weight_bounds(sh.opt.wf, wb, w / sum(w))
    return SchurComplementHierarchicalRiskParityResult(; pr = pr, wb = wb, clr = clr,
                                                       r = [rs...], gamma = gammas,
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

# Details

Unlike [`HierarchicalEqualRiskContribution`](@ref) and [`NestedClustered`](@ref), this optimiser accepts no `branchorder` keyword. Recursive bisection allocates by splitting the dendrogram's leaf permutation, so that permutation is the algorithm's input rather than a presentation detail, and the clusterisation always runs with the optimal ordering. A `branchorder` passed here is absorbed by `kwargs` and ignored. See ADR 0055.

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
