"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base clustering optimisation estimators.

These are intermediate configuration types used in hierarchical/clustering optimisation pipelines.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`HierarchicalOptimiser`](@ref)
"""
abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for clustering-based portfolio optimisation estimators.

Clustering optimisation estimators use asset clustering to decompose the portfolio optimisation problem. Subtypes include HRP, HERC, and SCHRP.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
"""
abstract type ClusteringOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the estimator's own problem-definition fields, the inner optimiser, or the fallback carry time-dependent constraints.
"""
function is_time_dependent(opt::ClusteringOptimisationEstimator)
    return (!isempty(time_dependent_fields(opt)) ||
            is_time_dependent(opt.opt) ||
            is_time_dependent(opt.fb))
end
function assert_time_dependent_fold_count(opt::ClusteringOptimisationEstimator, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    assert_time_dependent_fold_count(opt.opt, n, all_binds)
    assert_time_dependent_fold_count(opt.fb, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve time-dependent constraints for the fold described by `ctx`: the estimator's own scheduled fields (risk measures, scalarisers, fallback, …) are swapped for their per-fold values, then the inner optimiser and the (possibly just-swapped-in) fallback are recursed into with the same context.

[`NestedClustered`](@ref) overrides this with its own method resolving its own fields and inner estimators.
"""
function update_time_dependent_estimator(opt::ClusteringOptimisationEstimator,
                                         ctx::TimeDependentContext, all_binds::Bool = true)
    if !is_time_dependent(opt)
        return opt
    end
    opt = update_time_dependent_fields(opt, ctx, all_binds)
    return rebuild_estimator(opt,
                             (;
                              opt = update_time_dependent_estimator(opt.opt, ctx,
                                                                    all_binds),
                              fb = update_time_dependent_estimator(opt.fb, ctx, all_binds)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace time-dependent constraints with their static defaults, both on the estimator's own fields and by recursing into the inner optimiser and fallback.

[`NestedClustered`](@ref) overrides this with its own method resetting its own fields and inner estimators.
"""
function reset_time_dependent_estimator(opt::ClusteringOptimisationEstimator)
    if !is_time_dependent(opt)
        return opt
    end
    opt = reset_time_dependent_fields(opt)
    return rebuild_estimator(opt,
                             (; opt = reset_time_dependent_estimator(opt.opt),
                              fb = reset_time_dependent_estimator(opt.fb)))
end
"""
$(DocStringExtensions.TYPEDEF)

Shared field core for hierarchical (clustering-based) optimisation results.

Holds the fields common to [`HierarchicalRiskParityResult`](@ref) and [`HierarchicalEqualRiskContributionResult`](@ref), and is embedded as the first field (`hr`) of each — analogous to how [`JuMPOptimisationResult`](@ref) is embedded as `jr` on the JuMP side. Each leaf keeps only the measures and scalarisers its own estimator carries, plus the trailing `fb`.

## The core carries no `fb`

ADR 0011 fixes `fb` as the **last** field of each concrete result and keeps it out of the core, because every optimisation result already ends in `fb`. Both leaves end in `fb`, so ADR 0011's one generic `factory(res, fb)` rebuilds them by the same convention and needs no change. The core sits off [`AbstractResult`](@ref)'s optimisation branch, so that generic never reaches it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalResult(;
        pr::Option{<:AbstractPriorResult},
        clr::Option{<:AbstractClusteringResult},
        wb::Option{<:WeightBounds},
        fees::Option{<:Fees},
        retcode::OptimisationReturnCode,
        w::Option{<:VecNum}
    ) -> HierarchicalResult

Keywords correspond to the struct's fields.

# Related

  - [`BaseHierarchicalOptimisationResult`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
"""
@concrete struct HierarchicalResult <: BaseHierarchicalOptimisationResult
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:clr])
    """
    clr
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:fees])
    """
    fees
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:pw])
    """
    w
    function HierarchicalResult(pr::Option{<:AbstractPriorResult},
                                clr::Option{<:AbstractClusteringResult},
                                wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                                retcode::OptimisationReturnCode, w::Option{<:VecNum})
        return new{typeof(pr), typeof(clr), typeof(wb), typeof(fees), typeof(retcode),
                   typeof(w)}(pr, clr, wb, fees, retcode, w)
    end
end
function HierarchicalResult(; pr::Option{<:AbstractPriorResult},
                            clr::Option{<:AbstractClusteringResult},
                            wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                            retcode::OptimisationReturnCode,
                            w::Option{<:VecNum})::HierarchicalResult
    return HierarchicalResult(pr, clr, wb, fees, retcode, w)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for [`HierarchicalRiskParity`](@ref).

Carries the shared core as `hr`, plus the one measure and the one scalariser its estimator holds, both stored **resolved**.

Every property of the core forwards through this type, so `res.w`, `res.pr`, `res.clr`, `res.wb`, `res.fees` and `res.retcode` read as if the fields were flat.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalRiskParityResult(;
        hr::HierarchicalResult,
        r::BaseRM_VecBaseRM,
        sca::Scalariser,
        fb::Option{<:OptE_Opt}
    ) -> HierarchicalRiskParityResult

Keywords correspond to the struct's fields.

# Related

  - [`HierarchicalOptimisationResult`](@ref)
  - [`HierarchicalResult`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
"""
@concrete struct HierarchicalRiskParityResult <: HierarchicalOptimisationResult
    """
    Shared hierarchical result core, see [`HierarchicalResult`](@ref).
    """
    hr
    """
    $(field_dict[:r_res])
    """
    r
    """
    $(field_dict[:sca_res])
    """
    sca
    """
    $(field_dict[:fb])
    """
    fb
    function HierarchicalRiskParityResult(hr::HierarchicalResult, r::BaseRM_VecBaseRM,
                                          sca::Scalariser, fb::Option{<:OptE_Opt})
        return new{typeof(hr), typeof(r), typeof(sca), typeof(fb)}(hr, r, sca, fb)
    end
end
function HierarchicalRiskParityResult(; hr::HierarchicalResult, r::BaseRM_VecBaseRM,
                                      sca::Scalariser,
                                      fb::Option{<:OptE_Opt})::HierarchicalRiskParityResult
    return HierarchicalRiskParityResult(hr, r, sca, fb)
end
# Unique fields resolve directly; every other property forwards into the embedded core, so
# `res.w`, `res.pr`, `res.clr`, `res.wb`, `res.fees` and `res.retcode` stay source-compatible
# across the split. The rule is declared per leaf rather than on the abstract type, because
# `SchurComplementHierarchicalRiskParityResult` joins the family with flat fields and has no `hr`.
@forward_properties HierarchicalRiskParityResult begin
    forward(hr)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type for [`HierarchicalEqualRiskContribution`](@ref).

Carries the shared core as `hr`, plus the **two** measures and **two** scalarisers its estimator holds — the intra-cluster pair and the inter-cluster pair — all stored **resolved**.

The differing arity against [`HierarchicalRiskParityResult`](@ref) is why the shared `HierarchicalResult` split into two leaves rather than growing `Option` slots or union-typed fields.

Every property of the core forwards through this type, so `res.w`, `res.pr`, `res.clr`, `res.wb`, `res.fees` and `res.retcode` read as if the fields were flat.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalEqualRiskContributionResult(;
        hr::HierarchicalResult,
        ri::BaseRM_VecBaseRM,
        ro::BaseRM_VecBaseRM,
        scai::Scalariser,
        scao::Scalariser,
        fb::Option{<:OptE_Opt}
    ) -> HierarchicalEqualRiskContributionResult

Keywords correspond to the struct's fields.

# Related

  - [`HierarchicalOptimisationResult`](@ref)
  - [`HierarchicalResult`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
"""
@concrete struct HierarchicalEqualRiskContributionResult <: HierarchicalOptimisationResult
    """
    Shared hierarchical result core, see [`HierarchicalResult`](@ref).
    """
    hr
    """
    $(field_dict[:ri_res])
    """
    ri
    """
    $(field_dict[:ro_res])
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
    $(field_dict[:fb])
    """
    fb
    function HierarchicalEqualRiskContributionResult(hr::HierarchicalResult,
                                                     ri::BaseRM_VecBaseRM,
                                                     ro::BaseRM_VecBaseRM, scai::Scalariser,
                                                     scao::Scalariser,
                                                     fb::Option{<:OptE_Opt})
        return new{typeof(hr), typeof(ri), typeof(ro), typeof(scai), typeof(scao),
                   typeof(fb)}(hr, ri, ro, scai, scao, fb)
    end
end
function HierarchicalEqualRiskContributionResult(; hr::HierarchicalResult,
                                                 ri::BaseRM_VecBaseRM, ro::BaseRM_VecBaseRM,
                                                 scai::Scalariser, scao::Scalariser,
                                                 fb::Option{<:OptE_Opt})::HierarchicalEqualRiskContributionResult
    return HierarchicalEqualRiskContributionResult(hr, ri, ro, scai, scao, fb)
end
@forward_properties HierarchicalEqualRiskContributionResult begin
    forward(hr)
end
"""
$(DocStringExtensions.TYPEDEF)

Base configuration for hierarchical clustering-based portfolio optimisers.

`HierarchicalOptimiser` combines a prior estimator, a clustering estimator, and weight bound/fee specifications to provide a reusable base configuration for hierarchical optimisers (HRP, HERC, SCHRP, etc.).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalOptimiser(;
        pe::TD{<:PrE_Pr} = EmpiricalPrior(),
        cle::TD{<:HClE_HCl} = ClustersEstimator(),
        slv::Option{<:Slv_VecSlv} = nothing,
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        brt::Bool = false,
        x_src::Symbol = :prior,
        z_src::Symbol = :data,
        strict::Bool = false
    ) -> HierarchicalOptimiser

Keywords correspond to the struct's fields. Fields typed [`TD_Option`](@ref) or [`TD`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value; a cross-validation fold loop resolves it per fold, and a fold-less `optimise` runs with the field at its static default. The problem definition — the prior estimator, clustering estimator, weight finaliser and asset sets as much as the bounds and fees — may therefore vary over folds; execution control (`slv`, `brt`, `x_src`, `z_src`, `strict`) stays static.

## Validation

  - `x_src in (:prior, :data)`.
  - `z_src in (:prior, :data)`.
  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - If any field holds a [`TimeDependent`](@ref): every vector entry is test-substituted through this constructor so type compatibility errors surface immediately.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> HierarchicalOptimiser()
HierarchicalOptimiser
      pe ┼ EmpiricalPrior
         │        ce ┼ PortfolioOptimisersCovariance
         │           │   ce ┼ Covariance
         │           │      │    me ┼ SimpleExpectedReturns
         │           │      │       │   w ┴ nothing
         │           │      │    ce ┼ GeneralCovariance
         │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │           │      │       │    w ┴ nothing
         │           │      │   alg ┴ FullMoment()
         │           │   mp ┼ MatrixProcessing
         │           │      │     pdm ┼ Posdef
         │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │           │      │      dn ┼ nothing
         │           │      │      dt ┼ nothing
         │           │      │     alg ┼ nothing
         │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │        me ┼ SimpleExpectedReturns
         │           │   w ┴ nothing
         │   horizon ┴ nothing
     cle ┼ ClustersEstimator
         │    ce ┼ PortfolioOptimisersCovariance
         │       │   ce ┼ Covariance
         │       │      │    me ┼ SimpleExpectedReturns
         │       │      │       │   w ┴ nothing
         │       │      │    ce ┼ GeneralCovariance
         │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │       │      │       │    w ┴ nothing
         │       │      │   alg ┴ FullMoment()
         │       │   mp ┼ MatrixProcessing
         │       │      │     pdm ┼ Posdef
         │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │       │      │      dn ┼ nothing
         │       │      │      dt ┼ nothing
         │       │      │     alg ┼ nothing
         │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │    de ┼ Distance
         │       │   power ┼ nothing
         │       │     alg ┴ CanonicalDistance()
         │   alg ┼ HClustAlgorithm
         │       │   linkage ┴ Symbol: :ward
         │   onc ┼ OptimalNumberClusters
         │       │   max_k ┼ nothing
         │       │     alg ┼ SecondOrderDifference
         │       │         │   alg ┼ StandardisedValue
         │       │         │       │   mv ┼ MeanValue
         │       │         │       │      │   w ┴ nothing
         │       │         │       │   sv ┼ StdValue
         │       │         │       │      │           w ┼ nothing
         │       │         │       │      │   corrected ┴ Bool: true
     slv ┼ nothing
      wb ┼ WeightBounds
         │   lb ┼ Float64: 0.0
         │   ub ┴ Float64: 1.0
    fees ┼ nothing
    sets ┼ nothing
      wf ┼ IterativeWeightFinaliser
         │   iter ┴ Int64: 100
     brt ┼ Bool: false
   x_src ┼ Symbol: :prior
   z_src ┼ Symbol: :data
  strict ┴ Bool: false
```

# Related

  - [`BaseClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct HierarchicalOptimiser <: BaseClusteringOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    @vprop pe
    """
    $(field_dict[:cle])
    """
    cle
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:fees])
    """
    @fprop @vprop fees
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:x_src])
    """
    x_src
    """
    $(field_dict[:z_src])
    """
    z_src
    """
    $(field_dict[:strict_opt])
    """
    strict
    function HierarchicalOptimiser(pe::TD{<:PrE_Pr}, cle::TD{<:HClE_HCl},
                                   slv::Option{<:Slv_VecSlv}, wb::TD_Option{<:WbE_Wb},
                                   fees::TD_Option{<:FeesE_Fees},
                                   sets::TD_Option{<:UniverseSets},
                                   wf::TD{<:WeightFinaliser}, brt::Bool, x_src::Symbol,
                                   z_src::Symbol, strict::Bool)
        assert_source_selector(x_src, :x_src)
        assert_source_selector(z_src, :z_src)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        assert_time_dependent_substitution(HierarchicalOptimiser,
                                           (; pe, cle, slv, wb, fees, sets, wf, brt, x_src,
                                            z_src, strict),
                                           hierarchical_optimiser_td_defaults())
        return new{typeof(pe), typeof(cle), typeof(slv), typeof(wb), typeof(fees),
                   typeof(sets), typeof(wf), typeof(brt), typeof(x_src), typeof(z_src),
                   typeof(strict)}(pe, cle, slv, wb, fees, sets, wf, brt, x_src, z_src,
                                   strict)
    end
end
function HierarchicalOptimiser(; pe::TD{<:PrE_Pr} = EmpiricalPrior(),
                               cle::TD{<:HClE_HCl} = ClustersEstimator(),
                               slv::Option{<:Slv_VecSlv} = nothing,
                               wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                               fees::TD_Option{<:FeesE_Fees} = nothing,
                               sets::TD_Option{<:UniverseSets} = nothing,
                               wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                               brt::Bool = false, x_src::Symbol = :prior,
                               z_src::Symbol = :data,
                               strict::Bool = false)::HierarchicalOptimiser
    return HierarchicalOptimiser(pe, cle, slv, wb, fees, sets, wf, brt, x_src, z_src,
                                 strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`HierarchicalOptimiser`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`HierarchicalOptimiser`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function hierarchical_optimiser_td_defaults()::NamedTuple
    return (; pe = EmpiricalPrior(), cle = ClustersEstimator(), wb = WeightBounds(),
            wf = IterativeWeightFinaliser())
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the [`HierarchicalOptimiser`](@ref) requires previous portfolio weights (based on fee structure and time-dependent constraints).

# Related

  - [`needs_previous_weights`](@ref)
  - [`HierarchicalOptimiser`](@ref)
"""
function needs_previous_weights(opt::HierarchicalOptimiser)
    return needs_previous_weights(opt.fees) ||
           any(f -> needs_previous_weights(getfield(opt, f)), time_dependent_fields(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`HierarchicalOptimiser`](@ref) fields that may hold a [`TimeDependent`](@ref).

# Related

  - [`HierarchicalOptimiser`](@ref)
  - [`TimeDependent`](@ref)
  - [`TimeDependentContext`](@ref)
"""
function time_dependent_field_defaults(::HierarchicalOptimiser)::NamedTuple
    return hierarchical_optimiser_td_defaults()
end
"""
    unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum,
                           fees::Option{<:Fees} = nothing) -> Vector

Compute the expected risk of each asset held alone.

The ``i``-th entry is the risk of the portfolio whose weight vector is one in position ``i`` and zero everywhere else, so the result has one entry per **asset**, not per cluster. For [`Variance`](@ref) the vector is the diagonal of the covariance matrix. The hierarchical optimisers invert this vector to build a naive risk parity allocation inside a cluster.

# Arguments

  - `r`: Risk measure, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees to charge against each unit portfolio, or `nothing`.

# Returns

  - `rk::Vector`: Expected risk of each asset held alone, of length `size(X, 2)`.

# Related

  - [`unitary_expected_risks!`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
function unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum,
                                fees::Option{<:Fees} = nothing)
    wk = zeros(eltype(X), size(X, 2))
    rk = Vector{eltype(X)}(undef, size(X, 2))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return rk
end
"""
    unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure,
                            X::MatNum, fees::Option{<:Fees} = nothing) -> Nothing

Write the expected risk of each asset held alone into `rk`.

The in-place form of [`unitary_expected_risks`](@ref), for a caller that reuses one buffer across several risk measures.

# Arguments

  - `wk`: Scratch weight vector, of length `size(X, 2)`. It must arrive all zero, and it leaves all zero: each iteration raises one entry to one and lowers it again.
  - `rk`: Output risk vector, of length `size(X, 2)`. It is overwritten in full.
  - `r`: Risk measure, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees to charge against each unit portfolio, or `nothing`.

# Returns

  - `nothing`. The result is `rk`.

# Related

  - [`unitary_expected_risks`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
function unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure,
                                 X::MatNum, fees::Option{<:Fees} = nothing)
    fill!(rk, zero(eltype(X)))
    for i in eachindex(wk)
        wk[i] = one(eltype(X))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(X))
    end
    return nothing
end

export HierarchicalResult, HierarchicalRiskParityResult,
       HierarchicalEqualRiskContributionResult, HierarchicalOptimiser
