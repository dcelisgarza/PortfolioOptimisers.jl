"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the shared configuration that a hierarchical optimiser holds in its `opt` field.

A subtype is not an optimiser, so [`optimise`](@ref) does not take one. [`HierarchicalOptimiser`](@ref) is the one subtype, and [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) each hold one.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`HierarchicalOptimiser`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
"""
abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the optimisers that cluster the assets and allocate over the clusters.

It has four subtypes. [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) hold a [`HierarchicalOptimiser`](@ref) in the field `opt` and a fallback in the field `fb`. The methods of this file that dispatch on the family read those two fields, so a new subtype that holds them needs no method of its own for time-dependent constraints. [`NestedClustered`](@ref) holds neither field, and it overrides all four of those methods.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
  - [`SchurComplementHierarchicalRiskParity`](@ref)
  - [`NestedClustered`](@ref)
  - [`BaseClusteringOptimisationEstimator`](@ref)
"""
abstract type ClusteringOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a clustering optimiser, or an estimator it holds, carries a [`TimeDependent`](@ref) schedule.

The optimiser is time-dependent when any one of three things holds:

  - one of its own fields holds a schedule that [`time_dependent_fields`](@ref) returns;
  - its inner optimiser `opt.opt` is time-dependent;
  - its fallback `opt.fb` is time-dependent.

[`NestedClustered`](@ref) overrides this with its own method.

# Arguments

  - `opt`: Clustering optimiser that holds the fields `opt` and `fb`.

# Returns

  - `flag::Bool`: `true` when a fold loop must resolve a schedule before the optimiser runs.

# Related

  - [`update_time_dependent_estimator`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`time_dependent_fields`](@ref)
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

Replace every [`TimeDependent`](@ref) schedule of a clustering optimiser with its value at the fold that `ctx` describes.

The fallback is resolved after the optimiser's own fields, so a fallback that a schedule selects for this fold is itself resolved. [`NestedClustered`](@ref) overrides this with its own method.

# Algorithm

 1. Return `opt` unchanged when [`is_time_dependent`](@ref) is `false`.
 2. Replace each scheduled field of `opt` with its value at the fold, through [`update_time_dependent_fields`](@ref). This gives the new `opt`.
 3. Resolve the inner optimiser `opt.opt` and the fallback `opt.fb` of the new `opt` with the same `ctx` and `all_binds`.
 4. Rebuild `opt` with the two resolved estimators, through [`rebuild_estimator`](@ref).

# Arguments

  - `opt`: Clustering optimiser that holds the fields `opt` and `fb`.
  - `ctx`: Fold context of the current fold.
  - `all_binds`: Whether the fold loop resolves the schedules bound to `:nearest` as well as those bound to `:outermost`.

# Returns

  - `opt`: The optimiser with no schedule left for this fold loop, of the same type as the input.

# Related

  - [`is_time_dependent`](@ref)
  - [`reset_time_dependent_estimator`](@ref)
  - [`TimeDependentContext`](@ref)
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

Replace every [`TimeDependent`](@ref) schedule of a clustering optimiser with its static default.

A fold-less [`optimise`](@ref) calls it first, so the optimiser runs with each scheduled field at its default. [`NestedClustered`](@ref) overrides this with its own method.

# Algorithm

 1. Return `opt` unchanged when [`is_time_dependent`](@ref) is `false`.
 2. Replace each scheduled field of `opt` with its static default, through [`reset_time_dependent_fields`](@ref). This gives the new `opt`.
 3. Reset the inner optimiser `opt.opt` and the fallback `opt.fb` of the new `opt`.
 4. Rebuild `opt` with the two reset estimators, through [`rebuild_estimator`](@ref).

# Arguments

  - `opt`: Clustering optimiser that holds the fields `opt` and `fb`.

# Returns

  - `opt`: The optimiser with no schedule left, of the same type as the input.

# Related

  - [`is_time_dependent`](@ref)
  - [`update_time_dependent_estimator`](@ref)
  - [`hierarchical_optimiser_td_defaults`](@ref)
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

Holds the fields that the results of hierarchical risk parity and hierarchical equal risk contribution share.

[`HierarchicalRiskParityResult`](@ref) and [`HierarchicalEqualRiskContributionResult`](@ref) each hold it as their first field, `hr`, as a JuMP result holds a [`JuMPOptimisationResult`](@ref) as `jr`. Each of the two adds the risk measures and the scalarisers of its own estimator, and ends in the fallback field `fb`.

The core holds no `fb`. The generic `factory(res, fb)` rebuilds a result from all of its fields but the last, and puts `fb` in the last field. Each of the two results ends in `fb`, so that method keeps the core unchanged. The core is not an [`OptimisationResult`](@ref), so `factory(res, fb)` does not take it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalResult(;
        pr::Option{<:AbstractPriorResult},
        clr::Option{<:AbstractClusteringResult},
        wb::Option{<:WeightBounds},
        fees::Option{<:Fees},
        retcode::OptimisationReturnCode,
        w::Option{<:VecNum},
        imsk::Option{<:BitVector} = nothing
    ) -> HierarchicalResult

Keywords correspond to the struct's fields.

The keyword constructor expands `w` from the investable assets to the full universe, through [`expand_investable_weights`](@ref), and then calls the positional constructor. [`HierarchicalRiskParity`](@ref) and [`HierarchicalEqualRiskContribution`](@ref) build their results through the keyword constructor, so the stored `w` has one entry per asset of the universe, and a non-investable asset holds zero. When `imsk` is `nothing`, `w` is stored as it is given. The positional constructor always stores `w` as it is given, so it is the constructor for weights that are already expanded.

# Related

  - [`BaseHierarchicalOptimisationResult`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
  - [`investable_reduction`](@ref)
  - [`expand_investable_weights`](@ref)
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
    """
    $(field_dict[:imsk])
    """
    imsk
    function HierarchicalResult(pr::Option{<:AbstractPriorResult},
                                clr::Option{<:AbstractClusteringResult},
                                wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                                retcode::OptimisationReturnCode, w::Option{<:VecNum},
                                imsk::Option{<:BitVector})
        return new{typeof(pr), typeof(clr), typeof(wb), typeof(fees), typeof(retcode),
                   typeof(w), typeof(imsk)}(pr, clr, wb, fees, retcode, w, imsk)
    end
end
function HierarchicalResult(; pr::Option{<:AbstractPriorResult},
                            clr::Option{<:AbstractClusteringResult},
                            wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                            retcode::OptimisationReturnCode, w::Option{<:VecNum},
                            imsk::Option{<:BitVector} = nothing)::HierarchicalResult
    return HierarchicalResult(pr, clr, wb, fees, retcode,
                              expand_investable_weights(imsk, w), imsk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a clustering covers exactly the assets the optimisation allocates over.

[`clusterise`](@ref) returns a fitted [`AbstractClusteringResult`](@ref) unchanged, so a caller who states one instead of an estimator also states the leaf order, and nothing slices that order. Under an Investable Mask, or inside a subset or a nested view, the optimisation runs on fewer assets than the caller clustered. The leaf order would then index the wrong columns, and the optimisation would return wrong weights without an error. This check raises before the leaf order is read.

To keep one clustering method while the universe changes, state a [`ClustersEstimator`](@ref). It refits on each window, so its result always matches.

# Algorithm

 1. Read one cluster label per asset with [`assignments`](@ref).
 2. Throw a `DimensionMismatch` when their count is not `N`.

# Arguments

  - `clr`: Clustering result, fitted by [`clusterise`](@ref) or stated by the caller.
  - `N`: Number of assets the optimisation runs on, after the reduction.

# Validation

  - The clustering carries one label per asset of the reduced universe, else a `DimensionMismatch` is raised.

# Returns

  - `nothing`.

# Related

  - [`clusterise`](@ref)
  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`investable_reduction`](@ref)
"""
function assert_clustering_universe(clr::AbstractClusteringResult, N::Integer)::Nothing
    Nc = length(assignments(clr))
    @argcheck(Nc == N,
              DimensionMismatch("the clustering covers $(Nc) assets, but the optimisation runs on $(N). A fitted clustering result is used exactly as stated, so it must be stated over the universe the optimisation runs on. State a `ClustersEstimator` instead when the universe changes."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the weights and the fitted state of a hierarchical risk parity optimisation.

[`HierarchicalRiskParity`](@ref) returns it. It holds the shared [`HierarchicalResult`](@ref) as `hr`, and the risk measure and the scalariser of its estimator, with the measure stored resolved.

Every property of `hr` reads through this type, so `res.w`, `res.pr`, `res.clr`, `res.wb`, `res.fees`, `res.retcode` and `res.imsk` read as if they were fields of `res`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalRiskParityResult(;
        hr::HierarchicalResult,
        r::BaseRM_VecBaseRM,
        sca::Scalariser,
        fb::Option{<:OptE_Opt_FbChain}
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
    $(field_dict[:hr_core])
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
    $(field_dict[:fb_res])
    """
    fb
    function HierarchicalRiskParityResult(hr::HierarchicalResult, r::BaseRM_VecBaseRM,
                                          sca::Scalariser, fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(hr), typeof(r), typeof(sca), typeof(fb)}(hr, r, sca, fb)
    end
end
function HierarchicalRiskParityResult(; hr::HierarchicalResult, r::BaseRM_VecBaseRM,
                                      sca::Scalariser,
                                      fb::Option{<:OptE_Opt_FbChain})::HierarchicalRiskParityResult
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

Holds the weights and the fitted state of a hierarchical equal risk contribution optimisation.

[`HierarchicalEqualRiskContribution`](@ref) returns it. It holds the shared [`HierarchicalResult`](@ref) as `hr`, and the two risk measures and the two scalarisers of its estimator, one pair inside the clusters and one pair across them. Both measures are stored resolved.

Every property of `hr` reads through this type, so `res.w`, `res.pr`, `res.clr`, `res.wb`, `res.fees`, `res.retcode` and `res.imsk` read as if they were fields of `res`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalEqualRiskContributionResult(;
        hr::HierarchicalResult,
        ri::BaseRM_VecBaseRM,
        ro::BaseRM_VecBaseRM,
        scai::Scalariser,
        scao::Scalariser,
        fb::Option{<:OptE_Opt_FbChain}
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
    $(field_dict[:hr_core])
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
    $(field_dict[:fb_res])
    """
    fb
    function HierarchicalEqualRiskContributionResult(hr::HierarchicalResult,
                                                     ri::BaseRM_VecBaseRM,
                                                     ro::BaseRM_VecBaseRM, scai::Scalariser,
                                                     scao::Scalariser,
                                                     fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(hr), typeof(ri), typeof(ro), typeof(scai), typeof(scao),
                   typeof(fb)}(hr, ri, ro, scai, scao, fb)
    end
end
function HierarchicalEqualRiskContributionResult(; hr::HierarchicalResult,
                                                 ri::BaseRM_VecBaseRM, ro::BaseRM_VecBaseRM,
                                                 scai::Scalariser, scao::Scalariser,
                                                 fb::Option{<:OptE_Opt_FbChain})::HierarchicalEqualRiskContributionResult
    return HierarchicalEqualRiskContributionResult(hr, ri, ro, scai, scao, fb)
end
@forward_properties HierarchicalEqualRiskContributionResult begin
    forward(hr)
end
"""
    result_investable_mask(res::HierarchicalResult)
    result_investable_mask(res::HierarchicalRiskParityResult)
    result_investable_mask(res::HierarchicalEqualRiskContributionResult)

Read the Investable Mask a hierarchical result reduced on.

The core holds the mask in its field `imsk`, and each of the two leaves holds the core in its field `hr`. The generic method returns `nothing` for a result type that has no method of its own, and the property that a leaf forwards does not change the dispatch. So each leaf reads the mask through `hr`. Without these methods, a fold would keep the full weight vector and charge against it a per-asset fee that the result holds for the investable assets only.

# Arguments

  - `res`: A hierarchical result, or one of its two leaves.

# Returns

  - `imsk::Option{BitVector}`: The Investable Mask, or `nothing` when the optimisation reduced on nothing.

# Related

  - [`result_investable_mask`](@ref)
  - [`HierarchicalResult`](@ref)
  - [`HierarchicalRiskParityResult`](@ref)
  - [`HierarchicalEqualRiskContributionResult`](@ref)
"""
function result_investable_mask(res::HierarchicalResult)
    return res.imsk
end
function result_investable_mask(res::Union{<:HierarchicalRiskParityResult,
                                           <:HierarchicalEqualRiskContributionResult})
    return result_investable_mask(res.hr)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the prior, the clustering, the bounds and the fees that a hierarchical optimiser shares with its siblings.

[`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref) each hold one in their field `opt`. Each of them uses these fields to select the returns, fit the prior, cluster the assets, resolve the fees and the weight bounds, and finalise the weights.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalOptimiser(;
        pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
        cle::TD{<:HClE_HCl} = ClustersEstimator(),
        slv::Option{<:Slv_VecSlv} = nothing,
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        brt::Bool = false,
        x_src::Symbol = :prior,
        strict::Bool = false,
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> HierarchicalOptimiser

Keywords correspond to the struct's fields. A field typed [`TD`](@ref) or [`TD_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule, one value per fold, in place of a static value. A cross-validation fold loop resolves the schedule on each fold, and a fold-less `optimise` runs with the field at its static default. These fields are the problem definition: the prior estimator, the clustering estimator, the weight bounds, the fees, the asset sets and the weight finaliser. The fields `slv`, `brt`, `x_src` and `strict` control the execution, and they are always static.

## Validation

  - `x_src in (:prior, :data)`.
  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.
  - If a field holds a [`TimeDependent`](@ref) schedule: the constructor runs again with each value of the schedule in that field, so a value that the constructor refuses raises here and not in a later fold.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> HierarchicalOptimiser()
HierarchicalOptimiser
      pe ┼ EmpiricalPrior
         │           ce ┼ PortfolioOptimisersCovariance
         │              │   ce ┼ Covariance
         │              │      │    me ┼ SimpleExpectedReturns
         │              │      │       │   w ┴ nothing
         │              │      │    ce ┼ GeneralCovariance
         │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │              │      │       │    w ┴ nothing
         │              │      │   alg ┼ FullMoment()
         │              │      │     w ┴ nothing
         │              │   mp ┼ MatrixProcessing
         │              │      │     pdm ┼ Posdef
         │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │              │      │      dn ┼ nothing
         │              │      │      dt ┼ nothing
         │              │      │     alg ┼ nothing
         │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │           me ┼ SimpleExpectedReturns
         │              │   w ┴ nothing
         │      horizon ┼ nothing
         │   fill_limit ┴ nothing
     cle ┼ ClustersEstimator
         │    ce ┼ PortfolioOptimisersCovariance
         │       │   ce ┼ Covariance
         │       │      │    me ┼ SimpleExpectedReturns
         │       │      │       │   w ┴ nothing
         │       │      │    ce ┼ GeneralCovariance
         │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │       │      │       │    w ┴ nothing
         │       │      │   alg ┼ FullMoment()
         │       │      │     w ┴ nothing
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
    $(field_dict[:strict_opt])
    """
    strict
    """
    $(field_dict[:cache_opt])
    """
    @fprop @vprop cache
    function HierarchicalOptimiser(pe::Onl{<:TD{<:PrE_Pr}}, cle::TD{<:HClE_HCl},
                                   slv::Option{<:Slv_VecSlv}, wb::TD_Option{<:WbE_Wb},
                                   fees::TD_Option{<:FeesE_Fees},
                                   sets::TD_Option{<:UniverseSets},
                                   wf::TD{<:WeightFinaliser}, brt::Bool, x_src::Symbol,
                                   strict::Bool, cache::Option{<:ReturnsBufferState})
        assert_source_selector(x_src, :x_src)
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        assert_time_dependent_substitution(HierarchicalOptimiser,
                                           (; pe, cle, slv, wb, fees, sets, wf, brt, x_src,
                                            strict), hierarchical_optimiser_td_defaults())
        return new{typeof(pe), typeof(cle), typeof(slv), typeof(wb), typeof(fees),
                   typeof(sets), typeof(wf), typeof(brt), typeof(x_src), typeof(strict),
                   typeof(cache)}(pe, cle, slv, wb, fees, sets, wf, brt, x_src, strict,
                                  cache)
    end
end
function HierarchicalOptimiser(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
                               cle::TD{<:HClE_HCl} = ClustersEstimator(),
                               slv::Option{<:Slv_VecSlv} = nothing,
                               wb::TD_Option{<:WbE_Wb} = WeightBounds(),
                               fees::TD_Option{<:FeesE_Fees} = nothing,
                               sets::TD_Option{<:UniverseSets} = nothing,
                               wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                               brt::Bool = false, x_src::Symbol = :prior,
                               strict::Bool = false,
                               cache::Option{<:ReturnsBufferState} = nothing)::HierarchicalOptimiser
    return HierarchicalOptimiser(pe, cle, slv, wb, fees, sets, wf, brt, x_src, strict,
                                 cache)
end
function non_investable_universe(opt::HierarchicalOptimiser, ni::VecStr)
    return rebuild_estimator(opt, (; sets = non_investable_sets(opt.sets, ni)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the one copy of the static defaults of the schedulable fields of [`HierarchicalOptimiser`](@ref).

The constructor reads it to try each value of a schedule, and [`time_dependent_field_defaults`](@ref) returns it, so the two agree on the fold-less value of each field. It names `pe`, `cle`, `wb` and `wf`. The fields `fees` and `sets` default to `nothing`, so they have no entry.

# Returns

  - `defaults::NamedTuple`: The default value of each schedulable field whose default is not `nothing`.

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

Return `true` when a [`HierarchicalOptimiser`](@ref) reads the weights of the previous fold.

The optimiser reads them when its fees do, as a turnover fee does, or when one of its [`TimeDependent`](@ref) schedules does.

# Arguments

  - `opt`: The shared configuration of a hierarchical optimiser.

# Returns

  - `flag::Bool`: `true` when a fold loop must pass the previous weights through [`factory`](@ref).

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

Return the value that each schedulable field of a [`HierarchicalOptimiser`](@ref) takes when no fold resolves it.

It returns [`hierarchical_optimiser_td_defaults`](@ref), which the constructor also reads.

# Returns

  - `defaults::NamedTuple`: The static default of `pe`, `cle`, `wb` and `wf`.

# Related

  - [`HierarchicalOptimiser`](@ref)
  - [`hierarchical_optimiser_td_defaults`](@ref)
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

The result has one entry per asset, not per cluster. [`HierarchicalRiskParity`](@ref) and [`HierarchicalEqualRiskContribution`](@ref) weight the assets of a cluster in proportion to the inverse of these risks, which is the naive risk parity allocation inside the cluster.

# Mathematical definition

```math
\\begin{align}
\\rho(\\{i\\}) &= R(\\boldsymbol{e}_i)\\,, \\quad i = 1,\\, \\ldots,\\, N\\,.
\\end{align}
```

For [`Variance`](@ref), ``\\rho(\\{i\\}) = \\Sigma_{ii}``, so the vector is the diagonal of the covariance matrix. A measure that reads the returns evaluates ``R`` on the returns of asset ``i`` net of the fees that `fees` charges on ``\\boldsymbol{e}_i``.

Where:

  - ``\\rho(\\{i\\})``: Risk of asset ``i`` held alone.
  - $(math_dict[:R_w])
  - ``\\boldsymbol{e}_i``: Weight vector that is one at asset ``i`` and zero at every other asset.
  - ``\\Sigma_{ii}``: Variance of asset ``i``.
  - $(math_dict[:N])

# Algorithm

 1. Make the weight vector `wk`, all zero, of length `size(X, 2)`. Its entries take the type of the quotient of two returns, so an integer `X` gives fractional weights.
 2. For each asset `i`, set `wk[i]` to one, evaluate `expected_risk(r, wk, X, fees)` as the `i`-th risk, and set `wk[i]` back to zero.
 3. Collect the risks into `rk`, whose element type is the type that `expected_risk` returns.

# Arguments

  - `r`: Risk measure, already resolved by [`factory`](@ref).
  - `X`: Asset return matrix, observations by assets.
  - `fees`: Fees to charge against each unit portfolio, or `nothing`.

# Returns

  - `rk::Vector`: Expected risk of each asset held alone, of length `size(X, 2)`. Its element type is the type of the risk, not the element type of `X`.

# Related

  - [`unitary_expected_risks!`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`HierarchicalEqualRiskContribution`](@ref)
"""
function unitary_expected_risks(r::OptimisationRiskMeasure, X::MatNum,
                                fees::Option{<:Fees} = nothing)
    # A risk is not an entry of `X`: an integer returns matrix has a fractional variance.
    # The weights take the type of the returns, widened to a float only when it is an
    # integer, and `map` takes the type of the risks from the values `expected_risk` returns.
    wk = zeros(float_if_integer(eltype(X)), size(X, 2))
    return map(eachindex(wk)) do i
        wk[i] = one(eltype(wk))
        rki = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(wk))
        return rki
    end
end
"""
    unitary_expected_risks!(wk::VecNum, rk::VecNum, r::OptimisationRiskMeasure,
                            X::MatNum, fees::Option{<:Fees} = nothing) -> Nothing

Write the expected risk of each asset held alone into `rk`.

This is the in-place form of [`unitary_expected_risks`](@ref), which states the mathematics. A caller that evaluates several risk measures uses it to reuse one pair of buffers.

# Algorithm

 1. Set every entry of `rk` to zero.
 2. For each asset `i`, set `wk[i]` to one, evaluate `expected_risk(r, wk, X, fees)` into `rk[i]`, and set `wk[i]` back to zero.

# Arguments

  - `wk`: Scratch weight vector, of length `size(X, 2)`. It must be all zero on entry, and it is all zero on exit, because each step sets one entry to one and then back to zero.
  - `rk`: Output risk vector, of length `size(X, 2)`. It is overwritten in full. Its element type must hold a risk, so for an integer `X` it is a floating-point type.
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
    fill!(rk, zero(eltype(rk)))
    for i in eachindex(wk)
        wk[i] = one(eltype(wk))
        rk[i] = expected_risk(r, wk, X, fees)
        wk[i] = zero(eltype(wk))
    end
    return nothing
end

export HierarchicalResult, HierarchicalRiskParityResult,
       HierarchicalEqualRiskContributionResult, HierarchicalOptimiser
