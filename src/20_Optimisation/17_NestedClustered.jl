"""
$(DocStringExtensions.TYPEDEF)

Result type for Nested Clustered Optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`NestedClustered`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct NestedClusteredResult <: NonJuMPOptimisationResult
    """
    $(field_dict[:oe])
    """
    oe
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
    $(field_dict[:resi])
    """
    resi
    """
    $(field_dict[:reso])
    """
    reso
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    Final aggregated portfolio weights.
    """
    w
    """
    $(field_dict[:fb])
    """
    fb
    function NestedClusteredResult(oe::Type{<:OptimisationEstimator},
                                   pr::Option{<:AbstractPriorResult},
                                   clr::Option{<:AbstractClusteringResult},
                                   wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                                   resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                                   reso::OptimisationResult,
                                   cv::Option{<:OptimisationCrossValidation},
                                   retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                                   fb::Option{<:OptE_Opt})
        return new{typeof(oe), typeof(pr), typeof(clr), typeof(wb), typeof(fees),
                   typeof(resi), typeof(reso), typeof(cv), typeof(retcode), typeof(w),
                   typeof(fb)}(oe, pr, clr, wb, fees, resi, reso, cv, retcode, w, fb)
    end
end
function NestedClusteredResult(; oe::Type{<:OptimisationEstimator},
                               pr::Option{<:AbstractPriorResult},
                               clr::Option{<:AbstractClusteringResult},
                               wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                               resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                               reso::OptimisationResult,
                               cv::Option{<:OptimisationCrossValidation},
                               retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                               fb::Option{<:OptE_Opt})::NestedClusteredResult
    return NestedClusteredResult(oe, pr, clr, wb, fees, resi, reso, cv, retcode, w, fb)
end
"""
    assert_internal_optimiser(opt)

Assert that the inner (cluster-level) optimiser is valid for use in NCO.

Checks that the inner optimiser does not use pre-computed prior results or regression results, since these must be re-estimated for each cluster during NCO.

# Arguments

  - `opt`: Inner optimisation estimator.

# Returns

  - `nothing` on success; throws `ArgCheck` error otherwise.

# Related

  - [`NestedClustered`](@ref)
"""
function assert_internal_optimiser(opt::ClusteringOptimisationEstimator)::Nothing
    @argcheck(!isa(opt.opt.cle, AbstractClusteringResult),
              ArgumentError("opt.opt.cle cannot be a precomputed AbstractClusteringResult; use an estimator instead"))
    return nothing
end
"""
    assert_rc_variance(opt)

Assert that the optimiser does not use variance risk contribution for NCO outer optimisation.

Checks that risk budgeting-based JuMP optimisers do not use variance for risk contribution when used as the outer optimiser in NCO.

# Arguments

  - `opt`: Optimisation estimator.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
"""
function assert_rc_variance(::Any)::Nothing
    return nothing
end
function assert_rc_variance(opt::RiskJuMPOptimisationEstimator)::Nothing
    if isa(opt.r, Variance)
        @argcheck(!isa(opt.r.rc, LinearConstraint),
                  "`rc` cannot be a `LinearConstraint` because there is no way to only consider items from a specific group and because this would break factor risk contribution")
    elseif isa(opt.r, AbstractVector) && any(x -> isa(x, Variance), opt.r)
        idx = findall(x -> isa(x, Variance), opt.r)
        @argcheck(!any(x -> isa(x.rc, LinearConstraint), view(opt.r, idx)),
                  "`rc` cannot be a `LinearConstraint` because there is no way to only consider items from a specific group and because this would break factor risk contribution")
    end
    return nothing
end
"""
    assert_rc_pl(opt)

Assert that the optimiser does not use phylogeny risk contribution for NCO outer optimisation.

Checks that factor risk contribution optimisers do not use phylogeny-based constraints when used as the outer optimiser in NCO.

# Arguments

  - `opt`: Optimisation estimator.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
"""
function assert_rc_pl(::Any)::Nothing
    return nothing
end
function assert_rc_pl(opt::FactorRiskContribution)::Nothing
    @argcheck(!isa(opt.frc_ple, AbstractPhylogenyConstraintResult) ||
              isa(opt.frc_ple, AbstractVector) &&
              !any(x -> isa(x, AbstractPhylogenyConstraintResult), opt.frc_ple),
              ArgumentError("opt.frc_ple cannot be a precomputed AbstractPhylogenyConstraintResult in NCO outer optimiser; use an estimator instead"))
    return nothing
end
function assert_internal_optimiser(opt::JuMPOptimisationEstimator)::Nothing
    assert_rc_variance(opt)
    assert_rc_pl(opt)
    @argcheck(!(isa(opt.opt.lcse, LinearConstraint) ||
                isa(opt.opt.lcse, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.lcse)),
              ArgumentError("opt.opt.lcse cannot be a LinearConstraint in NCO inner optimiser"))
    @argcheck(!(isa(opt.opt.cte, LinearConstraint) ||
                isa(opt.opt.cte, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.cte)),
              ArgumentError("opt.opt.cte cannot be a LinearConstraint in NCO inner optimiser"))
    @argcheck(!isa(opt.opt.gcarde, LinearConstraint),
              ArgumentError("opt.opt.gcarde cannot be a LinearConstraint in NCO inner optimiser"))
    @argcheck(!(isa(opt.opt.sgcarde, LinearConstraint) ||
                isa(opt.opt.sgcarde, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.sgcarde)),
              ArgumentError("opt.opt.sgcarde cannot be a LinearConstraint in NCO inner optimiser"))
    @argcheck(!isa(opt.opt.ple, AbstractPhylogenyConstraintResult) ||
              isa(opt.opt.ple, AbstractVector) &&
              !any(x -> isa(x, AbstractPhylogenyConstraintResult), opt.opt.ple),
              ArgumentError("opt.opt.ple cannot be a precomputed AbstractPhylogenyConstraintResult in NCO inner optimiser; use an estimator instead"))
    return nothing
end
function assert_internal_optimiser(opt::VecOptE_Opt_TD)::Nothing
    assert_internal_optimiser.(opt)
    return nothing
end
"""
    assert_external_optimiser(opt)

Assert that the outer optimiser is valid for use in NCO.

Checks that the outer optimiser does not use pre-computed prior results, regression results, or unsupported variance/phylogeny risk contribution configurations.

# Arguments

  - `opt`: Outer optimisation estimator.

# Returns

  - `nothing` on success; throws `ArgCheck` error otherwise.

# Related

  - [`NestedClustered`](@ref)
"""
function assert_external_optimiser(opt::ClusteringOptimisationEstimator)::Nothing
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::JuMPOptimisationEstimator)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult),
              ArgumentError("opt.opt.pe cannot be a precomputed AbstractPriorResult; use an estimator instead"))
    assert_internal_optimiser(opt)
    return nothing
end
"""
    const RiskBudgetingOptimiser = Union{<:RiskBudgeting, <:RelaxedRiskBudgeting}

Alias for risk budgeting JuMP optimisers.

Matches either [`RiskBudgeting`](@ref) or [`RelaxedRiskBudgeting`](@ref). Used for dispatch in NCO validation and constraint generation.

# Related

  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
"""
const RiskBudgetingOptimiser = Union{<:RiskBudgeting, <:RelaxedRiskBudgeting}
function assert_external_optimiser(opt::RiskBudgetingOptimiser)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult),
              ArgumentError("opt.opt.pe cannot be a precomputed AbstractPriorResult; use an estimator instead"))
    if isa(opt.rba, FactorRiskBudgeting)
        @argcheck(!isa(opt.rba.re, AbstractRegressionResult),
                  ArgumentError("opt.rba.re cannot be a precomputed AbstractRegressionResult; use an estimator instead"))
    end
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::FactorRiskContribution)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult),
              ArgumentError("opt.opt.pe cannot be a precomputed AbstractPriorResult; use an estimator instead"))
    @argcheck(!isa(opt.re, AbstractRegressionResult),
              ArgumentError("opt.re cannot be a precomputed AbstractRegressionResult; use an estimator instead"))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::VecOptE_Opt_TD)::Nothing
    assert_external_optimiser.(opt)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Nested Clustered Optimisation (NCO) portfolio optimiser.

`NestedClustered` implements the Nested Clustered Optimisation algorithm. It first clusters assets, then solves a within-cluster (inner) optimisation for each cluster independently, and finally solves an across-cluster (outer) optimisation to combine the cluster portfolios into a final portfolio.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NestedClustered(;
        pe::PrE_Pr = EmpiricalPrior(),
        cle::ClE_Cl = ClustersEstimator(),
        wb::TD_Option{<:WbE_Wb} = WeightBounds(),
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::Option{<:AssetSets} = nothing,
        opti::OptE_TD,
        opto::OptE_TD,
        cv::Option{<:OptimisationCrossValidation} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        cle_pr::Bool = true,
        strict::Bool = false
    ) -> NestedClustered

Keywords correspond to the struct's fields.

## Time-dependent fields

`opti`, `opto` and `fb` may hold a [`TimeDependent`](@ref) per-fold schedule. `opto` and `fb` are `bind = :outermost` only — no inner fold loop consumes them. `opti` additionally admits `bind = :nearest`: the inner cross-validation is entered per cluster (`cross_val_predict(opti, …; cols = cl)`), so the field itself is the inner fold loop's entry point and a `:nearest` schedule is consumed there, per cluster. A `:nearest` `opti` schedule requires an explicit `default` and `cv !== nothing` at construction (see [`assert_nearest_optimiser_schedule`](@ref)) because the per-cluster optimise leg always resolves it fold-lessly to its `default`. `cv` itself stays static: it *is* the inner fold loop, not part of the per-fold problem definition, and the `:nearest` construction checks must be able to inspect it.

Schedule entries for `opti`/`opto` must be estimators, like the static fields: a vector schedule holding a precomputed result is rejected at construction by the entry substitution pass.

## Validation

  - `opto` must pass `assert_external_optimiser` and `assert_special_nco_requirements` (schedules delegate to their entries and `default`).
  - If `opti !== opto`: `opti` must pass `assert_internal_optimiser` and `assert_special_nco_requirements`.
  - If `cv` is provided: `opti` must also pass `assert_external_optimiser` and `assert_special_nco_requirements`.
  - `opto` and `fb` schedules: `bind !== :nearest`. A `bind = :nearest` `opti` schedule: explicit `default` and `cv !== nothing`.

# Mathematical definition

Let clusters ``C_1, \\ldots, C_K`` partition the ``N`` assets. The NCO algorithm:

 1. **Inner**: for each cluster ``k``, solve ``\\boldsymbol{w}_{C_k} = \\mathrm{opti}(\\mathbf{X}_{C_k})`` (sub-portfolio weights within ``C_k``).
 2. **Outer**: form a ``T \\times K`` synthetic returns matrix from cluster portfolios and solve ``\\boldsymbol{a} = \\mathrm{opto}(\\mathbf{X}_{\\mathrm{cluster}})`` (allocation across clusters).
 3. **Combine**: ``w_i = a_k \\cdot w_{C_k, i}`` for ``i \\in C_k``.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `opti`: Recursively updated via [`factory`](@ref).
  - `opto`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Related

  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`Stacking`](@ref)
  - [`NestedClusteredResult`](@ref)
"""
@propagatable @concrete struct NestedClustered <: ClusteringOptimisationEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:cle])
    """
    cle
    """
    $(field_dict[:wb_jmp])
    """
    wb
    """
    $(field_dict[:feese])
    """
    @fprop fees
    """
    $(field_dict[:sets])
    """
    sets
    """
    $(field_dict[:opti])
    """
    @fprop opti
    """
    $(field_dict[:opto])
    """
    @fprop opto
    """
    $(field_dict[:cv])
    """
    cv
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:fb])
    """
    @fprop fb
    """
    $(field_dict[:brt])
    """
    brt
    """
    $(field_dict[:cle_pr])
    """
    cle_pr
    """
    $(field_dict[:strict_opt])
    """
    strict
    function NestedClustered(pe::PrE_Pr, cle::ClE_Cl, wb::TD_Option{<:WbE_Wb},
                             fees::TD_Option{<:FeesE_Fees}, sets::Option{<:AssetSets},
                             opti::OptE_TD, opto::OptE_TD,
                             cv::Option{<:OptimisationCrossValidation}, wf::WeightFinaliser,
                             ex::FLoops.Transducers.Executor, fb::TDO_Option{<:OptE_Opt},
                             brt::Bool, cle_pr::Bool, strict::Bool)
        assert_nearest_optimiser_schedule(opti, :opti, cv, :NestedClustered)
        assert_no_nearest_bind_optimiser_schedule(opto, :opto, :NestedClustered)
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :NestedClustered)
        assert_external_optimiser(opto)
        assert_special_nco_requirements(opto)
        if !(opti === opto)
            assert_internal_optimiser(opti)
            assert_special_nco_requirements(opti)
        end
        if !isnothing(cv)
            assert_external_optimiser(opti)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        assert_time_dependent_substitution(NestedClustered,
                                           (; pe, cle, wb, fees, sets, opti, opto, cv, wf,
                                            ex, fb, brt, cle_pr, strict),
                                           nested_clustered_td_defaults())
        return new{typeof(pe), typeof(cle), typeof(wb), typeof(fees), typeof(sets),
                   typeof(opti), typeof(opto), typeof(cv), typeof(wf), typeof(ex),
                   typeof(fb), typeof(brt), typeof(cle_pr), typeof(strict)}(pe, cle, wb,
                                                                            fees, sets,
                                                                            opti, opto, cv,
                                                                            wf, ex, fb, brt,
                                                                            cle_pr, strict)
    end
end
function NestedClustered(; pe::PrE_Pr = EmpiricalPrior(), cle::ClE_Cl = ClustersEstimator(),
                         wb::TD_Option{<:WbE_Wb} = nothing,
                         fees::TD_Option{<:FeesE_Fees} = nothing,
                         sets::Option{<:AssetSets} = nothing, opti::OptE_TD, opto::OptE_TD,
                         cv::Option{<:OptimisationCrossValidation} = nothing,
                         wf::WeightFinaliser = IterativeWeightFinaliser(),
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         fb::TDO_Option{<:OptE_Opt} = nothing, brt::Bool = false,
                         cle_pr::Bool = true, strict::Bool = false)
    return NestedClustered(pe, cle, wb, fees, sets, opti, opto, cv, wf, ex, fb, brt, cle_pr,
                           strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`NestedClustered`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref). The optimiser-valued fields `opti` and `opto` are required and have no static default, so they are marked [`NoDefault`](@ref): a schedule there must carry its own `default` to be usable outside a fold loop. Fields whose static default is `nothing` (`wb`, `fees`, `fb`) are omitted.

# Related

  - [`NestedClustered`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function nested_clustered_td_defaults()::NamedTuple
    return (; opti = NoDefault(), opto = NoDefault())
end
function time_dependent_field_defaults(::NestedClustered)::NamedTuple
    return nested_clustered_td_defaults()
end
function inner_fold_fields(::NestedClustered)::Tuple
    return (:opti,)
end
function assert_internal_optimiser(opt::NestedClustered)::Nothing
    @argcheck(!isa(opt.cle, AbstractClusteringResult),
              ArgumentError("opt.cle cannot be a precomputed AbstractClusteringResult; use an estimator instead"))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    return nothing
end
function assert_external_optimiser(opt::NestedClustered)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult),
              ArgumentError("opt.pe cannot be a precomputed AbstractPriorResult; use an estimator instead"))
    @argcheck(!isa(opt.cle, AbstractClusteringResult),
              ArgumentError("opt.cle cannot be a precomputed AbstractClusteringResult; use an estimator instead"))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    if !isnothing(opt.cv)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (fees, inner optimiser, outer optimiser, or fallback).
"""
function needs_previous_weights(opt::NestedClustered)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opti) ||
            needs_previous_weights(opt.opto) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if the inner optimiser, outer optimiser, or fallback carries time-dependent constraints.
"""
function is_time_dependent(opt::NestedClustered)
    return (!isempty(time_dependent_fields(opt)) ||
            is_time_dependent(opt.opti) ||
            is_time_dependent(opt.opto) ||
            is_time_dependent(opt.fb))
end
function assert_time_dependent_fold_count(opt::NestedClustered, n::Integer,
                                          all_binds::Bool = true)::Nothing
    assert_time_dependent_fields_fold_count(opt, n, all_binds)
    assert_time_dependent_fold_count(opt.opti, n, false)
    assert_time_dependent_fold_count(opt.opto, n, all_binds)
    assert_time_dependent_fold_count(opt.fb, n, all_binds)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve time-dependent constraints for the fold described by `ctx` by recursing into the inner optimiser, outer optimiser, and fallback.
"""
function update_time_dependent_estimator(opt::NestedClustered, ctx::TimeDependentContext,
                                         all_binds::Bool = true)
    if !is_time_dependent(opt)
        return opt
    end
    opt = update_time_dependent_fields(opt, ctx, all_binds)
    return rebuild_estimator(opt,
                             (;
                              opti = update_time_dependent_estimator(opt.opti, ctx, false),
                              opto = update_time_dependent_estimator(opt.opto, ctx,
                                                                     all_binds),
                              fb = update_time_dependent_estimator(opt.fb, ctx, all_binds)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replace this meta-optimiser's own time-dependent fields with their static defaults.

Deliberately does **not** recurse into the wrapped optimisers: a standalone meta solve consumes inner per-fold schedules through its inner cross-validation leg, and its fold-less full-window inner solves reset themselves at their own `_optimise` seam. Only the meta's own fields (applied to the combined weights, resolved by an outer fold loop when one exists) are inert here. A `bind = :nearest` schedule in a field the meta hands across its own inner fold loop (see [`inner_fold_fields`](@ref)) is likewise left in place — resetting it here would replace it with its `default` before the inner cross-validation ever saw it.
"""
function reset_time_dependent_estimator(opt::NestedClustered)
    return reset_time_dependent_fields(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`NestedClustered`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(nco::NestedClustered, i, X::MatNum, args...)
    X = isa(nco.pe, AbstractPriorResult) ? nco.pe.X : X
    pe = port_opt_view(nco.pe, i)
    wb = port_opt_view(nco.wb, i)
    fees = port_opt_view(nco.fees, i)
    sets = port_opt_view(nco.sets, i)
    opti = port_opt_view(nco.opti, i, X)
    opto = port_opt_view(nco.opto, i, X)
    return NestedClustered(; pe = pe, cle = nco.cle, wb = wb, fees = fees, sets = sets,
                           opti = opti, opto = opto, cv = nco.cv, wf = nco.wf, ex = nco.ex,
                           fb = nco.fb, brt = nco.brt, cle_pr = nco.cle_pr,
                           strict = nco.strict)
end
"""
    predict_outer_nco_estimator_returns(
        nco::NestedClustered,
        rd::ReturnsResult,
        pr::AbstractPriorResult,
        fees::Option{<:Fees},
        wi::MatNum,
        resi::VecOpt,
        cls::VecVecInt
    )

Predict outer portfolio returns for [`NestedClustered`](@ref) optimisation. Overload this using `nco.cv` for custom cross-validation prediction.
"""
function predict_outer_nco_estimator_returns(nco::NestedClustered, rd::ReturnsResult,
                                             pr::AbstractPriorResult, fees::Option{<:Fees},
                                             wi::MatNum, resi::VecOpt, cls::VecVecInt)
    nb, B, iv, ivpa, X = prepare_outer_rd(rd, wi)
    for (i, (res, cl)) in enumerate(zip(resi, cls))
        pri = port_opt_view(pr, cl)
        feesi = port_opt_view(fees, cl)
        X[:, i] = calc_net_returns(res, pri, feesi)
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         nb = nb, B = B, ts = rd.ts, iv = iv, ivpa = ivpa)
end
function predict_outer_nco_estimator_returns(nco::NestedClustered{<:Any, <:Any, <:Any,
                                                                  <:Any, <:Any, <:Any,
                                                                  <:Any,
                                                                  <:OptimisationCrossValidation{<:NonCombOptCV}},
                                             rd::ReturnsResult, pr::AbstractPriorResult,
                                             fees::Option{<:Fees}, wi::MatNum, resi::VecOpt,
                                             cls::VecVecInt)
    (; opti, cv, ex) = nco
    cv = cv.cv
    predictions = Vector{MultiPeriodPredictionResult}(undef, length(cls))
    let cv = cv
        FLoops.@floop ex for (i, cl) in enumerate(cls)
            cvi = !hasfield(typeof(cv), :rng) ? cv : copy(cv)
            predictions[i] = cross_val_predict(opti, rd, cvi; cols = cl, ex = ex)
        end
    end
    return rebuild_returns_result(rd, predictions)
end
function predict_outer_nco_estimator_returns(nco::NestedClustered{<:Any, <:Any, <:Any,
                                                                  <:Any, <:Any, <:Any,
                                                                  <:Any,
                                                                  <:OptimisationCrossValidation{<:CombinatorialCrossValidation}},
                                             rd::ReturnsResult, pr::AbstractPriorResult,
                                             fees::Option{<:Fees}, wi::MatNum, resi::VecOpt,
                                             cls::VecVecInt)
    (; opti, cv, ex) = nco
    (; cv, scorer) = cv
    predictions = Vector{PopulationPredictionResult}(undef, length(cls))
    let cv = cv
        FLoops.@floop ex for (i, cl) in enumerate(cls)
            cvi = !hasfield(typeof(cv), :rng) ? cv : copy(cv)
            predictions[i] = cross_val_predict(opti, rd, cvi; cols = cl, ex = ex)
        end
    end
    if isnothing(scorer)
        scorer = NearestQuantilePrediction()
    end
    best_predictions = [scorer(prediction) for prediction in predictions]
    return rebuild_returns_result(rd, best_predictions)
end
function _optimise(nco::NestedClustered, rd::ReturnsResult; dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    nco = reset_time_dependent_estimator(nco)
    rd = returns_result_picker(rd, nco.brt)
    pr = prior(nco.pe, rd; dims = dims)
    X = pr.X
    clr = clusterise(nco.cle, pr; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     branchorder = branchorder, cle_pr = nco.cle_pr)
    fees = fees_constraints(nco.fees, nco.sets; datatype = eltype(X), strict = nco.strict)
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    wi = zeros(eltype(X), size(X, 2), clr.k)
    opti = nco.opti
    resi = Vector{NonFiniteAllocationOptimisationResult}(undef, clr.k)
    # FLoops.@floop nco.ex
    for (i, cl) in pairs(cls)
        optic = port_opt_view(opti, cl, X)
        rdc = port_opt_view(rd, cl)
        res = optimise(optic, rdc; dims = dims, branchorder = branchorder,
                       str_names = str_names, save = save, kwargs...)
        #! Support efficient frontier?
        @argcheck(!isa(res.retcode, AbstractVector),
                  ArgumentError("res.retcode cannot be an AbstractVector; efficient frontier results are not supported in NCO"))
        wi[cl, i] = res.w
        resi[i] = res
    end
    rdo = predict_outer_nco_estimator_returns(nco, rd, pr, fees, wi, resi, cls)
    reso = optimise(nco.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb = weight_bounds_constraints(nco.wb, nco.sets; N = clr.k, strict = nco.strict,
                                   datatype = eltype(X))
    retcode, w = outer_optimisation_finaliser(wb, nco.wf, resi, reso.retcode, reso.w, wi)
    return NestedClusteredResult(; oe = typeof(nco), pr = pr, clr = clr, wb = wb,
                                 fees = fees, resi = resi, reso = reso, cv = nco.cv,
                                 retcode = retcode, w = w, fb = nothing)
end
"""
    optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                      <:Any, <:Any, <:Any, Nothing
                  }, rd::ReturnsResult;
             dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
             save::Bool = true, kwargs...) -> NestedClusteredResult

Run the Nested Clustered Optimisation portfolio optimisation.

# Arguments

  - `nco`: The nested clustered optimiser to use.
  - $(arg_dict[:rd])
  - `dims`: The dimension along which observations advance in time.
  - `branchorder`: Passed to the inner and outer optimisers. If this optimiser uses hierarchical clustering, this applies to the clusterisation. The branch order to use for the clusterisation.
  - `str_names`: Passed to the inner and outer optimisers. Whether to use string names for the assets in the optimisation.
  - `save`: Passed to the inner and outer optimisers. Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`NestedClustered`](@ref)
  - [`NestedClusteredResult`](@ref)
"""
function optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(nco, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export NestedClusteredResult, NestedClustered
