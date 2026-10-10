"""
$(DocStringExtensions.TYPEDEF)

Holds the clustering, the intra-cluster results and the inter-cluster result of a Nested Clustered Optimisation.

[`optimise`](@ref) returns it for a [`NestedClustered`](@ref). `clr` holds the clustering that the algorithm found. `resi` holds one intra-cluster optimisation for each cluster, in cluster order. `reso` is the inter-cluster optimisation over the synthetic assets that the clusters define, so `reso.w` has one entry for each cluster. `w` is the product of the two weights, after the weight finaliser and the weight bounds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NestedClusteredResult(;
        pr::Option{<:AbstractPriorResult},
        clr::Option{<:AbstractClusteringResult},
        wb::Option{<:WeightBounds},
        fees::Option{<:Fees},
        resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
        reso::OptimisationResult,
        cv::Option{<:OptimisationCrossValidation},
        retcode::OptRetCode_VecOptRetCode,
        w::VecNum_VecVecNum,
        imsk::Option{<:BitVector} = nothing,
        fb::Option{<:OptE_Opt_FbChain}
    ) -> NestedClusteredResult

Keywords correspond to the struct's fields.

The keyword constructor expands `w` from the investable universe to the full asset universe with [`expand_investable_weights`](@ref), and [`optimise`](@ref) builds the result through it. The positional constructor does not expand. [`set_retcode`](@ref) rebuilds the result through the positional constructor, so the weights expand one time only. `pr`, `clr`, `wb`, `fees` and each entry of `resi` belong to the investable universe, because the algorithm runs on that universe.

# Related

  - [`NestedClustered`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`StackingResult`](@ref)
  - [`investable_reduction`](@ref)
  - [`expand_investable_weights`](@ref)

# References

  - $(ref_dict[:lopezdeprado2019robust])
  - $(ref_dict[:mlp1]) Chapter 7.
  - $(ref_dict[:cajas2025]) Section 12.3.
"""
@concrete struct NestedClusteredResult <: NonJuMPOptimisationResult
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
    Final portfolio weights over the full asset universe, zero at an asset outside the investable universe.
    """
    w
    """
    $(field_dict[:imsk])
    """
    imsk
    """
    $(field_dict[:fb_res])
    """
    fb
    function NestedClusteredResult(pr::Option{<:AbstractPriorResult},
                                   clr::Option{<:AbstractClusteringResult},
                                   wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                                   resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                                   reso::OptimisationResult,
                                   cv::Option{<:OptimisationCrossValidation},
                                   retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                                   imsk::Option{<:BitVector},
                                   fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(pr), typeof(clr), typeof(wb), typeof(fees), typeof(resi),
                   typeof(reso), typeof(cv), typeof(retcode), typeof(w), typeof(imsk),
                   typeof(fb)}(pr, clr, wb, fees, resi, reso, cv, retcode, w, imsk, fb)
    end
end
function NestedClusteredResult(; pr::Option{<:AbstractPriorResult},
                               clr::Option{<:AbstractClusteringResult},
                               wb::Option{<:WeightBounds}, fees::Option{<:Fees},
                               resi::AbstractVector{<:NonFiniteAllocationOptimisationResult},
                               reso::OptimisationResult,
                               cv::Option{<:OptimisationCrossValidation},
                               retcode::OptRetCode_VecOptRetCode, w::VecNum_VecVecNum,
                               imsk::Option{<:BitVector} = nothing,
                               fb::Option{<:OptE_Opt_FbChain})::NestedClusteredResult
    return NestedClusteredResult(pr, clr, wb, fees, resi, reso, cv, retcode,
                                 expand_investable_weights(imsk, w), imsk, fb)
end
# The nested-clustered family carries the mask on the result itself, so the fold reads it
# directly. The inner results are of the reduced universe, so the outer mask is the one the
# fold scores against.
function result_investable_mask(res::NestedClusteredResult)
    return res.imsk
end
"""
    set_retcode(res::NestedClusteredResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`NestedClusteredResult`](@ref) with a different return code.

A vector of return codes holds one code for each point of an efficient frontier, and a failure code in an entry drops that point. The method copies every other field unchanged. It rebuilds through the positional constructor, so `w` does not expand a second time.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one return code for each point of an efficient frontier.

# Returns

  - [`NestedClusteredResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`NestedClusteredResult`](@ref)
"""
function set_retcode(res::NestedClusteredResult, retcode::OptRetCode_VecOptRetCode)
    return NestedClusteredResult(res.pr, res.clr, res.wb, res.fees, res.resi, res.reso,
                                 res.cv, retcode, res.w, res.imsk, res.fb)
end
"""
    holds_precomputed(x, ::Type{T}) -> Bool

Report whether a slot of an estimator holds a value of type `T`.

The predicate reads the value in the slot and each entry of a vector in the slot. For a [`TimeDependent`](@ref) schedule in the slot, it reads each vector entry and the `default`, through [`time_dependent_entries`](@ref). A callable schedule has no value until it runs, so the predicate reads its `default` alone. The refusals of a nested optimiser pass a result type as `T`, so one predicate finds a precomputed result in each shape that a slot accepts.

# Arguments

  - `x`: The value of the slot.
  - `T`: The type to find.

# Returns

  - `true` when `x`, an entry of a vector in `x`, or an entry of a schedule in `x` is a `T`. `false` otherwise.

# Related

  - [`assert_internal_optimiser`](@ref)
  - [`assert_external_optimiser`](@ref)
  - [`assert_estimated_prior`](@ref)
"""
function holds_precomputed(x, ::Type{T})::Bool where {T}
    entries = isa(x, TimeDependent) ? time_dependent_entries(x) : (x,)
    return any(e -> isa(e, T) || isa(e, AbstractVector) && any(y -> isa(y, T), e), entries)
end
"""
    assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
    assert_internal_optimiser(opt::JuMPOptimisationEstimator)
    assert_internal_optimiser(opt::VecOptE_Opt_TD)
    assert_internal_optimiser(opt::NestedClustered)

Assert that an optimiser can solve the intra-cluster problem of a [`NestedClustered`](@ref).

Each cluster solves its problem over its own assets, so each input of the inner optimiser must follow the view onto those assets. An estimator fits again on the view. A precomputed result states one fixed universe, so the methods refuse it. A slot can hold such a result directly, in a vector, or in a [`TimeDependent`](@ref) schedule, and [`holds_precomputed`](@ref) reads each of these shapes. Other files add the methods of the other families of optimisers.

# Arguments

  - `opt`: The inner optimiser.

# Validation

  - A clustering optimiser: `opt.opt.cle` holds no `AbstractClusteringResult`.
  - A JuMP optimiser: `opt` passes [`assert_rc_variance`](@ref) and [`assert_rc_pl`](@ref). `opt.opt.lcse`, `opt.opt.cte`, `opt.opt.gcarde` and `opt.opt.sgcarde` hold no `LinearConstraint`, and `opt.opt.ple` holds no `AbstractPhylogenyConstraintResult`.
  - A vector of optimisers: each entry passes.
  - A [`NestedClustered`](@ref): `opt.cle` holds no `AbstractClusteringResult`, and `opt.opto` passes [`assert_external_optimiser`](@ref). When `opt.opti` is not `opt.opto`, it passes this function.

The methods throw an `ArgumentError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
  - [`assert_external_optimiser`](@ref)
  - [`holds_precomputed`](@ref)
"""
function assert_internal_optimiser(opt::ClusteringOptimisationEstimator)::Nothing
    @argcheck(!holds_precomputed(opt.opt.cle, AbstractClusteringResult),
              ArgumentError("opt.opt.cle cannot be a precomputed AbstractClusteringResult, or a TimeDependent schedule whose entries or default hold one; use an estimator instead"))
    return nothing
end
"""
    assert_rc_variance(opt)

Assert that no [`Variance`](@ref) of a JuMP optimiser holds a `LinearConstraint` in its `rc` field.

A precomputed risk contribution constraint states its rows over the full asset universe. A nested optimiser views that universe onto a cluster, or replaces it with the synthetic assets of the clusters, and no view of such a row keeps the assets of one group alone. The constraint also breaks a factor risk contribution, which reads the same rows. The method for any other value does nothing.

# Arguments

  - `opt`: An optimiser. The method for a risk-based JuMP optimiser reads `opt.r`, as one risk measure or as a vector of risk measures.

# Validation

  - No `Variance` in `opt.r` holds a `LinearConstraint` in `rc`. The method throws an `ArgumentError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
  - [`assert_internal_optimiser`](@ref)
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

Assert that a [`FactorRiskContribution`](@ref) holds no precomputed phylogeny constraint in `frc_ple`.

A precomputed phylogeny constraint states its matrix over one fixed universe, and a nested optimiser views that universe. The method for any other value does nothing.

# Arguments

  - `opt`: An optimiser.

# Validation

  - `opt.frc_ple` holds no `AbstractPhylogenyConstraintResult`, directly, in a vector or in a [`TimeDependent`](@ref) schedule. The method throws an `ArgumentError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
  - [`assert_internal_optimiser`](@ref)
  - [`holds_precomputed`](@ref)
"""
function assert_rc_pl(::Any)::Nothing
    return nothing
end
function assert_rc_pl(opt::FactorRiskContribution)::Nothing
    @argcheck(!holds_precomputed(opt.frc_ple, AbstractPhylogenyConstraintResult),
              ArgumentError("opt.frc_ple cannot be a precomputed AbstractPhylogenyConstraintResult, or a TimeDependent schedule whose entries or default hold one, in a nested optimiser; use an estimator instead"))
    return nothing
end
function assert_internal_optimiser(opt::JuMPOptimisationEstimator)::Nothing
    assert_rc_variance(opt)
    assert_rc_pl(opt)
    @argcheck(!holds_precomputed(opt.opt.lcse, LinearConstraint),
              ArgumentError("opt.opt.lcse cannot be a LinearConstraint, or a TimeDependent schedule whose entries or default hold one, in NCO inner optimiser"))
    @argcheck(!holds_precomputed(opt.opt.cte, LinearConstraint),
              ArgumentError("opt.opt.cte cannot be a LinearConstraint, or a TimeDependent schedule whose entries or default hold one, in NCO inner optimiser"))
    @argcheck(!holds_precomputed(opt.opt.gcarde, LinearConstraint),
              ArgumentError("opt.opt.gcarde cannot be a LinearConstraint, or a TimeDependent schedule whose entries or default hold one, in NCO inner optimiser"))
    @argcheck(!holds_precomputed(opt.opt.sgcarde, LinearConstraint),
              ArgumentError("opt.opt.sgcarde cannot be a LinearConstraint, or a TimeDependent schedule whose entries or default hold one, in NCO inner optimiser"))
    @argcheck(!holds_precomputed(opt.opt.ple, AbstractPhylogenyConstraintResult),
              ArgumentError("opt.opt.ple cannot be a precomputed AbstractPhylogenyConstraintResult, or a TimeDependent schedule whose entries or default hold one, in NCO inner optimiser; use an estimator instead"))
    return nothing
end
function assert_internal_optimiser(opt::VecOptE_Opt_TD)::Nothing
    assert_internal_optimiser.(opt)
    return nothing
end
"""
    assert_external_optimiser(opt::ClusteringOptimisationEstimator)
    assert_external_optimiser(opt::JuMPOptimisationEstimator)
    assert_external_optimiser(opt::RiskBudgetingOptimiser)
    assert_external_optimiser(opt::FactorRiskContribution)
    assert_external_optimiser(opt::VecOptE_Opt_TD)
    assert_external_optimiser(opt::NestedClustered)

Assert that an optimiser can solve the inter-cluster problem of a [`NestedClustered`](@ref), or fit again in each fold of its cross-validation.

The outer problem replaces the assets with one synthetic asset for each cluster. So the outer optimiser estimates its prior from the synthetic returns, and no input can be a result over the original assets. Each method also applies the refusals of [`assert_internal_optimiser`](@ref). [`NestedClustered`](@ref) applies this function to `opti` when `cv` is present, because each fold fits `opti` again, and a result fitted over every row puts the test rows of each fold into its fit.

# Arguments

  - `opt`: The outer optimiser, or the inner optimiser of a cross-validated [`NestedClustered`](@ref).

# Validation

  - Each method except the vector method: the `pe` slot passes [`assert_estimated_prior`](@ref). The slot is `opt.opt.pe`, or `opt.pe` for a `NestedClustered`.
  - A JuMP optimiser, a [`RiskBudgetingOptimiser`](@ref) and a [`FactorRiskContribution`](@ref): `opt` passes [`assert_external_lcse`](@ref).
  - A `RiskBudgetingOptimiser`: no [`FactorRiskBudgeting`](@ref) in `opt.rba`, or in an entry or the `default` of a schedule in `opt.rba`, holds an `AbstractLoadingsRegressionResult` in `re`.
  - A `FactorRiskContribution`: `opt.re` holds no `AbstractLoadingsRegressionResult`.
  - A `NestedClustered`: `opt.cle` holds no `AbstractClusteringResult`, and `opt.opto` passes this function. When `opt.opti` is not `opt.opto`, it passes [`assert_internal_optimiser`](@ref). When `opt.cv` is present, `opt.opti` passes this function.
  - A vector of optimisers: each entry passes.

The methods throw an `ArgumentError` otherwise.

# Returns

  - `nothing`.

# Related

  - [`NestedClustered`](@ref)
  - [`assert_internal_optimiser`](@ref)
  - [`holds_precomputed`](@ref)
"""
function assert_external_optimiser(opt::ClusteringOptimisationEstimator)::Nothing
    assert_estimated_prior(opt.opt.pe, "opt.opt.pe")
    assert_internal_optimiser(opt)
    return nothing
end
"""
    stated_constraint_space_basis(space::FactorSpace) -> Bool
    stated_constraint_space_basis(ece::ExposureConstraintEstimator) -> Bool
    stated_constraint_space_basis(lcse::AbstractVector) -> Bool
    stated_constraint_space_basis(td::TimeDependent) -> Bool
    stated_constraint_space_basis(::Any) -> Bool

Report whether an `lcse` slot holds a precomputed basis, so that an outer optimiser can refuse it.

A stated basis holds loadings indexed by the assets of one fixed universe. An inner solve takes a slice of the universe, and [`port_opt_view`](@ref) takes the same slice of the basis, so an inner optimiser can hold a stated basis. An outer solve replaces the universe with the names of the clusters, and no slice of the asset loadings follows that change. So an outer optimiser refuses a stated basis, as it refuses a precomputed `opt.re` and `opt.rba.re`.

The predicate is `false` for every other value, and for a space whose `re` is an estimator. An estimator fits again against the universe it receives, so the error message names it as the remedy. A vector and the entries and `default` of a [`TimeDependent`](@ref) schedule are read entry by entry.

# Related

  - [`assert_external_optimiser`](@ref)
  - [`FactorSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
function stated_constraint_space_basis(space::FactorSpace)::Bool
    return isa(space.re, AbstractLoadingsRegressionResult)
end
function stated_constraint_space_basis(::Any)::Bool
    return false
end
function stated_constraint_space_basis(ece::ExposureConstraintEstimator)::Bool
    return stated_constraint_space_basis(ece.space)
end
function stated_constraint_space_basis(lcse::AbstractVector)::Bool
    return any(stated_constraint_space_basis, lcse)
end
function stated_constraint_space_basis(td::TimeDependent)::Bool
    return any(stated_constraint_space_basis, time_dependent_entries(td))
end
"""
    assert_external_lcse(opt) -> Nothing

Assert that the `lcse` slot of an outer optimiser holds no precomputed basis.

The JuMP optimiser, the [`RiskBudgetingOptimiser`](@ref) and the [`FactorRiskContribution`](@ref) methods of [`assert_external_optimiser`](@ref) apply it. [`stated_constraint_space_basis`](@ref) states why an outer solve refuses a basis that an inner solve views.

# Validation

  - `stated_constraint_space_basis(opt.opt.lcse)` is `false`. The function throws an `ArgumentError` otherwise.

# Related

  - [`assert_external_optimiser`](@ref)
  - [`stated_constraint_space_basis`](@ref)
"""
function assert_external_lcse(opt)::Nothing
    @argcheck(!stated_constraint_space_basis(opt.opt.lcse),
              ArgumentError("a constraint space in opt.opt.lcse cannot hold a precomputed AbstractLoadingsRegressionResult in re; use an estimator instead. The outer problem replaces the asset universe with cluster names, so stated loadings cannot be sliced to follow it, and a row re-based through them would name assets that no longer exist"))
    return nothing
end
"""
    assert_estimated_prior(pe, name::AbstractString) -> Nothing

Assert that a `pe` slot holds a prior estimator, and no precomputed prior, directly or through a [`TimeDependent`](@ref) schedule.

A prior result is a fit over the rows that it saw. A fold loop that reads one gives every fold the same moments, so the test rows of each fold enter its fit. The outer solve of a nested optimiser cannot fit it again over the synthetic assets either. [`holds_precomputed`](@ref) reads the slot, and each vector entry and the `default` of a schedule. A callable schedule passes, because its values exist only when it runs.

# Arguments

  - `pe`: The value of the `pe` slot.
  - `name`: The path of the slot, as the error message names it.

# Validation

  - Neither `pe`, nor a vector entry of a schedule in `pe`, nor the `default` of that schedule, is an `AbstractPriorResult`. The function throws an `ArgumentError` that names `name` otherwise.

# Returns

  - `nothing`.

# Related

  - [`assert_external_optimiser`](@ref)
  - [`holds_precomputed`](@ref)
  - [`time_dependent_entries`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
function assert_estimated_prior(pe, name::AbstractString)::Nothing
    @argcheck(!holds_precomputed(pe, AbstractPriorResult),
              ArgumentError("$name cannot be a precomputed AbstractPriorResult, or a TimeDependent schedule whose entries or default hold one; use an estimator instead"))
    return nothing
end
function assert_external_optimiser(opt::JuMPOptimisationEstimator)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    assert_estimated_prior(opt.opt.pe, "opt.opt.pe")
    assert_external_lcse(opt)
    assert_internal_optimiser(opt)
    return nothing
end
"""
    const RiskBudgetingOptimiser = Union{<:RiskBudgeting, <:RelaxedRiskBudgeting}

Group the two JuMP optimisers that carry a risk budget algorithm in `rba`.

[`assert_external_optimiser`](@ref) dispatches on the group, because a [`FactorRiskBudgeting`](@ref) in `rba` holds a regression, and an outer optimiser must estimate that regression over the synthetic assets.

# Related

  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
"""
const RiskBudgetingOptimiser = Union{<:RiskBudgeting, <:RelaxedRiskBudgeting}
function assert_external_optimiser(opt::RiskBudgetingOptimiser)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    assert_estimated_prior(opt.opt.pe, "opt.opt.pe")
    rbas = isa(opt.rba, TimeDependent) ? time_dependent_entries(opt.rba) : (opt.rba,)
    @argcheck(!any(x -> isa(x, FactorRiskBudgeting) &&
                        isa(x.re, AbstractLoadingsRegressionResult), rbas),
              ArgumentError("opt.rba.re cannot be a precomputed AbstractLoadingsRegressionResult, or a TimeDependent schedule whose entries or default hold one; use an estimator instead"))
    assert_external_lcse(opt)
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::FactorRiskContribution)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    assert_estimated_prior(opt.opt.pe, "opt.opt.pe")
    @argcheck(!holds_precomputed(opt.re, AbstractLoadingsRegressionResult),
              ArgumentError("opt.re cannot be a precomputed AbstractLoadingsRegressionResult, or a TimeDependent schedule whose entries or default hold one; use an estimator instead"))
    assert_external_lcse(opt)
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::VecOptE_Opt_TD)::Nothing
    assert_external_optimiser.(opt)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Clusters the assets, optimises each cluster, and then optimises across the cluster portfolios.

Nested Clustered Optimisation (NCO) of [lopezdeprado2019robust](@cite) solves one intra-cluster problem for each cluster with the inner optimiser. It then solves one inter-cluster problem with the outer optimiser, over the synthetic assets that the cluster portfolios define. [mlp1](@cite) Chapter 7 states the inter-cluster problem through the reduced moments of the clusters. Section 12.3 of [cajas2025](@cite) lets each of the two problems take its own objective and its own constraints, and so does this type.

# Mathematical definition

```math
\\begin{align}
W_{ik} &= \\begin{cases} w^{(k)}_{i} & i \\in C_k\\,,\\\\ 0 & i \\notin C_k\\,,\\end{cases}\\\\
\\mathbf{X}^{o} &= \\mathbf{X} \\mathbf{W}\\,,\\\\
\\boldsymbol{w} &= \\mathbf{W} \\boldsymbol{v}\\,.
\\end{align}
```

Where:

  - ``C_1, \\ldots, C_K``: Clusters of the assets, a partition of the ``N`` assets.
  - ``w^{(k)}_{i}``: Weight of asset ``i`` in the solution of the inner optimiser over the assets of ``C_k``.
  - $(math_dict[:W_inner])
  - $(math_dict[:X_returns])
  - ``\\mathbf{X}^{o}``: Returns matrix of the synthetic assets, ``T \\times K``. Column ``k`` holds the returns of the portfolio of cluster ``C_k``.
  - $(math_dict[:v_outer])
  - ``\\boldsymbol{w}``: Combined weights over the ``N`` assets, before the weight finaliser and the weight bounds.
  - $(math_dict[:K_sub])
  - $(math_dict[:N])
  - $(math_dict[:T])

The outer optimiser solves for ``\\boldsymbol{v}`` over ``\\mathbf{X}^{o}``. The weight of asset ``i`` in ``C_k`` is ``w_i = v_k w^{(k)}_{i}``. The sample mean of ``\\mathbf{X}^{o}`` is ``\\mathbf{W}^\\intercal \\hat{\\boldsymbol{\\mu}}`` and its sample covariance is ``\\mathbf{W}^\\intercal \\hat{\\mathbf{\\Sigma}} \\mathbf{W}``, where ``\\hat{\\boldsymbol{\\mu}}`` and ``\\hat{\\mathbf{\\Sigma}}`` are the sample moments of ``\\mathbf{X}``. These are the reduced moments of [mlp1](@cite) Chapter 7, so an outer optimiser that estimates sample moments solves the inter-cluster problem of that chapter.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NestedClustered(;
        pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
        cle::TD{<:ClE_Cl} = ClustersEstimator(),
        wb::TD_Option{<:WbE_Wb} = nothing,
        fees::TD_Option{<:FeesE_Fees} = nothing,
        sets::TD_Option{<:UniverseSets} = nothing,
        opti::OptE_TD,
        opto::OptE_TD,
        cv::Option{<:OptimisationCrossValidation} = nothing,
        wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::TDO_Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        x_src::Symbol = :prior,
        strict::Bool = false,
        pcol::AbstractPanelCollapseAlgorithm = RenormaliseActive(),
        cache::Option{<:ReturnsBufferState} = nothing
    ) -> NestedClustered

Keywords correspond to the struct's fields.

## Time-dependent fields

`pe`, `cle`, `wb`, `fees`, `sets`, `wf`, `opti`, `opto` and `fb` can hold a [`TimeDependent`](@ref) schedule, one value for each fold. A schedule in `opto` or `fb` must bind `:outermost`, because no inner fold loop reads them. A schedule in `opti` can also bind `:nearest`. The inner cross-validation runs `cross_val_predict(opti, …; cols = cl)` for each cluster, so the inner fold loop reads `opti`. A `:nearest` schedule in `opti` resolves in that loop, one time for each cluster. The solve of each cluster outside the folds resolves the schedule to its `default`, so a `:nearest` schedule in `opti` needs an explicit `default` and a `cv` (see [`assert_nearest_optimiser_schedule`](@ref)). `cv` accepts no schedule, because it is the inner fold loop itself, and the construction checks read it.

An entry of a schedule in `opti` or `opto` must be an estimator, as the static value must be. The constructor refuses a vector schedule that holds a precomputed result.

## Validation

  - `x_src` is `:prior` or `:data`.
  - `opto` passes [`assert_external_optimiser`](@ref) and [`assert_special_nco_requirements`](@ref). A schedule passes when its entries and its `default` pass.
  - When `opti` is not `opto`, `opti` passes [`assert_internal_optimiser`](@ref) and `assert_special_nco_requirements`.
  - When `cv` is present, `opti` passes `assert_external_optimiser`.
  - A schedule in `opto` or `fb` does not bind `:nearest`. A schedule in `opti` that binds `:nearest` has an explicit `default`, and `cv` is present.
  - A `WeightBoundsEstimator` in `wb`, or a `FeesEstimator` in `fees`, needs `sets`. The constructor throws an `IsNothingError` otherwise.
  - A vector schedule holds no precomputed result, see [`assert_time_dependent_substitution`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `fees`: Recursively updated via [`factory`](@ref).
  - `opti`: Recursively updated via [`factory`](@ref).
  - `opto`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).
  - `cache`: Recursively updated via [`factory`](@ref).

## View parameters

`NestedClustered` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X`. When `pe` holds a prior result, the method uses `pe.X` in place of `X`, so the fields that read a matrix read the observations of the prior.
  - `pe`, `wb`, `sets` and `cache` recurse through [`port_opt_view`](@ref) with the index alone.
  - `fees`, `opti` and `opto` recurse through [`port_opt_view`](@ref) with the index and the matrix, and `fb` through [`view_child`](@ref).
  - `cle` passes through unchanged, because the clustering runs again over the viewed assets.

# Related

  - [`optimise`](@ref)
  - [`NestedClusteredResult`](@ref)
  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`Stacking`](@ref)
  - [`predict_outer_returns`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:lopezdeprado2019robust])
  - $(ref_dict[:mlp1]) Chapter 7.
  - $(ref_dict[:cajas2025]) Section 12.3.
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
    $(field_dict[:x_src])
    """
    x_src
    """
    $(field_dict[:strict_opt])
    """
    strict
    """
    $(field_dict[:pcol])
    """
    pcol
    """
    $(field_dict[:cache_opt])
    """
    @fprop cache
    function NestedClustered(pe::Onl{<:TD{<:PrE_Pr}}, cle::TD{<:ClE_Cl},
                             wb::TD_Option{<:WbE_Wb}, fees::TD_Option{<:FeesE_Fees},
                             sets::TD_Option{<:UniverseSets}, opti::OptE_TD, opto::OptE_TD,
                             cv::Option{<:OptimisationCrossValidation},
                             wf::TD{<:WeightFinaliser}, ex::FLoops.Transducers.Executor,
                             fb::TDO_Option{<:OptE_Opt}, brt::Bool, x_src::Symbol,
                             strict::Bool, pcol::AbstractPanelCollapseAlgorithm,
                             cache::Option{<:ReturnsBufferState})
        assert_source_selector(x_src, :x_src)
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
                                            ex, fb, brt, x_src, strict, pcol),
                                           nested_clustered_td_defaults())
        return new{typeof(pe), typeof(cle), typeof(wb), typeof(fees), typeof(sets),
                   typeof(opti), typeof(opto), typeof(cv), typeof(wf), typeof(ex),
                   typeof(fb), typeof(brt), typeof(x_src), typeof(strict), typeof(pcol),
                   typeof(cache)}(pe, cle, wb, fees, sets, opti, opto, cv, wf, ex, fb, brt,
                                  x_src, strict, pcol, cache)
    end
end
function NestedClustered(; pe::Onl{<:TD{<:PrE_Pr}} = EmpiricalPrior(),
                         cle::TD{<:ClE_Cl} = ClustersEstimator(),
                         wb::TD_Option{<:WbE_Wb} = nothing,
                         fees::TD_Option{<:FeesE_Fees} = nothing,
                         sets::TD_Option{<:UniverseSets} = nothing, opti::OptE_TD,
                         opto::OptE_TD, cv::Option{<:OptimisationCrossValidation} = nothing,
                         wf::TD{<:WeightFinaliser} = IterativeWeightFinaliser(),
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         fb::TDO_Option{<:OptE_Opt} = nothing, brt::Bool = false,
                         x_src::Symbol = :prior, strict::Bool = false,
                         pcol::AbstractPanelCollapseAlgorithm = RenormaliseActive(),
                         cache::Option{<:ReturnsBufferState} = nothing)
    return NestedClustered(pe, cle, wb, fees, sets, opti, opto, cv, wf, ex, fb, brt, x_src,
                           strict, pcol, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`NestedClustered`](@ref) fields that can hold a [`TimeDependent`](@ref) schedule.

The substitution pass of the constructor and [`time_dependent_field_defaults`](@ref) both read it. `opti` and `opto` have no keyword default, so they map to [`NoDefault`](@ref), and a schedule in them needs its own `default` to run outside a fold loop. `pe`, `cle` and `wf` map to their keyword defaults. The tuple omits `wb`, `fees`, `sets` and `fb`, whose keyword default is `nothing`.

# Related

  - [`NestedClustered`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function nested_clustered_td_defaults()::NamedTuple
    return (; pe = EmpiricalPrior(), cle = ClustersEstimator(), opti = NoDefault(),
            opto = NoDefault(), wf = IterativeWeightFinaliser())
end
function time_dependent_field_defaults(::NestedClustered)::NamedTuple
    return nested_clustered_td_defaults()
end
function inner_fold_fields(::NestedClustered)::Tuple
    return (:opti,)
end
function assert_internal_optimiser(opt::NestedClustered)::Nothing
    @argcheck(!holds_precomputed(opt.cle, AbstractClusteringResult),
              ArgumentError("opt.cle cannot be a precomputed AbstractClusteringResult, or a TimeDependent schedule whose entries or default hold one; use an estimator instead"))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    return nothing
end
function assert_external_optimiser(opt::NestedClustered)::Nothing
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    assert_estimated_prior(opt.pe, "opt.pe")
    @argcheck(!holds_precomputed(opt.cle, AbstractClusteringResult),
              ArgumentError("opt.cle cannot be a precomputed AbstractClusteringResult, or a TimeDependent schedule whose entries or default hold one; use an estimator instead"))
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

Return `true` when a part of `opt` needs the previous portfolio weights.

The parts are the schedules in the fields of `opt`, `fees`, `opti`, `opto` and `fb`.
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

Return `true` when a field of `opt` holds a [`TimeDependent`](@ref) schedule, or when `opti`, `opto` or `fb` holds one at any depth.
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

Resolve the [`TimeDependent`](@ref) schedules of `opt` for the fold that `ctx` describes.

# Algorithm

 1. When `opt` holds no schedule, return `opt` unchanged.
 2. Resolve the schedules in the fields of `opt` with [`update_time_dependent_fields`](@ref), giving `opt`. A schedule in `opti` that binds `:nearest` stays, because the inner fold loop reads `opti`.
 3. Resolve the schedules inside `opti` with `all_binds = false`, so that a schedule that binds `:nearest` stays for the inner fold loop.
 4. Resolve the schedules inside `opto` and `fb` with `all_binds`.
 5. Rebuild `opt` with the three resolved values through [`rebuild_estimator`](@ref), which runs the checks of the constructor again.
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

Replace the [`TimeDependent`](@ref) schedules in the fields of `opt` with their static defaults.

The method does not recurse into `opti`, `opto` or `fb`. A solve of `opt` outside a fold loop gives the schedules inside `opti` to its inner cross-validation, and each solve of a wrapped optimiser outside the folds resets its own schedules. A schedule that binds `:nearest` in a field that the inner fold loop reads stays too, see [`inner_fold_fields`](@ref). A reset of that schedule puts its `default` in place before the inner cross-validation reads it.
"""
function reset_time_dependent_estimator(opt::NestedClustered)
    return reset_time_dependent_fields(opt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of a [`NestedClustered`](@ref) viewed onto the assets `i`, with the returns matrix `X`.

The `## View parameters` subsection of [`NestedClustered`](@ref) states how the method views each field.
"""
function port_opt_view(nco::NestedClustered, i, X::MatNum, args...)
    X = isa(nco.pe, AbstractPriorResult) ? nco.pe.X : X
    pe = port_opt_view(nco.pe, i)
    wb = port_opt_view(nco.wb, i)
    fees = port_opt_view(nco.fees, i, X)
    sets = port_opt_view(nco.sets, i)
    opti = port_opt_view(nco.opti, i, X)
    opto = port_opt_view(nco.opto, i, X)
    return NestedClustered(; pe = pe, cle = nco.cle, wb = wb, fees = fees, sets = sets,
                           opti = opti, opto = opto, cv = nco.cv, wf = nco.wf, ex = nco.ex,
                           fb = view_child(nco.fb, i, X), brt = nco.brt, x_src = nco.x_src,
                           strict = nco.strict, pcol = nco.pcol,
                           cache = port_opt_view(nco.cache, i))
end
function non_investable_universe(nco::NestedClustered, ni::VecStr)::NestedClustered
    return rebuild_estimator(nco, (; sets = non_investable_sets(nco.sets, ni)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild the asset sets of the outer optimiser over the names of the synthetic assets.

The outer optimiser of a [`NestedClustered`](@ref) sees one synthetic asset for each cluster, and the outer returns result `rdo` carries their names, `_1`, `_2`, …. The method writes these names into the `xkey` entry of the [`UniverseSets`](@ref) of the outer optimiser. A constraint or a bound of the outer optimiser then names a cluster by the name of its synthetic asset, for example `_1 >= 0.3`.

# Algorithm

 1. When `nco.opto` has an `opt` field whose `sets` is present, and the `xkey` entry of those sets is not the vector `rdo.nx` itself, copy the dictionary of the sets to `ndict` and set `ndict[xkey] = rdo.nx`. Return `nco` with `ndict` in `nco.opto.opt.sets`.
 2. Otherwise, when `nco.opto` has a `sets` field that meets the same condition, do step 1 on `nco.opto.sets`.
 3. Otherwise, return `nco` unchanged.

The method copies the dictionary, so the [`UniverseSets`](@ref) of the caller does not change.

# Arguments

  - `nco::NestedClustered`: The nested clustered optimiser.
  - `rdo::ReturnsResult`: Outer returns result, whose `nx` holds the names of the synthetic assets.

# Returns

  - `nco::NestedClustered`: The optimiser, with the asset sets of `nco.opto` rebuilt over `rdo.nx`, or `nco` unchanged by step 3.

# Related

  - [`NestedClustered`](@ref)
  - [`UniverseSets`](@ref)
  - [`predict_outer_returns`](@ref)
"""
function _update_asset_sets(nco::NestedClustered, rdo::ReturnsResult)
    return if (hasproperty(nco.opto, :opt) &&
               hasproperty(nco.opto.opt, :sets) &&
               !isnothing(nco.opto.opt.sets) &&
               get(nco.opto.opt.sets.dict, nco.opto.opt.sets.xkey, nothing) !== rdo.nx)
        ndict = copy(nco.opto.opt.sets.dict)
        ndict[nco.opto.opt.sets.xkey] = rdo.nx
        Accessors.@reset nco.opto.opt.sets.dict = ndict
    elseif (hasproperty(nco.opto, :sets) &&
            !isnothing(nco.opto.sets) &&
            get(nco.opto.sets.dict, nco.opto.sets.xkey, nothing) !== rdo.nx)
        ndict = copy(nco.opto.sets.dict)
        ndict[nco.opto.sets.xkey] = rdo.nx
        Accessors.@reset nco.opto.sets.dict = ndict
    else
        nco
    end
end
function _optimise(nco::NestedClustered, rd::ReturnsResult; branchorder::Symbol = :optimal,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    nco = reset_time_dependent_estimator(nco)
    rd = returns_result_picker(rd, nco.brt)
    pr = prior(nco.pe, rd)
    # A weight and a fee are fractions, so integer returns take a floating point type.
    Tf = float_if_integer(eltype(pr.X))
    # Resolve the fee on the caller's universe before `investable_reduction` narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset. A liquidation charge keyed by name cannot resolve at all once its `w` sits on
    # the complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    # An asset that a `FeatureDistance` of the clustering cannot read in the window of its
    # Asset Panel departs with the non-investable assets (`feature_readable_mask`).
    imsk = feature_readable_mask(nco.cle, investable_mask(pr), rd)
    fees = investable_fees_view(fees_constraints(nco.fees, nco.sets; datatype = Tf,
                                                 strict = nco.strict), imsk, pr.X)
    # A forced exit is charged once, against the full-universe weight vector the fit
    # rebuilds, so only the result charges it. No sub-problem below holds that vector —
    # the exiting asset is in no cluster, its column being `NaN` — so none prices an exit.
    cfees = strip_liquidation_charges(fees, nothing)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here, so
    # that no cluster holds a non-investable asset and every cluster slice below indexes
    # the reduced axis. The weights are expanded back in `NestedClusteredResult`.
    # The reduced `rd` takes a name of its own. `rd` is assigned twice above, and a
    # variable that is reassigned and then captured by the fold's closure is boxed, which
    # FLoops reports as a correctness and performance problem on every call.
    _, pr, nco, rdr = investable_reduction(imsk, pr, nco, rd)
    X = pr.X
    clr = clusterise(nco.cle, pr; rd = rdr, iv = rdr.iv, ivpa = rdr.ivpa,
                     branchorder = branchorder, x_src = nco.x_src)
    assert_clustering_universe(clr, size(X, 2))
    idx = assignments(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    wi = zeros(Tf, size(X, 2), clr.k)
    opti = nco.opti
    resi = Vector{NonFiniteAllocationOptimisationResult}(undef, clr.k)
    FLoops.@floop nco.ex for (i, cl) in pairs(cls)
        optic = port_opt_view(opti, cl, X)
        rdc = port_opt_view(rdr, cl)
        res = optimise(optic, rdc; branchorder = branchorder, str_names = str_names,
                       save = save, kwargs...)
        #! Support efficient frontier?
        @argcheck(!isa(res.retcode, AbstractVector),
                  ArgumentError("res.retcode cannot be an AbstractVector; efficient frontier results are not supported in NCO"))
        wi[cl, i] = res.w
        resi[i] = res
    end
    rdo = predict_outer_returns(nco.cv, nco, ClusterUniverse(cls), rdr, pr, cfees, wi, resi)
    nco = _update_asset_sets(nco, rdo)
    reso = optimise(nco.opto, rdo; branchorder = branchorder, str_names = str_names,
                    save = save, kwargs...)
    wb = weight_bounds_constraints(nco.wb, nco.sets; N = size(X, 2), strict = nco.strict,
                                   datatype = Tf)
    retcode, w = outer_optimisation_finaliser(wb, nco.wf, resi, reso.retcode, reso.w, wi)
    return NestedClusteredResult(; pr = pr, clr = clr, wb = wb, fees = fees, resi = resi,
                                 reso = reso, cv = nco.cv, retcode = retcode, w = w,
                                 imsk = imsk, fb = nothing)
end
"""
    optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                      <:Any, <:Any, <:Any, Nothing
                  }, rd::ReturnsResult;
             branchorder::Symbol = :optimal, str_names::Bool = false,
             save::Bool = true, kwargs...) -> NestedClusteredResult

Optimise a portfolio with Nested Clustered Optimisation.

The `# Mathematical definition` of [`NestedClustered`](@ref) states the weights that the method computes.

# Algorithm

 1. Replace the schedules in the fields of `nco` with their static defaults, through [`reset_time_dependent_estimator`](@ref).
 2. Select the returns of `rd` with [`returns_result_picker`](@ref) and `nco.brt`, giving `rd`.
 3. Fit the prior of `nco.pe` to `rd`, giving `pr`.
 4. Take the element type of the weights, the fees and the weight bounds from `pr.X` with [`float_if_integer`](@ref), giving `Tf`. Integer returns take a floating point type, because a weight and a fee are fractions.
 5. Read the investable mask of `pr` with [`investable_mask`](@ref), giving `imsk`.
 6. Resolve the fees of `nco.fees` over the universe of the caller with [`fees_constraints`](@ref), and place them on the investable assets with [`investable_fees_view`](@ref), giving `fees`.
 7. Remove the liquidation charges from `fees` with [`strip_liquidation_charges`](@ref), giving `cfees`. The result charges a forced exit one time, and no cluster holds an asset that exits.
 8. Reduce `pr`, `nco` and `rd` to the investable assets with [`investable_reduction`](@ref), giving `pr`, `nco` and `rdr`.
 9. Cluster the assets with [`clusterise`](@ref), giving `clr`, and check with [`assert_clustering_universe`](@ref) that `clr` covers every asset. Collect the assets of each cluster, giving `cls`.
10. For each cluster `k`, view `nco.opti` and `rdr` onto the assets `cls[k]` with [`port_opt_view`](@ref), and optimise, giving `resi[k]`. Write its weights into column `k` of `wi`, which is zero outside the cluster.
11. Predict the returns of the synthetic assets with [`predict_outer_returns`](@ref), `cfees` and a [`ClusterUniverse`](@ref), giving `rdo`. Without `nco.cv` these are the returns of the cluster portfolios over the observations of `pr`. With `nco.cv` they are the out-of-sample predictions of the folds.
12. Rebuild the asset sets of `nco.opto` over the names of the synthetic assets with [`_update_asset_sets`](@ref).
13. Optimise `nco.opto` over `rdo`, giving `reso`.
14. Resolve the weight bounds of `nco.wb` over the investable assets with [`weight_bounds_constraints`](@ref), giving `wb`.
15. Combine `wi` and `reso.w` and finalise them with `nco.wf` under `wb`, through [`outer_optimisation_finaliser`](@ref), giving `retcode` and `w`.
16. Build the [`NestedClusteredResult`](@ref). Its keyword constructor expands `w` to the full asset universe.

# Arguments

  - `nco`: The nested clustered optimiser.
  - $(arg_dict[:rd])
  - `branchorder`: The branch order of a hierarchical clustering. The method also passes it to the inner and outer optimisers.
  - `str_names`: Whether the JuMP models name their variables with strings. The method passes it to the inner and outer optimisers.
  - `save`: Whether the results save their JuMP models. The method passes it to the inner and outer optimisers.
  - `kwargs`: More keyword arguments for the inner and outer optimisers.

# Validation

  - No field in the tree of `nco` holds an [`Online`](@ref). The method throws an `ArgumentError` that names the field otherwise, through [`assert_batch_entry`](@ref). A plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the online arm of a fold loop.
  - The clustering covers every investable asset. The method throws a `DimensionMismatch` otherwise.
  - No intra-cluster optimisation returns an efficient frontier. The method throws an `ArgumentError` otherwise.

# Returns

  - `res::NestedClusteredResult`: The combined portfolio. `retcode` is an [`OptimisationFailure`](@ref) when an intra-cluster optimisation, the inter-cluster optimisation or the weight finalisation fails.

# Related

  - [`NestedClustered`](@ref)
  - [`NestedClusteredResult`](@ref)
"""
function optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    assert_batch_entry(nco, "`optimise`")
    return _optimise(nco, rd; branchorder = branchorder, str_names = str_names, save = save,
                     kwargs...)
end

export NestedClusteredResult, NestedClustered
