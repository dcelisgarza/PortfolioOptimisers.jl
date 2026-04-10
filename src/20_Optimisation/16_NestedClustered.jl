"""
$(DocStringExtensions.TYPEDEF)

Result type for Nested Clustered Optimisation.

# Fields

  - `oe`: Type of the optimisation estimator that produced this result.
  - `pr`: Prior result used in optimisation.
  - `clr`: Clustering result.
  - `wb`: Weight bounds applied.
  - `fees`: Fee structure applied (or `nothing`).
  - `resi`: Inner (intra-cluster) optimisation results.
  - `reso`: Outer (inter-cluster) optimisation result.
  - `cv`: Cross-validation result (or `nothing`).
  - `retcode`: Overall optimisation return code.
  - `w`: Final aggregated portfolio weights.
  - `fb`: Fallback result.

# Related

  - [`NestedClustered`](@ref)
  - [`NonFiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct NestedClusteredResult <: NonFiniteAllocationOptimisationResult
    oe
    pr
    clr
    wb
    fees
    resi
    reso
    cv
    retcode
    w
    fb
end
function factory(res::NestedClusteredResult, fb::Option{<:OptE_Opt})
    return NestedClusteredResult(res.oe, res.pr, res.clr, res.wb, res.fees, res.resi,
                                 res.reso, res.cv, res.retcode, res.w, fb)
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
function assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
    @argcheck(!isa(opt.opt.cle, AbstractClusteringResult))
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
function assert_rc_variance(::Any)
    return nothing
end
function assert_rc_variance(opt::RiskJuMPOptimisationEstimator)
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
function assert_rc_pl(::Any)
    return nothing
end
function assert_rc_pl(opt::FactorRiskContribution)
    @argcheck(!isa(opt.frc_ple, AbstractPhylogenyConstraintResult) ||
              isa(opt.frc_ple, AbstractVector) &&
              !any(x -> isa(x, AbstractPhylogenyConstraintResult), opt.frc_ple))
    return nothing
end
function assert_internal_optimiser(opt::JuMPOptimisationEstimator)
    assert_rc_variance(opt)
    assert_rc_pl(opt)
    @argcheck(!(isa(opt.opt.lcse, LinearConstraint) ||
                isa(opt.opt.lcse, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.lcse)))
    @argcheck(!(isa(opt.opt.cte, LinearConstraint) ||
                isa(opt.opt.cte, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.cte)))
    @argcheck(!isa(opt.opt.gcarde, LinearConstraint))
    @argcheck(!(isa(opt.opt.sgcarde, LinearConstraint) ||
                isa(opt.opt.sgcarde, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.sgcarde)))
    @argcheck(!isa(opt.opt.ple, AbstractPhylogenyConstraintResult) ||
              isa(opt.opt.ple, AbstractVector) &&
              !any(x -> isa(x, AbstractPhylogenyConstraintResult), opt.opt.ple))
    return nothing
end
function assert_internal_optimiser(opt::VecOptE_Opt)
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
function assert_external_optimiser(opt::ClusteringOptimisationEstimator)
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::JuMPOptimisationEstimator)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult))
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
function assert_external_optimiser(opt::RiskBudgetingOptimiser)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult))
    if isa(opt.rba, FactorRiskBudgeting)
        @argcheck(!isa(opt.rba.re, AbstractRegressionResult))
    end
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::FactorRiskContribution)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult))
    @argcheck(!isa(opt.re, AbstractRegressionResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::VecOptE_Opt)
    assert_external_optimiser.(opt)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Nested Clustered Optimisation (NCO) portfolio optimiser.

`NestedClustered` implements the Nested Clustered Optimisation algorithm. It first clusters assets, then solves a within-cluster (inner) optimisation for each cluster independently, and finally solves an across-cluster (outer) optimisation to combine the cluster portfolios into a final portfolio.

# Fields

  - `pe`: Prior estimator or prior result.
  - `cle`: Clustering estimator or clustering result.
  - `wb`: Weight bounds estimator or bounds.
  - `fees`: Fee estimator or fee structure.
  - `sets`: Asset sets.
  - `opti`: Inner (intra-cluster) portfolio optimiser.
  - `opto`: Outer (inter-cluster) portfolio optimiser.
  - `cv`: Cross-validation configuration for model selection.
  - `wf`: Weight finaliser for enforcing bounds.
  - `ex`: FLoops executor for parallelism.
  - `fb`: Fallback optimiser.
  - `brt`: If `true`, uses bootstrap returns.
  - `cle_pr`: If `true`, passes the prior result to the clustering estimator.
  - `strict`: If `true`, strictly enforces weight bounds.

# Constructors

    NestedClustered(;
        pe::PrE_Pr = EmpiricalPrior(),
        cle::ClE_Cl = ClustersEstimator(),
        wb::Option{<:WbE_Wb} = WeightBounds(),
        fees::Option{<:FeesE_Fees} = nothing,
        sets::Option{<:AssetSets} = nothing,
        opti::NonFiniteAllocationOptimisationEstimator,
        opto::NonFiniteAllocationOptimisationEstimator = opti,
        cv::Option{<:OptimisationCrossValidation} = nothing,
        wf::WeightFinaliser = IterativeWeightFinaliser(),
        ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
        fb::Option{<:OptE_Opt} = nothing,
        brt::Bool = false,
        cle_pr::Bool = true,
        strict::Bool = false
    ) -> NestedClustered

Keywords correspond to the struct's fields.

# Related

  - [`ClusteringOptimisationEstimator`](@ref)
  - [`HierarchicalRiskParity`](@ref)
  - [`Stacking`](@ref)
  - [`NestedClusteredResult`](@ref)
"""
@concrete struct NestedClustered <: ClusteringOptimisationEstimator
    pe
    cle
    wb
    fees
    sets
    opti
    opto
    cv
    wf
    ex
    fb
    brt
    cle_pr
    strict
    function NestedClustered(pe::PrE_Pr, cle::ClE_Cl, wb::Option{<:WbE_Wb},
                             fees::Option{<:FeesE_Fees}, sets::Option{<:AssetSets},
                             opti::NonFiniteAllocationOptimisationEstimator,
                             opto::NonFiniteAllocationOptimisationEstimator,
                             cv::Option{<:OptimisationCrossValidation}, wf::WeightFinaliser,
                             ex::FLoops.Transducers.Executor, fb::Option{<:OptE_Opt},
                             brt::Bool, cle_pr::Bool, strict::Bool)
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
            @argcheck(!isnothing(sets))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets))
        end
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
                         wb::Option{<:WbE_Wb} = nothing,
                         fees::Option{<:FeesE_Fees} = nothing,
                         sets::Option{<:AssetSets} = nothing,
                         opti::NonFiniteAllocationOptimisationEstimator,
                         opto::NonFiniteAllocationOptimisationEstimator,
                         cv::Option{<:OptimisationCrossValidation} = nothing,
                         wf::WeightFinaliser = IterativeWeightFinaliser(),
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         fb::Option{<:OptE_Opt} = nothing, brt::Bool = false,
                         cle_pr::Bool = true, strict::Bool = false)
    return NestedClustered(pe, cle, wb, fees, sets, opti, opto, cv, wf, ex, fb, brt, cle_pr,
                           strict)
end
function assert_internal_optimiser(opt::NestedClustered)
    @argcheck(!isa(opt.cle, AbstractClusteringResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    return nothing
end
function assert_external_optimiser(opt::NestedClustered)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    @argcheck(!isa(opt.cle, AbstractClusteringResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    if !isnothing(opt.cv)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function needs_previous_weights(opt::NestedClustered)
    return (needs_previous_weights(opt.fees) ||
            needs_previous_weights(opt.opti) ||
            needs_previous_weights(opt.opto) ||
            needs_previous_weights(opt.fb))
end
function factory(nco::NestedClustered, w::AbstractVector)
    fees = factory(nco.fees, w)
    opti = factory(nco.opti, w)
    opto = factory(nco.opto, w)
    fb = factory(nco.fb, w)
    return NestedClustered(; pe = nco.pe, cle = nco.cle, wb = nco.wb, fees = fees,
                           sets = nco.sets, opti = opti, opto = opto, cv = nco.cv,
                           wf = nco.wf, ex = nco.ex, fb = fb, brt = nco.brt,
                           cle_pr = nco.cle_pr, strict = nco.strict)
end
function opt_view(nco::NestedClustered, i, X::MatNum)
    X = isa(nco.pe, AbstractPriorResult) ? nco.pe.X : X
    pe = prior_view(nco.pe, i)
    wb = weight_bounds_view(nco.wb, i)
    fees = fees_view(nco.fees, i)
    sets = asset_sets_view(nco.sets, i)
    opti = opt_view(nco.opti, i, X)
    opto = opt_view(nco.opto, i, X)
    return NestedClustered(; pe = pe, cle = nco.cle, wb = wb, fees = fees, sets = sets,
                           opti = opti, opto = opto, cv = nco.cv, wf = nco.wf, ex = nco.ex,
                           fb = nco.fb, brt = nco.brt, cle_pr = nco.cle_pr,
                           strict = nco.strict)
end
"""
    outer_optimisation_finaliser(wb, wf, w_inner, w_outer)

Finalise outer optimisation weights for the NCO algorithm.

Combines inner cluster weights `w_inner` with outer portfolio weights `w_outer`, applying weight bounds `wb` and finalisation algorithm `wf`.

# Arguments

  - `wb`: Weight bounds (optional).
  - `wf`: Weight finaliser.
  - `w_inner`: Inner (within-cluster) weights.
  - `w_outer`: Outer (across-cluster) weights.

# Returns

  - Final combined portfolio weights.

# Related

  - [`NestedClustered`](@ref)
  - [`WeightBounds`](@ref)
"""
function outer_optimisation_finaliser(wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                      ws::VecVecNum, wi::MatNum)
    retcode_w = [outer_optimisation_finaliser(wb, wf, resi, rco, w, wi)
                 for (rco, w) in zip(rcos, ws)]
    return map(x -> x[1], retcode_w), map(x -> x[2], retcode_w)
end
function outer_optimisation_finaliser(wb::Option{<:WeightBounds}, wf::WeightFinaliser,
                                      resi::VecOpt, rco::OptimisationReturnCode, w::VecNum,
                                      wi::MatNum)
    w = wi * w
    retcode, w = finalise_weight_bounds(wf, wb, w)
    wb_flag = isa(retcode, OptimisationFailure)
    opto_flag = isa(rco, OptimisationFailure)
    resi_flag = any(x -> isa(x, OptimisationFailure), getproperty.(resi, :retcode))
    if resi_flag || opto_flag || wb_flag
        msg = ""
        if resi_flag
            msg *= "opti failed.\n"
        end
        if opto_flag
            msg *= "opto failed.\n"
        end
        if wb_flag
            msg *= "Full optimisation failed.\n"
        end
        retcode = OptimisationFailure(; res = msg)
    end
    return retcode, w
end
"""
Overload this using nco.cv for custom cross-validation prediction
"""
function predict_outer_nco_estimator_returns(nco::NestedClustered, rd::ReturnsResult,
                                             pr::AbstractPriorResult, fees::Option{<:Fees},
                                             wi::MatNum, resi::VecOpt, cls::VecVecInt)
    nb, B = if !isa(rd.B, MatNum)
        rd.nb, rd.B
    else
        ["_b$(i)" for i in 1:size(wi, 2)], rd.B * wi
    end
    iv = rd.iv
    ivpa = rd.ivpa
    iv_flag = !isnothing(iv)
    ivpa_flag = isa(ivpa, AbstractVector)
    if iv_flag || ivpa_flag
        wi = abs.(wi)
        if iv_flag
            iv = iv * wi
        end
        if ivpa_flag
            ivpa = transpose(wi) * ivpa
        end
    end
    X = pr.X
    X = Matrix{eltype(X)}(undef, size(X, 1), size(wi, 2))
    for (i, (res, cl)) in enumerate(zip(resi, cls))
        pri = prior_view(pr, cl)
        feesi = fees_view(fees, cl)
        X[:, i] = calc_net_returns(res, pri, feesi)
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         nb = nb, B = B, ts = rd.ts, iv = iv, ivpa = ivpa)
end
"""
    rebuild_returns_result(rd, predictions)

Reconstruct a returns result from cross-validation predictions.

Combines individual fold predictions from `predictions` into a new `ReturnsResult` corresponding to the original data layout.

# Arguments

  - `rd`: Original [`ReturnsResult`](@ref).
  - `predictions`: Vector of [`MultiPeriodPredictionResult`](@ref) objects from cross-validation.

# Returns

  - Rebuilt [`ReturnsResult`](@ref).

# Related

  - [`NestedClustered`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function rebuild_returns_result(rd::ReturnsResult, predictions::VecMPredRes)
    N = length(predictions)
    nb = rd.nb
    B_flag = !isnothing(rd.B)
    iv_flag = !isnothing(rd.iv)
    ivpa_flag = !isnothing(rd.ivpa)
    rd1 = predictions[1].mrd
    X = rd1.X
    B = B_flag ? rd1.B : nothing
    iv = rd1.iv
    ivpa = ivpa_flag ? [rd1.ivpa] : nothing
    @inbounds for i in 2:N
        rdi = predictions[i].mrd
        append!(X, rdi.X)
        if iv_flag
            append!(iv, rdi.iv)
        end
        if ivpa_flag
            push!(ivpa, rdi.ivpa)
        end
        if B_flag
            append!(B, rdi.B)
        end
    end
    X = reshape(X, :, N)
    if B_flag
        B = reshape(B, :, N)
        nb = ["_b$(i)" for i in 1:N]
    end
    iv = iv_flag ? reshape(iv, :, N) : nothing
    return ReturnsResult(; nx = ["_$i" for i in 1:N], X = X, nf = rd1.nf, F = rd1.F,
                         nb = nb, B = B, ts = rd1.ts, iv = iv, ivpa = ivpa)
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
            cvi = !hasproperty(cv, :rng) ? cv : copy(cv)
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
            cvi = !hasproperty(cv, :rng) ? cv : copy(cv)
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
    FLoops.@floop nco.ex for (i, cl) in pairs(cls)
        optic = opt_view(opti, cl, X)
        rdc = returns_result_view(rd, cl)
        res = optimise(optic, rdc; dims = dims, branchorder = branchorder,
                       str_names = str_names, save = save, kwargs...)
        #! Support efficient frontier?
        @argcheck(!isa(res.retcode, AbstractVector))
        wi[cl, i] = res.w
        resi[i] = res
    end
    rdo = predict_outer_nco_estimator_returns(nco, rd, pr, fees, wi, resi, cls)
    reso = optimise(nco.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb = weight_bounds_constraints(nco.wb, nco.sets; N = clr.k, strict = nco.strict,
                                   datatype = eltype(X))
    retcode, w = outer_optimisation_finaliser(wb, nco.wf, resi, reso.retcode, reso.w, wi)
    return NestedClusteredResult(typeof(nco), pr, clr, wb, fees, resi, reso, nco.cv,
                                 retcode, w, nothing)
end
function optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(nco, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export NestedClusteredResult, NestedClustered
