struct NestedClusteredResult{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11} <:
       NonFiniteAllocationOptimisationResult
    oe::T1
    pr::T2
    clr::T3
    wb::T4
    fees::T5
    resi::T6
    reso::T7
    cv::T8
    retcode::T9
    w::T10
    fb::T11
end
function factory(res::NestedClusteredResult, fb::Option{<:OptE_Opt})
    return NestedClusteredResult(res.oe, res.pr, res.clr, res.wb, res.fees, res.resi,
                                 res.reso, res.cv, res.retcode, res.w, fb)
end
function assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
    @argcheck(!isa(opt.opt.cle, AbstractClusteringResult))
    return nothing
end
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
function assert_external_optimiser(opt::ClusteringOptimisationEstimator)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult))
    @argcheck(!isa(opt.opt.cle, AbstractClusteringResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::JuMPOptimisationEstimator)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
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
struct NestedClustered{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12} <:
       ClusteringOptimisationEstimator
    pe::T1
    cle::T2
    wb::T3
    fees::T4
    sets::T5
    opti::T6
    opto::T7
    cv::T8
    wf::T9
    strict::T10
    ex::T11
    fb::T12
    function NestedClustered(pe::PrE_Pr, cle::ClE_Cl, wb::Option{<:WbE_Wb},
                             fees::Option{<:FeesE_Fees}, sets::Option{<:AssetSets},
                             opti::NonFiniteAllocationOptimisationEstimator,
                             opto::NonFiniteAllocationOptimisationEstimator,
                             cv::Option{<:OptimisationCrossValidation}, wf::WeightFinaliser,
                             strict::Bool, ex::FLoops.Transducers.Executor,
                             fb::Option{<:OptE_Opt})
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
                   typeof(opti), typeof(opto), typeof(cv), typeof(wf), typeof(strict),
                   typeof(ex), typeof(fb)}(pe, cle, wb, fees, sets, opti, opto, cv, wf,
                                           strict, ex, fb)
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
                         strict::Bool = false,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         fb::Option{<:OptE_Opt} = nothing)
    return NestedClustered(pe, cle, wb, fees, sets, opti, opto, cv, wf, strict, ex, fb)
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
                           wf = nco.wf, strict = nco.strict, ex = nco.ex, fb = fb)
end
function opt_view(nco::NestedClustered, i, X::MatNum)
    X = isa(nco.pe, AbstractPriorResult) ? nco.pe.X : X
    pe = prior_view(nco.pe, i)
    wb = weight_bounds_view(nco.wb, i)
    fees = fees_view(nco.fees, i)
    sets = nothing_asset_sets_view(nco.sets, i)
    opti = opt_view(nco.opti, i, X)
    opto = opt_view(nco.opto, i, X)
    return NestedClustered(; pe = pe, cle = nco.cle, wb = wb, fees = fees, sets = sets,
                           opti = opti, opto = opto, cv = nco.cv, wf = nco.wf,
                           strict = nco.strict, ex = nco.ex, fb = nco.fb)
end
function outer_optimisation_finaliser(wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                                      wf::WeightFinaliser, strict::Bool, resi::VecOpt,
                                      rcos::AbstractVector{<:OptimisationReturnCode},
                                      ws::VecVecNum, wi::MatNum;
                                      datatype::DataType = Float64)
    res = [outer_optimisation_finaliser(wb, sets, wf, strict, resi, rco, w, wi;
                                        datatype = datatype) for (rco, w) in zip(rcos, ws)]
    return map(x -> x[1], res), map(x -> x[2], res), map(x -> x[3], res)
end
function outer_optimisation_finaliser(wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                                      wf::WeightFinaliser, strict::Bool, resi::VecOpt,
                                      rco::OptimisationReturnCode, w::VecNum, wi::MatNum;
                                      datatype::DataType = Float64)
    w = wi * w
    wb = weight_bounds_constraints(wb, sets; N = length(w), strict = strict,
                                   datatype = datatype)
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
    return wb, retcode, w
end
"""
Overload this using nco.cv for custom cross-validation prediction
"""
function predict_outer_nco_estimator_returns(nco::NestedClustered, rd::ReturnsResult,
                                             pr::AbstractPriorResult, fees::Option{<:Fees},
                                             wi::MatNum, resi::VecOpt, cls::VecVecInt)
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
    X = Matrix{eltype(pr.X)}(undef, size(pr.X, 1), size(wi, 2))
    for (i, (res, cl)) in enumerate(zip(resi, cls))
        pri = prior_view(pr, cl)
        feesi = fees_view(fees, cl)
        X[:, i] = calc_net_returns(res, pri, feesi)
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         ts = rd.ts, iv = iv, ivpa = ivpa)
end
function rebuild_returns_result(rd::ReturnsResult, predictions::VecMPredRes, N::Integer)
    iv_flag = !isnothing(rd.iv)
    ivpa_flag = !isnothing(rd.ivpa)
    rd1 = predictions[1].mrd
    X = rd1.X
    iv = rd1.iv
    ivpa = ivpa_flag ? [rd1.ivpa] : nothing
    @inbounds for i in 2:length(predictions)
        rdi = predictions[i].mrd
        append!(X, rdi.X)
        if iv_flag
            append!(iv, rdi.iv)
        end
        if ivpa_flag
            push!(ivpa, rdi.ivpa)
        end
    end
    X = reshape(X, :, N)
    iv = iv_flag ? reshape(iv, :, N) : nothing
    return ReturnsResult(; nx = ["_$i" for i in 1:N], X = X, nf = rd1.nf, F = rd1.F,
                         ts = rd1.ts, iv = iv, ivpa = ivpa)
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
    N = length(cls)
    predictions = Vector{MultiPeriodPredictionResult}(undef, N)
    let cv = cv
        FLoops.@floop ex for (i, cl) in enumerate(cls)
            cvi = hasproperty(cv, :rng) ? copy(cv) : cv
            predictions[i] = cross_val_predict(opti, rd, cvi; cols = cl, ex = ex)
        end
    end
    return rebuild_returns_result(rd, predictions, N)
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
    N = length(cls)
    predictions = Vector{PopulationPredictionResult}(undef, N)
    let cv = cv
        FLoops.@floop ex for (i, cl) in enumerate(cls)
            cvi = hasproperty(cv, :rng) ? copy(cv) : cv
            predictions[i] = cross_val_predict(opti, rd, cvi; cols = cl, ex = ex)
        end
    end
    if isnothing(scorer)
        scorer = NearestQuantilePrediction()
    end
    best_predictions = [scorer(prediction) for prediction in predictions]
    return rebuild_returns_result(rd, best_predictions, N)
end
function _optimise(nco::NestedClustered, rd::ReturnsResult; dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(nco.pe, rd; dims = dims)
    clr = clusterise(nco.cle, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     branchorder = branchorder)
    fees = fees_constraints(nco.fees, nco.sets; datatype = eltype(pr.X),
                            strict = nco.strict)
    idx = get_clustering_indices(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    wi = zeros(eltype(pr.X), size(pr.X, 2), clr.k)
    opti = nco.opti
    resi = Vector{NonFiniteAllocationOptimisationResult}(undef, clr.k)
    FLoops.@floop nco.ex for (i, cl) in pairs(cls)
        optic = opt_view(opti, cl, pr.X)
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
    wb, retcode, w = outer_optimisation_finaliser(nco.wb, nco.sets, nco.wf, nco.strict,
                                                  resi, reso.retcode, reso.w, wi;
                                                  datatype = eltype(pr.X))
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
