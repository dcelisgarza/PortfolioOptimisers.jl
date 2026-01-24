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
function factory(res::NestedClusteredResult, fb)
    return NestedClusteredResult(res.oe, res.pr, res.clr, res.wb, res.fees, res.resi,
                                 res.reso, res.cv, res.retcode, res.w, fb)
end
function assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
    @argcheck(!isa(opt.opt.clr, AbstractClusteringResult))
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
function assert_internal_optimiser(opt::JuMPOptimisationEstimator)
    assert_rc_variance(opt)
    @argcheck(!(isa(opt.opt.lcs, LinearConstraint) ||
                isa(opt.opt.lcs, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.lcs)))
    @argcheck(!(isa(opt.opt.ct, LinearConstraint) ||
                isa(opt.opt.ct, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.ct)))
    @argcheck(!isa(opt.opt.gcard, LinearConstraint))
    @argcheck(!(isa(opt.opt.sgcard, LinearConstraint) ||
                isa(opt.opt.sgcard, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.sgcard)))
    @argcheck(!isa(opt.opt.pl, AbstractPhylogenyConstraintResult) ||
              isa(opt.opt.pl, AbstractVector) &&
              !any(x -> isa(x, AbstractPhylogenyConstraintResult), opt.opt.pl))
    return nothing
end
function assert_internal_optimiser(opt::VecOptE_Opt)
    assert_internal_optimiser.(opt)
    return nothing
end
function assert_external_optimiser(opt::ClusteringOptimisationEstimator)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pr, AbstractPriorResult))
    @argcheck(!isa(opt.opt.clr, AbstractClusteringResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::JuMPOptimisationEstimator)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pr, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::FactorRiskContribution)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pr, AbstractPriorResult))
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
    pr::T1
    clr::T2
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
    function NestedClustered(pr::PrE_Pr, clr::ClE_Cl, wb::Option{<:WbE_Wb},
                             fees::Option{<:FeesE_Fees}, sets::Option{<:AssetSets},
                             opti::NonFiniteAllocationOptimisationEstimator,
                             opto::NonFiniteAllocationOptimisationEstimator,
                             cv::Option{<:CrossValidationEstimator}, wf::WeightFinaliser,
                             strict::Bool, ex::FLoops.Transducers.Executor,
                             fb::Option{<:NonFiniteAllocationOptimisationEstimator})
        assert_external_optimiser(opto)
        if !(opti === opto)
            assert_internal_optimiser(opti)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        if isa(fees, FeesEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pr), typeof(clr), typeof(wb), typeof(fees), typeof(sets),
                   typeof(opti), typeof(opto), typeof(cv), typeof(wf), typeof(strict),
                   typeof(ex), typeof(fb)}(pr, clr, wb, fees, sets, opti, opto, cv, wf,
                                           strict, ex, fb)
    end
end
function NestedClustered(; pr::PrE_Pr = EmpiricalPrior(), clr::ClE_Cl = ClustersEstimator(),
                         wb::Option{<:WbE_Wb} = nothing,
                         fees::Option{<:FeesE_Fees} = nothing,
                         sets::Option{<:AssetSets} = nothing,
                         opti::NonFiniteAllocationOptimisationEstimator,
                         opto::NonFiniteAllocationOptimisationEstimator,
                         cv::Option{<:CrossValidationEstimator} = nothing,
                         wf::WeightFinaliser = IterativeWeightFinaliser(),
                         strict::Bool = false,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         fb::Option{<:NonFiniteAllocationOptimisationEstimator} = nothing)
    return NestedClustered(pr, clr, wb, fees, sets, opti, opto, cv, wf, strict, ex, fb)
end
function assert_internal_optimiser(opt::NestedClustered)
    @argcheck(!isa(opt.clr, AbstractClusteringResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    return nothing
end
function assert_external_optimiser(opt::NestedClustered)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pr, AbstractPriorResult))
    @argcheck(!isa(opt.clr, AbstractClusteringResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function opt_view(nco::NestedClustered, i, X::MatNum)
    X = isa(nco.pr, AbstractPriorResult) ? nco.pr.X : X
    pr = prior_view(nco.pr, i)
    wb = weight_bounds_view(nco.wb, i)
    fees = fees_view(nco.fees, i)
    sets = nothing_asset_sets_view(nco.sets, i)
    opti = opt_view(nco.opti, i, X)
    opto = opt_view(nco.opto, i, X)
    return NestedClustered(; pr = pr, clr = nco.clr, wb = wb, fees = fees, sets = sets,
                           opti = opti, opto = opto, cv = nco.cv, wf = nco.wf,
                           strict = nco.strict, ex = nco.ex, fb = nco.fb)
end
function nested_clustering_finaliser(wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                                     wf::WeightFinaliser, strict::Bool, resi::VecOpt,
                                     res::NonFiniteAllocationOptimisationResult, w::VecNum;
                                     datatype::DataType = Float64)
    wb = weight_bounds_constraints(wb, sets; N = length(w), strict = strict,
                                   datatype = datatype)
    retcode, w = finalise_weight_bounds(wf, wb, w)
    wb_flag = isa(retcode, OptimisationFailure)
    opto_flag = isa(res.retcode, OptimisationFailure)
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
    iv = isnothing(rd.iv) ? rd.iv : rd.iv * wi
    ivpa = (isnothing(rd.ivpa) || isa(rd.ivpa, Number)) ? rd.ivpa : transpose(wi) * rd.ivpa
    X = zeros(eltype(pr.X), size(pr.X, 1), size(wi, 2))
    for (i, (res, cl)) in enumerate(zip(resi, cls))
        pri = prior_view(pr, cl)
        feesi = fees_view(fees, cl)
        X[:, i] = predict(res, pri, feesi)
    end
    return ReturnsResult(; nx = ["_$i" for i in 1:size(wi, 2)], X = X, nf = rd.nf, F = rd.F,
                         ts = rd.ts, iv = iv, ivpa = ivpa)
end
function _optimise(nco::NestedClustered, rd::ReturnsResult; dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(nco.pr, rd; dims = dims)
    clr = clusterise(nco.clr, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
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
    wb, retcode, w = nested_clustering_finaliser(nco.wb, nco.sets, nco.wf, nco.strict, resi,
                                                 reso, wi * reso.w; datatype = eltype(pr.X))
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
