struct NestedClusteredResult{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    clr::T4
    resi::T5
    reso::T6
    cv::T7
    retcode::T8
    w::T9
    fb::T10
end
function factory(res::NestedClusteredResult, fb)
    return NestedClusteredResult(res.oe, res.pr, res.wb, res.clr, res.resi, res.reso,
                                 res.cv, res.retcode, res.w, fb)
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
    @argcheck(!isa(opt.opt.cle, AbstractClusteringResult))
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
struct NestedClustered{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11} <:
       ClusteringOptimisationEstimator
    pr::T1
    cle::T2
    wb::T3
    sets::T4
    opti::T5
    opto::T6
    cv::T7
    wf::T8
    strict::T9
    ex::T10
    fb::T11
    function NestedClustered(pr::PrE_Pr, cle::ClE_Cl, wb::Option{<:WbE_Wb},
                             sets::Option{<:AssetSets}, opti::OptimisationEstimator,
                             opto::OptimisationEstimator,
                             cv::Option{<:CrossValidationEstimator}, wf::WeightFinaliser,
                             strict::Bool, ex::FLoops.Transducers.Executor,
                             fb::Option{<:OptimisationEstimator})
        assert_external_optimiser(opto)
        if !(opti === opto)
            assert_internal_optimiser(opti)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pr), typeof(cle), typeof(wb), typeof(sets), typeof(opti),
                   typeof(opto), typeof(cv), typeof(wf), typeof(strict), typeof(ex),
                   typeof(fb)}(pr, cle, wb, sets, opti, opto, cv, wf, strict, ex, fb)
    end
end
function NestedClustered(; pr::PrE_Pr = EmpiricalPrior(), cle::ClE_Cl = ClustersEstimator(),
                         wb::Option{<:WbE_Wb} = nothing,
                         sets::Option{<:AssetSets} = nothing, opti::OptimisationEstimator,
                         opto::OptimisationEstimator,
                         cv::Option{<:CrossValidationEstimator} = nothing,
                         wf::WeightFinaliser = IterativeWeightFinaliser(),
                         strict::Bool = false,
                         ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                         fb::Option{<:OptimisationEstimator} = nothing)
    return NestedClustered(pr, cle, wb, sets, opti, opto, cv, wf, strict, ex, fb)
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
    @argcheck(!isa(opt.pr, AbstractPriorResult))
    @argcheck(!isa(opt.cle, AbstractClusteringResult))
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
    sets = nothing_asset_sets_view(nco.sets, i)
    opti = opt_view(nco.opti, i, X)
    opto = opt_view(nco.opto, i, X)
    return NestedClustered(; pr = pr, cle = nco.cle, wb = wb, sets = sets, opti = opti,
                           opto = opto, cv = nco.cv, wf = nco.wf, strict = nco.strict,
                           ex = nco.ex, fb = nco.fb)
end
function nested_clustering_finaliser(wb::Option{<:WbE_Wb}, sets::Option{<:AssetSets},
                                     wf::WeightFinaliser, strict::Bool, resi::VecOpt,
                                     res::OptimisationResult, w::VecNum;
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
function _optimise(nco::NestedClustered, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(nco.pr, rd; dims = dims)
    clr = clusterise(nco.cle, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     branchorder = branchorder)
    idx = get_clustering_indices(clr)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    wi = zeros(eltype(pr.X), size(pr.X, 2), clr.k)
    opti = nco.opti
    resi = Vector{OptimisationResult}(undef, clr.k)
    FLoops.@floop nco.ex for (i, cl) in pairs(cls)
        if length(cl) == 1
            wi[cl, i] .= one(eltype(pr.X))
            resi[i] = SingletonOptimisation(OptimisationSuccess(nothing))
        else
            optic = opt_view(opti, cl, pr.X)
            rdc = returns_result_view(rd, cl)
            res = optimise(optic, rdc; dims = dims, branchorder = branchorder,
                           str_names = str_names, save = save, kwargs...)
            #! Support efficient frontier?
            @argcheck(!isa(res.retcode, AbstractVector))
            wi[cl, i] = res.w
            resi[i] = res
        end
    end
    rdo = predict_outer_estimator_returns(nco, rd, pr, wi, resi; cls = cls)
    reso = optimise(nco.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(nco.wb, nco.sets, nco.wf, nco.strict, resi,
                                                 reso, wi * reso.w; datatype = eltype(pr.X))
    return NestedClusteredResult(typeof(nco), pr, wb, clr, resi, reso, nco.cv, retcode, w,
                                 nothing)
end
function optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(nco, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export NestedClusteredResult, NestedClustered
