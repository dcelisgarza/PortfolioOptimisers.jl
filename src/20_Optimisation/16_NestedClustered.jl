struct NestedClusteredOptimisation{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       OptimisationResult
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
function opt_attempt_factory(res::NestedClusteredOptimisation, fb)
    return NestedClusteredOptimisation(res.oe, res.pr, res.wb, res.clr, res.resi, res.reso,
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
    @argcheck(!(isa(opt.opt.cent, LinearConstraint) ||
                isa(opt.opt.cent, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.cent)))
    @argcheck(!isa(opt.opt.gcard, LinearConstraint))
    @argcheck(!(isa(opt.opt.sgcard, LinearConstraint) ||
                isa(opt.opt.sgcard, AbstractVector) &&
                any(x -> isa(x, LinearConstraint), opt.opt.sgcard)))
    @argcheck(!isa(opt.opt.plg, AbstractPhylogenyConstraintResult) ||
              isa(opt.opt.plg, AbstractVector) &&
              !any(x -> isa(x, AbstractPhylogenyConstraintResult), opt.opt.plg))
    return nothing
end
function assert_internal_optimiser(opt::AbstractVector{<:Union{<:OptimisationEstimator,
                                                               <:OptimisationResult}})
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
function assert_external_optimiser(opt::FactorRiskContribution)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.opt.pe, AbstractPriorResult))
    @argcheck(!isa(opt.re, AbstractRegressionResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::AbstractVector{<:Union{<:OptimisationEstimator,
                                                               <:OptimisationResult}})
    assert_external_optimiser.(opt)
    return nothing
end
"""
"""
struct NestedClustered{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11} <:
       ClusteringOptimisationEstimator
    pe::T1
    cle::T2
    wb::T3
    sets::T4
    opti::T5
    opto::T6
    cv::T7
    cwf::T8
    strict::T9
    threads::T10
    fb::T11
    function NestedClustered(pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                             cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult},
                             wb::Union{Nothing, <:WbUWbE}, sets::Option{<:AssetSets},
                             opti::OptimisationEstimator, opto::OptimisationEstimator,
                             cv::Option{<:CrossValidationEstimator}, cwf::WeightFinaliser,
                             strict::Bool, threads::FLoops.Transducers.Executor,
                             fb::Option{<:OptimisationEstimator})
        assert_external_optimiser(opto)
        if !(opti === opto)
            assert_internal_optimiser(opti)
        end
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(pe), typeof(cle), typeof(wb), typeof(sets), typeof(opti),
                   typeof(opto), typeof(cv), typeof(cwf), typeof(strict), typeof(threads),
                   typeof(fb)}(pe, cle, wb, sets, opti, opto, cv, cwf, strict, threads, fb)
    end
end
function NestedClustered(;
                         pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                         cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult} = ClusteringEstimator(),
                         wb::Union{Nothing, <:WbUWbE} = nothing,
                         sets::Option{<:AssetSets} = nothing, opti::OptimisationEstimator,
                         opto::OptimisationEstimator,
                         cv::Option{<:CrossValidationEstimator} = nothing,
                         cwf::WeightFinaliser = IterativeWeightFinaliser(),
                         strict::Bool = false,
                         threads::FLoops.Transducers.Executor = ThreadedEx(),
                         fb::Option{<:OptimisationEstimator} = nothing)
    return NestedClustered(pe, cle, wb, sets, opti, opto, cv, cwf, strict, threads, fb)
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
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function opt_view(nco::NestedClustered, i, X::NumMat)
    X = isa(nco.pe, AbstractPriorResult) ? nco.pe.X : X
    pe = prior_view(nco.pe, i)
    wb = weight_bounds_view(nco.wb, i)
    sets = nothing_asset_sets_view(nco.sets, i)
    opti = opt_view(nco.opti, i, X)
    opto = opt_view(nco.opto, i, X)
    return NestedClustered(; pe = pe, cle = nco.cle, wb = wb, sets = sets, opti = opti,
                           opto = opto, cv = nco.cv, cwf = nco.cwf, strict = nco.strict,
                           threads = nco.threads, fb = nco.fb)
end
function nested_clustering_finaliser(wb::Union{Nothing, <:WeightBoundsEstimator,
                                               <:WeightBounds}, sets::Option{<:AssetSets},
                                     cwf::WeightFinaliser, strict::Bool,
                                     resi::AbstractVector{<:OptimisationResult},
                                     res::OptimisationResult, w::NumVec;
                                     datatype::DataType = Float64)
    wb = weight_bounds_constraints(wb, sets; N = length(w), strict = strict,
                                   datatype = datatype)
    retcode, w = clustering_optimisation_result(cwf, wb, w)
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
    pr = prior(nco.pe, rd; dims = dims)
    clr = clusterise(nco.cle, pr.X; iv = rd.iv, ivpa = rd.ivpa, dims = dims,
                     branchorder = branchorder)
    idx = cutree(clr.clustering; k = clr.k)
    cls = [findall(x -> x == i, idx) for i in 1:(clr.k)]
    wi = zeros(eltype(pr.X), size(pr.X, 2), clr.k)
    opti = nco.opti
    resi = Vector{OptimisationResult}(undef, clr.k)
    @floop nco.threads for (i, cl) in pairs(cls)
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
    X, F, ts, iv, ivpa = predict_outer_estimator_returns(nco, rd, pr, wi, resi; cls = cls)
    rdo = ReturnsResult(; nx = ["_$i" for i in 1:(clr.k)], X = X, nf = rd.nf, F = F,
                        ts = ts, iv = iv, ivpa = ivpa)
    reso = optimise(nco.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(nco.wb, nco.sets, nco.cwf, nco.strict,
                                                 resi, reso, wi * reso.w;
                                                 datatype = eltype(pr.X))
    return NestedClusteredOptimisation(typeof(nco), pr, wb, clr, resi, reso, nco.cv,
                                       retcode, w, nothing)
end
function optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, str_names::Bool = false,
                  save::Bool = true, kwargs...)
    return _optimise(nco, rd; dims = dims, branchorder = branchorder, str_names = str_names,
                     save = save, kwargs...)
end

export NestedClusteredOptimisation, NestedClustered
