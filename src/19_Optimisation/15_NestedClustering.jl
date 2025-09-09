struct NestedClusteringOptimisation{T1, T2, T3, T4, T5, T6, T7, T8} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    clr::T4
    resi::T5
    reso::T6
    retcode::T7
    w::T8
end
struct NestedClustering{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       ClusteringOptimisationEstimator
    pe::T1
    cle::T2
    wb::T3
    sets::T4
    opti::T5
    opto::T6
    cwf::T7
    strict::T8
    threads::T9
    fallback::T10
end
function assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
    @argcheck(!isa(opt.opt.cle, AbstractClusteringResult))
    return nothing
end
function assert_rc_variance(::Any)
    return nothing
end
function assert_rc_variance(opt::Union{<:MeanRisk, <:FactorRiskContribution,
                                       <:NearOptimalCentering, <:RiskBudgeting})
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
    @argcheck(!isa(opt.opt.lcs, LinearConstraint))
    @argcheck(!isa(opt.opt.lcm, LinearConstraint))
    @argcheck(!isa(opt.opt.cent, LinearConstraint))
    @argcheck(!isa(opt.opt.gcard, LinearConstraint))
    @argcheck(!isa(opt.opt.sgcard, LinearConstraint))
    # @argcheck(!isa(opt.opt.smtx, AbstractMatrix))
    @argcheck(!isa(opt.opt.nplg, PhylogenyResult))
    @argcheck(!isa(opt.opt.cplg, PhylogenyResult))
    return nothing
end
function assert_internal_optimiser(opt::NestedClustering)
    @argcheck(!isa(opt.cle, AbstractClusteringResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
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
function assert_external_optimiser(opt::NestedClustering)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    @argcheck(!isa(opt.cle, AbstractClusteringResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function assert_external_optimiser(opt::AbstractVector{<:Union{<:OptimisationEstimator,
                                                               <:OptimisationResult}})
    assert_external_optimiser.(opt)
    return nothing
end
function NestedClustering(;
                          pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                          cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult} = ClusteringEstimator(),
                          wb::Union{Nothing, <:WeightBoundsEstimator, <:WeightBounds} = nothing,
                          sets::Union{Nothing, <:AssetSets} = nothing,
                          opti::OptimisationEstimator, opto::OptimisationEstimator,
                          cwf::WeightFinaliser = IterativeWeightFinaliser(),
                          strict::Bool = false,
                          threads::FLoops.Transducers.Executor = ThreadedEx(),
                          fallback::Union{Nothing, <:OptimisationEstimator} = nothing)
    assert_external_optimiser(opto)
    if !(opti === opto)
        assert_internal_optimiser(opti)
    end
    if isa(wb, WeightBoundsEstimator)
        @argcheck(!isnothing(sets))
    end
    return NestedClustering(pe, cle, wb, sets, opti, opto, cwf, strict, threads, fallback)
end
function opt_view(nco::NestedClustering, i::AbstractVector, X::AbstractMatrix)
    X = isa(nco.pe, AbstractPriorResult) ? nco.pe.X : X
    pe = prior_view(nco.pe, i)
    wb = weight_bounds_view(nco.wb, i)
    sets = nothing_asset_sets_view(nco.sets, i)
    opti = opt_view(nco.opti, i, X)
    opto = opt_view(nco.opto, i, X)
    return NestedClustering(; pe = pe, cle = nco.cle, wb = wb, sets = sets, opti = opti,
                            opto = opto, cwf = nco.cwf, strict = nco.strict,
                            threads = nco.threads, fallback = nco.fallback)
end
function nested_clustering_finaliser(wb::Union{Nothing, <:WeightBoundsEstimator,
                                               <:WeightBounds},
                                     sets::Union{Nothing, <:AssetSets},
                                     cwf::WeightFinaliser, strict::Bool,
                                     resi::AbstractVector{<:OptimisationResult},
                                     res::OptimisationResult, w::AbstractVector;
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
function optimise!(nco::NestedClustering, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
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
            res = optimise!(optic, rdc; dims = dims, branchorder = branchorder,
                            str_names = str_names, save = save, kwargs...)
            #! Support efficient frontier?
            @argcheck(!isa(res.retcode, AbstractVector))
            wi[cl, i] = res.w
            resi[i] = res
        end
    end
    rdo = ReturnsResult(; nx = 1:(clr.k), X = pr.X * wi, nf = rd.nf, F = rd.F)
    reso = optimise!(nco.opto, rdo; dims = dims, branchorder = branchorder,
                     str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(nco.wb, nco.sets, nco.cwf, nco.strict,
                                                 resi, reso, wi * reso.w;
                                                 datatype = eltype(pr.X))
    return if isa(retcode, OptimisationSuccess) || isnothing(nco.fallback)
        NestedClusteringOptimisation(typeof(nco), pr, wb, clr, resi, reso, retcode, w)
    else
        optimise!(nco.fallback, rd; dims = dims, branchorder = branchorder,
                  str_names = str_names, save = save, kwargs...)
    end
end

export NestedClusteringOptimisation, NestedClustering
