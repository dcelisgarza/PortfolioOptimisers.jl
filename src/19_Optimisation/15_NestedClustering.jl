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
struct NestedClustering{T1, T2, T3, T4, T5, T6, T7, T8, T9} <:
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
end
function assert_internal_optimiser(opt::ClusteringOptimisationEstimator)
    @smart_assert(!isa(opt.opt.cle, AbstractClusteringResult))
    return nothing
end
function assert_internal_optimiser(opt::JuMPOptimisationEstimator)
    @smart_assert(!isa(opt.opt.lcs, LinearConstraint))
    @smart_assert(!isa(opt.opt.lcm, LinearConstraint))
    @smart_assert(!isa(opt.opt.cent, LinearConstraint))
    @smart_assert(!isa(opt.opt.gcard, LinearConstraint))
    @smart_assert(!isa(opt.opt.sgcard, LinearConstraint))
    # @smart_assert(!isa(opt.opt.smtx, AbstractMatrix))
    @smart_assert(!isa(opt.opt.nplg, PhilogenyResult))
    @smart_assert(!isa(opt.opt.cplg, PhilogenyResult))
    return nothing
end
function assert_internal_optimiser(opt::NestedClustering)
    @smart_assert(!isa(opt.cle, AbstractClusteringResult))
    return nothing
end
function assert_internal_optimiser(opt::AbstractVector{<:OptimisationEstimator})
    assert_internal_optimiser.(opt)
    return nothing
end
function assert_external_optimiser(opt::ClusteringOptimisationEstimator)
    @smart_assert(!isa(opt.opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::JuMPOptimisationEstimator)
    @smart_assert(!isa(opt.opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::NestedClustering)
    @smart_assert(!isa(opt.pe, AbstractPriorResult))
    assert_internal_optimiser(opt)
    return nothing
end
function assert_external_optimiser(opt::AbstractVector{<:OptimisationEstimator})
    assert_internal_optimiser.(opt)
    return nothing
end
function NestedClustering(;
                          pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                          cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult} = ClusteringEstimator(),
                          wb::Union{Nothing, <:WeightBoundsEstimator, <:WeightBounds} = nothing,
                          sets::Union{Nothing, <:AssetSets,
                                      #! Start: to delete
                                      <:DataFrame
                                      #! End: to delete
                                      } = nothing, opti::OptimisationEstimator,
                          opto::OptimisationEstimator = opti,
                          cwf::WeightFinaliser = IterativeWeightFiniliser(),
                          strict::Bool = false,
                          threads::FLoops.Transducers.Executor = ThreadedEx())
    assert_external_optimiser(opto)
    if !(opti === opto)
        assert_internal_optimiser(opti)
    end
    if isa(wb, WeightBoundsEstimator)
        @smart_assert(!isnothing(sets))
    end
    return NestedClustering(pe, cle, wb, sets, opti, opto, cwf, strict, threads)
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
                            threads = nco.threads)
end
function nested_clustering_finaliser(wb::Union{Nothing, <:WeightBoundsEstimator,
                                               <:WeightBounds},
                                     sets::Union{Nothing, <:AssetSets,
                                                 #! Start: to delete
                                                 <:DataFrame
                                                 #! End: to delete
                                                 }, cwf::WeightFinaliser, strict::Bool,
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
            @smart_assert(!isa(res.retcode, AbstractVector))
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
    return NestedClusteringOptimisation(typeof(nco), pr, wb, clr, resi, reso, retcode, w)
end

export NestedClusteringOptimisation, NestedClustering
