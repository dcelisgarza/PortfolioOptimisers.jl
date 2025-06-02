struct NestedClusteringResult{T1 <: Type, T2 <: AbstractPriorResult,
                              T3 <: Union{Nothing, <:WeightBoundsResult},
                              T4 <: AbstractClusteringResult,
                              T5 <: AbstractVector{<:OptimisationResult},
                              T6 <: OptimisationResult, T7 <: OptimisationReturnCode,
                              T8 <: AbstractVector} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    clr::T4
    resi::T5
    reso::T6
    retcode::T7
    w::T8
end
struct NestedClustering{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                        T2 <: Union{<:ClusteringEstimator, <:AbstractClusteringResult},
                        T3 <:
                        Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                        T4 <: Union{Nothing, <:DataFrame}, T5 <: OptimisationEstimator,
                        T6 <: OptimisationEstimator, T7 <: ClusteringWeightFinaliser,
                        T8 <: Bool, T9 <: Bool} <: ClusteringOptimisationEstimator
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
function assert_nested_clustering_optimiser(opt::ClusteringOptimisationEstimator)
    @smart_assert(!isa(opt.opt.cle, AbstractClusteringResult))
end
function assert_nested_clustering_optimiser(opt::JuMPOptimisationEstimator)
    @smart_assert(!isa(opt.opt.lcs, LinearConstraintResult))
    @smart_assert(!isa(opt.opt.lcm, LinearConstraintResult))
    @smart_assert(!isa(opt.opt.cent, LinearConstraintResult))
    @smart_assert(!isa(opt.opt.gcard, LinearConstraintResult))
    @smart_assert(!isa(opt.opt.nplg, PhilogenyConstraintResult))
    @smart_assert(!isa(opt.opt.cplg, PhilogenyConstraintResult))
end
function assert_nested_clustering_optimiser(opt::NestedClustering)
    @smart_assert(!isa(opt.pe, AbstractPriorResult))
    @smart_assert(!isa(opt.cle, AbstractClusteringResult))
end
function NestedClustering(;
                          pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                          cle::Union{<:ClusteringEstimator, <:AbstractClusteringResult} = ClusteringEstimator(),
                          wb::Union{Nothing, <:WeightBoundsResult,
                                    <:WeightBoundsConstraint} = nothing,
                          sets::Union{Nothing, <:DataFrame} = nothing,
                          opti::OptimisationEstimator = MeanRisk(),
                          opto::OptimisationEstimator = opti,
                          cwf::ClusteringWeightFinaliser = HeuristicClusteringWeightFiniliser(),
                          strict::Bool = false, threads::Bool = true)
    assert_nested_clustering_optimiser(opti)
    if !(opti === opto)
        assert_nested_clustering_optimiser(opto)
    end
    @smart_assert(!isa(opto.opt.pe, AbstractPriorResult))
    if isa(wb, WeightBoundsConstraint)
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    return NestedClustering{typeof(pe), typeof(cle), typeof(wb), typeof(sets), typeof(opti),
                            typeof(opto), typeof(cwf), typeof(strict), typeof(threads)}(pe,
                                                                                        cle,
                                                                                        wb,
                                                                                        sets,
                                                                                        opti,
                                                                                        opto,
                                                                                        cwf,
                                                                                        strict,
                                                                                        threads)
end
function opt_view(nco::NestedClustering, i::AbstractVector, X::AbstractMatrix)
    pe = prior_view(nco.pe, i)
    wb = weight_bounds_view(nco.wb, i)
    sets = nothing_dataframe_view(nco.sets, i)
    opti = opt_view(nco.opti, i, X)
    opto = opt_view(nco.opto, i, X)
    return NestedClustering(; pe = pe, cle = nco.cle, wb = wb, sets = sets, opti = opti,
                            opto = opto, cwf = nco.cwf, strict = nco.strict)
end
function nested_clustering_finaliser(wb::Union{Nothing, <:WeightBoundsResult,
                                               <:WeightBoundsConstraint},
                                     sets::Union{Nothing, <:DataFrame},
                                     cwf::ClusteringWeightFinaliser, strict::Bool,
                                     resi::AbstractVector{<:OptimisationResult},
                                     res::OptimisationResult, w::AbstractVector)
    wb = weight_bounds_constraints(wb, sets; N = length(w), strict = strict)
    retcode, w = clustering_optimisation_result(cwf, wb, w)
    if isa(retcode, OptimisationFailure) ||
       isa(res.retcode, OptimisationFailure) ||
       any(isa.(getproperty.(resi, Ref(:retcode)), Ref(OptimisationFailure)))
        msg = ""
        if any(isa.(getproperty.(resi, Ref(:retcode)), Ref(OptimisationFailure)))
            msg *= "opti failed.\n"
        end
        if isa(res.retcode, OptimisationFailure)
            msg *= "opto failed.\n"
        end
        if isa(retcode, OptimisationFailure)
            msg *= "Full optimisation failed.\n"
        end
        retcode = OptimisationFailure(; res = msg)
    end
    return wb, retcode, w
end
function optimise!(nco::NestedClustering, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(nco.pe, rd.X, rd.F; dims = dims)
    clr = clusterise(nco.cle, pr.X; dims = dims, branchorder = branchorder)
    idx = cutree(clr.clustering; k = clr.k)
    threads = nco.threads && clr.k > one(clr.k)
    cls = [findall(x -> x == i, idx) for i ∈ 1:(clr.k)]
    wi = zeros(eltype(pr.X), size(pr.X, 2), clr.k)
    opti = nco.opti
    resi = Vector{OptimisationResult}(undef, clr.k)
    @conditional_threading threads for i ∈ eachindex(cls)
        cl = cls[i]
        if length(cl) == 1
            wi[cl, i] = 1
        else
            optic = opt_view(opti, cl, pr.X)
            rdc = returns_result_view(rd, cl)
            res = optimise!(optic, rdc; dims = dims, branchorder = branchorder,
                            str_names = str_names, save = save, kwargs...)
            wi[cl, i] = res.w
            resi[i] = res
        end
    end
    rdo = ReturnsResult(; nx = 1:(clr.k), X = pr.X * wi, nf = rd.nf, F = rd.F)
    res = optimise!(nco.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(nco.wb, nco.sets, nco.cwf, nco.strict,
                                                 resi, res, wi * res.w)
    return NestedClusteringResult(typeof(nco), pr, wb, clr, resi, res, retcode, w)
end

export NestedClusteringResult, NestedClustering
