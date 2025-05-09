struct NestedClusteringOptimisation{T1 <:
                                    Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                                    T2 <: Union{<:ClusteringEstimator,
                                                <:AbstractClusteringResult},
                                    T3 <: OptimisationEstimator,
                                    T4 <: OptimisationEstimator} <: OptimisationEstimator
    pe::T1
    cle::T2
    opti::T3
    opto::T4
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
function assert_nested_clustering_optimiser(opt::NestedClusteringOptimisation)
    @smart_assert(!isa(opt.cle, AbstractClusteringResult))
end
function NestedClusteringOptimisation(;
                                      pe::Union{<:AbstractPriorEstimator,
                                                <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                                      cle::Union{<:ClusteringEstimator,
                                                 <:AbstractClusteringResult} = ClusteringEstimator(),
                                      opti::OptimisationEstimator = MeanRiskEstimator(),
                                      opto::OptimisationEstimator = opti)
    assert_nested_clustering_optimiser(opti)
    if !(opti === opto)
        assert_nested_clustering_optimiser(opto)
    end
    return NestedClusteringOptimisation{typeof(pe), typeof(cle), typeof(opti),
                                        typeof(opto)}(pe, cle, opti, opto)
end
function opt_view(nco::NestedClusteringOptimisation, i::AbstractVector)
    pe = prior_view(nco.pe, i)
    opti = opt_view(nco.opti, i)
    opto = opt_view(nco.opto, i)
    return NestedClusteringOptimisation(; pe = pe, cle = nco.cle, opti = opti, opto = opto)
end
function optimise!(nco::NestedClusteringOptimisation, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(nco.pe, rd.X, rd.F; dims = dims)
    clm = clusterise(nco.cle, pr.X; dims = dims, branchorder = branchorder)
    idx = cutree(clm.clustering; k = clm.k)
    cls = [findall(x -> x == i, idx) for i ∈ 1:(clm.k)]
    return cls
end