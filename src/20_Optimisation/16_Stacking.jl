abstract type BaseStackingOptimisationEstimator <: OptimisationEstimator end
struct StackingResult{T1 <: Type, T2 <: AbstractPriorResult,
                      T3 <: Union{Nothing, <:WeightBoundsResult},
                      T4 <: AbstractVector{<:OptimisationResult}, T5 <: OptimisationResult,
                      T6 <: OptimisationReturnCode, T7 <: AbstractVector} <:
       OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    resi::T4
    reso::T5
    retcode::T6
    w::T7
end
struct Stacking{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                T2 <: Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                T3 <: Union{Nothing, <:DataFrame},
                T4 <:
                Union{<:OptimisationEstimator, <:AbstractVector{<:OptimisationEstimator}},
                T5 <: OptimisationEstimator, T6 <: ClusteringWeightFinaliser, T7 <: Bool,
                T8 <: Bool} <: BaseStackingOptimisationEstimator
    pe::T1
    wb::T2
    sets::T3
    opti::T4
    opto::T5
    cwf::T6
    strict::T7
    threads::T8
end
function Stacking(;
                  pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                  wb::Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint} = nothing,
                  sets::Union{Nothing, <:DataFrame} = nothing,
                  opti::Union{<:OptimisationEstimator,
                              <:AbstractVector{<:OptimisationEstimator}} = MeanRisk(),
                  opto::OptimisationEstimator = MeanRisk(),
                  cwf::ClusteringWeightFinaliser = HeuristicClusteringWeightFiniliser(),
                  strict::Bool = false, threads::Bool = true)
    assert_nested_clustering_optimiser(opto)
    if isa(wb, WeightBoundsConstraint)
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    return Stacking{typeof(pe), typeof(wb), typeof(sets), typeof(opti), typeof(opto),
                    typeof(cwf), typeof(strict), typeof(threads)}(pe, wb, sets, opti, opto,
                                                                  cwf, strict, threads)
end
function opt_view(st::Stacking, i::AbstractVector, X::AbstractMatrix)
    pe = prior_view(st.pe, i)
    opti = opt_view(st.opti, i, X)
    opto = opt_view(st.opto, i, X)
    wb = weight_bounds_view(st.wb, i)
    sets = nothing_dataframe_view(st.sets, i)
    return Stacking(; pe = pe, opti = opti, opto = opto, wb = wb, cwf = st.cwf, sets = sets,
                    strict = st.strict)
end
function optimise!(st::Stacking, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(st.pe, rd.X, rd.F; dims = dims)
    opti = st.opti
    Ni = length(opti)
    threads = st.threads && Ni > one(Ni)
    wi = zeros(eltype(pr.X), size(pr.X, 2), Ni)
    resi = Vector{OptimisationResult}(undef, Ni)
    @conditional_threading threads for i ∈ eachindex(opti)
        res = optimise!(opti[i], rd; dims = dims, branchorder = branchorder,
                        str_names = str_names, save = save, kwargs...)
        wi[:, i] = res.w
        resi[i] = res
    end
    rdo = ReturnsResult(; nx = 1:Ni, X = pr.X * wi, nf = rd.nf, F = rd.F)
    res = optimise!(st.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(st.wb, st.sets, st.cwf, st.strict, resi,
                                                 res, wi * res.w)
    return StackingResult(typeof(st), pr, wb, resi, res, retcode, w)
end

export StackingResult, Stacking
