abstract type BaseStackingOptimisationEstimator <: OptimisationEstimator end
struct StackingResult{T1 <: Type, T2 <: AbstractPriorResult,
                      T3 <: AbstractVector{<:OptimisationResult}, T4 <: OptimisationResult,
                      T5 <: Union{Nothing, WeightBoundsResult},
                      T6 <: OptimisationReturnCode, T7 <: AbstractVector} <:
       OptimisationResult
    oe::T1
    pr::T2
    resi::T3
    reso::T4
    wb::T5
    retcode::T6
    w::T7
end
struct Stacking{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                T2 <:
                Union{<:OptimisationEstimator, <:AbstractVector{<:OptimisationEstimator}},
                T3 <: OptimisationEstimator,
                T4 <: Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint},
                T5 <: Union{Nothing, <:DataFrame}, T6 <: ClusteringWeightFinaliser,
                T7 <: Bool} <: BaseStackingOptimisationEstimator
    pe::T1
    opti::T2
    opto::T3
    wb::T4
    sets::T5
    cwf::T6
    strict::T7
end
function Stacking(;
                  pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPriorEstimator(),
                  opti::Union{<:OptimisationEstimator,
                              <:AbstractVector{<:OptimisationEstimator}} = MeanRisk(),
                  opto::OptimisationEstimator = MeanRisk(),
                  wb::Union{Nothing, <:WeightBoundsResult, <:WeightBoundsConstraint} = nothing,
                  sets::Union{Nothing, <:DataFrame} = nothing,
                  cwf::ClusteringWeightFinaliser = HeuristicClusteringWeightFiniliser(),
                  strict::Bool = false)
    assert_nested_clustering_optimiser(opto)
    if isa(wb, WeightBoundsConstraint)
        @smart_assert(isa(sets, DataFrame) && !isempty(sets))
    end
    return Stacking{typeof(pe), typeof(opti), typeof(opto), typeof(wb), typeof(sets),
                    typeof(cwf), typeof(strict)}(pe, opti, opto, wb, sets, cwf, strict)
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
    wi = zeros(eltype(pr.X), size(pr.X, 2), Ni)
    resi = Vector{OptimisationResult}(undef, Ni)
    for (i, opt) ∈ enumerate(opti)
        res = optimise!(opt, rd; dims = dims, branchorder = branchorder,
                        str_names = str_names, save = save, kwargs...)
        wi[:, i] = res.w
        resi[i] = res
    end
    rdo = ReturnsResult(; nx = 1:Ni, X = pr.X * wi, nf = rd.nf, F = rd.F)
    res = optimise!(st.opto, rdo; dims = dims, branchorder = branchorder,
                    str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(st.wb, st.sets, st.cwf, st.strict, resi,
                                                 res, wi * res.w)
    return StackingResult(typeof(st), pr, resi, res, wb, retcode, w)
end

export StackingResult, Stacking
