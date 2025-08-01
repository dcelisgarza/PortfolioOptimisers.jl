abstract type BaseStackingOptimisationEstimator <: OptimisationEstimator end
struct StackingOptimisation{T1 <: Type, T2 <: AbstractPriorResult,
                            T3 <: Union{Nothing, <:WeightBounds},
                            T4 <: AbstractVector{<:OptimisationResult},
                            T5 <: OptimisationResult, T6 <: OptimisationReturnCode,
                            T7 <: AbstractVector} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    resi::T4
    reso::T5
    retcode::T6
    w::T7
end
struct Stacking{T1 <: Union{<:AbstractPriorEstimator, <:AbstractPriorResult},
                T2 <: Union{Nothing, <:WeightBounds, <:AbstractString, Expr,
                            <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                            <:AbstractVector{<:Union{<:AbstractString, Expr}},
                  #! Start: to delete
                            <:WeightBoundsEstimator
                  #! End: to delete
                  }, T3 <: Union{Nothing, <:AssetSets,
                       #! Start: to delete
                                 <:DataFrame
                       #! End: to delete
                       },
                T4 <: Union{<:OptimisationResult, <:OptimisationEstimator,
                            <:AbstractVector{<:Union{<:OptimisationEstimator,
                                                     <:OptimisationResult}}},
                T5 <: OptimisationEstimator, T6 <: WeightFinaliser, T7 <: Bool,
                T8 <: FLoops.Transducers.Executor} <: BaseStackingOptimisationEstimator
    pe::T1
    wb::T2
    sets::T3
    opti::T4
    opto::T5
    cwf::T6
    strict::T7
    threads::T8
end
function assert_external_optimiser(opt::Stacking)
    @smart_assert(!isa(opt.pe, AbstractPriorResult))
    assert_external_optimiser(opt.opti)
    assert_external_optimiser(opt.opto)
    return nothing
end
function assert_internal_optimiser(opt::Stacking)
    assert_internal_optimiser(opt.opti)
    assert_internal_optimiser(opt.opto)
    return nothing
end
function Stacking(;
                  pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                  wb::Union{Nothing, <:WeightBounds, <:AbstractString, Expr,
                            <:AbstractVector{<:AbstractString}, <:AbstractVector{Expr},
                            <:AbstractVector{<:Union{<:AbstractString, Expr}},
                            #! Start: to delete
                            <:WeightBoundsEstimator
                            #! End: to delete
                            } = nothing,
                  sets::Union{Nothing, <:AssetSets,
                              #! Start: to delete
                              <:DataFrame
                              #! End: to delete
                              } = nothing,
                  opti::AbstractVector{<:Union{<:OptimisationEstimator,
                                               <:OptimisationResult}} = [MeanRisk()],
                  opto::OptimisationEstimator = MeanRisk(),
                  cwf::WeightFinaliser = IterativeWeightFiniliser(), strict::Bool = false,
                  threads::FLoops.Transducers.Executor = ThreadedEx())
    assert_external_optimiser(opto)
    if isa(wb, WeightBoundsEstimator)
        @smart_assert(!isnothing(sets))
    end
    return Stacking{typeof(pe), typeof(wb), typeof(sets), typeof(opti), typeof(opto),
                    typeof(cwf), typeof(strict), typeof(threads)}(pe, wb, sets, opti, opto,
                                                                  cwf, strict, threads)
end
function opt_view(st::Stacking, i::AbstractVector, X::AbstractMatrix)
    X = isa(st.pe, AbstractPriorResult) ? st.pe.X : X
    pe = prior_view(st.pe, i)
    opti = opt_view(st.opti, i, X)
    opto = opt_view(st.opto, i, X)
    wb = weight_bounds_view(st.wb, i)
    sets = nothing_asset_sets_view(st.sets, i)
    return Stacking(; pe = pe, opti = opti, opto = opto, wb = wb, cwf = st.cwf, sets = sets,
                    strict = st.strict, threads = st.threads)
end
function optimise!(st::Stacking, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   branchorder::Symbol = :optimal, str_names::Bool = false,
                   save::Bool = true, kwargs...)
    pr = prior(st.pe, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, dims = dims)
    opti = st.opti
    Ni = length(opti)
    wi = zeros(eltype(pr.X), size(pr.X, 2), Ni)
    resi = Vector{OptimisationResult}(undef, Ni)
    @floop st.threads for (i, opt) in pairs(opti)
        res = optimise!(opt, rd; dims = dims, branchorder = branchorder,
                        str_names = str_names, save = save, kwargs...)
        #! Support efficient frontier?
        @smart_assert(!isa(res.retcode, AbstractVector))
        wi[:, i] = res.w
        resi[i] = res
    end
    rdo = ReturnsResult(; nx = 1:Ni, X = pr.X * wi, nf = rd.nf, F = rd.F)
    reso = optimise!(st.opto, rdo; dims = dims, branchorder = branchorder,
                     str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(st.wb, st.sets, st.cwf, st.strict, resi,
                                                 reso, wi * reso.w; datatype = eltype(pr.X))
    return StackingOptimisation(typeof(st), pr, wb, resi, reso, retcode, w)
end

export StackingOptimisation, Stacking
