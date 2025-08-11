abstract type BaseStackingOptimisationEstimator <: OptimisationEstimator end
struct StackingOptimisation{T1, T2, T3, T4, T5, T6, T7} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    resi::T4
    reso::T5
    retcode::T6
    w::T7
end
struct Stacking{T1, T2, T3, T4, T5, T6, T7, T8} <: BaseStackingOptimisationEstimator
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
    @assert(!isa(opt.pe, AbstractPriorResult))
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
                  wb::Union{Nothing, <:WeightBoundsEstimator, <:WeightBounds} = nothing,
                  sets::Union{Nothing, <:AssetSets,
                              #! Start: to delete
                              <:DataFrame
                              #! End: to delete
                              } = nothing,
                  opti::AbstractVector{<:Union{<:OptimisationEstimator,
                                               <:OptimisationResult}},
                  opto::OptimisationEstimator,
                  cwf::WeightFinaliser = IterativeWeightFiniliser(), strict::Bool = false,
                  threads::FLoops.Transducers.Executor = ThreadedEx())
    assert_external_optimiser(opto)
    if isa(wb, WeightBoundsEstimator)
        @assert(!isnothing(sets))
    end
    return Stacking(pe, wb, sets, opti, opto, cwf, strict, threads)
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
    pr = prior(st.pe, rd; dims = dims)
    opti = st.opti
    Ni = length(opti)
    wi = zeros(eltype(pr.X), size(pr.X, 2), Ni)
    resi = Vector{OptimisationResult}(undef, Ni)
    @floop st.threads for (i, opt) in pairs(opti)
        res = optimise!(opt, rd; dims = dims, branchorder = branchorder,
                        str_names = str_names, save = save, kwargs...)
        #! Support efficient frontier?
        @assert(!isa(res.retcode, AbstractVector))
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
