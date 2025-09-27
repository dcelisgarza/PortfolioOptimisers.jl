abstract type BaseStackingOptimisationEstimator <: OptimisationEstimator end
struct StackingOptimisation{T1, T2, T3, T4, T5, T6, T7, T8} <: OptimisationResult
    oe::T1
    pr::T2
    wb::T3
    resi::T4
    reso::T5
    cv::T6
    retcode::T7
    w::T8
end
struct Stacking{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       BaseStackingOptimisationEstimator
    pe::T1
    wb::T2
    sets::T3
    opti::T4
    opto::T5
    cv::T6
    cwf::T7
    strict::T8
    threads::T9
    fallback::T10
end
function assert_external_optimiser(opt::Stacking)
    #! Maybe results can be allowed with a warning. This goes for other stuff like bounds and threshold vectors. And then the optimisation can throw a domain error when it comes to using them.
    @argcheck(!isa(opt.pe, AbstractPriorResult))
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_external_optimiser(opt.opti)
    end
    return nothing
end
function assert_internal_optimiser(opt::Stacking)
    assert_external_optimiser(opt.opto)
    if !(opt.opti === opt.opto)
        assert_internal_optimiser(opt.opti)
    end
    return nothing
end
function Stacking(;
                  pe::Union{<:AbstractPriorEstimator, <:AbstractPriorResult} = EmpiricalPrior(),
                  wb::Union{Nothing, <:WeightBoundsEstimator, <:WeightBounds} = nothing,
                  sets::Union{Nothing, <:AssetSets} = nothing,
                  opti::AbstractVector{<:Union{<:OptimisationEstimator,
                                               <:OptimisationResult}},
                  opto::OptimisationEstimator,
                  cv::Union{Nothing, <:CrossValidationEstimator} = nothing,
                  cwf::WeightFinaliser = IterativeWeightFinaliser(), strict::Bool = false,
                  threads::FLoops.Transducers.Executor = ThreadedEx(),
                  fallback::Union{Nothing, <:OptimisationEstimator} = nothing)
    assert_external_optimiser(opto)
    if isa(wb, WeightBoundsEstimator)
        @argcheck(!isnothing(sets))
    end
    return Stacking(pe, wb, sets, opti, opto, cv, cwf, strict, threads, fallback)
end
function opt_view(st::Stacking, i::AbstractVector, X::AbstractMatrix)
    X = isa(st.pe, AbstractPriorResult) ? st.pe.X : X
    pe = prior_view(st.pe, i)
    wb = weight_bounds_view(st.wb, i)
    opti = opt_view(st.opti, i, X)
    opto = opt_view(st.opto, i, X)
    sets = nothing_asset_sets_view(st.sets, i)
    return Stacking(; pe = pe, wb = wb, opti = opti, opto = opto, cv = st.cv, cwf = st.cwf,
                    sets = sets, strict = st.strict, threads = st.threads,
                    fallback = st.fallback)
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
        @argcheck(!isa(res.retcode, AbstractVector))
        wi[:, i] = res.w
        resi[i] = res
    end
    X, F, ts, iv, ivpa = predict_outer_estimator_returns(st, rd, wi, pr, resi)
    rdo = ReturnsResult(; nx = ["_$i" for i in 1:Ni], X = X, nf = rd.nf, F = F, ts = ts,
                        iv = iv, ivpa = ivpa)
    reso = optimise!(st.opto, rdo; dims = dims, branchorder = branchorder,
                     str_names = str_names, save = save, kwargs...)
    wb, retcode, w = nested_clustering_finaliser(st.wb, st.sets, st.cwf, st.strict, resi,
                                                 reso, wi * reso.w; datatype = eltype(pr.X))
    return if isa(retcode, OptimisationSuccess) || isnothing(st.fallback)
        StackingOptimisation(typeof(st), pr, wb, resi, reso, st.cv, retcode, w)
    else
        optimise!(st.fallback, rd; dims = dims, branchorder = branchorder,
                  str_names = str_names, save = save, kwargs...)
    end
end

export StackingOptimisation, Stacking
