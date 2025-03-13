struct HighOrderPriorModel{T1 <: LowOrderAbstractPriorModel,
                           T2 <: Union{Nothing, <:AbstractMatrix},
                           T3 <: Union{Nothing, <:AbstractMatrix},
                           T4 <: Union{Nothing, <:AbstractMatrix}} <:
       AbstractHighOrderPriorModel
    pm::T1
    kt::T2
    sk::T3
    V::T4
end
function HighOrderPriorModel(; pm::LowOrderAbstractPriorModel, kt::AbstractMatrix,
                             sk::AbstractMatrix, V::AbstractMatrix)
    issquarepermissive(sk)
    csk_invalid = isnothing(sk) || isempty(sk)
    v_invalid = isnothing(V) || isempty(V)
    if any((csk_invalid, v_invalid))
        @smart_assert(all((csk_invalid, v_invalid),
                          "If either sk or V, is nothing or empty, both must be nothing or empty."))
    else
        issquare(V)
    end
    return HighOrderPriorModel{typeof(pm), typeof(kt), typeof(sk), typeof(V)}(pm, kt, sk, V)
end
function Base.getproperty(obj::HighOrderPriorModel, sym::Symbol)
    return if sym == :X
        obj.pm.X
    elseif sym == :mu
        obj.pm.mu
    elseif sym == :sigma
        obj.pm.sigma
    else
        getfield(obj, sym)
    end
end
struct HighOrderPriorEstimator{T1 <: AbstractPriorEstimatorMap_1o2_1o2,
                               T2 <: CokurtosisEstimator, T3 <: CoskewnessEstimator}
    pe::T1
    ke::T2
    ske::T3
end
function HighOrderPriorEstimator(;
                                 pe::AbstractPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                                 ke::CokurtosisEstimator = FullCokurtosis(),
                                 ske::CoskewnessEstimator = FullCoskewness())
    return HighOrderPriorEstimator{typeof(pe), typeof(ke), typeof(ske)}(pe, ke, ske)
end
function prior(pe::HighOrderPriorEstimator, X::AbstractMatrix, args...; dims::Int = 1,
               kurt::Bool = true, skew::Bool = true, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    pm = prior(pe.pe, X, args...)
    kt = kurt ? cokurtosis(pe.ke, pm.X) : nothing
    sk, V = skew ? coskewness(pe.ske, pm.X) : (nothing, nothing)
    return HighOrderPriorModel(; pm = pm, kt = kt, sk = sk, V = V)
end

export HighOrderPriorModel, HighOrderPriorEstimator
