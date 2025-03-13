struct HighOrderPriorModel{T1 <: LowOrderAbstractPriorModel,
                           T2 <: Union{Nothing, <:AbstractMatrix},
                           T3 <: Union{Nothing, <:AbstractMatrix},
                           T4 <: Union{Nothing, <:AbstractMatrix},
                           T5 <: Union{Nothing, <:AbstractMatrix},
                           T6 <: Union{Nothing, <:AbstractMatrix},
                           T7 <: Union{Nothing, <:AbstractMatrix}} <:
       AbstractHighOrderPriorModel
    pm::T1
    kt::T2
    skt::T3
    sk::T4
    V::T5
    ssk::T6
    SV::T7
end
function HighOrderPriorModel(; pm::LowOrderAbstractPriorModel,
                             kt::Union{Nothing, <:AbstractMatrix},
                             skt::Union{Nothing, <:AbstractMatrix},
                             sk::Union{Nothing, <:AbstractMatrix},
                             V::Union{Nothing, <:AbstractMatrix},
                             ssk::Union{Nothing, <:AbstractMatrix},
                             SV::Union{Nothing, <:AbstractMatrix})
    issquarepermissive(kt)
    csk_invalid = isnothing(sk) || isempty(sk)
    v_invalid = isnothing(V) || isempty(V)
    if any((csk_invalid, v_invalid))
        @smart_assert(all((csk_invalid, v_invalid)),
                      "If either sk or V, is nothing or empty, both must be nothing or empty.")
    else
        @smart_assert(size(sk, 1)^2 == size(sk, 2))
        issquare(V)
    end
    issquarepermissive(skt)
    cssk_invalid = isnothing(ssk) || isempty(ssk)
    sv_invalid = isnothing(SV) || isempty(SV)
    if any((cssk_invalid, sv_invalid))
        @smart_assert(all((cssk_invalid, sv_invalid)),
                      "If either ssk or SV, is nothing or empty, both must be nothing or empty.")
    else
        @smart_assert(size(ssk, 1)^2 == size(ssk, 2))
        issquare(SV)
    end
    return HighOrderPriorModel{typeof(pm), typeof(kt), typeof(skt), typeof(sk), typeof(V),
                               typeof(ssk), typeof(SV)}(pm, kt, skt, sk, V, ssk, SV)
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
                               T2 <: CokurtosisEstimator, T3 <: CokurtosisEstimator,
                               T4 <: CoskewnessEstimator, T5 <: CoskewnessEstimator}
    pe::T1
    kte::T2
    skte::T3
    ske::T4
    sske::T5
end
function HighOrderPriorEstimator(;
                                 pe::AbstractPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                                 kte::CokurtosisEstimator = FullCokurtosis(),
                                 skte::CokurtosisEstimator = SemiCokurtosis(),
                                 ske::CoskewnessEstimator = FullCoskewness(),
                                 sske::CoskewnessEstimator = SemiCoskewness())
    return HighOrderPriorEstimator{typeof(pe), typeof(kte), typeof(skte), typeof(ske),
                                   typeof(sske)}(pe, kte, skte, ske, sske)
end
function prior(pe::HighOrderPriorEstimator, X::AbstractMatrix, args...; dims::Int = 1,
               kurt::Bool = true, skurt::Bool = true, skew::Bool = true, sskew = true,
               kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    pm = prior(pe.pe, X, args...)
    kt = kurt ? cokurtosis(pe.kte, pm.X) : nothing
    skt = skurt ? cokurtosis(pe.skte, pm.X) : nothing
    sk, V = skew ? coskewness(pe.ske, pm.X) : (nothing, nothing)
    ssk, SV = sskew ? coskewness(pe.sske, pm.X) : (nothing, nothing)
    return HighOrderPriorModel(; pm = pm, kt = kt, skt = skt, sk = sk, V = V, ssk = ssk,
                               SV = SV)
end

export HighOrderPriorModel, HighOrderPriorEstimator
