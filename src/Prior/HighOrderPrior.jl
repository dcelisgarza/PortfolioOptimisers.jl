struct HighOrderPriorModel{T1 <: AbstractLowOrderPriorModel,
                           T2 <: Union{Nothing, <:AbstractMatrix},
                           T3 <: Union{Nothing, <:AbstractMatrix},
                           T4 <: Union{Nothing, <:AbstractMatrix},
                           T5 <: Union{Nothing, <:AbstractMatrix},
                           T6 <: Union{Nothing, <:MatrixProcessing},
                           T7 <: Union{Nothing, <:AbstractMatrix},
                           T8 <: Union{Nothing, <:AbstractMatrix},
                           T9 <: Union{Nothing, <:MatrixProcessing}} <:
       AbstractHighOrderPriorModel
    pm::T1
    kt::T2
    skt::T3
    sk::T4
    V::T5
    skmp::T6
    ssk::T7
    SV::T8
    sskmp::T9
end
function HighOrderPriorModel(; pm::AbstractLowOrderPriorModel,
                             kt::Union{Nothing, <:AbstractMatrix},
                             skt::Union{Nothing, <:AbstractMatrix},
                             sk::Union{Nothing, <:AbstractMatrix},
                             skmp::Union{Nothing, <:MatrixProcessing},
                             V::Union{Nothing, <:AbstractMatrix},
                             ssk::Union{Nothing, <:AbstractMatrix},
                             SV::Union{Nothing, <:AbstractMatrix},
                             sskmp::Union{Nothing, <:MatrixProcessing})
    if isa(kt, AbstractMatrix)
        @smart_assert(!isempty(kt))
        issquare(kt)
        @smart_assert(length(pm.mu)^2 == size(kt, 1))
    end
    if isa(skt, AbstractMatrix)
        @smart_assert(!isempty(skt))
        issquare(skt)
        @smart_assert(length(pm.mu)^2 == size(skt, 1))
    end
    sk_flag = isa(sk, AbstractMatrix)
    V_flag = isa(V, AbstractMatrix)
    if sk_flag
        @smart_assert(!isempty(sk))
        @smart_assert(length(pm.mu)^2 == size(sk, 2))
    end
    if V_flag
        @smart_assert(!isempty(V))
        issquare(V)
    end
    if any((sk_flag, V_flag))
        @smart_assert(all((sk_flag, V_flag)),
                      "If either sk or V, is nothing, both must be nothing.")
    end
    ssk_flag = isa(ssk, AbstractMatrix)
    SV_flag = isa(SV, AbstractMatrix)
    if ssk_flag
        @smart_assert(!isempty(ssk))
        @smart_assert(length(pm.mu)^2 == size(ssk, 2))
    end
    if SV_flag
        @smart_assert(!isempty(SV))
        issquare(SV)
    end
    if any((ssk_flag, SV_flag))
        @smart_assert(all((ssk_flag, SV_flag)),
                      "If either ssk or SV, is nothing or empty, both must be nothing or empty.")
    end
    return HighOrderPriorModel{typeof(pm), typeof(kt), typeof(skt), typeof(sk), typeof(V),
                               typeof(skmp), typeof(ssk), typeof(SV), typeof(sskmp)}(pm, kt,
                                                                                     skt,
                                                                                     sk, V,
                                                                                     skmp,
                                                                                     ssk,
                                                                                     SV,
                                                                                     sskmp)
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
                               T4 <: CoskewnessEstimator, T5 <: CoskewnessEstimator} <:
       AbstractPriorEstimator_1o2_1o2
    pe::T1
    kte::T2
    skte::T3
    ske::T4
    sske::T5
end
function moment_factory_w(pe::HighOrderPriorEstimator,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return HighOrderPriorEstimator(; pe = moment_factory_w(pe.pe, w),
                                   kte = moment_factory_w(pe.kte, w),
                                   skte = moment_factory_w(pe.skte, w),
                                   ske = moment_factory_w(pe.ske, w),
                                   sske = moment_factory_w(pe.sske, w))
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
function Base.getproperty(obj::HighOrderPriorEstimator, sym::Symbol)
    return if sym == :me
        obj.pe.me
    elseif sym == :ce
        obj.pe.ce
    else
        getfield(obj, sym)
    end
end
function prior(pe::HighOrderPriorEstimator, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               kurt::Bool = true, skurt::Bool = true, skew::Bool = true, sskew::Bool = true,
               kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    pm = prior(pe.pe, X, F; kurt = kurt, skurt = skurt, skew = skew, sskew = sskew,
               kwargs...)
    kt = kurt ? cokurtosis(pe.kte, pm.X) : nothing
    skt = skurt ? cokurtosis(pe.skte, pm.X) : nothing
    sk, V = skew ? coskewness(pe.ske, pm.X) : (nothing, nothing)
    ssk, SV = sskew ? coskewness(pe.sske, pm.X) : (nothing, nothing)
    return HighOrderPriorModel(; pm = pm, kt = kt, skt = skt, sk = sk, V = V,
                               skmp = isnothing(sk) ? nothing : pe.ske.mp, ssk = ssk,
                               SV = SV, sskmp = isnothing(ssk) ? nothing : pe.sske.mp)
end

export HighOrderPriorModel, HighOrderPriorEstimator
