struct HighOrderPriorResult{T1 <: AbstractLowOrderPriorResult,
                            T2 <: Union{Nothing, <:AbstractMatrix},
                            T3 <: Union{Nothing, <:AbstractMatrix},
                            T4 <: Union{Nothing, <:AbstractMatrix},
                            T5 <: Union{Nothing, <:AbstractMatrixProcessingEstimator}} <:
       AbstractHighOrderPriorResult
    pm::T1
    kt::T2
    sk::T3
    V::T4
    skmp::T5
end
function HighOrderPriorResult(; pm::AbstractLowOrderPriorResult,
                              kt::Union{Nothing, <:AbstractMatrix},
                              sk::Union{Nothing, <:AbstractMatrix},
                              V::Union{Nothing, <:AbstractMatrix},
                              skmp::Union{Nothing, <:AbstractMatrixProcessingEstimator})
    if isa(kt, AbstractMatrix)
        @smart_assert(!isempty(kt))
        issquare(kt)
        @smart_assert(length(pm.mu)^2 == size(kt, 1))
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
    if sk_flag || V_flag
        @smart_assert(sk_flag && V_flag,
                      "If either sk or V, is nothing, both must be nothing.")
    end
    return HighOrderPriorResult{typeof(pm), typeof(kt), typeof(sk), typeof(V),
                                typeof(skmp)}(pm, kt, sk, V, skmp)
end
function prior_view(pm::HighOrderPriorResult, i::AbstractVector)
    idx = fourth_moment_index_factory(length(pm.mu), i)
    kt = pm.kt
    sk = pm.sk
    skmp = pm.skmp
    sk = nothing_scalar_array_view_odd_order(sk, i, idx)
    V = __coskewness(sk, pm.X, skmp)
    if !isnothing(V) && all(iszero, diag(V))
        V[diagind(V)] = I(size(V, 1))
    end
    return HighOrderPriorResult(; pm = prior_view(pm.pm, i),
                                kt = nothing_scalar_array_view(kt, idx), sk = sk, V = V,
                                skmp = skmp)
end
function Base.getproperty(obj::HighOrderPriorResult, sym::Symbol)
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
                               T2 <: Union{Nothing, CokurtosisEstimator},
                               T3 <: Union{Nothing, CoskewnessEstimator}} <:
       AbstractPriorEstimator_1o2_1o2
    pe::T1
    kte::T2
    ske::T3
end
function factory(pe::HighOrderPriorEstimator,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return HighOrderPriorEstimator(; pe = factory(pe.pe, w), kte = factory(pe.kte, w),
                                   ske = factory(pe.ske, w))
end
function HighOrderPriorEstimator(;
                                 pe::AbstractPriorEstimatorMap_1o2_1o2 = EmpiricalPriorEstimator(),
                                 kte::Union{Nothing, CokurtosisEstimator} = Cokurtosis(;
                                                                                       alg = Full()),
                                 ske::Union{Nothing, CoskewnessEstimator} = Coskewness(;
                                                                                       alg = Full()))
    return HighOrderPriorEstimator{typeof(pe), typeof(kte), typeof(ske)}(pe, kte, ske)
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
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    pm = prior(pe.pe, X, F)
    (; X, mu) = pm
    kt = cokurtosis(pe.kte, X; mean = transpose(mu))
    sk, V = coskewness(pe.ske, X; mean = transpose(mu))
    return HighOrderPriorResult(; pm = pm, kt = kt, sk = sk, V = V,
                                skmp = isnothing(sk) ? nothing : pe.ske.mp)
end

export HighOrderPriorResult, HighOrderPriorEstimator
