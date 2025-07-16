abstract type OpinionPoolingAlgorithm <: AbstractAlgorithm end
struct LinearOpinionPooling <: OpinionPoolingAlgorithm end
struct LogarithmicOpinionPooling <: OpinionPoolingAlgorithm end
struct OpinionPoolingPrior{T1 <:
                           AbstractVector{<:AbstractLowOrderPriorEstimatorMap_1o2_1o2},
                           T2 <:
                           Union{Nothing, <:AbstractLowOrderPriorEstimatorMap_1o2_1o2},
                           T3 <:
                           Union{Nothing, <:AbstractLowOrderPriorEstimatorMap_1o2_1o2},
                           T4 <: Union{Nothing, <:Real},
                           T5 <: Union{Nothing, <:AbstractVector},
                           T6 <: OpinionPoolingAlgorithm,
                           T7 <: FLoops.Transducers.Executor} <:
       AbstractLowOrderPriorEstimator_1o2_1o2
    pes::T1
    pe1::T2
    pe2::T3
    p::T4
    w::T5
    alg::T6
    threads::T7
end
function OpinionPoolingPrior(;
                             pes::AbstractVector{<:AbstractLowOrderPriorEstimator_1o2_1o2},
                             pe1::Union{Nothing, <:AbstractLowOrderPriorEstimator_1o2_1o2} = nothing,
                             pe2::Union{Nothing, <:AbstractLowOrderPriorEstimator_1o2_1o2} = nothing,
                             p::Union{Nothing, <:Real} = nothing,
                             w::Union{Nothing, <:AbstractWeights} = nothing,
                             alg::OpinionPoolingAlgorithm = LinearOpinionPooling(),
                             threads::FLoops.Transducers.Executor = ThreadedEx())
    @smart_assert(!isempty(pes))
    if !isnothing(p)
        @smart_assert(p > zero(p))
    end
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w) && length(w) == length(pes))
        @smart_assert(sum(w) <= one(eltype(w)))
    end
    return OpinionPoolingPrior{typeof(pes), typeof(pe1), typeof(pe2), typeof(p), typeof(w),
                               typeof(alg), typeof(threads)}(pes, pe1, pe2, p, w, alg,
                                                             threads)
end
function robust_probabilities(::AbstractMatrix, ow::AbstractVector, ::Nothing)
    return ow
end
function robust_probabilities(pw::AbstractMatrix, ow::AbstractVector, p::Real)
    c = pw * ow
    kldivs = [sum(kldivergence(view(pw, :, i), c)) for i in axes(pw, 2)]
    ow .*= exp.(-p * kldivs)
    ow /= sum(ow)
    return ow
end
function compute_pooling(::LinearOpinionPooling, pw::AbstractMatrix, ow::AbstractVector)
    return pw * ow
end
function compute_pooling(::LogarithmicOpinionPooling, pw::AbstractMatrix,
                         ow::AbstractVector)
    u = log.(pw) * ow
    lse = logsumexp(u)
    return vec(exp.(u .- lse))
end
function prior(pe::OpinionPoolingPrior, X::AbstractMatrix,
               F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1,
               strict::Bool = false, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    X = !isnothing(pe.pe1) ? prior(pe.pe1, X, F; strict = strict, kwargs...).X : X
    T = size(X, 1)
    M = length(pe.pes)
    pw = Vector{eltype(X)}(undef, T * M)
    @floop pe.threads for (i, pe) in enumerate(pe.pes)
        pr = prior(pe, X, F; strict = strict, kwargs...)
        @smart_assert(!isnothing(pr.w))
        pw[(1 + (i - 1) * M):(i * M)] = pr.w
    end
    ow = isnothing(pe.w) ? range(; start = inv(M), stop = inv(M), length = M) : pe.w
    rw = one(eltype(ow)) - sum(ow)
    if rw > eps(typeof(rw))
        push!(ow, rw)
        append!(pw, range(; start = inv(T), stop = inv(T), length = T))
    end
    pw = reshape(pw, T, :)
    ow = robust_probabilities(pw, ow, pe.p)
    w = pweights(compute_pooling(pe.alg, pw, ow))
    pe2 = if !isnothing(pe.pe2)
        factory(pe.pe2, w)
    else
        factory(EmpiricalPriorEstimator(), w)
    end
    (; X, mu, sigma, chol, loadings, f_mu, f_sigma) = prior(pe2, X, F; strict = strict,
                                                            kwargs...)
    return LowOrderPriorResult(; X = X, mu = mu, sigma = sigma, chol = chol, w = w,
                               loadings = loadings, f_mu = f_mu, f_sigma = f_sigma,
                               f_w = !isnothing(loadings) ? w : nothing)
    return nothing
end
