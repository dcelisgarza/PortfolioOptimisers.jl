abstract type OpinionPoolingAlgorithm <: AbstractAlgorithm end
struct LinearOpinionPooling <: OpinionPoolingAlgorithm end
struct LogarithmicOpinionPooling <: OpinionPoolingAlgorithm end
struct OpinionPoolingPrior{T1,T2,T3,T4,T5,T6,T7} <: AbstractLowOrderPriorEstimator_1o2_1o2
    pes::T1
    pe1::T2
    pe2::T3
    p::T4
    w::T5
    alg::T6
    threads::T7
end
function OpinionPoolingPrior(;
    pes::AbstractVector{<:AbstractLowOrderPriorEstimatorMap_1o2_1o2},
    pe1::Union{Nothing,<:AbstractLowOrderPriorEstimatorMap_1o2_1o2} = nothing,
    pe2::AbstractLowOrderPriorEstimatorMap_1o2_1o2 = EmpiricalPrior(),
    p::Union{Nothing,<:Real} = nothing,
    w::Union{Nothing,<:AbstractVector} = nothing,
    alg::OpinionPoolingAlgorithm = LinearOpinionPooling(),
    threads::FLoops.Transducers.Executor = ThreadedEx(),
)
    @argcheck(!isempty(pes))
    if !isnothing(p)
        @argcheck(p >= zero(p))
    end
    if isa(w, AbstractVector)
        @argcheck(!isempty(w) && length(w) == length(pes))
        @argcheck(
            all(x -> zero(x) <= x <= one(x), w),
            DomainError(
                w,
                range_msg("all entries of `w`", zero(w), one(w), nothing, true, true) * ".",
            )
        )
        @argcheck(sum(w) <= one(eltype(w)))
    end
    return OpinionPoolingPrior(pes, pe1, pe2, p, w, alg, threads)
end
function robust_probabilities(ow::AbstractVector, args...)
    return ow
end
function robust_probabilities(ow::AbstractVector, pw::AbstractMatrix, p::Real)
    c = pw * ow
    kldivs = [sum(kldivergence(view(pw, :, i), c)) for i in axes(pw, 2)]
    ow .*= exp.(-p * kldivs)
    ow /= sum(ow)
    return ow
end
function compute_pooling(::LinearOpinionPooling, ow::AbstractVector, pw::AbstractMatrix)
    return pweights(pw * ow)
end
function compute_pooling(
    ::LogarithmicOpinionPooling,
    ow::AbstractVector,
    pw::AbstractMatrix,
)
    u = log.(pw) * ow
    lse = logsumexp(u)
    return pweights(vec(exp.(u .- lse)))
end
function prior(
    pe::OpinionPoolingPrior,
    X::AbstractMatrix,
    F::Union{Nothing,<:AbstractMatrix} = nothing;
    dims::Int = 1,
    strict::Bool = false,
    kwargs...,
)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    X = !isnothing(pe.pe1) ? prior(pe.pe1, X, F; strict = strict, kwargs...).X : X
    T = size(X, 1)
    M = length(pe.pes)
    ow = isnothing(pe.w) ? range(; start = inv(M), stop = inv(M), length = M) : pe.w
    rw = one(eltype(ow)) - sum(ow)
    if rw > eps(typeof(rw))
        pw = Matrix{eltype(X)}(undef, T, M + 1)
        push!(ow, rw)
        pw[:, end] .= inv(T)
    else
        pw = Matrix{eltype(X)}(undef, T, M)
    end
    let X = X, F = F, pw = pw
        @floop pe.threads for (i, pe) in enumerate(pe.pes)
            pr = prior(pe, X, F; strict = strict, kwargs...)
            @argcheck(!isnothing(pr.w))
            pw[:, i] = pr.w
        end
    end
    ow = robust_probabilities(ow, pw, pe.p)
    w = compute_pooling(pe.alg, ow, pw)
    pe2 = factory(pe.pe2, w)
    (; X, mu, sigma, chol, rr, f_mu, f_sigma) = prior(pe2, X, F; strict = strict, kwargs...)
    ens = exp(entropy(w))
    kld = [kldivergence(w, view(pw, :, i)) for i in axes(pw, 2)]
    return LowOrderPrior(;
        X = X,
        mu = mu,
        sigma = sigma,
        chol = chol,
        w = w,
        ens = ens,
        kld = kld,
        ow = ow,
        rr = rr,
        f_mu = f_mu,
        f_sigma = f_sigma,
        f_w = ifelse(!isnothing(rr), w, nothing),
    )
end

export LinearOpinionPooling, LogarithmicOpinionPooling, OpinionPoolingPrior
