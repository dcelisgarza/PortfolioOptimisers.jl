"""
    abstract type OpinionPoolingAlgorithm <: AbstractAlgorithm end

Abstract supertype for opinion pooling algorithms.

`OpinionPoolingAlgorithm` is the base type for all algorithms that combine multiple prior estimations into a consensus prior using opinion pooling. All concrete opinion pooling algorithms should subtype this type to ensure a consistent interface for consensus formation in portfolio optimisation workflows.

# Related

  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
abstract type OpinionPoolingAlgorithm <: AbstractAlgorithm end
"""
    struct LinearOpinionPooling <: OpinionPoolingAlgorithm end

Linear opinion pooling algorithm for consensus prior estimation.

`LinearOpinionPooling` is a concrete subtype of [`OpinionPoolingAlgorithm`](@ref) that combines multiple prior probability distributions using a weighted linear average. This algorithm produces a consensus prior by averaging the input opinions according to their specified weights, resulting in a pooled distribution that reflects the collective beliefs of all contributors.

# Details

  - The consensus weights are computed as a linear combination of the individual prior weights, weighted by opinion confidence.
  - Is the weighted arithmetic mean of the individual opinions.
  - Suitable for scenarios where opinions are independent and additive.
  - The only way to force a zero in the final opinion for all opinions to assign it a zero probability.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
struct LinearOpinionPooling <: OpinionPoolingAlgorithm end
"""
    struct LogarithmicOpinionPooling <: OpinionPoolingAlgorithm end

Logarithmic opinion pooling algorithm for consensus prior estimation.

`LogarithmicOpinionPooling` is a concrete subtype of [`OpinionPoolingAlgorithm`](@ref) that combines multiple prior probability distributions using a weighted geometric mean. This algorithm produces a consensus prior by minimising the average Kullback-Leibler divergence from the individual opinions to the pooled distribution, resulting in an information-theoretically optimal consensus.

# Details

  - The consensus weights are computed as the weighted geometric mean of the individual prior weights, weighted by opinion confidence.
  - Robust to extremes, as it down-weights divergent or extreme views.
  - If any opinion assigns zero probability to an event, the pooled opinion will also assign zero probability.
  - Minimises the average Kullback-Leibler divergence from the individual opinions to the consensus.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
struct LogarithmicOpinionPooling <: OpinionPoolingAlgorithm end
"""
    struct OpinionPoolingPrior{T1, T2, T3, T4, T5, T6, T7} <: AbstractLowOrderPriorEstimator_AF
        pes::T1
        pe1::T2
        pe2::T3
        p::T4
        w::T5
        alg::T6
        threads::T7
    end

Opinion pooling prior estimator for asset returns.

`OpinionPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns by combining multiple prior estimations into a consensus prior using opinion pooling algorithms. It supports both linear and logarithmic pooling, flexible weighting of opinions, and optional pre- and post-processing estimators.

# Fields

  - `pes`: Vector of prior estimators to be pooled.
  - `pe1`: Optional pre-processing prior estimator.
  - `pe2`: Post-processing prior estimator.
  - `p`: Penalty parameter for penalising opinions which deviate from the consensus.
  - `w`: Vector of opinion probabilities.
  - `alg`: Opinion pooling algorithm.
  - `threads`: Parallel execution strategy.

# Constructor

    OpinionPoolingPrior(; pes, pe1, pe2, p, w, alg, threads)

Keyword arguments correspond to the fields above. All arguments are validated for type and value consistency.

## Validation

  - `pes` must be a non-empty vector of prior estimators.
  - If `w` is provided, `!isempty(w)`, `length(w) == length(pes)`, `all(x -> 0 <= x <= 1, w)`, and `sum(w) <= 1`.
  - If `p` is provided, `p > 0`.

# Details

  - If `w` is provided, and `sum(w) < 1`, the remaining weight is assigned to the uniform prior. Otherwise, all opinions are assumed to be equally weighted.
  - If `p` is `nothing`, the the opinion probabilities are used as given. Else they are adjusted according to their Kullback-Leibler divergence from the consensus.

# Examples

```jldoctest
julia> sets = AssetSets(; key = "nx", dict = Dict("nx" => ["A", "B", "C"]));

julia> OpinionPoolingPrior(;
                           pes = [EntropyPoolingPrior(; sets = sets,
                                                      mu_views = LinearConstraintEstimator(;
                                                                                           val = ["A == 0.03",
                                                                                                  "B + C == 0.04"])),
                                  EntropyPoolingPrior(; sets = sets,
                                                      mu_views = LinearConstraintEstimator(;
                                                                                           val = ["A == 0.05",
                                                                                                  "B + C >= 0.06"]))])
OpinionPoolingPrior
      pes ┼ EntropyPoolingPrior{EmpiricalPrior{PortfolioOptimisersCovariance{Covariance{SimpleExpectedReturns{Nothing}, GeneralCovariance{StatsBase.SimpleCovariance, Nothing}, Full}, DenoiseDetoneAlgMatrixProcessing{Posdef{UnionAll, @NamedTuple{}}, Nothing, Nothing, Nothing}}, SimpleExpectedReturns{Nothing}, Nothing}, LinearConstraintEstimator{Vector{String}, Nothing}, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, AssetSets{String, String, Dict{String, Vector{String}}}, Nothing, Nothing, OptimEntropyPooling{Tuple{}, @NamedTuple{}, Int64, Float64, ExpEntropyPooling}, Nothing, H1_EntropyPooling}[EntropyPoolingPrior
          │            pe ┼ EmpiricalPrior
          │               │        ce ┼ PortfolioOptimisersCovariance
          │               │           │   ce ┼ Covariance
          │               │           │      │    me ┼ SimpleExpectedReturns
          │               │           │      │       │   w ┴ nothing
          │               │           │      │    ce ┼ GeneralCovariance
          │               │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
          │               │           │      │       │    w ┴ nothing
          │               │           │      │   alg ┴ Full()
          │               │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
          │               │           │      │       pdm ┼ Posdef
          │               │           │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │               │           │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
          │               │           │      │   denoise ┼ nothing
          │               │           │      │    detone ┼ nothing
          │               │           │      │       alg ┴ nothing
          │               │        me ┼ SimpleExpectedReturns
          │               │           │   w ┴ nothing
          │               │   horizon ┴ nothing
          │      mu_views ┼ LinearConstraintEstimator
          │               │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
          │               │   key ┴ nothing
          │     var_views ┼ nothing
          │    cvar_views ┼ nothing
          │   sigma_views ┼ nothing
          │      sk_views ┼ nothing
          │      kt_views ┼ nothing
          │     rho_views ┼ nothing
          │     var_alpha ┼ nothing
          │    cvar_alpha ┼ nothing
          │          sets ┼ AssetSets
          │               │    key ┼ String: "nx"
          │               │   ukey ┼ String: "ux"
          │               │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
          │        ds_opt ┼ nothing
          │        dm_opt ┼ nothing
          │           opt ┼ OptimEntropyPooling
          │               │     args ┼ Tuple{}: ()
          │               │   kwargs ┼ @NamedTuple{}: NamedTuple()
          │               │      sc1 ┼ Int64: 1
          │               │      sc2 ┼ Float64: 1000.0
          │               │      alg ┴ ExpEntropyPooling()
          │             w ┼ nothing
          │           alg ┴ H1_EntropyPooling()
          │ , EntropyPoolingPrior
          │            pe ┼ EmpiricalPrior
          │               │        ce ┼ PortfolioOptimisersCovariance
          │               │           │   ce ┼ Covariance
          │               │           │      │    me ┼ SimpleExpectedReturns
          │               │           │      │       │   w ┴ nothing
          │               │           │      │    ce ┼ GeneralCovariance
          │               │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
          │               │           │      │       │    w ┴ nothing
          │               │           │      │   alg ┴ Full()
          │               │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
          │               │           │      │       pdm ┼ Posdef
          │               │           │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │               │           │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
          │               │           │      │   denoise ┼ nothing
          │               │           │      │    detone ┼ nothing
          │               │           │      │       alg ┴ nothing
          │               │        me ┼ SimpleExpectedReturns
          │               │           │   w ┴ nothing
          │               │   horizon ┴ nothing
          │      mu_views ┼ LinearConstraintEstimator
          │               │   val ┼ Vector{String}: ["A == 0.05", "B + C >= 0.06"]
          │               │   key ┴ nothing
          │     var_views ┼ nothing
          │    cvar_views ┼ nothing
          │   sigma_views ┼ nothing
          │      sk_views ┼ nothing
          │      kt_views ┼ nothing
          │     rho_views ┼ nothing
          │     var_alpha ┼ nothing
          │    cvar_alpha ┼ nothing
          │          sets ┼ AssetSets
          │               │    key ┼ String: "nx"
          │               │   ukey ┼ String: "ux"
          │               │   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
          │        ds_opt ┼ nothing
          │        dm_opt ┼ nothing
          │           opt ┼ OptimEntropyPooling
          │               │     args ┼ Tuple{}: ()
          │               │   kwargs ┼ @NamedTuple{}: NamedTuple()
          │               │      sc1 ┼ Int64: 1
          │               │      sc2 ┼ Float64: 1000.0
          │               │      alg ┴ ExpEntropyPooling()
          │             w ┼ nothing
          │           alg ┴ H1_EntropyPooling()
          │ ]
      pe1 ┼ nothing
      pe2 ┼ EmpiricalPrior
          │        ce ┼ PortfolioOptimisersCovariance
          │           │   ce ┼ Covariance
          │           │      │    me ┼ SimpleExpectedReturns
          │           │      │       │   w ┴ nothing
          │           │      │    ce ┼ GeneralCovariance
          │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
          │           │      │       │    w ┴ nothing
          │           │      │   alg ┴ Full()
          │           │   mp ┼ DenoiseDetoneAlgMatrixProcessing
          │           │      │       pdm ┼ Posdef
          │           │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │           │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
          │           │      │   denoise ┼ nothing
          │           │      │    detone ┼ nothing
          │           │      │       alg ┴ nothing
          │        me ┼ SimpleExpectedReturns
          │           │   w ┴ nothing
          │   horizon ┴ nothing
        p ┼ nothing
        w ┼ nothing
      alg ┼ LinearOpinionPooling()
  threads ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`prior`](@ref)
"""
struct OpinionPoolingPrior{T1, T2, T3, T4, T5, T6, T7} <: AbstractLowOrderPriorEstimator_AF
    pes::T1
    pe1::T2
    pe2::T3
    p::T4
    w::T5
    alg::T6
    threads::T7
    function OpinionPoolingPrior(pes::VecEP,
                                 pe1::Option{<:AbstractLowOrderPriorEstimator_A_F_AF},
                                 pe2::AbstractLowOrderPriorEstimator_A_F_AF,
                                 p::Option{<:Number}, w::Option{<:VecNum},
                                 alg::OpinionPoolingAlgorithm,
                                 threads::FLoops.Transducers.Executor)
        @argcheck(!isempty(pes))
        if !isnothing(p)
            @argcheck(p > zero(p))
        end
        if !isnothing(w)
            @argcheck(!isempty(w))
            @argcheck(length(w) == length(pes))
            @argcheck(all(x -> zero(x) <= x <= one(x), w), DomainError)
            @argcheck(sum(w) <= one(eltype(w)))
        end
        return new{typeof(pes), typeof(pe1), typeof(pe2), typeof(p), typeof(w), typeof(alg),
                   typeof(threads)}(pes, pe1, pe2, p, w, alg, threads)
    end
end
function OpinionPoolingPrior(; pes::VecEP,
                             pe1::Option{<:AbstractLowOrderPriorEstimator_A_F_AF} = nothing,
                             pe2::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             p::Option{<:Number} = nothing, w::Option{<:VecNum} = nothing,
                             alg::OpinionPoolingAlgorithm = LinearOpinionPooling(),
                             threads::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx())
    return OpinionPoolingPrior(pes, pe1, pe2, p, w, alg, threads)
end
"""
    robust_probabilities(ow::VecNum, args...)
    robust_probabilities(ow::VecNum, pw::MatNum, p::Number)

Compute robust opinion probabilities for consensus formation in opinion pooling.

`robust_probabilities` adjusts the vector of opinion probabilities (`ow`) used in opinion pooling algorithms to account for robustness against outlier or extreme opinions. If a penalty parameter `p` is provided, the method penalises opinions that diverge from the consensus by down-weighting them according to their Kullback-Leibler divergence from the pooled distribution. If no penalty parameter is set, the original opinion probabilities are returned unchanged.

# Arguments

  - `ow`: Vector of opinion probabilities (length = number of opinions).
  - `pw`: Matrix of prior weights for each opinion (observations × opinions).
  - `p`: Robustness penalty parameter.

# Returns

  - `ow::VecNum`: Opinion probabilities for pooling.

# Details

  - If `p` is `nothing`, i.e. the method with `args...`, returns the original opinion probabilities.
  - If `p` is provided, computes the consensus distribution, calculates the Kullback-Leibler divergence for each opinion, and applies an exponential penalty to each probability. The adjusted probabilities are normalised to sum to 1.
  - Used internally by [`OpinionPoolingPrior`](@ref) to ensure robust aggregation of opinions.

# Related

  - [`OpinionPoolingPrior`](@ref)
"""
function robust_probabilities(ow::VecNum, args...)
    return ow
end
function robust_probabilities(ow::VecNum, pw::MatNum, p::Number)
    c = pw * ow
    kldivs = [sum(kldivergence(view(pw, :, i), c)) for i in axes(pw, 2)]
    ow .*= exp.(-p * kldivs)
    ow /= sum(ow)
    return ow
end
"""
    compute_pooling(::LinearOpinionPooling, ow::VecNum, pw::MatNum)
    compute_pooling(::LogarithmicOpinionPooling, ow::VecNum, pw::MatNum)

Compute the consensus posterior return distribution from individual prior distributions using opinion pooling.

`compute_pooling` aggregates multiple prior probability distributions (`pw`) into a single consensus posterior distribution according to the specified opinion pooling algorithm and opinion probabilities (`ow`). Supports both linear and logarithmic pooling.

# Arguments

  - `alg`: Opinion pooling algorithm (`LinearOpinionPooling` or `LogarithmicOpinionPooling`).
  - `ow`: Vector of opinion probabilities (length = number of opinions).
  - `pw`: Matrix of prior weights for each opinion (observations × opinions).

# Returns

  - `w::ProbabilityWeights`: Consensus posterior probability weights.

# Details

  - For `LinearOpinionPooling`, computes the weighted arithmetic mean of the individual prior weights: `w = pw * ow`.
  - For `LogarithmicOpinionPooling`, computes the weighted geometric mean of the individual prior weights: `w = exp.(log.(pw) * ow - logsumexp(log.(pw) * ow))`.
  - Used internally by [`OpinionPoolingPrior`](@ref) to form the consensus prior distribution.

# Related

  - [`OpinionPoolingPrior`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
"""
function compute_pooling(::LinearOpinionPooling, ow::VecNum, pw::MatNum)
    return pweights(pw * ow)
end
function compute_pooling(::LogarithmicOpinionPooling, ow::VecNum, pw::MatNum)
    u = log.(pw) * ow
    lse = logsumexp(u)
    return pweights(vec(exp.(u .- lse)))
end
"""
    prior(pe::OpinionPoolingPrior, X::MatNum;
          F::Option{<:MatNum} = nothing, dims::Int = 1, strict::Bool = false,
          kwargs...)

Compute opinion pooling prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns by combining multiple prior estimations into a consensus prior using opinion pooling algorithms. Supports both linear and logarithmic pooling, robust opinion probability adjustment, and optional pre- and post-processing estimators.

# Arguments

  - `pe`: Opinion pooling prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix (default: `nothing`).
  - `dims`: Dimension along which to compute moments (`1` = columns/assets, `2` = rows). Default is `1`.
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, consensus weights, entropy, Kullback-Leibler divergence, opinion probabilities, and optional factor moments.

# Validation

  - `dims in (1, 2)`.

# Details

  - Optional pre-processing estimator `pe.pe1` is applied to asset returns before pooling, else the original returns are used.
  - Each prior estimator in `pe.pes` is applied to the asset returns, producing individual prior weights.
  - Opinion probabilities `ow` are initialised from `pe.w` or set uniformly if it is `nothing`; if their sum is less than 1, the remainder is assigned to a uniform prior.
  - Robust opinion probabilities are computed using [`robust_probabilities`](@ref) if a penalty parameter `pe.p` is provided.
  - Consensus posterior weights are computed using [`compute_pooling`](@ref) according to the specified pooling algorithm `pe.alg`.
  - Post-processing estimator `pe.pe2` is applied using the consensus weights.
  - The result includes the effective number of scenarios, Kullback-Leibler divergence to each opinion, robust opinion probabilities, and optional factor moments.

# Related

  - [`OpinionPoolingPrior`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`robust_probabilities`](@ref)
  - [`compute_pooling`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function prior(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
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
    ow = isnothing(pe.w) ? range(inv(M), inv(M); length = M) : pe.w
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
            pw[:, i] = pr.w
        end
    end
    ow = robust_probabilities(ow, pw, pe.p)
    w = compute_pooling(pe.alg, ow, pw)
    pe2 = factory(pe.pe2, w)
    (; X, mu, sigma, chol, rr, f_mu, f_sigma) = prior(pe2, X, F; strict = strict, kwargs...)
    ens = exp(entropy(w))
    kld = [kldivergence(w, view(pw, :, i)) for i in axes(pw, 2)]
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma, chol = chol, w = w, ens = ens,
                         kld = kld, ow = ow, rr = rr, f_mu = f_mu, f_sigma = f_sigma,
                         f_w = ifelse(!isnothing(rr), w, nothing))
end

export LinearOpinionPooling, LogarithmicOpinionPooling, OpinionPoolingPrior
