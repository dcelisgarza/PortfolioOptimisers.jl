"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for opinion pooling algorithms.

`OpinionPoolingAlgorithm` is the base type for all algorithms that combine multiple prior estimations into a consensus prior using opinion pooling. All concrete opinion pooling algorithms should subtype this type to ensure a consistent interface for consensus formation in portfolio optimisation workflows.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `OpinionPoolingAlgorithm` and implement the following method:

## Required method name

  - `compute_pooling(alg::OpinionPoolingAlgorithm, ow::VecNum, pw::MatNum) -> StatsBase.ProbabilityWeights`: Aggregate the columns of `pw` into one consensus scenario-weight vector.

### Arguments

  - `alg`: The concrete subtype instance.
  - `ow`: ``K \\times 1`` vector of opinion probabilities, summing to 1.
  - `pw`: ``T \\times K`` matrix whose column `k` holds expert `k`'s scenario weights.

### Returns

  - `w::StatsBase.ProbabilityWeights`: ``T \\times 1`` consensus scenario weights, summing to 1.

### Examples

```jldoctest
julia> struct MedianOpinionPooling <: PortfolioOptimisers.OpinionPoolingAlgorithm end

julia> function PortfolioOptimisers.compute_pooling(::MedianOpinionPooling, ow, pw)
           w = vec(mapslices(PortfolioOptimisers.Statistics.median, pw; dims = 2))
           return PortfolioOptimisers.StatsBase.pweights(w / sum(w))
       end

julia> PortfolioOptimisers.compute_pooling(MedianOpinionPooling(), [0.5, 0.5],
                                           [0.5 0.25; 0.25 0.25; 0.25 0.5])
3-element ProbabilityWeights{Float64, Float64, Vector{Float64}}:
 0.375
 0.25
 0.375
```

# Related

  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref)

# References

  - $(ref_dict[:dietrichlist2017])
  - $(ref_dict[:martinisprenger2017])
"""
abstract type OpinionPoolingAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Pools the opinions as a weighted arithmetic mean of their scenario weights.

Each scenario's consensus weight is the opinion-weighted average of what the experts assign to it, so the pooled distribution keeps every scenario any one expert believes in.

# Mathematical definition

```math
\\begin{align}
p_t^{*} &= \\sum_{k=1}^{K} \\alpha_k\\, p_{tk}\\,.
\\end{align}
```

# Details

  - Suitable where the opinions are independent and additive.
  - A scenario reaches zero in the consensus only when **every** opinion assigns it zero probability, because a sum of non-negative terms vanishes only when all of them do. This is the property that separates it from [`LogarithmicOpinionPooling`](@ref), where one zero is enough.
  - The pooled distribution is a mixture, so it is at least as dispersed as the most dispersed opinion.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref)

# References

  - $(ref_dict[:dietrichlist2017])
"""
struct LinearOpinionPooling <: OpinionPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Pools the opinions as a weighted geometric mean of their scenario weights, renormalised.

The result is the distribution that minimises the opinion-weighted Kullback-Leibler divergence to the individual opinions, which makes it the information-theoretic consensus.

# Mathematical definition

```math
\\begin{align}
p_t^{*} &= \\frac{\\exp\\!\\left(\\sum_{k=1}^{K} \\alpha_k \\log p_{tk}\\right)}{\\sum_{s=1}^{T} \\exp\\!\\left(\\sum_{k=1}^{K} \\alpha_k \\log p_{sk}\\right)}\\,.
\\end{align}
```

# Details

  - Robust to extremes, because it down-weights a scenario that any one opinion doubts.
  - A scenario reaches zero in the consensus as soon as **one** opinion assigns it zero probability, since a single ``\\log 0`` sends the exponent to ``-\\infty``. [`LinearOpinionPooling`](@ref) needs all of them to agree.
  - The normalisation runs through `LogExpFunctions.logsumexp`, so the exponent is shifted before it is exponentiated and a very negative sum does not underflow to a vector of zeros.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref)

# References

  - $(ref_dict[:good1952])
  - $(ref_dict[:dietrichlist2017])
"""
struct LogarithmicOpinionPooling <: OpinionPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Opinion pooling prior estimator for asset returns.

`OpinionPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns by combining multiple prior estimations into a consensus prior using opinion pooling algorithms. It supports both linear and logarithmic pooling, flexible weighting of opinions, and optional pre- and post-processing estimators.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OpinionPoolingPrior(;
        pes::VecEP,
        pe1::Option{<:AbstractLowOrderPriorEstimator_A_F_AF} = nothing,
        pe2::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        p::Option{<:Number} = nothing,
        w::Option{<:VecNum} = nothing,
        alg::OpinionPoolingAlgorithm = LinearOpinionPooling(),
        ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx()
    ) -> OpinionPoolingPrior

Keywords correspond to the struct's fields. All arguments are validated for type and value consistency.

## Validation

  - `pes` must be a non-empty vector of prior estimators.
  - If `w` is not `nothing`, `!isempty(w)`, `length(w) == length(pes)`, `all(x -> 0 <= x <= 1, w)`, and `sum(w) <= 1`.
  - If `p` is not `nothing`, `p > 0`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pes`: Recursively updated via [`factory`](@ref).
  - `pe1`: Recursively updated via [`factory`](@ref).
  - `pe2`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pes`: Recursively viewed via [`port_opt_view`](@ref).
  - `pe1`: Recursively viewed via [`port_opt_view`](@ref).
  - `pe2`: Recursively viewed via [`port_opt_view`](@ref).

# Details

  - If `w` is `nothing`, all opinions are equally weighted at `1/length(pes)`.
  - If `w` is not `nothing` and `sum(w) < 1`, the remaining weight is assigned to a uniform prior over the observations. That remainder is an opinion in its own right: it takes a column of `pw` and it is penalised alongside the others.
  - If `p` is `nothing`, the opinion probabilities are used as given. Otherwise they are adjusted by their Kullback-Leibler divergence from the consensus, through [`robust_probabilities`](@ref).
  - `p` is bounded below by zero strictly. `p = nothing` is how one asks for no penalty; there is no `p = 0`.
  - The opinions contribute **observation weights** alone. Every moment of the result comes from refitting `pe2` under the pooled weights, which is why `pes` is typed to the entropy-pooling estimators — they are the ones whose result carries a `w`.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> OpinionPoolingPrior(;
                           pes = [EntropyPoolingPrior(; sets = sets,
                                                      mu_views = LinearConstraintEstimator(;
                                                                                           val = [\"A == 0.03\",
                                                                                                  \"B + C == 0.04\"])),
                                  EntropyPoolingPrior(; sets = sets,
                                                      mu_views = LinearConstraintEstimator(;
                                                                                           val = [\"A == 0.05\",
                                                                                                  \"B + C >= 0.06\"]))])
OpinionPoolingPrior
  pes ┼ 2-element Vector{EntropyPoolingPrior}
      │ EntropyPoolingPrior ⋯
      │ EntropyPoolingPrior ⋯
  pe1 ┼ nothing
  pe2 ┼ EmpiricalPrior
      │        ce ┼ PortfolioOptimisersCovariance
      │           │   ce ┼ Covariance
      │           │      │    me ┼ SimpleExpectedReturns
      │           │      │       │   w ┴ nothing
      │           │      │    ce ┼ GeneralCovariance
      │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │           │      │       │    w ┴ nothing
      │           │      │   alg ┼ FullMoment()
      │           │      │     w ┴ nothing
      │           │   mp ┼ MatrixProcessing
      │           │      │     pdm ┼ Posdef
      │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │           │      │      dn ┼ nothing
      │           │      │      dt ┼ nothing
      │           │      │     alg ┼ nothing
      │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │        me ┼ SimpleExpectedReturns
      │           │   w ┴ nothing
      │   horizon ┴ nothing
    p ┼ nothing
    w ┼ nothing
  alg ┼ LinearOpinionPooling()
   ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`prior`](@ref)
  - [`robust_probabilities`](@ref)
  - [`compute_pooling`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:dietrichlist2017])
  - $(ref_dict[:martinisprenger2017])
"""
@propagatable @concrete struct OpinionPoolingPrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pes])
    """
    @fprop @vprop pes
    """
    $(field_dict[:pe1])
    """
    @fprop @vprop pe1
    """
    $(field_dict[:pe2])
    """
    @fprop @vprop pe2
    """
    $(field_dict[:p_pool])
    """
    p
    """
    $(field_dict[:op_w])
    """
    w
    """
    $(field_dict[:opalg])
    """
    alg
    """
    $(field_dict[:ex])
    """
    ex
    function OpinionPoolingPrior(pes::VecEP,
                                 pe1::Option{<:AbstractLowOrderPriorEstimator_A_F_AF},
                                 pe2::AbstractLowOrderPriorEstimator_A_F_AF,
                                 p::Option{<:Number}, w::Option{<:VecNum},
                                 alg::OpinionPoolingAlgorithm,
                                 ex::FLoops.Transducers.Executor)
        @argcheck(!isempty(pes), IsEmptyError("pes cannot be empty"))
        if !isnothing(p)
            @argcheck(p > zero(p), DomainError(p, "p must be > 0"))
        end
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
            @argcheck(length(w) == length(pes),
                      DimensionMismatch("length(w) ($(length(w))) must match length(pes) ($(length(pes)))"))
            @argcheck(all(x -> zero(x) <= x <= one(x), w), DomainError)
            @argcheck(sum(w) <= one(eltype(w)),
                      DomainError("sum(w) ($(sum(w))) must be <= 1"))
        end
        return new{typeof(pes), typeof(pe1), typeof(pe2), typeof(p), typeof(w), typeof(alg),
                   typeof(ex)}(pes, pe1, pe2, p, w, alg, ex)
    end
end
function OpinionPoolingPrior(; pes::VecEP,
                             pe1::Option{<:AbstractLowOrderPriorEstimator_A_F_AF} = nothing,
                             pe2::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             p::Option{<:Number} = nothing, w::Option{<:VecNum} = nothing,
                             alg::OpinionPoolingAlgorithm = LinearOpinionPooling(),
                             ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx())::OpinionPoolingPrior
    return OpinionPoolingPrior(pes, pe1, pe2, p, w, alg, ex)
end
"""
    robust_probabilities(ow::VecNum, args...)
    robust_probabilities(ow::VecNum, pw::MatNum, p::Number)

Compute robust opinion probabilities for consensus formation in opinion pooling.

`robust_probabilities` adjusts the vector of opinion probabilities (`ow`) used in opinion pooling algorithms to account for robustness against outlier or extreme opinions. If a penalty parameter `p` is not `nothing`, the method penalises opinions that diverge from the consensus by down-weighting them according to their Kullback-Leibler divergence from the pooled distribution. If no penalty parameter is set, the original opinion probabilities are returned unchanged.

# Mathematical definition

```math
\\begin{align}
D_k &= \\sum_{t=1}^{T} p_{tk} \\log\\!\\frac{p_{tk}}{c_t}\\,, \\quad c_t = \\sum_{k=1}^{K} \\alpha_k p_{tk}\\,, \\\\
\\tilde{\\alpha}_k &= \\frac{\\alpha_k \\exp(-\\rho D_k)}{\\sum_{j=1}^{K} \\alpha_j \\exp(-\\rho D_j)}\\,.
\\end{align}
```

Where:

  - ``\\alpha_k``: Opinion probability of expert ``k``, the input `ow`.
  - ``p_{tk}``: Scenario weight for scenario ``t`` from expert ``k``, column ``k`` of `pw`.
  - ``c_t``: Consensus scenario weight.
  - ``D_k``: Kullback-Leibler divergence from expert ``k`` to the consensus.
  - ``\\rho``: Penalty parameter, the argument `p`.
  - $(math_dict[:T])

The consensus ``\\boldsymbol{c}`` is always the **linear** pool, whatever [`OpinionPoolingAlgorithm`](@ref) the caller selected. The divergence is also directed: it reads ``D_k = \\mathrm{KL}(\\boldsymbol{p}_k \\,\\|\\, \\boldsymbol{c})``, from each opinion to the consensus. The `kld` field of the result runs the other way, from the consensus to each opinion, so the two are different numbers and neither is the other's mirror.

# Arguments

  - `ow`: Vector of opinion probabilities (length = number of opinions).
  - `pw`: Matrix of prior weights for each opinion (observations × opinions).
  - `p`: Robustness penalty parameter.

# Returns

  - `ow::VecNum`: A **new** vector of opinion probabilities for pooling, summing to 1. The argument is never modified, because it may be the estimator's own `w` field.

# Details

  - If `p` is `nothing`, i.e. the method with `args...`, returns the original opinion probabilities.
  - If `p` is not `nothing`, computes the consensus distribution, computes the Kullback-Leibler divergence for each opinion, and applies an exponential penalty to each probability. The adjusted probabilities are normalised to sum to 1.
  - A larger `p` concentrates the mass on the opinions nearest the consensus. As `p` grows without bound the pool tends to the single closest opinion.
  - Used internally by [`OpinionPoolingPrior`](@ref) to ensure robust aggregation of opinions.

# Related

  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref)
"""
function robust_probabilities(ow::VecNum, args...)
    return ow
end
function robust_probabilities(ow::VecNum, pw::MatNum, p::Number)
    c = pw * ow
    kldivs = [sum(StatsBase.kldivergence(view(pw, :, i), c)) for i in axes(pw, 2)]
    # Build a new vector rather than writing into `ow`. `ow` is the caller's, and when the
    # opinion weights already sum to one it *is* `pe.w`, the estimator's own stored field: an
    # in-place `.*=` there left the estimator holding penalised weights that no longer sum to
    # one, so a second `prior` call pooled a uniform-prior remainder nobody asked for. The
    # uniform-weight branch of `prior` also hands in an immutable `range`, which `.*=` cannot
    # write to at all.
    ow = ow .* exp.(-p * kldivs)
    return ow / sum(ow)
end
"""
    compute_pooling(::LinearOpinionPooling, ow::VecNum, pw::MatNum)
    compute_pooling(::LogarithmicOpinionPooling, ow::VecNum, pw::MatNum)

Compute the consensus posterior return distribution from individual prior distributions using opinion pooling.

`compute_pooling` aggregates multiple prior probability distributions (`pw`) into a single consensus posterior distribution according to the specified opinion pooling algorithm and opinion probabilities (`ow`). Supports both linear and logarithmic pooling.

# Mathematical definition

Let ``\\boldsymbol{\\alpha}`` be the opinion probabilities and ``\\mathbf{P}`` the ``T \\times K`` matrix of scenario weights for ``K`` experts:

Linear (weighted arithmetic mean):

```math
\\begin{align}
\\boldsymbol{p}^* &= \\mathbf{P} \\boldsymbol{\\alpha}\\,.
\\end{align}
```

Logarithmic (weighted geometric mean, normalised):

```math
\\begin{align}
p_t^* &= \\frac{\\exp\\!\\left(\\sum_{k=1}^{K} \\alpha_k \\log p_{tk}\\right)}{\\sum_{s=1}^{T} \\exp\\!\\left(\\sum_{k=1}^{K} \\alpha_k \\log p_{sk}\\right)}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{p}^*``: ``T \\times 1`` pooled posterior weight vector.
  - ``\\mathbf{P}``: ``T \\times K`` matrix of scenario weights for ``K`` experts.
  - ``\\boldsymbol{\\alpha}``: ``K \\times 1`` opinion probability vector (weights summing to 1).
  - ``p_{tk}``: Scenario weight for scenario ``t`` from expert ``k``.
  - $(math_dict[:T])

# Arguments

  - `alg`: Opinion pooling algorithm (`LinearOpinionPooling` or `LogarithmicOpinionPooling`).
  - `ow`: Vector of opinion probabilities (length = number of opinions).
  - `pw`: Matrix of prior weights for each opinion (observations × opinions).

# Returns

  - `w::StatsBase.ProbabilityWeights`: Consensus posterior probability weights.

# Details

  - For `LinearOpinionPooling`, computes the weighted arithmetic mean of the individual prior weights: `w = pw * ow`.
  - For `LogarithmicOpinionPooling`, computes the weighted geometric mean of the individual prior weights: `w = exp.(log.(pw) * ow - LogExpFunctions.logsumexp(log.(pw) * ow))`.
  - Used internally by [`OpinionPoolingPrior`](@ref) to form the consensus prior distribution.

# Related

  - [`OpinionPoolingPrior`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
"""
function compute_pooling(::LinearOpinionPooling, ow::VecNum, pw::MatNum)
    return StatsBase.pweights(pw * ow)
end
function compute_pooling(::LogarithmicOpinionPooling, ow::VecNum, pw::MatNum)
    u = log.(pw) * ow
    lse = LogExpFunctions.logsumexp(u)
    return StatsBase.pweights(vec(exp.(u .- lse)))
end
"""
    prior(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute opinion pooling prior moments for asset returns.

`prior` estimates the mean and covariance of asset returns by combining multiple prior estimations into a consensus prior using opinion pooling algorithms. Supports both linear and logarithmic pooling, robust opinion probability adjustment, and optional pre- and post-processing estimators.

# Arguments

  - `pe`: Opinion pooling prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `F`: Optional factor matrix.
  - $(arg_dict[:dims])
  - `strict`: If `true`, throws error for missing assets; otherwise, issues warnings. Default is `false`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators and solvers.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, posterior mean vector, posterior covariance matrix, consensus weights, entropy, Kullback-Leibler divergence, opinion probabilities, and optional factor moments.

# Details

  - Optional pre-processing estimator `pe.pe1` is applied to asset returns before pooling, else the original returns are used.
  - Each prior estimator in `pe.pes` is applied to the asset returns, producing individual prior weights.
  - Opinion probabilities `ow` are initialised from `pe.w` or set uniformly if it is `nothing`; if their sum is less than 1, the remainder is assigned to a uniform prior, which takes the last column of `pw` and is pooled and penalised like any other opinion.
  - Robust opinion probabilities are computed using [`robust_probabilities`](@ref) if a penalty parameter `pe.p` is not `nothing`. Neither `pe.w` nor any other field of `pe` is modified: calling `prior` twice on one estimator gives the same answer twice.
  - Consensus posterior weights are computed using [`compute_pooling`](@ref) according to the specified pooling algorithm `pe.alg`.
  - Post-processing estimator `pe.pe2` is applied using the consensus weights, via [`factory`](@ref). Every moment of the result is `pe.pe2`'s; the opinions contribute observation weights alone.
  - The result includes the effective number of scenarios, Kullback-Leibler divergence to each opinion, robust opinion probabilities, and optional factor moments.
  - `pr.ens` is `exp` of the entropy of the consensus weights, so it runs from 1 (all mass on one observation) to `T` (uniform).
  - `pr.kld[i]` is ``\\mathrm{KL}(\\boldsymbol{w} \\,\\|\\, \\boldsymbol{p}_i)``, from the **consensus to** opinion `i`. [`robust_probabilities`](@ref) reads the divergence in the other direction, so the two vectors are different numbers.

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
    X, F = dims_oriented(dims, X, F)
    X = !isnothing(pe.pe1) ? prior(pe.pe1, X, F; strict = strict, kwargs...).X : X
    T = size(X, 1)
    M = length(pe.pes)
    ow = isnothing(pe.w) ? range(inv(M), inv(M); length = M) : pe.w
    rw = one(eltype(ow)) - sum(ow)
    if rw > eps(typeof(rw))
        pw = Matrix{eltype(X)}(undef, T, M + 1)
        # `ow` may alias the struct's `pe.w`; build a new vector instead of `push!`ing
        # into it so repeated `prior` calls do not grow the stored weights (and so the
        # uniform-weight `range` branch, which is immutable, also works).
        ow = vcat(ow, rw)
        pw[:, end] .= inv(T)
    else
        pw = Matrix{eltype(X)}(undef, T, M)
    end
    let X = X, F = F, pw = pw
        FLoops.@floop pe.ex for (i, pe) in enumerate(pe.pes)
            pr = prior(pe, X, F; strict = strict, kwargs...)
            pw[:, i] = pr.w
        end
    end
    ow = robust_probabilities(ow, pw, pe.p)
    w = compute_pooling(pe.alg, ow, pw)
    pe2 = factory(pe.pe2, w)
    # Opinion pooling reweights observations without touching either axis of `Z`, so the
    # pooled prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole rather than stamped with the
    # pooled weights — see the note at the same seam in `12_EntropyPoolingPrior.jl`.
    (; X, o_X, mu, sigma, chol, rr, fpr, Z) = prior(pe2, X, F; strict = strict, kwargs...)
    ens = exp(StatsBase.entropy(w))
    kld = [StatsBase.kldivergence(w, view(pw, :, i)) for i in axes(pw, 2)]
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w,
                         ens = ens, kld = kld, ow = ow, rr = rr, fpr = fpr, Z = Z)
end

function factor_residual_config(pe::OpinionPoolingPrior)
    # The pooled `pe.pes` contribute observation weights alone; every moment of the result
    # comes from the refit `pe.pe2`, so the residual block is `pe.pe2`'s and this estimator
    # forwards it (see [`factor_residual_config`](@ref)).
    return factor_residual_config(pe.pe2)
end

export LinearOpinionPooling, LogarithmicOpinionPooling, OpinionPoolingPrior
