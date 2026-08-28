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

Each scenario's consensus weight is the opinion-weighted average of what the experts assign to it, so the pooled distribution keeps every scenario any one expert believes in. It suits opinions that are independent and additive.

# Mathematical definition

```math
\\begin{align}
p_t^{*} &= \\sum_{k=1}^{K} \\alpha_k\\, p_{tk}\\,.
\\end{align}
```

Where:

  - ``p_t^{*}``: Pooled weight of scenario ``t``.
  - ``\\alpha_k``: Opinion probability of expert ``k``.
  - ``p_{tk}``: Scenario weight for scenario ``t`` from expert ``k``.
  - ``K``: Number of opinions.

A sum of non-negative terms vanishes only when every term does, so ``p_t^{*} = 0`` requires that **every** opinion assigns scenario ``t`` zero probability. The pooled distribution is a mixture of the opinions, so it is at least as dispersed as the most dispersed one.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LogarithmicOpinionPooling`](@ref): the sibling rule, under which one zero opinion is enough to zero a scenario.
  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref)

# References

  - $(ref_dict[:dietrichlist2017])
"""
struct LinearOpinionPooling <: OpinionPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Pools the opinions as a weighted geometric mean of their scenario weights, renormalised.

The result is the distribution that minimises the opinion-weighted Kullback-Leibler divergence to the individual opinions, which makes it the information-theoretic consensus. It is robust to extremes, because it down-weights a scenario that any one opinion doubts.

# Mathematical definition

```math
\\begin{align}
p_t^{*} &= \\frac{\\exp\\!\\left(\\sum_{k=1}^{K} \\alpha_k \\log p_{tk}\\right)}{\\sum_{s=1}^{T} \\exp\\!\\left(\\sum_{k=1}^{K} \\alpha_k \\log p_{sk}\\right)}\\,.
\\end{align}
```

Where:

  - ``p_t^{*}``: Pooled weight of scenario ``t``.
  - ``\\alpha_k``: Opinion probability of expert ``k``.
  - ``p_{tk}``: Scenario weight for scenario ``t`` from expert ``k``.
  - ``K``: Number of opinions.
  - $(math_dict[:T])

A single ``\\log 0`` sends the exponent to ``-\\infty``, so ``p_t^{*} = 0`` as soon as **one** opinion assigns scenario ``t`` zero probability. The product ``\\alpha_k \\log p_{tk}`` is undefined when both factors vanish, which is the one case the form above does not cover.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LinearOpinionPooling`](@ref): the sibling rule, under which every opinion must agree before a scenario reaches zero.
  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref): its `# Validation` section states what the code does at the undefined product.

# References

  - $(ref_dict[:good1952])
  - $(ref_dict[:dietrichlist2017])
"""
struct LogarithmicOpinionPooling <: OpinionPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Opinion pooling prior estimator for asset returns.

`OpinionPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns by combining multiple prior estimations into a consensus prior using opinion pooling algorithms. It supports both linear and logarithmic pooling, flexible weighting of opinions, and optional pre- and post-processing estimators.

The opinions contribute **observation weights** alone. Every moment of the result comes from refitting `pe2` under the pooled weights, which is why `pes` is typed to the entropy-pooling estimators — they are the ones whose result carries a `w`. A `w` of `nothing` weights every opinion equally at `1/length(pes)`, and a `p` of `nothing` uses the opinion probabilities as given rather than adjusting them through [`robust_probabilities`](@ref).

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
  - The last is an inequality on purpose. When `sum(w) < 1`, [`prior`](@ref) gives the remaining weight to a uniform prior over the observations, which becomes an opinion in its own right: it takes a column of `pw`, and it is pooled and penalised alongside the others.
  - If `p` is not `nothing`, `p > 0`. The bound is strict, so `p = 0` raises; `p = nothing` is how one asks for no penalty.

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
            @argcheck(all(x -> zero(x) <= x <= one(x), w),
                      DomainError(w, "every entry of w must be in [0, 1]"))
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

``\\exp(-\\rho D_k)`` decreases in ``D_k``, so a larger ``\\rho`` concentrates the mass on the opinions nearest the consensus. As ``\\rho`` grows without bound the pool tends to the single opinion of smallest divergence.

# Algorithm

The steps below are those of the three-argument method. The `args...` method takes no step and returns `ow`.

 1. Pool the opinions linearly, giving the consensus `c`.
 2. Read the Kullback-Leibler divergence of each column of `pw` against `c`, giving `kldivs`.
 3. Scale each entry of `ow` by `exp(-p * kldivs)`, into a **new** vector.
 4. Divide that vector by its sum, giving the penalised opinion probabilities.

# Arguments

  - `ow`: Vector of opinion probabilities (length = number of opinions).
  - `pw`: Matrix of prior weights for each opinion (observations × opinions).
  - `p`: Robustness penalty parameter.

# Returns

  - `ow::VecNum`: The opinion probabilities for pooling, summing to 1. The three-argument method returns a **new** vector, and the `args...` method returns the argument itself. The argument is never modified either way, because it may be the estimator's own `w` field, or the immutable `range` that the uniform-weight branch of [`prior`](@ref) builds.

# Related

  - [`OpinionPoolingPrior`](@ref): the estimator that calls this to make its aggregation of opinions robust.
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

# Algorithm

Under [`LinearOpinionPooling`](@ref):

 1. Multiply `pw` by `ow`, giving the consensus weights `w`.

Under [`LogarithmicOpinionPooling`](@ref):

 1. Multiply the elementwise logarithm of `pw` by `ow`, giving the exponent vector `u`.
 2. Read `LogExpFunctions.logsumexp(u)` into `lse`. This shifts `u` by its own maximum before exponentiating it, so a very negative exponent does not underflow to a vector of zeros.
 3. Exponentiate `u .- lse`, giving the consensus weights `w`.

# Arguments

  - `alg`: Opinion pooling algorithm (`LinearOpinionPooling` or `LogarithmicOpinionPooling`).
  - `ow`: Vector of opinion probabilities (length = number of opinions).
  - `pw`: Matrix of prior weights for each opinion (observations × opinions).

# Validation

  - The result carries no `Inf` and no `NaN`, checked by `StatsBase.pweights`. Under [`LogarithmicOpinionPooling`](@ref) an opinion probability of exactly `0` against a scenario weight of exactly `0` makes the product `0 * log(0)`, which is `NaN`, and the call raises `ArgumentError`. A zero scenario weight alone is safe: it gives `-Inf`, which exponentiates to a consensus weight of `0`.

# Returns

  - `w::StatsBase.ProbabilityWeights`: Consensus posterior probability weights.

# Related

  - [`OpinionPoolingPrior`](@ref): the estimator that calls this to form its consensus prior distribution.
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

No field of `pe` is modified, so calling `prior` twice on one estimator gives the same answer twice. Every moment of the result is `pe.pe2`'s; the opinions contribute observation weights alone.

# Algorithm

 1. Orient `X` and `F` by `dims`.
 2. When `pe.pe1` is not `nothing`, replace `X` with the returns of that estimator's prior.
 3. Read the opinion probabilities `ow`, from `pe.w` when it is set and from a uniform `range` over `length(pe.pes)` when it is `nothing`.
 4. Take the remainder `rw` of `ow` against one. When `rw` exceeds `eps`, append it to `ow` and give `pw` a last column of `1/T`, the uniform prior over the observations.
 5. Fit every estimator of `pe.pes` over the executor `pe.ex`, writing each result's weights into a column of `pw`.
 6. Penalise `ow` through [`robust_probabilities`](@ref), which is the identity when `pe.p` is `nothing`.
 7. Pool the columns of `pw` under `pe.alg` through [`compute_pooling`](@ref), giving the consensus weights `w`.
 8. Refit `pe.pe2` under `w` through [`factory`](@ref), giving the moments of the result.
 9. Read `ens` as `exp` of the entropy of `w`, and `kld` as the divergence from `w` to each column of `pw`.

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
  - `pr.ens`: `exp` of the entropy of the consensus weights, so it runs from 1 (all mass on one observation) to `T` (uniform).
  - `pr.kld[i]`: ``\\mathrm{KL}(\\boldsymbol{w} \\,\\|\\, \\boldsymbol{p}_i)``, from the **consensus to** opinion `i`. [`robust_probabilities`](@ref) reads the divergence in the other direction, so the two vectors are different numbers.
  - `pr.ow`: The penalised opinion probabilities, one entry per column of `pw`. It is one entry longer than `pe.w` when the uniform-prior remainder took a column.

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
