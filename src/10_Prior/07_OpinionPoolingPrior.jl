"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for opinion pooling algorithms.

A subtype of `OpinionPoolingAlgorithm` is the rule by which an [`OpinionPoolingPrior`](@ref) pools the scenario weights of its opinions into one consensus weight vector. [`compute_pooling`](@ref) dispatches on it.

# Interfaces

To add a pooling rule, subtype `OpinionPoolingAlgorithm` and implement the following method:

## Required method name

  - `compute_pooling(alg::OpinionPoolingAlgorithm, ow::VecNum, pw::MatNum) -> StatsBase.AnalyticWeights`: Aggregate the columns of `pw` into one consensus scenario-weight vector.

### Arguments

  - `alg`: The concrete subtype instance.
  - `ow`: ``K \\times 1`` vector of opinion probabilities, summing to 1.
  - `pw`: ``T \\times K`` matrix whose column `k` holds expert `k`'s scenario weights.

### Returns

  - `w::StatsBase.AnalyticWeights`: ``T \\times 1`` consensus scenario weights, summing to 1. The type of the weights sets the bias correction of the covariance of the refit, see [`ScenarioProbabilities`](@ref).

### Examples

```jldoctest
julia> struct MedianOpinionPooling <: PortfolioOptimisers.OpinionPoolingAlgorithm end

julia> function PortfolioOptimisers.compute_pooling(::MedianOpinionPooling, ow, pw)
           w = vec(mapslices(PortfolioOptimisers.Statistics.median, pw; dims = 2))
           return PortfolioOptimisers.StatsBase.aweights(w / sum(w))
       end

julia> PortfolioOptimisers.compute_pooling(MedianOpinionPooling(), [0.5, 0.5],
                                           [0.5 0.25; 0.25 0.25; 0.25 0.5])
3-element AnalyticWeights{Float64, Float64, Vector{Float64}}:
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

  - $(ref_dict[:dietrichlist2017]) Section 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:dietrichlist2016]) Section 6. $(ref_dict[:no_original_source])
  - $(ref_dict[:martinisprenger2017]) Section 3, Theorem 2.
"""
abstract type OpinionPoolingAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Pools the opinions as a weighted arithmetic mean of their scenario weights.

The consensus weight of a scenario is the opinion-weighted average of the weights that the experts give it. So the pool keeps every scenario to which one expert gives a positive weight. The rule suits opinions that are independent and additive.

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

A sum of non-negative terms is zero only when every term is zero. So ``p_t^{*} = 0`` only when **every** opinion of positive probability gives scenario ``t`` a zero weight. The pool is a mixture of the opinions, and the entropy is concave, so the entropy of the pool is at least the opinion-weighted mean of the entropies of the opinions. It can be less than the entropy of the most dispersed opinion.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LogarithmicOpinionPooling`](@ref): the sibling rule, under which one opinion that gives a scenario a zero weight sets its pooled weight to zero.
  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref)

# References

  - $(ref_dict[:dietrichlist2017]) Section 2. $(ref_dict[:no_original_source])
"""
struct LinearOpinionPooling <: OpinionPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Pools the opinions as a weighted geometric mean of their scenario weights, renormalised.

The result is the distribution ``\\boldsymbol{q}`` that minimises ``\\sum_{k} \\alpha_k \\mathrm{KL}(\\boldsymbol{q} \\,\\|\\, \\boldsymbol{p}_k)``, the opinion-weighted Kullback-Leibler divergence from the pool to each opinion. The first-order condition of that problem gives the formula below. One opinion that gives a scenario a small weight makes the pooled weight of that scenario small, whatever weight the other opinions give it.

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

One ``\\log 0`` makes the exponent ``-\\infty``. So ``p_t^{*} = 0`` when **one** opinion of positive probability gives scenario ``t`` a zero weight. The product ``\\alpha_k \\log p_{tk}`` has no value when both factors are zero, and the form above does not cover that case. The source defines the pool only for opinions that give every scenario a positive weight. The library also accepts a zero weight, with the result above.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LinearOpinionPooling`](@ref): the sibling rule, under which the pooled weight of a scenario is zero only when every opinion gives it a zero weight.
  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref): its `# Validation` section states what the code does when both factors of the product are zero.

# References

  - $(ref_dict[:dietrichlist2016]) Section 6. $(ref_dict[:no_original_source])
  - $(ref_dict[:good1952])
"""
struct LogarithmicOpinionPooling <: OpinionPoolingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Opinion pooling prior estimator for asset returns.

`OpinionPoolingPrior` is a low order prior estimator. Each estimator in `pes` is one opinion, and each opinion gives a vector of observation weights. The estimator pools these vectors into one consensus vector under `alg`, and fits `pe2` under it.

The opinions contribute **observation weights** alone. Every moment of the result comes from the refit of `pe2` under the pooled weights. So `pes` holds entropy pooling estimators, because their results carry a `w`. A `w` of `nothing` gives every opinion the probability `1/length(pes)`. A `p` of `nothing` pools the opinion probabilities as given, and [`robust_probabilities`](@ref) does not penalise them.

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
        ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx(),
        cache::Option{<:AbstractPartialFitState} = nothing,
        x_src::Option{<:AbstractReturnsSource} = nothing
    ) -> OpinionPoolingPrior

Keywords correspond to the struct's fields. All arguments are validated for type and value consistency.

## Validation

  - `pes` must be a non-empty vector of prior estimators.
  - If `w` is not `nothing`, `!isempty(w)`, `length(w) == length(pes)`, `all(x -> 0 <= x <= 1, w)`, and `sum(w) <= 1`.
  - The last is an inequality on purpose. When `sum(w) < 1`, [`prior`](@ref) gives the remaining weight to a uniform prior over the observations. That prior is one more opinion. It takes a column of `pw`, and the pool penalises and pools it with the other opinions.
  - If `p` is not `nothing`, `p > 0`. The bound is strict, so `p = 0` raises an error. Set `p = nothing` for no penalty.
  - `x_src` is `nothing`, `DataSource()` or `ExcessSource()`, see [`assert_returns_source`](@ref).
  - The `x_src` of `pes`, `pe1`, `pe2` is `nothing`, because the prior picks the returns that it reads, see [`assert_inner_reader`](@ref).

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pes`: Recursively updated via [`factory`](@ref).
  - `pe1`: Recursively updated via [`factory`](@ref).
  - `pe2`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `pes`: Recursively viewed via [`port_opt_view`](@ref).
  - `pe1`: Recursively viewed via [`port_opt_view`](@ref).
  - `pe2`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

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
        │           ce ┼ PortfolioOptimisersCovariance
        │              │   ce ┼ Covariance
        │              │      │    me ┼ SimpleExpectedReturns
        │              │      │       │   w ┴ nothing
        │              │      │    ce ┼ GeneralCovariance
        │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │              │      │       │    w ┴ nothing
        │              │      │   alg ┼ FullMoment()
        │              │      │     w ┴ nothing
        │              │   mp ┼ MatrixProcessing
        │              │      │     pdm ┼ Posdef
        │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │              │      │      dn ┼ nothing
        │              │      │      dt ┼ nothing
        │              │      │     alg ┼ nothing
        │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │           me ┼ SimpleExpectedReturns
        │              │   w ┴ nothing
        │      horizon ┼ nothing
        │   fill_limit ┼ nothing
        │        x_src ┴ nothing
      p ┼ nothing
      w ┼ nothing
    alg ┼ LinearOpinionPooling()
     ex ┼ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
  x_src ┴ nothing
```

## The incremental fit

This prior has no exact incremental recursion, so it takes the online step by **refitting from a sample buffer**: [`Online`](@ref) seeds `cache`, [`partial_fit!`](@ref) appends each observation to it verbatim, and the one-argument [`prior`](@ref) runs this estimator's own batch verb over the rows the buffer kept. The answer is therefore exactly a batch fit over those rows, and a `max_history` on the wrapper windows the whole fit.

`cache` travels the three propagation channels as every partial-fit state does: [`factory`](@ref) carries it unchanged, [`port_opt_view`](@ref) slices it to the selected assets, and [`obs_weights_view`](@ref) drops it, because no slice of a state exists on the observation axis. It is not rendered, because a running buffer is not the configuration a reader looks the type up for.

# Related

  - [`OpinionPoolingAlgorithm`](@ref)
  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
  - [`prior`](@ref)
  - [`robust_probabilities`](@ref)
  - [`compute_pooling`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)

# References

  - $(ref_dict[:dietrichlist2017]) Section 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:dietrichlist2016]) Section 6. $(ref_dict[:no_original_source])
  - $(ref_dict[:martinisprenger2017]) Section 3, Theorem 2.
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
    $(field_dict[:op_w_pe])
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
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    """
    $(field_dict[:x_src_pe])
    """
    x_src
    function OpinionPoolingPrior(pes::VecEP,
                                 pe1::Option{<:AbstractLowOrderPriorEstimator_A_F_AF},
                                 pe2::AbstractLowOrderPriorEstimator_A_F_AF,
                                 p::Option{<:Number}, w::Option{<:VecNum},
                                 alg::OpinionPoolingAlgorithm,
                                 ex::FLoops.Transducers.Executor,
                                 cache::Option{<:AbstractPartialFitState},
                                 x_src::Option{<:AbstractReturnsSource})
        assert_returns_source(x_src, DATA_RETURNS_SOURCES, :OpinionPoolingPrior)
        for p in pes
            assert_inner_reader(p, :pes, :OpinionPoolingPrior)
        end
        assert_inner_reader(pe1, :pe1, :OpinionPoolingPrior)
        assert_inner_reader(pe2, :pe2, :OpinionPoolingPrior)
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
                      DomainError(sum(w),
                                  "`OpinionPoolingPrior.w` sums to $(sum(w)), and the weights of the pooled priors leave the remainder to the uniform prior, so they sum to at most one. State weights with `sum(w) <= 1`."))
        end
        return new{typeof(pes), typeof(pe1), typeof(pe2), typeof(p), typeof(w), typeof(alg),
                   typeof(ex), typeof(cache), typeof(x_src)}(pes, pe1, pe2, p, w, alg, ex,
                                                             cache, x_src)
    end
end
function OpinionPoolingPrior(; pes::VecEP,
                             pe1::Option{<:AbstractLowOrderPriorEstimator_A_F_AF} = nothing,
                             pe2::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             p::Option{<:Number} = nothing, w::Option{<:VecNum} = nothing,
                             alg::OpinionPoolingAlgorithm = LinearOpinionPooling(),
                             ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx(),
                             cache::Option{<:AbstractPartialFitState} = nothing,
                             x_src::Option{<:AbstractReturnsSource} = nothing)::OpinionPoolingPrior
    return OpinionPoolingPrior(pes, pe1, pe2, p, w, alg, ex, cache, x_src)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`OpinionPoolingPrior`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:OpinionPoolingPrior, true)` to render it.

# Arguments

  - `::OpinionPoolingPrior`: Prior estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:pes, :pe1, :pe2, :p, :w, :alg, :ex, :x_src)`.

# Related

  - [`OpinionPoolingPrior`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::OpinionPoolingPrior) = (:pes, :pe1, :pe2, :p, :w, :alg, :ex, :x_src)
"""
    robust_probabilities(ow::VecNum, args...)
    robust_probabilities(ow::VecNum, pw::MatNum, p::Number)

Penalise each opinion probability by the divergence of its opinion from the consensus.

The three-argument method multiplies the probability of each opinion by ``\\exp(-\\rho D_k)``, where ``D_k`` is the Kullback-Leibler divergence of the opinion from the linear pool, and renormalises. An opinion far from the consensus loses probability. When `p` is `nothing`, the `args...` method returns `ow` as it is.

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

The consensus ``\\boldsymbol{c}`` is always the **linear** pool, whatever [`OpinionPoolingAlgorithm`](@ref) the caller selected. The divergence has a direction. It is ``D_k = \\mathrm{KL}(\\boldsymbol{p}_k \\,\\|\\, \\boldsymbol{c})``, from each opinion to the consensus. The `kld` field of the result of [`prior`](@ref) has the other direction, from the consensus to each opinion, so the two vectors hold different numbers.

``\\exp(-\\rho D_k)`` decreases in ``D_k``, so a larger ``\\rho`` moves the mass to the opinions nearest the consensus. As ``\\rho`` increases without bound, all the mass goes to the opinion of smallest divergence.

# Algorithm

The steps below are those of the three-argument method. The `args...` method takes no step and returns `ow`.

 1. Pool the opinions linearly into the consensus `c`.
 2. Compute `kldivs`, the Kullback-Leibler divergence of each column of `pw` from `c`.
 3. Multiply each entry of `ow` by `exp(-p * kldivs)`, into a **new** vector.
 4. Divide that vector by its sum, and return it.

# Arguments

  - `ow`: Opinion probabilities, one entry per opinion.
  - `pw`: Scenario weights of the opinions, `observations × opinions`, one column per opinion.
  - `p`: Penalty, above zero.

# Returns

  - `ow::VecNum`: The opinion probabilities to pool with, summing to 1. The three-argument method returns a **new** vector, and the `args...` method returns the argument itself. Neither method writes to the argument, because it can be the estimator's own `w` field, or the immutable `range` that [`opinion_pooling_weights`](@ref) makes when `w` is `nothing`.

# Related

  - [`OpinionPoolingPrior`](@ref): the estimator that calls this function, through [`opinion_pooling_weights`](@ref).
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

Pool the scenario weights of several opinions into one consensus weight vector.

Each column of `pw` holds the scenario weights of one opinion, and `ow` holds the probability of each opinion. The type of the first argument selects the pooling rule.

# Mathematical definition

Let ``\\boldsymbol{\\alpha}`` be the opinion probabilities and ``\\mathbf{P}`` the ``T \\times K`` matrix of scenario weights for ``K`` experts.

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
  - ``\\boldsymbol{\\alpha}``: ``K \\times 1`` opinion probability vector, summing to 1.
  - ``p_{tk}``: Scenario weight for scenario ``t`` from expert ``k``.
  - $(math_dict[:T])

# Algorithm

Under [`LinearOpinionPooling`](@ref):

 1. Multiply `pw` by `ow` to get the consensus weights `w`.

Under [`LogarithmicOpinionPooling`](@ref):

 1. Multiply the elementwise logarithm of `pw` by `ow` to get the exponent vector `u`.
 2. Compute `lse = LogExpFunctions.logsumexp(u)`. The function subtracts the maximum of `u` before it exponentiates, so a very negative exponent does not underflow to a vector of zeros.
 3. Exponentiate `u .- lse` to get the consensus weights `w`.

# Arguments

  - `alg`: Opinion pooling algorithm, [`LinearOpinionPooling`](@ref) or [`LogarithmicOpinionPooling`](@ref).
  - `ow`: Opinion probabilities, one entry per opinion.
  - `pw`: Scenario weights of the opinions, `observations × opinions`, one column per opinion.

# Validation

  - `StatsBase.aweights` checks that the result carries no `Inf` and no `NaN`. Under [`LogarithmicOpinionPooling`](@ref), an opinion probability of exactly `0` against a scenario weight of exactly `0` makes the product `0 * log(0)`, which is `NaN`, and the call raises `ArgumentError`. A zero scenario weight alone is safe. It gives `-Inf`, which exponentiates to a consensus weight of `0`.

# Returns

  - `w::StatsBase.AnalyticWeights`: Consensus scenario weights. The logarithmic rule renormalises them to sum to 1, and the linear rule gives a sum of 1 when `ow` and every column of `pw` sum to 1. Their bias correction is the one of reliability weights, see [`ScenarioProbabilities`](@ref).

# Related

  - [`OpinionPoolingPrior`](@ref): the estimator that calls this function, through [`opinion_pooling_weights`](@ref).
  - [`LinearOpinionPooling`](@ref)
  - [`LogarithmicOpinionPooling`](@ref)
"""
function compute_pooling(::LinearOpinionPooling, ow::VecNum, pw::MatNum)
    return StatsBase.aweights(pw * ow)
end
function compute_pooling(::LogarithmicOpinionPooling, ow::VecNum, pw::MatNum)
    u = log.(pw) * ow
    lse = LogExpFunctions.logsumexp(u)
    return StatsBase.aweights(vec(exp.(u .- lse)))
end
"""
    prior(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing,
          pnl::Option{<:AssetPanel} = nothing;
          dims::Int = 1, strict::Bool = false, cache_reg::Bool = false, kwargs...)

Compute opinion pooling prior moments for asset returns.

`prior` fits every opinion of `pe.pes`, pools their observation weights into one consensus vector, and fits `pe.pe2` under that vector.

`prior` changes no field of `pe`, so two calls on one estimator give the same answer. Every moment of the result comes from `pe.pe2`. The opinions contribute observation weights alone.

# Algorithm

 1. Orient `X` and `F` by `dims`.
 2. When `pe.pe1` is not `nothing`, replace `X` with the returns of that estimator's prior.
 3. Pool the opinions of `pe.pes` over `X` with [`opinion_pooling_weights`](@ref), which returns the consensus weights `w`, the penalised opinion probabilities `ow` and the weights `pw` of each opinion.
 4. Refit `pe.pe2` under `w` through [`factory`](@ref). The moments of the result come from this refit. Check that the refit answered on the observation axis of the pool, see `# Validation`.
 5. Compute `ens` as `exp` of the entropy of `w`, and `kld` as the divergence from `w` to each column of `pw`.
 6. When `cache_reg` is `true`, store `pc = (; X, w, pe2)`. `X` holds the rows that the opinions read, or `nothing` when they read the `X` argument. `pe2` holds the cache of the result of `pe.pe2`. When `cache_reg` is `false`, `pc` is `nothing`.

The fit of `pe.pe2` receives `cache_reg` too, so a prior there stores its own cache. The refit of an entropy pooling prior around the pool reads both caches, and runs no opinion and no `pe.pe1`.

# Arguments

  - `pe`: Opinion pooling prior estimator.
  - `X`: Asset returns matrix, `observations × assets`.
  - `F`: Optional factor returns matrix.
  - $(arg_dict[:pnl_prior]) It goes to `pe.pe1`, to every opinion in `pe.pes`, and to the refit `pe.pe2`.
  - $(arg_dict[:dims])
  - `strict`: Passed to `pe.pe1`, to every opinion and to `pe.pe2`. If `true`, an opinion with a view that names an asset that its `sets` does not hold raises an error. If `false`, the opinion drops that view with a warning. Default is `false`.
  - $(arg_dict[:cache_reg])
  - `kwargs...`: More keyword arguments, passed to `pe.pe1`, to every opinion and to `pe.pe2`.

# Validation

  - `dims in (1, 2)`.
  - Every opinion answers on the observation axis of the pool. That axis is the rows of the result of `pe.pe1` when `pe.pe1` is set, and `X` when it is `nothing`. An opinion whose own wrapped estimator drops rows answers on a shorter axis, so its probabilities are over other scenarios. [`opinion_pooling_weights`](@ref) then raises a `DimensionMismatch`. The message names the opinion, both counts and the fix, which is to move the estimator that drops rows into `pe1`.
  - The refit `pe.pe2` answers on the observation axis of the pool too, because the pooled weights are over that axis. A `pe2` that drops rows, such as one that holds a Scenario Cap, answers on a shorter axis. `prior` then raises a `DimensionMismatch` that names `pe2`, both counts and the same fix.

# Returns

  - `pr::LowOrderPrior`: The moments of the refit `pe.pe2`, with the consensus weights `w`, their diagnostics `ens`, `kld` and `ow`, and the cache `pc`.
  - `pr.ens`: `exp` of the entropy of the consensus weights. It is 1 when all the mass is on one observation, and `T` when the weights are uniform.
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
function prior(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               cache_reg::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    X1 = if isnothing(pe.pe1)
        nothing
    else
        prior(pe.pe1, X, F, pnl; strict = strict, kwargs...).X
    end
    X = something(X1, X)
    w, ow, pw = opinion_pooling_weights(pe, X, F, pnl; strict = strict, kwargs...)
    pe2 = factory(pe.pe2, w)
    # Opinion pooling reweights observations without touching either axis of `Z`, so the
    # pooled prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole rather than stamped with the
    # pooled weights — see the note at the same step in `03_EntropyPoolingPrior_c.jl`.
    pr2 = prior(pe2, X, F, pnl; strict = strict, kwargs..., cache_reg = cache_reg)
    # The refit of an entropy pooling prior around the pool reads the rows of the opinions
    # and the pooled weights from the cache, so no opinion runs again (#1751). The cache of
    # `pe2` rides inside it, because a `pe2` that is a pool reads its own pooled weights.
    pc = ifelse(cache_reg, (; X = X1, w = w, pe2 = opinion_pooling_cache(pr2)), nothing)
    # The pooled weights are over the pool's own observation axis, so the refit must answer
    # on it, as every opinion must. A `pe2` that drops rows, such as one that holds a
    # Scenario Cap, answers on a shorter axis, and the weights then name no row of its
    # result. See ADR 0116.
    @argcheck(size(pr2.X, 1) == length(w),
              DimensionMismatch("`pe2`, a $(nameof(typeof(pe.pe2))), answered on $(size(pr2.X, 1)) observations and the pool has $(length(w)). The pool refits `pe2` under the pooled weights of its observations, so `pe2` must answer on the pool's axis. Move the estimator that drops rows, such as a Scenario Cap, into `pe1`, where it makes the scenario set for every opinion and for `pe2`."))
    (; X, o_X, mu, sigma, chol, rr, fpr, ss_gaps) = pr2
    ens = exp(StatsBase.entropy(w))
    kld = [StatsBase.kldivergence(w, view(pw, :, i)) for i in axes(pw, 2)]
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w,
                         ens = ens, kld = kld, ow = ow, rr = rr, fpr = fpr,
                         ss_gaps = ss_gaps, pc = pc)
end
"""
    opinion_pooling_cache(pr::Option{<:AbstractPriorResult})
    opinion_pooling_cache(pr::LowOrderPrior)

Return the cache of the opinion pool that answered `pr`, or `nothing`.

A fit of an [`OpinionPoolingPrior`](@ref) with `cache_reg = true` stores the rows that its opinions read, its pooled weights and the cache of its `pe2` in `pr.pc`. Its refits through [`scenario_prior`](@ref) and [`scenario_factory`](@ref) read them here, so they run no opinion and no `pe1`. A result with no cache gives `nothing`, and the refit then pools the opinions again.

# Arguments

  - `pr`: The result of the fit that the views read, or `nothing`.

# Returns

  - `pc::Option{<:NamedTuple}`: `pr.pc` of a [`LowOrderPrior`](@ref), and `nothing` for no result or a result of any other type.

# Related

  - [`OpinionPoolingPrior`](@ref)
  - [`scenario_prior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function opinion_pooling_cache(::Option{<:AbstractPriorResult})::Nothing
    return nothing
end
function opinion_pooling_cache(pr::LowOrderPrior)
    return pr.pc
end
"""
    opinion_pooling_weights(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum},
                            pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)

Pools the opinions of an [`OpinionPoolingPrior`](@ref) into the consensus weights of the observations.

[`prior`](@ref) refits `pe.pe2` under these weights. The online step of the refit route reads them too, because [`pin_prior_choice`](@ref) pins the regression of `pe.pe2` under the weights of its first fit.

# Algorithm

 1. Read the opinion probabilities `ow`, from `pe.w` when it is set and from a uniform `range` over `length(pe.pes)` when it is `nothing`.
 2. Compute the remainder `rw = 1 - sum(ow)`. When `rw` is more than `eps`, append it to `ow`, and fill a last column of `pw` with `1/T`, the uniform prior over the observations.
 3. Fit every estimator of `pe.pes` with the executor `pe.ex`. Check that each one answered on the observation axis of the pool, and write its weights into a column of `pw`.
 4. Penalise `ow` through [`robust_probabilities`](@ref), which returns `ow` as it is when `pe.p` is `nothing`.
 5. Pool the columns of `pw` under `pe.alg` through [`compute_pooling`](@ref) into the consensus weights `w`.

# Arguments

  - `pe`: Opinion pooling prior estimator.
  - `X`: The returns that the opinions read, `observations × assets`. These are the returns of `pe.pe1` when it is set.
  - `F`: Optional factor returns matrix.
  - $(arg_dict[:pnl_prior])
  - `strict`: Passed to every opinion. If `true`, an opinion with a view that names an asset that its `sets` does not hold raises an error. If `false`, the opinion drops that view with a warning. Default is `false`.
  - `kwargs...`: More keyword arguments, passed to every opinion.

# Validation

  - Every opinion answers on the observation axis of `X`. If an opinion does not, the function raises a `DimensionMismatch` that names the opinion.

# Returns

  - `w::StatsBase.AnalyticWeights`: The consensus weights of the observations.
  - `ow::VecNum`: The penalised opinion probabilities, one entry per column of `pw`.
  - `pw::MatNum`: The weights of each opinion, one column per opinion.

# Related

  - [`OpinionPoolingPrior`](@ref)
  - [`compute_pooling`](@ref)
  - [`robust_probabilities`](@ref)
  - [`pin_prior_choice`](@ref)
"""
function opinion_pooling_weights(pe::OpinionPoolingPrior, X::MatNum, F::Option{<:MatNum},
                                 pnl::Option{<:AssetPanel}; strict::Bool = false, kwargs...)
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
    let X = X, F = F, pnl = pnl, pw = pw, T = T
        FLoops.@floop pe.ex for (i, pe) in enumerate(pe.pes)
            pr = prior(pe, X, F, pnl; strict = strict, kwargs...)
            # An opinion pool weights ONE scenario set, so every opinion must answer on the
            # pool's own observation axis. An opinion whose nested prior drops rows answers
            # on a shorter one, and its probabilities are then over other scenarios. That is
            # refused here rather than one line below, where the write states nothing. See
            # ADR 0116.
            @argcheck(length(pr.w) == T,
                      DimensionMismatch("opinion $i, a $(typeof(pe)), answered on $(length(pr.w)) observations and the pool has $T. An opinion pool weights one scenario set, so every opinion must answer on the pool's axis. Move the estimator that drops rows into `pe1`, where it makes that scenario set for every opinion."))
            pw[:, i] = pr.w
        end
    end
    ow = robust_probabilities(ow, pw, pe.p)
    w = compute_pooling(pe.alg, ow, pw)
    return w, ow, pw
end

function factor_residual_config(pe::OpinionPoolingPrior)
    # The pooled `pe.pes` contribute observation weights alone; every moment of the result
    # comes from the refit `pe.pe2`, so the residual block is `pe.pe2`'s and this estimator
    # forwards it (see [`factor_residual_config`](@ref)).
    return factor_residual_config(pe.pe2)
end
# The pooled weights are the prior probabilities of a host that reweights the scenarios, so
# its refit holds the rows of `pe1` and the model of `pe2` under the pooled weights, and
# reweights only the scenarios of `pe2` under the posterior weights (#1649). The cache of the
# fit gives the rows and the pooled weights, and otherwise the opinions run once more, for
# the pooled weights alone (#1751). The result keeps the cache, so a nested entropy pooling
# prior reads it too. `scenario_prior` documents every method.
function scenario_prior(pe::OpinionPoolingPrior, w, pr::Option{<:AbstractPriorResult},
                        X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                        kwargs...)
    pc = opinion_pooling_cache(pr)
    sc = opinion_pooling_scenarios(pe, pc, X, F, pnl; kwargs...)
    res = scenario_prior(factory(pe.pe2, sc.w), w, opinion_pooling_inner(pr, pc), sc.X, F,
                         pnl; kwargs...)
    return opinion_pooling_kept(res, pc)
end
# A pool that a factor prior or a Black-Litterman prior embeds is rebuilt as an estimator,
# not refitted, so it pools on the rows of the fit that the rebuild sees. Those are its own
# rows only when it holds no `pe1`, which `assert_scenario_model` checks (#1652). The cache
# of the fit gives the pooled weights, so no opinion runs again (#1751).
function scenario_factory(pe::OpinionPoolingPrior, w, pr::Option{<:AbstractPriorResult},
                          X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                          kwargs...)
    pc = opinion_pooling_cache(pr)
    wp = opinion_pooling_scenarios(pe, pc, X, F, pnl; kwargs...).w
    return scenario_factory(factory(pe.pe2, wp), w, opinion_pooling_inner(pr, pc), X, F,
                            pnl; kwargs...)
end
"""
    opinion_pooling_scenarios(pe::OpinionPoolingPrior, pc::Nothing, X::MatNum,
                              F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)
    opinion_pooling_scenarios(pe::OpinionPoolingPrior, pc::NamedTuple, X::MatNum,
                              F::Option{<:MatNum}, pnl::Option{<:AssetPanel}; kwargs...)

Return the rows that the opinions of a pool read and its pooled weights, for its refit under new scenario weights.

The refit of [`scenario_prior`](@ref) and [`scenario_factory`](@ref) fits `pe.pe2` under the pooled weights, on the rows of `pe.pe1` when it is set. A fit with `cache_reg = true` stores both in `pc`, so the refit reads them and runs no opinion and no `pe.pe1`.

# Algorithm

The method that Julia selects is the algorithm.

 1. When `pc` is `nothing`, take the rows of `pe.pe1` when it is set, and `X` when it is not, as the fit of the pool does. Pool the opinions on those rows with [`opinion_pooling_weights`](@ref).
 2. When `pc` is a cache, take `pc.X`, or `X` when `pc.X` is `nothing`, and take `pc.w`.

# Arguments

  - `pe`: Opinion pooling prior estimator.
  - `pc`: The cache of the fit, or `nothing`.
  - $(arg_dict[:X])
  - `F`: Optional factor returns matrix.
  - $(arg_dict[:pnl_prior])
  - `kwargs...`: More keyword arguments, passed to `pe.pe1` and to the opinions when no cache is given.

# Returns

  - `X::MatNum`: The rows that the opinions read.
  - `w::StatsBase.AbstractWeights`: The pooled weights.

# Related

  - [`opinion_pooling_cache`](@ref)
  - [`opinion_pooling_weights`](@ref)
  - [`opinion_pooling_kept`](@ref)
"""
function opinion_pooling_scenarios(pe::OpinionPoolingPrior, ::Nothing, X::MatNum,
                                   F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                   kwargs...)
    X = isnothing(pe.pe1) ? X : prior(pe.pe1, X, F, pnl; kwargs...).X
    return (; X = X, w = first(opinion_pooling_weights(pe, X, F, pnl; kwargs...)))
end
function opinion_pooling_scenarios(::OpinionPoolingPrior, pc::NamedTuple, X::MatNum,
                                   ::Option{<:MatNum}, ::Option{<:AssetPanel}; kwargs...)
    return (; X = something(pc.X, X), w = pc.w)
end
"""
    opinion_pooling_kept(res::AbstractPriorResult, pc::Option{<:NamedTuple})
    opinion_pooling_kept(res::LowOrderPrior, pc::NamedTuple)

Keep the cache of a pool on the result of its refit, so an entropy pooling prior around a nested one reads it too.

The refit returns the result of `pe2`, which carries no cache of the pool.

# Algorithm

The method that Julia selects is the algorithm.

 1. When there is no cache, or the type of the result has no field `pc`, return `res`.
 2. When `res` is a [`LowOrderPrior`](@ref) and there is a cache, forward `res` with `pc` through [`forward_prior`](@ref).

# Arguments

  - `res`: The result of the refit.
  - `pc`: The cache of the fit, or `nothing`.

# Returns

  - `res::AbstractPriorResult`: The result of the refit, with the cache where it can carry one.

# Related

  - [`opinion_pooling_scenarios`](@ref)
  - [`opinion_pooling_cache`](@ref)
"""
function opinion_pooling_kept(res::AbstractPriorResult, ::Option{<:NamedTuple})
    return res
end
function opinion_pooling_kept(res::LowOrderPrior, pc::NamedTuple)
    return forward_prior(res; pc = pc)
end
"""
    opinion_pooling_inner(pr::Option{<:AbstractPriorResult}, pc::Option{<:NamedTuple})
    opinion_pooling_inner(pr::LowOrderPrior, pc::NamedTuple)

Give the refit of `pe2` the result of the pool with the cache that the fit of `pe2` stored.

The result of a pool carries the fields of the result of `pe2`, and its own cache in `pc`. The refit of `pe2` reads its cache from that result. A `pe2` that is a pool, or an entropy pooling prior over one, would then read the pooled weights of the outer pool as its own. So the cache of the pool keeps the cache of `pe2` in `pc.pe2`, and this function puts it back in `pc` before the refit of `pe2`.

# Algorithm

The method that Julia selects is the algorithm.

 1. When there is no cache, or the type of the result has no field `pc`, return `pr`.
 2. When `pr` is a [`LowOrderPrior`](@ref) and there is a cache, forward `pr` with `pc.pe2` through [`forward_prior`](@ref).

# Arguments

  - `pr`: The result of the fit of the pool, or `nothing`.
  - `pc`: The cache of the pool, or `nothing`.

# Returns

  - `pr::Option{<:AbstractPriorResult}`: The result that the refit of `pe2` reads.

# Related

  - [`opinion_pooling_cache`](@ref)
  - [`opinion_pooling_kept`](@ref)
  - [`scenario_prior`](@ref)
"""
function opinion_pooling_inner(pr::Option{<:AbstractPriorResult}, ::Option{<:NamedTuple})
    return pr
end
function opinion_pooling_inner(pr::LowOrderPrior, pc::NamedTuple)
    return forward_prior(pr; pc = pc.pe2)
end
# A caller's prior probabilities weigh the scenarios of the pool, the rows of `pe1` when it
# is set. A Scenario Cap in `pe1` keeps the last rows, so `pe1` takes them on the rows that
# it reads, and the opinions and `pe2` take them on the scenarios, as `factory` gives them.
function prior_probability_factory(pe::OpinionPoolingPrior, w, X::MatNum,
                                   F::Option{<:MatNum}, pnl::Option{<:AssetPanel};
                                   kwargs...)
    pool = factory(pe, w)
    pe1 = prior_probability_factory(pe.pe1, w, X, F, pnl; kwargs...)
    return rebuild_estimator(pool, (; pe1 = pe1))
end
function prior_probability_factory(pe::OpinionPoolingPrior, w, n::Integer)
    pool = factory(pe, w)
    return rebuild_estimator(pool, (; pe1 = prior_probability_factory(pe.pe1, w, n)))
end
# A pool with no first prior has nothing below it to reweight.
function prior_probability_factory(pe::OpinionPoolingPrior{<:Any, Nothing}, w, ::MatNum,
                                   ::Option{<:MatNum}, ::Option{<:AssetPanel}; kwargs...)
    return factory(pe, w)
end
function prior_probability_factory(pe::OpinionPoolingPrior{<:Any, Nothing}, w, ::Integer)
    return factory(pe, w)
end
function assert_scenario_model(pe::OpinionPoolingPrior)
    @argcheck(isnothing(pe.pe1),
              ArgumentError("An entropy pooling prior refits the opinion pool inside a factor prior or a Black-Litterman prior under its posterior weights, on the rows that the wrapping prior gives the pool. A pool with `pe1` reads the rows of `pe1` instead, so the refit cannot state its model. Set `pe1 = nothing` in this pool, or give the pool to the entropy pooling prior directly as its `pe`."))
    return assert_scenario_model(pe.pe2)
end
function assert_scenario_refit(pe::OpinionPoolingPrior)
    return assert_scenario_refit(pe.pe2)
end

export LinearOpinionPooling, LogarithmicOpinionPooling, OpinionPoolingPrior
public OpinionPoolingAlgorithm, compute_pooling
