"""
$(DocStringExtensions.TYPEDEF)

Fits an uncertainty set by resampling the return series, so the set assumes no law for the returns.

All concrete subtypes should subtype `BootstrapUncertaintySetEstimator`. It is the branch of [`AbstractUncertaintySetEstimator`](@ref) whose bounds come from a resample rather than from a closed form.

# Interfaces

A subtype implements the three methods of [`AbstractUncertaintySetEstimator`](@ref), and carries a [`ARCHBootstrapSet`](@ref) that says how the resample is drawn.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`ARCHBootstrapSet`](@ref)
"""
abstract type BootstrapUncertaintySetEstimator <: AbstractUncertaintySetEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Selects how a block bootstrap draws its blocks, which is what keeps the serial dependence of a return series in the resample.

All concrete subtypes should subtype `ARCHBootstrapSet`. The three that ship differ on three axes at once, and a caller chooses between them on those three axes alone.

| Scheme                        | Block length                                    | Wraps past the end | Start range              |
|:----------------------------- |:----------------------------------------------- |:------------------ |:------------------------ |
| [`StationaryBootstrap`](@ref) | geometric, restart probability `1 / block_size` | yes, by `mod1`     | `1:T`                    |
| [`CircularBootstrap`](@ref)   | fixed `block_size`                              | yes, by `mod1`     | `1:T`                    |
| [`MovingBootstrap`](@ref)     | fixed `block_size`                              | no                 | `1:(T - block_size + 1)` |

The wrapping column is the one that decides whether every observation is drawn equally often. The two schemes that wrap draw each of the `T` observations equally often; the one that does not draw the first and the last observations of the series less often than the middle, in the ramp [`MovingBootstrap`](@ref) states.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `ARCHBootstrapSet` and implement the following method:

## `bootstrap_indices`

  - `bootstrap_indices(alg::ARCHBootstrapSet, rng::Random.AbstractRNG, T::Integer, block_size::Integer) -> Vector{Int}`: Returns the row indices of one resample.

### Arguments

  - `alg`: The concrete subtype instance.
  - `rng`: Random number generator.
  - `T`: Number of observations in the sample being resampled.
  - `block_size`: Block length, or the mean block length when the length is random.

### Returns

  - `idx::Vector{Int}`: `T` indices in `1:T`, which select the rows of one resample.

# Related

  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
  - [`bootstrap_indices`](@ref)
"""
abstract type ARCHBootstrapSet <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Draws blocks of geometrically distributed random length, so the resample is itself a stationary series.

A block starts anywhere in `1:T`, continues to the next index by `mod1` so that it wraps past the end of the series, and restarts with probability `1 / block_size` at each index. The block length is therefore geometric with mean `block_size`: over 101170 blocks at `block_size = 10` the measured mean length is 9.884, and the measured length frequencies `[0.101, 0.0921, 0.082, 0.0733, 0.0671, 0.0584]` for lengths `1:6` match the geometric masses `[0.1, 0.09, 0.081, 0.0729, 0.0656, 0.059]`. The mean falls a little below `block_size` because the last block of an index vector is cut short at `T`.

Because a block wraps, every observation is drawn equally often. `block_size` is a mean and not a bound, so it may exceed `T` without a raise: the restart probability is then below `1 / T` and the scheme approaches a single wrapped block. The spread does not collapse, but it narrows — over 250 resamples of a 252-by-5 sample the standard deviation of the bootstrap means at `block_size = T + 1` is 0.377 of the value at `block_size = 3`.

# Related

  - [`ARCHBootstrapSet`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)

# References

  - $(ref_dict[:politis1994stationary])
"""
struct StationaryBootstrap <: ARCHBootstrapSet end
"""
$(DocStringExtensions.TYPEDEF)

Draws blocks of fixed length `block_size` that wrap past the end of the series, so every observation is drawn equally often.

A block starts anywhere in `1:T` and runs `block_size` indices forward, each taken by `mod1`, so the series is read as a circle. The wrap is what buys the equal coverage: over 20000 index vectors at `T = 100` and `block_size = 5` every observation is drawn between 0.985 and 1.015 of the average, against a ramp down to 0.192 under [`MovingBootstrap`](@ref).

**A `block_size` of `T` or more collapses the set to a point.** The first block already fills the whole index vector, so every resample is a cyclic shift of the series, and a cyclic shift is a permutation. The mean and the covariance do not change under a permutation of the rows, so every resample returns the same statistics. Measured over 50 resamples of a 252-by-5 sample at `block_size = T + 1`, the spread of the bootstrap means is 1.3e-18 and the box width is 1.7e-18. Nothing raises: the ellipsoidal route builds a shape matrix of order 1e-37 and a finite radius, which is an empty set rather than an error.

# Related

  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)

# References

  - $(ref_dict[:politis1992circular])
"""
struct CircularBootstrap <: ARCHBootstrapSet end
"""
$(DocStringExtensions.TYPEDEF)

Draws blocks of fixed length `block_size` that never wrap, so a resample holds no join between the end of the series and its start.

A block starts in `1:(T - block_size + 1)` and runs `block_size` indices forward, so the last index of a block never passes `T`. This is the one scheme of the three that guards `block_size`, and it is the one that needs a guard: the start range is empty as soon as `block_size` exceeds `T`, and the raise turns an `ArgumentError` about an empty range into a `DomainError` that names `block_size` and `T`. The two schemes that wrap take every index through `mod1`, which cannot leave `1:T`, so neither can build an out-of-range index and neither needs a guard.

**The price of the missing wrap is uneven coverage.** Observation `j` of the first `block_size` observations lies inside only `j` of the start positions, so it is drawn about `j / block_size` as often as an observation of the middle, and the last `block_size` observations mirror the ramp. Measured over 20000 index vectors at `T = 100` and `block_size = 5`, the first five observations are drawn at 0.192, 0.392, 0.594, 0.791 and 0.987 of the middle rate, and the last five at 1.007, 0.806, 0.605, 0.403 and 0.202. So the first and the last observations carry about one fifth of the weight of a middle one at this block size, and about `1 / block_size` of it in general. Prefer [`CircularBootstrap`](@ref) when that asymmetry is not wanted.

# Related

  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)

# References

  - $(ref_dict[:kunsch1989])
"""
struct MovingBootstrap <: ARCHBootstrapSet end
"""
    bootstrap_indices(alg::ARCHBootstrapSet, rng::Random.AbstractRNG, T::Integer,
                      block_size::Integer)

Generate a vector of `T` observation indices for one block bootstrap resample.

The three methods lay blocks of consecutive indices end to end until `T` indices are filled, and they differ only in how long a block is, in whether it wraps past the end of the series, and in where it may start. [`ARCHBootstrapSet`](@ref) tabulates the three axes.

# Algorithm

The method Julia selects on the type of `alg` is the algorithm, so the three procedures below are the three methods.

## `StationaryBootstrap`

 1. Set the restart probability `p` to `inv(block_size)`.
 2. Draw `idx[1]` uniformly from `1:T`, which starts the first block.
 3. For each later `t`, draw one uniform variate. Below `p`, draw `idx[t]` uniformly from `1:T`, which starts a new block. Otherwise set `idx[t]` to `mod1(idx[t - 1] + 1, T)`, which continues the block and wraps it past the end of the series.
 4. Return `idx`. Step 3 makes the block length geometric with mean `block_size`, and `mod1` keeps every index inside `1:T` whatever `block_size` is.

## `CircularBootstrap`

 1. Set the fill position `t` to zero.
 2. While `t` is below `T`, draw a start `s` uniformly from `1:T`.
 3. Fill the next `min(block_size, T - t)` entries with `mod1(s + k, T)` for `k` from zero, which lays one block and wraps it past the end of the series. The last block of the vector is cut short when fewer than `block_size` entries are left.
 4. Advance `t` by `block_size`, and go back to step 2 while `t` is below `T`.
 5. Return `idx`. `mod1` keeps every index inside `1:T`, so this method needs no guard on `block_size`.

## `MovingBootstrap`

 1. Check that `block_size` does not exceed `T`, and raise a `DomainError` otherwise. Step 3 would draw from an empty range.
 2. Set the fill position `t` to zero.
 3. While `t` is below `T`, draw a start `s` uniformly from `1:(T - block_size + 1)`, which is the last start whose block still ends at or before `T`.
 4. Fill the next `min(block_size, T - t)` entries with `s + k` for `k` from zero, which lays one block without a wrap. The last block of the vector is cut short when fewer than `block_size` entries are left.
 5. Advance `t` by `block_size`, and go back to step 3 while `t` is below `T`.
 6. Return `idx`. Step 3 bounds `s + k` by `T`, so no index leaves the range.

# Arguments

  - `alg`: Bootstrap algorithm type.
  - `rng`: Random number generator.
  - `T`: Number of observations in the sample being resampled.
  - `block_size`: Size of blocks for resampling. Mean block length for [`StationaryBootstrap`](@ref), fixed block length otherwise.

# Validation

  - [`MovingBootstrap`](@ref) requires `block_size <= T`, raising a `DomainError`. The other two methods take a `block_size` above `T` without a raise, and each states on its own type what it degenerates to.

# Returns

  - `idx::Vector{Int}`: Indices in `1:T` selecting the rows of one bootstrap resample.

# Related

  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
  - [`ARCHBootstrapSet`](@ref)

# References

  - $(ref_dict[:politis1994stationary])
  - $(ref_dict[:politis1992circular])
  - $(ref_dict[:kunsch1989])
"""
function bootstrap_indices(::StationaryBootstrap, rng::Random.AbstractRNG, T::Integer,
                           block_size::Integer)
    p = inv(block_size)
    idx = Vector{Int}(undef, T)
    idx[1] = rand(rng, 1:T)
    for t in 2:T
        idx[t] = rand(rng) < p ? rand(rng, 1:T) : mod1(idx[t - 1] + 1, T)
    end
    return idx
end
function bootstrap_indices(::CircularBootstrap, rng::Random.AbstractRNG, T::Integer,
                           block_size::Integer)
    idx = Vector{Int}(undef, T)
    t = 0
    while t < T
        s = rand(rng, 1:T)
        for k in 0:(min(block_size, T - t) - 1)
            idx[t + k + 1] = mod1(s + k, T)
        end
        t += block_size
    end
    return idx
end
function bootstrap_indices(::MovingBootstrap, rng::Random.AbstractRNG, T::Integer,
                           block_size::Integer)
    @argcheck(block_size <= T,
              DomainError(block_size,
                          "block_size must be <= the number of observations $T"))
    idx = Vector{Int}(undef, T)
    t = 0
    while t < T
        s = rand(rng, 1:(T - block_size + 1))
        for k in 0:(min(block_size, T - t) - 1)
            idx[t + k + 1] = s + k
        end
        t += block_size
    end
    return idx
end
"""
$(DocStringExtensions.TYPEDEF)

Fits a box or an ellipsoidal uncertainty set from the spread of the statistics over a block bootstrap of the return series.

It is the bootstrapping method of Equation 11.18 of the source, and it assumes no law for the returns. The `bootstrap` field picks one of the three block bootstraps, each of which the library implements itself in [`bootstrap_indices`](@ref) and cites its own paper for.

**The name carries no volatility model.** The method was ported from Riskfolio-Lib, which draws its resamples through the `bootstrap` sub-package of the Python `arch` package. That package is named for the volatility models it also ships, and its bootstrap sub-package fits none of them. This type fits none either: it refits `me` and `ce` on each resample and reads the spread of the refits, so no docstring in this file states a conditional variance recursion.

**The centre and the spread come from different estimators, and nothing reconciles them.** The centre `val` is the point estimate `pe` fits, while the bounds come from refitting `me` and `ce` on the resamples. So a box need not contain its own centre when the two disagree. With `pe = EmpiricalPrior()` and `me = MedianExpectedReturns()` over 250 resamples of a 252-by-5 sample at `block_size = 3` and `seed = 987654321`, one asset's `val` of -0.000934 sits above its `ub` of -0.001007. A consumer of the mean axis reads only `val` and the half-width `(ub - lb) / 2`, so the asymmetry is discarded and the set is centred on the prior's estimate with the bootstrap's width; a consumer of the covariance axis reads both bounds and never `val`.

**`ce` enters the ellipsoidal covariance axis twice.** It fits the covariance of every resample, and it then fits the shape matrix over the deviations of those covariances. Turning off its bias correction moves the resampled covariances by 0.397% over 252 observations and the covariance-axis shape matrix by 1.784% over 100 resamples. The mean axis reads `ce` once, over the deviations alone, and moves by exactly 1.0%.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ARCHUncertaintySet(;
        pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
        n_sim::Integer = 3_000,
        block_size::Integer = 3,
        q::Number = 0.05,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
        bootstrap::ARCHBootstrapSet = StationaryBootstrap(),
        kwargs::NamedTuple = (;),
    ) -> ARCHUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `n_sim > 0`.
  - `block_size > 0`.
  - `0 < q < 1`.

# Examples

```jldoctest
julia> ARCHUncertaintySet()
ARCHUncertaintySet
          pe ┼ EmpiricalPrior
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
          ce ┼ PortfolioOptimisersCovariance
             │   ce ┼ Covariance
             │      │    me ┼ SimpleExpectedReturns
             │      │       │   w ┴ nothing
             │      │    ce ┼ GeneralCovariance
             │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │      │       │    w ┴ nothing
             │      │   alg ┼ FullMoment()
             │      │     w ┴ nothing
             │   mp ┼ MatrixProcessing
             │      │     pdm ┼ Posdef
             │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      │      dn ┼ nothing
             │      │      dt ┼ nothing
             │      │     alg ┼ nothing
             │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
          me ┼ SimpleExpectedReturns
             │   w ┴ nothing
         alg ┼ BoxUncertaintySetAlgorithm()
       n_sim ┼ Int64: 3000
  block_size ┼ Int64: 3
           q ┼ Float64: 0.05
         rng ┼ Random.TaskLocalRNG: Random.TaskLocalRNG()
        seed ┼ nothing
   bootstrap ┼ StationaryBootstrap()
      kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`BootstrapUncertaintySetEstimator`](@ref)
  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)

# References

  - $(ref_dict[:cajas2025]) Equation 11.18.
"""
@concrete struct ARCHUncertaintySet <: BootstrapUncertaintySetEstimator
    """
    $(field_dict[:pe])
    """
    pe
    """
    $(field_dict[:ce])
    """
    ce
    """
    $(field_dict[:me])
    """
    me
    """
    $(field_dict[:ucsa])
    """
    alg
    """
    $(field_dict[:n_sim])
    """
    n_sim
    """
    $(field_dict[:block_size])
    """
    block_size
    """
    $(field_dict[:q_bs])
    """
    q
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    """
    $(field_dict[:bootstrap])
    """
    bootstrap
    """
    $(field_dict[:kwargs])
    """
    kwargs
    function ARCHUncertaintySet(pe::AbstractLowOrderPriorEstimator,
                                ce::StatsBase.CovarianceEstimator,
                                me::AbstractExpectedReturnsEstimator,
                                alg::AbstractUncertaintySetAlgorithm, n_sim::Integer,
                                block_size::Integer, q::Number, rng::Random.AbstractRNG,
                                seed::Option{<:Integer}, bootstrap::ARCHBootstrapSet,
                                kwargs::NamedTuple)
        @argcheck(n_sim > zero(n_sim), DomainError(n_sim, "n_sim must be > 0"))
        assert_resource_cap(n_sim, RESOURCE_LIMITS[].max_n_sim, :n_sim, :max_n_sim)
        @argcheck(block_size > zero(block_size),
                  DomainError(block_size, "block_size must be > 0"))
        assert_unit_interval(q, :q)
        return new{typeof(pe), typeof(ce), typeof(me), typeof(alg), typeof(n_sim),
                   typeof(block_size), typeof(q), typeof(rng), typeof(seed),
                   typeof(bootstrap), typeof(kwargs)}(pe, ce, me, alg, n_sim, block_size, q,
                                                      rng, seed, bootstrap, kwargs)
    end
end
function ARCHUncertaintySet(; pe::AbstractLowOrderPriorEstimator = EmpiricalPrior(),
                            ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                            me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                            alg::AbstractUncertaintySetAlgorithm = BoxUncertaintySetAlgorithm(),
                            n_sim::Integer = 3_000, block_size::Integer = 3,
                            q::Number = 0.05,
                            rng::Random.AbstractRNG = Random.default_rng(),
                            seed::Option{<:Integer} = nothing,
                            bootstrap::ARCHBootstrapSet = StationaryBootstrap(),
                            kwargs::NamedTuple = (;))::ARCHUncertaintySet
    return ARCHUncertaintySet(pe, ce, me, alg, n_sim, block_size, q, rng, seed, bootstrap,
                              kwargs)
end
"""
    bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)

Refits the mean and the covariance on every block bootstrap resample of `X`, in one pass over one index stream.

Both statistics are read from the same resample, so a caller that needs both axes gets them from `ue.n_sim` index vectors rather than from two independent runs of `ue.n_sim` each.

# Algorithm

 1. Read the observation count `T` from `X`, and allocate `mus` and `sigmas`.
 2. Resolve the generator with [`resolve_rng`](@ref), giving `rng`. A set `ue.seed` gives a private reseeded copy, so the stream restarts at the same place on every call and `ue.rng` is never advanced. An unset `ue.seed` gives `ue.rng` itself, so the stream continues where the previous call left it.
 3. For each of the `ue.n_sim` simulations, draw one index vector with [`bootstrap_indices`](@ref) and take those rows of `X`, giving the resample `Xi`.
 4. Fit `ue.me` on `Xi`, giving one column of `mus`.
 5. Fit `ue.ce` on `Xi`, giving one slice of `sigmas`. Steps 4 and 5 read the same `Xi`, which is what pairs the two statistics.
 6. Return `mus` and `sigmas`. `ue.pe` fits the point estimate the deviations are taken from, and takes no part here.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix to be resampled, one row per observation.
  - `kwargs...`: Additional keyword arguments passed to `ue.me` and `ue.ce`.

# Returns

  - `mus::Matrix{<:Number}`: Matrix of bootstrapped expected return vectors (`size(X, 2) × ue.n_sim`).
  - `sigmas::Array{<:Number, 3})`: Array of bootstrapped covariance matrices (`size(X, 2) × size(X, 2) × ue.n_sim`).

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`bootstrap_indices`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)
    T = size(X, 1)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    rng = resolve_rng(ue.rng, ue.seed)
    for i in 1:(ue.n_sim)
        Xi = X[bootstrap_indices(ue.bootstrap, rng, T, ue.block_size), :]
        mus[:, i] = vec(Statistics.mean(ue.me, Xi; dims = 1, kwargs...))
        sigmas[:, :, i] = Statistics.cov(ue.ce, Xi; dims = 1, kwargs...)
    end
    return mus, sigmas
end
"""
    mu_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)

Refits the mean on every block bootstrap resample of `X`, and fits no covariance.

The index stream is the one [`bootstrap_generator`](@ref) walks, drawn one vector per simulation. A set `ue.seed` restarts it at the same place, so this function sees the same resamples as its two siblings; an unset `ue.seed` does not, and the three then walk different parts of one shared stream.

# Algorithm

 1. Read the observation count `T` from `X`, and allocate `mus`.
 2. Resolve the generator with [`resolve_rng`](@ref), giving `rng`. A set `ue.seed` gives a private reseeded copy, so the stream restarts at the same place on every call and `ue.rng` is never advanced. An unset `ue.seed` gives `ue.rng` itself, so the stream continues where the previous call left it.
 3. For each of the `ue.n_sim` simulations, draw one index vector with [`bootstrap_indices`](@ref) and take those rows of `X`, giving the resample `Xi`.
 4. Fit `ue.me` on `Xi`, giving one column of `mus`.
 5. Return `mus`. `ue.ce` is not read, and `ue.pe` fits the point estimate the deviations are taken from and takes no part here.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix to be resampled, one row per observation.
  - `kwargs...`: Additional keyword arguments passed to `ue.me` and `ue.ce`.

# Returns

  - `mus::Matrix{<:Number}`: Matrix of bootstrapped expected return vectors (`size(X, 2) × ue.n_sim`).

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`bootstrap_indices`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)
    T = size(X, 1)
    mus = Matrix{eltype(X)}(undef, size(X, 2), ue.n_sim)
    rng = resolve_rng(ue.rng, ue.seed)
    for i in 1:(ue.n_sim)
        Xi = X[bootstrap_indices(ue.bootstrap, rng, T, ue.block_size), :]
        mus[:, i] = vec(Statistics.mean(ue.me, Xi; dims = 1, kwargs...))
    end
    return mus
end
"""
    sigma_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)

Refits the covariance on every block bootstrap resample of `X`, and fits no mean.

The index stream is the one [`bootstrap_generator`](@ref) walks, drawn one vector per simulation. A set `ue.seed` restarts it at the same place, so this function sees the same resamples as its two siblings; an unset `ue.seed` does not, and the three then walk different parts of one shared stream.

# Algorithm

 1. Read the observation count `T` from `X`, and allocate `sigmas`.
 2. Resolve the generator with [`resolve_rng`](@ref), giving `rng`. A set `ue.seed` gives a private reseeded copy, so the stream restarts at the same place on every call and `ue.rng` is never advanced. An unset `ue.seed` gives `ue.rng` itself, so the stream continues where the previous call left it.
 3. For each of the `ue.n_sim` simulations, draw one index vector with [`bootstrap_indices`](@ref) and take those rows of `X`, giving the resample `Xi`.
 4. Fit `ue.ce` on `Xi`, giving one slice of `sigmas`. This is the first of the two places the ellipsoidal route reads `ue.ce`.
 5. Return `sigmas`. `ue.me` is not read, and `ue.pe` fits the point estimate the deviations are taken from and takes no part here.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix to be resampled, one row per observation.
  - `kwargs...`: Additional keyword arguments passed to `ue.me` and `ue.ce`.

# Returns

  - `sigmas::Array{<:Number, 3}`: Array of bootstrapped covariance matrices (`size(X, 2) × size(X, 2) × ue.n_sim`).

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`bootstrap_indices`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
"""
function sigma_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)
    T = size(X, 1)
    sigmas = Array{eltype(X)}(undef, size(X, 2), size(X, 2), ue.n_sim)
    rng = resolve_rng(ue.rng, ue.seed)
    for i in 1:(ue.n_sim)
        Xi = X[bootstrap_indices(ue.bootstrap, rng, T, ue.block_size), :]
        sigmas[:, :, i] = Statistics.cov(ue.ce, Xi; dims = 1, kwargs...)
    end
    return sigmas
end
"""
    ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                               <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs box uncertainty sets for expected returns and covariance statistics using bootstrap resampling for time series data.

Both sets come from one pass over one index stream, so the mean and the covariance of a given simulation are read from the same resample. With `ue.seed` set, this method and the pair [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) return the same bounds bit for bit, because [`resolve_rng`](@ref) restarts each call at the same place and all three walk one index stream. With `ue.seed` unset they do not: over 200 resamples of a 252-by-5 sample the mean lower bound moved by 5.31e-4 against a set width of 9.53e-3. So a caller who splits one [`ucs`](@ref) call into two calls to save work keeps the answer only while a seed is set.

Generate ``M`` bootstrap samples, compute ``\\hat{\\boldsymbol{\\mu}}^{(m)}`` and ``\\hat{\\mathbf{\\Sigma}}^{(m)}``, then take element-wise quantile bounds:

```math
\\begin{align}
\\mu_{lb,i} &= Q_{q/2}\\!\\left(\\hat{\\mu}^{(m)}_i\\right)\\,, \\\\
\\mu_{ub,i} &= Q_{1-q/2}\\!\\left(\\hat{\\mu}^{(m)}_i\\right)\\,.
\\end{align}
```

```math
\\begin{align}
(\\mathbf{\\Sigma}_{lb})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{ub})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mu_{lb,i}``, ``\\mu_{ub,i}``: Lower/upper bounds for expected return of asset ``i``.
  - ``(\\mathbf{\\Sigma}_{lb})_{ij}``, ``(\\mathbf{\\Sigma}_{ub})_{ij}``: Lower/upper covariance bounds.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at level ``q/2``.
  - ``\\hat{\\mu}^{(m)}_i``: Bootstrap mean for asset ``i`` in sample ``m``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: Bootstrap covariance element ``(i,j)`` in sample ``m``.
  - ``q``: Significance level.

# Algorithm

 1. Fit `ue.pe` on `X` and `F`, giving the prior `pr`. Its `pr.mu` and `pr.sigma` become the centre `val` of the two sets, and `pr.X` replaces `X` for the resampling.
 2. Draw the resampled statistics with [`bootstrap_generator`](@ref), giving `mus` and `sigmas` from one index stream.
 3. Halve `ue.q`, giving the tail mass `q` that each side of a bound takes.
 4. Read the element-wise quantiles of `mus` with `vec_quantile_bounds`, giving `mu_l` and `mu_u`.
 5. Read the element-wise quantiles of `sigmas` with `box_quantile_bounds`, giving `sigma_l` and `sigma_u`.
 6. Return the two [`BoxUncertaintySet`](@ref) values. The bounds come from step 2 and the centres from step 1, so neither set is guaranteed to contain its own centre.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.pe` fits the centre `val` of both sets, and `ue.me` and `ue.ce` fit the bounds on the resamples, so the two need not agree.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm,
                                    <:Any, <:Any, <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    mus, sigmas = bootstrap_generator(ue, X; kwargs...)
    q = ue.q * 0.5
    mu_l, mu_u = vec_quantile_bounds(mus, q, ue.kwargs)
    sigma_l, sigma_u = box_quantile_bounds(eltype(X), (i, j) -> sigmas[i, j, :], N, q,
                                           ue.kwargs)
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u, val = pr.mu),
           BoxUncertaintySet(; lb = sigma_l, ub = sigma_u, val = pr.sigma)
end
"""
    mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                  <:Any, <:Any, <:Any}, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for expected returns using bootstrap resampling for time series data.

The method walks its own index stream. With `ue.seed` set it returns the same bounds as the mean half of [`ucs`](@ref), bit for bit, because [`resolve_rng`](@ref) restarts each call at the same place. With `ue.seed` unset it does not: over 200 resamples of a 252-by-5 sample the lower bound moved by 5.31e-4 against a set width of 9.53e-3.

# Mathematical definition

```math
\\begin{align}
\\mu_{lb,i} &= Q_{q/2}\\!\\left(\\hat{\\mu}^{(m)}_i\\right)\\,, \\\\
\\mu_{ub,i} &= Q_{1-q/2}\\!\\left(\\hat{\\mu}^{(m)}_i\\right)\\,.
\\end{align}
```

Where:

  - ``\\mu_{lb,i}``, ``\\mu_{ub,i}``: Lower/upper bounds for expected return of asset ``i``.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at level ``q/2``.
  - ``\\hat{\\mu}^{(m)}_i``: Bootstrap mean for asset ``i`` in sample ``m``.
  - ``q``: Significance level.

# Algorithm

 1. Fit `ue.pe` on `X` and `F`, giving the prior `pr`. Its `pr.mu` becomes the centre `val`, and `pr.X` replaces `X` for the resampling.
 2. Draw the resampled means with [`mu_bootstrap_generator`](@ref), giving `mus`. No covariance is fitted here, so `ue.ce` is not read.
 3. Halve `ue.q`, giving the tail mass `q` that each side of a bound takes.
 4. Read the element-wise quantiles of `mus` with `vec_quantile_bounds`, giving `mu_l` and `mu_u`.
 5. Return the [`BoxUncertaintySet`](@ref). The bounds come from step 2 and the centre from step 1, so the set is not guaranteed to contain its own centre.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.pe` fits the centre `val`, and `ue.me` fits the bounds on the resamples, so the two need not agree.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm,
                                       <:Any, <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    mus = mu_bootstrap_generator(ue, X; kwargs...)
    q = ue.q * 0.5
    mu_l, mu_u = vec_quantile_bounds(mus, q, ue.kwargs)
    return BoxUncertaintySet(; lb = mu_l, ub = mu_u, val = pr.mu)
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                     <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a box uncertainty set for covariance using bootstrap resampling for time series data.

The method walks its own index stream. With `ue.seed` set it returns the same bounds as the covariance half of [`ucs`](@ref), bit for bit, because [`resolve_rng`](@ref) restarts each call at the same place. With `ue.seed` unset it does not: over 200 resamples of a 252-by-5 sample the lower bound moved by 2.12e-5.

# Mathematical definition

```math
\\begin{align}
(\\mathbf{\\Sigma}_{lb})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{ub})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{lb})_{ij}``, ``(\\mathbf{\\Sigma}_{ub})_{ij}``: Lower/upper covariance bounds.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at level ``q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: Bootstrap covariance element ``(i,j)`` in sample ``m``.
  - ``q``: Significance level.

# Algorithm

 1. Fit `ue.pe` on `X` and `F`, giving the prior `pr`. Its `pr.sigma` becomes the centre `val`, and `pr.X` replaces `X` for the resampling.
 2. Draw the resampled covariances with [`sigma_bootstrap_generator`](@ref), giving `sigmas`. No mean is fitted here, so `ue.me` is not read.
 3. Halve `ue.q`, giving the tail mass `q` that each side of a bound takes.
 4. Read the element-wise quantiles of `sigmas` with `box_quantile_bounds`, giving `sigma_l` and `sigma_u`.
 5. Return the [`BoxUncertaintySet`](@ref). The bounds come from step 2 and the centre from step 1, so the set is not guaranteed to contain its own centre.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.pe` fits the centre `val`, and `ue.ce` fits the bounds on the resamples, so the two need not agree.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Covariance uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm,
                                          <:Any, <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    sigmas = sigma_bootstrap_generator(ue, X; kwargs...)
    q = ue.q * 0.5
    sigma_l, sigma_u = box_quantile_bounds(eltype(X), (i, j) -> sigmas[i, j, :], N, q,
                                           ue.kwargs)
    return BoxUncertaintySet(; lb = sigma_l, ub = sigma_u, val = pr.sigma)
end
"""
    ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                               <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs ellipsoidal uncertainty sets for expected returns and covariance statistics using bootstrap resampling for time series data.

Both sets come from one pass over one index stream. The shape matrices are the empirical covariances of the bootstrap deviations, fitted with `ue.ce`, so `ue.ce` fits the covariance axis twice: once inside every resample and once over the deviations of those resampled covariances. The mean axis reads it once, over the mean deviations alone. With `ue.seed` set, this method and the pair [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) agree; with `ue.seed` unset they do not.

# Mathematical definition

Compute bootstrap deviations ``\\boldsymbol{\\delta}_{\\mu}^{(m)} = \\hat{\\boldsymbol{\\mu}}^{(m)} - \\hat{\\boldsymbol{\\mu}}`` and ``\\boldsymbol{\\delta}_{\\Sigma}^{(m)} = \\operatorname{vec}(\\hat{\\mathbf{\\Sigma}}^{(m)} - \\hat{\\mathbf{\\Sigma}})``. Fit empirical covariances:

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\mu} &= \\operatorname{Cov}\\!\\left(\\boldsymbol{\\delta}_{\\mu}^{(m)}\\right)\\,, \\\\
\\mathbf{\\Sigma}_{\\Sigma} &= \\operatorname{Cov}\\!\\left(\\boldsymbol{\\delta}_{\\Sigma}^{(m)}\\right)\\,.
\\end{align}
```

Then form ellipsoidal sets:

```math
\\begin{align}
\\mathcal{E}_{\\mu} &= \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\mu}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\mu}^2\\right\\}\\,.
\\end{align}
```

```math
\\begin{align}
\\mathcal{E}_{\\Sigma} &= \\left\\{\\mathbf{\\Sigma} : \\left\\lVert \\mathbf{\\Sigma}_{\\Sigma}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}}) \\right\\rVert_2 \\leq k_{\\Sigma}\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\mu}``: Empirical covariance of bootstrap mean deviations.
  - ``\\mathbf{\\Sigma}_{\\Sigma}``: Empirical covariance of bootstrap covariance deviations (vectorised).
  - ``\\hat{\\boldsymbol{\\mu}}``, ``\\hat{\\mathbf{\\Sigma}}``: Estimated mean and covariance.
  - ``\\boldsymbol{\\delta}_{\\mu}^{(m)}``, ``\\boldsymbol{\\delta}_{\\Sigma}^{(m)}``: Bootstrap deviations for mean and covariance.
  - ``\\mathcal{E}_{\\mu}``: Ellipsoidal uncertainty set for expected returns.
  - ``\\mathcal{E}_{\\Sigma}``: Ellipsoidal uncertainty set for covariance.
  - ``k_{\\mu}``, ``k_{\\Sigma}``: Empirically fitted scaling parameters.

# Algorithm

 1. Fit `ue.pe` on `X` and `F`, giving the prior `pr`. Its `pr.mu` and `pr.sigma` become the centres of the two sets, and `pr.X` replaces `X` for the resampling.
 2. Draw the resampled statistics with [`bootstrap_generator`](@ref), giving `mus` and `sigmas` from one index stream.
 3. Subtract `pr.mu` from each column of `mus`, and the vectorised `pr.sigma` from each slice of `sigmas`, giving the deviation matrices `X_mu` and `X_sigma`. Transpose both, so a row is one simulation.
 4. Fit `ue.ce` on `X_mu`, giving the shape matrix `sigma_mu`. This is the second reading of `ue.ce` on the covariance axis and the only one on the mean axis, so the shape matrices are empirical and no asymptotic formula enters.
 5. Fit `ue.ce` on `X_sigma`, giving the shape matrix `sigma_sigma`.
 6. Build both sets with `ellipsoidal_set` under `ue.alg.diagonal` and `ue.alg.method`, which fits each radius `k` at the level `ue.q`.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.ce` fits both the covariance of every resample and the shape matrix over the deviations, so it enters the covariance axis twice and the mean axis once. `ue.pe` fits the centres, and `ue.me` and `ue.ce` fit the spread, so the two need not agree.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Ellipsoidal uncertainty set for expected returns.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Ellipsoidal uncertainty set for covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any,
                                    <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    mus, sigmas = bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pr.mu)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pr.sigma)
    end
    X_mu = transpose(X_mu)
    X_sigma = transpose(X_sigma)
    sigma_mu = Statistics.cov(ue.ce, X_mu)
    sigma_sigma = Statistics.cov(ue.ce, X_sigma)
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_mu, sigma_mu,
                           MuUncertaintySetClass(), pr.mu),
           ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_sigma, sigma_sigma,
                           SigmaUncertaintySetClass(), pr.sigma)
end
"""
    mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                  <:Any, <:Any, <:Any}, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for expected returns using bootstrap resampling for time series data.

The shape matrix is the empirical covariance of the bootstrap mean deviations, fitted with `ue.ce`, so `ue.ce` enters this axis once even though no covariance is fitted inside a resample. With `ue.seed` set the method returns the same set as the mean half of [`ucs`](@ref); with `ue.seed` unset it does not.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\mu} &= \\operatorname{Cov}\\!\\left(\\hat{\\boldsymbol{\\mu}}^{(m)} - \\hat{\\boldsymbol{\\mu}}\\right)\\,, \\\\
\\mathcal{E}_{\\mu} &= \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\mu}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\mu}^2\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\mu}``: Empirical covariance of bootstrap mean deviations.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - ``\\mathcal{E}_{\\mu}``: Ellipsoidal uncertainty set for expected returns.
  - ``k_{\\mu}``: Empirically fitted scaling parameter.

# Algorithm

 1. Fit `ue.pe` on `X` and `F`, giving the prior `pr`. Its `pr.mu` becomes the centre, and `pr.X` replaces `X` for the resampling.
 2. Draw the resampled means with [`mu_bootstrap_generator`](@ref), giving `mus`.
 3. Subtract `pr.mu` from each column of `mus`, giving the deviation matrix `X_mu`. Transpose it, so a row is one simulation.
 4. Fit `ue.ce` on `X_mu`, giving the shape matrix `sigma_mu`. The shape is empirical and no asymptotic formula enters.
 5. Build the set with `ellipsoidal_set` under `ue.alg.diagonal` and `ue.alg.method`, which fits the radius `k` at the level `ue.q`.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.me` fits the mean of every resample, and `ue.ce` fits the shape matrix over the deviations.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Ellipsoidal uncertainty set for expected returns.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any,
                                       <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    mus = mu_bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pr.mu)
    end
    X_mu = transpose(X_mu)
    sigma_mu = Statistics.cov(ue.ce, X_mu)
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_mu, sigma_mu,
                           MuUncertaintySetClass(), pr.mu)
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                     <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs an ellipsoidal uncertainty set for covariance using bootstrap resampling for time series data.

The shape matrix is the empirical covariance of the bootstrap covariance deviations, fitted with `ue.ce`, so `ue.ce` enters this axis twice: once inside every resample and once over the deviations. Turning off its bias correction moves the resampled covariances by 0.397% over 252 observations and the shape matrix by 1.784% over 100 resamples. With `ue.seed` set the method returns the same set as the covariance half of [`ucs`](@ref); with `ue.seed` unset it does not.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\Sigma} &= \\operatorname{Cov}\\!\\left(\\operatorname{vec}(\\hat{\\mathbf{\\Sigma}}^{(m)} - \\hat{\\mathbf{\\Sigma}})\\right)\\,, \\\\
\\mathcal{E}_{\\Sigma} &= \\left\\{\\mathbf{\\Sigma} : \\left\\lVert \\mathbf{\\Sigma}_{\\Sigma}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}}) \\right\\rVert_2 \\leq k_{\\Sigma}\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\Sigma}``: Empirical covariance of bootstrap covariance deviations (vectorised).
  - $(math_dict[:Sigma_hat])
  - ``\\mathcal{E}_{\\Sigma}``: Ellipsoidal uncertainty set for covariance.
  - ``k_{\\Sigma}``: Empirically fitted scaling parameter.

# Algorithm

 1. Fit `ue.pe` on `X` and `F`, giving the prior `pr`. Its `pr.sigma` becomes the centre, and `pr.X` replaces `X` for the resampling.
 2. Draw the resampled covariances with [`sigma_bootstrap_generator`](@ref), giving `sigmas`. This is the first reading of `ue.ce`.
 3. Subtract the vectorised `pr.sigma` from each slice of `sigmas`, giving the deviation matrix `X_sigma`. Transpose it, so a row is one simulation.
 4. Fit `ue.ce` on `X_sigma`, giving the shape matrix `sigma_sigma`. This is the second reading of `ue.ce`, and the shape is empirical rather than asymptotic.
 5. Build the set with `ellipsoidal_set` under `ue.alg.diagonal` and `ue.alg.method`, which fits the radius `k` at the level `ue.q`.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.ce` fits both the covariance of every resample and the shape matrix over the deviations, so it enters this axis twice.
  - `X`: Data matrix to be resampled.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Ellipsoidal uncertainty set for covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any,
                                          <:EllipsoidalUncertaintySetAlgorithm, <:Any,
                                          <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    sigmas = sigma_bootstrap_generator(ue, X; kwargs...)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_sigma, 2)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pr.sigma)
    end
    X_sigma = transpose(X_sigma)
    sigma_sigma = Statistics.cov(ue.ce, X_sigma)
    return ellipsoidal_set(ue.alg.diagonal, ue.alg.method, ue.q, X_sigma, sigma_sigma,
                           SigmaUncertaintySetClass(), pr.sigma)
end

"""
    ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm, <:Any,
                               <:Any, <:Any, <:Any, <:Any}, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs norm-ball uncertainty sets for expected returns and covariance statistics using bootstrap resampling for time series data.

**The bootstrap deviations are the geometry map, so this route builds no shape matrix.** The ellipsoidal sibling fits `ue.ce` on the deviations, which on the covariance axis is an ``N^{2} \\times N^{2}`` matrix of rank at most ``\\min(M - 1, N(N+1)/2)``, so it is rank deficient at every sample size and the default matrix processing repairs it. The map [`norm_ball_deviation_factor`](@ref) builds carries the same second moment exactly, at the rank the sample has, and `ue.ce` takes no part in it. `ue.ce` still fits the covariance of every resample, so it enters this axis once rather than twice. On the mean axis the map is of full rank once `ue.n_sim` exceeds ``N``, so the two shapes agree and the two sets reach the same weights.

# Mathematical definition

```math
\\mathbf{L} = \\dfrac{\\left(\\mathbf{X} - \\boldsymbol{1}\\bar{\\mathbf{x}}^{\\intercal}\\right)^{\\intercal}}{\\sqrt{M - 1}}\\,, \\qquad \\mathbf{L}\\mathbf{L}^{\\intercal} = \\operatorname{Cov}(\\mathbf{X})\\,, \\qquad U = \\left\\{ \\hat{\\mathbf{z}} + \\mathbf{L}\\mathbf{u} \\, \\vert \\, \\lVert \\mathbf{u} \\rVert_{p} \\leq \\kappa \\right\\}\\,.
```

Where:

  - ``\\mathbf{L}``: Geometry map.
  - ``\\mathbf{X}``: Bootstrap deviations, one row per resample.
  - ``\\bar{\\mathbf{x}}``: Column means of ``\\mathbf{X}``.
  - ``M``: Number of resamples, `ue.n_sim`.
  - ``\\hat{\\mathbf{z}}``: Point estimate the deviations are taken from.
  - ``\\kappa``, ``p``: Radius and norm order of the ball.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `X = pr.X` and `N = size(X, 2)`.
 2. Refit both statistics on every resample with [`bootstrap_generator`](@ref), giving `mus` and `sigmas` from one index stream.
 3. Subtract `pr.mu` from every resampled mean and `pr.sigma` from every resampled covariance, giving `X_mu` and `X_sigma`, one deviation per column.
 4. Assemble the two sets with [`norm_ball_deviation_set`](@ref) on the transposed deviations, and return them as a tuple, mean first.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator, `ue.me` and `ue.ce`.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Expected returns uncertainty set.
  - `sigma_ucs::NormBallUncertaintySet`: Covariance uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_deviation_set`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
"""
function ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm,
                                    <:Any, <:Any, <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    mus, sigmas = bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pr.mu)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pr.sigma)
    end
    return norm_ball_deviation_set(ue.alg, ue.q, transpose(X_mu), MuUncertaintySetClass(),
                                   pr.mu),
           norm_ball_deviation_set(ue.alg, ue.q, transpose(X_sigma),
                                   SigmaUncertaintySetClass(), pr.sigma)
end
"""
    mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm, <:Any,
                                  <:Any, <:Any, <:Any, <:Any}, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a norm-ball uncertainty set for expected returns using bootstrap resampling for time series data.

**The bootstrap deviations are the geometry map, so this route builds no shape matrix**, and `ue.ce` takes no part on this axis at all: the ellipsoidal sibling fits it on the mean deviations, and the map carries the same second moment without it. With `ue.seed` set the method sees the same resamples as the mean half of [`ucs`](@ref); with `ue.seed` unset it does not.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `X = pr.X` and `N = size(X, 2)`.
 2. Refit the mean on every resample with [`mu_bootstrap_generator`](@ref), giving `mus`.
 3. Subtract `pr.mu` from every resampled mean, giving `X_mu`, one deviation per column.
 4. Assemble and return the set with [`norm_ball_deviation_set`](@ref) on the transposed deviations, with `pr.mu` as the centre.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator and `ue.me`.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Expected returns uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_deviation_set`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
"""
function mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any,
                                       <:NormBallUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    mus = mu_bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(mus[:, i] - pr.mu)
    end
    return norm_ball_deviation_set(ue.alg, ue.q, transpose(X_mu), MuUncertaintySetClass(),
                                   pr.mu)
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm,
                                     <:Any, <:Any, <:Any, <:Any, <:Any}, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Constructs a norm-ball uncertainty set for covariance using bootstrap resampling for time series data.

**This is the one route of the library that bounds a covariance without a matrix of side ``N^{2}``.** The ellipsoidal sibling fits `ue.ce` on the ``M \\times N^{2}`` deviations, and its shape is rank deficient at **every** sample size, because a vectorised symmetric matrix spans only ``N(N+1)/2`` coordinates; the default matrix processing then repairs it into a matrix the sample never named, and the chi-squared radius reads ``N^{2}`` degrees of freedom where the errors have ``N(N+1)/2``. The map [`norm_ball_deviation_factor`](@ref) builds is the deviations themselves, scaled, so it carries the sample second moment exactly at rank ``\\min(M - 1, N(N+1)/2)``, and [`k_norm_ball`](@ref) reads that rank rather than the side of a shape.

# Algorithm

 1. Fit the prior with `prior(ue.pe, X, F; dims = dims, kwargs...)`, giving `pr`, and read `X = pr.X` and `N = size(X, 2)`.
 2. Refit the covariance on every resample with [`sigma_bootstrap_generator`](@ref), giving `sigmas`.
 3. Subtract `pr.sigma` from every resampled covariance and vectorise, giving `X_sigma`, one deviation per column.
 4. Assemble and return the set with [`norm_ball_deviation_set`](@ref) on the transposed deviations, with `pr.sigma` as the centre.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix.
  - `F`: Optional factor matrix. Used by the prior estimator.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the prior estimator and `ue.ce`.

# Returns

  - `sigma_ucs::NormBallUncertaintySet`: Covariance uncertainty set.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_deviation_set`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
  - [`mu_ucs`](@ref)

# References

  - $(ref_dict[:bentalnemirovski1998]) Section 3, Equation 14.
  - $(ref_dict[:goldfarbiyengar2003]) Section 5.
"""
function sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any,
                                          <:NormBallUncertaintySetAlgorithm, <:Any, <:Any,
                                          <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    pr = prior(ue.pe, X, F; dims = dims, kwargs...)
    X = pr.X
    N = size(X, 2)
    sigmas = sigma_bootstrap_generator(ue, X; kwargs...)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_sigma, 2)
        X_sigma[:, i] = vec(sigmas[:, :, i] - pr.sigma)
    end
    return norm_ball_deviation_set(ue.alg, ue.q, transpose(X_sigma),
                                   SigmaUncertaintySetClass(), pr.sigma)
end

export StationaryBootstrap, CircularBootstrap, MovingBootstrap, ARCHUncertaintySet
