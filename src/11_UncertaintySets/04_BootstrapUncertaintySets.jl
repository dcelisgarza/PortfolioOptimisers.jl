"""
$(DocStringExtensions.TYPEDEF)

Fits an uncertainty set from resamples of the return series, so that the set assumes no distribution for the returns.

All concrete subtypes must subtype `BootstrapUncertaintySetEstimator`. A subtype reads its bounds from the resamples, and not from a closed-form expression.

# Interfaces

A subtype implements the three methods of [`AbstractUncertaintySetEstimator`](@ref). It holds an [`ARCHBootstrapSet`](@ref), which selects how the estimator draws the resamples.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`AbstractUncertaintySetEstimator`](@ref)
  - [`ARCHBootstrapSet`](@ref)
"""
abstract type BootstrapUncertaintySetEstimator <: AbstractUncertaintySetEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Selects how a block bootstrap draws its blocks of consecutive rows, which keep the serial dependence of the return series in the resample.

All concrete subtypes must subtype `ARCHBootstrapSet`. The library has three subtypes. They differ in the block length, in the wrap past the end of the series, and in the range of the start index. The table gives each scheme on these three axes.

| Scheme                        | Block length                                    | Wraps past the end | Start range              |
|:----------------------------- |:----------------------------------------------- |:------------------ |:------------------------ |
| [`StationaryBootstrap`](@ref) | geometric, restart probability `1 / block_size` | yes, by `mod1`     | `1:T`                    |
| [`CircularBootstrap`](@ref)   | fixed `block_size`                              | yes, by `mod1`     | `1:T`                    |
| [`MovingBootstrap`](@ref)     | fixed `block_size`                              | no                 | `1:(T - block_size + 1)` |

The wrap column decides if the scheme draws every observation equally often. The two schemes that wrap draw each of the `T` observations equally often. [`MovingBootstrap`](@ref) does not wrap, so it draws the first and the last observations of the series less often than the middle ones. Its docstring gives the measured rates.

# Interfaces

To add a new scheme, subtype `ARCHBootstrapSet` and implement this method:

## `bootstrap_indices`

  - `bootstrap_indices(alg::ARCHBootstrapSet, rng::Random.AbstractRNG, T::Integer, block_size::Integer) -> Vector{Int}`: Returns the row indices of one resample.

### Arguments

  - `alg`: The concrete subtype instance.
  - `rng`: Random number generator.
  - `T`: Number of observations in the sample that the method resamples.
  - `block_size`: Block length, or the mean block length if the length is random.

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

Draws blocks of random length with a geometric distribution, so that the resample is a stationary series.

A block starts anywhere in `1:T`. Each next index is the previous index plus one, taken by `mod1`, so a block wraps past the end of the series. At each index, a new block starts with probability `1 / block_size`. So the block length is geometric with mean `block_size`.

Over 101170 blocks at `block_size = 10`, the measured mean length is 9.884. The measured frequencies `[0.101, 0.0921, 0.082, 0.0733, 0.0671, 0.0584]` of the lengths `1:6` agree with the geometric masses `[0.1, 0.09, 0.081, 0.0729, 0.0656, 0.059]`. The mean is a little below `block_size`, because the last block of an index vector stops at `T`.

A block wraps, so the scheme draws every observation equally often. `block_size` is a mean and not a bound, so a `block_size` larger than `T` raises no error. The restart probability is then below `1 / T`, and the index vector is almost one wrapped block. The spread becomes smaller, but not zero. Over 250 resamples of a 252-by-5 sample, the standard deviation of the bootstrap means at `block_size = T + 1` is 0.377 of the value at `block_size = 3`.

# Related

  - [`ARCHBootstrapSet`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)

# References

  - $(ref_dict[:politis1994stationary]) Section 2.
"""
struct StationaryBootstrap <: ARCHBootstrapSet end
"""
$(DocStringExtensions.TYPEDEF)

Draws blocks of fixed length `block_size` that wrap past the end of the series, so that the scheme draws every observation equally often.

A block starts anywhere in `1:T` and holds the next `block_size` indices, each taken by `mod1`. So the scheme reads the series as a circle, and the wrap gives the equal rates. Over 20000 index vectors at `T = 100` and `block_size = 5`, the scheme draws every observation between 0.985 and 1.015 times the mean rate. Under [`MovingBootstrap`](@ref), the rate of the first observation is 0.192 of the middle rate.

**A `block_size` of `T` or more gives a set of zero width.** The first block fills the whole index vector, so every resample is a cyclic shift of the series. A cyclic shift is a permutation of the rows, and a permutation of the rows does not change the mean or the covariance. So every resample gives the same statistics, and the box width and the ellipsoidal shape matrix are zero to within rounding error. The library raises no error, because the constructor does not know `T`.

# Related

  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)

# References

  - $(ref_dict[:politis1992circular]) Section 2.
"""
struct CircularBootstrap <: ARCHBootstrapSet end
"""
$(DocStringExtensions.TYPEDEF)

Draws blocks of fixed length `block_size` that do not wrap, so that no block joins the end of the series to its start.

A block starts in `1:(T - block_size + 1)` and holds the next `block_size` indices, so the last index of a block is never above `T`. This scheme is the only one of the three that checks `block_size`, and it is the only one that needs the check. If `block_size` is larger than `T`, the start range is empty. Without the check, `rand` raises an `ArgumentError` about an empty range. The check raises a `DomainError` that names `block_size` and `T`. The two schemes that wrap take every index through `mod1`, which cannot leave `1:T`, so they need no check.

Without the wrap, the scheme draws the observations at different rates. For `j` up to `block_size`, only `j` start positions give a block that holds observation `j`. So the scheme draws observation `j` about `j / block_size` as often as a middle observation, and the last `block_size` observations have the same ramp in the opposite order. At `T = 100` and `block_size = 5`, the measured rates of the first five observations are 0.192, 0.392, 0.594, 0.791 and 0.987 of the middle rate. Use [`CircularBootstrap`](@ref) if you do not want this asymmetry.

Section 2.3 of [kunsch1989](@cite) assumes that `block_size` divides `T`. This scheme does not make that assumption, and its last block stops at `T`.

# Related

  - [`ARCHBootstrapSet`](@ref)
  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)

# References

  - $(ref_dict[:kunsch1989]) Section 2.3, Equation 2.7.
"""
struct MovingBootstrap <: ARCHBootstrapSet end
"""
    bootstrap_indices(alg::ARCHBootstrapSet, rng::Random.AbstractRNG, T::Integer,
                      block_size::Integer)

Makes the vector of `T` observation indices for one resample of a block bootstrap.

The three methods put blocks of consecutive indices one after the other until the vector holds `T` indices. They differ only in the block length, in the wrap past the end of the series, and in the range of the start index. [`ARCHBootstrapSet`](@ref) gives the three axes in a table.

# Algorithm

Julia selects the method on the type of `alg`. Each procedure below is one of the three methods.

## `StationaryBootstrap`

 1. Set the restart probability `p` to `inv(block_size)`.
 2. Draw `idx[1]` uniformly from `1:T`. This starts the first block.
 3. For each later `t`, draw one uniform variate. If the variate is below `p`, draw `idx[t]` uniformly from `1:T`, which starts a new block. If not, set `idx[t]` to `mod1(idx[t - 1] + 1, T)`. This continues the block, and wraps it past the end of the series.
 4. Return `idx`. Step 3 makes the block length geometric with mean `block_size`. For each value of `block_size`, `mod1` keeps every index in `1:T`.

## `CircularBootstrap`

 1. Set the fill position `t` to zero.
 2. While `t` is below `T`, draw a start `s` uniformly from `1:T`.
 3. Fill the next `min(block_size, T - t)` entries with `mod1(s + k, T)`, for `k` from zero. This puts one block in the vector, and wraps it past the end of the series. If fewer than `block_size` entries are free, the last block of the vector is shorter.
 4. Increase `t` by `block_size`. Go back to step 2 while `t` is below `T`.
 5. Return `idx`. `mod1` keeps every index in `1:T`, so this method needs no check on `block_size`.

## `MovingBootstrap`

 1. Make sure that `block_size` is not larger than `T`. If it is larger, raise a `DomainError`, because step 3 then has an empty range.
 2. Set the fill position `t` to zero.
 3. While `t` is below `T`, draw a start `s` uniformly from `1:(T - block_size + 1)`. The upper end is the last start whose block ends at or before `T`.
 4. Fill the next `min(block_size, T - t)` entries with `s + k`, for `k` from zero. This puts one block in the vector, with no wrap. If fewer than `block_size` entries are free, the last block of the vector is shorter.
 5. Increase `t` by `block_size`. Go back to step 3 while `t` is below `T`.
 6. Return `idx`. The range of step 3 keeps `s + k` at or below `T`, so no index leaves `1:T`.

# Arguments

  - `alg`: Bootstrap scheme. Its type selects the method.
  - `rng`: Random number generator.
  - `T`: Number of observations in the sample that the method resamples.
  - `block_size`: Block length. It is the mean block length for [`StationaryBootstrap`](@ref), and the fixed block length for the other two schemes.

# Validation

  - [`MovingBootstrap`](@ref) needs `block_size <= T`, and raises a `DomainError` otherwise. The other two methods accept a `block_size` larger than `T` with no error. The docstring of each type states the result.

# Returns

  - `idx::Vector{Int}`: Indices in `1:T` that select the rows of one resample.

# Related

  - [`StationaryBootstrap`](@ref)
  - [`CircularBootstrap`](@ref)
  - [`MovingBootstrap`](@ref)
  - [`ARCHBootstrapSet`](@ref)

# References

  - $(ref_dict[:politis1994stationary]) Section 2.
  - $(ref_dict[:politis1992circular]) Section 2.
  - $(ref_dict[:kunsch1989]) Section 2.3, Equation 2.7.
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

Fits a box, an ellipsoidal or a norm-ball uncertainty set from the spread of the statistics over block bootstrap resamples of the return series.

The method is the bootstrap method of Equation 11.18 of [cajas2025](@cite), and it assumes no distribution for the returns. Section 4 of [tutuncu2004](@cite) is the first work to read the bounds of a box from the percentiles of bootstrap means and covariances. That work resamples single observations, and it uses the percentile interval of the mean with no centre. Equation 11.18 of [cajas2025](@cite) resamples blocks, and it puts the half-width of the interval on the point estimate. Its Listing 11.11 builds the ellipsoid from the same resamples.

The `bootstrap` field selects one of the three block bootstraps. The library implements all three in [`bootstrap_indices`](@ref), and the docstring of each bootstrap type cites the work that defines it.

The name does not refer to a volatility model. It comes from the `bootstrap` module of the Python package `arch`, which fits volatility models, but this type fits no such model. It refits `me` and `ce` on each resample, and it reads the spread of the refits.

The centre and the bounds come from different estimators, and the library does not make them agree. `pe` fits the centre `val`, and `me` and `ce` fit the bounds on the resamples. If the two disagree, a box does not always contain its own centre. Take `pe = EmpiricalPrior()`, `me = MedianExpectedReturns()`, `block_size = 3` and `seed = 987654321`. Over 250 resamples of a 252-by-5 sample, the `val` of one asset is -0.000934, which is above its `ub` of -0.001007.

On the mean axis, the consumer reads only `val` and the half-width `(ub - lb) / 2`. So the consumer ignores the asymmetry, and the set has the centre of the prior and the width of the bootstrap. On the covariance axis, the consumer reads both bounds, and it does not read `val`.

`ce` enters the ellipsoidal covariance axis twice. It fits the covariance of every resample, and then it fits the shape matrix over the deviations of those covariances. The mean axis uses `ce` once, over the mean deviations. A `ce` with no bias correction changes the resampled covariances by 0.397% over 252 observations. It changes the shape matrix of the covariance axis by 1.784% over 100 resamples, and the shape matrix of the mean axis by exactly 1.0%.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ARCHUncertaintySet(;
        pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
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

The keywords are the fields of the struct.

## Validation

  - $(val_dict[:ce_cov])
  - `n_sim > 0`.
  - `n_sim > 1` if `alg` is an [`EllipsoidalUncertaintySetAlgorithm`](@ref) or a [`NormBallUncertaintySetAlgorithm`](@ref). Both shapes read the covariance of the bootstrap deviations, and one resample has no covariance. [`BoxUncertaintySetAlgorithm`](@ref) reads quantiles, so it accepts one resample.
  - `block_size > 0`.
  - `0 < q < 1`.

# Examples

```jldoctest
julia> ARCHUncertaintySet()
ARCHUncertaintySet
          pe ┼ EmpiricalPrior
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

  - $(ref_dict[:tutuncu2004]) Sections 2.1 and 4, Equations 4 and 5.
  - $(ref_dict[:cajas2025]) Equation 11.18 and Listing 11.11. $(ref_dict[:no_original_source])
"""
@concrete struct ARCHUncertaintySet <: BootstrapUncertaintySetEstimator
    """
    $(field_dict[:pe_ucs])
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
    function ARCHUncertaintySet(pe::Option{<:AbstractLowOrderPriorEstimator},
                                ce::StatsBase.CovarianceEstimator,
                                me::AbstractExpectedReturnsEstimator,
                                alg::AbstractUncertaintySetAlgorithm, n_sim::Integer,
                                block_size::Integer, q::Number, rng::Random.AbstractRNG,
                                seed::Option{<:Integer}, bootstrap::ARCHBootstrapSet,
                                kwargs::NamedTuple)
        assert_covariance_estimator(ce, :ce)
        @argcheck(n_sim > zero(n_sim), DomainError(n_sim, "n_sim must be > 0"))
        if alg isa
           Union{EllipsoidalUncertaintySetAlgorithm, NormBallUncertaintySetAlgorithm}
            @argcheck(n_sim > one(n_sim),
                      DomainError(n_sim,
                                  "an ellipsoid or a norm ball reads the covariance of the resampled errors, and one resample has no covariance. Set n_sim to at least 2, or use BoxUncertaintySetAlgorithm, which reads quantiles and accepts one resample"))
        end
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
function ARCHUncertaintySet(;
                            pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
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

The function fits both statistics on the same resample. So a caller that needs both axes gets them from `ue.n_sim` index vectors, and not from two runs of `ue.n_sim` index vectors each.

# Algorithm

 1. Read the observation count `T` from `X`. Allocate `mus` and `sigmas`.
 2. Resolve the generator with [`resolve_rng`](@ref) to get `rng`. If `ue.seed` is set, `rng` is a private reseeded copy. The stream then starts at the same place on every call, and `ue.rng` does not advance. If `ue.seed` is `nothing`, `rng` is `ue.rng`, and the stream continues from the end of the previous call.
 3. For each of the `ue.n_sim` simulations, draw one index vector with [`bootstrap_indices`](@ref). Take those rows of `X` to get the resample `Xi`.
 4. Fit `ue.me` on `Xi` to get one column of `mus`.
 5. Fit `ue.ce` on `Xi` to get one slice of `sigmas`. Steps 4 and 5 read the same `Xi`, so the two statistics of one simulation come from one resample.
 6. Return `mus` and `sigmas`. This function does not read `ue.pe`.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix that the function resamples, with one row per observation.
  - `kwargs...`: More keyword arguments, which the function gives to `ue.me` and `ue.ce`.

# Returns

  - `mus::Matrix{<:Number}`: Bootstrap expected returns, with one column per simulation. Its size is `size(X, 2) × ue.n_sim`.
  - `sigmas::Array{<:Number, 3}`: Bootstrap covariance matrices, with one slice per simulation. Its size is `size(X, 2) × size(X, 2) × ue.n_sim`.

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
        sigmas[:, :, i] = Statistics.cov(library_covariance_estimator(ue.ce), Xi; dims = 1,
                                         kwargs...)
    end
    return mus, sigmas
end
"""
    mu_bootstrap_generator(ue::ARCHUncertaintySet, X::MatNum; kwargs...)

Refits the mean on every block bootstrap resample of `X`, and fits no covariance.

The function draws its index vectors as [`bootstrap_generator`](@ref) does, one vector per simulation. If `ue.seed` is set, the stream starts at the same place on every call, so this function sees the same resamples as the other two generators. If `ue.seed` is `nothing`, the three generators read different parts of one shared stream.

# Algorithm

 1. Read the observation count `T` from `X`. Allocate `mus`.
 2. Resolve the generator with [`resolve_rng`](@ref) to get `rng`. If `ue.seed` is set, `rng` is a private reseeded copy. The stream then starts at the same place on every call, and `ue.rng` does not advance. If `ue.seed` is `nothing`, `rng` is `ue.rng`, and the stream continues from the end of the previous call.
 3. For each of the `ue.n_sim` simulations, draw one index vector with [`bootstrap_indices`](@ref). Take those rows of `X` to get the resample `Xi`.
 4. Fit `ue.me` on `Xi` to get one column of `mus`.
 5. Return `mus`. This function reads neither `ue.ce` nor `ue.pe`.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix that the function resamples, with one row per observation.
  - `kwargs...`: More keyword arguments, which the function gives to `ue.me`.

# Returns

  - `mus::Matrix{<:Number}`: Bootstrap expected returns, with one column per simulation. Its size is `size(X, 2) × ue.n_sim`.

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

The function draws its index vectors as [`bootstrap_generator`](@ref) does, one vector per simulation. If `ue.seed` is set, the stream starts at the same place on every call, so this function sees the same resamples as the other two generators. If `ue.seed` is `nothing`, the three generators read different parts of one shared stream.

# Algorithm

 1. Read the observation count `T` from `X`. Allocate `sigmas`.
 2. Resolve the generator with [`resolve_rng`](@ref) to get `rng`. If `ue.seed` is set, `rng` is a private reseeded copy. The stream then starts at the same place on every call, and `ue.rng` does not advance. If `ue.seed` is `nothing`, `rng` is `ue.rng`, and the stream continues from the end of the previous call.
 3. For each of the `ue.n_sim` simulations, draw one index vector with [`bootstrap_indices`](@ref). Take those rows of `X` to get the resample `Xi`.
 4. Fit `ue.ce` on `Xi` to get one slice of `sigmas`. On the ellipsoidal route, this is the first of the two fits of `ue.ce`.
 5. Return `sigmas`. This function reads neither `ue.me` nor `ue.pe`.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Data matrix that the function resamples, with one row per observation.
  - `kwargs...`: More keyword arguments, which the function gives to `ue.ce`.

# Returns

  - `sigmas::Array{<:Number, 3}`: Bootstrap covariance matrices, with one slice per simulation. Its size is `size(X, 2) × size(X, 2) × ue.n_sim`.

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
        sigmas[:, :, i] = Statistics.cov(library_covariance_estimator(ue.ce), Xi; dims = 1,
                                         kwargs...)
    end
    return sigmas
end
# A ARCHUncertaintySet with no prior of its own reads the prior result it is handed
# (see [`reads_prior_result`](@ref)).
function reads_prior_result(::ARCHUncertaintySet{Nothing})::Bool
    return true
end
"""
    ucs(ue::ARCHUncertaintySet, X::MatNum,
        F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    mu_ucs(ue::ARCHUncertaintySet, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    sigma_ucs(ue::ARCHUncertaintySet, X::MatNum,
              F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Fits the sets of an [`ARCHUncertaintySet`](@ref) from returns data, through a fit of the prior of the set.

These are the returns-data methods of the three verbs. Each verb has one such method for all three shapes, because the next call selects the shape. Each method fits `ue.pe` once through [`ucs_prior`](@ref). It then gives the Prior Result to the prior-result method of the same verb, on the same set with `pe = nothing`. That method selects the box, the ellipsoid or the norm ball on the type of `ue.alg`.

The method resamples the `X` of that Prior Result. So a set with a prior of its own fits on that prior, and the prior fits on the returns that the caller gives. The keyword arguments go to the prior fit and to the estimators of the resamples.

# Algorithm

 1. Fit the prior with [`ucs_prior`](@ref) on `ue.pe`, `X` and `F` to get `pr`.
 2. Give `pr` to the prior-result method of the verb, on the set with `pe = nothing`. That method builds the set of the shape that `ue.alg` names.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `X`: Matrix of asset returns.
  - `F`: Optional matrix of factor returns, which a factor prior needs.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method gives to the prior estimator, `ue.me` and `ue.ce`.

# Validation

  - `ue.pe` is not `nothing`. Otherwise [`ucs_prior`](@ref) raises an `ArgumentError` that names the prior-result calls.

# Returns

  - `mu_ucs::AbstractUncertaintySetResult`: Uncertainty set of the expected returns, from `mu_ucs`.
  - `sigma_ucs::AbstractUncertaintySetResult`: Uncertainty set of the covariance, from `sigma_ucs`.
  - `(mu_ucs, sigma_ucs)`: Both sets, from `ucs`.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`ucs_prior`](@ref)
  - [`reads_prior_result`](@ref)
  - [`BoxUncertaintySetAlgorithm`](@ref)
  - [`EllipsoidalUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
"""
function ucs(ue::ARCHUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
             dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return ucs(rebuild_estimator(ue, (; pe = nothing)), pr; kwargs...)
end
function mu_ucs(ue::ARCHUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return mu_ucs(rebuild_estimator(ue, (; pe = nothing)), pr; kwargs...)
end
function sigma_ucs(ue::ARCHUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                   dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return sigma_ucs(rebuild_estimator(ue, (; pe = nothing)), pr; kwargs...)
end
"""
    ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                               <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the box uncertainty sets of the expected returns and the covariance from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

Both sets come from one pass over one index stream, so the mean and the covariance of a simulation come from the same resample. If `ue.seed` is set, this method and the pair [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) return the same bounds bit for bit. The reason is that [`resolve_rng`](@ref) starts each call at the same place of one index stream. If `ue.seed` is `nothing`, the bounds differ. Over 200 resamples of a 252-by-5 sample, the lower bound of the mean changed by 5.31e-4, and the width of the set is 9.53e-3. So a caller who splits one [`ucs`](@ref) call into two calls gets the same answer only if a seed is set.

# Mathematical definition

The method fits ``\\hat{\\boldsymbol{\\mu}}^{(m)}`` and ``\\hat{\\mathbf{\\Sigma}}^{(m)}`` on each of the ``M`` resamples. The bounds are the quantiles of each entry:

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

  - ``\\mu_{lb,i}``, ``\\mu_{ub,i}``: Lower and upper bounds of the expected return of asset ``i``.
  - ``(\\mathbf{\\Sigma}_{lb})_{ij}``, ``(\\mathbf{\\Sigma}_{ub})_{ij}``: Lower and upper bounds of the covariance entry ``(i,j)``.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at the levels ``q/2`` and ``1-q/2``.
  - ``\\hat{\\mu}^{(m)}_i``: Bootstrap mean of asset ``i`` in resample ``m``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: Bootstrap covariance entry ``(i,j)`` in resample ``m``.
  - ``q``: Significance level.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` from the reduced Prior Result. The method resamples this `X`, so under a Scenario Cap it resamples only the rows that the prior keeps.
 3. Draw the resampled statistics with [`bootstrap_generator`](@ref) to get `mus` and `sigmas` from one index stream.
 4. Halve `ue.q` to get the tail mass `q` of each side of a bound.
 5. Take the quantiles of each entry of `mus` with `vec_quantile_bounds` to get `mu_l` and `mu_u`.
 6. Take the quantiles of each entry of `sigmas` with `box_quantile_bounds` to get `sigma_l` and `sigma_u`.
 7. Build the two [`BoxUncertaintySet`](@ref) values. The bounds come from steps 5 and 6. The centres `val` are the `mu` and the `sigma` of the reduced Prior Result. So a set does not always contain its own centre.
 8. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref), and return them. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator. The centres `val` come from `pr`, and `ue.me` and `ue.ce` fit the bounds on the resamples. So the centres and the bounds can disagree.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.me` and `ue.ce`.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Box of the expected returns.
  - `sigma_ucs::BoxUncertaintySet`: Box of the covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm,
                                    <:Any, <:Any, <:Any, <:Any, <:Any},
             pr::AbstractPriorResult; rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    mus, sigmas = bootstrap_generator(ue, X; kwargs...)
    q = ue.q * 0.5
    mu_l, mu_u = vec_quantile_bounds(mus, q, ue.kwargs)
    sigma_l, sigma_u = box_quantile_bounds(eltype(X), (i, j) -> sigmas[i, j, :], N, q,
                                           ue.kwargs)
    mu_set, sigma_set = BoxUncertaintySet(; lb = mu_l, ub = mu_u, val = prr.mu),
                        BoxUncertaintySet(; lb = sigma_l, ub = sigma_u, val = prr.sigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                  <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the box uncertainty set of the expected returns from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

The method draws its own index stream. If `ue.seed` is set, it returns the same bounds as the mean set of [`ucs`](@ref), bit for bit, because [`resolve_rng`](@ref) starts each call at the same place. If `ue.seed` is `nothing`, the bounds differ. Over 200 resamples of a 252-by-5 sample, the lower bound changed by 5.31e-4, and the width of the set is 9.53e-3.

# Mathematical definition

```math
\\begin{align}
\\mu_{lb,i} &= Q_{q/2}\\!\\left(\\hat{\\mu}^{(m)}_i\\right)\\,, \\\\
\\mu_{ub,i} &= Q_{1-q/2}\\!\\left(\\hat{\\mu}^{(m)}_i\\right)\\,.
\\end{align}
```

Where:

  - ``\\mu_{lb,i}``, ``\\mu_{ub,i}``: Lower and upper bounds of the expected return of asset ``i``.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at the levels ``q/2`` and ``1-q/2``.
  - ``\\hat{\\mu}^{(m)}_i``: Bootstrap mean of asset ``i`` in resample ``m``.
  - ``q``: Significance level.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` from the reduced Prior Result. The method resamples this `X`, so under a Scenario Cap it resamples only the rows that the prior keeps.
 3. Draw the resampled means with [`mu_bootstrap_generator`](@ref) to get `mus`. This method fits no covariance, so it does not read `ue.ce`.
 4. Halve `ue.q` to get the tail mass `q` of each side of a bound.
 5. Take the quantiles of each entry of `mus` with `vec_quantile_bounds` to get `mu_l` and `mu_u`.
 6. Build the [`BoxUncertaintySet`](@ref). The bounds come from step 5, and the centre `val` is the `mu` of the reduced Prior Result. So the set does not always contain its own centre.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator. The centre `val` comes from `pr`, and `ue.me` fits the bounds on the resamples. So the centre and the bounds can disagree.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.me`.

# Returns

  - `mu_ucs::BoxUncertaintySet`: Box of the expected returns.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm,
                                       <:Any, <:Any, <:Any, <:Any, <:Any},
                pr::AbstractPriorResult; rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    mus = mu_bootstrap_generator(ue, X; kwargs...)
    q = ue.q * 0.5
    mu_l, mu_u = vec_quantile_bounds(mus, q, ue.kwargs)
    set = BoxUncertaintySet(; lb = mu_l, ub = mu_u, val = prr.mu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                     <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the box uncertainty set of the covariance from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

The method draws its own index stream. If `ue.seed` is set, it returns the same bounds as the covariance set of [`ucs`](@ref), bit for bit, because [`resolve_rng`](@ref) starts each call at the same place. If `ue.seed` is `nothing`, the bounds differ. Over 200 resamples of a 252-by-5 sample, the lower bound changed by 2.12e-5.

# Mathematical definition

```math
\\begin{align}
(\\mathbf{\\Sigma}_{lb})_{ij} &= Q_{q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,, \\\\
(\\mathbf{\\Sigma}_{ub})_{ij} &= Q_{1-q/2}\\!\\left(\\hat{\\Sigma}^{(m)}_{ij}\\right)\\,.
\\end{align}
```

Where:

  - ``(\\mathbf{\\Sigma}_{lb})_{ij}``, ``(\\mathbf{\\Sigma}_{ub})_{ij}``: Lower and upper bounds of the covariance entry ``(i,j)``.
  - ``Q_{q/2}``, ``Q_{1-q/2}``: Quantile functions at the levels ``q/2`` and ``1-q/2``.
  - ``\\hat{\\Sigma}^{(m)}_{ij}``: Bootstrap covariance entry ``(i,j)`` in resample ``m``.
  - ``q``: Significance level.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` from the reduced Prior Result. The method resamples this `X`, so under a Scenario Cap it resamples only the rows that the prior keeps.
 3. Draw the resampled covariances with [`sigma_bootstrap_generator`](@ref) to get `sigmas`. This method fits no mean, so it does not read `ue.me`.
 4. Halve `ue.q` to get the tail mass `q` of each side of a bound.
 5. Take the quantiles of each entry of `sigmas` with `box_quantile_bounds` to get `sigma_l` and `sigma_u`.
 6. Build the [`BoxUncertaintySet`](@ref). The bounds come from step 5, and the centre `val` is the `sigma` of the reduced Prior Result. So the set does not always contain its own centre.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator. The centre `val` comes from `pr`, and `ue.ce` fits the bounds on the resamples. So the centre and the bounds can disagree.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.ce`.

# Returns

  - `sigma_ucs::BoxUncertaintySet`: Box of the covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any,
                                          <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any,
                                          <:Any, <:Any}, pr::AbstractPriorResult;
                   rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    sigmas = sigma_bootstrap_generator(ue, X; kwargs...)
    q = ue.q * 0.5
    sigma_l, sigma_u = box_quantile_bounds(eltype(X), (i, j) -> sigmas[i, j, :], N, q,
                                           ue.kwargs)
    set = BoxUncertaintySet(; lb = sigma_l, ub = sigma_u, val = prr.sigma)
    return expand_investable_ucs(set, imsk, pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the [`EllipsoidalUncertaintySet`](@ref) of a bootstrap estimator from the bootstrap deviations and their sample covariance.

The shape is the sample covariance of the ``M`` rows of `X`, so its rank is at most ``M - 1``. If ``M - 1`` is less than the dimension of the set, the repaired shape is flat in the other directions. A chi-squared radius at the full dimension is then too large for the set. So the degrees of freedom are the dimension of the set, capped at ``M - 1``. This is the rank that [`norm_ball_deviation_set`](@ref) measures on the same sample, so the two routes of one bootstrap set give one radius.

# Algorithm

 1. Calculate `df = ucs_dimension(class, ue.alg.diagonal, size(X, 2), size(X, 1))`. This is the dimension of the set, capped at the number of rows of `X` minus one.
 2. Build the set with [`ellipsoidal_set`](@ref) on `ue.alg`, `ue.q`, `X`, `cov`, `class`, `val` and `df`.

# Arguments

  - `ue`: Bootstrap uncertainty set estimator. It gives the set algorithm and the significance level.
  - `X`: Bootstrap deviations, with one row per resample.
  - `cov`: Sample covariance of `X`. It becomes the shape matrix.
  - `class`: Axis tag of the set.
  - `val`: Centre of the set. It is the fitted expected returns on the mean axis, and the fitted covariance on the covariance axis.

# Returns

  - `ucs::EllipsoidalUncertaintySet`: The set.

# Related

  - [`ellipsoidal_set`](@ref)
  - [`ucs_dimension`](@ref)
  - [`norm_ball_deviation_set`](@ref)
  - [`ARCHUncertaintySet`](@ref)
"""
function bootstrap_ellipsoidal_set(ue::ARCHUncertaintySet, X::MatNum, cov::MatNum,
                                   class::AbstractUncertaintySetClass, val::ArrNum)
    df = ucs_dimension(class, ue.alg.diagonal, size(X, 2), size(X, 1))
    return ellipsoidal_set(ue.alg, ue.q, X, cov, class, val; df = df)
end
"""
    ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                               <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal uncertainty sets of the expected returns and the covariance from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

Both sets come from one pass over one index stream. The shape matrices are the empirical covariances of the bootstrap deviations, and `ue.ce` fits them. So `ue.ce` enters the covariance axis twice. It fits every resample, and then it fits the deviations of the resampled covariances. The mean axis uses `ue.ce` once, over the mean deviations.

If `ue.seed` is set, this method and the pair [`mu_ucs`](@ref) and [`sigma_ucs`](@ref) give the same sets. If `ue.seed` is `nothing`, they do not.

# Mathematical definition

The bootstrap deviations are ``\\boldsymbol{\\delta}_{\\mu}^{(m)} = \\hat{\\boldsymbol{\\mu}}^{(m)} - \\hat{\\boldsymbol{\\mu}}`` and ``\\boldsymbol{\\delta}_{\\Sigma}^{(m)} = \\operatorname{vec}(\\hat{\\mathbf{\\Sigma}}^{(m)} - \\hat{\\mathbf{\\Sigma}})``. Their empirical covariances are:

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\mu} &= \\operatorname{Cov}\\!\\left(\\boldsymbol{\\delta}_{\\mu}^{(m)}\\right)\\,, \\\\
\\mathbf{\\Sigma}_{\\Sigma} &= \\operatorname{Cov}\\!\\left(\\boldsymbol{\\delta}_{\\Sigma}^{(m)}\\right)\\,.
\\end{align}
```

The ellipsoidal sets are:

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

  - ``\\mathbf{\\Sigma}_{\\mu}``: Empirical covariance of the bootstrap deviations of the mean.
  - ``\\mathbf{\\Sigma}_{\\Sigma}``: Empirical covariance of the vectorised bootstrap deviations of the covariance.
  - ``\\hat{\\boldsymbol{\\mu}}``, ``\\hat{\\mathbf{\\Sigma}}``: Estimated mean vector and covariance matrix.
  - ``\\boldsymbol{\\delta}_{\\mu}^{(m)}``, ``\\boldsymbol{\\delta}_{\\Sigma}^{(m)}``: Bootstrap deviations of the mean and of the covariance.
  - ``\\mathcal{E}_{\\mu}``: Ellipsoidal uncertainty set of the expected returns.
  - ``\\mathcal{E}_{\\Sigma}``: Ellipsoidal uncertainty set of the covariance.
  - ``k_{\\mu}``, ``k_{\\Sigma}``: Radii of the two sets. [`k_ucs`](@ref) fits each radius from `ue.alg.method` at the level `ue.q`.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` from the reduced Prior Result. The method resamples this `X`, so under a Scenario Cap it resamples only the rows that the prior keeps.
 3. Draw the resampled statistics with [`bootstrap_generator`](@ref) to get `mus` and `sigmas` from one index stream.
 4. Subtract the `mu` of the reduced Prior Result from each column of `mus` to get `X_mu`. Subtract its vectorised `sigma` from each slice of `sigmas` to get `X_sigma`. Transpose both, so that each row is one simulation.
 5. Fit `ue.ce` on `X_mu` to get the shape matrix `sigma_mu`. This is the only fit of `ue.ce` on the mean axis. The shape is empirical, and no asymptotic formula enters.
 6. Fit `ue.ce` on `X_sigma` to get the shape matrix `sigma_sigma`. This is the second fit of `ue.ce` on the covariance axis.
 7. Build both sets with [`bootstrap_ellipsoidal_set`](@ref). It fits each radius `k` at the level `ue.q`, with the degrees of freedom capped at `ue.n_sim - 1`.
 8. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref), and return them. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.ce` fits the covariance of every resample and the shape matrix over the deviations. So it enters the covariance axis twice and the mean axis once. The centres come from `pr`, and `ue.me` and `ue.ce` fit the spread, so the centres and the spread can disagree.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.me` and `ue.ce`.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Ellipsoid of the expected returns.
  - `sigma_ucs::EllipsoidalUncertaintySet`: Ellipsoid of the covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any,
                                    <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
             rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    mus, sigmas = bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(@view(mus[:, i]) - prr.mu)
        X_sigma[:, i] = vec(@view(sigmas[:, :, i]) - prr.sigma)
    end
    X_mu = transpose(X_mu)
    X_sigma = transpose(X_sigma)
    sigma_mu = Statistics.cov(ue.ce, X_mu)
    sigma_sigma = Statistics.cov(ue.ce, X_sigma)
    mu_set, sigma_set = bootstrap_ellipsoidal_set(ue, X_mu, sigma_mu,
                                                  MuUncertaintySetClass(), prr.mu),
                        bootstrap_ellipsoidal_set(ue, X_sigma, sigma_sigma,
                                                  SigmaUncertaintySetClass(), prr.sigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                  <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal uncertainty set of the expected returns from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

The shape matrix is the empirical covariance of the bootstrap deviations of the mean, and `ue.ce` fits it. So `ue.ce` enters this axis once, although the method fits no covariance on a resample. If `ue.seed` is set, the method returns the same set as the mean set of [`ucs`](@ref). If `ue.seed` is `nothing`, it does not.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\mu} &= \\operatorname{Cov}\\!\\left(\\hat{\\boldsymbol{\\mu}}^{(m)} - \\hat{\\boldsymbol{\\mu}}\\right)\\,, \\\\
\\mathcal{E}_{\\mu} &= \\left\\{\\boldsymbol{\\mu} : (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^{\\intercal} \\mathbf{\\Sigma}_{\\mu}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq k_{\\mu}^2\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\mu}``: Empirical covariance of the bootstrap deviations of the mean.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
  - ``\\mathcal{E}_{\\mu}``: Ellipsoidal uncertainty set of the expected returns.
  - ``k_{\\mu}``: Radius of the set. [`k_ucs`](@ref) fits it from `ue.alg.method` at the level `ue.q`.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` from the reduced Prior Result. The method resamples this `X`, so under a Scenario Cap it resamples only the rows that the prior keeps.
 3. Draw the resampled means with [`mu_bootstrap_generator`](@ref) to get `mus`.
 4. Subtract the `mu` of the reduced Prior Result from each column of `mus` to get the deviation matrix `X_mu`. Transpose it, so that each row is one simulation.
 5. Fit `ue.ce` on `X_mu` to get the shape matrix `sigma_mu`. The shape is empirical, and no asymptotic formula enters.
 6. Build the set with [`bootstrap_ellipsoidal_set`](@ref). It fits the radius `k` at the level `ue.q`, with the degrees of freedom capped at `ue.n_sim - 1`.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.me` fits the mean of every resample, and `ue.ce` fits the shape matrix over the deviations.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.me`.

# Returns

  - `mu_ucs::EllipsoidalUncertaintySet`: Ellipsoid of the expected returns.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any,
                                       <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    mus = mu_bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(@view(mus[:, i]) - prr.mu)
    end
    X_mu = transpose(X_mu)
    sigma_mu = Statistics.cov(ue.ce, X_mu)
    set = bootstrap_ellipsoidal_set(ue, X_mu, sigma_mu, MuUncertaintySetClass(), prr.mu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                     <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the ellipsoidal uncertainty set of the covariance from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

The shape matrix is the empirical covariance of the bootstrap deviations of the covariance, and `ue.ce` fits it. So `ue.ce` enters this axis twice. It fits every resample, and then it fits the deviations. A `ue.ce` with no bias correction changes the resampled covariances by 0.397% over 252 observations, and the shape matrix by 1.784% over 100 resamples. If `ue.seed` is set, the method returns the same set as the covariance set of [`ucs`](@ref). If `ue.seed` is `nothing`, it does not.

# Mathematical definition

```math
\\begin{align}
\\mathbf{\\Sigma}_{\\Sigma} &= \\operatorname{Cov}\\!\\left(\\operatorname{vec}(\\hat{\\mathbf{\\Sigma}}^{(m)} - \\hat{\\mathbf{\\Sigma}})\\right)\\,, \\\\
\\mathcal{E}_{\\Sigma} &= \\left\\{\\mathbf{\\Sigma} : \\left\\lVert \\mathbf{\\Sigma}_{\\Sigma}^{-1/2} \\operatorname{vec}(\\mathbf{\\Sigma} - \\hat{\\mathbf{\\Sigma}}) \\right\\rVert_2 \\leq k_{\\Sigma}\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{\\Sigma}_{\\Sigma}``: Empirical covariance of the vectorised bootstrap deviations of the covariance.
  - $(math_dict[:Sigma_hat])
  - ``\\mathcal{E}_{\\Sigma}``: Ellipsoidal uncertainty set of the covariance.
  - ``k_{\\Sigma}``: Radius of the set. [`k_ucs`](@ref) fits it from `ue.alg.method` at the level `ue.q`.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` from the reduced Prior Result. The method resamples this `X`, so under a Scenario Cap it resamples only the rows that the prior keeps.
 3. Draw the resampled covariances with [`sigma_bootstrap_generator`](@ref) to get `sigmas`. This is the first fit of `ue.ce`.
 4. Subtract the vectorised `sigma` of the reduced Prior Result from each slice of `sigmas` to get the deviation matrix `X_sigma`. Transpose it, so that each row is one simulation.
 5. Fit `ue.ce` on `X_sigma` to get the shape matrix `sigma_sigma`. This is the second fit of `ue.ce`. The shape is empirical, and no asymptotic formula enters.
 6. Build the set with [`bootstrap_ellipsoidal_set`](@ref). It fits the radius `k` at the level `ue.q`, with the degrees of freedom capped at `ue.n_sim - 1`.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator. `ue.ce` fits the covariance of every resample and the shape matrix over the deviations, so it enters this axis twice.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.ce`.

# Returns

  - `sigma_ucs::EllipsoidalUncertaintySet`: Ellipsoid of the covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
"""
function sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any,
                                          <:EllipsoidalUncertaintySetAlgorithm, <:Any,
                                          <:Any, <:Any, <:Any, <:Any},
                   pr::AbstractPriorResult; rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    sigmas = sigma_bootstrap_generator(ue, X; kwargs...)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_sigma, 2)
        X_sigma[:, i] = vec(@view(sigmas[:, :, i]) - prr.sigma)
    end
    X_sigma = transpose(X_sigma)
    sigma_sigma = Statistics.cov(ue.ce, X_sigma)
    set = bootstrap_ellipsoidal_set(ue, X_sigma, sigma_sigma, SigmaUncertaintySetClass(),
                                    prr.sigma)
    return expand_investable_ucs(set, imsk, pr)
end

"""
    ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm, <:Any,
                               <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball uncertainty sets of the expected returns and the covariance from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

This route builds no shape matrix, because the scaled bootstrap deviations are the geometry map. The ellipsoidal route fits `ue.ce` on the deviations. On the covariance axis, the result is an ``N^{2} \\times N^{2}`` matrix of rank at most ``\\min(M - 1, N(N+1)/2)``. That matrix is rank deficient at every sample size, and the default matrix processing repairs it. The map that [`norm_ball_deviation_factor`](@ref) builds has the same second moment exactly, at the rank of the sample, and `ue.ce` does not fit it. `ue.ce` still fits the covariance of every resample, so it enters this axis once and not twice.

On the mean axis, the map has full rank if `ue.n_sim` is larger than ``N``. The two shapes then agree, and the two sets give the same optimal weights.

# Mathematical definition

```math
\\mathbf{L} = \\dfrac{\\left(\\mathbf{X} - \\boldsymbol{1}\\bar{\\mathbf{x}}^{\\intercal}\\right)^{\\intercal}}{\\sqrt{M - 1}}\\,, \\qquad \\mathbf{L}\\mathbf{L}^{\\intercal} = \\operatorname{Cov}(\\mathbf{X})\\,, \\qquad U = \\left\\{ \\hat{\\mathbf{z}} + \\mathbf{L}\\mathbf{u} \\, \\vert \\, \\lVert \\mathbf{u} \\rVert_{p} \\leq \\kappa \\right\\}\\,.
```

Where:

  - ``\\mathbf{L}``: Geometry map.
  - ``\\mathbf{X}``: Bootstrap deviations, one row per resample.
  - ``\\bar{\\mathbf{x}}``: Column means of ``\\mathbf{X}``.
  - ``M``: Number of resamples, `ue.n_sim`.
  - ``\\hat{\\mathbf{z}}``: Point estimate that the method subtracts to get the deviations.
  - ``\\kappa``, ``p``: Radius and norm order of the ball.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` and `N = size(X, 2)` from the reduced Prior Result. The method resamples this `X`.
 3. Refit both statistics on every resample with [`bootstrap_generator`](@ref) to get `mus` and `sigmas` from one index stream.
 4. Subtract the `mu` of the reduced Prior Result from every resampled mean to get `X_mu`. Subtract its `sigma` from every resampled covariance to get `X_sigma`. Each column holds one deviation.
 5. Build the two sets with [`norm_ball_deviation_set`](@ref) on the transposed deviations.
 6. Write both sets back onto the full universe with [`expand_investable_ucs`](@ref). Return them as a tuple, with the mean set first. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.me` and `ue.ce`.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Norm ball of the expected returns.
  - `sigma_ucs::NormBallUncertaintySet`: Norm ball of the covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_deviation_set`](@ref)
  - [`bootstrap_generator`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2, Listing 11.11. $(ref_dict[:no_original_source])
  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equation 50.
  - $(ref_dict[:elghaoui2003]) Theorem 1, Equation 8.
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any,
                                    <:NormBallUncertaintySetAlgorithm, <:Any, <:Any, <:Any,
                                    <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing,
             kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    mus, sigmas = bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(@view(mus[:, i]) - prr.mu)
        X_sigma[:, i] = vec(@view(sigmas[:, :, i]) - prr.sigma)
    end
    mu_set, sigma_set = norm_ball_deviation_set(ue.alg, ue.q, transpose(X_mu),
                                                MuUncertaintySetClass(), prr.mu),
                        norm_ball_deviation_set(ue.alg, ue.q, transpose(X_sigma),
                                                SigmaUncertaintySetClass(), prr.sigma)
    return expand_investable_ucs(mu_set, imsk, pr),
           expand_investable_ucs(sigma_set, imsk, pr)
end
"""
    mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm, <:Any,
                                  <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball uncertainty set of the expected returns from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

This route builds no shape matrix, because the scaled bootstrap deviations are the geometry map. `ue.ce` does not enter this axis. The ellipsoidal route fits `ue.ce` on the mean deviations, and the map has the same second moment without it. If `ue.seed` is set, the method sees the same resamples as the mean set of [`ucs`](@ref). If `ue.seed` is `nothing`, it does not.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` and `N = size(X, 2)` from the reduced Prior Result. The method resamples this `X`.
 3. Refit the mean on every resample with [`mu_bootstrap_generator`](@ref) to get `mus`.
 4. Subtract the `mu` of the reduced Prior Result from every resampled mean to get `X_mu`. Each column holds one deviation.
 5. Build the set with [`norm_ball_deviation_set`](@ref) on the transposed deviations, with the `mu` of the reduced Prior Result as the centre.
 6. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.me`.

# Returns

  - `mu_ucs::NormBallUncertaintySet`: Norm ball of the expected returns.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_deviation_set`](@ref)
  - [`mu_bootstrap_generator`](@ref)
  - [`sigma_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2, Listing 11.11. $(ref_dict[:no_original_source])
  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equation 50.
  - $(ref_dict[:elghaoui2003]) Theorem 1, Equation 8.
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any,
                                       <:NormBallUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    mus = mu_bootstrap_generator(ue, X; kwargs...)
    X_mu = Matrix{eltype(X)}(undef, N, ue.n_sim)
    for i in axes(X_mu, 2)
        X_mu[:, i] = vec(@view(mus[:, i]) - prr.mu)
    end
    set = norm_ball_deviation_set(ue.alg, ue.q, transpose(X_mu), MuUncertaintySetClass(),
                                  prr.mu)
    return expand_investable_ucs(set, imsk, pr)
end
"""
    sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm,
                                     <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits the norm-ball uncertainty set of the covariance from block bootstrap resamples of the returns.

This is the prior-result method of the verb. Only a set whose `pe` is `nothing` has it. Inside an optimiser, `pr` is the prior that the optimiser solves on. The set then has the same centre as the objective, and the method resamples the rows of that prior. Outside an optimiser, give `pr = prior(pe, X)`. The returns-data method of the same verb fits the prior of the set through [`ucs_prior`](@ref) and calls this method.

This route is the only route of the library that bounds a covariance with no matrix of side ``N^{2}``. The ellipsoidal route fits `ue.ce` on the ``M \\times N^{2}`` deviations. A vectorised symmetric matrix has only ``N(N+1)/2`` free coordinates, so that shape is rank deficient at every sample size. The default matrix processing then repairs it into a matrix that the sample does not give.

The map that [`norm_ball_deviation_factor`](@ref) builds is the scaled deviations. So it has the second moment of the sample exactly, at the rank ``\\min(M - 1, N(N+1)/2)``. [`k_norm_ball`](@ref) reads that rank, and not the side of a shape matrix. The chi-squared radius of the ellipsoidal route reads the same rank by default. It reads ``N^{2}`` degrees of freedom only if its [`ChiSqKUncertaintyAlgorithm`](@ref) has `ambient = true`.

# Algorithm

 1. Reduce the Prior Result to the Investable Mask with [`investable_ucs_reduction`](@ref). Inside an optimiser, the result is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes.
 2. Take `X` and `N = size(X, 2)` from the reduced Prior Result. The method resamples this `X`.
 3. Refit the covariance on every resample with [`sigma_bootstrap_generator`](@ref) to get `sigmas`.
 4. Subtract the `sigma` of the reduced Prior Result from every resampled covariance, and vectorise the difference, to get `X_sigma`. Each column holds one deviation.
 5. Build the set with [`norm_ball_deviation_set`](@ref) on the transposed deviations, with the `sigma` of the reduced Prior Result as the centre.
 6. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

# Arguments

  - `ue`: ARCH uncertainty set estimator.
  - `pr`: Prior Result that the set fits on. The method resamples `pr.X`.
  - `rd`: Returns Result that the three-argument form gives beside the prior. The reduction takes a view of it, and the method does not use the view.
  - `kwargs...`: More keyword arguments, which the method gives to `ue.ce`.

# Returns

  - `sigma_ucs::NormBallUncertaintySet`: Norm ball of the covariance.

# Related

  - [`ARCHUncertaintySet`](@ref)
  - [`NormBallUncertaintySetAlgorithm`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_deviation_set`](@ref)
  - [`sigma_bootstrap_generator`](@ref)
  - [`mu_ucs`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 11.3.2, Listing 11.11. $(ref_dict[:no_original_source])
  - $(ref_dict[:goldfarbiyengar2003]) Section 5, Equation 50.
  - $(ref_dict[:elghaoui2003]) Theorem 1, Equation 8.
  - $(ref_dict[:bertsimas2011]) Section 2.2, Theorem 2. $(ref_dict[:no_original_source])
  - $(ref_dict[:bertsimas2004])
"""
function sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any,
                                          <:NormBallUncertaintySetAlgorithm, <:Any, <:Any,
                                          <:Any, <:Any, <:Any}, pr::AbstractPriorResult;
                   rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    X = prr.X
    N = size(X, 2)
    sigmas = sigma_bootstrap_generator(ue, X; kwargs...)
    X_sigma = Matrix{eltype(X)}(undef, N^2, ue.n_sim)
    for i in axes(X_sigma, 2)
        X_sigma[:, i] = vec(@view(sigmas[:, :, i]) - prr.sigma)
    end
    set = norm_ball_deviation_set(ue.alg, ue.q, transpose(X_sigma),
                                  SigmaUncertaintySetClass(), prr.sigma)
    return expand_investable_ucs(set, imsk, pr)
end

export StationaryBootstrap, CircularBootstrap, MovingBootstrap, ARCHUncertaintySet
public BootstrapUncertaintySetEstimator, ARCHBootstrapSet, bootstrap_indices
