"""
$(DocStringExtensions.TYPEDEF)

Names how a regime statistic of standardised squared returns becomes the multiplier of a volatility.

The estimator scores each observation against the variance before it. The method turns the scores
of one observation into one statistic. The regime state smooths that statistic, and the method
turns the state into a volatility multiplier. This is the Volatility Regime Adjustment of
[menchero2011](@cite), and [`RootMeanSquaredAdjusted`](@ref) is its statistic. The other methods
replace the mean square with a statistic that one outlier moves less. They are the Short-Term
Volatility Updating of [paleologo2025](@cite).

# Interfaces

To implement a new regime adjustment method, subtype `RegimeAdjustedMethod` and implement the
methods below. [`RegimeAdjustedExpWeightedVariance`](@ref) calls `regime_multiplier` and the first
form of `get_regime_state`, and [`RegimeAdjustedExpWeightedCovariance`](@ref) calls
`regime_multiplier` and the second form.

A new method takes `debias = RawStatistic()`. [`ExactDebias`](@ref) and [`LawDebias`](@ref)
divide the statistic by the bias of the moment that the method reads. The library computes that
bias for its own three methods alone, so the two constructors refuse a new method under either
rule.

## `regime_multiplier`

  - `regime_multiplier(method::MyRegimeMethod, regime_state::Number) -> Number`: Turns the
    smoothed regime state into the volatility multiplier. The estimator scales its variance or its
    covariance by the square of the multiplier.

## `get_regime_state`

  - `get_regime_state(method::MyRegimeMethod, z2::VecNum, min_val::Number) -> Number`: Turns the
    standardised squared returns of the scored assets of one observation into the statistic that
    the regime state smooths.
  - `get_regime_state(method::MyRegimeMethod, target::RegimeAdjustedTarget, stats::VecNum, n::Integer, min_val::Number) -> VecNum`:
    Turns each statistic of the target, over `n` scored assets, into the value that the regime
    state smooths. The estimator smooths the mean of the values.

### Examples

```jldoctest
julia> struct MyRegimeMethod <: PortfolioOptimisers.RegimeAdjustedMethod end

julia> PortfolioOptimisers.regime_multiplier(::MyRegimeMethod, s::Number) = sqrt(max(s, 0));

julia> function PortfolioOptimisers.get_regime_state(::MyRegimeMethod, z2::AbstractVector,
                                                     ::Number)
           return sum(z2) / length(z2)
       end;

julia> function PortfolioOptimisers.get_regime_state(::MyRegimeMethod,
                                                     ::PortfolioOptimisers.RegimeAdjustedTarget,
                                                     stats::AbstractVector, n::Integer, ::Number)
           return stats ./ n
       end;

julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03];

julia> kw = (; decay = 0.9, min_obs = 2, regime_min_obs = 1, debias = RawStatistic());

julia> var(RegimeAdjustedExpWeightedVariance(; regime_method = MyRegimeMethod(), kw...), X) ≈
       var(RegimeAdjustedExpWeightedVariance(; regime_method = RootMeanSquaredAdjusted(), kw...),
           X)
true

julia> cov(RegimeAdjustedExpWeightedCovariance(; regime_method = MyRegimeMethod(), kw...), X) ≈
       cov(RegimeAdjustedExpWeightedCovariance(; regime_method = RootMeanSquaredAdjusted(), kw...),
           X)
true

julia> RegimeAdjustedExpWeightedVariance(; regime_method = MyRegimeMethod())
ERROR: ArgumentError: `RegimeAdjustedExpWeightedVariance.regime_method` is a `MyRegimeMethod`, and `RegimeAdjustedExpWeightedVariance.debias` is `ExactDebias()`. That rule divides the regime statistic by the bias of the moment that the method reads, and the library computes that bias for `RootMeanSquaredAdjusted`, `FirstMomentRegimeAdjusted` and `LogRegimeAdjusted` alone. Set `debias = RawStatistic()` for this method.
[...]
```

# Related

  - [`LogRegimeAdjusted`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)

# References

  - $(ref_dict[:menchero2011]) Section 4.3.
  - $(ref_dict[:paleologo2025]) Section 6.3.3, Model 6.8 and Equations 6.9 to 6.11.
"""
abstract type RegimeAdjustedMethod <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Scales the volatility by the exponential of half the smoothed mean log of the standardised squared returns.

The log compresses a large square, so one outlier moves this multiplier less than it moves the
multiplier of [`RootMeanSquaredAdjusted`](@ref). The constant ``\\kappa`` is the expectation of
``\\ln z^{2}`` when ``z^{2}`` is a ``\\mathrm{Gamma}(x, y)`` variate. The defaults ``x = 1/2`` and
``y = 2`` make ``z^{2}`` a ``\\chi^{2}(1)`` variate, the square of a standard normal return. The
regime state is then a log variance ratio for every law, so the volatility multiplier is the
exponential of half the state.

This is the Short-Term Volatility Updating of [paleologo2025](@cite), Section 6.3.3. Model 6.8
scales a correctly calibrated return by ``e^{x_{t}/2}``, so ``\\ln z^{2} - \\kappa`` is an
unbiased observation of the latent log variance ratio ``x_{t}``. The smoothed state is the
estimate ``\\hat{x}_{t}`` of the book, and the multiplier is its ``e^{\\hat{x}_{t}/2}``. The book
reads the squared norm of the whole standardised vector of ``m`` returns, with
``\\kappa = \\psi(m/2) + \\ln 2``. This method reads each asset at ``m = 1`` and takes the mean of
the values. [`RegimeAdjustedExpWeightedCovariance`](@ref) with a [`MahalanobisTarget`](@ref)
reads the norm of the book.

# Mathematical definition

```math
\\begin{align}
\\kappa &= \\psi(x) + \\ln y\\,, \\\\
g_{t} &= \\frac{1}{n_{t}} \\sum_{i \\in \\mathcal{A}_{t}} \\ln \\max(z_{i,t}^{2}, \\varepsilon) - \\kappa\\,, \\\\
\\mathrm{mult}(s) &= \\exp(s / 2)\\,.
\\end{align}
```

Where:

  - ``\\kappa``: Expectation of ``\\ln z^{2}`` for a ``\\mathrm{Gamma}(x, y)`` variate.
  - ``\\psi``: Digamma function.
  - ``x``, ``y``: Shape and scale of the law of ``z^{2}``, the fields of the same names.
  - $(math_dict[:g_t_regime])
  - $(math_dict[:A_t_regime])
  - $(math_dict[:z2_it_regime])
  - ``\\varepsilon``: The `min_val` field of the estimator.
  - $(math_dict[:mult_regime])

[`RegimeAdjustedExpWeightedVariance`](@ref) states ``\\mathcal{A}_{t}``, ``z_{i,t}^{2}`` and the
smoothing of ``g_{t}``. [`RegimeAdjustedExpWeightedCovariance`](@ref) replaces the mean over the
assets with the log of the statistic of its [`RegimeAdjustedTarget`](@ref), and ``\\kappa`` with
the expectation of that log.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogRegimeAdjusted(;
        x::Number = 0.5,
        y::Number = 2.0
    ) -> LogRegimeAdjusted

Keywords correspond to the struct's fields. The constructor derives the `kappa` field from `x`
and `y`, so a caller cannot set it.

## Validation

  - $(val_dict[:ra_x])
  - $(val_dict[:ra_y])

# Examples

```jldoctest
julia> LogRegimeAdjusted()
LogRegimeAdjusted
      x ┼ Float64: 0.5
      y ┼ Float64: 2.0
  kappa ┴ Float64: -1.2703628454614782
```

# Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)

# References

  - $(ref_dict[:paleologo2025]) Section 6.3.3, pages 122 and 123, Model 6.8 and the estimate of the state.
"""
@concrete struct LogRegimeAdjusted <: RegimeAdjustedMethod
    """
    $(field_dict[:ra_x])
    """
    x
    """
    $(field_dict[:ra_y])
    """
    y
    """
    $(field_dict[:ra_kappa])
    """
    kappa
    function LogRegimeAdjusted(x::Number, y::Number)
        assert_nonempty_gt0_finite_val(x, :x)
        assert_nonempty_gt0_finite_val(y, :y)
        kappa = SpecialFunctions.digamma(x) + log(y)
        return new{typeof(x), typeof(y), typeof(kappa)}(x, y, kappa)
    end
end
function LogRegimeAdjusted(; x::Number = 0.5, y::Number = 2.0)::LogRegimeAdjusted
    return LogRegimeAdjusted(x, y)
end
"""
$(DocStringExtensions.TYPEDEF)

Scales the volatility by the smoothed mean absolute standardised return over its expectation `x`.

The absolute value grows linearly in a return, so one outlier moves this multiplier less than it
moves the multiplier of [`RootMeanSquaredAdjusted`](@ref) and more than it moves the multiplier of
[`LogRegimeAdjusted`](@ref). The default ``x = \\sqrt{2/\\pi}`` is the expectation of ``|z|`` for a
standard normal ``z``.

This is Equation 6.11 of [paleologo2025](@cite), the linear form of the Short-Term Volatility
Updating. The book divides the norm of the standardised vector of ``m`` returns by ``\\sqrt{m}``,
the limit of its expectation as ``m`` grows. This method reads each asset at ``m = 1`` and
divides by the exact expectation ``x``, 0.798 by default, where the book divides by 1.
[`RegimeAdjustedExpWeightedCovariance`](@ref) divides the root of the statistic of its target by
the exact expectation of that root. At ``m = 5`` this is 2.128 against ``\\sqrt{5} = 2.236``.

# Mathematical definition

```math
\\begin{align}
g_{t} &= \\frac{1}{x\\, n_{t}} \\sum_{i \\in \\mathcal{A}_{t}} \\sqrt{z_{i,t}^{2}}\\,, \\\\
\\mathrm{mult}(s) &= s\\,.
\\end{align}
```

Where:

  - ``x``: Expectation of ``|z|`` under correct calibration, the field of the same name.
  - $(math_dict[:g_t_regime])
  - $(math_dict[:A_t_regime])
  - $(math_dict[:z2_it_regime])
  - $(math_dict[:mult_regime])

[`RegimeAdjustedExpWeightedVariance`](@ref) states ``\\mathcal{A}_{t}``, ``z_{i,t}^{2}`` and the
smoothing of ``g_{t}``. [`RegimeAdjustedExpWeightedCovariance`](@ref) replaces the mean over the
assets with the root of the statistic of its [`RegimeAdjustedTarget`](@ref), and ``x`` with the
expectation of that root.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FirstMomentRegimeAdjusted(;
        x::Number = sqrt(2 * inv(π))
    ) -> FirstMomentRegimeAdjusted

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ra_x])

# Examples

```jldoctest
julia> FirstMomentRegimeAdjusted()
FirstMomentRegimeAdjusted
  x ┴ Float64: 0.7978845608028654
```

# Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`LogRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)

# References

  - $(ref_dict[:paleologo2025]) Section 6.3.3, page 124, Equation 6.11.
"""
@concrete struct FirstMomentRegimeAdjusted <: RegimeAdjustedMethod
    """
    $(field_dict[:ra_norm_x])
    """
    x
    function FirstMomentRegimeAdjusted(x::Number)
        assert_nonempty_gt0_finite_val(x, :x)
        return new{typeof(x)}(x)
    end
end
function FirstMomentRegimeAdjusted(;
                                   x::Number = sqrt(2 * inv(pi)))::FirstMomentRegimeAdjusted
    return FirstMomentRegimeAdjusted(x)
end
"""
$(DocStringExtensions.TYPEDEF)

Scales the volatility by the root of the smoothed mean of the standardised squared returns.

This is the Volatility Regime Adjustment of [menchero2011](@cite). The statistic of one
observation is the square of the cross-sectional bias statistic of Equation 4.3, and the
multiplier is the root of its exponentially weighted mean, Equation 4.4. The specific-risk form,
Equations 5.10 and 5.11, weighs each asset by its capitalisation, and this method weighs each
asset equally. The recursion of the regime state starts at the first statistic. The square
grows fastest in a return, so one outlier moves this multiplier more than it moves the multipliers
of [`FirstMomentRegimeAdjusted`](@ref) and [`LogRegimeAdjusted`](@ref).

# Mathematical definition

```math
\\begin{align}
g_{t} &= \\frac{1}{n_{t}} \\sum_{i \\in \\mathcal{A}_{t}} z_{i,t}^{2}\\,, \\\\
\\mathrm{mult}(s) &= \\sqrt{\\max(s, 0)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:g_t_regime])
  - $(math_dict[:A_t_regime])
  - $(math_dict[:z2_it_regime])
  - $(math_dict[:mult_regime])

[`RegimeAdjustedExpWeightedVariance`](@ref) states ``\\mathcal{A}_{t}``, ``z_{i,t}^{2}`` and the
smoothing of ``g_{t}``. [`RegimeAdjustedExpWeightedCovariance`](@ref) replaces ``g_{t}`` with the
statistic of its [`RegimeAdjustedTarget`](@ref) over the count of the assets.

# Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`LogRegimeAdjusted`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)

# References

  - $(ref_dict[:menchero2011]) Section 4.3, Equations 4.3 to 4.5, and Section 5.3, Equations 5.10 to 5.12.
"""
struct RootMeanSquaredAdjusted <: RegimeAdjustedMethod end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the volatility multiplier of the log regime adjustment method.

The regime state is a log variance ratio for every law of the squares, so the multiplier takes
half of it. The estimator scales the variance by the square of the multiplier.

# Arguments

  - `::LogRegimeAdjusted`: Log regime adjustment method (unused).
  - `regime_state::Number`: Current smoothed regime state.

# Returns

  - `mult::Number`: Volatility multiplier `exp(regime_state / 2)`.

# Related

  - [`LogRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_multiplier(::LogRegimeAdjusted, regime_state::Number)
    # The state is a log variance ratio whatever the law, so the volatility takes half of it.
    return exp(regime_state / 2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the volatility multiplier of the first-moment regime adjustment method.

# Arguments

  - `::FirstMomentRegimeAdjusted`: First-moment regime adjustment method (unused).
  - `regime_state::Number`: Current smoothed regime state.

# Returns

  - `mult::Number`: Volatility multiplier, equal to `regime_state`.

# Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_multiplier(::FirstMomentRegimeAdjusted, regime_state::Number)
    return regime_state
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the volatility multiplier of the root-mean-squared regime adjustment method.

# Arguments

  - `::RootMeanSquaredAdjusted`: Root-mean-squared regime adjustment method (unused).
  - `regime_state::Number`: Current smoothed regime state.

# Returns

  - `mult::Number`: Volatility multiplier `sqrt(max(regime_state, 0))`.

# Related

  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_multiplier(::RootMeanSquaredAdjusted, regime_state::Number)
    return sqrt(max(regime_state, zero(regime_state)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Checks that the bias correction of `debias` can read the moment of `regime_method`.

[`ExactDebias`](@ref) and [`LawDebias`](@ref) divide the regime statistic by the bias of the
moment that the method reads, and the library computes that bias for its own three methods
alone. A regime method that a caller defines therefore takes [`RawStatistic`](@ref). The library
methods, and a `regime_method` of `nothing`, take every rule.

# Arguments

  - `regime_method::Option{<:RegimeAdjustedMethod}`: Regime adjustment method of the estimator.
  - `debias::AbstractRegimeDebias`: Bias correction of the estimator.
  - `name::Symbol`: Name of the estimator, for the message.

# Validation

  - Where `regime_method` is not a library method, `debias` corrects no bias. An
    `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function assert_regime_debias(regime_method::Option{<:RegimeAdjustedMethod},
                              debias::AbstractRegimeDebias, name::Symbol)
    # A `Union` whose members carry a `<:` bound is built again on every call, and an online
    # step reaches this check through the constructor. Each member here is a `UnionAll` over
    # all its parameters, so the plain `Union` admits the same methods (#1824).
    @argcheck(isnothing(regime_method) ||
              isa(regime_method,
                  Union{RootMeanSquaredAdjusted, FirstMomentRegimeAdjusted,
                        LogRegimeAdjusted}) ||
              !debiases(debias),
              ArgumentError("`$name.regime_method` is a `$(nameof(typeof(regime_method)))`, and `$name.debias` is `$(nameof(typeof(debias)))()`. That rule divides the regime statistic by the bias of the moment that the method reads, and the library computes that bias for `RootMeanSquaredAdjusted`, `FirstMomentRegimeAdjusted` and `LogRegimeAdjusted` alone. Set `debias = RawStatistic()` for this method."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates the exponentially weighted variance of each asset, and scales it by one regime multiplier.

At each observation the estimator scores each asset by its squared return over the variance
before it, `z²`. Then it steps the variance. The regime state smooths a statistic of the scores
with `regime_decay`. The report scales each variance by the square of
`regime_multiplier(regime_method, regime_state)`.

A `regime_method` of `nothing` turns the adjustment off. No regime state advances, so the
multiplier stays at one, and the estimator is the plain exponentially weighted recursion. A
consumer that reads a volatility, and not a risk figure that the regime scales, needs this form.
[`EWVolatility`](@ref) is that consumer in the library.

Under `debias = ExactDebias()`, the default, the estimator divides `z²` by the bias that the
inverse of an estimated variance puts in the moment that the method reads. It does not score an
estimate of four terms or fewer. Under `debias = RawStatistic()` it scores the raw `z²`.

Under `centring = EstimatedCentring()`, the default, the estimator takes each deviation from the
normalised location of the returns before it. It divides each square by its exact factor, so the
variance is unbiased from the second valid return whatever the mean. Under HAC the factor also
reads the mean of each lagged product. Under `centring = PreCentred()` each return is its own
deviation.

The estimator is mask-aware. A prior fitted with it keeps a young asset investable, and
[`scenario_fill`](@ref) writes a zero in each row that the asset was missing. Every consumer of a
Prior Result reads its returns matrix. A measure that reads the scenarios then reads a zero return
where the asset had none, and it understates the risk of that asset over those rows. The variance
stays the estimate that this recursion made from the rows that it saw.

The `fill_limit` field of the fitting prior is a share of the observations of the asset. A fill at
or below it is silent, a fill above it warns, and `strict` refuses every fill. `fill_limit`
defaults to `nothing`, and this family carries no `CoveragePolicy` to derive a limit from, so the
fit warns about every fill.

The estimator is the Volatility Regime Adjustment of specific risk of [menchero2011](@cite), on
an exponentially weighted variance. The log and the first-moment methods are the Short-Term
Volatility Updating of [paleologo2025](@cite). The HAC adjustment is the estimator of
[neweywest1987](@cite), applied to each term of the recursion.

# Mathematical definition

The variance of each asset steps on the observations where its return is valid:

```math
\\begin{align}
q_{i,t} &= u_{i,t}^{2} + 2 \\sum_{j=1}^{L} \\left(1 - \\frac{j}{L + 1}\\right) u_{i,t}\\, u_{i,t-j}\\,, \\\\
v_{i,t} &= \\lambda v_{i,t-1} + (1 - \\lambda)\\, q_{i,t}\\,.
\\end{align}
```

Each observation scores the assets of ``\\mathcal{A}_{t}`` against the variance before it, and the
regime state smooths the statistic of the method:

```math
\\begin{align}
z_{i,t}^{2} &= \\frac{u_{i,t}^{2}}{\\beta(K_{i,t})\\, v_{i,t-1} / (1 - \\lambda^{K_{i,t}})}\\,, \\\\
s_{t} &= \\lambda_{r} s_{t-1} + (1 - \\lambda_{r})\\, g_{t}\\,, \\quad s_{t_{0}} = g_{t_{0}}\\,.
\\end{align}
```

The report divides each variance by the weight that its zero seed leaves, and scales it by the
square of the multiplier:

```math
\\begin{align}
\\hat{\\sigma}_{i}^{2} &= m^{2}\\, \\frac{\\max(v_{i,T}, 0)}{1 - \\lambda^{K_{i,T}}}\\,, \\quad
m = \\mathrm{clamp}(\\mathrm{mult}(s_{T}), l, h)\\,.
\\end{align}
```

Where:

  - ``u_{i,t}``: Deviation of the return of asset ``i`` at observation ``t``. Under
    [`EstimatedCentring`](@ref) it is taken from the location of the returns before ``t``, the
    first return gives no term, and each square, in ``q_{i,t}`` and in ``z_{i,t}^{2}``, is divided
    by its exact factor, as [`ExpWeightedVariance`](@ref) states. Under [`PreCentred`](@ref) it is
    the return.
  - ``q_{i,t}``: Term of the recursion. Where `hac_lags` is `nothing`, ``L = 0`` and the term is
    ``u_{i,t}^{2}``. A lag that reaches before the first observation of the asset gives no term.
  - ``L``: `hac_lags`.
  - ``v_{i,t}``: Raw exponentially weighted variance of asset ``i``, seeded at zero.
  - ``\\lambda``: `decay`.
  - $(math_dict[:A_t_regime]) An asset is scored where its return is valid, where it is in the
    estimation mask, where ``K_{i,t}`` is at least `min_obs`, and where ``v_{i,t-1}`` is at least
    `min_val`.
  - $(math_dict[:z2_it_regime])
  - ``K_{i,t}``: Count of the terms in ``v_{i,t-1}``.
  - ``\\beta(K)``: Bias factor of [`regime_bias_table`](@ref) for the moment that the method
    reads, or one under [`RawStatistic`](@ref). The inverse of an estimated variance is too large
    on average (Jensen's inequality). Without the factor, the mean of ``z_{i,t}^{2}`` is 1.07 at a
    half-life of 10 and 1.017 at a half-life of 40. The factor is exact for this statistic. Under
    [`ExactDebias`](@ref) the estimator does not score an estimate of four terms or fewer, because
    the variance of ``z_{i,t}^{2}`` is not finite there. A HAC estimate is a quadratic form with a
    banded weight matrix, and the table reads the spectrum of that matrix, so the factor is exact
    for it too. The estimator does not score an estimate with too few effective observations. On
    10 independent normal assets over 4000 rows, at two lags and a half-life of 10, the mean
    squared multiplier of `RootMeanSquaredAdjusted` over 8 seeds is 0.999 with the factor and
    1.140 without it.
  - $(math_dict[:g_t_regime]) The [`RegimeAdjustedMethod`](@ref) states it from the
    ``z_{i,t}^{2}`` of ``\\mathcal{A}_{t}``. An observation that scores no asset leaves the
    state as it is.
  - $(math_dict[:s_t_regime]) ``t_{0}`` is the first observation that scores an asset.
  - $(math_dict[:lambda_r_regime])
  - ``\\hat{\\sigma}_{i}^{2}``: Regime-adjusted variance of asset ``i``. It is `NaN` where
    ``K_{i,T}`` is below `min_obs`, or where the asset is not active.
  - ``m``: Multiplier of the report. It is one while fewer than `regime_min_obs` observations
    have scored an asset, and where `regime_method` is `nothing`.
  - $(math_dict[:mult_regime])
  - ``l``, ``h``: The two bounds of `regime_lohi_mult`. Where that field is `nothing`, no clamp
    runs.
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegimeAdjustedExpWeightedVariance(;
        decay::Number             = exp2(-inv(40.0)),
        min_obs::Integer          = round(Int, max(1, decay_half_life(decay))),
        hac_lags::Option{<:Integer} = nothing,
        hac_floor::AbstractHacFloor = NoHacFloor(),
        regime_method::Option{<:RegimeAdjustedMethod} = FirstMomentRegimeAdjusted(),
        regime_decay::Number      = exp2(-2 / decay_half_life(decay)),
        regime_min_obs::Integer   = round(Int, max(1, decay_half_life(decay) / 2)),
        regime_lohi_mult::Option{<:Tuple{Number, Number}} = (0.7, 1.6),
        min_val::Number           = 1e-12,
        centring::AbstractCentring = EstimatedCentring(),
        debias::AbstractRegimeDebias = ExactDebias(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> RegimeAdjustedExpWeightedVariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - $(val_dict[:hac_lags])
  - $(val_dict[:regime_decay])
  - $(val_dict[:regime_min_obs])
  - $(val_dict[:regime_lohi_mult])

# Examples

```jldoctest
julia> ce = RegimeAdjustedExpWeightedVariance();

julia> ce.decay ≈ exp2(-inv(40.0))
true

julia> ce.min_obs
40
```

# Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`LogRegimeAdjusted`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`partial_fit!`](@ref)
  - [`scenario_fill`](@ref)
  - [`EmpiricalPrior`](@ref)

# References

  - $(ref_dict[:menchero2011]) Section 5.3, Equations 5.10 to 5.12.
  - $(ref_dict[:paleologo2025]) Section 6.3.3, Model 6.8 and Equations 6.9 to 6.11.
  - $(ref_dict[:neweywest1987]) Equation 5.
"""
@concrete struct RegimeAdjustedExpWeightedVariance <: AbstractVarianceEstimator
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    $(field_dict[:hac_lags])
    """
    hac_lags
    """
    $(field_dict[:hac_floor])
    """
    hac_floor
    """
    $(field_dict[:regime_method])
    """
    regime_method
    """
    $(field_dict[:regime_decay])
    """
    regime_decay
    """
    $(field_dict[:regime_min_obs])
    """
    regime_min_obs
    """
    $(field_dict[:regime_lohi_mult])
    """
    regime_lohi_mult
    """
    $(field_dict[:min_val])
    """
    min_val
    """
    $(field_dict[:centring])
    """
    centring
    """
    $(field_dict[:ra_debias])
    """
    debias
    """
    Running state of an incremental fit, or `nothing` before the first call to [`partial_fit!`](@ref). It is the one Result this estimator holds, and its type bound is the enforcement of that exception. [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance)`](@ref) reads it, and a fit over a matrix ignores it.
    """
    cache
    function RegimeAdjustedExpWeightedVariance(decay::Number, min_obs::Integer,
                                               hac_lags::Option{<:Integer},
                                               hac_floor::AbstractHacFloor,
                                               regime_method::Option{<:RegimeAdjustedMethod},
                                               regime_decay::Number,
                                               regime_min_obs::Integer,
                                               regime_lohi_mult::Option{<:Tuple{Number,
                                                                                Number}},
                                               min_val::Number, centring::AbstractCentring,
                                               debias::AbstractRegimeDebias,
                                               cache::Option{<:AbstractPartialFitState})
        assert_unit_interval(decay, :decay)
        assert_unit_interval(regime_decay, :regime_decay)
        assert_regime_debias(regime_method, debias, :RegimeAdjustedExpWeightedVariance)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_nonempty_gt0_finite_val(regime_min_obs, :regime_min_obs)
        if !isnothing(regime_lohi_mult)
            @argcheck(zero(regime_lohi_mult[1]) < regime_lohi_mult[1] < regime_lohi_mult[2],
                      DomainError(regime_lohi_mult,
                                  "`RegimeAdjustedExpWeightedVariance.regime_lohi_mult` is $regime_lohi_mult, and it clamps the regime multiplier to `(lo, hi)`, so the pair must satisfy `0 < lo < hi`. State such a pair, or `nothing` for no clamp."))
        end
        if !isnothing(hac_lags)
            assert_nonempty_gt0_finite_val(hac_lags, :hac_lags)
        end
        return new{typeof(decay), typeof(min_obs), typeof(hac_lags), typeof(hac_floor),
                   typeof(regime_method), typeof(regime_decay), typeof(regime_min_obs),
                   typeof(regime_lohi_mult), typeof(min_val), typeof(centring),
                   typeof(debias), typeof(cache)}(decay, min_obs, hac_lags, hac_floor,
                                                  regime_method, regime_decay,
                                                  regime_min_obs, regime_lohi_mult, min_val,
                                                  centring, debias, cache)
    end
end
# `Tuple` is covariant, so `Tuple{Number, Number}` admits every pair of numbers. A form with a
# `<:` bound on each parameter makes each keyword call build its type again, and an online step
# rebuilds the estimator through this constructor (`rebuild_estimator`).
function RegimeAdjustedExpWeightedVariance(; decay::Number = exp2(-inv(40.0)),
                                           min_obs::Integer = round(Int,
                                                                    max(1,
                                                                        decay_half_life(decay))),
                                           hac_lags::Option{<:Integer} = nothing,
                                           hac_floor::AbstractHacFloor = NoHacFloor(),
                                           regime_method::Option{<:RegimeAdjustedMethod} = FirstMomentRegimeAdjusted(),
                                           regime_decay::Number = exp2(-2 /
                                                                       decay_half_life(decay)),
                                           regime_min_obs::Integer = round(Int,
                                                                           max(1,
                                                                               decay_half_life(decay) /
                                                                               2)),
                                           regime_lohi_mult::Option{<:Tuple{Number, Number}} = (0.7,
                                                                                                1.6),
                                           min_val::Number = 1e-12,
                                           centring::AbstractCentring = EstimatedCentring(),
                                           debias::AbstractRegimeDebias = ExactDebias(),
                                           cache::Option{<:AbstractPartialFitState} = nothing)::RegimeAdjustedExpWeightedVariance
    return RegimeAdjustedExpWeightedVariance(decay, min_obs, hac_lags, hac_floor,
                                             regime_method, regime_decay, regime_min_obs,
                                             regime_lohi_mult, min_val, centring, debias,
                                             cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the running state of an incremental fit of a [`RegimeAdjustedExpWeightedVariance`](@ref).

[`partial_fit!`](@ref) builds it and keeps it in the `cache` field of the estimator, and
`var(ce, state)` reads it. A caller can hold a state outside the estimator, but only the forward
pass of the estimator writes it. The fold mutates the arrays in place, and it builds a new state
with the two new scalar fields, `regime_state` and `n_regime_obs`, through the positional
constructor.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
@concrete struct RegimeAdjustedVarianceState <: AbstractPartialFitState
    """
    $(field_dict[:ret_buffer])
    """
    ret_buffer
    """
    $(field_dict[:ra_variance])
    """
    variance
    """
    $(field_dict[:ra_X2])
    """
    X2
    """
    $(field_dict[:ra_X_old_i])
    """
    X_old_i
    """
    $(field_dict[:ra_z2])
    """
    z2
    """
    $(field_dict[:ra_location])
    """
    location
    """
    $(field_dict[:obs_count])
    """
    obs_count
    """
    $(field_dict[:old_obs_count])
    """
    old_obs_count
    """
    $(field_dict[:ra_active])
    """
    active
    """
    $(field_dict[:regime_state])
    """
    regime_state
    """
    $(field_dict[:n_regime_obs])
    """
    n_regime_obs
    """
    $(field_dict[:ra_bias])
    """
    bias
    """
    $(field_dict[:ra_lag_records])
    """
    lag_records
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the statistic ``g_{t}`` of the root-mean-squared adjustment, which the regime state smooths.

# Arguments

  - `::RootMeanSquaredAdjusted`: Root-mean-squared regime adjustment method (unused).
  - `z2_valid::VecNum`: Vector of valid (non-NaN) standardised squared returns.
  - `::Any`: Ignored minimum value argument.

# Returns

  - `s::Number`: Mean of `z2_valid`.

# Related

  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function get_regime_state(::RootMeanSquaredAdjusted, z2_valid::VecNum, ::Any)
    return Statistics.mean(z2_valid)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the statistic ``g_{t}`` of the first-moment adjustment, which the regime state smooths.

# Arguments

  - `method::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `z2_valid::VecNum`: Vector of valid (non-NaN) standardised squared returns.
  - `::Any`: Ignored minimum value argument.

# Returns

  - `s::Number`: Mean absolute standardised return over its expectation, `mean(sqrt(max.(z², 0))) / method.x`.

# Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function get_regime_state(method::FirstMomentRegimeAdjusted, z2_valid::VecNum, ::Any)
    return Statistics.mean(z -> sqrt(max(z, zero(eltype(z2_valid)))), z2_valid) / method.x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the statistic ``g_{t}`` of the log adjustment, which the regime state smooths.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `z2_valid::VecNum`: Vector of valid (non-NaN) standardised squared returns.
  - `min_val::Number`: Floor of each square before its log.

# Returns

  - `s::Number`: Mean log of the squares less its expectation, `mean(log(max.(z², min_val))) - method.kappa`.

# Related

  - [`LogRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function get_regime_state(method::LogRegimeAdjusted, z2_valid::VecNum,
                          min_val::Number = sqrt(eps(eltype(z2_valid))))
    log_z2 = log.(max.(z2_valid, min_val))
    return Statistics.mean(log_z2) - method.kappa
end
"""
    blend_regime_state(prev::Nothing, new::Number, decay::Number) -> Number
    blend_regime_state(prev::Number, new::Number, decay::Number) -> Number

Blends the regime state of the last step with the regime state of one new observation.

[`process_observation!`](@ref) and the regime update of [`RegimeAdjustedExpWeightedCovariance`](@ref) call it. The first step that advances the regime has no state before it, so `prev` is `nothing`, and the function returns `new`. A later step weights `prev` by `decay` and `new` by one minus `decay`.

# Arguments

  - `prev`: The regime state of the last step, or `nothing` before the first step.
  - `new`: The regime state of the new observation.
  - `decay`: The decay of the regime state, the `regime_decay` of the estimator.

# Returns

  - `s::Number`: The blended regime state.

# Related

  - [`get_regime_state`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
blend_regime_state(::Nothing, new::Number, ::Number) = new
function blend_regime_state(prev::Number, new::Number, decay::Number)
    return decay * prev + (one(decay) - decay) * new
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor that the root-mean-squared adjustment reads, ``\\mathbb{E}[Q^{-1}]``.

`Q` is the estimated variance of one return over its true variance, and
[`regime_bias_table`](@ref) states it. The method smooths ``z^{2} = u^{2} / \\hat{v}`` and its
calibration assumes ``\\mathbb{E}[z^{2}] = 1``, but ``\\mathbb{E}[z^{2}] = \\mathbb{E}[Q^{-1}]``.

# Arguments

  - `::RootMeanSquaredAdjusted`: Root-mean-squared regime adjustment method.
  - `s::VecNum`: Grid of the integration variable ``s = e^{x}``.
  - `G::VecNum`: ``G(s)``, the Laplace transform of `Q` on the grid.
  - `c::Number`: Normalised weight of the newest observation.
  - `h::Number`: Step of the grid in ``x``.

# Returns

  - `factor::Number`: ``\\int_{0}^{\\infty} G(s)\\, \\mathrm{d}s / (2 c)``.

# Related

  - [`regime_bias_table`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
"""
function regime_bias_factor(::RootMeanSquaredAdjusted, s::VecNum, G::VecNum, c::Number,
                            h::Number)
    return h * sum(i -> G[i] * s[i], eachindex(s, G)) / (2 * c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor that the first-moment adjustment reads, ``\\mathbb{E}[Q^{-1/2}]^{2}``.

The method smooths ``|z| = |u| / \\sqrt{\\hat{v}}``, and ``\\mathbb{E}[|z|] = \\mathbb{E}[|u|]\\, \\mathbb{E}[Q^{-1/2}]``, so the statistic ``z^{2}`` is divided by the square of that moment.

# Arguments

  - `::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `s::VecNum`: Grid of the integration variable ``s = e^{x}``.
  - `G::VecNum`: ``G(s)``, the Laplace transform of `Q` on the grid.
  - `c::Number`: Normalised weight of the newest observation.
  - `h::Number`: Step of the grid in ``x``.

# Returns

  - `factor::Number`: ``\\left(\\int_{0}^{\\infty} s^{-1/2} G(s)\\, \\mathrm{d}s\\right)^{2} / (2 \\pi c)``.

# Related

  - [`regime_bias_table`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
"""
function regime_bias_factor(::FirstMomentRegimeAdjusted, s::VecNum, G::VecNum, c::Number,
                            h::Number)
    return (h * sum(i -> G[i] * sqrt(s[i]), eachindex(s, G)))^2 / (2 * c * pi)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor that the log adjustment reads, ``\\exp(-\\mathbb{E}[\\ln Q])``.

The method smooths ``\\ln z^{2} = \\ln u^{2} - \\ln Q``, so the statistic ``z^{2}`` is divided by
``\\exp(-\\mathbb{E}[\\ln Q])``, whatever the parameters of the method.

# Arguments

  - `::LogRegimeAdjusted`: Log regime adjustment method.
  - `s::VecNum`: Grid of the integration variable ``s = e^{x}``.
  - `G::VecNum`: ``G(s)``, the Laplace transform of `Q` on the grid.
  - `c::Number`: Normalised weight of the newest observation.
  - `h::Number`: Step of the grid in ``x``.

# Returns

  - `factor::Number`: ``\\exp\\left(-\\int_{0}^{\\infty} (e^{-s} - G(s))\\, s^{-1}\\, \\mathrm{d}s - \\ln(2 c)\\right)``.

# Related

  - [`regime_bias_table`](@ref)
  - [`LogRegimeAdjusted`](@ref)
"""
function regime_bias_factor(::LogRegimeAdjusted, s::VecNum, G::VecNum, c::Number, h::Number)
    return exp(-(h * sum(i -> exp(-s[i]) - G[i], eachindex(s, G)) + log(2 * c)))
end
"""
$(DocStringExtensions.TYPEDEF)

Names the table of the moments of one term of a sum of regime statistics, which the table of a
[`DiagonalTarget`](@ref) holds.

A term ``u_{i}^{2} / \\hat{v}_{i}`` reads one estimated variance ``\\hat{v}_{i} = \\sigma_{i}^{2} Q_{i}``.
The estimator divides the term by the factor of the mean, ``f = \\mathbb{E}[Q^{-1}]``, so the term
carries the noise ``a = 1 / (f Q)``, with ``\\mathbb{E}[a] = 1``.

The root and the log of the sum read that noise through a variable of their own. In that variable
the statistic is linear along the direction where every term carries the same noise. The variable
is ``a^{1/2}`` for [`FirstMomentRegimeAdjusted`](@ref) and ``\\ln a`` for [`LogRegimeAdjusted`](@ref).
For each count, [`regime_bias_factor`](@ref) gives the triple ``(f, b, v)``. These are the factor
of the mean, the scale ``b`` at which the variable of the method has its mean, and the relative
variance ``v`` of that variable.

| Method                              | ``b``                               | ``v``                                 |
|:----------------------------------- |:----------------------------------- |:------------------------------------- |
| [`FirstMomentRegimeAdjusted`](@ref) | ``\\mathbb{E}[Q^{-1/2}]^{2} / f``   | ``f / \\mathbb{E}[Q^{-1/2}]^{2} - 1`` |
| [`LogRegimeAdjusted`](@ref)         | ``\\exp(-\\mathbb{E}[\\ln Q]) / f`` | ``\\mathrm{Var}[\\ln Q]``             |
| any other method                    | ``1``                               | ``0``                                 |

# Fields

  - `method`: $(field_dict[:ra_term_method])

# Related

  - [`regime_bias_factor`](@ref)
  - [`regime_bias_table`](@ref)
  - [`diagonal_law_factor`](@ref)
"""
@concrete struct RegimeTermMoments
    method
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the moments of one term for a method that reads the mean of the sum: the mean's factor, a
scale of one and no variance.

# Arguments

  - `::RegimeTermMoments`: Table of the moments of one term.
  - `s::VecNum`: Grid of the integration variable ``s = e^{x}``.
  - `G::VecNum`: ``G(s)``, the Laplace transform of `Q` on the grid.
  - `c::Number`: Normalised weight of the newest observation.
  - `h::Number`: Step of the grid in ``x``.

# Returns

  - `moments::Tuple`: ``(\\mathbb{E}[Q^{-1}], 1, 0)``.

# Related

  - [`RegimeTermMoments`](@ref)
"""
function regime_bias_factor(::RegimeTermMoments, s::VecNum, G::VecNum, c::Number, h::Number)
    f = regime_bias_factor(RootMeanSquaredAdjusted(), s, G, c, h)
    return (f, one(f), zero(f))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the moments of one term for the first-moment method, which reads the root of the noise
``a = 1 / (f Q)``.

# Arguments

  - `m::RegimeTermMoments{<:FirstMomentRegimeAdjusted}`: Table of the moments of one term.
  - `s::VecNum`: Grid of the integration variable ``s = e^{x}``.
  - `G::VecNum`: ``G(s)``, the Laplace transform of `Q` on the grid.
  - `c::Number`: Normalised weight of the newest observation.
  - `h::Number`: Step of the grid in ``x``.

# Returns

  - `moments::Tuple`: ``(f, q / f, f / q - 1)``, with ``f = \\mathbb{E}[Q^{-1}]`` and
    ``q = \\mathbb{E}[Q^{-1/2}]^{2}``, so ``b = \\mathbb{E}[a^{1/2}]^{2}`` and ``v`` is the relative
    variance of ``a^{1/2}``.

# Related

  - [`RegimeTermMoments`](@ref)
"""
function regime_bias_factor(m::RegimeTermMoments{<:FirstMomentRegimeAdjusted}, s::VecNum,
                            G::VecNum, c::Number, h::Number)
    f = regime_bias_factor(RootMeanSquaredAdjusted(), s, G, c, h)
    q = regime_bias_factor(m.method, s, G, c, h)
    return (f, q / f, f / q - one(f))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the moments of one term for the log method, which reads the log of the noise
``a = 1 / (f Q)``.

# Arguments

  - `m::RegimeTermMoments{<:LogRegimeAdjusted}`: Table of the moments of one term.
  - `s::VecNum`: Grid of the integration variable ``s = e^{x}``.
  - `G::VecNum`: ``G(s)``, the Laplace transform of `Q` on the grid.
  - `c::Number`: Normalised weight of the newest observation.
  - `h::Number`: Step of the grid in ``x``.

# Returns

  - `moments::Tuple`: ``(f, e^{-\\mathbb{E}[\\ln Q]} / f, \\mathrm{Var}[\\ln Q])``, with
    ``f = \\mathbb{E}[Q^{-1}]``, so ``b = \\exp(\\mathbb{E}[\\ln a])``, from
    [`regime_log_variance`](@ref).

# Related

  - [`RegimeTermMoments`](@ref)
  - [`regime_log_variance`](@ref)
"""
function regime_bias_factor(m::RegimeTermMoments{<:LogRegimeAdjusted}, s::VecNum, G::VecNum,
                            c::Number, h::Number)
    f = regime_bias_factor(RootMeanSquaredAdjusted(), s, G, c, h)
    return (f, regime_bias_factor(m.method, s, G, c, h) / f, regime_log_variance(s, G, h))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the variance of the log of an estimated variance, from the Laplace transform of `Q`.

The Mellin transform of ``Q' = 2 c Q`` expands about zero as
``\\mathbb{E}[Q'^{-\\sigma}] = 1 + \\sigma I_{0} + \\sigma^{2} (I_{1} + \\gamma I_{0}) + O(\\sigma^{3})``,
so its first two coefficients are the first two moments of ``-\\ln Q'``. A scale does not move a
variance of a log, so the variance of ``\\ln Q'`` is that of ``\\ln Q``. At equal weights it agrees
with the trigamma function ``\\psi'(K / 2)``.

# Mathematical definition

```math
\\begin{align}
I_{0} &= \\int_{0}^{\\infty} \\left(G(s) - e^{-s}\\right) s^{-1}\\, \\mathrm{d}s\\,, \\quad
I_{1} = \\int_{0}^{\\infty} \\ln s \\left(G(s) - e^{-s}\\right) s^{-1}\\, \\mathrm{d}s\\,, \\\\
\\mathrm{Var}[\\ln Q] &= 2 (I_{1} + \\gamma I_{0}) - I_{0}^{2}\\,.
\\end{align}
```

Where ``\\gamma`` is the Euler-Mascheroni constant.

# Arguments

  - `s::VecNum`: Grid of the integration variable ``s = e^{x}``.
  - `G::VecNum`: ``G(s)``, the Laplace transform of `Q` on the grid.
  - `h::Number`: Step of the grid in ``x``.

# Returns

  - `v::Number`: ``\\mathrm{Var}[\\ln Q]``.

# Related

  - [`regime_bias_factor`](@ref)
  - [`regime_bias_table`](@ref)
"""
function regime_log_variance(s::VecNum, G::VecNum, h::Number)
    I0 = h * sum(i -> G[i] - exp(-s[i]), eachindex(s, G))
    I1 = h * sum(i -> log(s[i]) * (G[i] - exp(-s[i])), eachindex(s, G))
    return 2 * (I1 + Base.MathConstants.eulergamma * I0) - I0^2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor of a regime statistic that reads one estimated variance, for every count
of observations from one to `K`.

An exponentially weighted variance estimate of ``K`` observations is ``\\hat{v} = \\sigma^{2} Q``,
with ``Q = \\sum_{j} w_{j} z_{j}^{2}`` on correctly calibrated returns. The inverse of an estimate
is too large on average (Jensen's inequality), and each [`RegimeAdjustedMethod`](@ref) reads its
own moment of it, so [`regime_bias_factor`](@ref) names the factor of each method. The factors
are exact, not asymptotic. At equal weights they agree with ``K / (K - 2)``, with the ratio of
gamma functions and with the digamma form to ``10^{-12}``.

# Mathematical definition

```math
\\begin{align}
w_{j} &= c\\, \\lambda^{j}\\,, \\quad c = \\frac{1 - \\lambda}{1 - \\lambda^{K}}\\,, \\\\
\\mathbb{E}[e^{-tQ}] &= G(2 c t)\\,, \\quad G(s) = \\prod_{j=0}^{K-1} (1 + \\lambda^{j} s)^{-1/2}\\,.
\\end{align}
```

Where:

  - ``w_{j}``: Normalised weight of the observation ``j`` steps before the newest.
  - ``\\lambda``: `decay`.
  - ``G``: Laplace transform of ``Q``, on the scale ``s = 2 c t``.

# Algorithm

 1. Lay a grid of step ``h = 1/10`` on ``x = \\ln s \\in [-60, 50]``. The integrands are smooth
    on it and decay at both ends, the slowest as ``e^{x/2}`` below, so the trapezoid rule on it
    agrees with a grid of step ``1/200`` on ``[-75, 60]`` to ``1.1 \\times 10^{-12}`` relative from
    ``K = 5``, at half-lives of 2, 10 and 40.
 2. For each ``K``, multiply ``G`` by ``(1 + \\lambda^{K - 1} s)^{-1/2}`` on the log scale, so the
    whole table costs one pass.
 3. Evaluate the factor of the method with [`regime_bias_factor`](@ref).

The mean of ``Q^{-1}`` is finite from ``K = 3`` and its variance from ``K = 5``, so the estimator
does not read an entry at ``K \\le 4``.

# Arguments

  - `method::Union{RegimeAdjustedMethod, RegimeTermMoments}`: Regime adjustment method,
    which names the moment, or the table of the moments of one term.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Largest count of observations in the table.

# Returns

  - `table::AbstractVector`: The factor for each count from one to `K`, or the moments of
    [`RegimeTermMoments`](@ref).

# Related

  - [`regime_bias_factor`](@ref)
  - [`regime_bias!`](@ref)
  - [`mahalanobis_bias`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_bias_table(method::Union{RegimeAdjustedMethod, RegimeTermMoments},
                           decay::Number, K::Integer)
    h = one(decay) / 10
    s = exp.(range(-60 * one(h), 50 * one(h); step = h))
    lG = zero(s)
    G = similar(s)
    lam = one(decay)
    return map(1:K) do k
        lG .-= log1p.(lam .* s) ./ 2
        lam *= decay
        G .= exp.(lG)
        return regime_bias_factor(method, s, G,
                                  (one(decay) - decay) / (one(decay) - decay^k), h)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the bias factor of a regime statistic at a count of observations, and grows the table of
the state where the count is past its end.

The table grows to twice the count, and to 64 entries at the least. It stops at the count
``K_{\\mathrm{sat}}`` where ``\\lambda^{K}`` falls below the machine epsilon of the table. Past that
count the factor does not change, so the last entry serves every larger count. A fit over ``T``
rows thus makes the table once or a few times, and the table holds no more than
``\\min(\\max(2 T, 64), K_{\\mathrm{sat}})`` entries.

# Arguments

  - `bias::AbstractVector`: Table of the state (mutated where it grows).
  - `method::Union{RegimeAdjustedMethod, RegimeTermMoments}`: Regime adjustment method,
    which names the moment, or the table of the moments of one term.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `hac_lags::Option{<:Union{Integer, VecNum}}`: Count of HAC lags of the estimate, or the
    weight of each lag, whose table [`regime_bias_table`](@ref) computes from its banded weight
    matrix, or `nothing`.
  - `centring::AbstractCentring`: The centring of the estimate. Under [`EstimatedCentring`](@ref)
    the table reads the exact law of the estimated deviations. The default, [`PreCentred`](@ref),
    reads the law of pre-centred terms.
  - `debias::AbstractRegimeDebias`: Whether the table reads the dependence of the deviation on the
    estimate, by [`reads_dependence`](@ref). [`ExactDebias`](@ref) by default.

# Returns

  - `factor`: The factor at `K`, or the moments of [`RegimeTermMoments`](@ref) at `K`.

# Related

  - [`regime_bias_table`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
"""
function regime_bias!(bias::AbstractVector,
                      method::Union{RegimeAdjustedMethod, RegimeTermMoments}, decay::Number,
                      K::Integer, hac_lags::Option{<:Union{Integer, VecNum}} = nothing,
                      centring::AbstractCentring = PreCentred(),
                      debias::AbstractRegimeDebias = ExactDebias())
    Ksat = ceil(Int, log(eps(eltype(eltype(bias)))) / log(decay))
    if K > length(bias) && length(bias) < Ksat
        table = regime_bias_table(method, decay, min(max(2 * K, 64), Ksat), hac_lags,
                                  centring, debias)
        resize!(bias, length(table))
        copyto!(bias, table)
    end

    return bias[min(K, length(bias))]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns one, the factor of a state that takes no bias correction.

# Arguments

  - `::Nothing`: The table of a state whose estimator has `debias = RawStatistic()` or no regime method.
  - `::Any`: Ignored regime adjustment method.
  - `decay::Number`: Decay of the weights.
  - `::Integer`: Ignored count of observations.
  - `::Any`: Ignored count of HAC lags.
  - `::Any`: Ignored centring.
  - `::Any`: Ignored debias rule.

# Returns

  - `one(decay)`.

# Related

  - [`regime_bias!`](@ref)
"""
function regime_bias!(::Nothing, ::Any, decay::Number, ::Integer, ::Any = nothing,
                      ::Any = nothing, ::Any = nothing)
    return one(decay)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns whether an estimate of `K` observations is old enough for its regime statistic to be
scored.

The statistic of `n` directions reads the inverse of an estimate, and the variance of that inverse
is finite from ``K > n + 3``. Before that count, one observation can dominate the smoothed regime
state for many half-lives. A HAC estimate has a weight matrix with negative eigenvalues, so it can
fail to be positive definite at any count, and no moment of its inverse is finite. Its effective
count of observations, ``1 / \\operatorname{tr}(A^{2})`` from [`exp_weight_cross_sum`](@ref), must
also pass ``n + 1``. That is the count at which the mean of the inverse of a plain estimate is
finite.

On pre-centred normal returns at 12 assets, a half-life of 10 and two lags, `K > n + 3` alone
scores the estimate at ``K = 16``, where 19 % of draws are not positive definite. With the effective count the first scored row is ``K = 54``, and no draw of 20000 fails
there. Over half-lives of 5, 10 and 40, one to five lags and one to 12 assets, the first scored
row is not positive definite in at most ``1.3 \\times 10^{-3}`` of draws at a half-life of 5. The
share is at most ``4 \\times 10^{-4}`` at 10, and no draw of 5000 fails at 40. The gate never
scores an estimate whose steady state has too few effective observations, such as 12 assets at a
half-life of 5. Without the correction this gate opens for every estimate, and `min_obs` alone
gates it.

# Arguments

  - `debias::AbstractRegimeDebias`: Whether the estimator corrects the bias of its statistic.
  - `n::Integer`: Count of directions that the statistic reads.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of the terms in the estimate.
  - `hac_lags::Option{<:Union{Integer, VecNum}}`: Count of HAC lags, the weight of each lag, or
    `nothing`.

# Returns

  - `open::Bool`: Whether the statistic is scored.

# Related

  - [`regime_bias_table`](@ref)
  - [`exp_weight_cross_sum`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function regime_bias_open(debias::AbstractRegimeDebias, n::Integer, decay::Number,
                          K::Integer, hac_lags::Option{<:Union{Integer, VecNum}})
    return !debiases(debias) ||
           K > n + 3 && (isnothing(hac_lags) ||
                         exp_weight_cross_sum(decay, decay, K, hac_lags) * (n + 1) < one(decay))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the terms of the current observation, and stores them in `cache.X2`.

A term is the squared deviation, plus the HAC products where `hac_lags` is set, over its exact
factor.

The factor is the mean of the term in units of the variance, for returns that are independent in
time: [`centring_factor`](@ref) of the square plus [`centring_lag_factor`](@ref) of the lagged
products. Both are one and zero under [`PreCentred`](@ref). The floor rule of `ce.hac_floor` then
acts on the terms of the valid assets.

# Arguments

  - `cache::RegimeAdjustedVarianceState`: Online variance computation cache.
  - `ce::RegimeAdjustedExpWeightedVariance`: Variance estimator configuration.
  - `X::VecNum`: The deviations of the current observation.
  - `valid::AbstractVector{<:Bool}`: Mask of the valid assets.
  - `dvalid::AbstractVector{<:Bool}`: Mask of the assets whose deviation exists.
  - `f::VecNum`: The factor of each squared deviation.

# Returns

  - `X2::VecNum`: The terms, stored in `cache.X2`.

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function hac_squared_returns!(cache::RegimeAdjustedVarianceState,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              valid::AbstractVector{<:Bool}, dvalid::AbstractVector{<:Bool},
                              f::VecNum)
    copyto!(cache.X2, X .^ 2)
    if isnothing(cache.ret_buffer) || isempty(cache.ret_buffer)
        cache.X2 ./= f
        return cache.X2
    end

    w = l -> one(eltype(X)) - l / (ce.hac_lags + 1)
    for (i, X_old) in enumerate(Iterators.reverse(cache.ret_buffer))
        cache.X_old_i .= replace(X_old, NaN => zero(eltype(X_old)))
        cache.X2 .+= 2 * w(i) * X .* cache.X_old_i
    end
    cache.X2 ./= f .+
                 centring_lag_factor(ce.centring, ce.decay, cache.old_obs_count, dvalid,
                                     cache.lag_records, w, true)

    return hac_floor!(ce.hac_floor, cache.X2, valid)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into the online variance cache.

The function steps the location, the variance and the standardised squared returns in `cache`,
and then it advances the smoothed regime state. A `regime_method` of `nothing` advances no regime
state, so the variance stays the plain exponentially weighted recursion. A row with no valid asset
changes no estimate and no count. Under HAC it still enters the buffer as one lag, so a lag counts
observations whatever the assets hold.

# Algorithm

 1. Mark the assets whose return is finite, and that `active_mask` holds where it is set, giving
    `valid`.
 2. Reset the variance, the count, the location, the HAC buffer and the lagged records of each
    asset that becomes inactive, and store the mask in `cache.active`.
 3. Copy `cache.obs_count` to `cache.old_obs_count`. Step the location with
    [`centring_location!`](@ref), giving the deviations `Xi` and the mask `dvalid` of the assets
    whose deviation exists.
 4. Compute the exact factor `f` of each squared deviation, and the count `K` of the terms in each
    variance.
 5. Form `regime_mask`, the assets of `dvalid` that `min_obs` admits and that
    [`regime_bias_open`](@ref) scores. The mask is empty where `regime_method` is `nothing`.
 6. Write ``z^{2}`` to `cache.z2` for each asset of `regime_mask` whose variance is at least
    `min_val`, with the weight of the zero seed and the factor of [`regime_bias!`](@ref).
 7. Compute the terms `X2` with [`hac_squared_returns!`](@ref). Step the variance of each asset of
    `dvalid`, and add one to the count of each asset of `valid`.
 8. Under HAC, push the deviations to the buffer, with `NaN` where no deviation exists, and push
    the lagged record.
 9. Keep the assets of `regime_mask` that `estimation_mask` holds. Return the cache where no
    asset is left.
10. Compute the statistic with [`get_regime_state`](@ref), and blend it into the regime state
    with [`blend_regime_state`](@ref).
11. Return a new state that holds the blended regime state and one more regime observation.

# Arguments

  - `cache::RegimeAdjustedVarianceState`: Online variance computation cache (mutated).
  - `ce::RegimeAdjustedExpWeightedVariance`: Variance estimator configuration.
  - `X::VecNum`: Returns vector for the current observation.
  - `estimation_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of the assets that the
    regime state reads.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of the active assets. An asset
    that becomes inactive restarts its variance and its counts from zero.

# Returns

  - `cache::RegimeAdjustedVarianceState`: The cache to read, and to pass to the next observation.
    The function mutates the arrays in place. `regime_state` and `n_regime_obs` are immutable
    fields, so a new state holds them, and the caller must rebind the cache to this return value.
    A caller that discards it freezes the regime state at its initial value.

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function process_observation!(cache::RegimeAdjustedVarianceState,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              estimation_mask::Option{<:AbstractVector{<:Bool}},
                              active_mask::Option{<:AbstractVector{<:Bool}})
    finite_mask = isfinite.(X)
    valid = isnothing(active_mask) ? finite_mask : (finite_mask .& active_mask)

    if !isnothing(active_mask)
        newly_inactive = .!active_mask .& cache.active
        if any(newly_inactive)
            cache.variance[newly_inactive] .= zero(eltype(cache.variance))
            cache.obs_count[newly_inactive] .= 0
            cache.location[newly_inactive] .= NaN
            hac_buffer_reset!(cache.ret_buffer, newly_inactive)
            centring_lag_reset!(cache.lag_records, newly_inactive)
        end
        cache.active .= active_mask
    else
        cache.active .= true
    end

    # A row with no valid asset takes this path too, as in the covariance. It writes no
    # estimate and no count, but its blank deviations enter the buffer as one HAC lag.
    copyto!(cache.old_obs_count, cache.obs_count)
    Xi, dvalid = centring_location!(ce.centring, cache.location, cache.obs_count, X, valid,
                                    ce.decay)
    f = centring_factor.(Ref(ce.centring), ce.decay, cache.old_obs_count)
    K = centring_terms.(Ref(ce.centring), cache.old_obs_count)

    # A `regime_method` of `nothing` empties the mask, so no `z²` is formed and no regime state
    # advances. It is a conjunction rather than a branch, which keeps one code path. The gate
    # skips an estimate too young for its statistic to have a finite variance.
    regime_mask = dvalid .&
                  centring_ready.(Ref(ce.centring), cache.old_obs_count, ce.min_obs) .&
                  regime_bias_open.(Ref(ce.debias), 1, ce.decay, K, ce.hac_lags) .&
                  !isnothing(ce.regime_method)
    fill!(cache.z2, NaN)
    var_idx = regime_mask .& (cache.variance .>= ce.min_val)
    if any(var_idx)
        Kv = view(K, var_idx)
        factor = inv.(max.(one(ce.decay) .- ce.decay .^ Kv, eps(ce.decay)))
        var_corrected = view(cache.variance, var_idx) .* factor .*
                        regime_bias!.(Ref(cache.bias), Ref(ce.regime_method), ce.decay, Kv,
                                      ce.hac_lags, Ref(ce.centring), Ref(ce.debias))
        cache.z2[var_idx] = view(Xi, var_idx) .^ 2 ./ view(f, var_idx) ./ var_corrected
    end

    X2 = hac_squared_returns!(cache, ce, Xi, valid, dvalid, f)
    cache.variance[dvalid] .= ce.decay * view(cache.variance, dvalid) +
                              (one(ce.decay) - ce.decay) * view(X2, dvalid)
    cache.obs_count[valid] .+= 1

    if !isnothing(cache.ret_buffer)
        X_new = copyto!(similar(cache.X2), Xi)
        X_new[.!dvalid] .= NaN
        push!(cache.ret_buffer, X_new)
        centring_lag_push!(cache.lag_records, cache.old_obs_count, valid, nothing)
    end

    if !isnothing(estimation_mask)
        regime_mask .&= estimation_mask
    end

    if !any(regime_mask)
        return cache
    end

    z2_valid = filter(!isnan, view(cache.z2, regime_mask))
    if isempty(z2_valid)
        return cache
    end

    c = cache
    regime_state = get_regime_state(ce.regime_method, z2_valid, ce.min_val)
    rs = blend_regime_state(c.regime_state, regime_state, ce.regime_decay)
    return RegimeAdjustedVarianceState(c.ret_buffer, c.variance, c.X2, c.X_old_i, c.z2,
                                       c.location, c.obs_count, c.old_obs_count, c.active,
                                       rs, c.n_regime_obs + 1, c.bias, c.lag_records)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run one forward pass of the online variance update over the observations of `X`, and call `f`
after each observation.

The pass holds the argument validation, the orientation and the cache, so the two verbs that read
it, `var` and `variance_series`, state the recursion once. `f` receives the observation index and
the cache after that observation. So a point-in-time series is one forward pass, and not one
refit for each observation.

# Arguments

  - `f`: Function of `(i, cache)` called after observation `i` is processed. A caller that wants
    only the final cache reads the method that takes no `f`.
  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix of the size of `X`. Where it is set, the regime
    state at observation `i` reads only the assets where `estimation_mask[i, :]` (or `[:, i]`)
    is `true`.
  - `active_mask`: Optional boolean matrix of the size of `X`. Where it is set, an asset that
    becomes inactive restarts its variance and its observation count from zero.
  - `state`: Optional cache to continue from. The default `nothing` builds the cold cache, which
    is what a fit over a whole sample needs. [`partial_fit!`](@ref) passes the estimator's own
    state instead, so the pass continues where the last call stopped.

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `state` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `cache::RegimeAdjustedVarianceState`: The cache after the last observation.

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`process_observation!`](@ref)
  - [`regime_adjusted_variance`](@ref)
  - [`partial_fit!`](@ref)
"""
function regime_adjusted_variance_pass!(f, ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                                        dims::Int,
                                        estimation_mask::Option{<:AbstractMatrix{<:Bool}},
                                        active_mask::Option{<:AbstractMatrix{<:Bool}},
                                        state::Option{<:RegimeAdjustedVarianceState} = nothing)
    assert_dims(dims)
    est_flag = !isnothing(estimation_mask)
    act_flag = !isnothing(active_mask)
    itr, v = ifelse(isone(dims), (eachrow, (x, y) -> view(x, y, :)),
                    (eachcol, (x, y) -> view(x, :, y)))
    if est_flag
        @argcheck(size(X) == size(estimation_mask),
                  DimensionMismatch("size(X) ($(size(X))) must match size(estimation_mask) ($(size(estimation_mask)))"))
    end
    if act_flag
        @argcheck(size(X) == size(active_mask),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(active_mask)))"))
    end
    N = size(X, setdiff((1, 2), (dims,))[1])

    # The state holds a variance and a `NaN`, so an integer panel computes in its floating
    # point type, and every other type is kept: a `Float32` panel keeps a `Float32` state.
    Tf = float_if_integer(eltype(X))
    # The location starts as `NaN`, which marks an asset with no valid observation.
    cache = if isnothing(state)
        RegimeAdjustedVarianceState(if isnothing(ce.hac_lags)
                                        nothing
                                    else
                                        DataStructures.CircularBuffer{Vector{Tf}}(ce.hac_lags)
                                    end, zeros(Tf, N), zeros(Tf, N), zeros(Tf, N),
                                    fill(convert(Tf, NaN), N), fill(convert(Tf, NaN), N),
                                    zeros(Int, N), zeros(Int, N), trues(N), nothing,
                                    zero(Tf), regime_bias_state(ce, Tf),
                                    centring_lag_records(ce.centring, ce.hac_lags, nothing))
    else
        @argcheck(length(state.variance) == N,
                  DimensionMismatch("the state holds $(length(state.variance)) assets, and `X` holds $N"))
        state
    end
    for (i, Xi) in enumerate(itr(X))
        emi = est_flag ? v(estimation_mask, i) : nothing
        ami = act_flag ? v(active_mask, i) : nothing
        cache = process_observation!(cache, ce, Xi, emi, ami)
        f(i, cache)
    end

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run one forward pass of the online variance update over the observations of `X`, and read no
intermediate cache.

This is the callback method with a callback that does nothing, so a verb that wants the last
cache alone states no callback of its own. `var` and `partial_fit!` read the pass this way, and
`variance_series` reads the callback method.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix of the size of `X`. Where it is set, the regime
    state at observation `i` reads only the assets where `estimation_mask[i, :]` (or `[:, i]`)
    is `true`.
  - `active_mask`: Optional boolean matrix of the size of `X`. Where it is set, an asset that
    becomes inactive restarts its variance and its observation count from zero.
  - `state`: Optional cache to continue from. The default `nothing` builds the cold cache, which
    is what a fit over a whole sample needs. [`partial_fit!`](@ref) passes the estimator's own
    state instead, so the pass continues where the last call stopped.

# Returns

  - `cache::RegimeAdjustedVarianceState`: The cache after the last observation.

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`regime_adjusted_variance_pass!`](@ref)
  - [`regime_adjusted_variance`](@ref)
  - [`partial_fit!`](@ref)
"""
function regime_adjusted_variance_pass!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                                        dims::Int,
                                        estimation_mask::Option{<:AbstractMatrix{<:Bool}},
                                        active_mask::Option{<:AbstractMatrix{<:Bool}},
                                        state::Option{<:RegimeAdjustedVarianceState} = nothing)
    return regime_adjusted_variance_pass!((args...) -> nothing, ce, X, dims,
                                          estimation_mask, active_mask, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the regime-adjusted variance out of a cache, as it stands.

A HAC state is a quadratic form with an indefinite weight matrix, so it can fall below zero. The
function floors the variance at zero here, and not in the recursion, so the recursion stays
unbiased. The function reads the cache and never writes it, so the same cache answers this call
after every observation of a forward pass.

Where `ce.regime_method` is `nothing`, [`process_observation!`](@ref) advances no regime state, so
`cache.n_regime_obs` stays at zero, which is below every admissible `regime_min_obs`. The
multiplier is then one, and the variance is the plain recursion.

Where `ce.regime_lohi_mult` is not `nothing`, the function clamps the multiplier to that
`(lo, hi)` range, and then squares it. Where it is `nothing`, no clamp runs. The clamp bounds an
estimate, so the multiplier of one before `regime_min_obs` holds even where `lo > 1` or `hi < 1`.

# Algorithm

 1. Floor the raw variance of each asset at zero, giving `variance`.
 2. Divide the variance of each asset that holds a term by `1 - decay^K`, the weight that the
    zero seed leaves.
 3. Set the variance to `NaN` for each asset that is not active, or that `min_obs` does not admit.
 4. Compute the multiplier `factor`. It is one before `regime_min_obs` regime observations, and
    otherwise [`regime_multiplier`](@ref) of the regime state, clamped where `regime_lohi_mult`
    is set.
 5. Return `variance * factor^2`.

# Arguments

  - `cache::RegimeAdjustedVarianceState`: Online variance computation cache.
  - `ce::RegimeAdjustedExpWeightedVariance`: Variance estimator configuration.

# Returns

  - `variance::Vector{<:Number}`: Per-asset regime-adjusted variance vector of length `assets`.
    An asset with fewer than `ce.min_obs` observations, or one that is not active, is `NaN`.

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_adjusted_variance(cache::RegimeAdjustedVarianceState,
                                  ce::RegimeAdjustedExpWeightedVariance)
    variance = max.(cache.variance, zero(eltype(cache.variance)))
    correction = ones(eltype(variance), length(variance))
    K = centring_terms.(Ref(ce.centring), cache.obs_count)
    counted = K .> zero(eltype(K))
    correction[counted] .= inv.(max.(one(ce.decay) .- ce.decay .^ view(K, counted),
                                     eps(eltype(variance))))
    variance .*= correction
    not_ready = .!cache.active .| .!centring_ready.(Ref(ce.centring), cache.obs_count, ce.min_obs)

    if any(not_ready)
        variance[not_ready] .= NaN
    end

    # The clamp bounds an estimate; the warm-up factor of one is no estimate, so it holds.
    factor = if cache.n_regime_obs < ce.regime_min_obs
        one(eltype(variance))
    elseif isnothing(ce.regime_lohi_mult)
        regime_multiplier(ce.regime_method, cache.regime_state)
    else
        clamp(regime_multiplier(ce.regime_method, cache.regime_state),
              ce.regime_lohi_mult[1], ce.regime_lohi_mult[2])
    end

    return variance * factor^2
end
"""
    Statistics.var(
        ce::RegimeAdjustedExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Vector{<:Number}

Compute the regime-adjusted exponentially weighted variance for each asset.

The method steps an online variance cache over the observations of `X`. After the last
observation, it divides each variance by the weight that the zero seed leaves, and it scales the
result by the square of the regime multiplier.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix of the size of `X`. Where it is set, the regime
    state at observation `i` reads only the assets where `estimation_mask[i, :]` (or `[:, i]`)
    is `true`.
  - `active_mask`: Optional boolean matrix of the size of `X`. Where it is set, an asset that
    becomes inactive restarts its variance and its observation count from zero.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `var::Vector{<:Number}`: Per-asset regime-adjusted exponentially weighted variance
    vector of length `assets`. The variance of an asset with fewer than `ce.min_obs`
    observations, or of an asset that is not active, is `NaN`.

# Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
  - [`variance_series(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = regime_adjusted_variance_pass!(ce, X, dims, estimation_mask, active_mask)
    return regime_adjusted_variance(cache, ce)
end
"""
    variance_series(
        ce::RegimeAdjustedExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time regime-adjusted exponentially weighted variance series.

Row `t` holds what `var` returns for the first `t` observations of `X`, so no row reads an
observation after its own. The update is a recursion over one observation, so this method
replaces the expanding-window fallback with one forward pass. It reads the cache after each
observation, and it does not refit.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix of the size of `X`. Where it is set, the regime
    state at observation `i` reads only the assets where `estimation_mask[i, :]` (or `[:, i]`)
    is `true`.
  - `active_mask`: Optional boolean matrix of the size of `X`. Where it is set, an asset that
    becomes inactive restarts its variance and its observation count from zero.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if
    `dims == 2`. An asset with fewer than `ce.min_obs` observations at row `t` is `NaN` there.

# Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                         estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    val = Matrix{float_if_integer(eltype(X))}(undef, size(X, dims),
                                              size(X, setdiff((1, 2), (dims,))[1]))
    regime_adjusted_variance_pass!(ce, X, dims, estimation_mask, active_mask) do i, cache
        val[i, :] = regime_adjusted_variance(cache, ce)
        return nothing
    end

    return isone(dims) ? val : permutedims(val)
end
"""
    partial_fit!(
        ce::RegimeAdjustedExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> RegimeAdjustedExpWeightedVariance

Fold every observation of `X` into the estimator's own online variance state.

The state is the cache that the forward pass builds. The estimator keeps it, so a second call continues the recursion where the first call stopped. A call on an estimator whose `cache` field holds `nothing` builds the cold cache. So one call over a whole sample gives the state that `var(ce, X)` reads and then drops.

The method mutates the arrays of the state in place, and a new state holds the new scalar fields. So the caller must rebind the estimator to the return value. Two estimators that successive calls return share one state object, and the `!` in the name says so.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix of the size of `X`. Where it is set, the regime
    state at observation `i` reads only the assets where `estimation_mask[i, :]` (or `[:, i]`)
    is `true`.
  - `active_mask`: Optional boolean matrix of the size of `X`. Where it is set, an asset that
    becomes inactive restarts its variance and its observation count from zero.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `ce::RegimeAdjustedExpWeightedVariance`: The estimator, with `cache` holding the state after
    the last observation of `X`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03];

julia> ce = RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> isequal(var(partial_fit!(ce, X)), var(ce, X))
true

julia> halves = partial_fit!(partial_fit!(ce, X[1:3, :]), X[4:6, :]);

julia> isequal(var(halves), var(ce, X))
true
```

# Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
  - [`regime_adjusted_variance_pass!`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance)`](@ref)
"""
function partial_fit!(ce::RegimeAdjustedExpWeightedVariance{<:Any, <:Any, <:Any, <:Any,
                                                            <:Any, <:Any, <:Any, <:Any,
                                                            <:Any, <:Any, <:Any,
                                                            <:Option{<:RegimeAdjustedVarianceState}},
                      X::MatNum; dims::Int = 1,
                      estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = regime_adjusted_variance_pass!(ce, X, dims, estimation_mask, active_mask,
                                           ce.cache)
    return cache === ce.cache ? ce : rebuild_estimator(ce, (; cache = cache))
end
"""
    partial_fit!(
        ce::RegimeAdjustedExpWeightedVariance,
        x::VecNum;
        estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        kwargs...
    ) -> RegimeAdjustedExpWeightedVariance

Fold one observation into the estimator's own online variance state.

The entries of `x` are the assets of one observation, the row that the matrix method folds one at a time. A caller whose observations arrive one by one has this shape.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - `x::VecNum`: One observation, with one entry per asset.
  - `estimation_mask`: Optional boolean vector of the length of `x`, of the assets that the
    regime state reads.
  - `active_mask`: Optional boolean vector of the length of `x`. An asset that becomes inactive
    restarts its variance and its observation count from zero.
  - $(arg_dict[:ignkwargs])

# Validation

  - If `estimation_mask` is not `nothing`, `length(x) == length(estimation_mask)`.
  - If `active_mask` is not `nothing`, `length(x) == length(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `x`.

# Returns

  - `ce::RegimeAdjustedExpWeightedVariance`: The estimator, with `cache` holding the state after
    the observation.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03];

julia> ce = RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> one_at_a_time = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1); init = ce);

julia> isequal(var(one_at_a_time), var(ce, X))
true
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance)`](@ref)
"""
function partial_fit!(ce::RegimeAdjustedExpWeightedVariance{<:Any, <:Any, <:Any, <:Any,
                                                            <:Any, <:Any, <:Any, <:Any,
                                                            <:Any, <:Any, <:Any,
                                                            <:Option{<:RegimeAdjustedVarianceState}},
                      x::VecNum;
                      estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
    return partial_fit!(ce, permutedims(x); dims = 1,
                        estimation_mask = if isnothing(estimation_mask)
                            nothing
                        else
                            permutedims(estimation_mask)
                        end, active_mask = if isnothing(active_mask)
                            nothing
                        else
                            permutedims(active_mask)
                        end)
end
"""
    Statistics.var(
        ce::RegimeAdjustedExpWeightedVariance,
        state::RegimeAdjustedVarianceState;
        kwargs...
    ) -> Vector{<:Number}

Read the regime-adjusted variance out of a state held by hand.

This is [`regime_adjusted_variance`](@ref) under the public verb of the family. So a state that a caller keeps outside an estimator answers the same call as a state that the estimator holds. The method reads the state and never writes it.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - `state`: Running state of an incremental fit.
  - $(arg_dict[:ignkwargs])

# Returns

  - `var::Vector{<:Number}`: Per-asset regime-adjusted exponentially weighted variance vector of
    length `assets`. An asset with fewer than `ce.min_obs` observations is `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2,
                                                           regime_min_obs = 2), X);

julia> isequal(var(ce, ce.cache), var(ce))
true
```

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`regime_adjusted_variance`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance)`](@ref)
"""
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance,
                        state::RegimeAdjustedVarianceState; kwargs...)
    return regime_adjusted_variance(state, ce)
end
"""
    Statistics.var(ce::RegimeAdjustedExpWeightedVariance; kwargs...) -> Vector{<:Number}

Read the regime-adjusted variance out of the estimator's own state.

[`partial_fit!`](@ref) leaves the state in the `cache` field, and this one-argument form turns it into the variance. An estimator that has folded no observation carries no state, so the method refuses the call and does not return a zero.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator that holds a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `var::Vector{<:Number}`: Per-asset regime-adjusted exponentially weighted variance vector of
    length `assets`. An asset with fewer than `ce.min_obs` observations is `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2,
                                                           regime_min_obs = 2), X);

julia> length(var(ce))
2

julia> var(RegimeAdjustedExpWeightedVariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `var(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
"""
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `var(ce, X)` for a fit over a whole sample."))
    return var(ce, state)
end
"""
    Statistics.std(
        ce::RegimeAdjustedExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Vector{<:Number}

Compute the regime-adjusted exponentially weighted standard deviation for each asset.

It is the root of [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref). The variance is a non-negative exponentially weighted average scaled by a squared multiplier, so the root is real. An asset that `var` answers with `NaN` stays `NaN`.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix of the size of `X`. Where it is set, the regime
    state at observation `i` reads only the assets where `estimation_mask[i, :]` (or `[:, i]`)
    is `true`.
  - `active_mask`: Optional boolean matrix of the size of `X`. Where it is set, an asset that
    becomes inactive restarts its variance and its observation count from zero.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `std::Vector{<:Number}`: Per-asset regime-adjusted exponentially weighted standard
    deviation vector of length `assets`. The standard deviation of an asset with fewer than
    `ce.min_obs` observations, or of an asset that is not active, is `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> isapprox(std(ce, X), sqrt.(var(ce, X)))
true
```

# Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.std(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return sqrt.(var(ce, X; dims = dims, estimation_mask = estimation_mask,
                     active_mask = active_mask, kwargs...))
end
"""
    Statistics.std(
        ce::RegimeAdjustedExpWeightedVariance,
        state::RegimeAdjustedVarianceState;
        kwargs...
    ) -> Vector{<:Number}

Read the regime-adjusted standard deviation out of a state held by hand.

It is the root of [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)`](@ref). So a state that a caller keeps outside an estimator answers the same call as a state that the estimator holds. The method reads the state and never writes it.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - `state`: Running state of an incremental fit.
  - $(arg_dict[:ignkwargs])

# Returns

  - `std::Vector{<:Number}`: Per-asset regime-adjusted exponentially weighted standard
    deviation vector of length `assets`. An asset with fewer than `ce.min_obs` observations
    is `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2,
                                                           regime_min_obs = 2), X);

julia> isequal(std(ce, ce.cache), std(ce))
true
```

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)`](@ref)
  - [`Statistics.std(ce::RegimeAdjustedExpWeightedVariance; kwargs...)`](@ref)
"""
function Statistics.std(ce::RegimeAdjustedExpWeightedVariance,
                        state::RegimeAdjustedVarianceState; kwargs...)
    return sqrt.(var(ce, state; kwargs...))
end
"""
    Statistics.std(ce::RegimeAdjustedExpWeightedVariance; kwargs...) -> Vector{<:Number}

Read the regime-adjusted standard deviation out of the estimator's own state.

[`partial_fit!`](@ref) leaves the state in the `cache` field, and this one-argument form turns it into the standard deviation. An estimator that has folded no observation carries no state, so the method refuses the call and does not return a zero.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator that holds a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `std::Vector{<:Number}`: Per-asset regime-adjusted exponentially weighted standard
    deviation vector of length `assets`. An asset with fewer than `ce.min_obs` observations
    is `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2,
                                                           regime_min_obs = 2), X);

julia> length(std(ce))
2

julia> std(RegimeAdjustedExpWeightedVariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `std(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.std(ce::RegimeAdjustedExpWeightedVariance, state::RegimeAdjustedVarianceState; kwargs...)`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedVariance; kwargs...)`](@ref)
"""
function Statistics.std(ce::RegimeAdjustedExpWeightedVariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `std(ce, X)` for a fit over a whole sample."))
    return std(ce, state; kwargs...)
end
"""
    merge_states(
        a::RegimeAdjustedVarianceState,
        b::RegimeAdjustedVarianceState
    ) -> Union{}

Refuses a pair of regime-adjusted states, because this family does not merge.

A merge needs the state of a block to be a sufficient statistic for what that block contributes, and this state is not one. The regime state reads the standardised squared return of each observation. That score divides by the running variance, and the running observation count gates it. So a block fitted from a cold start weighs its first observations differently from the same block fitted after another block. The exponentially weighted accumulator folds as ``\\lambda^{n_B} v_A + v_B``, but the regime state that scales it does not fold. A fit under [`EstimatedCentring`](@ref) also holds its running location, and a HAC fit holds its buffer of recent returns.

Fold the second block into the first with [`partial_fit!`](@ref). A sequential fit is exact.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref), so a mismatched pair is named as such.
  - The method then refuses the pair with an `ArgumentError`, whatever it holds.

# Returns

  - The method always throws, so it returns no value.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2,
                                                           regime_min_obs = 2), X);

julia> try
           PortfolioOptimisers.merge_states(ce.cache, ce.cache)
       catch err
           err isa ArgumentError
       end
true
```

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_mergeable_states`](@ref)
  - [`partial_fit!(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function merge_states(a::RegimeAdjustedVarianceState, b::RegimeAdjustedVarianceState)
    assert_mergeable_states(a, b)
    return throw(ArgumentError("a `RegimeAdjustedVarianceState` pair does not merge, because a block fitted from a cold start is not what the same block contributes after another one. The regime state reads each observation's standardised squared innovation, which divides by the running variance and is gated by the running observation count. Fold the second block into the first with `partial_fit!` instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`RegimeAdjustedVarianceState`](@ref), so the copy shares no array with the original.

This is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, and [`partial_fit`](@ref) calls it before it folds. The method copies every array field, and it passes the two scalar fields through. It rebuilds the circular buffer of recent centred returns at the same capacity, and it copies each observation that the buffer holds. So a fold on the copy pushes into a buffer of its own.

This family answers `copy` and refuses `merge_states`. The two methods are independent. The merge asks whether two blocks fold into one. The copy asks only that a fold on one estimator leaves another estimator as it is.

# Algorithm

 1. Rebuild the circular buffer at the capacity of `x.ret_buffer`, and push a copy of each observation it holds. Take `nothing` when `x.ret_buffer` is `nothing`, which is the estimator that runs no HAC correction.
 2. Name the constructor, and pass a copy of each array field and the two scalar fields unchanged. The table of bias factors and the buffer of lagged records are each `nothing` or a container, and `deepcopy` copies a container and returns `nothing` unchanged.

# Arguments

  - `x`: The cache to copy.

# Returns

  - `state::RegimeAdjustedVarianceState`: A fresh cache, equal to `x`, whose arrays are fresh.

# Related

  - [`RegimeAdjustedVarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::RegimeAdjustedVarianceState)
    ret_buffer = if isnothing(x.ret_buffer)
        nothing
    else
        buffer = DataStructures.CircularBuffer{eltype(x.ret_buffer)}(DataStructures.capacity(x.ret_buffer))
        for X_old in x.ret_buffer
            push!(buffer, copy(X_old))
        end
        buffer
    end
    return RegimeAdjustedVarianceState(ret_buffer, copy(x.variance), copy(x.X2),
                                       copy(x.X_old_i), copy(x.z2), copy(x.location),
                                       copy(x.obs_count), copy(x.old_obs_count),
                                       copy(x.active), x.regime_state, x.n_regime_obs,
                                       deepcopy(x.bias), deepcopy(x.lag_records))
end
"""
    Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                   pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...) -> MatNum
    Statistics.std(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                   pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...) -> MatNum
    variance_series(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                    pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...) -> MatNum

Take the whole window, and read the two universe masks of the Asset Panel.

[`RegimeAdjustedExpWeightedVariance`](@ref) is mask-aware, so it overrides the reduce-and-expand root of its verb. Its recursion holds its own warm-up, its freeze on a holiday and its reset over an inactive period. So it returns its own frame over the whole window, and it does not take a clean block from [`coverage_reduction`](@ref). An asset outside the Coverage Universe therefore keeps a number here, where a plain estimator gives `NaN`.

The panel is the third positional argument. The method unpacks it onto the `active_mask` and `estimation_mask` keywords of the two-argument method. No panel and a static panel both give the path with no mask.

# Algorithm

 1. Read the two masks with [`panel_moment_masks`](@ref).
 2. Delegate to the two-argument method with them on its keywords.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the two-argument method.

# Returns

  - `sigma::MatNum`: The variance, or the standard deviation, of every asset of the window.

# Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`coverage_reduction`](@ref)
"""
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.var(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
function Statistics.std(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.std(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
function variance_series(ce::RegimeAdjustedExpWeightedVariance, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return variance_series(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                           kwargs...)
end

# Folds in every configuration; only its merge refuses (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(::RegimeAdjustedExpWeightedVariance)
    return true
end
function variance_count(ve::RegimeAdjustedExpWeightedVariance, X::MatNum)
    return exp_weighted_variance_count(ve.decay, X, ve.centring)
end
export LogRegimeAdjusted, FirstMomentRegimeAdjusted, RootMeanSquaredAdjusted,
       RegimeAdjustedExpWeightedVariance
public RegimeAdjustedMethod, regime_multiplier, get_regime_state
