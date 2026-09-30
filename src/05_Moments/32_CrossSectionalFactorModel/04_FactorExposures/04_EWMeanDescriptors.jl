"""
    half_life_decay(half_life::Real, sym::Sym_Str = :half_life) -> Real

Convert a half-life in observations into an exponential decay factor.

A half-life is the number of observations after which a weight halves, and a decay factor is the number the recursion multiplies its state by. Each fixes the other. A named Descriptor takes the half-life, because a reader of a factor model knows that number.

# Arguments

  - `half_life`: The half-life, in observations.
  - `sym::Sym_Str`: Name that the `DomainError` gives the half-life. A constructor with two half-lives passes the keyword it reads.

# Validation

  - `0 < half_life < Inf`, for the half-life that `sym` names. Raises a `DomainError`.

# Returns

  - `decay::Real`: The decay factor `2^(-1 / half_life)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.half_life_decay(1.0)
0.5

julia> PortfolioOptimisers.half_life_decay(2.0)
0.7071067811865476
```

# Related

  - [`half_life_min_obs`](@ref)
  - [`decay_half_life`](@ref)
  - [`EWMean`](@ref)
  - [`EWVolatility`](@ref)
"""
function half_life_decay(half_life::Real, sym::Sym_Str = :half_life)::Real
    assert_nonempty_gt0_finite_val(half_life, sym)
    return exp2(-inv(half_life))
end
"""
    half_life_min_obs(half_life::Real, sym::Sym_Str = :half_life) -> Int

Convert a half-life in observations into the warm-up of an exponentially weighted Descriptor.

The recursion starts from zero, so its early values lie closer to zero than the data does. Every exponentially weighted Descriptor returns `NaN` for an asset until that asset has this many valid observations.

# Arguments

  - `half_life`: The half-life, in observations.
  - `sym::Sym_Str`: Name that the `DomainError` gives the half-life. A caller that takes the warm-up from its own argument passes that argument's name.

# Validation

  - `0 < half_life < Inf`, for the half-life that `sym` names. Raises a `DomainError`.

# Returns

  - `min_obs::Int`: The warm-up, `ceil(Int, half_life)`, and at least one.

# Examples

```jldoctest
julia> PortfolioOptimisers.half_life_min_obs(40.0)
40

julia> PortfolioOptimisers.half_life_min_obs(0.5)
1
```

# Related

  - [`half_life_decay`](@ref)
  - [`EWMean`](@ref)
  - [`EWVolatility`](@ref)
"""
function half_life_min_obs(half_life::Real, sym::Sym_Str = :half_life)::Int
    assert_nonempty_gt0_finite_val(half_life, sym)
    return max(1, ceil(Int, half_life))
end
"""
    assert_ew_decay(decay::Real) -> nothing

Check that an exponential decay factor lies strictly between zero and one.

A decay of one keeps the full weight of every past observation, and a decay at or below zero is not a weight. Each constructor of an exponentially weighted Descriptor runs the check once.

# Arguments

  - `decay`: The decay factor.

# Validation

  - `0 < decay < 1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`half_life_decay`](@ref)
  - [`EWMean`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`DaysToCover`](@ref)
"""
function assert_ew_decay(decay::Real)::Nothing
    @argcheck(zero(decay) < decay < one(decay),
              DomainError(decay,
                          "decay is the factor an exponentially weighted recursion multiplies its state by, so it must lie strictly between zero and one, got $decay"))
    return nothing
end
"""
    ew_mean_series(R::AbstractMatrix{<:Real}, decay::Real, min_obs::Integer) -> Matrix{<:Real}

Run the exponentially weighted mean recursion down each column of a matrix.

This is the one recursion every exponentially weighted mean Descriptor shares, so the three archetypes of this file differ only in the matrix they build before they call it. Each asset carries its own state and its own count of valid observations. A cell that is not finite advances neither. The state holds there, and the recursion neither restarts nor reads the cell as a zero.

# Mathematical definition

```math
\\begin{align}
S_{t,i} &= \\begin{cases} \\lambda S_{t-1,i} + (1 - \\lambda) y_{t,i} & \\text{if } y_{t,i} \\text{ is finite} \\\\ S_{t-1,i} & \\text{otherwise} \\end{cases}\\,, \\quad S_{0,i} = 0\\,, \\\\
n_{t,i} &= \\begin{cases} n_{t-1,i} + 1 & \\text{if } y_{t,i} \\text{ is finite} \\\\ n_{t-1,i} & \\text{otherwise} \\end{cases}\\,, \\quad n_{0,i} = 0\\,, \\\\
m_{t,i} &= \\begin{cases} S_{t,i} & \\text{if } n_{t,i} \\geq \\texttt{min\\_obs} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

The recursion applies no bias correction, so the first value of an asset is ``(1 - \\lambda) y_{t,i}`` and not ``y_{t,i}``. The start at zero biases the first values toward zero, and the warm-up `min_obs` replaces them with `NaN`.

Where:

  - ``y_{t,i}``: Input of the recursion for asset ``i`` at observation ``t``, the entry `R[t, i]`.
  - $(math_dict[:lambda_ew])
  - ``S_{t,i}``: State of the recursion of asset ``i`` after observation ``t``.
  - ``n_{t,i}``: Count of the finite inputs of asset ``i`` up to observation ``t``.
  - ``m_{t,i}``: Value of the series for asset ``i`` at observation ``t``.

# Arguments

  - `R`: The input, `observations × assets`.
  - `decay`: The decay factor.
  - `min_obs`: The warm-up, in valid observations per asset.

# Returns

  - `S::Matrix{<:Real}`: The series, `observations × assets`, `NaN` before the warm-up of an asset ends. An integer input gives a floating point series.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_mean_series([1.0; 3.0; NaN; 5.0;;], 0.5, 1)
4×1 Matrix{Float64}:
 0.5
 1.75
 1.75
 3.375
```

# Related

  - [`EWMean`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`DaysToCover`](@ref)
"""
function ew_mean_series(R::AbstractMatrix{<:Real}, decay::Real,
                        min_obs::Integer)::Matrix{<:Real}
    Tf = float_if_integer(eltype(R))
    S = fill(Tf(NaN), size(R))
    for i in axes(R, 2)
        s = zero(Tf)
        n = 0
        for t in axes(R, 1)
            r = R[t, i]
            if isfinite(r)
                s = decay * s + (one(Tf) - decay) * r
                n += 1
            end
            if n >= min_obs
                S[t, i] = s
            end
        end
    end
    return S
end
"""
    assert_ew_ratio_side(x::Nothing, sym::Sym_Str) -> nothing
    assert_ew_ratio_side(x::Union{<:AbstractString,
                                  <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                         sym::Sym_Str) -> nothing
    assert_ew_ratio_side(x::AbstractVector{<:AbstractString}, sym::Sym_Str) -> nothing

Check one side of an exponentially weighted ratio.

This function checks the four forms of one side, so [`EWVolumeRatio`](@ref) states each rule once. `nothing` names the returns and reads nothing out of the Asset Panel, so it needs no check.

# Arguments

  - `x`: One side of the ratio, in any of the forms [`ew_ratio_values`](@ref) reads.
  - `sym`: The field's name, for the message.

# Validation

  - A name is not the empty string, and a combination holds at least one term whose name is not empty and whose coefficient is finite. Raises an [`IsEmptyError`](@ref) or a `DomainError`.
  - A product holds at least one name, and no name is the empty string. Raises an [`IsEmptyError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`EWVolumeRatio`](@ref)
  - [`ew_ratio_values`](@ref)
  - [`assert_panel_terms`](@ref)
"""
function assert_ew_ratio_side(::Nothing, ::Sym_Str)::Nothing
    return nothing
end
function assert_ew_ratio_side(x::Union{<:AbstractString,
                                       <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                              sym::Sym_Str)::Nothing
    assert_panel_terms(x, sym)
    return nothing
end
function assert_ew_ratio_side(x::AbstractVector{<:AbstractString}, sym::Sym_Str)::Nothing
    @argcheck(!isempty(x),
              IsEmptyError("$sym is a product of Panel Fields, so it needs at least one name"))
    for (k, name) in enumerate(x)
        @argcheck(!isempty(name),
                  IsEmptyError("factor $k of $sym names a Panel Field, so its name cannot be the empty string"))
    end
    return nothing
end
"""
    ew_ratio_values(rd::ReturnsResult, x::Nothing) -> Matrix{<:Real}
    ew_ratio_values(rd::ReturnsResult, x::AbstractString) -> Matrix{<:Real}
    ew_ratio_values(rd::ReturnsResult,
                    x::AbstractVector{<:Pair{<:AbstractString, <:Real}}) -> Matrix{<:Real}
    ew_ratio_values(rd::ReturnsResult, x::AbstractVector{<:AbstractString}) -> Matrix{<:Real}

Read one side of an exponentially weighted ratio out of a [`ReturnsResult`](@ref).

A side takes four forms. Julia selects the method from the form, and each method reads the side as follows.

 1. `nothing` reads the absolute returns `abs.(rd.X)`. Returns are not a Panel Field, so this is how a ratio names them.
 2. A name reads that one Panel Field, through [`descriptor_field_values`](@ref).
 3. A vector of `name => coefficient` pairs reads their sum, through [`descriptor_field_values`](@ref). `["a" => 1, "b" => -1]` reads `a - b`.
 4. A vector of names reads their product. `["adj_close", "adj_volume"]` reads a traded amount out of a price and a volume, which no single Panel Field carries.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl` unless `x` is `nothing`.
  - `x`: One side of the ratio.

# Validation

  - The rules of [`descriptor_field_values`](@ref) for every Panel Field the side names.

# Returns

  - `V::Matrix{<:Real}`: The values, `observations × assets`, `NaN` where a Panel Field the side names was not observed.

# Related

  - [`EWVolumeRatio`](@ref)
  - [`assert_ew_ratio_side`](@ref)
  - [`descriptor_field_values`](@ref)
"""
function ew_ratio_values(rd::ReturnsResult, ::Nothing)::Matrix{<:Real}
    return abs.(rd.X)
end
function ew_ratio_values(rd::ReturnsResult, x::AbstractString)::Matrix{<:Real}
    return descriptor_field_values(rd, x)
end
function ew_ratio_values(rd::ReturnsResult,
                         x::AbstractVector{<:Pair{<:AbstractString, <:Real}})::Matrix{<:Real}
    return descriptor_field_values(rd, x)
end
function ew_ratio_values(rd::ReturnsResult,
                         x::AbstractVector{<:AbstractString})::Matrix{<:Real}
    V = descriptor_field_values(rd, x[1])
    for k in 2:length(x)
        V .*= descriptor_field_values(rd, x[k])
    end
    return V
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted mean of the log returns, at every observation, with an optional skip.

This is the archetype of every momentum Descriptor. The skip separates a medium-term trend from the short-term reversal that follows it. At observation `t` the recursion reads the log return of observation `t - skip`, so the most recent `skip` observations never enter. The output is in log units, or in return units when `exponentiate` is set.

# Mathematical definition

```math
\\begin{align}
y_{t,i} &= \\begin{cases} \\log(1 + x_{t-s,\\,i}) & \\text{if } t > s \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,, \\\\
S_{t,i} &= \\lambda S_{t-1,i} + (1 - \\lambda) y_{t,i}\\,, \\quad S_{0,i} = 0\\,, \\\\
d_{t,i} &= \\begin{cases} \\exp(S_{t,i}) - 1 & \\text{if } \\texttt{exponentiate} \\\\ S_{t,i} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

The state moves only where ``y_{t,i}`` is finite, and holds everywhere else. The delay counts rows of the panel, not valid observations of the asset, so a gap in the returns of asset ``i`` reaches the recursion ``s`` rows later as a hold. ``d_{t,i}`` is `NaN` until ``n_{t,i} \\geq \\texttt{min\\_obs}``, and where asset ``i`` is inactive. [`ew_mean_series`](@ref) states the hold and the count in full.

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - $(math_dict[:x_ti_ret])
  - ``y_{t,i}``: Input of the recursion for asset ``i`` at observation ``t``.
  - ``s``: `skip`, the delay in observations.
  - $(math_dict[:lambda_ew])
  - ``S_{t,i}``: State of the recursion of asset ``i`` after observation ``t``.
  - ``n_{t,i}``: Count of the finite inputs ``y`` of asset ``i`` up to observation ``t``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWMean(; decay::Real, min_obs::Integer, skip::Integer = 0,
           exponentiate::Bool = false) -> EWMean

Keywords correspond to the struct's fields. `decay` and `min_obs` take no default, because they depend on the data frequency. [`EWMomentum`](@ref) states a half-life instead and converts it through [`half_life_decay`](@ref) and [`half_life_min_obs`](@ref).

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `skip >= 0`.

# Examples

```jldoctest
julia> EWMean(; decay = 0.5, min_obs = 2)
EWMean
         decay ┼ Float64: 0.5
       min_obs ┼ Int64: 2
          skip ┼ Int64: 0
  exponentiate ┴ Bool: false
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWMomentum`](@ref)
  - [`ew_mean_series`](@ref)
  - [`EWVolatility`](@ref)
"""
@concrete struct EWMean <: AbstractDescriptorEstimator
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    Number of the most recent observations the recursion does not read. A skip of zero reads the current return.
    """
    skip
    """
    Whether the Descriptor is in return units, `exp(S) - 1`, rather than in log units. The map from log units is increasing, so a cross-sectional ranking is the same either way.
    """
    exponentiate
    function EWMean(decay::Real, min_obs::Integer, skip::Integer, exponentiate::Bool)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        @argcheck(skip >= zero(skip),
                  DomainError(skip,
                              "skip is the number of the most recent observations the recursion does not read, so it cannot be negative, got $skip"))
        return new{typeof(decay), typeof(min_obs), typeof(skip), typeof(exponentiate)}(decay,
                                                                                       min_obs,
                                                                                       skip,
                                                                                       exponentiate)
    end
end
function EWMean(; decay::Real, min_obs::Integer, skip::Integer = 0,
                exponentiate::Bool = false)::EWMean
    return EWMean(decay, min_obs, skip, exponentiate)
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted mean of a ratio of Panel Fields, at every observation.

This is the archetype of every liquidity Descriptor built from a daily ratio. A raw ratio of a volume to a share count, or of a return to a traded amount, spikes around an earnings date or an index rebalance, so the Descriptor smooths it rather than reading one observation. Each side of the ratio is a Panel Field, a weighted sum of Panel Fields, a product of Panel Fields, or the absolute returns.

# Mathematical definition

```math
\\begin{align}
q_{t,i} &= \\begin{cases} \\dfrac{a_{t,i}}{b_{t,i}} & \\text{if } b_{t,i} > 0 \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,, \\\\
S_{t,i} &= \\lambda S_{t-1,i} + (1 - \\lambda) q_{t,i}\\,, \\quad S_{0,i} = 0\\,, \\\\
d_{t,i} &= S_{t,i}\\,.
\\end{align}
```

The state moves only where ``q_{t,i}`` is finite, and holds everywhere else. A missing Panel Field on either side, and a denominator that is not strictly positive, both hold the state. A negative value of a Panel Field that `nonneg` names is a data error, so [`descriptor`](@ref) refuses it before the recursion starts. ``d_{t,i}`` is `NaN` until ``n_{t,i} \\geq \\texttt{min\\_obs}``, and where asset ``i`` is inactive. [`ew_mean_series`](@ref) states the hold and the count in full.

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``a_{t,i}``: The numerator of asset ``i`` at observation ``t``.
  - ``b_{t,i}``: The denominator of asset ``i`` at observation ``t``.
  - ``q_{t,i}``: The ratio of asset ``i`` at observation ``t``.
  - $(math_dict[:lambda_ew])
  - ``S_{t,i}``: State of the recursion of asset ``i`` after observation ``t``.
  - ``n_{t,i}``: Count of the finite ratios ``q`` of asset ``i`` up to observation ``t``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWVolumeRatio(;
        num::Union{Nothing, <:AbstractString,
                   <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                   <:AbstractVector{<:AbstractString}},
        den::Union{Nothing, <:AbstractString,
                   <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                   <:AbstractVector{<:AbstractString}},
        nonneg::Option{<:VecStr} = nothing,
        gt0::Option{<:VecStr} = nothing,
        decay::Real,
        min_obs::Integer
    ) -> EWVolumeRatio

Keywords correspond to the struct's fields. `decay` and `min_obs` take no default, because they depend on the data frequency. [`EWShareTurnover`](@ref) and [`EWAmihudIlliquidity`](@ref) state a half-life instead. `nonneg` is `nothing` by default, because a side that combines Panel Fields, such as `["adj_volume" => 1, "short_interest" => -1]`, can be negative by design, and `gt0` is `nothing` for the same reason. The two presets name every Panel Field they read, and put a share count or a price in `gt0`.

## Validation

  - The rules of [`assert_ew_ratio_side`](@ref) for `num` and for `den`.
  - Every name in `nonneg` and in `gt0` is a Panel Field that `num` or `den` reads, see [`assert_panel_guard_names`](@ref).
  - $(val_dict[:decay])
  - $(val_dict[:min_obs])

# Examples

```jldoctest
julia> EWVolumeRatio(; num = \"adj_volume\", den = \"adj_shares_outstanding\", decay = 0.5,
                     min_obs = 2)
EWVolumeRatio
      num ┼ String: "adj_volume"
      den ┼ String: "adj_shares_outstanding"
   nonneg ┼ nothing
      gt0 ┼ nothing
    decay ┼ Float64: 0.5
  min_obs ┴ Int64: 2
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWShareTurnover`](@ref)
  - [`EWAmihudIlliquidity`](@ref)
  - [`ew_ratio_values`](@ref)
  - [`DaysToCover`](@ref)
"""
@concrete struct EWVolumeRatio <: AbstractDescriptorEstimator
    """
    The numerator, in one of four forms: `nothing` for the absolute returns, the name of one Panel Field, a vector of `name => coefficient` pairs read as their sum, or a vector of names read as their product.
    """
    num
    """
    The denominator, in the same forms. The ratio is `NaN` wherever it is not strictly positive.
    """
    den
    """
    $(field_dict[:nonneg_pnl])
    """
    nonneg
    """
    $(field_dict[:gt0_pnl])
    """
    gt0
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    function EWVolumeRatio(num::Union{Nothing, <:AbstractString,
                                      <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                                      <:AbstractVector{<:AbstractString}},
                           den::Union{Nothing, <:AbstractString,
                                      <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                                      <:AbstractVector{<:AbstractString}},
                           nonneg::Option{<:VecStr}, gt0::Option{<:VecStr}, decay::Real,
                           min_obs::Integer)
        assert_ew_ratio_side(num, :num)
        assert_ew_ratio_side(den, :den)
        known = vcat(panel_term_names(num), panel_term_names(den))
        assert_panel_guard_names(nonneg, known, :nonneg)
        assert_panel_guard_names(gt0, known, :gt0)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(num), typeof(den), typeof(nonneg), typeof(gt0), typeof(decay),
                   typeof(min_obs)}(num, den, nonneg, gt0, decay, min_obs)
    end
end
function EWVolumeRatio(;
                       num::Union{Nothing, <:AbstractString,
                                  <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                                  <:AbstractVector{<:AbstractString}},
                       den::Union{Nothing, <:AbstractString,
                                  <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                                  <:AbstractVector{<:AbstractString}},
                       nonneg::Option{<:VecStr} = nothing, gt0::Option{<:VecStr} = nothing,
                       decay::Real, min_obs::Integer)::EWVolumeRatio
    return EWVolumeRatio(num, den, nonneg, gt0, decay, min_obs)
end
"""
$(DocStringExtensions.TYPEDEF)

Ratio of a Panel Field to the exponentially weighted mean of a second one, at every observation.

The days to cover of a short position is the shares held short divided by a smoothed daily volume, so it measures how many days of ordinary trading it would take to buy the position back. [`EWVolumeRatio`](@ref) smooths the ratio. This archetype smooths the denominator alone and forms the ratio at the current observation, so the Descriptor shows the full change in the short interest at the observation where it happens. Only a strictly positive value of the denominator advances the recursion.

# Mathematical definition

```math
\\begin{align}
V_{t,i} &= \\begin{cases} \\lambda V_{t-1,i} + (1 - \\lambda) b_{t,i} & \\text{if } b_{t,i} > 0 \\\\ V_{t-1,i} & \\text{otherwise} \\end{cases}\\,, \\quad V_{0,i} = 0\\,, \\\\
d_{t,i} &= \\begin{cases} \\dfrac{a_{t,i}}{V_{t,i}} & \\text{if } V_{t,i} > 0 \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

A zero or a missing volume holds the state and does not count toward the warm-up. A missing short interest gives a `NaN` Descriptor at that observation, and the state of the volume moves all the same. A negative short interest or a negative volume is a data error, so [`descriptor`](@ref) refuses it before the recursion starts, for every Panel Field that `nonneg` names. With `nonneg = nothing`, a negative volume holds the state, and a negative short interest gives a negative Descriptor. ``d_{t,i}`` is `NaN` until ``n_{t,i} \\geq \\texttt{min\\_obs}``, and where asset ``i`` is inactive.

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``a_{t,i}``: Value of the Panel Field `num` for asset ``i`` at observation ``t``.
  - ``b_{t,i}``: Value of the Panel Field `den` for asset ``i`` at observation ``t``.
  - $(math_dict[:lambda_ew])
  - ``V_{t,i}``: Smoothed denominator of asset ``i`` after observation ``t``.
  - ``n_{t,i}``: Count of the strictly positive values ``b`` of asset ``i`` up to observation ``t``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DaysToCover(; num::AbstractString = "short_interest",
                den::AbstractString = "adj_volume",
                nonneg::Option{<:VecStr} = [num, den], half_life::Real = 21.0,
                decay::Real = half_life_decay(half_life),
                min_obs::Integer = half_life_min_obs(half_life)) -> DaysToCover

`num`, `den`, `nonneg`, `decay` and `min_obs` correspond to the struct's fields. `half_life` is not a field. It fixes the defaults of `decay` and `min_obs`, and the constructor keeps a `decay` or a `min_obs` that the caller passes as it is. The default half-life of `21` is about one month of daily observations. `nonneg` names both Panel Fields by default, because neither a short interest nor a volume can be negative.

## Validation

  - `!isempty(num)` and `!isempty(den)`.
  - Every name in `nonneg` is `num` or `den`, see [`assert_panel_guard_names`](@ref).
  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `0 < half_life < Inf`, where it fixes a default. Raises a `DomainError`.

# Examples

```jldoctest
julia> DaysToCover(; half_life = 2)
DaysToCover
      num ┼ String: "short_interest"
      den ┼ String: "adj_volume"
   nonneg ┼ Vector{String}: ["short_interest", "adj_volume"]
    decay ┼ Float64: 0.7071067811865476
  min_obs ┴ Int64: 2
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`ew_mean_series`](@ref)
"""
@concrete struct DaysToCover <: AbstractDescriptorEstimator
    """
    Name of the Panel Field the ratio divides, the number of shares held short.
    """
    num
    """
    Name of the Panel Field the recursion smooths, the traded volume. Only a strictly positive value advances it.
    """
    den
    """
    $(field_dict[:nonneg_pnl])
    """
    nonneg
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    function DaysToCover(num::AbstractString, den::AbstractString, nonneg::Option{<:VecStr},
                         decay::Real, min_obs::Integer)
        assert_panel_terms(num, :num)
        assert_panel_terms(den, :den)
        assert_panel_guard_names(nonneg, [String(num), String(den)], :nonneg)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(num), typeof(den), typeof(nonneg), typeof(decay),
                   typeof(min_obs)}(num, den, nonneg, decay, min_obs)
    end
end
function DaysToCover(; num::AbstractString = "short_interest",
                     den::AbstractString = "adj_volume",
                     nonneg::Option{<:VecStr} = [num, den], half_life::Real = 21.0,
                     decay::Real = half_life_decay(half_life),
                     min_obs::Integer = half_life_min_obs(half_life))::DaysToCover
    return DaysToCover(num, den, nonneg, decay, min_obs)
end
"""
    descriptor(de::EWMean, rd::ReturnsResult) -> Matrix{<:Real}
    descriptor(de::EWVolumeRatio, rd::ReturnsResult) -> Matrix{<:Real}
    descriptor(de::DaysToCover, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted mean Descriptor from a [`ReturnsResult`](@ref).

Each of the three archetypes builds one `observations × assets` matrix and runs it through [`ew_mean_series`](@ref). [`descriptor_active_fill!`](@ref) then writes `NaN` into the inactive cells. An asset that is listed but has no value at an observation holds its state there, so a gap in the data neither restarts the recursion nor enters it as a zero.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`EWMean`](@ref): check the returns through [`assert_log_returns`](@ref), then run the recursion over `log1p(rd.X)` delayed by `skip` observations. The first `skip` rows have no input. Take `expm1` of the series when `exponentiate` is set.
 2. [`EWVolumeRatio`](@ref): check the `nonneg` and `gt0` guards through [`assert_panel_field_sign`](@ref), read both sides through [`ew_ratio_values`](@ref), divide them through [`positive_divide`](@ref), and run the recursion over the ratio.
 3. [`DaysToCover`](@ref): check the `nonneg` guard through [`assert_panel_field_sign`](@ref), run the recursion over the denominator's Panel Field, with every value that is not strictly positive read as a `NaN`, then divide the numerator's Panel Field by the series through [`positive_divide`](@ref).

Every method then writes `NaN` into the inactive cells.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`descriptor_field_values`](@ref) for every Panel Field the estimator names.
  - The rule of [`assert_log_returns`](@ref) for an [`EWMean`](@ref).
  - The rule of [`assert_panel_field_sign`](@ref) for the `nonneg` guard of an [`EWVolumeRatio`](@ref) or a [`DaysToCover`](@ref), and for the `gt0` guard of an [`EWVolumeRatio`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"adj_volume\",
                                            vals = [10.0 20.0; 30.0 40.0; 50.0 60.0]),
                          NumericPanelInput(; name = \"adj_shares_outstanding\",
                                            vals = [100.0 100.0; 100.0 200.0;
                                                    100.0 200.0])]; amsk = trues(3, 2),
                         emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], pnl = pnl);

julia> descriptor(EWMean(; decay = 0.5, min_obs = 1), rd)
3×2 Matrix{Float64}:
  0.0476551   0.0911608
 -0.0288527   0.0455804
  0.00996873  0.0471853

julia> descriptor(EWVolumeRatio(; num = \"adj_volume\", den = \"adj_shares_outstanding\", decay = 0.5,
                                min_obs = 1), rd)
3×2 Matrix{Float64}:
 0.05    0.1
 0.175   0.15
 0.3375  0.225
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`EWMean`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`DaysToCover`](@ref)
  - [`ew_mean_series`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::EWMean, rd::ReturnsResult)::Matrix{<:Real}
    pnl = descriptor_asset_panel(rd)
    X = rd.X
    assert_log_returns(X)
    Tf = float_if_integer(eltype(X))
    R = fill(Tf(NaN), size(X))
    skip = de.skip
    for i in axes(X, 2), t in (skip + 1):size(X, 1)
        R[t, i] = log1p(X[t - skip, i])
    end
    S = ew_mean_series(R, de.decay, de.min_obs)
    D = de.exponentiate ? expm1.(S) : S
    descriptor_active_fill!(D, pnl)
    return D
end
function descriptor(de::EWVolumeRatio, rd::ReturnsResult)::Matrix{<:Real}
    pnl = descriptor_asset_panel(rd)
    assert_panel_field_sign(rd, de.nonneg, false)
    assert_panel_field_sign(rd, de.gt0, true)
    A = ew_ratio_values(rd, de.num)
    B = ew_ratio_values(rd, de.den)
    D = ew_mean_series(positive_divide.(A, B), de.decay, de.min_obs)
    descriptor_active_fill!(D, pnl)
    return D
end
function descriptor(de::DaysToCover, rd::ReturnsResult)::Matrix{<:Real}
    pnl = descriptor_asset_panel(rd)
    assert_panel_field_sign(rd, de.nonneg, false)
    A = descriptor_field_values(rd, de.num)
    B = descriptor_field_values(rd, de.den)
    Tf = eltype(B)
    V = [b > zero(b) ? b : Tf(NaN) for b in B]
    D = positive_divide.(A, ew_mean_series(V, de.decay, de.min_obs))
    descriptor_active_fill!(D, pnl)
    return D
end
"""
    EWMomentum(; half_life::Real = 87.0, skip::Integer = 21,
               exponentiate::Bool = false,
               decay::Real = half_life_decay(half_life),
               min_obs::Integer = half_life_min_obs(half_life)) -> EWMean

Exponentially weighted mean of the log returns of the past year, less the past month.

The value is the recursion of [`EWMean`](@ref) over `log1p(r)` delayed by `skip` observations. A stock that has risen over the medium term tends to keep rising, and the skip drops the most recent month, where the opposite holds. The default half-life of `87` weights about as far back as a window of `252` daily observations.

# Arguments

  - `half_life`: Half-life of the recursion, in observations. It fixes the defaults of `decay` and `min_obs`.
  - `skip`: Number of the most recent observations the recursion does not read. `21` is about one month of daily observations.
  - `exponentiate`: Whether the Descriptor is in return units.
  - `decay`: Decay factor of the recursion.
  - `min_obs`: Warm-up, in valid observations per asset.

# Validation

  - The rules of [`EWMean`](@ref).
  - `0 < half_life < Inf`, where it fixes a default. Raises a `DomainError`.

# Returns

  - `de::EWMean`: The estimator, with the half-life and the skip fixed.

# Examples

```jldoctest
julia> EWMomentum(; half_life = 5)
EWMean
         decay ┼ Float64: 0.8705505632961241
       min_obs ┼ Int64: 5
          skip ┼ Int64: 21
  exponentiate ┴ Bool: false
```

# Related

  - [`EWMean`](@ref)
  - [`descriptor`](@ref)
  - [`half_life_decay`](@ref)
  - [`EWVolatility`](@ref)
"""
function EWMomentum(; half_life::Real = 87.0, skip::Integer = 21,
                    exponentiate::Bool = false, decay::Real = half_life_decay(half_life),
                    min_obs::Integer = half_life_min_obs(half_life))::EWMean
    return EWMean(; decay = decay, min_obs = min_obs, skip = skip,
                  exponentiate = exponentiate)
end
"""
    EWShareTurnover(; num::AbstractString = "adj_volume",
                    den::AbstractString = "adj_shares_outstanding",
                    nonneg::Option{<:VecStr} = [num, den],
                    gt0::Option{<:VecStr} = [den], half_life::Real = 21.0,
                    decay::Real = half_life_decay(half_life),
                    min_obs::Integer = half_life_min_obs(half_life)) -> EWVolumeRatio

Exponentially weighted share turnover, the fraction of the shares outstanding that changes hands.

The value is the recursion of [`EWVolumeRatio`](@ref) over `adj_volume / adj_shares_outstanding`. A stock whose shares turn over slowly is harder to trade, and its expected return includes a premium for that illiquidity. The default half-life of `21` is about one month of daily observations.

# Arguments

  - `num`: Name of the traded volume Panel Field.
  - `den`: Name of the shares outstanding Panel Field. Both must use the same split adjustment.
  - `nonneg`: Names of the Panel Fields that must not be negative, both by default, because neither a volume nor a share count can be negative. `nothing` turns the check off.
  - `gt0`: Names of the Panel Fields that must be strictly positive, the share count by default, because a listed asset has shares outstanding. A volume can be zero on a day with no trade, so it is not in `gt0`. `nothing` turns the check off, and a share count at or below zero then gives no ratio.
  - `half_life`: Half-life of the recursion, in observations. It fixes the defaults of `decay` and `min_obs`.
  - `decay`: Decay factor of the recursion.
  - `min_obs`: Warm-up, in valid observations per asset.

# Validation

  - The rules of [`EWVolumeRatio`](@ref).
  - `0 < half_life < Inf`, where it fixes a default. Raises a `DomainError`.

# Returns

  - `de::EWVolumeRatio`: The estimator, with the two Panel Fields and the half-life fixed.

# Examples

```jldoctest
julia> EWShareTurnover(; half_life = 2)
EWVolumeRatio
      num ┼ String: \"adj_volume\"
      den ┼ String: \"adj_shares_outstanding\"
   nonneg ┼ Vector{String}: [\"adj_volume\", \"adj_shares_outstanding\"]
      gt0 ┼ Vector{String}: [\"adj_shares_outstanding\"]
    decay ┼ Float64: 0.7071067811865476
  min_obs ┴ Int64: 2
```

# Related

  - [`EWVolumeRatio`](@ref)
  - [`descriptor`](@ref)
  - [`EWAmihudIlliquidity`](@ref)
  - [`DaysToCover`](@ref)
"""
function EWShareTurnover(; num::AbstractString = "adj_volume",
                         den::AbstractString = "adj_shares_outstanding",
                         nonneg::Option{<:VecStr} = [num, den],
                         gt0::Option{<:VecStr} = [den], half_life::Real = 21.0,
                         decay::Real = half_life_decay(half_life),
                         min_obs::Integer = half_life_min_obs(half_life))::EWVolumeRatio
    return EWVolumeRatio(; num = num, den = den, nonneg = nonneg, gt0 = gt0, decay = decay,
                         min_obs = min_obs)
end
"""
    EWAmihudIlliquidity(; den::AbstractVector{<:AbstractString} = ["adj_close",
                                                                   "adj_volume"],
                        nonneg::Option{<:VecStr} = copy(den),
                        gt0::Option{<:VecStr} = den[1:1], half_life::Real = 63.0,
                        decay::Real = half_life_decay(half_life),
                        min_obs::Integer = half_life_min_obs(half_life)) -> EWVolumeRatio

Exponentially weighted price impact, the absolute return earned per unit of traded amount.

The value is the recursion of [`EWVolumeRatio`](@ref) over `abs(r) / (adj_close * adj_volume)`. A stock whose price moves far on a small amount traded is expensive to trade, and the illiquidity premium is the return that pays for that cost. The numerator is the absolute returns. Returns are not a Panel Field, so the numerator is `nothing`. A day with zero volume has no ratio, so it holds the state. The default half-life of `63` is about three months of daily observations.

# Arguments

  - `den`: Names of the two Panel Fields whose product is the traded amount, a price and a volume on the same split adjustment.
  - `nonneg`: Names of the Panel Fields that must not be negative, both names of `den` by default, because neither a price nor a volume can be negative. `nothing` turns the check off.
  - `gt0`: Names of the Panel Fields that must be strictly positive, the first name of `den`, the price, by default, because a price is positive by construction. A volume can be zero on a day with no trade, so it is not in `gt0`. `nothing` turns the check off.
  - `half_life`: Half-life of the recursion, in observations. It fixes the defaults of `decay` and `min_obs`.
  - `decay`: Decay factor of the recursion.
  - `min_obs`: Warm-up, in valid observations per asset.

# Validation

  - The rules of [`EWVolumeRatio`](@ref).
  - `0 < half_life < Inf`, where it fixes a default. Raises a `DomainError`.

# Returns

  - `de::EWVolumeRatio`: The estimator, with the denominator and the half-life fixed.

# Examples

```jldoctest
julia> EWAmihudIlliquidity(; half_life = 2)
EWVolumeRatio
      num ┼ nothing
      den ┼ Vector{String}: [\"adj_close\", \"adj_volume\"]
   nonneg ┼ Vector{String}: [\"adj_close\", \"adj_volume\"]
      gt0 ┼ Vector{String}: [\"adj_close\"]
    decay ┼ Float64: 0.7071067811865476
  min_obs ┴ Int64: 2
```

# Related

  - [`EWVolumeRatio`](@ref)
  - [`descriptor`](@ref)
  - [`EWShareTurnover`](@ref)
  - [`ew_ratio_values`](@ref)
"""
function EWAmihudIlliquidity(;
                             den::AbstractVector{<:AbstractString} = ["adj_close",
                                                                      "adj_volume"],
                             nonneg::Option{<:VecStr} = copy(den),
                             gt0::Option{<:VecStr} = den[1:1], half_life::Real = 63.0,
                             decay::Real = half_life_decay(half_life),
                             min_obs::Integer = half_life_min_obs(half_life))::EWVolumeRatio
    return EWVolumeRatio(; num = nothing, den = den, nonneg = nonneg, gt0 = gt0,
                         decay = decay, min_obs = min_obs)
end

export EWMean, EWVolumeRatio, DaysToCover, EWMomentum, EWShareTurnover, EWAmihudIlliquidity
