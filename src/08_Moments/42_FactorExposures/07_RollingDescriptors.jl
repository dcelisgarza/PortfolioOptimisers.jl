"""
    assert_rolling_sign(sign::Real) -> nothing

Check that the sign of a rolling log-return Descriptor is `1` or `-1`.

The Descriptor sums log returns over a window, and the optional exponentiation inverts that sum with `expm1`. A multiplier other than `1` or `-1` scales the sum, and the exponentiation of a scaled sum is no longer a return, so the field takes the two values that keep the reading meaningful: `1` reads the window as momentum, and `-1` reads it as reversal.

# Arguments

  - `sign`: The multiplier applied to the window sum.

# Validation

  - `sign == 1 || sign == -1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`RollingLogReturn`](@ref)
  - [`RollingMomentum`](@ref)
  - [`Reversal`](@ref)
"""
function assert_rolling_sign(sign::Real)::Nothing
    @argcheck(isone(sign) || isone(-sign),
              DomainError(sign,
                          "the sign of a rolling log-return Descriptor reads the window as momentum or as reversal, so it must be 1 or -1, got $sign"))
    return nothing
end
"""
    assert_log_returns(X::AbstractMatrix{<:Real}) -> nothing

Check that every return that is not missing is greater than `-1`.

A rolling log-return Descriptor takes the logarithm of one plus each return. A return of `-1` is a total loss, and the logarithm is undefined below it, so the check refuses the whole matrix rather than write an infinity into one cell of the Descriptor. A missing return is a `NaN`, and it passes the check.

# Arguments

  - `X`: The returns, `observations × assets`.

# Validation

  - Every entry of `X` that is not `NaN` is greater than `-1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`RollingLogReturn`](@ref)
  - [`descriptor`](@ref)
"""
function assert_log_returns(X::AbstractMatrix{<:Real})::Nothing
    k = findfirst(x -> !isnan(x) && x <= -one(x), X)
    @argcheck(isnothing(k),
              DomainError(isnothing(k) ? NaN : X[k],
                          "a rolling log-return Descriptor takes the logarithm of one plus each return, so every return that is not missing must be greater than -1, and it is $(isnothing(k) ? NaN : X[k]) at observation $(isnothing(k) ? 0 : k[1]) for asset $(isnothing(k) ? 0 : k[2]). A return at or below -1 is a data error, so clean the input rather than pass it through."))
    return nothing
end
"""
    descriptor_returns(rd::ReturnsResult) -> Tuple{Matrix{<:Real}, AssetPanel}

Read the returns and the Asset Panel a rolling Descriptor works on.

Returns are not a Panel Field, so a Descriptor that reads them reads `rd.X` rather than the feature matrix. This is the one route to that pair, and every rolling Descriptor reads through it. The returns come back as a floating point matrix of their own, so a missing return is a `NaN` the Descriptor tests directly.

# Arguments

  - $(arg_dict[:rd]) It must carry returns in `rd.X` and an Asset Panel in `rd.pnl`.

# Validation

  - `!isnothing(rd.X)`. Raises an [`IsNothingError`](@ref).
  - `!isnothing(rd.pnl)`. Raises an [`IsNothingError`](@ref).

The two shapes need no check of their own: [`ReturnsResult`](@ref) binds the observation axis and the asset axis of the feature matrix to those of the returns, so an Asset Panel that reaches a carrier always matches the returns beside it.

# Returns

  - `X::Matrix{<:Real}`: The returns, `observations × assets`.
  - `pnl::AssetPanel`: The Asset Panel the carrier holds.

# Related

  - [`descriptor`](@ref)
  - [`RollingLogReturn`](@ref)
  - [`RollingMax`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_returns(rd::ReturnsResult)
    X = rd.X
    pnl = rd.pnl
    @argcheck(!isnothing(X),
              IsNothingError("a rolling Descriptor reads returns, and rd.X is nothing. Build the carrier with the returns matrix the Asset Panel was drawn on."))
    @argcheck(!isnothing(pnl),
              IsNothingError("a rolling Descriptor reads the active mask of an Asset Panel, and rd.pnl is nothing. Build the carrier with the `pnl`, `nz` and `Z` that asset_panel returns."))
    return Matrix{float(eltype(X))}(X), pnl
end
"""
    rolling_window_max(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                       i::Integer, rows) -> Real

Take the largest return of one asset over one window of observations.

This is the window scan of [`RollingMax`](@ref), written once so the verb reads as one loop over the observations. The scan stops at the first observation the active mask refuses, because the window is then `NaN` whatever the rest of it holds.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `amsk`: The active mask of the Asset Panel, `observations × assets`.
  - `i`: The asset's column.
  - `rows`: The observations of the window.

# Returns

  - `m::Real`: The largest return that is not missing, or `NaN` where one observation of the window is inactive or every return of the window is missing.

# Related

  - [`RollingMax`](@ref)
  - [`descriptor`](@ref)
"""
function rolling_window_max(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                            i::Integer, rows)::Real
    Tf = eltype(X)
    m = Tf(NaN)
    for k in rows
        if !(amsk[k, i])
            return Tf(NaN)
        end
        x = X[k, i]
        if !isnan(x) && (isnan(m) || x > m)
            m = x
        end
    end
    return m
end
"""
$(DocStringExtensions.TYPEDEF)

Sum of log returns over a fixed window that ends a fixed number of observations back, at every observation.

This is the archetype of every rolling log-return Descriptor: medium-term momentum, which skips the most recent month, and short-term reversal, which negates the sum of the last month. The window holds `window` observations and ends `skip` observations before the current one, so the two readings differ only in their window, their skip and their sign.

The Descriptor is `NaN` unless every observation of the window is active, which is one rule for three cases: the warm-up at the start of the sample, an asset that lists late, and a gap in the middle of a listing. An active observation whose return is missing contributes zero to the sum, because a holiday is not a loss.

The output is a log return by default. A log cumulative return is more symmetric than a simple one, which suits the cross-sectional standardisation that reads it, and the logarithm is monotone, so the two orderings agree. Set `exponentiate` to read the simple return instead.

# Mathematical definition

```math
\\begin{align}
x_{t,i} &= \\begin{cases} \\log(1 + r_{t,i}) & \\text{if } r_{t,i} \\text{ is observed} \\\\ 0 & \\text{otherwise} \\end{cases}\\,,\\\\
S_{t,i} &= \\sum_{k = t - s - w + 1}^{t - s} x_{k,i}\\,,\\\\
d_{t,i} &= \\begin{cases} \\sigma S_{t,i} & \\text{if } t \\ge s + w \\text{, every } a_{k,i} \\text{ of the window holds, and not } \\texttt{exponentiate} \\\\ \\exp(\\sigma S_{t,i}) - 1 & \\text{same, and } \\texttt{exponentiate} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``r_{t,i}``: Return of asset ``i`` at observation ``t``.
  - ``a_{t,i}``: Active mask of the Asset Panel.
  - ``w``: The window, in observations.
  - ``s``: The skip, in observations.
  - ``\\sigma``: The sign.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RollingLogReturn(; window::Integer, skip::Integer = 0, sign::Real = 1,
                     exponentiate::Bool = false) -> RollingLogReturn

Keywords correspond to the struct's fields. `window` takes no default, because it depends on the data frequency: `252` is one year of daily observations. The named Descriptors fix all four.

## Validation

  - `window > 0`.
  - `skip >= 0`.
  - `sign == 1 || sign == -1`.

# Examples

```jldoctest
julia> RollingLogReturn(; window = 252, skip = 21)
RollingLogReturn
        window ┼ Int64: 252
          skip ┼ Int64: 21
          sign ┼ Int64: 1
  exponentiate ┴ Bool: false
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`RollingMax`](@ref)
  - [`RollingMomentum`](@ref)
  - [`Reversal`](@ref)
"""
@concrete struct RollingLogReturn <: AbstractDescriptorEstimator
    """
    Number of observations in the window.
    """
    window
    """
    Number of most recent observations the window excludes. The window ends at observation `t - skip`.
    """
    skip
    """
    Multiplier of the window sum. `1` reads the window as momentum, `-1` reads it as reversal.
    """
    sign
    """
    Whether the output is the simple return over the window rather than the log return.
    """
    exponentiate
    function RollingLogReturn(window::Integer, skip::Integer, sign::Real,
                              exponentiate::Bool)
        assert_gt0(window, :window)
        assert_nonneg(skip, :skip)
        assert_rolling_sign(sign)
        return new{typeof(window), typeof(skip), typeof(sign), typeof(exponentiate)}(window,
                                                                                     skip,
                                                                                     sign,
                                                                                     exponentiate)
    end
end
function RollingLogReturn(; window::Integer, skip::Integer = 0, sign::Real = 1,
                          exponentiate::Bool = false)::RollingLogReturn
    return RollingLogReturn(window, skip, sign, exponentiate)
end
"""
$(DocStringExtensions.TYPEDEF)

Maximum return over a fixed trailing window, at every observation.

This is the lottery demand Descriptor. An asset whose recent path holds one very large positive return attracts speculative demand, and the cross-section of that reading prices it.

The Descriptor is `NaN` unless every observation of the window is active, which covers the warm-up at the start of the sample, an asset that lists late, and a gap in the middle of a listing. A missing return inside an active window is ignored rather than counted as zero, so the maximum reads the returns that exist. A window whose returns are all missing is `NaN`, because a maximum of nothing is not a number.

# Mathematical definition

```math
\\begin{align}
d_{t,i} &= \\begin{cases} \\max \\{ r_{k,i} : k \\in [t - w + 1,\\, t] \\text{, } r_{k,i} \\text{ observed} \\} & \\text{if } t \\ge w \\text{, every } a_{k,i} \\text{ of the window holds, and one } r_{k,i} \\text{ is observed} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``r_{t,i}``: Return of asset ``i`` at observation ``t``.
  - ``a_{t,i}``: Active mask of the Asset Panel.
  - ``w``: The window, in observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RollingMax(; window::Integer) -> RollingMax

Keywords correspond to the struct's fields. `window` takes no default, because it depends on the data frequency: `21` is one month of daily observations. [`MaxReturn`](@ref) fixes it there.

## Validation

  - `window > 0`.

# Examples

```jldoctest
julia> RollingMax(; window = 21)
RollingMax
  window ┴ Int64: 21
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`RollingLogReturn`](@ref)
  - [`MaxReturn`](@ref)
  - [`rolling_window_max`](@ref)
"""
@concrete struct RollingMax <: AbstractDescriptorEstimator
    """
    Number of observations in the trailing window.
    """
    window
    function RollingMax(window::Integer)
        assert_gt0(window, :window)
        return new{typeof(window)}(window)
    end
end
function RollingMax(; window::Integer)::RollingMax
    return RollingMax(window)
end
"""
    descriptor(de::RollingLogReturn, rd::ReturnsResult) -> Matrix{<:Real}
    descriptor(de::RollingMax, rd::ReturnsResult) -> Matrix{<:Real}

Compute a rolling Descriptor of the return path.

Both members read the returns of `rd.X` and the active mask of the Asset Panel, and no Panel Field. Both answer `NaN` on every observation whose window is not wholly active, and on every inactive cell.

# Algorithm

 1. Read the returns and the Asset Panel through [`descriptor_returns`](@ref).
 2. For [`RollingLogReturn`](@ref), refuse a return at or below `-1` through [`assert_log_returns`](@ref), then take the cumulative sums of `log1p` of the returns and of the active mask along the observations. A missing return contributes zero. The window of observation `t` runs from `t - skip - window + 1` to `t - skip`, and its sum is one difference of the cumulative sums. Write that sum, multiplied by the sign, where the active count of the window equals the window, and `expm1` of it when `exponentiate` is set.
 3. For [`RollingMax`](@ref), scan the trailing window of each observation through [`rolling_window_max`](@ref), which writes the largest return that is not missing where every observation of the window is active and one return exists.
 4. Write `NaN` into every inactive cell through [`descriptor_active_fill!`](@ref).

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry returns in `rd.X` and an Asset Panel in `rd.pnl`.

# Validation

  - The validation of [`descriptor_returns`](@ref), and of [`assert_log_returns`](@ref) for [`RollingLogReturn`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 1.0; 1.0 1.0; 1.0 1.0; 1.0 1.0])];
                         amsk = trues(4, 2), emsk = trues(4, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 -0.05; 0.2 0.1; -0.1 0.05; 0.05 NaN], res...);

julia> descriptor(RollingLogReturn(; window = 2), rd)
4×2 Matrix{Float64}:
 NaN          NaN
   0.277632     0.0440169
   0.076961     0.1441
  -0.0565704    0.0487902

julia> descriptor(Reversal(; window = 2), rd)
4×2 Matrix{Float64}:
 NaN          NaN
  -0.277632    -0.0440169
  -0.076961    -0.1441
   0.0565704   -0.0487902

julia> descriptor(RollingMax(; window = 2), rd)
4×2 Matrix{Float64}:
 NaN     NaN
   0.2     0.1
   0.2     0.1
   0.05    0.05
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`RollingLogReturn`](@ref)
  - [`RollingMax`](@ref)
  - [`descriptor_returns`](@ref)
  - [`assert_log_returns`](@ref)
  - [`rolling_window_max`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::RollingLogReturn, rd::ReturnsResult)::Matrix{<:Real}
    X, pnl = descriptor_returns(rd)
    assert_log_returns(X)
    amsk = pnl.amsk
    Tf = eltype(X)
    T, N = size(X)
    D = fill(Tf(NaN), T, N)
    cs = zeros(Tf, T + 1, N)
    ac = zeros(Int, T + 1, N)
    for i in axes(X, 2), t in 1:T
        x = X[t, i]
        cs[t + 1, i] = cs[t, i] + (isnan(x) ? zero(Tf) : log1p(x))
        ac[t + 1, i] = ac[t, i] + amsk[t, i]
    end
    window, skip, sgn = de.window, de.skip, de.sign
    for i in axes(X, 2), t in (skip + window):T
        e = t - skip + 1
        s = e - window
        if ac[e, i] - ac[s, i] == window
            v = sgn * (cs[e, i] - cs[s, i])
            D[t, i] = de.exponentiate ? expm1(v) : v
        end
    end
    descriptor_active_fill!(D, pnl)
    return D
end
function descriptor(de::RollingMax, rd::ReturnsResult)::Matrix{<:Real}
    X, pnl = descriptor_returns(rd)
    amsk = pnl.amsk
    T, N = size(X)
    D = fill(eltype(X)(NaN), T, N)
    window = de.window
    for i in axes(X, 2), t in window:T
        D[t, i] = rolling_window_max(X, amsk, i, (t - window + 1):t)
    end
    descriptor_active_fill!(D, pnl)
    return D
end
"""
    RollingMomentum(; window::Integer = 252, skip::Integer = 21, sign::Real = 1,
                    exponentiate::Bool = false) -> RollingLogReturn

Sum of log returns over one year, ending one month back.

This is the classic twelve-minus-one momentum reading. The skip separates the medium-term momentum the window measures from the short-term reversal of the most recent month, which [`Reversal`](@ref) reads on its own.

# Arguments

  - `window`: Number of observations in the window. `252` is one year of daily observations.
  - `skip`: Number of most recent observations the window excludes. `21` is one month of daily observations.
  - `sign`: Multiplier of the window sum.
  - `exponentiate`: Whether the output is the simple return over the window rather than the log return.

# Returns

  - `de::RollingLogReturn`: The estimator, with the window, the skip and the sign fixed.

# Examples

```jldoctest
julia> RollingMomentum()
RollingLogReturn
        window ┼ Int64: 252
          skip ┼ Int64: 21
          sign ┼ Int64: 1
  exponentiate ┴ Bool: false
```

# Related

  - [`RollingLogReturn`](@ref)
  - [`descriptor`](@ref)
  - [`Reversal`](@ref)
  - [`MaxReturn`](@ref)
"""
function RollingMomentum(; window::Integer = 252, skip::Integer = 21, sign::Real = 1,
                         exponentiate::Bool = false)::RollingLogReturn
    return RollingLogReturn(; window = window, skip = skip, sign = sign,
                            exponentiate = exponentiate)
end
"""
    Reversal(; window::Integer = 21, skip::Integer = 0, sign::Real = -1,
             exponentiate::Bool = false) -> RollingLogReturn

Negated sum of log returns over one month, ending at the current observation.

This is the short-term reversal reading. A high value says that the asset lost ground recently, which temporary price pressure, the provision of liquidity and the microstructure of the market tend to reverse. It is the counterpart of [`RollingMomentum`](@ref), whose skip excludes the window this Descriptor reads.

# Arguments

  - `window`: Number of observations in the window. `21` is one month of daily observations, `5` is one week, and `1` is one day.
  - `skip`: Number of most recent observations the window excludes.
  - `sign`: Multiplier of the window sum.
  - `exponentiate`: Whether the output is the simple return over the window rather than the log return.

# Returns

  - `de::RollingLogReturn`: The estimator, with the window, the skip and the sign fixed.

# Examples

```jldoctest
julia> Reversal()
RollingLogReturn
        window ┼ Int64: 21
          skip ┼ Int64: 0
          sign ┼ Int64: -1
  exponentiate ┴ Bool: false
```

# Related

  - [`RollingLogReturn`](@ref)
  - [`descriptor`](@ref)
  - [`RollingMomentum`](@ref)
  - [`MaxReturn`](@ref)
"""
function Reversal(; window::Integer = 21, skip::Integer = 0, sign::Real = -1,
                  exponentiate::Bool = false)::RollingLogReturn
    return RollingLogReturn(; window = window, skip = skip, sign = sign,
                            exponentiate = exponentiate)
end
"""
    MaxReturn(; window::Integer = 21) -> RollingMax

Maximum return over one month.

This is the lottery demand reading at the horizon its source uses. An asset whose last month holds one very large positive return earns a lower return afterwards, which the cross-section reads as the price of a lottery-like payoff.

# Arguments

  - `window`: Number of observations in the trailing window. `21` is one month of daily observations, and `5` is one week.

# Returns

  - `de::RollingMax`: The estimator, with the window fixed.

# Examples

```jldoctest
julia> MaxReturn()
RollingMax
  window ┴ Int64: 21
```

# Related

  - [`RollingMax`](@ref)
  - [`descriptor`](@ref)
  - [`RollingMomentum`](@ref)
  - [`Reversal`](@ref)
"""
function MaxReturn(; window::Integer = 21)::RollingMax
    return RollingMax(; window = window)
end

export RollingLogReturn, RollingMax, RollingMomentum, Reversal, MaxReturn
