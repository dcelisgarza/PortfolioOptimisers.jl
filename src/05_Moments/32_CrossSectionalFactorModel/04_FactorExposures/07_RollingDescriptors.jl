"""
    assert_rolling_sign(sign::Real) -> nothing

Check that the sign of a rolling log-return Descriptor is `1` or `-1`.

The Descriptor sums log returns over a window, and the optional exponentiation turns that sum back into a simple return with `expm1`. A multiplier other than `1` or `-1` scales the sum, and the exponential of a scaled sum is not a return. So the field takes only two values. A sign of `1` reads the window as momentum, and a sign of `-1` reads it as reversal.

# Arguments

  - $(arg_dict[:sign_roll])

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
    descriptor_returns(rd::ReturnsResult) -> Tuple{Matrix{<:Real}, AssetPanel}

Read the returns and the Asset Panel a rolling Descriptor works on.

Returns are not a Panel Field, so a Descriptor that reads them reads `rd.X` and not the feature matrix. Every rolling Descriptor reads the pair through this function. The returns come back as a new matrix in `float_if_integer(eltype(rd.X))`. An integer panel is floated, so its Descriptor can hold the `NaN` of the warm-up. A `Rational` panel is kept, and it cannot hold that `NaN`.

A return that is missing is a `NaN`, and a Descriptor tests for it. A return that is infinite is refused, because a sum over a window is one difference of two cumulative sums. One infinite return makes every later cumulative sum infinite, and the difference of two of them is `NaN`, also in a window that does not hold the infinite return.

# Arguments

  - $(arg_dict[:rd]) It must carry returns in `rd.X` and an Asset Panel in `rd.pnl`.

# Validation

  - `!isnothing(rd.X)`. Raises an [`IsNothingError`](@ref).
  - `!isnothing(rd.pnl)`. Raises an [`IsNothingError`](@ref).
  - Every entry of `rd.X` that is not `NaN` is finite. Raises a `DomainError`.

The two shapes need no check of their own. [`ReturnsResult`](@ref) binds the observation axis and the asset axis of the feature matrix to those of the returns, so an Asset Panel that reaches a `ReturnsResult` always matches the returns beside it.

# Returns

  - `X::Matrix{<:Real}`: The returns, `observations × assets`, in `float_if_integer(eltype(rd.X))`.
  - `pnl::AssetPanel`: The Asset Panel of the `ReturnsResult`.

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
              IsNothingError("a rolling Descriptor reads returns, and rd.X is nothing. Build the ReturnsResult with the returns matrix the Asset Panel was drawn on."))
    @argcheck(!isnothing(pnl),
              IsNothingError("a rolling Descriptor reads the active mask of an Asset Panel, and rd.pnl is nothing. Build the ReturnsResult with the `pnl` that asset_panel returns."))
    Xf = Matrix{float_if_integer(eltype(X))}(X)
    k = findfirst(isinf, Xf)
    @argcheck(isnothing(k),
              DomainError(Xf[k],
                          "a rolling Descriptor reads every return that is not missing as a finite number, and it is $(Xf[k]) at observation $(k[1]) for asset $(k[2]). An infinite return is a data error, so clean the input rather than pass it through."))
    return Xf, pnl
end
"""
    rolling_window_max(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                       i::Integer, rows) -> Real

Take the largest return of one asset over one window of observations.

This is the scan of one window for [`RollingMax`](@ref). The scan stops at the first inactive observation, because the value of the window is then `NaN` whatever the rest of the window holds.

# Mathematical definition

```math
\\begin{align}
m &= \\begin{cases} \\max \\{ x_{k,\\,i} : k \\in K \\text{, } x_{k,\\,i} \\text{ observed} \\} & \\text{if } a_{ki} = 1 \\text{ for every } k \\in K \\text{, and one } x_{k,\\,i} \\text{ is observed} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``m``: Value of the window.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:a_ti_pnl])
  - ``K``: `rows`, the observations of the window.

# Arguments

  - `X`: The returns, `observations × assets`, in a type that holds `NaN`.
  - `amsk`: The active mask of the Asset Panel, `observations × assets`.
  - `i`: The column of the asset.
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

This is the archetype of every rolling log-return Descriptor. Medium-term momentum skips the most recent month, and short-term reversal negates the sum of the last month. The window holds `window` observations and ends `skip` observations before the current one, so momentum and reversal differ only in their window, their skip and their sign.

The Descriptor is `NaN` unless every observation of the window is active. This one rule covers three cases: the warm-up at the start of the sample, an asset that lists late, and a gap in the middle of a listing. A window with a skip excludes the current observation, and the Descriptor is also `NaN` where the current observation is inactive. An active observation whose return is missing adds zero to the sum, because a holiday is not a loss.

The output is a log return by default. A log cumulative return is more symmetric than a simple one, and the cross-sectional standardisation that reads it works better on a symmetric value. The logarithm is increasing, so the two orders of the assets agree. Set `exponentiate` to get the simple return.

# Mathematical definition

```math
\\begin{align}
y_{t,i} &= \\begin{cases} \\log(1 + x_{t,\\,i}) & \\text{if } x_{t,\\,i} \\text{ is observed} \\\\ 0 & \\text{otherwise} \\end{cases}\\,,\\\\
S_{t,i} &= \\sum_{k = t - s - w + 1}^{t - s} y_{k,i}\\,,\\\\
d_{t,i} &= \\begin{cases} \\sigma S_{t,i} & \\text{if } t \\ge s + w \\text{, } a_{ti} = 1 \\text{, } a_{ki} = 1 \\text{ for every } k \\text{ of the window, and not } \\texttt{exponentiate} \\\\ \\exp(\\sigma S_{t,i}) - 1 & \\text{if the same conditions hold, and } \\texttt{exponentiate} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Every return that is observed is finite and greater than ``-1``, so every ``y_{t,i}`` is finite.

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - $(math_dict[:x_ti_ret])
  - ``y_{t,i}``: Log return of asset ``i`` at observation ``t``, zero where the return is missing.
  - ``S_{t,i}``: Sum of the log returns of the window of asset ``i`` at observation ``t``.
  - $(math_dict[:a_ti_pnl])
  - $(math_dict[:w_roll])
  - ``s``: `skip`, the number of the most recent observations that the window excludes.
  - ``\\sigma``: `sign`, the multiplier of the window sum.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RollingLogReturn(; window::Integer, skip::Integer = 0, sign::Real = 1,
                     exponentiate::Bool = false) -> RollingLogReturn

Keywords correspond to the struct's fields. `window` takes no default, because it depends on the data frequency. For example, `252` is one year of daily observations. The named Descriptors [`RollingMomentum`](@ref) and [`Reversal`](@ref) give a default to all four.

## Validation

  - `window > 0`.
  - `skip >= 0`.
  - `sign == 1 || sign == -1`, through [`assert_rolling_sign`](@ref).

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
    $(field_dict[:window_roll])
    """
    window
    """
    $(field_dict[:skip_roll])
    """
    skip
    """
    $(field_dict[:sign_roll])
    """
    sign
    """
    $(field_dict[:exponentiate_roll])
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

This is the lottery demand Descriptor. An asset whose recent returns hold one very large positive return attracts speculative demand, and the cross-section of this Descriptor prices that demand.

The Descriptor is `NaN` unless every observation of the window is active. This rule covers the warm-up at the start of the sample, an asset that lists late, and a gap in the middle of a listing. The window ends at the current observation, so an inactive current observation makes the Descriptor `NaN` too. The maximum ignores a missing return inside an active window, and does not count it as zero. A window whose returns are all missing is `NaN`, because a maximum of no value is not a number.

# Mathematical definition

```math
\\begin{align}
d_{t,i} &= \\begin{cases} \\max \\{ x_{k,\\,i} : k \\in [t - w + 1,\\, t] \\text{, } x_{k,\\,i} \\text{ observed} \\} & \\text{if } t \\ge w \\text{, } a_{ki} = 1 \\text{ for every } k \\text{ of the window, and one } x_{k,\\,i} \\text{ is observed} \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Every return that is observed is finite.

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:a_ti_pnl])
  - $(math_dict[:w_roll])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RollingMax(; window::Integer) -> RollingMax

Keywords correspond to the struct's fields. `window` takes no default, because it depends on the data frequency. For example, `21` is one month of daily observations, and [`MaxReturn`](@ref) gives that default.

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
    $(field_dict[:window_roll])
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

Both methods read the returns of `rd.X` and the active mask of the Asset Panel, and no Panel Field. Both write `NaN` at every observation whose window is not wholly active, and in every inactive cell.

# Algorithm

 1. Read the returns and the Asset Panel through [`descriptor_returns`](@ref), which floats integer returns and refuses an infinite return.
 2. For [`RollingLogReturn`](@ref), refuse a return at or below `-1` through [`assert_log_returns`](@ref). Then take the cumulative sums of `log1p` of the returns and of the active mask along the observations. A missing return adds zero. The window of observation `t` runs from `t - skip - window + 1` to `t - skip`, and its sum is one difference of the cumulative sums. Where the active count of the window equals `window`, write that sum multiplied by `sign`, or `expm1` of that product when `exponentiate` is set.
 3. For [`RollingMax`](@ref), scan the window of each observation through [`rolling_window_max`](@ref). It gives the largest return that is not missing where every observation of the window is active and one return exists.
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
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 1.0; 1.0 1.0; 1.0 1.0; 1.0 1.0])];
                         amsk = trues(4, 2), emsk = trues(4, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 -0.05; 0.2 0.1; -0.1 0.05; 0.05 NaN],
                          pnl = pnl);

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

This is the classic twelve-minus-one momentum. The skip separates the medium-term momentum that the window measures from the short-term reversal of the most recent month. [`Reversal`](@ref) measures that reversal on its own.

# Arguments

  - $(arg_dict[:window_roll]) `252` is one year of daily observations.
  - $(arg_dict[:skip_roll]) `21` is one month of daily observations.
  - $(arg_dict[:sign_roll])
  - $(arg_dict[:exponentiate_roll])

# Validation

  - The validation of [`RollingLogReturn`](@ref).

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

This is the short-term reversal. A high value shows that the asset lost value recently. Temporary price pressure, the provision of liquidity and the microstructure of the market tend to reverse such a loss. It is the counterpart of [`RollingMomentum`](@ref), whose skip excludes the window of this Descriptor.

# Arguments

  - $(arg_dict[:window_roll]) `21` is one month of daily observations, `5` is one week, and `1` is one day.
  - $(arg_dict[:skip_roll])
  - $(arg_dict[:sign_roll])
  - $(arg_dict[:exponentiate_roll])

# Validation

  - The validation of [`RollingLogReturn`](@ref).

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

This is the lottery demand Descriptor at the horizon of its source. An asset whose last month holds one very large positive return earns a lower return after that month. The cross-section prices this as the cost of a payoff that resembles a lottery ticket.

# Arguments

  - $(arg_dict[:window_roll]) `21` is one month of daily observations, and `5` is one week.

# Validation

  - The validation of [`RollingMax`](@ref).

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
