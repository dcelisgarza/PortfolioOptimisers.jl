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

The Descriptor is a difference of cumulative sums from the first observation, so it folds one observation at a time. [`partial_fit!`](@ref) carries the last `window + skip + 1` rows of the sums in a [`RollingLogReturnState`](@ref), and the Descriptor of each new observation equals the one of the batch call to the last bit. The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads it that way, so it keeps no panel row for it.

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
                     exponentiate::Bool = false,
                     cache::Option{<:AbstractPartialFitState} = nothing) -> RollingLogReturn

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
  - [`RollingLogReturnState`](@ref)
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
    """
    $(field_dict[:roll_cache])
    """
    cache
    function RollingLogReturn(window::Integer, skip::Integer, sign::Real,
                              exponentiate::Bool, cache::Option{<:AbstractPartialFitState})
        assert_gt0(window, :window)
        assert_nonneg(skip, :skip)
        assert_rolling_sign(sign)
        return new{typeof(window), typeof(skip), typeof(sign), typeof(exponentiate),
                   typeof(cache)}(window, skip, sign, exponentiate, cache)
    end
end
function RollingLogReturn(; window::Integer, skip::Integer = 0, sign::Real = 1,
                          exponentiate::Bool = false,
                          cache::Option{<:AbstractPartialFitState} = nothing)::RollingLogReturn
    return RollingLogReturn(window, skip, sign, exponentiate, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`RollingLogReturn`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one, such as a [`CompositeExposure`](@ref). Set `set_show_nothing_fields!(:RollingLogReturn, true)` to render it.

# Arguments

  - `de`: The estimator.

# Returns

  - `fields::Tuple`: The field names to render, `(:window, :skip, :sign, :exponentiate)`.

# Related

  - [`RollingLogReturn`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(::RollingLogReturn)
    return (:window, :skip, :sign, :exponentiate)
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
function lookback(de::RollingLogReturn)::Integer
    return de.skip + de.window
end
function lookback(de::RollingMax)::Integer
    return de.window
end
"""
$(DocStringExtensions.TYPEDEF)

Carried state of a [`RollingLogReturn`](@ref): the last `window + skip + 1` rows of the cumulative sums that its batch call takes.

The batch call [`descriptor`](@ref) takes the cumulative sums of `log1p` of the returns and of the active mask from the first observation. The Descriptor of an observation is one difference of two rows of each. So the last `window + skip + 1` rows give the Descriptor of the next observation, and the state holds no other value. The state adds each new row to the last carried row, as the batch call does, so the Descriptor of a new observation equals the one of the batch call over every folded observation to the last bit. A running window sum that adds the new row and subtracts the old one drifts by round-off, so the state keeps no such sum.

Each row is a vector of its own, and no verb changes a row after the state carries it. A step copies the two buffers, which copies the references to the rows and no row, so the state before the step stays as it was.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RollingLogReturnState(; cs::DataStructures.CircularBuffer{<:AbstractVector{<:Real}},
                          ac::DataStructures.CircularBuffer{<:AbstractVector{<:Integer}}) -> RollingLogReturnState

Keywords correspond to the struct's fields. [`descriptor_step`](@ref) seeds a state whose one row is zero, the row before the first observation of the batch call.

## Validation

  - `cs` and `ac` have the same capacity, and hold the same number of rows. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`RollingLogReturn`](@ref)
  - [`descriptor_step`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct RollingLogReturnState <: AbstractPartialFitState
    """
    The cumulative sums of `log1p` of the returns of each asset, one row per observation, oldest first. A missing return adds zero.
    """
    cs
    """
    The cumulative counts of the active observations of each asset, one row per observation, oldest first.
    """
    ac
    function RollingLogReturnState(cs::DataStructures.CircularBuffer{<:AbstractVector{<:Real}},
                                   ac::DataStructures.CircularBuffer{<:AbstractVector{<:Integer}})
        @argcheck(DataStructures.capacity(cs) == DataStructures.capacity(ac) &&
                  length(cs) == length(ac),
                  DimensionMismatch("the cumulative sums and the active counts of a RollingLogReturnState hold the same rows, got $(length(cs)) of $(DataStructures.capacity(cs)) and $(length(ac)) of $(DataStructures.capacity(ac))"))
        return new{typeof(cs), typeof(ac)}(cs, ac)
    end
end
function RollingLogReturnState(;
                               cs::DataStructures.CircularBuffer{<:AbstractVector{<:Real}},
                               ac::DataStructures.CircularBuffer{<:AbstractVector{<:Integer}})::RollingLogReturnState
    return RollingLogReturnState(cs, ac)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`RollingLogReturnState`](@ref) fitted on disjoint blocks.

The cumulative sums of the second block start from the last row of the first block, so the state of the second block alone does not give its Descriptors. The two blocks fold in sequence.

# Arguments

  - `a`: The state of the first block.
  - `b`: The state of the second block.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`RollingLogReturnState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(::RollingLogReturnState, ::RollingLogReturnState)
    return throw(ArgumentError("a RollingLogReturnState cannot merge two states fitted on disjoint blocks: the cumulative sums of the second block start from the last row of the first. Fold the second block into the state of the first with partial_fit!."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`RollingLogReturnState`](@ref), so that the copy shares no row with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::RollingLogReturnState`: A new state, equal to `x`.

# Related

  - [`RollingLogReturnState`](@ref)
"""
function Base.copy(x::RollingLogReturnState)
    return RollingLogReturnState(rolling_state_buffer(copy, x.cs),
                                 rolling_state_buffer(copy, x.ac))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds a buffer of the capacity of `cb` that holds `f` of each of its rows, in order.

# Arguments

  - `f`: The function of a row, `identity` to share the rows or `copy` to copy them.
  - `cb`: The buffer.

# Returns

  - `cb::DataStructures.CircularBuffer`: The new buffer.

# Related

  - [`RollingLogReturnState`](@ref)
  - [`LagDescriptorState`](@ref)
"""
function rolling_state_buffer(f, cb::DataStructures.CircularBuffer)
    out = DataStructures.CircularBuffer{eltype(cb)}(DataStructures.capacity(cb))
    for r in cb
        push!(out, f(r))
    end
    return out
end
"""
    rolling_state_seed(de::RollingLogReturn, X::MatNum)

Returns the state that a step of a [`RollingLogReturn`](@ref) folds its rows into: a seeded state when the estimator carries none, or a copy of its buffers that shares their rows.

# Arguments

  - `de`: The estimator.
  - `X`: The returns of the step, `observations × assets`.

# Validation

  - A carried state holds the assets of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::RollingLogReturnState`: A state that no other estimator holds.

# Related

  - [`descriptor_step`](@ref)
"""
function rolling_state_seed(de::RollingLogReturn{<:Any, <:Any, <:Any, <:Any, Nothing},
                            X::MatNum)
    k = de.window + de.skip + 1
    N = size(X, 2)
    cs = DataStructures.CircularBuffer{Vector{eltype(X)}}(k)
    ac = DataStructures.CircularBuffer{Vector{Int}}(k)
    push!(cs, zeros(eltype(X), N))
    push!(ac, zeros(Int, N))
    return RollingLogReturnState(cs, ac)
end
function rolling_state_seed(de::RollingLogReturn{<:Any, <:Any, <:Any, <:Any,
                                                 <:RollingLogReturnState}, X::MatNum)
    st = de.cache
    @argcheck(length(st.cs[end]) == size(X, 2),
              DimensionMismatch("the state of this RollingLogReturn carries $(length(st.cs[end])) assets, and the step brings $(size(X, 2))"))
    return RollingLogReturnState(rolling_state_buffer(identity, st.cs),
                                 rolling_state_buffer(identity, st.ac))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the rows of a step into a [`RollingLogReturnState`](@ref) in place, and returns the Descriptor of each row.

Each row adds its log returns and its active mask to the last carried row, with the arithmetic of the batch call [`descriptor`](@ref). Once the state holds `window + skip + 1` rows, the Descriptor of the new row is the difference of the row `skip` rows before the last and the oldest row, where the active count of the window equals `window`.

# Arguments

  - `st`: The state, changed in place.
  - `de`: The estimator, which fixes the window, the skip and the sign.
  - `X`: The returns of the step, `observations × assets`.
  - `amsk`: The active mask of the step.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor of each row of the step, `NaN` where the window is not complete.

# Related

  - [`descriptor_step`](@ref)
  - [`RollingLogReturnState`](@ref)
"""
function rolling_state_fold!(st::RollingLogReturnState, de::RollingLogReturn, X::MatNum,
                             amsk::AbstractMatrix{Bool})::Matrix{<:Real}
    D = fill(eltype(X)(NaN), size(X))
    (; cs, ac) = st
    for t in axes(X, 1)
        rolling_state_push!(st, view(X, t, :), view(amsk, t, :))
        if DataStructures.isfull(cs)
            rolling_state_value!(view(D, t, :), st, de)
        end
    end
    return D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Writes the Descriptor of the newest row of a full [`RollingLogReturnState`](@ref), where the active count of its window equals `window`.

# Arguments

  - `d`: The Descriptor of the row, changed in place.
  - `st`: The state, which holds `window + skip + 1` rows.
  - `de`: The estimator, which fixes the window, the skip and the sign.

# Returns

  - `nothing`.

# Related

  - [`rolling_state_fold!`](@ref)
"""
function rolling_state_value!(d::AbstractVector{<:Real}, st::RollingLogReturnState,
                              de::RollingLogReturn)::Nothing
    (; cs, ac) = st
    cs0, ce, as, ae = cs[1], cs[end - de.skip], ac[1], ac[end - de.skip]
    for i in eachindex(d)
        if ae[i] - as[i] == de.window
            v = de.sign * (ce[i] - cs0[i])
            d[i] = de.exponentiate ? expm1(v) : v
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pushes the cumulative sums and the active counts of one observation onto a [`RollingLogReturnState`](@ref). Each new row is the last row plus the log returns and the active mask of the observation, with the arithmetic of the batch call [`descriptor`](@ref). A missing return adds zero.

# Arguments

  - `st`: The state, changed in place.
  - `x`: The returns of the observation.
  - `a`: The active mask of the observation.

# Returns

  - `nothing`.

# Related

  - [`rolling_state_fold!`](@ref)
"""
function rolling_state_push!(st::RollingLogReturnState, x::AbstractVector{<:Real},
                             a::AbstractVector{Bool})::Nothing
    c0, a0 = st.cs[end], st.ac[end]
    c, n = similar(c0), similar(a0)
    for i in eachindex(c0)
        xi = x[i]
        c[i] = c0[i] + (isnan(xi) ? zero(eltype(c0)) : log1p(xi))
        n[i] = a0[i] + a[i]
    end
    push!(st.cs, c)
    push!(st.ac, n)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of a [`RollingLogReturn`](@ref), and returns the Descriptor of each one.

The Descriptor of an observation equals the one of the batch call [`descriptor`](@ref) over every observation that the state folded and the observations before it, to the last bit, because the state takes the cumulative sums with the same arithmetic from the same first observation. The step copies the buffers of the state and no row, so the estimator it gets keeps its state.

# Algorithm

 1. Take the returns and the Asset Panel with [`descriptor_returns`](@ref), and refuse a return at or below `-1` through [`assert_log_returns`](@ref).
 2. Take the state with [`rolling_state_seed`](@ref), and fold the rows into it with [`rolling_state_fold!`](@ref).
 3. Write `NaN` where an observation is inactive, with [`descriptor_active_fill!`](@ref).

# Arguments

  - `de`: The estimator, with or without a state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`descriptor_returns`](@ref), [`assert_log_returns`](@ref) and [`rolling_state_seed`](@ref).

# Returns

  - `step::NamedTuple`: `de`, the estimator with the state after the observations in `cache`, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`RollingLogReturn`](@ref)
  - [`RollingLogReturnState`](@ref)
  - [`partial_fit!`](@ref)
  - [`descriptor_carry`](@ref)
"""
function descriptor_step(de::RollingLogReturn, rd::ReturnsResult)
    X, pnl = descriptor_returns(rd)
    assert_log_returns(X)
    st = rolling_state_seed(de, X)
    D = rolling_state_fold!(st, de, X, pnl.amsk)
    descriptor_active_fill!(D, pnl)
    return (; de = RollingLogReturn(de.window, de.skip, de.sign, de.exponentiate, st),
            D = D)
end
"""
    descriptor_step(::AbstractDescriptorEstimator, ::ReturnsResult)

Answers `nothing`: a Descriptor carries no state unless it implements this verb.

`descriptor_step` is the verb of a Descriptor that folds new observations from a carried state, and it is `public`. A subtype of [`AbstractDescriptorEstimator`](@ref) that folds implements it, with the contract of the methods of [`RollingLogReturn`](@ref), of the exponentially weighted mean Descriptors and of the lag Descriptors: it takes the estimator with its state in a field, and the returns data of the new observations alone, and it answers `(; de, D)`, the estimator with the state after them and the Descriptor of each one, equal to the batch call [`descriptor`](@ref) over every observation from the first one. It also makes [`carry_lookback`](@ref) answer one. The carry fold of a [`CrossSectionalFactorPrior`](@ref) then reads the Descriptor off the state with [`descriptor_carry`](@ref), and keeps no panel row for it.

# Arguments

  - `de`: The Descriptor.
  - $(arg_dict[:rd])

# Returns

  - `nothing`.

# Related

  - [`descriptor_carry`](@ref)
  - [`carry_lookback`](@ref)
"""
function descriptor_step(::AbstractDescriptorEstimator, ::ReturnsResult)::Nothing
    return nothing
end
"""
    partial_fit!(de::RollingLogReturn{<:Any, <:Any, <:Any, <:Any,
                                      <:Option{<:RollingLogReturnState}},
                 rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of a [`RollingLogReturn`](@ref), and returns the estimator with the state after them in `cache`. [`descriptor_step`](@ref) states the fold, and also returns the Descriptor of each observation.

# Arguments

  - `de`: The estimator, with no state or with its state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`descriptor_step`](@ref).

# Returns

  - `de::RollingLogReturn`: The estimator, with its `cache` field set to the state after the observations.

# Related

  - [`descriptor_step`](@ref)
  - [`RollingLogReturnState`](@ref)
"""
function partial_fit!(de::RollingLogReturn{<:Any, <:Any, <:Any, <:Any,
                                           <:Option{<:RollingLogReturnState}},
                      rd::ReturnsResult)
    return descriptor_step(de, rd).de
end
"""
$(DocStringExtensions.TYPEDEF)

The Descriptor of the new observations of a step of the carry fold, read off a carried state.

The carry fold of a [`CrossSectionalFactorPrior`](@ref) puts it in place of a Descriptor that carries a state, as [`descriptor_carry`](@ref) does, so the code of the batch fit computes the Factor Exposures of the new observations. It reads no panel row, so its [`lookback`](@ref) is one.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`descriptor_carry`](@ref)
  - [`descriptor_step`](@ref)
"""
@concrete struct CarriedDescriptor <: AbstractDescriptorEstimator
    """
    The Descriptor of the new observations, `observations × assets`.
    """
    D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the Descriptor of a [`CarriedDescriptor`](@ref) on the rows of a [`ReturnsResult`](@ref). The carried rows are the last rows of `rd`, and every row before them is `NaN`.

# Arguments

  - `de`: The carried Descriptor.
  - $(arg_dict[:rd])

# Validation

  - `rd` holds the assets of `de.D` and at least its rows. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Related

  - [`CarriedDescriptor`](@ref)
"""
function descriptor(de::CarriedDescriptor, rd::ReturnsResult)::Matrix{<:Real}
    T, N = size(rd.X)
    m = size(de.D, 1)
    @argcheck(T >= m && N == size(de.D, 2),
              DimensionMismatch("a carried Descriptor of $(size(de.D)) reads returns data of $((T, N))"))
    if T == m
        return de.D
    end
    D = fill(eltype(de.D)(NaN), T, N)
    D[(T - m + 1):T, :] = de.D
    return D
end
function lookback(::CarriedDescriptor)::Integer
    return 1
end
"""
    descriptor_carry(xe, rd::ReturnsResult, m)
    descriptor_carry(de::AbstractDescriptorEstimator, rd::ReturnsResult, m::Integer)
    descriptor_carry(xe::CompositeExposure, rd::ReturnsResult, m::Integer)
    descriptor_carry(ds::DescriptorScores, rd::ReturnsResult, m::Integer)

Folds the last `m` observations of a [`ReturnsResult`](@ref) into every Descriptor of an Exposure Estimator that carries a state, on the carry fold of a [`CrossSectionalFactorPrior`](@ref).

The generic method returns the estimator as it is twice: it carries no state. A Descriptor folds the observations with [`descriptor_step`](@ref) when that verb answers a step, as it does for a [`RollingLogReturn`](@ref), an [`EWMean`](@ref), an [`EWVolumeRatio`](@ref), a [`DaysToCover`](@ref) and the lag Descriptors, and [`descriptor_carried`](@ref) reads the answer. Every other Descriptor answers `nothing`, and stays as it is. The [`CompositeExposure`](@ref) method folds each of its Descriptors. The [`DescriptorScores`](@ref) method folds each Descriptor of the scores of a Return Forecast the same way, so the code of the batch fit scores the new observations.

# Arguments

  - `xe`: The Exposure Estimator, the Descriptor or the Descriptor Scores, as the carry folded it so far.
  - $(arg_dict[:rd]) It holds the panel rows that the carry carries, followed by the new observations.
  - `m`: The number of new observations.

# Returns

  - `carry::NamedTuple`: `xf`, the estimator with the state after the new observations, and `xv`, the estimator whose stateful Descriptors are [`CarriedDescriptor`](@ref) of the new observations, which the code of the batch fit computes the Factor Exposures with.

# Related

  - [`carry_lookback`](@ref)
  - [`CarriedDescriptor`](@ref)
  - [`descriptor_step`](@ref)
"""
function descriptor_carry(xe, ::ReturnsResult, ::Any)
    return (; xf = xe, xv = xe)
end
function descriptor_carry(de::AbstractDescriptorEstimator, rd::ReturnsResult, m::Integer)
    T = size(rd.X, 1)
    return descriptor_carried(de, descriptor_step(de, port_opt_view(rd, (T - m + 1):T, :)))
end
"""
    descriptor_carried(de::AbstractDescriptorEstimator, ::Nothing)
    descriptor_carried(::AbstractDescriptorEstimator, step::NamedTuple)

Returns the carry of a Descriptor from the answer of [`descriptor_step`](@ref), as [`descriptor_carry`](@ref) gives it: the Descriptor as it is twice for `nothing`, because it carries no state, or the Descriptor with its new state and the [`CarriedDescriptor`](@ref) of the new observations for a step.

# Arguments

  - `de`: The Descriptor, as the carry folded it so far.
  - `step`: The answer of [`descriptor_step`](@ref), `nothing` or a `NamedTuple` with the fields `de` and `D`.

# Returns

  - `carry::NamedTuple`: `xf` and `xv`, as [`descriptor_carry`](@ref) states them.

# Related

  - [`descriptor_carry`](@ref)
  - [`descriptor_step`](@ref)
"""
function descriptor_carried(de::AbstractDescriptorEstimator, ::Nothing)
    return (; xf = de, xv = de)
end
function descriptor_carried(::AbstractDescriptorEstimator, step::NamedTuple)
    return (; xf = step.de, xv = CarriedDescriptor(step.D))
end
"""
    carry_lookback(x)
    carry_lookback(::Union{RollingLogReturn, EWMean, EWVolumeRatio, DaysToCover, GrowthRate,
                           ChangeToScale, ChangeInIntensity})
    carry_lookback(xe::CompositeExposure)
    carry_lookback(pe::CrossSectionalFactorPrior)
    carry_lookback(rfe::Union{FixedWeightedReturnForecast, ExpWeightedReturnForecast,
                              TargetReturnForecast})
    carry_lookback(ests::AbstractVector)

Returns the number of panel rows that an estimator reads to give the Factor Exposure of one new observation on the carry fold of a [`CrossSectionalFactorPrior`](@ref), or `nothing` for every row.

A Descriptor that carries a state, as [`descriptor_carry`](@ref) folds it, reads the new observation alone, so a [`RollingLogReturn`](@ref), an [`EWMean`](@ref), an [`EWVolumeRatio`](@ref), a [`DaysToCover`](@ref) and the lag Descriptors [`GrowthRate`](@ref), [`ChangeToScale`](@ref) and [`ChangeInIntensity`](@ref) answer one. A [`CompositeExposure`](@ref) answers it over its Descriptors, and a Return Forecast with Descriptor Scores over the Descriptors of its scores. Every other estimator answers its [`lookback`](@ref), and a vector answers the largest one of its members, or `nothing` when one member answers `nothing`. A [`CrossSectionalFactorPrior`](@ref) answers the look-back of its factors on the carry fold, as [`cross_sectional_lookback`](@ref) counts it, and [`cross_sectional_carry_rows`](@ref) reads it.

# Arguments

  - `x`: The estimator, or a vector of estimators.

# Returns

  - `n::Option{<:Integer}`: The number of rows, or `nothing`.

# Related

  - [`lookback`](@ref)
  - [`descriptor_carry`](@ref)
"""
function carry_lookback(x)::Option{<:Integer}
    return lookback(x)
end
function carry_lookback(::Union{RollingLogReturn, EWMean, EWVolumeRatio, DaysToCover,
                                GrowthRate, ChangeToScale, ChangeInIntensity})::Integer
    return 1
end
function carry_lookback(ests::AbstractVector)::Option{<:Integer}
    return lookback_max(carry_lookback, ests)
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
public descriptor_step, carry_lookback
