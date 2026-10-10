"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all Descriptor Estimator types.

A Descriptor Estimator produces one Descriptor, a per-asset value at every observation that the estimator computes from one or more Panel Fields of an Asset Panel. A point-in-time ratio of two fundamentals, the logarithm of a market capitalisation and the growth of a field over a lag are each one Descriptor. The estimator is configuration, so it names the Panel Fields it reads and holds no data.

All concrete types producing a Descriptor should be subtypes of `AbstractDescriptorEstimator`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractDescriptorEstimator` and implement the following methods:

## `descriptor`

  - [`descriptor(de::AbstractDescriptorEstimator, rd::ReturnsResult)`](@ref): Computes the Descriptor of a [`ReturnsResult`](@ref).

### Arguments

  - `de`: The concrete subtype instance.
  - `rd`: The returns result that carries the Asset Panel.

### Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`, `NaN` wherever the active mask is `false`.

## `lookback` (optional)

  - `lookback(de::MyDescriptor) -> Option{<:Integer}`: The number of trailing observations that the value of the Descriptor at one observation reads, or `nothing` when the number is unbounded.

The fallback returns `nothing`, which is exact for every Descriptor. Its cost is on the carry fold of a [`CrossSectionalFactorPrior`](@ref). With no finite look-back, [`FoldOrRefit`](@ref) keeps every panel row, and computes the Factor Exposures over all of them at each step. [`FoldOnly`](@ref) refuses the prior.

# Related

  - [`AbstractEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`lookback`](@ref)
  - [`PanelFieldRatio`](@ref)
  - [`PanelFieldLog`](@ref)
  - [`Passthrough`](@ref)
  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`EWMean`](@ref)
  - [`EWVolumeRatio`](@ref)
  - [`DaysToCover`](@ref)
  - [`EWVolatility`](@ref)
  - [`RollingLogReturn`](@ref)
  - [`RollingMax`](@ref)
  - [`AssetPanel`](@ref)
"""
abstract type AbstractDescriptorEstimator <: AbstractEstimator end
"""
    descriptor(de::AbstractDescriptorEstimator, rd::ReturnsResult) -> Matrix{<:Real}

Compute the Descriptor of a [`ReturnsResult`](@ref).

Every Descriptor Estimator implements this function. It reads the Panel Fields that the estimator names from `rd.pnl`. Returns are not a Panel Field, so a member that reads them reads `rd.X`. Every member follows two conventions. The value at an observation uses information up to and including that observation, and every cell where the active mask of the Asset Panel is `false` is `NaN`.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor_field_values`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor end
"""
    descriptor_field_values(rd::ReturnsResult, name::AbstractString) -> Matrix{<:Real}
    descriptor_field_values(rd::ReturnsResult,
                            terms::AbstractVector{<:Pair{<:AbstractString, <:Real}}) -> Matrix{<:Real}

Read one numeric Panel Field, or a linear combination of numeric Panel Fields, out of a [`ReturnsResult`](@ref), for a Descriptor.

Every Descriptor Estimator reads its Panel Fields through this function. A blank cell never reaches a `ReturnsResult`, because [`asset_panel`](@ref) resolves each one to a fill value and records the resolution in the observed-mask column of the field. A cell outside the universe holds a finite value too. This function writes `NaN` into both kinds of cell, through [`panel_field_values`](@ref) with `inactive = NaN`, so a Descriptor cannot mistake a fill value or the value of an inactive cell for data. A lag or a recursion that reaches back to a cell before a listing, or into a stretch where the asset is out of the universe, reads `NaN` there.

# Algorithm

 1. Check that the Panel Field is numeric and time-varying.
 2. Read it through [`panel_field_values`](@ref) with `inactive = NaN` and `unobserved = NaN`. An integer field reads in `Float64`, and a `Float32` field stays `Float32`.
 3. For a vector of `name => coefficient` pairs, read each named field the same way and multiply it by its coefficient. Return the sum of those terms in the type that all of them promote to, so the order of the terms does not change the result. A `NaN` in any term is a `NaN` in the sum.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `name`: The Panel Field's name.
  - `terms`: The Panel Fields to combine, each paired with its coefficient. `[\"a\" => 1, \"b\" => -1]` reads `a - b`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - `name` names a Panel Field, and its kind is [`NumericPanelField`](@ref). Raises a `KeyError` or an `ArgumentError`.
  - `!isempty(terms)`. Raises an [`IsEmptyError`](@ref).

# Returns

  - `V::Matrix{<:Real}`: The values, `observations × assets`, `NaN` where the Panel Field was not observed and where the asset is outside the universe.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"mcap\", vals = [1.0 2.0; NaN 4.0],
                                            alg = ForwardPanelFill(; val = 0.0)),
                          NumericPanelInput(; name = \"debt\", vals = [0.5 1.0; 1.5 2.0])];
                         amsk = [true false; true true], emsk = [true false; true true]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> PortfolioOptimisers.descriptor_field_values(rd, \"mcap\")
2×2 Matrix{Float64}:
   1.0  NaN
 NaN      4.0

julia> PortfolioOptimisers.descriptor_field_values(rd, [\"mcap\" => 1, \"debt\" => 1])
2×2 Matrix{Float64}:
   1.5  NaN
 NaN      6.0
```

# Related

  - [`descriptor`](@ref)
  - [`panel_field_values`](@ref)
  - [`panel_field`](@ref)
  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`NumericPanelField`](@ref)
"""
function descriptor_field_values(rd::ReturnsResult, name::AbstractString)::Matrix{<:Real}
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Descriptor reads its Panel Fields off an Asset Panel, and rd.pnl is nothing. Build the ReturnsResult with the `pnl` that asset_panel returns."))
    f = panel_field(pnl, name)
    @argcheck(isa(f, NumericPanelField),
              ArgumentError("a Descriptor reads one number per observation and asset, so the Panel Field \"$name\" must be a NumericPanelField, got a $(nameof(typeof(f)))"))
    @argcheck(ndims(f.vals) == 2,
              DimensionMismatch("a Descriptor reads one number per observation and asset, so the Panel Field \"$name\" must be time-varying; this Asset Panel is static"))
    return panel_field_values(pnl, name; inactive = NaN, unobserved = NaN)
end
function descriptor_field_values(rd::ReturnsResult,
                                 terms::AbstractVector{<:Pair{<:AbstractString, <:Real}})::Matrix{<:Real}
    @argcheck(!isempty(terms),
              IsEmptyError("a Panel Field combination needs at least one `name => coefficient` term"))
    Vs = [descriptor_field_values(rd, name) .* c for (name, c) in terms]
    V = similar(Vs[1], mapreduce(eltype, promote_type, Vs))
    V .= Vs[1]
    for k in 2:length(Vs)
        V .+= Vs[k]
    end
    return V
end
"""
    descriptor_asset_panel(rd::ReturnsResult) -> AssetPanel

Read the Asset Panel a Descriptor needs out of a [`ReturnsResult`](@ref).

[`descriptor_field_values`](@ref) refuses a `ReturnsResult` that holds no Asset Panel. A Descriptor over the returns reads no Panel Field, so it calls this function to get the same refusal and the active mask that [`descriptor_active_fill!`](@ref) reads.

# Arguments

  - $(arg_dict[:rd])

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel the `ReturnsResult` holds.

# Related

  - [`descriptor`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`descriptor_field_values`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_asset_panel(rd::ReturnsResult)::AssetPanel
    pnl = rd.pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Descriptor is `NaN` wherever the active mask of an Asset Panel is `false`, and rd.pnl is nothing. Build the ReturnsResult with the `pnl` that asset_panel returns."))
    return pnl
end
"""
    assert_log_returns(X::AbstractMatrix{<:Real}) -> nothing

Check that every return that is not missing is greater than `-1`.

A Descriptor that compounds returns takes the logarithm of one plus each return. A return of `-1` is a total loss, and the logarithm is undefined below it, so the check refuses the whole matrix rather than write an infinity into one cell of the Descriptor. A missing return is a `NaN`, and it passes the check. Every Descriptor that reads `log1p(rd.X)` runs it, exponentially weighted and rolling alike.

# Arguments

  - `X`: The returns, `observations × assets`.

# Validation

  - Every entry of `X` that is not `NaN` is greater than `-1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`descriptor`](@ref)
  - [`EWMean`](@ref)
  - [`RollingLogReturn`](@ref)
"""
function assert_log_returns(X::AbstractMatrix{<:Real})::Nothing
    k = findfirst(x -> !isnan(x) && x <= -one(x), X)
    @argcheck(isnothing(k),
              DomainError(X[k],
                          "a Descriptor over log returns takes the logarithm of one plus each return, so every return that is not missing must be greater than -1, and it is $(X[k]) at observation $(k[1]) for asset $(k[2]). A return at or below -1 is a data error, so clean the input rather than pass it through."))
    return nothing
end
"""
    nan_fill_value(A::AbstractArray{<:Number}) -> Number

Return `NaN` in the element type of an array that a fill writes into in place, and refuse an element type that cannot hold it.

A fill changes an array that its caller owns, so it cannot widen the element type the way a function that allocates its result does with `float_if_integer`. An `Integer`, a `Rational` or a `Complex{Int}` holds no `NaN`, and the conversion would raise an `InexactError` at the first inactive cell. This function converts once, before the fill writes anything. Every element type that holds `NaN` passes, a number type from another package included. Every other type raises an `ArgumentError` whose message names the repair.

# Arguments

  - `A`: The array the fill writes into.

# Validation

  - `eltype(A)` holds `NaN`. Raises an `ArgumentError`.

# Returns

  - `nan::Number`: `NaN` converted to `eltype(A)`.

# Related

  - [`descriptor_active_fill!`](@ref)
  - [`exposure_active_fill!`](@ref)
  - [`one_hot_level_fill!`](@ref)
  - [`one_hot_observed_fill!`](@ref)
"""
function nan_fill_value(A::AbstractArray{<:Number})::Number
    T = eltype(A)
    try
        return convert(T, NaN)
    catch err
        if !(err isa InexactError)
            rethrow()
        end
        throw(ArgumentError("a fill writes NaN into the cells that hold no value, in place, so the element type of the array must hold NaN, and $T does not. Convert the array to a floating point type before the call, for example with float_if_integer."))
    end
end
"""
    descriptor_active_fill!(D::AbstractMatrix{<:Number}, pnl::AssetPanel) -> nothing

Write `NaN` into every cell of a Descriptor where the active mask of the Asset Panel is `false`, in place.

Every Descriptor Estimator ends with this call, so the library states the convention that an inactive cell is `NaN` in one place. An asset that is not listed at an observation has no Descriptor there, whatever its Panel Fields hold.

# Arguments

  - `D`: The Descriptor, `observations × assets`, changed in place. Its element type must hold `NaN`.
  - `pnl`: The Asset Panel whose active mask is read.

# Validation

  - `size(D) == size(pnl.amsk)`. Raises a `DimensionMismatch`.
  - `eltype(D)` holds `NaN`, through [`nan_fill_value`](@ref). Raises an `ArgumentError`.

# Returns

  - `nothing`. `D` carries the filled Descriptor.

# Examples

```jldoctest
julia> pnl = AssetPanel(; pf = [NumericPanelField(; name = \"a\", vals = [1.0 2.0; 3.0 4.0])],
                        amsk = [true false; true true], emsk = [true false; true true]);

julia> D = [1.0 2.0; 3.0 4.0];

julia> PortfolioOptimisers.descriptor_active_fill!(D, pnl)

julia> D
2×2 Matrix{Float64}:
 1.0  NaN
 3.0    4.0
```

# Related

  - [`descriptor`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_active_fill!(D::AbstractMatrix{<:Number}, pnl::AssetPanel)::Nothing
    amsk = pnl.amsk
    @argcheck(size(D) == size(amsk),
              DimensionMismatch("a Descriptor is observations × assets, so it must match the active mask of the Asset Panel, got size(D) = $(size(D)) and size(pnl.amsk) = $(size(amsk))"))
    nan = nan_fill_value(D)
    for k in CartesianIndices(D)
        if !amsk[k]
            D[k] = nan
        end
    end
    return nothing
end
"""
    positive_divide(a::Real, b::Real) -> Real

Divide `a` by `b` where `b` is strictly positive and the quotient is finite, and return `NaN` otherwise.

A ratio Descriptor is undefined where its denominator is zero, and it is meaningless where a quantity that is positive by construction, a market capitalisation or a total of assets, is negative. Both cases give `NaN` rather than a number or an error, so one bad cell costs one cell of the Descriptor and not the whole fit. A `NaN` denominator compares `false` against zero, so it also gives `NaN`.

A quotient of two finite values can overflow, for example `1e300 / 1e-10`. That cell also gives `NaN`. The Asset Panel refuses an infinity in its input, and every cross-sectional transform refuses one in a Descriptor, so an infinite cell would cost the whole fit.

# Mathematical definition

```math
\\begin{align}
q &= \\begin{cases} a / b & b > 0 \\text{ and } a / b \\text{ is finite}\\,, \\\\ \\mathrm{NaN} & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``a``: The numerator.
  - ``b``: The denominator.

# Arguments

  - `a`: The numerator.
  - `b`: The denominator.

# Returns

  - `q::Real`: The quotient, in the type of `a / b`. Where `b` is not positive, that type must hold `NaN`, so a `Rational` quotient raises an `InexactError` there.

# Examples

```jldoctest
julia> PortfolioOptimisers.positive_divide(1.0, 4.0)
0.25

julia> PortfolioOptimisers.positive_divide(1.0, 0.0)
NaN

julia> PortfolioOptimisers.positive_divide(1.0, -2.0)
NaN

julia> PortfolioOptimisers.positive_divide(1.0e300, 1.0e-10)
NaN
```

# Related

  - [`descriptor`](@ref)
  - [`PanelFieldRatio`](@ref)
"""
function positive_divide(a::Real, b::Real)::Real
    q = a / b
    return b > zero(b) && isfinite(q) ? q : oftype(q, NaN)
end

"""
    market_return_series(rd::ReturnsResult, mcap::AbstractString) -> Vector{<:Real}

Build the market return of every observation from a [`ReturnsResult`](@ref).

Every market-relative Descriptor reads this series, so its definition is in one place. The market return is the capitalisation-weighted mean of the returns over the estimation universe, and every member rebuilds it from the Asset Panel rather than take it from the caller. A weight need not be positive. A negative capitalisation is a data error and not a missing value, so it enters the sum as it stands.

# Mathematical definition

```math
\\begin{align}
r_{m,t} &= \\frac{\\sum_{i \\in \\mathcal{E}_{t}} c_{t,i}\\, x_{t,\\,i}}{\\sum_{i \\in \\mathcal{E}_{t}} c_{t,i}}\\,, \\\\
\\mathcal{E}_{t} &= \\left\\{ i : e_{t,i} \\text{ is true, and } x_{t,\\,i} \\text{ and } c_{t,i} \\text{ are finite} \\right\\}\\,.
\\end{align}
```

Where:

  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:x_ti_ret])
  - ``c_{t,i}``: Capitalisation of asset ``i`` at observation ``t``, `NaN` where the Panel Field was not observed.
  - ``e_{t,i}``: Entry of the estimation mask of the Asset Panel.
  - ``\\mathcal{E}_{t}``: The assets that enter the mean at observation ``t``.

# Algorithm

 1. Read the capitalisation ``c`` through [`descriptor_field_values`](@ref), so a cell that a fill touched is `NaN` and leaves ``\\mathcal{E}_{t}``.
 2. At each observation, add up the numerator and the denominator of ``r_{m,t}`` over ``\\mathcal{E}_{t}``.
 3. Refuse a denominator at or below zero, and store the quotient as ``r_{m,t}``.

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - $(arg_dict[:mcap])

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The total weight ``\\sum_{i \\in \\mathcal{E}_{t}} c_{t,i}`` of every observation is strictly positive, because a total of zero or below divides no mean. Raises an `ArgumentError` naming the observation.

# Returns

  - `rm::Vector{<:Real}`: The market return, one entry per observation.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\", vals = [1.0 3.0; 2.0 2.0])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.3], pnl = pnl);

julia> PortfolioOptimisers.market_return_series(rd, \"market_cap\")
2-element Vector{Float64}:
 0.17500000000000002
 0.09999999999999999
```

# Related

  - [`descriptor`](@ref)
  - [`ew_beta_series`](@ref)
  - [`descriptor_field_values`](@ref)
  - [`AssetPanel`](@ref)
"""
function market_return_series(rd::ReturnsResult, mcap::AbstractString)
    pnl = descriptor_asset_panel(rd)
    W = descriptor_field_values(rd, mcap)
    X = rd.X
    emsk = pnl.emsk
    Tf = promote_type(eltype(X), eltype(W))
    rm = Vector{Tf}(undef, size(X, 1))
    for t in axes(X, 1)
        s = zero(Tf)
        w = zero(Tf)
        for i in axes(X, 2)
            x = X[t, i]
            m = W[t, i]
            if emsk[t, i] && isfinite(x) && isfinite(m)
                s += m * x
                w += m
            end
        end
        @argcheck(w > zero(w),
                  ArgumentError("the market return is the capitalisation-weighted mean of the returns over the estimation universe, so an observation needs at least one estimable asset whose return and whose \"$mcap\" are both finite, and whose weights total strictly more than zero. The total weight is $w at observation $t"))
        rm[t] = s / w
    end
    return rm
end
"""
    ew_beta_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real}, decay::Real,
                   min_obs::Integer, min_val::Real,
                   amsk::Option{<:AbstractMatrix{Bool}} = nothing) -> Tuple{Matrix{<:Real}, Vector{<:Real}}

Run the exponentially weighted market beta recursion down each column of a return matrix.

[`EWBeta`](@ref) and [`EWResidualVolatility`](@ref) both call this recursion rather than repeat it. Each asset has its own state and its own count of valid observations, and the market has one state for the whole panel.

# Mathematical definition

```math
\\begin{align}
\\beta_{t,i} &= \\frac{C_{t,i}}{V_{m,t} + \\texttt{min\\_val}}\\,, \\\\
C_{t,i} &= \\lambda C_{t-1,i} + (1 - \\lambda)\\left(x_{t,\\,i} - \\mu_{t-1,i}\\right)\\left(r_{m,t} - \\mu_{m,t-1}\\right)\\,, \\\\
V_{m,t} &= \\lambda V_{m,t-1} + (1 - \\lambda)\\left(r_{m,t} - \\mu_{m,t-1}\\right)^2\\,, \\\\
\\mu_{t,i} &= \\lambda \\mu_{t-1,i} + (1 - \\lambda)\\, x_{t,\\,i}\\,, \\\\
\\mu_{m,t} &= \\lambda \\mu_{m,t-1} + (1 - \\lambda)\\, r_{m,t}\\,.
\\end{align}
```

Where:

  - ``C_{t,i}``: Exponentially weighted covariance of asset ``i`` with the market.
  - ``\\mu_{t,i}``, ``\\mu_{m,t}``: Exponentially weighted means of asset ``i`` and of the market.
  - $(math_dict[:beta_ti_ewb])
  - $(math_dict[:V_mt_ewb])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:lambda_ew])
  - $(math_dict[:min_val_ewb])
  - $(math_dict[:n_i_ew])

Every state starts from zero, and each deviation uses the mean of the previous observation. This is the exponentially weighted form of Welford's recursion, and it has none of the downward bias of a deviation from the updated mean. The states of asset ``i`` advance only at an observation where its return is valid, and they hold their values at every other observation. ``n_i`` counts the observations where they advance. The market states advance at every observation. The call is [`ew_beta_series!`](@ref) from a new state at zero, so a step of the carry fold from an [`EWBetaState`](@ref) runs the same arithmetic.

# Algorithm

 1. Where `amsk` is given, call [`ew_beta_reset!`](@ref), which sets ``\\mu_{t-1,i}``, ``C_{t-1,i}`` and ``n_i`` to zero for an asset that turns inactive. A valid return then needs an active cell. Where `amsk` is `nothing`, no state restarts and a valid return is any finite return.
 2. Update ``\\mu_{m,t}`` and ``V_{m,t}`` from ``r_{m,t}``, and store ``V_{m,t}``.
 3. For each asset with a valid return, update ``\\mu_{t,i}`` and ``C_{t,i}``, and add one to ``n_i``.
 4. Where ``t`` and ``n_i`` have both reached `min_obs`, write a new ``\\beta_{t,i}``. Every other cell holds the last beta of its asset, and an asset that has never reached `min_obs` holds `NaN`. A restart does not clear the held beta, so it holds until ``n_i`` reaches `min_obs` again.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:min_val])
  - `amsk`: Optional active mask, `observations × assets`. The recursion restarts the state of an asset where it turns inactive.

# Returns

  - `B::Matrix{<:Real}`: The betas ``\\beta_{t,i}``, `observations × assets`, before the caller applies a mask.
  - `Vm::Vector{<:Real}`: The market variance ``V_{m,t}``, one entry per observation.

# Examples

```jldoctest
julia> B, Vm = PortfolioOptimisers.ew_beta_series([0.1 0.2; -0.1 0.3; 0.05 -0.05],
                                                  [0.1, -0.05, 0.02], 0.5, 1, 1e-12);

julia> B
3×2 Matrix{Float64}:
 1.0       2.0
 1.33333  -0.666667
 1.4557   -1.26582
```

# Related

  - [`market_return_series`](@ref)
  - [`ew_beta_reset!`](@ref)
  - [`descriptor`](@ref)
  - [`EWBeta`](@ref)
  - [`EWResidualVolatility`](@ref)
"""
function ew_beta_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real}, decay::Real,
                        min_obs::Integer, min_val::Real,
                        amsk::Option{<:AbstractMatrix{Bool}} = nothing)
    (; B, Vm) = ew_beta_series!(ew_beta_state(nothing, X, rm, decay), X, rm, decay, min_obs,
                                min_val, amsk)
    return B, Vm
end
"""
$(DocStringExtensions.TYPEDEF)

The carried state of the exponentially weighted market beta recursion of [`ew_beta_series`](@ref), after the observations that it folded.

The state holds every quantity that the recursion reads from the observation before: the held beta, the mean, the covariance with the market, the count and the activity of each asset, the mean and the variance of the market, and the count of the observations. [`ew_beta_series!`](@ref) folds new observations from it with the arithmetic of the batch call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWBetaState(; b::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                cv::AbstractVector{<:Real}, n::AbstractVector{<:Integer},
                act::AbstractVector{Bool}, mu_m::Real, var_m::Real, t::Integer) -> EWBetaState

Keywords correspond to the struct's fields.

## Validation

  - `b`, `mu`, `cv`, `n` and `act` hold the same number of assets. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`ew_beta_series`](@ref)
  - [`ew_beta_series!`](@ref)
  - [`EWVolatilityState`](@ref)
"""
@concrete struct EWBetaState <: AbstractPartialFitState
    """
    The last beta ``\\beta_{t,i}`` of each asset, `NaN` before the asset reaches `min_obs`.
    """
    b
    """
    The exponentially weighted mean ``\\mu_{t,i}`` of each asset.
    """
    mu
    """
    The exponentially weighted covariance ``C_{t,i}`` of each asset with the market.
    """
    cv
    """
    The count ``n_i`` of the valid observations of each asset since its last restart.
    """
    n
    """
    The activity of each asset at the last observation.
    """
    act
    """
    The exponentially weighted mean ``\\mu_{m,t}`` of the market.
    """
    mu_m
    """
    $(math_dict[:V_mt_ewb])
    """
    var_m
    """
    The count of the observations that the state folded.
    """
    t
    function EWBetaState(b::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                         cv::AbstractVector{<:Real}, n::AbstractVector{<:Integer},
                         act::AbstractVector{Bool}, mu_m::Real, var_m::Real, t::Integer)
        @argcheck(length(b) == length(mu) == length(cv) == length(n) == length(act),
                  DimensionMismatch("the vectors of an EWBetaState hold one entry per asset, got $(map(length, (b, mu, cv, n, act)))"))
        return new{typeof(b), typeof(mu), typeof(cv), typeof(n), typeof(act), typeof(mu_m),
                   typeof(var_m), typeof(t)}(b, mu, cv, n, act, mu_m, var_m, t)
    end
end
function EWBetaState(; b::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                     cv::AbstractVector{<:Real}, n::AbstractVector{<:Integer},
                     act::AbstractVector{Bool}, mu_m::Real, var_m::Real,
                     t::Integer)::EWBetaState
    return EWBetaState(b, mu, cv, n, act, mu_m, var_m, t)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWBetaState`](@ref), so that the copy shares no vector with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::EWBetaState`: A new state, equal to `x`.

# Related

  - [`EWBetaState`](@ref)
"""
function Base.copy(x::EWBetaState)
    return EWBetaState(copy(x.b), copy(x.mu), copy(x.cv), copy(x.n), copy(x.act), x.mu_m,
                       x.var_m, x.t)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`EWBetaState`](@ref): the recursion of the second block starts from the state after the first one, so two states fitted on disjoint blocks do not give the state of their union to the last bit.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`EWBetaState`](@ref)
  - [`ew_beta_series!`](@ref)
"""
function merge_states(::EWBetaState, ::EWBetaState)
    return throw(ArgumentError("an EWBetaState cannot merge two states fitted on disjoint blocks: the recursion of the second block starts from the state after the first one. Fold the second block into the state of the first with ew_beta_series!."))
end
"""
    ew_beta_state(::Nothing, X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                  decay::Real)
    ew_beta_state(st::EWBetaState, X::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                  ::Real)

Returns the state that a fold of the beta recursion of [`ew_beta_series!`](@ref) starts from: a new state at zero for no carried state, or a copy of the carried state.

The vectors of a new state take the type of the returns and of the market return, and its two market scalars also take the type of the decay, as the recursion gives them after its first observation. The first observation reads the same values from both, so the batch call keeps every bit.

# Arguments

  - `st`: The carried state, or `nothing`.
  - `X`: The returns of the step, `observations × assets`.
  - `rm`: The market return of the step, one entry per observation.
  - `decay`: The decay factor of the recursion.

# Validation

  - A carried state holds the assets of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `st::EWBetaState`: The state to fold into.

# Related

  - [`EWBetaState`](@ref)
  - [`ew_beta_series!`](@ref)
"""
function ew_beta_state(::Nothing, X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                       decay::Real)::EWBetaState
    Tf = float_if_integer(promote_type(eltype(X), eltype(rm)))
    N = size(X, 2)
    Tm = promote_type(Tf, typeof(decay))
    return EWBetaState(fill(Tf(NaN), N), zeros(Tf, N), zeros(Tf, N), zeros(Int, N),
                       trues(N), zero(Tm), zero(Tm), 0)
end
function ew_beta_state(st::EWBetaState, X::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                       ::Real)::EWBetaState
    @argcheck(length(st.b) == size(X, 2),
              DimensionMismatch("the beta state carries $(length(st.b)) assets, and the step brings $(size(X, 2))"))
    return copy(st)
end
"""
    ew_beta_series!(st::EWBetaState, X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                    decay::Real, min_obs::Integer, min_val::Real,
                    amsk::Option{<:AbstractMatrix{Bool}})

Runs the exponentially weighted market beta recursion of [`ew_beta_series`](@ref) over the observations of `X` from a state, and returns the betas, the market variance and the state after them.

The batch call [`ew_beta_series`](@ref) is this function from a new state at zero, and a step of the carry fold is this function from the carried state, so a step equals the batch call by construction. The test against `min_obs` counts the observations from the first one that the state folded, not from the first row of `X`.

# Arguments

  - `st`: The state to fold into. Its vectors change in place.
  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:min_val])
  - `amsk`: Optional active mask of the rows of `X`, `observations × assets`.

# Returns

  - `fold::NamedTuple`: `B` and `Vm`, as [`ew_beta_series`](@ref) states them, and `st`, the state after the observations.

# Related

  - [`ew_beta_series`](@ref)
  - [`EWBetaState`](@ref)
  - [`ew_beta_reset!`](@ref)
"""
function ew_beta_series!(st::EWBetaState, X::AbstractMatrix{<:Real},
                         rm::AbstractVector{<:Real}, decay::Real, min_obs::Integer,
                         min_val::Real, amsk::Option{<:AbstractMatrix{Bool}})
    (; b, mu, cv, n, act) = st
    Tf = eltype(b)
    T, N = size(X)
    B = Matrix{Tf}(undef, T, N)
    Vm = Vector{Tf}(undef, T)
    mu_m = st.mu_m
    var_m = st.var_m
    om = one(Tf) - decay
    for t in 1:T
        ew_beta_reset!(amsk, mu, cv, n, act, t)
        r = rm[t]
        dm = r - mu_m
        mu_m = decay * mu_m + om * r
        var_m = decay * var_m + om * dm * dm
        Vm[t] = var_m
        k = st.t + t
        for i in 1:N
            x = X[t, i]
            if isfinite(x) && act[i]
                d = x - mu[i]
                mu[i] = decay * mu[i] + om * x
                cv[i] = decay * cv[i] + om * d * dm
                n[i] += 1
                if k >= min_obs && n[i] >= min_obs
                    b[i] = cv[i] / (var_m + min_val)
                end
            end
            B[t, i] = b[i]
        end
    end
    return (; B = B, Vm = Vm, st = EWBetaState(b, mu, cv, n, act, mu_m, var_m, st.t + T))
end
"""
    ew_beta_reset!(amsk::Nothing, mu::AbstractVector{<:Number},
                   cv::AbstractVector{<:Number},
                   n::AbstractVector{<:Integer}, act::AbstractVector{Bool},
                   t::Integer) -> nothing
    ew_beta_reset!(amsk::AbstractMatrix{Bool}, mu::AbstractVector{<:Number},
                   cv::AbstractVector{<:Number}, n::AbstractVector{<:Integer},
                   act::AbstractVector{Bool}, t::Integer) -> nothing

Restart the state of every asset that turns inactive at one observation, in place.

This function reads the optional active mask of [`ew_beta_series`](@ref) by dispatch, so the recursion needs no branch on it. With no mask, no state restarts and `act` stays `true` everywhere, so any finite return is valid. With a mask, `act` holds the activity of the observation, so the recursion tests one vector and not both the mask and its absence.

# Arguments

  - `amsk`: The active mask, `observations × assets`, or `nothing`.
  - `mu`: Exponentially weighted mean of each asset, changed in place.
  - `cv`: Exponentially weighted covariance of each asset with the market, changed in place.
  - `n`: Count of the valid observations of each asset, changed in place.
  - `act`: Activity of each asset at the previous observation, changed in place.
  - `t`: The observation.

# Returns

  - `nothing`. The four vectors carry the state of observation `t`.

# Related

  - [`ew_beta_series`](@ref)
  - [`EWResidualVolatility`](@ref)
"""
function ew_beta_reset!(::Nothing, ::AbstractVector{<:Number}, ::AbstractVector{<:Number},
                        ::AbstractVector{<:Integer}, ::AbstractVector{Bool},
                        ::Integer)::Nothing
    return nothing
end
function ew_beta_reset!(amsk::AbstractMatrix{Bool}, mu::AbstractVector{<:Number},
                        cv::AbstractVector{<:Number}, n::AbstractVector{<:Integer},
                        act::AbstractVector{Bool}, t::Integer)::Nothing
    for i in eachindex(mu, cv, n, act)
        if act[i] && !amsk[t, i]
            mu[i] = zero(eltype(mu))
            cv[i] = zero(eltype(cv))
            n[i] = 0
        end
        act[i] = amsk[t, i]
    end
    return nothing
end
"""
    lookback(est::AbstractEstimator) -> Option{<:Integer}
    lookback(ests::AbstractVector) -> Option{<:Integer}

Return the look-back of an estimator: the number of trailing observations that its value at one observation reads.

The count includes the observation of the value, so a Descriptor that reads its own row has a look-back of one. A value that is computed from the last `lookback` rows of a panel alone equals the value at the last row of the full panel. A fold of a [`CrossSectionalFactorPrior`](@ref) can therefore keep only those rows. `nothing` states an unbounded look-back, because the value is a recursion from the first observation. An estimator that states no look-back of its own takes the fallback, which returns `nothing`. Its fold keeps every row, so its value stays exact. Under [`FoldOnly`](@ref) the prior refuses such an estimator in its constructor, as [`carry_growing_parts`](@ref) states.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`PanelFieldRatio`](@ref), a [`PanelFieldLog`](@ref) and a [`Passthrough`](@ref) read their own row, so each returns one. So do a [`ConstantExposure`](@ref), a [`OneHotExposure`](@ref), a [`CurrencyExposure`](@ref) and a [`CustomValueReturnForecast`](@ref).
 2. A [`GrowthRate`](@ref), a [`ChangeToScale`](@ref) and a [`ChangeInIntensity`](@ref) read their own row and the row `lag` observations earlier, so each returns `lag + 1`.
 3. A [`RollingLogReturn`](@ref) returns `skip + window`, and a [`RollingMax`](@ref) returns `window`.
 4. A vector of estimators returns the largest look-back of its members, and `nothing` when one member returns `nothing`. A [`CompositeExposure`](@ref) returns it over its Descriptors, and a [`FixedWeightedReturnForecast`](@ref) over the Descriptors of its scores. An [`ObservedExposure`](@ref) returns the look-back of the member it wraps.
 5. A [`DerivedExposure`](@ref) returns one. The factor it is derived from is in the same factor list, so the look-back of the prior counts the rows of that factor.
 6. A [`CrossSectionalFactorPrior`](@ref) returns the look-back of its factors plus `lag`, because the regression of one observation reads the Factor Exposures `lag` observations earlier. The look-back of its Return Forecast Estimator replaces that sum when it is larger. `nothing` from either part gives `nothing`.
 7. Every other estimator returns `nothing`. This covers every exponentially weighted Descriptor, an [`ExpWeightedReturnForecast`](@ref) and a [`TargetReturnForecast`](@ref), which fit over every observation.

The look-back of the prior counts the Asset Panel rows that its Factor Exposures and its Return Forecast read. Its moment estimators `pe` and `ve` read every factor return and every idiosyncratic return, and a fold keeps their state rather than their rows.

# Arguments

  - `est`: A Descriptor Estimator, an Exposure Estimator, a Return Forecast Estimator or a [`CrossSectionalFactorPrior`](@ref).
  - `ests`: A vector of estimators.

# Returns

  - `L::Option{<:Integer}`: The look-back, at least one, or `nothing` when it is unbounded.

# Examples

```jldoctest
julia> PortfolioOptimisers.lookback(RollingMomentum())
273

julia> PortfolioOptimisers.lookback(GrowthRate(; field = \"sales_ttm\", lag = 252))
253

julia> isnothing(PortfolioOptimisers.lookback(EWMomentum()))
true

julia> PortfolioOptimisers.lookback(CompositeExposure(; descriptors = [Reversal(), MaxReturn()]))
21

julia> PortfolioOptimisers.lookback(CrossSectionalFactorPrior(;
                                                              factors = [\"momentum\" =>
                                                                             CompositeExposure(;
                                                                                               descriptors = [RollingMomentum()])],
                                                              lag = 2))
275
```

# Related

  - [`descriptor`](@ref)
  - [`factor_exposure`](@ref)
  - [`return_forecast`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`AbstractDescriptorEstimator`](@ref)
"""
function lookback(::AbstractEstimator)::Nothing
    return nothing
end
function lookback(ests::AbstractVector)::Option{<:Integer}
    return lookback_max(lookback, ests)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the largest look-back of a vector of estimators, as the verb `f` counts it, or `nothing` when one member answers `nothing`. An empty vector answers one.

# Arguments

  - `f`: The verb of the look-back, [`lookback`](@ref) or [`carry_lookback`](@ref).
  - `ests`: A vector of estimators.

# Returns

  - `L::Option{<:Integer}`: The largest look-back, or `nothing`.

# Related

  - [`lookback`](@ref)
  - [`carry_lookback`](@ref)
"""
function lookback_max(f, ests::AbstractVector)::Option{<:Integer}
    L = 1
    for est in ests
        l = f(est)
        if isnothing(l)
            return nothing
        end
        L = max(L, l)
    end
    return L
end

export descriptor
public AbstractDescriptorEstimator, lookback
