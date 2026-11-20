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
    descriptor_field_values(rd::ReturnsResult, x::Nothing) -> Matrix{<:Real}
    descriptor_field_values(rd::ReturnsResult, name::AbstractString) -> Matrix{<:Real}
    descriptor_field_values(rd::ReturnsResult,
                            terms::AbstractVector{<:Pair{<:AbstractString, <:Real}}) -> Matrix{<:Real}
    descriptor_field_values(rd::ReturnsResult,
                            names::AbstractVector{<:AbstractString}) -> Matrix{<:Real}
    descriptor_field_values(rd::ReturnsResult, de::AbstractDescriptorEstimator) -> Matrix{<:Real}

Read one numeric Panel Field, a linear combination or a product of numeric Panel Fields, the absolute returns, or a Descriptor, out of a [`ReturnsResult`](@ref), for a Descriptor.

A term takes five forms, and Julia selects the method from the form. Forms 2 to 4 are a [`PanelTermType`](@ref).

 1. `nothing` reads the absolute returns `abs.(rd.X)`. Returns are not a Panel Field, so this is how a side of an [`EWVolumeRatio`](@ref) names them.
 2. A name reads that one Panel Field.
 3. A vector of `name => coefficient` pairs reads their sum. `[\"a\" => 1, \"b\" => -1]` reads `a - b`.
 4. A vector of names reads their product. `[\"adj_close\", \"adj_volume\"]` reads a traded amount out of a price and a volume, which no single Panel Field carries.
 5. A Descriptor Estimator reads its Descriptor through [`descriptor`](@ref). The Descriptor is `NaN` outside the universe, as a Panel Field is.

Every Descriptor Estimator reads its Panel Fields through this function, and so does a [`PanelFieldScorer`](@ref). A blank cell never reaches a `ReturnsResult`, because [`asset_panel`](@ref) resolves each one to a fill value and records the resolution in the observed-mask column of the field. A cell outside the universe holds a finite value too. This function writes `NaN` into both kinds of cell, through [`panel_field_values`](@ref) with `inactive = NaN`, so a Descriptor cannot mistake a fill value or the value of an inactive cell for data. A lag or a recursion that reaches back to a cell before a listing, or into a stretch where the asset is out of the universe, reads `NaN` there.

# Algorithm

 1. Check that the Panel Field is numeric and time-varying.
 2. Read it through [`panel_field_values`](@ref) with `inactive = NaN` and `unobserved = NaN`. A `Float32` field stays `Float32`.
 3. For a vector of `name => coefficient` pairs, read each named field the same way and multiply it by its coefficient. Return the sum of those terms in the type that all of them promote to, so the order of the terms does not change the result. A `NaN` in any term is a `NaN` in the sum.
 4. For a vector of names, read each named field the same way. Return the product of those fields in the type that all of them promote to. A `NaN` in any field is a `NaN` in the product.

`nothing` and a Descriptor Estimator skip the steps above. A Descriptor Estimator reads its Panel Fields through these steps inside its own [`descriptor`](@ref).

# Arguments

  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`, unless the term is `nothing`.
  - `x`: `nothing`, for the absolute returns.
  - `name`: The Panel Field's name.
  - `terms`: The Panel Fields to combine, each paired with its coefficient. `[\"a\" => 1, \"b\" => -1]` reads `a - b`.
  - `names`: The Panel Fields to multiply.
  - `de`: The Descriptor Estimator.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - `name` names a Panel Field, and its kind is [`NumericPanelField`](@ref). Raises a `KeyError` or an `ArgumentError`.
  - `!isempty(terms)` and `!isempty(names)`. Raises an [`IsEmptyError`](@ref).
  - The rules of the [`descriptor`](@ref) of `de`.

# Returns

  - `V::Matrix{<:Real}`: The values, `observations × assets`, `NaN` where a Panel Field that the term names was not observed and where the asset is outside the universe.

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

julia> PortfolioOptimisers.descriptor_field_values(rd, [\"mcap\", \"debt\"])
2×2 Matrix{Float64}:
   0.5  NaN
 NaN      8.0
```

# Related

  - [`descriptor`](@ref)
  - [`panel_field_values`](@ref)
  - [`panel_field`](@ref)
  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`NumericPanelField`](@ref)
  - [`assert_panel_terms`](@ref)
"""
function descriptor_field_values(rd::ReturnsResult, ::Nothing)::Matrix{<:Real}
    return abs.(rd.X)
end
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
function descriptor_field_values(rd::ReturnsResult,
                                 names::AbstractVector{<:AbstractString})::Matrix{<:Real}
    @argcheck(!isempty(names),
              IsEmptyError("a Panel Field product needs at least one name"))
    Vs = [descriptor_field_values(rd, name) for name in names]
    V = similar(Vs[1], mapreduce(eltype, promote_type, Vs))
    V .= Vs[1]
    for k in 2:length(Vs)
        V .*= Vs[k]
    end
    return V
end
function descriptor_field_values(rd::ReturnsResult,
                                 de::AbstractDescriptorEstimator)::Matrix{<:Real}
    return descriptor(de, rd)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the state of a Descriptor after the observations that it folded, and the Descriptor of each of those observations.

This type is an implementation detail and is not intended for direct use. The fold of each Descriptor that carries a state makes it: [`lag_descriptor_fold`](@ref), [`ew_mean_fold`](@ref), [`ew_volatility_fold`](@ref) and [`ew_beta_fold`](@ref). The batch call of [`descriptor`](@ref) reads `D`, and [`descriptor_step`](@ref) reads both. One shape is shared by the four folds, so it is one type.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`descriptor_step`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct DescriptorFold
    """
    `st`: The state after the observations, or `nothing` where the estimator does not fold.
    """
    st
    """
    `D`: The Descriptor of each observation, `observations × assets`.
    """
    D
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the two forms of an Exposure Estimator after a step of the carry fold of a [`CrossSectionalFactorPrior`](@ref): the form that folds the next step, and the form that the code of the batch fit reads.

This type is an implementation detail and is not intended for direct use. [`descriptor_carry`](@ref), [`descriptor_carried`](@ref) and [`cross_sectional_descriptor_carry`](@ref) make it, and the carry fold reads both fields. The two forms are estimators of one family, so a record names them where a plain tuple would let a swap pass silently.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`descriptor_carry`](@ref)
  - [`CarriedDescriptor`](@ref)
"""
@concrete struct CarriedEstimators
    """
    `xf`: The estimator with the state after the new observations.
    """
    xf
    """
    `xv`: The estimator whose stateful Descriptors are [`CarriedDescriptor`](@ref)s of the new observations.
    """
    xv
end
"""
$(DocStringExtensions.TYPEDEF)

The answer of a step of a **Descriptor Estimator** that folds new observations from a carried state: the estimator with the state after them, and the Descriptor of each of them.

[`descriptor_step`](@ref) returns it, and a Descriptor of a caller that folds builds it in its own method of that verb. So the type is the contract of the verb, and its keyword constructor checks the types of the two values. `partial_fit!` keeps `de`, and the carry fold of a [`CrossSectionalFactorPrior`](@ref) reads both fields through [`descriptor_carried`](@ref). A caller reads the fields by name, as `res.D` or `(; de, D) = res`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DescriptorStepResult(de::AbstractDescriptorEstimator, D::MatNum)
    DescriptorStepResult(; de::AbstractDescriptorEstimator, D::MatNum)

The keyword constructor takes each field by name, with no default, and calls the positional constructor.

## Validation

  - `de` is an [`AbstractDescriptorEstimator`](@ref), and `D` is a matrix of numbers. Otherwise, the keyword constructor raises a `TypeError`.

# Examples

```jldoctest
julia> res = PortfolioOptimisers.DescriptorStepResult(; de = RollingLogReturn(; window = 2),
                                                      D = [0.1 0.2; 0.3 NaN]);

julia> res.D
2×2 Matrix{Float64}:
 0.1    0.2
 0.3  NaN
```

# Related

  - [`descriptor_step`](@ref)
  - [`descriptor_carried`](@ref)
  - [`AbstractDescriptorEstimator`](@ref)
"""
@concrete struct DescriptorStepResult <: AbstractResult
    """
    The estimator with the state after the new observations in its state field.
    """
    de
    """
    The Descriptor of each new observation, `observations × assets`, equal to the batch call over every observation from the first one.
    """
    D
    function DescriptorStepResult(de::AbstractDescriptorEstimator, D::MatNum)
        return new{typeof(de), typeof(D)}(de, D)
    end
end
function DescriptorStepResult(; de::AbstractDescriptorEstimator,
                              D::MatNum)::DescriptorStepResult
    return DescriptorStepResult(de, D)
end
"""
    descriptor_asset_panel(rd::ReturnsResult) -> AssetPanel

Read the Asset Panel a Descriptor needs out of a [`ReturnsResult`](@ref).

[`descriptor_field_values`](@ref) refuses a `ReturnsResult` that holds no Asset Panel. A Descriptor over the returns reads no Panel Field, so it calls this function to get the active mask that [`descriptor_active_fill!`](@ref) reads. A `ReturnsResult` with no panel gives the mask-only panel of [`returns_with_universe`](@ref), which the ingestion layer would have emitted.

# Arguments

  - $(arg_dict[:rd])

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref), or `rd.X` is not `nothing`. Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel the `ReturnsResult` holds, or the mask-only panel of its returns.

# Related

  - [`descriptor`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`descriptor_field_values`](@ref)
  - [`AssetPanel`](@ref)
"""
function descriptor_asset_panel(rd::ReturnsResult)::AssetPanel
    pnl = returns_with_universe(rd).pnl
    @argcheck(!isnothing(pnl),
              IsNothingError("a Descriptor is `NaN` wherever the active mask of an Asset Panel is `false`, and rd holds neither a panel nor asset returns. Build the ReturnsResult with its returns, or with the `pnl` that asset_panel returns."))
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

A fill changes an array that its caller owns, so it cannot widen the element type. An `Integer`, a `Rational` or a `Complex{Int}` holds no `NaN`, and the conversion would raise an `InexactError` at the first inactive cell. This function converts once, before the fill writes anything. Every element type that holds `NaN` passes, a number type from another package included. Every other type raises an `ArgumentError` whose message names the repair.

# Arguments

  - `A`: The array the fill writes into.

# Validation

  - `eltype(A)` holds `NaN`. Raises an `ArgumentError`.

# Returns

  - `nan::Number`: `NaN` converted to `eltype(A)`.

# Related

  - [`descriptor_active_fill!`](@ref)
  - [`exposure_active_fill!`](@ref)
  - [`one_hot_fill!`](@ref)
"""
function nan_fill_value(A::AbstractArray{<:Number})::Number
    T = eltype(A)
    try
        return convert(T, NaN)
    catch err
        if !(err isa InexactError)
            rethrow()
        end
        throw(ArgumentError("a fill writes NaN into the cells that hold no value, in place, so the element type of the array must hold NaN, and $T does not. Convert the array to a floating point type before the call."))
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
  - [`ew_regression_series!`](@ref)
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
$(DocStringExtensions.TYPEDEF)

The exponentially weighted moments of the return series that a market beta Descriptor regresses, one set per asset.

Each asset carries its own observation count, its own weight sum, and the means and the covariance numerator of the `K` series it tracks. The market return is the first series and the asset return is the last. An [`EWMacroSensitivity`](@ref) puts its reference return between them. An update adds one observation of all `K` series of one asset, so the series of an asset always share their observations, and an asset that is listed late or misses an observation weights the market over its own observations alone. The weighted Welford recursion updates the moments, so the first observation sets the means and every later one corrects them. A mean is stored as the last value of its series plus an offset, which keeps a small variation of a large mean.

This type is an implementation detail and is not intended for direct use. An [`EWBetaState`](@ref) holds it.

# Mathematical definition

```math
\\begin{align}
W_{t,i} &= \\lambda W_{t-1,i} + \\omega\\,, \\\\
\\boldsymbol{\\delta}_{t,i} &= \\boldsymbol{y}_{t,i} - \\boldsymbol{\\mu}_{t-1,i}\\,, \\\\
\\mathbf{S}_{t,i} &= \\lambda \\mathbf{S}_{t-1,i} + \\omega \\frac{\\lambda W_{t-1,i}}{W_{t,i}}\\, \\boldsymbol{\\delta}_{t,i} \\boldsymbol{\\delta}_{t,i}^{\\intercal}\\,, \\\\
\\boldsymbol{\\mu}_{t,i} &= \\boldsymbol{y}_{t,i} - \\frac{\\lambda W_{t-1,i}}{W_{t,i}}\\, \\boldsymbol{\\delta}_{t,i}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{y}_{t,i}``: The `K` returns of asset ``i`` at an observation that updates it.
  - ``W_{t,i}``: Weight sum of the observations of asset ``i``.
  - ``\\boldsymbol{\\mu}_{t,i}``: Exponentially weighted means of the `K` series of asset ``i``.
  - ``\\mathbf{S}_{t,i}``: Covariance numerator of the `K` series of asset ``i``. The exponentially weighted covariance is ``\\mathbf{S}_{t,i} / W_{t,i}``.
  - ``\\lambda``: `decay`, the factor that multiplies the weights of the earlier observations.
  - ``\\omega``: `om`, the weight of a new observation, `1 - decay`. The squared weights of the shrinkage use `decay^2` and `(1 - decay)^2`.

An observation that does not update asset ``i`` leaves its moments as they are, with no decay.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`EWBetaState`](@ref)
  - [`ew_moments_update!`](@ref)
  - [`ew_model_beta`](@ref)
  - [`EWReturnWindow`](@ref)
"""
@concrete struct EWMoments
    """
    The count of the observations of each asset since its last reset.
    """
    n
    """
    The weight sum ``W_{t,i}`` of each asset.
    """
    w
    """
    The last value of each series of each asset, `K × assets`.
    """
    r
    """
    The offset of each mean from the last value of its series, `K × assets`.
    """
    o
    """
    The covariance numerator ``\\mathbf{S}_{t,i}`` of each asset, `K × K × assets`.
    """
    C
    """
    The deviations of one update, `K` entries. The update writes them before it reads them.
    """
    d
    """
    The decay ``\\lambda`` of the weights.
    """
    decay
    """
    The weight ``\\omega`` of a new observation.
    """
    om
    function EWMoments(n::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                       r::AbstractMatrix{<:Real}, o::AbstractMatrix{<:Real},
                       C::AbstractArray{<:Real, 3}, d::AbstractVector{<:Real}, decay::Real,
                       om::Real)
        @argcheck(length(n) == length(w) == size(r, 2) == size(o, 2) == size(C, 3) &&
                  size(r, 1) == size(o, 1) == size(C, 1) == size(C, 2) == length(d),
                  DimensionMismatch("the moments of an EWMoments hold K series of each asset, got $(length(n)) counts, $(length(w)) weight sums, $(size(r)) last values, $(size(o)) offsets, $(size(C)) covariance numerators and $(length(d)) deviations"))
        return new{typeof(n), typeof(w), typeof(r), typeof(o), typeof(C), typeof(d),
                   typeof(decay), typeof(om)}(n, w, r, o, C, d, decay, om)
    end
end
"""
    ew_moments(K::Integer, N::Integer, ::Type{Tf}, decay::Real, om::Real) -> EWMoments

Returns new [`EWMoments`](@ref) of `K` series of `N` assets at zero, of element type `Tf`.

# Arguments

  - `K`: The number of series of each asset.
  - `N`: The number of assets.
  - `Tf`: The element type of the moments.
  - $(arg_dict[:decay])
  - `om`: The weight of a new observation.

# Returns

  - `mo::EWMoments`: The moments, every count and every sum at zero.

# Related

  - [`EWMoments`](@ref)
  - [`ew_beta_state`](@ref)
"""
function ew_moments(K::Integer, N::Integer, ::Type{Tf}, decay::Real,
                    om::Real)::EWMoments where {Tf}
    return EWMoments(zeros(Int, N), zeros(Tf, N), zeros(Tf, K, N), zeros(Tf, K, N),
                     zeros(Tf, K, K, N), zeros(Tf, K), decay, om)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWMoments`](@ref), so that the copy shares no array with the original.

# Arguments

  - `x`: The moments to copy.

# Returns

  - `mo::EWMoments`: New moments, equal to `x`.

# Related

  - [`EWMoments`](@ref)
"""
function Base.copy(x::EWMoments)
    return EWMoments(copy(x.n), copy(x.w), copy(x.r), copy(x.o), copy(x.C), copy(x.d),
                     x.decay, x.om)
end
"""
    ew_moments_reset!(mo::EWMoments, i::Integer) -> nothing
    ew_moments_reset!(::Nothing, ::Integer) -> nothing

Clears the moments of asset `i`, in place, so its next observation starts a new history. `nothing` holds no moments, so it has nothing to clear.

# Arguments

  - `mo`: The moments, or `nothing`.
  - `i`: The asset.

# Returns

  - `nothing`.

# Related

  - [`EWMoments`](@ref)
  - [`ew_regression_reset!`](@ref)
"""
function ew_moments_reset!(mo::EWMoments, i::Integer)::Nothing
    mo.n[i] = 0
    mo.w[i] = zero(eltype(mo.w))
    mo.r[:, i] .= zero(eltype(mo.r))
    mo.o[:, i] .= zero(eltype(mo.o))
    mo.C[:, :, i] .= zero(eltype(mo.C))
    return nothing
end
function ew_moments_reset!(::Nothing, ::Integer)::Nothing
    return nothing
end
"""
    ew_moments_update!(mo::EWMoments, y::AbstractVector{<:Real}, i::Integer) -> nothing
    ew_moments_update!(::Nothing, ::AbstractVector{<:Real}, ::Integer) -> nothing

Adds one observation of the series of asset `i` to its moments, in place, by the recursion that [`EWMoments`](@ref) states. `nothing` holds no moments, so it adds nothing.

The moments read the first `K` entries of `y`, so the moments of the squared weights, which track the market alone, read the market return off the vector of every series.

# Arguments

  - `mo`: The moments, or `nothing`.
  - `y`: The values of the series, the market return first.
  - `i`: The asset.

# Returns

  - `nothing`.

# Related

  - [`EWMoments`](@ref)
  - [`ew_regression_update!`](@ref)
"""
function ew_moments_update!(mo::EWMoments, y::AbstractVector{<:Real}, i::Integer)::Nothing
    (; r, o, C, d, decay, om) = mo
    K = length(d)
    w0 = decay * mo.w[i]
    wt = w0 + om
    f = w0 / wt
    for a in 1:K
        d[a] = (y[a] - r[a, i]) - o[a, i]
    end
    for b in 1:K, a in 1:K
        C[a, b, i] = decay * C[a, b, i] + om * f * d[a] * d[b]
    end
    for a in 1:K
        o[a, i] = -f * d[a]
        r[a, i] = y[a]
    end
    mo.w[i] = wt
    mo.n[i] += 1
    return nothing
end
function ew_moments_update!(::Nothing, ::AbstractVector{<:Real}, ::Integer)::Nothing
    return nothing
end
"""
    ew_moments_vary(mo::EWMoments, a::Integer, i::Integer) -> Bool

Returns whether series `a` of asset `i` varies by more than the rounding noise of its mean.

A series that does not vary cannot be regressed on. The test compares the weighted standard deviation of the series with `32 eps` times the absolute value of its weighted mean, on the scale of the weight sum, so it needs no square of the mean and no division. A series of no observation does not vary.

# Arguments

  - `mo`: The moments.
  - `a`: The series, `1` for the market.
  - `i`: The asset.

# Returns

  - `flag::Bool`: `true` where the series varies.

# Related

  - [`ew_model_beta`](@ref)
  - [`EWMoments`](@ref)
"""
function ew_moments_vary(mo::EWMoments, a::Integer, i::Integer)::Bool
    mu = mo.r[a, i] + mo.o[a, i]
    return sqrt(mo.C[a, a, i]) > sqrt(mo.w[i]) * (32 * eps(typeof(mu)) * abs(mu))
end
"""
$(DocStringExtensions.TYPEDEF)

The running means of the return series of each asset over the window of `agg_obs` observations that is not yet complete.

An exponentially weighted beta Descriptor with `agg_obs > 1` regresses one mean per window rather than one return per observation. Two series that close at different times of day record one move on different observations, and a window holds the move of both. Each asset averages its series over the observations of the window where all of them are finite and the asset is active, so the series of an asset share their observations here too. An inactive observation clears the open window of the asset. A mean is stored as the last value plus an offset, as in [`EWMoments`](@ref).

This type is an implementation detail and is not intended for direct use. An [`EWBetaState`](@ref) holds it, and holds `nothing` in its place where `agg_obs = 1`.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`EWBetaState`](@ref)
  - [`ew_regression_take!`](@ref)
  - [`ew_regression_close!`](@ref)
"""
@concrete struct EWReturnWindow
    """
    The count of the observations of each asset in the open window.
    """
    n
    """
    The last value of each series of each asset, `K × assets`.
    """
    r
    """
    The offset of each mean from the last value of its series, `K × assets`.
    """
    o
    function EWReturnWindow(n::AbstractVector{<:Integer}, r::AbstractMatrix{<:Real},
                            o::AbstractMatrix{<:Real})
        @argcheck(length(n) == size(r, 2) && size(r) == size(o),
                  DimensionMismatch("the window of an EWReturnWindow holds K series of each asset, got $(length(n)) counts, $(size(r)) last values and $(size(o)) offsets"))
        return new{typeof(n), typeof(r), typeof(o)}(n, r, o)
    end
end
"""
    ew_return_window(agg_obs::Integer, K::Integer, N::Integer, ::Type{Tf})
        -> Option{<:EWReturnWindow}

Returns a new empty [`EWReturnWindow`](@ref) of `K` series of `N` assets, or `nothing` where `agg_obs = 1`, because a window of one observation holds the observation itself.

# Arguments

  - $(arg_dict[:agg_obs])
  - `K`: The number of series of each asset.
  - `N`: The number of assets.
  - `Tf`: The element type of the means.

# Returns

  - `wn::Option{<:EWReturnWindow}`: The window, or `nothing`.

# Related

  - [`EWReturnWindow`](@ref)
  - [`ew_beta_state`](@ref)
"""
function ew_return_window(agg_obs::Integer, K::Integer, N::Integer,
                          ::Type{Tf})::Option{<:EWReturnWindow} where {Tf}
    return if isone(agg_obs)
        nothing
    else
        EWReturnWindow(zeros(Int, N), zeros(Tf, K, N), zeros(Tf, K, N))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWReturnWindow`](@ref), so that the copy shares no array with the original.

# Arguments

  - `x`: The window to copy.

# Returns

  - `wn::EWReturnWindow`: A new window, equal to `x`.

# Related

  - [`EWReturnWindow`](@ref)
"""
function Base.copy(x::EWReturnWindow)
    return EWReturnWindow(copy(x.n), copy(x.r), copy(x.o))
end
"""
    ew_window_reset!(wn::EWReturnWindow, i::Integer) -> nothing
    ew_window_reset!(::Nothing, ::Integer) -> nothing

Clears the open window of asset `i`, in place. `nothing` holds no window, so it has nothing to clear.

# Arguments

  - `wn`: The window, or `nothing`.
  - `i`: The asset.

# Returns

  - `nothing`.

# Related

  - [`EWReturnWindow`](@ref)
  - [`ew_regression_reset!`](@ref)
"""
function ew_window_reset!(wn::EWReturnWindow, i::Integer)::Nothing
    wn.n[i] = 0
    wn.r[:, i] .= zero(eltype(wn.r))
    wn.o[:, i] .= zero(eltype(wn.o))
    return nothing
end
function ew_window_reset!(::Nothing, ::Integer)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

The carried state of the exponentially weighted regression of the returns on the market return, after the observations that it folded.

An [`EWBeta`](@ref) and the beta of an [`EWResidualVolatility`](@ref) carry it, and an [`EWMacroSensitivityState`](@ref) holds it with its own counts. The state holds every quantity that [`ew_regression_series!`](@ref) reads from the observation before: the moments of each asset, the moments of the squared weights that the shrinkage reads, the open window of the averages, the held beta of each asset, and the count of the observations. A step of the carry fold of a [`CrossSectionalFactorPrior`](@ref) folds its new observations from it with the arithmetic of the batch call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWBetaState(; mo::EWMoments, sq::Option{<:EWMoments} = nothing,
                wn::Option{<:EWReturnWindow} = nothing, b::AbstractVector{<:Real},
                y::AbstractVector{<:Real}, t::Integer) -> EWBetaState

Keywords correspond to the struct's fields.

## Validation

  - `mo`, `sq`, `wn` and `b` hold the same number of assets, and `y` one entry per series of `mo`. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`ew_regression_series!`](@ref)
  - [`ew_beta_state`](@ref)
  - [`EWMoments`](@ref)
  - [`EWReturnWindow`](@ref)
  - [`EWVolatilityState`](@ref)
"""
@concrete struct EWBetaState <: AbstractPartialFitState
    """
    The [`EWMoments`](@ref) of the market, of the reference series where there is one, and of each asset.
    """
    mo
    """
    The [`EWMoments`](@ref) of the market under squared weights, for the shrinkage of an [`EWBeta`](@ref) with a `group`, or `nothing`.
    """
    sq
    """
    The [`EWReturnWindow`](@ref) of the window that is not yet complete, or `nothing` where `agg_obs = 1`.
    """
    wn
    """
    The beta of each asset after the last complete window, shrunk where the estimator shrinks, `NaN` where the asset has none.
    """
    b
    """
    The values of the series of one observation, one entry per series. A row of the recursion writes them before it reads them.
    """
    y
    """
    The count of the observations that the state folded.
    """
    t
    function EWBetaState(mo::EWMoments, sq::Option{<:EWMoments},
                         wn::Option{<:EWReturnWindow}, b::AbstractVector{<:Real},
                         y::AbstractVector{<:Real}, t::Integer)
        N = length(b)
        @argcheck(length(mo.n) == N &&
                  (isnothing(sq) || length(sq.n) == N) &&
                  (isnothing(wn) || length(wn.n) == N) &&
                  length(y) == length(mo.d),
                  DimensionMismatch("the parts of an EWBetaState hold one entry per asset and the values one entry per series, got $N betas, $(length(mo.n)) assets of the moments and $(length(y)) values of $(length(mo.d)) series"))
        return new{typeof(mo), typeof(sq), typeof(wn), typeof(b), typeof(y), typeof(t)}(mo,
                                                                                        sq,
                                                                                        wn,
                                                                                        b,
                                                                                        y,
                                                                                        t)
    end
end
function EWBetaState(; mo::EWMoments, sq::Option{<:EWMoments} = nothing,
                     wn::Option{<:EWReturnWindow} = nothing, b::AbstractVector{<:Real},
                     y::AbstractVector{<:Real}, t::Integer)::EWBetaState
    return EWBetaState(mo, sq, wn, b, y, t)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWBetaState`](@ref), so that the copy shares no array with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::EWBetaState`: A new state, equal to `x`.

# Related

  - [`EWBetaState`](@ref)
"""
function Base.copy(x::EWBetaState)
    return EWBetaState(copy(x.mo), isnothing(x.sq) ? nothing : copy(x.sq),
                       isnothing(x.wn) ? nothing : copy(x.wn), copy(x.b), copy(x.y), x.t)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`EWBetaState`](@ref): the recursion of the second block starts from the state after the first one, so two states fitted on disjoint blocks do not give the state of their union to the last bit.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`EWBetaState`](@ref)
  - [`ew_regression_series!`](@ref)
"""
function merge_states(::EWBetaState, ::EWBetaState)
    return throw(ArgumentError("an EWBetaState cannot merge two states fitted on disjoint blocks: the recursion of the second block starts from the state after the first one. Fold the second block into the state of the first with partial_fit!."))
end
"""
    ew_beta_state(::Nothing, X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                  rf::Option{<:AbstractVector{<:Real}}, decay::Real,
                  group::Option{<:AbstractString}, agg_obs::Integer)
    ew_beta_state(st::EWBetaState, X::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                  rf::Option{<:AbstractVector{<:Real}}, ::Real,
                  group::Option{<:AbstractString}, agg_obs::Integer)
    ew_beta_state(st::AbstractPartialFitState, args...)

Returns the state that a fold of [`ew_regression_series!`](@ref) starts from: a new state at zero for no carried state, or a copy of the carried state.

The moments of a new state track two series, the market and the asset, or three where a reference return `rf` joins them. They take the type of the returns, of the market return, of the reference return and of the decay. The moments of the squared weights exist where a `group` asks for the shrinkage, and the window exists where `agg_obs > 1`.

# Arguments

  - `st`: The carried state, or `nothing`.
  - `X`: The returns of the step, `observations × assets`.
  - `rm`: The market return of the step, one entry per observation.
  - `rf`: The reference return of the step, or `nothing`.
  - $(arg_dict[:decay])
  - $(arg_dict[:group_ewb])
  - $(arg_dict[:agg_obs])

# Validation

  - A carried state holds the assets of `X`. A `DimensionMismatch` is thrown otherwise.
  - A carried state was folded by the same configuration: a reference return, a `group` and an `agg_obs > 1` each exactly where the state holds their part. An `ArgumentError` is thrown otherwise, and also for a carried state of another type, because the state belongs to another Descriptor.

# Returns

  - `st::EWBetaState`: The state to fold into.

# Related

  - [`EWBetaState`](@ref)
  - [`ew_regression_series!`](@ref)
"""
function ew_beta_state(::Nothing, X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                       rf::Option{<:AbstractVector{<:Real}}, decay::Real,
                       group::Option{<:AbstractString}, agg_obs::Integer)::EWBetaState
    Tf = promote_type(eltype(X), eltype(rm), isnothing(rf) ? Union{} : eltype(rf),
                      typeof(decay))
    N = size(X, 2)
    K = isnothing(rf) ? 2 : 3
    om = one(decay) - decay
    sq = isnothing(group) ? nothing : ew_moments(1, N, Tf, decay^2, om^2)
    return EWBetaState(ew_moments(K, N, Tf, decay, om), sq,
                       ew_return_window(agg_obs, K, N, Tf), fill(Tf(NaN), N), zeros(Tf, K),
                       0)
end
function ew_beta_state(st::EWBetaState, X::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                       rf::Option{<:AbstractVector{<:Real}}, ::Real,
                       group::Option{<:AbstractString}, agg_obs::Integer)::EWBetaState
    @argcheck(length(st.b) == size(X, 2),
              DimensionMismatch("the beta state carries $(length(st.b)) assets, and the step brings $(size(X, 2))"))
    @argcheck(length(st.y) == (isnothing(rf) ? 2 : 3) &&
              isnothing(st.sq) == isnothing(group) &&
              isnothing(st.wn) == isone(agg_obs),
              ArgumentError("the beta state was folded by another configuration of the Descriptor: a reference return, a group and agg_obs > 1 each need their own part of the state. A Descriptor folds from its own state, or from no state."))
    return copy(st)
end
function ew_beta_state(::AbstractPartialFitState, args...)
    return throw(ArgumentError("the state of this exponentially weighted beta Descriptor is not an EWBetaState, so it belongs to another Descriptor. A Descriptor folds from its own state, or from no state."))
end
"""
    ew_regression_series!(st::EWBetaState, md, X::AbstractMatrix{<:Real},
                          rm::AbstractVector{<:Real}, rf::Option{<:AbstractVector{<:Real}},
                          amsk::AbstractMatrix{Bool}, min_obs::Integer, agg_obs::Integer)

Runs the exponentially weighted regression of the returns on the market return over the observations of `X` from a state, and returns the betas and the state after them.

Every market beta Descriptor runs this one recursion. The batch call runs it from a new state at zero, and a step of the carry fold from the carried state, so a step equals the batch call by construction. The model `md` selects what a complete window writes, by dispatch: `nothing` writes the market beta of [`ew_model_beta`](@ref), an [`EWBetaShrinkage`](@ref) writes it shrunk toward the mean of its group, and the constant-market flags of an [`EWMacroSensitivity`](@ref) write the partial beta of [`ew_macro_beta`](@ref).

# Algorithm

At each observation `t`:

 1. Reset every asset that is inactive at `t` through [`ew_regression_reset!`](@ref): its moments, its open window and its beta. Its history then starts again, so an asset that is listed late, delisted or relisted regresses only the observations of its current listing.
 2. Take every asset that is active and whose market return, reference return and own return are all finite through [`ew_regression_take!`](@ref). With `agg_obs = 1` the observation updates the regression of the asset. Otherwise it joins the open window of the asset.
 3. Where `t` closes a window of `agg_obs` observations, counted from the first observation that the state folded, fold the mean of each open window into its regression through [`ew_regression_close!`](@ref), write the beta of every asset through [`ew_regression_refresh!`](@ref), and shrink it through [`ew_beta_shrink_row!`](@ref).
 4. Write the held beta of every asset into row `t`.

# Arguments

  - `st`: The state to fold into. Its arrays change in place.
  - `md`: The model, `nothing`, an [`EWBetaShrinkage`](@ref) or the constant-market flags of an [`EWMacroSensitivityState`](@ref).
  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `rf`: The reference return, one entry per observation, or `nothing`.
  - `amsk`: The active mask of the observations, `observations × assets`.
  - $(arg_dict[:min_obs])
  - $(arg_dict[:agg_obs])

# Returns

  - `(B, st)::Tuple`: The plain tuple of `B`, the betas, `observations × assets`, and `st`, the state after the observations.

# Related

  - [`EWBetaState`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
  - [`EWResidualVolatility`](@ref)
"""
function ew_regression_series!(st::EWBetaState, md, X::AbstractMatrix{<:Real},
                               rm::AbstractVector{<:Real},
                               rf::Option{<:AbstractVector{<:Real}},
                               amsk::AbstractMatrix{Bool}, min_obs::Integer,
                               agg_obs::Integer)
    T = size(X, 1)
    B = Matrix{eltype(st.b)}(undef, size(X))
    for t in 1:T
        ew_regression_reset!(md, st, amsk, t)
        ew_regression_add!(md, st, X, rm, rf, amsk, t)
        if iszero((st.t + t) % agg_obs)
            ew_regression_close!(st.wn, md, st)
            ew_regression_refresh!(md, st, min_obs)
            ew_beta_shrink_row!(md, st, t)
        end
        B[t, :] = st.b
    end
    return B, EWBetaState(st.mo, st.sq, st.wn, st.b, st.y, st.t + T)
end
"""
    ew_regression_reset!(md, st::EWBetaState, amsk::AbstractMatrix{Bool}, t::Integer)
        -> nothing

Resets, in place, every asset that is inactive at observation `t`: its moments, its moments of the squared weights, its open window, the constant-market flag of the model, and its beta, which turns `NaN`.

# Arguments

  - `md`: The model of [`ew_regression_series!`](@ref).
  - `st`: The state.
  - `amsk`: The active mask, `observations × assets`.
  - `t`: The observation.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`ew_model_reset!`](@ref)
"""
function ew_regression_reset!(md, st::EWBetaState, amsk::AbstractMatrix{Bool},
                              t::Integer)::Nothing
    for i in axes(amsk, 2)
        if !amsk[t, i]
            ew_moments_reset!(st.mo, i)
            ew_moments_reset!(st.sq, i)
            ew_window_reset!(st.wn, i)
            ew_model_reset!(md, i)
            st.b[i] = eltype(st.b)(NaN)
        end
    end
    return nothing
end
"""
    ew_regression_add!(md, st::EWBetaState, X::AbstractMatrix{<:Real},
                       rm::AbstractVector{<:Real}, rf::Option{<:AbstractVector{<:Real}},
                       amsk::AbstractMatrix{Bool}, t::Integer) -> nothing

Takes observation `t` of every asset that is active there and whose series are all finite, through [`ew_regression_take!`](@ref). An asset whose return, market return or reference return is missing takes nothing, so its regression neither advances nor decays.

# Arguments

  - `md`: The model of [`ew_regression_series!`](@ref).
  - `st`: The state. Its vector `y` takes the values of the series.
  - `X`: The returns, `observations × assets`.
  - `rm`: The market return.
  - `rf`: The reference return, or `nothing`.
  - `amsk`: The active mask.
  - `t`: The observation.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`ew_regression_take!`](@ref)
"""
function ew_regression_add!(md, st::EWBetaState, X::AbstractMatrix{<:Real},
                            rm::AbstractVector{<:Real},
                            rf::Option{<:AbstractVector{<:Real}},
                            amsk::AbstractMatrix{Bool}, t::Integer)::Nothing
    y = st.y
    y[1] = rm[t]
    ew_reference_load!(y, rf, t)
    for i in axes(X, 2)
        y[end] = X[t, i]
        if amsk[t, i] && all(isfinite, y)
            ew_regression_take!(st.wn, md, st, i)
        end
    end
    return nothing
end
"""
    ew_reference_load!(y::AbstractVector{<:Real}, ::Nothing, ::Integer) -> nothing
    ew_reference_load!(y::AbstractVector{<:Real}, rf::AbstractVector{<:Real},
                       t::Integer) -> nothing

Writes the reference return of observation `t` into the second entry of the values `y`, or nothing where the regression has no reference series.

# Arguments

  - `y`: The values of the series of one observation.
  - `rf`: The reference return, or `nothing`.
  - `t`: The observation.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_add!`](@ref)
"""
function ew_reference_load!(::AbstractVector{<:Real}, ::Nothing, ::Integer)::Nothing
    return nothing
end
function ew_reference_load!(y::AbstractVector{<:Real}, rf::AbstractVector{<:Real},
                            t::Integer)::Nothing
    y[2] = rf[t]
    return nothing
end
"""
    ew_regression_take!(::Nothing, md, st::EWBetaState, i::Integer) -> nothing
    ew_regression_take!(wn::EWReturnWindow, ::Any, st::EWBetaState, i::Integer) -> nothing

Takes the values `st.y` of one observation of asset `i`, in place. Where there is no window, the observation updates the regression of the asset through [`ew_regression_update!`](@ref). Otherwise it joins the running means of the open window of the asset: each mean is held as the last value `r` plus an offset `o`, as in [`EWMoments`](@ref), and the count `n` of the window grows by one.

# Arguments

  - `wn`: The window of the state, or `nothing`.
  - `md`: The model of [`ew_regression_series!`](@ref).
  - `st`: The state.
  - `i`: The asset.

# Validation

  - The rules of [`ew_regression_update!`](@ref).
  - Every mean of an open window stays finite. Raises an [`IsNonFiniteError`](@ref), because a difference of two finite returns can overflow.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_add!`](@ref)
  - [`EWReturnWindow`](@ref)
"""
function ew_regression_take!(::Nothing, md, st::EWBetaState, i::Integer)::Nothing
    return ew_regression_update!(md, st, st.y, i)
end
function ew_regression_take!(wn::EWReturnWindow, ::Any, st::EWBetaState,
                             i::Integer)::Nothing
    (; r, o) = wn
    y = st.y
    k = wn.n[i]
    f = k / (k + 1)
    for a in axes(r, 1)
        d = (y[a] - r[a, i]) - o[a, i]
        o[a, i] = -f * d
        r[a, i] = y[a]
    end
    wn.n[i] += 1
    @argcheck(all(isfinite, view(o, :, i)),
              IsNonFiniteError("the mean of the returns of asset $i over its window of observations is not finite, because the returns overflow the arithmetic. Rescale the returns."))
    return nothing
end
"""
    ew_regression_close!(::Nothing, ::Any, ::EWBetaState) -> nothing
    ew_regression_close!(wn::EWReturnWindow, md, st::EWBetaState) -> nothing

Folds the window that closes into the regression of every asset, in place, and clears the window. An asset whose window holds at least one observation updates its regression with the means of its series over that window. With no window, each observation already updated the regression, so there is nothing to fold.

# Arguments

  - `wn`: The window of the state, or `nothing`.
  - `md`: The model of [`ew_regression_series!`](@ref).
  - `st`: The state.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`EWReturnWindow`](@ref)
"""
function ew_regression_close!(::Nothing, ::Any, ::EWBetaState)::Nothing
    return nothing
end
function ew_regression_close!(wn::EWReturnWindow, md, st::EWBetaState)::Nothing
    y = st.y
    for i in eachindex(wn.n)
        if wn.n[i] > 0
            for a in eachindex(y)
                y[a] = wn.r[a, i] + wn.o[a, i]
            end
            ew_regression_update!(md, st, y, i)
        end
        ew_window_reset!(wn, i)
    end
    return nothing
end
"""
    ew_regression_update!(md, st::EWBetaState, y::AbstractVector{<:Real}, i::Integer)
        -> nothing

Adds one observation `y` of the series of asset `i` to its regression, in place: the constant-market flag of the model first, through [`ew_model_update!`](@ref), then its moments and its moments of the squared weights, through [`ew_moments_update!`](@ref).

# Arguments

  - `md`: The model of [`ew_regression_series!`](@ref).
  - `st`: The state.
  - `y`: The values of the series.
  - `i`: The asset.

# Validation

  - The covariance numerator of the asset stays finite. Raises an [`IsNonFiniteError`](@ref), because a product of two finite returns can overflow.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`EWMoments`](@ref)
"""
function ew_regression_update!(md, st::EWBetaState, y::AbstractVector{<:Real},
                               i::Integer)::Nothing
    ew_model_update!(md, st.mo, y, i)
    ew_moments_update!(st.mo, y, i)
    ew_moments_update!(st.sq, y, i)
    @argcheck(all(isfinite, view(st.mo.C, :, :, i)),
              IsNonFiniteError("the exponentially weighted moments of asset $i are not finite, because the returns overflow the arithmetic. Rescale the returns."))
    return nothing
end
"""
    ew_regression_refresh!(md, st::EWBetaState, min_obs::Integer) -> nothing

Writes the beta of every asset from its moments, in place, through [`ew_model_beta`](@ref), which the model selects.

# Arguments

  - `md`: The model of [`ew_regression_series!`](@ref).
  - `st`: The state.
  - $(arg_dict[:min_obs])

# Returns

  - `nothing`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`ew_model_beta`](@ref)
"""
function ew_regression_refresh!(md, st::EWBetaState, min_obs::Integer)::Nothing
    for i in eachindex(st.b)
        st.b[i] = ew_model_beta(md, st.mo, i, min_obs)
    end
    return nothing
end
"""
    ew_model_reset!(::Any, ::Integer) -> nothing

Resets the part of the model of [`ew_regression_series!`](@ref) that belongs to one asset. The fallback is the market beta, whose model holds no part of an asset, so it resets nothing. The constant-market flags of an [`EWMacroSensitivity`](@ref) own a method.

# Arguments

  - `md`: The model.
  - `i`: The asset.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_reset!`](@ref)
  - [`EWMacroSensitivityState`](@ref)
"""
function ew_model_reset!(::Any, ::Integer)::Nothing
    return nothing
end
"""
    ew_model_update!(::Any, ::EWMoments, ::AbstractVector{<:Real}, ::Integer) -> nothing

Updates the part of the model of [`ew_regression_series!`](@ref) that belongs to one asset, before its moments take a new observation. The fallback is the market beta, whose model holds no part of an asset, so it updates nothing. The constant-market flags of an [`EWMacroSensitivity`](@ref) own a method.

# Arguments

  - `md`: The model.
  - `mo`: The moments, before the update.
  - `y`: The values of the series.
  - `i`: The asset.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_update!`](@ref)
  - [`EWMacroSensitivityState`](@ref)
"""
function ew_model_update!(::Any, ::EWMoments, ::AbstractVector{<:Real}, ::Integer)::Nothing
    return nothing
end
"""
    ew_model_beta(::Any, mo::EWMoments, i::Integer, min_obs::Integer) -> Real

Returns the beta that the model of [`ew_regression_series!`](@ref) writes for asset `i`. The fallback is the market beta, the slope of the regression of the asset on the market with an intercept, from its moments. The constant-market flags of an [`EWMacroSensitivity`](@ref) own a method, the partial beta of [`ew_macro_beta`](@ref).

# Mathematical definition

```math
\\begin{align}
\\beta_{t,i} &= \\frac{S^{mx}_{t,i}}{S^{mm}_{t,i}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:beta_ti_ewb])
  - ``S^{mx}_{t,i}``, ``S^{mm}_{t,i}``: Entries of the covariance numerator of [`EWMoments`](@ref), of the market with the asset and of the market with itself. Both sums run over the same observations with the same weights, so their weight sums cancel.

The market beta is `NaN` before the asset has `max(2, min_obs)` observations, where the market does not vary over them as [`ew_moments_vary`](@ref) states, and where the quotient is not finite.

# Arguments

  - `md`: The model.
  - `mo`: The moments.
  - `i`: The asset.
  - $(arg_dict[:min_obs])

# Returns

  - `b::Real`: The beta, or `NaN`.

# Related

  - [`ew_regression_refresh!`](@ref)
  - [`EWMoments`](@ref)
"""
function ew_model_beta(::Any, mo::EWMoments, i::Integer, min_obs::Integer)::Real
    K = size(mo.C, 1)
    b = mo.C[1, K, i] / mo.C[1, 1, i]
    ok = mo.n[i] >= max(2, min_obs) && ew_moments_vary(mo, 1, i) && isfinite(b)
    return ok ? b : oftype(b, NaN)
end
"""
    ew_beta_shrink_row!(::Any, ::EWBetaState, ::Integer) -> nothing

Shrinks the betas of a complete window where the model of [`ew_regression_series!`](@ref) asks for it. The fallback leaves the raw betas, and an [`EWBetaShrinkage`](@ref) owns a method.

# Arguments

  - `md`: The model.
  - `st`: The state.
  - `t`: The observation that closes the window.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`EWBetaShrinkage`](@ref)
"""
function ew_beta_shrink_row!(::Any, ::EWBetaState, ::Integer)::Nothing
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
 4. A vector of estimators returns the largest look-back of its members, and `nothing` when one member returns `nothing`. A [`CompositeExposure`](@ref) returns it over its Descriptors, and a [`FixedWeightedReturnForecast`](@ref), an [`ExpWeightedReturnForecast`](@ref) and a [`TargetReturnForecast`](@ref) over the Descriptors of their scores. An [`ObservedExposure`](@ref) returns the look-back of the member it wraps.
 5. A [`DerivedExposure`](@ref) returns one. The factor it is derived from is in the same factor list, so the look-back of the prior counts the rows of that factor.
 6. A [`CrossSectionalFactorPrior`](@ref) returns the look-back of its factors plus `lag`, because the regression of one observation reads the Factor Exposures `lag` observations earlier. The look-back of its Return Forecast Estimator replaces that sum when it is larger. `nothing` from either part gives `nothing`.
 7. Every other estimator returns `nothing`. This covers every exponentially weighted Descriptor, which is a recursion from the first observation.

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
public AbstractDescriptorEstimator, DescriptorStepResult, lookback
