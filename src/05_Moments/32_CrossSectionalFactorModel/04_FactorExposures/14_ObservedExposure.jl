"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Exposure Estimators of observed factors.

An observed factor is a factor whose return the caller observes, and which the Cross-Sectional Regression does not estimate. A Currency Factor is one: its return is the Currency Excess Return of its currency. A macro factor and an observed market factor are others. The member gives the exposure of every asset to the factor, as every Exposure Estimator does, and it names the column of the Exogenous Series `rd.E` that holds the return of each of its factors. A [`CrossSectionalFactorPrior`](@ref) that holds such a member in its factor list regresses the returns net of the observed factors on the other exposures, and then appends the observed factors to the factor model.

The kind is the declaration. So the Factor Family label of an observed factor is a property of its member, and no label is reserved. A family holds either estimated factors or observed ones, and never both, because the attribution of a family states a standard error only for factors the fit estimated.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractObservedExposureEstimator`, give it a `family` field that holds its Factor Family label, and implement the following methods:

## `factor_exposure`

  - `factor_exposure(xe::MyObservedExposure, rd::ReturnsResult) -> Array{<:Real}`: The Factor Exposure, `observations × assets` for one factor or `observations × assets × factors` for several, `NaN` wherever the active mask is `false`.

## `observed_series`

  - `observed_series(xe::MyObservedExposure, rd::ReturnsResult) -> Vector{String}`: The name of the column of `rd.E` that holds the return of each factor of the member, in column order.

### Arguments

  - `xe`: The concrete subtype instance.
  - `rd`: The returns data that carries the Asset Panel and the Exogenous Series.

A member that gives one factor takes the name the caller pairs with it, and a member that gives several names each factor by its series. [`exposure_axis_names`](@ref) states both, so the member needs no method of it.

# Related

  - [`AbstractExposureEstimator`](@ref)
  - [`ObservedExposure`](@ref)
  - [`CurrencyExposure`](@ref)
  - [`observed_series`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
abstract type AbstractObservedExposureEstimator <: AbstractExposureEstimator end
"""
    observed_series(xe::AbstractObservedExposureEstimator, rd::ReturnsResult) -> Vector{String}

Return the name of the column of the Exogenous Series that holds the return of each factor of an observed member, in column order.

This is the verb that makes a member observed. [`CrossSectionalFactorPrior`](@ref) selects the returns of the observed factors from `rd.E` by these names.

# Arguments

  - `xe`: Observed Exposure Estimator.
  - $(arg_dict[:rd]) The currency method reads the levels of its Panel Field off the Asset Panel.

# Returns

  - `ne::Vector{String}`: One series name per factor of the member.

# Related

  - [`AbstractObservedExposureEstimator`](@ref)
  - [`ObservedExposure`](@ref)
  - [`CurrencyExposure`](@ref)
"""
function observed_series end
function exposure_axis_names(nm::AbstractString, xe::AbstractObservedExposureEstimator,
                             rd::ReturnsResult)
    s = observed_series(xe, rd)
    n = isone(length(s)) ? [String(nm)] : s
    return n, fill(String(xe.family), length(n))
end
"""
$(DocStringExtensions.TYPEDEF)

An observed factor whose exposure another Exposure Estimator gives, and whose return is one named column of the Exogenous Series.

The member wraps an estimated member that gives one factor, for example a [`CompositeExposure`](@ref) over a Descriptor of the sensitivity of each asset to a macro series, or of its beta to the market. The wrapped member gives the exposure, and the column `series` of `rd.E` gives the return of the factor. The Cross-Sectional Regression does not estimate that return, so the observed series enters the factor model as it is.

Inside a [`CrossSectionalFactorPrior`](@ref), the wrapped member reads the returns net of the observed members that read no returns, such as the Currency Factors of a [`CurrencyExposure`](@ref). So a Descriptor of the returns measures the local move of an asset, and not the currency it holds. It never reads returns net of its own factor, as [`cross_sectional_observed`](@ref) states.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ObservedExposure(; xe::AbstractExposureEstimator, series::AbstractString,
                     family::AbstractString = xe.family) -> ObservedExposure

Keywords correspond to the struct's fields.

## Validation

  - `xe` is neither an observed member nor a [`DerivedExposure`](@ref), which reads the exposure of another factor of the list.
  - `!isempty(series)` and `!isempty(family)`.

# Examples

```jldoctest
julia> ObservedExposure(; xe = ConstantExposure(), series = \"SPX\")
ObservedExposure
      xe ┼ ConstantExposure
         │   family ┴ String: \"market\"
  series ┼ String: \"SPX\"
  family ┴ String: \"market\"
```

# Related

  - [`AbstractObservedExposureEstimator`](@ref)
  - [`CurrencyExposure`](@ref)
  - [`observed_series`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
@concrete struct ObservedExposure <: AbstractObservedExposureEstimator
    """
    Exposure Estimator that gives the exposure of every asset to the factor. It gives one factor.
    """
    xe
    """
    Name of the column of the Exogenous Series that holds the return of the factor.
    """
    series
    """
    Label of the Factor Family the observed factor belongs to.
    """
    family
    function ObservedExposure(xe::AbstractExposureEstimator, series::AbstractString,
                              family::AbstractString)
        @argcheck(!isa(xe, Union{<:AbstractObservedExposureEstimator, <:DerivedExposure}),
                  ArgumentError("an ObservedExposure wraps an estimated Exposure Estimator that reads the Asset Panel, and got a $(nameof(typeof(xe))). An observed member is observed already, and a DerivedExposure reads the exposure of another factor of the list."))
        @argcheck(!isempty(series),
                  IsEmptyError("series names a column of the Exogenous Series, so it cannot be the empty string"))
        assert_exposure_family(family)
        return new{typeof(xe), typeof(series), typeof(family)}(xe, series, family)
    end
end
function ObservedExposure(; xe::AbstractExposureEstimator, series::AbstractString,
                          family::AbstractString = xe.family)::ObservedExposure
    return ObservedExposure(xe, series, family)
end
"""
    factor_exposure(xe::ObservedExposure, rd::ReturnsResult) -> Matrix{<:Real}

Compute the Factor Exposure of an observed factor, through the Exposure Estimator it wraps.

# Arguments

  - `xe`: Observed Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The wrapped member gives one factor, so its exposure is a matrix. Raises a `DimensionMismatch`.
  - The rules of the wrapped member.

# Returns

  - `L::Matrix{<:Real}`: The Factor Exposure, `observations × assets`.

# Related

  - [`ObservedExposure`](@ref)
  - [`factor_exposure`](@ref)
"""
function factor_exposure(xe::ObservedExposure, rd::ReturnsResult)::Matrix{<:Real}
    L = factor_exposure(xe.xe, rd)
    @argcheck(ndims(L) == 2,
              DimensionMismatch("an ObservedExposure reads one column of the Exogenous Series, \"$(xe.series)\", so the member it wraps must give one factor, and a $(nameof(typeof(xe.xe))) gave $(size(L, 3))"))
    return L
end
function observed_series(xe::ObservedExposure, ::ReturnsResult)::Vector{String}
    return [String(xe.series)]
end
function lookback(xe::ObservedExposure)::Option{<:Integer}
    return lookback(xe.xe)
end
function carry_lookback(xe::ObservedExposure)::Option{<:Integer}
    return carry_lookback(xe.xe)
end
function descriptor_carry(xe::ObservedExposure, rd::ReturnsResult, m::Integer)
    c = descriptor_carry(xe.xe, rd, m)
    return (; xf = ObservedExposure(c.xf, xe.series, xe.family),
            xv = ObservedExposure(c.xv, xe.series, xe.family))
end
"""
$(DocStringExtensions.TYPEDEF)

The observed one-hot exposure of the currency of each asset, which gives the Currency Factors of a [`CrossSectionalFactorPrior`](@ref).

A universe held across currencies earns, in the base currency, the return of each exchange rate beside the local return of each asset. No style or classification factor explains that return. The member reads a categorical Panel Field that names the currency of each asset at each observation, and gives one factor per currency: an asset carries a one on the currency it is denominated in and a zero on every other currency. The block is the one [`OneHotExposure`](@ref) gives on the same field.

The factors are observed. The return of each is the Currency Excess Return of its currency, which the prior reads from the Exogenous Series `rd.E` by the currency level, for example `\"USD\"`. [`currency_excess_index`](@ref) builds the level series that [`prices_to_returns`](@ref) converts to those returns.

The base currency earns no Currency Excess Return, so its index is constant and a factor of it would have no variance. `base` names it. The member then gives the base currency no factor, and an asset of the base currency carries a zero exposure on every Currency Factor, so its local return is its base-currency return. No rule refuses a factor of the base currency, but its zero variance makes the default covariance of the factor prior throw, because it divides by a zero volatility.

# Mathematical definition

```math
\\begin{align}
x^{\\mathrm{ccy}}_{t,\\,i,\\,c} &= \\begin{cases}
1 & C_{t,\\,i} = c\\,, \\\\
0 & C_{t,\\,i} \\neq c\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``x^{\\mathrm{ccy}}_{t,\\,i,\\,c}``: Exposure of asset ``i`` to the Currency Factor ``c`` at observation ``t``.
  - ``C_{t,\\,i}``: Currency of asset ``i`` at observation ``t``, the level of the Panel Field.

The currencies ``c`` are the levels of the Panel Field, less `base` when it is stated.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CurrencyExposure(; field::AbstractString = \"currency\",
                     family::AbstractString = \"currency\",
                     base::Option{<:AbstractString} = nothing) -> CurrencyExposure

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(field)` and `!isempty(family)`.
  - `base`, when it is stated, is not empty.

# Examples

```jldoctest
julia> CurrencyExposure(; base = \"USD\")
CurrencyExposure
   field ┼ String: \"currency\"
  family ┼ String: \"currency\"
    base ┴ String: \"USD\"
```

# Related

  - [`AbstractObservedExposureEstimator`](@ref)
  - [`OneHotExposure`](@ref)
  - [`ObservedExposure`](@ref)
  - [`observed_series`](@ref)
  - [`currency_excess_index`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
@concrete struct CurrencyExposure <: AbstractObservedExposureEstimator
    """
    Name of the categorical Panel Field that states the currency of each asset at each observation. Each of its levels names a currency, and the column of the Exogenous Series that holds its Currency Excess Return.
    """
    field
    """
    Label of the Factor Family the Currency Factors belong to.
    """
    family
    """
    Level of the base currency, which takes no factor, or `nothing` to give every level a factor. An asset of the base currency carries a zero exposure on every Currency Factor.
    """
    base
    function CurrencyExposure(field::AbstractString, family::AbstractString,
                              base::Option{<:AbstractString})
        assert_panel_terms(field, :field)
        assert_exposure_family(family)
        if !isnothing(base)
            assert_panel_terms(base, :base)
        end
        return new{typeof(field), typeof(family), typeof(base)}(field, family, base)
    end
end
function CurrencyExposure(; field::AbstractString = "currency",
                          family::AbstractString = "currency",
                          base::Option{<:AbstractString} = nothing)::CurrencyExposure
    return CurrencyExposure(field, family, base)
end
"""
    factor_exposure(xe::CurrencyExposure, rd::ReturnsResult) -> Array{<:Real, 3}

Compute the one-hot Factor Exposure of the currency of each asset, one factor per currency.

The block is the one [`OneHotExposure`](@ref) gives on the same Panel Field, so every rule of that member holds: a cell the Panel Field does not observe, a cell that sets no level and a cell the active mask does not activate carry `NaN` on every level. The column of the base currency is dropped, so an asset of the base currency carries a zero on every column left.

# Algorithm

 1. Compute the one-hot block with [`factor_exposure`](@ref) of a [`OneHotExposure`](@ref) on `xe.field`.
 2. Keep the columns of the levels that [`currency_level_columns`](@ref) keeps.

# Arguments

  - `xe`: Currency Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`one_hot_field`](@ref) and of [`currency_level_columns`](@ref).

# Returns

  - `L::Array{<:Real, 3}`: The Factor Exposure, `observations × assets × currencies`, in the level order the Panel Field declares, less the base currency.

# Examples

```jldoctest
julia> pnl = asset_panel([CategoricalPanelInput(; name = \"currency\",
                                                vals = [\"USD\" \"EUR\"; \"USD\" \"EUR\"])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> factor_exposure(CurrencyExposure(), rd)[1, :, :]
2×2 Matrix{Float64}:
 0.0  1.0
 1.0  0.0

julia> PortfolioOptimisers.observed_series(CurrencyExposure(), rd)
2-element Vector{String}:
 \"EUR\"
 \"USD\"

julia> factor_exposure(CurrencyExposure(; base = \"USD\"), rd)[1, :, :]
2×1 Matrix{Float64}:
 0.0
 1.0
```

# Related

  - [`CurrencyExposure`](@ref)
  - [`OneHotExposure`](@ref)
  - [`observed_series`](@ref)
  - [`currency_level_columns`](@ref)
"""
function factor_exposure(xe::CurrencyExposure, rd::ReturnsResult)::Array{<:Real, 3}
    B = factor_exposure(OneHotExposure(xe.field, xe.family), rd)
    return B[:, :, currency_level_columns(xe, rd)]
end
function observed_series(xe::CurrencyExposure, rd::ReturnsResult)::Vector{String}
    return String.(one_hot_field(rd, xe.field).levels[currency_level_columns(xe, rd)])
end
function lookback(::CurrencyExposure)::Integer
    return 1
end
"""
    currency_level_columns(xe::CurrencyExposure, rd::ReturnsResult) -> Vector{Int}

Return the levels of the Panel Field of a [`CurrencyExposure`](@ref) that take a Currency Factor: every level but the base currency.

# Arguments

  - `xe`: Currency Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`one_hot_field`](@ref).
  - `xe.base`, when it is stated, is a level of the Panel Field. Raises an `ArgumentError`.

# Returns

  - `idx::Vector{Int}`: The positions of the levels that take a factor, in level order.

# Related

  - [`CurrencyExposure`](@ref)
  - [`observed_series`](@ref)
"""
function currency_level_columns(xe::CurrencyExposure, rd::ReturnsResult)::Vector{Int}
    lv = one_hot_field(rd, xe.field).levels
    if isnothing(xe.base)
        return collect(eachindex(lv))
    end
    @argcheck(xe.base in lv,
              ArgumentError("the base currency \"$(xe.base)\" is not a level of the Panel Field \"$(xe.field)\", whose levels are $lv"))
    return findall(!=(xe.base), lv)
end
"""
    observed_reads_returns(xe::AbstractObservedExposureEstimator) -> Bool
    observed_reads_returns(xe::CurrencyExposure) -> Bool

Return whether the exposures of an observed member can read the asset returns.

A [`CrossSectionalFactorPrior`](@ref) reads its observed members in two stages, with [`cross_sectional_observed`](@ref). The members that read no returns come first, and the returns net of their factors are derived from their exposures. The members that can read returns then read those net returns. A member that reads returns thus never enters the net returns it reads.

The answer is `true` unless the member states otherwise. That is the safe side: a member that reads no returns but answers `true` only reads returns net of fewer observed factors, and it gives the same exposure. [`ObservedExposure`](@ref) answers `true`, because the member it wraps can hold a Descriptor of the returns. [`CurrencyExposure`](@ref) answers `false`, because it reads only a categorical Panel Field.

# Arguments

  - `xe`: Observed Exposure Estimator.

# Returns

  - `flag::Bool`: `true` if the exposures of the member can read the asset returns.

# Related

  - [`AbstractObservedExposureEstimator`](@ref)
  - [`cross_sectional_observed`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function observed_reads_returns(::AbstractObservedExposureEstimator)
    return true
end
function observed_reads_returns(::CurrencyExposure)
    return false
end

"""
    cross_sectional_factor_partition(factors::AbstractVector{<:Pair}) -> NamedTuple

Split the factor list of a [`CrossSectionalFactorPrior`](@ref) into the factors the fit estimates and the factors it observes.

A member of [`AbstractObservedExposureEstimator`](@ref) declares that its factors are observed, and every other member gives estimated factors. The fit puts the observed factors after the estimated ones on every axis, so the partition keeps the order the caller wrote within each part.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.

# Returns

  - `est::Vector{<:Pair}`: The Pairs of the estimated factors, in the order of `factors`.
  - `obs::Vector{<:Pair}`: The Pairs of the observed factors, in the order of `factors`.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`AbstractObservedExposureEstimator`](@ref)
  - [`cross_sectional_observed`](@ref)
"""
function cross_sectional_factor_partition(factors::AbstractVector{<:Pair})
    isobs = [isa(last(p), AbstractObservedExposureEstimator) for p in factors]
    return (; est = factors[.!isobs], obs = factors[isobs])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Join the two parts of the factor list of a [`CrossSectionalFactorPrior`](@ref) that [`cross_sectional_factor_partition`](@ref) split, in the order of the list.

The carry fold folds the members of each part, and keeps the folded list in its state, so a later step partitions it again.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`, the list that was split.
  - `est`: The Pairs of the estimated factors, in the order of `factors`.
  - `obs`: The Pairs of the observed factors, in the order of `factors`.

# Returns

  - `factors::Vector{<:Pair}`: The Pairs of `est` and `obs`, each one at the place of its member in `factors`.

# Related

  - [`cross_sectional_factor_partition`](@ref)
  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_factor_join(factors::AbstractVector{<:Pair},
                                     est::AbstractVector{<:Pair},
                                     obs::AbstractVector{<:Pair})
    isobs = [isa(last(p), AbstractObservedExposureEstimator) for p in factors]
    io = cumsum(isobs)
    ie = cumsum(.!isobs)
    return [isobs[i] ? obs[io[i]] : est[ie[i]] for i in eachindex(factors)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuse a Factor Family of a [`CrossSectionalFactorPrior`](@ref) that holds both estimated and observed factors, or that a constrained family names although it is observed.

The family label of an observed factor is a property of its member, so no label is reserved. A family is still one kind or the other: the attribution of a family states a standard error only for factors the fit estimated, and the zero-sum condition of a constrained family reaches only the regression. So a label that an estimated member and an observed member both carry is refused, and so is a constrained family whose label an observed member carries.

# Arguments

  - `factors`: Pairs of `factor name => Exposure Estimator`.
  - `families`: Pairs of `family label => dropped member`, or `nothing`.

# Validation

  - No family label belongs to an estimated member and to an observed member. Raises an `ArgumentError`.
  - No key of `families` is the family label of an observed member. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_factor_partition`](@ref)
  - [`AbstractObservedExposureEstimator`](@ref)
"""
function assert_cross_sectional_observed_families(factors::AbstractVector{<:Pair},
                                                  families::Option{<:AbstractVector{<:Pair}})::Nothing
    (; est, obs) = cross_sectional_factor_partition(factors)
    fo = unique(String[String(last(p).family) for p in obs])
    both = intersect(fo, String[String(last(p).family) for p in est])
    @argcheck(isempty(both),
              ArgumentError("the Factor Families $both hold both estimated and observed factors. A family is estimated or observed, because its attribution states a standard error only for the factors the fit estimates. Give the observed members, or the estimated ones, another family."))
    if !isnothing(families)
        bad = intersect(fo, String[String(first(p)) for p in families])
        @argcheck(isempty(bad),
                  ArgumentError("families cannot constrain the observed Factor Families $bad: their returns are observed, and the zero-sum condition of a constrained family reaches only the factors the regression estimates"))
    end
    return nothing
end
"""
    cross_sectional_observed(obs::AbstractVector{<:Pair}, rd::ReturnsResult, lag::Integer,
                             kept::Option{<:NamedTuple} = nothing, g0::Integer = 0,
                             n::Option{<:Integer} = nothing) -> Option{<:NamedTuple}

Read the observed factors of a [`CrossSectionalFactorPrior`](@ref) off the returns data: their exposures, their names, their families and their returns.

A prior with no observed factor gets `nothing`. Otherwise each member gives its exposures and names, with [`factor_exposure`](@ref) and [`exposure_axis_names`](@ref), and the column of the Exogenous Series of each of its factors, with [`observed_series`](@ref).

The members are read in two stages, as [`observed_reads_returns`](@ref) sorts them. The members that read no returns, such as [`CurrencyExposure`](@ref), read `rd`. The members that can read returns read a copy of `rd` whose `X` is net of the factors of the first stage, derived with [`cross_sectional_local_returns`](@ref). So a Descriptor of the returns inside an [`ObservedExposure`](@ref) measures the local move of an asset, as the estimated members do, and never the net returns its own factor enters. The copy is derived also when `lx` names a Panel Field of net returns, because that field is net of every observed factor, the member's own among them. When one stage is empty, every member reads `rd`.

The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads the rows that it carries and the rows of a step. It gives the derived rows of the carried rows in `kept`, and the number of observations before the first row of `rd` in `g0`. So the function derives the rows of the step alone, as the batch fit derives them. The carry gives the number of rows of the step in `n` too, and the members as it folded them in `obs`. Each member folds the rows of the step into its Descriptors that carry a state, with [`descriptor_carry`](@ref), and computes the exposures of those rows alone, from the last rows of `rd` that [`cross_sectional_exposure_rows`](@ref) names. The derivation of a row reads the exposures of the row `lag` observations before it, and a Descriptor that carries a state cannot give a row before the step again. So `kept` holds the observed exposures of the last `lag` rows before the step too, and the exposures of the function are those rows followed by the rows of the step.

# Algorithm

 1. `obs` is empty: return `nothing`.
 2. Read the members for which [`observed_reads_returns`](@ref) answers `false` on `rd`, with [`cross_sectional_observed_member`](@ref). When `n` is an integer, fold the rows of the step into each member with [`descriptor_carry`](@ref), compute the exposures of those rows alone with [`cross_sectional_observed_rows`](@ref), and put the columns of the member in `kept.Zo` before them.
 3. If a member answers `true` and a member answers `false`, stack the members of step 2 with [`cross_sectional_observed_stack`](@ref), and derive the returns net of their factors with [`cross_sectional_local_returns`](@ref). Copy `rd` with those returns in `X`.
 4. Read the members for which [`observed_reads_returns`](@ref) answers `true` on the copy, or on `rd` when step 3 made no copy, as step 2 reads them.
 5. Stack every member, in the order of `obs`, with [`cross_sectional_observed_stack`](@ref).

# Arguments

  - `obs`: The Pairs of the observed factors, from [`cross_sectional_factor_partition`](@ref). The carry fold gives them as it folded them.
  - $(arg_dict[:rd]) It carries the Asset Panel and the Exogenous Series.
  - `lag`: Number of observations by which the exposures lag the returns.
  - `kept`: The derived series that a carry fold carries, `(; Xn, Xl, Zo)` as the field `der` of a [`CrossSectionalCarryState`](@ref) holds them, or `nothing`. `Xn` holds the returns net of the first stage at the first rows of `rd`, and the function derives the rows after them. `Zo` holds the observed exposures of the last `lag` rows before the step.
  - `g0`: Number of observations before the first row of `rd`.
  - `n`: Number of last rows of `rd` whose exposures to compute, or `nothing` for every row. The carry fold gives the number of rows of a step.

# Validation

  - Each member names one series per factor. Raises a `DimensionMismatch`.
  - The rules of [`cross_sectional_observed_stack`](@ref).
  - The rules of [`factor_exposure`](@ref) and [`observed_series`](@ref) of each member.

# Returns

  - `cc::Option{<:NamedTuple}`: `nothing`, or `(; Z, R, lv, nf, fam, Xn, xf)`: the exposures `observations × assets × factors` at the last rows of `rd`, or at every row when `n` is `nothing`; the observed returns `observations × factors` over every observation of `rd`; the series names; the factor names; the family labels; the returns net of the first stage that the second stage read, or `nothing` when step 3 derived none; and the Pairs of `obs` with the state of each Descriptor after the rows of the step.

# Related

  - [`AbstractObservedExposureEstimator`](@ref)
  - [`observed_reads_returns`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_local_returns`](@ref)
  - [`cross_sectional_observed_block`](@ref)
  - [`cross_sectional_observed_rows`](@ref)
  - [`ReturnsResult`](@ref)
"""
function cross_sectional_observed(obs::AbstractVector{<:Pair}, rd::ReturnsResult,
                                  lag::Integer, kept::Option{<:NamedTuple} = nothing,
                                  g0::Integer = 0, n::Option{<:Integer} = nothing)
    if isempty(obs)
        return nothing
    end
    srs = [observed_series(last(p), rd) for p in obs]
    # The carried exposures of the last `lag` rows hold the factors in the order of `obs`.
    col = cumsum(vcat(0, length.(srs)))
    kx = something(kept, (; Xn = nothing, Zo = nothing))
    function member(k, r)
        return cross_sectional_observed_member(obs[k], r, n, srs[k], kx.Zo,
                                               (col[k] + 1):col[k + 1])
    end
    rr = [observed_reads_returns(last(p)) for p in obs]
    ms = Vector{Any}(undef, length(obs))
    for k in findall(!, rr)
        ms[k] = member(k, rd)
    end
    # The members that can read returns read the returns net of the members that read none,
    # so a Descriptor of the returns measures the local move of an asset, and no member reads
    # net returns that its own factor enters.
    Xn = nothing
    rdn = rd
    if any(rr) && !all(rr)
        Xn = cross_sectional_local_returns(nothing,
                                           cross_sectional_observed_stack(ms[.!rr], rd), rd.X,
                                           rd, lag, kx.Xn, g0)
        rdn = ReturnsResult(; nx = rd.nx, X = Xn, nf = rd.nf, F = rd.F, nb = rd.nb,
                            B = rd.B, ne = rd.ne, E = rd.E, ts = rd.ts, iv = rd.iv,
                            ivpa = rd.ivpa, pnl = rd.pnl)
    end
    for k in findall(rr)
        ms[k] = member(k, rdn)
    end
    return merge(cross_sectional_observed_stack(ms, rd),
                 (; Xn = Xn, xf = [m.xf for m in ms]))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read one observed member of a [`CrossSectionalFactorPrior`](@ref) for [`cross_sectional_observed`](@ref): its exposures, its names and its family labels.

When `n` is an integer, the member folds the rows of the step into its Descriptors that carry a state with [`descriptor_carry`](@ref), and computes the exposures of those rows alone with [`cross_sectional_observed_rows`](@ref). The carried exposures of the last `lag` rows then come before them.

# Arguments

  - `p`: The Pair `factor name => observed member`.
  - $(arg_dict[:rd])
  - `n`: Number of last rows of `rd` whose exposures to compute, or `nothing` for every row.
  - `sr`: The names of the series of the member, from [`observed_series`](@ref).
  - `Zo`: The observed exposures of every member at the last `lag` rows before the step, or `nothing`.
  - `cols`: The columns of the member in `Zo`.

# Validation

  - The member names one series per factor. Raises a `DimensionMismatch`.

# Returns

  - `m::NamedTuple`: `(; Z, lv, nf, fam, xf)`: the exposures `observations × assets × factors`, the series names, the factor names, the family labels, and the Pair `p` with the state of each Descriptor after the rows of the step.

# Related

  - [`cross_sectional_observed`](@ref)
  - [`cross_sectional_observed_rows`](@ref)
  - [`descriptor_carry`](@ref)
"""
function cross_sectional_observed_member(p::Pair, rd::ReturnsResult, n::Option{<:Integer},
                                         sr::AbstractVector{<:AbstractString},
                                         Zo::Option{<:AbstractArray{<:Any, 3}},
                                         cols::AbstractVector{<:Integer})
    (key, xe) = p
    # The generic method of `descriptor_carry` returns the member twice for `n = nothing`.
    c = descriptor_carry(xe, rd, n)
    Z = cross_sectional_observed_rows(c.xv, rd, n)
    nm, f = exposure_axis_names(String(key), xe, rd)
    @argcheck(length(sr) == length(nm) == size(Z, 3),
              DimensionMismatch("the observed factor \"$key\" gives $(size(Z, 3)) exposures and $(length(nm)) names, and names $(length(sr)) series of the Exogenous Series; each factor reads one series"))
    if !isnothing(Zo)
        Z = cat(Zo[:, :, cols], Z; dims = 1)
    end
    return (; Z = Z, lv = sr, nf = nm, fam = f, xf = key => c.xf)
end
"""
    cross_sectional_observed_rows(xe::AbstractObservedExposureEstimator, rd::ReturnsResult,
                                  n::Nothing) -> Array{<:Real, 3}
    cross_sectional_observed_rows(xe::AbstractObservedExposureEstimator, rd::ReturnsResult,
                                  n::Integer) -> Array{<:Real, 3}

Compute the exposures of an observed member at every row of a [`ReturnsResult`](@ref), or at its last `n` rows alone, with a factor axis.

The batch fit gives `n = nothing`, and the member reads every row. The carry fold of a [`CrossSectionalFactorPrior`](@ref) gives the number of rows of a step, and the member reads the last rows that [`cross_sectional_exposure_rows`](@ref) names, as an estimated member does. Those rows give the exposures of the batch fit at the rows of the step.

# Arguments

  - `xe`: Observed Exposure Estimator. On the carry fold, a Descriptor of it that carries a state is the [`CarriedDescriptor`](@ref) of the step, as [`descriptor_carry`](@ref) gives it.
  - $(arg_dict[:rd])
  - `n`: Number of last rows, or `nothing` for every row.

# Validation

  - The rules of [`factor_exposure`](@ref) of the member.

# Returns

  - `Z::Array{<:Real, 3}`: The exposures, `observations × assets × factors`.

# Related

  - [`cross_sectional_observed`](@ref)
  - [`cross_sectional_exposure_rows`](@ref)
"""
function cross_sectional_observed_rows(xe::AbstractObservedExposureEstimator,
                                       rd::ReturnsResult, ::Nothing)::Array{<:Real, 3}
    A = factor_exposure(xe, rd)
    return ndims(A) == 2 ? reshape(A, size(A, 1), size(A, 2), 1) : A
end
function cross_sectional_observed_rows(xe::AbstractObservedExposureEstimator,
                                       rd::ReturnsResult, n::Integer)::Array{<:Real, 3}
    (; rdi, k) = cross_sectional_exposure_rows(xe, rd, n)
    A = cross_sectional_observed_rows(xe, rdi, nothing)
    return k == n ? A : A[(k - n + 1):k, :, :]
end
"""
    cross_sectional_observed_carry(cc::Nothing, Xl, n::Integer, lag::Integer) -> NamedTuple
    cross_sectional_observed_carry(cc::NamedTuple, Xl::MatNum, n::Integer, lag::Integer)
        -> NamedTuple

Split the observed factors of a step of the carry fold of a [`CrossSectionalFactorPrior`](@ref) into the exposures of the rows of the step and the derived series that the state carries.

On the carry fold, [`cross_sectional_observed`](@ref) gives the exposures of the rows of the step and of the `lag` rows before them. The fit of the step reads the rows of the step. The derivation of the next step reads the exposures of the last `lag` rows, and a Descriptor that carries a state cannot give them again, so the state carries them in `Zo`.

# Arguments

  - `cc`: The observed factors that [`cross_sectional_observed`](@ref) read, or `nothing`.
  - `Xl`: The returns net of every observed factor, at every row of the window of the step.
  - `n`: Number of rows of the step.
  - `lag`: Number of observations by which the exposures lag the returns.

# Returns

  - `carry::NamedTuple`: `Zn`, the observed exposures of the `n` rows of the step, and `der`, the derived series `(; Xn, Xl, Zo)` that the field `der` of a [`CrossSectionalCarryState`](@ref) holds. Both are `nothing` without an observed factor.

# Related

  - [`cross_sectional_observed`](@ref)
  - [`cross_sectional_fold_rows`](@ref)
  - [`CrossSectionalCarryState`](@ref)
"""
function cross_sectional_observed_carry(::Nothing, ::Any, ::Integer, ::Integer)
    return (; Zn = nothing, der = nothing)
end
function cross_sectional_observed_carry(cc::NamedTuple, Xl::MatNum, n::Integer,
                                        lag::Integer)
    T = size(cc.Z, 1)
    return (; Zn = cc.Z[(T - n + 1):T, :, :],
            der = (; Xn = cc.Xn, Xl = Xl, Zo = cc.Z[(T - min(lag, T) + 1):T, :, :]))
end
"""
    cross_sectional_observed_stack(ms::AbstractVector, rd::ReturnsResult) -> NamedTuple

Stack the observed members that [`cross_sectional_observed`](@ref) read on the factor axis, and select the return of each factor from the Exogenous Series by name.

The function selects the columns of the series from `rd.E` and ignores every other column. It refuses a series with no column, even when no asset loads on the factor, because the factor enters the factor covariance with zero loadings and its return must still be a number.

# Arguments

  - `ms`: One NamedTuple `(; Z, lv, nf, fam)` for each member: its exposures `observations × assets × factors`, its series names, its factor names and its family labels.
  - $(arg_dict[:rd]) It carries the Exogenous Series.

# Validation

  - `rd.E` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - `rd.ne` names every series. Raises an `ArgumentError` that names the missing ones.

# Returns

  - `cc::NamedTuple`: `(; Z, R, lv, nf, fam)`: the exposures `observations × assets × factors`, the observed returns `observations × factors` over every observation of `rd`, the series names, the factor names and the family labels, in the order of `ms`.

# Related

  - [`cross_sectional_observed`](@ref)
  - [`cross_sectional_local_returns`](@ref)
"""
function cross_sectional_observed_stack(ms::AbstractVector, rd::ReturnsResult)
    lv = String[]
    nf = String[]
    fam = String[]
    for m in ms
        append!(lv, m.lv)
        append!(nf, m.nf)
        append!(fam, m.fam)
    end
    @argcheck(!isnothing(rd.E),
              IsNothingError("the observed factors read their returns from the Exogenous Series of the returns data, and rd.E is nothing. Give the PricesResult or the ReturnsResult an E with one column per series: $lv. currency_excess_index builds the levels of the Currency Factors."))
    j = indexin(lv, rd.ne)
    miss = unique(lv[isnothing.(j)])
    @argcheck(isempty(miss),
              ArgumentError("the observed factors read their returns from the columns of the Exogenous Series with these names, and rd.ne has no column for $miss. Every series needs a column, even one no asset loads on, because its factor enters the factor covariance. Got rd.ne => $(rd.ne)"))
    return (; Z = reduce((a, b) -> cat(a, b; dims = 3), [m.Z for m in ms]),
            R = rd.E[:, Int.(j)], lv = lv, nf = nf, fam = fam)
end
"""
    cross_sectional_local_returns(lx::Nothing, cc::Nothing, X::MatNum, rd::ReturnsResult,
                                  lag::Integer, kept::Option{<:MatNum} = nothing,
                                  g0::Integer = 0) -> MatNum
    cross_sectional_local_returns(lx::Nothing, cc::NamedTuple, X::MatNum, rd::ReturnsResult,
                                  lag::Integer, kept::Option{<:MatNum} = nothing,
                                  g0::Integer = 0) -> Matrix{<:Real}
    cross_sectional_local_returns(lx::AbstractString, cc::NamedTuple, X::MatNum,
                                  rd::ReturnsResult, lag::Integer,
                                  kept::Option{<:MatNum} = nothing,
                                  g0::Integer = 0) -> Matrix{<:Real}

Return the returns a Cross-Sectional Factor Prior regresses on its estimated exposures: the returns net of its observed factors.

The observed factors carry the part of each return that their observed returns explain, so the regression reads what is left. A prior with no observed factor regresses on `X` itself. A prior whose `lx` names a Panel Field reads the net returns off that field, so a caller's own measure enters the fit unchanged. Otherwise the prior derives them from `X` and the observed returns. Under Currency Factors the net returns are the local returns of the assets, and `X` stays in the base currency.

# Mathematical definition

```math
\\begin{align}
x^{\\mathrm{net}}_{t,\\,i} &= x_{t,\\,i} - \\sum_{c \\,:\\, z_{s,\\,i,\\,c} \\neq 0} z_{s,\\,i,\\,c} \\, r_{t,\\,c}\\,, \\\\
s &= \\begin{cases} t - \\ell & t > \\ell\\,, \\\\ t & t \\leq \\ell\\,. \\end{cases}
\\end{align}
```

Where:

  - ``x^{\\mathrm{net}}_{t,\\,i}``: Return of asset ``i`` at observation ``t`` net of the observed factors.
  - $(math_dict[:x_ti_ret])
  - ``z_{t,\\,i,\\,c}``: Exposure of asset ``i`` to the observed factor ``c`` at observation ``t``.
  - ``r_{t,\\,c}``: Observed return of the factor ``c`` at observation ``t``.
  - ``s``: The observation whose exposure the derivation removes.
  - $(math_dict[:ell_lag_cs])

The index ``t`` counts the observations from the first observation of the sample.

The exposures lag the returns, so the derivation removes each observed return through the exposure of ``t - \\ell``, which is the exposure the model loads for observation ``t``. The lagged combined exposures then reproduce `X` exactly. The sum runs over the non-zero exposures, so a non-finite return of a currency the asset does not hold does not reach it. An asset whose exposure is `NaN`, because it is inactive or carries no currency label, gets a `NaN` net return, and it leaves the regression of that observation. The first ``\\ell`` observations have no lagged exposure, so the derivation takes the exposure of the same observation there. The fit never regresses them, but the estimated members read them: a Descriptor of the returns reads the net returns, and a row with no finite return leaves the market return undefined.

Under Currency Factors and simple returns the identity of the local return is not exact: the base-currency return of an asset also holds the cross term of its local return and the exchange-rate return, see [`currency_excess_index`](@ref), and the derived net return keeps it.

The carry fold of a [`CrossSectionalFactorPrior`](@ref) gives the rows that it carries and the rows of a step. A window that derived its first rows again would take the exposure of the same row there, and the batch fit took the exposure of the row ``\\ell`` observations before. So the carry gives the derived rows of its carried rows in `kept`, and the number of observations before the first row of `X` in `g0`. The function copies `kept` and derives the rows after it. The exposures `cc.Z` of the carry hold the last rows of `X` alone, the rows of the step and the `lag` rows before them, and the function reads each row of `cc.Z` at its row of `X`.

# Arguments

  - `lx`: The name of the Panel Field of net returns, or `nothing`.
  - `cc`: The observed factors [`cross_sectional_observed`](@ref) read, or `nothing`.
  - `X`: Asset returns, `observations × assets`.
  - $(arg_dict[:rd]) The named method reads the Panel Field off its Asset Panel.
  - `lag`: Number of observations by which the exposures lag the returns.
  - `kept`: The derived rows of the first rows of `X`, or `nothing`. Only the derived method reads it, because a Panel Field gives each row from its own row.
  - `g0`: Number of observations before the first row of `X`.

# Validation

  - The rules of [`descriptor_field_values`](@ref) for the named method.

# Returns

  - `Xl::MatNum`: The returns the regression reads, `observations × assets`.

# Related

  - [`CrossSectionalFactorPrior`](@ref)
  - [`cross_sectional_observed`](@ref)
  - [`currency_excess_index`](@ref)
"""
function cross_sectional_local_returns(::Nothing, ::Nothing, X::MatNum, ::ReturnsResult,
                                       ::Integer, ::Option{<:MatNum} = nothing,
                                       ::Integer = 0)
    return X
end
function cross_sectional_local_returns(::Nothing, cc::NamedTuple, X::MatNum,
                                       ::ReturnsResult, lag::Integer,
                                       kept::Option{<:MatNum} = nothing, g0::Integer = 0)
    (; Z, R) = cc
    Xl = similar(X, promote_type(eltype(X), eltype(Z), eltype(R)))
    # The carried rows are copied, and an absent `kept` copies zero rows.
    K = something(kept, view(X, 1:0, :))
    k = size(K, 1)
    Xl[1:k, :] .= K
    # On the carry fold `Z` holds the last rows of `X` alone.
    zo = size(X, 1) - size(Z, 1)
    for t in (k + 1):size(X, 1), i in axes(X, 2)
        x = X[t, i]
        s = g0 + t > lag ? t - lag : t
        for c in axes(Z, 3)
            z = Z[s - zo, i, c]
            if !iszero(z)
                x -= z * R[t, c]
            end
        end
        Xl[t, i] = x
    end
    return Xl
end
function cross_sectional_local_returns(lx::AbstractString, ::NamedTuple, ::MatNum,
                                       rd::ReturnsResult, ::Integer,
                                       ::Option{<:MatNum} = nothing, ::Integer = 0)
    return descriptor_field_values(rd, lx)
end
"""
    cross_sectional_observed_block(cc::Nothing, rw, r, o::Integer = 0) -> nothing
    cross_sectional_observed_block(cc::NamedTuple, rw, r, o::Integer = 0) -> NamedTuple

Trim the observed factors to the observations a fit reads, and refuse an infinite observed return among them.

A `NaN` marks a gap, such as a series that starts after the first fitted observation, and the trimmed returns keep it. The derived net returns of every asset that holds the factor are then `NaN` at that observation, so the eligibility mask drops those pairs from the regression. The factor prior states the moments of the factor by its own rule. A mask-aware estimator, such as the default one, estimates them from the observations the series has. A plain estimator leaves the factor out of its Coverage Universe, so the moments are `NaN`, and the assets that hold the factor leave the Investable Mask. An infinity marks no gap, so it is refused. The check reads only the observations left after the Descriptor warm-up and the exposure lag, and it runs before the regression, so the error names the series.

# Arguments

  - `cc`: The observed factors [`cross_sectional_observed`](@ref) read, or `nothing`.
  - `rw`: The observations left after the Descriptor warm-up, as rows of the returns data.
  - `r`: The fitted observations, as an index into `rw`.
  - `o`: Number of observations of the returns data before the first row of `cc`, which the message adds to the row that it names. The carry fold carries the observations after the warm-up alone.

# Validation

  - No observed return of the fitted observations is infinite. Raises an [`IsNonFiniteError`](@ref) that names the series and the first observation, as a row of the returns data.

# Returns

  - `cb::Option{<:NamedTuple}`: `nothing`, or `(; R, Zr, Zt, nf, fam)`: the observed returns of the fitted observations, the exposures of the fitted observations as a view of `cc.Z`, those of the latest observation, the factor names and the family labels.

# Related

  - [`cross_sectional_observed`](@ref)
  - [`cross_sectional_observed_append`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_observed_block(::Nothing, ::Any, ::Any, ::Integer = 0)
    return nothing
end
function cross_sectional_observed_block(cc::NamedTuple, rw, r, o::Integer = 0)
    rows = rw[r]
    R = cc.R[rows, :]
    bad = [c for c in axes(R, 2) if any(isinf, view(R, :, c))]
    if !isempty(bad)
        t = findfirst(t -> any(isinf, view(R, t, :)), axes(R, 1))
        throw(IsNonFiniteError("the observed factor returns must not be infinite on an observation the fit reads, and the series $(unique(cc.lv[bad])) are, first at observation $(o + rows[t]) of the returns data. A NaN marks a gap and is accepted."))
    end
    return (; R = R, Zr = view(cc.Z, rows, :, :), Zt = cc.Z[rows[end], :, :], nf = cc.nf,
            fam = cc.fam)
end
"""
    cross_sectional_observed_gap(cb::Nothing, t::Integer) -> String
    cross_sectional_observed_gap(cb::NamedTuple, t::Integer) -> String

State the observed factors that have no return at one fitted observation, as a sentence a refusal appends.

A `NaN` observed return makes the net return of every asset that holds the factor `NaN` at that observation, so the eligibility mask drops those assets there. An observed factor that every asset holds, such as an [`ObservedExposure`](@ref) of a [`ConstantExposure`](@ref), then leaves the observation with no eligible asset, and [`assert_cross_sectional_coverage`](@ref) refuses the fit. The sentence names the cause, which the advice of that refusal does not reach. A prior with no observed factor, or an observation where every observed return is a number, gives the empty string.

# Arguments

  - `cb`: The observed factors of the fitted observations, from [`cross_sectional_observed_block`](@ref), or `nothing`.
  - `t`: The fitted observation, as a row of `cb.R`.

# Returns

  - `note::String`: The sentence, with a leading space, or the empty string.

# Related

  - [`cross_sectional_observed_block`](@ref)
  - [`assert_cross_sectional_coverage`](@ref)
"""
function cross_sectional_observed_gap(::Nothing, ::Integer)
    return ""
end
function cross_sectional_observed_gap(cb::NamedTuple, t::Integer)
    gap = cb.nf[isnan.(view(cb.R, t, :))]
    return if isempty(gap)
        ""
    else
        " The observed factors $gap have no return at that observation, so the net return of every asset that holds them is NaN there. Start the fit where those series start, or give them a return there."
    end
end
"""
    cross_sectional_observed_append(cb::Nothing, f, L, Ms, nf, fam, fcb) -> NamedTuple
    cross_sectional_observed_append(cb::NamedTuple, f, L, Ms, nf, fam, fcb) -> NamedTuple

Append the observed factors to the fitted factor model, after the factors the regression estimated.

The regression runs on the estimated exposures alone. This function puts the observed factors on every axis the factor model states, as its trailing columns: the observed returns after the estimated factor returns, the observed exposures after the estimated loadings and after the exposure history, and their names and family labels after the estimated ones. The Factor Family Basis takes them as pass-through factors with [`append_passthrough_factors`](@ref), so no constrained family re-bases them. A prior with no observed factor takes the method over `Nothing`, which returns the inputs unchanged and no `fx`.

# Arguments

  - `cb`: The observed factors of the fitted observations, from [`cross_sectional_observed_block`](@ref), or `nothing`.
  - `f`: The estimated factor returns on the reduced axis, `observations × factors`.
  - `L`: The estimated loadings of the latest observation on the reduced axis, `assets × factors`.
  - `Ms`: The exposure history of the fitted observations on the raw axis, `observations × assets × factors`.
  - `nf`: Name of each raw factor.
  - `fam`: Family label of each raw factor.
  - `fcb`: The Factor Family Basis over the post-warm-up observations, or `nothing`.

# Returns

  - `ca::NamedTuple`: `(; f, L, Ms, nf, fam, fcb, fx)`, the six inputs with the observed factors appended, and the observed returns in `fx`, or `nothing` in `fx` when there is no observed factor.

# Related

  - [`cross_sectional_observed_block`](@ref)
  - [`append_passthrough_factors`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalFactorPrior`](@ref)
"""
function cross_sectional_observed_append(::Nothing, f::MatNum, L::MatNum, Ms::Arr3Num,
                                         nf::VecStr, fam::VecStr,
                                         fcb::Option{<:AbstractFactorFamilyBasis})
    return (; f = f, L = L, Ms = Ms, nf = nf, fam = fam, fcb = fcb, fx = nothing)
end
function cross_sectional_observed_append(cb::NamedTuple, f::MatNum, L::MatNum, Ms::Arr3Num,
                                         nf::VecStr, fam::VecStr,
                                         fcb::Option{<:AbstractFactorFamilyBasis})
    nfa = vcat(nf, cb.nf)
    @argcheck(allunique(nfa),
              ArgumentError("the factor axis repeats a name once the observed factors join it. Got $nfa"))
    return (; f = hcat(f, cb.R), L = hcat(L, cb.Zt),
            Ms = cross_sectional_joined_exposures(Ms, cb.Zr), nf = nfa,
            fam = vcat(fam, cb.fam), fcb = append_passthrough_factors(fcb, length(cb.nf)),
            fx = cb.R)
end
"""
    cross_sectional_joined_exposures(Ms::AbstractArray{<:Any, 3}, Zo::AbstractArray{<:Any, 3})
    cross_sectional_joined_exposures(Ms::SubArray{<:Any, 3}, Zo::SubArray{<:Any, 3})

Join the exposure history of the estimated factors and the exposure history of the observed factors along the factor axis, the estimated factors first.

The carry fold keeps the two histories as views of one backing, whose factor axis holds the estimated factors and then the observed ones, with [`cross_sectional_fold_join`](@ref). When `Ms` and `Zo` are views of the same rows and the same assets of one backing, and its factor axis holds `Ms` and then `Zo` and nothing else, the joined history is a view of that backing, and the method copies nothing. Any other pair joins by a copy, as the batch fit joins its histories.

# Arguments

  - `Ms`: The exposure history of the estimated factors, `observations × assets × factors`.
  - `Zo`: The exposure history of the observed factors over the same observations and assets.

# Returns

  - `M::AbstractArray{<:Any, 3}`: `Ms` followed by `Zo` along the factor axis.

# Related

  - [`cross_sectional_observed_append`](@ref)
  - [`cross_sectional_fold_join`](@ref)
"""
function cross_sectional_joined_exposures(Ms::AbstractArray{<:Any, 3},
                                          Zo::AbstractArray{<:Any, 3})
    return cat(Ms, Zo; dims = 3)
end
function cross_sectional_joined_exposures(Ms::SubArray{<:Any, 3}, Zo::SubArray{<:Any, 3})
    P = parent(Ms)
    i, j = parentindices(Ms), parentindices(Zo)
    K = size(Ms, 3)
    return if P === parent(Zo) &&
              i[1] == j[1] &&
              i[2] == j[2] &&
              i[3] == 1:K &&
              j[3] == (K + 1):size(P, 3)
        view(P, i[1], i[2], :)
    else
        cat(Ms, Zo; dims = 3)
    end
end

export ObservedExposure, CurrencyExposure
public AbstractObservedExposureEstimator, observed_series
