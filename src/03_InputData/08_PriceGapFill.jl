"""
$(DocStringExtensions.TYPEDEF)

States that a price did not move across a gap, so the fill holds the last observed price forward.

`CarriedPrice` is the **Held Price** convention. Inside a Held Gap the price is the last observed price of the listed run, so every return inside the gap is zero and the return at the observation that ends the gap carries the whole move. The fill thus conserves wealth. A constant per asset cannot conserve it, because a fill with the median of an asset makes one large move into a suspension and a second one out of it.

It names a convention and not a value, so it is the one member of the `fill` slot of [`PriceGapFill`](@ref) that is not a number. The fill reads each price off the data. The fitted value is only the seed for a later window that opens inside a gap.

# Mathematical definition

```math
\\begin{align}
\\tilde{p}_{t,\\,i} &= \\begin{cases}
p_{s_{t,\\,i},\\,i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i}\\,,\\ s_{t,\\,i} \\text{ exists} \\\\
v_{i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i}\\,,\\ s_{t,\\,i} \\text{ does not exist}\\,,\\ t \\geq t_{0}\\,,\\ r_{t,\\,i} = 1\\,,\\ v_{i} \\text{ exists} \\\\
p_{t,\\,i} & \\text{otherwise}
\\end{cases}\\,, \\\\
r_{t,\\,i} &= \\min\\left\\{u \\leq t : a_{u',\\,i} = 1 \\text{ for } u \\leq u' \\leq t\\right\\}\\,, \\\\
s_{t,\\,i} &= \\max\\left(\\mathcal{O}_{i} \\cap \\{r_{t,\\,i}, \\ldots, t - 1\\}\\right)\\,.
\\end{align}
```

The third case holds a gap that no case fills, so that gap stays absent. Let ``b < c`` be two consecutive members of ``\\mathcal{O}_{i}`` in one listed run. Then ``\\tilde{p}_{t,\\,i} = p_{b,\\,i}`` for ``b < t < c``, so the returns at ``b + 1, \\ldots, c - 1`` are zero and the return at ``c`` is ``p_{c,\\,i} / p_{b,\\,i} - 1``. A gap after an absence from the listing takes no price from before the absence.

Where:

  - $(math_dict[:p_tilde_ti_fill])
  - $(math_dict[:p_ti_price])
  - $(math_dict[:a_ti_span])
  - $(math_dict[:O_i_fill])
  - $(math_dict[:r_ti_run])
  - $(math_dict[:s_ti_carry])
  - $(math_dict[:v_i_fill])
  - $(math_dict[:t0_fill])
  - $(math_dict[:n_span])

# Constructors

    CarriedPrice() -> CarriedPrice

# Examples

```jldoctest
julia> CarriedPrice()
CarriedPrice()
```

# Related

  - [`PriceGapFill`](@ref)
  - [`PriceGapFillResult`](@ref)
  - [`Num_VecToScaM`](@ref)
"""
struct CarriedPrice end
"""
$(DocStringExtensions.TYPEDEF)

Fills the price gaps inside the listing of each asset with a stated convention, and changes no other price.

`PriceGapFill` is the one fill of the ingestion layer, and it runs only when a caller adds it. A fill states a price convention across a suspension or a holiday. Its purpose is not to remove a gap, because the conversion keeps a gap and no later step needs it gone. The **Listing Span** bounds the fill, so the fill writes into **Held Gaps** alone and never makes a price before an asset is listed or after it is delisted.

The fill runs on prices, before [`PricesToReturns`](@ref), because a carried price is a statement about a price and the returns cannot state it. Across a run of prices `p₀, _, _, p₃` the unfilled returns are all non-finite, and a zero in their place loses the move from `p₀` to `p₃`. The carried prices give the returns `0, 0, p₃/p₀ - 1`. A filled cell thus has a finite return and a `true` estimation mask entry. That is the intent, because a caller who fills states that the asset traded.

Under [`CarriedPrice`](@ref) the fitted value is only a seed. A window that opens inside a gap after the training window takes the last observed training price, and every later gap of that window takes the most recent price that the window observed in the same listed run. The fill writes the seed onto no observation at or before the end of the training window. A [`Pipeline`](@ref) transforms the training window with the step that it just fitted, and on that window a gap that opens the window stays a Held Gap. It does not take the last price of the window. No fill thus reads a later price.

An asset gets no seed when an absence from its listing follows its last observed training price, or when the training window ends outside its listing. A later window cannot see that absence, so a seed from before it would book a return across the absence. The asset keeps its entry in the result, and a later window fills its gaps from the prices of that window alone.

# Mathematical definition

Under [`CarriedPrice`](@ref):

```math
\\begin{align}
\\tilde{p}_{t,\\,i} &= \\begin{cases}
p_{s_{t,\\,i},\\,i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i}\\,,\\ s_{t,\\,i} \\text{ exists} \\\\
v_{i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i}\\,,\\ s_{t,\\,i} \\text{ does not exist}\\,,\\ t \\geq t_{0}\\,,\\ r_{t,\\,i} = 1\\,,\\ v_{i} \\text{ exists} \\\\
p_{t,\\,i} & \\text{otherwise}
\\end{cases}\\,, \\\\
v_{i} &= p_{m_{i},\\,i} \\quad \\text{when } a_{u,\\,i} = 1 \\text{ for } m_{i} \\leq u \\leq n^{\\mathrm{tr}}\\,, \\\\
m_{i} &= \\max \\mathcal{O}_{i}^{\\mathrm{tr}}\\,.
\\end{align}
```

When an absence falls between ``m_{i}`` and ``n^{\\mathrm{tr}}``, ``v_{i}`` does not exist.

Under a [`Num_VecToScaM`](@ref):

```math
\\begin{align}
\\tilde{p}_{t,\\,i} &= \\begin{cases}
v_{i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i} \\\\
p_{t,\\,i} & \\text{otherwise}
\\end{cases}\\,, \\\\
v_{i} &= \\phi\\left(\\left\\{p_{u,\\,i} : u \\in \\mathcal{O}_{i}^{\\mathrm{tr}}\\right\\}\\right)\\,.
\\end{align}
```

An asset with an empty ``\\mathcal{O}_{i}^{\\mathrm{tr}}`` has no ``v_{i}``, and every price of that asset stays as it is.

Where:

  - $(math_dict[:p_tilde_ti_fill])
  - $(math_dict[:p_ti_price])
  - $(math_dict[:a_ti_span])
  - $(math_dict[:O_i_fill])
  - ``\\mathcal{O}_{i}^{\\mathrm{tr}}``: Observed set of asset ``i`` over the training window.
  - ``m_{i}``: Last observation of the training window at which asset ``i`` has an observed price.
  - ``n^{\\mathrm{tr}}``: Last observation of the training window.
  - $(math_dict[:r_ti_run])
  - $(math_dict[:s_ti_carry])
  - $(math_dict[:v_i_fill])
  - $(math_dict[:phi_fill])
  - $(math_dict[:t0_fill])
  - $(math_dict[:n_span])

# Algorithm

## Fit

 1. For each asset column, find `t`, the row of the last observed price of the training window. The step leaves out every entry that [`is_missing_value`](@ref) accepts.
 2. Skip an asset whose column holds no observed price. The asset gets no fitted value and no entry in the result, so the apply step does not change it.
 3. Collect the observed prices of the column into `obs`. Read with [`PortfolioOptimisers.gap_fill_open`](@ref) whether the Listing Span of the carrier holds from `t` to the end of the training window. A carrier that states no span states no absence.
 4. Reduce `obs` to one value with [`PortfolioOptimisers.gap_fill_seed`](@ref), giving the fitted value of the asset. Under [`CarriedPrice`](@ref) the value is `missing` when the span does not hold from `t` to the end.
 5. Return a [`PriceGapFillResult`](@ref) that holds the fitted asset names, their values, the last timestamp of the training window, the convention and `strict`.

## Apply

 1. Copy the price values of the window into `vals`, so the input does not change.
 2. Read the Listing Span `span` that bounds the fill with [`PortfolioOptimisers.gap_fill_span`](@ref). A carrier that states no span gives an all-`false` span, and the fill writes no cell.
 3. Find `t0`, the first observation of the window after the end of the training window. On the training window itself `t0` is one past the last row. On a window that follows the training window, `t0` is the first row.
 4. For each fitted asset name, find its column `j` in the window. Skip a name that the window does not carry.
 5. Fill column `j` of `vals` with [`PortfolioOptimisers.gap_fill_column!`](@ref), with the fitted value as the seed from `t0` and `span` as the bound. A `missing` seed fills the gaps of the column from the prices of the window alone.
 6. Rebuild `X` from `vals` with the same timestamps and column names, then rebuild the [`PricesResult`](@ref). Every other field passes through unchanged.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceGapFill(;
        fill::Union{CarriedPrice, Num_VecToScaM} = CarriedPrice(),
        strict::Bool = false,
        cache::Option{<:AbstractPartialFitState} = nothing,
    ) -> PriceGapFill

Keywords correspond to the struct's fields.

# Online form

A `PriceGapFill` with a [`CarriedPrice`](@ref) fill has an online step. [`partial_fit_transform`](@ref) fills a block of prices from the carried prices and moves the carried prices forward. [`fit_preprocessing`](@ref) with no data reads the [`PriceGapFillResult`](@ref) of the whole history out of `cache`. A statistic fill has no online form, because a longer window changes the value of every earlier gap. [`supports_partial_fit`](@ref) returns `false` for it, and a [`Pipeline`](@ref) refuses it at warm-up with a message that names the step.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 4),
                     [100.0 1.0; NaN 3.0; NaN 5.0; 130.0 7.0], [\"A\", \"B\"]);

julia> res = fit_preprocessing(PriceGapFill(), PricesResult(; X = X));

julia> res.v
2-element Vector{Float64}:
 130.0
   7.0
```

# Related

  - [`PriceGapFillResult`](@ref)
  - [`CarriedPrice`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`listing_span`](@ref)
  - [`PricesToReturns`](@ref)
  - [`Num_VecToScaM`](@ref)
"""
@concrete struct PriceGapFill <: AbstractPricesPreprocessingEstimator
    """
    Convention for a gap inside the Listing Span. [`CarriedPrice`](@ref) holds the last observed price forward. A [`Num_VecToScaM`](@ref) reduces the observed training prices of an asset to one value, and every gap of that asset takes the value.
    """
    fill
    """
    $(field_dict[:strict_span])
    """
    strict
    """
    $(field_dict[:pfcache])
    """
    cache
    function PriceGapFill(fill::Union{CarriedPrice, Num_VecToScaM}, strict::Bool,
                          cache::Option{<:AbstractPartialFitState})
        return new{typeof(fill), typeof(strict), typeof(cache)}(fill, strict, cache)
    end
end
function PriceGapFill(; fill::Union{CarriedPrice, Num_VecToScaM} = CarriedPrice(),
                      strict::Bool = false,
                      cache::Option{<:AbstractPartialFitState} = nothing)::PriceGapFill
    return PriceGapFill(fill, strict, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the value that a price gap fill fitted for each asset, with the end of the training window and the convention.

[`PriceGapFill`](@ref) makes it, and [`apply_preprocessing`](@ref) replays it on a window. One result type serves both conventions, as [`AssetSelectorResult`](@ref) serves the whole selector family. The convention is a field and not a second type, because [`PortfolioOptimisers.gap_fill_column!`](@ref) dispatches on it and the two conventions differ in nothing else.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`PriceGapFill`](@ref)
  - [`CarriedPrice`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
"""
@concrete struct PriceGapFillResult <: AbstractPricesPreprocessingResult
    """
    Names of the assets with an observed training price.
    """
    nx
    """
    Fitted values, aligned with `nx`. Under [`CarriedPrice`](@ref) each value is the last observed training price of its asset, or `missing` when an absence from the listing follows that price in the training window. A `missing` seed is never written. Under a [`Num_VecToScaM`](@ref) each value is the reduction of the observed training prices of its asset.
    """
    v
    """
    Last timestamp of the training window. Under [`CarriedPrice`](@ref) the fill writes a seed of `v` only onto an observation after `te`. A gap at or before `te` thus never takes the seed. A [`Num_VecToScaM`](@ref) does not read `te`.
    """
    te
    """
    Convention that reads `v`, copied from the estimator.
    """
    fill
    """
    $(field_dict[:strict_span])
    """
    strict
end
"""
    carrier_listing_span(pr::AbstractPricesResult) -> Nothing
    carrier_listing_span(pr::PricesResult) -> Option{<:AbstractMatrix{Bool}}

Read the Listing Span that a price carrier states, or `nothing` when it states none.

The price carrier holds the Listing Span, so a step that needs a span asks the carrier and does not derive one. A [`PricesResult`](@ref) returns its `span` field. [`price_ingestion`](@ref) fills that field, and a carrier that a caller builds by hand leaves it `nothing`. Every other member of the family returns `nothing`, because it states no listing calendar. The step that asked then uses its fallback and reports it.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any other price carrier: return `nothing`, because the family states no span of its own.
 2. A [`PricesResult`](@ref): return its `span` field.

# Arguments

  - `pr`: The price carrier that the fill reads.

# Returns

  - `span::Option{<:AbstractMatrix{Bool}}`: The Listing Span of the carrier, or `nothing`.

# Related

  - [`PortfolioOptimisers.gap_fill_span`](@ref)
  - [`listing_span`](@ref)
  - [`price_ingestion`](@ref)
  - [`PricesResult`](@ref)
  - [`Option`](@ref)
"""
function carrier_listing_span(::AbstractPricesResult)
    return nothing
end
function carrier_listing_span(pr::PricesResult)
    return pr.span
end
"""
    gap_fill_span(span::AbstractMatrix{Bool}, X::AbstractMatrix, strict::Bool) -> AbstractMatrix{Bool}
    gap_fill_span(span::Nothing, X::AbstractMatrix, strict::Bool) -> BitMatrix

Resolve the Listing Span that bounds a [`PriceGapFill`](@ref) over one window.

The span comes from the carrier, because a listing calendar is a fact about the instruments and a window cannot see all of it. A carrier that states no span leaves only the window, and the window cannot give the span. A suspension across the edge of a window reads there as an inception or a delisting, so a span derived from the window fills the wrong cells, not fewer cells. The fill therefore takes an all-`false` span, under which every cell lies outside a listing and the fill writes no price. The function reports this with a warning, or with an error under `strict`. It reports only when the window holds a gap, because a window with no gap has nothing to fill.

[`strict_diagnostic`](@ref) raises the warning or the error. It is the one strictness policy of the library.

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is an `AbstractMatrix{Bool}`: check its shape against `X` and return it. A caller's own declaration and a derived [`PortfolioOptimisers.ListingSpan`](@ref) enter alike, under the public bound.
 2. `span` is `nothing`: report through [`strict_diagnostic`](@ref) when `X` holds a gap, then return an all-`false` span of the shape of `X`, which fills nothing.

# Arguments

  - `span`: The listing statement that the carrier holds, `observations × assets`, or `nothing`.
  - `X`: The price values of the window that the fill transforms, `observations × assets`.
  - $(arg_dict[:strict_span])

# Validation

  - `size(span) == size(X)`. Raises a `DimensionMismatch`.
  - The carrier states a span when `X` holds a gap. Raises an `ArgumentError` under `strict`.

# Returns

  - `span`: The listing statement on the price clock of `X`, or an all-`false` span when the carrier states none.

# Related

  - [`PriceGapFill`](@ref)
  - [`PortfolioOptimisers.carrier_listing_span`](@ref)
  - [`listing_span`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function gap_fill_span(span::AbstractMatrix{Bool}, X::AbstractMatrix, ::Bool)
    @argcheck(size(span) == size(X),
              DimensionMismatch("a listing statement and the window it bounds have the same shape; got size(span) = $(size(span)) and size(X) = $(size(X))"))
    return span
end
function gap_fill_span(::Nothing, X::AbstractMatrix, strict::Bool)
    if any(is_missing_value, X)
        strict_diagnostic("`PriceGapFill` is bounded by the Listing Span, and the price carrier states none, so nothing was filled. The window cannot supply one: a suspension straddling its edge reads there as an inception or a delisting, so a window-local derivation fills the wrong cells rather than fewer of them. Build the carrier through the ingestion layer, which states a span.",
                          strict)
    end
    return falses(size(X))
end
"""
    gap_fill_open(span::AbstractMatrix{Bool}, j::Integer, t::Integer) -> Bool
    gap_fill_open(span::Nothing, j::Integer, t::Integer) -> Bool

Tell whether the listing of column `j` holds from observation `t` to the end of a window, so that a price at `t` can seed the next window.

A seed from before an absence books a return across that absence. A later window cannot see an absence at the end of the window before it, so the window that observes the absence must retire the seed. [`PriceGapFill`](@ref) reads this at the end of the training window, and its online step reads it at the end of each block.

# Mathematical definition

```math
\\begin{align}
o_{i} &= \\prod_{u = t}^{n} a_{u,\\,i}\\,.
\\end{align}
```

Here ``i`` is the asset of column `j`, and ``t`` is the row of its last observed price.

Where:

  - $(math_dict[:o_i_fill])
  - $(math_dict[:a_ti_span])
  - $(math_dict[:n_span])

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is an `AbstractMatrix{Bool}`: return `true` when every entry of column `j` from row `t` to the last row is `true`.
 2. `span` is `nothing`: return `true`. A carrier that states no span states no absence.

# Arguments

  - `span`: The listing statement of the carrier, `observations × assets`, or `nothing`.
  - `j`: Index of the column.
  - `t`: Row of the price that seeds the next window. It is `1` when the window observes no price of the column and the seed comes from an earlier window.

# Returns

  - `open::Bool`: `true` when the seed reaches the end of the window inside one listed run.

# Related

  - [`PriceGapFill`](@ref)
  - [`PortfolioOptimisers.gap_fill_seed`](@ref)
  - [`PortfolioOptimisers.carrier_listing_span`](@ref)
"""
function gap_fill_open(span::AbstractMatrix{Bool}, j::Integer, t::Integer)
    return all(view(span, t:size(span, 1), j))
end
function gap_fill_open(::Nothing, ::Integer, ::Integer)
    return true
end
"""
    gap_fill_seed(fill::CarriedPrice, obs::VecNum, open::Bool) -> Union{Missing, Number}
    gap_fill_seed(fill::Num_VecToScaM, obs::VecNum, open::Bool) -> Number

Reduce the observed training prices of one asset to the value that [`apply_preprocessing`](@ref) replays.

# Mathematical definition

Under [`CarriedPrice`](@ref):

```math
\\begin{align}
v_{i} &= p_{\\max \\mathcal{O}_{i},\\,i} \\quad \\text{when } o_{i} = 1\\,.
\\end{align}
```

When ``o_{i} = 0``, ``v_{i}`` does not exist.

Under a [`Num_VecToScaM`](@ref):

```math
\\begin{align}
v_{i} &= \\phi\\left(\\left\\{p_{u,\\,i} : u \\in \\mathcal{O}_{i}\\right\\}\\right)\\,.
\\end{align}
```

Here ``\\mathcal{O}_{i}`` is the observed set of asset ``i`` over the training window, and it is not empty.

Where:

  - $(math_dict[:v_i_fill])
  - $(math_dict[:p_ti_price])
  - $(math_dict[:O_i_fill])
  - $(math_dict[:o_i_fill])
  - $(math_dict[:phi_fill])

# Algorithm

The method that Julia selects is the algorithm.

 1. [`CarriedPrice`](@ref): return the last entry of `obs` when `open` is `true`. It is the seed for a later window that opens inside a gap. Return `missing` otherwise, because the seed would cross an absence.
 2. [`Num_VecToScaM`](@ref): return the reduction of `obs` through [`vec_to_real_measure`](@ref). The value carries no price, so an absence does not retire it.

# Arguments

  - `fill`: The convention, read off [`PriceGapFill`](@ref).
  - `obs`: The observed training prices of one asset, in observation order.
  - `open`: Whether the listing holds from the last observed price to the end of the training window, from [`PortfolioOptimisers.gap_fill_open`](@ref).

# Returns

  - `v::Union{Missing, Number}`: The fitted value of the asset, or `missing` under [`CarriedPrice`](@ref) when `open` is `false`.

# Related

  - [`PriceGapFill`](@ref)
  - [`PriceGapFillResult`](@ref)
  - [`PortfolioOptimisers.gap_fill_open`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
function gap_fill_seed(::CarriedPrice, obs::VecNum, open::Bool)
    return open ? obs[end] : missing
end
function gap_fill_seed(fill::Num_VecToScaM, obs::VecNum, ::Bool)
    return vec_to_real_measure(fill, obs)
end
"""
    gap_fill_column!(fill::CarriedPrice, X::AbstractMatrix, span::AbstractMatrix{Bool}, j::Integer, v::Union{Missing, Number}, t0::Integer) -> AbstractMatrix
    gap_fill_column!(fill::Num_VecToScaM, X::AbstractMatrix, span::AbstractMatrix{Bool}, j::Integer, v::Number, t0::Integer) -> AbstractMatrix

Write the fill of one column in place, inside the listing of the asset and nowhere else.

Both methods read `span` before they read the price, so they never write an observation outside the listing, whatever the convention. The Span Rule puts every observation before the first listing of an asset and after its delisting outside the span, so the fill cannot make a price there.

Under [`CarriedPrice`](@ref) the seed `v` is a price that the training window observed last, so the method writes it only onto an observation after that window. `t0` names the first such observation. A gap before `t0` with no observed price before it stays a gap. On the training window, which a [`Pipeline`](@ref) transforms with the step that it just fitted, every observation comes before `t0`, so a gap that opens the window does not take the last price of the window.

The carry also stops at an observation outside the listing. A caller's own span can take an asset out of the listing and back in. A gap after the return takes no price from before the absence and no seed, because with such a price the next return is a move across the absence. [`PortfolioOptimisers.project_span`](@ref) books no return across an absence, and the fill keeps that statement true. The window cannot see an absence before its first row, so the fit that saw the absence gives a `missing` seed. The method never writes a `missing` seed.

# Mathematical definition

Under [`CarriedPrice`](@ref):

```math
\\begin{align}
\\tilde{p}_{t,\\,i} &= \\begin{cases}
p_{s_{t,\\,i},\\,i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i}\\,,\\ s_{t,\\,i} \\text{ exists} \\\\
v_{i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i}\\,,\\ s_{t,\\,i} \\text{ does not exist}\\,,\\ t \\geq t_{0}\\,,\\ r_{t,\\,i} = 1\\,,\\ v_{i} \\text{ exists} \\\\
p_{t,\\,i} & \\text{otherwise}
\\end{cases}\\,.
\\end{align}
```

Under a [`Num_VecToScaM`](@ref):

```math
\\begin{align}
\\tilde{p}_{t,\\,i} &= \\begin{cases}
v_{i} & a_{t,\\,i} = 1\\,,\\ t \\notin \\mathcal{O}_{i} \\\\
p_{t,\\,i} & \\text{otherwise}
\\end{cases}\\,.
\\end{align}
```

Here ``i`` is the asset of the column, and ``\\mathcal{O}_{i}`` is its observed set over the window.

Where:

  - $(math_dict[:p_tilde_ti_fill])
  - $(math_dict[:p_ti_price])
  - $(math_dict[:a_ti_span])
  - $(math_dict[:O_i_fill])
  - $(math_dict[:r_ti_run])
  - $(math_dict[:s_ti_carry])
  - $(math_dict[:v_i_fill])
  - $(math_dict[:t0_fill])
  - $(math_dict[:n_span])

# Algorithm

The method that Julia selects is the algorithm, and the two methods differ in what they write.

 1. [`CarriedPrice`](@ref): walk the observation axis with the carry `carry`, which starts as the seed `v`, and the flag `carried`, which states whether the walk can write `carry`. A `missing` seed is retired from the start. At an observation outside the listing, clear `carried` and retire the seed. Inside the listing, set `carried` from `t0` on while the seed is not retired. Take an observed price as the new `carry` and set `carried`, or write `carry` onto a gap when `carried` is set.
 2. [`Num_VecToScaM`](@ref): write `v` onto every gap inside the listing. The value is already the reduction, so the method carries no price and reads neither an observed price nor `t0`.

# Arguments

  - `fill`: The convention, read off [`PriceGapFillResult`](@ref).
  - `X`: The price values of the window. The method changes them in place.
  - `span`: The listing statement that bounds the fill, `observations × assets`.
  - `j`: Index of the column to fill.
  - `v`: The fitted value of the asset. Under [`CarriedPrice`](@ref) it can be `missing`, and then the column fills from the prices of the window alone.
  - `t0`: Index of the first observation after the training window. It is `size(X, 1) + 1` when the window holds none, as on the training window itself, and `1` when every observation follows the training window.

# Returns

  - `X::AbstractMatrix`: The same matrix, with column `j` filled.

# Related

  - [`PriceGapFill`](@ref)
  - [`PriceGapFillResult`](@ref)
  - [`PortfolioOptimisers.gap_fill_span`](@ref)
  - [`is_missing_value`](@ref)
"""
function gap_fill_column!(::CarriedPrice, X::AbstractMatrix, span::AbstractMatrix{Bool},
                          j::Integer, v::Union{Missing, Number}, t0::Integer)
    #! `carry` holds the seed from the start and `carried` says whether it may be written:
    #! it may once the walk reaches `t0`, the first observation the seed precedes, or once
    #! an observed price has replaced the seed. A gap before either stays a gap. An absence
    #! from the listing ends the carry and retires the seed, so a gap after the asset
    #! rejoins waits for a price of its own run. A `missing` seed is retired from the start.
    carry = v
    carried = false
    seeded = !ismissing(v)
    for t in axes(X, 1)
        if !span[t, j]
            carried = seeded = false
            continue
        end
        carried |= seeded & (t >= t0)
        x = X[t, j]
        if !is_missing_value(x)
            carry = x
            carried = true
        elseif carried
            X[t, j] = carry
        end
    end
    return X
end
function gap_fill_column!(::Num_VecToScaM, X::AbstractMatrix, span::AbstractMatrix{Bool},
                          j::Integer, v::Number, ::Integer)
    for t in axes(X, 1)
        if span[t, j] && is_missing_value(X[t, j])
            X[t, j] = v
        end
    end
    return X
end
function fit_preprocessing(est::PriceGapFill, pr::PricesResult)::PriceGapFillResult
    names = TimeSeries.colnames(pr.X)
    vals = values(pr.X)
    keep = Vector{Int}(undef, 0)
    v = Vector{Any}(undef, 0)
    obs = Vector{nonmissingtype(eltype(vals))}(undef, 0)
    span = carrier_listing_span(pr)
    for i in axes(vals, 2)
        t = findlast(!is_missing_value, view(vals, :, i))
        if isnothing(t)
            continue
        end
        append!(empty!(obs), Iterators.filter(!is_missing_value, view(vals, :, i)))
        #! A later window cannot see an absence at the end of this one, so the seed is
        #! retired here when the listing breaks after the last observed price.
        push!(keep, i)
        push!(v, gap_fill_seed(est.fill, obs, gap_fill_open(span, i, t)))
    end
    return PriceGapFillResult(names[keep], identity.(v), last(TimeSeries.timestamp(pr.X)),
                              est.fill, est.strict)
end
function apply_preprocessing(res::PriceGapFillResult, pr::PricesResult)::PricesResult
    names = TimeSeries.colnames(pr.X)
    vals = copy(values(pr.X))
    span = gap_fill_span(carrier_listing_span(pr), vals, res.strict)
    #! The seed is a training price, so it is written only after the training window. A
    #! `TimeArray`'s clock is sorted, so the first observation past `te` is the first the
    #! seed precedes; on the training window itself that is one past the end.
    t0 = searchsortedlast(TimeSeries.timestamp(pr.X), res.te) + 1
    for (name, v) in zip(res.nx, res.v)
        j = findfirst(==(name), names)
        if isnothing(j)
            continue
        end
        gap_fill_column!(res.fill, vals, span, j, v, t0)
    end
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X), vals, names)
    #! A fill states a price, not a listing, so the span passes through untouched: it is
    #! what bounded the fill, and the Span Rule reads the same listing off the filled
    #! panel as off the raw one.
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                        pnl = pr.pnl, span = pr.span)
end

export CarriedPrice, PriceGapFill, PriceGapFillResult
