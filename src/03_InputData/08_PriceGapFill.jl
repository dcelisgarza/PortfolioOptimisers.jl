"""
$(DocStringExtensions.TYPEDEF)

States that a price did not move across a gap, so the last priced observation is held forward.

`CarriedPrice` is the **Held Price** convention: inside a Held Gap the price is the last observed one, so the returns across the gap are flat and the whole move lands on the observation that ends it. Wealth is conserved, which a per-asset constant cannot do — filling a suspension with an asset's median manufactures one large move into the gap and another out of it.

It names a convention rather than a value, so it is the one member of [`PriceGapFill`](@ref)'s `fill` slot that is not a number: the value it fills with is read off the data, and the fitted state is only the seed a window that opens inside a gap starts from.

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

Fills the price gaps inside an asset's listing with a stated convention, and touches nothing outside it.

`PriceGapFill` is the ingestion layer's one fill, and it is off unless a caller adds it. A fill exists to **state a price convention** across a suspension or a holiday, not to remove a gap: a gap is carried through the conversion, so nothing downstream needs it gone. The fill is bounded by the **Listing Span**, so it touches **Held Gaps** alone and can never fabricate a price where an asset was not yet listed or has been delisted. ADR 0130 owns both rules.

It runs at the price level, before [`PricesToReturns`](@ref), because a carried price is a statement about a price and cannot be expressed after the conversion: across a gapped run `p₀, _, _, p₃` the unfilled returns are all non-finite, and zeroing them discards the `p₀ → p₃` move entirely. Carried forward at the price level the same run gives `0, 0, p₃/p₀ - 1`. A filled cell is therefore finite in the returns and its estimation mask entry is `true`, which is the design rather than an oversight: a caller who filled has said the asset traded.

# Algorithm

## Fit

 1. For each asset column, collect the observed prices of the training window. An entry [`is_missing_value`](@ref) accepts is left out.
 2. Skip an asset whose column holds no observed price. It gets no fitted value, and no entry in the result, so it is left untouched at apply time.
 3. Reduce the observed prices of the column to one value with [`PortfolioOptimisers.gap_fill_seed`](@ref), giving that asset's fitted value.
 4. Return a [`PriceGapFillResult`](@ref) holding the fitted asset names, their values, the convention and `strict`.

## Apply

 1. Copy the price values of the window, so the input is not mutated.
 2. Read the Listing Span that bounds the fill with [`PortfolioOptimisers.gap_fill_span`](@ref). A carrier that states none bounds the fill by nothing, so no cell is written.
 3. For each fitted asset name, find its column in the window. Skip a name the window does not carry.
 4. Run the convention forward through that column with [`PortfolioOptimisers.gap_fill_column!`](@ref), seeded by the fitted value and bounded by the span.
 5. Rebuild `X` from the filled values, keeping the timestamps and the column names, then rebuild the [`PricesResult`](@ref). Every other field passes through untouched.

The fitted value is a **seed**, and under [`CarriedPrice`](@ref) it is only that: a window that opens inside a gap fills from the last observed *training* price, and every later gap of that window fills from the most recent price the window itself observed. So no fill reads the future, and the convention still tracks the data it is replayed on.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceGapFill(;
        fill::Union{CarriedPrice, Num_VecToScaM} = CarriedPrice(),
        strict::Bool = false,
    ) -> PriceGapFill

Keywords correspond to the struct's fields.

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
    The convention a gap inside the Listing Span takes. [`CarriedPrice`](@ref) holds the last priced observation forward; a [`Num_VecToScaM`](@ref) reduces the asset's observed training prices to the one value every gap of that column takes.
    """
    fill
    """
    Whether a price carrier that states no Listing Span is refused (`true`) or warned about (`false`).
    """
    strict
    function PriceGapFill(fill::Union{CarriedPrice, Num_VecToScaM}, strict::Bool)
        return new{typeof(fill), typeof(strict)}(fill, strict)
    end
end
function PriceGapFill(; fill::Union{CarriedPrice, Num_VecToScaM} = CarriedPrice(),
                      strict::Bool = false)::PriceGapFill
    return PriceGapFill(fill, strict)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of a [`PriceGapFill`](@ref).

Carries the per-asset value fitted on the training window, together with the convention that value is read under. One result type serves both conventions, as [`AssetSelectorResult`](@ref) serves the whole selector family: the convention is a field rather than a second type, because it is what [`apply_preprocessing`](@ref) dispatches on and nothing else about the two differs.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`PriceGapFill`](@ref)
  - [`CarriedPrice`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
"""
@concrete struct PriceGapFillResult <: AbstractPricesPreprocessingResult
    """
    Names of the assets with a fitted value.
    """
    nx
    """
    Fitted values, aligned with `nx`. The last observed training price under [`CarriedPrice`](@ref), and the reduced scalar under a [`Num_VecToScaM`](@ref).
    """
    v
    """
    The convention `v` is read under, copied from the estimator.
    """
    fill
    """
    Whether a price carrier that states no Listing Span is refused (`true`) or warned about (`false`).
    """
    strict
end
"""
    carrier_listing_span(pr::AbstractPricesResult) -> Nothing
    carrier_listing_span(pr::PricesResult) -> Option{<:AbstractMatrix{Bool}}

Read the Listing Span a price carrier states, or `nothing` when it states none.

ADR 0129 rides the span on the price carrier, so a step that needs one asks the carrier rather than deriving its own. A [`PricesResult`](@ref) answers with its `span` field, which [`price_ingestion`](@ref) fills and a carrier assembled by hand leaves `nothing`; every other member of the family answers `nothing`, because a carrier that carries no span states no listing calendar and the step that asked must fall back and say so.

# Algorithm

The method that Julia selects is the algorithm.

 1. Any price carrier: `nothing`. The family states no span of its own.
 2. A [`PricesResult`](@ref): its `span` field.

# Arguments

  - `pr`: The price carrier the fill is running on.

# Returns

  - `span::Option{<:AbstractMatrix{Bool}}`: The carrier's Listing Span, or `nothing`.

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

The span is the carrier's, because a listing calendar is a fact about the instruments and a window cannot see all of it. A carrier that states none leaves only the window in hand, and the window cannot answer: a suspension straddling its edge reads there as an inception or a delisting, so a window-local derivation fills the wrong cells rather than fewer of them. So the fill is bounded by nothing at all — an all-`false` span, under which every cell lies outside a listing and no price is written — and it says so by name, refusing under `strict`. The diagnostic fires only when the window actually holds a gap, since a gapless window has nothing to fill and nothing to get wrong.

The refusal is [`strict_diagnostic`](@ref)'s, which is the library's one strictness policy.

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is an `AbstractMatrix{Bool}`: check its shape against `X` and answer it. A caller's own declaration and a derived [`PortfolioOptimisers.ListingSpan`](@ref) enter alike, under the public bound.
 2. `span` is `nothing`: report through [`strict_diagnostic`](@ref) when `X` holds a gap, then answer an all-`false` span of `X`'s shape, which fills nothing.

# Arguments

  - `span`: The listing statement the carrier holds, `observations × assets`, or `nothing`.
  - `X`: The price values of the window being transformed, `observations × assets`.
  - `strict`: If `true`, throws an `ArgumentError` when the carrier states no span; if `false`, issues a warning.

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
    gap_fill_seed(fill::CarriedPrice, obs::VecNum) -> Number
    gap_fill_seed(fill::Num_VecToScaM, obs::VecNum) -> Number

Reduce one asset's observed training prices to the value [`apply_preprocessing`](@ref) replays.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`CarriedPrice`](@ref): the last observed training price, which seeds a carry-forward on a window that opens inside a gap.
 2. [`Num_VecToScaM`](@ref): the reduction of the observed training prices, through [`vec_to_real_measure`](@ref).

# Arguments

  - `fill`: The convention, read off [`PriceGapFill`](@ref).
  - `obs`: One asset's observed training prices, in observation order.

# Returns

  - `v::Number`: The asset's fitted value.

# Related

  - [`PriceGapFill`](@ref)
  - [`PriceGapFillResult`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
function gap_fill_seed(::CarriedPrice, obs::VecNum)
    return obs[end]
end
function gap_fill_seed(fill::Num_VecToScaM, obs::VecNum)
    return vec_to_real_measure(fill, obs)
end
"""
    gap_fill_column!(fill::CarriedPrice, X::AbstractMatrix, span::AbstractMatrix{Bool}, j::Integer, v::Number) -> AbstractMatrix
    gap_fill_column!(fill::Num_VecToScaM, X::AbstractMatrix, span::AbstractMatrix{Bool}, j::Integer, v::Number) -> AbstractMatrix

Write one column's fill in place, inside the asset's listing and nowhere else.

Both methods read `span` before they read the price, so an observation outside the listing is never written whatever the convention states. That is where the guarantee sits: the fill cannot fabricate a price before an asset's first listing or after its delisting, because those observations are outside the span by the Span Rule.

# Algorithm

The method that Julia selects is the algorithm, and the two differ in what they write.

 1. [`CarriedPrice`](@ref): walk the observation axis carrying a price, seeded by `v`. Inside the listing, write the carried price onto a gap, and take an observed price as the new carried price.
 2. [`Num_VecToScaM`](@ref): write `v` onto every gap inside the listing. The value is already the reduction, so nothing is carried and an observed price is read by nothing.

# Arguments

  - `fill`: The convention, read off [`PriceGapFillResult`](@ref).
  - `X`: The price values of the window, mutated in place.
  - `span`: The listing statement bounding the fill, `observations × assets`.
  - `j`: Index of the column to fill.
  - `v`: The asset's fitted value.

# Returns

  - `X::AbstractMatrix`: The same matrix, with column `j` filled.

# Related

  - [`PriceGapFill`](@ref)
  - [`PriceGapFillResult`](@ref)
  - [`PortfolioOptimisers.gap_fill_span`](@ref)
  - [`is_missing_value`](@ref)
"""
function gap_fill_column!(::CarriedPrice, X::AbstractMatrix, span::AbstractMatrix{Bool},
                          j::Integer, v::Number)
    carry = v
    for t in axes(X, 1)
        if !span[t, j]
            continue
        end
        x = X[t, j]
        if is_missing_value(x)
            X[t, j] = carry
        else
            carry = x
        end
    end
    return X
end
function gap_fill_column!(::Num_VecToScaM, X::AbstractMatrix, span::AbstractMatrix{Bool},
                          j::Integer, v::Number)
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
    for i in axes(vals, 2)
        obs = identity.([x for x in view(vals, :, i) if !is_missing_value(x)])
        if isempty(obs)
            continue
        end
        push!(keep, i)
        push!(v, gap_fill_seed(est.fill, obs))
    end
    return PriceGapFillResult(names[keep], identity.(v), est.fill, est.strict)
end
function apply_preprocessing(res::PriceGapFillResult, pr::PricesResult)::PricesResult
    names = TimeSeries.colnames(pr.X)
    vals = copy(values(pr.X))
    span = gap_fill_span(carrier_listing_span(pr), vals, res.strict)
    for (name, v) in zip(res.nx, res.v)
        j = findfirst(==(name), names)
        if isnothing(j)
            continue
        end
        gap_fill_column!(res.fill, vals, span, j, v)
    end
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X), vals, names)
    #! A fill states a price, not a listing, so the span passes through untouched: it is
    #! what bounded the fill, and the Span Rule reads the same listing off the filled
    #! panel as off the raw one.
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                        pnl = pr.pnl, span = pr.span)
end

export CarriedPrice, PriceGapFill, PriceGapFillResult
