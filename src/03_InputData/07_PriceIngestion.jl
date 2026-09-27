"""
    series_value_type(A::Nothing) -> Type{Union{}}
    series_value_type(A::TimeSeries.TimeArray) -> Type

Read the value type of one optional price series, without the `Missing` a wide table may carry.

The ingestion layer derives the type it carries prices in from the series themselves. Each series contributes the type of its values without `Missing`, because the target type spells an absence and the source does not. A series the caller omitted contributes `Union{}`, the identity of `promote_type`, so the site that promotes needs no branch for an absent table.

# Algorithm

The method that Julia selects is the algorithm.

 1. `A` is `nothing`: return `Union{}`.
 2. `A` is a table: return the element type of its values with `Missing` removed. A table whose element type is `Missing` alone gives `Union{}` too, so it contributes nothing to the promotion.

# Arguments

  - `A`: One price series, or `nothing`.

# Returns

  - `T::Type`: The value type, `Union{}` for no series.

# Related

  - [`absence_type`](@ref)
  - [`unify_gaps`](@ref)
  - [`price_ingestion`](@ref)
"""
function series_value_type(::Nothing)
    return Union{}
end
function series_value_type(A::TimeSeries.TimeArray)
    return Base.nonmissingtype(eltype(values(A)))
end
"""
    absence_type(::Type{T}) -> Type
    absence_type(::Type{Union{}}) -> Union{}

Derive the type the ingestion layer carries a series of value type `T` in.

The layer spells an absent price `NaN`, so the carried type must be able to hold one. The layer reads that type off the data and does not name it: the type is [`float_if_integer`](@ref) of `T`. A floating-point type gives itself, so a `Float32` panel stays `Float32`. An integer type gives its floating-point type, `Float64` for `Int` and `BigFloat` for `BigInt`. A `Rational` and a number type the library does not know give themselves.

This function does not decide whether the type can hold an absence, because a series with no gap and no padding never spells one. [`absent_value`](@ref) refuses a type by name when the layer writes an absence in it.

# Algorithm

The method that Julia selects is the algorithm.

 1. `T` is `Union{}`: refuse. Every price table holds only `missing`, so no value type exists to derive.
 2. Otherwise return [`float_if_integer`](@ref) of `T`.

# Arguments

  - `T`: The value type of a series, as [`series_value_type`](@ref) reads it.

# Validation

  - `T !== Union{}`, which [`series_value_type`](@ref) gives for a table whose element type is `Missing` alone. Raises a `DomainError`.

# Returns

  - `S::Type`: [`float_if_integer`](@ref) of `T`.

# Related

  - [`series_value_type`](@ref)
  - [`absent_value`](@ref)
  - [`unify_gaps`](@ref)
  - [`price_ingestion`](@ref)
"""
function absence_type(::Type{T}) where {T}
    return float_if_integer(T)
end
function absence_type(::Type{Union{}})
    return throw(DomainError(Union{},
                             "the ingestion layer derives the type it carries prices in from the element type of the price series, and the series it was handed hold `missing` alone, so there is no value type to derive. Give the table a value type, for example `Matrix{Union{Missing, Float64}}`, or hand the layer a table that holds a price."))
end
"""
    absent_value(::Type{T}) -> T

Return the one spelling of an absent value in type `T`, and refuse by name a type that cannot hold one.

The layer spells an absence one way, `NaN`, so a type that cannot represent `NaN` cannot state a gap. This function refuses such a type, for example a `Rational` panel that holds a gap or that a join must pad. Without it, a conversion raises an `InexactError` that names nothing. The refusal names the type and what the caller can do. An integer panel never reaches the refusal, because [`absence_type`](@ref) widens it to a floating-point type first.

The layer stays open to number types it does not know. A type carries an absence when `convert(T, NaN)` returns a value for which `isnan` holds.

# Algorithm

 1. Convert `NaN` to `T`. A conversion that throws gives `nothing`.
 2. Refuse `T` unless the conversion gave a value and `isnan` holds for it.
 3. Return that value.

# Arguments

  - `T`: The type an absence must be spelled in.

# Validation

  - `convert(T, NaN)` succeeds and `isnan` of it holds. Otherwise raises a `DomainError` naming `T`.

# Returns

  - `x::T`: The absence, `convert(T, NaN)`.

# Related

  - [`absence_type`](@ref)
  - [`unify_gaps`](@ref)
  - [`align_series`](@ref)
  - [`price_ingestion`](@ref)
"""
function absent_value(::Type{T}) where {T}
    x = try
        convert(T, NaN)
    catch
        nothing
    end
    @argcheck(!isnothing(x) && isnan(x),
              DomainError(T,
                          "the ingestion layer spells an absent price `NaN`, and `$T` cannot carry one, so a gap or a padded observation cannot be stated in it. Hand the layer a panel whose value type carries an absence, or fill every gap and align every series before the call."))
    return x
end
"""
    assert_no_infinite_price(v::AbstractArray, names, ts) -> nothing

Refuse an infinite value in a price series by name.

The layer spells an absent price `NaN`, and an infinity is not that spelling. Two pieces of the layer read an infinity differently. [`listing_span`](@ref) reads a non-finite cell as unpriced, so the **Span Rule** keeps an interior infinity inside the listing. The conversion reads the same cell as a price, and the return that follows it is a finite `-100 %`. This function refuses the value where the layer fixes the spelling, so every later piece reads one definition of a gap.

# Algorithm

 1. Find the first value that is not `missing`, not `NaN` and not finite, giving `k`.
 2. No such value: return.
 3. Read the observation and the column of `k` from its Cartesian index. A one-column series can hold its values in a vector, and then the column is the first.
 4. Raise a `DomainError` that names the value, the column and the timestamp.

# Arguments

  - `v`: The values of one series, `observations × columns`.
  - `names`: The series' column names.
  - `ts`: The series' timestamps.

# Validation

  - Every value is `missing`, `NaN`, or finite. Raises a `DomainError` naming the first offending column and observation.

# Returns

  - `nothing`.

# Related

  - [`unify_gaps`](@ref)
  - [`listing_span`](@ref)
  - [`is_missing_value`](@ref)
"""
function assert_no_infinite_price(v::AbstractArray, names, ts)::Nothing
    k = findfirst(x -> !ismissing(x) && !isnan(x) && !isfinite(x), v)
    if isnothing(k)
        return nothing
    end
    #! A one-column series may hold its values in a vector, so the column is read off the
    #! Cartesian index where there is one and is the first column otherwise.
    idx = Tuple(CartesianIndices(v)[k])
    col = length(idx) == 1 ? 1 : idx[2]
    return throw(DomainError(v[k],
                             "an infinite value is neither a price nor the marker of an absence, and the ingestion layer spells an absent price `NaN`; got $(v[k]) in column `$(names[col])` at $(ts[idx[1]]). Spell the absence `NaN` or `missing`, or correct the price, before the call."))
end
"""
    unify_gaps(A::TimeSeries.TimeArray) -> TimeSeries.TimeArray
    unify_gaps(A::TimeSeries.TimeArray, ::Type{T}) -> TimeSeries.TimeArray

Spell every absent price of one series the one way the ingestion layer carries, in the value type `T`.

A source spells an absent price one of two ways. An outer join of ragged per-asset histories pads with `NaN`, and a wide table built from a tidy one leaves `missing`. The layer unifies both as `NaN`, which no deletion step reads. This defines a gap for every later piece, so it runs before the Span Rule and before the join.

The caller does not name the target type. The one-argument form derives it from the series alone with [`absence_type`](@ref), and the conversion uses this form for a carrier built by hand. [`price_ingestion`](@ref) first promotes the asset, factor and benchmark series to one type, because the join lays them in one table, and passes that type to the two-argument form.

# Algorithm

 1. Refuse an infinite value by name with [`assert_no_infinite_price`](@ref). The **Span Rule** reads a non-finite cell as unpriced and the conversion reads it as a price, so the two would disagree about the cell.
 2. The values are already of type `T`, so their gaps are already `NaN`: return `A` untouched.
 3. At least one value is `missing`: rebuild the values. Map each `missing` to [`absent_value`](@ref)`(T)` and convert every other entry to `T`. This is the one place where a `missing` becomes an absence, so the refusal of a type that cannot hold one happens here, by name.
 4. Otherwise convert every entry to `T`. Nothing is absent, so the function spells nothing and does not ask whether `T` can spell an absence. This includes an element type that admits `missing` when no value is `missing`.

# Arguments

  - `A`: One price series.
  - `T`: The value type to carry it in. Defaults to `absence_type(series_value_type(A))`.

# Validation

  - Every value is `missing`, `NaN`, or finite. An infinity raises a `DomainError` naming the column and the observation.
  - `T` carries an absence when a value is `missing`. Raises a `DomainError` naming `T`, from [`absent_value`](@ref).
  - The one-argument form: the element type of the series is not `Missing` alone. Raises a `DomainError`, from [`absence_type`](@ref).

# Returns

  - `A′::TimeSeries.TimeArray`: The same series in type `T`, with every absent price spelled `NaN`.

# Related

  - [`PriceIngestion`](@ref)
  - [`price_ingestion`](@ref)
  - [`series_value_type`](@ref)
  - [`absence_type`](@ref)
  - [`absent_value`](@ref)
  - [`assert_no_infinite_price`](@ref)
  - [`listing_span`](@ref)
"""
function unify_gaps(A::TimeSeries.TimeArray)
    return unify_gaps(A, absence_type(series_value_type(A)))
end
function unify_gaps(A::TimeSeries.TimeArray, ::Type{T}) where {T}
    v = values(A)
    assert_no_infinite_price(v, TimeSeries.colnames(A), TimeSeries.timestamp(A))
    if eltype(v) === T
        return A
    end
    w = if any(ismissing, v)
        nan = absent_value(T)
        [ismissing(x) ? nan : convert(T, x) for x in v]
    else
        map(Base.Fix1(convert, T), v)
    end
    return TimeSeries.TimeArray(TimeSeries.timestamp(A), w, TimeSeries.colnames(A))
end
"""
    assert_pad_spellable(::Type{T}, method::Symbol) -> nothing

Refuse by name, before a join that can pad, a value type that cannot spell the pad.

`TimeSeries.merge` pads with `NaN` converted to the table's value type, which has the same bits as [`absent_value`](@ref)`(T)`. For a type that cannot hold an absence, `merge` raises an `InexactError` that names nothing. This function asks first, so the refusal names the type. It asks before every join that can pad, whether or not the join then pads a row. An inner join drops an observation and does not pad it, so it asks for nothing.

# Algorithm

 1. `method` is `:inner`: return.
 2. Otherwise ask [`absent_value`](@ref)`(T)`, which refuses by name.

# Arguments

  - `T`: The value type of the tables the layer joins, the layer's unification target.
  - `method`: The join, as `TimeSeries.merge` takes it.

# Validation

  - `T` carries an absence when `method` is not `:inner`. Raises a `DomainError` naming `T`, from [`absent_value`](@ref).

# Returns

  - `nothing`.

# Related

  - [`absent_value`](@ref)
  - [`absence_type`](@ref)
  - [`price_ingestion`](@ref)
"""
function assert_pad_spellable(::Type{T}, method::Symbol) where {T}
    if method != :inner
        absent_value(T)
    end
    return nothing
end
"""
    align_series(A::TimeSeries.TimeArray{T}, ts::AbstractVector) -> TimeSeries.TimeArray

Put one carried series on the clock the ingestion layer emits, and pad the observations where it is silent.

The layer carries the implied volatilities beside the assets and does not join them into the asset table, because they carry the asset names and a join would rename them. The alignment works as a left join works for a factor. The emitted clock decides. The function keeps each observation the series states, and each observation the series does not state becomes an absence that the layer carries.

# Algorithm

 1. Fill a table of `length(ts)` rows and the series' columns with [`absent_value`](@ref) of its type. This asks for an absence even when the series covers the clock, as a join that can pad does.
 2. Write each row of the series whose timestamp is in `ts` at that timestamp's row.

# Arguments

  - `A`: The series to align.
  - `ts`: The emitted clock.

# Validation

  - `T` carries an absence. Raises a `DomainError` naming `T`, from [`absent_value`](@ref).

# Returns

  - `A′::TimeSeries.TimeArray`: The series on `ts`, `NaN` where it was silent.

# Related

  - [`absent_value`](@ref)
  - [`padded_observations`](@ref)
  - [`price_ingestion`](@ref)
"""
function align_series(A::TimeSeries.TimeArray{T}, ts::AbstractVector) where {T}
    v = values(A)
    out = fill(absent_value(T)::T, length(ts), size(v, 2))
    for (r, k) in pairs(indexin(ts, TimeSeries.timestamp(A)))
        if !isnothing(k)
            out[r, :] .= view(v, k, :)
        end
    end
    return TimeSeries.TimeArray(ts, out, TimeSeries.colnames(A))
end
"""
    padded_observations(A::Nothing, ts::AbstractVector) -> Int
    padded_observations(A::TimeSeries.TimeArray, ts::AbstractVector) -> Int

Count the observations of the emitted clock at which one series is silent.

The join or the alignment padded each of those observations, and the layer carries each padded observation as an absence and names it. A series the caller omitted is silent nowhere.

# Algorithm

The method that Julia selects is the algorithm.

 1. `A` is `nothing`: `0`.
 2. `A` is a table: the number of timestamps in `ts` absent from `A`'s.

# Arguments

  - `A`: One price series, or `nothing`.
  - `ts`: The clock the layer emitted.

# Returns

  - `n::Int`: The number of padded observations.

# Related

  - [`padding_report_line`](@ref)
  - [`assert_join_padding`](@ref)
  - [`align_series`](@ref)
"""
function padded_observations(::Nothing, ::AbstractVector)
    return 0
end
function padded_observations(A::TimeSeries.TimeArray, ts::AbstractVector)
    return count(!in(Set(TimeSeries.timestamp(A))), ts)
end
"""
    padding_report_line(name::String, A::Option{<:TimeSeries.TimeArray}, ts::AbstractVector) -> Option{String}

Write the line of the padding report that one series owes, or `nothing` when the layer padded it nowhere.

# Algorithm

 1. Count the padded observations with [`padded_observations`](@ref).
 2. None: return `nothing`.
 3. Otherwise write a line that names the table, the count out of the clock's length, and every column. A series silent at an observation is silent in all its columns.

# Arguments

  - `name`: How the caller spells the series, `"X"`, `"F"`, `"B"` or `"iv"`.
  - `A`: The series, or `nothing`.
  - `ts`: The clock the layer emitted.

# Returns

  - `line::Option{String}`: The report line, or `nothing`.

# Related

  - [`padded_observations`](@ref)
  - [`assert_join_padding`](@ref)
"""
function padding_report_line(::String, ::Nothing, ::AbstractVector)
    return nothing
end
function padding_report_line(name::String, A::TimeSeries.TimeArray, ts::AbstractVector)
    n = padded_observations(A, ts)
    if iszero(n)
        return nothing
    end
    return "`$name` at $n of $(length(ts)) observations, in every column ($(join(TimeSeries.colnames(A), ", ")))"
end
"""
    assert_join_padding(lines::AbstractVector, strict::Bool) -> nothing

Name what the join and the alignment padded, warning by default and refusing under `strict`.

A carried absence raises no error, and the factor, benchmark and implied-volatility axes state no universe that would name it later. So this function names it through [`strict_diagnostic`](@ref), which a Held Gap and [`PriceGapFill`](@ref) also use to report. A caller who runs a walk-forward and sets `strict` knows that the layer padded no covariate.

# Algorithm

 1. Drop the `nothing` entries. None is left, so the layer padded nothing: return.
 2. Otherwise join the lines into one report that names the three join methods, and pass it to [`strict_diagnostic`](@ref).

# Arguments

  - `lines`: The report lines [`padding_report_line`](@ref) wrote, `nothing` entries included.
  - `strict`: If `true`, throws an `ArgumentError`; if `false`, issues a warning.

# Validation

  - The layer padded nothing. Raises an `ArgumentError` under `strict`.

# Returns

  - `nothing`.

# Related

  - [`padding_report_line`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`PriceIngestion`](@ref)
  - [`price_ingestion`](@ref)
"""
function assert_join_padding(lines::AbstractVector, strict::Bool)::Nothing
    kept = String[l for l in lines if !isnothing(l)]
    if isempty(kept)
        return nothing
    end
    strict_diagnostic("the ingestion layer padded the series the join and the alignment found silent on the emitted clock, and a padded observation is an absence carried as `NaN` on an axis that states no universe: " *
                      join(kept, "; ") *
                      ". Under `join_method = :left` the asset table states the clock; `:outer` takes the union and `:inner` the intersection, and a caller who wants no padding aligns the series before the call.",
                      strict)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Estimator that assembles raw price series into the price carrier and its Listing Span, which the conversion then reads.

`PriceIngestion` runs once on the whole panel, and it is not a [`Pipeline`](@ref) step. The observation clock decides this. The factor and benchmark join can add or drop observations, and the frequency collapse renumbers them. Cross-validation cuts its folds once on the carrier's clock. A hyperparameter that moves the clock changes the test set, and a score on one test set cannot be compared with a score on another. A step that changes only values leaves the clock alone, so it stays inside the `Pipeline`.

Because it runs outside the `Pipeline`, the **Span Rule** can read the whole panel. A step sees only a window, and a span derived from a window reads a delisting that straddles the window end as an asset that was never listed. A split forbids work on the data before it, but a listing calendar is a fact about the instruments and not an estimate from returns, so the panel-wide read is valid.

The carrier holds the derived **Listing Span** in its `span` field. The conversion projects it onto the returns clock and gives the returns carrier an [`AssetPanel`](@ref) that states the universe. A caller who holds a listing calendar passes it as `span`, and it replaces the Span Rule's answer.

**The asset table states the clock.** It states the universe, so it also states the clock. Under the default `join_method = :left`, the layer aligns the factor, benchmark and implied-volatility series to that clock and pads them where they are silent. The layer names what it padded: the table, the number of observations and the columns. It warns by default and refuses under `strict`, because a carried absence raises no error and those axes state no universe that would name it later.

Each field states a fact about the sources that the layer cannot derive. `join_method` states which clock decides, `collapse_args` the frequency the analysis wants, `span` the caller's listing calendar, and `strict` whether a padded covariate is acceptable.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceIngestion(;
        join_method::Symbol = :left,
        collapse_args::Tuple = (),
        span::Option{<:AbstractMatrix{Bool}} = nothing,
        strict::Bool = false,
    ) -> PriceIngestion

Keywords correspond to the struct's fields.

## Validation

  - `join_method in (:left, :outer, :inner)`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [NaN 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = price_ingestion(PriceIngestion(), X);

julia> pr.span
ListingSpan(3 × 2)

julia> Matrix(pr.span)
3×2 Matrix{Bool}:
 0  1
 1  1
 1  1
```

# Related

  - [`price_ingestion`](@ref)
  - [`PricesResult`](@ref)
  - [`PricesToReturns`](@ref)
  - [`listing_span`](@ref)
  - [`universe_masks`](@ref)
  - [`Option`](@ref)
"""
@concrete struct PriceIngestion <: AbstractEstimator
    """
    How the layer joins the factor and benchmark series onto the asset clock, as `TimeSeries.merge` takes it. `:left` keeps the asset clock and pads the others where they are silent. `:outer` takes the union of the clocks and can pad every table. `:inner` takes their intersection and pads none.
    """
    join_method
    """
    Arguments for collapsing the joined series to a lower frequency, as `TimeSeries.collapse` takes them. An empty tuple leaves the clock alone.
    """
    collapse_args
    """
    Optional listing statement of the caller's own, `price observations × assets`. It replaces the answer of the **Span Rule**. Any `AbstractMatrix{Bool}` is valid, for example a listing calendar, or a constituency that leaves and rejoins. `nothing` derives the span from the gaps.
    """
    span
    """
    Whether a padded observation refuses the ingestion. `false` names what the join and the alignment padded in a warning, and `true` raises an `ArgumentError` with the same report.
    """
    strict
    function PriceIngestion(join_method::Symbol, collapse_args::Tuple,
                            span::Option{<:AbstractMatrix{Bool}}, strict::Bool)
        @argcheck(join_method in (:left, :outer, :inner),
                  ArgumentError("join_method must be :left, :outer or :inner, got $(repr(join_method))"))
        return new{typeof(join_method), typeof(collapse_args), typeof(span),
                   typeof(strict)}(join_method, collapse_args, span, strict)
    end
end
function PriceIngestion(; join_method::Symbol = :left, collapse_args::Tuple = (),
                        span::Option{<:AbstractMatrix{Bool}} = nothing,
                        strict::Bool = false)::PriceIngestion
    return PriceIngestion(join_method, collapse_args, span, strict)
end
"""
    price_ingestion(est::PriceIngestion, X::TimeSeries.TimeArray;
                    F::Option{<:TimeSeries.TimeArray} = nothing,
                    B::Option{<:TimeSeries.TimeArray} = nothing,
                    iv::Option{<:TimeSeries.TimeArray} = nothing,
                    ivpa::Option{<:Num_VecNum} = nothing,
                    pnl::Option{<:AssetPanel} = nothing) -> PricesResult
    price_ingestion(est::PriceIngestion, pr::PricesResult) -> PricesResult

Assemble raw price series into the price carrier and its Listing Span.

The three steps that move the clock run here and not in a [`Pipeline`](@ref), and the **Span Rule** reads the whole panel after them. The result is an ordinary [`PricesResult`](@ref). Its `span` field states which assets are listed at each observation, and [`PricesToReturns`](@ref) projects it onto the returns clock.

The layer spells an absent price `NaN`, the library's one spelling for absence. The conversion carries it into the returns and does not delete the observation or the asset that holds it. Every axis the layer touches carries its absences. If a factor, benchmark or implied-volatility series is silent at an observation of the emitted clock, the layer pads it there and names the padding. It warns by default and refuses under `strict`, because those axes state no universe that would name the padding later.

The caller does not name the value type of the carrier. The layer promotes the asset, factor and benchmark types to one type, because the join lays them in one table. It then widens that type only as the return arithmetic would widen it. A `Float32` panel stays `Float32`, `Float32` beside `Float64` joins in `Float64`, and an integer panel takes the floating-point type that represents it. The layer refuses by name a type that cannot spell an absence wherever it might have to spell one: at a `missing` value, before a join that can pad, and when it aligns the implied volatilities. The join and the alignment ask before they know whether they pad a row.

# Algorithm

 1. Check with [`assert_distinct_series_names`](@ref) that the asset, factor and benchmark series keep their names through the join. The join renames a name that two tables share, and a block would then take another block's column.
 2. Promote the value types that the three price tables contribute through [`series_value_type`](@ref), and widen the result with [`absence_type`](@ref), giving the target type. Unify the absent-price spelling of the three tables in that type with [`unify_gaps`](@ref), so a gap means one thing from here on.
 3. Join the factor and the benchmark series onto the asset clock under `join_method`. A left join keeps the asset clock and pads the others where they are silent. An outer join adds the rows that one series has and another does not, and can pad every table. An inner join keeps the rows that every table has. Before a join that can pad, ask [`assert_pad_spellable`](@ref), which refuses by name a type that cannot spell the pad.
 4. Write the padding report lines of the asset, factor and benchmark tables against the joined clock with [`padding_report_line`](@ref), before the collapse renumbers it.
 5. Collapse the joined series to a lower frequency when `collapse_args` is not empty. This renumbers every observation.
 6. Split the joined series back into the asset, factor and benchmark blocks, all on one clock.
 7. Align the implied volatilities to that clock with [`align_series`](@ref), in the type that [`absence_type`](@ref) derives from their own type, and pad the observations where they are silent. Write their report line, and carry `ivpa` through. An implied volatility is a volatility and not a price, so the layer carries it and never converts it to a return.
 8. Name what the layer padded with [`assert_join_padding`](@ref), which warns, or refuses under `strict`.
 9. Put a caller's [`AssetPanel`](@ref) on the emitted clock with [`project_panel_clock`](@ref). A caller states a Panel Field on the clock of the table they hold, and the join and the collapse move that clock, so the projection happens here. Each period of a collapse takes the Panel Field values of the last row of its group, as [`LastObservation`](@ref) does. The timestamp function of `collapse_args` does not change this row, so under `(Dates.week, first, last)` the prices and the Panel Field values of a week come from one day. A static panel has no observation axis, and the layer carries it through unchanged.
10. Read the **Listing Span** off the asset block with [`listing_span`](@ref), unless the caller declared one. A declared span replaces the answer of the Span Rule.
11. Return the [`PricesResult`](@ref) that carries all of it.

Under the default join the emitted clock is the asset table's, so `timestamp(pr.X) == timestamp(X)` unless `collapse_args` is not empty. A caller who declares a listing calendar can therefore size it against the table they hold.

# Arguments

  - `est`: The [`PriceIngestion`](@ref) estimator.
  - `X`: Asset prices, `observations × assets`.
  - `F`: Optional factor prices.
  - `B`: Optional benchmark prices, one column or one per asset.
  - `iv`: Optional implied volatilities, one column per asset, on any clock. The layer aligns them to the emitted clock and pads `NaN` where they are silent.
  - `ivpa`: Optional implied volatility adjustment.
  - `pnl`: Optional [`AssetPanel`](@ref) of Panel Fields the caller already holds. The layer keeps its Panel Fields and does not read its masks. A time-varying panel carries an `amsk` by construction, so that field cannot carry a declaration. A listing statement goes in `span`.
  - `pr`: A [`PricesResult`](@ref), for the second form. The layer ingests its series again. It keeps a span the carrier states unless `est.span` overrides it, and the Span Rule runs only where neither states one.

# Validation

  - `!isempty(X)`. Raises an [`IsEmptyError`](@ref).
  - The asset, factor and benchmark column names are pairwise disjoint, and none of them is `timestamp`. Raises a [`ConflictingArgumentError`](@ref) naming the offending columns.
  - At least one of `X`, `F` and `B` has an element type that is not `Missing` alone. Raises a `DomainError`, from [`absence_type`](@ref).
  - A declared `span` has the size of the asset block after the join and the collapse. Raises a `DimensionMismatch`.
  - The value type carries an absence wherever the layer might spell one. Raises a `DomainError` naming the type, from [`absent_value`](@ref).
  - `pnl` describes the assets of `X`, and a time-varying `pnl` also its observations. Raises a `DimensionMismatch`, from [`project_panel_clock`](@ref).
  - Each row that a time-varying `pnl` takes is a row of `X`. Without a collapse, this is each emitted row. With a collapse, it is the last joined row of each period. An outer join that adds a row breaks this. Raises an `ArgumentError`, from [`matched_row_indices`](@ref).
  - The layer padded nothing, under `strict`. Raises an `ArgumentError` carrying the padding report, from [`assert_join_padding`](@ref). Otherwise the layer warns.

# Returns

  - `pr::PricesResult`: The price carrier, holding the Listing Span in its `span` field.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 4),
                     [100.0 missing; 102.0 103.0; 101.0 104.0; 103.0 102.0], [\"A\", \"B\"]);

julia> pr = price_ingestion(PriceIngestion(), X);

julia> values(pr.X)[1, :]
2-element Vector{Float64}:
 100.0
 NaN

julia> Matrix(pr.span)
4×2 Matrix{Bool}:
 1  0
 1  1
 1  1
 1  1
```

# Related

  - [`PriceIngestion`](@ref)
  - [`PricesResult`](@ref)
  - [`PricesToReturns`](@ref)
  - [`listing_span`](@ref)
  - [`unify_gaps`](@ref)
  - [`series_value_type`](@ref)
  - [`absence_type`](@ref)
  - [`assert_pad_spellable`](@ref)
  - [`align_series`](@ref)
  - [`padding_report_line`](@ref)
  - [`assert_join_padding`](@ref)
  - [`project_panel_clock`](@ref)
  - [`LastObservation`](@ref)
"""
function price_ingestion(est::PriceIngestion, X::TimeSeries.TimeArray;
                         F::Option{<:TimeSeries.TimeArray} = nothing,
                         B::Option{<:TimeSeries.TimeArray} = nothing,
                         iv::Option{<:TimeSeries.TimeArray} = nothing,
                         ivpa::Option{<:Num_VecNum} = nothing,
                         pnl::Option{<:AssetPanel} = nothing)::PricesResult
    @argcheck(!isempty(X),
              IsEmptyError("`X` cannot be empty: the ingestion layer states a universe from a price panel"))
    assert_distinct_series_names(X, F, B)
    nx = TimeSeries.colnames(X)
    #! The three price tables are laid in one table by the join, so they take one value
    #! type, and it is the one the return arithmetic would widen them to: derived from the
    #! series, never named.
    T = absence_type(promote_type(series_value_type(X), series_value_type(F),
                                  series_value_type(B)))
    M = unify_gaps(X, T)
    if !isnothing(F)
        assert_pad_spellable(T, est.join_method)
        M = TimeSeries.merge(M, unify_gaps(F, T); method = est.join_method)
    end
    if !isnothing(B)
        assert_pad_spellable(T, est.join_method)
        M = TimeSeries.merge(M, unify_gaps(B, T); method = est.join_method)
    end
    #! The report reads the joined clock before the collapse renumbers it, because the
    #! join is what padded, and it names the asset table too: an outer join pads it with
    #! observations a covariate's length invented, which the Span Rule then reads.
    tsj = TimeSeries.timestamp(M)
    lines = Option{String}[padding_report_line("X", X, tsj),
                           padding_report_line("F", F, tsj),
                           padding_report_line("B", B, tsj)]
    if !isempty(est.collapse_args)
        M = TimeSeries.collapse(M, est.collapse_args...)
    end
    Xa = M[nx]
    Fa = isnothing(F) ? nothing : M[TimeSeries.colnames(F)]
    Ba = isnothing(B) ? nothing : M[TimeSeries.colnames(B)]
    ts = TimeSeries.timestamp(Xa)
    iva = if isnothing(iv)
        nothing
    else
        push!(lines, padding_report_line("iv", iv, ts))
        align_series(unify_gaps(iv, absence_type(series_value_type(iv))), ts)
    end
    assert_join_padding(lines, est.strict)
    #! The collapse is the one step here that renumbers an observation, and a Panel Field is
    #! stated on the clock of the table the caller holds, so the projection is owed here
    #! rather than by the conversion: after the door, the carrier states one clock and
    #! everything it carries is on it.
    pnl = project_panel_clock(pnl, tsj, X, est.collapse_args)
    span = isnothing(est.span) ? listing_span(values(Xa)) : est.span
    return PricesResult(; X = Xa, F = Fa, B = Ba, iv = iva, ivpa = ivpa, pnl = pnl,
                        span = span)
end
function price_ingestion(est::PriceIngestion, pr::PricesResult)::PricesResult
    #! A carrier's span is a declaration already made, and a declaration is never
    #! second-guessed: the estimator's own `span` overrides it, and the Span Rule runs
    #! only where neither states one. A declared span is on the clock the caller handed
    #! in, so a collapse that moves the clock refuses it by shape at the carrier.
    span = isnothing(est.span) ? pr.span : est.span
    est = PriceIngestion(; join_method = est.join_method, collapse_args = est.collapse_args,
                         span = span, strict = est.strict)
    return price_ingestion(est, pr.X; F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                           pnl = pr.pnl)
end
"""
    project_panel_clock(pnl::Nothing, tsj, X::TimeSeries.TimeArray, ca::Tuple) -> nothing
    project_panel_clock(pnl::AssetPanel, tsj, X::TimeSeries.TimeArray, ca::Tuple) -> AssetPanel

Put a caller's [`AssetPanel`](@ref) on the clock the ingestion emits.

A caller states a Panel Field on the clock of the table they hold. The join and the collapse of [`price_ingestion`](@ref) move that clock, so the projection happens at the door of the layer. After it, the carrier states one clock and everything it holds is on that clock. A static panel has no observation axis, so [`feature_row_indices`](@ref) returns `Colon()` for it and the panel passes through unchanged.

A collapse puts many rows into one period. Each period takes the Panel Field values of the last row of its group, as [`LastObservation`](@ref) does. The timestamp function of `ca` does not change this row. So under `(Dates.week, first, last)` a week pairs the prices of its last day with the Panel Field values of the same day.

# Algorithm

 1. `pnl` is `nothing`: return `nothing`.
 2. Otherwise read the incoming clock `ts_old` and the asset names off `X`, and check the panel against them with [`check_asset_panel`](@ref).
 3. Find the timestamp of the row that each emitted observation takes, giving `ts_row`. Without a collapse, `ts_row` is `tsj`. With a collapse, collapse the row positions of `tsj` with the period and timestamp functions of `ca` and the value function `last`. This gives the last row of each group, as `TimeSeries.collapse` forms the groups, and `ts_row` holds the timestamps of those rows.
 4. Find the rows of `ts_row` in the incoming clock with [`feature_row_indices`](@ref), and view the panel over them with [`port_opt_view`](@ref). The view takes the whole asset axis and its names, so that it also cuts a square tensor Panel Field on its label axis.

# Arguments

  - `pnl`: The caller's [`AssetPanel`](@ref), or `nothing`.
  - `tsj`: The timestamps of the joined table, before the collapse.
  - `X`: The asset table the caller passed. Its timestamps are the incoming clock, and its column names are the asset names.
  - `ca`: The `collapse_args` of the [`PriceIngestion`](@ref). An empty tuple states no collapse.

# Validation

  - The panel describes the assets of `X`, and a time-varying panel also its observations. Raises a `DimensionMismatch`.
  - Every entry of `ts_row` is a timestamp of `X`, when the panel is time-varying. Raises an `ArgumentError`, from [`matched_row_indices`](@ref).

# Returns

  - `pnl′::Option{<:AssetPanel}`: The panel on the emitted clock, or `nothing`.

# Related

  - [`price_ingestion`](@ref)
  - [`AssetPanel`](@ref)
  - [`feature_row_indices`](@ref)
  - [`port_opt_view`](@ref)
  - [`check_asset_panel`](@ref)
  - [`LastObservation`](@ref)
"""
function project_panel_clock(::Nothing, ::Any, ::TimeSeries.TimeArray, ::Tuple)
    return nothing
end
function project_panel_clock(pnl::AssetPanel, tsj, X::TimeSeries.TimeArray,
                             ca::Tuple)::AssetPanel
    ts_old = TimeSeries.timestamp(X)
    nx = string.(TimeSeries.colnames(X))
    check_asset_panel(pnl, length(nx), length(ts_old), "the number of asset price columns")
    #! The collapse of the row positions groups them as the collapse of the prices does, so
    #! each period takes its last row whatever the timestamp and value functions pick.
    ts_row = if isempty(ca)
        tsj
    else
        tsj[values(TimeSeries.collapse(TimeSeries.TimeArray(tsj, collect(eachindex(tsj))),
                                       ca[1], ca[2], last))]
    end
    return port_opt_view(pnl, feature_row_indices(pnl, ts_row, ts_old),
                         collect(eachindex(nx)), nx)
end
"""
    span_carrier_view(span::Nothing, ts_new, ts_old, j) -> nothing
    span_carrier_view(span::AbstractMatrix{Bool}, ts_new, ts_old, j) -> SubArray
    span_carrier_view(span::ListingSpan, ts_new, ts_old, j) -> Union{ListingSpan, SubArray}

View a price carrier's Listing Span over the surviving timestamps and the assets `j`, or return `nothing` when the carrier holds none.

The span states a fact about the instruments, so the span of a window is a view of the panel-wide span, never a span derived again from the window. A span derived from a window reads a delisting that straddles the window end as an asset that was never listed. A view is exact, and over a [`PortfolioOptimisers.ListingSpan`](@ref) it stores no boolean cell.

The span is parallel to the price clock by position. So the function finds its rows from the surviving timestamps with [`matched_row_indices`](@ref), as it does for a time-varying [`AssetPanel`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: return `nothing`, and match no timestamp.
 2. `span` is a matrix: find its rows with [`matched_row_indices`](@ref), and view it at those rows and the assets `j`.
 3. `span` is a [`PortfolioOptimisers.ListingSpan`](@ref) and the rows are a contiguous window in clock order, for example a fold's window or the whole clock. Subtract the number of rows before the window from both bounds, keep the bounds of the assets `j`, and return a new `ListingSpan` of the window's length. The result keeps two integers per asset and does not expand them into a view of booleans. A bound outside the window states the same listings inside it as a bound at its edge. Any other row selection can split an interval in two, which one interval cannot state, so it takes step 2.

# Arguments

  - `span`: The Listing Span on the price clock, or `nothing`.
  - `ts_new`: Timestamps that survived the selection.
  - `ts_old`: Timestamps of the price clock the span is parallel to.
  - `j`: Asset indices, a vector or `Colon()`.

# Returns

  - `s::Option{<:AbstractMatrix{Bool}}`: A new `ListingSpan` for a contiguous window of a `ListingSpan`, a view of `span` for any other selection, or `nothing`.

# Related

  - [`PricesResult`](@ref)
  - [`PortfolioOptimisers.ListingSpan`](@ref)
  - [`listing_span`](@ref)
  - [`matched_row_indices`](@ref)
  - [`port_opt_view`](@ref)
"""
function span_carrier_view(::Nothing, ::Any, ::Any, ::Any)
    return nothing
end
function span_carrier_view(span::AbstractMatrix{Bool}, ts_new, ts_old, j)
    return view(span, matched_row_indices(ts_new, ts_old), j)
end
function span_carrier_view(span::ListingSpan, ts_new, ts_old, j)
    i = matched_row_indices(ts_new, ts_old)
    #! A contiguous row window in clock order cuts every interval to an interval, so the
    #! two integers per asset survive the cut: the bounds shift by the rows the window
    #! dropped in front, and a bound that falls outside the window says the same thing
    #! there as a bound at its edge. A fold's window is this case. Any other selection
    #! can split an interval in half, which no interval can say, and falls back to the
    #! ordinary view.
    return if !isempty(i) && i == first(i):last(i)
        o = first(i) - 1
        ListingSpan(span.first[j] .- o, span.last[j] .- o, length(i))
    else
        view(span, i, j)
    end
end
"""
    assert_span_shape(span::Nothing, nobs, na) -> nothing
    assert_span_shape(span::AbstractMatrix{Bool}, nobs::Integer, na::Integer) -> nothing

Check that a Listing Span has the shape of the price panel it describes.

A span states which assets are listed at each observation of the price clock, so it has the shape of the asset prices. A carrier that holds no span states no universe, and there is nothing to check.

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: the carrier states no universe, so there is nothing to check.
 2. Otherwise check the span's shape against the price panel's.

# Arguments

  - `span`: The Listing Span on the price clock, or `nothing`.
  - `nobs`: Observation count of the price panel.
  - `na`: Asset count of the price panel.

# Validation

  - `size(span) == (nobs, na)`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`prices_to_returns`](@ref)
  - [`PricesToReturns`](@ref)
  - [`price_ingestion`](@ref)
  - [`returns_universe_masks`](@ref)
"""
function assert_span_shape(::Nothing, ::Any, ::Any)::Nothing
    return nothing
end
function assert_span_shape(span::AbstractMatrix{Bool}, nobs::Integer, na::Integer)::Nothing
    @argcheck(size(span) == (nobs, na),
              DimensionMismatch("a Listing Span states which assets are listed at each observation of the price clock, so it is the shape of the asset prices; got size(span) = $(size(span)) and $nobs × $na prices"))
    return nothing
end
"""
    series_names(A::Nothing) -> Vector{Symbol}
    series_names(A::TimeSeries.TimeArray) -> Vector{Symbol}

Name the columns of one optional price table, and name none when the caller passed none.

A series the caller omitted contributes no column name. It contributes an empty list, so no site that reads the names needs an `isnothing` branch.

# Algorithm

The method that Julia selects is the algorithm.

 1. `A` is `nothing`: return an empty name list.
 2. `A` is a table: return its column names.

# Arguments

  - `A`: One price table, or `nothing`.

# Returns

  - `n::Vector{Symbol}`: The table's column names, empty when there is no table.

# Related

  - [`assert_distinct_series_names`](@ref)
  - [`price_ingestion`](@ref)
  - [`prices_to_returns`](@ref)
"""
function series_names(::Nothing)
    return Symbol[]
end
function series_names(A::TimeSeries.TimeArray)
    return TimeSeries.colnames(A)
end
"""
    assert_disjoint_series_names(a::AbstractVector{Symbol}, b::AbstractVector{Symbol}, na::String, nb::String) -> nothing

Refuse a column name that two of the price tables both carry.

A column name says which series a column came from. The layer merges the asset, factor and benchmark tables onto one clock and then splits the blocks by name. `TimeSeries.merge` renames the second of two columns that share a name by appending `_1`. So a shared name makes one block take the other's column with no error, and the new name belongs to no block. The conversion also cannot choose between the two columns. `X`'s `AAPL` and `F`'s `AAPL` are different series, and to keep either one is worse than to refuse both.

# Arguments

  - `a`: Column names of the first table.
  - `b`: Column names of the second table.
  - `na`: How the caller spells the first table.
  - `nb`: How the caller spells the second table.

# Validation

  - `isdisjoint(a, b)`. Raises a [`ConflictingArgumentError`](@ref) naming the shared columns.

# Returns

  - `nothing`.

# Related

  - [`assert_distinct_series_names`](@ref)
  - [`assert_unreserved_series_names`](@ref)
  - [`series_names`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
function assert_disjoint_series_names(a::AbstractVector{Symbol}, b::AbstractVector{Symbol},
                                      na::String, nb::String)::Nothing
    shared = intersect(a, b)
    @argcheck(isempty(shared),
              ConflictingArgumentError("the asset, factor and benchmark series are joined onto one clock and a column name is what says which series a column came from, so `$na` and `$nb` cannot share one; both carry $(shared). Rename the colliding columns before the call."))
    return nothing
end
"""
    assert_unreserved_series_names(n::AbstractVector{Symbol}, nn::String) -> nothing

Refuse a series named after the observation clock.

The conversion writes the clock into a column named `timestamp`. A series with the same name then becomes `timestamp_1`, and the clock keeps the name. So a lookup of the series by its name reads the dates as prices. The layer owns the name, and a caller who holds a series with that name must rename it.

# Arguments

  - `n`: Column names of one table.
  - `nn`: How the caller spells that table.

# Validation

  - `:timestamp ∉ n`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_distinct_series_names`](@ref)
  - [`assert_disjoint_series_names`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
function assert_unreserved_series_names(n::AbstractVector{Symbol}, nn::String)::Nothing
    @argcheck(:timestamp ∉ n,
              ConflictingArgumentError("the conversion writes the observation clock into a column named `timestamp`, so a series cannot carry that name; `$nn` does. Rename it before the call."))
    return nothing
end
"""
    assert_distinct_series_names(X::TimeSeries.TimeArray, F::Option{<:TimeSeries.TimeArray} = nothing, B::Option{<:TimeSeries.TimeArray} = nothing) -> nothing

Check that every price series reaching the layer can still be named after the join.

Both doors of the layer run this check before they merge, [`price_ingestion`](@ref) and [`prices_to_returns`](@ref). After it, every later piece can split the merged table by name. Each name belongs to exactly one of the asset, factor and benchmark blocks, and no name is the clock's.

This function cannot check two series of one table. The `TimeSeries.TimeArray` constructor runs `TimeSeries.replace_dupes!` over its column names, so it renames the duplicates of a table before the table exists, and no duplicate reaches this function.

# Algorithm

 1. Read the three name lists, an absent table naming none, with [`series_names`](@ref).
 2. Refuse a name shared by two of them with [`assert_disjoint_series_names`](@ref), over all three pairs.
 3. Refuse the clock's own name in any of them with [`assert_unreserved_series_names`](@ref).

# Arguments

  - `X`: Asset prices, `observations × assets`.
  - `F`: Optional factor prices.
  - `B`: Optional benchmark prices.

# Validation

  - The asset, factor and benchmark names are pairwise disjoint. Raises a [`ConflictingArgumentError`](@ref) naming the shared columns.
  - None of them is `timestamp`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`assert_disjoint_series_names`](@ref)
  - [`assert_unreserved_series_names`](@ref)
  - [`series_names`](@ref)
  - [`price_ingestion`](@ref)
  - [`prices_to_returns`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
function assert_distinct_series_names(X::TimeSeries.TimeArray,
                                      F::Option{<:TimeSeries.TimeArray} = nothing,
                                      B::Option{<:TimeSeries.TimeArray} = nothing)::Nothing
    nx = TimeSeries.colnames(X)
    nf = series_names(F)
    nb = series_names(B)
    assert_disjoint_series_names(nx, nf, "X", "F")
    assert_disjoint_series_names(nx, nb, "X", "B")
    assert_disjoint_series_names(nf, nb, "F", "B")
    assert_unreserved_series_names(nx, "X")
    assert_unreserved_series_names(nf, "F")
    assert_unreserved_series_names(nb, "B")
    return nothing
end
"""
    compress_all_true(amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool})

Store a pair of universe masks that is true everywhere as the constant it is.

A gapless ingestion states a universe in which every asset is listed at every observation and every return is finite, so both masks are `true` everywhere. A [`PortfolioOptimisers.AllTrueMask`](@ref) stores that in two integers. So the panel that the layer always emits costs `O(1)` memory, not one bit per observation and asset. The estimation mask is a subset of the active mask. When the estimation mask is all true, the active mask is all true too, so the function tests the estimation mask alone.

# Algorithm

 1. Some entry of `emsk` is `false`: return the two masks unchanged.
 2. Otherwise return one [`PortfolioOptimisers.AllTrueMask`](@ref) of the same size as both masks.

# Arguments

  - `amsk`: The active mask.
  - `emsk`: The estimation mask.

# Returns

  - `(amsk, emsk)`: The two masks, compressed when they admit it.

# Related

  - [`PortfolioOptimisers.AllTrueMask`](@ref)
  - [`universe_masks`](@ref)
  - [`returns_universe_masks`](@ref)
  - [`AssetPanel`](@ref)
"""
function compress_all_true(amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool})
    if !all(emsk)
        return amsk, emsk
    end
    msk = AllTrueMask(size(emsk, 1), size(emsk, 2))
    return msk, msk
end
"""
    returns_universe_masks(span::Nothing, R::AbstractMatrix)
    returns_universe_masks(span::AbstractMatrix{Bool}, R::AbstractMatrix)

Derive the two universe masks the returns carrier's [`AssetPanel`](@ref) states.

A carrier from the layer arrives with a **Listing Span**. This function projects the span onto the returns clock and intersects it with finiteness. A carrier built outside the layer arrives without a span and states **no universe**, whether or not its prices hold a gap. The function does not derive a span for it, because a derivation from one window reads a delisting that straddles the window end as an asset that was never listed. The library still handles the gaps of such a carrier, because with no panel the Coverage Universe reads finiteness alone. So `pnl === nothing` means only that the layer did not build the carrier.

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: return `nothing, nothing`.
 2. `span` is given: project and intersect it with [`universe_masks`](@ref), then compress the pair with [`compress_all_true`](@ref).

# Arguments

  - `span`: The Listing Span over the surviving price rows and assets, or `nothing`.
  - `R`: The returns panel the conversion produced, `observations × assets`.

# Returns

  - `(amsk, emsk)`: The two masks, or `nothing, nothing` when the carrier states no universe.

# Related

  - [`universe_masks`](@ref)
  - [`compress_all_true`](@ref)
  - [`attach_universe_masks`](@ref)
  - [`prices_to_returns`](@ref)
"""
function returns_universe_masks(::Nothing, ::AbstractMatrix)
    return nothing, nothing
end
function returns_universe_masks(span::AbstractMatrix{Bool}, R::AbstractMatrix)
    return compress_all_true(universe_masks(span, R)...)
end
"""
    attach_universe_masks(pnl, amsk::Nothing, emsk::Nothing)
    attach_universe_masks(pnl::Nothing, amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool}) -> AssetPanel
    attach_universe_masks(pnl::AssetPanel, amsk::AbstractMatrix{Bool}, emsk::AbstractMatrix{Bool}) -> AssetPanel

Put the two universe masks onto the [`AssetPanel`](@ref) the returns carrier holds.

The layer emits one carrier, and the masks go on it. A [`Pipeline`](@ref) step has one output, and [`PricesToReturns`](@ref) cannot emit the masks by another route without putting them on the price clock, where the masks are not stated. Because the carrier holds the masks, [`port_opt_view`](@ref) slices the universe together with the returns and needs no extra code.

The function keeps a caller's Panel Fields and replaces both masks with the layer's. The active mask cannot be a declaration that a panel brings to the door. A time-varying panel holds an `amsk` by construction, and [`asset_panel`](@ref) writes an all-true one when the caller states none. So an all-true `amsk` beside a time-varying Panel Field looks the same as a declared calendar in which every asset is listed. The one door for a listing statement is `span` on [`PriceIngestion`](@ref). The estimation mask states a fact about the data and not about the instruments, so the conversion derives it again in every case with [`universe_masks`](@ref), and `emsk ⊆ amsk` holds by construction.

# Algorithm

The method that Julia selects is the algorithm.

 1. No masks: return the panel unchanged. The carrier states no universe, so `pnl === nothing` means only that the layer did not build the carrier.
 2. Masks and no panel: return an [`AssetPanel`](@ref) of the two masks and no Panel Field. This is the layer's common case: a caller holding only prices has no feature data.
 3. Masks and a panel: keep its Panel Fields and replace both its masks. A static panel's fields carry no observation axis, so the function first lifts them onto the clock of the masks with [`panel_field_lift`](@ref).

# Arguments

  - `pnl`: The Asset Panel the price carrier held, or `nothing`.
  - `amsk`: The active mask, or `nothing`.
  - `emsk`: The estimation mask, or `nothing`.

# Returns

  - `pnl′::Option{<:AssetPanel}`: The panel the returns carrier holds.

# Related

  - [`AssetPanel`](@ref)
  - [`asset_panel`](@ref): writes the all-true `amsk` a time-varying build carries when the caller states none.
  - [`PriceIngestion`](@ref): its `span` is the door for a listing statement.
  - [`returns_universe_masks`](@ref)
  - [`panel_field_lift`](@ref)
  - [`prices_to_returns`](@ref)
"""
function attach_universe_masks(pnl, ::Nothing, ::Nothing)
    return pnl
end
function attach_universe_masks(::Nothing, amsk::AbstractMatrix{Bool},
                               emsk::AbstractMatrix{Bool})::AssetPanel
    return AssetPanel(; amsk = amsk, emsk = emsk)
end
function attach_universe_masks(pnl::AssetPanel, amsk::AbstractMatrix{Bool},
                               emsk::AbstractMatrix{Bool})::AssetPanel
    pf = if panel_is_static(pnl)
        [panel_field_lift(f, size(amsk, 1)) for f in pnl.pf]
    else
        pnl.pf
    end
    return AssetPanel(; pf = pf, amsk = amsk, emsk = emsk)
end

export PriceIngestion, price_ingestion
