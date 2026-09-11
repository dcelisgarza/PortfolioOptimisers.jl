"""
    series_value_type(A::Nothing) -> Type{Union{}}
    series_value_type(A::TimeSeries.TimeArray) -> Type

Read the value type of one optional price series, without the `Missing` a wide table may carry.

The ingestion layer derives its unification target from the series it is handed rather than naming one, and this is what each series contributes to that derivation: the type of its values, with the absence convention stripped, because an absence is spelled by the target and not by the source. A series the caller omitted contributes `Union{}`, which is the identity of `promote_type`, so an absent table is not a branch at the site that promotes.

# Algorithm

The method that Julia selects is the algorithm.

 1. `A` is `nothing`: return `Union{}`.
 2. `A` is a table: return the element type of its values with `Missing` removed.

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

Derive the type the ingestion layer carries a series of value type `T` in.

The layer spells an absent price `NaN`, so the type it carries a series in must be able to hold one, and that type is read off the arithmetic rather than named: a return divides one price by another, so `oneunit(T) / one(T)` is the widening the conversion applies to every price anyway, and its type is the widest the layer needs. A floating-point type is its own answer, so a `Float32` panel stays `Float32`; an integer panel takes the floating-point type that represents it; a number type the library has never seen takes whatever its own division returns. Whether that type can hold an absence is not decided here, because a series with no gap and no padding never spells one: [`absent_value`](@ref) refuses by name at the moment an absence is written.

# Arguments

  - `T`: The value type of a series, as [`series_value_type`](@ref) reads it.

# Returns

  - `S::Type`: `typeof(oneunit(T) / one(T))`.

# Related

  - [`series_value_type`](@ref)
  - [`absent_value`](@ref)
  - [`unify_gaps`](@ref)
  - [`price_ingestion`](@ref)
"""
function absence_type(::Type{T}) where {T}
    return typeof(oneunit(T) / one(T))
end
"""
    absent_value(::Type{T}) -> T

The one spelling of an absent value in type `T`, or a refusal by name when `T` cannot carry one.

The layer has one spelling for absence, `NaN`, and a type that cannot represent it cannot state a gap. A `Rational` panel holding a gap, or an integer panel joined onto a longer clock, is refused here rather than by an `InexactError` from inside a conversion, and the refusal names the type and what the caller can do. The library stays open to number types it has never seen: any type whose `convert` from `NaN` answers a value `isnan` recognises carries an absence, and only a type that cannot is refused.

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
    unify_gaps(A::TimeSeries.TimeArray) -> TimeSeries.TimeArray
    unify_gaps(A::TimeSeries.TimeArray, ::Type{T}) -> TimeSeries.TimeArray

Spell every absent price of one series the one way the ingestion layer carries, in the value type `T`.

A source spells an absent price either way — an outer join of ragged per-asset histories pads with `NaN`, a wide table built from a tidy one leaves `missing` — and the layer unifies them as `NaN`, which no deletion step reads. This is what defines a gap for every later piece, so it runs before the Span Rule and before the join.

The target type is derived, never named. The one-argument form reads it off the series alone with [`absence_type`](@ref), which is what a carrier built by hand gets at the conversion; [`price_ingestion`](@ref) promotes the asset, factor and benchmark series to one type first, because the join lays them in one table, and hands that type to the two-argument form.

# Algorithm

 1. A series whose values are already of type `T` carries its gaps as `NaN`. Return it untouched.
 2. A series holding `missing` is rebuilt, mapping `missing` to [`absent_value`](@ref)`(T)` and converting every other entry to `T`. This is the one place a `missing` becomes an absence, so a type that cannot carry one is refused here, by name.
 3. Any other series is converted to `T` entry by entry. Nothing is absent, so nothing is spelled, and a type that could not spell one is not asked to.

# Arguments

  - `A`: One price series.
  - `T`: The value type to carry it in. Defaults to `absence_type(series_value_type(A))`.

# Returns

  - `A′::TimeSeries.TimeArray`: The same series in type `T`, with every absent price spelled `NaN`.

# Related

  - [`PriceIngestion`](@ref)
  - [`price_ingestion`](@ref)
  - [`series_value_type`](@ref)
  - [`absence_type`](@ref)
  - [`absent_value`](@ref)
  - [`listing_span`](@ref)
"""
function unify_gaps(A::TimeSeries.TimeArray)
    return unify_gaps(A, absence_type(series_value_type(A)))
end
function unify_gaps(A::TimeSeries.TimeArray, ::Type{T}) where {T}
    v = values(A)
    if eltype(v) === T
        return A
    end
    w = if Missing <: eltype(v)
        nan = absent_value(T)
        [ismissing(x) ? nan : convert(T, x) for x in v]
    else
        map(Base.Fix1(convert, T), v)
    end
    return TimeSeries.TimeArray(TimeSeries.timestamp(A), w, TimeSeries.colnames(A))
end
"""
    assert_pad_spellable(::Type{T}, method::Symbol) -> nothing

Refuse by name, before a join that pads, a value type that cannot spell the pad.

`TimeSeries.merge` pads with `NaN` converted to the table's value type, which is [`absent_value`](@ref)`(T)` to the bit, so a type that cannot carry an absence would be refused from inside it by an `InexactError` that names nothing. Asking first is what makes it a refusal by name. An inner join drops an observation rather than padding it, so it spells no absence and asks for none.

# Algorithm

 1. `method` is `:inner`: return.
 2. Otherwise ask [`absent_value`](@ref)`(T)`, which refuses by name.

# Arguments

  - `T`: The value type of the table being joined, the layer's unification target.
  - `method`: The join, as `TimeSeries.merge` takes it.

# Validation

  - `T` carries an absence when `method` pads. Raises a `DomainError` naming `T`, from [`absent_value`](@ref).

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

Put one carried series on the clock the ingestion layer emits, padding the observations it is silent at.

The implied volatilities are carried beside the assets rather than joined into their table, because they are named after the assets and a join would rename them. They are aligned the way a left join aligns a factor: the emitted clock is authoritative, an observation the series states is taken, and one it does not is an absence the layer carries.

# Algorithm

 1. Fill a table of `length(ts)` rows and the series' columns with [`absent_value`](@ref) of its type.
 2. Write each row of the series whose timestamp is in `ts` at that timestamp's row.

# Arguments

  - `A`: The series to align, in a type `T` that carries an absence.
  - `ts`: The emitted clock.

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

Each of those is an observation the join or the alignment padded, and a padded observation is an absence the layer carries and names. A series the caller omitted is silent nowhere.

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
 3. Otherwise name the table, the count against the clock's length, and every column, because a series silent at an observation is silent in all of its columns.

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

A carried absence is invisible where a refused one was not, and the factor, benchmark and implied-volatility axes carry no universe that would name it later. So the layer names it here, through [`strict_diagnostic`](@ref), which is the shape a Held Gap and a [`PriceGapFill`](@ref) report with. A caller running a walk-forward sets `strict` and is then certain no covariate was padded.

# Algorithm

 1. Drop the `nothing` entries. None left: nothing was padded, return.
 2. Otherwise join the lines into one report naming the join in force and the two others, and hand it to [`strict_diagnostic`](@ref).

# Arguments

  - `lines`: The report lines [`padding_report_line`](@ref) wrote, `nothing` entries included.
  - `strict`: If `true`, throws an `ArgumentError`; if `false`, issues a warning.

# Validation

  - Nothing was padded. Raises an `ArgumentError` under `strict`.

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

Estimator assembling raw price series into the span-carrying price carrier the ingestion layer converts.

`PriceIngestion` runs **once on the whole panel**, and it is deliberately *not* a [`Pipeline`](@ref) step. The rule that puts it outside is the observation clock: unification, the factor and benchmark join, and the frequency collapse each move or renumber the observations, folds are cut once on the carrier's clock, and a hyperparameter that changes the test set cannot be scored against one that does not. A step that only touches values leaves the clock alone and stays inside.

Running outside the `Pipeline` is also what lets the **Span Rule** read the whole panel. A step only ever sees a window, and a window-local span reads a delisting straddling the window end as an asset that was never listed. What licenses the panel-wide read where a split otherwise forbids work before it is that a listing calendar is a fact about the *instruments*, not an estimate from returns.

The carrier it emits holds the derived **Listing Span** in its `span` field, so the conversion can project it onto the returns clock and hand the returns carrier an [`AssetPanel`](@ref) stating the universe. A caller holding their own listing calendar passes it as `span` and replaces the Span Rule's answer outright.

**The asset table states the clock.** It states the universe, so it states the clock, and the factor, benchmark and implied-volatility series are aligned to it under the default `join_method = :left`, padded where they are silent. The layer names what it padded — which table, how many observations, which columns — warning by default and refusing under `strict`, because a carried absence is invisible where a refused one was not and those axes carry no universe that would name it later. Each field states something about the sources the layer cannot derive: which clock is authoritative, what frequency an analysis wants, which listing calendar the caller holds, and whether a padded covariate is acceptable.

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
    How the factor and benchmark series are joined onto the asset clock, as `TimeSeries.merge` takes it. `:left` keeps the asset clock and pads the others where they are silent; `:outer` takes the union of the clocks, padding every table; `:inner` takes their intersection, padding none.
    """
    join_method
    """
    Arguments for collapsing the joined series to a lower frequency, as `TimeSeries.collapse` takes them. Empty leaves the clock alone.
    """
    collapse_args
    """
    Optional listing statement of the caller's own, `price observations × assets`, replacing the **Span Rule**'s answer outright. A listing calendar, or a constituency that leaves and rejoins, is any `AbstractMatrix{Bool}`. `nothing` derives the span from the gaps.
    """
    span
    """
    Whether a padded observation refuses the ingestion. `false` names what the join and the alignment padded in a warning; `true` raises an `ArgumentError` carrying the same report.
    """
    strict
    function PriceIngestion(join_method::Symbol, collapse_args::Tuple,
                            span::Option{<:AbstractMatrix{Bool}}, strict::Bool)
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

Assemble raw price series into the span-carrying price carrier.

The three clock-moving steps run here rather than in a [`Pipeline`](@ref), and the **Span Rule** reads the whole panel once they have. What the caller gets back is an ordinary [`PricesResult`](@ref) whose `span` field states which assets are listed at each observation, ready for [`PricesToReturns`](@ref) to project onto the returns clock.

The layer spells an absent price `NaN`, which is the library's one spelling for absence, and the conversion carries it into the returns rather than deleting the observation or the asset that holds one. An absence is carried on every axis the layer touches: a factor, benchmark or implied-volatility series silent at an observation of the emitted clock is padded there, and the padding is named — warned by default, refused under `strict` — because those axes carry no universe that would name it later.

The value type the carrier holds is derived from the series rather than named. The asset, factor and benchmark types are promoted to one, because the join lays them in one table, and that type is widened only as the return arithmetic would widen it: a `Float32` panel stays `Float32`, `Float32` beside `Float64` joins in `Float64`, and an integer panel takes the floating-point type that represents it. A type that cannot spell an absence is refused by name at the first gap or padded observation it would have to spell.

# Algorithm

 1. Check that the asset, factor and benchmark series can still be named after the join with [`assert_distinct_series_names`](@ref). The join renames a name two tables share, so a block would otherwise be taken apart into another block's column.
 2. Derive the unification target: promote the value types the three price tables contribute through [`series_value_type`](@ref), and widen the result with [`absence_type`](@ref). Unify the absent-price convention of every series in that type with [`unify_gaps`](@ref), so that a gap means one thing from here on.
 3. Join the factor and the benchmark series onto the asset clock under `join_method`. A left join keeps the asset clock and pads the others where they are silent; an outer join adds the rows one series has and another does not, padding every table; an inner join keeps the rows every table has. A join that pads asks [`assert_pad_spellable`](@ref) first, which refuses by name a type that cannot spell the pad.
 4. Write the padding report for the asset, factor and benchmark tables against the joined clock with [`padding_report_line`](@ref), before the collapse renumbers it.
 5. Collapse the joined series to a lower frequency when `collapse_args` is non-empty, which renumbers every observation.
 6. Split the joined series back into their asset, factor and benchmark blocks, all now on one clock.
 7. Align the implied volatilities to that clock with [`align_series`](@ref), in the type [`absence_type`](@ref) derives from their own, padding the observations they are silent at; write their report line; and carry `ivpa` through. An implied volatility is a volatility rather than a price, so it is carried, never converted.
 8. Name what was padded with [`assert_join_padding`](@ref), which warns, or refuses under `strict`.
 9. Put a caller's [`AssetPanel`](@ref) on the emitted clock with [`project_panel_clock`](@ref). A caller states a Panel Field on the clock of the table they hold, and the collapse is the only step here that renumbers it, so this is where it is projected: the aggregated period takes the values of the row at its representative timestamp, which is last-observation semantics and matches [`LastObservation`](@ref). A static panel has no observation axis and is carried through untouched.
10. Read the **Listing Span** off the asset block with [`listing_span`](@ref), unless the caller declared one, in which case theirs is taken outright.
11. Return the [`PricesResult`](@ref) carrying all of it.

The emitted clock is the asset table's under the default join, so `timestamp(pr.X) == timestamp(X)` unless `collapse_args` is non-empty, and a caller declaring their own listing calendar can size it against the table they hold.

# Arguments

  - `est`: The [`PriceIngestion`](@ref) estimator.
  - `X`: Asset prices, `observations × assets`.
  - `F`: Optional factor prices.
  - `B`: Optional benchmark prices, one column or one per asset.
  - `iv`: Optional implied volatilities, one column per asset, on any clock: aligned to the emitted one and padded `NaN` where silent.
  - `ivpa`: Optional implied volatility adjustment.
  - `pnl`: Optional [`AssetPanel`](@ref) of Panel Fields the caller already holds.
  - `pr`: A [`PricesResult`](@ref), for the second form, whose series are re-ingested.

# Validation

  - `!isempty(X)`. Raises an [`IsEmptyError`](@ref).
  - The asset, factor and benchmark column names are pairwise disjoint, and none of them is `timestamp`. Raises a [`ConflictingArgumentError`](@ref) naming the offending columns.
  - A declared `span` is `size(values(X))` after the join and the collapse. Raises a `DimensionMismatch`.
  - The value type carries an absence, wherever one must be spelled. Raises a `DomainError` naming the type, from [`absent_value`](@ref).
  - Nothing was padded, under `strict`. Raises an `ArgumentError` carrying the padding report, from [`assert_join_padding`](@ref); a warning otherwise.

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
    pnl = project_panel_clock(pnl, TimeSeries.timestamp(Xa), TimeSeries.timestamp(X),
                              string.(nx))
    span = isnothing(est.span) ? listing_span(values(Xa)) : est.span
    return PricesResult(; X = Xa, F = Fa, B = Ba, iv = iva, ivpa = ivpa, pnl = pnl,
                        span = span)
end
function price_ingestion(est::PriceIngestion, pr::PricesResult)::PricesResult
    return price_ingestion(est, pr.X; F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                           pnl = pr.pnl)
end
"""
    project_panel_clock(pnl::Nothing, ts_new, ts_old, nx::VecStr) -> nothing
    project_panel_clock(pnl::AssetPanel, ts_new, ts_old, nx::VecStr) -> AssetPanel

Put a caller's [`AssetPanel`](@ref) on the clock the ingestion emits.

A caller states a Panel Field on the clock of the table they hold, and the collapse is the one step of [`price_ingestion`](@ref) that renumbers an observation, so the projection is owed at the door: after it, the carrier states one clock and everything the carrier holds is on it. A static panel has no observation axis, so [`feature_row_indices`](@ref) answers `Colon()` for one and it rides through unchanged.

# Algorithm

 1. A carrier holding no panel projects to none.
 2. Otherwise check the panel against the asset axis and the incoming clock with [`check_asset_panel`](@ref).
 3. Recover the emitted clock's rows in the incoming one with [`feature_row_indices`](@ref), and view the panel over them with [`port_opt_view`](@ref). The asset axis is whole, and it is still named so that a square tensor Panel Field is cut on its label axis too.

# Arguments

  - `pnl`: The caller's [`AssetPanel`](@ref), or `nothing`.
  - `ts_new`: The timestamps the ingestion emits.
  - `ts_old`: The timestamps of the asset table the caller handed in.
  - `nx`: The asset names.

# Validation

  - The panel describes `length(nx)` assets and `length(ts_old)` observations. Raises a `DimensionMismatch`.

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
function project_panel_clock(::Nothing, ::Any, ::Any, ::VecStr)
    return nothing
end
function project_panel_clock(pnl::AssetPanel, ts_new, ts_old, nx::VecStr)::AssetPanel
    check_asset_panel(pnl, length(nx), length(ts_old), "the number of asset price columns")
    return port_opt_view(pnl, feature_row_indices(pnl, ts_new, ts_old),
                         collect(eachindex(nx)), nx)
end
"""
    span_carrier_view(span::Nothing, ts_new, ts_old, j) -> nothing
    span_carrier_view(span::AbstractMatrix{Bool}, ts_new, ts_old, j) -> SubArray
    span_carrier_view(span::ListingSpan, ts_new, ts_old, j)

View a price carrier's Listing Span over the surviving timestamps and the assets `j`, or return `nothing` when the carrier holds none.

The span is a statement about the **instruments**, so a window's span is the panel-wide one *viewed*, never one re-derived from the window: re-deriving reads a delisting that straddles the window end as an asset that was never listed, which is the divergence a panel-wide derivation exists to avoid. A view is exact and, over a [`PortfolioOptimisers.ListingSpan`](@ref), allocates no cell.

The span is held positionally parallel to the price clock, so its rows are recovered from the surviving timestamps the same way a time-varying [`AssetPanel`](@ref)'s are, by [`matched_row_indices`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `span` is `nothing`: return `nothing`, without matching any timestamp.
 2. `span` is a matrix: recover its rows with [`matched_row_indices`](@ref), and view it at those rows and the assets `j`.
 3. `span` is a [`PortfolioOptimisers.ListingSpan`](@ref) and the recovered rows are the whole clock in order: subset the two bound vectors instead, so the two integers per asset survive the cut rather than being expanded into a view of booleans. Any other row selection can split an interval in half, which no interval can say, and falls back to step 2.

# Arguments

  - `span`: The Listing Span on the price clock, or `nothing`.
  - `ts_new`: Timestamps that survived the selection.
  - `ts_old`: Timestamps of the price clock the span is parallel to.
  - `j`: Asset index.

# Returns

  - A view of `span`, or `nothing`.

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
    #! A row selection that keeps the whole clock in order leaves every interval intact,
    #! so the two integers per asset survive the cut. Any other selection can split one
    #! in half, which no interval can say, and falls back to the ordinary view.
    return if i == axes(span, 1)
        ListingSpan(span.first[j], span.last[j], span.n)
    else
        view(span, i, j)
    end
end
"""
    assert_span_shape(span::Nothing, nobs, na) -> nothing
    assert_span_shape(span::AbstractMatrix{Bool}, nobs::Integer, na::Integer) -> nothing

Check that a Listing Span fits the price panel it rides on.

A span states which assets are listed at each observation of the price clock, so it is the shape of the asset prices. A carrier that states none states no universe, and there is nothing to check.

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

A series the caller omitted contributes no column name, so it contributes an empty list rather than an `isnothing` branch at every site that reads one.

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

A column name is what says which series a column came from: the asset, factor and benchmark tables are merged onto one clock, and the blocks are taken apart by name afterwards. `TimeSeries.merge` renames the second of two columns that share a name, appending `_1`, so a shared name silently makes one block take the other's column, and the minted name belongs to no block at all. A shared name is also not the conversion's to resolve — `X`'s `AAPL` and `F`'s `AAPL` are different series, and keeping either one is worse than refusing both.

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

The conversion writes the clock into a column named `timestamp`, and a series of the same name takes that column's place: the clock keeps the name and the series is renamed `timestamp_1`, so the block the series belongs to reads the dates as prices. The name is the layer's, and a caller holding a series of that name renames it.

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

This is the door's check, and both doors take it: [`price_ingestion`](@ref) before it merges, and [`prices_to_returns`](@ref) before it does. What it buys is that every later piece may split the merged table by name — a name belongs to exactly one of the asset, factor and benchmark blocks, and none of them is the clock's.

Two series of one table cannot be checked here. Every `TimeSeries.TimeArray` constructor runs `TimeSeries.replace_dupes!` over its column names, so a table's own duplicates are renamed before the table exists and no duplicate reaches this function.

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

A gapless ingestion states a universe in which every asset is listed at every observation and every return is finite, so both masks are `true` throughout. [`PortfolioOptimisers.AllTrueMask`](@ref) says exactly that in two integers, which is what makes *always emit a panel* cost `O(1)` rather than `observations × assets` bits. The estimation mask is a subset of the active one, so testing the estimation mask alone answers for both.

# Algorithm

 1. Every entry of `emsk` is `true`: return one [`PortfolioOptimisers.AllTrueMask`](@ref) as both masks.
 2. Otherwise return the two masks unchanged.

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

The layer's carrier arrives with a **Listing Span**, and this projects it onto the returns clock and intersects it with finiteness. A carrier built outside the layer arrives without one and states **no universe**, whether or not its prices hold a gap: a window-local derivation reads a delisting straddling the window end as an asset that was never listed, so it answers a question it cannot answer correctly. The gaps of such a carrier are still handled — with no panel the Coverage Universe reads finiteness alone — and `pnl === nothing` keeps its single meaning: the carrier was not built by the layer.

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

The layer emits **one carrier**, and the masks ride on it: a [`Pipeline`](@ref) step has one out-slot, and mask emission cannot leave [`PricesToReturns`](@ref) without landing on the price clock, where the masks are not stated. Carrying them makes [`port_opt_view`](@ref) slice the universe in step with the returns for free.

A caller's Panel Fields are kept. Only the masks are the layer's to state: an estimation mask is a statement about the *data* rather than about the instruments, so it is re-derived in every case and `emsk ⊆ amsk` holds by construction rather than by refusal.

# Algorithm

The method that Julia selects is the algorithm.

 1. No masks: return the panel unchanged. The carrier states no universe, so `pnl === nothing` keeps its one meaning — the carrier was not built by the layer.
 2. Masks and no panel: return an [`AssetPanel`](@ref) of the two masks and no Panel Field. This is the layer's common case: a caller holding only prices has no feature data.
 3. Masks and a panel: keep its Panel Fields and replace its masks. A static panel's fields carry no observation axis, so they are lifted onto the masks' clock with [`panel_field_lift`](@ref) first.

# Arguments

  - `pnl`: The Asset Panel the price carrier held, or `nothing`.
  - `amsk`: The active mask, or `nothing`.
  - `emsk`: The estimation mask, or `nothing`.

# Returns

  - `pnl′::Option{<:AssetPanel}`: The panel the returns carrier holds.

# Related

  - [`AssetPanel`](@ref)
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
