"""
$(DocStringExtensions.TYPEDEF)

Reads a per-asset listing interval as a matrix of booleans, storing two integers per asset.

`ListingSpan` is the **Listing Span**: the interval of the price clock over which an asset is listed, from its first priced observation to its last. It answers `size` as `(observations, assets)` and `getindex(s, t, i)` as `first[i] <= t <= last[i]`, so it is an ordinary `AbstractMatrix{Bool}` to every reader, and it costs two integers per asset rather than one boolean per cell. The compression is **exact** rather than approximate: under the **Span Rule** an interior gap leaves an asset listed, so an asset's active set *is* that interval. A column that is a gap throughout is the empty interval, `first[i] > last[i]`.

It is unexported and Base-only: it owns `size`, `getindex` and `show`, and nothing else in the library dispatches on it. The public bound is `AbstractMatrix{Bool}`, which is what a caller writes against and what a caller's own declaration — a listing calendar, or a constituency that leaves and rejoins — enters as.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ListingSpan(first::VecInt, last::VecInt, n::Integer) -> ListingSpan

## Validation

  - `length(first) == length(last)`. Raises a `DimensionMismatch`.
  - `n >= 0`. Raises a `DomainError`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ListingSpan([1, 2], [3, 3], 3)
ListingSpan(3 × 2)
```

# Related

  - [`listing_span`](@ref)
  - [`universe_masks`](@ref)
  - [`PortfolioOptimisers.project_span`](@ref)
"""
struct ListingSpan <: AbstractMatrix{Bool}
    """
    Index of each asset's first priced observation, on the clock the span is stated on.
    """
    first::Vector{Int}
    """
    Index of each asset's last priced observation, on the clock the span is stated on. An asset that is a gap throughout carries `last[i] < first[i]`, the empty interval.
    """
    last::Vector{Int}
    """
    Length of the clock the span is stated on.
    """
    n::Int
    function ListingSpan(first::VecInt, last::VecInt, n::Integer)
        @argcheck(length(first) == length(last),
                  DimensionMismatch("a Listing Span names one interval per asset, so its two bounds are the same length; got length(first) = $(length(first)) and length(last) = $(length(last))"))
        @argcheck(n >= zero(n),
                  DomainError(n,
                              "a Listing Span is stated on a clock of non-negative length"))
        return new(collect(Int, first), collect(Int, last), Int(n))
    end
end
function Base.size(s::ListingSpan)
    return (s.n, length(s.first))
end
Base.@propagate_inbounds function Base.getindex(s::ListingSpan, t::Integer, i::Integer)
    @boundscheck checkbounds(s, t, i)
    return s.first[i] <= t <= s.last[i]
end
function Base.IndexStyle(::Type{<:ListingSpan})
    return IndexCartesian()
end
function Base.show(io::IO, s::ListingSpan)
    return print(io, "ListingSpan($(s.n) × $(length(s.first)))")
end
function Base.show(io::IO, ::MIME"text/plain", s::ListingSpan)
    return show(io, s)
end
"""
    listing_span(X::AbstractMatrix{<:Union{Missing, <:Real}}) -> ListingSpan

Derive the Listing Span of every asset column of a price panel by the Span Rule.

The rule reads a gap's *position* in a price column, which is what lets a caller holding only prices state a universe: a **leading** run of gaps is an asset not yet listed, a **trailing** run is a delisting, and an **interior** gap — priced on both sides — is a suspension or a holiday on an asset that is still listed and still held. The first two fall outside the span and the third inside it, so the span is exactly the interval from the first priced observation to the last. A cell counts as priced when it is neither `missing` nor non-finite, which unifies the two conventions a source spells an absent price with.

The derivation runs once over the whole panel rather than per window. A listing calendar is a fact about the instruments, not an estimate from returns, which is what licenses that; a window-local derivation instead reads a delisting straddling the window end as dead rather than held.

# Algorithm

 1. For each asset column, find the first and the last observation whose price is neither `missing` nor non-finite.
 2. A column with no priced observation takes the empty interval, `first = 1` and `last = 0`.
 3. Return the two bound vectors as a [`PortfolioOptimisers.ListingSpan`](@ref) over the price clock.

# Arguments

  - `X`: The price panel, `observations × assets`. An absent price is spelled `missing` or non-finite.

# Validation

  - `X` is not empty. Raises an [`IsEmptyError`](@ref).

# Returns

  - `span::ListingSpan`: The Listing Span of each asset column, on the price clock.

# Examples

```jldoctest
julia> X = [NaN 1.0 3.0
            1.0 NaN 3.1
            1.1 2.0 NaN];

julia> span = listing_span(X)
ListingSpan(3 × 3)

julia> Matrix(span)
3×3 Matrix{Bool}:
 0  1  1
 1  1  1
 1  1  0
```

# Related

  - [`universe_masks`](@ref)
  - [`PortfolioOptimisers.ListingSpan`](@ref)
  - [`prices_to_returns`](@ref)
"""
function listing_span(X::AbstractMatrix{<:Union{Missing, <:Real}})
    @argcheck(!isempty(X),
              IsEmptyError("`X` cannot be empty: a Listing Span names one interval per asset column of a price panel"))
    T, N = size(X)
    fst = Vector{Int}(undef, N)
    lst = Vector{Int}(undef, N)
    for i in 1:N
        priced = t -> !ismissing(X[t, i]) && isfinite(X[t, i])
        fi = findfirst(priced, 1:T)
        li = findlast(priced, 1:T)
        fst[i] = isnothing(fi) ? 1 : fi
        lst[i] = isnothing(li) ? 0 : li
    end
    return ListingSpan(fst, lst, T)
end
"""
    project_span(span::ListingSpan, m::Integer) -> ListingSpan
    project_span(span::AbstractMatrix{Bool}, m::Integer) -> BitMatrix

Project a price-clock listing statement onto a returns clock of `m` observations.

A return is the change between two **consecutive** observations, so a return is active exactly when *both* prices of its pair lie inside the asset's listing. That reading fixes the projection, and it is the same one under either padding convention: with padding the clocks align row for row and the pair of observation `t` is `(t - 1, t)`, so the leading row is active for nobody; without padding the returns clock is one observation shorter and the pair of observation `s` is `(s, s + 1)`. Under it the active mask bounds exactly the run of a column's finite returns at both ends, and no asset's inception emits a **Held Gap**.

On a [`PortfolioOptimisers.ListingSpan`](@ref) the projection is `[first + 1, last]` under padding and `[first, last - 1]` without it, so the two integers per asset survive the crossing rather than being expanded and discarded at it. An asset with a single priced observation gives the empty interval, which is correct: one price yields no return. On any other `AbstractMatrix{Bool}`, which is how a caller's own declaration enters, the same rule is applied cell by cell, so a constituency that leaves and rejoins books no return across its absence.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`PortfolioOptimisers.ListingSpan`](@ref): shift the opening bound up by one when the clocks align row for row, and the closing bound down by one when they do not.
 2. Any other `AbstractMatrix{Bool}`: write `span[a, i] && span[a + 1, i]` into row `a + 1 - o` of the result, where `o` is the row count the returns clock lost.

# Arguments

  - `span`: The listing statement, `price observations × assets`.
  - `m::Integer`: Length of the returns clock to project onto.

# Returns

  - `amsk`: The active mask, `m × assets`, on the returns clock.

# Related

  - [`universe_masks`](@ref)
  - [`listing_span`](@ref)
  - [`PortfolioOptimisers.ListingSpan`](@ref)
"""
function project_span(span::ListingSpan, m::Integer)
    return if size(span, 1) == m
        ListingSpan(span.first .+ 1, span.last, m)
    else
        ListingSpan(span.first, span.last .- 1, m)
    end
end
function project_span(span::AbstractMatrix{Bool}, m::Integer)
    n, N = size(span)
    o = n - Int(m)
    amsk = falses(m, N)
    for i in 1:N, a in 1:(n - 1)
        amsk[a + 1 - o, i] = span[a, i] && span[a + 1, i]
    end
    return amsk
end
"""
    universe_masks(
        span::AbstractMatrix{Bool},
        R::AbstractMatrix{<:Union{Missing, <:Real}}
    ) -> Tuple{AbstractMatrix{Bool}, BitMatrix}

Project a listing statement onto the returns clock and intersect it with finiteness, giving an Asset Panel's two universe masks.

The **active mask** says which assets are in the universe at each observation, and it is `span` projected by [`PortfolioOptimisers.project_span`](@ref). A caller who passes their own `AbstractMatrix{Bool}` rather than a derived [`PortfolioOptimisers.ListingSpan`](@ref) therefore replaces the Span Rule's answer **outright**, and is never second-guessed. The **estimation mask** is the active mask intersected with the finiteness of the returns: an asset inside a **Held Gap** is active — still listed, still held — but has no return at that observation, so it cannot enter that observation's cross-section. It is **never the caller's to state** and is always re-derived, which is what makes `emsk ⊆ amsk` hold by construction rather than by refusal, and what keeps the library from quietly disagreeing with the numbers it emitted.

The estimation mask is a **snapshot** of what the conversion produced, not a view over whatever `R` later holds, so a value-level step that rewrites a return does not move the estimation universe under a fold already scored against it.

Which padding convention the conversion used is read off the two row counts, and this is the one place that knows: `size(span, 1) == size(R, 1)` is the padded case, in which the first observation survives with a non-finite return, and `size(span, 1) == size(R, 1) + 1` is the unpadded case.

# Algorithm

 1. Read the padding convention off the row counts of `span` and `R`.
 2. Project `span` onto the returns clock with [`PortfolioOptimisers.project_span`](@ref), giving the active mask.
 3. Intersect the active mask with the finiteness of `R`, giving the estimation mask.

# Arguments

  - `span`: The listing statement, `price observations × assets`. A [`PortfolioOptimisers.ListingSpan`](@ref) from [`listing_span`](@ref), or a caller's own declaration.
  - `R`: The returns panel the conversion produced, `observations × assets`. An absent return is spelled `missing` or non-finite.

# Validation

  - `size(span, 2) == size(R, 2)`. Raises a `DimensionMismatch`.
  - `size(span, 1)` is `size(R, 1)` or `size(R, 1) + 1`. Raises a `DimensionMismatch`.

# Returns

  - `amsk`: The active mask, `observations × assets` on the returns clock. A [`PortfolioOptimisers.ListingSpan`](@ref) when `span` is one, and a `BitMatrix` otherwise.
  - `emsk::BitMatrix`: The estimation mask, `observations × assets` on the returns clock.

# Examples

```jldoctest
julia> X = [NaN 1.0 3.0
            1.0 NaN 3.1
            1.1 2.0 NaN];

julia> R = [NaN NaN NaN
            NaN NaN 0.1
            0.1 NaN NaN];

julia> amsk, emsk = universe_masks(listing_span(X), R);

julia> Matrix(amsk)
3×3 Matrix{Bool}:
 0  0  0
 0  1  1
 1  1  0

julia> emsk
3×3 BitMatrix:
 0  0  0
 0  0  1
 1  0  0
```

# Related

  - [`listing_span`](@ref)
  - [`PortfolioOptimisers.project_span`](@ref)
  - [`AssetPanel`](@ref)
  - [`PortfolioOptimisers.assert_panel_masks`](@ref)
"""
function universe_masks(span::AbstractMatrix{Bool},
                        R::AbstractMatrix{<:Union{Missing, <:Real}})
    n, N = size(span)
    m, Nr = size(R)
    @argcheck(N == Nr,
              DimensionMismatch("a listing statement and a returns panel name the same assets; got size(span, 2) = $N and size(R, 2) = $Nr"))
    @argcheck(n == m || n == m + 1,
              DimensionMismatch("a returns clock is the price clock under padding, and one observation shorter without it; got size(span, 1) = $n and size(R, 1) = $m"))
    amsk = project_span(span, m)
    emsk = falses(m, N)
    for i in 1:N, t in 1:m
        x = R[t, i]
        emsk[t, i] = amsk[t, i] && !ismissing(x) && isfinite(x)
    end
    return amsk, emsk
end

export listing_span, universe_masks
