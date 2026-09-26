"""
$(DocStringExtensions.TYPEDEF)

Stores the listing interval of each asset as two integers and reads it as a matrix of booleans.

`ListingSpan` holds the **Listing Span** of each asset, the interval of the price clock from the first priced observation of the asset to its last. `size` answers `(observations, assets)`, and `getindex(s, t, i)` answers `first[i] <= t <= last[i]`, so every reader sees an ordinary `AbstractMatrix{Bool}`. The type holds two integers per asset in place of one boolean per cell. The compression is exact, because the **Span Rule** keeps an interior gap inside the listing, so the active set of an asset is one interval. A column that is a gap throughout takes the empty interval, `first[i] > last[i]`.

The type is unexported. It owns `size`, `getindex`, `IndexStyle` and `show`. [`PortfolioOptimisers.project_span`](@ref) and [`PortfolioOptimisers.span_carrier_view`](@ref) carry a library-internal fast path for it, which keeps the two integers per asset through a projection and through a window. A caller writes against the public bound, `AbstractMatrix{Bool}`, and a caller's own declaration, such as a listing calendar or a constituency that leaves and joins again, enters under that bound.

# Mathematical definition

```math
\\begin{align}
a_{t,\\,i} &= \\begin{cases}
1 & f_{i} \\leq t \\leq l_{i} \\\\
0 & \\text{otherwise}
\\end{cases}\\,, \\quad t = 1, \\ldots, n\\,.
\\end{align}
```

An empty interval, ``f_{i} > l_{i}``, gives ``a_{t,\\,i} = 0`` at every observation.

Where:

  - $(math_dict[:a_ti_span])
  - $(math_dict[:f_i_span])
  - $(math_dict[:l_i_span])
  - $(math_dict[:n_span])

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
  - [`PortfolioOptimisers.span_carrier_view`](@ref): cuts a span to a window of its clock and keeps the two integers per asset.
"""
struct ListingSpan <: AbstractMatrix{Bool}
    """
    Index of the first priced observation of each asset, on the clock of the span. A window of a longer clock puts this bound at zero or below for an asset that is listed before the window opens.
    """
    first::Vector{Int}
    """
    Index of the last priced observation of each asset, on the clock of the span. An asset that is a gap throughout carries `last[i] < first[i]`, the empty interval. A window of a longer clock puts this bound above `n` for an asset that is still listed when the window closes.
    """
    last::Vector{Int}
    """
    Length of the clock of the span, in observations.
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

The Span Rule reads the position of a gap in a price column, so a caller who holds only prices can state a universe. A leading run of gaps is an asset that is not listed yet, and a trailing run is a delisting. An interior gap, with prices on both sides, is a suspension or a holiday on an asset that is still listed and still held. The first two fall outside the span and the third falls inside it. A cell counts as priced when it is neither `missing` nor non-finite, so the two spellings of an absent price read alike.

The derivation runs once over the whole panel, not once per window. A listing calendar is a fact about the instruments and not an estimate from returns, so one derivation serves every window. A derivation over one window reads a gap that runs past the end of the window as a delisting, although the asset resumes after the window and is still held.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P}_{i} &= \\left\\{t \\in \\{1, \\ldots, n\\} : \\text{asset } i \\text{ is priced at observation } t\\right\\}\\,, \\\\
f_{i} &= \\min \\mathcal{P}_{i}\\,, \\\\
l_{i} &= \\max \\mathcal{P}_{i}\\,.
\\end{align}
```

An asset with an empty ``\\mathcal{P}_{i}`` takes an empty interval, ``f_{i} > l_{i}``. Every observation from ``f_{i}`` to ``l_{i}`` is listed, so an interior gap stays inside the listing.

Where:

  - ``\\mathcal{P}_{i}``: Priced observations of asset ``i``, those at which its price is neither `missing` nor non-finite.
  - $(math_dict[:f_i_span])
  - $(math_dict[:l_i_span])
  - $(math_dict[:n_span])

# Algorithm

 1. For each asset column, find the first and the last observation whose price is neither `missing` nor non-finite.
 2. A column with no priced observation takes the empty interval, `first = 1` and `last = 0`.
 3. Return the two bound vectors as a [`PortfolioOptimisers.ListingSpan`](@ref) over the price clock.

# Arguments

  - `X`: The price panel, `observations × assets`. A source spells an absent price as `missing` or as a non-finite value.

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

A return is the change between two consecutive prices, so a return is active when both prices of its pair lie inside the listing of the asset. The projection follows from that rule under either padding convention. With padding, the two clocks align row for row and return `t` reads prices `t - 1` and `t`, so the first row is active for no asset. Without padding, the returns clock is one observation shorter and return `t` reads prices `t` and `t + 1`. The active mask thus starts at the first return whose two prices lie inside the listing, and it stops at the last one. The first return of an asset is therefore never a **Held Gap**. An interior gap next to either end of the listing does make one, because the asset is still listed there.

On a [`PortfolioOptimisers.ListingSpan`](@ref) the projection moves the two bounds, so the two integers per asset stay two integers. A bound lies outside the clock when the span is a window of a longer clock, and the projection clamps the opening bound to the clock before it moves it. The projection of a window thus equals the projection of the matrix of booleans that the window holds. An asset with one priced observation gets the empty interval, because one price gives no return. On any other `AbstractMatrix{Bool}`, which is how a caller's own declaration enters, the rule applies cell by cell, so a constituency that leaves and joins again books no return across its absence.

# Mathematical definition

```math
\\begin{align}
\\tilde{a}_{t,\\,i} &= a_{t+o-1,\\,i} \\wedge a_{t+o,\\,i}\\,, \\quad t = 1, \\ldots, m\\,, \\\\
\\tilde{f}_{i} &= \\max(f_{i}, 1) + 1 - o\\,, \\\\
\\tilde{l}_{i} &= l_{i} - o\\,.
\\end{align}
```

The first line is the rule for a matrix of booleans, with ``a_{0,\\,i} = 0``. The second and the third lines are the same rule on an interval: ``\\tilde{a}_{t,\\,i} = 1`` exactly when ``\\tilde{f}_{i} \\leq t \\leq \\tilde{l}_{i}``. Under padding, ``\\tilde{a}_{1,\\,i} = 0`` for every asset, because the first return reads a price before the clock.

Where:

  - $(math_dict[:a_tilde_ti_act])
  - $(math_dict[:a_ti_span])
  - ``\\tilde{f}_{i}``, ``\\tilde{l}_{i}``: First and last active observation of asset ``i``, on the returns clock.
  - $(math_dict[:f_i_span])
  - $(math_dict[:l_i_span])
  - $(math_dict[:o_span])
  - $(math_dict[:n_span])
  - $(math_dict[:m_span])

# Algorithm

The method that Julia selects is the algorithm.

 1. [`PortfolioOptimisers.ListingSpan`](@ref): read the clock offset `o` off the two row counts. Clamp each opening bound to the clock and move it up by `1 - o`, and move each closing bound down by `o`. Return the moved bounds as a `ListingSpan` of length `m`.
 2. Any other `AbstractMatrix{Bool}`: read the clock offset `o` off the two row counts, and write `span[a, i] && span[a + 1, i]` into row `a + 1 - o` of `amsk`.

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
    o = size(span, 1) - Int(m)
    return ListingSpan(max.(span.first, 1) .+ (1 - o), span.last .- o, m)
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

Project a listing statement onto the returns clock and intersect it with finiteness to make the two universe masks of an Asset Panel.

The **active mask** states which assets are in the universe at each observation. It is `span` projected by [`PortfolioOptimisers.project_span`](@ref). A caller who passes their own `AbstractMatrix{Bool}` in place of a derived [`PortfolioOptimisers.ListingSpan`](@ref) replaces the answer of the Span Rule, and the function does not change that answer. The **estimation mask** is the active mask intersected with the finiteness of the returns. An asset inside a **Held Gap** is active, because it is still listed and still held, but it has no return at that observation, so it cannot enter the cross-section of that observation. A caller cannot state the estimation mask. The function always derives it, so `emsk ⊆ amsk` holds by construction, and the mask always agrees with the returns that the conversion made.

The estimation mask is a copy of the finiteness of `R` when the function runs, not a view of `R`. A later step that rewrites a return thus does not change the estimation universe of a fold that already has a score.

The function reads the padding convention off the two row counts, and no other function does. `size(span, 1) == size(R, 1)` is the padded case, where the first observation stays with a non-finite return. `size(span, 1) == size(R, 1) + 1` is the unpadded case.

# Mathematical definition

```math
\\begin{align}
e_{t,\\,i} &= \\tilde{a}_{t,\\,i} \\wedge \\mathbb{1}\\left[x_{t,\\,i} \\text{ is present and finite}\\right]\\,, \\quad t = 1, \\ldots, m\\,, \\\\
o &\\in \\{0, 1\\}\\,.
\\end{align}
```

It follows that ``e_{t,\\,i} \\leq \\tilde{a}_{t,\\,i}`` at every observation.

Where:

  - ``e_{t,\\,i}``: Estimation mask entry of asset ``i`` at observation ``t`` of the returns clock.
  - $(math_dict[:a_tilde_ti_act])
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:o_span])
  - $(math_dict[:n_span])
  - $(math_dict[:m_span])

# Algorithm

 1. Read the padding convention off the row counts of `span` and `R`.
 2. Project `span` onto the returns clock with [`PortfolioOptimisers.project_span`](@ref), giving the active mask `amsk`.
 3. Intersect `amsk` with the finiteness of `R`, giving the estimation mask `emsk`.

# Arguments

  - `span`: The listing statement, `price observations × assets`. A [`PortfolioOptimisers.ListingSpan`](@ref) from [`listing_span`](@ref), or a caller's own declaration.
  - `R`: The returns panel the conversion produced, `observations × assets`. An absent return is `missing` or a non-finite value.

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
