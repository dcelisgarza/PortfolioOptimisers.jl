"""
$(DocStringExtensions.TYPEDEF)

Names the past periods on which a follow-the-leader rule re-solves its held estimator.

A [`FollowTheLeader`](@ref) rule holds a Sample Selector on `sel`. The solved rules of the online selection literature solve one programme and differ in the sample alone. Follow the leader takes every period so far, and the successive variable rebalanced portfolio takes the last `W`. The pattern-matching rules take the periods whose preceding window resembles the latest window, by histogram cell, kernel radius, nearest neighbours, correlation or cluster. A selector reads the price relatives that the head holds and returns row indices. It holds no rows of its own.

# Interfaces

In order to implement a new selector, subtype `AbstractSampleSelector` with the paper's parameters as part of the struct, and implement:

  - `select_rows(sel::AbstractSampleSelector, X::AbstractMatrix) -> AbstractVector{<:Integer}`: The indices of the rows of `X` in the sample, in time order. An empty vector is the empty sample.
  - `rows_needed(sel::AbstractSampleSelector) -> Union{Nothing, Integer}`: The number of rows that the selector reads at a step, or `nothing` for every row folded so far.

## Arguments

  - `sel`: The selector.
  - `X`: The price relatives `1 .+ r` of every row that the head holds, `observations × assets`, in time order. A gap reads as one, through [`price_relative`](@ref). So a window that spans a listing compares as a window in which that asset sat in cash, which is the number that the step reads there.

## Returns

  - `idx::AbstractVector{<:Integer}`: The row indices of the sample, in time order.

# Related

  - [`FollowTheLeader`](@ref)
  - [`Prefix`](@ref)
  - [`LastRows`](@ref)
  - [`HistogramMatch`](@ref)
  - [`KernelMatch`](@ref)
  - [`NearestNeighbourMatch`](@ref)
  - [`CorrelationMatch`](@ref)
  - [`ClusterMatch`](@ref)
"""
abstract type AbstractSampleSelector <: AbstractAlgorithm end
"""
    select_rows(sel::AbstractSampleSelector, X::AbstractMatrix)

Returns the row indices of the sample that a Sample Selector names, from the price relatives that the head holds.

# Arguments

  - `sel`: The selector.
  - `X`: The price relatives, `observations × assets`, in time order.

# Returns

  - `idx::AbstractVector{<:Integer}`: The row indices of the sample ``C_t``, in time order, with ``t`` the row count of `X`.

# Related

  - [`AbstractSampleSelector`](@ref)
  - [`FollowTheLeader`](@ref)
"""
function select_rows end
"""
$(DocStringExtensions.TYPEDEF)

Selects every period so far, the sample of follow the leader.

Under the default log-optimal estimator the leader is the best constant rebalanced portfolio to date. This is the successive constant rebalanced portfolio (SCRP) of Gaivoronski and Stella (2000).

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace 1, \\ldots, t \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - $(math_dict[:t_period])

# Examples

```jldoctest
julia> Prefix()
Prefix()
```

# Related

  - [`AbstractSampleSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`LastRows`](@ref): the sample of the last `W` periods, which equals this sample while ``t \\leq W``.

# References

  - $(ref_dict[:gaivoronski2000])
  - $(ref_dict[:lihoi2014]) Section 3.2.3.
"""
struct Prefix <: AbstractSampleSelector end
function select_rows(::Prefix, X::AbstractMatrix)
    return axes(X, 1)
end
function rows_needed(::Prefix)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the last `W` periods, the sample of the successive variable rebalanced portfolio.

Under the default log-optimal estimator the leader is the best constant rebalanced portfolio of a moving window. Gaivoronski and Stella (2000) call the rule the successive variable rebalanced portfolio, and Li and Hoi (2014) call it the variable rebalanced portfolio (VRP). Before `W` periods exist the sample is every period so far, as in the paper of Gaivoronski and Stella.

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace \\max(1,\\, t - W + 1), \\ldots, t \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - ``W``: Window of the sample, the `W` field.
  - $(math_dict[:t_period])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LastRows(; W::Integer = 30) -> LastRows

Keywords correspond to the struct's fields.

## Validation

  - `W >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> LastRows()
LastRows
  W ┴ Int64: 30
```

# Related

  - [`AbstractSampleSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`Prefix`](@ref): the sample of every period so far, which this sample equals while ``t \\leq W``.

# References

  - $(ref_dict[:gaivoronski2000])
  - $(ref_dict[:lihoi2014]) Section 3.2.3.
"""
struct LastRows{T1 <: Integer} <: AbstractSampleSelector
    """
    The number of rows of the window.
    """
    W::T1
    function LastRows(W::Integer)
        @argcheck(W >= 1, DomainError(W, "W must be at least 1"))
        return new{typeof(W)}(W)
    end
end
function LastRows(; W::Integer = 30)::LastRows
    return LastRows(W)
end
function select_rows(sel::LastRows, X::AbstractMatrix)
    T = size(X, 1)
    return max(1, T - sel.W + 1):T
end
function rows_needed(sel::LastRows)
    return sel.W
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the periods whose preceding window resembles the latest window, the pattern-matching family.

The selector compares each candidate period through the window of the `window` periods before it. The latest window holds the last `window` periods, and it precedes the next period, which has no price relative yet. Before `window + 1` periods exist no candidate exists, the sample is empty, and the rule plays the uniform portfolio. Every selector of the family reads every row folded so far.

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : m\\left(\\boldsymbol{x}_{i-w}^{i-1},\\, \\boldsymbol{x}_{t-w+1}^{t}\\right) \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - $(math_dict[:w_win])
  - $(math_dict[:x_win])
  - ``m``: Similarity test of the selector, true when a candidate window matches the latest window.
  - $(math_dict[:t_period])

# Algorithm

The steps of [`select_rows`](@ref) on every selector of the family:

 1. Call `pattern_candidates` on `sel.window` and `X`, giving the candidate rows `rows`, their flattened windows `cands` and the flattened latest window `latest`. When it returns `nothing`, return the empty vector.
 2. Call `matched_rows` on `sel`, `X`, `cands` and `latest`, giving the matched positions.
 3. Return `rows` at those positions.

# Interfaces

In order to implement a new pattern-matching selector, subtype `AbstractPatternMatchSelector` with a `window` field and the paper's similarity parameters, and implement:

  - `matched_rows(sel::AbstractPatternMatchSelector, X::AbstractMatrix, cands::AbstractVector, latest::AbstractVector) -> AbstractVector{<:Integer}`: The positions in `cands` of the candidates that match. `cands` holds one flattened window for each candidate, and `latest` is the latest window flattened the same way.

# Related

  - [`AbstractSampleSelector`](@ref)
  - [`HistogramMatch`](@ref)
  - [`KernelMatch`](@ref)
  - [`NearestNeighbourMatch`](@ref)
  - [`CorrelationMatch`](@ref)
  - [`ClusterMatch`](@ref)
"""
abstract type AbstractPatternMatchSelector <: AbstractSampleSelector end
"""
    matched_rows(sel::AbstractPatternMatchSelector, X::AbstractMatrix, cands::AbstractVector, latest::AbstractVector)

Returns the positions of the candidates that a pattern-matching selector matches against the latest window.

# Arguments

  - `sel`: The selector.
  - `X`: The price relatives that the head holds, for a selector that reads more than the windows.
  - `cands`: One flattened window for each candidate period, in time order.
  - `latest`: The latest window, flattened the same way.

# Returns

  - `idx::AbstractVector{<:Integer}`: The positions in `cands` of the matched candidates, increasing.

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`select_rows`](@ref)
"""
function matched_rows end
function rows_needed(::AbstractPatternMatchSelector)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Forms the candidate periods of a pattern-matching selector, their flattened windows and the flattened latest window.

A window flattens column by column, so the entries of one asset stay together.

# Arguments

  - `window`: The number of rows in each window.
  - `X`: The price relatives, `T × N`, in time order.

# Returns

  - `nothing` when `T <= window`, because no candidate exists.
  - `(rows, cands, latest)` otherwise. `rows` is `(window + 1):T`. The entry of `cands` for the candidate row `i` is `vec(X[(i - window):(i - 1), :])`. `latest` is `vec(X[(T - window + 1):T, :])`.

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`select_rows`](@ref)
"""
function pattern_candidates(window::Integer, X::AbstractMatrix)
    T = size(X, 1)
    if T <= window
        return nothing
    end
    latest = vec(X[(T - window + 1):T, :])
    rows = (window + 1):T
    cands = map(i -> vec(X[(i - window):(i - 1), :]), rows)
    return rows, cands, latest
end
function select_rows(sel::AbstractPatternMatchSelector, X::AbstractMatrix)
    c = pattern_candidates(sel.window, X)
    if isnothing(c)
        return Int[]
    end
    rows, cands, latest = c
    return rows[matched_rows(sel, X, cands, latest)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Checks that the window of a pattern-matching selector holds at least one row.

# Arguments

  - `window`: The number of rows in each window.

# Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`AbstractPatternMatchSelector`](@ref)
"""
function assert_pattern_window(window::Integer)::Nothing
    @argcheck(window >= 1, DomainError(window, "window must be at least 1"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the periods whose preceding window falls in the same histogram cell as the latest window (BH).

This is the histogram rule of Györfi and Schäfer (2003). The paper quantises each price relative vector by a partition of the positive orthant, and a period matches when its window gives the same string of cell labels as the latest window. The paper leaves the partition to the user. Its consistency theorem asks for a nested sequence of partitions whose cells shrink, and the paper mixes the experts of that sequence by wealth. That mixture is an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each.

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : G\\left(\\boldsymbol{x}_{i-w}^{i-1}\\right) = G\\left(\\boldsymbol{x}_{t-w+1}^{t}\\right) \\right\\rbrace\\,, \\\\
G(\\boldsymbol{v})_j &= \\left\\lvert \\left\\lbrace k : e_k \\leq v_j \\right\\rbrace \\right\\rvert\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - $(math_dict[:w_win])
  - $(math_dict[:x_win])
  - ``G``: Cell map, which sends a flattened window to the bin index of each of its entries.
  - ``e_1 < \\cdots < e_m``: Cell edges, the `edges` field.
  - ``v_j``: Entry ``j`` of a flattened window ``\\boldsymbol{v}``.
  - $(math_dict[:t_period])

The bins of every entry form a partition of the positive orthant into products of intervals, which is one partition of the paper's kind. At the default `edges = [1]` the bin of an entry shows whether the asset rose or fell. So two windows match when every asset moved the same way in every period of the window. More edges give a finer partition and a smaller sample.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HistogramMatch(; window::Integer = 5, edges::AbstractVector{<:Real} = [1]) -> HistogramMatch

Keywords correspond to the struct's fields.

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `!isempty(edges)`. An `IsEmptyError` is thrown otherwise.
  - `edges` is finite and strictly increasing. An `ArgumentError` is thrown otherwise.

# Examples

```jldoctest
julia> HistogramMatch()
HistogramMatch
  window ┼ Int64: 5
   edges ┴ Vector{Int64}: [1]
```

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`KernelMatch`](@ref): the rule that matches by distance in place of a cell.
  - [`ExpertMixture`](@ref): the mixture by wealth over a grid of these selectors.

# References

  - $(ref_dict[:gyorfischafer2003])
  - $(ref_dict[:lihoi2014]) Section 3.4.1.
"""
struct HistogramMatch{T1 <: Integer, T2 <: AbstractVector{<:Real}} <:
       AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window::T1
    """
    The cell boundaries of every entry's binning, increasing.
    """
    edges::T2
    function HistogramMatch(window::Integer, edges::AbstractVector{<:Real})
        assert_pattern_window(window)
        @argcheck(!isempty(edges), IsEmptyError("edges cannot be empty"))
        @argcheck(all(isfinite, edges) && issorted(edges; lt = <=),
                  ArgumentError("edges must be finite and strictly increasing, got $edges"))
        return new{typeof(window), typeof(edges)}(window, edges)
    end
end
function HistogramMatch(; window::Integer = 5,
                        edges::AbstractVector{<:Real} = [1])::HistogramMatch
    return HistogramMatch(window, edges)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the histogram cell of a flattened window, the bin index of each of its entries.

# Arguments

  - `edges`: The cell edges, increasing.
  - `v`: The flattened window.

# Returns

  - `cell::AbstractVector{<:Integer}`: Entry `j` is the number of `edges` at or below `v[j]`.

# Related

  - [`HistogramMatch`](@ref)
"""
function histogram_cell(edges::AbstractVector, v::AbstractVector)
    return map(x -> searchsortedlast(edges, x), v)
end
function matched_rows(sel::HistogramMatch, ::AbstractMatrix, cands::AbstractVector,
                      latest::AbstractVector)
    cell = histogram_cell(sel.edges, latest)
    return findall(c -> histogram_cell(sel.edges, c) == cell, cands)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the periods whose preceding window lies within `radius` of the latest window (BK).

This is the kernel rule of Györfi, Lugosi and Udina (2006) under the uniform kernel, in the Euclidean norm of the flattened windows. The paper's experts run over a grid of windows ``w`` and radii ``c / \\ell``, and the paper mixes them by wealth. That mixture is an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each, and each selector holds one radius.

Three later rules use this selector under another objective, which the held estimator on the rule's `opt` states. The semi-log-optimal rule of Györfi, Urbán and Vajda (2007) maximises the second-order expansion of the log at one. The Markowitz-type rule of Ottucsák and Vajda (2007) maximises a mean-variance utility. The rule of Györfi and Vajda (2008) compares windows of one period and maximises the log return net of proportional costs.

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : \\left\\lVert \\boldsymbol{x}_{i-w}^{i-1} - \\boldsymbol{x}_{t-w+1}^{t} \\right\\rVert_2 \\leq r \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - $(math_dict[:w_win])
  - $(math_dict[:x_win])
  - ``r``: Radius of the uniform kernel, the `radius` field.
  - $(math_dict[:t_period])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KernelMatch(; window::Integer = 5, radius::Real) -> KernelMatch

Keywords correspond to the struct's fields. The radius has no default, because the papers scan it. Its scale is the scale of a window of price relatives.

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `radius > 0` and finite. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> KernelMatch(; radius = 0.1)
KernelMatch
  window ┼ Int64: 5
  radius ┴ Float64: 0.1
```

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`NearestNeighbourMatch`](@ref): the rule that keeps a count of the nearest windows in place of a radius.
  - [`ExpertMixture`](@ref): the mixture by wealth over a grid of these selectors.

# References

  - $(ref_dict[:gyorfi2006])
  - $(ref_dict[:gyorfi2007semilog])
  - $(ref_dict[:ottucsakvajda2007])
  - $(ref_dict[:gyorfivajda2008])
"""
struct KernelMatch{T1 <: Integer, T2 <: Real} <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window::T1
    """
    The radius of the uniform kernel, in the Euclidean norm of the concatenated windows.
    """
    radius::T2
    function KernelMatch(window::Integer, radius::Real)
        assert_pattern_window(window)
        @argcheck(isfinite(radius) && radius > zero(radius),
                  DomainError(radius, "radius must be positive and finite"))
        return new{typeof(window), typeof(radius)}(window, radius)
    end
end
function KernelMatch(; window::Integer = 5, radius::Real)::KernelMatch
    return KernelMatch(window, radius)
end
function matched_rows(sel::KernelMatch, ::AbstractMatrix, cands::AbstractVector,
                      latest::AbstractVector)
    return findall(c -> LinearAlgebra.norm(c - latest) <= sel.radius, cands)
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the periods whose preceding windows are the `neighbours` nearest to the latest window (BNN).

This is the nearest-neighbour rule of Györfi, Udina and Walk (2008), in the Euclidean norm of the flattened windows. The paper invests at period ``n = t + 1`` with ``\\lfloor p\\, n \\rfloor`` neighbours, a count that grows with the history. So `neighbours` is an `Integer` count or a fraction in `(0, 1)`. The fraction applies to the ``t - w`` candidates and not to the period, so the count never exceeds the candidates. It is below the paper's count by at most ``\\lceil p (w + 1) \\rceil``. A fraction that rounds down to zero gives the empty sample. The paper assumes that ties have zero probability. Here the earlier period wins a tie at the boundary.

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : \\boldsymbol{x}_{i-w}^{i-1} \\text{ is one of the } \\ell \\text{ windows nearest to } \\boldsymbol{x}_{t-w+1}^{t} \\right\\rbrace\\,, \\\\
\\ell &= \\begin{cases} \\min(k,\\, t - w) & \\text{for a count } k\\,, \\\\ \\lfloor p\\, (t - w) \\rfloor & \\text{for a fraction } p\\,. \\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - $(math_dict[:w_win])
  - $(math_dict[:x_win])
  - ``\\ell``: Neighbour count at period ``t``.
  - ``k``: Count, the `neighbours` field when it is an `Integer`.
  - ``p``: Fraction, the `neighbours` field when it is a real in `(0, 1)`.
  - $(math_dict[:t_period])

The distance between two windows is the Euclidean norm of their difference.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NearestNeighbourMatch(; window::Integer = 5, neighbours::Real = 0.1) -> NearestNeighbourMatch

Keywords correspond to the struct's fields.

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `neighbours >= 1` for an `Integer`, and `0 < neighbours < 1` for another real. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> NearestNeighbourMatch()
NearestNeighbourMatch
      window ┼ Int64: 5
  neighbours ┴ Float64: 0.1
```

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`KernelMatch`](@ref): the rule that keeps every window within a radius in place of a count.

# References

  - $(ref_dict[:gyorfi2008])
"""
struct NearestNeighbourMatch{T1 <: Integer, T2 <: Real} <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window::T1
    """
    The number of neighbours to keep, an `Integer` count or a fraction in `(0, 1)` of the candidate rows.
    """
    neighbours::T2
    function NearestNeighbourMatch(window::Integer, neighbours::Real)
        assert_pattern_window(window)
        if isa(neighbours, Integer)
            @argcheck(neighbours >= 1,
                      DomainError(neighbours, "a count must be at least 1"))
        else
            @argcheck(zero(neighbours) < neighbours < one(neighbours),
                      DomainError(neighbours, "a fraction must lie in (0, 1)"))
        end
        return new{typeof(window), typeof(neighbours)}(window, neighbours)
    end
end
function NearestNeighbourMatch(; window::Integer = 5,
                               neighbours::Real = 0.1)::NearestNeighbourMatch
    return NearestNeighbourMatch(window, neighbours)
end
"""
    neighbour_count(neighbours::Integer, n::Integer)
    neighbour_count(neighbours::Real, n::Integer)

Returns the number of neighbours that a nearest-neighbour selector keeps out of `n` candidates.

# Arguments

  - `neighbours`: The count, an `Integer`, or the fraction, a real in `(0, 1)`.
  - `n`: The number of candidates.

# Returns

  - `l::Integer`: `min(neighbours, n)` for a count, and `floor(Int, neighbours * n)` for a fraction.

# Related

  - [`NearestNeighbourMatch`](@ref)
"""
function neighbour_count(neighbours::Integer, n::Integer)
    return min(neighbours, n)
end
function neighbour_count(neighbours::Real, n::Integer)
    return floor(Int, neighbours * n)
end
function matched_rows(sel::NearestNeighbourMatch, ::AbstractMatrix, cands::AbstractVector,
                      latest::AbstractVector)
    l = neighbour_count(sel.neighbours, length(cands))
    if l < 1
        return Int[]
    end
    d = map(c -> LinearAlgebra.norm(c - latest), cands)
    return sort!(partialsortperm(d, 1:l))
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the periods whose preceding window correlates with the latest window at `rho` or more (CORN).

This is the correlation-driven rule of Li, Hoi and Gopalkrishnan (2011). The correlation is the Pearson correlation of the two flattened windows.

The paper's CORN-U mixes the experts of the windows 1 to ``W`` at one threshold, from a uniform start and by wealth. Its CORN-K runs over the windows 1 to ``W`` and the thresholds ``0, 1/P, \\ldots, (P - 1)/P``, and it keeps the ``K`` wealthiest experts, weighted by wealth. Both are an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each, CORN-K under the [`TopK`](@ref) weighting. The paper sets ``W = 5``, ``P = 10`` and ``K = 5``, and it runs CORN-U at the threshold 0.1 with no tuning. The defaults of one selector take the window 5 and the threshold 0.1 from these values.

Wang, Wang, Wang and Zhang (2018) add a risk penalty to each expert (RACORN-K). Their programme maximises the mean log return over the sample less ``\\lambda`` times the standard deviation of the log return ``\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle``. Their experts run over windows, thresholds and values of ``\\lambda``, and the wealthiest tenth of them combine as in CORN-K. The nearest programme of the library is [`MeanRisk`](@ref) under [`MaximumUtility`](@ref) at `l = λ` over [`StandardDeviation`](@ref), with [`LogarithmicReturn`](@ref). It penalises the standard deviation of the return ``\\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle - 1`` in place of the log return, and the two penalties agree to the first order in the return.

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : \\rho\\left(\\boldsymbol{x}_{i-w}^{i-1},\\, \\boldsymbol{x}_{t-w+1}^{t}\\right) \\geq \\bar{\\rho} \\right\\rbrace\\,, \\\\
\\rho(\\boldsymbol{u}, \\boldsymbol{v}) &= \\begin{cases} \\dfrac{\\operatorname{cov}(\\boldsymbol{u}, \\boldsymbol{v})}{\\sigma(\\boldsymbol{u})\\, \\sigma(\\boldsymbol{v})} & \\text{if } \\sigma(\\boldsymbol{u})\\, \\sigma(\\boldsymbol{v}) > 0\\,, \\\\ 0 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - $(math_dict[:w_win])
  - $(math_dict[:x_win])
  - ``\\rho(\\boldsymbol{u}, \\boldsymbol{v})``: Correlation of two flattened windows, taken over their entries.
  - ``\\operatorname{cov}``, ``\\sigma``: Covariance and standard deviation over the entries of a flattened window.
  - ``\\bar{\\rho}``: Threshold, the `rho` field.
  - $(math_dict[:t_period])

A window with no variation correlates at zero, as the paper sets it. So it matches at a threshold of zero or less, and at no positive threshold.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CorrelationMatch(; window::Integer = 5, rho::Real = 0.1) -> CorrelationMatch

Keywords correspond to the struct's fields.

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `-1 <= rho <= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> CorrelationMatch()
CorrelationMatch
  window ┼ Int64: 5
     rho ┴ Float64: 0.1
```

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`ExpertMixture`](@ref): CORN-U and CORN-K combine rules over this selector.
  - [`TopK`](@ref): the weighting of CORN-K and RACORN-K.

# References

  - $(ref_dict[:li2011corn]) Equation 6, Algorithms 1 to 3 and Section 5.2.
  - $(ref_dict[:wang2018racorn]) Equation 3.
"""
struct CorrelationMatch{T1 <: Integer, T2 <: Real} <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window::T1
    """
    The correlation threshold a candidate window must reach.
    """
    rho::T2
    function CorrelationMatch(window::Integer, rho::Real)
        assert_pattern_window(window)
        @argcheck(-one(rho) <= rho <= one(rho), DomainError(rho, "rho must lie in [-1, 1]"))
        return new{typeof(window), typeof(rho)}(window, rho)
    end
end
function CorrelationMatch(; window::Integer = 5, rho::Real = 0.1)::CorrelationMatch
    return CorrelationMatch(window, rho)
end
function matched_rows(sel::CorrelationMatch, ::AbstractMatrix, cands::AbstractVector,
                      latest::AbstractVector)
    return findall(cands) do c
        r = Statistics.cor(c, latest)
        # A window with no variation has no correlation, and the paper sets it to zero.
        return ifelse(isnan(r), zero(r), r) >= sel.rho
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Selects the periods whose preceding window falls in the same cluster as the latest window (KMNLOG).

This follows the cluster-based rule of Khedmati and Azin (2020). The paper clusters the windows by k-means (KMNLOG), k-medoids (KMDLOG), spectral clustering (SPCLOG) and hierarchical clustering (HRCLOG). The `clusterer` field takes one of the library's clustering estimators, which partitions the windows through a distance that it forms from them. [`KMeansAlgorithm`](@ref) and [`HClustAlgorithm`](@ref) are among its algorithms. The paper puts a transaction cost into its programme, and the `fees` slot of the rule's `opt` holds that cost. Fewer than two candidates cannot be partitioned, so they give the empty sample.

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : \\kappa\\left(\\boldsymbol{x}_{i-w}^{i-1}\\right) = \\kappa\\left(\\boldsymbol{x}_{t-w+1}^{t}\\right) \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:C_t_sample])
  - $(math_dict[:w_win])
  - $(math_dict[:x_win])
  - ``\\kappa``: Cluster label of a window, from one clustering of the ``t - w`` candidate windows and the latest window together.
  - $(math_dict[:t_period])

# Algorithm

The steps of `matched_rows` on this selector:

 1. When `cands` holds fewer than two windows, return the empty vector.
 2. Lay the candidate windows and `latest` out as the columns of `M`, with `latest` in the last column.
 3. Cluster the columns of `M` with `clusterise` on `sel.clusterer`, giving `labels`.
 4. Return the positions of the candidates whose label is the label of `latest`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClusterMatch(;
        window::Integer = 5,
        clusterer::AbstractClustersEstimator = ClustersEstimator(; alg = KMeansAlgorithm())
    ) -> ClusterMatch

Keywords correspond to the struct's fields.

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ClusterMatch(; clusterer = ClustersEstimator()).window
5
```

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`ClustersEstimator`](@ref): the default clusterer.
  - [`KMeansAlgorithm`](@ref): the default algorithm of the clusterer, the paper's KMNLOG.
  - [`HClustAlgorithm`](@ref): the paper's HRCLOG.

# References

  - $(ref_dict[:khedmatiazin2020])
"""
struct ClusterMatch{T1 <: Integer, T2 <: AbstractClustersEstimator} <:
       AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window::T1
    """
    The clustering estimator that partitions the windows.
    """
    clusterer::T2
    function ClusterMatch(window::Integer, clusterer::AbstractClustersEstimator)
        assert_pattern_window(window)
        return new{typeof(window), typeof(clusterer)}(window, clusterer)
    end
end
function ClusterMatch(; window::Integer = 5,
                      clusterer::AbstractClustersEstimator = ClustersEstimator(;
                                                                               alg = KMeansAlgorithm()))::ClusterMatch
    return ClusterMatch(window, clusterer)
end
function matched_rows(sel::ClusterMatch, ::AbstractMatrix, cands::AbstractVector,
                      latest::AbstractVector)
    n = length(cands)
    if n < 2
        return Int[]
    end
    M = Matrix{eltype(latest)}(undef, length(latest), n + 1)
    for (j, c) in enumerate(cands)
        M[:, j] .= c
    end
    M[:, n + 1] .= latest
    labels = Clustering.assignments(clusterise(sel.clusterer, M; dims = 1))
    return findall(==(labels[n + 1]), view(labels, 1:n))
end
"""
$(DocStringExtensions.TYPEDEF)

Adds the head's Allocation Set to the programme of a follow-the-leader re-solve, the Allocation Set Constraint.

The type is a [`CustomJuMPConstraint`](@ref) that holds the resolved set, the Price-Adjusted Allocation of the step and the head's rows. A [`FollowTheLeader`](@ref) rule appends it to the `ccnt` of its held optimiser for one solve. [`add_custom_constraint!`](@ref) on it calls [`add_allocation_set_constraints!`](@ref), which runs the set's own builders on the leader's model. So the feasible region of the programme is the set intersected with the region of the held optimiser. The turnover ceiling measures from the Price-Adjusted Allocation. The variance ceiling and the tracking error read the set's own `pe` on the head's rows, as they do in a projection. A constraint can live on the set or on the held optimiser, and a caller who wants one home leaves the other bare.

# Fields

$(DocStringExtensions.FIELDS)

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets. `set` takes its own view, against the unreduced returns matrix of the carrier when one exists, because the view of a tracking estimator asks for it. The view slices `w`, and `X` takes the view of the carrier. The reduction of the held estimator to its Investable Mask views its `ccnt` with the rest of the optimiser. So the set stays in the reduced programme. A custom constraint with no view drops out of it.

# Related

  - [`FollowTheLeader`](@ref)
  - [`add_allocation_set_constraints!`](@ref): the builders that this constraint calls.
  - [`CustomJuMPConstraint`](@ref)
  - [`AbstractAllocationSet`](@ref)
  - [`investable_reduction`](@ref): the reduction that views this constraint with the held optimiser.
"""
struct AllocationSetConstraint{T1 <: AbstractAllocationSet, T2 <: AbstractVector,
                               T3 <: Option{<:ReturnsResult}} <: CustomJuMPConstraint
    """
    The Allocation Set, resolved over the pinned universe.
    """
    set::T1
    """
    The Price-Adjusted Allocation the step trades from.
    """
    w::T2
    """
    The rows carrier the head holds through the period, a [`ReturnsResult`](@ref), or `nothing`.
    """
    X::T3
end
function add_custom_constraint!(model::JuMP.Model, ccnt::AllocationSetConstraint, ::Any,
                                ::Any)::Nothing
    add_allocation_set_constraints!(model, ccnt.set, ccnt.w, ccnt.X)
    return nothing
end
function port_opt_view(c::AllocationSetConstraint{<:Any, <:Any, Nothing}, i, args...)
    return AllocationSetConstraint(port_opt_view(c.set, i, args...), c.w[i], nothing)
end
function port_opt_view(c::AllocationSetConstraint{<:Any, <:Any, <:ReturnsResult}, i,
                       args...)
    return AllocationSetConstraint(port_opt_view(c.set, i, c.X.X), c.w[i],
                                   port_opt_view(c.X, i))
end
"""
    const LeaderOptimiser = Union{<:BestConstantRebalancedPortfolio, <:JuMPOptimisationEstimator}

Groups the optimisation estimators that a follow-the-leader rule can re-solve under the head's Allocation Set.

A leader of a [`FollowTheLeader`](@ref) rule must honour the set. The solver-free [`BestConstantRebalancedPortfolio`](@ref) takes a bounded set on its own `wb` and `sets` slots. A JuMP estimator takes any set through the [`AllocationSetConstraint`](@ref) on its `ccnt`. No other estimator has a slot for the set, so the alias excludes it.

# Related

  - [`FollowTheLeader`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
const LeaderOptimiser = Union{<:BestConstantRebalancedPortfolio,
                              <:JuMPOptimisationEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Re-solves an optimisation estimator on a sample of past periods at every step, the follow-the-leader rule.

The Sample Selector on `sel` names the sample, and the estimator on `opt` solves on it. The solution is the leader. The rule plays the leader, or at `gamma > 0` the leader damped towards the allocation that it held.

Every solved rule of the online selection literature is this rule under one selector and one objective. [`Prefix`](@ref) under the log-optimal objective is follow the leader, the successive constant rebalanced portfolio of Gaivoronski and Stella (2000). At `gamma > 0` it is their weighted rule (WSCRP). [`LastRows`](@ref) gives their successive variable rebalanced portfolio. The pattern-matching selectors give the histogram, kernel, nearest-neighbour, correlation and cluster rules. The papers of those rules mix a grid of selectors by wealth, which is an [`ExpertMixture`](@ref) over rules of this kind.

The objective is the held estimator's. [`BestConstantRebalancedPortfolio`](@ref), the default, is the log-optimal portfolio with no solver. [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref) solves the same programme on a solver. Under [`Prefix`](@ref) and on the simplex, that estimator with `l2 = L2Regularisation(; val = 1 / (2t), alg = QuadRiskExpr())` on its [`JuMPOptimiser`](@ref) and ``t`` rows is the Exp-Concave-FTL of Hazan and Kale (2015). The paper subtracts ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} \\rVert^2`` from the sum of the log returns, and [`LogarithmicReturn`](@ref) states their mean, so the weight is ``1 / (2t)``. One fixed `val` is that leader at one sample size alone. [`MaximumUtility`](@ref) over a risk measure gives a mean-risk leader, and `fees` on the estimator gives a cost-aware leader.

The solver-free default runs Cover's fixed point, which stops on a certificate, a bound on the shortfall of its log wealth from the leader's. At a leader that drops an asset, the weight of that asset decays slowly. So the default budget can stop before the certificate meets `tol`, and the rule then warns once and names the certificate. On the first 30 rows of the returns `0.02 .* randn(StableRNG(11), 40, 4)` the default stops 0.066 from the leader in weight and 6.2e-5 short of it in log wealth. At `iters = 10_000_000` the certificate meets `tol = 1e-12` after 1 437 030 steps, 1.6e-9 from the leader in weight. A JuMP head solves the leader to the solver's tolerance, and the two forms agree to that tolerance at a leader inside the simplex.

**The re-solve takes the head's Allocation Set as its feasible region.** The rule wraps the set, the Price-Adjusted Allocation and the head's rows into an [`AllocationSetConstraint`](@ref), and appends it to the `ccnt` of the held JuMP estimator for the solve. It passes the Price-Adjusted Allocation through [`factory`](@ref) as a fold loop does, so a turnover term or a fee on `opt` reads the same book. It plays the optimum with no projection, because the projected leader is not the leader. The solver-free estimator has no model. So a [`BoundedAllocationSet`](@ref) goes into its `wb` and `sets`, and the rule plays its repaired fixed point. The head refuses a [`ProgrammeAllocationSet`](@ref) over the solver-free estimator at construction. Under a programme set the constrained leader is [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref).

**The rule projects the damped mix alone.** It uses the Euclidean geometry on `proj`, with the Price-Adjusted Allocation as the reference. On a static convex set the projection returns the mix unchanged. A turnover ceiling or a MIP kind needs the repair on a day when the mix leaves the set. The rule skips the projection by type on a [`BoundedAllocationSet`](@ref), and at `gamma = 0`. An empty sample, or a sample too small for the estimator, gives the uniform portfolio projected onto the set, as the papers do. A JuMP head fits a covariance, so it needs two rows. A re-solve that fails after the fallback chain of the held estimator is a Held Step. The rule then plays the Price-Adjusted Allocation, so the fund trades nothing that period.

**Under a time-varying panel the re-solve is the batch path.** The estimator reads the head's rows carrier viewed at the sample, with the returns as they are, `NaN` where no return exists, and the active mask. So the estimator reduces to its own universe, as it does on any carrier, and writes a zero at every asset outside it. That universe is the Coverage Universe of the sample for the solver-free leader and for a head with no prior, and the Investable Mask of its prior for a JuMP head. An asset that is not listed for a part of the sample stays outside the universe of a plain estimator until the sample clears that span. A [`Prefix`](@ref) sample never clears it, and a [`LastRows`](@ref) sample clears it when its window has passed the listing. A prior that reads the mask admits the asset after its own warm-up. The selector reads a gap as a price relative of one, as if the asset sat in cash, so it compares a window that spans a listing and does not drop it. The damped mix and the projection read a zero from the leader at such an asset as they read any other zero.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^\\star_t &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\max}\\; \\sum_{i \\in C_t} \\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( (1 - \\gamma)\\, \\boldsymbol{w}^\\star_t + \\gamma\\, \\boldsymbol{w}_t \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_star_lead]) The first line is the leader under the default log-optimal estimator. Another estimator on `opt` replaces the objective.
  - $(math_dict[:w_var_lead])
  - $(math_dict[:W_aset])
  - $(math_dict[:C_t_sample])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - ``\\gamma``: Damping towards the held allocation, in ``[0, 1]``, the `gamma` field.
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:t_period])

At ``\\gamma = 0`` the rule plays the leader. The leader lies in ``\\mathcal{W}``, so the mix leaves ``\\mathcal{W}`` only when ``\\boldsymbol{w}_t`` is outside it, as under a turnover ceiling, or when ``\\mathcal{W}`` is not convex, as under a cardinality bound. Under the log-optimal estimator and [`Prefix`](@ref), the leader on ``\\mathcal{W} = \\Delta_N`` is the best constant rebalanced portfolio of the first ``t`` periods. The Exp-Concave-FTL of Hazan and Kale is

```math
\\begin{align}
\\boldsymbol{w}^\\star_t &= \\underset{\\boldsymbol{w} \\in \\Delta_N}{\\arg\\max}\\; \\sum_{i=1}^{t} \\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle - \\frac{1}{2} \\lVert \\boldsymbol{w} \\rVert_2^2\\,.
\\end{align}
```

Where:

  - $(math_dict[:Delta_N_simplex])
  - $(math_dict[:N])

# Algorithm

The steps of the Online Update:

 1. Compute `wh`, the Price-Adjusted Allocation of `w` over `x`.
 2. Select `idx`, the rows of the sample, with `select_rows` on `alg.sel` and the price relatives of `rows`.
 3. When `idx` holds fewer rows than `leader_min_rows(alg.opt)`, set `wstar` to the uniform portfolio projected onto `set` with `alg.proj`, from `wh`. Otherwise set `wstar` to the result of `leader_allocation` on `alg.opt`, the view of `rows` at `idx`, `wh`, `set` and `rows`.
 4. When `wstar` is `nothing`, the step is a Held Step. Return the carrier `st` and a copy of `wh`.
 5. When `alg.gamma` is zero, return `st` and `wstar`.
 6. Set `mix` to `(1 - alg.gamma) .* wstar .+ alg.gamma .* w`. Return `st` and the result of `blend_projection` of `mix` onto `set` with `alg.proj`, from `wh`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FollowTheLeader(;
        sel::AbstractSampleSelector = Prefix(),
        opt::LeaderOptimiser = BestConstantRebalancedPortfolio(),
        gamma::Real = 0,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> FollowTheLeader

Keywords correspond to the struct's fields.

## Validation

  - `0 <= gamma <= 1`. A `DomainError` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, `opt` is viewed to the selected indices and `sel`, `gamma` and `proj` are carried unchanged.

# Examples

```jldoctest
julia> FollowTheLeader()
FollowTheLeader
    sel ┼ Prefix()
    opt ┼ BestConstantRebalancedPortfolio
        │       wb ┼ WeightBounds
        │          │   lb ┼ Float64: 0.0
        │          │   ub ┴ Float64: 1.0
        │     fees ┼ nothing
        │     sets ┼ nothing
        │       wf ┼ IterativeWeightFinaliser
        │          │   iter ┴ Int64: 100
        │       fb ┼ nothing
        │    iters ┼ Int64: 20000
        │      tol ┼ Float64: 1.0e-12
        │   strict ┴ Bool: false
  gamma ┼ Int64: 0
   proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`AbstractSampleSelector`](@ref)
  - [`AllocationSetConstraint`](@ref): the adapter that adds the head's set to the programme.
  - [`BestConstantRebalancedPortfolio`](@ref): the default estimator.
  - [`ExpertMixture`](@ref): the mixture by wealth over a grid of these rules.
  - [`FollowTheLeadingHistory`](@ref)
  - [`ShortTermLossControlPortfolio`](@ref): this rule over [`LastRows`](@ref) under a loss-control programme.
  - [`LowDimensionEnsemblePortfolio`](@ref): this rule over [`LastRows`](@ref) under an ensemble forecast.

# References

  - $(ref_dict[:gaivoronski2000])
  - $(ref_dict[:hazankale2012]) Section 2.2, Equation 1.
  - $(ref_dict[:lihoi2014]) Sections 3.2.3 and 3.4.
"""
struct FollowTheLeader{T1 <: AbstractSampleSelector, T2 <: LeaderOptimiser, T3 <: Real,
                       T4 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The Sample Selector that names the rows the estimator re-solves on.
    """
    sel::T1
    """
    The optimisation estimator re-solved on the selected rows.
    """
    opt::T2
    """
    The damping towards the held allocation, in `[0, 1]`. At `0` the rule plays the leader.
    """
    gamma::T3
    """
    $(field_dict[:proj])
    """
    proj::T4
    function FollowTheLeader(sel::T1, opt::T2, gamma::T3,
                             proj::T4) where {T1 <: AbstractSampleSelector,
                                              T2 <: LeaderOptimiser, T3 <: Real,
                                              T4 <: EuclideanProjection}
        @argcheck(zero(gamma) <= gamma <= one(gamma),
                  DomainError(gamma, "gamma must lie in [0, 1]"))
        return new{T1, T2, T3, T4}(sel, opt, gamma, proj)
    end
end
function FollowTheLeader(; sel::AbstractSampleSelector = Prefix(),
                         opt::LeaderOptimiser = BestConstantRebalancedPortfolio(),
                         gamma::Real = 0,
                         proj::EuclideanProjection = EuclideanProjection())::FollowTheLeader
    return FollowTheLeader(sel, opt, gamma, proj)
end
function port_opt_view(alg::FollowTheLeader, i, args...)
    return FollowTheLeader(; sel = alg.sel, opt = port_opt_view(alg.opt, i, args...),
                           gamma = alg.gamma, proj = alg.proj)
end
function rows_needed(alg::FollowTheLeader)
    return rows_needed(alg.sel)
end
function assert_rule_admits_set(alg::FollowTheLeader{<:Any,
                                                     <:BestConstantRebalancedPortfolio},
                                ::ProgrammeAllocationSet)::Nothing
    return throw(ArgumentError("a `FollowTheLeader` over the solver-free `BestConstantRebalancedPortfolio` has no model a `ProgrammeAllocationSet` can enter: the constrained leader under a programme set is `MeanRisk` under `LogarithmicReturn` and `MaximumReturn`, with the set's solver, on `opt`."))
end
"""
    leader_min_rows(opt::LeaderOptimiser)

Returns the fewest sample rows on which the held estimator can re-solve.

An estimator whose tree holds a second moment needs two rows, because the covariance of one observation does not exist. Any estimator in the tree can state a larger floor through [`fit_min_rows`](@ref).

# Arguments

  - `opt`: The held estimator.

# Returns

  - `n::Integer`: The larger of `2` for a tree that holds a second moment, `1` otherwise, and `fit_min_rows(opt)`.

# Related

  - [`FollowTheLeader`](@ref)
  - [`holds_second_moment`](@ref)
  - [`fit_min_rows`](@ref)
"""
function leader_min_rows(opt::LeaderOptimiser)
    return max(holds_second_moment(opt) ? 2 : 1, fit_min_rows(opt))
end
"""
    append_custom_constraint(ccnt::Nothing, c::CustomJuMPConstraint)
    append_custom_constraint(ccnt::CustomJuMPConstraint, c::CustomJuMPConstraint)
    append_custom_constraint(ccnt::AbstractVector{<:CustomJuMPConstraint}, c::CustomJuMPConstraint)

Appends the Allocation Set Constraint to the custom constraints of the held estimator.

# Arguments

  - `ccnt`: The custom constraints of the held estimator, `nothing`, one constraint, or a vector of them.
  - `c`: The [`AllocationSetConstraint`](@ref) to append.

# Returns

  - `c` alone when `ccnt` is `nothing`, the vector `[ccnt, c]` for one constraint, and `vcat(ccnt, c)` for a vector.

# Related

  - [`FollowTheLeader`](@ref)
"""
function append_custom_constraint(::Nothing, c::CustomJuMPConstraint)
    return c
end
function append_custom_constraint(ccnt::CustomJuMPConstraint, c::CustomJuMPConstraint)
    return [ccnt, c]
end
function append_custom_constraint(ccnt::AbstractVector{<:CustomJuMPConstraint},
                                  c::CustomJuMPConstraint)
    return vcat(ccnt, c)
end
"""
    leader_allocation(opt::BestConstantRebalancedPortfolio, rd::ReturnsResult, w::AbstractVector, set::BoundedAllocationSet, X)
    leader_allocation(opt::JuMPOptimisationEstimator, rd::ReturnsResult, w::AbstractVector, set::AbstractAllocationSet, X)

Re-solves the held estimator on the sample under the head's Allocation Set, and returns the leader, or `nothing` on a Held Step.

The estimator runs the batch path that it runs on any carrier. An estimator with no prior reduces to the Coverage Universe of the sample, an estimator with a prior reduces to the Investable Mask of its prior, and both return a zero at every asset outside it. A leader that stopped short of its certificate still trades, so the solver-free method warns and plays the fixed point.

# Algorithm

The solver-free method:

 1. Build `bcrp`, a copy of `opt` whose `wb` and `sets` are those of `set`.
 2. Optimise `factory(bcrp, w)` on `rd`, giving `res`.
 3. When the fixed point stopped at `iters` before its certificate met `tol`, warn once and name the certificate.
 4. Return `res.w`.

The JuMP method:

 1. Pass `w` into `opt` with [`factory`](@ref).
 2. Append an [`AllocationSetConstraint`](@ref) over `set`, `w` and `X` to the `ccnt` of the estimator.
 3. Optimise on `rd`, giving `res`. When its retcode is an [`OptimisationSuccess`](@ref), return `res.w`.
 4. Otherwise record a Held Step through [`record_held_step!`](@ref) with the trials of the solver, and return `nothing`.

# Arguments

  - `opt`: The held estimator.
  - `rd`: The head's rows carrier viewed at the sample. It holds the returns as they are, `NaN` where no return exists, under the pinned names and the Asset Panel of the buffer.
  - `w`: The Price-Adjusted Allocation of the step.
  - `set`: The resolved Allocation Set of the head, a [`BoundedAllocationSet`](@ref) for the solver-free method.
  - `X`: The head's rows carrier over every row that it holds, or `nothing`. The JuMP method passes it to the set's builders, and the solver-free method does not read it.

# Returns

  - `w::Union{Nothing, AbstractVector}`: The leader, or `nothing` on a Held Step.

# Related

  - [`FollowTheLeader`](@ref)
  - [`AllocationSetConstraint`](@ref)
  - [`HeldStep`](@ref)
"""
function leader_allocation(opt::BestConstantRebalancedPortfolio, rd::ReturnsResult,
                           w::AbstractVector, set::BoundedAllocationSet, ::Any)
    bcrp = BestConstantRebalancedPortfolio(; wb = set.wb, fees = opt.fees, sets = set.sets,
                                           wf = opt.wf, fb = opt.fb, iters = opt.iters,
                                           tol = opt.tol, strict = opt.strict,
                                           cache = opt.cache)
    res = optimise(factory(bcrp, w), rd)
    fp = isa(res.retcode, OptimisationSuccess) ? res.retcode.res : nothing
    if isa(fp, NamedTuple) && !get(fp, :converged, true)
        @warn("the solver-free leader stopped at `iters = $(opt.iters)` on $(size(rd.X, 1)) selected rows, $(fp.gap) or less short of the optimal log wealth; later short steps are not reported. The exact leader is `MeanRisk` under `LogarithmicReturn` and `MaximumReturn` on a solver.",
              maxlog = 1)
    end
    return res.w
end
function leader_allocation(opt::JuMPOptimisationEstimator, rd::ReturnsResult,
                           w::AbstractVector, set::AbstractAllocationSet,
                           X::Option{<:ReturnsResult})
    opt = factory(opt, w)
    ccnt = append_custom_constraint(opt.opt.ccnt, AllocationSetConstraint(set, w, X))
    opt = Accessors.@set opt.opt.ccnt = ccnt
    res = optimise(opt, rd)
    if isa(res.retcode, OptimisationSuccess)
        return res.w
    end
    record_held_step!("the follow-the-leader programme of the `$(nameof(typeof(opt)))` did not solve on the $(size(rd.X, 1)) selected rows, and the step trades nothing",
                      res.retcode.res)
    return nothing
end
function online_update!(alg::FollowTheLeader, st, w::AbstractVector, x::AbstractVector,
                        rows::ReturnsResult, set::AbstractAllocationSet)
    wh = price_adjusted_allocation(w, x)
    idx = select_rows(alg.sel, price_relative.(rows.X))
    wstar = if length(idx) < leader_min_rows(alg.opt)
        project(alg.proj, set, fill(one(eltype(w)) / length(w), length(w)), wh)
    else
        leader_allocation(alg.opt, port_opt_view(rows, idx, :), wh, set, rows)
    end
    if isnothing(wstar)
        return st, copy(wh)
    end
    if iszero(alg.gamma)
        return st, wstar
    end
    mix = (one(alg.gamma) - alg.gamma) .* wstar .+ alg.gamma .* w
    return st, blend_projection(alg.proj, set, mix, wh)
end
"""
    ShortTermLossControlPortfolio(; window::Integer = 5, gamma::Real = 0.025, slv::Slv_VecSlv, pe::AbstractPriorEstimator = EmpiricalPrior(; ce = RankOneCovariance()), proj::EuclideanProjection = EuclideanProjection())

Builds the short-term portfolio optimisation with loss control of Lai, Tan, Wu and Fang (2020) (SPOLC).

The rule is a [`FollowTheLeader`](@ref) over the last `window` periods. Its programme maximises the worst increasing factor of the window less `gamma` times the portfolio variance under the [`RankOneCovariance`](@ref). [`MeanRisk`](@ref) states it as the minimum of [`WorstRealisation`](@ref) plus [`Variance`](@ref) scaled by `gamma`. The variance reads the rank-one matrix as a quadratic form under [`QuadRiskExpr`](@ref), which needs no factor of the singular matrix. The head's Allocation Set enters the programme as in every follow-the-leader programme. The paper does not state the rule before the window fills. Here the rule re-solves on the periods that it has, and on one period it plays the uniform portfolio, because the programme reads a covariance.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^\\star_t &= \\underset{\\boldsymbol{w} \\in \\Delta_N}{\\arg\\max}\\; \\min_{i \\in C_t} \\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle - \\gamma\\, \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}}_{\\mathrm{RO}} \\boldsymbol{w}\\,, \\\\
C_t &= \\left\\lbrace \\max(1,\\, t - w + 1), \\ldots, t \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_star_lead])
  - $(math_dict[:w_var_lead])
  - $(math_dict[:Delta_N_simplex])
  - $(math_dict[:C_t_sample])
  - $(math_dict[:x_t_rel])
  - ``\\gamma``: Weight of the variance against the worst increasing factor, the paper's ``\\gamma``.
  - ``\\hat{\\mathbf{\\Sigma}}_{\\mathrm{RO}}``: Rank-one covariance estimate of the paper on the sample.
  - ``w``: Window of the rule, the paper's ``w``.
  - $(math_dict[:N])
  - $(math_dict[:t_period])

This is the paper's equation 51 at period ``t``. On the simplex ``\\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle - 1`` is the portfolio return of period ``i``. So the worst increasing factor less one is the negative of the worst realisation over the sample, and the two programmes have one maximiser.

# Arguments

  - `window`: The paper's ``w``, the number of periods of the window.
  - `gamma`: The paper's ``\\gamma``, the weight of the variance against the worst increasing factor.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme, whose covariance is the rank-one estimate.
  - `proj`: The geometry of the damped mix. The rule's `gamma = 0` does not read it.

# Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `gamma >= 0`. A `DomainError` is thrown otherwise.

# Returns

  - `alg::FollowTheLeader`: The rule over [`LastRows`](@ref) at `W = window`, whose `opt` is the loss-control programme.

# Examples

```jldoctest
julia> alg = ShortTermLossControlPortfolio(; slv = Solver(; solver = nothing));

julia> alg.sel
LastRows
  W ┴ Int64: 5

julia> alg.opt.r[2].settings.scale
0.025
```

# Related

  - [`FollowTheLeader`](@ref)
  - [`LastRows`](@ref)
  - [`RankOneCovariance`](@ref): the paper's rank-one estimate.
  - [`WorstRealisation`](@ref)
  - [`Variance`](@ref)
  - [`MeanRisk`](@ref)

# References

  - $(ref_dict[:lai2020spolc]) Equation 51 and Section 4.2.1.
"""
function ShortTermLossControlPortfolio(; window::Integer = 5, gamma::Real = 0.025,
                                       slv::Slv_VecSlv,
                                       pe::AbstractPriorEstimator = EmpiricalPrior(;
                                                                                   ce = RankOneCovariance()),
                                       proj::EuclideanProjection = EuclideanProjection())::FollowTheLeader
    @argcheck(gamma >= zero(gamma), DomainError(gamma, "gamma must be non-negative"))
    return FollowTheLeader(; sel = LastRows(; W = window),
                           opt = loss_control_programme(gamma, slv, pe), proj = proj)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the programme of the short-term portfolio optimisation with loss control.

The programme is [`MeanRisk`](@ref) under [`MinimumRisk`](@ref) over [`WorstRealisation`](@ref) plus [`Variance`](@ref) scaled by `gamma` under [`QuadRiskExpr`](@ref). The method is typed on the solver and the prior, so the optimiser that it builds is concrete for them.

# Arguments

  - `gamma`: The weight of the variance.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme.

# Returns

  - `opt::MeanRisk`: The programme.

# Related

  - [`ShortTermLossControlPortfolio`](@ref)
"""
function loss_control_programme(gamma::Real, slv::T1,
                                pe::T2) where {T1 <: Slv_VecSlv,
                                               T2 <: AbstractPriorEstimator}
    return MeanRisk(;
                    r = [WorstRealisation(),
                         Variance(; alg = QuadRiskExpr(),
                                  settings = RiskMeasureSettings(; scale = gamma))],
                    obj = MinimumRisk(), opt = JuMPOptimiser(; pe = pe, slv = slv))
end
"""
    LowDimensionEnsemblePortfolio(; N::Integer, window::Integer = 5, gamma::Real = 0.25, xi::Real = 0.002, slv::Slv_VecSlv, pe::AbstractPriorEstimator = LowDimensionEnsemblePrior(), proj::EuclideanProjection = EuclideanProjection())

Builds the online low-dimension ensemble method of Xi, Li, Song and Ning (2023) (OLDEM).

The rule is a [`FollowTheLeader`](@ref) over the last `window` regression pairs. Its programme maximises the forecast return of the ensemble, less `gamma` times the portfolio variance under the predictive covariance of the ensemble, less a linear turnover fee at the rate `xi`. [`MeanRisk`](@ref) states it as [`MaximumUtility`](@ref) at the risk aversion `gamma` over [`Variance`](@ref) under [`QuadRiskExpr`](@ref). The ensemble prior on `pe` gives the forecast and the covariance from one fit. A [`Fees`](@ref) whose [`Turnover`](@ref) rate is `xi` gives the ``\\ell_1`` term, and the head passes the reference allocation through [`factory`](@ref), as it does for every fee. The head's Allocation Set enters the programme as in every follow-the-leader programme.

The paper relaxes ``\\boldsymbol{w} \\geq \\boldsymbol{0}``, runs a coordinate-wise descent on the change of the allocation, and projects the result onto the simplex. The programme here keeps ``\\boldsymbol{w} \\geq \\boldsymbol{0}`` inside the solve. So it returns the minimiser of the paper's programme, which the relaxed and projected result is not in general. `window` regression pairs need `window + 1` rows, so the selector holds one row more than the paper's ``w``. Before the window fills the rule re-solves on the rows that it has. Below three rows the covariance of the regressors does not exist, and the rule plays the uniform portfolio.

The turnover fee holds a reference allocation of the universe's length from construction, and the head replaces it at every update. So the constructor takes the number of assets `N`, as [`UniversalPortfolio`](@ref) does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^\\star_t &= \\underset{\\boldsymbol{w} \\in \\Delta_N}{\\arg\\min}\\; -\\hat{\\boldsymbol{x}}_{t+1}^\\intercal \\boldsymbol{w} + \\gamma\\, \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}}_{t+1} \\boldsymbol{w} + \\xi\\, \\lVert \\boldsymbol{w} - \\hat{\\boldsymbol{w}}_t \\rVert_1\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_star_lead])
  - $(math_dict[:w_var_lead])
  - $(math_dict[:Delta_N_simplex])
  - $(math_dict[:xhat_fc])
  - ``\\gamma``: Risk aversion over the predictive variance, the paper's ``\\gamma``.
  - $(math_dict[:Sigma_hat_pred])
  - ``\\xi``: Rate of the linear turnover fee, the paper's ``\\xi``.
  - $(math_dict[:w_hat_t_padj])
  - $(math_dict[:N])
  - $(math_dict[:t_period])

This is the paper's equation 14, with one change. The paper measures the turnover from its own last allocation ``\\hat{\\boldsymbol{b}}_t``, and the programme here measures it from the Price-Adjusted Allocation ``\\hat{\\boldsymbol{w}}_t``, the book that the fund holds when it trades. On the simplex ``\\hat{\\boldsymbol{x}}^\\intercal \\boldsymbol{w}`` and ``(\\hat{\\boldsymbol{x}} - \\boldsymbol{1})^\\intercal \\boldsymbol{w}`` differ by one, so the programme over the forecast return has the same minimiser.

# Arguments

  - `N`: The number of assets, the length of the reference allocation of the fee.
  - `window`: The paper's ``w``, the number of regression pairs of the window. The selector holds `window + 1` rows.
  - `gamma`: The paper's ``\\gamma``, the risk aversion over the predictive variance.
  - `xi`: The paper's ``\\xi``, the rate of the linear turnover fee.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme, the ensemble by default. Any prior gives the two moments that the programme reads.
  - `proj`: The geometry of the damped mix. The rule's `gamma = 0` does not read it.

# Validation

  - `N >= 1`. A `DomainError` is thrown otherwise.
  - `window >= 2`, the fewest pairs whose regressor covariance exists. A `DomainError` is thrown otherwise.
  - `gamma >= 0`. A `DomainError` is thrown otherwise.
  - `xi >= 0`. A `DomainError` is thrown otherwise.

# Returns

  - `alg::FollowTheLeader`: The rule over [`LastRows`](@ref) at `W = window + 1`, whose `opt` is the ensemble programme.

# Examples

```jldoctest
julia> alg = LowDimensionEnsemblePortfolio(; N = 3, slv = Solver(; solver = nothing));

julia> alg.sel
LastRows
  W ┴ Int64: 6

julia> alg.opt.obj
MaximumUtility
  l ┴ Float64: 0.25

julia> alg.opt.opt.fees.tn
Turnover
      w ┼ Vector{Float64}: [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
    val ┼ Float64: 0.002
  fixed ┴ Bool: false
```

# Related

  - [`FollowTheLeader`](@ref)
  - [`LastRows`](@ref)
  - [`LowDimensionEnsemblePrior`](@ref): the ensemble that gives the forecast and the predictive covariance.
  - [`MaximumUtility`](@ref)
  - [`Variance`](@ref)
  - [`Fees`](@ref)
  - [`Turnover`](@ref)
  - [`MeanRisk`](@ref)

# References

  - $(ref_dict[:xi2023oldem]) Equation 14, Section 4.2 and Section 5.3.
"""
function LowDimensionEnsemblePortfolio(; N::Integer, window::Integer = 5,
                                       gamma::Real = 0.25, xi::Real = 0.002,
                                       slv::Slv_VecSlv,
                                       pe::AbstractPriorEstimator = LowDimensionEnsemblePrior(),
                                       proj::EuclideanProjection = EuclideanProjection())::FollowTheLeader
    @argcheck(N >= 1, DomainError(N, "N must be at least 1"))
    @argcheck(window >= 2,
              DomainError(window,
                          "window must be at least 2: two regression pairs are the fewest whose regressor covariance exists"))
    @argcheck(gamma >= zero(gamma), DomainError(gamma, "gamma must be non-negative"))
    @argcheck(xi >= zero(xi), DomainError(xi, "xi must be non-negative"))
    w = fill(one(xi) / N, N)
    return FollowTheLeader(; sel = LastRows(; W = window + 1),
                           opt = ensemble_programme(gamma,
                                                    Fees(;
                                                         tn = Turnover(; w = w, val = xi)),
                                                    slv, pe), proj = proj)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the programme of the online low-dimension ensemble method.

The programme is [`MeanRisk`](@ref) under [`MaximumUtility`](@ref) at the risk aversion `gamma` over [`Variance`](@ref) under [`QuadRiskExpr`](@ref), and it charges `fees`. The method is typed on the solver and the prior, so the optimiser that it builds is concrete for them.

# Arguments

  - `gamma`: The risk aversion over the predictive variance.
  - `fees`: The fees of the programme, a [`Turnover`](@ref) fee for the rule.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme.

# Returns

  - `opt::MeanRisk`: The programme.

# Related

  - [`LowDimensionEnsemblePortfolio`](@ref)
"""
function ensemble_programme(gamma::Real, fees::Fees, slv::T1,
                            pe::T2) where {T1 <: Slv_VecSlv, T2 <: AbstractPriorEstimator}
    return MeanRisk(; r = Variance(; alg = QuadRiskExpr()),
                    obj = MaximumUtility(; l = gamma),
                    opt = JuMPOptimiser(; pe = pe, slv = slv, fees = fees))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the live experts of a follow-the-leading-history rule, with their Rule States, start periods and weights.

It is the carrier of [`FollowTheLeadingHistory`](@ref). The rule refuses a merge of two carriers, because the head's allocation depends on the order of the rows.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`FollowTheLeadingHistory`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct FollowTheLeadingHistoryState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The carrier of each live expert, `nothing` when the base rule carries nothing.
    """
    st
    """
    The allocation that each live expert holds during the current period.
    """
    h
    """
    The weight over the live experts during the current period.
    """
    p
    """
    The start period of each live expert, from which its lifetime counts.
    """
    born
end
function merge_states(::FollowTheLeadingHistoryState, ::FollowTheLeadingHistoryState)
    return throw(ArgumentError("a `FollowTheLeadingHistoryState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::FollowTheLeadingHistoryState)
    return FollowTheLeadingHistoryState(x.n, copy_column.(x.st), copy.(x.h), copy(x.p),
                                        copy(x.born))
end
function port_opt_view(x::FollowTheLeadingHistoryState, i, args...)
    return FollowTheLeadingHistoryState(x.n, map(s -> rule_state_view(s, i, args...), x.st),
                                        map(h -> renormalised_view(h, i), x.h), copy(x.p),
                                        copy(x.born))
end
"""
$(DocStringExtensions.TYPEDEF)

Mixes copies of one rule started at different periods, with a fixed share for the newest copy (FLH).

This is follow the leading history of Hazan and Seshadhri (2009). A multiplicative update weights the copies by their returns, and at every period a new copy starts with a fixed share of the weight. Under `prune` the rule keeps a working set of copies of logarithmic size. The paper's experiments use the Online Newton Step as the base rule, which is [`NewtonStep`](@ref) here, and any rule of the family serves.

A new copy starts at the base rule's own start from the uniform portfolio, projected onto the Allocation Set in the base rule's geometry. So a constant rebalanced base starts at the projection of its own `w`. The rule reads nothing of a given Start Allocation beyond its first copy, which starts there through [`project_start`](@ref). The rule projects the weight vector onto the Expert Set on `eset` in the entropic geometry, after the share and the pruning. So a cap on `eset` caps the trust in one copy. The working set changes size, so `eset` admits a scalar bound alone. The rule projects the blend onto the head's Allocation Set once more in the Euclidean geometry on `proj`, as an [`ExpertMixture`](@ref) does, and skips it by type on a [`BoundedAllocationSet`](@ref).

# Mathematical definition

```math
\\begin{align}
\\hat{p}_{t+1, k} &= \\frac{p_{t, k}\\, r_{t, k}^{\\alpha}}{\\sum_{j \\in S_t} p_{t, j}\\, r_{t, j}^{\\alpha}}\\,, \\quad k \\in S_t\\,, \\\\
\\bar{p}_{t+1, k} &= \\begin{cases} \\left( 1 - \\dfrac{1}{t + 1} \\right) \\hat{p}_{t+1, k} & k \\in S_t\\,, \\\\ \\dfrac{1}{t + 1} & k = t + 1\\,, \\end{cases} \\\\
p_{t+1, k} &= \\frac{\\bar{p}_{t+1, k}}{\\sum_{j \\in S_{t+1}} \\bar{p}_{t+1, j}}\\,, \\quad k \\in S_{t+1}\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\sum_{k \\in S_{t+1}} p_{t+1, k}\\, \\boldsymbol{h}_k(t+1)\\,.
\\end{align}
```

Where:

  - $(math_dict[:p_t_expert]) Here expert ``k`` is the copy started at period ``k``, and ``p_{1, 1} = 1``.
  - ``\\hat{p}_{t+1, k}``, ``\\bar{p}_{t+1, k}``: Weights after the multiplicative update, and after the share of the new copy.
  - $(math_dict[:r_t_expert])
  - $(math_dict[:h_kt_expert])
  - $(math_dict[:x_t_rel])
  - ``\\alpha``: Exponent of the multiplicative update, the `alpha` field.
  - ``S_t``: Working set of period ``t``, the start periods of the live copies, with ``S_1 = \\lbrace 1 \\rbrace``.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:t_period])

The factor ``r_{t, k}^{\\alpha}`` is the paper's ``e^{-\\alpha f_t}`` under the log-wealth loss ``f_t(\\boldsymbol{w}) = -\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle``, and at ``\\alpha = 1`` it is the wealth weighting. Without `prune`, ``S_{t+1} = \\lbrace 1, \\ldots, t + 1 \\rbrace`` and the last normalisation divides by one. Under `prune`, ``S_{t+1}`` holds the start periods that are alive at ``t + 1`` under the lifetime of [`expert_alive`](@ref), which the paper takes from Woodruff. Then ``S_t`` holds ``O(\\log t)`` copies, and for every ``s \\leq t`` it holds a start period in ``[s, (s + t)/2]``.

Suppose that the base rule's regret on every interval ``I`` is ``\\alpha^{-1} \\log |I|``, and that every loss is ``\\alpha``-exp-concave. When the copy started at ``r`` lives through ``I = [r, s]``, the paper bounds the regret of the rule on ``I`` by ``O(\\alpha^{-1} (\\ln r + \\ln |I|))`` (Lemma 3.1). Without `prune` every copy lives, so this bound holds on every interval. Under `prune` the paper's bound on every interval is ``O(\\alpha^{-1} \\log s \\cdot \\log |I| + 1)`` (Lemma 3.2).

# Algorithm

The steps of the Online Update:

 1. Set `t` to `st.n + 1`, and `r` to the return ``\\langle \\boldsymbol{h}, \\boldsymbol{x} \\rangle`` of the allocation `h` of each live copy.
 2. Set `phat` to `st.p .* r .^ alg.alpha`, divided by its sum.
 3. Step every live copy with its own Online Update, which gives its Rule State and its allocation for the next period.
 4. Start a new copy at the base rule's start from the uniform portfolio, projected onto `set` in the base rule's geometry, with the start period `t + 1`.
 5. Set `p` to `(1 - 1 / (t + 1)) .* phat`, followed by the share `1 / (t + 1)` of the new copy.
 6. Under `alg.prune`, keep the copies that `expert_alive` finds alive at `t + 1`, with their entries of `p`.
 7. Project `p` onto the Expert Set with the entropic projection. On the bare simplex over the copies this divides `p` by its sum.
 8. Set `q` to the sum of `p[k] .* h` over the copies. Return the new carrier, and the result of `blend_projection` of `q` onto `set` with `alg.proj`, from the Price-Adjusted Allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FollowTheLeadingHistory(;
        alg::AbstractOnlinePortfolioSelectionAlgorithm = NewtonStep(),
        alpha::Real = 1,
        prune::Bool = true,
        eset::Option{<:BoundedAllocationSet} = nothing,
        proj::EuclideanProjection = EuclideanProjection()
    ) -> FollowTheLeadingHistory

Keywords correspond to the struct's fields.

## Validation

  - `alpha > 0`. A `DomainError` is thrown otherwise.
  - `eset`, when given, holds a [`WeightBounds`](@ref) whose bounds are scalars and no `sets`: the working set changes size, so a vector bound has no length to match. An `ArgumentError` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the base rule is viewed and the rest is carried unchanged.

# Examples

```jldoctest
julia> FollowTheLeadingHistory()
FollowTheLeadingHistory
    alg ┼ NewtonStep
        │    beta ┼ Int64: 1
        │   delta ┼ Float64: 0.125
        │     eta ┼ Int64: 0
        │    proj ┴ EuclideanProjection()
  alpha ┼ Int64: 1
  prune ┼ Bool: true
   eset ┼ nothing
   proj ┴ EuclideanProjection()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`FollowTheLeadingHistoryState`](@ref)
  - [`ExpertMixture`](@ref): the mixture over a fixed set of experts.
  - [`NewtonStep`](@ref): the base rule of the paper's experiments.
  - [`expert_alive`](@ref): the lifetime that the pruning reads.

# References

  - $(ref_dict[:hazanseshadhri2009]) Algorithm 1, Lemmas 3.1 and 3.2, and Appendix A.
"""
struct FollowTheLeadingHistory{T1 <: AbstractOnlinePortfolioSelectionAlgorithm, T2 <: Real,
                               T3 <: Bool, T4 <: Option{<:BoundedAllocationSet},
                               T5 <: EuclideanProjection} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The base rule every expert is a copy of.
    """
    alg::T1
    """
    The exponent of the multiplicative update, the paper's ``\\alpha``. At `1` the update is the wealth weighting.
    """
    alpha::T2
    """
    Whether the rule prunes the working set to the paper's streaming set of logarithmic size.
    """
    prune::T3
    """
    The Expert Set onto which the rule projects the weight vector, a scalar bound over the live experts, or `nothing` for the bare simplex over them.
    """
    eset::T4
    """
    The geometry of the last projection of the blend onto the head's Allocation Set.
    """
    proj::T5
    function FollowTheLeadingHistory(alg::AbstractOnlinePortfolioSelectionAlgorithm,
                                     alpha::Real, prune::Bool,
                                     eset::Option{<:BoundedAllocationSet},
                                     proj::EuclideanProjection)
        @argcheck(alpha > zero(alpha), DomainError(alpha, "alpha must be positive"))
        if !isnothing(eset)
            @argcheck(isa(eset.wb, WeightBounds) &&
                      !isa(eset.wb.lb, AbstractVector) &&
                      !isa(eset.wb.ub, AbstractVector) &&
                      isnothing(eset.sets),
                      ArgumentError("the Expert Set of a FollowTheLeadingHistory is stated over a working set that changes size, so it holds a `WeightBounds` of scalar bounds and no `sets`: a vector bound has no length to match."))
        end
        return new{typeof(alg), typeof(alpha), typeof(prune), typeof(eset), typeof(proj)}(alg,
                                                                                          alpha,
                                                                                          prune,
                                                                                          eset,
                                                                                          proj)
    end
end
function FollowTheLeadingHistory(;
                                 alg::AbstractOnlinePortfolioSelectionAlgorithm = NewtonStep(),
                                 alpha::Real = 1, prune::Bool = true,
                                 eset::Option{<:BoundedAllocationSet} = nothing,
                                 proj::EuclideanProjection = EuclideanProjection())::FollowTheLeadingHistory
    return FollowTheLeadingHistory(alg, alpha, prune, eset, proj)
end
function port_opt_view(alg::FollowTheLeadingHistory, i, args...)
    return FollowTheLeadingHistory(; alg = port_opt_view(alg.alg, i, args...),
                                   alpha = alg.alpha, prune = alg.prune, eset = alg.eset,
                                   proj = alg.proj)
end
function rows_needed(alg::FollowTheLeadingHistory)
    return rows_needed(alg.alg)
end
function assert_rule_admits_set(alg::FollowTheLeadingHistory,
                                set::AbstractAllocationSet)::Nothing
    return assert_rule_admits_set(alg.alg, set)
end
function projection_geometry(alg::FollowTheLeadingHistory)
    return alg.proj
end
function rule_state_seed(alg::FollowTheLeadingHistory, w::AbstractVector,
                         set::Option{<:AbstractAllocationSet} = nothing)
    h = project_start(projection_geometry(alg.alg), set,
                      expert_start_allocation(alg.alg, w))
    return FollowTheLeadingHistoryState(0, [rule_state_seed(alg.alg, h, set)], [h],
                                        [one(eltype(w))], [1])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Checks whether the copy started at period `born` is alive at period `t`, under the lifetime of Hazan and Seshadhri (2009).

The function reads `t >= born` as given.

# Mathematical definition

```math
\\begin{align}
k &= r\\, 2^{j}\\,, \\quad r \\text{ odd}\\,, \\\\
k &\\in S_t \\iff k \\leq t \\leq k + 2^{j + 2} + 1\\,.
\\end{align}
```

Where:

  - ``k``: Start period of the copy, `born`.
  - ``r``, ``j``: Odd part of ``k`` and the exponent of the largest power of two that divides ``k``.
  - ``S_t``: Working set of period ``t``.
  - $(math_dict[:t_period])

The paper calls ``2^{j + 2} + 1`` the lifetime of ``k``, so the copy is alive in ``2^{j + 2} + 2`` periods.

# Arguments

  - `born`: The start period of the copy.
  - `t`: The period.

# Returns

  - `alive::Bool`: `t <= born + 2^(j + 2) + 1`, with `j` the number of trailing zeros of `born`.

# Related

  - [`FollowTheLeadingHistory`](@ref): the rule whose pruning reads the lifetime.
"""
function expert_alive(born::Integer, t::Integer)
    k = trailing_zeros(born)
    return t <= born + (1 << (k + 2)) + 1
end
function online_update!(alg::FollowTheLeadingHistory, st::FollowTheLeadingHistoryState,
                        w::AbstractVector, x::AbstractVector, rows,
                        set::AbstractAllocationSet)
    t = st.n + 1
    K = length(st.h)
    r = map(h -> LinearAlgebra.dot(h, x), st.h)
    q = st.p .* r .^ alg.alpha
    phat = q ./ sum(q)
    for k in 1:K
        st.st[k], st.h[k] = online_update!(alg.alg, st.st[k], st.h[k], x, rows, set)
    end
    # The newcomer starts where the base rule starts, on the set: a constant rebalanced
    # base answers its own allocation, so the projection follows the start, not the reverse.
    hn = expert_start_allocation(alg.alg, fill(one(eltype(w)) / length(w), length(w)))
    hn = project(projection_geometry(alg.alg), set, hn, hn)
    push!(st.st, rule_state_seed(alg.alg, hn, set))
    push!(st.h, hn)
    push!(st.born, t + 1)
    share = one(eltype(phat)) / (t + 1)
    p = vcat((one(share) - share) .* phat, share)
    if alg.prune
        keep = findall(b -> expert_alive(b, t + 1), st.born)
        keepat!(st.st, keep)
        keepat!(st.h, keep)
        keepat!(st.born, keep)
        p = p[keep]
    end
    p = project(EntropicProjection(), expert_allocation_set(alg.eset, length(p), eltype(w)),
                p, p)
    q = zeros(eltype(w), length(w))
    for (k, h) in enumerate(st.h)
        q .+= p[k] .* h
    end
    wn = blend_projection(alg.proj, set, q, price_adjusted_allocation(w, x))
    return FollowTheLeadingHistoryState(t, st.st, st.h, p, st.born), wn
end
export Prefix, LastRows, HistogramMatch, KernelMatch, NearestNeighbourMatch,
       CorrelationMatch, ClusterMatch, FollowTheLeader, ShortTermLossControlPortfolio,
       LowDimensionEnsemblePortfolio, FollowTheLeadingHistory
public AbstractSampleSelector, AbstractPatternMatchSelector, select_rows, matched_rows,
       AllocationSetConstraint
