"""
$(DocStringExtensions.TYPEDEF)

Names the past periods on which a follow-the-leader rule re-solves its held estimator.

A [`FollowTheLeader`](@ref) rule holds a Sample Selector on `sel`. The solved rules of the online selection literature solve one programme and differ in the sample alone. Follow the leader takes every period so far, and the successive variable rebalanced portfolio takes the last `W`. The pattern-matching rules take the periods whose preceding window resembles the latest window, by histogram cell, kernel radius, nearest neighbours, correlation or cluster. A selector reads the price relatives that the head holds and returns row indices. It holds no rows of its own.

# Interfaces

In order to implement a new selector, subtype `AbstractSampleSelector` with the parameters of the selector as part of the struct, and implement:

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

Under the default log-optimal estimator the leader is the best constant rebalanced portfolio to date. This is the successive constant rebalanced portfolio (SCRP) of [gaivoronski2000](@citet).

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

  - $(ref_dict[:gaivoronski2000]) Section 2, Definition 2, Equations 5 and 6.
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

Under the default log-optimal estimator the leader is the best constant rebalanced portfolio of a moving window. [gaivoronski2000](@citet) call the rule the successive variable rebalanced portfolio. Before `W` periods exist the sample is every period so far, as in [gaivoronski2000](@cite).

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

  - $(ref_dict[:gaivoronski2000]) Section 4, Definition 8, Equations 47 and 48.
"""
@concrete struct LastRows <: AbstractSampleSelector
    """
    The number of rows of the window.
    """
    W
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

In order to implement a new pattern-matching selector, subtype `AbstractPatternMatchSelector` with a `window` field and the parameters of its similarity test, and implement:

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

This is the partitioning portfolio of [gyorfischafer2003](@citet). [lihoi2014](@citet) state it as a similarity set and call the selector histogram-based, and they name its log-optimal rule BH. [gyorfischafer2003](@citet) quantise each price relative vector by a finite partition of the feature space, and a period matches when its window gives the same string of cell labels as the latest window. [gyorfischafer2003](@citet) leave the partition to the user. The consistency theorem of [gyorfischafer2003](@cite) asks for a nested sequence of partitions whose cells shrink. [gyorfischafer2003](@citet) mix the experts of that sequence by wealth. That mixture is an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each.

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

The bins of every entry form a partition of the positive orthant into products of intervals, which is one partition of the kind in [gyorfischafer2003](@cite). At the default `edges = [1]` the bin of an entry shows whether the asset rose or fell. So two windows match when every asset moved the same way in every period of the window. More edges give a finer partition and a smaller sample.

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

  - $(ref_dict[:gyorfischafer2003]) Section 16.4, Theorem 2.
  - $(ref_dict[:lihoi2014]) Sections 3.4.1 and 3.4.3.
"""
@concrete struct HistogramMatch <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window
    """
    The cell boundaries of every entry's binning, increasing.
    """
    edges
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

This is the kernel rule of [gyorfi2006](@citet) under the uniform kernel, in the Euclidean norm of the flattened windows. [lihoi2014](@citet) call its log-optimal rule BK. The experts of [gyorfi2006](@cite) run over a grid of windows ``w`` and radii ``c / \\ell``, and [gyorfi2006](@citet) mix them by wealth. That mixture is an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each, and each selector holds one radius.

Three later rules use this selector under another objective, which the held estimator on the rule's `opt` states. The semi-log-optimal rule of [gyorfi2007semilog](@citet) maximises the second-order expansion of the log at one. The Markowitz-type rule of [ottucsakvajda2007](@citet) maximises a mean-variance utility. The rule of [gyorfivajda2008](@citet) compares windows of one period and maximises the log return net of proportional costs.

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

Keywords correspond to the struct's fields. The radius has no default, because [gyorfi2006,gyorfi2007semilog,ottucsakvajda2007,gyorfivajda2008](@cite) scan it. Its scale is the scale of a window of price relatives.

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

  - $(ref_dict[:gyorfi2006]) Section 3, Equation 3.1 and Theorem 3.1.
  - $(ref_dict[:gyorfi2007semilog]) Section 3 and Equations 4.1 to 4.3.
  - $(ref_dict[:ottucsakvajda2007]) Sections 3.1 and 5, Equation 5.3.
  - $(ref_dict[:gyorfivajda2008]) Section 2, Algorithms 1 and 2, Equations 6 and 7.
  - $(ref_dict[:lihoi2014]) Sections 3.4.1 and 3.4.3.
"""
@concrete struct KernelMatch <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window
    """
    The radius of the uniform kernel, in the Euclidean norm of the concatenated windows.
    """
    radius
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

This is the nearest-neighbour rule of [gyorfi2008](@citet), in the Euclidean norm of the flattened windows. [lihoi2014](@citet) call its log-optimal rule BNN. The rule of [gyorfi2008](@cite) invests at period ``n = t + 1`` with ``\\lfloor p\\, n \\rfloor`` neighbours, a count that grows with the history. So `neighbours` is a [`CountOf`](@ref) or a [`FractionOf`](@ref). The fraction applies to the ``t - w`` candidates and not to the period, so the count never exceeds the candidates. It is below the count of [gyorfi2008](@cite) by at most ``\\lceil p (w + 1) \\rceil``. A fraction that rounds down to zero gives the empty sample. [gyorfi2008](@citet) assume that ties have zero probability. Here the earlier period wins a tie at the boundary.

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
  - ``k``: Count, the `val` of the `neighbours` field when it is a [`CountOf`](@ref).
  - ``p``: Fraction, the `val` of the `neighbours` field when it is a [`FractionOf`](@ref).
  - $(math_dict[:t_period])

The distance between two windows is the Euclidean norm of their difference.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NearestNeighbourMatch(; window::Integer = 5,
                          neighbours::Real_CountFraction = FractionOf(0.1)) -> NearestNeighbourMatch

Keywords correspond to the struct's fields. A plain `neighbours` is read through [`count_or_fraction`](@ref).

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `neighbours` passes [`count_or_fraction`](@ref).

$(val_dict[:count_fraction_grid])

# Examples

```jldoctest
julia> NearestNeighbourMatch()
NearestNeighbourMatch
      window ┼ Int64: 5
  neighbours ┼ FractionOf
             │   val ┴ Float64: 0.1
```

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`KernelMatch`](@ref): the rule that keeps every window within a radius in place of a count.

# References

  - $(ref_dict[:gyorfi2008]) Section 3, Equations 3.1 to 3.3 and Theorem 3.1.
  - $(ref_dict[:lihoi2014]) Sections 3.4.1 and 3.4.3.
"""
@concrete struct NearestNeighbourMatch <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window
    """
    The number of neighbours to keep, a [`CountOf`](@ref) or a [`FractionOf`](@ref) of the candidate rows.
    """
    neighbours
    function NearestNeighbourMatch(window::Integer, neighbours::Real_CountFraction)
        assert_pattern_window(window)
        neighbours = count_or_fraction(neighbours)
        return new{typeof(window), typeof(neighbours)}(window, neighbours)
    end
end
function NearestNeighbourMatch(; window::Integer = 5,
                               neighbours::Real_CountFraction = FractionOf(0.1))::NearestNeighbourMatch
    return NearestNeighbourMatch(window, neighbours)
end
"""
    neighbour_count(neighbours::CountOf, n::Integer)
    neighbour_count(neighbours::FractionOf, n::Integer)

Returns the number of neighbours that a nearest-neighbour selector keeps out of `n` candidates.

# Arguments

  - `neighbours`: The [`CountOf`](@ref) or the [`FractionOf`](@ref) of the candidates.
  - `n`: The number of candidates.

# Returns

  - `l::Integer`: `min(neighbours.val, n)` for a count, and `floor(Int, neighbours.val * n)` for a fraction.

# Related

  - [`NearestNeighbourMatch`](@ref)
"""
function neighbour_count(neighbours::CountOf, n::Integer)
    return min(neighbours.val, n)
end
function neighbour_count(neighbours::FractionOf, n::Integer)
    return floor(Int, neighbours.val * n)
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

This is the correlation-driven rule of [li2011corn](@citet). The correlation is the Pearson correlation of the two flattened windows.

The CORN-U of [li2011corn](@cite) mixes the experts of the windows 1 to ``W`` at one threshold, from a uniform start and by wealth. The CORN-K of [li2011corn](@cite) runs over the windows 1 to ``W`` and the thresholds ``0, 1/P, \\ldots, (P - 1)/P``, and it keeps the ``K`` wealthiest experts, weighted by wealth. Both are an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each, CORN-K under the [`TopK`](@ref) weighting. [li2011corn](@citet) set ``W = 5``, ``P = 10`` and ``K = 5``, and run CORN-U at the threshold 0.1 with no tuning. The defaults of one selector take the window 5 and the threshold 0.1 from these values.

[wang2018racorn](@citet) add a risk penalty to each expert (RACORN-K). Their programme maximises the mean log return over the sample less ``\\lambda`` times the standard deviation of the log return ``\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle``. Their experts run over windows, thresholds and values of ``\\lambda``, and the wealthiest tenth of them combine as in CORN-K. The nearest programme of the library is [`MeanRisk`](@ref) under [`MaximumUtility`](@ref) at `l = λ` over [`StandardDeviation`](@ref), with [`LogarithmicReturn`](@ref). It penalises the standard deviation of the return ``\\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle - 1`` in place of the log return, and the two penalties agree to the first order in the return.

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

A window with no variation correlates at zero, as [li2011corn](@cite) sets it. So it matches at a threshold of zero or less, and at no positive threshold.

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
  - $(ref_dict[:wang2018racorn]) Section 3.2, Equation 3 and Section 4.2.
"""
@concrete struct CorrelationMatch <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window
    """
    The correlation threshold a candidate window must reach.
    """
    rho
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

This follows the cluster-based rule of [khedmatiazin2020](@citet). [khedmatiazin2020](@citet) cluster the windows by k-means (KMNLOG), k-medoids (KMDLOG), spectral clustering (SPCLOG) and hierarchical clustering (HRCLOG). The `clusterer` field takes one of the library's clustering estimators, which partitions the windows through a distance that it forms from them. [`KMeansAlgorithm`](@ref) and [`HClustAlgorithm`](@ref) are among its algorithms. The programme of [khedmatiazin2020](@cite) holds a transaction cost, and the `fees` slot of the rule's `opt` holds that cost. Fewer than two candidates cannot be partitioned, so they give the empty sample.

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
  - `returns_source(clusterer) === nothing`: the clustering reads the windows that the selector lays out, in the frame of the head's step, so it states no source of its own, see [`assert_inner_reader`](@ref). An `ArgumentError` is thrown otherwise.

# Examples

```jldoctest
julia> ClusterMatch(; clusterer = ClustersEstimator()).window
5
```

# Related

  - [`AbstractPatternMatchSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`ClustersEstimator`](@ref): the default clusterer.
  - [`KMeansAlgorithm`](@ref): the default algorithm of the clusterer, KMNLOG in [khedmatiazin2020](@cite).
  - [`HClustAlgorithm`](@ref): HRCLOG in [khedmatiazin2020](@cite).

# References

  - $(ref_dict[:khedmatiazin2020]) Page 113546. $(ref_dict[:no_original_source])
"""
@concrete struct ClusterMatch <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window
    """
    The clustering estimator that partitions the windows.
    """
    clusterer
    function ClusterMatch(window::Integer, clusterer::AbstractClustersEstimator)
        assert_pattern_window(window)
        assert_inner_reader(clusterer, :clusterer, :ClusterMatch)
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
    const LeaderOptimiser = Union{BestConstantRebalancedPortfolio, JuMPOptimisationEstimator}

Groups the optimisation estimators that a follow-the-leader rule can re-solve under the head's Allocation Set.

A leader of a [`FollowTheLeader`](@ref) rule must honour the set. The solver-free [`BestConstantRebalancedPortfolio`](@ref) takes a bounded set on its own `wb` and `sets` slots. A JuMP estimator takes any set as a second block of its entry, the keyword `blocks` of [`resolve_jump_optimiser`](@ref). No other estimator has a slot or an entry for the set, so the alias excludes it. Each member of the fallback chain of the held estimator is a member of the alias too, see [`assert_leader_chain`](@ref).

# Related

  - [`FollowTheLeader`](@ref)
  - [`leader_allocation`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
const LeaderOptimiser = Union{BestConstantRebalancedPortfolio, JuMPOptimisationEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Re-solves an optimisation estimator on a sample of past periods at every step, the follow-the-leader rule.

The Sample Selector on `sel` names the sample, and the estimator on `opt` solves on it. The solution is the leader. The rule plays the leader, or at `gamma > 0` the leader damped towards the allocation that it held.

Every solved rule of the online selection literature is this rule under one selector and one objective. [`Prefix`](@ref) under the log-optimal objective is follow the leader, the successive constant rebalanced portfolio of [gaivoronski2000](@citet). At `gamma > 0` it is the weighted rule (WSCRP) of [gaivoronski2000](@cite). [`LastRows`](@ref) gives the successive variable rebalanced portfolio of [gaivoronski2000](@cite). The pattern-matching selectors give the histogram, kernel, nearest-neighbour, correlation and cluster rules of [gyorfischafer2003,gyorfi2006,gyorfi2008,li2011corn,khedmatiazin2020](@cite). [gyorfischafer2003,gyorfi2006,gyorfi2008,li2011corn,khedmatiazin2020](@citet) mix a grid of selectors by wealth, which is an [`ExpertMixture`](@ref) over rules of this kind.

The objective is the held estimator's. [`BestConstantRebalancedPortfolio`](@ref), the default, is the log-optimal portfolio with no solver. [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref) solves the same programme on a solver. Under [`Prefix`](@ref) and on the simplex, that estimator with `l2 = L2Regularisation(; val = 1 / (2t), alg = QuadRiskExpr())` on its [`JuMPOptimiser`](@ref) and ``t`` rows is the Exp-Concave-FTL of [hazankale2012](@citet). [hazankale2012](@citet) subtract ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} \\rVert^2`` from the sum of the log returns, and [`LogarithmicReturn`](@ref) states their mean, so the weight is ``1 / (2t)``. One fixed `val` is that leader at one sample size alone. [`MaximumUtility`](@ref) over a risk measure gives a mean-risk leader, and `fees` on the estimator gives a cost-aware leader.

The solver-free default runs Cover's fixed point, which stops on a certificate, a bound on the shortfall of its log wealth from the leader's. At a leader that drops an asset, the weight of that asset decays slowly. So the default budget can stop before the certificate meets `tol`, and the rule then warns once and names the certificate. On the first 30 rows of the returns `0.02 .* randn(StableRNG(11), 40, 4)` the default stops 0.066 from the leader in weight and 6.2e-5 short of it in log wealth. At `iters = 10_000_000` the certificate meets `tol = 1e-12` after 1 437 030 steps, 1.6e-9 from the leader in weight. A JuMP head solves the leader to the solver's tolerance, and the two forms agree to that tolerance at a leader inside the simplex.

**The re-solve takes the head's Allocation Set as its feasible region.** The rule passes the set as a second block of the entry of the held JuMP estimator, the keyword `blocks` of [`resolve_jump_optimiser`](@ref), with the rows that the head holds, see [`leader_block`](@ref). The block of the set reads the prior of the set on those rows, and the block of the held estimator reads its own prior on the sample. The rule passes the Price-Adjusted Allocation through [`factory`](@ref) to the held estimator and to the set, as a fold loop does, so a turnover term or a fee on `opt` reads the same book as the set. It plays the optimum with no projection, because the projected leader is not the leader. A constraint of the set and one of the held estimator meet in one model as two constraints, each in its own namespace. So the leader is the optimum over the set and the feasible set of the held estimator together. The constraints of the held estimator restrict the leader, but they do not bind the allocation that the rule plays. At `gamma > 0` the rule projects the damped mix onto the set alone, so the mix can leave the feasible set of the held estimator. A held estimator with no constraint of its own gives the leader over the set alone, which is the rule of the literature. The default `wb` of a [`JuMPOptimiser`](@ref) is `WeightBounds()`, the long-only bounds. So on a set that admits a negative weight, state `wb = nothing` on the held estimator, or the leader stays long-only. The budget is the exception, because the head owns it. The re-solve sets the net budget of the held estimator, and of each member of its fallback chain, to the budget of one, so the leader's model holds the row `Σw = k`, see [`leader_chain`](@ref). So the held estimator takes `bgt = nothing`, a budget of one, or a [`BudgetRange`](@ref) that holds one, and a gross budget that the budget of one can reach. The constructor refuses every other budget and every [`BudgetCostEstimator`](@ref), see [`assert_leader_budget`](@ref). The fallback chain passes the block to each member. So the constructor refuses a member that cannot take the set, see [`assert_leader_chain`](@ref). The solver-free estimator has no model. So a [`BoundedAllocationSet`](@ref) goes into its `wb` and `sets`, and the rule plays its repaired fixed point. The head refuses a [`ProgrammeAllocationSet`](@ref) over the solver-free estimator at construction. Under a programme set the constrained leader is [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref).

**The rule projects the damped mix alone.** It uses the Euclidean geometry on `proj`, with the Price-Adjusted Allocation as the reference. On a static convex set the projection returns the mix unchanged. A turnover ceiling or a MIP kind needs the repair on a day when the mix leaves the set. The rule skips the projection by type on a [`BoundedAllocationSet`](@ref), and at `gamma = 0`. An empty sample, or a sample too small for the estimator, gives the uniform portfolio projected onto the set, as in [gyorfischafer2003,gyorfi2006,gyorfi2008,li2011corn](@cite). A JuMP head fits a covariance, so it needs two rows. A re-solve that fails after the fallback chain of the held estimator is a Held Step. The rule then plays the Price-Adjusted Allocation, so the fund trades nothing that period.

**Under a time-varying panel the re-solve is the batch path.** The estimator reads the head's rows viewed at the sample, with the returns as they are, `NaN` where no return exists, and the active mask. So the estimator reduces to its own universe, as it does on any `ReturnsResult`, and writes a zero at every asset outside it. That universe is the Coverage Universe of the sample for the solver-free leader and for a head with no prior, and the Investable Mask of its prior for a JuMP head. An asset that is not listed for a part of the sample stays outside the universe of a plain estimator until the sample clears that span. A [`Prefix`](@ref) sample never clears it, and a [`LastRows`](@ref) sample clears it when its window has passed the listing. A prior that reads the mask admits the asset after its own warm-up. The selector reads a gap as a price relative of one, as if the asset sat in cash, so it compares a window that spans a listing and does not drop it. The damped mix and the projection read a zero from the leader at such an asset as they read any other zero.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^\\star_t &= \\underset{\\boldsymbol{w} \\in \\mathcal{W} \\cap \\mathcal{W}_{\\mathrm{opt}}}{\\arg\\max}\\; \\sum_{i \\in C_t} \\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\mathrm{Proj}_{\\mathcal{W}}\\left( (1 - \\gamma)\\, \\boldsymbol{w}^\\star_t + \\gamma\\, \\boldsymbol{w}_t \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_star_lead]) The first line is the leader under the default log-optimal estimator. Another estimator on `opt` replaces the objective.
  - $(math_dict[:w_var_lead])
  - $(math_dict[:W_aset])
  - ``\\mathcal{W}_{\\mathrm{opt}}``: Feasible set of the constraints of the held estimator on `opt`. With no constraint of its own, it holds every allocation, and the leader is the optimum over ``\\mathcal{W}`` alone.
  - $(math_dict[:C_t_sample])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - ``\\gamma``: Damping towards the held allocation, in ``[0, 1]``, the `gamma` field.
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:t_period])

At ``\\gamma = 0`` the rule plays the leader. The leader lies in ``\\mathcal{W}``, so the mix leaves ``\\mathcal{W}`` only when ``\\boldsymbol{w}_t`` is outside it, as under a turnover ceiling, or when ``\\mathcal{W}`` is not convex, as under a cardinality bound. The projection reads ``\\mathcal{W}`` alone, so at ``\\gamma > 0`` the played allocation can leave ``\\mathcal{W}_{\\mathrm{opt}}``. Under the log-optimal estimator and [`Prefix`](@ref), the leader on ``\\mathcal{W} = \\Delta_N`` is the best constant rebalanced portfolio of the first ``t`` periods. The Exp-Concave-FTL of [hazankale2012](@citet) is

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
 2. Select `idx`, the rows of the sample, with `select_rows` on `alg.sel` and the price relatives of `rows` in the head's frame, which [`head_frame_returns`](@ref) gives. The leader in step 3 reads the raw rows, and its prior states its own source.
 3. When `idx` holds fewer rows than `leader_min_rows(alg.opt)`, set `wstar` to the uniform portfolio projected onto `set` with `alg.proj`, from `wh`. Otherwise set `wstar` to the result of `leader_allocation` on `alg.opt`, the view of `rows` at `idx`, `wh`, `set` and `rows`.
 4. When `wstar` is `nothing`, the step is a Held Step. Return `st` and a copy of `wh`. `st` is the state of the rule: what the rule keeps from one update to the next, or `nothing` for a rule that keeps nothing. This rule returns it unchanged.
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
  - The budgets of a JuMP estimator on `opt` admit the budget of one of the head, see [`assert_leader_budget`](@ref):
      + `bgt` is `nothing`, one, or a [`BudgetRange`](@ref) that holds one. A `DomainError` is thrown otherwise.
      + `bgt` is not a [`BudgetCostEstimator`](@ref). An `ArgumentError` is thrown otherwise.
      + `sbgt`, `gbgt` and `wb` pass [`assert_gross_budget_admissible`](@ref) with a net budget of one. So `gbgt` reaches one, and it is not given beside a number in `sbgt`. An `ArgumentError` is thrown otherwise.
      + The same holds for each entry and the explicit `default` of a [`TimeDependent`](@ref) schedule in these fields.
  - Each member of the fallback chain of `opt` can take the head's Allocation Set, see [`assert_leader_chain`](@ref):
      + Under a JuMP estimator, each member is a JuMP estimator, and its budgets meet the rules above. An `ArgumentError` is thrown otherwise.
      + Under the solver-free [`BestConstantRebalancedPortfolio`](@ref), each member is a `BestConstantRebalancedPortfolio`. An `ArgumentError` is thrown otherwise.
      + No member is a precomputed result. An `ArgumentError` is thrown otherwise.

## View parameters

[`port_opt_view`](@ref) takes the view of `opt` at the selected indices and carries `sel`, `gamma` and `proj` unchanged.

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
  - [`leader_allocation`](@ref): the re-solve that gives the head's set to the held estimator.
  - [`BestConstantRebalancedPortfolio`](@ref): the default estimator.
  - [`ExpertMixture`](@ref): the mixture by wealth over a grid of these rules.
  - [`FollowTheLeadingHistory`](@ref)
  - [`ShortTermLossControlPortfolio`](@ref): this rule over [`LastRows`](@ref) under a loss-control programme.
  - [`LowDimensionEnsemblePortfolio`](@ref): this rule over [`LastRows`](@ref) under an ensemble forecast.

# References

  - $(ref_dict[:gaivoronski2000]) Section 2, Definitions 2 and 4, and Section 4, Definition 8.
  - $(ref_dict[:hazankale2012]) Section 2.2, Equation 1.
  - $(ref_dict[:gyorfischafer2003]) Section 16.4.
  - $(ref_dict[:gyorfi2006]) Section 3.
  - $(ref_dict[:gyorfi2008]) Section 3.
  - $(ref_dict[:li2011corn]) Section 4.3.
  - $(ref_dict[:khedmatiazin2020]) Page 113546. $(ref_dict[:no_original_source])
"""
@concrete struct FollowTheLeader <: AbstractOnlinePortfolioSelectionAlgorithm
    """
    The Sample Selector that names the rows the estimator re-solves on.
    """
    sel
    """
    The optimisation estimator re-solved on the selected rows.
    """
    opt
    """
    The damping towards the held allocation, in `[0, 1]`. At `0` the rule plays the leader.
    """
    gamma
    """
    $(field_dict[:proj])
    """
    proj
    function FollowTheLeader(sel::T1, opt::T2, gamma::T3,
                             proj::T4) where {T1 <: AbstractSampleSelector,
                                              T2 <: LeaderOptimiser, T3 <: Real,
                                              T4 <: EuclideanProjection}
        @argcheck(zero(gamma) <= gamma <= one(gamma),
                  DomainError(gamma, "gamma must lie in [0, 1]"))
        assert_leader_budget(opt)
        assert_leader_chain(opt, opt.fb)
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
    assert_leader_budget(opt::BestConstantRebalancedPortfolio)
    assert_leader_budget(opt::JuMPOptimisationEstimator)

Assert that the budgets of the held estimator of a [`FollowTheLeader`](@ref) rule admit the budget of one that the online head owns.

The head owns the budget: `Σw = 1` holds on every Allocation Set, and cash is an asset with price relative one. The re-solve keeps that budget as the row `Σw = k` in the leader's model. It sets the net budget of the held optimiser, and of each member of its fallback chain, to one, see [`leader_chain`](@ref). A budget that excludes one would then lose its effect with no message, and before the re-solve took the set as a block it left an empty feasible set, so that every step was a Held Step. The configuration alone decides it, so the constructor refuses it.

The solver-free estimator states no budget, because its fixed point lies on the simplex. A JuMP estimator is checked through the budgets of its [`JuMPOptimiser`](@ref) with [`leader_budget_admits_one`](@ref). Each entry and the explicit `default` of a [`TimeDependent`](@ref) schedule in `bgt`, `sbgt`, `gbgt` or `wb` are checked with the other fields at their static values, through [`assert_time_dependent_substitution`](@ref). A function schedule has a value only at a fold, so the check reads it only through its `default`.

# Algorithm

The JuMP method:

 1. Collect `args`, the `bgt`, `sbgt`, `gbgt` and `wb` of `jo`, the [`JuMPOptimiser`](@ref) of `opt`.
 2. Check the values in `args` with [`leader_budget_admits_one`](@ref). A schedule passes this step.
 3. Check each entry and the explicit `default` of each schedule in `args` with [`leader_budget_admits_one`](@ref), through [`assert_time_dependent_substitution`](@ref). The other fields take their static values, and `jump_optimiser_td_defaults` gives the static defaults.

# Arguments

  - `opt`: The held estimator.

# Validation

  - The rules of [`leader_budget_admits_one`](@ref) for the budgets of a JuMP estimator.

# Returns

  - `nothing`.

# Related

  - [`FollowTheLeader`](@ref)
  - [`leader_budget_admits_one`](@ref)
  - [`assert_net_budget_admits_one`](@ref)
"""
function assert_leader_budget(::BestConstantRebalancedPortfolio)::Nothing
    return nothing
end
function assert_leader_budget(opt::JuMPOptimisationEstimator)::Nothing
    jo = opt.opt
    args = (; bgt = jo.bgt, sbgt = jo.sbgt, gbgt = jo.gbgt, wb = jo.wb)
    leader_budget_admits_one(; args...)
    assert_time_dependent_substitution(leader_budget_admits_one, args,
                                       jump_optimiser_td_defaults())
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that the static budgets of a held optimiser admit the budget of one of the online head.

The net budget goes through [`assert_net_budget_admits_one`](@ref). The short and the gross budgets go through [`assert_gross_budget_admissible`](@ref) with the net budget of one in the place of `bgt`, as a [`ProgrammeAllocationSet`](@ref) checks its own. The head fixes the net exposure at one. So a number in `sbgt` and any `gbgt` over-determine the gross exposure, and a gross budget below one leaves no feasible weights.

# Arguments

  - `bgt`: The net budget of the held optimiser.
  - `sbgt`: The short budget of the held optimiser.
  - `gbgt`: The gross budget of the held optimiser.
  - `wb`: The weight bounds of the held optimiser.

# Validation

  - The rules of [`assert_net_budget_admits_one`](@ref) for `bgt`.
  - The rules of [`assert_gross_budget_admissible`](@ref) for `1`, `sbgt`, `gbgt` and `wb`.

# Related

  - [`assert_leader_budget`](@ref)
"""
function leader_budget_admits_one(; bgt, sbgt, gbgt, wb)::Nothing
    assert_net_budget_admits_one(bgt)
    assert_gross_budget_admissible(1, sbgt, gbgt, wb)
    return nothing
end
"""
    assert_net_budget_admits_one(bgt::Number)
    assert_net_budget_admits_one(bgt::BudgetRange)
    assert_net_budget_admits_one(bgt::BudgetCostEstimator)
    assert_net_budget_admits_one(bgt)

Assert that the net budget `bgt` of the held optimiser of a [`FollowTheLeader`](@ref) rule admits the budget of one.

The re-solve sets `bgt` to one, so the leader's model holds the head's row `Σw = k` in the place of the rows of `bgt`, see [`leader_chain`](@ref). A number writes `Σw = k · bgt`, so the replacement keeps its feasible set only when it is one. A [`BudgetRange`](@ref) writes `k · lb ≤ Σw ≤ k · ub`, so the replacement narrows it to the budget of one only when its interval holds one. `nothing` writes no row, and the head's row fixes the budget.

A [`BudgetCostEstimator`](@ref) is refused in every form. It charges its cost against the budget, `Σw + c(w) ∈ k · [lb, ub]`, with `c(w) ≥ 0` the cost of the move from its reference `w`. The head's row `Σw = k` turns this into `c(w) ∈ k · [lb − 1, ub − 1]`, a band on the cost that the wealth recursion of the head never pays, and the replacement of `bgt` with one would drop the cost with no message. [`factory`](@ref factory(bgt::BudgetCosts, w::VecNum)) sets the reference to the Price-Adjusted Allocation of the step, or keeps the caller's vector when `fixed` is `true`. A budget of one pins the leader to that reference, so the leader never trades, a budget below one leaves no feasible weights, and a budget above one bounds the cost of the move from the reference. None of these is a cost charged against the budget. A cost that the series pays is the `fees` of the head. A cost that the leader weighs is the `fees` of the held [`JuMPOptimiser`](@ref), which `factory` measures from the Price-Adjusted Allocation of the step, and a ceiling on the trade is a [`Turnover`](@ref) on `tn`.

`nothing`, a [`TimeDependent`](@ref) schedule and any other [`BudgetEstimator`](@ref) pass. [`assert_leader_budget`](@ref) checks the entries of a schedule.

# Arguments

  - `bgt`: The net budget of the held optimiser.

# Validation

  - A number is one. A `DomainError` is thrown otherwise.
  - The interval of a `BudgetRange` holds one. A `DomainError` is thrown otherwise.
  - `bgt` is not a `BudgetCostEstimator`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`assert_leader_budget`](@ref)
  - [`set_budget_constraints!`](@ref)
  - [`leader_chain`](@ref)
"""
function assert_net_budget_admits_one(bgt::Number)::Nothing
    @argcheck(isone(bgt),
              DomainError(bgt,
                          "the held optimiser of a `FollowTheLeader` has the net budget `bgt = $bgt`. The online head owns the budget of one: the re-solve sets the net budget of the held optimiser to one, so the leader's model holds the row `sum(w) == k`, and `bgt = $bgt` would have no effect. Set `bgt = nothing` or `bgt = 1` on the held `JuMPOptimiser`. A budget below one is a cash column with price relative one, and a leverage cap is a `gbgt` of at least one."))
    return nothing
end
function assert_net_budget_admits_one(bgt::BudgetRange)::Nothing
    lb = bgt.lb
    ub = bgt.ub
    @argcheck((isnothing(lb) || lb <= one(lb)) && (isnothing(ub) || one(ub) <= ub),
              DomainError((lb, ub),
                          "the held optimiser of a `FollowTheLeader` has the net budget range `lb = $lb`, `ub = $ub`, which excludes one. The online head owns the budget of one: the re-solve sets the net budget of the held optimiser to one, so the leader's model holds the row `sum(w) == k`, which the range excludes. Use a range that holds one, `bgt = nothing` or `bgt = 1` on the held `JuMPOptimiser`."))
    return nothing
end
function assert_net_budget_admits_one(::BudgetCostEstimator)::Nothing
    return throw(ArgumentError("the held optimiser of a `FollowTheLeader` has a `BudgetCostEstimator` in `bgt`. It charges its cost against the budget, `sum(w) + c(w)` in `k * [lb, ub]`, but the online head owns the budget of one: the re-solve sets the net budget of the held optimiser to one, so the leader's model holds the row `sum(w) == k`, and the cost would have no effect. With that row, the cost would leave `c(w)` in `k * [lb - 1, ub - 1]`, a band on the cost of the move from the estimator's reference `w`, which the head's wealth recursion never pays. The reference is the step's Price-Adjusted Allocation, or the caller's vector when `fixed` is `true`. A budget of one pins the leader to that reference, and a budget below one leaves no feasible weights. A cost that the series pays is the `fees` of the head. A cost that the leader weighs is the `fees` of the held `JuMPOptimiser`, measured from the step's Price-Adjusted Allocation, and a ceiling on the trade is a `Turnover` on `tn`."))
end
function assert_net_budget_admits_one(::Any)::Nothing
    return nothing
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
    assert_leader_chain(opt::LeaderOptimiser, fb::Nothing)
    assert_leader_chain(opt::JuMPOptimisationEstimator, fb::JuMPOptimisationEstimator)
    assert_leader_chain(opt::BestConstantRebalancedPortfolio, fb::BestConstantRebalancedPortfolio)
    assert_leader_chain(opt::LeaderOptimiser, fb)

Assert that each member of the fallback chain of the held estimator of a [`FollowTheLeader`](@ref) rule can take the head's Allocation Set.

[`optimise`](@ref) runs the fallback `fb` of a member that fails, and it passes its keyword arguments to each member. So the second block that [`leader_allocation`](@ref) passes to a JuMP estimator reaches each member of its chain, and the leader of each member honours the set. A member of another kind does not read the block, and a precomputed result has nothing to solve, so either one answers outside the set. The solver-free estimator takes a bounded set through its own bounds and passes no block. So each member of its chain is solver-free too, and [`leader_chain`](@ref) gives each one the bounds of the set. The rule is that each member has the kind of the held estimator.

# Algorithm

 1. When `fb` is `nothing`, return.
 2. When `fb` has the kind of `opt`, check the budgets of a JuMP member with [`assert_leader_budget`](@ref), and check the chain of `fb`.
 3. Otherwise, throw.

# Arguments

  - `opt`: The held estimator, or the member of the chain before `fb`.
  - `fb`: The fallback of `opt`.

# Validation

  - `fb` is `nothing` or has the kind of `opt`. An `ArgumentError` is thrown otherwise.
  - The rules of [`assert_leader_budget`](@ref) for a JuMP member.

# Returns

  - `nothing`.

# Related

  - [`FollowTheLeader`](@ref)
  - [`LeaderOptimiser`](@ref)
  - [`leader_chain`](@ref)
"""
function assert_leader_chain(::LeaderOptimiser, ::Nothing)::Nothing
    return nothing
end
function assert_leader_chain(::JuMPOptimisationEstimator,
                             fb::JuMPOptimisationEstimator)::Nothing
    assert_leader_budget(fb)
    return assert_leader_chain(fb, fb.fb)
end
function assert_leader_chain(::BestConstantRebalancedPortfolio,
                             fb::BestConstantRebalancedPortfolio)::Nothing
    return assert_leader_chain(fb, fb.fb)
end
function assert_leader_chain(opt::LeaderOptimiser, fb)::Nothing
    return throw(ArgumentError("the fallback chain of the held estimator of a `FollowTheLeader` holds a `$(nameof(typeof(fb)))` after a `$(nameof(typeof(opt)))`. The re-solve gives the head's Allocation Set to a JuMP estimator as a block of its entry, and the chain passes that block to each member. The solver-free `BestConstantRebalancedPortfolio` takes a bounded set through its own bounds. A member of another kind, or a precomputed result, would answer outside the set. Give each member of the chain the kind of the held estimator: a JuMP estimator under a JuMP estimator, and a `BestConstantRebalancedPortfolio` under a `BestConstantRebalancedPortfolio`."))
end
"""
    leader_chain(opt::Nothing, set::AbstractAllocationSet, k::Number)
    leader_chain(opt::JuMPOptimisationEstimator, set::AbstractAllocationSet, k::Number)
    leader_chain(opt::BestConstantRebalancedPortfolio, set::BoundedAllocationSet, k::Number)

Gives the held estimator of a [`FollowTheLeader`](@ref) rule, and each member of its fallback chain, the parts of the head's Allocation Set that do not reach it as a block.

The online head owns the budget of one, and the block of the set carries no budget row. So the function sets the net budget `bgt` of each JuMP member to `k`, and the leader's model holds the row `Σw = k`. [`assert_leader_budget`](@ref) admits only a budget that holds one, so the replacement changes no feasible set that the constructor admits. Without the replacement, `bgt = nothing` would solve away from the budget of one, and the wealth recursion would read weights that do not sum to one. The solver-free estimator has no model and takes no block. So each member of its chain takes the bounds `wb` and the `sets` of the bounded set.

# Arguments

  - `opt`: The held estimator after `factory(opt, w)`, a member of its chain, or `nothing` at the end of the chain.
  - `set`: The head's Allocation Set, resolved once per fold.
  - `k`: The budget of one, `one(eltype(w))` for the Price-Adjusted Allocation `w` of the step.

# Returns

  - `opt'`: `opt` and its chain, with the budget or the bounds of the set, or `nothing` at the end of the chain.

# Related

  - [`leader_allocation`](@ref)
  - [`assert_leader_chain`](@ref)
  - [`assert_leader_budget`](@ref)
"""
function leader_chain(::Nothing, ::AbstractAllocationSet, ::Number)::Nothing
    return nothing
end
function leader_chain(opt::JuMPOptimisationEstimator, set::AbstractAllocationSet, k::Number)
    return rebuild_estimator(opt,
                             (; opt = rebuild_estimator(opt.opt, (; bgt = k)),
                              fb = leader_chain(opt.fb, set, k)))
end
function leader_chain(opt::BestConstantRebalancedPortfolio, set::BoundedAllocationSet,
                      k::Number)
    return rebuild_estimator(opt,
                             (; wb = set.wb, sets = set.sets,
                              fb = leader_chain(opt.fb, set, k)))
end
"""
    leader_block(set::ProgrammeAllocationSet, opt::JuMPOptimisationEstimator, w::AbstractVector, X::ReturnsResult)
    leader_block(set::BoundedAllocationSet, opt::JuMPOptimisationEstimator, w::AbstractVector, X::ReturnsResult)

Gives the entry of the head's Allocation Set in the keyword `blocks` of the entry of a held JuMP estimator.

The entry is `(set, X)`, as [`fit_allocation_set_block`](@ref) states: the set after stage 1 and after `factory(set, w)`, and the rows that the head holds. So the block of the set reads its own prior on the rows of the head, and its turnover ceilings and its tracking errors read the Price-Adjusted Allocation of the step. The entry is an argument of one solve, so no estimator holds a Result. A [`BoundedAllocationSet`](@ref) holds no block. Its feasible set is the budget of one and its bounds, so the function puts its bounds `wb` and its `sets` into a [`ProgrammeAllocationSet`](@ref) whose block holds those two slots alone. That set takes the solver of the held estimator. Its block reads no rows and resolves no Calibration Rule, so no reader reads the solver, and the set fits no prior.

# Arguments

  - `set`: The head's Allocation Set, resolved once per fold.
  - `opt`: The held estimator.
  - `w`: The Price-Adjusted Allocation of the step.
  - `X`: Every row that the head holds, as a [`ReturnsResult`](@ref).

# Returns

  - `(set', X)::Tuple`: The entry.

# Related

  - [`leader_allocation`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`fit_allocation_set_block`](@ref)
"""
function leader_block(set::ProgrammeAllocationSet, ::JuMPOptimisationEstimator,
                      w::AbstractVector, X::ReturnsResult)::Tuple
    return factory(set, w), X
end
function leader_block(set::BoundedAllocationSet, opt::JuMPOptimisationEstimator,
                      ::AbstractVector, X::ReturnsResult)::Tuple
    return ProgrammeAllocationSet(; slv = opt.opt.slv, wb = set.wb, sets = set.sets), X
end
"""
    leader_allocation(opt::BestConstantRebalancedPortfolio, rd::ReturnsResult, w::AbstractVector, set::BoundedAllocationSet, X)
    leader_allocation(opt::JuMPOptimisationEstimator, rd::ReturnsResult, w::AbstractVector, set::AbstractAllocationSet, X::ReturnsResult)

Re-solves the held estimator on the sample under the head's Allocation Set, and returns the leader, or `nothing` on a Held Step.

The estimator runs the batch path that it runs on any `ReturnsResult`. An estimator with no prior reduces to the Coverage Universe of the sample, an estimator with a prior reduces to the Investable Mask of its prior, and both return a zero at every asset outside it. A JuMP estimator takes the set as a second block of its entry, so the one refusal branch of the entry covers the readers of the set, and an asset that the prior of the set cannot price departs with a warning, as in a batch fit with two blocks. A leader that stopped short of its certificate still trades, so the solver-free method warns and plays the fixed point.

# Algorithm

The solver-free method:

 1. Give `opt` and each member of its chain the bounds and the `sets` of `set` with [`leader_chain`](@ref).
 2. Optimise the result of `factory` with `w` on `rd`, giving `res`.
 3. When the fixed point stopped at `iters` before its certificate met `tol`, warn once and name the certificate.
 4. Return `res.w`.

The JuMP method:

 1. Pass `w` into `opt` with [`factory`](@ref).
 2. Set the net budget of `opt` and of each member of its chain to one with [`leader_chain`](@ref).
 3. Optimise on `rd` with the entry of the set, [`leader_block`](@ref), as the keyword `blocks`, giving `res`. When its retcode is an [`OptimisationSuccess`](@ref), return `res.w`.
 4. Otherwise record a Held Step through [`record_held_step!`](@ref) with the trials of the solver, and return `nothing`.

# Arguments

  - `opt`: The held estimator.
  - `rd`: The rows that the head holds, as a [`ReturnsResult`](@ref), viewed at the sample. It holds the returns as they are, `NaN` where no return exists, under the pinned names and the Asset Panel of the buffer.
  - `w`: The Price-Adjusted Allocation of the step.
  - `set`: The resolved Allocation Set of the head, a [`BoundedAllocationSet`](@ref) for the solver-free method.
  - `X`: Every row that the head holds, as a [`ReturnsResult`](@ref). The JuMP method gives it to the block of the set, and the solver-free method does not read it.

# Returns

  - `w::Union{Nothing, AbstractVector}`: The leader, or `nothing` on a Held Step.

# Related

  - [`FollowTheLeader`](@ref)
  - [`leader_block`](@ref)
  - [`leader_chain`](@ref)
  - [`HeldStep`](@ref)
"""
function leader_allocation(opt::BestConstantRebalancedPortfolio, rd::ReturnsResult,
                           w::AbstractVector, set::BoundedAllocationSet, ::Any)
    res = optimise(factory(leader_chain(opt, set, one(eltype(w))), w), rd)
    fp = isa(res.retcode, OptimisationSuccess) ? res.retcode.res : nothing
    if isa(fp, NamedTuple) && !get(fp, :converged, true)
        @warn("the solver-free leader stopped at `iters = $(opt.iters)` on $(size(rd.X, 1)) selected rows, $(fp.gap) or less short of the optimal log wealth; later short steps at this `iters` are not reported. The exact leader is `MeanRisk` under `LogarithmicReturn` and `MaximumReturn` on a solver.",
              maxlog = log_limit(:warn, 1), _id = @log_id(opt.iters))
    end
    return res.w
end
function leader_allocation(opt::JuMPOptimisationEstimator, rd::ReturnsResult,
                           w::AbstractVector, set::AbstractAllocationSet, X::ReturnsResult)
    lead = leader_chain(factory(opt, w), set, one(eltype(w)))
    res = optimise(lead, rd; blocks = (leader_block(set, opt, w, X),))
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
    idx = select_rows(alg.sel, price_relative.(head_frame_returns(rows)))
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
    ShortTermLossControlPortfolio(; window::Integer = 5, gamma::Real = 0.025, slv::Slv_VecSlv, pe::AbstractPriorEstimator = EmpiricalPrior(; ce = RankOneCovariance()), proj::EuclideanProjection = EuclideanProjection(), sc::Number = 1, so::Number = 1)

Builds the short-term portfolio optimisation with loss control (SPOLC).

It is the rule of [lai2020spolc](@citet). The rule is a [`FollowTheLeader`](@ref) over the last `window` periods. Its programme maximises the worst increasing factor of the window less `gamma` times the portfolio variance under the [`RankOneCovariance`](@ref). [`MeanRisk`](@ref) states it as the minimum of [`WorstRealisation`](@ref) plus [`Variance`](@ref) scaled by `gamma`. The variance reads the rank-one matrix as a quadratic form under [`QuadRiskExpr`](@ref), which needs no factor of the singular matrix. The head's Allocation Set enters the programme as in every follow-the-leader programme. [lai2020spolc](@citet) do not state the rule before the window fills. Here the rule re-solves on the periods that it has, and on one period it plays the uniform portfolio, because the programme reads a covariance.

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
  - ``\\gamma``: Weight of the variance against the worst increasing factor, as in [lai2020spolc](@cite).
  - $(math_dict[:Sigma_hat_RO])
  - ``w``: Window of the rule, as in [lai2020spolc](@cite).
  - $(math_dict[:N])
  - $(math_dict[:t_period])

This is Equation 51 of [lai2020spolc](@cite) at period ``t``. On the simplex ``\\langle \\boldsymbol{w}, \\boldsymbol{x}_i \\rangle - 1`` is the portfolio return of period ``i``. So the worst increasing factor less one is the negative of the worst realisation over the sample, and the two programmes have one maximiser.

# Arguments

  - `window`: ``w`` in [lai2020spolc](@cite), the number of periods of the window.
  - `gamma`: ``\\gamma`` in [lai2020spolc](@cite), the weight of the variance against the worst increasing factor.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme, whose covariance is the rank-one estimate. When it states `x_src = ExcessSource()`, the leader fits on the returns in excess of the benchmark that the head's rows carry, so the programme bounds the worst excess return and the tracking variance.
  - `proj`: The geometry of the damped mix. The rule's `gamma = 0` does not read it.
  - $(arg_dict[:sc]) The [`JuMPOptimiser`](@ref) of the programme takes it.
  - $(arg_dict[:so]) The [`JuMPOptimiser`](@ref) of the programme takes it. The rank-one covariance is near ``\\zeta \\boldsymbol{1} \\boldsymbol{1}^\\intercal / N`` on a window of correlated assets, with a large ``\\zeta``, so the objective can be large against the constraints. A value below one can then help the solver, and it does not change the leader.

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
  - [`RankOneCovariance`](@ref): the rank-one estimate of [lai2020spolc](@cite).
  - [`WorstRealisation`](@ref)
  - [`Variance`](@ref)
  - [`MeanRisk`](@ref)

# References

  - $(ref_dict[:lai2020spolc]) Section 3.4, Equation 51 and Section 4.2.1.
"""
function ShortTermLossControlPortfolio(; window::Integer = 5, gamma::Real = 0.025,
                                       slv::Slv_VecSlv,
                                       pe::AbstractPriorEstimator = EmpiricalPrior(;
                                                                                   ce = RankOneCovariance()),
                                       proj::EuclideanProjection = EuclideanProjection(),
                                       sc::Number = 1, so::Number = 1)::FollowTheLeader
    @argcheck(gamma >= zero(gamma), DomainError(gamma, "gamma must be non-negative"))
    return FollowTheLeader(; sel = LastRows(; W = window),
                           opt = loss_control_programme(gamma, slv, pe, sc, so),
                           proj = proj)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the programme of the short-term portfolio optimisation with loss control.

The programme is [`MeanRisk`](@ref) under [`MinimumRisk`](@ref) over [`WorstRealisation`](@ref) plus [`Variance`](@ref) scaled by `gamma` under [`QuadRiskExpr`](@ref). Its prior states its own source. The method is typed on the solver and the prior, so the optimiser that it builds is concrete for them.

# Arguments

  - `gamma`: The weight of the variance.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme.
  - $(arg_dict[:sc])
  - $(arg_dict[:so])

# Returns

  - `opt::MeanRisk`: The programme.

# Related

  - [`ShortTermLossControlPortfolio`](@ref)
"""
function loss_control_programme(gamma::Real, slv::T1, pe::T2, sc::Number,
                                so::Number) where {T1 <: Slv_VecSlv,
                                                   T2 <: AbstractPriorEstimator}
    return MeanRisk(;
                    r = [WorstRealisation(),
                         Variance(; alg = QuadRiskExpr(),
                                  settings = RiskMeasureSettings(; scale = gamma))],
                    obj = MinimumRisk(),
                    opt = JuMPOptimiser(; pe = pe, slv = slv, sc = sc, so = so))
end
"""
    LowDimensionEnsemblePortfolio(; N::Integer, window::Integer = 5, gamma::Real = 0.25, xi::Real = 0.002, slv::Slv_VecSlv, pe::AbstractPriorEstimator = LowDimensionEnsemblePrior(), proj::EuclideanProjection = EuclideanProjection(), sc::Number = 1, so::Number = 1)

Builds the online low-dimension ensemble method (OLDEM).

It is the method of [xi2023oldem](@citet). The rule is a [`FollowTheLeader`](@ref) over the last `window` regression pairs. Its programme maximises the forecast return of the ensemble, less `gamma` times the portfolio variance under the predictive covariance of the ensemble, less a linear turnover fee at the rate `xi`. [`MeanRisk`](@ref) states it as [`MaximumUtility`](@ref) at the risk aversion `gamma` over [`Variance`](@ref) under [`QuadRiskExpr`](@ref). The ensemble prior on `pe` gives the forecast and the covariance from one fit. A [`Fees`](@ref) whose [`Turnover`](@ref) rate is `xi` gives the ``\\ell_1`` term, and the head passes the reference allocation through [`factory`](@ref), as it does for every fee. The head's Allocation Set enters the programme as in every follow-the-leader programme.

[xi2023oldem](@citet) relax ``\\boldsymbol{w} \\geq \\boldsymbol{0}``, run a coordinate-wise descent on the change of the allocation, and project the result onto the simplex. The programme here keeps ``\\boldsymbol{w} \\geq \\boldsymbol{0}`` inside the solve. So it returns the minimiser of the programme of [xi2023oldem](@cite), which the relaxed and projected result is not in general. `window` regression pairs need `window + 1` rows, so the selector holds one row more than ``w`` in [xi2023oldem](@cite). Before the window fills the rule re-solves on the rows that it has. Below three rows the covariance of the regressors does not exist, and the rule plays the uniform portfolio.

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
  - ``\\gamma``: Risk aversion over the predictive variance, as in [xi2023oldem](@cite).
  - $(math_dict[:Sigma_hat_pred])
  - ``\\xi``: Rate of the linear turnover fee, as in [xi2023oldem](@cite).
  - $(math_dict[:w_hat_t_padj])
  - $(math_dict[:N])
  - $(math_dict[:t_period])

This is Equation 14 of [xi2023oldem](@cite), with one change. [xi2023oldem](@citet) measure the turnover from the last allocation ``\\hat{\\boldsymbol{b}}_t`` of the rule, and the programme here measures it from the Price-Adjusted Allocation ``\\hat{\\boldsymbol{w}}_t``, the book that the fund holds when it trades. On the simplex ``\\hat{\\boldsymbol{x}}^\\intercal \\boldsymbol{w}`` and ``(\\hat{\\boldsymbol{x}} - \\boldsymbol{1})^\\intercal \\boldsymbol{w}`` differ by one, so the programme over the forecast return has the same minimiser.

# Arguments

  - `N`: The number of assets, the length of the reference allocation of the fee.
  - `window`: ``w`` in [xi2023oldem](@cite), the number of regression pairs of the window. The selector holds `window + 1` rows.
  - `gamma`: ``\\gamma`` in [xi2023oldem](@cite), the risk aversion over the predictive variance.
  - `xi`: ``\\xi`` in [xi2023oldem](@cite), the rate of the linear turnover fee.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme, the ensemble by default. Any prior gives the two moments that the programme reads. When it states `x_src = ExcessSource()`, the leader fits on the returns in excess of the benchmark that the head's rows carry, so the programme maximises the forecast excess return less the tracking variance.
  - `proj`: The geometry of the damped mix. The rule's `gamma = 0` does not read it.
  - $(arg_dict[:sc]) The [`JuMPOptimiser`](@ref) of the programme takes it.
  - $(arg_dict[:so]) The [`JuMPOptimiser`](@ref) of the programme takes it.

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
                                       proj::EuclideanProjection = EuclideanProjection(),
                                       sc::Number = 1, so::Number = 1)::FollowTheLeader
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
                                                    slv, pe, sc, so), proj = proj)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the programme of the online low-dimension ensemble method.

The programme is [`MeanRisk`](@ref) under [`MaximumUtility`](@ref) at the risk aversion `gamma` over [`Variance`](@ref) under [`QuadRiskExpr`](@ref), and it charges `fees`. Its prior states its own source. The method is typed on the solver and the prior, so the optimiser that it builds is concrete for them.

# Arguments

  - `gamma`: The risk aversion over the predictive variance.
  - `fees`: The fees of the programme, a [`Turnover`](@ref) fee for the rule.
  - `slv`: The solver of the programme.
  - `pe`: The prior estimator of the programme.
  - $(arg_dict[:sc])
  - $(arg_dict[:so])

# Returns

  - `opt::MeanRisk`: The programme.

# Related

  - [`LowDimensionEnsemblePortfolio`](@ref)
"""
function ensemble_programme(gamma::Real, fees::Fees, slv::T1, pe::T2, sc::Number,
                            so::Number) where {T1 <: Slv_VecSlv,
                                               T2 <: AbstractPriorEstimator}
    return MeanRisk(; r = Variance(; alg = QuadRiskExpr()),
                    obj = MaximumUtility(; l = gamma),
                    opt = JuMPOptimiser(; pe = pe, slv = slv, fees = fees, sc = sc,
                                        so = so))
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the live experts of a follow-the-leading-history rule, with their Rule States, start periods and weights.

It is the state of [`FollowTheLeadingHistory`](@ref): what the rule keeps from one update to the next. The rule refuses a merge of two such states, because the head's allocation depends on the order of the rows.

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
    The state of each live expert, `nothing` when the base rule keeps nothing.
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
    return throw(ArgumentError("a `FollowTheLeadingHistoryState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the state of the rule follows it."))
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

This is follow the leading history of [hazanseshadhri2009](@citet). A multiplicative update weights the copies by their returns, and at every period a new copy starts with a fixed share of the weight. Under `prune` the rule keeps a working set of copies of logarithmic size. The experiments of [hazanseshadhri2009](@cite) use the Online Newton Step as the base rule, which is [`NewtonStep`](@ref) here, and any rule of the family serves.

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

The factor ``r_{t, k}^{\\alpha}`` is the factor ``e^{-\\alpha f_t}`` of [hazanseshadhri2009](@cite) under the log-wealth loss ``f_t(\\boldsymbol{w}) = -\\log \\langle \\boldsymbol{w}, \\boldsymbol{x}_t \\rangle``, and at ``\\alpha = 1`` it is the wealth weighting. Without `prune`, ``S_{t+1} = \\lbrace 1, \\ldots, t + 1 \\rbrace`` and the last normalisation divides by one. Under `prune`, ``S_{t+1}`` holds the start periods that are alive at ``t + 1`` under the lifetime of [`expert_alive`](@ref), which Appendix A of [hazanseshadhri2009](@cite) states. Then ``S_t`` holds ``O(\\log t)`` copies, and for every ``s \\leq t`` it holds a start period in ``[s, (s + t)/2]``.

Suppose that the base rule's regret on every interval ``I`` is ``\\alpha^{-1} \\log |I|``, and that every loss is ``\\alpha``-exp-concave. When the copy started at ``r`` lives through ``I = [r, s]``, Lemma 3.1 of [hazanseshadhri2009](@cite) bounds the regret of the rule on ``I`` by ``O(\\alpha^{-1} (\\ln r + \\ln |I|))``. Without `prune` every copy lives, so this bound holds on every interval. Under `prune`, Lemma 3.2 of [hazanseshadhri2009](@cite) bounds the regret on every interval by ``O(\\alpha^{-1} \\log s \\cdot \\log |I| + 1)``.

# Algorithm

The steps of the Online Update:

 1. Set `t` to `st.n + 1`, and `r` to the return ``\\langle \\boldsymbol{h}, \\boldsymbol{x} \\rangle`` of the allocation `h` of each live copy. `st` is the state of the rule: what the rule keeps from one update to the next, here a [`FollowTheLeadingHistoryState`](@ref).
 2. Set `phat` to `st.p .* r .^ alg.alpha`, divided by its sum.
 3. Step every live copy with its own Online Update, which gives its Rule State and its allocation for the next period.
 4. Start a new copy at the base rule's start from the uniform portfolio, projected onto `set` in the base rule's geometry, with the start period `t + 1`.
 5. Set `p` to `(1 - 1 / (t + 1)) .* phat`, followed by the share `1 / (t + 1)` of the new copy.
 6. Under `alg.prune`, keep the copies that `expert_alive` finds alive at `t + 1`, with their entries of `p`.
 7. Project `p` onto the Expert Set with the entropic projection. On the bare simplex over the copies this divides `p` by its sum.
 8. Set `q` to the sum of `p[k] .* h` over the copies. Return the new state of the rule, and the result of `blend_projection` of `q` onto `set` with `alg.proj`, from the Price-Adjusted Allocation.

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

[`port_opt_view`](@ref) takes the view of the base rule and carries the rest unchanged.

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
  - [`NewtonStep`](@ref): the base rule of the experiments of [hazanseshadhri2009](@cite).
  - [`expert_alive`](@ref): the lifetime that the pruning reads.

# References

  - $(ref_dict[:hazanseshadhri2009]) Algorithm 1, Property 3.1, Lemmas 3.1 and 3.2, Section 4 and Appendix A.
"""
@concrete struct FollowTheLeadingHistory <: AbstractOnlinePortfolioSelectionAlgorithm
    """
    The base rule every expert is a copy of.
    """
    alg
    """
    The exponent of the multiplicative update, ``\\alpha`` in [hazanseshadhri2009](@cite). At `1` the update is the wealth weighting.
    """
    alpha
    """
    Whether the rule prunes the working set to the streaming set of logarithmic size of [hazanseshadhri2009](@cite).
    """
    prune
    """
    The Expert Set onto which the rule projects the weight vector, a scalar bound over the live experts, or `nothing` for the bare simplex over them.
    """
    eset
    """
    The geometry of the last projection of the blend onto the head's Allocation Set.
    """
    proj
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

Checks whether the copy started at period `born` is alive at period `t`, under the lifetime of follow the leading history.

The lifetime is that of [hazanseshadhri2009](@cite). The function reads `t >= born` as given.

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

[hazanseshadhri2009](@citet) call ``2^{j + 2} + 1`` the lifetime of ``k``, so the copy is alive in ``2^{j + 2} + 2`` periods.

# Arguments

  - `born`: The start period of the copy.
  - `t`: The period.

# Returns

  - `alive::Bool`: `t <= born + 2^(j + 2) + 1`, with `j` the number of trailing zeros of `born`.

# Related

  - [`FollowTheLeadingHistory`](@ref): the rule whose pruning reads the lifetime.

# References

  - $(ref_dict[:hazanseshadhri2009]) Section 3.1 and Appendix A.
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
public AbstractSampleSelector, AbstractPatternMatchSelector, select_rows, matched_rows
