"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Sample Selectors a [`FollowTheLeader`](@ref) rule holds on `sel`: the part of a solved rule that names which past rows the held optimisation estimator re-solves on.

Every solved row of the online selection literature is one programme over a sample of past price relatives, and the rows differ in the sample alone: every row so far (follow the leader), the last `W` (the variable rebalanced portfolio), or the rows whose preceding window resembles the latest one — by histogram cell, kernel radius, nearest neighbours, correlation or cluster — which is the pattern-matching family. A selector reads the price relatives the head holds and answers row indices; it holds no rows of its own.

# Interfaces

In order to implement a new selector, subtype `AbstractSampleSelector` with the paper's parameters as part of the struct, and implement:

  - `select_rows(sel::AbstractSampleSelector, X::AbstractMatrix) -> AbstractVector{<:Integer}`: The indices of the rows of `X` the rule re-solves on, `X` the price relatives the head holds through the period, `observations × assets`, in time order. An empty vector is the empty selection.
  - `rows_needed(sel::AbstractSampleSelector) -> Union{Nothing, Integer}`: The rows the selector reads at a step, `nothing` for every row folded so far.

## Arguments

  - `sel`: The selector.
  - `X`: The price relatives, `1 .+ r`, every row the head holds.

## Returns

  - `idx::AbstractVector{<:Integer}`: The selected row indices, in time order.

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

The indices of the rows a Sample Selector names, from the price relatives the head holds through the period.

# Related

  - [`AbstractSampleSelector`](@ref)
  - [`FollowTheLeader`](@ref)
"""
function select_rows end
"""
$(DocStringExtensions.TYPEDEF)

The Sample Selector of follow the leader: every row so far, so the re-solve is the best constant rebalanced portfolio to date (Gaivoronski and Stella 2000's successive constant rebalanced portfolio, SCRP).

# Examples

```jldoctest
julia> Prefix()
Prefix()
```

# Related

  - [`AbstractSampleSelector`](@ref)
  - [`FollowTheLeader`](@ref)
  - [`LastRows`](@ref)

# References

  - $(ref_dict[:gaivoronski2000])
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

The Sample Selector of the variable rebalanced portfolio: the last `W` rows, so the re-solve is the best constant rebalanced portfolio of a moving window (Gaivoronski and Stella 2000, VRP).

Before `W` rows exist the selection is every row so far, the window filling as the rows arrive: the rule re-solves on what it holds, as the forecast-reading rules read a statistic over the rows they have.

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
  - [`Prefix`](@ref)

# References

  - $(ref_dict[:gaivoronski2000])
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

Abstract supertype for the pattern-matching Sample Selectors: the ones that name the rows whose preceding window of `window` price relatives resembles the latest window.

# Interfaces

In order to implement a new pattern-matching selector, subtype `AbstractPatternMatchSelector` with a `window` field and the paper's similarity parameters, and implement:

  - `matched_rows(sel::AbstractPatternMatchSelector, X::AbstractMatrix, cands::AbstractVector, latest::AbstractVector) -> AbstractVector{<:Integer}`: Which of the candidate rows match, from the candidates' flattened preceding windows `cands` — one vector per candidate, the window's rows concatenated — and the latest window flattened the same way.

[`select_rows`](@ref) forms the candidates for it: at `T` rows, the candidate rows are `window + 1` to `T`, each compared through its preceding `window` rows, and the latest window is the last `window` rows. Before `window + 1` rows exist there is no candidate and the selection is empty, which the rule answers with the uniform portfolio. Every selector of the kind reads every row folded so far.

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

Which candidate rows a pattern-matching selector matches, as indices into `cands`, the candidates' flattened preceding windows, against `latest`, the latest window flattened the same way.

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

The candidate rows of a pattern-matching selector and their flattened preceding windows, beside the latest window: rows `window + 1` to `T`, each window the `window` rows before the row concatenated column-major, `nothing` when fewer than `window + 1` rows exist.

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

Refuses a window of less than one row.

# Related

  - [`AbstractPatternMatchSelector`](@ref)
"""
function assert_pattern_window(window::Integer)::Nothing
    @argcheck(window >= 1, DomainError(window, "window must be at least 1"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

The histogram-based Sample Selector of Györfi and Schäfer (2003): the rows whose preceding window falls in the same cell of a partition of the price relatives as the latest window (BH).

# Mathematical definition

With ``G`` the partition of the positive orthant into product cells, every entry of a window binned by `edges`,

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : G\\left(\\boldsymbol{x}_{i-w}^{i-1}\\right) = G\\left(\\boldsymbol{x}_{t-w+1}^{t}\\right) \\right\\rbrace\\,.
\\end{align}
```

The paper leaves the partition to the user and asks that a finer one exists; here it is the product of one binning of every entry, an entry's cell being the number of `edges` at or below it. The default `edges = [1]` splits every price relative at one — the asset rose or it fell — so two windows match when every asset moved in the same direction on every row, the coarsest partition that reads the market at all; more edges make a finer partition and a smaller sample.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HistogramMatch(; window::Integer = 5, edges::AbstractVector{<:Real} = [1]) -> HistogramMatch

Keywords correspond to the struct's fields.

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `edges` is non-empty, finite and sorted in increasing order. An `ArgumentError` is thrown otherwise.

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
  - [`KernelMatch`](@ref)

# References

  - $(ref_dict[:gyorfischafer2003])
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

The cell of a flattened window under a binning: one bin index per entry, the number of `edges` at or below it.

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

The kernel-based Sample Selector of Györfi, Lugosi and Udina (2006): the rows whose preceding window lies within `radius` of the latest window, in the Euclidean norm of the concatenated windows, a uniform kernel (BK).

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : \\left\\lVert \\boldsymbol{x}_{i-w}^{i-1} - \\boldsymbol{x}_{t-w+1}^{t} \\right\\rVert \\leq r \\right\\rbrace\\,.
\\end{align}
```

The paper's experts run over a grid of windows and radii `r = c / l` and are aggregated by wealth, which is an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each; a selector holds one radius. The kernel-based semi-log-optimal (Györfi, Urbán and Vajda 2007), Markowitz-type (Ottucsák and Vajda 2007) and transaction-cost (Györfi and Vajda 2008) rules are this selector under another objective on the rule's `opt`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KernelMatch(; window::Integer = 5, radius::Real) -> KernelMatch

Keywords correspond to the struct's fields. The radius has no paper default: the papers scan it, and its scale is the scale of a window of price relatives.

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
  - [`NearestNeighbourMatch`](@ref)

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

The nearest-neighbour Sample Selector of Györfi, Udina and Walk (2008): the rows whose preceding window is among the `neighbours` nearest to the latest window, in the Euclidean norm of the concatenated windows (BNN).

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : \\boldsymbol{x}_{i-w}^{i-1} \\text{ is among the } \\ell \\text{ nearest neighbours of } \\boldsymbol{x}_{t-w+1}^{t} \\right\\rbrace\\,.
\\end{align}
```

The paper takes ``\\ell = \\lfloor p_\\ell\\, n \\rfloor`` neighbours at period ``n``, a count that grows with the history, so `neighbours` is either an `Integer` count or a fraction in `(0, 1)`. The fraction is taken of the candidate rows, ``n - 1 - w`` of them, and not of the period, so the count never exceeds the candidates and is below the paper's by at most ``\\lceil p_\\ell (w + 1) \\rceil``; a fraction that rounds to no neighbour is the empty selection. Ties at the boundary are broken by the order of the rows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NearestNeighbourMatch(; window::Integer = 5, neighbours::Real = 0.1) -> NearestNeighbourMatch

Keywords correspond to the struct's fields.

## Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `neighbours`: an `Integer` at least `1`, or a real in `(0, 1)`. A `DomainError` is thrown otherwise.

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
  - [`KernelMatch`](@ref)

# References

  - $(ref_dict[:gyorfi2008])
"""
struct NearestNeighbourMatch{T1 <: Integer, T2 <: Real} <: AbstractPatternMatchSelector
    """
    The number of rows of the windows compared.
    """
    window::T1
    """
    The neighbours kept: an `Integer` count, or a fraction in `(0, 1)` of the candidate rows.
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

The number of neighbours kept out of `n` candidates: a count capped at `n`, or the floor of a fraction of `n`.

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

The correlation-driven Sample Selector of Li, Hoi and Gopalkrishnan (2011): the rows whose preceding window correlates with the latest window at `rho` or above, the Pearson correlation of the two concatenated windows (CORN).

# Mathematical definition

```math
\\begin{align}
C_t &= \\left\\lbrace w < i \\leq t : \\operatorname{corr}\\left(\\boldsymbol{x}_{i-w}^{i-1},\\, \\boldsymbol{x}_{t-w+1}^{t}\\right) \\geq \\rho \\right\\rbrace\\,.
\\end{align}
```

A window with no variation has no correlation and never matches. The paper's `CORN-U` mixes the experts of windows `1` to `W` uniformly by wealth and `CORN-K` keeps the top `K`, both an [`ExpertMixture`](@ref) over [`FollowTheLeader`](@ref) rules with one selector each; the defaults are the paper's `w = 5`, `ρ = 0.1`. The risk-aversion rule of Wang, Wang, Wang and Zhang (2018) is the same selector under a standard-deviation penalty on the log return — [`MeanRisk`](@ref) with a [`MaximumUtility`](@ref) objective over [`StandardDeviation`](@ref) under [`LogarithmicReturn`](@ref) on the rule's `opt` — one expert per threshold, and the [`TopK`](@ref) weighting over them (RACORN-K).

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
  - [`ExpertMixture`](@ref)

# References

  - $(ref_dict[:li2011corn])
  - $(ref_dict[:wang2018racorn])
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
        return isfinite(r) && r >= sel.rho
    end
end
"""
$(DocStringExtensions.TYPEDEF)

The cluster-based Sample Selector of Khedmati and Azin (2020): the rows whose preceding window falls in the same cluster as the latest window, the windows clustered by one of the library's clustering estimators (KMNLOG).

The candidate windows and the latest one are laid out as the columns of one matrix, each window's rows concatenated, and handed to the clusterer as the units it partitions; a candidate whose label is the latest window's is selected. The paper clusters the windows by k-means, k-medoids, spectral and hierarchical methods; the slot is bound to the library's [`AbstractClustersEstimator`](@ref), which partitions its units through a distance the estimator forms from them, `KMeansAlgorithm` and `HClustAlgorithm` among its algorithms. The paper's transaction-cost term is the `fees` slot of the rule's `opt`. Fewer than two candidates cannot be partitioned and are the empty selection.

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
  - [`ClustersEstimator`](@ref)
  - [`KMeansAlgorithm`](@ref)

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

The Allocation Set Constraint: the adapter through which a [`FollowTheLeader`](@ref) rule's re-solve honours the head's Allocation Set.

It wraps the resolved set, the step's Price-Adjusted Allocation and the head's rows as a [`CustomJuMPConstraint`](@ref), which the rule appends to its held optimiser's `ccnt` for that solve; [`add_custom_constraint!`](@ref) on it calls [`add_allocation_set_constraints!`](@ref), the set's own builders on the leader's model, so the programme's feasible region is the set intersected with whatever the held optimiser carries itself. The turnover ceiling measures from the Price-Adjusted Allocation and the variance ceiling and tracking error read the set's own `pe` on the head's rows, as they do in a projection. Both homes hold: a caller who wants one leaves the held optimiser bare of constraints and writes them on the set.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`FollowTheLeader`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`CustomJuMPConstraint`](@ref)
  - [`AbstractAllocationSet`](@ref)
"""
struct AllocationSetConstraint{T1 <: AbstractAllocationSet, T2 <: AbstractVector, T3} <:
       CustomJuMPConstraint
    """
    The Allocation Set, resolved over the pinned universe.
    """
    set::T1
    """
    The Price-Adjusted Allocation the step trades from.
    """
    w::T2
    """
    The rows of returns the head holds through the period, or `nothing`.
    """
    X::T3
end
function add_custom_constraint!(model::JuMP.Model, ccnt::AllocationSetConstraint, ::Any,
                                ::Any)::Nothing
    add_allocation_set_constraints!(model, ccnt.set, ccnt.w, ccnt.X)
    return nothing
end
"""
    const LeaderOptimiser = Union{<:BestConstantRebalancedPortfolio, <:JuMPOptimisationEstimator}

The optimisation estimators a [`FollowTheLeader`](@ref) rule re-solves: the solver-free [`BestConstantRebalancedPortfolio`](@ref), and any JuMP head, whose `ccnt` takes the [`AllocationSetConstraint`](@ref).

# Related

  - [`FollowTheLeader`](@ref)
"""
const LeaderOptimiser = Union{<:BestConstantRebalancedPortfolio,
                              <:JuMPOptimisationEstimator}
"""
$(DocStringExtensions.TYPEDEF)

The follow-the-leader rule: at every period, the optimisation estimator on `opt` is re-solved on the past rows the Sample Selector on `sel` names, and the answer is the solved allocation damped towards the current one, ``(1 - \\gamma)\\, \\boldsymbol{w}^\\star_t + \\gamma\\, \\boldsymbol{w}_t``.

Every solved row of the literature is this rule under one selector and one objective. [`Prefix`](@ref) with the log-optimal objective is follow the leader, the successive constant rebalanced portfolio of Gaivoronski and Stella (2000), and `gamma > 0` their weighted one (WSCRP); [`LastRows`](@ref) is their variable rebalanced portfolio; the pattern-matching selectors are the histogram, kernel, nearest-neighbour, correlation and cluster rules, whose papers aggregate a grid of selectors by wealth — an [`ExpertMixture`](@ref) over rules of this kind. The objective is the held estimator's: [`BestConstantRebalancedPortfolio`](@ref), the default, is the log-optimal portfolio with no solver, and [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref) and [`MaximumReturn`](@ref) is the same programme on a solver, where `L2Regularisation(; val = 1 / (2n), alg = QuadRiskExpr())` on `n` selected rows is the exp-concave leader of Hazan and Kale (2012) — the log return is the mean over the rows, so the paper's ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} \\rVert^2`` against the sum of the logs is `1 / (2n)` against their mean, and one fixed `val` is that leader at one sample size alone — a [`MaximumUtility`](@ref) objective over a risk the Markowitz-type row, a quadratic objective the semi-log-optimal row, and `fees` the cost-aware row.

The solver-free default runs Cover's fixed point, whose stop rule reads the change in log wealth. At a leader that drops an asset the weight of that asset decays sublinearly and the log wealth is flat to second order in it, so the fixed point stops at `tol` or at `iters` short of the leader: on forty rows of four assets with 2 % Gaussian returns the default budget stops up to `7e-2` from the leader in weight and `6e-5` in log wealth, `tol = 1e-12` reports convergence `3e-4` from it, and `tol = 1e-16` under a budget of two million steps reaches `2e-6`. The solved form on a JuMP head is exact to the solver's tolerance, and an interior leader is exact under both.

**The re-solve takes the head's Allocation Set as its feasible region.** The update forms the step's Price-Adjusted Allocation, wraps the resolved set, that allocation and the head's rows into an [`AllocationSetConstraint`](@ref), appends it to the held JuMP estimator's `ccnt` for the solve, threads the allocation through [`factory`](@ref) as a fold loop does — so a turnover term or a fee on `opt` reads the same book — and takes the optimum as it is, with no projection: the projected leader is not the leader. The solver-free [`BestConstantRebalancedPortfolio`](@ref) has no model to add to, so a [`BoundedAllocationSet`](@ref) goes into its `wb` and `sets` and the answer is its repaired fixed point, and a [`ProgrammeAllocationSet`](@ref) is refused by name at the head's construction: the constrained leader under a programme set is [`MeanRisk`](@ref) under [`LogarithmicReturn`](@ref).

**Only the damped mix is projected**, in the Euclidean geometry on `proj`, with the Price-Adjusted Allocation as the reference: the identity on every static convex kind, the repair a turnover ceiling or a MIP kind needs on the day the mix leaves the set, skipped by type on a [`BoundedAllocationSet`](@ref) and at `gamma = 0`. An empty selection, or one too small for the estimator — a JuMP head fits a covariance and needs two rows — answers the uniform portfolio projected onto the set in the same geometry, as the papers do; [`LastRows`](@ref) re-solves on the rows it has until its window fills, and a pattern-matching selector has no candidate until `window + 1` rows exist. A re-solve that does not succeed after the held estimator's own fallback chain is a Held Step: the update answers the Price-Adjusted Allocation, so the fund trades nothing that period.

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
  - [`AllocationSetConstraint`](@ref)
  - [`BestConstantRebalancedPortfolio`](@ref)
  - [`ExpertMixture`](@ref)
  - [`FollowTheLeadingHistory`](@ref)
  - [`ShortTermLossControlPortfolio`](@ref)

# References

  - $(ref_dict[:gaivoronski2000])
  - $(ref_dict[:hazankale2012])
  - $(ref_dict[:lihoi2014])
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
    The damping towards the current allocation, in `[0, 1]`; `0` plays the solved allocation.
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

The fewest selected rows the held estimator re-solves on: two for an estimator whose tree holds a second moment, whose covariance of one observation does not exist, one otherwise, and the floor any estimator in the tree states through [`fit_min_rows`](@ref) where that is larger.

# Related

  - [`FollowTheLeader`](@ref)
  - [`holds_second_moment`](@ref)
  - [`fit_min_rows`](@ref)
"""
function leader_min_rows(opt::LeaderOptimiser)
    return max(holds_second_moment(opt) ? 2 : 1, fit_min_rows(opt))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The carrier a [`FollowTheLeader`](@ref) rule hands its estimator: the selected rows of returns under the names the current [`ProjectionStep`](@ref) pins, or the column indices as names outside a step.

# Related

  - [`FollowTheLeader`](@ref)
  - [`projection_step_names`](@ref)
"""
function leader_carrier(R::AbstractMatrix)
    nx = projection_step_names()
    if isnothing(nx)
        nx = string.(axes(R, 2))
    end
    return ReturnsResult(; nx = nx, X = R)
end
"""
    append_custom_constraint(ccnt::Nothing, c::CustomJuMPConstraint)
    append_custom_constraint(ccnt::CustomJuMPConstraint, c::CustomJuMPConstraint)
    append_custom_constraint(ccnt::AbstractVector{<:CustomJuMPConstraint}, c::CustomJuMPConstraint)

The held estimator's custom constraints with the [`AllocationSetConstraint`](@ref) appended: the constraint alone, a vector of two, or the vector with one more.

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
    leader_allocation(opt::BestConstantRebalancedPortfolio, R::AbstractMatrix, w::AbstractVector, set::BoundedAllocationSet, X)
    leader_allocation(opt::JuMPOptimisationEstimator, R::AbstractMatrix, w::AbstractVector, set::AbstractAllocationSet, X)

The leader: the held estimator re-solved on the selected rows `R` under the head's set, or `nothing` on a Held Step.

The solver-free arm takes the bounded set's `wb` and `sets` onto the fixed point's own slots and answers the repaired fixed point. The JuMP arm threads the Price-Adjusted Allocation `w` through [`factory`](@ref), appends the [`AllocationSetConstraint`](@ref) over the set, `w` and the head's rows `X` to the estimator's `ccnt`, and takes the optimum as it is; a result whose retcode is not an [`OptimisationSuccess`](@ref) records a Held Step through [`record_held_step!`](@ref) with the solver's trials and answers `nothing`.

# Related

  - [`FollowTheLeader`](@ref)
  - [`AllocationSetConstraint`](@ref)
  - [`HeldStep`](@ref)
"""
function leader_allocation(opt::BestConstantRebalancedPortfolio, R::AbstractMatrix,
                           w::AbstractVector, set::BoundedAllocationSet, ::Any)
    bcrp = BestConstantRebalancedPortfolio(; wb = set.wb, fees = opt.fees, sets = set.sets,
                                           wf = opt.wf, fb = opt.fb, iters = opt.iters,
                                           tol = opt.tol, strict = opt.strict,
                                           cache = opt.cache)
    return optimise(factory(bcrp, w), leader_carrier(R)).w
end
function leader_allocation(opt::JuMPOptimisationEstimator, R::AbstractMatrix,
                           w::AbstractVector, set::AbstractAllocationSet, X)
    opt = factory(opt, w)
    ccnt = append_custom_constraint(opt.opt.ccnt, AllocationSetConstraint(set, w, X))
    opt = Accessors.@set opt.opt.ccnt = ccnt
    res = optimise(opt, leader_carrier(R))
    if isa(res.retcode, OptimisationSuccess)
        return res.w
    end
    record_held_step!("the follow-the-leader programme of the `$(nameof(typeof(opt)))` did not solve on the $(size(R, 1)) selected rows, and the step trades nothing",
                      res.retcode.res)
    return nothing
end
function online_update!(alg::FollowTheLeader, st, w::AbstractVector, x::AbstractVector,
                        rows, set::AbstractAllocationSet)
    wh = price_adjusted_allocation(w, x)
    idx = select_rows(alg.sel, one(eltype(rows)) .+ rows)
    wstar = if length(idx) < leader_min_rows(alg.opt)
        project(alg.proj, set, fill(one(eltype(w)) / length(w), length(w)), wh)
    else
        leader_allocation(alg.opt, rows[idx, :], wh, set, rows)
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

The short-term portfolio optimisation with loss control of Lai, Tan, Wu and Fang (2020): a [`FollowTheLeader`](@ref) over the last `window` rows whose programme maximises the worst increasing factor of the window minus `gamma` times the portfolio variance under the [`RankOneCovariance`](@ref) (SPOLC).

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{b}} &= \\underset{\\boldsymbol{b} \\in \\Delta^N}{\\arg\\max} \\min_{1 \\leq i \\leq w} \\boldsymbol{x}_i^\\intercal \\boldsymbol{b} - \\gamma\\, \\boldsymbol{b}^\\intercal \\hat{\\Sigma}_{\\mathrm{RO}} \\boldsymbol{b}\\,,
\\end{align}
```

the paper's equation 51, which [`MeanRisk`](@ref) states as the minimum of [`WorstRealisation`](@ref) beside [`Variance`](@ref) scaled by `gamma`, the sum scalarisation of the two: the worst realisation over the window's returns is the worst increasing factor less one, and the variance takes the rank-one matrix as a quadratic form under [`QuadRiskExpr`](@ref), which needs no factor of a singular matrix. The head's Allocation Set enters the programme as every follow-the-leader programme takes it; before the window fills the rule re-solves on the rows it has, and on one row answers the uniform portfolio.

# Arguments

  - `window`: The paper's `w`, the rows of the window.
  - `gamma`: The paper's `γ`, the weight of the variance against the worst increasing factor.
  - `slv`: The solver the programme runs on.
  - `pe`: The prior estimator of the programme, whose covariance is the rank-one estimate.
  - `proj`: The geometry of the damped mix, unused at the rule's `gamma = 0`.

# Validation

  - `window >= 1`. A `DomainError` is thrown otherwise.
  - `gamma >= 0`. A `DomainError` is thrown otherwise.

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
  - [`RankOneCovariance`](@ref)
  - [`WorstRealisation`](@ref)
  - [`Variance`](@ref)
  - [`MeanRisk`](@ref)

# References

  - $(ref_dict[:lai2020spolc])
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

The programme of [`ShortTermLossControlPortfolio`](@ref): [`MeanRisk`](@ref) minimising [`WorstRealisation`](@ref) beside [`Variance`](@ref) scaled by `gamma` under [`QuadRiskExpr`](@ref), on the solver and prior given. Typed on both, so the optimiser it builds is concrete for the solver and prior it was given.

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

The online low-dimension ensemble method of Xi, Li, Song and Ning (2023): a [`FollowTheLeader`](@ref) over the last `window` regression pairs whose programme maximises the ensemble's forecast return less `gamma` times the portfolio variance under the ensemble's predictive covariance, less a linear turnover fee of `xi` against the Price-Adjusted Allocation.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{b}}_{t + 1} &= \\underset{\\boldsymbol{b} \\in \\Delta^N}{\\arg\\min} \\left\\{ -\\hat{\\boldsymbol{x}}_{t + 1}^\\intercal \\boldsymbol{b} + \\gamma\\, \\boldsymbol{b}^\\intercal \\hat{\\Sigma}_{t + 1} \\boldsymbol{b} + \\xi\\, \\lVert \\boldsymbol{b} - \\hat{\\boldsymbol{w}}_t \\rVert_1 \\right\\}\\,,
\\end{align}
```

the paper's equation 14, which [`MeanRisk`](@ref) states as [`MaximumUtility`](@ref) at risk aversion `gamma` over [`Variance`](@ref) under [`QuadRiskExpr`](@ref), with the ensemble prior on `pe` supplying both ``\\hat{\\boldsymbol{x}}_{t + 1} - \\boldsymbol{1}`` and ``\\hat{\\Sigma}_{t + 1}`` from one fit, and a [`Fees`](@ref) whose [`Turnover`](@ref) rate `xi` is the ``\\ell_1`` penalty against the current allocation ``\\hat{\\boldsymbol{w}}_t``, which the head threads through [`factory`](@ref) as every fee reads it. On the simplex ``\\hat{\\boldsymbol{x}}^\\intercal \\boldsymbol{b}`` and ``(\\hat{\\boldsymbol{x}} - \\boldsymbol{1})^\\intercal \\boldsymbol{b}`` differ by a constant, so the two programmes share their minimiser.

The paper relaxes ``\\boldsymbol{b} \\geq \\boldsymbol{0}``, runs a coordinate-wise descent on ``\\boldsymbol{c} = \\boldsymbol{b} - \\hat{\\boldsymbol{w}}_t`` and projects the answer onto the simplex; the programme here keeps the non-negativity inside the solve, so it answers the minimiser of the stated problem, which the relax-then-project answer is not in general. The head's Allocation Set enters the programme as every follow-the-leader programme takes it. `window` regression pairs need `window + 1` rows, so the selector holds one row more than the paper's `w`; before the window fills the rule re-solves on the rows it has, and below three rows, where the regressor covariance does not exist, it answers the uniform portfolio.

The turnover fee reads a reference allocation of the universe's length at construction, replaced by the Price-Adjusted Allocation at every update, so the constructor takes the number of assets `N`, as [`UniversalPortfolio`](@ref) does.

# Arguments

  - `N`: The number of assets, the length of the fee's reference allocation.
  - `window`: The paper's `w`, the regression pairs of the window; the selector holds `window + 1` rows.
  - `gamma`: The paper's `γ`, the risk aversion over the predictive variance.
  - `xi`: The paper's `ξ`, the linear turnover fee rate.
  - `slv`: The solver the programme runs on.
  - `pe`: The prior estimator of the programme, the ensemble by default; any prior supplies the two moments the programme reads.
  - `proj`: The geometry of the damped mix, unused at the rule's `gamma = 0`.

# Validation

  - `N >= 1`. A `DomainError` is thrown otherwise.
  - `window >= 2`, the fewest pairs whose regressor covariance exists. A `DomainError` is thrown otherwise.
  - `gamma >= 0`. A `DomainError` is thrown otherwise.
  - `xi >= 0`. A `DomainError` is thrown otherwise.

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
  - [`LowDimensionEnsemblePrior`](@ref)
  - [`MaximumUtility`](@ref)
  - [`Variance`](@ref)
  - [`Fees`](@ref)
  - [`Turnover`](@ref)
  - [`MeanRisk`](@ref)

# References

  - $(ref_dict[:xi2023oldem])
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

The programme of [`LowDimensionEnsemblePortfolio`](@ref): [`MeanRisk`](@ref) under [`MaximumUtility`](@ref) at risk aversion `gamma` over [`Variance`](@ref) under [`QuadRiskExpr`](@ref), charging `fees`, on the solver and prior given. Typed on both, so the optimiser it builds is concrete for the solver and prior it was given.

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

The carrier of [`FollowTheLeadingHistory`](@ref): the working set of experts, each with its Rule State and the period it was started, and the weight over them.

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
    Each live expert's carrier, one entry per expert, `nothing` where the base rule carries nothing.
    """
    st
    """
    Each live expert's allocation held during the current period, one vector per expert.
    """
    h
    """
    The weight over the live experts held during the current period.
    """
    p
    """
    The period each live expert was started at, which its lifetime is counted from.
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

Follow the leading history of Hazan and Seshadhri (2009): a mixture over copies of one rule started at different periods, weighted by a multiplicative update with a fixed share for the newest copy, and pruned to a working set of logarithmic size (FLH).

# Mathematical definition

With ``\\boldsymbol{h}_j(t)`` the allocation of the expert started at period ``j`` and ``v_t^{(j)}`` its weight,

```math
\\begin{align}
\\hat{v}_{t+1}^{(j)} &\\propto v_t^{(j)} \\left\\langle \\boldsymbol{h}_j(t), \\boldsymbol{x}_t \\right\\rangle^{\\alpha}\\,,\\quad
v_{t+1}^{(t+1)} = \\frac{1}{t + 1}\\,,\\quad
v_{t+1}^{(j)} = \\left( 1 - \\frac{1}{t + 1} \\right) \\hat{v}_{t+1}^{(j)}\\,,\\\\
\\boldsymbol{w}_{t+1} &= \\sum_{j \\in S_{t+1}} v_{t+1}^{(j)}\\, \\boldsymbol{h}_j(t+1)\\,.
\\end{align}
```

The multiplicative update is the paper's ``e^{-\\alpha f_t}`` on the log-wealth loss, the wealth weighting at `alpha = 1`, and the fixed share is what lets a newly started expert win an interval: the paper's adaptive regret on every interval `[r, s]` is `O(α⁻¹(log r + log |I|))` beyond the base rule's regret on that interval. A new expert is started at every period from the uniform portfolio in the base rule's own start, projected onto the Allocation Set in the base rule's geometry. Under `prune`, the working set is Woodruff's streaming set the paper adopts: an expert started at period ``i = r\\,2^k`` with ``r`` odd lives for ``2^{k+2} + 1`` periods, so the set holds `O(log t)` experts at period `t`, every interval still holds an expert started within its first half, and the weights are renormalised over the survivors; without it every expert lives, which is the paper's first algorithm at `O(t)` experts. The base rule of the paper is the [`NewtonStep`](@ref); any rule of the family serves.

The weight vector is projected onto the Expert Set on `eset` in the entropic geometry after the fixed share and the pruning, so a cap on `eset` caps the trust in any one copy; the working set changes size, so `eset` admits a scalar bound alone. The blend is projected onto the head's Allocation Set once more in the Euclidean geometry on `proj`, as an [`ExpertMixture`](@ref)'s is, skipped by type on a [`BoundedAllocationSet`](@ref). The rule reads nothing of a given Start Allocation beyond its first expert, which starts there.

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
  - [`ExpertMixture`](@ref)
  - [`NewtonStep`](@ref)

# References

  - $(ref_dict[:hazanseshadhri2009])
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
    The exponent of the multiplicative update, the paper's ``\\alpha``; `1` is the wealth weighting.
    """
    alpha::T2
    """
    Whether the working set is pruned to the paper's streaming set of logarithmic size.
    """
    prune::T3
    """
    The Expert Set the weight vector is projected onto, a scalar bound over the live experts, or `nothing` for the bare simplex over them.
    """
    eset::T4
    """
    The geometry the blend is projected onto the head's Allocation Set in, once more.
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
function rule_state_seed(alg::FollowTheLeadingHistory, w::AbstractVector)
    h = expert_start_allocation(alg.alg, w)
    return FollowTheLeadingHistoryState(0, [rule_state_seed(alg.alg, h)], [h],
                                        [one(eltype(w))], [1])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Whether the expert started at period `born` is alive at period `t` under the paper's streaming rule: with `born = r 2^k`, `r` odd, it lives from `born` through `born + 2^(k + 2) + 1`.

# Related

  - [`FollowTheLeadingHistory`](@ref)
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
    # The newcomer starts where the base rule starts, on the set.
    u = fill(one(eltype(w)) / length(w), length(w))
    hn = expert_start_allocation(alg.alg, project(projection_geometry(alg.alg), set, u, u))
    push!(st.st, rule_state_seed(alg.alg, hn))
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
