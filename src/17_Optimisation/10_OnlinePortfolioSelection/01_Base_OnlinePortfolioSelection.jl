"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Online Selection Rules an [`OnlinePortfolioSelection`](@ref) head holds on `alg`.

A rule is the one thing that varies across the online portfolio selection family: a struct of the paper's parameters, a private carrier for what the update accumulates, and one update method. Everything shared — the Allocation Set, the Causal Pass, the Block Step, the Recursion Read-out, the refusals — is written once on the head. A rule projects its raw step onto the head's Allocation Set in its own Projection Geometry, held on its `proj` slot, whose type bound names the geometries the rule's theorem covers.

# Interfaces

In order to implement a new rule, subtype `AbstractOnlinePortfolioSelectionAlgorithm` with the paper's parameters and a `proj` slot as part of the struct, and implement the following methods:

  - `online_update!(alg::AbstractOnlinePortfolioSelectionAlgorithm, st, w::AbstractVector, x::AbstractVector, rows, set::AbstractAllocationSet) -> Tuple`: The Online Update: from the rule's carrier `st`, the allocation `w` held during the period, the finite price relative `x` of the period and the rows the head holds through it, answer `(st', w')`, the carrier and the allocation for the next period. `st` is written in place where the rule can, and `w'` is always a new vector. A first-order rule also implements the seven-argument form, whose last argument is the Gradient Point an [`ExpertMixture`](@ref) hands it; the generic method drops the point.
  - `rule_state_seed(alg::AbstractOnlinePortfolioSelectionAlgorithm, w::AbstractVector)`: The carrier before the first update, or `nothing` for a rule that carries nothing; `w` is the Start Allocation, whose length and element type the carrier takes.
  - `rows_needed(alg::AbstractOnlinePortfolioSelectionAlgorithm) -> Union{Nothing, Integer}`: The number of rows the rule reads at a step: `0` for one that reads none, `nothing` for one that reads every row folded so far.
  - `projection_geometry(alg::AbstractOnlinePortfolioSelectionAlgorithm) -> AbstractProjectionGeometry`: The geometry the head projects the Start Allocation in, the rule's `proj` slot by default.

A rule whose carrier holds a per-asset axis also implements `port_opt_view(st, i, args...)` on the carrier's type, so a view of the head slices it, and `Base.copy(st)`, so the value form of the fold aliases nothing.

## Arguments

  - `alg`: The concrete rule.
  - `st`: The rule's private carrier, or `nothing`.
  - `w`: The allocation held during the period, over the full pinned universe, summing to one.
  - `x`: The price relative of the period, `1 .+ r`, finite at every asset.
  - `rows`: The rows of returns the head holds through the period, `observations × assets`, or `nothing` when the rule reads none.
  - `set`: The head's Allocation Set, which the rule projects onto through [`project`](@ref).

## Returns

  - `(st', w')::Tuple`: The carrier and the allocation for the next period.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`online_update!`](@ref)
  - [`AbstractProjectionGeometry`](@ref)
  - [`AbstractAllocationSet`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
"""
abstract type AbstractOnlinePortfolioSelectionAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Projection Geometries an Online Selection Rule projects its raw step back onto the Allocation Set in.

The geometry is the theorem's, not decoration: exponentiated gradient's regret bound is a relative-entropy argument and its multiplicative update followed by normalisation is the exact Kullback–Leibler projection onto the simplex; the reversion rules are Euclidean, and their sparsity is the Euclidean projection zeroing entries. The `proj` slot of every rule is bound to the geometries its theorem covers.

# Interfaces

In order to implement a new geometry, subtype `AbstractProjectionGeometry` and implement, for every Allocation Set it projects onto:

  - `project(proj::AbstractProjectionGeometry, set::AbstractAllocationSet, q::AbstractVector, w::AbstractVector) -> AbstractVector`: The projection of the raw step `q` onto `set` in this geometry, a new vector.

## Arguments

  - `proj`: The geometry.
  - `set`: The Allocation Set, its bounds resolved to vectors.
  - `q`: The raw step of the rule, before the projection.
  - `w`: The Price-Adjusted Allocation the step trades from, the reference a turnover ceiling on a [`ProgrammeAllocationSet`](@ref) reads and the allocation a Held Step answers; the bounded set does not read it.

## Returns

  - `w'::AbstractVector`: The projected allocation.

A geometry with no closed form on any set carries its own solver, as [`GramProjection`](@ref) does, and [`projection_solver`](@ref) reads it ahead of the set's.

# Related

  - [`EuclideanProjection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`GramProjection`](@ref)
  - [`project`](@ref)
  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
"""
abstract type AbstractProjectionGeometry <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The Euclidean Projection Geometry: the raw step is projected onto the Allocation Set in squared Euclidean distance, ``\\min \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert^2``.

On the simplex this is the sort of Duchi, Shalev-Shwartz, Singer and Chandra (2008), and it zeroes every entry below its threshold, which is what makes the reversion rules sparse. It is the geometry of the reversion rules, of the constant rebalanced portfolio and of buy-and-hold, and the prototype's answer for the Newton step.

# Examples

```jldoctest
julia> EuclideanProjection()
EuclideanProjection()
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`EntropicProjection`](@ref)
  - [`project`](@ref)
"""
struct EuclideanProjection <: AbstractProjectionGeometry end
"""
$(DocStringExtensions.TYPEDEF)

The entropic Projection Geometry: the raw step is projected onto the Allocation Set in relative entropy, ``\\min \\sum_i w_i \\log (w_i / q_i)``.

On the simplex this is plain normalisation, `q / Σq`, so exponentiated gradient's multiplicative update followed by normalisation is exactly this projection. It cannot zero a positive entry, and a zero entry stays zero, so a Start Allocation with a zero under it holds that zero for the run. It refuses a raw step with a negative entry, whose logarithm is undefined.

# Examples

```jldoctest
julia> EntropicProjection()
EntropicProjection()
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`EuclideanProjection`](@ref)
  - [`project`](@ref)
"""
struct EntropicProjection <: AbstractProjectionGeometry end
"""
$(DocStringExtensions.TYPEDEF)

The Gram Projection Geometry: the raw step is projected onto the Allocation Set in the norm of the rule's Gram matrix, ``\\min (\\boldsymbol{w} - \\boldsymbol{q})^\\intercal A (\\boldsymbol{w} - \\boldsymbol{q})``.

It is the geometry the online Newton step's logarithmic regret is proved in (Agarwal, Hazan, Kale and Schapire 2006), and it has no closed form on any set, the bare simplex included, so it carries its own solver and every projection in it is a programme. The matrix is the rule's: [`NewtonStep`](@ref) binds its carrier's ``A_t`` onto the geometry's `A` slot at every step through [`gram_geometry`](@ref), and a geometry with no matrix bound refuses to project. The Start Allocation is projected before any gradient is seen, where ``A_0 = I`` and the geometry is the Euclidean one, so [`projection_geometry`](@ref) answers [`EuclideanProjection`](@ref) for it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GramProjection(;
        slv::Slv_VecSlv,
        A::Option{<:AbstractMatrix} = nothing
    ) -> GramProjection

Keywords correspond to the struct's fields. `A` is bound by the rule, never by the caller.

## Validation

  - If `A` is given: `size(A, 1) == size(A, 2)`. A `DimensionMismatch` is thrown otherwise.

# Examples

```jldoctest
julia> GramProjection(; slv = Solver(; solver = nothing))
GramProjection
  slv ┼ Solver
      │          name ┼ String: \"\"
      │        solver ┼ nothing
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
    A ┴ nothing
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`EuclideanProjection`](@ref)
  - [`NewtonStep`](@ref)
  - [`project`](@ref)
  - [`gram_geometry`](@ref)

# References

  - $(ref_dict[:agarwal2006])
"""
struct GramProjection{T1 <: Slv_VecSlv, T2 <: Option{<:AbstractMatrix}} <:
       AbstractProjectionGeometry
    """
    $(field_dict[:slv])
    """
    slv::T1
    """
    The Gram matrix the projection is taken in the norm of, bound by the rule at each step, or `nothing` before the rule binds one.
    """
    A::T2
    function GramProjection(slv::Slv_VecSlv, A::Option{<:AbstractMatrix})
        if !isnothing(A)
            @argcheck(size(A, 1) == size(A, 2),
                      DimensionMismatch("the Gram matrix must be square, got $(size(A))"))
        end
        return new{typeof(slv), typeof(A)}(slv, A)
    end
end
function GramProjection(; slv::Slv_VecSlv,
                        A::Option{<:AbstractMatrix} = nothing)::GramProjection
    return GramProjection(slv, A)
end
"""
    gram_geometry(proj::EuclideanProjection, A::AbstractMatrix)
    gram_geometry(proj::GramProjection, A::AbstractMatrix)

The geometry a rule with a Gram matrix projects in at this step: a [`GramProjection`](@ref) with the rule's current matrix bound onto its `A` slot, and any other geometry unchanged.

# Related

  - [`GramProjection`](@ref)
  - [`NewtonStep`](@ref)
"""
function gram_geometry(proj::AbstractProjectionGeometry, ::AbstractMatrix)
    return proj
end
function gram_geometry(proj::GramProjection, A::AbstractMatrix)
    return GramProjection(; slv = proj.slv, A = A)
end
"""
$(DocStringExtensions.TYPEDEF)

The Allocation Set of weight bounds alone: `Σw = 1` and `lb ≤ w ≤ ub`, with no solver field, because every projection onto it is closed form.

The default, `BoundedAllocationSet()`, is the simplex. Every projection onto the set is closed form: under the simplex bounds the sort of Duchi and co-authors in the Euclidean geometry and plain normalisation in the entropic one, and under any other bound a scalar root — `w = clip(q − θ, lb, ub)` for the `θ` that restores the budget, or `w = clip(q / Z, lb, ub)` for the `Z` that does — found by bisection. A negative lower bound is admitted under the Euclidean geometry and refused under the entropic one, whose `log w` is undefined below zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BoundedAllocationSet(;
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:UniverseSets} = nothing
    ) -> BoundedAllocationSet

Keywords correspond to the struct's fields.

## Validation

  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `wb`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> BoundedAllocationSet()
BoundedAllocationSet
    wb ┼ WeightBounds
       │   lb ┼ Float64: 0.0
       │   ub ┴ Float64: 1.0
  sets ┴ nothing
```

# Related

  - [`AbstractAllocationSet`](@ref)
  - [`project`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`WeightBounds`](@ref)
"""
@propagatable @concrete struct BoundedAllocationSet <: AbstractAllocationSet
    """
    $(field_dict[:wb])
    """
    @vprop wb
    """
    $(field_dict[:sets])
    """
    @vprop sets
    function BoundedAllocationSet(wb::Option{<:WbE_Wb}, sets::Option{<:UniverseSets})
        if isa(wb, WeightBoundsEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb is a WeightBoundsEstimator"))
        end
        return new{typeof(wb), typeof(sets)}(wb, sets)
    end
end
function BoundedAllocationSet(; wb::Option{<:WbE_Wb} = WeightBounds(),
                              sets::Option{<:UniverseSets} = nothing)::BoundedAllocationSet
    return BoundedAllocationSet(wb, sets)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves the weight bounds of a [`BoundedAllocationSet`](@ref) to vectors over `N` assets.

# Related

  - [`BoundedAllocationSet`](@ref)
  - [`weight_bounds_constraints`](@ref)
"""
function resolve_allocation_set(set::BoundedAllocationSet, N::Integer, strict::Bool,
                                datatype::DataType)
    wb = weight_bounds_constraints(set.wb, set.sets; N = N, strict = strict,
                                   datatype = datatype)
    return BoundedAllocationSet(; wb = wb, sets = set.sets)
end
function rows_needed(::BoundedAllocationSet)
    return 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Whether a resolved bound is the simplex's `(0, 1)` at every asset, where the projection is a sort rather than a root.

# Related

  - [`project`](@ref)
  - [`BoundedAllocationSet`](@ref)
"""
function simplex_bounds(wb::WeightBounds)::Bool
    return all(iszero, wb.lb) && all(isone, wb.ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a resolved bound no allocation summing to one satisfies: `Σ lb ≤ 1 ≤ Σ ub`.

# Related

  - [`project`](@ref)
  - [`BoundedAllocationSet`](@ref)
"""
function assert_feasible_bounds(wb::WeightBounds)::Nothing
    @argcheck(sum(wb.lb) <= one(eltype(wb.lb)) <= sum(wb.ub),
              ArgumentError("no allocation summing to one lies between the resolved bounds: `Σ lb = $(sum(wb.lb))` and `Σ ub = $(sum(wb.ub))`, and the budget is one."))
    return nothing
end
"""
    project(::EuclideanProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(::EntropicProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)

Projects a rule's raw step onto the Allocation Set in the rule's Projection Geometry.

The one dispatch of the Constrained Update. On the bounded set under the simplex bounds, the Euclidean arm is the sort of Duchi and co-authors ([`project_simplex`](@ref)) and the entropic arm is normalisation; under any other bound each arm is a scalar root, `clip(q − θ, lb, ub)` ([`bounded_quadratic_projection`](@ref) at unit weights) and `clip(q / Z, lb, ub)`, found by bisection through [`bounded_root`](@ref). The entropic arm refuses a raw step with a negative entry, a step with no positive entry at all, which no normalisation puts on the simplex, and a negative lower bound.

# Arguments

  - `proj`: The geometry.
  - `set`: The set, its bounds resolved.
  - `q`: The raw step.
  - `w`: The Price-Adjusted Allocation the step trades from, unread by the bounded set.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds. An `ArgumentError` is thrown otherwise.
  - Under [`EntropicProjection`](@ref), `all(>= 0, q)`, `sum(q) > 0` and `all(>= 0, lb)`. A `DomainError` is thrown otherwise.

# Returns

  - `w'::Vector`: The projected allocation, a new vector.

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`project_simplex`](@ref)
"""
function project(::EuclideanProjection, set::BoundedAllocationSet, q::AbstractVector,
                 ::AbstractVector)
    wb = set.wb
    if simplex_bounds(wb)
        return project_simplex(q)
    end
    return bounded_quadratic_projection(q, wb)
end
function project(::EntropicProjection, set::BoundedAllocationSet, q::AbstractVector,
                 ::AbstractVector)
    @argcheck(all(x -> x >= zero(x), q),
              DomainError(q,
                          "the entropic projection is defined on non-negative raw steps alone: `log w` is undefined below zero"))
    s = sum(q)
    @argcheck(s > zero(s),
              DomainError(q,
                          "the entropic projection needs a positive entry to normalise: a raw step of zeros has no projection onto the simplex"))
    wb = set.wb
    if simplex_bounds(wb)
        return q ./ s
    end
    @argcheck(all(x -> x >= zero(x), wb.lb),
              DomainError(wb.lb,
                          "the entropic projection admits no negative lower bound: `log w` is undefined below zero"))
    assert_feasible_bounds(wb)
    # `Σ clip(q / Z, lb, ub)` falls from `Σ ub` to `Σ lb` as `Z` grows, and the plain
    # normalisation `Z = s` is where it would sit unclipped; the root is bracketed by
    # widening from there.
    f = z -> sum(clamp.(q ./ z, wb.lb, wb.ub))
    lo = s
    hi = s
    while f(lo) < one(s)
        lo /= 2
    end
    while f(hi) > one(s)
        hi *= 2
    end
    return clamp.(q ./ bounded_root(f, lo, hi), wb.lb, wb.ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The projection of a raw step onto a bounded set in the norm of a positive diagonal `h`, ``\\min \\tfrac{1}{2} \\sum_i h_i (w_i - q_i)^2`` subject to the budget and the bounds: the scalar root ``w_i = \\mathrm{clip}(q_i - \\theta / h_i, lb_i, ub_i)`` at the ``\\theta`` that restores the budget, found by bisection through [`bounded_root`](@ref). At `h = 1`, the default, it is the Euclidean arm of [`project`](@ref) off the simplex bounds; with the gradient mass of [`AdaptiveSubgradient`](@ref) it is the [`DiagonalProjection`](@ref) on every bound.

# Arguments

  - `q`: The raw step.
  - `wb`: The resolved bounds.
  - `h`: The diagonal of the norm, positive.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds. An `ArgumentError` is thrown otherwise.

# Returns

  - `w'::Vector`: The projected allocation, a new vector.

# Related

  - [`project`](@ref)
  - [`bounded_root`](@ref)
  - [`EuclideanProjection`](@ref)
  - [`DiagonalProjection`](@ref)
"""
function bounded_quadratic_projection(q::AbstractVector, wb::WeightBounds,
                                      h::AbstractVector = fill(one(eltype(q)), length(q)))
    assert_feasible_bounds(wb)
    # `Σ clip(q − θ / h, lb, ub)` falls from `Σ ub` to `Σ lb` as `θ` runs from
    # `min h (q − ub)` to `max h (q − lb)`, so the budget's root is bracketed there.
    lo = minimum(h .* (q .- wb.ub))
    hi = maximum(h .* (q .- wb.lb))
    theta = bounded_root(t -> sum(clamp.(q .- t ./ h, wb.lb, wb.ub)), lo, hi)
    return clamp.(q .- theta ./ h, wb.lb, wb.ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The root of a non-increasing budget function `f` on `[lo, hi]` where it crosses one, by bisection to the interval's floating-point resolution.

The projections onto a bounded set are each a clip of the raw step at one scalar, and the budget is monotone in that scalar, so a bisection between a value where the budget is at least one and one where it is at most one finds the scalar to machine precision in some sixty steps.

# Arguments

  - `f`: The budget as a function of the scalar, non-increasing.
  - `lo`, `hi`: A bracket with `f(lo) ≥ 1 ≥ f(hi)`.

# Returns

  - `x`: The scalar at which `f` crosses one.

# Related

  - [`project`](@ref)
  - [`BoundedAllocationSet`](@ref)
"""
function bounded_root(f, lo, hi)
    for _ in 1:200
        mid = (lo + hi) / 2
        if mid == lo || mid == hi
            break
        end
        if f(mid) >= one(mid)
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The Euclidean projection of a vector onto the probability simplex, by the sort of Duchi, Shalev-Shwartz, Singer and Chandra (2008).

# Mathematical definition

The projection solves ``\\min \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{v} \\rVert^2`` subject to ``\\boldsymbol{1}^\\intercal \\boldsymbol{w} = 1`` and ``\\boldsymbol{w} \\geq 0``. Its Lagrangian gives ``w_i = \\max(v_i - \\theta, 0)`` for the scalar ``\\theta`` that restores the budget: with ``\\boldsymbol{u}`` the entries of ``\\boldsymbol{v}`` in decreasing order,

```math
\\begin{align}
\\rho &= \\max \\left\\lbrace j : u_j - \\frac{1}{j} \\left( \\sum_{k \\leq j} u_k - 1 \\right) > 0 \\right\\rbrace\\,,\\quad
\\theta = \\frac{1}{\\rho} \\left( \\sum_{k \\leq \\rho} u_k - 1 \\right)\\,.
\\end{align}
```

Entries pushed below ``\\theta`` become exactly zero, which is the sparsity of every rule under this geometry.

# Arguments

  - `v`: The vector to project.

# Returns

  - `w::Vector`: The projection, a new vector summing to one.

# Related

  - [`project`](@ref)
  - [`EuclideanProjection`](@ref)

# References

  - $(ref_dict[:duchi2008])
"""
function project_simplex(v::AbstractVector)
    u = sort(v; rev = true)
    css = zero(eltype(u))
    rho = 1
    theta = zero(css) / one(rho)
    for j in eachindex(u)
        css += u[j]
        t = (css - one(css)) / j
        if u[j] - t > zero(t)
            rho = j
            theta = t
        end
    end
    return max.(v .- theta, zero(theta))
end
"""
    online_update!(alg::AbstractOnlinePortfolioSelectionAlgorithm, st, w::AbstractVector, x::AbstractVector, rows, set::AbstractAllocationSet)
    online_update!(alg::AbstractOnlinePortfolioSelectionAlgorithm, st, w::AbstractVector, x::AbstractVector, rows, set::AbstractAllocationSet, point::Option{<:AbstractVector})

The Online Update: one row of the online portfolio selection recursion, written once per rule.

Takes the rule, its private carrier `st`, the allocation `w` held during the period, the finite price relative `x` of that period, the rows the head holds through it, and the head's Allocation Set; answers `(st', w')`, the carrier and the allocation for the next period. The carrier is written in place where the rule can, `nothing` is a legal carrier, and `w'` is always a new vector, because the projection allocates one — so a `w` that is a view after [`port_opt_view`](@ref) is never written.

Every rule takes the same two halves — the unconstrained step to a raw vector, then [`project`](@ref) onto the set in the rule's geometry — and a rule that reads `w` continues from whatever it is handed, the Start Allocation on the first row.

The seven-argument form names the **Gradient Point**: an [`ExpertMixture`](@ref) under [`BlendPoint`](@ref) hands every expert its played blend as `point`, and a first-order rule evaluates its gradient there while stepping from its own iterate, the shared gradient the dynamic-regret mixtures are stated on. The generic method drops the point and takes the six-argument update, so a rule that has no gradient ignores it, and a `nothing` point — the mixture under [`OwnPoint`](@ref) — is the six-argument update for every rule; [`MirrorDescent`](@ref), [`OptimisticStep`](@ref) and [`AdaptiveSubgradient`](@ref) read a given point.

# Arguments

  - `alg`: The rule.
  - `st`: The rule's carrier, or `nothing`.
  - `w`: The allocation held during the period.
  - `x`: The price relative of the period, `1 .+ r`.
  - `rows`: The returns the head holds through the period, or `nothing`.
  - `set`: The Allocation Set, resolved.
  - `point`: The Gradient Point, the allocation a first-order rule evaluates its gradient at, or `nothing` for its own iterate.

# Returns

  - `(st', w')::Tuple`: The carrier and the allocation for the next period.

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`project`](@ref)
  - [`ExpertMixture`](@ref)
"""
function online_update! end
function online_update!(alg::AbstractOnlinePortfolioSelectionAlgorithm, st,
                        w::AbstractVector, x::AbstractVector, rows,
                        set::AbstractAllocationSet, ::Option{<:AbstractVector})
    return online_update!(alg, st, w, x, rows, set)
end
"""
    rule_state_seed(::AbstractOnlinePortfolioSelectionAlgorithm, ::AbstractVector)

The carrier a rule holds before its first update, `nothing` for a rule that carries nothing, which is the default.

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
"""
function rule_state_seed(::AbstractOnlinePortfolioSelectionAlgorithm, ::AbstractVector)
    return nothing
end
"""
    projection_geometry(alg::AbstractOnlinePortfolioSelectionAlgorithm)

The Projection Geometry the head projects the Start Allocation in: the rule's `proj` slot.

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
function projection_geometry(alg::AbstractOnlinePortfolioSelectionAlgorithm)
    return alg.proj
end
"""
    rows_needed(alg::AbstractOnlinePortfolioSelectionAlgorithm)

The number of rows a rule reads at a step. A rule with no method reads none.

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
"""
function rows_needed(::AbstractOnlinePortfolioSelectionAlgorithm)
    return 0
end
"""
    rows_needed_max(a, b)

The larger of two row needs, `nothing` being unbounded.

# Related

  - [`rows_needed`](@ref)
"""
function rows_needed_max(::Nothing, ::Any)
    return nothing
end
function rows_needed_max(::Integer, ::Nothing)
    return nothing
end
function rows_needed_max(a::Integer, b::Integer)
    return max(a, b)
end
"""
    rule_state_view(st::Nothing, i, args...)
    rule_state_view(st::AbstractPartialFitState, i, args...)
    rule_state_view(st, i, args...)

Slices a rule's carrier to the selected assets, forwarding a typed carrier to its own [`port_opt_view`](@ref) and refusing one that has none.

# Related

  - [`OnlinePortfolioSelectionState`](@ref)
  - [`port_opt_view`](@ref)
"""
function rule_state_view(::Nothing, ::Any, args...)
    return nothing
end
function rule_state_view(st::AbstractPartialFitState, i, args...)
    return port_opt_view(st, i, args...)
end
function rule_state_view(st, ::Any, args...)
    return throw(ArgumentError("a rule carrier of type `$(typeof(st))` has no `port_opt_view`, so the head's state cannot be sliced by asset: a carrier with a per-asset axis is an `AbstractPartialFitState` that writes its own view, and one with none is `nothing`."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The Price-Adjusted Allocation: the book the fund holds at the end of a period before it trades, `w .* x / ⟨w, x⟩`, one row of the self-financing Weight Drift at budget one.

It is the reference of the family's trade — the update of [`BuyAndHold`](@ref) *is* this vector, and a turnover ceiling on a later set kind bounds the distance from it — and it is handed to every projection as its `w` argument.

# Arguments

  - `w`: The allocation held during the period.
  - `x`: The price relative of the period.

# Returns

  - `ŵ::Vector`: The price-adjusted allocation, a new vector.

# Related

  - [`online_update!`](@ref)
  - [`BuyAndHold`](@ref)
  - [`SelfFinancingDrift`](@ref)
"""
function price_adjusted_allocation(w::AbstractVector, x::AbstractVector)
    return w .* x ./ LinearAlgebra.dot(w, x)
end
"""
$(DocStringExtensions.TYPEDEF)

The record of a Held Step: a projection programme of one row that did not solve, so the step traded nothing.

A projection onto a [`ProgrammeAllocationSet`](@ref), or in a [`GramProjection`](@ref), is a programme, and a programme can fail — infeasible on the day a turnover ceiling and a cap cannot both hold, timed out, a MIP at its limit, a covariance cone with no finite data on its first rows. The step then answers the Price-Adjusted Allocation it was handed, the book the fund already holds, so the fund trades nothing; the rule's carrier still absorbs the row. The head warns once with the row's timestamp, and its Recursion Read-out carries the record of the last folded row inside an [`OptimisationSuccess`](@ref), so a fallback chain never runs on a hold. A step never throws on a failed solve, and never falls back to a weaker set: a constraint dropped on the day it binds is not a constraint.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`project`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
  - [`JuMPResult`](@ref)
"""
struct HeldStep{T1, T2 <: AbstractString, T3}
    """
    The timestamp of the row whose projection was held, or its index in the fold when the carrier has no timestamps, or `nothing` outside a fold.
    """
    ts::T1
    """
    What was held: which projection of the row, and why.
    """
    reason::T2
    """
    The solver trials of the failed programme, a [`JuMPResult`](@ref)'s `trials`, or `nothing` when no programme was built.
    """
    trials::T3
end
"""
$(DocStringExtensions.TYPEDEF)

The step a projection runs inside: the rows the head holds through the period, the pinned asset names, the row's timestamp, and the Held Steps recorded so far.

The Online Update's projection is `project(proj, set, q, w)`, four arguments and no more, so a programme set's covariance cone and tracking error — which read the head's rows — and the Held Step's record — which the head must see — travel outside the signature, on the task-scoped [`PROJECTION_STEP`](@ref). The head opens one step per row through [`with_projection_step`](@ref), around the whole Online Update, so every projection of that row — a mixture's experts' and its own blend's — reads one set of rows and writes one log. A rule that re-solves an optimisation estimator on the rows it selects, [`FollowTheLeader`](@ref), reads the names here too, so the carrier it hands the estimator is named as the head's was. Outside a step, a projection reads no rows and reports a hold as a warning.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`PROJECTION_STEP`](@ref)
  - [`with_projection_step`](@ref)
  - [`HeldStep`](@ref)
  - [`project`](@ref)
"""
struct ProjectionStep{T1, T2, T3}
    """
    The rows of returns the head holds through the period, `observations × assets`, or `nothing`.
    """
    rows::T1
    """
    The pinned asset names, or `nothing`.
    """
    nx::T3
    """
    The row's timestamp, or its index in the fold.
    """
    ts::T2
    """
    The Held Steps recorded during this row, in the order their programmes failed.
    """
    held::Vector{HeldStep}
end
"""
    const PROJECTION_STEP

The task-scoped [`ProjectionStep`](@ref) a projection reads, `nothing` outside a step.

# Related

  - [`ProjectionStep`](@ref)
  - [`with_projection_step`](@ref)
"""
const PROJECTION_STEP = ScopedValue{Union{Nothing, ProjectionStep}}(nothing)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs `f()` inside a [`ProjectionStep`](@ref) over `rows` at `ts`, the names `nx` pinned, and answers `(f(), held)`: the update's result and the Held Steps its projections recorded, an empty vector when every programme solved.

# Related

  - [`PROJECTION_STEP`](@ref)
  - [`ProjectionStep`](@ref)
  - [`fold_online_selection`](@ref)
"""
function with_projection_step(f, rows, ts; nx = nothing)
    step = ProjectionStep(rows, nx, ts, HeldStep[])
    out = Base.ScopedValues.with(f, PROJECTION_STEP => step)
    return out, step.held
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Records a Held Step on the current [`ProjectionStep`](@ref), or warns at once when the projection runs outside one.

# Related

  - [`HeldStep`](@ref)
  - [`with_projection_step`](@ref)
"""
function record_held_step!(reason::AbstractString, trials)
    step = PROJECTION_STEP[]
    if isnothing(step)
        @warn("Held Step outside an Online Update: $reason")
        return nothing
    end
    push!(step.held, HeldStep(step.ts, reason, trials))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The rows the current [`ProjectionStep`](@ref) holds, or `nothing` outside a step.

# Related

  - [`ProjectionStep`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
"""
function projection_step_rows()
    step = PROJECTION_STEP[]
    return isnothing(step) ? nothing : step.rows
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The asset names the current [`ProjectionStep`](@ref) pins, or `nothing` outside a step or under a carrier with none.

# Related

  - [`ProjectionStep`](@ref)
  - [`FollowTheLeader`](@ref)
"""
function projection_step_names()
    step = PROJECTION_STEP[]
    return isnothing(step) ? nothing : step.nx
end
"""
    assert_rule_admits_set(alg::AbstractOnlinePortfolioSelectionAlgorithm, set::AbstractAllocationSet)

Refuses, at the head's construction, a rule and an Allocation Set that cannot meet: the default admits every pair, and a rule that holds an optimisation estimator with no model for a programme set refuses that set by name ([`FollowTheLeader`](@ref)). A rule over other rules forwards the check to each of them.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`assert_geometry_admits_set`](@ref)
  - [`FollowTheLeader`](@ref)
"""
function assert_rule_admits_set(::AbstractOnlinePortfolioSelectionAlgorithm,
                                ::AbstractAllocationSet)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

The Partial Fit State of an [`OnlinePortfolioSelection`](@ref) head: a Rule State beside the rows held once, the Fold Context pinned by the first step, and every timestamp folded.

The pair `(st, w)` is the Rule State, the unit the family recurses over. The rows any rule of the tree reads are held once, on this state, as a [`SampleBufferState`](@ref) of **returns** — the Returns Result's rows verbatim, a non-finite cell filled with zero before the push — capped by the rule tree's [`rows_needed`](@ref), and `nothing` for a tree that reads none. The timestamps are never capped, so [`Resume`](@ref) works for a carrier-free rule. The last folded row's active mask is the Investable Mask the read-out reduces on, `nothing` under a static panel; the recursion itself keeps the full `w` and never carries a forced zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OnlinePortfolioSelectionState(;
        n::Integer = 0,
        w::AbstractVector,
        st = nothing,
        X::Option{<:SampleBufferState} = nothing,
        nx::Option{<:VecStr} = nothing,
        pnl::Option{<:AssetPanel} = nothing,
        amsk::Option{<:AbstractVector{<:Bool}} = nothing,
        ts::Option{<:AbstractVector} = nothing,
        hold::Option{<:HeldStep} = nothing
    ) -> OnlinePortfolioSelectionState

Keywords correspond to the struct's fields.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `length(nx) == length(w)` when `nx` is given, and `length(amsk) == length(w)` when `amsk` is given. A `DimensionMismatch` is thrown otherwise.
  - `pnl` is static when it is not `nothing`. An `ArgumentError` is thrown otherwise: a time-varying panel's masks ride with the rows.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets: `w` is sliced and renormalised, `st` is forwarded to the carrier's own view through [`rule_state_view`](@ref), `X`, `nx`, `pnl` and `amsk` are sliced, and `n`, `ts` and `hold` are copied.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`SampleBufferState`](@ref)
  - [`rule_state_view`](@ref)
  - [`merge_states`](@ref)
"""
@concrete struct OnlinePortfolioSelectionState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The allocation held during the current period, over the full pinned universe, summing to one.
    """
    w
    """
    The rule's private carrier, or `nothing`.
    """
    st
    """
    The rows of returns any rule of the tree reads, a [`SampleBufferState`](@ref) capped by [`rows_needed`](@ref), or `nothing` when the tree reads none.
    """
    X
    """
    Names of the asset columns, pinned by the first step.
    """
    nx
    """
    The static [`AssetPanel`](@ref) of the universe, pinned by the first step, or `nothing`. A time-varying panel is never held here.
    """
    pnl
    """
    The active mask of the last folded row under a time-varying panel, the Investable Mask of the read-out, or `nothing` under a static one.
    """
    amsk
    """
    Timestamps of the observations folded, in order and uncapped, or `nothing` when the carrier holds none.
    """
    ts
    """
    The [`HeldStep`](@ref) record of the last folded row when a projection of that row was held, or `nothing` when every projection of it solved; the Recursion Read-out's retcode carries it.
    """
    hold
end
function OnlinePortfolioSelectionState(; n::Integer = 0, w::AbstractVector, st = nothing,
                                       X::Option{<:SampleBufferState} = nothing,
                                       nx::Option{<:VecStr} = nothing,
                                       pnl::Option{<:AssetPanel} = nothing,
                                       amsk::Option{<:AbstractVector{<:Bool}} = nothing,
                                       ts::Option{<:AbstractVector} = nothing,
                                       hold::Option{<:HeldStep} = nothing)::OnlinePortfolioSelectionState
    assert_nonneg(n, :n)
    if !isnothing(nx)
        @argcheck(length(nx) == length(w),
                  DimensionMismatch("the pinned names ($(length(nx))) and the allocation ($(length(w))) must have one length"))
    end
    if !isnothing(amsk)
        @argcheck(length(amsk) == length(w),
                  DimensionMismatch("the active mask ($(length(amsk))) and the allocation ($(length(w))) must have one length"))
    end
    if !isnothing(pnl)
        @argcheck(panel_is_static(pnl),
                  ArgumentError("an OnlinePortfolioSelectionState pins a static Asset Panel and never a time-varying one: the masks of a time-varying panel are per-observation, so the last row's rides on `amsk` and the rest with the rows."))
    end
    return OnlinePortfolioSelectionState(n, w, st, X, nx, pnl, amsk, ts, hold)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two online selection states, by name.

An Online Update is not a sufficient statistic for its block: even a carrier that is — a Gram sum, a prefix of rows — sits beside an allocation that is not, so two states folded on disjoint blocks describe no single run. The family's parallel route is the Causal Pass, which is `O(N)` a row.

# Related

  - [`OnlinePortfolioSelectionState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(::OnlinePortfolioSelectionState, ::OnlinePortfolioSelectionState)
    return throw(ArgumentError("an `OnlinePortfolioSelectionState` cannot be merged: an online update is order-dependent, so two states folded on disjoint blocks describe no single run. Run the Causal Pass over the concatenated rows instead, `optimise(opt, rd)`."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`OnlinePortfolioSelectionState`](@ref) so the copy shares no array with the original: `w`, `st`, `X` and `ts` are deep-copied.

# Related

  - [`OnlinePortfolioSelectionState`](@ref)
  - [`partial_fit`](@ref)
"""
function Base.copy(x::OnlinePortfolioSelectionState)
    return OnlinePortfolioSelectionState(; n = x.n, w = copy(x.w), st = copy_column(x.st),
                                         X = copy_column(x.X), nx = copy_column(x.nx),
                                         pnl = x.pnl, amsk = copy_column(x.amsk),
                                         ts = copy_column(x.ts), hold = x.hold)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices an [`OnlinePortfolioSelectionState`](@ref) to the selected assets: the state of the same observations over them.

`w` is sliced and renormalised, the carrier is forwarded to its own view through [`rule_state_view`](@ref), the rows buffer takes [`SampleBufferState`](@ref)'s view, the names, the panel and the mask are sliced, and the count, the timestamps and the hold record are copied. The view is a copy for the read-out; the recursion keeps the full `w`.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments, forwarded to the carrier's view.

# Returns

  - `state::OnlinePortfolioSelectionState`: The state over the selected assets.

# Related

  - [`OnlinePortfolioSelectionState`](@ref)
  - [`rule_state_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(x::OnlinePortfolioSelectionState, i, args...)
    return OnlinePortfolioSelectionState(; n = x.n, w = renormalised_view(x.w, i),
                                         st = rule_state_view(x.st, i, args...),
                                         X = if isnothing(x.X)
                                             nothing
                                         else
                                             port_opt_view(x.X, i, args...)
                                         end, nx = nothing_scalar_array_view(x.nx, i),
                                         pnl = if isnothing(x.pnl)
                                             nothing
                                         else
                                             port_opt_view(x.pnl, i)
                                         end, amsk = nothing_scalar_array_view(x.amsk, i),
                                         ts = copy_column(x.ts), hold = x.hold)
end
"""
    renormalised_view(w::Nothing, i)
    renormalised_view(w::AbstractVector, i)

Slices an allocation to the selected assets and renormalises it to sum to one, as a copy; passes an absent one through.

The mass the selection leaves out is spread over the kept assets in proportion, which is the reading every per-asset view of an allocation takes in the family: the read-out under a time-varying panel, and the view of a rule's own allocation. A selection that keeps none of the mass, as a view of a one-hot allocation over the other assets does, has no proportion to spread it in and answers the uniform allocation over the kept assets, the family's Start Allocation.

# Related

  - [`OnlinePortfolioSelectionState`](@ref)
  - [`port_opt_view`](@ref)
"""
function renormalised_view(::Nothing, ::Any)
    return nothing
end
function renormalised_view(w::AbstractVector, i)
    v = w[i]
    s = sum(v)
    return iszero(s) ? fill(one(s) / length(v), length(v)) : v ./ s
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Learning-Rate Schedules a first-order Online Selection Rule may hold on `eta` in place of a number.

A schedule is read at every update as a function of the period count and of the rule's carrier, so it may follow the count alone, read a running statistic of the run that the carrier keeps for it, or name a stage boundary at which the rule restarts. It owns no state: whatever it accumulates lives on the rule's carrier, seeded by the schedule and written by it once per row — after the step by default, so the rate of a period reads the past alone, or before the rate is read when the schedule declares that it chooses the rate from the period's own row.

# Interfaces

In order to implement a new schedule, subtype `AbstractLearningRateSchedule` and implement:

  - `learning_rate(sched::AbstractLearningRateSchedule, t::Integer, st) -> Real`: The rate of the update at period `t`, the `t`-th row the rule has seen; `st` is the rule's carrier, whose `s` field is the schedule's own statistic.
  - `restart(sched::AbstractLearningRateSchedule, t::Integer) -> Bool`: Whether the update at period `t` closes a stage, at which the rule answers the Start Allocation for period `t + 1` and puts its carrier back at its seed; `false` by default.
  - `schedule_state_seed(sched::AbstractLearningRateSchedule, w::AbstractVector)`: The statistic the carrier holds for the schedule before the first row, or `nothing`, the default.
  - `schedule_update!(sched::AbstractLearningRateSchedule, s, w::AbstractVector, x::AbstractVector)`: Writes the row into the statistic `s`, from the allocation `w` played during the period and its price relative `x`; the identity by default.
  - `reads_period_row(sched::AbstractLearningRateSchedule) -> Bool`: Whether the rate of period `t` is chosen from the row of period `t`, so the statistic is written before the rate is read rather than after the step; `false` by default. A rule whose update needs the rate of the *next* period as well, as the online form of [`ExpectationMaximisation`](@ref) does, refuses a schedule that answers `true`, because the next period's row is not there to read.
  - `mixing_share(sched::AbstractLearningRateSchedule, t::Integer, alpha::Real) -> Real`: The uniform-mix share of the update at period `t`, the rule's own `alpha` by default; a schedule that sets the share from a stage length answers its own.

A statistic whose entries lie over the assets, as the expert allocations of a replaying schedule do, also implements [`schedule_state_view`](@ref), which slices it to the selected assets with the rule's carrier; a statistic over the run alone is copied as it stands.

A number on `eta` is the constant schedule: every verb answers the number, `false`, `nothing`, the identity, `false` and `alpha`.

# Related

  - [`MirrorDescent`](@ref)
  - [`ExpectationMaximisation`](@ref)
  - [`InverseSquareRootRate`](@ref)
  - [`DoublingTrickRate`](@ref)
  - [`SelfConfidentRate`](@ref)
  - [`WindowedBestRate`](@ref)
"""
abstract type AbstractLearningRateSchedule <: AbstractAlgorithm end
"""
    learning_rate(eta::Real, t::Integer, st)
    learning_rate(sched::AbstractLearningRateSchedule, t::Integer, st)

The step size a first-order rule takes at period `t`: the number itself when `eta` is one, and the schedule's answer otherwise, read from the period count and the rule's carrier.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`MirrorDescent`](@ref)
"""
function learning_rate(eta::Real, ::Integer, ::Any)
    return eta
end
"""
    restart(eta::Real, t::Integer)
    restart(sched::AbstractLearningRateSchedule, t::Integer)

Whether the update at period `t` closes a stage of the schedule, so the rule answers the Start Allocation for period `t + 1` and puts its carrier back at its seed. A number and every schedule without stages answer `false`.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`DoublingTrickRate`](@ref)
"""
function restart(::Union{<:Real, <:AbstractLearningRateSchedule}, ::Integer)
    return false
end
"""
    schedule_state_seed(eta::Real, w::AbstractVector)
    schedule_state_seed(sched::AbstractLearningRateSchedule, w::AbstractVector)

The statistic a rule's carrier holds for its schedule before the first row: `nothing` for a number and for a schedule that reads none.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`SelfConfidentRate`](@ref)
"""
function schedule_state_seed(::Union{<:Real, <:AbstractLearningRateSchedule},
                             ::AbstractVector)
    return nothing
end
"""
    schedule_update!(eta::Real, s, w::AbstractVector, x::AbstractVector)
    schedule_update!(sched::AbstractLearningRateSchedule, s, w::AbstractVector, x::AbstractVector)

Writes one row into the schedule's statistic `s`, from the allocation `w` played during the period and its price relative `x`, and answers the statistic; the identity for a number and for a schedule that reads none.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`SelfConfidentRate`](@ref)
"""
function schedule_update!(::Union{<:Real, <:AbstractLearningRateSchedule}, s,
                          ::AbstractVector, ::AbstractVector)
    return s
end
"""
    mixing_share(eta::Real, t::Integer, alpha::Real)
    mixing_share(sched::AbstractLearningRateSchedule, t::Integer, alpha::Real)

The uniform-mix share of the update at period `t`: the rule's own `alpha` under a number and under every schedule that does not set it, and the stage's share under [`DoublingTrickRate`](@ref), whose share is a function of the stage length and takes precedence.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`MirrorDescent`](@ref)
"""
function mixing_share(::Union{<:Real, <:AbstractLearningRateSchedule}, ::Integer,
                      alpha::Real)
    return alpha
end
"""
    reads_period_row(eta::Real)
    reads_period_row(sched::AbstractLearningRateSchedule)

Whether the schedule chooses the rate of period `t` from the row of period `t`: `true` for a schedule that replays the period before it answers, as [`WindowedBestRate`](@ref) does, and `false` for a number and for every schedule that reads the past alone. A rule writes the statistic before it reads the rate when the answer is `true`, and after its step otherwise; [`statistic_before_rate`](@ref) and [`statistic_after_step`](@ref) are the two calls.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`schedule_update!`](@ref)
"""
function reads_period_row(::Union{<:Real, <:AbstractLearningRateSchedule})
    return false
end
"""
    statistic_before_rate(eta, s, w::AbstractVector, x::AbstractVector)

The first of the two points at which a rule writes its schedule's row: the statistic the rate of the period is read from. The row enters here when [`reads_period_row`](@ref) answers `true`, and the call answers `s` unchanged otherwise; [`statistic_after_step`](@ref) is the other point, so a rule makes both calls and the schedule decides which one writes.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`schedule_update!`](@ref)
  - [`MirrorDescent`](@ref)
"""
function statistic_before_rate(eta, s, w::AbstractVector, x::AbstractVector)
    return reads_period_row(eta) ? schedule_update!(eta, s, w, x) : s
end
"""
    statistic_after_step(eta, s, w::AbstractVector, x::AbstractVector)

The second of the two points at which a rule writes its schedule's row: the statistic the carrier keeps after the step. The row enters here when [`reads_period_row`](@ref) answers `false`, the default, and the call answers `s` unchanged otherwise; [`statistic_before_rate`](@ref) is the other point.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`schedule_update!`](@ref)
  - [`MirrorDescent`](@ref)
"""
function statistic_after_step(eta, s, w::AbstractVector, x::AbstractVector)
    return reads_period_row(eta) ? s : schedule_update!(eta, s, w, x)
end
"""
    schedule_state_view(s, i)

Slices a schedule's statistic to the selected assets when a rule's carrier is viewed: a copy of the statistic as it stands by default, because the statistics of the count-reading and run-reading schedules lie over the run and not over the assets; a statistic that holds allocations, as [`WindowedBestRate`](@ref)'s does, slices and renormalises each of them through [`renormalised_view`](@ref).

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`renormalised_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function schedule_state_view(s, ::Any)
    return copy_column(s)
end
export EuclideanProjection, EntropicProjection, GramProjection, BoundedAllocationSet
public AbstractOnlinePortfolioSelectionAlgorithm, AbstractProjectionGeometry,
       AbstractAllocationSet, online_update!, rule_state_seed, projection_geometry, project,
       resolve_allocation_set, AbstractLearningRateSchedule, learning_rate, restart,
       schedule_state_seed, schedule_update!, mixing_share, reads_period_row,
       schedule_state_view
