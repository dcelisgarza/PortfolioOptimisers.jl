"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Online Selection Rules that an [`OnlinePortfolioSelection`](@ref) head holds in its `alg` field.

The rule is the only part of the online portfolio selection family that changes from one paper to the next. It is a struct of the paper's parameters, a private carrier for what the update accumulates, and one update method. The head owns the parts that every rule shares: the Allocation Set, the Causal Pass, the Block Step, the Recursion Read-out and the checks that refuse a bad input. A rule projects its raw step onto the head's Allocation Set in its own Projection Geometry, which it holds in its `proj` field. The type bound of that field names the geometries that the rule's theorem covers.

# Interfaces

To implement a new rule, subtype `AbstractOnlinePortfolioSelectionAlgorithm` with the paper's parameters and a `proj` field, and implement the following methods:

  - `online_update!(alg::AbstractOnlinePortfolioSelectionAlgorithm, st, w::AbstractVector, x::AbstractVector, rows, set::AbstractAllocationSet) -> Tuple`: The Online Update. It reads the rule's carrier `st`, the allocation `w` held during the period, the finite price relative `x` of the period and the rows carrier that the head holds through the period. It returns `(st', w')`, the carrier and the allocation for the next period. It may write `st` in place, and it always returns a new vector for `w'`. A first-order rule also implements the seven-argument form, whose last argument is the Gradient Point that an [`ExpertMixture`](@ref) gives it. The generic method of that form drops the point.
  - `rule_state_seed(alg::AbstractOnlinePortfolioSelectionAlgorithm, w::AbstractVector)`: The carrier before the first update, or `nothing` for a rule that carries nothing. `w` is the Start Allocation, and the carrier takes its length and its element type. The head calls the three-argument form, `rule_state_seed(alg, w, set)`, with its Allocation Set, and the default method of that form drops `set`. A rule that holds allocations of its own over the assets, as the experts of an [`ExpertMixture`](@ref) are, implements the three-argument form and projects them onto `set`.
  - `rows_needed(alg::AbstractOnlinePortfolioSelectionAlgorithm) -> Union{Nothing, Integer}`: The number of rows that the rule reads at a step. It is `0` for a rule that reads none, and `nothing` for a rule that reads every row folded so far.
  - `projection_geometry(alg::AbstractOnlinePortfolioSelectionAlgorithm) -> AbstractProjectionGeometry`: The geometry in which the head projects the Start Allocation. It is the rule's `proj` field by default.

A rule whose carrier has a per-asset axis also implements `port_opt_view(st, i, args...)` on the type of the carrier, so that a view of the head slices the carrier. It also implements `Base.copy(st)`, so that the copy of the state that [`partial_fit`](@ref) makes shares no array with the original.

## Arguments

  - `alg`: The concrete rule.
  - `st`: The rule's private carrier, or `nothing`.
  - `w`: The allocation held during the period. It covers the full pinned universe and sums to one.
  - `x`: The price relative of the period, [`price_relative`](@ref) of the row. It is `1 .+ r`, and one at a gap, so it is finite at every asset.
  - `rows`: The rows carrier that the head holds through the period, or `nothing` when the rule reads no rows. It is a [`ReturnsResult`](@ref) whose `X` holds the rows of returns as they arrived, with `NaN` where there was no return, under the pinned names and the Asset Panel of the buffer. A statistic over it, such as the mean of a forecaster, a prior or a re-solve, reads it as the batch verb reads a carrier, and reduces to its own Coverage Universe. A kernel over the price relatives reads a gap as one through [`price_relative`](@ref).
  - `set`: The head's Allocation Set. The rule projects onto it through [`project`](@ref).

## Returns

  - `(st', w')::Tuple`: The carrier and the allocation for the next period.

# Examples

A rule that projects one fixed raw step at every period. Under a cap of `0.5`, the Euclidean projection moves the excess of the first asset to the other two assets in equal parts.

```jldoctest
julia> struct FixedTarget{T1, T2} <: PortfolioOptimisers.AbstractOnlinePortfolioSelectionAlgorithm
           q::T1
           proj::T2
       end

julia> function PortfolioOptimisers.online_update!(alg::FixedTarget, st, w::AbstractVector,
                                                   x::AbstractVector, rows,
                                                   set::PortfolioOptimisers.AbstractAllocationSet)
           wh = PortfolioOptimisers.price_adjusted_allocation(w, x)
           return st, PortfolioOptimisers.project(alg.proj, set, alg.q, wh)
       end

julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\"], X = [0.1 -0.1 0.0; 0.0 0.2 -0.05]);

julia> opt = OnlinePortfolioSelection(; alg = FixedTarget([0.8, 0.1, 0.1], EuclideanProjection()),
                                      set = BoundedAllocationSet(;
                                                                 wb = WeightBounds(; lb = 0,
                                                                                   ub = 0.5)));

julia> optimise(opt, rd).w
3-element Vector{Float64}:
 0.5
 0.25
 0.25
```

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

Abstract supertype for the Projection Geometries in which an Online Selection Rule projects its raw step back onto the Allocation Set.

The geometry belongs to the theorem of the rule. The regret bound of exponentiated gradient is a relative-entropy argument, and its multiplicative update followed by normalisation is the exact Kullback–Leibler projection onto the simplex. The reversion rules are Euclidean, and their sparsity comes from the entries that the Euclidean projection sets to zero. The type bound of the `proj` field of every rule names the geometries that its theorem covers.

# Mathematical definition

A geometry is a potential ``\\Psi``. It projects the raw step onto the Allocation Set in the Bregman divergence of that potential.

```math
\\begin{align}
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q})\\,, \\\\
D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q}) &= \\Psi(\\boldsymbol{w}) - \\Psi(\\boldsymbol{q}) - \\nabla \\Psi(\\boldsymbol{q})^\\intercal (\\boldsymbol{w} - \\boldsymbol{q})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_plus_proj])
  - $(math_dict[:W_aset])
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:q_raw])

# Interfaces

To implement a new geometry, subtype `AbstractProjectionGeometry` and implement the following method for every Allocation Set that it projects onto:

  - `project(proj::AbstractProjectionGeometry, set::AbstractAllocationSet, q::AbstractVector, w::AbstractVector) -> AbstractVector`: The projection of the raw step `q` onto `set` in this geometry, as a new vector.

## Arguments

  - `proj`: The geometry.
  - `set`: The Allocation Set, with its bounds resolved to vectors.
  - `q`: The raw step of the rule, before the projection.
  - `w`: The Price-Adjusted Allocation from which the step trades. A turnover ceiling on a [`ProgrammeAllocationSet`](@ref) reads it as its reference, and a Held Step returns it. The bounded set reads it only on a Held Step.

## Returns

  - `w'::AbstractVector`: The projected allocation.

A geometry with no closed form on any set carries its own solver, as [`GramProjection`](@ref) does, and [`projection_solver`](@ref) reads that solver before the solver of the set.

# Examples

A geometry that projects in the norm of a fixed positive diagonal. The asset with the larger weight in the norm moves less.

```jldoctest
julia> struct FixedDiagonal{T} <: PortfolioOptimisers.AbstractProjectionGeometry
           h::T
       end

julia> function PortfolioOptimisers.project(proj::FixedDiagonal, set::BoundedAllocationSet,
                                            q::AbstractVector, w::AbstractVector)
           return PortfolioOptimisers.bounded_quadratic_projection(q, set.wb, proj.h)
       end

julia> set = PortfolioOptimisers.resolve_allocation_set(BoundedAllocationSet(), 2, false, Float64);

julia> PortfolioOptimisers.project(FixedDiagonal([1.0, 3.0]), set, [0.8, 0.8], [0.5, 0.5])
2-element Vector{Float64}:
 0.35
 0.65
```

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

Projects the raw step onto the Allocation Set in squared Euclidean distance.

On the simplex this projection is the sort of Duchi, Shalev-Shwartz, Singer and Chandra (2008). It sets every entry below its threshold to zero, and that is the source of the sparsity of the reversion rules. It is the default geometry of most rules, among them the reversion and tracking rules, the constant rebalanced portfolio, buy-and-hold and the Newton step.

# Mathematical definition

```math
\\begin{align}
\\Psi(\\boldsymbol{w}) &= \\tfrac{1}{2} \\lVert \\boldsymbol{w} \\rVert_2^2\\,, \\\\
\\mathrm{Proj}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert_2^2\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(q_i - \\theta,\\, l_i\\right),\\, u_i\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:Psi_pot])
  - $(math_dict[:Proj_W_euclid])
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])

On a [`BoundedAllocationSet`](@ref) the projection is ``\\boldsymbol{w}(\\theta)`` at the root. The budget ``\\sum_i w_i(\\theta)`` does not increase in ``\\theta``, and it is linear between the kinks at ``q_i - u_i`` and ``q_i - l_i``. The geometry admits a negative lower bound.

# Examples

```jldoctest
julia> EuclideanProjection()
EuclideanProjection()
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`EntropicProjection`](@ref)
  - [`project`](@ref)
  - [`project_simplex`](@ref)

# References

  - $(ref_dict[:duchi2008])
"""
struct EuclideanProjection <: AbstractProjectionGeometry end
"""
$(DocStringExtensions.TYPEDEF)

Projects the raw step onto the Allocation Set in relative entropy.

On the simplex this projection is normalisation, `q / sum(q)`. So the multiplicative update of exponentiated gradient, followed by normalisation, is this projection. Under the simplex bounds the projection keeps every positive entry positive and every zero entry at zero, so a zero in the Start Allocation stays zero for the whole run. The projection refuses a raw step with a negative entry, because the logarithm of a negative number is not defined.

# Mathematical definition

```math
\\begin{align}
\\Psi(\\boldsymbol{w}) &= \\sum_i w_i \\log w_i\\,, \\\\
\\mathrm{Proj}^{\\mathrm{KL}}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\sum_i w_i \\log \\frac{w_i}{q_i}\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(q_i e^{-\\theta},\\, l_i\\right),\\, u_i\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:Psi_pot])
  - ``\\mathrm{Proj}^{\\mathrm{KL}}_{\\mathcal{W}}``: Projection onto ``\\mathcal{W}`` in relative entropy.
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])

The Bregman divergence of ``\\Psi`` is ``\\sum_i w_i \\log (w_i / q_i) - \\sum_i w_i + \\sum_i q_i``, and on the budget its last two terms are constant. On a [`BoundedAllocationSet`](@ref) the projection is ``\\boldsymbol{w}(\\theta)`` at the root. Under the simplex bounds the root is ``e^{-\\theta} = 1 / \\sum_i q_i``. Under other bounds a floor ``l_i > 0`` lifts a zero entry to ``l_i``, and a cap ``u_i = 0`` sets a positive entry to zero. The budget ``\\sum_i w_i(\\theta)`` does not increase in ``\\theta``, and it is linear in ``e^{-\\theta}`` between the kinks at ``l_i / q_i`` and ``u_i / q_i``. The geometry refuses a negative lower bound.

# Examples

```jldoctest
julia> EntropicProjection()
EntropicProjection()
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`EuclideanProjection`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:helmbold1998])
"""
struct EntropicProjection <: AbstractProjectionGeometry end
"""
$(DocStringExtensions.TYPEDEF)

Projects the raw step onto the Allocation Set in the norm of the rule's Gram matrix.

The online Newton step of Agarwal, Hazan, Kale and Schapire (2006) proves its logarithmic regret in this geometry. The projection has no closed form on any set, the bare simplex included, so the geometry carries its own solver and every projection in it is a programme. The rule owns the matrix. [`NewtonStep`](@ref) binds the Gram matrix of its carrier to the `A` field of the geometry at every step, through [`gram_geometry`](@ref). A geometry with no matrix refuses to project. The head projects the Start Allocation before the rule sees a gradient. The Gram matrix is the identity there, and the geometry is the Euclidean one, so [`projection_geometry`](@ref) returns [`EuclideanProjection`](@ref) for the Newton step.

# Mathematical definition

```math
\\begin{align}
\\Psi(\\boldsymbol{w}) &= \\tfrac{1}{2} \\boldsymbol{w}^\\intercal \\mathbf{A} \\boldsymbol{w}\\,, \\\\
\\mathrm{Proj}^{\\mathbf{A}}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; (\\boldsymbol{w} - \\boldsymbol{q})^\\intercal \\mathbf{A} (\\boldsymbol{w} - \\boldsymbol{q})\\,.
\\end{align}
```

Where:

  - $(math_dict[:Psi_pot])
  - ``\\mathbf{A}``: Gram matrix that the rule binds at the step, symmetric and positive definite.
  - ``\\mathrm{Proj}^{\\mathbf{A}}_{\\mathcal{W}}``: Projection onto ``\\mathcal{W}`` in the norm of ``\\mathbf{A}``.
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])

At ``\\mathbf{A} = I`` the projection is the Euclidean one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GramProjection(;
        slv::Slv_VecSlv,
        A::Option{<:AbstractMatrix} = nothing
    ) -> GramProjection

Keywords correspond to the struct's fields. The rule binds `A` at each step, and a caller leaves it at `nothing`.

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
    The Gram matrix in whose norm the geometry projects. The rule binds it at each step, and it is `nothing` before the rule binds one.
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
    gram_geometry(proj::AbstractProjectionGeometry, A::AbstractMatrix)
    gram_geometry(proj::GramProjection, A::AbstractMatrix)

Returns the geometry in which a rule with a Gram matrix projects at this step.

For a [`GramProjection`](@ref) it is a new geometry with the same solver and with the rule's current matrix `A` in its `A` field. Every other geometry comes back unchanged.

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

Limits the allocation to the budget and to per-asset weight bounds, with closed-form projections and no solver.

The default, `BoundedAllocationSet()`, is the simplex. Under the simplex bounds the Euclidean projection is the sort of Duchi and co-authors, and the entropic projection is normalisation. Under any other bound each projection clips the raw step at one scalar, the root of the budget, which [`breakpoint_root`](@ref) finds exactly from the kinks of the budget. The barrier geometries find their root by bisection. The Euclidean geometry admits a negative lower bound. The entropic and barrier geometries refuse it, because their potentials are not defined below zero.

# Mathematical definition

```math
\\begin{align}
\\mathcal{W} &= \\left\\lbrace \\boldsymbol{w} \\in \\mathbb{R}^N : \\boldsymbol{1}^\\intercal \\boldsymbol{w} = 1,\\; l_i \\leq w_i \\leq u_i \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:W_aset])
  - $(math_dict[:N])
  - $(math_dict[:lu_i_aset])

The set is not empty when ``\\sum_i l_i \\leq 1 \\leq \\sum_i u_i``. At ``l_i = 0`` and ``u_i = 1`` it is the probability simplex.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BoundedAllocationSet(;
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:UniverseSets} = nothing
    ) -> BoundedAllocationSet

Keywords correspond to the struct's fields.

## Validation

  - If `wb` is a [`WeightBoundsEstimator`](@ref): `!isnothing(sets)`. An `IsNothingError` is thrown otherwise.

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

[`weight_bounds_constraints`](@ref) does the work, with the set's `sets` for a bound that names assets. The set that comes back keeps its `sets`.

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

Checks whether a resolved bound is `(0, 1)` at every asset. The bound is then the simplex, and the projection is a sort or a normalisation, not a root.

# Related

  - [`project`](@ref)
  - [`BoundedAllocationSet`](@ref)
"""
function simplex_bounds(wb::WeightBounds)::Bool
    return all(iszero, wb.lb) && all(isone, wb.ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a resolved bound that no allocation on the budget satisfies.

# Validation

  - `sum(wb.lb) <= 1 <= sum(wb.ub)`. An `ArgumentError` is thrown otherwise.

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
    project(proj::EuclideanProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::EntropicProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)

Projects a rule's raw step onto a [`BoundedAllocationSet`](@ref) in the Euclidean or the entropic geometry.

These are the closed-form arms of the Constrained Update. Under the simplex bounds the Euclidean arm is the sort of [`project_simplex`](@ref), and the entropic arm is normalisation. Under any other bound each arm finds the root of its budget exactly, from the kinks of the budget, through [`breakpoint_root`](@ref). A root whose allocation misses the budget in floating point gives the Held Step, through [`budget_or_held_step`](@ref).

# Mathematical definition

```math
\\begin{align}
w^{+}_i &= \\min\\left(\\max\\left(q_i - \\theta,\\, l_i\\right),\\, u_i\\right) \\quad \\text{(Euclidean)}\\,, \\\\
w^{+}_i &= \\min\\left(\\max\\left(q_i e^{-\\theta},\\, l_i\\right),\\, u_i\\right) \\quad \\text{(entropic)}\\,, \\\\
\\sum_i w^{+}_i &= 1\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_plus_proj])
  - $(math_dict[:q_raw])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])

Each form is the KKT solution of the projection in its geometry, and ``\\theta`` is the multiplier of the budget. In the entropic form a zero entry stays at ``l_i`` for every ``\\theta``. So the entropic arm needs ``q_i \\geq 0``, ``\\sum_i q_i > 0``, ``l_i \\geq 0``, and ``\\sum_{q_i > 0} u_i + \\sum_{q_i = 0} l_i \\geq 1``.

# Algorithm

Under [`EuclideanProjection`](@ref):

 1. When the bounds are the simplex bounds, refuse a non-finite `q` with [`assert_finite_raw_step`](@ref), and return [`project_simplex`](@ref) of `q`.
 2. Otherwise, find the allocation at the root with [`bounded_quadratic_projection`](@ref) at unit weights.
 3. Return that allocation, or the Held Step `w`, with [`budget_or_held_step`](@ref).

Under [`EntropicProjection`](@ref):

 1. Refuse a non-finite `q`, a `q` with a negative entry, and a `q` that sums to zero.
 2. When the bounds are the simplex bounds, return `q ./ sum(q)`.
 3. Refuse a negative lower bound, and bounds that no allocation on the budget satisfies, with [`assert_feasible_bounds`](@ref).
 4. Refuse a `q` that cannot reach the budget, because its zeros stay at their floors and its positive entries stop at their caps.
 5. Find the scale `t` at which `sum(clamp.(t .* q, lb, ub))` is one, with [`breakpoint_root`](@ref) over the kinks `lb ./ q` and `ub ./ q`. The search runs in `-t`, in which the budget does not increase.
 6. Return `clamp.(t .* q, lb, ub)`, or the Held Step `w`, with [`budget_or_held_step`](@ref).

# Arguments

  - `proj`: The geometry.
  - `set`: The set, with its bounds resolved.
  - `q`: The raw step.
  - `w`: The Price-Adjusted Allocation from which the step trades. A Held Step returns it.

# Validation

  - `all(isfinite, q)`. A `DomainError` is thrown otherwise.
  - `sum(lb) <= 1 <= sum(ub)` over the resolved bounds, off the simplex bounds. An `ArgumentError` is thrown otherwise.
  - Under [`EntropicProjection`](@ref): `all(>=(0), q)`, `sum(q) > 0`, `all(>=(0), lb)` and `sum(ifelse.(q .> 0, ub, lb)) >= 1`. A `DomainError` is thrown otherwise.

# Returns

  - `w'::Vector`: The projected allocation as a new vector, or a copy of `w` on a Held Step.

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`project_simplex`](@ref)
  - [`bounded_quadratic_projection`](@ref)
  - [`breakpoint_root`](@ref)
  - [`budget_or_held_step`](@ref)
"""
function project(proj::EuclideanProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    wb = set.wb
    if simplex_bounds(wb)
        assert_finite_raw_step(q)
        return project_simplex(q)
    end
    return budget_or_held_step(bounded_quadratic_projection(q, wb), w, proj, set)
end
function project(proj::EntropicProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    assert_finite_raw_step(q)
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
    # A zero entry sits at its floor at every scale, so the positive entries must reach the
    # budget at their caps.
    @argcheck(sum(ifelse.(q .> zero(s), wb.ub, wb.lb)) >= one(s),
              DomainError(q,
                          "the zeros of the raw step stay at their floors under the entropic projection, and the caps of the remaining assets do not reach the budget"))
    # `Σ clip(t q, lb, ub)` is piecewise linear and non-decreasing in the scale `t`, with its
    # kinks at `lb / q` and `ub / q`; the root is taken in `-t`, where it is non-increasing.
    t = -breakpoint_root(u -> sum(clamp.(-u .* q, wb.lb, wb.ub)),
                         [-wb.lb ./ q; -wb.ub ./ q])
    return budget_or_held_step(clamp.(t .* q, wb.lb, wb.ub), w, proj, set)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a raw step with a non-finite entry, because no projection of an infinite or undefined step lies on the budget.

# Validation

  - `all(isfinite, q)`. A `DomainError` is thrown otherwise.

# Related

  - [`project`](@ref)
  - [`bounded_quadratic_projection`](@ref)
"""
function assert_finite_raw_step(q::AbstractVector)::Nothing
    @argcheck(all(isfinite, q),
              DomainError(q,
                          "the projection is defined on finite raw steps alone: an infinite or `NaN` entry has no projection onto the budget"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the projected allocation `wn` when it meets the budget, and the Held Step otherwise.

A root found from the kinks is exact on every input whose kinks floating point can hold. The root of a barrier geometry can lose its budget to cancellation, when a base is far larger than the step it restores. This function catches that loss, so no allocation off the budget reaches the rule.

# Algorithm

 1. When `isapprox(sum(wn), 1)` holds, return `wn`. The default tolerance of `isapprox` is a relative error of `sqrt(eps)` of the element type, `1.49e-8` for `Float64`.
 2. Otherwise, record a Held Step on the current [`ProjectionStep`](@ref) with [`record_held_step!`](@ref). Its reason names the set, the geometry and the sum, and it has no solver trials.
 3. Return a copy of `w`, as a failed [`projection_programme`](@ref) does.

# Arguments

  - `wn`: The projected allocation.
  - `w`: The Price-Adjusted Allocation the step trades from.
  - `proj`: The geometry, named in the record.
  - `set`: The set, named in the record.

# Returns

  - `w'::Vector`: `wn`, or `w` copied on a Held Step.

# Related

  - [`project`](@ref)
  - [`HeldStep`](@ref)
"""
function budget_or_held_step(wn::AbstractVector, w::AbstractVector,
                             proj::AbstractProjectionGeometry, set::AbstractAllocationSet)
    if isapprox(sum(wn), one(eltype(wn)))
        return wn
    end
    record_held_step!("the projection onto the `$(nameof(typeof(set)))` in the `$(nameof(typeof(proj)))` geometry did not meet the budget in floating point, its allocation sums to $(sum(wn)), and the step trades nothing",
                      nothing)
    return copy(w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Projects a raw step onto a bounded set in the norm of a positive diagonal `h`, at the exact root of the budget.

At `h = 1`, the default, it is the Euclidean arm of [`project`](@ref) off the simplex bounds. With the gradient mass of [`AdaptiveSubgradient`](@ref) in `h`, it is the [`DiagonalProjection`](@ref) on every bound.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\tfrac{1}{2} \\sum_i h_i (w_i - q_i)^2\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(q_i - \\theta / h_i,\\, l_i\\right),\\, u_i\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_plus_proj])
  - $(math_dict[:W_aset])
  - $(math_dict[:h_diag_norm])
  - $(math_dict[:q_raw])
  - $(math_dict[:lu_i_aset])
  - $(math_dict[:theta_aset])

The projection is ``\\boldsymbol{w}(\\theta)`` at the root. The budget ``\\sum_i w_i(\\theta)`` does not increase in ``\\theta``, and it is linear between the kinks at ``h_i (q_i - u_i)`` and ``h_i (q_i - l_i)``, so the line through the two kinks that bracket the root gives the root exactly.

# Algorithm

 1. Refuse a non-finite `q` with [`assert_finite_raw_step`](@ref), and bounds that no allocation on the budget satisfies with [`assert_feasible_bounds`](@ref).
 2. Find `theta`, the root of `sum(clamp.(q .- theta ./ h, lb, ub))` at one, with [`breakpoint_root`](@ref) over the kinks `h .* (q .- ub)` and `h .* (q .- lb)`.
 3. Return `clamp.(q .- theta ./ h, lb, ub)`.

# Arguments

  - `q`: The raw step.
  - `wb`: The resolved bounds.
  - `h`: The diagonal of the norm, positive.

# Validation

  - `all(isfinite, q)`. A `DomainError` is thrown otherwise.
  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds. An `ArgumentError` is thrown otherwise.

# Returns

  - `w'::Vector`: The projected allocation, a new vector.

# Related

  - [`project`](@ref)
  - [`breakpoint_root`](@ref)
  - [`EuclideanProjection`](@ref)
  - [`DiagonalProjection`](@ref)
"""
function bounded_quadratic_projection(q::AbstractVector, wb::WeightBounds,
                                      h::AbstractVector = fill(one(eltype(q)), length(q)))
    assert_finite_raw_step(q)
    assert_feasible_bounds(wb)
    # `Σ clip(q − θ / h, lb, ub)` is piecewise linear and non-increasing in `θ`, with its
    # kinks at `h (q − ub)` and `h (q − lb)`.
    theta = breakpoint_root(t -> sum(clamp.(q .- t ./ h, wb.lb, wb.ub)),
                            [h .* (q .- wb.ub); h .* (q .- wb.lb)])
    return clamp.(q .- theta ./ h, wb.lb, wb.ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds, from its kinks `x`, the point at which a non-increasing, piecewise linear budget function `f` crosses one.

The Euclidean, diagonal and entropic projections onto a bounded set each clip the raw step at one scalar. The budget is linear in that scalar between the kinks at which an asset reaches a bound, so the line through the two kinks that bracket the root gives the root exactly. The search reads the kinks and not the width of a bracket, so a raw step whose entries span hundreds of binades is as exact as a step near the simplex. A kink at an infinite bound is not a kink, and the budget is linear past the last finite kink.

# Algorithm

 1. Drop the non-finite kinks and sort the rest, giving `x`. When no kink is left, use the one kink `0`.
 2. Search the indices of `x` by bisection for `lo`, the last kink at which `f` is at least one, and `hi = lo + 1`. The index `0` stands for the linear tail below the first kink, and the index `length(x) + 1` for the tail above the last kink.
 3. Take the segment `(a, b)` that holds the root with [`breakpoint_segment`](@ref).
 4. When `f(a) == f(b)`, return `a`. Otherwise, return the point at which the line through `(a, f(a))` and `(b, f(b))` crosses one.

# Arguments

  - `f`: The budget as a function of the scalar, non-increasing and linear between kinks.
  - `x`: The kinks. The non-finite ones are dropped.

# Returns

  - `r`: The scalar at which `f` crosses one.

# Related

  - [`project`](@ref)
  - [`bounded_quadratic_projection`](@ref)
"""
function breakpoint_root(f, x::AbstractVector)
    x = sort!(filter(isfinite, x))
    if isempty(x)
        push!(x, zero(eltype(x)))
    end
    # The last kink at which the budget is still at least one, `0` and `n + 1` standing for
    # the linear tails below the first kink and above the last.
    lo, hi = 0, length(x) + 1
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        f(x[mid]) >= one(eltype(x)) ? (lo = mid) : (hi = mid)
    end
    a, b = breakpoint_segment(x, lo, hi)
    fa, fb = f(a), f(b)
    return fa == fb ? a : a + (fa - one(fa)) * (b - a) / (fa - fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the segment of the budget of [`breakpoint_root`](@ref) that holds the root, from the kinks `x[lo]` and `x[hi]`.

An index past either end of `x` stands for a linear tail. The second point on a tail lies beyond the outer kink by the magnitude of that kink, and by at least one.

# Algorithm

 1. When `lo == 0`, return `(x[1] - max(1, abs(x[1])), x[1])`.
 2. When `hi > length(x)`, return `(x[end], x[end] + max(1, abs(x[end])))`.
 3. Otherwise, return `(x[lo], x[hi])`.

# Related

  - [`breakpoint_root`](@ref)
"""
function breakpoint_segment(x::AbstractVector, lo::Integer, hi::Integer)
    if lo == 0
        return x[1] - max(one(x[1]), abs(x[1])), x[1]
    elseif hi > length(x)
        return x[end], x[end] + max(one(x[end]), abs(x[end]))
    end
    return x[lo], x[hi]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finds the point on `[lo, hi]` at which a non-increasing budget function `f` crosses one, by bisection until the bracket collapses to adjacent floating-point values.

The projection of a barrier geometry onto a bounded set clips the raw step at one scalar. The budget is monotone in that scalar but not linear, so the function bisects it. [`bisection_cap`](@ref) of the type of the bracket caps the number of halvings. The cap is not less than the number of halvings in which a bracket over the whole range of the type collapses, so no bracket stops before it reaches its resolution. The midpoint is `lo / 2 + hi / 2`, which does not overflow on a bracket near `floatmax`.

# Algorithm

 1. Take the midpoint `mid = lo / 2 + hi / 2`. Stop when `mid` equals `lo` or `hi`, because the bracket then holds two adjacent values.
 2. Set `lo = mid` when `f(mid) >= 1`, and `hi = mid` otherwise.
 3. Repeat steps 1 and 2, at most [`bisection_cap`](@ref) times.
 4. Return `lo / 2 + hi / 2`.

# Arguments

  - `f`: The budget as a function of the scalar, non-increasing.
  - `lo`, `hi`: A bracket with `f(lo) ≥ 1 ≥ f(hi)`.

# Returns

  - `x`: The scalar at which `f` crosses one.

# Related

  - [`project`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`breakpoint_root`](@ref)
  - [`bisection_cap`](@ref)
"""
function bounded_root(f, lo, hi)
    # `(lo + hi) / 2` overflows to `Inf` when `lo + hi` passes `floatmax`; the halves do not,
    # and in the normal range the two midpoints are the same number.
    for _ in 1:bisection_cap(typeof(lo / 2 + hi / 2))
        mid = lo / 2 + hi / 2
        if mid == lo || mid == hi
            break
        end
        if f(mid) >= one(mid)
            lo = mid
        else
            hi = mid
        end
    end
    return lo / 2 + hi / 2
end
"""
    bisection_cap(::Type{T}) where {T <: AbstractFloat}
    bisection_cap(::Type)

Returns a number of halvings that is enough for any bisection bracket of type `T` to collapse to adjacent values of `T`.

# Mathematical definition

```math
\\begin{align}
c(T) &= e_{\\max}(T) - e_{\\min}(T) + 2 p(T)\\,.
\\end{align}
```

Where:

  - ``c(T)``: Cap on the halvings for the type ``T``.
  - ``e_{\\max}(T)``, ``e_{\\min}(T)``: Exponents of `floatmax(T)` and `floatmin(T)`. Their difference counts the normal binades of ``T``.
  - ``p(T)``: Precision of ``T`` in bits. Twice the precision covers the significand and the subnormal binades below `floatmin(T)`.

The cap is 2151 for `Float64`, 301 for `Float32` and 51 for `Float16`. The widest bracket, `[-floatmax(T), floatmax(T)]`, collapses at the smallest subnormal after 2099, 278 and 41 halvings, so the cap is an upper bound and not the exact count.

The cap only guards [`bounded_root`](@ref). On a floating-point type the collapse check stops the loop first. Any other type takes the cap of `Float64`, and the loop does not need a tight cap there. An AD dual and a quantity with units compare by their values, so the collapse check still stops the loop. A `Rational` bracket never collapses, so no cap comes from it, and a `Rational{Int}` bracket overflows its integers after about sixty halvings.

# Related

  - [`bounded_root`](@ref)
"""
function bisection_cap(::Type{T}) where {T <: AbstractFloat}
    return exponent(floatmax(T)) - exponent(floatmin(T)) + 2 * precision(T)
end
function bisection_cap(::Type)
    return bisection_cap(Float64)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Projects a vector onto the probability simplex in Euclidean distance, by the sort of Duchi, Shalev-Shwartz, Singer and Chandra (2008).

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\Delta_N}{\\arg\\min} \\; \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{v} \\rVert_2^2\\,, \\\\
w^{+}_i &= \\max(v_i - \\theta, 0)\\,, \\\\
\\rho &= \\max \\left\\lbrace j : u_j - \\frac{1}{j} \\left( \\sum_{k \\leq j} u_k - 1 \\right) > 0 \\right\\rbrace\\,, \\\\
\\theta &= \\frac{1}{\\rho} \\left( \\sum_{k \\leq \\rho} u_k - 1 \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_plus_proj])
  - $(math_dict[:Delta_N_simplex])
  - ``\\boldsymbol{v}``: Vector to project.
  - $(math_dict[:theta_aset])
  - ``\\rho``: Number of positive entries of the projection.
  - ``\\boldsymbol{u}``: Entries of ``\\boldsymbol{v}`` in decreasing order.

The entries of ``\\boldsymbol{v}`` below ``\\theta`` become exactly zero. That is the sparsity of every rule under the Euclidean geometry.

A non-negative ``\\boldsymbol{v}`` whose sum misses one by rounding alone is on the simplex already, and the projection renormalises it:

```math
\\begin{align}
\\boldsymbol{w}^{+} &= \\frac{\\boldsymbol{v}}{s} \\quad \\text{if } \\boldsymbol{v} \\geq 0 \\text{ and } \\lvert s - 1 \\rvert \\leq (N + 1) \\varepsilon\\,,\\quad s = \\boldsymbol{1}^\\intercal \\boldsymbol{v}\\,.
\\end{align}
```

Where:

  - ``s``: Sum of the entries of ``\\boldsymbol{v}``.
  - $(math_dict[:eps_machine])

The bound ``(N + 1) \\varepsilon`` is [`budget_rounding`](@ref).

The exact projection of such a vector lifts every zero entry by ``(1 - s) / N`` when ``s < 1``, and a later step compounds that residue as a real holding. The quotient keeps a zero exactly zero. It differs from the exact projection by at most ``\\lVert \\boldsymbol{v} - \\boldsymbol{v} / s \\rVert``, because a projection is non-expansive and ``\\boldsymbol{v} / s`` is its own projection. That distance is within the rounding of ``\\boldsymbol{v}`` itself.

# Algorithm

 1. Sum `v`, giving `s`. When every entry of `v` is non-negative and `abs(s - 1)` is at most [`budget_rounding`](@ref), return `v ./ s`.
 2. Sort `v` in decreasing order, giving `u`.
 3. For each `j`, add `u[j]` to the running sum `css` and take `t = (css - 1) / j`. When `u[j] - t > 0`, set `rho = j` and `theta = t`.
 4. Return `max.(v .- theta, 0)`.

# Arguments

  - `v`: The vector to project.

# Returns

  - `w::Vector`: The projection, a new vector summing to one.

# Related

  - [`project`](@ref)
  - [`EuclideanProjection`](@ref)
  - [`budget_rounding`](@ref)

# References

  - $(ref_dict[:duchi2008])
"""
function project_simplex(v::AbstractVector)
    s = sum(v)
    if all(x -> x >= zero(x), v) && abs(s - one(s)) <= budget_rounding(typeof(s), length(v))
        return v ./ s
    end
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
    budget_rounding(::Type{T}, n::Integer) where {T <: Union{Integer, Rational}}
    budget_rounding(::Type{T}, n::Integer)

Returns the rounding bound on the sum of an `n`-vector of type `T` that sums to one in exact arithmetic.

# Mathematical definition

```math
\\begin{align}
b(T, n) &= (n + 1) \\varepsilon\\,.
\\end{align}
```

Where:

  - ``b(T, n)``: Bound on ``\\lvert \\boldsymbol{1}^\\intercal \\boldsymbol{v} - 1 \\rvert`` for a computed ``\\boldsymbol{v}`` of ``n`` entries of type ``T``.
  - $(math_dict[:eps_machine])

The bound is deterministic, to first order in ``u``. The Price-Adjusted Allocation `w .* x ./ ⟨w, x⟩` sums to one in exact arithmetic whatever `Σ w` is. Each of its entries carries a relative error of at most ``(n + 2) u``: one rounding in the product, one in the quotient, and ``n`` in the dot product. The pairwise `sum` adds at most ``\\lceil \\log_2 n \\rceil u``. The total, ``(n + 2 + \\lceil \\log_2 n \\rceil) u``, is at most ``2 (n + 1) u = b(T, n)`` for every ``n \\geq 1``.

The probabilistic bound ``\\sqrt{n} \\varepsilon`` holds at one step with high probability, but not at every step of a long recursion. One miss leaves a residue that later steps compound. An integer or `Rational` type takes zero, because its sums carry no rounding. An AD dual takes the bound of its value type, through its own `eps`.

# Related

  - [`project_simplex`](@ref)
  - [`price_adjusted_allocation`](@ref)
"""
function budget_rounding(::Type{T}, n::Integer) where {T <: Union{Integer, Rational}}
    return zero(T)
end
function budget_rounding(::Type{T}, n::Integer) where {T}
    return (n + 1) * eps(T)
end
"""
    online_update!(alg::AbstractOnlinePortfolioSelectionAlgorithm, st, w::AbstractVector, x::AbstractVector, rows, set::AbstractAllocationSet)
    online_update!(alg::AbstractOnlinePortfolioSelectionAlgorithm, st, w::AbstractVector, x::AbstractVector, rows, set::AbstractAllocationSet, point::Option{<:AbstractVector})

Runs the Online Update, one row of the online portfolio selection recursion, which each rule writes once.

The function reads the rule, its private carrier `st`, the allocation `w` held during the period, the finite price relative `x` of that period, the rows carrier that the head holds through the period, and the head's Allocation Set. It returns `(st', w')`, the carrier and the allocation for the next period. A rule may write its carrier in place, and `nothing` is a legal carrier. `w'` is always a new vector, because the projection makes one, so the update never writes a `w` that is a view after [`port_opt_view`](@ref).

Every rule makes the same two steps. It takes an unconstrained step to a raw vector, and it then calls [`project`](@ref) onto the set in its geometry. A rule that reads `w` continues from the allocation it gets, which is the Start Allocation on the first row.

The seven-argument form names the **Gradient Point**. An [`ExpertMixture`](@ref) under [`BlendPoint`](@ref) gives every expert its played blend as `point`. A first-order rule then evaluates its gradient at that point and steps from its own iterate, so every expert reads the one gradient on which the dynamic-regret mixtures state their bounds. The generic method drops the point and runs the six-argument update. So a rule with no gradient ignores the point, and a `nothing` point, which the mixture gives under [`OwnPoint`](@ref), runs the six-argument update for every rule. [`MirrorDescent`](@ref), [`OptimisticStep`](@ref) and [`AdaptiveSubgradient`](@ref) read a given point.

# Mathematical definition

```math
\\begin{align}
(s_{t+1}, \\boldsymbol{q}) &= U(s_t, \\boldsymbol{w}_t, \\boldsymbol{x}_t)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q})\\,.
\\end{align}
```

Where:

  - ``U``: Unconstrained step of the rule.
  - ``s_t``: Carrier of the rule before the update of period ``t``.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - $(math_dict[:D_Psi_breg])

# Arguments

  - `alg`: The rule.
  - `st`: The rule's carrier, or `nothing`.
  - `w`: The allocation held during the period.
  - `x`: The price relative of the period, `1 .+ r`, one at a gap.
  - `rows`: The rows carrier that the head holds through the period, a [`ReturnsResult`](@ref) of the rows as they arrived, or `nothing`.
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
    price_relative(r::Number)

Returns the price relative of one return, `1 + r`, and one at a gap.

A non-finite return is a period in which the leg held cash, which is how the Held Gap reads it. So the Online Update sees a finite `x` at every asset, and a kernel over the rows reads the same number at the gap.

# Mathematical definition

```math
\\begin{align}
x_{tj} &= \\begin{cases} 1 + r_{tj} & \\text{if } r_{tj} \\text{ is finite}\\,, \\\\ 1 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``x_{tj}``: Price relative of asset ``j`` in period ``t``.
  - $(math_dict[:r_tj])

# Related

  - [`online_update!`](@ref)
  - [`online_selection_row!`](@ref)
"""
function price_relative(r::Number)
    return isfinite(r) ? one(r) + r : one(r)
end
"""
    rule_state_seed(::AbstractOnlinePortfolioSelectionAlgorithm, ::AbstractVector)
    rule_state_seed(alg::AbstractOnlinePortfolioSelectionAlgorithm, w::AbstractVector,
                    set::Option{<:AbstractAllocationSet})

Returns the carrier that a rule holds before its first update. The default is `nothing`, for a rule that carries nothing.

The head calls the three-argument form with its Allocation Set, and the default method of that form drops `set`. So a rule whose carrier reads only the Start Allocation implements the two-argument form. A rule that holds allocations of its own over the assets, as an [`ExpertMixture`](@ref) and a [`FollowTheLeadingHistory`](@ref) hold their experts, implements the three-argument form. It projects each allocation onto `set`, as the head projects its `w0`. When `set` is `nothing`, the method projects nothing.

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
  - [`project_start`](@ref)
"""
function rule_state_seed(::AbstractOnlinePortfolioSelectionAlgorithm, ::AbstractVector)
    return nothing
end
function rule_state_seed(alg::AbstractOnlinePortfolioSelectionAlgorithm, w::AbstractVector,
                         ::Option{<:AbstractAllocationSet})
    return rule_state_seed(alg, w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Projects a start allocation `w` once onto the Allocation Set `set` in the geometry `proj`, with `w` as its own reference.

The head projects its Start Allocation with this function, and so does each expert of an [`ExpertMixture`](@ref) or a [`FollowTheLeadingHistory`](@ref). So the allocation held during the first period lies in the set. When `set` is `nothing`, or when it reads the head's rows and so has no constraints before the first row, the function returns `w` as it is, and the first Online Update projects it.

# Related

  - [`rule_state_seed`](@ref)
  - [`project`](@ref)
  - [`online_selection_seed`](@ref)
"""
function project_start(proj::AbstractProjectionGeometry,
                       set::Option{<:AbstractAllocationSet}, w::AbstractVector)
    if isnothing(set) || isnothing(rows_needed(set))
        return w
    end
    return project(proj, set, w, w)
end
"""
    projection_geometry(alg::AbstractOnlinePortfolioSelectionAlgorithm)

Returns the Projection Geometry in which the head projects the Start Allocation. The default is the rule's `proj` field.

A rule overrides it when its start needs another geometry. [`NewtonStep`](@ref) returns [`EuclideanProjection`](@ref), because its Gram matrix is the identity before the first gradient.

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
function projection_geometry(alg::AbstractOnlinePortfolioSelectionAlgorithm)
    return alg.proj
end
"""
    rows_needed(alg::AbstractOnlinePortfolioSelectionAlgorithm)

Returns the number of rows that a rule reads at a step. The default is `0`, so a rule with no method of its own reads no rows.

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelectionState`](@ref)
"""
function rows_needed(::AbstractOnlinePortfolioSelectionAlgorithm)
    return 0
end
"""
    rows_needed_max(a, b)

Returns the larger of two row needs. `nothing` stands for every row, so it is larger than any number.

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

Slices a rule's carrier to the selected assets.

A `nothing` carrier stays `nothing`. A carrier that subtypes [`AbstractPartialFitState`](@ref) goes to its own [`port_opt_view`](@ref). The function refuses any other carrier.

# Validation

  - `st` is `nothing` or an [`AbstractPartialFitState`](@ref). An `ArgumentError` is thrown otherwise.

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

Returns the Price-Adjusted Allocation, the book that the fund holds at the end of a period before it trades.

It is one row of the self-financing Weight Drift at budget one. Every trade of the family starts from it. The update of [`BuyAndHold`](@ref) returns this vector, a turnover ceiling on a [`ProgrammeAllocationSet`](@ref) bounds the distance from it, and every projection gets it as its `w` argument.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{w}}_t &= \\frac{\\boldsymbol{w}_t \\odot \\boldsymbol{x}_t}{\\boldsymbol{w}_t^\\intercal \\boldsymbol{x}_t}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_hat_t_padj])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel])
  - ``\\odot``: Elementwise product.

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

Records a Held Step, a projection of one row that did not give an allocation on the budget, so the step traded nothing.

A projection onto a [`ProgrammeAllocationSet`](@ref), or in a [`GramProjection`](@ref), is a programme, and a programme can fail. It is infeasible on a day when a turnover ceiling and a cap cannot both hold. It can also time out, stop a MIP at its limit, or meet a covariance cone with no finite data on its first rows. A closed-form root can also miss the budget in floating point, as [`budget_or_held_step`](@ref) states. The step then returns the Price-Adjusted Allocation that it got, the book that the fund already holds, so the fund trades nothing. The rule's carrier still takes in the row. The head warns once with the timestamp of the row. Its Recursion Read-out carries the record of the last folded row inside an [`OptimisationSuccess`](@ref), so a fallback chain never runs on a hold. A step never throws on a failed solve, and never falls back to a weaker set, because a constraint that the step drops on the day it binds does not constrain.

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
    The timestamp of the row whose projection the step held. It is the index of the row in the fold when the carrier has no timestamps, and `nothing` outside a fold.
    """
    ts::T1
    """
    The projection of the row that the step held, and the reason.
    """
    reason::T2
    """
    The solver trials of the failed programme, the `trials` of a [`JuMPResult`](@ref), or `nothing` when the projection built no programme.
    """
    trials::T3
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the context of one row for every projection of that row: the rows carrier, the timestamp of the row, and the Held Steps recorded so far.

The projection of the Online Update is `project(proj, set, q, w)`, with four arguments and no more. Two things must reach it outside that signature. The covariance cone and the tracking error of a programme set read the head's rows, and the head must see the record of a Held Step. Both go on the task-scoped [`PROJECTION_STEP`](@ref). The head opens one step for each row through [`with_projection_step`](@ref), around the whole Online Update. So every projection of that row, those of a mixture's experts and that of its blend, reads one carrier and writes one log. The carrier is the one that the update gets, a [`ReturnsResult`](@ref) of the rows as they arrived, under the pinned names and the Asset Panel of the buffer. So a programme set fits its prior through [`rows_carrier`](@ref), as a batch head does. Outside a step, a projection reads no rows and reports a hold as a warning.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`PROJECTION_STEP`](@ref)
  - [`with_projection_step`](@ref)
  - [`rows_carrier`](@ref)
  - [`HeldStep`](@ref)
  - [`project`](@ref)
"""
struct ProjectionStep{T1, T2}
    """
    The rows carrier that the head holds through the period, a [`ReturnsResult`](@ref) of the rows as they arrived, or `nothing`.
    """
    rows::T1
    """
    The timestamp of the row, or the index of the row in the fold.
    """
    ts::T2
    """
    The head's `strict` flag. A programme set reads it when it resolves its rows, so it refuses a name that an exposure row states and the universe does not carry, as the head refuses one.
    """
    strict::Bool
    """
    The Held Steps recorded during this row, in the order in which their projections failed.
    """
    held::Vector{HeldStep}
end
"""
    const PROJECTION_STEP

Holds the task-scoped [`ProjectionStep`](@ref) that a projection reads. It is `nothing` outside a step.

# Related

  - [`ProjectionStep`](@ref)
  - [`with_projection_step`](@ref)
"""
const PROJECTION_STEP = ScopedValue{Union{Nothing, ProjectionStep}}(nothing)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Runs `f()` inside a [`ProjectionStep`](@ref) over the rows carrier `rows` at `ts`, with the head's `strict` flag.

It returns `(f(), held)`, the result of the update and the Held Steps that its projections recorded. `held` is empty when every projection met the budget.

# Related

  - [`PROJECTION_STEP`](@ref)
  - [`ProjectionStep`](@ref)
  - [`fold_online_selection`](@ref)
"""
function with_projection_step(f, rows::Option{<:ReturnsResult}, ts; strict::Bool = false)
    step = ProjectionStep(rows, ts, strict, HeldStep[])
    out = Base.ScopedValues.with(f, PROJECTION_STEP => step)
    return out, step.held
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Records a Held Step on the current [`ProjectionStep`](@ref). Outside a step, it warns at once and records nothing.

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

Returns the rows that the current [`ProjectionStep`](@ref) holds, or `nothing` outside a step.

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

Returns the head's `strict` flag that the current [`ProjectionStep`](@ref) carries, or `false` outside a step.

# Related

  - [`ProjectionStep`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
"""
function projection_step_strict()
    step = PROJECTION_STEP[]
    return isnothing(step) ? false : step.strict
end
"""
    assert_rule_admits_set(alg::AbstractOnlinePortfolioSelectionAlgorithm, set::AbstractAllocationSet)

Refuses, when the head is built, a rule and an Allocation Set that cannot work together.

The default admits every pair. A rule that holds an optimisation estimator with no model for a programme set, such as [`FollowTheLeader`](@ref), refuses that set by name. A rule over other rules sends the check to each of them.

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

Holds the Partial Fit State of an online portfolio selection head: the Rule State, the rows, the Fold Context and every folded timestamp.

The pair `(st, w)` is the Rule State, the unit over which the family recurses. The state holds once the rows that any rule of the tree reads, as a [`SampleBufferState`](@ref) of returns. The buffer keeps the rows of the Returns Result as they arrived, with `NaN` where there was no return, and under a time-varying panel it keeps the active mask of each row beside it. [`rows_needed`](@ref) of the rule tree caps the buffer, and the buffer is `nothing` for a tree that reads no rows. At every row the head reads the buffer out as a [`ReturnsResult`](@ref) through [`rows_carrier`](@ref), so a statistic over the rows reduces to its Coverage Universe as a batch fit over the same window does. No cap applies to the timestamps, so [`Resume`](@ref) works for a rule with no carrier. The active mask of the last folded row is the Investable Mask on which the read-out reduces, and it is `nothing` under a static panel. The recursion keeps the full `w` and never carries a forced zero.

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
  - `pnl` is static when it is not `nothing`. An `ArgumentError` is thrown otherwise, because the buffer holds the masks of a time-varying panel with the rows.

## View parameters

`OnlinePortfolioSelectionState` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method passes `args...` to the view of the carrier and to the view of the rows buffer.
  - `w` is sliced and renormalised through [`renormalised_view`](@ref), as a copy. The recursion keeps the full `w`.
  - `st` goes to the view of the carrier through [`rule_state_view`](@ref).
  - `X`, `nx`, `pnl` and `amsk` are sliced along the asset axis.
  - `n`, `ts` and `hold` pass through, and `ts` is copied.

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
    The allocation held during the current period. It covers the full pinned universe and sums to one.
    """
    w
    """
    The rule's private carrier, or `nothing`.
    """
    st
    """
    The rows of returns that any rule of the tree reads, a [`SampleBufferState`](@ref) of the rows as they arrived and of their active masks. [`rows_needed`](@ref) caps it. It is `nothing` when the tree reads no rows.
    """
    X
    """
    Names of the asset columns, pinned by the first step.
    """
    nx
    """
    The static [`AssetPanel`](@ref) of the universe, pinned by the first step, or `nothing`. The state never holds a time-varying panel here.
    """
    pnl
    """
    The active mask of the last folded row under a time-varying panel, which is the Investable Mask of the read-out, or `nothing` under a static panel.
    """
    amsk
    """
    Timestamps of the folded observations, in order and with no cap, or `nothing` when the carrier holds none.
    """
    ts
    """
    The [`HeldStep`](@ref) record of the last folded row when the step held a projection of that row, or `nothing` when every projection of it met the budget. The return code of the Recursion Read-out carries it.
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

Refuses to merge two online selection states.

An Online Update is not a sufficient statistic for its block. A carrier can be one, such as a Gram sum or a prefix of rows, but the allocation beside it is not. So two states folded on disjoint blocks describe no single run. To fold the rows in one pass, run the Causal Pass, which costs `O(N)` a row.

# Validation

  - The function always throws an `ArgumentError`.

# Related

  - [`OnlinePortfolioSelectionState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(::OnlinePortfolioSelectionState, ::OnlinePortfolioSelectionState)
    return throw(ArgumentError("an `OnlinePortfolioSelectionState` cannot be merged: an online update is order-dependent, so two states folded on disjoint blocks describe no single run. Run the Causal Pass over the concatenated rows instead, `optimise(opt, rd)`."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`OnlinePortfolioSelectionState`](@ref) so that the copy shares no array with the original.

The method calls `copy` on `w`, `st`, `X`, `nx`, `amsk` and `ts`. A carrier and the rows buffer copy their own arrays through their `Base.copy` methods. `n`, `pnl` and `hold` pass through unchanged.

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

Slices an [`OnlinePortfolioSelectionState`](@ref) to the selected assets, giving the state of the same observations over those assets.

The method slices `w` and renormalises it. It sends the carrier to its own view through [`rule_state_view`](@ref), and the rows buffer to the view of [`SampleBufferState`](@ref). It slices the names, the panel and the mask, and it keeps the count, the timestamps and the hold record. The view is a copy for the read-out, and the recursion keeps the full `w`.

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

Slices an allocation to the selected assets and renormalises it to sum to one, as a copy. A `nothing` allocation stays `nothing`.

The function spreads the mass that the selection leaves out over the kept assets in proportion. Every per-asset view of an allocation in the family reads it this way, the read-out under a time-varying panel and the view of a rule's own allocation among them. A selection that keeps none of the mass, such as a view of a one-hot allocation to the other assets, has no proportion. It returns the uniform allocation over the kept assets, the Start Allocation of the family.

# Mathematical definition

```math
\\begin{align}
v_i &= \\begin{cases} w_i \\big/ \\sum_{j \\in I} w_j & \\text{if } \\sum_{j \\in I} w_j \\neq 0\\,, \\\\ 1 / \\lvert I \\rvert & \\text{otherwise}\\,, \\end{cases} \\quad i \\in I\\,.
\\end{align}
```

Where:

  - ``v_i``: Entry of the view for the kept asset ``i``.
  - ``w_i``: Entry of the allocation for asset ``i``.
  - ``I``: Set of the kept assets.

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

Abstract supertype for the Learning-Rate Schedules that a first-order Online Selection Rule can hold in `eta` in place of a number.

The rule reads its schedule at every update, as a function of the period count and of the rule's carrier. So a schedule can follow the count alone, read a running statistic of the run that the carrier keeps for it, or name a stage boundary at which the rule restarts. A schedule owns no state. What it accumulates lives on the rule's carrier. The schedule seeds that statistic and writes it once for each row. By default it writes after the step, so the rate of a period reads only the past. A schedule that chooses the rate from the row of the period itself writes before the rule reads the rate.

# Interfaces

To implement a new schedule, subtype `AbstractLearningRateSchedule` and implement the following methods:

  - `learning_rate(sched::AbstractLearningRateSchedule, t::Integer, st) -> Real`: The rate of the update at period `t`, the `t`-th row that the rule has seen. `st` is the rule's carrier, and its `s` field is the schedule's own statistic.
  - `restart(sched::AbstractLearningRateSchedule, t::Integer) -> Bool`: Whether the update at period `t` closes a stage. At a stage boundary the rule returns the Start Allocation for period `t + 1` and puts its carrier back at its seed. The default is `false`.
  - `schedule_state_seed(sched::AbstractLearningRateSchedule, w::AbstractVector)`: The statistic that the carrier holds for the schedule before the first row. The default is `nothing`.
  - `schedule_update!(sched::AbstractLearningRateSchedule, s, w::AbstractVector, x::AbstractVector)`: Writes the row into the statistic `s`, from the allocation `w` played during the period and its price relative `x`. The default returns `s` unchanged.
  - `reads_period_row(sched::AbstractLearningRateSchedule) -> Bool`: Whether the schedule chooses the rate of period `t` from the row of period `t`. When it does, the rule writes the statistic before it reads the rate, and not after the step. The default is `false`. A rule whose update also needs the rate of the *next* period, as the online form of [`ExpectationMaximisation`](@ref) does, refuses a schedule that returns `true`, because the row of the next period is not there to read.
  - `mixing_share(sched::AbstractLearningRateSchedule, t::Integer, alpha::Real) -> Real`: The share of the uniform mix in the update at period `t`. The default is the rule's own `alpha`, and a schedule that sets the share from a stage length returns its own.

A statistic with entries over the assets, such as the expert allocations of a schedule that replays the run, also implements [`schedule_state_view`](@ref). That method slices the statistic to the selected assets with the rule's carrier. The default copies a statistic over the run as it is.

A number in `eta` is the constant schedule. The six verbs return the number, `false`, `nothing`, the statistic unchanged, `false` and `alpha`.

# Examples

A schedule that halves the rate at every period, from `0.4`. Two steps of exponentiated gradient at the rates `0.4` and `0.2` give the allocation below.

```jldoctest
julia> struct HalvingRate{T} <: PortfolioOptimisers.AbstractLearningRateSchedule
           eta::T
       end

julia> function PortfolioOptimisers.learning_rate(sched::HalvingRate, t::Integer, st)
           return sched.eta / 2^(t - 1)
       end

julia> PortfolioOptimisers.learning_rate(HalvingRate(0.4), 3, nothing)
0.1

julia> rd = ReturnsResult(; nx = [\"A\", \"B\", \"C\"], X = [0.1 -0.1 0.0; 0.0 0.2 -0.05]);

julia> round.(optimise(OnlinePortfolioSelection(; alg = MirrorDescent(; eta = HalvingRate(0.4))),
                       rd).w; digits = 4)
3-element Vector{Float64}:
 0.3436
 0.3295
 0.327
```

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

Returns the step size that a first-order rule takes at period `t`.

When `eta` is a number, the step size is that number. Otherwise the schedule returns it, from the period count and the rule's carrier.

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

Checks whether the update at period `t` closes a stage of the schedule.

At a stage boundary the rule returns the Start Allocation for period `t + 1` and puts its carrier back at its seed. A number and every schedule with no stages return `false`.

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

Returns the statistic that a rule's carrier holds for its schedule before the first row. It is `nothing` for a number and for a schedule that reads no statistic.

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

Writes one row into the schedule's statistic `s`, from the allocation `w` played during the period and its price relative `x`, and returns the statistic.

For a number, and for a schedule that reads no statistic, it returns `s` unchanged.

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

Returns the share of the uniform mix in the update at period `t`.

It is the rule's own `alpha` under a number and under every schedule that does not set the share. Under [`DoublingTrickRate`](@ref) the share is a function of the stage length, and it replaces `alpha`.

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

Checks whether the schedule chooses the rate of period `t` from the row of period `t`.

It is `true` for a schedule that replays the period before it returns a rate, as [`WindowedBestRate`](@ref) does. It is `false` for a number and for every schedule that reads only the past. When it is `true`, a rule writes the statistic before it reads the rate. Otherwise the rule writes it after its step. [`statistic_before_rate`](@ref) and [`statistic_after_step`](@ref) are the two calls.

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`schedule_update!`](@ref)
"""
function reads_period_row(::Union{<:Real, <:AbstractLearningRateSchedule})
    return false
end
"""
    statistic_before_rate(eta, s, w::AbstractVector, x::AbstractVector)

Returns the statistic from which the rule reads the rate of the period. This is the first of the two points at which a rule writes the row of its schedule.

The row goes in here when [`reads_period_row`](@ref) returns `true`. Otherwise the call returns `s` unchanged. [`statistic_after_step`](@ref) is the other point. A rule makes both calls, and the schedule decides which one writes.

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

Returns the statistic that the carrier keeps after the step. This is the second of the two points at which a rule writes the row of its schedule.

The row goes in here when [`reads_period_row`](@ref) returns `false`, the default. Otherwise the call returns `s` unchanged. [`statistic_before_rate`](@ref) is the other point.

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

Slices a schedule's statistic to the selected assets when a view slices the rule's carrier.

By default it returns a copy of the statistic as it is, because a schedule that reads the count or the run keeps a statistic over the run and not over the assets. A statistic that holds allocations, as the statistic of [`WindowedBestRate`](@ref) does, slices and renormalises each allocation through [`renormalised_view`](@ref).

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
