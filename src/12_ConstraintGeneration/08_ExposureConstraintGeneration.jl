"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the bases a linear constraint can be re-based into.

A constraint can be re-based into another basis **if and only if it is a linear form in the weights**. Under a change of basis ``\\boldsymbol{w}_b = \\mathbf{P}^\\intercal \\boldsymbol{w}``, the row ``\\boldsymbol{a}`` becomes ``\\mathbf{P}\\boldsymbol{a}``, and nothing else about the problem changes.

The boundary is a property of the **mechanism**: a re-basis rewrites a row and leaves the model untouched, so a constraint that reaches the model through its own variables is outside it even where a change of basis is well defined for the quantity constrained. Cardinality, sub-group cardinality and buy-in thresholds index the binary held-indicators rather than the weights. A per-asset weight box has no counterpart because ``lb \\leq \\mathbf{P}^\\intercal\\boldsymbol{w} \\leq ub`` *is* a linear constraint and already has a home. Turnover and tracking error are norm forms: ``\\lVert \\mathbf{P}^\\intercal(\\boldsymbol{w} - \\boldsymbol{w}_0) \\rVert`` is meaningful, but it needs its own variables and cones, so it is re-basable in mathematics and not by this mechanism. A fee is priced per traded position and is subtracted from the return, so it is not a constraint on the weights at all. The list illustrates the rule rather than exhausting it, and these absences are a property of the constraint, not a gap in the implementation; see ADR 0047 and its 2026-08-16 amendment.

Tracking a factor needs no re-basis: [`ReturnsTracking`](@ref) takes a benchmark *return series*, and a factor's return series is a column of the factor matrix, so it is passed directly.

The family has one member today, [`FactorSpace`](@ref), and the abstract type exists so it can gain more: any linear change of basis in the weights — currency, sector, a caller-supplied exposure table — lands here without a new pathway.

There is deliberately **no `AssetSpace`**. The asset frame is the *absence* of a re-basis, spelled by using a bare [`LinearConstraintEstimator`](@ref); an `AssetSpace` member would make [`ExposureConstraintEstimator`](@ref) a no-op decorator computing bit-for-bit what it wraps.

# Interfaces

In order to implement a new basis which will work seamlessly with the library, subtype `AbstractConstraintSpace` with the source of its change of basis as its fields, and implement the following methods.

## Basis resolution

  - `PortfolioOptimisers.constraint_space_basis(space::AbstractConstraintSpace, sets::UniverseSets, rr, rd) -> (basis, key)`: The change of basis the rows project through, and the key their names resolve against.

There is no fallback. Every check the basis needs is made here, so that the assembly loop after it can assume a consistent basis, and a space that checks nothing states so rather than omitting the method.

### Arguments

  - `space`: The basis.
  - `sets`: The declared universe the names resolve against.
  - `rr`: The prior's loadings, or `nothing`.
  - `rd`: Returns the space may refit from, or `nothing`.

### Returns

  - `basis`: The change of basis, in whatever form [`rebase_linear_constraints`](@ref) reads for this space.
  - `key::AbstractString`: The key of `sets.dict` the row's names resolve against.

## Asset view

  - `PortfolioOptimisers.port_opt_view(space::AbstractConstraintSpace, i, args...) -> AbstractConstraintSpace`: An asset-sliced copy of the space.

A basis is indexed by asset on one side and by the space's own names on the other, so the generic fallback is wrong for every member of this family. [`FactorSpace`](@ref)'s method is the reference: it slices the loadings by asset and leaves the factor axis whole.

### Arguments

  - `space`: The basis.
  - `i`: The asset index the view keeps.

### Returns

  - `space::AbstractConstraintSpace`: The basis over the selected assets.

# Related

  - [`FactorSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
abstract type AbstractConstraintSpace <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The factor basis: a constraint written in factor names, re-based through a regression's loadings.

A space carries the source of its own basis in `re`. A precomputed [`Regression`](@ref) states the basis outright, an estimator refits it from the returns, and `nothing` reads the prior's own loadings. [`factor_space_regression`](@ref) ranks the three and owns the order.

!!! warning

    A **precomputed** `re` does not refit. The rows are re-based through exactly the loadings handed in, whatever data the problem is later solved on, and nothing at generation time can see that it is inside a cross-validation fold: the projection is silently stale rather than wrong-shaped, because the loadings are the right shape for the full universe. Two spellings already exist for a basis that must move — `re = <an estimator>`, which refits per fold and per subproblem, and a time-dependent schedule on `lcse`, which is already [`TD_Option`](@ref) and can carry a different basis per period. A stated result is also refused outright at a [`NestedClustered`](@ref) outer solve, where the universe is *replaced* by cluster names rather than sliced, so no view of the loadings can follow it.

# Mathematical definition

```math
\\begin{align}
  \\boldsymbol{w}_f &= \\mathbf{M}^\\intercal \\boldsymbol{w} \\\\
  \\boldsymbol{a}^\\intercal \\boldsymbol{w}_f &= (\\mathbf{M}\\boldsymbol{a})^\\intercal \\boldsymbol{w} \\\\
  \\mathbf{A}\\boldsymbol{w}_f \\leq \\boldsymbol{b} &\\iff \\left(\\mathbf{A}\\mathbf{M}^\\intercal\\right)\\boldsymbol{w} \\leq \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}``: ``Na \\times Nf`` loadings matrix, [`Regression`](@ref)'s `M`.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_f``: ``Nf \\times 1`` factor weights (exposures).
  - ``\\boldsymbol{a}``: ``Nf \\times 1`` row the user wrote in factor names.
  - ``\\mathbf{A}``: ``Nc \\times Nf`` coefficient matrix, one row per constraint.
  - ``\\boldsymbol{b}``: ``Nc \\times 1`` right-hand side of those constraints.

The third line is the whole of the re-basis, and it says two things. A constraint written over the factors becomes a constraint over the assets when its coefficient matrix is post-multiplied by ``\\mathbf{M}^\\intercal``. And ``\\boldsymbol{b}`` is the same on both sides: a change of basis acts on the row, never on the bound. An equality reads the same way, with ``\\leq`` replaced by ``=``.

A factor exposure is ``\\mathbf{M}^\\intercal\\boldsymbol{w}`` and nothing more. It is defined without the intercept a [`Regression`](@ref) also carries, which appears in no line above. ``\\boldsymbol{b}`` above is the right-hand side of the constraints and is a different quantity from that intercept.

The names resolve against the factor axis the loadings themselves name — [`factor_axis_key`](@ref) reads `sets.tfkey` off a [`Regression`](@ref) and `sets.cfkey` off a [`CrossSectionalFactorModel`](@ref), so a mandate written for one family can never resolve against the other's names. The loadings are `rr.M` rather than `rr.L` — `M`'s columns are the named original factors, and a constraint must be *written* in names a user can put in an equation, whereas `L`'s columns are principal components under [`DimensionReductionRegression`](@ref). Risk decomposition reads `L` and is correct to; the two are the two sides of one projection, not a drift between conventions.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorSpace(; re::Option{<:RegE_Reg} = nothing) -> FactorSpace

Keywords correspond to the struct's fields.

## Validation

  - `re` is bounded by its type, `Option{<:RegE_Reg}`, and by nothing else. The constructor checks no condition of its own. A value outside the bound — a matrix of loadings passed raw, for instance — raises a `TypeError` from the keyword constructor, so a space that is built is a space whose source is of a kind the resolution can read.

# Related

  - [`AbstractConstraintSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`Regression`](@ref)
  - [`factor_space_regression`](@ref): ranks the three carriers `re` competes with, and states the order.
  - [`resolve_factor_regression`](@ref): the shared precedence, which [`FactorRiskContribution`](@ref) and [`FactorRiskBudgeting`](@ref) read too.

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.6.
"""
@concrete struct FactorSpace <: AbstractConstraintSpace
    """
    $(field_dict[:fs_re])
    """
    re
end
function FactorSpace(; re::Option{<:RegE_Reg} = nothing)::FactorSpace
    return FactorSpace(re)
end
"""
$(DocStringExtensions.TYPEDEF)

Decorator that re-bases a linear constraint into another basis.

`ExposureConstraintEstimator` wraps whatever `lcse` itself accepts — a [`LinearConstraintEstimator`](@ref), a precomputed [`LinearConstraint`](@ref), or a vector of either — and declares the basis its rows are written in. It **wraps rather than reimplements**, so equation parsing, `val`/`key` validation and group expansion are inherited, and the bound being exactly `lcse`'s means no shape can reach the optimiser un-re-based.

The projection happens **at generation time**, so what leaves constraint generation is an ordinary asset-space [`LinearConstraint`](@ref). It flows into the existing `lcsr` slot and the existing model assembly: there is no second constraint pathway, and Near Optimal Centering, time-dependent schedules and every optimiser sharing [`JuMPOptimiser`](@ref) get factor exposure constraints without knowing they exist.

The re-basis lives here and not on [`LinearConstraintEstimator`](@ref) on purpose. That type also feeds `gcarde` and `sgcarde`, which build rows over the binary held-indicators; a projected row is neither integral nor an index into them. Leaving it unmarked is what keeps it the only thing those slots admit, and therefore what makes the category error **unrepresentable** rather than validated. See ADR 0047.

An `ExposureConstraintEstimator` is also usable as a bare [`Pipeline`](@ref) step, where it reads the `prior` slot for its basis and writes an ordinary asset-space [`LinearConstraint`](@ref) into `constraints`. The factor names resolve against the `nf` axis [`pipeline_asset_sets`](@ref) builds from `rd.nf`, so the axis and the loadings agree by construction. **Prefer the `lcse` field of a [`JuMPOptimiser`](@ref) to the step** for a factor mandate: the step projects once, against the prior it saw, and the field re-projects against the optimiser's own prior, per fold. [`constraint_step_value`](@ref) states the trade-off in full.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExposureConstraintEstimator(;
        lce::LcE_Lc_VecLcE_Lc,
        space::AbstractConstraintSpace
    ) -> ExposureConstraintEstimator

Keywords correspond to the struct's fields. `space` is required and has no default: a re-basis is something the user asks for, never something inferred.

## Validation

  - If `lce` is an `AbstractVector`, `!isempty(lce)`.

The constructor checks that one condition and no more. Every check the *basis* needs is made when the constraint is generated, because the names sit on the factor axis while the basis comes from the prior, and a single object cannot see both. [`constraint_space_basis`](@ref) owns those three checks and states them.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"nf\" => [\"MTUM\", \"VLUE\"]));

julia> rr = Regression(; M = [1.0 0.0; 0.5 0.5; 0.0 1.0]);

julia> ece = ExposureConstraintEstimator(; lce = LinearConstraintEstimator(; val = \"MTUM <= 0.3\"),
                                         space = FactorSpace());

julia> lcr = linear_constraints(ece, sets; rr = rr);

julia> lcr.ineq.A
1×3 transpose(::Matrix{Float64}) with eltype Float64:
 1.0  0.5  0.0

julia> lcr.ineq.B
1-element Vector{Float64}:
 0.3
```

# Related

  - [`AbstractConstraintSpace`](@ref)
  - [`FactorSpace`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`LinearConstraint`](@ref)
  - [`linear_constraints`](@ref)
  - [`UniverseSets`](@ref)
  - [`Pipeline`](@ref)
  - [`constraint_space_basis`](@ref): every check the basis needs, made when the constraint is generated.
  - [`constraint_step_value`](@ref): the pipeline step, and why a step-generated constraint is pinned to the pipeline's prior.
  - [`pipeline_asset_sets`](@ref): the factor axis the step's names resolve against.

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.6.
"""
@concrete struct ExposureConstraintEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:ece_lce])
    """
    lce
    """
    $(field_dict[:ece_space])
    """
    space
    function ExposureConstraintEstimator(lce::LcE_Lc_VecLcE_Lc,
                                         space::AbstractConstraintSpace)::ExposureConstraintEstimator
        if isa(lce, AbstractVector)
            @argcheck(!isempty(lce), IsEmptyError("lce cannot be empty"))
        end
        return new{typeof(lce), typeof(space)}(lce, space)
    end
end
function ExposureConstraintEstimator(; lce::LcE_Lc_VecLcE_Lc,
                                     space::AbstractConstraintSpace)::ExposureConstraintEstimator
    return ExposureConstraintEstimator(lce, space)
end
"""
    const EcE_LcE_Lc = Union{<:ExposureConstraintEstimator, <:LcE_Lc}

Alias for a union of an [`ExposureConstraintEstimator`](@ref) and the types it decorates.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`LcE_Lc`](@ref)
"""
const EcE_LcE_Lc = Union{<:ExposureConstraintEstimator, <:LcE_Lc}
"""
$(DocStringExtensions.TYPEDEF)

Alias for an abstract vector of [`EcE_LcE_Lc`](@ref) elements.

# Related

  - [`EcE_LcE_Lc`](@ref)
"""
const VecEcE_LcE_Lc = AbstractVector{<:EcE_LcE_Lc}
"""
    const EcE_LcE_Lc_VecEcE_LcE_Lc = Union{<:EcE_LcE_Lc, <:VecEcE_LcE_Lc}

Alias for a union of [`EcE_LcE_Lc`](@ref) or a vector of them.

This is the widened bound for `lcse` — the one slot that admits a re-basis. `gcarde` and `sgcarde` keep the narrower [`LcE_Lc`](@ref)-based aliases, which is the enforcement described in [`ExposureConstraintEstimator`](@ref).

# Related

  - [`EcE_LcE_Lc`](@ref)
  - [`VecEcE_LcE_Lc`](@ref)
"""
const EcE_LcE_Lc_VecEcE_LcE_Lc = Union{<:EcE_LcE_Lc, <:VecEcE_LcE_Lc}
"""
    port_opt_view(space::FactorSpace, i, args...) -> FactorSpace

Return an asset-sliced copy of a [`FactorSpace`](@ref).

Only the basis is data. `re` is sliced by [`port_opt_view`](@ref)'s [`Regression`](@ref) method, which takes the rows of `M` belonging to `i` — the loadings are assets-major, so a subset of assets is a subset of rows and the projection over the subset is exact. A space that states no basis comes back bit-identical, because a `nothing` and an estimator both carry no universe.

This is why a space is **viewed** where the universe is sliced and **refused** where it is replaced: a [`NestedClustered`](@ref) outer solve writes cluster names over the asset universe, and no row-slice of asset loadings follows that. See [`assert_external_optimiser`](@ref).

# Arguments

  - `space`: The [`FactorSpace`](@ref) to view.
  - `i`: The asset index the view keeps.
  - `args...`: Additional positional arguments, forwarded to the [`Regression`](@ref) method.

# Returns

  - `space::FactorSpace`: A space whose `re` holds the loadings of the selected assets, over the whole factor axis.

# Related

  - [`port_opt_view`](@ref)
  - [`FactorSpace`](@ref)
  - [`Regression`](@ref)
"""
function port_opt_view(space::FactorSpace, i, args...)::FactorSpace
    return FactorSpace(; re = port_opt_view(space.re, i, args...))
end
"""
    port_opt_view(ece::ExposureConstraintEstimator, i, args...) -> ExposureConstraintEstimator

Return an asset-sliced copy of the `lcse` slot.

The wrapped constraint is passed through and the **space** is viewed. That split is the whole content of the method: the wrapped shape is written in the space's names, so an asset index means nothing to it, while the space's basis is the one thing in the slot that is indexed by asset.

A vector slot ([`VecEcE_LcE_Lc`](@ref)) is covered by the generic vector [`port_opt_view`](@ref) method, mixed or not, which views each element in turn instead of slicing the list of constraints by asset index.

# Arguments

  - `ece`: The [`ExposureConstraintEstimator`](@ref) to view.
  - `i`: The asset index the view keeps.
  - `args...`: Additional positional arguments, forwarded to the space's method.

# Returns

  - `ece::ExposureConstraintEstimator`: The same wrapped shape, over a viewed space.

# Related

  - [`port_opt_view`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`FactorSpace`](@ref)
"""
function port_opt_view(ece::ExposureConstraintEstimator, i,
                       args...)::ExposureConstraintEstimator
    return ExposureConstraintEstimator(; lce = ece.lce,
                                       space = port_opt_view(ece.space, i, args...))
end
"""
    factor_space_regression(re::Option{<:RegE_Reg}, rr::Option{<:AbstractLoadingsRegressionResult},
                            rd::Option{<:ReturnsResult})
        -> Option{<:AbstractLoadingsRegressionResult}

Apply [`resolve_factor_regression`](@ref)'s precedence to the three carriers a [`FactorSpace`](@ref) can read its loadings from, and return `nothing` when none of them holds any.

The precedence is the one [`FactorRiskContribution`](@ref) already uses — a stated result, then the prior's own `rr`, then a refit — and the refit arm delegates to [`resolve_factor_regression`](@ref) rather than repeating it. What differs is the carriers: constraint generation is handed the loadings as `rr`, already read off the prior, so the prior arm is a plain argument rather than a field read, and both `re` and `rd` are optional here because a space that names no source and a route that carries no returns are both ordinary.

Returning `nothing` rather than throwing is deliberate: the diagnosis for "no basis anywhere" belongs to [`constraint_space_basis`](@ref), which knows it is assembling a constraint.

# Algorithm

 1. Return `re` when it is a precomputed [`Regression`](@ref). The caller has stated the basis, and no data is needed.
 2. Otherwise return `rr` when the prior carries one. This is what a `re === nothing` always resolves to, and it is the behaviour of every [`FactorSpace`](@ref) written before the field existed.
 3. Otherwise return `nothing` when `re` is `nothing` too. No carrier holds a basis, and [`constraint_space_basis`](@ref) writes the diagnosis.
 4. Otherwise `re` is an estimator. Raise when `rd` is `nothing`, and return `resolve_factor_regression(re, rd)` when it is not, which refits the loadings from the returns.

Step 4 is a capability the field adds rather than a fallback: `FactorSpace(; re = StepwiseRegression())` is **a factor mandate on a prior that carries no loadings**, which previously threw.

# Arguments

  - `re::Option{<:RegE_Reg}`: The space's stated basis source, or `nothing` to read `rr`.
  - `rr::Option{<:AbstractLoadingsRegressionResult}`: The prior's loadings, or `nothing`.
  - `rd::Option{<:ReturnsResult}`: Returns to refit from, or `nothing` on the standalone route.

# Validation

  - An estimator in `re` with no `rd` throws an [`IsNothingError`](@ref) naming both fixes. The refit needs the returns, so it is available at a [`JuMPOptimiser`](@ref) and as a [`Pipeline`](@ref) step, and refused on the standalone [`linear_constraints`](@ref) route, which receives none.

# Returns

  - `rr::Option{<:AbstractLoadingsRegressionResult}`: The loadings the highest-ranked carrier holds, or `nothing` when none of the three holds any.

# Related

  - [`constraint_space_basis`](@ref)
  - [`FactorSpace`](@ref)
  - [`resolve_factor_regression`](@ref)
"""
function factor_space_regression(re::Option{<:RegE_Reg},
                                 rr::Option{<:AbstractLoadingsRegressionResult},
                                 rd::Option{<:ReturnsResult})
    if isa(re, AbstractLoadingsRegressionResult)
        return re
    end
    if !isnothing(rr)
        return rr
    end
    if isnothing(re)
        return nothing
    end
    @argcheck(!isnothing(rd),
              IsNothingError("a `FactorSpace` whose `re` is a regression *estimator* ($(nameof(typeof(re)))) must fit the loadings from returns, and this route carries none. `linear_constraints(ece, sets; rr = ...)` is the standalone entry point: it receives the loadings directly and never the data they would be fitted from.\nEither state the basis outright, `FactorSpace(; re = Regression(; M = ...))`, or pass the loadings as `rr` and leave `re === nothing`, or generate the constraint through a `JuMPOptimiser` or a `Pipeline` step, where the returns are in scope and the refit happens per fold."))
    return resolve_factor_regression(re, rd)
end
"""
    constraint_space_basis(space::AbstractConstraintSpace, sets::UniverseSets, rr, rd)
        -> (basis, key)

Resolve the basis a re-based constraint projects through, and the key its names resolve against, throwing if either is unusable.

Every check a space needs before a single row is assembled lives here, so the assembly loop can assume a consistent basis. For [`FactorSpace`](@ref) that is: the loadings resolve from one of the three carriers [`factor_space_regression`](@ref) ranks, and — via [`factor_universe`](@ref), shared with every other consumer of the declared axis — the factor axis is declared and agrees with those loadings on how many factors there are.

`rd` is the returns the space may refit from. It is threaded from the [`JuMPOptimiser`](@ref) and from the [`Pipeline`](@ref) step, and is `nothing` on the standalone route.

# Algorithm

 1. Resolve `rr`, the loadings, with [`factor_space_regression`](@ref) over the three carriers `space.re`, `rr` and `rd`.
 2. Raise when `rr` is `nothing`. No carrier holds a basis, and no row can be assembled.
 3. Check the declared factor axis with [`factor_universe`](@ref), against `size(rr.M, 2)`. The call raises and returns no value the assembly reads.
 4. Return `rr` and the factor axis key [`factor_axis_key`](@ref) reads off `rr`, which is the key the row's names resolve against.

# Arguments

  - `space`: The [`FactorSpace`](@ref) whose basis is resolved.
  - `sets`: The declared universe, which must carry the factor axis `rr` names, under `sets.tfkey` or `sets.cfkey`.
  - `rr`: The prior's loadings, or `nothing`.
  - `rd`: Returns the space may refit from, or `nothing` on the standalone route.

# Validation

The names are looked up on the factor axis, but the *basis* comes from the prior, so the two can disagree in ways a single object cannot check. That is why all three of these are checked here and not at construction.

  - **A missing basis throws, ignoring `strict`.** `strict` governs unknown *names*: a per-row, recoverable condition where the offending row is dropped and the rest of the problem is still the problem the caller described. A missing regression is not that — it makes every row unbuildable, and dropping them silently yields a feasible, plausible-looking portfolio carrying none of the requested exposure. "Missing" means no carrier holds any: the space can supply its own through [`FactorSpace`](@ref)'s `re`, which is what makes a factor mandate legal on a prior with no factor block. Raises an [`IsNothingError`](@ref).
  - **A missing factor axis throws**, naming the key [`factor_axis_key`](@ref) chose. The axis is optional on [`UniverseSets`](@ref); it is not optional for a constraint written against it. [`factor_universe`](@ref) raises a `KeyError`.
  - **`size(rr.M, 2) == length(sets.dict[key])` always.** [`factor_universe`](@ref) raises a `DimensionMismatch` otherwise. The name-level cross-check against `rd.nf` needs the returns and lives at the optimiser.

# Returns

  - `basis::AbstractLoadingsRegressionResult`: The loadings the rows project through.
  - `key::AbstractString`: The key the row's names resolve against, [`factor_axis_key`](@ref) of `rr` for a [`FactorSpace`](@ref).

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`FactorSpace`](@ref)
  - [`factor_space_regression`](@ref)
  - [`factor_universe`](@ref)
"""
function constraint_space_basis(space::FactorSpace, sets::UniverseSets,
                                rr::Option{<:AbstractLoadingsRegressionResult},
                                rd::Option{<:ReturnsResult} = nothing)
    rr = factor_space_regression(space.re, rr, rd)
    @argcheck(!isnothing(rr),
              IsNothingError("a factor exposure constraint is written in factor names and re-based through the regression loadings, so it needs a source for them, and none of the three carriers holds any: the space states none (`space.re === nothing`) and the prior carries none (`rr === nothing`). Unlike an unknown name, this is not recoverable per row and is not governed by `strict`: every row of the constraint would be dropped, leaving a feasible portfolio with none of the requested exposure.\nState the basis on the space instead, `FactorSpace(; re = Regression(; M = ...))` to pin it or `FactorSpace(; re = StepwiseRegression())` to refit it from the returns. $prior_regression_remedy"))
    key = factor_axis_key(sets, rr)
    factor_universe(sets, key, size(rr.M, 2), "a $(FactorSpace) constraint", "rr.M")
    return rr, key
end
"""
    project_linear_constraint(lc::LinearConstraint, M::MatNum) -> LinearConstraint
    project_linear_constraint(plc::PartialLinearConstraint, M::MatNum) -> PartialLinearConstraint

Re-base an **already assembled** constraint by projecting its coefficient matrix, `A * transpose(M)`.

This is the precomputed-constraint half of the estimator path's per-term projection, and it is the one place [`linear_constraints`](@ref) stops being a pass-through for a [`LinearConstraint`](@ref): a precomputed constraint reaching the optimiser through an [`ExposureConstraintEstimator`](@ref) was written in the wrapped basis and must still be re-based. The right-hand side is untouched — a change of basis acts on the row, not on the bound.

`size(A, 2) == size(M, 2)` is checked rather than assumed, because a precomputed constraint carries no names and nothing else would catch an asset-space matrix handed to a [`FactorSpace`](@ref) wrapper whose asset and factor counts happen to differ.

# Algorithm

The [`PartialLinearConstraint`](@ref) method is the one that computes, and the [`LinearConstraint`](@ref) method is its fan-out over the two halves.

 1. Check that `plc.A` has one column per factor, `size(plc.A, 2) == size(M, 2)`.
 2. Return a [`PartialLinearConstraint`](@ref) whose `A` is the re-based coefficient matrix and whose `B` is `plc.B`, taken over unchanged.
 3. For a whole [`LinearConstraint`](@ref), apply steps 1 and 2 to `ineq` and to `eq`, skipping a half that is `nothing`, and rebuild the constraint from the two.

# Arguments

  - `lc`: The assembled [`LinearConstraint`](@ref), written over the factors.
  - `plc`: One half of it, a [`PartialLinearConstraint`](@ref).
  - `M`: The loadings, `Na × Nf`.

# Validation

  - `size(plc.A, 2) == size(M, 2)`. Otherwise a `DimensionMismatch` names both counts.

# Returns

  - `lc::LinearConstraint` or `plc::PartialLinearConstraint`: The same shape, over the assets. `A` is re-based and `B` is the input's own.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
"""
function project_linear_constraint(plc::PartialLinearConstraint,
                                   M::MatNum)::PartialLinearConstraint
    @argcheck(size(plc.A, 2) == size(M, 2),
              DimensionMismatch("a precomputed constraint being re-based must have one column per factor. Got\nsize(A, 2) => $(size(plc.A, 2))\nsize(M, 2) => $(size(M, 2))"))
    return PartialLinearConstraint(; A = plc.A * transpose(M), B = plc.B)
end
function project_linear_constraint(lc::LinearConstraint, M::MatNum)::LinearConstraint
    ineq = isnothing(lc.ineq) ? nothing : project_linear_constraint(lc.ineq, M)
    eq = isnothing(lc.eq) ? nothing : project_linear_constraint(lc.eq, M)
    return LinearConstraint(; ineq = ineq, eq = eq)
end
"""
    rebase_linear_constraints(lce, sets::UniverseSets, basis, key; datatype, strict, bl_flag)

Re-base one wrapped shape. Dispatches on what [`ExposureConstraintEstimator`](@ref) is decorating:

  - A [`LinearConstraintEstimator`](@ref) parses and assembles against `key`, projecting each term as it goes. A `key === nothing` on the wrapped estimator resolves to the space's key — the factor axis of `rr` for [`FactorSpace`](@ref) — rather than to `sets.xkey`.
  - A precomputed [`LinearConstraint`](@ref) is projected wholesale by [`project_linear_constraint`](@ref).
  - A vector is mapped elementwise, matching what [`linear_constraints`](@ref) returns for a vector of estimators.

# Algorithm

The two single-shape methods reach the same row by two routes. The estimator route builds the row term by term, and the precomputed route projects a finished row. Measured on one mandate written both ways, the rows agree.

 1. For a [`LinearConstraintEstimator`](@ref), pick the key: `lce.key` when the wrapped estimator states one, and `key` otherwise.
 2. Call [`linear_constraints`](@ref) on `lce.val` against that key, with `basis` passed as `rr`. [`constraint_row_term`](@ref) then sums the selected columns of `basis.M` for each named term, which is the projection done one term at a time.
 3. For a precomputed [`LinearConstraint`](@ref), call [`project_linear_constraint`](@ref) on it with `basis.M`, which projects the assembled coefficient matrix in one product. `basis.M` is the only field of the basis either route reads, so the regression's intercept never enters.
 4. For a vector, apply steps 1 to 3 to each element in turn and return the results in the order of the input.

# Arguments

  - `lce`: The wrapped shape — a [`LinearConstraintEstimator`](@ref), a [`LinearConstraint`](@ref), or a vector of either.
  - `sets`: The declared universe the names resolve against.
  - `basis`: The loadings, as [`constraint_space_basis`](@ref) resolved them.
  - `key`: The key the names resolve against, when the wrapped estimator states none of its own.
  - `datatype`: Data type of the assembled row.
  - `strict`: If `true`, a name the universe does not resolve throws; if `false`, it warns and the term is dropped.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.

# Returns

  - `lc`: An asset-space [`LinearConstraint`](@ref), `nothing` when every row was dropped, or a vector of either in the order of `lce`.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`project_linear_constraint`](@ref)
  - [`constraint_row_term`](@ref): the per-term half of the projection, which the estimator route reaches through [`linear_constraints`](@ref).
"""
function rebase_linear_constraints(lce::LinearConstraintEstimator, sets::UniverseSets,
                                   basis::AbstractLoadingsRegressionResult,
                                   key::AbstractString; datatype::DataType = Float64,
                                   strict::Bool = false,
                                   bl_flag::Bool = false)::Option{<:LinearConstraint}
    return linear_constraints(lce.val, sets, ifelse(isnothing(lce.key), key, lce.key);
                              datatype = datatype, strict = strict, bl_flag = bl_flag,
                              rr = basis)
end
function rebase_linear_constraints(lce::LinearConstraint, ::UniverseSets,
                                   basis::AbstractLoadingsRegressionResult,
                                   ::AbstractString; kwargs...)::LinearConstraint
    return project_linear_constraint(lce, basis.M)
end
function rebase_linear_constraints(lce::VecLcE_Lc, sets::UniverseSets,
                                   basis::AbstractLoadingsRegressionResult,
                                   key::AbstractString; kwargs...)
    return [rebase_linear_constraints(lc, sets, basis, key; kwargs...) for lc in lce]
end
"""
    linear_constraints(lcs::ExposureConstraintEstimator, sets::UniverseSets;
                       datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false,
                       rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)

Generate the asset-space constraint a re-based one is equivalent to.

Validates the space's basis once via [`constraint_space_basis`](@ref), then re-bases the wrapped shape. What comes back is an ordinary [`LinearConstraint`](@ref) — or a vector of them, when a vector was wrapped — indistinguishable from one written in asset names, which is why nothing downstream of constraint generation needs to know a re-basis happened.

`rd` is the returns a space may refit its basis from. It is `nothing` here, which is the standalone route: a space whose `re` is an estimator throws rather than refitting, and the message names the fixes. See [`factor_space_regression`](@ref).

# Arguments

  - `lcs`: The [`ExposureConstraintEstimator`](@ref) whose rows are re-based.
  - `sets`: The declared universe, carrying the factor axis `rr` names, under `sets.tfkey` or `sets.cfkey`.
  - `datatype`: Data type of the assembled row.
  - `strict`: If `true`, a name the universe does not resolve throws; if `false`, it warns and the term is dropped.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
  - `rr`: The loadings, when the caller holds them and the space states none.
  - `rd`: Returns the space may refit from. `nothing` on this route.

# Returns

  - `lc`: An asset-space [`LinearConstraint`](@ref), `nothing` when every row was dropped, or a vector of either when a vector was wrapped.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`constraint_space_basis`](@ref)
  - [`factor_space_regression`](@ref)
  - [`rebase_linear_constraints`](@ref)
"""
function linear_constraints(lcs::ExposureConstraintEstimator, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)
    basis, key = constraint_space_basis(lcs.space, sets, rr, rd)
    return rebase_linear_constraints(lcs.lce, sets, basis, key; datatype = datatype,
                                     strict = strict, bl_flag = bl_flag)
end
"""
    linear_constraints(lcs::VecEcE_LcE_Lc, sets::UniverseSets; datatype::DataType = Float64,
                       strict::Bool = false, bl_flag::Bool = false,
                       rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)

Broadcast over a vector that may mix re-based and asset-space constraints, forwarding the loadings and the returns to each. The narrower `VecLcE` method still wins for a vector that holds only [`LinearConstraintEstimator`](@ref)s.

Each element resolves its own basis, so a vector may mix a space that reads the prior with one that states or refits its own.

# Arguments

  - `lcs`: The vector of shapes, re-based and asset-space mixed.
  - `sets`: The declared universe the names resolve against.
  - `datatype`: Data type of the assembled rows.
  - `strict`: If `true`, a name the universe does not resolve throws; if `false`, it warns and the term is dropped.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
  - `rr`: The loadings, forwarded to every element.
  - `rd`: Returns an element's space may refit from, forwarded to every element.

# Returns

  - `lcs`: One result per entry of the input, in the order of the input. An entry is a [`LinearConstraint`](@ref) or `nothing`.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::VecEcE_LcE_Lc, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)
    return [linear_constraints(lc, sets; datatype = datatype, strict = strict,
                               bl_flag = bl_flag, rr = rr, rd = rd) for lc in lcs]
end

export FactorSpace, ExposureConstraintEstimator
