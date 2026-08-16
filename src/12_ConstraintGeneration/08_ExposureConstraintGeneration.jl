"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the bases a linear constraint can be re-based into.

A constraint can be re-based into another basis **if and only if it is a linear form in the weights**. Under a change of basis `\\boldsymbol{w}_b = \\mathbf{P}^\\intercal \\boldsymbol{w}`, the row `\\boldsymbol{a}` becomes `\\mathbf{P}\\boldsymbol{a}`, and nothing else about the problem changes.

The boundary is a property of the **mechanism**: a re-basis rewrites a row and leaves the model untouched, so a constraint that reaches the model through its own variables is outside it even where a change of basis is well defined for the quantity constrained. Cardinality, sub-group cardinality and buy-in thresholds index the binary held-indicators rather than the weights. A per-asset weight box has no counterpart because `lb \\leq \\mathbf{P}^\\intercal\\boldsymbol{w} \\leq ub` *is* a linear constraint and already has a home. Turnover and tracking error are norm forms: `\\lVert \\mathbf{P}^\\intercal(\\boldsymbol{w} - \\boldsymbol{w}_0) \\rVert` is meaningful, but it needs its own variables and cones, so it is re-basable in mathematics and not by this mechanism. A fee is priced per traded position and is subtracted from the return, so it is not a constraint on the weights at all. The list illustrates the rule rather than exhausting it, and these absences are a property of the constraint, not a gap in the implementation; see ADR 0047 and its 2026-08-16 amendment.

Tracking a factor needs no re-basis: [`ReturnsTracking`](@ref) takes a benchmark *return series*, and a factor's return series is a column of the factor matrix, so it is passed directly.

The family has one member today, [`FactorSpace`](@ref), and the abstract type exists so it can gain more: any linear change of basis in the weights — currency, sector, a caller-supplied exposure table — lands here without a new pathway.

There is deliberately **no `AssetSpace`**. The asset frame is the *absence* of a re-basis, spelled by using a bare [`LinearConstraintEstimator`](@ref); an `AssetSpace` member would make [`ExposureConstraintEstimator`](@ref) a no-op decorator computing bit-for-bit what it wraps.

# Related

  - [`FactorSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
abstract type AbstractConstraintSpace <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The factor basis: a constraint written in factor names, re-based through a regression's loadings.

# Mathematical definition

```math
\\begin{align}
  \\boldsymbol{w}_f &= \\mathbf{M}^\\intercal \\boldsymbol{w} \\\\
  \\boldsymbol{a}^\\intercal \\boldsymbol{w}_f &= (\\mathbf{M}\\boldsymbol{a})^\\intercal \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - `\\mathbf{M}`: `Na×Nf` loadings matrix, [`Regression`](@ref)'s `M`.
  - `\\boldsymbol{w}`: `Na×1` asset weights.
  - `\\boldsymbol{w}_f`: `Nf×1` factor weights (exposures).
  - `\\boldsymbol{a}`: `Nf×1` row the user wrote in factor names.

The names resolve against the declared factor axis, `sets.dict[sets.fkey]`, and the loadings are `rr.M` rather than `rr.L` — `M`'s columns are the named original factors, and a constraint must be *written* in names a user can put in an equation, whereas `L`'s columns are principal components under [`DimensionReductionRegression`](@ref). Risk decomposition reads `L` and is correct to; the two are the two sides of one projection, not a drift between conventions.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorSpace(; re::Option{<:RegE_Reg} = nothing) -> FactorSpace

Keywords correspond to the struct's fields.

# Where the loadings come from

A space carries the source of its own basis. `re` is that source, and its precedence is [`resolve_factor_regression`](@ref)'s, shared with [`FactorRiskContribution`](@ref) and [`FactorRiskBudgeting`](@ref):

 1. `re` when it is a precomputed [`Regression`](@ref). The caller has stated the basis, and no data is needed.
 2. The prior's `rr`, when the prior carries a factor block. This is what `re === nothing` always resolves to, and it is the behaviour of every `FactorSpace` written before the field existed.
 3. `regression(re, rd)` otherwise — an estimator refits the loadings from the returns.

Arm 3 is a capability the field adds rather than a fallback: `FactorSpace(; re = StepwiseRegression())` is **a factor mandate on a prior that carries no loadings**, which previously threw. It needs `rd`, so it is available at a [`JuMPOptimiser`](@ref) and as a [`Pipeline`](@ref) step, and refused on the standalone [`linear_constraints`](@ref) route, which receives no returns.

!!! warning

    A **precomputed** `re` does not refit. The rows are re-based through exactly the loadings handed in, whatever data the problem is later solved on, and nothing at generation time can see that it is inside a cross-validation fold: the projection is silently stale rather than wrong-shaped, because the loadings are the right shape for the full universe. Two spellings already exist for a basis that must move — `re = <an estimator>`, which refits per fold and per subproblem, and a time-dependent schedule on `lcse`, which is already [`TD_Option`](@ref) and can carry a different basis per period. A stated result is also refused outright at a [`NestedClustered`](@ref) outer solve, where the universe is *replaced* by cluster names rather than sliced, so no view of the loadings can follow it.

# Related

  - [`AbstractConstraintSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`Regression`](@ref)
  - [`resolve_factor_regression`](@ref)
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

# Failure modes

The names are looked up on the factor axis, but the *basis* comes from the prior, so the two can disagree in ways a single object cannot check. All three are checked when the constraint is generated:

  - **Missing loadings throw, ignoring `strict`.** `strict` governs unknown *names*: a per-row, recoverable condition where the offending row is dropped and the rest of the problem is still the problem the caller described. A missing regression is not that — it makes every row unbuildable, and dropping them silently yields a feasible, plausible-looking portfolio carrying none of the requested exposure. "Missing" means no carrier holds any: the space can supply its own through [`FactorSpace`](@ref)'s `re`, which is what makes a factor mandate legal on a prior with no factor block.
  - **A missing factor axis throws**, naming `fkey`. The axis is optional on [`UniverseSets`](@ref); it is not optional for a constraint written against it.
  - **`size(rr.M, 2) == length(sets.dict[fkey])` always.** The name-level cross-check against `rd.nf` needs the returns and lives at the optimiser.

# As a pipeline step

An `ExposureConstraintEstimator` is also usable as a bare [`Pipeline`](@ref) step, where it reads the `prior` slot for its basis and writes an ordinary asset-space [`LinearConstraint`](@ref) into `constraints`. The factor names resolve against the `nf` axis [`pipeline_asset_sets`](@ref) builds from `rd.nf`, so the axis and the loadings agree by construction.

!!! warning

    A step-generated constraint is **pinned to the pipeline's prior**. The projection happens once, when the step runs, and a downstream optimiser that refits its own prior receives rows computed against the loadings the *step* saw — under [`cross_val_predict`](@ref) that is the loadings of whatever prior step preceded it in the same fold, which is right only if the optimiser shares that prior. Passing the estimator to a [`JuMPOptimiser`](@ref)'s `lcse` field instead re-projects it with the optimiser's own prior, and is the default advice for a factor mandate. This is the same trade-off a phylogeny constraint step already makes.

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
    port_opt_view(lcse::VecEcE_LcE_Lc, i, args...) -> AbstractVector

Return an asset-sliced copy of the `lcse` slot.

The wrapped constraint is passed through and the **space** is viewed. That split is the whole content of the method: the wrapped shape is written in the space's names, so an asset index means nothing to it, while the space's basis is the one thing in the slot that is indexed by asset.

The vector method exists because the universal fallback would `view` the vector itself, slicing a list of constraints by asset indices. It covers every vector shape `lcse` admits, mixed or not.

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
function port_opt_view(lcse::VecEcE_LcE_Lc, i, args...)
    return [port_opt_view(lc, i, args...) for lc in lcse]
end
"""
    factor_space_regression(re::Option{<:RegE_Reg}, rr::Option{<:AbstractRegressionResult},
                            rd::Option{<:ReturnsResult})
        -> Option{<:AbstractRegressionResult}

Apply [`resolve_factor_regression`](@ref)'s precedence to the three carriers a [`FactorSpace`](@ref) can read its loadings from, and return `nothing` when none of them holds any.

The precedence is the one [`FactorRiskContribution`](@ref) already uses — a stated result, then the prior's own `rr`, then a refit — and the refit arm delegates to [`resolve_factor_regression`](@ref) rather than repeating it. What differs is the carriers: constraint generation is handed the loadings as `rr`, already read off the prior, so the prior arm is a plain argument rather than a field read, and both `re` and `rd` are optional here because a space that names no source and a route that carries no returns are both ordinary.

Returning `nothing` rather than throwing is deliberate: the diagnosis for "no basis anywhere" belongs to [`constraint_space_basis`](@ref), which knows it is assembling a constraint.

# Arguments

  - `re::Option{<:RegE_Reg}`: The space's stated basis source, or `nothing` to read `rr`.
  - `rr::Option{<:AbstractRegressionResult}`: The prior's loadings, or `nothing`.
  - `rd::Option{<:ReturnsResult}`: Returns to refit from, or `nothing` on the standalone route.

# Validation

  - An estimator in `re` with no `rd` throws an [`IsNothingError`](@ref) naming both fixes.

# Related

  - [`constraint_space_basis`](@ref)
  - [`FactorSpace`](@ref)
  - [`resolve_factor_regression`](@ref)
"""
function factor_space_regression(re::Option{<:RegE_Reg},
                                 rr::Option{<:AbstractRegressionResult},
                                 rd::Option{<:ReturnsResult})
    if isa(re, AbstractRegressionResult)
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

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`FactorSpace`](@ref)
  - [`factor_space_regression`](@ref)
  - [`factor_universe`](@ref)
"""
function constraint_space_basis(space::FactorSpace, sets::UniverseSets,
                                rr::Option{<:AbstractRegressionResult},
                                rd::Option{<:ReturnsResult} = nothing)
    rr = factor_space_regression(space.re, rr, rd)
    @argcheck(!isnothing(rr),
              IsNothingError("a factor exposure constraint is written in factor names and re-based through the regression loadings, so it needs a source for them, and none of the three carriers holds any: the space states none (`space.re === nothing`) and the prior carries none (`rr === nothing`). Unlike an unknown name, this is not recoverable per row and is not governed by `strict`: every row of the constraint would be dropped, leaving a feasible portfolio with none of the requested exposure.\nState the basis on the space instead, `FactorSpace(; re = Regression(; M = ...))` to pin it or `FactorSpace(; re = StepwiseRegression())` to refit it from the returns. $prior_regression_remedy"))
    factor_universe(sets, size(rr.M, 2), "a $(FactorSpace) constraint", "rr.M")
    return rr, sets.fkey
end
"""
    project_linear_constraint(lc::LinearConstraint, M::MatNum) -> LinearConstraint
    project_linear_constraint(plc::PartialLinearConstraint, M::MatNum) -> PartialLinearConstraint

Re-base an **already assembled** constraint by projecting its coefficient matrix, `A * transpose(M)`.

This is the precomputed-constraint half of the estimator path's per-term projection, and it is the one place [`linear_constraints`](@ref) stops being a pass-through for a [`LinearConstraint`](@ref): a precomputed constraint reaching the optimiser through an [`ExposureConstraintEstimator`](@ref) was written in the wrapped basis and must still be re-based. The right-hand side is untouched — a change of basis acts on the row, not on the bound.

`size(A, 2) == size(M, 2)` is checked rather than assumed, because a precomputed constraint carries no names and nothing else would catch an asset-space matrix handed to a [`FactorSpace`](@ref) wrapper whose asset and factor counts happen to differ.

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

  - A [`LinearConstraintEstimator`](@ref) parses and assembles against `key`, projecting each term as it goes. A `key === nothing` on the wrapped estimator resolves to the space's key — `sets.fkey` for [`FactorSpace`](@ref) — rather than to `sets.xkey`.
  - A precomputed [`LinearConstraint`](@ref) is projected wholesale by [`project_linear_constraint`](@ref).
  - A vector is mapped elementwise, matching what [`linear_constraints`](@ref) returns for a vector of estimators.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`project_linear_constraint`](@ref)
"""
function rebase_linear_constraints(lce::LinearConstraintEstimator, sets::UniverseSets,
                                   basis::AbstractRegressionResult, key::AbstractString;
                                   datatype::DataType = Float64, strict::Bool = false,
                                   bl_flag::Bool = false)::Option{<:LinearConstraint}
    return linear_constraints(lce.val, sets, ifelse(isnothing(lce.key), key, lce.key);
                              datatype = datatype, strict = strict, bl_flag = bl_flag,
                              rr = basis)
end
function rebase_linear_constraints(lce::LinearConstraint, ::UniverseSets,
                                   basis::AbstractRegressionResult, ::AbstractString;
                                   kwargs...)::LinearConstraint
    return project_linear_constraint(lce, basis.M)
end
function rebase_linear_constraints(lce::VecLcE_Lc, sets::UniverseSets,
                                   basis::AbstractRegressionResult, key::AbstractString;
                                   kwargs...)
    return [rebase_linear_constraints(lc, sets, basis, key; kwargs...) for lc in lce]
end
"""
    linear_constraints(lcs::ExposureConstraintEstimator, sets::UniverseSets;
                       datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false,
                       rr::Option{<:AbstractRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)

Generate the asset-space constraint a re-based one is equivalent to.

Validates the space's basis once via [`constraint_space_basis`](@ref), then re-bases the wrapped shape. What comes back is an ordinary [`LinearConstraint`](@ref) — or a vector of them, when a vector was wrapped — indistinguishable from one written in asset names, which is why nothing downstream of constraint generation needs to know a re-basis happened.

`rd` is the returns a space may refit its basis from. It is `nothing` here, which is the standalone route: a space whose `re` is an estimator throws rather than refitting, and the message names the fixes. See [`factor_space_regression`](@ref).

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`constraint_space_basis`](@ref)
  - [`factor_space_regression`](@ref)
  - [`rebase_linear_constraints`](@ref)
"""
function linear_constraints(lcs::ExposureConstraintEstimator, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)
    basis, key = constraint_space_basis(lcs.space, sets, rr, rd)
    return rebase_linear_constraints(lcs.lce, sets, basis, key; datatype = datatype,
                                     strict = strict, bl_flag = bl_flag)
end
"""
    linear_constraints(lcs::VecEcE_LcE_Lc, sets::UniverseSets; datatype::DataType = Float64,
                       strict::Bool = false, bl_flag::Bool = false,
                       rr::Option{<:AbstractRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)

Broadcast over a vector that may mix re-based and asset-space constraints, forwarding the loadings and the returns to each. The narrower `VecLcE` method still wins for a vector that holds only [`LinearConstraintEstimator`](@ref)s.

Each element resolves its own basis, so a vector may mix a space that reads the prior with one that states or refits its own.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::VecEcE_LcE_Lc, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)
    return [linear_constraints(lc, sets; datatype = datatype, strict = strict,
                               bl_flag = bl_flag, rr = rr, rd = rd) for lc in lcs]
end

export AbstractConstraintSpace, FactorSpace, ExposureConstraintEstimator
