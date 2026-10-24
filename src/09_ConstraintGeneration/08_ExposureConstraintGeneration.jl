"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the bases that a linear constraint can be re-based into.

A re-basis applies to a constraint if and only if the constraint is a linear form in the weights. Under a change of basis ``\\boldsymbol{w}_b = \\mathbf{P}^\\intercal \\boldsymbol{w}``, the row ``\\boldsymbol{a}`` becomes ``\\mathbf{P}\\boldsymbol{a}``, and nothing else in the problem changes.

The limit comes from the mechanism. A re-basis rewrites a row and does not change the model. A constraint that enters the model through its own variables is outside the mechanism, even when a change of basis of the constrained quantity is well defined. These examples show the rule, and the list is not complete:

  - Cardinality, sub-group cardinality and buy-in thresholds index the binary held-indicators, not the weights.
  - The weight bounds of each asset need no counterpart in another basis. ``lb \\leq \\mathbf{P}^\\intercal\\boldsymbol{w} \\leq ub`` is a linear constraint, and this mechanism already takes it.
  - Turnover and tracking error are norm forms. ``\\lVert \\mathbf{P}^\\intercal(\\boldsymbol{w} - \\boldsymbol{w}_0) \\rVert`` has a meaning, but it needs its own variables and cones. Its re-basis is possible in mathematics, but not with this mechanism.
  - A fee is a price per traded position, which the model subtracts from the return. It is not a constraint on the weights.

Each absence is a property of the constraint, not a gap in the code.

The tracking of a factor needs no re-basis. [`ReturnsTracking`](@ref) takes a benchmark return series, and the return series of a factor is a column of the factor matrix. Pass that column directly.

The family has one member now, [`FactorSpace`](@ref). The abstract type lets the family get more members. Any linear change of basis in the weights can be a member and use the same code path, for example a currency, a sector, or an exposure table from the caller.

The family has no `AssetSpace`, by design. The asset frame is the absence of a re-basis, and a bare [`LinearConstraintEstimator`](@ref) states it. An `AssetSpace` member would make [`ExposureConstraintEstimator`](@ref) a wrapper that changes nothing and returns the rows of the constraint it wraps, bit for bit.

# Interfaces

To make a new basis, subtype `AbstractConstraintSpace` and put the source of its change of basis in its fields. Then implement the methods below.

## Basis resolution

  - `PortfolioOptimisers.constraint_space_basis(space::AbstractConstraintSpace, sets::UniverseSets, rr, rd) -> (basis, key)`: The change of basis through which the rows project, and the key against which their names resolve.

There is no fallback. This method makes every check that the basis needs, so the assembly loop after it can assume a consistent basis. A space that checks nothing still implements the method, and its docstring states that it checks nothing.

### Arguments

  - `space`: The basis.
  - `sets`: The declared universe the names resolve against.
  - `rr`: The loadings of the prior, or `nothing`.
  - `rd`: Returns from which the space can refit, or `nothing`. A space that refits from `rd` states the source of the refit on its own reader, for example in the `x_src` field of a regression estimator.

### Returns

  - `basis`: The change of basis, in the form that `rebase_linear_constraints` reads for this space.
  - `key::AbstractString`: The key of `sets.dict` against which the names of the row resolve.

## Asset view

  - `PortfolioOptimisers.port_opt_view(space::AbstractConstraintSpace, i, args...) -> AbstractConstraintSpace`: A copy of the space over the selected assets.

A basis has the assets on one axis and the names of the space on the other. The generic fallback returns an algorithm unchanged, so it is wrong for every member of this family. [`FactorSpace`](@ref) has the reference method. It slices the loadings by asset and keeps the whole factor axis.

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

The factor basis. It re-bases a constraint written in factor names through the loadings of a regression.

The field `re` holds the source of the basis. A precomputed [`Regression`](@ref) states the basis, an estimator refits it from the returns, and `nothing` reads the loadings of the prior. [`factor_space_regression`](@ref) ranks the three sources and holds the order.

!!! warning

    A precomputed `re` does not refit. The rows use exactly the loadings that the caller gave, whatever data the problem is solved on later. Constraint generation cannot see that it runs inside a cross-validation fold. The loadings have the correct shape for the full universe, so the projection is stale with no error and no warning. Two forms already give a basis that changes. The first is `re = <an estimator>`, which refits per fold and per subproblem. The second is a time-dependent schedule on `lcse`, which is already [`TD_Option`](@ref) and can hold a different basis per period. A [`NestedClustered`](@ref) outer solve refuses a stated result. That solve replaces the universe with cluster names and does not slice it, so no view of the loadings can follow it.

# Mathematical definition

```math
\\begin{align}
  \\boldsymbol{w}_f &= \\mathbf{M}^\\intercal \\boldsymbol{w} \\\\
  \\boldsymbol{a}^\\intercal \\boldsymbol{w}_f &= (\\mathbf{M}\\boldsymbol{a})^\\intercal \\boldsymbol{w} \\\\
  \\mathbf{A}\\boldsymbol{w}_f \\leq \\boldsymbol{b} &\\iff \\left(\\mathbf{A}\\mathbf{M}^\\intercal\\right)\\boldsymbol{w} \\leq \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{M}``: ``Na \\times Nf`` matrix of loadings, the `M` field of a [`Regression`](@ref).
  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_f``: ``Nf \\times 1`` vector of factor exposures.
  - ``\\boldsymbol{a}``: ``Nf \\times 1`` row that the user wrote in factor names.
  - ``\\mathbf{A}``: ``Nc \\times Nf`` coefficient matrix, with one row per constraint.
  - ``\\boldsymbol{b}``: ``Nc \\times 1`` right-hand side of those constraints.

The third line is the complete re-basis, and it states two facts. The first fact is that the product of the coefficient matrix and ``\\mathbf{M}^\\intercal`` turns a constraint over the factors into a constraint over the assets. The second fact is that ``\\boldsymbol{b}`` is the same on both sides, because a change of basis acts on the row and never on the bound. An equality has the same form, with ``=`` in place of ``\\leq``.

A factor exposure is ``\\mathbf{M}^\\intercal\\boldsymbol{w}`` and nothing more. Its definition does not use the intercept that a [`Regression`](@ref) also holds, and no line above contains that intercept. The ``\\boldsymbol{b}`` above is the right-hand side of the constraints, a different quantity from the intercept.

The names resolve against the factor axis that the loadings name. [`factor_axis_key`](@ref) reads `sets.tfkey` for a [`Regression`](@ref) and `sets.cfkey` for a [`CrossSectionalFactorModel`](@ref). Thus a mandate written for one family never resolves against the names of the other.

The loadings are `rr.M`, not `rr.L`. The columns of `M` are the named original factors, and a user must write a constraint in names that can go in an equation. Under [`DimensionReductionRegression`](@ref), the columns of `L` are principal components. Risk decomposition reads `L`, and that is correct. `M` and `L` are the two sides of one projection, not two conventions that differ.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorSpace(; re::Option{<:RegE_Reg} = nothing) -> FactorSpace

Keywords correspond to the struct's fields.

## Validation

  - `re` has the type bound `Option{<:RegE_Reg}`, and the constructor makes no other check. A value outside the bound, for example a raw matrix of loadings, raises a `TypeError` from the keyword constructor. The resolution can read the source of every space that the constructor builds.

# Related

  - [`AbstractConstraintSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`Regression`](@ref)
  - [`factor_space_regression`](@ref): ranks `re` against the other two sources of loadings, and states the order.
  - [`resolve_factor_regression`](@ref): the same precedence, which [`FactorRiskContribution`](@ref) and [`FactorRiskBudgeting`](@ref) also use.

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.6. $(ref_dict[:no_original_source])
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

Wrapper that re-bases a linear constraint into another basis.

`ExposureConstraintEstimator` wraps a [`LinearConstraintEstimator`](@ref), a precomputed [`LinearConstraint`](@ref), or a vector of either. It also declares the basis in which the rows are written. It wraps and does not reimplement, so it reuses the equation parser, the checks of `val` and `key`, and the expansion of groups. The bound of `lce` holds every shape that `lcse` takes except this wrapper, so every such shape can take a re-basis.

The projection happens at generation time, and constraint generation gives an ordinary [`LinearConstraint`](@ref) over the assets. That constraint goes into the existing `lcsr` slot and the existing model assembly, and there is no second path for constraints. Near Optimal Centering, time-dependent schedules and every optimiser that uses [`JuMPOptimiser`](@ref) take factor exposure constraints with no change.

The re-basis is on this type, and not on [`LinearConstraintEstimator`](@ref), by design. That type also goes to `gcarde` and `sgcarde`, which build rows over the binary held-indicators. A projected row is not integral, and it is not an index into those indicators. The bounds of those two slots do not admit this wrapper. Thus the type system makes the category error impossible, and no check at run time is necessary.

An `ExposureConstraintEstimator` is also a [`Pipeline`](@ref) step without a wrapper. The step reads the `prior` slot for its basis, and writes an ordinary [`LinearConstraint`](@ref) over the assets into `constraints`. The factor names resolve against the `nf` axis that [`pipeline_asset_sets`](@ref) builds from `rd.nf`, so the axis and the loadings agree by construction. For a factor mandate, use the `lcse` field of a [`JuMPOptimiser`](@ref), not the step. The step projects once, against the prior that it read. The field projects again against the prior of the optimiser, in each fold. [`constraint_step_value`](@ref) states the trade-off in full.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExposureConstraintEstimator(;
        lce::LcE_Lc_VecLcE_Lc,
        space::AbstractConstraintSpace
    ) -> ExposureConstraintEstimator

Keywords correspond to the struct's fields. `space` has no default, and the caller must give it. A re-basis is a choice of the user, and the library never infers it.

## Validation

  - If `lce` is an `AbstractVector`, `!isempty(lce)`.

The constructor checks only that condition. Constraint generation makes every check that the basis needs. The names are on the factor axis and the basis comes from the prior, and one object cannot see both. [`constraint_space_basis`](@ref) makes those three checks and states them.

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
  - [`constraint_space_basis`](@ref): makes every check that the basis needs, at generation time.
  - [`constraint_step_value`](@ref): the pipeline step, and the reason that a constraint from the step stays fixed to the prior of the pipeline.
  - [`pipeline_asset_sets`](@ref): builds the factor axis against which the names of the step resolve.

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.6. $(ref_dict[:no_original_source])
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

Alias for a union of an [`ExposureConstraintEstimator`](@ref) and the types that it wraps.

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

Alias for an [`EcE_LcE_Lc`](@ref) or a vector of them.

It is the bound of `lcse`, the only slot that admits a re-basis. `gcarde` and `sgcarde` keep the narrower aliases built on [`LcE_Lc`](@ref). That difference is the enforcement that [`ExposureConstraintEstimator`](@ref) describes.

# Related

  - [`EcE_LcE_Lc`](@ref)
  - [`VecEcE_LcE_Lc`](@ref)
"""
const EcE_LcE_Lc_VecEcE_LcE_Lc = Union{<:EcE_LcE_Lc, <:VecEcE_LcE_Lc}
"""
    port_opt_view(space::FactorSpace, i, args...) -> FactorSpace

Return a copy of a [`FactorSpace`](@ref) over the selected assets.

Only the basis is data. The [`Regression`](@ref) method of [`port_opt_view`](@ref) slices `re` and keeps the rows of `M` at `i`. The loadings have one row per asset, so a subset of assets is a subset of rows, and the projection over the subset is exact. A space with `re === nothing` comes back unchanged. An estimator in `re` goes through its own view method, and a default estimator comes back as the same object.

A slice of the universe thus gives a slice of the space. A [`NestedClustered`](@ref) outer solve does not slice the universe. It writes cluster names over the asset universe, and no slice of the rows of the loadings can follow that. For that reason, [`assert_external_optimiser`](@ref) refuses a space whose `re` is a precomputed result.

# Arguments

  - `space`: The [`FactorSpace`](@ref) to view.
  - `i`: The asset index the view keeps.
  - `args...`: More positional arguments, which the method passes on to the view of `re`.

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

Return a copy of an [`ExposureConstraintEstimator`](@ref) over the selected assets.

The method passes the wrapped constraint through and views the space, and it does nothing more. The wrapped shape is written in the names of the space, so an asset index has no meaning for it. The basis of the space is the only part of the estimator that has an asset index.

The generic vector method of [`port_opt_view`](@ref) covers a [`VecEcE_LcE_Lc`](@ref) slot, mixed or not. It views each element in turn, and does not slice the list of constraints by asset index.

# Arguments

  - `ece`: The [`ExposureConstraintEstimator`](@ref) to view.
  - `i`: The asset index the view keeps.
  - `args...`: More positional arguments, which the method passes on to the view of the space.

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

Apply the precedence of [`resolve_factor_regression`](@ref) to the three sources of loadings of a [`FactorSpace`](@ref). Return `nothing` when no source holds loadings.

[`FactorRiskContribution`](@ref) uses the same precedence. At `x_src = nothing`, a stated result comes first, then the `rr` of the prior, then a refit. [`resolve_factor_regression`](@ref) gives the order for each other source. This function calls the same two parts as that function, [`source_loadings`](@ref) and [`refit_factor_regression`](@ref), and does not repeat them. The sources are different. Constraint generation receives the loadings as `rr`, which the caller already read from the prior. The prior arm reads that argument, not a field. Both `re` and `rd` are optional here, because a space that names no source and a route with no returns are both ordinary.

The function returns `nothing` and does not throw, by design. [`constraint_space_basis`](@ref) diagnoses the case with no basis, because it knows that it assembles a constraint.

# Algorithm

 1. Return `re` when it is a precomputed [`Regression`](@ref). The caller stated the basis, and no data is necessary.
 2. Otherwise return `rr` when `re` is `nothing`. This is the behaviour of `FactorSpace()`. When `rr` is also `nothing`, no source holds a basis, and [`constraint_space_basis`](@ref) writes the diagnosis.
 3. Otherwise `re` is an estimator. Return the loadings that [`source_loadings`](@ref) takes from `rr` under the source of `re`, when they are not `nothing`. At `x_src = nothing` the loadings are `rr`, and [`PriorSource`](@ref) refuses `rr === nothing`.
 4. Otherwise raise when `rd` is `nothing`. When `rd` is present, return `refit_factor_regression(re, rd)`. It refits the loadings from the returns that the source of `re` names, which are `rd.X`, or `rd.X - rd.B` under [`ExcessSource`](@ref).

Step 4 is a capability of the field, not a fallback. With `FactorSpace(; re = StepwiseRegression())`, a factor mandate works on a prior that holds no loadings.

# Arguments

  - `re::Option{<:RegE_Reg}`: The source of the basis that the space states, or `nothing` to read `rr`.
  - `rr::Option{<:AbstractLoadingsRegressionResult}`: The loadings of the prior, or `nothing`.
  - `rd::Option{<:ReturnsResult}`: Returns from which to refit, or `nothing` on the standalone route.

# Validation

  - The rule of [`source_loadings`](@ref). An estimator in `re` at `x_src = PriorSource()` with `rr === nothing` throws an `ArgumentError`.
  - An estimator in `re` that must refit with no `rd` throws an [`IsNothingError`](@ref) that names the three fixes. The refit needs the returns. A [`JuMPOptimiser`](@ref) and a [`Pipeline`](@ref) step pass the returns and can refit. The standalone [`linear_constraints`](@ref) route receives no returns and cannot refit.

# Returns

  - `rr::Option{<:AbstractLoadingsRegressionResult}`: The loadings of the source with the highest rank, or `nothing` when no source holds loadings.

# Related

  - [`constraint_space_basis`](@ref)
  - [`FactorSpace`](@ref)
  - [`resolve_factor_regression`](@ref)
  - [`source_loadings`](@ref)
  - [`refit_factor_regression`](@ref)
"""
function factor_space_regression(re::AbstractTimeSeriesRegressionEstimator,
                                 rr::Option{<:AbstractLoadingsRegressionResult},
                                 rd::Option{<:ReturnsResult})
    rr = source_loadings(returns_source(re), rr, re)
    if !isnothing(rr)
        return rr
    end
    @argcheck(!isnothing(rd),
              IsNothingError("a `FactorSpace` whose `re` is a regression *estimator* ($(nameof(typeof(re)))) must fit the loadings from returns, and this route carries none. `linear_constraints(ece, sets; rr = ...)` is the standalone entry point: it receives the loadings directly and never the data they would be fitted from.\nEither state the basis outright, `FactorSpace(; re = Regression(; M = ...))`, or pass the loadings as `rr` and leave `re === nothing`, or generate the constraint through a `JuMPOptimiser` or a `Pipeline` step, where the returns are in scope and the refit happens per fold."))
    return refit_factor_regression(re, rd)
end
# A stated result is the basis, and no stated source reads the loadings of the prior.
function factor_space_regression(re::AbstractLoadingsRegressionResult,
                                 ::Option{<:AbstractLoadingsRegressionResult},
                                 ::Option{<:ReturnsResult})
    return re
end
function factor_space_regression(::Nothing, rr::Option{<:AbstractLoadingsRegressionResult},
                                 ::Option{<:ReturnsResult})
    return rr
end
"""
    constraint_space_basis(space::AbstractConstraintSpace, sets::UniverseSets, rr, rd)
        -> (basis, key)

Resolve the basis through which a re-based constraint projects, and the key against which its names resolve. Throw if either is not usable.

This function makes every check that a space needs before the assembly of the first row, so the assembly loop can assume a consistent basis. A [`FactorSpace`](@ref) needs two checks. The loadings must come from one of the three sources that [`factor_space_regression`](@ref) ranks. The factor axis must be declared, and it must agree with the loadings on the number of factors. [`factor_universe`](@ref) makes the second check, and every other reader of the declared axis also uses it.

`rd` holds the returns from which the space can refit. The [`JuMPOptimiser`](@ref) and the [`Pipeline`](@ref) step pass it, and the standalone route passes `nothing`.

# Algorithm

 1. Resolve `rr`, the loadings, with [`factor_space_regression`](@ref) over the three sources `space.re`, `rr` and `rd`, under the source of `space.re`.
 2. Raise when `rr` is `nothing`. No source holds a basis, and no row can be assembled.
 3. Check the declared factor axis with [`factor_universe`](@ref), against `size(rr.M, 2)`. The call can raise, and the assembly reads no value that it returns.
 4. Return `rr` and the key of the factor axis that [`factor_axis_key`](@ref) reads for `rr`. The names of the row resolve against that key.

# Arguments

  - `space`: The [`FactorSpace`](@ref) whose basis the function resolves.
  - `sets`: The declared universe. It must hold the factor axis that `rr` names, under `sets.tfkey` or `sets.cfkey`.
  - `rr`: The loadings of the prior, or `nothing`.
  - `rd`: Returns from which the space can refit, or `nothing` on the standalone route.

# Validation

The names come from the factor axis, but the basis comes from the prior. The two can disagree in ways that no single object can check. Thus this function makes the three checks below, and the constructor makes none of them.

  - A missing basis throws, whatever the value of `strict`. `strict` controls unknown names, and an unknown name affects one row. The generator drops that row, and the rest is still the problem that the caller described. A missing regression is different, because no row can be built. A silent drop of all rows gives a feasible portfolio that looks plausible and has none of the requested exposure. "Missing" means that no source holds loadings. The space can give its own through the `re` field of [`FactorSpace`](@ref), so a factor mandate works on a prior with no factor block. Raises an [`IsNothingError`](@ref).
  - A missing factor axis throws. [`factor_universe`](@ref) raises a `KeyError` that names the key that [`factor_axis_key`](@ref) chose. The axis is optional on [`UniverseSets`](@ref), but a constraint written against the axis needs it.
  - `size(rr.M, 2) == length(sets.dict[key])`. Otherwise [`factor_universe`](@ref) raises a `DimensionMismatch`. The check of the names against `rd.nf` needs the returns, so the optimiser makes it.

# Returns

  - `basis::AbstractLoadingsRegressionResult`: The loadings through which the rows project.
  - `key::AbstractString`: The key against which the names of the row resolve. For a [`FactorSpace`](@ref) it is the [`factor_axis_key`](@ref) of `rr`.

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
              IsNothingError("a factor exposure constraint is written in factor names and re-based through the regression loadings, so it needs a source for them, and none of the three sources holds any: the space states none (`space.re === nothing`) and the prior carries none (`rr === nothing`). Unlike an unknown name, this is not recoverable per row and is not governed by `strict`: every row of the constraint would be dropped, leaving a feasible portfolio with none of the requested exposure.\nState the basis on the space instead, `FactorSpace(; re = Regression(; M = ...))` to pin it or `FactorSpace(; re = StepwiseRegression())` to refit it from the returns. $prior_regression_remedy"))
    key = factor_axis_key(sets, rr)
    factor_universe(sets, key, size(rr.M, 2), "a $(FactorSpace) constraint", "rr.M")
    return rr, key
end
"""
    project_linear_constraint(lc::LinearConstraint, M::MatNum; strict::Bool = false)
        -> Option{<:LinearConstraint}
    project_linear_constraint(plc::PartialLinearConstraint, M::MatNum; strict::Bool = false)
        -> Option{<:PartialLinearConstraint}
    project_linear_constraint(::Nothing, ::MatNum; kwargs...) -> Nothing

Re-base an assembled constraint by the projection of its coefficient matrix, `A * transpose(M)`.

The estimator route projects term by term, and this function is the half of the projection for a precomputed constraint. It is the one place where [`linear_constraints`](@ref) does more than pass a [`LinearConstraint`](@ref) through. A precomputed constraint that reaches the optimiser through an [`ExposureConstraintEstimator`](@ref) is written in the wrapped basis, so it still needs a re-basis. The right-hand side does not change, because a change of basis acts on the row and not on the bound.

The function checks `size(A, 2) == size(M, 2)` and does not assume it. A precomputed constraint holds no names. Without this check, a matrix over the assets given to a [`FactorSpace`](@ref) wrapper goes through when the numbers of assets and factors differ.

A re-based row whose coefficients are all zero does not depend on the weights, so the function drops it. The estimator route drops such a row in [`get_linear_constraints`](@ref) too. The loadings make such a row when every factor that the row weights has zero loadings.

# Algorithm

The [`PartialLinearConstraint`](@ref) method does the computation, and the [`LinearConstraint`](@ref) method applies it to the two halves.

 1. Check that `plc.A` has one column per factor, `size(plc.A, 2) == size(M, 2)`.
 2. Project the coefficient matrix, `A = plc.A * transpose(M)`.
 3. Report each row of `A` whose coefficients are all zero through [`strict_diagnostic`](@ref) with [`IsZeroError`](@ref), and drop it with its entry of `plc.B`.
 4. Return a [`PartialLinearConstraint`](@ref) of the rows that remain, with their entries of `plc.B` unchanged. Return `nothing` when no row remains.
 5. For a whole [`LinearConstraint`](@ref), apply steps 1 to 4 to `ineq` and to `eq`, and rebuild the constraint from the two. A half that is `nothing` stays `nothing`, through the `Nothing` method. Return `nothing` when both halves are `nothing`.

# Arguments

  - `lc`: The assembled [`LinearConstraint`](@ref), written over the factors.
  - `plc`: One half of it, a [`PartialLinearConstraint`](@ref).
  - `M`: The loadings, `Na × Nf`.
  - `strict`: If `true`, a re-based row whose coefficients are all zero raises an [`IsZeroError`](@ref). If `false`, it warns. The row is dropped either way.

# Validation

  - `size(plc.A, 2) == size(M, 2)`. Otherwise a `DimensionMismatch` names both counts.
  - A re-based row whose coefficients are all zero raises an [`IsZeroError`](@ref) when `strict` is `true`, and warns otherwise. The row is dropped either way.

# Returns

  - `lc::Option{<:LinearConstraint}` or `plc::Option{<:PartialLinearConstraint}`: The same shape over the assets, or `nothing` when the function dropped every row. `A` is re-based, and `B` is the `B` of the input.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
"""
function project_linear_constraint(plc::PartialLinearConstraint, M::MatNum;
                                   strict::Bool = false)::Option{<:PartialLinearConstraint}
    @argcheck(size(plc.A, 2) == size(M, 2),
              DimensionMismatch("a precomputed constraint being re-based must have one column per factor. Got\nsize(A, 2) => $(size(plc.A, 2))\nsize(M, 2) => $(size(M, 2))"))
    A = plc.A * transpose(M)
    keep = map(r -> any(!iszero, r), eachrow(A))
    foreach(i -> strict_diagnostic("row $(i) of $(size(A, 1)) of a precomputed constraint projected to an all-zero row over $(size(A, 2)) assets: every factor it weights has zero loadings; row dropped",
                                   strict; id = @log_id(), exception = IsZeroError),
            findall(!, keep))
    return any(keep) ? PartialLinearConstraint(; A = A[keep, :], B = plc.B[keep]) : nothing
end
function project_linear_constraint(::Nothing, ::MatNum; kwargs...)::Nothing
    return nothing
end
function project_linear_constraint(lc::LinearConstraint, M::MatNum;
                                   strict::Bool = false)::Option{<:LinearConstraint}
    ineq = project_linear_constraint(lc.ineq, M; strict = strict)
    eq = project_linear_constraint(lc.eq, M; strict = strict)
    return if isnothing(ineq) && isnothing(eq)
        nothing
    else
        LinearConstraint(; ineq = ineq, eq = eq)
    end
end
"""
    rebase_linear_constraints(lce, sets::UniverseSets, basis, key; datatype, strict, bl_flag)

Re-base one wrapped shape. The method depends on the shape that the [`ExposureConstraintEstimator`](@ref) wraps:

  - The method parses a [`LinearConstraintEstimator`](@ref) and assembles it against `key`, and projects each term during the assembly. When the wrapped estimator has `key === nothing`, the key of the space applies, not `sets.xkey`. For a [`FactorSpace`](@ref), that key is the factor axis of `rr`.
  - The method projects a precomputed [`LinearConstraint`](@ref) as a whole with [`project_linear_constraint`](@ref).
  - The method maps a vector element by element, in the same way as [`linear_constraints`](@ref) for a vector of estimators.

# Algorithm

The two single-shape methods reach the same row by two routes. The estimator route builds the row term by term, and the precomputed route projects a finished row.

 1. For a [`LinearConstraintEstimator`](@ref), pick the key: `lce.key` when the wrapped estimator states one, and `key` otherwise.
 2. Call [`linear_constraints`](@ref) on `lce.val` against that key, with `basis` passed as `rr`. [`constraint_row_term`](@ref) then sums the selected columns of `basis.M` for each named term, which is the projection done one term at a time.
 3. For a precomputed [`LinearConstraint`](@ref), call [`project_linear_constraint`](@ref) on it with `basis.M` and `strict`, which projects the assembled coefficient matrix in one product. `basis.M` is the only field of the basis either route reads, so the regression's intercept never enters.
 4. For a vector, apply steps 1 to 3 to each element in turn and return the results in the order of the input.

# Arguments

  - `lce`: The wrapped shape. It is a [`LinearConstraintEstimator`](@ref), a [`LinearConstraint`](@ref), or a vector of either.
  - `sets`: The declared universe the names resolve against.
  - `basis`: The loadings that [`constraint_space_basis`](@ref) resolved.
  - `key`: The key against which the names resolve, when the wrapped estimator states no key.
  - `datatype`: Data type of the assembled row.
  - `strict`: If `true`, a name that the universe does not resolve throws, and a row whose coefficients are all zero raises an [`IsZeroError`](@ref). If `false`, each case warns. The function drops the row in both cases.
  - `bl_flag`: If `true`, a group expands to the mean over its members, as the Black-Litterman views need. If `false`, it expands to the sum.

# Returns

  - `lc`: A [`LinearConstraint`](@ref) over the assets, `nothing` when the function dropped every row, or a vector of either in the order of `lce`.

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
                                   ::AbstractString; strict::Bool = false,
                                   kwargs...)::Option{<:LinearConstraint}
    return project_linear_constraint(lce, basis.M; strict = strict)
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

Make the constraint over the assets that is equivalent to a re-based constraint.

The method checks the basis of the space once with [`constraint_space_basis`](@ref), then re-bases the wrapped shape. The result is an ordinary [`LinearConstraint`](@ref), or a vector of them when the estimator wraps a vector. It is the same as a constraint written in asset names, and no step after constraint generation needs to know about the re-basis.

`rd` holds the returns from which a space can refit its basis. A [`JuMPOptimiser`](@ref) passes its returns. A standalone call keeps the default `nothing`, and then a space whose `re` must refit throws, with a message that names the fixes. See [`factor_space_regression`](@ref).

# Arguments

  - `lcs`: The [`ExposureConstraintEstimator`](@ref) whose rows the method re-bases.
  - `sets`: The declared universe. It holds the factor axis that `rr` names, under `sets.tfkey` or `sets.cfkey`.
  - `datatype`: Data type of the assembled row.
  - `strict`: If `true`, a name that the universe does not resolve throws, and a row whose coefficients are all zero raises an [`IsZeroError`](@ref). If `false`, each case warns. The function drops the row in both cases.
  - `bl_flag`: If `true`, a group expands to the mean over its members, as the Black-Litterman views need. If `false`, it expands to the sum.
  - `rr`: The loadings, when the caller holds them and the space states none.
  - `rd`: Returns from which the space can refit, or `nothing` on the standalone route.

# Returns

  - `lc`: A [`LinearConstraint`](@ref) over the assets, `nothing` when the function dropped every row, or a vector of either when the estimator wraps a vector.

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

Apply [`linear_constraints`](@ref) to each element of a vector that can mix re-based constraints and constraints over the assets. The method passes the loadings and the returns to each element. The narrower `VecLcE` method still applies to a vector that holds only [`LinearConstraintEstimator`](@ref)s.

Each element resolves its own basis, so one vector can hold a space that reads the prior and a space that states or refits its own basis.

# Arguments

  - `lcs`: The vector of shapes, a mix of re-based constraints and constraints over the assets.
  - `sets`: The declared universe the names resolve against.
  - `datatype`: Data type of the assembled rows.
  - `strict`: If `true`, a name that the universe does not resolve throws, and a row whose coefficients are all zero raises an [`IsZeroError`](@ref). If `false`, each case warns. The function drops the row in both cases.
  - `bl_flag`: If `true`, a group expands to the mean over its members, as the Black-Litterman views need. If `false`, it expands to the sum.
  - `rr`: The loadings, which the method passes to every element.
  - `rd`: Returns from which the space of an element can refit, which the method passes to every element.

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
"""
    reader_frames(space::FactorSpace, path::AbstractString, refit::Bool)
    reader_frames(ece::ExposureConstraintEstimator, path::AbstractString, refit::Bool)

Return the frames that the regression reader of a constraint space adds to the group of the optimiser that holds the constraint.

A [`FactorSpace`](@ref) holds its reader in `re`, and an [`ExposureConstraintEstimator`](@ref) holds the space in `space`. Each method reads the field at its path, so the warning of [`warn_raw_readers`](@ref) names a reader with its path, for example `"lcse.space.re"`. A precomputed [`Regression`](@ref) and `re = nothing` are not readers, and they add no frames.

# Arguments

  - `space`, `ece`: The constraint space, or the constraint estimator that holds it.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the reader in the field.

# Related

  - [`reader_frames`](@ref)
  - [`FactorSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
function reader_frames(space::FactorSpace, path::AbstractString, refit::Bool)
    return reader_frames(space.re, join_path(path, :re), refit)
end
function reader_frames(ece::ExposureConstraintEstimator, path::AbstractString, refit::Bool)
    return reader_frames(ece.space, join_path(path, :space), refit)
end

export FactorSpace, ExposureConstraintEstimator
public AbstractConstraintSpace, constraint_space_basis
