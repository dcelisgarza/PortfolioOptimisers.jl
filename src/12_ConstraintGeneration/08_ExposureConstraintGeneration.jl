"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the bases a linear constraint can be re-based into.

A constraint can be re-based into another basis **if and only if it is a linear form in the weights**. Under a change of basis `\\boldsymbol{w}_b = \\mathbf{P}^\\intercal \\boldsymbol{w}`, the row `\\boldsymbol{a}` becomes `\\mathbf{P}\\boldsymbol{a}`, and nothing else about the problem changes. Constraints that index the binary held-indicators rather than the weights — cardinality, sub-group cardinality, buy-in thresholds — have no counterpart, and a per-asset weight box has none either, because `lb \\leq \\mathbf{P}^\\intercal\\boldsymbol{w} \\leq ub` *is* a linear constraint and already has a home. Their absence is a property of the constraint, not a gap in the implementation; see ADR 0047.

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

# Related

  - [`AbstractConstraintSpace`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
  - [`Regression`](@ref)
"""
struct FactorSpace <: AbstractConstraintSpace end
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

  - **Missing loadings throw, ignoring `strict`.** `strict` governs unknown *names*: a per-row, recoverable condition where the offending row is dropped and the rest of the problem is still the problem the caller described. A missing regression is not that — it makes every row unbuildable, and dropping them silently yields a feasible, plausible-looking portfolio carrying none of the requested exposure.
  - **A missing factor axis throws**, naming `fkey`. The axis is optional on [`UniverseSets`](@ref); it is not optional for a constraint written against it.
  - **`size(rr.M, 2) == length(sets.dict[fkey])` always.** The name-level cross-check against `rd.nf` needs the returns and lives at the optimiser.

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
    constraint_space_basis(space::AbstractConstraintSpace, sets::UniverseSets, rr)
        -> (basis, key)

Resolve the basis a re-based constraint projects through, and the key its names resolve against, throwing if either is unusable.

Every check a space needs before a single row is assembled lives here, so the assembly loop can assume a consistent basis. For [`FactorSpace`](@ref) that is: the loadings exist, the factor axis is declared, and the two agree on how many factors there are.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`FactorSpace`](@ref)
"""
function constraint_space_basis(::FactorSpace, sets::UniverseSets,
                                rr::Option{<:AbstractRegressionResult})
    @argcheck(!isnothing(rr),
              IsNothingError("a factor exposure constraint is written in factor names and re-based through the regression loadings, so it needs a prior that carries one, but `rr === nothing`. Unlike an unknown name, this is not recoverable per row and is not governed by `strict`: every row of the constraint would be dropped, leaving a feasible portfolio with none of the requested exposure. $prior_regression_remedy"))
    fkey = sets.fkey
    @argcheck(haskey(sets.dict, fkey),
              KeyError("$fkey (the factor universe), required by a $(FactorSpace) constraint. The factor axis is optional on UniverseSets, but a constraint written against it is not: add `sets.fkey => <factor names>` to `sets.dict`, in the column order of the loadings."))
    nf = sets.dict[fkey]
    @argcheck(size(rr.M, 2) == length(nf),
              DimensionMismatch("the loadings and the declared factor axis disagree on how many factors there are. Got\nsize(rr.M, 2) => $(size(rr.M, 2))\nlength(sets.dict[$fkey]) => $(length(nf))"))
    return rr, fkey
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
                       rr::Option{<:AbstractRegressionResult} = nothing)

Generate the asset-space constraint a re-based one is equivalent to.

Validates the space's basis once via [`constraint_space_basis`](@ref), then re-bases the wrapped shape. What comes back is an ordinary [`LinearConstraint`](@ref) — or a vector of them, when a vector was wrapped — indistinguishable from one written in asset names, which is why nothing downstream of constraint generation needs to know a re-basis happened.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`constraint_space_basis`](@ref)
  - [`rebase_linear_constraints`](@ref)
"""
function linear_constraints(lcs::ExposureConstraintEstimator, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractRegressionResult} = nothing)
    basis, key = constraint_space_basis(lcs.space, sets, rr)
    return rebase_linear_constraints(lcs.lce, sets, basis, key; datatype = datatype,
                                     strict = strict, bl_flag = bl_flag)
end
"""
    linear_constraints(lcs::VecEcE_LcE_Lc, sets::UniverseSets; datatype::DataType = Float64,
                       strict::Bool = false, bl_flag::Bool = false,
                       rr::Option{<:AbstractRegressionResult} = nothing)

Broadcast over a vector that may mix re-based and asset-space constraints, forwarding the loadings to each. The narrower `VecLcE` method still wins for a vector that holds only [`LinearConstraintEstimator`](@ref)s.

# Related

  - [`ExposureConstraintEstimator`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::VecEcE_LcE_Lc, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractRegressionResult} = nothing)
    return [linear_constraints(lc, sets; datatype = datatype, strict = strict,
                               bl_flag = bl_flag, rr = rr) for lc in lcs]
end

export AbstractConstraintSpace, FactorSpace, ExposureConstraintEstimator
