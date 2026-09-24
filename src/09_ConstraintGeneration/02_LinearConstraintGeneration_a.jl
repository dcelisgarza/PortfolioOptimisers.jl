"""
$(DocStringExtensions.TYPEDEF)

Holds the coefficient matrix `A` and the right-hand side vector `B` of one half of a linear constraint block.

The half is an inequality or an equality according to the field of [`LinearConstraint`](@ref) that carries it, `ineq` or `eq`, and [`LinearConstraint`](@ref) states the form of each half. One row of `A` and the entry of `B` beside it are one constraint, so a pair holding more bounds than rows, or more rows than bounds, is satisfied by no value of the constrained variable.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PartialLinearConstraint(;
        A::MatNum,
        B::VecNum
    ) -> PartialLinearConstraint

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:A])
  - $(val_dict[:B])
  - $(val_dict[:A_B])

# Examples

```jldoctest
julia> PartialLinearConstraint(; A = [1.0 2.0; 3.0 4.0], B = [5.0, 6.0])
PartialLinearConstraint
  A ┼ 2×2 Matrix{Float64}
  B ┴ Vector{Float64}: [5.0, 6.0]
```

# Related

  - [`LinearConstraint`](@ref)
  - [`LinearConstraintEstimator`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.1.
"""
@concrete struct PartialLinearConstraint <: AbstractConstraintResult
    """
    $(field_dict[:A])
    """
    A
    """
    $(field_dict[:B])
    """
    B
    function PartialLinearConstraint(A::MatNum, B::VecNum)::PartialLinearConstraint
        @argcheck(!isempty(A), IsEmptyError)
        @argcheck(!isempty(B), IsEmptyError)
        @argcheck(size(A, 1) == length(B),
                  DimensionMismatch("a linear constraint half must have one row of `A` per entry of `B`. Got\nsize(A, 1) => $(size(A, 1))\nlength(B) => $(length(B))"))
        return new{typeof(A), typeof(B)}(A, B)
    end
end
function PartialLinearConstraint(; A::MatNum, B::VecNum)::PartialLinearConstraint
    return PartialLinearConstraint(A, B)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the inequality half and the equality half of a linear constraint block.

Each half is a [`PartialLinearConstraint`](@ref), and either one may be absent. The optimiser writes every row scaled and homogenised, as `sc * (A * w - k * B) <= 0` for the inequality half and `== 0` for the equality half, where `sc` is the constraint scale and `k` is the homogenisation scalar of a ratio objective. The returned solution is de-homogenised, so it satisfies the form below whatever the objective is.

# Mathematical definition

```math
\\begin{align}
  \\mathbf{A}_\\text{ineq} \\boldsymbol{x} &\\leq \\boldsymbol{B}_\\text{ineq} \\\\
  \\mathbf{A}_\\text{eq} \\boldsymbol{x} &= \\boldsymbol{B}_\\text{eq}\\,.
\\end{align}
```

Where:

  - $(math_dict[:A])
  - $(math_dict[:B])
  - $(math_dict[:ineq])
  - $(math_dict[:eq])
  - $(math_dict[:x])
  - ``\\boldsymbol{a}^\\intercal``: One row of a coefficient matrix.
  - ``b``: The entry of a response vector beside that row.

One row and the entry beside it are one constraint. The row runs over the entries of ``\\boldsymbol{x}``, in the order of the universe the constraint is written against.

The inequality half is defined in the ``\\leq`` sense, so the sense a row is written in fixes the half that holds it. The row ``\\boldsymbol{a}^\\intercal \\boldsymbol{x} = b`` is an equality and belongs to the ``\\text{eq}`` half. The row ``\\boldsymbol{a}^\\intercal \\boldsymbol{x} \\leq b`` belongs to the ``\\text{ineq}`` half as it stands. The row ``\\boldsymbol{a}^\\intercal \\boldsymbol{x} \\geq b`` is the same constraint as ``-\\boldsymbol{a}^\\intercal \\boldsymbol{x} \\leq -b``, so it belongs to the ``\\text{ineq}`` half with both sides negated.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearConstraint(;
        ineq::Option{<:PartialLinearConstraint} = nothing,
        eq::Option{<:PartialLinearConstraint} = nothing
    ) -> LinearConstraint

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:eqineq])

## View parameters

`LinearConstraint` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the index and drops it. Both halves are carried through unchanged, and `A` is never sliced along the asset axis.
  - A row is written over the whole universe it was assembled against, so slicing `A` would change what the row asserts. [`port_opt_view`](@ref) states why the identity is the behaviour this slot needs.

# Examples

```jldoctest
julia> ineq = PartialLinearConstraint(; A = [1.0 2.0; 3.0 4.0], B = [5.0, 6.0]);

julia> eq = PartialLinearConstraint(; A = [7.0 8.0; 9.0 10.0], B = [11.0, 12.0]);

julia> LinearConstraint(; ineq = ineq, eq = eq)
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 2×2 Matrix{Float64}
       │   B ┴ Vector{Float64}: [5.0, 6.0]
    eq ┼ PartialLinearConstraint
       │   A ┼ 2×2 Matrix{Float64}
       │   B ┴ Vector{Float64}: [11.0, 12.0]
```

# Related

  - [`PartialLinearConstraint`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`merge_linear_constraints`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equation 9.1.
"""
@concrete struct LinearConstraint <: AbstractConstraintResult
    """
    $(field_dict[:ineq])
    """
    ineq
    """
    $(field_dict[:eq])
    """
    eq
    function LinearConstraint(ineq::Option{<:PartialLinearConstraint},
                              eq::Option{<:PartialLinearConstraint})::LinearConstraint
        @argcheck(!(isnothing(ineq) && isnothing(eq)),
                  IsNothingError("ineq and eq cannot both be nothing. Got\nisnothing(ineq) => $(isnothing(ineq))\nisnothing(eq) => $(isnothing(eq))"))
        return new{typeof(ineq), typeof(eq)}(ineq, eq)
    end
end
function LinearConstraint(; ineq::Option{<:PartialLinearConstraint} = nothing,
                          eq::Option{<:PartialLinearConstraint} = nothing)::LinearConstraint
    return LinearConstraint(ineq, eq)
end
"""
    const VecLc = AbstractVector{<:LinearConstraint}

Every abstract vector whose elements are [`LinearConstraint`](@ref)s. The group exists so that one method signature accepts a whole block of assembled constraints, which is what a caller holds after several constraint steps have each produced one.

# Related

  - [`LinearConstraint`](@ref)
  - [`Lc_VecLc`](@ref)
  - [`merge_linear_constraints`](@ref)
"""
const VecLc = AbstractVector{<:LinearConstraint}
"""
    const Lc_VecLc = Union{<:LinearConstraint, <:VecLc}

One assembled [`LinearConstraint`](@ref), or a vector of them. The group exists because a caller that ran one constraint step and a caller that ran several reach the same slot, so every method that reads that slot must accept both arities.

# Related

  - [`LinearConstraint`](@ref)
  - [`VecLc`](@ref)
  - [`linear_constraints`](@ref)
"""
const Lc_VecLc = Union{<:LinearConstraint, <:VecLc}
# Flattened constraint matrices as virtual properties: `:A_ineq`, `:B_ineq`, `:A_eq`,
# `:B_eq` extract the corresponding sub-matrices from `obj.ineq` / `obj.eq`, returning
# `nothing` when the relevant constraint set is absent (the function form of `compute`
# returns `nothing` rather than throwing `PropertyPathError`; see [`@forward_properties`](@ref)).
@forward_properties LinearConstraint begin
    compute(A_ineq, obj -> isnothing(obj.ineq) ? nothing : obj.ineq.A)
    compute(B_ineq, obj -> isnothing(obj.ineq) ? nothing : obj.ineq.B)
    compute(A_eq, obj -> isnothing(obj.eq) ? nothing : obj.eq.A)
    compute(B_eq, obj -> isnothing(obj.eq) ? nothing : obj.eq.B)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Concatenate the rows of the same half of several [`PartialLinearConstraint`](@ref)s, skipping the absent ones.

# Algorithm

 1. Collect the entries of `ps` that are not `nothing`, giving `kept`.
 2. Return `nothing` when `kept` is empty, because the half is absent from every input.
 3. Read the row width of the first entry of `kept`, giving `N`, and check every other entry against it.
 4. Stack the `A` matrices of `kept` in input order, and stack their `B` vectors the same way.
 5. Return the [`PartialLinearConstraint`](@ref) built from the two stacks.

# Arguments

  - `ps`: The halves to concatenate, each a [`PartialLinearConstraint`](@ref) or `nothing`.

# Validation

  - Every kept half is written over the same number of variables, `size(p.A, 2) == N`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - A [`PartialLinearConstraint`](@ref), or `nothing` when every input was absent.

# Related

  - [`merge_linear_constraints`](@ref)
  - [`PartialLinearConstraint`](@ref)
"""
function merge_partial_linear_constraints(ps)
    kept = [p for p in ps if !isnothing(p)]
    if isempty(kept)
        return nothing
    end
    N = size(kept[1].A, 2)
    @argcheck(all(p -> size(p.A, 2) == N, kept),
              DimensionMismatch("every constraint being merged must be written over the same variables, but the row widths differ: $(unique(size(p.A, 2) for p in kept))"))
    return PartialLinearConstraint(; A = reduce(vcat, (p.A for p in kept)),
                                   B = reduce(vcat, (p.B for p in kept)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Combine several [`LinearConstraint`](@ref)s into the single one that holds all their rows.

A `LinearConstraint` is a block of rows, and applying two blocks is the same as applying the block that stacks them — the inequality halves concatenate, the equality halves concatenate, and an absent half contributes nothing. This is exactly what generation already does when it is handed several estimators at once: [`centrality_constraints`](@ref) over a vector of [`CentralityConstraint`](@ref)s appends every row into one result rather than returning one result per estimator.

That equivalence is what this function exists to preserve. A caller that computes its constraints separately — a [`Pipeline`](@ref) running one step per estimator — can merge them here and reach the optimiser with the value it would have had from the vector form.

# Algorithm

 1. Return the one element unchanged when `lcs` holds a single constraint.
 2. Merge the `ineq` half of every element with [`merge_partial_linear_constraints`](@ref), giving the inequality half of the result.
 3. Merge the `eq` half of every element the same way, giving the equality half.
 4. Return the [`LinearConstraint`](@ref) built from the two halves.

# Arguments

  - `lcs`: The constraints to merge.

# Validation

  - `lcs` is non-empty.
  - Every merged half is written over the same number of variables.

# Returns

  - `lc::LinearConstraint`: One constraint carrying every row, in input order.

# Examples

```jldoctest
julia> lc1 = LinearConstraint(; ineq = PartialLinearConstraint(; A = [1.0 0.0], B = [0.5]));

julia> lc2 = LinearConstraint(; ineq = PartialLinearConstraint(; A = [0.0 1.0], B = [0.25]));

julia> PortfolioOptimisers.merge_linear_constraints([lc1, lc2])
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 2×2 Matrix{Float64}
       │   B ┴ Vector{Float64}: [0.5, 0.25]
    eq ┴ nothing
```

# Related

  - [`LinearConstraint`](@ref)
  - [`merge_partial_linear_constraints`](@ref)
  - [`centrality_constraints`](@ref)
"""
function merge_linear_constraints(lcs::AbstractVector{<:LinearConstraint})::LinearConstraint
    @argcheck(!isempty(lcs), IsEmptyError("lcs cannot be empty"))
    if length(lcs) == 1
        return lcs[1]
    end
    return LinearConstraint(;
                            ineq = merge_partial_linear_constraints(lc.ineq for lc in lcs),
                            eq = merge_partial_linear_constraints(lc.eq for lc in lcs))
end
function merge_linear_constraints(lc::LinearConstraint)::LinearConstraint
    return lc
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all equation parsing result types.

All concrete and/or abstract types representing parsing results should be subtypes of `AbstractParsingResult`. Every member carries one parsed equation in canonical form — the variable names, their coefficients, the comparison operator and the right-hand side — so that the stages after [`parse_equation`](@ref) read one shape whatever the equation was written in.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`parse_equation`](@ref)
"""
abstract type AbstractParsingResult <: AbstractConstraintResult end
"""
$(DocStringExtensions.TYPEDEF)

Structured result for standard linear constraint equation parsing.

It is the canonical output of [`parse_equation`](@ref) for standard linear constraints, and it carries everything [`get_linear_constraints`](@ref) needs to assemble a row: the variable names, their coefficients, the comparison operator, the right-hand side value, and a formatted equation string.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ParsingResult(
        vars::VecStr,
        coef::VecNum,
        op::AbstractString,
        rhs::Number,
        eqn::AbstractString
    ) -> ParsingResult

Positional arguments correspond to the struct's fields. There is no keyword constructor, because [`parse_equation`](@ref) is the producer of this type.

## Validation

  - `length(vars) == length(coef)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ParsingResult([\"w_A\", \"w_B\"], [1.0, 2.0], \"<=\", 1.0,
                                         \"w_A + 2.0*w_B <= 1.0\")
ParsingResult
  vars ┼ Vector{String}: [\"w_A\", \"w_B\"]
  coef ┼ Vector{Float64}: [1.0, 2.0]
    op ┼ String: \"<=\"
   rhs ┼ Float64: 1.0
   eqn ┴ String: \"w_A + 2.0*w_B <= 1.0\"
```

# Related

  - [`AbstractParsingResult`](@ref)
  - [`parse_equation`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`get_linear_constraints`](@ref)
"""
@concrete struct ParsingResult <: AbstractParsingResult
    """
    $(field_dict[:vars])
    """
    vars
    """
    $(field_dict[:coef_c])
    """
    coef
    """
    $(field_dict[:op])
    """
    op
    """
    $(field_dict[:rhs])
    """
    rhs
    """
    $(field_dict[:eqn])
    """
    eqn
    function ParsingResult(vars::VecStr, coef::VecNum, op::AbstractString, rhs::Number,
                           eqn::AbstractString)::ParsingResult
        @argcheck(length(vars) == length(coef), DimensionMismatch)
        return new{typeof(vars), typeof(coef), typeof(op), typeof(rhs), typeof(eqn)}(vars,
                                                                                     coef,
                                                                                     op,
                                                                                     rhs,
                                                                                     eqn)
    end
end
"""
    const VecPR = AbstractVector{<:ParsingResult}

Every abstract vector whose elements are [`ParsingResult`](@ref)s. The group exists because [`parse_equation`](@ref) answers a vector of equations with a vector of results, and every stage after it is broadcast over that vector.

# Related

  - [`ParsingResult`](@ref)
  - [`PR_VecPR`](@ref)
  - [`parse_equation`](@ref)
"""
const VecPR = AbstractVector{<:ParsingResult}
"""
    const PR_VecPR = Union{<:ParsingResult, <:VecPR}

One [`ParsingResult`](@ref), or a vector of them. The group exists because an equation may be written singly or in a list, and every stage after [`parse_equation`](@ref) carries whichever arity it was given through to [`get_linear_constraints`](@ref).

# Related

  - [`ParsingResult`](@ref)
  - [`VecPR`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`get_linear_constraints`](@ref)
"""
const PR_VecPR = Union{<:ParsingResult, <:VecPR}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Collect the `dict` keys that start with `prefix`, as the candidate pool of a [`suggest_declared_key`](@ref) suggestion inside [`UniverseSets`](@ref).

A missing partition key is reported by the *group* that asked for it, so the whole key set is the wrong pool: the nearest neighbour of `nx_sector` in `Dict("ux_sector" => …)` is `ux_sector`, the very key under validation, and the caller would be told to rename the one thing that is correct. Narrowing the pool to the prefix the missing key must carry leaves only keys that could genuinely have been meant.

# Algorithm

 1. Return the keys of `dict` that start with `prefix`, as strings, in the order `dict` iterates in.

# Arguments

  - `dict`: The [`UniverseSets`](@ref) dictionary being validated.
  - `prefix`: The axis prefix the missing key must carry, `xkey`, `tfkey` or `cfkey`.

# Returns

  - `candidates::Vector{String}`: The keys of `dict` that start with `prefix`.

# Related

  - [`UniverseSets`](@ref)
  - [`unclaimed_sets_keys`](@ref)
  - [`suggest_declared_key`](@ref)
"""
function prefixed_sets_keys(dict::AbstractDict, prefix::AbstractString)
    return String[string(k) for k in keys(dict) if startswith(string(k), prefix)]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Collect the `dict` keys that no axis in `claimed` has taken, as the candidate pool of the missing-`xkey` suggestion inside [`UniverseSets`](@ref).

The counterpart of [`prefixed_sets_keys`](@ref) for the one key with no prefix of its own. The asset universe is whichever key holds the asset names, so it cannot be found by a prefix; what *can* be ruled out is every key another declared axis already speaks for. Without that, a dict carrying only a feature axis answers a mistyped `xkey` with the feature key, which is a different axis and never the right fix.

# Algorithm

 1. Return the keys of `dict` that start with no entry of `claimed`, as strings, in the order `dict` iterates in.

# Arguments

  - `dict`: The [`UniverseSets`](@ref) dictionary being validated.
  - `claimed`: The other declared axis prefixes, `uxkey`, `tfkey`, `utfkey`, `cfkey` and `ucfkey`.

# Returns

  - `candidates::Vector{String}`: The keys of `dict` that start with no entry of `claimed`.

# Related

  - [`UniverseSets`](@ref)
  - [`prefixed_sets_keys`](@ref)
  - [`suggest_declared_key`](@ref)
"""
function unclaimed_sets_keys(dict::AbstractDict, claimed)
    return String[string(k)
                  for k in keys(dict) if !any(p -> startswith(string(k), p), claimed)]
end
"""
    assert_factor_partition(dict::AbstractDict, k::AbstractString, fkey::AbstractString,
                            axis::AbstractString) -> Nothing

Assert that the factor partition `k` names a declared factor axis `fkey`, and that the two agree on how many factors there are.

[`UniverseSets`](@ref) carries **two** factor axes, and both obey this one rule, so the rule is written once and called twice. `axis` names the axis in both messages — a caller who declared the time-series axis and wrote a cross-sectional partition is told which of the two is missing, which the key value alone does not say.

# Arguments

  - `dict`: The [`UniverseSets`](@ref) dictionary being validated.
  - `k`: The `fkey`-prefixed key under validation.
  - `fkey`: The factor axis key the prefix belongs to, `tfkey` or `cfkey`.
  - `axis`: Names the axis in both diagnostic messages, for example `"time-series factor"`.

# Validation

  - `haskey(dict, fkey)`. A `KeyError` naming `axis` is thrown otherwise.
  - `length(dict[k]) == length(dict[fkey])`. A `DimensionMismatch` naming `axis` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`UniverseSets`](@ref)
  - [`assert_factor_unique_group`](@ref)
"""
function assert_factor_partition(dict::AbstractDict, k::AbstractString,
                                 fkey::AbstractString, axis::AbstractString)::Nothing
    @argcheck(haskey(dict, fkey),
              KeyError("$fkey (the $axis universe), required by the $axis partition $k. A `$fkey`-prefixed key declares a partition of the $axis axis, so the axis itself must be declared: add `$fkey => <factor names>` to `dict`, or rename `$k` if it was never meant to be a $axis partition."))
    @argcheck(length(dict[k]) == length(dict[fkey]),
              DimensionMismatch("the $axis partition `$k` and the $axis universe `$fkey` disagree on how many factors there are. Got\nlength(dict[$k]) => $(length(dict[k]))\nlength(dict[$fkey]) => $(length(dict[fkey]))"))
    return nothing
end
"""
    assert_factor_unique_group(dict::AbstractDict, k::AbstractString, fkey::AbstractString,
                               ufkey::AbstractString, axis::AbstractString) -> Nothing

Assert that the unique-entry factor group `k` names a declared factor axis `fkey`, that the partition it draws its entries from exists, and that the partition has the length of the axis.

The sibling of [`assert_factor_partition`](@ref), and written for the same reason: [`UniverseSets`](@ref) carries two factor axes and both obey this one rule, so the rule is written once and called twice.

# Arguments

  - `dict`: The [`UniverseSets`](@ref) dictionary being validated.
  - `k`: The `ufkey`-prefixed key under validation.
  - `fkey`: The factor axis key the group summarises, `tfkey` or `cfkey`.
  - `ufkey`: The unique-entry prefix `k` carries, `utfkey` or `ucfkey`.
  - `axis`: Names the axis in every diagnostic message, for example `"cross-sectional factor"`.

# Validation

  - `haskey(dict, fkey)`. A `KeyError` naming `axis` is thrown otherwise.
  - `haskey(dict, fkey * chopprefix(k, ufkey))`. A `KeyError` carrying a spelling suggestion is thrown otherwise.
  - `length(dict[fkey * chopprefix(k, ufkey)]) == length(dict[fkey])`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`UniverseSets`](@ref)
  - [`assert_factor_partition`](@ref)
  - [`prefixed_sets_keys`](@ref)
"""
function assert_factor_unique_group(dict::AbstractDict, k::AbstractString,
                                    fkey::AbstractString, ufkey::AbstractString,
                                    axis::AbstractString)::Nothing
    @argcheck(haskey(dict, fkey),
              KeyError("$fkey (the $axis universe), required by the unique-entry $axis group $k. A `$ufkey`-prefixed key summarises a partition of the $axis axis, so the axis itself must be declared: add `$fkey => <factor names>` to `dict`, or rename `$k` if it was never meant to be a $axis group."))
    tmp_key = fkey * chopprefix(k, ufkey)
    @argcheck(haskey(dict, tmp_key),
              KeyError("$tmp_key (the $axis partition), required by the unique-entry $axis group $k. Every `$ufkey`-prefixed group names the `$fkey`-prefixed partition it draws its entries from: correct the spelling$(suggest_declared_key(tmp_key, prefixed_sets_keys(dict, fkey))), or add `$tmp_key => <one group per factor>` to `dict`."))
    @argcheck(length(dict[tmp_key]) == length(dict[fkey]),
              DimensionMismatch("the $axis partition `$tmp_key`, required by the unique-entry $axis group `$k`, and the $axis universe `$fkey` disagree on how many factors there are. Got\nlength(dict[$tmp_key]) => $(length(dict[tmp_key]))\nlength(dict[$fkey]) => $(length(dict[fkey]))"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Declares the universes a portfolio problem is written against, and any groupings or partitions of them.

Constraint generation and the estimator routines read it to expand group references, to map a group name to its member list, and to validate membership.

It **declares every axis it carries**: `xkey`/`uxkey` for assets, `tfkey`/`utfkey` for time-series factors, `cfkey`/`ucfkey` for cross-sectional factors, and `nikey` for the Non-Investable Axis. Assets are the *primary* axis — `haskey(dict, xkey)` is required, and it is the axis a view slices. The factor axes are **optional**: requiring either would invalidate every sets object built for a problem with no factor model, so a consumer that needs one and does not find it throws at the point of need rather than at construction.

There are **two factor axes** because the two factor families name different things. A time-series regression fits one loading vector per asset over the observations, so its factors are the columns of `rd.F` and a caller copies `rd.nf` into the dict under `tfkey`. A cross-sectional regression fits one loading vector per observation across the assets, so its factors are the exposures the fit was built from, and they exist only inside the fitted block. One axis carrying both would make a single key mean two different lists on one problem. A consumer never chooses between them by hand: [`factor_axis_key`](@ref) reads the key off the loadings result it already holds.

If a key in `dict` starts with the same value as `xkey`, it means that the corresponding group must have the same length as the asset universe, `dict[xkey]`. This is useful for defining partitions of the asset universe, for example when using [`asset_sets_matrix`](@ref) with [`NestedClustered`](@ref).

If a key in `dict` starts with the same value as `uxkey`, it identifies a unique-entry group variant. The corresponding `xkey`-prefixed group must exist in `dict` with the same length as the asset universe, and is used to match each asset to a unique entry from the `uxkey`-prefixed group. This enables constraint generation using unique entries even in [`NestedClustered`](@ref) optimisations.

The `tfkey`/`utfkey` prefixes mean the same thing on the time-series factor axis, and `cfkey`/`ucfkey` mean the same thing again on the cross-sectional one. They buy something different from the asset pair. On the asset side the conventions serve *views*; factors are never sliced by an asset index, so on either factor side they buy length validation at construction and one shared mental model. The two factor axes are validated alike, and neither is validated against the other: a problem may declare one, both, or neither.

A taxonomy reaches the Asset Panel through [`panel_input`](@ref), which reads one `xkey`-prefixed key as one Panel Field. The panel names its own columns, so no key declares a feature axis.

`nikey` declares the **Non-Investable Axis**: the names the Investable Mask left out, which is the axis a forced liquidation is priced on. It is unlike the other six in three ways, and each is deliberate. It is **minted, not authored** — a door writes it after it reduces an optimiser to the Investable Mask, so a caller who states one by hand is overwritten there; authoring one is still the way to resolve a liquidation rate outside a door. It is **dropped by every view**, so a sets that carries it was reduced by exactly one door for exactly one problem, and a cluster of a nested optimisation can never inherit its parent's departures and charge them again. And it is **bare** — no prefixed partition and no unique-entry twin — because its entries are unique by construction, and a plain group already reaches it.

A key matching none of the seven prefixes is a plain group: expanded by name and **axis-blind**, which is why a factor group needs no machinery of its own, and why a group resolves on the Non-Investable Axis with no `nikey`-prefixed machinery at all.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    UniverseSets(;
        xkey::AbstractString = "nx",
        uxkey::AbstractString = "ux",
        tfkey::AbstractString = "nf",
        utfkey::AbstractString = "uf",
        cfkey::AbstractString = "ncf",
        ucfkey::AbstractString = "ucf",
        nikey::AbstractString = "ni",
        dict::AbstractDict{<:AbstractString, <:Any}
    ) -> UniverseSets

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(dict)`.
  - `haskey(dict, xkey)`.
  - No two of `xkey`, `uxkey`, `tfkey`, `utfkey`, `cfkey`, `ucfkey`, `nikey` may be a prefix of one another (42 ordered checks, which also rules out any two being equal).
  - If a key in `dict` starts with the same value as `xkey`, `length(dict[k]) == length(dict[xkey])`.
  - If a key in `dict` starts with the same value as `uxkey`, there must be a corresponding key in `dict` where the `uxkey` prefix is replaced by the `xkey` prefix, and its length must equal `length(dict[xkey])`.
  - If a key in `dict` starts with the same value as `tfkey`, `haskey(dict, tfkey)` and `length(dict[k]) == length(dict[tfkey])`.
  - If a key in `dict` starts with the same value as `utfkey`, there must be a corresponding key in `dict` where the `utfkey` prefix is replaced by the `tfkey` prefix, and its length must equal `length(dict[tfkey])`.
  - If a key in `dict` starts with the same value as `cfkey`, `haskey(dict, cfkey)` and `length(dict[k]) == length(dict[cfkey])`.
  - If a key in `dict` starts with the same value as `ucfkey`, there must be a corresponding key in `dict` where the `ucfkey` prefix is replaced by the `cfkey` prefix, and its length must equal `length(dict[cfkey])`.
  - If `dict` carries `nikey`, its entries are unique, and none of them is also in `dict[xkey]`. An asset is investable or it is not, and a name on both axes would be priced twice — once as a holding and once as a forced exit.

## View parameters

`UniverseSets` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the asset index alone. It drops every further positional argument, because no axis but the asset axis is sliced.
  - Every `xkey`-prefixed entry of `dict` is sliced to the selected assets, and every `uxkey`-prefixed entry is rebuilt from the sliced partition it names.
  - The `tfkey`-, `utfkey`-, `cfkey`- and `ucfkey`-prefixed entries, and every plain group, are carried through unchanged. [`port_opt_view`](@ref) states why each axis is exempt.
  - The `nikey` entry is **dropped**, because only a door mints one. The key itself is matched exactly rather than by prefix, so a plain group whose name merely starts with it survives.
  - The seven key prefixes are carried through unchanged, so the viewed value declares the same axes as the original.

# Examples

```jldoctest
julia> UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"group1\" => [\"A\", \"B\"]))
UniverseSets
    xkey ┼ String: "nx"
   uxkey ┼ String: "ux"
   tfkey ┼ String: "nf"
  utfkey ┼ String: "uf"
   cfkey ┼ String: "ncf"
  ucfkey ┼ String: "ucf"
   nikey ┼ String: "ni"
    dict ┴ Dict{String, Vector{String}}: Dict("group1" => ["A", "B"], "nx" => ["A", "B", "C"])
```

# Related

  - [`replace_group_by_assets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`linear_constraints`](@ref)
  - [`factor_axis_key`](@ref)
  - [`factor_universe`](@ref)
  - [`panel_input`](@ref)
  - [`prefixed_sets_keys`](@ref)
  - [`unclaimed_sets_keys`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct UniverseSets <: AbstractEstimator
    """
    $(field_dict[:us_xkey])
    """
    xkey
    """
    $(field_dict[:us_uxkey])
    """
    uxkey
    """
    $(field_dict[:us_tfkey])
    """
    tfkey
    """
    $(field_dict[:us_utfkey])
    """
    utfkey
    """
    $(field_dict[:us_cfkey])
    """
    cfkey
    """
    $(field_dict[:us_ucfkey])
    """
    ucfkey
    """
    $(field_dict[:us_nikey])
    """
    nikey
    """
    $(field_dict[:dict])
    """
    dict
    function UniverseSets(xkey::AbstractString, uxkey::AbstractString,
                          tfkey::AbstractString, utfkey::AbstractString,
                          cfkey::AbstractString, ucfkey::AbstractString,
                          nikey::AbstractString,
                          dict::AbstractDict{<:AbstractString, <:Any})::UniverseSets
        @argcheck(!isempty(dict), IsEmptyError)
        @argcheck(haskey(dict, xkey),
                  KeyError("$xkey (the asset universe), required by UniverseSets. The asset axis is the one mandatory axis: correct the spelling$(suggest_declared_key(xkey, unclaimed_sets_keys(dict, (uxkey, tfkey, utfkey, cfkey, ucfkey, nikey)))), pass `xkey = <the key you wrote>`, or add `$xkey => <asset names>` to `dict`."))
        knames = ("xkey", "uxkey", "tfkey", "utfkey", "cfkey", "ucfkey", "nikey")
        kvals = (xkey, uxkey, tfkey, utfkey, cfkey, ucfkey, nikey)
        for i in eachindex(kvals), j in eachindex(kvals)
            i == j && continue
            @argcheck(!startswith(kvals[i], kvals[j]),
                      ArgumentError("$(knames[i]) ($(kvals[i])) must not start with $(knames[j]) ($(kvals[j]))"))
        end
        if haskey(dict, nikey)
            ni = dict[nikey]
            @argcheck(allunique(ni),
                      ArgumentError("the non-investable axis `$nikey` names an asset twice. A departure happens once, so a repeated name would price one forced exit more than once: deduplicate `dict[$nikey]`."))
            @argcheck(isdisjoint(ni, dict[xkey]),
                      ArgumentError("$(length(intersect(ni, dict[xkey]))) name(s) are on both the asset universe `$xkey` and the non-investable axis `$nikey`. An asset is investable or it is not, and a name on both would be priced twice, once as a holding and once as a forced exit: remove it from whichever axis it does not belong to."))
        end
        for k in setdiff(keys(dict), (xkey, tfkey, cfkey, nikey))
            if startswith(k, xkey)
                @argcheck(length(dict[k]) == length(dict[xkey]),
                          DimensionMismatch("the asset partition `$k` and the asset universe `$xkey` disagree on how many assets there are. Got\nlength(dict[$k]) => $(length(dict[k]))\nlength(dict[$xkey]) => $(length(dict[xkey]))"))
            elseif startswith(k, uxkey)
                tmp_key = xkey * chopprefix(k, uxkey)
                @argcheck(haskey(dict, tmp_key),
                          KeyError("$tmp_key (the asset partition), required by the unique-entry asset group $k. Every `$uxkey`-prefixed group names the `$xkey`-prefixed partition it draws its entries from: correct the spelling$(suggest_declared_key(tmp_key, prefixed_sets_keys(dict, xkey))), or add `$tmp_key => <one group per asset>` to `dict`."))
                @argcheck(length(dict[tmp_key]) == length(dict[xkey]),
                          DimensionMismatch("the asset partition `$tmp_key`, required by the unique-entry asset group `$k`, and the asset universe `$xkey` disagree on how many assets there are. Got\nlength(dict[$tmp_key]) => $(length(dict[tmp_key]))\nlength(dict[$xkey]) => $(length(dict[xkey]))"))
            elseif startswith(k, tfkey)
                assert_factor_partition(dict, k, tfkey, "time-series factor")
            elseif startswith(k, utfkey)
                assert_factor_unique_group(dict, k, tfkey, utfkey, "time-series factor")
            elseif startswith(k, cfkey)
                assert_factor_partition(dict, k, cfkey, "cross-sectional factor")
            elseif startswith(k, ucfkey)
                assert_factor_unique_group(dict, k, cfkey, ucfkey, "cross-sectional factor")
            end
        end
        return new{typeof(xkey), typeof(uxkey), typeof(tfkey), typeof(utfkey),
                   typeof(cfkey), typeof(ucfkey), typeof(nikey), typeof(dict)}(xkey, uxkey,
                                                                               tfkey,
                                                                               utfkey,
                                                                               cfkey,
                                                                               ucfkey,
                                                                               nikey, dict)
    end
end
function UniverseSets(; xkey::AbstractString = "nx", uxkey::AbstractString = "ux",
                      tfkey::AbstractString = "nf", utfkey::AbstractString = "uf",
                      cfkey::AbstractString = "ncf", ucfkey::AbstractString = "ucf",
                      nikey::AbstractString = "ni",
                      dict::AbstractDict{<:AbstractString, <:Any})::UniverseSets
    return UniverseSets(xkey, uxkey, tfkey, utfkey, cfkey, ucfkey, nikey, dict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`UniverseSets`](@ref) restricted to the assets at index `i`.

The asset axis is the only axis this view slices, and the other three are exempt for two different reasons. **Both** factor axes are exempt because an asset index has no meaning on either, and they are treated alike: a `cfkey`-prefixed entry comes back bit-identical exactly as a `tfkey`-prefixed one does. Declaring an axis is what makes the exemption a property of the *data*: before the declaration, a factor-flavoured sets sitting in a `@vprop` field was sliced by asset indices and failed with a length mismatch, and the only defence was omitting the annotation by hand, field by field. There is deliberately **no factor-index arity** either. `port_opt_view(rd, i, j, k)` can slice `rd.nf`, but no internal caller passes a non-colon `k`, so a user who slices factors updates their sets themselves.

# Algorithm

 1. Read `xkey` and `uxkey` from `sets`, and open an empty dictionary `dict` of the type `sets.dict` has.
 2. For an entry of `sets.dict` whose key starts with `xkey`, take `view(v, i)`, the group restricted to the selected assets.
 3. For an entry whose key starts with `uxkey`, take the unique entries of the `xkey`-prefixed partition it names, restricted to `i`. The unique-entry group is therefore derived from the sliced partition and never from the original one.
 4. Skip the `nikey` entry, matched **exactly**. Only a door mints the Non-Investable Axis, so a view never carries one: a cluster of a nested optimisation would otherwise inherit its parent's departures and charge every one of them again, once per cluster. The match is exact rather than by prefix so that a plain group whose name merely starts with `nikey` — `"nikkei225"` under the default `"ni"` — is not silently dropped with it.
 5. Carry every other entry through unchanged, into the same `dict`. The `tfkey`-, `utfkey`-, `cfkey`- and `ucfkey`-prefixed entries, and every plain group, come back bit-identical.
 6. Return the [`UniverseSets`](@ref) built from `dict` and the eight unchanged key prefixes, which revalidates the prefix grammar over the viewed universe.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) to view.
  - `i`: The asset index selection.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `sets::UniverseSets`: A new [`UniverseSets`](@ref) over the selected assets, declaring the same seven key prefixes as the original.

# Related

  - [`UniverseSets`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(sets::UniverseSets, i, args...)::UniverseSets
    xkey = sets.xkey
    uxkey = sets.uxkey
    nikey = sets.nikey
    dict = typeof(sets.dict)()
    for (k, v) in sets.dict
        if startswith(k, xkey)
            v = view(v, i)
        elseif startswith(k, uxkey)
            v = unique(view(sets.dict[xkey * chopprefix(k, uxkey)], i))
        elseif k == nikey
            continue
        end
        push!(dict, k => v)
    end
    return UniverseSets(; xkey = xkey, uxkey = uxkey, tfkey = sets.tfkey,
                        utfkey = sets.utfkey, cfkey = sets.cfkey, ucfkey = sets.ucfkey,
                        nikey = nikey, dict = dict)
end
"""
    non_investable_sets(sets::Nothing, ni) -> Nothing
    non_investable_sets(sets::UniverseSets, ni::VecStr) -> UniverseSets

Mint the Non-Investable Axis on `sets`: declare `ni`, the names the Investable Mask left out, under `sets.nikey`.

This is the **only** way the axis comes to exist. A door calls it after it has reduced an optimiser to the Investable Mask, so a [`UniverseSets`](@ref) that carries the axis was reduced by exactly one door, for exactly one problem, and [`port_opt_view`](@ref) drops it rather than pass it to a sub-problem that did not earn it.

A caller may still declare the axis by hand, and outside a door that is the only way to resolve a forced-liquidation rate — [`fees_constraints`](@ref) called directly, with no optimisation around it. Inside a door the mask is the truth, so a hand-authored entry is **overwritten** here rather than merged: the two can only disagree, and the mask is the one derived from the data.

An empty `ni` returns `sets` untouched. Nothing left the universe, so nothing is owed, and declaring an empty axis would make [`fees_constraints`](@ref) resolve a carrier that prices no position.

# Algorithm

 1. Return `sets` unchanged when `ni` is empty.
 2. Otherwise copy `sets.dict`, write `ni` under `sets.nikey`, and rebuild the [`UniverseSets`](@ref) from it and the seven unchanged key prefixes, which revalidates uniqueness and disjointness over the minted axis.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) to mint the axis on, or `nothing`.
  - `ni`: The names the Investable Mask left out, in the order the complement of the mask visits them.

# Returns

  - `sets`: The [`UniverseSets`](@ref) carrying the Non-Investable Axis, or `nothing`.

# Related

  - [`UniverseSets`](@ref)
  - [`port_opt_view`](@ref)
  - [`investable_reduction`](@ref)
  - [`coverage_reduction`](@ref)
  - [`fees_constraints`](@ref)
"""
function non_investable_sets(::Nothing, ::Any)
    return nothing
end
function non_investable_sets(sets::UniverseSets, ni::VecStr)::UniverseSets
    if isempty(ni)
        return sets
    end
    dict = copy(sets.dict)
    dict[sets.nikey] = ni
    return UniverseSets(; xkey = sets.xkey, uxkey = sets.uxkey, tfkey = sets.tfkey,
                        utfkey = sets.utfkey, cfkey = sets.cfkey, ucfkey = sets.ucfkey,
                        nikey = sets.nikey, dict = dict)
end
"""
    non_investable_names(nx::Nothing, imsk::BitVector) -> VecStr
    non_investable_names(nx::VecStr, imsk::BitVector) -> VecStr

Read the names the Investable Mask leaves out, in the order the complement of the mask visits them.

The order is the whole point. A forced-liquidation carrier is sliced to the complement of the mask by index, and the rate that prices it is resolved against these names by position, so the two must walk the complement the same way. Both do: this indexes `nx` with `.!imsk`, which is ascending, and [`port_opt_view`](@ref)`(::Fees, i, X)` takes the complement of `i` over the width of `X`, which is ascending too.

Unnamed returns data answers an empty vector rather than throwing. Names are what the axis is made of, so a problem with no names has no Non-Investable Axis to mint — and no name-keyed constraint to resolve against one either.

It lives here, beside [`non_investable_sets`](@ref), rather than beside the optimisation door that first needed it, because a wrapping prior mints the axis at its own entry too and loads seven directories earlier. The vocabulary of the Non-Investable Axis is therefore one file, and no layer reaches it by a back reference.

# Arguments

  - `nx`: The asset names of the *unreduced* returns data, or `nothing`.
  - $(arg_dict[:imsk])

# Returns

  - `ni::VecStr`: The names the mask leaves out, or an empty vector.

# Related

  - [`investable_mask`](@ref)
  - [`non_investable_sets`](@ref)
  - [`non_investable_universe`](@ref)
  - [`investable_reduction`](@ref)
  - [`announce_non_investable`](@ref)
"""
function non_investable_names(::Nothing, ::BitVector)::VecStr
    return String[]
end
function non_investable_names(nx::VecStr, imsk::BitVector)::VecStr
    return nx[.!imsk]
end
"""
    record_non_investable_drop!(ledger::Nothing, what::AbstractString) -> Nothing
    record_non_investable_drop!(ledger::AbstractVector, what::AbstractString) -> Nothing

Record, for the door to report, one thing a departure cost.

A departed name is dropped where it is met — a view row here, a group member there — and each of those places is far from the door that derived the mask and knows the departure happened *as an event*. Reporting at the site would say the same thing once per row per window of a walk-forward, and that repetition is refused. Reporting nothing leaves a caller who wrote three views and got one fitted with no way to learn it. So the site writes what it dropped into a **ledger**, and the door reads the ledger once and says both things together, through [`announce_non_investable`](@ref).

A `nothing` ledger is the no-collection path, and it is the default everywhere: a caller who assembles constraints outside a door has no door to report to, and pays nothing for the ledger it does not keep. The branch is dispatch rather than a condition, as it is throughout the reduction machinery.

`what` is a noun phrase naming the casualty, not a sentence: the door joins them into one message and supplies the verb.

# Arguments

  - `ledger`: The door's ledger, or `nothing` when nobody is collecting.
  - `what`: A noun phrase naming what was dropped, for example ``"the view row `a + c == 0.05`"``.

# Returns

  - `nothing`. A vector ledger is appended to in place.

# Related

  - [`announce_non_investable`](@ref)
  - [`get_linear_constraints`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`counterpart_axis_names`](@ref)
"""
function record_non_investable_drop!(::Nothing, ::AbstractString)::Nothing
    return nothing
end
function record_non_investable_drop!(ledger::AbstractVector, what::AbstractString)::Nothing
    push!(ledger, what)
    return nothing
end
"""
    record_group_shed!(ledger::Option{<:AbstractVector}, group::AbstractString,
                       shed::Integer, kept::Integer, eqn::AbstractString) -> Nothing

Record what a group shed to a departure, for the door to report through [`announce_non_investable`](@ref).

A group that loses some of its members still describes the rest, so its row survives at a coefficient spread over the survivors; a group that loses **all** of them describes nothing, and its row goes with it. The two are different news to a caller, so they are phrased differently, and this is the one place either sentence is written. [`replace_group_by_assets`](@ref) is the only caller, at each of its four expansion branches.

Counts, not names: the departed assets are named once by the door, and repeating them per group would make the message longer than what it reports.

A shed of nothing records nothing, so the all-investable path costs one comparison.

# Arguments

  - `ledger`: The door's ledger, or `nothing` when nobody is collecting.
  - `group`: The group name as the caller wrote it, or the pair `"(a, b)"` for a correlation view.
  - `shed`: How many members the group lost.
  - `kept`: How many members survived.
  - `eqn`: The row the group appears in, as the caller wrote it.

# Returns

  - `nothing`.

# Related

  - [`record_non_investable_drop!`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`shed_departed_members`](@ref)
  - [`announce_non_investable`](@ref)
"""
function record_group_shed!(ledger::Option{<:AbstractVector}, group::AbstractString,
                            shed::Integer, kept::Integer, eqn::AbstractString)::Nothing
    if iszero(shed)
        return nothing
    end
    what = if iszero(kept)
        "the row `$(eqn)`, whose group `$(group)` lost every member"
    else
        "$(shed) departed member(s) of the group `$(group)` in the row `$(eqn)`"
    end
    return record_non_investable_drop!(ledger, what)
end
"""
    announce_non_investable(ni::VecStr, drops::VecStr = String[],
                            process::AbstractString = "optimisation",
                            consequence::AbstractString = "…";
                            warn::Bool = false) -> Nothing

Announce, once per door, the assets that left the investable universe and what their leaving cost.

The door is the only place that knows a departure happened *as an event* rather than as a shape. Downstream, a departed asset is simply absent: a bound stated for it resolves on the Non-Investable Axis and is skipped, a view row naming it is dropped whole. Reporting each of those where it happens would say the same thing once per row per window of a walk-forward, so each site writes its casualty into a ledger with [`record_non_investable_drop!`](@ref) and the door says everything once, here.

`process` names the work the departure is excluded from, because more than one kind of door mints the axis: an optimisation reduces at its entry, and a wrapping prior reduces at its own before it builds a view. Hard-coding `"optimisation"` made the message wrong for the second. `consequence` states what a departure means to *this* door — a forced-liquidation carrier is priced by an optimisation and by nothing else — and both are ordinary defaults, so the optimisation door reads as it always did.

It is `@info` by default, not a warning and not a [`strict_diagnostic`](@ref). Nothing is wrong: the data moved, and the work is proceeding correctly over what is left. Making it raise under `strict` would put back the refusal this whole path exists to remove. `warn` raises it to `@warn` for the one case that is not routine — a departure that took the **last** of something the caller asked for, such as the final view of a view set, because handing back the unconditioned answer changes the result and the caller has no other way to learn it.

An empty `ni` says nothing at all, which is the all-investable path and the unnamed-data path alike. An empty `drops` says who left and stops there, which is the door that has not yet resolved anything over them.

# Arguments

  - `ni`: The names the Investable Mask left out.
  - `drops`: The ledger of casualties, as [`record_non_investable_drop!`](@ref) filled it.
  - `process`: Noun phrase naming the work, for example `"optimisation"` or `"entropy pooling fit"`.
  - `consequence`: Sentence stating what a departure means to this door.
  - `warn`: Raise the message to `@warn`, for a departure that changed the model rather than trimming it.

# Returns

  - `nothing`.

# Related

  - [`non_investable_names`](@ref)
  - [`record_non_investable_drop!`](@ref)
  - [`investable_reduction`](@ref)
  - [`coverage_reduction`](@ref)
"""
function announce_non_investable(ni::VecStr, drops::VecStr = String[],
                                 process::AbstractString = "optimisation",
                                 consequence::AbstractString = "A constraint, bound or rate stated for one of them is dropped, and a forced-liquidation carrier is priced over them.";
                                 warn::Bool = false)::Nothing
    if isempty(ni)
        return nothing
    end
    msg = "$(length(ni)) asset(s) left the investable universe and are excluded from this $(process): $(ni). $(consequence)"
    if !isempty(drops)
        msg *= " Dropped over them: $(join(drops, "; "))."
    end
    if warn
        @warn(msg)
    else
        @info(msg)
    end
    return nothing
end
"""
    factor_universe(sets::UniverseSets, key::AbstractString, K::Integer,
                    need::AbstractString, source::AbstractString) -> VecStr

Read the **declared** factor universe `sets.dict[key]`, checking that it exists and that it agrees with `source` — the `observations × factors` matrix whose `K` columns it must name — on how many factors there are.

A factor axis is optional on [`UniverseSets`](@ref) but is not optional for a consumer written against it, so the failure has to be diagnosed at the point of need. Both messages name `key` and the matrix, because the two are what a caller has to reconcile: a user arriving from the pre-declaration shape put the factor names under `xkey` and would otherwise be told about an *asset* universe they never wrote in.

`key` is stated rather than read off `sets`, because [`UniverseSets`](@ref) declares **two** factor axes and this helper cannot tell which one a caller means. A caller that holds a loadings result reads the key from it with [`factor_axis_key`](@ref); a caller written against the returns data's own `F` states `sets.tfkey`, the axis those columns live on.

One helper therefore serves every consumer of either axis, and none of them re-encodes the checks.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) whose factor axis is read.
  - `key`: The factor axis key to read, `sets.tfkey` or `sets.cfkey`.
  - `K`: The number of columns of `source`, which the declared axis must name.
  - `need`: Names the consumer in both diagnostic messages, for example `"a FactorSpace constraint"`.
  - `source`: Names the matrix in both diagnostic messages, for example `"rr.M"` or `"F"`.

# Validation

  - `haskey(sets.dict, key)`. A `KeyError` naming `need` is thrown otherwise.
  - `length(sets.dict[key]) == K`. A `DimensionMismatch` naming `source` is thrown otherwise.

# Returns

  - `nf::VecStr`: The declared factor names, in the column order of `source`.

# Related

  - [`UniverseSets`](@ref)
  - [`factor_axis_key`](@ref)
  - [`constraint_space_basis`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
"""
function factor_universe(sets::UniverseSets, key::AbstractString, K::Integer,
                         need::AbstractString, source::AbstractString)
    @argcheck(haskey(sets.dict, key),
              KeyError("$key (the factor universe), required by $need. A factor axis is optional on UniverseSets; it is not optional here: add `$key => <factor names>` to `sets.dict`, in the column order of `$source`."))
    nf = sets.dict[key]
    @argcheck(length(nf) == K,
              DimensionMismatch("`$source` and the declared factor axis disagree on how many factors there are. Got\nsize($source, 2) => $K\nlength(sets.dict[$key]) => $(length(nf))"))
    return nf
end
"""
    factor_axis_key(sets::UniverseSets, rr::Regression) -> AbstractString
    factor_axis_key(sets::UniverseSets, rr::CrossSectionalFactorModel) -> AbstractString
    factor_axis_key(sets::UniverseSets,
                    re::AbstractTimeSeriesRegressionEstimator) -> AbstractString

Return the [`UniverseSets`](@ref) key naming the factor axis that `rr`'s loadings are written on.

[`UniverseSets`](@ref) declares two factor axes, so a consumer that resolves factor names has to say which one it means. It never says so by hand. The key follows the block that carries `M`: a [`Regression`](@ref) is fitted per asset over the observations, so its columns are the columns of `rd.F` and it answers `sets.tfkey`; a [`CrossSectionalFactorModel`](@ref) is fitted per observation across the assets, so its columns are the exposures the fit was built from and it answers `sets.cfkey`. A caller therefore cannot name the wrong axis, and no consumer gains a field to state it in.

The third method serves a consumer that holds an unfitted specification rather than a result. Only the time-series family names a specification here, because [`RegE_Reg`](@ref) admits an [`AbstractTimeSeriesRegressionEstimator`](@ref) and no other estimator, and every result that family fits is a [`Regression`](@ref). The three methods therefore cover [`RegE_Reg`](@ref) exactly.

There is deliberately **no fallback on [`AbstractLoadingsRegressionResult`](@ref)**. A future member of the root would silently inherit whichever axis the fallback named, and half the time that is the wrong list of names with the right length — a constraint written against it would still solve and would constrain the wrong factors. A missing method is a `MethodError` that names the type.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) whose factor axis key is read.
  - `rr` / `re`: The loadings result, or the specification whose verb produces one.

# Returns

  - `key::AbstractString`: `sets.tfkey` for the time-series family, `sets.cfkey` for the cross-sectional one.

# Related

  - [`UniverseSets`](@ref)
  - [`factor_universe`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`constraint_space_basis`](@ref)
  - [`risk_budget_universe_key`](@ref)
"""
function factor_axis_key(sets::UniverseSets, ::Regression)
    return sets.tfkey
end
function factor_axis_key(sets::UniverseSets, ::CrossSectionalFactorModel)
    return sets.cfkey
end
function factor_axis_key(sets::UniverseSets, ::AbstractTimeSeriesRegressionEstimator)
    return sets.tfkey
end
"""
    counterpart_axis_names(sets::UniverseSets, nxkey::AbstractString) -> VecStr

Return the asset axis that `nxkey` is the counterpart of, or an empty vector when it has none.

[`UniverseSets`](@ref) declares two axes over assets: the investable universe under `xkey`, and the Non-Investable Axis under `nikey`, which a door mints from the complement of the Investable Mask. A name-keyed estimator resolves against one of them, and a name it does not find there may still be a perfectly good name on the other — a bound stated for an asset that has since left, or a forced-liquidation rate stated for one that stayed. [`name_to_val!`](@ref) needs that list to tell such a name from a typo, and this is where the pairing is written down.

The relation is symmetric and covers only these two. A factor axis names factors, so no asset name is ever a departed factor and the answer is empty; a caller-supplied `key` naming some other list is treated the same way.

An axis that `sets` does not declare answers empty, which is the common case: a problem in which every asset is investable carries no `nikey` entry at all.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) whose axes are read.
  - `nxkey`: The key of the axis being resolved against.

# Returns

  - `other::VecStr`: The counterpart axis, or an empty vector.

# Related

  - [`UniverseSets`](@ref)
  - [`name_to_val!`](@ref)
  - [`estimator_to_val`](@ref)
  - [`non_investable_sets`](@ref)
"""
function counterpart_axis_names(sets::UniverseSets, nxkey::AbstractString)::VecStr
    key = if nxkey == sets.xkey
        sets.nikey
    elseif nxkey == sets.nikey
        sets.xkey
    else
        return String[]
    end
    return get(sets.dict, key, String[])
end
"""
    shed_departed_members(members::AbstractVector, other::VecStr,
                          ledger::Option{<:AbstractVector}, group::AbstractString,
                          eqn::AbstractString) -> AbstractVector
    shed_departed_members(members1::AbstractVector, members2::AbstractVector,
                          other::VecStr, ledger::Option{<:AbstractVector},
                          group::AbstractString, eqn::AbstractString) -> Tuple

Strike from a group's member list the names that sit on the counterpart axis, and tell the door's ledger what went.

A group is a **description the data resolves**, not a term the caller chose: `"tech"` means the technology assets of this problem, and when one of them delists the description still names the rest. [`replace_group_by_assets`](@ref) therefore sheds the departed members *before* it spreads the group's coefficient, so a Black–Litterman mean divides by the surviving count and an entropy pooling sum runs over the survivors — the row still computes what its right-hand side asserts. Striking a member afterwards would leave `k - 1` legs of `c/k` against an unchanged target. That is why a group differs from a written-out name, which takes its row with it.

A group that loses **every** member keeps the first of them rather than answering empty, because a group that describes nobody *is* a row naming a departed asset, and saying so is what makes it drop by the counterpart rule one door later — whole, and in silence. Answering empty would leave a row with no variable in it, which is what a caller writing `1 == 0.004` produces, and that one still has to be diagnosed.

The second method is the **pair** form, for a correlation view written over two groups. The two lists are walked together, and a position is kept only when *both* of its names survived: a pair is one correlation, so a pair that has lost either side has nothing left to measure, and shedding jointly is also what keeps the two lists the same length, which [`replace_group_by_assets`](@ref) has already checked. The all-lost case keeps the first pair, on the same reasoning.

An empty `other` is the all-investable path, and both methods then return their arguments untouched and record nothing, so a problem with no departure pays one comparison and no allocation.

The recording lives here rather than at the four call sites so that [`replace_group_by_assets`](@ref) spends one line per branch on the whole of it. The four branches are otherwise identical, and four copies of the shed, the record and the all-lost fallback is where they would drift.

# Arguments

  - `members` / `members1`, `members2`: The group's member names, as `sets.dict` holds them.
  - `other`: The counterpart axis, read with [`counterpart_axis_names`](@ref). Usually the Non-Investable Axis.
  - `ledger`: The door's ledger, or `nothing` when nobody is collecting.
  - `group`: The group name as the caller wrote it, for the ledger.
  - `eqn`: The row the group appears in, for the ledger.

# Returns

  - `members::AbstractVector`: The members that are not on `other`, in their original order, or the first departed member when none survived.
  - `(members1, members2)::Tuple`: The pair form, restricted to the positions both lists survived, or the first pair when none did.

# Related

  - [`replace_group_by_assets`](@ref)
  - [`record_group_shed!`](@ref)
  - [`counterpart_axis_names`](@ref)
  - [`non_investable_sets`](@ref)
  - [`UniverseSets`](@ref)
"""
function shed_departed_members(members::AbstractVector, other::VecStr,
                               ledger::Option{<:AbstractVector}, group::AbstractString,
                               eqn::AbstractString)
    if isempty(other)
        return members
    end
    kept = filter(!in(other), members)
    record_group_shed!(ledger, group, length(members) - length(kept), length(kept), eqn)
    return isempty(kept) ? members[1:1] : kept
end
function shed_departed_members(members1::AbstractVector, members2::AbstractVector,
                               other::VecStr, ledger::Option{<:AbstractVector},
                               group::AbstractString, eqn::AbstractString)
    if isempty(other)
        return members1, members2
    end
    keep = [m1 ∉ other && m2 ∉ other for (m1, m2) in zip(members1, members2)]
    record_group_shed!(ledger, group, count(!, keep), count(keep), eqn)
    return if any(keep)
        members1[keep], members2[keep]
    else
        members1[1:1], members2[1:1]
    end
end
"""
    name_to_val!(nx::VecStr, sdict::AbstractDict, key::Any, val::Number,
                 arr::VecNum, strict::Bool, nxkey::AbstractString,
                 other::VecStr = String[])

Set values in a vector for the asset or the group of assets that `key` names.

`name_to_val!` resolves `key` through [`resolve_axis_name`](@ref) — an asset name resolves to itself, a group name expands to its members — maps the result to indices in the asset universe `nx`, and sets the corresponding entries of `arr` to `val`. If `key` names neither, the function either throws an error or issues a warning, depending on the `strict` flag. Every diagnostic message names the *size* of the universe and never the universe itself or the input value dictionary, because each is routed through a shared message builder in `01_Base/06_Messages.jl`.

`other` is the **counterpart axis**: the asset names that this call is not resolving against, but that the same [`UniverseSets`](@ref) declares. A name found there is **skipped in silence**, under `strict` or not, and that is the whole of what `strict` gives up. `strict` exists to catch a caller's typo, and a name on the counterpart axis is the opposite of a typo: it was a correct name over the universe the caller was given, and the data moved it. A caller cannot know in advance which asset a prior will fail to estimate, so refusing them — or even warning, once per constraint, per window of a walk-forward — reports something no one can act on. The departure itself is announced once, by the door that derived the mask.

The two asset axes are counterparts of each other, and the relation is symmetric. Resolving on the asset universe, `other` is the Non-Investable Axis, so a bound stated for an asset that left is dropped. Resolving on the Non-Investable Axis — which is how a forced-liquidation rate is priced — `other` is the asset universe, so a liquidation rate stated for an asset that stayed is dropped by the same rule. A factor axis has no counterpart, and `other` is then empty.

# Algorithm

 1. Resolve `key` through [`resolve_axis_name`](@ref), giving `members`. An asset name resolves to itself, and a group name expands to a copy of its member list. An asset name takes precedence over a group name of the same spelling.
 2. Return in silence when `members` is `nothing` and `key` names an entry of `other`, because the name is on the counterpart axis: it is known-good, and this axis has no entry to write it into.
 3. Report through [`strict_diagnostic`](@ref) and return when `members` is `nothing`, because `key` names neither an asset nor a group. The suggestion pool is widened from `nx` to `nx` together with the keys of `sdict`, because a missing name may be a mistyped asset or a mistyped group.
 4. Map `members` to positions in `nx` with [`axis_name_indices`](@ref), giving `idx`. Members that miss the universe are dropped. Those on `other` are struck from the report by the same rule as step 2, and any that remain are reported once through [`strict_diagnostic`](@ref) — so a group whose departed members are all accounted for is silent, and one holding a genuine typo still names it.
 5. Set the entries of `arr` at `idx` to `val`.

# Arguments

  - `nx`: Vector of asset names.
  - `sdict`: Dictionary mapping group names to vectors of asset names. It is never modified, because [`resolve_axis_name`](@ref) returns a copy of the member list.
  - `key`: Name of the asset or the group of assets to set values for.
  - `val`: The value to assign.
  - `arr`: The array to be modified in-place.
  - `strict`: If `true`, throws an error if `key` resolves to nothing; if `false`, issues a warning.
  - `nxkey`: Name of the asset-universe key in `sets.dict` (e.g. `"nx"`), used only to name the universe in the diagnostic message — see [`unknown_variable_msg`](@ref) / [`missing_group_assets_msg`](@ref).
  - `other`: The counterpart asset axis, whose names are skipped in silence rather than reported.

# Validation

  - `key` names an asset of `nx`, a group of `sdict`, or an entry of `other`. An `ArgumentError` is thrown when `strict` is `true`, and a warning is issued otherwise.
  - Every member of a resolved group names an entry of `nx` or of `other`. A member that misses both is dropped, and the drop raises when `strict` is `true` and issues a warning otherwise.

# Returns

  - `nothing`. The operation is performed in-place on `arr`.

# Related

  - [`estimator_to_val`](@ref)
  - [`resolve_axis_name`](@ref)
  - [`axis_name_indices`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`UniverseSets`](@ref)
  - [`unknown_variable_msg`](@ref)
  - [`missing_group_assets_msg`](@ref)
"""
function name_to_val!(nx::VecStr, sdict::AbstractDict, key::Any, val::Number, arr::VecNum,
                      strict::Bool, nxkey::AbstractString,
                      other::VecStr = String[])::Nothing
    members = resolve_axis_name(key, nx, sdict)
    if isnothing(members)
        # A name on the counterpart axis is known-good, not a typo: it was correct over the
        # universe the caller was given, and the data moved it to the other axis. Silent
        # under `strict` too — the door that derived the mask announces the departure once.
        if any(isequal(key), other)
            return nothing
        end
        # A missing key may be a mistyped asset *or* a mistyped group/set name, so widen the
        # suggestion pool beyond the raw universe to include the group/set keys.
        return strict_diagnostic(unknown_variable_msg(key, nx, nxkey;
                                                      candidates = [nx;
                                                                    collect(keys(sdict))]),
                                 strict)
    end
    idx = axis_name_indices(members, nx,
                            function (m)
                                m = filter(x -> !any(isequal(x), other), m)
                                return if isempty(m)
                                    nothing
                                else
                                    strict_diagnostic(missing_group_assets_msg(key, m, nx,
                                                                               nxkey),
                                                      strict)
                                end
                            end)
    arr[idx] .= val
    return nothing
end
"""
    estimator_to_val(dict::MultiEstValType, sets::UniverseSets,
                     val::Option{<:Number} = nothing,
                     key::Option{<:AbstractString} = nothing;
                     datatype::DataType = Float64, strict::Bool = false)
    estimator_to_val(dict::PairStrNum, sets::UniverseSets,
                     val::Option{<:Number} = nothing,
                     key::Option{<:AbstractString} = nothing;
                     datatype::DataType = Float64, strict::Bool = false)

Return value for assets or groups, based on a mapping and asset sets.

The function creates the vector and sets the values for assets or groups as specified by `dict`, using the asset universe and groupings in `sets`. If a key in `dict` is not found in the asset sets, the function either throws an error or issues a warning, depending on the `strict` flag.

!!! warning

    If the same asset is found in subsequent iterations, its value will be overwritten in favour of the most recent one. To ensure determinism, use an [`OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/stable/#OrderedDicts) or a vector of pairs.

# Algorithm

 1. Take `val` as the fill value, or `zero(datatype)` when `val` is `nothing`.
 2. Take `key` as the universe key `nxkey`, or `sets.xkey` when `key` is `nothing`, and read the universe `nx` from `sets.dict` under it.
 3. Allocate `arr`, one entry per name of `nx`, filled with the value of step 1.
 4. Read the counterpart axis with [`counterpart_axis_names`](@ref), the other of the two asset axes [`UniverseSets`](@ref) declares.
 5. For each `(key, val)` pair of `dict`, in the order `dict` iterates in, write `val` into `arr` through [`name_to_val!`](@ref). A key that names an asset writes one entry, a key that names a group writes one entry per member, a key that names the counterpart axis is skipped in silence, and a key that names none of them is reported through the `strict` flag.
 6. Return `arr`.

# Arguments

  - `dict`: A dictionary, vector of pairs, or single pair mapping asset or group names to values.
  - `sets`: The [`UniverseSets`](@ref) containing the asset universe and group definitions.
  - `val`: The value assigned to every asset before `dict` is applied. `nothing` means `zero(datatype)`.
  - `key`: (Optional) Key in the [`UniverseSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`UniverseSets`](@ref).
  - `datatype`: Element type of the value the array is filled with when `val` is `nothing`.
  - `strict`: If `true`, throws an error if a key in `dict` is not found in the asset sets; if `false`, issues a warning.

# Validation

  - A key of `dict` that names neither an asset, nor a group, nor an entry of the counterpart axis raises an `ArgumentError` when `strict` is `true`. A warning is issued otherwise.

# Returns

  - `arr::VecNum`: Value array, one entry per name of the universe.

# Related

  - [`name_to_val!`](@ref)
  - [`UniverseSets`](@ref)
  - [`estimator_to_val`](@ref)
"""
function estimator_to_val(dict::MultiEstValType, sets::UniverseSets,
                          val::Option{<:Number} = nothing,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, strict::Bool = false)
    val = ifelse(isnothing(val), zero(datatype), val)
    nxkey = ifelse(isnothing(key), sets.xkey, key)
    nx = sets.dict[nxkey]
    other = counterpart_axis_names(sets, nxkey)
    arr = fill(val, length(nx))
    for (key, val) in dict
        name_to_val!(nx, sets.dict, key, val, arr, strict, nxkey, other)
    end
    return arr
end
function estimator_to_val(dict::PairStrNum, sets::UniverseSets,
                          val::Option{<:Number} = nothing,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, strict::Bool = false)
    val = ifelse(isnothing(val), zero(datatype), val)
    nxkey = ifelse(isnothing(key), sets.xkey, key)
    nx = sets.dict[nxkey]
    arr = fill(val, length(nx))
    key, val = dict
    name_to_val!(nx, sets.dict, key, val, arr, strict, nxkey,
                 counterpart_axis_names(sets, nxkey))
    return arr
end
"""
    estimator_to_val(val::Option{<:Number}, args...; kwargs...)

Fallback no-op for value mapping in asset/group estimators.

This method returns the input value `val` as-is, without modification or mapping. It serves as a fallback for cases where the input is already a numeric value, a vector of numeric values, or `nothing`, and no further processing is required.

# Algorithm

 1. Return `val`. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `val`: A value of type `Nothing` or a single numeric value.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `val::Option{<:Number}`: The input `val`, unchanged.

# Related

  - [`estimator_to_val`](@ref)
  - [`name_to_val!`](@ref)
  - [`UniverseSets`](@ref)
"""
function estimator_to_val(val::Option{<:Number}, args...; kwargs...)::Option{<:Number}
    return val
end
"""
    estimator_to_val(val::VecNum, sets::UniverseSets, ::Any = nothing,
                     key::Option{<:AbstractString} = nothing; kwargs...)

Return a numeric vector for asset/group estimators, validating length against asset universe.

This method checks that the input vector `val` matches the length of the asset universe in `sets`, and returns it unchanged if valid. It is used as a fast path for workflows where the value vector is already constructed and requires only defensive validation.

# Algorithm

 1. Take `key` as the universe key, or `sets.xkey` when `key` is `nothing`, and read the universe from `sets.dict` under it.
 2. Check `val` against the length of that universe.
 3. Return `val`.

# Arguments

  - `val`: Numeric vector to be mapped to assets/groups.
  - `sets`: [`UniverseSets`](@ref) containing the asset universe and group definitions.
  - `::Any`: Fill value for API consistency (ignored).
  - `key`: (Optional) Key in the [`UniverseSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`UniverseSets`](@ref).
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `length(val) == length(sets.dict[ifelse(isnothing(key), sets.xkey, key)]`.

# Returns

  - `val::VecNum`: The input vector, unchanged.

# Related

  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
  - [`name_to_val!`](@ref)
"""
function estimator_to_val(val::VecNum, sets::UniverseSets, ::Any = nothing,
                          key::Option{<:AbstractString} = nothing; kwargs...)
    @argcheck(length(val) == length(sets.dict[ifelse(isnothing(key), sets.xkey, key)]),
              DimensionMismatch)
    return val
end
"""
    estimator_to_val(val::MatNum, sets::UniverseSets, ::Any = nothing,
                     key::Option{<:AbstractString} = nothing; dims::Int = 2, kwargs...)

Return a numeric matrix for asset/group estimators, validating length against asset universe.

This method checks that size of `dims` of the input matrix `val` matches the length of the asset universe in `sets`, and returns it unchanged if valid. It is used as a fast path for workflows where the value matrix is already constructed and requires only defensive validation.

# Algorithm

 1. Take `key` as the universe key, or `sets.xkey` when `key` is `nothing`, and read the universe from `sets.dict` under it.
 2. Check the size of `val` along `dims` against the length of that universe.
 3. Return `val`.

# Arguments

  - `val`: Numeric matrix to be mapped to assets/groups.
  - `sets`: [`UniverseSets`](@ref) containing the asset universe and group definitions.
  - `::Any`: Fill value for API consistency (ignored).
  - `key`: (Optional) Key in the [`UniverseSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`UniverseSets`](@ref).
  - `dims`: Dimension along which to validate the matrix size.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `size(val, dims) == length(sets.dict[ifelse(isnothing(key), sets.xkey, key)]`.

# Returns

  - `val::MatNum`: The input matrix, unchanged.

# Related

  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
  - [`name_to_val!`](@ref)
"""
function estimator_to_val(val::MatNum, sets::UniverseSets, ::Any = nothing,
                          key::Option{<:AbstractString} = nothing; dims::Int = 2, kwargs...)
    @argcheck(size(val, dims) == length(sets.dict[ifelse(isnothing(key), sets.xkey, key)]),
              DimensionMismatch)
    return val
end
"""
$(DocStringExtensions.TYPEDEF)

Fills every entry of a value vector with `1/N`, where `N` is the number of assets in the universe.

The same value is produced whatever slot the algorithm sits in. `lb = UniformValues()` floors every weight at the equal-weight level and `ub = UniformValues()` caps every weight there. Neither slot is a special case in [`estimator_to_val`](@ref).

# Mathematical definition

```math
\\begin{align}
v_i &= \\frac{1}{N}\\,, \\quad i = 1,\\, \\ldots,\\, N\\,.
\\end{align}
```

Where:

  - ``v_i``: Entry ``i`` of the value vector.
  - $(math_dict[:N])

The entries sum to one, so the vector is the equal-weight portfolio whenever the slot it fills is a set of weights.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> PortfolioOptimisers.estimator_to_val(UniformValues(), sets)
StepRangeLen(0.3333333333333333, 0.0, 3)
```

# Related

  - [`VectorAbstractEstimatorValueAlgorithm`](@ref)
  - [`AbstractEstimatorValueAlgorithm`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
"""
struct UniformValues <: VectorAbstractEstimatorValueAlgorithm end
"""
    estimator_to_val(::UniformValues, sets::UniverseSets, ::Any = nothing,
                     key::Option{<:AbstractString} = nothing;
                     datatype::DataType = Float64, kwargs...)

Return a uniform value vector for all assets in the universe defined by `sets`.

[`UniformValues`](@ref) states the closed form the entries take. The value is a range rather than a vector, so no array is allocated.

# Algorithm

 1. Take `key` as the universe key, or `sets.xkey` when `key` is `nothing`, and read the universe from `sets.dict` under it, giving its length `N`.
 2. Compute `iN`, the reciprocal of `N` in `datatype`.
 3. Return the range of length `N` whose start and stop are both `iN`.

# Arguments

  - `::UniformValues`: The algorithm that selects this method.
  - `sets`: The [`UniverseSets`](@ref) whose universe gives `N`.
  - `::Any`: Fill value for API consistency (ignored).
  - `key`: (Optional) Key in the [`UniverseSets`](@ref) naming the universe the value is written over. When provided, takes precedence over `sets.xkey`.
  - `datatype`: Element type of the returned range.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `val::StepRangeLen`: A range of length `N`, each entry the reciprocal of `N`.

# Related

  - [`UniformValues`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
"""
function estimator_to_val(::UniformValues, sets::UniverseSets, ::Any = nothing,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, kwargs...)
    N = length(sets.dict[ifelse(isnothing(key), sets.xkey, key)])
    iN = datatype(inv(N))
    return range(; start = iN, stop = iN, length = N)
end
"""
    allowed_functions = Dict{Symbol, Function}(:+ => +, :- => -, :* => *, :/ => /,
                                               :^ => ^, :sqrt => sqrt, :cbrt => cbrt,
                                               :exp => exp, :exp2 => exp2, :exp10 => exp10,
                                               :log => log, :log2 => log2, :log10 => log10,
                                               :abs => abs, :min => min, :max => max)

Enumerated table of the functions permitted in equation parsing, mapping each allowed name directly to its function object. Evaluating constraint/view strings crosses a trust boundary (config files, spreadsheets, UI), so the parser must be able to call *only* these 16 mathematical functions. Using an explicit `Symbol => Function` table — rather than resolving a name against `Base` with `getfield(Base, fname)` — bounds that capability to exactly this table: a name absent from the keys fails closed with a `Meta.ParseError`, and the set of callable functions cannot drift from the set of allowed names, because they are the same list.

The `prior(...)` marker is deliberately absent from this table: it names assets/groups (not numbers) and is expanded structurally by [`eval_numeric_functions`](@ref)/[`replace_group_by_assets`](@ref), never evaluated numerically.

# Related

  - [`eval_numeric_functions`](@ref)
  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
"""
const allowed_functions = Dict{Symbol, Function}(:+ => +, :- => -, :* => *, :/ => /,
                                                 :^ => ^, :sqrt => sqrt, :cbrt => cbrt,
                                                 :exp => exp, :exp2 => exp2,
                                                 :exp10 => exp10, :log => log,
                                                 :log2 => log2, :log10 => log10,
                                                 :abs => abs, :min => min, :max => max)
"""
    eval_numeric_functions(expr, datatype::DataType = Float64)

Recursively evaluate numeric functions and constants in a Julia expression.

`eval_numeric_functions` traverses a Julia expression tree and evaluates any sub-expressions that are purely numeric, including standard mathematical functions and constants (such as `Inf`). This is used to simplify constraint equations before further parsing and canonicalisation.

When an allowlisted function is actually evaluated (all its arguments are numeric), its arguments are coerced to `datatype` (a float type) *first*, so the arithmetic happens in the same numeric domain the optimiser will use rather than in machine `Int64`. This prevents integer literals from combining and wrapping — e.g. `2^64` yields `1.8446744073709552e19` rather than silently wrapping to `0`, and `2^-1` yields `0.5` rather than a `DomainError`. Numeric literals that survive inside an *unevaluated* (nonlinear) subexpression are left untouched, so `2^z` still renders as `2 ^ z`.

Only the functions enumerated in [`allowed_functions`](@ref) may be evaluated; any other call head fails closed with a `Meta.ParseError`. The `prior(...)` marker is handled structurally (see [`replace_group_by_assets`](@ref)) and throws a `Meta.ParseError` if given purely numeric arguments.

# Algorithm

 1. Fold every argument of `expr` first, by applying this function to each, when `expr` is an `Expr`. A node whose head is not `:call` is rebuilt from its folded arguments and returned.
 2. Rebuild a `:call` node whose head is `prior` from its folded arguments, and return it. The marker names assets or groups, so it is never folded to a number.
 3. Look the head of any other `:call` node up in [`allowed_functions`](@ref), giving `f`. A head that the table does not hold raises.
 4. Rebuild the call and return it when any folded argument is not a `Number`, so a nonlinear subexpression keeps its own literals untouched.
 5. Coerce every folded argument to `datatype`, apply `f` to them, and return the value that comes back.
 6. Return the value `Inf` when `expr` is the symbol `:Inf`. Return `expr` itself in every other case, so a number stands and a variable name survives as a symbol.

# Arguments

  - `expr`: The Julia expression to evaluate. Can be a `Number`, `Symbol`, or `Expr`.
  - `datatype`: Float type into which numeric arguments are coerced before an allowlisted function is evaluated.

# Validation

  - The head of a `:call` node is a key of [`allowed_functions`](@ref), or the `prior` marker. A `Meta.ParseError` naming the head is thrown otherwise.
  - `prior(...)` carries at least one argument that is not a number. A `Meta.ParseError` is thrown otherwise.

# Returns

  - The evaluated expression, with all numeric sub-expressions replaced by their computed values. Non-numeric or symbolic expressions are returned in their original or partially simplified form.

# Related

  - [`_collect_terms`](@ref)
  - [`_parse_equation`](@ref)
  - [`allowed_functions`](@ref)
  - [`replace_group_by_assets`](@ref)
"""
function eval_numeric_functions(expr, datatype::DataType = Float64)
    return if isa(expr, Expr)
        if expr.head == :call
            fname = expr.args[1]
            args = [eval_numeric_functions(arg, datatype) for arg in expr.args[2:end]]
            if fname === :prior
                # `prior(...)` names assets/groups and is expanded structurally later; it
                # must never be evaluated numerically, so all-numeric args are a user error.
                if all(x -> isa(x, Number), args)
                    throw(Meta.ParseError("`prior(...)` takes asset/group names, not numbers."))
                end
                Expr(:call, fname, args...)
            else
                f = get(allowed_functions, fname, nothing)
                if isnothing(f)
                    throw(Meta.ParseError("Function `$(fname)` is not allowed in constraint expressions."))
                end
                # Only evaluate if all arguments are numeric. Coerce them to `datatype`
                # first so arithmetic happens in the optimiser's float domain rather than
                # machine `Int64` (`2^64` would otherwise wrap; `2^-1` would `DomainError`).
                if all(x -> isa(x, Number), args)
                    f((datatype(a) for a in args)...)
                else
                    Expr(:call, fname, args...)
                end
            end
        else
            Expr(expr.head, map(a -> eval_numeric_functions(a, datatype), expr.args)...)
        end
    elseif isa(expr, Symbol) && expr == :Inf
        Inf
    else
        expr
    end
end
"""
    _collect_terms(expr::Union{Symbol, Expr, <:Number}, datatype::DataType = Float64)

Expand and collect all terms from a Julia expression representing a linear constraint equation.

`_collect_terms` takes a Julia expression (such as the left-hand side of a constraint equation), recursively traverses its structure, and returns a vector of `(coefficient, variable)` pairs. It supports numeric constants, variables, and arithmetic operations (`+`, `-`, `*`, `/`), and is used to canonicalise linear constraint equations for further processing.

The starting coefficient is `one(datatype)`, so every coefficient the walk builds is of that type. The caller asked for the numeric domain the optimiser works in, and the coefficients belong to it as much as the right-hand side does.

# Algorithm

 1. Open an empty vector `terms`.
 2. Walk `expr` with [`collect_terms!`](@ref), from the starting coefficient `one(datatype)`. The walk appends one pair to `terms` per term it reaches: a constant as `(coefficient, nothing)`, and anything else as `(coefficient, name)`.
 3. Return `terms`.

# Arguments

  - `expr`: The Julia expression to expand.
  - `datatype`: Numeric type of the coefficients the walk builds.

# Returns

  - `terms::Vector{Tuple{datatype, Option{<:String}}}`: A vector of `(coefficient, variable)` pairs, where `variable` is a string for variable terms or `nothing` for constant terms.

# Related

  - [`collect_terms!`](@ref)
  - [`_parse_equation`](@ref)
"""
function _collect_terms(expr, datatype::DataType = Float64)
    terms = []
    collect_terms!(expr, one(datatype), terms)
    return terms
end
"""
    collect_terms!(expr, coeff, terms)

Recursively collect and expand terms from a Julia expression for linear constraint parsing.

`collect_terms!` traverses a Julia expression tree representing a linear equation, expanding and collecting all terms into a vector of `(coefficient, variable)` pairs. It handles numeric constants, variables, and arithmetic operations (`+`, `-`, `*`, `/`), supporting canonicalisation of linear constraint equations for further processing.

# Algorithm

 1. Append `(coeff * expr, nothing)` when `expr` is a `Number`, so a constant carries the coefficient and no variable.
 2. Append `(coeff, string(expr))` when `expr` is a `Symbol`, so a bare variable carries the coefficient it arrived with.
 3. For a multiplication `a * b`, recurse into the side that is not a number, with `coeff` multiplied by the side that is. A product of two non-numeric sides is opaque, so append it whole as `(coeff, string(expr))`.
 4. For a division `a / b`, recurse into `a` with `coeff` divided by `b`, when `b` is a number. A division by a denominator that is not a number is opaque, so append it whole.
 5. For an addition, recurse into every argument with `coeff` unchanged.
 6. For a subtraction, recurse into every argument but the last with `coeff`, and into the last with `-coeff`. A unary minus holds no argument but the last, so this negates its one operand.
 7. Append any other expression whole, as `(coeff, string(expr))`. This is what makes a term such as `sqrt(x)` opaque: it becomes one variable named by its own text, and the row builder resolves that text against the universe like any other name.

# Arguments

  - `expr`: The Julia expression to traverse.
  - `coeff`: The current numeric coefficient to apply.
  - `terms`: A vector to which `(coefficient, variable)` pairs are appended in-place. Each pair is of the form `(typeof(coeff), Option{<:String})`, where `Nothing` indicates a constant term.

# Returns

  - `nothing`. The function modifies `terms` in-place.

# Related

  - [`_collect_terms`](@ref)
  - [`_parse_equation`](@ref)
"""
function collect_terms!(expr, coeff, terms)
    if isa(expr, Number)
        push!(terms, (coeff * oftype(coeff, expr), nothing))
    elseif isa(expr, Symbol)
        push!(terms, (coeff, string(expr)))
    elseif isa(expr, Expr)
        if expr.head == :call && expr.args[1] == :*
            # Multiplication: find numeric and variable part
            a, b = expr.args[2], expr.args[3]
            if isa(a, Number)
                collect_terms!(b, coeff * oftype(coeff, a), terms)
            elseif isa(b, Number)
                collect_terms!(a, coeff * oftype(coeff, b), terms)
            else
                # e.g. x*y, treat as variable
                push!(terms, (coeff, string(expr)))
            end
        elseif expr.head == :call && expr.args[1] == :/
            a, b = expr.args[2], expr.args[3]
            if isa(b, Number)
                collect_terms!(a, coeff / oftype(coeff, b), terms)
            else
                # e.g. x/y, treat as variable
                push!(terms, (coeff, string(expr)))
            end
        elseif expr.head == :call && expr.args[1] == :+
            for i in 2:length(expr.args)
                # Collect terms from addition
                collect_terms!(expr.args[i], coeff, terms)
            end
        elseif expr.head == :call && expr.args[1] == :-
            for i in 2:(length(expr.args) - 1)
                # Collect terms from addition
                collect_terms!(expr.args[i], coeff, terms)
            end
            collect_terms!(expr.args[length(expr.args)], -coeff, terms)
        else
            # treat as variable (e.g. sin(x))
            push!(terms, (coeff, string(expr)))
        end
    end
end
"""
    format_term(coeff, var)

Format a single term in a linear constraint equation as a string.

`format_term` takes a coefficient and a variable name and returns a string representation suitable for display in a canonicalised linear constraint equation. Handles special cases for coefficients of `1` and `-1` to avoid redundant notation.

# Algorithm

 1. Return the variable name alone when `coeff` is one.
 2. Return the variable name behind a minus sign when `coeff` is minus one.
 3. Return the coefficient, a `*`, and the variable name, in every other case.

# Arguments

  - `coeff`: Numeric coefficient for the variable.
  - `var`: Variable name as a string.

# Returns

  - `term_str::String`: The formatted term as a string.

# Related

  - [`_parse_equation`](@ref)
  - [`ParsingResult`](@ref)
"""
function format_term(coeff, var)::String
    return if isone(coeff)
        "$var"
    elseif isone(-coeff)
        "-$var"
    else
        "$(coeff)*$var"
    end
end
"""
    rethrow_parse_error(expr; side = :lhs)

Internal utility for error handling during equation parsing.

`rethrow_parse_error` is used to detect and handle incomplete or invalid expressions encountered while parsing constraint equations. It is called on both sides of an equation during parsing to ensure that the expressions are valid and complete. If an incomplete expression is detected, a `Meta.ParseError` is thrown; otherwise, the function returns `nothing`. The parser fails closed on an empty side rather than assuming zero, because a silently assumed zero is a constraint the author never wrote. A caller who means zero writes it.

# Algorithm

The method that Julia selects is the algorithm, and one method answers each shape a parsed side can take.

 1. `expr` is `Nothing`, which is what an empty side gives: raise, and name `side` in the message.
 2. `expr` is an `Expr`: raise when its head is `:incomplete`, and return `nothing` otherwise.
 3. `expr` is anything else, a number or a symbol among them: return `nothing`.

# Arguments

  - `expr`: The parsed Julia expression to check. Can be an `Expr`, `Nothing`, or any other type.
  - `side`: Symbol indicating which side of the equation is being checked (`:lhs` or `:rhs`). Used for error messages.

# Validation

  - `expr` is not `Nothing`. A `Meta.ParseError` naming `side` is thrown otherwise.
  - `expr.head != :incomplete`. A `Meta.ParseError` naming `side` and the expression is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`parse_equation`](@ref)
  - [`_parse_equation`](@ref)
"""
function rethrow_parse_error(::Any, side = :lhs)::Nothing
    return nothing
end
function rethrow_parse_error(::Nothing, side = :lhs)::Nothing
    # Fail closed: an empty side comes from a malformed (e.g. truncated) equation
    # string, never a legitimate constraint; assuming zero would silently create a
    # constraint the author did not write.
    return throw(Meta.ParseError("$side of equation is empty; write an explicit zero if that is intended."))
end
function rethrow_parse_error(expr::Expr, side = :lhs)::Nothing
    @argcheck(expr.head != :incomplete,
              Meta.ParseError("$side is an incomplete expression.\n$expr"))
    return nothing
end

export UniverseSets, PartialLinearConstraint, LinearConstraint, ParsingResult,
       estimator_to_val, UniformValues
