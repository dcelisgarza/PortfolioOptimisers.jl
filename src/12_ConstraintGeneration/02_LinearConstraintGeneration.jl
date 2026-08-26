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
  - `prefix`: The axis prefix the missing key must carry, `xkey` or `fkey`.

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
  - `claimed`: The other declared axis prefixes, `uxkey`, `fkey`, `ufkey` and `zkey`.

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
$(DocStringExtensions.TYPEDEF)

Declares the universes a portfolio problem is written against, and any groupings or partitions of them.

Constraint generation and the estimator routines read it to expand group references, to map a group name to its member list, and to validate membership.

It **declares every axis it carries**: `xkey`/`uxkey` for assets, `fkey`/`ufkey` for factors, and `zkey` for features. Assets are the *primary* axis — `haskey(dict, xkey)` is required, and it is the axis a view slices. The factor and feature axes are **optional**: requiring either would invalidate every sets object built for a problem with no factor model or no feature program, so a consumer that needs one and does not find it throws at the point of need rather than at construction.

If a key in `dict` starts with the same value as `xkey`, it means that the corresponding group must have the same length as the asset universe, `dict[xkey]`. This is useful for defining partitions of the asset universe, for example when using [`asset_sets_matrix`](@ref) with [`NestedClustered`](@ref).

If a key in `dict` starts with the same value as `uxkey`, it identifies a unique-entry group variant. The corresponding `xkey`-prefixed group must exist in `dict` with the same length as the asset universe, and is used to match each asset to a unique entry from the `uxkey`-prefixed group. This enables constraint generation using unique entries even in [`NestedClustered`](@ref) optimisations.

The `fkey`/`ufkey` prefixes mean the same thing on the factor axis, but they buy something different. On the asset side the conventions serve *views*; factors are never sliced by an asset index, so on the factor side they buy length validation at construction and one shared mental model.

`zkey` has **no prefix convention at all**, and that asymmetry is the point. `xkey` and `fkey` each have a unique-entry sibling because each names an axis that *partitions are written over*; nothing is written over the feature axis. A graded feature program's taxonomy keys are `xkey`-prefixed and asset-length, and its column nodes are named directly out of the flat list `dict[zkey]`. So `zkey` carries exactly one rule — `allunique(dict[zkey])`, so [`ReturnsResult`](@ref)'s own uniqueness check cannot be reached with a duplicate — and no length rule whatever.

A key matching none of the four prefixes is a plain group: expanded by name and **axis-blind**, which is why a factor group needs no machinery of its own.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    UniverseSets(;
        xkey::AbstractString = "nx",
        uxkey::AbstractString = "ux",
        fkey::AbstractString = "nf",
        ufkey::AbstractString = "uf",
        zkey::AbstractString = "nz",
        dict::AbstractDict{<:AbstractString, <:Any}
    ) -> UniverseSets

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(dict)`.
  - `haskey(dict, xkey)`.
  - No two of `xkey`, `uxkey`, `fkey`, `ufkey`, `zkey` may be a prefix of one another (20 ordered checks, which also rules out any two being equal).
  - If `haskey(dict, zkey)`, `allunique(dict[zkey])`.
  - If a key in `dict` starts with the same value as `xkey`, `length(dict[k]) == length(dict[xkey])`.
  - If a key in `dict` starts with the same value as `uxkey`, there must be a corresponding key in `dict` where the `uxkey` prefix is replaced by the `xkey` prefix, and its length must equal `length(dict[xkey])`.
  - If a key in `dict` starts with the same value as `fkey`, `haskey(dict, fkey)` and `length(dict[k]) == length(dict[fkey])`.
  - If a key in `dict` starts with the same value as `ufkey`, there must be a corresponding key in `dict` where the `ufkey` prefix is replaced by the `fkey` prefix, and its length must equal `length(dict[fkey])`.

## View parameters

`UniverseSets` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the asset index alone. It drops every further positional argument, because no axis but the asset axis is sliced.
  - Every `xkey`-prefixed entry of `dict` is sliced to the selected assets, and every `uxkey`-prefixed entry is rebuilt from the sliced partition it names.
  - The `fkey`-, `ufkey`- and `zkey`-prefixed entries, and every plain group, are carried through unchanged. [`port_opt_view`](@ref) states why each axis is exempt.
  - The five key prefixes are carried through unchanged, so the viewed value declares the same axes as the original.

# Examples

```jldoctest
julia> UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"group1\" => [\"A\", \"B\"]))
UniverseSets
   xkey ┼ String: "nx"
  uxkey ┼ String: "ux"
   fkey ┼ String: "nf"
  ufkey ┼ String: "uf"
   zkey ┼ String: "nz"
   dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"], "group1" => ["A", "B"])
```

# Related

  - [`replace_group_by_assets`](@ref)
  - [`estimator_to_val`](@ref)
  - [`linear_constraints`](@ref)
  - [`factor_universe`](@ref)
  - [`feature_universe`](@ref)
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
    $(field_dict[:us_fkey])
    """
    fkey
    """
    $(field_dict[:us_ufkey])
    """
    ufkey
    """
    $(field_dict[:us_zkey])
    """
    zkey
    """
    $(field_dict[:dict])
    """
    dict
    function UniverseSets(xkey::AbstractString, uxkey::AbstractString, fkey::AbstractString,
                          ufkey::AbstractString, zkey::AbstractString,
                          dict::AbstractDict{<:AbstractString, <:Any})::UniverseSets
        @argcheck(!isempty(dict), IsEmptyError)
        @argcheck(haskey(dict, xkey),
                  KeyError("$xkey (the asset universe), required by UniverseSets. The asset axis is the one mandatory axis: correct the spelling$(suggest_declared_key(xkey, unclaimed_sets_keys(dict, (uxkey, fkey, ufkey, zkey)))), pass `xkey = <the key you wrote>`, or add `$xkey => <asset names>` to `dict`."))
        knames = ("xkey", "uxkey", "fkey", "ufkey", "zkey")
        kvals = (xkey, uxkey, fkey, ufkey, zkey)
        for i in eachindex(kvals), j in eachindex(kvals)
            i == j && continue
            @argcheck(!startswith(kvals[i], kvals[j]),
                      ArgumentError("$(knames[i]) ($(kvals[i])) must not start with $(knames[j]) ($(kvals[j]))"))
        end
        if haskey(dict, zkey)
            @argcheck(allunique(dict[zkey]),
                      ArgumentError("the declared feature axis `$zkey` must not repeat a node, because a duplicate would silently merge two columns and would be rejected by `ReturnsResult`'s own `nz` uniqueness check anyway"))
        end
        for k in setdiff(keys(dict), (xkey, fkey, zkey))
            if startswith(k, xkey)
                @argcheck(length(dict[k]) == length(dict[xkey]),
                          DimensionMismatch("the asset partition `$k` and the asset universe `$xkey` disagree on how many assets there are. Got\nlength(dict[$k]) => $(length(dict[k]))\nlength(dict[$xkey]) => $(length(dict[xkey]))"))
            elseif startswith(k, uxkey)
                tmp_key = xkey * chopprefix(k, uxkey)
                @argcheck(haskey(dict, tmp_key),
                          KeyError("$tmp_key (the asset partition), required by the unique-entry asset group $k. Every `$uxkey`-prefixed group names the `$xkey`-prefixed partition it draws its entries from: correct the spelling$(suggest_declared_key(tmp_key, prefixed_sets_keys(dict, xkey))), or add `$tmp_key => <one group per asset>` to `dict`."))
                @argcheck(length(dict[tmp_key]) == length(dict[xkey]),
                          DimensionMismatch("the asset partition `$tmp_key`, required by the unique-entry asset group `$k`, and the asset universe `$xkey` disagree on how many assets there are. Got\nlength(dict[$tmp_key]) => $(length(dict[tmp_key]))\nlength(dict[$xkey]) => $(length(dict[xkey]))"))
            elseif startswith(k, fkey)
                @argcheck(haskey(dict, fkey),
                          KeyError("$fkey (the factor universe), required by the factor partition $k. A `$fkey`-prefixed key declares a partition of the factor axis, so the axis itself must be declared: add `$fkey => <factor names>` to `dict`, or rename `$k` if it was never meant to be a factor partition."))
                @argcheck(length(dict[k]) == length(dict[fkey]),
                          DimensionMismatch("the factor partition `$k` and the factor universe `$fkey` disagree on how many factors there are. Got\nlength(dict[$k]) => $(length(dict[k]))\nlength(dict[$fkey]) => $(length(dict[fkey]))"))
            elseif startswith(k, ufkey)
                @argcheck(haskey(dict, fkey),
                          KeyError("$fkey (the factor universe), required by the unique-entry factor group $k. A `$ufkey`-prefixed key summarises a partition of the factor axis, so the axis itself must be declared: add `$fkey => <factor names>` to `dict`, or rename `$k` if it was never meant to be a factor group."))
                tmp_key = fkey * chopprefix(k, ufkey)
                @argcheck(haskey(dict, tmp_key),
                          KeyError("$tmp_key (the factor partition), required by the unique-entry factor group $k. Every `$ufkey`-prefixed group names the `$fkey`-prefixed partition it draws its entries from: correct the spelling$(suggest_declared_key(tmp_key, prefixed_sets_keys(dict, fkey))), or add `$tmp_key => <one group per factor>` to `dict`."))
                @argcheck(length(dict[tmp_key]) == length(dict[fkey]),
                          DimensionMismatch("the factor partition `$tmp_key`, required by the unique-entry factor group `$k`, and the factor universe `$fkey` disagree on how many factors there are. Got\nlength(dict[$tmp_key]) => $(length(dict[tmp_key]))\nlength(dict[$fkey]) => $(length(dict[fkey]))"))
            end
        end
        return new{typeof(xkey), typeof(uxkey), typeof(fkey), typeof(ufkey), typeof(zkey),
                   typeof(dict)}(xkey, uxkey, fkey, ufkey, zkey, dict)
    end
end
function UniverseSets(; xkey::AbstractString = "nx", uxkey::AbstractString = "ux",
                      fkey::AbstractString = "nf", ufkey::AbstractString = "uf",
                      zkey::AbstractString = "nz",
                      dict::AbstractDict{<:AbstractString, <:Any})::UniverseSets
    return UniverseSets(xkey, uxkey, fkey, ufkey, zkey, dict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`UniverseSets`](@ref) restricted to the assets at index `i`.

The asset axis is the only axis this view slices, and the other two are exempt for two different reasons. The factor axis is exempt because an asset index has no meaning on it. Declaring that axis is what makes the exemption a property of the *data*: before the declaration, a factor-flavoured sets sitting in a `@vprop` field was sliced by asset indices and failed with a length mismatch, and the only defence was omitting the annotation by hand, field by field. There is deliberately **no factor-index arity** either. `port_opt_view(rd, i, j, k)` can slice `rd.nf`, but no internal caller passes a non-colon `k`, so a user who slices factors updates their sets themselves.

The feature axis is exempt although some of its nodes *are* assets. It is left alone because the axis is **declared** rather than derived: the caller wrote the node list down, so it is the program's coordinate system and not a summary of the current universe. That is what makes `size(Z, 2)` **fold-invariant** for a graded [`asset_sets_features`](@ref) program — exactly the opposite of the group-name-key path, where the viewed producer rebuilds the axis from the viewed taxonomy and a group left with no members disappears. The consequence is accepted rather than filtered: an asset node whose asset the view dropped survives as an **all-zero column**.

# Algorithm

 1. Read `xkey` and `uxkey` from `sets`, and open an empty dictionary `dict` of the type `sets.dict` has.
 2. For an entry of `sets.dict` whose key starts with `xkey`, take `view(v, i)`, the group restricted to the selected assets.
 3. For an entry whose key starts with `uxkey`, take the unique entries of the `xkey`-prefixed partition it names, restricted to `i`. The unique-entry group is therefore derived from the sliced partition and never from the original one.
 4. Carry every other entry through unchanged, into the same `dict`. The `fkey`-, `ufkey`- and `zkey`-prefixed entries, and every plain group, come back bit-identical.
 5. Return the [`UniverseSets`](@ref) built from `dict` and the five unchanged key prefixes, which revalidates the prefix grammar over the viewed universe.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) to view.
  - `i`: The asset index selection.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `sets::UniverseSets`: A new [`UniverseSets`](@ref) over the selected assets, declaring the same five axes as the original.

# Related

  - [`UniverseSets`](@ref)
  - [`asset_sets_features`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(sets::UniverseSets, i, args...)::UniverseSets
    xkey = sets.xkey
    uxkey = sets.uxkey
    dict = typeof(sets.dict)()
    for (k, v) in sets.dict
        if startswith(k, xkey)
            v = view(v, i)
        elseif startswith(k, uxkey)
            v = unique(view(sets.dict[xkey * chopprefix(k, uxkey)], i))
        end
        push!(dict, k => v)
    end
    return UniverseSets(; xkey = xkey, uxkey = uxkey, fkey = sets.fkey, ufkey = sets.ufkey,
                        zkey = sets.zkey, dict = dict)
end
"""
    factor_universe(sets::UniverseSets, K::Integer, need::AbstractString,
                    source::AbstractString) -> VecStr

Read the **declared** factor universe, `sets.dict[sets.fkey]`, checking that it exists and that it agrees with `source` — the `observations × factors` matrix whose `K` columns it must name — on how many factors there are.

The factor axis is optional on [`UniverseSets`](@ref) but is not optional for a consumer written against it, so the failure has to be diagnosed at the point of need. Both messages name `sets.fkey` and the matrix, because the two are what a caller has to reconcile: a user arriving from the pre-declaration shape put the factor names under `xkey` and would otherwise be told about an *asset* universe they never wrote in.

One helper therefore serves every consumer of the axis, and none of them re-encodes the checks.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) whose factor axis is read.
  - `K`: The number of columns of `source`, which the declared axis must name.
  - `need`: Names the consumer in both diagnostic messages, for example `"a FactorSpace constraint"`.
  - `source`: Names the matrix in both diagnostic messages, for example `"rr.M"` or `"F"`.

# Validation

  - `haskey(sets.dict, sets.fkey)`. A `KeyError` naming `need` is thrown otherwise.
  - `length(sets.dict[sets.fkey]) == K`. A `DimensionMismatch` naming `source` is thrown otherwise.

# Returns

  - `nf::VecStr`: The declared factor names, in the column order of `source`.

# Related

  - [`UniverseSets`](@ref)
  - [`constraint_space_basis`](@ref)
  - [`FactorBlackLittermanPrior`](@ref)
  - [`feature_universe`](@ref): the same helper for the feature axis, which carries no arity to reconcile.
"""
function factor_universe(sets::UniverseSets, K::Integer, need::AbstractString,
                         source::AbstractString)
    fkey = sets.fkey
    @argcheck(haskey(sets.dict, fkey),
              KeyError("$fkey (the factor universe), required by $need. The factor axis is optional on UniverseSets; it is not optional here: add `sets.fkey => <factor names>` to `sets.dict`, in the column order of `$source`."))
    nf = sets.dict[fkey]
    @argcheck(length(nf) == K,
              DimensionMismatch("`$source` and the declared factor axis disagree on how many factors there are. Got\nsize($source, 2) => $K\nlength(sets.dict[$fkey]) => $(length(nf))"))
    return nf
end
"""
    feature_universe(sets::UniverseSets, need::AbstractString) -> VecStr

Read the **declared** feature axis, `sets.dict[sets.zkey]`, checking that it exists.

The sibling of [`factor_universe`](@ref), written the same way and for the same reason: the axis is optional on [`UniverseSets`](@ref) but is not optional for a consumer written against it, so the failure is diagnosed at the point of need, by one shared helper whose message names the key and says what to add. Existence is the whole check, because the feature axis has no matrix to be reconciled against. It **defines** the width instead: [`asset_sets_features`](@ref) allocates `assets × length(nz)` from this list.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) whose feature axis is read.
  - `need`: Names the consumer in the diagnostic message, for example `"a graded feature program"`.

# Validation

  - `haskey(sets.dict, sets.zkey)`. A `KeyError` naming `need` is thrown otherwise.

# Returns

  - `nz::VecStr`: The declared feature node names, in the column order the feature matrix is to have.

# Related

  - [`UniverseSets`](@ref)
  - [`factor_universe`](@ref): the same helper for the factor axis, which also reconciles the axis against a matrix that already exists.
  - [`asset_sets_features`](@ref)
  - [`asset_sets_feature_names`](@ref)
"""
function feature_universe(sets::UniverseSets, need::AbstractString)
    zkey = sets.zkey
    @argcheck(haskey(sets.dict, zkey),
              KeyError("$zkey (the declared feature axis), required by $need. The feature axis is optional on UniverseSets; it is not optional here: add `sets.zkey => <feature node names>` to `sets.dict`, in the column order the feature matrix is to have."))
    return sets.dict[zkey]
end
"""
    name_to_val!(nx::VecStr, sdict::AbstractDict, key::Any, val::Number,
                 arr::VecNum, strict::Bool, nxkey::AbstractString)

Set values in a vector for the asset or the group of assets that `key` names.

`name_to_val!` resolves `key` through [`resolve_axis_name`](@ref) — an asset name resolves to itself, a group name expands to its members — maps the result to indices in the asset universe `nx`, and sets the corresponding entries of `arr` to `val`. If `key` names neither, the function either throws an error or issues a warning, depending on the `strict` flag. Every diagnostic message names the *size* of the universe and never the universe itself or the input value dictionary, because each is routed through a shared message builder in `01_Base.jl`.

# Algorithm

 1. Resolve `key` through [`resolve_axis_name`](@ref), giving `members`. An asset name resolves to itself, and a group name expands to a copy of its member list. An asset name takes precedence over a group name of the same spelling.
 2. Report through [`strict_diagnostic`](@ref) and return when `members` is `nothing`, because `key` names neither an asset nor a group. The suggestion pool is widened from `nx` to `nx` together with the keys of `sdict`, because a missing name may be a mistyped asset or a mistyped group.
 3. Map `members` to positions in `nx` with [`axis_name_indices`](@ref), giving `idx`. Members that miss the universe are dropped, and they are reported once through [`strict_diagnostic`](@ref).
 4. Set the entries of `arr` at `idx` to `val`.

# Arguments

  - `nx`: Vector of asset names.
  - `sdict`: Dictionary mapping group names to vectors of asset names. It is never modified, because [`resolve_axis_name`](@ref) returns a copy of the member list.
  - `key`: Name of the asset or the group of assets to set values for.
  - `val`: The value to assign.
  - `arr`: The array to be modified in-place.
  - `strict`: If `true`, throws an error if `key` resolves to nothing; if `false`, issues a warning.
  - `nxkey`: Name of the asset-universe key in `sets.dict` (e.g. `"nx"`), used only to name the universe in the diagnostic message — see [`unknown_variable_msg`](@ref) / [`missing_group_assets_msg`](@ref).

# Validation

  - `key` names an asset of `nx` or a group of `sdict`. An `ArgumentError` is thrown when `strict` is `true`, and a warning is issued otherwise.
  - Every member of a resolved group names an entry of `nx`. A member that misses the universe is dropped, and the drop raises when `strict` is `true` and issues a warning otherwise.

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
                      strict::Bool, nxkey::AbstractString)::Nothing
    members = resolve_axis_name(key, nx, sdict)
    if isnothing(members)
        # A missing key may be a mistyped asset *or* a mistyped group/set name, so widen the
        # suggestion pool beyond the raw universe to include the group/set keys.
        return strict_diagnostic(unknown_variable_msg(key, nx, nxkey;
                                                      candidates = [nx;
                                                                    collect(keys(sdict))]),
                                 strict)
    end
    idx = axis_name_indices(members, nx,
                            m -> strict_diagnostic(missing_group_assets_msg(key, m, nx,
                                                                            nxkey), strict))
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
 4. For each `(key, val)` pair of `dict`, in the order `dict` iterates in, write `val` into `arr` through [`name_to_val!`](@ref). A key that names an asset writes one entry, a key that names a group writes one entry per member, and a key that names neither is reported through the `strict` flag.
 5. Return `arr`.

# Arguments

  - `dict`: A dictionary, vector of pairs, or single pair mapping asset or group names to values.
  - `sets`: The [`UniverseSets`](@ref) containing the asset universe and group definitions.
  - `val`: The value assigned to every asset before `dict` is applied. `nothing` means `zero(datatype)`.
  - `key`: (Optional) Key in the [`UniverseSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`UniverseSets`](@ref).
  - `datatype`: Element type of the value the array is filled with when `val` is `nothing`.
  - `strict`: If `true`, throws an error if a key in `dict` is not found in the asset sets; if `false`, issues a warning.

# Validation

  - A key of `dict` that names neither an asset nor a group raises an `ArgumentError` when `strict` is `true`. A warning is issued otherwise.

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
    arr = fill(val, length(nx))
    for (key, val) in dict
        name_to_val!(nx, sets.dict, key, val, arr, strict, nxkey)
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
    name_to_val!(nx, sets.dict, key, val, arr, strict, nxkey)
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

  - [`AbstractEstimatorValueAlgorithm`](@ref)
  - [`WeightBoundsEstimator`](@ref)
  - [`WeightBounds`](@ref)
"""
struct UniformValues <: AbstractEstimatorValueAlgorithm end
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

Enumerated table of the functions permitted in equation parsing, mapping each allowed name directly to its function object. Evaluating constraint/view strings crosses a trust boundary (config files, spreadsheets, UI), so the parser must be able to call *only* these 16 mathematical functions. Using an explicit `Symbol => Function` table — rather than resolving a name against `Base` with `getfield(Base, fname)` — bounds that capability to exactly this table: a name absent from the keys fails closed with a `Meta.ParseError`, and the set of callable functions cannot drift from the set of allowed names, because they are the same list. See `docs/adr/0025-enumerated-parser-allowlist.md`.

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
"""
    _parse_equation(lhs, opstr::AbstractString, rhs; datatype::DataType = Float64)

Parse and canonicalise a linear constraint equation from Julia expressions.

`_parse_equation` takes the left-hand side (`lhs`) and right-hand side (`rhs`) of a constraint equation, both as Julia expressions, and a comparison operator string (`opstr`). It evaluates numeric functions, moves all terms to the left-hand side, collects coefficients and variables, and returns a [`ParsingResult`](@ref) with the canonicalised equation.

# Algorithm

 1. Fold the constant subexpressions of both sides with [`eval_numeric_functions`](@ref), giving `lexpr` and `rexpr`, and check each with [`rethrow_parse_error`](@ref).
 2. Build `diff_expr`, the expression `lexpr - (rexpr)`. This moves every term of the equation to the left-hand side.
 3. Walk `diff_expr` with [`_collect_terms`](@ref), giving `terms`, one `(coefficient, variable)` pair per term.
 4. Accumulate `terms` into `varmap`, which holds the summed coefficient of each variable name, and into `constant`, the sum of the coefficients that carry no variable.
 5. Read `variables` and `coefficients` off `varmap`, and take `rhs_val` as the negated `constant`. This moves the constant to the right-hand side.
 6. Render each pair with [`format_term`](@ref), join the renderings with `+`, and fold `+ -` into `-`, giving the canonical string `formatted`.
 7. Return the [`ParsingResult`](@ref) built from `variables`, `coefficients`, `opstr`, `rhs_val` and `formatted`.

# Arguments

  - `lhs`: Left-hand side of the equation as a Julia expression.
  - `opstr`: Comparison operator as a string.
  - `rhs`: Right-hand side of the equation as a Julia expression.
  - `datatype`: Numeric type for coefficients and right-hand side.

# Returns

  - `res::ParsingResult`: Structured result with canonicalised variables, coefficients, operator, right-hand side, and formatted equation. The order of `vars` is the order the variable map iterates in, and it is not the order the equation was written in.

# Related

  - [`ParsingResult`](@ref)
  - [`parse_equation`](@ref)
  - [`eval_numeric_functions`](@ref)
  - [`_collect_terms`](@ref)
  - [`format_term`](@ref)
"""
function _parse_equation(lhs, opstr::AbstractString, rhs,
                         datatype::DataType = Float64)::ParsingResult
    # 3. Evaluate numeric functions on both sides
    lexpr = eval_numeric_functions(lhs, datatype)
    rethrow_parse_error(lexpr, :lhs)
    rexpr = eval_numeric_functions(rhs, datatype)
    rethrow_parse_error(rexpr, :rhs)

    # 4. Move all terms to LHS: lhs - rhs == 0
    diff_expr = :($lexpr - ($rexpr))

    # 5. Expand and collect like terms
    terms = _collect_terms(diff_expr, datatype)

    # 6. Separate variables and constant
    varmap = Dict{String, datatype}()
    constant::datatype = 0.0
    for (coeff, var) in terms
        if isnothing(var)
            constant += coeff
        else
            varmap[var] = get(varmap, var, zero(datatype)) + coeff
        end
    end

    # 7. Move constant to RHS, variables to LHS
    variables = collect(keys(varmap))
    coefficients = [varmap[v] for v in variables]
    rhs_val = -constant

    # 8. Format the simplified expression
    lhs_str = join([format_term(coeff, var)
                    for (coeff, var) in zip(coefficients, variables)], " + ")
    lhs_str = replace(lhs_str, "+ -" => "-", "  " => " ")
    rhs_str = string(rhs_val)
    formatted = strip("$lhs_str $opstr $rhs_str")
    return ParsingResult(variables, coefficients, opstr, rhs_val, formatted)
end
"""
    parse_equation(eqn::EqnType;
                   ops1::Tuple = ("==", "<=", ">="), ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                   datatype::DataType = Float64, kwargs...)

Parse a linear constraint equation from a string into a structured [`ParsingResult`](@ref).

An equation string crosses a trust boundary, so both entry shapes carry a limit from `EQUATION_LIMITS[]` before any recursive walk runs. The string form is capped on length, which bounds the depth its parse can reach, and the pre-built `Expr` form is capped on depth directly, because no length applies to it. `docs/adr/0027-cap-equation-parser-recursion.md` owns both limits.

# Algorithm

The method that Julia selects is the algorithm, and one method answers each shape of `eqn`.

 1. `eqn` is a vector: apply this function to each element, and return the vector of results.
 2. `eqn` is a string: check its length against `EQUATION_LIMITS[].max_length`, and refuse the pattern `++`.
 3. Find the first operator of `ops1` that occurs in the string, giving `opstr`, and split the string on it into `lhs` and `rhs`.
 4. Parse both parts with `Meta.parse`, giving `lexpr` and `rexpr`, and check each with [`rethrow_parse_error`](@ref).
 5. `eqn` is an `Expr`: check its depth against `EQUATION_LIMITS[].max_depth` with [`_expr_depth_exceeds`](@ref), and refuse a `++` pattern with [`has_invalid_plus`](@ref).
 6. Check that the head of the expression is a call and is exactly one operator of `ops2`, giving `opstr`, and read `lhs` and `rhs` off the arguments of the call.
 7. Hand `opstr` and the two sides to [`_parse_equation`](@ref), which canonicalises them and builds the [`ParsingResult`](@ref).

# Arguments

  - `eqn`: The equation string to parse.

      + `eqn::AbstractVector`: Each element needs to meet the criteria below.

      + `eqn::AbstractString`: Must contain exactly one comparison operator from `ops1`.

          * `ops1`: Tuple of valid comparison operators as strings.

      + `eqn::Expr`: Must contain exactly one comparison operator from `ops1`.

          * `ops2`: Tuple of valid comparison operator expressions.

  - `datatype`: The numeric type to use for coefficients and right-hand side.

  - `kwargs...`: Additional keyword arguments, ignored.

# Validation

  - `length(eqn) <= EQUATION_LIMITS[].max_length`, for the string form. A `Meta.ParseError` naming both lengths is thrown otherwise.
  - The expression tree of `eqn` is no deeper than `EQUATION_LIMITS[].max_depth`, for the `Expr` form. A `Meta.ParseError` naming the limit is thrown otherwise.
  - `eqn` holds no `++` pattern.
  - `eqn` holds exactly one comparison operator, from `ops1` for the string form and from `ops2` for the `Expr` form.
  - The head of the `Expr` form is a call.
  - Neither side of the equation is empty or incomplete, which [`rethrow_parse_error`](@ref) checks.

# Returns

  - If `eqn::Str_Expr`:

      + `res::ParsingResult`: Structured parsing result.

  - If `eqn::AbstractVector`:

      + `res::Vector{ParsingResult}`: Vector of structured parsing results.

# Examples

```jldoctest
julia> parse_equation(\"w_A + 2w_B <= 1\")
ParsingResult
  vars ┼ Vector{String}: ["w_A", "w_B"]
  coef ┼ Vector{Float64}: [1.0, 2.0]
    op ┼ String: "<="
   rhs ┼ Float64: 1.0
   eqn ┴ SubString{String}: "w_A + 2.0*w_B <= 1.0"
```

# Related

  - [`ParsingResult`](@ref)
  - [`_parse_equation`](@ref)
  - [`rethrow_parse_error`](@ref)
  - [`has_invalid_plus`](@ref)
  - [`_expr_depth_exceeds`](@ref)
  - [`replace_group_by_assets`](@ref)
"""
function parse_equation(eqn::AbstractString; ops1::Tuple = ("==", "<=", ">="),
                        datatype::DataType = Float64, kwargs...)::ParsingResult
    # Trust boundary: cap the untrusted string length before `Meta.parse` and the
    # recursive expression walks, so a deeply nested string cannot exhaust the stack.
    # Bounding the length bounds the achievable AST depth of the string form.
    lim = EQUATION_LIMITS[]
    @argcheck(length(eqn) <= lim.max_length,
              Meta.ParseError("Equation string is too long ($(length(eqn)) > $(lim.max_length) characters)."))
    @argcheck(!occursin("++", eqn),
              Meta.ParseError("Invalid operator '++' detected in equation."))
    # 1. Identify the comparison operator
    op = findfirst(op -> occursin(op, eqn), ops1)
    @argcheck(!isnothing(op),
              Meta.ParseError("Equation must contain a valid comparison operator $(join(ops1,", ")) .\n$(eqn)"))
    opstr = ops1[op]
    parts = split(eqn, opstr)
    @argcheck(length(parts) == 2,
              Meta.ParseError("Equation must have exactly one comparison operator.\n$(eqn)"))
    lhs, rhs = strip.(parts)
    # 2. Parse both sides into Julia expressions
    lexpr = Meta.parse(lhs)
    rethrow_parse_error(lexpr, :lhs)
    rexpr = Meta.parse(rhs)
    rethrow_parse_error(rexpr, :rhs)
    return _parse_equation(lexpr, opstr, rexpr, datatype)
end
"""
    has_invalid_plus(expr)

Check whether a Julia expression contains an invalid `+` operator in a constraint context.

Internal helper used during linear constraint parsing to detect unsupported `+` operator usage in constraint expressions. It is the `Expr` counterpart of the `++` check the string form of [`parse_equation`](@ref) runs on the raw text.

# Algorithm

 1. Return `false` when `expr` is not a call, because only a call can carry the head this function refuses.
 2. Return `true` when the head of the call is the `++` operator.
 3. Apply this function to every argument of the call that is itself an expression, and return `true` when any of them does.

# Arguments

  - `expr`: Julia expression to check.

# Returns

  - `Bool`: `true` if the expression contains an invalid `+`, `false` otherwise.

# Related

  - [`parse_equation`](@ref)
  - [`_expr_depth_exceeds`](@ref)
"""
function has_invalid_plus(expr)::Bool
    if !(isa(expr, Expr) && expr.head == :call)
        return false
    end
    # Check for nested :+ calls (e.g., :(+(+(a, b), c))) or more than two arguments
    if expr.args[1] == :++
        # If any argument is itself a :+ call, that's suspicious (from "++")
        return true
    end
    # Recurse into sub-expressions
    return any(has_invalid_plus(arg) for arg in expr.args[2:end] if isa(arg, Expr))
end
"""
    _expr_depth_exceeds(x, limit::Integer) -> Bool

Return `true` if the expression tree `x` is deeper than `limit`.

Guards the `Expr` form of [`parse_equation`](@ref) against a deeply nested AST that no
string length cap covers. The check itself recurses at most `limit + 1` frames deep and
short-circuits the moment the limit is breached, so it cannot exhaust the stack it protects.
`docs/adr/0027-cap-equation-parser-recursion.md` owns the limit this function is called with.

# Algorithm

 1. Return `true` when `limit` is negative, because the walk has already gone one level past the cap.
 2. Return `false` when `x` is not an expression, because a leaf adds no depth.
 3. Apply this function to every argument of `x`, with `limit` lowered by one, and return `true` when any of them does. The scan stops at the first argument that answers `true`.

# Arguments

  - `x`: The expression tree to measure.
  - `limit`: The greatest depth the tree may have.

# Returns

  - `Bool`: `true` when the tree is deeper than `limit`, `false` otherwise.

# Related

  - [`parse_equation`](@ref)
  - [`has_invalid_plus`](@ref)
"""
function _expr_depth_exceeds(x, limit::Integer)::Bool
    if limit < 0
        return true
    end
    if !(isa(x, Expr))
        return false
    end
    return any(_expr_depth_exceeds(a, limit - 1) for a in x.args)
end
function parse_equation(expr::Expr; ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64, kwargs...)::ParsingResult
    # Trust-boundary defence for the pre-built-AST form (no string length cap applies):
    # reject an over-deep tree before the recursive walks below can exhaust the stack.
    lim = EQUATION_LIMITS[]
    @argcheck(!_expr_depth_exceeds(expr, lim.max_depth),
              Meta.ParseError("Equation expression is too deeply nested (exceeds depth $(lim.max_depth))."))
    # Recursively check for invalid "++" pattern in the expression tree
    @argcheck(!has_invalid_plus(expr),
              Meta.ParseError("Invalid operator pattern '++' detected in equation expression:\n$expr"))
    # Ensure the expression is a call to a valid comparison operator
    @argcheck(expr.head == :call,
              Meta.ParseError("Expression must be a function call (comparison operator expected):\n$expr"))
    # Count how many valid operators are present
    op_count = count(op -> expr.args[1] == op, ops2[2:end])
    @argcheck(op_count == 1,
              Meta.ParseError("Expression must contain a valid comparison operator $(join(ops2[2:end], ", ")) .\n$expr"))
    opstr = string(expr.args[1])
    lhs, rhs = expr.args[2], expr.args[3]
    return _parse_equation(lhs, opstr, rhs, datatype)
end
function parse_equation(eqn::VecStr_Expr; ops1::Tuple = ("==", "<=", ">="),
                        ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64)
    return parse_equation.(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
end
"""
    replace_group_by_assets(res::PR_VecPR, sets::UniverseSets, bl_flag::Bool = false,
                            ep_flag::Bool = false, rho_flag::Bool = false)

Expand group or special variable references in a [`ParsingResult`](@ref) to their corresponding asset names.

This function takes a [`ParsingResult`](@ref) containing variable names (which may include group names, `prior(...)` expressions, or correlation views like `(A, B)`), and replaces these with the actual asset names from the provided [`UniverseSets`](@ref). It supports Black-Litterman-style group expansion, entropy pooling prior views, and correlation view parsing for advanced constraint generation. When `res` is a vector of [`ParsingResult`](@ref) objects, the function is applied to each element of the vector.

# Mathematical definition

```math
\\begin{align}
c\\, g &\\to \\sum_{j=1}^{k} c\\, m_j\\,, \\\\
c\\, g &\\to \\sum_{j=1}^{k} \\frac{c}{k}\\, m_j\\,.
\\end{align}
```

Where:

  - ``g``: A group name written in the equation.
  - ``m_j``: The ``j``-th member of the group ``g``.
  - ``k``: The number of members of the group ``g``.
  - ``c``: The coefficient the group name carries.

The two lines are different operations. The first repeats the coefficient on every member, so the expanded row constrains the **sum** over the group. The second divides the coefficient by the member count, so the expanded row constrains the **mean** over the group. A group of one member is the only case in which the two agree.

# Algorithm

 1. Copy `res.vars` and `res.coef` into `variables_new` and `coeffs_new`, and open the empty accumulators `variables_tmp`, `coeffs_tmp` and `idx_rm`.
 2. For each variable name of `res.vars`, match it against the prior pattern `prior(...)` and against the correlation pattern `(a, b)`. The four combinations of the two matches select steps 3 to 6.
 3. A name matching neither pattern, with `rho_flag` false, is a plain name. Look it up in `sets.dict`, and leave it where it stands when the dictionary does not hold it, because a name that is not a group is already the name of one column. A group name expands to its members, each carrying the coefficient the mathematics above gives, and the index of the group joins `idx_rm`.
 4. A name matching the correlation pattern expands to one entry naming the two member lists, and that entry carries the coefficient of the view unchanged. A correlation view is one row over a pair of universes, so no coefficient is spread over members.
 5. A name matching the prior pattern expands the name inside `prior(...)` exactly as step 3 does, and wraps each member back in `prior(...)`.
 6. A name matching both patterns expands as step 4 does, and wraps each of the two member lists in `prior(...)`.
 7. Return `res` unchanged when nothing expanded, so an equation written in asset names costs no allocation.
 8. Delete the entries at `idx_rm` from `variables_new` and `coeffs_new`, append the two accumulators to them, and render the expanded equation string.
 9. Return the [`ParsingResult`](@ref) built from the new names and coefficients, together with the operator and the right-hand side of `res`, which the expansion leaves untouched.

# Arguments

  - `res`: A [`ParsingResult`](@ref) object containing variables and coefficients to be expanded.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `bl_flag`: Selects which of the two expansions above runs. `false` takes the first, which constrains the sum over the group. `true` takes the second, the Black-Litterman-style expansion, which constrains the mean.
  - `ep_flag`: If `true`, enables expansion of `prior(...)` expressions for entropy pooling.
  - `rho_flag`: If `true`, enables expansion of correlation views `(A, B)` for entropy pooling.

# Validation

The three flags are not independent, and five guards hold the grammar they describe.

  - `bl_flag` can only be `true` if both `ep_flag` and `rho_flag` are `false`.
  - `rho_flag` can only be `true` if `ep_flag` is also `true`.
  - The pattern `(a, b)` can only be used when `ep_flag` and `rho_flag` are both `true`.
  - The pattern `prior(a)` can only be used when `ep_flag` is `true`.
  - The pattern `prior(a, b)` can only be used when `rho_flag` is `true`.

Two further guards hold the shape of a correlation view.

  - A correlation view is written `(a, b)`, and a correlation view prior is written `prior(a, b)`.
  - Both sides of a correlation view name a group that `sets.dict` holds, and the two groups have the same number of members. A view whose two sides are both absent from `sets.dict` is skipped instead of raised on.

# Returns

  - `res::ParsingResult`: A new [`ParsingResult`](@ref) with all group and special variable references expanded to asset names.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\",
                           dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"group1\" => [\"A\", \"B\"]));

julia> res = parse_equation(\"group1 + 2C == 1\")
ParsingResult
  vars ┼ Vector{String}: ["C", "group1"]
  coef ┼ Vector{Float64}: [2.0, 1.0]
    op ┼ String: "=="
   rhs ┼ Float64: 1.0
   eqn ┴ SubString{String}: "2.0*C + group1 == 1.0"

julia> replace_group_by_assets(res, sets)
ParsingResult
  vars ┼ Vector{String}: ["C", "A", "B"]
  coef ┼ Vector{Float64}: [2.0, 1.0, 1.0]
    op ┼ String: "=="
   rhs ┼ Float64: 1.0
   eqn ┴ String: "2.0*C + A + B == 1.0"
```

# Related

  - [`UniverseSets`](@ref)
  - [`ParsingResult`](@ref)
  - [`parse_equation`](@ref)
  - [`get_linear_constraints`](@ref)
  - [`linear_constraints`](@ref)
"""
function replace_group_by_assets(res::ParsingResult, sets::UniverseSets,
                                 bl_flag::Bool = false, ep_flag::Bool = false,
                                 rho_flag::Bool = false)::ParsingResult
    @argcheck(!(bl_flag && (rho_flag || ep_flag)),
              ArgumentError("bl_flag can only be true if ep_flag and rho_flag are false. Got\nbl_flag => $(bl_flag)\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
    @argcheck(!(rho_flag && !ep_flag),
              ArgumentError("rho_flag can only be true if ep_flag is also true. Got\nrho_flag => $rho_flag\nep_flag => $ep_flag"))
    variables, coeffs = res.vars, res.coef
    variables_new = copy(variables)
    coeffs_new = copy(coeffs)
    variables_tmp = Vector{eltype(variables)}(undef, 0)
    coeffs_tmp = Vector{eltype(coeffs)}(undef, 0)
    idx_rm = Vector{Int}(undef, 0)
    prior_pattern = r"prior\(([^()]*)\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            n = match(corr_pattern, v)
            if isnothing(n) && !rho_flag
                asset = get(sets.dict, v, nothing)
                if isnothing(asset)
                    continue
                end
                c = !bl_flag ? coeffs[i] : coeffs[i] / length(asset)
                append!(variables_tmp, asset)
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
                push!(idx_rm, i)
            else
                @argcheck(ep_flag && rho_flag,
                          ArgumentError("The pattern '(a, b)' can only be used for rho_views (rho_flag is true) in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
                @argcheck(!isnothing(n),
                          ArgumentError("Correlation views can only be of the form '(a, b)'. Got\nv => $v"))
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1), IsNothingError)
                @argcheck(!isnothing(asset2), IsNothingError)
                @argcheck(length(asset1) == length(asset2), DimensionMismatch)
                push!(variables_tmp, "([$(join(asset1, ", "))], [$(join(asset2, ", "))])")
                push!(coeffs_tmp, coeffs[i])
                push!(idx_rm, i)
            end
        else
            @argcheck(ep_flag,
                      ArgumentError("The pattern 'prior(a)' can only be used in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)."))
            n = match(corr_pattern, v)
            if isnothing(n) && !rho_flag
                asset = get(sets.dict, v[7:(end - 1)], nothing)
                if isnothing(asset)
                    continue
                end
                c = !bl_flag ? coeffs[i] : coeffs[i] / length(asset)
                append!(variables_tmp, ["prior($a)" for a in asset])
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
                push!(idx_rm, i)
            else
                @argcheck(rho_flag,
                          ArgumentError("The pattern 'prior(a, b)' can only be used for rho_views (rho_flag is true) in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
                @argcheck(!isnothing(n),
                          ArgumentError("Correlation views prior can only be of the form 'prior(a, b)'. Got\nv => $v"))
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1), IsNothingError)
                @argcheck(!isnothing(asset2), IsNothingError)
                @argcheck(length(asset1) == length(asset2), DimensionMismatch)
                push!(variables_tmp,
                      "prior([$(join(asset1, ", "))], [$(join(asset2, ", "))])")
                push!(coeffs_tmp, coeffs[i])
                push!(idx_rm, i)
            end
        end
    end
    if isempty(variables_tmp)
        return res
    end
    deleteat!(variables_new, idx_rm)
    deleteat!(coeffs_new, idx_rm)
    append!(variables_new, variables_tmp)
    append!(coeffs_new, coeffs_tmp)
    # Render through the same `format_term` the unexpanded string uses, so one constraint
    # prints one way whether or not a group expanded.
    eqn = replace(join(format_term.(coeffs_new, variables_new), " + "), "+ -" => "-",
                  "  " => " ")
    return ParsingResult(variables_new, coeffs_new, res.op, res.rhs,
                         "$(eqn) $(res.op) $(res.rhs)")
end
function replace_group_by_assets(res::VecPR, sets::UniverseSets, args...)
    return replace_group_by_assets.(res, sets, args...)
end
"""
    universe_axis(sets::UniverseSets, key::AbstractString) -> String

Name of the axis the universe stored under `key` belongs to, read off the key itself: `"factor"` for anything carrying the `fkey` prefix, `"asset"` otherwise. It exists only so [`unknown_variable_msg`](@ref) and [`empty_row_msg`](@ref) can name the axis the user wrote in.

The **key** is the evidence, for both callers, and the reason is that both resolve names against `sets.dict[key]` and nothing else: whatever axis that universe belongs to is the axis a failed lookup failed on. [`get_black_litterman_views`](@ref) takes the key from the estimator that owns the views, and [`get_linear_constraints`](@ref) from the constraint space — [`FactorSpace`](@ref) resolving at `sets.fkey`. Reading it off the *re-basis* instead would be a second encoding of the same fact, and a worse one: a wrapped estimator carrying its own `key` overrides the space's, so a re-based row can legitimately resolve against a universe the loadings are not written in, and the message must name the universe that was searched.

The **prefix** rather than equality is what makes a factor group key (`"nf_sector"`) resolve as the factor axis too, and the disjoint-prefix rule [`UniverseSets`](@ref) enforces at construction is what makes that unambiguous.

# Algorithm

 1. Return `"factor"` when `key` starts with `sets.fkey`.
 2. Return `"asset"` in every other case.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) whose `fkey` names the factor axis.
  - `key`: The key the names were resolved against.

# Returns

  - `axis::String`: `"factor"` or `"asset"`, the word a diagnostic message uses to name the axis.

# Related

  - [`get_linear_constraints`](@ref)
  - [`get_black_litterman_views`](@ref)
  - [`UniverseSets`](@ref)
"""
function universe_axis(sets::UniverseSets, key::AbstractString)::String
    return ifelse(startswith(key, sets.fkey), "factor", "asset")
end
"""
    constraint_row_length(rr, nx::VecStr) -> Int

Length of the assembled constraint row. Without a re-basis this is the size of the universe the names resolve against; with one it is the number of *assets* the loadings project onto, because the projection is applied while the row is assembled and what leaves is an ordinary asset-space row.

# Algorithm

The method that Julia selects is the algorithm, and the re-basis selects it.

 1. `rr` is `nothing`: return the length of `nx`, the universe the names resolve against.
 2. `rr` is a regression result: return the number of rows of `rr.M`, which is the number of assets the loadings project onto.

# Arguments

  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space row.
  - `nx`: The universe the names resolve against.

# Returns

  - `N::Int`: The number of entries one assembled row has.

# Related

  - [`get_linear_constraints`](@ref)
  - [`constraint_row_term`](@ref)
"""
function constraint_row_length(::Nothing, nx::VecStr)::Int
    return length(nx)
end
function constraint_row_length(rr::AbstractRegressionResult, ::VecStr)::Int
    return size(rr.M, 1)
end
"""
    constraint_row_term(::Nothing, Ai, c)
    constraint_row_term(rr::AbstractRegressionResult, Ai, c)

Contribution of one matched variable to a constraint row.

Without a re-basis the contribution is the indicator `Ai` scaled by the coefficient `c`. With one it is the columns of the loadings that `Ai` selects, summed and scaled. The columns are **summed** rather than indexed by `findfirst`, so a factor universe carrying a duplicated name contributes every column bearing it, matching how the asset path treats a duplicated asset name.

`rr.M` is used, never `rr.L`: `M`'s columns are the named original factors, and a constraint must be *written* in names a user can put in an equation. Risk decomposition wants `L` and is right to; see ADR 0047.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{a}^\\intercal \\boldsymbol{w}_f &= \\boldsymbol{a}^\\intercal \\mathbf{M}^\\intercal \\boldsymbol{w}_a = (\\mathbf{M} \\boldsymbol{a})^\\intercal \\boldsymbol{w}_a\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{a}``: A constraint row written in factor names.
  - ``\\boldsymbol{w}_f``: The factor weights that row is written against.
  - ``\\boldsymbol{w}_a``: The asset weights the optimiser holds.
  - ``\\mathbf{M}``: The factor loadings, one column per named factor and one row per asset.

The identity is what lets a row written in factor names bind asset weights with no change of variables: the re-based row is an ordinary asset-space row over ``\\boldsymbol{w}_a``.

# Algorithm

The method that Julia selects is the algorithm, and the re-basis selects it.

 1. `rr` is `nothing`: return `Ai` scaled by `c`, one entry per name of the universe.
 2. `rr` is a regression result: sum the columns of `rr.M` that `Ai` selects, scale the sum by `c`, and return it. The value is asset-length whatever the row was written in.

# Arguments

  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space row.
  - `Ai`: The indicator of the matched name over the universe the names resolve against.
  - `c`: The coefficient the matched name carries.

# Returns

  - The contribution of this term to the row, of the length [`constraint_row_length`](@ref) gives.

# Related

  - [`get_linear_constraints`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`constraint_row_length`](@ref)
"""
function constraint_row_term(::Nothing, Ai, c)
    return Ai * c
end
function constraint_row_term(rr::AbstractRegressionResult, Ai, c)
    return vec(sum(view(rr.M, :, Ai); dims = 2)) * c
end
"""
    get_linear_constraints(lcs::PR_VecPR, sets::UniverseSets,
                           key::Option{<:AbstractString} = nothing;
                           datatype::DataType = Float64, strict::Bool = false,
                           rr::Option{<:AbstractRegressionResult} = nothing)

Convert parsed linear constraint equations into a `LinearConstraint` object.

`get_linear_constraints` takes one or more [`ParsingResult`](@ref) objects (as produced by [`parse_equation`](@ref)), expands variable names using the provided [`UniverseSets`](@ref), and assembles the corresponding constraint matrices and right-hand side vectors. The result is a [`LinearConstraint`](@ref) object containing both equality and inequality constraints, suitable for use in portfolio optimisation routines.

A row takes one of two shapes. Without `rr` it runs over the universe the names resolve against. With `rr` it runs over the assets, because the loadings re-base each term as the row is assembled and what leaves the function is an ordinary asset-space row.

# Algorithm

 1. Take `k` as `key`, or `sets.xkey` when `key` is `nothing`, read the universe `nx` from `sets.dict` under it, and name the axis with [`universe_axis`](@ref).
 2. Take `N`, the row length, from [`constraint_row_length`](@ref), and allocate the working row `At` of that length.
 3. Zero `At` for each parsing result, and start that result with no matched name.
 4. Build the indicator of each variable name of the result over `nx`. A name that matches no entry is reported through [`strict_diagnostic`](@ref) and is dropped, and the row is assembled from whatever did match.
 5. Add the contribution [`constraint_row_term`](@ref) gives for the name and its coefficient to `At`. With `rr` the contribution arrives already projected, so `At` is asset-length while it is accumulated.
 6. Report the row through [`strict_diagnostic`](@ref) and drop it when `At` is still zero. The message separates a row whose names missed the universe from a row whose names hit it and whose loadings annihilated it, because the second is not a typo for the reader to hunt.
 7. Read the sign and the inequality flag of the operator from [`comparison_sign_ineq_flag`](@ref), and scale the row and its right-hand side by the sign. That negates a `>=` row, so both senses of an inequality are written in the `<=` sense, which is the convention [`LinearConstraint`](@ref) states.
 8. Append the row to the inequality accumulator when the flag is `true`, and to the equality accumulator when it is `false`.
 9. Reshape each accumulator that holds a row into a matrix of `N` columns, and build the [`PartialLinearConstraint`](@ref) of that half.
10. Return the [`LinearConstraint`](@ref) holding the halves that were built, or `nothing` when neither half holds a row.

# Arguments

  - `lcs`: A single [`ParsingResult`](@ref) or a vector of such objects, representing parsed constraint equations.
  - `sets`: A [`UniverseSets`](@ref) object specifying the universes and groupings.
  - `key`: Key naming the universe the variables resolve against. Defaults to `sets.xkey`; a re-based constraint passes `sets.fkey`.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.
  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space constraint. See [`ExposureConstraintEstimator`](@ref) — callers do not pass this directly.

# Validation

  - `lcs` is non-empty, when it is a vector.
  - A variable name that matches no entry of the universe raises when `strict` is `true`, and issues a warning otherwise. The row is assembled from the names that did match either way.
  - A row whose terms all fall away raises when `strict` is `true`, and issues a warning otherwise. The row is dropped either way.
  - Each `op` is one of `"=="`, `"<="` or `">="`, which [`comparison_sign_ineq_flag`](@ref) enforces.

# Returns

  - `lcs::LinearConstraint`: An object containing the assembled equality and inequality constraints, or `nothing` if no constraints are present.

# Related

  - [`ParsingResult`](@ref)
  - [`LinearConstraint`](@ref)
  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`constraint_row_term`](@ref)
  - [`constraint_row_length`](@ref)
  - [`universe_axis`](@ref)
  - [`comparison_sign_ineq_flag`](@ref)
"""
function get_linear_constraints(lcs::PR_VecPR, sets::UniverseSets,
                                key::Option{<:AbstractString} = nothing;
                                datatype::DataType = Float64, strict::Bool = false,
                                rr::Option{<:AbstractRegressionResult} = nothing)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs), IsEmptyError)
    end
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    k = ifelse(isnothing(key), sets.xkey, key)
    nx = sets.dict[k]
    axis = universe_axis(sets, k)
    N = constraint_row_length(rr, nx)
    At = Vector{datatype}(undef, N)
    for lc in lcs
        fill!(At, zero(eltype(At)))
        matched = false
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = unknown_variable_msg(v, nx, k; axis = axis)
                strict_diagnostic(msg, strict)
                continue
            end
            matched = true
            At .+= constraint_row_term(rr, Ai, c)
        end
        if !any(!iszero, At)
            # Two distinct failures land here once a re-basis is possible: the names missed the
            # universe (`matched === false`, the pre-existing diagnosis), or they hit it and the
            # loadings annihilated them. Reporting the first for the second would send a user
            # hunting for a typo that is not there.
            msg = if matched && !isnothing(rr)
                empty_projected_row_msg(lc.eqn, nx, k, N)
            else
                empty_row_msg(lc.eqn, nx, k; axis = axis)
            end
            strict_diagnostic(msg, strict)
            continue
        end
        d, flag = comparison_sign_ineq_flag(lc.op)
        A = At .* d
        B = lc.rhs * d
        if flag
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    ineq_flag = !isempty(A_ineq)
    eq_flag = !isempty(A_eq)
    ineq = nothing
    eq = nothing
    if ineq_flag
        A_ineq = transpose(reshape(A_ineq, N, :))
        ineq = PartialLinearConstraint(; A = A_ineq, B = B_ineq)
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, N, :))
        eq = PartialLinearConstraint(; A = A_eq, B = B_eq)
    end
    return if ineq_flag || eq_flag
        LinearConstraint(; ineq = ineq, eq = eq)
    else
        nothing
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the linear constraint equations to parse, and the universe key their names resolve against.

[`linear_constraints`](@ref) parses `val` and assembles the coefficient matrices of a [`LinearConstraint`](@ref) from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearConstraintEstimator(;
        val::EqnType,
        key::Option{<:AbstractString} = nothing
    ) -> LinearConstraintEstimator

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(val)`.

# Examples

```jldoctest
julia> lce = LinearConstraintEstimator(; val = [\"w_A + w_B == 1\", \"w_A >= 0.1\"]);

julia> sets = UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"w_A\", \"w_B\"]));

julia> linear_constraints(lce, sets)
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [-0.1]
    eq ┼ PartialLinearConstraint
       │   A ┼ 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [1.0]
```

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`parse_equation`](@ref)
  - [`linear_constraints`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1.
"""
@concrete struct LinearConstraintEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:lce_val])
    """
    val
    """
    $(field_dict[:ekey])
    """
    key
    function LinearConstraintEstimator(val::EqnType,
                                       key::Option{<:AbstractString} = nothing)::LinearConstraintEstimator
        if isa(val, Str_Vec)
            @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
        end
        if !isnothing(key)
            @argcheck(!isempty(key), IsEmptyError("key cannot be empty"))
        end
        return new{typeof(val), typeof(key)}(val, key)
    end
end
function LinearConstraintEstimator(; val::EqnType,
                                   key::Option{<:AbstractString} = nothing)::LinearConstraintEstimator
    return LinearConstraintEstimator(val, key)
end
"""
    const LcE_Lc = Union{<:LinearConstraintEstimator, <:LinearConstraint}

An unparsed [`LinearConstraintEstimator`](@ref), or an assembled [`LinearConstraint`](@ref). The group exists because a constraint slot accepts both: [`linear_constraints`](@ref) parses the first and passes the second through untouched, so a caller may hand over equations or a block it built earlier.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`LinearConstraint`](@ref)
  - [`linear_constraints`](@ref)
"""
const LcE_Lc = Union{<:LinearConstraintEstimator, <:LinearConstraint}
"""
    const VecLcE_Lc = AbstractVector{<:LcE_Lc}

Every abstract vector whose elements are [`LcE_Lc`](@ref)s. The group exists so that one slot may hold a mixed list of equations still to parse and constraints already assembled.

# Related

  - [`LcE_Lc`](@ref)
  - [`LcE_Lc_VecLcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const VecLcE_Lc = AbstractVector{<:LcE_Lc}
"""
    const VecLcE = AbstractVector{<:LinearConstraintEstimator}

Every abstract vector whose elements are [`LinearConstraintEstimator`](@ref)s. The group is narrower than [`VecLcE_Lc`](@ref) on purpose: every element still has to be parsed, so [`linear_constraints`](@ref) is broadcast over it and answers one constraint per element.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`LcE_VecLcE`](@ref)
  - [`VecLcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const VecLcE = AbstractVector{<:LinearConstraintEstimator}
"""
    const LcE_Lc_VecLcE_Lc = Union{<:LcE_Lc, <:VecLcE_Lc}

One [`LcE_Lc`](@ref), or a vector of them. The group is the widest linear-constraint slot the library declares: it names every shape a user may write into such a field, so it is what the type bound of that field is written against.

# Related

  - [`LcE_Lc`](@ref)
  - [`VecLcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const LcE_Lc_VecLcE_Lc = Union{<:LcE_Lc, <:VecLcE_Lc}
"""
    const LcE_VecLcE = Union{<:LinearConstraintEstimator, <:VecLcE}

One [`LinearConstraintEstimator`](@ref), or a vector of them. The group excludes an assembled [`LinearConstraint`](@ref), so a method that dispatches on it knows that every element still carries equations to parse.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`VecLcE`](@ref)
  - [`LcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const LcE_VecLcE = Union{<:LinearConstraintEstimator, <:VecLcE}
"""
    linear_constraints(lcs::Option{<:LinearConstraint}, args...; kwargs...)
    linear_constraints(lcs::AbstractVector{<:LinearConstraint}, ::Nothing, args...; kwargs...)

No-op fallback for returning an existing `LinearConstraint` object, `nothing`, or a vector of them.

This method is used to pass through an already constructed [`LinearConstraint`](@ref) object or `nothing` without modification. It enables composability and uniform interface handling in constraint generation workflows, allowing functions to accept either raw equations or pre-built constraint objects.

The vector arity is narrowed to a `nothing` universe on purpose. A vector needs no [`UniverseSets`](@ref) precisely because every element is already assembled, and that is the shape a [`Pipeline`](@ref) hands an optimiser when more than one constraint step ran; with a real `UniverseSets` the broader vector methods take over and map this one over the elements.

# Algorithm

 1. Return `lcs`. Neither method reads its further positional arguments or its keywords.

# Arguments

  - `lcs`: An existing [`LinearConstraint`](@ref) object, `nothing`, or a vector of constraints.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `lcs`: The input, unchanged.

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::Option{<:LinearConstraint}, args...;
                            kwargs...)::Option{<:LinearConstraint}
    return lcs
end
"""
    port_opt_view(lc::LinearConstraint, i, args...) -> LinearConstraint

Return a precomputed [`LinearConstraint`](@ref) unchanged under an asset sub-selection.

The identity is deliberate, and it is **not** the claim that a full-universe row means the same thing over a subset — it does not. It is what the `lcse` slot already did: the slot was passed unviewed until a constraint space gained a basis a view has to follow, and slicing `A` here would change the behaviour of a path this method exists only to leave alone. A [`NestedClustered`](@ref) inner solve refuses a bare precomputed constraint outright for exactly this reason; [`Stacking`](@ref) and [`SubsetResampling`](@ref) carry no such guard, and that gap pre-dates the view.

A constraint reaching a meta-optimiser through an [`ExposureConstraintEstimator`](@ref) is a different case and is handled: its `A` is factor-width and is re-projected against the viewed prior's loadings, so the view it needs is of the *basis*, not of the row.

# Algorithm

 1. Return `lc`. The method reads neither the index nor the tail that follows it.

# Arguments

  - `lc`: The precomputed [`LinearConstraint`](@ref).
  - `::Any`: The asset index selection (ignored).
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `lc::LinearConstraint`: The input, unchanged.

# Related

  - [`port_opt_view`](@ref)
  - [`LinearConstraint`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
function port_opt_view(lc::LinearConstraint, ::Any, args...)::LinearConstraint
    return lc
end
function linear_constraints(lcs::AbstractVector{<:LinearConstraint}, ::Nothing, args...;
                            kwargs...)::AbstractVector{<:LinearConstraint}
    return lcs
end
"""
    linear_constraints(eqn::EqnType,
                       sets::UniverseSets; ops1::Tuple = ("==", "<=", ">="),
                       key::Option{<:AbstractString} = nothing;
                       ops2::Tuple = (:call, :(==), :(<=), :(>=)), datatype::DataType = Float64,
                       strict::Bool = false, bl_flag::Bool = false)

Parse and convert one or more linear constraint equations into a [`LinearConstraint`](@ref) object.

This function parses one or more constraint equations (as strings, expressions, or vectors thereof), replaces group or asset references using the provided [`UniverseSets`](@ref), and constructs the corresponding constraint matrices. The result is a [`LinearConstraint`](@ref) object containing both equality and inequality constraints, suitable for use in portfolio optimisation routines.

# Algorithm

This method is the whole pipeline, and each step names the stage that owns it.

 1. Parse `eqn` with [`parse_equation`](@ref), giving `lcs`, one [`ParsingResult`](@ref) per equation. Each result carries the equation in canonical form.
 2. Expand every group name of `lcs` into its members with [`replace_group_by_assets`](@ref), giving results written in names of the universe. `bl_flag` selects which of the two expansions runs.
 3. Assemble the coefficient matrices and the right-hand sides from `lcs` with [`get_linear_constraints`](@ref), which resolves each name against the universe `key` names and separates the equality rows from the inequality rows.
 4. Return what [`get_linear_constraints`](@ref) gives: a [`LinearConstraint`](@ref), or `nothing` when no row survived.

# Arguments

  - `eqn`: A single constraint equation (as `AbstractString` or `Expr`), or a vector of such equations.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `ops1`: Tuple of valid comparison operators as strings.
  - `ops2`: Tuple of valid comparison operators as expression heads.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
  - `key`: Key naming the universe the variables resolve against. Defaults to `sets.xkey`.
  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space constraint.

# Validation

  - Every stage validates its own input: [`parse_equation`](@ref) the equation text, [`replace_group_by_assets`](@ref) the flag grammar, and [`get_linear_constraints`](@ref) the names against the universe.

# Returns

  - `lcs::LinearConstraint`: An object containing the assembled equality and inequality constraints, or `nothing` if no constraints are present.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"w_A\", \"w_B\", \"w_C\"]));

julia> linear_constraints([\"w_A + w_B == 1\", \"w_A >= 0.1\"], sets)
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [-0.1]
    eq ┼ PartialLinearConstraint
       │   A ┼ 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [1.0]
```

# Related

  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`UniverseSets`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(eqn::EqnType, sets::UniverseSets,
                            key::Option{<:AbstractString} = nothing;
                            ops1::Tuple = ("==", "<=", ">="),
                            ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractRegressionResult} = nothing)::Option{<:LinearConstraint}
    lcs = parse_equation(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, bl_flag)
    return get_linear_constraints(lcs, sets, key; datatype = datatype, strict = strict,
                                  rr = rr)
end
function linear_constraints(lcs::LinearConstraintEstimator{<:AbstractEstimatorValueAlgorithm},
                            sets::UniverseSets, key::Option{<:AbstractString} = nothing;
                            datatype::DataType = Float64, strict::Bool = false,
                            args...)::Option{<:LinearConstraint}
    return estimator_to_val(lcs.val, sets,
                            !hasproperty(lcs.val, :default) ? nothing : lcs.val.default,
                            key; datatype = datatype, strict = strict)
end
"""
    linear_constraints(lcs::LinearConstraintEstimator, sets::UniverseSets;
                       datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false,
                       rr::Option{<:AbstractRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)
    linear_constraints(lcs::VecLcE, sets::UniverseSets;
                       datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false,
                       rr::Option{<:AbstractRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)

Parse the equations a [`LinearConstraintEstimator`](@ref) carries, against the universe key that estimator names.

The method reads `val` and `key` off the estimator and hands both to the equation method, which gives one uniform interface for a single constraint estimator and for a vector of them. A vector is answered element by element, and the result is a vector of the same length.

`rr` is accepted so that a caller holding loadings — [`processed_jump_optimiser_attributes`](@ref) does — can pass them uniformly to whatever sits in `lcse`, without inspecting its type first. A bare [`LinearConstraintEstimator`](@ref) **drops** them: the asset frame is the absence of a re-basis, and an estimator that quietly re-based itself because loadings happened to be available would make the space depend on the prior rather than on what the user wrote. A re-basis is asked for by wrapping in an [`ExposureConstraintEstimator`](@ref) and by nothing else. `rd` rides along for the same reason and is dropped for a stronger one: only a space can ask for a refit, and a bare estimator has no space.

# Algorithm

 1. Read `val` and `key` off `lcs`.
 2. Drop `rr` and `rd`, for the reason the paragraph above gives.
 3. Return the [`LinearConstraint`](@ref) that the equation method builds from `val`, `sets` and `key`.
 4. Apply steps 1 to 3 to each element, and return the vector of results, when `lcs` is a vector. `rr` and `rd` reach every element, and every element drops them.

# Arguments

  - `lcs`: The [`LinearConstraintEstimator`](@ref) to parse, or a vector of them.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
  - `rr`: Accepted and dropped. A bare estimator never re-bases.
  - `rd`: Accepted and dropped. A bare estimator never asks for a refit.

# Returns

  - `lcs::Option{<:LinearConstraint}`: The assembled constraint, or `nothing` when no row survived. A vector input gives one such value per element.

# Related

  - [`linear_constraints`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
"""
function linear_constraints(lcs::LinearConstraintEstimator, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)::Option{<:LinearConstraint}
    return linear_constraints(lcs.val, sets, lcs.key; datatype = datatype, strict = strict,
                              bl_flag = bl_flag)
end
function linear_constraints(lcs::VecLcE, sets::UniverseSets; datatype::DataType = Float64,
                            strict::Bool = false, bl_flag::Bool = false,
                            rr::Option{<:AbstractRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)
    return [linear_constraints(lc, sets; datatype = datatype, strict = strict,
                               bl_flag = bl_flag, rr = rr, rd = rd) for lc in lcs]
end

export UniverseSets, PartialLinearConstraint, LinearConstraint, LinearConstraintEstimator,
       ParsingResult, parse_equation, replace_group_by_assets, estimator_to_val,
       linear_constraints, UniformValues
