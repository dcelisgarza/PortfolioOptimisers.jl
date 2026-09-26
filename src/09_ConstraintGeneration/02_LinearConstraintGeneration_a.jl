"""
$(DocStringExtensions.TYPEDEF)

Holds the coefficient matrix `A` and the right-hand side vector `B` of one half of a linear constraint block.

The field of [`LinearConstraint`](@ref) that holds the half, `ineq` or `eq`, makes it an inequality or an equality, and [`LinearConstraint`](@ref) states the form of each half. One row of `A` and the entry of `B` beside it are one constraint, so the constructor refuses a pair whose count of rows and count of entries differ.

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

  - The method reads the index and ignores it. It returns both halves unchanged, and it never slices `A` along the asset axis.
  - Each row runs over the whole universe that it was assembled against, so a slice of `A` changes what the row states. [`port_opt_view`](@ref) states why this slot needs the identity.

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

A `LinearConstraint` is a block of rows, and two blocks applied one after the other are the same as the one block that stacks them. The inequality halves concatenate, the equality halves concatenate, and an absent half adds no row. Generation does the same when it takes several estimators at once. For example, [`centrality_constraints`](@ref) over a vector of [`CentralityConstraint`](@ref)s appends every row to one result, and does not return one result per estimator.

A caller that computes its constraints one at a time, such as a [`Pipeline`](@ref) that runs one step per estimator, can merge them here. The optimiser then gets the value that the vector form gives.

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

Every type that holds a parsed equation subtypes `AbstractParsingResult`. Each member holds one parsed equation in canonical form, which is the variable names, their coefficients, the comparison operator and the right-hand side. The stages after [`parse_equation`](@ref) therefore read one shape, whatever form the equation had.

# Related

  - [`ParsingResult`](@ref)
  - [`RhoParsingResult`](@ref)
  - [`parse_equation`](@ref)
"""
abstract type AbstractParsingResult <: AbstractConstraintResult end
"""
$(DocStringExtensions.TYPEDEF)

Holds one linear constraint equation in canonical form.

[`parse_equation`](@ref) returns it, and it holds what [`get_linear_constraints`](@ref) needs to assemble a row. That is the variable names, their coefficients, the comparison operator, the right-hand side and the equation as a formatted string.

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

Every abstract vector whose elements are [`ParsingResult`](@ref)s. The group exists because [`parse_equation`](@ref) returns a vector of results for a vector of equations, and every later stage maps over that vector.

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

The group that needs a missing partition key reports it, so the whole key set is the wrong pool. In `Dict("ux_sector" => …)` the nearest neighbour of the missing `nx_sector` is `ux_sector`, the key under validation, and a suggestion from the whole set tells the caller to rename the one key that is correct. A pool of the keys that carry the prefix of the missing key holds only keys that the caller can have meant.

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

It is the counterpart of [`prefixed_sets_keys`](@ref) for `xkey`, the one key with no prefix of its own. Any key can hold the asset names, so no prefix finds the asset universe. The function removes every key that another declared axis claims. Without that step, a dict that holds only a factor axis answers a mistyped `xkey` with the factor key, which names a different axis.

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

[`UniverseSets`](@ref) carries two factor axes, and one rule holds on both, so this function states the rule once and the constructor calls it once per axis. `axis` names the axis in both messages. A caller who declared the time-series axis and wrote a cross-sectional partition learns which of the two is missing, which the key value alone does not show.

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

It is the sibling of [`assert_factor_partition`](@ref), for the same reason. [`UniverseSets`](@ref) carries two factor axes, and one rule holds on both, so this function states the rule once and the constructor calls it once per axis.

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

It declares every axis that it carries. `xkey` and `uxkey` are for assets, `tfkey` and `utfkey` for time-series factors, `cfkey` and `ucfkey` for cross-sectional factors, and `nikey` for the Non-Investable Axis. The asset axis is the primary axis. The constructor requires `haskey(dict, xkey)`, and a view slices this axis. The factor axes are optional, because a problem with no factor model has no factor names. A consumer that needs a factor axis and does not find it throws when it reads the axis, not at construction.

The two factor families name different things, so there are two factor axes. A time-series regression fits one loading vector per asset over the observations, so its factors are the columns of `rd.F`, and a caller copies `rd.nf` into the dict under `tfkey`. A cross-sectional regression fits one loading vector per observation across the assets, so its factors are the exposures of the fit, and they exist only inside the fitted block. One axis for both gives one key two different lists on one problem. A consumer never chooses between the axes by hand. [`factor_axis_key`](@ref) reads the key from the loadings result that the consumer holds.

A key of `dict` that starts with `xkey` is a partition of the asset universe, so its group must have the length of `dict[xkey]`. [`asset_sets_matrix`](@ref) with [`NestedClustered`](@ref) reads such partitions.

A key that starts with `uxkey` is a unique-entry group. Its `xkey`-prefixed partition must exist in `dict` and have the length of the asset universe, and that partition maps each asset to one entry of the unique-entry group. Constraint generation can then write constraints over the unique entries, also inside a [`NestedClustered`](@ref) optimisation.

The `tfkey` and `utfkey` prefixes mean the same on the time-series factor axis, and `cfkey` and `ucfkey` mean the same on the cross-sectional axis. On the asset axis the prefixes let a view slice the partitions. An asset index never slices a factor axis, so on a factor axis the prefixes give only the length checks at construction. The constructor checks the two factor axes alike and never checks one against the other, so a problem can declare one, both or neither.

A taxonomy reaches the Asset Panel through [`panel_input`](@ref), which reads one `xkey`-prefixed key as one Panel Field. The panel names its own columns, so no key declares a feature axis.

`nikey` declares the Non-Investable Axis, the names that the Investable Mask left out. A forced liquidation is priced on this axis. It differs from the other six keys in three ways.

  - A door writes it after it reduces an optimiser to the Investable Mask, and the door overwrites an entry that the caller wrote. Outside a door, an entry that the caller writes is the way to resolve a liquidation rate.
  - Every view drops it. A sets that carries it was reduced by one door for one problem, so a cluster of a nested optimisation cannot inherit the departures of its parent and charge them again.
  - It has no prefixed partition and no unique-entry group, because its entries are unique by construction and a plain group already reaches it.

A key that matches none of the seven prefixes is a plain group. It expands by name on any axis, so a factor group and a group on the Non-Investable Axis need no code of their own.

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
  - No one of `xkey`, `uxkey`, `tfkey`, `utfkey`, `cfkey`, `ucfkey` and `nikey` starts with another. The constructor makes the 42 ordered checks, and they also refuse two equal keys.
  - If a key in `dict` starts with the same value as `xkey`, `length(dict[k]) == length(dict[xkey])`.
  - If a key in `dict` starts with the same value as `uxkey`, there must be a corresponding key in `dict` where the `uxkey` prefix is replaced by the `xkey` prefix, and its length must equal `length(dict[xkey])`.
  - If a key in `dict` starts with the same value as `tfkey`, `haskey(dict, tfkey)` and `length(dict[k]) == length(dict[tfkey])`.
  - If a key in `dict` starts with the same value as `utfkey`, there must be a corresponding key in `dict` where the `utfkey` prefix is replaced by the `tfkey` prefix, and its length must equal `length(dict[tfkey])`.
  - If a key in `dict` starts with the same value as `cfkey`, `haskey(dict, cfkey)` and `length(dict[k]) == length(dict[cfkey])`.
  - If a key in `dict` starts with the same value as `ucfkey`, there must be a corresponding key in `dict` where the `ucfkey` prefix is replaced by the `cfkey` prefix, and its length must equal `length(dict[cfkey])`.
  - If `dict` carries `nikey`, its entries are unique, and none of them is also in `dict[xkey]`. An asset is investable or it is not, and a name on both axes is priced twice, once as a holding and once as a forced exit.

## View parameters

`UniverseSets` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the asset index alone. It ignores every further positional argument, because it slices no axis but the asset axis.
  - The view slices every `xkey`-prefixed entry of `dict` to the selected assets, and rebuilds every `uxkey`-prefixed entry from the sliced partition that it names.
  - The view keeps the `tfkey`-, `utfkey`-, `cfkey`- and `ucfkey`-prefixed entries and every plain group unchanged. [`port_opt_view`](@ref) states why each axis is exempt.
  - The view drops the `nikey` entry, because only a door writes one. It matches the key exactly and not by prefix, so it keeps a plain group whose name starts with `nikey`.
  - The view keeps the seven key prefixes, so the viewed value declares the same axes as the original.

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
            # `isdisjoint` over two untyped values leads JET into `Transducers` (issue #1341).
            both = intersect(ni, dict[xkey])
            @argcheck(isempty(both),
                      ArgumentError("$(length(both)) name(s) are on both the asset universe `$xkey` and the non-investable axis `$nikey`. An asset is investable or it is not, and a name on both would be priced twice, once as a holding and once as a forced exit: remove it from whichever axis it does not belong to."))
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

The view slices the asset axis and no other. An asset index has no meaning on a factor axis, so the view returns a `tfkey`-prefixed entry and a `cfkey`-prefixed entry bit-identical. The view drops the Non-Investable Axis, as step 4 states. The sets declares the axis of each entry by its key, so a sets in a `@vprop` field is safe to view. The method takes no factor index. `port_opt_view(rd, i, j, k)` can slice `rd.nf`, but no internal caller passes a `k` that is not a colon, so a user who slices factors must update the sets.

# Algorithm

 1. Read `xkey` and `uxkey` from `sets`, and open an empty dictionary `dict` of the type `sets.dict` has.
 2. For an entry of `sets.dict` whose key starts with `xkey`, take `view(v, i)`, the group restricted to the selected assets.
 3. For an entry whose key starts with `uxkey`, take the unique entries of the `xkey`-prefixed partition it names, restricted to `i`. The unique-entry group is therefore derived from the sliced partition and never from the original one.
 4. Skip the entry whose key equals `nikey`. Only a door writes the Non-Investable Axis, so a view never carries one. Otherwise a cluster of a nested optimisation inherits the departures of its parent and charges each of them again, once per cluster. The match is exact and not by prefix, so the view keeps a plain group such as `"nikkei225"`, whose name starts with the default `"ni"`.
 5. Copy every other entry unchanged into the same `dict`. The `tfkey`-, `utfkey`-, `cfkey`- and `ucfkey`-prefixed entries and every plain group come back bit-identical.
 6. Return the [`UniverseSets`](@ref) built from `dict` and the seven unchanged key prefixes, which revalidates the prefix grammar over the viewed universe.

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

Write the Non-Investable Axis on `sets`, which declares `ni`, the names that the Investable Mask left out, under `sets.nikey`.

A door calls this function after it reduces an optimiser to the Investable Mask, so a [`UniverseSets`](@ref) that carries the axis was reduced by one door for one problem. [`port_opt_view`](@ref) drops the axis, so no sub-problem inherits it.

A caller can also declare the axis by hand. Outside a door, that is the only way to resolve a forced-liquidation rate, for example in a direct call of [`fees_constraints`](@ref) with no optimisation around it. Inside a door the mask comes from the data, so this function overwrites an entry that the caller wrote and does not merge the two.

An empty `ni` returns `sets` unchanged. No asset left the universe, and an empty axis makes [`fees_constraints`](@ref) resolve a carrier that prices no position.

# Algorithm

 1. Return `sets` unchanged when `ni` is empty.
 2. Otherwise copy `sets.dict`, write `ni` under `sets.nikey`, and rebuild the [`UniverseSets`](@ref) from it and the seven unchanged key prefixes, which revalidates uniqueness and disjointness over the minted axis.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) to write the axis on, or `nothing`.
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

The order matters. A forced-liquidation carrier is sliced by index to the complement of the mask, and the rate that prices it resolves against these names by position, so both must walk the complement in one order. This function indexes `nx` with `.!imsk`, which is ascending, and [`port_opt_view`](@ref)`(::Fees, i, X)` takes the complement of `i` over the width of `X`, which is also ascending.

Returns data with no names gives an empty vector and does not throw. The axis is a list of names, so a problem with no names has no Non-Investable Axis and no name-keyed constraint to resolve against it.

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

Record one thing that a departure cost, for the door to report.

Each site drops a departed name where it finds it, for example a view row or a group member. The site is far from the door, and only the door knows that a departure happened. A report at the site repeats once per row and once per window of a walk-forward. No report at all leaves a caller who wrote three views and got one fitted with no way to learn it. So the site writes what it dropped into a ledger, and the door reads the ledger once and reports the departures and the drops together, through [`announce_non_investable`](@ref).

A `nothing` ledger collects nothing, and it is the default everywhere. A caller who assembles constraints outside a door has no door to report to, and pays nothing for a ledger that it does not keep. Dispatch on the type of the ledger selects the branch.

`what` is a noun phrase that names the dropped item, not a sentence. The door joins the phrases into one message and gives the verb.

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

A group that loses some of its members still describes the rest, so its row stays, with the coefficient spread over the survivors. A group that loses all of its members describes nothing, and its row goes. The two cases tell the caller different things, so the function writes a different phrase for each, and no other function writes either phrase. [`shed_departed_members`](@ref) is the only caller, and [`replace_group_by_assets`](@ref) reaches it from each of its four expansion branches.

The phrases give counts and not names. The door names the departed assets once, and a list per group makes the message longer than the facts that it reports.

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

Only the door knows that a departure happened. After the door, a departed asset is absent. A bound stated for it resolves on the Non-Investable Axis and is skipped, and a view row that names it is dropped whole. A report at each of those sites repeats once per row and once per window of a walk-forward, so each site writes its drop into a ledger with [`record_non_investable_drop!`](@ref), and the door reports everything once, here.

`process` names the work that excludes the departed assets, because more than one kind of door writes the axis. An optimisation reduces at its entry, and a wrapping prior reduces at its own entry before it builds a view. `consequence` states what a departure means to this door. Only an optimisation prices a forced-liquidation carrier, so the defaults of the two arguments give the message of the optimisation door.

The message is an `@info` by default, not a warning and not a [`strict_diagnostic`](@ref). Nothing is wrong. The data changed, and the work runs correctly over the assets that are left. A raise under `strict` refuses the departure, and the reduction to the Investable Mask exists to remove that refusal. `warn = true` gives a `@warn` for the one case that is not routine, a departure that removed the last of something the caller asked for, such as the last view of a view set. The result is then the unconditioned answer, and the caller has no other way to learn it.

An empty `ni` logs nothing. This is the path when every asset is investable and when the data has no names. An empty `drops` gives a message that names the departed assets and no drops, which is what a door logs before it resolves anything over them.

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

Read the declared factor universe `sets.dict[key]`, and check that it exists and names one factor per column of `source`.

`source` is the `observations × factors` matrix, and `K` is its number of columns. A factor axis is optional on [`UniverseSets`](@ref), but a consumer that reads one needs it, so this function reports the failure where the consumer reads the axis. Both messages name `key` and the matrix, because the caller must make these two agree. A caller who put the factor names under `xkey` learns that the factor key is missing, and not that an asset universe is wrong.

The caller gives `key`, because [`UniverseSets`](@ref) declares two factor axes and this function cannot tell which one the caller means. A caller that holds a loadings result reads the key from it with [`factor_axis_key`](@ref). A caller that reads the columns of the returns data `F` gives `sets.tfkey`, the axis of those columns.

Every consumer of either axis calls this one function, and none of them repeats the checks.

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

[`UniverseSets`](@ref) declares two factor axes, so a consumer that resolves factor names must say which one it means. The consumer does not choose by hand. The key follows the result that carries `M`. A [`Regression`](@ref) is fitted per asset over the observations, so its columns are the columns of `rd.F`, and it gives `sets.tfkey`. A [`CrossSectionalFactorModel`](@ref) is fitted per observation across the assets, so its columns are the exposures of the fit, and it gives `sets.cfkey`. A caller therefore cannot name the wrong axis, and no consumer needs a field for the key.

The third method is for a consumer that holds an unfitted specification and not a result. Only the time-series family has a specification here, because [`RegE_Reg`](@ref) takes an [`AbstractTimeSeriesRegressionEstimator`](@ref) and no other estimator, and every result that family fits is a [`Regression`](@ref). The three methods therefore cover [`RegE_Reg`](@ref) exactly.

No method takes [`AbstractLoadingsRegressionResult`](@ref) as a fallback. A new member of that root inherits the axis of such a fallback, and that axis can be the wrong list of names with the right length. A constraint written against it then solves and constrains the wrong factors. A missing method is a `MethodError` that names the type.

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

[`UniverseSets`](@ref) declares two axes over assets. The investable universe is under `xkey`, and the Non-Investable Axis is under `nikey`, which a door writes from the complement of the Investable Mask. A name-keyed estimator resolves against one of them. A name that it does not find there can still be a correct name on the other axis, for example a bound for an asset that left, or a forced-liquidation rate for an asset that stayed. [`name_to_val!`](@ref) reads this list to tell such a name from a typo.

The relation is symmetric and covers only these two axes. A factor axis names factors, so it has no counterpart and the answer is empty. A `key` that names any other list also gives an empty answer.

An axis that `sets` does not declare gives an empty answer. This is the common case, because a problem in which every asset is investable has no `nikey` entry.

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

Remove from the member list of a group the names that are on the counterpart axis, and record the removal in the ledger of the door.

The data resolves a group, and the caller does not choose its members. `"tech"` means the technology assets of this problem, and when one of them delists, the group still names the rest. [`replace_group_by_assets`](@ref) therefore sheds the departed members before it spreads the coefficient of the group. A Black-Litterman mean then divides by the surviving count, an entropy pooling sum runs over the survivors, and the row still computes what its right-hand side states. A removal after the spread leaves `k - 1` terms of `c/k` against an unchanged target. A name that the caller writes out is different, because a departed name drops its whole row.

A group that loses every member keeps its first member and does not return empty. The row then names a departed asset, so [`get_linear_constraints`](@ref) drops the whole row in silence, by the counterpart rule. An empty return leaves a row with no variable, the same row as a caller who writes `1 == 0.004`, and that row gets a diagnostic. A group that holds no member at all stays empty.

The second method is the pair form, for a correlation view over two groups. It walks the two lists together and keeps a position only when both of its names survived. A pair is one correlation, so a pair that lost one side has nothing to measure. The joint shed also keeps the two lists the same length, which [`replace_group_by_assets`](@ref) checked before. When no pair survives, the method keeps the first pair, for the same reason as the first method.

An empty `other` means that every asset is investable. Both methods then return their arguments unchanged and record nothing, so a problem with no departure costs one comparison and no allocation.

The record is written here and not at the four call sites, so each branch of [`replace_group_by_assets`](@ref) needs one line for the shed, the record and the fallback. Four copies of those steps can drift apart.

# Arguments

  - `members` / `members1`, `members2`: The group's member names, as `sets.dict` holds them.
  - `other`: The counterpart axis, read with [`counterpart_axis_names`](@ref). Usually the Non-Investable Axis.
  - `ledger`: The door's ledger, or `nothing` when nobody is collecting.
  - `group`: The group name as the caller wrote it, for the ledger.
  - `eqn`: The row the group appears in, for the ledger.

# Returns

  - `members::AbstractVector`: The members that are not on `other`, in their original order, or the first departed member when none survived. An empty `members` stays empty.
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
    return isempty(kept) ? first(members, 1) : kept
end
function shed_departed_members(members1::AbstractVector, members2::AbstractVector,
                               other::VecStr, ledger::Option{<:AbstractVector},
                               group::AbstractString, eqn::AbstractString)
    if isempty(other)
        return members1, members2
    end
    keep = map((m1, m2) -> m1 ∉ other && m2 ∉ other, members1, members2)
    record_group_shed!(ledger, group, count(!, keep), count(keep), eqn)
    return if any(keep)
        members1[keep], members2[keep]
    else
        first(members1, 1), first(members2, 1)
    end
end
"""
    name_to_val!(nx::VecStr, sdict::AbstractDict, key::Any, val::Number,
                 arr::VecNum, strict::Bool, nxkey::AbstractString,
                 other::VecStr = String[])

Set values in a vector for the asset or the group of assets that `key` names.

`name_to_val!` resolves `key` through [`resolve_axis_name`](@ref), maps the members to indices in the asset universe `nx`, and sets those entries of `arr` to `val`. An asset name resolves to itself, and a group name expands to its members. If `key` names neither, the function throws or warns, as `strict` selects. Each diagnostic message names the size of the universe and never the universe itself or the value dictionary, because a shared message builder writes each message.

`other` is the counterpart axis, the asset names that the same [`UniverseSets`](@ref) declares on the axis that this call does not resolve against. The function skips a name that it finds there in silence, whatever `strict` is, and this is the only exception to `strict`. `strict` catches a typo of the caller, and a name on the counterpart axis is not a typo. It was a correct name over the universe that the caller had, and the data moved it. A caller cannot know in advance which asset a prior will fail to estimate, so a refusal, or a warning once per constraint and once per window of a walk-forward, reports a thing that no one can act on. The door that derived the mask reports the departure once.

The two asset axes are counterparts of each other, and the relation is symmetric. When the call resolves on the asset universe, `other` is the Non-Investable Axis, so the function drops a bound for an asset that left. When the call resolves on the Non-Investable Axis, which is how a forced-liquidation rate is priced, `other` is the asset universe, so the same rule drops a liquidation rate for an asset that stayed. A factor axis has no counterpart, and `other` is then empty.

# Algorithm

 1. Resolve `key` through [`resolve_axis_name`](@ref), giving `members`. An asset name resolves to itself, and a group name expands to a copy of its member list. An asset name takes precedence over a group name of the same spelling.
 2. Return in silence when `members` is `nothing` and `key` names an entry of `other`, because the name is on the counterpart axis. It is a correct name, and this axis has no entry to write it into.
 3. Report through [`strict_diagnostic`](@ref) and return when `members` is `nothing`, because `key` names neither an asset nor a group. The suggestion pool is widened from `nx` to `nx` together with the keys of `sdict`, because a missing name may be a mistyped asset or a mistyped group.
 4. Map `members` to positions in `nx` with [`axis_name_indices`](@ref), giving `idx`, and drop the members that miss the universe. Remove from the report the members on `other`, by the rule of step 2, and report the rest once through [`strict_diagnostic`](@ref). A group whose departed members are all on `other` is therefore silent, and a group that holds a real typo names it.
 5. Set the entries of `arr` at `idx` to `val`.

# Arguments

  - `nx`: Vector of asset names.
  - `sdict`: Dictionary mapping group names to vectors of asset names. It is never modified, because [`resolve_axis_name`](@ref) returns a copy of the member list.
  - `key`: Name of the asset or the group of assets to set values for.
  - `val`: The value to assign.
  - `arr`: The array that the function changes in place.
  - `strict`: If `true`, a `key` that resolves to nothing throws. If `false`, it warns.
  - `nxkey`: The key of the asset universe in `sets.dict`, for example `"nx"`. The messages of [`unknown_variable_msg`](@ref) and [`missing_group_assets_msg`](@ref) read it to name the universe.
  - `other`: The counterpart asset axis. The function skips its names in silence and does not report them.

# Validation

  - `key` names an asset of `nx`, a group of `sdict`, or an entry of `other`. An `ArgumentError` is thrown when `strict` is `true`, and a warning is issued otherwise.
  - Every member of a resolved group names an entry of `nx` or of `other`. A member that misses both is dropped, and the drop raises when `strict` is `true` and issues a warning otherwise.

# Returns

  - `nothing`. The function changes `arr` in place.

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

Return one value per asset of the universe, from a mapping of asset names and group names to values.

The function fills a vector with `val` and writes the values of `dict` into it, through the universe and the groups of `sets`. A key of `dict` that `sets` does not hold throws or warns, as `strict` selects.

!!! warning

    A later key that writes the same asset overwrites the earlier value. A `Dict` has no fixed order, so use an [`OrderedDict`](https://juliacollections.github.io/OrderedCollections.jl/stable/#OrderedDicts) or a vector of pairs to fix which value is kept.

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
  - `key`: The key of the universe in `sets.dict`, or `nothing`. When it is not `nothing`, it replaces `sets.xkey`.
  - `datatype`: Element type of the value the array is filled with when `val` is `nothing`.
  - `strict`: If `true`, a key of `dict` that `sets` does not hold throws. If `false`, it warns.

# Validation

  - A key of `dict` that names neither an asset, nor a group, nor an entry of the counterpart axis raises an `ArgumentError` when `strict` is `true`, and warns otherwise.

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

Return a number or `nothing` unchanged.

This method is the fallback for an input that is already a number or `nothing`, which needs no mapping. A vector of numbers goes to the vector method, which checks its length.

# Algorithm

 1. Return `val`. The method reads none of its other arguments and none of its keywords.

# Arguments

  - `val`: `nothing` or a number.
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

Check a value vector against the length of the universe, and return it unchanged.

A caller that already holds one value per asset needs no mapping, so this method checks the length and nothing else.

# Algorithm

 1. Take `key` as the universe key, or `sets.xkey` when `key` is `nothing`, and read the universe from `sets.dict` under it.
 2. Check `val` against the length of that universe.
 3. Return `val`.

# Arguments

  - `val`: One value per asset of the universe.
  - `sets`: The [`UniverseSets`](@ref) that holds the universe.
  - `::Any`: The fill value of the mapping methods, ignored here.
  - `key`: The key of the universe in `sets.dict`, or `nothing`. When it is not `nothing`, it replaces `sets.xkey`.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `length(val) == length(sets.dict[ifelse(isnothing(key), sets.xkey, key)])`.

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

Check the size of a value matrix along `dims` against the length of the universe, and return it unchanged.

A caller that already holds one row or column per asset needs no mapping, so this method checks the size and nothing else.

# Algorithm

 1. Take `key` as the universe key, or `sets.xkey` when `key` is `nothing`, and read the universe from `sets.dict` under it.
 2. Check the size of `val` along `dims` against the length of that universe.
 3. Return `val`.

# Arguments

  - `val`: A matrix with one row or one column per asset, as `dims` selects.
  - `sets`: The [`UniverseSets`](@ref) that holds the universe.
  - `::Any`: The fill value of the mapping methods, ignored here.
  - `key`: The key of the universe in `sets.dict`, or `nothing`. When it is not `nothing`, it replaces `sets.xkey`.
  - `dims`: The dimension whose size must equal the length of the universe.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `size(val, dims) == length(sets.dict[ifelse(isnothing(key), sets.xkey, key)])`.

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

The value is the same in every slot. `lb = UniformValues()` floors every weight at the equal-weight level and `ub = UniformValues()` caps every weight there. Neither slot is a special case in [`estimator_to_val`](@ref).

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
 2. Convert `N` to `datatype` and take its reciprocal, giving `iN`. The division runs in `datatype`, so a `Rational{Int}` gives `1//N` exactly.
 3. Return the range of length `N` whose start and stop are both `iN`.

# Arguments

  - `::UniformValues`: The algorithm that selects this method.
  - `sets`: The [`UniverseSets`](@ref) whose universe gives `N`.
  - `::Any`: Fill value for API consistency (ignored).
  - `key`: The key of the universe in `sets.dict`, or `nothing`. When it is not `nothing`, it replaces `sets.xkey`.
  - `datatype`: Element type of the returned range.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `val::AbstractRange`: A range of length `N`, each entry the reciprocal of `N`. A float `datatype` gives a `StepRangeLen`, and a `Rational` one gives a `LinRange`.

# Related

  - [`UniformValues`](@ref)
  - [`estimator_to_val`](@ref)
  - [`UniverseSets`](@ref)
"""
function estimator_to_val(::UniformValues, sets::UniverseSets, ::Any = nothing,
                          key::Option{<:AbstractString} = nothing;
                          datatype::DataType = Float64, kwargs...)
    N = length(sets.dict[ifelse(isnothing(key), sets.xkey, key)])
    iN = inv(datatype(N))
    return range(; start = iN, stop = iN, length = N)
end
"""
    allowed_functions = Dict{Symbol, Function}(:+ => +, :- => -, :* => *, :/ => /,
                                               :^ => ^, :sqrt => sqrt, :cbrt => cbrt,
                                               :exp => exp, :exp2 => exp2, :exp10 => exp10,
                                               :log => log, :log2 => log2, :log10 => log10,
                                               :abs => abs, :min => min, :max => max)

Maps the name of each of the 16 functions that an equation can call to its function object.

An equation string can come from a configuration file, a spreadsheet or a user interface, so the parser calls only the functions of this table. The table maps each name to its function directly and does not look the name up in `Base`. A name that is not a key fails closed with a `Meta.ParseError`, and the callable functions cannot drift from the allowed names, because the two are one list.

The `prior(...)` marker is not in the table. It names assets or groups and not numbers, so [`eval_numeric_functions`](@ref) keeps it and [`replace_group_by_assets`](@ref) expands it, and no function evaluates it as a number.

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

The function walks an expression tree and evaluates each call whose arguments are all numbers, and it reads the symbol `Inf` as the number. [`_parse_equation`](@ref) calls it on both sides of an equation before it collects the terms.

Before a call runs, the function converts every argument to `datatype`, so the arithmetic runs in the number type of the optimiser and not in machine `Int64`. Integer literals therefore do not wrap. `2^64` gives `1.8446744073709552e19` and not `0`, and `2^-1` gives `0.5` and not a `DomainError`. A numeric literal inside a call that is not evaluated keeps its type, so `2^z` stays `2 ^ z`.

A call head that [`allowed_functions`](@ref) does not hold fails closed with a `Meta.ParseError`. The function keeps the `prior(...)` marker for [`replace_group_by_assets`](@ref), and raises a `Meta.ParseError` when all its arguments are numbers.

# Algorithm

 1. Fold every argument of `expr` first, by applying this function to each, when `expr` is an `Expr`. A node whose head is not `:call` is rebuilt from its folded arguments and returned.
 2. Rebuild a `:call` node whose head is `prior` from its folded arguments, and return it. The marker names assets or groups, so it is never folded to a number.
 3. Look the head of any other `:call` node up in [`allowed_functions`](@ref), giving `f`. A head that the table does not hold raises.
 4. Rebuild the call and return it when any folded argument is not a `Number`, so a nonlinear subexpression keeps its own literals untouched.
 5. Coerce every folded argument to `datatype`, apply `f` to them, and return the value that comes back.
 6. Return the `Float64` value `Inf` when `expr` is the symbol `:Inf`. Return `expr` itself in every other case, so a number stays a number and a variable name stays a symbol.

# Arguments

  - `expr`: The Julia expression to evaluate. Can be a `Number`, `Symbol`, or `Expr`.
  - `datatype`: The number type that the arguments of an evaluated call are converted to.

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

The function walks the expression and returns one `(coefficient, variable)` pair per term. It reads numeric constants, variables and the operations `+`, `-`, `*` and `/`, and [`_parse_equation`](@ref) calls it to put an equation in canonical form.

The walk starts from the coefficient `one(datatype)`, so the coefficients that it builds come from the arithmetic of `datatype`, the number type of the optimiser, as the right-hand side does.

# Algorithm

 1. Open an empty vector `terms`.
 2. Walk `expr` with [`collect_terms!`](@ref), from the starting coefficient `one(datatype)`. The walk appends one pair to `terms` per term that it reaches. A constant gives `(coefficient, nothing)`, and any other term gives `(coefficient, name)`.
 3. Return `terms`.

# Arguments

  - `expr`: The Julia expression to expand.
  - `datatype`: Numeric type of the coefficients the walk builds.

# Returns

  - `terms::Vector{Any}`: A vector of `(coefficient, variable)` pairs, where `variable` is a string for variable terms or `nothing` for constant terms. The vector is untyped, and each coefficient has the type that the arithmetic of the walk gives from `one(datatype)`.

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

The function walks the expression tree of a linear equation and appends one `(coefficient, variable)` pair per term to `terms`. It reads numeric constants, variables and the operations `+`, `-`, `*` and `/`.

# Algorithm

 1. Append `(coeff * expr, nothing)` when `expr` is a `Number`, so a constant carries the coefficient and no variable.
 2. Append `(coeff, string(expr))` when `expr` is not a call. A `Symbol` is a bare variable that carries the coefficient it arrived with, and any other leaf, a string literal among them, is a variable named by its text.
 3. For a multiplication, multiply `coeff` by every numeric factor, giving `c`. The parser writes `2*3*x` as one call with three factors, so the step reads every factor and not only the first two. Recurse into the one factor that is not a number with `c`. When no factor is left, append `(c, nothing)`. When more than one factor is left, the product is opaque, so append the product of those factors whole, as `(c, text)`.
 4. For a division `a / b`, recurse into `a` with `coeff` divided by `b`, when `b` is a number.
 5. For an addition, recurse into every argument with `coeff` unchanged.
 6. For a subtraction, recurse into every argument but the last with `coeff`, and into the last with `-coeff`. A unary minus holds no argument but the last, so this negates its one operand.
 7. Append any other call whole, as `(coeff, string(expr))`. A division by a denominator that is not a number is such a call. This step makes a term such as `sqrt(x)` opaque. It becomes one variable named by its own text, and the row builder resolves that text against the universe like any other name.

# Arguments

  - `expr`: The Julia expression to traverse.
  - `coeff`: The current numeric coefficient to apply.
  - `terms`: The vector that the function appends the pairs to, in place. A pair holds a coefficient and a variable name, or `nothing` in place of the name for a constant term.

# Returns

  - `nothing`. The function changes `terms` in place.

# Related

  - [`_collect_terms`](@ref)
  - [`_parse_equation`](@ref)
"""
function collect_terms!(expr, coeff, terms)::Nothing
    if isa(expr, Number)
        push!(terms, (coeff * oftype(coeff, expr), nothing))
    elseif !(isa(expr, Expr) && expr.head == :call)
        # A symbol, a non-call expression, or any other leaf, such as a string literal, is
        # one variable named by its own text.
        push!(terms, (coeff, string(expr)))
    elseif expr.args[1] == :*
        # `2*3*x` parses as one call with three factors, so every numeric factor joins the
        # coefficient, and the factors that are left make the term.
        c = coeff
        rest = Any[]
        for a in view(expr.args, 2:length(expr.args))
            isa(a, Number) ? (c *= oftype(coeff, a)) : push!(rest, a)
        end
        if isone(length(rest))
            collect_terms!(only(rest), c, terms)
        else
            push!(terms, (c, isempty(rest) ? nothing : string(Expr(:call, :*, rest...))))
        end
    elseif expr.args[1] == :/ && isa(expr.args[3], Number)
        collect_terms!(expr.args[2], coeff / oftype(coeff, expr.args[3]), terms)
    elseif expr.args[1] == :+
        for i in 2:length(expr.args)
            collect_terms!(expr.args[i], coeff, terms)
        end
    elseif expr.args[1] == :-
        for i in 2:(length(expr.args) - 1)
            collect_terms!(expr.args[i], coeff, terms)
        end
        collect_terms!(expr.args[length(expr.args)], -coeff, terms)
    else
        # Any other call, `x/y` and `sqrt(x)` among them, is one opaque variable.
        push!(terms, (coeff, string(expr)))
    end
    return nothing
end
"""
    format_term(coeff, var)

Format a single term in a linear constraint equation as a string.

The function writes a coefficient of one as the bare name, a coefficient of minus one as the name behind a minus sign, and every other coefficient as `c*name`.

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
    rethrow_parse_error(expr, side = :lhs)

Raise when a parsed side of an equation is empty or incomplete.

The parser calls it on both sides of an equation. An empty or incomplete side raises a `Meta.ParseError`, and any other side returns `nothing`. The parser fails closed on an empty side and does not assume zero, because an assumed zero is a constraint that the author never wrote. A caller who means zero writes it.

# Algorithm

The method that Julia selects is the algorithm, and one method handles each shape that a parsed side can take.

 1. When `expr` is `Nothing`, which an empty side gives, raise and name `side` in the message.
 2. When `expr` is an `Expr`, raise when its head is `:incomplete`, and return `nothing` otherwise.
 3. When `expr` is anything else, a number or a symbol among them, return `nothing`.

# Arguments

  - `expr`: The parsed Julia expression to check. Can be an `Expr`, `Nothing`, or any other type.
  - `side`: The side of the equation, `:lhs` or `:rhs`, which the message names.

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
