"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the two forms of the size of a part of a set: a [`CountOf`](@ref) of its members and a [`FractionOf`](@ref) of the set.

A field that takes a part of a set holds one of the two members, such as the assets that a subset draws, the rows of a window or the neighbours of a period. The type of the member selects the reading, and the type of the number inside it does not. Julia promotes the elements of a vector literal to one type, so a search grid can turn the plain count `3` into `3.0`. A `CountOf(3)` keeps its type in the same grid.

A member checks its value and does no rounding. The size of the set is known only to the reader, and each reader rounds a fraction of its own set by the rule of its source. So one fraction can give a different count at two sites. A reader that uses `floor` keeps no more than the fraction, and a reader that uses `ceil` keeps no less.

A field of this kind also takes a plain number, through [`count_or_fraction`](@ref). An `Integer` is a count and any other real number is a fraction. The plain form is a shorthand, and the wrapper is the form to prefer. A plain `1.0` is refused, because a grid can promote the count `1` to it.

# Interfaces

The family is closed. Each reader dispatches on [`CountOf`](@ref) and on [`FractionOf`](@ref), and holds one method for each. A new member needs a method at every reader, so the library adds no member.

# Related

  - [`CountOf`](@ref)
  - [`FractionOf`](@ref)
  - [`count_or_fraction`](@ref)
"""
abstract type AbstractCountOrFraction <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The size of a part of a set, stated as a number of its members.

The reader takes `val` members. It clamps `val` to the size of the set where the set can be smaller, and the docstring of each field states whether it clamps or refuses.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CountOf(;
        val::Integer
    ) -> CountOf

Keywords correspond to the struct's fields. No count suits every set, so `val` has no default.

## Validation

  - `val >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> CountOf(3)
CountOf
  val ┴ Int64: 3
```

# Related

  - [`AbstractCountOrFraction`](@ref)
  - [`FractionOf`](@ref)
  - [`count_or_fraction`](@ref)
"""
struct CountOf{T} <: AbstractCountOrFraction
    """
    The number of members of the set.
    """
    val::T
    # The struct is written by hand, and not with `@concrete`, because `@concrete` appends a
    # generic positional constructor that skips this one. `CountOf(5.0)` would then hold a
    # count that is not an `Integer`. The positional form is the main form of this type.
    function CountOf(val::Integer)
        @argcheck(val >= one(val),
                  DomainError(val,
                              "`CountOf.val` is $val, and a count of the members of a set is at least one. State a positive `Integer`, or state the part as a `FractionOf` of the set."))
        return new{typeof(val)}(val)
    end
end
function CountOf(; val::Integer)::CountOf
    return CountOf(val)
end
"""
$(DocStringExtensions.TYPEDEF)

The size of a part of a set, stated as a fraction of the size of the set.

The reader multiplies `val` by the size of its set, and rounds the product to a count by the rule of its source. The docstring of each field states the rounding.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FractionOf(;
        val::Real
    ) -> FractionOf

Keywords correspond to the struct's fields. No fraction suits every set, so `val` has no default.

## Validation

  - `0 < val <= 1`. A `DomainError` is thrown otherwise. `val = 1` is the whole set.

# Examples

```jldoctest
julia> FractionOf(0.2)
FractionOf
  val ┴ Float64: 0.2
```

# Related

  - [`AbstractCountOrFraction`](@ref)
  - [`CountOf`](@ref)
  - [`count_or_fraction`](@ref)
"""
struct FractionOf{T} <: AbstractCountOrFraction
    """
    The fraction of the set, in `(0, 1]`.
    """
    val::T
    # Written by hand for the reason that `CountOf` states: the generic constructor of
    # `@concrete` would take `FractionOf(0.5 + 0im)`.
    function FractionOf(val::Real)
        @argcheck(zero(val) < val <= one(val),
                  DomainError(val,
                              "`FractionOf.val` is $val, and a fraction of a set is in `(0, 1]`. A value of one is the whole set. State a number of members as a `CountOf`."))
        return new{typeof(val)}(val)
    end
end
function FractionOf(; val::Real)::FractionOf
    return FractionOf(val)
end
"""
    count_or_fraction(x::Option{<:UnresolvedValue})
    count_or_fraction(x::Integer)
    count_or_fraction(x::Real)

Return the [`AbstractCountOrFraction`](@ref) that a field of a part of a set holds, or the value that the field holds in its place.

The constructor of each such field calls this function, so a plain number becomes a [`CountOf`](@ref) or a [`FractionOf`](@ref), and each reader dispatches on the two members alone. A plain number is the shorthand of [`AbstractCountOrFraction`](@ref): the type of the number selects the member.

The function is a repair at a constructor, so it returns unchanged each value that needs no repair: `nothing`, a member of the family, a rule or a function that computes the part, and a [`TimeDependent`](@ref) schedule. Each of these is an [`UnresolvedValue`](@ref) or `nothing`. Each other type is a `MethodError`.

# Algorithm

 1. Return `nothing`, a member of the family, or another [`UnresolvedValue`](@ref) unchanged.
 2. Return `CountOf(x)` for an `Integer`. `Bool` is an `Integer`, so `true` is a count of one.
 3. Refuse `x == 1` for any other real number. Return `FractionOf(x)` otherwise.

# Validation

  - A plain real number that is not an `Integer` and equals one throws a `DomainError`. A search grid promotes the count `1` to `1.0`, and the whole set is a different part. State `CountOf(1)` or `FractionOf(1.0)`.
  - The constructors of [`CountOf`](@ref) and [`FractionOf`](@ref) check the range.

# Returns

  - `x::Option{<:UnresolvedValue}`: A [`CountOf`](@ref) for an `Integer`, a [`FractionOf`](@ref) for each other real number, and `x` unchanged otherwise.

# Examples

```jldoctest
julia> PortfolioOptimisers.count_or_fraction(3)
CountOf
  val ┴ Int64: 3

julia> PortfolioOptimisers.count_or_fraction(0.2)
FractionOf
  val ┴ Float64: 0.2
```

# Related

  - [`AbstractCountOrFraction`](@ref)
  - [`CountOf`](@ref)
  - [`FractionOf`](@ref)
  - [`UnresolvedValue`](@ref)
"""
function count_or_fraction(x::Option{<:UnresolvedValue})
    return x
end
function count_or_fraction(x::Integer)::CountOf
    return CountOf(x)
end
function count_or_fraction(x::Real)::FractionOf
    @argcheck(x != one(x),
              DomainError(x,
                          "a plain `$x` is not read as a part of a set. A search grid promotes the count `1` to `1.0`, and the whole set is a different part, so the library refuses to guess. State `CountOf(1)` for one member or `FractionOf(1.0)` for the whole set."))
    return FractionOf(x)
end
"""
    const Real_CountFraction = Union{Real, AbstractCountOrFraction}

Alias for every value that a field of a part of a set accepts: a plain number, which [`count_or_fraction`](@ref) reads by its type, or a member of [`AbstractCountOrFraction`](@ref).

# Related

  - [`AbstractCountOrFraction`](@ref)
  - [`count_or_fraction`](@ref)
"""
const Real_CountFraction = Union{Real, AbstractCountOrFraction}
"""
    const VecReal_CountFraction = AbstractVector{<:Real_CountFraction}

Alias for a vector whose entries are each a [`Real_CountFraction`](@ref).

# Related

  - [`Real_CountFraction`](@ref)
  - [`Real_CountFraction_VecReal_CountFraction`](@ref)
"""
const VecReal_CountFraction = AbstractVector{<:Real_CountFraction}
"""
    const Real_CountFraction_VecReal_CountFraction = Union{Real_CountFraction,
                                                           VecReal_CountFraction}

Alias for one [`Real_CountFraction`](@ref), or a vector of them. A slot of cardinality caps takes this type, with one cap for each set.

# Related

  - [`Real_CountFraction`](@ref)
  - [`VecReal_CountFraction`](@ref)
  - [`cardinality_cap`](@ref)
"""
const Real_CountFraction_VecReal_CountFraction = Union{Real_CountFraction,
                                                       VecReal_CountFraction}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the largest number of members whose share of a set of size `n` is at most `part.val`.

A plain `floor(part.val * n)` can be one too small, because the product can round below a whole number. The product of `0.29` and `100` is `28.999999999999996`. This function compares the share of the count with the fraction, so `FractionOf(0.29)` of `100` members is `29`.

# Algorithm

 1. Round `part.val * n` to the nearest integer `c`.
 2. When `c / n > part.val`, subtract one from `c`.
 3. Return `c`.

# Arguments

  - `part`: The fraction of the set.
  - `n`: The size of the set.

# Returns

  - `c::Int`: The number of members. It is zero when the share of one member is more than the fraction.

# Examples

```jldoctest
julia> PortfolioOptimisers.fraction_floor(FractionOf(0.29), 100)
29

julia> PortfolioOptimisers.fraction_floor(FractionOf(0.5), 7)
3
```

# Related

  - [`FractionOf`](@ref)
  - [`split_count`](@ref)
  - [`resolve_cardinality_cap`](@ref)
"""
function fraction_floor(part::FractionOf, n::Real)::Int
    f = part.val
    c = round(Int, f * n)
    return c * one(f) / n > f ? c - 1 : c
end
"""
    cardinality_cap(x::Option{<:Union{UnresolvedValue, Integer, VecInt}})
    cardinality_cap(x::Real)
    cardinality_cap(x::VecReal_CountFraction)

Return the value that a slot of cardinality caps holds.

A cap is the largest number of members of a set that a portfolio holds. A plain `Integer` is the resolved cap, so the slot holds it unchanged, and it reads as a [`CountOf`](@ref) of the same number. The Result of a fit holds the integer too, and a sub-problem writes it back to the slot. Any other plain number goes through [`count_or_fraction`](@ref), which reads it as a [`FractionOf`](@ref) and refuses `1.0`. A member of the family, a [`TimeDependent`](@ref) schedule and `nothing` need no repair, so the function returns them unchanged.

# Returns

  - `x`: The value of the slot. A vector that holds a value other than an `Integer` gives a vector of the repaired entries.

# Related

  - [`AbstractCountOrFraction`](@ref)
  - [`count_or_fraction`](@ref)
  - [`assert_cardinality_cap`](@ref)
  - [`resolve_cardinality_cap`](@ref)
"""
function cardinality_cap(x::Option{<:Union{UnresolvedValue, Integer, VecInt}})
    return x
end
function cardinality_cap(x::Real)
    return count_or_fraction(x)
end
function cardinality_cap(x::VecReal_CountFraction)
    return concrete_typed_array(map(cardinality_cap, x))
end
"""
    assert_cardinality_cap(check, cap, name::Symbol)
    assert_cardinality_cap(check, cap::AbstractVector, name::Symbol)

Check a slot of cardinality caps with `check`, entry by entry when the slot holds a vector.

`check` is the value check of the slot, [`assert_nonempty_gt0_finite_val`](@ref) or [`assert_nonempty_nonneg_finite_val`](@ref). It checks an `Integer`. It passes a member of [`AbstractCountOrFraction`](@ref), because the constructor of the member checked its value.

# Validation

  - `!isempty(cap)` for a vector. An [`IsEmptyError`](@ref) is thrown otherwise.
  - The rules of `check` for each `Integer`.

# Returns

  - `nothing`.

# Related

  - [`cardinality_cap`](@ref)
  - [`resolve_cardinality_cap`](@ref)
"""
function assert_cardinality_cap(check::F, cap, name::Symbol)::Nothing where {F}
    return check(cap, name)
end
function assert_cardinality_cap(check::F, cap::AbstractVector,
                                name::Symbol)::Nothing where {F}
    @argcheck(!isempty(cap), IsEmptyError("$name cannot be empty"))
    for c in cap
        check(c, name)
    end
    return nothing
end
"""
    resolve_cardinality_cap(cap::Option{<:Int_VecInt}, set)
    resolve_cardinality_cap(cap::CountOf, set)
    resolve_cardinality_cap(cap::FractionOf, n::Real)
    resolve_cardinality_cap(cap::FractionOf, ns::AbstractVector)
    resolve_cardinality_cap(cap::FractionOf, smtx::MatNum)
    resolve_cardinality_cap(cap::FractionOf, set::Nothing)
    resolve_cardinality_cap(caps::VecReal_CountFraction, sets)

Resolves a cardinality cap against its set, to the largest number of members that the portfolio holds.

Only a [`FractionOf`](@ref) reads the set. It rounds down with [`fraction_floor`](@ref), because a cap that rounds up holds more than the stated fraction. The resolved number is at least one, so a small set keeps one member. A [`CountOf`](@ref) above the size of the set is a cap that does not bind, and it passes.

# Algorithm

 1. An `Integer`, a vector of them, and `nothing` need no resolution. Return `cap`.

 2. A [`CountOf`](@ref) gives its number.

 3. A [`FractionOf`](@ref) gives `max(fraction_floor(cap, n), 1)`, with `n` the size of its set:

      + A number is the size itself.
      + A vector of sizes gives a vector, with one cap for each size.
      + A selection matrix of sub-groups has the size of its rows that select an asset. A sub-group that a view or the Investable Mask leaves with no asset is not a member of the set.

 4. A vector of caps resolves each entry against the entry of `sets` at the same position.

# Arguments

  - `cap`: The cap that the slot holds.
  - `set`, `sets`, `n`, `ns`, `smtx`: The set behind a fraction.

# Validation

  - A [`FractionOf`](@ref) with `set === nothing`, where the universe is not known. An [`IsNothingError`](@ref) is thrown.

# Returns

  - `cap::Option{<:Int_VecInt}`: The resolved cap, one cap for each set, or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.resolve_cardinality_cap(FractionOf(0.2), 18)
3

julia> PortfolioOptimisers.resolve_cardinality_cap(CountOf(30), 18)
30

julia> PortfolioOptimisers.resolve_cardinality_cap(FractionOf(0.5), [1, 3, 4])
3-element Vector{Int64}:
 1
 1
 2
```

# Related

  - [`cardinality_cap`](@ref)
  - [`fraction_floor`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`IntegerPhylogeny`](@ref)
"""
function resolve_cardinality_cap(cap::Option{<:Int_VecInt}, ::Any)
    return cap
end
function resolve_cardinality_cap(cap::CountOf, ::Any)
    return cap.val
end
function resolve_cardinality_cap(cap::FractionOf, n::Real)::Int
    return max(fraction_floor(cap, n), 1)
end
function resolve_cardinality_cap(cap::FractionOf, ns::AbstractVector)
    return [resolve_cardinality_cap(cap, n) for n in ns]
end
function resolve_cardinality_cap(cap::FractionOf, smtx::MatNum)::Int
    return max(fraction_floor(cap, count(row -> any(!iszero, row), eachrow(smtx))), 1)
end
function resolve_cardinality_cap(cap::FractionOf, ::Nothing)
    return throw(IsNothingError("the cardinality cap $(cap.val) is a fraction of the assets, and the block resolves with no rows and no number of assets. State the cap as a `CountOf`, or resolve the block on its rows."))
end
function resolve_cardinality_cap(caps::VecReal_CountFraction, sets)
    return [resolve_cardinality_cap(cap, set) for (cap, set) in zip(caps, sets)]
end

export CountOf, FractionOf
public AbstractCountOrFraction
