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

export CountOf, FractionOf
public AbstractCountOrFraction
