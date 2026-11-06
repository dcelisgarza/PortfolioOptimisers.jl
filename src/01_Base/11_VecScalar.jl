"""
$(DocStringExtensions.TYPEDEF)

A result that holds a vector and a scalar.

The moment risk measures read a `VecScalar` as a target `mu`. The target of a portfolio with weights `w` is `dot(w, mu.v) + mu.s`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VecScalar(;
        v::VecNum,
        s::Number
    ) -> VecScalar

Each keyword sets the field of the same name.

## Validation

  - `v`: `!isempty(v)` and `all(isfinite, v)`.
  - `s`: `isfinite(s)`.

# Examples

```jldoctest
julia> VecScalar([1.0, 2.0, 3.0], 4.2)
VecScalar
  v ┼ Vector{Float64}: [1.0, 2.0, 3.0]
  s ┴ Float64: 4.2
```

# Related

  - [`AbstractResult`](@ref)
  - [`VecNum`](@ref)
"""
@concrete struct VecScalar <: AbstractResult
    """
    The vector part.
    """
    v
    """
    The scalar part.
    """
    s
    function VecScalar(v::VecNum, s::Number)
        assert_nonempty_finite_val(v, :v)
        assert_nonempty_finite_val(s, :s)
        return new{typeof(v), typeof(s)}(v, s)
    end
end
function VecScalar(; v::VecNum, s::Number)
    return VecScalar(v, s)
end
"""
    const Num_VecNum_VecScalar = Union{Num_VecNum, VecScalar}

Alias for a union of a numeric type, a vector of numeric types and a `VecScalar` result.

# Related

  - [`Num_VecNum`](@ref)
  - [`VecScalar`](@ref)
"""
const Num_VecNum_VecScalar = Union{Num_VecNum, VecScalar}
"""
    const Num_ArrNum_VecScalar_DynWeights = Union{Num_ArrNum, VecScalar, DynamicAbstractWeights}

Alias for a union of a numeric type, an array of numeric types, a `VecScalar` result and a [`DynamicAbstractWeights`](@ref).

# Related

  - [`Num_ArrNum`](@ref)
  - [`VecScalar`](@ref)
  - [`DynamicAbstractWeights`](@ref)
"""
const Num_ArrNum_VecScalar_DynWeights = Union{Num_ArrNum, VecScalar, DynamicAbstractWeights}

export VecScalar
