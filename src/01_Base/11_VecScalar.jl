"""
$(DocStringExtensions.TYPEDEF)

Represents a composite result containing a vector and a scalar.

Encapsulates a vector and a scalar value, commonly used for storing results that combine both types of data (e.g., weighted statistics, risk measures).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VecScalar(;
        v::VecNum,
        s::Number
    ) -> VecScalar

Keywords correspond to the struct's fields.

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
    Vector component.
    """
    v
    """
    Scalar component.
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
    const Num_VecNum_VecScalar = Union{<:Num_VecNum, <:VecScalar}

Alias for a union of a numeric type, a vector of numeric types, or a `VecScalar` result.

# Related

  - [`Num_VecNum`](@ref)
  - [`VecScalar`](@ref)
"""
const Num_VecNum_VecScalar = Union{<:Num_VecNum, <:VecScalar}
"""
    const Num_ArrNum_VecScalar_DynWeights = Union{<:Num_ArrNum, <:VecScalar, <:DynamicAbstractWeights}

Alias for a union of a numeric type, an array of numeric types, or a `VecScalar` result.

# Related

  - [`Num_ArrNum`](@ref)
  - [`VecScalar`](@ref)
"""
const Num_ArrNum_VecScalar_DynWeights = Union{<:Num_ArrNum, <:VecScalar,
                                              <:DynamicAbstractWeights}

export VecScalar
