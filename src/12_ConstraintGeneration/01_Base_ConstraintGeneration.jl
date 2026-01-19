"""
    abstract type AbstractConstraintResult <: AbstractResult end

Abstract supertype for all constraint result types in PortfolioOptimisers.jl.

All concrete types representing the result of constraint generation or evaluation should subtype `AbstractConstraintResult`.

# Related

  - [`AbstractConstraintEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractConstraintResult <: AbstractResult end
"""
    abstract type AbstractConstraintEstimator <: AbstractEstimator end

Abstract supertype for all constraint estimator types in PortfolioOptimisers.jl.

All concrete types implementing constraint generation or estimation algorithms should subtype `AbstractConstraintEstimator`. This enables extensible and composable workflows for constraint construction and validation.

# Related

  - [`AbstractConstraintResult`](@ref)
  - [`AbstractEstimator`](@ref)
"""
abstract type AbstractConstraintEstimator <: AbstractEstimator end
"""
    const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}

Union type representing supported comparison operators for constraint generation.

This type is used to specify which comparison operators are valid for defining constraints in PortfolioOptimisers.jl. It includes equality and both directions of inequality.

# Related

  - [`comparison_sign_ineq_flag`](@ref)
"""
const ComparisonOperator = Union{typeof(==), typeof(<=), typeof(>=)}
"""
    comparison_sign_ineq_flag(op::ComparisonOperator)

Return the multiplicative sign and inequality flag for a given comparison operator.

# Arguments

  - `op::ComparisonOperator`: The comparison operator.

# Returns

  - `sign::Int`: The multiplicative sign for the constraint.
  - `is_inequality::Bool`: `true` if the operator is an inequality, `false` for equality.

# Examples

```jldoctest
julia> PortfolioOptimisers.comparison_sign_ineq_flag(==)
(1, false)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(<=)
(1, true)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(>=)
(-1, true)
```

# Related

  - [`ComparisonOperator`](@ref)
"""
function comparison_sign_ineq_flag(::typeof(==))
    return 1, false
end
function comparison_sign_ineq_flag(::typeof(<=))
    return 1, true
end
function comparison_sign_ineq_flag(::typeof(>=))
    return -1, true
end
