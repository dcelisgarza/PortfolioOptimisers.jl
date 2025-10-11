"""
```julia
abstract type AbstractConstraintResult <: AbstractResult end
```

Abstract supertype for all constraint result types in PortfolioOptimisers.jl.

All concrete types representing the result of constraint generation or evaluation should subtype `AbstractConstraintResult`. This enables a consistent interface for handling constraint results across different estimators and algorithms.

# Related

  - [`AbstractConstraintEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractConstraintResult <: AbstractResult end
"""
```julia
abstract type AbstractConstraintEstimator <: AbstractEstimator end
```

Abstract supertype for all constraint estimator types in PortfolioOptimisers.jl.

All concrete types implementing constraint generation or estimation algorithms should subtype `AbstractConstraintEstimator`. This enables extensible and composable workflows for constraint construction and validation.

# Related

  - [`AbstractConstraintResult`](@ref)
  - [`AbstractEstimator`](@ref)
"""
abstract type AbstractConstraintEstimator <: AbstractEstimator end
"""
```julia
abstract type ComparisonOperator end
```

Abstract supertype for all comparison operator types used in constraint generation.

Concrete subtypes represent specific comparison semantics (e.g., equality, inequality) for use in constraint definitions and evaluation.

# Related

  - [`EqualityComparisonOperator`](@ref)
  - [`InequalityComparisonOperator`](@ref)
"""
abstract type ComparisonOperator end
"""
```julia
abstract type EqualityComparisonOperator <: ComparisonOperator end
```

Abstract supertype for all equality comparison operator types.

Concrete subtypes represent equality-based comparison semantics (e.g., `EQ`) for use in constraint definitions.

# Related

  - [`ComparisonOperator`](@ref)
  - [`EQ`](@ref)
"""
abstract type EqualityComparisonOperator <: ComparisonOperator end
"""
```julia
abstract type InequalityComparisonOperator <: ComparisonOperator end
```

Abstract supertype for all inequality comparison operator types.

Concrete subtypes represent inequality-based comparison semantics (e.g., `LEQ`, `GEQ`) for use in constraint definitions.

# Related

  - [`ComparisonOperator`](@ref)
  - [`LEQ`](@ref)
  - [`GEQ`](@ref)
"""
abstract type InequalityComparisonOperator <: ComparisonOperator end
"""
```julia
struct EQ <: EqualityComparisonOperator end
```

Equality comparison operator for constraint generation.

# Related

  - [`EqualityComparisonOperator`](@ref)
  - [`LEQ`](@ref)
  - [`GEQ`](@ref)
"""
struct EQ <: EqualityComparisonOperator end
"""
```julia
struct LEQ <: InequalityComparisonOperator end
```

Less-than-or-equal-to comparison operator for constraint generation.

# Related

  - [`InequalityComparisonOperator`](@ref)
  - [`EQ`](@ref)
  - [`GEQ`](@ref)
"""
struct LEQ <: InequalityComparisonOperator end
"""
```julia
struct GEQ <: InequalityComparisonOperator end
```

Greater-than-or-equal-to comparison operator for constraint generation.

# Related

  - [`InequalityComparisonOperator`](@ref)
  - [`EQ`](@ref)
  - [`LEQ`](@ref)
"""
struct GEQ <: InequalityComparisonOperator end
"""
```julia
comparison_sign_ineq_flag(op::ComparisonOperator)
```

Return the multiplicative sign and inequality flag for a given comparison operator.

# Arguments

  - `op::ComparisonOperator`: The comparison operator.

# Returns

  - `sign::Int`: The multiplicative sign for the constraint.
  - `is_inequality::Bool`: `true` if the operator is an inequality, `false` for equality.

# Examples

```jldoctest
julia> PortfolioOptimisers.comparison_sign_ineq_flag(EQ())
(1, false)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(LEQ())
(1, true)

julia> PortfolioOptimisers.comparison_sign_ineq_flag(GEQ())
(-1, true)
```

# Related

  - [`EQ`](@ref)
  - [`LEQ`](@ref)
  - [`GEQ`](@ref)
  - [`ComparisonOperator`](@ref)
"""
function comparison_sign_ineq_flag(::EQ)
    return 1, false
end
function comparison_sign_ineq_flag(::LEQ)
    return 1, true
end
function comparison_sign_ineq_flag(::GEQ)
    return -1, true
end

export EQ, LEQ, GEQ
