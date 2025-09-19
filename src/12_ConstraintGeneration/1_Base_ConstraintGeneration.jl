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
abstract type ComparisonOperators end
```

Abstract supertype for all comparison operator types used in constraint generation.

Concrete subtypes represent specific comparison semantics (e.g., equality, inequality) for use in constraint definitions and evaluation.

# Related

  - [`EqualityComparisonOperators`](@ref)
  - [`InequalityComparisonOperators`](@ref)
"""
abstract type ComparisonOperators end

"""
```julia
abstract type EqualityComparisonOperators <: ComparisonOperators end
```

Abstract supertype for all equality comparison operator types.

Concrete subtypes represent equality-based comparison semantics (e.g., `EQ`) for use in constraint definitions.

# Related

  - [`ComparisonOperators`](@ref)
  - [`EQ`](@ref)
"""
abstract type EqualityComparisonOperators <: ComparisonOperators end

"""
```julia
abstract type InequalityComparisonOperators <: ComparisonOperators end
```

Abstract supertype for all inequality comparison operator types.

Concrete subtypes represent inequality-based comparison semantics (e.g., `LEQ`, `GEQ`) for use in constraint definitions.

# Related

  - [`ComparisonOperators`](@ref)
  - [`LEQ`](@ref)
  - [`GEQ`](@ref)
"""
abstract type InequalityComparisonOperators <: ComparisonOperators end

"""
```julia
struct EQ <: EqualityComparisonOperators end
```

Equality comparison operator for constraint generation.

# Related

  - [`EqualityComparisonOperators`](@ref)
  - [`LEQ`](@ref)
  - [`GEQ`](@ref)
"""
struct EQ <: EqualityComparisonOperators end

"""
```julia
struct LEQ <: InequalityComparisonOperators end
```

Less-than-or-equal-to comparison operator for constraint generation.

# Related

  - [`InequalityComparisonOperators`](@ref)
  - [`EQ`](@ref)
  - [`GEQ`](@ref)
"""
struct LEQ <: InequalityComparisonOperators end

"""
```julia
struct GEQ <: InequalityComparisonOperators end
```

Greater-than-or-equal-to comparison operator for constraint generation.

# Related

  - [`InequalityComparisonOperators`](@ref)
  - [`EQ`](@ref)
  - [`LEQ`](@ref)
"""
struct GEQ <: InequalityComparisonOperators end

"""
```julia
comparison_sign_ineq_flag(op::ComparisonOperators)
```

Return the multiplicative sign and inequality flag for a given comparison operator.

# Arguments

  - `op::ComparisonOperators`: The comparison operator.

# Returns

  - `sign`: The multiplicative sign for the constraint.
  - `is_inequality`: `true` if the operator is an inequality, `false` for equality.

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
  - [`ComparisonOperators`](@ref)
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
