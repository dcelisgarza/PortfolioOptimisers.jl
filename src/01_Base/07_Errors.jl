"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom exception types.

All error types of `PortfolioOptimisers.jl` must subtype `PortfolioOptimisersError`.

# Related

  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`IsZeroError`](@ref)
  - [`ConflictingArgumentError`](@ref)
  - [`PropertyPathError`](@ref)
  - [`ObservationWeightsError`](@ref)
  - [`NonPositiveWealthError`](@ref)
  - [`PosdefRepairError`](@ref)
"""
abstract type PortfolioOptimisersError <: Exception end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IsNothingError(msg) -> IsNothingError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNothingError(\"Input data must not be nothing\"))
ERROR: IsNothingError: Input data must not be nothing
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
@concrete struct IsNothingError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly empty.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IsEmptyError(msg) -> IsEmptyError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsEmptyError(\"Input array must not be empty\"))
ERROR: IsEmptyError: Input array must not be empty
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
@concrete struct IsEmptyError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly non-finite, for example when it holds `NaN` or `Inf`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IsNonFiniteError(msg) -> IsNonFiniteError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsNonFiniteError(\"Input array contains non-finite values\"))
ERROR: IsNonFiniteError: Input array contains non-finite values
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`ConflictingArgumentError`](@ref)
"""
@concrete struct IsNonFiniteError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when an argument or value is unexpectedly zero, such as a constraint row whose coefficients are all zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IsZeroError(msg) -> IsZeroError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(IsZeroError(\"Constraint row must not be all zero\"))
ERROR: IsZeroError: Constraint row must not be all zero
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`strict_diagnostic`](@ref)
"""
@concrete struct IsZeroError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when the caller gives two arguments that exclude each other. One argument must be `nothing` when the other is set, and the caller set both.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConflictingArgumentError(msg) -> ConflictingArgumentError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(ConflictingArgumentError(\"sbgt must be nothing when bgt is a BudgetCostEstimator\"))
ERROR: ConflictingArgumentError: sbgt must be nothing when bgt is a BudgetCostEstimator
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
"""
@concrete struct ConflictingArgumentError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when [`@forward_properties`](@ref) cannot follow a nested path, because a node in the path is `nothing`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PropertyPathError(msg) -> PropertyPathError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(PropertyPathError(\"cannot descend path `sol.w` on `JuMPOptimisationResult`: intermediate `sol` is `nothing`\"))
ERROR: PropertyPathError: cannot descend path `sol.w` on `JuMPOptimisationResult`: intermediate `sol` is `nothing`
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`@forward_properties`](@ref)
"""
@concrete struct PropertyPathError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when a [`DynamicAbstractWeights`](@ref) cannot resolve the observation weights for its data, because [`get_observation_weights`](@ref) has no method for the shape of that data.

A `nothing` from [`get_observation_weights`](@ref) means that the caller asked for no weights. It never means that the weights are not available. Each `isnothing` branch after it reads `nothing` in that way and computes an unweighted result. A `DynamicAbstractWeights` that resolves to `nothing` thus gives an unweighted answer that looks correct, with no message. The library raises this error instead.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ObservationWeightsError(msg) -> ObservationWeightsError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(ObservationWeightsError(\"MyWeights has no `get_observation_weights` method for a 2-dimensional input of size (3, 10)\"))
ERROR: ObservationWeightsError: MyWeights has no `get_observation_weights` method for a 2-dimensional input of size (3, 10)
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`DynamicAbstractWeights`](@ref)
  - [`get_observation_weights`](@ref)
"""
@concrete struct ObservationWeightsError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when a drifted portfolio's wealth reaches zero or turns negative over the observations it is scored on.

A drifted series divides by the wealth of the previous observation, so a wealth of zero or below is outside the domain of the series rather than a large loss inside it. The check runs before any return is formed, so a ruined window gives no partial series. A negative wealth is **finite**, and every leg of the record flips its sign, so nothing downstream reads the failure from a `NaN`. That is why the drift raises rather than returning a value.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NonPositiveWealthError(msg) -> NonPositiveWealthError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(NonPositiveWealthError(\"the drifted wealth must satisfy `all(>(0), wealth)`, but the wealth is -0.3975 at row 2 of the window\"))
ERROR: NonPositiveWealthError: the drifted wealth must satisfy `all(>(0), wealth)`, but the wealth is -0.3975 at row 2 of the window
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`SelfFinancingDrift`](@ref)
  - [`assert_positive_wealth`](@ref)
  - [`non_positive_wealth_index`](@ref)
"""
@concrete struct NonPositiveWealthError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDEF)

Exception type thrown when a positive definite repair returns a matrix that is not positive semidefinite.

A consumer factorises the matrix that a repair returns, or optimises over it. A matrix with an eigenvalue below the round-off of a positive semidefinite matrix gives a negative variance in one direction. Each consumer then reads a wrong number, or gets a LAPACK error that does not name the cause. The repair thus raises this error, and does not return the matrix. A singular result that is positive semidefinite to round-off is a valid covariance, and the repair returns it with no message.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PosdefRepairError(msg) -> PosdefRepairError

Arguments correspond to the fields above.

# Examples

```jldoctest
julia> throw(PosdefRepairError(\"the repair of `Newton()` must give a matrix whose smallest eigenvalue is at least -8.881784197001252e-16, but it is -1.0\"))
ERROR: PosdefRepairError: the repair of `Newton()` must give a matrix whose smallest eigenvalue is at least -8.881784197001252e-16, but it is -1.0
Stacktrace:
 [1] top-level scope
   @ none:1
```

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`Posdef`](@ref)
  - [`posdef!`](@ref)
  - [`assert_posdef_repair`](@ref)
"""
@concrete struct PosdefRepairError <: PortfolioOptimisersError
    """
    $(field_dict[:msg])
    """
    msg
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Print a subtype of `PortfolioOptimisersError` to `io` as its type name and its message, with no type parameters.

# Algorithm

 1. Take `name`, the string of the concrete type of `err`.
 2. Cut `name` at the first `{` or `(`, so that a parametric subtype prints with its wrapper name alone.
 3. Print `name`, a colon, and the `msg` field of `err`.

# Arguments

  - `io`: The stream that the function prints the message to.
  - `err`: The error to render.

# Returns

  - `nothing`.

# Related

  - [`PortfolioOptimisersError`](@ref)
  - [`first_error_line`](@ref)
"""
function Base.showerror(io::IO, err::PortfolioOptimisersError)
    name = string(typeof(err))
    name = name[1:(findfirst(x -> (x == '{' || x == '('), name) - 1)]
    return print(io, "$name: $(err.msg)")
end

export IsEmptyError, IsNothingError, IsNonFiniteError, IsZeroError,
       ConflictingArgumentError, PropertyPathError, ObservationWeightsError,
       NonPositiveWealthError, PosdefRepairError
