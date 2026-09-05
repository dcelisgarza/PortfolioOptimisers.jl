"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all custom exception types.

All error types specific to `PortfolioOptimisers.jl` should be subtypes of `PortfolioOptimisersError`.

# Related

  - [`IsNothingError`](@ref)
  - [`IsEmptyError`](@ref)
  - [`IsNonFiniteError`](@ref)
  - [`ConflictingArgumentError`](@ref)
  - [`PropertyPathError`](@ref)
  - [`ObservationWeightsError`](@ref)
  - [`NonPositiveWealthError`](@ref)
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

Exception type thrown when an argument or value is unexpectedly non-finite (e.g., contains `NaN` or `Inf`).

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

Exception type thrown when an argument or value is mutually exclusive with another and both were supplied — a "must-be-absent" constraint was violated (e.g. an argument that must be `nothing` because a conflicting one is set).

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

Exception type thrown when a [`@forward_properties`](@ref) nested path cannot be descended because an intermediate node is `nothing`.

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

Exception type thrown when a [`DynamicAbstractWeights`](@ref) cannot resolve observation weights for the data it was handed, because no [`get_observation_weights`](@ref) method is implemented for that input's shape.

[`get_observation_weights`](@ref) returns `nothing` to mean *no weights were requested*, never *weights were unavailable*. Every `isnothing` branch downstream reads it the first way and computes an unweighted result, so a `DynamicAbstractWeights` that resolved to `nothing` would silently produce a numerically plausible but unweighted answer with no diagnostic. It raises instead.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Print human-readable representation of `PortfolioOptimisersError` subtypes to `io`, stripping parametric type suffixes.

# Algorithm

 1. Take `name`, the string of the concrete type of `err`.
 2. Cut `name` at the first `{` or `(`, so a parametric subtype prints under its wrapper name alone.
 3. Print `name`, a colon, and the `msg` field of `err`.

# Arguments

  - `io`: Stream the message is printed to.
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

export IsEmptyError, IsNothingError, IsNonFiniteError, ConflictingArgumentError,
       PropertyPathError, ObservationWeightsError, NonPositiveWealthError
