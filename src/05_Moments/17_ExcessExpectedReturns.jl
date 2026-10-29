"""
$(DocStringExtensions.TYPEDEF)

Subtracts a risk-free rate from the expected returns that a nested estimator computes.

The nested estimator `me` computes the expected returns, and this type subtracts `rf` from each element of its result. `me` can hold any expected returns estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExcessExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        rf::Number = 0.0
    ) -> ExcessExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:rf])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ExcessExpectedReturns()
ExcessExpectedReturns
  me ┼ SimpleExpectedReturns
     │   w ┴ nothing
  rf ┴ Float64: 0.0
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct ExcessExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:rf])
    """
    rf
    function ExcessExpectedReturns(me::AbstractExpectedReturnsEstimator, rf::Number)
        @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
        return new{typeof(me), typeof(rf)}(me, rf)
    end
end
function ExcessExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               rf::Number = 0.0)::ExcessExpectedReturns
    return ExcessExpectedReturns(me, rf)
end
"""
    Statistics.mean(me::ExcessExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute excess expected returns by subtracting the risk-free rate.

This method applies the nested estimator `me.me` to the data, and subtracts the risk-free rate `me.rf` from every element of the result. The method reads the rate from the estimator alone, and never from the data or from a keyword.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{\\text{excess}} &= \\hat{\\boldsymbol{\\mu}} - r_f \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{\\text{excess}}``: The ``N \\times 1`` vector of excess expected returns.
  - ``\\hat{\\boldsymbol{\\mu}}``: The ``N \\times 1`` vector of expected returns that `me.me` estimates.
  - ``r_f``: The risk-free rate `me.rf`.
  - ``\\boldsymbol{1}``: The ``N \\times 1`` vector of ones.

# Arguments

  - `me`: The excess expected returns estimator.
  - `X`: The data matrix, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method passes to the nested estimator `me.me`.

# Validation

  - $(val_dict[:dims])

# Returns

  - `mu::ArrNum`: The excess expected returns. The subtraction is elementwise, so the result has the shape of the result of `me.me`.

# Examples

```jldoctest
julia> me = ExcessExpectedReturns(; rf = 0.01);

julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> mean(me, X)
1×2 Matrix{Float64}:
 0.01  0.02
```

# Related

  - [`ExcessExpectedReturns`](@ref)
"""
function Statistics.mean(me::ExcessExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    return Statistics.mean(me.me, X; dims = dims, kwargs...) .- me.rf
end
function effective_observation_count(me::ExcessExpectedReturns, X::MatNum)
    return effective_observation_count(me.me, X)
end

export ExcessExpectedReturns
