"""
$(DocStringExtensions.TYPEDEF)

Container type for excess expected returns estimators.

`ExcessExpectedReturns` encapsulates a mean estimator and a risk-free rate for computing excess expected returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExcessExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        rf::Number = 0.0
    ) -> ExcessExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ExcessExpectedReturns()
ExcessExpectedReturns
  me ┼ SimpleExpectedReturns
     │   w ┴ nothing
  rf ┴ Float64: 0.0
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct ExcessExpectedReturns <:
                               AbstractShrunkExpectedReturnsEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:rf])
    """
    rf
    function ExcessExpectedReturns(me::AbstractExpectedReturnsEstimator, rf::Number)
        return new{typeof(me), typeof(rf)}(me, rf)
    end
end
#= Old factory function:
function factory(me::ExcessExpectedReturns, w::ObsWeights)::ExcessExpectedReturns
    return ExcessExpectedReturns(; me = factory(me.me, w), rf = me.rf)
end
=#
function ExcessExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               rf::Number = 0.0)::ExcessExpectedReturns
    return ExcessExpectedReturns(me, rf)
end
"""
    Statistics.mean(me::ExcessExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute excess expected returns by subtracting the risk-free rate.

This method applies the mean estimator to the data and subtracts the risk-free rate from the resulting expected returns.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{\\text{excess}} &= \\hat{\\boldsymbol{\\mu}} - r_f \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{\\text{excess}}``: ``N \\times 1`` vector of excess expected returns.
  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` vector of estimated expected returns.
  - ``r_f``: Risk-free rate.
  - ``\\boldsymbol{1}``: ``N \\times 1`` vector of ones.

# Arguments

  - `me`: Excess expected returns estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - `mu::ArrNum`: Excess expected returns vector.

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
    return Statistics.mean(me.me, X; dims = dims, kwargs...) .- me.rf
end

export ExcessExpectedReturns
