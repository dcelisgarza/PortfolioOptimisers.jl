"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that restricts computation to a rolling or indexed observation window.

`WindowedExpectedReturns` wraps another expected returns estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted expected returns estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        w::Option{<:ObsWeights} = nothing,
        window::Option{<:Int_VecInt} = nothing
    ) -> WindowedExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - If `window` is provided, it must be nonempty, nonnegative, and finite.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `me`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> WindowedExpectedReturns()
WindowedExpectedReturns
      me ┼ SimpleExpectedReturns
         │   w ┴ nothing
       w ┼ nothing
  window ┴ nothing
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct WindowedExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    Window specification: an integer (last `window` observations) or a vector of indices.
    """
    window
    function WindowedExpectedReturns(me::AbstractExpectedReturnsEstimator,
                                     w::Option{<:ObsWeights}, window::Option{<:Int_VecInt})
        assert_nonempty_nonneg_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(me), typeof(w), typeof(window)}(me, w, window)
    end
end
function WindowedExpectedReturns(;
                                 me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                 w::Option{<:ObsWeights} = nothing,
                                 window::Option{<:Int_VecInt} = nothing)::WindowedExpectedReturns
    return WindowedExpectedReturns(me, w, window)
end
"""
    Statistics.mean(me::WindowedExpectedReturns, X::MatNum; dims::Int = 1, iv::Option{<:MatNum} = nothing, kwargs...)

Compute expected returns using a rolling or indexed observation window.

This method selects a window of observations from `X` (and applies observation weights if specified), then delegates to the underlying mean estimator.

# Arguments

  - `me`: Windowed expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - $(arg_dict[:oiv])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - `mu::ArrNum`: Expected returns vector for the selected window.

# Related

  - [`WindowedExpectedReturns`](@ref)
  - [`windowed_preamble`](@ref)
"""
function Statistics.mean(me::WindowedExpectedReturns, X::MatNum; dims::Int = 1,
                         iv::Option{<:MatNum} = nothing, kwargs...)
    inner, X, iv = windowed_preamble(me.me, me.w, me.window, X; iv = iv, dims = dims,
                                     kwargs...)
    return Statistics.mean(inner, X; dims = dims, iv = iv, kwargs...)
end

export WindowedExpectedReturns
