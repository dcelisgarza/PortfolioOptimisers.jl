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
"""
@curryable @concrete struct WindowedExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:me])
    """
    @c me
    """
    $(field_dict[:oow])
    """
    @c w
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
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the expected returns estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:me])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:mev])

# Related

  - [`WindowedExpectedReturns`](@ref)
"""
function moment_view(me::WindowedExpectedReturns, i)::WindowedExpectedReturns
    return WindowedExpectedReturns(; me = moment_view(me.me, i), w = me.w,
                                   window = me.window)
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
"""
function Statistics.mean(me::WindowedExpectedReturns, X::MatNum; dims::Int = 1,
                         iv::Option{<:MatNum} = nothing, kwargs...)
    window = get_window(me.window, X, dims)
    X, w = moment_window_and_weights(X, me.w, window; dims = dims, kwargs...)
    me = factory(me.me, w)
    if !isnothing(iv) && isa(window, VecInt)
        iv = isone(dims) ? view(iv, window, :) : view(iv, :, window)
    end
    return Statistics.mean(me, X; dims = dims, iv = iv, kwargs...)
end

export WindowedExpectedReturns
