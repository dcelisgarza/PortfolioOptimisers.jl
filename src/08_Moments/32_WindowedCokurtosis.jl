"""
$(DocStringExtensions.TYPEDEF)

Cokurtosis estimator that restricts computation to a rolling or indexed observation window.

`WindowedCokurtosis` wraps a [`Cokurtosis`](@ref) estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted cokurtosis estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedCokurtosis(;
        ke::CokurtosisEstimator = Cokurtosis(),
        w::Option{<:ObsWeights} = nothing,
        window::Option{<:Int_VecInt} = nothing
    ) -> WindowedCokurtosis

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - If `window` is provided, it must be nonempty, nonnegative, and finite.

# Examples

```jldoctest
julia> WindowedCokurtosis()
WindowedCokurtosis
      ke ┼ Cokurtosis
         │    me ┼ SimpleExpectedReturns
         │       │   w ┴ nothing
         │    mp ┼ MatrixProcessing
         │       │     pdm ┼ Posdef
         │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
         │       │      dn ┼ nothing
         │       │      dt ┼ nothing
         │       │     alg ┼ nothing
         │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
         │   alg ┼ FullMoment()
         │     w ┴ nothing
       w ┼ nothing
  window ┴ nothing
```

# Related

  - [`CokurtosisEstimator`](@ref)
  - [`Cokurtosis`](@ref)
"""
@propagatable @concrete struct WindowedCokurtosis <: CokurtosisEstimator
    """
    Cokurtosis estimator.
    """
    @fprop @vprop ke
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    Window specification: an integer (last `window` observations) or a vector of indices.
    """
    window
    function WindowedCokurtosis(ke::CokurtosisEstimator, w::Option{<:ObsWeights},
                                window::Option{<:Int_VecInt})
        assert_nonempty_nonneg_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ke), typeof(w), typeof(window)}(ke, w, window)
    end
end
function WindowedCokurtosis(; ke::CokurtosisEstimator = Cokurtosis(),
                            w::Option{<:ObsWeights} = nothing,
                            window::Option{<:Int_VecInt} = nothing)::WindowedCokurtosis
    return WindowedCokurtosis(ke, w, window)
end
"""
    cokurtosis(ke::WindowedCokurtosis, X::MatNum; dims::Int = 1, iv::Option{<:MatNum} = nothing, kwargs...)

Compute the cokurtosis tensor using a rolling or indexed observation window.

This method selects a window of observations from `X` (and applies observation weights if specified), then delegates to the underlying cokurtosis estimator.

# Arguments

  - `ke`: Windowed cokurtosis estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - $(arg_dict[:oiv])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - `ckurt::Matrix{<:Number}`: Cokurtosis tensor (assets² × assets²).

# Related

  - [`WindowedCokurtosis`](@ref)
  - [`Cokurtosis`](@ref)
  - [`cokurtosis`](@ref)
  - [`windowed_preamble`](@ref)
"""
function cokurtosis(ke::WindowedCokurtosis, X::MatNum; dims::Int = 1,
                    iv::Option{<:MatNum} = nothing, kwargs...)
    inner, X, iv = windowed_preamble(ke.ke, ke.w, ke.window, X; iv = iv, dims = dims,
                                     kwargs...)
    return cokurtosis(inner, X; dims = dims, iv = iv, kwargs...)
end

export WindowedCokurtosis
