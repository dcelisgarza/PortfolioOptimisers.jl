"""
$(DocStringExtensions.TYPEDEF)

Cokurtosis estimator that restricts computation to a rolling or indexed observation window.

`WindowedCokurtosis` wraps a [`Cokurtosis`](@ref) estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted cokurtosis estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedCokurtosis(;
        ke::Cokurtosis = Cokurtosis(),
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
         │   alg ┼ Full()
         │     w ┴ nothing
       w ┼ nothing
  window ┴ nothing
```

# Related

  - [`CokurtosisEstimator`](@ref)
  - [`Cokurtosis`](@ref)
"""
@concrete struct WindowedCokurtosis <: CokurtosisEstimator
    """
    Cokurtosis estimator.
    """
    ke
    """
    $(field_dict[:oow])
    """
    w
    """
    Window specification: an integer (last `window` observations) or a vector of indices.
    """
    window
    function WindowedCokurtosis(ke::Cokurtosis, w::Option{<:ObsWeights},
                                window::Option{<:Int_VecInt})
        assert_nonempty_nonneg_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ke), typeof(w), typeof(window)}(ke, w, window)
    end
end
function WindowedCokurtosis(; ke::Cokurtosis = Cokurtosis(),
                            w::Option{<:ObsWeights} = nothing,
                            window::Option{<:Int_VecInt} = nothing)::WindowedCokurtosis
    return WindowedCokurtosis(ke, w, window)
end
"""
    factory(ke::WindowedCokurtosis, w::ObsWeights) -> WindowedCokurtosis

Return a new [`WindowedCokurtosis`](@ref) estimator with observation weights `w` applied to the underlying cokurtosis estimator and stored as the windowed weights.

# Arguments

  - `ke`: Windowed cokurtosis estimator.
  - $(arg_dict[:ow])

# Returns

  - `ke::WindowedCokurtosis`: Updated estimator with weights applied.

# Related

  - [`WindowedCokurtosis`](@ref)
  - [`factory`](@ref)
"""
function factory(ke::WindowedCokurtosis, w::ObsWeights)::WindowedCokurtosis
    return WindowedCokurtosis(; ke = factory(ke.ke, w), w = w, window = ke.window)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the coskewness estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:kte])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:ktev])

# Related

  - [`WindowedCokurtosis`](@ref)
"""
function port_opt_view(kte::WindowedCokurtosis, i, args...)::WindowedCokurtosis
    return WindowedCokurtosis(; kte = port_opt_view(kte.kte, i), w = kte.w,
                              window = kte.window)
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
"""
function cokurtosis(ke::WindowedCokurtosis, X::MatNum; dims::Int = 1,
                    iv::Option{<:MatNum} = nothing, kwargs...)
    window = get_window(ke.window, X, dims)
    X, w = moment_window_and_weights(X, ke.w, window; dims = dims, kwargs...)
    ke = factory(ke.ke, w)
    if !isnothing(iv) && isa(window, VecInt)
        iv = isone(dims) ? view(iv, window, :) : view(iv, :, window)
    end
    return cokurtosis(ke, X; dims = dims, iv = iv, kwargs...)
end

export WindowedCokurtosis
