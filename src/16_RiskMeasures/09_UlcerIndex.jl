"""
$(DocStringExtensions.TYPEDEF)

Measures the Ulcer Index, the root mean square of the uncompounded drawdown over the sample.

The drawdown series is the one of [`absolute_drawdown_vec`](@ref): the cumulative return is a sum of the returns, and its running peak starts at ``c_0 = 0``. The square gives a deep drawdown more weight than [`AverageDrawdown`](@ref) does, and a long drawdown adds one term for each period that it lasts. [`RelativeUlcerIndex`](@ref) takes the same root mean square of the compounded drawdown series.

# Mathematical definition

```math
\\begin{align}
\\mathrm{UI}(\\boldsymbol{x}) &= \\sqrt{\\frac{1}{T} \\sum_{t=1}^{T} d_t^2} = \\frac{\\lVert \\boldsymbol{d} \\rVert_2}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{UI}(\\boldsymbol{x})``: Ulcer Index (root-mean-square drawdown).
  - $(math_dict[:xret])
  - $(math_dict[:dtdd]) [`absolute_drawdown_vec`](@ref) states the series.
  - ``\\boldsymbol{d}``: Absolute drawdown series vector ``T \\times 1``.
  - $(math_dict[:T])

With observation weights, the index takes the weighted mean of the squared drawdowns:

```math
\\begin{align}
\\mathrm{UI}(\\boldsymbol{x}) &= \\sqrt{\\frac{1}{W_{T}} \\sum_{t=1}^{T} w_{t} d_t^2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_obs])
  - $(math_dict[:W_T_total])

The drawdown series reads the full return path, so an observation with zero weight still moves every later drawdown. A path that never falls below its running peak has an Ulcer Index of zero. The index is never below [`AverageDrawdown`](@ref) on the same path and weights, because a root mean square is never below a mean of the same nonnegative values.

Equation 7.88 of [cajas2025](@cite) states the index without weights. The weighted mean of the squares is the library's extension of that form.

# Algorithm

 1. Compute the drawdown series with [`absolute_drawdown_vec`](@ref).
 2. Resolve the observation weights with [`checked_observation_weights`](@ref).
 3. Return [`ulcer_index`](@ref) of the series and the weights.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    UlcerIndex(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing
    ) -> UlcerIndex

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow_rm])

# Functor

    (r::UlcerIndex)(x::VecNum)

Computes the Ulcer Index of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> UlcerIndex()
UlcerIndex
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┴ nothing
```

On the path of [`AverageDrawdown`](@ref), the drawdown series is `[0, -0.25, -0.1875, -0.3125, 0]`, so the sum of the squares is 0.1953125.

```jldoctest
julia> x = [0.125, -0.25, 0.0625, -0.125, 0.375];

julia> UlcerIndex()(x) ≈ sqrt(0.1953125 / 5)
true

julia> UlcerIndex(; w = pweights([1.0, 1.0, 1.0, 1.0, 4.0]))(x)
0.15625
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`AverageDrawdown`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`RelativeUlcerIndex`](@ref)
  - [`ulcer_index`](@ref)

# References

  - $(ref_dict[:ulcer])
  - $(ref_dict[:cajas2025]) Section 7.2.4.2, Equation 7.88. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct UlcerIndex <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    @pprop w
    function UlcerIndex(settings::RiskMeasureSettings, w::Option{<:ObsWeights})
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function UlcerIndex(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                    w::Option{<:ObsWeights} = nothing)::UlcerIndex
    return UlcerIndex(settings, w)
end
"""
    ulcer_index(dd::VecNum, ::Nothing) -> Number
    ulcer_index(dd::VecNum, w::VecNum) -> Number

Aggregate a drawdown series into its root-mean-square drawdown.

[`UlcerIndex`](@ref) and [`RelativeUlcerIndex`](@ref) both call this kernel. They differ only in the series that they give it: [`absolute_drawdown_vec`](@ref) for the first, and [`relative_drawdown_vec`](@ref) for the second. [`average_drawdown`](@ref) is the kernel of the mean drawdown.

The type of the second argument selects the method. Callers resolve the observation weights with [`checked_observation_weights`](@ref) first, so the kernel sees a weight vector or `nothing`, and never a [`DynamicAbstractWeights`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{UI}(\\boldsymbol{\\delta}) &= \\frac{\\lVert \\boldsymbol{\\delta} \\rVert_2}{\\sqrt{T}}\\,, \\\\
\\mathrm{UI}_{w}(\\boldsymbol{\\delta}) &= \\sqrt{\\frac{1}{W_{T}} \\sum_{t=1}^{T} w_{t} \\delta_t^2}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\delta}``: Drawdown series `dd` ``T \\times 1``, absolute or relative, every entry ``\\leq 0``.
  - ``\\mathrm{UI}(\\boldsymbol{\\delta})``, ``\\mathrm{UI}_{w}(\\boldsymbol{\\delta})``: The root-mean-square drawdown of the methods without and with weights.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:W_T_total])
  - $(math_dict[:T])

# Arguments

  - `dd::VecNum`: Drawdown series, every entry ``\\leq 0``. The function does not write it.
  - `w`: Resolved observation weights, or `nothing` for the index without weights.

# Returns

  - `Number`: Ulcer index, returned as a nonnegative loss.

# Related

  - [`UlcerIndex`](@ref)
  - [`RelativeUlcerIndex`](@ref)
  - [`average_drawdown`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`checked_observation_weights`](@ref)
"""
function ulcer_index(dd::VecNum, ::Nothing)
    return LinearAlgebra.norm(dd, 2) / sqrt(length(dd))
end
function ulcer_index(dd::VecNum, w::VecNum)
    return sqrt(sum(t -> w[t] * dd[t]^2, axes(dd, 1)) / sum(w))
end
function (r::UlcerIndex)(x::VecNum)
    return ulcer_index(absolute_drawdown_vec(x), checked_observation_weights(r.w, x))
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the Relative Ulcer Index, the root mean square of the compounded drawdown over the sample.

The drawdown series is the one of [`relative_drawdown_vec`](@ref): the wealth compounds the returns, and its running peak starts at ``C_0 = 1``. Each drawdown is thus a fraction of the peak wealth. [`UlcerIndex`](@ref) takes the same root mean square of the uncompounded series. The compounded wealth is a product of linear functions of the weights, so the library states no JuMP formulation of the measure, and it is a [`HierarchicalRiskMeasure`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{RUI}(\\boldsymbol{x}) &= \\frac{\\lVert \\boldsymbol{rd} \\rVert_2}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RUI}(\\boldsymbol{x})``: Relative Ulcer Index (root-mean-square relative drawdown).
  - $(math_dict[:xret])
  - $(math_dict[:rdt]) [`relative_drawdown_vec`](@ref) states the series.
  - ``\\boldsymbol{rd}``: Relative drawdown series vector ``T \\times 1``.
  - $(math_dict[:T])

With observation weights, the index takes the weighted mean of the squares, as [`UlcerIndex`](@ref) states, with ``rd_t`` in place of ``d_t``.

Equation 7.88 of [cajas2025](@cite) states the index on the uncompounded drawdown. The root mean square of the relative drawdown series is the library's extension of that form.

# Algorithm

 1. Compute the relative drawdown series with [`relative_drawdown_vec`](@ref).
 2. Resolve the observation weights with [`checked_observation_weights`](@ref).
 3. Return [`ulcer_index`](@ref) of the series and the weights.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RelativeUlcerIndex(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing
    ) -> RelativeUlcerIndex

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow_rm])

# Functor

    (r::RelativeUlcerIndex)(x::VecNum)

Computes the Relative Ulcer Index of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> RelativeUlcerIndex()
RelativeUlcerIndex
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
         w ┴ nothing
```

On the path of [`AverageDrawdown`](@ref), the relative drawdown series is `[0, -0.25, -0.203125, -0.302734375, -0.041259765625]`.

```jldoctest
julia> x = [0.125, -0.25, 0.0625, -0.125, 0.375];

julia> rd = [0, -0.25, -0.203125, -0.302734375, -0.041259765625];

julia> RelativeUlcerIndex()(x) ≈ sqrt(sum(abs2, rd) / 5)
true
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`UlcerIndex`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
  - [`ulcer_index`](@ref)

# References

  - $(ref_dict[:ulcer])
  - $(ref_dict[:cajas2025]) Section 7.2.4.2, Equation 7.88. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct RelativeUlcerIndex <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:oow])
    """
    @pprop w
    function RelativeUlcerIndex(settings::HierarchicalRiskMeasureSettings,
                                w::Option{<:ObsWeights})
        assert_observation_weights(w, :w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function RelativeUlcerIndex(;
                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                            w::Option{<:ObsWeights} = nothing)::RelativeUlcerIndex
    return RelativeUlcerIndex(settings, w)
end
function (r::RelativeUlcerIndex)(x::VecNum)
    return ulcer_index(relative_drawdown_vec(x), checked_observation_weights(r.w, x))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::UlcerIndex) = NetReturnsInput()
risk_input_kind(::RelativeUlcerIndex) = NetReturnsInput()

export UlcerIndex, RelativeUlcerIndex
