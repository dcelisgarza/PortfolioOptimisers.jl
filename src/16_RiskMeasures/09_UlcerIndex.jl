"""
$(DocStringExtensions.TYPEDEF)

Represents the Ulcer Index risk measure.

`UlcerIndex` measures the depth and duration of drawdowns in portfolio returns. It is defined as the root-mean-square of the absolute drawdown series, capturing both the magnitude and persistence of losses.

# Mathematical definition

Define the absolute drawdown series:

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} x_s\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])

The Ulcer Index is:

```math
\\begin{align}
\\mathrm{UI}(\\boldsymbol{x}) &= \\sqrt{\\frac{1}{T} \\sum_{t=1}^{T} d_t^2} = \\frac{\\lVert \\boldsymbol{d} \\rVert_2}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{UI}(\\boldsymbol{x})``: Ulcer Index (root-mean-square drawdown).
  - $(math_dict[:T])
  - $(math_dict[:dtdd])
  - ``\\boldsymbol{d}``: Absolute drawdown series vector ``T \\times 1``.

For observation-weighted samples, the weighted mean of the squared drawdowns is used instead:

```math
\\begin{align}
\\mathrm{UI}(\\boldsymbol{x}) &= \\sqrt{\\frac{1}{W_{T}} \\sum_{t=1}^{T} w_{t} d_t^2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_t_obs])
  - $(math_dict[:W_T_total])

The drawdowns are taken on the full return path, so an observation with zero weight still moves every later drawdown.

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

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`AverageDrawdown`](@ref)
  - [`MaximumDrawdown`](@ref)
  - [`RelativeUlcerIndex`](@ref)
  - [`ulcer_index`](@ref)

# References

  - $(ref_dict[:ulcer])
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

This is the shared aggregation kernel behind [`UlcerIndex`](@ref) and [`RelativeUlcerIndex`](@ref). The two measures differ only in the drawdown series they feed it, [`absolute_drawdown_vec`](@ref) and [`relative_drawdown_vec`](@ref), so the aggregation lives here once, as [`average_drawdown`](@ref) does for the average drawdown.

Dispatch on the second argument selects the weighting scheme. Callers resolve observation weights with [`checked_observation_weights`](@ref) first, so the kernel only sees a concrete weight vector or `nothing`.

  - `::Nothing`: `norm(dd, 2) / sqrt(length(dd))`.
  - `w::VecNum`: `sqrt(sum(w .* dd .^ 2) / sum(w))`.

# Arguments

  - `dd::VecNum`: Drawdown series, all entries ≤ 0. Not modified.
  - `w`: Resolved observation weights, or `nothing` for the unweighted index.

# Returns

  - `Number`: Ulcer index, returned as a nonnegative loss.

# Related

  - [`UlcerIndex`](@ref)
  - [`RelativeUlcerIndex`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
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

Represents the Relative Ulcer Index risk measure for hierarchical optimisation.

`RelativeUlcerIndex` applies the Ulcer Index framework to the relative (compounded) drawdown series.

# Mathematical definition

Define the relative drawdown series:

```math
\\begin{align}
C_t &= \\prod_{s=1}^{t} (1 + x_s)\\,, \\\\
rd_t &= \\frac{C_t}{\\max_{0 \\leq s \\leq t} C_s} - 1 \\leq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:Ct])
  - $(math_dict[:rdt])

The Relative Ulcer Index is:

```math
\\begin{align}
\\mathrm{RUI}(\\boldsymbol{x}) &= \\frac{\\lVert \\boldsymbol{rd} \\rVert_2}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RUI}(\\boldsymbol{x})``: Relative Ulcer Index (root-mean-square relative drawdown).
  - $(math_dict[:T])
  - $(math_dict[:rdt])
  - ``\\boldsymbol{rd}``: Relative drawdown series vector ``T \\times 1``.

For observation-weighted samples, the weighted mean of the squared relative drawdowns is used instead, as [`UlcerIndex`](@ref) states.

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

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`UlcerIndex`](@ref)
  - [`RelativeAverageDrawdown`](@ref)
  - [`ulcer_index`](@ref)

# References

  - $(ref_dict[:ulcer])
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
