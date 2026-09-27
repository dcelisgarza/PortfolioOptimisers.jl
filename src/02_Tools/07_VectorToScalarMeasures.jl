"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for algorithms mapping a vector of real values to a single real value.

`VectorToScalarMeasure` provides a unified interface for algorithms that reduce a vector of real numbers to a scalar, such as minimum, mean, median, or maximum. These are used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Interfaces

In order to implement a new vector-to-scalar measure that works seamlessly with the library, subtype `VectorToScalarMeasure` and implement the following method:

## Reduction method

  - `vec_to_real_measure(measure::VectorToScalarMeasure, val::VecNum) -> Number`: Reduces `val` to a single scalar.

### Arguments

  - `measure`: Concrete subtype instance.
  - `val`: Vector of real values to reduce.

### Returns

  - `score::Number`: Computed scalar.

# Related

  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`StdValue`](@ref)
  - [`VarValue`](@ref)
  - [`SumValue`](@ref)
  - [`ProdValue`](@ref)
  - [`ModeValue`](@ref)
  - [`StandardisedValue`](@ref)
  - [`Num_VecToScaM`](@ref)
  - [`CentralityConstraint`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
abstract type VectorToScalarMeasure <: AbstractAlgorithm end
"""
    const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure, <:Function}

Union type representing a numeric value, a `VectorToScalarMeasure`, or a `Function`.

This type lets functions and fields accept all three, so a caller can give a fixed number, an object that implements the `VectorToScalarMeasure` interface, or a plain reduction function. [`vec_to_real_measure`](@ref) returns a `Number` unchanged, dispatches a `VectorToScalarMeasure` to its reduction, and applies a `Function` to the vector.

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
const Num_VecToScaM = Union{<:Number, <:VectorToScalarMeasure, <:Function}
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its minimum.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MinValue}(\\boldsymbol{v}) &= \\underset{i}{\\min}\\ v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights, so a weighted call gives the same value as an unweighted one.

# Constructors

    MinValue() -> MinValue

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MinValue(), [1.2, 3.4, 0.7])
0.7
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MinValue <: VectorToScalarMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted mean.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MeanValue}(\\boldsymbol{v}) &= \\frac{1}{n} \\sum_{i=1}^{n} v_{i}\\,,
&&w = \\mathrm{nothing}\\,, \\\\
\\mathrm{MeanValue}(\\boldsymbol{v}) &= \\frac{\\sum_{i=1}^{n} w_{i} v_{i}}{\\sum_{i=1}^{n} w_{i}}\\,,
&&\\mathrm{otherwise}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])
  - ``w_{i}``: The ``i``-th observation weight, from the field `w`.

The weighted form normalises by the total weight, so a weight vector scaled by a positive constant gives the same value. `w` must carry one entry per entry of ``\\boldsymbol{v}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanValue(;
        w::Option{<:ObsWeights} = nothing,
    ) -> MeanValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MeanValue(), [1.2, 3.4, 0.7])
1.7666666666666666
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MedianValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct MeanValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    function MeanValue(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function MeanValue(; w::Option{<:ObsWeights} = nothing)
    return MeanValue(w)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted median.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MedianValue}(\\boldsymbol{v}) &= Q_{\\boldsymbol{v}}(0.5)\\,, \\\\
\\mathrm{MedianValue}(\\boldsymbol{v}) &= Q_{\\boldsymbol{v}, \\boldsymbol{w}}(0.5)\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``\\boldsymbol{w}``: The observation weights, from the field `w`. The first line is the case `w = nothing`.
  - ``Q_{\\boldsymbol{v}}(p)``: The ``p``-quantile of ``\\boldsymbol{v}``.
  - ``Q_{\\boldsymbol{v}, \\boldsymbol{w}}(p)``: The weighted ``p``-quantile of ``\\boldsymbol{v}``, as `StatsBase` defines it.

**Both forms are quantiles, and both interpolate.** Neither is an order statistic, so the result need not be an entry of ``\\boldsymbol{v}``. On a vector of even length the unweighted form averages the two middle entries, and the weighted form interpolates between the two entries that bracket half the weight mass.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MedianValue(;
        w::Option{<:ObsWeights} = nothing,
    ) -> MedianValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MedianValue(), [1.2, 3.4, 0.7])
1.2
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MaxValue`](@ref)
  - [`vec_to_real_measure`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct MedianValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    function MedianValue(w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function MedianValue(; w::Option{<:ObsWeights} = nothing)
    return MedianValue(w)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its maximum.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MaxValue}(\\boldsymbol{v}) &= \\underset{i}{\\max}\\ v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights, so a weighted call gives the same value as an unweighted one.

# Constructors

    MaxValue() -> MaxValue

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MaxValue(), [1.2, 3.4, 0.7])
3.4
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MinValue`](@ref)
  - [`MeanValue`](@ref)
  - [`MedianValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct MaxValue <: VectorToScalarMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted standard deviation. The unweighted default is safe and the weighted default is not: `corrected = true` under a plain `StatsBase.Weights` raises an `ArgumentError`, because that type declares no bias correction. Pass an `AnalyticWeights`, a `FrequencyWeights` or a `ProbabilityWeights`, or set `corrected = false`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{StdValue}(\\boldsymbol{v}) &= \\sqrt{\\mathrm{VarValue}(\\boldsymbol{v})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``\\mathrm{VarValue}(\\boldsymbol{v})``: The variance under the same `w` and the same `corrected`, whose four denominators [`VarValue`](@ref) states.

`corrected` selects the denominator of the variance, and the square root carries that choice through. The unweighted default `corrected = true` divides by ``n - 1``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StdValue(;
        w::Option{<:ObsWeights} = nothing,
        corrected::Bool = true,
    ) -> StdValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(StdValue(), [1.2, 3.4, 0.7])
1.4364307617610164
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`VarValue`](@ref): the four denominators that `corrected` and the type of `w` select.
  - [`StandardisedValue`](@ref): reaches `Statistics.std` with a `mean` keyword, which is how it makes the deviation be taken about the mean that its `mv` produced.
  - [`vec_to_real_measure`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct StdValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:corrected])
    """
    corrected
    function StdValue(w::Option{<:ObsWeights}, corrected::Bool)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(corrected)}(w, corrected)
    end
end
function StdValue(; w::Option{<:ObsWeights} = nothing, corrected::Bool = true)
    return StdValue(w, corrected)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted variance. The weighted default raises: a plain `StatsBase.Weights` declares no bias correction, so `corrected = true` under it raises an `ArgumentError` rather than returning a value. Pass one of the three corrected weight types below, or set `corrected = false`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{VarValue}(\\boldsymbol{v}) &= \\frac{1}{d} \\sum_{i=1}^{n} w_{i} \\left(v_{i} - \\bar{v}\\right)^{2}\\,, \\\\
\\bar{v} &= \\frac{\\sum_{i=1}^{n} w_{i} v_{i}}{\\sum_{i=1}^{n} w_{i}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``w_{i}``: The ``i``-th observation weight. The unweighted case is ``w_{i} = 1``.
  - ``\\bar{v}``: The mean of ``\\boldsymbol{v}`` under those weights.
  - ``d``: The denominator, which `corrected` and the **type** of `w` together select.

``d`` takes one of four values:

  - `w = nothing`: ``d = n - 1`` when `corrected` is `true`, and ``d = n`` when it is `false`.
  - `w::AnalyticWeights`: ``d = \\sum w_{i} - \\sum w_{i}^{2} / \\sum w_{i}`` when `corrected` is `true`.
  - `w::FrequencyWeights`: ``d = \\sum w_{i} - 1`` when `corrected` is `true`.
  - `w::ProbabilityWeights`: ``d = \\left(\\sum w_{i}\\right)(m - 1) / m`` when `corrected` is `true`, where ``m`` is the count of non-zero weights.

With `corrected = false` every weighted case takes ``d = \\sum w_{i}``.

``d`` is selected by the **type** of ``\\boldsymbol{w}`` and not by its values, so two numerically identical weight vectors of different types give different variances.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarValue(;
        w::Option{<:ObsWeights} = nothing,
        corrected::Bool = true,
    ) -> VarValue

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(VarValue(), [1.2, 3.4, 0.7])
2.0633333333333335
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`StdValue`](@ref)
  - [`StandardisedValue`](@ref)
  - [`vec_to_real_measure`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct VarValue <: VectorToScalarMeasure
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:corrected])
    """
    corrected
    function VarValue(w::Option{<:ObsWeights}, corrected::Bool)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(corrected)}(w, corrected)
    end
end
function VarValue(; w::Option{<:ObsWeights} = nothing, corrected::Bool = true)
    return VarValue(w, corrected)
end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its sum.

# Mathematical definition

```math
\\begin{align}
\\mathrm{SumValue}(\\boldsymbol{v}) &= \\sum_{i=1}^{n} v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights. [`MeanValue`](@ref) is the weighted sum normalised by the total weight, so a weighted sum is that value multiplied by the total weight.

# Constructors

    SumValue() -> SumValue

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(SumValue(), [1.2, 3.4, 0.7])
5.3
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`ProdValue`](@ref)
  - [`ModeValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct SumValue <: VectorToScalarMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its product.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ProdValue}(\\boldsymbol{v}) &= \\prod_{i=1}^{n} v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction carries no weights. One zero entry gives zero, and the product of many entries below one underflows, so this reduction is for a short vector of values near one.

# Constructors

    ProdValue() -> ProdValue

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(ProdValue(), [1.2, 3.4, 0.7])
2.856
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`SumValue`](@ref)
  - [`ModeValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct ProdValue <: VectorToScalarMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its mode.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ModeValue}(\\boldsymbol{v}) &= \\underset{u \\in \\boldsymbol{v}}{\\arg\\max}\\ \\left| \\left\\{ i : v_{i} = u \\right\\} \\right|\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``u``: A value that ``\\boldsymbol{v}`` carries.
  - ``\\left| \\cdot \\right|``: The count of a set.

`StatsBase.mode` breaks a tie by the **first** value that reaches the highest count, so the result is a value of the input and never an average of two. The comparison is exact equality, so this reduction is for a vector of repeated exact values and not for a continuous one.

# Constructors

    ModeValue() -> ModeValue

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(ModeValue(), [1.2, 3.4, 0.7, 1.2])
1.2
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`SumValue`](@ref)
  - [`ProdValue`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
struct ModeValue <: VectorToScalarMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Algorithm for reducing a vector of real values to its optionally weighted mean divided by its optionally weighted standard deviation. A weighted [`factory`](@ref) call can make the reduction raise: `factory` replaces the `w` field of both `mv` and `sv` with the incoming [`ObsWeights`](@ref), and `sv` keeps its default `corrected = true`, which raises an `ArgumentError` under a plain `StatsBase.Weights`. Thread an `AnalyticWeights`, a `FrequencyWeights` or a `ProbabilityWeights`, or declare `sv = StdValue(; corrected = false)`.

# Mathematical definition

```math
\\begin{align}
z &= \\frac{\\hat{\\mu}}{\\tilde{\\sigma}}\\,, \\\\
\\tilde{\\sigma} &= \\begin{cases} 1 & \\hat{\\sigma} \\ \\mathrm{undefined} \\\\ \\sqrt{\\varepsilon} & \\hat{\\sigma} = 0 \\\\ \\hat{\\sigma} & \\mathrm{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``z``: Standardised value.
  - ``\\hat{\\mu}``: The value computed by `mv`.
  - ``\\hat{\\sigma}``: The value computed by `sv`, taken about ``\\hat{\\mu}``.
  - ``\\tilde{\\sigma}``: The guarded denominator.
  - ``\\varepsilon``: Machine epsilon of the element type of ``\\hat{\\sigma}``.

``\\hat{\\sigma}`` is undefined on a vector of one entry, because a corrected standard deviation needs two. The first case then gives ``\\tilde{\\sigma} = 1`` and ``z = \\hat{\\mu}``, so the reduction is defined on every non-empty vector.

# Algorithm

 1. Reduce `val` with `mv`, giving `m`.
 2. Reduce `val` with `sv`, and pass `m` as the `mean` keyword, giving `s`. The deviation is therefore always taken about the mean that step 1 produced, so weighting `mv` without weighting `sv` changes the denominator too.
 3. Guard `s`:
     1. `s` is `NaN`: replace it with `one(s)`.
     2. `s` is an exact zero: replace it with `sqrt(eps(eltype(s)))`. The test is an equality, so a small `s` is not guarded: on the constant vector `[2.0, 2.0, 2.0]` the result is `1.342e8`, which is `2 / sqrt(eps(Float64))`.
     3. Otherwise: keep `s`.
 4. Return `m / s`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardisedValue(;
        mv::MeanValue = MeanValue(),
        sv::StdValue = StdValue(),
    ) -> StandardisedValue

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `mv`: Recursively updated via [`factory`](@ref).
  - `sv`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), [1.2, 3.4, 0.7])
1.2299003291330186

julia> PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), [0.37])
0.37
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`StdValue`](@ref)
  - [`VarValue`](@ref)
  - [`vec_to_real_measure`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct StandardisedValue <: VectorToScalarMeasure
    """
    The mean value measure used for the numerator.
    """
    @fprop mv
    """
    The standard deviation measure used for the denominator.
    """
    @fprop sv
    function StandardisedValue(mv::MeanValue, sv::StdValue)
        return new{typeof(mv), typeof(sv)}(mv, sv)
    end
end
function StandardisedValue(; mv::MeanValue = MeanValue(), sv::StdValue = StdValue())
    return StandardisedValue(mv, sv)
end
"""
    vec_to_real_measure(
        measure::Num_VecToScaM,
        val::Union{<:VecNum, NTuple{N, <:Number} where {N}};
        kwargs...
    ) -> Number

Reduce a vector of real values to a single real value using a specified measure.

`vec_to_real_measure` applies a reduction algorithm (such as minimum, mean, median, or maximum) to a vector of real numbers, as specified by the concrete subtype of [`VectorToScalarMeasure`](@ref). This is used in constraint generation and centrality-based portfolio constraints to aggregate asset-level metrics.

# Algorithm

The method that Julia selects is the algorithm. `measure` names the reduction, and the type parameter of a weighted measure names the branch, `MeanValue{Nothing}` against `MeanValue{<:ObsWeights}`, so the branch is chosen at compile time and the field is never tested at run time.

 1. `measure` is a `Number`: return it, and read nothing of `val`.
 2. `measure` is a `Function`: return `measure(val)`.
 3. `measure` is a [`MinValue`](@ref), a [`MaxValue`](@ref), a [`SumValue`](@ref) or a [`ProdValue`](@ref): return `minimum`, `maximum`, `sum` or `prod` of `val`.
 4. `measure` is a [`ModeValue`](@ref): return `StatsBase.mode` of `val`.
 5. `measure` is a [`MeanValue`](@ref) or a [`MedianValue`](@ref): return `Statistics.mean` or `Statistics.median` of `val`, with the weights `measure.w` when the measure carries them. A tuple is `collect`ed first on the weighted branch.
 6. `measure` is a [`StdValue`](@ref) or a [`VarValue`](@ref): return `Statistics.std` or `Statistics.var` of `val`, with `corrected = measure.corrected`, with the weights `measure.w` when the measure carries them, and with `kwargs...` forwarded. A tuple is `collect`ed first on the weighted branch.
 7. `measure` is a [`StandardisedValue`](@ref): follow that type's own algorithm, which reduces twice and guards the denominator.

Step 1 is the case that makes a plain number a legal `measure`: a caller that already holds the value writes it where a reduction goes, and the seam needs no second signature.

# Arguments

  - `measure`: One of three things.

      + `::VectorToScalarMeasure`: The reduction to apply to `val`.
      + `::Number`: The value to return, whatever `val` holds.
      + `::Function`: Applied to `val` directly, as `measure(val)`.

  - `val`: A vector or tuple of real values to be reduced. A tuple is accepted wherever a vector is, and the weighted reductions `collect` it first, because `Statistics` needs an `AbstractVector` beside its weights. It is ignored when `measure` is a `Number`.

  - `kwargs...`: Forwarded to the underlying reduction. Only the [`StdValue`](@ref) and [`VarValue`](@ref) reductions read them.

# Returns

  - `score::Number`: Computed value according to `measure`.

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(MaxValue(), [1.2, 3.4, 0.7])
3.4

julia> PortfolioOptimisers.vec_to_real_measure(0.9, [1.2, 3.4, 0.7])
0.9
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`Num_VecToScaM`](@ref)
"""
function vec_to_real_measure(::MinValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return minimum(val)
end
function vec_to_real_measure(mv::MeanValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.mean(val)
end
function vec_to_real_measure(mv::MeanValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.mean(val, mv.w)
end
function vec_to_real_measure(mv::MeanValue{<:ObsWeights},
                             val::NTuple{N, <:Number} where {N}; kwargs...)
    return Statistics.mean(collect(val), mv.w)
end
function vec_to_real_measure(mdv::MedianValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.median(val)
end
function vec_to_real_measure(mdv::MedianValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.median(val, mdv.w)
end
function vec_to_real_measure(mdv::MedianValue{<:ObsWeights},
                             val::NTuple{N, <:Number} where {N}; kwargs...)
    return Statistics.median(collect(val), mdv.w)
end
function vec_to_real_measure(::MaxValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return maximum(val)
end
function vec_to_real_measure(val::Number, ::Union{<:VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return val
end
function vec_to_real_measure(sv::StdValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.std(val; corrected = sv.corrected, kwargs...)
end
function vec_to_real_measure(sv::StdValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.std(val, sv.w; corrected = sv.corrected, kwargs...)
end
function vec_to_real_measure(sv::StdValue{<:ObsWeights}, val::NTuple{N, <:Number} where {N};
                             kwargs...)
    return Statistics.std(collect(val), sv.w; corrected = sv.corrected, kwargs...)
end
function vec_to_real_measure(vv::VarValue{Nothing},
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.var(val; corrected = vv.corrected, kwargs...)
end
function vec_to_real_measure(vv::VarValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.var(val, vv.w; corrected = vv.corrected, kwargs...)
end
function vec_to_real_measure(vv::VarValue{<:ObsWeights}, val::NTuple{N, <:Number} where {N};
                             kwargs...)
    return Statistics.var(collect(val), vv.w; corrected = vv.corrected, kwargs...)
end
function vec_to_real_measure(msv::StandardisedValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    m = vec_to_real_measure(msv.mv, val)
    s = vec_to_real_measure(msv.sv, val; mean = m)
    s = if isnan(s)
        one(s)
    elseif iszero(s)
        sqrt(eps(eltype(s)))
    else
        s
    end
    return m / s
end
function vec_to_real_measure(::SumValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return sum(val)
end
function vec_to_real_measure(::ProdValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return prod(val)
end
function vec_to_real_measure(::ModeValue,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return StatsBase.mode(val)
end
function vec_to_real_measure(f::Function,
                             val::Union{<:VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return f(val)
end

export MinValue, MeanValue, MedianValue, MaxValue, StandardisedValue, StdValue, VarValue,
       SumValue, ProdValue, ModeValue
public VectorToScalarMeasure, vec_to_real_measure
