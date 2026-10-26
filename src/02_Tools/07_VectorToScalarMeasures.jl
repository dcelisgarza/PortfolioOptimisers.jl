"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for algorithms that reduce a vector of real values to one real value.

A `VectorToScalarMeasure` reduces a vector of real numbers to a scalar, for example its minimum, mean, median or maximum. The constraint generation and the centrality constraints use these measures to reduce a metric of each asset to one value.

# Interfaces

To implement a new vector-to-scalar measure, subtype `VectorToScalarMeasure` and implement this method:

## Reduction method

  - `vec_to_real_measure(measure::VectorToScalarMeasure, val::VecNum) -> Number`: Reduces `val` to a single scalar.

### Arguments

  - `measure`: An instance of the concrete subtype.
  - `val`: Vector of real values to reduce.

### Returns

  - `score::Number`: The scalar result.

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
    const Num_VecToScaM = Union{Number, VectorToScalarMeasure, Function}

Union of a number, a `VectorToScalarMeasure` and a `Function`.

A function or a field with this type accepts all three. A caller can give a fixed number, an object that implements the `VectorToScalarMeasure` interface, or a plain reduction function. [`vec_to_real_measure`](@ref) returns a `Number` with no change, sends a `VectorToScalarMeasure` to its reduction, and applies a `Function` to the vector.

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`vec_to_real_measure`](@ref)
"""
const Num_VecToScaM = Union{Number, VectorToScalarMeasure, Function}
"""
$(DocStringExtensions.TYPEDEF)

Algorithm that reduces a vector of real values to its minimum.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MinValue}(\\boldsymbol{v}) &= \\underset{i}{\\min}\\ v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction reads no weights. A call of [`factory`](@ref) with observation weights returns it with no change.

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

Algorithm that reduces a vector of real values to its mean, with or without weights.

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

The weighted form divides by the total weight, so a weight vector multiplied by a positive constant gives the same value. `w` must have one entry for each entry of ``\\boldsymbol{v}``.

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

[`factory`](@ref) passes its arguments on to this `@wprop` field:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

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

Algorithm that reduces a vector of real values to its median, with or without weights.

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

**Both forms are quantiles, and both interpolate.** Neither is an order statistic, so the result is not always an entry of ``\\boldsymbol{v}``. On a vector of even length, the unweighted form takes the average of the two middle entries. The weighted form interpolates between the two entries on each side of half the total weight.

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

[`factory`](@ref) passes its arguments on to this `@wprop` field:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

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

Algorithm that reduces a vector of real values to its maximum.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MaxValue}(\\boldsymbol{v}) &= \\underset{i}{\\max}\\ v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction reads no weights. A call of [`factory`](@ref) with observation weights returns it with no change.

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

Algorithm that reduces a vector of real values to its standard deviation, with or without weights. The default `corrected = true` works without weights. With a plain `StatsBase.Weights`, it raises an `ArgumentError`, because that type declares no bias correction. Pass an `AnalyticWeights`, a `FrequencyWeights` or a `ProbabilityWeights`, or set `corrected = false`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{StdValue}(\\boldsymbol{v}) &= \\sqrt{\\mathrm{VarValue}(\\boldsymbol{v})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``\\mathrm{VarValue}(\\boldsymbol{v})``: The variance under the same `w` and the same `corrected`, whose four denominators [`VarValue`](@ref) states.

`corrected` selects the denominator of the variance, and the square root keeps that choice. Without weights, the default `corrected = true` divides by ``n - 1``.

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

[`factory`](@ref) passes its arguments on to this `@wprop` field:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> PortfolioOptimisers.vec_to_real_measure(StdValue(), [1.2, 3.4, 0.7])
1.4364307617610164
```

# Related

  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`VarValue`](@ref): the four denominators that `corrected` and the type of `w` select.
  - [`StandardisedValue`](@ref): calls `Statistics.std` with a `mean` keyword, so the deviation is about the mean that its `mv` gives.
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

Algorithm that reduces a vector of real values to its variance, with or without weights. A plain `StatsBase.Weights` declares no bias correction, so the default `corrected = true` with it raises an `ArgumentError` and returns no value. Pass one of the three weight types below that have a correction, or set `corrected = false`.

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

With `corrected = false`, every weighted case takes ``d = \\sum w_{i}``.

The **type** of ``\\boldsymbol{w}`` selects ``d``, not its values. So two weight vectors with the same values and different types give different variances.

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

[`factory`](@ref) passes its arguments on to this `@wprop` field:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

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

Algorithm that reduces a vector of real values to its sum.

# Mathematical definition

```math
\\begin{align}
\\mathrm{SumValue}(\\boldsymbol{v}) &= \\sum_{i=1}^{n} v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction reads no weights. The weighted form of [`MeanValue`](@ref) is the weighted sum divided by the total weight, and a weighted sum is that value multiplied by the total weight.

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

Algorithm that reduces a vector of real values to its product.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ProdValue}(\\boldsymbol{v}) &= \\prod_{i=1}^{n} v_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - $(math_dict[:v_i_entry])

The reduction reads no weights. One zero entry gives zero. The product of many entries below one underflows to zero, and the product of many entries above one overflows. So this reduction is for a short vector of values near one.

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

Algorithm that reduces a vector of real values to its mode.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ModeValue}(\\boldsymbol{v}) &= \\underset{u \\in \\boldsymbol{v}}{\\arg\\max}\\ \\left| \\left\\{ i : v_{i} = u \\right\\} \\right|\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_reduce])
  - ``u``: A value of ``\\boldsymbol{v}``.
  - ``\\left| \\cdot \\right|``: The number of elements of a set.

When two values have the highest count, `StatsBase.mode` returns the **first** value that reaches that count. The result is a value of the input and never an average of two values. For example, `[3.0, 1.0, 1.0, 3.0]` gives `1.0`. The comparison is exact equality, so this reduction is for a vector of repeated exact values, not for continuous values.

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

Algorithm that reduces a vector of real values to its mean divided by its standard deviation, with or without weights. After a call of [`factory`](@ref) with weights, the reduction can raise an error. `factory` replaces the `w` field of both `mv` and `sv` with the incoming [`ObsWeights`](@ref), and `sv` keeps its default `corrected = true`. With a plain `StatsBase.Weights`, that default raises an `ArgumentError`. Thread an `AnalyticWeights`, a `FrequencyWeights` or a `ProbabilityWeights`, or declare `sv = StdValue(; corrected = false)`.

# Mathematical definition

```math
\\begin{align}
z &= \\frac{\\hat{\\mu}}{\\tilde{\\sigma}}\\,, \\\\
\\tilde{\\sigma} &= \\begin{cases} 1 & \\hat{\\sigma} \\ \\mathrm{undefined} \\\\ \\sqrt{\\varepsilon} & \\hat{\\sigma} = 0 \\\\ \\hat{\\sigma} & \\mathrm{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``z``: Standardised value.
  - ``\\hat{\\mu}``: The value that `mv` gives.
  - ``\\hat{\\sigma}``: The value that `sv` gives, about ``\\hat{\\mu}``.
  - ``\\tilde{\\sigma}``: The denominator after the guard.
  - ``\\varepsilon``: Machine epsilon of the element type of ``\\hat{\\sigma}``.

``\\hat{\\sigma}`` is undefined on a vector of one entry, because a corrected standard deviation needs two entries. The first case then gives ``\\tilde{\\sigma} = 1`` and ``z = \\hat{\\mu}``, so the reduction has a value on every non-empty vector.

# Algorithm

 1. Reduce `val` with `mv`, which gives `m`.
 2. Reduce `val` with `sv`, and pass `m` as the `mean` keyword, which gives `s`. So the deviation is always about the mean of step 1. Weights on `mv` and no weights on `sv` also change the denominator.
 3. Guard `s`:
     1. `s` is `NaN`: replace it with `one(s)`.
     2. `s` is an exact zero: replace it with `sqrt(eps(eltype(s)))`. On the constant vector `[2.0, 2.0, 2.0]`, `s` is an exact zero, and the result is `1.342e8`, which is `2 / sqrt(eps(Float64))`. The test is an equality, so the step keeps a small positive `s`, and the result can be larger than this.
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

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `mv`: Updated through a recursive call of [`factory`](@ref).
  - `sv`: Updated through a recursive call of [`factory`](@ref).

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
    The mean measure of the numerator.
    """
    @fprop mv
    """
    The standard deviation measure of the denominator.
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
        val::Union{VecNum, NTuple{N, <:Number} where {N}};
        kwargs...
    ) -> Number

Reduce a vector of real values to one real value with a measure.

`vec_to_real_measure` applies the reduction of the concrete subtype of [`VectorToScalarMeasure`](@ref), such as the minimum, the mean, the median or the maximum, to a vector of real numbers. The constraint generation and the centrality constraints use it to reduce a metric of each asset to one value.

# Algorithm

The method that Julia selects is the algorithm. `measure` names the reduction. The type parameter of a weighted measure names the branch, for example `MeanValue{Nothing}` or `MeanValue{<:ObsWeights}`. So the compiler selects the branch, and no code tests the field at run time.

 1. `measure` is a `Number`: return it, and read nothing of `val`.
 2. `measure` is a `Function`: return `measure(val)`.
 3. `measure` is a [`MinValue`](@ref), a [`MaxValue`](@ref), a [`SumValue`](@ref) or a [`ProdValue`](@ref): return `minimum`, `maximum`, `sum` or `prod` of `val`.
 4. `measure` is a [`ModeValue`](@ref): return `StatsBase.mode` of `val`.
 5. `measure` is a [`MeanValue`](@ref) or a [`MedianValue`](@ref): return `Statistics.mean` or `Statistics.median` of `val`, with weights when the measure holds them. On the weighted branch, `collect` a tuple first, and resolve the weights against `val` with [`get_observation_weights`](@ref). That call returns a `StatsBase.AbstractWeights` itself, and it evaluates a [`DynamicAbstractWeights`](@ref) on `val`.
 6. `measure` is a [`StdValue`](@ref) or a [`VarValue`](@ref): return `Statistics.std` or `Statistics.var` of `val`, with `corrected = measure.corrected`, with weights when the measure holds them, and with `kwargs...`. The weighted branch resolves its weights as step 5 does.
 7. `measure` is a [`StandardisedValue`](@ref): follow the algorithm of that type, which reduces two times and guards the denominator.

Because of step 1, a plain number is a valid `measure`. A caller that already has the value writes it where a reduction goes, and `vec_to_real_measure` needs no second signature.

# Arguments

  - `measure`: One of three values.

      + `::VectorToScalarMeasure`: The reduction to apply to `val`.
      + `::Number`: The value to return, whatever `val` holds.
      + `::Function`: The function to apply to `val`, as `measure(val)`.

  - `val`: A vector or a tuple of real values to reduce. Every reduction accepts a tuple, and the weighted reductions `collect` it first, because `Statistics` needs an `AbstractVector` with its weights. The function ignores `val` when `measure` is a `Number`.

  - `kwargs...`: Keyword arguments for the reduction. Only the [`StdValue`](@ref) and [`VarValue`](@ref) reductions read them.

# Returns

  - `score::Number`: The value that `measure` gives.

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
  - [`get_observation_weights`](@ref)
"""
function vec_to_real_measure(::MinValue, val::Union{VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return minimum(val)
end
function vec_to_real_measure(mv::MeanValue{Nothing},
                             val::Union{VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.mean(val)
end
function vec_to_real_measure(mv::MeanValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.mean(val, get_observation_weights(mv.w, val))
end
function vec_to_real_measure(mv::MeanValue{<:ObsWeights},
                             val::NTuple{N, <:Number} where {N}; kwargs...)
    return vec_to_real_measure(mv, collect(val))
end
function vec_to_real_measure(mdv::MedianValue{Nothing},
                             val::Union{VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.median(val)
end
function vec_to_real_measure(mdv::MedianValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.median(val, get_observation_weights(mdv.w, val))
end
function vec_to_real_measure(mdv::MedianValue{<:ObsWeights},
                             val::NTuple{N, <:Number} where {N}; kwargs...)
    return vec_to_real_measure(mdv, collect(val))
end
function vec_to_real_measure(::MaxValue, val::Union{VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return maximum(val)
end
function vec_to_real_measure(val::Number, ::Union{VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return val
end
function vec_to_real_measure(sv::StdValue{Nothing},
                             val::Union{VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.std(val; corrected = sv.corrected, kwargs...)
end
function vec_to_real_measure(sv::StdValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.std(val, get_observation_weights(sv.w, val); corrected = sv.corrected,
                          kwargs...)
end
function vec_to_real_measure(sv::StdValue{<:ObsWeights}, val::NTuple{N, <:Number} where {N};
                             kwargs...)
    return vec_to_real_measure(sv, collect(val); kwargs...)
end
function vec_to_real_measure(vv::VarValue{Nothing},
                             val::Union{VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
    return Statistics.var(val; corrected = vv.corrected, kwargs...)
end
function vec_to_real_measure(vv::VarValue{<:ObsWeights}, val::VecNum; kwargs...)
    return Statistics.var(val, get_observation_weights(vv.w, val); corrected = vv.corrected,
                          kwargs...)
end
function vec_to_real_measure(vv::VarValue{<:ObsWeights}, val::NTuple{N, <:Number} where {N};
                             kwargs...)
    return vec_to_real_measure(vv, collect(val); kwargs...)
end
function vec_to_real_measure(msv::StandardisedValue,
                             val::Union{VecNum, NTuple{N, <:Number} where {N}}; kwargs...)
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
function vec_to_real_measure(::SumValue, val::Union{VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return sum(val)
end
function vec_to_real_measure(::ProdValue, val::Union{VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return prod(val)
end
function vec_to_real_measure(::ModeValue, val::Union{VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return StatsBase.mode(val)
end
function vec_to_real_measure(f::Function, val::Union{VecNum, NTuple{N, <:Number} where {N}};
                             kwargs...)
    return f(val)
end

export MinValue, MeanValue, MedianValue, MaxValue, StandardisedValue, StdValue, VarValue,
       SumValue, ProdValue, ModeValue
public VectorToScalarMeasure, vec_to_real_measure
