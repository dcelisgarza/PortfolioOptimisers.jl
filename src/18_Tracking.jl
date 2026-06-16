"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all tracking result types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing tracking error or tracking constraint results should be subtypes of `AbstractTracking`.

# Related

  - [`AbstractResult`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
abstract type AbstractTracking <: AbstractResult end
"""
    const VecTr = AbstractVector{<:AbstractTracking}

Alias for a vector of tracking result types.

# Related

  - [`AbstractTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
const VecTr = AbstractVector{<:AbstractTracking}
"""
    const Tr_VecTr = Union{<:AbstractTracking, <:VecTr}

Union type for a single tracking result or a vector of tracking results.

# Related

  - [`AbstractTracking`](@ref)
  - [`VecTr`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
const Tr_VecTr = Union{<:AbstractTracking, <:VecTr}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all tracking algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing tracking algorithms (such as weights or returns tracking) should be subtypes of `AbstractTrackingAlgorithm`.

# Interfaces

In order to implement a new tracking algorithm that works seamlessly with the library, subtype `AbstractTrackingAlgorithm` and implement the following methods:

  - `tracking_benchmark(tr::AbstractTrackingAlgorithm, X::MatNum) -> VecNum`: Compute benchmark returns from the asset return matrix `X`.
  - `factory(tr::AbstractTrackingAlgorithm, w::VecNum) -> AbstractTrackingAlgorithm`: Construct a new instance with updated portfolio weights `w`.
  - `port_opt_view(tr::AbstractTrackingAlgorithm, i) -> AbstractTrackingAlgorithm`: Create a view of the tracking algorithm for the subset of assets at indices `i`.

## Arguments

  - `tr`: The concrete tracking algorithm instance.
  - $(arg_dict[:sigrhoX])
  - `w`: New portfolio weights.
  - `i`: Index or indices for asset subset.

## Returns

  - `b::VecNum`: Benchmark returns (for `tracking_benchmark`).
  - `tr::AbstractTrackingAlgorithm`: Updated or viewed tracking algorithm (for `factory`, `port_opt_view`).

## Examples

```jldoctest
julia> struct MyTracking <: PortfolioOptimisers.AbstractTrackingAlgorithm
           w::Vector{Float64}
       end

julia> function PortfolioOptimisers.tracking_benchmark(tr::MyTracking,
                                                       X::PortfolioOptimisers.MatNum)
           return X * tr.w
       end

julia> PortfolioOptimisers.factory(tr::MyTracking, w) = MyTracking(w)

julia> PortfolioOptimisers.port_opt_view(tr::MyTracking, i) = MyTracking(tr.w[i])

julia> tr = MyTracking([0.5, 0.5]);

julia> X = [0.01 0.02; 0.03 0.04];

julia> PortfolioOptimisers.tracking_benchmark(tr, X)
2-element Vector{Float64}:
 0.015
 0.035
```

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
  - [`TrackingError`](@ref)
"""
abstract type AbstractTrackingAlgorithm <: AbstractAlgorithm end
function needs_previous_weights(::AbstractTrackingAlgorithm)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all tracking formulation algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing tracking formulation algorithms (such as norm-based or variable-based tracking) should be subtypes of `TrackingFormulation`.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`NormTracking`](@ref)
  - [`VariableTracking`](@ref)
  - [`L2Tracking`](@ref)
  - [`SquaredL2Tracking`](@ref)
  - [`L1Tracking`](@ref)
"""
abstract type TrackingFormulation <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all norm-based tracking formulation algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing norm-based tracking algorithms (such as second-order cone or norm-one tracking) should be subtypes of `NormTracking`.

# Related

  - [`TrackingFormulation`](@ref)
  - [`L2Tracking`](@ref)
  - [`SquaredL2Tracking`](@ref)
  - [`L1Tracking`](@ref)
"""
abstract type NormTracking <: TrackingFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all variable-based tracking formulation algorithms in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing variable-based tracking algorithms (such as independent or dependent variable tracking) should be subtypes of `VariableTracking`.

# Related

  - [`TrackingFormulation`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
"""
abstract type VariableTracking <: TrackingFormulation end
"""
$(DocStringExtensions.TYPEDEF)

Second-order cone (SOC) norm-based tracking formulation.

`L2Tracking` implements a norm-based tracking error formulation using the Euclidean (L2) norm, scaled by the square root of the number of assets minus the degrees of freedom (`ddof`). This is commonly used for tracking error constraints and objectives in portfolio optimisation.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_2}{\\sqrt{T - d}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b})``: L2-norm tracking error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L2Tracking(;
        ddof::Integer = 1
    ) -> L2Tracking

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> L2Tracking()
L2Tracking
  ddof ┴ Int64: 1
```

# Related

  - [`NormTracking`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`L1Tracking`](@ref)
  - [`norm_tracking`](@ref)
"""
@concrete struct L2Tracking <: NormTracking
    """
    $(field_dict[:ddof])
    """
    ddof
    function L2Tracking(ddof::Integer)::L2Tracking
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function L2Tracking(; ddof::Integer = 1)::L2Tracking
    return L2Tracking(ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

Second-order cone (SOC) squared norm-based tracking formulation.

`SquaredL2Tracking` implements a norm-based tracking error formulation using the squared Euclidean (L2) norm, scaled by the number of assets minus the degrees of freedom (`ddof`). This is commonly used for tracking error constraints and objectives in portfolio optimisation where squared error is preferred.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_2^2}{T - d}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b})``: Squared L2-norm tracking error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SquaredL2Tracking(;
        ddof::Integer = 1,
    ) -> SquaredL2Tracking

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> SquaredL2Tracking()
SquaredL2Tracking
  ddof ┴ Int64: 1
```

# Related

  - [`NormTracking`](@ref)
  - [`L2Tracking`](@ref)
  - [`L1Tracking`](@ref)
  - [`norm_tracking`](@ref)
"""
@concrete struct SquaredL2Tracking <: NormTracking
    """
    $(field_dict[:ddof])
    """
    ddof
    function SquaredL2Tracking(ddof::Integer)::SquaredL2Tracking
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof)}(ddof)
    end
end
function SquaredL2Tracking(; ddof::Integer = 1)::SquaredL2Tracking
    return SquaredL2Tracking(ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

Norm-one (NOC) tracking formulation.

`L1Tracking` implements a norm-based tracking error formulation using the L1 (norm-one) distance between portfolio and benchmark weights. This is commonly used for tracking error constraints and objectives in portfolio optimisation where sparsity or absolute deviations are preferred.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_1}{T}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b})``: L1-norm tracking error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T]) When ``T`` is not provided the denominator is 1.

# Constructors

    L1Tracking() -> L1Tracking

# Examples

```jldoctest
julia> L1Tracking()
L1Tracking()
```

# Related

  - [`NormTracking`](@ref)
  - [`L2Tracking`](@ref)
  - [`SquaredL2Tracking`](@ref)
  - [`norm_tracking`](@ref)
"""
struct L1Tracking <: NormTracking end
"""
$(DocStringExtensions.TYPEDEF)

L-p norm tracking error estimator.

Computes the Lp-norm of the difference between portfolio and benchmark returns: ``\\lvert\\mathbf{X} \\boldsymbol{w} - \\boldsymbol{b}\\rvert_p``.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_p}{(T - d)^{1/p}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b})``: Lp-norm tracking error.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.
  - ``p``: Norm order.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LpTracking(; p::Number = 3, ddof::Integer = 0) -> LpTracking

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> LpTracking()
LpTracking
     p ┼ Int64: 3
  ddof ┴ Int64: 0
```

# Related

  - [`NormTracking`](@ref)
  - [`L1Tracking`](@ref)
  - [`L2Tracking`](@ref)
  - [`LInfTracking`](@ref)
"""
@concrete struct LpTracking <: NormTracking
    """
    $(field_dict[:p_rm])
    """
    p
    """
    $(field_dict[:ddof])
    """
    ddof
    function LpTracking(p::Number, ddof::Integer)::LpTracking
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(p), typeof(ddof)}(p, ddof)
    end
end
function LpTracking(; p::Number = 3, ddof::Integer = 0)::LpTracking
    return LpTracking(p, ddof)
end
"""
$(DocStringExtensions.TYPEDEF)

L-infinity norm (maximum absolute deviation) tracking error estimator.

Computes the L∞-norm (maximum absolute deviation) of the difference between portfolio and benchmark returns.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_\\infty}{T - d}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b})``: L∞-norm tracking error. `pos = true` uses ``+\\infty``, `pos = false` uses ``-\\infty``.
  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`. When ``T`` is not provided the denominator is 1.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LInfTracking(; ddof::Integer = 0, pos::Bool = true) -> LInfTracking

Keywords correspond to the struct's fields.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> LInfTracking()
LInfTracking
  ddof ┼ Int64: 0
   pos ┴ Bool: true
```

# Related

  - [`NormTracking`](@ref)
  - [`LpTracking`](@ref)
  - [`L1Tracking`](@ref)
  - [`L2Tracking`](@ref)
"""
@concrete struct LInfTracking <: NormTracking
    """
    $(field_dict[:ddof])
    """
    ddof
    """
    $(field_dict[:pos])
    """
    pos
    function LInfTracking(ddof::Integer, pos::Bool)::LInfTracking
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(ddof), typeof(pos)}(ddof, pos)
    end
end
function LInfTracking(; ddof::Integer = 0, pos::Bool = true)::LInfTracking
    return LInfTracking(ddof, pos)
end
"""
    norm_tracking(f::L2Tracking, a, b, T::Option{<:Number} = nothing)
    norm_tracking(f::SquaredL2Tracking, a, b, T::Option{<:Number} = nothing)
    norm_tracking(::L1Tracking, a, b, T::Option{<:Number} = nothing)
    norm_tracking(f::LpTracking, a, b, T::Option{<:Number} = nothing)
    norm_tracking(f::LInfTracking, a, b, T::Option{<:Number} = nothing)

Compute the norm-based tracking error between portfolio and benchmark weights.

`norm_tracking` computes the tracking error using either the Euclidean (L2) norm for [`L2Tracking`](@ref), squared Euclidean (L2) norm for [`SquaredL2Tracking`](@ref), or the L1 (norm-one) distance for [`L1Tracking`](@ref). The error is optionally scaled by the number of assets and degrees of freedom for SOC, or by the number of assets for NOC.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}_{L_2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_2}{\\sqrt{T - d}}\\,, \\\\
\\mathrm{TE}_{L_2^2}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_2^2}{T - d}\\,, \\\\
\\mathrm{TE}_{L_1}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_1}{T}\\,, \\\\
\\mathrm{TE}_{L_p}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_p}{(T-d)^{1/p}}\\,, \\\\
\\mathrm{TE}_{L_\\infty}(\\boldsymbol{a},\\boldsymbol{b}) &= \\frac{\\|\\boldsymbol{a} - \\boldsymbol{b}\\|_\\infty}{T - d}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{a}``: Portfolio weight or return vector ``T \\times 1``.
  - ``\\boldsymbol{b}``: Benchmark vector ``T \\times 1``.
  - $(math_dict[:T])
  - ``d``: Degrees of freedom, `ddof`.
  - ``p``: Norm order.

# Arguments

  - `f`: Tracking formulation algorithm.
  - `a`: Portfolio weights.
  - `b`: Benchmark weights.
  - `T`: Optional number of observations.

# Returns

  - `err::Number`: Norm-based tracking error.

# Details

  - For `L2Tracking`, computes `LinearAlgebra.norm(a - b, 2) / sqrt(T - f.ddof)` if `T` is not `nothing`, else unscaled.
  - For `SquaredL2Tracking`, computes `LinearAlgebra.norm(a - b, 2)^2 / (T - f.ddof)` if `T` is not `nothing`, else unscaled.
  - For `L1Tracking`, computes `LinearAlgebra.norm(a - b, 1) / T` if `T` is not `nothing`, else unscaled.

# Examples

```jldoctest
julia> PortfolioOptimisers.norm_tracking(L2Tracking(), [0.5, 0.5], [0.6, 0.4], 2)
0.14142135623730948

julia> PortfolioOptimisers.norm_tracking(L1Tracking(), [0.5, 0.5], [0.6, 0.4], 2)
0.09999999999999998
```

# Related

  - [`L2Tracking`](@ref)
  - [`L1Tracking`](@ref)
  - [`NormTracking`](@ref)
  - [`Option`](@ref)
"""
function norm_tracking(f::L2Tracking, a, b, T::Option{<:Number} = nothing)
    factor = isnothing(T) ? 1 : sqrt(T - f.ddof)
    return LinearAlgebra.norm(a - b, 2) / factor
end
function norm_tracking(f::SquaredL2Tracking, a, b, T::Option{<:Number} = nothing)
    factor = isnothing(T) ? 1 : (T - f.ddof)
    val = LinearAlgebra.norm(a - b, 2)
    return val^2 / factor
end
function norm_tracking(::L1Tracking, a, b, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T)
    return LinearAlgebra.norm(a - b, 1) / factor
end
function norm_tracking(f::LpTracking, a, b, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T - f.ddof)
    factor = if f.p == 3
        cbrt(factor)
    else
        factor^(inv(f.p))
    end
    return LinearAlgebra.norm(a - b, f.p) / factor
end
function norm_tracking(f::LInfTracking, a, b, T::Option{<:Number} = nothing)
    factor = ifelse(isnothing(T), 1, T - f.ddof)
    ty = promote_type(eltype(a), eltype(b))
    p = ifelse(f.pos, typemax(ty), typemin(ty))
    return LinearAlgebra.norm(a - b, p) / factor
end
"""
$(DocStringExtensions.TYPEDEF)

Independent variable-based tracking formulation.

`IndependentVariableTracking` tracks the independent variables of a measurement.

# Constructors

    IndependentVariableTracking() -> IndependentVariableTracking

# Examples

```jldoctest
julia> IndependentVariableTracking()
IndependentVariableTracking()
```

# Related

  - [`VariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
"""
struct IndependentVariableTracking <: VariableTracking end
"""
$(DocStringExtensions.TYPEDEF)

Dependent variable-based tracking formulation.

`DependentVariableTracking` tracks the measurement.

# Constructors

    DependentVariableTracking() -> DependentVariableTracking

# Examples

```jldoctest
julia> DependentVariableTracking()
DependentVariableTracking()
```

# Related

  - [`VariableTracking`](@ref)
  - [`IndependentVariableTracking`](@ref)
"""
struct DependentVariableTracking <: VariableTracking end
"""
$(DocStringExtensions.TYPEDEF)

Asset weights-based tracking algorithm.

`WeightsTracking` represents a tracking algorithm that operates directly on portfolio weights, optionally incorporating transaction fees. This is used for tracking error measurement and constraint generation where the comparison is made between portfolio weights and benchmark weights, with optional adjustment for fees.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WeightsTracking(;
        fees::Option{<:Fees} = nothing,
        w::VecNum,
        fixed::Bool = false
    ) -> WeightsTracking

## Validation

  - `!isempty(w)`.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `fees`: Recursively viewed via [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> WeightsTracking(; w = [0.5, 0.5])
WeightsTracking
   fees ┼ nothing
      w ┼ Vector{Float64}: [0.5, 0.5]
  fixed ┴ Bool: false
```

# Related

  - [`ReturnsTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`AbstractTrackingAlgorithm`](@ref)
  - [`Fees`](@ref)
  - [`Option`](@ref)
  - [`tracking_benchmark`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct WeightsTracking <: AbstractTrackingAlgorithm
    """
    $(field_dict[:fees])
    """
    @vprop fees
    """
    $(field_dict[:w_tn])
    """
    @vprop w
    """
    $(field_dict[:fixed])
    """
    fixed
    function WeightsTracking(fees::Option{<:Fees}, w::VecNum, fixed::Bool)::WeightsTracking
        assert_nonempty_finite_val(w, :w)
        return new{typeof(fees), typeof(w), typeof(fixed)}(fees, w, fixed)
    end
end
function WeightsTracking(; fees::Option{<:Fees} = nothing, w::VecNum,
                         fixed::Bool = false)::WeightsTracking
    return WeightsTracking(fees, w, fixed)
end
function needs_previous_weights(tr::WeightsTracking)::Bool
    return !tr.fixed
end
"""
    factory(tr::WeightsTracking, w::VecNum)

Construct a new `WeightsTracking` object with updated portfolio weights.

This function creates a new [`WeightsTracking`](@ref) instance by copying the fees from the input `tr` object and replacing the portfolio weights with `w`. The fees field is updated using the `factory` function on the existing fees and weights.

# Arguments

  - `tr`: A [`WeightsTracking`](@ref) object to copy fees from.
  - `w`: Portfolio weights.

# Returns

  - `tr::WeightsTracking`: New tracking algorithm object with updated weights.

# Details

  - Copies and updates the `fees` field using `factory(tr.fees, tr.w)`.
  - Replaces the `w` field with the provided weights.

# Examples

```jldoctest
julia> tr = WeightsTracking(; fees = Fees(; l = 0.002), w = [0.5, 0.5])
WeightsTracking
   fees ┼ Fees
        │       tn ┼ nothing
        │        l ┼ Float64: 0.002
        │        s ┼ nothing
        │       fl ┼ nothing
        │       fs ┼ nothing
        │   kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
      w ┼ Vector{Float64}: [0.5, 0.5]
  fixed ┴ Bool: false

julia> PortfolioOptimisers.factory(tr, [0.6, 0.4])
WeightsTracking
   fees ┼ Fees
        │       tn ┼ nothing
        │        l ┼ Float64: 0.002
        │        s ┼ nothing
        │       fl ┼ nothing
        │       fs ┼ nothing
        │   kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
      w ┼ Vector{Float64}: [0.6, 0.4]
  fixed ┴ Bool: false

julia> tr = WeightsTracking(; fees = Fees(; l = 0.002), w = [0.5, 0.5], fixed = true)
WeightsTracking
   fees ┼ Fees
        │       tn ┼ nothing
        │        l ┼ Float64: 0.002
        │        s ┼ nothing
        │       fl ┼ nothing
        │       fs ┼ nothing
        │   kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
      w ┼ Vector{Float64}: [0.5, 0.5]
  fixed ┴ Bool: true

julia> PortfolioOptimisers.factory(tr, [0.1, 0.1])
WeightsTracking
   fees ┼ Fees
        │       tn ┼ nothing
        │        l ┼ Float64: 0.002
        │        s ┼ nothing
        │       fl ┼ nothing
        │       fs ┼ nothing
        │   kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
      w ┼ Vector{Float64}: [0.5, 0.5]
  fixed ┴ Bool: true
```

# Related

  - [`WeightsTracking`](@ref)
  - [`VecNum`](@ref)
  - [`factory`](@ref)
"""
function factory(tr::WeightsTracking, w::VecNum)
    return if tr.fixed
        tr
    else
        WeightsTracking(; fees = factory(tr.fees, tr.w), w = w, fixed = tr.fixed)
    end
end
"""
    tracking_benchmark(tr::WeightsTracking, X::MatNum)

Compute the benchmark portfolio returns for a weights-based tracking algorithm.

`tracking_benchmark` computes the net portfolio returns for the benchmark weights stored in a [`WeightsTracking`](@ref) object, optionally adjusting for transaction fees if specified. The asset return matrix `X` is multiplied by the benchmark weights, and fees are deducted if present.

# Arguments

  - `tr`: [`WeightsTracking`](@ref) tracking algorithm containing benchmark weights and optional fees.
  - `X`: Asset return matrix (assets × periods).

# Returns

  - `b::VecNum`: Net benchmark portfolio returns.

# Details

  - If `tr.fees` is not `nothing`, net returns are computed using `calc_net_returns`.
  - Otherwise, returns are computed as `X * tr.w`.

# Examples

```jldoctest
julia> tr = WeightsTracking(; w = [0.5, 0.5]);

julia> X = [0.01 0.02; 0.03 0.04];

julia> PortfolioOptimisers.tracking_benchmark(tr, X)
2-element Vector{Float64}:
 0.015
 0.035
```

# Related

  - [`WeightsTracking`](@ref)
  - [`MatNum`](@ref)
  - [`calc_net_returns`](@ref)
  - [`tracking_benchmark`](@ref)
"""
function tracking_benchmark(tr::WeightsTracking, X::MatNum)
    return calc_net_returns(tr.w, X, tr.fees)
end
"""
$(DocStringExtensions.TYPEDEF)

Returns-based tracking algorithm.

`ReturnsTracking` represents a tracking algorithm that operates directly on portfolio returns, rather than asset weights. This is used for tracking error measurement and constraint generation where the comparison is made between portfolio returns and benchmark returns.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsTracking(;
        w::VecNum
    ) -> ReturnsTracking

## Validation

  - `!isempty(w)`.

# Examples

```jldoctest
julia> ReturnsTracking(; w = [0.01, 0.02, 0.03])
ReturnsTracking
  w ┴ Vector{Float64}: [0.01, 0.02, 0.03]
```

# Related

  - [`WeightsTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`AbstractTrackingAlgorithm`](@ref)
  - [`tracking_benchmark`](@ref)
"""
@concrete struct ReturnsTracking <: AbstractTrackingAlgorithm
    """
    $(field_dict[:w_bm_ret])
    """
    w
    function ReturnsTracking(w::VecNum)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function ReturnsTracking(; w::VecNum)
    return ReturnsTracking(w)
end
"""
    tracking_benchmark(tr::ReturnsTracking, args...)

Return the benchmark portfolio returns for a returns-based tracking algorithm.

`tracking_benchmark` extracts the benchmark portfolio returns stored in a [`ReturnsTracking`](@ref) object. This is used for tracking error measurement and constraint generation where the comparison is made directly between portfolio returns and benchmark returns.

# Arguments

  - `tr`: [`ReturnsTracking`](@ref) tracking algorithm containing benchmark returns.
  - `args...`: For interface compatibility (ignored).

# Returns

  - `b::VecNum`: Benchmark portfolio returns.

# Examples

```jldoctest
julia> tr = ReturnsTracking(; w = [0.01, 0.02, 0.03]);

julia> PortfolioOptimisers.tracking_benchmark(tr)
3-element Vector{Float64}:
 0.01
 0.02
 0.03
```

# Related

  - [`ReturnsTracking`](@ref)
  - [`WeightsTracking`](@ref)
  - [`TrackingError`](@ref)
"""
function tracking_benchmark(tr::ReturnsTracking, args...)
    return tr.w
end
"""
$(DocStringExtensions.TYPEDEF)

Tracking error result type.

`TrackingError` represents the result of a tracking error computation, including the tracking algorithm used, the computed error value, and the tracking formulation algorithm. This type is used to store and propagate tracking error results in portfolio analytics and optimisation workflows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrackingError(;
        tr::AbstractTrackingAlgorithm,
        err::Number = 0.0,
        alg::NormTracking = L2Tracking()
    ) -> TrackingError

## Validation

  - `isfinite(err)` and `err >= 0`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `tr`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `tr`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> tr = WeightsTracking(; w = [0.5, 0.5]);

julia> TrackingError(; tr = tr, err = 0.01)
TrackingError
   tr ┼ WeightsTracking
      │    fees ┼ nothing
      │       w ┼ Vector{Float64}: [0.5, 0.5]
      │   fixed ┴ Bool: false
  err ┼ Float64: 0.01
  alg ┼ L2Tracking
      │   ddof ┴ Int64: 1
```

# Related

  - [`set_tracking_error_constraints!`](@ref)
  - [`AbstractTracking`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
  - [`NormTracking`](@ref)
  - [`L2Tracking`](@ref)
  - [`L1Tracking`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct TrackingError <: AbstractTracking
    """
    $(field_dict[:tr])
    """
    @fprop @vprop tr
    """
    $(field_dict[:err])
    """
    err
    """
    $(field_dict[:tralg])
    """
    alg
    function TrackingError(tr::AbstractTrackingAlgorithm, err::Number, alg::NormTracking)
        assert_nonempty_nonneg_finite_val(err, :err)
        return new{typeof(tr), typeof(err), typeof(alg)}(tr, err, alg)
    end
end
function TrackingError(; tr::AbstractTrackingAlgorithm, err::Number = 0.0,
                       alg::NormTracking = L2Tracking())
    return TrackingError(tr, err, alg)
end
"""
    port_opt_view(tr::VecTr, args...)

Return a vector of tracking views for each element in a vector of tracking results.

This function applies `port_opt_view` to each element of the input vector of tracking results, passing any additional arguments. It enables efficient subsetting and composability for collections of tracking error or tracking constraint results.

# Arguments

  - `tr`: A vector of tracking result objects (`VecTr`).
  - `args...`: Additional arguments to pass to each `port_opt_view` call.

# Returns

  - `tres::Vector{<:AbstractTracking}`: Vector of tracking view results.

# Details

  - Applies `port_opt_view` to each element in the input vector.
  - Passes all additional arguments to each call.
  - Returns a vector of the results.

# Examples

```jldoctest
julia> tr = TrackingError(; tr = WeightsTracking(; w = [0.5, 0.5, 0.5]), err = 0.01);

julia> PortfolioOptimisers.port_opt_view([tr], 1:2)
1-element Vector{TrackingError{WeightsTracking{Nothing, SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, Bool}, Float64, L2Tracking{Int64}}}:
 TrackingError
   tr ┼ WeightsTracking
      │    fees ┼ nothing
      │       w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.5, 0.5]
      │   fixed ┴ Bool: false
  err ┼ Float64: 0.01
  alg ┼ L2Tracking
      │   ddof ┴ Int64: 1
```

# Related

  - [`port_opt_view`](@ref)
  - [`AbstractTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
function port_opt_view(tr::VecTr, i, args...)
    return [port_opt_view(t, i, args...) for t in tr]
end
function needs_previous_weights(tr::TrackingError)
    return needs_previous_weights(tr.tr)
end
function factory(tr::VecTr, w::VecNum)
    return [factory(t, w) for t in tr]
end
function needs_previous_weights(tr::VecTr)
    return any(needs_previous_weights.(tr))
end

export L2Tracking, SquaredL2Tracking, L1Tracking, LpTracking, LInfTracking,
       IndependentVariableTracking, DependentVariableTracking, WeightsTracking,
       ReturnsTracking, TrackingError
