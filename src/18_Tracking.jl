"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all tracking result types.

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

Abstract supertype for all tracking algorithm types.

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

Abstract supertype for all variable-based tracking formulation algorithms.

A variable tracking algorithm states **which quantity** a risk-tracking measure compares against its benchmark: the weights that go into the risk measure, or the risk that comes out of it. It is orthogonal to the norm that measures the comparison, which a tracking measure holds in a separate field as a [`NormError`](@ref).

All concrete and/or abstract types representing variable-based tracking algorithms should be subtypes of `VariableTracking`.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
  - [`NormError`](@ref)
"""
abstract type VariableTracking <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Applies the risk measure to the difference between the portfolio weights and the benchmark weights.

The weights are the independent variable of a risk measure, so this formulation compares the two portfolios before the measure is evaluated: it reports ``R(\\boldsymbol{w} - \\boldsymbol{w}_{b})``.

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

# References

  - $(ref_dict[:cajas2025]) Section 9.2.
"""
struct IndependentVariableTracking <: VariableTracking end
"""
$(DocStringExtensions.TYPEDEF)

Applies the risk measure to each portfolio, then takes the absolute difference of the two risks.

The risk is the dependent variable of a risk measure, so this formulation compares the two portfolios after the measure is evaluated: it reports ``\\left\\lvert R(\\boldsymbol{w}) - R(\\boldsymbol{w}_{b}) \\right\\rvert``.

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

Builds the benchmark return series by holding a fixed weight vector, net of its own fees.

The benchmark is stated as a portfolio rather than as a return series, so [`tracking_benchmark`](@ref) computes ``\\mathbf{X}\\boldsymbol{w}_{b}`` on whichever return matrix the caller supplies. Use it when the benchmark is a known allocation. Use [`ReturnsTracking`](@ref) when only its realised returns are known.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{b} &= \\mathbf{X}\\boldsymbol{w}_{b} - \\boldsymbol{F}(\\boldsymbol{w}_{b})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{b}``: `T × 1` benchmark return vector.
  - ``\\mathbf{X}``: `T × N` asset return matrix.
  - ``\\boldsymbol{w}_{b}``: `N × 1` benchmark weight vector, the `w` field.
  - ``\\boldsymbol{F}(\\boldsymbol{w}_{b})``: Per-period fee charged on the benchmark, from the `fees` field. It is zero when `fees` is `nothing`. See [`calc_net_returns`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WeightsTracking(;
        fees::Option{<:Fees} = nothing,
        w::VecNum,
        fixed::Bool = false
    ) -> WeightsTracking

Keywords correspond to the struct's fields.

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
  - [`calc_net_returns`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equations 9.16 and 9.18.
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

Carries the benchmark return series itself, for a benchmark whose weights are unknown.

[`tracking_benchmark`](@ref) returns the `w` field unchanged, so no return matrix is read and no fee is applied. This is the case the book states first: an index whose published series is all the caller has. Use [`WeightsTracking`](@ref) when the benchmark allocation is known.

The `w` field holds `T` returns, one per observation, not `N` weights. Its length must match the number of rows of the return matrix the model is built on.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsTracking(;
        w::VecNum
    ) -> ReturnsTracking

Keywords correspond to the struct's fields.

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

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equations 9.16 and 9.17.
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

Bounds how far the portfolio return series may drift from a benchmark return series.

`err` is an upper bound, not a computed value: [`set_tracking_error_constraints!`](@ref) writes one cone per `alg` and holds the scaled deviation below `err`. `tr` supplies the benchmark and `alg` names the norm that measures the deviation.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TE}(\\boldsymbol{w}) &= \\lVert \\boldsymbol{r}(\\boldsymbol{w}) - \\boldsymbol{b} \\rVert \\cdot c^{-1} \\leq \\mathrm{err}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TE}(\\boldsymbol{w})``: Tracking error.
  - ``\\boldsymbol{r}(\\boldsymbol{w})``: `T × 1` net portfolio return series.
  - ``\\boldsymbol{b}``: `T × 1` benchmark return series, from [`tracking_benchmark`](@ref) on the `tr` field.
  - ``\\lVert \\cdot \\rVert``, ``c``: The norm and the scaling factor that `alg` names. [`norm_error`](@ref) computes the pair and [`norm_factor`](@ref) gives ``c``.
  - ``\\mathrm{err}``: The `err` field.

!!! warning

    `err` is stated in the units of `alg`, and [`SquaredL2Norm`](@ref) squares. The same number therefore means two different bounds: `TrackingError(; alg = SquaredL2Norm(), err = 5e-6)` admits an [`L2Norm`](@ref) error up to `sqrt(5e-6)`, about `0.00224`, where `TrackingError(; alg = L2Norm(), err = 5e-6)` admits `5e-6`. Convert with the square, not by reusing the tolerance. The model, [`norm_error`](@ref) and `set_risk_constraints!` all read `err` the same way.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrackingError(;
        tr::AbstractTrackingAlgorithm,
        err::Number = 0.0,
        alg::NormError = L2Norm()
    ) -> TrackingError

Keywords correspond to the struct's fields.

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
  alg ┼ L2Norm
      │   ddof ┴ Int64: 1
```

# Related

  - [`set_tracking_error_constraints!`](@ref)
  - [`AbstractTracking`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
  - [`NormError`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
  - [`tracking_benchmark`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equations 9.19 to 9.21.
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
    function TrackingError(tr::AbstractTrackingAlgorithm, err::Number, alg::NormError)
        assert_nonempty_nonneg_finite_val(err, :err)
        return new{typeof(tr), typeof(err), typeof(alg)}(tr, err, alg)
    end
end
function TrackingError(; tr::AbstractTrackingAlgorithm, err::Number = 0.0,
                       alg::NormError = L2Norm())
    return TrackingError(tr, err, alg)
end
function needs_previous_weights(tr::TrackingError)
    return needs_previous_weights(tr.tr)
end
function needs_previous_weights(tr::VecTr)
    return any(needs_previous_weights.(tr))
end

export IndependentVariableTracking, DependentVariableTracking, WeightsTracking,
       ReturnsTracking, TrackingError
