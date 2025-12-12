"""
    abstract type AbstractTracking <: AbstractResult end

Abstract supertype for all tracking result types in `PortfolioOptimisers.jl`.

All concrete types representing tracking error or tracking constraint results should subtype `AbstractTracking`. This enables a consistent, composable interface for tracking error measurement, tracking constraint generation, and related portfolio analytics.

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

# Related Types

  - [`AbstractTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
const VecTr = AbstractVector{<:AbstractTracking}
"""
    const Tr_VecTr = Union{<:AbstractTracking, <:VecTr}

Union type for a single tracking result or a vector of tracking results.

# Related Types

  - [`AbstractTracking`](@ref)
  - [`VecTr`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
const Tr_VecTr = Union{<:AbstractTracking, <:VecTr}
"""
    abstract type AbstractTrackingAlgorithm <: AbstractAlgorithm end

Abstract supertype for all tracking algorithm types in PortfolioOptimisers.jl.

All concrete types representing tracking algorithms (such as weights or returns tracking) should subtype `AbstractTrackingAlgorithm`. This enables a consistent, composable interface for tracking error computation, tracking constraint generation, and related portfolio analytics.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
  - [`TrackingError`](@ref)
"""
abstract type AbstractTrackingAlgorithm <: AbstractAlgorithm end
"""
    abstract type TrackingFormulation <: AbstractAlgorithm end

Abstract supertype for all tracking formulation algorithm types in PortfolioOptimisers.jl.

All concrete types representing tracking formulation algorithms (such as norm-based or variable-based tracking) should subtype `TrackingFormulation`. This enables a consistent, composable interface for tracking error computation, constraint generation, and related portfolio analytics.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`NormTracking`](@ref)
  - [`VariableTracking`](@ref)
  - [`SOCTracking`](@ref)
  - [`SquaredSOCTracking`](@ref)
  - [`NOCTracking`](@ref)
"""
abstract type TrackingFormulation <: AbstractAlgorithm end
"""
    abstract type NormTracking <: TrackingFormulation end

Abstract supertype for all norm-based tracking formulation algorithms in PortfolioOptimisers.jl.

All concrete types representing norm-based tracking algorithms (such as second-order cone or norm-one tracking) should subtype `NormTracking`. This enables a consistent, composable interface for tracking error computation and constraint generation using norm-based formulations.

# Related

  - [`TrackingFormulation`](@ref)
  - [`SOCTracking`](@ref)
  - [`SquaredSOCTracking`](@ref)
  - [`NOCTracking`](@ref)
"""
abstract type NormTracking <: TrackingFormulation end
"""
    abstract type VariableTracking <: TrackingFormulation end

Abstract supertype for all variable-based tracking formulation algorithms in PortfolioOptimisers.jl.

All concrete types representing variable-based tracking algorithms (such as independent or dependent variable tracking) should subtype `VariableTracking`. This enables a consistent, composable interface for tracking error computation and constraint generation using variable-based formulations.

# Related

  - [`TrackingFormulation`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
"""
abstract type VariableTracking <: TrackingFormulation end
"""
    struct SOCTracking{T1} <: NormTracking
        ddof::T1
    end

Second-order cone (SOC) norm-based tracking formulation.

`SOCTracking` implements a norm-based tracking error formulation using the Euclidean (L2) norm, scaled by the square root of the number of assets minus the degrees of freedom (`ddof`). This is commonly used for tracking error constraints and objectives in portfolio optimisation.

# Fields

  - `ddof`: Degrees of freedom adjustment.

# Constructor

    SOCTracking(; ddof::Integer = 1)

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> SOCTracking()
SOCTracking
  ddof ┴ Int64: 1
```

# Related

  - [`NormTracking`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`NOCTracking`](@ref)
  - [`norm_tracking`](@ref)
"""
struct SOCTracking{T1} <: NormTracking
    ddof::T1
    function SOCTracking(ddof::Integer)
        @argcheck(zero(ddof) <= ddof, DomainError)
        return new{typeof(ddof)}(ddof)
    end
end
function SOCTracking(; ddof::Integer = 1)
    return SOCTracking(ddof)
end
"""
    struct SquaredSOCTracking{T1} <: NormTracking
        ddof::T1
    end

Second-order cone (SOC) squared norm-based tracking formulation.

`SquaredSOCTracking` implements a norm-based tracking error formulation using the squared Euclidean (L2) norm, scaled by the number of assets minus the degrees of freedom (`ddof`). This is commonly used for tracking error constraints and objectives in portfolio optimisation where squared error is preferred.

# Fields

  - `ddof`: Degrees of freedom adjustment for scaling.

# Constructors

```julia
SquaredSOCTracking(; ddof::Integer = 1)
```

  - `ddof`: Sets the degrees of freedom adjustment.

## Validation

  - `0 <= ddof`.

# Examples

```jldoctest
julia> SquaredSOCTracking()
SquaredSOCTracking
  ddof ┴ Int64: 1
```

# Related

  - [`NormTracking`](@ref)
  - [`SOCTracking`](@ref)
  - [`NOCTracking`](@ref)
  - [`norm_tracking`](@ref)
"""
struct SquaredSOCTracking{T1} <: NormTracking
    ddof::T1
    function SquaredSOCTracking(ddof::Integer)
        @argcheck(zero(ddof) <= ddof, DomainError)
        return new{typeof(ddof)}(ddof)
    end
end
function SquaredSOCTracking(; ddof::Integer = 1)
    return SquaredSOCTracking(ddof)
end
"""
    struct NOCTracking <: NormTracking end

Norm-one (NOC) tracking formulation.

`NOCTracking` implements a norm-based tracking error formulation using the L1 (norm-one) distance between portfolio and benchmark weights. This is commonly used for tracking error constraints and objectives in portfolio optimisation where sparsity or absolute deviations are preferred.

# Examples

```jldoctest
julia> NOCTracking()
NOCTracking()
```

# Related

  - [`NormTracking`](@ref)
  - [`SOCTracking`](@ref)
  - [`SquaredSOCTracking`](@ref)
  - [`norm_tracking`](@ref)
"""
struct NOCTracking <: NormTracking end
"""
    norm_tracking(f::SOCTracking, a, b; N::Option{<:Number} = nothing)
    norm_tracking(f::SquaredSOCTracking, a, b; N::Option{<:Number} = nothing)
    norm_tracking(::NOCTracking, a, b; N::Option{<:Number} = nothing)

Compute the norm-based tracking error between portfolio and benchmark weights.

`norm_tracking` calculates the tracking error using either the Euclidean (L2) norm for [`SOCTracking`](@ref), squared Euclidean (L2) norm for [`SquaredSOCTracking`](@ref), or the L1 (norm-one) distance for [`NOCTracking`](@ref). The error is optionally scaled by the number of assets and degrees of freedom for SOC, or by the number of assets for NOC.

# Arguments

  - `f`: Tracking formulation algorithm.
  - `a`: Portfolio weights.
  - `b`: Benchmark weights.
  - `N`: Optional number of assets.

# Returns

  - `err::Number`: Norm-based tracking error.

# Details

  - For `SOCTracking`, computes `norm(a - b, 2) / sqrt(N - f.ddof)` if `N` is provided, else unscaled.
  - For `SquaredSOCTracking`, computes `norm(a - b, 2)^2 / (N - f.ddof)` if `N` is provided, else unscaled.
  - For `NOCTracking`, computes `norm(a - b, 1) / N` if `N` is provided, else unscaled.

# Examples

```jldoctest
julia> PortfolioOptimisers.norm_tracking(SOCTracking(), [0.5, 0.5], [0.6, 0.4], 2)
0.14142135623730948

julia> PortfolioOptimisers.norm_tracking(NOCTracking(), [0.5, 0.5], [0.6, 0.4], 2)
0.09999999999999998
```

# Related

  - [`SOCTracking`](@ref)
  - [`NOCTracking`](@ref)
  - [`NormTracking`](@ref)
  - [`Option`](@ref)
"""
function norm_tracking(f::SOCTracking, a, b, N::Option{<:Number} = nothing)
    factor = isnothing(N) ? 1 : sqrt(N - f.ddof)
    return norm(a - b, 2) / factor
end
function norm_tracking(f::SquaredSOCTracking, a, b, N::Option{<:Number} = nothing)
    factor = isnothing(N) ? 1 : (N - f.ddof)
    val = norm(a - b, 2)
    return val^2 / factor
end
function norm_tracking(::NOCTracking, a, b, N::Option{<:Number} = nothing)
    factor = ifelse(isnothing(N), 1, N)
    return norm(a - b, 1) / factor
end
"""
    struct IndependentVariableTracking <: VariableTracking end

Independent variable-based tracking formulation.

`IndependentVariableTracking` tracks the independent variables of a measurement.

# Related

  - [`VariableTracking`](@ref)
  - [`DependentVariableTracking`](@ref)
"""
struct IndependentVariableTracking <: VariableTracking end
"""
    struct DependentVariableTracking <: VariableTracking end

Dependent variable-based tracking formulation.

`DependentVariableTracking` tracks the measurement.

# Related

  - [`VariableTracking`](@ref)
  - [`IndependentVariableTracking`](@ref)
"""
struct DependentVariableTracking <: VariableTracking end
"""
    tracking_view(::Nothing, ::Any)

Return a `nothing` value for tracking view when the input tracking object is `nothing`.

Used as a fallback method for missing tracking constraints or estimators, ensuring composability and uniform interface handling in constraint processing workflows.

# Arguments

  - `::Nothing`: Indicates absence of a tracking object.
  - `::Any`: Index or argument (ignored).

# Returns

  - `nothing`: Returns `nothing`.

# Details

  - Used to propagate `nothing` in tracking view operations.
  - Ensures interface consistency for missing tracking objects.

# Related

  - [`tracking_view`](@ref)
  - [`AbstractTracking`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
function tracking_view(::Nothing, ::Any)
    return nothing
end
"""
    struct WeightsTracking{T1, T2} <: AbstractTrackingAlgorithm
        fees::T1
        w::T2
    end

Asset weights-based tracking algorithm.

`WeightsTracking` represents a tracking algorithm that operates directly on portfolio weights, optionally incorporating transaction fees. This is used for tracking error measurement and constraint generation where the comparison is made between portfolio weights and benchmark weights, with optional adjustment for fees.

# Fields

  - `fees`: Optional fees estimator or result to apply to the weights tracking.
  - `w`: Portfolio weights (vector of real numbers).

# Constructor

    WeightsTracking(; fees::Option{<:Fees} = nothing, w::VecNum)

## Validation

  - `!isempty(w)`.

# Examples

```jldoctest
julia> WeightsTracking(; w = [0.5, 0.5])
WeightsTracking
  fees ┼ nothing
     w ┴ Vector{Float64}: [0.5, 0.5]
```

# Related

  - [`ReturnsTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`AbstractTrackingAlgorithm`](@ref)
  - [`Fees`](@ref)
  - [`Option`](@ref)
  - [`tracking_benchmark`](@ref)
"""
struct WeightsTracking{T1, T2} <: AbstractTrackingAlgorithm
    fees::T1
    w::T2
    function WeightsTracking(fees::Option{<:Fees}, w::VecNum)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(fees), typeof(w)}(fees, w)
    end
end
function WeightsTracking(; fees::Option{<:Fees} = nothing, w::VecNum)
    return WeightsTracking(fees, w)
end
"""
    factory(tr::WeightsTracking, w::VecNum)

Construct a new `WeightsTracking` object with updated portfolio weights.

This function creates a new [`WeightsTracking`](@ref) instance by copying the fees from the input `tr` object and replacing the portfolio weights with `w`. The fees field is updated using the `factory` function on the existing fees and weights.

# Arguments

  - `tr`: A [`WeightsTracking`](@ref) object to copy fees from.
  - `w`: Portfolio weights (vector of real numbers).

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
     w ┴ Vector{Float64}: [0.5, 0.5]

julia> PortfolioOptimisers.factory(tr, [0.6, 0.4])
WeightsTracking
  fees ┼ Fees
       │       tn ┼ nothing
       │        l ┼ Float64: 0.002
       │        s ┼ nothing
       │       fl ┼ nothing
       │       fs ┼ nothing
       │   kwargs ┴ @NamedTuple{atol::Float64}: (atol = 1.0e-8,)
     w ┴ Vector{Float64}: [0.6, 0.4]
```

# Related

  - [`WeightsTracking`](@ref)
  - [`VecNum`](@ref)
  - [`factory`](@ref)
"""
function factory(tr::WeightsTracking, w::VecNum)
    return WeightsTracking(; fees = factory(tr.fees, tr.w), w = w)
end
"""
    tracking_view(tr::WeightsTracking, i)

Return a view of a `WeightsTracking` object for the given index or indices.

This function creates a new [`WeightsTracking`](@ref) instance by extracting a view of the fees and portfolio weights fields at the specified index or indices. This enables efficient subsetting and composability for tracking algorithms in portfolio analytics workflows.

# Arguments

  - `tr`: A [`WeightsTracking`](@ref) object containing fees and portfolio weights.
  - `i`: Index or indices to subset the fees and weights.

# Returns

  - `tr::WeightsTracking`: New tracking algorithm object with fees and weights viewed at `i`.

# Details

  - Uses `fees_view` to subset the `fees` field.
  - Uses `view` to subset the `w` field.
  - Returns a new `WeightsTracking` object with the subsetted fields.

# Examples

```jldoctest
julia> tr = WeightsTracking(; w = [0.5, 0.5, 0.6]);

julia> PortfolioOptimisers.tracking_view(tr, 2:3)
WeightsTracking
  fees ┼ nothing
     w ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.5, 0.6]
```

# Related

  - [`WeightsTracking`](@ref)
  - [`fees_view`](@ref)
  - [`tracking_view`](@ref)
"""
function tracking_view(tr::WeightsTracking, i)
    fees = fees_view(tr.fees, i)
    w = view(tr.w, i)
    return WeightsTracking(; fees = fees, w = w)
end
"""
    tracking_benchmark(tr::WeightsTracking, X::MatNum)

Compute the benchmark portfolio returns for a weights-based tracking algorithm.

`tracking_benchmark` calculates the net portfolio returns for the benchmark weights stored in a [`WeightsTracking`](@ref) object, optionally adjusting for transaction fees if specified. The asset return matrix `X` is multiplied by the benchmark weights, and fees are deducted if present.

# Arguments

  - `tr`: [`WeightsTracking`](@ref) tracking algorithm containing benchmark weights and optional fees.
  - `X`: Asset return matrix (assets × periods).

# Returns

  - `b::VecNum`: Net benchmark portfolio returns.

# Details

  - If `tr.fees` is provided, net returns are computed using `calc_net_returns`.
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
    struct ReturnsTracking{T1} <: AbstractTrackingAlgorithm
        w::T1
    end

Returns-based tracking algorithm.

`ReturnsTracking` represents a tracking algorithm that operates directly on portfolio returns, rather than asset weights. This is used for tracking error measurement and constraint generation where the comparison is made between portfolio returns and benchmark returns.

# Fields

  - `w`: Benchmark portfolio returns (vector of real numbers).

# Constructor

    ReturnsTracking(; w::VecNum)

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
struct ReturnsTracking{T1} <: AbstractTrackingAlgorithm
    w::T1
    function ReturnsTracking(w::VecNum)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(w)}(w)
    end
end
function ReturnsTracking(; w::VecNum)
    return ReturnsTracking(w)
end
"""
    tracking_view(tr::ReturnsTracking, ::Any)

Return a view of a `ReturnsTracking` object.

This function returns the input [`ReturnsTracking`](@ref) object unchanged. It is used to provide a consistent interface for tracking view operations on returns-based tracking algorithms.

# Arguments

  - `tr`: A [`ReturnsTracking`](@ref) object containing benchmark portfolio returns.
  - `::Any`: Index or argument (ignored).

# Returns

  - `ReturnsTracking`: The input tracking algorithm object.

# Details

  - Returns the input object unchanged.
  - Ensures interface consistency for tracking view operations.

# Related

  - [`ReturnsTracking`](@ref)
  - [`tracking_view`](@ref)
"""
function tracking_view(tr::ReturnsTracking, ::Any)
    return tr
end
"""
    tracking_benchmark(tr::ReturnsTracking, args...)

Return the benchmark portfolio returns for a returns-based tracking algorithm.

`tracking_benchmark` extracts the benchmark portfolio returns stored in a [`ReturnsTracking`](@ref) object. This is used for tracking error measurement and constraint generation where the comparison is made directly between portfolio returns and benchmark returns.

# Arguments

  - `tr`: [`ReturnsTracking`](@ref) tracking algorithm containing benchmark returns.
  - `args...`: For interface compatibility (ignored).

# Returns

  - `VecNum`: Benchmark portfolio returns.

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
    factory(tr::ReturnsTracking, ::Any)

Return the input `ReturnsTracking` object unchanged.

This function provides a consistent interface for updating or copying returns-based tracking algorithms. It is used for composability in portfolio analytics workflows where the tracking object does not require modification.

# Arguments

  - `tr`: A [`ReturnsTracking`](@ref) object containing benchmark portfolio returns.
  - `::Any`: Index or argument (ignored).

# Returns

  - `ReturnsTracking`: The input tracking algorithm object.

# Details

  - Returns the input object unchanged.
  - Ensures interface consistency for factory operations.

# Related

  - [`ReturnsTracking`](@ref)
  - [`factory`](@ref)
"""
function factory(tr::ReturnsTracking, ::Any)
    return tr
end
"""
    struct TrackingError{T1, T2, T3} <: AbstractTracking
        tr::T1
        err::T2
        alg::T3
    end

Tracking error result type.

`TrackingError` represents the result of a tracking error computation, including the tracking algorithm used, the computed error value, and the tracking formulation algorithm. This type is used to store and propagate tracking error results in portfolio analytics and optimisation workflows.

# Fields

  - `tr`: Tracking algorithm object.
  - `err`: Tracking error value.
  - `alg`: Tracking formulation algorithm.

# Constructor

    TrackingError(; tr::AbstractTrackingAlgorithm, err::Number = 0.0,
                  alg::NormTracking = SOCTracking())

## Validation

  - `isfinite(err)` and `err >= 0`.

# Examples

```jldoctest
julia> tr = WeightsTracking(; w = [0.5, 0.5]);

julia> TrackingError(; tr = tr, err = 0.01)
TrackingError
   tr ┼ WeightsTracking
      │   fees ┼ nothing
      │      w ┴ Vector{Float64}: [0.5, 0.5]
  err ┼ Float64: 0.01
  alg ┼ SOCTracking
      │   ddof ┴ Int64: 1
```

# Related

  - [`AbstractTracking`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
  - [`NormTracking`](@ref)
  - [`SOCTracking`](@ref)
  - [`NOCTracking`](@ref)
"""
struct TrackingError{T1, T2, T3} <: AbstractTracking
    tr::T1
    err::T2
    alg::T3
    function TrackingError(tr::AbstractTrackingAlgorithm, err::Number, alg::NormTracking)
        assert_nonempty_nonneg_finite_val(err, :err)
        return new{typeof(tr), typeof(err), typeof(alg)}(tr, err, alg)
    end
end
function TrackingError(; tr::AbstractTrackingAlgorithm, err::Number = 0.0,
                       alg::NormTracking = SOCTracking())
    return TrackingError(tr, err, alg)
end
"""
    tracking_view(tr::TrackingError, i)

Return a view of a `TrackingError` object for the given index or indices.

This function creates a new [`TrackingError`](@ref) instance by extracting a view of the underlying tracking algorithm at the specified index or indices. The error value and formulation algorithm are preserved. This enables efficient subsetting and composability for tracking error results in portfolio analytics workflows.

# Arguments

  - `tr`: A [`TrackingError`](@ref) object containing a tracking algorithm, error value, and formulation algorithm.
  - `i`: Index or indices to subset the underlying tracking algorithm.

# Returns

  - `tre::TrackingError`: New tracking error result object with the underlying tracking algorithm viewed at `i`.

# Details

  - Uses `tracking_view` to subset the `tr` field.
  - Preserves the `err` and `alg` fields.
  - Returns a new `TrackingError` object with the subsetted tracking algorithm.

# Examples

```jldoctest
julia> tr = WeightsTracking(; w = [0.5, 0.5, 0.6]);

julia> err = TrackingError(; tr = tr, err = 0.01)
TrackingError
   tr ┼ WeightsTracking
      │   fees ┼ nothing
      │      w ┴ Vector{Float64}: [0.5, 0.5, 0.6]
  err ┼ Float64: 0.01
  alg ┼ SOCTracking
      │   ddof ┴ Int64: 1

julia> PortfolioOptimisers.tracking_view(err, 2:3)
TrackingError
   tr ┼ WeightsTracking
      │   fees ┼ nothing
      │      w ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.5, 0.6]
  err ┼ Float64: 0.01
  alg ┼ SOCTracking
      │   ddof ┴ Int64: 1
```

# Related

  - [`TrackingError`](@ref)
  - [`tracking_view`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
function tracking_view(tr::TrackingError, i)
    return TrackingError(; tr = tracking_view(tr.tr, i), err = tr.err, alg = tr.alg)
end
"""
    tracking_view(tr::VecTr, args...)

Return a vector of tracking views for each element in a vector of tracking results.

This function applies `tracking_view` to each element of the input vector of tracking results, passing any additional arguments. It enables efficient subsetting and composability for collections of tracking error or tracking constraint results.

# Arguments

  - `tr`: A vector of tracking result objects (`VecTr`).
  - `args...`: Additional arguments to pass to each `tracking_view` call.

# Returns

  - `tres::Vector{<:AbstractTracking}`: Vector of tracking view results.

# Details

  - Applies `tracking_view` to each element in the input vector.
  - Passes all additional arguments to each call.
  - Returns a vector of the results.

# Related

  - [`tracking_view`](@ref)
  - [`AbstractTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
function tracking_view(tr::VecTr, args...)
    return [tracking_view(t, args...) for t in tr]
end
"""
    factory(tr::TrackingError, w::VecNum)

Construct a new `TrackingError` object with updated tracking algorithm weights.

This function creates a new [`TrackingError`](@ref) instance by updating the underlying tracking algorithm using the provided weights `w`. The error value and formulation algorithm are preserved. This enables composable updates to tracking error results in portfolio analytics workflows.

# Arguments

  - `tr`: A [`TrackingError`](@ref) object containing a tracking algorithm, error value, and formulation algorithm.
  - `w`: Portfolio weights (vector of real numbers) to update the underlying tracking algorithm.

# Returns

  - `tre::TrackingError`: New tracking error result object with the underlying tracking algorithm updated using `w`.

# Details

  - Uses `factory(tr.tr, w)` to update the underlying tracking algorithm.
  - Preserves the `err` and `alg` fields.
  - Returns a new `TrackingError` object with the updated tracking algorithm.

# Examples

```jldoctest
julia> tr = WeightsTracking(; w = [0.5, 0.5]);

julia> err = TrackingError(; tr = tr, err = 0.01)
TrackingError
   tr ┼ WeightsTracking
      │   fees ┼ nothing
      │      w ┴ Vector{Float64}: [0.5, 0.5]
  err ┼ Float64: 0.01
  alg ┼ SOCTracking
      │   ddof ┴ Int64: 1

julia> PortfolioOptimisers.factory(err, [0.6, 0.4])
TrackingError
   tr ┼ WeightsTracking
      │   fees ┼ nothing
      │      w ┴ Vector{Float64}: [0.6, 0.4]
  err ┼ Float64: 0.01
  alg ┼ SOCTracking
      │   ddof ┴ Int64: 1
```

# Related

  - [`TrackingError`](@ref)
  - [`factory`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
function factory(tr::TrackingError, w::VecNum)
    return TrackingError(; tr = factory(tr.tr, w), err = tr.err, alg = tr.alg)
end

export SOCTracking, SquaredSOCTracking, NOCTracking, IndependentVariableTracking,
       DependentVariableTracking, WeightsTracking, ReturnsTracking, TrackingError
