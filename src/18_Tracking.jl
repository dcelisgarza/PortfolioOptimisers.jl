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

  - `ddof`: Degrees of freedom adjustment (integer, must be > 0).

# Constructor

    SOCTracking(; ddof::Integer = 1)

## Validation

  - `ddof > 0`.

# Examples

```jldoctest
julia> SOCTracking()
SOCTracking
  ddof ┴ Int64: 1
```

# Related

  - [`NormTracking`](@ref)
  - [`NOCTracking`](@ref)
  - [`norm_tracking`](@ref)
"""
struct SOCTracking{T1} <: NormTracking
    ddof::T1
    function SOCTracking(ddof::Integer)
        @argcheck(ddof > 0, DomainError("`ddof` must be greater than 0:\nddof => $ddof"))
        return new{typeof(ddof)}(ddof)
    end
end
function SOCTracking(; ddof::Integer = 1)
    return SOCTracking(ddof)
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
  - [`norm_tracking`](@ref)
"""
struct NOCTracking <: NormTracking end
"""
    norm_tracking(f::SOCTracking, a, b; N::Union{Nothing, <:Real} = nothing)
    norm_tracking(::NOCTracking, a, b; N::Union{Nothing, <:Real} = nothing)

Compute the norm-based tracking error between portfolio and benchmark weights.

`norm_tracking` calculates the tracking error using either the Euclidean (L2) norm for [`SOCTracking`](@ref) or the L1 (norm-one) distance for [`NOCTracking`](@ref). The error is optionally scaled by the number of assets and degrees of freedom for SOC, or by the number of assets for NOC.

# Arguments

  - `f`: Tracking formulation algorithm.
  - `a`: Portfolio weights.
  - `b`: Benchmark weights.
  - `N`: Optional number of assets.

# Returns

  - `err::Real`: Norm-based tracking error.

# Details

  - For `SOCTracking`, computes `norm(a - b, 2) / sqrt(N - f.ddof)` if `N` is provided, else unscaled.
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
"""
function norm_tracking(f::SOCTracking, a, b, N::Union{Nothing, <:Real} = nothing)
    factor = isnothing(N) ? 1 : sqrt(N - f.ddof)
    return norm(a - b, 2) / factor
end
function norm_tracking(::NOCTracking, a, b, N::Union{Nothing, <:Real} = nothing)
    factor = isnothing(N) ? 1 : N
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

    WeightsTracking(; fees::Union{Nothing, <:Fees} = nothing, w::AbstractVector{<:Real})

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
  - [`tracking_benchmark`](@ref)
"""
struct WeightsTracking{T1, T2} <: AbstractTrackingAlgorithm
    fees::T1
    w::T2
    function WeightsTracking(fees::Union{Nothing, <:Fees}, w::AbstractVector{<:Real})
        @argcheck(!isempty(w), IsEmptyError(non_empty_msg("`w`") * "."))
        return new{typeof(fees), typeof(w)}(fees, w)
    end
end
function WeightsTracking(; fees::Union{Nothing, <:Fees} = nothing,
                         w::AbstractVector{<:Real})
    return WeightsTracking(fees, w)
end
function factory(tracking::WeightsTracking, w::AbstractVector)
    return WeightsTracking(; fees = factory(tracking.fees, tracking.w), w = w)
end
function tracking_view(tracking::WeightsTracking, i::AbstractVector)
    fees = fees_view(tracking.fees, i)
    w = view(tracking.w, i)
    return WeightsTracking(; fees = fees, w = w)
end
"""
    tracking_benchmark(tracking::WeightsTracking, X::AbstractMatrix{<:Real})

Compute the benchmark portfolio returns for a weights-based tracking algorithm.

`tracking_benchmark` calculates the net portfolio returns for the benchmark weights stored in a [`WeightsTracking`](@ref) object, optionally adjusting for transaction fees if specified. The asset return matrix `X` is multiplied by the benchmark weights, and fees are deducted if present.

# Arguments

  - `tracking`: [`WeightsTracking`](@ref) tracking algorithm containing benchmark weights and optional fees.
  - `X`: Asset return matrix (assets × periods).

# Returns

  - `Vector{<:Real}`: Net benchmark portfolio returns.

# Details

  - If `tracking.fees` is provided, net returns are computed using `calc_net_returns`.
  - Otherwise, returns are computed as `X * tracking.w`.

# Examples

```jldoctest
julia> tracking = WeightsTracking(; w = [0.5, 0.5]);

julia> X = [0.01 0.02; 0.03 0.04];

julia> PortfolioOptimisers.tracking_benchmark(tracking, X)
2-element Vector{Float64}:
 0.015
 0.035
```

# Related

  - [`WeightsTracking`](@ref)
  - [`calc_net_returns`](@ref)
  - [`tracking_benchmark`](@ref)
"""
function tracking_benchmark(tracking::WeightsTracking, X::AbstractMatrix{<:Real})
    return calc_net_returns(tracking.w, X, tracking.fees)
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

    ReturnsTracking(; w::AbstractVector{<:Real})

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
    function ReturnsTracking(w::AbstractVector{<:Real})
        @argcheck(!isempty(w), IsEmptyError(non_empty_msg("`w`") * "."))
        return new{typeof(w)}(w)
    end
end
function ReturnsTracking(; w::AbstractVector{<:Real})
    return ReturnsTracking(w)
end
function tracking_view(tracking::ReturnsTracking, ::Any)
    return tracking
end
"""
    tracking_benchmark(tracking::ReturnsTracking, args...)

Return the benchmark portfolio returns for a returns-based tracking algorithm.

`tracking_benchmark` extracts the benchmark portfolio returns stored in a [`ReturnsTracking`](@ref) object. This is used for tracking error measurement and constraint generation where the comparison is made directly between portfolio returns and benchmark returns.

# Arguments

  - `tracking`: [`ReturnsTracking`](@ref) tracking algorithm containing benchmark returns.
  - `args...`: For interface compatibility (ignored).

# Returns

  - `Vector{<:Real}`: Benchmark portfolio returns.

# Examples

```jldoctest
julia> tracking = ReturnsTracking(; w = [0.01, 0.02, 0.03]);

julia> PortfolioOptimisers.tracking_benchmark(tracking)
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
function tracking_benchmark(tracking::ReturnsTracking, args...)
    return tracking.w
end
function factory(tracking::ReturnsTracking, ::Any)
    return tracking
end
"""
    struct TrackingError{T1, T2, T3} <: AbstractTracking
        tracking::T1
        err::T2
        alg::T3
    end

Tracking error result type.

`TrackingError` represents the result of a tracking error computation, including the tracking algorithm used, the computed error value, and the tracking formulation algorithm. This type is used to store and propagate tracking error results in portfolio analytics and optimisation workflows.

# Fields

  - `tracking`: Tracking algorithm object.
  - `err`: Tracking error value.
  - `alg`: Tracking formulation algorithm.

# Constructor

    TrackingError(; tracking::AbstractTrackingAlgorithm, err::Real = 0.0,
                  alg::NormTracking = SOCTracking())

## Validation

  - `isfinite(err)` and `err >= 0`.

# Examples

```jldoctest
julia> tracking = WeightsTracking(; w = [0.5, 0.5]);

julia> TrackingError(; tracking = tracking, err = 0.01)
TrackingError
  tracking ┼ WeightsTracking
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
    tracking::T1
    err::T2
    alg::T3
    function TrackingError(tracking::AbstractTrackingAlgorithm, err::Real,
                           alg::NormTracking)
        @argcheck(isfinite(err) && err >= zero(err),
                  DomainError("`err` must be finite and non-negative:\nerr => $err"))
        return new{typeof(tracking), typeof(err), typeof(alg)}(tracking, err, alg)
    end
end
function TrackingError(; tracking::AbstractTrackingAlgorithm, err::Real = 0.0,
                       alg::NormTracking = SOCTracking())
    return TrackingError(tracking, err, alg)
end
function tracking_view(tracking::TrackingError, i::AbstractVector, args...)
    return TrackingError(; tracking = tracking_view(tracking.tracking, i),
                         err = tracking.err, alg = tracking.alg)
end
function tracking_view(tracking::AbstractVector{<:AbstractTracking}, args...)
    return [tracking_view(t, args...) for t in tracking]
end
function factory(tracking::TrackingError, w::AbstractVector)
    return TrackingError(; tracking = factory(tracking.tracking, w), err = tracking.err,
                         alg = tracking.alg)
end

export WeightsTracking, ReturnsTracking, TrackingError
