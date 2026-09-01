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

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `any(isfinite, w)`. One finite entry is enough, so `[0.5, NaN]` is admitted and `[NaN, NaN]` is not.

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

The `fixed` flag decides whether the benchmark moves, and it obeys it the way [`factory(tn::Turnover, w::VecNum)`](@ref) obeys its own. A `fixed` benchmark is a fixed allocation, so the object is returned unchanged and `===` holds. A benchmark that is not `fixed` takes `w` as its new reference, and its `fees` advance one step: the nested [`Turnover`](@ref) takes the **old** `tr.w` as its reference, because that is the allocation the portfolio is turning over from.

# Algorithm

 1. Read `tr.fixed`.
 2. When `tr.fixed` is `true`, return `tr` itself. The argument `w` is not read.
 3. When `tr.fixed` is `false`, advance the fees with `factory(tr.fees, tr.w)`, which hands the **old** weights to the nested turnover as its reference.
 4. Build a new [`WeightsTracking`](@ref) from the advanced fees, the new `w`, and the unchanged `tr.fixed`.

# Arguments

  - `tr`: A [`WeightsTracking`](@ref) object to copy fees from.
  - `w`: Portfolio weights.

# Returns

  - `tr::WeightsTracking`: New tracking algorithm object with updated weights, or `tr` itself when `tr.fixed` is `true`.

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
  - [`factory(tn::Turnover, w::VecNum)`](@ref): The verb that reads the same `fixed` flag, one level down.
  - [`needs_previous_weights`](@ref): Reads the same flag to decide whether the optimiser must supply a previous weight vector.
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

This method restates no definition of its own. It is [`calc_net_returns(tr.w, X, tr.fees)`](@ref), so the fee is the one scalar that function subtracts from every period. On the three-period matrix `[0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]` and ``\\boldsymbol{w}_{b} = [0.3,\\, 0.2,\\, 0.4,\\, 0.1]``, the benchmark measured `[0.006, 0.023, 0.006]` with no fee, and `[0.005, 0.022, 0.005]` under `Fees(; l = 0.001)`, whose fee is `0.001`.

# Algorithm

 1. Forward `tr.w`, `X` and `tr.fees` to [`calc_net_returns`](@ref).
 2. A `nothing` `tr.fees` reaches the `args...` method of [`calc_net_returns`](@ref), which returns `X * tr.w`. It does not reach the `Fees` method and charge a zero fee.
 3. A [`Fees`](@ref) `tr.fees` reaches the `Fees` method, which subtracts the one scalar `calc_fees(tr.w, tr.fees)` from every entry of `X * tr.w`.

# Arguments

  - `tr`: [`WeightsTracking`](@ref) tracking algorithm containing benchmark weights and optional fees.
  - `X`: Asset return matrix (observations × assets).

# Returns

  - `b::VecNum`: Net benchmark portfolio returns, one entry per row of `X`.

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
  - [`calc_net_returns`](@ref): The single definition this method forwards to.
  - [`Fees`](@ref)
  - [`tracking_benchmark(tr::ReturnsTracking, args...)`](@ref): The sibling, which reads no return matrix at all.
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

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `any(isfinite, w)`. One finite entry is enough, so `[0.01, NaN]` is admitted and `[NaN, NaN]` is not.

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

**No return matrix is read, and no fee is applied.** Every trailing argument is ignored, so a matrix whose number of rows does not match `length(tr.w)` still returns `tr.w` unchanged: a three-entry `w` against a seven-row matrix succeeds here. That is deliberate. A length mismatch is a property of the model the series is put into, so it is raised there rather than by this function.

# Algorithm

 1. Return `tr.w`. No argument after `tr` is read.

# Arguments

  - `tr`: [`ReturnsTracking`](@ref) tracking algorithm containing benchmark returns.
  - `args...`: For interface compatibility (ignored).

# Returns

  - `b::VecNum`: Benchmark portfolio returns. It is `tr.w` itself, not a copy, so `tracking_benchmark(tr) === tr.w`.

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
  - [`tracking_benchmark(tr::WeightsTracking, X::MatNum)`](@ref): The sibling, which does read the return matrix and does charge a fee.
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

The conversion is the square root, it carries no dependence on `T`, and the two norms then write the **same** cone bound. [`tracking_error_soc_factor`](@ref) is where they meet: on `T = 252` it returned `0.04752893855326458` for both `(SquaredL2Norm(), 9e-6)` and `(L2Norm(), 3e-3)`. Two minimum-variance models over 252 observations of 20 assets, tracking an index series under those two settings, returned **identical** weight vectors, and their realised deviations satisfied the square exactly: `9.0e-6` against `3.0e-3` squared, both measured `8.999996766182225e-6`. The `ddof` field of `alg` moves that bound. At `ddof = 0` the same problem realised `0.04762352278968468` and at `ddof = 1` it realised `0.047528930014379016`, each matching its cone bound to eight digits.

The keys the model registers are picked by `alg`, and each carries the constraint index appended. Every branch registers `:t_te_` for the cone variable, `:te_` for the deviation expression ``\\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b}k``, and `:cte_` for the row that holds the cone variable below the scaled tolerance. The cone, its row, and the rows a branch adds beyond those three, are:

| `alg`                                     | Cone                        | Cone row        | Rows the branch adds |
|:----------------------------------------- |:--------------------------- |:--------------- |:-------------------- |
| [`L1Norm`](@ref)                          | `JuMP.MOI.NormOneCone`      | `:cte_noc_`     | none                 |
| [`L2Norm`](@ref), [`SquaredL2Norm`](@ref) | `JuMP.SecondOrderCone`      | `:cte_soc_`     | none                 |
| [`LpNorm`](@ref)                          | `JuMP.MOI.PowerCone`        | `:cte_pnorm_`   | `:r_te_`, `:cste_`   |
| [`LInfNorm`](@ref)                        | `JuMP.MOI.NormInfinityCone` | `:cte_infnorm_` | none                 |

`:cte_soc_` is therefore the key of the default `alg = L2Norm()` and of [`SquaredL2Norm`](@ref) alone. The model registers no `:tracking_risk_` and no `:sq_tracking_risk_`: those two keys belong to [`TrackingRiskMeasure`](@ref), which measures a risk difference rather than a return-series deviation.

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

  - `err`, through [`assert_nonempty_nonneg_finite_val`](@ref): `isfinite(err)` and `err >= 0`, each raising a `DomainError` naming `err`. `err = 0.0` is admitted, and it pins the portfolio to the benchmark.

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
  - [`tracking_error_soc_factor`](@ref): Turns `err` into the cone bound, and is where the two `L2` norms meet.
  - [`TrackingRiskMeasure`](@ref): The risk-difference measure that owns `:tracking_risk_` and `:sq_tracking_risk_`.
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
"""
    needs_previous_weights(tr::AbstractTrackingAlgorithm) -> Bool
    needs_previous_weights(tr::WeightsTracking) -> Bool
    needs_previous_weights(tr::TrackingError) -> Bool
    needs_previous_weights(tr::VecTr) -> Bool

Check whether a tracking algorithm or tracking result needs the previous portfolio weights.

Only a [`WeightsTracking`](@ref) that is not `fixed` needs them, because only it moves its reference allocation when [`factory`](@ref) runs. A [`ReturnsTracking`](@ref) carries a return series and reaches the fallback, which answers `false`. The vector method answers `any` and not `all`, so one entry that needs the previous weights makes the whole vector need them: a vector holding one `fixed` and one free [`WeightsTracking`](@ref) answers `true` where `all` would answer `false`.

# Algorithm

 1. On any [`AbstractTrackingAlgorithm`](@ref) that no method below claims, return `false`.
 2. On a [`WeightsTracking`](@ref), return `!tr.fixed`.
 3. On a [`TrackingError`](@ref), forward to the `tr.tr` field and return its answer.
 4. On a [`VecTr`](@ref), apply steps 1 to 3 to every entry and reduce with `any`.

# Arguments

  - `tr`: One tracking algorithm, one tracking result, or a vector of tracking results.

# Returns

  - `Bool`: `true` if previous weights are needed, `false` otherwise.

# Related

  - [`AbstractTrackingAlgorithm`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`VecTr`](@ref)
  - [`factory(tr::WeightsTracking, w::VecNum)`](@ref): The verb that reads the same `fixed` flag.
  - [`Turnover`](@ref): Carries the same `fixed` flag and the same `any` rule one family across.
"""
function needs_previous_weights(tr::TrackingError)
    return needs_previous_weights(tr.tr)
end
function needs_previous_weights(tr::VecTr)
    return any(needs_previous_weights.(tr))
end

export IndependentVariableTracking, DependentVariableTracking, WeightsTracking,
       ReturnsTracking, TrackingError
