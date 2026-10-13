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
  - [`DataTracking`](@ref)
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

The functor reports the exact absolute difference. In an optimisation the model states the difference from one side only, because the absolute difference between a convex risk and the benchmark risk, a constant, is not convex. The model reaches it through the epigraph of the risk measure, an upper bound on ``R(\\boldsymbol{w})`` that the solver can raise at no cost. So the model states ``\\max(0,\\ R(\\boldsymbol{w}) - R(\\boldsymbol{w}_{b}))``: it penalises a portfolio riskier than the benchmark, and it reports zero for one that is less risky. This is true of every measure whose model is an upper bound, among them [`ConditionalValueatRisk`](@ref), [`StandardDeviation`](@ref), [`LowOrderMoment`](@ref) and a [`Variance`](@ref) in the semidefinite form, ``\\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W}) \\geq \\boldsymbol{w}^\\intercal\\boldsymbol{\\Sigma}\\boldsymbol{w}``:

  - As a bound, a `RiskTrackingRiskMeasure` with `settings.ub` or a [`RiskTrackingError`](@ref), it is the ceiling ``R(\\boldsymbol{w}) \\leq R(\\boldsymbol{w}_{b}) + \\varepsilon``, where ``\\varepsilon`` is the `err` of the tracking error or the `settings.ub` of the measure. The model evaluates ``R(\\boldsymbol{w}_{b})`` on the prior of the optimisation, and a [`WeightsTracking`](@ref) benchmark that is not `fixed` takes the previous weights. So the ceiling moves with the data and with the benchmark, which a fixed `settings.ub` on the measure cannot do.
  - In the objective, it penalises only the risk in excess of the benchmark risk. The semidefinite [`Variance`](@ref) is the one exception, because its inner build shares the lifted matrix ``\\mathbf{W}`` of the head. The difference is exact when the terms that price the growth of ``\\mathbf{W}`` outweigh the tracking term: ``p + s_v \\Sigma_{ii} \\geq s \\Sigma_{ii}`` for every ``i``. Here ``p`` is the penalty of a [`SemiDefinitePhylogeny`](@ref), ``s`` the tracking scale, and ``s_v`` the scale of a variance that the objective minimises on the same weights, `0` when there is none.

A [`Variance`](@ref) outside the semidefinite form is a quadratic expression, and it does not solve in this mode. The variance takes the semidefinite form under a [`SemiDefinitePhylogeny`](@ref) or risk-contribution rows. The dependent method of `set_risk_constraints!` for a [`RiskTrackingRiskMeasure`](@ref) states the bound in its `## Relaxation`.

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

The `fees` field reads no fold, and it needs none. [`charge_fees`](@ref) charges this fee over the benchmark series it builds, so an [`AmortisedFees`](@ref) `fa` spreads the two fixed charges over that series, and a `nothing` `fa` charges them on its first observation. The count is never stored on the fee, so this site needs no fold to settle one.

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

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`. Every entry must be finite, so `[0.5, NaN]` is refused.

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
        │       lq ┼ nothing
        │      flq ┼ nothing
        │       fa ┼ nothing
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
        │       lq ┼ nothing
        │      flq ┼ nothing
        │       fa ┼ nothing
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
        │       lq ┼ nothing
        │      flq ┼ nothing
        │       fa ┼ nothing
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
        │       lq ┼ nothing
        │      flq ┼ nothing
        │       fa ┼ nothing
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

This method restates no definition of its own. It is [`calc_net_returns(tr.w, X, tr.fees)`](@ref), so the fee falls on the clock that function reads from `tr.fees.fa`. On the three-period matrix `[0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]` and ``\\boldsymbol{w}_{b} = [0.3,\\, 0.2,\\, 0.4,\\, 0.1]``, the benchmark measured `[0.006, 0.023, 0.006]` with no fee, and `[0.005, 0.022, 0.005]` under `Fees(; l = 0.001)`, whose fee is `0.001`.

# Algorithm

 1. Forward `tr.w`, `X` and `tr.fees` to [`calc_net_returns`](@ref).
 2. A `nothing` `tr.fees` reaches the `args...` method of [`calc_net_returns`](@ref), which returns `X * tr.w`. It does not reach the `Fees` method and charge a zero fee.
 3. A [`Fees`](@ref) `tr.fees` reaches the `Fees` method, which hands `X * tr.w` to [`charge_fees`](@ref).

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
  - [`tracking_benchmark(tr::ReturnsTracking, X::VecNum_MatNum)`](@ref): The sibling, which reads only the number of rows of the return matrix.
"""
function tracking_benchmark(tr::WeightsTracking, X::MatNum)
    return calc_net_returns(tr.w, X, tr.fees)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the benchmark return series itself, for a benchmark whose weights are unknown.

[`tracking_benchmark`](@ref) returns the `w` field unchanged, so no return matrix is read and no fee is applied. This is the case the book states first: an index whose published series is all the caller has. Use [`WeightsTracking`](@ref) when the benchmark allocation is known.

The `w` field holds `T` returns, one per observation, not `N` weights. Its length must match the number of rows of the return matrix the model is built on, and [`tracking_benchmark`](@ref) refuses a mismatch. The series is fixed, so it cannot follow rows that change: the online step of an optimiser refuses it, and so do the risk ceiling of a [`ProgrammeAllocationSet`](@ref) and a [`RiskLoss`](@ref). Use [`DataTracking`](@ref) to track the benchmark that the returns data carries, which follows the rows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsTracking(;
        w::VecNum
    ) -> ReturnsTracking

Keywords correspond to the struct's fields.

## Validation

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`. Every entry must be finite, so `[0.01, NaN]` is refused.

# Examples

```jldoctest
julia> ReturnsTracking(; w = [0.01, 0.02, 0.03])
ReturnsTracking
  w ┴ Vector{Float64}: [0.01, 0.02, 0.03]
```

# Related

  - [`WeightsTracking`](@ref)
  - [`DataTracking`](@ref)
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
    tracking_benchmark(tr::ReturnsTracking)
    tracking_benchmark(tr::ReturnsTracking, X::VecNum_MatNum)

Return the benchmark portfolio returns for a returns-based tracking algorithm.

`tracking_benchmark` extracts the benchmark portfolio returns stored in a [`ReturnsTracking`](@ref) object. This is used for tracking error measurement and constraint generation where the comparison is made directly between portfolio returns and benchmark returns.

**No return matrix is read, and no fee is applied.** The form with `X` reads only the number of rows of `X`. The series holds one return per row, so the form refuses a matrix whose number of rows is not `length(tr.w)`. Every reader of a tracking error subtracts the series from a series of the same rows. Without the check, a three-entry `w` against a seven-row matrix fails inside the reader with a `DimensionMismatch`, or, when the lengths agree by chance, gives a tracking error against the wrong rows.

The series is fixed at construction, so it does not follow the rows of an online step. The online step of an optimiser refuses it, see [`partial_fit!`](@ref). Use [`DataTracking`](@ref) to track the benchmark that the returns data carries, or [`WeightsTracking`](@ref) to track a benchmark whose weights are known. Both read the rows that the reader holds.

# Algorithm

 1. Refuse `X` when its number of rows is not `length(tr.w)`.
 2. Return `tr.w`.

# Arguments

  - `tr`: [`ReturnsTracking`](@ref) tracking algorithm containing benchmark returns.
  - `X`: The asset return matrix, or the portfolio return series, that the benchmark is compared with. Only its number of rows is read.

# Validation

  - `size(X, 1) == length(tr.w)`. An `ArgumentError` is thrown otherwise.

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

julia> PortfolioOptimisers.tracking_benchmark(tr, zeros(3, 2)) === tr.w
true
```

# Related

  - [`ReturnsTracking`](@ref)
  - [`WeightsTracking`](@ref)
  - [`tracking_benchmark(tr::WeightsTracking, X::MatNum)`](@ref): The sibling, which does read the return matrix and does charge a fee.
  - [`TrackingError`](@ref)
"""
function tracking_benchmark(tr::ReturnsTracking)
    return tr.w
end
function tracking_benchmark(tr::ReturnsTracking, X::VecNum_MatNum)
    T = size(X, 1)
    @argcheck(length(tr.w) == T,
              ArgumentError("a `ReturnsTracking` series holds one benchmark return per row of the returns it is compared with, but `w` has $(length(tr.w)) entries and the returns have $(T) rows. The series is fixed at construction, so give it the benchmark returns of exactly these rows, track the benchmark that the returns data carries with `DataTracking`, or track a benchmark portfolio with `WeightsTracking`, which reads the rows themselves."))
    return tr.w
end
"""
$(DocStringExtensions.TYPEDEF)

Tracks the benchmark that the returns data carries, so the benchmark follows the rows that each reader holds.

A [`ReturnsTracking`](@ref) series is fixed at construction, so it aligns with the rows of one fit only. `DataTracking` reads the `B` field of the [`ReturnsResult`](@ref) instead. Every reader that holds the returns data binds it to the `B` of its own rows with [`bind_data_benchmark`](@ref): an optimiser to the rows it fits on, the online step to the rows of its buffer, a fold of a cross-validation to the rows of the fold, and an online portfolio selection head to the rows of its window. It is a returns series that updates itself.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{b} &= \\begin{cases}
\\mathbf{B} & \\text{a vector benchmark}\\,,\\\\
\\mathbf{B}\\boldsymbol{w} & \\text{a matrix benchmark}\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\boldsymbol{b}``: `T × 1` benchmark return series.
  - ``\\mathbf{B}``: Benchmark returns of the rows, the `B` field. A vector `T × 1` is one benchmark that every asset shares, and a matrix `T × N` holds one benchmark per asset.
  - $(math_dict[:w_port])

A matrix benchmark gives the portfolio the same weights in the benchmark of each asset, so the tracking error is ``\\lVert (\\mathbf{X} - \\mathbf{B})\\boldsymbol{w} \\rVert``. It is linear in the weights, so every model stays conic. [`collapse_benchmark`](@ref) applies the same rule to the benchmark of a prediction. The benchmark pays no fee.

A reader that fits on the returns in excess of the benchmark, `brt = true`, refuses a data benchmark, because the tracking error would subtract the benchmark twice. A reader that holds no returns data leaves the benchmark unbound, and [`tracking_benchmark`](@ref) refuses it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DataTracking(;
        B::Option{<:VecNum_MatNum} = nothing
    ) -> DataTracking

Keywords correspond to the struct's fields.

## View parameters

[`port_opt_view`](@ref) views the columns of a matrix `B` at the selected indices. A vector `B` is a series of rows, so it passes through unchanged.

# Examples

```jldoctest
julia> DataTracking()
DataTracking
  B ┴ nothing

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.01 0.03; 0.02 -0.01], nb = [\"BM\"],
                          B = [0.02, 0.0]);

julia> PortfolioOptimisers.bind_data_benchmark(DataTracking(), rd, false)
DataTracking
  B ┴ Vector{Float64}: [0.02, 0.0]
```

# Related

  - [`ReturnsTracking`](@ref)
  - [`WeightsTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`AbstractTrackingAlgorithm`](@ref)
  - [`bind_data_benchmark`](@ref)
  - [`tracking_benchmark`](@ref)
  - [`ReturnsResult`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.2, Equations 9.16 and 9.17.
"""
@concrete struct DataTracking <: AbstractTrackingAlgorithm
    """
    $(field_dict[:B_data_tr])
    """
    B
    function DataTracking(B::Option{<:VecNum_MatNum})
        return new{typeof(B)}(B)
    end
end
function DataTracking(; B::Option{<:VecNum_MatNum} = nothing)
    return DataTracking(B)
end
"""
    tracking_benchmark(tr::DataTracking, X::VecNum_MatNum)

Return the benchmark returns that a [`DataTracking`](@ref) holds, after the check that they cover the rows of `X`.

A vector is the benchmark series. A matrix holds one benchmark per asset, and the reader forms the series ``\\mathbf{B}\\boldsymbol{w}`` against its own weights. No fee is applied.

# Algorithm

 1. Refuse an unbound benchmark, whose `B` is `nothing`.
 2. Refuse `B` when its number of rows is not the number of rows of `X`.
 3. Refuse a matrix `B` when its number of columns is not the number of columns of `X`.
 4. Return `tr.B`.

# Arguments

  - `tr`: [`DataTracking`](@ref) bound to the benchmark of the rows.
  - `X`: The asset return matrix, or the portfolio return series, that the benchmark is compared with. Only its size is read.

# Validation

  - `tr.B` is not `nothing`. An `ArgumentError` is thrown otherwise.
  - `size(tr.B, 1) == size(X, 1)`, and `size(tr.B, 2) == size(X, 2)` when `tr.B` is a matrix. An `ArgumentError` is thrown otherwise.

# Returns

  - `B::VecNum_MatNum`: `tr.B` itself.

# Examples

```jldoctest
julia> tr = DataTracking(; B = [0.01, 0.02]);

julia> PortfolioOptimisers.tracking_benchmark(tr, zeros(2, 3)) === tr.B
true
```

# Related

  - [`DataTracking`](@ref)
  - [`bind_data_benchmark`](@ref)
  - [`tracking_benchmark(tr::ReturnsTracking, X::VecNum_MatNum)`](@ref): The sibling, whose series is fixed at construction.
"""
function tracking_benchmark(::DataTracking{Nothing}, ::VecNum_MatNum)
    return throw(ArgumentError("a `DataTracking` reads the benchmark `B` of the returns data, and no returns data bound it. Optimise with a `ReturnsResult` that carries `B`, or evaluate with `expected_risk(r, w, rd)`, which binds it to `rd.B`."))
end
function tracking_benchmark(tr::DataTracking, X::VecNum_MatNum)
    B = tr.B
    @argcheck(size(B, 1) == size(X, 1),
              ArgumentError("a `DataTracking` holds one benchmark return per row of the returns it is compared with, but `B` has $(size(B, 1)) rows and the returns have $(size(X, 1)). Bind it to the returns data of these rows."))
    @argcheck(ndims(B) == 1 || size(B, 2) == size(X, 2),
              ArgumentError("a matrix `B` of a `DataTracking` holds one benchmark per asset, but `B` has $(size(B, 2)) columns and the returns have $(size(X, 2))."))
    return B
end
function port_opt_view(tr::DataTracking{<:MatNum}, i, args...)
    return DataTracking(; B = view(tr.B, :, i))
end
"""
    returns_tracking_path(est)
    returns_tracking_path(::ReturnsTracking)
    returns_tracking_path(v::AbstractVector)
    returns_tracking_path(p::Pair)
    returns_tracking_path(::Any)

Names the first field of an estimator tree that holds a [`ReturnsTracking`](@ref), or answers `nothing`.

A returns series is fixed at construction, and it aligns only with the rows of one batch fit. The online step of an optimiser and the two row readers of an online portfolio selection head, the risk ceiling of a [`ProgrammeAllocationSet`](@ref) and a [`RiskLoss`](@ref), use this walk to refuse it. The walk reads every field of an estimator and of a Result, because a [`TrackingError`](@ref) is a Result and a [`TrackingRiskMeasure`](@ref) is an estimator. A vector answers with the index of its element in brackets, and a Pair answers through its value, as [`element_path`](@ref) joins them. A series at the root answers `""`, the empty path. A numeric array and every other value answer `nothing`.

# Algorithm

 1. Answer `""` when `est` is a [`ReturnsTracking`](@ref).
 2. Walk every field of `est` in order, and find the path in each field with this function.
 3. Answer the first path found, with the field name before it, as [`tree_path`](@ref) joins them.
 4. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, the Result, or any value a field holds.
  - `v`: A vector that a field holds.
  - `p`: A Pair that a vector holds.

# Returns

  - `path::Option{<:String}`: The path of the first returns series found, `""` when `est` is itself one, or `nothing`.

# Related

  - [`ReturnsTracking`](@ref)
  - [`partial_fit!`](@ref)
  - [`element_path`](@ref)
  - [`tree_path`](@ref)
"""
function returns_tracking_path(est::Union{<:AbstractEstimator, <:AbstractResult})
    paths = map(f -> returns_tracking_path(getfield(est, f)), fieldnames(typeof(est)))
    i = findfirst(!isnothing, paths)
    return if isnothing(i)
        nothing
    else
        tree_path(fieldname(typeof(est), i), something(paths[i]))
    end
end
function returns_tracking_path(::ReturnsTracking)
    return ""
end
function returns_tracking_path(v::AbstractVector)
    return eltype(v) <: Number ? nothing : element_path(returns_tracking_path, v)
end
function returns_tracking_path(p::Pair)
    return returns_tracking_path(last(p))
end
function returns_tracking_path(::Any)
    return nothing
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

    `err` is stated in the units of `alg`, and [`SquaredL2Norm`](@ref) squares. The same number therefore means two different bounds: `TrackingError(; alg = SquaredL2Norm(), err = 5e-6)` admits an [`L2Norm`](@ref) error up to `sqrt(5e-6)`, where `TrackingError(; alg = L2Norm(), err = 5e-6)` admits `5e-6`. Convert with the square, not by reusing the tolerance. The model, [`norm_error`](@ref) and `set_risk_constraints!` all read `err` the same way.

The conversion is the square root, it carries no dependence on `T`, and the two norms then write the **same** cone bound. [`tracking_error_soc_factor`](@ref) is where they meet: `(SquaredL2Norm(), err^2)` and `(L2Norm(), err)` give one factor, one weight vector, and realised deviations that satisfy the square. The `ddof` field of `alg` moves that bound.

The keys the model registers are picked by `alg`, and each carries the constraint index appended. Every branch registers `:t_tr_` for the cone variable, `:tr_` for the deviation expression ``\\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b}k``, and `:ctr_` for the row that holds the cone variable below the scaled tolerance. The cone, its row, and the rows a branch adds beyond those three, are:

| `alg`                                     | Cone                        | Cone row        | Rows the branch adds |
|:----------------------------------------- |:--------------------------- |:--------------- |:-------------------- |
| [`L1Norm`](@ref)                          | `JuMP.MOI.NormOneCone`      | `:ctr_noc_`     | none                 |
| [`L2Norm`](@ref), [`SquaredL2Norm`](@ref) | `JuMP.SecondOrderCone`      | `:ctr_soc_`     | none                 |
| [`LpNorm`](@ref)                          | `JuMP.MOI.PowerCone`        | `:ctr_pnorm_`   | `:r_tr_`, `:cstr_`   |
| [`LInfNorm`](@ref)                        | `JuMP.MOI.NormInfinityCone` | `:ctr_infnorm_` | none                 |

`:ctr_soc_` is therefore the key of the default `alg = L2Norm()` and of [`SquaredL2Norm`](@ref) alone. The model registers no `:tracking_risk_` and no `:sq_tracking_risk_`: those two keys belong to [`TrackingRiskMeasure`](@ref), which measures a risk difference rather than a return-series deviation.

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
    return any(needs_previous_weights, tr)
end
"""
    bind_data_benchmark(x, rd, brt::Bool)
    bind_data_benchmark(tr::DataTracking, rd::ReturnsResult, brt::Bool)
    bind_data_benchmark(tr::TrackingError, rd, brt::Bool)
    bind_data_benchmark(v::VecNum, rd, brt::Bool)
    bind_data_benchmark(v::AbstractVector, rd, brt::Bool)

Bind every [`DataTracking`](@ref) that `x` holds to the benchmark `B` of the returns data `rd`.

A data benchmark has no series of its own, so each reader that holds the returns data binds it to the rows it reads: an optimiser at its entry, the value-level [`expected_risk`](@ref) on a [`ReturnsResult`](@ref), and an online portfolio selection head on its rows window. The binding replaces a `B` bound earlier, so a benchmark never reads the rows of another reader.

The function rebuilds an object through its keyword constructor only when a part changes. An object that holds no data benchmark comes back as itself, so `===` holds. The risk measures and the optimisers that hold a tracking object add their own methods.

# Algorithm

 1. A value that holds no data benchmark, or `rd` that is not a [`ReturnsResult`](@ref), returns `x` unchanged. A data benchmark that a reader with no returns data reads stays unbound, and [`tracking_benchmark`](@ref) refuses it there.
 2. A [`DataTracking`](@ref) refuses `brt = true`, and refuses `rd` that carries no benchmark. Otherwise it returns a new `DataTracking` that holds `rd.B`.
 3. A [`TrackingError`](@ref) binds its `tr` field.
 4. A vector of numbers returns unchanged. Any other vector binds each element, and returns itself when no element changes.

# Arguments

  - `x`: The value to bind: a tracking algorithm, a tracking error, a risk measure, an optimiser, or a vector of them.
  - `rd`: The returns data of the rows that the reader holds.
  - $(arg_dict[:brt_read]) A data benchmark refuses `true`.

# Validation

  - A [`DataTracking`](@ref) with `brt = true`. An `ArgumentError` is thrown: the reader fits on `X - B`, and a tracking error against `B` would subtract the benchmark twice.
  - A [`DataTracking`](@ref) with `rd.B === nothing`. An `ArgumentError` is thrown.

# Returns

  - `x'`: `x` with every data benchmark bound to `rd.B`, or `x` itself when it holds none.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.01 0.03; 0.02 -0.01], nb = [\"BM\"],
                          B = [0.02, 0.0]);

julia> te = TrackingError(; tr = DataTracking(), err = 0.01);

julia> PortfolioOptimisers.bind_data_benchmark(te, rd, false).tr.B === rd.B
true

julia> tw = TrackingError(; tr = WeightsTracking(; w = [0.5, 0.5]));

julia> PortfolioOptimisers.bind_data_benchmark(tw, rd, false) === tw
true
```

# Related

  - [`DataTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`tracking_benchmark`](@ref)
  - [`ReturnsResult`](@ref)
  - [`rebuild_estimator`](@ref)
"""
function bind_data_benchmark(x::Any, ::Any, ::Bool)
    return x
end
function bind_data_benchmark(::DataTracking, rd::ReturnsResult, brt::Bool)
    @argcheck(!brt,
              ArgumentError("a `DataTracking` tracks the benchmark `B` of the returns data, and `brt = true` fits the optimiser on the returns in excess of `B`. A tracking error against `B` would then subtract the benchmark twice. Set `brt = false` on the optimiser that holds the data benchmark."))
    @argcheck(!isnothing(rd.B),
              ArgumentError("a `DataTracking` tracks the benchmark `B` of the returns data, and the returns data carries none. Give the returns data a benchmark `B`, or track a fixed series with `ReturnsTracking` or a benchmark portfolio with `WeightsTracking`."))
    return DataTracking(; B = rd.B)
end
function bind_data_benchmark(tr::TrackingError, rd, brt::Bool)
    bound = bind_data_benchmark(tr.tr, rd, brt)
    return bound === tr.tr ? tr : rebuild_estimator(tr, (; tr = bound))
end
function bind_data_benchmark(v::VecNum, ::Any, ::Bool)
    return v
end
function bind_data_benchmark(v::AbstractVector, rd, brt::Bool)
    bound = [bind_data_benchmark(x, rd, brt) for x in v]
    return all(splat(===), zip(bound, v)) ? v : bound
end

export IndependentVariableTracking, DependentVariableTracking, WeightsTracking,
       ReturnsTracking, DataTracking, TrackingError
public AbstractTrackingAlgorithm, tracking_benchmark
