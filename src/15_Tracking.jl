"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the tracking Results.

A tracking Result holds a benchmark and a bound on the distance of the portfolio from it. [`TrackingError`](@ref) bounds a norm of the difference of the return series, and [`RiskTrackingError`](@ref) bounds a difference of risks.

# Related

  - [`AbstractResult`](@ref)
  - [`TrackingError`](@ref)
  - [`RiskTrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
abstract type AbstractTracking <: AbstractResult end
"""
    const VecTr = AbstractVector{<:AbstractTracking}

Alias for a vector of tracking Results.

# Related

  - [`AbstractTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
const VecTr = AbstractVector{<:AbstractTracking}
"""
    const Tr_VecTr = Union{AbstractTracking, VecTr}

Alias for one tracking Result or a vector of tracking Results.

# Related

  - [`AbstractTracking`](@ref)
  - [`VecTr`](@ref)
  - [`TrackingError`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
"""
const Tr_VecTr = Union{AbstractTracking, VecTr}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the tracking algorithms. A tracking algorithm gives the benchmark of a tracking measure.

Every form compares the portfolio series `pr.X * w` with its benchmark series, so the source of the prior that the measure reads decides the frame of the comparison. A raw prior states `x_src = nothing` or `x_src = DataSource()`. An excess prior states `x_src = ExcessSource()`, and its `pr.X` is `X - B`. The table gives the series that each pair tracks, where `s = sum(w)` and `s_b = sum(w_b)`.

| Prior  | Form                            | Vector `B`                      | Matrix `B`            | Status                 |
|:------ |:------------------------------- |:------------------------------- |:--------------------- |:---------------------- |
| raw    | a plain risk measure            | `X * w`                         | `X * w`               | absolute risk          |
| excess | a plain risk measure            | `X * w - B * s`                 | `(X - B) * w`         | active risk            |
| raw    | [`WeightsTracking`](@ref)       | `X * (w - w_b)`                 | `X * (w - w_b)`       | valid                  |
| excess | [`WeightsTracking`](@ref)       | `X * (w - w_b) - B * (s - s_b)` | `(X - B) * (w - w_b)` | valid                  |
| raw    | [`ReturnsTracking`](@ref)       | `X * w - b`                     | `X * w - b`           | valid                  |
| excess | [`ReturnsTracking`](@ref)       | `X * w - B * s - b`             | `(X - B) * w - b`     | valid if `b` is excess |
| raw    | [`DataBenchmarkTracking`](@ref) | `X * w - B`                     | `(X - B) * w`         | valid                  |
| excess | [`DataBenchmarkTracking`](@ref) | `X * w - B * (s + 1)`           | `(X - 2B) * w`        | refused                |

The constructor of the holder refuses the last row, because that series subtracts the benchmark two times. No check can see a wrong use of the two rows of [`WeightsTracking`](@ref) and [`ReturnsTracking`](@ref) under an excess prior, so their docstrings state the meaning. A raw prior with a data benchmark and an excess prior with a plain measure can track the same series, but they are different models. The excess prior also moves `pr.mu`, `pr.sigma` and every other measure of the optimiser, and the raw prior keeps the objective absolute.

# Interfaces

To add a tracking algorithm, subtype `AbstractTrackingAlgorithm` and add these methods:

  - `tracking_benchmark(tr::AbstractTrackingAlgorithm, X::MatNum) -> VecNum`: Returns the benchmark series for the asset return matrix `X`.
  - `factory(tr::AbstractTrackingAlgorithm, w::VecNum) -> AbstractTrackingAlgorithm`: Returns a tracking algorithm that holds the new weights `w`.
  - `port_opt_view(tr::AbstractTrackingAlgorithm, i) -> AbstractTrackingAlgorithm`: Returns the tracking algorithm of the assets at the indices `i`.

## Arguments

  - `tr`: The tracking algorithm.
  - $(arg_dict[:X])
  - `w`: The new portfolio weights.
  - `i`: The index or the indices of the assets in the view.

## Returns

  - `b::VecNum`: The benchmark series that `tracking_benchmark` returns.
  - `tr::AbstractTrackingAlgorithm`: The tracking algorithm that `factory` or `port_opt_view` returns.

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
  - [`DataBenchmarkTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`TrackingRiskMeasure`](@ref)
  - [`ExcessSource`](@ref): The source of an excess prior.
"""
abstract type AbstractTrackingAlgorithm <: AbstractAlgorithm end
function needs_previous_weights(::AbstractTrackingAlgorithm)::Bool
    return false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the variable tracking algorithms.

A variable tracking algorithm states the quantity that a risk tracking measure compares with its benchmark. The quantity is the weights that the risk measure reads, or the risk that the risk measure returns. The norm that measures the difference is a separate choice. A tracking measure holds the norm in a different field, as a [`NormError`](@ref).

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

The weights are the independent variable of a risk measure, so this formulation compares the two portfolios before the risk measure reads them. It reports ``R(\\boldsymbol{w} - \\boldsymbol{w}_{b})``. Under a [`Variance`](@ref) it is the variance of the tracking error of page 15 of [roll1992](@cite), ``(\\boldsymbol{w} - \\boldsymbol{w}_{b})^{\\intercal} \\mathbf{\\Sigma} (\\boldsymbol{w} - \\boldsymbol{w}_{b})``. Any other risk measure generalises it.

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
  - [`Variance`](@ref)

# References

  - $(ref_dict[:roll1992]) Page 15.
"""
struct IndependentVariableTracking <: VariableTracking end
"""
$(DocStringExtensions.TYPEDEF)

Applies the risk measure to each portfolio, then takes the absolute difference of the two risks.

The risk is the dependent variable of a risk measure, so this formulation compares the two portfolios after the risk measure reads them. It reports ``\\left\\lvert R(\\boldsymbol{w}) - R(\\boldsymbol{w}_{b}) \\right\\rvert``.

The functor reports the exact absolute difference. In an optimisation the model states the difference from one side only, because the absolute difference between a convex risk and the benchmark risk, a constant, is not convex. The model reads ``R(\\boldsymbol{w})`` through the epigraph of the risk measure. The epigraph variable is an upper bound on ``R(\\boldsymbol{w})``, and the solver can raise it at no cost. So the model states ``\\max(0,\\ R(\\boldsymbol{w}) - R(\\boldsymbol{w}_{b}))``. It penalises a portfolio with more risk than the benchmark, and it reports zero for a portfolio with less risk. This is true of every measure whose model is an upper bound. Examples are [`ConditionalValueatRisk`](@ref), [`StandardDeviation`](@ref), [`LowOrderMoment`](@ref) and a [`Variance`](@ref) in the semidefinite form, ``\\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W}) \\geq \\boldsymbol{w}^\\intercal\\boldsymbol{\\Sigma}\\boldsymbol{w}``. The one-sided difference has two uses:

  - In a `RiskTrackingRiskMeasure` with `settings.ub`, or in a [`RiskTrackingError`](@ref), it is the ceiling ``R(\\boldsymbol{w}) \\leq R(\\boldsymbol{w}_{b}) + \\varepsilon``. Here ``\\varepsilon`` is the `err` of the tracking error or the `settings.ub` of the measure. The model evaluates ``R(\\boldsymbol{w}_{b})`` on the prior of the optimisation, and a [`WeightsTracking`](@ref) benchmark that is not `fixed` takes the previous weights. So the ceiling changes with the data and with the benchmark. A fixed `settings.ub` on the measure cannot do this.
  - In the objective, it penalises only the risk above the benchmark risk. The semidefinite [`Variance`](@ref) is the one exception, because its inner build shares the lifted matrix ``\\mathbf{W}`` of the head. For this variance, the difference is exact when ``p + s_v \\Sigma_{ii} \\geq s \\Sigma_{ii}`` for every ``i``. The left side prices the growth of ``\\mathbf{W}``, and the right side is the tracking term. Here ``p`` is the penalty of a [`SemiDefinitePhylogeny`](@ref) and ``s`` is the tracking scale. ``s_v`` is the scale of a variance that the objective minimises on the same weights, and it is `0` when the objective has no such variance.

A [`Variance`](@ref) outside the semidefinite form is a quadratic expression, and it does not solve in this mode. The variance takes the semidefinite form under a [`SemiDefinitePhylogeny`](@ref) or under risk-contribution rows. The dependent method of `set_risk_constraints!` for a [`RiskTrackingRiskMeasure`](@ref) states the bound in its `## Relaxation`.

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

Builds the benchmark return series from a vector of benchmark weights, net of the fees of those weights.

The benchmark is a portfolio, not a return series. So [`tracking_benchmark`](@ref) computes ``\\mathbf{X}\\boldsymbol{w}_{b}`` on the return matrix that the caller gives. Page 15 of [roll1992](@cite) states the benchmark as such a portfolio. Equation 1 of [rudolf1999](@cite) states the deviation of the portfolio returns from the benchmark returns. Use it when the benchmark is a known allocation. Use [`ReturnsTracking`](@ref) when only its realised returns are known.

Under a prior that states `x_src = ExcessSource()` with a vector benchmark `B`, the measure tracks `X * (w - w_b) - B * (sum(w) - sum(w_b))`. It equals the tracking error on the raw returns only when the two budgets are equal. Under a matrix benchmark it tracks `(X - B) * (w - w_b)`. No check can see a difference of budgets, so the table of combinations of [`AbstractTrackingAlgorithm`](@ref) states the meaning of each pair.

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
  - ``\\boldsymbol{F}(\\boldsymbol{w}_{b})``: The fee that each period charges on the benchmark, from the `fees` field. It is zero when `fees` is `nothing`. See [`calc_net_returns`](@ref).

The `fees` field reads no fold, and it needs none. [`charge_fees`](@ref) charges this fee over the benchmark series that it builds. An [`AmortisedFees`](@ref) `fa` spreads the two fixed charges over that series, and a `nothing` `fa` charges them on the first observation. The fee does not store the count of observations, so this site needs no fold to set one.

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

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`. Every entry must be finite, so the constructor refuses `[0.5, NaN]`.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `fees`: Recursively viewed through [`port_opt_view`](@ref).
  - `w`: Sliced to the selected indices through [`port_opt_view`](@ref).

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

  - $(ref_dict[:roll1992]) Page 15.
  - $(ref_dict[:rudolf1999]) Section 2, Equation 1.
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

Construct a `WeightsTracking` that holds the new benchmark weights `w`.

The `fixed` flag decides whether the benchmark changes. [`factory(tn::Turnover, w::VecNum)`](@ref) reads its own `fixed` flag in the same way. A `fixed` benchmark is a constant allocation, so the function returns the object unchanged, and `===` holds. A benchmark that is not `fixed` takes `w` as its new reference, and its `fees` advance one step. The nested [`Turnover`](@ref) takes the old `tr.w` as its reference, because the portfolio turns over from that allocation.

# Algorithm

 1. Read `tr.fixed`.
 2. When `tr.fixed` is `true`, return `tr` itself. The function does not read `w`.
 3. When `tr.fixed` is `false`, advance the fees with `factory(tr.fees, tr.w)`. This gives the old weights to the nested turnover as its reference.
 4. Build a new [`WeightsTracking`](@ref) from the advanced fees, the new `w`, and the unchanged `tr.fixed`.

# Arguments

  - `tr`: The [`WeightsTracking`](@ref) to update.
  - `w`: The new benchmark weights.

# Returns

  - `tr::WeightsTracking`: A new tracking algorithm that holds `w`, or `tr` itself when `tr.fixed` is `true`.

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
  - [`factory(tn::Turnover, w::VecNum)`](@ref): Reads the `fixed` flag of the nested turnover in the same way.
  - [`needs_previous_weights`](@ref): Reads the same flag to decide whether the optimiser must give a previous weight vector.
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

Compute the net return series of the benchmark weights of a [`WeightsTracking`](@ref).

This method states no definition of its own. It is [`calc_net_returns(tr.w, X, tr.fees)`](@ref), so the fee follows the amortisation that `calc_net_returns` reads from `tr.fees.fa`. Take the three-period matrix `[0.01 0.02 -0.01 0.03; 0.03 0.04 0.02 -0.02; -0.01 0.005 0.01 0.04]` and ``\\boldsymbol{w}_{b} = [0.3,\\, 0.2,\\, 0.4,\\, 0.1]``. The benchmark is `[0.006, 0.023, 0.006]` with no fee, and `[0.005, 0.022, 0.005]` under `Fees(; l = 0.001)`, whose fee is `0.001`.

# Algorithm

 1. Forward `tr.w`, `X` and `tr.fees` to [`calc_net_returns`](@ref).
 2. A `nothing` `tr.fees` reaches the `args...` method of [`calc_net_returns`](@ref), which returns `X * tr.w`. It does not reach the `Fees` method, so it charges no fee.
 3. A [`Fees`](@ref) `tr.fees` reaches the `Fees` method, which gives `X * tr.w` to [`charge_fees`](@ref).

# Arguments

  - `tr`: The [`WeightsTracking`](@ref) that holds the benchmark weights and the fees.
  - `X`: Asset return matrix `observations × assets`.

# Returns

  - `b::VecNum`: The net return series of the benchmark, one entry for each row of `X`.

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
  - [`calc_net_returns`](@ref): The definition that this method calls.
  - [`Fees`](@ref)
  - [`tracking_benchmark(tr::ReturnsTracking, X::VecNum_MatNum)`](@ref): The sibling, which reads only the number of rows of the return matrix.
"""
function tracking_benchmark(tr::WeightsTracking, X::MatNum)
    return calc_net_returns(tr.w, X, tr.fees)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the return series of a benchmark whose weights are unknown.

[`tracking_benchmark`](@ref) returns the `w` field unchanged. It reads no return matrix and applies no fee. Equation 1 of [rudolf1999](@cite) has this case. Its benchmark is the return series of an index, and the caller has only the published series. Use [`WeightsTracking`](@ref) when the benchmark allocation is known.

The `w` field holds `T` returns, one for each observation, not `N` weights. Its length must equal the number of rows of the return matrix of the model, and [`tracking_benchmark`](@ref) refuses a different length. The series is constant, so it cannot follow rows that change. The online step of an optimiser refuses it. The risk ceiling of a [`ProgrammeAllocationSet`](@ref) and a [`RiskLoss`](@ref) also refuse it. A batch cross-validation with more than one fold refuses it through [`assert_folds_track_no_series`](@ref), because each fold refits on its own rows. [`DataBenchmarkTracking`](@ref) tracks the benchmark that the returns data carries, and it follows the rows.

The series must be in the frame of the prior that the measure reads. Under a prior that states `x_src = ExcessSource()`, the portfolio series is `(X - B) * w`, so the series `w` must be an excess series too. A series equal to `B` then subtracts the benchmark two times. No check can see the frame of a series, so the table of combinations of [`AbstractTrackingAlgorithm`](@ref) states the meaning of each pair.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsTracking(;
        w::VecNum
    ) -> ReturnsTracking

Keywords correspond to the struct's fields.

## Validation

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`. Every entry must be finite, so the constructor refuses `[0.01, NaN]`.

# Examples

```jldoctest
julia> ReturnsTracking(; w = [0.01, 0.02, 0.03])
ReturnsTracking
  w ┴ Vector{Float64}: [0.01, 0.02, 0.03]
```

# Related

  - [`WeightsTracking`](@ref)
  - [`DataBenchmarkTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`AbstractTrackingAlgorithm`](@ref)
  - [`tracking_benchmark`](@ref)

# References

  - $(ref_dict[:rudolf1999]) Section 2, Equation 1.
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

Return the benchmark return series that a [`ReturnsTracking`](@ref) holds.

The function reads no return matrix and applies no fee. The form with `X` reads only the number of rows of `X`. The series holds one return for each row, so the form refuses a matrix whose number of rows is not `length(tr.w)`. Every reader of a tracking error subtracts the series from a series of the same rows. Without the check, a three-entry `w` against a seven-row matrix fails inside the reader with a `DimensionMismatch`. The check cannot see a series of other rows whose length is correct by chance. Such a series gives a tracking error against the wrong rows.

The series is constant after construction, so it does not follow the rows of an online step. The online step of an optimiser refuses it. See [`partial_fit!`](@ref). Use [`DataBenchmarkTracking`](@ref) to track the benchmark that the returns data carries, or [`WeightsTracking`](@ref) to track a benchmark whose weights are known. Both read the rows that the reader holds.

# Algorithm

 1. Refuse `X` when its number of rows is not `length(tr.w)`.
 2. Return `tr.w`.

# Arguments

  - `tr`: The [`ReturnsTracking`](@ref) that holds the benchmark return series.
  - `X`: The asset return matrix, or the portfolio return series, that the reader compares with the benchmark. The function reads only its number of rows.

# Validation

  - `size(X, 1) == length(tr.w)`. The function throws an `ArgumentError` otherwise.

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
  - [`tracking_benchmark(tr::WeightsTracking, X::MatNum)`](@ref): The sibling, which reads the return matrix and charges a fee.
  - [`TrackingError`](@ref)
"""
function tracking_benchmark(tr::ReturnsTracking)
    return tr.w
end
function tracking_benchmark(tr::ReturnsTracking, X::VecNum_MatNum)
    T = size(X, 1)
    @argcheck(length(tr.w) == T,
              ArgumentError("a `ReturnsTracking` series holds one benchmark return per row of the returns it is compared with, but `w` has $(length(tr.w)) entries and the returns have $(T) rows. The series is fixed at construction, so give it the benchmark returns of exactly these rows, track the benchmark that the returns data carries with `DataBenchmarkTracking`, or track a benchmark portfolio with `WeightsTracking`, which reads the rows themselves."))
    return tr.w
end
"""
$(DocStringExtensions.TYPEDEF)

Tracks the benchmark that the returns data carries, so the benchmark follows the rows that each reader holds.

A [`ReturnsTracking`](@ref) series is constant after construction, so it aligns with the rows of one fit only. `DataBenchmarkTracking` reads the `B` field of the [`ReturnsResult`](@ref) instead. Every reader that holds the returns data binds it to the `B` of its own rows with [`bind_data_benchmark`](@ref):

  - An optimiser binds it to the rows that it fits on.
  - The online step binds it to the rows of its buffer.
  - A fold of a cross-validation binds it to the rows of the fold.
  - An online portfolio selection head binds it to the rows of its window.

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
  - ``\\mathbf{B}``: Benchmark returns of the rows, the `B` field. A vector `T × 1` is one benchmark that every asset shares, and a matrix `T × N` holds one benchmark for each asset.
  - $(math_dict[:w_port])

A matrix benchmark gives the portfolio the same weights in the benchmark of each asset, so the tracking error is ``\\lVert (\\mathbf{X} - \\mathbf{B})\\boldsymbol{w} \\rVert``. It is linear in the weights, so every model stays conic. [`collapse_benchmark`](@ref) applies the same rule to the benchmark of a prediction. The benchmark pays no fee.

A data benchmark refuses a prior that states `x_src = ExcessSource()`, because the tracking error then subtracts the benchmark two times. The constructor of the optimiser that holds both refuses the pair. See the table of combinations of [`AbstractTrackingAlgorithm`](@ref). A reader that holds no returns data leaves the benchmark unbound, and [`tracking_benchmark`](@ref) refuses it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DataBenchmarkTracking(;
        B::Option{<:VecNum_MatNum} = nothing
    ) -> DataBenchmarkTracking

Keywords correspond to the struct's fields.

## View parameters

[`port_opt_view`](@ref) views the columns of a matrix `B` at the selected indices. A vector `B` is a series of rows, so the view keeps it unchanged.

# Examples

```jldoctest
julia> DataBenchmarkTracking()
DataBenchmarkTracking
  B ┴ nothing

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.01 0.03; 0.02 -0.01], nb = [\"BM\"],
                          B = [0.02, 0.0]);

julia> PortfolioOptimisers.bind_data_benchmark(DataBenchmarkTracking(), rd)
DataBenchmarkTracking
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

  - $(ref_dict[:rudolf1999]) Section 2, Equation 1.
"""
@concrete struct DataBenchmarkTracking <: AbstractTrackingAlgorithm
    """
    $(field_dict[:B_data_tr])
    """
    B
    function DataBenchmarkTracking(B::Option{<:VecNum_MatNum})
        return new{typeof(B)}(B)
    end
end
function DataBenchmarkTracking(; B::Option{<:VecNum_MatNum} = nothing)
    return DataBenchmarkTracking(B)
end
"""
    tracking_benchmark(tr::DataBenchmarkTracking, X::VecNum_MatNum)

Return the benchmark returns that a [`DataBenchmarkTracking`](@ref) holds, after the check that they cover the rows of `X`.

A vector is the benchmark series. A matrix holds one benchmark for each asset, and the reader forms the series ``\\mathbf{B}\\boldsymbol{w}`` with its own weights. The function applies no fee.

# Algorithm

 1. Refuse an unbound benchmark, whose `B` is `nothing`.
 2. Refuse `B` when its number of rows is not the number of rows of `X`.
 3. Refuse a matrix `B` when its number of columns is not the number of columns of `X`.
 4. Return `tr.B`.

# Arguments

  - `tr`: A [`DataBenchmarkTracking`](@ref) bound to the benchmark of the rows.
  - `X`: The asset return matrix, or the portfolio return series, that the reader compares with the benchmark. The function reads only its size.

# Validation

  - `tr.B` is not `nothing`. The function throws an `ArgumentError` otherwise.
  - `size(tr.B, 1) == size(X, 1)`, and `size(tr.B, 2) == size(X, 2)` when `tr.B` is a matrix. The function throws an `ArgumentError` otherwise.

# Returns

  - `B::VecNum_MatNum`: `tr.B` itself.

# Examples

```jldoctest
julia> tr = DataBenchmarkTracking(; B = [0.01, 0.02]);

julia> PortfolioOptimisers.tracking_benchmark(tr, zeros(2, 3)) === tr.B
true
```

# Related

  - [`DataBenchmarkTracking`](@ref)
  - [`bind_data_benchmark`](@ref)
  - [`tracking_benchmark(tr::ReturnsTracking, X::VecNum_MatNum)`](@ref): The sibling, whose series is constant after construction.
"""
function tracking_benchmark(::DataBenchmarkTracking{Nothing}, ::VecNum_MatNum)
    return throw(ArgumentError("a `DataBenchmarkTracking` reads the benchmark `B` of the returns data, and no returns data bound it. Optimise with a `ReturnsResult` that carries `B`, or evaluate with `expected_risk(r, w, rd)`, which binds it to `rd.B`."))
end
function tracking_benchmark(tr::DataBenchmarkTracking, X::VecNum_MatNum)
    B = tr.B
    @argcheck(size(B, 1) == size(X, 1),
              ArgumentError("a `DataBenchmarkTracking` holds one benchmark return per row of the returns it is compared with, but `B` has $(size(B, 1)) rows and the returns have $(size(X, 1)). Bind it to the returns data of these rows."))
    @argcheck(ndims(B) == 1 || size(B, 2) == size(X, 2),
              ArgumentError("a matrix `B` of a `DataBenchmarkTracking` holds one benchmark per asset, but `B` has $(size(B, 2)) columns and the returns have $(size(X, 2))."))
    return B
end
function port_opt_view(tr::DataBenchmarkTracking{<:MatNum}, i, args...)
    return DataBenchmarkTracking(; B = view(tr.B, :, i))
end
"""
    returns_tracking_path(est)
    returns_tracking_path(::ReturnsTracking)
    returns_tracking_path(v::AbstractVector)
    returns_tracking_path(p::Pair)
    returns_tracking_path(td::TimeDependent)
    returns_tracking_path(::AbstractPartialFitState)
    returns_tracking_path(::Any)

Names the first field of an estimator tree that holds a [`ReturnsTracking`](@ref), or answers `nothing`.

A returns series is constant after construction, and it aligns only with the rows of one batch fit. These callers use this walk to refuse it:

  - The online step of an optimiser.
  - A batch cross-validation with more than one fold, through [`assert_folds_track_no_series`](@ref).
  - The two row readers of an online portfolio selection head, the risk ceiling of a [`ProgrammeAllocationSet`](@ref) and a [`RiskLoss`](@ref).

The walk reads every field of an estimator, of a Result and of an algorithm. A [`TrackingError`](@ref) is a Result, a [`TrackingRiskMeasure`](@ref) is an estimator, and a [`ProgrammeAllocationSet`](@ref) is an algorithm that holds tracking errors. A vector answers with the index of its element in brackets, and a Pair answers through its value, as [`element_path`](@ref) joins them. A series at the root answers `""`, the empty path. A numeric array and every other value answer `nothing`.

A partial-fit state answers `nothing` without a walk. It holds observations and folded copies of the members of its estimator, and never a configuration that a caller writes. The walk already reads each member through the estimator. The online step of an optimiser calls the walk at every step, so a walk into the buffers costs one dispatch for each of their fields at each step.

A [`TimeDependent`](@ref) schedule answers through its `default` only. Entry `i` of the schedule is the value of fold `i` alone, so the caller can give each fold a series of its own rows. The `default` is one value that every solve with no fold reads, so it can misalign as a field can.

# Algorithm

 1. Answer `""` when `est` is a [`ReturnsTracking`](@ref), and `nothing` when it is an [`AbstractPartialFitState`](@ref).
 2. When `est` is a [`TimeDependent`](@ref), answer the path in its `default`, after the name `default`.
 3. Walk every field of `est` in order, and find the path in each field with this function.
 4. Answer the first path found, with the field name before it, as [`tree_path`](@ref) joins them.
 5. Answer `nothing` when no field gives a path.

# Arguments

  - `est`: The estimator, the Result, the algorithm, or any value a field holds.
  - `v`: A vector that a field holds.
  - `p`: A Pair that a vector holds.
  - `td`: A schedule that a field holds.

# Returns

  - `path::Option{<:String}`: The path of the first returns series found, `""` when `est` is itself one, or `nothing`.

# Related

  - [`ReturnsTracking`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_folds_track_no_series`](@ref)
  - [`TimeDependent`](@ref)
  - [`element_path`](@ref)
  - [`tree_path`](@ref)
"""
function returns_tracking_path(est::Union{AbstractEstimator, AbstractResult,
                                          AbstractAlgorithm})
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
function returns_tracking_path(::AbstractPartialFitState)
    return nothing
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
    assert_folds_track_no_series(est, n::Integer)

Refuses a [`ReturnsTracking`](@ref) in the tree of `est` when a batch cross-validation runs more than one fold.

Each fold of a batch cross-validation refits on its own training rows. A returns series is constant after construction, so it covers the rows of one fold at most. The length check of [`tracking_benchmark`](@ref) cannot see the misalignment when the folds share one training length. The folds of a rolling walk-forward and of a `KFold` share one length, and the series then passes the check at every fold. So [`fold_loop`](@ref) calls this function once before the folds run, and the function refuses the series by the name of its field. The walk is [`returns_tracking_path`](@ref), so the function also refuses the series of an online portfolio selection head. That head cuts the series to its first rows, which align only with a fold that starts at the first row.

One fold is one batch fit, so a series of its training rows aligns, and the function accepts it. An entry of a [`TimeDependent`](@ref) schedule is the value of one fold only, so the walk does not read it.

# Algorithm

 1. When `n > 1`, find the first returns series in the tree of `est` with [`returns_tracking_path`](@ref).
 2. Refuse `est` when there is one.

# Arguments

  - `est`: The estimator that the fold loop takes.
  - `n`: The number of folds.

# Validation

  - When `n > 1`, no field in the tree of `est` holds a [`ReturnsTracking`](@ref). The function throws an `ArgumentError` that names the field otherwise.

# Returns

  - `nothing`.

# Related

  - [`ReturnsTracking`](@ref)
  - [`DataBenchmarkTracking`](@ref): The form that follows the rows, because each fold binds it to the benchmark of its training rows.
  - [`returns_tracking_path`](@ref)
  - [`fold_loop`](@ref)
"""
function assert_folds_track_no_series(est, n::Integer)
    if n > 1
        path = returns_tracking_path(est)
        @argcheck(isnothing(path),
                  ArgumentError("`$(typeof(est).name.name)` runs $n folds of a batch cross-validation holding a `ReturnsTracking` at `$(path)`, and a fixed series covers the rows of one fold at most: each fold refits on its own training rows, and when the folds share one training length, the series passes the length check at every fold and misaligns with no error. Track the benchmark that the returns data carries with `DataBenchmarkTracking`, which binds to the `B` of the training rows of each fold, track a benchmark portfolio with `WeightsTracking`, or give each fold its own series in a `TimeDependent` schedule."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Bounds the deviation of the portfolio return series from a benchmark return series.

`err` is an upper bound, not a computed value. [`set_tracking_error_constraints!`](@ref) writes one cone for the norm that `alg` names, and the cone holds the scaled deviation at or below `err`. The `tr` field gives the benchmark, and the `alg` field names the norm that measures the deviation.

The norms come from [rudolf1999](@cite) and [beasley2003](@cite). The docstring of each [`NormError`](@ref) gives the location of its norm in the source. Those works minimise the tracking error. The bound on the tracking error inside another problem follows Section 9.2 of [cajas2025](@cite).

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
  - ``\\lVert \\cdot \\rVert``, ``c``: The norm and the scaling factor that `alg` names. [`norm_error`](@ref) computes the scaled norm, and [`norm_factor`](@ref) gives ``c``.
  - ``\\mathrm{err}``: The `err` field.

!!! warning

    `err` has the units of `alg`, and [`SquaredL2Norm`](@ref) squares the norm. So one number gives two different bounds. `TrackingError(; alg = SquaredL2Norm(), err = 5e-6)` admits an [`L2Norm`](@ref) error up to `sqrt(5e-6)`. `TrackingError(; alg = L2Norm(), err = 5e-6)` admits `5e-6`. To change an `L2Norm` tolerance into a `SquaredL2Norm` tolerance, square it. Do not use the same number for both. The model, [`norm_error`](@ref) and `set_risk_constraints!` read `err` in the same way.

The conversion does not depend on `T`. After the conversion, the two norms write the same cone bound. [`tracking_error_soc_factor`](@ref) computes that bound, and it returns one factor for `(SquaredL2Norm(), err^2)` and for `(L2Norm(), err)`. So the two models give one weight vector, and the realised deviation of the `SquaredL2Norm` model is the square of the realised deviation of the `L2Norm` model. The `ddof` field of `alg` changes that bound.

`alg` selects the keys that the model registers, and each key ends with the index of the constraint. Every branch registers `:t_tr_` for the cone variable, `:tr_` for the deviation expression ``\\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b}k``, and `:ctr_` for the row that holds the cone variable at or below the scaled tolerance. The table gives the cone of each branch, the row of the cone, and the rows that a branch adds to those three:

| `alg`                                     | Cone                        | Cone row        | Rows the branch adds |
|:----------------------------------------- |:--------------------------- |:--------------- |:-------------------- |
| [`L1Norm`](@ref)                          | `JuMP.MOI.NormOneCone`      | `:ctr_noc_`     | none                 |
| [`L2Norm`](@ref), [`SquaredL2Norm`](@ref) | `JuMP.SecondOrderCone`      | `:ctr_soc_`     | none                 |
| [`LpNorm`](@ref)                          | `JuMP.MOI.PowerCone`        | `:ctr_pnorm_`   | `:r_tr_`, `:cstr_`   |
| [`LInfNorm`](@ref)                        | `JuMP.MOI.NormInfinityCone` | `:ctr_infnorm_` | none                 |

So `:ctr_soc_` is the key of the default `alg = L2Norm()` and of [`SquaredL2Norm`](@ref) only. The model registers no `:tracking_risk_` and no `:sq_tracking_risk_`. Those two keys belong to [`TrackingRiskMeasure`](@ref), which measures a difference of risks, not the deviation of a return series.

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

  - `err`, through [`assert_nonempty_nonneg_finite_val`](@ref): `isfinite(err)` and `err >= 0`. A failed check throws a `DomainError` that names `err`. The constructor accepts `err = 0.0`, which permits no deviation from the benchmark.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `tr`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `tr`: Recursively viewed through [`port_opt_view`](@ref).

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
  - [`AbstractTrackingAlgorithm`](@ref): The table of the series that each form tracks under the source of the prior.
  - [`NormError`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
  - [`L1Norm`](@ref)
  - [`norm_error`](@ref)
  - [`norm_factor`](@ref)
  - [`tracking_error_soc_factor`](@ref): Turns `err` into the cone bound. The two `L2` norms give the same bound there.
  - [`TrackingRiskMeasure`](@ref): The risk-difference measure that owns `:tracking_risk_` and `:sq_tracking_risk_`.
  - [`tracking_benchmark`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:rudolf1999]) Section 2, Equation 2. Section 3, Equations 5, 6, 7a and 7c.
  - $(ref_dict[:beasley2003]) Section 3.5.1, Equation 9.
  - $(ref_dict[:cajas2025]) Section 9.2, Equations 9.19 and 9.20. $(ref_dict[:no_original_source])
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

Only a [`WeightsTracking`](@ref) that is not `fixed` needs them, because only it changes its reference allocation when [`factory`](@ref) runs. A [`ReturnsTracking`](@ref) and a [`DataBenchmarkTracking`](@ref) reach the fallback, which returns `false`. The vector method reduces with `any`, not `all`. So one entry that needs the previous weights makes the whole vector need them. A vector with one `fixed` [`WeightsTracking`](@ref) and one that is not `fixed` returns `true`, where `all` gives `false`.

# Algorithm

 1. On any [`AbstractTrackingAlgorithm`](@ref) that no method below claims, return `false`.
 2. On a [`WeightsTracking`](@ref), return `!tr.fixed`.
 3. On a [`TrackingError`](@ref), forward to the `tr.tr` field and return its answer.
 4. On a [`VecTr`](@ref), apply steps 1 to 3 to every entry and reduce with `any`.

# Arguments

  - `tr`: One tracking algorithm, one tracking result, or a vector of tracking results.

# Returns

  - `Bool`: `true` if `tr` needs the previous weights, `false` otherwise.

# Related

  - [`AbstractTrackingAlgorithm`](@ref)
  - [`WeightsTracking`](@ref)
  - [`ReturnsTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`VecTr`](@ref)
  - [`factory(tr::WeightsTracking, w::VecNum)`](@ref): The verb that reads the same `fixed` flag.
  - [`Turnover`](@ref): Has the same `fixed` flag and the same `any` rule.
"""
function needs_previous_weights(tr::TrackingError)
    return needs_previous_weights(tr.tr)
end
function needs_previous_weights(tr::VecTr)
    return any(needs_previous_weights, tr)
end
"""
    bind_data_benchmark(x, rd)
    bind_data_benchmark(tr::DataBenchmarkTracking, rd::ReturnsResult)
    bind_data_benchmark(tr::TrackingError, rd)
    bind_data_benchmark(v::AbstractVector, rd)

Bind every [`DataBenchmarkTracking`](@ref) that `x` holds to the benchmark `B` of the returns data `rd`.

A data benchmark has no series of its own, so each reader that holds the returns data binds it to the rows that the reader reads:

  - An optimiser binds it at its entry.
  - The value-level [`expected_risk`](@ref) binds it on a [`ReturnsResult`](@ref).
  - An online portfolio selection head binds it on the rows of its window.

The bind replaces a `B` bound earlier, so a benchmark never reads the rows of another reader.

The bind does not read the source of the prior. A data benchmark with a prior that states `x_src = ExcessSource()` subtracts the benchmark two times. The configuration alone shows this, so the constructor of the holder refuses the pair before any data arrives. See the table of combinations of [`AbstractTrackingAlgorithm`](@ref).

The function rebuilds an object through its keyword constructor only when a part changes. The function returns an object that holds no data benchmark as itself, so `===` holds. The risk measures and the optimisers that hold a tracking object add their own methods.

# Algorithm

 1. When `x` holds no data benchmark, or `rd` is not a [`ReturnsResult`](@ref), return `x` unchanged. A data benchmark of a reader with no returns data stays unbound, and [`tracking_benchmark`](@ref) refuses it there.
 2. On a [`DataBenchmarkTracking`](@ref), refuse an `rd` that carries no benchmark. Otherwise return a new `DataBenchmarkTracking` that holds `rd.B`.
 3. On a [`TrackingError`](@ref), bind its `tr` field.
 4. Bind each element of a vector, and return the vector itself when no element changes.

# Arguments

  - `x`: The value to bind: a tracking algorithm, a tracking error, a risk measure, an optimiser, or a vector of them.
  - `rd`: The returns data of the rows that the reader holds.

# Validation

  - `rd.B` is not `nothing` when `x` is a [`DataBenchmarkTracking`](@ref). The function throws an `ArgumentError` otherwise.

# Returns

  - `x'`: `x` with every data benchmark bound to `rd.B`, or `x` itself when it holds none.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.01 0.03; 0.02 -0.01], nb = [\"BM\"],
                          B = [0.02, 0.0]);

julia> te = TrackingError(; tr = DataBenchmarkTracking(), err = 0.01);

julia> PortfolioOptimisers.bind_data_benchmark(te, rd).tr.B === rd.B
true

julia> tw = TrackingError(; tr = WeightsTracking(; w = [0.5, 0.5]));

julia> PortfolioOptimisers.bind_data_benchmark(tw, rd) === tw
true
```

# Related

  - [`DataBenchmarkTracking`](@ref)
  - [`TrackingError`](@ref)
  - [`tracking_benchmark`](@ref)
  - [`ReturnsResult`](@ref)
  - [`rebuild_estimator`](@ref)
"""
function bind_data_benchmark(x::Any, ::Any)
    return x
end
function bind_data_benchmark(::DataBenchmarkTracking, rd::ReturnsResult)
    @argcheck(!isnothing(rd.B),
              ArgumentError("a `DataBenchmarkTracking` tracks the benchmark `B` of the returns data, and the returns data carries none. Give the returns data a benchmark `B`, or track a fixed series with `ReturnsTracking` or a benchmark portfolio with `WeightsTracking`."))
    return DataBenchmarkTracking(; B = rd.B)
end
function bind_data_benchmark(tr::TrackingError, rd)
    bound = bind_data_benchmark(tr.tr, rd)
    return bound === tr.tr ? tr : rebuild_estimator(tr, (; tr = bound))
end
function bind_data_benchmark(v::AbstractVector, rd)
    bound = [bind_data_benchmark(x, rd) for x in v]
    return all(splat(===), zip(bound, v)) ? v : bound
end

export IndependentVariableTracking, DependentVariableTracking, WeightsTracking,
       ReturnsTracking, DataBenchmarkTracking, TrackingError
public AbstractTrackingAlgorithm, tracking_benchmark
