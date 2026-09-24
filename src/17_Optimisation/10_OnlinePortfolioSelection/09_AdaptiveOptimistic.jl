"""
$(DocStringExtensions.TYPEDEF)

Projects the raw step onto the Allocation Set in the norm of a positive diagonal matrix.

This is the geometry of the diagonal adaptive subgradient method of Duchi, Hazan and Singer (2011). [`AdaptiveSubgradient`](@ref) builds one at every step from the gradient mass it has accrued, as [`NewtonStep`](@ref) binds its Gram matrix onto [`GramProjection`](@ref). The rule projects the Start Allocation before it sees a gradient, so [`projection_geometry`](@ref) returns [`EuclideanProjection`](@ref) for that projection. The geometry admits a negative lower bound, as the Euclidean geometry does.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Proj}^{\\boldsymbol{h}}_{\\mathcal{W}}(\\boldsymbol{q}) &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\tfrac{1}{2} (\\boldsymbol{w} - \\boldsymbol{q})^\\intercal \\mathrm{diag}(\\boldsymbol{h}) (\\boldsymbol{w} - \\boldsymbol{q})\\,, \\\\
w_i(\\theta) &= \\min\\left(\\max\\left(q_i - \\theta / h_i,\\, l_i\\right),\\, u_i\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:q_raw])
  - $(math_dict[:W_aset])
  - ``\\boldsymbol{h}``: Positive diagonal of the norm, one entry for each asset.
  - ``l_i``, ``u_i``: Lower and upper bounds of asset ``i`` on a [`BoundedAllocationSet`](@ref).
  - ``\\theta``: Multiplier of the budget, the root of ``\\sum_i w_i(\\theta) = 1``.

On a [`BoundedAllocationSet`](@ref) the projection is ``\\boldsymbol{w}(\\theta)`` at that root. At ``\\boldsymbol{h} = \\boldsymbol{1}`` it is the Euclidean projection.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DiagonalProjection(h::AbstractVector) -> DiagonalProjection

The positional argument is the struct's field. The rule builds the geometry at each step, and a caller never holds one.

## Validation

  - `all(> 0, h)`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> PortfolioOptimisers.DiagonalProjection([1.0, 2.0])
PortfolioOptimisers.DiagonalProjection
  h ┴ Vector{Float64}: [1.0, 2.0]
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`EuclideanProjection`](@ref)
  - [`GramProjection`](@ref): the full-matrix geometry, whose programme arm this geometry shares.
  - [`AdaptiveSubgradient`](@ref)
  - [`project`](@ref)
  - [`set_projection_objective!`](@ref)

# References

  - $(ref_dict[:duchi2011])
"""
struct DiagonalProjection{T1 <: AbstractVector} <: AbstractProjectionGeometry
    """
    The diagonal of the norm of the projection. The rule sets it to `delta` plus its gradient mass at the step.
    """
    h::T1
    function DiagonalProjection(h::AbstractVector)
        @argcheck(all(x -> x > zero(x), h),
                  DomainError(h, "the diagonal of the projection norm must be positive"))
        return new{typeof(h)}(h)
    end
end
"""
    project(proj::DiagonalProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::DiagonalProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)

Projects the raw step `q` onto the Allocation Set `set` in the norm of `diag(proj.h)`. These are the diagonal arms of the Constrained Update, and `w` is the Price-Adjusted Allocation.

On a bounded set the weights of the norm break the sort that projects onto the simplex in the Euclidean geometry. So the bounded arm finds the scalar root on every bound, the simplex included.

# Algorithm

On a [`BoundedAllocationSet`](@ref):

 1. Find the allocation `wn` at the weighted scalar root with [`bounded_quadratic_projection`](@ref), on the bounds of `set` and the weights `proj.h`.
 2. Return `wn`, or the Held Step `w` when `wn` misses the budget in floating point, with [`budget_or_held_step`](@ref).

On a [`ProgrammeAllocationSet`](@ref):

 1. Solve the programme of [`projection_programme`](@ref), whose objective [`set_projection_objective!`](@ref) writes, and return its allocation.

# Related

  - [`project`](@ref)
  - [`DiagonalProjection`](@ref)
  - [`bounded_quadratic_projection`](@ref)
  - [`budget_or_held_step`](@ref)
  - [`projection_programme`](@ref)
"""
function project(proj::DiagonalProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return budget_or_held_step(bounded_quadratic_projection(q, set.wb, proj.h), w, proj,
                               set)
end
function project(proj::DiagonalProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
function set_projection_objective!(model::JuMP.Model, proj::DiagonalProjection,
                                   q::AbstractVector)::Nothing
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t_proj)
    set_distance_cone!(model, t_proj, sc * (sqrt.(proj.h) .* (w - q)), sc)
    set_divergence_objective!(model, t_proj, so)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the gradient mass of each asset, the carrier of the adaptive subgradient rule.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`AdaptiveSubgradient`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct AdaptiveSubgradientState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The gradient mass ``\\boldsymbol{s}_t``, with ``s_{t, i}^2 = \\sum_{s \\leq t} g_{s, i}^2``, `assets × 1`. The rule writes it in place.
    """
    s
end
function merge_states(::AdaptiveSubgradientState, ::AdaptiveSubgradientState)
    return throw(ArgumentError("an `AdaptiveSubgradientState` is not merged on its own: it sits beside an allocation that is order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::AdaptiveSubgradientState)
    return AdaptiveSubgradientState(x.n, copy(x.s))
end
function port_opt_view(x::AdaptiveSubgradientState, i, args...)
    return AdaptiveSubgradientState(x.n, x.s[i])
end
"""
$(DocStringExtensions.TYPEDEF)

Steps each asset at its own rate, set by the gradient mass that the asset has accrued (AdaGrad).

This is the diagonal adaptive subgradient method of Duchi, Hazan and Singer (2011, Algorithm 1) with its composite mirror descent update. The rule projects in the norm of its gradient mass, through a [`DiagonalProjection`](@ref) that it builds at every step, so it holds no `proj` slot.

At `delta = 0` an asset whose every price relative so far is zero has no gradient mass. The diagonal of the norm then has a zero entry, and [`DiagonalProjection`](@ref) refuses it with a `DomainError`. The paper takes a pseudo-inverse there, but the projection in a seminorm has no unique answer, so the rule refuses instead.

# Mathematical definition

```math
\\begin{align}
s_{t, i} &= \\sqrt{\\textstyle\\sum_{s \\leq t} g_{s, i}^2}\\,, \\\\
H_t &= \\delta I + \\mathrm{diag}(\\boldsymbol{s}_t)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta \\langle \\boldsymbol{g}_t, \\boldsymbol{w} \\rangle + \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{w}_t \\rVert^2_{H_t} = \\mathrm{Proj}^{H_t}_{\\mathcal{W}} \\left( \\boldsymbol{w}_t - \\eta H_t^{-1} \\boldsymbol{g}_t \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:g_t_loss]) Under [`LogWealth`](@ref) it is ``-\\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``.
  - $(math_dict[:x_t_rel])
  - $(math_dict[:w_t_iter])
  - $(math_dict[:t_period])
  - ``\\boldsymbol{s}_t``: Gradient mass after period ``t``, with one entry for each asset.
  - ``\\delta``: Offset that the rule adds to the gradient mass of every asset.
  - ``H_t``: Diagonal matrix of the norm of the step at period ``t``.
  - ``\\eta``: Fixed learning rate of the rule.
  - $(math_dict[:W_aset])
  - ``\\lVert \\boldsymbol{z} \\rVert^2_{H} = \\boldsymbol{z}^\\intercal H \\boldsymbol{z}``: Squared norm of a vector ``\\boldsymbol{z}`` in a positive diagonal matrix ``H``.
  - ``\\mathrm{Proj}^{H}_{\\mathcal{W}}``: Projection onto ``\\mathcal{W}`` in the norm of ``H``, which is [`DiagonalProjection`](@ref) with the diagonal of ``H``.

Under log wealth the raw step is ``\\boldsymbol{w}_t + \\eta \\boldsymbol{x}_t / (\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle (\\delta + \\boldsymbol{s}_t))``, with the division taken asset by asset. At ``\\delta = 0`` the first raw step is ``\\boldsymbol{w}_1 + \\eta \\boldsymbol{1}``, a uniform shift, because ``s_{1, i} = \\lvert g_{1, i} \\rvert``. The projection in the norm of ``H_1`` then takes more of the shift back from the assets with the smaller gradient, so the first step moves weight toward the assets that rose the most. It leaves the allocation unchanged only when every price relative is equal.

Let ``D_\\infty = \\sup_{\\boldsymbol{w} \\in \\mathcal{W}} \\lVert \\boldsymbol{w} - \\boldsymbol{w}^* \\rVert_\\infty`` for a fixed comparator ``\\boldsymbol{w}^*``. At ``\\delta = 0`` and ``\\eta = D_\\infty / \\sqrt{2}``, Corollary 6 of the paper bounds the regret against ``\\boldsymbol{w}^*`` by ``\\sqrt{2} D_\\infty \\sum_i \\lVert \\boldsymbol{g}_{1:T, i} \\rVert_2``, with ``\\boldsymbol{g}_{1:T, i}`` the gradients of asset ``i`` over ``T`` periods. On the simplex ``D_\\infty \\leq 1``, and the default ``\\eta = 1 / \\sqrt{2}`` is that rate. The bound is small when a few assets carry most of the gradient mass. The paper shows that the bound is below the bound of online gradient descent on a box. On the simplex, with dense gradients, it can be larger by a factor of up to ``\\sqrt{N / 2}``, and log-wealth gradients are dense.

The full-matrix variant of the paper (Algorithm 2) uses ``H_t = \\delta I + (\\sum_{s \\leq t} \\boldsymbol{g}_s \\boldsymbol{g}_s^\\intercal)^{1/2}``, the root of the sum that [`NewtonStep`](@ref) accrues. This rule keeps the diagonal alone, and the diagonal keeps the step at ``O(N)``.

# Algorithm

The seven-argument [`online_update!`](@ref) runs these steps at the period's row `x`. The six-argument form calls it with the book `w` as the Gradient Point `point`.

 1. Take the gradient `g` of `obj` at `point`, from `x` and the head's `rows`, with [`loss_gradient`](@ref).
 2. Form the Price-Adjusted Allocation `wh` of the book `w` after the row.
 3. When every entry of `g` is zero, project `w` onto the set from `wh` in the Euclidean geometry. Count the period, keep the gradient mass, and return that allocation. Stop.
 4. Add `g` to the gradient mass `st.s` in place, as the root of the sum of squares of each asset.
 5. Form the diagonal `h = delta .+ st.s`.
 6. Form the raw step `q = w .- eta .* g ./ h`.
 7. Project `q` onto the set from `wh` in the geometry `DiagonalProjection(h)`, which gives the new allocation.
 8. Return the state, with the period count raised by one, and the new allocation.

A zero gradient accrues no mass, and the norm of step 7 has no meaning there. So step 3 takes the zero step of every first-order rule. A [`RiskLoss`](@ref) gives a zero gradient while the head holds fewer than two rows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AdaptiveSubgradient(;
        eta::Real = 1 / sqrt(2),
        delta::Real = 0,
        obj::AbstractOnlineObjective = LogWealth()
    ) -> AdaptiveSubgradient

Keywords correspond to the struct's fields. The paper fixes no default. `eta = 1 / sqrt(2)` is the rate of its Corollary 6 on the simplex, and the paper states that `delta = 0` works in practice. `obj` is the slot of [`MirrorDescent`](@ref). It takes log wealth, or a [`RiskLoss`](@ref) over the head's rows, whose gradient [`loss_gradient`](@ref) gives at the Gradient Point. The rule needs as many rows as its objective.

## Validation

  - `eta > 0`, `delta >= 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> AdaptiveSubgradient()
AdaptiveSubgradient
    eta ┼ Float64: 0.7071067811865475
  delta ┼ Int64: 0
    obj ┴ LogWealth()
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`AdaptiveSubgradientState`](@ref)
  - [`DiagonalProjection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`NewtonStep`](@ref)

# References

  - $(ref_dict[:duchi2011])
"""
struct AdaptiveSubgradient{T1 <: Real, T2 <: Real, T3 <: AbstractOnlineObjective} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The learning rate of the step. A larger rate reacts faster and is less stable.
    """
    eta::T1
    """
    The paper's ``\\delta``, which the rule adds to the gradient mass of every asset.
    """
    delta::T2
    """
    The objective whose gradient the rule takes, log wealth or a Risk Loss over the head's rows.
    """
    obj::T3
    function AdaptiveSubgradient(eta::Real, delta::Real, obj::AbstractOnlineObjective)
        @argcheck(eta > zero(eta), DomainError(eta, "eta must be positive"))
        @argcheck(delta >= zero(delta), DomainError(delta, "delta must be non-negative"))
        return new{typeof(eta), typeof(delta), typeof(obj)}(eta, delta, obj)
    end
end
function AdaptiveSubgradient(; eta::Real = 1 / sqrt(2), delta::Real = 0,
                             obj::AbstractOnlineObjective = LogWealth())::AdaptiveSubgradient
    return AdaptiveSubgradient(eta, delta, obj)
end
function rows_needed(alg::AdaptiveSubgradient)
    return rows_needed(alg.obj)
end
function projection_geometry(::AdaptiveSubgradient)
    # The Start Allocation is projected before any gradient, where no mass has accrued.
    return EuclideanProjection()
end
function rule_state_seed(::AdaptiveSubgradient, w::AbstractVector)
    return AdaptiveSubgradientState(0, zeros(eltype(w), length(w)))
end
function online_update!(alg::AdaptiveSubgradient, st::AdaptiveSubgradientState,
                        w::AbstractVector, x::AbstractVector, rows,
                        set::AbstractAllocationSet)
    return online_update!(alg, st, w, x, rows, set, w)
end
# The seven-argument form is the primitive: the gradient is read at `point`, the played
# allocation on the head and the mixture's played blend under `BlendPoint`.
function online_update!(alg::AdaptiveSubgradient, st::AdaptiveSubgradientState,
                        w::AbstractVector, x::AbstractVector, rows,
                        set::AbstractAllocationSet, point::AbstractVector)
    g = loss_gradient(alg.obj, point, x, rows)
    wh = price_adjusted_allocation(w, x)
    if all(iszero, g)
        # A zero gradient is a zero step, and no mass accrues: the norm is undefined, so
        # the iterate is projected in the Euclidean geometry, as a first-order rule's zero
        # step is.
        return AdaptiveSubgradientState(st.n + 1, st.s),
               project(EuclideanProjection(), set, w, wh)
    end
    st.s .= hypot.(st.s, g)
    h = alg.delta .+ st.s
    q = w .- alg.eta .* g ./ h
    wn = project(DiagonalProjection(h), set, q, wh)
    return AdaptiveSubgradientState(st.n + 1, st.s), wn
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the Gradient Predictors, which give the hint of an optimistic step for the gradient of the next period.

An [`OptimisticStep`](@ref) takes its second half-step along the hint ``M_{t+1}``.

# Interfaces

To implement a new predictor, subtype `AbstractGradientPredictor` and implement:

  - `predictor_state_seed(pred::AbstractGradientPredictor, w::AbstractVector)`: Returns the carrier that the predictor keeps on the Rule State before the first row, `nothing` by default. `w` is the Start Allocation, whose length and element type the carrier takes.
  - `predict_gradient!(pred::AbstractGradientPredictor, ps, obj::AbstractOnlineObjective, g::AbstractVector, v::AbstractVector, x::AbstractVector, xm::AbstractVector, rows, t::Integer) -> Tuple`: Returns the carrier after the period and the hint. The arguments are the objective `obj` of the wrapped rule, the gradient `g` of the period at the played allocation, the secondary iterate `v` that the second half-step starts from, the price relative `x` of the period as traded, the same relative `xm` after the uniform mix of the wrapped rule, the rows that the head holds through the period, and the period count `t`. `g` reads `xm`, and `xm` is `x` where the wrapped rule does not mix. A predictor that reads the relative as a price folds `x`. A predictor that evaluates the loss of the wrapped rule again reads `xm`, so the hint sees the loss that `g` saw.
  - `rows_needed(pred::AbstractGradientPredictor)`: The rows the predictor reads at a step, `0` by default.

The head's state already slices and copies a carrier that is `nothing`, a vector or a Partial Fit State. A carrier of another shape needs its own methods of [`predictor_state_view`](@ref) and [`copy_predictor_state`](@ref).

# Related

  - [`LastGradient`](@ref)
  - [`MeanGradient`](@ref)
  - [`ForecastGradient`](@ref)
  - [`OptimisticStep`](@ref)
"""
abstract type AbstractGradientPredictor <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Takes the last gradient as the hint of the optimistic step.

This is the default predictor of [`OptimisticStep`](@ref). Under it the optimistic step is the two-projection algorithm of Chiang, Yang, Lee, Mahdavi, Lu, Jin and Zhu (2012), and the residual sum ``\\sum_t \\lVert \\boldsymbol{g}_t - \\boldsymbol{g}_{t-1} \\rVert_*^2`` is the path length of the gradients.

With `at_played = false` the predictor reads the loss of the period again at the new secondary iterate, through [`loss_gradient`](@ref). This is the online form of the Mirror-Prox method of Nemirovski (2004). Rakhlin and Sridharan [rakhlin2013nips](@cite) recover Mirror Prox from the optimistic step with the hint at the secondary point, for one fixed objective, so the two agree when every period has the same loss. The secondary iterate ``\\boldsymbol{v}_{t+1}`` is not the played allocation ``\\boldsymbol{w}_t``, so the two hints of this type differ in general from the first period on.

# Mathematical definition

```math
\\begin{align}
M_{t+1} &= \\boldsymbol{g}_t\\,, \\\\
M_{t+1}^{\\mathrm{MP}} &= -\\boldsymbol{x}_t / \\langle \\boldsymbol{v}_{t+1}, \\boldsymbol{x}_t \\rangle\\,.
\\end{align}
```

Where:

  - $(math_dict[:M_t_hint]) ``M_{t+1}`` is the hint at `at_played = true`.
  - ``M_{t+1}^{\\mathrm{MP}}``: Hint at `at_played = false` under [`LogWealth`](@ref).
  - $(math_dict[:g_t_loss]) The optimistic step takes it at the played allocation ``\\boldsymbol{w}_t``.
  - $(math_dict[:w_t_iter])
  - $(math_dict[:x_t_rel]) Where the wrapped rule mixes, it is the mixed price relative that ``\\boldsymbol{g}_t`` reads.
  - $(math_dict[:v_t_sec])
  - $(math_dict[:t_period])
  - $(math_dict[:dual_norm_geo])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LastGradient(; at_played::Bool = true) -> LastGradient

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> LastGradient()
LastGradient
  at_played ┴ Bool: true
```

# Related

  - [`AbstractGradientPredictor`](@ref)
  - [`OptimisticStep`](@ref)
  - [`MeanGradient`](@ref)

# References

  - $(ref_dict[:chiang2012])
  - $(ref_dict[:rakhlin2013nips])
  - $(ref_dict[:nemirovski2004])
"""
struct LastGradient <: AbstractGradientPredictor
    """
    Whether the hint is the gradient at the played allocation. When it is `false`, the hint is the gradient of the loss of the period at the new secondary iterate.
    """
    at_played::Bool
end
function LastGradient(; at_played::Bool = true)::LastGradient
    return LastGradient(at_played)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes the running mean of the past gradients as the hint of the optimistic step.

Rakhlin and Sridharan [rakhlin2013colt](@cite) call the regret bound under this hint a variance bound. The carrier holds the mean, and the predictor updates it in one pass.

# Mathematical definition

```math
\\begin{align}
M_{t+1} &= \\frac{1}{t} \\sum_{s \\leq t} \\boldsymbol{g}_s\\,.
\\end{align}
```

Where:

  - $(math_dict[:M_t_hint])
  - $(math_dict[:g_t_loss])
  - $(math_dict[:t_period])

With ``M_1 = \\boldsymbol{0}``, the residual sum is at most the squared first gradient plus twice the spread of the gradients about their mean, ``\\sum_{t \\leq T} \\lVert \\boldsymbol{g}_t - M_t \\rVert_2^2 \\leq \\lVert \\boldsymbol{g}_1 \\rVert_2^2 + 2 \\sum_{t \\leq T} \\lVert \\boldsymbol{g}_t - \\bar{\\boldsymbol{g}}_T \\rVert_2^2``, with ``\\bar{\\boldsymbol{g}}_T`` the mean of the first ``T`` gradients. The maximum norm is at most the Euclidean norm, so the bound holds for the residuals in the maximum norm too.

# Examples

```jldoctest
julia> MeanGradient()
MeanGradient()
```

# Related

  - [`AbstractGradientPredictor`](@ref)
  - [`OptimisticStep`](@ref)
  - [`LastGradient`](@ref)

# References

  - $(ref_dict[:rakhlin2013colt])
"""
struct MeanGradient <: AbstractGradientPredictor end
"""
$(DocStringExtensions.TYPEDEF)

Forms the hint of the optimistic step from a Price Relative Forecast.

The hint is the gradient that the log-wealth loss of the next period would have at the secondary iterate, if the forecast were the outcome. The forecaster is any expected-returns estimator. The carrier folds it where it has an exact fold, and the predictor fits it again from the head's rows otherwise, as the `me` slot of [`ForecastReversion`](@ref) does. A flat forecast gives a hint with equal entries, and the projection onto the budget undoes the uniform shift of such a hint.

The optimistic-step papers take their hints from the past gradients, and this hint is not in them. It is the natural hint of the portfolio problem, where every reversion rule and every tracking rule already holds a forecast.

# Mathematical definition

```math
\\begin{align}
M_{t+1} &= -\\hat{\\boldsymbol{x}}_{t+1} / \\langle \\boldsymbol{v}_{t+1}, \\hat{\\boldsymbol{x}}_{t+1} \\rangle\\,.
\\end{align}
```

Where:

  - $(math_dict[:M_t_hint])
  - ``\\hat{\\boldsymbol{x}}_{t+1}``: Price Relative Forecast for period ``t + 1``, made after the row of period ``t``.
  - $(math_dict[:v_t_sec])
  - $(math_dict[:t_period])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastGradient(; me::AbstractExpectedReturnsEstimator) -> ForecastGradient

Keywords correspond to the struct's fields. The predictor needs as many rows as `me`, and the wrapper needs as many rows as the larger of the predictor and the wrapped rule.

## Validation

  - `me` is not an [`Online`](@ref) wrapper, and holds none. An `ArgumentError` is thrown otherwise.

# Examples

```jldoctest
julia> ForecastGradient(; me = PriceLevelExpectedReturns(; alg = MovingAverage(; window = 3)))
ForecastGradient
  me ┼ PriceLevelExpectedReturns
     │   alg ┼ MovingAverage
     │       │   window ┴ Int64: 3
```

# Related

  - [`AbstractGradientPredictor`](@ref)
  - [`OptimisticStep`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`PriorExpectedReturns`](@ref)
  - [`forecast_relative`](@ref)
"""
struct ForecastGradient{T1 <: AbstractExpectedReturnsEstimator} <: AbstractGradientPredictor
    """
    The forecaster whose Price Relative Forecast gives the hint.
    """
    me::T1
    function ForecastGradient(me::AbstractExpectedReturnsEstimator)
        assert_forecaster(me)
        return new{typeof(me)}(me)
    end
end
function ForecastGradient(; me::AbstractExpectedReturnsEstimator)::ForecastGradient
    return ForecastGradient(me)
end
function port_opt_view(pred::ForecastGradient, i, args...)
    return ForecastGradient(; me = port_opt_view(pred.me, i, args...))
end
function rows_needed(::AbstractGradientPredictor)
    return 0
end
function rows_needed(pred::ForecastGradient)
    return rows_needed(pred.me)
end
"""
    predictor_state_seed(pred::AbstractGradientPredictor, w::AbstractVector)
    predictor_state_seed(pred::MeanGradient, w::AbstractVector)
    predictor_state_seed(pred::ForecastGradient, w::AbstractVector)

Returns the carrier that a Gradient Predictor keeps on the Rule State before the first row. It is `nothing` by default, the zero mean under [`MeanGradient`](@ref), and the carrier of the forecaster from [`forecaster_seed`](@ref) under [`ForecastGradient`](@ref).

# Related

  - [`AbstractGradientPredictor`](@ref)
  - [`predict_gradient!`](@ref)
"""
function predictor_state_seed(::AbstractGradientPredictor, ::AbstractVector)
    return nothing
end
function predictor_state_seed(::MeanGradient, w::AbstractVector)
    return zeros(eltype(w), length(w))
end
function predictor_state_seed(pred::ForecastGradient, ::AbstractVector)
    return forecaster_seed(pred.me)
end
"""
    predict_gradient!(pred::LastGradient, ps::Nothing, obj::AbstractOnlineObjective, g::AbstractVector, v::AbstractVector, x::AbstractVector, xm::AbstractVector, rows, t::Integer)
    predict_gradient!(pred::MeanGradient, m::AbstractVector, obj::AbstractOnlineObjective, g::AbstractVector, v::AbstractVector, x::AbstractVector, xm::AbstractVector, rows, t::Integer)
    predict_gradient!(pred::ForecastGradient, ps, obj::AbstractOnlineObjective, g::AbstractVector, v::AbstractVector, x::AbstractVector, xm::AbstractVector, rows, t::Integer)

Returns the carrier after the period and the hint of the period, and writes the carrier in place.

  - Under [`LastGradient`](@ref) the hint is a copy of `g`. At `at_played = false` it is the gradient of `obj` at the secondary iterate `v`, on the mixed relative `xm` that `g` reads.
  - Under [`MeanGradient`](@ref) the method updates the running mean `m`, and the hint is a copy of it.
  - Under [`ForecastGradient`](@ref) the method folds or fits the forecaster on the traded relative `x` through [`forecast_relative`](@ref), and the hint is the log-wealth gradient of the forecast at `v`. It reads no objective, because a forecast is a price relative.

A forecaster folds the path that traded, not the mixed one, so the mix does not pull its statistic toward one. The rule applies no mix to the forecast.

# Returns

  - `(ps', M)::Tuple`: The carrier after the period, and the hint, a new vector.

# Related

  - [`AbstractGradientPredictor`](@ref)
  - [`predictor_state_seed`](@ref)
  - [`OptimisticStep`](@ref)
"""
function predict_gradient!(pred::LastGradient, ::Nothing, obj::AbstractOnlineObjective,
                           g::AbstractVector, v::AbstractVector, ::AbstractVector,
                           xm::AbstractVector, rows, ::Integer)
    if pred.at_played
        return nothing, copy(g)
    end
    return nothing, loss_gradient(obj, v, xm, rows)
end
function predict_gradient!(::MeanGradient, m::AbstractVector, ::AbstractOnlineObjective,
                           g::AbstractVector, ::AbstractVector, ::AbstractVector,
                           ::AbstractVector, ::Any, t::Integer)
    m .+= (g .- m) ./ t
    return m, copy(m)
end
function predict_gradient!(pred::ForecastGradient, ps, ::AbstractOnlineObjective,
                           ::AbstractVector, v::AbstractVector, x::AbstractVector,
                           ::AbstractVector, rows, ::Integer)
    ps, xhat = forecast_relative(pred.me, ps, x, rows)
    return ps, -xhat ./ LinearAlgebra.dot(v, xhat)
end
"""
    predictor_state_view(ps::Nothing, i)
    predictor_state_view(ps::AbstractVector, i)
    predictor_state_view(ps::AbstractPartialFitState, i)

Returns the carrier of a Gradient Predictor sliced to the assets `i`, as a copy.

# Related

  - [`AbstractGradientPredictor`](@ref)
  - [`OptimisticStepState`](@ref)
"""
function predictor_state_view(::Nothing, ::Any)
    return nothing
end
function predictor_state_view(ps::AbstractVector, i)
    return ps[i]
end
function predictor_state_view(ps::AbstractPartialFitState, i)
    return port_opt_view(ps, i)
end
"""
    copy_predictor_state(ps::Nothing)
    copy_predictor_state(ps::AbstractVector)
    copy_predictor_state(ps::AbstractPartialFitState)

Returns a copy of the carrier of a Gradient Predictor that shares no array with it.

# Related

  - [`AbstractGradientPredictor`](@ref)
  - [`OptimisticStepState`](@ref)
"""
function copy_predictor_state(::Nothing)
    return nothing
end
function copy_predictor_state(ps::AbstractVector)
    return copy(ps)
end
function copy_predictor_state(ps::AbstractPartialFitState)
    return copy(ps)
end
"""
$(DocStringExtensions.TYPEDEF)

Sets the learning rate of an optimistic step from the hint residuals that the step has accrued.

This is the adaptive rate of Corollary 2 of [rakhlin2013nips](@cite). It needs no horizon and no bound on the residual sum ahead of time. The schedule reads the residuals that an [`OptimisticStep`](@ref) keeps on its carrier, and it refuses the carrier of any other rule by name. [`hint_residual`](@ref) states the norm of the residuals in each geometry.

# Mathematical definition

```math
\\begin{align}
S_t &= \\sum_{i \\leq t} \\lVert \\boldsymbol{g}_i - M_i \\rVert_*^2\\,, \\\\
\\eta_t &= R_{\\max} \\min \\left\\lbrace \\left( \\sqrt{S_{t-1}} + \\sqrt{S_{t-2}} \\right)^{-1},\\, 1 \\right\\rbrace\\,.
\\end{align}
```

Where:

  - $(math_dict[:eta_t_lr])
  - $(math_dict[:t_period])
  - ``S_t``: Sum of the squared hint residuals through period ``t``, zero for ``t \\leq 0``.
  - $(math_dict[:g_t_loss])
  - $(math_dict[:M_t_hint])
  - $(math_dict[:dual_norm_geo])
  - ``R_{\\max}``: Scale and cap of the rate. The paper sets ``R_{\\max}^2 = \\sup_{\\boldsymbol{w}, \\boldsymbol{w}' \\in \\mathcal{W}} D_\\Psi(\\boldsymbol{w}, \\boldsymbol{w}')``.
  - $(math_dict[:W_aset])
  - $(math_dict[:D_Psi_breg])

The rate is the cap ``R_{\\max}`` while ``S_{t-1} = 0``. With ``R_{\\max}`` at that supremum, the paper bounds the regret over ``T`` periods by ``3.5 R_{\\max} (\\sqrt{S_T} + 1)``. Under the Euclidean geometry on the simplex the supremum is one, the default `rmax`. Under the entropic geometry the supremum is infinite, and the uniform mix `alpha` of the wrapped rule does not bound it, because the mix applies to the played allocation and not to the secondary iterate. There `rmax` is a tuning constant, and the bound does not hold.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HintResidualRate(; rmax::Real = 1) -> HintResidualRate

Keywords correspond to the struct's fields.

## Validation

  - `rmax > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> HintResidualRate()
HintResidualRate
  rmax ┴ Int64: 1
```

# Related

  - [`AbstractLearningRateSchedule`](@ref)
  - [`OptimisticStep`](@ref)
  - [`InverseSquareRootRate`](@ref)

# References

  - $(ref_dict[:rakhlin2013nips])
"""
struct HintResidualRate{T1 <: Real} <: AbstractLearningRateSchedule
    """
    The paper's ``R_{\\max}``, which scales the rate and caps it.
    """
    rmax::T1
    function HintResidualRate(rmax::Real)
        @argcheck(rmax > zero(rmax), DomainError(rmax, "rmax must be positive"))
        return new{typeof(rmax)}(rmax)
    end
end
function HintResidualRate(; rmax::Real = 1)::HintResidualRate
    return HintResidualRate(rmax)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the two iterates of an optimistic step, its hint residuals and the carriers of its schedule and its predictor.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`OptimisticStep`](@ref)
  - [`MirrorDescentState`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct OptimisticStepState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The secondary iterate ``\\boldsymbol{v}_t`` over the full pinned universe, summing to one. The first half-step of the period starts from it.
    """
    v
    """
    The unmixed played iterate ``\\boldsymbol{w}_t``, at which the rule reads the gradient of the period.
    """
    u
    """
    The Start Allocation projected onto the Allocation Set. A restart returns both iterates to it.
    """
    w0
    """
    The schedule statistic of the wrapped rule, or `nothing`.
    """
    s
    """
    The sums of the squared hint residuals through the last period and the period before it, ``[S_{t-1}, S_{t-2}]`` at period ``t``. [`HintResidualRate`](@ref) reads them.
    """
    res
    """
    The hint ``M_t`` along which the rule stepped the played iterate, `assets × 1`. The rule measures the gradient of the period against it.
    """
    m
    """
    The carrier of the Gradient Predictor, or `nothing`.
    """
    ps
end
function merge_states(::OptimisticStepState, ::OptimisticStepState)
    return throw(ArgumentError("an `OptimisticStepState` is not merged on its own: it holds two iterates that are order-dependent, so the head's state refuses the merge, and the carrier follows it."))
end
function Base.copy(x::OptimisticStepState)
    return OptimisticStepState(x.n, copy(x.v), copy(x.u), copy(x.w0), copy_column(x.s),
                               copy(x.res), copy(x.m), copy_predictor_state(x.ps))
end
function port_opt_view(x::OptimisticStepState, i, args...)
    return OptimisticStepState(x.n, renormalised_view(x.v, i), renormalised_view(x.u, i),
                               renormalised_view(x.w0, i), schedule_state_view(x.s, i),
                               copy(x.res), x.m[i], predictor_state_view(x.ps, i))
end
function learning_rate(sched::HintResidualRate, ::Integer, st::OptimisticStepState)
    a = sqrt(st.res[1]) + sqrt(st.res[2])
    if !(a > zero(a))
        return sched.rmax
    end
    return sched.rmax * min(inv(a), one(a))
end
function learning_rate(::HintResidualRate, ::Integer, st)
    return throw(ArgumentError("a `HintResidualRate` reads the hint residuals an `OptimisticStep` keeps on its carrier, and a `$(typeof(st).name.name)` holds none: the schedule is the optimistic step's, on the `eta` of the rule it wraps."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the squared hint residual of a period in the dual norm of the geometry.

A [`MirrorDescent`](@ref) rule takes only the four geometries below, so an [`OptimisticStep`](@ref) never reads the residual under [`GramProjection`](@ref) or [`DiagonalProjection`](@ref), and no method serves them.

# Mathematical definition

```math
\\begin{align}
r_{\\mathrm{E}} &= \\lVert \\boldsymbol{g}_t - M_t \\rVert_2^2\\,, \\\\
r_{\\infty} &= \\lVert \\boldsymbol{g}_t - M_t \\rVert_\\infty^2\\,.
\\end{align}
```

Where:

  - ``r_{\\mathrm{E}}``: Squared residual under [`EuclideanProjection`](@ref).
  - ``r_{\\infty}``: Squared residual under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref).
  - $(math_dict[:g_t_loss])
  - $(math_dict[:M_t_hint])
  - $(math_dict[:t_period])

Each is the square of the dual norm ``\\lVert \\boldsymbol{g}_t - M_t \\rVert_*``. On the simplex the Euclidean potential is 1-strongly convex against the ``\\ell_2`` norm, whose dual is the ``\\ell_2`` norm. The entropic, Tsallis and log-barrier potentials are strongly convex against the ``\\ell_1`` norm with a modulus of at least one, and the dual of the ``\\ell_1`` norm is the ``\\ell_\\infty`` norm.

# Related

  - [`HintResidualRate`](@ref)
  - [`OptimisticStepState`](@ref)
"""
function hint_residual(::EuclideanProjection, d::AbstractVector)
    return sum(abs2, d)
end
function hint_residual(::Union{<:EntropicProjection, <:TsallisProjection,
                               <:LogBarrierProjection}, d::AbstractVector)
    return abs2(maximum(abs, d))
end
"""
$(DocStringExtensions.TYPEDEF)

Plays two half-steps of a first-order rule each period, the second along a hint for the gradient of the next period (OMD).

This is the optimistic mirror descent of Rakhlin and Sridharan (2013), as a wrapper over a [`MirrorDescent`](@ref) rule. The carrier holds two iterates. The secondary iterate is where the step of the next period starts, and the played allocation is one hint step ahead of it.

`eta`, `proj`, `alpha` and `obj` are those of the wrapped rule. Both half-steps are its mirror step in its geometry, and the gradient is that of its objective, through [`loss_gradient`](@ref). A [`RiskLoss`](@ref) reads the head's rows, and the wrapper needs as many rows as the larger of the objective and the predictor. A schedule on `eta` counts the periods of the wrapper and reads its carrier. The wrapper reads the uniform mix `alpha` as the wrapped rule does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{v}_{t+1} &= \\underset{\\boldsymbol{v} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta_t \\langle \\boldsymbol{g}_t, \\boldsymbol{v} \\rangle + D_\\Psi(\\boldsymbol{v}, \\boldsymbol{v}_t)\\,, \\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta_{t+1} \\langle M_{t+1}, \\boldsymbol{w} \\rangle + D_\\Psi(\\boldsymbol{w}, \\boldsymbol{v}_{t+1})\\,.
\\end{align}
```

Where:

  - $(math_dict[:v_t_sec])
  - $(math_dict[:w_t_iter]) Here it is the played allocation before the uniform mix.
  - $(math_dict[:g_t_loss]) The rule takes it at ``\\boldsymbol{w}_t``. Under [`LogWealth`](@ref) it is ``-\\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle``.
  - $(math_dict[:x_t_rel])
  - $(math_dict[:M_t_hint])
  - $(math_dict[:eta_t_lr])
  - $(math_dict[:t_period])
  - $(math_dict[:W_aset])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:dual_norm_geo])

The rates follow equation (1) of [rakhlin2013nips](@cite), which forms ``\\boldsymbol{w}_{t+1}`` at ``\\eta_{t+1}``. A fixed rate is one rate for both half-steps. Under the zero hint the second half-step returns ``\\boldsymbol{v}_{t+1}``, so the rule is the wrapped rule.

For a fixed rate ``\\eta`` and a comparator ``\\boldsymbol{w}^*`` with ``D_\\Psi(\\boldsymbol{w}^*, \\boldsymbol{v}_1) \\leq R^2``, Lemma 1 of [rakhlin2013nips](@cite), with its equation (3) at ``\\rho = \\eta``, bounds the regret by ``\\eta^{-1} R^2 + \\tfrac{\\eta}{2} \\sum_t \\lVert \\boldsymbol{g}_t - M_t \\rVert_*^2``. At the rate tuned to the residual sum, the bound is ``O(\\sqrt{\\sum_t \\lVert \\boldsymbol{g}_t - M_t \\rVert_*^2})``. The residual sum is the path length of the gradients under [`LastGradient`](@ref), and the spread of the gradients about their mean bounds it under [`MeanGradient`](@ref). [rakhlin2013colt](@cite) states that a hint that misses keeps the bound of mirror descent up to a constant, for hints no larger than the gradients in the dual norm. [`HintResidualRate`](@ref) on the `eta` of the wrapped rule is the adaptive rate of [rakhlin2013nips](@cite).

# Algorithm

The seven-argument [`online_update!`](@ref) runs these steps at the period's row `x`. The six-argument form calls it with the unmixed played iterate `st.u` as the Gradient Point `point`.

 1. Set the period `t = st.n + 1`, and form the Price-Adjusted Allocation `wh` of the book `w` after the row.
 2. When the schedule of `eta` names a restart at `t`, re-enter the Start Allocation `st.w0` onto the set from `wh` with [`reprojection`](@ref). Put both iterates at that allocation and every carrier at its seed, keep `t`, and return that allocation. Stop.
 3. When the schedule reads the period's row before the rate, write its statistic from `w` and `x`.
 4. Read the rate `eta` and the share `alpha` of period `t`.
 5. Mix the price relatives into `xm` with [`mixed_relatives`](@ref).
 6. Take the gradient `g` of `obj` at `point`, from `xm` and the head's `rows`, with [`loss_gradient`](@ref).
 7. Take the first half-step from the secondary iterate `st.v` on `eta .* g` with [`half_step`](@ref), which gives the new secondary iterate `v`.
 8. Form the hint `m` and the carrier `ps` of the predictor with [`predict_gradient!`](@ref), from `g`, `v`, `x` and `xm`.
 9. Measure the residual `r` of `g` against the last hint `st.m` with [`hint_residual`](@ref), and shift the residual sums into `res`.
10. When the schedule reads the past alone, write its statistic `s` from `w` and `x`.
11. Read the rate of period `t + 1` from a carrier that holds the period's row. Take the second half-step from `v` on that rate times `m`, which gives the unmixed played iterate `u`.
12. Form the played allocation from `u` with [`played_allocation`](@ref). When `alpha` is positive, project it with [`reprojection`](@ref).
13. Return the new state and the played allocation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimisticStep(;
        alg::MirrorDescent = MirrorDescent(),
        predictor::AbstractGradientPredictor = LastGradient()
    ) -> OptimisticStep

Keywords correspond to the struct's fields. The Gradient Transform of the wrapped rule must be the identity. A transform keeps a carrier that one gradient writes each period, and the optimistic step reads two gradients a period.

## Validation

  - `alg.grad` is a [`PlainGradient`](@ref). An `ArgumentError` is thrown otherwise.

# Examples

```jldoctest
julia> OptimisticStep()
OptimisticStep
        alg ┼ MirrorDescent
            │     eta ┼ Float64: 0.05
            │    proj ┼ EntropicProjection()
            │   alpha ┼ Int64: 0
            │     obj ┼ LogWealth()
            │    grad ┴ PlainGradient()
  predictor ┼ LastGradient
            │   at_played ┴ Bool: true
```

# Related

  - [`AbstractOnlinePortfolioSelectionAlgorithm`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`OptimisticStepState`](@ref)
  - [`AbstractGradientPredictor`](@ref)
  - [`HintResidualRate`](@ref)

# References

  - $(ref_dict[:rakhlin2013colt])
  - $(ref_dict[:rakhlin2013nips])
  - $(ref_dict[:chiang2012])
"""
struct OptimisticStep{T1 <: MirrorDescent, T2 <: AbstractGradientPredictor} <:
       AbstractOnlinePortfolioSelectionAlgorithm
    """
    The first-order rule that takes both half-steps.
    """
    alg::T1
    """
    The Gradient Predictor whose hint the second half-step follows.
    """
    predictor::T2
    function OptimisticStep(alg::MirrorDescent, predictor::AbstractGradientPredictor)
        @argcheck(isa(alg.grad, PlainGradient),
                  ArgumentError("the optimistic step wraps a `MirrorDescent` whose Gradient Transform is the identity: a `$(typeof(alg.grad).name.name)` keeps a carrier written once a period from one gradient, and the optimistic step reads two. Wrap the rule with `grad = PlainGradient()`."))
        return new{typeof(alg), typeof(predictor)}(alg, predictor)
    end
end
function OptimisticStep(; alg::MirrorDescent = MirrorDescent(),
                        predictor::AbstractGradientPredictor = LastGradient())::OptimisticStep
    return OptimisticStep(alg, predictor)
end
function port_opt_view(alg::OptimisticStep, i, args...)
    return OptimisticStep(; alg = alg.alg,
                          predictor = port_opt_view(alg.predictor, i, args...))
end
function projection_geometry(alg::OptimisticStep)
    return alg.alg.proj
end
function rows_needed(alg::OptimisticStep)
    return rows_needed_max(rows_needed(alg.predictor), rows_needed(alg.alg))
end
function rule_state_seed(alg::OptimisticStep, w::AbstractVector)
    return OptimisticStepState(0, copy(w), copy(w), copy(w),
                               schedule_state_seed(alg.alg.eta, w), zeros(eltype(w), 2),
                               zeros(eltype(w), length(w)),
                               predictor_state_seed(alg.predictor, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Takes one half-step of an [`OptimisticStep`](@ref). This is the mirror step of the wrapped rule from `u` on the scaled gradient `s`, projected onto the set in the rule's geometry from the Price-Adjusted Allocation `w`.

# Related

  - [`OptimisticStep`](@ref)
  - [`mirror_step`](@ref)
  - [`project`](@ref)
"""
function half_step(alg::MirrorDescent, set::AbstractAllocationSet, u::AbstractVector,
                   s::AbstractVector, w::AbstractVector)
    return project(alg.proj, set, mirror_step(alg.proj, u, s), w)
end
function online_update!(alg::OptimisticStep, st::OptimisticStepState, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet)
    return online_update!(alg, st, w, x, rows, set, st.u)
end
# The seven-argument form is the primitive: the gradient is read at `point`, the unmixed
# played iterate on the head and the mixture's played blend under `BlendPoint`; the hint is
# the predictor's and reads the secondary iterate as before. The predictor is handed both the
# traded relative and the mixed one: a forecaster folds the path that traded, and a
# re-evaluated loss reads the vector the gradient was read on (#1210).
function online_update!(alg::OptimisticStep, st::OptimisticStepState, w::AbstractVector,
                        x::AbstractVector, rows, set::AbstractAllocationSet,
                        point::AbstractVector)
    md = alg.alg
    t = st.n + 1
    wh = price_adjusted_allocation(w, x)
    if restart(md.eta, t)
        # The period count survives the restart: the schedule's stages are cumulative. The
        # start re-enters the set from the book the fund holds, as it entered at the seed.
        u0 = reprojection(md.proj, set, st.w0, wh)
        return OptimisticStepState(t, copy(u0), copy(u0), st.w0,
                                   schedule_state_seed(md.eta, st.w0), zero(st.res),
                                   zero(st.m), predictor_state_seed(alg.predictor, st.w0)),
               copy(u0)
    end
    # The schedule's row enters before the rate or after the step, as the schedule says.
    st = OptimisticStepState(st.n, st.v, st.u, st.w0,
                             statistic_before_rate(md.eta, st.s, w, x), st.res, st.m, st.ps)
    eta = learning_rate(md.eta, t, st)
    alpha = mixing_share(md.eta, t, md.alpha)
    xm = mixed_relatives(x, alpha)
    g = loss_gradient(md.obj, point, xm, rows)
    v = half_step(md, set, st.v, eta .* g, wh)
    ps, m = predict_gradient!(alg.predictor, st.ps, md.obj, g, v, x, xm, rows, t)
    r = hint_residual(md.proj, g .- st.m)
    res = [st.res[1] + r, st.res[1]]
    s = statistic_after_step(md.eta, st.s, w, x)
    # The played iterate is formed at the rate of the next period, which the schedule
    # answers from the carrier after the row: the paper's `f_{t+1}` at `η_{t+1}`.
    nst = OptimisticStepState(t, v, v, st.w0, s, res, m, ps)
    u = half_step(md, set, v, learning_rate(md.eta, t + 1, nst) .* m, wh)
    played = played_allocation(u, alpha)
    return OptimisticStepState(t, v, u, st.w0, s, res, m, ps),
           iszero(alpha) ? played : reprojection(md.proj, set, played, wh)
end
export AdaptiveSubgradient, OptimisticStep, LastGradient, MeanGradient, ForecastGradient,
       HintResidualRate
public AbstractGradientPredictor, predictor_state_seed, predict_gradient!
