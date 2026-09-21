"""
$(DocStringExtensions.TYPEDEF)

The diagonal Projection Geometry: the raw step is projected onto the Allocation Set in the norm of a positive diagonal matrix, ``\\min \\tfrac{1}{2} (\\boldsymbol{w} - \\boldsymbol{q})^\\intercal \\mathrm{diag}(\\boldsymbol{h}) (\\boldsymbol{w} - \\boldsymbol{q})``.

It is the geometry of the diagonal adaptive subgradient method (Duchi, Hazan and Singer 2011), whose proximal term is ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{w}_t \\rVert^2_{H_t}`` with ``H_t = \\delta I + \\mathrm{diag}(\\boldsymbol{s}_t)`` the accrued gradient mass. The weight vector is the rule's: [`AdaptiveSubgradient`](@ref) constructs the geometry from its carrier's ``\\boldsymbol{h}_t`` at every step, as [`NewtonStep`](@ref) binds its Gram matrix onto [`GramProjection`](@ref). [`EuclideanProjection`](@ref) is the case ``\\boldsymbol{h} = \\boldsymbol{1}``: on a [`BoundedAllocationSet`](@ref) the projection is the weighted scalar root ``w_i = \\mathrm{clip}(q_i - \\theta / h_i, lb_i, ub_i)`` with ``\\theta`` the budget's root, which shares its bisection with the Euclidean one ([`bounded_quadratic_projection`](@ref)) and costs what it costs; on a [`ProgrammeAllocationSet`](@ref) it is the weighted quadratic programme through a second-order cone, on the set's solver. The Start Allocation is projected before any gradient is seen, where no mass has accrued, so [`projection_geometry`](@ref) answers [`EuclideanProjection`](@ref) for it. A negative lower bound is admitted, as under the Euclidean geometry.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DiagonalProjection(h::AbstractVector) -> DiagonalProjection

The positional argument is the struct's field. The rule constructs the geometry at each step; a caller never holds one.

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
  - [`AdaptiveSubgradient`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:duchi2011])
"""
struct DiagonalProjection{T1 <: AbstractVector} <: AbstractProjectionGeometry
    """
    The diagonal of the norm the projection is taken in, the rule's accrued gradient mass at the step.
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

The diagonal arms of the Constrained Update: the weighted scalar root of [`bounded_quadratic_projection`](@ref) on the bounded set, on every bound, the simplex included, because the weights break the sort; and the bare-model programme of [`projection_programme`](@ref) on the programme set.

# Related

  - [`project`](@ref)
  - [`DiagonalProjection`](@ref)
  - [`bounded_quadratic_projection`](@ref)
  - [`projection_programme`](@ref)
"""
function project(proj::DiagonalProjection, set::BoundedAllocationSet, q::AbstractVector,
                 ::AbstractVector)
    return bounded_quadratic_projection(q, set.wb, proj.h)
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
    JuMP.@constraint(model, proj_soc,
                     [sc * t_proj; sc * (sqrt.(proj.h) .* (w - q))] in
                     JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t_proj)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

The carrier of [`AdaptiveSubgradient`](@ref): the per-asset gradient mass.

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
    The gradient mass ``\\boldsymbol{s}_t``, ``s_{t, i}^2 = \\sum_{s \\leq t} g_{s, i}^2``, `assets × 1`, written in place.
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

The diagonal adaptive subgradient method of Duchi, Hazan and Singer (2011) in its composite-mirror-descent form (their Algorithm 1 and Corollary 6): a first-order step whose rate is set per asset by the gradient mass that asset has accrued (AdaGrad).

# Mathematical definition

With ``\\boldsymbol{g}_t = -\\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` the gradient of the period's log-wealth loss,

```math
\\begin{align}
s_{t, i}^2 &= \\sum_{s \\leq t} g_{s, i}^2\\,,\\quad
H_t = \\delta I + \\mathrm{diag}(\\boldsymbol{s}_t)\\,,\\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta \\langle \\boldsymbol{g}_t, \\boldsymbol{w} \\rangle + \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{w}_t \\rVert^2_{H_t} = \\mathrm{Proj}^{H_t}_{\\mathcal{W}} \\left( \\boldsymbol{w}_t - \\eta H_t^{-1} \\boldsymbol{g}_t \\right)\\,,
\\end{align}
```

the raw step ``\\boldsymbol{w}_t + \\eta \\boldsymbol{x}_t / (\\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle (\\delta + \\boldsymbol{s}_t))`` followed by the projection onto the Allocation Set **in the norm of ``H_t``**, the [`DiagonalProjection`](@ref) the rule constructs from its accrued gradient mass at every step. The regret is ``O(\\sum_i \\lVert \\boldsymbol{g}_{1:T, i} \\rVert_2)`` (Corollary 6 at ``\\eta = 1 / \\sqrt{2}`` on a set of ``\\ell_\\infty`` diameter one), small when the gradients are sparse or a few assets carry most of their mass, and never worse than online gradient descent's ``O(\\sqrt{T N})`` by more than a constant. The rule is [`NewtonStep`](@ref) one rank down: the full-matrix variant of the paper is the Newton carrier under a square root, and the diagonal keeps the step at ``O(N)``.

The per-coordinate normalisation cancels the first gradient exactly, so the first raw step is a uniform shift of the Start Allocation; the projection in the norm of ``H_1 = \\delta I + \\mathrm{diag}(\\lvert \\boldsymbol{g}_1 \\rvert)`` then lets the assets with the smaller gradient absorb more of the shift back, so the first step moves weight toward the assets that rose the most, and is a no-op only when every price relative is equal. At `delta = 0`, the paper's default, the projection's diagonal is the gradient mass alone, which is positive from the first row in every asset whose price relative is positive; a price relative of zero in an asset that has never moved gives that asset no mass, and the projection refuses it. A zero gradient — a [`RiskLoss`](@ref) while the head holds fewer than two rows — is a zero step: no mass accrues, and the iterate is projected in the Euclidean geometry, as the zero step of every first-order rule is.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AdaptiveSubgradient(;
        eta::Real = 1 / sqrt(2),
        delta::Real = 0,
        obj::AbstractOnlineObjective = LogWealth()
    ) -> AdaptiveSubgradient

Keywords correspond to the struct's fields. The defaults are the paper's. The rule projects in its own geometry, a [`DiagonalProjection`](@ref) it constructs at each step, so it holds no `proj` slot. `obj` is the slot of [`MirrorDescent`](@ref): log wealth, or a [`RiskLoss`](@ref) over the head's rows, whose gradient [`loss_gradient`](@ref) answers at the iterate; the rule's `rows_needed` is the objective's.

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
    Learning rate. Larger reacts faster and is less stable.
    """
    eta::T1
    """
    The paper's ``\\delta``, added to every asset's gradient mass; `0` is the paper's rule.
    """
    delta::T2
    """
    The objective the gradient is taken of: log wealth, or a Risk Loss over the head's rows.
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

Abstract supertype for the Gradient Predictors an [`OptimisticStep`](@ref) plays its second half-step along: the hint ``M_{t+1}`` for the gradient of the period not yet seen.

# Interfaces

In order to implement a new predictor, subtype `AbstractGradientPredictor` and implement:

  - `predictor_state_seed(pred::AbstractGradientPredictor, w::AbstractVector)`: The carrier the predictor keeps on the Rule State before the first row, or `nothing`, the default; `w` is the Start Allocation, whose length and element type the carrier takes.
  - `predict_gradient!(pred::AbstractGradientPredictor, ps, obj::AbstractOnlineObjective, g::AbstractVector, v::AbstractVector, x::AbstractVector, xm::AbstractVector, rows, t::Integer) -> Tuple`: The carrier after the period and the hint, from the wrapped rule's objective `obj`, the period's gradient `g` at the played allocation, the secondary iterate `v` the second half-step is taken from, the period's price relative `x` as traded, the same relative `xm` with the uniform mix applied where the wrapped rule mixes — the vector `g` was read on, and `x` itself where the rule does not mix — the rows the head holds through the period and the period count `t`. A predictor that reads the relative as a price folds `x`; one that re-evaluates the wrapped rule's loss reads `xm`, so the hint sees the loss `g` saw.
  - `rows_needed(pred::AbstractGradientPredictor)`: The rows the predictor reads at a step, `0` by default.

A carrier that is `nothing`, a vector or a Partial Fit State is sliced and copied with the head's state already; a carrier of another shape needs a slice and a copy of its own, through the two private helpers [`OptimisticStepState`](@ref) names.

# Related

  - [`LastGradient`](@ref)
  - [`MeanGradient`](@ref)
  - [`ForecastGradient`](@ref)
  - [`OptimisticStep`](@ref)
"""
abstract type AbstractGradientPredictor <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The last gradient as the hint: ``M_{t+1} = \\boldsymbol{g}_t``, the gradient of the period's loss at the played allocation, under which the optimistic step is the two-projection algorithm of Chiang, Yang, Lee, Mahdavi, Lu, Jin and Zhu (2012) and the regret sum ``\\sum_t \\lVert \\boldsymbol{g}_t - M_t \\rVert_*^2`` is the path length of the gradients. The default of [`OptimisticStep`](@ref).

With `at_played = false` the hint is the period's loss re-evaluated at the new secondary iterate through [`loss_gradient`](@ref), ``M_{t+1} = -\\boldsymbol{x}_t / \\langle \\boldsymbol{v}_{t+1}, \\boldsymbol{x}_t \\rangle`` under log wealth: the Mirror-Prox form of Nemirovski (2004), which Rakhlin and Sridharan (2013) recover from the optimistic step by taking the hint at the secondary point. The two coincide when the played and the secondary allocations do, so they part only after the first period.

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
    Whether the hint is the gradient at the played allocation, or the period's loss re-evaluated at the new secondary iterate.
    """
    at_played::Bool
end
function LastGradient(; at_played::Bool = true)::LastGradient
    return LastGradient(at_played)
end
"""
$(DocStringExtensions.TYPEDEF)

The running mean of the gradients as the hint: ``M_{t+1} = \\frac{1}{t} \\sum_{s \\leq t} \\boldsymbol{g}_s``, under which the regret sum of Rakhlin and Sridharan (2013) is the variance of the gradients about their mean, the bound of Hazan and Kale (2010) up to a constant. The mean is kept on the carrier and updated in one pass.

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

A gradient formed from a Price Relative Forecast as the hint: ``M_{t+1} = -\\hat{\\boldsymbol{x}}_{t+1} / \\langle \\boldsymbol{v}_{t+1}, \\hat{\\boldsymbol{x}}_{t+1} \\rangle``, the gradient the next period's log-wealth loss would have at the secondary iterate the second half-step is taken from, were the forecast the outcome. The forecaster is any expected-returns estimator, folded on the carrier where it has an exact fold and refit from the head's rows otherwise, exactly as on the `me` slot of [`ForecastReversion`](@ref); a flat forecast is a zero hint up to a constant, on which the second half-step is a uniform shift the projection undoes. Not in the optimistic-step papers, whose hints are functions of the past gradients: the natural hint of the portfolio problem, where every reversion and tracking rule already holds one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForecastGradient(; me::AbstractExpectedReturnsEstimator) -> ForecastGradient

Keywords correspond to the struct's fields. `rows_needed` forwards to `me`; the wrapper's is the larger of the predictor's and the wrapped rule's.

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
    The forecaster whose Price Relative Forecast the hint is formed from.
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

The carrier a Gradient Predictor keeps on the Rule State before the first row: `nothing`, the zero mean, or the forecaster's own carrier through [`forecaster_seed`](@ref).

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

The hint of the period, the carrier written in place: the gradient `g` itself or the period's loss `obj` at the secondary iterate `v` on the mixed relative `xm` the gradient `g` was read on, the updated running mean `m`, or the log-wealth gradient of the forecast at `v` from the forecaster folded on or refit through [`forecast_relative`](@ref) on the traded relative `x`, which reads no objective because a forecast is a price relative. A forecaster folds the path that traded, not the mixed one, so its statistic is not compressed toward one by the mix; the hint `-x̂ / ⟨v, x̂⟩` is scale-free, so the forecast needs no mix applied after.

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

A Gradient Predictor's carrier sliced to the assets `i`, as a copy.

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

A copy of a Gradient Predictor's carrier sharing no array with it.

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

The adaptive rate of Rakhlin and Sridharan (2013, Corollary 2), read from the hint residuals an [`OptimisticStep`](@ref) keeps on its carrier:

```math
\\begin{align}
\\eta_t &= R_{\\max} \\min \\left\\lbrace \\left( \\sqrt{\\textstyle\\sum_{i \\leq t-1} \\lVert \\boldsymbol{g}_i - M_i \\rVert_*^2} + \\sqrt{\\textstyle\\sum_{i \\leq t-2} \\lVert \\boldsymbol{g}_i - M_i \\rVert_*^2} \\right)^{-1},\\, 1 \\right\\rbrace\\,,
\\end{align}
```

with ``R_{\\max}^2`` the diameter of the Allocation Set in the geometry's divergence, which the caller supplies: one for the Euclidean geometry on the simplex, and for the entropic geometry a bound the uniform mix `alpha` of the wrapped rule makes finite. Under it the regret is ``3.5 R_{\\max} \\sqrt{\\sum_t \\lVert \\boldsymbol{g}_t - M_t \\rVert_*^2 + 1}`` with no horizon and no residual sum known ahead. The residual is measured in the dual norm the geometry's potential is strongly convex against: the Euclidean norm under [`EuclideanProjection`](@ref), and the maximum norm under the entropic and barrier maps, whose potentials are taken against the ``\\ell_1`` norm. The rate is the cap ``R_{\\max}`` while the residual sum is zero.

The schedule reads the residuals an [`OptimisticStep`](@ref) alone accrues, and refuses any other rule's carrier by name.

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
    The paper's ``R_{\\max}``, the root of the set's diameter in the geometry's divergence, which scales the rate and caps it.
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

The carrier of [`OptimisticStep`](@ref): the secondary iterate, the wrapped rule's own carrier laid flat beside it, the last hint and the running hint residuals, and the Gradient Predictor's carrier.

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
    The secondary iterate ``\\boldsymbol{v}_t``, over the full pinned universe, summing to one, which both half-steps are taken from.
    """
    v
    """
    The unmixed played iterate ``\\boldsymbol{w}_t``, at which the period's gradient is read.
    """
    u
    """
    The Start Allocation projected onto the Allocation Set, which a restart returns both iterates to.
    """
    w0
    """
    The wrapped rule's schedule statistic, or `nothing`.
    """
    s
    """
    The running sums of the squared hint residuals through the last period and the one before, `[Σ_{i ≤ t-1}, Σ_{i ≤ t-2}]`, which [`HintResidualRate`](@ref) reads.
    """
    res
    """
    The hint ``M_t`` the played iterate was stepped along, `assets × 1`, against which the period's gradient is measured.
    """
    m
    """
    The Gradient Predictor's carrier, or `nothing`.
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

The squared hint residual ``\\lVert \\boldsymbol{g}_t - M_t \\rVert_*^2`` of a period in the dual norm of the geometry: the Euclidean norm under [`EuclideanProjection`](@ref), the maximum norm under every other map.

# Related

  - [`HintResidualRate`](@ref)
  - [`OptimisticStepState`](@ref)
"""
function hint_residual(::EuclideanProjection, d::AbstractVector)
    return sum(abs2, d)
end
function hint_residual(::AbstractProjectionGeometry, d::AbstractVector)
    return abs2(maximum(abs, d))
end
"""
$(DocStringExtensions.TYPEDEF)

The optimistic mirror descent of Rakhlin and Sridharan (2013): a wrapper over a [`MirrorDescent`](@ref) rule that plays two half-steps a period, the second along a hint for the gradient not yet seen (OMD).

# Mathematical definition

With ``\\boldsymbol{g}_t = -\\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` the gradient of the period's log-wealth loss at the played allocation, ``M_{t+1}`` the Gradient Predictor's hint, ``D_\\Psi`` the divergence of the wrapped rule's Projection Geometry and ``\\eta`` its rate,

```math
\\begin{align}
\\boldsymbol{v}_{t+1} &= \\underset{\\boldsymbol{v} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta \\langle \\boldsymbol{g}_t, \\boldsymbol{v} \\rangle + D_\\Psi(\\boldsymbol{v}, \\boldsymbol{v}_t)\\,,\\\\
\\boldsymbol{w}_{t+1} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; \\eta \\langle M_{t+1}, \\boldsymbol{w} \\rangle + D_\\Psi(\\boldsymbol{w}, \\boldsymbol{v}_{t+1})\\,.
\\end{align}
```

The secondary sequence ``\\boldsymbol{v}_t`` is what the next period's step continues from, and the played ``\\boldsymbol{w}_t`` is one hint-step ahead of it; the carrier holds both. The regret is ``\\eta^{-1} R_{\\max}^2 + \\tfrac{\\eta}{2} \\sum_t \\lVert \\boldsymbol{g}_t - M_t \\rVert_*^2`` (their Lemma 2), ``O(\\sqrt{\\sum_t \\lVert \\boldsymbol{g}_t - M_t \\rVert_*^2})`` at the rate tuned to the residuals — the gradient path length under [`LastGradient`](@ref), where the step is the two-projection algorithm of Chiang and co-authors (2012), and the gradient variance under [`MeanGradient`](@ref) — and never worse than the wrapped rule's bound by more than a constant when the hints are wrong. A zero hint is the wrapped rule exactly. [`HintResidualRate`](@ref) on the wrapped rule's `eta` is the paper's adaptive rate, which reads the residuals the carrier accrues.

`eta`, `proj`, `alpha` and `obj` are the wrapped rule's: both half-steps are its mirror step in its geometry, the gradient is its objective's through [`loss_gradient`](@ref) — a [`RiskLoss`](@ref) reads the head's rows, and the wrapper's `rows_needed` is the larger of the objective's and the predictor's — a schedule on its `eta` counts the wrapper's periods and reads the wrapper's carrier, and the uniform mix `alpha` is read as the wrapped rule reads it — mixed relatives ([`mixed_relatives`](@ref)) at the unmixed played iterate, and the played mix from the unmixed ``\\boldsymbol{w}_{t+1}``, projected once more where the set excludes it ([`reprojection`](@ref)). The first half-step is taken at the rate of the period, ``\\eta_t``, and the second at the rate of the next, ``\\eta_{t+1}``, which every schedule answers from the carrier once the period's row is in it — the paper forms ``\\boldsymbol{w}_{t+1}`` with ``\\eta_{t+1}``, and its adaptive rate reads the residual of the period just closed — so a number is one rate for both and a schedule is two. A restart the schedule names returns both iterates to the Start Allocation, re-entered onto the set from the book the fund holds, and every carrier to its seed. The wrapped rule's Gradient Transform must be the identity, because the transforms keep a carrier written once a period from one gradient, and the optimistic step reads two.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimisticStep(;
        alg::MirrorDescent = MirrorDescent(),
        predictor::AbstractGradientPredictor = LastGradient()
    ) -> OptimisticStep

Keywords correspond to the struct's fields.

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
    The first-order rule both half-steps are taken with.
    """
    alg::T1
    """
    The Gradient Predictor whose hint the second half-step is taken along.
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

One half-step of an [`OptimisticStep`](@ref): the wrapped rule's mirror step from `u` along the scaled gradient `s`, projected onto the set in its geometry from the Price-Adjusted Allocation `w`.

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
