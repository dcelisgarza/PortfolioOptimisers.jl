"""
$(DocStringExtensions.TYPEDEF)

Ordered Weights Array Range (OWA Range) risk measure.

Computes portfolio risk as the difference between two OWA linear combinations of sorted portfolio returns, providing a range-based risk measure.

The constructor reverses `w2` unless the caller declares that it is already reversed (see `rev`). That reversal is what makes the *difference* of the two weight vectors a *range*: `w1` addresses the lower tail of the sorted returns, and the reversed `w2` addresses the upper tail, so `w1 - w2` sums the two tails rather than cancelling them. It is the weight-space form of the convention [`ValueatRiskRange`](@ref) states on the returns themselves.

# Mathematical definition

```math
\\begin{align}
\\mathrm{OWARange}_{\\boldsymbol{w}_{1},\\boldsymbol{w}_{2}}(\\boldsymbol{x}) &= \\sum_{t=1}^{T} \\left(w_{1,t} - w_{2,t}\\right) x_{(t)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{OWARange}_{\\boldsymbol{w}_{1},\\boldsymbol{w}_{2}}(\\boldsymbol{x})``: Ordered weights array range.
  - $(math_dict[:xret])
  - $(math_dict[:T])
  - ``\\boldsymbol{w}_{1}``: Lower-tail OWA weight vector ``T \\times 1``.
  - ``\\boldsymbol{w}_{2}``: Upper-tail OWA weight vector ``T \\times 1``, held reversed by the constructor.
  - ``x_{(t)}``: ``t``-th smallest entry of ``\\boldsymbol{x}``, so ``x_{(1)} \\leq \\dots \\leq x_{(T)}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OrderedWeightsArrayRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w1::OWA_Func_VecNum = owa_tg,
        w2::OWA_Func_VecNum = owa_tg,
        alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
        rev::Bool = false
    ) -> OrderedWeightsArrayRange

Keywords correspond to the struct's fields.

## Validation

  - If `w1` is a `VecNum`: `!isempty(w1)`.
  - If `w2` is a `VecNum`: `!isempty(w2)`.
  - If both `w1` and `w2` are `VecNum`: `length(w1) == length(w2)`.

# Functor

    (r::OrderedWeightsArrayRange)(x::VecNum)

Computes the OWA range of a portfolio returns vector `x`, as the sorted returns weighted by `w1 - w2`. A `w1` or `w2` that is a callable is evaluated at `length(x)` first, so the weight vectors always match the sample.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> OrderedWeightsArrayRange()
OrderedWeightsArrayRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
        w1 ┼ typeof(owa_tg): PortfolioOptimisers.owa_tg
        w2 ┼ ComposedFunction{typeof(reverse), typeof(owa_tg)}: reverse ∘ PortfolioOptimisers.owa_tg
       alg ┼ ApproxOrderedWeightsArray
           │   p ┴ Vector{Float64}: [2.0, 3.0, 4.0, 10.0, 50.0]
       rev ┴ Bool: true
```

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`OrderedWeightsArrayFormulation`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`RiskMeasureSettings`](@ref)

# References

  - $(ref_dict[:owaog])
  - $(ref_dict[:owa1])
  - $(ref_dict[:cajas2025]) Section 7.2.3.
"""
@concrete struct OrderedWeightsArrayRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w1_owa])
    """
    w1
    """
    $(field_dict[:w2_owa])
    """
    w2
    """
    $(field_dict[:alg])
    """
    alg
    """
    $(field_dict[:rev_owa])
    """
    rev
    function OrderedWeightsArrayRange(settings::RiskMeasureSettings, w1::OWA_Func_VecNum,
                                      w2::OWA_Func_VecNum,
                                      alg::OrderedWeightsArrayFormulation, rev::Bool)
        w1_flag = isa(w1, VecNum)
        w2_flag = isa(w2, VecNum)
        if w1_flag
            @argcheck(!isempty(w1), IsEmptyError("w1 cannot be empty"))
        end
        if w2_flag
            @argcheck(!isempty(w2), IsEmptyError("w2 cannot be empty"))
            # Non-mutating on purpose: `w2` belongs to the caller. An in-place reverse here
            # corrupts their vector, and building two measures from one vector reverses it
            # twice. It also makes the `w1 === w2` aliasing case need no special handling.
            if !rev
                w2 = reverse(w2)
            end
        else
            # Same rule for the builder branch: the closure is the caller's, so a cached
            # return value must not be reversed underneath them.
            if !rev
                w2 = reverse ∘ w2
            end
        end
        if w1_flag && w2_flag
            @argcheck(length(w1) == length(w2),
                      DimensionMismatch("w1 ($(length(w1))) must match w2 ($(length(w2)))"))
        end
        return new{typeof(settings), typeof(w1), typeof(w2), typeof(alg), typeof(rev)}(settings,
                                                                                       w1,
                                                                                       w2,
                                                                                       alg,
                                                                                       true)
    end
end
function OrderedWeightsArrayRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  w1::OWA_Func_VecNum = owa_tg,
                                  w2::OWA_Func_VecNum = owa_tg,
                                  alg::OrderedWeightsArrayFormulation = ApproxOrderedWeightsArray(),
                                  rev::Bool = false)
    return OrderedWeightsArrayRange(settings, w1, w2, alg, rev)
end
# Deferrable slots — see `deferred_slots`. Both builders carry them. `rev` is stored as a
# done flag, so the positional rebuild of `rebuild_with_slots` passes `true` and reverses
# nothing a second time.
deferred_slots(r::OrderedWeightsArrayRange) = (; w1 = r.w1, w2 = r.w2)
# Calibration slots — see `calibration_slots`. The container names its two children, on the
# terms the single-builder container above states.
function calibration_slots(r::OrderedWeightsArrayRange)
    return merge(isa(r.w1, OWA_CalOccupant) ? (; w1 = r.w1) : (;),
                 isa(r.w2, OWA_CalOccupant) ? (; w2 = r.w2) : (;))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the two weight builders of an [`OrderedWeightsArrayRange`](@ref) against prior result `pr`.

It carries the reading of [`factory`](@ref) on [`OrderedWeightsArray`](@ref) unchanged, over two builders instead of one.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`OrderedWeightsArray`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
"""
function factory(x::OrderedWeightsArrayRange, pr::AbstractPriorResult, args...; kwargs...)
    return resolve_deferred_quantities(x, pr)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the weight builder inside the reversal that [`OrderedWeightsArrayRange`](@ref) wraps its `w2` in.

The Range constructor stores `w2` as `reverse ∘ w2` whenever the caller has not already reversed it, so the object the `w2` slot holds is a composition and the builder a rule sits in is its inner half. Without this method the recursion would stop at the composition and a rule in the gain-side builder would never resolve, while the same rule in `w1` did.

The composition is rebuilt around the resolved half, so the reversal survives. The bound names `reverse` and an [`AbstractOrderedWeightsArrayFunction`](@ref), so no other composition reaches it.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`calibration_slots`](@ref)
"""
function resolve_deferred_quantities(x::ComposedFunction{typeof(reverse),
                                                         <:AbstractOrderedWeightsArrayFunction},
                                     pr::AbstractPriorResult, slv = nothing)
    inner = resolve_deferred_quantities(x.inner, pr, slv)
    return inner === x.inner ? x : reverse ∘ inner
end
# Calibration slots — see `calibration_slots`. The reversal states no quantity of its own,
# so it names the builder it wraps and that builder names its own slots.
calibration_slots(x::OWA_RevFunc) = (; inner = x.inner)
# Tail decomposition — see `range_tails`. Declared for the approximate formulation only: the
# exact one collapses both tails into a single constraint on `w1 - w2`, so it fuses rather
# than duplicating. `w2` is already reversed by the constructor (`rev` is a done flag), so
# each tail is an ordinary `OrderedWeightsArray` over its own weight vector.
function range_tails(r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                 <:ApproxOrderedWeightsArray})
    settings = RiskMeasureSettings(; rke = false)
    return (; loss = OrderedWeightsArray(; settings = settings, w = r.w1, alg = r.alg),
            gain = OrderedWeightsArray(; settings = settings, w = r.w2, alg = r.alg))
end
function (r::OrderedWeightsArrayRange)(x::VecNum)
    w1 = isa(r.w1, VecNum) ? r.w1 : r.w1(length(x))
    w2 = isa(r.w2, VecNum) ? r.w2 : r.w2(length(x))
    w = w1 - w2
    return LinearAlgebra.dot(w, sort(x))
end

# Expected-risk input kind — see `risk_input_kind`.
risk_input_kind(::OrderedWeightsArray) = NetReturnsInput()
risk_input_kind(::OrderedWeightsArrayRange) = NetReturnsInput()

# The `# Interfaces`-marked type of #1137 (ADR 0154). Its section names only the callable
# shape, so no separate verb is promoted alongside it.

export OrderedWeightsArrayRange
