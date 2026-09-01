"""
$(DocStringExtensions.TYPEDEF)

Carries one return term's own weight in the return sum, its own lower bound, and the two charges netted out of it.

A [`JuMPOptimiser`](@ref) takes one return term or a vector of them, and the model's single
scalar return expression is the weighted sum ``\\mathrm{ret} = \\sum_i s_i\\, \\mathrm{ret}_i``
over the terms whose `rte` is `true`. This bundle carries everything that belongs to *one*
term rather than to the optimiser: its weight in that sum, its own lower bound, whether it
enters the sum at all, and which of the two portfolio charges are netted out of it.

The bundle sits in a field called `settings`, placed **first** on every return estimator, in
the same position [`RiskMeasureSettings`](@ref) takes on a risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPReturnsSettings(;
        scale::Number = 1.0,
        lb::Option{<:RkRtBounds} = nothing,
        rte::Bool = true,
        fee::Bool = true,
        mic::Bool = true
    ) -> JuMPReturnsSettings

Keywords correspond to the struct's fields.

## Details

  - `scale` is the term's weight in the sum, and it is *not* normalised. Two terms at
    `scale = 1` charge their flagged fees twice; a blend of two terms at `scale = 0.5`
    charges them once. That is a statement about the configuration, not a defect.
  - `fee` and `mic` are independent because market impact already constrains the budget, so a
    caller may net the fees into a term while leaving the impact cost out of it.
  - `rte = false` is the route for a term that is not in return units: it still takes its own
    `lb`, so a **constraint-only** return term is expressible.

## Validation

  - `isfinite(scale)`.
  - If `lb` is a number: `isfinite(lb)`.
  - If `lb` is a vector: `!isempty(lb)` and `all(isfinite, lb)`.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
@concrete struct JuMPReturnsSettings <: AbstractEstimator
    """
    $(field_dict[:scale_rt])
    """
    scale
    """
    $(field_dict[:lb_rts])
    """
    lb
    """
    $(field_dict[:rte])
    """
    rte
    """
    $(field_dict[:fee_rts])
    """
    fee
    """
    $(field_dict[:mic_rts])
    """
    mic
    function JuMPReturnsSettings(scale::Number, lb::Option{<:RkRtBounds}, rte::Bool,
                                 fee::Bool, mic::Bool)
        @argcheck(isfinite(scale), IsNonFiniteError("scale must be finite, got $scale"))
        if isa(lb, Number)
            @argcheck(isfinite(lb), IsNonFiniteError("lb must be finite, got $lb"))
        elseif isa(lb, VecNum)
            @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
            @argcheck(all(isfinite, lb),
                      IsNonFiniteError("all elements of lb must be finite"))
        end
        return new{typeof(scale), typeof(lb), typeof(rte), typeof(fee), typeof(mic)}(scale,
                                                                                     lb,
                                                                                     rte,
                                                                                     fee,
                                                                                     mic)
    end
end
function JuMPReturnsSettings(; scale::Number = 1.0, lb::Option{<:RkRtBounds} = nothing,
                             rte::Bool = true, fee::Bool = true, mic::Bool = true)
    return JuMPReturnsSettings(scale, lb, rte, fee, mic)
end
"""
    const VecJRE = AbstractVector{<:JuMPReturnsEstimator}

Alias for a vector of return terms.

Mirrors [`VecRM`](@ref) on the risk side.

The vector is the multiplicity carrier ([`JRE_VecJRE`](@ref)), so every seam that reaches one
term reaches all of them. [`factory`](@ref) and [`port_opt_view`](@ref) therefore need no method
of their own here: their generic vector methods rebuild and view each term in turn. Each term
keeps its own settings, its own uncertainty set and its own characteristic; the outer `ucs`
argument, when there is one, is the same for all of them, because only a single-term
configuration can be routed a bare mean uncertainty set (see [`pipe_route`](@ref)).

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`JRE_VecJRE`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
const VecJRE = AbstractVector{<:JuMPReturnsEstimator}
"""
    const JRE_VecJRE = Union{<:JuMPReturnsEstimator, <:VecJRE}

Field bound for [`JuMPOptimiser`](@ref)'s `ret` slot: one return term or several.

Mirrors [`RM_VecRM`](@ref) on the risk side.

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`VecJRE`](@ref)
"""
const JRE_VecJRE = Union{<:JuMPReturnsEstimator, <:VecJRE}
"""
    const ArithRetMu = Union{<:Num_VecNum, <:AbstractExpectedReturnsEstimator, <:AbstractPriorEstimator}

Field bound for [`ArithmeticReturn`](@ref)'s `mu` slot: the expected returns themselves, or the Estimator that computes them (a **Deferred Quantity** — see [`DeferredQuantity`](@ref)).

Narrower than [`MuSlot`](@ref) by a [`VecScalar`](@ref). A `VecScalar` is a centring target for a moment risk measure, and the return expression is `dot_scalar(mu, w)`, which takes a number or a vector. It is also an [`AbstractResult`](@ref), and an Estimator must not hold one.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`MuSlot`](@ref)
  - [`DeferredQuantity`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
const ArithRetMu = Union{<:Num_VecNum, <:AbstractExpectedReturnsEstimator,
                         <:AbstractPriorEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Computes the portfolio return as the arithmetic mean return, the dot product of the expected returns and the weights.

Optionally supports an uncertainty set on the mean vector (box, ellipsoidal or
``\\ell_1``). When `ucs` is set the optimiser maximises the **worst-case** expected return
over the set instead of the point estimate `μ`, giving a robust return.

# Mathematical definition

```math
\\begin{align}
r(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``r(\\boldsymbol{w})``: Expected portfolio return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])

Each `ucs` shape replaces this expression with its own worst case; the four are stated on
the [`set_ucs_return_constraints!`](@ref) methods.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ArithmeticReturn(;
        settings::JuMPReturnsSettings = JuMPReturnsSettings(),
        ucs::Option{<:UcSE_UcS} = nothing,
        mu::Option{<:ArithRetMu} = nothing
    ) -> ArithmeticReturn

Keywords correspond to the struct's fields.

## Details

  - `ucs` accepts either a pre-built mean uncertainty set (the result of [`mu_ucs`](@ref), e.g. a `BoxUncertaintySet` or `EllipsoidalUncertaintySet`) or an uncertainty-set *estimator*. A pre-built set is the simplest path — symmetric with how [`UncertaintySetVariance`](@ref) takes a pre-built [`sigma_ucs`](@ref) result. Passing an estimator defers construction to solve time and requires the returns data (`rd`) to be threaded through the optimiser.
  - `mu` accepts a **Deferred Quantity**: an expected-returns estimator or a prior estimator that computes the vector against the optimisation's own prior at [`factory`](@ref) time. See [`resolve_deferred_quantities`](@ref).
  - A `ucs` that carries its own centre outranks `mu`, and `mu` outranks `pr.mu` (ADR 0050). A Deferred Quantity is a state of the `mu` rung, not a rung of its own: beside a centre-carrying set it is resolved and then goes unused, exactly as a stated vector does.
  - The lower bound lives on `settings.lb`, not on the term itself (ADR 0052).

## Validation

  - If `ucs` is an `EllipsoidalUncertaintySet`: must be parameterised by `MuEllipsoidalUncertaintySet`.
  - If `mu` is a number: `isfinite(mu)`.
  - If `mu` is a vector: `!isempty(mu)` and `all(isfinite, mu)`.

!!! warning

    A stated `mu` is pinned: it crosses a Cross-Validation fold or a subset view as the whole universe's answer, so it does not follow the refit the optimisation runs on. A caller who wants it to follow the fit names a **Deferred Quantity** in `mu`, or leaves the slot `nothing` and lets the prior supply it.

# Related

  - [`JuMPReturnsSettings`](@ref)
  - [`bounds_returns_estimator`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.1.1.
  - $(ref_dict[:markowitz1952])
"""
@concrete struct ArithmeticReturn <: JuMPReturnsEstimator
    """
    $(field_dict[:settings_rt])
    """
    settings
    """
    $(field_dict[:ucs])
    """
    ucs
    """
    $(field_dict[:mu_ret_slot])
    """
    mu
    function ArithmeticReturn(settings::JuMPReturnsSettings, ucs::Option{<:UcSE_UcS},
                              mu::Option{<:ArithRetMu})
        if isa(ucs, EllipsoidalUncertaintySet)
            @argcheck(isa(ucs,
                          EllipsoidalUncertaintySet{<:Any, <:Any,
                                                    <:MuEllipsoidalUncertaintySet}),
                      ArgumentError("ucs must be parameterised by MuEllipsoidalUncertaintySet, got $(typeof(ucs))"))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu),
                      IsNonFiniteError("all elements of mu must be finite"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        return new{typeof(settings), typeof(ucs), typeof(mu)}(settings, ucs, mu)
    end
end
function ArithmeticReturn(; settings::JuMPReturnsSettings = JuMPReturnsSettings(),
                          ucs::Option{<:UcSE_UcS} = nothing,
                          mu::Option{<:ArithRetMu} = nothing)
    return ArithmeticReturn(settings, ucs, mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a **Deferred Quantity** in [`ArithmeticReturn`](@ref)'s `mu` slot against prior result `pr`. The estimator carries one prior-derived slot, so the slot itself admits the Estimator and there is no fan-out to make.

Every `JuMP` path reaches this through [`factory`](@ref), which [`processed_jump_optimiser_attributes`](@ref) calls on `opt.ret` before any model is built. A return term needs no second entry point, unlike a risk measure.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`ArithRetMu`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_deferred_quantities(rt::ArithmeticReturn, pr::AbstractPriorResult,
                                     ::Any = nothing)
    if !isa(rt.mu, DeferredQuantity)
        return rt
    end
    return rebuild_with_slots(rt, (; mu = resolve_slot(rt.mu, :mu, pr)))
end
# Deferrable slots — see `deferred_slots`. `ucs` holds an Estimator by design, not a Deferred
# Quantity, so it is not declared here. The declaration is what carries this slot into the
# containers that hold a return term — `ExpectedReturn` and `ExpectedReturnRiskRatio`.
deferred_slots(rt::ArithmeticReturn) = (; mu = rt.mu)
function factory(rt::ArithmeticReturn, pr::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcSE_UcS} = nothing, args...; kwargs...)
    rt = resolve_deferred_quantities(rt, pr)
    return ArithmeticReturn(; settings = rt.settings, ucs = ucs_selector(rt.ucs, ucs),
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu))
end
function factory(rt::ArithmeticReturn, pr::AbstractPriorResult,
                 ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
    rt = resolve_deferred_quantities(rt, pr)
    return ArithmeticReturn(; settings = rt.settings, ucs = ucs_selector(rt.ucs, ucs),
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu))
end
function factory(rt::ArithmeticReturn, ucs::UcSE_UcS, pr::AbstractPriorResult; kwargs...)
    rt = resolve_deferred_quantities(rt, pr)
    return ArithmeticReturn(; settings = rt.settings, ucs = ucs_selector(rt.ucs, ucs),
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu))
end
function factory(rt::ArithmeticReturn, ucs::UcSE_UcS, args...; kwargs...)
    # No prior in hand, so a Deferred Quantity cannot resolve here. It travels on unchanged
    # and the prior-carrying `factory` the sub-problem runs resolves it.
    return ArithmeticReturn(; settings = rt.settings, ucs = ucs_selector(rt.ucs, ucs),
                            mu = rt.mu)
end
function port_opt_view(r::ArithmeticReturn, i, args...)
    uset = port_opt_view(r.ucs, i)
    # A Deferred Quantity crosses the view unsliced: `nothing_scalar_array_view` is the
    # identity on an Estimator. It then computes on the subset, which is the whole
    # fold-stability argument for the feature.
    mu = nothing_scalar_array_view(r.mu, i)
    return ArithmeticReturn(; settings = r.settings, ucs = uset, mu = mu)
end
"""
    no_bounds_returns_estimator(r, args...)

Create a version of the return term with its lower bound removed.

Used internally in frontier and near-optimal-centering sub-problems, where the corner solves
must range freely over the feasible set.

Only `lb` and — when `flag` is `false` — `ucs` are stripped. Everything else the term carries
survives, `mu` included: dropping the characteristic would silently re-centre the term on the
prior's own vector, which is the ADR 0050 defect class, and with several terms it
would collapse every one of them onto the same corner.

# Arguments

  - `r`: One return term, or a vector of them.
  - `flag::Bool`: When `false` the uncertainty set is stripped too.

# Returns

  - The term(s) without bounds.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`no_bounds_optimiser`](@ref)
"""
function no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool = true)
    return ArithmeticReturn(; settings = no_bounds_returns_settings(r.settings),
                            ucs = ifelse(flag, r.ucs, nothing), mu = r.mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of `settings` with its lower bound cleared.

`scale`, `rte`, `fee` and `mic` are not bounds, so they survive: a corner solve must charge
the same fees and weight the same terms as the sweep it seeds.

# Related

  - [`no_bounds_returns_estimator`](@ref)
  - [`JuMPReturnsSettings`](@ref)
"""
function no_bounds_returns_settings(settings::JuMPReturnsSettings)
    return JuMPReturnsSettings(; scale = settings.scale, lb = nothing, rte = settings.rte,
                               fee = settings.fee, mic = settings.mic)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of return term `r` with its `scale` set to `one(scale)`. A term that already
carries a unit scale is returned unchanged, so the common path allocates nothing.

`scale` is a combination weight: it says how much this term contributes to the return
expression built from several terms. One term is not a combination, so the weight has
nothing to weigh and the singular route drops it. `lb`, `rte`, `fee` and `mic` are not
weights, so they survive: the bound still binds on the term's own expression and the term
still charges the same fees.

This is the return-axis twin of [`unit_scale_risk_measure`](@ref).

# Arguments

  - `r`: A [`JuMPReturnsEstimator`](@ref).

# Returns

  - Return term carrying a unit scale.

# Related

  - [`unit_scale_risk_measure`](@ref)
  - [`no_bounds_returns_settings`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function unit_scale_returns_estimator(r::JuMPReturnsEstimator)
    settings = r.settings
    scale = settings.scale
    return if isone(scale)
        r
    else
        Accessors.@set r.settings = JuMPReturnsSettings(; scale = one(scale),
                                                        lb = settings.lb,
                                                        rte = settings.rte,
                                                        fee = settings.fee,
                                                        mic = settings.mic)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the portfolio return as the **mean logarithmic return**, the Kelly criterion's objective.

Optionally supports observation weights.

Unlike [`ArithmeticReturn`](@ref) this term holds **no per-asset quantity at all**, which is
why the plural noun of this family is the *return term* rather than the characteristic.

# Mathematical definition

The value the term reports is the mean of the log gross returns:

```math
\\begin{align}
r(\\boldsymbol{w}) &= \\frac{1}{T} \\sum_{t=1}^{T} \\ln\\left(1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}\\right)\\,.
\\end{align}
```

The model raises it as an exponential cone programme, one cone per observation:

```math
\\begin{align}
(q_t,\\; k,\\; k + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}) &\\in \\mathcal{K}_{\\exp} \\quad \\forall t = 1,\\dots,T\\,, \\\\
r(\\boldsymbol{w}) &= \\frac{1}{T} \\sum_{t=1}^{T} q_t\\,.
\\end{align}
```

Where:

  - ``r(\\boldsymbol{w})``: Mean logarithmic portfolio return.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:w_port])
  - $(math_dict[:T])
  - $(math_dict[:k_budget])
  - ``q_t``: Auxiliary model variable that the cone bounds by ``k \\ln(1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w} / k)``.
  - ``\\mathcal{K}_{\\exp}``: Exponential cone.

!!! warning

    This is the mean **logarithmic** return, not the geometric mean net return
    ``\\prod_t (1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w})^{1/T} - 1``. The two are one
    ``\\exp(\\cdot) - 1`` apart, so they order portfolios alike but carry different units. On a
    200-observation sample the term reports **0.0019487260** where the geometric mean is
    **0.0019506260**. `settings.lb` and [`MaximumRatio`](@ref)'s `rf` are therefore stated in
    log units. Apply `exp(r) - 1` to read the value as a net return.

The cone is a relaxation that a maximising objective closes: on that sample the model's own
`:ret` is **0.0019487202** against the value-level **0.0019487260**, a gap of **5.8e-9**.
[`expected_return`](@ref) computes the same quantity in closed form.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogarithmicReturn(;
        settings::JuMPReturnsSettings = JuMPReturnsSettings(),
        w::Option{<:ObsWeights} = nothing
    ) -> LogarithmicReturn

Keywords correspond to the struct's fields.

## Validation

  - If `w` is provided: `!isempty(w)`, all elements non-negative and finite.

# Related

  - [`JuMPReturnsSettings`](@ref)
  - [`bounds_returns_estimator`](@ref)
  - [`ArithmeticReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`expected_return`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.1.2, equations 8.2 and 8.5.
  - $(ref_dict[:kelly1956])
  - $(ref_dict[:thorp2008])
  - $(ref_dict[:chares2009])
"""
@concrete struct LogarithmicReturn <: JuMPReturnsEstimator
    """
    $(field_dict[:settings_rt])
    """
    settings
    """
    $(field_dict[:oow])
    """
    w
    function LogarithmicReturn(settings::JuMPReturnsSettings, w::Option{<:ObsWeights})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function LogarithmicReturn(; settings::JuMPReturnsSettings = JuMPReturnsSettings(),
                           w::Option{<:ObsWeights} = nothing)
    return LogarithmicReturn(settings, w)
end
function factory(rt::LogarithmicReturn, pr::AbstractPriorResult, args...; kwargs...)
    return LogarithmicReturn(; settings = rt.settings,
                             w = nothing_scalar_array_selector(rt.w, pr.w))
end
function no_bounds_returns_estimator(r::LogarithmicReturn, args...)
    return LogarithmicReturn(; settings = no_bounds_returns_settings(r.settings), w = r.w)
end
"""
$(DocStringExtensions.TYPEDEF)

Return term that contributes no return.

`NoReturn` computes nothing: its value-level twin returns zero and its optimisation
formulation adds a zero return expression. It exists so that an optimiser which genuinely has
no return term can say so, without a vestigial one changing the model class. The return-side
mirror of [`NoRisk`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NoReturn(; settings::JuMPReturnsSettings = JuMPReturnsSettings()) -> NoReturn

Keywords correspond to the struct's fields.

# Details

[`set_return_constraints!`](@ref) runs from the shared Model Assembly whatever the optimiser
is, and [`JuMPOptimiser`](@ref)'s `ret` slot defaults to [`ArithmeticReturn`](@ref). Three
optimisers never read `:ret` at all — [`RiskBudgeting`](@ref), [`RelaxedRiskBudgeting`](@ref)
and [`FactorRiskContribution`](@ref) — so today they build the whole expression, mean
uncertainty-set cones included, and discard it. That is [`NoRisk`](@ref)'s own argument from
the other side: a vestigial term drags constraints into a model that does not need them, and
a conic solver onto a problem that is a linear program. `NoReturn` keeps such problems in the
class they belong to, and it is the main use of the type.

It also makes "no return term" something a caller **states**, rather than something that falls
out of setting `settings.rte = false` on every term.

# Notes

`NoReturn` is only coherent where nothing reads the return expression:

| Optimiser and objective                                     | `NoReturn` |
|:----------------------------------------------------------- |:---------- |
| [`RiskBudgeting`](@ref), [`RelaxedRiskBudgeting`](@ref)     | ok         |
| [`FactorRiskContribution`](@ref) + [`MinimumRisk`](@ref)    | ok         |
| [`FactorRiskContribution`](@ref) + [`MaximumUtility`](@ref) | ok         |
| [`FactorRiskContribution`](@ref) + [`MaximumReturn`](@ref)  | **throws** |
| [`FactorRiskContribution`](@ref) + [`MaximumRatio`](@ref)   | **throws** |
| [`MeanRisk`](@ref) + [`MinimumRisk`](@ref)                  | ok         |
| [`MeanRisk`](@ref) + [`MaximumUtility`](@ref)               | ok         |
| [`MeanRisk`](@ref) + [`MaximumReturn`](@ref)                | **throws** |
| [`MeanRisk`](@ref) + [`MaximumRatio`](@ref)                 | **throws** |
| [`NearOptimalCentering`](@ref)                              | **throws** |

[`RiskBudgeting`](@ref) and [`RelaxedRiskBudgeting`](@ref) hold no objective at all, so
nothing there can read `:ret`. [`FactorRiskContribution`](@ref) does hold one, so it is
refused on exactly the same two objectives as [`MeanRisk`](@ref).

The objective refusals come from [`assert_no_return_objective_compatibility`](@ref), at model
build: a [`MaximumReturn`](@ref) objective would be identically zero, so the solver returns an
arbitrary feasible portfolio and reports success, and a [`MaximumRatio`](@ref) numerator would
vanish. The [`NearOptimalCentering`](@ref) refusal comes from
[`assert_return_term_required`](@ref) at its constructor, and it is an **infeasibility**, not
a degeneracy: the barrier constrains `exp(log_ret) <= ret - rt`, and with no return term both
sides are zero.

Every refusal above is reached by `settings.rte = false` on every term too, because the guards
test the state of the expression and not the type of the term (ADR 0054).

The term holds no per-asset quantity, so `settings.scale`, `settings.fee` and `settings.mic`
are inert — zero scaled is still zero, and a charge subtracted here would make the expression
non-zero, which every guard above rests on. Setting `settings.lb` is legal but binds on a
quantity that is always zero, so a positive bound makes the model infeasible; this is
[`NoRisk`](@ref)'s `settings.ub`, which is legal and pointless for the same reason.

# Related

  - [`NoRisk`](@ref)
  - [`JuMPReturnsSettings`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`zero_return_expression_flag`](@ref)
  - [`assert_no_return_objective_compatibility`](@ref)
  - [`assert_return_term_required`](@ref)
"""
@concrete struct NoReturn <: JuMPReturnsEstimator
    """
    $(field_dict[:settings_rt])
    """
    settings
    function NoReturn(settings::JuMPReturnsSettings)
        return new{typeof(settings)}(settings)
    end
end
function NoReturn(; settings::JuMPReturnsSettings = JuMPReturnsSettings())::NoReturn
    return NoReturn(settings)
end
function no_bounds_returns_estimator(r::NoReturn, args...)
    return NoReturn(; settings = no_bounds_returns_settings(r.settings))
end
function no_bounds_returns_estimator(r::VecJRE, args...)
    return [no_bounds_returns_estimator(ri, args...) for ri in r]
end
"""
    bounds_returns_estimator(r, lb)

Return a copy of return term `r` with its lower bound set to `lb`.

The pairing is **term by term**. One term takes a scalar bound or `nothing`; *k* terms take
`nothing` (which clears all *k*) or a vector of *k* bounds, one per term.

A scalar **number** against *k* terms is **refused**. The bound binds on each term's own
expression, and the terms are heterogeneous in unit, so no check could tell whether one
number means the same thing to all of them.

# Arguments

  - `r`: One return term, or a vector of them.
  - `lb`: The lower bound; a number, `nothing`, or a vector of one per term.

# Returns

  - The term(s) with the updated lower bound.

# Related

  - [`JuMPReturnsSettings`](@ref)
  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
"""
function bounds_returns_estimator(r::JuMPReturnsEstimator, lb::Option{<:RkRtBounds})
    return Accessors.@set r.settings.lb = lb
end
function bounds_returns_estimator(r::VecJRE, lb::Nothing)
    return [bounds_returns_estimator(ri, nothing) for ri in r]
end
function bounds_returns_estimator(r::VecJRE, lb::Number)
    return throw(ArgumentError("cannot apply the single bound $lb to $(length(r)) return terms: a bound binds on one term's own expression, and the terms are not guaranteed to share a unit. Pass a vector of $(length(r)) bounds, or `nothing` to clear them all."))
end
function bounds_returns_estimator(r::VecJRE, lb::AbstractVector)
    @argcheck(length(lb) == length(r),
              DimensionMismatch("`lb` must have one entry per return term:\n`length(lb)` => $(length(lb))\n`length(r)` => $(length(r))"))
    return [bounds_returns_estimator(ri, lbi) for (ri, lbi) in zip(r, lb)]
end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that minimises portfolio risk.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min}\\; R(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:R_w])

The objective reads no return expression, so it admits a [`NoReturn`](@ref) term and a zero
`:ret`. It also states no floor on the return, so a term's `settings.lb` is the only thing
that keeps the solution off a portfolio whose expected return is negative.

# Related

  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)
  - [`JuMPReturnsSettings`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.1, equation 8.7.
  - $(ref_dict[:markowitz1952])
"""
struct MinimumRisk <: ObjectiveFunction end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises risk-adjusted utility.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - l\\, R(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``l``: Risk-aversion coefficient.
  - $(math_dict[:R_w])

The risk carries the coefficient **whole**: the model builds `ret - l * risk`, with no factor
of one half. A caller porting a ``\\tfrac{\\lambda}{2}`` convention halves its own ``\\lambda``. At
`l = 3.0` on a five-asset sample the model's own objective is **0.0015922030484060072**, which
is `ret - 3.0 * risk` exactly; the halved form would give **0.0018692372364207165**.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumUtility(; l::Number = 2) -> MaximumUtility

Keywords correspond to the struct's fields.

## Validation

  - `l >= 0`.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.3, equation 8.12.
  - $(ref_dict[:markowitz1952])
"""
@concrete struct MaximumUtility <: ObjectiveFunction
    """
    $(field_dict[:l])
    """
    l
    function MaximumUtility(l::Number)
        @argcheck(l >= zero(l), DomainError(l, "l must be >= 0"))
        return new{typeof(l)}(l)
    end
end
function MaximumUtility(; l::Number = 2)
    return MaximumUtility(l)
end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises the risk-adjusted Sharpe-type ratio.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\frac{\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f}{R(\\boldsymbol{w})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``r_f``: Risk-free rate.
  - $(math_dict[:R_w])

The quotient is not convex, so the model solves the equivalent fractional programme instead.
It homogenises the whole feasible set by a scalar ``k \\geq 0``, optimises in
``\\boldsymbol{y} = k \\boldsymbol{w}``, and de-homogenises with
``\\boldsymbol{w} = \\boldsymbol{y} / k``. One of two normalisations pins ``k``:

  - **Return form.** ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{y} - r_f k = \\mathrm{ohf}``, and the model minimises the risk. This is the branch [`set_max_ratio_return_constraints!`](@ref) registers as `sr_ret`.
  - **Risk form.** ``R(\\boldsymbol{y}) \\leq \\mathrm{ohf}``, and the model maximises ``\\mathrm{ret} - r_f k``. This is the branch registered as `sr_risk`.

The return form fails whenever the portfolio's expected return cannot exceed ``r_f``, so the
risk form takes over there, and wherever a term raises a cone the return form cannot carry.
[`set_max_ratio_return_constraints!`](@ref) states the exact test.

The ratio is taken at the **aggregate** level: its numerator is the model's single `ret`
expression, whatever number of terms built it. `rf` is therefore a single rate on that
aggregate, and a term that is not in return units belongs out of the numerator through
`settings.rte = false`, not through a per-term rate.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumRatio(; rf::Number = 0.0, ohf::Option{<:Number} = nothing) -> MaximumRatio

Keywords correspond to the struct's fields.

## Validation

  - If `ohf` is provided: `ohf > 0`.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)
  - [`set_max_ratio_return_constraints!`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.4, equations 8.13 to 8.16.
  - $(ref_dict[:sharpe1964])
  - $(ref_dict[:schaibleibaraki1983])
  - $(ref_dict[:charnescooper1962])
"""
@concrete struct MaximumRatio <: ObjectiveFunction
    """
    $(field_dict[:rf])
    """
    rf
    """
    $(field_dict[:ohf])
    """
    ohf
    function MaximumRatio(rf::Number, ohf::Option{<:Number})
        if !isnothing(ohf)
            @argcheck(ohf > zero(ohf), DomainError(ohf, "ohf must be > 0"))
        end
        return new{typeof(rf), typeof(ohf)}(rf, ohf)
    end
end
function MaximumRatio(; rf::Number = 0.0, ohf::Option{<:Number} = nothing)
    return MaximumRatio(rf, ohf)
end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises the model's return expression.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\mathrm{ret}(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\mathrm{ret}(\\boldsymbol{w})``: The model's single scalar return expression, the weighted sum over the return terms.

The objective reads the aggregate expression, not ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``:
a [`LogarithmicReturn`](@ref) term contributes a mean logarithmic return, and several terms
contribute their weighted sum. It states no ceiling on the risk, so a risk measure's
`settings.ub` is the only thing that keeps the solution off the single highest-return asset.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`ObjectiveFunction`](@ref)
  - [`MaximumElementReturn`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.2, equation 8.10.
  - $(ref_dict[:markowitz1952])
"""
struct MaximumReturn <: ObjectiveFunction end
"""
$(DocStringExtensions.TYPEDEF)

Internal objective that maximises the expression of **one** return term.

Used only by the return-frontier corner solves. With *k* terms the span of term *i* has to be
read off a portfolio that maximised term *i* alone: reading it off the aggregate
maximum-return corner makes it an artefact of the other terms' `scale`, and can leave
`rt_min > rt_max`, so the sweep range descends.

Not part of the user-facing API.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumElementReturn(i::Integer) -> MaximumElementReturn

The argument corresponds to the struct's field. The type takes a positional argument alone,
because it carries one field and no caller outside the frontier builds it.

## Validation

  - `i > 0`. The other half of the domain, `i <= length(ret)`, is checked at model build by
    [`assert_no_return_objective_compatibility`](@ref), which is the first site that sees the
    return terms.

# Related

  - [`MaximumReturn`](@ref)
  - [`compute_ret_lbs`](@ref)
  - [`assert_no_return_objective_compatibility`](@ref)
"""
@concrete struct MaximumElementReturn <: ObjectiveFunction
    """
    $(field_dict[:i_ret_term])
    """
    i
    function MaximumElementReturn(i::Integer)
        @argcheck(i > zero(i), DomainError(i, "i must be > 0"))
        return new{typeof(i)}(i)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the model's `:ret` expression is identically zero.

The degeneracy guard tests the **state** of the expression, not the **type** of the term (ADR
0054). Two routes reach that state — a [`NoReturn`](@ref) term, and a term taken out of the
sum by `settings.rte = false` — and this predicate is one **fused** test over both.

The fusion is load-bearing. `:ret` is the weighted sum of the terms, so it is zero exactly
when *every* term is out of it, by either route. Testing the routes separately and composing
them is wrong:

```julia
r = [NoReturn(), ArithmeticReturn(; settings = JuMPReturnsSettings(; rte = false))]
```

Every term here is out of `:ret`, yet `all(isa NoReturn) || all(!rte)` returns `false`. Only
`all(isa NoReturn || !rte)` sees it. The risk axis composes instead, because its two halves
carry different quantifiers; see [`zero_risk_expression_flag`](@ref).

One real term beside a `NoReturn` leaves `:ret` non-zero, so `[ArithmeticReturn(), NoReturn()]` solves correctly under every objective and is not refused. An empty vector is
refused separately by [`set_return_constraints!`](@ref).

# Related

  - [`NoReturn`](@ref)
  - [`assert_no_return_objective_compatibility`](@ref)
  - [`assert_return_term_required`](@ref)
  - [`zero_risk_expression_flag`](@ref)
"""
function zero_return_expression_flag(r)::Bool
    return if isa(r, AbstractVector)
        !isempty(r) && all(x -> isa(x, NoReturn) || !x.settings.rte, r)
    else
        isa(r, NoReturn) || !r.settings.rte
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that an objective which reads the return expression is given a non-zero one.

Dispatches on the objective. Three of them read `:ret` and are refused when it is degenerate;
every other objective takes the no-op fallback, because a zero `:ret` is legitimate under
[`MinimumRisk`](@ref) and [`MaximumUtility`](@ref).

| Objective                      | Refused when                     |
|:------------------------------ |:-------------------------------- |
| [`MaximumReturn`](@ref)        | every term is out of `:ret`      |
| [`MaximumRatio`](@ref)         | every term is out of `:ret`      |
| [`MaximumElementReturn`](@ref) | term `i` is a [`NoReturn`](@ref) |

The first two use [`zero_return_expression_flag`](@ref), so both routes to a zero expression
are covered (ADR 0054). [`MaximumElementReturn`](@ref) is **per index** and ignores
`settings.rte` entirely: it maximises `ret_i` directly, which the builder registers whatever
the flag says, so a `false` flag removes that term from the *sum* without touching the
objective. Only the sentinel type makes `ret_i` itself zero.

[`MaximumElementReturn`](@ref)'s method **range-checks `i` first**, because the `NoReturn`
test indexes `ret[i]`. The refusal is a `DomainError`, matching the constructor's own
`i > zero(i)` guard: the two halves of one domain, spelled alike. The constructor cannot
check the upper half, because it never sees `ret`. A [`TimeDependent`](@ref) schedule is
checked **per fold**, by construction — a callable schedule has no value until its fold
exists, so an eager scan could cover vector schedules only.

Called from [`set_return_constraints!`](@ref), the shared model-build hook every JuMP
optimiser reaches, **not** from a constructor. The hook is the only site that sees all three
objective-carrying optimisers: [`MeanRisk`](@ref), [`FactorRiskContribution`](@ref) — which
carries its own `obj` and had no return-side guard at all — and
[`NearOptimalCentering`](@ref). The cost is that a refusal now arrives at `optimise` time
rather than at construction time.

The seam is also where a [`TimeDependent`](@ref) schedule is already resolved, so neither side
needs a schedule skip here.

# Related

  - [`NoReturn`](@ref)
  - [`zero_return_expression_flag`](@ref)
  - [`assert_return_term_required`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function assert_no_return_objective_compatibility(ret, ::ObjectiveFunction)::Nothing
    return nothing
end
function assert_no_return_objective_compatibility(ret, ::MaximumReturn)::Nothing
    @argcheck(!zero_return_expression_flag(ret),
              ArgumentError("MaximumReturn needs a non-zero return expression, and every return term is out of it: each term is either a `NoReturn` or carries `settings.rte = false`. The objective would be identically zero, so every feasible portfolio is optimal and the solver would return an arbitrary one while reporting success. Use obj = MinimumRisk() or obj = MaximumUtility(), or give a return term that is in the expression."))
    return nothing
end
function assert_no_return_objective_compatibility(ret, ::MaximumRatio)::Nothing
    @argcheck(!zero_return_expression_flag(ret),
              ArgumentError("MaximumRatio needs a non-zero return expression, and every return term is out of it: each term is either a `NoReturn` or carries `settings.rte = false`. The ratio's homogenisation variable `k` collapses to zero when `rf > 0`, and the problem returns an arbitrary feasible point when `rf = 0`. Use obj = MinimumRisk() or obj = MaximumUtility(), or give a return term that is in the expression."))
    return nothing
end
function assert_no_return_objective_compatibility(ret, obj::MaximumElementReturn)::Nothing
    i = obj.i
    rets = isa(ret, AbstractVector) ? ret : (ret,)
    @argcheck(i <= length(rets),
              DomainError(i,
                          "i must be <= the number of return terms; ret has $(length(rets)) $(length(rets) == 1 ? "term" : "terms")"))
    @argcheck(!isa(rets[i], NoReturn),
              ArgumentError("MaximumElementReturn($i) needs a non-zero return term at index $i, and that term is a `NoReturn`, whose expression is identically zero. The objective would be identically zero, so every feasible portfolio is optimal and the solver would return an arbitrary one while reporting success. Name a different term, or give a real return term at index $i. `settings.rte` is not consulted here: it removes a term from the summed `:ret` expression, while this objective reads `ret_$i` directly."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `ret` gives a non-zero return expression, for optimisers built around one.

A zero `:ret` is coherent in the three optimisers that never read it, and in
[`MeanRisk`](@ref) under an objective that never reads it. [`NearOptimalCentering`](@ref) is
the one optimiser that neither describes: its logarithmic barrier constrains
`exp(log_ret) <= ret - rt`, so with a zero return expression the model is **infeasible**
rather than degenerate, and without this check the failure arrives as a solver
`OptimisationFailure` naming nothing.

The criterion is [`zero_return_expression_flag`](@ref), so both routes to a zero expression
are covered (ADR 0054). This guard stays at the **constructor**, unlike the objective-level
[`assert_no_return_objective_compatibility`](@ref), and the split is principled: this one asks
whether the *formulation* needs a return term, which is knowable from the estimator alone,
while the other asks whether the *objective* does, and objective and term first meet at model
build.

`T` names the calling optimiser, for the error message. [`TimeDependent`](@ref) schedules are
skipped here and reached instead through [`assert_time_dependent_substitution`](@ref), which
re-runs the host's own constructor on each resolved entry.

# Related

  - [`NoReturn`](@ref)
  - [`zero_return_expression_flag`](@ref)
  - [`assert_no_return_objective_compatibility`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function assert_return_term_required(ret, T::Symbol)::Nothing
    if isa(ret, TimeDependent) || !zero_return_expression_flag(ret)
        return nothing
    end
    return throw(ArgumentError("$T needs a non-zero return expression, and every return term is out of it: each term is either a `NoReturn` or carries `settings.rte = false`. $T's logarithmic barrier constrains exp(log_ret) <= ret - rt, and with a zero return expression the reference return rt and the model's return expression are both zero, so the constraint reads exp(log_ret) <= 0, which no real log_ret satisfies. The model is infeasible, not merely degenerate. Give a return term that is in the expression. A zero return expression is for the optimisers whose formulation never reads it — RiskBudgeting, RelaxedRiskBudgeting and FactorRiskContribution — and for MeanRisk under MinimumRisk or MaximumUtility."))
end
"""
    set_maximum_ratio_factor_variables!(model, obj)

Register the homogenisation variable `k` for the maximum ratio objective.

This is the head-level producer of `k`. Every head that shapes `w` from an objective calls
it exactly once, so the two spellings of `k` live here rather than at each head: `k >= 0`
under [`MaximumRatio`](@ref), and the literal `1` under every other objective. The heads
whose formulation is fixed pass their own fixed objective ([`MinimumRisk`](@ref)) and take
the second branch. [`RiskBudgeting`](@ref) is the one head that does not come through here
— see [`get_k`](@ref).

This runs **before** the model is assembled, because [`set_weight_constraints!`](@ref) reads
`k` immediately afterwards. The other half of the old bundle — the normalisation factor `ohf`
— is sized from the *resolved* return characteristic, which does not exist until the return
builders have run, so [`set_maximum_ratio_normalisation!`](@ref) registers it later.

The second method takes exactly one objective, not `args...`. A variadic fallback used to
absorb a wrong-arity call silently and register `k = 1` under a `MaximumRatio` objective, so
the ratio branch went untested while the call still looked correct.

# Arguments

  - `model`: JuMP optimisation model.
  - `obj`: Objective function (e.g., [`MaximumRatio`](@ref)).

# Returns

  - `nothing`.

# Related

  - [`MaximumRatio`](@ref)
  - [`get_k`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
  - [`ObjectiveFunction`](@ref)
"""
function set_maximum_ratio_factor_variables!(model::JuMP.Model, obj::MaximumRatio)
    JuMP.@variable(model, k >= 0)
    return nothing
end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, obj)
    JuMP.@expression(model, k, 1)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Register the ratio problem's normalisation factor `ohf`.

Sized from the resolved aggregate characteristic ``\\sum_{i:\\,\\mathrm{rte}} s_i \\boldsymbol{\\mu}_i`` when there is one, and from the prior's own vector when no term carries
a per-asset quantity. Reading it off the resolved aggregate corrects a pre-existing mismatch:
the factor used to come from `pr.mu` alone, so it ignored a term's own `mu` and a set's
carried centre. The change is numerical, not semantic — any `ohf > 0` recovers the same
`w / k`.

# Arguments

  - $(arg_dict[:model])
  - `obj::MaximumRatio`: The ratio objective.
  - `mu`: The resolved aggregate characteristic, or `nothing`.
  - `pr`: Prior result, the fallback when `mu` is `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_maximum_ratio_factor_variables!`](@ref)
  - [`set_max_ratio_return_constraints!`](@ref)
"""
function set_maximum_ratio_normalisation!(model::JuMP.Model, obj::MaximumRatio,
                                          mu::Option{<:Num_VecNum}, pr::AbstractPriorResult)
    ohf = if isnothing(obj.ohf)
        mu = isnothing(mu) ? pr.mu : mu
        min(1e3, max(1e-3, Statistics.mean(abs.(mu))))
    else
        @argcheck(obj.ohf > zero(obj.ohf), DomainError(obj.ohf, "obj.ohf must be > 0"))
        obj.ohf
    end
    JuMP.@expression(model, ohf, ohf)
    return nothing
end
"""
    set_return_bounds!(model, i, ret_expr, lb)

Bound the expression of return term `i` from below.

The `Nothing` overload does nothing. With a scalar `lb`, adds `ret_i >= lb * k`. With a
[`Frontier`](@ref) or a vector, pushes the term onto the `:ret_frontier` Model State entry
for a later sweep, exactly as [`set_risk_upper_bound!`](@ref) does on the risk side.

The bound binds on the term's **own** expression, net of that term's own flagged charges and
before `settings.scale` is applied, and it binds whether or not `settings.rte` is `true` — so
a term may constrain the portfolio without entering the objective at all.

# Arguments

  - $(arg_dict[:model])
  - `i`: Index of the return term.
  - `ret_expr`: The term's own JuMP return expression.
  - `lb`: Lower bound on the term (scalar, vector, or `Frontier`).

# Returns

  - `nothing`.

# Related

  - [`set_return_constraints!`](@ref)
  - [`JuMPReturnsSettings`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
function set_return_bounds!(::JuMP.Model, ::Any, ::Any, ::Nothing)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, i, ret_expr, lb::Number)
    sc = get_constraint_scale(model)
    k = get_k(model)
    state_set!(model, Symbol(""), :ret_lb_, i,
               JuMP.@constraint(model, sc * (ret_expr - lb * k) >= 0))
    return nothing
end
function set_return_bounds!(model::JuMP.Model, i, ret_expr, lb::Front_NumVec)
    bound_key = state_key(Symbol(""), :ret_lb_, i)
    bound_var_key = state_key(Symbol(""), :ret_lb_var_, i)
    if !shared_has(model, :ret_frontier)
        JuMP.@expression(model, ret_frontier,
                         Pair{Tuple{Symbol, Symbol},
                              Tuple{<:JuMP.AbstractJuMPScalar, <:Front_NumVec, <:Integer}}[(bound_var_key, bound_key) => (ret_expr,
                                                                                                                          lb,
                                                                                                                          i)])
    else
        push!(shared_get(model, :ret_frontier),
              (bound_var_key, bound_key) => (ret_expr, lb, i))
    end
    return nothing
end
"""
    set_return_expression!(model, i, ret_expr, scale, rte)

Push the scaled expression of return term `i` onto the `:ret_vec` Model State entry.

If `rte` is `false` the function does nothing, so the term contributes nothing to the model's
return expression while its own bound still binds. The twin of
[`set_risk_expression!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `i`: Index of the return term.
  - `ret_expr`: The term's own JuMP return expression.
  - `scale::Number`: The term's weight in the sum.
  - `rte::Bool`: When `false` this method is a no-op.

# Returns

  - `nothing`.

# Related

  - [`scalarise_return_expression!`](@ref)
  - [`set_return_bounds!`](@ref)
"""
function set_return_expression!(model::JuMP.Model, i, ret_expr, scale::Number, rte::Bool)
    if !rte
        return nothing
    end
    if !shared_has(model, :ret_vec)
        JuMP.@expression(model, ret_vec, JuMP.AffExpr[])
    end
    push!(shared_get(model, :ret_vec), scale * ret_expr)
    return nothing
end
"""
    scalarise_return_expression!(model)

Collapse the `:ret_vec` entries into the model's single scalar `:ret` expression.

The collapse is always the **weighted sum** ``\\sum_i s_i\\, \\mathrm{ret}_i``. There is no
scalariser on this side, and there is no configuration in which there is one (ADR 0052): the
package's scalarisers follow cvxpy's `scalarize` transforms, whose `max` and `log_sum_exp`
discard the objective's sense and so fail on a maximised concave expression, and cvxpy ships
no `min`. Normalising the sense to rescue them is barred, because `:ret` is a model-global
name that the objective, the bounds, the ratio and [`NearOptimalCentering`](@ref) all read,
and a stored `-ret` leads every one of them astray.

An empty `:ret_vec` — every term opted out through `settings.rte = false` — gives a zero
return expression rather than an error here. The refusal belongs to the objective, not to the
collapse: [`MinimumRisk`](@ref) and [`MaximumUtility`](@ref) read a zero `:ret` legitimately,
while [`MaximumReturn`](@ref) and [`MaximumRatio`](@ref) are refused upstream by
[`assert_no_return_objective_compatibility`](@ref).

# Arguments

  - $(arg_dict[:model])

# Returns

  - `nothing`.

# Related

  - [`set_return_expression!`](@ref)
  - [`set_return_constraints!`](@ref)
  - [`scalarise_risk_expression!`](@ref)
"""
function scalarise_return_expression!(model::JuMP.Model)
    JuMP.@expression(model, ret, zero(JuMP.AffExpr))
    if !shared_has(model, :ret_vec)
        return nothing
    end
    for ret_i in shared_get(model, :ret_vec)
        JuMP.add_to_expression!(ret, ret_i)
    end
    return nothing
end
"""
    set_max_ratio_return_constraints!(model, obj, rets, mus, robust, pr)

Add the maximum-ratio homogenisation constraint to the model.

The constraint is **hoisted** out of the per-term builders and runs exactly once. It reads
the model-global `:ret` and registers the model-global names `sr_ret` and `sr_risk`, so *k*
copies of it would collide and each would read the wrong expression.

Which of the two forms is used is decided by a **structural `any`** and a **numeric
aggregate**:

 1. If any term has no per-asset characteristic (a [`LogarithmicReturn`](@ref)) or builds a
    robust cone (a box or ellipsoidal uncertainty set), the risk form is used.
 2. Otherwise the aggregate ``\\sum_{i:\\,\\mathrm{rte}} s_i \\boldsymbol{\\mu}_i`` decides:
    `all(x -> x <= rf, ·)` selects the risk form.

Step 2 is the exact generalisation of the single-term test. A per-term `any` would send two
terms at `0.9 r_f` and `scale = 1` down the weaker branch, though their sum is `1.8 r_f`.

An **empty numerator** is refused, mirroring [`NoRisk`](@ref) under this objective: with every
term out of `:ret`, `k` collapses to `0` at `rf > 0` and the problem returns an arbitrary
feasible point at `rf = 0`.

# Arguments

  - $(arg_dict[:model])
  - `obj`: Objective function; a no-op unless it is a [`MaximumRatio`](@ref).
  - `rets`: The return terms.
  - `mus`: Each term's resolved characteristic, `nothing` where it has none.
  - `robust`: Whether each term built a robust cone.
  - `pr`: Prior result, the fallback for sizing `ohf`.

# Returns

  - `nothing`.

# Related

  - [`MaximumRatio`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
"""
function set_max_ratio_return_constraints!(::JuMP.Model, ::ObjectiveFunction, args...)
    return nothing
end
function set_max_ratio_return_constraints!(model::JuMP.Model, obj::MaximumRatio, rets,
                                           mus::AbstractVector, robust::AbstractVector,
                                           pr::AbstractPriorResult)
    # The empty-numerator refusal is not here: it is one of the three objective refusals
    # `assert_no_return_objective_compatibility` makes at the top of this seam.
    mu = aggregate_return_characteristic(rets, mus)
    set_maximum_ratio_normalisation!(model, obj, mu, pr)
    sc = get_constraint_scale(model)
    k = get_k(model)
    ohf = shared_get(model, :ohf)
    ret = get_ret(model)
    rf = obj.rf
    if any(robust) || isnothing(mu) || all(x -> x <= rf, mu)
        risk = get_risk(model)
        JuMP.@constraint(model, sr_risk, sc * (risk - ohf) <= 0)
    else
        JuMP.@constraint(model, sr_ret, sc * (ret - rf * k - ohf) == 0)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Sum the characteristics of the terms that are in the return expression, each at its own scale.

Returns `nothing` when no included term carries a per-asset quantity, which is the state a
pure [`LogarithmicReturn`](@ref) problem is in. A term whose `settings.rte` is `false` is
skipped: it contributes nothing to `:ret`, so it must contribute nothing to the aggregate the
ratio's tests read.

# Related

  - [`set_max_ratio_return_constraints!`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
"""
function aggregate_return_characteristic(rets, mus::AbstractVector)
    mu = nothing
    for (r, mu_i) in zip(rets, mus)
        if !r.settings.rte || isnothing(mu_i)
            continue
        end
        term = r.settings.scale * mu_i
        mu = isnothing(mu) ? term : mu .+ term
    end
    return mu
end
"""
    add_fees_to_ret!(model, ret, fee::Bool)

Subtract the fees expression from one term's return expression.

Does nothing when the term's `settings.fee` is `false`, or when no fees are registered.

The charge stays **inside** each builder, so with *k* terms the multiplier on the fee is
``\\sum_{i:\\,\\mathrm{fee}} s_i``. That multiplier is deliberately unconstrained: a blend of
two terms at `scale = 0.5` charges the fee once, and two terms at `scale = 1` charge it
twice.

# Arguments

  - $(arg_dict[:model])
  - `ret`: JuMP return expression to modify in-place.
  - `fee::Bool`: The term's `settings.fee`.

# Returns

  - `nothing`.

# Related

  - [`add_market_impact_cost!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_fees_to_ret!(model::JuMP.Model, ret, fee::Bool)
    if !fee || !shared_has(model, :fees)
        return nothing
    end
    JuMP.add_to_expression!(ret, -shared_get(model, :fees))
    return nothing
end
"""
    add_market_impact_cost!(model, ret, mic::Bool)

Subtract the market impact cost from one term's return expression.

Does nothing when the term's `settings.mic` is `false`, or when no market impact cost is
registered. Only [`BudgetMarketImpact`](@ref) registers one; a plain budget cost constrains
the budget and never reaches the return expression, despite sharing the `cost_bgt_expr` name.

# Arguments

  - $(arg_dict[:model])
  - `ret`: JuMP return expression to modify in-place.
  - `mic::Bool`: The term's `settings.mic`.

# Returns

  - `nothing`.

# Related

  - [`add_fees_to_ret!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_market_impact_cost!(model::JuMP.Model, ret, mic::Bool)
    if !mic || !shared_has(model, :wip)
        return nothing
    end
    JuMP.add_to_expression!(ret, -shared_get(model, :cost_bgt_expr))
    return nothing
end
"""
    set_return_constraints!(model, pret, obj, pr; kwargs...)
    set_return_constraints!(model, i, pret, pr; kwargs...)

Build the model's return expression and the constraints that go with it.

The four-argument methods are the seam every JuMP optimiser reaches. They run the per-term
builder once per return term, collapse the results into the single `:ret` expression, and
then add the hoisted maximum-ratio constraint. The five-argument methods are the per-term
builders, which dispatch on the term's type and on the shape of its uncertainty set.

Each per-term builder registers its own index-suffixed names (`ret_1`, `t_l1ucs_2`, …),
applies that term's own flagged charges, bounds that term, and pushes the scaled expression
onto `:ret_vec`.

# Arguments

  - $(arg_dict[:model])
  - `pret`: One return term, or a vector of them.
  - `i`: Index of the return term (per-term builders).
  - `obj::ObjectiveFunction`: Portfolio objective function.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `kwargs...`: Additional keyword arguments (e.g. `rd` for uncertainty sets).

# Returns

  - The four-argument methods return `nothing`. A per-term builder returns
    `(mu, robust)`: the characteristic it resolved (or `nothing`), and whether it built a
    robust cone.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`NoReturn`](@ref)
  - [`scalarise_return_expression!`](@ref)
  - [`set_return_bounds!`](@ref)
  - [`add_fees_to_ret!`](@ref)
"""
function set_return_constraints!(model::JuMP.Model, pret::JuMPReturnsEstimator,
                                 obj::ObjectiveFunction, pr::AbstractPriorResult; kwargs...)
    # `scale` is a combination weight, so it is dropped here: one term is not a combination
    # and the weight has nothing to weigh. The vector method below keeps every element's
    # weight, because there the terms really do combine.
    #
    # `pret` is rebound before *both* uses on purpose. The second use feeds
    # `aggregate_return_characteristic`, which applies `settings.scale` to `mu_i` in its own
    # right. Dropping the weight at the first call alone would leave `MaximumRatio`'s
    # normalisation scaled while `:ret` is not — worse than not dropping it at all.
    assert_no_return_objective_compatibility(pret, obj)
    pret = unit_scale_returns_estimator(pret)
    mu, robust = set_return_constraints!(model, 1, pret, pr; kwargs...)
    scalarise_return_expression!(model)
    set_max_ratio_return_constraints!(model, obj, (pret,), [mu], [robust], pr)
    return nothing
end
function set_return_constraints!(model::JuMP.Model, pret::VecJRE, obj::ObjectiveFunction,
                                 pr::AbstractPriorResult; kwargs...)
    @argcheck(!isempty(pret), IsEmptyError("`ret` cannot be an empty vector"))
    assert_no_return_objective_compatibility(pret, obj)
    mus = Vector{Any}(undef, length(pret))
    robust = Vector{Bool}(undef, length(pret))
    for (i, pret_i) in enumerate(pret)
        mus[i], robust[i] = set_return_constraints!(model, i, pret_i, pr; kwargs...)
    end
    scalarise_return_expression!(model)
    set_max_ratio_return_constraints!(model, obj, pret, mus, robust, pr)
    return nothing
end
function set_return_constraints!(model::JuMP.Model, i,
                                 pret::ArithmeticReturn{<:Any, Nothing, <:Any},
                                 pr::AbstractPriorResult; kwargs...)
    w = get_w(model)
    settings = pret.settings
    mu = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, dot_scalar(mu, w)))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    return mu, false
end
"""
    set_ucs_return_constraints!(model, i, ucs::BoxUncertaintySet, mu, settings)

Build one term's box-robust return expression.

Introduces a norm-1 cone constraint to model the worst-case characteristic under a box
uncertainty set. The family dispatches on the set type: an [`EllipsoidalUncertaintySet`](@ref)
raises a second-order cone, and the two ``\\ell_1`` sets raise an infinity-norm cone and a pair
of linear epigraphs respectively.

# Mathematical definition

Box uncertainty set (worst-case return):

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\boldsymbol{\\Delta}^\\intercal |\\boldsymbol{w}|\\,, \\\\
\\boldsymbol{\\Delta} &= \\frac{\\boldsymbol{u} - \\boldsymbol{\\ell}}{2}\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\Delta}``: Half-width of the box uncertainty set.
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bounds of the box uncertainty set.

# Arguments

  - $(arg_dict[:model])
  - `i`: Index of the return term, which suffixes every name the builder registers.
  - `ucs`: The uncertainty set.
  - `mu`: Fallback characteristic vector, used when the set carries none of its own.
  - `settings::JuMPReturnsSettings`: The term's settings, read for `fee` and `mic`.

# Returns

  - `(ret, mu, robust)`: the term's expression, the characteristic the set is centred on —
    the set's own field wins over the fallback (ADR 0050) — and whether the builder raised a
    cone the ratio's `ret == rf k + ohf` normalisation cannot be used with.

# Related

  - [`set_return_constraints!`](@ref)
  - [`ArithmeticReturn`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::BoxUncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
    sc = get_constraint_scale(model)
    w = get_w(model)
    N = length(w)
    mu = something(ucs.val, mu)
    d_mu = (ucs.ub - ucs.lb) * 0.5
    bucs_w = state_set!(model, Symbol(""), :bucs_w_, i, JuMP.@variable(model, [1:N]))
    state_set!(model, Symbol(""), :bucs_ret_, i,
               JuMP.@constraint(model, [j = 1:N],
                                [sc * bucs_w[j]; sc * w[j]] in JuMP.MOI.NormOneCone(2)))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model,
                                      dot_scalar(mu, w) - LinearAlgebra.dot(d_mu, bucs_w)))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, true
end
"""
    set_ucs_return_constraints!(model, i, ucs::EllipsoidalUncertaintySet, mu, settings)

Build one term's ellipsoid-robust return expression.

Introduces a second-order cone constraint to model the worst-case characteristic under an
ellipsoidal uncertainty set. The cone is not linear, so the term is reported as `robust`, and
the ratio's `ret == rf k + ohf` normalisation cannot be used with it.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\kappa \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\kappa``: Ellipsoidal uncertainty set radius.
  - ``\\mathbf{G}``: Upper Cholesky factor of the uncertainty set covariance.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::EllipsoidalUncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
    sc = get_constraint_scale(model)
    w = get_w(model)
    mu = something(ucs.val, mu)
    G = LinearAlgebra.cholesky(ucs.sigma).U
    k = ucs.k
    x_eucs_w = state_set!(model, Symbol(""), :x_eucs_w_, i, JuMP.@expression(model, G * w))
    t_eucs_gw = state_set!(model, Symbol(""), :t_eucs_gw_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :eucs_ret_, i,
               JuMP.@constraint(model,
                                [sc * t_eucs_gw; sc * x_eucs_w] in JuMP.SecondOrderCone()))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, dot_scalar(mu, w) - k * t_eucs_gw))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, true
end
"""
    set_ucs_return_constraints!(model, i, ucs::L1UncertaintySet, mu, settings)

Build one term's ``\\ell_1``-robust return expression.

Introduces an infinity-norm cone constraint to model the worst-case characteristic under an
``\\ell_1`` uncertainty set. The constraint is linear, so the resulting model is an LP
whenever the rest of the problem is (see [`NoRisk`](@ref)).

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\epsilon \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\epsilon``: Radius of the ``\\ell_1`` uncertainty set.
  - ``\\boldsymbol{\\sigma}``: Per-asset scaling (`sd`); ``\\boldsymbol{1}`` when `sd` is `nothing`.

Two ``\\ell_1`` terms whose `sd` differ do **not** collapse into one: the sum of their
penalties is not a single infinity norm unless every `sd` matches.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::L1UncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
    sc = get_constraint_scale(model)
    w = get_w(model)
    mu = something(ucs.mu, mu)
    sd = ucs.sd
    sw = isnothing(sd) ? w : sd .* w
    t_l1ucs = state_set!(model, Symbol(""), :t_l1ucs_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :l1ucs_ret_, i,
               JuMP.@constraint(model,
                                [sc * t_l1ucs;
                                 sc * sw] in JuMP.MOI.NormInfinityCone(1 + length(w))))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, dot_scalar(mu, w) - ucs.eps * t_l1ucs))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, false
end
"""
    set_ucs_return_constraints!(model, i, ucs::SignedL1UncertaintySet, mu, settings)

Build one term's signed-``\\ell_1``-robust return expression.

Introduces one epigraph variable per error sign. Because the objective maximises the return expression, each variable is driven down to its lower bound, so `t_sl1ucs_p` attains ``[\\max_i(-\\sigma_i w_i)]_+`` and `t_sl1ucs_m` attains ``[\\max_i(\\sigma_i w_i)]_+`` at the optimum. The constraints are linear.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\epsilon_{+} \\left[\\underset{i}{\\max}\\, (-\\sigma_i w_i)\\right]_{+} - \\epsilon_{-} \\left[\\underset{i}{\\max}\\, (\\sigma_i w_i)\\right]_{+}\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\epsilon_{+}``, ``\\epsilon_{-}``: Radii of the positive- and negative-error sides.
  - ``\\boldsymbol{\\sigma}``: Per-asset scaling (`sd`); ``\\boldsymbol{1}`` when `sd` is `nothing`.

Modelling this worst case directly keeps the long-short problem *coupled*, so it does not need the decoupling of equations (27) and (28) of [quintile](@cite), nor the complementary-support caveat its Remark 12 attaches to recombining them.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::SignedL1UncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
    sc = get_constraint_scale(model)
    w = get_w(model)
    mu = something(ucs.mu, mu)
    sd = ucs.sd
    sw = isnothing(sd) ? w : sd .* w
    t_sl1ucs_p = state_set!(model, Symbol(""), :t_sl1ucs_p_, i,
                            JuMP.@variable(model, lower_bound = 0))
    t_sl1ucs_m = state_set!(model, Symbol(""), :t_sl1ucs_m_, i,
                            JuMP.@variable(model, lower_bound = 0))
    state_set!(model, Symbol(""), :sl1ucs_ret_p_, i,
               JuMP.@constraint(model, sc * (-sw .- t_sl1ucs_p) <= 0))
    state_set!(model, Symbol(""), :sl1ucs_ret_m_, i,
               JuMP.@constraint(model, sc * (sw .- t_sl1ucs_m) <= 0))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model,
                                      dot_scalar(mu, w) - ucs.ep * t_sl1ucs_p -
                                      ucs.en * t_sl1ucs_m))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, false
end
function set_return_constraints!(model::JuMP.Model, i,
                                 pret::ArithmeticReturn{<:Any, <:UcSE_UcS, <:Any},
                                 pr::AbstractPriorResult; rd::ReturnsResult, kwargs...)
    settings = pret.settings
    # The set is a neighbourhood of the quantity it was calibrated on, so it names the
    # centre. The term's own field and then the prior are the fallbacks (ADR 0050).
    fb = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    ret, mu, robust = set_ucs_return_constraints!(model, i, mu_ucs(pret.ucs, rd; kwargs...),
                                                  fb, settings)
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    return mu, robust
end
function set_return_constraints!(model::JuMP.Model, i, pret::LogarithmicReturn,
                                 pr::AbstractPriorResult; kwargs...)
    k = get_k(model)
    sc = get_constraint_scale(model)
    settings = pret.settings
    X = set_portfolio_returns!(model, pr.X)
    T = length(X)
    t_elog_ret = state_set!(model, Symbol(""), :t_elog_ret_, i,
                            JuMP.@variable(model, [1:T]))
    wi = nothing_scalar_array_selector(pret.w, pr.w)
    wi = get_observation_weights(wi, X)
    ret = if isnothing(wi)
        JuMP.@expression(model, Statistics.mean(t_elog_ret))
    else
        JuMP.@expression(model, Statistics.mean(t_elog_ret, wi))
    end
    state_set!(model, Symbol(""), :ret_, i, ret)
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    kret = state_set!(model, Symbol(""), :kret_, i, JuMP.@expression(model, k .+ X))
    state_set!(model, Symbol(""), :elog_ret_ret_, i,
               JuMP.@constraint(model, [j = 1:T],
                                [sc * t_elog_ret[j], sc * k, sc * kret[j]] in
                                JuMP.MOI.ExponentialCone()))
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    # A logarithmic term holds no per-asset quantity, which forces the ratio's risk form.
    return nothing, false
end
function set_return_constraints!(model::JuMP.Model, i, pret::NoReturn,
                                 ::AbstractPriorResult; kwargs...)
    settings = pret.settings
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, zero(JuMP.AffExpr)))
    # No charge is applied here, and this is why `settings.fee` and `settings.mic` are inert:
    # the term's expression is identically zero by construction, every guard `NoReturn`
    # carries rests on that, and a fee subtracted here would make it non-zero.
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    # No per-asset quantity, and no robust cone.
    return nothing, false
end
"""
    add_to_objective_penalty!(model::JuMP.Model, expr)

Accumulate an expression into the objective penalty term `op` in the JuMP model.

Creates the `op` expression if it does not yet exist, then adds `expr` to it.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `expr`: JuMP expression to add to the penalty.

# Returns

  - `nothing`.

# Related

  - [`add_penalty_to_objective!`](@ref)
  - [`set_portfolio_objective_function!`](@ref)
"""
function add_to_objective_penalty!(model::JuMP.Model, expr)
    op = if !shared_has(model, :op) && isa(expr, JuMP.AffExpr)
        JuMP.@expression(model, op, zero(JuMP.AffExpr))
    elseif !shared_has(model, :op) && isa(expr, JuMP.QuadExpr)
        JuMP.@expression(model, op, zero(JuMP.QuadExpr))
    elseif shared_has(model, :op)
        shared_get(model, :op)
    else
        throw(ArgumentError("expr must be a JuMP.AffExpr or JuMP.QuadExpr"))
    end
    if isa(expr, JuMP.QuadExpr) && !isa(op, JuMP.QuadExpr)
        JuMP.unregister(model, :op)
        op = JuMP.@expression(model, op, JuMP.QuadExpr(op))
    end
    JuMP.add_to_expression!(op, expr)
    return nothing
end
"""
    add_penalty_to_objective!(model::JuMP.Model, factor::Integer, expr)

Add the accumulated objective penalty to the main objective expression.

If an `op` penalty term exists in the model, adds `factor * op` to `expr`. Returns `expr` unchanged if no penalty term has been registered.

A quadratic penalty cannot be accumulated into an affine objective in-place, so an affine `expr` is promoted to a `JuMP.QuadExpr` when `op` is quadratic. Promotion allocates a new expression, which is why the caller must use the returned value rather than the one it passed in.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `factor::Integer`: Sign factor (`1` for minimisation, `-1` for maximisation).
  - `expr`: JuMP objective expression.

# Returns

  - `expr`: The objective expression with the penalty added, promoted to a `JuMP.QuadExpr` if that was needed to hold a quadratic penalty.

# Related

  - [`add_to_objective_penalty!`](@ref)
  - [`set_portfolio_objective_function!`](@ref)
"""
function add_penalty_to_objective!(model::JuMP.Model, factor::Integer, expr)
    if !shared_has(model, :op)
        return expr
    end
    op = shared_get(model, :op)
    if !isa(expr, JuMP.QuadExpr) && isa(op, JuMP.QuadExpr)
        JuMP.unregister(model, :obj_expr)
        expr = JuMP.@expression(model, obj_expr, JuMP.QuadExpr(expr))
    end
    JuMP.add_to_expression!(expr, factor, op)
    return expr
end
"""
    set_portfolio_objective_function!(model, obj, optimiser, attrs)

Set the portfolio objective function in the JuMP model.

Dispatches on the objective function type to build the appropriate JuMP objective expression, then folds in the [Objective Penalty](@ref add_to_objective_penalty!) accumulated by the regularisation, soft-constraint, and custom-term builders.

Custom objective terms are applied *before* the penalty is folded in, because they contribute to the same accumulator: `add_penalty_to_objective!` applies the sign factor matching this method's optimisation sense, so a contribution always worsens the objective regardless of which objective is being built. See ADR 0036.

The return term is **not** a parameter of this function. It used to be passed positionally so that a [`LogarithmicReturn`](@ref) ratio problem could dispatch to its own method, because the logarithmic builder registered `sr_elog_ret_risk` where every other branch registered `sr_risk`. The hoisted ratio constraint registers one name for every term shape, so the two methods collapse into one keyed on the presence of `:sr_risk`, and the positional lost its only dispatch reason.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `obj::ObjectiveFunction`: Portfolio objective (e.g. [`MinimumRisk`](@ref), [`MaximumUtility`](@ref)). This is the objective *being built*, which during a [`Frontier`](@ref) sweep differs from the one the user declared.
  - `optimiser::JuMPOptimisationEstimator`: The outer optimisation estimator (e.g. the [`MeanRisk`](@ref) itself). Supplies the custom objective terms as `optimiser.opt.cobj`.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed constraint and prior bundle.

# Returns

  - `nothing`.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`add_penalty_to_objective!`](@ref)
  - [`add_custom_objective_term!`](@ref)
"""
function set_portfolio_objective_function!(model::JuMP.Model, obj::MinimumRisk,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    risk = get_risk(model)
    JuMP.@expression(model, obj_expr, risk)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumUtility,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    ret = get_ret(model)
    risk = get_risk(model)
    l = obj.l
    JuMP.@expression(model, obj_expr, ret - l * risk)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    if shared_has(model, :sr_risk)
        ret = get_ret(model)
        k = get_k(model)
        rf = obj.rf
        JuMP.@expression(model, obj_expr, ret - rf * k)
        add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
        obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
        JuMP.@objective(model, Max, so * obj_expr)
    else
        risk = get_risk(model)
        JuMP.@expression(model, obj_expr, risk)
        add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
        obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
        JuMP.@objective(model, Min, so * obj_expr)
    end
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumReturn,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    ret = get_ret(model)
    JuMP.@expression(model, obj_expr, ret)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumElementReturn,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    ret = state_get(model, Symbol(""), :ret_, obj.i)
    JuMP.@expression(model, obj_expr, ret)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end

export JuMPReturnsSettings, ArithmeticReturn, LogarithmicReturn, NoReturn, MinimumRisk,
       MaximumUtility, MaximumRatio, MaximumReturn, bounds_returns_estimator
