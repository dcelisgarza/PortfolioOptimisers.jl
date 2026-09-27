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
  - A `ucs` that carries its own centre outranks `mu`, and `mu` outranks `pr.mu`. A Deferred Quantity is a state of the `mu` rung, not a rung of its own: beside a centre-carrying set it is resolved and then goes unused, exactly as a stated vector does.
  - The lower bound lives on `settings.lb`, not on the term itself.

## Validation

  - If `ucs` is an `EllipsoidalUncertaintySet` or a `NormBallUncertaintySet`: must be parameterised by `MuUncertaintySetClass`.
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
                          EllipsoidalUncertaintySet{<:Any, <:Any, <:MuUncertaintySetClass}),
                      ArgumentError("ucs must be parameterised by MuUncertaintySetClass, got $(typeof(ucs))"))
        elseif isa(ucs, NormBallUncertaintySet)
            @argcheck(isa(ucs,
                          NormBallUncertaintySet{<:Any, <:Any, <:Any,
                                                 <:MuUncertaintySetClass}),
                      ArgumentError("ucs must be parameterised by MuUncertaintySetClass, got $(typeof(ucs))"))
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
prior's own vector — a set is a neighbourhood of the one quantity it was calibrated on, not a
shared default — and with several terms it would collapse every one of them onto the same
corner.

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
    ``\\exp(\\cdot) - 1`` apart, so they order portfolios alike but carry different units.
    `settings.lb` and [`MaximumRatio`](@ref)'s `rf` are therefore stated in log units.
    Apply `exp(r) - 1` to read the value as a net return.

The cone is a relaxation that a maximising objective closes, so the model's own `:ret` sits
within solver tolerance of the value-level figure. [`expected_return`](@ref) computes the
same quantity in closed form.

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
test the state of the expression and not the type of the term.

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

export JuMPReturnsSettings, ArithmeticReturn, LogarithmicReturn, NoReturn,
       bounds_returns_estimator
