"""
$(DocStringExtensions.TYPEDEF)

Carries one return term's own weight in the return sum, its own lower bound, and the two charges netted out of it.

A [`JuMPOptimiser`](@ref) takes one return term or a vector of them. The settings hold the
data of one term, not of the optimiser. They hold these items:

  - The weight of the term in the return expression.
  - The lower bound of the term.
  - Whether the term enters the return expression.
  - Which of the two portfolio charges the term nets.

Every return estimator holds the settings in its first field, `settings`. A risk measure holds
its [`RiskMeasureSettings`](@ref) in the same way.

The library does not normalise the scales. Two terms at `scale = 1` charge their flagged fees
two times, and two terms at `scale = 0.5` charge them one time. `fee` and `mic` are separate
flags, because the market impact cost already constrains the budget. Thus a term can net the
fees and not the impact cost. A term with `rte = false` stays out of the return expression,
and its own `lb` still binds. Use it to state a term that only constrains the portfolio.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ret} &= \\sum_{i \\,:\\, \\mathrm{rte}_i} s_i\\, \\mathrm{ret}_i\\,, \\\\
\\mathrm{ret}_i &\\geq \\mathrm{lb}_i \\quad \\forall i\\,.
\\end{align}
```

Where:

  - $(math_dict[:ret_model])
  - $(math_dict[:ret_i_term])
  - $(math_dict[:s_i_ret])
  - ``\\mathrm{rte}_i``: The `rte` flag of term ``i``.
  - ``\\mathrm{lb}_i``: The lower bound `lb` of term ``i``. A term with no bound takes no row.

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

It is the return-side twin of [`VecRM`](@ref). An optimiser holds several return terms as
this vector, so a function that reaches one term reaches all of them.
[`factory`](@ref) and [`port_opt_view`](@ref) need no method of their own for it, because their
generic vector methods rebuild and view each term in turn. Each term keeps its own settings,
its own uncertainty set and its own characteristic. An outer `ucs` argument is the same for all
of the terms, because [`pipe_route`](@ref) routes a bare mean uncertainty set to a single term
alone.

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`JRE_VecJRE`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
const VecJRE = AbstractVector{<:JuMPReturnsEstimator}
"""
    const JRE_VecJRE = Union{<:JuMPReturnsEstimator, <:VecJRE}

Field bound for [`JuMPOptimiser`](@ref)'s `ret` slot, which takes one return term or several.

It is the return-side twin of [`RM_VecRM`](@ref).

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`VecJRE`](@ref)
"""
const JRE_VecJRE = Union{<:JuMPReturnsEstimator, <:VecJRE}
"""
    const ArithRetMu = Union{<:Num_VecNum, <:AbstractExpectedReturnsEstimator, <:AbstractPriorEstimator}

Field bound for [`ArithmeticReturn`](@ref)'s `mu` slot, which holds the expected returns or the Estimator that computes them, a [`DeferredQuantity`](@ref).

The bound is [`MuSlot`](@ref) less a [`VecScalar`](@ref), for two reasons. A `VecScalar` is the centre target of a moment risk measure. The return expression is `dot_scalar(mu, w)`, which takes a number or a vector. A `VecScalar` is also an [`AbstractResult`](@ref), and an Estimator must not hold one.

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

The term takes an optional uncertainty set on the mean vector, a box, ellipsoidal,
``\\ell_1``, signed ``\\ell_1`` or norm-ball set. With a set, the term is the worst-case
expected return over the set instead of the point estimate, which gives a robust return.

The `ucs` field takes a mean uncertainty set that [`mu_ucs`](@ref) built, or an estimator of
one. A built set is the simpler route, as a built [`sigma_ucs`](@ref) result is for
[`UncertaintySetVariance`](@ref). An estimator builds the set when the model is built, so the
optimiser must pass it the returns data `rd`. The term takes its centre from the first of
these three items that exists:

 1. The centre that the set carries.
 2. The field `mu`.
 3. The expected returns of the prior.

A Deferred Quantity in `mu` resolves when a set with its own centre outranks it. The term then
does not use it, as for a stated vector. The lower bound of the term is `settings.lb`.

# Mathematical definition

```math
\\begin{align}
r(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}\\,, \\\\
\\hat{r}(\\boldsymbol{w}) &= \\underset{\\boldsymbol{\\mu} \\in \\mathcal{U}}{\\min}\\; \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``r(\\boldsymbol{w})``: Expected portfolio return, the term without a set.
  - $(math_dict[:rhat_worst]) It is the term with a set.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\mathcal{U}``: The mean uncertainty set `ucs`.

The five [`set_ucs_return_constraints!`](@ref) methods state the closed form of
``\\hat{r}(\\boldsymbol{w})`` for each shape.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ArithmeticReturn(;
        settings::JuMPReturnsSettings = JuMPReturnsSettings(),
        ucs::Option{<:UcSE_UcS} = nothing,
        mu::Option{<:ArithRetMu} = nothing,
        mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot()
    ) -> ArithmeticReturn

Keywords correspond to the struct's fields.

## Validation

  - If `ucs` is an `EllipsoidalUncertaintySet` or a `NormBallUncertaintySet`: must be parameterised by `MuUncertaintySetClass`.
  - If `mu` is a number: `isfinite(mu)`.
  - If `mu` is a vector: `!isempty(mu)` and `all(isfinite, mu)`.

!!! warning

    A stated `mu` stays fixed. A Cross-Validation fold or a subset view receives the vector of the whole universe, so the vector does not follow the refit that the optimisation runs on. To make it follow the fit, put a Deferred Quantity in `mu`, or leave the slot `nothing` so that the prior supplies it.

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
    """
    Square-root algorithm of the matrix of an [`EllipsoidalUncertaintySet`](@ref) in `ucs`, or `nothing` for the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. The default takes the square root of the eigendecomposition of a singular positive semidefinite matrix. [`matrix_square_root`](@ref) states each algorithm. The other sets read no square root.
    """
    mtx_sqrt
    function ArithmeticReturn(settings::JuMPReturnsSettings, ucs::Option{<:UcSE_UcS},
                              mu::Option{<:ArithRetMu},
                              mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm})
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
        return new{typeof(settings), typeof(ucs), typeof(mu), typeof(mtx_sqrt)}(settings,
                                                                                ucs, mu,
                                                                                mtx_sqrt)
    end
end
function ArithmeticReturn(; settings::JuMPReturnsSettings = JuMPReturnsSettings(),
                          ucs::Option{<:UcSE_UcS} = nothing,
                          mu::Option{<:ArithRetMu} = nothing,
                          mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())
    return ArithmeticReturn(settings, ucs, mu, mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve a Deferred Quantity in [`ArithmeticReturn`](@ref)'s `mu` slot against prior result `pr`.

The estimator has one slot that the prior can supply, so the slot itself admits the Estimator, and the method resolves that one slot. Every `JuMP` path calls it through [`factory`](@ref), which [`processed_jump_optimiser_attributes`](@ref) calls on `opt.ret` before it builds a model. A risk measure has a second entry point, and a return term needs none.

# Algorithm

 1. When `rt.mu` is not a [`DeferredQuantity`](@ref), return `rt` unchanged.
 2. Compute the vector with [`resolve_slot`](@ref) against `pr`.
 3. Rebuild `rt` with the vector in `mu`, through [`rebuild_with_slots`](@ref), and return it.

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
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu),
                            mtx_sqrt = rt.mtx_sqrt)
end
function factory(rt::ArithmeticReturn, pr::AbstractPriorResult,
                 ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
    rt = resolve_deferred_quantities(rt, pr)
    return ArithmeticReturn(; settings = rt.settings, ucs = ucs_selector(rt.ucs, ucs),
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu),
                            mtx_sqrt = rt.mtx_sqrt)
end
function factory(rt::ArithmeticReturn, ucs::UcSE_UcS, pr::AbstractPriorResult; kwargs...)
    rt = resolve_deferred_quantities(rt, pr)
    return ArithmeticReturn(; settings = rt.settings, ucs = ucs_selector(rt.ucs, ucs),
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu),
                            mtx_sqrt = rt.mtx_sqrt)
end
function factory(rt::ArithmeticReturn, ucs::UcSE_UcS, args...; kwargs...)
    # No prior in hand, so a Deferred Quantity cannot resolve here. It travels on unchanged
    # and the prior-carrying `factory` the sub-problem runs resolves it.
    return ArithmeticReturn(; settings = rt.settings, ucs = ucs_selector(rt.ucs, ucs),
                            mu = rt.mu, mtx_sqrt = rt.mtx_sqrt)
end
function port_opt_view(r::ArithmeticReturn, i, args...)
    uset = port_opt_view(r.ucs, i)
    # A Deferred Quantity crosses the view unsliced: `nothing_scalar_array_view` is the
    # identity on an Estimator. It then computes on the subset, which is the whole
    # fold-stability argument for the feature.
    mu = nothing_scalar_array_view(r.mu, i)
    return ArithmeticReturn(; settings = r.settings, ucs = uset, mu = mu,
                            mtx_sqrt = r.mtx_sqrt)
end
"""
    no_bounds_returns_estimator(r, args...)

Create a version of the return term with its lower bound removed.

The sub-problems of the frontier and of [`NearOptimalCentering`](@ref) call it, because their
corner solves must range over the whole feasible set.

The function removes `lb`, and it removes `ucs` when `flag` is `false`. The term keeps all
other fields, `mu` included. Without `mu`, the term takes the prior's own vector as its
centre. But a set describes the neighbourhood of the one vector that it was calibrated on.
Also, with several terms, each term then gives the same corner.

# Arguments

  - `r`: One return term, or a vector of them.
  - `flag::Bool`: When `false`, the copy drops the uncertainty set too.

# Returns

  - The term, or the vector of terms, without bounds.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`no_bounds_optimiser`](@ref)
"""
function no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool = true)
    return ArithmeticReturn(; settings = no_bounds_returns_settings(r.settings),
                            ucs = ifelse(flag, r.ucs, nothing), mu = r.mu,
                            mtx_sqrt = r.mtx_sqrt)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a copy of `settings` with its lower bound cleared.

`scale`, `rte`, `fee` and `mic` are not bounds, so the copy keeps them. A corner solve must
charge the same fees and weigh the same terms as the sweep that it starts.

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

Return a copy of return term `r` with its `scale` set to `one(scale)`.

`scale` is the weight of the term in a return expression that several terms build. One term
has nothing to weigh against, so the route for a single term drops the weight. `lb`, `rte`,
`fee` and `mic` are not weights, so the copy keeps them. The bound still binds on the term's
own expression, and the term still charges the same fees. This is the return-side twin of
[`unit_scale_risk_measure`](@ref).

# Algorithm

 1. When `isone(r.settings.scale)`, return `r` unchanged, which allocates nothing.
 2. Otherwise build a [`JuMPReturnsSettings`](@ref) with `scale = one(scale)` and the other four fields of `r.settings`.
 3. Return a copy of `r` that holds the new settings.

# Arguments

  - `r`: A [`JuMPReturnsEstimator`](@ref).

# Returns

  - The return term with a unit scale.

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

Computes the portfolio return as the mean logarithmic return, the Kelly criterion's objective.

The term takes optional observation weights. Unlike [`ArithmeticReturn`](@ref), it holds no
per-asset quantity, which is why this family is named for the return term and not for the
characteristic. [`expected_return`](@ref) computes the same quantity in closed form, and the
model's `:ret` agrees with it when the objective raises the return. The formulation is on
[`set_return_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
r(\\boldsymbol{w}) &= \\frac{\\sum_{t=1}^{T} w_{t} \\ln\\left(1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}\\right)}{\\sum_{t=1}^{T} w_{t}}\\,.
\\end{align}
```

Where:

  - ``r(\\boldsymbol{w})``: Mean logarithmic portfolio return.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:w_port])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is ``1`` when `w` is `nothing`.
  - $(math_dict[:T])

!!! warning

    This is the mean logarithmic return, not the geometric mean net return
    ``\\prod_t (1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w})^{1/T} - 1``. The two are one
    ``\\exp(\\cdot) - 1`` apart, so they order portfolios alike but carry different units.
    `settings.lb` and [`MaximumRatio`](@ref)'s `rf` are therefore stated in log units.
    Apply `exp(r) - 1` to read the value as a net return.

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

`NoReturn` computes nothing. Its value-level twin returns zero, and its formulation adds a
zero return expression. It is the return-side twin of [`NoRisk`](@ref). An optimiser with no
return term states that with it, and then no unused term changes the model class.

[`set_return_constraints!`](@ref) runs in the shared Model Assembly for every optimiser, and
[`JuMPOptimiser`](@ref)'s `ret` slot defaults to [`ArithmeticReturn`](@ref).
[`RiskBudgeting`](@ref), [`RelaxedRiskBudgeting`](@ref) and [`FactorRiskContribution`](@ref)
never read `:ret`. With the default term, they build the whole expression and do not use it.
The expression includes the cones of a mean uncertainty set. An unused term adds rows that the model does not
need, and it can force a conic solver onto a linear programme. `NoReturn` keeps such a model
in its own class, which is the main use of the type. It also lets a caller state that there is
no return term, where the alternative is `settings.rte = false` on every term.

`NoReturn` is coherent only where nothing reads the return expression:

| Optimiser and objective                                     | `NoReturn` |
|:----------------------------------------------------------- |:---------- |
| [`RiskBudgeting`](@ref), [`RelaxedRiskBudgeting`](@ref)     | ok         |
| [`FactorRiskContribution`](@ref) + [`MinimumRisk`](@ref)    | ok         |
| [`FactorRiskContribution`](@ref) + [`MaximumUtility`](@ref) | ok         |
| [`FactorRiskContribution`](@ref) + [`MaximumReturn`](@ref)  | throws     |
| [`FactorRiskContribution`](@ref) + [`MaximumRatio`](@ref)   | throws     |
| [`MeanRisk`](@ref) + [`MinimumRisk`](@ref)                  | ok         |
| [`MeanRisk`](@ref) + [`MaximumUtility`](@ref)               | ok         |
| [`MeanRisk`](@ref) + [`MaximumReturn`](@ref)                | throws     |
| [`MeanRisk`](@ref) + [`MaximumRatio`](@ref)                 | throws     |
| [`NearOptimalCentering`](@ref)                              | throws     |

[`RiskBudgeting`](@ref) and [`RelaxedRiskBudgeting`](@ref) hold no objective, so nothing in
them can read `:ret`. [`FactorRiskContribution`](@ref) holds one, so it refuses the same two
objectives as [`MeanRisk`](@ref).

[`assert_no_return_objective_compatibility`](@ref) makes the objective refusals when the model
is built. Without the refusal, a [`MaximumReturn`](@ref) objective is zero everywhere, and the
solver returns an arbitrary feasible portfolio and reports success. The numerator of a
[`MaximumRatio`](@ref) objective is zero. [`assert_return_term_required`](@ref) makes the
[`NearOptimalCentering`](@ref) refusal in its constructor. That model is infeasible, not
degenerate, because its barrier constrains `exp(log_ret) <= ret - rt`. With no return term,
`ret` and `rt` are both zero. Each refusal also applies to a vector of terms that all carry
`settings.rte = false`. The guards test the expression, not the type of the term.

The term holds no per-asset quantity, so `settings.scale`, `settings.fee` and `settings.mic`
have no effect. A scaled zero is still zero. The builder subtracts no charge, because a charge
makes the expression non-zero, and each guard above uses the zero. A `settings.lb` is legal,
but it binds on a quantity that is always zero, so a positive bound makes the model
infeasible. [`NoRisk`](@ref)'s `settings.ub` has the same effect.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NoReturn(; settings::JuMPReturnsSettings = JuMPReturnsSettings()) -> NoReturn

Keywords correspond to the struct's fields.

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

The function pairs bounds with terms one to one. One term takes a scalar bound or `nothing`.
A vector of ``n`` terms takes `nothing`, which clears all ``n`` bounds, or a vector of ``n``
bounds, one for each term. It refuses a single number for a vector of terms. The bound binds on
each term's own expression, and the terms can have different units. Thus no check can find
whether one number has the same meaning for all of the terms.

# Arguments

  - `r`: One return term, or a vector of them.
  - `lb`: The lower bound, which is a number, `nothing`, or a vector with one entry for each term.

# Validation

  - If `r` is a vector and `lb` is a number, the function raises an `ArgumentError`.
  - If `r` and `lb` are vectors: `length(lb) == length(r)`. Otherwise the function raises a `DimensionMismatch`.

# Returns

  - The term, or the vector of terms, with the new lower bound.

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
