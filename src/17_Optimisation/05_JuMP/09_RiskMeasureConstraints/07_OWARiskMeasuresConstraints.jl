"""
$(DocStringExtensions.TYPEDSIGNATURES)

Set up the OWA portfolio returns variable and equality constraint.

Introduces a vector variable `owa` of length `T` and adds the equality constraint
`sc * (net_X - owa) == 0`. Returns the existing `owa` if already present.

# Arguments

  - $(arg_dict[:model])
  - `X::MatNum`: Asset returns matrix (`T × N`).

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `owa`: JuMP vector variable of length `T` for OWA portfolio returns.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
"""
function set_owa_constraints!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))
    return state_build!(model, prefix, :owa) do
        sc = get_constraint_scale(model)
        net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
        T = get_T(model)
        owa = JuMP.@variable(model, [1:T])
        state_set!(model, prefix, :owac, JuMP.@constraint(model, sc * (net_X - owa) == 0))
        return owa
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Ordered Weights Array (OWA) risk constraints to `model` using the exact formulation.

The exact formulation is the linear programme that is dual to the assignment problem which
sorts the sample. It adds two `T × 1` auxiliary variables and one `T × T` block of linear
constraints, and the risk is the sum of the two auxiliary variables. The block costs
``T^{2}`` constraints. For a long sample, [`SortingNetworkOrderedWeightsArray`](@ref) gives the
same exact risk at ``O(T \\log^{2} T)`` rows.

The programme pairs the largest weight with the largest sorted return, so it attains
``\\mathrm{sort}(\\boldsymbol{\\omega})^{\\intercal} \\mathrm{sort}(\\hat{\\boldsymbol{r}})``. This equals the OWA
risk when, and only when, ``\\boldsymbol{\\omega}`` is monotonic non-decreasing, which is also the
condition for the risk measure to be convex. Every weight builder in this package returns
such a vector.

# Mathematical definition

```math
\\begin{align}
\\hat{r}_{t} &= \\boldsymbol{x}_{t}^{\\intercal} \\boldsymbol{w}\\\\
\\mathrm{OWA}(\\boldsymbol{w}) &= \\boldsymbol{\\omega}^{\\intercal} \\mathrm{sort}\\left(\\hat{\\boldsymbol{r}}\\right)\\\\
&= \\begin{cases}
\\underset{\\boldsymbol{a},\\, \\boldsymbol{b}}{\\min} & \\sum\\limits_{t=1}^{T} \\left(a_{t} + b_{t}\\right)\\\\
\\text{s.t.} & \\hat{r}_{i} \\omega_{j} - a_{j} - b_{i} \\leq 0 \\quad \\forall i,\\, j = 1,\\, \\ldots,\\, T\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\mathrm{OWA}(\\boldsymbol{w})``: Is the ordered weighted average risk of the portfolio.
  - ``\\boldsymbol{\\omega}``: Is the OWA weight vector, `r.w` if it is a vector, `r.w(T)` if it is a callable.
  - ``\\hat{\\boldsymbol{r}}``: Is the vector of net portfolio returns, which ``\\mathrm{sort}`` places in ascending order.
  - ``\\boldsymbol{x}_{t}``: Is the `N × 1` vector of asset returns at time ``t``.
  - ``\\boldsymbol{w}``: Is the `N × 1` vector of portfolio weights.
  - ``\\boldsymbol{a},\\, \\boldsymbol{b}``: Are the two `T × 1` auxiliary variables of the assignment dual.
  - ``T``: Is the total number of observations.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArray{<:Any, <:Any, <:ExactOrderedWeightsArray}`: The OWA risk measure
    with the exact formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Validation

  - The weight vector has one entry for each observation, through [`owa_weights`](@ref). Raises a `DimensionMismatch`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `owa_risk`: The OWA risk expression added to the model.

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`set_owa_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ExactOrderedWeightsArray},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    T = get_T(model)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    ovec = range(one(eltype(X)), one(eltype(X)); length = T)
    owa_a = state_set!(model, prefix, :owa_a_, i, JuMP.@variable(model, [1:T]))
    owa_b = state_set!(model, prefix, :owa_b_, i, JuMP.@variable(model, [1:T]))
    owa_risk = state_set!(model, prefix, :owa_risk_, i,
                          JuMP.@expression(model, sum(owa_a + owa_b)))
    owa_w = owa_weights(r.w, T, :w)
    state_set!(model, prefix, :cowa_, i,
               JuMP.@constraint(model,
                                sc * (owa * transpose(owa_w) - ovec * transpose(owa_a) -
                                      owa_b * transpose(ovec)) in JuMP.Nonpositives()))
    set_risk_bounds_and_expression!(model, opt, owa_risk, r.settings, :owa_risk_, i;
                                    prefix = prefix)
    return owa_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `OrderedWeightsArrayRange` using the exact OWA formulation
to `model`.

The exact formulation fuses the two tails rather than duplicating them. It forms the single
weight vector ``\\boldsymbol{\\omega}_{1} - \\boldsymbol{\\omega}_{2}`` and builds one assignment dual
over it, so the range costs the same `T × T` block as a single tail. [`OrderedWeightsArrayRange`](@ref)
holds `r.w2` reversed, so the difference is the
range weight vector.

# Mathematical definition

```math
\\begin{align}
\\hat{r}_{t} &= \\boldsymbol{x}_{t}^{\\intercal} \\boldsymbol{w}\\\\
\\boldsymbol{\\omega} &= \\boldsymbol{\\omega}_{1} - \\boldsymbol{\\omega}_{2}\\\\
\\mathrm{OWA}_{\\mathrm{rg}}(\\boldsymbol{w}) &= \\boldsymbol{\\omega}^{\\intercal} \\mathrm{sort}\\left(\\hat{\\boldsymbol{r}}\\right)\\\\
&= \\begin{cases}
\\underset{\\boldsymbol{a},\\, \\boldsymbol{b}}{\\min} & \\sum\\limits_{t=1}^{T} \\left(a_{t} + b_{t}\\right)\\\\
\\text{s.t.} & \\hat{r}_{i} \\omega_{j} - a_{j} - b_{i} \\leq 0 \\quad \\forall i,\\, j = 1,\\, \\ldots,\\, T\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\mathrm{OWA}_{\\mathrm{rg}}(\\boldsymbol{w})``: Is the ordered weighted average range risk of the portfolio.
  - ``\\boldsymbol{\\omega}_{1}``: Is the loss-tail OWA weight vector, `r.w1` if it is a vector, `r.w1(T)` if it is a callable.
  - ``\\boldsymbol{\\omega}_{2}``: Is the reversed gain-tail OWA weight vector, `r.w2` if it is a vector, `r.w2(T)` if it is a callable.
  - ``\\hat{\\boldsymbol{r}}``: Is the vector of net portfolio returns, which ``\\mathrm{sort}`` places in ascending order.
  - ``\\boldsymbol{a},\\, \\boldsymbol{b}``: Are the two `T × 1` auxiliary variables of the assignment dual.
  - ``T``: Is the total number of observations.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ExactOrderedWeightsArray}`: The
    OWA range risk measure with exact formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Validation

  - Each tail weight vector has one entry for each observation, through [`owa_weights`](@ref). Raises a `DimensionMismatch`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `owa_range_risk`: The OWA range risk expression added to the model.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`set_owa_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:ExactOrderedWeightsArray},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    T = get_T(model)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    ovec = range(one(eltype(X)), one(eltype(X)); length = T)
    owa_a = state_set!(model, prefix, :owa_range_a_, i, JuMP.@variable(model, [1:T]))
    owa_b = state_set!(model, prefix, :owa_range_b_, i, JuMP.@variable(model, [1:T]))
    owa_range_risk = state_set!(model, prefix, :owa_range_risk_, i,
                                JuMP.@expression(model, sum(owa_a + owa_b)))
    owa_w1 = owa_weights(r.w1, T, :w1)
    owa_w2 = owa_weights(r.w2, T, :w2)
    owa_w = owa_w1 - owa_w2
    state_set!(model, prefix, :cowa_range_, i,
               JuMP.@constraint(model,
                                sc * (owa * transpose(owa_w) - ovec * transpose(owa_a) -
                                      owa_b * transpose(ovec)) in JuMP.Nonpositives()))
    set_risk_bounds_and_expression!(model, opt, owa_range_risk, r.settings,
                                    :owa_range_risk_, i; prefix = prefix)
    return owa_range_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `OrderedWeightsArray` using the approximate OWA formulation
to `model`.

The exact formulation orders the sample with a `T × T` block of constraints. This one drops
that block. It keeps only the properties of the OWA weight vector that any ordering leaves
unchanged — the minimum, the maximum, the sum, and one p-norm for each entry of `r.alg.p` —
and it writes each p-norm as a power cone. The cost falls to `T × M` variables, where `M` is
the length of `r.alg.p`.

Every permutation of the weight vector satisfies those properties, so the feasible set is a
superset of the exact one and the risk is an upper bound on the exact OWA risk. The gap
closes as the weight vector approaches a line. [owa3](@citet) studies that linear case.
Tables 1 and 3 of [owa3](@cite) compare the objective value with the exact formulations for
the Gini mean difference and the tail Gini over samples of 500 to 1,500 observations. The
values agree to the printed precision in every cell but one, 12.94 % against 12.93 % in
Table 3. At 10,000 observations the exact formulations do not solve, so those rows hold no
comparison.

The gap was measured here against the functor at `T = 100`, `N = 8` with the default `p`. The
Gini mean difference is 0.06 % high, the tail Gini is 1.7e-5 % high, and the tail Gini range
is 0.37 % high. A fourth-order L-moment weight vector, which is not linear, is 4.5 % high, so
prefer [`SortingNetworkOrderedWeightsArray`](@ref) for a weight vector that is far from a line.

# Mathematical definition

```math
\\begin{align}
\\hat{r}_{t} &= \\boldsymbol{x}_{t}^{\\intercal} \\boldsymbol{w}\\\\
\\mathrm{OWA}(\\boldsymbol{w}) &\\approx \\begin{cases}
\\underset{t,\\, \\boldsymbol{\\nu},\\, \\boldsymbol{\\eta},\\, \\boldsymbol{\\varepsilon},\\, \\boldsymbol{\\psi},\\, \\boldsymbol{\\zeta},\\, \\boldsymbol{y}}{\\min} & c_{1} t - c_{2} \\boldsymbol{1}^{\\intercal} \\boldsymbol{\\nu} + c_{3} \\boldsymbol{1}^{\\intercal} \\boldsymbol{\\eta} + \\sum\\limits_{k \\in S} d_{k} y_{k}\\\\
\\text{s.t.} & \\hat{\\boldsymbol{r}} + t \\boldsymbol{1} - \\boldsymbol{\\nu} + \\boldsymbol{\\eta} - \\sum\\limits_{k \\in S} \\boldsymbol{\\varepsilon}_{k} = \\boldsymbol{0}\\\\
 & \\zeta_{k} + y_{k} - \\boldsymbol{1}^{\\intercal} \\boldsymbol{\\psi}_{k} = 0 \\quad \\forall k \\in S\\\\
 & \\left(-k \\zeta_{k},\\, \\dfrac{k}{k-1} \\psi_{k,\\, t},\\, \\varepsilon_{k,\\, t}\\right) \\in \\mathcal{P}_{3}^{1/k,\\, 1-1/k} \\quad \\forall k \\in S,\\, \\forall t = 1,\\, \\ldots,\\, T\\\\
 & \\boldsymbol{\\nu},\\, \\boldsymbol{\\eta},\\, \\boldsymbol{y} \\geq \\boldsymbol{0}
\\end{cases}\\\\
c_{1} &= \\boldsymbol{1}^{\\intercal} \\left(-\\boldsymbol{\\omega}\\right)\\\\
c_{2} &= \\min\\left(-\\boldsymbol{\\omega}\\right)\\\\
c_{3} &= \\max\\left(-\\boldsymbol{\\omega}\\right)\\\\
d_{k} &= \\lVert -\\boldsymbol{\\omega} \\rVert_{k} \\quad \\forall k \\in S\\\\
\\mathcal{P}_{3}^{\\alpha,\\, 1-\\alpha} &\\coloneqq \\left\\{\\boldsymbol{u} \\in \\mathbb{R}^{3} : u_{1}^{\\alpha} u_{2}^{1-\\alpha} \\geq \\lvert u_{3} \\rvert,\\, u_{1},\\, u_{2} \\geq 0\\right\\}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{\\omega}``: Is the OWA weight vector, `r.w` if it is a vector, `r.w(T)` if it is a callable.
  - ``\\hat{\\boldsymbol{r}}``: Is the vector of net portfolio returns, negated when `loss` is `false`.
  - ``S``: Is the set of p-norm orders, `r.alg.p`, or the empty set when at most one entry of ``\\boldsymbol{\\omega}`` lies strictly between its minimum and its maximum. `## Relaxation` gives the reason.
  - ``t``: Is the scalar auxiliary variable, `owa_t`.
  - ``\\boldsymbol{\\nu},\\, \\boldsymbol{\\eta}``: Are the `T × 1` non-negative auxiliary variables, `owa_nu` and `owa_eta`.
  - ``\\boldsymbol{\\varepsilon},\\, \\boldsymbol{\\psi}``: Are the `T × M` auxiliary variables, `owa_epsilon` and `owa_psi`.
  - ``\\boldsymbol{\\zeta},\\, \\boldsymbol{y}``: Are the `M × 1` auxiliary variables, `owa_z` and `owa_y`.
  - ``\\mathcal{P}_{3}^{\\alpha,\\, 1-\\alpha}``: Is the three-dimensional power cone.
  - ``T``: Is the total number of observations.
  - ``M``: Is the number of p-norm orders, `length(r.alg.p)`.

# JuMP formulation

## Variables

  - The net portfolio returns ``\\hat{\\boldsymbol{r}}``: Read, through [`set_net_portfolio_returns!`](@ref).
  - `owa_t_`, `owa_nu_`, `owa_eta_`, `owa_epsilon_`, `owa_psi_`, `owa_z_`, `owa_y_`: Created under no name, and stored under these keys with the index `i`. They are ``t``, ``\\boldsymbol{\\nu} \\geq 0``, ``\\boldsymbol{\\eta} \\geq 0``, ``\\boldsymbol{\\varepsilon}``, ``\\boldsymbol{\\psi}``, ``\\boldsymbol{\\zeta}`` and ``\\boldsymbol{y} \\geq 0``.

## Expressions

  - `aowa_risk_`: ``c_{1} t - c_{2} \\boldsymbol{1}^{\\intercal} \\boldsymbol{\\nu} + c_{3} \\boldsymbol{1}^{\\intercal} \\boldsymbol{\\eta} + \\sum_{k \\in S} d_{k} y_{k}``, the risk.
  - `neg_owa_z_owa_p_`: ``-k \\zeta_{k}`` for each ``k \\in S``.
  - `owa_p_o_owa_pm1_`: ``k / (k - 1)`` for each ``k \\in S``.

## Constraints

Every row is registered under no name, and stored under its key with the index `i`.

  - `ca1_owa_`: ``s_c \\left(\\hat{\\boldsymbol{r}} - \\boldsymbol{\\nu} + \\boldsymbol{\\eta} - \\sum_{k \\in S} \\boldsymbol{\\varepsilon}_{k} + t \\boldsymbol{1}\\right) = \\boldsymbol{0}``.
  - `ca2_owa_`: ``s_c \\left(\\zeta_{k} + y_{k} - \\boldsymbol{1}^{\\intercal} \\boldsymbol{\\psi}_{k}\\right) = 0`` for each ``k \\in S``.
  - `ca_owa_pcone_`: ``\\left(-s_c k \\zeta_{k},\\, s_c \\tfrac{k}{k-1} \\psi_{k,\\, t},\\, s_c \\varepsilon_{k,\\, t}\\right) \\in \\mathcal{P}_{3}^{1/k,\\, 1-1/k}`` for each ``k \\in S`` and each ``t``.

Where:

  - $(math_dict[:sc_scale])
  - The other symbols are those of the `Where:` list of `# Mathematical definition`.

## Relaxation

$(val_dict[:relax])

  - The bounded quantity is `aowa_risk_`. At its minimum over the auxiliary variables, it lies at or **above** the exact OWA risk of the same portfolio, ``\\boldsymbol{\\omega}^{\\intercal} \\mathrm{sort}(\\hat{\\boldsymbol{r}})``. The dual of the programme maximises over every vector with the minimum, the maximum, the sum and the p-norms of ``-\\boldsymbol{\\omega}``, and every reordering of ``-\\boldsymbol{\\omega}`` is such a vector.
  - The bound is exact when at most one entry of the weight vector lies strictly between its minimum and its maximum, as the vectors of [`owa_wr`](@ref) and of [`owa_cvar`](@ref) at every ``\\alpha`` do. The box and the sum then describe the convex hull of the reorderings, and every p-norm constraint is redundant. The method then sets ``S`` to the empty set, so ``M = 0`` and the model is a linear programme. The redundant cones would put the optimum at the apex of every cone, ``\\boldsymbol{\\varepsilon} = \\boldsymbol{0}`` and ``\\boldsymbol{y} = \\boldsymbol{0}``, where an interior point solver can stall. For other vectors the gap closes as the vector approaches a line, and the gaps above give its size.
  - The bound holds while the objective or a bound pulls `aowa_risk_` down, as a minimum-risk objective does. Under [`MaximumReturn`](@ref) with no risk bound, nothing pulls, and the value of `aowa_risk_` can lie further above. Read the exact figure back with [`expected_risk`](@ref).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArray{<:Any, <:Any, <:ApproxOrderedWeightsArray}`: The OWA risk
    measure with approximate formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Validation

  - The weight vector has one entry for each observation, through [`owa_weights`](@ref). Raises a `DimensionMismatch`. The formulation reads only the sum, the extremes and the norms of the weights, so without the check a vector of the wrong length solves a different problem with no error.

# Keyword arguments

  - `loss::Bool`: If `true` (default), the measure is applied to the net portfolio returns;
    if `false`, to their negation. [`set_range_risk_constraints!`](@ref) sets it to `false`
    to build the gain tail of [`OrderedWeightsArrayRange`](@ref).
  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `aowa_risk`: The OWA risk expression added to the model.

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`set_risk_constraints!`](@ref)

# References

  - $(ref_dict[:owa3]) Sections 2.3 and 2.4, Equations 4 and 5; Section 3, Equation 7; Section 4, Tables 1 and 3.
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ApproxOrderedWeightsArray},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               loss::Bool = true, prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    T = get_T(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    owa_w = -owa_weights(r.w, T, :w)
    owa_s = sum(owa_w)
    owa_l, owa_h = extrema(owa_w)
    # With at most one entry strictly inside the box, the box and the sum describe the convex
    # hull of the reorderings. The p-norm cones are then redundant, and they put the optimum
    # at the apex of every cone, where the interior point solver can stall (#1895).
    owa_p = count(x -> owa_l < x < owa_h, owa_w) <= 1 ? empty(r.alg.p) : r.alg.p
    M = length(owa_p)
    owa_t = state_set!(model, prefix, :owa_t_, i, JuMP.@variable(model))
    owa_nu = state_set!(model, prefix, :owa_nu_, i,
                        JuMP.@variable(model, [1:T], lower_bound = 0))
    owa_eta = state_set!(model, prefix, :owa_eta_, i,
                         JuMP.@variable(model, [1:T], lower_bound = 0))
    owa_epsilon = state_set!(model, prefix, :owa_epsilon_, i,
                             JuMP.@variable(model, [1:T, 1:M]))
    owa_psi = state_set!(model, prefix, :owa_psi_, i, JuMP.@variable(model, [1:T, 1:M]))
    owa_z = state_set!(model, prefix, :owa_z_, i, JuMP.@variable(model, [1:M]))
    owa_y = state_set!(model, prefix, :owa_y_, i,
                       JuMP.@variable(model, [1:M], lower_bound = 0))
    owa_d = [LinearAlgebra.norm(owa_w, p) for p in owa_p]
    aowa_risk = state_set!(model, prefix, :aowa_risk_, i,
                           JuMP.@expression(model,
                                            owa_s * owa_t - owa_l * sum(owa_nu) +
                                            owa_h * sum(owa_eta) +
                                            LinearAlgebra.dot(owa_d, owa_y)))
    neg_owa_z_owa_p = state_set!(model, prefix, :neg_owa_z_owa_p_, i,
                                 JuMP.@expression(model, -owa_z .* owa_p))
    owa_p_o_owa_pm1 = state_set!(model, prefix, :owa_p_o_owa_pm1_, i,
                                 JuMP.@expression(model,
                                                  owa_p ./ (owa_p .- one(eltype(owa_p)))))
    # A sum over one axis of a JuMP array is one `sum(view(…))` for each row or column.
    # `sum(owa_psi; dims = 1)` reduces with a plain `+` and copies each partial sum, which
    # cost 144 MiB on 1000 × 3 (#2062).
    eps_sum = [sum(view(owa_epsilon, t, :)) for t in 1:T]
    psi_sum = [sum(view(owa_psi, :, j)) for j in 1:M]
    # The gain tail negates `net_X` inside the macro, which builds the row on one copy of the
    # series. A negated copy made before the macro costs a second copy (#2062).
    ca1_owa = if loss
        JuMP.@constraint(model, sc * ((net_X - owa_nu + owa_eta - eps_sum) .+ owa_t) == 0)
    else
        JuMP.@constraint(model, sc * ((-net_X - owa_nu + owa_eta - eps_sum) .+ owa_t) == 0)
    end
    state_set!(model, prefix, :ca1_owa_, i, ca1_owa)
    state_set!(model, prefix, :ca2_owa_, i,
               JuMP.@constraint(model, sc * (owa_z + owa_y - psi_sum) == 0))
    state_set!(model, prefix, :ca_owa_pcone_, i,
               JuMP.@constraint(model, [i = 1:M, j = 1:T],
                                [sc * neg_owa_z_owa_p[i],
                                 sc * owa_psi[j, i] * owa_p_o_owa_pm1[i],
                                 sc * owa_epsilon[j, i]] in
                                JuMP.MOI.PowerCone(inv(owa_p[i]))))
    set_risk_bounds_and_expression!(model, opt, aowa_risk, r.settings, :aowa_risk_, i;
                                    prefix = prefix)
    return aowa_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `OrderedWeightsArrayRange` using the approximate OWA
formulation to `model`.

Delegates to [`set_range_risk_constraints!`](@ref), which builds the loss tail from `w1` on
the net portfolio returns and the gain tail from `w2` on their negation, then sums the two
OWA expressions. Each tail brings its own power cone block.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ApproxOrderedWeightsArray}`: The
    OWA range risk measure with approximate formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Validation

  - Each tail weight vector has one entry for each observation, through [`owa_weights`](@ref). Raises a `DimensionMismatch` that names `w1` or `w2`.

# Returns

  - `aowa_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:ApproxOrderedWeightsArray},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    # Each tail checks its own weights, but under the name `w` of the tail. Check them here
    # under the names that the caller gave.
    T = get_T(model)
    owa_weights(r.w1, T, :w1)
    owa_weights(r.w2, T, :w2)
    return set_range_risk_constraints!(model, i, r, :aowa_range_risk_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the relaxed sorting network of a [`SortingNetworkOrderedWeightsArray`](@ref) to `model`, and return the OWA risk expression of the weight vector `owa_w`.

The network reads the `owa` variable of [`set_owa_constraints!`](@ref), so the fees and the costs enter as they do for [`ExactOrderedWeightsArray`](@ref). The inputs are divided by the largest absolute asset return and the risk is multiplied by it, which keeps the network values near one. The OWA risk is positively homogeneous, so the scale does not change it.

# JuMP formulation

  - `z`: Vector variable of length `2K`, the two values that each of the `K` comparators writes, in the columns that `alg.ordering` gives.
  - `c_eq`: The `K` rows `sc * (z_l + z_h - z_a - z_b) == 0`, where comparator `k` reads `z_a` and `z_b`, and writes the smaller value `z_l` and the larger value `z_h`.
  - `c_a`, `c_b`: The `K` rows `sc * (z_h - z_a) >= 0` and the `K` rows `sc * (z_h - z_b) >= 0`.
  - `risk`: The expression `s * dot(owa_w, z_out)`, where `z_out` holds the last value that the network writes to each position, in ascending position, and `s` is the largest absolute asset return.

# Arguments

  - $(arg_dict[:model])
  - `alg`: The sorting-network formulation.
  - $(arg_dict[:pr_X])
  - `owa_w`: Nondecreasing OWA weight vector of length `T`.

# Validation

  - `issorted(owa_w)`, through [`assert_owa_formulation_weights`](@ref). Raises a `DomainError`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `(; z, c_eq, c_a, c_b, risk)`: The network variable, the three blocks of rows and the risk expression, which the caller registers.

# Related

  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`sorting_network`](@ref)
  - [`sorting_network_columns`](@ref)
  - [`set_owa_constraints!`](@ref)

# References

  - $(ref_dict[:goemans2015])
"""
function set_sorting_network_owa!(model::JuMP.Model, alg::SortingNetworkOrderedWeightsArray,
                                  X::MatNum, owa_w::VecNum; prefix::Symbol = Symbol(""))
    assert_owa_formulation_weights(alg, owa_w)
    sc = get_constraint_scale(model)
    T = get_T(model)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    c = sorting_network(alg.merge, T)
    K = length(c)
    cols = sorting_network_columns(alg.ordering, c)
    z = JuMP.@variable(model, [1:(2 * K)])
    s = maximum(abs, X)
    s = iszero(s) ? one(s) : s
    # The inputs, then the values of the network in the order of the comparators.
    x = [owa / s; z[cols]]
    # The value that each position holds. Comparator `k` reads `a[k]` and `b[k]`.
    wire = collect(1:T)
    a = Vector{Int}(undef, K)
    b = Vector{Int}(undef, K)
    for (k, (i, j)) in enumerate(c)
        a[k] = wire[i]
        b[k] = wire[j]
        wire[i] = T + 2 * k - 1
        wire[j] = T + 2 * k
    end
    l = (T + 1):2:(T + 2 * K)
    h = (T + 2):2:(T + 2 * K)
    c_eq = JuMP.@constraint(model, sc * (x[l] + x[h] - x[a] - x[b]) == 0)
    c_a = JuMP.@constraint(model, sc * (x[h] - x[a]) in JuMP.Nonnegatives())
    c_b = JuMP.@constraint(model, sc * (x[h] - x[b]) in JuMP.Nonnegatives())
    risk = JuMP.@expression(model, s * LinearAlgebra.dot(owa_w, x[wire]))
    return (; z, c_eq, c_a, c_b, risk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Ordered Weights Array (OWA) risk constraints to `model` using the sorting-network formulation.

The formulation is exact for a nondecreasing weight vector, as [`ExactOrderedWeightsArray`](@ref) is, at ``O(T \\log^{2} T)`` linear rows in place of ``T^{2}``. [`set_sorting_network_owa!`](@ref) builds the network.

# Mathematical definition

```math
\\begin{align}
\\mathrm{OWA}(\\boldsymbol{w}) &= \\begin{cases}
\\underset{\\boldsymbol{z}}{\\min} & s\\, \\boldsymbol{\\omega}^{\\intercal} \\boldsymbol{z}_{\\mathrm{out}}\\\\
\\text{s.t.} & z_{l(k)} + z_{h(k)} = z_{a(k)} + z_{b(k)} \\quad \\forall k = 1,\\, \\ldots,\\, K\\\\
 & z_{h(k)} \\geq z_{a(k)},\\; z_{h(k)} \\geq z_{b(k)} \\quad \\forall k = 1,\\, \\ldots,\\, K\\\\
 & \\boldsymbol{z}_{\\mathrm{in}} = \\hat{\\boldsymbol{r}} / s\\,.
\\end{cases}
\\end{align}
```

Where:

  - ``\\boldsymbol{\\omega}``: Is the OWA weight vector, `r.w` if it is a vector, `r.w(T)` if it is a callable.
  - ``\\hat{\\boldsymbol{r}}``: Is the vector of net portfolio returns.
  - ``s``: Is the largest absolute asset return.
  - ``K``: Is the number of comparators of `r.alg.merge` for `T` values.
  - ``a(k),\\, b(k)``: Are the two values that comparator ``k`` reads.
  - ``l(k),\\, h(k)``: Are the smaller and the larger value that comparator ``k`` writes.
  - ``\\boldsymbol{z}_{\\mathrm{in}},\\, \\boldsymbol{z}_{\\mathrm{out}}``: Are the values that the network reads and the last value that it writes to each position, in ascending position.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArray{<:Any, <:Any, <:SortingNetworkOrderedWeightsArray}`: The OWA risk
    measure with the sorting-network formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Validation

  - The weight vector has one entry for each observation, through [`owa_weights`](@ref). Raises a `DimensionMismatch`.
  - The weight vector is nondecreasing. Raises a `DomainError`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `sowa_risk`: The OWA risk expression added to the model.

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`set_sorting_network_owa!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)

# References

  - $(ref_dict[:goemans2015])
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:SortingNetworkOrderedWeightsArray},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    T = get_T(model)
    owa_w = owa_weights(r.w, T, :w)
    net = set_sorting_network_owa!(model, r.alg, pr.X, owa_w; prefix = prefix)
    state_set!(model, prefix, :sowa_z_, i, net.z)
    state_set!(model, prefix, :csowa_eq_, i, net.c_eq)
    state_set!(model, prefix, :csowa_a_, i, net.c_a)
    state_set!(model, prefix, :csowa_b_, i, net.c_b)
    sowa_risk = state_set!(model, prefix, :sowa_risk_, i, net.risk)
    set_risk_bounds_and_expression!(model, opt, sowa_risk, r.settings, :sowa_risk_, i;
                                    prefix = prefix)
    return sowa_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `OrderedWeightsArrayRange` using the sorting-network formulation to `model`.

The formulation fuses the two tails, as [`ExactOrderedWeightsArray`](@ref) does. It forms the single weight vector ``\\boldsymbol{\\omega}_{1} - \\boldsymbol{\\omega}_{2}`` and builds one network over it. [`OrderedWeightsArrayRange`](@ref) holds `r.w2` reversed, so the difference is the range weight vector, and it is nondecreasing when both tails are.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\omega} &= \\boldsymbol{\\omega}_{1} - \\boldsymbol{\\omega}_{2}\\\\
\\mathrm{OWA}_{\\mathrm{rg}}(\\boldsymbol{w}) &= \\boldsymbol{\\omega}^{\\intercal} \\mathrm{sort}\\left(\\hat{\\boldsymbol{r}}\\right)\\,.
\\end{align}
```

[`set_sorting_network_owa!`](@ref) states the rows that compute the sorted term.

Where:

  - ``\\mathrm{OWA}_{\\mathrm{rg}}(\\boldsymbol{w})``: Is the ordered weighted average range risk of the portfolio.
  - ``\\boldsymbol{\\omega}_{1}``: Is the loss-tail OWA weight vector, `r.w1` if it is a vector, `r.w1(T)` if it is a callable.
  - ``\\boldsymbol{\\omega}_{2}``: Is the reversed gain-tail OWA weight vector, `r.w2` if it is a vector, `r.w2(T)` if it is a callable.
  - ``\\hat{\\boldsymbol{r}}``: Is the vector of net portfolio returns, which ``\\mathrm{sort}`` places in ascending order.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:SortingNetworkOrderedWeightsArray}`: The
    OWA range risk measure with the sorting-network formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Validation

  - Each tail weight vector has one entry for each observation, through [`owa_weights`](@ref). Raises a `DimensionMismatch`.
  - The range weight vector is nondecreasing. Raises a `DomainError`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `sowa_range_risk`: The OWA range risk expression added to the model.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`SortingNetworkOrderedWeightsArray`](@ref)
  - [`set_sorting_network_owa!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)

# References

  - $(ref_dict[:goemans2015])
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:SortingNetworkOrderedWeightsArray},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    T = get_T(model)
    owa_w1 = owa_weights(r.w1, T, :w1)
    owa_w2 = owa_weights(r.w2, T, :w2)
    net = set_sorting_network_owa!(model, r.alg, pr.X, owa_w1 - owa_w2; prefix = prefix)
    state_set!(model, prefix, :sowa_range_z_, i, net.z)
    state_set!(model, prefix, :csowa_range_eq_, i, net.c_eq)
    state_set!(model, prefix, :csowa_range_a_, i, net.c_a)
    state_set!(model, prefix, :csowa_range_b_, i, net.c_b)
    sowa_range_risk = state_set!(model, prefix, :sowa_range_risk_, i, net.risk)
    set_risk_bounds_and_expression!(model, opt, sowa_range_risk, r.settings,
                                    :sowa_range_risk_, i; prefix = prefix)
    return sowa_range_risk
end
