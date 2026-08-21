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
        T = size(X, 1)
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
``T^{2}`` constraints, so [`ApproxOrderedWeightsArray`](@ref) is the cheaper formulation for
a long sample.

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
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    T = size(X, 1)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    ovec = range(one(eltype(X)), one(eltype(X)); length = T)
    owa_a, owa_b = JuMP.@variables(model, begin
                                       [1:T]
                                       [1:T]
                                   end)
    state_set!(model, prefix, :owa_a_, i, owa_a)
    state_set!(model, prefix, :owa_b_, i, owa_b)
    owa_risk = state_set!(model, prefix, :owa_risk_, i,
                          JuMP.@expression(model, sum(owa_a + owa_b)))
    owa_w = isa(r.w, VecNum) ? r.w : r.w(T)
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
over it, so the range costs the same `T × T` block as a single tail. `r.w2` is already
reversed by the constructor of [`OrderedWeightsArrayRange`](@ref), so the difference is the
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
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    T = size(X, 1)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    ovec = range(one(eltype(X)), one(eltype(X)); length = T)
    owa_a, owa_b = JuMP.@variables(model, begin
                                       [1:T]
                                       [1:T]
                                   end)
    state_set!(model, prefix, :owa_range_a_, i, owa_a)
    state_set!(model, prefix, :owa_range_b_, i, owa_b)
    owa_range_risk = state_set!(model, prefix, :owa_range_risk_, i,
                                JuMP.@expression(model, sum(owa_a + owa_b)))
    owa_w1 = isa(r.w1, VecNum) ? r.w1 : r.w1(T)
    owa_w2 = isa(r.w2, VecNum) ? r.w2 : r.w2(T)
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
closes as the weight vector approaches a line, which is the case the source paper studies: it
reports the same objective value as the exact formulation, to its printed precision, for the
Gini mean difference and the tail Gini over samples of 500 to 10,000 observations.

The gap was measured here against the functor at `T = 100`, `N = 8` with the default `p`. The
Gini mean difference is 0.06 % high, the tail Gini is 1.7e-5 % high, and the tail Gini range
is 0.37 % high. A fourth-order L-moment weight vector, which is not linear, is 4.5 % high, so
prefer [`ExactOrderedWeightsArray`](@ref) for a weight vector that is far from a line.

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
  - ``S``: Is the set of p-norm orders, `r.alg.p`.
  - ``t``: Is the scalar auxiliary variable, `owa_t`.
  - ``\\boldsymbol{\\nu},\\, \\boldsymbol{\\eta}``: Are the `T × 1` non-negative auxiliary variables, `owa_nu` and `owa_eta`.
  - ``\\boldsymbol{\\varepsilon},\\, \\boldsymbol{\\psi}``: Are the `T × M` auxiliary variables, `owa_epsilon` and `owa_psi`.
  - ``\\boldsymbol{\\zeta},\\, \\boldsymbol{y}``: Are the `M × 1` auxiliary variables, `owa_z` and `owa_y`.
  - ``\\mathcal{P}_{3}^{\\alpha,\\, 1-\\alpha}``: Is the three-dimensional power cone.
  - ``T``: Is the total number of observations.
  - ``M``: Is the number of p-norm orders, `length(r.alg.p)`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArray{<:Any, <:Any, <:ApproxOrderedWeightsArray}`: The OWA risk
    measure with approximate formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `loss::Bool`: If `true` (default), the measure is applied to the net portfolio returns;
    if `false`, to their negation. This is the seam [`set_range_risk_constraints!`](@ref)
    builds the gain tail of [`OrderedWeightsArrayRange`](@ref) through.
  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `aowa_risk`: The OWA risk expression added to the model.

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`ExactOrderedWeightsArray`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ApproxOrderedWeightsArray},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    T = size(X, 1)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    owa_p = r.alg.p
    M = length(owa_p)
    owa_t, owa_nu, owa_eta, owa_epsilon, owa_psi, owa_z, owa_y = JuMP.@variables(model,
                                                                                 begin
                                                                                     ()
                                                                                     [1:T],
                                                                                     (lower_bound = 0)
                                                                                     [1:T],
                                                                                     (lower_bound = 0)
                                                                                     [1:T,
                                                                                      1:M]
                                                                                     [1:T,
                                                                                      1:M]
                                                                                     [1:M]
                                                                                     [1:M],
                                                                                     (lower_bound = 0)
                                                                                 end)
    state_set!(model, prefix, :owa_t_, i, owa_t)
    state_set!(model, prefix, :owa_nu_, i, owa_nu)
    state_set!(model, prefix, :owa_eta_, i, owa_eta)
    state_set!(model, prefix, :owa_epsilon_, i, owa_epsilon)
    state_set!(model, prefix, :owa_psi_, i, owa_psi)
    state_set!(model, prefix, :owa_z_, i, owa_z)
    state_set!(model, prefix, :owa_y_, i, owa_y)
    owa_w = isa(r.w, VecNum) ? -r.w : -r.w(T)
    owa_s = sum(owa_w)
    owa_l = minimum(owa_w)
    owa_h = maximum(owa_w)
    owa_d = [LinearAlgebra.norm(owa_w, p) for p in owa_p]
    aowa_risk, neg_owa_z_owa_p, owa_p_o_owa_pm1 = JuMP.@expressions(model,
                                                                    begin
                                                                        owa_s * owa_t -
                                                                        owa_l *
                                                                        sum(owa_nu) +
                                                                        owa_h *
                                                                        sum(owa_eta) +
                                                                        LinearAlgebra.dot(owa_d,
                                                                                          owa_y)
                                                                        -owa_z .* owa_p
                                                                        owa_p ./ (owa_p .-
                                                                                  one(eltype(owa_p)))
                                                                    end)
    state_set!(model, prefix, :aowa_risk_, i, aowa_risk)
    state_set!(model, prefix, :neg_owa_z_owa_p_, i, neg_owa_z_owa_p)
    state_set!(model, prefix, :owa_p_o_owa_pm1_, i, owa_p_o_owa_pm1)
    ca1_owa, ca2_owa, ca_owa_pcone = JuMP.@constraints(model,
                                                       begin
                                                           sc *
                                                           ((net_X - owa_nu + owa_eta -
                                                             vec(sum(owa_epsilon; dims = 2))) .+
                                                            owa_t) == 0
                                                           sc * (owa_z + owa_y -
                                                                 vec(sum(owa_psi; dims = 1))) ==
                                                           0
                                                           [i = 1:M, j = 1:T],
                                                           [sc * neg_owa_z_owa_p[i],
                                                            sc *
                                                            owa_psi[j, i] *
                                                            owa_p_o_owa_pm1[i],
                                                            sc * owa_epsilon[j, i]] in
                                                           JuMP.MOI.PowerCone(inv(owa_p[i]))
                                                       end)
    state_set!(model, prefix, :ca1_owa_, i, ca1_owa)
    state_set!(model, prefix, :ca2_owa_, i, ca2_owa)
    state_set!(model, prefix, :ca_owa_pcone_, i, ca_owa_pcone)
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
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :aowa_range_risk_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
