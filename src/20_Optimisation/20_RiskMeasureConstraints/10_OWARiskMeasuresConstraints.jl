"""
$(DocStringExtensions.TYPEDSIGNATURES)

Set up the OWA portfolio returns variable and equality constraint.

Introduces a vector variable `owa` of length `T` and adds the equality constraint
`sc * (net_X - owa) == 0`. Returns the existing `owa` if already present.

# Arguments

  - $(arg_dict[:model])
  - `X::MatNum`: Asset returns matrix (`T × N`).

# Returns

  - `owa`: JuMP vector variable of length `T` for OWA portfolio returns.

# Related

  - [`set_risk_constraints!`](@ref)
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

Add Ordered Weights Array (OWA) risk constraints to `model`.

The exact overloads introduce auxiliary matrices and use a bilinear constraint to encode the
exact OWA risk. The approximate overloads use the Wasserstein-based approximation via power
cone constraints parameterised by `r.alg.p`. Range variants compute the difference between
two OWA expressions (e.g. tail-Gini range).

# Mathematical definition

```math
\\begin{align}
\\mathrm{OWA}(\\boldsymbol{w}) &= \\boldsymbol{\\omega}^\\intercal \\mathrm{sort}(\\hat{\\boldsymbol{r}})\\,, \\\\
\\hat{r}_t &= \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{OWA}(\\boldsymbol{w})``: Ordered Weighted Average risk measure.
  - ``\\boldsymbol{\\omega}``: OWA weight vector.
  - ``\\hat{\\boldsymbol{r}}``: Vector of portfolio returns at each time step.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}``: Portfolio return at time ``t``.

where ``\\boldsymbol{\\omega}`` is the OWA weight vector and ``\\mathrm{sort}(\\hat{\\boldsymbol{r}})`` sorts the portfolio returns in ascending order.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: OWA or OWA-range risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

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

Introduces auxiliary matrix variables and a bilinear constraint to encode the exact OWA
range risk as the difference between two OWA tail expressions (e.g. tail-Gini range).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ExactOrderedWeightsArray}`: The
    OWA range risk measure with exact formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

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

Uses the Wasserstein-based power cone approximation parameterised by `r.alg.p` to encode
the OWA risk as a weighted sum of p-norm terms.

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
