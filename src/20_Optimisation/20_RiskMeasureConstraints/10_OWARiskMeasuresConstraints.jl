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
    if haskey(model, Symbol(prefix, :owa))
        return model[Symbol(prefix, :owa)]
    end
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = size(X, 1)
    owa = preg!(model, prefix, :owa, JuMP.@variable(model, [1:T]))
    preg!(model, prefix, :owac, JuMP.@constraint(model, sc * (net_X - owa) == 0))
    return owa
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
    key = Symbol(:owa_risk_, i)
    sc = get_constraint_scale(model)
    X = pr.X
    T = size(X, 1)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    ovec = range(one(eltype(X)), one(eltype(X)); length = T)
    owa_a, owa_b = model[Symbol(:owa_a_, i)], model[Symbol(:owa_b_, i)] = JuMP.@variables(model,
                                                                                          begin
                                                                                              [1:T]
                                                                                              [1:T]
                                                                                          end)
    owa_risk = model[key] = JuMP.@expression(model, sum(owa_a + owa_b))
    owa_w = isa(r.w, VecNum) ? r.w : r.w(T)
    model[Symbol(:cowa_, i)] = JuMP.@constraint(model,
                                                sc * (owa * transpose(owa_w) -
                                                      ovec * transpose(owa_a) -
                                                      owa_b * transpose(ovec)) in
                                                JuMP.Nonpositives())
    set_risk_bounds_and_expression!(model, opt, owa_risk, r.settings, key)
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
    key = Symbol(:owa_range_risk_, i)
    sc = get_constraint_scale(model)
    X = pr.X
    T = size(X, 1)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    ovec = range(one(eltype(X)), one(eltype(X)); length = T)
    owa_a, owa_b = model[Symbol(:owa_range_a_, i)], model[Symbol(:owa_range_b_, i)] = JuMP.@variables(model,
                                                                                                      begin
                                                                                                          [1:T]
                                                                                                          [1:T]
                                                                                                      end)
    owa_range_risk = model[key] = JuMP.@expression(model, sum(owa_a + owa_b))
    owa_w1 = isa(r.w1, VecNum) ? r.w1 : r.w1(T)
    owa_w2 = isa(r.w2, VecNum) ? r.w2 : r.w2(T)
    owa_w = owa_w1 - owa_w2
    model[Symbol(:cowa_range_, i)] = JuMP.@constraint(model,
                                                      sc * (owa * transpose(owa_w) -
                                                            ovec * transpose(owa_a) -
                                                            owa_b * transpose(ovec)) in
                                                      JuMP.Nonpositives())
    set_risk_bounds_and_expression!(model, opt, owa_range_risk, r.settings, key)
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

# Returns

  - `nothing`.

# Related

  - [`OrderedWeightsArray`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ApproxOrderedWeightsArray},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:aowa_risk_, i)
    sc = get_constraint_scale(model)
    X = pr.X
    T = size(X, 1)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    owa_p = r.alg.p
    M = length(owa_p)
    owa_t, owa_nu, owa_eta, owa_epsilon, owa_psi, owa_z, owa_y = model[Symbol(:owa_t_, i)], model[Symbol(:owa_nu_, i)], model[Symbol(:owa_eta_, i)], model[Symbol(:owa_epsilon_, i)], model[Symbol(:owa_psi_, i)], model[Symbol(:owa_z_, i)], model[Symbol(:owa_y_, i)] = JuMP.@variables(model,
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
    owa_w = isa(r.w, VecNum) ? -r.w : -r.w(T)
    owa_s = sum(owa_w)
    owa_l = minimum(owa_w)
    owa_h = maximum(owa_w)
    owa_d = [LinearAlgebra.norm(owa_w, p) for p in owa_p]
    aowa_risk, neg_owa_z_owa_p, owa_p_o_owa_pm1 = model[key], model[Symbol(:neg_owa_z_owa_p_, i)], model[Symbol(:owa_p_o_owa_pm1_, i)] = JuMP.@expressions(model,
                                                                                                                                                           begin
                                                                                                                                                               owa_s *
                                                                                                                                                               owa_t -
                                                                                                                                                               owa_l *
                                                                                                                                                               sum(owa_nu) +
                                                                                                                                                               owa_h *
                                                                                                                                                               sum(owa_eta) +
                                                                                                                                                               LinearAlgebra.dot(owa_d,
                                                                                                                                                                                 owa_y)
                                                                                                                                                               -owa_z .*
                                                                                                                                                               owa_p
                                                                                                                                                               owa_p ./
                                                                                                                                                               (owa_p .-
                                                                                                                                                                one(eltype(owa_p)))
                                                                                                                                                           end)
    model[Symbol(:ca1_owa_, i)], model[Symbol(:ca2_owa_, i)], model[Symbol(:ca_owa_pcone_, i)] = JuMP.@constraints(model,
                                                                                                                   begin
                                                                                                                       sc *
                                                                                                                       ((net_X -
                                                                                                                         owa_nu +
                                                                                                                         owa_eta -
                                                                                                                         vec(sum(owa_epsilon;
                                                                                                                                 dims = 2))) .+
                                                                                                                        owa_t) ==
                                                                                                                       0
                                                                                                                       sc *
                                                                                                                       (owa_z +
                                                                                                                        owa_y -
                                                                                                                        vec(sum(owa_psi;
                                                                                                                                dims = 1))) ==
                                                                                                                       0
                                                                                                                       [i = 1:M,
                                                                                                                        j = 1:T],
                                                                                                                       [sc *
                                                                                                                        neg_owa_z_owa_p[i],
                                                                                                                        sc *
                                                                                                                        owa_psi[j,
                                                                                                                                i] *
                                                                                                                        owa_p_o_owa_pm1[i],
                                                                                                                        sc *
                                                                                                                        owa_epsilon[j,
                                                                                                                                    i]] in
                                                                                                                       JuMP.MOI.PowerCone(inv(owa_p[i]))
                                                                                                                   end)
    set_risk_bounds_and_expression!(model, opt, aowa_risk, r.settings, key)
    return aowa_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `OrderedWeightsArrayRange` using the approximate OWA
formulation to `model`.

Uses the Wasserstein-based power cone approximation parameterised by `r.alg.p` to encode
both OWA tail expressions, then computes their difference as the range risk.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ApproxOrderedWeightsArray}`: The
    OWA range risk measure with approximate formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`OrderedWeightsArrayRange`](@ref)
  - [`ApproxOrderedWeightsArray`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:ApproxOrderedWeightsArray},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:aowa_range_risk_, i)
    sc = get_constraint_scale(model)
    X = pr.X
    T = size(X, 1)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    owa_p = r.alg.p
    M = length(owa_p)
    owa_l_t, owa_l_nu, owa_l_eta, owa_l_epsilon, owa_l_psi, owa_l_z, owa_l_y, owa_h_t, owa_h_nu, owa_h_eta, owa_h_epsilon, owa_h_psi, owa_h_z, owa_h_y = model[Symbol(:owa_l_t_, i)], model[Symbol(:owa_l_nu_, i)], model[Symbol(:owa_l_eta_, i)], model[Symbol(:owa_l_epsilon_, i)], model[Symbol(:owa_l_psi_, i)], model[Symbol(:owa_l_z_, i)], model[Symbol(:owa_l_y_, i)], model[Symbol(:owa_h_t_, i)], model[Symbol(:owa_h_nu_, i)], model[Symbol(:owa_h_eta_, i)], model[Symbol(:owa_h_epsilon_, i)], model[Symbol(:owa_h_psi_, i)], model[Symbol(:owa_h_z_, i)], model[Symbol(:owa_h_y_, i)] = JuMP.@variables(model,
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
    owa_l_w = isa(r.w1, VecNum) ? -r.w1 : -r.w1(T)
    owa_l_s = sum(owa_l_w)
    owa_l_l = minimum(owa_l_w)
    owa_l_h = maximum(owa_l_w)
    owa_l_d = [LinearAlgebra.norm(owa_l_w, p) for p in owa_p]
    owa_h_w = isa(r.w2, VecNum) ? -r.w2 : -r.w2(T)
    owa_h_s = sum(owa_h_w)
    owa_h_l = minimum(owa_h_w)
    owa_h_h = maximum(owa_h_w)
    owa_h_d = [LinearAlgebra.norm(owa_h_w, p) for p in owa_p]
    owa_l_risk, neg_owa_l_z_owa_p, owa_h_risk, neg_owa_h_z_owa_p, owa_p_o_owa_pm1 = model[Symbol(:owa_l_risk_, i)], model[Symbol(:neg_owa_l_z_owa_p_, i)], model[Symbol(:owa_h_risk_, i)], model[Symbol(:neg_owa_h_z_owa_p_, i)], model[Symbol(:owa_p_o_owa_pm1_, i)] = JuMP.@expressions(model,
                                                                                                                                                                                                                                                                                          begin
                                                                                                                                                                                                                                                                                              owa_l_s *
                                                                                                                                                                                                                                                                                              owa_l_t -
                                                                                                                                                                                                                                                                                              owa_l_l *
                                                                                                                                                                                                                                                                                              sum(owa_l_nu) +
                                                                                                                                                                                                                                                                                              owa_l_h *
                                                                                                                                                                                                                                                                                              sum(owa_l_eta) +
                                                                                                                                                                                                                                                                                              LinearAlgebra.dot(owa_l_d,
                                                                                                                                                                                                                                                                                                                owa_l_y)
                                                                                                                                                                                                                                                                                              -owa_l_z .*
                                                                                                                                                                                                                                                                                              owa_p
                                                                                                                                                                                                                                                                                              owa_h_s *
                                                                                                                                                                                                                                                                                              owa_h_t -
                                                                                                                                                                                                                                                                                              owa_h_l *
                                                                                                                                                                                                                                                                                              sum(owa_h_nu) +
                                                                                                                                                                                                                                                                                              owa_h_h *
                                                                                                                                                                                                                                                                                              sum(owa_h_eta) +
                                                                                                                                                                                                                                                                                              LinearAlgebra.dot(owa_h_d,
                                                                                                                                                                                                                                                                                                                owa_h_y)
                                                                                                                                                                                                                                                                                              -owa_h_z .*
                                                                                                                                                                                                                                                                                              owa_p
                                                                                                                                                                                                                                                                                              owa_p ./
                                                                                                                                                                                                                                                                                              (owa_p .-
                                                                                                                                                                                                                                                                                               one(eltype(owa_p)))
                                                                                                                                                                                                                                                                                          end)
    model[Symbol(:ca1_owa_l_, i)], model[Symbol(:ca2_owa_l_, i)], model[Symbol(:ca_owa_pcone_l_, i)], model[Symbol(:ca1_owa_h_, i)], model[Symbol(:ca2_owa_h_, i)], model[Symbol(:ca_owa_pcone_h_, i)] = JuMP.@constraints(model,
                                                                                                                                                                                                                           begin
                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                               ((net_X -
                                                                                                                                                                                                                                 owa_l_nu +
                                                                                                                                                                                                                                 owa_l_eta -
                                                                                                                                                                                                                                 vec(sum(owa_l_epsilon;
                                                                                                                                                                                                                                         dims = 2))) .+
                                                                                                                                                                                                                                owa_l_t) ==
                                                                                                                                                                                                                               0
                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                               (owa_l_z +
                                                                                                                                                                                                                                owa_l_y -
                                                                                                                                                                                                                                vec(sum(owa_l_psi;
                                                                                                                                                                                                                                        dims = 1))) ==
                                                                                                                                                                                                                               0
                                                                                                                                                                                                                               [i = 1:M,
                                                                                                                                                                                                                                j = 1:T],
                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                neg_owa_l_z_owa_p[i],
                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                owa_l_psi[j,
                                                                                                                                                                                                                                          i] *
                                                                                                                                                                                                                                owa_p_o_owa_pm1[i],
                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                owa_l_epsilon[j,
                                                                                                                                                                                                                                              i]] in
                                                                                                                                                                                                                               JuMP.MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                               ((-net_X -
                                                                                                                                                                                                                                 owa_h_nu +
                                                                                                                                                                                                                                 owa_h_eta -
                                                                                                                                                                                                                                 vec(sum(owa_h_epsilon;
                                                                                                                                                                                                                                         dims = 2))) .+
                                                                                                                                                                                                                                owa_h_t) ==
                                                                                                                                                                                                                               0
                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                               (owa_h_z +
                                                                                                                                                                                                                                owa_h_y -
                                                                                                                                                                                                                                vec(sum(owa_h_psi;
                                                                                                                                                                                                                                        dims = 1))) ==
                                                                                                                                                                                                                               0
                                                                                                                                                                                                                               [i = 1:M,
                                                                                                                                                                                                                                j = 1:T],
                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                neg_owa_h_z_owa_p[i],
                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                owa_h_psi[j,
                                                                                                                                                                                                                                          i] *
                                                                                                                                                                                                                                owa_p_o_owa_pm1[i],
                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                owa_h_epsilon[j,
                                                                                                                                                                                                                                              i]] in
                                                                                                                                                                                                                               JuMP.MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                                                                           end)
    aowa_range_risk = model[key] = JuMP.@expression(model, owa_l_risk + owa_h_risk)
    set_risk_bounds_and_expression!(model, opt, aowa_range_risk, r.settings, key)
    return aowa_range_risk
end
