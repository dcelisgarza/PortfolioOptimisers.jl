function set_owa_constraints!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :owa)
        return model[:owa]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    T = size(X, 1)
    @variable(model, owa[1:T])
    @constraint(model, owac, sc * (net_X - owa) == 0)
    return owa
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ExactOrderedWeightsArray},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:owa_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    owa = set_owa_constraints!(model, pr.X)
    ovec = range(1; stop = 1, length = T)
    owa_a, owa_b = model[Symbol(:owa_a_, i)], model[Symbol(:owa_b_, i)] = @variables(model,
                                                                                     begin
                                                                                         [1:T]
                                                                                         [1:T]
                                                                                     end)
    owa_risk = model[key] = @expression(model, sum(owa_a + owa_b))
    owa_w = isnothing(r.w) ? owa_gmd(T) : r.w
    model[Symbol(:cowa_, i)] = @constraint(model,
                                           sc * (owa * transpose(owa_w) -
                                                 ovec * transpose(owa_a) -
                                                 owa_b * transpose(ovec)) in Nonpositives())
    set_risk_bounds_and_expression!(model, opt, owa_risk, r.settings, key)
    return owa_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:ExactOrderedWeightsArray},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:owa_range_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    owa = set_owa_constraints!(model, pr.X)
    ovec = range(1; stop = 1, length = T)
    owa_a, owa_b = model[Symbol(:owa_range_a_, i)], model[Symbol(:owa_range_b_, i)] = @variables(model,
                                                                                                 begin
                                                                                                     [1:T]
                                                                                                     [1:T]
                                                                                                 end)
    owa_range_risk = model[key] = @expression(model, sum(owa_a + owa_b))
    owa_w1 = isnothing(r.w1) ? owa_tg(T) : r.w1
    owa_w2 = isnothing(r.w2) ? reverse(owa_w1) : r.w2
    owa_w = owa_w1 - owa_w2
    model[Symbol(:cowa_range_, i)] = @constraint(model,
                                                 sc * (owa * transpose(owa_w) -
                                                       ovec * transpose(owa_a) -
                                                       owa_b * transpose(ovec)) in
                                                 Nonpositives())
    set_risk_bounds_and_expression!(model, opt, owa_range_risk, r.settings, key)
    return owa_range_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArray{<:Any, <:Any,
                                                      <:ApproxOrderedWeightsArray},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:aowa_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    net_X = set_net_portfolio_returns!(model, pr.X)
    owa_p = r.alg.p
    M = length(owa_p)
    owa_t, owa_nu, owa_eta, owa_epsilon, owa_psi, owa_z, owa_y = model[Symbol(:owa_t_, i)], model[Symbol(:owa_nu_, i)], model[Symbol(:owa_eta_, i)], model[Symbol(:owa_epsilon_, i)], model[Symbol(:owa_psi_, i)], model[Symbol(:owa_z_, i)], model[Symbol(:owa_y_, i)] = @variables(model,
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
    owa_w = isnothing(r.w) ? -owa_gmd(T) : -r.w
    owa_s = sum(owa_w)
    owa_l = minimum(owa_w)
    owa_h = maximum(owa_w)
    owa_d = [norm(owa_w, p) for p in owa_p]
    aowa_risk, neg_owa_z_owa_p, owa_p_o_owa_pm1 = model[key], model[Symbol(:neg_owa_z_owa_p_, i)], model[Symbol(:owa_p_o_owa_pm1_, i)] = @expressions(model,
                                                                                                                                                      begin
                                                                                                                                                          owa_s *
                                                                                                                                                          owa_t -
                                                                                                                                                          owa_l *
                                                                                                                                                          sum(owa_nu) +
                                                                                                                                                          owa_h *
                                                                                                                                                          sum(owa_eta) +
                                                                                                                                                          dot(owa_d,
                                                                                                                                                              owa_y)
                                                                                                                                                          -owa_z .*
                                                                                                                                                          owa_p
                                                                                                                                                          owa_p ./
                                                                                                                                                          (owa_p .-
                                                                                                                                                           one(eltype(owa_p)))
                                                                                                                                                      end)
    model[Symbol(:ca1_owa_, i)], model[Symbol(:ca2_owa_, i)], model[Symbol(:ca_owa_pcone_, i)] = @constraints(model,
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
                                                                                                                  MOI.PowerCone(inv(owa_p[i]))
                                                                                                              end)
    set_risk_bounds_and_expression!(model, opt, aowa_risk, r.settings, key)
    return aowa_risk
end
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any,
                                                           <:ApproxOrderedWeightsArray},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:aowa_range_risk_, i)
    sc = model[:sc]
    T = size(pr.X, 1)
    net_X = set_net_portfolio_returns!(model, pr.X)
    owa_p = r.alg.p
    M = length(owa_p)
    owa_l_t, owa_l_nu, owa_l_eta, owa_l_epsilon, owa_l_psi, owa_l_z, owa_l_y, owa_h_t, owa_h_nu, owa_h_eta, owa_h_epsilon, owa_h_psi, owa_h_z, owa_h_y = model[Symbol(:owa_l_t_, i)], model[Symbol(:owa_l_nu_, i)], model[Symbol(:owa_l_eta_, i)], model[Symbol(:owa_l_epsilon_, i)], model[Symbol(:owa_l_psi_, i)], model[Symbol(:owa_l_z_, i)], model[Symbol(:owa_l_y_, i)], model[Symbol(:owa_h_t_, i)], model[Symbol(:owa_h_nu_, i)], model[Symbol(:owa_h_eta_, i)], model[Symbol(:owa_h_epsilon_, i)], model[Symbol(:owa_h_psi_, i)], model[Symbol(:owa_h_z_, i)], model[Symbol(:owa_h_y_, i)] = @variables(model,
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
    owa_l_w = isnothing(r.w1) ? -owa_tg(T) : -r.w1
    owa_l_s = sum(owa_l_w)
    owa_l_l = minimum(owa_l_w)
    owa_l_h = maximum(owa_l_w)
    owa_l_d = [norm(owa_l_w, p) for p in owa_p]
    owa_h_w = isnothing(r.w2) ? reverse(owa_l_w) : -r.w2
    owa_h_s = sum(owa_h_w)
    owa_h_l = minimum(owa_h_w)
    owa_h_h = maximum(owa_h_w)
    owa_h_d = [norm(owa_h_w, p) for p in owa_p]
    owa_l_risk, neg_owa_l_z_owa_p, owa_h_risk, neg_owa_h_z_owa_p, owa_p_o_owa_pm1 = model[Symbol(:owa_l_risk_, i)], model[Symbol(:neg_owa_l_z_owa_p_, i)], model[Symbol(:owa_h_risk_, i)], model[Symbol(:neg_owa_h_z_owa_p_, i)], model[Symbol(:owa_p_o_owa_pm1_, i)] = @expressions(model,
                                                                                                                                                                                                                                                                                     begin
                                                                                                                                                                                                                                                                                         owa_l_s *
                                                                                                                                                                                                                                                                                         owa_l_t -
                                                                                                                                                                                                                                                                                         owa_l_l *
                                                                                                                                                                                                                                                                                         sum(owa_l_nu) +
                                                                                                                                                                                                                                                                                         owa_l_h *
                                                                                                                                                                                                                                                                                         sum(owa_l_eta) +
                                                                                                                                                                                                                                                                                         dot(owa_l_d,
                                                                                                                                                                                                                                                                                             owa_l_y)
                                                                                                                                                                                                                                                                                         -owa_l_z .*
                                                                                                                                                                                                                                                                                         owa_p
                                                                                                                                                                                                                                                                                         owa_h_s *
                                                                                                                                                                                                                                                                                         owa_h_t -
                                                                                                                                                                                                                                                                                         owa_h_l *
                                                                                                                                                                                                                                                                                         sum(owa_h_nu) +
                                                                                                                                                                                                                                                                                         owa_h_h *
                                                                                                                                                                                                                                                                                         sum(owa_h_eta) +
                                                                                                                                                                                                                                                                                         dot(owa_h_d,
                                                                                                                                                                                                                                                                                             owa_h_y)
                                                                                                                                                                                                                                                                                         -owa_h_z .*
                                                                                                                                                                                                                                                                                         owa_p
                                                                                                                                                                                                                                                                                         owa_p ./
                                                                                                                                                                                                                                                                                         (owa_p .-
                                                                                                                                                                                                                                                                                          one(eltype(owa_p)))
                                                                                                                                                                                                                                                                                     end)
    model[Symbol(:ca1_owa_l_, i)], model[Symbol(:ca2_owa_l_, i)], model[Symbol(:ca_owa_pcone_l_, i)], model[Symbol(:ca1_owa_h_, i)], model[Symbol(:ca2_owa_h_, i)], model[Symbol(:ca_owa_pcone_h_, i)] = @constraints(model,
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
                                                                                                                                                                                                                          MOI.PowerCone(inv(owa_p[i]))
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
                                                                                                                                                                                                                          MOI.PowerCone(inv(owa_p[i]))
                                                                                                                                                                                                                      end)
    aowa_range_risk = model[key] = @expression(model, owa_l_risk + owa_h_risk)
    set_risk_bounds_and_expression!(model, opt, aowa_range_risk, r.settings, key)
    return aowa_range_risk
end
