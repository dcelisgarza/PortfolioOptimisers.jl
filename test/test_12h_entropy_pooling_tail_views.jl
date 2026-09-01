include(joinpath(@__DIR__, "test12_setup.jl"))
using Pajarito, HiGHS, JuMP

# The tail views of `EntropyPoolingPrior` are the CVaR and EVaR views of Cajas (2026).
# Every one of them is a constraint of the single entropy pooling problem, so a view is
# checked by reading its risk measure off the posterior weights, not by trusting a residual
# the solver reports.

const PO = PortfolioOptimisers
const ep_a = 0.05
ep_jopt = JuMPEntropyPooling(; slv = slv)
ep_w0 = StatsBase.pweights(fill(inv(T), T))
ep_cvar(j, wi) = ConditionalValueatRisk(; alpha = ep_a, w = wi)(rd.X[:, j])
ep_evar_of(j, wi) = PO.ep_evar(-rd.X[:, j], wi, ep_a).evar
const ep_pcvar = ep_cvar(1, ep_w0)
const ep_pevar = ep_evar_of(1, ep_w0)
ep_pr0 = prior(EmpiricalPrior(), rd)
# A view on a group is the *sum* of the risk measures of its members, so it names as many
# assets as the group holds. Two members keep the count of binary variables affordable;
# `gS` holds one, which is what makes a group view take the single-asset path.
ep_gsets = UniverseSets(;
                        dict = Dict("nx" => rd.nx, "gA" => rd.nx[1:2], "gB" => rd.nx[3:4],
                                    "gS" => rd.nx[5:5]))
ep_gcvar(wi, js) = sum(ep_cvar(j, wi) for j in js)
# The same two measures at an arbitrary significance level, for the views that carry their
# own.
ep_cvar_at(j, wi, a) = ConditionalValueatRisk(; alpha = a, w = wi)(rd.X[:, j])
ep_evar_at(j, wi, a) = PO.ep_evar(-rd.X[:, j], wi, a).evar
const ep_k = 0.3
ep_rlvar_of(j, wi) = PO.ep_rlvar(-rd.X[:, j], wi, ep_a, ep_k).rlvar
ep_rlvar_at(j, wi, a, k) = PO.ep_rlvar(-rd.X[:, j], wi, a, k).rlvar
const ep_prlvar = ep_rlvar_of(1, ep_w0)
# Resolve one view equation the way `ep_tail_views!` does, without solving anything.
function ep_terms(eqn, key, sets = ep_gsets)
    res = PO.parse_equation(eqn; ops1 = ("==", ">=", "<="),
                            ops2 = (:call, :(==), :(>=), :(<=)), datatype = eltype(rd.X))
    res = PO.replace_group_by_assets(res, sets, false, true, false)
    res = PO.replace_prior_views(res, ep_pr0, sets, key, ep_a)
    return PO.ep_view_terms(res, sets, rd.X)
end

@testset "ep_evar matches the EVaR risk measure" begin
    for j in (1, 5, 20)
        @test isapprox(PO.ep_evar(-rd.X[:, j], ep_w0, ep_a).evar,
                       EntropicValueatRisk(; slv = slv, alpha = ep_a)(rd.X[:, j]),
                       rtol = 1e-5)
    end
    # The dual variable it returns is the one that attains the value.
    res = PO.ep_evar(-rd.X[:, 1], ep_w0, ep_a)
    f = z -> z * (log(sum(ep_w0 .* exp.(-rd.X[:, 1] ./ z))) - log(ep_a))
    @test f(res.z) <= f(res.z * 1.05)
    @test f(res.z) <= f(res.z * 0.95)
    @test isapprox(f(res.z), res.evar, rtol = 1e-8)
    # Non-uniform prior weights.
    wv = StatsBase.pweights(LinearAlgebra.normalize(range(0.5, 1.5; length = T), 1))
    @test isapprox(PO.ep_evar(-rd.X[:, 1], wv, ep_a).evar,
                   EntropicValueatRisk(; slv = slv, alpha = ep_a, w = wv)(rd.X[:, 1]),
                   rtol = 1e-5)
end

@testset "CVaR views, linear formulation" begin
    tgt = ep_pcvar * 1.3
    # A lower bound binds at the target: the entropy minimiser never overshoots it.
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL >= $tgt"))),
               rd)
    @test isapprox(ep_cvar(1, pr.w), tgt, rtol = 1e-4)
    @test isapprox(sum(pr.w), 1)
    @test all(>=(-eps()), pr.w)
    @test pr.kld > 0

    # An equality above the prior is the same problem.
    pr2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    cvar_views = ConditionalValueatRiskView(;
                                                                            views = LinearConstraintEstimator(;
                                                                                                              val = "AAPL == $tgt"))),
                rd)
    @test isapprox(pr.w, pr2.w, rtol = 1e-6)

    # A `prior(...)` reference resolves to the prior CVaR of the asset.
    pr3 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    cvar_views = ConditionalValueatRiskView(;
                                                                            views = LinearConstraintEstimator(;
                                                                                                              val = "AAPL == 1.37*prior(AAPL)"))),
                rd)
    @test isapprox(ep_cvar(1, pr3.w), 1.37 * ep_pcvar, rtol = 1e-4)

    # A negative coefficient flips the operator rather than the answer.
    pr4 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    cvar_views = ConditionalValueatRiskView(;
                                                                            views = LinearConstraintEstimator(;
                                                                                                              val = "-AAPL <= $(-tgt)"))),
                rd)
    @test isapprox(pr.w, pr4.w, rtol = 1e-6)
end

@testset "EVaR views, conic formulation" begin
    tgt = ep_pevar * 1.2
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   evar_views = EntropicValueatRiskView(;
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "AAPL >= $tgt"))),
               rd)
    @test isapprox(ep_evar_of(1, pr.w), tgt, rtol = 1e-4)

    pr2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    evar_views = EntropicValueatRiskView(;
                                                                         views = LinearConstraintEstimator(;
                                                                                                           val = "AAPL == $tgt"))),
                rd)
    @test isapprox(pr.w, pr2.w, rtol = 1e-6)

    pr3 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    evar_views = EntropicValueatRiskView(;
                                                                         views = LinearConstraintEstimator(;
                                                                                                           val = "AAPL == 1.2*prior(AAPL)"))),
                rd)
    @test isapprox(ep_evar_of(1, pr3.w), tgt, rtol = 1e-4)
end

@testset "ep_rlvar matches the RLVaR risk measure" begin
    # `ep_rlvar` minimises the primal objective of the sample RLVaR with the two power
    # cones of each observation already minimised out in closed form, so it has to agree
    # with the conic risk measure a solver answers.
    for j in (1, 5, 20)
        @test isapprox(PO.ep_rlvar(-rd.X[:, j], ep_w0, ep_a, ep_k).rlvar,
                       RelativisticValueatRisk(; slv = slv, alpha = ep_a, kappa = ep_k)(rd.X[:,
                                                                                             j]),
                       rtol = 1e-5)
    end
    # Several deformation parameters and significance levels.
    for a in (0.01, 0.05, 0.1), k in (0.1, 0.3, 0.6)
        @test isapprox(PO.ep_rlvar(-rd.X[:, 1], ep_w0, a, k).rlvar,
                       RelativisticValueatRisk(; slv = slv, alpha = a, kappa = k)(rd.X[:,
                                                                                       1]),
                       rtol = 1e-5)
    end
    # The primal pair it returns is the one that attains the value.
    res = PO.ep_rlvar(-rd.X[:, 1], ep_w0, ep_a, ep_k)
    lnk = PO.kappa_log(inv(ep_a * T), ep_k)
    g = (t, z) -> t +
                  z * lnk +
                  T * sum(ep_w0[j] * PO.ep_rlvar_tail(t + rd.X[j, 1], z, ep_k) for j in 1:T)
    @test isapprox(g(res.t, res.z), res.rlvar, rtol = 1e-8)
    @test g(res.t, res.z) <= g(res.t * 1.01, res.z)
    @test g(res.t, res.z) <= g(res.t * 0.99, res.z)
    @test g(res.t, res.z) <= g(res.t, res.z * 1.05)
    @test g(res.t, res.z) <= g(res.t, res.z * 0.95)
    # Non-uniform prior weights. The Kaniadakis logarithm has no multiplication-to-addition
    # property, so the weights cannot be folded into the normalisation the way they are for
    # EVaR: they are normalised to sum to one and enter as `T * w`.
    wv = StatsBase.pweights(LinearAlgebra.normalize(range(0.5, 1.5; length = T), 1))
    @test isapprox(PO.ep_rlvar(-rd.X[:, 1], wv, ep_a, ep_k).rlvar,
                   RelativisticValueatRisk(; slv = slv, alpha = ep_a, kappa = ep_k, w = wv)(rd.X[:,
                                                                                                 1]),
                   rtol = 1e-5)
    # Uniform weights passed explicitly reproduce the unweighted answer.
    @test isapprox(PO.ep_rlvar(-rd.X[:, 1], ep_w0, ep_a, ep_k).rlvar,
                   RelativisticValueatRisk(; slv = slv, alpha = ep_a, kappa = ep_k,
                                           w = ep_w0)(rd.X[:, 1]), rtol = 1e-5)
end

@testset "RLVaR views, conic formulation" begin
    # RLVaR sits close to the worst realisation of the sample, so the headroom a lower
    # bound has is small. A modest multiple keeps the view reachable.
    tgt = ep_prlvar * 1.05
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   rlvar_views = RelativisticValueatRiskView(;
                                                                             views = LinearConstraintEstimator(;
                                                                                                               val = "AAPL >= $tgt"))),
               rd)
    # The power cones this formulation writes are harder than the exponential cones of
    # the EVaR view, so the solver lands a little above the target rather than on it. The
    # direction is the safe one: the view holds.
    @test ep_rlvar_of(1, pr.w) >= tgt * (1 - 1e-8)
    @test isapprox(ep_rlvar_of(1, pr.w), tgt, rtol = 2e-3)
    @test isapprox(sum(pr.w), 1)
    @test all(>=(-eps()), pr.w)
    @test pr.kld > 0

    # An equality above the prior is the same problem.
    pr2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL == $tgt"))),
                rd)
    @test isapprox(pr.w, pr2.w, rtol = 1e-6)

    # A `prior(...)` reference resolves to the prior RLVaR of the asset, read at this
    # group's `alpha` *and* `kappa`.
    pr3 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL == 1.05*prior(AAPL)"))),
                rd)
    @test isapprox(ep_rlvar_of(1, pr3.w), tgt, rtol = 2e-3)

    # A negative coefficient flips the operator rather than the answer.
    pr4 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "-AAPL <= $(-tgt)"))),
                rd)
    @test isapprox(pr.w, pr4.w, rtol = 1e-6)

    # A group of one member resolves to that member, so it takes the single-asset path.
    pr5 = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "gS >= 1.04*prior(gS)"))),
                rd)
    @test isapprox(ep_rlvar_of(5, pr5.w), 1.04 * ep_rlvar_of(5, ep_w0), rtol = 2e-3)
end

@testset "RLVaR view guards and constructors" begin
    v = LinearConstraintEstimator(; val = "AAPL >= $(ep_prlvar * 1.05)")
    @test RelativisticValueatRiskView(; views = v).alpha == 0.05
    @test RelativisticValueatRiskView(; views = v).kappa == 0.3
    @test RelativisticValueatRiskView(; views = v).alg === nothing
    @test RelativisticValueatRiskView(; views = v, alpha = 0.01, kappa = 0.7).kappa == 0.7
    @test RelativisticValueatRiskView(; views = v,
                                      alg = GridRelativisticValueatRiskView(; pct = 0.8,
                                                                            K = 21)).alg.K ==
          21
    @test_throws DomainError RelativisticValueatRiskView(; views = v, alpha = 1.5)
    @test_throws DomainError RelativisticValueatRiskView(; views = v, kappa = 0.0)
    @test_throws DomainError RelativisticValueatRiskView(; views = v, kappa = 1.0)
    @test_throws DomainError GridRelativisticValueatRiskView(; K = 10)
    @test_throws DomainError GridRelativisticValueatRiskView(; pct = 1.0)
    @test_throws DomainError GridRelativisticValueatRiskView(; M = 0)
    @test_throws PortfolioOptimisers.IsEmptyError RelativisticValueatRiskView(; views = v,
                                                                              alg = ConicRelativisticValueatRiskView[])
    @test_throws UndefKeywordError RelativisticValueatRiskView(; alpha = 0.05)
    @test_throws PortfolioOptimisers.IsEmptyError EntropyPoolingPrior(; sets = sets,
                                                                      opt = ep_jopt,
                                                                      rlvar_views = RelativisticValueatRiskView[])
    @test_throws PortfolioOptimisers.IsNothingError EntropyPoolingPrior(; opt = ep_jopt,
                                                                        rlvar_views = RelativisticValueatRiskView(;
                                                                                                                  views = v))

    # `nothing` takes the exact conic formulation where it applies, and the grid elsewhere.
    @test PO.ep_rlvar_formulation(nothing, :geq, 0.1, 0.05) ===
          ConicRelativisticValueatRiskView()
    @test isa(PO.ep_rlvar_formulation(nothing, :leq, 0.1, 0.05),
              GridRelativisticValueatRiskView)
    @test isa(PO.ep_rlvar_formulation(nothing, :eq, 0.01, 0.05),
              GridRelativisticValueatRiskView)
    @test PO.ep_rlvar_formulation(ConicRelativisticValueatRiskView(), :leq, 0.0, 0.0) ===
          ConicRelativisticValueatRiskView()

    # The conic formulation bounds the RLVaR from below only.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         rlvar_views = RelativisticValueatRiskView(;
                                                                                                   alg = ConicRelativisticValueatRiskView(),
                                                                                                   views = LinearConstraintEstimator(;
                                                                                                                                     val = "AAPL <= $(ep_prlvar * 0.9)"))),
                                     rd)
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         rlvar_views = RelativisticValueatRiskView(;
                                                                                                   alg = ConicRelativisticValueatRiskView(),
                                                                                                   views = LinearConstraintEstimator(;
                                                                                                                                     val = "AAPL == $(ep_prlvar * 0.9)"))),
                                     rd)
    # There is no formulation for a relative RLVaR view.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         rlvar_views = RelativisticValueatRiskView(;
                                                                                                   views = LinearConstraintEstimator(;
                                                                                                                                     val = "AAPL - XOM >= 0.01"))),
                                     rd)
    # A view no reweighting of the sample can reach.
    @test_throws DomainError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                       rlvar_views = RelativisticValueatRiskView(;
                                                                                                 views = LinearConstraintEstimator(;
                                                                                                                                   val = "AAPL >= $(maximum(-rd.X[:, 1]) * 1.01)"))),
                                   rd)
    # A tail view needs auxiliary variables, which the dual formulation has no room for.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets,
                                                         opt = OptimEntropyPooling(),
                                                         rlvar_views = RelativisticValueatRiskView(;
                                                                                                   views = v)),
                                     rd)
end

@testset "an RLVaR view naming an asset the universe does not hold" begin
    using Logging
    # `ep_view_terms` places every name a view holds. Under `strict = false` a name the
    # universe does not hold is reported and dropped, so the view produces no constraint
    # and the loop moves on. Nothing else in this file reaches that branch of
    # `ep_tail_views!`.
    epc = Dict{Symbol, Any}()
    tvs = PO.AbstractEntropyPoolingTailView[]
    v = RelativisticValueatRiskView(;
                                    views = LinearConstraintEstimator(;
                                                                      val = "NOTANASSET >= 0.1"),
                                    alpha = ep_a, kappa = ep_k)
    Logging.with_logger(Logging.NullLogger()) do
        return PO.ep_tail_views!(v, epc, tvs, ep_pr0, ep_gsets, ep_w0; strict = false)
    end
    @test isempty(epc)
    @test isempty(tvs)
    # The same name under `strict = true` raises instead.
    @test_throws ArgumentError PO.ep_tail_views!(v, epc, tvs, ep_pr0, ep_gsets, ep_w0;
                                                 strict = true)
end

@testset "prior references in tail views" begin
    # A `prior(...)` reference is replaced by the prior value of the risk measure the view
    # family names, so a CVaR view reads the prior CVaR and an EVaR view the prior EVaR.
    # The view is then stated as a multiple of, or an offset from, what the prior says.
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL >= 1.25*prior(AAPL)"))),
               rd)
    @test isapprox(ep_cvar(1, pr.w), 1.25 * ep_pcvar, rtol = 1e-4)

    # An additive offset. The reference is part of the constant of the view, so it lands on
    # the target rather than on the left hand side.
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL == prior(AAPL) + 0.01"))),
               rd)
    @test isapprox(ep_cvar(1, pr.w), ep_pcvar + 0.01, rtol = 1e-4)

    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   evar_views = EntropicValueatRiskView(;
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "AAPL >= 1.15*prior(AAPL)"))),
               rd)
    @test isapprox(ep_evar_of(1, pr.w), 1.15 * ep_pevar, rtol = 1e-4)

    # A group of one member resolves to that member, so a group view takes the single-asset
    # path and both continuous formulations apply to it.
    pr = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "gS >= 1.2*prior(gS)"))),
               rd)
    @test isapprox(ep_cvar(5, pr.w), 1.2 * ep_cvar(5, ep_w0), rtol = 1e-4)

    pr = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                   evar_views = EntropicValueatRiskView(;
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "gS >= 1.15*prior(gS)"))),
               rd)
    @test isapprox(ep_evar_of(5, pr.w), 1.15 * ep_evar_of(5, ep_w0), rtol = 1e-4)
end

@testset "group and multi-asset views resolve to sums of asset risk measures" begin
    # This is the whole of the group semantics, and it is settled before any solve: the
    # group expansion hands each member the coefficient the group carried, so a group view
    # reads as the sum of the members' risk measures, never their average.
    t = ep_terms("gA >= 0.09", :cvar)
    @test t.idx == [1, 2]
    @test t.coef == [1.0, 1.0]
    @test t.op == :geq
    @test isapprox(t.rhs, 0.09)

    # `prior(<group>)` expands the same way, so it is the sum of the members' prior values.
    t = ep_terms("gA == prior(gA)", :cvar)
    @test t.op == :eq
    @test isapprox(t.rhs, ep_gcvar(ep_w0, 1:2))

    t = ep_terms("gA == 1.2*prior(gA)", :cvar)
    @test isapprox(t.rhs, 1.2 * ep_gcvar(ep_w0, 1:2))

    # A relative view on two groups.
    t = ep_terms("gA - gB >= 0.01", :cvar)
    @test t.idx == [1, 2, 3, 4]
    @test t.coef == [1.0, 1.0, -1.0, -1.0]

    # A relative view naming several assets directly.
    t = ep_terms("$(rd.nx[1]) + $(rd.nx[2]) - $(rd.nx[3]) >= 0.01", :cvar)
    @test t.idx == [1, 2, 3]
    @test t.coef == [1.0, 1.0, -1.0]

    # A group and a bare asset in the same view.
    t = ep_terms("gA - $(rd.nx[3]) >= 0.02", :cvar)
    @test t.idx == [1, 2, 3]
    @test t.coef == [1.0, 1.0, -1.0]

    # A coefficient on the group reaches every member.
    t = ep_terms("2*gA >= 0.3", :cvar)
    @test t.coef == [2.0, 2.0]

    # A one-member group leaves one asset, and its prior reference reads the EVaR.
    t = ep_terms("gS >= 1.1*prior(gS)", :evar)
    @test t.idx == [5]
    @test isapprox(t.rhs, 1.1 * ep_evar_of(5, ep_w0))

    # Every view above names more than one asset, and that alone sends the choice to the
    # integer formulation, lower bound or not.
    @test isa(PO.ep_cvar_formulation(nothing, false, :geq, 0.1, 0.05),
              IntegerConditionalValueatRiskView)
end

@testset "views naming several assets are refused by the single-asset formulations" begin
    gv = LinearConstraintEstimator(; val = "gA >= 0.2")
    # A group of two is a relative view. Cajas (2026) gives no formulation for a
    # relative EVaR view, so every EVaR route refuses it.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                                         evar_views = EntropicValueatRiskView(;
                                                                                              views = gv)),
                                     rd)
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                                         evar_views = EntropicValueatRiskView(;
                                                                                              alg = ConicEntropicValueatRiskView(),
                                                                                              views = gv)),
                                     rd)
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                                         evar_views = EntropicValueatRiskView(;
                                                                                              alg = GridEntropicValueatRiskView(),
                                                                                              views = gv)),
                                     rd)
    # The linear CVaR formulation writes the CVaR of one asset.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                                         cvar_views = ConditionalValueatRiskView(;
                                                                                                 alg = LinearConditionalValueatRiskView(),
                                                                                                 views = gv)),
                                     rd)
end

@testset "tail views at several significance levels" begin
    # The significance level is part of the statistic, not a setting of the solve: the CVaR
    # at 1% and at 10% are different numbers on the same series. A `ConditionalValueatRiskView` carries its
    # own level, so one estimator can hold groups stated at several of them.
    p1 = ep_cvar_at(1, ep_w0, 0.01)
    p10 = ep_cvar_at(20, ep_w0, 0.10)
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = [ConditionalValueatRiskView(; alpha = 0.01,
                                                                            views = LinearConstraintEstimator(;
                                                                                                              val = "AAPL >= $(p1 * 1.3)")),
                                                 ConditionalValueatRiskView(; alpha = 0.10,
                                                                            views = LinearConstraintEstimator(;
                                                                                                              val = "XOM >= $(p10 * 1.2)"))]),
               rd)
    # Both groups bind, each read at its own level, out of the one solve.
    @test isapprox(ep_cvar_at(1, pr.w, 0.01), p1 * 1.3, rtol = 1e-4)
    @test isapprox(ep_cvar_at(20, pr.w, 0.10), p10 * 1.2, rtol = 1e-4)

    # A `prior(...)` reference resolves at *its own group's* level. The two levels are far
    # apart here, so a reference read at the wrong one could not pass this.
    @test !isapprox(p1, ep_cvar_at(1, ep_w0, 0.10), rtol = 0.5)
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(; alpha = 0.01,
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL >= 1.3*prior(AAPL)"))),
               rd)
    @test isapprox(ep_cvar_at(1, pr.w, 0.01), p1 * 1.3, rtol = 1e-4)

    # A group that states no level takes the 0.05 default. There is no estimator-level
    # setting to fall back to any more: the level lives on the view and nowhere else.
    p5 = ep_cvar_at(1, ep_w0, 0.05)
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL >= $(p5 * 1.3)"))),
               rd)
    @test isapprox(ep_cvar_at(1, pr.w, 0.05), p5 * 1.3, rtol = 1e-4)
end

@testset "EVaR view groups carry their own level and grid" begin
    # `alg` on the group is where the grid of dual variables and the big-M live, so one
    # group can take a different grid from the rest.
    e1 = ep_evar_at(1, ep_w0, 0.01)
    e10 = ep_evar_at(20, ep_w0, 0.10)
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   evar_views = [EntropicValueatRiskView(; alpha = 0.01,
                                                                         views = LinearConstraintEstimator(;
                                                                                                           val = "AAPL >= $(e1 * 1.2)")),
                                                 EntropicValueatRiskView(; alpha = 0.10,
                                                                         alg = GridEntropicValueatRiskView(;
                                                                                                           pct = 0.8,
                                                                                                           K = 21),
                                                                         views = LinearConstraintEstimator(;
                                                                                                           val = "XOM >= $(e10 * 1.15)"))]),
               rd)
    # The first group takes the exact conic formulation.
    @test isapprox(ep_evar_at(1, pr.w, 0.01), e1 * 1.2, rtol = 1e-4)
    # The second takes the grid its own `alg` names, which holds at the grid points and may
    # fall a little short between them.
    @test isapprox(ep_evar_at(20, pr.w, 0.10), e10 * 1.15, rtol = 1e-3)
end

@testset "tail view group constructors" begin
    v = LinearConstraintEstimator(; val = "AAPL >= 0.07")
    # `alpha` defaults to 0.05; `alg` defaults to `nothing`, which lets each view take the
    # cheapest formulation that expresses it exactly.
    @test ConditionalValueatRiskView(; views = v).alpha == 0.05
    @test ConditionalValueatRiskView(; views = v).alg === nothing
    @test EntropicValueatRiskView(; views = v).alpha == 0.05
    @test EntropicValueatRiskView(; views = v).alg === nothing
    @test ValueatRiskView(; views = v).alpha == 0.05
    @test ConditionalValueatRiskView(; views = v, alpha = 0.01).alpha == 0.01
    @test ConditionalValueatRiskView(; views = v,
                                     alg = IntegerConditionalValueatRiskView(; sbar = 40)).alg.sbar ==
          40
    @test EntropicValueatRiskView(; views = v,
                                  alg = GridEntropicValueatRiskView(; pct = 0.8, K = 21)).alg.K ==
          21

    @test_throws DomainError ConditionalValueatRiskView(; views = v, alpha = 1.5)
    @test_throws DomainError ConditionalValueatRiskView(; views = v, alpha = 0.0)
    @test_throws DomainError EntropicValueatRiskView(; views = v, alpha = 1.0)
    @test_throws PortfolioOptimisers.IsEmptyError ConditionalValueatRiskView(; views = v,
                                                                             alg = LinearConditionalValueatRiskView[])
    @test_throws PortfolioOptimisers.IsEmptyError EntropicValueatRiskView(; views = v,
                                                                          alg = ConicEntropicValueatRiskView[])
    # A group with no views has nothing to state.
    @test_throws UndefKeywordError ConditionalValueatRiskView(; alpha = 0.05)
    @test_throws UndefKeywordError EntropicValueatRiskView(; alpha = 0.05)

    # An empty group vector on the estimator.
    @test_throws PortfolioOptimisers.IsEmptyError EntropyPoolingPrior(; sets = sets,
                                                                      opt = ep_jopt,
                                                                      cvar_views = ConditionalValueatRiskView[])
    @test_throws PortfolioOptimisers.IsEmptyError EntropyPoolingPrior(; sets = sets,
                                                                      opt = ep_jopt,
                                                                      evar_views = EntropicValueatRiskView[])
    # A CVaR group needs auxiliary variables just as a bare estimator does.
    @test_throws ArgumentError EntropyPoolingPrior(; sets = sets,
                                                   cvar_views = ConditionalValueatRiskView(;
                                                                                           alpha = 0.05,
                                                                                           views = v))
    # A group still needs `sets` to resolve its names against.
    @test_throws PortfolioOptimisers.IsNothingError EntropyPoolingPrior(; opt = ep_jopt,
                                                                        cvar_views = ConditionalValueatRiskView(;
                                                                                                                views = v))
end

@testset "EVaR lower bound, grid formulation, reaches the Optim dual" begin
    # The lower-bound half of the grid formulation is linear in the posterior
    # probabilities, so it is the one tail view the dual formulation can carry.
    tgt = ep_pevar * 1.2
    v = LinearConstraintEstimator(; val = "AAPL >= $tgt")
    w = nothing
    for opt in (OptimEntropyPooling(), ep_jopt)
        pr = prior(EntropyPoolingPrior(; sets = sets, opt = opt,
                                       evar_views = EntropicValueatRiskView(;
                                                                            alg = GridEntropicValueatRiskView(),
                                                                            views = v)), rd)
        # The grid enforces the bound at its points alone, so the answer is approximate.
        @test isapprox(ep_evar_of(1, pr.w), tgt, rtol = 5e-3)
        w = isnothing(w) ? pr.w : (@test isapprox(w, pr.w, rtol = 1e-4); w)
    end
    # A denser grid closes the gap.
    coarse = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                       evar_views = EntropicValueatRiskView(;
                                                                            alg = GridEntropicValueatRiskView(;
                                                                                                              pct = 0.9,
                                                                                                              K = 3),
                                                                            views = v)), rd)
    fine = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                     evar_views = EntropicValueatRiskView(;
                                                                          alg = GridEntropicValueatRiskView(;
                                                                                                            pct = 0.9,
                                                                                                            K = 41),
                                                                          views = v)), rd)
    @test abs(ep_evar_of(1, fine.w) - tgt) <= abs(ep_evar_of(1, coarse.w) - tgt)
end

@testset "tail views alongside moment views" begin
    tgt = ep_pcvar * 1.2
    pe = EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                             mu_views = LinearConstraintEstimator(; val = "AAPL == 0.002"),
                             sigma_views = LinearConstraintEstimator(;
                                                                     val = "XOM == 1.2*prior(XOM)"),
                             cvar_views = ConditionalValueatRiskView(;
                                                                     views = LinearConstraintEstimator(;
                                                                                                       val = "AAPL >= $tgt")))
    for alg in (H0_EntropyPooling(), H1_EntropyPooling(), H2_EntropyPooling())
        pr = prior(EntropyPoolingPrior(; pe.pe, pe.sets, pe.opt, pe.mu_views, pe.cvar_views,
                                       pe.sigma_views, alg = alg), rd)
        # The tail view is accumulated into the constraint set, so it still holds after
        # the variance stage re-solves.
        @test ep_cvar(1, pr.w) >= tgt - 1e-6
        @test isapprox(pr.mu[1], 0.002, rtol = 1e-4)
    end
end

@testset "the two divergence formulations agree" begin
    tgt = ep_pcvar * 1.25
    v = LinearConstraintEstimator(; val = "AAPL >= $tgt")
    p1 = prior(EntropyPoolingPrior(; sets = sets,
                                   opt = JuMPEntropyPooling(; slv = slv,
                                                            alg = ExpEntropyPooling()),
                                   cvar_views = ConditionalValueatRiskView(; views = v)),
               rd)
    p2 = prior(EntropyPoolingPrior(; sets = sets,
                                   opt = JuMPEntropyPooling(; slv = slv,
                                                            alg = LogEntropyPooling()),
                                   cvar_views = ConditionalValueatRiskView(; views = v)),
               rd)
    @test isapprox(p1.w, p2.w, rtol = 1e-3, atol = 1e-8)
end

@testset "formulation choice and its guards" begin
    # `nothing` takes the exact dual formulation where it applies, and the general one
    # everywhere else.
    @test PO.ep_cvar_formulation(nothing, true, :geq, 0.1, 0.05) ===
          LinearConditionalValueatRiskView()
    @test PO.ep_cvar_formulation(nothing, true, :eq, 0.1, 0.05) ===
          LinearConditionalValueatRiskView()
    @test isa(PO.ep_cvar_formulation(nothing, true, :eq, 0.01, 0.05),
              IntegerConditionalValueatRiskView)
    @test isa(PO.ep_cvar_formulation(nothing, true, :leq, 0.1, 0.05),
              IntegerConditionalValueatRiskView)
    @test isa(PO.ep_cvar_formulation(nothing, false, :geq, 0.1, 0.05),
              IntegerConditionalValueatRiskView)
    @test PO.ep_cvar_formulation(LinearConditionalValueatRiskView(), false, :leq, 0.0,
                                 0.0) === LinearConditionalValueatRiskView()
    @test PO.ep_evar_formulation(nothing, :geq, 0.1, 0.05) ===
          ConicEntropicValueatRiskView()
    @test isa(PO.ep_evar_formulation(nothing, :leq, 0.1, 0.05), GridEntropicValueatRiskView)
    @test isa(PO.ep_evar_formulation(nothing, :eq, 0.01, 0.05), GridEntropicValueatRiskView)

    # A vector of formulations is spread one per view.
    @test length(PO.ep_view_formulations(nothing, 3, :alg)) == 3
    @test PO.ep_view_formulations([LinearConditionalValueatRiskView()], 1, :alg) ==
          [LinearConditionalValueatRiskView()]
    @test_throws DimensionMismatch PO.ep_view_formulations([LinearConditionalValueatRiskView()],
                                                           2, :alg)

    # A CVaR view always needs auxiliary variables, so it is refused at construction
    # against the dual formulation.
    @test_throws ArgumentError EntropyPoolingPrior(; sets = sets,
                                                   cvar_views = ConditionalValueatRiskView(;
                                                                                           views = LinearConstraintEstimator(;
                                                                                                                             val = "AAPL == 0.07")))
    # An upper bound is outside the linear formulation.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         cvar_views = ConditionalValueatRiskView(;
                                                                                                 alg = LinearConditionalValueatRiskView(),
                                                                                                 views = LinearConstraintEstimator(;
                                                                                                                                   val = "AAPL <= $(ep_pcvar * 0.9)"))),
                                     rd)
    # So is an equality below the prior value.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         cvar_views = ConditionalValueatRiskView(;
                                                                                                 alg = LinearConditionalValueatRiskView(),
                                                                                                 views = LinearConstraintEstimator(;
                                                                                                                                   val = "AAPL == $(ep_pcvar * 0.9)"))),
                                     rd)
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         evar_views = EntropicValueatRiskView(;
                                                                                              alg = ConicEntropicValueatRiskView(),
                                                                                              views = LinearConstraintEstimator(;
                                                                                                                                val = "AAPL <= $(ep_pevar * 0.9)"))),
                                     rd)
    # A relative view is CVaR only, and integer only.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         evar_views = EntropicValueatRiskView(;
                                                                                              views = LinearConstraintEstimator(;
                                                                                                                                val = "AAPL - XOM >= 0.01"))),
                                     rd)
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                         cvar_views = ConditionalValueatRiskView(;
                                                                                                 alg = LinearConditionalValueatRiskView(),
                                                                                                 views = LinearConstraintEstimator(;
                                                                                                                                   val = "AAPL - XOM >= 0.01"))),
                                     rd)
    # A view no reweighting of the sample reaches.
    @test_throws DomainError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                       cvar_views = ConditionalValueatRiskView(;
                                                                                               views = LinearConstraintEstimator(;
                                                                                                                                 val = "AAPL >= 10"))),
                                   rd)
    @test_throws DomainError prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                       evar_views = EntropicValueatRiskView(;
                                                                                            views = LinearConstraintEstimator(;
                                                                                                                              val = "AAPL >= 10"))),
                                   rd)
end

@testset "tail view formulation constructors" begin
    @test IntegerConditionalValueatRiskView().sbar === nothing
    @test IntegerConditionalValueatRiskView(; sbar = 25).sbar == 25
    @test_throws DomainError IntegerConditionalValueatRiskView(; sbar = 0)
    @test_throws DomainError IntegerConditionalValueatRiskView(; sbar = 1.5)
    @test GridEntropicValueatRiskView().K == 11
    @test_throws DomainError GridEntropicValueatRiskView(; K = 10)
    @test_throws DomainError GridEntropicValueatRiskView(; pct = 1.5)
    @test_throws DomainError GridEntropicValueatRiskView(; M = 0)
    # The anchor that centres the grid runs from a grid formulation alone, so its number of
    # steps and its tolerance live there rather than on the view group.
    for T in (GridEntropicValueatRiskView, GridRelativisticValueatRiskView)
        # `K` sizes a binary block of the mixed-integer program an upper-bound or equality
        # view builds, so it takes a resource cap of its own (ADR 0041). The parity check
        # alone admitted a grid of ten million binaries from one view.
        @test_throws DomainError T(; K = 10_000_001)
        @test T().iters == 50
        @test T().tol == 1e-10
        @test T().tilt_iters == 200
        @test_throws DomainError T(; iters = 0)
        @test_throws DomainError T(; tol = -1)
        @test_throws DomainError T(; tilt_iters = 0)
    end
    # The tilt of one row bisects to the resolution of the floating-point type, which for
    # `Float64` it reaches near step 64, so the cap binds only below that.
    let c = exp.(range(0, 1; length = 8)), w = fill(inv(8), 8)
        b = (minimum(c) + maximum(c)) / 2
        @test PO.ep_row_tilt(w, c, b) ≈ PO.ep_row_tilt(w, c, b; iters = 200)
        @test PO.ep_row_tilt(w, c, b) ≈ PO.ep_row_tilt(w, c, b; iters = 64)
        @test !(PO.ep_row_tilt(w, c, b) ≈ PO.ep_row_tilt(w, c, b; iters = 4))
        @test_throws DomainError PO.ep_row_tilt(w, c, b; iters = 0)
    end
    # `ep_sbar` resolves a count, a fraction and the rule of thumb.
    # `ord` is ascending, as the paper writes it, so the rule of thumb counts positions
    # from the end. Pin the count itself, which is what fixes the direction of the scan.
    ord = sortperm(-rd.X[:, 1])
    s = findfirst(>=(ep_a), cumsum(ep_w0[reverse(ord)]))
    @test PO.ep_sbar(37, T) == 37
    @test PO.ep_sbar(10 * T, T) == T
    @test PO.ep_sbar(0.25, T) == ceil(Int, 0.25 * T)
    @test PO.ep_sbar(nothing, T, ep_a, ep_w0, ord) ==
          min(T, max(2 * s, ceil(Int, 2 * ep_a * T), 1))
    @test PO.ep_sbar(nothing, T, ep_a, ep_w0, ord) >= ceil(Int, 2 * ep_a * T)
    @test PO.ep_sbar(nothing, T, ep_a, ep_w0, ord) <= T
end

# --------------------------------------------------------------------------------------
# The mixed-integer formulations. They need a solver that handles mixed-integer
# exponential cone programs, and the outer approximation Pajarito runs costs one conic
# solve per round, so these run on a short window rather than on `rd`.
# --------------------------------------------------------------------------------------
ep_mip_slv = [Solver(; name = :ep_mip,
                     solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                        "verbose" => false,
                                                        "oa_solver" =>
                                                            optimizer_with_attributes(HiGHS.Optimizer,
                                                                                      JuMP.MOI.Silent() =>
                                                                                          true),
                                                        "conic_solver" =>
                                                            optimizer_with_attributes(Clarabel.Optimizer,
                                                                                      "verbose" =>
                                                                                          false)),
                     check_sol = (; allow_local = true, allow_almost = true))]
ep_mopt = JuMPEntropyPooling(; slv = ep_mip_slv)
ep_sX = rd.X[(end - 200):end, :]
ep_srd = ReturnsResult(; nx = rd.nx, X = ep_sX)
ep_sT = size(ep_sX, 1)
ep_sw0 = StatsBase.pweights(fill(inv(ep_sT), ep_sT))
ep_scvar(j, wi) = ConditionalValueatRisk(; alpha = ep_a, w = wi)(ep_sX[:, j])
ep_sevar(j, wi) = PO.ep_evar(-ep_sX[:, j], wi, ep_a).evar
const ep_spcvar = ep_scvar(1, ep_sw0)
const ep_spcvarN = ep_scvar(size(ep_sX, 2), ep_sw0)
const ep_spevar = ep_sevar(1, ep_sw0)
ep_srlvar(j, wi) = PO.ep_rlvar(-ep_sX[:, j], wi, ep_a, ep_k).rlvar
const ep_sprlvar = ep_srlvar(1, ep_sw0)
ep_srlvar_at(j, wi, a, k) = PO.ep_rlvar(-ep_sX[:, j], wi, a, k).rlvar

@testset "RLVaR view groups carry their own level and deformation" begin
    # The deformation parameter is part of the statistic, exactly as the significance
    # level is, so it lives on the view group. One estimator holds groups at several
    # pairs, and one solve answers them all. Two power cone blocks in one model is a
    # demanding solve, so this runs on the short window.
    sopt = JuMPEntropyPooling(; slv = slv)
    r1 = ep_srlvar_at(1, ep_sw0, 0.05, 0.1)
    r2 = ep_srlvar_at(20, ep_sw0, 0.20, 0.5)
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = sopt,
                                   rlvar_views = [RelativisticValueatRiskView(;
                                                                              alpha = 0.05,
                                                                              kappa = 0.1,
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL >= $(r1 * 1.04)")),
                                                  RelativisticValueatRiskView(;
                                                                              alpha = 0.20,
                                                                              kappa = 0.5,
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "XOM >= $(r2 * 1.04)"))]),
               ep_srd)
    @test isapprox(ep_srlvar_at(1, pr.w, 0.05, 0.1), r1 * 1.04, rtol = 2e-3)
    @test isapprox(ep_srlvar_at(20, pr.w, 0.20, 0.5), r2 * 1.04, rtol = 2e-3)

    # A `prior(...)` reference read at the wrong deformation could not pass this: the two
    # values are far apart.
    @test !isapprox(r1, ep_srlvar_at(1, ep_sw0, 0.05, 0.5), rtol = 1e-2)
    pr2 = prior(EntropyPoolingPrior(; sets = sets, opt = sopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              alpha = 0.05,
                                                                              kappa = 0.1,
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL >= 1.04*prior(AAPL)"))),
                ep_srd)
    @test isapprox(ep_srlvar_at(1, pr2.w, 0.05, 0.1), r1 * 1.04, rtol = 2e-3)

    # A group that states neither takes the 0.05 and 0.3 defaults.
    pr3 = prior(EntropyPoolingPrior(; sets = sets, opt = sopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL >= $(ep_sprlvar * 1.04)"))),
                ep_srd)
    @test isapprox(ep_srlvar_at(1, pr3.w, 0.05, 0.3), ep_sprlvar * 1.04, rtol = 2e-3)
end

@testset "integer CVaR agrees with the linear formulation on a lower bound" begin
    # The two formulations describe the same set, so they must land on the same
    # posterior. This is what pins the direction of the monotonicity constraint: the
    # marked observations have to be a suffix of the ascending losses, or the model's
    # expression is the mean of some other subset and the two answers part company.
    tgt = ep_spcvar * 1.2
    v = LinearConstraintEstimator(; val = "AAPL >= $tgt")
    pl = prior(EntropyPoolingPrior(; sets = sets, opt = JuMPEntropyPooling(; slv = slv),
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           alg = LinearConditionalValueatRiskView(),
                                                                           views = v)),
               ep_srd)
    pi_ = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                    cvar_views = ConditionalValueatRiskView(;
                                                                            alg = IntegerConditionalValueatRiskView(),
                                                                            views = v)),
                ep_srd)
    @test isapprox(ep_scvar(1, pl.w), tgt, rtol = 1.0e-4)
    @test isapprox(ep_scvar(1, pi_.w), tgt, rtol = 1.0e-3)
    # The two constraint sets describe the same feasible posterior, so the integer solve
    # cannot beat the continuous one. What separates them is the outer approximation gap
    # Pajarito stops on, which shows up as a slightly larger divergence and a visibly
    # different split of the tail mass across the same observations.
    @test pi_.kld >= pl.kld * (1 - 1.0e-6)
    @test isapprox(pi_.kld, pl.kld, rtol = 1.0e-2)
    @test isapprox(pl.w, pi_.w, rtol = 5.0e-2, atol = 1.0e-6)
end

@testset "integer CVaR expresses the operators the linear formulation cannot" begin
    tgt = ep_spcvar * 0.85
    for op in ("<=", "==")
        pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                       cvar_views = ConditionalValueatRiskView(;
                                                                               views = LinearConstraintEstimator(;
                                                                                                                 val = "AAPL $op $tgt"))),
                   ep_srd)
        @test isapprox(ep_scvar(1, pr.w), tgt, rtol = 1e-3)
    end
    # An explicit `sbar` is honoured.
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           alg = IntegerConditionalValueatRiskView(;
                                                                                                                   sbar = 40),
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL <= $tgt"))),
               ep_srd)
    @test isapprox(ep_scvar(1, pr.w), tgt, rtol = 1e-3)
end

@testset "relative CVaR view over two assets" begin
    gap = ep_spcvar - ep_spcvarN
    tgt = gap + 0.01
    nN = rd.nx[size(ep_sX, 2)]
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL - $nN == $tgt"))),
               ep_srd)
    @test isapprox(ep_scvar(1, pr.w) - ep_scvar(size(ep_sX, 2), pr.w), tgt, rtol = 5e-3)
end

@testset "grid EVaR expresses the operators the conic formulation cannot" begin
    tgt = ep_spevar * 0.9
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                   evar_views = EntropicValueatRiskView(;
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "AAPL <= $tgt"))),
               ep_srd)
    # The upper bound holds at one grid point, so it is conservative, never violated.
    @test ep_sevar(1, pr.w) <= tgt * (1 + 1e-6)

    pr2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                    evar_views = EntropicValueatRiskView(;
                                                                         views = LinearConstraintEstimator(;
                                                                                                           val = "AAPL == $(ep_spevar * 0.92)"))),
                ep_srd)
    @test isapprox(ep_sevar(1, pr2.w), ep_spevar * 0.92, rtol = 2e-2)
end

@testset "CVaR views on a group" begin
    # A group view is the sum of the members' CVaRs, so it names two assets and takes the
    # integer formulation whatever its operator. `prior(gA)` is the sum of their prior
    # CVaRs, which is what the target is stated against.
    pA = sum(ep_scvar(j, ep_sw0) for j in 1:2)
    pr = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_mopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "gA == 1.2*prior(gA)"))),
               ep_srd)
    @test isapprox(sum(ep_scvar(j, pr.w) for j in 1:2), 1.2 * pA, rtol = 5e-3)
end

@testset "relative CVaR view on two groups" begin
    pA = sum(ep_scvar(j, ep_sw0) for j in 1:2)
    pB = sum(ep_scvar(j, ep_sw0) for j in 3:4)
    tgt = (pA - pB) + 0.02
    pr = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_mopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "gA - gB == $tgt"))),
               ep_srd)
    @test isapprox(sum(ep_scvar(j, pr.w) for j in 1:2) -
                   sum(ep_scvar(j, pr.w) for j in 3:4), tgt, rtol = 5e-3)
end

@testset "relative CVaR view over several assets" begin
    # Three assets, two of them on the same side of the view.
    tgt = ep_scvar(1, ep_sw0) + ep_scvar(2, ep_sw0) - ep_scvar(3, ep_sw0) + 0.02
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "$(rd.nx[1]) + $(rd.nx[2]) - $(rd.nx[3]) >= $tgt"))),
               ep_srd)
    @test isapprox(ep_scvar(1, pr.w) + ep_scvar(2, pr.w) - ep_scvar(3, pr.w), tgt,
                   rtol = 5e-3)
end

@testset "grid RLVaR expresses the operators the conic formulation cannot" begin
    # The grid fixes the primal pair, which is what turns the objective into a row that is
    # linear in the posterior probabilities. Every grid point is a feasible point of the
    # primal programme, so the upper bound it writes is conservative, never violated.
    tgt = ep_sprlvar * 0.85
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                   rlvar_views = RelativisticValueatRiskView(;
                                                                             views = LinearConstraintEstimator(;
                                                                                                               val = "AAPL <= $tgt"))),
               ep_srd)
    @test ep_srlvar(1, pr.w) <= tgt * (1 + 1e-6)
    @test isapprox(ep_srlvar(1, pr.w), tgt, rtol = 1e-2)

    # An equality below the prior writes both blocks: the lower-bound rows and the
    # upper-bound selector.
    pr2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL == $tgt"))),
                ep_srd)
    @test isapprox(ep_srlvar(1, pr2.w), tgt, rtol = 1e-2)

    # A grid of its own on the group.
    pr3 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              alg = GridRelativisticValueatRiskView(;
                                                                                                                    pct = 0.8,
                                                                                                                    K = 21),
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL <= $tgt"))),
                ep_srd)
    @test ep_srlvar(1, pr3.w) <= tgt * (1 + 1e-6)
end

@testset "RLVaR lower bound, grid formulation, reaches the dual formulation" begin
    # The lower-bound half of the grid formulation is linear in the posterior
    # probabilities alone, so it is one of the two tail views the dual formulation carries.
    tgt = ep_sprlvar * 1.03
    v = LinearConstraintEstimator(; val = "AAPL >= $tgt")
    w = nothing
    for opt in (OptimEntropyPooling(), JuMPEntropyPooling(; slv = slv))
        pr = prior(EntropyPoolingPrior(; sets = sets, opt = opt,
                                       rlvar_views = RelativisticValueatRiskView(;
                                                                                 alg = GridRelativisticValueatRiskView(),
                                                                                 views = v)),
                   ep_srd)
        # The grid enforces the bound at its points alone, so the answer is approximate.
        @test isapprox(ep_srlvar(1, pr.w), tgt, rtol = 5e-3)
        w = isnothing(w) ? pr.w : (@test isapprox(w, pr.w, rtol = 1e-4); w)
    end
end

@testset "the grid RLVaR defaults are measured, not copied" begin
    # Issue #525. The row is divided by its largest coefficient, so its coefficients sit in
    # `(0, 1]` and the posterior sums to one: the left-hand side of a row cannot exceed one
    # whatever the data. The default `M` clears that bound by an order of magnitude, and the
    # measurement over the fixture found no right-hand side that narrows the margin.
    dflt = GridRelativisticValueatRiskView()
    @test dflt.pct == 0.5
    @test dflt.K == 11
    @test dflt.M == 10
    x = -ep_sX[:, 1]
    for a in (0.01, 0.05, 0.1), k in (0.1, 0.3, 0.5, 0.7), m in (0.5, 0.75, 0.95)
        r = PO.ep_rlvar(x, ep_sw0, a, k)
        vbar = m * r.rlvar
        delta = r.rlvar - vbar
        lnk = PO.kappa_log(inv(a * ep_sT), k)
        for zk in range(r.z * (1 - dflt.pct), r.z * (1 + dflt.pct); length = dflt.K)
            tk = PO.ep_rlvar_shift(x, ep_sw0, k, lnk, zk).t - delta
            c, b = PO.ep_rlvar_grid_row(x, vbar, tk, zk, a, k)
            @test all(isfinite, c)
            @test isfinite(b)
            @test all(>(0), c)
            @test isapprox(maximum(c), 1)
            # A deselected row must be slack, and `1 - b` is the largest it can ask for.
            @test dflt.M >= 1 - b
        end
    end
end

@testset "a grid point whose row overflows is dropped" begin
    # Issue #525. `ep_rlvar_tail` overflows at a dual variable near zero, which is where the
    # grid sits when `kappa` approaches one. Before the guard the coefficients reached JuMP
    # as `NaN * x[j]`, which it refuses with a message that names neither the view nor
    # `kappa`. A dropped point is one the grid never had, so the upper bound still holds at
    # the points it keeps: each of those is a feasible point of the primal programme.
    x = -ep_sX[:, 1]
    add(kappa, m, op) = begin
        r = PO.ep_rlvar(x, ep_sw0, ep_a, kappa)
        epc = Dict{Any, Any}()
        tvs = Any[]
        PO.ep_add_rlvar_view!(epc, tvs, GridRelativisticValueatRiskView(), x, ep_a, kappa,
                              op, m * r.rlvar, ep_sw0, r.z, r.rlvar,
                              "AAPL <= $(m * r.rlvar)")
        return epc, tvs
    end
    # Nothing overflows here, so the grid keeps every point it built.
    _, tvs = add(0.7, 0.75, :leq)
    @test length(only(tvs).z) == GridRelativisticValueatRiskView().K
    # Here six of the eleven points overflow. The five that remain carry finite rows.
    _, tvs = add(0.9, 0.75, :leq)
    tv = only(tvs)
    @test length(tv.z) == length(tv.t) < GridRelativisticValueatRiskView().K
    for k in eachindex(tv.z)
        c, b = PO.ep_rlvar_grid_row(x, tv.rhs, tv.t[k], tv.z[k], ep_a, 0.9)
        @test all(isfinite, c)
        @test isfinite(b)
    end
    # The lower-bound block writes its rows straight into the constraint set, and it drops
    # the same points.
    epc, _ = add(0.9, 0.75, :geq)
    lhs, rhs = epc[:ineq]
    @test size(lhs, 1) == length(rhs) == length(tv.z)
    @test all(isfinite, lhs)
    @test all(isfinite, rhs)
    # Every point overflows here, and the raise names `kappa` rather than leaving JuMP to
    # refuse a `NaN` coefficient.
    @test_throws ArgumentError add(0.9, 0.5, :leq)
end

@testset "one row is made tight by an exponential tilt" begin
    # Issue #530. The row of a grid point is linear in the posterior probabilities, so the
    # posterior that makes it tight at the smallest relative entropy is an exponential tilt
    # along its coefficients. It is the entropy pooling answer to that one row, and the
    # anchor of the RLVaR grid iterates on it.
    x = -ep_sX[:, 1]
    r = PO.ep_rlvar(x, ep_sw0, ep_a, ep_k)
    c, b = PO.ep_rlvar_grid_row(x, 0.75 * r.rlvar, r.t, r.z, ep_a, ep_k)
    lo, hi = extrema(c)
    # A value inside the range of the coefficients is attained exactly, from either side of
    # the prior, and the answer is a probability vector. The last of the three sits close to
    # the smallest coefficient, which is where the bracket has to travel furthest.
    for tgt in (0.25 * lo + 0.75 * hi, 0.5 * (lo + hi), 0.999 * lo + 0.001 * hi)
        q = PO.ep_row_tilt(ep_sw0, c, tgt)
        @test isapprox(sum(q), 1)
        @test all(>=(0), q)
        @test isapprox(LinearAlgebra.dot(c, q), tgt, rtol = 1e-10)
    end
    # The prior itself is the answer when the prior already attains the value.
    q = PO.ep_row_tilt(ep_sw0, c, LinearAlgebra.dot(c, ep_sw0))
    @test isapprox(q, collect(ep_sw0), rtol = 1e-8)
    # No probability vector reaches outside the range of the coefficients, so there is no
    # tilt to return.
    @test isnothing(PO.ep_row_tilt(ep_sw0, c, lo))
    @test isnothing(PO.ep_row_tilt(ep_sw0, c, hi))
    @test isnothing(PO.ep_row_tilt(ep_sw0, c, lo - one(lo)))
    @test isnothing(PO.ep_row_tilt(ep_sw0, c, hi + one(hi)))
end

# The entropy pooling answer to a grid, without a solver. The grid asks for one of its rows,
# so the posterior is the cheapest tilt over the rows that are finite and reachable. Issue
# #530 checked this reading against a Pajarito solve and the two agreed to four digits.
function ep_grid_post(x, w, alpha, kappa, rhs, t, z)
    bq, bd = nothing, Inf
    for k in eachindex(t, z)
        c, b = PO.ep_rlvar_grid_row(x, rhs, t[k], z[k], alpha, kappa)
        if !(all(isfinite, c) && isfinite(b))
            continue
        end
        q = LinearAlgebra.dot(c, w) <= b ? collect(w) : PO.ep_row_tilt(w, c, b)
        if isnothing(q)
            continue
        end
        d = sum(qj > zero(qj) ? qj * log(qj / wj) : zero(qj) for (qj, wj) in zip(q, w))
        d < bd && (bd = d; bq = q)
    end
    return bq, bd
end

@testset "the RLVaR grid is centred on the point the posterior attains" begin
    # Issue #530. An upper-bound view reaches its target only where the grid holds the pair
    # the posterior itself attains, and that pair is not the prior's: over this fixture its
    # dual variable reaches five hundred times the prior's, which no `pct` below one spans.
    # The anchor solves for the pair instead of guessing it, so the width covers only the
    # movement the other views of the model cause.
    #
    # The cell below is the one issue #530 reports: `XOM`, `alpha = 0.01`, `kappa = 0.7`, a
    # target five per cent below the prior. The grid centred on the prior lands twenty-three
    # per cent below it. A lower-bound view keeps that centre, so the two calls are the grid
    # before the fix and the grid after it.
    x = -ep_sX[:, 20]
    r = PO.ep_rlvar(x, ep_sw0, 0.01, 0.7)
    tgt = 0.95 * r.rlvar
    told, zold = PO.ep_rlvar_grid(x, ep_sw0, 0.01, 0.7, :geq, tgt, r.z, r.rlvar, 0.5, 11)
    tnew, znew = PO.ep_rlvar_grid(x, ep_sw0, 0.01, 0.7, :leq, tgt, r.z, r.rlvar, 0.5, 11)
    qold, dold = ep_grid_post(x, ep_sw0, 0.01, 0.7, tgt, told, zold)
    qnew, dnew = ep_grid_post(x, ep_sw0, 0.01, 0.7, tgt, tnew, znew)
    @test isapprox(abs(PO.ep_rlvar(x, qold, 0.01, 0.7).rlvar - tgt) / tgt, 0.22995,
                   rtol = 1e-3)
    @test abs(PO.ep_rlvar(x, qnew, 0.01, 0.7).rlvar - tgt) / tgt <= 1e-8
    # The anchored grid reaches the target at a *lower* relative entropy than the grid
    # centred on the prior reaches its own answer, so the target is not bought with
    # divergence.
    @test dnew < dold
    # A lower-bound view is centred on the prior's dual variable, and `K` is odd, so the
    # centre is a grid point.
    @test isapprox(zold[(length(zold) + 1) ÷ 2], r.z, rtol = 1e-12)
    # An upper-bound view is centred on the anchor instead, and the anchor's dual variable
    # leaves the span the prior's admits, which is `2 * zstar` at most.
    anc = PO.ep_rlvar_anchor(x, ep_sw0, 0.01, 0.7, tgt, r.t - (r.rlvar - tgt), r.z)
    @test isapprox(znew[(length(znew) + 1) ÷ 2], anc.z, rtol = 1e-12)
    @test anc.z > 100 * r.z
    @test isapprox(PO.ep_rlvar(x, anc.w, 0.01, 0.7).rlvar, tgt, rtol = 1e-10)
    # The shift of the centre is the shift that minimises at the anchor's own posterior, so
    # the grid passes through the pair rather than through its dual variable alone.
    @test isapprox(tnew[(length(tnew) + 1) ÷ 2], anc.t, rtol = 1e-6)

    # The anchor holds over the design issue #530 measured: two levels, four deformations
    # and two targets below the prior, on two assets.
    for j in (1, 20), a in (0.01, 0.05), k in (0.1, 0.3, 0.5, 0.7), m in (0.6, 0.85)
        xj = -ep_sX[:, j]
        rj = PO.ep_rlvar(xj, ep_sw0, a, k)
        mj = m * rj.rlvar
        t, z = PO.ep_rlvar_grid(xj, ep_sw0, a, k, :leq, mj, rj.z, rj.rlvar, 0.5, 11)
        q, _ = ep_grid_post(xj, ep_sw0, a, k, mj, t, z)
        @test abs(PO.ep_rlvar(xj, q, a, k).rlvar - mj) / mj <= 1e-6
    end
end

@testset "the RLVaR anchor gives up rather than answer wrongly" begin
    # Issue #530. The iteration answers nothing on three paths, and each falls back to the
    # grid centred on the prior, which is the grid a lower-bound view takes.
    x = -ep_sX[:, 1]
    r = PO.ep_rlvar(x, ep_sw0, ep_a, ep_k)
    tgt = 0.75 * r.rlvar
    # It reaches the target from the prior's pair, and it takes more than one step to do it.
    @test !isnothing(PO.ep_rlvar_anchor(x, ep_sw0, ep_a, ep_k, tgt, r.t - (r.rlvar - tgt),
                                        r.z))
    @test isnothing(PO.ep_rlvar_anchor(x, ep_sw0, ep_a, ep_k, tgt, r.t - (r.rlvar - tgt),
                                       r.z; iters = 1))
    # A target above the prior asks one row to take a value outside the range of its
    # coefficients, which no probability vector attains.
    up = 1.25 * r.rlvar
    @test isnothing(PO.ep_rlvar_anchor(x, ep_sw0, ep_a, ep_k, up, r.t - (r.rlvar - up),
                                       r.z))
    # `ep_rlvar_tail` overflows at the dual variable that attains the RLVaR as `kappa`
    # approaches one, so the row the iteration reads is not finite.
    r9 = PO.ep_rlvar(x, ep_sw0, ep_a, 0.9)
    d9 = 0.5 * r9.rlvar
    @test isnothing(PO.ep_rlvar_anchor(x, ep_sw0, ep_a, 0.9, d9, r9.t - (r9.rlvar - d9),
                                       r9.z))
    # Both of those cells fall back, so an upper-bound view builds the grid a lower-bound
    # view builds.
    for (k, m) in ((ep_k, 1.25), (0.9, 0.5))
        rk = PO.ep_rlvar(x, ep_sw0, ep_a, k)
        mk = m * rk.rlvar
        @test PO.ep_rlvar_grid(x, ep_sw0, ep_a, k, :leq, mk, rk.z, rk.rlvar, 0.5, 11) ==
              PO.ep_rlvar_grid(x, ep_sw0, ep_a, k, :geq, mk, rk.z, rk.rlvar, 0.5, 11)
    end
end

# The entropy pooling answer to an EVaR grid, without a solver. The grid asks for one of its
# rows, so the posterior is the cheapest tilt over the rows that are reachable. It is the
# reading `ep_grid_post` gives the RLVaR grid, on the row `ep_evar_grid_row` builds.
function ep_evar_grid_post(x, w, alpha, rhs, z)
    bq, bd = nothing, Inf
    for zk in z
        c, isc = PO.ep_evar_grid_row(x, rhs, zk)
        b = alpha * isc
        if !(all(isfinite, c) && isfinite(b))
            continue
        end
        q = LinearAlgebra.dot(c, w) <= b ? collect(w) : PO.ep_row_tilt(w, c, b)
        if isnothing(q)
            continue
        end
        d = sum(qj > zero(qj) ? qj * log(qj / wj) : zero(qj) for (qj, wj) in zip(q, w))
        d < bd && (bd = d; bq = q)
    end
    return bq, bd
end

@testset "the EVaR grid is centred on the point the posterior attains" begin
    # The RLVaR grid carries two devices, and an EVaR grid needs one of them. The shift of
    # the primal programme of EVaR is closed form in the target and the dual variable, and
    # `ep_evar_grid_row` already carries it, so the translation `ep_rlvar_grid` applies to
    # its shift buys an EVaR grid nothing. The anchor is the half that does buy something:
    # an upper-bound view reaches its target only where the grid holds the dual variable the
    # posterior itself attains, and that one is not the prior's.
    x = -rd.X[:, 1]
    e = PO.ep_evar(x, ep_w0, ep_a)
    tgt = 0.4 * e.evar
    zold = PO.ep_evar_grid(x, ep_w0, ep_a, :geq, tgt, e.z, 0.5, 11)
    znew = PO.ep_evar_grid(x, ep_w0, ep_a, :leq, tgt, e.z, 0.5, 11)
    qold, dold = ep_evar_grid_post(x, ep_w0, ep_a, tgt, zold)
    qnew, dnew = ep_evar_grid_post(x, ep_w0, ep_a, tgt, znew)
    # The grid centred on the prior lands seventeen per cent below the target.
    @test isapprox(PO.ep_evar(x, qold, ep_a).evar / tgt, 0.82608, rtol = 1e-4)
    # The anchored grid reaches it.
    @test isapprox(PO.ep_evar(x, qnew, ep_a).evar, tgt, rtol = 1e-8)
    # It reaches it at half the relative entropy the grid centred on the prior pays for its
    # own answer, so the target is not bought with divergence.
    @test dnew < dold
    @test isapprox(dold, 0.074531, rtol = 1e-4)
    @test isapprox(dnew, 0.037877, rtol = 1e-4)
    # A lower-bound view keeps the prior's dual variable as the centre, and `K` is odd, so
    # the centre is a grid point.
    @test isapprox(zold[(length(zold) + 1) ÷ 2], e.z, rtol = 1e-12)
    # An upper-bound view is centred on the anchor instead, whose dual variable is under a
    # quarter of the prior's. That leaves the span the prior's admits, which is `2 * zstar`
    # at most.
    anc = PO.ep_evar_anchor(x, ep_w0, ep_a, tgt, e.z)
    @test isapprox(znew[(length(znew) + 1) ÷ 2], anc.z, rtol = 1e-12)
    @test isapprox(anc.z / e.z, 0.23115, rtol = 1e-4)
    @test isapprox(PO.ep_evar(x, anc.w, ep_a).evar, tgt, rtol = 1e-10)
    # The anchor holds over three assets, two levels and four targets below the prior.
    for j in (1, 5, 20), a in (0.01, 0.05), m in (0.4, 0.6, 0.8, 0.95)
        xj = -rd.X[:, j]
        ej = PO.ep_evar(xj, ep_w0, a)
        mj = m * ej.evar
        zj = PO.ep_evar_grid(xj, ep_w0, a, :leq, mj, ej.z, 0.5, 11)
        q, _ = ep_evar_grid_post(xj, ep_w0, a, mj, zj)
        @test abs(PO.ep_evar(xj, q, a).evar - mj) / mj <= 1e-6
    end
end

@testset "the EVaR anchor gives up rather than answer wrongly" begin
    # The iteration answers nothing on two paths, and each falls back to the grid centred on
    # the prior, which is the grid a lower-bound view takes.
    x = -rd.X[:, 1]
    e = PO.ep_evar(x, ep_w0, ep_a)
    tgt = 0.75 * e.evar
    # It reaches the target from the prior's dual variable, and it takes more than one step.
    @test !isnothing(PO.ep_evar_anchor(x, ep_w0, ep_a, tgt, e.z))
    @test isnothing(PO.ep_evar_anchor(x, ep_w0, ep_a, tgt, e.z; iters = 1))
    # A target far above the prior asks the row to take a value outside the range of its
    # coefficients, which no probability vector attains.
    up = 2 * e.evar
    @test isnothing(PO.ep_evar_anchor(x, ep_w0, ep_a, up, e.z))
    # That cell falls back, so an upper-bound view builds the grid a lower-bound view builds.
    @test PO.ep_evar_grid(x, ep_w0, ep_a, :leq, up, e.z, 0.5, 11) ==
          PO.ep_evar_grid(x, ep_w0, ep_a, :geq, up, e.z, 0.5, 11)
end

@testset "a tail view group carries its own Optim settings" begin
    # `args` and `kwargs` reach every `Optim.optimize` the group's views run. The spans those
    # searches run over are situational, so each is optional and `nothing` resolves at the
    # point of use: the EVaR carries the one number it needs as a field of the view group,
    # and the RLVaR carries its three in a `RelativisticValueatRiskViewBracket`.
    x = -rd.X[:, 1]
    r = PO.ep_rlvar(x, ep_w0, ep_a, ep_k)
    @test PO.ep_evar(x, ep_w0, ep_a) ==
          PO.ep_evar(x, ep_w0, ep_a; args = (PO.Optim.Brent(),))
    # The bracket a caller states by hand is the bracket `nothing` resolves to.
    @test r == PO.ep_rlvar(x, ep_w0, ep_a, ep_k; args = (PO.Optim.Brent(),),
                           bracket = RelativisticValueatRiskViewBracket())
    @test r == PO.ep_rlvar(x, ep_w0, ep_a, ep_k;
                           bracket = RelativisticValueatRiskViewBracket(; tspan = 2,
                                                                        log_zlo = -20,
                                                                        log_zhi = 10))
    # A search stopped before it converges raises, rather than answering with the point it
    # reached.
    @test_throws ErrorException PO.ep_evar(x, ep_w0, ep_a; kwargs = (; iterations = 1))
    @test_throws ErrorException PO.ep_rlvar(x, ep_w0, ep_a, ep_k;
                                            kwargs = (; iterations = 1))
    # The constructor of the bracket holds the defaults and the validation, so a span the
    # search cannot use never reaches it.
    @test RelativisticValueatRiskViewBracket().tspan == 2
    @test RelativisticValueatRiskViewBracket().log_zlo == -20
    @test RelativisticValueatRiskViewBracket().log_zhi == 10
    @test_throws DomainError RelativisticValueatRiskViewBracket(; tspan = 0)
    @test_throws DomainError RelativisticValueatRiskViewBracket(; log_zlo = 20)
    @test_throws DomainError RelativisticValueatRiskViewBracket(; log_zlo = 5, log_zhi = 5)
    # The EVaR carries one number rather than a struct, and the view group validates it.
    @test_throws DomainError PO.ep_evar(x, ep_w0, ep_a; zlo_frac = 2.0)
    @test_throws DomainError PO.ep_evar(x, ep_w0, ep_a; zlo_frac = 0.0)
    # A wider span answers the same value, because the default already holds the minimiser.
    @test isapprox(PO.ep_evar(x, ep_w0, ep_a; zlo_frac = 1e-6).evar, ep_pevar, rtol = 1e-10)
    @test isapprox(PO.ep_rlvar(x, ep_w0, ep_a, ep_k;
                               bracket = RelativisticValueatRiskViewBracket(; tspan = 5)).rlvar,
                   r.rlvar, rtol = 1e-8)
    # The fields carry the settings, and both spans default to `nothing`.
    v = LinearConstraintEstimator(; val = "AAPL >= $(1.2 * ep_pevar)")
    ev = EntropicValueatRiskView(; views = v, args = (PO.Optim.Brent(),),
                                 kwargs = (; abs_tol = 1e-10), zlo_frac = 1e-6)
    @test ev.args == (PO.Optim.Brent(),)
    @test ev.kwargs == (; abs_tol = 1e-10)
    @test ev.zlo_frac == 1e-6
    @test isnothing(EntropicValueatRiskView(; views = v).zlo_frac)
    @test_throws DomainError EntropicValueatRiskView(; views = v, zlo_frac = 1.5)
    bkt = RelativisticValueatRiskViewBracket(; tspan = 3)
    rv = RelativisticValueatRiskView(; views = v, bracket = bkt)
    @test rv.bracket === bkt
    @test rv.args == ()
    @test rv.kwargs == (;)
    @test isnothing(RelativisticValueatRiskView(; views = v).bracket)
    # A bracket built for the other statistic is a type the field does not admit. The
    # keyword carries the bound, so the raise names the field and the type it wanted.
    @test_throws TypeError RelativisticValueatRiskView(; views = v, bracket = (; tspan = 3))
    # They reach the solve. A group that states the default answers as a group that leaves
    # it out.
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   evar_views = EntropicValueatRiskView(;
                                                                        alg = GridEntropicValueatRiskView(),
                                                                        views = v)), rd)
    pr2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    evar_views = EntropicValueatRiskView(;
                                                                         alg = GridEntropicValueatRiskView(),
                                                                         views = v,
                                                                         args = (PO.Optim.Brent(),),
                                                                         zlo_frac = sqrt(eps(Float64)))),
                rd)
    @test isapprox(pr.w, pr2.w, rtol = 1e-10)
    # A group that moves the span answers near the same place, and not at the same place:
    # the end of the bracket moves where Brent samples, so it moves the dual variable the
    # grid is centred on in the last digits.
    pr3 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    evar_views = EntropicValueatRiskView(;
                                                                         alg = GridEntropicValueatRiskView(),
                                                                         views = v,
                                                                         zlo_frac = 1e-8)),
                rd)
    @test isapprox(pr.w, pr3.w, rtol = 1e-3)
    @test !isapprox(pr.w, pr3.w, rtol = 1e-9)
    # A group whose search cannot converge raises rather than building a wrong grid.
    @test_throws ErrorException prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                                          evar_views = EntropicValueatRiskView(;
                                                                                               views = v,
                                                                                               kwargs = (;
                                                                                                         iterations = 1))),
                                      rd)
    # The grid formulation's `iters` and `tol` reach the anchor. One step is not enough to
    # reach the target, so the anchor gives up and the grid falls back to the one a
    # lower-bound view builds, centred on the prior's dual variable.
    e = PO.ep_evar(x, ep_w0, ep_a)
    tgt = 0.4 * e.evar
    grid_of(alg) = begin
        epc, tvs = Dict{Symbol, Any}(), Any[]
        PO.ep_add_evar_view!(epc, tvs, alg, x, ep_a, :leq, tgt, ep_w0, e.z, e.evar, "AAPL")
        tvs[1].z
    end
    zanc = grid_of(GridEntropicValueatRiskView())
    zone = grid_of(GridEntropicValueatRiskView(; iters = 1))
    @test zone == PO.ep_evar_grid(x, ep_w0, ep_a, :geq, tgt, e.z, 0.5, 11)
    @test zanc != zone
    @test isapprox(zanc[6] / e.z, 0.23115, rtol = 1e-4)
    # A tolerance the iteration cannot miss stops it at the first step, which lands short of
    # the anchor the default tolerance reaches.
    @test grid_of(GridEntropicValueatRiskView(; tol = 1.0)) != zanc
end

@testset "a CVaR or EVaR view naming an asset the universe does not hold" begin
    using Logging
    # The counterpart of the RLVaR case above, for the two families whose walkers carry the
    # same guard. Under `strict = false` the name is reported and dropped, the view produces
    # no constraint, and the loop moves on.
    lce = LinearConstraintEstimator(; val = "NOTANASSET >= 0.1")
    for V in (ConditionalValueatRiskView, EntropicValueatRiskView)
        v = V(; views = lce, alpha = ep_a)
        epc = Dict{Symbol, Any}()
        tvs = PO.AbstractEntropyPoolingTailView[]
        Logging.with_logger(Logging.NullLogger()) do
            return PO.ep_tail_views!(v, epc, tvs, ep_pr0, ep_gsets, ep_w0; strict = false)
        end
        @test isempty(epc)
        @test isempty(tvs)
        # The same name under `strict = true` raises instead.
        @test_throws ArgumentError PO.ep_tail_views!(v, epc, tvs, ep_pr0, ep_gsets, ep_w0;
                                                     strict = true)
    end
end

@testset "an EVaR grid point whose row overflows is dropped" begin
    # The entropic counterpart of the relativistic case above. `ep_evar_grid_row` evaluates
    # `exp((x - rhs) / z)`, which overflows at a dual variable near zero. The grid spans
    # `zc * (1 - pct)` to `zc * (1 + pct)`, so its lower end sits there when `pct`
    # approaches one, and `pct` is only bound to the open unit interval. Before the guard
    # the scaled coefficients were `Inf / Inf`, and they reached JuMP as `NaN * x[j]`, whose
    # message names neither the view nor the setting that caused it.
    x = -ep_sX[:, 1]
    e = PO.ep_evar(x, ep_sw0, ep_a)
    tgt = 0.7 * e.evar
    add(pct, op; a = ep_a, ea = e) = begin
        epc = Dict{Symbol, Any}()
        tvs = Any[]
        PO.ep_add_evar_view!(epc, tvs, GridEntropicValueatRiskView(; pct = pct), x, a, op,
                             0.7 * ea.evar, ep_sw0, ea.z, ea.evar,
                             "AAPL <= $(0.7 * ea.evar)")
        return epc, tvs
    end
    # Nothing overflows at the default half-width, so the grid keeps every point it built.
    _, tvs = add(GridEntropicValueatRiskView().pct, :leq)
    @test length(only(tvs).z) == GridEntropicValueatRiskView().K
    # The lowest point overflows here, and the ones that remain carry finite rows.
    _, tvs = add(0.999, :leq)
    tv = only(tvs)
    @test length(tv.z) < GridEntropicValueatRiskView().K
    for zk in tv.z
        c, isc = PO.ep_evar_grid_row(x, tv.rhs, zk)
        @test all(isfinite, c)
        @test isfinite(isc)
    end
    # The lower-bound block writes its rows straight into the constraint set, and it drops
    # the same points.
    epc, _ = add(0.999, :geq)
    lhs, rhs = epc[:ineq]
    @test size(lhs, 1) == length(rhs) == length(tv.z)
    @test all(isfinite, lhs)
    @test all(isfinite, rhs)
    # A level that leaves fewer than one observation in the tail puts the whole grid there:
    # `ep_evar`'s minimiser is then at the end of its own bracket, a factor `sqrt(eps)`
    # below the top. Every point overflows, and the raise names the two settings that reach
    # it rather than leaving JuMP to refuse a `NaN` coefficient.
    small = PO.ep_evar(x, ep_sw0, 1e-3)
    @test ep_a * length(x) > 1 > 1e-3 * length(x)
    @test small.z / e.z < 1e-6
    @test_throws ArgumentError add(0.5, :leq; a = 1e-3, ea = small)
    @test_throws ArgumentError add(0.5, :geq; a = 1e-3, ea = small)
end

@testset "a grid of one point is the centre alone" begin
    # `K` is odd and at least one, so `K = 1` is a legal setting: one binary variable and
    # one row, at the point the anchor found. `range(lo, hi; length = 1)` refuses two ends
    # that differ, so both grids write that case out rather than raising from `Base`.
    x = -rd.X[:, 1]
    e = PO.ep_evar(x, ep_w0, ep_a)
    tgt = 0.7 * e.evar
    # A lower-bound view is centred on the prior's dual variable.
    @test PO.ep_evar_grid(x, ep_w0, ep_a, :geq, tgt, e.z, 0.5, 1) == [e.z]
    # An upper-bound view is centred on the anchor's.
    anc = PO.ep_evar_anchor(x, ep_w0, ep_a, tgt, e.z)
    @test PO.ep_evar_grid(x, ep_w0, ep_a, :leq, tgt, e.z, 0.5, 1) == [anc.z]
    # The half-width does not reach a grid of one point.
    @test PO.ep_evar_grid(x, ep_w0, ep_a, :geq, tgt, e.z, 0.9, 1) ==
          PO.ep_evar_grid(x, ep_w0, ep_a, :geq, tgt, e.z, 0.1, 1)
    # The relativistic grid answers the same way, and its shift follows its one point.
    r = PO.ep_rlvar(x, ep_w0, ep_a, ep_k)
    t1, z1 = PO.ep_rlvar_grid(x, ep_w0, ep_a, ep_k, :geq, 0.7 * r.rlvar, r.z, r.rlvar, 0.5,
                              1)
    @test z1 == [r.z]
    @test length(t1) == 1
    # The centre of a grid of `K` points is the one point a grid of one holds.
    z11 = PO.ep_evar_grid(x, ep_w0, ep_a, :leq, tgt, e.z, 0.5, 11)
    @test z11[6] == only(PO.ep_evar_grid(x, ep_w0, ep_a, :leq, tgt, e.z, 0.5, 1))
    # It reaches the view machinery, which builds one row from it. That path normalises the
    # weights first, and the renormalised last bits move where Brent samples, so the anchor
    # it finds agrees with the one above to about `sqrt(tol)`: the objective is flat of
    # second order at its minimum, so a `tol` of `1e-10` on the value buys far less on `z`.
    epc = Dict{Symbol, Any}()
    tvs = Any[]
    PO.ep_add_evar_view!(epc, tvs, GridEntropicValueatRiskView(; K = 1), x, ep_a, :leq, tgt,
                         ep_w0, e.z, e.evar, "AAPL <= $tgt")
    @test isapprox(only(only(tvs).z), anc.z, rtol = 1e-6)
end

@testset "the closed forms answer to numbers computed by hand" begin
    # Issue #541, condition 2. Each block below fixes one closed form against a number that
    # does not come from the library.

    # `ep_evar` against a `BigFloat` minimisation of the same objective. The value is flat
    # of second order at its minimum, so the minimiser is the looser of the two.
    function evar_bf(xf, wf, a)
        xb, wb, ab = BigFloat.(xf), BigFloat.(wf), BigFloat(a)
        wb ./= sum(wb)
        f = z -> z * (log(sum(wb .* exp.(xb ./ z))) - log(ab))
        hi = (maximum(xb) - sum(wb .* xb)) / (-log(ab))
        r = PO.Optim.optimize(f, hi * BigFloat(1e-8), hi, PO.Optim.Brent();
                              abs_tol = BigFloat(1e-40))
        return Float64(PO.Optim.minimum(r))
    end
    for a in (0.01, 0.05, 0.25), j in (1, 5, 20)
        @test isapprox(PO.ep_evar(-rd.X[:, j], ep_w0, a).evar,
                       evar_bf(-rd.X[:, j], collect(ep_w0), a), rtol = 1e-14)
    end

    # `ep_evar_grid_row` term by term. The row `dot(c, p) <= alpha * isc` is the condition
    # `sum_j p_j exp((x_j - rhs) / z) <= alpha`, which is the entropic value at risk at that
    # dual variable, and the scaling divides both sides by the same number.
    wv = StatsBase.pweights(LinearAlgebra.normalize(range(0.5, 1.5; length = T), 1))
    for a in (0.01, 0.05, 0.1), j in (1, 5, 20)
        x = -rd.X[:, j]
        e = PO.ep_evar(x, ep_w0, a)
        for m in (0.8, 1.0, 1.25), zm in (0.7, 1.0, 1.4)
            rhs, z = e.evar * m, e.z * zm
            c, isc = PO.ep_evar_grid_row(x, rhs, z)
            row = LinearAlgebra.dot(c, wv) - a * isc
            direct = (sum(wv .* exp.((x .- rhs) ./ z)) - a) * isc
            @test abs(row - direct) <= 1e-13 * a * isc
            # The row's sign is the sign of the entropic value at risk condition it states.
            @test sign(row) == sign(z * (log(sum(wv .* exp.((x .- rhs) ./ z))) - log(a)))
        end
    end

    # The big-M headroom. A row's coefficients sit in `(0, 1]` and the probabilities sum to
    # one, so the largest value the left hand side of a released row takes is one.
    Mneed = 0.0
    for a in (0.01, 0.05, 0.25), j in (1, 5, 20)
        x = -rd.X[:, j]
        e = PO.ep_evar(x, ep_w0, a)
        for m in (0.7, 0.85, 1.0)
            rhs = e.evar * m
            for zk in PO.ep_evar_grid(x, ep_w0, a, :leq, rhs, e.z, 0.5, 11)
                c, isc = PO.ep_evar_grid_row(x, rhs, zk)
                Mneed = max(Mneed, maximum(c) - a * isc)
            end
        end
    end
    @test Mneed <= 1
    @test isapprox(Mneed, 1; rtol = 1e-7)
    @test GridEntropicValueatRiskView().M >= 10 * Mneed

    # The Brent bracket of `ep_evar` holds the minimiser strictly inside, over every asset
    # and four levels. The upper end is a proof, so only the margin below it is a choice.
    for a in (0.01, 0.05, 0.1, 0.25), j in axes(rd.X, 2)
        x = -rd.X[:, j]
        hi = (maximum(x) - LinearAlgebra.dot(collect(ep_w0), x)) / (-log(a))
        lo = hi * sqrt(eps(Float64))
        z = PO.ep_evar(x, ep_w0, a).z
        @test lo < z < hi
        @test z / hi <= 0.6
        @test z / lo >= 1e6
    end

    # `ep_row_tilt` against the root of the tilt written out by hand. With uniform prior
    # weights and `c = [1, 2, 3]`, `u = exp(-theta)` solves `u^2 - u - 3 = 0` at `b = 2.5`.
    u = (1 + sqrt(13)) / 2
    qhand = [u, u^2, u^3] ./ (u + u^2 + u^3)
    q = PO.ep_row_tilt(fill(1 / 3, 3), [1.0, 2.0, 3.0], 2.5)
    @test isapprox(q, qhand, rtol = 1e-15)
    @test LinearAlgebra.dot(q, [1.0, 2.0, 3.0]) == 2.5
    # A target at or outside an end of the range of `c` is attained by no probability
    # vector, including the ends themselves.
    @test isnothing(PO.ep_row_tilt(fill(1 / 3, 3), [1.0, 2.0, 3.0], 1.0))
    @test isnothing(PO.ep_row_tilt(fill(1 / 3, 3), [1.0, 2.0, 3.0], 3.0))

    # The three `ep_sbar` methods against the counts stated by hand. Uniform weights reach
    # `alpha` at `ceil(alpha * T)`, the rule doubles that, and `ceil(2 * alpha * T)` floors
    # it.
    ord = sortperm(-rd.X[:, 1])
    @test PO.ep_sbar(nothing, T, ep_a, ep_w0, ord) ==
          max(2 * ceil(Int, ep_a * T), ceil(Int, 2 * ep_a * T))
    @test PO.ep_sbar(nothing, T, ep_a, ep_w0, ord) == 102
    @test PO.ep_sbar(5, T) == 5
    @test PO.ep_sbar(2 * T, T) == T
    @test PO.ep_sbar(0.1, T) == ceil(Int, 0.1 * T)
    @test PO.ep_sbar(1e-6, T) == 1

    # `ep_assert_reachable_view` is the open band between the best and the worst
    # realisation. Both ends are refused, and everything strictly inside passes.
    x = -rd.X[:, 1]
    lo, hi = extrema(x)
    @test isnothing(PO.ep_assert_reachable_view(:eq, (lo + hi) / 2, x, "e", "EVaR"))
    @test isnothing(PO.ep_assert_reachable_view(:geq, prevfloat(hi), x, "e", "EVaR"))
    @test isnothing(PO.ep_assert_reachable_view(:leq, nextfloat(lo), x, "e", "EVaR"))
    for (op, v) in ((:geq, hi), (:leq, lo), (:eq, hi), (:eq, lo))
        @test_throws DomainError PO.ep_assert_reachable_view(op, v, x, "e", "EVaR")
    end

    # `ep_normalise_view_term` flips the operator on a negative coefficient, and leaves an
    # equality alone. All four cells.
    @test PO.ep_normalise_view_term(2.0, :geq, 0.06) == (:geq, 0.03)
    @test PO.ep_normalise_view_term(-2.0, :geq, -0.06) == (:leq, 0.03)
    @test PO.ep_normalise_view_term(-2.0, :leq, -0.06) == (:geq, 0.03)
    @test PO.ep_normalise_view_term(-2.0, :eq, -0.06) == (:eq, 0.03)
    # A view written with an unnormalised coefficient resolves to the normalised one.
    for (raw, norm) in
        (("2*AAPL >= 0.06", "AAPL >= 0.03"), ("-3*AAPL <= -0.09", "AAPL >= 0.03"),
         ("-2*AAPL >= -0.06", "AAPL <= 0.03"), ("4*AAPL == 0.12", "AAPL == 0.03"))
        t1, t2 = ep_terms(raw, :cvar), ep_terms(norm, :cvar)
        @test t1.idx == t2.idx
        @test PO.ep_normalise_view_term(t1.coef[1], t1.op, t1.rhs) ==
              PO.ep_normalise_view_term(t2.coef[1], t2.op, t2.rhs)
    end

    # `ep_evar_anchor` is a fixed point: the tilted posterior attains the target, and the
    # dual variable it returns makes the row of that grid point tight at that posterior.
    for a in (0.01, 0.05, 0.1), j in (1, 5, 20), m in (0.75, 0.85, 0.95)
        x = -rd.X[:, j]
        e = PO.ep_evar(x, ep_w0, a)
        rhs = e.evar * m
        anc = PO.ep_evar_anchor(x, ep_w0, a, rhs, e.z)
        @test !isnothing(anc)
        post = PO.ep_evar(x, anc.w, a)
        @test abs(post.evar - rhs) <= 1e-10 * abs(rhs)
        @test post.z == anc.z
        c, isc = PO.ep_evar_grid_row(x, rhs, anc.z)
        @test abs(LinearAlgebra.dot(c, anc.w) - a * isc) <= 1e-9 * a * isc
    end
    # The iteration gives up on three paths, and each returns `nothing`. A dual variable
    # small enough to overflow the row is the first: it ends the iteration rather than
    # tilting along a vector of `NaN`s.
    @test isnothing(PO.ep_evar_anchor(-rd.X[:, 1], ep_w0, ep_a, 0.5 * ep_pevar, 1e-8))
    # The second is a row no probability vector makes tight. The row's coefficients are
    # `exp((x_j - max(x)) / z)`, which spans `exp((min(x) - max(x)) / z)` to one, and its
    # target is `alpha * exp((rhs - max(x)) / z)`. So a target at or below
    # `min(x) + z * ln(1 / alpha)` puts the target under the smallest coefficient, and that
    # bound sits **inside** the band `ep_assert_reachable_view` admits: a legitimate view
    # reaches it, and the grid then falls back to the prior's dual variable.
    e1 = PO.ep_evar(-rd.X[:, 1], ep_w0, ep_a)
    lo1 = minimum(-rd.X[:, 1])
    @test lo1 + e1.z * log(inv(ep_a)) > lo1
    @test isnothing(PO.ep_evar_anchor(-rd.X[:, 1], ep_w0, ep_a, 0.98 * lo1, e1.z))
    # The third is `iters` steps that never reach the target, which the testset above pins.

    # `get_pr_value` agrees with the closed form in this file, and reads uniform weights
    # rather than `pr.w`, which is what its `# Arguments` states.
    for a in (0.01, 0.05, 0.1), i in (1, 5, 20)
        @test isapprox(PO.get_pr_value(ep_pr0, i, Val(:evar), a),
                       PO.ep_evar(-rd.X[:, i], ep_w0, a).evar, rtol = 1e-14)
        @test isapprox(PO.get_pr_value(ep_pr0, i, Val(:rlvar), a, ep_k),
                       PO.ep_rlvar(-rd.X[:, i], ep_w0, a, ep_k).rlvar, rtol = 1e-14)
    end

    # `factor_residual_config` forwards the estimator its moments come from.
    @test PO.factor_residual_config(EntropyPoolingPrior(; pe = FactorPrior())) ==
          PO.factor_residual_config(FactorPrior())
    @test isnothing(PO.factor_residual_config(EntropyPoolingPrior()))
end

@testset "each method of add_ep_tail_view! registers the rows its formulation names" begin
    # The `# JuMP formulation` of `add_ep_tail_view!` names every entry each of the six
    # methods registers. The counts below are those lists read off a model.
    counts(tv, Tn) = begin
        m = JuMP.Model()
        pw = JuMP.@variable(m, [1:Tn], lower_bound = 0)
        n0 = JuMP.num_variables(m)
        PO.add_ep_tail_view!(m, pw, tv, 1.0)
        d = Dict(k => JuMP.num_constraints(m, k...)
                 for k in JuMP.list_of_constraint_types(m))
        return JuMP.num_variables(m) - n0, d
    end
    LE = (JuMP.AffExpr, JuMP.MOI.LessThan{Float64})
    EQ = (JuMP.AffExpr, JuMP.MOI.EqualTo{Float64})
    BIN = (JuMP.VariableRef, JuMP.MOI.ZeroOne)
    UB = (JuMP.VariableRef, JuMP.MOI.LessThan{Float64})
    x = -ep_sX[:, 1]
    Ts = length(x)

    # Three rows: `T` bounds on `nu`, the budget, and the target.
    nv, d = counts(PO.LinearConditionalValueatRiskViewConstraint(x, ep_a, 0.05), Ts)
    @test nv == Ts
    @test d[LE] == Ts + 1
    @test d[EQ] == 1

    # Three rows and the relative entropy cone, with `nu` bounded to `[0, 1]`.
    nv, d = counts(PO.ConicEntropicValueatRiskViewConstraint(x, ep_a, 0.05), Ts)
    @test nv == Ts
    @test d[UB] == Ts
    @test d[LE] == 1
    @test d[EQ] == 1
    @test d[(Vector{JuMP.AffExpr}, JuMP.MOI.RelativeEntropyCone)] == 1

    # Five rows per asset over its window, and one row carrying the operator.
    o = sortperm(x)
    sb = PO.ep_sbar(nothing, Ts, ep_a, ep_sw0, o)
    ordw = o[(Ts - sb + 1):Ts]
    nv, d = counts(PO.IntegerConditionalValueatRiskViewConstraint([ordw], [x[ordw]], [1.0],
                                                                  ep_a, :geq, 0.05), Ts)
    @test nv == 2 * sb
    @test d[BIN] == sb
    @test d[LE] == 3 * sb + (sb - 1) + 1
    @test d[EQ] == 1

    # Two rows: the selector, and one row per grid point.
    e = PO.ep_evar(x, ep_sw0, ep_a)
    zg = PO.ep_evar_grid(x, ep_sw0, ep_a, :leq, 0.9 * e.evar, e.z, 0.5, 11)
    nv, d = counts(PO.GridEntropicValueatRiskViewConstraint(x, zg, ep_a, 0.9 * e.evar,
                                                            10.0), Ts)
    @test nv == 11
    @test d[BIN] == 11
    @test d[LE] == 11
    @test d[EQ] == 1

    # Five rows, two of them a power cone per observation, and three auxiliary vectors.
    nv, d = counts(PO.ConicRelativisticValueatRiskViewConstraint(x, ep_a, ep_k, 0.05), Ts)
    @test nv == 3 * Ts
    @test d[UB] == Ts
    @test d[LE] == 2
    @test d[EQ] == 1
    @test d[(Vector{JuMP.AffExpr}, JuMP.MOI.PowerCone{Float64})] == 2 * Ts
end

@testset "two tail views of different families are met on one posterior" begin
    # Each tail view is a constraint of the single entropy pooling problem, so a mandate
    # carrying two of them is one solve, and both targets hold on its answer.
    tc = ep_spcvar * 1.12
    te = ep_sevar(5, ep_sw0) * 1.12
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL >= $tc")),
                                   evar_views = EntropicValueatRiskView(;
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "$(rd.nx[5]) >= $te"))),
               ep_srd)
    @test isapprox(ep_scvar(1, pr.w), tc, rtol = 1e-6)
    @test isapprox(ep_sevar(5, pr.w), te, rtol = 1e-6)
    @test isapprox(sum(pr.w), 1)
    @test pr.kld > 0
    # The two views together cost more divergence than either alone.
    one_c = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                      cvar_views = ConditionalValueatRiskView(;
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL >= $tc"))),
                  ep_srd)
    one_e = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                      evar_views = EntropicValueatRiskView(;
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "$(rd.nx[5]) >= $te"))),
                  ep_srd)
    @test pr.kld > one_c.kld
    @test pr.kld > one_e.kld
end
