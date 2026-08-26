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
# Resolve one view equation the way `ep_cvar_views!` does, without solving anything.
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
    lnk = PO.ep_kappa_log(inv(ep_a * T), ep_k)
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
