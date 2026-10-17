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

@testset "grid EVaR expresses the operators the conic formulation cannot" begin
    tgt = ep_spevar * 0.9
    # The upper bound reaches the outer approximation of Pajarito. On the developer host it
    # finds its incumbent at the second iteration, then gets the same integral solution back
    # while its bound wanders about 1.5% below, and it ran 48 minutes without closing the
    # default gap of 1e-5 (#1635). The test claims feasibility alone: a selector row holds at
    # one grid point, so every feasible posterior meets the bound. A gap of 5% stops the solve
    # at the third iteration, and the claim still holds.
    gap_mopt = JuMPEntropyPooling(;
                                  slv = Solver(; name = :ep_mip_gap,
                                               solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                                                  "verbose" =>
                                                                                      false,
                                                                                  "tol_rel_gap" =>
                                                                                      0.05,
                                                                                  "oa_solver" =>
                                                                                      optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                                JuMP.MOI.Silent() =>
                                                                                                                    true),
                                                                                  "conic_solver" =>
                                                                                      optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                                "verbose" =>
                                                                                                                    false)),
                                               check_sol = (; allow_local = true,
                                                            allow_almost = true)))
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = gap_mopt,
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

@testset "a grid row reads against one, so a solver resolves it at every point" begin
    # Issue #1264. A grid row is scaled so its largest coefficient is one, and at a small
    # dual variable its bound then falls to 1e-11, far inside the feasibility tolerance of
    # the solver. The solver took that row as met by any posterior, the selector picked it,
    # and the posterior missed an upper bound at half the prior EVaR by 60%. Every row now
    # reaches the model divided by its bound. The pinned divergence is the minimum of a scan
    # over the dual variable of one-row tilts, which is where the anchor centres the grid.
    rng = StableRNG(1254)
    gT = 100
    X = 0.01 .* randn(rng, gT, 3) .+ 0.0005
    X[:, 1] .-= 0.02 .* (rand(rng, gT) .< 0.05)
    grd = ReturnsResult(; nx = ["A", "B", "C"], X = X)
    gsets = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"]))
    gw0 = StatsBase.pweights(fill(inv(gT), gT))
    x = -X[:, 1]
    e0 = PO.ep_evar(x, gw0, ep_a)
    r0 = PO.ep_rlvar(x, gw0, ep_a, ep_k)
    etgt, rtgt = 0.5 * e0.evar, 0.5 * r0.rlvar

    # The grid reaches the bounds that the solver could not resolve at their own scale.
    tvs = Any[]
    PO.ep_add_evar_view!(Dict{Any, Any}(), tvs, GridEntropicValueatRiskView(), [x], [1.0],
                         ep_a, :leq, etgt, gw0, [e0.z], e0.evar, "A <= $etgt")
    tv = only(tvs)
    @test minimum(ep_a * PO.ep_evar_grid_row(x, etgt, z)[2] for z in tv.z) < 1e-10

    # In the model, row `k` reads `(c / b)' p + M * M_k * y_k <= 1 + M * M_k`, with `M_k` the
    # largest scaled coefficient less one: the smallest constant that releases the row.
    rows = (M -> begin
                tvs = Any[]
                PO.ep_add_evar_view!(Dict{Any, Any}(), tvs,
                                     GridEntropicValueatRiskView(; M = M), [x], [1.0], ep_a,
                                     :leq, etgt, gw0, [e0.z], e0.evar, "A <= $etgt")
                tv = only(tvs)
                return tv, k -> begin
                           c, isc = PO.ep_evar_grid_row(x, etgt, tv.z[k])
                           return c ./ (ep_a * isc)
                       end
            end,
            M -> begin
                tvs = Any[]
                PO.ep_add_rlvar_view!(Dict{Any, Any}(), tvs,
                                      GridRelativisticValueatRiskView(; M = M), [x], [1.0],
                                      ep_a, ep_k, :leq, rtgt, gw0, [r0.z], r0.rlvar,
                                      "A <= $rtgt")
                tv = only(tvs)
                return tv,
                       k -> begin
                           c, b = PO.ep_rlvar_grid_row(x, rtgt, tv.t[k], tv.z[k], ep_a,
                                                       ep_k)
                           return c ./ b
                       end
            end)
    for mk in rows, M in (1, 2.5)
        tv, scaled = mk(M)
        m = JuMP.Model()
        pw = JuMP.@variable(m, [1:gT], lower_bound = 0)
        PO.add_ep_tail_view!(m, pw, tv, 1.0)
        ys = setdiff(JuMP.all_variables(m), pw)
        cons = JuMP.all_constraints(m, JuMP.AffExpr, JuMP.MOI.LessThan{Float64})
        @test length(cons) == length(tv.z) == length(ys)
        for (k, con) in pairs(cons)
            a = JuMP.normalized_coefficient.(Ref(con), pw)
            yc = sum(JuMP.normalized_coefficient(con, y) for y in ys)
            @test isapprox(a, scaled(k); rtol = 1e-12)
            @test isapprox(yc, M * (maximum(a) - 1); rtol = 1e-12)
            @test isapprox(JuMP.normalized_rhs(con), 1 + yc; rtol = 1e-12)
        end
    end

    # The reproduction of the issue. The upper bound is met, and at the posterior of least
    # divergence rather than the one the tolerance let through.
    fit = (key, v) -> prior(EntropyPoolingPrior(; sets = gsets, opt = ep_mopt,
                                                (key => v,)...), grd)
    pr = fit(:evar_views,
             EntropicValueatRiskView(; alpha = ep_a, alg = GridEntropicValueatRiskView(),
                                     views = LinearConstraintEstimator(;
                                                                       val = "A <= $etgt")))
    @test PO.ep_evar(x, pr.w, ep_a).evar <= etgt * (1 + 1e-6)
    @test isapprox(PO.ep_evar(x, pr.w, ep_a).evar, etgt; rtol = 1e-6)
    @test isapprox(pr.kld, 0.088217; rtol = 1e-4)
    # The relativistic grid met this view before the fix, with its smallest bound at 1e-7.
    # It stays met.
    pr = fit(:rlvar_views,
             RelativisticValueatRiskView(; alpha = ep_a, kappa = ep_k,
                                         alg = GridRelativisticValueatRiskView(),
                                         views = LinearConstraintEstimator(;
                                                                           val = "A <= $rtgt")))
    @test PO.ep_rlvar(x, pr.w, ep_a, ep_k).rlvar <= rtgt * (1 + 1e-5)
    @test isapprox(PO.ep_rlvar(x, pr.w, ep_a, ep_k).rlvar, rtgt; rtol = 1e-5)
end
