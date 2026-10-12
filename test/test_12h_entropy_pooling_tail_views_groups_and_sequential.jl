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

@testset "an upper-bound grid point whose bound is at or below zero is dropped" begin
    # Issue #1264. A grid row has positive coefficients and the posterior sums to one, so a
    # row whose bound is at or below zero holds at no posterior as an upper bound, and the
    # upper-bound half drops it. As a lower bound it holds at every posterior, so the
    # lower-bound half keeps it: it costs nothing, and dropping it would only add a raise.
    row = g -> ([1.0, 0.5], g)
    grid = [0.2, -0.1, 0.0, 0.4]
    for (op, kept, nrows) in ((:leq, [0.2, 0.4], 0), (:geq, grid, 4), (:eq, [0.2, 0.4], 4))
        epc = Dict{Any, Any}()
        @test PO.ep_add_grid_tail_view!(epc, grid, op, row, () -> "none") == kept
        @test (haskey(epc, :ineq) ? size(epc[:ineq][1], 1) : 0) == nrows
    end
    # The lower-bound rows keep the norm scale every other entropy pooling row takes.
    epc = Dict{Any, Any}()
    PO.ep_add_grid_tail_view!(epc, [0.4], :geq, row, () -> "none")
    @test isapprox(epc[:ineq][1], -[1.0 0.5] / norm([1.0, 0.5]))
    @test isapprox(epc[:ineq][2], [-0.4 / norm([1.0, 0.5])])
    # Only a half that keeps no point raises.
    for op in (:leq, :eq)
        @test_throws ArgumentError("none") PO.ep_add_grid_tail_view!(Dict{Any, Any}(),
                                                                     [-0.1, 0.0], op, row,
                                                                     () -> "none")
    end
    epc = Dict{Any, Any}()
    @test PO.ep_add_grid_tail_view!(epc, [-0.1, 0.0], :geq, row, () -> "none") ==
          [-0.1, 0.0]
    @test size(epc[:ineq][1], 1) == 2
end

@testset "CVaR views on a group" begin
    # A group view is the sum of the members' CVaRs, so it names two assets. A positive
    # combination of CVaRs is concave in the posterior probabilities, so an equality at or
    # above the prior takes the linear formulation and needs no integer variable.
    # `prior(gA)` is the sum of their prior CVaRs, which is what the target is stated
    # against.
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
    # `(0, 1]` and the row does not overflow before it is built. Issue #1264: the row then
    # reaches the model divided by its bound, and the constant that releases it is the
    # largest scaled coefficient less one, which the data fix row by row. The default `M`
    # multiplies that constant by one.
    dflt = GridRelativisticValueatRiskView()
    @test dflt.pct == 0.5
    @test dflt.K == 11
    @test dflt.M == 1
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
        PO.ep_add_rlvar_view!(epc, tvs, GridRelativisticValueatRiskView(), [x], [1.0], ep_a,
                              kappa, op, m * r.rlvar, ep_sw0, [r.z], r.rlvar,
                              "AAPL <= $(m * r.rlvar)")
        return epc, tvs
    end
    # The grid of each half, built again, and whether the row of each of its points is finite.
    # The two halves centre their grids on different points: the upper bound on the point the
    # posterior attains, the lower bound on the dual variable of the prior. So they need not
    # keep the same points. Where a point overflows sits on a knife edge that moves with the
    # host (#1635), so each half is checked against its own grid, never against a count.
    grid(kappa, m, op) = begin
        r = PO.ep_rlvar(x, ep_sw0, ep_a, kappa)
        alg = GridRelativisticValueatRiskView()
        t, z = PO.ep_rlvar_grid(x, ep_sw0 ./ sum(ep_sw0), ep_a, kappa, op, m * r.rlvar, r.z,
                                r.rlvar, alg.pct, alg.K; iters = alg.iters, tol = alg.tol,
                                tilt_iters = alg.tilt_iters)
        rows = [PO.ep_rlvar_grid_row(x, m * r.rlvar, t[k], z[k], ep_a, kappa)
                for k in eachindex(t)]
        return t, z, [all(isfinite, c) && isfinite(b) for (c, b) in rows], last.(rows)
    end
    # Nothing overflows here, so the grid keeps every point it built.
    _, tvs = add(0.7, 0.75, :leq)
    @test length(only(tvs).z) == GridRelativisticValueatRiskView().K
    # The upper-bound half keeps exactly the points of its grid whose row is finite and whose
    # bound is positive, and each kept row is finite.
    _, tvs = add(0.9, 0.75, :leq)
    tv = only(tvs)
    t, z, fin, bd = grid(0.9, 0.75, :leq)
    keep = findall(k -> fin[k] && bd[k] > 0, eachindex(z))
    @test tv.z == z[keep]
    @test tv.t == t[keep]
    for k in eachindex(tv.z)
        c, b = PO.ep_rlvar_grid_row(x, tv.rhs, tv.t[k], tv.z[k], ep_a, 0.9)
        @test all(isfinite, c)
        @test isfinite(b)
    end
    # The lower-bound half sits at a dual variable near zero, where some points overflow on
    # every host measured. It writes the rows of the points it keeps straight into the
    # constraint set, so it writes one row per finite point of its grid.
    epc, _ = add(0.9, 0.75, :geq)
    lhs, rhs = epc[:ineq]
    _, _, fin, _ = grid(0.9, 0.75, :geq)
    @test 0 < count(fin) < GridRelativisticValueatRiskView().K
    @test size(lhs, 1) == length(rhs) == count(fin)
    @test all(isfinite, lhs)
    @test all(isfinite, rhs)
    # Every point overflows here, and the raise names `kappa` rather than leaving JuMP to
    # refuse a `NaN` coefficient.
    @test_throws ArgumentError add(0.9, 0.5, :leq)
end

@testset "the grid guard drops a row that is not finite, and an upper bound at or below zero" begin
    # The guard on rows that are not finite by construction, so no knife edge decides which
    # point it drops (#1635). Point 2 has an infinite coefficient, point 3 a `NaN` bound, and
    # point 4 a bound below zero, which only an upper bound refuses.
    row(g) = (g == 2 ? [1.0, Inf] : [1.0, float(g)], g == 3 ? NaN : g == 4 ? -1.0 : 1.0)
    msg = () -> "no usable grid point"
    epc = Dict{Any, Any}()
    @test PO.ep_add_grid_tail_view!(epc, collect(1:5), :leq, row, msg) == [1, 5]
    @test !haskey(epc, :ineq)
    epc = Dict{Any, Any}()
    @test PO.ep_add_grid_tail_view!(epc, collect(1:5), :geq, row, msg) == [1, 4, 5]
    # The rows of points 1, 4 and 5, negated into the `<=` sense, each scaled to unit norm.
    lhs, rhs = epc[:ineq]
    @test size(lhs) == (3, 2)
    @test all(<(0), lhs[:, 1])
    @test lhs[:, 2] ./ lhs[:, 1] ≈ [1.0, 4.0, 5.0]
    @test rhs ./ lhs[:, 1] ≈ [1.0, -1.0, 1.0]
    epc = Dict{Any, Any}()
    @test PO.ep_add_grid_tail_view!(epc, collect(1:5), :eq, row, msg) == [1, 5]
    @test size(epc[:ineq][1], 1) == 3
    @test_throws ArgumentError(msg()) PO.ep_add_grid_tail_view!(Dict{Any, Any}(), [2, 3],
                                                                :geq, row, msg)
    @test_throws ArgumentError(msg()) PO.ep_add_grid_tail_view!(Dict{Any, Any}(), [2, 3, 4],
                                                                :leq, row, msg)
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
        PO.ep_add_evar_view!(epc, tvs, alg, [x], [1.0], ep_a, :leq, tgt, ep_w0, [e.z],
                             e.evar, "AAPL")
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
        PO.ep_add_evar_view!(epc, tvs, GridEntropicValueatRiskView(; pct = pct), [x], [1.0],
                             a, op, 0.7 * ea.evar, ep_sw0, [ea.z], ea.evar,
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
    PO.ep_add_evar_view!(epc, tvs, GridEntropicValueatRiskView(; K = 1), [x], [1.0], ep_a,
                         :leq, tgt, ep_w0, [e.z], e.evar, "AAPL <= $tgt")
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
    @test isnothing(PO.ep_assert_reachable_view(:eq, (lo + hi) / 2, [x], [1.0], ep_w0, "e",
                                                "EVaR"))
    @test isnothing(PO.ep_assert_reachable_view(:geq, prevfloat(hi), [x], [1.0], ep_w0, "e",
                                                "EVaR"))
    @test isnothing(PO.ep_assert_reachable_view(:leq, nextfloat(lo), [x], [1.0], ep_w0, "e",
                                                "EVaR"))
    for (op, v) in ((:geq, hi), (:leq, lo), (:eq, hi), (:eq, lo))
        @test_throws DomainError PO.ep_assert_reachable_view(op, v, [x], [1.0], ep_w0, "e",
                                                             "EVaR")
    end
    # Over several assets the band is the coefficient-weighted sum of the per-asset ends,
    # with the ends exchanged where the coefficient is negative.
    y = -rd.X[:, 2]
    lo2, hi2 = extrema(y)
    hib = hi - 2 * lo2
    lob = lo - 2 * hi2
    @test isnothing(PO.ep_assert_reachable_view(:eq, (lob + hib) / 2, [x, y], [1.0, -2.0],
                                                ep_w0, "e", "CVaR"))
    @test_throws DomainError PO.ep_assert_reachable_view(:geq, hib, [x, y], [1.0, -2.0],
                                                         ep_w0, "e", "CVaR")
    @test_throws DomainError PO.ep_assert_reachable_view(:leq, lob, [x, y], [1.0, -2.0],
                                                         ep_w0, "e", "CVaR")

    # A posterior puts no mass where the prior puts none, so an observation of zero prior
    # probability widens neither band. Issue #1260: with the worst loss at zero prior, a
    # target between the worst loss on the support and the worst loss overall passed the
    # guard, and the tilt doubled until every `exp` underflowed and returned all `NaN`.
    wz = [0.25, 0.25, 0.5, 0.0]
    cz = [1.0, 2.0, 3.0, 10.0]
    @test isnothing(PO.ep_row_tilt(wz, cz, 5.0))
    @test isnothing(PO.ep_row_tilt(wz, cz, 3.0))
    qz = PO.ep_row_tilt(wz, cz, 2.9)
    @test all(isfinite, qz)
    @test iszero(qz[4])
    @test LinearAlgebra.dot(qz, cz) ≈ 2.9
    @test_throws DomainError PO.ep_assert_reachable_view(:geq, 5.0, [cz], [1.0], wz, "e",
                                                         "EVaR")
    @test isnothing(PO.ep_assert_reachable_view(:geq, 2.9, [cz], [1.0], wz, "e", "EVaR"))

    # `ep_check_tail_window` warns where the window of an integer CVaR carrier binds, which
    # is where it holds no more than `alpha`, and never for a window of the whole sample.
    tvz = PO.IntegerConditionalValueatRiskViewConstraint([[3, 4, 5]], [[1.0, 2.0, 3.0]],
                                                         [1.0], 0.1, :leq, 2.0)
    @test_logs (:warn, r"window binds") PO.ep_check_tail_window(tvz,
                                                                [0.45, 0.45, 0.0, 0.05,
                                                                 0.05])
    @test_logs PO.ep_check_tail_window(tvz, [0.3, 0.3, 0.2, 0.1, 0.1])
    tvw = PO.IntegerConditionalValueatRiskViewConstraint([[1, 2]], [[1.0, 2.0]], [1.0], 0.1,
                                                         :leq, 2.0)
    @test_logs PO.ep_check_tail_window(tvw, [0.95, 0.05])
    @test isnothing(PO.ep_check_tail_window(PO.LinearConditionalValueatRiskViewConstraint([[1.0,
                                                                                            2.0]],
                                                                                          [1.0],
                                                                                          0.1,
                                                                                          2.0),
                                            [0.5, 0.5]))

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
    nv, d = counts(PO.LinearConditionalValueatRiskViewConstraint([x], [1.0], ep_a, 0.05),
                   Ts)
    @test nv == Ts
    @test d[LE] == Ts + 1
    @test d[EQ] == 1

    # Three rows and the relative entropy cone, with `nu` bounded to `[0, 1]`.
    nv, d = counts(PO.ConicEntropicValueatRiskViewConstraint([x], [1.0], ep_a, 0.05), Ts)
    @test nv == Ts
    @test d[UB] == Ts
    @test d[LE] == 1
    @test d[EQ] == 1
    @test d[(Vector{JuMP.AffExpr}, JuMP.MOI.RelativeEntropyCone)] == 1

    # Five rows per asset over its window, and one row carrying the operator. The row that
    # puts an observation in the tail in full reads the observation below it, so the lowest
    # observation of the window has none. Issue #1260.
    o = sortperm(x)
    sb = PO.ep_sbar(nothing, Ts, ep_a, ep_sw0, o)
    ordw = o[(Ts - sb + 1):Ts]
    nv, d = counts(PO.IntegerConditionalValueatRiskViewConstraint([ordw], [x[ordw]], [1.0],
                                                                  ep_a, :geq, 0.05), Ts)
    @test nv == 2 * sb
    @test d[BIN] == sb
    @test d[LE] == 3 * sb - 1 + (sb - 1) + 1
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
    nv, d = counts(PO.ConicRelativisticValueatRiskViewConstraint([x], [1.0], ep_a, ep_k,
                                                                 0.05), Ts)
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

# --------------------------------------------------------------------------------------
# The sequential convex formulations. Each measure is concave in the posterior
# probabilities, so a view whose coefficients carry both signs, an upper bound, and an
# equality below the prior have no convex feasible set. The sequential formulation bounds
# every measure on the wrong side of the inequality by a row that is linear in the
# probabilities, solves the convex problem that results, and re-solves with the row re-read
# at the posterior until it is tight. Every posterior of the sequence meets the view.
# --------------------------------------------------------------------------------------
@testset "sequential formulation constructors" begin
    for F in (SequentialConditionalValueatRiskView, SequentialEntropicValueatRiskView,
              SequentialRelativisticValueatRiskView)
        f = F()
        @test f.iters == 20
        @test f.tol == 1e-8
        @test F(; iters = 0, tol = 1e-4).iters == 0
        @test_throws DomainError F(; iters = -1)
        @test_throws DomainError F(; tol = 0.0)
    end
    @test isa(SequentialConditionalValueatRiskView(),
              PO.AbstractConditionalValueatRiskViewFormulation)
    @test isa(SequentialEntropicValueatRiskView(),
              PO.AbstractEntropicValueatRiskViewFormulation)
    @test isa(SequentialRelativisticValueatRiskView(),
              PO.AbstractRelativisticValueatRiskViewFormulation)
end

@testset "ep_sequential_sides orients the view as a lower bound and splits its assets" begin
    x1, x2 = -ep_sX[:, 1], -ep_sX[:, 2]
    # A lower bound is kept, and the sides follow the signs.
    xd, cd, xp, cp, r = PO.ep_sequential_sides([x1, x2], [1.0, -2.0], :geq, 0.01, 0.0)
    @test xd == [x1] && cd == [1.0] && xp == [x2] && cp == [-2.0] && r == 0.01
    # An upper bound is negated on both sides, which exchanges the sides.
    xd, cd, xp, cp, r = PO.ep_sequential_sides([x1, x2], [1.0, -2.0], :leq, 0.01, 0.0)
    @test xd == [x2] && cd == [2.0] && xp == [x1] && cp == [-1.0] && r == -0.01
    # An equality is written as the bound the prior violates.
    xd, cd, xp, cp, r = PO.ep_sequential_sides([x1], [1.0], :eq, 0.05, 0.04)
    @test xd == [x1] && isempty(xp) && r == 0.05
    xd, cd, xp, cp, r = PO.ep_sequential_sides([x1], [1.0], :eq, 0.03, 0.04)
    @test isempty(xd) && xp == [x1] && cp == [-1.0] && r == -0.03
end

@testset "the surrogate row is tight where it is read and bounds the measure elsewhere" begin
    x = -ep_sX[:, 1]
    T = length(x)
    rng = StableRNG(350)
    w0 = ep_sw0
    w1 = StatsBase.pweights(normalize(rand(rng, T) .+ 0.2, 1))
    w2 = StatsBase.pweights(normalize(rand(rng, T) .^ 3 .+ 0.05, 1))
    # The CVaR multiplier is the minimiser of the Rockafellar-Uryasev function, which no
    # other loss beats, and the value there is the risk measure.
    ru(eta, w) = eta + LinearAlgebra.dot(w, max.(x .- eta, 0)) / ep_a
    for w in (w0, w1, w2)
        eta = PO.ep_var_multiplier(x, w, ep_a)
        @test eta in x
        @test all(ru(eta, w) <= ru(e, w) + 1e-14 for e in x)
        @test isapprox(ru(eta, w), ConditionalValueatRisk(; alpha = ep_a, w = w)(-x),
                       rtol = 1e-10)
    end
    tvc = PO.SequentialConditionalValueatRiskViewConstraint([], Float64[], [x], [-1.0],
                                                            zeros(T), 0.0, ep_a, 0.0, 20,
                                                            1e-8)
    tve = PO.SequentialEntropicValueatRiskViewConstraint([], Float64[], [x], [-1.0],
                                                         zeros(T), 0.0, ep_a, 0.0, 20, 1e-8,
                                                         (), (;), nothing)
    tvr = PO.SequentialRelativisticValueatRiskViewConstraint([], Float64[], [x], [-1.0],
                                                             zeros(T), 0.0, ep_a, ep_k, 0.0,
                                                             20, 1e-8, (), (;), nothing)
    measure = (tv, w) -> if tv isa PO.SequentialConditionalValueatRiskViewConstraint
        ConditionalValueatRisk(; alpha = ep_a, w = w)(-x)
    elseif tv isa PO.SequentialEntropicValueatRiskViewConstraint
        PO.ep_evar(x, w, ep_a).evar
    else
        PO.ep_rlvar(x, w, ep_a, ep_k).rlvar
    end
    for tv in (tvc, tve, tvr), w in (w0, w1, w2)
        r, r0 = PO.ep_tail_surrogate_row(tv, x, w)
        @test isapprox(r0 + LinearAlgebra.dot(r, w), measure(tv, w), rtol = 1e-8)
        for v in (w0, w1, w2)
            @test r0 + LinearAlgebra.dot(r, v) >= measure(tv, v) * (1 - 1e-8)
        end
    end
    # A carrier with an empty primal side has nothing to re-read, and a fixed carrier
    # asks for no re-solve.
    tv0 = PO.SequentialConditionalValueatRiskViewConstraint([x], [1.0], [], Float64[],
                                                            zeros(T), 0.0, ep_a, 0.0, 20,
                                                            1e-8)
    @test PO.ep_refine_tail_view(tv0, w1) === (tv0, true)
    @test PO.ep_refine_iters(tv0) == 20
    lc = PO.LinearConditionalValueatRiskViewConstraint([x], [1.0], ep_a, 0.05)
    @test PO.ep_refine_iters(lc) == 0
    @test PO.ep_refine_tail_view(lc, w1) === (lc, true)
    @test PO.ep_refine_iters(PO.AbstractEntropyPoolingTailView[lc, tv0]) == 20
    @test PO.ep_refine_iters(PO.AbstractEntropyPoolingTailView[]) == 0
    # A re-read at the probabilities the row was read at reports it tight, and a re-read
    # elsewhere does not. The EVaR row is the one to read elsewhere: a CVaR row is tight
    # wherever the value at risk lands on the same loss.
    tvc1, _ = PO.ep_refine_tail_view(tvc, w1)
    @test PO.ep_refine_tail_view(tvc1, w1)[2]
    tve1, _ = PO.ep_refine_tail_view(tve, w1)
    @test PO.ep_refine_tail_view(tve1, w1)[2]
    @test !PO.ep_refine_tail_view(tve1, w2)[2]
end

@testset "the re-solves never raise the divergence and stop at a tight row" begin
    tgt = (ep_spevar - ep_sevar(size(ep_sX, 2), ep_sw0)) + 0.01
    v = LinearConstraintEstimator(; val = "AAPL - $(rd.nx[end]) >= $tgt")
    pr0 = prior(EmpiricalPrior(), ep_srd)
    epc = Dict{Symbol, Tuple{<:AbstractMatrix, <:AbstractVector}}()
    tvs = PO.AbstractEntropyPoolingTailView[]
    PO.ep_tail_views!(EntropicValueatRiskView(; views = v), epc, tvs, pr0, sets, ep_sw0)
    @test only(tvs) isa PO.SequentialEntropicValueatRiskViewConstraint
    klds = Float64[]
    gaps = Float64[]
    w1 = PO.ep_jump_entropy_pooling(ep_sw0, epc, tvs, ep_jopt)
    for _ in 1:8
        push!(klds, StatsBase.kldivergence(w1, ep_sw0))
        tv = only(tvs)
        tv2, tight = PO.ep_refine_tail_view(tv, w1)
        push!(gaps,
              abs((LinearAlgebra.dot(tv2.c, w1) + tv2.b) -
                  (LinearAlgebra.dot(tv.c, w1) + tv.b)))
        tight && break
        tvs = PO.AbstractEntropyPoolingTailView[tv2]
        w1 = PO.ep_jump_entropy_pooling(ep_sw0, epc, tvs, ep_jopt)
    end
    @test length(klds) < 8
    @test all(diff(klds) .<= 1e-9)
    @test all(diff(gaps) .< 0)
    @test gaps[end] <= 1e-8 * max(abs(tgt), maximum(abs, -ep_sX[:, 1]))
    # The library loop lands where the trace does, and with no re-solve it keeps the
    # first posterior, on which the view holds with room to spare.
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   evar_views = EntropicValueatRiskView(; views = v)),
               ep_srd)
    @test isapprox(pr.kld, klds[end], rtol = 1e-6)
    @test isapprox(ep_sevar(1, pr.w) - ep_sevar(size(ep_sX, 2), pr.w), tgt, rtol = 1e-5)
    pr0_ = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                     evar_views = EntropicValueatRiskView(;
                                                                          alg = SequentialEntropicValueatRiskView(;
                                                                                                                  iters = 0),
                                                                          views = v)),
                 ep_srd)
    @test isapprox(pr0_.kld, klds[1], rtol = 1e-6)
    @test ep_sevar(1, pr0_.w) - ep_sevar(size(ep_sX, 2), pr0_.w) > tgt * (1 + 1e-2)
    @test pr0_.kld > pr.kld
end

@testset "sequential CVaR agrees with the integer formulation" begin
    # An upper bound on one asset. Both land on the target; the sequential answer is a
    # local minimiser of the divergence and the integer one is an outer approximation over
    # a window, so the two divergences agree loosely.
    tgt = ep_spcvar * 0.85
    v = LinearConstraintEstimator(; val = "AAPL <= $tgt")
    ps = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           alg = SequentialConditionalValueatRiskView(),
                                                                           views = v)),
               ep_srd)
    pi_ = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                    cvar_views = ConditionalValueatRiskView(; views = v)),
                ep_srd)
    @test isapprox(ep_scvar(1, ps.w), tgt, rtol = 1e-6)
    @test isapprox(ep_scvar(1, pi_.w), tgt, rtol = 1e-3)
    @test isapprox(ps.kld, pi_.kld, rtol = 2e-2)
    # A relative view stated as an equality.
    tgt2 = (ep_spcvar - ep_spcvarN) + 0.01
    nN = rd.nx[size(ep_sX, 2)]
    v2 = LinearConstraintEstimator(; val = "AAPL - $nN == $tgt2")
    ps2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    cvar_views = ConditionalValueatRiskView(;
                                                                            alg = SequentialConditionalValueatRiskView(),
                                                                            views = v2)),
                ep_srd)
    pi2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_mopt,
                                    cvar_views = ConditionalValueatRiskView(; views = v2)),
                ep_srd)
    @test isapprox(ep_scvar(1, ps2.w) - ep_scvar(size(ep_sX, 2), ps2.w), tgt2, rtol = 1e-5)
    @test isapprox(ep_scvar(1, pi2.w) - ep_scvar(size(ep_sX, 2), pi2.w), tgt2, rtol = 5e-3)
    @test isapprox(ps2.kld, pi2.kld, rtol = 5e-2)
    # A convex view under the sequential formulation is the linear one with no re-solve.
    v3 = LinearConstraintEstimator(; val = "AAPL >= $(ep_spcvar * 1.2)")
    ps3 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    cvar_views = ConditionalValueatRiskView(;
                                                                            alg = SequentialConditionalValueatRiskView(),
                                                                            views = v3)),
                ep_srd)
    pl3 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    cvar_views = ConditionalValueatRiskView(; views = v3)),
                ep_srd)
    @test isapprox(ps3.w, pl3.w, rtol = 1e-6, atol = 1e-8)
end

@testset "relative EVaR and RLVaR views are met with no integer variable" begin
    N = size(ep_sX, 2)
    nN = rd.nx[N]
    # EVaR. `nothing` picks the sequential formulation, which is the one that expresses a
    # view whose coefficients carry both signs.
    tgt = (ep_spevar - ep_sevar(N, ep_sw0)) + 0.01
    pe = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   evar_views = EntropicValueatRiskView(;
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "AAPL - $nN >= $tgt"))),
               ep_srd)
    @test isapprox(ep_sevar(1, pe.w) - ep_sevar(N, pe.w), tgt, rtol = 1e-5)
    # The other direction moves the other asset, and an equality below the prior is met.
    tgt2 = (ep_spevar - ep_sevar(N, ep_sw0)) - 0.01
    pe2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    evar_views = EntropicValueatRiskView(;
                                                                         views = LinearConstraintEstimator(;
                                                                                                           val = "AAPL - $nN == $tgt2"))),
                ep_srd)
    @test isapprox(ep_sevar(1, pe2.w) - ep_sevar(N, pe2.w), tgt2, rtol = 1e-5)
    # RLVaR.
    tgt3 = (ep_sprlvar - ep_srlvar(N, ep_sw0)) + 0.01
    pr_ = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              kappa = ep_k,
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL - $nN >= $tgt3"))),
                ep_srd)
    # The target sits near zero, so the tolerance is read against the prior RLVaR.
    @test isapprox(ep_srlvar(1, pr_.w) - ep_srlvar(N, pr_.w), tgt3,
                   atol = 2e-4 * ep_sprlvar)
    # An upper bound on one asset under the sequential formulation lands on the target
    # exactly, where the grid holds it at a grid point and is conservative.
    tgt4 = ep_spevar * 0.9
    pu = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   evar_views = EntropicValueatRiskView(;
                                                                        alg = SequentialEntropicValueatRiskView(),
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "AAPL <= $tgt4"))),
               ep_srd)
    @test isapprox(ep_sevar(1, pu.w), tgt4, rtol = 1e-5)
    tgt5 = ep_sprlvar * 0.9
    pu2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              kappa = ep_k,
                                                                              alg = SequentialRelativisticValueatRiskView(),
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "AAPL <= $tgt5"))),
                ep_srd)
    @test isapprox(ep_srlvar(1, pu2.w), tgt5, rtol = 1e-4)
    # A group RLVaR lower bound is a positive combination, written by the conic
    # formulation over two assets.
    pAr = ep_srlvar(1, ep_sw0) + ep_srlvar(2, ep_sw0)
    pg = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                   rlvar_views = RelativisticValueatRiskView(; kappa = ep_k,
                                                                             views = LinearConstraintEstimator(;
                                                                                                               val = "gA >= $(1.03 * pAr)"))),
               ep_srd)
    @test isapprox(ep_srlvar(1, pg.w) + ep_srlvar(2, pg.w), 1.03 * pAr, rtol = 5e-4)
    # The sequential carriers need the JuMP route, like every tail view.
    @test_throws ArgumentError prior(EntropyPoolingPrior(; sets = sets,
                                                         opt = OptimEntropyPooling(),
                                                         evar_views = EntropicValueatRiskView(;
                                                                                              views = LinearConstraintEstimator(;
                                                                                                                                val = "AAPL - $nN >= $tgt"))),
                                     ep_srd)
end

@testset "a group upper bound with the default formulation is met" begin
    # Issue #1287. `nothing` sent an upper bound over several assets of one sign to the grid,
    # which is one asset's, and the grid refused the view. The EVaR and RLVaR views now take
    # the sequential formulation. The CVaR views take the integer one, whose window is per
    # asset.
    pAe = ep_sevar(1, ep_sw0) + ep_sevar(2, ep_sw0)
    pAr = ep_srlvar(1, ep_sw0) + ep_srlvar(2, ep_sw0)
    pAc = ep_scvar(1, ep_sw0) + ep_scvar(2, ep_sw0)
    for (op, f) in (("<=", 0.95), ("==", 0.97))
        v = LinearConstraintEstimator(; val = "gA $op $(f)*prior(gA)")
        pe = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                       evar_views = EntropicValueatRiskView(; views = v)),
                   ep_srd)
        @test isapprox(ep_sevar(1, pe.w) + ep_sevar(2, pe.w), f * pAe, rtol = 1e-5)
        pr_ = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_jopt,
                                        rlvar_views = RelativisticValueatRiskView(;
                                                                                  kappa = ep_k,
                                                                                  views = v)),
                    ep_srd)
        @test isapprox(ep_srlvar(1, pr_.w) + ep_srlvar(2, pr_.w), f * pAr, rtol = 1e-4)
        pc = prior(EntropyPoolingPrior(; sets = ep_gsets, opt = ep_mopt,
                                       cvar_views = ConditionalValueatRiskView(; views = v)),
                   ep_srd)
        @test isapprox(ep_scvar(1, pc.w) + ep_scvar(2, pc.w), f * pAc, rtol = 5e-3)
    end
end

@testset "the sequential carriers register the rows their formulation names" begin
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
    x = -ep_sX[:, 1]
    y = -ep_sX[:, 2]
    Ts = length(x)
    # A dual carrier over two assets: one block per asset and one target row.
    nv, d = counts(PO.LinearConditionalValueatRiskViewConstraint([x, y], [1.0, 1.0], ep_a,
                                                                 0.05), Ts)
    @test nv == 2 * Ts
    @test d[LE] == 2 * Ts + 1
    @test d[EQ] == 2
    # A sequential carrier with one asset on each side: one block and one row.
    c = ones(Ts)
    nv, d = counts(PO.SequentialConditionalValueatRiskViewConstraint([x], [1.0], [y],
                                                                     [-1.0], c, 0.0, ep_a,
                                                                     0.01, 20, 1e-8), Ts)
    @test nv == Ts
    @test d[LE] == Ts + 1
    @test d[EQ] == 1
    # With no asset on the dual side the view is one linear row and no variable.
    nv, d = counts(PO.SequentialEntropicValueatRiskViewConstraint([], Float64[], [y],
                                                                  [-1.0], c, 0.0, ep_a,
                                                                  -0.05, 20, 1e-8, (), (;),
                                                                  nothing), Ts)
    @test nv == 0
    @test d[LE] == 1
    @test !haskey(d, EQ)
    nv, d = counts(PO.SequentialRelativisticValueatRiskViewConstraint([x], [1.0], [y],
                                                                      [-1.0], c, 0.0, ep_a,
                                                                      ep_k, 0.01, 20, 1e-8,
                                                                      (), (;), nothing), Ts)
    @test nv == 3 * Ts
    @test d[LE] == 2
    @test d[EQ] == 1
    @test d[(Vector{JuMP.AffExpr}, JuMP.MOI.PowerCone{Float64})] == 2 * Ts
end
