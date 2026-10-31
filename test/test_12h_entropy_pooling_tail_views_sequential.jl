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
    f = SequentialTailView()
    @test f.iters == 20
    @test f.tol == 1e-8
    @test SequentialTailView(; iters = 0, tol = 1e-4).iters == 0
    @test_throws DomainError SequentialTailView(; iters = -1)
    @test_throws DomainError SequentialTailView(; tol = 0.0)
    # One sequential formulation serves every measure, and the view group names the
    # measure. Issue #1872.
    @test isa(f, PO.AbstractEntropyPoolingViewFormulation)
    v = LinearConstraintEstimator(; val = "AAPL <= 0.1")
    @test ConditionalValueatRiskView(; views = v, alg = f).alg === f
    @test EntropicValueatRiskView(; views = v, alg = f).alg === f
    @test RelativisticValueatRiskView(; views = v, alg = f).alg === f
    # A vector that mixes a formulation of the measure with the shared one keeps working,
    # and an entry of another measure is refused at construction.
    @test length(ConditionalValueatRiskView(; views = v,
                                            alg = [LinearConditionalValueatRiskView(), f]).alg) ==
          2
    @test length(EntropicValueatRiskView(; views = v,
                                         alg = [ConicEntropicValueatRiskView(), f,
                                                GridTailView()]).alg) == 3
    @test length(RelativisticValueatRiskView(; views = v,
                                             alg = [ConicRelativisticValueatRiskView(), f,
                                                    GridTailView()]).alg) == 3
    @test_throws ArgumentError ConditionalValueatRiskView(; views = v,
                                                          alg = [f, GridTailView()])
    @test_throws ArgumentError EntropicValueatRiskView(; views = v,
                                                       alg = [f,
                                                              LinearConditionalValueatRiskView()])
    @test_throws ArgumentError RelativisticValueatRiskView(; views = v,
                                                           alg = [f,
                                                                  ConicEntropicValueatRiskView()])
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

@testset "the value at risk multiplier and the start of the tilt chain at their limits (#1912)" begin
    # A level that the running mass never reaches answers the smallest loss. The 201 equal
    # weights sum to a hair under one, so a level just below one is never reached.
    al = prevfloat(1.0)
    @test foldl(+, ep_sw0 ./ sum(ep_sw0)) < al
    # At that level the measure is the mean loss, so the view fixes the mean of the posterior.
    cv1 = ConditionalValueatRisk(; alpha = al, w = ep_sw0)(ep_sX[:, 1])
    @test isapprox(cv1, -LinearAlgebra.dot(ep_sw0, ep_sX[:, 1]); rtol = 1e-10)
    tgt = cv1 * 0.85
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                   cvar_views = ConditionalValueatRiskView(;
                                                                           alg = SequentialTailView(),
                                                                           alpha = al,
                                                                           views = LinearConstraintEstimator(;
                                                                                                             val = "AAPL <= $tgt"))),
               ep_srd)
    @test isapprox(-LinearAlgebra.dot(pr.w, ep_sX[:, 1]), tgt; rtol = 1e-5)
    @test isapprox(ConditionalValueatRisk(; alpha = al, w = pr.w)(ep_sX[:, 1]), tgt;
                   rtol = 1e-5)
    @test pr.kld > 0
    # An asset that no scenario tells apart has a constant row, so no tilt of the posterior
    # reaches a target above it. The view that the prior meets leaves the posterior at the
    # prior, and the view that no reweighting meets is refused by name.
    X2 = copy(ep_sX[(end - 60):end, :])
    X2[:, 1] .= 0.001
    rd2 = ReturnsResult(; nx = rd.nx, X = X2)
    mk(t) = EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                cvar_views = ConditionalValueatRiskView(;
                                                                        alg = SequentialTailView(),
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "AAPL <= $t")))
    flat = prior(mk(0.5), rd2)
    @test isapprox(flat.w, fill(inv(size(X2, 1)), size(X2, 1)); rtol = 1e-6)
    @test flat.kld < 1e-9
    @test_throws DomainError prior(mk(-0.5), rd2)
    # A posterior on one scenario has no room to tilt: the target sits on the end of the
    # support, no tilt exists, and the start returns the constraint it read at the prior.
    # A prior whose worst scenario holds at least `alpha` reaches this exit through `prior`,
    # and the fit then fails on a feasible view (#1923). The direct call stays until then.
    x = -ep_sX[:, 1]
    T = length(x)
    tv = PO.SequentialConditionalValueatRiskViewConstraint([], Float64[], [x], [-1.0],
                                                           zeros(T), 0.0, ep_a, 10.0, 20,
                                                           1e-8)
    wp = zeros(T)
    wp[1] = 1.0
    out = PO.ep_sequential_start(tv, wp)
    @test out isa PO.SequentialConditionalValueatRiskViewConstraint
    @test out.rhs == tv.rhs
    @test length(out.c) == T
    @test out.c == PO.ep_refine_tail_view(tv, wp)[1].c
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
                                                                          alg = SequentialTailView(;
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
                                                                           alg = SequentialTailView(),
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
                                                                            alg = SequentialTailView(),
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
                                                                            alg = SequentialTailView(),
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
                                                                        alg = SequentialTailView(),
                                                                        views = LinearConstraintEstimator(;
                                                                                                          val = "AAPL <= $tgt4"))),
               ep_srd)
    @test isapprox(ep_sevar(1, pu.w), tgt4, rtol = 1e-5)
    tgt5 = ep_sprlvar * 0.9
    pu2 = prior(EntropyPoolingPrior(; sets = sets, opt = ep_jopt,
                                    rlvar_views = RelativisticValueatRiskView(;
                                                                              kappa = ep_k,
                                                                              alg = SequentialTailView(),
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
