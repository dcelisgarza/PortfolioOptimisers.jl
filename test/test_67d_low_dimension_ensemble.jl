#=
The low-dimension ensemble prior and the online low-dimension ensemble portfolio (issue #1186 on
map #1148): a prior whose mean and covariance come off the same random low-dimensional lagged
regressions, and the `FollowTheLeader` configuration whose programme is the paper's
mean-variance-turnover objective over that prior.

Parity: the moments are checked against the paper's equations 4 to 10 written out by hand on
one full-system subsystem, where the kernel weights are all one, and on two single-asset
subsystems, where they are not; the programme is checked against the paper's own solver, the
relaxed coordinate-wise descent followed by a simplex projection, and the tolerance that
relaxation costs is stated; at `xi = 0`, `gamma = 0` the programme is the one-hot argmax of the
forecast.
=#
using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, StatsBase, Dates,
      Clarabel
@testset "Online portfolio selection: the low-dimension ensemble" begin
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection
    rows(r, i) = po.port_opt_view(r, i, :)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                 "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                 "max_iter" => 500),
                 check_sol = (; allow_local = true, allow_almost = true))
    # The Euclidean projection onto the simplex, for the paper's last step.
    function simplex(v)
        u = sort(v; rev = true)
        css = cumsum(u)
        rho = findlast(k -> u[k] + (1 - css[k]) / k > 0, eachindex(u))
        return max.(v .+ (1 - css[rho]) / rho, 0)
    end

    rng = StableRNG(11)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)

    @testset "Construction" begin
        pe = LowDimensionEnsemblePrior()
        @test pe.n_subsystems == 300 && pe.subsystem_size == 3 && pe.sigma == 0.025
        @test isa(pe.pdm, Posdef) && isnothing(pe.seed)
        @test_throws DomainError LowDimensionEnsemblePrior(; n_subsystems = 0)
        @test_throws DomainError LowDimensionEnsemblePrior(; subsystem_size = 0)
        @test_throws DomainError LowDimensionEnsemblePrior(; sigma = 0)
        @test_throws DomainError LowDimensionEnsemblePrior(; sigma = Inf)
        @test isnothing(po.factor_residual_config(pe))
        # The floor a tree states reaches the rule that holds it.
        @test po.fit_min_rows(pe) == 3
        @test po.fit_min_rows(EmpiricalPrior()) == 1
        @test po.fit_min_rows(nothing) == 1
        @test po.fit_min_rows(prior(EmpiricalPrior(), R)) == 1
        @test po.fit_min_rows(PriorExpectedReturns(; pe = pe)) == 3
        @test po.forecast_min_rows(PriorExpectedReturns(; pe = pe)) == 3
        @test po.forecast_min_rows(SimpleExpectedReturns()) == 1
        alg = LowDimensionEnsemblePortfolio(; N = N, slv = slv)
        @test po.leader_min_rows(alg.opt) == 3
        @test po.leader_min_rows(MeanRisk(;
                                          opt = JuMPOptimiser(; pe = EmpiricalPrior(),
                                                              slv = slv))) == 2
        @test alg.sel == LastRows(; W = 6)
        @test alg.gamma == 0
        @test isa(alg.opt.obj, MaximumUtility) && alg.opt.obj.l == 0.25
        @test isa(alg.opt.r, Variance) && isa(alg.opt.r.alg, QuadRiskExpr)
        @test isa(alg.opt.opt.pe, LowDimensionEnsemblePrior)
        @test alg.opt.opt.fees.tn.val == 0.002
        @test alg.opt.opt.fees.tn.w == fill(0.25, N)
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = 0, slv = slv)
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = N, window = 1,
                                                               slv = slv)
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = N, gamma = -1,
                                                               slv = slv)
        @test_throws DomainError LowDimensionEnsemblePortfolio(; N = N, xi = -1, slv = slv)
    end

    @testset "The prior on one full-system subsystem" begin
        # One subsystem over every asset: the kernel weights are all one, the forecast is
        # the regression's, and the covariance is the regressors' sample covariance pushed
        # through the coefficients (equations 4, 5, 8 and 9).
        W6 = R[1:6, :]
        P = 1 .+ W6
        Y = P[2:end, :]
        Z = P[1:(end - 1), :]
        B = Z \ Y
        pe1 = LowDimensionEnsemblePrior(; n_subsystems = 1, subsystem_size = N,
                                        pdm = nothing)
        pr = prior(pe1, W6)
        @test isa(pr, LowOrderPrior)
        @test pr.X === W6 || pr.X == W6
        @test isapprox(pr.mu, vec(P[end, :]' * B) .- 1; atol = 1e-12)
        @test isapprox(pr.sigma, B' * cov(Y) * B; atol = 1e-12)
        @test issymmetric(pr.sigma)
        # A subsystem wider than the universe is clipped to it.
        pe9 = LowDimensionEnsemblePrior(; n_subsystems = 1, subsystem_size = 9,
                                        pdm = nothing)
        @test isapprox(prior(pe9, W6).sigma, pr.sigma; atol = 1e-12)
        # Orientation.
        @test isapprox(prior(pe1, permutedims(W6); dims = 2).mu, pr.mu; atol = 1e-12)
        # Too few rows for the regressor covariance.
        @test_throws ArgumentError prior(pe1, W6[1:2, :])
        # Factor returns and a panel are ignored.
        @test isapprox(prior(pe1, W6, W6[:, 1:1]).mu, pr.mu; atol = 1e-12)
    end

    @testset "The prior on two single-asset subsystems" begin
        # Two subsystems of one asset each, drawn as the estimator draws them, so the kernel
        # weights differ and equations 6, 7 and 10 are exercised by hand.
        W6 = R[1:6, 1:3]
        P = 1 .+ W6
        Y = P[2:end, :]
        Z = P[1:(end - 1), :]
        z = P[end, :]
        w = 5
        sigma = 0.025
        pe2 = LowDimensionEnsemblePrior(; n_subsystems = 2, subsystem_size = 1,
                                        sigma = sigma, pdm = nothing, seed = 7)
        drng = po.resolve_rng(pe2.rng, 7)
        idxs = [StatsBase.sample(drng, 1:3, 1; replace = false) for _ in 1:2]
        Bs = [Z[:, idx] \ Y for idx in idxs]
        Rl = [vec(sum(abs2, Y .- Z[:, idxs[l]] * Bs[l]; dims = 1)) ./ w for l in 1:2]
        Fl = [vec(Bs[l]' * z[idxs[l]]) for l in 1:2]
        V = [exp.(-Rl[l] ./ sigma^2) for l in 1:2]
        Vs = V[1] .+ V[2]
        xhat = (V[1] .* Fl[1] .+ V[2] .* Fl[2]) ./ Vs
        Sl = [Bs[l]' * cov(Y[:, idxs[l]]) * Bs[l] for l in 1:2]
        num = (V[1] * V[1]') .* Sl[1] .+ (V[2] * V[2]') .* Sl[2]
        den = V[1] * V[1]' .+ V[2] * V[2]'
        pr = prior(pe2, W6)
        @test isapprox(pr.mu, xhat .- 1; atol = 1e-12)
        @test isapprox(pr.sigma, num ./ den; atol = 1e-12)
        # The two weights are not equal, so the hand check read the kernel.
        @test !isapprox(V[1] ./ Vs, fill(0.5, 3); atol = 1e-3)
        # A seed draws the same subsystems at every fit; no seed re-draws.
        @test prior(pe2, W6).mu == pr.mu
        pe_free = LowDimensionEnsemblePrior(; n_subsystems = 2, subsystem_size = 1,
                                            pdm = nothing, rng = StableRNG(3))
        @test prior(pe_free, W6).mu != prior(pe_free, W6).mu
    end

    @testset "The repair and the default" begin
        # The aggregate under entry-wise weights need not be positive semidefinite; the
        # default repairs it and `nothing` leaves it.
        W = R[1:6, :]
        pe = LowDimensionEnsemblePrior(; n_subsystems = 40, subsystem_size = 2, seed = 3)
        pr = prior(pe, W)
        @test isposdef(pr.sigma)
        @test issymmetric(pr.sigma)
        pe0 = LowDimensionEnsemblePrior(; n_subsystems = 40, subsystem_size = 2, seed = 3,
                                        pdm = nothing)
        pr0 = prior(pe0, W)
        @test pr0.mu == pr.mu
        @test issymmetric(pr0.sigma)
        # Errors far above the bandwidth: the shift keeps the weights finite and summing to
        # one, where the unshifted kernel underflows every weight.
        Wbig = 5 .* W
        prb = prior(LowDimensionEnsemblePrior(; n_subsystems = 40, subsystem_size = 2,
                                              seed = 3, sigma = 1e-3), Wbig)
        @test all(isfinite, prb.mu) && all(isfinite, prb.sigma)
        # The prior serves an expected-returns slot through the library-wide adapter.
        me = PriorExpectedReturns(; pe = pe)
        @test isapprox(vec(mean(me, W)), pr.mu; atol = 1e-12)
    end

    @testset "The programme against the paper's solver" begin
        # The paper's coordinate-wise descent (equations 15 to 23): relax `b >= 0`, descend
        # on `c = b - w` under a Lagrange multiplier started at `1.1 xi` and updated from
        # the active set, then project onto the simplex. Run to a tighter stop than the
        # paper's `0.01` so the comparison reads the method, not its stopping rule.
        function paper_step(xhat, S, w, gamma, xi; iters = 2000, tol = 1e-12)
            d = length(w)
            z = 2 * gamma * (S * w) .- xhat
            c = fill(1 / d, d) .- w
            lam = 1.1 * xi
            f(c) = gamma * dot(c, S, c) + dot(z, c) + xi * sum(abs, c)
            fo = f(c)
            for _ in 1:iters
                for i in 1:d
                    off = 2 * gamma * (dot(S[i, :], c) - S[i, i] * c[i])
                    if c[i] > 0
                        c[i] = (lam - xi - z[i] - off) / (2 * gamma * S[i, i])
                    elseif c[i] < 0
                        c[i] = (lam + xi - z[i] - off) / (2 * gamma * S[i, i])
                    end
                end
                act = findall(!iszero, c)
                if !isempty(act)
                    inv = [1 / (2 * gamma * S[i, i]) for i in act]
                    offs = [(z[i] + 2 * gamma * (dot(S[i, :], c) - S[i, i] * c[i])) /
                            (2 * gamma * S[i, i]) for i in act]
                    sgn = [c[i] > 0 ? 1 : -1 for i in act]
                    lam = (sum(offs) - xi * dot(sgn, inv)) / sum(inv)
                end
                fn = f(c)
                if abs(fn - fo) <= tol
                    break
                end
                fo = fn
            end
            return simplex(c .+ w)
        end
        gamma, xi = 0.25, 0.002
        pe = LowDimensionEnsemblePrior(; seed = 5)
        alg = LowDimensionEnsemblePortfolio(; N = N, gamma = gamma, xi = xi, slv = slv,
                                            pe = pe)
        W6 = R[10:15, :]
        pr = prior(pe, W6)
        xhat = 1 .+ pr.mu
        S = pr.sigma
        # A reference allocation away from uniform, so the fee binds.
        wref = [0.5, 0.3, 0.1, 0.1]
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        b = po.leader_allocation(alg.opt, W6, wref, set, W6)
        @test isapprox(sum(b), 1; atol = 1e-8)
        @test minimum(b) >= -1e-8
        obj(b) = -dot(xhat, b) + gamma * dot(b, S, b) + xi * norm(b .- wref, 1)
        bp = paper_step(xhat, S, wref, gamma, xi)
        # The programme minimises the stated problem; the relax-then-project answer is
        # feasible for it, so it is no better.
        @test obj(b) <= obj(bp) + 1e-9
        # The tolerance the relaxation costs on this window: the two answers agree to
        # 0.02 in the one-norm.
        @test norm(b .- bp, 1) <= 0.02
        # The fee is the paper's penalty against the reference: with it the answer trades
        # less from `wref` than without it.
        alg0 = LowDimensionEnsemblePortfolio(; N = N, gamma = gamma, xi = 0, slv = slv,
                                             pe = pe)
        b0 = po.leader_allocation(alg0.opt, W6, wref, set, W6)
        @test norm(b .- wref, 1) <= norm(b0 .- wref, 1) + 1e-8
        # The variance term pulls towards the variance-minimising direction: at a large
        # risk aversion the answer moves away from the argmax.
        algg = LowDimensionEnsemblePortfolio(; N = N, gamma = 500, xi = 0, slv = slv,
                                             pe = pe)
        bg = po.leader_allocation(algg.opt, W6, wref, set, W6)
        @test obj(bg) >= obj(b0) - 1e-9
        @test !isapprox(bg, b0; atol = 1e-3)
        # Without fee and variance the programme is the one-hot argmax of the forecast.
        alg00 = LowDimensionEnsemblePortfolio(; N = N, gamma = 0, xi = 0, slv = slv,
                                              pe = pe)
        b00 = po.leader_allocation(alg00.opt, W6, wref, set, W6)
        onehot = zeros(N)
        onehot[argmax(xhat)] = 1
        @test isapprox(b00, onehot; atol = 1e-6)
    end

    @testset "The head" begin
        pe = LowDimensionEnsemblePrior(; n_subsystems = 60, seed = 5)
        alg = LowDimensionEnsemblePortfolio(; N = N, slv = slv, pe = pe)
        # Below three rows the rule answers the uniform portfolio; from three it solves.
        @test optimise(OPS(; alg = alg), rows(rd, 1:1)).w == fill(0.25, N)
        @test optimise(OPS(; alg = alg), rows(rd, 1:2)).w == fill(0.25, N)
        w3 = optimise(OPS(; alg = alg), rows(rd, 1:3)).w
        @test isapprox(sum(w3), 1; atol = 1e-8) && !isapprox(w3, fill(0.25, N); atol = 1e-3)
        res = optimise(OPS(; alg = alg), rd)
        @test isa(res.retcode, OptimisationSuccess)
        @test isapprox(sum(res.w), 1; atol = 1e-8)
        @test minimum(res.w) >= -1e-8
        # The last update re-solves on the last six rows with the Price-Adjusted Allocation
        # of the previous target as the fee's reference.
        prev = optimise(OPS(; alg = alg), rows(rd, 1:(T - 1))).w
        X = 1 .+ R
        wh = prev .* X[T, :] ./ dot(prev, X[T, :])
        set = po.resolve_allocation_set(BoundedAllocationSet(), N, false, Float64)
        bhand = po.leader_allocation(alg.opt, R[(T - 5):T, :], wh, set, R)
        @test isapprox(res.w, bhand; atol = 1e-6)
        # The head's set enters the programme.
        wcap = optimise(OPS(; alg = alg,
                            set = ProgrammeAllocationSet(; slv = slv,
                                                         wb = WeightBounds(0, 0.5))),
                        rows(rd, 1:10)).w
        @test maximum(wcap) <= 0.5 + 1e-6
        @test isapprox(sum(wcap), 1; atol = 1e-8)
        # The rule and its prior view onto the investable assets.
        v = po.port_opt_view(alg, [1, 3], R)
        @test isa(v, FollowTheLeader)
        @test length(v.opt.opt.fees.tn.w) == 2
        @test v.opt.opt.pe === alg.opt.opt.pe
    end

    @testset "Docstrings" begin
        docs = [string(@doc(LowDimensionEnsemblePrior)),
                string(@doc(LowDimensionEnsemblePortfolio)), string(@doc(po.fit_min_rows)),
                string(@doc(po.ensemble_programme))]
        @test all(d -> !occursin("No documentation found", d), docs)
        @test occursin("Forecasts the next price relative", docs[1])
    end
end
