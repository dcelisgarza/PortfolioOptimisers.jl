using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Clarabel, Logging

# The investable reduction and the weight expansion, issue #678, under the contract #647 fixed.
#
# A Prior Estimator fits on the coverage universe and returns a result on the FULL asset
# universe, where an asset it could not estimate carries `NaN` in `mu` and on the diagonal of
# `sigma`. The optimiser derives the Investable Mask from that result, reduces once at its
# entry, solves over the investable assets alone, and expands the solved weights back into a
# zero vector of the full length.
#
# The oracle is the same optimisation with the non-investable asset removed by hand. The two
# must agree weight for weight, so the tolerances are the solver's own and nothing wider. A
# single Clarabel at tightened tolerances is used for every parity comparison: two identical
# programmes drift apart at the shipped defaults, and the drift is the solver's, not the
# reduction's.

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             check_sol = (; allow_local = true, allow_almost = true),
             settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                             "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                             "tol_infeas_abs" => 1e-12, "tol_infeas_rel" => 1e-12))

rng = StableRNG(987654321)
T, N = 200, 5
X = randn(rng, T, N) ./ 100 .+ 0.0005
F = randn(StableRNG(123456789), T, 3) ./ 100
nx = ["a", "b", "c", "d", "e"]
nf = ["f1", "f2", "f3"]
rd = ReturnsResult(; nx = nx, X = X, nf = nf, F = F)
pr = prior(EmpiricalPrior(), rd)

# The non-investable asset, and the universe that survives it.
k = 3
keep = [1, 2, 4, 5]

# The full-universe prior a cross-sectional fit would hand the optimiser.
function nan_prior(pr, k)
    mu = collect(pr.mu)
    sigma = collect(pr.sigma)
    X = collect(pr.X)
    mu[k] = NaN
    sigma[k, :] .= NaN
    sigma[:, k] .= NaN
    X[:, k] .= NaN
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end
prn = nan_prior(pr, k)
prk = LowOrderPrior(; X = pr.X[:, keep], mu = pr.mu[keep], sigma = pr.sigma[keep, keep])
rdk = ReturnsResult(; nx = nx[keep], X = X[:, keep], nf = nf, F = F)

@testset "Investable Mask: derived from the result, never a field on it" begin
    # No prior result carries a mask. The all-investable case is the `nothing` sentinel, not
    # a vector of `true`, and that is what keeps every existing path allocation-free.
    @test isnothing(PortfolioOptimisers.investable_mask(pr))
    @test isnothing(PortfolioOptimisers.investable_mask(prk))
    @test PortfolioOptimisers.investable_mask(prn) == BitVector([1, 1, 0, 1, 1])
    # Either half of the conjunction excludes an asset on its own.
    mu = collect(pr.mu)
    mu[2] = NaN
    @test PortfolioOptimisers.investable_mask(LowOrderPrior(; X = pr.X, mu = mu,
                                                            sigma = pr.sigma)) ==
          BitVector([1, 0, 1, 1, 1])
    sigma = collect(pr.sigma)
    sigma[4, 4] = NaN
    @test PortfolioOptimisers.investable_mask(LowOrderPrior(; X = pr.X, mu = pr.mu,
                                                            sigma = sigma)) ==
          BitVector([1, 1, 1, 0, 1])
    # An off-diagonal `NaN` is not read: the diagonal alone decides.
    sigma = collect(pr.sigma)
    sigma[1, 2] = NaN
    @test isnothing(PortfolioOptimisers.investable_mask(LowOrderPrior(; X = pr.X,
                                                                      mu = pr.mu,
                                                                      sigma = sigma)))
    # An empty investable set has no optimisation to state, so it throws where it is derived
    # rather than passing a zero-asset problem downstream.
    @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.investable_mask(LowOrderPrior(;
                                                                                                    X = pr.X,
                                                                                                    mu = fill(NaN,
                                                                                                              N),
                                                                                                    sigma = pr.sigma))
end

@testset "The reduction and the expansion reproduce the hand-reduced problem" begin
    res = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prn, slv = slv)), rd)
    ref = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prk, slv = slv)), rdk)
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    # The weights come back on the caller's own universe.
    @test length(res.w) == N
    @test length(ref.w) == length(keep)
    # A non-investable asset holds a zero: the optimiser could not trade it.
    @test iszero(res.w[k])
    # And every other weight is the weight the reduced problem solved for.
    @test isapprox(res.w[keep], ref.w; rtol = 1e-8)
    @test isapprox(sum(res.w), 1; rtol = 1e-8)
    # The mask reaches the result through the bundle, so a reader of a walk-forward can tell
    # which assets each fold could trade.
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test isnothing(ref.imsk)
    # The bundle carries the reduced prior, so the model was built over four assets.
    @test size(res.pr.X, 2) == length(keep)
end

@testset "A constraint stated on the full universe binds on the right asset" begin
    sets = UniverseSets(; dict = Dict("nx" => nx))
    # A weight bound written against the full universe, naming an asset that survives.
    wb = WeightBoundsEstimator(; ub = Dict("e" => 0.1))
    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; pe = prn, slv = slv, wb = wb,
                                                sets = sets)), rd)
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test res.w[5] <= 0.1 + 1e-8
    @test isapprox(res.w[5], 0.1; atol = 1e-7)
    @test iszero(res.w[k])
    # A linear constraint written against the full universe, naming two surviving assets.
    lcse = LinearConstraintEstimator(; val = "a - b == 0")
    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; pe = prn, slv = slv, lcse = lcse,
                                                sets = sets)), rd)
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test isapprox(res.w[1], res.w[2]; atol = 1e-8)
    @test iszero(res.w[k])
end

@testset "Every JuMP head reduces and expands" begin
    # The reduction lives in the shared prelude, and each head takes the same view of itself
    # and of `rd`, so every family that reaches `JuMPOptimisationResult` expands.
    for est in (RiskBudgeting(; opt = JuMPOptimiser(; pe = prn, slv = slv)),
                RelaxedRiskBudgeting(; opt = JuMPOptimiser(; pe = prn, slv = slv)),
                NearOptimalCentering(; opt = JuMPOptimiser(; pe = prn, slv = slv)))
        res = optimise(est, rd)
        @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test length(res.w) == N
        @test iszero(res.w[k])
        @test isapprox(sum(res.w), 1; rtol = 1e-8)
        @test res.imsk == BitVector([1, 1, 0, 1, 1])
    end
    fsets = UniverseSets(; dict = Dict("nx" => nf))
    res = optimise(FactorRiskContribution(; opt = JuMPOptimiser(; pe = prn, slv = slv),
                                          sets = fsets), rd)
    @test length(res.w) == N
    @test iszero(res.w[k])
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
end

@testset "The efficient frontier expands every sweep point" begin
    lb = Frontier(; N = 3)
    res = optimise(MeanRisk(;
                            opt = JuMPOptimiser(; pe = prn, slv = slv,
                                                ret = ArithmeticReturn(;
                                                                       settings = JuMPReturnsSettings(;
                                                                                                      lb = lb)))),
                   rd)
    @test length(res.w) == 3
    @test all(w -> length(w) == N, res.w)
    @test all(w -> iszero(w[k]), res.w)
end

@testset "The expansion on its own terms" begin
    sol = PortfolioOptimisers.JuMPOptimisationSolution(; w = [0.25, 0.25, 0.5])
    imsk = BitVector([1, 1, 0, 1, 0])
    # The `nothing` sentinel returns the very object it was given: the all-investable path
    # copies nothing.
    @test PortfolioOptimisers.expand_investable_weights(nothing, sol) === sol
    out = PortfolioOptimisers.expand_investable_weights(imsk, sol)
    @test out.w == [0.25, 0.25, 0.0, 0.5, 0.0]
    # A failed solve carries `NaN` at each solved position. The expansion keeps the
    # distinction between an asset the optimiser tried and one it never could.
    nan_sol = PortfolioOptimisers.JuMPOptimisationSolution(; w = fill(NaN, 3))
    out = PortfolioOptimisers.expand_investable_weights(imsk, nan_sol)
    @test iszero(out.w[3])
    @test iszero(out.w[5])
    @test all(isnan, out.w[[1, 2, 4]])
    # The vector route, one solution per frontier point.
    outs = PortfolioOptimisers.expand_investable_weights(imsk, [sol, sol])
    @test length(outs) == 2
    @test all(o -> o.w == [0.25, 0.25, 0.0, 0.5, 0.0], outs)
    # A mask and a solution from different optimisations are refused where they meet.
    @test_throws DimensionMismatch PortfolioOptimisers.expand_investable_weights(BitVector([1,
                                                                                            1,
                                                                                            1,
                                                                                            1,
                                                                                            0]),
                                                                                 sol)
end

@testset "The all-investable path is the path it was" begin
    # Nothing about an ordinary optimisation changes: no mask, no reduction, no expansion.
    res = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv)), rd)
    @test isnothing(res.imsk)
    @test length(res.w) == N
    @test size(res.pr.X, 2) == N
    @test isapprox(sum(res.w), 1; rtol = 1e-8)
    # And the bundle keeps its own default, so a caller building one by hand states nothing.
    attrs = PortfolioOptimisers.processed_jump_optimiser_attributes(JuMPOptimiser(; pe = pr,
                                                                                  slv = slv),
                                                                    rd)
    @test isnothing(attrs.imsk)
end

@testset "The reduction reaches the optimiser view and the bundle" begin
    # `investable_reduction` is the seam: it derives the mask and takes the three views. The
    # `nothing` method returns its arguments untouched, which is what dispatch buys.
    opt = JuMPOptimiser(; pe = pr, slv = slv)
    imsk, pro, opto, rdo = PortfolioOptimisers.investable_reduction(pr, opt, rd)
    @test isnothing(imsk)
    @test pro === pr
    @test opto === opt
    @test rdo === rd
    imsk, pro, opto, rdo = PortfolioOptimisers.investable_reduction(prn, opt, rd)
    @test imsk == BitVector([1, 1, 0, 1, 1])
    @test size(pro.X, 2) == length(keep)
    @test size(rdo.X, 2) == length(keep)
    @test rdo.nx == nx[keep]
    # The factor axis is untouched: the mask indexes assets.
    @test rdo.nf == nf
    @test size(rdo.F, 2) == length(nf)
    # And the head-side view, which reduces what the bundle does not carry.
    mr = MeanRisk(; wi = collect(1:N) ./ sum(1:N), opt = opt)
    mro, rdo = PortfolioOptimisers.investable_view(mr, rd, prn, imsk)
    @test length(mro.wi) == length(keep)
    @test mro.wi == (collect(1:N) ./ sum(1:N))[keep]
    @test size(rdo.X, 2) == length(keep)
    # The `nothing` method leaves both alone.
    mro, rdo = PortfolioOptimisers.investable_view(mr, rd, pr, nothing)
    @test mro === mr
    @test rdo === rd
end

# The naive and meta families, issue #676, under ADR 0115. `InverseVolatility`, `Stacking`
# and `SubsetResampling` each reduce once at their entry through the same shared verb the
# JuMP prelude uses, and each result expands in its keyword constructor. A meta family
# composes two masks: its own at its entry, and each inner head's inside its own solve.

@testset "The plain-vector expansion on its own terms" begin
    imsk = BitVector([1, 1, 0, 1, 1])
    w = [0.1, 0.2, 0.3, 0.4]
    # The `nothing` sentinel returns the very object it was given, as the JuMP route does.
    @test PortfolioOptimisers.expand_investable_weights(nothing, w) === w
    # A naive head whose finaliser gave up records no weights, and no mask makes them.
    @test isnothing(PortfolioOptimisers.expand_investable_weights(nothing, nothing))
    @test isnothing(PortfolioOptimisers.expand_investable_weights(imsk, nothing))
    @test PortfolioOptimisers.expand_investable_weights(imsk, w) ==
          [0.1, 0.2, 0.0, 0.3, 0.4]
    # A failed solve carries `NaN` at each solved position, and the expansion keeps the
    # distinction between an asset the optimiser tried and one it never could.
    out = PortfolioOptimisers.expand_investable_weights(imsk, fill(NaN, 4))
    @test iszero(out[k])
    @test all(isnan, out[keep])
    # The frontier route, one weight vector per sweep point.
    @test PortfolioOptimisers.expand_investable_weights(imsk, [w, reverse(w)]) ==
          [[0.1, 0.2, 0.0, 0.3, 0.4], [0.4, 0.3, 0.0, 0.2, 0.1]]
    # A mask and a weight vector from different optimisations are refused where they meet.
    @test_throws DimensionMismatch PortfolioOptimisers.expand_investable_weights(imsk,
                                                                                 [0.5, 0.5])
end

@testset "InverseVolatility reduces and expands" begin
    res = optimise(InverseVolatility(; pe = prn), rd)
    oracle = optimise(InverseVolatility(; pe = prk), rdk)
    # The oracle is the same optimisation with the non-investable asset removed by hand.
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test isnothing(oracle.imsk)
    @test length(res.w) == N
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w)
    # The result carries the objects of the reduced universe beside the mask.
    @test size(res.pr.X, 2) == length(keep)
    # The all-investable path is the path it was.
    plain = optimise(InverseVolatility(; pe = pr), rd)
    @test isnothing(plain.imsk)
    @test length(plain.w) == N
    # No investable asset throws where the mask is derived.
    prz = LowOrderPrior(; X = pr.X, mu = fill(NaN, N), sigma = pr.sigma)
    @test_throws PortfolioOptimisers.IsEmptyError optimise(InverseVolatility(; pe = prz),
                                                           rd)
end

# The two prior-free naive heads, issue #859, under ADR 0120. `EqualWeighted` and
# `RandomWeighted` fit no prior, so no Prior Result yields an Investable Mask for them. Each
# derives the Coverage Universe of its own window instead, through the verb of ADR 0117,
# weights the assets it keeps, and carries that universe as `imsk`. Every optimisation
# result of the library therefore carries a mask, and a reader has one idiom.
#
# The oracle is the one every other family here uses: the same head on the panel with the
# dead column removed by hand, weight for weight.

# A listing inside the window: asset `k` has no quote over the first thirty observations.
Xg = copy(X)
Xg[1:30, k] .= NaN
rdg = ReturnsResult(; nx = nx, X = Xg, nf = nf, F = F)

# The same asset, live and quoted, but outside the universe over an inactive spell. Its
# price is finite at every row, so only the panel's active mask says it must not be traded.
stale_amsk = trues(T, N)
stale_amsk[10:20, k] .= false
function naive_panel(amsk)
    return AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(T, N))],
                      amsk = amsk, emsk = amsk)
end
rds = ReturnsResult(; nx = nx, X = X, nf = nf, F = F, pnl = naive_panel(stale_amsk))

@testset "EqualWeighted reduces to the Coverage Universe and expands" begin
    res = optimise(EqualWeighted(), rdg)
    oracle = optimise(EqualWeighted(), rdk)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test isnothing(oracle.imsk)
    # The weights come back on the caller's own universe, and the dead asset holds a zero.
    @test length(res.w) == N
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w)
    @test isapprox(sum(res.w), 1)
    # The result carries the objects of the reduced universe beside the mask.
    @test size(res.pr.X, 2) == length(keep)
    # A stale finite price during an inactive spell weights nothing: the active mask of the
    # Asset Panel excludes the asset on its own, with every return finite.
    @test all(isfinite, X)
    res = optimise(EqualWeighted(), rds)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w)
    # A complete window is the path it was: no mask, no reduction, no expansion.
    plain = optimise(EqualWeighted(), rd)
    @test isnothing(plain.imsk)
    @test length(plain.w) == N
    @test isapprox(plain.w, fill(inv(N), N))
    # A weight bound stated over the full universe binds on the right asset, because the
    # sets and the bounds are viewed by the same index the returns are.
    sets = UniverseSets(; dict = Dict("nx" => nx))
    wb = WeightBoundsEstimator(; ub = Dict("e" => 0.1))
    res = optimise(EqualWeighted(; wb = wb, sets = sets), rdg)
    @test iszero(res.w[k])
    @test isapprox(res.w[5], 0.1)
    @test isapprox(sum(res.w), 1)
    # An all-dead window has no universe to weight, and the refusal is the one the mask is
    # derived with.
    Xdead = copy(X)
    Xdead[1, :] .= NaN
    @test_throws PortfolioOptimisers.IsEmptyError optimise(EqualWeighted(),
                                                           ReturnsResult(; nx = nx,
                                                                         X = Xdead))
    @test_throws PortfolioOptimisers.IsEmptyError optimise(EqualWeighted(),
                                                           ReturnsResult(; nx = nx, X = X,
                                                                         pnl = naive_panel(falses(T,
                                                                                                  N))))
end

@testset "RandomWeighted draws over the Coverage Universe" begin
    res = optimise(RandomWeighted(; seed = 42), rdg)
    oracle = optimise(RandomWeighted(; seed = 42), rdk)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test isnothing(oracle.imsk)
    @test length(res.w) == N
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w)
    @test isapprox(sum(res.w), 1)
    # A vector `alpha` is one concentration per asset of the full universe, so the draw over
    # the reduced universe is the draw the hand-reduced concentrations give.
    alpha = collect(1.0:N)
    res = optimise(RandomWeighted(; alpha = alpha, seed = 42), rdg)
    oracle = optimise(RandomWeighted(; alpha = alpha[keep], seed = 42), rdk)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w)
    # And its length is checked against the full width, because that is the universe the
    # caller states it over.
    @test_throws DimensionMismatch optimise(RandomWeighted(; alpha = alpha[keep],
                                                           seed = 42), rdg)
    # The inactive spell excludes the asset here too.
    res = optimise(RandomWeighted(; seed = 42), rds)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test iszero(res.w[k])
    # A complete window is the path it was.
    plain = optimise(RandomWeighted(; seed = 42), rd)
    @test isnothing(plain.imsk)
    @test length(plain.w) == N
end

@testset "A walk-forward over a changing universe weights each fold's live assets" begin
    # One panel with a listing and a delisting, stated in the returns and in the active mask
    # alike, as a real panel states them.
    Xw = copy(X)
    wamsk = trues(T, N)
    Xw[1:40, 5] .= NaN            # asset 5 lists at observation 41
    wamsk[1:40, 5] .= false
    Xw[151:end, 3] .= NaN         # asset 3 delists after observation 150
    wamsk[151:end, 3] .= false
    rdw = ReturnsResult(; nx = nx, X = Xw,
                        pnl = AssetPanel(;
                                         pf = [NumericPanelField(; name = "mcap",
                                                                 vals = ones(T, N))],
                                         amsk = wamsk, emsk = wamsk))
    cv = IndexWalkForward(60, 20)
    (; train_idx) = split(cv, rdw)
    mpr = PortfolioOptimisers.fit_and_predict(EqualWeighted(), rdw, cv)
    @test length(mpr.pred) == length(train_idx)
    for (pred, tr) in zip(mpr.pred, train_idx)
        # The fold's mask is the Coverage Universe of its own training window, written out
        # by hand: finite at every row of the window, and active at every row of it.
        hand = BitVector([all(isfinite, view(Xw, tr, j)) && all(view(wamsk, tr, j))
                          for j in 1:N])
        expected = all(hand) ? nothing : hand
        @test pred.res.imsk == expected
        # The weights come back on the caller's universe, and an asset the fold could not
        # trade holds a zero.
        @test length(pred.res.w) == N
        @test isapprox(sum(pred.res.w), 1)
        @test all(iszero, view(pred.res.w, .!hand))
        @test all(isapprox(inv(count(hand))), view(pred.res.w, hand))
    end
    # The two ends of the run disagree about the universe, which is the point of the test.
    @test mpr.pred[1].res.imsk == BitVector([1, 1, 1, 1, 0])
    @test isnothing(mpr.pred[3].res.imsk)
    @test mpr.pred[end].res.imsk == BitVector([1, 1, 0, 1, 1])
end

@testset "SubsetResampling draws its subsets from the investable universe" begin
    sr(pe) = SubsetResampling(; pe = pe, opt = InverseVolatility(), subset_size = 3,
                              n_subsets = 3, seed = 42)
    res = optimise(sr(prn), rd)
    oracle = optimise(sr(prk), rdk)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test length(res.w) == N
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w)
    # The subset index is in reduced positions, as `pr`, `wb` and `fees` are, so no subset
    # can name a dead asset and the draw needs no second filter.
    @test size(res.idx) == (3, 3)
    @test all(res.idx .<= length(keep))
    @test res.idx == oracle.idx
    # A subset larger than the investable universe refuses where the count is checked,
    # rather than drawing a dead asset to make up the number.
    @test_throws ArgumentError optimise(SubsetResampling(; pe = prn,
                                                         opt = InverseVolatility(),
                                                         subset_size = 3, n_subsets = 5,
                                                         seed = 42), rd)
    # A rebuild never expands a second time: it goes through the positional constructor.
    @test PortfolioOptimisers.factory(res, nothing).w == res.w
    @test PortfolioOptimisers.set_retcode(res, OptimisationFailure()).w == res.w
end

@testset "Stacking reduces once, and its candidates compose their own masks" begin
    st(pe) = Stacking(; pe = pe, opti = [InverseVolatility(), EqualWeighted()],
                      opto = InverseVolatility())
    res = optimise(st(prn), rd)
    oracle = optimise(st(prk), rdk)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test length(res.w) == N
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w)
    # Every candidate solved the reduced universe, so its own record is the reduced width.
    @test all(r -> length(r.w) == length(keep), res.resi)
    @test PortfolioOptimisers.set_retcode(res, OptimisationFailure()).w == res.w
    # A JuMP candidate reduces again inside its own solve, and the two masks compose to the
    # same answer as the hand-reduced problem.
    stj(pe) = Stacking(; pe = pe,
                       opti = [MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv)),
                               EqualWeighted()], opto = InverseVolatility())
    res = optimise(stj(prn), rd)
    oracle = optimise(stj(prk), rdk)
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test iszero(res.w[k])
    @test isapprox(res.w[keep], oracle.w; rtol = 5e-6)
end

# The hierarchical, and the nested clustered, families, issue #675, under ADR 0115. The rule
# is the JuMP one, carried outward: reduce once at the entry, immediately after the prior fit
# and before the clustering, then expand in the keyword constructor of the result. The oracle
# is again the same optimisation with the non-investable asset removed by hand.

# Every hierarchical head takes the same pair of configurations: the full universe with a
# `NaN` asset, and the universe that survives it.
optn = HierarchicalOptimiser(; pe = prn)
optk = HierarchicalOptimiser(; pe = prk)

function test_hierarchical_parity(res, ref)
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    @test isa(ref.retcode, PortfolioOptimisers.OptimisationSuccess)
    # The weights come back on the caller's own universe.
    @test length(res.w) == N
    @test length(ref.w) == length(keep)
    # A non-investable asset holds a zero: the optimiser could not trade it.
    @test iszero(res.w[k])
    # And every other weight is the weight the hand-reduced problem solved for.
    @test isapprox(res.w[keep], ref.w; rtol = 1e-10)
    @test isapprox(sum(res.w), 1; rtol = 1e-10)
    # The mask reaches the result, so a reader of a walk-forward can tell which assets each
    # fold could trade, with the same idiom the JuMP results answer.
    @test res.imsk == BitVector([1, 1, 0, 1, 1])
    @test isnothing(ref.imsk)
    # The result carries the objects of the reduced universe beside the mask.
    @test size(res.pr.X, 2) == length(keep)
    @test length(res.wb.lb) == length(keep)
    return nothing
end

@testset "Hierarchical Risk Parity reduces and expands" begin
    # One measure.
    test_hierarchical_parity(optimise(HierarchicalRiskParity(; opt = optn), rd),
                             optimise(HierarchicalRiskParity(; opt = optk), rdk))
    # And a vector of them, which takes the scalarised branch.
    rs = [Variance(), ConditionalValueatRisk()]
    test_hierarchical_parity(optimise(HierarchicalRiskParity(; r = rs, opt = optn), rd),
                             optimise(HierarchicalRiskParity(; r = rs, opt = optk), rdk))
    # The clustering is built from the reduced prior, so it holds one leaf per live asset.
    res = optimise(HierarchicalRiskParity(; opt = optn), rd)
    @test length(assignments(res.clr)) == length(keep)
end

@testset "Hierarchical Equal Risk Contribution reduces and expands" begin
    test_hierarchical_parity(optimise(HierarchicalEqualRiskContribution(; opt = optn), rd),
                             optimise(HierarchicalEqualRiskContribution(; opt = optk), rdk))
    test_hierarchical_parity(optimise(HierarchicalEqualRiskContribution(; ri = [Variance()],
                                                                        ro = Variance(),
                                                                        opt = optn), rd),
                             optimise(HierarchicalEqualRiskContribution(; ri = [Variance()],
                                                                        ro = Variance(),
                                                                        opt = optk), rdk))
end

@testset "Schur Complement HRP reduces and expands" begin
    # One parameter bundle.
    test_hierarchical_parity(optimise(SchurComplementHierarchicalRiskParity(; opt = optn),
                                      rd),
                             optimise(SchurComplementHierarchicalRiskParity(; opt = optk),
                                      rdk))
    # And a vector of them, which blends over portfolios.
    ps = [SchurComplementParams(; gamma = 0.5), SchurComplementParams(; gamma = 0.25)]
    test_hierarchical_parity(optimise(SchurComplementHierarchicalRiskParity(; params = ps,
                                                                            opt = optn),
                                      rd),
                             optimise(SchurComplementHierarchicalRiskParity(; params = ps,
                                                                            opt = optk),
                                      rdk))
end

@testset "Nested Clustered reduces once, and the cluster slice indexes the reduced axis" begin
    # The reduction happens before the clustering, so no cluster holds a non-investable
    # asset and every `port_opt_view(opti, cl, X)` below indexes the reduced universe.
    res = optimise(NestedClustered(; pe = prn, opti = HierarchicalRiskParity(),
                                   opto = HierarchicalRiskParity()), rd)
    ref = optimise(NestedClustered(; pe = prk, opti = HierarchicalRiskParity(),
                                   opto = HierarchicalRiskParity()), rdk)
    test_hierarchical_parity(res, ref)
    # Every inner result is a result of its own cluster of live assets, so the widths sum to
    # the reduced universe rather than to the full one.
    @test sum(length(r.w) for r in res.resi) == length(keep)
end

@testset "A pre-fitted clustering of the wrong width is refused" begin
    # `clusterise` returns a fitted result unchanged and nothing slices its leaf order, so a
    # clustering of the full universe against a reduced one would index the wrong columns.
    clr = optimise(HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = pr)), rd).clr
    @test length(assignments(clr)) == N
    # The same clustering is correct when nothing is reduced.
    res = optimise(HierarchicalRiskParity(;
                                          opt = HierarchicalOptimiser(; pe = pr, cle = clr)),
                   rd)
    @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
    # And refused when the mask has narrowed the universe under it.
    for opt in
        (HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = prn, cle = clr)),
         HierarchicalEqualRiskContribution(;
                                           opt = HierarchicalOptimiser(; pe = prn,
                                                                       cle = clr)),
         SchurComplementHierarchicalRiskParity(;
                                               opt = HierarchicalOptimiser(; pe = prn,
                                                                           cle = clr)),
         NestedClustered(; pe = prn, cle = clr, opti = HierarchicalRiskParity(),
                         opto = HierarchicalRiskParity()))
        @test_throws DimensionMismatch optimise(opt, rd)
    end
end

@testset "The hierarchical all-investable path is the path it was" begin
    # No mask, no reduction, no expansion, and every result of the family says so the same
    # way.
    for opt in (HierarchicalRiskParity(; opt = optk),
                HierarchicalEqualRiskContribution(; opt = optk),
                SchurComplementHierarchicalRiskParity(; opt = optk),
                NestedClustered(; pe = prk, opti = HierarchicalRiskParity(),
                                opto = HierarchicalRiskParity()))
        res = optimise(opt, rdk)
        @test isnothing(res.imsk)
        @test length(res.w) == length(keep)
        @test isapprox(sum(res.w), 1; rtol = 1e-10)
    end
end

@testset "The reduction verb takes a hierarchical head" begin
    # One verb, bound to the root, so the head itself is what reduces outside the JuMP
    # families: its own `port_opt_view` slices every estimator it holds.
    hrp = HierarchicalRiskParity(; opt = optn)
    imsk, pro, hrpo, rdo = PortfolioOptimisers.investable_reduction(prn, hrp, rd)
    @test imsk == BitVector([1, 1, 0, 1, 1])
    @test size(pro.X, 2) == length(keep)
    @test size(rdo.X, 2) == length(keep)
    @test size(hrpo.opt.pe.X, 2) == length(keep)
    @test isa(hrpo, HierarchicalRiskParity)
    # And the `nothing` method returns the head untouched.
    hrp = HierarchicalRiskParity(; opt = optk)
    imsk, pro, hrpo, rdo = PortfolioOptimisers.investable_reduction(prk, hrp, rdk)
    @test isnothing(imsk)
    @test pro === prk
    @test hrpo === hrp
    @test rdo === rdk
end

# The Non-Investable Axis, issue #911 and ADR 0124.
#
# Every name-keyed estimator but the fee is resolved AFTER the door, against a `sets` the
# view has already narrowed. A name the caller stated over the universe they were given is
# gone from it by then, so the term was dropped under `strict = false` and refused with an
# `ArgumentError` under `strict = true` — on a name that was correct when it was written,
# for a departure the caller could not foresee.
#
# The door now declares the departed names on the sets under `nikey`, after it takes the
# view. Resolution finds them there and skips them in silence; a name on neither axis is
# still refused, because that is what `strict` is for.

@testset "The Non-Investable Axis is minted, validated, and dropped by a view" begin
    # Bare and unique: a departure happens once, and an asset is investable or it is not.
    @test_throws ArgumentError UniverseSets(;
                                            dict = Dict("nx" => ["a", "b"],
                                                        "ni" => ["c", "c"]))
    @test_throws ArgumentError UniverseSets(;
                                            dict = Dict("nx" => ["a", "b", "c"],
                                                        "ni" => ["c"]))
    # The seventh prefix joins the mutual-exclusion grammar.
    @test_throws ArgumentError UniverseSets(; xkey = "ni", nikey = "n",
                                            dict = Dict("ni" => ["a", "b"]))

    sred = UniverseSets(; dict = Dict("nx" => nx[keep]))
    # Minting declares the axis; an empty complement declares nothing, because nothing left.
    smint = PortfolioOptimisers.non_investable_sets(sred, [nx[k]])
    @test smint.dict["ni"] == [nx[k]]
    @test smint.nikey == "ni"
    @test PortfolioOptimisers.non_investable_sets(sred, String[]) === sred
    @test isnothing(PortfolioOptimisers.non_investable_sets(nothing, [nx[k]]))
    # The mask is the truth inside a door: a hand-authored axis is overwritten, not merged.
    shand = UniverseSets(; dict = Dict("nx" => nx[keep], "ni" => ["zz"]))
    @test PortfolioOptimisers.non_investable_sets(shand, [nx[k]]).dict["ni"] == [nx[k]]

    # A view drops the axis, matched EXACTLY: only a door mints one, and a plain group whose
    # name merely starts with the key is not swept away with it.
    slook = UniverseSets(;
                         dict = Dict("nx" => nx[keep], "ni" => [nx[k]],
                                     "nikkei225" => ["a", "b"]))
    sview = PortfolioOptimisers.port_opt_view(slook, [1, 2])
    @test !haskey(sview.dict, "ni")
    @test sview.dict["nikkei225"] == ["a", "b"]
    @test sview.dict["nx"] == ["a", "b"]
    @test sview.nikey == "ni"
end

@testset "A door declares the axis, and announces the departure once" begin
    sets = UniverseSets(; dict = Dict("nx" => nx))
    # The prior-fitting door.
    opt = JuMPOptimiser(; pe = prn, sets = sets, slv = slv)
    _, _, opto, _ = @test_logs (:info,) PortfolioOptimisers.investable_reduction(prn, opt,
                                                                                 rd)
    @test opto.sets.dict["ni"] == [nx[k]]
    @test opto.sets.dict["nx"] == nx[keep]
    # A head that reaches its sets through a nested optimiser is served by a forwarder.
    hrp = HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = prn, sets = sets))
    _, _, hrpo, _ = PortfolioOptimisers.investable_reduction(prn, hrp, rd)
    @test hrpo.opt.sets.dict["ni"] == [nx[k]]
    # The all-investable path declares nothing and says nothing.
    optf = JuMPOptimiser(; pe = prk, sets = UniverseSets(; dict = Dict("nx" => nx[keep])),
                         slv = slv)
    _, _, optfo, _ = PortfolioOptimisers.investable_reduction(prk, optf, rdk)
    @test !haskey(optfo.sets.dict, "ni")
    # The prior-free door is a door too: a Coverage Universe declares the axis as well.
    Xc = collect(X)
    Xc[:, k] .= NaN
    rdc = ReturnsResult(; nx = nx, X = Xc)
    ew = EqualWeighted(; sets = sets)
    _, ewo, _ = PortfolioOptimisers.coverage_reduction(ew, rdc)
    @test ewo.sets.dict["ni"] == [nx[k]]
end

@testset "A departed name resolves in silence, and a typo still refuses" begin
    sets = UniverseSets(;
                        dict = Dict("nx" => nx[keep], "ni" => [nx[k]],
                                    "grp" => [nx[1], nx[k]]))
    # The name is on the counterpart axis, so it is known-good rather than mistyped: no
    # term is written, nothing is logged, and `strict` does not refuse.
    @test PortfolioOptimisers.estimator_to_val(Dict(nx[k] => 0.5), sets, nothing, nothing;
                                               strict = true) == zeros(length(keep))
    # A group whose departed members are all accounted for is silent, and its live members
    # still bind.
    @test PortfolioOptimisers.estimator_to_val(Dict("grp" => 0.3), sets, nothing, nothing;
                                               strict = true) == [0.3, 0.0, 0.0, 0.0]
    # A name on neither axis is what `strict` exists for, and it is refused as before.
    @test_throws ArgumentError PortfolioOptimisers.estimator_to_val(Dict("zz" => 0.5), sets,
                                                                    nothing, nothing;
                                                                    strict = true)
    # The relation is symmetric: resolving ON the axis, a name that stayed is skipped.
    @test PortfolioOptimisers.estimator_to_val(Dict(nx[k] => 0.5, nx[1] => 0.9), sets,
                                               nothing, "ni"; strict = true) == [0.5]
end

@testset "A constraint naming a departed asset no longer refuses under strict" begin
    # Resolution is what #911 is about, so each builder is asked directly, on the sets a
    # door leaves. Under `strict = true` every one of these raised an `ArgumentError` on a
    # name that was correct over the universe the caller was handed.
    sred = UniverseSets(; dict = Dict("nx" => nx[keep], "ni" => [nx[k]]))
    dep = Dict(nx[k] => 0.3)
    wbr = weight_bounds_constraints(WeightBoundsEstimator(; lb = 0, ub = dep), sred;
                                    N = length(keep), strict = true)
    @test all(iszero, wbr.lb)
    @test all(isone, wbr.ub)
    thr = threshold_constraints(ThresholdEstimator(; val = dep), sred; strict = true)
    @test all(iszero, thr.val)
    tnr = turnover_constraints(TurnoverEstimator(; w = fill(0.2, length(keep)), val = dep),
                               sred; strict = true)
    @test all(iszero, tnr.val)
    rkb = risk_budget_constraints(RiskBudgetEstimator(; val = dep), sred, "nx";
                                  N = length(keep), strict = true)
    @test length(rkb.val) == length(keep)
    # The same names on the same sets, one of them mistyped, is still refused: `strict`
    # keeps the job it exists for.
    @test_throws ArgumentError weight_bounds_constraints(WeightBoundsEstimator(; lb = 0,
                                                                               ub = Dict("zz" =>
                                                                                             0.3)),
                                                         sred; N = length(keep),
                                                         strict = true)

    # And end to end. The bound names the asset that leaves, `strict` is on, and the solve
    # reaches the hand-reduced oracle rather than an `ArgumentError`.
    sets = UniverseSets(; dict = Dict("nx" => nx))
    oracle = optimise(MeanRisk(;
                               opt = JuMPOptimiser(; pe = prk, slv = slv, bgt = 1,
                                                   wb = WeightBounds(; lb = 0, ub = 1))),
                      rdk)
    opt = JuMPOptimiser(; pe = prn, sets = sets, slv = slv, strict = true, bgt = 1,
                        wb = WeightBoundsEstimator(; lb = 0, ub = Dict(nx[k] => 0.3)))
    res = optimise(MeanRisk(; opt = opt), rd)
    @test length(res.w) == length(nx)
    @test iszero(res.w[k])
    @test isapprox(sum(res.w), 1; atol = 1e-8)
    @test isapprox(res.w[keep], oracle.w; rtol = 1e-5)
    optz = JuMPOptimiser(; pe = prn, sets = sets, slv = slv, strict = true, bgt = 1,
                         wb = WeightBoundsEstimator(; lb = 0, ub = Dict("zz" => 0.3)))
    @test_throws ArgumentError optimise(MeanRisk(; opt = optz), rd)
end

@testset "A precomputed linear constraint is refused by name, not by DimensionMismatch" begin
    # `port_opt_view(::LinearConstraint, i)` is deliberately the identity, because position
    # is the only link between a column of `A` and an asset. So the row survives the door at
    # its original width and meets a shorter weight vector. Say so at the seam.
    sets = UniverseSets(; dict = Dict("nx" => nx))
    lc = LinearConstraint(;
                          ineq = PartialLinearConstraint(; A = ones(1, length(nx)),
                                                         B = [1.0]))
    opt = JuMPOptimiser(; pe = prn, sets = sets, slv = slv, lcse = lc, bgt = 1,
                        wb = WeightBounds(; lb = 0, ub = 1))
    @test_throws DimensionMismatch optimise(MeanRisk(; opt = opt), rd)
    # The same row over the universe the door leaves is not this failure, and passes.
    lck = LinearConstraint(;
                           ineq = PartialLinearConstraint(; A = ones(1, length(keep)),
                                                          B = [1.0]))
    PortfolioOptimisers.assert_investable_constraint_width(lck, length(keep), "lcse")
    @test isnothing(PortfolioOptimisers.assert_investable_constraint_width(nothing,
                                                                           length(keep),
                                                                           "lcse"))
end

@testset "Every head that owns a sets declares the axis, and one that owns none is untouched" begin
    # The verb is written per type, so every head that carries a `UniverseSets` needs its
    # own line and the generic method must leave everything else alone. One assertion per
    # method, because a head that silently keeps the caller's sets refuses the name that
    # this whole path exists to accept.
    sets = UniverseSets(; dict = Dict("nx" => nx[keep]))
    ni = [nx[k]]
    mrk = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
    heads = (JuMPOptimiser(; pe = prk, sets = sets, slv = slv),
             HierarchicalOptimiser(; pe = prk, sets = sets),
             HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = prk, sets = sets)),
             HierarchicalEqualRiskContribution(;
                                               opt = HierarchicalOptimiser(; pe = prk,
                                                                           sets = sets)),
             SchurComplementHierarchicalRiskParity(;
                                                   opt = HierarchicalOptimiser(; pe = prk,
                                                                               sets = sets)),
             NestedClustered(; pe = prk, sets = sets, opti = mrk, opto = mrk),
             Stacking(; pe = prk, sets = sets, opti = [mrk], opto = mrk),
             SubsetResampling(; pe = prk, sets = sets, opt = mrk),
             InverseVolatility(; pe = prk, sets = sets), EqualWeighted(; sets = sets),
             RandomWeighted(; sets = sets),
             RiskBudgeting(; opt = JuMPOptimiser(; pe = prk, slv = slv),
                           rba = AssetRiskBudgeting(; sets = sets)),
             RelaxedRiskBudgeting(; opt = JuMPOptimiser(; pe = prk, slv = slv),
                                  rba = AssetRiskBudgeting(; sets = sets)))
    function head_sets(h)
        return if hasproperty(h, :rba)
            h.rba.sets
        elseif hasproperty(h, :sets) && isa(getfield(h, :sets), UniverseSets)
            h.sets
        else
            h.opt.sets
        end
    end
    for h in heads
        @test head_sets(PortfolioOptimisers.non_investable_universe(h, ni)).dict["ni"] == ni
    end
    # A head that carries no sets is returned untouched by the generic method.
    @test PortfolioOptimisers.non_investable_universe(mrk, ni) === mrk
    # And an unnamed returns result mints nothing, because names are what the axis is.
    @test isempty(PortfolioOptimisers.non_investable_names(nothing, BitVector([1, 0, 1])))
    # The vector method of the width guard walks its elements.
    lck = LinearConstraint(;
                           ineq = PartialLinearConstraint(; A = ones(1, length(keep)),
                                                          B = [1.0]))
    @test isnothing(PortfolioOptimisers.assert_investable_constraint_width([lck, lck],
                                                                           length(keep),
                                                                           "gcarde"))
    lcn = LinearConstraint(;
                           eq = PartialLinearConstraint(; A = ones(1, length(nx)),
                                                        B = [1.0]))
    @test_throws DimensionMismatch PortfolioOptimisers.assert_investable_constraint_width([lcn],
                                                                                          length(keep),
                                                                                          "gcarde")
end

@testset "HighOrderFactorPriorEstimator reduces before the lift and expands its co-moments" begin
    # Issue #923. The lift squares the reach of a non-investable asset: one `NaN` loading row
    # makes a `NaN` band of `kron(M, M)`, and `kM * f_kt * transpose(kM)` carries it across
    # 369 of the 625 cokurtosis entries of a five-asset panel. `matrix_processing!` then
    # refused the whole fit with `ArgumentError: matrix contains Infs or NaNs`, which names
    # LAPACK rather than the asset that caused it.
    #
    # The gapped panel a delisting makes. `FactorPrior` reduces to the Coverage Universe and
    # expands, so its `rr.M` carries the `NaN` row that the lift squares.
    Xg = copy(X)
    Xg[141:end, k] .= NaN
    # The fourth-moment indices that name the non-investable asset, and its own row.
    kpairs = [(a - 1) * N + b for a in 1:N for b in 1:N if a == k || b == k]
    kmask = falses(N^2)
    kmask[kpairs] .= true

    pe = HighOrderFactorPriorEstimator()
    hop = prior(pe, Xg, F; dims = 1)
    hok = prior(pe, Xg[:, keep], F; dims = 1)

    # The carrier lives on the FULL asset universe, and the mask is still derivable from it.
    @test length(hop.mu) == N
    @test PortfolioOptimisers.investable_mask(hop) == BitVector([1, 1, 0, 1, 1])
    # The structure matrices are sized from the full asset count, which is what the
    # constructor validates the expanded `kt` against.
    @test size(hop.kt) == (N^2, N^2)
    @test size(hop.sk) == (N, N^2)
    @test size(hop.V) == (N, N)
    @test size(hop.L2) == size(hop.S2) == (div(N * (N + 1), 2), N^2)
    @test size(hop.D2) == (N^2, div(N * (N + 1), 2))

    # The `NaN` reaches exactly the fourth-moment indices of the non-investable asset, and
    # nothing else. That is the carrier `HighOrderPriorEstimator` makes on the same panel.
    @test [!isfinite(hop.kt[i, j]) for i in 1:(N ^ 2), j in 1:(N ^ 2)] ==
          [kmask[i] || kmask[j] for i in 1:(N ^ 2), j in 1:(N ^ 2)]
    @test count(!isfinite, hop.kt) == 369
    @test [!isfinite(hop.sk[i, j]) for i in 1:N, j in 1:(N ^ 2)] ==
          [i == k || kmask[j] for i in 1:N, j in 1:(N ^ 2)]
    @test count(!isfinite, hop.sk) == 61
    @test [!isfinite(hop.V[i, j]) for i in 1:N, j in 1:N] ==
          [i == k || j == k for i in 1:N, j in 1:N]
    @test count(!isfinite, hop.V) == 9

    # The oracle is the same fit with the non-investable asset removed by hand, and the
    # reduction of the returned carrier must reach it block for block, at 0.0.
    v = PortfolioOptimisers.port_opt_view(hop, keep)
    for f in (:mu, :sigma, :X, :kt, :sk, :V)
        @test all(isfinite, getproperty(v, f))
        @test getproperty(v, f) == getproperty(hok, f)
    end

    # The factor block sits on the factor axis, which the reduction does not touch, and the
    # two routes to the low order factor prior are still the same object.
    @test hop.fpr.pr === hop.pr.fpr
    @test hop.f_kt == hok.f_kt
    @test hop.f_sk == hok.f_sk
    @test hop.f_V == hok.f_V

    # Every co-moment configuration reduces and expands: the residual correction reads the
    # reconstruction error and the wrapped declaration, both of which follow the reduction,
    # and either estimator set to `nothing` drops its moment from both ends.
    for cpe in (HighOrderFactorPriorEstimator(; rsd = false),
                HighOrderFactorPriorEstimator(; ske = nothing),
                HighOrderFactorPriorEstimator(; kte = nothing))
        chop = prior(cpe, Xg, F; dims = 1)
        chok = prior(cpe, Xg[:, keep], F; dims = 1)
        cv = PortfolioOptimisers.port_opt_view(chop, keep)
        @test length(chop.mu) == N
        for f in (:mu, :sigma, :X)
            @test getproperty(cv, f) == getproperty(chok, f)
        end
        for f in (:kt, :sk, :V)
            a, b = getproperty(cv, f), getproperty(chok, f)
            @test isnothing(a) == isnothing(b)
            if !isnothing(a)
                @test all(isfinite, a)
                @test a == b
            end
        end
    end

    # The all-investable path is the path it was: the `nothing` sentinel reduces nothing,
    # expands nothing, and the co-moments are finite throughout.
    hoa = prior(pe, X, F; dims = 1)
    @test isnothing(PortfolioOptimisers.investable_mask(hoa))
    @test all(isfinite, hoa.kt)
    @test all(isfinite, hoa.sk)
    @test all(isfinite, hoa.V)
    @test size(hoa.kt) == (N^2, N^2)
end

# ---------------------------------------------------------------------------------------
# Issue #922: an entropy pooling view is built on the investable universe.
#
# A view is a dense linear form over the asset axis, and `0 * NaN` is `NaN`, so before this
# a view naming only LIVE assets came back an all-`NaN` constraint row and the solve failed
# naming the solver. The builders now receive a carrier reduced by `port_opt_view` and sets
# carrying the Non-Investable Axis, so a row is written over the investable columns alone.
#
# Nothing is expanded: an `epc` row runs over observations, and the moments come from the
# refit wrapped prior, which already carries the full-universe frame. The oracle is
# therefore the same fit with the dead column removed by hand, and the parity is exact.
# ---------------------------------------------------------------------------------------

# A real gapped panel: `c` delists after observation 150, in the returns and in the active
# mask alike, so `EmpiricalPrior` mints the `NaN` column itself.
Xd = copy(X)
damsk = trues(T, N)
Xd[151:end, k] .= NaN
damsk[151:end, k] .= false
rdd = ReturnsResult(; nx = nx, X = Xd,
                    pnl = AssetPanel(;
                                     pf = [NumericPanelField(; name = "mcap",
                                                             vals = ones(T, N))],
                                     amsk = damsk, emsk = damsk))
rddk = ReturnsResult(; nx = nx[keep], X = Xd[:, keep],
                     pnl = AssetPanel(;
                                      pf = [NumericPanelField(; name = "mcap",
                                                              vals = ones(T, length(keep)))],
                                      amsk = damsk[:, keep], emsk = damsk[:, keep]))
setsd = UniverseSets(; dict = Dict("nx" => nx))
setsdk = UniverseSets(; dict = Dict("nx" => nx[keep]))
epview = LinearConstraintEstimator(; val = "a == 0.001")

@testset "The counterpart axis sheds a group's departed members, before the spread" begin
    other = ["c"]
    led = String[]
    shed = PortfolioOptimisers.shed_departed_members
    # The survivors keep their order, and an empty counterpart axis is the identity that
    # records nothing.
    @test shed(["a", "c", "b"], other, led, "grp", "grp == 1") == ["a", "b"]
    @test length(led) == 1
    @test shed(["a", "b"], String[], led, "grp", "grp == 1") == ["a", "b"]
    @test length(led) == 1
    # A group that lost every member keeps the first of them, so the row it leaves names a
    # departed asset rather than nothing at all.
    @test shed(["c"], other, led, "grp", "grp == 1") == ["c"]
    @test occursin("lost every member", led[end])
    # A pair sheds jointly: a position goes when either of its names did, which is what
    # keeps the two lists the same length.
    m1, m2 = shed(["a", "b"], ["c", "d"], other, nothing, "grp", "e")
    @test m1 == ["b"]
    @test m2 == ["d"]
    m1, m2 = shed(["a", "c"], ["d", "e"], other, nothing, "grp", "e")
    @test m1 == ["a"]
    @test m2 == ["d"]
    # And an all-lost pair group keeps its first pair, on the same reasoning.
    m1, m2 = shed(["c"], ["c"], other, nothing, "grp", "e")
    @test m1 == ["c"]
    @test m2 == ["c"]
    # An empty counterpart axis is the identity for the pair form too, and records nothing.
    m1, m2 = shed(["a", "c"], ["d", "e"], String[], led, "grp", "e")
    @test m1 == ["a", "c"]
    @test m2 == ["d", "e"]
    # The mean divides by the SURVIVING count, so the row still computes what its
    # right-hand side asserts. `bl_flag = true` is the mean expansion.
    sred = UniverseSets(;
                        dict = Dict("nx" => nx[keep], "ni" => [nx[k]],
                                    "grp" => [nx[1], nx[k], nx[2]]))
    res = PortfolioOptimisers.replace_group_by_assets(parse_equation("grp == 0.05"), sred,
                                                      true)
    @test res.vars == [nx[1], nx[2]]
    @test res.coef == [0.5, 0.5]
    # The sum expansion repeats the coefficient over the survivors instead.
    res = PortfolioOptimisers.replace_group_by_assets(parse_equation("grp == 0.05"), sred,
                                                      false, true, false)
    @test res.coef == [1.0, 1.0]
    # A group that loses every member *is* a row naming a departed asset, so it expands to
    # one — not to nothing — and the counterpart rule drops the row whole, in silence, one
    # door later. Expanding to nothing would make it indistinguishable from a caller's
    # constant row, `1 == 0.004`, which must still be diagnosed.
    sall = UniverseSets(; dict = Dict("nx" => nx[keep], "ni" => [nx[k]], "grp" => [nx[k]]))
    res = PortfolioOptimisers.replace_group_by_assets(parse_equation("grp == 0.05"), sall,
                                                      true)
    @test res.vars == [nx[k]]
    @test isnothing(@test_logs PortfolioOptimisers.get_linear_constraints([res], sall;
                                                                          strict = true))
    # The constant row keeps its own diagnosis, under both settings of `strict`.
    const_row = parse_equation("1 == 0.004")
    @test isempty(const_row.vars)
    @test isnothing(@test_logs (:warn,) PortfolioOptimisers.get_linear_constraints([const_row],
                                                                                   sall))
    @test_throws ArgumentError PortfolioOptimisers.get_linear_constraints([const_row], sall;
                                                                          strict = true)
end

@testset "A row naming a departed asset is dropped whole, and a typo still names itself" begin
    sred = UniverseSets(; dict = Dict("nx" => nx[keep], "ni" => [nx[k]]))
    # The whole row goes, not the term: fitting `a + c == 0.05` as `a == 0.05` would assert
    # something the caller never wrote. Silent, and `strict` does not refuse.
    rows = parse_equation(["a + c == 0.05", "b == 0.02"])
    lcs = @test_logs PortfolioOptimisers.get_linear_constraints(rows, sred; strict = true)
    @test size(lcs.eq.A, 1) == 1
    @test lcs.eq.A[1, :] == [0.0, 1.0, 0.0, 0.0]
    @test lcs.eq.B == [0.02]
    # A name on neither axis is a typo, is reported, and now takes its row with it. One
    # warning, not two: the row never reaches the empty-row report.
    zz = parse_equation(["a + zz == 0.05"])
    @test isnothing(@test_logs (:warn,) PortfolioOptimisers.get_linear_constraints(zz,
                                                                                   sred))
    @test_throws ArgumentError PortfolioOptimisers.get_linear_constraints(zz, sred;
                                                                          strict = true)
    # The message says what the failure cost, and the unit differs by shape.
    @test occursin("row dropped",
                   PortfolioOptimisers.unknown_variable_msg("zz", nx[keep], "nx";
                                                            consequence = "row dropped"))
    @test occursin("term dropped",
                   PortfolioOptimisers.unknown_variable_msg("zz", nx[keep], "nx"))
end

@testset "The departure ledger records what a drop cost, and nothing when nobody collects" begin
    led = String[]
    @test isnothing(PortfolioOptimisers.record_non_investable_drop!(nothing, "x"))
    PortfolioOptimisers.record_non_investable_drop!(led, "the row `a == 1`")
    @test led == ["the row `a == 1`"]
    # A shed of nothing records nothing, so the all-investable path costs one comparison.
    PortfolioOptimisers.record_group_shed!(led, "grp", 0, 3, "grp == 1")
    @test length(led) == 1
    # Some members lost: the row survives, and the ledger counts them.
    PortfolioOptimisers.record_group_shed!(led, "grp", 1, 2, "grp == 1")
    @test occursin("1 departed member(s) of the group `grp`", led[2])
    # Every member lost: the row went, and the ledger says so instead.
    PortfolioOptimisers.record_group_shed!(led, "grp", 3, 0, "grp == 1")
    @test occursin("lost every member", led[3])
    @test isnothing(PortfolioOptimisers.record_group_shed!(nothing, "grp", 1, 1, "e"))
end

@testset "The entropy pooling door reduces, and no view is expanded back" begin
    prd = prior(EmpiricalPrior(), rdd)
    @test PortfolioOptimisers.investable_mask(prd) == BitVector([1, 1, 0, 1, 1])
    # The door hands back the Investable Mask itself, not the positions: it is what
    # `investable_prior` views at and what `expand_moment` writes back through, so a caller
    # that reduces and expands — the Black-Litterman family does both — needs nothing else.
    imsk, vsets, ni = PortfolioOptimisers.investable_views(prd, setsd)
    @test imsk == BitVector([1, 1, 0, 1, 1])
    @test findall(imsk) == keep
    @test ni == [nx[k]]
    @test vsets.dict["nx"] == nx[keep]
    @test vsets.dict["ni"] == [nx[k]]
    # The prior is viewed at the same mask, and the view is exact.
    vpr = PortfolioOptimisers.investable_prior(imsk, prd)
    @test vpr.mu == prd.mu[keep]
    @test vpr.sigma == prd.sigma[keep, keep]
    # The all-investable path returns its arguments untouched, and views nothing.
    prk2 = prior(EmpiricalPrior(), rddk)
    imskk, vsetsk, nik = PortfolioOptimisers.investable_views(prk2, setsdk)
    @test isnothing(imskk)
    @test vsetsk === setsdk
    @test isempty(nik)
    @test PortfolioOptimisers.investable_prior(imskk, prk2) === prk2
    # Sets that do not cover the fitted universe are named, not a BoundsError.
    @test_throws DimensionMismatch PortfolioOptimisers.investable_views(prd, setsdk)
    # No sets is no views, so there is nothing to reduce for.
    imskn, vsetsn, nin = PortfolioOptimisers.investable_views(prd, nothing)
    @test isnothing(imskn)
    @test isnothing(vsetsn)
    @test isempty(nin)
end

@testset "An entropy pooling view on live assets reaches the hand-reduced fit exactly" begin
    # This is the ticket's defect: on `dev` the row came back all-`NaN` and the solve raised
    # `ErrorException` blaming the caller's views. The parity is exact, not approximate,
    # because the reduction is a slice and the solve sees identical numbers.
    oracle = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = epview,
                                       sets = setsdk), rddk)
    gapped = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = epview,
                                       sets = setsd), rdd)
    @test gapped.w == oracle.w
    @test gapped.ens == oracle.ens
    @test gapped.kld == oracle.kld
    # The posterior lives on the full universe, and the departed asset keeps its `NaN`.
    @test length(gapped.mu) == N
    @test isnan(gapped.mu[k])
    @test PortfolioOptimisers.investable_mask(gapped) == BitVector([1, 1, 0, 1, 1])
    # The Meucci route takes the same door.
    moracle = prior(MeucciEntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = epview,
                                              sets = setsdk), rddk)
    mgapped = prior(MeucciEntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = epview,
                                              sets = setsd), rdd)
    @test mgapped.w == moracle.w
end

@testset "A view naming a departed asset drops its row, and strict does not refuse" begin
    oracle = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = epview,
                                       sets = setsdk), rddk)
    # The departed row goes; the live row is fitted, and the answer is the oracle's.
    both = LinearConstraintEstimator(; val = ["a == 0.001", "c == 0.002"])
    res = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = both, sets = setsd),
                rdd)
    @test res.w == oracle.w
    # `strict` refuses a typo and not a departure, which is the whole of the distinction.
    @test res.w ==
          prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = both, sets = setsd),
                rdd; strict = true).w
    typo = LinearConstraintEstimator(; val = ["a == 0.001", "zz == 0.002"])
    @test_throws ArgumentError prior(EntropyPoolingPrior(; pe = EmpiricalPrior(),
                                                         mu_views = typo, sets = setsd),
                                     rdd; strict = true)
    # A joint row goes whole rather than being fitted without its departed leg.
    joint = LinearConstraintEstimator(; val = "a + c == 0.001")
    viewless = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = joint,
                                         sets = setsd), rdd)
    @test viewless.kld == 0
    @test viewless.w == fill(1 / T, T)
    # A group sheds its departed member instead, and the surviving sum is the oracle.
    setsg = UniverseSets(; dict = Dict("nx" => nx, "grp" => [nx[1], nx[k]]))
    grouped = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(),
                                        mu_views = LinearConstraintEstimator(;
                                                                             val = "grp == 0.002"),
                                        sets = setsg), rdd)
    goracle = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(),
                                        mu_views = LinearConstraintEstimator(;
                                                                             val = "a == 0.002"),
                                        sets = setsdk), rddk)
    @test grouped.w == goracle.w
end

@testset "The door reports once, naming its own process and what the departure cost" begin
    # A standalone prior fit used to say nothing at all: only an optimisation door
    # announced. It now names itself, and the casualties its departures caused.
    both = LinearConstraintEstimator(; val = ["a == 0.001", "c == 0.002"])
    logs, _ = Test.collect_test_logs() do
        return prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = both,
                                         sets = setsd), rdd)
    end
    msgs = [string(l.message) for l in logs]
    @test length(msgs) == 1
    @test occursin("entropy pooling fit", msgs[1])
    @test occursin("[\"c\"]", msgs[1])
    @test occursin("the row `c == 0.002`", msgs[1])
    @test logs[1].level == Logging.Info
    # A departure that takes the LAST view changes the model rather than trimming it, so
    # the one report is raised to a warning and says the posterior is unconditioned.
    joint = LinearConstraintEstimator(; val = "a + c == 0.001")
    logs, _ = Test.collect_test_logs() do
        return prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = joint,
                                         sets = setsd), rdd)
    end
    @test length(logs) == 1
    @test logs[1].level == Logging.Warn
    @test occursin("no view at all", string(logs[1].message))
    # A shed group is counted rather than named: the departed asset is named once already.
    setsg = UniverseSets(; dict = Dict("nx" => nx, "grp" => [nx[1], nx[k]]))
    logs, _ = Test.collect_test_logs() do
        return prior(EntropyPoolingPrior(; pe = EmpiricalPrior(),
                                         mu_views = LinearConstraintEstimator(;
                                                                              val = "grp == 0.002"),
                                         sets = setsg), rdd)
    end
    @test occursin("1 departed member(s) of the group `grp`", string(logs[1].message))
    # The all-investable path says nothing, and pays nothing.
    logs, _ = Test.collect_test_logs() do
        return prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), mu_views = epview,
                                         sets = setsdk), rddk)
    end
    @test isempty(logs)
end

@testset "announce_non_investable names the process and the consequence it is given" begin
    logs, _ = Test.collect_test_logs() do
        return PortfolioOptimisers.announce_non_investable(["c"], ["the row `x`"],
                                                           "widget fit", "Nothing breaks.")
    end
    msg = string(logs[1].message)
    @test occursin("excluded from this widget fit", msg)
    @test occursin("Nothing breaks.", msg)
    @test occursin("Dropped over them: the row `x`.", msg)
    @test logs[1].level == Logging.Info
    # The optimisation door's wording is the default, so it reads as it always did.
    logs, _ = Test.collect_test_logs() do
        return PortfolioOptimisers.announce_non_investable(["c"])
    end
    @test occursin("excluded from this optimisation", string(logs[1].message))
    @test !occursin("Dropped over them", string(logs[1].message))
    # Nothing left, nothing said.
    logs, _ = Test.collect_test_logs() do
        return PortfolioOptimisers.announce_non_investable(String[], ["the row `x`"])
    end
    @test isempty(logs)
end

@testset "A prior reference and a pair group shed their departed members too" begin
    setsp = UniverseSets(; dict = Dict("nx" => nx, "grp" => [nx[1], nx[k]]))
    # `prior(grp)` expands exactly as the plain group does, and sheds the same members
    # before its coefficient is spread, so the reference resolves over the survivors.
    res = PortfolioOptimisers.replace_group_by_assets(parse_equation("prior(grp) + b == 0.002"),
                                                      UniverseSets(;
                                                                   dict = Dict("nx" =>
                                                                                   nx[keep],
                                                                               "ni" =>
                                                                                   [nx[k]],
                                                                               "grp" =>
                                                                                   [nx[1],
                                                                                    nx[k]])),
                                                      false, true, false)
    @test "prior($(nx[1]))" ∈ res.vars
    @test "prior($(nx[k]))" ∉ res.vars
    # A pair group sheds jointly, so the two sides stay the same length and every surviving
    # pair still names two live assets.
    sredp = UniverseSets(;
                         dict = Dict("nx" => nx[keep], "ni" => [nx[k]],
                                     "gA" => [nx[1], nx[k]], "gB" => [nx[2], nx[4]]))
    res = PortfolioOptimisers.replace_group_by_assets(parse_equation("(gA, gB) == 0.1"),
                                                      sredp, false, true, true)
    @test res.vars == ["([$(nx[1])], [$(nx[2])])"]
    # And end to end: a correlation view over groups, one of whose members delisted, equals
    # the same view stated over the survivors on the hand-reduced panel.
    setsg = UniverseSets(;
                         dict = Dict("nx" => nx, "gA" => [nx[1], nx[k]],
                                     "gB" => [nx[2], nx[4]]))
    setsgk = UniverseSets(; dict = Dict("nx" => nx[keep], "gA" => [nx[1]], "gB" => [nx[2]]))
    rv = LinearConstraintEstimator(; val = "(gA, gB) == 0.1")
    gapped = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), rho_views = rv,
                                       sets = setsg), rdd)
    oracle = prior(EntropyPoolingPrior(; pe = EmpiricalPrior(), rho_views = rv,
                                       sets = setsgk), rddk)
    @test gapped.w == oracle.w
end

# ---------------------------------------------------------------------------------------
# Issue #921: the Black-Litterman family reduces once at its entry and expands its
# posterior back to the full universe.
#
# A view is a dense linear form over the asset axis, and `0 * NaN` is `NaN`, so before this
# a departed asset poisoned `omega` and with it every entry of both posteriors — under a
# view naming only LIVE assets exactly as thoroughly as under one naming the asset that
# left. #919 measured all four members on `dev`: `BlackLittermanPrior` returned an
# all-`NaN` result whose Investable Mask was empty, and the other three raised an unnamed
# `ArgumentError` or `BoundsError` from LAPACK, the regression or `posdef!`.
#
# Unlike the entropy pooling family, this one owes an EXPANSION: an `epc` row runs over
# observations, but a Black-Litterman posterior is a moment pair over the reduced assets,
# and the contract is that a prior result lives on the full asset universe.
#
# The oracle throughout is the same fit with the dead column removed by hand.
# ---------------------------------------------------------------------------------------

# The family needs factors, which `rdd` does not carry.
rddf = ReturnsResult(; nx = nx, X = Xd, nf = nf, F = F,
                     pnl = AssetPanel(;
                                      pf = [NumericPanelField(; name = "mcap",
                                                              vals = ones(T, N))],
                                      amsk = damsk, emsk = damsk))
rddfk = ReturnsResult(; nx = nx[keep], X = Xd[:, keep], nf = nf, F = F,
                      pnl = AssetPanel(;
                                       pf = [NumericPanelField(; name = "mcap",
                                                               vals = ones(T, length(keep)))],
                                       amsk = damsk[:, keep], emsk = damsk[:, keep]))
setsdf = UniverseSets(; dict = Dict("nx" => nx, "nf" => nf))
setsdfk = UniverseSets(; dict = Dict("nx" => nx[keep], "nf" => nf))
blview = LinearConstraintEstimator(; val = ["a == 0.001", "b == 0.0008"])
blfview = LinearConstraintEstimator(; val = ["f1 == 0.002", "f2 == 0.001"])

@testset "A Black-Litterman view on live assets reaches the hand-reduced fit exactly" begin
    # This is the ticket's defect. On `dev` the gapped fit came back all-`NaN` with an empty
    # Investable Mask, and said nothing at all about it. The parity is exact, not
    # approximate, because the reduction is a slice and the master equations see identical
    # numbers.
    oracle = prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = blview,
                                       sets = setsdfk), rddfk)
    gapped = prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = blview,
                                       sets = setsdf), rddf)
    @test gapped.mu[keep] == oracle.mu
    @test gapped.sigma[keep, keep] == oracle.sigma
    # The posterior lives on the full universe, and the departed asset keeps its `NaN`, so
    # the next layer derives the same mask this one did.
    @test length(gapped.mu) == N
    @test isnan(gapped.mu[k])
    @test isnan(gapped.sigma[k, k])
    @test PortfolioOptimisers.investable_mask(gapped) == BitVector([1, 1, 0, 1, 1])
    # The reduction takes columns and never rows, so the observation axis is untouched and
    # the wrapped weighting still describes the rows of the returned `X`.
    @test size(gapped.X) == (T, N)
end

@testset "A Black-Litterman row naming a departed asset goes whole, and strict does not refuse" begin
    oracle = prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = blview,
                                       sets = setsdfk), rddfk)
    both = LinearConstraintEstimator(; val = ["a == 0.001", "b == 0.0008", "c == 0.002"])
    res = prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = both, sets = setsdf),
                rddf)
    @test res.mu[keep] == oracle.mu
    # `strict` refuses a typo and not a departure, which is the whole of the distinction.
    @test res.mu[keep] ==
          prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = both, sets = setsdf),
                rddf; strict = true).mu[keep]
    typo = LinearConstraintEstimator(; val = ["a == 0.001", "zz == 0.002"])
    @test_throws ArgumentError prior(BlackLittermanPrior(; pe = EmpiricalPrior(),
                                                         views = typo, sets = setsdf), rddf;
                                     strict = true)
    # A joint row goes whole rather than being fitted without its departed leg: `a + c`
    # assembled without `c` would assert `a == 0.001`, which the caller never wrote.
    joint = LinearConstraintEstimator(; val = ["a + c == 0.001", "b == 0.0008"])
    jref = prior(BlackLittermanPrior(; pe = EmpiricalPrior(),
                                     views = LinearConstraintEstimator(;
                                                                       val = ["b == 0.0008"]),
                                     sets = setsdfk), rddfk)
    @test prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = joint, sets = setsdf),
                rddf).mu[keep] == jref.mu
    # A per-view confidence vector keeps its alignment across the dropped row: the dropped
    # index joins `excl`, so `remove_excl_views` drops the matching entry.
    cnf = LinearConstraintEstimator(; val = ["c == 0.002", "a == 0.001", "b == 0.0008"])
    cref = prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = blview,
                                     sets = setsdfk, views_conf = [0.2, 0.7]), rddfk)
    @test prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = cnf, sets = setsdf,
                                    views_conf = [0.9, 0.2, 0.7]), rddf).mu[keep] == cref.mu
    # A group sheds its departed member before the coefficient is spread, so the mean
    # divides by the SURVIVING count.
    setsg = UniverseSets(; dict = Dict("nx" => nx, "nf" => nf, "grp" => [nx[1], nx[k]]))
    grouped = prior(BlackLittermanPrior(; pe = EmpiricalPrior(),
                                        views = LinearConstraintEstimator(;
                                                                          val = "grp == 0.002"),
                                        sets = setsg), rddf)
    goracle = prior(BlackLittermanPrior(; pe = EmpiricalPrior(),
                                        views = LinearConstraintEstimator(;
                                                                          val = "a == 0.002"),
                                        sets = setsdfk), rddfk)
    @test grouped.mu[keep] == goracle.mu
end

@testset "A departure that takes the last view leaves the wrapped prior, and warns" begin
    # ADR 0125's singled-out case: nothing is left to condition on, so the posterior is the
    # wrapped prior and the caller is told, because they have no other way to learn it.
    wrapped = prior(EmpiricalPrior(), rddf)
    onlyc = LinearConstraintEstimator(; val = "c == 0.002")
    viewless = prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = onlyc,
                                         sets = setsdf), rddf)
    @test all(isequal(0), filter(isfinite, viewless.mu .- wrapped.mu))
    @test all(isequal(0), filter(isfinite, viewless.sigma .- wrapped.sigma))
    @test isnan(viewless.mu[k])
    logs, _ = Test.collect_test_logs() do
        return prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = onlyc,
                                         sets = setsdf), rddf)
    end
    warns = filter(l -> l.level == Logging.Warn, logs)
    @test length(warns) == 1
    @test occursin("no view at all", warns[1].message)
    @test occursin("its wrapped prior", warns[1].message)
    @test occursin("the view row `c == 0.002`", warns[1].message)
    # A view set that was empty or mistyped FROM THE START is still a refusal, which is
    # #852's case: no departure wrote into the ledger, so nothing distinguishes it from a
    # caller who stated views the universe never held.
    @test_throws PortfolioOptimisers.IsNothingError prior(BlackLittermanPrior(;
                                                                              pe = EmpiricalPrior(),
                                                                              views = LinearConstraintEstimator(;
                                                                                                                val = "zz == 0.002"),
                                                                              sets = setsdf),
                                                          rddf)
end

@testset "A precomputed view matrix over a gapped universe is refused by name" begin
    # A precomputed `P` resolves no name, so there is no way to tell which of its rows the
    # departed asset belonged to and no way to reduce it. Left alone this is the ticket's
    # defect, unfixed and silent, so it refuses instead.
    pre = PortfolioOptimisers.BlackLittermanViews(; P = [1.0 0 0 0 0; 0 1 0 0 0],
                                                  Q = [0.001, 0.0008])
    @test_throws ArgumentError prior(BlackLittermanPrior(; pe = EmpiricalPrior(),
                                                         views = pre), rddf)
    # And it says nothing on a universe with no gap in it, which is the path it was.
    prek = PortfolioOptimisers.BlackLittermanViews(; P = [1.0 0 0 0; 0 1 0 0],
                                                   Q = [0.001, 0.0008])
    @test all(isfinite,
              prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = prek), rddfk).mu)
end

@testset "The three factor-view members reduce their asset side and expand it back" begin
    # These three write their views on the FACTOR axis, so no view row can be dropped for a
    # departed asset and the counterpart axis never bites. What a departure poisons is the
    # asset side, and it poisons all of it.
    #
    # `BayesianBlackLittermanPrior` inverts the asset covariance twice, so on `dev` this
    # raised an unnamed `ArgumentError` out of LAPACK.
    bo = prior(BayesianBlackLittermanPrior(; pe = FactorPrior(), views = blfview,
                                           sets = setsdfk), rddfk)
    bg = prior(BayesianBlackLittermanPrior(; pe = FactorPrior(), views = blfview,
                                           sets = setsdf), rddf)
    @test isapprox(bg.mu[keep], bo.mu; rtol = 1e-12)
    @test isapprox(bg.sigma[keep, keep], bo.sigma; rtol = 1e-12)
    # The factor block is not expanded, because the reduction never touched the factor axis.
    @test bg.fpr.mu == bo.fpr.mu
    @test length(bg.mu) == N
    @test isnan(bg.mu[k])
    @test PortfolioOptimisers.investable_mask(bg) == BitVector([1, 1, 0, 1, 1])

    # `FactorBlackLittermanPrior` wraps a FACTOR prior, so there is no asset-side prior
    # result to read a mask off: the gap arrives in `X` itself and `coverage_mask` reads it,
    # panel included. On `dev` the `NaN` column reached `StepwiseRegression`, which selected
    # zero factors and raised `BoundsError … at index [1:200, [0]]`.
    fo = prior(FactorBlackLittermanPrior(; views = blfview, sets = setsdfk), rddfk)
    fg = prior(FactorBlackLittermanPrior(; views = blfview, sets = setsdf), rddf)
    @test fg.mu[keep] == fo.mu
    @test fg.sigma[keep, keep] == fo.sigma
    # Every asset-axis block it produced goes back: the loadings, the intercept and the
    # reconstruction, not only the moment pair.
    @test fg.rr.M[keep, :] == fo.rr.M
    @test fg.rr.b[keep] == fo.rr.b
    @test fg.X[:, keep] == fo.X
    @test all(isnan, fg.rr.M[k, :])
    @test size(fg.X) == (T, N)
    # `chol` factorises `sigma`, and a `NaN` frame has no factorisation, so it is dropped on
    # the gapped path and kept on the clean one.
    @test isnothing(fg.chol)
    @test !isnothing(fo.chol)

    # `AugmentedBlackLittermanPrior` stacks `[assets; factors]`, so the reduction applies to
    # the asset half and the truncation back to it is where the expansion goes.
    ao = prior(AugmentedBlackLittermanPrior(; a_views = blview, f_views = blfview,
                                            sets = setsdfk), rddfk)
    ag = prior(AugmentedBlackLittermanPrior(; a_views = blview, f_views = blfview,
                                            sets = setsdf), rddf)
    @test ag.mu[keep] == ao.mu
    @test ag.sigma[keep, keep] == ao.sigma
    @test ag.fpr.mu == ao.fpr.mu
    @test ag.rr.M[keep, :] == ao.rr.M
    @test isnan(ag.mu[k])
    @test size(ag.X) == (T, N)
    # Its asset half drops a departed row whole, like the plain member's.
    withc = LinearConstraintEstimator(; val = ["a == 0.001", "b == 0.0008", "c == 0.002"])
    @test prior(AugmentedBlackLittermanPrior(; a_views = withc, f_views = blfview,
                                             sets = setsdf), rddf).mu[keep] == ao.mu
    # When the departure empties the ASSET half, the stack does NOT collapse to the prior:
    # the factor views the departure never touched still condition the joint posterior, so
    # the answer moves with them.
    onlyc = LinearConstraintEstimator(; val = "c == 0.002")
    emptied = prior(AugmentedBlackLittermanPrior(; a_views = onlyc, f_views = blfview,
                                                 sets = setsdf), rddf)
    other = prior(AugmentedBlackLittermanPrior(; a_views = onlyc,
                                               f_views = LinearConstraintEstimator(;
                                                                                   val = ["f1 == 0.01"]),
                                               sets = setsdf), rddf)
    @test all(isfinite, emptied.mu[keep])
    @test isnan(emptied.mu[k])
    @test !isapprox(emptied.mu[keep], other.mu[keep])
end

@testset "The Black-Litterman all-investable path is the path it was" begin
    # No mask, no reduction, no expansion, no message. `expand_moment`'s `nothing` methods
    # are the identity, so the gap-free fit allocates nothing for a contract it does not use.
    logs, res = Test.collect_test_logs() do
        return prior(BlackLittermanPrior(; pe = EmpiricalPrior(), views = blview,
                                         sets = setsdfk), rddfk)
    end
    @test isnothing(PortfolioOptimisers.investable_mask(res))
    @test isempty(filter(l -> occursin("left the investable universe", string(l.message)),
                         logs))
    # The same holds one member along, where the announcement reads its names off the sets.
    @test isempty(PortfolioOptimisers.investable_universe_names(setsdfk, nothing))
    @test PortfolioOptimisers.investable_universe_names(setsdf,
                                                        BitVector([1, 1, 0, 1, 1])) ==
          [nx[k]]
    # An estimator whose views land on the factor axis need state no asset universe at all,
    # and one that has not stated one has named nobody to report.
    @test isempty(PortfolioOptimisers.investable_universe_names(nothing,
                                                                BitVector([1, 1, 0, 1, 1])))
    # And a universe of the wrong length is the same silence, not a refusal: this reads
    # names for a message, and refusing a message is not its job.
    @test isempty(PortfolioOptimisers.investable_universe_names(setsdfk,
                                                                BitVector([1, 1, 0, 1, 1])))
end

@testset "The Black-Litterman collapse and the empty view block on their own terms" begin
    mu = [0.1, 0.2]
    sigma = [1.0 0.2; 0.2 1.0]
    # No view is the prior pair itself, and NOT the empty-view algebra: `vanilla_posteriors`
    # adds the estimation-error term, so an empty `P` would answer `(1 + tau) * sigma`, a
    # wider covariance produced by views that no longer exist.
    pmu, psigma = PortfolioOptimisers.bl_posteriors(nothing, mu, sigma)
    @test pmu === mu
    @test psigma == sigma
    # The covariance is a copy, because every caller hands it to `matrix_processing!`, which
    # writes in place, and the prior result must not be mutated under a caller holding it.
    @test psigma !== sigma
    psigma[1, 1] = 99
    @test sigma[1, 1] == 1.0
    # The empty block is `0 x n`, so the stack it joins carries no phantom view.
    P, Q, omega = PortfolioOptimisers.bl_view_block(nothing, 2, Float64)
    @test size(P) == (0, 2)
    @test isempty(Q)
    @test size(omega) == (0, 0)
    # The stacked products the augmented member forms are all well defined over it.
    @test size(transpose(P) * (omega \ P)) == (2, 2)
    @test all(iszero, transpose(P) * (omega \ P))
    blp = (; P = [1.0 0.0], Q = [0.5], omega = Diagonal([0.1]), tau = 0.01)
    @test PortfolioOptimisers.bl_view_block(blp, 2, Float64) === (blp.P, blp.Q, blp.omega)
end

@testset "reduce_columns is the inverse of expand_columns" begin
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    msk = BitVector([1, 0, 1])
    @test PortfolioOptimisers.reduce_columns(A, msk) == [1.0 3.0; 4.0 6.0]
    # A copy, not a view: the block goes on to a regression.
    @test isa(PortfolioOptimisers.reduce_columns(A, msk), Matrix)
    # The round trip restores the covered columns and marks the rest.
    B = PortfolioOptimisers.expand_columns(PortfolioOptimisers.reduce_columns(A, msk), msk)
    @test B[:, msk] == A[:, msk]
    @test all(isnan, B[:, 2])
    # `nothing` is the identity on both halves, and the same object comes back.
    @test PortfolioOptimisers.reduce_columns(A, nothing) === A
    @test PortfolioOptimisers.expand_columns(A, nothing) === A
end

@testset "A prior estimator reduces its own per-asset weights" begin
    # `pe.w` is per-asset configuration written against the caller's full universe, and it
    # meets a reduced axis for the same reason a per-asset bound does. Unsliced it raises a
    # bare `DimensionMismatch` from a matrix product that names no asset.
    w = fill(1 / N, N)
    fo = prior(FactorBlackLittermanPrior(; views = blfview, sets = setsdfk, l = 1,
                                         w = fill(1 / N, length(keep))), rddfk)
    fg = prior(FactorBlackLittermanPrior(; views = blfview, sets = setsdf, l = 1, w = w),
               rddf)
    @test fg.mu[keep] == fo.mu
    ao = prior(AugmentedBlackLittermanPrior(; a_views = blview, f_views = blfview,
                                            sets = setsdfk, l = 1,
                                            w = fill(1 / N, length(keep))), rddfk)
    ag = prior(AugmentedBlackLittermanPrior(; a_views = blview, f_views = blfview,
                                            sets = setsdf, l = 1, w = w), rddf)
    @test ag.mu[keep] == ao.mu
    # Both `nothing` sentinels answer the argument they were handed.
    @test PortfolioOptimisers.investable_weights_view(nothing, w) === w
    @test isnothing(PortfolioOptimisers.investable_weights_view(BitVector([1, 1, 0, 1, 1]),
                                                                nothing))
end

@testset "A view row that resolves and cancels is still dropped, and is not a typo" begin
    # The empty-row path narrowed rather than disappeared. A name that does not resolve now
    # takes its row at the name, so the only way left to reach this branch is a row whose
    # names all resolve and whose coefficients annihilate it -- `A - A`, which
    # `parse_equation` folds to a single term with coefficient zero.
    bsets = UniverseSets(; xkey = "nx", dict = Dict("nx" => ["A", "B", "C"]))
    blv = PortfolioOptimisers.get_black_litterman_views(parse_equation(["A - A == 0.0",
                                                                        "B == 0.01"]),
                                                        bsets)
    @test size(blv.P) == (1, 3)
    @test blv.Q == [0.01]
    # It joins `excl`, so a per-view confidence vector keeps its alignment.
    @test blv.excl == [1]
    @test collect(PortfolioOptimisers.remove_excl_views([0.1, 0.2], blv.excl)) == [0.2]
    # And `strict` still refuses it, because nothing about it is a departure.
    @test_throws ArgumentError PortfolioOptimisers.get_black_litterman_views(parse_equation("A - A == 0.0"),
                                                                             bsets;
                                                                             strict = true)
end
