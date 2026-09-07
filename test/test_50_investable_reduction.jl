using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Clarabel

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
