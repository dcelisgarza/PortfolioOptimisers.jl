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
