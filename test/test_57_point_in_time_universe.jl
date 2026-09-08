#=
The closing verification of the point-in-time universe, issue #677.

A `WalkForward` over a synthetic panel with a listing and a delisting inside its window runs
end to end, and every fold reaches the same weights through a JuMP optimiser and through a
hierarchical one. The map's rule is one line -- handle it, or throw a named error -- and this
file is where the whole chain is asked the question at once, rather than one layer at a time.

The identity every fold must satisfy is the one the map fixed in #647 and #669:

 1. The fold's Investable Mask is the Coverage Universe of its own training window: an asset
    whose return is finite at every row of the window, and whose Asset Panel says it is active
    at every row of it.
 2. An asset outside that mask holds **exactly** zero, not a small number.
 3. The assets inside it hold the weights of the same optimisation run on the listed subset
    alone -- the panel with the dead columns removed by hand, weight for weight.

The oracle is the hand-reduced problem rather than the reference implementation, because the
reference reduces in its convex family only: its hierarchical base shadows the input cleaner
with a version that omits the mask, so running it on a gapped panel would measure its defect
and not this port. `test_57b_reference_walk_forward_parity.jl` pins what the reference *can*
answer, and is where the cross-language numbers live.

The panel comes from the generator of map #643 ticket #656 (`test06c_setup.jl`) rather than a
second one written here, as #677 asks.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))
using Clarabel, Statistics, Dates

# Two identical programmes drift apart at the shipped solver defaults, and the drift is the
# solver's rather than the reduction's, so every parity comparison here runs one Clarabel at
# tightened tolerances. See `test_50_investable_reduction.jl`, which sets the same bar.
const PIT_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       check_sol = (; allow_local = true, allow_almost = true),
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                       "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                                       "tol_infeas_abs" => 1e-12,
                                       "tol_infeas_rel" => 1e-12))

# One panel, drawn once. `missing_ratio = 0` keeps every blank cell a listing or a delisting
# rather than a scattered hole, so the mask a fold derives is the listing window and nothing
# else. The two probabilities are raised above their defaults to make the universe move
# inside a 300-observation panel.
const PIT = synthetic_asset_panel(; n_assets = 8, n_observations = 300, n_industries = 3,
                                  late_listing_proba = 0.4, delisting_proba = 0.4,
                                  missing_ratio = 0.0, rng = StableRNG(3))
const PIT_RD = PIT.rd
const PIT_AMSK = PIT_RD.pnl.amsk
const PIT_CV = IndexWalkForward(120, 40)
const PIT_TRAIN = [1:120, 41:160, 81:200, 121:240]

# The Coverage Universe of a window, written out by hand: finite at every row of it, and
# active at every row of it. This is the mask the library derives, re-derived independently
# so no assertion below compares the library against itself.
function pit_coverage(tr)
    return BitVector([all(isfinite, view(PIT_RD.X, tr, j)) && all(view(PIT_AMSK, tr, j))
                      for j in axes(PIT_RD.X, 2)])
end

# The same optimisation on the listed subset alone, with the dead columns removed by hand.
function pit_oracle(opt, tr, keep)
    return optimise(opt, ReturnsResult(; nx = PIT_RD.nx[keep], X = PIT_RD.X[tr, keep])).w
end

@testset "The panel moves its universe inside the walk-forward window" begin
    T, N = size(PIT_RD.X)
    @test (T, N) == (300, 8)
    # Three assets list after the first observation, and three delist before the last, so
    # every fold below sees a different universe.
    @test [findfirst(view(PIT_AMSK, :, j)) for j in 1:N] == [1, 34, 42, 1, 67, 10, 175, 1]
    @test [findlast(view(PIT_AMSK, :, j)) for j in 1:N] ==
          [300, 300, 300, 252, 300, 291, 300, 264]
    # The returns state the same thing the mask does: a blank cell is a `NaN`, not a zero.
    @test all(isequal.(isfinite.(PIT_RD.X), PIT_AMSK))
    # The walk-forward the whole file runs, and the windows it splits into.
    (; train_idx, test_idx) = split(PIT_CV, PIT_RD)
    @test train_idx == PIT_TRAIN
    @test [first(t):last(t) for t in test_idx] == [121:160, 161:200, 201:240, 241:280]
    # A listing and a delisting each land inside a training window, which is what makes this
    # a point-in-time test rather than a static one.
    @test [findall(pit_coverage(tr)) for tr in PIT_TRAIN] ==
          [[1, 4, 8], [1, 2, 4, 6, 8], [1, 2, 3, 4, 5, 6, 8], [1, 2, 3, 4, 5, 6, 8]]
end

@testset "A walk-forward through a JuMP optimiser reduces and expands at every fold" begin
    mr = MeanRisk(; obj = MinimumRisk(), r = Variance(),
                  opt = JuMPOptimiser(; slv = PIT_SLV))
    mpr = PortfolioOptimisers.fit_and_predict(mr, PIT_RD, PIT_CV)
    @test length(mpr.pred) == length(PIT_TRAIN)
    for (pred, tr) in zip(mpr.pred, PIT_TRAIN)
        hand = pit_coverage(tr)
        keep = findall(hand)
        @test isa(pred.res.retcode, PortfolioOptimisers.OptimisationSuccess)
        # 1. The mask is the Coverage Universe of the fold's own training window.
        @test pred.res.imsk == hand
        # 2. The weights come back on the caller's universe, and a dead asset holds an
        #    exact zero rather than a small one.
        @test length(pred.res.w) == size(PIT_RD.X, 2)
        @test all(iszero, view(pred.res.w, .!hand))
        @test isapprox(sum(pred.res.w), 1)
        # 3. The live assets hold the hand-reduced problem's weights.
        @test isapprox(pred.res.w[keep], pit_oracle(mr, tr, keep); rtol = 1e-8)
    end
    # The two ends of the run disagree about the universe, which is the point of the test.
    @test mpr.pred[1].res.imsk == BitVector([1, 0, 0, 1, 0, 0, 0, 1])
    @test mpr.pred[end].res.imsk == BitVector([1, 1, 1, 1, 1, 1, 0, 1])
end

@testset "A walk-forward through the hierarchical families reduces and expands at every fold" begin
    # Every hierarchical head of the library, and the nested-clustered one that slices per
    # cluster, so the contract is checked where the reduction is hardest rather than only
    # where it is easiest.
    for opt in
        (HierarchicalRiskParity(; r = Variance()), HierarchicalEqualRiskContribution(),
         SchurComplementHierarchicalRiskParity(),
         NestedClustered(; opti = InverseVolatility(), opto = InverseVolatility()),
         InverseVolatility())
        mpr = PortfolioOptimisers.fit_and_predict(opt, PIT_RD, PIT_CV)
        @test length(mpr.pred) == length(PIT_TRAIN)
        for (pred, tr) in zip(mpr.pred, PIT_TRAIN)
            hand = pit_coverage(tr)
            keep = findall(hand)
            @test pred.res.imsk == hand
            @test length(pred.res.w) == size(PIT_RD.X, 2)
            @test all(iszero, view(pred.res.w, .!hand))
            @test isapprox(sum(pred.res.w), 1)
            @test isapprox(pred.res.w[keep], pit_oracle(opt, tr, keep); rtol = 1e-10)
        end
    end
end

@testset "The stitched out-of-sample series is finite through both routes" begin
    # The map closes on a run that reaches the end, not only on folds that solve. A test
    # window holds delistings of its own, and the Held Gap filter of ADR 0120 zeroes those
    # weights once, so the series carries a number at every observation.
    for opt in (MeanRisk(; obj = MinimumRisk(), r = Variance(),
                         opt = JuMPOptimiser(; slv = PIT_SLV)),
                HierarchicalRiskParity(; r = Variance()))
        pred = cross_val_predict(opt, PIT_RD, PIT_CV)
        @test length(pred.mrd.X) == 160
        @test all(isfinite, pred.mrd.X)
        # A realised out-of-sample risk exists, which is what a caller runs this for.
        @test isfinite(expected_risk(LowOrderMoment(; alg = SecondMoment()), pred))
    end
end

@testset "A layer that cannot answer refuses by name" begin
    # The map's rule has two halves, and this is the second. A plain moment estimator has no
    # correct answer for a gapped sample, so it refuses rather than returning one.
    Xg = copy(PIT_RD.X)
    @test_throws PortfolioOptimisers.IsNonFiniteError cov(PortfolioOptimisersCovariance(),
                                                          Xg)
    @test_throws PortfolioOptimisers.IsNonFiniteError cov(Covariance(), Xg)
    # The prior is the layer that *can* answer: it reduces to the Coverage Universe, fits
    # there, and writes the gap back as a `NaN` the mask is derived from.
    pr = prior(EmpiricalPrior(), PIT_RD)
    @test PortfolioOptimisers.investable_mask(pr) == pit_coverage(1:size(PIT_RD.X, 1))
    # A window with no live asset has no universe to weight, and the refusal names that.
    Xdead = copy(PIT_RD.X)
    Xdead[1, :] .= NaN
    @test_throws PortfolioOptimisers.IsEmptyError optimise(EqualWeighted(),
                                                           ReturnsResult(; nx = PIT_RD.nx,
                                                                         X = Xdead))
end
