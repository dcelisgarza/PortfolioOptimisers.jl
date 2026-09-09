using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, StatsBase, Statistics, Clarabel

# The scenario fill of a mask-aware prior, issue #858, under ADR 0118.
#
# A mask-aware moment estimator answers a young asset from the observations it has, so the
# asset is investable and its column of the returns matrix still carries a `NaN` at every row
# before it listed. The prior writes `0` there ONCE, so that every consumer -- the JuMP model,
# the hierarchical families and the value-level doors -- reads a finite investable column.
#
# A non-investable asset keeps its whole `NaN` column, because the Investable Mask is derived
# from that gap. `mu` and `sigma` are untouched, because the estimator computed them from the
# rows it saw.
#
# The fill is a trade: a scenario-based measure reads a zero return where the asset had none.
# How much of that trade passes in silence is the FITTING ESTIMATOR'S OWN ANSWER, carried in
# `EmpiricalPrior`'s `fill_limit` field, issue #975. It is silent at or below `fill_limit`,
# warns above it, and refuses any fill under `strict`. `fill_limit` defaults to `nothing`,
# which accepts no share in silence: an investable asset is asked to cover every observation.
#
# `EmpiricalPrior` is the only estimator eligible for the field. A prior is eligible when it
# holds a mask-aware moment estimator DIRECTLY and puts the CALLER'S OWN returns matrix into
# the result's `X`; every other prior wraps an inner prior estimator and synthesises its `X`,
# so the fill is paid once, at the `EmpiricalPrior` at the bottom of the chain.
#
# The reference implementation gives no oracle for the share: it zero-fills its portfolio
# return series at one line and announces nothing, so it has no fill limit and no denominator
# to read off. The denominator is ADR 0118's own -- filled entries over the entries of the
# returns matrix -- and it is pinned numerically below.

const PO = PortfolioOptimisers

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             check_sol = (; allow_local = true, allow_almost = true),
             settings = Dict("verbose" => false))

rng = StableRNG(987654321)
T, N = 60, 4
X0 = randn(rng, T, N) ./ 100 .+ 0.0005
nx = ["a", "b", "c", "d"]

# Asset 4 lists at observation 31, and asset 3 delists at observation 50. The young asset is
# investable under a mask-aware estimator; the dead one is not, whatever the estimator.
Xmix = copy(X0)
Xmix[1:30, 4] .= NaN
Xmix[50:end, 3] .= NaN
amsk_mix = trues(T, N)
amsk_mix[1:30, 4] .= false
amsk_mix[50:end, 3] .= false

# A holiday: asset 2 misses one quote mid-window. The recursion freezes over it, so the asset
# stays investable and exactly one entry is filled.
Xhol = copy(X0)
Xhol[15, 2] = NaN

function make_panel(amsk)
    return AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(T, N))],
                      amsk = amsk, emsk = amsk)
end

pnl_mix = make_panel(amsk_mix)
pnl_full = make_panel(trues(T, N))

me = ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 2)
ce = ExpWeightedCovariance(; decay = 0.9, min_obs = 2, centred = true)
# The estimator that accepts the whole matrix in silence, and the one that accepts none.
pe = EmpiricalPrior(; me = me, ce = ce, fill_limit = 1)
pe0 = EmpiricalPrior(; me = me, ce = ce)

@testset "The fill writes zero at the investable gap alone" begin
    pr = prior(pe, Xmix, nothing, pnl_mix)
    # The young asset is investable, the dead one is not, and the fill did not move the mask.
    @test PO.investable_mask(pr) == BitVector([1, 1, 0, 1])
    @test all(isfinite, view(pr.X, :, [1, 2, 4]))
    @test count(!isfinite, view(pr.X, :, 3)) == 11
    # Zero exactly where the young asset's gap was, and nowhere else.
    @test all(iszero, view(pr.X, 1:30, 4))
    @test view(pr.X, 31:T, 4) == view(Xmix, 31:T, 4)
    @test view(pr.X, :, [1, 2]) == view(Xmix, :, [1, 2])
    @test isequal(view(pr.X, :, 3), view(Xmix, :, 3))
    # `mu` and `sigma` are the estimator's own answer, untouched by the fill.
    @test isequal(pr.mu, vec(Statistics.mean(me, Xmix, pnl_mix; dims = 1)))
    @test isequal(pr.sigma, Statistics.cov(ce, Xmix, pnl_mix; dims = 1))

    # The horizon variant travels the same seam, on the arithmetic returns it carries.
    prh = prior(EmpiricalPrior(; me = me, ce = ce, horizon = 5, fill_limit = 1), Xmix,
                nothing, pnl_mix)
    @test PO.investable_mask(prh) == BitVector([1, 1, 0, 1])
    @test all(isfinite, view(prh.X, :, [1, 2, 4]))
    @test all(iszero, view(prh.X, 1:30, 4))
    @test count(!isfinite, view(prh.X, :, 3)) == 11

    # A plain estimator never reaches the fill: the young asset leaves the Coverage Universe,
    # so its gap belongs to a non-investable column and the matrix is the caller's own.
    pp = @test_logs prior(EmpiricalPrior(), Xmix, nothing, pnl_mix)
    @test PO.investable_mask(pp) == BitVector([1, 1, 0, 0])
    @test pp.X === Xmix

    # A complete window returns the caller's matrix untouched, with no scan of the mask.
    pc = @test_logs prior(pe, X0, nothing, pnl_full)
    @test pc.X === X0
    # The default `fill_limit` reaches the same conclusion by the same short circuit: there is
    # nothing to fill, so there is nothing to name.
    @test (@test_logs prior(pe0, X0, nothing, pnl_full)).X === X0
end

@testset "The share, the diagnostic and the refusal" begin
    # The denominator is the count of entries of the returns matrix, not of the gapped column.
    filled = PO.scenario_fill_pairs(Xmix, BitVector([1, 1, 0, 1]))
    @test length(filled) == 30
    @test length(Xmix) == 240
    @test length(filled) / length(Xmix) == 0.125
    @test unique(last.(filled)) == [4]
    # The dead column is never a pair, whatever its gap.
    @test isempty(PO.scenario_fill_pairs(view(Xmix, :, 3:3), BitVector([0])))

    # Above the estimator's own share: one warning, naming the asset, the count, the limit it
    # was measured against and the consequence.
    pe5 = EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.05)
    @test_logs (:warn, r"Assets \[4\]") prior(pe5, Xmix, nothing, pnl_mix)
    @test_logs (:warn, r"30 \(observation, asset\) pair") prior(pe5, Xmix, nothing, pnl_mix)
    @test_logs (:warn, r"against a limit of 0\.05") prior(pe5, Xmix, nothing, pnl_mix)
    @test_logs (:warn, r"understates its risk") prior(pe5, Xmix, nothing, pnl_mix)
    # The remedy the message names is the field, not a global.
    @test_logs (:warn, r"EmpiricalPrior\(; fill_limit = \.\.\.\)") prior(pe5, Xmix, nothing,
                                                                         pnl_mix)
    # Raise the share above it and the same fit is silent.
    @test_logs prior(EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.2), Xmix, nothing,
                     pnl_mix)

    # TWO PRIORS IN ONE PROGRAM, TWO ANSWERS. This is what the field buys over the global that
    # preceded it: the share travels with the estimator that fills, not with the session.
    loud = EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.05)
    quiet = EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.2)
    @test_logs (:warn, r"Assets \[4\]") prior(loud, Xmix, nothing, pnl_mix)
    @test_logs prior(quiet, Xmix, nothing, pnl_mix)
    @test isequal(prior(quiet, Xmix, nothing, pnl_mix).X,
                  (@test_logs (:warn, r"Assets \[4\]") prior(loud, Xmix, nothing, pnl_mix)).X)

    # One filled entry is 1/240 of the matrix, so a 5% share passes it in silence.
    ph = @test_logs prior(pe5, Xhol, nothing, pnl_full)
    @test iszero(ph.X[15, 2])
    @test count(!isfinite, ph.X) == 0
    @test isnothing(PO.investable_mask(ph))

    # THE DEFAULT. `fill_limit = nothing` accepts no share at all, so a single filled entry is
    # named, and the message says what was asked rather than printing a share no caller chose.
    @test_logs (:warn, r"Assets \[2\]") prior(pe0, Xhol, nothing, pnl_full)
    @test_logs (:warn, r"`fill_limit = nothing`") prior(pe0, Xhol, nothing, pnl_full)
    @test_logs (:warn, r"cover every observation") prior(pe0, Xhol, nothing, pnl_full)
    @test_logs (:warn, r"Assets \[4\]") prior(pe0, Xmix, nothing, pnl_mix)
    # It changes what is SAID, never what is COMPUTED.
    @test isequal((@test_logs (:warn, r"Assets \[4\]") prior(pe0, Xmix, nothing, pnl_mix)).X,
                  prior(pe, Xmix, nothing, pnl_mix).X)

    # `strict` refuses any fill, whatever the share.
    @test_throws ArgumentError prior(pe5, Xhol, nothing, pnl_full; strict = true)
    @test_throws ArgumentError prior(pe, Xmix, nothing, pnl_mix; strict = true)
    @test_throws ArgumentError prior(pe0, Xmix, nothing, pnl_mix; strict = true)
    # A fit with nothing to fill is silent under `strict` too.
    @test_logs prior(pe, X0, nothing, pnl_full; strict = true)
    @test_logs prior(pe0, X0, nothing, pnl_full; strict = true)
    @test_logs prior(EmpiricalPrior(), Xmix, nothing, pnl_mix; strict = true)
end

@testset "The field, its validation and its default" begin
    # The default is `nothing`: no share passes in silence.
    @test isnothing(EmpiricalPrior().fill_limit)
    @test isnothing(EmpiricalPrior(; me = me, ce = ce).fill_limit)
    @test EmpiricalPrior(; fill_limit = 0.2).fill_limit == 0.2
    # The value is inspectable on the estimator, which is the whole point of the field.
    @test :fill_limit in fieldnames(EmpiricalPrior)
    @test occursin("fill_limit", sprint(show, EmpiricalPrior(; fill_limit = 0.2)))

    # The share is a fraction, so nothing outside the unit interval names a reachable one.
    @test_throws DomainError EmpiricalPrior(; fill_limit = -0.1)
    @test_throws DomainError EmpiricalPrior(; fill_limit = 1.5)
    @test_throws DomainError EmpiricalPrior(; fill_limit = -eps())
    @test_throws DomainError EmpiricalPrior(; fill_limit = 2)
    # ZERO IS NOT A VALUE. `nothing` already means that no fill passes in silence, so a `0`
    # that also meant it would be a second spelling of one answer.
    @test_throws DomainError EmpiricalPrior(; fill_limit = 0)
    @test_throws DomainError EmpiricalPrior(; fill_limit = 0.0)
    # The closed upper end is reachable, and it accepts the whole matrix.
    @test EmpiricalPrior(; fill_limit = 1).fill_limit == 1
    @test EmpiricalPrior(; fill_limit = 1.0).fill_limit == 1.0

    # The field rides along a nested prior through `factory`, so a caller who buries an
    # `EmpiricalPrior` inside another prior still sets the share in one place. It survives a
    # view too, because a share is not per-asset configuration.
    inner = EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.2)
    @test PO.factory(inner).fill_limit == 0.2
    @test PO.factory(FactorPrior(; pe = inner)).pe.fill_limit == 0.2
    @test PO.factory(HighOrderPriorEstimator(; pe = inner)).pe.fill_limit == 0.2
    @test PO.port_opt_view(inner, [1, 2]).fill_limit == 0.2

    # THE GLOBAL IS GONE. Nothing in the package answers to the scoped config that preceded
    # the field, and no preference key seeds it.
    @test !isdefined(PO, :SCENARIO_FILL_LIMIT)
    @test !isdefined(PO, :set_scenario_fill_limit!)
    @test !isdefined(PO, :with_scenario_fill_limit)
    @test !isdefined(PO, :assert_scenario_fill_limit)
    @test !("scenario_fill_limit" in PO.PREFERENCE_KEYS)
    @test !occursin("scenario_fill_limit",
                    PO.relaxed_preferences_msg([("max_bins", 500, 900)]))
end

@testset "The filled prior reaches every consumer" begin
    rd = ReturnsResult(; nx = nx, X = Xmix, pnl = pnl_mix)
    resj = optimise(MeanRisk(; r = ConditionalValueatRisk(),
                             opt = JuMPOptimiser(; pe = pe, slv = slv)), rd)
    resh = optimise(HierarchicalRiskParity(; r = ConditionalValueatRisk(),
                                           opt = HierarchicalOptimiser(; pe = pe)), rd)
    # A scenario measure over a `NaN` column has no answer at all, so both solves are the
    # proof that the fill reached the model.
    @test isa(resj.retcode, PO.OptimisationSuccess)
    @test all(isfinite, resj.w)
    @test all(isfinite, resh.w)
    # The dead asset is expanded back as a zero by ADR 0115's rule, and the young one is held.
    @test iszero(resj.w[3])
    @test iszero(resh.w[3])
    @test !iszero(resj.w[4])
    @test !iszero(resh.w[4])

    # The value-level door reduces to the Investable Mask and reads the filled matrix, so it
    # equals the same measure on the filled columns by hand.
    pr = prior(pe, Xmix, nothing, pnl_mix)
    w = fill(0.25, N)
    keep = [1, 2, 4]
    r = @test_logs (:warn, r"not investable") expected_risk(ConditionalValueatRisk(), w, pr)
    @test r == expected_risk(ConditionalValueatRisk(), w[keep], pr.X[:, keep])
end
