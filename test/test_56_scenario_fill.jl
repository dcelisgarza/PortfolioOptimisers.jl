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
# warns above it, and refuses any fill under `strict`.
#
# THE SHARE IS PER ASSET, issue #1000: the worst investable column's own count of filled
# entries over the number of observations. A matrix-wide denominator scales with the universe,
# so the column the notice exists to catch disappears inside it -- one asset of a hundred whose
# column is seven-tenths invented is seven thousandths of the matrix. The matrix-wide share is
# reported in the message as context and trips nothing.
#
# `fill_limit` AND `min_coverage` ARE ONE NUMBER. `admits` reads an asset's coverage share as
# its own observation count over the observations folded, and the fill counts that column's
# non-finite entries over the same denominator, so admission IS the fill test. `fill_limit`
# defaults to `nothing`, which therefore DERIVES at the fit: `1 - maximum(min_coverage)` over
# the arms that state a floor, which never fires; and, where no arm states one -- the
# exponentially weighted family gates on `min_obs`, a count -- every fill is named. An explicit
# value must be TIGHTER than admission, and a looser one refuses as dead by construction.
#
# `EmpiricalPrior` is the only estimator eligible for the field. A prior is eligible when it
# holds a mask-aware moment estimator DIRECTLY and puts the CALLER'S OWN returns matrix into
# the result's `X`; every other prior wraps an inner prior estimator and synthesises its `X`,
# so the fill is paid once, at the `EmpiricalPrior` at the bottom of the chain.
#
# The reference implementation gives no oracle for the share: it zero-fills its portfolio
# return series at one line and announces nothing, so it has no fill limit and no denominator
# to read off. The reference's own denominator -- filled entries over the entries of the whole
# returns matrix -- is what ADR 0118 replaced, and it is pinned below as the context the
# message reports rather than as the number that trips.

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
    # THE DENOMINATOR IS THE GAPPED COLUMN'S OWN OBSERVATIONS, not the entries of the matrix.
    # Asset 4 is half invented; the matrix is an eighth invented, which is only context.
    filled = PO.scenario_fill_pairs(Xmix, BitVector([1, 1, 0, 1]))
    @test length(filled) == 30
    @test length(Xmix) == 240
    @test length(filled) / length(Xmix) == 0.125
    @test count(i -> last(i) == 4, filled) / T == 0.5
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
    # The worst column and its own share are named, and the matrix-wide share rides along as
    # context, labelled as context.
    @test_logs (:warn, r"The worst column is asset 4, 0\.5 of whose own observations") prior(pe5,
                                                                                             Xmix,
                                                                                             nothing,
                                                                                             pnl_mix)
    @test_logs (:warn,
                r"the whole matrix is 0\.125 invented, which is context and trips nothing") prior(pe5,
                                                                                                  Xmix,
                                                                                                  nothing,
                                                                                                  pnl_mix)
    # THE NOTICE NAMES EVERY CONSUMER OF AN INVENTED COLUMN, not the tail alone: the four the
    # census of the readers of `pr.X` found, and the tail's cost as the identity.
    @test_logs (:warn, r"four consumers read them") prior(pe5, Xmix, nothing, pnl_mix)
    @test_logs (:warn, r"they inflate the denominator") prior(pe5, Xmix, nothing, pnl_mix)
    @test_logs (:warn, r"at `c = 0\.3` a 5% CVaR is a 16\.7% CVaR") prior(pe5, Xmix,
                                                                          nothing, pnl_mix)
    @test_logs (:warn, r"hierarchical optimiser may branch the asset alone") prior(pe5,
                                                                                   Xmix,
                                                                                   nothing,
                                                                                   pnl_mix)
    @test_logs (:warn, r"entropy pooling view on the asset is calibrated") prior(pe5, Xmix,
                                                                                 nothing,
                                                                                 pnl_mix)
    @test_logs (:warn, r"meta-optimiser carries the fill into the outer problem") prior(pe5,
                                                                                        Xmix,
                                                                                        nothing,
                                                                                        pnl_mix)
    # The remedy the message names is the field, not a global.
    @test_logs (:warn, r"EmpiricalPrior\(; fill_limit = \.\.\.\)") prior(pe5, Xmix, nothing,
                                                                         pnl_mix)
    # A limit the MATRIX-WIDE share would have passed in silence still warns, because the
    # column is half invented. This is the blindness the per-asset denominator removes.
    @test_logs (:warn, r"Assets \[4\]") prior(EmpiricalPrior(; me = me, ce = ce,
                                                             fill_limit = 0.2), Xmix,
                                              nothing, pnl_mix)
    # Raise the share above the COLUMN'S OWN and the same fit is silent.
    @test_logs prior(EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.6), Xmix, nothing,
                     pnl_mix)

    # TWO PRIORS IN ONE PROGRAM, TWO ANSWERS. This is what the field buys over the global that
    # preceded it: the share travels with the estimator that fills, not with the session.
    loud = EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.05)
    quiet = EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.6)
    @test_logs (:warn, r"Assets \[4\]") prior(loud, Xmix, nothing, pnl_mix)
    @test_logs prior(quiet, Xmix, nothing, pnl_mix)
    @test isequal(prior(quiet, Xmix, nothing, pnl_mix).X,
                  (@test_logs (:warn, r"Assets \[4\]") prior(loud, Xmix, nothing, pnl_mix)).X)

    # One filled entry is 1/60 of asset 2's own observations, so a 5% share passes it in
    # silence under the per-asset denominator as it did under the matrix-wide one.
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

# The panel for a window of `t` observations over `N2` assets, so an expanding walk-forward can
# be driven a fold at a time without the cross-validation machinery in the way: a fold's fit is
# exactly this call.
function window_panel(amsk, t)
    return AssetPanel(;
                      pf = [NumericPanelField(; name = "mcap",
                                              vals = ones(t, size(amsk, 2)))],
                      amsk = view(amsk, 1:t, :), emsk = view(amsk, 1:t, :))
end

@testset "The share the notice tests is the worst column's own, not the matrix's" begin
    # THE CASE THE MATRIX-WIDE DENOMINATOR IS BLIND TO. One asset of twenty is seven-tenths
    # invented, and that is thirty-five thousandths of the matrix -- under any limit a caller
    # would set. The per-asset denominator gives one number with one meaning whatever the
    # width of the panel.
    rngw = StableRNG(24680)
    Tw, Nw = 100, 20
    Xw = randn(rngw, Tw, Nw) ./ 100 .+ 0.0005
    Xw[1:70, 1] .= NaN
    amsk_w = trues(Tw, Nw)
    amsk_w[1:70, 1] .= false
    pnl_w = window_panel(amsk_w, Tw)

    @test 70 / length(Xw) == 0.035
    @test 70 / Tw == 0.7

    # A 5% limit passes the matrix-wide share and fails the column's own, so it warns.
    wide = EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.05)
    @test_logs (:warn, r"Assets \[1\]") prior(wide, Xw, nothing, pnl_w)
    @test_logs (:warn, r"The worst column is asset 1, 0\.7 of whose own observations") prior(wide,
                                                                                             Xw,
                                                                                             nothing,
                                                                                             pnl_w)
    @test_logs (:warn, r"the whole matrix is 0\.035 invented") prior(wide, Xw, nothing,
                                                                     pnl_w)
    # Only a limit above the COLUMN'S share buys silence.
    @test_logs prior(EmpiricalPrior(; me = me, ce = ce, fill_limit = 0.75), Xw, nothing,
                     pnl_w)
    @test_logs (:warn, r"Assets \[1\]") prior(EmpiricalPrior(; me = me, ce = ce,
                                                             fill_limit = 0.65), Xw,
                                              nothing, pnl_w)
end

@testset "The dilution identity a scenario measure reads" begin
    # THE COST OF THE FILL IS EXACT, and the invented zeros are not the reason a reader might
    # expect: they do not enter the tail, they inflate the denominator. For an admitted column
    # of coverage `c` whose asset carries at least `ceil(alpha * T)` losses, the measure at
    # level `alpha` over the FILLED column equals the measure at level `alpha / c` over that
    # column's OBSERVED rows -- to the last bit, because `alpha * T` and `(alpha / c) * (c * T)`
    # are the same count, the same order statistic and the same denominator.
    #
    # The shares are exact binary fractions so that the two `ceil(Int, alpha * T)` agree
    # without a rounding argument. ADR 0118's own illustration is the same statement at
    # `c = 0.3`, where a 5% CVaR is a 16.7% CVaR.
    rngd = StableRNG(13579)
    Td, Nd = 64, 3
    Xd = randn(rngd, Td, Nd) ./ 100
    Xd[1:32, 2] .= NaN
    amsk_d = trues(Td, Nd)
    amsk_d[1:32, 2] .= false
    pnl_d = window_panel(amsk_d, Td)

    c = 0.5
    alpha = 0.0625
    @test alpha * Td == 4.0
    @test (alpha / c) * (c * Td) == 4.0

    prd = @test_logs prior(EmpiricalPrior(; me = me, ce = ce, fill_limit = 1), Xd, nothing,
                           pnl_d)
    @test isnothing(PO.investable_mask(prd))
    @test all(iszero, view(prd.X, 1:32, 2))
    observed = Xd[33:Td, 2]
    # The column carries more than the four losses the identity needs.
    @test count(<(0), observed) >= ceil(Int, alpha * Td)

    filled_col = reshape(prd.X[:, 2], :, 1)
    obs_col = reshape(observed, :, 1)
    @test expected_risk(ConditionalValueatRisk(; alpha = alpha), [1.0], filled_col) ==
          expected_risk(ConditionalValueatRisk(; alpha = alpha / c), [1.0], obs_col)
    # And the naive reading -- the same level over the observed rows -- is a different number,
    # which is what makes the identity worth stating.
    @test expected_risk(ConditionalValueatRisk(; alpha = alpha), [1.0], filled_col) !=
          expected_risk(ConditionalValueatRisk(; alpha = alpha), [1.0], obs_col)
end

@testset "The notice's truth table, and the limit that derives" begin
    cvgp = CoveragePolicy(; min_coverage = 0.1)
    me_p = SimpleExpectedReturns(; cvg = cvgp)
    ce_p = Covariance(; cvg = cvgp)

    # THE FLOOR IS READ OFF THE ARMS, and the binding one is the maximum, because admission is
    # their conjunction and both arms read the same per-asset count.
    @test PO.coverage_floor(EmpiricalPrior(; me = me_p, ce = ce_p)) == 0.1
    @test PO.coverage_floor(EmpiricalPrior(; me = me_p)) == 0.1
    @test PO.coverage_floor(EmpiricalPrior(; ce = ce_p)) == 0.1
    @test PO.coverage_floor(EmpiricalPrior(; me = me_p,
                                           ce = Covariance(;
                                                           cvg = CoveragePolicy(;
                                                                                min_coverage = 0.4)))) ==
          0.4
    # A policy inside the default wrapper is still the estimator's own floor.
    @test PO.coverage_floor(EmpiricalPrior(;
                                           ce = PortfolioOptimisersCovariance(; ce = ce_p))) ==
          0.1
    # No arm states one, so there is no floor -- not a floor of zero.
    @test isnothing(PO.coverage_floor(EmpiricalPrior()))
    @test isnothing(PO.coverage_floor(EmpiricalPrior(; me = me, ce = ce)))
    # The verb reads the policy off a carrier and off the policy itself, and answers `nothing`
    # for an arm that carries none.
    @test PO.coverage_floor(cvgp) == 0.1
    @test isnothing(PO.coverage_floor(nothing))
    @test PO.coverage_floor(SimpleVariance(; cvg = cvgp)) == 0.1
    @test isnothing(PO.coverage_floor(SimpleVariance()))
    @test isnothing(PO.coverage_floor(SimpleExpectedReturns()))
    @test isnothing(PO.coverage_floor(Covariance()))

    # `nothing` DERIVES `1 - min_coverage` where a floor is stated, and keeps its original
    # meaning where none is.
    @test PO.resolve_fill_limit(nothing, 0.1) == 0.9
    @test isnothing(PO.resolve_fill_limit(nothing, nothing))
    @test PO.resolve_fill_limit(0.5, nothing) == 0.5
    @test PO.resolve_fill_limit(0.9, 0.1) == 0.9

    # ROW 1. No policy and `fill_limit = nothing` names every fill: the exponentially weighted
    # family is mask-aware without a policy, gating on `min_obs`, a count.
    @test_logs (:warn, r"Assets \[4\]") prior(pe0, Xmix, nothing, pnl_mix)
    @test_logs (:warn, r"`fill_limit = nothing`") prior(pe0, Xhol, nothing, pnl_full)

    # ROW 2. A policy and `fill_limit = nothing` names NONE, on every fold of an expanding
    # walk-forward over a panel with a listing and a delisting. The derived limit cannot fire:
    # an admitted column satisfies it by construction.
    pe_p = EmpiricalPrior(; me = me_p, ce = ce_p)
    filled_any = false
    for t in 10:5:T
        pnl_t = window_panel(amsk_mix, t)
        prt = @test_logs prior(pe_p, Xmix[1:t, :], nothing, pnl_t)
        filled_any |= any(iszero, view(prt.X, :, 4))
    end
    # The silence is not vacuous: at least one fold actually filled a column.
    @test filled_any
    pnl_40 = window_panel(amsk_mix, 40)
    pr40 = @test_logs prior(pe_p, Xmix[1:40, :], nothing, pnl_40)
    # Every asset is admitted at this window -- asset 3 still quotes, and asset 4 clears the
    # floor at 10/40 -- so the mask is the all-investable `nothing`.
    @test isnothing(PO.investable_mask(pr40))
    @test count(iszero, view(pr40.X, :, 4)) == 30
    @test 30 / 40 == 0.75

    # ROW 3. An explicit limit LOOSER than admission refuses at the fit, because nothing that
    # reaches the fill could trip it. It refuses whether or not this window has a gap: the
    # configuration is dead by construction, not by data.
    dead = EmpiricalPrior(; me = me_p, ce = ce_p, fill_limit = 0.95)
    @test_throws DomainError prior(dead, Xmix[1:40, :], nothing, pnl_40)
    @test_throws DomainError prior(dead, X0, nothing, pnl_full)
    @test_throws DomainError prior(EmpiricalPrior(; me = me_p, ce = ce_p, horizon = 5,
                                                  fill_limit = 0.95), X0, nothing, pnl_full)
    # The boundary is closed: exactly `1 - min_coverage` is admissible.
    @test_logs prior(EmpiricalPrior(; me = me_p, ce = ce_p, fill_limit = 0.9),
                     Xmix[1:40, :], nothing, pnl_40)
    # A TIGHTER value is the one configuration the derivation cannot express: admit broadly and
    # be told anyway.
    @test_logs (:warn, r"Assets \[4\]") prior(EmpiricalPrior(; me = me_p, ce = ce_p,
                                                             fill_limit = 0.5),
                                              Xmix[1:40, :], nothing, pnl_40)

    # ROW 4. `strict` refuses any fill under either configuration, whatever the limit derives.
    @test_throws ArgumentError prior(pe_p, Xmix[1:40, :], nothing, pnl_40; strict = true)
    @test_throws ArgumentError prior(EmpiricalPrior(; me = me_p, ce = ce_p,
                                                    fill_limit = 0.9), Xmix[1:40, :],
                                     nothing, pnl_40; strict = true)
    @test_throws ArgumentError prior(pe0, Xmix, nothing, pnl_mix; strict = true)
    # A fold with nothing to fill is silent under `strict` too.
    @test_logs prior(pe_p, X0, nothing, pnl_full; strict = true)
end
