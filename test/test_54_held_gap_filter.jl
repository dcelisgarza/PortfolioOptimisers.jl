using Clarabel, Statistics
using InteractiveUtils: subtypes

#=
The Held Gap filter, the value-level doors, and the tripwire that keeps them honest.

`#856` builds the first half of `#673` (ADR 0118) and the first decision of `#674` (ADR 0120):

  - A fold views its test window and its weights at the Investable Mask, then zeroes the Held
    Gaps of the reduced window **once**, before the series is formed and before a Weight Drift
    compounds on it. A Held Gap is an (observation, asset) pair at which the weight is non-zero and
    the return is missing, which is what an asset that delists *inside* the test window makes. The
    fees are not viewed: the result carries them on the universe it solved on (#892).
  - A value-level verb against a Prior Result reduces to the Investable Mask at its entry and
    expands a per-asset answer back to the full length.
  - A series the caller holds is documented, not checked.

The oracle everywhere is the same problem with the non-investable asset removed **by hand**, which
is the oracle `test/test_50_investable_reduction.jl` uses for the optimiser side.
=#

const PO = PortfolioOptimisers

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             check_sol = (; allow_local = true, allow_almost = true),
             settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                             "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12,
                             "tol_infeas_abs" => 1e-12, "tol_infeas_rel" => 1e-12))

rng = StableRNG(198654321)
T, N = 180, 5
X = randn(rng, T, N) ./ 100 .+ 0.0005
nx = ["a", "b", "c", "d", "e"]
rd = ReturnsResult(; nx = nx, X = X)
pr = prior(EmpiricalPrior(), rd)

# Asset 3 is the one the prior could not estimate, and `keep` is the universe that survives it.
k = 3
keep = [1, 2, 4, 5]
imsk = BitVector([1, 1, 0, 1, 1])

function nan_prior(pr, k)
    mu = collect(pr.mu)
    sigma = collect(pr.sigma)
    Xn = collect(pr.X)
    mu[k] = NaN
    sigma[k, :] .= NaN
    sigma[:, k] .= NaN
    Xn[:, k] .= NaN
    return LowOrderPrior(; X = Xn, mu = mu, sigma = sigma)
end
prn = nan_prior(pr, k)
prk = LowOrderPrior(; X = pr.X[:, keep], mu = pr.mu[keep], sigma = pr.sigma[keep, keep])

# The weights: the same portfolio, once on the full universe with a zero at the dead asset, and
# once on the hand-reduced one.
w_full = [0.3, 0.2, 0.0, 0.35, 0.15]
w_keep = w_full[keep]
# The same portfolio, but holding the asset the prior could not estimate.
w_held = [0.3, 0.2, 0.1, 0.3, 0.1]

@testset "filter_held_gaps: the window is cleaned once, and only a held pair is named" begin
    Y = [0.01 0.02 0.03
         0.04 NaN 0.06
         0.07 0.08 NaN]
    # A finite window is returned unchanged, and it is the very same object: the filter costs a
    # scan and nothing else on the path every existing fold takes.
    Z = [0.01 0.02; 0.03 0.04]
    @test PO.filter_held_gaps([0.5, 0.5], Z, false) === Z
    # A zero weight at a gap is silent, and the gap is still zeroed, because `0 * NaN` is `NaN`
    # and the product would poison the whole observation.
    Yf = @test_logs min_level=Logging.Warn PO.filter_held_gaps([1.0, 0.0, 0.0], Y, false)
    @test all(isfinite, Yf)
    @test Yf[2, 2] == 0
    @test Yf[3, 3] == 0
    @test Yf[1, :] == Y[1, :]
    # A non-zero weight at a gap is a Held Gap: one warning, naming the assets.
    Yg = @test_logs (:warn, r"non-finite return at 2 held \(observation, asset\) pair\(s\)") PO.filter_held_gaps([0.5,
                                                                                                                  0.3,
                                                                                                                  0.2],
                                                                                                                 Y,
                                                                                                                 false)
    @test all(isfinite, Yg)
    @test Yg == Yf
    # Under `strict` it refuses instead.
    @test_throws ArgumentError PO.filter_held_gaps([0.5, 0.3, 0.2], Y, true)
    # The pairs themselves, so the message cannot drift from what it counted.
    @test PO.held_gap_pairs([0.5, 0.3, 0.2], Y) == [(2, 2), (3, 3)]
    @test PO.held_gap_pairs([0.5, 0.0, 0.5], Y) == [(3, 3)]
    # A population holds a column when any member does.
    @test PO.held_gap_pairs([[0.5, 0.0, 0.5], [0.4, 0.6, 0.0]], Y) == [(2, 2), (3, 3)]
    # A weight path judges each pair by the weights held through its own observation.
    U = [0.5 0.3 0.2
         0.5 0.0 0.5
         0.5 0.3 0.2]
    @test PO.held_gap_pairs(U, Y) == [(3, 3)]
end

@testset "investable_reduction: the value-level door reduces, names and passes through" begin
    # A bare matrix and a `ReturnsResult` carry no moments, so no mask exists.
    m, Xr, wr, fr = PO.investable_reduction(X, w_full, nothing, false)
    @test isnothing(m)
    @test Xr === X
    @test wr === w_full
    m, rdr, wr, fr = PO.investable_reduction(rd, w_full, nothing, false)
    @test isnothing(m)
    @test rdr === rd
    # An all-investable prior takes the path it took before the mask existed.
    m, prr, wr, fr = PO.investable_reduction(pr, w_full, nothing, false)
    @test isnothing(m)
    @test prr === pr
    @test wr === w_full
    # A gapped prior reduces, and a zero weight at the dead asset is silent.
    m, prr, wr, fr = @test_logs min_level=Logging.Warn PO.investable_reduction(prn, w_full,
                                                                               nothing,
                                                                               false)
    @test m == imsk
    @test wr == w_keep
    @test prr.mu == prk.mu
    @test prr.sigma == prk.sigma
    @test prr.X == prk.X
    # A held non-investable asset is named once.
    m, prr, wr, fr = @test_logs (:warn, r"Assets \[3\] are not investable") PO.investable_reduction(prn,
                                                                                                    w_held,
                                                                                                    nothing,
                                                                                                    false)
    @test wr == w_held[keep]
    @test_throws ArgumentError PO.investable_reduction(prn, w_held, nothing, true)
    # The fees ride the same axis as the weights, so they are sliced with them.
    fees = Fees(; l = [0.001, 0.002, 0.003, 0.004, 0.005])
    m, prr, wr, fr = PO.investable_reduction(prn, w_full, fees, false)
    @test fr.l == [0.001, 0.002, 0.004, 0.005]
    # The expansion is the inverse of the view on a per asset answer.
    @test PO.expand_investable_weights(imsk, w_keep) == w_full
    @test PO.expand_investable_weights(nothing, w_keep) === w_keep
    @test_throws DimensionMismatch PO.expand_investable_weights(imsk, w_full)
end

#=
The tripwire. It walks the concrete subtypes of `AbstractBaseRiskMeasure` rather than a written
list, because a written list is exactly what a new measure is forgotten in.

Two filters keep it honest. Only types defined in `PortfolioOptimisers` are taken, so a probe type
left in a shared session cannot join the census; and only measures that answer both sides of the
oracle are compared, so a measure that needs a slot the fixture does not state is skipped rather
than failed.
=#
function all_concrete(@nospecialize(T))
    acc = Any[]
    for S in subtypes(T)
        isabstracttype(S) ? append!(acc, all_concrete(S)) : push!(acc, S)
    end
    return acc
end

@testset "The tripwire: every value-level measure reduces to the Investable Mask" begin
    measures = Any[]
    for U in all_concrete(PO.AbstractBaseRiskMeasure)
        if parentmodule(U) !== PO
            continue
        end
        r = try
            U()
        catch
            continue
        end
        # The oracle side must answer, else the fixture does not state what this measure needs.
        ok = try
            isfinite(expected_risk(r, w_keep, prk))
        catch
            false
        end
        if ok
            push!(measures, r)
        end
    end
    # The census is worthless if it walked nothing.
    @test length(measures) >= 10
    for r in measures
        # A zero weight at the dead asset: the same answer as the hand-reduced problem.
        rk = @test_logs min_level=Logging.Warn expected_risk(r, w_full, prn)
        @test isapprox(rk, expected_risk(r, w_keep, prk))
        # A held weight at the dead asset: one warning, and the answer of the reduced problem.
        rh = @test_logs (:warn, r"Assets \[3\] are not investable") expected_risk(r, w_held,
                                                                                  prn)
        @test isapprox(rh, expected_risk(r, w_held[keep], prk))
        # And a refusal under `strict`.
        @test_throws ArgumentError expected_risk(r, w_held, prn; strict = true)
    end
end

@testset "risk_contribution expands, and a dead asset reports exactly zero" begin
    r = Variance()
    rc = @test_logs min_level=Logging.Warn risk_contribution(r, w_full, prn)
    @test length(rc) == N
    @test rc[k] === zero(eltype(rc))
    @test isapprox(rc[keep], risk_contribution(r, w_keep, prk))
    # The marginal figures expand the same way.
    mrc = risk_contribution(r, w_full, prn; marginal = true)
    @test length(mrc) == N
    @test mrc[k] === zero(eltype(mrc))
    # A held weight names the asset once and refuses under `strict`.
    @test_logs (:warn, r"Assets \[3\] are not investable") risk_contribution(r, w_held, prn)
    @test_throws ArgumentError risk_contribution(r, w_held, prn; strict = true)
    # A bare matrix is unchanged: no mask exists, so nothing is dropped or expanded.
    @test length(risk_contribution(ConditionalValueatRisk(), w_full, X)) == N
end

@testset "factor_risk_contribution reduces the prior and the factor block together" begin
    # The factor door is the one value-level verb that also has to cut the returns it
    # regresses against, because the loadings are fitted per asset. The oracle is the same
    # regression over the hand-reduced universe.
    rngf = StableRNG(56781234)
    F = randn(rngf, T, 2) ./ 100
    rdf = ReturnsResult(; nx = nx, X = X, nf = ["f1", "f2"], F = F)
    rdk = PO.port_opt_view(rdf, keep)
    r = Variance()
    frc = factor_risk_contribution(r, w_full, prn; rd = rdf)
    @test length(frc) == size(F, 2) + 1
    @test isapprox(frc, factor_risk_contribution(r, w_keep, prk; rd = rdk))
    # A held weight names the asset, and refuses under `strict`. The stepwise regression
    # warns on its own over these synthetic factors, so the match is loose on purpose.
    @test_logs (:warn, r"Assets \[3\] are not investable") match_mode=:any factor_risk_contribution(r,
                                                                                                    w_held,
                                                                                                    prn;
                                                                                                    rd = rdf)
    @test_throws ArgumentError factor_risk_contribution(r, w_held, prn; rd = rdf,
                                                        strict = true)
end

@testset "expected_return reduces, and a composite names a held asset once" begin
    @test isapprox(expected_return(ArithmeticReturn(), w_full, prn),
                   expected_return(ArithmeticReturn(), w_keep, prk))
    @test isapprox(expected_return(LogarithmicReturn(), w_full, prn),
                   expected_return(LogarithmicReturn(), w_keep, prk))
    # The ratio composite calls both doors, and it is the outermost entry that names the asset,
    # so the caller reads one diagnostic and not two.
    @test_logs (:warn, r"Assets \[3\] are not investable") expected_ratio(Variance(),
                                                                          ArithmeticReturn(),
                                                                          w_held, prn)
    @test isapprox(expected_ratio(Variance(), ArithmeticReturn(), w_full, prn),
                   expected_ratio(Variance(), ArithmeticReturn(), w_keep, prk))
end

@testset "result_investable_mask: a result that carries a mask cannot hide it" begin
    fallback = which(PO.result_investable_mask, Tuple{PO.OptimisationResult})
    base_getproperty = which(Base.getproperty, Tuple{Any, Symbol})
    # Every concrete result that carries an `imsk` field must answer it through the verb the
    # fold reads. Nothing else keeps a family that gains a mask from being scored on the wrong
    # window, in silence. A leaf that overrides `getproperty` carries its core's mask as a
    # forwarded property, which `fieldnames` cannot see and the verb cannot dispatch on, so
    # such a leaf is held to the same rule (#892: the two hierarchical leaves fell back).
    for U in all_concrete(PO.OptimisationResult)
        parentmodule(U) === PO || continue
        forwards = which(Base.getproperty, Tuple{U, Symbol}) !== base_getproperty
        if :imsk in fieldnames(U) || forwards
            @test which(PO.result_investable_mask, Tuple{U}) !== fallback
        end
    end
    # The JuMP side answers the mask its own reduction derived.
    res = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prn, slv = slv)), rd)
    @test PO.result_investable_mask(res) == imsk
    @test length(res.w) == N
    @test res.w[k] == 0
    res_all = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv)), rd)
    @test isnothing(PO.result_investable_mask(res_all))
    # Every family that carries the mask on the result itself answers it directly. The
    # results are built by hand, because the verb is what the fold reads and a family that
    # gains a mask must answer it whatever route built the result.
    nres = NaiveOptimisationResult(; pr = prk, wb = nothing,
                                   retcode = OptimisationSuccess(), w = w_keep, imsk = imsk,
                                   fb = nothing)
    @test PO.result_investable_mask(nres) == imsk
    sres = SubsetResamplingResult(; pr = prk, wb = nothing, fees = nothing, ress = [nres],
                                  idx = reshape([1.0, 2.0], 2, 1),
                                  retcode = OptimisationSuccess(), w = w_keep, imsk = imsk,
                                  fb = nothing)
    @test PO.result_investable_mask(sres) == imsk
    kres = StackingResult(; pr = prk, wb = nothing, fees = nothing, resi = [nres],
                          reso = nres, cv = nothing, retcode = OptimisationSuccess(),
                          w = w_keep, imsk = imsk, fb = nothing)
    @test PO.result_investable_mask(kres) == imsk
    # The hierarchical core answers directly, and its two leaves answer through the core.
    hres = HierarchicalResult(; pr = prk, clr = nothing, wb = nothing, fees = nothing,
                              retcode = OptimisationSuccess(), w = w_keep, imsk = imsk)
    @test PO.result_investable_mask(hres) == imsk
    @test hres.w == w_full
    hrp = HierarchicalRiskParityResult(; hr = hres, r = Variance(), sca = SumScalariser(),
                                       fb = nothing)
    @test PO.result_investable_mask(hrp) == imsk
    herc = HierarchicalEqualRiskContributionResult(; hr = hres, ri = Variance(),
                                                   ro = Variance(), scai = SumScalariser(),
                                                   scao = SumScalariser(), fb = nothing)
    @test PO.result_investable_mask(herc) == imsk
    ncres = NestedClusteredResult(; pr = prk, clr = nothing, wb = nothing, fees = nothing,
                                  resi = [nres], reso = nres, cv = nothing,
                                  retcode = OptimisationSuccess(), w = w_keep, imsk = imsk,
                                  fb = nothing)
    @test PO.result_investable_mask(ncres) == imsk
    scres = SchurComplementHierarchicalRiskParityResult(; pr = prk, wb = nothing,
                                                        clr = nothing, r = Variance(),
                                                        gamma = 0.5,
                                                        retcode = OptimisationSuccess(),
                                                        w = w_keep, imsk = imsk,
                                                        fb = nothing)
    @test PO.result_investable_mask(scres) == imsk
end

#=
The fold. A walk-forward whose training window sees a non-investable asset and whose **test**
window holds a delisting of an investable one. The two gaps are taken in order: the dead column is
never read, and the delisting is a Held Gap that is zeroed once and named once.
=#
@testset "The fold zeroes a Held Gap once, and expands its Held Weights record" begin
    # Asset 5 delists inside the test window; asset 3 is the column the prior could not estimate.
    Xd = copy(X)
    delist = 160
    Xd[delist:end, 5] .= NaN
    Xd[:, k] .= NaN
    rdd = ReturnsResult(; nx = nx, X = Xd)
    test_idx = collect(150:T)

    res = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prn, slv = slv)), rd)
    @test PO.result_investable_mask(res) == imsk

    pred = @test_logs (:warn, r"Assets \[4\] carry a non-finite return") predict(res, rdd,
                                                                                 test_idx)
    # The series is finite, and it is the identity ADR 0118 pins, with the dead column never
    # read and every gap of the reduced window contributing zero.
    @test all(isfinite, pred.rd.X)
    wk = res.w[keep]
    Xk = Xd[test_idx, keep]
    hand = [sum(wk[i] * (isfinite(Xk[t, i]) ? Xk[t, i] : zero(eltype(Xk)))
                for i in eachindex(wk)) for t in axes(Xk, 1)]
    @test isapprox(pred.rd.X, hand)
    # An all-gap row earns nothing at all.
    Xa = copy(Xd)
    Xa[test_idx, keep] .= NaN
    rda = ReturnsResult(; nx = nx, X = Xa)
    preda = @test_logs (:warn, r"carry a non-finite return") predict(res, rda, test_idx)
    @test all(iszero, preda.rd.X)
    # Under `strict` the fold refuses rather than warning.
    @test_throws ArgumentError predict(res, rdd, test_idx; strict = true)
    # A window with no delisting reads the dead column, warns nothing, and scores the live set.
    Xc = copy(X)
    Xc[:, k] .= NaN
    rdc = ReturnsResult(; nx = nx, X = Xc)
    predc = @test_logs min_level=Logging.Warn predict(res, rdc, test_idx)
    @test isapprox(predc.rd.X, Xc[test_idx, keep] * wk)
end

@testset "The fold's Held Weights record comes back on the caller's universe" begin
    Xd = copy(X)
    Xd[160:end, 5] .= NaN
    Xd[:, k] .= NaN
    rdd = ReturnsResult(; nx = nx, X = Xd)
    test_idx = collect(150:T)
    res = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prn, slv = slv)), rd)
    pred = @test_logs (:warn, r"carry a non-finite return") predict(res, rdd, test_idx;
                                                                    wd = SelfFinancingDrift(),
                                                                    store_weight_path = true)
    # The next fold's turnover reads these weights against the caller's own universe.
    @test length(pred.hw.w) == N
    @test pred.hw.w[k] == 0
    @test size(pred.hw.X, 2) == N
    @test all(iszero, view(pred.hw.X, :, k))
    @test size(pred.hw.U, 2) == N
    @test all(iszero, view(pred.hw.U, :, k))
    # The rebuilt path is the stored one, on the expanded record as on the reduced one.
    @test PO.weight_path(pred.hw, res.w) == pred.hw.U
end

@testset "The fold charges the fees the result carries, and views them no second time" begin
    # A result carries its `Fees` on the universe it solved on (ADR 0115), so the fold must not
    # view them again at the mask: a per-asset rate indexed by full-universe positions is a
    # `BoundsError`, and a scalar rate hides it because a scalar passes through the view (#892).
    # The oracle is the hand-reduced fee: the same rates and previous weights on `keep`.
    w_prev = [0.2, 0.2, 0.2, 0.2, 0.2]
    rate = [0.001, 0.002, 0.010, 0.003, 0.004]
    l = [0.0005, 0.0010, 0.0015, 0.0020, 0.0025]
    fees = Fees(; tn = Turnover(; w = w_prev, val = rate), l = l)
    fees_keep = Fees(; tn = Turnover(; w = w_prev[keep], val = rate[keep]), l = l[keep])
    Xc = copy(X)
    Xc[:, k] .= NaN
    rdc = ReturnsResult(; nx = nx, X = Xc)
    test_idx = collect(150:T)
    for opt in (MeanRisk(; opt = JuMPOptimiser(; pe = prn, slv = slv, fees = fees)),
                HierarchicalRiskParity(; opt = HierarchicalOptimiser(; pe = prn, fees = fees)))
        res = optimise(opt, rd)
        @test PO.result_investable_mask(res) == imsk
        @test length(res.w) == N
        @test res.w[k] == 0
        @test length(res.fees.tn.val) == length(keep)
        wk = res.w[keep]
        # The whole-sample door and the fold door both charge the reduced fee once.
        pred = predict(res, rdc)
        @test isapprox(pred.rd.X, Xc[:, keep] * wk .- PO.calc_periodic_fees(wk, fees_keep))
        predf = predict(res, rdc, test_idx)
        @test isapprox(predf.rd.X,
                       Xc[test_idx, keep] * wk .- PO.calc_periodic_fees(wk, fees_keep))
    end
    # Issue #898: the clock reaches the two fixed terms alone, and this fee carries none,
    # so an `AmortisedFees()` leaves every number where it was. The pass-through still
    # carries the clock through to the reduced fee on the result.
    feesa = Fees(; tn = Turnover(; w = w_prev, val = rate), l = l, fa = AmortisedFees())
    resa = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prn, slv = slv, fees = feesa)),
                    rd)
    wk = resa.w[keep]
    feesa_keep = Fees(; tn = Turnover(; w = w_prev[keep], val = rate[keep]), l = l[keep],
                      fa = AmortisedFees())
    preda = predict(resa, rdc, test_idx)
    @test isa(resa.fees.fa, AmortisedFees)
    @test isapprox(preda.rd.X,
                   Xc[test_idx, keep] * wk .- PO.calc_periodic_fees(wk, feesa_keep))
end

@testset "The scheme carries `strict`, and the fold reads it" begin
    for cv in (KFold(), CombinatorialCrossValidation(), IndexWalkForward(60, 20),
               DateWalkForward(60, 20))
        @test cv.strict === false
        @test PO.fold_evaluation(cv).strict === false
    end
    @test KFold(; strict = true).strict === true
    @test CombinatorialCrossValidation(; strict = true).strict === true
    @test IndexWalkForward(60, 20; strict = true).strict === true
    @test DateWalkForward(60, 20; strict = true).strict === true
    @test PO.fold_evaluation(IndexWalkForward(60, 20; strict = true)).strict === true
    # A scheme that states nothing keeps the library's default.
    @test PO.fold_evaluation(nothing).strict === false
end

#=
The doors that take a caller's series take no finiteness check, so what is pinned here is the
documented compaction identity: dropping the gaps first is what reproduces the reference
implementation's drop-per-column answer. One measure of each kernel class.
=#
@testset "The Precomputed-returns contract: compaction is the documented cure" begin
    xc = collect(range(; start = -0.05, stop = 0.05, length = 20))
    x = vcat(xc, fill(NaN, 10))
    xf = x[isfinite.(x)]
    @test xf == xc
    for r in (ConditionalValueatRisk(), ValueatRisk(), MeanReturn())
        @test isfinite(expected_risk_from_returns(r, xf))
        @test expected_risk_from_returns(r, xf) == r(xf)
    end
    # The gapped series is not refused, and it is not silently right either: a tail measure
    # answers a finite wrong number, which is what the docstrings state. `partialsort` orders
    # a `NaN` after every real, so the order statistic is read off the finite prefix and
    # divided by the poisoned length.
    @test isfinite(expected_risk_from_returns(ConditionalValueatRisk(), x))
    @test expected_risk_from_returns(ConditionalValueatRisk(), x) !=
          expected_risk_from_returns(ConditionalValueatRisk(), xf)
    # The summary and the two accumulations carry the gap forward rather than dropping it.
    @test isnan(performance_summary(x).ann_return)
    @test isfinite(performance_summary(xf).ann_return)
    @test !all(isfinite, cumulative_returns(x))
    @test all(isfinite, cumulative_returns(xf))
    @test !all(isfinite, drawdowns(x))
    @test all(isfinite, drawdowns(xf))
end
