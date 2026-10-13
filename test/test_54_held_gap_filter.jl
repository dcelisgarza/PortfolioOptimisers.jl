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
    # A fold filters the window viewed at the Investable Mask, so a column index of it is a
    # reduced one that names nothing to the caller: given the view's names, the message names
    # the assets, and with none it reports the columns it was handed.
    @test_logs (:warn, r"Assets \[\"b\", \"c\"\] carry") PO.filter_held_gaps([0.5, 0.3,
                                                                              0.2], Y,
                                                                             false;
                                                                             nx = ["a", "b",
                                                                                   "c"])
    @test occursin("Assets [2, 3] carry", PO.held_gap_msg([(2, 2), (3, 3)]))
    @test occursin("Assets [\"b\", \"c\"] carry",
                   PO.held_gap_msg([(2, 2), (3, 3)], ["a", "b", "c"]))
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
    # `BudgetedHindsightPathResult` is a path, so its `w` and its `imsk` are `rows × assets`,
    # one allocation and one mask per row. No fold reads it: its `predict` answers one
    # `NaiveOptimisationResult` per row with that row's mask, and those leaves are held here.
    per_row = (PO.BudgetedHindsightPathResult,)
    for U in all_concrete(PO.OptimisationResult)
        parentmodule(U) === PO || continue
        U in per_row && continue
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
                                                        clr = nothing, fees = nothing,
                                                        r = Variance(), gamma = 0.5,
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

    # The window is viewed at the Investable Mask, so asset 5 is column 4 of it; the message
    # names the asset rather than reporting that column.
    pred = @test_logs (:warn, r"Assets \[\"e\"\] carry a non-finite return") predict(res,
                                                                                     rdd,
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

@testset "The fold-taking consumers read the expanded record at the result's mask" begin
    # The record and `res.w` come back on the caller's universe, and `res.fees` stays on the
    # investable one (ADR 0115), so a consumer that hands the three to a value-level verb
    # unviewed meets a `DimensionMismatch` on the fee, or pairs the fold's reduced `nx` with
    # a full-width matrix. The oracle is the same fold solved on the hand-reduced universe.
    Xc = copy(X)
    Xc[:, k] .= NaN
    rdc = ReturnsResult(; nx = nx, X = Xc)
    rdk = ReturnsResult(; nx = nx[keep], X = X[:, keep])
    test_idx = collect(150:T)
    rate = [0.001, 0.002, 0.010, 0.003, 0.004]
    fees = Fees(; tn = Turnover(; w = fill(0.2, N), val = rate), l = 0.001)
    fees_keep = Fees(; tn = Turnover(; w = fill(0.2, 4), val = rate[keep]), l = 0.001)
    res = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = prn, slv = slv, fees = fees)), rd)
    resk = optimise(MeanRisk(;
                             opt = JuMPOptimiser(; pe = prk, slv = slv, fees = fees_keep)),
                    rdk)
    pred = @test_logs min_level=Logging.Warn predict(res, rdc, test_idx;
                                                     wd = SelfFinancingDrift())
    predk = predict(resk, rdk, test_idx; wd = SelfFinancingDrift())
    @test isapprox(res.w[keep], resk.w; atol = 1e-6)
    r = LowOrderMoment(; alg = SecondMoment())
    # The per asset split is on the caller's universe, with a zero column at the dead asset,
    # and its rows still sum to the fold's series.
    nar = calc_net_asset_returns(pred)
    @test size(nar, 2) == N
    @test all(iszero, view(nar, :, k))
    @test isapprox(vec(sum(nar; dims = 2)), pred.rd.X)
    # The risk contributions expand, and the dead asset reports exactly zero.
    rc = risk_contribution(r, pred)
    @test length(rc) == N
    @test rc[k] == 0
    @test isapprox(rc[keep], risk_contribution(r, predk); rtol = 1e-4)
    mrc = risk_contribution(r, pred; marginal = true)
    @test length(mrc) == N
    @test mrc[k] == 0
    # The factor contributions fit the loadings over the live assets alone, from the fold's
    # own block and from a caller's `rd` on the full universe alike.
    F = X[:, keep] * [0.5, 0.3, 0.1, 0.1] .+ 0.001 .* randn(StableRNG(54), T)
    rdf = ReturnsResult(; nx = nx, X = Xc, nf = ["f"], F = reshape(F, :, 1))
    rdfk = ReturnsResult(; nx = nx[keep], X = X[:, keep], nf = ["f"], F = reshape(F, :, 1))
    predf = @test_logs min_level=Logging.Warn predict(res, rdf, test_idx;
                                                      wd = SelfFinancingDrift())
    predfk = predict(resk, rdfk, test_idx; wd = SelfFinancingDrift())
    frc = factor_risk_contribution(r, predf)
    @test length(frc) == 2
    @test isapprox(frc, factor_risk_contribution(r, predfk); rtol = 1e-4)
    @test isapprox(factor_risk_contribution(r, predf; rd = rdf),
                   factor_risk_contribution(r, predfk; rd = rdfk); rtol = 1e-4)
    # A caller's own `fees` is stated on the caller's universe, as a caller's `rd` is, and
    # the three consumers view it at the result's mask through `fold_fees`. A per-asset rate
    # on the full universe therefore charges the live assets at their own rates, exactly as
    # the same rate stated on the hand-reduced universe does; unviewed it met the reduced
    # weights with a `BoundsError`.
    lrate = [0.001, 0.002, 0.010, 0.003, 0.004]
    fees_c = Fees(; l = lrate)
    fees_ck = Fees(; l = lrate[keep])
    narc = calc_net_asset_returns(pred, fees_c)
    @test all(iszero, view(narc, :, k))
    @test isapprox(narc[:, keep], calc_net_asset_returns(predk, fees_ck); atol = 1e-6)
    @test isapprox(risk_contribution(r, pred, fees_c)[keep],
                   risk_contribution(r, predk, fees_ck); rtol = 1e-4)
    @test isapprox(factor_risk_contribution(r, predf, fees_c),
                   factor_risk_contribution(r, predfk, fees_ck); rtol = 1e-4)
    # A caller's carrier on the full universe lands on the complement, where the exit is,
    # and nowhere else: the dead asset's column holds the exit charge alone, and a fold whose
    # result reduced nothing strips the carrier rather than liquidating the whole book.
    fees_lq = Fees(; lq = Turnover(; w = fill(0.2, N), val = lrate))
    narq = calc_net_asset_returns(pred, fees_lq)
    @test all(==(-0.2 * lrate[k]), view(narq, :, k))
    @test isapprox(narq[:, keep], calc_net_asset_returns(predk, Fees()); atol = 1e-6)
    res_full = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv)), rd)
    pred_full = predict(res_full, rd, test_idx; wd = SelfFinancingDrift())
    @test isnothing(PO.result_investable_mask(res_full))
    @test isapprox(calc_net_asset_returns(pred_full, fees_lq),
                   calc_net_asset_returns(pred_full); atol = 1e-12)
    @test isnothing(PO.fold_fees(res_full, fees_lq, pred_full.hw.X).lq)
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
documented compaction identity: dropping the gaps first is what reproduces the oracle's
drop-per-column answer. One measure of each kernel class.
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
# The oracle fills the same scenario cells with zero and warns once for each asset whose own
# filled share is above a fixed `0.05`; it never refuses, so `fill_limit = 0.05` gives its
# decision to warn or stay silent (#1416). It is silent only on its portfolio return series,
# which it fills at one line. The matrix-wide share -- filled entries over the entries of the
# whole returns matrix -- is pinned below as the context the message reports rather than as
# the number that trips.

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
ce = ExpWeightedCovariance(; decay = 0.9, min_obs = 2, centring = PreCentred())
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
    @test_logs (:warn,
                r"whose graph reader reads the prior, `x_src = :prior`, may branch the asset alone") prior(pe5,
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
    # The clustering states the prior as its source, so it reads the filled matrix too. A
    # clustering of the returns data meets the gap itself (ADR 0211, decision 8).
    resh = optimise(HierarchicalRiskParity(; r = ConditionalValueatRisk(),
                                           opt = HierarchicalOptimiser(; pe = pe,
                                                                       x_src = :prior)), rd)
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

@testset "A graph reader decides a gap with its own moment estimator (#1694)" begin
    # The prior fills the gap of the young asset `d` in its own returns only. The clustering
    # reads the returns data, which keep the gap, so its `ce` decides (ADR 0211, decision 8).
    rd = ReturnsResult(; nx = nx, X = Xmix, pnl = pnl_mix)
    hrp(; kwargs...) = HierarchicalRiskParity(; r = ConditionalValueatRisk(),
                                              opt = HierarchicalOptimiser(; pe = pe,
                                                                          kwargs...))
    cvg_ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = CoveragePolicy()))
    left_c = (:info, r"left the investable universe.*\[\"c\"\]")

    # A plain clustering leaves `d` out of the Investable Mask at the entry of the head, and
    # names it with the two remedies.
    plain = (:warn,
             r"plain moment estimator, and the rows that it reads hold a gap for 1 asset\(s\) that the prior kept: \[\"d\"\].*`CoveragePolicy`.*`x_src = :prior`")
    res = @test_logs plain (:info, r"left the investable universe.*\[\"c\", \"d\"\]") optimise(hrp(),
                                                                                               rd)
    @test res.imsk == BitVector([1, 1, 0, 0])
    @test iszero(res.w[4])
    @test isapprox(sum(res.w), 1)
    # Under `strict` the head refuses the gap.
    @test_throws IsNonFiniteError optimise(hrp(; strict = true), rd)

    # A mask-aware clustering reads the observed cells, and the prior source reads the
    # filled matrix. Both keep `d`, and only the dead asset departs.
    aware = @test_logs left_c optimise(hrp(; cle = ClustersEstimator(; ce = cvg_ce)), rd)
    @test aware.imsk == BitVector([1, 1, 0, 1])
    @test aware.w[4] > 0
    filled = @test_logs left_c optimise(hrp(; x_src = :prior), rd)
    @test filled.imsk == aware.imsk
    @test filled.w[4] > 0

    # The JuMP head narrows on its phylogeny estimator as the hierarchical head does.
    mr = MeanRisk(;
                  opt = JuMPOptimiser(; pe = pe, slv = slv,
                                      ple = SemiDefinitePhylogenyEstimator()))
    resj = @test_logs plain (:info, r"left the investable universe") match_mode = :any optimise(mr,
                                                                                                rd)
    @test resj.imsk == BitVector([1, 1, 0, 0])
    @test iszero(resj.w[4])

    # Every asset that the prior kept holds a gap, so the plain clustering leaves none.
    Xall = copy(X0)
    amsk_all = trues(T, N)
    for k in 1:N
        Xall[k, k] = NaN
        amsk_all[k, k] = false
    end
    rd_all = ReturnsResult(; nx = nx, X = Xall, pnl = make_panel(amsk_all))
    @test isnothing(PO.investable_mask(prior(pe, rd_all)))
    @test_throws IsEmptyError optimise(hrp(), rd_all)

    # A direct call has no entry of a fit, so it refuses the gap and names the remedies.
    pr = prior(pe, rd)
    pr3 = PO.port_opt_view(pr, [1, 2, 4])
    rd3 = PO.port_opt_view(rd, [1, 2, 4])
    @test_throws IsNonFiniteError clusterise(ClustersEstimator(), pr3; rd = rd3)
    @test_throws "x_src = :prior" clusterise(ClustersEstimator(), pr3; rd = rd3)
    @test isa(clusterise(ClustersEstimator(), pr3; rd = rd3, x_src = :prior),
              PO.AbstractClusteringResult)
    @test isa(clusterise(ClustersEstimator(; ce = cvg_ce), pr3; rd = rd3),
              PO.AbstractClusteringResult)

    # A `ce` is mask-aware when a policy sits anywhere in its tree, so the wrapper reads the
    # gapped sample as the `Covariance` inside it does.
    X3 = Xmix[:, [1, 2, 4]]
    @test isnan(PO.gap_fill_value(cvg_ce))
    @test iszero(PO.gap_fill_value(PortfolioOptimisersCovariance()))
    @test cor(cvg_ce, X3) == cor(Covariance(; cvg = CoveragePolicy()), X3)

    # The readers that the heads hold, one method each.
    @test !PO.plain_graph_reader(nothing)
    @test PO.plain_graph_reader(ClustersEstimator())
    @test !PO.plain_graph_reader(ClustersEstimator(; ce = cvg_ce))
    @test !PO.plain_graph_reader(ClustersEstimator(; ce = ExpWeightedCovariance()))
    @test !PO.plain_graph_reader(ClustersEstimator(; de = FeatureDistance()))
    @test PO.plain_graph_reader(NetworkClustersEstimator())
    @test !PO.plain_graph_reader(SemiDefinitePhylogenyEstimator(;
                                                                pl = NetworkEstimator(;
                                                                                      ce = cvg_ce)))
    @test PO.plain_graph_reader((nothing, [CentralityConstraint()]))
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
    # of coverage `c` whose asset carries at least `floor(alpha * T) + 1` losses, the measure at
    # level `alpha` over the FILLED column equals the measure at level `alpha / c` over that
    # column's OBSERVED rows -- to the last bit, because `alpha * T` and `(alpha / c) * (c * T)`
    # are the same count, the same order statistic and the same denominator.
    #
    # The shares are exact binary fractions so that the two `floor(Int, alpha * T)` agree
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
    # The column carries more than the five losses the identity needs: the four of the tail
    # and the boundary loss, which the tail takes at zero weight.
    @test count(<(0), observed) >= floor(Int, alpha * Td) + 1

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
