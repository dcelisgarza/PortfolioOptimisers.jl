#=
The variance split of a Leverage-One Pair in a factor attribution (#1579, decided by #1577 on map
#1375).

A pair that the fit marks in `h1` has a direction of the design of its own, so its total is exact
and its split between the systematic and the idiosyncratic part is not identified. Under the
default `EntrywiseUnknown()` every variance number that reads the split of a held pair is `NaN`.
`ZeroUnknown()`, or `KindwiseUnknown(; leverage = ZeroUnknown())`, reads the split as fitted, which
is the oracle's attribution.

The measured case is `parity_panel(; T = 1200, N = 40, seed = 1471)`, first 880 rows, with the
factors of map #1562. Asset 3 is the only member of the Utilities level at all 607 rows of the
block, so the panel marks no other pair: the last testset marks a pair of asset 5 by hand.
=#
using Test, PortfolioOptimisers, LinearAlgebra, Logging, Statistics
const PO = PortfolioOptimisers
include(joinpath(@__DIR__, "parity_harness.jl"))

# The variance numbers of a component, and the numbers the rule keeps.
split_numbers(c) = [c.vol, c.vol_contrib, c.pct_var, c.corr]
quiet(f) = with_logger(f, NullLogger())
# The parts of a Result hold arrays, so two equal parts compare field by field.
function same(a, b)
    return a === b ||
           all(isequal(getfield(a, k), getfield(b, k)) for k in fieldnames(typeof(a)))
end

@testset "The rule of each kind, helper by helper (#1579)" begin
    @test PO.attribution_unknown_kinds(EntrywiseUnknown()) ===
          (; unstated = EntrywiseUnknown(), leverage = EntrywiseUnknown())
    @test PO.attribution_unknown_kinds(ZeroUnknown()) ===
          (; unstated = ZeroUnknown(), leverage = ZeroUnknown())
    @test PO.attribution_unknown_kinds(KindwiseUnknown(; leverage = ZeroUnknown())) ===
          (; unstated = EntrywiseUnknown(), leverage = ZeroUnknown())
    @test KindwiseUnknown() === KindwiseUnknown(EntrywiseUnknown(), EntrywiseUnknown())
    @test_throws ArgumentError KindwiseUnknown(; unstated = KindwiseUnknown())
    @test_throws ArgumentError KindwiseUnknown(; leverage = KindwiseUnknown())

    # Asset 2 is the only member of factor 2. Factor 1 has three members, and a `NaN` or a zero
    # exposure is no membership.
    B = [1.0 0.0 0.5; 1.0 2.0 0.0; 1.0 0.0 NaN]
    @test PO.attribution_sole_factors!(falses(3), B, 2) == [false, true, false]
    @test PO.attribution_sole_factors!(falses(3), B, 1) == [false, false, true]

    lev = [false, true, false]
    @test isnothing(PO.attribution_leverage_split(ZeroUnknown(), lev, [0.2, 0.5, 0.3], B))
    @test isnothing(PO.attribution_leverage_split(EntrywiseUnknown(), nothing,
                                                  [0.2, 0.5, 0.3], B))
    @test isnothing(PO.attribution_leverage_split(EntrywiseUnknown(), lev, [0.5, 0.0, 0.5],
                                                  B))
    lv = PO.attribution_leverage_split(EntrywiseUnknown(), lev, [0.2, 0.5, 0.3], B)
    @test lv.assets == lev && lv.factors == [false, true, false]

    # The realised method reads the mask row by row: a weight of zero at the marked row is no
    # holding.
    h1 = BitMatrix([0 0 0; 0 1 0])
    Bh = permutedims(cat(B, B; dims = 3), (3, 1, 2))
    @test isnothing(PO.attribution_leverage_split(EntrywiseUnknown(), h1,
                                                  [0.5 0.5 0.0; 0.5 0.0 0.5], Bh))
    lv = PO.attribution_leverage_split(EntrywiseUnknown(), h1, [0.5 0.0 0.5; 0.5 0.5 0.0],
                                       Bh)
    @test lv.assets == lev && lv.factors == [false, true, false]
    @test PO.attribution_marked_assets(h1) == lev
    @test isnothing(PO.attribution_marked_assets(nothing))
end

@testset "The leverage test of a standard error, helper by helper (#1580)" begin
    ses = (; sys = 0.5, factor = [0.1, 0.2, NaN], family = nothing)
    # A rule that reads the plug-in variance, or a block with no mask, runs no more passes.
    nopass(s2) = error("no pass")
    reg = trues(2, 3)
    @test PO.attribution_leverage_errors(ZeroUnknown(), falses(2, 3), reg, ses, nopass) ===
          ses
    @test PO.attribution_leverage_errors(EntrywiseUnknown(), nothing, reg, ses, nopass) ===
          ses
    # A mask of pairs outside the regression marks nothing.
    h1 = BitMatrix([0 0 1; 0 0 0])
    regh = BitMatrix([1 1 0; 1 1 1])
    @test PO.attribution_leverage_errors(EntrywiseUnknown(), h1, regh, ses, nopass) === ses
    # A marked pair of the regression: the first pass reads its indicator, the second every
    # pair of the regression.
    lev = (; sys = 1.0, factor = [0.0, 1.0, NaN], family = nothing)
    scl = (; sys = 1.0, factor = [1.0, 1.0, NaN], family = nothing)
    stub(s2) = count(s2) == 1 ? lev : scl
    out = PO.attribution_leverage_errors(EntrywiseUnknown(), h1, reg, ses, stub)
    @test isnan(out.sys) && isequal(out.factor, [0.1, NaN, NaN]) && isnothing(out.family)
    # The ratio of the squares against `eps`: a ratio of round-off keeps the error, a ratio
    # above `eps` is `NaN`, and a `NaN` or a missing family axis stays as it is.
    e = eps(Float64)
    @test PO.attribution_leverage_error_nan(0.5, sqrt(e / 10), 1.0) == 0.5
    @test isnan(PO.attribution_leverage_error_nan(0.5, sqrt(10e), 1.0))
    @test PO.attribution_leverage_error_nan(0.5, 0.0, 0.0) == 0.5
    @test isnothing(PO.attribution_leverage_error_nan(nothing, nothing, nothing))
    @test isequal(PO.attribution_leverage_error_nan([0.1, 0.2, NaN], [0.0, 1.0, NaN],
                                                    [1.0, 1.0, NaN]), [0.1, NaN, NaN])
    @test PO.attribution_leverage_error_nan(0.5f0, sqrt(10e), 1.0) == 0.5f0
end

@testset "The bare-array predicted method takes the marked assets (#1579)" begin
    B = [1.0 0.0; 1.0 1.0; 1.0 0.0]
    F = [4.0e-4 1.0e-5; 1.0e-5 2.0e-4]
    d = [1.0e-4, 0.0, 2.0e-4]
    lev = [false, true, false]
    w = [0.3, 0.4, 0.3]
    fz = factor_attribution(w, B, F, d; assets = true)
    fe = factor_attribution(w, B, F, d; lev = lev, assets = true)
    # `ZeroUnknown()` for the kind of the pair gives the numbers of an unmarked model.
    fk = factor_attribution(w, B, F, d; lev = lev, assets = true,
                            unknown = KindwiseUnknown(; leverage = ZeroUnknown()))
    for c in (:sys, :idio, :unattr, :total, :fbd, :abd, :afc)
        @test same(getfield(fk, c), getfield(fz, c))
    end
    @test all(isnan, split_numbers(fe.sys)) && all(isnan, split_numbers(fe.idio))
    @test fe.sys.mu_contrib == fz.sys.mu_contrib && fe.idio.mu_contrib == fz.idio.mu_contrib
    @test same(fe.total, fz.total) && same(fe.unattr, fz.unattr)
    # Factor 2 has asset 2 alone, so its split is unknown. Factor 1 keeps its numbers.
    @test isnan.(fe.fbd.vol_contrib) == [false, true]
    @test isnan.(fe.fbd.pct_var) == [false, true] && isnan.(fe.fbd.corr) == [false, true]
    @test fe.fbd.vol == fz.fbd.vol && fe.fbd.mu_contrib == fz.fbd.mu_contrib
    @test fe.fbd.vol_contrib[1] == fz.fbd.vol_contrib[1]
    @test isnan.(fe.abd.sys_vol_contrib) == lev && isnan.(fe.abd.idio_vol_contrib) == lev
    @test fe.abd.vol_contrib == fz.abd.vol_contrib
    @test isnan.(fe.afc.vol_contrib) == [false false; false true; false false]
    # A portfolio that does not hold the pair is unchanged.
    w0 = [0.5, 0.0, 0.5]
    @test same(factor_attribution(w0, B, F, d; lev = lev, assets = true).sys,
               factor_attribution(w0, B, F, d; assets = true).sys)
end

@testset "The measured case of #1577" begin
    sty(dsc) = CompositeExposure(; descriptors = [dsc], family = "style")
    factors = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "size" => sty(LogMarketCap()), "value" => sty(BookToPrice()),
               "momentum" => sty(RollingMomentum()), "reversal" => sty(Reversal())]
    rd = PO.port_opt_view(parity_panel(; T = 1200, N = 40, seed = 1471).rd, 1:880, :)
    pe = CrossSectionalFactorPrior(; factors = factors, families = ["industry" => nothing],
                                   minra = 5)
    pr = prior(pe, rd)
    h1 = pr.rr.csr.h1
    @test count(h1) == 607 && all(view(h1, :, 3)) && size(h1) == (607, 40)
    N = 40
    w3 = zeros(N)
    w3[3] = 1
    w5 = zeros(N)
    w5[5] = 1
    weq = fill(1 / N, N)
    # Utilities is the level whose only member is asset 3, factor 5 of the raw axis.
    ut = 5

    @testset "ZeroUnknown() is the oracle's attribution" begin
        zp = factor_attribution(w3, pr; unknown = ZeroUnknown())
        @test zp.sys.pct_var ≈ 1.0 rtol = 1e-12
        @test zp.fbd.pct_var[ut] ≈ 0.8604 atol = 5e-5
        zr = quiet(() -> factor_attribution(w3, pr, rd; unknown = ZeroUnknown()))
        @test zr.sys.pct_var ≈ 1.0 rtol = 1e-12
        @test zr.fbd.pct_var[ut] ≈ 0.8265 atol = 5e-5
        z5 = factor_attribution(w5, pr, rd; unknown = ZeroUnknown())
        @test z5.sys.pct_var ≈ 0.6436 atol = 5e-5
        @test z5.idio.pct_var ≈ 0.3564 atol = 5e-5
        zq = quiet(() -> factor_attribution(weq, pr, rd; unknown = ZeroUnknown()))
        @test zq.sys.pct_var ≈ 0.9775 atol = 5e-5
        @test zq.idio.pct_var ≈ -0.00211 atol = 5e-6
        # The preset and the rule per kind agree when the unstated kind reads nothing.
        kq = quiet(() -> factor_attribution(weq, pr, rd;
                                            unknown = KindwiseUnknown(;
                                                                      unstated = ZeroUnknown(),
                                                                      leverage = ZeroUnknown())))
        @test same(kq.sys, zq.sys) && same(kq.fbd, zq.fbd)
    end

    @testset "The default gives NaN for the split of a holder, $(side)" for side in
                                                                            (:predicted,
                                                                             :realised)
        args = side === :predicted ? (pr,) : (pr, rd)
        att(w, u) = quiet(() -> factor_attribution(w, args...; unknown = u, assets = true))
        # Asset 4 is in the warm-up of its variance, so a predicted holding of it is unknown for
        # the other kind of entry. The equal weight leaves it out on the predicted side.
        wq = copy(weq)
        if side === :predicted
            wq[4] = 0
            wq ./= sum(wq)
        end
        for w in (w3, wq)
            fe = att(w, EntrywiseUnknown())
            fz = att(w, KindwiseUnknown(; leverage = ZeroUnknown()))
            @test all(isnan, split_numbers(fe.sys)) && all(isnan, split_numbers(fe.idio))
            @test fe.sys.mu_contrib == fz.sys.mu_contrib
            @test fe.idio.mu_contrib == fz.idio.mu_contrib
            @test same(fe.total, fz.total) && same(fe.unattr, fz.unattr)
            @test findall(isnan, fe.fbd.pct_var) == [ut]
            @test findall(isnan, fe.fbd.vol_contrib) == [ut]
            @test isequal(fe.fbd.vol, fz.fbd.vol) && fe.fbd.mu_contrib == fz.fbd.mu_contrib
            @test findall(isnan, fe.abd.sys_vol_contrib) == [3]
            @test findall(isnan, fe.abd.idio_vol_contrib) == [3]
            @test isequal(fe.abd.vol_contrib, fz.abd.vol_contrib)
            @test findall(isnan, fe.afc.vol_contrib) == [CartesianIndex(3, ut)]
        end
        # The holder of asset 5 holds no marked pair, so the default changes nothing.
        f5 = att(w5, EntrywiseUnknown())
        z5 = att(w5, KindwiseUnknown(; leverage = ZeroUnknown()))
        for c in (:sys, :idio, :unattr, :total, :fbd, :fmbd, :abd, :afc)
            @test same(getfield(f5, c), getfield(z5, c))
        end
    end

    @testset "A pair marked by hand, on the bare-array realised method" begin
        # The panel marks asset 3 alone, at every row. Asset 5 shares its level, so a mark of it
        # gives the split of the asset and the portfolio, and no factor row.
        blk = PO.attribution_block_arrays(pr.rr, pr)
        Tb = size(blk.f, 1)
        X = ifelse.(isfinite.(rd.X), rd.X, 0.0)[(end - Tb + 1):end, :]
        W = repeat(transpose(w5), Tb)
        W[100:200, 5] .= 0
        W[100:200, 6] .= 1
        ret = vec(sum(W .* X; dims = 2))
        kw = (; lag = blk.lag, fam = blk.fam, assets = true)
        base = factor_attribution(W, blk.B, blk.f, blk.eps, ret; h1 = blk.h1, kw...)
        @test all(isfinite, split_numbers(base.sys))
        # Asset 5 marked only where the history holds none of it: unchanged.
        h1a = copy(blk.h1)
        h1a[150, 5] = true
        fa = factor_attribution(W, blk.B, blk.f, blk.eps, ret; h1 = h1a, kw...)
        for c in (:sys, :idio, :unattr, :total, :fbd, :fmbd, :abd, :afc)
            @test same(getfield(fa, c), getfield(base, c))
        end
        # Asset 5 marked at a held row: the split is unknown, the factor rows stay.
        h1b = copy(blk.h1)
        h1b[300, 5] = true
        fb = factor_attribution(W, blk.B, blk.f, blk.eps, ret; h1 = h1b, kw...)
        @test all(isnan, split_numbers(fb.sys)) && all(isnan, split_numbers(fb.idio))
        @test same(fb.total, base.total) && fb.sys.mu_contrib == base.sys.mu_contrib
        @test !any(isnan, fb.fbd.pct_var)
        @test findall(isnan, fb.abd.sys_vol_contrib) == [5]
        # The rolling twin applies the rule window by window.
        fr = factor_attribution(W, blk.B, blk.f, blk.eps, ret, 100; step = 100, h1 = h1b,
                                kw...)
        @test [isnan(x.sys.pct_var) for x in fr] ==
              [(t - 99) <= 300 - blk.lag <= t for t in 100:100:(Tb - blk.lag)]
    end

    @testset "The standard errors that read the pair are NaN (#1580)" begin
        # A variance estimate with no warm-up, so no error is `NaN` for an unstated variance
        # (#1388).
        ve1 = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(), min_obs = 1)
        pe1 = CrossSectionalFactorPrior(; factors = factors,
                                        families = ["industry" => nothing], minra = 5,
                                        ve = ve1)
        pr1 = prior(pe1, rd)
        blk = PO.attribution_block_arrays(pr1.rr, pr1)
        al = PO.attribution_align(blk, size(rd.X, 1))
        reg = al.act .& .!iszero.(al.rw)
        # The leverage index over the scale of each output, as squares of the two indicator
        # passes.
        function ratios(w)
            g = reduce(vcat,
                       [transpose(transpose(PO.attribution_slice(al.B, t)) * w)
                        for t in axes(al.f, 1)])
            red = PO.attribution_reduce_for_errors(al.fcb, al.B, g, al.no, size(g, 1))
            pass(s2) = PO.attribution_error_pass(s2, g, al, red, blk.fam, 1)
            l, s = pass(reg .& al.h1), pass(reg)
            return (; sys = abs2(l.sys) / abs2(s.sys),
                    factor = abs2.(l.factor) ./ abs2.(s.factor))
        end
        # Measured: a coefficient through the zero-sum constraint gives the market factor a
        # ratio of 2.4e-3 and Utilities 0.76. A coefficient of round-off gives a style factor
        # 2.2e-19 at most, and the holder of asset 5 a systematic ratio of zero.
        r3, r5 = ratios(w3), ratios(w5)
        @test r3.sys ≈ 1 rtol = 1e-10
        @test r3.factor[1] ≈ 2.42e-3 rtol = 1e-2
        @test r3.factor[ut] ≈ 0.7607 rtol = 1e-3
        @test r5.sys < 1e-18
        @test r5.factor[3] ≈ 1.42e-3 rtol = 1e-2
        for r in (r3, r5)
            @test all(<(1e-18), r.factor[6:9])
        end
        att(w, u) = quiet(() -> factor_attribution(w, pr1, rd; se = true, unknown = u))
        lz = KindwiseUnknown(; leverage = ZeroUnknown())
        for (w, sys, fac) in ((w3, true, [1, ut]), (w5, false, [1, 3]), (weq, true, 1:5))
            fe, fz = att(w, EntrywiseUnknown()), att(w, lz)
            @test isnan(fe.sys.mu_se) == sys && isnan(fe.idio.mu_se) == sys
            @test findall(isnan, fe.fbd.mu_se) == fac
            # Every other error keeps the plug-in value, and so do the style factors.
            @test all(isfinite, fz.fbd.mu_se) && isfinite(fz.sys.mu_se)
            keep = setdiff(1:9, fac)
            @test fe.fbd.mu_se[keep] == fz.fbd.mu_se[keep]
            @test isequal(isnan.(fe.fmbd.mu_se), [true, true, false])
            @test fe.fmbd.mu_se[3] == fz.fmbd.mu_se[3]
            # Every number that is not an error is the number of the split rule alone.
            fs = quiet(() -> factor_attribution(w, pr1, rd; unknown = EntrywiseUnknown()))
            @test same(fs.total, fe.total) && fs.sys.mu_contrib == fe.sys.mu_contrib
        end
        # A holder of asset 5 keeps its systematic error, equal to the plug-in one.
        @test att(w5, EntrywiseUnknown()).sys.mu_se == att(w5, lz).sys.mu_se
        # The oracle reads the plug-in variance of the pair, zero, so a holder has no error.
        @test iszero(att(w3, lz).sys.mu_se)
    end
end
