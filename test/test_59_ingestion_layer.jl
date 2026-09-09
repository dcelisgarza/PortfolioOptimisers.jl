#=
The ingestion layer's emission, issue #973 of map #955.

`PriceIngestion` assembles raw price series into the carrier the conversion reads, and the
conversion hands the returns carrier an `AssetPanel` stating the universe. What the file pins
is the set of statements ADR 0129, 0131 and 0132 make about that path, each one asserted
rather than assumed:

  - The layer emits **one carrier**, and it **always** carries a panel. A gapless price table
    is not a special case that skips one; it is a panel whose masks are all true. So
    `pnl === nothing` on a returns carrier means one thing only: the carrier was not built by
    the layer.

  - Saying so costs nothing. A gapless ingestion's two masks are one `AllTrueMask`, which
    stores no cell, and a gapped one's active mask is the projected `ListingSpan`, two
    integers per asset. Both are ordinary `AbstractMatrix{Bool}`s to every reader.

  - `emsk` is a **snapshot** taken after the conversion, not a view over `X`. A value-level
    rewrite of the returns therefore does not move the estimation universe under a fold that
    was already scored against it.

  - The alignment guarantee has **split**. The asset axis is fixed before the split, so
    `assert_universe_aligned` narrows to a provenance check: `nx` equality plus panel-presence
    parity, reaching only a carrier built outside the layer or a step that changes the asset
    set.

  - An `AssetPanel` with **no Panel Field** is the layer's common case. A caller holding only
    prices has no market capitalisation and no sector, and the panel states a universe and
    nothing else.

The closing test is the destination itself: a caller holding a gapped price table reaches a
walk-forward fold with no hand-built `ReturnsResult` and no hand-built mask.
=#
using Dates

# The fixture carries all three span cases in one table, because the Span Rule reads position.
# `a` is priced throughout, so no window's universe is ever empty.
T59, N59 = 40, 4
ts59 = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T59 - 1))
nx59 = ["a", "b", "c", "d"]
P59 = 100.0 .+ cumsum(randn(StableRNG(973), T59, N59) ./ 10; dims = 1)
P59[1:8, 2] .= NaN             # b lists at observation 9: a LEADING run of gaps
P59[33:40, 3] .= NaN           # c delists at observation 33: a TRAILING run of gaps
P59[18:20, 4] .= NaN           # d is suspended and prices again: an INTERIOR gap
X59 = TimeArray(collect(ts59), P59, nx59)

# The same table with every gap spelled the other way, which is what a wide table built from a
# tidy one leaves.
Pm59 = Matrix{Union{Missing, Float64}}(P59)
Pm59[.!isfinite.(P59)] .= missing
Xm59 = TimeArray(collect(ts59), Pm59, nx59)

# A table with no gap at all, the case the layer must still emit a panel for.
Pg59 = 100.0 .+ cumsum(randn(StableRNG(9731), T59, N59) ./ 10; dims = 1)
Xg59 = TimeArray(collect(ts59), Pg59, nx59)

ptr59 = PricesToReturns(; nan_to_missing = false)
rd59 = prices_to_returns(ptr59, price_ingestion(PriceIngestion(), X59))

@testset "PriceIngestion assembles the span-carrying price carrier" begin
    pr = price_ingestion(PriceIngestion(), X59)
    @test isa(pr, PricesResult)
    @test isa(pr.span, PortfolioOptimisers.ListingSpan)
    @test size(pr.span) == (T59, N59)

    # The Span Rule: leading and trailing runs fall outside the span, an interior one inside.
    expected = trues(T59, N59)
    expected[1:8, 2] .= false
    expected[33:40, 3] .= false
    @test Matrix(pr.span) == expected
    @test pr.span.first == [1, 9, 1, 1]
    @test pr.span.last == [40, 40, 32, 40]

    @testset "the two absent-price conventions unify, and unify to NaN" begin
        prm = price_ingestion(PriceIngestion(), Xm59)
        # A `missing` source and a `NaN` source give the same carrier and the same span.
        @test isa(values(prm.X), Matrix{Float64})
        @test Matrix(prm.span) == Matrix(pr.span)
        @test isequal(values(prm.X), values(pr.X))
        # The unification is what no deletion step reads, so the gaps survive the conversion.
        @test count(!isfinite, values(prm.X)) == count(!isfinite, P59)
        # A float table is already spelled that way and is returned untouched.
        @test PortfolioOptimisers.unify_gaps(X59) === X59
        @test !isa(PortfolioOptimisers.unify_gaps(Xm59), typeof(Xm59))
    end

    @testset "a caller's declaration replaces the Span Rule's answer outright" begin
        # A listing calendar that says `c` is listed to the end, against the gaps.
        own = trues(T59, N59)
        own[1:8, 2] .= false
        prd = price_ingestion(PriceIngestion(; span = own), X59)
        @test prd.span === own
        @test Matrix(prd.span) != Matrix(pr.span)
    end

    @testset "the join and the collapse move the clock, so they run here" begin
        # A factor priced on every second day: an outer join adds the asset rows it lacks and
        # pads them, and the emitted carrier has one clock for both blocks.
        F = TimeArray(collect(ts59)[1:2:end], reshape(P59[1:2:end, 1], :, 1), ["f"])
        prj = price_ingestion(PriceIngestion(), X59; F = F)
        @test TimeSeries.timestamp(prj.X) == TimeSeries.timestamp(prj.F)
        @test size(values(prj.X)) == (T59, N59)
        @test size(prj.span) == (T59, N59)
        # Half the factor rows are gaps the join padded, and the asset span is unchanged.
        @test count(!isfinite, values(prj.F)) == T59 - length(1:2:T59)
        @test Matrix(prj.span) == Matrix(pr.span)

        # A shared benchmark joins on the same clock, and it is one column whatever the
        # asset count is.
        B = TimeArray(collect(ts59), reshape(P59[:, 1], :, 1), ["bm"])
        prb = price_ingestion(PriceIngestion(), X59; F = F, B = B)
        @test TimeSeries.timestamp(prb.B) == TimeSeries.timestamp(prb.X)
        @test size(values(prb.B), 2) == 1
        @test Matrix(prb.span) == Matrix(pr.span)

        # A weekly collapse renumbers every observation, and the span is derived after it.
        prc = price_ingestion(PriceIngestion(; collapse_args = (Dates.week, first, last)),
                              X59)
        @test size(values(prc.X), 1) < T59
        @test size(prc.span) == size(values(prc.X))
    end

    @testset "implied volatilities are carried and clock-aligned, never converted" begin
        iv = TimeArray(collect(ts59), fill(0.2, T59, N59), nx59)
        pri = price_ingestion(PriceIngestion(), X59; iv = iv, ivpa = 1.5)
        @test TimeSeries.timestamp(pri.iv) == TimeSeries.timestamp(pri.X)
        @test all(values(pri.iv) .== 0.2)
        @test pri.ivpa == 1.5
        # The emitted clock must be a subset of the implied volatilities' own.
        ivs = iv[collect(ts59)[1:(T59 - 1)]]
        @test_throws PortfolioOptimisers.IsEmptyError price_ingestion(PriceIngestion(), X59;
                                                                      iv = ivs)
    end

    @testset "a PricesResult is re-ingested through the same verb" begin
        pr2 = price_ingestion(PriceIngestion(), PricesResult(; X = X59))
        @test Matrix(pr2.span) == Matrix(pr.span)
        @test_throws PortfolioOptimisers.IsEmptyError price_ingestion(PriceIngestion(),
                                                                      X59[Date[]])
    end
end

@testset "The layer emits one carrier, and it always carries a panel" begin
    pr = price_ingestion(PriceIngestion(), X59)
    rd = prices_to_returns(ptr59, pr)
    @test isa(rd, ReturnsResult)
    @test !isnothing(rd.pnl)
    @test rd.nx == nx59

    # `padding = false`, so the returns clock is one observation shorter than the prices.
    m = T59 - 1
    @test size(rd.X) == (m, N59)

    @testset "the active mask is the projected span, and the estimation mask follows the values" begin
        # A return consumes the earlier price of its pair, so the unpadded projection is
        # [first, last - 1]: b's inception emits no Held Gap, and c's delisting is booked one
        # observation earlier than its last price.
        amsk = falses(m, N59)
        amsk[:, 1] .= true
        amsk[9:m, 2] .= true
        amsk[1:31, 3] .= true
        amsk[:, 4] .= true
        @test Matrix(rd.pnl.amsk) == amsk
        # The estimation mask is the active mask intersected with finiteness, by construction.
        @test Matrix(rd.pnl.emsk) == amsk .& isfinite.(rd.X)
        # d's interior gap of k = 3 prices leaves k + 1 = 4 non-finite returns, and those four
        # are active but not estimable: the Held Gap.
        @test count(view(amsk, :, 4) .& .!view(Matrix(rd.pnl.emsk), :, 4)) == 4
    end

    @testset "the compressions: two integers per asset, and none at all" begin
        # A gapped ingestion keeps ADR 0129's exact compression across the clock crossing.
        @test isa(rd.pnl.amsk, PortfolioOptimisers.ListingSpan)
        # A gapless one says "all of them" in O(1) for both masks.
        rdg = prices_to_returns(ptr59, price_ingestion(PriceIngestion(), Xg59))
        @test isa(rdg.pnl.amsk, PortfolioOptimisers.AllTrueMask)
        @test rdg.pnl.emsk === rdg.pnl.amsk
        @test all(rdg.pnl.amsk)
        @test size(rdg.pnl.amsk) == (m, N59)
        # Both are ordinary matrices of booleans to every reader.
        @test isa(rdg.pnl.amsk, AbstractMatrix{Bool})
        @test Base.IndexStyle(typeof(rdg.pnl.amsk)) === IndexCartesian()
        @test sprint(show, rdg.pnl.amsk) == "AllTrueMask($m × $N59)"
        @test sprint(show, MIME"text/plain"(), rdg.pnl.amsk) == "AllTrueMask($m × $N59)"
        @test_throws DomainError PortfolioOptimisers.AllTrueMask(-1, 2)
        @test_throws BoundsError rdg.pnl.amsk[m + 1, 1]
    end

    @testset "the masks slice in step with the returns under port_opt_view" begin
        v = PortfolioOptimisers.port_opt_view(rd, 10:20, [1, 4])
        @test size(v.X) == (11, 2)
        @test size(v.pnl.amsk) == (11, 2)
        @test Matrix(v.pnl.amsk) == Matrix(rd.pnl.amsk)[10:20, [1, 4]]
        @test Matrix(v.pnl.emsk) == Matrix(rd.pnl.emsk)[10:20, [1, 4]]
        # A price carrier's span slices the same way, and it is viewed rather than re-derived:
        # `c`'s delisting still reads as a delisting inside a window that ends before it.
        pv = PortfolioOptimisers.port_opt_view(pr, 1:20, 1:N59)
        @test size(pv.span) == (20, N59)
        @test Matrix(pv.span) == Matrix(pr.span)[1:20, :]
        @test all(view(Matrix(pv.span), :, 3))
        # A row cut can split an interval in half, which no interval can say, so the cut span
        # is an ordinary view of booleans rather than a Listing Span.
        @test !isa(pv.span, PortfolioOptimisers.ListingSpan)
        # Keeping the whole clock in order leaves every interval intact, so the two integers
        # per asset survive the cut.
        pw = PortfolioOptimisers.port_opt_view(pr, 1:T59, [1, 3])
        @test isa(pw.span, PortfolioOptimisers.ListingSpan)
        @test Matrix(pw.span) == Matrix(pr.span)[:, [1, 3]]

        # A caller's own declaration is any matrix of booleans, and it slices the same way.
        own = trues(T59, N59)
        own[1:8, 2] .= false
        pd = PortfolioOptimisers.port_opt_view(price_ingestion(PriceIngestion(; span = own),
                                                               X59), 5:25, [2, 4])
        @test Matrix(pd.span) == own[5:25, [2, 4]]

        # A price-level filter that drops rows and columns cuts the span with them, and one
        # that only fills leaves it alone.
        mdf = fit_preprocessing(MissingDataFilter(; row_thr = 0.2), pr)
        pf = apply_preprocessing(mdf, pr)
        @test size(pf.span) == size(values(pf.X))
        imp = fit_preprocessing(Imputer(), pr)
        @test apply_preprocessing(imp, pr).span === pr.span
    end

    @testset "the estimation mask is a snapshot, not a view over the returns" begin
        # A value-level rewrite of every return leaves both masks exactly where they were.
        before_a = Matrix(rd.pnl.amsk)
        before_e = Matrix(rd.pnl.emsk)
        rd2 = ReturnsResult(; nx = rd.nx, X = fill(0.01, size(rd.X)), ts = rd.ts,
                            pnl = rd.pnl)
        @test Matrix(rd2.pnl.amsk) == before_a
        @test Matrix(rd2.pnl.emsk) == before_e
        # The snapshot still records the Held Gap the rewritten values no longer show.
        @test !all(rd2.pnl.emsk)
        @test all(isfinite, rd2.X)
    end

    @testset "a caller's Panel Fields are kept and only the masks are the layer's" begin
        # A static panel joins at the panel's observation count, lazily lifted.
        stat = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(N59))])
        prs = price_ingestion(PriceIngestion(), X59; pnl = stat)
        rds = prices_to_returns(ptr59, prs)
        @test length(rds.pnl.pf) == 1
        @test PortfolioOptimisers.panel_field(rds.pnl, "mcap") isa NumericPanelField
        @test size(PortfolioOptimisers.panel_field(rds.pnl, "mcap").vals) == (m, N59)
        @test Matrix(rds.pnl.amsk) == Matrix(rd.pnl.amsk)
        @test !PortfolioOptimisers.panel_is_static(rds.pnl)

        # A time-varying panel keeps its fields, and its masks are replaced rather than kept:
        # a declared estimation mask marking a non-finite return estimable cannot survive.
        tv = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(T59, N59))],
                        amsk = trues(T59, N59), emsk = trues(T59, N59))
        prt = price_ingestion(PriceIngestion(), X59; pnl = tv)
        rdt = prices_to_returns(ptr59, prt)
        @test length(rdt.pnl.pf) == 1
        @test !all(rdt.pnl.emsk)
        @test Matrix(rdt.pnl.emsk) == Matrix(rd.pnl.emsk)
    end
end

@testset "The layer's span bounds the layer's fill" begin
    # `PriceGapFill` asks the carrier for its Listing Span rather than deriving one, and a
    # carrier the layer built answers. That is what keeps a fill off a window's edge: a
    # window-local derivation reads `d`'s suspension at the edge as an inception there.
    pr = price_ingestion(PriceIngestion(), X59)
    @test PortfolioOptimisers.carrier_listing_span(pr) === pr.span
    @test isnothing(PortfolioOptimisers.carrier_listing_span(PricesResult(; X = X59)))

    res = fit_preprocessing(PriceGapFill(), pr)
    pf = apply_preprocessing(res, pr)
    # The fill states a price and not a listing, so the span rides through untouched and the
    # Span Rule reads the same listing off the filled panel as off the raw one.
    @test pf.span === pr.span
    @test Matrix(listing_span(values(pf.X))) == Matrix(pr.span)
    # `d`'s interior gap is filled, and neither `b`'s inception nor `c`'s delisting is.
    @test all(isfinite, view(values(pf.X), 18:20, 4))
    @test all(!isfinite, view(values(pf.X), 1:8, 2))
    @test all(!isfinite, view(values(pf.X), 33:40, 3))

    # A filled Held Gap is estimable, because a caller who filled has said the asset traded.
    rdf = prices_to_returns(ptr59, pf)
    @test all(view(Matrix(rdf.pnl.emsk), :, 4))
    @test !all(view(Matrix(rd59.pnl.emsk), :, 4))
end

@testset "An Asset Panel with no Panel Field states a universe and nothing else" begin
    a = trues(5, 3)
    a[1, 1] = false
    e = copy(a)
    e[2, 2] = false
    pnl = AssetPanel(; amsk = a, emsk = e)
    @test isempty(pnl.pf)
    @test PortfolioOptimisers.panel_axes(pnl) == (5, 3)
    @test !PortfolioOptimisers.panel_is_static(pnl)

    # It slices like any other panel, and the slice is still field-less.
    v = PortfolioOptimisers.port_opt_view(pnl, 2:4, [1, 3])
    @test isempty(v.pf)
    @test size(v.amsk) == (3, 2)

    # It derives a Feature Matrix with no column, because it holds no feature data.
    nz, Z = PortfolioOptimisers.panel_feature_matrix(pnl)
    @test isempty(nz)
    @test size(Z) == (5, 3, 0)
    @test isempty(PortfolioOptimisers.panel_feature_names(pnl))

    # A panel with neither a field nor a mask carries nothing at all and is refused.
    @test_throws PortfolioOptimisers.IsEmptyError AssetPanel()
    @test_throws PortfolioOptimisers.IsEmptyError AssetPanel(;
                                                             pf = PortfolioOptimisers.AbstractPanelField[])
    # A static panel still needs a field: nothing else would state its asset axis.
    @test PortfolioOptimisers.panel_axes([NumericPanelField(; name = "m", vals = ones(3))],
                                         nothing) == (3,)
end

@testset "A carrier the layer did not build states no universe" begin
    # No span and no gap: nothing to state, and no panel is emitted. This is what keeps
    # `pnl === nothing` meaning one thing only.
    @test isnothing(prices_to_returns(ptr59, PricesResult(; X = Xg59)).pnl)

    # No span and gaps: the span is derived from this window alone, and that warns.
    bare = PricesResult(; X = X59)
    rdw = @test_logs (:warn,) match_mode=:any prices_to_returns(ptr59, bare)
    @test !isnothing(rdw.pnl)
    @test_throws PortfolioOptimisers.IsNothingError prices_to_returns(PricesToReturns(;
                                                                                      nan_to_missing = false,
                                                                                      strict = true),
                                                                      bare)

    # A span whose gaps `nan_to_missing = true` would delete: warned about, refused under
    # strict, because the panel that survives says the universe was never gapped.
    pr = price_ingestion(PriceIngestion(), X59)
    rdn = @test_logs (:warn,) match_mode=:any prices_to_returns(PricesToReturns(), pr)
    @test all(rdn.pnl.emsk)
    @test_throws PortfolioOptimisers.ConflictingArgumentError prices_to_returns(PricesToReturns(;
                                                                                                strict = true),
                                                                                pr)

    # A span that does not fit the price clock is refused outright.
    @test_throws DimensionMismatch PricesResult(; X = X59, span = trues(T59 + 1, N59))
    @test_throws DimensionMismatch prices_to_returns(X59; span = trues(T59, N59 + 1),
                                                     nan_to_missing = false)
end

@testset "assert_universe_aligned narrows to a provenance check" begin
    pr = price_ingestion(PriceIngestion(), X59)
    pipe = Pipeline(; steps = (ptr59, EmpiricalPrior(), EqualWeighted()))
    res = fit(pipe, pr)
    rd = res.ctx.returns
    @test !isnothing(rd.pnl)
    @test PortfolioOptimisers.assert_universe_aligned(res, rd) === nothing

    # A window carrying a different asset set is still refused by name.
    other = ReturnsResult(; nx = nx59[1:3], X = rd.X[:, 1:3])
    @test_throws ArgumentError PortfolioOptimisers.assert_universe_aligned(res, other)

    # And so is one that states no universe where the fitted context states one, which is the
    # half the check gained: the two did not come from one ingestion.
    stripped = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts)
    @test_throws ArgumentError PortfolioOptimisers.assert_universe_aligned(res, stripped)

    # A fitted context with no returns slot has nothing to compare against, so the check
    # is silent rather than guessing.
    price_only = fit(Pipeline(; steps = (MissingDataFilter(),)), pr)
    @test isnothing(price_only.ctx.returns)
    @test PortfolioOptimisers.assert_universe_aligned(price_only, rd) === nothing
end

@testset "A gapped price table reaches a walk-forward fold with nothing hand-built" begin
    # The destination of map #955: no hand-built ReturnsResult, and no hand-built mask.
    pr = price_ingestion(PriceIngestion(), X59)
    pipe = Pipeline(; steps = (ptr59, EmpiricalPrior(), EqualWeighted()))
    iwf = IndexWalkForward(20, 8)
    p = cross_val_predict(pipe, pr, iwf)
    @test length(p.pred) >= 2
    rd = prices_to_returns(ptr59, pr)
    cv = split(iwf, pr)
    for (k, f) in pairs(p.pred)
        # Reduce-and-expand answers on the full universe, and the panel the layer emitted is
        # what drove the reduction: no mask was written by hand anywhere on this path.
        @test length(f.res.w) == N59
        @test all(isfinite, f.res.w)
        @test !isnothing(f.res.pr.pnl)

        # The fold's Investable Mask is the Coverage Universe the ingested panel states, and
        # every asset outside it holds exactly zero rather than a small number.
        tr = cv.train_idx[k]
        rows = tr[1]:(last(tr) - 1)
        cm = PortfolioOptimisers.coverage_mask(view(rd.X, rows, :),
                                               PortfolioOptimisers.port_opt_view(rd.pnl,
                                                                                 rows,
                                                                                 1:N59))
        @test f.res.imsk == cm
        @test all(iszero, view(f.res.w, .!cm))
        @test isapprox(sum(f.res.w), 1; rtol = 1e-8)
    end

    # The universe each fold states is the panel-wide one cut to the window, never one
    # re-derived from it: `c` is still listed inside a window that ends before its delisting.
    for tr in split(iwf, rd).train_idx
        v = PortfolioOptimisers.port_opt_view(rd, tr, 1:N59)
        @test Matrix(v.pnl.amsk) == Matrix(rd.pnl.amsk)[tr, :]
    end
end
