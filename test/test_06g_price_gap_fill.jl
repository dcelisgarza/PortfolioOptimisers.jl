#=
The span-bounded price fill (issue #970, map #955).

ADR 0130 gives the ingestion layer exactly one fill, off unless a caller asks for it. It exists to
state a price convention across a suspension rather than to remove a gap, it is bounded by the
Listing Span so it touches Held Gaps alone, and it is fitted on a training window and replayed.

`CarriedPrice` states the Held Price convention and conserves wealth across a gap; a
`Num_VecToScaM` reduction states a constant and manufactures two moves the market never printed.
Both are asserted below, so the difference between them is pinned rather than assumed.

The span the fill is bounded by is the carrier's, always. A carrier that states none bounds the
fill by nothing at all and no price is written, because the window cannot supply one: a suspension
straddling its edge reads there as an inception or a delisting, so a window-local derivation fills
the wrong cells rather than fewer of them (ADR 0133). The carriers below therefore state a span, as
`price_ingestion` builds them, and the span-less case is asserted in its own testset. The seed path
— a window that opens inside a gap — is driven through `gap_fill_span` and `gap_fill_column!` with
a stated span, which is the same public `AbstractMatrix{Bool}` bound a caller's listing calendar
enters at.
=#
function pgf_970_prices()
    # observations × assets, one column per case:
    #  A interior gap, k = 2   B leading gap (not yet listed)
    #  C trailing gap (delisted)   D a gap throughout
    return [100.0 NaN 300.0 NaN
            101.0 NaN 301.0 NaN
            102.0 200.0 302.0 NaN
            NaN 201.0 303.0 NaN
            NaN 202.0 304.0 NaN
            120.0 203.0 NaN NaN
            121.0 204.0 NaN NaN
            122.0 205.0 NaN NaN]
end
function pgf_970_carrier(X::AbstractMatrix)
    ts = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(size(X, 1) - 1))
    # The carrier states the Span Rule's reading of the whole panel, which is what
    # `price_ingestion` puts in the field. Nothing below relies on a window deriving one.
    return PricesResult(; X = TimeArray(collect(ts), X, ["A", "B", "C", "D"]),
                        span = listing_span(X))
end
function pgf_970_quiet_apply(res, pr)
    # A carrier that states a span reports nothing, so this is a belt-and-braces silence: the
    # one carrier that reports is the span-less one, and it has its own testset.
    return Logging.with_logger(Logging.NullLogger()) do
        return apply_preprocessing(res, pr)
    end
end
function pgf_970_returns(X::AbstractMatrix)
    T = size(X, 1)
    return X[2:T, :] ./ X[1:(T - 1), :] .- 1
end
@testset "A Held Price conserves wealth across a gap; a constant manufactures two moves" begin
    X = pgf_970_prices()
    pr = pgf_970_carrier(X)
    res = fit_preprocessing(PriceGapFill(), pr)
    @test isa(res, PriceGapFillResult)
    @test isa(res.fill, CarriedPrice)
    # A column with no observed price gets no fitted value and no entry, so it is untouched.
    @test res.nx == [:A, :B, :C]
    @test res.v == [122.0, 205.0, 304.0]
    Xc = values(pgf_970_quiet_apply(res, pr).X)
    # The k = 2 interior gap takes the last priced observation, and nothing else moved.
    @test Xc[4:5, 1] == [102.0, 102.0]
    @test Xc[[1, 2, 3, 6, 7, 8], 1] == X[[1, 2, 3, 6, 7, 8], 1]
    Rc = pgf_970_returns(Xc)
    # `k` zero returns inside the gap, and the whole move on the observation that ends it.
    @test Rc[3:4, 1] == [0.0, 0.0]
    @test Rc[5, 1] ≈ 120.0 / 102.0 - 1
    # Wealth is conserved: the compounded product over the gap is exactly the price ratio.
    @test prod(1 .+ Rc[3:5, 1]) ≈ 120.0 / 102.0
    # The same gap under a per-asset reduction: two fabricated moves instead of one real one.
    resm = fit_preprocessing(PriceGapFill(; fill = MedianValue()), pr)
    @test resm.fill === MedianValue()
    @test resm.v[1] ≈ 111.0
    Xm = values(pgf_970_quiet_apply(resm, pr).X)
    @test Xm[4:5, 1] ≈ [111.0, 111.0]
    Rm = pgf_970_returns(Xm)
    @test Rm[3, 1] ≈ 111.0 / 102.0 - 1
    @test Rm[4, 1] == 0.0
    @test Rm[5, 1] ≈ 120.0 / 111.0 - 1
    # Wealth is conserved over the run either way — the conventions differ in where the move
    # lands, not in how much of it there is.
    @test prod(1 .+ Rm[3:5, 1]) ≈ 120.0 / 102.0
    # A plain number and a plain function reach the same slot.
    resn = fit_preprocessing(PriceGapFill(; fill = 99.0), pr)
    @test values(pgf_970_quiet_apply(resn, pr).X)[4:5, 1] == [99.0, 99.0]
    resf = fit_preprocessing(PriceGapFill(; fill = minimum), pr)
    @test values(pgf_970_quiet_apply(resf, pr).X)[4:5, 1] == [100.0, 100.0]
end
@testset "The fill is bounded by the Listing Span under both conventions" begin
    X = pgf_970_prices()
    pr = pgf_970_carrier(X)
    for fill in (CarriedPrice(), MedianValue())
        res = fit_preprocessing(PriceGapFill(; fill = fill), pr)
        Xf = values(pgf_970_quiet_apply(res, pr).X)
        # A leading run is an asset not yet listed and a trailing run is a delisting. Neither is
        # inside the span, so neither is written whatever the convention states.
        @test all(isnan, Xf[1:2, 2])
        @test all(isnan, Xf[6:8, 3])
        # A column with no observed price at all is untouched.
        @test all(isnan, Xf[:, 4])
        # The span the Span Rule reads off the filled panel is the one it read off the raw panel,
        # so the fill cannot erase a delisting or invent a listing.
        span_raw = listing_span(X)
        span_filled = listing_span(Xf)
        @test span_filled.first == span_raw.first
        @test span_filled.last == span_raw.last
        # The active mask still reports the leading and the trailing run inactive after the fill.
        amsk, emsk = universe_masks(span_filled, pgf_970_returns(Xf))
        @test !any(Matrix(amsk)[1:2, 2])
        @test !any(Matrix(amsk)[5:7, 3])
        # A filled cell is finite in the returns, so its estimation mask entry is true — the
        # design ADR 0130 states, not an oversight — and the subset invariant still holds.
        @test all(emsk[3:5, 1])
        @test all(Matrix(emsk) .<= Matrix(amsk))
        @test isnothing(PortfolioOptimisers.assert_panel_masks(size(amsk), amsk, emsk))
    end
end
@testset "A window that opens inside a gap fills from the fitted seed" begin
    X = pgf_970_prices()
    # Fit on the head of the panel, then replay on a tail whose first observations are a gap. The
    # tail states a span of its own — the public `AbstractMatrix{Bool}` bound, which is where a
    # caller's listing calendar enters — saying the asset is listed throughout it.
    res = fit_preprocessing(PriceGapFill(), pgf_970_carrier(X[1:3, :]))
    @test res.v[1] == 102.0
    # The result records where the training window ended, which is what tells a replay
    # whether the seed precedes the window it is written onto.
    @test res.te == Date(2020, 1, 3)
    tail = copy(X[4:8, :])
    # A deliberately different in-window price: were the fill reading the window rather than the
    # seed, the two gapped observations would take 120.0 and not 102.0.
    @test tail[3, 1] == 120.0
    span = trues(size(tail))
    @test PortfolioOptimisers.gap_fill_span(span, tail, false) === span
    # Every observation of the tail follows the training window, so the seed applies from
    # its first row.
    PortfolioOptimisers.gap_fill_column!(res.fill, tail, span, 1, res.v[1], 1)
    @test tail[1:2, 1] == [102.0, 102.0]
    @test tail[3:5, 1] == X[6:8, 1]
    # Once the window observes a price of its own, the convention tracks it: a later gap fills
    # from the most recent observed price rather than from the seed.
    later = [NaN, 130.0, NaN, 140.0, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(later, 5, 1), trues(5, 1),
                                         1, 102.0, 1)
    @test later == [102.0, 130.0, 130.0, 140.0, 140.0]
    # A reduction carries nothing, so every gap of the column takes the one fitted value.
    flat = [NaN, 130.0, NaN, 140.0, NaN]
    PortfolioOptimisers.gap_fill_column!(MedianValue(), reshape(flat, 5, 1), trues(5, 1), 1,
                                         111.0, 1)
    @test flat == [111.0, 130.0, 111.0, 140.0, 111.0]
    # The bound is read before the price, so a stated span that excludes an observation keeps the
    # fill out of it whatever the convention says.
    bounded = [NaN, NaN, 130.0, NaN, NaN]
    span_b = reshape(Bool[0, 1, 1, 1, 0], 5, 1)
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(bounded, 5, 1), span_b, 1,
                                         102.0, 1)
    @test isnan(bounded[1])
    @test isnan(bounded[5])
    @test bounded[2:4] == [102.0, 130.0, 130.0]
end
@testset "The seed is written only after the training window (#1068)" begin
    # Issue #1068. The seed is the last observed training price, so on a window that follows
    # the training window it precedes every row it is written onto. On the training window
    # itself, which a Pipeline transforms with the step it just fitted, it is a price from the
    # window's end, and a gap that opens the window inside an asset's suspension must not take
    # it: the first return the conversion would compute is a move the market never printed,
    # computed with a price observations in the future.
    ts = Date(2020, 1, 1):Day(1):Date(2020, 1, 12)
    P = [100.0 NaN 50.0 10.0; 101.0 NaN 51.0 10.1; 102.0 NaN 52.0 10.2; 103.0 20.0 53.0 10.3
         NaN 20.5 54.0 10.4; NaN 21.0 55.0 10.5; 106.0 21.5 56.0 10.6; 107.0 22.0 57.0 10.7
         108.0 22.5 58.0 10.8; 109.0 23.0 NaN 10.9; 110.0 23.5 NaN 11.0;
         111.0 24.0 NaN 11.1]
    pr = price_ingestion(PriceIngestion(), TimeArray(collect(ts), P, ["A", "B", "C", "D"]))
    # A training window that opens inside A's suspension: the span is `true` there and the
    # price is `NaN`.
    w = PortfolioOptimisers.port_opt_view(pr, 5:12)
    @test all(Matrix(w.span)[1:2, 1])
    @test all(isnan, values(w.X)[1:2, 1])
    res = fit_preprocessing(PriceGapFill(), w)
    a = findfirst(==(:A), res.nx)
    @test res.v[a] == 111.0
    @test res.te == Date(2020, 1, 12)
    # On the training window the two leading cells stay a Held Gap: no price precedes them.
    # Every later cell of the column is observed, and the other columns are untouched.
    Xw = values(apply_preprocessing(res, w).X)
    @test all(isnan, Xw[1:2, 1])
    @test Xw[3:8, 1] == P[7:12, 1]
    @test isequal(Xw[:, 2:4], P[5:12, 2:4])
    # The returns the conversion computes for A then start at its first observed price, and
    # the -4.5 % move from 111 to 106 the seed manufactured is not among them.
    rd = prices_to_returns(PricesToReturns(), apply_preprocessing(res, w))
    ra = Matrix(rd.X)[:, 1]
    @test all(isnan, ra[1:2])
    @test ra[3] ≈ 107.0 / 106.0 - 1
    @test !any(x -> isfinite(x) && x ≈ 106.0 / 111.0 - 1, ra)
    # A window that follows the training window is seeded exactly as ADR 0130 describes: its
    # leading in-span gap takes the last training price.
    head = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(pr, 1:4))
    @test head.te == Date(2020, 1, 4)
    @test head.v[findfirst(==(:A), head.nx)] == 103.0
    Xt = values(apply_preprocessing(head, w).X)
    @test Xt[1:2, 1] == [103.0, 103.0]
    @test Xt[3:8, 1] == P[7:12, 1]
    # A window that overlaps the training window interpolates: the rows at or before its end
    # are not seeded, and the rows after it are. Fitted on 1:5, applied to 4:8, the gap at row
    # 5 (at the training end) stays, and the gap at row 6 (after it) takes the seed.
    mid = fit_preprocessing(PriceGapFill(), PortfolioOptimisers.port_opt_view(pr, 1:5))
    @test mid.te == Date(2020, 1, 5)
    @test mid.v[findfirst(==(:A), mid.nx)] == 103.0
    ov = PortfolioOptimisers.port_opt_view(pr, 5:8)
    Xo = values(apply_preprocessing(mid, ov).X)
    @test isnan(Xo[1, 1])
    @test Xo[2, 1] == 103.0
    @test Xo[3:4, 1] == P[7:8, 1]
    # A gap the window's own earlier price precedes fills from that price whether or not the
    # seed applies: fitted and applied on 1:12, A's suspension takes 103 and C's trailing run
    # is outside the span and untouched.
    full = fit_preprocessing(PriceGapFill(), pr)
    Xf = values(apply_preprocessing(full, pr).X)
    @test Xf[5:6, 1] == [103.0, 103.0]
    @test all(isnan, Xf[10:12, 3])
    # The reduction arm is unaffected: replaying a fitted statistic on the window it was
    # fitted on is the meaning of fitted state, so the leading in-span gap takes the median.
    resm = fit_preprocessing(PriceGapFill(; fill = MedianValue()), w)
    Xm = values(apply_preprocessing(resm, w).X)
    @test Xm[1:2, 1] == fill(resm.v[findfirst(==(:A), resm.nx)], 2)
    # The walk itself: before `t0` a gap no observed price precedes stays a gap, from `t0` on
    # the seed is the carry, and an observed price replaces it either side of `t0`.
    col = [NaN, NaN, 130.0, NaN, NaN, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(col, 6, 1), trues(6, 1), 1,
                                         102.0, 5)
    @test all(isnan, col[1:2])
    @test col[3:6] == [130.0, 130.0, 130.0, 130.0]
    col2 = [NaN, NaN, NaN, NaN, 130.0, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(col2, 6, 1), trues(6, 1),
                                         1, 102.0, 3)
    @test all(isnan, col2[1:2])
    @test col2[3:6] == [102.0, 102.0, 130.0, 130.0]
    col3 = [NaN, NaN, NaN]
    PortfolioOptimisers.gap_fill_column!(CarriedPrice(), reshape(col3, 3, 1), trues(3, 1),
                                         1, 102.0, 4)
    @test all(isnan, col3)
end
@testset "A carrier that states no Listing Span fills nothing, and refuses under strict" begin
    X = pgf_970_prices()
    ts = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(size(X, 1) - 1))
    bare = PricesResult(; X = TimeArray(collect(ts), copy(X), ["A", "B", "C", "D"]))
    @test isnothing(PortfolioOptimisers.carrier_listing_span(bare))
    res = fit_preprocessing(PriceGapFill(), bare)
    @test !res.strict
    out = @test_logs (:warn,) match_mode = :any apply_preprocessing(res, bare)
    # An all-`false` span bounds the fill by nothing, so not one cell is written: the window
    # cannot answer the question, and answering it wrongly is worse than not answering it.
    @test isequal(values(out.X), X)
    @test PortfolioOptimisers.gap_fill_span(nothing, X, false) == falses(size(X))
    strict = fit_preprocessing(PriceGapFill(; strict = true), bare)
    @test strict.strict
    @test_throws ArgumentError apply_preprocessing(strict, bare)
    # A window that holds no gap has nothing to fill and nothing to get wrong, so a carrier
    # stating no span is silent there and the values pass through untouched.
    cleanX = 100.0 .+ collect(1.0:8.0) * ones(1, 4)
    clean = PricesResult(; X = TimeArray(collect(ts), cleanX, ["A", "B", "C", "D"]))
    quiet = fit_preprocessing(PriceGapFill(; strict = true), clean)
    cout = @test_logs apply_preprocessing(quiet, clean)
    @test values(cout.X) == values(clean.X)
    # The carrier is not mutated by an apply, and every field but `X` passes through.
    pr = pgf_970_carrier(X)
    @test all(isnan, values(pr.X)[4:5, 1])
    filled = pgf_970_quiet_apply(fit_preprocessing(PriceGapFill(), pr), pr)
    @test TimeSeries.timestamp(filled.X) == TimeSeries.timestamp(pr.X)
    @test TimeSeries.colnames(filled.X) == TimeSeries.colnames(pr.X)
    @test isnothing(filled.F) && isnothing(filled.B) && isnothing(filled.pnl)
    # A stated span must fit the window it bounds.
    @test_throws DimensionMismatch PortfolioOptimisers.gap_fill_span(trues(3, 4), X, false)
end
@testset "A fitted universe is replayed by name" begin
    X = pgf_970_prices()
    res = fit_preprocessing(PriceGapFill(), pgf_970_carrier(X))
    # A window that carries only some of the fitted names fills those and skips the rest.
    ts = Date(2021, 1, 1):Day(1):(Date(2021, 1, 1) + Day(size(X, 1) - 1))
    sub = PricesResult(; X = TimeArray(collect(ts), X[:, [1, 4]], ["A", "D"]),
                       span = listing_span(X[:, [1, 4]]))
    out = pgf_970_quiet_apply(res, sub)
    @test TimeSeries.colnames(out.X) == [:A, :D]
    @test values(out.X)[4:5, 1] == [102.0, 102.0]
    @test all(isnan, values(out.X)[:, 2])
    # A window whose names the fit never saw is left alone entirely.
    other = PricesResult(; X = TimeArray(collect(ts), X[:, 1:1], ["Z"]),
                         span = listing_span(X[:, 1:1]))
    @test isequal(values(pgf_970_quiet_apply(res, other).X), values(other.X))
end
