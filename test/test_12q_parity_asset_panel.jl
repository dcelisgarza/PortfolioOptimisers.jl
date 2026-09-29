#=
Parity of map #1375 for the Asset Panel, its builder, its views, the ingestion and the DataFrame
export (#1378). Every `Parity_*` file this test reads is an output of the oracle, stored with the
harness of #1376, and each testset states how its case was made.

A measured difference is an open child issue of the map, and the test pins today's value of it:

  - #1413: `ForwardPanelFill` and `BackwardPanelFill` carry a value across an inactive stretch of
    the active mask, where the oracle stops at the edge of the stretch.

#1414 is closed. A simple return is `(p_t - p_{t-1}) / p_{t-1}`, which is correctly rounded on all
58 finite returns of the ingestion case, where the oracle's `p_t / p_{t-1} - 1` is exact on 2. The
returns comparisons keep `rtol = 1e-12`, and the difference left is the oracle's own rounding.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

# One numeric Panel Field whose blanks sit at every edge the active mask makes.
function parity_fill_case()
    T, N = 12, 5
    rng = StableRNG(1378)
    amsk = trues(T, N)
    amsk[1:3, 2] .= false
    amsk[5:7, 3] .= false
    amsk[10:12, 4] .= false
    x = 10 .+ randn(rng, T, N)
    x[.!amsk] .= NaN
    # An active run of three blanks, the first active cell of a late listing, the last active
    # cell before an inactive stretch and the first after it, the last active cell before a
    # delisting, and a holiday.
    x[4:6, 1] .= NaN
    x[4, 2] = NaN
    x[4, 3] = NaN
    x[8, 3] = NaN
    x[9, 4] = NaN
    x[6, 5] = NaN
    X = ifelse.(amsk, 0.01 .* randn(rng, T, N), NaN)
    return (; T, N, amsk, x, X)
end
function parity_fill_field(fc, alg)
    pnl = asset_panel([NumericPanelInput(; name = "x", vals = fc.x, alg = alg)];
                      amsk = fc.amsk, emsk = fc.amsk)
    return panel_field(pnl, "x")
end

# One price table with a late listing, two interior gaps and a delisting.
function parity_ingest_case()
    rng = StableRNG(13780)
    T, N = 15, 5
    P = 100 .* exp.(cumsum(0.02 .* randn(rng, T, N); dims = 1))
    P[1:3, 2] .= NaN
    P[7:8, 3] .= NaN
    P[12:15, 4] .= NaN
    P[10, 5] = NaN
    ts = collect(Date(2020, 1, 1) .+ Day.(0:(T - 1)))
    return (; T, N, P, X = TimeArray(ts, P, ["a$i" for i in 1:N]))
end

@testset "Parity: the Asset Panel, its builder, its views, the ingestion and the export (#1378)" begin
    PO = PortfolioOptimisers
    fx = parity_small_panel()
    pnl = fx.rd.pnl
    A = pnl.amsk
    num = [f.name for f in pnl.pf if f isa NumericPanelField]
    # The oracle leaves a blank that its fill cannot reach as NaN, and ADR 0102 writes the fill
    # value there, with the observed mask false.
    blank_to_val(o, act, val) = ifelse.(act .& isnan.(o), val, o)

    @testset "Panel Fields, observed masks and the Feature Matrix" begin
        # Small panel, written raw. The oracle built its panel from the exchange and forward
        # filled each numeric field with no limit (Fields); Missing is its missing mask of the
        # raw field; OneHot is one 0/1 column per level of `industry`, then of `currency`.
        of = parity_load("asset_panel", "SmallRaw", "Fields")
        om = parity_load("asset_panel", "SmallRaw", "Missing") .== 1
        T, N = size(A)
        for (k, name) in enumerate(num)
            f = panel_field(pnl, name)
            cols = ((k - 1) * N + 1):(k * N)
            # Measured: maxrel = 0.0 on every field.
            r = parity_compare(ifelse.(A, f.vals, NaN),
                               ifelse.(A, blank_to_val(of[:, cols], A, 0.0), NaN);
                               name = name)
            @test r.ok
            @test (.!f.omsk .& A) == (om[:, cols] .& A)
        end
        nz, Z = panel_feature_matrix(pnl)
        oh = parity_load("panel_feature_matrix", "SmallRaw", "OneHot")
        lv = [("industry", l) for l in panel_field(pnl, "industry").levels]
        append!(lv, [("currency", l) for l in panel_field(pnl, "currency").levels])
        for (k, (name, l)) in enumerate(lv)
            @test vec(Z[:, :, findfirst(==("$(name)=$(l)"), nz)])[vec(A)] == oh[vec(A), k]
        end
        for (k, name) in enumerate(num)
            cols = ((k - 1) * N + 1):(k * N)
            @test Z[:, :, findfirst(==("$(name)::observed"), nz)][A] ==
                  Float64.(.!om[:, cols])[A]
            @test Z[:, :, findfirst(==(name), nz)][A] ==
                  blank_to_val(of[:, cols], A, 0.0)[A]
        end
    end

    @testset "The fill policies at every edge of the active mask" begin
        # The fill case, written raw. The oracle forward and backward filled `x`, with no limit
        # and with a limit of two.
        fc = parity_fill_case()
        cases = (("ForwardNone", ForwardPanelFill(), (8, 3), (3, 3)),
                 ("ForwardLim2", ForwardPanelFill(; lim = 2), nothing, nothing),
                 ("BackwardNone", BackwardPanelFill(), (4, 3), (9, 3)),
                 ("BackwardLim2", BackwardPanelFill(; lim = 2), nothing, nothing))
        for (case, alg, cross, src) in cases
            o = parity_load("asset_panel", "FillEdges", case)
            f = parity_fill_field(fc, alg)
            e = blank_to_val(o, fc.amsk, 0.0)
            if !isnothing(cross)
                # #1413: the oracle stops at the inactive stretch 5:7 of asset 3, and our fill
                # carries the value of the other side across it.
                @test isnan(o[cross...])
                @test f.vals[cross...] == fc.x[src...]
                e[cross...] = f.vals[cross...]
            end
            # Measured: maxrel = 0.0, bit-equal on every active cell that both fill.
            r = parity_compare(ifelse.(fc.amsk, f.vals, NaN), ifelse.(fc.amsk, e, NaN);
                               name = case)
            @test r.ok && iszero(r.maxabs)
            @test f.omsk == .!isnan.(fc.x)
        end
    end

    @testset "Views" begin
        # Small panel, written raw. The oracle selected observations 11:50 and assets
        # [2, 3, 4, 7, 9] by position; Fields holds every field of the view in panel order (the
        # categoricals as zero-based codes), and Masks holds its active then estimation mask.
        rows = 11:50
        cols = [2, 3, 4, 7, 9]
        v = PO.port_opt_view(pnl, rows, cols)
        vr = PO.port_opt_view(fx.rd, rows, cols)
        of = parity_load("port_opt_view", "SmallRaw", "Fields")
        ms = parity_load("port_opt_view", "SmallRaw", "Masks") .== 1
        va = ms[:, 1:5]
        @test v.amsk == va
        @test v.emsk == ms[:, 6:10]
        @test vr.pnl.amsk == va
        @test isequal(vr.X, fx.rd.X[rows, cols])
        for (k, f) in enumerate(v.pf)
            o = of[:, ((k - 1) * 5 + 1):(k * 5)]
            if f isa NumericPanelField
                @test all((f.vals[va] .== o[va]) .| isnan.(o[va]))
            else
                @test f.codes[va] .- 1 == o[va]
            end
        end
        # A view equals a panel built from the selected rows and columns.
        sub = asset_panel([NumericPanelInput(; name = n,
                                             vals = panel_field(pnl, n).vals[rows, cols])
                           for n in num]; amsk = A[rows, cols], emsk = pnl.emsk[rows, cols])
        @test all(panel_field(sub, n).vals == panel_field(v, n).vals for n in num)
        @test PO.feature_row_indices(pnl, [3, 7, 20], collect(1:80)) == [3, 7, 20]
        @test PO.asset_panel_view(nothing, rows, cols, nothing) === nothing
    end

    @testset "The carriers keep the panel" begin
        # No oracle counterpart: the oracle has no prices carrier. Checked against the
        # definition: the conversion drops the first price row of every Panel Field.
        ic = parity_ingest_case()
        ppnl = asset_panel([NumericPanelInput(; name = "shares", vals = ic.P .* 1e6,
                                              alg = ForwardPanelFill()),
                            CategoricalPanelInput(; name = "sector",
                                                  vals = repeat(["A" "B" "A" "C" "B"], 15))])
        pr = price_ingestion(PriceIngestion(), ic.X; pnl = ppnl,
                             B = TimeArray(timestamp(ic.X), ic.P[:, 1], ["bm"]))
        rd = prices_to_returns(pr)
        sh = panel_field(rd.pnl, "shares")
        @test sh.vals == panel_field(ppnl, "shares").vals[2:end, :]
        @test sh.omsk == panel_field(ppnl, "shares").omsk[2:end, :]
        @test panel_field(rd.pnl, "sector").codes ==
              panel_field(ppnl, "sector").codes[2:end, :]
        @test rd.pnl.amsk == prices_to_returns(ic.X).pnl.amsk
        rp = returns_result_picker(rd, true)
        @test rp.pnl === rd.pnl
        @test isequal(rp.X, rd.X .- rd.B)
        rv = prices_to_returns(PO.port_opt_view(pr, timestamp(ic.X)[5:12], :))
        @test panel_field(rv.pnl, "shares").vals == sh.vals[5:11, :]
    end

    @testset "Ingestion" begin
        # The price table of `parity_ingest_case`. The oracle converted it with no fill and no
        # inception cut (IngestNoFill), with a forward fill and no cut (IngestFillKeep), and with
        # its defaults, a forward fill and an inception cut (IngestDefault, whose Rows are the
        # zero-based price rows it kept). ActiveAligned is the active mask its panel derives from
        # the no-fill returns, from an all-active mask.
        ic = parity_ingest_case()
        r0 = prices_to_returns(ic.X)
        live = r0.pnl.amsk
        # The default conversion keeps every gap: a run of k gapped prices is k + 1 NaN returns.
        # rtol = 1e-12; measured maxrel = 1.1e-13, maxabs = 1.0e-16 on the four returns cases,
        # the oracle's rounding of its quotient (#1414).
        on = parity_load("prices_to_returns", "IngestNoFill", "X")
        @test parity_compare(r0.X, on; rtol = 1e-12, name = "no fill").ok
        # PriceGapFill carries a price inside the Listing Span and not past it. On the active
        # cells it is the oracle's forward fill; after the delisting the oracle carries the last
        # price, a return of zero, and the Span Rule makes the asset inactive (#955).
        pr = price_ingestion(PriceIngestion(), ic.X)
        r2 = prices_to_returns(PO.apply_preprocessing(PO.fit_preprocessing(PriceGapFill(),
                                                                           pr), pr))
        ok = parity_load("prices_to_returns", "IngestFillKeep", "X")
        @test parity_compare(ifelse.(live, r2.X, NaN), ifelse.(live, ok, NaN); rtol = 1e-12,
                             name = "fill").ok
        @test all(iszero, ok[11:14, 4]) &&
              !any(live[11:14, 4]) &&
              all(isnan, r2.X[11:14, 4])
        # The oracle's default cut drops the rows before the latest listing; the library keeps
        # them with the asset inactive (#955).
        od = parity_load("prices_to_returns", "IngestDefault", "X")
        rows = Int.(vec(parity_load("prices_to_returns", "IngestDefault", "Rows")))
        @test rows == 4:14
        @test parity_compare(ifelse.(live[rows, :], r2.X[rows, :], NaN),
                             ifelse.(live[rows, :], od, NaN); rtol = 1e-12, name = "cut").ok
        # CatchUpGapReturn books the catch-up of the oracle's fill route on the first priced
        # observation and keeps the Held Gap, where the fill route books zeros (#963).
        r1 = prices_to_returns(ic.X; gap_return_alg = CatchUpGapReturn())
        m = isfinite.(r1.X)
        @test parity_compare(r1.X[m], ok[m]; rtol = 1e-12, name = "catch-up").ok
        @test all(isnan, r1.X[6:7, 3]) && all(iszero, ok[6:7, 3]) && isnan(r1.X[9, 5])
        # The listing half of the Span Rule is the oracle's; the delisting half is an addition.
        oa = parity_load("universe_masks", "Ingest", "ActiveAligned") .== 1
        @test live[:, [1, 2, 3, 5]] == oa[:, [1, 2, 3, 5]]
        @test all(oa[:, 4]) && !all(live[:, 4])
        @test r0.pnl.emsk == live .& isfinite.(r0.X)
    end

    @testset "DataFrame export" begin
        # Small panel, written raw. The oracle exported its panel long (the active rows, the
        # fields, then the estimation mask) and wide (one column per field and asset, then the
        # active and estimation masks). Codes holds each categorical as a zero-based code of its
        # levels, -1 for the oracle's missing label.
        L = panel_dataframe(pnl; nx = fx.rd.nx, layout = :long)
        W = panel_dataframe(pnl; nx = fx.rd.nx, layout = :wide)
        ind = panel_field(pnl, "industry").levels
        ccy = panel_field(pnl, "currency").levels
        lv = parity_load("panel_dataframe", "SmallRawLong", "Values")
        lc = parity_load("panel_dataframe", "SmallRawLong", "Codes")
        @test nrow(L) == size(lv, 1) == count(A)
        for (k, n) in enumerate(num)
            @test all((L[!, n] .== lv[:, k]) .| isnan.(lv[:, k]))
            @test isnan.(lv[:, k]) == .!L[!, n * "::observed"]
        end
        @test L.emsk == (lv[:, end] .== 1)
        @test L.industry == ind[Int.(lc[:, 1]) .+ 1]
        @test L.currency == ccy[Int.(lc[:, 2]) .+ 1]
        wv = parity_load("panel_dataframe", "SmallRawWide", "Values")
        wc = parity_load("panel_dataframe", "SmallRawWide", "Codes")
        N = length(fx.rd.nx)
        for (k, n) in enumerate([num; "amsk"; "emsk"]), i in 1:N
            ours = W[!, "$(n)@a$(i)"]
            o = wv[:, (k - 1) * N + i]
            if n in ("amsk", "emsk")
                @test ours == (o .== 1)
            else
                @test all((ours .== o) .| isnan.(o))
            end
        end
        for (k, (n, levels)) in enumerate((("industry", ind), ("currency", ccy))), i in 1:N
            o = wc[:, (k - 1) * N + i]
            @test W[!, "$(n)@a$(i)"][A[:, i]] == levels[Int.(o[A[:, i]]) .+ 1]
        end
        # The observed-mask columns are an addition: the oracle's export has none.
        @test count(endswith("::observed"), names(L)) == length(num)
    end

    @testset "currency_excess_index against its definition" begin
        # No oracle counterpart. The index against its definition in BigFloat.
        rng = StableRNG(1366)
        T = 60
        ts = collect(Date(2020, 1, 1) .+ Day.(0:(T - 1)))
        S = exp.(cumsum(0.006 .* randn(rng, T, 3); dims = 1)) .* [1.1 0.009 0.7]
        rc = 0.0002 .* rand(rng, T, 3)
        rb = 0.0001 .* rand(rng, T)
        K = cumprod(1 .+ rc; dims = 1)
        Kb = cumprod(1 .+ rb)
        I = values(currency_excess_index(TimeArray(ts, S, ["EUR", "JPY", "GBP"]),
                                         TimeArray(ts, K[:, [3, 1, 2]],
                                                   ["GBP", "EUR", "JPY"]),
                                         TimeArray(ts, reshape(Kb, :, 1), ["USD"])))
        # Measured: maxrel = 2.0e-16.
        @test parity_compare(I, Float64.(big.(S) .* big.(K) ./ big.(Kb)); name = "level").ok
        # A return near zero from two rounded levels cancels, so it is measured against the
        # largest return. Measured: maxscaled = 2.0e-14, maxrel = 4.6e-12.
        Rb = big.(S[2:end, :]) ./ big.(S[1:(end - 1), :]) .* (1 .+ big.(rc[2:end, :])) ./
             (1 .+ big.(rb[2:end])) .- 1
        @test parity_compare(I[2:end, :] ./ I[1:(end - 1), :] .- 1, Float64.(Rb);
                             scale = :array, name = "return").ok
    end
end
