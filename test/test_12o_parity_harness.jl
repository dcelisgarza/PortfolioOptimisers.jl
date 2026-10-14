#=
The parity harness of map #1375 (#1376): the fixtures, the exchange by CSV, the comparison and the
storage that every parity ticket of the map reads from `parity_harness.jl`.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))

@testset "The parity harness (#1376)" begin
    PO = PortfolioOptimisers
    fx = parity_small_panel()
    rd = fx.rd
    at = fx.at

    @testset "The small panel holds every condition of the fixture" begin
        @test size(rd.X) == (80, 12)
        # A late listing, a delisting, and a delisting that lists again.
        @test !fx.amsk[at.late[2] - 1, at.late[1]] &&
              all(fx.amsk[at.late[2]:end, at.late[1]])
        @test all(fx.amsk[1:at.delist[2], at.delist[1]]) &&
              !any(fx.amsk[(at.delist[2] + 1):end, at.delist[1]])
        i, a, b = at.relist
        @test fx.amsk[a - 1, i] && !any(fx.amsk[a:(b - 1), i]) && all(fx.amsk[b:end, i])
        # A holiday gap inside an active stretch.
        @test fx.amsk[at.holiday...] && isnan(rd.X[at.holiday...])
        @test all(isfinite,
                  rd.X[fx.amsk .& .!(CartesianIndices(rd.X) .== CartesianIndex(at.holiday))])
        @test all(isnan, rd.X[.!fx.amsk])
        # The estimation mask is a strict subset of the active mask.
        @test all(fx.emsk .<= fx.amsk) && fx.emsk != fx.amsk
        @test all(fx.amsk[at.outside_estimation...]) &&
              !any(fx.emsk[at.outside_estimation...])
        # A zero and a negative denominator, a negative volume and a negative short interest.
        vals(name) = PO.panel_field(rd.pnl, name).vals
        @test iszero(vals("book_equity")[at.zero_denominator...])
        @test all(<(0), vals("book_equity")[at.negative_denominator...])
        @test vals("adj_volume")[at.negative_volume...] < 0
        @test vals("short_interest")[at.negative_short_interest...] < 0
        # Ties in one cross-section.
        t, js = at.ties
        @test allequal(vals("style1")[t, js])
        # The level "Utilities" is empty on the active universe after the delisting.
        f = PO.panel_field(rd.pnl, "industry")
        k = findfirst(==(at.empty_level[1]), f.levels)
        @test findall(==(k), f.codes[1, :]) == [at.empty_level[2]]
        @test !any(((f.codes .== k) .& fx.amsk)[(at.delist[2] + 1):end, :])
        # The Exogenous Series: three currencies and one macro series.
        @test rd.ne == ["EUR", "JPY", "USD", "MACRO"]
        @test rd.X[2, 1] ≈ fx.loc[2, 1] + fx.R[2, fx.code[1]] rtol = 1e-15
        lx = parity_large_panel()
        @test size(lx.rd.X) == (250, 40)
        @test lx.emsk != lx.amsk && lx.rd.ne == rd.ne
    end

    @testset "The exchange writes every file the oracle side reads" begin
        dir = parity_write(mktempdir(), rd)
        X = parity_read(joinpath(dir, "returns.csv"))
        @test isequal(X, rd.X)
        @test parity_read(joinpath(dir, "active_mask.csv")) == fx.amsk
        @test parity_read(joinpath(dir, "estimation_mask.csv")) == fx.emsk
        @test readlines(joinpath(dir, "assets.csv")) == rd.nx
        kinds = split.(readlines(joinpath(dir, "fields.csv")), ',')
        @test length(kinds) == length(rd.pnl.pf)
        @test count(k -> k[2] == "categorical", kinds) == 2
        # A numeric cell outside the active mask, or not observed, is NaN.
        V = parity_read(joinpath(dir, "field_adj_close.csv"))
        @test all(isnan, V[.!fx.amsk]) && isnan(V[at.holiday...])
        @test V[fx.amsk .& .!isnan.(V)] ==
              PO.panel_field(rd.pnl, "adj_close").vals[fx.amsk .& .!isnan.(V)]
        # Under `filled = true` the holiday holds the value of the fill policy, the last close.
        Vf = parity_read(joinpath(parity_write(mktempdir(), rd; filled = true),
                                  "field_adj_close.csv"))
        @test Vf[at.holiday...] == V[at.holiday[1] - 1, at.holiday[2]]
        @test all(isnan, Vf[.!fx.amsk])
        @test isequal(parity_read(joinpath(dir, "field_book_equity.csv")),
                      ifelse.(fx.amsk, PO.panel_field(rd.pnl, "book_equity").vals, NaN))
        # A categorical cell is a zero-based code, and -1 outside the active mask.
        C = parity_read(joinpath(dir, "field_industry.csv"))
        f = PO.panel_field(rd.pnl, "industry")
        @test C[fx.amsk] == f.codes[fx.amsk] .- 1
        @test all(==(-1), C[.!fx.amsk])
        @test readlines(joinpath(dir, "levels_industry.csv")) == f.levels
        E = CSV.read(joinpath(dir, "exogenous.csv"), DataFrame)
        @test names(E) == rd.ne && Matrix(E) == rd.E
    end

    @testset "The comparison checks the pattern and the relative difference" begin
        a = [1.0 NaN; -2.0 0.0]
        @test parity_compare(a, copy(a); name = "equal") ==
              (; ok = true, pattern = true, maxrel = 0.0, maxscaled = 0.0, maxabs = 0.0)
        r = parity_compare(a, a .* (1 + 1e-13); name = "1e-13")
        @test r.ok
        @test r.maxrel ≈ 1e-13 rtol = 1e-3
        r = parity_compare(a, a .* (1 + 1e-11); quiet = true)
        @test !r.ok && r.pattern
        @test parity_compare(a, a .* (1 + 1e-11); rtol = 1e-10).ok
        # A zero against a tiny value is a relative difference of one, which only `atol` passes.
        b = [1.0 NaN; -2.0 1e-18]
        @test !parity_compare(a, b; quiet = true).ok
        @test parity_compare(a, b; atol = 1e-16).ok
        # Under `scale = :array` a cell is measured against the largest entry, 2 here.
        r = parity_compare(a, b; scale = :array)
        @test r.ok && r.maxrel == 1 && r.maxscaled == 5e-19
        @test !parity_compare(a, [1.0 NaN; -2.0 3e-12]; scale = :array, quiet = true).ok
        # The non-finite cells must agree exactly.
        r = parity_compare(a, [1.0 2.0; -2.0 0.0]; quiet = true)
        @test !r.ok && !r.pattern
        @test !parity_compare([Inf], [-Inf]; quiet = true).ok &&
              parity_compare([Inf], [Inf]).ok
        @test parity_compare(a, a[:, 1]; quiet = true) ==
              (; ok = false, pattern = false, maxrel = Inf, maxscaled = Inf, maxabs = Inf)
        # A check that fails prints one line under its name, and `quiet = true` prints none,
        # so a negative check that passes adds no output.
        out = mktemp() do path, io
            redirect_stdout(io) do
                parity_compare([1.0], [2.0]; name = "loud")
                parity_compare([1.0], [2.0]; name = "hush", quiet = true)
                parity_compare([1.0], [1.0]; name = "pass")
                parity_compare(a, a[:, 1]; name = "shape", quiet = true)
                return nothing
            end
            close(io)
            return read(path, String)
        end
        @test startswith(out, "parity loud: maxrel = 0.5") && count('\n', out) == 1
    end

    @testset "The storage round-trips a stored output exactly" begin
        dir = mktempdir()
        p = parity_asset("CrossSectionalFactorPrior", "Small", "Sigma"; dir = dir)
        @test basename(p) == "Parity_CrossSectionalFactorPrior_Small_Sigma.csv.gz"
        @test dirname(parity_asset("U", "C", "O")) == joinpath(@__DIR__, "assets")
        A = randn(StableRNG(1376), 5, 3)
        A[2, 2] = NaN
        parity_store(p, A)
        @test isequal(parity_load("CrossSectionalFactorPrior", "Small", "Sigma"; dir = dir),
                      A)
        v = [0.1, 1e-300, -3.0]
        parity_store(parity_asset("U", "C", "Mu"; dir = dir), v)
        @test parity_load("U", "C", "Mu"; dir = dir) == reshape(v, :, 1)
        # A plain output of the oracle side carries no header.
        q = joinpath(dir, "out.csv")
        write(q, "1.5,NaN\n-2.0e-7,3.0\n")
        @test isequal(parity_read(q), [1.5 NaN; -2.0e-7 3.0])
    end
end

#=
Parity of the Cross-Sectional Factor Prior on a panel with gaps (#1377, map #1375). The prior
passes both universe masks to the idiosyncratic variance and to the refinement of the regression
weights. Without them an asset that delists and lists again kept its variance over the inactive
stretch, and the regime statistic read an active asset outside the estimation universe.
=#

@testset "The prior reads both masks on a panel with gaps, at parity (#1377)" begin
    # Each stored case is the fit of `parity_small_panel()` with a market factor and the two
    # passthrough styles, `minra = 5`, and the default of every other parameter. `MasksOnePass`
    # keeps the market-cap weights, and `MasksTwoPass` blends them half-way towards the inverse
    # idiosyncratic variance. The outputs are the idiosyncratic variance history, the
    # regression weights, `mu` and `sigma`.
    fx = parity_small_panel()
    at = fx.at
    mpass(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                 outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(), "style1" => mpass("style1"),
               "style2" => mpass("style2")]
    # The factor covariance clips its regime multiplier to (0.7, 1.6), the default on both
    # sides since #1383. The case states the clip because the two-pass case is the first
    # measured case where it binds: without it the factor covariance is 0.991 times the stored
    # one. This file measures the masks.
    pe = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                        ce = RegimeAdjustedExpWeightedCovariance(; centring = PreCentred(),
                                                                 debias = RawStatistic(),
                                                                 regime_lohi_mult = (0.7,
                                                                                     1.6)))
    load(c, o) = parity_load("CrossSectionalFactorPrior", c, o)
    i4 = at.relist[1]

    @testset "$(c)" for (c, wa) in (("MasksOnePass", MarketCapWeights()),
                                    ("MasksTwoPass", BlendedInverseVarianceWeights(; lambda = 0.5)))
        pr = prior(CrossSectionalFactorPrior(; lambda = 1, factors = factors, minra = 5,
                                             wa = wa, pe = pe, ve = PARITY_VE), fx.rd)
        # The variance resets when the asset delists, and warms up again after it lists, so
        # the non-finite cells of the relisted asset match. The regime statistic reads the
        # estimation universe alone. Measured maxrel 1.5e-15.
        @test parity_compare(pr.rr.vs, load(c, "IdioVariances"); name = "$(c) vs").ok
        # Measured maxrel 4.2e-16.
        @test parity_compare(pr.rr.rw, load(c, "RegressionWeights"); name = "$(c) rw").ok
        # The relisted asset is still in its warm-up at the latest observation, so neither side
        # states its idiosyncratic variance. Both sides state its mean and its covariances
        # with the other assets, which the model determines, and neither states its variance
        # (#1384).
        @test isnan(pr.rr.vs[end, i4]) && isnan(load(c, "IdioVariances")[end, i4])
        @test isfinite(pr.mu[i4]) && isnan(pr.sigma[i4, i4])
        # Measured maxrel 2.9e-14.
        @test parity_compare(pr.mu, vec(load(c, "Mu")); name = "$(c) mu").ok
        # A covariance compares against its largest entry, because its small off-diagonal
        # entries come from a cancellation (#1376). Measured maxscaled 7.9e-16, and maxrel
        # 1.6e-14 cell by cell. They were 3.3e-13 and 1.6e-11 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576).
        @test parity_compare(pr.sigma, load(c, "Sigma"); scale = :array,
                             name = "$(c) sigma").ok
    end
end

#=
Parity of map #1375 for the Asset Panel, its builder, its views, the ingestion and the DataFrame
export (#1378). Every `Parity_*` file this test reads is an output of the oracle, stored with the
harness of #1376, and each testset states how its case was made.

The two measured differences of this test were child issues of the map, and both are closed:

  - #1413: `ForwardPanelFill` and `BackwardPanelFill` carried a value across an inactive stretch of
    the active mask. The fill now stops at the edge of the stretch, as the oracle does, and the
    fill case is at parity on every active cell.
  - #1414: a simple return is `(p_t - p_{t-1}) / p_{t-1}`, which is correctly rounded on all 58
    finite returns of the ingestion case, where the oracle's `p_t / p_{t-1} - 1` is exact on 2.
    The returns comparisons keep `rtol = 1e-12`, and the difference left is the oracle's own
    rounding.
=#

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
        cases = (("ForwardNone", ForwardPanelFill(), (8, 3)),
                 ("ForwardLim2", ForwardPanelFill(; lim = 2), (8, 3)),
                 ("BackwardNone", BackwardPanelFill(), (4, 3)),
                 ("BackwardLim2", BackwardPanelFill(; lim = 2), (4, 3)))
        for (case, alg, edge) in cases
            o = parity_load("asset_panel", "FillEdges", case)
            f = parity_fill_field(fc, alg)
            e = blank_to_val(o, fc.amsk, 0.0)
            # #1413: the blank next to the inactive stretch 5:7 of asset 3 has nothing to carry
            # on its own side of the stretch, so both sides leave it to the fill value.
            @test isnan(o[edge...])
            @test f.vals[edge...] == 0.0
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
        rp = PO.prior_step_returns(rd, true)
        @test rp.pnl === rd.pnl
        @test isequal(rp.X, rd.X .- rd.B)
        @test isequal(benchmark_relative_returns(rd, ExcessSource()), rd.X .- rd.B)
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

#=
Parity of the cross-sectional diagnostics and the factor model summary on the block of a real fit
(#1387, map #1375).

Every stored case under `test/assets/Parity_CrossSectionalFactorModel_Diag<Panel><Case>_<Output>.csv.gz`
is the oracle's diagnostics of its own fit of a configuration of `parity_grid.jl`, on the exchange
`parity_write(dir, grid_fixture(fx); filled = true)` writes. The fit of each configuration is at
parity (#1385), so each case measures the diagnostics of two blocks that agree. The panels list
late, delist, list again, skip a holiday and keep an asset outside the estimation universe, and the
cases carry an exposure lag of two, a constrained family with a level that empties, two constrained
families, the Currency Factors and an observed factor.

The oracle ranks a tie in the order of an unstable sort, which no rule states. Its rank is replaced
by a stable one, which `ties = :ordinal` reproduces, and a case whose name ends in `Average` replaces
it by the midrank, which is the default `ties = :average`.

A group of outputs is stored side by side in one file:

  - `Regression`: the t-statistics, the variance inflation factors, the condition number and the
    four scores `r2`, `adjusted_r2`, `aic`, `bic`. `TRate`: the exceedance rate at the thresholds
    `2` and `1.5`, for the cases without a constrained family. In a family case a singular row
    changes the rate (#1421), and the factor model summary below checks that difference.
    `Gram`: the Gram of `DiagSmallBase`, one row per observation.
  - `Correlation`, `Stability`, `Dispersion`: the benchmark, identity and regression weightings.
    `Stability5`: the stability at a step of five.
  - `IC1`, `IC3`: the rank and the linear coefficient at the horizons one and three. `ICSummary`:
    their mean, deviation, ratio and hit rate, in that order. `ICRed`: the same on the reduced axis.
  - `Summary`: the nine columns of the summary at `ppy = 252`, and again at `threshold = 1.5` and
    `step = 5`.
  - `Idio`: the calibration, the tail rates at three and two, the excess kurtosis and the skewness,
    with each return divided by the volatility of its own observation. `IdioDependence`: the
    information coefficient and the residual dependence of the volatility. `AheadIdio`: the oracle's
    own kernels on each return divided by the volatility of the previous observation.
=#
include(joinpath(@__DIR__, "parity_grid.jl"))

@testset "The cross-sectional diagnostics on a fitted block, at parity (#1387)" begin
    PO = PortfolioOptimisers
    U = "CrossSectionalFactorModel"
    load(c, o) = parity_load(U, c, o)
    loadv(c, o) = vec(load(c, o))
    pc(a, b, name; rtol = 1e-12, scale = :cell) = parity_compare(a, b; rtol = rtol,
                                                                 scale = scale, name = name).ok
    fxs = parity_small_panel()
    fxl = parity_large_panel()
    rds = grid_fixture(fxs)
    rdl = grid_fixture(fxl)
    blocks = Dict{String, Any}()
    function block(fix, c)
        return get!(blocks, "$(fix)$(c)") do
            rd = fix == "Small" ? rds : rdl
            return prior(grid_prior(c, rd), rd).rr
        end
    end
    fam = ("FamOne", "FamTwo")
    cases = [("Small", "Base"), ("Small", "Lag2"), ("Small", "FamOne"), ("Small", "FamTwo"),
             ("Small", "Currency"), ("Small", "Macro"), ("Large", "Base"),
             ("Large", "FamOne")]
    weightings = (BenchmarkWeightMetric(), IdentityMetric(), RegressionWeightMetric())

    @testset "The regression group, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        R = load(n, "Regression")
        t = cs_regression_t_stats(csfm).X
        K = size(t, 2)
        kap = exposure_condition_number(csfm)
        okap = R[:, 2K + 1]
        # The constrained family's level that empties makes rows whose design is singular,
        # and both sides flag them. Such a row identifies only the coefficients whose unit
        # vector lies in the row space of its Gram. The oracle's t-statistic of another
        # coefficient is a ratio of round-off (7.87 for a level with no member), and it
        # divides every residual variance by `n - K`. Ours is `NaN` for an unidentified
        # coefficient, its variance inflation factor is `Inf` (`R_k^2 = 1`) or `NaN` (a
        # zero column), and the unbiased variance divides by `n - rank` (#1421), Better.
        idf = (kap .< 1e12) .& (okap .< 1e12)
        @test all(>=(1e12), kap[.!idf]) && all(>=(1e12), okap[.!idf])
        @test c in fam || all(idf)
        # A t-statistic near zero carries the relative round-off of the fit, so the check
        # reads the array scale. Measured maxrel 1.6e-12 on the large panel on one host, 2.1e-12
        # on the CI host, and maxscaled 7.4e-15 on both.
        @test pc(t[idf, :], R[idf, 1:K], "$(n) t"; scale = :array)
        vif = exposure_vif(csfm).X
        @test pc(vif[idf, :], R[idf, (K + 1):(2K)], "$(n) vif")
        @test pc(kap[idf], okap[idf], "$(n) kappa")
        # An identified coefficient has one variance whatever basis of the column space the
        # fit reads, so on a singular row its t-statistic equals that of the fit on a
        # pivoted basis of the weighted design, which has full rank. Every coefficient
        # outside the basis is unidentified. Measured maxrel 3.3e-15 over 233 coefficients.
        d = PO.cs_regression_data(csfm)
        mask, u = PO.cs_diagnostic_mask_weights(d.B, d.eps, d.w)
        for r in findall(.!idf)
            A = sqrt.(u[r, mask[r, :]]) .* d.B[r, mask[r, :], :]
            kb = sort(qr(A, ColumnNorm()).p[1:rank(A)])
            wr = isnothing(d.w) ? nothing : d.w[r:r, :]
            tb = vec(cs_regression_t_stats(d.B[r:r, :, kb], d.f[r:r, kb], d.eps[r:r, :],
                                           wr))
            ib = .!isnan.(t[r, kb])
            @test any(ib) && pc(t[r, kb][ib], tb[ib], "$(n) basis t $(r)"; rtol = 1e-13)
            nb = setdiff(1:K, kb)
            @test all(isnan, t[r, nb]) && all(v -> isinf(v) || isnan(v), vif[r, nb])
            @test all(v -> isnan(v) || v >= 1, vif[r, :])
        end
        # `1 - rss / tss` of a small score loses digits. Measured maxrel 2.5e-12 at a score
        # near 0.02 under the exposure lag of two, and maxscaled 6.5e-16.
        @test pc(cs_regression_r2(csfm), R[:, 2K + 2], "$(n) r2"; rtol = 3e-12)
        # A score charges the rank of each row, which the oracle's column count `k = K`
        # exceeds on a singular row. So the default is at parity on the rows of full rank,
        # `k = K` is at parity on every row, and on a singular row the AIC is the oracle's
        # less `2 (K - rank)`. Measured maxrel 4.3e-14 on the adjusted score.
        rk = PO.cs_score_regressors(nothing, d.B,
                                    PO.cs_regression_score_parts(d.B, d.f, d.eps, d.w)[4])
        @test all(==(K), rk[idf]) && all(<(K), rk[.!idf])
        aic = cs_regression_aic(csfm)
        @test pc(aic[.!idf], R[.!idf, 2K + 4] .- 2 .* (K .- rk[.!idf]), "$(n) aic rank")
        for (j, verb) in ((3, cs_regression_adjusted_r2), (4, cs_regression_aic),
                          (5, cs_regression_bic))
            @test pc(verb(csfm)[idf], R[idf, 2K + j], "$(n) score $(j)")
            @test pc(verb(csfm; k = K), R[:, 2K + j], "$(n) score $(j) k")
        end
        if !(c in fam)
            tr = load(n, "TRate")
            @test pc(cs_regression_t_stat_exceedance_rate(csfm).X, tr[:, 1], "$(n) rate")
            @test pc(cs_regression_t_stat_exceedance_rate(csfm; threshold = 1.5).X,
                     tr[:, 2], "$(n) rate 1.5")
        end
    end

    @testset "The Gram of the regression group" begin
        # A cross-product of a centred exposure comes from a cancellation, so the Gram
        # compares against its largest entry. Measured maxscaled 4.1e-16, and maxrel 1.1e-10
        # cell by cell on the market-by-style entries.
        csfm = block("Small", "Base")
        d = PO.cs_regression_data(csfm)
        u = ifelse.(isfinite.(d.eps), d.w, 0.0)
        G = cs_gram(d.B, u)
        K = size(G, 2)
        @test pc(reshape(permutedims(G, (1, 3, 2)), size(G, 1), K * K),
                 load("DiagSmallBase", "Gram"), "gram"; scale = :array)
    end

    @testset "The exposure group, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        B = csfm.Ms
        K = size(B, 3)
        # The oracle's benchmark weight needs a finite return. Its observed-factor path reads
        # the returns net of the observed factor, `X_t - z_(t-1) m_t`, which the stored case
        # builds for it, and that return is `NaN` at the first active observation of a listing,
        # where `z_(t-1)` is absent. So the oracle's weight is zero on two cells where ours is
        # the capitalisation weight of an asset whose base-currency return is finite. The
        # stored `BenchmarkWeightChanges` holds those cells, and every benchmark-weighted
        # output of this case reads the oracle's weights through the verb's array form, which
        # measures the kernel alone.
        bw = csfm.bw
        if c == "Macro"
            ch = load(n, "BenchmarkWeightChanges")
            @test size(ch, 1) == 2 && all(iszero, ch[:, 3])
            bw = copy(bw)
            for r in eachrow(ch)
                bw[Int(r[1]), Int(r[2])] = r[3]
            end
        end
        C = load(n, "Correlation")
        S = load(n, "Stability")
        D = load(n, "Dispersion")
        for (j, (wn, wt)) in
            enumerate(zip(("bench", "identity", "regression"), weightings))
            u = j == 1 ? bw : PO.cs_diagnostic_weights(wt, csfm)
            cols = ((j - 1) * K + 1):(j * K)
            @test pc(exposure_correlation(B, u), C[cols, :], "$(n) correlation $(wn)")
            @test pc(exposure_stability(B, u), S[:, cols], "$(n) stability $(wn)")
            # A constant exposure has a dispersion of exactly zero: the mean of the entering
            # values is clamped to their range, so the deviations are zero. The oracle reads
            # the round-off of the mean, at most 4.5e-16. Every other entry is at parity.
            dp = exposure_dispersion(B, u)
            od = D[:, cols]
            z = dp .== 0
            @test all(<=(1e-15), abs.(od[z]))
            @test pc(dp[.!z], od[.!z], "$(n) dispersion $(wn)")
            if j == 1
                @test pc(exposure_stability(B, u; step = 5), load(n, "Stability5"),
                         "$(n) stability 5")
            end
        end
        # The verbs over the block read the block's own weights through the same kernels.
        u = PO.cs_diagnostic_weights(BenchmarkWeightMetric(), csfm)
        @test isequal(exposure_correlation(csfm).X, exposure_correlation(B, u))
        @test isequal(exposure_stability(csfm).X, exposure_stability(B, u))
        @test isequal(exposure_dispersion(csfm).X, exposure_dispersion(B, u))
    end

    @testset "The exposure information coefficient, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        K = size(csfm.Ms, 3)
        Q = load(n, "ICSummary")
        for (h, o) in ((1, load(n, "IC1")), (3, load(n, "IC3"))),
            (j, rank) in enumerate((true, false))

            ic = exposure_ic(csfm; horizon = h, rank = rank, ties = :ordinal).X
            # A linear coefficient near zero carries the round-off of the fit's returns.
            # Measured maxrel 1.4e-12 on the large panel, and maxscaled 7.4e-16.
            @test pc(ic, o[:, ((j - 1) * K + 1):(j * K)], "$(n) ic $(h) $(rank)";
                     rtol = 1.5e-12)
            s = exposure_ic_summary(csfm; horizon = h, rank = rank, ties = :ordinal)
            q = Q[:, (8 * (h == 3) + 4 * (j - 1) + 1):(8 * (h == 3) + 4 * j)]
            @test pc(s.mean_ic, q[:, 1], "$(n) mean ic $(h) $(rank)")
            @test pc(s.std_ic, q[:, 2], "$(n) std ic $(h) $(rank)")
            @test pc(s.ic_ir, q[:, 3], "$(n) ic ir $(h) $(rank)")
            # The oracle counts an observation whose coefficient is `NaN` as a miss, so its hit
            # rate is a share of every observation. Ours is a share of the observations that
            # have a coefficient (ADR 0149). The two agree on a factor with no `NaN`.
            P = size(ic, 1)
            @test q[:, 4] == [count(>(0), view(ic, :, k)) / P for k in 1:K]
            nn = [count(!isnan, view(ic, :, k)) for k in 1:K]
            @test isequal(s.hit_rate,
                          [nn[k] > 0 ? count(>(0), view(ic, :, k)) / nn[k] : NaN
                           for k in 1:K])
        end
        if c in fam
            # The rank coefficient on the reduced axis is bit-equal on the three panels. The
            # linear one carries the round-off of the fit's returns, as on the raw axis:
            # measured maxrel 1.1e-12 on the large panel, and maxscaled 2.3e-15.
            o = load(n, "ICRed")
            Kr = size(o, 2) ÷ 2
            @test pc(exposure_ic(csfm; ties = :ordinal, reduced = true).X, o[:, 1:Kr],
                     "$(n) reduced rank ic")
            @test pc(exposure_ic(csfm; rank = false, reduced = true).X, o[:, (Kr + 1):end],
                     "$(n) reduced linear ic"; rtol = 1.5e-12)
        end
    end

    @testset "The factor model summary, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        n = "Diag$(fix)$(c)"
        O = load(n, "Summary")
        nf = csfm.nf
        cols = (:ann_return, :ann_volatility, :sharpe, :autocorr, :mean_abs_t, :t_rate,
                :mean_vif, :coverage, :stability)
        keep = trues(length(nf), length(cols))
        if c in fam
            # The factor return of a level the basis drops is fixed by the constraint of its
            # family. The block carries it on the raw axis in `fr`, so the summary states the
            # four return statistics of that level too, at parity (#1422).
            red = PO.reduce_factor_names(csfm.fcb, nf)
            drop = [!(x in red) for x in nf]
            @test any(drop)
            @test all(isfinite, factor_model_summary(csfm).ann_return[drop])
            # A level that empties leaves rows where the design is singular. There a
            # coefficient the design does not identify has no t-statistic and an infinite
            # VIF, and the others divide by `n - rank`, where the oracle reads round-off and
            # `n - K` (#1421, see the regression group). So a singular row changes the
            # t-statistic of every factor, and the oracle's mean absolute t and exceedance
            # rate differ for each one, by 0.33 and 0.26 relative on the large panel. A factor
            # that a singular row does not identify has a mean VIF of `Inf`, where the oracle's
            # is finite. The three columns compare instead against the same statistics of the
            # oracle's t-statistics and VIFs on the rows of full rank and ours on the singular
            # rows, which the regression group pins to the fit on a pivoted basis. A factor
            # that no singular row identifies reads the oracle's rows of full rank alone
            # (1.7500 for the level of one member, where the round-off rows gave 1.7755).
            # Every row of `FamTwo` is singular, so there the check reads ours alone and tests
            # the statistics. Measured maxrel 8.2e-16 on the mean absolute t, 2.9e-16 on the
            # mean VIF, and an exact rate.
            t = cs_regression_t_stats(csfm).X
            vif = exposure_vif(csfm).X
            R = load(n, "Regression")
            K = size(t, 2)
            sing = .!((exposure_condition_number(csfm) .< 1e12) .& (R[:, 2K + 1] .< 1e12))
            @test any(sing) && (c == "FamTwo" || !all(sing))
            th = R[:, 1:K]
            th[sing, :] .= t[sing, :]
            vh = R[:, (K + 1):(2K)]
            vh[sing, :] .= vif[sing, :]
            pos = PO.factor_summary_positions(csfm)
            colmean(A) = PO.factor_summary_mapped([PO.factor_summary_column_mean(A, j)
                                                   for j in 1:K], pos)
            mabs = colmean(abs.(th))
            mvif = colmean(vh)
            @test any(isinf, mvif)
            for (thr, step) in ((2, 1), (1.5, 5))
                fs = factor_model_summary(csfm; ppy = 252, threshold = thr, step = step)
                rate = PO.factor_summary_mapped(cs_regression_t_stat_exceedance_rate(th;
                                                                                     threshold = thr),
                                                pos)
                @test pc(fs.mean_abs_t, mabs, "$(n) mean abs t $(thr)")
                @test pc(fs.t_rate, rate, "$(n) t rate $(thr)")
                @test pc(fs.mean_vif, mvif, "$(n) mean vif $(thr)")
            end
            keep[:, 5:7] .= false
        end
        if c == "Macro"
            # The stability reads the benchmark weights of the fit, which differ on two cells
            # (see the exposure group).
            keep[:, 9] .= false
        end
        for (off, fs) in ((0, factor_model_summary(csfm; ppy = 252)),
                          (9, factor_model_summary(csfm; ppy = 252, threshold = 1.5, step = 5)))
            for (j, cn) in enumerate(cols)
                k = keep[:, j]
                @test pc(getfield(fs, cn)[k], O[k, off + j], "$(n) $(cn) $(off)")
            end
        end
    end

    @testset "The idiosyncratic group, $(fix) $(c)" for (fix, c) in cases
        csfm = block(fix, c)
        if c in fam
            # Better (#1423). Asset 3 is the only member of its industry until it delists,
            # so the design gives it a direction of its own and the fit reproduces its
            # return. Its residual is zero by construction, and so is its variance under the
            # design: its standardised return is 0 / 0. The oracle divides the round-off of
            # one by the round-off of the other, near 1e-18 / 1e-17, and enters an order-one
            # number into every cross-section, so no stored case compares here. The fit
            # marks the pair, writes an exact zero, and the group leaves the asset out, so
            # each verb equals the same verb on the panel without asset 3.
            h = csfm.csr.h1
            act = .!isnan.(csfm.csr.eps)
            @test findall(vec(any(h; dims = 1))) == [3]
            @test h[:, 3] == act[:, 3]
            @test all(iszero, csfm.csr.eps[h])
            @test all(x -> iszero(x) || isnan(x), csfm.vs[:, 3])
            @test all(isnan,
                      standardised_idio_returns(PO.idio_diagnostic_data(csfm)...)[:, 3])
            k = setdiff(axes(csfm.csr.eps, 2), 3)
            ek = csfm.csr.eps[:, k]
            vk = csfm.vs[:, k]
            for ahead in (true, false)
                @test isequal(idio_calibration(csfm; ahead = ahead),
                              idio_calibration(ek, vk; ahead = ahead))
                @test isequal(idio_tail_rate(csfm; ahead = ahead),
                              idio_tail_rate(ek, vk; ahead = ahead))
                @test isequal(idio_kurtosis(csfm; ahead = ahead),
                              idio_kurtosis(ek, vk; ahead = ahead))
                @test isequal(idio_skewness(csfm; ahead = ahead),
                              idio_skewness(ek, vk; ahead = ahead))
                @test isequal(idio_calibration_summary(csfm; ahead = ahead),
                              idio_calibration_summary(ek, vk; ahead = ahead))
            end
            @test isequal(idio_vol_ic(csfm), idio_vol_ic(ek, vk))
            continue
        end
        n = "Diag$(fix)$(c)"
        I = load(n, "Idio")
        @test pc(idio_calibration(csfm; ahead = false), I[:, 1], "$(n) calibration")
        @test pc(idio_tail_rate(csfm; ahead = false), I[:, 2], "$(n) tail 3")
        @test pc(idio_tail_rate(csfm; threshold = 2, ahead = false), I[:, 3], "$(n) tail 2")
        # The excess kurtosis subtracts three from a ratio near three, so an entry near zero
        # comes from a cancellation and compares against the largest entry. Measured maxrel
        # 2.4e-11 on the large panel, and maxscaled 1.4e-15.
        @test pc(idio_kurtosis(csfm; ahead = false), I[:, 4], "$(n) kurtosis";
                 scale = :array)
        @test pc(idio_skewness(csfm; ahead = false), I[:, 5], "$(n) skewness")
        @test pc(collect(values(idio_calibration_summary(csfm; ahead = false))),
                 loadv(n, "IdioSummary"), "$(n) summary")
        Dp = load(n, "IdioDependence")
        @test pc(idio_vol_ic(csfm; ties = :ordinal), Dp[:, 1], "$(n) vol ic")
        @test pc(idio_vol_residual_dependence(csfm; ties = :ordinal), Dp[:, 2],
                 "$(n) residual dependence")
        # Better: the default divides each return by the volatility of the previous row, whose
        # estimate has not read it. The oracle's own kernels on that standardisation give the
        # stored `Ahead` outputs.
        A = load(n, "AheadIdio")
        @test pc(idio_calibration(csfm), A[:, 1], "$(n) ahead calibration")
        @test pc(idio_tail_rate(csfm), A[:, 2], "$(n) ahead tail")
        # An excess kurtosis near zero is a difference of near-equal terms: measured cell
        # maxrel 2.4e-11 and maxscaled 2.6e-15, so the check is `:array`.
        @test pc(idio_kurtosis(csfm), A[:, 3], "$(n) ahead kurtosis"; scale = :array)
        @test pc(idio_skewness(csfm), A[:, 4], "$(n) ahead skewness")
        @test pc(collect(values(idio_calibration_summary(csfm))),
                 loadv(n, "AheadIdioSummary"), "$(n) ahead summary")
    end

    @testset "The same-row standardisation understates the tails" begin
        # The panels draw Gaussian residuals. The variance of a row has read the return of
        # that row, so the same-row tail rate sits below the Gaussian reference of 0.27 %, and
        # the one-step-ahead rate sits above it, as the error of an estimated variance makes
        # it. Measured on the large panel: 0.12 % against 0.55 %, and a mean deviation of 0.988
        # against 1.009.
        csfm = block("Large", "Base")
        sr = idio_calibration_summary(csfm; ahead = false)
        sa = idio_calibration_summary(csfm)
        # The Gaussian reference of a threshold of three, `2Φ(-3)`.
        @test sr.mean_tail_rate < 0.002699796063260207 < sa.mean_tail_rate
        @test sr.mean_cs_std < 1 < sa.mean_cs_std
    end

    @testset "The midrank of a tie" begin
        # The default `ties = :average` against the oracle's kernels with the midrank. The
        # constant market exposure has constant midranks, so it has no coefficient at any
        # observation, on either side.
        csfm = block("Small", "Base")
        n = "DiagSmallBaseAverage"
        Q = load(n, "ICSummary")
        for (h, o) in ((1, load(n, "IC1")), (3, load(n, "IC3")))
            ic = exposure_ic(csfm; horizon = h).X
            @test pc(ic, o, "average ic $(h)")
            @test all(isnan, ic[:, 1])
            s = exposure_ic_summary(csfm; horizon = h)
            q = Q[:, (4 * (h == 3) + 1):(4 * (h == 3) + 4)]
            @test pc(s.mean_ic, q[:, 1], "average mean ic $(h)")
            @test pc(s.std_ic, q[:, 2], "average std ic $(h)")
            @test pc(s.ic_ir, q[:, 3], "average ic ir $(h)")
            # The market factor has no coefficient, so it has no hit rate. The oracle counts
            # each `NaN` as a miss and reports zero.
            @test isnan(s.hit_rate[1]) && q[1, 4] == 0
            @test s.hit_rate[2:end] == q[2:end, 4]
        end
        Dp = load(n, "IdioDependence")
        @test pc(idio_vol_ic(csfm), Dp[:, 1], "average vol ic")
        @test pc(idio_vol_residual_dependence(csfm), Dp[:, 2],
                 "average residual dependence")
    end
end

#=
Parity of map #1375 for the point-in-time ratio Descriptors and the lag Descriptors (#1379). Every
`Parity_ratio_lag_descriptors_*` file this test reads is an output of the oracle, stored with the
harness of #1376, and each testset states how its case was made.

The measure found one defect, which this ticket fixes: a quotient of two finite values that
overflows gave an infinity, and every cross-sectional transform refuses an infinity, so one cell
cost the whole fit. `positive_divide` now gives `NaN` there, as the oracle does.

A data error refuses, and an undefined ratio is `NaN` (ADR 0108). A field that is positive by
construction (a price, a market capitalisation, a share count, a total of assets) refuses a value
at or below zero through the `gt0` guard of its named constructor. A denominator that valid data
can make zero or negative (a book equity, an enterprise value) gives `NaN`. Three differences
from the oracle are deliberate, and ADR 0108 records them:

  - the ratios over `total_assets` and `MarketLeverage` refuse a non-positive total of assets or
    market capitalisation, where the oracle gives `NaN`; `GrossMargin` refuses a negative sales
    figure;
  - the archetypes `ChangeToScale` and `ChangeInIntensity` give `NaN` at a non-positive scale by
    default, where the oracle refuses; their named constructors refuse, as the oracle does;
  - the non-negative guard reads the active observed cells alone, where the oracle reads every
    finite cell, a cell outside the active mask included.

Every refusal keeps the oracle's output one keyword away: `parity_unguarded` rebuilds each named
Descriptor with its `gt0` fields moved to `pos` and its sign guards off, and that estimator equals
the oracle on every case.
=#

# The small fixture of the harness, with the Panel Fields that the named Descriptors read by
# default and the fixture does not hold. `kind` selects the case:
#
#   - `:clean`: every guarded field is valid. The negative short interest of the fixture is made
#     positive, so `ShortInterest` has a value to compare.
#   - `:negative`: a negative value, at one active observed cell, in each field that a
#     Descriptor refuses to read below zero.
#   - `:nonpositive`: a zero or a negative value in each denominator that the oracle refuses at
#     or below zero.
function parity_ratio_case(kind::Symbol = :clean; fx = parity_small_panel())
    pnl = fx.rd.pnl
    amsk = pnl.amsk
    T, N = size(amsk)
    rng = StableRNG(1379)
    raw(name) = parity_field_rows(panel_field(pnl, name),
                                  amsk .& something(panel_field(pnl, name).omsk, amsk))
    mcap = raw("market_cap")
    shares = raw("adj_shares_outstanding")
    assets = raw("total_assets")
    income = raw("net_income_ttm")
    sales = raw("sales_ttm")
    debt = 0.3 .* assets .* exp.(0.3 .* randn(rng, 1, N) .+ 0.05 .* randn(rng, T, N))
    ocf = income .+ 0.02 .* assets .* randn(rng, T, N)
    cogs = sales .* (0.7 .+ 0.15 .* randn(rng, T, N))
    ebitda = 0.12 .* assets .* (1 .+ 0.4 .* randn(rng, T, N))
    ev = mcap .+ debt .- 0.2 .* assets .* exp.(0.5 .* randn(rng, T, N))
    eps = income ./ shares .* (1 .+ 0.1 .* randn(rng, T, N))
    dps = max.(0.4 .* eps, 0.0)
    dividends = dps .* shares
    buybacks = 0.01 .* mcap .* randn(rng, T, N)
    eps_std = 0.2 .* abs.(eps) .* exp.(0.3 .* randn(rng, T, N))
    capex = 0.05 .* assets .* exp.(0.2 .* randn(rng, T, N))
    # An enterprise value at or below zero: the ratio over it is NaN on both sides.
    ev[50, 11] = -abs(ev[50, 11])
    ev[51, 12] = 0.0
    # A zero dividend is valid, and the ratio over the price is zero.
    dividends[30:32, 1] .= 0.0
    # Blanks in an active stretch of the new fields.
    debt[45, 5] = NaN
    capex[30, 1] = NaN
    si = raw("short_interest")
    nsi = fx.at.negative_short_interest
    si[nsi...] = abs(si[nsi...])
    close = raw("adj_close")
    if kind === :negative
        dividends[25, 2] = -1.0
        dps[21, 3] = -1.0e-3
        eps_std[22, 4] = -1.0e-3
        capex[23, 5] = -1.0
        assets[24, 6] = -assets[24, 6]
        sales[25, 7] = -sales[25, 7]
        shares[26, 8] = -shares[26, 8]
        si[nsi...] = -si[nsi...]
    elseif kind === :nonpositive
        mcap[60, 1] = -mcap[60, 1]
        mcap[61, 2] = 0.0
        close[62, 5] = 0.0
        shares[63, 6] = 0.0
        assets[64, 7] = 0.0
        sales[65, 9] = 0.0
    end
    numeric = ["market_cap" => mcap, "adj_close" => close,
               "adj_shares_outstanding" => shares, "short_interest" => si,
               "book_equity" => raw("book_equity"), "total_assets" => assets,
               "net_income_ttm" => income, "sales_ttm" => sales, "style1" => raw("style1"),
               "total_debt" => debt, "operating_cash_flow_ttm" => ocf,
               "cost_of_revenue_ttm" => cogs, "ebitda_ttm" => ebitda,
               "enterprise_value" => ev, "eps_ntm" => eps, "dps_ntm" => dps,
               "dividends_ttm" => dividends, "net_buybacks_ttm" => buybacks,
               "eps_ntm_std" => eps_std, "capex_ttm" => capex]
    for (_, A) in numeric
        A[.!amsk] .= NaN
    end
    pf = [NumericPanelInput(; name = k, vals = A, alg = ForwardPanelFill(; val = 0.0))
          for (k, A) in numeric]
    rd = ReturnsResult(; nx = fx.rd.nx, X = fx.rd.X,
                       pnl = asset_panel(pf; amsk = amsk, emsk = pnl.emsk))
    return (; rd, amsk)
end

# Each Descriptor of the ticket, under the name of its column in the stored files. A named
# constructor reads its default Panel Fields, which are the field names of the oracle. The lag
# Descriptors run at lag 1 and at lag 30, which reaches back across the inactive stretch of
# asset 4.
function parity_ratio_descriptors()
    ds = Pair{String, Any}["BookToPrice" => BookToPrice(),
                           "CashFlowToPrice" => CashFlowToPrice(),
                           "SalesToPrice" => SalesToPrice(),
                           "EarningsToPrice" => EarningsToPrice(),
                           "ForwardEarningsToPrice" => ForwardEarningsToPrice(),
                           "EbitdaToEnterpriseValue" => EbitdaToEnterpriseValue(),
                           "DividendToPrice" => DividendToPrice(),
                           "ForwardDividendToPrice" => ForwardDividendToPrice(),
                           "ShareholderYield" => ShareholderYield(),
                           "BookLeverage" => BookLeverage(),
                           "MarketLeverage" => MarketLeverage(),
                           "DebtToAssets" => DebtToAssets(),
                           "GrossProfitability" => GrossProfitability(),
                           "GrossMargin" => GrossMargin(),
                           "ReturnOnAssets" => ReturnOnAssets(),
                           "ReturnOnEquity" => ReturnOnEquity(),
                           "AssetTurnover" => AssetTurnover(),
                           "CashFlowToAssets" => CashFlowToAssets(),
                           "SalesToEnterpriseValue" => SalesToEnterpriseValue(),
                           "AccrualsCashFlow" => AccrualsCashFlow(),
                           "AnalystDispersionToPrice" => AnalystDispersionToPrice(),
                           "LogMarketCap" => LogMarketCap(),
                           "ShortInterest" => ShortInterest(),
                           "PassthroughStyle1" => Passthrough(; field = "style1"),
                           "PassthroughBookEquity" => Passthrough(; field = "book_equity")]
    for L in (1, 30)
        append!(ds,
                ["AssetsGrowthRateL$L" => AssetsGrowthRate(; lag = L),
                 "SalesGrowthRateL$L" => SalesGrowthRate(; lag = L),
                 "IssuanceGrowthRateL$L" => IssuanceGrowthRate(; lag = L),
                 "EarningsChangeToPriceL$L" => EarningsChangeToPrice(; lag = L),
                 "CapexToAssetsChangeInIntensityL$L" =>
                     CapexToAssetsChangeInIntensity(; lag = L),
                 "GrowthRateCapexL$L" => GrowthRate(; field = "capex_ttm", lag = L),
                 "ChangeToScaleSalesAssetsL$L" =>
                     ChangeToScale(; field = "sales_ttm", scale = "total_assets", lag = L),
                 "ChangeInIntensityDebtMcapL$L" =>
                     ChangeInIntensity(; field = "total_debt", scale = "market_cap",
                                       lag = L)])
    end
    return ds
end

# The route with no refusal: a ratio with its `gt0` fields moved to `pos` and no `nonneg`
# guard, a logarithm and a scale with `gt0 = false`. It gives NaN in every cell that a guard
# would refuse, which is the oracle's rule for a denominator it does not refuse.
function parity_unguarded(de::PanelFieldRatio)
    pos = unique([something(de.pos, String[]); something(de.gt0, String[])])
    return PanelFieldRatio(; num = de.num, den = de.den, pos = isempty(pos) ? nothing : pos)
end
parity_unguarded(de::PanelFieldLog) = PanelFieldLog(; field = de.field)
function parity_unguarded(de::ChangeToScale)
    return ChangeToScale(; field = de.field, scale = de.scale, lag = de.lag)
end
function parity_unguarded(de::ChangeInIntensity)
    return ChangeInIntensity(; field = de.field, scale = de.scale, lag = de.lag)
end
parity_unguarded(de) = de

# The stored oracle of a case: the clean outputs, with the cells of `Changes` replaced. A row of
# `Changes` is the index of the Descriptor in `parity_ratio_descriptors()`, the linear index of the
# cell, and the oracle's value of the cell in that case.
function parity_ratio_oracle(clean::AbstractMatrix, changes::AbstractMatrix, k::Integer)
    o = clean[:, k]
    for r in eachrow(changes)
        if Int(r[1]) == k
            o[Int(r[2])] = r[3]
        end
    end
    return o
end

@testset "Parity: the point-in-time ratio Descriptors and the lag Descriptors (#1379)" begin
    PO = PortfolioOptimisers
    ds = parity_ratio_descriptors()
    names = first.(ds)
    # The clean case: `parity_ratio_case(:clean)`, every Descriptor of `ds` with its default
    # Panel Fields, one column per Descriptor, `vec` of the 80 × 12 output. Measured
    # `maxrel = 0.0` on all 41: bit-equal, and the NaN pattern is equal.
    clean = parity_load("ratio_lag_descriptors", "SmallClean", "Descriptors")
    @testset "Clean panel" begin
        c = parity_ratio_case(:clean)
        @test size(clean) == (80 * 12, length(ds))
        for (k, (name, de)) in enumerate(ds)
            @test parity_compare(vec(descriptor(de, c.rd)), clean[:, k]; name = name).ok
        end
    end
    # The non-positive case: `parity_ratio_case(:nonpositive)`. The oracle refuses 19 Descriptors,
    # and the stored output of each is the oracle run on the panel with the non-positive cells of
    # its refused field blanked to NaN, which is the output under the oracle's own NaN rule. The
    # 22 named Descriptors below refuse the panel through `gt0`. The four generic lag archetypes
    # have `gt0 = false` and give NaN. Every Descriptor with no refusal, `parity_unguarded`,
    # equals the stored oracle. Measured `maxrel = 0.0`.
    @testset "A non-positive price, market cap, share count or total assets refuses" begin
        refused = ["BookToPrice", "CashFlowToPrice", "SalesToPrice", "EarningsToPrice",
                   "ForwardEarningsToPrice", "DividendToPrice", "ForwardDividendToPrice",
                   "ShareholderYield", "MarketLeverage", "DebtToAssets",
                   "GrossProfitability", "ReturnOnAssets", "AssetTurnover",
                   "CashFlowToAssets", "AccrualsCashFlow", "AnalystDispersionToPrice",
                   "LogMarketCap", "ShortInterest", "EarningsChangeToPriceL1",
                   "CapexToAssetsChangeInIntensityL1", "EarningsChangeToPriceL30",
                   "CapexToAssetsChangeInIntensityL30"]
        @test refused ⊆ names
        c = parity_ratio_case(:nonpositive)
        changes = parity_load("ratio_lag_descriptors", "SmallNonpositive", "Changes")
        for (k, (name, de)) in enumerate(ds)
            oracle = parity_ratio_oracle(clean, changes, k)
            if name in refused
                @test_throws DomainError descriptor(de, c.rd)
            else
                @test parity_compare(vec(descriptor(de, c.rd)), oracle; name = name).ok
            end
            @test parity_compare(vec(descriptor(parity_unguarded(de), c.rd)), oracle;
                                 name = "unguarded " * name).ok
        end
        # Without the guard, the cell of a non-positive market capitalisation is NaN, and its
        # neighbours are not.
        D = descriptor(parity_unguarded(BookToPrice()), c.rd)
        @test isnan(D[60, 1]) && isnan(D[61, 2])
        @test isfinite(D[59, 1]) && isfinite(D[62, 2])
        # A book equity at or below zero is a state of the firm, not a data error: the
        # fixture holds both, and the return on equity is NaN there with no refusal.
        R = descriptor(ReturnOnEquity(), c.rd)
        @test isnan(R[40, 7]) && all(isnan, R[41:50, 8])
    end
    # The negative case: `parity_ratio_case(:negative)`. Ours and the oracle refuse the same 13
    # Descriptors, whose negative field the oracle also refuses. Ours also refuses the nine
    # below, over a negative total of assets or a negative sales figure, where the oracle gives
    # NaN or refuses the scale; the stored output of each is the oracle's, and
    # `parity_unguarded` equals it. Measured `maxrel = 0.0`.
    @testset "Negative guarded fields refuse" begin
        both = ["DividendToPrice", "ForwardDividendToPrice", "ShareholderYield",
                "AnalystDispersionToPrice", "ShortInterest", "AssetsGrowthRateL1",
                "SalesGrowthRateL1", "IssuanceGrowthRateL1", "GrowthRateCapexL1",
                "AssetsGrowthRateL30", "SalesGrowthRateL30", "IssuanceGrowthRateL30",
                "GrowthRateCapexL30"]
        ours = ["DebtToAssets", "GrossProfitability", "GrossMargin", "ReturnOnAssets",
                "AssetTurnover", "CashFlowToAssets", "AccrualsCashFlow",
                "CapexToAssetsChangeInIntensityL1", "CapexToAssetsChangeInIntensityL30"]
        @test both ⊆ names && ours ⊆ names
        c = parity_ratio_case(:negative)
        changes = parity_load("ratio_lag_descriptors", "SmallNegative", "Changes")
        for (k, (name, de)) in enumerate(ds)
            if name in both
                @test_throws DomainError descriptor(de, c.rd)
                continue
            end
            oracle = parity_ratio_oracle(clean, changes, k)
            if name in ours
                @test_throws DomainError descriptor(de, c.rd)
            else
                @test parity_compare(vec(descriptor(de, c.rd)), oracle; name = name).ok
            end
            @test parity_compare(vec(descriptor(parity_unguarded(de), c.rd)), oracle;
                                 name = "unguarded " * name).ok
        end
    end
    # The non-negative guard reads the active observed cells alone (ADR 0108). A negative value
    # outside the active mask never reaches the output, so it does not refuse the panel.
    @testset "The non-negative guard reads the active cells" begin
        amsk = trues(4, 2)
        amsk[1:2, 2] .= false
        div = [1.0 -5.0; 1.0 -5.0; 1.0 2.0; 1.0 2.0]
        mcap = fill(10.0, 4, 2)
        pnl = asset_panel([NumericPanelInput(; name = "dividends_ttm", vals = div),
                           NumericPanelInput(; name = "market_cap", vals = mcap)];
                          amsk = amsk, emsk = amsk)
        rd = ReturnsResult(; nx = ["a", "b"], X = zeros(4, 2), pnl = pnl)
        @test panel_field(pnl, "dividends_ttm").vals[1, 2] == -5.0
        @test isequal(descriptor(DividendToPrice(), rd),
                      [0.1 NaN; 0.1 NaN; 0.1 0.2; 0.1 0.2])
        div[3, 2] = -5.0
        pnl = asset_panel([NumericPanelInput(; name = "dividends_ttm", vals = div),
                           NumericPanelInput(; name = "market_cap", vals = mcap)];
                          amsk = amsk, emsk = amsk)
        rd = ReturnsResult(; nx = ["a", "b"], X = zeros(4, 2), pnl = pnl)
        @test_throws DomainError descriptor(DividendToPrice(), rd)
    end
    # A quotient that overflows is NaN, as the oracle gives, and not an infinity that refuses the
    # exposure (the defect this ticket fixes).
    @testset "An overflowing quotient is NaN" begin
        c = parity_ratio_case(:clean)
        pnl = c.rd.pnl
        flds = map(pnl.pf) do f
            v = copy(f.vals)
            if f.name == "book_equity"
                v[50, 1] = 1.0e300
            elseif f.name == "market_cap"
                v[50, 1] = 1.0e-10
            end
            return NumericPanelInput(; name = f.name, vals = v)
        end
        rd = ReturnsResult(; nx = c.rd.nx, X = c.rd.X,
                           pnl = asset_panel(flds; amsk = pnl.amsk, emsk = pnl.emsk))
        D = descriptor(BookToPrice(), rd)
        @test isnan(D[50, 1])
        @test isequal(D[:, 2:end], descriptor(BookToPrice(), c.rd)[:, 2:end])
        E = factor_exposure(CompositeExposure(; descriptors = [BookToPrice()],
                                              bw = "market_cap"), rd)
        @test all(isfinite, E[50, 2:end])
    end
end

#=
Parity of map #1375 for the exponentially weighted and the rolling Descriptors (#1380). Every
`Parity_*` file this test reads is an output of the oracle, stored with the harness of #1376,
and each case in the table below states how it was made. The conventions of the harness are in
the comment that #1376 links; this file does not repeat them.

Every case reads the raw exchange: a Panel Field is `NaN` where it is not observed, because a
Descriptor reads its Panel Fields through the observed mask and never sees a fill value. The
small fixture holds a late listing, a delisting, a relisting, a holiday gap, an asset outside
the estimation mask and a tie, so each recursion meets each of them.

THE CLEAN FIXTURE. The fixture carries a negative volume and a negative short interest. The
oracle refuses both in `EWShareTurnover`, `EWAmihudIlliquidity` and `DaysToCover`, and since
#1380 the library refuses them too, through the `nonneg` guard that each of the three presets
sets on every Panel Field it reads. So their parity cases read `parity_1380_clean`, which
makes the two cells positive, and a testset below pins the refusal on the raw cells.

THE MEASURE. All 55 cases of the ticket agreed on the `NaN` pattern. The largest relative
difference of each family, over the cases stored here and the large-panel defaults measured
with them:

| Family | `maxrel` |
| --- | --- |
| `EWMean`, `EWMomentum` | 4.2e-13 (large panel, default) |
| `EWVolumeRatio`, `EWShareTurnover`, `EWAmihudIlliquidity`, `DaysToCover` | 0.0 |
| `EWVolatility`, `EWDownsideVolatility` | 3.3e-16 |
| `EWResidualVolatility`, `EWResidualDownsideVolatility` | 7.5e-16 |
| `EWBeta`, `EWMarketBeta` | 7.4e-14 |
| `EWMacroSensitivity` | 1.9e-13 |
| `EWDownsideBeta` | 1.1e-15 |
| `RollingLogReturn`, `RollingMomentum`, `Reversal` | 3.0e-12 cell by cell, 9.6e-17 against the largest entry |
| `RollingMax`, `MaxReturn` | 0.0 |

A rolling log return is the difference of two cumulative sums, on both sides. A window whose
sum is near zero keeps the round-off of the cumulative sums, so its relative error is large
while the error against the largest entry stays at the last bit. The rolling cases therefore
compare with `scale = :array`.
=#

# The fixture with its negative volume and its negative short interest made positive.
function parity_1380_clean(fx)
    rd = deepcopy(fx.rd)
    at = fx.at
    fv = panel_field(rd.pnl, "adj_volume").vals
    fv[at.negative_volume...] = abs(fv[at.negative_volume...])
    fs = panel_field(rd.pnl, "short_interest").vals
    fs[at.negative_short_interest...] = abs(fs[at.negative_short_interest...])
    return rd
end

# The MACRO series of the Exogenous Series, with a gap at observations 10, 30 and 31.
function parity_1380_gap(r::AbstractVector)
    g = copy(r)
    g[[10, 30, 31]] .= NaN
    return g
end

@testset "Parity: the exponentially weighted and the rolling Descriptors (#1380)" begin
    fxs = parity_small_panel()
    rds = fxs.rd
    rdsc = parity_1380_clean(fxs)
    rdl = parity_large_panel().rd
    macro_s = rds.E[:, findfirst(==("MACRO"), rds.ne)]
    # A volatility estimator whose warm-up is set apart from its half-life.
    vol(h, w) = RegimeAdjustedExpWeightedVariance(;
                                                  decay = PortfolioOptimisers.half_life_decay(h),
                                                  min_obs = w, centring = PreCentred(),
                                                  regime_method = nothing)
    # Each case: the unit, the case, the returns data, and the Descriptor. The case names the
    # fixture (the small raw panel, its clean copy, or the large raw panel) and the
    # configuration. The output is the Descriptor, `observations × assets`.
    cases = [("EWMomentum", "SmallH5S3", rds,
              r -> descriptor(EWMomentum(; half_life = 5, skip = 3), r)),
             ("EWMomentum", "SmallH5Min8Exp", rds,
              r -> descriptor(EWMomentum(; half_life = 5, skip = 0, min_obs = 8,
                                         exponentiate = true), r)),
             ("EWMomentum", "LargeDefault", rdl, r -> descriptor(EWMomentum(), r)),
             # The general form at a decay no half-life of the presets names.
             ("EWMean", "SmallDecay085", rds,
              r -> descriptor(EWMean(; decay = 0.85, min_obs = 6, skip = 2), r)),
             ("EWShareTurnover", "SmallCleanDefault", rdsc,
              r -> descriptor(EWShareTurnover(), r)),
             ("EWShareTurnover", "SmallCleanH5", rdsc,
              r -> descriptor(EWShareTurnover(; half_life = 5), r)),
             ("EWVolumeRatio", "SmallCleanDecay08", rdsc,
              r -> descriptor(EWVolumeRatio(; num = "adj_volume",
                                            den = "adj_shares_outstanding", decay = 0.8,
                                            min_obs = 4), r)),
             ("EWAmihudIlliquidity", "SmallCleanDefault", rdsc,
              r -> descriptor(EWAmihudIlliquidity(), r)),
             ("EWAmihudIlliquidity", "SmallCleanH5", rdsc,
              r -> descriptor(EWAmihudIlliquidity(; half_life = 5), r)),
             ("DaysToCover", "SmallCleanDefault", rdsc, r -> descriptor(DaysToCover(), r)),
             ("DaysToCover", "SmallCleanH5Min2", rdsc,
              r -> descriptor(DaysToCover(; half_life = 5, min_obs = 2), r)),
             ("EWVolatility", "SmallDefault", rds, r -> descriptor(EWVolatility(), r)),
             ("EWVolatility", "SmallH5", rds,
              r -> descriptor(EWVolatility(; half_life = 5), r)),
             ("EWVolatility", "SmallH8Min3", rds,
              r -> descriptor(EWVolatility(; ce = vol(8, 3)), r)),
             ("EWDownsideVolatility", "SmallDefault", rds,
              r -> descriptor(EWDownsideVolatility(), r)),
             ("EWDownsideVolatility", "SmallH5Mar", rds,
              r -> descriptor(EWDownsideVolatility(; half_life = 5, mar = 0.001), r)),
             ("EWResidualVolatility", "SmallDefault", rds,
              r -> descriptor(EWResidualVolatility(), r)),
             ("EWResidualVolatility", "SmallH5B8", rds,
              r -> descriptor(EWResidualVolatility(; half_life = 5, beta_half_life = 8), r)),
             ("EWResidualDownsideVolatility", "SmallDefault", rds,
              r -> descriptor(EWResidualDownsideVolatility(), r)),
             ("EWResidualDownsideVolatility", "SmallH5B8Mar", rds,
              r -> descriptor(EWResidualDownsideVolatility(; half_life = 5,
                                                           beta_half_life = 8,
                                                           mar = -0.001), r)),
             ("EWMarketBeta", "SmallDefault", rds, r -> descriptor(EWMarketBeta(), r)),
             ("EWMarketBeta", "SmallH8", rds,
              r -> descriptor(EWMarketBeta(; half_life = 8), r)),
             # Groups of three or four assets: a group prior at `min_group_size = 3`, and the
             # global prior for every group at the default of five.
             ("EWMarketBeta", "SmallH8Group3", rds,
              r -> descriptor(EWMarketBeta(; half_life = 8, group = "industry",
                                           min_group_size = 3), r)),
             ("EWMarketBeta", "SmallH8Group5", rds,
              r -> descriptor(EWMarketBeta(; half_life = 8, group = "industry"), r)),
             ("EWMarketBeta", "SmallH4Agg3", rds,
              r -> descriptor(EWMarketBeta(; half_life = 4, agg_obs = 3), r)),
             ("EWMarketBeta", "SmallH4Agg3Group3", rds,
              r -> descriptor(EWMarketBeta(; half_life = 4, agg_obs = 3, group = "industry",
                                           min_group_size = 3, bounds = (0.2, 0.9)), r)),
             ("EWBeta", "SmallDecay09", rds,
              r -> descriptor(EWBeta(; decay = 0.9, min_obs = 6), r)),
             # The reference return through `ref` and through `series`.
             ("EWMacroSensitivity", "SmallDefault", rds,
              r -> descriptor(EWMacroSensitivity(), r; ref = macro_s)),
             ("EWMacroSensitivity", "SmallH8", rds,
              r -> descriptor(EWMacroSensitivity(; series = "MACRO", half_life = 8), r)),
             ("EWMacroSensitivity", "SmallH8Gap", rds,
              r -> descriptor(EWMacroSensitivity(; half_life = 8), r;
                              ref = parity_1380_gap(macro_s))),
             ("EWMacroSensitivity", "SmallH4Agg3Gap", rds,
              r -> descriptor(EWMacroSensitivity(; half_life = 4, agg_obs = 3), r;
                              ref = parity_1380_gap(macro_s))),
             ("EWDownsideBeta", "SmallDefault", rds, r -> descriptor(EWDownsideBeta(), r)),
             ("EWDownsideBeta", "SmallH8Mar", rds,
              r -> descriptor(EWDownsideBeta(; half_life = 8, mar = -0.002), r)),
             ("RollingMomentum", "SmallW10S3", rds,
              r -> descriptor(RollingMomentum(; window = 10, skip = 3), r)),
             ("RollingMomentum", "SmallW10Exp", rds,
              r -> descriptor(RollingMomentum(; window = 10, skip = 0, exponentiate = true),
                              r)),
             ("RollingMomentum", "LargeW120", rdl,
              r -> descriptor(RollingMomentum(; window = 120), r)),
             ("Reversal", "SmallDefault", rds, r -> descriptor(Reversal(), r)),
             ("Reversal", "SmallW5", rds, r -> descriptor(Reversal(; window = 5), r)),
             ("RollingLogReturn", "SmallW7S2NegExp", rds,
              r -> descriptor(RollingLogReturn(; window = 7, skip = 2, sign = -1,
                                               exponentiate = true), r)),
             ("MaxReturn", "SmallDefault", rds, r -> descriptor(MaxReturn(), r)),
             ("MaxReturn", "SmallW5", rds, r -> descriptor(MaxReturn(; window = 5), r)),
             ("RollingMax", "SmallW2", rds, r -> descriptor(RollingMax(; window = 2), r))]
    rolling = ("RollingMomentum", "Reversal", "RollingLogReturn")
    for (unit, case, rd, f) in cases
        @testset "$unit $case" begin
            D = f(rd)
            O = parity_load(unit, case, "D")
            scale = unit in rolling ? :array : :cell
            r = parity_compare(D, O; scale = scale, name = "$(unit)_$(case)")
            @test r.pattern
            @test r.ok
        end
    end

    @testset "An inactive cell is missing on both sides, so no recursion reads it (#719)" begin
        # The oracle refuses a finite return outside the active mask, and the library masks the
        # returns before a recursion starts. The cases above cross a delisting and a
        # relisting; here the delisted asset of the fixture is `NaN` to its last observation.
        (i, t) = fxs.at.delist
        for (unit, case, rd, f) in cases
            rd === rds || continue
            @test all(isnan, f(rd)[(t + 1):end, i])
        end
    end

    @testset "A rolling maximum over one observation is the return itself (#720)" begin
        # The oracle refuses `window = 1`, a window that is well defined: the maximum of one
        # return. The library gives it, so the unit is better than the oracle there.
        D = descriptor(RollingMax(; window = 1), rds)
        E = ifelse.(rds.pnl.amsk, rds.X, NaN)
        @test isequal(D, E)
        @test isequal(descriptor(MaxReturn(; window = 1), rds), E)
    end

    @testset "A negative volume or short interest is refused, as the oracle refuses it" begin
        # A volume, a share count, a price and a short interest cannot be negative, so the
        # presets refuse a negative value of every Panel Field they read. Before #1380 the
        # share turnover read the negative volume as a negative turnover, and the days to
        # cover gave a negative Descriptor.
        for de in (EWShareTurnover(; half_life = 5), EWAmihudIlliquidity(; half_life = 5),
                   DaysToCover(; half_life = 5))
            @test_throws DomainError descriptor(de, rds)
        end
        @test_throws r"short_interest" descriptor(DaysToCover(;
                                                              nonneg = ["short_interest"]),
                                                  rds)
        @test_throws r"adj_volume" descriptor(DaysToCover(; nonneg = ["adj_volume"]), rds)
        # With the guard off, a negative volume in a denominator holds the state, and a
        # negative volume in a numerator enters the mean.
        (t, i) = fxs.at.negative_volume
        Dc = descriptor(EWAmihudIlliquidity(; half_life = 5), rdsc)
        Da = descriptor(EWAmihudIlliquidity(; half_life = 5, nonneg = nothing), rds)
        @test Da[t, i] == Da[t - 1, i]
        @test Da[t, i] != Dc[t, i]
        Dt = descriptor(EWShareTurnover(; half_life = 5, nonneg = nothing), rds)
        @test Dt[t, i] < Dt[t - 1, i]
        (ts, is) = fxs.at.negative_short_interest
        Dd = descriptor(DaysToCover(; half_life = 5, nonneg = nothing), rds)
        @test Dd[ts, is] < 0
        # The guard reads the active observed cells alone (ADR 0108): a negative volume that is
        # observed on an inactive cell reaches no output, so it is not refused, and the
        # Descriptor equals the one of the clean panel.
        rdo = deepcopy(rdsc)
        (i3, t3) = fxs.at.delist
        f = panel_field(rdo.pnl, "adj_volume")
        f.vals[t3 + 5, i3] = -1.0
        f.omsk[t3 + 5, i3] = true
        for de in (EWShareTurnover(; half_life = 5), EWAmihudIlliquidity(; half_life = 5),
                   DaysToCover(; half_life = 5))
            @test isequal(descriptor(de, rdo), descriptor(de, rdsc))
        end
        # The generic ratio does not guard by default: a difference of Panel Fields can be
        # negative by design.
        @test isnothing(EWVolumeRatio(; num = ["adj_volume" => 1, "short_interest" => -1],
                                      den = "adj_shares_outstanding", decay = 0.8,
                                      min_obs = 2).nonneg)
        # A guard must name a Panel Field the ratio reads.
        @test_throws ArgumentError EWShareTurnover(; nonneg = ["adj_close"])
        @test_throws ArgumentError EWAmihudIlliquidity(;
                                                       nonneg = ["adj_shares_outstanding"])
        @test_throws ArgumentError DaysToCover(; nonneg = ["market_cap"])
        @test_throws PortfolioOptimisers.IsEmptyError DaysToCover(; nonneg = String[])
        @test PortfolioOptimisers.panel_term_names(nothing) == String[]
        @test PortfolioOptimisers.panel_term_names(["adj_close", "adj_volume"]) ==
              ["adj_close", "adj_volume"]
    end

    @testset "An infinite reference return is refused on both paths (#1365)" begin
        rf = copy(macro_s)
        rf[6] = Inf
        @test_throws PortfolioOptimisers.IsNonFiniteError descriptor(EWMacroSensitivity(;
                                                                                        half_life = 8),
                                                                     rds; ref = rf)
        E = copy(rds.E)
        E[6, findfirst(==("MACRO"), rds.ne)] = Inf
        rdi = ReturnsResult(; nx = rds.nx, X = rds.X, ne = rds.ne, E = E, pnl = rds.pnl)
        @test_throws PortfolioOptimisers.IsNonFiniteError descriptor(EWMacroSensitivity(;
                                                                                        series = "MACRO",
                                                                                        half_life = 8),
                                                                     rdi)
    end
end
