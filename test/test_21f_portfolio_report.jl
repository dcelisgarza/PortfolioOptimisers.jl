#=
Issue #1943 (ADR 0228): a Portfolio Report. `portfolio_report` builds a `Report` from
prediction results of every type, a single fold, a walk-forward path and a population, and
`write_report` writes it as Markdown or as one self-contained HTML file. The HTML tests read
the output as a string, and need no browser.
=#
using Dates, Statistics, Clarabel, StatsPlots, GraphRecipes
const PO = PortfolioOptimisers

const RP_X = randn(StableRNG(987654321), 60, 4) .* 0.02 .+ 0.001
const RP_TS = Date(2024, 1, 1) .+ Day.(0:59)
const RP_RD = ReturnsResult(; nx = ["A", "B", "C", "D"], X = RP_X, ts = RP_TS)
const RP_ONE = predict(optimise(InverseVolatility(), RP_RD), RP_RD)
const RP_PATH = cross_val_predict(InverseVolatility(), RP_RD, IndexWalkForward(20, 10))
const RP_POP = cross_val_predict(InverseVolatility(), RP_RD,
                                 MultipleRandomised(IndexWalkForward(30, 10);
                                                    subset_size = 3, n_subsets = 3,
                                                    rng = StableRNG(11), seed = 7))
const RP_SLV = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false))
const RP_FRONT = cross_val_predict(MeanRisk(;
                                            opt = JuMPOptimiser(; slv = RP_SLV,
                                                                ret = ArithmeticReturn(;
                                                                                       settings = JuMPReturnsSettings(;
                                                                                                                      lb = Frontier(;
                                                                                                                                    N = 3))))),
                                   RP_RD, IndexWalkForward(30, 10))

# Two figures with one image MIME each, to reach the fallback of the writer.
struct RPPngOnly end
Base.show(io::IO, ::MIME"image/png", ::RPPngOnly) = print(io, "png")
struct RPSvgOnly end
Base.show(io::IO, ::MIME"image/svg+xml", ::RPSvgOnly) = print(io, "<svg/>")

rp_html(x) = sprint(show, MIME"text/html"(), x)
rp_md(x) = sprint(show, MIME"text/markdown"(), x)
rp_section(e, title) = e.sections[findfirst(s -> s.title == title, e.sections)]
function rp_alloc(n)
    return PO.GreedyAllocationResult(; retcode = OptimisationSuccess(),
                                     shares = collect(1.0:n), cost = collect(10.0:10:(10n)),
                                     w = fill(1 / n, n), cash = 2.5, fees = 0.5,
                                     fb = nothing)
end

@testset "#1943: the overview and the sections of every prediction type" begin
    preds = [RP_ONE, RP_PATH, RP_POP]
    nm = ["one", "path", "pop"]
    r = portfolio_report(preds; names = nm, groups = ["g1", "g2", "g2"])
    tab = r.overview[1]
    df = DataFrame(performance_summary(preds; names = nm))
    @test names(tab.df)[1] == "group"
    @test isequal(tab.df[:, 2:end], df)
    @test tab.df.group == [Dict(nm .=> ["g1", "g2", "g2"])[x] for x in df.name]
    @test tab.key == :name && tab.sortable
    @test DataFrames.nrow(tab.df) == 2 + length(RP_POP.pred)
    @test [e.name for e in r.entries] == nm
    @test [e.group for e in r.entries] == ["g1", "g2", "g2"]
    for e in r.entries
        @test [s.title for s in e.sections] == ["Statistics", "Weights"]
    end
    # A single fold gives a table of two columns, the figure and its value.
    st = r.entries[1].sections[1].blocks[1]
    @test names(st.df) == ["figure", "value"]
    @test st.df.value[st.df.figure .== "sharpe"][1] == df.sharpe[1]
    @test !("path" in st.df.figure) && !("name" in st.df.figure)
    # The four exposures of the book come from the summary, so the report computes none (#1962).
    for c in ("long_exposure", "short_exposure", "gross_exposure", "net_exposure")
        @test isequal(st.df.value[st.df.figure .== c][1], df[1, c])
    end
    # A population gives one sortable row for each path, and the distribution of each figure.
    pst = r.entries[3].sections[1].blocks
    @test length(pst) == 2
    @test DataFrames.nrow(pst[1].df) == length(RP_POP.pred) && pst[1].sortable
    @test pst[1].df.path == [p.id for p in RP_POP.pred]
    dist = pst[2].df
    srow = findfirst(==("sharpe"), dist.figure)
    sh = filter(isfinite, df.sharpe[df.name .== "pop"])
    @test dist.count[srow] == length(sh)
    @test dist.median[srow] ≈ median(sh)
    @test dist.min[srow] == minimum(sh) && dist.max[srow] == maximum(sh)
    @test !("date0" in dist.figure) && !("path" in dist.figure)
    # A figure with no finite value shows empty cells.
    trow = findfirst(==("excess_ret"), dist.figure)
    @test dist.count[trow] == 0 && ismissing(dist.median[trow])
    # The single-prediction method is a report of one entry, with no group column.
    r1 = portfolio_report(RP_ONE; name = "solo", meta = ["k" => "v"])
    @test length(r1.entries) == 1 && r1.entries[1].name == "solo"
    @test names(r1.overview[1].df)[1] == "name"
    @test r1.entries[1].meta == ["k" => "v"]
    @test portfolio_report(RP_ONE; group = "g").entries[1].group == "g"
end

@testset "#1943: the weights of a fold, a path, a population and the members" begin
    @testset "a single fold reads its investable view" begin
        w = PO.fold_target(RP_ONE)
        t = sort(abs.(w))[2]
        r = portfolio_report(RP_ONE; weight_threshold = t)
        wt = r.entries[1].sections[2].blocks[1]
        o = sortperm(abs.(w); rev = true)
        @test wt.df.asset == RP_ONE.rd.nx[o][1:3]
        @test wt.df.weight == w[o][1:3]
        @test occursin("1 asset(s)", wt.caption)
        _, iw, _, _, inx = PO.result_investable_view(RP_ONE.res, nothing, nothing,
                                                     RP_ONE.rd.nx)
        w0 = portfolio_report(RP_ONE; weight_threshold = 0).entries[1].sections[2].blocks[1]
        @test Set(w0.df.asset) == Set(inx)
        for (a, x) in zip(inx, iw)
            @test w0.df.weight[w0.df.asset .== a][1] == x
        end
        @test !occursin("not shown", w0.caption)
    end
    @testset "a path gives one column for each fold" begin
        wt = portfolio_report(RP_PATH; weight_threshold = 0).entries[1].sections[2].blocks[1]
        @test names(wt.df)[1:2] == ["asset", "mean"]
        @test names(wt.df)[3:end] == [string(first(f.rd.ts)) for f in RP_PATH.pred]
        W = PO.stacked_fold_weights(RP_PATH, PO.fold_target)
        nx = RP_PATH.pred[1].rd.nx
        for (i, a) in enumerate(nx)
            row = findfirst(==(a), wt.df.asset)
            @test wt.df.mean[row] ≈ mean(W[:, i])
            @test collect(wt.df[row, 3:end]) == W[:, i]
        end
        @test issorted(abs.(wt.df.mean); rev = true)
        # A fold with no timestamps is labelled by its position.
        @test PO.fold_label(nothing, 3) == "fold 3"
    end
    @testset "a population gives one column for each path, on the union of the names" begin
        wt = portfolio_report(RP_POP; weight_threshold = 0).entries[1].sections[2].blocks[1]
        allnames = unique(reduce(vcat,
                                 [collect(f.rd.nx) for p in RP_POP.pred for f in p.pred]))
        @test Set(wt.df.asset) == Set(allnames)
        @test names(wt.df)[3:end] ==
              [string(PO.population_path_key(p, k)) for (k, p) in enumerate(RP_POP.pred)]
        for (k, p) in enumerate(RP_POP.pred)
            col = string(PO.population_path_key(p, k))
            W = PO.stacked_fold_weights(p, PO.fold_target)
            nmk = unique(reduce(vcat, [collect(f.rd.nx) for f in p.pred]))
            m = vec(mean(W; dims = 1))
            for a in allnames
                row = findfirst(==(a), wt.df.asset)
                i = findfirst(==(a), nmk)
                @test wt.df[row, col] ≈ (isnothing(i) ? 0.0 : m[i])
            end
        end
        # A single-fold path is keyed by its position, and gives its target.
        pp = PopulationPredictionResult(; pred = [RP_ONE, RP_ONE])
        nmp, Wp, lp = PO.prediction_weights(pp)
        @test lp == ["1", "2"] && Wp[1, :] == PO.fold_target(RP_ONE)
        @test_throws PO.IsEmptyError PO.prediction_weights(PopulationPredictionResult())
        @test_throws PO.IsEmptyError PO.prediction_members(PopulationPredictionResult())
    end
    @testset "a prediction with many members gives one table for each member" begin
        @test PO.prediction_members(RP_FRONT) == 3
        @test PO.prediction_members(PopulationPredictionResult(; pred = [RP_FRONT])) == 3
        wts = portfolio_report(RP_FRONT; weight_threshold = 0).entries[1].sections[2].blocks
        @test length(wts) == 3
        for j in 1:3
            @test startswith(wts[j].caption, "Member $(j). ")
            W = PO.stacked_fold_weights(RP_FRONT, f -> PO.fold_target(f, j))
            @test wts[j].df.mean ≈ sort(vec(mean(W; dims = 1)); by = abs, rev = true)
        end
        st = portfolio_report(RP_FRONT).entries[1].sections[1].blocks[1]
        @test st.df.member == [1, 2, 3] && !("path" in names(st.df))
    end
    @testset "a fold with no asset names takes the positions" begin
        res = PO.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                         retcode = OptimisationSuccess(), w = [0.25, 0.75],
                                         fb = nothing)
        p = PredictionResult(; res = res,
                             rd = PO.PredictionReturnsResult(; X = [0.01, -0.02, 0.03]))
        nx, W, l = PO.prediction_weights(p)
        @test nx == ["1", "2"] && W == [0.25 0.75] && l == ["weight"]
        mp = MultiPeriodPredictionResult(; pred = [p, p])
        nx2, W2, l2 = PO.prediction_weights(mp)
        @test nx2 == ["1", "2"] && size(W2) == (2, 2) && l2 == ["fold 1", "fold 2"]
    end
end

@testset "#1943: the allocation of a fold, a path and a population" begin
    r = portfolio_report(RP_ONE; alloc = rp_alloc(4))
    e = r.entries[1]
    @test [s.title for s in e.sections] == ["Statistics", "Weights", "Allocation"]
    at = e.sections[3].blocks[1]
    @test at.df.asset == RP_ONE.rd.nx
    @test at.df.shares == 1.0:4 && at.df.cost == 10.0:10:40 && at.df.weight == fill(0.25, 4)
    @test occursin("Cash left: 2.5", at.caption) && occursin("Fees: 0.5", at.caption)
    @test_throws DimensionMismatch portfolio_report(RP_ONE; alloc = rp_alloc(3))
    # A path allocates the target of its last fold.
    rp = portfolio_report([RP_PATH]; allocs = [rp_alloc(4)])
    @test rp_section(rp.entries[1], "Allocation").blocks[1].df.asset ==
          RP_PATH.pred[end].rd.nx
    @test_throws ArgumentError portfolio_report(RP_POP; alloc = rp_alloc(4))
end

const RP_SETS = UniverseSets(;
                             dict = Dict("nx" => ["A", "B", "C", "D"],
                                         "nx_sector" =>
                                             Union{Missing, String}["x", "x", "y", missing],
                                         "nx_cur" => ["usd", "eur", "usd", "usd"]))
function rp_exposure(e, key)
    return rp_section(e, "Exposure").blocks[findfirst(s -> s.title == key,
                                                      rp_section(e, "Exposure").blocks)]
end

@testset "the Exposure section sums the weights of each group of a taxonomy column" begin
    @testset "a single fold, with a short and an asset with no label" begin
        res = PO.NaiveOptimisationResult(; pr = nothing, wb = nothing,
                                         retcode = OptimisationSuccess(),
                                         w = [0.5, 0.7, -0.3, 0.1], fb = nothing)
        p = predict(res, RP_RD)
        r = portfolio_report(p; sets = RP_SETS, taxonomy = ["nx_sector", "nx_cur"])
        e = r.entries[1]
        @test [s.title for s in e.sections] == ["Statistics", "Weights", "Exposure"]
        @test [s.title for s in rp_section(e, "Exposure").blocks] == ["nx_sector", "nx_cur"]
        t = only(rp_exposure(e, "nx_sector").blocks)
        @test names(t.df) == ["group", "long", "short", "net"] && t.sortable
        @test t.df.group == ["x", "y", "(none)"]
        @test t.df.long == [1.2, 0.0, 0.1] && t.df.short == [0.0, -0.3, 0.0]
        @test t.df.net ≈ [1.2, -0.3, 0.1]
        @test sum(t.df.net) ≈ sum(res.w)
        @test occursin("Gross exposure: 1.6.", t.caption)
        c = only(rp_exposure(e, "nx_cur").blocks)
        @test Set(c.df.group) == Set(["usd", "eur"])
        @test c.df.net[c.df.group .== "usd"][1] ≈ 0.3 &&
              c.df.short[c.df.group .== "usd"][1] ≈ -0.3
        # An asset that the universe does not name counts in `(none)`.
        s2 = UniverseSets(; dict = Dict("nx" => ["A", "B", "C"], "nx_g" => ["k", "k", "m"]))
        t2 = only(rp_exposure(portfolio_report(p; sets = s2, taxonomy = ["nx_g"]).entries[1],
                              "nx_g").blocks)
        @test t2.df.group == ["k", "m", "(none)"] && t2.df.net ≈ [1.2, -0.3, 0.1]
        # A label that is not a string shows as its text.
        s3 = UniverseSets(;
                          dict = Dict("nx" => ["A", "B", "C", "D"], "nx_n" => [1, 1, 2, 2]))
        t3 = only(rp_exposure(portfolio_report(p; sets = s3, taxonomy = ["nx_n"]).entries[1],
                              "nx_n").blocks)
        @test t3.df.group == ["1", "2"] && t3.df.net ≈ [1.2, -0.2]
        # The writers show the section.
        @test occursin("### Exposure\n\n#### nx\\_sector", rp_md(e))
        @test occursin("<h3>Exposure</h3>", rp_html(e))
    end
    @testset "the allocation of a fold gives a second set of columns" begin
        t = only(rp_exposure(portfolio_report(RP_ONE; alloc = rp_alloc(4), sets = RP_SETS,
                                              taxonomy = ["nx_cur"]).entries[1], "nx_cur").blocks)
        @test names(t.df) ==
              ["group", "long", "short", "net", "alloc_long", "alloc_short", "alloc_net"]
        @test t.df.alloc_net[t.df.group .== "usd"][1] ≈ 0.75
        @test t.df.alloc_net[t.df.group .== "eur"][1] ≈ 0.25
        @test all(iszero, t.df.alloc_short)
        @test occursin("realised weights of the allocation, with a gross exposure of 1.0.",
                       t.caption)
        e = portfolio_report(RP_ONE; alloc = rp_alloc(4), sets = RP_SETS,
                             taxonomy = ["nx_cur"]).entries[1]
        @test [s.title for s in e.sections] ==
              ["Statistics", "Weights", "Exposure", "Allocation"]
    end
    @testset "a path gives one row for each fold, and a row for its allocation" begin
        r = portfolio_report([RP_PATH]; allocs = [rp_alloc(4)], sets = [RP_SETS],
                             taxonomy = ["nx_sector"])
        t = only(rp_exposure(r.entries[1], "nx_sector").blocks)
        @test names(t.df) == ["portfolio", "x", "y", "(none)", "gross"]
        @test t.df.portfolio ==
              [[string(first(f.rd.ts)) for f in RP_PATH.pred]; "allocation"]
        for (k, f) in enumerate(RP_PATH.pred)
            w = PO.fold_target(f)
            @test t.df.x[k] ≈ w[1] + w[2] && t.df.y[k] ≈ w[3] && t.df[k, "(none)"] ≈ w[4]
            @test t.df.gross[k] ≈ sum(abs, w)
        end
        @test t.df.x[end] ≈ 0.5 && t.df.y[end] ≈ 0.25 && t.df.gross[end] ≈ 1.0
        @test occursin("The row `allocation`", t.caption)
        # A path with no allocation has no `allocation` row.
        t0 = only(rp_exposure(portfolio_report(RP_PATH; sets = RP_SETS,
                                               taxonomy = ["nx_sector"]).entries[1],
                              "nx_sector").blocks)
        @test DataFrames.nrow(t0.df) == length(RP_PATH.pred)
        @test !occursin("allocation", t0.caption)
        # A group with the label of a fixed column takes the suffix, and the fixed column keeps
        # its name. The column `gross` held the net of the group `gross` (R33 of #1817).
        sg = UniverseSets(;
                          dict = Dict("nx" => ["A", "B", "C", "D"],
                                      "nx_s" => ["gross", "gross", "portfolio", "v"]))
        tg = only(rp_exposure(portfolio_report(RP_PATH; sets = sg, taxonomy = ["nx_s"]).entries[1],
                              "nx_s").blocks)
        @test names(tg.df) == ["portfolio", "gross_1", "portfolio_1", "v", "gross"]
        for (k, f) in enumerate(RP_PATH.pred)
            w = PO.fold_target(f)
            @test tg.df.portfolio[k] == string(first(f.rd.ts))
            @test tg.df.gross[k] ≈ sum(abs, w)
            @test tg.df.gross_1[k] ≈ w[1] + w[2] && tg.df.portfolio_1[k] ≈ w[3]
        end
        # The suffix skips a name that a group already holds: the group `gross` beside the
        # group `gross_1` takes `gross_2`, and the group `gross_1` keeps its name.
        st = UniverseSets(;
                          dict = Dict("nx" => ["A", "B", "C", "D"],
                                      "nx_s" => ["gross", "gross_1", "portfolio", "v"]))
        tt = only(rp_exposure(portfolio_report(RP_PATH; sets = st, taxonomy = ["nx_s"]).entries[1],
                              "nx_s").blocks)
        @test names(tt.df) ==
              ["portfolio", "gross_2", "gross_1", "portfolio_1", "v", "gross"]
        for (k, f) in enumerate(RP_PATH.pred)
            w = PO.fold_target(f)
            @test tt.df.gross_2[k] ≈ w[1] && tt.df.gross_1[k] ≈ w[2]
            @test tt.df.portfolio_1[k] ≈ w[3] && tt.df.gross[k] ≈ sum(abs, w)
        end
    end
    @testset "a population gives one row for each path" begin
        t = only(rp_exposure(portfolio_report(RP_POP; sets = RP_SETS,
                                              taxonomy = ["nx_cur"]).entries[1], "nx_cur").blocks)
        nm, W, labels = PO.prediction_weights(RP_POP)
        @test t.df.portfolio == labels
        cur = Dict("A" => "usd", "B" => "eur", "C" => "usd", "D" => "usd")
        for i in axes(W, 1)
            @test t.df.usd[i] ≈ sum(W[i, j] for j in eachindex(nm) if cur[nm[j]] == "usd")
            @test t.df.usd[i] + t.df.eur[i] ≈ sum(view(W, i, :))
        end
    end
    @testset "a prediction with many members gives one table for each member" begin
        x = rp_exposure(portfolio_report(RP_FRONT; sets = RP_SETS, taxonomy = ["nx_cur"]).entries[1],
                        "nx_cur")
        @test length(x.blocks) == 3
        @test [startswith(t.caption, "Member $(j). ") for (j, t) in enumerate(x.blocks)] ==
              trues(3)
    end
    @testset "no sets or no taxonomy gives no section" begin
        r = portfolio_report([RP_ONE, RP_PATH]; names = ["a", "b"],
                             sets = [RP_SETS, nothing], taxonomy = ["nx_cur"])
        @test any(s -> s.title == "Exposure", r.entries[1].sections)
        @test !any(s -> s.title == "Exposure", r.entries[2].sections)
        @test !any(s -> s.title == "Exposure",
                   portfolio_report(RP_ONE; sets = RP_SETS).entries[1].sections)
        @test !any(s -> s.title == "Exposure",
                   portfolio_report(RP_ONE; taxonomy = ["nx_cur"]).entries[1].sections)
    end
    @testset "the refusals" begin
        @test_throws DimensionMismatch portfolio_report([RP_ONE, RP_PATH]; sets = [RP_SETS])
        err = try
            portfolio_report(RP_ONE; sets = RP_SETS, taxonomy = ["nx_sectr"])
        catch e
            e
        end
        @test err isa KeyError && occursin("Exposure section", string(err.key))
        bad = UniverseSets(; dict = Dict("nx" => ["A", "B", "C", "D"], "g" => ["k", "m"]))
        @test_throws DimensionMismatch portfolio_report(RP_ONE; sets = bad,
                                                        taxonomy = ["g"])
    end
end

@testset "#1943: the refusals" begin
    @test_throws PO.IsEmptyError portfolio_report(PredictionResult[])
    @test_throws ArgumentError portfolio_report([RP_ONE, RP_PATH]; names = ["a", "a"])
    @test_throws DimensionMismatch portfolio_report([RP_ONE, RP_PATH]; names = ["a"])
    @test_throws DimensionMismatch portfolio_report([RP_ONE, RP_PATH]; groups = ["a"])
    @test_throws DimensionMismatch portfolio_report([RP_ONE, RP_PATH]; meta = [nothing])
    @test_throws DimensionMismatch portfolio_report([RP_ONE, RP_PATH]; allocs = [nothing])
    err = try
        portfolio_report([RP_ONE, RP_PATH]; groups = ["a"])
    catch e
        e
    end
    @test occursin("`groups`", err.msg)
    @test_throws DomainError portfolio_report(RP_ONE; weight_threshold = -1)
    @test_throws DomainError portfolio_report(RP_ONE; weight_threshold = Inf)
    @test_throws ArgumentError portfolio_report(RP_ONE; entry_figures = [p -> nothing])
    @test_throws ArgumentError portfolio_report(RP_ONE; overview_figures = [:a => identity])
    @test_throws ArgumentError ReportFigure(1)
    err = try
        ReportFigure(1)
    catch e
        e
    end
    @test occursin("Int64", err.msg)
    @test_throws DomainError ReportTable(DataFrame(; a = [1]); digits = 0)
    @test_throws ArgumentError ReportTable(DataFrame(; a = [1]); key = :b)
    e = ReportEntry("x", ReportSection[])
    @test_throws ArgumentError Report("t", [e, e])
    r = portfolio_report(RP_ONE)
    dir = mktempdir()
    @test_throws ArgumentError write_report(joinpath(dir, "r.txt"), r)
    @test_throws ArgumentError write_report(joinpath(dir, "r.md"), r;
                                            format = MIME"text/plain"())
    @test_throws ArgumentError write_report(joinpath(dir, "r.md"), r;
                                            figure_mime = MIME"image/jpeg"())
    # A figure function that throws stops the build with its own error.
    @test_throws ErrorException portfolio_report(RP_ONE;
                                                 entry_figures = ["x" => p -> error("boom")])
end

@testset "#1943: every string of the caller is escaped" begin
    bad = "<script>alert(1)</script> & \" | x"
    rd = ReturnsResult(; nx = ["A", bad, "C", "D"], X = RP_X, ts = RP_TS)
    p = predict(optimise(InverseVolatility(), rd), rd)
    r = portfolio_report([p, RP_PATH]; names = [bad, "path"], groups = [bad, "g"],
                         meta = [[bad => bad], nothing])
    h = rp_html(r)
    m = rp_md(r)
    for s in (h, m)
        @test !occursin("<script>alert", s)
        @test occursin("&lt;script&gt;alert", s) && occursin("&amp;", s)
        @test occursin("&quot;", s)
    end
    @test occursin("\\|", m) && occursin("\\(1\\)", m)
    # A line break stays in its paragraph or its table cell.
    @test PO.md_escape("a\nb\r\nc") == "a b  c"
    # The script of the file is the only `<script>` element.
    @test count("<script>", h) == 1
end

@testset "#1943: the HTML file is self-contained" begin
    r = portfolio_report([RP_ONE, RP_PATH]; names = ["one", "path"])
    h = rp_html(r)
    @test startswith(h, "<!doctype html>")
    @test occursin("<meta http-equiv=\"Content-Security-Policy\" content=\"$(PO.REPORT_CSP)\">",
                   h)
    @test !occursin("<link", h) && !occursin("<script src", h)
    @test !occursin("http://", h) && !occursin("https://", h)
    @test occursin(PO.REPORT_CSS, h) && occursin(PO.REPORT_JS, h)
    @test count("<article class=\"po-entry\"", h) == 2
    @test occursin("id=\"entry-1\"", h) && occursin("id=\"entry-2\"", h)
    @test occursin("data-entry=\"one\"", h)
end

@testset "#1943: the selector and the groups" begin
    h1 = rp_html(portfolio_report(RP_ONE))
    @test !occursin("id=\"po-select\"", h1) && !occursin("class=\"po-groups\"", h1)
    @test occursin("id=\"po-theme\"", h1)
    h2 = rp_html(portfolio_report([RP_ONE, RP_PATH, RP_POP]; names = ["a", "b", "c"],
                                  groups = [nothing, "g1", "g2"]))
    @test count("<option", h2) == 3 && count("<optgroup", h2) == 2
    @test occursin("<option value=\"entry-1\">a</option><optgroup label=\"g1\">", h2)
    @test count("type=\"checkbox\"", h2) == 2
    h3 = rp_html(portfolio_report([RP_ONE, RP_PATH]; names = ["a", "b"]))
    @test count("<option", h3) == 2 && !occursin("<optgroup", h3)
    @test !occursin("class=\"po-groups\"", h3)
end

@testset "#1943: the blocks and the writers" begin
    sec = ReportSection("Outer",
                        [ReportText("first para\n\nsecond *para*"),
                         ReportSection("Inner",
                                       [ReportTable(DataFrame(;
                                                              a = [1.234567, NaN, Inf,
                                                                   -Inf],
                                                              b = [missing, nothing, true,
                                                                   Date(2024, 1, 1)],
                                                              c = Any[1 // 2, 2, 3, 4]);
                                                    caption = "cap", digits = 3)]),
                         ReportFigure(RPPngOnly(); caption = "png"),
                         ReportFigure(RPSvgOnly())])
    e = ReportEntry("e", [sec])
    m = rp_md(e)
    @test occursin("## e\n\n### Outer\n\nfirst para\n\nsecond \\*para\\*\n\n#### Inner", m)
    @test occursin("*cap*\n\n| a | b | c |\n| --- | --- | --- |\n", m)
    @test occursin("| 1.23 | — | 1//2 |", m) && occursin("| Inf | true | 3 |", m)
    @test occursin("| -Inf | 2024\\-01\\-01 | 4 |", m)
    @test occursin("![png](data:image/png;base64,", m)
    @test occursin("![](data:image/svg+xml;base64,", m)
    h = rp_html(e)
    @test occursin("<h3>Outer</h3>", h) && occursin("<h4>Inner</h4>", h)
    @test occursin("<p>first para</p>\n<p>second *para*</p>", h)
    @test occursin("<td class=\"po-num\" data-v=\"1.234567\">1.23</td>", h)
    @test occursin("<td>—</td>", h)
    @test occursin("data-v=\"Infinity\">Inf</td>", h) && occursin("data-v=\"-Infinity\"", h)
    @test occursin("<td data-v=\"true\">true</td>", h)
    @test occursin("<td data-v=\"2024-01-01\">2024-01-01</td>", h)
    @test occursin("<td class=\"po-num\" data-v=\"2\">2</td>", h)
    @test occursin("<td data-v=\"1//2\">1//2</td>", h)
    @test occursin("src=\"data:image/png;base64,", h)
    # The PNG preference takes the SVG image of a figure with no PNG.
    io = IOBuffer()
    show(IOContext(io, :report_figure_mime => MIME"image/png"()), MIME"text/markdown"(),
         ReportFigure(RPSvgOnly()))
    @test occursin("data:image/svg+xml;base64,", String(take!(io)))
    @test PO.report_level(IOContext(IOBuffer(), :report_level => 9)) == 6
    # One line in the REPL.
    @test repr(ReportText(repeat("x", 50))) == "ReportText(\"$(repeat("x", 40))\"…)"
    @test repr(ReportText("short")) == "ReportText(\"short\")"
    @test repr(sec.blocks[2].blocks[1]) == "ReportTable(4×3 table)"
    @test repr(sec.blocks[3]) == "ReportFigure($(RPPngOnly))"
    @test repr(sec) == "ReportSection(\"Outer\", 4 blocks)"
    @test repr(e) == "ReportEntry(\"e\", 1 sections)"
    r = Report("T", [e]; overview = [ReportText("ov")], meta = ["by" => "me"])
    @test repr(r) == "Report(\"T\", 1 entries)"
    mr = rp_md(r)
    @test startswith(mr, "# T\n\n- **by**: me\n\n## Overview\n\nov\n\n## e")
    @test occursin("<ul class=\"po-meta\"><li><b>by</b>: me</li></ul>", rp_html(r))
    @test occursin("Group: g", rp_md(ReportEntry("e", ReportSection[]; group = "g")))
end

@testset "#1943: write_report writes the bytes of show" begin
    r = Report("T", [ReportEntry("e", [ReportSection("S", [ReportFigure(RPSvgOnly())])])])
    dir = mktempdir()
    for (f, mime) in (("r.html", MIME"text/html"()), ("r.htm", MIME"text/html"()),
                      ("r.md", MIME"text/markdown"()))
        path = joinpath(dir, f)
        @test write_report(path, r) == path
        @test read(path, String) == sprint(show, mime, r)
    end
    path = joinpath(dir, "x.out")
    @test write_report(path, r; format = MIME"text/markdown"()) == path
    @test read(path, String) == rp_md(r)
    write_report(path, r; format = MIME"text/html"(), figure_mime = MIME"image/png"())
    @test occursin("data:image/svg+xml;base64,", read(path, String))
end

@testset "#1943: the default figures of the plotting extension" begin
    f = default_report_figures()
    @test first.(f.overview) == ["Cumulative returns", "Risk and return"]
    @test length(f.entry) == 4
    held = cross_val_predict(InverseVolatility(), RP_RD,
                             IndexWalkForward(20, 10; wd = SelfFinancingDrift())).pred[1]
    preds = [RP_ONE, RP_PATH, RP_POP, held, RP_FRONT]
    nm = ["one", "path", "pop", "held", "front"]
    r = portfolio_report(preds; names = nm, overview_figures = f.overview,
                         entry_figures = f.entry)
    @test count(b -> b isa ReportFigure, r.overview) == 2
    nfig(k) = length(rp_section(r.entries[k], "Figures").blocks)
    # The risk contribution reads only a fold with a Held Weights record.
    @test nfig(1) == 3 && nfig(2) == 3 && nfig(3) == 3 && nfig(4) == 4
    # A prediction with many members draws none of the four.
    @test !any(s -> s.title == "Figures", r.entries[5].sections)
    dir = mktempdir()
    h = write_report(joinpath(dir, "r.html"), r)
    m = write_report(joinpath(dir, "r.md"), r)
    sh = read(h, String)
    @test count("src=\"data:image/svg+xml;base64,", sh) == 2 + 3 + 3 + 3 + 4
    @test !occursin("http://", sh) && !occursin("https://", sh)
    @test count("](data:image/", read(m, String)) == 15
    hp = write_report(joinpath(dir, "rp.html"), r; figure_mime = MIME"image/png"())
    @test count("src=\"data:image/png;base64,", read(hp, String)) == 15
end

@testset "#2056: the default figures draw a prediction with a NaN return" begin
    # A walk-forward fold that fails to solve has `NaN` weights, and so `NaN` returns for its
    # test rows. The histogram of the returns read the whole series and threw, which stopped
    # the build of the report.
    path = deepcopy(RP_PATH)
    path.mrd.X[5] = NaN
    pop = deepcopy(RP_POP)
    pop.pred[1].mrd.X[3] = NaN
    f = default_report_figures()
    r = portfolio_report([path, pop]; names = ["path", "pop"],
                         overview_figures = f.overview, entry_figures = f.entry)
    nfig(k) = length(rp_section(r.entries[k], "Figures").blocks)
    @test nfig(1) == 3 && nfig(2) == 3
end

@testset "#1943: the composition of a path and of a population draws each weight under its own name" begin
    # The rows are built by hand from the weights and the names of each fold, so the test
    # does not read `prediction_weights`. A bar series holds `NaN` separators, so the values
    # compare one entry at a time.
    labels(p) = [s[:label] for s in p.series_list]
    function same_plot(p, q)
        return labels(p) == labels(q) &&
               all(all(isapprox.(a[:y], b[:y]; nans = true, atol = 1e-12))
                   for (a, b) in zip(p.series_list, q.series_list))
    end
    # The paths of the population hold the subsets [1, 2, 4], [1, 3, 4] and [1, 2, 4], and the
    # mean of a path sits on the union of the names, in the order of first appearance. The old
    # plot put the weights of every path on the names of the first path.
    paths = RP_POP.pred
    nms = unique(reduce(vcat, [collect(f.rd.nx) for p in paths for f in p.pred]))
    @test nms == ["A", "B", "D", "C"]
    rows = map(paths) do p
        m = zeros(length(nms))
        for f in p.pred, (i, n) in enumerate(f.rd.nx)
            m[findfirst(==(n), nms)] += f.res.w[i] / length(p.pred)
        end
        return m
    end
    @test same_plot(plot_composition(RP_POP), plot_stacked_bar_composition(rows, nms))
    @test same_plot(plot_stacked_area_composition(RP_POP),
                    plot_stacked_area_composition(rows, nms))
    # Asset D has no returns over the fit window of the first fold, so the Investable Mask
    # drops it there. The weights of a fold sit on the universe of the caller, with a zero for
    # D, and the old plot read them beside the four names of the first fold.
    Xg = copy(RP_X)
    Xg[1:25, 4] .= NaN
    rdg = ReturnsResult(; nx = ["A", "B", "C", "D"], X = Xg, ts = RP_TS)
    pg = cross_val_predict(InverseVolatility(), rdg, IndexWalkForward(20, 10))
    @test collect(pg.pred[1].rd.nx) == ["A", "B", "C"]
    wg = [f.res.w for f in pg.pred]
    @test iszero(wg[1][4]) && !iszero(wg[end][4])
    @test same_plot(plot_composition(pg),
                    plot_stacked_bar_composition(wg, ["A", "B", "C", "D"]))
    @test same_plot(plot_stacked_area_composition(pg),
                    plot_stacked_area_composition(wg, ["A", "B", "C", "D"]))
end
