#=
```@meta
Description = "A portfolio report in PortfolioOptimisers.jl: compare searched candidates, a population of paths and one allocated fold in one HTML file."
```

# [Portfolio report](@id example-portfolio-report)

[`portfolio_report`](@ref) puts several predictions into one report. The report has an overview
table with one row for each portfolio, and one entry for each prediction with its statistics, its
weights and its figures. An entry also shows a finite allocation when you give one. [`write_report`](@ref) writes the report as
Markdown, or as one HTML file that holds its style, its script and its figures. The HTML file opens
in any browser with no network access.

This page builds one report from the three kinds of prediction:

 1. the best candidates of a grid search, each one predicted over a walk-forward path,
 2. a population of paths from a combinatorial cross-validation,
 3. one prediction on the last month, with a finite allocation of whole shares.

!!! tip "When to reach for this"
    Reach for a report when you want to compare many predictions side by side, or to give the
    results to someone who does not run Julia. Each figure and each statistic of the report comes
    from a function of the library, so a report shows nothing that you cannot compute yourself. Use
    [`performance_summary`](@ref) when you want the numbers alone.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, Clarabel, HiGHS, StatsPlots,
      GraphRecipes

#=
## 1. The data

We use two years of daily prices. A short window keeps the report small, because a figure of a
long daily series is large.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252 * 2):end]
rd = prices_to_returns(X)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

#=
## 2. The candidates of a search

We search the risk measure of a minimum-risk portfolio. The search scores each candidate on a
walk-forward scheme with a training window of one year and a test window of one month.
`n_best = 3` keeps the three best candidates, best first, in `ranked`. The
[hyperparameter tuning example](@ref example-hyperparameter-tuning) explains the search.
=#

mr = MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; slv = slv))
wf = IndexWalkForward(252, 21)
gs = GridSearchCrossValidation(["r" => [Variance(), ConditionalValueatRisk(),
                                        ConditionalDrawdownatRisk(), UlcerIndex(),
                                        MaximumDrawdown()]];
                               r = MeanReturnRiskRatio(; rk = ConditionalValueatRisk(),
                                                       rf = rf), cv = wf, n_best = 3)
gs_res = search_cross_validation(mr, gs, rd)
[nameof(typeof(e.r)) for e in gs_res.ranked]

#=
A search scores the candidates, but it keeps no prediction. We predict each ranked candidate over
the same walk-forward scheme. Each prediction is a
[`MultiPeriodPredictionResult`](@ref), one path of consecutive test windows.
=#

cands = [cross_val_predict(e, rd, wf) for e in gs_res.ranked]

#=
## 3. A population of paths

A walk-forward path is one history out of many. [`CombinatorialCrossValidation`](@ref) tests the
best candidate on several paths, and the result is a [`PopulationPredictionResult`](@ref). Its
entry in the report gives one row for each path, and the distribution of each statistic over the
paths.
=#

ppred = cross_val_predict(gs_res.opt, rd,
                          CombinatorialCrossValidation(; n_folds = 5, n_test_folds = 2))
length(ppred.pred)

#=
## 4. One fold and its allocation

We fit the best candidate on every row but the last month, and predict the last month. The
result is a [`PredictionResult`](@ref). [`DiscreteAllocation`](@ref) turns its weights into whole
shares of a budget of \$100,000 at the last prices. The
[finite allocation example](@ref example-finite-allocation) explains the allocation.
=#

train, test = train_test_split(rd; test_size = 21)
res = optimise(gs_res.opt, train)
pred = predict(res, test)

mip_slv = Solver(; name = :highs, solver = HiGHS.Optimizer,
                 settings = Dict("log_to_console" => false))
alloc = optimise(DiscreteAllocation(; slv = mip_slv),
                 FiniteAllocationInput(; w = res.w, prices = vec(values(X)[end, :]),
                                       cash = 100_000.0))

#=
## 5. The report

`portfolio_report` takes the predictions as one vector and their names. The keywords that it does
not use pass to [`performance_summary`](@ref). We keep them in `ps_kw`: `ppy` annualises the daily
figures, `rf` is the risk-free rate of one day, and `measures` adds two risk measures by name.
`groups` labels each prediction, and the HTML file shows or hides the portfolios of each group.
`allocs` gives the finite allocation of each prediction, or `nothing`.

[`default_report_figures`](@ref) gives the default figures when `StatsPlots` is loaded:
`overview` draws every prediction on one figure, and `entry` draws one prediction. A figure of
your own is a pair `caption => f`. An entry figure calls `f(pred)`, and a function that returns
`nothing` adds no figure. Here the turnover figure draws a walk-forward path and skips the other
two kinds of prediction.
=#

turnover_figure(p::MultiPeriodPredictionResult) = plot_turnover(p)
turnover_figure(::Any) = nothing

figs = default_report_figures()
preds = [cands; ppred; pred]
labels = ["search rank 1", "search rank 2", "search rank 3", "best, all paths",
          "best, last month"]
ps_kw = (; ppy = 252, rf = rf,
         measures = [:cvar_95 => ConditionalValueatRisk(), :ulcer => UlcerIndex()])
r = portfolio_report(preds; names = labels, title = "Minimum risk portfolios",
                     groups = ["search", "search", "search", "population", "single fold"],
                     allocs = [nothing, nothing, nothing, nothing, alloc],
                     overview_figures = figs.overview,
                     entry_figures = [figs.entry; "Turnover" => turnover_figure], ps_kw...);

#=
A `Report` shows itself as the whole HTML document, so the cell above ends with a semicolon. The
next cell lists the sections of each entry. Only the single fold has an Allocation section.
=#

[e.name => [s.title for s in e.sections] for e in r.entries]

#=
The overview table of the report is the table of `performance_summary` with the same keywords,
so the numbers of the report are the numbers that you compute yourself.
=#

DataFrame(performance_summary(preds; names = labels, ps_kw...))

#=
## 6. Write the report

The extension of the path chooses the format. The figures are SVG images by default, and an SVG
image holds every point of its series. PNG images keep the file small, so we write both files with
PNG figures and print their sizes in MiB.
=#

html = write_report("portfolio_report.html", r; figure_mime = MIME"image/png"())
md = write_report("portfolio_report.md", r; figure_mime = MIME"image/png"())
(html = filesize(html) / 2^20, md = filesize(md) / 2^20)

#=
The Markdown file starts with the title and the overview table, and a figure is an image with its
data in the link.
=#

foreach(println, first(eachline(md), 6))

#=
## 7. The report on this page

The frame below shows the HTML file that the cell above wrote. Choose a portfolio in the list at
the top, or click its row of the overview, to open its entry. Click a column head to sort a table.
Clear the box of a group to hide its portfolios, and use the theme button to change between light
and dark. You can also [open the report in its own tab](portfolio_report.html).

```@raw html
<iframe src="../portfolio_report.html" title="Portfolio report" loading="lazy" style="width: 100%; height: 80vh; border: 0"></iframe>
```
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - #1951. The `@example` blocks run in docs/build/examples/6_post_processing, so the
#src   relative write_report path puts the file next to the page directory. The Markdown link
#src   `portfolio_report.html` is resolved by Documenter against the build dir (file must exist
#src   when CrossReferences runs, after ExpandTemplates) and rewritten for prettyurls. The raw
#src   iframe is not rewritten, so it carries `../`.
#src - The notebook run writes the two files into examples/6_post_processing; .gitignore holds them.
#src - MeanAbsoluteDeviation is not a RiskMeasure, so MeanRisk refuses it in the grid.
