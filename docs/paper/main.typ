#import "@preview/clean-math-paper:0.2.7": *
#import "@preview/jlyfish:0.1.0": *

// ===========================================================================
// BUILDING THIS PAPER
//
// The listing in figure 1 is executed at build time by jlyfish, so it cannot
// drift away from the library. Two halves must run: TypstJlyfish evaluates the
// Julia and writes main-jlyfish.json, and typst renders the document.
//
// CI BUILDS THIS, in .github/workflows/Paper.yml, and that job is the gate on
// drift. It moves main-jlyfish.json aside before it builds, because the file is
// committed and `recompute: false` keys a cell on its own text, so a build that
// kept the cache would render the stored answer and prove nothing. It then fails
// if any cell reports "failed": true, and fails again if the rebuilt cells
// disagree with the committed ones.
//
// So COMMIT A REBUILT main-jlyfish.json whenever the listing or its answer
// changes. A stale one fails the job by design. Locally that is the one-shot
// command below, run from docs/paper.
//
// Run both from THIS directory (docs/paper). Inside a cell, pwd() and @__DIR__
// resolve to the directory Julia was launched from, and the listing reaches the
// price data through @__DIR__.
//
// One shot, evaluate and then build the PDF:
//
//   julia --project=@jlyfishdoc -e 'import TypstJlyfish; TypstJlyfish.compile("main.typ")'
//
// Live, in two terminals:
//
//   julia --project=@jlyfishdoc -e 'import TypstJlyfish; TypstJlyfish.watch("main.typ")'
//   typst watch main.typ
//
// @jlyfishdoc is a shared environment holding only TypstJlyfish. Create it once:
//
//   julia --project=@jlyfishdoc -e 'using Pkg; Pkg.add("TypstJlyfish")'
//
// These are Julia expressions. Typing TypstJlyfish.compile("main.typ") straight
// at a shell prompt gives `bash: syntax error near unexpected token`.
//
// PREVIEWS
//
// An editor preview needs no extra setup: tinymist in VS Code, or the typst
// extension in Zed. Typst tracks json() as a file dependency, so the preview
// re-renders by itself every time TypstJlyfish.watch rewrites
// main-jlyfish.json. Leave that watcher running beside the preview and the
// figures refresh as the code changes. There is no need for `typst watch` as
// well; the preview already watches. Only the Julia half must be started by
// hand, because a preview never evaluates Julia on its own.
//
// FOUR THINGS THAT BREAK IT
//
//   1. fig1.svg must stay in git, and figure 2 must keep loading it from disk.
//      TypstJlyfish runs `typst query` BEFORE it evaluates any Julia, so if the
//      file is missing the query fails, Julia never runs, and the file is never
//      written. That deadlock's only symptom is "Info: Typst query failed".
//      Returning the plot from the cell would dodge it, and jlyfish supports
//      that, but its image path calls image.decode, which typst 0.15 REMOVED.
//      An embedded figure therefore renders under the 0.12 that jlyfish bundles
//      and breaks every 0.15 preview. A file on disk is the only form both
//      compilers agree on. Delete fig1.svg and the build wedges.
//   2. main-jlyfish.json is committed, like fig1.svg and main.pdf: without it a
//      bare `typst compile` and every editor preview error at
//      #read-julia-output. Two rules follow. Never commit one whose cells carry
//      "failed": true, or the paper renders error text. And keep the cells free
//      of absolute paths, which is why the listing ends in `nothing`: a cell's
//      RESULT is stored, and Plots.savefig returns an abspath.
//   3. `recompute: false` makes a cell re-run exactly when its own code text
//      changes. The cache key is the CELL text, so a cell reading
//      include("paper.jl") would never notice that paper.jl had changed.
//      It also caches FAILURES. A cell that errored once is skipped for ever,
//      with "Info: Skipping recomputation of ...", even after you fix the cause
//      somewhere else, such as in the package set. Delete main-jlyfish.json and
//      build again to clear it.
//   4. Cells run in a throwaway environment, so a plain #jl-pkg("PortfolioOptimisers")
//      would resolve the REGISTERED release. The listing uses API that is newer
//      than any release, so the paper devs this checkout instead, through the
//      relative path "../..". That is resolved against the directory Julia was
//      launched from, which is one more reason to launch from docs/paper. Once a
//      release catches up, a plain #jl-pkg("PortfolioOptimisers") is the better
//      default for a paper: it then shows what a reader gets from the registry.
//
// TypstJlyfish drives its own bundled typst, 0.12 through Typst_jll, and not the
// one on PATH, and no keyword overrides it. So this document is compiled by two
// different typst versions: 0.12 for the jlyfish query and its own PDF, 0.15 for
// tinymist and a bare `typst compile`. Keep it working under both. That is what
// rules out image.decode above, and it holds while clean-math-paper declares
// compiler = "0.12.0". A Typst package needing something newer would fail the
// jlyfish query while `typst compile` kept working, a confusing pair of symptoms.
// ===========================================================================

#let date = datetime.today().display("[month repr:long] [day], [year]")
#let repo = "https://github.com/dcelisgarza/PortfolioOptimisers.jl/"
#let docs = "https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/"
#let catalogue_link = "https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/capability_catalogue/"
#let examples_link = "https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples/"
#let user_guide_link = "https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/user_guide/00_User_Guide/"
#let api_link = "https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/00_API/"
#let codecov = "https://app.codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl"
#let po = link(repo)[PortfolioOptimisers.jl]
#let podocs = link(docs)[documentation]
#let catalogue = link(catalogue_link)[capability catalogue]
#let examples = link(examples_link)[examples]
#let user_guide = link(user_guide_link)[user guide]
#let api = link(api_link)[API]
#let coverage = link(codecov)[coverage]
#let jump = link("https://github.com/jump-dev/JuMP.jl")[JuMP.jl]
#let statsbase = link("https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator")[StatsBase.jl]

// Modify some arguments, which can be overwritten in the template call
#page-args.insert("numbering", "1/1")
#text-args-title.insert("size", 2em)
#text-args-title.insert("fill", black)
#text-args-authors.insert("size", 12pt)

#show: template.with(
  title: link(repo)[PortfolioOptimisers.jl],
  authors: (
    (name: "Daniel Celis Garza", affiliation-id: 1, orcid: "0000-0003-4622-2234"),
  ),
  affiliations: (
    (id: 1, name: "Independent researcher, Oxford, UK"),
  ),
  date: date,
  heading-color: rgb("#0000ff"),
  link-color: rgb("#008002"),
  abstract: [Portfolio optimisation is the science of either: _1)_ Minimising risk whilst keeping returns to acceptable levels. _2)_ Maximising returns whilst keeping risk to acceptable levels. To some definition of acceptable, and with any number of additional constraints available to the optimisation type. There exist myriad statistical, pre- and post-processing algorithms, optimisations, and constraints that allow one to explore an extensive landscape of "optimal" portfolios. #po is an attempt at providing as many of these as possible under a single banner and making it accessible to all. The library is highly composable, so the many different components can be mixed and matched in arbitrary ways as long as they are compatible, providing users with the ability to create complex portfolio optimisation workflows in accordance to their needs and creativity. We make extensive use of Julia's type system, module extensions, and multiple dispatch to achieve excellent performance, robustness, correctness and testability, without sacrificing maintainability or usability.],
  keywords: ("Portfolio Optimisation", "Quantitative Investment", "Conic Optimisation", "Parameter Estimation"),
)
#show heading: set block(above: 1em, below: 1em)
#show heading.where(level: 1): set block(spacing: 1.5em)
#show heading.where(level: 2): set block(spacing: 1.25em)
#set par(first-line-indent: 2em, spacing: 2em, leading: 1.2em)
#set text(font: "New Computer Modern")
#set enum(numbering: "1.a.i.")
#set math.equation(numbering: "(1)")

= Introduction
The field of portfolio optimisation was introduced by Harry Markowitz's seminal 1952 paper, _Portfolio Selection_ @markowitz1952. The field has grown enormously since then, and so has the catalogue of that model's failure modes. It attempts to summarise the entire distribution of returns in highly compressed summary statistics, both of which can be sensitive to outliers and atypical conditions for the lookback period. The optimiser then behaves as an error maximiser rather than a diversifier @michaud1989, is acutely sensitive to the expected returns it is given @best1991, and suffers roughly an order of magnitude more from errors in the means than from errors in the covariance @chopra1993. The equal weighted portfolio remains a stubborn out of sample benchmark @demiguel2009naive. Sixty years of responses, from constraints @jagannathan2003 and shrinkage @ledoit2004 to norm regularisation @demiguel2009norm, robust formulations and hierarchical ones, are by now a literature of their own @kolm2014 @cajasbook @lopezdepradobook @dppalomarbook.

Unfortunately, most implementations live in disparate, unconnected, often proprietary, and/or bespoke codebases, which limit their applicability. It is only recently that various advanced yet very usable libraries have been published @dppalomargithub @riskfolio @skfolio. However, each has its own strengths, weaknesses, scope, and idiosyncrasies. That is no different from #po, but it is our hope that this library serves as a unifying framework for the various techniques and methods available in the field, while also eventually providing a simple and intuitive interface for users, with advanced features for experts.

There also exist myriad optimisation methods, pre-filtering, distribution and moment estimators, machine learning techniques, validation, and parameter tuning methods that can all be used together to improve the out-of sample performance of a portfolio. To this day, only skfolio @skfolio has succeeded in providing a unified framework for this. #po provides an alternative that is not tied to the scikit-learn @scikit-learn or cvxpy @cvxpy1 @cvxpy2 APIs and ecosystems, as well as providing different functionality and a different architectural philosophy.

@tab1 gives a sense of the scale at the time of writing. The library is actively maintained so its capabilities are constantly expanded and improved. Some have reached maturity, but new features in line with new research, user feedback, my own ideas, and AI assisted reviews are added regularly.

#show table: set par(first-line-indent: 0em, spacing: 0.6em, leading: 0.6em)
#show table: set text(size: 9pt)

#figure(
  // placement: auto,
  caption: [Concrete estimator and algorithm types per family, counted from the library's own type tree; the Result types they return are not counted. The 20 optimisers are 6 naïve, 5 #jump based, 4 clustering, 2 ensemble, 2 finite allocators, and 1 hindsight comparator. One of the naïve optimisers holds any of the 22 online portfolio selection rules, and 24 named constructors fix a rule to its published form. Composition multiplies them: any covariance estimator feeds any distance, any distance feeds any clustering or network algorithm, and any of those feeds any optimiser that asks for one.],
  table(
    columns: (0.5fr, 0.2fr, 2.3fr),
    align: (left + top, center + top, left + top),
    stroke: none,
    table.hline(),
    table.header([*Family*], [*Types*], [*A few of them*]),
    table.hline(stroke: 0.5pt),
    [Risk measures], [57],
    [Variance, CVaR, EVaR, RLVaR, their range and drawdown twins, ordered weights, high order moments],
    [Covariance, correlation], [22],
    [Gerber, Smyth-Broby, Kendall, Spearman, mutual information, distance covariance, lower tail dependence, geodesic shrinkage, regime adjusted exponential weighting],
    [Expected returns], [12],
    [Equilibrium, excess, median, windowed, exponentially weighted, price level forecasts, and shrunk towards three targets],
    [Prior statistics], [13],
    [Empirical, time series and cross-sectional factor, high order, four Black-Litterman variants, entropy pooling, opinion pooling],
    [Cross-sectional factor model], [27],
    [Descriptor estimators behind 37 named descriptors, such as book to price or momentum, factor exposures including observed and currency factors, and return forecasts],
    [Phylogeny algorithms], [14],
    [Distances and distances of distances, minimum spanning trees, Direct Bubble Hierarchical Trees, hierarchical clustering, k-means, eight centrality measures],
    [Uncertainty sets], [5],
    [Normal distribution, ARCH bootstrapped, delta, characteristic, and orthogonal to a factor model, in box, ellipsoidal, norm ball or compact form],
    [Calibration rules], [14],
    [Data-driven heuristics for significance levels, deformation parameters, ambiguity radii, norm ceilings, factor mean blends],
    [Constraints, costs, penalties], [18],
    [Weight bounds, three budget forms, linear equations, thresholds, risk budgets, factor exposure, two phylogeny forms, centrality, fees, turnover, two allocation sets of an online rule, and L2 or Lp weight penalties. Custom constraints and objectives are an abstract type to subtype, so they are not counted],
    [Optimisers], [20],
    [Equal weighted, inverse volatility, online portfolio selection, best constant rebalanced portfolio, mean risk, near optimal centering, risk budgeting, HRP, HERC, Schur, nested clustering, stacking, discrete allocation],
    [Online portfolio selection rules], [22],
    [Buy and hold, constant rebalancing, mirror descent, Newton step, passive aggressive and confidence weighted mean reversion, pattern matching, expert mixtures],
    [Cross-validation and tuning], [8],
    [K-fold, combinatorial, walk-forward by index or by date, hindsight split, multiple randomised, grid and randomised search. Every sequential scheme can also step online],
    [Preprocessing and pipeline], [9],
    [Price ingestion, prices to returns, missing data filter, price gap fill, three asset selectors, train and test split, pipelines that bundle end-to-end workflows and can be cross-validated and tuned as a single unit],
    table.hline(),
  ),
)<tab1>

= Design and implementation

#po is built with modularity and extensibility in mind, whilst guaranteeing performance, robustness, and correctness. With Julia, and careful software architecture design, we need not make compromises in any of these.

== Basic example

We can demonstrate a simple improvement over the Markowitz model by adding an L2 regularisation term to the optimisation problem, which shrinks the weights towards the equal weighted portfolio and is known to improve out of sample performance @demiguel2009norm.

$
  & min_bold(w) quad && bold(w)^top bold(Sigma) bold(w) + lambda ||bold(w)||_2^2 \
  & upright(s.t.)    && bold(w)^top bold(mu) >= mu_i \
  &                  && bold(w)^top bold(1) = 1 \
  &                  && 0 <= w_j <= 1 quad forall j != "AAPL" \
  &                  && 0 <= w_("AAPL") <= 0.2
$<eq1>

The problem is solved once per return level $mu_i$, for $N = 100$ levels evenly spaced between the minimum and maximum attainable returns, which traces the efficient frontier of @fig1.

#read-julia-output(json("main-jlyfish.json"))
// No #jl-pkg. This hidden cell reuses the environment that builds the docs,
// reached by a relative path. docs/Project.toml already carries every package
// the listing needs, and its manifest already points PortfolioOptimisers at
// this checkout, so the paper and the documentation agree on versions by
// construction.
#jl(code: false, result: false, stdout: false, logs: false, ```julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
```)

#{
  show figure: set block(breakable: true)
  set figure(gap: 1.5em)
  figure(
    // placement: auto,
    caption: [End-to-end example of a regularised Markowitz model. The listing is not a transcript: it is executed by the build, so it cannot drift away from the library.],
    jl(code: true, result: false, stdout: false, logs: false, recompute: false, ```julia
    # Import packages.
    using PortfolioOptimisers, CSV, TimeSeries, Clarabel, StatsPlots, GraphRecipes
    # Load prices and turn them into returns.
    X = TimeArray(CSV.File(joinpath(@__DIR__, "../../examples/SP500.csv.gz"));
                  timestamp = :Date)
    rd = prices_to_returns(X)
    rd_train, rd_test = train_test_split(rd; test_size = 0.2)
    # Mean risk optimisation, which minimises risk by default.
    mr = MR(;
      # Variance, written directly as a quadratic expression.
      r = Variance(; alg = QuadRiskExpr()),
      opt = JuMPOpt(;
        # Solvers are tried in order until one of them succeeds.
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     settings = Dict("verbose" => false)),
        # An estimator, so the bounds are built from the data: every asset
        # is capped at 1 and floored at 0, except AAPL, capped at 0.2.
        wb = WBE(; ub = "AAPL" => 0.2),
        # Maps asset names to their columns, and names to sets of assets.
        # Constraint estimators and some priors are built against this.
        sets = UniverseSets(; dict = Dict("nx" => rd.nx)),
        # Squared L2 penalty on the weights, lambda = 1e-4.
        l2 = L2Reg(; val = 0.0001, alg = QuadRiskExpr()),
        # Sweep 100 return levels: one solve each, one efficient frontier.
        ret = ArithmeticReturn(;
                settings = JuMPReturnsSettings(; lb = Frontier(; N = 100)))
      ) # opt
    ) # mr
    # Fit on the training set, then score both sets.
    res = optimise(mr, rd_train)
    pred_train = predict(res, rd_train)
    pred_test = predict(res, rd_test)
    # Scenario based standard deviation, as a second order cone expression.
    r = SCM(; alg = SOCRiskExpr())
    plt = plot_measures(pred_train; x = r, label = "Training", zcolor = nothing)
    plt = plot_measures(pred_test; x = r, plt = plt, label = "Test", zcolor = nothing,
                        markercolor = :red, ylabel = "Mean Return",
                        xlabel = "Standard Deviation")
    savefig(plt, joinpath(@__DIR__, "fig1.svg"))
    nothing
    ```),
  )
}<code1>

#figure(
  image("fig1.svg", width: 65%),
  caption: [Train and test efficient frontier for an L2 regularised Markowitz model with a scenario-based standard deviation risk measure. The training set does much better than the test set, this is why other portfolio optimisation modalities have been invented.],
)<fig1>

== Data flow

The library is configured to allow for different workflows. There is no rigid structure to the data flow. It is possible to use the library as simply as shown above or to build extremely complex workflows either manually, or using a pipeline. The functionality is grouped into three main stages, which break down into a hierarchy of standalone processes that can be used in isolation or as part of a larger workflow.

+ Preprocessing: data cleanup, filtering, and transformation.
  + Ingestion: joins the asset, factor and benchmark tables, collapses the frequency, and records the listing span of every asset. It runs once over the whole history, because the folds of a cross-validation are cut on the dates it produces.
  + Gap treatment: fills a missing price with something tractable, or drops the assets and observations with too many gaps.
  + Preselection: uses various non-optimisation methods to preselect assets. Can operate at the price or returns stage.
  + Returns computation: computes returns, and records which assets are listed and which can be estimated at every observation.
+ Processing: these can be provided precomputed, which is valid for certain optimisers and standalone optimisations, or computed inside an optimisation, which is required for certain meta optimisers and for cross-validation. The only requirement to perform an optimisation is a prior, and for certain optimisers, a clustering.
  + Prior statistics: standalone moment estimation, integrative returns distribution estimation, and adjustments to the moment estimations.
  + Phylogeny: uncovers the relational structure of the asset universe through clustering and network analysis.
  + Constraint generation: lets users define constraints, some of which consume prior and phylogeny statistics.
  + Uncertainty sets: used for robust optimisation, they ameliorate the sensitivity to estimation errors.
  + Optimisation: consumes or internally carries out all the aforementioned steps and produces a result carrying every computed quantity.
+ Postprocessing: performance reporting, attribution, forecast evaluation, regret, and visualisation.

The preprocessing and processing steps can be bundled into a single pipeline object, which is itself an estimator. That pipeline can be cross-validated and tuned as a unit, so the preprocessing and processing stacks are tuned together rather than one at a time.

== Point-in-time data

Real universes change. Assets list late, delist, and are suspended, and a backtest that drops them from the whole sample, or holds them after they are gone, suffers from survivorship and look-ahead bias. #po never deletes a gap. The position of a gap in a price column says whether the asset was in the universe: a leading run means it had not yet listed, a trailing run that it delisted, and an interior run that it was suspended. The returns then carry an asset panel whose two masks say which assets are listed at each observation, and which of those have a finite return.

The estimators downstream read those masks. A moment estimator fits on the assets it can estimate over its window, and an optional coverage policy replaces that rule with available-case estimation, in which each entry of a covariance is fitted on the observations its two assets share. Every optimiser fits on the assets it can trade and gives every other asset a weight of exactly zero, so each fold of a walk-forward runs on the universe of that fold. The fees follow the same calendar: proportional and turnover fees are charged on every observation, fixed fees once per holding period, and a liquidation charge on a position whose asset leaves the universe.

The same panel holds the features of the assets, such as a market capitalisation, a sector, or a graded classification, as named numeric, categorical or tensor fields. Feature distances and the cross-sectional factor model below read them from there. Each reader of the returns, such as a prior, a clustering, or an online rule, states whether it fits on the returns themselves, on their excess over a benchmark, or on the returns of the prior.

== Moments

#po comes with a large number of moment estimation and prior statistics methods. It leverages #statsbase's covariance estimator definitions and API to allow interoperability with the Julia ecosystem, for example shrinkage is supported by composing with #link("https://github.com/mateuszbaran/CovarianceEstimation.jl")[CovarianceEstimation.jl]. They are highly composable with each other, as well as with different priors, and uncertainty sets.

- Full or downside co-moments.
- Weighted, unweighted.
- Shrunk, excess, equilibrium returns.
- Geodesic shrinkage @musolas2021.
- Exponentially weighted, and regime adjusted with an exact correction of the bias of the regime statistic.
- Probability-theoretic, including the Gerber statistic and its descendants @gerber @smyth2022enhanced @gerber2025squeezing.
- Denoising and detoning against the Marchenko-Pastur spectrum @mpdist @lopezdepradobook.
- J-sparse, through information filtering networks @J_LoGo.
- Regression-based.
- Rank based.
- Coskewness and cokurtosis.
- Expected returns forecast from the price levels, such as moving averages, spatial medians, and trend switches, as the online portfolio selection features use them @lihoi2014.

== Distance and phylogeny statistics

These bundle clustering and network analysis pipelines to explore the relational structure of the asset universe, and to derive insights that constraints and optimisers can consume. Clustering and network analysis are different, but they tackle similar issues from complementary angles, and the same analysis techniques exploit the structure that each uncovers. As such we choose to categorise them under the same umbrella. The distance matrices include the first and second order distances defined in @lopezdepradobook, and are compatible with every single covariance estimator that follows the #statsbase API. The clustering and network analysis methods include:

- Distance, distance-of-distance and similarity matrices.
- Feature distances, which measure the assets by their exposures, memberships, factor loadings or graph neighbourhoods rather than by their returns.
- Hierarchical agglomerative clustering, with an exact optimal leaf order @barjoseph2001.
- Direct bubble hierarchy trees @DBHTs @NHPG.
- K-means clustering.
- Minimum spanning trees.
- Planar maximally filtered graphs @PMFG.
- Centrality and community detection.

== Prior statistics

A prior is the returns distribution the optimisation actually sees, and it is the one component every optimiser requires. They fall under three categories:

- Frequentist: empirical, timeseries factor models for low and high moments, and cross-sectional (fundamental analysis) factor models.
- Bayesian: Black-Litterman @blacklitterman1992, in vanilla, factor, Bayesian and augmented forms.
- Information-theoretic: entropy pooling @meucci2008, including views on Conditional Value at Risk, Entropic Value at Risk @EPTail and Relativistic Value at Risk @EPRLVaR, and linear or logarithmic opinion pooling @genest1986.

Again these can be composed with each other to form more sophisticated priors.

== Risk measures

A risk measure is an object, not a hard-coded branch. Any optimiser that takes one takes a vector of them, scalarised into a single expression by a sum, a maximum, a minimum, or a log-sum-exp. Every prior-derived quantity a measure needs, such as the expected returns, the covariance, the coskewness or the cokurtosis, may be given as a value or as the estimator that computes it. In the second case it is refitted against the optimisation's own prior, once per cross-validation fold and once per subset of a meta-optimiser, which a pasted matrix cannot do.

- Dispersion: variance, standard deviation, low order moments, range, Brownian distance variance, and a variance taken over an uncertainty set.
- Tail: value at risk, conditional value at risk @rockafellar2000, entropic value at risk @ahmadijavid2012, power norm and relativistic value at risk @cajasbook, and worst realisation. Most have a range twin, and the conditional ones have distributionally robust variants @drcvar.
- Drawdown: the same tail family applied to the drawdown series @chekhlov2005, plus average drawdown, maximum drawdown, and the ulcer index.
- Ordered weights: Gini mean difference, tail Gini, and L-moments of arbitrary order, as disciplined convex programmes @owa1 @owa2 @owa3, including an exact formulation through a sorting network whose number of constraints grows as $T log^2 T$ in the number of observations @batcher1968.
- High order: coskewness and cokurtosis, full or semi, standardised or not.
- Relative: tracking error, risk tracking error and turnover, which measure deviation from a reference portfolio rather than from zero.

Twelve further measures are exclusive to the hierarchical optimisers, which can use measures that are not convex in the weights because they never place them in a solver.

The scalar parameters need not be numbers either. The tail probability or deformation parameter of a measure, the ambiguity radius of a measure or of a weight penalty, and a ceiling on the norm of the weights may each be a rule that the optimisation evaluates on its own prior, so it is recalibrated in every fold. The rules compute a significance level from the length of the sample, a deformation parameter from an estimate of the tail index, an ambiguity radius from the sampling error of the empirical measure @drcvar, and a norm ceiling from a floor on the effective number of assets.

== Uncertainty sets

Uncertainty sets are used for robust optimisation, where the parameters are only known to lie within a set. They ameliorate the sensitivity to estimation errors, and they currently cover the expected returns and the covariance.

- Box: the simplest set, defined by a lower and upper bound for each parameter @tutuncu2004.
- Ellipsoidal: defined by a radius and a covariance matrix, forming an ellipsoid in the parameter space @goldfarb2003.
- Characteristic: applies a specially callibrated L1 uncertainty set to the expected returns, which recovers the quintile portfolio as a special case @quintile. This supports any characteristics vector, as long as there exists an overload for `Statistics.mean` for that vector.
- Norm ball: the image of a norm ball under a map of any rank, so a set that is flat in the directions a factor model does not span needs no full rank shape matrix.
- Compact: holds the worst-case variance of a covariance set as a quadratic penalty on the weights, so the problem stays a second order cone programme with no semidefinite constraint.

The bounds themselves come from a normal distribution approximation, from a delta method, or from an ARCH bootstrap of the estimator. A further estimator reads no returns at all: it takes the factor model of the optimisation's prior, and limits both sets to the directions that the factors do not span.

== Optimisers

#po provides a large number of optimisers, in six families.

+ Naïve: equal weighted, inverse volatility, random weighted, and previous weights, which holds the weights of the last fold. Speed and robustness, but not necessarily optimality. Interesting as sub-optimisers to more complex optimisers, as fallbacks, and as the benchmark that is hardest to beat out of sample @demiguel2009naive. The family also holds the best constant rebalanced portfolio in hindsight, computed from Cover's fixed point without a solver @cover1991.
+ Online portfolio selection: rules that move the weights after every new price relative, most of them without a solver @lihoi2014. A batch fit makes one causal pass over its window, and an online fit continues from the state the last one reached, so both give one path. The rules cover:
  + Benchmarks: buy and hold, and constant rebalancing.
  + Follow the winner: mirror descent in a Euclidean, entropic, Tsallis or log-barrier geometry @helmbold1998, the Newton step @agarwal2006, and the universal portfolio @cover1991, with learning-rate schedules, optimistic gradient predictions, and any risk measure as the loss.
  + Follow the loser: passive aggressive @li2012pamr, moving average @li2015olmar, robust median @huang2016 and confidence weighted @li2013cwmr mean reversion, anticorrelation @borodin2004, and transaction cost optimisation @li2018tco.
  + Pattern matching: solves for the leader over the past rows that resemble the present ones @gyorfi2006, through a solver free fixed point or any #jump based optimiser.
  + Meta-learning: mixes any of the above as experts, including mixtures built for changing environments @zhang2018ader.

  A prediction charges the fees of the head, with each trade measured against the held book. The optimiser keeps the weights of every rule inside an allocation set, either weight bounds that project in closed form or the constraints of a #jump based optimiser. A missing return is read as cash, so a rule never trades on a return that did not happen.
+ #jump based @JuMP: the most flexible and powerful, but they require a solver, such as Clarabel @clarabel, and are typically slower than first-order optimisers. They support a wide range of constraints and objectives, and they use conic reformulations throughout. The family covers mean risk, near optimal centering, risk budgeting @maillard2010 and its relaxed form, and factor risk contribution.
+ Hierarchical: these use the relational structure of the universe to compute the risk of each group and sub-group, producing a diversified portfolio that encodes the relational structure as well as the risk characteristics of each group. They are typically faster than #jump based optimisers and support fewer constraints, but they are very good for large universes. The family covers hierarchical risk parity @hrp2016, hierarchical equal risk contribution @raffinot2017, and Schur complementary allocation @cotton2024, which interpolates between hierarchical risk parity and minimum variance.
+ Meta-optimisers: these consume other optimisers and combine their results. They are typically slower, but can be very effective. The family covers nested clustering @lopezdepradobook, stacking @wolpert1992, and subset resampling @shen2017.
+ Finite allocators: these do not optimise portfolios in the traditional sense. The user provides a finite cash amount and the asset prices, and the allocator computes the best portfolio attainable with that cash. They run after the others have produced a result. A long-short book either spends the proceeds of its short sales, or caps its gross exposure at a collateral amount.

== Constraints, objectives, and penalties

There is a huge variety of constraints. Every optimiser supports weight bounds, but most of the richness is exclusive to #jump based estimators. Constraints are built by estimators rather than written as matrices, so a constraint is stated once and rebuilt against whatever data it meets. Linear constraints can be written as equations in near-natural language, for instance `"AAPL <= 0.1"`, `"MSFT >= AMD"`, or `"tech >= 0.15"` for a set of assets declared in the universe sets.

+ Weight bounds and budgets: the maximum and minimum weights, and the total value of the weights. Together they express leveraged, dollar-neutral, and long-short portfolios, and they integrate with the finite allocators, which adjust the available cash to the budget.
+ Linear constraints: used for relative weights, group exposures, risk contributions, and relational structure.
+ Cardinality constraints: sparsity, buy-in thresholds, and inclusion or exclusion rules, at the level of assets or of sets of assets. They are implemented as mixed integer linear constraints.
+ Fees and turnover: long and short proportional fees, fixed fees, and turnover fees, which penalise the returns and therefore, indirectly, the risk measures and the objective. The fixed fees are charged once per holding period, or amortised over it.
+ Weight penalties: L-norm penalties on the objective, or hard limits on the norm of the weights, which regulate sparsity and robustness.
+ Tracking: limits on how far a portfolio may drift from a reference, in weights or in risk.
+ Objectives: minimum risk, maximum return, maximum risk-adjusted return, and maximum utility, where the risk term may itself be a scalarised vector of risk measures.
+ Risk upper and return lower bounds: applicable simultaneously, and bounded by a scalar, a precomputed vector or range, or a frontier object that lets the optimiser compute the bounds on the fly. Bounding more than one quantity by a range produces a grid, and therefore a Pareto surface returned as a vector of results.
+ Custom constraints and objectives: user-defined terms that receive the optimiser's own context, and enter the objective as penalty contributions.
+ Time dependent: every constraint and every non-finite optimiser can be wrapped so that it varies across cross-validation folds. The wrapper takes a predefined list, or a function of the fold's context and the previous portfolio's weights, which allows a strategy to change its own constraints as it walks forward.

== Incremental estimation

Some statistics have an exact update, in which the new value follows from a running state and one new observation. The sample mean, the variance, the full covariance, the coskewness and the cokurtosis are among them, and their estimators fold each new observation into that state rather than read the entire sample again. Every other estimator keeps the observations it has seen, optionally capped to a rolling window, and runs its batch computation over them.

When used, incremental estimators allow the strategy to update its statistics with new data as it arrives, without recomputing on the entire history, falling back to batch computation for estimators that do not support exact incremental updates. Users only need to declare an estimator as incremental, the library uses multiple dispatch to select the appropriate update or recomputation method automatically.

== Cross-validation, hyperparameter tuning, and pipelines

The library comes equipped with a pipeline framework that can run an entire strategy end to end, and with a cross-validation framework that evaluates the out of sample performance of an optimisation or of a whole pipeline. The schemes include k-fold, index and date based walk-forward with purging, multiple randomised splits, and combinatorial splits that produce many backtest paths rather than one @lopezdepradoml. The same machinery drives the nested clustering and stacking meta-optimisers, which need out of sample predictions from their inner estimators before the outer optimisation can run.

Every sequential scheme also has an online form, in which each fold updates one estimator with its new observations instead of refitting it, and a finished run can be resumed over a longer history. A hindsight split trains each fold on the row it tests, which yields the per-row comparators of dynamic regret. A fold can also be scored on the weights a fund actually holds, which drift with the returns of their assets between trades, and those held weights can be passed on to the next fold.

The hyperparameter tuning framework is built on top of the cross-validation framework. Users address the steps of a pipeline by index or by name, and the fields to tune by name. Grid search takes grids of values, and random search takes grids, distributions from #link("https://github.com/JuliaStats/Distributions.jl")[Distributions.jl] @distributionsjl, or both.

== Evaluation and attribution

The postprocessing stage scores what the processing stage produced. The forecast evaluations also accept bare arrays, so they score a forecast from outside the library in the same way as one the library made.

- Performance: annualised return and volatility, the Sharpe, Sortino and Calmar ratios, the maximum drawdown and the CVaR, and, against a benchmark, the excess return, tracking error, information ratio and turnover. The probabilistic and deflated Sharpe ratios ask whether a Sharpe ratio shows skill, or the luck of a search over many trials @sharpe_stderr @bailey2014deflated.
- Attribution: the predicted or realised volatility and mean return of a portfolio, split into systematic, idiosyncratic and unexplained parts, by factor, by factor family and by asset, as well as Brinson attribution.
- Forecast evaluation: the information coefficient, coverage, quantile spread, decay and calibration of a return forecast out of sample, with a test of the mean information coefficient that pairs a fixed-bandwidth long-run variance with a wild bootstrap, so the overlap of the forward windows does not inflate its significance. Covariance forecasts are scored against the returns realised after them, step by step through a walk-forward.
- Regret: the log-wealth regret of a strategy against any comparator, with a Newey-West test of equal expected log growth @neweywest1987. A comparator fitted in hindsight on each row gives the dynamic regret, and the best sequence of allocations under a budget on its path length gives the regret against the budgeted comparator of @zinkevich2003.
- These include textual printouts as well as plots and charts.

== Design philosophy

#po is implemented in Julia @Julia-2017, using the #jump mathematical optimisation embedded language @JuMP, and leverages other well-established Julia libraries for much of its functionality. Aside from a strong commitment to FOSS principles, the library is designed atop four pillars:

+ Well-defined type hierarchies. This lets us take advantage of generic fallbacks, and enables fearless extensibility, even at the user-level thanks to multiple dispatch.
+ Strongly typed, immutable structs. This lets the compiler aggressively optimise code, provides a single source of truth, and enables construction-time data validation that will not expire.
+ Compositional design. This lets us build complex objects from simpler ones, enables code reuse, and aids in extensibility.
+ Defensive programming. This lets us catch errors as early as possible, and provides a clear and consistent interface to the user.

With the exception of covariance estimators, which are defined in #statsbase, every type in the library is a subtype of one of three types:

+ `AbstractEstimator`: these contain all the parameters needed to estimate a particular quantity, and are consumed by the functions that perform the estimation. They are the user-level vocabulary, they nest within each other, and they build more complex estimators.
+ `AbstractResult`: many estimators return more than one quantity, and those are returned in a result type. Estimators are configuration; results are data.
+ `AbstractAlgorithm`: these modify the behaviour of estimators via multiple dispatch. They are not used standalone, but as part of an estimator.

The strict adherence to a well-defined type hierarchy lets developers and users alike extend and modify the library's behaviour without modifying the source code. A new risk measure, estimator, or optimiser can live in a user's own package and still compose with everything described here.

== Future developments

There are several planned features that have not yet been implemented these include:

+ Vine copula based synthetic data prior. This would allow the generation of realistic synthetic datasets that preserve complex dependency structures observed in historical data whilst letting users perform robust stress testing and scenario analysis without relying solely on historical samples.
+ Fixed income and derivative risk models. This would enable the library to handle a broader range of financial instruments, allowing for more comprehensive portfolio optimisation and risk management strategies.
+ Characteristics based portfolio construction. This would allow the library to build portfolios directly from cross-sectional factors rather than returns data.

= High standards of quality in the age of AI

The library has an extensive and growing test suite that as of now has 98% #coverage. It has a fully documented public and private #api, very comprehensive #examples and #user_guide, and an entire toolchain dedicated to automatically finding and reporting regressions, type instabilities, missing documentation, code smells, and other potential issues that unequivocally improve code quality and maintainability as the library evolves. These tools are run both when code is committed---serving as a measure of immediate quality control---but also as part of weekly continuous integration checks which serve to improve the overall code quality and maintainability of the existing codebase. These tools are deterministic in their reporting, so they do not require an AI to interpret, analyse or fix their findings, however we have found frontier models to be particularly good at fixing these measurable, interpretable results.

Even before AI tools became as prevalent and---for frontier models---unambiguously beneficial as they are today, the library defined high standards of quality. These have been greatly expanded and reinforced with automated checks and continuous integration, ensuring that both human and AI contributions maintain the library's high standards. There are also entire workflows that use AI tools to assist in code review; improve testing and documentation; add new features; find and fix bugs; improve the user experience by simplifying interfaces, improving logging and error messages; and overall doing the menial and boring maintenance refactors that unlock new possibilities and fix latent issues before they manifest.

By being so thorough with our standards and automatic checks, we have found great success at integrating AI tools into our development workflow, allowing the library to integrate previously out of scope features, and improving the quality of existing ones. These workflows are continuously refined and expanded to maximize the benefits of AI assistance without losing our unwavering commitment to the highest standards of quality. A cursory glance at the issue tracker will reveal the lengths to which we go to ensure additions meet our quality standards, but also how much effort is put into maintaining and improving the existing codebase over time.

= Availability

#po is registered under an MIT license in the Julia General registry, so `Pkg.add("PortfolioOptimisers")` is all that is needed. The listings in this paper are executed when it is built, so the paper is checked by the same standard as the rest of the project. The library is under active development, the issue tracker keeps ideas and roadmaps for new features, maintenance tasks, bug reports, etc. contributions are very much appreciated and welcome.

We hope readers will find in #po an interesting, valuable, and trustworthy tool for their needs.

= AI use disclosure

AI was used in the validation of the counts provided in @tab1, in ensuring the paper covers a comprehensive and accurate representation of the library's features, capabilities, and development practices, as well as checking for gaps in the literature. It was not used to write any part of this article.

#show bibliography: set par(first-line-indent: 0em, spacing: 0.75em, leading: 0.6em)
#show bibliography: set text(size: 9pt)
#bibliography("refs.bib")
