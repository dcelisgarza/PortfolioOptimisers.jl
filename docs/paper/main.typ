#import "@preview/clean-math-paper:0.2.7": *
#import "@preview/jlyfish:0.1.0": *

// ===========================================================================
// BUILDING THIS PAPER
//
// The listing in figure 1 is executed at build time by jlyfish, so it cannot
// drift away from the library. Two halves must run: TypstJlyfish evaluates the
// Julia and writes main-jlyfish.json, and typst renders the document.
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
#let po = link(repo)[PortfolioOptimisers.jl]
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
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
  abstract: [Portfolio optimisation is the science of either: _1)_ Minimising risk whilst keeping returns to acceptable levels. _2)_ Maximising returns whilst keeping risk to acceptable levels. To some definition of acceptable, and with any number of additional constraints available to the optimisation type. There exist myriad statistical, pre- and post-processing, optimisations, and constraints that allow one to explore an extensive landscape of "optimal" portfolios. PortfolioOptimisers.jl is an attempt at providing as many of these as possible under a single banner and making it accessible to all: 57 risk measures, 18 covariance estimators, 12 prior estimators and 16 optimisers, which compose with each other rather than sit in separate silos, and which a single cross-validation framework can tune as one object. We make extensive use of Julia's type system, module extensions, and multiple dispatch to simplify development and maintenance, while keeping robustness, testability, and usability high.],
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

@tab1 gives a sense of the scale. What follows is a tour rather than a manual: the documentation carries the detail, and the point of this paper is to show that the detail is there.

#show table: set par(first-line-indent: 0em, spacing: 0.6em, leading: 0.6em)
#show table: set text(size: 9pt)

#figure(
  placement: auto,
  caption: [Concrete estimator and algorithm types per family, counted from the library's own type tree; the Result types they return are not counted. The 16 optimisers are 3 naïve, 5 #jump based, 4 clustering, 2 ensemble, and 2 finite allocators. Composition multiplies them: any covariance estimator feeds any distance, any distance feeds any clustering or network algorithm, and any of those feeds any optimiser that asks for one.],
  table(
    columns: (auto, auto, 1fr),
    align: (left + top, center + top, left + top),
    stroke: none,
    table.hline(),
    table.header([*Family*], [*Types*], [*A few of them*]),
    table.hline(stroke: 0.5pt),
    [Risk measures], [57],
    [Variance, CVaR, EVaR, RLVaR, their range and drawdown twins, ordered weights, high order moments],
    [Covariance and correlation], [18],
    [Gerber, Smyth-Broby, Kendall, Spearman, mutual information, distance covariance, lower tail dependence],
    [Expected returns], [9],
    [Equilibrium, excess, median, windowed, and shrunk towards three targets],
    [Prior statistics], [12],
    [Empirical, factor, high order, four Black-Litterman variants, entropy pooling, opinion pooling],
    [Phylogeny algorithms], [14],
    [Three minimum spanning trees, DBHT, hierarchical clustering, k-means, eight centrality measures],
    [Uncertainty sets], [4],
    [Normal, ARCH bootstrapped, delta and characteristic, in box or ellipsoidal form],
    [Constraints, costs, and penalties], [16],
    [Weight bounds, three budget forms, linear equations, thresholds, risk budgets, factor exposure, two phylogeny forms, centrality, fees, turnover, and L2 or Lp weight penalties. Custom constraints and objectives are an abstract type to subtype, so they are not counted],
    [Optimisers], [16],
    [Equal weighted, inverse volatility, mean risk, near optimal centering, risk budgeting, HRP, HERC, Schur, nested clustering, stacking, discrete allocation],
    [Cross-validation and tuning], [7],
    [K-fold, combinatorial, walk-forward by index or by date, multiple randomised, grid and randomised search],
    [Preprocessing and pipeline], [8],
    [Prices to returns, missing data filter, imputer, three asset selectors, train and test split, pipelines that bundle end-to-end workflows and can be cross-validated and tuned as a single unit],
    table.hline(),
  ),
)<tab1>

= Design and implementation

== Basic example

#po is built with modularity and extensibility in mind. We can demonstrate a simple improvement over the Markowitz model by adding an L2 regularisation term to the optimisation problem, which shrinks the weights towards the equal weighted portfolio and is known to improve out of sample performance @demiguel2009norm.

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
  + Transformation: fills out pathological data with something tractable.
  + Preselection: uses various non-optimisation methods to preselect assets. Can operate at the price or returns stage.
  + Returns computation: computes returns, and can impute, drop, or fill missing data.
+ Processing: these can be provided precomputed, which is valid for certain optimisers and standalone optimisations, or computed inside an optimisation, which is required for certain meta optimisers and for cross-validation. The only requirement to perform an optimisation is a prior, and for certain optimisers, a clustering.
  + Prior statistics: standalone moment estimation, integrative returns distribution estimation, and adjustments to the moment estimations.
  + Phylogeny: uncovers the relational structure of the asset universe through clustering and network analysis.
  + Constraint generation: lets users define constraints, some of which consume prior and phylogeny statistics.
  + Uncertainty sets: used for robust optimisation, they ameliorate the sensitivity to estimation errors.
  + Optimisation: consumes or internally carries out all the aforementioned steps and produces a result carrying every computed quantity.
+ Postprocessing: reporting and visualisation, which is by far the least mature aspect of the library.

The preprocessing and processing steps can be bundled into a single pipeline object, which is itself an estimator. That pipeline can be cross-validated and tuned as a unit, so the preprocessing and processing stacks are tuned together rather than one at a time.

== Moments

#po comes with a large number of moment estimation and prior statistics methods. It leverages #statsbase's covariance estimator definitions and API to allow interoperability with the Julia ecosystem, for example by composing with #link("https://github.com/mateuszbaran/CovarianceEstimation.jl")[CovarianceEstimation.jl]. They can be combined in many ways: it is possible to compute the expected returns using a specific covariance method and vice-versa.

- Full or downside co-moments.
- Weighted, unweighted.
- Shrinkage @ledoit2004.
- Probability-theoretic, including the Gerber statistic and its descendants @gerber @smyth2022enhanced @gerber2025squeezing.
- Denoising and detoning against the Marchenko-Pastur spectrum @mpdist @lopezdepradobook.
- J-sparse, through information filtering networks @J_LoGo.
- Regression-based.
- Rank based.
- Coskewness and cokurtosis.

== Distance and phylogeny statistics

These bundle clustering and network analysis pipelines to explore the relational structure of the asset universe, and to derive insights that constraints and optimisers can consume. Clustering and network analysis are different, but they tackle similar issues from complementary angles, and the same analysis techniques exploit the structure that each uncovers. As such we choose to categorise them under the same umbrella. The distance matrices include the first and second order distances defined in @lopezdepradobook, and are compatible with every single covariance estimator that follows the #statsbase API. The clustering and network analysis methods include:

- Distance, distance-of-distance and similarity matrices.
- Hierarchical agglomerative clustering.
- Direct bubble hierarchy trees @DBHTs @NHPG.
- K-means clustering.
- Minimum spanning trees.
- Planar maximally filtered graphs @PMFG.
- Centrality and community detection.

== Prior statistics

A prior is the returns distribution the optimisation actually sees, and it is the one component every optimiser requires. They fall under three categories:

- Frequentist: empirical and timeseries factor models, for low and high moments.
- Bayesian: Black-Litterman @blacklitterman1992, in vanilla, factor, Bayesian and augmented forms.
- Information-theoretic: entropy pooling @meucci2008, including views on CVaR and EVaR @EPTail, and linear or logarithmic opinion pooling @genest1986.

== Risk measures

A risk measure is an object, not a hard-coded branch. Any optimiser that takes one takes a vector of them, scalarised into a single expression by a sum, a maximum, a minimum, or a log-sum-exp. Every prior-derived quantity a measure needs, such as the expected returns, the covariance, the coskewness or the cokurtosis, may be given as a value or as the estimator that computes it. In the second case it is refitted against the optimisation's own prior, once per cross-validation fold and once per subset of a meta-optimiser, which a pasted matrix cannot do.

- Dispersion: variance, standard deviation, low order moments, range, Brownian distance variance, and a variance taken over an uncertainty set.
- Tail: value at risk, conditional value at risk @rockafellar2000, entropic value at risk @ahmadijavid2012, power norm and relativistic value at risk @cajasbook, and worst realisation. Most have a range twin, and the conditional ones have distributionally robust variants @drcvar.
- Drawdown: the same tail family applied to the drawdown series @chekhlov2005, plus average drawdown, maximum drawdown, and the ulcer index.
- Ordered weights: Gini mean difference, tail Gini, and L-moments of arbitrary order, as disciplined convex programmes @owa1 @owa2 @owa3.
- High order: coskewness and cokurtosis, full or semi, standardised or not.
- Relative: tracking error, risk tracking error and turnover, which measure deviation from a reference portfolio rather than from zero.

Twelve further measures are exclusive to the hierarchical optimisers, which can use measures that are not convex in the weights because they never place them in a solver.

== Uncertainty sets

Uncertainty sets are used for robust optimisation, where the parameters are only known to lie within a set. They ameliorate the sensitivity to estimation errors, and they currently cover the expected returns and the covariance.

- Box: the simplest set, defined by a lower and upper bound for each parameter @tutuncu2004.
- Ellipsoidal: defined by a radius and a covariance matrix, forming an ellipsoid in the parameter space @goldfarb2003.
- Characteristic: applies an L1 uncertainty set to the expected returns, which recovers the quintile portfolio as a special case @quintile. This will eventually be generalised to any characteristic vector.

The bounds themselves come from a normal approximation, from a delta method, or from an ARCH bootstrap of the estimator.

== Optimisers

#po provides a large number of optimisers, in five families.

+ Naïve: equal weighted, inverse volatility, and random weighted. Speed and robustness, but not necessarily optimality. Interesting as sub-optimisers to more complex optimisers, as fallbacks, and as the benchmark that is hardest to beat out of sample @demiguel2009naive.
+ #jump based @JuMP: the most flexible and powerful, but they require a solver, such as Clarabel @clarabel, and are typically slower than first-order optimisers. They support a wide range of constraints and objectives, and they use conic reformulations throughout. The family covers mean risk, near optimal centering, risk budgeting @maillard2010 and its relaxed form, and factor risk contribution.
+ Hierarchical: these use the relational structure of the universe to compute the risk of each group and sub-group, producing a diversified portfolio that encodes the relational structure as well as the risk characteristics of each group. They are typically faster than #jump based optimisers and support fewer constraints, but they are very good for large universes. The family covers hierarchical risk parity @hrp2016, hierarchical equal risk contribution @raffinot2017, and Schur complementary allocation @cotton2024, which interpolates between hierarchical risk parity and minimum variance.
+ Meta-optimisers: these consume other optimisers and combine their results. They are typically slower, but can be very effective. The family covers nested clustering @lopezdepradobook, stacking @wolpert1992, and subset resampling @shen2017.
+ Finite allocators: these do not optimise portfolios in the traditional sense. The user provides a finite cash amount and the asset prices, and the allocator computes the best portfolio attainable with that cash. They run after the others have produced a result.

== Constraints, objectives, and penalties

There is a huge variety of constraints. Every optimiser supports weight bounds, but most of the richness is exclusive to #jump based estimators. Constraints are built by estimators rather than written as matrices, so a constraint is stated once and rebuilt against whatever data it meets. Linear constraints can be written as equations in near-natural language, for instance `"AAPL <= 0.1"`, `"MSFT >= AMD"`, or `"tech >= 0.15"` for a set of assets declared in the universe sets.

+ Weight bounds and budgets: the maximum and minimum weights, and the total value of the weights. Together they express leveraged, dollar-neutral, and long-short portfolios, and they integrate with the finite allocators, which adjust the available cash to the budget.
+ Linear constraints: used for relative weights, group exposures, risk contributions, and relational structure.
+ Cardinality constraints: sparsity, buy-in thresholds, and inclusion or exclusion rules, at the level of assets or of sets of assets. They are implemented as mixed integer linear constraints.
+ Fees and turnover: long and short proportional fees, fixed fees, and turnover fees, which penalise the returns and therefore, indirectly, the risk measures and the objective.
+ Weight penalties: L-norm penalties on the objective, or hard limits on the norm of the weights, which regulate sparsity and robustness.
+ Tracking: limits on how far a portfolio may drift from a reference, in weights or in risk.
+ Objectives: minimum risk, maximum return, maximum risk-adjusted return, and maximum utility, where the risk term may itself be a scalarised vector of risk measures.
+ Risk upper and return lower bounds: applicable simultaneously, and bounded by a scalar, a precomputed vector or range, or a frontier object that lets the optimiser compute the bounds on the fly. Bounding more than one quantity by a range produces a grid, and therefore a Pareto surface returned as a vector of results.
+ Custom constraints and objectives: user-defined terms that receive the optimiser's own context, and enter the objective as penalty contributions.
+ Time dependent: every constraint and every non-finite optimiser can be wrapped so that it varies across cross-validation folds. The wrapper takes a predefined list, or a function of the fold's context and the previous portfolio's weights, which allows a strategy to change its own constraints as it walks forward.

== Cross-validation, hyperparameter tuning, and pipelines

The library comes equipped with a pipeline framework that can run an entire strategy end to end, and with a cross-validation framework that evaluates the out of sample performance of an optimisation or of a whole pipeline. The schemes include k-fold, index and date based walk-forward with purging, multiple randomised splits, and combinatorial splits that produce many backtest paths rather than one @lopezdepradoml. The same machinery drives the nested clustering and stacking meta-optimisers, which need out of sample predictions from their inner estimators before the outer optimisation can run.

The hyperparameter tuning framework is built on top of the cross-validation framework. Users address the steps of a pipeline by index or by name, and the fields to tune by name. Grid search takes grids of values, and random search takes grids, distributions from #link("https://github.com/JuliaStats/Distributions.jl")[Distributions.jl] @distributionsjl, or both.

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

= Availability

#po is registered in the Julia General registry, so `Pkg.add("PortfolioOptimisers")` is all that is needed, and it is available under an MIT license. It follows common software development best practices:

+ An extensive and high #link("https://app.codecov.io/gh/dcelisgarza/PortfolioOptimisers.jl")[coverage] test suite.
+ Automated documentation builds with a programmatically generated #link("https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/capability_catalogue")[capability catalogue], an introductory #link("https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/user_guide/00_User_Guide")[user guide], #link("https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/examples/00_Examples")[deep examples], and fully documented public and private #link("https://dcelisgarza.github.io/PortfolioOptimisers.jl/stable/api/00_API")[APIs].
+ Miscellaneous code quality checks.

All as part of a GitHub continuous integration pipeline. The listings in this paper are executed when it is built, so the paper is checked by the same standard as the rest of the project. The library is under active development, the roadmap is public, and contributions are welcome. We hope readers will find it a good place to run their own experiments.

#show bibliography: set par(first-line-indent: 0em, spacing: 0.75em, leading: 0.6em)
#show bibliography: set text(size: 9pt)
#bibliography("refs.bib")
