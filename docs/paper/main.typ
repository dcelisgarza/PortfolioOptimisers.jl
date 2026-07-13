#import "@preview/clean-math-paper:0.2.7": *
#import "@preview/jlyfish:0.1.0": *

#let date = datetime.today().display("[month repr:long] [day], [year]")

// Modify some arguments, which can be overwritten in the template call
#page-args.insert("numbering", "1/1")
#text-args-title.insert("size", 2em)
#text-args-title.insert("fill", black)
#text-args-authors.insert("size", 12pt)

#set text(font: "New Computer Modern")
#set enum(numbering: "1.")

#show: template.with(
  title: link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl],
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
  abstract: [Portfolio optimisation is the science of either: _1)_ Minimising risk whilst keeping returns to acceptable levels. _2)_ Maximising returns whilst keeping risk to acceptable levels. To some definition of acceptable, and with any number of additional constraints available to the optimisation type. There exist myriad statistical, pre- and post-processing, optimisations, and constraints that allow one to explore an extensive landscape of "optimal" portfolios. PortfolioOptimisers.jl is an attempt at providing as many of these as possible under a single banner and making it accessible to all. We make extensive use of Julia's type system, module extensions, and multiple dispatch to simplify development and maintenance, while keeping robustness, testability, and usability high.],
  keywords: ("Portfolio Optimisation", "Quantitative Investment", "Conic Optimisation", "Parameter Estimation"),
)
#show heading: set block(above: 1em, below: 1em)
#show heading.where(level: 1): set block(spacing: 1.5em)
#show heading.where(level: 2): set block(spacing: 1.25em)
#set par(first-line-indent: 2em, spacing: 2em, leading: 1.2em)

= Introduction
The field of portfolio optimisation was introduced by Harry Markowitz's seminal 1952 paper "Modern portfolio theory" @markowitz1952. However, the field has grown significantly since then, with many new techniques and methods being developed. The original Markowitz model is highly sensitive to estimation errors in the input parameters, particularly the expected returns. After all, it attempts to summarise the entire distribution of returns in highly compressed summary statistics, both of which can be sensitive to outliers and atypical conditions for the lookback period. Many have studied and addressed the strengths and weaknesses of the Markowitz model @cajasbook @lopezdepradobook @dppalomarbook.

Unfortunately, most implementations live in disparate, unconnected, often propriatary, bespoke codebases, which limit their applicability. It is only recently that various advanced yet very usable libraries have been published @dppalomargithub @riskfolio @skfolio. However, each has its own strengths, weaknesses, scope, and ideosyncrasies. That is no different from #link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl], but it is our hope that this library serves as a unifying framework for the various techniques and methods available in the field, while also eventually providing a simple and intuitive interface for users, with advanced features for experts.

There also exist myriad optimisation methods, pre-filtering, distribution and moment estimators, machine learning techniques, validation, and parameter tuning methods that can all be used together to improve the out-of sample performance of a portfolio. To this day, only @skfolio has succeeded in providing a unified framework for this. #link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl] provides an alternative that is not tied to the scikit-learn @scikit-learn or cvxpy @cvxpy1 @cvxpy2 APIs and ecosystems, as well as providing different functionality and a different architectural philosphy.

= Design and implementation

== Basic usage

#link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl] is built with modularity and extensibility in mind. We can demonstrate a simple improvement over the Markowitz model by using by adding an L2 regularisation term to the optimisation problem.

$
  & min_bold(w) #h(1.5em) && bold(w)^top bold(Sigma) bold(w) + lambda ||bold(w)||_2^2 \
  & upright(s.t.)         && bold(w)^top bold(mu) >= mu_i forall i in {1, ..., N} \
  &                       && bold(w)^top bold(1) = 1 \
  &                       && bold(w) >= 0 \
  &                       && bold(w)_(i != "AAPL") <= 1 \
  &                       && w_("AAPL") <= 0.2
$

// #read-julia-output(json("main-jlyfish.json"))
// #jl-pkg(
//   "Pkg",
//   "Revise",
//   "CSV",
//   "TimeSeries",
//   "PrettyTables",
//   "Clarabel",
//   "StatsPlots",
//   "GraphRecipes",
// )
// #set image(width: 10em)
// #jl(recompute: false,
```julia
# Import packages.
using PortfolioOptimisers, CSV, TimeSeries, Clarabel, StatsPlots, GraphRecipes
# Load data.
X = TimeArray(CSV.File(joinpath(@__DIR__, "examples/SP500.csv.gz"));
              timestamp = :Date)
# Compute returns.
rd = prices_to_returns(X)
# Split into training and test sets.
rd_train, rd_test = train_test_split(rd; test_size = 0.2)
# Define solver. Use clarabel with no verbose output and default settings.
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false))
# Define assets sets used to match constraints to the asset universe.
sets = AssetSets(; dict = Dict("nx" => rd.nx))
# Lower bounds of 0 % for all assets, APPL has upper bounds of 20 %, the other assets default to 100 %.
wb = WBE(; lb = 0, ub = "AAPL" => 0.2)
# L2 norm penalty, lambda × ||w ⋅ w||^2, is added to the objective function. The sign depends on whether the optimisation is a minimisation or maximisation problem. It reduces the overfitting of a model to the training data and improves out-of-sample performance.
l2 = L2Reg(; val = 0.0005, alg = SquaredSOCRiskExpr())
# Define the return with 100 points in the pareto front. Risk measures can take an `lb` parameter which can also be used to define a pareto front. When using the returns, the optimisation must be a minimisation of risk for the efficient frontier to be fully computed. if using risk measures, the optimisations must be a maximisation of return.
ret = ArithmeticReturn(; lb = Frontier(; N = 100))
# Define the optimiser with all the above settings.
opt = JuMPOpt(; slv = slv, wb = wb, sets = sets, l2 = l2, ret = ret)
# Markowitz model.
r = Variance()
# Mean Risk optimisation with all the above settings.
mr = MR(; r = r, opt = opt)
# Perform optimisation on the training set.
res = optimise(mr, rd_train)
# Predict on training data.
pred_train = predict(res, rd_train)
# Predict on test data.
pred_test = predict(res, rd_test)
# Scenario based standard deviation.
r = SCM(; alg = SOCRiskExpr())
plt = plot_measures(pred_train; x = r, label = "Training", zcolor = nothing)
plt = plot_measures(pred_test; x = r, plt = plt, label = "Test", zcolor = nothing,
                    markercolor = :red, ylabel = "Daily Mean Return",
                    xlabel = "Daily Standard Deviation")
```<code1>
// )
#figure(
  image("fig1.svg", width: 65%),
  caption: [Train and test efficient frontier for an L2 regularised Markowitz model with a scenario-based standard deviation risk measure.],
)<fig1>

== Design philosophy

#link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl] is implemented in Julia @Julia-2017, using the #link("https://github.com/jump-dev/JuMP.jl")[JuMP.jl] @JuMP mathematical optimisation embedded language, and leverages other well-established Julia libraries for much of its functionality. Aside from a strong commitment to FOSS principles, the library is designed atop four pillars:

+ Well-defined type hierachies. This lets us take advantage of generic fallbacks, and enables fearless extensibility, even at the user-level thanks to multiple dispatch.
+ Strongly typed, immutable structs. This lets the compiler agressively optimise code, provides a single source of truth, and enables construction-time data validation that will not expire.
+ Compositional design. This lets us build complex objects from simpler ones, enables code reuse, and aids in extensibility.
+ Defensive programming. This lets us catch errors as early as possible, and provides a clear and consistent interface to the user.

With the exception of covariance estimators, which are defined in #link("https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator")[StatsBase.jl], every type in the library is a subtype of one of three types:

+ `AbstractEstimator`: These contain all the parameters to estimate a particular quantity. They are consumed by functions that perform the estimation. Estimators are used at user-level, they can be used by the public API to return a result. Estimators can be nested within each other, and can be used to build more complex estimators.
+ `AbstractResult`: Many of the estimators return more than one quantity, these are returned in a result type.
+ `AbstractAlgorithm`: These modify the behaviour of estimators via multiple dispatch. They are not used standalone or directly by the public API, but as part of an estimator.

The strict adherence to a well-defined type hierarchy lets developers and users alike extend and modify the library's behaviour without modifying the source code.

The code is available under an MIT license via the Julia General registry. It follows common software development best practices:

+ An extensive and high coverage test suite.
+ Automated documentation builds with an introductory user guide and deep examples, as well as a completely documented public and private API.
+ Miscelaneous code quality checks.

All as part of a GitHub continuous integration pipeline. It is under active development, and receives regular updates and improvements.

= Equations

The template uses #link("https://typst.app/universe/package/i-figured/")[`i-figured`] for labeling equations. Equations will be numbered only if they are labelled. Here is an equation with a label:

$
  sum_(k=1)^n k = (n(n+1)) / 2
$<equation>

We can reference it by `@eq:label` like this: @eq:equation, i.e., we need to prepend the label with `eq:`. The number of an equation is determined by the section it is in, i.e. the first digit is the section number and the second digit is the equation number within that section.

Here is an equation without a label:

$
  exp(x) = sum_(n=0)^oo (x^n) / n!
$

As we can see, it is not numbered.

= Theorems

The template uses #link("https://typst.app/universe/package/great-theorems/")[`great-theorems`] for theorems. Here is an example of a theorem:

#theorem(title: "Example Theorem")[
  This is an example theorem.
]<th:example>
#proof[
  This is the proof of the example theorem.
]


We also provide `definition`, `lemma`, `remark`, `example`, and `question`s among others. Here is an example of a definition:

#definition(title: "Example Definition")[
  This is an example definition.
]

#question(title: "Custom mathblock?")[
  How do you define a custom mathblock?
]

#let answer = my-mathblock(
  blocktitle: "Answer",
  bodyfmt: text.with(style: "italic"),
)

#answer[
  You can define a custom mathblock like this:
  ```typst
  #let answer = my-mathblock(
    blocktitle: "Answer",
    bodyfmt: text.with(style: "italic"),
  )
  ```
]

Similar as for the equations, the numbering of the theorems is determined by the section they are in. We can reference theorems by `@label` like this: @th:example.

To get a bibliography, we also add a citation.

#lorem(50)

#bibliography("bibliography.bib")

// Create appendix section
#show: appendices
=

If you have appendices, you can add them after `#show: appendices`. The appendices are started with an empty heading `=` and will be numbered alphabetically. Any appendix can also have different subsections.

== Appendix section

#lorem(100)
