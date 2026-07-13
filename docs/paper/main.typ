#import "@preview/clean-math-paper:0.2.7": *
#import "@preview/jlyfish:0.1.0": *

#let date = datetime.today().display("[month repr:long] [day], [year]")

// Modify some arguments, which can be overwritten in the template call
#page-args.insert("numbering", "1/1")
#text-args-title.insert("size", 2em)
#text-args-title.insert("fill", black)
#text-args-authors.insert("size", 12pt)

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
#set text(font: "New Computer Modern")
#set enum(numbering: "1.")
#set math.equation(numbering: "(1)")

= Introduction
The field of portfolio optimisation was introduced by Harry Markowitz's seminal 1952 paper "Modern portfolio theory" @markowitz1952. However, the field has grown significantly since then, with many new techniques and methods being developed. The original Markowitz model is highly sensitive to estimation errors in the input parameters, particularly the expected returns. After all, it attempts to summarise the entire distribution of returns in highly compressed summary statistics, both of which can be sensitive to outliers and atypical conditions for the lookback period. Many have studied and addressed the strengths and weaknesses of the Markowitz model @cajasbook @lopezdepradobook @dppalomarbook.

Unfortunately, most implementations live in disparate, unconnected, often propriatary, bespoke codebases, which limit their applicability. It is only recently that various advanced yet very usable libraries have been published @dppalomargithub @riskfolio @skfolio. However, each has its own strengths, weaknesses, scope, and ideosyncrasies. That is no different from #link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl], but it is our hope that this library serves as a unifying framework for the various techniques and methods available in the field, while also eventually providing a simple and intuitive interface for users, with advanced features for experts.

There also exist myriad optimisation methods, pre-filtering, distribution and moment estimators, machine learning techniques, validation, and parameter tuning methods that can all be used together to improve the out-of sample performance of a portfolio. To this day, only @skfolio has succeeded in providing a unified framework for this. #link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl] provides an alternative that is not tied to the scikit-learn @scikit-learn or cvxpy @cvxpy1 @cvxpy2 APIs and ecosystems, as well as providing different functionality and a different architectural philosphy.

= Design and implementation

== Basic usage

#link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl] is built with modularity and extensibility in mind. We can demonstrate a simple improvement over the Markowitz model by using by adding an L2 regularisation term to the optimisation problem.

$
  & min_bold(w) quad && bold(w)^top bold(Sigma) bold(w) + lambda ||bold(w)||_2^2 \
  & upright(s.t.)    && bold(w)^top bold(mu) >= mu_i forall i in {1, ..., N} \
  &                  && bold(w)^top bold(1) = 1 \
  &                  && bold(w) >= 0 \
  &                  && bold(w)_(i != "AAPL") <= 1 \
  &                  && w_("AAPL") <= 0.2
$<eq1>

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
X = TimeArray(CSV.File(joinpath(@__DIR__, "../../examples/SP500.csv.gz"));
              timestamp = :Date)
# Compute returns.
rd = prices_to_returns(X)
# Split into training and test sets.
rd_train, rd_test = train_test_split(rd; test_size = 0.2)
# Mean Risk optimisation. Defaults to minimising risk.
mr = MR(;
  # Variance (default)
  r = Variance(),
  # Configured optimiser.
  opt = JuMPOpt(;
    # Using Clarabel as the solver. It's possible to provide fallbacks in the form of
    # a vector, `PortfolioOptimisers.jl` will iterate until it finds one that works
    # or they all fail.
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
      settings = Dict("verbose" => false)
    ), # slv
    # Weight bounds using an estimator (it builds the constraint based on the data).
    # Lower bound for all assets is 0, upper bound for AAPL is 0.2, for the rest it's
    # 1.
    wb = WBE(; lb = 0, ub = "AAPL" => 0.2),
    # This maps asset names to their indices in the data, as well as sets to which
    # they belong. It is needed to build constraints from estimators and is used by
    # other components such as some prior statistics.
    sets = AssetSets(; dict = Dict("nx" => rd.nx)),
    # L2 regularisation using a squared L2 norm with a scalar value of 0.0001. This
    # is used to prevent weight concentration and thus reduce overfitting and improve
    #generalisation.
    l2 = L2Reg(; val = 0.0001, alg = SquaredSOCRiskExpr()),
    # Arithmetic returns with 100 evenly distributed points between the minimum and
    # maximum returns in the training set. This way we can compute the efficient
    # frontier, which is a subset of pareto fronts.
    ret = ArithmeticReturn(; lb = Frontier(; N = 100))
  ) # opt
) # mr
# Perform optimisation on the training set.
res = optimise(mr, rd_train)
# Predict on training data.
pred_train = predict(res, rd_train)
# Predict on test data.
pred_test = predict(res, rd_test)
# Scenario based standard deviation.
r = SCM(; alg = SOCRiskExpr())
# Plot the results.
plt = plot_measures(pred_train; x = r, label = "Training", zcolor = nothing)
plt = plot_measures(pred_test; x = r, plt = plt, label = "Test", zcolor = nothing,
                    markercolor = :red, ylabel = "Mean Return",
                    xlabel = "Standard Deviation")
savefig(plt, joinpath(@__DIR__, "fig1.svg"))

```<code1>
// )
#figure(
  image("fig1.svg", width: 65%),
  caption: [Train and test efficient frontier for an L2 regularised Markowitz model with a scenario-based standard deviation risk measure. The training set does much better than the test set, this is why other portfolio optimisation modalities have been invented.],
)<fig1>

== Optimisers

#link("https://github.com/dcelisgarza/PortfolioOptimisers.jl/")[PortfolioOptimisers.jl] provides a large number of optimisers. They can be categorised into 4 types.

+ Naïve: Speed and robustness, but not necessarily optimal. Can be interesting as sub-optimisers to more complex optimisers, and as fallbacks.
+ #link("https://github.com/jump-dev/JuMP.jl")[JuMP]-based: These are the most flexible and powerful, but they also require a solver, and are typically slower than other first-order optimisers. They support a wide range of constraints and objectives, and can be used to solve complex optimisation problems. They all use conic reformulations of problems.
+ Hierarchical: These utilise the relational structure of the universe in order to compute the risk of each group and sub-group, eventually leading to a diversified portfolio which encodes the relational structure as well as the risk characteristics of each group/subgroup. They are typically faster than #link("https://github.com/jump-dev/JuMP.jl")[JuMP]-based optimisers, but support fewer constraints. They are very good for large numbers of assets and build well-diversified portfolios.
+ Meta-optimisers: These consume other optimisers and combine their results in order to produce a better result. They are typically slower than other optimisers, but can be very effective in finding good solutions.
+ Finite optimisers: These do not optimise portfolios in the traditional sense, but rather let the user specify a finite cash amount, as well as the asset prices and other options, and compute the best portfolio that can be constructed with the given cash amount. They are used after the others have produced a result.

== Constraints and penalties

There is a huge variety of constraints. Every optimiser supports weight bounds constraints, all but the naïve ones also support fees. However, the richness of the constraints is in the #link("https://github.com/jump-dev/JuMP.jl")[JuMP.jl]-based estimators. They can be divided into a few categories:

+ Upper/lower bounds: These limit the maximum and minimum weights of the assets in the portfolio. Their implementation differs depending on the optimiser, but in #link("https://github.com/jump-dev/JuMP.jl")[JuMP.jl]-based estimators they are implemented as linear constraints.
+ Budget: These limit the total value of the weights of the portfolio. They integrate seamlessly with the finite optimisers in that they adjust the available cash based on the portfolio's budget. Together with bounds and linear constraints, they can be used to implement leveraged, dollar-neutral, and other types of portfolios.
+ Linear constraints: These are used in a variety of contexts, as risk contribution constraints, weight bounds, and relational structure constraints.
+ Cardinality constraints: These these can enforce sparsity, buy-in thresholds, inclusion/exclusion constraints. They can operate at the level of assets, or sets of assets. They are implemented as linear mixed integer constraints.
+ Fees: These penalise the return of the portfolio (thus indirectly penalise certain risk measures and objective functions). They include long and short proportional, long and short fixed, and turnover fees.
+ Weight penalties: These apply L-norm penalties to the objective function, or limit the L-norm of the weights of the portfolio. They can be used to regulate the sparsity and robustness of the portfolio.
+ Tracking and turnover: These limit how much a portfolio can deviate from a reference.

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

#bibliography("refs.bib")

// Create appendix section
#show: appendices
=

If you have appendices, you can add them after `#show: appendices`. The appendices are started with an empty heading `=` and will be numbered alphabetically. Any appendix can also have different subsections.

== Appendix section

#lorem(100)
