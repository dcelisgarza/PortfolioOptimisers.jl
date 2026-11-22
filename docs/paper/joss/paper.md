---
title: 'PortfolioOptimisers.jl: composable portfolio optimisation in Julia'
tags:
  - Julia
  - portfolio optimisation
  - quantitative investment
  - conic optimisation
  - parameter estimation
authors:
  - name: Daniel Celis Garza
    orcid: 0000-0003-4622-2234
    affiliation: 1
affiliations:
  - name: Independent researcher, Oxford, UK
    index: 1
date: 22 November 2026
bibliography: paper.bib
---

<!-- JOSS needs a top-level heading for each section, and the title of the front matter counts as
one, so MD025 cannot hold in this file. -->
<!-- markdownlint-disable MD025 -->

# Summary

Portfolio optimisation is the science of either minimising risk whilst keeping returns to
acceptable levels, or maximising returns whilst keeping risk to acceptable levels, to some
definition of acceptable, and with any number of additional constraints. There exist myriad
statistical, pre- and post-processing algorithms, optimisations, and constraints that allow one to
explore an extensive landscape of "optimal" portfolios. PortfolioOptimisers.jl aims to provide as
many of these as possible under a single banner, and to make them accessible to all.

The library is highly composable, so the many different components can be mixed and matched in
arbitrary ways as long as they are compatible, providing users with the ability to create complex
portfolio optimisation workflows in accordance with their needs and creativity. It makes extensive
use of Julia's type system, module extensions, and multiple dispatch [@Julia-2017] to achieve
performance, robustness, correctness and testability, without sacrificing maintainability or
usability.

# Statement of need

The field of portfolio optimisation was introduced by Markowitz [@markowitz1952]. The field has
grown enormously since then, and so has the catalogue of that model's failure modes. The model
summarises the entire distribution of returns in its first two moments, the mean and the
covariance, both of which can be sensitive to outliers and atypical conditions of the lookback
period. The optimiser then behaves as an error maximiser rather than a diversifier [@michaud1989],
is acutely sensitive to the expected returns it is given [@best1991], and suffers roughly an order
of magnitude more from errors in the means than from errors in the covariance [@chopra1993]. In
contrast, the completely naïve, equal weighted portfolio remains a stubborn out of sample benchmark
[@demiguel2009naive] when the universe is sufficiently curated to remove obvious underperformers.
Seven decades of responses, from constraints [@jagannathan2003] and shrinkage [@ledoit2004] to norm
regularisation [@demiguel2009norm], robust formulations and hierarchical ones, are by now a
literature of their own [@kolm2014; @cajasbook; @lopezdepradoml; @dppalomarbook].

Unfortunately, most implementations live in disparate, unconnected, often proprietary, and/or
bespoke research codebases, which limits their applicability. Yet the optimisation methods,
pre-filtering, distribution and moment estimators, machine learning techniques, validation, and
parameter tuning methods can all be used together to improve the out of sample performance of a
portfolio.

PortfolioOptimisers.jl serves researchers and practitioners who need these methods in one framework,
where any estimator feeds any optimiser that asks for it, and where a whole workflow can be
cross-validated and tuned as a single unit. Its documented and tested interfaces let users and
maintainers extend it, and its modularity lets them swap components in and out.

# State of the field

Various advanced and very usable libraries have been published only relatively recently
[@riskfolio; @skfolio; @cvxportfolio1; @cvxportfolio2], each with its own strengths, weaknesses,
scope, and idiosyncrasies. Earlier general-purpose libraries include PyPortfolioOpt
[@pyportfolioopt] in Python, and PortfolioAnalytics [@portfolioanalytics] and fPortfolio
[@fportfolio] in R. To the author's knowledge, only skfolio [@skfolio] provides a unified framework
for the estimators, optimisers, validation and tuning of a portfolio. PortfolioOptimisers.jl
provides an alternative that is not tied to the scikit-learn [@scikit-learn] or cvxpy
[@cvxpy1; @cvxpy2] APIs and ecosystems, as well as providing different functionality and a
different architectural philosophy.

A contribution to an existing package would not have satisfied the author's goals in
providing such a flexible and powerful library. Among the most relevant reasons are the following:

- The solver-agnostic declarative modelling of JuMP [@JuMP], alongside its rich and vibrant
  ecosystem, is extremely powerful. With open-source solvers such as Clarabel [@clarabel], HiGHS
  [@highs], and Pajarito [@pajarito] it solves problems that have traditionally required commercial
  solvers, or that were out of reach for lack of suitable implementations.
- Multiple dispatch lets users expand any and all functionality as long as they follow and/or
  implement the necessary public interfaces, which are documented and tested.
- The complete freedom to create and implement generalisations, extensions, improvements, and novel
  methods without approval or being forced to maintain a fork.

# Software design

The design rests on well-defined type hierarchies, strongly typed immutable structs that validate
their data at construction, compositional design, and defensive programming that catches errors as
early as possible.

When it comes to typing, most public types in the library are subtypes of one of three types:

- `AbstractEstimator` contains all the parameters needed to estimate a particular quantity. These
  types are the user-level vocabulary. They often nest within each other, but have meaningful
  standalone use.
- `AbstractResult` holds the quantities that an estimation returns: estimators are configuration,
  results are data.
- `AbstractAlgorithm` modifies the behaviour of an estimator via multiple dispatch. These types are
  not used standalone, but as part of an estimator.

The exceptions are the covariance estimators, which subtype the abstract covariance estimator of
StatsBase.jl; a distance metric, which subtypes a metric of Distances.jl; and the error types, which
follow Julia's error interfaces.

The library's functionality is grouped into three stages. The first two can be bundled into a
pipeline, and some estimators of each of them can use the functionality of the other.

1. Preprocessing ingests prices, treats gaps, preselects assets and computes returns.
2. Processing estimates the prior statistics, uncovers the relational structure of the asset
   universe through clustering and network analysis, builds the constraints and the uncertainty
   sets, optimises, cross-validates, predicts, and tunes.
3. Postprocessing scores the result: performance, attribution, forecast evaluation and regret.

One of the central novelties of PortfolioOptimisers.jl is the concept that risk and return measures
are fully fledged objects with their own scales, bounds, prior statistics, and parameters. Each
measure therefore carries its own parameters, rather than placing them in the configuration of an
optimiser or the arguments of a function. Many estimators accept vectors of measures, which they
scalarise with one of several algorithms that use the scale of each measure, so users can compose
risk and return expressions from different constituents. Some measures can also be pre- or
dynamically configured, or take their configuration from the prior statistics. For example, a
variance can use a preconfigured covariance matrix, take the prior covariance matrix from the
optimisation estimator, or compute its own covariance matrix from input data given to the
optimisation.

Constraints and views are constructed by estimators using near natural language, for instance
`"AAPL <= 0.1"`, rather than written as matrices, though explicit linear algebra constructs are
also accepted. A constraint is therefore stated once and rebuilt against whatever data each fold of
a cross-validation, or subset of the universe an optimisation meets.

At the time of writing, the library counts 57 risk measures including robust ones [@drcvar], 22
covariance and correlation estimators, 13 priors, including Black-Litterman variants
[@blacklitterman1992] and entropy pooling [@meucci2008], 20 optimisers, from mean risk with
conditional value at risk [@rockafellar2000] and distributionally robust measures to hierarchical
risk parity [@hrp2016], 21 online portfolio selection rules [@lihoi2014], and eight
cross-validation and tuning schemes, including combinatorial splits [@lopezdepradoml]. It also
offers 20 types of constraints, costs and penalties, among them static and score-scaled bounds,
budgets, joint, exclusion and set constraints, buy-in thresholds, lot units, fees including
delisting and borrowing costs, graph, network and centrality constraints, tracking, turnover, and
regularisation.

Furthermore, the library is built with real use cases in mind, which often means dealing with messy
data due to late listings, delistings, mismatched holidays, suspensions, etc. PortfolioOptimisers.jl
never deletes a gap, though it can fill it during preprocessing: the returns carry an asset panel
whose masks say which assets are listed at each observation, and which of those have a finite
return. Every optimiser fits on the assets it can trade and gives every other asset a weight of
exactly zero, so each fold of a walk-forward runs on the universe of that fold, without
survivorship or look-ahead bias.

# Research impact statement

PortfolioOptimisers.jl is registered in the Julia General registry under the EUPL-1.2 licence.
Where a method has a reference implementation, the library measures its results against it. Its
test suite covers 99% of the lines. Its public and private API are documented, and its examples
cover the foundations, the moments and priors, the optimisers, the constraints and costs, the
validation and tuning, and the post-processing, each executed by the documentation build. A
toolchain of deterministic checks finds regressions, type instabilities, missing documentation and
references, and other defects on every commit, and every week. The library is the engine of a
separate data and strategy pipeline by the same author.

At the time of writing, the package receives about 136 downloads a month, mostly from the USA and
Central Europe, and some from South Korea, Japan, India and Australia. With 31 stars on GitHub, it
is, to the author's knowledge, the most starred and most downloaded portfolio optimisation package
in the Julia ecosystem.

# AI usage disclosure

Until the release of Claude Sonnet 4.6 (Anthropic), the library was written entirely by hand, and
its tests covered around 86% of the lines. From then on, AI tools were used to standardise the API
documentation and to find and fix simple bugs. From Claude Opus 4.8 onwards, they assisted with the
implementation of features and examples, and with code quality. From Claude Opus 5 onwards, they
have been used heavily for new features, tests, documentation, the user guide, examples, bug fixes,
and code review. Only the latest frontier models are used for new features and code review;
smaller models do simple tasks that do not affect code quality. AI tools also helped to draft and
proofread parts of this paper, and to check its feature counts.

The author designs and plans every feature, and reviews and edits all generated code and text. The
issue tracker records the design, development, and testing of each change. Skills, standards, and
deterministic checks, refined since the start, gate every change whatever its origin. They check
code and prose quality, type stability, and docstring structure. They require full coverage and
tests against the sources of each method, and they doctest the docstrings, which explain the API,
algorithms and mathematics with references. Pull requests are extensively reviewed, and all issues
found during review are fixed before they are merged.

# Acknowledgements

This work received no external funding.
