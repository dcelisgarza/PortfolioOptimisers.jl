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
  - name: Independent researcher
    index: 1
date: 11 November 2026
bibliography: paper.bib
---

<!-- JOSS needs a top-level heading for each section, and the title of the front matter counts as
one, so MD025 cannot hold in this file. -->
<!-- markdownlint-disable MD025 -->

<!--
DRAFT for the author (issue #2026). The prose of most sections comes from docs/paper/main.typ.
Each comment below marks the text that is new. Rewrite it in your own words before submission.
-->

# Summary

Portfolio optimisation is the science of either minimising risk whilst keeping returns to
acceptable levels, or maximising returns whilst keeping risk to acceptable levels, to some
definition of acceptable, and with any number of additional constraints. There exist myriad
statistical, pre- and post-processing algorithms, optimisations, and constraints that allow one to
explore an extensive landscape of "optimal" portfolios. PortfolioOptimisers.jl is an attempt at
providing as many of these as possible under a single banner and making it accessible to all.

The library is highly composable, so the many different components can be mixed and matched in
arbitrary ways as long as they are compatible, providing users with the ability to create complex
portfolio optimisation workflows in accordance with their needs and creativity. It makes extensive
use of Julia's type system, module extensions, and multiple dispatch [@Julia-2017] to achieve
performance, robustness, correctness and testability, without sacrificing maintainability or
usability.

# Statement of need

The field of portfolio optimisation was introduced by Markowitz [@markowitz1952]. The field has
grown enormously since then, and so has the catalogue of that model's failure modes. The model
summarises the entire distribution of returns in highly compressed summary statistics, which can
be sensitive to outliers and atypical conditions of the lookback period. The optimiser then
behaves as an error maximiser rather than a diversifier [@michaud1989], is acutely sensitive to
the expected returns it is given [@best1991], and suffers roughly an order of magnitude more from
errors in the means than from errors in the covariance [@chopra1993]. The equal weighted portfolio
remains a stubborn out of sample benchmark [@demiguel2009naive]. Sixty years of responses, from
constraints [@jagannathan2003] and shrinkage [@ledoit2004] to norm regularisation
[@demiguel2009norm], robust formulations and hierarchical ones, are by now a literature of their
own [@kolm2014; @cajasbook; @lopezdepradoml; @dppalomarbook].

Unfortunately, most implementations live in disparate, unconnected, often proprietary, and/or
bespoke codebases, which limit their applicability. There also exist myriad optimisation methods,
pre-filtering, distribution and moment estimators, machine learning techniques, validation, and
parameter tuning methods that can all be used together to improve the out of sample performance
of a portfolio. PortfolioOptimisers.jl serves researchers and practitioners who need these
methods in one framework, where any estimator feeds any optimiser that asks for it, and where a
whole workflow can be cross-validated and tuned as a single unit.

# State of the field

Various advanced yet very usable libraries have been published only recently
[@riskfolio; @skfolio]. Each has its own strengths, weaknesses, scope, and idiosyncrasies. To
this day, only skfolio [@skfolio] has succeeded in providing a unified framework for the
estimators, optimisers, validation and tuning of a portfolio. PortfolioOptimisers.jl provides an
alternative that is not tied to the scikit-learn [@scikit-learn] or cvxpy [@cvxpy1; @cvxpy2]
APIs and ecosystems, as well as providing different functionality and a different architectural
philosophy.

<!-- DRAFT: new text. JOSS asks why a new package and not a contribution to an existing one. -->
A contribution to an existing package could not give three properties that motivated a new
library. First, the solver-agnostic conic modelling of JuMP [@JuMP] lets every risk measure be a
reformulation that any compatible solver, such as Clarabel [@clarabel], accepts. Second, multiple
dispatch lets a user add a risk measure, a prior or an optimiser in their own package, and it
composes with everything else without a change to the library. Third, the library is built around
point-in-time data and online estimation, which change the inputs of every estimator, and which
are hard to add to a design that assumes a fixed, complete panel of returns.

# Software design

Every type in the library, with the exception of the covariance estimators defined in StatsBase.jl,
is a subtype of one of three types. An `AbstractEstimator` contains all the parameters needed to
estimate a particular quantity. An `AbstractResult` holds the quantities that an estimation
returns: estimators are configuration, results are data. An `AbstractAlgorithm` modifies the
behaviour of an estimator via multiple dispatch. The design rests on four pillars: well-defined
type hierarchies, strongly typed immutable structs that validate their data at construction,
compositional design, and defensive programming.

The functionality is grouped into three stages. Preprocessing ingests prices, treats gaps,
preselects assets and computes returns. Processing estimates the prior statistics, uncovers the
relational structure of the asset universe through clustering and network analysis, builds the
constraints and the uncertainty sets, and optimises. Postprocessing scores the result:
performance, attribution, forecast evaluation and regret. The preprocessing and processing steps
can be bundled into a single pipeline object, which is itself an estimator, so it can be
cross-validated and tuned as a unit.

A risk measure is an object, not a hard-coded branch. Any optimiser that takes one takes a vector
of them, and every prior-derived quantity a measure needs may be given as a value or as the
estimator that computes it, in which case it is refitted against the optimisation's own prior in
every cross-validation fold. Constraints are built by estimators rather than written as matrices,
so a constraint is stated once and rebuilt against whatever data it meets. Linear constraints can
be written as equations in near-natural language, for instance `"AAPL <= 0.1"`.

At the time of writing, the library counts 57 risk measures, 22 covariance and correlation
estimators, 13 priors, including Black-Litterman variants [@blacklitterman1992] and entropy
pooling [@meucci2008], 20 optimisers, from mean risk with conditional value at risk
[@rockafellar2000] and distributionally robust measures [@drcvar] to hierarchical risk parity
[@hrp2016], 22 online portfolio selection rules [@lihoi2014], and eight cross-validation and
tuning schemes, including combinatorial splits [@lopezdepradobook].

Real universes change. Assets list late, delist, and are suspended. PortfolioOptimisers.jl never
deletes a gap: the returns carry an asset panel whose masks say which assets are listed at each
observation, and which of those have a finite return. Every optimiser fits on the assets it can
trade and gives every other asset a weight of exactly zero, so each fold of a walk-forward runs
on the universe of that fold, without survivorship or look-ahead bias.

# Research impact statement

<!--
DRAFT: new text. Add the evidence that you have: downloads from the Julia package server,
GitHub stars, citations, talks, and users outside your own projects. Do not claim a number
that you cannot show.
-->
PortfolioOptimisers.jl is registered in the Julia General registry under the MIT licence. Its
test suite covers 98% of the lines. Its public and private API are documented, and its examples
cover the foundations, the moments and priors, the optimisers, the constraints and costs, the
validation and tuning, and the post-processing, each executed by the documentation build. A
toolchain of deterministic checks finds regressions, type instabilities, missing documentation
and other defects on every commit and every week. Where a method has a reference implementation,
the library measures its results against it. The library is the engine of a separate data and
strategy pipeline by the same author.

# AI usage disclosure

<!--
DRAFT: new text. Check every fact below. JOSS asks for the tools and their versions, where they
were used, and a confirmation that humans reviewed the output and made the core design decisions.
-->
Generative AI tools were used to write code, tests, documentation and reviews of this library,
and to draft this paper. Since September 2025, 360 of the 447 commits of the repository carry a
co-author trailer for Claude Opus 5.5 (Anthropic), used through Claude Code. One commit carries
Claude Sonnet 5.5 and one carries GitHub Copilot. The tools were used for new features, bug
fixes, tests, documentation, code review, and maintenance refactors.

The author made the core design decisions. The architecture, the type hierarchy and each design
choice are recorded in architecture decision records in the repository, and the author decided
each one. The author reviewed the AI-assisted output. Its correctness is checked by the test
suite, by the deterministic code-health checks described above, by doctests, and by comparison
with reference implementations where they exist.

For the long paper of the library, AI was used solely in an editorial capacity: to check that the
feature list is comprehensive, accurate and current, to mend gaps in the references, and to
proofread. This JOSS paper was drafted by Claude Opus 5.5 from the prose of that paper, and the
author reviewed and edited it.

# Acknowledgements

<!-- DRAFT: write your acknowledgements, or state that there are none. -->

# References
