---
status: accepted
---

# A file over the size threshold that mixes subjects becomes a directory by subject

## Context

[ADR 0178](0178-a-file-over-the-size-threshold-that-reads-in-order-is-cut-into-letter-parts.md)
cut eight files over the size threshold into letter parts, because each held one subject in
order. Six other files over the threshold mix subjects, and the maintainer asked for them to be
split by subject, as the docstring dictionaries were
([ADR 0177](0177-the-docstring-dictionaries-are-a-directory-one-subject-to-a-file.md)).

- `src/02_Tools.jl` holds the element-wise operators, the array views, the type utilities,
  `factory`, `@propagatable`, `@forward_properties` and the vector-to-scalar measures.
- `src/17_Optimisation/01_Base_Optimisation.jl` holds the optimisation types, pipe routing, the
  time-dependent machinery, the weight finalisers, the investable universe, the panel collapse
  and the investable view of a result, with the two return codes placed apart from their type.
- `src/17_Optimisation/05_JuMP/01_Base_JuMPOptimisation.jl` holds the JuMP types, the model
  accessors, the model state (in two runs), the frontier sweep and the model setup.
- `src/17_Optimisation/05_JuMP/02_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl` holds
  the return estimators, the objective functions (in two runs) and the return constraints.
- `src/10_Prior/06_EntropyPooling/01_Base_EntropyPoolingPrior.jl` holds the algorithms, the mean
  views, the view types, the solvers, the moment views and the tail-view solve, with
  `RhoParsingResult` apart from the rho views that read it.
- `ext/PortfolioOptimisersPlotsExt.jl` spreads the methods of one verb over several subjects.

## Decision

**Each of the five `src/` files becomes a directory of the same name, one subject to a file,
numbered in load order.**

| Old file | New files |
| -------- | --------- |
| `02_Tools.jl` | `01_Operators`, `02_ArrayViews`, `03_TypeUtilities`, `04_Factory`, `05_Propagatable`, `06_ForwardProperties`, `07_VectorToScalarMeasures` |
| `01_Base_Optimisation.jl` | `01_OptimisationTypes`, `02_PipeRouting`, `03_TimeDependent`, `04_WeightFinalisers`, `05_InvestableUniverse`, `06_PanelCollapse`, `07_ResultInvestableView` |
| `01_Base_JuMPOptimisation.jl` | `01_JuMPOptimisationTypes`, `02_JuMPModelAccessors`, `03_JuMPModelState`, `04_FrontierSweep`, `05_JuMPModelSetup` |
| `02_Returns_and_ObjectiveFunctions.jl` | `01_ReturnEstimators`, `02_ObjectiveFunctions`, `03_ReturnConstraints` |
| `01_Base_EntropyPoolingPrior.jl` | `01_EntropyPoolingAlgorithms`, `02_EntropyPoolingMeanViews`, `03_EntropyPoolingViewTypes`, `04_EntropyPoolingSolvers`, `05_EntropyPoolingMomentViews`, `06_EntropyPoolingTailViewSolve` |

**The unit of a split is a top-level definition with the lines before it.** A specification
assigns each definition of the old file to one new file, and inside a file the definitions keep
their old order. The split then proves that the new files hold every non-blank line of the old
file exactly once, apart from the `export` and `public` statements.

**A definition moves across a load position only when every name Julia reads at load time is
defined before it.** Julia resolves the names inside a method body when the method runs, so a
move can break only a supertype, a field type, a `const` value, a signature type, or a macro
call. The six definitions that move later or earlier here are the two return codes, which need
only their abstract type; the objective penalty and setters, which read the objective types of
the same new file; the model-state registry, which the frontier sweep calls only at run time;
and `RhoParsingResult`, which only the moment views read. The package loads with the new
include list, and that load is the check.

**Each file ends with the `export` and `public` lines of the names it defines**, as ADR 0178
states for a letter part. A name that no new file defines, such as a result type that a later
file declares, stays in the first file, beside its old neighbours in the statement.

**A bare `@docs` entry whose docstrings now sit in two files goes to the page of the first.** A
bare entry renders every docstring of its name, so it cannot be divided. `get_pr_value`,
`entropy_pooling` and `assert_special_nco_requirements` are the three cases. A section's prose
follows the first page that takes an entry from the section, as ADR 0150 states.

**`CodeHealth.DECLARING_FILES` names the files that now hold the Declaration Macros:**
`02_Tools/05_Propagatable.jl`, `02_Tools/06_ForwardProperties.jl` and
`01_Base_Optimisation/02_PipeRouting.jl`.

Everything else keyed on an old path moves as ADR 0178 states: the pages, the manifest rows, the
baselines, the Exemptions and the perf dismissals, the link targets in the ADRs, the file list of
`test_28` (the JuMP interface is now its directory), the method file that `test_40` checks, and
the comments and prompts that named a file.

**The Plots extension takes the same rule in a commit of its own**, because its code sits inside
a `module` block. `ext/PortfolioOptimisersPlotsExt.jl` becomes `ext/PortfolioOptimisersPlotsExt/`,
where Julia also finds an extension: the module file `PortfolioOptimisersPlotsExt.jl` keeps the
`using` and `import` lines and the two helpers every subject reads, and includes twelve files in
order. Its returns subject alone stood at 810 code lines, so it takes three files, and the factor
moments leave the moment plots, which stood at 495.

| New file | What it holds |
| -------- | ------------- |
| `01_CumulativeReturnsPlots` | cumulative portfolio and asset returns, the benchmark |
| `02_DrawdownAndHistogramPlots` | drawdowns, rolling drawdowns, histograms |
| `03_MeasurePlots` | risk measures, rolling measures, the performance summary |
| `04_CompositionPlots` | composition, stacked composition, risk and factor risk contribution |
| `05_ClusteringPlots` | network, dendrogram, clusters, centrality |
| `06_MomentPlots` | correlation, expected returns, covariance, eigenspectrum, prior, coskewness, cokurtosis |
| `07_FactorMomentPlots` | factor loadings, factor covariance, factor expected returns |
| `08_AttributionPlots` | the factor attribution plots |
| `09_CrossValidationPlots` | cross-validation scores, turnover, weight stability, the dashboard |
| `10_FrontierPlots` | the portfolio dashboard and the efficient frontier |
| `11_FactorDiagnosticsPlots` | the cross-sectional regression, exposure and idiosyncratic diagnostics, the factor summary and forecasts |
| `12_ForecastEvaluationPlots` | the forecast evaluation plots |

Every definition of the extension is a method or a `const` string that a method reads when it
runs, so no move can break the load order. The extension loads, and `test_25_plotting.jl` passes.

## Consequences

- No file under `src/` or `ext/` stands over the size threshold.
- A new definition goes to the file of its subject. A new subject in one of these directories
  takes the next number.
- The entry test of ADR 0074 reads each new file as an addition, so each definition over a
  complexity threshold takes an Exemption citing `a-definition-a-file-split-moved-unchanged`.
