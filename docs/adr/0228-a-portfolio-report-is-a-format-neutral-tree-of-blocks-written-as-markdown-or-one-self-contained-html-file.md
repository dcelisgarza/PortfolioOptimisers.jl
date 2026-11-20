---
status: accepted
---

# A Portfolio Report is a format-neutral tree of blocks, written as Markdown or as one self-contained HTML file

## Context

The library can fit, predict, summarise, allocate and plot portfolios, but it could not put these
results in one document. A caller who compared ten portfolios wrote each table and each plot by
hand, and then put them together in a notebook or a page of their own. Issue
[#1943](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1943) asked for a report: a
document model, a builder that fills it from prediction results, and two writers, Markdown and one
HTML file with no server.

The inputs are the three prediction types: a single fold (`PredictionResult`), a walk-forward path
(`MultiPeriodPredictionResult`) and a population of paths (`PopulationPredictionResult`). Each can
hold many members, when its optimiser gives many portfolios. `performance_summary` (ADR 0227)
already summarises all of them, with one row for each candidate, path and member.

## Decision

**1. The dependency boundary.** The report model, the builder, the Markdown writer, the HTML writer
and the inline style and script are in the core, in `src/24_Report/`. The one new dependency is the
standard library `Base64`, for the figures as data URIs. Markdown and HTML are text, so a writer
needs no package, and a package extension needs a trigger package. The default figures are methods
of the plotting extension, which exists. A future format that needs a package, such as a PDF
through a typesetter, is its own extension, with methods of the block interface for its MIME.

**2. A report is a tree of blocks with no knowledge of portfolios.** `AbstractReportBlock` is
`public`, and its `# Interfaces` section names the extension point: a block has
`show(io, MIME"text/markdown"(), b)` and `show(io, MIME"text/html"(), b)`. The blocks are
`ReportText`, `ReportTable`, `ReportFigure` and `ReportSection`. A `ReportEntry` is one portfolio,
and a `Report` holds the entries and an overview. Every writer escapes every string of the caller,
so a name cannot add markup. A `ReportFigure` takes any object that shows itself as an SVG or a PNG
image, so the block does not depend on Plots.

**3. One candidate is one entry, whatever its type.** The builder `portfolio_report` gives one
entry for each element of its vector. The paths of a population and the members of a prediction
are rows inside the entry, and the builder never splits a candidate. The overview table is the
table of `performance_summary`, so a statistic has one source, and a population entry adds the
distribution of each figure over its paths. The weights of every type come from `fold_target`,
on the names of each fold, which `performance_summary` reads for `n_eff` and `turnover`. The folds
of a path and the paths of a population can hold different assets, so `weights_by_name` puts them
on the union of their names. The composition plots of a path and of a population read the same
rows, which corrects them: before, they put the weights of every fold on the names of the first
fold. A finite allocation is the allocation of one portfolio, so a path allocates the target of
its last fold, and a population has no allocation.

**4. A figure is a function of the prediction.** An entry figure is `f(pred)` and an overview
figure is `f(preds, names)`, so a legend names the predictions. Each gives a figure or `nothing`.
A default figure gives `nothing` for an input that it cannot draw, by dispatch and never by a
caught error, and a function that throws stops the build.

**5. Figures are data URIs in an image tag, in both formats.** An inline `<svg>` element shares the
identifiers and the styles of its page, so two inline figures with one identifier clip each other.
GR numbers its clip paths from a counter of the process, so the figures of one process do not
collide (see *Measurements*). Another backend, or figures drawn in two processes, can. The data
URI stays for one rule in both formats: a Markdown file has no inline SVG, and an image tag
isolates each figure for every backend.

**6. The HTML file loads nothing.** It holds one `<!doctype html>` document with an inline
`<style>` and an inline `<script>`, read at load time from `report.css` and `report.js`. A Content
Security Policy, `default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src
'unsafe-inline'`, makes the browser refuse any load from the network. The script shows one entry at
a time, with a selector grouped by `<optgroup>`, keeps the selection in `location.hash`, moves
between the entries with the arrow keys, sorts a table by a click on a header, filters the groups,
and toggles the theme. With no script, and in print, every entry shows.

## Measurements

The prototype built a report of 30 walk-forward paths of `RandomWeighted` portfolios, over 2520
daily observations of 10 assets, with folds of 63 observations after a training window of 252.
It used the default figures: 2 in the overview and 4 in each entry, 122 figures in all. One
single-threaded run gave these numbers:

| Output | Size | Time to write |
| --- | --- | --- |
| HTML, SVG figures (the default) | 29.33 MB | 6.2 s |
| HTML, PNG figures | 8.04 MB | 3.4 s |
| Markdown, SVG figures | 28.7 MB | not measured |

The build of the report took 17.0 s. An SVG image holds every point of its series, so the SVG
file grows with the length of the series and the PNG file does not. `figure_mime =
MIME"image/png"()` is the control for a large report. The report does no downsampling of a
series.

Two drawdown figures of one process carried the clip identifiers `clip920` to `clip923` and the
next four numbers, and shared none.

## Rejected

- **A core that writes Markdown and an HTML extension.** No package triggers the HTML extension,
  so it needs an artificial weak dependency that the writer does not use.
- **A separate report package.** The report reads the internals of the results, the investable
  view and the summary columns, and moves with them.
- **Plotly figures.** A self-contained file must inline a bundle of about 3.5 MB, the library has
  no Plotly dependency, and the Plotly backend of Plots writes a script tag that loads the bundle
  from a network. An interactive figure block can come later as its own extension.
- **Hand-written SVG charts in the core.** They copy the charts of the plotting extension, and two
  chart paths drift.
- **One entry for each path of a population.** A population of many paths gives a selector of
  many entries, and the comparison that a population exists for, the spread of a figure over its
  paths, is then in no entry. The paths are rows of one entry.
