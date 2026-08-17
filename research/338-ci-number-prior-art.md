# 338 — How other projects hold a JET or complexity number in CI

Research ticket #338 of wayfinder map #335. Written 2026-08-17.

Sources are the installed package sources (JET 0.12.1 at
`/home/danielcelisgarza/.julia/packages/JET/vOTu0/`, CodeComplexity 0.2.0 at
`/home/danielcelisgarza/.julia/packages/CodeComplexity/JGFE3/`), the published
documentation sites of JET, ESLint, detekt, PHPStan, Codecov and GitHub, and two
Julia Discourse threads. No Julia code was executed for this report. Claims that
need execution are marked **UNVERIFIED**.

---

## Summary — what already exists, and what this map must not rebuild

- **The report filter exists.** JET 0.12 exports seven `ReportMatcher` types and
  applies them through `target_modules` and `ignored_modules`. This map must not
  write its own module or method filter.
- **A report identity exists, but it carries a line number.** JET's
  `default_aggregation_policy` keys on `(report type, Signature, file, line)`.
  Line numbers do not survive an edit. This identity suits deduplication, not a
  committed baseline.
- **A second identity exists and it is not serialisable.** `reportkey` returns
  `(sig.tt, MethodInstance)`. A `MethodInstance` cannot go into a file that is
  committed to the repository.
- **JET ships no baseline and no ratchet.** It ships `broken=true` (a one-bit
  ratchet with no number) and scope narrowing. The `.JET.toml` configuration
  file was *removed* in 0.12.0. Nothing in the map's plan is duplicated work.
- **`test_package` asserts zero, but zero is measured after filtering.** The
  assertion is `length(get_reports(result)) == 0`, and `get_reports` applies
  `target_modules`/`ignored_modules` first. A package with many raw reports can
  therefore pass. Top-level errors are the exception; no filter removes them.
- **The dominant baseline identity outside Julia is `(path, rule, count)`.**
  ESLint, PHPStan and detekt all key on a file path plus a rule or message. Only
  detekt adds a code signature. None of the three keys on a line number. The
  map's per-file baseline is the industry shape, not an invention.
- **The Julia ecosystem's one published number ratchet stores no baseline file.**
  The `julia-invalidations` action recomputes both sides in the same run and
  compares the branch against the default branch. That is a valid alternative to
  a committed baseline.
- **`check_measure` is a gate, but it is a *fixed-threshold* gate, not a
  ratchet.** It takes one scalar `max_value::Int` per metric per call and it
  throws. `throw_on_violation=false` returns the violating files, so the map can
  build a per-file ratchet on top of it. It must not rebuild the AST walk.

---

## 1. What JET itself offers for CI

### The entry points

JET 0.12.1 exports four assertion entry points. Their exact signatures are:

```julia
function test_call(args...; jetconfigs...)                          # src/analyzers/jetanalyzer.jl:1643
function test_file(args...; jetconfigs...)                          # src/analyzers/jetanalyzer.jl:1700
function test_package(args...; toplevel_logger=nothing, jetconfigs...)  # src/analyzers/jetanalyzer.jl:1786
function test_text(args...; jetconfigs...)                          # src/analyzers/jetanalyzer.jl:1819
```

All four delegate to one internal function:

```julia
function func_test(func, testname::Symbol, @nospecialize(args...);
    broken::Bool = false, skip::Bool = false,
    jetconfigs...)
```

(`/home/danielcelisgarza/.julia/packages/JET/vOTu0/src/JETBase.jl:1309`)

### What they assert

The assertion is one line:

```julia
result = func(args...; jetconfigs...)
if length(get_reports(result)) == 0
    Pass(testname, orig_expr, nothing, nothing, source)
else
    JETTestFailure(orig_expr, source, result)
end
```

(`src/JETBase.jl:1323-1329`)

The entry points assert **exactly zero reports**. There is no keyword for a
tolerated count, and there is no comparison against a stored number. This is a
fact from the source.

The count is *not* the raw analysis count. `get_reports` applies the report
configuration first:

```julia
function get_reports(result::JETToplevelResult)
    res = result.res
    if !isempty(res.toplevel_error_reports)
        return res.toplevel_error_reports
    else
        return configured_reports(res.inference_error_reports; result.jetconfigs...)
    end
end
```

(`src/JETBase.jl:501-511`)

Two consequences follow.

**A package with a non-zero raw report count can pass.** The mechanisms are:

1. `target_modules` and `ignored_modules`. These run inside `configured_reports`
   (`src/JETBase.jl:753-771`) and they remove reports before the length check.
   Section 2 describes them.
2. `report_config`. This replaces the standard filter completely with a
   user-defined strategy (`src/JETBase.jl:639-645`). The user extends
   `JET.configured_reports(::T, ::Vector{InferenceErrorReport})`.
3. Analysis-time suppression. `mode = :typo` reports a strict subset of
   `mode = :basic` (`src/analyzers/jetanalyzer.jl:17-22`). `ignore_throws` and
   `ignore_missing_comparison` suppress whole report classes
   (`src/analyzers/jetanalyzer.jl:28-43`). `report_package` turns both of these
   on by default:

   ```julia
   function report_package(pkgmod::Module;
                           ignore_missing_comparison::Bool=true,
                           ignore_throws::Bool=true,
                           jetconfigs...)
   ```

   (`src/analyzers/jetanalyzer.jl:1753-1756`)
4. `broken = true`. This is the closest thing to a ratchet that JET ships, and
   it holds one bit, not a number. See below.
5. `skip = true`. This records a `Broken(:skipped, …)` and runs no analysis
   (`src/JETBase.jl:1320-1321`).
6. For `OptAnalyzer` only, `function_filter` and `skip_noncompileable_calls`
   (`src/analyzers/optanalyzer.jl:132-141`, `:7-23`). The map has already ruled
   `report_opt` out, so these are context only.

**No filter can remove a top-level error.** When `res.toplevel_error_reports` is
non-empty, `get_reports` returns it and never reaches `configured_reports`
(`src/JETBase.jl:503-507`). A parse failure or a concretisation failure therefore
always fails `test_package`, whatever the configuration says. This is a fact from
the source.

### The `broken` mechanism, exactly

```julia
if broken
    if isa(testres, JETTestFailure)
        testres = Broken(testname, orig_expr)
    elseif isa(testres, Pass)
        testres = Error(:test_unbroken, orig_expr, nothing, nothing, source)
    end
else
    isa(testres, Pass) || Test.trigger_test_failure_break(testres)
end
```

(`src/JETBase.jl:1335-1343`)

`broken = true` converts a failure into `Broken`, which does not fail the test
set. It converts a *pass* into an `Error(:test_unbroken, …)`, which does fail the
test set. `broken = true` is therefore a one-bit ratchet: it detects the
transition from "reports exist" to "no reports exist", and it forces the
maintainer to remove the annotation. It cannot hold a number, and it cannot
detect the transition from 40 reports to 41.

### The failure output

The failure is a dedicated `Result` subtype:

```julia
struct JETTestFailure <: Result
    orig_expr::Expr
    source::LineNumberNode
    result::Union{JETCallResult,JETToplevelResult}
end
```

(`src/JETBase.jl:1350-1354`)

Its `show` method prints the header, the expression and the full rendered report
list, each line indented by two spaces (`src/JETBase.jl:1358-1367`). The JET
documentation's *Error Analysis* page gives the rendered form:

```text
JET-test failed at none:1
  Expression: #= none:1 =# JET.@test_call mode = :sound f(10)
  ═════ 1 possible error found ═════
  ┌ f(n::Int64) @ Main ./none:2
  │ non-boolean `Any` may be used in boolean context: goto %5 if not cond
  └────────────────────

ERROR: There was an error during testing
```

(`src/analyzers/jetanalyzer.jl:1583-1595`)

Inside a `DefaultTestSet`, JET prints the failure and then converts it to a plain
`Fail` so that the summary counts work:

```julia
# HACK convert to `Fail` so that test summarization works correctly
fail = Fail(:test_call, t.orig_expr, nothing, nothing, nothing, t.source, false)
Test.record(ts, fail; print_result=false)
```

(`src/JETBase.jl:1380-1382`)

Note the hard-coded `:test_call` label. A `test_package` failure therefore
appears in the test-set summary as a `test_call` failure. **UNVERIFIED** as to
the exact rendered summary line; the source is unambiguous but I did not run it.

The report count itself appears only inside the rendered banner, produced by
`pluralize(length(reports), "possible error")` (`src/ui/print.jl:204`). No entry
point returns the count as a number. A gate that wants the number must call
`length(JET.get_reports(result))` on the result of `report_package`.
`JET.get_reports` carries a docstring (`src/JETBase.jl:486-496`) but it is **not**
in the export list (`src/JET.jl:73-81`) and **not** in the `JETInterface` re-export
list (`src/JETBase.jl:1421-1431`). It is documented internal API.

### Version gating

JET 0.12 adds `const JET_AVAILABLE` (`src/JET.jl:67`). On an unsupported Julia
version JET loads empty stubs and the analysis APIs throw. The README recommends
`if JET.JET_AVAILABLE; include("jet_tests.jl"); end`
(`/home/danielcelisgarza/.julia/packages/JET/vOTu0/README.md:48-57`). A CI gate
that pins Julia exactly does not need this guard, but a gate that floats does.

---

## 2. JET 0.12's report-matching surface

### The type family

```julia
abstract type ReportMatcher end

struct LastFrameModule <: ReportMatcher
    mod::Union{Module,Symbol}
end
struct AnyFrameModule <: ReportMatcher
    mod::Union{Module,Symbol}
end
struct LastFrameModuleExact <: ReportMatcher
    mod::Union{Module,Symbol}
end
struct AnyFrameModuleExact <: ReportMatcher
    mod::Union{Module,Symbol}
end

struct LastFrameMethod <: ReportMatcher
    meth::Union{Function,Method,Symbol}
end
struct AnyFrameMethod <: ReportMatcher
    meth::Union{Function,Method,Symbol}
end
```

(`src/JETBase.jl:653-673`)

All seven names are exported (`src/JET.jl:78-80`).

### What each one matches on

A report carries a `vst::VirtualStackTrace`, which is a `Vector{VirtualFrame}`
ordered from the analysis entry point to the error point
(`src/abstractinterpret/inferenceerrorreport.jl:62-70`). "Last frame" means the
frame where JET detected the error. "Any frame" means any frame on that path.

| Matcher | Frames examined | Match rule | Source |
| --- | --- | --- | --- |
| `LastFrameModule` | last only | module of the frame is the given module *or a submodule*; with a `Symbol`, the frame's module or one of its parents has that name | `src/JETBase.jl:694-698` |
| `AnyFrameModule` | all | same rule, over `any(report.vst)` | `src/JETBase.jl:699-710` |
| `LastFrameModuleExact` | last only | `m === mod`, or `nameof(m) === mod` | `src/JETBase.jl:711-715` |
| `AnyFrameModuleExact` | all | same rule, over `any(report.vst)` | `src/JETBase.jl:716-727` |
| `LastFrameMethod` | last only | `def.name === meth` (Symbol), `def === meth` (Method), or `def in methods(meth)` (Function) | `src/JETBase.jl:728-738` |
| `AnyFrameMethod` | all | same rule, over `any(report.vst)` | `src/JETBase.jl:739-749` |

Submodule containment follows lexical nesting and stops at namespace roots
(`issubmodule`, `src/JETBase.jl:679-686`). `Base` is therefore not a submodule of
`Main`. The 0.12.0 changelog records this as a breaking change
(`CHANGELOG.md:87-91`).

### How they compose

They compose as a flat `any` over one iterator, twice — once for keeping and once
for dropping:

```julia
function configured_reports(config::ReportConfig, reports::Vector{InferenceErrorReport})
    if config.target_modules !== nothing
        reports = filter(reports) do @nospecialize report
            return any(config.target_modules) do m
                m isa ReportMatcher || (m = LastFrameModule(m))
                match_report(m,report)
            end
        end
    end
    if config.ignored_modules !== nothing
        reports = filter(reports) do @nospecialize report
            return !any(config.ignored_modules) do m
                m isa ReportMatcher || (m = LastFrameModule(m))
                match_report(m,report)
            end
        end
    end
    return reports
end
```

(`src/JETBase.jl:753-771`)

Facts that follow:

- The composition is **OR within a list, and AND across the two lists**. There is
  no `not`, no nesting, and no per-matcher conjunction.
- A bare `Module` or `Symbol` is promoted to `LastFrameModule`. That promotion is
  the historical default.
- `ignored_modules` runs **after** `target_modules`. The docstring states this
  (`src/JETBase.jl:559-563`).
- Extension is possible: a user type `T <: JET.ReportMatcher` works once the user
  adds a method to `JET.match_report(::T, ::InferenceErrorReport)`
  (`src/JETBase.jl:541-543`, fallback at `src/JETBase.jl:750-751`). `match_report`
  is **not** exported and **not** in the `JETInterface` list, so the extension
  must spell `JET.match_report`.

### Which keywords accept them

Only two: `target_modules` and `ignored_modules`
(`src/JETBase.jl:639-645`). Both are members of `GENERAL_CONFIGURATIONS`
(`src/JETBase.jl:1389-1396`), so every entry point accepts them.

`ignore_missing_comparison` does **not** accept a matcher. It is a `Bool` field of
`JETAnalyzerConfig` (`src/analyzers/jetanalyzer.jl:46-57`). The same holds for
`ignore_throws`. These act during analysis, not during filtering.

`report_config` accepts an arbitrary object and bypasses both matcher lists
completely (`src/JETBase.jl:565-572`).

### Can matchers replace a hand-maintained exclusion list?

**Partly, and not in the shape the map needs.** This is the analysis, stated as
analysis rather than as fact.

A matcher list *is* an exclusion list. Every accepted report needs an entry, so
the maintenance cost does not disappear; it moves from a report list to a matcher
list.

The matchers key on **module** or on **method**. Neither key is fine enough for a
per-report accept:

- `LastFrameMethod(f)` with a `Function` excludes *every* report in *every* method
  of `f`, including reports added later. The exclusion is over-broad.
- `LastFrameMethod(m)` with a `Method` object is precise, but a `Method` cannot be
  written into a committed configuration file. It must be constructed at run time,
  and it changes identity when the signature changes.
- `LastFrameMethod(:name)` matches by unqualified method name across all modules.
  That is over-broad in the same way `CodeComplexity`'s `ignore=` is (section 6).
- There is no file matcher and no line matcher. A per-file exclusion is not
  expressible.

The matchers also cannot express "accept up to N reports here". They are boolean.
Any number lives outside them.

Conclusion: matchers are the right tool for *whole-module* noise, for example
reports whose last frame sits in a dependency. They are the wrong tool for a
per-file accepted-count baseline. The map should use `target_modules =
(PortfolioOptimisers,)` to define the scope, and hold the number separately.

---

## 3. A baseline file or a ratchet in JET

### In JET itself: none

I searched the JET 0.12.1 source, its `docs/src/` pages, its `README.md` and its
`CHANGELOG.md` for `baseline`, `ratchet` and `regression`. There are no matches
that describe such a feature. The only near-neighbour is `broken`, described in
section 1.

Two further facts point the same way:

- The `.JET.toml` configuration file was **removed** in 0.12.0, with the parent
  directory lookup that `report_file` performed
  (`CHANGELOG.md:98-104`). JET 0.12 has no on-disk configuration mechanism at all.
  Every configuration is a keyword argument.
- JET's own self-check does not hold a number. It narrows the scope and lists
  functions:

  ```julia
  let target_modules = (JET,)
      …
      test_call(JET.report_package, (Module,); target_modules)
  ```

  (`/home/danielcelisgarza/.julia/packages/JET/vOTu0/test/self_check.jl:5-21`)

  The optimisation half is disabled behind an early `return`, with a
  `function_filter` that lists thirteen functions by identity
  (`test/self_check.jl:24-51`). That list is a hand-maintained exclusion list, and
  it is the closest thing to prior art inside JET itself.

### In the community: no baseline pattern either

Two Julia Discourse threads cover this ground directly.

The thread *Testing type stability across Julia and JET versions* records only
scope narrowing. The advice is `target_modules=(@__MODULE__,)` and a
parametrised `function_filter`, and one participant notes the trade-off that
filtering hides real problems. The thread's author of JET states that JET "is not
a sound analyzer to begin with" and that it "prioritizes reducing false positives".
No baseline, no accepted-report list and no allowed-failure count appears in the
thread.

The thread *What to do when packages fail PkgEval because of JET?* records the
opposite conclusion — that the number should not gate at all. One maintainer
writes that JET "is just much too unstable to be part of any reliable toolchain"
and calls it "a tool best used manually during local development". Another
separates CI into a released-version workflow and a nightly workflow and does not
mind when the nightly one fails. The thread's resolution is upstream: JET now
loads as an empty module on unsupported Julia versions.

This is direct support for the map's decision to pin Julia exactly in the gate
while `Test.yml` floats.

### What identity JET *could* key on

JET ships two report identities. Neither is designed for a committed baseline,
but the map must know both.

**`reportkey`** — exported (`src/JET.jl:76`):

```julia
reportkey(report::InferenceErrorReport) = (report.sig.tt, report.vst[end].linfo)
```

(`src/abstractinterpret/inferenceerrorreport.jl:617`)

The docstring says: "The first component is the reported call tuple type … The
second component is the `MethodInstance` of the final virtual frame"
(`src/abstractinterpret/inferenceerrorreport.jl:605-616`). A `MethodInstance` is
a run-time object. It cannot be written to a file and read back. `reportkey` is
for `unique(reportkey, reports)` in one session, not for a baseline.

**`default_aggregation_policy`** — its `aggregation_policy` accessor is
re-exported through `JETInterface` (`src/JETBase.jl:1424`):

```julia
const default_aggregation_policy = function (report::InferenceErrorReport)
    @nospecialize report
    return DefaultReportIdentity(
        typeof(Base.inferencebarrier(report)),
        report.sig,
        VirtualFrameNoLinfo(last(report.vst)))
end
```

(`src/abstractinterpret/abstractanalyzer.jl:412-419`)

Its docstring states the key: "1. The same concrete report type 2. Equal
expression `Signature`s 3. The same file and line in their final `VirtualFrame`s"
(`src/abstractinterpret/abstractanalyzer.jl:398-409`).

This identity **contains a line number**. `VirtualFrameNoLinfo` holds
`file::Symbol` and `line::Int` (`src/abstractinterpret/abstractanalyzer.jl:420-424`).
An edit that inserts one line above a report invalidates the key. This identity
is correct for its purpose, which is to deduplicate reports in one run
(`src/toplevel/virtualprocess.jl:723`). It is wrong for a baseline.

**A hash of the rendered text is worse still.** Two facts support this. First, the
rendered text embeds the same file and line. Second, I found no `sort` or
`sortperm` over the report vector in `src/ui/print.jl`, and `report_package`
appends reports from parallel jobs (`src/toplevel/virtualprocess.jl:870`, with the
0.11.2 changelog recording the parallelisation, `CHANGELOG.md:168-176`). The
report *order* may therefore vary between runs. I did **not** verify this by
running JET. Treat "the rendered text is not order-stable" as an
**UNVERIFIED assumption**, but treat it as a reason not to key on rendered text.

**The identity that does survive an edit** is the pair `(file, report type)`, or
just `file`. Both are derivable from public data: `last(report.vst).file` and
`typeof(report)`. This is exactly the identity that the tools in section 4
converged on.

---

## 4. Prior art outside Julia

Four mechanisms are worth copying. In each case the important part is the
identity the accepted finding keys on, and what happens to that identity after an
edit.

### The suppression file keyed on `(path, rule, count)`

ESLint's *bulk suppressions* feature, documented on the ESLint site as *Bulk
Suppressions* and introduced in the ESLint blog post *Introducing bulk
suppressions*, writes `eslint-suppressions.json`:

```json
{
  "src/file1.js": {
    "no-undef": {
      "count": 1
    }
  },
  "src/file2.js": {
    "no-unused-expressions": {
      "count": 2
    }
  }
}
```

The identity is **file path, then rule identifier, then an integer count**. There
is no line number. The blog post states the reasoning: suppressions track
violations "per file and rule — not per specific line", so ESLint cannot tell a
new violation from a pre-existing one inside a file. Its answer is to report
*all* violations in that file and rule when the count rises above the recorded
number, and never to hide anything speculatively.

Edit behaviour follows from the identity. Moving code inside a file does not
change the key. Adding a violation raises the count and fails. Fixing one lowers
the count and leaves a stale entry, which ESLint warns about and
`--prune-suppressions` removes. `--suppress-all` and `--suppress-rule <rule>`
generate the file.

PHPStan's baseline, documented on the PHPStan site as *The Baseline*, uses the
same shape in NEON: an entry is a `message` regular expression, a `path`, and a
`count`. The identity is again file-plus-rule-plus-count, with the rule replaced
by a regular expression over the rendered message. The PHPStan documentation
notes explicitly that the baseline has no line-number specificity, unlike an
inline ignore annotation. It also warns against casual regeneration, because
regeneration silently accepts new violations.

### The baseline keyed on a code signature

detekt's baseline, documented on the detekt site as *Code Smell Baseline*, keys on
a rule identifier plus a *finding signature*:

```xml
<SmellBaseline>
  <ManuallySuppressedIssues>
    <ID>RuleID:Finding_Signature</ID>
  </ManuallySuppressedIssues>
  <CurrentIssues>
    <ID>RuleID:Finding_Signature</ID>
  </CurrentIssues>
</SmellBaseline>
```

The signature names the file and the enclosing declaration, for example
`Junk.kt$e: RuntimeException`. This is finer than ESLint's key, and it does
survive a pure line move. The documentation warns that it does *not* survive
reformatting: "auto formatting cannot be combined with the `baseline`. The
signatures for a `;` for example would be too ambiguous."

detekt also separates two lists. `CurrentIssues` is the grandfathered debt.
`ManuallySuppressedIssues` is the deliberate false-positive list. That separation
is worth copying: it distinguishes "we will fix this" from "this is not a bug".

### The rebased baseline

The tool *Static Analysis Results Baseliner* (SARB), by Dave Liddament, addresses
the line-number problem head-on. It keys the baseline on file and line, and then
uses the version-control history to *rebase* those line numbers forward when the
file changes between commits. It is the only mechanism found that keeps a
line-level identity alive across edits, and it does so by paying a
version-control-integration cost.

### The coverage ratchet with no stored file

Codecov's *Commit Status* documentation describes a `project` status with
`target: auto` and a `threshold`. `auto` means "use the coverage from the base
commit … to compare against". The baseline is **not a file in the repository**; it
is the previously uploaded report for the base commit. `threshold: 5` permits a
five-point regression while still reporting success. The number can be keyed at
several granularities: the whole project, a flag, or a path or component. A second
status, `patch`, measures only the lines the change touched, so new code is held
to a rule that old code is not.

The Julia ecosystem has one instance of this shape. The `julia-invalidations`
action in the `julia-actions` organisation counts method invalidations with
SnoopCompile, runs the same script on the pull request and on the default branch
**in the same workflow and on the same Julia version**, and fails when the
branch's total exceeds the default branch's total. It stores **no baseline file**;
it recomputes both sides every run. The failure step is a plain comparison:

```yaml
- name: Check if the PR does increase number of invalidations
  if: ${{ fromJSON(steps.invs_pr.outputs.total) > fromJSON(steps.invs_default.outputs.total) }}
  run: exit 1
```

This is the strongest ecosystem-local precedent for holding a number in Julia CI.
Its lesson for map #335 is that the "same Julia version on both sides" property is
what makes the number comparable — the same property the map wants to buy by
pinning Julia exactly.

### The trade-off, stated plainly

The recomputed baseline (Codecov, `julia-invalidations`) never goes stale, and it
needs no maintenance. It costs two analysis runs per pull request, and it cannot
express "this specific finding is accepted forever". The committed baseline
(ESLint, PHPStan, detekt) costs one run, and it names each accepted finding, but
it goes stale and it needs a prune step. The per-file count is the identity that
minimises staleness while staying reviewable in a diff.

---

## 5. A scheduled quality job that opens issues

**I could not establish a Julia package that publishes a scheduled quality job
which opens issues.** I searched for the pattern and found no instance. State this
as an open item, not as a negative result: absence of a search hit is weak
evidence.

What I can state:

**The Julia-ecosystem scheduled bots open pull requests, not issues.** This
repository runs three scheduled workflows, and all three are of that kind:

- `.github/workflows/CompatHelper.yml` — `cron: 0 0 * * *`, `permissions:` with
  `contents: write` and `pull-requests: write`, and `continue-on-error: true` on
  its step.
- `.github/workflows/Copier.yml` — `cron: 0 7 1/7 * *`.
- `.github/workflows/PreCommitUpdate.yml` — `cron: "0 7 1/7 * *"`.

No workflow in `.github/workflows/` requests `issues: write`. `TagBot.yml`
requests `issues: read` only. Any issue-opening gate therefore needs a new
permission grant.

CompatHelper's duplicate behaviour is **not settled**. Its documentation describes
a `master_branch` option and a per-dependency branch naming scheme, and its issue
tracker carries an open report titled "Avoid multiple PRs suggesting the same
change". I could not confirm the exact deduplication key from a primary source.
Do not model the map's dedup on it.

**The general GitHub Actions patterns are these**, and they are documented:

1. **Search by title, then update.** The `create-an-issue` action by JasonEtco
   takes an `update_existing` input. Set to `true`, it updates an open issue with
   the *exact same title* instead of creating a second one. The identity is the
   issue title string.
2. **Close the previous one.** GitHub's own tutorial *Scheduling issue creation*
   does not dedup by title. It offers `CLOSE_PREVIOUS`: "The workflow will close
   the most recent issue that has the labels defined in the `labels` field." The
   identity is a **marker label**, and the strategy is reactive — it closes the
   old issue after creating the new one, rather than preventing the duplicate. The
   documented permission is `permissions: issues: write`.
3. **Query first with the CLI.** GitHub's *Managing your work with GitHub Actions*
   documentation shows the `gh` CLI used inside a workflow to create issues. A
   `gh issue list --label <marker> --state open` guard before the create call is
   the hand-rolled idempotent upsert. The identity is again a marker label.
4. **Guard against overlapping runs.** Independent of dedup, a scheduled workflow
   needs a `concurrency` group with `cancel-in-progress: true`, or two slow runs
   can both open an issue.

Recommendation for the map, stated as a recommendation: use a **marker label** as
the identity, not the title. A title that carries the current number changes every
time the number changes, so title matching would open a fresh issue on every
drift. A label is stable, and the body can carry the number.

---

## 6. Is `check_measure` already the gate this map wants?

### The signatures, exactly

```julia
function check_measure(
    metric::AbstractMetric,
    path::AbstractString;
    max_value::Int = default_max_value(metric),
    throw_on_violation::Bool = true,
    kwargs...,
)
```

(`/home/danielcelisgarza/.julia/packages/CodeComplexity/JGFE3/src/common.jl:112-118`)

```julia
function check_measure(
    metric::AbstractMetric,
    pkg::Module;
    max_value::Int = default_max_value(metric),
    throw_on_violation::Bool = true,
    kwargs...,
)
```

(`src/common.jl:137-143`)

`measure_check` is not a second function. It is a binding:

```julia
const measure_check = check_measure
```

(`src/api.jl`, in the `measure_check` docstring block)

The docstring calls it an "Exact alias of [`check_measure`](@ref)" and says it
exists "for discoverability so the assert / threshold-check verb shows up under
`measure_<TAB>`" (`src/api.jl`). Both names are exported (`src/api.jl`, the
`export` block).

### What the threshold is

The threshold is **one scalar `Int`, per metric, per call**. It is not per-metric
within a call, because a call takes one metric. It is not per-definition and it is
not per-file.

The defaults are three constants:

```julia
default_max_value(::CyclomaticComplexity) = 10
default_max_value(::CognitiveComplexity) = 15
default_max_value(::ArgumentCountComplexity) = 5
```

(`src/common.jl:20-22`)

The comparison is strict:

```julia
return filter(f -> f.value > max_value, fns)
```

(`src/internals/common.jl:17`)

A definition with `value == max_value` passes.

### What it does on violation

It **throws**. It does not return a `Bool` and it does not emit a `@test`.

```julia
function _handle_violations(
    metric::AbstractMetric,
    violations::Vector{<:FileMeasure},
    max_value::Int,
    throw_on_violation::Bool,
)
    (throw_on_violation && !isempty(violations)) || return
    msg = IOBuffer()
    println(msg, _violation_header(metric, max_value))
    for fc in violations
        for func in fc.functions
            line_info = func.line > 0 ? ":$(func.line)" : ""
            println(msg, "  ", _violation_line(metric, fc.path, func, line_info))
        end
    end
    error(String(take!(msg)))
end
```

(`src/internals/common.jl:144-160`)

The message format is:

```text
<metric label> violations (max_value=<N>):
  <path>:<line>: <name> has value <value>
```

(`src/internals/common.jl:164-168`)

The comment above those two functions says the wording is deliberately stable "for
downstream code that greps for it" (`src/internals/common.jl:162-163`). A gate may
therefore parse the message, but parsing is not necessary — see below.

`error(…)` raises an `ErrorException`. Inside a `@testset` it is recorded as an
`Error`, not a `Fail`, and the test set fails. The README's *Use in tests and CI*
section shows exactly that usage:

```julia
@testset "Complexity" begin
    check_measure(CyclomaticComplexity(),    MyPackage; max_value = 10)
    check_measure(CognitiveComplexity(),     MyPackage; max_value = 15)
    check_measure(ArgumentCountComplexity(), MyPackage; max_value = 5)
end
```

(`/home/danielcelisgarza/.julia/packages/CodeComplexity/JGFE3/README.md:205-217`)

### Is it the shape the map wants?

**It is a gate, but it is a fixed-threshold gate, not a ratchet.** The difference:

- It answers "does any definition exceed N?". It does not answer "did the count of
  definitions above N rise since the last commit?".
- It has no notion of a baseline, of an accepted list, or of a previous run.
- The only per-definition escape hatch is `ignore=`, and only for
  `ArgumentCountComplexity`. The README states that matching "is by **unqualified
  name** only: every definition with that name is skipped, even across modules"
  (`README.md:200`). The implementation is a `Set{String}` of names
  (`src/internals/argument_counts.jl:100-127`). This is over-broad in exactly the
  same way `JET.LastFrameMethod(:name)` is.
- The general pre-filter hook `_measure_pre_filter(::AbstractMetric, fns; kwargs...)`
  (`src/internals/common.jl:174`) lives in the `Internals` submodule and is not
  exported. Extending it for a built-in metric would be type piracy.

**What the map should reuse**, and must not rebuild:

- The AST walk, the three metrics, and the file/directory/package traversal.
- `throw_on_violation = false`. With that keyword, `check_measure` returns
  `Vector{FileMeasure}` — the violating files, each with its violating
  `FunctionMeasure`s — and throws nothing (`src/common.jl:112-135`). That return
  value is exactly the per-file data a ratchet needs. The map's gate can compare
  `length(fc.functions)` per `fc.path` against a committed per-file baseline, in
  the ESLint `(path, rule, count)` shape from section 4.
- `FunctionMeasure` carries `name::String`, `value::Int` and `line::Int`
  (`src/api.jl`). A baseline that keys on `(path, name)` survives a line move; one
  that keys on `line` does not.

**Two gate weaknesses to know about:**

1. `measure_directory` swallows per-file failures:

   ```julia
   catch e
       @warn "Failed to analyze $filepath" exception = e
   end
   ```

   (`src/common.jl:62-70`)

   A file that fails to parse contributes zero violations and only a warning. The
   gate would pass. This is a fail-open path. Compare it to JET, where a top-level
   error always fails (section 1). The map's gate should count the analysed files
   and assert that count, or it inherits the fail-open behaviour.
2. When `max_value` is set, `FileMeasure.total_value` is the sum over the
   *surviving* (violating) definitions only, not over the whole file
   (`src/common.jl:38-49` with the `FileMeasure` outer constructor in `src/api.jl`).
   Do not read `total_value` from a `check_measure` result as a whole-file
   complexity total. To get the true total, call `measure_file` with
   `max_value = nothing`.

---

## Local precedent: how `Aqua.yml` is shaped

`/mnt/storage/dev/PortfolioOptimisers.jl/dev/.github/workflows/Aqua.yml` is the
repository's existing quality gate. Its shape:

- **Triggers**: `push` to `main`, `dev*` and `agents/*`, plus tags; `pull_request`;
  and `workflow_dispatch`. There is no `schedule`.
- **One job, `build`, on `ubuntu-latest`.** No matrix.
- **Julia floats.** `julia-actions/setup-julia@latest` with `version: '1'`. Note the
  map's decision to pin the JET gate exactly: `Aqua.yml` is the counter-example,
  and the #325 memory records that a floating `'1'` is what let Julia 1.12.7 drift
  into CI.
- **The step body is an inline Julia script**, run with
  `shell: julia --color=yes {0}`. It does `Pkg.add` for Aqua, `Pkg.develop` for the
  repository, and then calls eight `Aqua.test_*` functions in sequence.
- **The gate is assertion-shaped, not measurement-shaped.** Each `Aqua.test_*` call
  throws or emits a failing test; nothing holds a number, and nothing is compared
  against a stored value.
- **No `permissions:` block.** The job runs with the workflow default token scope,
  and it cannot open an issue.
- **The step is not wrapped in a `@testset`.** The first failing assertion aborts
  the script, so later checks do not run.

A JET or complexity gate that follows this template gets triggers, the inline
Julia script and the `Pkg.add`/`Pkg.develop` pair for free. It must add the pinned
Julia version, the baseline file read, and — for the issue-opening variant of
section 5 — a `permissions: issues: write` block.

---

## What I could not settle

- **The exact rendered `test_package` failure line in a test-set summary.** The
  source hard-codes the label `:test_call` (`src/JETBase.jl:1381`). I did not run
  Julia, so I did not see the printed summary. Marked UNVERIFIED.
- **Whether JET's report order is stable between runs.** I found no sort in the
  printing path and I found a parallel append in `report_package`. This is a
  reason to distrust a rendered-text hash, but it is an inference from source, not
  an observation. Marked UNVERIFIED.
- **Whether any Julia package publishes a scheduled quality job that opens
  issues.** I found none. This is a null search result, not a proof of absence.
- **CompatHelper's exact duplicate-pull-request key.** Its documentation describes
  a branch naming scheme, and its tracker carries an open duplicate-PR report. I
  could not confirm the key from a primary source.
