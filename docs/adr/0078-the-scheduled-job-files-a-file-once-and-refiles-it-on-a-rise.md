---
status: accepted
---

# The scheduled job files a file once, and refiles it when the number rises

## Context

[ADR 0076](0076-the-code-health-pass-rule-is-a-ratchet.md) separates the two mechanisms: the
ratchet stops decay, and a scheduled job drives improvement. The ratchet is green the day it lands,
so without a second mechanism no number ever falls.

The job runs against a tracker that already carries several live wayfinder maps, so its noise is
the design problem. **38** of the 196 files in scope stand above at least one threshold today, and
**30** remain once the 17 constructor Exemptions drop. **73** more carry a JET report that no
Dismissal covers. The union is **80** files. A job that files every offender every week would post
all of them in its first run and repost them every Monday after that.

## Decision

### The offender is a file, and the job is a standing queue

| Question | Answer |
| --- | --- |
| Cadence | **Weekly**, Monday, plus `workflow_dispatch`. |
| Unit | One **file**, never a definition and never a report. |
| How many | **Top up to five open**, not a per-run cap. |
| Label | **`code-health`**, one label, standalone. No parent issue and no map. |
| Body | The path, each breached metric with its value, baseline and threshold, the worst definitions by name and line, the JET reports by kind and site, and the reproduce command. |

The workflow lives on `main` so the schedule fires, and it checks out `dev`. It uses the default
`GITHUB_TOKEN` with `contents: read` and `issues: write`, guarded by `github.repository` so a fork
never files into anyone's tracker.

**The issue body is never machine-read.** It is written for a person, and every fact the job needs
on a later run comes from a committed file or from the tracker's own metadata.

### The thresholds are conventional, and they live as data

The thresholds are **cyclomatic 10, cognitive 15 and argument count 10**. JET needs no threshold:
one reviewed-real report makes the file a candidate. A percentile was rejected because it never
terminates — a fixed share of the codebase is always in the worst decile.

The first two are the textbook numbers. Argument count was frozen at the textbook 5 while this map
was charted, and it rejected **101** of the 196 files, so it was re-derived from this repository's
own distribution. Counting files by their maximum argument count gives 22 at 4, 29 at 5, 18 at 6,
18 at 7, 22 at 8, 18 at 9, then **4 at 10**. The body of the codebase sits in the 4-to-9 band and
the count collapses above it, and the other two metrics break in the same place. At 10 all three
metrics report **21** files each, which matters because the ranking below compares
`value / threshold` **across** metrics — a ratio is only comparable when a ratio of 1 means the same
share of the codebase for each.

**The cliff is why 10 was chosen once. It is not a rule to re-run.** A threshold re-derived from the
live distribution never terminates, which is the same reason a percentile was rejected.

The numbers live as data in a committed configuration file rather than as literals in the job's
script, so a change is one reviewed line.

### Candidacy re-measures, and only the refile clause reads a committed number

The job measures the tree itself. It has no choice for candidacy: an Exemption is dropped **before**
the file's maximum is taken, and a baseline row records the maximum alone, so the per-definition
numbers an Exemption keys on exist in no committed file. The same run supplies the rest of the body,
which the baseline also does not carry — the worst definitions by name and line, and the reports by
kind and site.

The refile clause below is the one place a committed number is read, because the number a file
carried at a **past** commit exists nowhere else. The two sources agree whenever the baseline is
current, and the loop keeps it current: an issue closes on a commit, and one of the three closing
commits is a refreshed baseline.

The cost is one full JET measurement per run, so the job runs about as long as `JET.yml`. It is
weekly, and it buys a job that never files a file whose number has already fallen.

### The cap is on the open count

```text
open = count(label:code-health state:open)
room = 5 - open
```

A per-run cap is blind to throughput, so an unworked backlog would grow at a fixed rate. This cap
is self-limiting. If five are open the job files nothing, and the queue paces itself to what the
maintainer actually closes. The tracker never holds more than five of these at once.

### The ranking is JET first, then the excess ratio

```text
rank key = (has_reviewed_jet_report, max_excess)
  max_excess = max over metrics of value / threshold
```

A file with a reviewed-real JET report outranks every complexity-only file, because a suspected
defect outranks a refactor. The JET pool is **73** files, not the 1 to 10 that carry a suspected
*real* finding: a report no Dismissal covers counts as reviewed-real by arithmetic, and no
Dismissal has shipped yet. So the JET key orders most of the queue rather than draining in a run or
two, and it drains fastest as the first Dismissals land. The excess **ratio** is used rather than
the raw value, because a cyclomatic 12 and an argument count 12 are not the same distance past
their own thresholds.

### Deduplication searches `state:all`, so a file is filed once

The search is `label:code-health state:all` plus the exact path in the title. Any match, open or
closed, means skip. **This rule bounds the noise, not the cap and not the cadence.**

The lifetime total is the candidate pool: **80** files, measured by a dry run of the job on
`ce5abb6e0f`. Most of them are JET-only, and most of those reports belong to the single systematic
class the loop expects to close with a Dismissal rather than a fix. The pool shrinks as Dismissals
land, because a dismissed report stops making its file a candidate — but only for a file the job
has not yet filed. So the number falls fastest if the first Dismissals land early.

### A rise since the last issue closed refiles the file

One case escapes that rule. [ADR 0076](0076-the-code-health-pass-rule-is-a-ratchet.md) ranks
correctness above maintainability, so a JET fix that raises a complexity number is a correct
outcome, recorded with `refresh --accept-rise`. Under `state:all` alone the file was already filed
and closed, so it would carry its new, higher number for ever with no standing reminder.

So the rule has a second clause. A closed issue means skip **unless** the file's recorded number
now stands above the number it carried when that issue closed.

The job needs no new state to see that. It already holds the closed issue, so it holds `closed_at`.
It reads the file's baseline row at the last commit before that time and compares it against the
row at the checked-out commit.

```text
skip  if  a match exists  and  baseline_at(closed_at) >= baseline_now
file  if  a match exists  and  baseline_at(closed_at) <  baseline_now
```

Three alternatives were rejected. Carrying the old number in a **label** puts machine state where a
person can edit it by hand. A **fifth committed file** written by the job needs `contents: write`
and a commit onto `dev` from a workflow, a capability nothing here has. **Reading the issue body**
is forbidden above. Reading git is free, needs only `contents: read`, and works for a file that was
filed before this rule existed.

A refile is rare by construction: it needs a reviewed `--accept-rise` commit on a file whose issue
is already closed.

### An Exemption declares a number irreducible, and the maintainer approves it

An Exemption is keyed `(path, definition, metric)` and it **cites an approved Rationale**. It is
dropped **before** the file's maximum is taken, so it binds **candidacy only and never the
baseline**. A ruling is a commit, not a closed issue.

The citation reuses [ADR 0071](0071-a-dismissed-jet-report-is-keyed-by-file-kind-and-message.md)'s
mechanism in one shared namespace, so the two escapes follow **one rule**: **anybody may add an
Exemption or a Dismissal that cites an approved Rationale, and a new Rationale needs the
maintainer.** CI flags a diff that adds one. Free prose was rejected, because a required reason that
nothing obliges anyone to read is not an approval gate.

**17 constructor Exemptions ship before the job first runs**, over 19 offending definitions, all on
argument count alone and all citing one shared Rationale: a struct's inner constructor takes one
argument per field, because Julia gives it no other form. Their files stay candidates on the other
two metrics, where the number comes from validation branches and a branch **is** extractable.

Re-affirmation on a Julia bump binds the JET side alone. A Rationale cited only by Exemptions claims
a syntactic fact that no analyser bump can falsify, and a CodeComplexity release that changed the
counting is already caught by ADR 0073's provenance comparison.

## Consequences

- A contributor without write access can lower a number, and can declare a stalemate whenever an
  approved Rationale already covers the case. Only a **new** Rationale needs the maintainer, which
  is exactly route 2 of the red-gate message in
  [ADR 0075](0075-a-run-that-trips-publishes-the-refresh-artifact.md).
- No `code-health` issue closes without a commit. The three committed outcomes are a number that
  fell, a Dismissal, or an Exemption. The procedure is in
  [`docs/src/contribute/3-code-health.md`](../src/contribute/3-code-health.md).
- The job reads git history, so its checkout needs the depth to reach the commit before the oldest
  `closed_at` it will consult. A shallow clone of depth 1 is not enough.
- The queue is finite and self-limiting, and the refile clause adds a bounded, reviewed trickle
  rather than a second source of noise.
