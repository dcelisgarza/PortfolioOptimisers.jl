---
status: accepted
---

# A run that trips publishes the Refresh Artifact that clears it

## Context

The ratchet is a gate a contributor is expected to act on. Two earlier decisions left one class of
contributor with no way to act.

[ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) fixed that a refresh **lowers by
default**, that recording a rise needs a second named act, and that **any provenance mismatch turns
the gate red**. The provenance includes the Julia version, so a local refresh needs the exact pinned
Julia even for the 4.8 s complexity measurement.

The gate's shape decision then pinned the tools to `code_health/Manifest.toml` and gave the
workflows one artifact rule: a **green** run uploads the fresh numbers **on `workflow_dispatch`
only**.

Together those left a fork contributor whose number legitimately rose with two routes, and neither
one works from a fork.

| Route | Why it fails from a fork |
| --- | --- |
| Run the refresh locally | 5 minutes 46 seconds and a 2.4 GiB peak for JET, and every route needs Julia 1.12.7 |
| Take the artifact from a `workflow_dispatch` run | `workflow_dispatch` needs write access to the repository |

Four facts constrain the answer.

1. **A fork pull request already runs a tool-installing quality gate here.** `Aqua.yml` and
   `FormatCheck.yml` both trigger on a bare `pull_request` with `permissions: contents: read`, and
   both `Pkg.add` a tool from the public General registry. A registry install needs no token, so
   `contents: read` is enough to measure.
2. **A fork pull request can receive a file.** `actions/upload-artifact` authenticates with the
   Actions runtime token, not with `GITHUB_TOKEN`. The artifact attaches to the run in the base
   repository, and this repository is public, so anyone can download it from the run page.
   `workflow_dispatch` blocks the **trigger**, not the **receipt**.
3. **A fork contributor cannot re-run a workflow.** Re-running needs write access. After a transient
   failure their only lever is a new push.
4. **No check is required in this repository.** The ruleset "Protect Main" carries `deletion`,
   `non_fast_forward`, `update`, `code_scanning` and `pull_request` with
   `required_approving_review_count: 0`. There is no `required_status_checks` rule.

## Decision

### The gate measures a fork pull request, and no workflow tests for a fork

Both workflows keep their bare `pull_request` trigger and their `contents: read` ceiling. Nothing in
either file reads `github.event.pull_request.head.repo.full_name`.

Guarding the gate on `github.repository`, as the scheduled job is guarded, was rejected. That guard
is right for a job that **writes** — the scheduled job opens issues. This gate only measures, and a
gate a contributor cannot see before the merge moves every refresh onto the maintainer by
construction.

### A run that trips publishes the Refresh Artifact

A **Refresh Artifact** is the generated baseline file as `refresh --accept-rise` would write it,
uploaded from the run that failed. A contributor downloads it, puts it at its committed path,
commits it, and the gate goes green. They run no analyser and they need no Julia.

One rule covers both workflows.

| Workflow | Refresh Artifact |
| --- | --- |
| `JET.yml` | `jet_baseline.toml` |
| `Complexity.yml` | `complexity_baseline.toml`, `expansion_bound.toml`, or both |

Serving only the 5 minutes 46 seconds route was rejected. The complexity measurement is cheap in
time alone: ADR 0073's provenance rule binds it to Julia 1.12.7 as tightly as it binds JET. A
JET-only rule leaves the pinned-Julia install on the contributor's path and needs two failure
messages instead of one.

Publishing raw measurement, for the contributor to accept with a new script mode that reads a file
instead of running the tool, was rejected. It preserves the named act at the cost of one more mode,
and the act it preserves is a single command — which is the thing ADR 0073 called unremarkable.

### The artifact is whatever the refresh writes, and a provenance mismatch writes nothing

A run publishes whenever a refresh is a legitimate fix, and it publishes exactly what
`refresh --accept-rise` wrote. No rule decides publication case by case.

| Failure | Publishes | Why |
| --- | --- | --- |
| The ratchet trips | yes | the refresh records the rise |
| A row names a file that is gone | yes | the refresh drops the row, or pairs it with a rename |
| A file in scope has no row, and it passes candidacy | yes | the refresh adds the row |
| A file in scope has no row, and it fails candidacy | **no** | the refresh itself errors, so no file exists |
| A tracked file is neither in scope nor an Unmeasured Path | yes | but the artifact does not clear it |
| The provenance does not match | **no** | the numbers came from the wrong tools |

The candidacy row is
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md)'s rule reaching
this one. An added file over a threshold stops the refresh, so there is nothing to upload, and the
two fixes it names — lower the number, or declare an **Exemption** — are a source edit and a
`rulings.toml` edit. Neither is a generated file. The rule needs no exception written for it: a run
publishes what the refresh wrote, and here the refresh wrote nothing.

The coverage row differs. ADR 0072's assertion is checked, not refreshed, so the refresh still
writes valid baselines and the run still publishes them. The message says plainly that the artifact
alone will not turn the gate green, because the fix is an **Unmeasured Path** entry, and a human's
reason cannot come out of a generator.

A provenance-mismatched run measured with the wrong Julia, the wrong JET or the wrong load set, so
its numbers are worthless. Publishing them behind a warning name was rejected: it puts a poisoned
baseline one download away from a commit, and only a filename stands between them.

### The local spelling stands, and the diff is the protection

`refresh --accept-rise` remains the local act, unchanged. The Refresh Artifact is the same act
performed by CI for a contributor who cannot perform it.

ADR 0073 added the flag to close the hole where a contributor whose build is red goes green with one
unremarkable command. Publishing a drop-in file reopens that hole for everyone, and the decision
accepts it, because the flag was never what closed it. **A risen number is a changed line in a
committed TOML file, visible in the pull request, whichever route wrote it.** The reviewer's control
is the diff, and the diff is unchanged.

### The gate measures the merge commit

Both workflows keep the bare `actions/checkout@v7` that `Aqua.yml`, `FormatCheck.yml` and
`TestOnPRs.yml` all use, so a pull-request run measures the merge of the head with the base. A gate
gates what will land.

One consequence is named and accepted. ADR 0073 fixed that a refresh rewrites a **whole** generated
file, so a Refresh Artifact carries a row for every file in scope, including files the contributor
never touched. When the base branch is **already red** — a rise reached it without a refresh — that
row is in the artifact, and committing the artifact records a rise the contributor did not cause.
The hole opens only when the base is already red, and it closes when the base is refreshed.

Pinning the checkout to the head was rejected: the artifact would then match a local refresh exactly,
but the gate would stop seeing a rise that only the merge creates. Measuring twice, the merge for the
verdict and the head for the artifact, was rejected on cost: it makes `JET.yml` about 11 minutes 30
seconds.

### One message, three ordered routes

A trip prints one `::error` annotation per offending file and a `$GITHUB_STEP_SUMMARY` table. The
summary then gives the routes, in order of preference, in one text that everybody reads.

```text
## JET ratchet tripped

| file | baseline | measured |
| --- | --- | --- |
| src/08_Moments/01_Base_Moments.jl | 3 | 5 |

Three ways to green, in order of preference:

1. Lower the number. Fix what JET found.
2. Dismiss it. Add a Dismissal to code_health/rulings.toml citing an APPROVED
   Rationale. A NEW Rationale needs a maintainer -- say so in this pull request
   and one will rule.
3. Record the rise, if it is legitimate. Download the `jet_baseline` Refresh
   Artifact from this run, put it at code_health/jet_baseline.toml, and commit
   it. The risen row will be visible in your diff. Locally the same act is
   `julia code_health/jet.jl refresh --accept-rise`.
```

The complexity message drops route 2, because a Dismissal is a JET-only instrument.

Route 2 is named in the message rather than left to the developer documentation, because it is the
one route a fork contributor cannot finish alone. A new Rationale needs the maintainer, and a
contributor should meet that in CI rather than in review.

A message that branches on whether the pull request came from a fork was rejected. It puts back the
fork test that the first decision removed, and it makes two audiences read different instructions
for one gate.

## Consequences

**A fork contributor needs no Julia to clear any gate.** Every route to green that does not change
`src/` is a download and a commit.

**The Refresh Artifact does not always clear the gate on its own.** Two cases need a `rulings.toml`
edit beside it. A renamed file's JET row is carried with its **old** reviewed number, so the gate
stays red until the Dismissals name the new path, and
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md) has the refresh
print that repair rather than make it. A coverage failure needs an Unmeasured Path entry. In both,
the artifact is correct and worth committing. It is not the whole fix, and the message says so.

**The `workflow_dispatch` clause survives untouched, and nothing needs it.** A green run still
uploads its numbers on `workflow_dispatch` only. The only run left with no artifact is a green one,
which needs none.

**Both workflows gain an upload step that runs on failure**, and `Complexity.yml` uploads whichever
of its two generated files the run changed.

**The developer documentation gains a section**, "When a code-health gate turns red", written when
the gate ships rather than before it. Until then no branch carries a `JET.yml` for the text to
describe.

**Making either gate a required check stays out of the gate's scope.** Branch protection is a
repository setting, not a file. Two facts belong to whoever configures it later. A workflow skipped
by a `paths:` filter leaves its check **Pending**, not skipped, and a pending required check blocks
the pull request; GitHub documents the fix as a companion workflow with the same job name on the
inverse paths. `JET.yml` carries a `paths:` filter and is exposed; `Complexity.yml` carries none and
is not. Separately, a fork contributor cannot re-run a check, so a transient failure needs a new
push from them or a re-run by a maintainer.

**The vocabulary stays out of `CONTEXT.md`**, for the reason ADRs 0071, 0072 and 0073 all gave. That
file's preamble scopes it to the library's domain, and a published artifact is repository process.

**"Published Baseline" and "Remedial Baseline" were considered and rejected.** "Published Baseline"
reads redundantly against its own verb, and the gate's shape decision already uses "publish" for the
provenance block a green run prints. "Remedial" composes with no other word in this map's
vocabulary, while "refresh" is already the name of the act that writes the file.
