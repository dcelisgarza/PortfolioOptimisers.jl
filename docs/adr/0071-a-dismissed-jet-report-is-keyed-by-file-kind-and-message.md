---
status: accepted
---

# A dismissed JET report is keyed by file, kind and message

## Context

The JET and complexity gate records **two numbers per file**: the raw report count and the
reviewed-real count. The ratchet binds on the reviewed number, and the raw number is context.
That split only works if a human judgement of "this report is not a real defect" can be attached
to something that survives an ordinary edit.

JET does not hand you one. `reportkey(report) = (report.sig.tt, report.vst[end].linfo)` is
documented for `unique(reportkey, reports)`, which deduplicates inside a **single run**. A
`MethodInstance` does not serialise, and a call tuple type printed to a file is long and
version-sensitive. JET 0.12 removed the `.JET.toml` file that earlier versions used, and the seven
`ReportMatcher` types are boolean predicates with no file matcher and no line matcher, so they
cannot hold a per-report judgement either.

Two measurements constrain the answer. Both come from two fresh processes on the same Julia, the
same JET and the same load set.

1. **A line number is useless as an identity.** It moves whenever anything above it changes. The
   mainstream identity used by static-analysis baselines outside Julia is the path, the rule and a
   count, and it carries no line.
2. **A stack trace already flickers without a source change.** Of 315 reports, all 315 appeared at
   the same index in both runs, and the multiset of kind, reported site and attribution site was
   identical. Exactly one report held **7 frames in one run and 8 in the other**. An identity built
   on frames is therefore not stable even between two runs of the same code.

The scale matters for the design. The package emits 315 reports over 68 source files. About 180 of
them are a **single systematic class**: a `getfield` call on a container that inference narrowed to
`Union{}`, which arises from an abstract element type such as
`AbstractMatrix{<:Union{Number, JuMP.AbstractJuMPScalar}}`. The honest estimate of real findings is
one to ten. So the mechanism must make the systematic case cheap, or nobody will ever use it.

## Decision

### A Report Fingerprint holds three components and no site

A **Report Fingerprint** identifies a *problem*, not a report. Many reports share one.

| Component | Source |
| --- | --- |
| The attribution file | the attribution rule the baseline uses |
| The report kind | the `InferenceErrorReport` subtype |
| The report message | `JETInterface.print_report_message` |

The line, the stack trace and the **signature render** are all excluded. The last exclusion is the
one that does the work, and JET's own interface makes the cut for us: `print_report_message` and
`print_signature` are two separate interface functions, and `print_signature` is an overridable
predicate returning a `Bool`. A rendered report concatenates them:

```text
no matching method found `-(::Nothing, ::Int64)` (1/2 union split):
  (PortfolioOptimisers.findfirst(#23, name)::Union{Nothing, Int64} PortfolioOptimisers.:- 1)
```

The first line is the message. The second is the signature render, and every fragile piece of text
lives there. `#23` is a compiler-generated name whose number comes from lowering order rather than
from the source text. `name` is a local variable that a rename can change. Neither carries meaning
that a reader of a dismissal needs.

The message alone keeps the substance for every kind the package emits:

```text
invalid builtin function call: getfield(x::FillArrays.Fill{Union{}, 1}, f::Symbol)
no matching method found `convert(::Type{TopologyOnly}, ::Nothing)` (1/4 union split)
```

Because the fragile text is cut rather than masked, **no normalisation rule exists to write, test or
maintain**.

### A Dismissal covers a class, and its count does not bind

A **Dismissal** records that every report matching one Report Fingerprint is not a real defect. A
report whose fingerprint carries no Dismissal counts as **real**. Nothing ever records "this is
real", so the record is one-sided.

The gate computes `reviewed = raw − matched dismissals`. A Dismissal matches **every** report with
its fingerprint. The count written beside it is for a human reader and does not bind.

Take a file with `raw = 21` and one dismissed class at `count = 14`, so `reviewed = 7`. A commit
adds a fifteenth instance of that same class:

| Rule | raw | dismissed | reviewed | Gate |
| --- | --- | --- | --- | --- |
| The class binds | 21 → 22 | 14 → 15 | 7 → 7 | green |
| The count binds | 21 → 22 | 14 → 14 | 7 → 8 | red |

The class rule is chosen. The systematic `getfield` class grows whenever new code takes an abstract
element type, so a count-bound rule would turn CI red for a report already judged not real, during
ordinary work.

### Every Dismissal cites a shared, named Rationale

A **Rationale** is a named paragraph saying why a class of reports is not a real defect. It is
written once and cited by many Dismissals.

The fingerprint holds the file, so one systematic class needs **one Dismissal per file it touches**.
The `getfield` class alone needs dozens. A free-text reason on each entry would copy one paragraph
dozens of times, and the copies would drift.

This split also fixes the trust boundary. Reusing an approved Rationale is bookkeeping. Inventing
one is a claim that some code is correct.

| Change to the file | Who |
| --- | --- |
| A Dismissal citing an approved Rationale | any contributor, ordinary review |
| A new Rationale | the maintainer; CI flags a diff that adds a rationale block |

The rule is mechanically enforceable, because a diff that adds a rationale block is easy to detect.

**An Exemption cites a Rationale in the same namespace.** The complexity side of the gate has the
same shape of problem: a class of definitions whose number cannot fall, and one sentence that
explains all of them. [ADR 0073](0073-the-code-health-baseline-is-four-toml-files.md) writes 17
Exemptions for the struct inner constructors, all of one class, so a free-text reason on each entry
would copy one paragraph 17 times. Reusing the Rationale gives the Exemption the approval gate it
otherwise lacks, where any contributor could write any prose and no rule asked anybody to read it.

One namespace was chosen over two. A Rationale is a named, approved paragraph that many keyed
entries cite, and what cites it is evident from where the citation sits.

### The record is a committed data file, never an annotation in the source

Dismissals and Rationales live in a committed file outside `src/`, in a standard machine-parseable
format. A bespoke text format needing its own parser is not acceptable.

An annotation at the reported site was rejected for two independent reasons. It changes `src/`,
which the gate is built without doing. And an annotation lives **at a line**, so it would restore
the line to the identity through the back door.

### A broken fingerprint always fails closed

A fingerprint breaks when the message text changes, after a dependency or Julia bump, or when the
file is renamed.

The failure direction is **not a policy choice**. It follows from the subtraction. A Dismissal that
stops matching contributes zero, so the reviewed number rises and the gate turns red. A broken
identity cannot silently relax the gate, and no rule has to be written or enforced to make that
true.

### A dead baseline row fails; a dead Dismissal does not

Three file-level events behave differently.

| Event | Gate | Why |
| --- | --- | --- |
| The file is deleted | **red**, and the row must be pruned | keeps the baseline honest at no extra cost |
| The file is renamed | **red** until the Dismissals name the new path | reports reappear unmatched under the new path |
| A Dismissal matches nothing, file still present | **green**; the scheduled job prunes it | the class went away because someone improved the code |

The asymmetry is deliberate. A rename already forces an edit to add the new path, so requiring the
old row to go adds no separate chore. But a Dismissal stops matching when a whole noise class
disappears, and the gate must not turn red on exactly the work it exists to encourage. That case
cannot relax the gate either: if the message merely drifted, the reports return unmatched and the
gate turns red anyway.

The rename row is refined by
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md). The baseline
row itself no longer needs a hand edit: the refresh pairs the dead row with the new path on the
**raw** count and carries it. The red stands, because the carried row keeps its old reviewed number
while the measured one has risen, and it clears when the Dismissals name the new path. The refresh
prints the exact lines to edit and never writes `rulings.toml` itself.

The rule for a file the baseline does **not** name is settled by ADR 0074.

### A Julia bump migrates the Dismissals and re-affirms the Rationales

Dismissals carry over untouched. The subtraction already turns the gate red wherever a message
actually changed, so the bump commit fixes exactly what moved.

Each Rationale a Dismissal cites records the Julia version it was last affirmed against, and the
gate stays red until every such Rationale names the new pin. A new inference engine can invalidate a
claim that some code is correct without changing one character of source, and only the Rationale
layer carries such claims.

**Re-affirmation binds the JET side alone.** A Rationale cited only by Exemptions records no pin. It
claims a syntactic fact — an inner constructor takes one argument per field — and no analyser bump
can falsify it. A CodeComplexity release that changed the counting is already caught by the
provenance comparison, which fails the gate and forces a refresh. Making every Rationale re-affirm
was rejected as a ritual that can never fail.

The counts make this affordable. Dismissals will number in the hundreds. Rationales will number
under ten, so a bump costs a human under ten paragraphs to re-read.

Discarding the reviewed set wholesale on a bump was rejected. Re-triaging some 315 reports at every
bump means the version is never bumped and the gate rots.

## Consequences

**The vocabulary stays out of `CONTEXT.md`.** That file's preamble scopes it to the library's
domain, the workflow from data to post-processing, and states that decisions live here. A
static-analysis dismissal is repository process and sits beside no entry in any of its sections.
The three nouns are defined in this ADR alone.

**Two reports that share a file, a kind and a message merge into one fingerprint.** They are judged
together and dismissed together. This is the price of an identity that survives an edit, and under
the class rule it costs nothing extra.

**A dismissed class can grow without bound and nothing turns red.** The raw number records the
growth as context, which is the role it already had.

**A contributor who meets genuinely new noise is blocked** until a Rationale exists for it, because
inventing one is the maintainer's call.
[ADR 0074](0074-the-baseline-row-set-is-total-and-a-rename-pairs-by-measurement.md) confirmed this
consequence rather than softening it, because its entry test needs a new file to arrive with
`reviewed = 0`.

**"Verdict", "class", "signature" and "suppression" were all considered and rejected.** "Verdict"
implies a two-sided judgement, and nothing ever records "this is real". "Class" sits one word from
"kind", which is a component of the fingerprint. "Signature" is JET's own name for the offending
expression, which is the exact part cut out of the key. "Suppression" says the report is hidden,
whereas the claim being made is stronger: there is no defect.

## Amendment (2026-08-22)

**For a `BuiltinErrorReport`, the message component is JET's message followed by the builtin.**
Every other kind keeps `print_report_message` unchanged. Issue #357.

### The first example above is not a message

The body gives this as an example of "the message":

```text
invalid builtin function call: getfield(x::FillArrays.Fill{Union{}, 1}, f::Symbol)
```

It is not one. `JETInterface.print_report_message(io, r::BuiltinErrorReport) = print(io, r.msg)`
(`src/analyzers/jetanalyzer.jl:1146` in JET 0.12.1), and the usual `r.msg` is the constant
`GENERAL_BUILTIN_ERROR_MSG = "invalid builtin function call"`. Everything after the colon is the
signature render, which this ADR cuts on purpose. The second example is correct:
`MethodErrorReport` does carry the call signature and the union-split marker in its message.

### One constant covered five different builtins

Measured against the committed baseline, over all three runs of the load set:

| Fact | Value |
| --- | --- |
| `BuiltinErrorReport` reports | 200 of 364 |
| Carrying the bare constant | 175 |
| Builtins behind that one constant | `getfield` 163, `memoryrefget` 5, `typeassert` 4, `apply_type` 2, `fieldtype` 1 |
| Carrying a richer message | 25, each with exactly one builtin |
| Distinct fingerprints, before and after | 176 → 179 |
| Files where one Dismissal covered two builtins | 3 |

So a Dismissal keyed `(file, "BuiltinErrorReport", "invalid builtin function call")` covered every
general builtin error in the file, not the `getfield` class the body describes. It still failed
closed, because `reviewed = raw − matched dismissals` never rises above `raw`. The cost was the
other direction: a new builtin report of a different shape, in a file already carrying that
Dismissal, would have been absorbed with no signal. That is the one place the instrument could
lose a finding silently.

### `r.f` is the discriminator, and it re-admits nothing

`BuiltinErrorReport` declares two fields, `f` and `msg`. `r.f` is the builtin itself, and it
renders as `getfield`, `memoryrefget` or `bitcast`. It carries no gensym, no local variable name
and no type render, so appending it narrows the class without restoring any of the fragile text
this ADR cut. The fingerprint stays at three components and `rulings.toml` keeps its shape.

The append is unconditional for the kind. A rule that fired only on the bare constant would name a
JET internal string inside `code_health/jet.jl`, which is the standing chore this ADR avoided when
it cut the signature render instead of masking it. The measured cost of firing always is four
reports: JET's intrinsic message already opens with the builtin, so `bitcast: target type not a
leaf primitive type` becomes `bitcast: target type not a leaf primitive type: bitcast`.

The kind is matched by its **name**, not by its type. A type in a method signature resolves at load
time, and JET's stubs define no such type on an unsupported Julia, so a signature would replace
`assert_environment`'s deliberate message with an `UndefVarError`.

### What this amendment does not cover

`UnsoundBuiltinErrorReport` carries the same shape — a constant message and a field `f` — but JET
emits it only from `_report_builtin_error_sound!`, which runs in `:sound` mode. `report_package`
runs `:basic`, so the kind cannot reach this gate's baseline and no rule is written for it. A
future JET kind whose message is a bare constant gets its own ticket, the way this one did. The
alternative, a general rule over any kind whose message is constant, was rejected: it needs a human
to re-read every kind's message at each JET bump, and this ADR does not create standing chores.

### No number moved

`rulings.toml` carries no Dismissal, so `reviewed` equals `raw` for every file in every run. The
fingerprint binds the subtraction alone, so changing it left all 588 baseline rows unchanged —
196 files times three runs.
The change was therefore free at the moment it was made, and it would have been a migration after
the first Dismissal was written.
