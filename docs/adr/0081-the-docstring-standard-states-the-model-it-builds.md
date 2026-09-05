---
status: accepted
---

# The docstring standard states the model a docstring builds, and a one-user field description may be prose

## Context

[#404](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/404) sweeps every file in
`src/` and `ext/` for three things at once: its documentation states the mathematics, its code
agrees with that statement when it is checked with real numbers, and its lines are covered or
exempted. Thirteen child maps carry the work. This ADR records the rules that the sweep applies,
so that thirteen maps apply one standard rather than thirteen.

Four gaps in the standard made a sweeping session invent its own answer.

**The JuMP layer is undocumented as a model.** `set_gross_budget_constraints!` registers the rows
`gbgt`, `gbgt_lb` and `gbgt_ub`, reads the shared variables `lw` and `sw`, homogenises with `k` and
scales with `sc`. Its docstring names none of that. Thirteen files sit under
`src/20_Optimisation/09_JuMPConstraints/`, and this is the layer in which `set_model_scales!` had
`sc` and `so` swapped for nine callers. A row name is public: a caller reads a row back out of the
model by the name in its `JuMP.@constraint` call.

**An encoding is not always exact, and nothing said so.** Four cases were already known when the
map was charted, and each surprised the reader who found it. `BrownianDistanceVariance` is a DCP
upper bound. The `Max` and log-sum-exp scalarisers put an upper bound in `model[:risk]`, which is
exact only while a minimising objective pulls on it. The SDP phylogeny penalty's `p` sets the
relaxation gap and not the constraint. The two `Squared` weight finalisers minimise the norm. In
each case the documentation stated the exact quantity, and the code built a bound on it.

**A procedure had nowhere to go.** `# Mathematical definition` holds a closed form. A type whose
content is a sequence of operations either compressed the sequence into prose, or left it out.
The opposite failure is as bad: `AbstractAlgorithm` is one of the four roots of the type hierarchy,
and most of its subtypes are selector tags that carry no procedure at all. A rule that forces
numbered steps onto a marker type makes the standard unusable in the place where most of the types
live.

**The field dictionary's rule was never written down.** `field_dict` is derived from `arg_dict`,
and the two exist to stop one sentence being pasted twice and then drifting. Neither the standard
nor `CLAUDE.md` said when a field description must interpolate a key, so
[#351](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/351) left the question open
after sweeping 719 type docstrings. Measured in `src/` while #404 was charted: 1230 field
docstrings interpolate a key, and up to 368 over 48 files are prose, of which 121 are in
`src/25_Aliases.jl` alone. A blanket "always interpolate" rule turns 368 prose sentences into 368
new dictionary keys, most of which would have one user each.

`ext/` is the last gap. It carries the plot recipes and **zero docstrings**, and the standard's
Scope stopped at `src/`.

## Decision

The Authority for all four rules is
`.github/instructions/julia-docstrings.instructions.md`. `STANDARDS.md` routes to it.

**1. `# JuMP formulation`.** Any code that adds rows to a `JuMP.Model` carries this section. It
sits after `# Mathematical definition` and after `# Algorithm`, and before `# Fields` or
`# Arguments`. It has three subsections:

- `## Variables` — one bullet per model variable that the code reads or creates.
- `## Constraints` — **one bullet per row**, in the order the body registers them, each carrying
  **the row's JuMP name** and the row's mathematics, closed by a `Where:` list.
- `## Relaxation` — present **only when the encoding is not exact**. It opens with
  `$(val_dict[:relax])` and then states the direction of the bound, the quantity bounded by its
  model key, and the condition under which the bound is tight.

**2. `# Algorithm`.** A procedure is documented as numbered steps, one step per operation, each
step naming the quantity that it produces. A closed form stays in `# Mathematical definition` and
is not restated as a step. A selector tag carries **neither** section, and its summary sentence
states which branch it selects.

**3. A field description interpolates when it has more than one user.** A description used by more
than one field must interpolate `field_dict`. A description used by exactly one field may be prose.
When a prose description gains a second user, it moves into `arg_dict` and both copies become the
interpolation. A new description is added as a **new** `arg_dict` key, because editing a value
already in `arg_dict` moves every docstring that interpolates it.

**4. The Scope is `src/`, `ext/` and `docs/`.** A package extension is documented on the same terms
as `src/`.

Two dictionary keys in `src/01_Base.jl` serve the new sections: `val_dict[:relax]`, the fixed
opening of a `## Relaxation` subsection, and `math_dict[:sc_scale]`, the constraint scale that
every scaled row carries. `math_dict[:k_budget]` already held the homogenisation variable.

## Consequences

**The rules are written before they are applied.** No docstring in `src/` carries the two new
sections today. The child maps of #404 apply them file by file, and a file's `swept` flag in the
sweep manifest is what makes the new rules binding on that file. A presence gate that demanded the
sections on day one would red the build across every selector tag in the library.

**The gate follows the standard, not the reverse.** `test/test_26_docs.jl` reads `src/` only, so
the widened Scope is unenforced in `ext/` until
[#409](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/409) teaches it about `ext/`.
This is a known unenforced state, in the sense of `STANDARDS.md`, and not a hidden one.

**The prose exception keeps 368 sentences out of the dictionary**, and pays for it with a rule that
holds only while a second user is noticed. The migration it does demand is the reverse one: a prose
sentence that a second field starts to share moves into `arg_dict` at that moment.

**A `## Constraints` list is prose, not data.** Nothing compares the row names in a docstring with
the names that the body registers through `JuMP.@constraint`. That census was offered while #404
was charted and set aside as the largest build; it stands in the map's *Not yet specified*, to be
revisited once enough `# JuMP formulation` blocks exist to show whether they drift.

## Amendment (2026-08-22)

The gate this decision promised is built, and it is two checks rather than one.

The Consequences above say that "a file's `swept` flag in the sweep manifest is what makes the new
rules binding on that file". Nothing read that flag for either section.
[#437](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/437) built the gate as the
`Swept file section completeness` testset of `test/test_26_docs.jl`, and it does not treat the two
sections alike. **The trigger, not the section, decides the shape of the check.**

**`# JuMP formulation` gets a per-unit presence rule.** A body that calls `JuMP.@constraint` or
`JuMP.@constraints` registers a row, and the parser sees the call, so the trigger needs no
judgement and no exemption list. Measured over `src/` and `ext/`, **82 functions register a row and
every one of the 82 is documented in its own file**, so the rule demands the section of 82
docstrings and of nothing else.

The attribution is **per name, not per definition**, because a docstring is often not on the method
that registers the row. `04_WeightConstraints.jl` documents a dispatch-error stub and the methods
that register `w_lb` and `w_ub` follow it undocumented; `06_XatRiskConstraints.jl` documents five
separate methods of `set_risk_constraints!`, each registering its own rows. A docstring therefore
speaks for its own definition and for every later definition of that name until the next docstring
of that name. Read per definition instead, four of the thirteen files under `09_JuMPConstraints/`
register rows under no documented definition at all.

**`# Algorithm` gets a per-file floor.** This decision's own exemption is a *selector tag*, and no
parser rule defines one, so a presence rule cannot be written without first defining a selector tag
mechanically. A row marked `swept = true` in `sweep/manifest.toml` therefore carries one more key,
`algorithm = N`, and the measured count of the file's `# Algorithm` sections may not fall below it.
It is a floor and not an equality, because an equality would red an *improvement*: adding the
section to an existing unit adds no unit, so the row's `units` count would not move and the only
red would ask for a number to be updated. The floor leaves the stronger rule available — a later
ticket can raise it to a presence rule without moving the key.

**The gate covers the trigger this decision wrote, and no more.** 40 further documented functions
touch the model without registering a row: they create a variable, register an expression, or set
the objective. `set_model_scales!` is one of them, and so are
`set_risk_constraints!` for `BrownianDistanceVariance` and the scalarisers — **two of the four
`## Relaxation` cases this decision cites sit on the exempt side of the trigger it wrote**. Whether
the trigger should widen is a change to this decision, so it is raised as
[#443](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/443) rather than settled by a
gate that demands more than its Authority states. This is a known unenforced state, in the sense of
`STANDARDS.md`, and not a hidden one.

## Amendment (2026-08-24)

This decision wrote four rules.
[#478](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/478) changes one of them, and
it removes a section that this decision assumed.

**The `# Mathematical definition` boundary now points both ways.** Rule 2 above states one
direction only: a closed form stays in `# Mathematical definition`, and no step restates it.
Nothing stated the reverse, so nothing kept an implementation fact out of the mathematics.
`src/05_Denoise.jl` shows the cost. `SpectralDenoise` writes the sort order of the eigenvalues
into its mathematical prose, and the sort is the body's choice. `ShrunkDenoise` repeats its own
`# Algorithm` step 5 there. Neither breaks a rule that this decision wrote. #478 adds the reverse
rule. The section names no identifier from the body, states no order of operations, and states no
property that the implementation chose rather than the mathematics. A mathematical consequence of
the definition stays, so `ShrunkDenoise`'s sentence about the two `alpha` weights survives.

**`# Details` is no longer a section.** This decision's Scope covered the sections that it wrote,
and it left `# Details` as it found it. `# Details` carried no rule at all, in this ADR or at its
Authority, and the only text that described it was the placeholder `Additional implementation
notes.` in a template. 299 docstrings over 84 files carry it. #478 abolishes it and names four
destinations. [ADR 0085](0085-the-docstring-standard-is-rules-and-pointers.md) owns that decision
and its two gates.

**Rules 1, 3 and 4 stand.** The `# JuMP formulation` trigger, the one-user field prose exception
and the widened Scope are unchanged.
[#443](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/443) still holds the open
question that the 2026-08-22 amendment raised, and #478 re-parents it without answering it.

**A selector tag is never forced to carry a section, and it is not forbidden one.** Rule 2 above
reads *a selector tag carries neither section*. The Context that produced it says only that a rule
must never **force** numbered steps onto a marker type. The two are not the same sentence, and the
stronger reading contradicts the tree it governs: `SpectralDenoise`, `FixedDenoise`, `MaxValue` and
the four members of `AbstractDistanceAlgorithm` are all fieldless tags, and every one of them
states the closed form of the branch it selects.
[#485](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/485) sharpens the Authority to
the permission the Context intended. A tag that names nothing beyond its branch stops at its
summary sentence. A tag whose branch **is** a closed form states that form under
`# Mathematical definition`, with the steps of the branch under `# Algorithm`. `SpectralDenoise` is
the Authority's reference for that shape, and #478's own motivation asks for its mathematics to be
corrected rather than deleted.

**The Authority keeps its name and loses its worked example.**
`.github/instructions/julia-docstrings.instructions.md` remains the Authority for every rule above.
Its 380-line `## Complete Example` is deleted, because it is fictional, it can never be gated, and
it breaks this repository's summary-sentence rule twice in its own text. ADR 0085 records what
replaces it.

## Amendment (2026-08-24): the `# JuMP formulation` trigger covers the whole formulation

The 2026-08-22 amendment above raised a question and left it open. This amendment answers it.

**The trigger widens from a row to any model entry.** Rule 1 above reads *any code that adds
rows to a `JuMP.Model`*. It now reads: **any code that builds part of a `JuMP.Model`**, and the
mechanical form of that is a body that calls `JuMP.@variable`, `JuMP.@variables`,
`JuMP.@expression`, `JuMP.@expressions`, `JuMP.@constraint`, `JuMP.@constraints` or
`JuMP.@objective`.

The reason is this decision's own justification for the section. It says that the section exists
because *the rows carry names, a caller reads them back by those names*. **That is not a property
of a row.** `model[:sc]`, `model[:w]`, `model[:ret]` and `model[:risk]` are each registered as a
variable or as an expression, `src/` reads a model key back by name in 51 places over 16 distinct
keys, and `src/20_Optimisation/08_Base_JuMPOptimisation.jl` wraps nine of those keys in an
accessor that raises a named `ArgumentError` when its builder has not run. A row name is public,
and so is every one of those.

Measured over `src/` and `ext/` at the tip that widened it: 137 documented units call one of the
seven macros, 96 register a row, and 41 touch the model without one. Of the 41, 31 register only
an expression, 3 only a variable, 4 both, 2 only the objective, and 1 an expression and the
objective. They sit in 16 files, and **no file marked `swept = true` is one of them**, so the
widening reds nothing on the day it lands.

**The section gains `## Expressions` and `## Objective`, and no subsection is unconditional.**
Rule 1 said `## Variables` and `## Constraints` are *always present*, which would put an empty
`## Constraints` under a function that registers no row. Each of the four register subsections is
now present exactly when the body calls the macro that owns it: `@variable` owes `## Variables`,
`@expression` owes `## Expressions`, `@constraint` owes `## Constraints`, and `@objective` owes
`## Objective`. `## Variables` is also permitted when the body only *reads* a variable, because a
formulation that reads `w` and never names it is unreadable. One `Where:` list closes the last
subsection present and serves the whole section.

An expression had no home before this. `set_model_scales!` registers `sc` and `so` and is the
function whose two scales were swapped for nine callers — the defect #404's charter names when it
says that the JuMP layer is undocumented as a model — and under the old rule its docstring had
nowhere to say so.

**`@objective` is in.** A mathematical program is variables, constraints and an objective, and the
third was missing from a section named *formulation*. The case that settles it is
`owa_l_moment_crm_sumsq_obj` in `src/19_RiskMeasures/10_OWARiskMeasures.jl`: two methods that
differ in `Min so * t` against `Min so * t^2` and in nothing else. Only an objective bullet tells
them apart.

**`## Relaxation` is unchanged in shape and widened in reach.** The bound does not have to sit in
a row. `BrownianDistanceVariance` relaxes inside an expression, and the `Max` and log-sum-exp
scalarisers put an upper bound in `model[:risk]`. Those are two of the four cases this decision
cites, and both sat on the exempt side of the trigger it wrote. `val_dict[:relax]` therefore reads
*the entries below bound the quantity* rather than *the rows below*. No docstring interpolated it
yet, so the wording moved at no cost.

**The gate widens with the standard, and gains a subsection check.** The `Swept file section
completeness` testset of `test/test_26_docs.jl` reads a macro-to-subsection table rather than a
constraint-macro tuple. It demands the section of a unit whose definitions call any of the seven,
and then demands each subsection that unit's own macros name. The widening keeps the property that
made the narrow rule worth building: a parser sees a macro call, so the trigger needs no judgement
and no exemption list.

**What is still ungated.** The *content* of a subsection. Nothing compares the model keys a
docstring names with the keys the body registers, and nothing reads `## Relaxation` at all — an
inexact encoding is a fact about the mathematics and not a token. The key census stands in the
map's *Not yet specified*, as the Consequences above already record. `## Relaxation` holds by
review, in the sense of `STANDARDS.md`. This is a known unenforced state and not a hidden one.

`math_dict` gains `:so_scale`, the objective scale, beside the `:sc_scale` this decision added.
