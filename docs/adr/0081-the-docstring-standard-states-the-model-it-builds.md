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
