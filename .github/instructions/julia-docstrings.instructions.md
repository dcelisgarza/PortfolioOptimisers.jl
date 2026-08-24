---
applyTo: 'src/**/*.jl, ext/**/*.jl, docs/**/*.md'
---

# Docstring and Documentation Guidelines for PortfolioOptimisers.jl

## How to read this file

This file is the Authority for docstrings, in the sense of [`STANDARDS.md`](../../STANDARDS.md). It carries the rules, and it names where to read a real docstring. It holds no worked docstring of its own: a copy inside a standards file drifts away from the code, and a pointer cannot.

Three kinds of block appear below.

- **A rule** is prose or a bullet list. It is normative.
- **A template** is a fenced block marked `**Template.**`. Its section names, its order and its shape are normative. The identifiers inside it are invented, and they are not.
- **An example** is a fenced block marked `**Example.**`. It illustrates the rule beside it, and it is not normative.

A fenced block inside a bullet belongs to that bullet.

To read a complete docstring, open a Unit that [Reference docstrings](#reference-docstrings) names.

## Vocabulary

Five words are used precisely in this file.

- **Unit** — one documented name: a module, a type, a function, a macro or a constant. [`sweep/manifest.toml`](../../sweep/manifest.toml) records the unit count of every file under `src/` and `ext/`, and `test/test_45_sweep_census.jl` fails when a file's count leaves its row.
- **Family** — a leaf abstract supertype, together with the concrete types that subtype it. Leaf-most means that no abstract type subtypes it, so a generic root such as `RiskMeasure` or `AbstractResult` is not a Family: its members span many files and share no notation. Most families sit inside one file, so a rule about a Family is local to one sweep ticket. [Notation is fixed by symbol and by family](#notation-is-fixed-by-symbol-and-by-family) is such a rule.
- **Reference docstring** — a docstring that the [Reference docstrings](#reference-docstrings) table names. Its file is marked `swept = true` in the sweep manifest, so a Gate holds it. Read one in place of a worked example.
- **Capability Catalogue** — the user-facing inventory of everything the package offers, built by [`docs/capability_catalogue.jl`](../../docs/capability_catalogue.jl) under ADR 0040. It extracts the first sentence of a type's summary paragraph verbatim.
- **Coverage Exemption** — a named ruling in [`code_health/rulings.toml`](../../code_health/rulings.toml) that excuses a stated number of uncovered lines in one file. [ADR 0082](../../docs/adr/0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md) owns it.

## General Guidelines

- Look at how other docstrings are implemented and follow similar patterns.
- Write clear and concise documentation.
- Use consistent terminology and style.
- Include code examples where applicable.
- **All public types, functions, and macros must have docstrings.**
- **Scope**: `src/**/*.jl`, `ext/**/*.jl` and `docs/**/*.md`. A package extension in `ext/` is documented on the same terms as `src/`.
- **An extension documents the names it declares itself, and nothing else.** An extension implements a seam that `src/` declares, and the declaration carries the docstring, the `# References` section and the API-page entry. So a method of a function declared in `src/` gets no docstring of its own in `ext/`, and the extension's own module, constants, types and macros each get one. Write a citation in the `src/` declaration, where the API page that renders it carries the bibliography block; an extension needs neither an API page nor a bibliography block of its own. `test/test_26_docs.jl` gates this per file, through the `swept` flag in [`sweep/manifest.toml`](../../sweep/manifest.toml).

## The summary sentence (load-bearing — read this before writing a type docstring)

Every type docstring opens with `$(DocStringExtensions.TYPEDEF)`, a blank line, then a **summary paragraph**. The **first sentence of that paragraph is extracted verbatim** and rendered as the type's one-line description in the [Capability Catalogue](../../docs/capability_catalogue.jl) (ADR 0040), which is the user-facing inventory of everything the package can do.

This is a real contract, not a convention that merely happens to hold. It is what lets the catalogue carry prose without keeping a second, drifting copy of every description. If a docstring loses its summary paragraph, the docs build fails with an error naming the type.

Write the first sentence so it stands alone in a bullet list:

- **Lead with what it does**, in the active voice. `Denoises by setting the smallest \`num_factors\` eigenvalues to zero.` — not `A denoising algorithm that sets...`.
- **Keep it under ~120 characters.** If the idea needs more, put a crisp first sentence and move the detail into a *second* sentence, which still renders on the API page but not in the catalogue. Do not compress by deleting information.
- **Avoid filler openers**: `A flexible container type for...`, `A concrete estimator type for...`. They cost a line and say nothing.
- **Do not append `in \`PortfolioOptimisers.jl\``.** Every docstring in the package is in`PortfolioOptimisers.jl`.
- **Do not put `@ref` links in the first sentence.** The catalogue appends the type's own links after the description, so a link in the summary renders twice. Put cross-references in a second sentence or in `# Related`.
- **Do not open with a display formula.** Inline maths is fine; a full `$...$` equation belongs in `# Mathematical definition`.
- **Never leave a bare `_` outside a code span.** Markdown reads `_` as emphasis and will pair it with the underscore inside a neighbouring `` `snake_case` `` link, eating both and destroying the link. `(f_μ vector)` sitting next to `` [`plot_factor_mu`](@ref) `` rendered as ``(fμ vector … [`plotfactor_`` — a dead link that Documenter cannot resolve and the site builder reports only as a single anonymous `./@ref`. Write `` `f_mu` `` instead.
- **Siblings should not all share a prefix.** If every algorithm in a family starts `Centrality algorithm type for ...`, the catalogue shows that boilerplate eight times over. Say what distinguishes each one.

## Grammar

- Use present tense verbs (is, open) instead of past tense (was, opened).
- Write factual statements and direct commands. Avoid hypotheticals like "could" or "would".
- Use active voice where the subject performs the action.
- Write in third person (one, the user) to keep statements consistent.

## Markdown Guidelines

- Use headings to organise content.
- Use bullet points for lists.
- Include links to related resources.
- Use code blocks for code snippets.

---

## DocStringExtensions.jl Macros

Always use `DocStringExtensions.jl` macros where applicable rather than writing boilerplate manually:

| Macro | Use for |
| --- | --- |
| `$(DocStringExtensions.TYPEDEF)` | Docstring header for any `abstract type` or `struct` |
| `$(DocStringExtensions.TYPEDSIGNATURES)` | Docstring header for internal/private functions (auto-generates signature) |
| `$(DocStringExtensions.FIELDS)` | `# Fields` section body — auto-generates field list from inline field docstrings |
| `$(DocStringExtensions.README)` | Module-level docstring (top of main module file) |

Public functions use a **manually written signature** in the docstring header (not `TYPEDSIGNATURES`), because the manual form allows showing default values and grouping overloads clearly.

---

## Documentation Dictionaries

Five dictionaries in `src/01_Base.jl` provide standardised, consistent descriptions. **Always interpolate from them** instead of writing ad-hoc text.

- `arg_dict` — argument descriptions. Use as `$(arg_dict[:key])` in `# Arguments` sections.
- `field_dict` — field descriptions (derived from `arg_dict`). Use as `"$(field_dict[:key])"` in inline field docstrings inside structs.
- `val_dict` — validation rule descriptions. Use as `$(val_dict[:key])` in `## Validation` sections.
- `ret_dict` — return value descriptions. Use as `$(ret_dict[:key])` in `# Returns` sections.
- `math_dict` — LaTeX mathematical notation. Use as `$(math_dict[:key])` in the `Where:` list of a `# Mathematical definition` or `# JuMP formulation` section.

If a needed key is missing, add it to the appropriate dictionary in `01_Base.jl` before writing the docstring.

### Inline field docstrings

Fields in `@concrete` structs are documented inline using `field_dict`:

**Example.**

```julia
@concrete struct Covariance <: AbstractCovarianceEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:malg])"
    alg
    ...
end
```

### When a field description may be prose

The dictionary exists to stop two copies of one sentence drifting apart. One copy cannot drift, so the rule is keyed on the number of users:

- A description used by **more than one field must interpolate** `field_dict`.
- A description used by **exactly one field may be written as prose** in the struct body.
- When a prose description gains a second user, move it into `arg_dict` and replace both copies with the interpolation.

`field_dict` is derived from `arg_dict` by stripping everything up to and including the first `:`, so a new field description is added to `arg_dict`. Editing a value already in `arg_dict` moves every docstring that interpolates the key, so **add a new key rather than rewriting one that is in use**.

**Example.**

```julia
@concrete struct MyType <: AbstractMyType
    # `me` is described by many types, so it interpolates.
    "$(field_dict[:me])"
    me
    # `tag` is described here and nowhere else, so prose is permitted.
    "Selector that names the branch this type takes."
    tag
end
```

---

## What each section holds

A docstring has one home for each kind of fact. Put a sentence where its **subject** belongs. A true sentence in the wrong section is still a defect, because the reader who needs it looks somewhere else.

| Section | Holds | Does not hold |
| --- | --- | --- |
| summary paragraph | what the unit is, in the first sentence; a trap that applies to the unit as a whole, in a later sentence | a formula, a step, or a fact about one field |
| `# Mathematical definition` | the closed form that defines the unit, and a consequence of that form | an identifier from the body, an order of operations, or a choice the implementation made |
| `# Algorithm` | the numbered steps the body runs, each naming the quantity it produces | a closed form restated as a step |
| `# JuMP formulation` | the variables the code reads or creates, one bullet per row it registers, and the relaxation when the encoding is not exact | a row the body does not register |
| `# Interfaces` | on an abstract type, the methods a concrete subtype must implement, one subsection per method | a method the family does not dispatch on |
| `# Fields` | one description per field, written where the field is declared | a fact about the type as a whole |
| `# Constructors` | the signature, `## Validation`, and the propagation subsections | a rule the constructor does not enforce |
| `# Arguments` | the contract of each argument | the shape of the result |
| `# Validation` | every precondition and every raise | a precondition the code does not check |
| `# Returns` | the shape and the meaning of each returned value | how the value was computed |
| `# Examples` | a `jldoctest` block | prose that a section above owns |
| `# Related` | one bullet per related unit, annotated when the relation is not obvious | a copy of the related unit's own text |
| `# References` | one `ref_dict` interpolation per published work | reference prose written out |

### `# Details` is abolished

**There is no `# Details` section.** It held facts that the sections above already own, and it held them because nothing said where they belonged. Write no new one, and move an existing one by the **subject** of each fact it carries.

| The fact is about… | It goes to |
| --- | --- |
| one field | that field's description |
| a raise or a precondition | `# Validation` |
| the unit as a whole | the summary paragraph, in the second sentence or later |
| another unit | `# Related`, as an annotated entry |

A mis-filed step goes to `# Algorithm`, an argument contract to `# Arguments`, the shape of a result to `# Returns`, and a model row to `# JuMP formulation`.

The Capability Catalogue extracts the **first sentence only** of the summary paragraph, so a later sentence of that paragraph is a safe home for a trap that applies to the whole unit.

`test/test_26_docs.jl` gates the abolition twice. A file marked `swept = true` in [`sweep/manifest.toml`](../../sweep/manifest.toml) carries **zero** `# Details` sections, and the library-wide count of the section **may not rise**. The second check retires when that count reaches zero. [ADR 0085](../../docs/adr/0085-the-docstring-standard-is-rules-and-pointers.md) records the decision.

---

## Section Structure for Types (abstract and concrete)

### Abstract types

**Template.**

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this abstract type represents.

All concrete subtypes should subtype `MyAbstractType`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype
`MyAbstractType` and implement the following methods:

## Required method name

  - `method_name(x::MyAbstractType, arg::Type) -> ReturnType`: What the method does.

### Arguments

  - `x`: The concrete subtype instance.
  - `arg`: Description.

### Returns

  - `result::ReturnType`: Description.

### Examples

```jldoctest
julia> struct MyConcreteType <: PortfolioOptimisers.MyAbstractType end
...
```

# Related

  - [`ConcreteSubtype1`](@ref)
  - [`related_function`](@ref)
"""
abstract type MyAbstractType <: AbstractEstimator end
````

### Concrete struct types

**Template.**

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this type does.

Optional longer explanation with mathematical notation if needed.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MyType(;
        field1::Type1 = default1,
        field2::Type2 = default2
    ) -> MyType

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:key1])
  - $(val_dict[:key2])

# Examples

```jldoctest
julia> MyType()
MyType
  field1 ┴ default1
```

# Related

  - [`AbstractMyType`](@ref)
  - [`related_function`](@ref)
"""
@concrete struct MyType <: AbstractMyType
    "$(field_dict[:key1])"
    field1
    "$(field_dict[:key2])"
    field2
    function MyType(field1::Type1, field2::Type2)
        # validation
        return new{typeof(field1), typeof(field2)}(field1, field2)
    end
end
function MyType(; field1::Type1 = default1, field2::Type2 = default2)
    return MyType(field1, field2)
end
````

### `@propagatable` concrete struct types

When a struct is decorated with `@propagatable`, its fields carry stackable tags inside the struct body. Three of them drive a generated method whose behaviour a reader of the type needs to know:

- `@fprop` — **factory propagation**: the field is automatically propagated when [`factory`](@ref) is called on the enclosing struct (see `factory_child` for dispatch rules).
- `@vprop` — **view propagation**: the field is automatically subset along the **asset** axis when [`port_opt_view`](@ref) is called on the enclosing struct (recursing into composed children, slicing data arrays).
- `@wprop` — **observation weights**: the field holds the weights themselves. It drives **two** channels, and they do different things to it: [`factory`](@ref) **replaces** it with an incoming [`ObsWeights`](@ref) value, and [`obs_weights_view`](@ref) **indexes** the value already there, along the **observation** axis.

(`@pprop` and `@cprop` drive prior selection. They are documented on the prior families, not here.)

A field may carry any combination (`@fprop @vprop field`, in either order) — the field sets of the channels genuinely diverge, so a field can be factory-propagated but view-passthrough, or vice versa. **Document each channel in its own subsection inside `# Constructors`**, placed **after** `## Validation` (or directly after "Keywords correspond to the struct's fields." when there is no `## Validation`), in the order below.

**A subsection documents a channel, not a tag.** A channel is gated on more than one tag, so the field list of a subsection is the union of the tags that channel reads. Write a subsection when at least one field carries a tag that gates it, and omit it otherwise. The view channel carries one more trigger: a hand-written [`port_opt_view`](@ref) method earns the subsection even when no field carries `@vprop`.

`## Propagated parameters` — the **factory** channel, gated on `@fprop` and `@wprop`. Lists each field and how it is propagated:

- **Observation-weight fields** (`@wprop`; type `ObsWeights`, `Nothing`, or `Option{<:ObsWeights}`): write `` `fieldname`: Replaced with the incoming [`ObsWeights`](@ref). ``
- **Estimator, algorithm, or result fields** (`@fprop`; subtypes of `AbstractEstimator`, `AbstractAlgorithm`, or `AbstractResult`): write `` `fieldname`: Recursively updated via [`factory`](@ref). ``

`## View parameters` — the **view** channel. Write it when the type has a [`port_opt_view`](@ref) method, whether [`@propagatable`](@ref) generates that method from a `@vprop` tag or the file defines one by hand. The two cases carry different content.

**A generated method** — at least one field carries `@vprop`. List each tagged field and how it is viewed:

- **Estimator, algorithm, or result fields** (subtypes of `AbstractEstimator`, `AbstractAlgorithm`, or `AbstractResult`): write `` `fieldname`: Recursively viewed via [`port_opt_view`](@ref). ``
- **Data fields** (arrays, scalars, or `Option` thereof): write `` `fieldname`: Sliced to the selected indices via [`port_opt_view`](@ref). ``

**A hand-written method** — no field carries `@vprop`, and the file defines `port_opt_view(x::MyType, i, args...)`. A field list misdescribes such a method: it reads arguments no tag can name, and it applies rules no tag can encode. Open the subsection with

> `MyType` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

and follow that with a bullet list of high-level details. Cover these points, in this order, and only where the method has something to say:

- Which arguments beyond `i` the method reads, and what it does with them.
- Which fields recurse through [`port_opt_view`](@ref), and with which arguments.
- Which fields are sliced, and along which axis when that axis is not the asset axis.
- Which fields pass through unchanged, where a reader would expect otherwise.
- Any rule the method enforces or preserves that a field list cannot show.

Describe the method, do not transcribe it. A field carried through with nothing to say about it needs no bullet.

`## Observation weight parameters` — the **observation** channel, gated on `@wprop` alone. A type with no `@wprop` field never gains this subsection, whatever else it carries. Lists each field and how it is indexed:

- **Observation-weight fields** (`@wprop`): write `` `fieldname`: Indexed to the selected observations via [`obs_weights_view`](@ref). ``
- **Estimator, algorithm, or result fields** (`@fprop`): write `` `fieldname`: Recursively indexed via [`obs_weights_view`](@ref). ``

> **The same field appears in two subsections, saying two different things.** A `@wprop` field is *replaced* under `## Propagated parameters` and *indexed* under `## Observation weight parameters`. That is not a duplication to collapse: it is the one place a reader learns that `factory` and `obs_weights_view` treat the field differently. A `@fprop` sibling appears in both too, and recurses in both.

List fields in the same order they appear in the struct body, and add [`factory`](@ref), [`port_opt_view`](@ref) and/or [`obs_weights_view`](@ref) to `# Related` to match the subsections present.

**Template.**

````julia
"""
$(DocStringExtensions.TYPEDEF)

One-sentence description of what this type does.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MyType(;
        weight_field::Option{<:ObsWeights} = nothing,
        nested_est::AbstractEstimator = MyEstimator(),
        config::Bool = true
    ) -> MyType

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `nested_est`: Recursively updated via [`factory`](@ref).
  - `weight_field`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `nested_est`: Recursively viewed via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `nested_est`: Recursively indexed via [`obs_weights_view`](@ref).
  - `weight_field`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> MyType()
MyType
  nested_est ┼ MyEstimator
  weight_field ┴ nothing
```

# Related

  - [`AbstractMyType`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct MyType <: AbstractMyType
    "$(field_dict[:oow])"
    @fprop weight_field
    "$(field_dict[:nested])"
    @fprop @vprop nested_est
    "$(field_dict[:cfg])"
    config
    function MyType(weight_field::Option{<:ObsWeights}, nested_est::AbstractEstimator,
                    config::Bool)
        assert_nonempty_nonneg_finite_val(weight_field, :weight_field)
        return new{typeof(weight_field), typeof(nested_est), typeof(config)}(weight_field,
                                                                             nested_est,
                                                                             config)
    end
end
function MyType(;
                weight_field::Option{<:ObsWeights} = nothing,
                nested_est::AbstractEstimator = MyEstimator(),
                config::Bool = true)::MyType
    return MyType(weight_field, nested_est, config)
end
````

---

## Section Structure for Functions

Public functions use a manually written signature as the docstring header.

**Template.**

````julia
"""
    function_name(
        arg1::Type1,
        arg2::Type2;
        kwarg1::Type3 = default
    ) -> ReturnType

One-sentence description of what this function does.

Longer explanation if needed.

# Arguments

  - $(arg_dict[:key1])
  - $(arg_dict[:key2])
  - `kwarg1::Type3 = default`: Description.

# Validation

  - $(val_dict[:key])

# Returns

  - $(ret_dict[:key])

# Examples

```jldoctest
julia> function_name(...)
...
```

# Related

  - [`RelatedType`](@ref)
  - [`related_function`](@ref)
"""
function function_name(arg1::Type1, arg2::Type2; kwarg1::Type3 = default)
    ...
end
````

Internal/private functions may use `$(DocStringExtensions.TYPEDSIGNATURES)` as the header instead of a manually written signature.

---

## Section Structure for Aliases

An **alias** is a second name for something another docstring already documents. Its docstring routes the reader to that documentation. It restates none of it, because a second copy drifts away from the first — the argument that put the field descriptions into `field_dict`.

Three kinds of alias exist, and the sections differ by kind.

| Kind | Declaration | Lives in | Header line | Sections it carries |
| --- | --- | --- | --- | --- |
| **Acronym alias** | `const HRP = HierarchicalRiskParity` | `src/25_Aliases.jl` only | the alias name alone | none |
| **Factory alias** | `MAD(; kwargs...)::LowOrderMoment` | `src/25_Aliases.jl` only | the signature, ending `-> T` | `# Validation`, and only when its own body raises |
| **Dispatch alias** | `const RhoDistanceAlgorithm = Union{...}` | any file under `src/` | the declaration, `const NAME = <type expression>` | `# Related`, and `# References` when the grouping itself is published |

A **dispatch alias** is a `const` bound to a type expression rather than to a bare name. A `Union`, a container such as `AbstractVector{<:LinearConstraint}`, and a parametrised form such as `const RMCVaR{T} = Union{...}` are all one kind, because a caller meets all three the same way: as the type a method signature dispatches on.

### The summary paragraph

- An **acronym alias** carries exactly one sentence, `Alias for [`Canonical`](@ref).` and nothing more. The alias and its target are the same object, so a second sentence describes the target and belongs on the target.
- A **factory alias** carries one sentence naming what it builds. That sentence `@ref`s every type the factory composes. A later sentence is permitted, and only for a choice the composition fixes that a reader would otherwise get wrong. Read `ZeroVarianceFilter` in [`src/25_Aliases.jl`](../../src/25_Aliases.jl), which states why it scores with `SCM()` and not with `Variance`.
- A **dispatch alias** carries what the alias groups and why the group exists. The *why* is the load-bearing half: a reader who sees only the member list learns nothing the declaration did not already show.

### Why an alias carries so little

- **No `# Fields`, no `# Constructors`, no `# Arguments`, no `# Returns`, no `# Examples`, on any kind.** The canonical unit owns each of them, and the summary sentence puts it one click away. A copy on the alias is a second text to maintain and the one a reader reaches first when it goes stale.
- **No `# Related` on an acronym alias or on a factory alias.** Its summary sentence already `@ref`s every canonical name, and `## What each section holds` states that `# Related` does not hold a copy of the related unit's own text. One link, once.
- **`# Related` on a dispatch alias**, one bullet per grouped member. A group of members is a list, and a summary sentence must not be one. A function that dispatches on the alias may take a bullet of its own.
- **`# Algorithm`, `# Mathematical definition` and `# JuMP formulation` reach no alias.** An alias runs no steps and registers no row. A factory alias composes types and takes no branch worth numbering.

The Capability Catalogue lists an alias in `NOT_A_FEATURE` with the reason `:alias`, under [`.github/instructions/julia-source-code.instructions.md`](julia-source-code.instructions.md), so it never reads an alias summary sentence.

### The Gate

`test/test_26_docs.jl` gates the rule with three checks, in the shape [ADR 0086](../../docs/adr/0086-an-alias-docstring-links-its-canonical-unit-and-restates-nothing.md) records.

 1. **Library-wide, absolute.** No alias carries a section outside its kind's row of the table above. A new breach is the only way this check can red.
 2. **A swept file, presence.** A dispatch alias in a file marked `swept = true` in [`sweep/manifest.toml`](../../sweep/manifest.toml) carries `# Related`.
 3. **Library-wide, a ratchet.** The count of dispatch aliases carrying no `# Related` may not rise. The check retires when that count reaches zero.

Check 2 reads the `swept` flag because it demands a section, and a presence demand may not red a file that no child map of issue #404 has swept. Check 1 forbids a section instead, so it needs no flag.

---

## The `# Validation` Section

- Include a `## Validation` sub-section in struct docstrings and a `# Validation` section in function docstrings whenever the function or constructor enforces preconditions.
- Use `val_dict` entries for common validation rules.
- For custom validation, describe the condition clearly: `` `x > 0` ``.

---

## The `# References` Section

A docstring that rests on a published source names it. The section is **last**, after `# Related`, and it holds one bullet per work.

- **Never paste the reference prose.** Every bullet is one interpolation of `ref_dict` (`src/01_Base.jl`), which holds a single copy of each reference text:

  ```julia
  # References

    - $(ref_dict[:mlp1]) Chapter 2.
  ```

  A locator such as `Chapter 2.` may follow the interpolation. Nothing else may.
- **If the work has no key**, add the BibTeX entry to `docs/src/References.bib`, then add the matching `ref_dict` entry. The `ref_dict` value is the citation marker followed by the reference formatted as `DocumenterCitations` renders it in the bibliography.
- **If the type has no source**, write no section. A `ref_dict` entry with no user is a test failure, and so is a citation whose key is not in `References.bib`.
- **Add the page's bibliography block.** An API page whose prose or whose included docstrings cite anything carries this block, and a page that cites nothing must not:

  ````markdown
  ## References

  ```@bibliography
  Pages = [@__FILE__]
  Canonical = false
  ```
  ````

  `Canonical = false` is load-bearing. `docs/src/99_references.md` holds the one canonical block; a second canonical block silently skips every entry the first claimed. A `Pages` block on a page that cites nothing is a docs-**build error**, not a warning.
- An inline citation in prose — `as described in [DBHTs](@cite)` — needs no bullet of its own, but the work still needs a `# References` bullet on the type that owns it.

The four rules above are checked by the `"References"` testset in `test/test_26_docs.jl`.

---

## `jldoctest` Examples

- All examples in docstrings must use `jldoctest` blocks so they are testable via `julia --doctest`.
- Output of pretty-printed structs must match exactly what `@define_pretty_show` produces.
- For abstract types with an `# Interfaces` section, the `jldoctest` must demonstrate a complete working implementation of the interface.
- Keep examples minimal but complete enough to be useful.

---

## `docs/src/api/` Markdown Files

Each source file `src/SomeFeature.jl` has a corresponding `docs/src/api/SomeFeature.md`. Every public symbol defined in the source file must be listed under an appropriate heading using the Documenter.jl `@docs` block:

**Template.**

````markdown
## My section heading

```@docs
MyType
my_function
MyAbstractType
```
````

When adding a new symbol, also add it to the corresponding API markdown file.

---

## Mathematical Notation

When a type or function has a mathematical formulation, include a `# Mathematical definition` section immediately before `# Fields` (for structs) or before `# Arguments` (for functions).

### What the section may not state

The section states the mathematics and nothing else. It **names no identifier from the body, states no order of operations, and states no property that the implementation chose rather than the mathematics.** A consequence of the definition stays, because a consequence is mathematics.

`# Algorithm` carries the reverse rule — *do not restate a closed form as a step*. The two rules bound one border from opposite sides, so a fact that fails this rule usually already stands as a step, and the fix is to delete it here rather than to move it.

**Example.** Read against `ShrunkDenoise` in [`src/05_Denoise.jl`](../../src/05_Denoise.jl):

- CUT — *the eigenvalues sorted ascending*. The sort is the body's choice, and it is already an `# Algorithm` step.
- CUT — *the diagonal is pinned to one afterwards to shed the eigendecomposition round-off*. That is an `# Algorithm` step too.
- CUT — `vals`, `vecs`, `corr0`. Each names a local of the body.
- KEEP — *the two `alpha` weights sum to one on the diagonal, so the reconstruction preserves it in exact arithmetic*. That is a consequence of the definition, and it holds whatever the body does.

An implementation fact is not a token, so no parser finds one. This rule is **unenforced**: it holds by review, in the sense of [`STANDARDS.md`](../../STANDARDS.md). That is a known state and not a hidden one.

### LaTeX conventions

| Notation | Use for |
| --- | --- |
| `\boldsymbol{x}` | Vectors (e.g., ``\boldsymbol{w}``, ``\boldsymbol{\mu}``) |
| `\mathbf{A}` | Matrices (e.g., ``\mathbf{\Sigma}``, ``\mathbf{F}``) |
| `\mathbb{R}` | Number domains (e.g., ``\mathbb{R}^N``, ``\mathbb{Z}_{\geq 0}``) |
| `\mathcal{W}` | Sets (e.g., ``\mathcal{W}``, ``\mathcal{K}_{\mathrm{SOC}}``) |
| `\underset{\boldsymbol{w}}{\min}` | Optimisation formulations (not `\min_{\boldsymbol{w}}`) |
| `\intercal` | Transpose (e.g., ``\boldsymbol{w}^\intercal``) |

### `\begin{align}` environment

All math blocks use `\begin{align}...\end{align}` with `&` alignment markers and `\\` line breaks. Each separate equation goes on its own line. Use `\quad` for in-equation spacing only; split distinct equations onto separate lines (never `\qquad` between two equations in the same block).

**Example.**

````julia
# Good — each equation on its own line
```math
\begin{align}
\hat{\boldsymbol{\mu}} &= \frac{1}{T} \sum_{t=1}^{T} \boldsymbol{x}_t\,, \\
\hat{\mathbf{\Sigma}} &= \frac{1}{T-1} \sum_{t=1}^{T}
    (\boldsymbol{x}_t - \hat{\boldsymbol{\mu}})
    (\boldsymbol{x}_t - \hat{\boldsymbol{\mu}})^\intercal\,.
\end{align}
```

# Bad — \qquad to cram two equations on one line
```math
\begin{align}
\hat{\boldsymbol{\mu}} &= \frac{1}{T} \sum_{t=1}^T \boldsymbol{x}_t \qquad
\hat{\mathbf{\Sigma}} = \frac{1}{T-1} \sum_{t=1}^T \ldots
\end{align}
```
````

### `Where:` section

Immediately after each math block (or after the **last** math block when multiple consecutive blocks belong to the same docstring), add a `Where:` bullet list defining every symbol. Use `$(math_dict[:key])` for common variables.

**Template.**

````julia
"""
# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\frac{1}{T} \\sum_{t=1}^{T} \\boldsymbol{x}_t\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\frac{1}{T-1} \\sum_{t=1}^{T}
    (\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})
    (\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})^\\intercal\\,.
\\end{align}
```

Where:

- ``\\hat{\\boldsymbol{\\mu}}``: Estimated mean vector.
- ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
- $(math_dict[:x_t])
- $(math_dict[:T])
"""
````

Key rules:

- One comprehensive `Where:` after the last block is acceptable when multiple blocks appear in the same docstring.
- Every symbol that appears in any block must be defined.
- Interpolate `$(math_dict[:key])` for standardised variables (``T``, ``\boldsymbol{x}_t``, ``\alpha``, etc.).
- If a key is missing from `math_dict`, add it to `src/01_Base.jl` first.

---

### Notation is fixed by symbol and by family

The rules above fix the glyphs. The two below fix the content, so that two docstrings that state one quantity state it once and state it alike.

**A shared symbol becomes a `math_dict` key.** A symbol that appears in the docstrings of two or more Units gets a key in [`src/01_Base.jl`](../../src/01_Base.jl), and every site interpolates it. A symbol that exactly one Unit uses may stay inline, on the reasoning of [When a field description may be prose](#when-a-field-description-may-be-prose): one copy cannot drift. When a second Unit needs it, move it into `math_dict` and replace both copies with the interpolation.

A new description is a **new** key. Editing a value already in `math_dict` moves every docstring that interpolates it, which is the reason [ADR 0081](../../docs/adr/0081-the-docstring-standard-states-the-model-it-builds.md) gives for `arg_dict`.

**A key owns a definition, not a glyph.** One glyph carries different quantities in different families: ``\boldsymbol{w}`` is the portfolio weights vector in a risk measure and the observation weights in a moment estimator. A key is therefore claimed by the whole definition — the symbol together with the sentence that defines it — and a second quantity on the same glyph takes its own key under its own symbol. It never takes a second meaning on the first.

**Gate.** `test/test_26_docs.jl` reds when a `Where:` bullet copies a `math_dict` value instead of interpolating it. It matches the whole bullet against the whole value, so it fires on a copy and never on a glyph that two families share. A file marked `swept = true` in [`sweep/manifest.toml`](../../sweep/manifest.toml) carries no such copy, and the library-wide count may not rise. The copies that remain migrate file by file, inside each file's own sweep ticket of issue #404.

**Siblings of one Family state a shared quantity in the same form.** Not the same symbol alone — the same shape of equation. A condition that one sibling writes in a parenthesis, the second in set notation and the third in its `Where:` list is one fact written three ways, and the three do not read as one Family. Take the shape from the sibling that states it most completely, and write the others to match it.

The Family is the leaf-most abstract supertype, never a generic root. `RiskMeasure` and `AbstractResult` span many files and their members share no notation, so neither is a Family in this sense.

An equation's form is not a token, so no parser finds a breach. This rule is **unenforced**: it holds by review, in the sense of [`STANDARDS.md`](../../STANDARDS.md). That is a known state and not a hidden one.

---

## The `# Algorithm` Section

A **procedure** is documented in `# Algorithm`. A **closed form** stays in `# Mathematical definition`. A docstring may carry both sections, and a marker type carries neither.

`# Algorithm` sits immediately after `# Mathematical definition`, and before `# Fields` (for structs) or before `# Arguments` (for functions).

Rules:

- Write **numbered steps, one step per operation**. Each step names the quantity that the operation produces.
- Name each quantity by the name that the body gives it, so a reader can follow the steps in the code.
- Do not restate a closed form as a step. The formula belongs in `# Mathematical definition`, and the step that applies it names it.
- A **selector tag** — a type whose only job is to name the branch that a caller takes — carries **neither** section. Its summary sentence states which branch it selects. Most subtypes of `AbstractAlgorithm` are selector tags, so the rule must never force numbered steps onto a marker type.

**Example.** The following is the algorithm of `denoise!(dn::Denoise, X::MatNum, q::Number)` in `src/05_Denoise.jl`:

````julia
"""
# Algorithm

 1. Check that `X` is square.
 2. Read the diagonal of `X` into `s`. When any entry of `s` is not one, `X` is a covariance matrix: replace `s` with its square roots and convert `X` to a correlation matrix with `StatsBase.cov2cor!`.
 3. Eigendecompose `X`, giving the ascending eigenvalues `vals` and the eigenvectors `vecs`.
 4. Fit the Marčenko-Pastur density to `vals`, giving `max_val`, the upper edge of the noise band.
 5. Count the eigenvalues that do not exceed `max_val`, giving `num_factors`, the number of noise eigenvalues.
 6. Rebuild `X` from the split spectrum, through the branch that `alg` selects.
 7. Repair the rebuilt matrix with `posdef!`.
 8. When step 2 converted a covariance matrix, convert `X` back with `StatsBase.cor2cov!`.
"""
````

---

## The `# JuMP formulation` Section

Any code that **adds rows to a `JuMP.Model`** carries this section. It states the model that the code builds, which the mathematics alone does not: the rows carry names, a caller reads them back by those names, and the encoding is not always exact.

`# JuMP formulation` sits after `# Mathematical definition` and after `# Algorithm`, and before `# Fields` (for structs) or before `# Arguments` (for functions).

It has three subsections, in this order. The first two are always present. The third is present only when the encoding is not exact.

### `## Variables`

One bullet per model variable that the code reads or creates. Name each variable by its model key, and say whether the code reads it or creates it.

### `## Constraints`

**One bullet per row that the code registers**, in the order in which the body registers them. Each bullet carries **the row's JuMP name** and the mathematics of the row. The name is the one written in the `JuMP.@constraint` call, because that is the key with which a caller reads the row back out of the model.

Close the subsection with a `Where:` list that defines every symbol, under the rules of the `Where:` section above. Interpolate `$(math_dict[:key])` for a standardised symbol.

### `## Relaxation`

Present **only when the encoding is not exact**. An exact encoding carries no `## Relaxation` subsection at all.

Open the subsection with `$(val_dict[:relax])`, so that the opening cannot drift from one docstring to the next. Then state three things:

 1. The **direction** of the bound: whether the model quantity lies above or below the exact quantity.
 2. The **quantity** that is bounded, named by its model key.
 3. The **condition** under which the bound is tight.

**Example.** The following is the formulation of `set_gross_budget_constraints!` in `src/20_Optimisation/09_JuMPConstraints/03_BudgetConstraints.jl`:

````julia
"""
# JuMP formulation

## Variables

  - `lw`, `sw`: long and short weight vectors, read from the model.
  - `k`: homogenisation scalar.

## Constraints

  - `gbgt_lb`: ``s_c \\left(\\sum lw + \\sum sw - k b_l\\right) \\geq 0``
  - `gbgt_ub`: ``s_c \\left(\\sum lw + \\sum sw - k b_u\\right) \\leq 0``

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - ``b_l``, ``b_u``: lower and upper gross budget bounds.
"""
````

---

## Reference docstrings

Read a real docstring, not a copy of one. Each row names a Unit whose file is marked `swept = true` in [`sweep/manifest.toml`](../../sweep/manifest.toml), so a Gate holds the target and the pointer cannot drift.

| Kind | Unit | File |
| --- | --- | --- |
| Abstract type | `AbstractDenoiseAlgorithm` | [`src/05_Denoise.jl`](../../src/05_Denoise.jl) |
| Selector tag | `SpectralDenoise` | [`src/05_Denoise.jl`](../../src/05_Denoise.jl) |
| Struct with fields | `ShrunkDenoise` | [`src/05_Denoise.jl`](../../src/05_Denoise.jl) |
| Public function | `denoise!` | [`src/05_Denoise.jl`](../../src/05_Denoise.jl) |
| Private function | `_denoise!` | [`src/05_Denoise.jl`](../../src/05_Denoise.jl) |
| Dispatch alias | `RhoDistanceAlgorithm` | [`src/09_Distance/02_Distance.jl`](../../src/09_Distance/02_Distance.jl) |

`# JuMP formulation` carries no row. Every file that calls a `JuMP` macro is unswept, so no Gate holds a pointer into one. The row is added when the first such file is swept.
