---
applyTo: 'src/**/*.jl, ext/**/*.jl, docs/**/*.md'
---

# Docstring and Documentation Guidelines for PortfolioOptimisers.jl

## How to read this file

This file is the Authority for docstrings, in the sense of [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md). It carries the rules, and it names where to read a real docstring. It holds no worked docstring of its own. A copy inside a standards file drifts away from the code, and a pointer cannot drift.

Three kinds of block appear below.

- **A rule** is prose or a bullet list. It is normative.
- **A template** is a fenced block marked `**Template.**`. Its section names, its order and its shape are normative. The identifiers inside it are invented, and they are not normative.
- **An example** is a fenced block marked `**Example.**`. It illustrates the rule beside it, and it is not normative.

A fenced block inside a bullet belongs to that bullet.

To read a complete docstring, open a Unit that [Reference docstrings](#reference-docstrings) names.

## Vocabulary

This file gives five words a precise meaning.

- **Unit.** One documented name: a module, a type, a function, a macro or a constant. [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml) records the unit count of every file under `src/` and `ext/`, and `test/test_45_sweep_census.jl` fails when the count of a file leaves its row.
- **Family.** A leaf abstract supertype, together with the concrete types that subtype it. Leaf-most means that no abstract type subtypes it. So a generic root such as `RiskMeasure` or `AbstractResult` is not a Family, because its members are in many files and share no notation. Most families are inside one file, so a rule about a Family is local to one sweep ticket. [Notation is fixed by symbol and by family](#notation-is-fixed-by-symbol-and-by-family) is such a rule.
- **Reference docstring.** A docstring that the [Reference docstrings](#reference-docstrings) table names. The sweep manifest marks its file `swept = true`, so a Gate holds it. Read one in place of a worked example.
- **Capability Catalogue.** The user-facing inventory of everything the package offers. [`docs/capability_catalogue.jl`](../../docs/capability_catalogue.jl) builds it under ADR 0040. It extracts the first sentence of the summary paragraph of a type verbatim.
- **Coverage Exemption.** A named ruling in [`code_health/rulings.toml`](../../code_health/rulings.toml) that excuses a stated number of uncovered lines in one file. [ADR 0082](../../docs/adr/0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md) owns it.

## General Guidelines

- Read the docstrings of other units, and follow their patterns.
- Write short, clear sentences.
- Use one word for one concept, in the same style as the other docstrings.
- Give a code example where the unit has one.
- **All public types, functions, and macros must have docstrings.**
- **Scope.** The rules cover `src/**/*.jl`, `ext/**/*.jl` and `docs/**/*.md`. A package extension in `ext/` follows the same rules as `src/`.
- **An extension documents the names it declares itself, and nothing else.** An extension adds methods to a name that `src/` declares, and the declaration carries the docstring, the `# References` section and the API-page entry. So a method of a function declared in `src/` gets no docstring of its own in `ext/`. The module of the extension gets one, and so does each function, constant, type and macro that the extension declares itself, such as a private helper. Write a citation in the `src/` declaration, because the API page that renders the declaration carries the bibliography block. An extension needs neither an API page nor a bibliography block of its own. `test/test_26_docs.jl` gates this per file, through the `swept` flag in [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml).

## The summary sentence

Read this section before you write a type docstring. The Capability Catalogue reads the summary sentence, so a defect in it reaches a page. Every type docstring opens with `$(DocStringExtensions.TYPEDEF)`, a blank line, and then a **summary paragraph**. The [Capability Catalogue](../../docs/capability_catalogue.jl) (ADR 0040) **extracts the first sentence of that paragraph verbatim** and renders it as the one-line description of the type. The catalogue is the user-facing inventory of everything the package can do.

The docs build enforces this contract. Because of it, the catalogue carries prose and keeps no second copy of a description that can drift. If a docstring loses its summary paragraph, the docs build fails with an error that names the type.

Write the first sentence so that it reads correctly alone as an entry of a bullet list:

- **Lead with what it does**, in the active voice. Write `Sets the smallest \`num_factors\` eigenvalues to zero to denoise the matrix.`, not`A denoising algorithm that sets...`.
- **Keep it under ~120 characters.** If the idea needs more, write a short first sentence and move the detail into a *second* sentence. The API page renders the second sentence, and the catalogue does not. Do not delete information to make the sentence short.
- **Do not open with filler**, such as `A flexible container type for...` or `A concrete estimator type for...`. Such an opener costs a line and says nothing.
- **Do not append `in \`PortfolioOptimisers.jl\``.** Every docstring in the package is in`PortfolioOptimisers.jl`.
- **Do not put `@ref` links in the first sentence.** The catalogue appends the links of the type after the description, so a link in the summary renders twice. Put cross-references in a second sentence or in `# Related`.
- **Do not open with a display formula.** Inline maths is permitted. A full `$...$` equation belongs in `# Mathematical definition`.
- **Never leave a bare `_` outside a code span.** Markdown reads `_` as emphasis. It pairs the bare `_` with the underscore inside a neighbouring `` `snake_case` `` link, removes both, and breaks the link. In one docstring, `(f_μ vector)` next to `` [`plot_factor_mu`](@ref) `` rendered as ``(fμ vector … [`plotfactor_``. Documenter cannot resolve that dead link, and the site builder reports it only as one anonymous `./@ref`. Write `` `f_mu` `` instead.
- **Do not give every sibling the same prefix.** If every algorithm in a family starts `Centrality algorithm type for ...`, the catalogue shows that text eight times. State what makes each one different.

## The prose follows ASD-STE100

The prose of a docstring follows ASD-STE100, Simplified Technical English, as the current issue of the specification states it. **The specification is the Authority for the words and the grammar.** This section states the Scope, the exception for the domain vocabulary, and the rules that a docstring adds to the specification.

**Scope.** The Scope is that of [The prose passes `/unslop`](#the-prose-passes-unslop). It holds the docstrings of `src/**/*.jl` and `ext/**/*.jl`, and the dictionary values of [`src/01_Base/01_DocstringDictionaries/`](../../src/01_Base/01_DocstringDictionaries/) that they interpolate. The rule reads the prose alone. It does not read a code span, an identifier, a `jldoctest` block, a LaTeX expression, a heading that a template of this file fixes, or a `# References` bullet.

**The words of `GLOSSARY.md` are approved.** A term that [`GLOSSARY.md`](../../GLOSSARY.md) defines is an approved technical name, and a verb that it defines is an approved technical verb. The exception holds when the dictionary of the specification does not contain the word, and when the dictionary gives the word a different meaning or a different part of speech.

- Use the glossary term with the meaning that the glossary gives it, and with no other meaning. In a docstring, the words Prior, Result and Estimator have the glossary meaning only.
- Do not use a word that the glossary lists under *Avoid*, even when the dictionary of the specification approves it.
- The exception covers the glossary words only. Every other word follows the dictionary, or the rules of the specification for technical names and technical verbs. The name of a published method is a technical name.
- Do not coin a domain term in a docstring. Add the term to `GLOSSARY.md` in the same change, and then use it.

**The rules that a docstring breaks most often.** This list is a digest. It does not replace the specification.

- Use each word with one meaning and one part of speech. Use the same word for the same thing each time.
- Use the simple present tense, the imperative, the infinitive, or a past participle as an adjective. Do not use an "-ing" form of a verb, except in a technical name such as "floating point".
- Use the active voice.
- Use "must" for a requirement and "can" for a possibility. Do not use "should", "may", "shall", "could" or "would".
- Keep an instruction to 20 words or fewer, and a descriptive sentence to 25 words or fewer.
- Keep the articles "a", "an" and "the". Keep "that" at the start of a noun clause.
- Do not make a noun cluster of more than three nouns. Break a longer cluster with "of" or "for".
- Do not use a semicolon, a dash or a parenthesis to join two thoughts. Write two sentences.
- Keep a paragraph to six sentences or fewer, about one topic.

**What a docstring adds to the specification.**

- Use the simple present tense for a fact about the unit. The specification also permits the simple past and the simple future, but a docstring describes the unit as it is now.
- Write in the third person, or give a command in the imperative.

**Where this rule meets the other rules of this file.**

- A template of this file fixes the shape of a section. The specification applies to the sentences inside it. A `# Fields` bullet, a `Where:` bullet and a `# Related` entry stay list entries.
- The rules of [The summary sentence](#the-summary-sentence) and the specification both apply to the first sentence.
- The `/unslop` pass and this rule both apply to the same prose. One rewrite can satisfy both.
- A dictionary value is one text with many users. Rewrite it once, in its dictionary.

**The Gate.** None. No parser in the repository reads the dictionary of the specification, so the rule holds by review, in the sense of [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md).

## Markdown Guidelines

- Use headings to organise content.
- Use bullet points for lists.
- Include links to related resources.
- Use code blocks for code snippets.

---

## DocStringExtensions.jl Macros

Where a macro of `DocStringExtensions.jl` applies, always use it in place of hand-written boilerplate:

| Macro | Use for |
| --- | --- |
| `$(DocStringExtensions.TYPEDEF)` | Docstring header for any `abstract type` or `struct` |
| `$(DocStringExtensions.TYPEDSIGNATURES)` | Docstring header for an internal or private function. It writes the signature. |
| `$(DocStringExtensions.FIELDS)` | Body of the `# Fields` section. It writes the field list from the inline field docstrings. |
| `$(DocStringExtensions.README)` | Module-level docstring, at the top of the main module file |

A public function has a **hand-written signature** in its docstring header, not `TYPEDSIGNATURES`. The hand-written form can show the default values and group the overloads.

---

## Documentation Dictionaries

Five dictionaries in `src/01_Base/01_DocstringDictionaries/` hold the standard descriptions. **Always interpolate from them**, and do not write new text in their place.

- `arg_dict` holds the argument descriptions. Use it as `$(arg_dict[:key])` in `# Arguments` sections.
- `field_dict` holds the field descriptions, which it derives from `arg_dict`. Use it as `"$(field_dict[:key])"` in inline field docstrings inside structs.
- `val_dict` holds the descriptions of validation rules. Use it as `$(val_dict[:key])` in `## Validation` sections.
- `ret_dict` holds the descriptions of return values. Use it as `$(ret_dict[:key])` in `# Returns` sections.
- `math_dict` holds the LaTeX notation. Use it as `$(math_dict[:key])` in the `Where:` list of a `# Mathematical definition` or `# JuMP formulation` section.

If a needed key is missing, add it to the appropriate dictionary, in the file of `src/01_Base/01_DocstringDictionaries/` that holds its subject, before writing the docstring.

### Inline field docstrings

Document the fields of a `@concrete` struct inline, with `field_dict`:

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

The dictionary exists to stop two copies of one sentence drifting apart. One copy cannot drift, so the rule depends on the number of users:

- A description used by **more than one field must interpolate** `field_dict`.
- A description used by **exactly one field may be written as prose** in the struct body.
- When a prose description gains a second user, move it into `arg_dict` and replace both copies with the interpolation.

The package derives `field_dict` from `arg_dict`. It removes the text up to and including the first `:`. So add a new field description to `arg_dict`. An edit to a value already in `arg_dict` changes every docstring that interpolates the key. So **add a new key, and do not rewrite a key that is in use**.

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
| summary paragraph | what the unit is, in the first sentence, and a trap that applies to the whole unit, in a later sentence | a formula, a step, or a fact about one field |
| `# Mathematical definition` | the closed form that defines the unit, and a consequence of that form | an identifier from the body, an order of operations, or a choice the implementation made |
| `# Algorithm` | the numbered steps the body runs, each naming the quantity it produces | a closed form restated as a step |
| `# JuMP formulation` | the model the code builds: the variables, expressions and rows it registers, the objective it sets, and the relaxation when the encoding is not exact | an entry the body does not register |
| `# Interfaces` | on an abstract type, the methods that a concrete subtype must implement, one subsection per method. A `## \`verb\`` heading names a function that the package defines. A `- \`verb(args...) -> Ret\`` bullet names that verb and only types that exist, with a `My`-prefixed name as the placeholder for the subtype of the reader (`test/test_61_interfaces_section_census.jl`) | a method the family does not dispatch on, or a name the package no longer defines |
| `# Fields` | one description per field, written where the field is declared | a fact about the type as a whole |
| `# Constructors` | the signature, `## Validation`, and the propagation subsections | a rule the constructor does not enforce |
| `# Arguments` | the contract of each argument | the shape of the result |
| `# Validation` | every precondition and every raise | a precondition the code does not check |
| `# Returns` | the shape and the meaning of each returned value | how the value was computed |
| `# Examples` | a `jldoctest` block | prose that a section above owns |
| `# Related` | one bullet per related unit, annotated when the relation is not obvious | a copy of the related unit's own text |
| `# References` | one `ref_dict` interpolation per published work | reference prose written out |

### `# Details` is abolished

**There is no `# Details` section.** It held facts that the sections above already own, and it held them because nothing said where they belonged. Write no new one. To remove an existing one, move each fact that it carries by the **subject** of the fact.

| The fact is about… | It goes to |
| --- | --- |
| one field | that field's description |
| a raise or a precondition | `# Validation` |
| the unit as a whole | the summary paragraph, in the second sentence or later |
| another unit | `# Related`, as an annotated entry |

A mis-filed step goes to `# Algorithm`, an argument contract to `# Arguments`, the shape of a result to `# Returns`, and a model entry to `# JuMP formulation`.

The Capability Catalogue extracts the **first sentence only** of the summary paragraph. So a later sentence of that paragraph is a safe place for a trap that applies to the whole unit.

`test/test_26_docs.jl` gates the abolition twice. A file marked `swept = true` in [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml) carries **zero** `# Details` sections, and the library-wide count of the section **may not rise**. The second check retires when that count reaches zero. [ADR 0085](../../docs/adr/0085-the-docstring-standard-is-rules-and-pointers.md) records the decision.

---

## A docstring cites no process

A docstring documents the released unit, not the discussion that produced it. **It never names a GitHub issue, a pull request, an ADR, or an unpublished numerical experiment**, in any section. This includes an admonition (`!!! note`, `!!! warning`). An admonition renders on the API page like the paragraph beside it, so it is not an exception.

The Scope of the rule is every page that a library user reaches without opening `docs/adr/`. It is wider than `src/**/*.jl`, `ext/**/*.jl` and `docs/**/*.md`. The same clause governs the `Prose` text in [`docs/capability_catalogue.jl`](../../docs/capability_catalogue.jl). It governs the rendered prose and admonitions of the Literate sources that build the example and user-guide pages, `examples/**/*.jl` and `user_guide/*.jl`. It also governs error text. That is the message of a `throw`, an `error`, an error constructor such as `ArgumentError(…)`, an `@argcheck` or an `@assert` under `src/` and `ext/`, and a `const` remedy string whose name ends in `_remedy` or `_message`. An error message reaches the user through the REPL, so it is as user-facing as a page. A contributor reads an ADR to decide what to build next. A caller of the finished function is a different reader. An ADR number means nothing to that reader, and it sends them to look for a file that is not part of the package.

- **State the fact, not its provenance.** Write the rule, the bound or the defect as a reader with no access to the repository needs it. State what holds, what fails, and why. Do not point at the ticket that set the rule. `The clamp is a necessary and sufficient condition on the template.` is complete alone. `This closes #494 and #500.` sends the reader to a page that the docstring does not need.
- **A decision belongs in `docs/adr/`.** Link an ADR from another ADR, from `CODING_STANDARDS.md`, or from `GLOSSARY.md`. Never link one from a docstring, an example, a user-guide page or the Capability Catalogue.
- **`# References` cites only a published external source**, through `ref_dict`. A GitHub issue or pull request is neither published nor external in that sense, so it gets no bullet there either.
- **The contributor guide is the exception.** `docs/src/contribute/` and `docs/adr/` are written for a contributor, not a library user. They can cite an ADR, an issue or a PR.

---

## A docstring names a source by its citation

A reader who meets a source in a docstring must be able to open that source. So a docstring names a published work through a citation that links to the bibliography. It never names a work by words that point to nothing, such as "the paper", "the authors" or an author list with a year.

**Scope.** The prose of the docstrings of `src/**/*.jl` and `ext/**/*.jl`, and the dictionary values of [`src/01_Base/01_DocstringDictionaries/`](../../src/01_Base/01_DocstringDictionaries/) that they interpolate. The rule does not read a code span, a `jldoctest` block, a LaTeX expression or a `# References` bullet.

- **Cite the work in the sentence that uses it.** Write `[key](@cite)` when the work is a source for the sentence. Write `[key](@citet)` when the authors are the subject of the sentence. The key is the BibTeX key in `docs/src/References.bib`. The docs build uses the numeric style, so `[lihoi2012](@citet)` renders as "Li and Hoi [n]", and the number links to the bibliography.
- **Do not write an author list or a year in place of a citation.** `Györfi and Schäfer (2003)` and `Damian et al. 2023` link to nothing. Write `[gyorfischafer2003](@citet)`.
- **Do not refer to a work as "the paper", "the article", "the source paper" or "the authors".** Repeat the citation. Put a locator before the citation: `Equation 27 of [key](@cite)`.
- **State what the library takes from the work, and then what it adds.** First say which part of the work the unit makes: `Each rule makes one target of [schaferstrimmer2005](@cite).` Then state a generalisation or a new method in a different sentence.
- **Tell an implementer what to supply, not where it comes from.** Write ``subtype it with the parameters of the rule and a `proj` field``. Do not write `with the paper's parameters`. The reader of an abstract type has no paper.
- **Define each word that an argument or a field description uses.** Use a word of [`GLOSSARY.md`](../../GLOSSARY.md) or a word that the docstring defines. A private name for a value, such as "the carrier of the rule", tells the reader nothing about the value. Never use the words for the mechanism of the code that [`julia-prose.instructions.md`](julia-prose.instructions.md) lists, such as "carrier", "seam", "host", "read-out" and "door". Name the argument, the type or the function instead: "`rd`", "the [`ReturnsResult`](@ref)", "`optimise(opt)` with no data", "[`investable_fees_view`](@ref)". The rule holds on an error message too.
- **Do not cite in the first sentence of a type docstring.** The Capability Catalogue copies that sentence to a page that has no bibliography. Name the method in the first sentence, and cite its source in the second sentence.
- **Every work that the prose cites has a bullet in `# References` of the same docstring.** A private unit can cite. Its page under `docs/src/private_api/` then carries the bibliography block that [The `# References` Section](#the--references-section) states.

A method that has the name of a person, such as the Weiszfeld iteration or Welford's recursion, is a technical name. The name is not a citation, and the rule does not read it.

**The Gate.** [`test/test_74_source_citation_census.jl`](../../test/test_74_source_citation_census.jl) reads the literal text of each docstring under `src/` and `ext/`, field docstrings included, and each value in the docstring dictionaries. It fails on the words "paper", "article", "co-author" and "the authors", and on a name with a capital letter that a year in parentheses follows. The same census reads every string literal under `src/` and `ext/`, messages included, and fails on a word of the `mechanism` pattern of `code_health/prose.jl`. The census does not find an author list that has no year, or a private name for a value that the pattern does not list. Those two hold by review, in the sense of [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md). `test/test_26_docs.jl` checks that the key of each `[key](@cite)` and `[key](@citet)` is in `docs/src/References.bib`.

### The original source

A reviewer checks a formulation against its **original source**. The original source is the work that first states the formulation that the code builds. A book, a survey, a software manual or another library that restates the formulation is a **secondary source** of it. A check against a secondary source cannot find an error that the secondary source carries.

A work is secondary for one formulation, not by its kind. A book is the original source of a formulation that it states first. An article is a secondary source of a formulation that it takes from an earlier work.

- **Find the original.** Follow the bibliography of the secondary source back to the work that first states the formulation. Search the web when the bibliography is not enough.
- **Check the code against the original, and cite the original.** Cite a secondary source too only where it adds something, such as a generalisation or a conic encoding. The sentence that cites it says what it adds.
- **When the original cannot be found, write the standard note.** It closes the `# References` bullet of the secondary source that the docstring follows. [The `# References` Section](#the--references-section) states its form.

So a `# References` bullet with no note states that a reviewer checked every formulation that the docstring takes from that work against its original source.

---

## The prose passes `/unslop`

The prose of a docstring reads as a person wrote it for a reader. The `unslop` skill states the patterns that mark generated text. Every rule of the skill applies to the prose in the Scope of this section. **The skill is the Authority for the patterns.** This section cites it by name, copies none of its rules, and names a rule of the skill by the number the skill gives it.

**Scope.** The docstrings of `src/**/*.jl` and `ext/**/*.jl`, and the dictionary values of [`src/01_Base/01_DocstringDictionaries/`](../../src/01_Base/01_DocstringDictionaries/) that they interpolate. Every text a user reads as a page is outside this section: the hand-written Markdown pages under `docs/src/`, `README.md`, the `Prose` text of the Capability Catalogue, and the Literate sources of the examples and the user guide. [`.github/instructions/julia-prose.instructions.md`](julia-prose.instructions.md) owns them all.

**What the pass reads.** The pass reads the prose alone. That is the summary paragraph, the sentences of every section, and the text of every bullet and of every admonition. It does not read a code span, a `jldoctest` block, a LaTeX expression, a heading that a template of this file fixes, or a `# References` bullet, which `ref_dict` writes.

**Where this file and the skill meet.**

- A template of this file fixes the shape of a section, and the rules of the skill apply to the sentences inside it. A `# Fields` bullet, a `Where:` bullet and a `# Related` entry are list entries by template. Rule 33 asks each to carry its articles and its verb. It does not ask each to become a paragraph.
- A dictionary value is one text with many users. Rewrite it once, in its dictionary. Every docstring that interpolates it then changes with it.
- Rules 20 and 22 describe a chat reply. They never arise in a docstring.
- Every other rule applies as written.

**How to apply it.** Invoke `/unslop` on the file. When the skill cannot be invoked, read its rules and apply them by hand. Rewrite the prose, keep the meaning, and end with the skill's self-audit, "What makes this obviously AI generated?". A pass changes prose alone. A line of code, a `jldoctest` block and its output, a LaTeX expression and an interpolated key stay as they are.

**The Gate.** None. The rules that matter most, such as rules 27, 28 and 32, are judgements that no parser reads, so the rule holds by review, in the sense of [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md). A file whose sweep-manifest row reads `swept = true` owes the pass, and the file's sweep ticket carries it.

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

When `@propagatable` decorates a struct, its fields carry stackable tags inside the struct body. Each of three tags controls a generated method, and a reader of the type needs to know what that method does:

- `@fprop` selects **factory propagation**. [`factory`](@ref) on the enclosing struct passes its arguments on to the field. `factory_child` states the dispatch rules.
- `@vprop` selects **view propagation**. [`port_opt_view`](@ref) on the enclosing struct takes the view of the field along the **asset** axis. It recurses into composed children and slices data arrays.
- `@wprop` marks the **observation weights**. The field holds the weights. The tag controls **two** channels, and they do different things to the field. [`factory`](@ref) **replaces** it with an incoming [`ObsWeights`](@ref) value, and [`obs_weights_view`](@ref) **indexes** the value that is already there, along the **observation** axis.

`@pprop` and `@cprop` control the selection of the prior. The docstrings of the prior families document them, and this file does not.

A field can carry any combination of tags, such as `@fprop @vprop field`, in either order. The channels read different field sets. So the factory channel can propagate a field that the view channel passes through unchanged, and the reverse is also possible. **Document each channel in its own subsection inside `# Constructors`**, in the order below. Put the subsections **after** `## Validation`. When there is no `## Validation`, put them directly after "Keywords correspond to the struct's fields.".

**A subsection documents a channel, not a tag.** More than one tag can open a channel, so the field list of a subsection is the union of the tags that the channel reads. Write a subsection when at least one field carries a tag that opens its channel, and omit it otherwise. The view channel has one more trigger. A hand-written [`port_opt_view`](@ref) method needs the subsection even when no field carries `@vprop`.

`## Propagated parameters` documents the **factory** channel, which `@fprop` and `@wprop` open. It lists each field and how the channel propagates it:

- For an **observation-weight field**, which carries `@wprop` and has the type `ObsWeights`, `Nothing`, or `Option{<:ObsWeights}`, write `` `fieldname`: Replaced with the incoming [`ObsWeights`](@ref). ``
- For an **estimator, algorithm, or result field**, which carries `@fprop` and is a subtype of `AbstractEstimator`, `AbstractAlgorithm`, or `AbstractResult`, write `` `fieldname`: Recursively updated through [`factory`](@ref). ``
- For **the partial-fit state**, `@fprop cache` of the type `Option{<:AbstractPartialFitState}`, write `` `cache`: Carried unchanged through [`factory`](@ref). ``

`## View parameters` documents the **view** channel. Write it when the type has a [`port_opt_view`](@ref) method. [`@propagatable`](@ref) can generate that method from a `@vprop` tag, or the file can define one by hand. The two cases need different content.

**A generated method.** At least one field carries `@vprop`. List each tagged field and how the method takes its view:

- For an **estimator, algorithm, or result field**, a subtype of `AbstractEstimator`, `AbstractAlgorithm`, or `AbstractResult`, write `` `fieldname`: Recursively viewed through [`port_opt_view`](@ref). ``
- For a **data field**, an array, a scalar, or an `Option` of one, write `` `fieldname`: Sliced to the selected indices through [`port_opt_view`](@ref). ``
- For **the partial-fit state**, `@vprop cache`, write `` `cache`: Sliced to the selected assets through [`port_opt_view`](@ref). ``

**A hand-written method.** No field carries `@vprop`, and the file defines `port_opt_view(x::MyType, i, args...)`. A field list gives a wrong description of such a method. The method reads arguments that no tag can name, and it applies rules that no tag can encode. Open the subsection with

> `MyType` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

Then write a bullet list of high-level details. Cover these points, in this order, and only where the method has something to state:

- Which arguments beyond `i` the method reads, and what it does with them.
- Which fields recurse through [`port_opt_view`](@ref), and with which arguments.
- Which fields are sliced, and along which axis when that axis is not the asset axis.
- Which fields pass through unchanged, where a reader would expect otherwise.
- Any rule the method enforces or preserves that a field list cannot show.

Describe the method. Do not copy its code into prose. A field that the method passes through, with nothing to state about it, needs no bullet.

`## Observation weight parameters` documents the **observation** channel, which `@wprop` alone opens. A type with no `@wprop` field never has this subsection, whatever other tags it carries. The subsection lists each field and how the channel indexes it:

- For an **observation-weight field**, which carries `@wprop`, write `` `fieldname`: Indexed to the selected observations through [`obs_weights_view`](@ref). ``
- For an **estimator, algorithm, or result field**, which carries `@fprop`, write `` `fieldname`: Recursively indexed through [`obs_weights_view`](@ref). ``

> **The same field appears in two subsections, and the two state different things.** A `@wprop` field is *replaced* under `## Propagated parameters` and *indexed* under `## Observation weight parameters`. Do not merge the two. This is the one place where a reader learns that `factory` and `obs_weights_view` treat the field differently. A `@fprop` sibling also appears in both, and it recurses in both.

List the fields in the order of the struct body. Add [`factory`](@ref), [`port_opt_view`](@ref) or [`obs_weights_view`](@ref) to `# Related`, one for each subsection that is present. `test/test_83_propagation_list_census.jl` checks the two lists of the `factory` and `view` channels. Each list must name exactly the fields that its channel reads, in the order of the body.

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

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `nested_est`: Recursively updated through [`factory`](@ref).
  - `weight_field`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `nested_est`: Recursively viewed through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `nested_est`: Recursively indexed through [`obs_weights_view`](@ref).
  - `weight_field`: Indexed to the selected observations through [`obs_weights_view`](@ref).

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

An internal or private function can use `$(DocStringExtensions.TYPEDSIGNATURES)` as the header in place of a hand-written signature.

---

## Section Structure for Aliases

An **alias** is a second name for something another docstring already documents. Its docstring sends the reader to that documentation, and it restates none of it. A second copy drifts away from the first. The same argument put the field descriptions into `field_dict`.

Three kinds of alias exist, and the sections differ by kind.

| Kind | Declaration | Lives in | Header line | Sections it carries |
| --- | --- | --- | --- | --- |
| **Acronym alias** | `const HRP = HierarchicalRiskParity` | `src/23_Aliases.jl` only | the alias name alone | none |
| **Factory alias** | `MAD(; kwargs...)::LowOrderMoment` | `src/23_Aliases.jl` only | the signature, ending `-> T` | `# Validation`, and only when its own body raises |
| **Dispatch alias** | `const RhoDistanceAlgorithm = Union{...}` | any file under `src/` | the declaration, `const NAME = <type expression>` | `# Related`, and `# References` when the grouping itself is published |

A **dispatch alias** is a `const` bound to a type expression rather than to a bare name. A `Union`, a container such as `AbstractVector{<:LinearConstraint}`, and a parametrised form such as `const RMCVaR{T} = Union{...}` are all one kind. A caller meets all three the same way, as the type that a method signature dispatches on.

### The summary paragraph

- An **acronym alias** carries exactly one sentence, `Alias for [`Canonical`](@ref).` and nothing more. The alias and its target are the same object, so a second sentence describes the target and belongs on the target.
- A **factory alias** carries one sentence naming what it builds. That sentence `@ref`s every type the factory composes. A later sentence is permitted only for a choice that the composition fixes and that a reader can otherwise get wrong. Read `ZeroVarianceFilter` in [`src/23_Aliases.jl`](../../src/23_Aliases.jl), which states why it scores with `SCM()` and not with `Variance`.
- A **dispatch alias** carries what the alias groups and why the group exists. The *why* is the important half. A reader who sees only the member list learns nothing that the declaration does not already show.

### Why an alias carries so little

- **No `# Fields`, no `# Constructors`, no `# Arguments`, no `# Returns`, no `# Examples`, on any kind.** The canonical unit has each of them, and the summary sentence links to that unit. A copy on the alias is a second text to maintain. When the copy becomes stale, it is still the text that a reader finds first.
- **No `# Related` on an acronym alias or on a factory alias.** Its summary sentence already `@ref`s every canonical name. `## What each section holds` states that `# Related` does not hold a copy of the text of the related unit. Write each link once.
- **`# Related` on a dispatch alias** has one bullet for each grouped member. A group of members is a list, and a summary sentence must not be one. A function that dispatches on the alias may take a bullet of its own.
- **`# Algorithm`, `# Mathematical definition` and `# JuMP formulation` reach no alias.** An alias runs no steps and registers no model entry. A factory alias composes types and takes no branch worth numbering.

The Capability Catalogue lists an alias in `NOT_A_FEATURE` with the reason `:alias`, under [`.github/instructions/julia-source-code.instructions.md`](julia-source-code.instructions.md), so it never reads an alias summary sentence.

### The Gate

`test/test_26_docs.jl` gates the rule with three checks, in the shape [ADR 0086](../../docs/adr/0086-an-alias-docstring-links-its-canonical-unit-and-restates-nothing.md) records.

 1. **Library-wide, absolute.** No alias carries a section outside the row of its kind in the table above. Only a new breach can make this check fail.
 2. **A swept file, presence.** A dispatch alias in a file marked `swept = true` in [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml) carries `# Related`.
 3. **Library-wide, a count that may not rise.** The count of dispatch aliases carrying no `# Related` may not rise. The check retires when that count reaches zero.

Check 2 reads the `swept` flag because it demands a section. A demand for a section must not fail a file that no child map of issue #404 swept. Check 1 forbids a section instead, so it needs no flag.

---

## The `# Validation` Section

- Include a `## Validation` sub-section in struct docstrings and a `# Validation` section in function docstrings whenever the function or constructor enforces preconditions.
- Use `val_dict` entries for common validation rules.
- For custom validation, state the condition as code, such as `` `x > 0` ``.

---

## The `# References` Section

A docstring that rests on a published source names it. The section is **last**, after `# Related`, and it holds one bullet per work.

- **Never paste the reference prose.** Every bullet is one interpolation of `ref_dict` (`src/01_Base/01_DocstringDictionaries/13_References.jl`), which holds a single copy of each reference text:

  ```julia
  # References

    - $(ref_dict[:mlp1]) Chapter 2.
  ```

  A locator such as `Chapter 2.` may follow the interpolation. Nothing else may, except the standard note below.
- **The standard note marks a formulation whose original source was not found.** [The original source](#the-original-source) states when to write it. It closes the bullet of the secondary source that the docstring follows, after a locator that names the equation, the section or the page that the code follows:

  ```julia
  # References

    - $(ref_dict[:cajas2025]) Equation 3.45. $(ref_dict[:no_original_source])
  ```

  The note is one key of `ref_dict`, `no_original_source`, so its wording cannot drift and a search for the key lists every formulation that is still open. Its text is "The original source of this formulation was not found. The formulation follows this work and is not checked against the original." It is the one key of `ref_dict` that is not a key of `docs/src/References.bib`. The locator is mandatory, because a later search starts from it. A docstring that follows a secondary source for one formulation and an original source for another carries the note on the bullet of the secondary source alone, and the locator names the formulation.

  The note records the result of a search, so no test limits the number of notes. Remove a note when a later search finds the original. The docstring then cites the original, and a reviewer checks the code against it.
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

  `Canonical = false` is necessary. `docs/src/99_references.md` holds the one canonical block. A second canonical block skips every entry that the first block claimed, and it gives no warning. A `Pages` block on a page that cites nothing is a docs-**build error**, not a warning.
- An inline citation in prose, such as `as described in [DBHTs](@cite)`, needs no bullet of its own. The work still needs a `# References` bullet on the type that owns it.

The `"References"` testset in `test/test_26_docs.jl` checks the rules above. It also checks the form of the note. The note closes a `# References` bullet, a locator comes before it, and it appears nowhere else. The check of a formulation against its original source holds by review.

---

## `jldoctest` Examples

- All examples in docstrings must use `jldoctest` blocks, so that `julia --doctest` can test them.
- Output of pretty-printed structs must match exactly what `@define_pretty_show` produces.
- For abstract types with an `# Interfaces` section, the `jldoctest` must demonstrate a complete working implementation of the interface.
- Keep an example small, but complete enough to be useful.

---

## `docs/src/public_api/` and `docs/src/private_api/` Markdown Files

Each source file `src/SomeFeature.jl` has a corresponding page on both mirror trees:
`docs/src/public_api/SomeFeature.md` and `docs/src/private_api/SomeFeature.md`.
`classify_binding` in `docs/api_classification.jl` puts a symbol on exactly one of the two
(ADR 0128). A symbol is public when it is `export`ed, `public`-declared, or extends a foreign
generic such as `Base.iterate` or `StatsAPI.fit`. Every other symbol is private. A side with no
symbol still gets its page. That page holds a one-line note that the file has no public names, or
no private names.

List the symbol in a Documenter.jl `@docs` block under a suitable heading, on the mirror that its
classification names:

**Template.**

````markdown
## My section heading

```@docs
MyType
my_function
MyAbstractType
```
````

When you add a new symbol, add it to its mirror page too.
`test/test_65_docs_public_private_placement_census.jl` checks that every `@docs` entry is on the
side that its classification names.

**The H1 names the file of the page.** The H1 is the name of the page's file, without the numeric
prefix and the extension, with its words separated. Split the words at each underscore and at each
change of case: `01_Base_OnlinePortfolioSelection.md` takes `# Base Online Portfolio Selection`,
and `03_GerberIQCovariance_a.md` takes `# Gerber IQ covariance (a)`. The case, the spaces and the
punctuation are free, so `01_XatRisk.md` can take `# X at risk`. A word cannot change. Do not
expand `Exp` to `Exponentially`, and do not write `ℓ1` for `L1`. The H1 does not describe the page.
Put the description in the first paragraph under the H1. The private mirror adds `: private API`
to the same text. Documenter shows the H1 as the label of the page in the navigation, and a
description there hides the file that the page documents. The `Description` line of the page
comes from the H1, so a change to the H1 changes that line. Take the new line from
`mirror_description` in `docs/page_metadata.jl`. If the H1 carries an `@id` label, the label is
`api-` or `private-api-` followed by the H1 in lower case with hyphens between the words. Change
every `@ref` to the label in the same commit. `names_its_file` in `docs/page_metadata.jl` states
the rule, and `test/test_64_docs_page_metadata_census.jl` gates it on both trees.

---

## Mathematical Notation

When a type or function has a mathematical formulation, include a `# Mathematical definition` section immediately before `# Fields` (for structs) or before `# Arguments` (for functions).

### What the section may not state

The section states the mathematics and nothing else. It **names no identifier from the body, states no order of operations, and states no property that the implementation chose rather than the mathematics.** A consequence of the definition stays, because a consequence is mathematics.

`# Algorithm` has the reverse rule, *do not restate a closed form as a step*. The two rules mark one border from its two sides. So a fact that fails this rule is usually already a step, and the fix is to delete it here, not to move it.

**Example.** Compare these sentences with `ShrunkDenoise` in [`src/04_MatrixProcessing/02_Denoise.jl`](../../src/04_MatrixProcessing/02_Denoise.jl):

- Cut *the eigenvalues sorted ascending*. The sort is a choice of the body, and it is already an `# Algorithm` step.
- Cut *the diagonal is pinned to one afterwards to shed the eigendecomposition round-off*. That is an `# Algorithm` step too.
- Cut `vals`, `vecs`, `corr0`. Each names a local of the body.
- Keep *the two `alpha` weights sum to one on the diagonal, so the reconstruction preserves it in exact arithmetic*. That is a consequence of the definition, and it holds whatever the body does.

An implementation fact is not a token, so no parser finds one. This rule is **unenforced**. It holds by review, in the sense of [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md). That state is known, and it is not hidden.

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

All math blocks use `\begin{align}...\end{align}` with `&` alignment markers and `\\` line breaks. Put each equation on its own line. Use `\quad` only for spacing inside one equation. Put two equations on two lines, and never put `\qquad` between two equations in the same block.

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

Directly after each math block, add a `Where:` bullet list that defines every symbol. When many consecutive blocks belong to the same docstring, add the list after the **last** math block. Use `$(math_dict[:key])` for common variables.

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

- When many blocks are in the same docstring, one complete `Where:` list after the last block is acceptable.
- The list must define every symbol that appears in any block.
- Interpolate `$(math_dict[:key])` for standardised variables (``T``, ``\boldsymbol{x}_t``, ``\alpha``, etc.).
- If a key is missing from `math_dict`, first add it to the `*_Math*.jl` file of `src/01_Base/01_DocstringDictionaries/` that holds its subject.

---

### Notation is fixed by symbol and by family

The rules above fix the glyphs. The two below fix the content, so that two docstrings that state one quantity state it once and state it alike.

**A shared symbol becomes a `math_dict` key.** A symbol that appears in the docstrings of two or more Units gets a key in [`src/01_Base/01_DocstringDictionaries/`](../../src/01_Base/01_DocstringDictionaries/), and every site interpolates it. A symbol that exactly one Unit uses may stay inline, on the reasoning of [When a field description may be prose](#when-a-field-description-may-be-prose). One copy cannot drift. When a second Unit needs it, move it into `math_dict` and replace both copies with the interpolation.

A new description is a **new** key. An edit to a value already in `math_dict` changes every docstring that interpolates it. [ADR 0081](../../docs/adr/0081-the-docstring-standard-states-the-model-it-builds.md) gives the same reason for `arg_dict`.

**A key owns a definition, not a glyph.** One glyph carries different quantities in different families. ``\boldsymbol{w}`` is the portfolio weights vector in a risk measure and the observation weights in a moment estimator. So the whole definition claims a key. The definition is the symbol together with the sentence that defines it. A second quantity on the same glyph takes its own key under its own symbol. It never adds a second meaning to the first key.

**One quantity takes one key.** This is the converse of the rule above. Two keys that define one quantity are drift inside the table that exists to stop drift. The docstrings that read them state one thing in two spellings. `:w_t_moment`, `:w_t_obsweight` and `:cal_w_i` each defined the observation weight, under ``w_t``, ``w_{t}`` and ``w_{i}``, and as "observation ``t``" against "period ``i``". They are one key, `:w_t_obs`. A new key states a quantity that the table does not already carry. A new *description* of a quantity that the table carries is a change to the key that owns it, under the rule above.

**Gate.** Two checks in `test/test_26_docs.jl`, one for each direction. The first fails when a `Where:` bullet copies a `math_dict` value and does not interpolate it. It matches the whole bullet against the whole value. So it finds a copy, and it never fires on a glyph that two families share. A file marked `swept = true` in [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml) carries no such copy, and the library-wide count may not rise. The copies that remain move file by file, inside the sweep ticket of each file under issue #404.

The second check reads the table itself. It fails when two keys open with one definition head. The head is the noun phrase before the first punctuation mark and the first function word. That testset records each head that several keys share, with the reason that the keys are not one. So a new key may not restate a definition that the table already carries. A head is a coarse check. Two keys with different words do not match, and the sweep ticket of the file reads that pair by hand.

**Siblings of one Family state a shared quantity in the same form.** They use the same symbol and the same shape of equation. A condition that one sibling writes in a parenthesis, the second in set notation and the third in its `Where:` list is one fact written three ways, and the three do not read as one Family. Take the shape from the sibling that states it most completely, and write the others to match it.

The Family is the leaf-most abstract supertype, never a generic root. `RiskMeasure` and `AbstractResult` span many files and their members share no notation, so neither is a Family in this sense.

The form of an equation is not a token, so no parser finds a breach. This rule is **unenforced**. It holds by review, in the sense of [`CODING_STANDARDS.md`](../../CODING_STANDARDS.md). That state is known, and it is not hidden.

---

## The `# Algorithm` Section

Document a **procedure** in `# Algorithm`. A **closed form** stays in `# Mathematical definition`. A docstring can carry both sections, and a marker type carries neither.

`# Algorithm` sits immediately after `# Mathematical definition`, and before `# Fields` (for structs) or before `# Arguments` (for functions).

Rules:

- Write **numbered steps, one step per operation**. Each step names the quantity that the operation produces.
- Name each quantity by the name that the body gives it, so a reader can follow the steps in the code.
- Do not restate a closed form as a step. The formula belongs in `# Mathematical definition`, and the step that applies it names it.
- A **selector tag** is a type whose only job is to name the branch that a caller takes. The rule never *forces* a selector tag to carry either section. Its summary sentence states which branch it selects, and a tag that names nothing more stops there. Most subtypes of `AbstractAlgorithm` are selector tags, so the rule must never force numbered steps onto a marker type. A tag whose branch *is* a closed form states that form. The form goes under `# Mathematical definition`, and the steps of that branch go under `# Algorithm`. `SpectralDenoise` is the reference for that shape, and a row of the table in [Reference docstrings](#reference-docstrings) names it.

**Example.** The following is the algorithm of `denoise!(dn::Denoise, X::MatNum, q::Number)` in `src/04_MatrixProcessing/02_Denoise.jl`:

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

Any code that **builds part of a `JuMP.Model`** carries this section. It states the model that the code builds, and the mathematics alone does not state the model. A model entry has a name, a caller reads the entry back by that name, and the encoding is not always exact.

The trigger is mechanical. The section is necessary when the body calls `JuMP.@variable`, `JuMP.@variables`, `JuMP.@expression`, `JuMP.@expressions`, `JuMP.@constraint`, `JuMP.@constraints` or `JuMP.@objective`. **A row is not the only public entry.** The code registers `model[:sc]`, `model[:w]`, `model[:ret]` and `model[:risk]` as a variable or as an expression, and the library reads each one back by name. `set_model_scales!` registers `sc` and `so` for every builder that scales a row or an objective.

`# JuMP formulation` sits after `# Mathematical definition` and after `# Algorithm`, and before `# Fields` (for structs) or before `# Arguments` (for functions).

It has five subsections, in this order. Each of the first four is present whenever the body calls the macro that owns it. The fifth is present only when the encoding is not exact.

| Subsection | Present when the body calls | Holds |
| --- | --- | --- |
| `## Variables` | `JuMP.@variable`, `JuMP.@variables` | the model variables the code creates, and the ones it reads |
| `## Expressions` | `JuMP.@expression`, `JuMP.@expressions` | the expressions the code registers |
| `## Constraints` | `JuMP.@constraint`, `JuMP.@constraints` | the rows the code registers |
| `## Objective` | `JuMP.@objective` | the sense and the expression the code sets |
| `## Relaxation` | no macro | the bound, when the encoding is not exact |

`## Variables` is also permitted when the body only **reads** a variable. A formulation that reads `w` and never names it is unreadable.

The **last** of the first four subsections that is present closes with a `Where:` list. It defines every symbol the section uses, under the rules of the `Where:` section above. Interpolate `$(math_dict[:key])` for a standardised symbol. One list serves the whole section.

### `## Variables`

Write one bullet for each model variable that the code creates, and one for each model variable that it reads. Name each variable by its model key, and state whether the code reads it or creates it. Name a variable that the macro creates under no key by the symbol that the body binds it to.

### `## Expressions`

Write one bullet for each expression that the code registers, in the order in which the body registers them. Each bullet carries **the model key of the expression** and the mathematics of the expression. An expression that the macro does not name carries the key that the code stores it under. Name an expression that the function only returns by the returned value.

### `## Constraints`

Write **one bullet for each row that the code registers**, in the order in which the body registers them. Each bullet carries **the JuMP name of the row** and the mathematics of the row. The name is the one written in the `JuMP.@constraint` call, because that is the key with which a caller reads the row back out of the model.

### `## Objective`

Write one bullet. It states the sense that the code sets, `Min` or `Max`, and the expression that the code minimises or maximises, named by its model key. Two methods of one function can differ in nothing but the objective, and then this bullet is the only text that tells them apart.

### `## Relaxation`

Write this subsection **only when the encoding is not exact**. An exact encoding carries no `## Relaxation` subsection at all.

Open the subsection with `$(val_dict[:relax])`, so that the opening cannot drift from one docstring to the next. Then state three things:

 1. The **direction** of the bound: whether the model quantity lies above or below the exact quantity.
 2. The **quantity** that is bounded, named by its model key.
 3. The **condition** under which the bound is tight.

**A bound does not have to sit in a row.** `BrownianDistanceVariance` relaxes inside an expression, and the `Max` and log-sum-exp scalarisers put an upper bound in `model[:risk]` that is tight only while the objective minimises it.

**Example.** The following is the formulation of `set_gross_budget_constraints!` in `src/17_Optimisation/05_JuMP/02_JuMPConstraints/03_BudgetConstraints.jl`:

````julia
"""
# JuMP formulation

## Variables

  - `lw`, `sw`: long and short weight vectors, read from the model.
  - `k`: homogenisation scalar.

## Constraints

  - `gbgt_lb`: ``s_c \left(\sum lw + \sum sw - k b_l\right) \geq 0``
  - `gbgt_ub`: ``s_c \left(\sum lw + \sum sw - k b_u\right) \leq 0``

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - ``b_l``, ``b_u``: lower and upper gross budget bounds.
"""
````

**Example that registers no row.** The following is the formulation of `set_model_scales!` in `src/17_Optimisation/05_JuMP/01_Base_JuMPOptimisation/02_JuMPModelAccessors.jl`. It registers two expressions and nothing else, so it carries one subsection:

````julia
"""
# JuMP formulation

## Expressions

  - `sc`: ``s_c``, multiplied into both sides of every scaled row.
  - `so`: ``s_o``, multiplied into the objective.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:so_scale])
"""
````

---

## Reference docstrings

Read a real docstring, not a copy of one. Each row names a Unit whose file [`code_health/sweep_manifest.toml`](../../code_health/sweep_manifest.toml) marks `swept = true`. So a Gate holds the target, and the pointer cannot drift.

| Kind | Unit | File |
| --- | --- | --- |
| Abstract type | `AbstractDenoiseAlgorithm` | [`src/04_MatrixProcessing/02_Denoise.jl`](../../src/04_MatrixProcessing/02_Denoise.jl) |
| Selector tag | `SpectralDenoise` | [`src/04_MatrixProcessing/02_Denoise.jl`](../../src/04_MatrixProcessing/02_Denoise.jl) |
| Struct with fields | `ShrunkDenoise` | [`src/04_MatrixProcessing/02_Denoise.jl`](../../src/04_MatrixProcessing/02_Denoise.jl) |
| Public function | `denoise!` | [`src/04_MatrixProcessing/02_Denoise.jl`](../../src/04_MatrixProcessing/02_Denoise.jl) |
| Private function | `_denoise!` | [`src/04_MatrixProcessing/02_Denoise.jl`](../../src/04_MatrixProcessing/02_Denoise.jl) |
| Dispatch alias | `RhoDistanceAlgorithm` | [`src/06_Distance/02_Distance.jl`](../../src/06_Distance/02_Distance.jl) |
| `# JuMP formulation` | `set_distance_cone!` | [`src/17_Optimisation/10_OnlinePortfolioSelection/12_AllocationSetProgramme.jl`](../../src/17_Optimisation/10_OnlinePortfolioSelection/12_AllocationSetProgramme.jl) |
| `# JuMP formulation` with a `## Relaxation` | `ep_tail_dual_block!` | [`src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_a.jl`](../../src/10_Prior/06_EntropyPooling/03_EntropyPoolingPrior_a.jl) |
