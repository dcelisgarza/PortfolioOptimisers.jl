"""
    CodeHealth

Shared machinery for the four code-health entry scripts, `complexity.jl`, `expansion.jl`, `jet.jl`
and `coverage.jl`. It reads and writes the generated TOML files, compares provenance, applies the
ratchet, pairs renames, and renders a failure for a terminal and for GitHub Actions.

The decisions this module implements are recorded in ADRs 0071 to 0077 and ADR 0082 under
`docs/adr/`. The maintenance procedure that calls it is `docs/src/contribute/3-code-health.md`.
"""
module CodeHealth

using TOML

export Definition, Reviewed, Rise, RefreshRefused, run_script

# --- what one measurement says about one definition -------------------------
#
# The two record types below are the vocabulary the measuring scripts and the scheduled job share.
# They live here rather than in the script that fills them because `code_health/triage.jl` reads
# both, and a type is only one type when one module defines it.

"""
    Definition

One definition, as `code_health/complexity.jl` measures it and the scheduled job reports it. ADR
0078 asks the issue body for the worst definitions **by name and line**, so the line is carried
through the measurement rather than measured a second time. Nothing here reaches a baseline file: a
row records the maximum and the sum alone.
"""
struct Definition
    name::String
    value::Int
    line::Int
end

"""
    Reviewed

One JET report that survived the Dismissals, as `code_health/jet.jl` measures it and the scheduled
job reports it: the run it came from, its kind, its message, and the line of the frame the
attribution chose.

ADR 0078 asks the issue body for the reports **by kind and site**. The line is a reader's
convenience and is never part of the Report Fingerprint, which ADR 0071 keys on the file, the kind
and the message alone.
"""
struct Reviewed
    run::String
    kind::String
    message::String
    line::Int
end

# --- paths and scope -------------------------------------------------------

const DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(DIR, ".."))
const RULINGS_PATH = joinpath(DIR, "rulings.toml")
const ARTIFACT_DIR = joinpath(DIR, "_refresh")

"""
The measured roots of ADR 0072. A tracked `.jl` file outside them must be a named Unmeasured Path.
"""
const MEASURED_ROOTS = ("src/", "ext/")

"""
The four files that declare a Declaration Macro, per ADR 0072.
"""
const DECLARING_FILES = ("src/01_Base/03_PrettyShow.jl", "src/02_Tools.jl",
                         "src/08_Moments/01_Base_Moments.jl",
                         "src/20_Optimisation/01_Base_Optimisation.jl")

in_scope(path::AbstractString) = any(r -> startswith(path, r), MEASURED_ROOTS)

is_ci() = haskey(ENV, "GITHUB_ACTIONS")

# --- reading source text ---------------------------------------------------
#
# Every census in this repository reads the SOURCE TEXT rather than a loaded binding: one parse per
# file, no package loaded, and a reading that cannot move under a reformat. The recursion and the
# docstring predicate are written once here. `code_health/`, `test/test_26_docs.jl`,
# `test/test_41_constructor_docstring_drift.jl`, `test/test_45_sweep_census.jl` and
# `test/test_47_alias_and_module_census.jl` all cross this seam to reach them.
#
# This module loads only `TOML`, which is stdlib and which `test/Project.toml` already carries, so
# a test file includes this one file at no dependency cost. `test/test_49_coverage_attribution_
# census.jl` loads `code_health/coverage.jl` the same way.

"""
    parse_file(file) -> Expr

Parse one file with `Meta.parseall`. A relative `file` is read from the repository root. An
absolute one is read as it stands, which is what a test file under `test/` passes.
"""
parse_file(file) = Meta.parseall(read(joinpath(REPO_ROOT, file), String); filename = file)

"""
    Prune

The one value a [`walk_ast`](@ref) visitor returns to stop the descent below the node it was given.
It is a type of its own rather than `false`, because a visitor's last expression is whatever its
body happened to leave behind, and a stray `false` must never prune the walk.
"""
struct Prune end

"""
The single [`Prune`](@ref).
"""
const PRUNE = Prune()

"""
    walk_ast(f, node)

Apply `f` to `node` and to every `Expr` under it, parents before children and in source order. A
non-`Expr` is skipped, so a visitor never sees a `Symbol`, a literal or a `LineNumberNode`.

Return [`PRUNE`](@ref) from `f` to leave the subtree below that node unvisited. Every other return
value is ignored.
"""
function walk_ast(f, node)
    if !(node isa Expr)
        return nothing
    end
    if f(node) === PRUNE
        return nothing
    end
    for a in node.args
        walk_ast(f, a)
    end
    return nothing
end

"""
The `Core.@doc` macro, as the parser writes it. A docstring at the top level of a file parses to
this `GlobalRef`. One inside a `module` block parses to the bare `Symbol`.
"""
const DOC_MACRO = GlobalRef(Core, Symbol("@doc"))

"""
    isdocstring(x) -> Bool

Whether `x` is the macrocall a docstring parses to. A FIELD docstring is not one: inside a struct
body a docstring parses as a bare string literal, so it never answers `true` here.
"""
function isdocstring(x)
    return Meta.isexpr(x, :macrocall) &&
           !(isempty(x.args)) &&
           (x.args[1] === DOC_MACRO || x.args[1] === Symbol("@doc"))
end

"""
    docstring_text(x) -> String

The literal prose of the docstring `x`, or `""` when it carries none.

**This is the one reader.** `test/test_26_docs.jl` reads a section heading with it and
`test/test_47_alias_and_module_census.jl` reads a summary sentence and an `@ref` with it, so the
rule for what counts as prose moves in one edit.

A docstring that interpolates parses to an `Expr(:string, ...)` rather than to a `String`, and a
section heading is a literal line inside it. So the literal pieces alone carry every heading and
every `@ref`, and an interpolated piece is replaced by a space rather than dropped.
"""
function docstring_text(x)
    for a in x.args[2:end]
        if a isa AbstractString
            return String(a)
        end
        Meta.isexpr(a, :string) &&
            return join(p isa AbstractString ? p : " " for p in a.args)
    end
    return ""
end

# --- the name a definition binds --------------------------------------------
#
# **This is the one resolver.** `code_health/coverage.jl` writes the `definition` key of a
# Coverage Exemption with it, and `test/test_26_docs.jl` keys a docstring by the name it
# binds with it. Issue #521 corrected the `::` branch in one of the two copies that used to
# stand here, and the other kept naming a functor method after its receiver VARIABLE, so
# every functor with the same receiver collapsed onto one key.
# `test/test_49_coverage_attribution_census.jl` is the differential oracle over this code.

"""
    is_signature(e) -> Bool

Whether an expression is the left side of a **definition** rather than the left side of an
assignment. `f(x)`, `f(x) where {T}` and `f(x)::T` are signatures. `x` and `x::T` are not.

A return type is what makes the two hard to tell apart, and telling them apart is the whole of issue
#521. `f(x)::T` and the receiver slot `r::T` of a functor method are the same `Expr` head, so the
side that carries the name differs between them, and `x::T = 1` is an assignment that looks like
both.
"""
function is_signature(e)
    if !(e isa Expr)
        return false
    end
    if e.head === :call || e.head === :where
        return true
    end
    if e.head === :(::)
        return is_signature(e.args[1])
    end
    return false
end

defname(x::Symbol) = String(x)
defname(x::QuoteNode) = defname(x.value)
# A docstring parses to `Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), …)`, and a `GlobalRef`
# is neither a `Symbol` nor an `Expr`. Without this method every documented definition in the
# library falls to the empty default and is attributed to `<toplevel>`.
defname(x::GlobalRef) = defname(x.name)
function defname(e::Expr)
    if isempty(e.args)
        return ""
    end
    if e.head === :call
        return defname(e.args[1])
    end
    if e.head === :where
        return defname(e.args[1])
    end
    if e.head === :(<:)
        return defname(e.args[1])
    end
    if e.head === :(::)
        # The two shapes a `::` node arrives in are named from opposite sides. A signature with a
        # declared return type, `f(x)::T`, is named `f`, because `T` names no method and one file
        # holds many methods that return the same type. A receiver slot, `(r::T)(x)`, is named `T`,
        # because that is what a reader calls the functor method. Issue #521.
        return defname(is_signature(e.args[1]) ? e.args[1] : e.args[end])
    end
    if e.head === :curly
        return defname(e.args[1])
    end
    if e.head === :.
        return defname(e.args[end])
    end
    if e.head === :macrocall
        return defname(e.args[1])
    end
    return ""
end
defname(x) = ""

"""
    unwrap(e) -> Expr

A docstring parses as a `Core.@doc` macro call wrapping the definition, so a documented definition
would otherwise be named after the doc macro. Nearly every definition in this library is documented,
so dropping the wrapper is not a corner case.
"""
function unwrap(e)
    if isdocstring(e)
        return unwrap(e.args[end])
    end
    return e
end

"""
    definition_name(e) -> String

The name a Coverage Exemption row writes for one top-level expression, or `""` when the expression
declares nothing a reader would name.
"""
function definition_name(e0)
    e = unwrap(e0)
    if !(e isa Expr)
        return ""
    end
    if e.head in (:function, :macro)
        return defname(e.args[1])
    elseif e.head === :(=) && is_signature(e.args[1])
        # A short form that declares a return type, `f(x)::Bool = true`, is a definition and not an
        # assignment. Reading it as an assignment attributed its lines to `<toplevel>`, which is the
        # second face of issue #521: a Coverage Exemption could name neither method.
        return defname(e.args[1])
    elseif e.head === :struct
        # `Expr(:struct, mutable, name, body)`.
        return length(e.args) >= 2 ? defname(e.args[2]) : ""
    elseif e.head in (:abstract, :primitive)
        # `Expr(:abstract, :(T <: S))` carries one argument, not the three a struct carries.
        return isempty(e.args) ? "" : defname(e.args[1])
    elseif e.head === :const
        return e.args[1] isa Expr ? defname(e.args[1].args[1]) : defname(e.args[1])
    elseif e.head === :macrocall
        # A Declaration Macro wraps the definition it declares, so name the definition. Naming the
        # macro instead would collapse every `@concrete` call in a file onto one ambiguous key.
        # A macro that declares nothing, such as `@define_pretty_show T`, keeps its own name.
        inner = definition_name(e.args[end])
        return isempty(inner) ? defname(e.args[1]) : inner
    end
    return ""
end

# --- Declaration Macros ----------------------------------------------------
#
# Found by parsing alone, so `complexity.jl` stays a pure parser and loads no package.

macro_name(x::Symbol) = String(x)
macro_name(x::Expr) = x.head === :. ? macro_name(x.args[end]) : ""
macro_name(x::QuoteNode) = macro_name(x.value)
macro_name(x) = ""

"""
    declared_macros(file) -> Vector{String}

The macros a file declares.
"""
function declared_macros(file)
    names = String[]
    walk_ast(parse_file(file)) do e
        if e.head === :macro
            sig = e.args[1]
            n = sig isa Expr ? sig.args[1] : sig
            if n isa Symbol
                push!(names, "@" * String(n))
            end
        end
        return nothing
    end
    return names
end

"""
    macro_call_sites(file) -> Vector{Tuple{String, Expr}}

Every macro call a file makes outside a struct body, with the name and the call expression. The
five field-prefix macros — `@fprop`, `@vprop`, `@pprop`, `@cprop` and `@wprop` — declare nothing of
their own and appear only inside a struct body, so this is what separates them from the seven
Declaration Macros of ADR 0072.
"""
function macro_call_sites(file)
    found = Tuple{String, Expr}[]
    walk_ast(parse_file(file)) do e
        if e.head === :struct
            return PRUNE
        end
        if e.head === :macrocall
            n = macro_name(e.args[1])
            if !(isempty(n))
                push!(found, (n, e))
            end
        end
        return nothing
    end
    return found
end

called_macros(file) = Set(n for (n, _) in macro_call_sites(file))

"""
    declaration_macros(files) -> Vector{String}

A **Declaration Macro** turns a declaration into definitions the parser cannot see. ADR 0072 names
seven of them and sites all seven in four declaring files. The list is derived rather than written
down, so a new one is found on the day it is declared and called, which is what ADR 0074's rule for
the Expansion Bound's key set needs.
"""
function declaration_macros(files)
    declared = Set{String}()
    for f in DECLARING_FILES
        union!(declared, declared_macros(f))
    end
    called = Set{String}()
    for f in files
        union!(called, called_macros(f))
    end
    return sort!(collect(intersect(declared, called)))
end

"""
    names_path(title, path) -> Bool

Whether an issue title names `path` as a whole word. Both scheduled jobs search the titles they
already hold rather than asking the tracker a second time, so the search happens here.

The test is a token match rather than a substring: a substring test would let a title naming
`src/A.jl.orig` skip `src/A.jl` for ever. ADR 0078 states it for the code-health job and ADR 0084
reuses it for the sweep job, so it is defined once.
"""
function names_path(title::AbstractString, path::AbstractString)
    return any(==(path), split(title, r"[\s`,;()\[\]]+"; keepempty = false))
end

# --- git -------------------------------------------------------------------

"""
    tracked_jl_files() -> Vector{String}

Every tracked `.jl` file, as a path relative to the repository root. ADR 0072's coverage assertion
and ADR 0074's set equality both read this list. `git ls-files` is used rather than `walkdir`
because `measure_directory` ignores `.gitignore` and picks up the untracked `NOTRACK_*` scratch
files (issue #336).
"""
function tracked_jl_files()
    out = read(Cmd(`git ls-files -z -- '*.jl'`; dir = REPO_ROOT), String)
    return sort!(filter!(!isempty, split(out, '\0')))
end

"""
    source_files() -> Vector{String}

Every tracked `.jl` file under the measured roots, as a path relative to the repository root. This
is ADR 0074's expected row set for `sweep/manifest.toml`, and `test/test_45_sweep_census.jl`
compares it against the rows the manifest holds.
"""
source_files() = String[f for f in tracked_jl_files() if in_scope(f)]

function git_short_commit()
    try
        return strip(read(Cmd(`git rev-parse --short HEAD`; dir = REPO_ROOT), String))
    catch
        return "unknown"
    end
end

# --- the sweep manifest ----------------------------------------------------

"""
    documented_units(path) -> Int

The file's count of documented units: a docstring that attaches to a binding, counted from the
source text by parsing with `Meta.parseall` and counting the `Core.@doc` macrocalls at any depth.
It is the `units` key of a `sweep/manifest.toml` row.

**This is the one definition.** `test/test_45_sweep_census.jl` states what a unit is and why, and
`code_health/sweep_check.jl` measures the same number before the commit. Both call this function,
so the rule moves in one edit.

A field docstring is NOT a unit: inside a struct body a docstring parses as a bare string literal
rather than as a macrocall, so it never reaches this count.
"""
function documented_units(path::AbstractString)
    n = 0
    walk_ast(parse_file(path)) do node
        if isdocstring(node)
            (n += 1)
        end
        return nothing
    end
    return n
end

"""
    row_line(f, m, u, s; algorithm) -> String

The `sweep/manifest.toml` line a person pastes back. A swept row also carries the `algorithm` key
that `test/test_26_docs.jl` ratchets, so the printer takes it: a line pasted without that key would
delete the ratchet's floor, and the deletion would read as a correction. An unswept row has no such
key, and a file that has no row at all is never swept.

`test/test_45_sweep_census.jl` and `code_health/sweep_check.jl` both print this line, before the
commit and after it, so the two printers are one printer.
"""
function row_line(f, m, u, s; algorithm = nothing)
    a = isnothing(algorithm) ? "" : string(", algorithm = ", algorithm)
    return string("\"", f, "\" = { map = ", m, ", units = ", u, a, ", swept = ", s, " }")
end

"""
    candidate_maps(rows, map_names, f) -> Vector{Int}

The child maps a file's own directory already uses. **A `map` is not derivable from a path**, so the
census prints the candidates and a person chooses by subject. Each of the nine subdirectories of
`src/` and `ext/` uses exactly one map, and there the answer is an answer. The top level of `src/`
holds sixteen files across five maps, and the numeric prefix does not rescue the lookup: the blocks
are not contiguous. `10_` sits between map 2 and map 8, and `25_` returns to map 1. #428 measured
this.

A file in a brand-new directory has no sibling row, and then every map is a candidate.
"""
function candidate_maps(rows, map_names, f::AbstractString)
    d = dirname(f)
    ms = sort(unique(r["map"] for (g, r) in rows if dirname(g) == d))
    return isempty(ms) ? sort(parse.(Int, collect(keys(map_names)))) : ms
end

# --- the sweep, as the two scheduled jobs name it ---------------------------
#
# `code_health/sweep_check.jl` runs before the commit and `code_health/sweep_triage.jl` runs on a
# schedule. They read the same two files and write into the same tracker, so the five names below
# are stated here once and aliased into each script.

"""
The umbrella of the map of maps. It is reopened with every child map either job reopens, because a
child map that reopens makes the umbrella's own terminal condition false again.
"""
const SWEEP_UMBRELLA = 404

"""
The label ADR 0084 puts on every sweep issue. Both jobs search the titles that carry it.
"""
const SWEEP_LABEL = "sweep"

"""
Where the sweep jobs write their plan.
"""
const SWEEP_PLAN_DIR = joinpath(DIR, "_sweep")

"""
The sweep manifest of ADR 0074: one row per file under the measured roots.
"""
const MANIFEST_PATH = joinpath(REPO_ROOT, "sweep", "manifest.toml")

"""
The coverage baseline of ADR 0082, which both sweep jobs read for a file's coverage row.
"""
const COVERAGE_BASELINE_PATH = joinpath(DIR, "coverage_baseline.toml")

# --- the tracker dump ------------------------------------------------------

"""
    read_tracker_dump(path, fields) -> Vector{Vector{String}}

The issues a workflow dumped, one per line, as `fields` tab-separated columns. An absent file means
an empty tracker, which is what the first run sees. A blank line is skipped, and a line of the
wrong width raises with its number, because a short line would otherwise be read as a missing issue
and the job would file a duplicate.

`code_health/triage.jl` dumps four columns and `code_health/sweep_triage.jl` dumps three, so the
width is an argument. The parse is the same parse.
"""
function read_tracker_dump(path::AbstractString, fields::Integer)
    out = Vector{String}[]
    if !(isfile(path))
        return out
    end
    for (i, line) in enumerate(eachline(path))
        if isempty(strip(line))
            continue
        end
        parts = split(line, '\t')
        if !(length(parts) == fields)
            error("$path line $i has $(length(parts)) fields, not $fields: " * repr(line))
        end
        push!(out, String.(parts))
    end
    return out
end

# --- TOML rendering --------------------------------------------------------
#
# ADR 0073 rules that the generator prints the lines itself. `TOML.print` emits one line per key in
# hash order, which makes every regeneration a whole-file diff.

toml_scalar(x::Integer) = string(x)
toml_scalar(x::AbstractString) = "\"" * escape_string(x) * "\""
toml_scalar(x::Bool) = x ? "true" : "false"
toml_scalar(x::AbstractVector) = "[" * join(map(toml_scalar, x), ", ") * "]"

"""
    inline_table(pairs) -> String

Render `pairs` as a TOML inline table, keeping the order given. The stdlib parser accepts an inline
table; `TOML.print` cannot write one.
"""
function inline_table(pairs)
    body = join(("$k = $(toml_scalar(v))" for (k, v) in pairs), ", ")
    return "{ " * body * " }"
end

toml_key(k::AbstractString) = "\"" * escape_string(k) * "\""

"""
    emit_section(io, header, entries)

Write one TOML table whose every entry is a keyed inline table, sorted by key.
"""
function emit_section(io::IO, header::AbstractString, entries)
    println(io, "[", header, "]")
    for (k, v) in sort(collect(entries); by = first)
        println(io, toml_key(k), " = ", inline_table(v))
    end
    return nothing
end

function emit_provenance(io::IO, entries)
    println(io, "[provenance]")
    for (k, v) in entries
        println(io, k, " = ", toml_scalar(v))
    end
    return nothing
end

read_toml(path::AbstractString) = isfile(path) ? TOML.parsefile(path) : Dict{String, Any}()

# --- rulings ---------------------------------------------------------------

"""
    read_rulings() -> Dict

The hand-written file of ADR 0073: the thresholds, the Unmeasured Paths, the Exemptions, the
Rationales and the Dismissals. Nothing in it is measured.
"""
function read_rulings()
    if !(isfile(RULINGS_PATH))
        error("code_health/rulings.toml is missing. The gate cannot read its thresholds.")
    end
    return TOML.parsefile(RULINGS_PATH)
end

thresholds(rulings) = rulings["thresholds"]

"""
    unmeasured_paths(rulings) -> Vector{String}

The path prefixes ADR 0072 names as never measured.
"""
unmeasured_paths(rulings) = String[e["path"] for e in get(rulings, "unmeasured_path", [])]

"""
    exempt_definitions(rulings, metric) -> Set{Tuple{String, String}}

The Exemptions for one metric, keyed `(path, definition)`. ADR 0072 and issue #356 bind an
Exemption to candidacy alone. It is dropped before the file's maximum is taken for the candidacy
test, and it never touches the baseline.
"""
function exempt_definitions(rulings, metric::AbstractString)
    keys = Set{Tuple{String, String}}()
    for e in get(rulings, "exemption", [])
        if e["metric"] == metric
            push!(keys, (e["path"], e["definition"]))
        end
    end
    return keys
end

"""
    check_rationale_citations(rulings)

Every Dismissal, every Exemption and every Coverage Exemption must cite a Rationale that the file
defines. ADR 0071 makes a new Rationale the maintainer's act, so an unknown citation is a broken
record, not a new claim.
"""
function check_rationale_citations(rulings)
    known = Set(keys(get(rulings, "rationale", Dict{String, Any}())))
    bad = String[]
    for (kind, entries) in (("dismissal", get(rulings, "dismissal", [])),
                            ("exemption", get(rulings, "exemption", [])),
                            ("coverage_exemption", get(rulings, "coverage_exemption", [])))
        for e in entries
            r = get(e, "rationale", nothing)
            if r === nothing || !(r in known)
                push!(bad, "$kind entry cites unknown rationale $(repr(r))")
            end
        end
    end
    return bad
end

# --- coverage --------------------------------------------------------------

"""
    coverage_failures(rulings) -> Vector{String}

ADR 0072's total-coverage assertion. Every tracked `.jl` file is measured or is matched by an
Unmeasured Path. A file that is neither turns the gate red, so a new top-level directory cannot
fall through unmeasured and silent.
"""
function coverage_failures(rulings)
    prefixes = unmeasured_paths(rulings)
    bad = String[]
    for f in tracked_jl_files()
        if in_scope(f)
            continue
        end
        if any(p -> startswith(f, p), prefixes)
            continue
        end
        push!(bad, f)
    end
    return bad
end

# --- provenance ------------------------------------------------------------

"""
    provenance_failures(recorded, measured) -> Vector{String}

ADR 0073 compares every provenance field except `commit`, and any mismatch fails. The baseline is
always older than the tree, so the commit is context for a reader and never binds.
"""
function provenance_failures(recorded, measured)
    bad = String[]
    for (k, v) in measured
        if k == "commit"
            continue
        end
        r = get(recorded, k, nothing)
        if r != v
            push!(bad, "  $k: baseline $(repr(r)), now $(repr(v))")
        end
    end
    return bad
end

function provenance_message(bad)
    io = IOBuffer()
    println(io, "The analyser moved under the baseline.")
    for line in bad
        println(io, line)
    end
    print(io,
          "Refresh the baseline in the SAME commit that moves code_health/Manifest.toml.")
    return String(take!(io))
end

# --- the ratchet -----------------------------------------------------------

"""
    Rise

One number that stands above the number the baseline records for it. `group` names the JET run, or
is empty where the generated file has no runs.
"""
struct Rise
    group::String
    key::String
    metric::String
    old::Int
    new::Int
end

"""
    rises(recorded, measured, metrics; group = "") -> Vector{Rise}

ADR 0076's pass rule, applied per key and per metric. The comparison is never per total, so a fall
in one file cannot pay for a rise in another. Keys absent from either side are the business of
ADR 0074's set equality and are skipped here.
"""
function rises(recorded, measured, metrics; group::AbstractString = "")
    out = Rise[]
    for (key, new) in measured
        old = get(recorded, key, nothing)
        if old === nothing
            continue
        end
        for m in metrics
            if new[m] > old[m]
                push!(out, Rise(group, key, m, old[m], new[m]))
            end
        end
    end
    return sort!(out; by = r -> (r.group, r.key, r.metric))
end

# --- set equality and rename pairing ---------------------------------------

"""
    set_differences(expected, recorded) -> (missing_rows, dead_rows)

ADR 0074's rule. `missing_rows` are in-scope keys the generated file does not name, and `dead_rows`
are rows that name nothing. Both are the same failure, and the gate never invents a number.
"""
function set_differences(expected, recorded)
    exp_set, rec_set = Set(expected), Set(recorded)
    return sort!(collect(setdiff(exp_set, rec_set))),
           sort!(collect(setdiff(rec_set, exp_set)))
end

"""
    pair_renames(dead, added, recorded, measured, equal) -> (pairs, unpaired_dead, unpaired_added)

ADR 0074 pairs a dead row with an unnamed file when `equal` holds between the recorded row and the
measured one, and carries the row across with no flag. The pairing is safe by arithmetic: when the
numbers are equal it does not matter which dead row takes which new path, so the multiset of
recorded numbers cannot rise.
"""
function pair_renames(dead, added, recorded, measured, equal)
    pairs = Pair{String, String}[]
    left, right = copy(dead), copy(added)
    for d in dead
        i = findfirst(a -> equal(recorded[d], measured[a]), right)
        if i === nothing
            continue
        end
        a = right[i]
        push!(pairs, d => a)
        deleteat!(right, i)
        deleteat!(left, findfirst(==(d), left))
    end
    return pairs, left, right
end

# --- refusing a refresh ----------------------------------------------------

"""
    RefreshRefused

A refresh that must not write a file. ADR 0075 makes the Refresh Artifact whatever the refresh
wrote, so a refused refresh publishes nothing.
"""
struct RefreshRefused <: Exception
    message::String
end

Base.showerror(io::IO, e::RefreshRefused) = print(io, e.message)

"""
    refuse_rise(kind, rises)

ADR 0073's refresh contract. A bare refresh lowers a number, so its diff is always an improvement
and it is safe to run without thought. Recording a rise is a second, named act.
"""
function refuse_rise(kind::AbstractString, rs::Vector{Rise})
    io = IOBuffer()
    for r in rs
        g = isempty(r.group) ? "" : "[$(r.group)] "
        println(io, "ERROR: $g$(r.key) $(r.metric) rose $(r.old) -> $(r.new).")
    end
    print(io, "Re-run with --accept-rise to record ", length(rs), " rise(s) in $kind.")
    return throw(RefreshRefused(String(take!(io))))
end

# --- reporting -------------------------------------------------------------

"""
    annotate(path, message)

One `::error` annotation per offending file, per ADR 0077. Outside CI the annotation syntax is
dropped and the same text is printed plainly.
"""
function annotate(path::AbstractString, message::AbstractString)
    if is_ci()
        println("::error file=", path, "::", message)
    else
        println("  ", path, ": ", message)
    end
    return nothing
end

function step_summary(text::AbstractString)
    path = get(ENV, "GITHUB_STEP_SUMMARY", "")
    if !(isempty(path))
        open(io -> println(io, text), path, "a")
    end
    return nothing
end

"""
    rise_table(rises) -> String

The markdown table ADR 0077 writes to the step summary. It names every offending file, including
one whose number rose because a Declaration Macro in another file changed, since such a file is
absent from the diff and gets no annotation.
"""
function rise_table(rs::Vector{Rise})
    io = IOBuffer()
    grouped = any(r -> !isempty(r.group), rs)
    println(io, if grouped
                "| run | file | metric | baseline | now |"
            else
                "| file | metric | baseline | now |"
            end)
    println(io, grouped ? "| --- | --- | --- | --- | --- |" : "| --- | --- | --- | --- |")
    for r in rs
        if grouped
            println(io, "| ", r.group, " | ", r.key, " | ", r.metric, " | ", r.old, " | ",
                    r.new, " |")
        else
            println(io, "| ", r.key, " | ", r.metric, " | ", r.old, " | ", r.new, " |")
        end
    end
    return String(take!(io))
end

"""
    routes(; dismissal) -> String

ADR 0075's one message for everyone, in its three ordered routes. Route 2 is named in CI rather
than left to the documentation, because a new Rationale needs the maintainer and that is the one
route a contributor without write access cannot finish alone. The complexity message drops it,
because a Dismissal is a JET-only instrument.
"""
function routes(; dismissal::Bool)
    io = IOBuffer()
    println(io, "A refresh is not the fix. Take one of these routes, in order.")
    println(io, "  1. Lower the number.")
    if dismissal
        println(io,
                "  2. Add a Dismissal to code_health/rulings.toml citing an approved Rationale.")
        println(io,
                "     Say so in the pull request if the Rationale is new: a new one needs the maintainer.")
        println(io,
                "  3. Record the rise. Download the Refresh Artifact from this run, put the file at")
        println(io,
                "     its committed path under code_health/, and commit it. Locally the same act is")
        println(io,
                "     `julia --project=code_health code_health/<tool>.jl refresh --accept-rise`.")
    else
        println(io,
                "  2. Record the rise. Download the Refresh Artifact from this run, put the file at")
        println(io,
                "     its committed path under code_health/, and commit it. Locally the same act is")
        println(io,
                "     `julia --project=code_health code_health/<tool>.jl refresh --accept-rise`.")
    end
    print(io, "See docs/src/contribute/3-code-health.md.")
    return String(take!(io))
end

# --- the command line ------------------------------------------------------

struct Command
    verb::Symbol
    accept_rise::Bool
end

function parse_command(args)
    usage = "usage: <tool>.jl check | refresh | refresh --accept-rise"
    if isempty(args)
        error(usage)
    end
    verb = args[1]
    if verb == "check"
        if !(length(args) == 1)
            error(usage)
        end
        return Command(:check, false)
    elseif verb == "refresh"
        if length(args) == 1
            return Command(:refresh, false)
        elseif length(args) == 2 && args[2] == "--accept-rise"
            return Command(:refresh, true)
        end
    end
    return error(usage)
end

"""
    write_artifact(name, text)

Stage the Refresh Artifact of ADR 0075: the generated baseline as `refresh --accept-rise` would
write it, uploaded from the failing run. The contributor puts the file at its committed path and
commits it, and needs no Julia at all.
"""
function write_artifact(name::AbstractString, text::AbstractString)
    mkpath(ARTIFACT_DIR)
    path = joinpath(ARTIFACT_DIR, name)
    write(path, text)
    println("Wrote the Refresh Artifact to code_health/_refresh/", name, ".")
    return path
end

"""
    run_script(; name, measure, render, verify, refresh)

The one flow the three entry scripts share.

`measure` returns the measurement. `render(measurement, recorded, accept_rise)` returns the text
the refresh would write, or throws `RefreshRefused`. `verify(measurement, recorded)` returns the
failure text of a check, empty when the check is green, and a flag saying whether provenance
matched. A provenance mismatch publishes nothing, because the numbers came from the wrong tools.
"""
function run_script(args; name::AbstractString, measure, verify, render, publish)
    cmd = parse_command(args)
    path = joinpath(DIR, name)
    recorded = read_toml(path)
    measurement = measure()
    if cmd.verb === :refresh
        local text
        try
            text = render(measurement, recorded, cmd.accept_rise)
        catch e
            # A refused refresh is an ordinary answer, not a crash. ADR 0073 fixes the text, and a
            # Julia stacktrace on top of it would bury the one line that says what to do.
            if !(e isa RefreshRefused)
                rethrow()
            end
            println(stderr, e.message)
            return 1
        end
        write(path, text)
        println("Refreshed code_health/", name, ".")
        return 0
    end
    failures, provenance_ok = verify(measurement, recorded)
    if isempty(failures)
        publish(measurement)
        return 0
    end
    for f in failures
        println(f)
    end
    if provenance_ok
        try
            write_artifact(name, render(measurement, recorded, true))
        catch e
            if !(e isa RefreshRefused)
                rethrow()
            end
            println("No Refresh Artifact: ", e.message)
        end
    else
        println("No Refresh Artifact: the provenance does not match.")
    end
    return 1
end

end # module
