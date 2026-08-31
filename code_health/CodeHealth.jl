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
const DECLARING_FILES = ("src/01_Base.jl", "src/02_Tools.jl",
                         "src/08_Moments/01_Base_Moments.jl",
                         "src/20_Optimisation/01_Base_Optimisation.jl")

in_scope(path::AbstractString) = any(r -> startswith(path, r), MEASURED_ROOTS)

is_ci() = haskey(ENV, "GITHUB_ACTIONS")

# --- Declaration Macros ----------------------------------------------------
#
# Found by parsing alone, so `complexity.jl` stays a pure parser and loads no package.

macro_name(x::Symbol) = String(x)
macro_name(x::Expr) = x.head === :. ? macro_name(x.args[end]) : ""
macro_name(x::QuoteNode) = macro_name(x.value)
macro_name(x) = ""

parse_file(file) = Meta.parseall(read(joinpath(REPO_ROOT, file), String); filename = file)

"""
    declared_macros(file) -> Vector{String}

The macros a file declares.
"""
function declared_macros(file)
    names = String[]
    walk(x) = nothing
    function walk(e::Expr)
        if e.head === :macro
            sig = e.args[1]
            n = sig isa Expr ? sig.args[1] : sig
            n isa Symbol && push!(names, "@" * String(n))
        end
        foreach(walk, e.args)
        return nothing
    end
    walk(parse_file(file))
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
    walk(x) = nothing
    function walk(e::Expr)
        if e.head === :struct
            return nothing
        end
        if e.head === :macrocall
            n = macro_name(e.args[1])
            isempty(n) || push!(found, (n, e))
        end
        foreach(walk, e.args)
        return nothing
    end
    walk(parse_file(file))
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

**`test/test_45_sweep_census.jl` is the Authority for this definition**, and it holds a second copy
of these lines. That test loads no file of `code_health/`, deliberately, so `code_health` may not
become a dependency of the test suite. The two copies must move together, and the comment beside
the test's copy says so.

A field docstring is NOT a unit: inside a struct body a docstring parses as a bare string literal
rather than as a macrocall, so it never reaches this count.
"""
function documented_units(path::AbstractString)
    doc_macro = GlobalRef(Core, Symbol("@doc"))
    isdocstring(x) = Meta.isexpr(x, :macrocall) &&
                     !(isempty(x.args)) &&
                     (x.args[1] === doc_macro || x.args[1] === Symbol("@doc"))
    n = 0
    function walk(node)
        if !(node isa Expr)
            return nothing
        end
        if isdocstring(node)
            (n += 1)
        end
        foreach(walk, node.args)
        return nothing
    end
    walk(Meta.parseall(read(path, String)))
    return n
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
