#!/usr/bin/env julia
#
# The line-coverage half of the code-health gate.
#
#     julia --project=code_health code_health/coverage.jl check
#     julia --project=code_health code_health/coverage.jl refresh
#     julia --project=code_health code_health/coverage.jl refresh --accept-rise
#     julia --project=code_health code_health/coverage.jl terminal [path ...]
#
# It reads `lcov.info`, which `julia-actions/julia-processcoverage` writes in the test job, and it
# writes `code_health/coverage_baseline.toml`. It loads no package under measurement and parses with
# `Meta.parseall`, so it costs about five seconds.
#
# The decisions this file implements are ADR 0082, over ADRs 0073 to 0077.

if !(isdefined(Main, :CodeHealth))
    Main.include(joinpath(@__DIR__, "CodeHealth.jl"))
end

using Main.CodeHealth
# The name a definition binds is `CodeHealth`'s, so this file and `test/test_26_docs.jl` write one
# answer. `test/test_49_coverage_attribution_census.jl` reads these four names through this file.
using Main.CodeHealth: definition_name, defname, is_signature, unwrap
using TOML

const NAME = "coverage_baseline.toml"

"""
The metric the ratchet compares. A file's **miss count** binds and its relevant-line count is
context, on the same split `complexity.jl` draws between a maximum and a sum.

The miss count is the number the map drives to zero, and it moves in the right direction on its own.
Adding a covered function raises `lines` and leaves `misses` where it was, so ordinary work is
quiet. Adding an uncovered one raises `misses`, which is the case the gate exists for.

The percentage is deliberately not the metric. It rises when an uncovered line is deleted and it
rises when a covered one is added, so a file can improve its percentage while gaining misses.
"""
const BINDING = ("misses",)

"""
Every number a row carries. A rename pairs on both, per ADR 0074.
"""
const ROW_NUMBERS = ("lines", "misses")

"""
The name a miss line carries when it lies inside no named top-level definition. A `const`, an
`include` and a `precompile` block all land here, and it is a legitimate Coverage Exemption target.
"""
const TOPLEVEL = "<toplevel>"

# --- reading lcov.info -----------------------------------------------------

"""
    lcov_path(; root = CodeHealth.REPO_ROOT) -> String

Where the gate looks for `lcov.info`. `COVERAGE_LCOV` overrides it, which is how the CI job hands
over the file it downloaded from the test job's artifact.
"""
function lcov_path(; root = CodeHealth.REPO_ROOT)
    return get(ENV, "COVERAGE_LCOV", joinpath(root, "lcov.info"))
end

"""
    relative(path; root = CodeHealth.REPO_ROOT) -> String

An `SF:` record's path as a path relative to `root`. `Coverage.jl` writes the path it was given, so
it may be absolute or relative depending on where `julia-processcoverage` ran.
"""
function relative(path::AbstractString; root = CodeHealth.REPO_ROOT)
    p = replace(String(path), '\\' => '/')
    base = replace(String(root), '\\' => '/')
    base = endswith(base, "/") ? base : base * "/"
    if startswith(p, base)
        p = p[(length(base) + 1):end]
    end
    return startswith(p, "./") ? p[3:end] : p
end

"""
    parse_lcov(path; root = CodeHealth.REPO_ROOT) -> Dict{String, Dict{Int, Int}}

The `DA:<line>,<count>` records of an LCOV file, per source file. `Coverage.jl` writes `SF`, `DA`,
`LH`, `LF` and `end_of_record` and no function records at all, so a line is the only unit there is.
A line with no `DA` record is not executable and is never counted.
"""
function parse_lcov(path::AbstractString; root = CodeHealth.REPO_ROOT)
    if !(isfile(path))
        error("No coverage data at $path.\n" *
              "The gate reads the lcov.info that julia-processcoverage writes. Set COVERAGE_LCOV " *
              "to point at it, or run the suite with --code-coverage=user first.")
    end
    out = Dict{String, Dict{Int, Int}}()
    current = ""
    for line in eachline(path)
        if startswith(line, "SF:")
            current = relative(strip(line[4:end]); root)
            get!(out, current, Dict{Int, Int}())
        elseif startswith(line, "DA:") && !(isempty(current))
            body = strip(line[4:end])
            comma = findfirst(==(','), body)
            if comma === nothing
                continue
            end
            ln = tryparse(Int, body[1:(comma - 1)])
            n = tryparse(Int, body[(comma + 1):end])
            if (ln === nothing || n === nothing)
                continue
            end
            d = out[current]
            # A file measured by more than one upload session appears twice. The hits add up, so a
            # line covered by either session is covered.
            d[ln] = get(d, ln, 0) + n
        elseif startswith(line, "end_of_record")
            current = ""
        end
    end
    return out
end

# --- attributing a line to a definition ------------------------------------

function line_numbers!(acc::Vector{Int}, e)
    if e isa LineNumberNode
        push!(acc, e.line)
    elseif e isa Expr
        for a in e.args
            line_numbers!(acc, a)
        end
    end
    return acc
end

"""
    definition_ranges(file; root = CodeHealth.REPO_ROOT) -> Vector{Tuple{String, Int, Int}}

One entry per named top-level definition: its name and the first and last source line it holds. The
range is taken from the `LineNumberNode`s the parser leaves in the expression, and every executable
line carries one, so a miss line always falls inside the range of the definition that holds it.

The walk is deliberately top-level only. A closure inside a function is attributed to that function,
because a Coverage Exemption is written and read by a human and a human names the method.
"""
function definition_ranges(file::AbstractString; root = CodeHealth.REPO_ROOT)
    top = CodeHealth.parse_file(file; root)
    out = Tuple{String, Int, Int}[]
    if !(top isa Expr)
        return out
    end
    for a in top.args
        if !(a isa Expr)
            continue
        end
        name = definition_name(a)
        if isempty(name)
            continue
        end
        ls = line_numbers!(Int[], a)
        if isempty(ls)
            continue
        end
        push!(out, (name, minimum(ls), maximum(ls)))
    end
    return out
end

"""
    attribute(ranges, misses) -> Dict{String, Int}

The miss count per definition. A line inside no named definition is attributed to `<toplevel>`, and
a line inside more than one takes the narrowest range.
"""
function attribute(ranges, misses)
    out = Dict{String, Int}()
    for ln in misses
        best, width = TOPLEVEL, typemax(Int)
        for (name, lo, hi) in ranges
            if lo <= ln <= hi && (hi - lo) < width
                best, width = name, hi - lo
            end
        end
        out[best] = get(out, best, 0) + 1
    end
    return out
end

# --- measurement -----------------------------------------------------------

struct FileCoverage
    lines::Int
    misses::Vector{Int}
    by_definition::Dict{String, Int}
end

"""
    measure(; root, files, lcov) -> NamedTuple

Count the uncovered lines of `files` from the LCOV file at `lcov`, and attribute each one to the
definition that holds it. Every path is read relative to `root`. The defaults are the live checkout,
every file in scope and the `lcov.info` the test job wrote, so the entry script calls `measure()`
unchanged; a test passes a fixture tree and a fixture LCOV file instead. `CodeHealth.REPO_ROOT`
states why the seam is written this way.
"""
function measure(; root = CodeHealth.REPO_ROOT, files = CodeHealth.source_files(; root),
                 lcov = lcov_path(; root))
    hits_by_file = parse_lcov(lcov; root)
    numbers = Dict{String, FileCoverage}()
    for f in files
        hits = get(hits_by_file, f, Dict{Int, Int}())
        misses = sort!([ln for (ln, n) in hits if n == 0])
        by_def = if isempty(misses)
            Dict{String, Int}()
        else
            attribute(definition_ranges(f; root), misses)
        end
        numbers[f] = FileCoverage(length(hits), misses, by_def)
    end
    # An lcov record naming a file outside the scope is not a failure. `julia-processcoverage`
    # walks the package directory, and a stray record costs the gate nothing.
    foreign = sort!(collect(setdiff(Set(keys(hits_by_file)), Set(files))))
    return (; files, numbers, foreign, provenance = provenance(; root))
end

"""
    provenance() -> Vector{Pair}

**Nothing here binds.** The other three baselines pin their analyser and fail on a mismatch, because
a parser that moves invalidates the numbers it took. Coverage is produced by Julia itself in the
test job, and ADR 0056 floats that job on the newest release, so a pin here would turn the gate red
on a Julia patch release for no defect. A move in Julia's line attribution shows up as an ordinary
rise instead, and the Refresh Artifact clears it on the same route as any other rise. ADR 0082.
"""
function provenance(; root = CodeHealth.REPO_ROOT)
    return ["julia" => string(VERSION), "commit" => CodeHealth.git_short_commit(; root)]
end

row(c::FileCoverage) = ["lines" => c.lines, "misses" => length(c.misses)]

rows(m) = Dict(f => Dict(row(m.numbers[f])) for f in m.files)

recorded_rows(recorded) = get(recorded, "file", Dict{String, Any}())

# --- Coverage Exemptions ---------------------------------------------------

"""
    exemptions(rulings) -> Dict{String, Dict{String, Int}}

The `[[coverage_exemption]]` rows of `code_health/rulings.toml`, keyed by path and then by
definition. A row states how many uncovered lines in that definition may stand, so the count is part
of the claim and not a free pass on the whole definition.
"""
function exemptions(rulings)
    out = Dict{String, Dict{String, Int}}()
    for e in get(rulings, "coverage_exemption", [])
        d = get!(out, e["path"], Dict{String, Int}())
        d[e["definition"]] = get(d, e["definition"], 0) + e["misses"]
    end
    return out
end

"""
    exemption_failures(m, rulings) -> Vector{String}

A Coverage Exemption states an exact number, and the gate holds it to it in **both** directions.

A claim that stands above the truth is stale: the lines were covered and the row outlived its
reason. A claim that stands below it is the leak the file total cannot see, because a line covered
elsewhere in the file pays for a new uncovered line inside the exempted definition and the ratchet
reads a flat number. Equality closes both.
"""
function exemption_failures(m, rulings)
    bad = String[]
    in_tree = Set(m.files)
    for (path, defs) in exemptions(rulings)
        if !(path in in_tree)
            push!(bad,
                  "ERROR: a Coverage Exemption names $path, which is not a file in scope.")
            continue
        end
        actual = m.numbers[path].by_definition
        for (name, claimed) in defs
            have = get(actual, name, 0)
            if have == claimed
                continue
            elseif have == 0
                push!(bad,
                      "ERROR: $path: the Coverage Exemption for $name claims $claimed " *
                      "uncovered line(s), and none are uncovered. Remove the row.")
            else
                push!(bad,
                      "ERROR: $path: the Coverage Exemption for $name claims $claimed " *
                      "uncovered line(s), and $have are uncovered.")
            end
        end
    end
    return sort!(bad)
end

"""
    residue(m, rulings, file) -> Int

The uncovered lines of a file that no Coverage Exemption accounts for. A file is **terminal** when
its residue is zero, which is #404's closing condition for a child map's coverage.
"""
function residue(m, rulings, file)
    exempt = get(exemptions(rulings), file, Dict{String, Int}())
    total = 0
    for (name, n) in m.numbers[file].by_definition
        total += max(0, n - get(exempt, name, 0))
    end
    return total
end

is_terminal(m, rulings, file) = residue(m, rulings, file) == 0

# --- candidacy -------------------------------------------------------------

"""
    candidacy_failures(m, rulings, file) -> Vector{String}

ADR 0074's entry test, in the form coverage takes. **A file added to the tree enters terminal**:
every one of its lines is covered, or carries a Coverage Exemption.

A new file has no legacy to plead. This is the rule that would have stopped the two files standing
at exactly zero, which is the case #404 names: a repository-wide number hides them, and a per-file
ratchet that lets a file enter at whatever it measures records the zero rather than refusing it.
"""
function candidacy_failures(m, rulings, file)
    r = residue(m, rulings, file)
    if r == 0
        return String[]
    end
    return ["ERROR: $file enters with $r uncovered line(s) that no Coverage Exemption " *
            "accounts for.\n" *
            "       Cover them, or add a [[coverage_exemption]] row to code_health/rulings.toml."]
end

# --- render ----------------------------------------------------------------

function render(m, recorded, accept_rise::Bool; rulings = CodeHealth.read_rulings())
    measured = rows(m)
    rec = recorded_rows(recorded)
    if !(isempty(rec))
        missing_rows, dead_rows = CodeHealth.set_differences(m.files, collect(keys(rec)))
        equal(a, b) = all(k -> a[k] == b[k], ROW_NUMBERS)
        pairs, _, added = CodeHealth.pair_renames(dead_rows, missing_rows, rec, measured,
                                                  equal)
        for (dead, new) in pairs
            println("Paired the row of ", dead, " with ", new, ", which measures the same.")
        end
        refusals = String[]
        for f in added
            append!(refusals, candidacy_failures(m, rulings, f))
        end
        isempty(refusals) || throw(CodeHealth.RefreshRefused(join(refusals, "\n")))
    end
    rs = CodeHealth.rises(rec, measured, BINDING)
    if !(isempty(rs)) && !(accept_rise)
        CodeHealth.refuse_rise(NAME, rs)
    end
    io = IOBuffer()
    println(io, "# Generated by code_health/coverage.jl. Do not edit by hand.")
    println(io, "# One row per file in scope: `misses` binds and `lines` is context.")
    println(io,
            "# A file with no lcov record has no executable line and records zero of both.")
    println(io)
    CodeHealth.emit_provenance(io, m.provenance)
    println(io)
    CodeHealth.emit_section(io, "file",
                            (f => [k => measured[f][k] for k in ROW_NUMBERS]
                             for f in m.files))
    return String(take!(io))
end

# --- verify ----------------------------------------------------------------

function verify(m, recorded; rulings = CodeHealth.read_rulings())
    failures = String[]
    for line in CodeHealth.check_rationale_citations(rulings)
        push!(failures, "ERROR: " * line)
    end
    append!(failures, exemption_failures(m, rulings))
    rec = recorded_rows(recorded)
    missing_rows, dead_rows = CodeHealth.set_differences(m.files, collect(keys(rec)))
    for f in missing_rows
        CodeHealth.annotate(f,
                            "no row in $NAME. The baseline must name every file in scope.")
    end
    for f in dead_rows
        println("  $NAME names $f, which no longer exists.")
    end
    if !(isempty(missing_rows) && isempty(dead_rows))
        push!(failures,
              "The baseline's file set and the tree's file set differ: " *
              "$(length(missing_rows)) file(s) with no row, $(length(dead_rows)) row(s) naming no file.")
    end
    rs = CodeHealth.rises(rec, rows(m), BINDING)
    for r in rs
        CodeHealth.annotate(r.key, "$(r.metric) rose from $(r.old) to $(r.new).")
    end
    if !(isempty(rs))
        push!(failures, "The coverage ratchet tripped on $(length(rs)) file(s).")
        CodeHealth.step_summary("### Coverage ratchet\n\n" * CodeHealth.rise_table(rs))
    end
    if !(isempty(failures))
        push!(failures, CodeHealth.routes(; dismissal = false))
    end
    # Provenance never binds here, so the second value says so. A failing run always publishes the
    # Refresh Artifact.
    return failures, true
end

function publish(m)
    rulings = CodeHealth.read_rulings()
    lines = sum(c -> c.lines, values(m.numbers); init = 0)
    misses = sum(c -> length(c.misses), values(m.numbers); init = 0)
    done = count(f -> is_terminal(m, rulings, f), m.files)
    pct = lines == 0 ? 100.0 : round(100 * (lines - misses) / lines; digits = 2)
    text = """
    ### Coverage gate

    | field | value |
    | --- | --- |
    | julia | $(VERSION) |
    | files measured | $(length(m.files)) |
    | relevant lines | $(lines) |
    | misses | $(misses) |
    | coverage | $(pct) % |
    | files terminal | $(done) of $(length(m.files)) |
    """
    println("Green. ", length(m.files), " files measured, ", misses, " miss(es), ", pct,
            " %, ", done, " file(s) terminal.")
    CodeHealth.step_summary(text)
    return nothing
end

# --- the terminal condition ------------------------------------------------

"""
    report_terminal(paths) -> Int

#404's closing condition, per file. A child map closes on coverage when every file it owns is
terminal. With no argument the whole tree is reported; with arguments, only the paths given, so a
child map asks about its own files alone.
"""
function report_terminal(paths)
    m = measure()
    rulings = CodeHealth.read_rulings()
    wanted = if isempty(paths)
        m.files
    else
        filter(f -> any(p -> f == p || startswith(f, rstrip(p, '/') * "/"), paths), m.files)
    end
    if isempty(wanted)
        println("No file in scope matches ", join(paths, ", "), ".")
        return 1
    end
    open_files = 0
    for f in sort(wanted)
        r = residue(m, rulings, f)
        if r == 0
            continue
        end
        open_files += 1
        println(f, ": ", r, " uncovered line(s) with no Coverage Exemption.")
        exempt = get(exemptions(rulings), f, Dict{String, Int}())
        for (name, n) in sort(collect(m.numbers[f].by_definition); by = first)
            left = n - get(exempt, name, 0)
            left > 0 && println("    ", name, ": ", left)
        end
    end
    println(length(wanted) - open_files, " of ", length(wanted), " file(s) terminal.")
    return open_files == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !(isempty(ARGS)) && ARGS[1] == "terminal"
        exit(report_terminal(ARGS[2:end]))
    end
    exit(CodeHealth.run_script(ARGS; name = NAME, measure = measure, verify = verify,
                               render = render, publish = publish))
end
