#!/usr/bin/env julia
#
# The CodeComplexity half of the code-health gate.
#
#     julia --project=code_health code_health/complexity.jl check
#     julia --project=code_health code_health/complexity.jl refresh
#     julia --project=code_health code_health/complexity.jl refresh --accept-rise
#
# It writes `code_health/complexity_baseline.toml`. CodeComplexity is a pure JuliaSyntax parser, so
# this script loads no package under measurement and costs about five seconds.

# `code_health/triage.jl` includes this file into a module of its own, so the include happens once
# and always into `Main`. CodeHealth must be ONE module rather than three: a `Definition` built by a
# second copy would not be the same type as one built by the first.
if !(isdefined(Main, :CodeHealth))
    Main.include(joinpath(@__DIR__, "CodeHealth.jl"))
end

using Main.CodeHealth
using CodeComplexity, JuliaSyntax, Pkg, TOML

const NAME = "complexity_baseline.toml"

"""
The three metrics, in the order a baseline row prints them. The key is the row's field name and the
value is the metric CodeComplexity measures it with.
"""
const METRICS = ("cyc" => CyclomaticComplexity(), "cog" => CognitiveComplexity(),
                 "arg" => ArgumentCountComplexity())

"""
The metrics the ratchet compares. The **maximum** over the definitions in a file binds, and the
three sums beside it are context that never binds.

Issue #336 measured why. The maximum is quiet under ordinary work: a new helper of complexity 2 in
a file whose maximum is 9 moves nothing. The sum moves by 2 and would turn the gate red, and that
failure is noise. A noisy gate gets switched off. The sum is still recorded, because the maximum is
blind to sprawl and the scheduled job ranks with it.
"""
const BINDING = ("cyc", "cog", "arg")

"""
Every number a row carries. A rename pairs on all of them, which is stricter than pairing on the
binding three alone and cannot pair two files that merely share a worst definition.
"""
const ROW_NUMBERS = ("cyc", "cog", "arg", "cyc_sum", "cog_sum", "arg_sum")

# --- parseability ----------------------------------------------------------

"""
    parse_failures(files) -> Vector{String}

CodeComplexity parses with `ignore_errors = true`, so a file with a syntax error is measured as if
it were correct and reports a low number (issue #336). The gate therefore checks parseability
itself, with errors on.
"""
function parse_failures(files)
    bad = String[]
    for f in files
        try
            JuliaSyntax.parseall(Expr, read(joinpath(CodeHealth.REPO_ROOT, f), String);
                                 filename = f, ignore_errors = false)
        catch e
            push!(bad, "$f: $(sprint(showerror, e))")
        end
    end
    return bad
end

# --- measurement -----------------------------------------------------------

struct FileNumbers
    max::Dict{String, Int}
    sum::Dict{String, Int}
    definitions::Dict{String, Vector{CodeHealth.Definition}}
    macros::Vector{String}
end

function measure()
    files = filter(CodeHealth.in_scope, CodeHealth.tracked_jl_files())
    bad = parse_failures(files)
    if !(isempty(bad))
        error("A file in scope does not parse:\n" * join(bad, "\n"))
    end
    decl = CodeHealth.declaration_macros(files)
    numbers = Dict{String, FileNumbers}()
    for f in files
        mx, sm = Dict{String, Int}(), Dict{String, Int}()
        defs = Dict{String, Vector{CodeHealth.Definition}}()
        for (key, metric) in METRICS
            fns = measure_file(metric, joinpath(CodeHealth.REPO_ROOT, f)).functions
            mx[key] = isempty(fns) ? 0 : maximum(fn -> fn.value, fns)
            sm[key] = sum(fn -> fn.value, fns; init = 0)
            defs[key] = [CodeHealth.Definition(String(fn.name), fn.value, fn.line)
                         for fn in fns]
        end
        marker = sort!(collect(intersect(CodeHealth.called_macros(f), decl)))
        numbers[f] = FileNumbers(mx, sm, defs, marker)
    end
    return (; files, numbers, provenance = provenance())
end

function provenance()
    deps = Pkg.dependencies()
    version(name) = string(only(v.version for v in values(deps) if v.name == name))
    return ["julia" => string(VERSION), "code_complexity" => version("CodeComplexity"),
            "julia_syntax" => version("JuliaSyntax"),
            "commit" => CodeHealth.git_short_commit()]
end

function row(n::FileNumbers)
    return ["cyc" => n.max["cyc"], "cog" => n.max["cog"], "arg" => n.max["arg"],
            "cyc_sum" => n.sum["cyc"], "cog_sum" => n.sum["cog"], "arg_sum" => n.sum["arg"],
            "macros" => n.macros]
end

rows(m) = Dict(f => Dict(row(m.numbers[f])) for f in m.files)

recorded_rows(recorded) = get(recorded, "file", Dict{String, Any}())

# --- candidacy -------------------------------------------------------------

"""
    candidacy_failures(file, numbers, rulings) -> Vector{String}

ADR 0074's entry test. An added file with no row enters when the scheduled job would not file it as
an offender, with Exemptions dropped first. It invents no flag and no threshold: the two ways to
satisfy it already exist, which are to lower the number or to declare an Exemption.
"""
function candidacy_failures(file, numbers::FileNumbers, rulings)
    th = CodeHealth.thresholds(rulings)
    limits = ("cyc" => th["cyclomatic"], "cog" => th["cognitive"], "arg" => th["argcount"])
    bad = String[]
    for (key, limit) in limits
        exempt = CodeHealth.exempt_definitions(rulings, key)
        vals = [d.value for d in numbers.definitions[key] if !((file, d.name) in exempt)]
        value = isempty(vals) ? 0 : maximum(vals)
        if value > limit
            push!(bad,
                  "ERROR: $file enters at $key = $value, over the threshold of $limit.\n" *
                  "       Lower it, or add an Exemption naming (path, definition, $key).")
        end
    end
    return bad
end

# --- render ----------------------------------------------------------------

function render(m, recorded, accept_rise::Bool)
    rulings = CodeHealth.read_rulings()
    measured = rows(m)
    rec = recorded_rows(recorded)
    out = Dict{String, Any}()
    if !isempty(rec)
        missing_rows, dead_rows = CodeHealth.set_differences(m.files, collect(keys(rec)))
        equal(a, b) = all(k -> a[k] == b[k], ROW_NUMBERS) && a["macros"] == b["macros"]
        pairs, _, added = CodeHealth.pair_renames(dead_rows, missing_rows, rec, measured,
                                                  equal)
        for (dead, new) in pairs
            println("Paired the row of ", dead, " with ", new, ", which measures the same.")
        end
        refusals = String[]
        for f in added
            append!(refusals, candidacy_failures(f, m.numbers[f], rulings))
        end
        isempty(refusals) || throw(CodeHealth.RefreshRefused(join(refusals, "\n")))
    end
    rs = CodeHealth.rises(rec, measured, BINDING)
    if !isempty(rs) && !accept_rise
        CodeHealth.refuse_rise(NAME, rs)
    end
    # A refresh that gets this far records the truth: it either lowered a number or the rise was
    # asked for by name.
    for f in m.files
        out[f] = measured[f]
    end
    io = IOBuffer()
    println(io, "# Generated by code_health/complexity.jl. Do not edit by hand.")
    println(io,
            "# The maximum over the definitions in the file binds. The sums are context.")
    println(io)
    CodeHealth.emit_provenance(io, m.provenance)
    println(io)
    CodeHealth.emit_section(io, "file",
                            (f => [k => out[f][k] for (k, _) in row(m.numbers[f])]
                             for f in m.files))
    return String(take!(io))
end

# --- verify ----------------------------------------------------------------

function verify(m, recorded)
    rulings = CodeHealth.read_rulings()
    failures = String[]
    bad_prov = CodeHealth.provenance_failures(get(recorded, "provenance", Dict()),
                                              Dict(m.provenance))
    if !isempty(bad_prov)
        push!(failures, CodeHealth.provenance_message(bad_prov))
        return failures, false
    end
    for line in CodeHealth.check_rationale_citations(rulings)
        push!(failures, "ERROR: " * line)
    end
    uncovered = CodeHealth.coverage_failures(rulings)
    if !isempty(uncovered)
        push!(failures,
              "A tracked .jl file is neither measured nor a named Unmeasured Path.\n" *
              join(("  " * f for f in uncovered), "\n") *
              "\nAdd an [[unmeasured_path]] entry to code_health/rulings.toml, with a reason.")
    end
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
    if !isempty(rs)
        push!(failures, "The complexity ratchet tripped on $(length(rs)) number(s).")
        CodeHealth.step_summary("### Complexity ratchet\n\n" * CodeHealth.rise_table(rs))
    end
    if !(isempty(failures))
        push!(failures, CodeHealth.routes(; dismissal = false))
    end
    return failures, true
end

function publish(m)
    total(k) = sum(n -> n.max[k], values(m.numbers); init = 0)
    text = """
    ### Complexity gate

    | field | value |
    | --- | --- |
    | julia | $(VERSION) |
    | files measured | $(length(m.files)) |
    | cyclomatic, worst file | $(maximum(n -> n.max["cyc"], values(m.numbers))) |
    | cognitive, worst file | $(maximum(n -> n.max["cog"], values(m.numbers))) |
    | argument count, worst file | $(maximum(n -> n.max["arg"], values(m.numbers))) |
    """
    println("Green. ", length(m.files), " files measured, cyclomatic total ", total("cyc"),
            ".")
    CodeHealth.step_summary(text)
    return nothing
end

# The scheduled job of ADR 0078 reuses this file's `measure` rather than carrying a second copy
# of it, so the command line runs only when this file is the program. `code_health/triage.jl`
# includes it into a module of its own and calls `measure` directly.
if abspath(PROGRAM_FILE) == @__FILE__
    exit(CodeHealth.run_script(ARGS; name = NAME, measure = measure, verify = verify,
                               render = render, publish = publish))
end
