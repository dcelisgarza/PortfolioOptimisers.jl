#!/usr/bin/env julia
#
# The JET half of the code-health gate.
#
#     julia --project=code_health code_health/jet.jl check
#     julia --project=code_health code_health/jet.jl refresh
#     julia --project=code_health code_health/jet.jl refresh --accept-rise
#
# It writes `code_health/jet_baseline.toml`. `report_package` analyses a package and cannot be
# pointed at one file, so this script is the whole inner loop and it costs about six minutes at
# 2.4 GiB of peak resident memory.
#
# The baseline is keyed by run and by file. Each run holds a row for every file in scope, zeros
# included, and no number is ever summed, so the four Plots-extension reports that attribute back
# into `src/` cannot double count. ADR 0073.

# `code_health/triage.jl` includes this file into a module of its own, so the include happens once
# and always into `Main`. CodeHealth must be ONE module rather than three: a `Definition` built by a
# second copy would not be the same type as one built by the first.
if !(isdefined(Main, :CodeHealth))
    Main.include(joinpath(@__DIR__, "CodeHealth.jl"))
end

using Main.CodeHealth
using Pkg, TOML

# The load set, loaded before anything measures. `report_package`'s number moves with it: 315
# reports with the package alone and 312 once the three extension triggers load (issue #337), so
# the baseline pins the set and the script asserts it.
const LOAD_SET = ["GraphRecipes", "Impute", "StatsPlots"]

using GraphRecipes, Impute, StatsPlots
using PortfolioOptimisers
using JET

const NAME = "jet_baseline.toml"

"""
The ratchet binds on the **reviewed** number alone. The raw count is context, because a Dismissal
covers a class and a fifteenth instance of a dismissed class must stay green (ADR 0071).
"""
const BINDING = ("reviewed",)

"""
The attribution rule, recorded in the provenance block so a baseline is never read against a run
that used a different one.
"""
const ATTRIBUTION = "deepest_repository_frame"
const FILTER = "AnyFrameModule"

# --- the environment -------------------------------------------------------

"""
    assert_environment()

JET 0.12.1 needs Julia 1.12 and degrades silently: on an unsupported version it loads
`JETEmpty.jl`, whose stubs warn on load and throw on call. A gate that stops measuring without
failing is worse than no gate, so `JET_AVAILABLE` is asserted before anything is measured.
"""
function assert_environment()
    if !(JET.JET_AVAILABLE)
        error("JET.JET_AVAILABLE is false. JET loaded its empty stubs, so nothing would be\n" *
              "measured and the gate would report zero. Julia is $(VERSION); JET 0.12.1 needs 1.12.")
    end
    return nothing
end

"""
    assert_load_set(set)

Each run's recorded load set must be loaded. A trigger package that silently fails to load makes
the run measure a different thing, and 312 becomes 315 with no source change.
"""
function assert_load_set(set)
    loaded = Set(String(k.name) for k in keys(Base.loaded_modules))
    absent = filter(n -> !(n in loaded), set)
    if !(isempty(absent))
        error("The recorded load set is not loaded.\n  expected: $(join(set, ", "))\n" *
              "  absent:   $(join(absent, ", "))")
    end
    return nothing
end

# --- attribution and fingerprint -------------------------------------------

"""
    attribute(report) -> Union{String, Nothing}

The deepest frame of the report's virtual stack trace that names a file under `src/` or `ext/`.

`report.vst` runs from the outermost frame to the innermost, so the deepest repository frame is the
last one in scope. Keying on the **last** frame instead discards 238 of the 315 reports, 164 of
them ending in `Base_compiler.jl`, which throws away a defect in package code whenever execution
ends inside a dependency. Keying on the **shallowest** frame piles hundreds of reports onto the few
entry points, so the per-file key stops localising anything. ADR 0073.
"""
function attribute_frame(report)
    hit, line = nothing, 0
    for frame in report.vst
        path = String(frame.file)
        rel = if startswith(path, CodeHealth.REPO_ROOT)
            lstrip(path[(length(CodeHealth.REPO_ROOT) + 1):end], '/')
        else
            path
        end
        CodeHealth.in_scope(rel) && ((hit, line) = (rel, Int(frame.line)))
    end
    return hit, line
end

attribute(report) = attribute_frame(report)[1]

report_kind(report) = String(nameof(typeof(report)))

"""
    report_message(report) -> String

`JETInterface.print_report_message`, and for a `BuiltinErrorReport` the builtin as well.
`print_signature` is a separate JET interface function, and every fragile piece of text — the
gensym `#23`, a local variable name — lives there, so the fingerprint cuts it rather than masking
it. Because the text is cut, no normalisation rule exists to write, test or maintain.

A `BuiltinErrorReport` prints `r.msg` alone, and that message is often the bare constant
`invalid builtin function call`: 175 of this package's 200 reports of the kind carry it, and five
builtins sit behind it. One Dismissal would cover all five. `r.f` is the builtin itself, and it
carries no gensym, no local name and no type render, so it narrows the class without re-admitting
the fragile text. It is appended unconditionally, because a rule that fires only on JET's constant
would name a JET internal string in this script.

The kind is matched by its name rather than by its type. A type in a method signature resolves at
load time, and JET's stubs define no such type on an unsupported Julia, so a signature would turn
`assert_environment`'s clean message into an `UndefVarError`. ADR 0071, amended by issue #357.
"""
function report_message(report)
    msg = sprint(io -> JET.JETInterface.print_report_message(io, report))
    if report_kind(report) == "BuiltinErrorReport"
        return msg * ": " * string(report.f)
    end
    return msg
end

fingerprint(report) = (attribute(report), report_kind(report), report_message(report))

# --- dismissals ------------------------------------------------------------

"""
    dismissal_set(rulings) -> Set

A Report Fingerprint carries no run, so one Dismissal matches in every run that produces the class.
The count written beside a Dismissal is for a human reader and does not bind.
"""
function dismissal_set(rulings)
    return Set((e["file"], e["kind"], e["message"]) for e in get(rulings, "dismissal", []))
end

# --- measurement -----------------------------------------------------------

const RUNS = ("main" => :package, "plots_ext" => :PortfolioOptimisersPlotsExt,
              "impute_ext" => :PortfolioOptimisersImputeExt)

function target_module(name::Symbol)
    m = Base.get_extension(PortfolioOptimisers, name)
    if m === nothing
        error("The extension $name is not loaded, so its run cannot measure.")
    end
    return m
end

function measure()
    assert_environment()
    assert_load_set(LOAD_SET)
    rulings = CodeHealth.read_rulings()
    dismissed = dismissal_set(rulings)
    files = filter(CodeHealth.in_scope, CodeHealth.tracked_jl_files())
    runs = Pair{String, Any}[]
    reviewed = Dict(f => CodeHealth.Reviewed[] for f in files)
    unattributed = String[]
    for (run, which) in RUNS
        target = which === :package ? PortfolioOptimisers : target_module(which)
        result = JET.report_package(target; target_modules = (JET.AnyFrameModule(target),))
        counts = Dict(f => Dict("raw" => 0, "reviewed" => 0) for f in files)
        for report in JET.get_reports(result)
            file, line = attribute_frame(report)
            kind, message = report_kind(report), report_message(report)
            if file === nothing
                push!(unattributed, "$run: $kind: $message")
                continue
            end
            row = counts[file]
            row["raw"] += 1
            if !((file, kind, message) in dismissed)
                row["reviewed"] += 1
                push!(reviewed[file], CodeHealth.Reviewed(run, kind, message, line))
            end
        end
        push!(runs, run => counts)
    end
    if !(isempty(unattributed))
        error("$(length(unattributed)) report(s) name no file under src/ or ext/, so they have\n" *
              "nowhere to be recorded and the gate would lose them silently:\n" *
              join(first(unattributed, 10), "\n"))
    end
    return (; files, runs, reviewed, provenance = provenance())
end

function provenance()
    deps = Pkg.dependencies()
    version(name) = string(only(v.version for v in values(deps) if v.name == name))
    return ["julia" => string(VERSION), "jet" => version("JET"), "filter" => FILTER,
            "attribution" => ATTRIBUTION, "load_set" => LOAD_SET,
            "commit" => CodeHealth.git_short_commit()]
end

row(counts, f) = ["raw" => counts[f]["raw"], "reviewed" => counts[f]["reviewed"]]

recorded_runs(recorded) = get(recorded, "run", Dict{String, Any}())

function recorded_rows(recorded, run)
    return get(get(recorded_runs(recorded), run, Dict()), "file", Dict{String, Any}())
end

# --- render ----------------------------------------------------------------

"""
    dismissal_repairs(rulings, files)

ADR 0074 prints the Dismissal lines whose file field is now dead, and never edits them.
`rulings.toml` is the hand-written file of ADR 0073's authorship split, and a generated edit landing
in the file that holds human paragraphs is the coupling that split exists to prevent.
"""
function dismissal_repairs(rulings, files)
    live = Set(files)
    dead = [e for e in get(rulings, "dismissal", []) if !(e["file"] in live)]
    if isempty(dead)
        return nothing
    end
    println("NOTE: ", length(dead), " Dismissals still name a dead path.")
    for e in dead
        println("  ", e["file"], "  ", e["kind"], ": ", e["message"])
    end
    println("Edit code_health/rulings.toml, then re-run.")
    return nothing
end

function render(m, recorded, accept_rise::Bool)
    rulings = CodeHealth.read_rulings()
    dismissal_repairs(rulings, m.files)
    out = Pair{String, Any}[]
    for (run, counts) in m.runs
        rec = recorded_rows(recorded, run)
        measured = Dict(f => Dict(row(counts, f)) for f in m.files)
        if !isempty(rec)
            missing_rows, dead_rows = CodeHealth.set_differences(m.files,
                                                                 collect(keys(rec)))
            # ADR 0074: a rename pairs on the raw count alone, because a rename breaks every
            # Dismissal on the file and the reviewed count therefore does not survive the move.
            equal(a, b) = a["raw"] == b["raw"]
            pairs, _, added = CodeHealth.pair_renames(dead_rows, missing_rows, rec,
                                                      measured, equal)
            for (dead, new) in pairs
                # The carried row keeps its old reviewed number, so the gate stays red until the
                # Dismissals name the new path. No new rule creates the red; the subtraction does.
                measured[new] = Dict("raw" => rec[dead]["raw"],
                                     "reviewed" => rec[dead]["reviewed"])
                println("Paired the row of ", dead, " with ", new,
                        ", which measures the same raw",
                        " count. Its reviewed number is carried unchanged.")
            end
            th = CodeHealth.thresholds(rulings)
            for f in added
                if counts[f]["reviewed"] >= th["jet_reviewed"]
                    throw(CodeHealth.RefreshRefused("ERROR: $f enters with reviewed = $(counts[f]["reviewed"]), and an added " *
                                                    "file must enter at 0.\n       Lower it, or add a Dismissal citing an " *
                                                    "approved Rationale."))
                end
            end
        end
        rs = CodeHealth.rises(rec, measured, BINDING; group = run)
        if !isempty(rs) && !accept_rise
            CodeHealth.refuse_rise(NAME, rs)
        end
        push!(out, run => measured)
    end
    io = IOBuffer()
    println(io, "# Generated by code_health/jet.jl. Do not edit by hand.")
    println(io,
            "# Keyed by run and by file. Every run is total over the whole scope, and no")
    println(io,
            "# number is ever summed. The ratchet binds on `reviewed`; `raw` is context.")
    println(io)
    CodeHealth.emit_provenance(io, m.provenance)
    for (run, measured) in out
        println(io)
        println(io, "[run.", run, "]")
        println(io, "load_set = ", CodeHealth.toml_scalar(LOAD_SET))
        println(io)
        CodeHealth.emit_section(io, "run.$run.file",
                                (f => ["raw" => measured[f]["raw"],
                                       "reviewed" => measured[f]["reviewed"]]
                                 for f in m.files))
    end
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
    all_rises = CodeHealth.Rise[]
    for (run, counts) in m.runs
        rec = recorded_rows(recorded, run)
        recorded_set = get(get(recorded_runs(recorded), run, Dict()), "load_set", nothing)
        if recorded_set !== nothing && Set(recorded_set) != Set(LOAD_SET)
            push!(failures,
                  "Run $run records a load set the process does not hold.\n" *
                  "  baseline: $(join(recorded_set, ", "))\n  now:      $(join(LOAD_SET, ", "))")
        end
        measured = Dict(f => Dict(row(counts, f)) for f in m.files)
        missing_rows, dead_rows = CodeHealth.set_differences(m.files, collect(keys(rec)))
        for f in missing_rows
            CodeHealth.annotate(f, "no row in $NAME under run $run.")
        end
        for f in dead_rows
            println("  $NAME names $f under run $run, which no longer exists.")
        end
        if !(isempty(missing_rows) && isempty(dead_rows))
            push!(failures,
                  "Run $run's file set and the tree's file set differ: " *
                  "$(length(missing_rows)) with no row, $(length(dead_rows)) naming no file.")
        end
        rs = CodeHealth.rises(rec, measured, BINDING; group = run)
        for r in rs
            CodeHealth.annotate(r.key,
                                "reviewed JET reports rose from $(r.old) to $(r.new) in run $run.")
        end
        append!(all_rises, rs)
    end
    if !isempty(all_rises)
        push!(failures, "The JET ratchet tripped on $(length(all_rises)) number(s).")
        CodeHealth.step_summary("### JET ratchet\n\n" * CodeHealth.rise_table(all_rises))
    end
    if !(isempty(failures))
        push!(failures, CodeHealth.routes(; dismissal = true))
    end
    return failures, true
end

function publish(m)
    io = IOBuffer()
    println(io, "### JET gate\n")
    println(io, "| field | value |")
    println(io, "| --- | --- |")
    println(io, "| julia | ", VERSION, " |")
    println(io, "| filter | ", FILTER, " |")
    println(io, "| attribution | ", ATTRIBUTION, " |")
    println(io, "| load set | ", join(LOAD_SET, ", "), " |")
    println(io, "| files measured, per run | ", length(m.files), " |")
    for (run, counts) in m.runs
        raw = sum(c -> c["raw"], values(counts); init = 0)
        rev = sum(c -> c["reviewed"], values(counts); init = 0)
        println(io, "| run `", run, "` | raw ", raw, ", reviewed ", rev, " |")
    end
    println("Green. ", length(m.runs), " runs over ", length(m.files), " files each.")
    CodeHealth.step_summary(String(take!(io)))
    return nothing
end

# The scheduled job of ADR 0078 reuses this file's `measure` rather than carrying a second copy
# of it, so the command line runs only when this file is the program. `code_health/triage.jl`
# includes it into a module of its own and calls `measure` directly.
if abspath(PROGRAM_FILE) == @__FILE__
    exit(CodeHealth.run_script(ARGS; name = NAME, measure = measure, verify = verify,
                               render = render, publish = publish))
end
