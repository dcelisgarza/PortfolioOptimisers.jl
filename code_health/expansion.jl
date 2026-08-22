#!/usr/bin/env julia
#
# The Expansion Bound half of the code-health gate.
#
#     julia --project=code_health code_health/expansion.jl check
#     julia --project=code_health code_health/expansion.jl refresh
#     julia --project=code_health code_health/expansion.jl refresh --accept-rise
#
# It writes `code_health/expansion_bound.toml`, one row per (Declaration Macro, metric), recording
# the worst that macro's expansion adds at any call site beyond what the parser already sees.
#
# ADR 0072 rules that the gate does not expand a file. A Declaration Macro is measured once, where
# it is declared, and that measurement stands for every site that calls it. This script measures
# only the residue that the declaration cannot bound. It loads the package, so it costs about 80
# seconds against the complexity measurement's five.

# `code_health/triage.jl` includes this file into a module of its own, so the include happens once
# and always into `Main`. CodeHealth must be ONE module rather than three: a `Definition` built by a
# second copy would not be the same type as one built by the first.
if !(isdefined(Main, :CodeHealth))
    Main.include(joinpath(@__DIR__, "CodeHealth.jl"))
end

using Main.CodeHealth
using CodeComplexity, Pkg, TOML
using PortfolioOptimisers

const NAME = "expansion_bound.toml"

const METRICS = ("cyc" => CyclomaticComplexity(), "cog" => CognitiveComplexity(),
                 "arg" => ArgumentCountComplexity())

const BINDING = ("cyc", "cog", "arg")

"""
    peak(metric, code) -> Int

The largest value any definition in `code` measures. Zero when the code holds no definition.
"""
function peak(metric, code::AbstractString)
    fns = measure_report(metric, code)
    return isempty(fns) ? 0 : maximum(fn -> fn.value, fns)
end

"""
    addition(expr) -> Dict{String, Int}

What one call site's expansion adds, per metric, beyond what the parser already sees.

The parser descends into a macro call's arguments, so a hand-written inner constructor inside
`@propagatable struct … end` is already measured and already carried by the baseline. Subtracting
the unexpanded call from the expanded one leaves exactly the emitted code.

The expansion is one step deep. A nested macro the expansion emits is left as a macro call, which
is how every other measurement in this gate treats one. ADR 0072 rejected a recursive expansion on
measurement: it pulls `ArgCheck` and `Logging` boilerplate into the numbers, and a gate that counts
them measures those packages rather than this library.
"""
function addition(expr::Expr)
    before = string(expr)
    after = string(macroexpand(PortfolioOptimisers, expr; recursive = false))
    return Dict(key => max(0, peak(metric, after) - peak(metric, before))
                for (key, metric) in METRICS)
end

function measure()
    files = filter(CodeHealth.in_scope, CodeHealth.tracked_jl_files())
    decl = Set(CodeHealth.declaration_macros(files))
    bound = Dict{String, Dict{String, Int}}()
    sites = Dict{String, Int}()
    failed = String[]
    for f in files
        for (name, expr) in CodeHealth.macro_call_sites(f)
            if !(name in decl)
                continue
            end
            sites[name] = get(sites, name, 0) + 1
            local add
            try
                add = addition(expr)
            catch e
                push!(failed, "$name at $f: $(sprint(showerror, e))")
                continue
            end
            row = get!(() -> Dict(k => 0 for k in BINDING), bound, name)
            for k in BINDING
                row[k] = max(row[k], add[k])
            end
        end
    end
    if !(isempty(failed))
        error("A Declaration Macro would not expand:\n" *
              join(failed, "\n") *
              "\nThe Expansion Bound cannot be measured, so the gate reports nothing rather than a\n" *
              "number that is too low.")
    end
    for name in decl
        get!(() -> Dict(k => 0 for k in BINDING), bound, name)
    end
    return (; macros = sort!(collect(decl)), bound, sites, provenance = provenance())
end

function provenance()
    deps = Pkg.dependencies()
    version(name) = string(only(v.version for v in values(deps) if v.name == name))
    return ["julia" => string(VERSION), "code_complexity" => version("CodeComplexity"),
            "julia_syntax" => version("JuliaSyntax"),
            "commit" => CodeHealth.git_short_commit()]
end

row(m, name) = [k => m.bound[name][k] for k in BINDING]

rows(m) = Dict(name => Dict(row(m, name)) for name in m.macros)

recorded_rows(recorded) = get(recorded, "macro", Dict{String, Any}())

# --- render ----------------------------------------------------------------

function render(m, recorded, accept_rise::Bool)
    measured = rows(m)
    rec = recorded_rows(recorded)
    out = Dict{String, Any}()
    rs = CodeHealth.rises(rec, measured, BINDING)
    if !isempty(rs) && !accept_rise
        CodeHealth.refuse_rise(NAME, rs)
    end
    # ADR 0074: no entry test applies to a new key, because the thresholds measure a definition and
    # this file records an addition. A fresh bound is green on arrival.
    for name in m.macros
        out[name] = measured[name]
    end
    io = IOBuffer()
    println(io, "# Generated by code_health/expansion.jl. Do not edit by hand.")
    println(io,
            "# One row per Declaration Macro: the worst its expansion adds at any call site,")
    println(io, "# beyond what the parser already sees at the call. ADR 0072.")
    println(io)
    CodeHealth.emit_provenance(io, m.provenance)
    println(io)
    CodeHealth.emit_section(io, "macro", (name => row(m, name) for name in m.macros))
    return String(take!(io))
end

# --- verify ----------------------------------------------------------------

function verify(m, recorded)
    failures = String[]
    bad_prov = CodeHealth.provenance_failures(get(recorded, "provenance", Dict()),
                                              Dict(m.provenance))
    if !isempty(bad_prov)
        push!(failures, CodeHealth.provenance_message(bad_prov))
        return failures, false
    end
    rec = recorded_rows(recorded)
    missing_rows, dead_rows = CodeHealth.set_differences(m.macros, collect(keys(rec)))
    for name in missing_rows
        println("  $NAME has no row for $name, which is declared and called.")
    end
    for name in dead_rows
        println("  $NAME names $name, which no macro declares or calls.")
    end
    if !(isempty(missing_rows) && isempty(dead_rows))
        push!(failures,
              "The Expansion Bound's key set and the tree's Declaration Macros differ: " *
              "$(length(missing_rows)) missing, $(length(dead_rows)) dead.")
    end
    rs = CodeHealth.rises(rec, rows(m), BINDING)
    for r in rs
        # The offending line is not in the file that changed, so ADR 0077 drops the annotation and
        # leaves the step-summary table to name it.
        println("  ", r.key,
                " now adds $(r.metric) = $(r.new), against a bound of $(r.old).")
    end
    if !isempty(rs)
        push!(failures,
              "The Expansion Bound tripped on $(length(rs)) number(s). " *
              "A Declaration Macro's expansion became worse, and the files that call it " *
              "are not in this diff.")
        CodeHealth.step_summary("### Expansion Bound\n\n" * CodeHealth.rise_table(rs))
    end
    if !(isempty(failures))
        push!(failures, CodeHealth.routes(; dismissal = false))
    end
    return failures, true
end

function publish(m)
    io = IOBuffer()
    println(io, "### Expansion Bound\n")
    println(io, "| macro | call sites | cyc | cog | arg |")
    println(io, "| --- | --- | --- | --- | --- |")
    for name in m.macros
        b = m.bound[name]
        println(io, "| `", name, "` | ", get(m.sites, name, 0), " | ", b["cyc"], " | ",
                b["cog"], " | ", b["arg"], " |")
    end
    println("Green. ", length(m.macros), " Declaration Macros over ",
            sum(values(m.sites); init = 0), " call sites.")
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
