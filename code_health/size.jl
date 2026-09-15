#!/usr/bin/env julia
#
# The file-size half of the code-health gate.
#
#     julia --project=code_health code_health/size.jl check
#     julia --project=code_health code_health/size.jl refresh
#     julia --project=code_health code_health/size.jl refresh --accept-rise
#
# It writes `code_health/size_baseline.toml`, one row per file in scope. `code` is the count of
# lines that carry code, and it is the only number that binds. `doc`, `comment`, `blank` and
# `total` are context.
#
# A raw line count is the wrong instrument on this tree, which is mostly docstring. It fires on
# documentation, which issue #404 asks for, and it stays silent on sprawl. This script counts what
# a reader must hold in their head instead. It is a pure JuliaSyntax parser, so it loads no package
# under measurement and costs about five seconds. ADR 0101.

# `code_health/triage.jl` includes this file into a module of its own, so the include happens once
# and always into `Main`. CodeHealth must be ONE module rather than three: a `Definition` built by a
# second copy would not be the same type as one built by the first.
if !(isdefined(Main, :CodeHealth))
    Main.include(joinpath(@__DIR__, "CodeHealth.jl"))
end

using Main.CodeHealth
using JuliaSyntax, TOML

const NAME = "size_baseline.toml"

"""
The four kinds a line falls into, in the order a baseline row prints them, followed by the total.
The kinds partition the file, so the four always sum to `total`.
"""
const KINDS = ("code", "doc", "comment", "blank")

"""
Every number a row carries. A rename pairs on all of them, which cannot pair two files that merely
share a code-line count.
"""
const ROW_NUMBERS = ("code", "doc", "comment", "blank", "total")

# --- classifying a line ----------------------------------------------------
#
# The classification reads the SOURCE TEXT twice, once as tokens and once as a tree, and never a
# loaded binding. A line-by-line scan for `\"\"\"` and `#` cannot do this job: it reads a long
# string literal in the middle of a function as a docstring. That mistake is common in this
# library, and the tree knows the difference.

"""
    mark!(flags, range)

Set every byte of `range` in `flags`. A token carries a `UInt32` range and a tree node carries an
`Int` one, so the range is taken as it comes and indexed as an `Int`.
"""
function mark!(flags::BitVector, range::AbstractUnitRange{<:Integer})
    for b in range
        i = Int(b)
        if 1 <= i <= length(flags)
            flags[i] = true
        end
    end
    return nothing
end

"""
    walk_syntax(f, node)

Apply `f` to `node` and to every node under it, parents before children and in source order.
"""
function walk_syntax(f, node)
    f(node)
    cs = JuliaSyntax.children(node)
    if cs !== nothing
        for c in cs
            walk_syntax(f, c)
        end
    end
    return nothing
end

"""
    mark_documentation!(isdoc, tree)

Flag the bytes of every docstring in `tree`. A docstring is a `doc` node, and its first child is
the string.

**One rule covers a FIELD docstring too, and that is a property of this parser.**
`CodeHealth.isdocstring` reads a tree from `Meta.parseall`, where a field docstring is a bare
string literal: a struct body binds nothing for `Core.@doc` to attach to, so the `Expr` front end
drops the wrapper. `JuliaSyntax.parseall(SyntaxNode, …)` keeps the `doc` node in both places, so
this reader needs no second shape. Measured over the whole tree: adding a rule for a string
literal standing in a struct body moves no line.

A second rule is not merely unnecessary here. It is **harmful** if it is written for any block
rather than for a struct body, because a string literal in any other block is a VALUE. Reading one
as documentation undercounted `src/01_Base/09_ObservationWeights.jl` by two lines, where the two
branches of an `if` block each build a message string.
`test/test_52_size_classification_census.jl` holds that case.
"""
function mark_documentation!(isdoc::BitVector, tree)
    walk_syntax(tree) do n
        if kind(n) === K"doc"
            cs = JuliaSyntax.children(n)
            if !(cs === nothing || isempty(cs))
                mark!(isdoc, JuliaSyntax.byte_range(first(cs)))
            end
        end
        return nothing
    end
    return nothing
end

is_space(u::UInt8) = u == 0x20 || u == 0x09 || u == 0x0d || u == 0x0a

"""
    line_counts(path) -> Dict{String, Int}

The four counts of [`KINDS`](@ref) for one file, plus `total`.

A line is **code** when it carries one byte that is neither whitespace, nor inside a comment, nor
inside a docstring. It is **doc** when what it carries is docstring alone, **comment** when what it
carries is comment alone, and **blank** when it carries nothing. Code wins a line it shares, so
`\"\"\"docstring\"\"\" f(x) = 1` on one line counts once, as code.
"""
function line_counts(path::AbstractString)
    text = read(path, String)
    n = ncodeunits(text)
    isdoc, iscomment = falses(n), falses(n)
    for t in JuliaSyntax.tokenize(text)
        if kind(t) === K"Comment"
            mark!(iscomment, t.range)
        end
    end
    tree = JuliaSyntax.parseall(JuliaSyntax.SyntaxNode, text; filename = path,
                                ignore_errors = false)
    mark_documentation!(isdoc, tree)
    counts = Dict(k => 0 for k in KINDS)
    total = 0
    first_byte = 1
    # A file that does not end in a newline still ends a line, so the loop closes the last one
    # after the scan rather than inside it.
    for b in 1:(n + 1)
        if b <= n && codeunit(text, b) != 0x0a
            continue
        end
        last_byte = min(b, n)
        code, doc, comment = false, false, false
        for i in first_byte:last_byte
            if is_space(codeunit(text, i))
                continue
            elseif iscomment[i]
                comment = true
            elseif isdoc[i]
                doc = true
            else
                code = true
            end
        end
        if b <= n || last_byte >= first_byte
            total += 1
            key = code ? "code" : doc ? "doc" : comment ? "comment" : "blank"
            counts[key] += 1
        end
        first_byte = b + 1
    end
    counts["total"] = total
    return counts
end

# --- measurement -----------------------------------------------------------

function measure()
    files = filter(CodeHealth.in_scope, CodeHealth.tracked_jl_files())
    counts = Dict{String, Dict{String, Int}}()
    bad = String[]
    for f in files
        try
            counts[f] = line_counts(joinpath(CodeHealth.REPO_ROOT, f))
        catch e
            push!(bad, "$f: $(sprint(showerror, e))")
        end
    end
    if !(isempty(bad))
        error("A file in scope does not parse:\n" *
              join(bad, "\n") *
              "\nThe size ratchet cannot be measured, so the gate reports nothing rather than a\n" *
              "number that is too low.")
    end
    return (; files, counts, provenance = provenance())
end

"""
    provenance() -> Vector{Pair{String, String}}

What measured the numbers. The analyser's version is read with `pkgversion` rather than with
`Pkg.dependencies`, which is what `complexity.jl` and `expansion.jl` use.

The reading is the same and the cost is not. `Pkg` is a dependency of `code_health/Project.toml`
alone, so a script that loads it cannot be loaded from `test/`, and
`test/test_52_size_classification_census.jl` is the fixture adapter for this gate. `pkgversion` is
in `Base`, so the gate loads under any environment that carries `JuliaSyntax` and `TOML`.
"""
function provenance()
    return ["julia" => string(VERSION), "julia_syntax" => string(pkgversion(JuliaSyntax)),
            "commit" => CodeHealth.git_short_commit()]
end

row(c) = [k => c[k] for k in ROW_NUMBERS]

rows(m) = Dict(f => Dict(row(m.counts[f])) for f in m.files)

recorded_rows(recorded) = get(recorded, "file", Dict{String, Any}())

# --- the pass rule ---------------------------------------------------------

"""
    ceiling(recorded_row, limit) -> Int

The largest `code` a file may carry: the greater of the threshold and the number the baseline
records for it. A file with no row takes the threshold.

**This is the whole pass rule, and it is one sentence.** It says two things at once.

  - A file under the threshold is free. Its `code` is context, and ordinary work moves it without
    turning the gate red. Issue #336 measured why that matters: a gate that reddens on every added
    helper is noise, and a noisy gate gets switched off.
  - A file over the threshold is held where it stands. It may fall and it may not rise, so the
    largest files in the tree cannot grow further while no one is looking.

A file that crosses the threshold trips on the crossing, which is the moment a person should be
asked whether the file wants splitting.
"""
function ceiling(recorded_row, limit::Integer)
    if recorded_row === nothing
        return Int(limit)
    end
    return max(Int(limit), Int(recorded_row["code"]))
end

"""
    size_rises(recorded, measured, limit) -> Vector{Rise}

Every file whose `code` stands over its [`ceiling`](@ref). `old` is the ceiling rather than the
recorded number, so a message reads as the rule reads.
"""
function size_rises(recorded, measured, limit::Integer)
    out = CodeHealth.Rise[]
    for (f, new) in measured
        cap = ceiling(get(recorded, f, nothing), limit)
        if new["code"] > cap
            push!(out, CodeHealth.Rise("", f, "code", cap, new["code"]))
        end
    end
    return sort!(out; by = r -> r.key)
end

"""
    code_limit(rulings) -> Int

The size threshold of `code_health/rulings.toml`. It sits in a `[size]` section of its own rather
than in `[thresholds]`, because a `[thresholds]` number never reaches a pass rule and this one
does.
"""
code_limit(rulings) = rulings["size"]["code_lines"]

# --- render ----------------------------------------------------------------

function render(m, recorded, accept_rise::Bool)
    rulings = CodeHealth.read_rulings()
    limit = code_limit(rulings)
    measured = rows(m)
    rec = recorded_rows(recorded)
    if !isempty(rec)
        missing_rows, dead_rows = CodeHealth.set_differences(m.files, collect(keys(rec)))
        equal(a, b) = all(k -> a[k] == b[k], ROW_NUMBERS)
        pairs, _, _ = CodeHealth.pair_renames(dead_rows, missing_rows, rec, measured, equal)
        for (dead, new) in pairs
            println("Paired the row of ", dead, " with ", new, ", which measures the same.")
        end
    end
    rs = size_rises(rec, measured, limit)
    if !isempty(rs) && !accept_rise
        CodeHealth.refuse_rise(NAME, rs)
    end
    # A refresh that gets this far records the truth: every ceiling it writes either fell, or the
    # rise was asked for by name.
    io = IOBuffer()
    println(io, "# Generated by code_health/size.jl. Do not edit by hand.")
    println(io, "# One row per file in scope: `code` binds and the rest is context.")
    println(io,
            "# `code` binds only where it stands over the threshold in code_health/rulings.toml.")
    println(io, "# ADR 0101.")
    println(io)
    CodeHealth.emit_provenance(io, m.provenance)
    println(io)
    CodeHealth.emit_section(io, "file", (f => row(m.counts[f]) for f in m.files))
    return String(take!(io))
end

# --- verify ----------------------------------------------------------------

function verify(m, recorded)
    rulings = CodeHealth.read_rulings()
    limit = code_limit(rulings)
    failures = String[]
    bad_prov = CodeHealth.provenance_failures(get(recorded, "provenance", Dict()),
                                              Dict(m.provenance))
    if !isempty(bad_prov)
        push!(failures, CodeHealth.provenance_message(bad_prov))
        return failures, false
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
    rs = size_rises(rec, rows(m), limit)
    for r in rs
        CodeHealth.annotate(r.key,
                            "code lines rose to $(r.new), over a ceiling of $(r.old). " *
                            "The threshold is $limit.")
    end
    if !isempty(rs)
        push!(failures,
              "The size ratchet tripped on $(length(rs)) file(s). " *
              "A file over $limit code lines may fall and may not rise.")
        CodeHealth.step_summary("### Size ratchet\n\n" * CodeHealth.rise_table(rs))
    end
    if !(isempty(failures))
        push!(failures, CodeHealth.routes(; dismissal = false))
    end
    return failures, true
end

function publish(m)
    rulings = CodeHealth.read_rulings()
    limit = code_limit(rulings)
    code(f) = m.counts[f]["code"]
    over = sort!(filter(f -> code(f) > limit, m.files); by = code, rev = true)
    total(k) = sum(f -> m.counts[f][k], m.files; init = 0)
    worst = isempty(m.files) ? "" : argmax(code, m.files)
    io = IOBuffer()
    println(io, "### Size ratchet\n")
    println(io, "| field | value |")
    println(io, "| --- | --- |")
    println(io, "| files measured | ", length(m.files), " |")
    println(io, "| code lines | ", total("code"), " |")
    println(io, "| docstring lines | ", total("doc"), " |")
    println(io, "| threshold | ", limit, " |")
    println(io, "| files over the threshold | ", length(over), " |")
    if !(isempty(over))
        println(io, "\n| file | code | total |")
        println(io, "| --- | --- | --- |")
        for f in over
            println(io, "| `", f, "` | ", code(f), " | ", m.counts[f]["total"], " |")
        end
    end
    doc_share = total("total") == 0 ? 0.0 : 100 * total("doc") / total("total")
    println("Green. ", length(m.files), " files measured, ", total("code"), " code lines, ",
            round(doc_share; digits = 1), "% docstring. Worst file ", worst, " at ",
            isempty(worst) ? 0 : code(worst), ".")
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
