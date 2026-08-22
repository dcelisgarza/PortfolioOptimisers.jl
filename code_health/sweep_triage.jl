#!/usr/bin/env julia
#
# The deciding half of the scheduled sweep job of ADR 0084.
#
#     julia --project=code_health code_health/sweep_triage.jl --maps <tsv> --existing <tsv>
#
# It reconciles `sweep/manifest.toml` against the tracker, finds every file whose row still reads
# `swept = false` under a CLOSED child map of #404, and writes a plan: the maps to reopen and the
# sub-issues to open. **It opens nothing, it reopens nothing and it writes no row.** The tracker is
# read and written by `gh` in the workflow around it, so every decision this job makes lives here,
# in one place a person can run and read.
#
# It is a sibling of `triage.jl`, not a fifth instrument. The four instruments measure the tree and
# hold a baseline; this one measures nothing. It reads two committed files and a dump of the
# tracker, so it needs no package instantiate: `CodeHealth.jl` loads only `TOML`, which is stdlib.
#
# There is no `READY` file here. `triage.jl` writes one because `--no-jet` succeeds while producing
# a plan that must not be filed. This job has no such half-measured state: it either reconciles the
# manifest against the tracker and writes a plan, or it throws and writes none.

include(joinpath(@__DIR__, "CodeHealth.jl"))

using .CodeHealth
using TOML

const PLAN_DIR = joinpath(CodeHealth.DIR, "_sweep")
const LABEL = "sweep"

"""
The umbrella of the map of maps. It is reopened with every child map the job reopens, because a
child map that reopens makes the umbrella's own terminal condition false again.
"""
const UMBRELLA = 404

const MANIFEST_PATH = joinpath(CodeHealth.REPO_ROOT, "sweep", "manifest.toml")
const COVERAGE_PATH = joinpath(CodeHealth.DIR, "coverage_baseline.toml")

# --- the tracker's side, as data -------------------------------------------

"""
    ChildMap

One child map of #404: the number the manifest's `[map]` table uses, the issue that carries it, its
state, and its name.

**The issue number is read from the tracker, not from a committed file.** The job must know whether
the map is open, and only the tracker holds that, so the dump is needed whatever else is written
down. Keying it by the number in the title costs nothing more, and it cannot go stale the way a
committed table of thirteen issue numbers can.
"""
struct ChildMap
    number::Int
    issue::Int
    open::Bool
    name::String
end

"""
    read_maps(path) -> Dict{String, ChildMap}, Union{Bool, Nothing}

The `wayfinder:map` issues the workflow dumped, one per line as
`number<TAB>state<TAB>title`. The thirteen child maps are the titles that read
`Child map <n>: <name>`; every other map in the tracker is skipped, and the umbrella is returned
separately as its own open flag.

The umbrella's state is read here rather than assumed, because `gh issue reopen` fails on an issue
that is already open. The plan lists a number only when reopening it is a real act.
"""
function read_maps(path::AbstractString)
    out = Dict{String, ChildMap}()
    umbrella = nothing
    if !(isfile(path))
        return out, umbrella
    end
    for (i, line) in enumerate(eachline(path))
        if isempty(strip(line))
            continue
        end
        parts = split(line, '\t')
        if !(length(parts) == 3)
            error("$path line $i has $(length(parts)) fields, not 3: " * repr(line))
        end
        number, state, title = parts
        issue = parse(Int, number)
        isopen = uppercase(state) == "OPEN"
        if issue == UMBRELLA
            umbrella = isopen
            continue
        end
        m = match(r"^Child map (\d+): (.+)$", title)
        if m === nothing
            continue
        end
        n = parse(Int, m.captures[1])
        out[string(n)] = ChildMap(n, issue, isopen, String(m.captures[2]))
    end
    return out, umbrella
end

"""
    read_existing(path) -> Vector{Tuple{Bool, String}}

The `sweep` issues the workflow dumped, one per line as `number<TAB>state<TAB>title`, reduced to
what the safeguard reads: whether each is open, and its title. An absent file means an empty
tracker, which is what the first run sees.
"""
function read_existing(path::AbstractString)
    out = Tuple{Bool, String}[]
    if !(isfile(path))
        return out
    end
    for (i, line) in enumerate(eachline(path))
        if isempty(strip(line))
            continue
        end
        parts = split(line, '\t')
        if !(length(parts) == 3)
            error("$path line $i has $(length(parts)) fields, not 3: " * repr(line))
        end
        push!(out, (uppercase(parts[2]) == "OPEN", String(parts[3])))
    end
    return out
end

"""
    already_filed(existing, path) -> Bool

The safeguard of ADR 0084: skip a path that an **open** sweep issue already names. A closed one is
no match, because a sub-issue closed while the flag stays `false` is exactly the case the job's
self-healing property is there to refile.

The match is `CodeHealth.names_path`, so the path stands as a whole word and a title naming
`src/A.jl.orig` can never suppress `src/A.jl`.
"""
function already_filed(existing, path::AbstractString)
    return any(e -> e[1] && CodeHealth.names_path(e[2], path), existing)
end

# --- reconciliation --------------------------------------------------------

"""
    reconcile(rows, map_names, maps) -> nothing

The manifest and the tracker must agree before a single issue is planned. Three ways they can
disagree, and each one throws:

 1. A child map the manifest names has no issue in the dump. The job cannot reopen what it cannot
    find, so it files nothing rather than filing under the wrong parent.
 2. An issue's name differs from the manifest's. One of the two was renamed alone, and the job has
    no way to tell which.
 3. A row names a map that `[map]` does not list. `test/test_45_sweep_census.jl` already reds the
    build on this, so reaching it here means the census was bypassed.

A loud refusal is the right outcome for all three. This job writes to a tracker, and a wrong parent
or a wrong title is work a person must undo by hand.
"""
function reconcile(rows, map_names, maps)
    bad = String[]
    for (key, name) in sort(collect(map_names); by = first)
        m = get(maps, key, nothing)
        if m === nothing
            push!(bad,
                  "child map $key (\"$name\") has no issue titled \"Child map $key: $name\"")
        elseif !(m.name == name)
            push!(bad,
                  "child map $key is \"$name\" in the manifest and \"$(m.name)\" on issue #$(m.issue)")
        end
    end
    for (path, row) in sort(collect(rows); by = first)
        if !(haskey(map_names, string(row["map"])))
            push!(bad, "`$path` names map $(row["map"]), which `[map]` does not list")
        end
    end
    if !(isempty(bad))
        error("The sweep manifest and the tracker disagree, so nothing is planned:\n  " *
              join(bad, "\n  "))
    end
    return nothing
end

# --- the candidates --------------------------------------------------------

"""
    Candidate

One file to file: its path, the child map that owns it, and the four generated numbers of the body.
`lines` and `misses` are `nothing` for a file that carries no coverage row, which ADR 0074's total
row set makes an anomaly rather than a normal state.
"""
struct Candidate
    path::String
    map::ChildMap
    units::Int
    lines::Union{Int, Nothing}
    misses::Union{Int, Nothing}
end

"""
    candidates(rows, maps, coverage, existing) -> Vector{Candidate}, Vector{Tuple{String, String}}

Every file the trigger fires on, and the files it fired on that the safeguard then skipped.

The trigger is the one condition of ADR 0084: a row that reads `swept = false` **and** a child map
that is **closed**. It is also the deduplication. Run 1 sees a closed map, reopens it and files;
run 2 sees an open map, so the trigger is false and nothing is filed. `already_filed` is a cheap
safeguard on top of it, never the mechanism.

**There is no cap.** `triage.jl` tops up to a five-open queue, because its candidates are a standing
backlog that the ratchet already holds still. These are not: each one is a file that entered the
library after its child map finished, and the first run is the only run that sees the trigger true.
Capping the run would drop the remainder for ever, because run 2 sees the map it just reopened.
"""
function candidates(rows, maps, coverage, existing)
    chosen, skipped = Candidate[], Tuple{String, String}[]
    for (path, row) in sort(collect(rows); by = first)
        if row["swept"]
            continue
        end
        m = maps[string(row["map"])]
        if m.open
            continue
        end
        if already_filed(existing, path)
            push!(skipped, (path, "an open `$LABEL` issue already names it"))
            continue
        end
        cov = get(coverage, path, nothing)
        push!(chosen,
              Candidate(path, m, row["units"], cov === nothing ? nothing : cov["lines"],
                        cov === nothing ? nothing : cov["misses"]))
    end
    return chosen, skipped
end

# --- the issue -------------------------------------------------------------

function title_of(c::Candidate)
    return string("Sweep `", c.path, "` into child map ", c.map.number, ": ", c.map.name)
end

number_or_dash(x) = x === nothing ? "—" : string(x)

"""
    body_of(candidate, commit) -> String

The sub-issue of ADR 0084. It mirrors the child map that owns it, one file wide: a Destination
naming the file and its measured row, the three conditions of #404 restated compactly, the sentence
that the committed files are the authority, and Notes that point at #404 without copying a rule.

**Every field is generated**, so the job needs no judgement: the path, `map` and `units` from
`sweep/manifest.toml`, and `lines` and `misses` from `code_health/coverage_baseline.toml`.

**Nothing here is ever machine-read.** The safeguard reads the title, and every other fact a later
run needs comes from a committed file or from the tracker's own metadata.
"""
function body_of(c::Candidate, commit::AbstractString)
    io = IOBuffer()
    println(io, "## Destination\n")
    println(io, "This file is swept. It joined child map ", c.map.number,
            " after that map closed, so the map and #", UMBRELLA, " were reopened.\n")
    println(io, "| File | Units | Misses | Lines |")
    println(io, "| --- | ---: | ---: | ---: |")
    println(io, "| `", c.path, "` | ", c.units, " | ", number_or_dash(c.misses), " | ",
            number_or_dash(c.lines), " |\n")
    println(io, "Three conditions, from #", UMBRELLA, ":\n")
    println(io, "1. Its documentation states the mathematics.")
    println(io, "2. Its code agrees with that statement, checked with real numbers.")
    println(io, "3. Its lines are covered, or exempted with a reason.\n")
    println(io, "Take the numbers from `sweep/manifest.toml` and ",
            "`code_health/coverage_baseline.toml`, not from this table.\n")
    println(io, "## Notes\n")
    println(io, "Every rule for this effort lives on #", UMBRELLA,
            ". Read it first. This ticket closes when the file's row reads `swept = true`.\n")
    println(io, "---\n")
    println(io, "Filed by the sweep job against `", commit, "`, under child map #",
            c.map.issue, ".")
    return String(take!(io))
end

# --- the plan --------------------------------------------------------------

"""
    write_plan(dir, chosen, umbrella_open, commit) -> nothing

One title file and one body file per sub-issue to open, plus two tab-separated files the workflow
reads and nothing more.

  - `plan.tsv` — one row per issue, as `stem<TAB>path<TAB>parent issue`. It is not called
    `manifest.tsv`, as `triage.jl`'s is, because `sweep/manifest.toml` already owns that noun here.
  - `reopen.tsv` — the issues to reopen, one number per line, **each of them closed**. `gh issue reopen` fails on an issue that is already open, so an open umbrella is left out rather than
    reopened defensively.

An empty plan writes an empty `plan.tsv` and an empty `reopen.tsv`, which is the first run's own
outcome: all thirteen child maps are open today.
"""
function write_plan(dir::AbstractString, chosen, umbrella_open, commit::AbstractString)
    mkpath(dir)
    # Only this job's own files are cleared. The workflow puts the tracker's two dumps in the same
    # directory, and a blanket `rm` of a path that arrives as an option would take them with it.
    for f in readdir(dir)
        if occursin(r"^\d\d\.(title|body)$", f) ||
           f in ("plan.tsv", "reopen.tsv", "summary.md")
            rm(joinpath(dir, f))
        end
    end
    plan = IOBuffer()
    for (i, c) in enumerate(chosen)
        stem = string(i; pad = 2)
        write(joinpath(dir, stem * ".title"), title_of(c))
        write(joinpath(dir, stem * ".body"), body_of(c, commit))
        println(plan, stem, '\t', c.path, '\t', c.map.issue)
    end
    write(joinpath(dir, "plan.tsv"), String(take!(plan)))
    reopen = IOBuffer()
    if !(isempty(chosen))
        for issue in sort(unique(c.map.issue for c in chosen))
            println(reopen, issue)
        end
        # The umbrella closes only when the last child map closes, so a child map that reopens
        # makes its terminal condition false again.
        if umbrella_open === false
            println(reopen, UMBRELLA)
        end
    end
    write(joinpath(dir, "reopen.tsv"), String(take!(reopen)))
    return nothing
end

function summarise(io::IO, rows, chosen, skipped, maps, umbrella_open)
    closed_maps = sort([m.number for m in values(maps) if !(m.open)])
    unswept = count(r -> !(r["swept"]), values(rows))
    println(io, "### The sweep job\n")
    println(io, "| field | value |")
    println(io, "| --- | ---: |")
    println(io, "| files in the manifest | ", length(rows), " |")
    println(io, "| `swept = false` | ", unswept, " |")
    println(io, "| child maps closed | ",
            isempty(closed_maps) ? "none" : join(closed_maps, ", "), " |")
    println(io, "| umbrella #", UMBRELLA, " | ", if umbrella_open === nothing
                "**not in the dump**"
            else
                (umbrella_open ? "open" : "closed")
            end, " |")
    println(io, "| to open | ", length(chosen), " |")
    println(io)
    if !(isempty(chosen))
        println(io, "| file | child map | units | misses | lines |")
        println(io, "| --- | --- | ---: | ---: | ---: |")
        for c in chosen
            println(io, "| `", c.path, "` | #", c.map.issue, " map ", c.map.number, " | ",
                    c.units, " | ", number_or_dash(c.misses), " | ",
                    number_or_dash(c.lines), " |")
        end
        println(io)
    end
    if !(isempty(skipped))
        println(io, "<details><summary>", length(skipped),
                " candidate(s) skipped</summary>\n")
        for (path, reason) in skipped
            println(io, "- `", path, "` — ", reason)
        end
        println(io, "\n</details>")
    end
    return nothing
end

# --- the command line ------------------------------------------------------

struct Options
    maps::String
    existing::String
    out::String
end

"""
    parse_options(args) -> Options

`--maps` is the only required input: without the tracker's state no file can be a candidate, and an
absent dump would make every run vacuously empty rather than loud.
"""
function parse_options(args)
    usage = "usage: sweep_triage.jl --maps <tsv> [--existing <tsv>] [--out <dir>]"
    maps, existing, out = "", "", PLAN_DIR
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("--maps", "--existing", "--out") && i < length(args)
            if a == "--maps"
                (maps = args[i + 1])
            elseif a == "--existing"
                (existing = args[i + 1])
            else
                (out = args[i + 1])
            end
            i += 2
        else
            error(usage)
        end
    end
    if isempty(maps)
        error(usage)
    end
    return Options(maps, existing, out)
end

function main(args)
    opts = parse_options(args)
    manifest = CodeHealth.read_toml(MANIFEST_PATH)
    rows = get(manifest, "file", Dict{String, Any}())
    map_names = get(manifest, "map", Dict{String, Any}())
    coverage = get(CodeHealth.read_toml(COVERAGE_PATH), "file", Dict{String, Any}())
    maps, umbrella_open = read_maps(opts.maps)
    existing = read_existing(opts.existing)

    reconcile(rows, map_names, maps)
    chosen, skipped = candidates(rows, maps, coverage, existing)
    write_plan(opts.out, chosen, umbrella_open, CodeHealth.git_short_commit())

    println(length(rows), " file(s) in the manifest, ", count(m -> !(m.open), values(maps)),
            " child map(s) closed, ", length(chosen), " sub-issue(s) to open.")
    for c in chosen
        println("  ", c.path, " — child map ", c.map.number, ", issue #", c.map.issue)
    end
    for (path, reason) in skipped
        println("  skipped ", path, " — ", reason)
    end
    io = IOBuffer()
    summarise(io, rows, chosen, skipped, maps, umbrella_open)
    text = String(take!(io))
    CodeHealth.step_summary(text)
    write(joinpath(opts.out, "summary.md"), text)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
