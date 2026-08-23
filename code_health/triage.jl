#!/usr/bin/env julia
#
# The deciding half of the scheduled job of ADR 0078.
#
#     julia --project=code_health code_health/triage.jl --existing <tsv>
#     julia --project=code_health code_health/triage.jl --existing <tsv> --no-jet
#
# It measures the tree, finds the files that stand above a threshold, ranks them, applies the
# deduplication rule, and writes a plan of the issues that should be opened. **It opens nothing and
# it reads no issue body.** The tracker is read and written by `gh` in the workflow around it, so
# every decision this job makes lives here, in one place a person can run and read.
#
# The two measuring scripts are reused rather than copied. Each is included into a module of its
# own, because all three scripts name their entry point `measure`.
#
# The input is what `gh issue list --label code-health --state all` produced, one issue per line as
# `number<TAB>state<TAB>closedAt<TAB>title`. The output is a directory of title and body files.

include(joinpath(@__DIR__, "CodeHealth.jl"))

using .CodeHealth
using TOML

# CodeHealth is included FIRST and into `Main`, so the two scripts below find it already defined
# and share it. `Definition` and `Reviewed` cross the boundary between them and this file, and a
# type is only one type when one module defines it.
module Complexity
include(joinpath(@__DIR__, "complexity.jl"))
end

# JET is loaded only when it will be used. Its module loads PortfolioOptimisers and the whole plot
# and imputation load set, which is the entire cost of `--no-jet` avoiding it. A top-level `@eval`
# is what puts the module in `Main` from inside a conditional.
const WITH_JET = !("--no-jet" in ARGS)

if WITH_JET
    @eval module Jet
    include(joinpath($(@__DIR__), "jet.jl"))
    end
end

const PLAN_DIR = joinpath(CodeHealth.DIR, "_triage")
const LABEL = "code-health"

"""
The three complexity metrics, as the row key, the name a person reads, and the ruling that limits
it. JET is not here: it carries no ratio, and one reviewed report makes a file a candidate.
"""
const METRICS = ("cyc" => ("cyclomatic", "cyclomatic"), "cog" => ("cognitive", "cognitive"),
                 "arg" => ("argument count", "argcount"))

"""
The complexity numbers the refile clause reads out of a baseline row. The JET side is not listed
here, because a run's name is not known until the file is parsed: `recorded_numbers` adds one
`jet:<run>` key per run it finds. Together the two make the ratchet's binding set of ADR 0076.
"""
const REFILE_METRICS = ("cyc", "cog", "arg")

"""
How many worst definitions the body lists for one breached metric. The maximum is what binds, so
the list is a starting point for the session rather than a census.
"""
const WORST_SHOWN = 5

"""
How many JET reports the body lists. A file with more than this has a systematic class, and the
reproduce command shows the rest.
"""
const REPORTS_SHOWN = 25

# --- the rulings this job reads --------------------------------------------

struct Config
    limits::Dict{String, Int}
    jet_reviewed::Int
    open_queue::Int
end

"""
    read_config(rulings) -> Config

ADR 0078 keeps every number of this job in `code_health/rulings.toml` rather than as a literal
here, so changing one is a single reviewed line. Nothing is defaulted: a missing key is a broken
ruling, not a reason to invent a number.
"""
function read_config(rulings)
    th = CodeHealth.thresholds(rulings)
    job = rulings["scheduled_job"]
    limits = Dict(key => th[ruling] for (key, (_, ruling)) in METRICS)
    return Config(limits, th["jet_reviewed"], job["open_queue"])
end

# --- the tracker's side, as data -------------------------------------------

struct Existing
    number::Int
    open::Bool
    closed_at::String
    title::String
end

"""
    read_existing(path) -> Vector{Existing}

The `code-health` issues the workflow dumped, open and closed alike. An absent file means an empty
tracker, which is what the first run sees.

`closed_at` stays the string GitHub printed. It is fixed-width ISO-8601 in UTC, so it sorts
lexicographically and `git rev-list --before=` reads it as it stands. Parsing it would buy nothing
and would add a dependency to a job that needs none.
"""
function read_existing(path::AbstractString)
    out = Existing[]
    if !(isfile(path))
        return out
    end
    for (i, line) in enumerate(eachline(path))
        if isempty(strip(line))
            continue
        end
        parts = split(line, '\t')
        if !(length(parts) == 4)
            error("$path line $i has $(length(parts)) fields, not 4: " * repr(line))
        end
        number, state, closed, title = parts
        if closed in ("null", "0001-01-01T00:00:00Z")
            (closed = "")
        end
        push!(out,
              Existing(parse(Int, number), uppercase(state) == "OPEN", String(closed),
                       String(title)))
    end
    return out
end

# --- git, for the refile clause --------------------------------------------

"""
    commit_before(when) -> Union{String, Nothing}

The last commit at or before `when` on the checked-out history. **The checkout needs the depth to
reach it**, so the workflow fetches the full history: a shallow clone answers nothing here and the
job would silently never refile.
"""
function commit_before(when::AbstractString)
    sha = try
        strip(read(Cmd(`git rev-list -1 --before=$when HEAD`; dir = CodeHealth.REPO_ROOT),
                   String))
    catch
        ""
    end
    return isempty(sha) ? nothing : String(sha)
end

function file_at(commit::AbstractString, path::AbstractString)
    try
        return read(Cmd(`git show $commit:$path`; dir = CodeHealth.REPO_ROOT), String)
    catch
        return nothing
    end
end

"""
    recorded_numbers(commit, path) -> Dict{String, Int}

The baseline row one file carried at one commit, as a flat map of metric to number. The three
complexity numbers keep their own keys and each JET run contributes `jet:<run>`.

The **committed baseline** is read rather than a fresh measurement, because the number a file
carried at a past commit exists nowhere else. ADR 0078 chose git over a label, a fifth committed
file and the issue body, and it needs only `contents: read`.
"""
function recorded_numbers(commit::AbstractString, path::AbstractString)
    out = Dict{String, Int}()
    text = file_at(commit, "code_health/complexity_baseline.toml")
    if text !== nothing
        row = get(get(TOML.parse(text), "file", Dict{String, Any}()), path, nothing)
        if row !== nothing
            for k in REFILE_METRICS
                haskey(row, k) && (out[k] = row[k])
            end
        end
    end
    text = file_at(commit, "code_health/jet_baseline.toml")
    if text !== nothing
        for (run, block) in get(TOML.parse(text), "run", Dict{String, Any}())
            row = get(get(block, "file", Dict{String, Any}()), path, nothing)
            if row === nothing
                continue
            end
            haskey(row, "reviewed") && (out["jet:" * run] = row["reviewed"])
        end
    end
    return out
end

"""
    rose(before, now) -> Bool

ADR 0078's second clause, as one predicate.

```text
skip  if  a match exists  and  baseline_at(closed_at) >= baseline_now
file  if  a match exists  and  baseline_at(closed_at) <  baseline_now
```

A metric the older commit does not record cannot be shown to have risen, so it is skipped. The job
never invents a number, which is ADR 0074's rule applied to history.
"""
function rose(before::Dict{String, Int}, now::Dict{String, Int})
    return any(k -> haskey(before, k) && now[k] > before[k], keys(now))
end

# --- candidacy -------------------------------------------------------------

struct Breach
    name::String
    value::Int
    baseline::Int
    threshold::Int
end

struct Candidate
    path::String
    breaches::Vector{Breach}
    worst::Vector{Pair{String, Vector{CodeHealth.Definition}}}
    reports::Vector{CodeHealth.Reviewed}
    max_excess::Float64
end

has_jet(c::Candidate) = !(isempty(c.reports))

"""
    surviving(numbers, path, rulings, key) -> Vector{Definition}

The definitions of one file under one metric, with the Exemptions dropped and the worst first. An
Exemption is dropped **before** the maximum is taken, so it binds candidacy alone and never the
baseline, which is the whole of its authority under ADR 0078.
"""
function surviving(numbers, path::AbstractString, rulings, key::AbstractString)
    exempt = CodeHealth.exempt_definitions(rulings, key)
    kept = [d for d in numbers.definitions[key] if !((path, d.name) in exempt)]
    return sort!(kept; by = d -> (-d.value, d.line, d.name))
end

"""
    candidates(cm, jm, rulings, cfg, baseline, jet_baseline) -> Vector{Candidate}

Every file that stands above a threshold, worst first.

The ranking is ADR 0078's: a file carrying a reviewed JET report outranks every complexity-only
file, because a suspected defect outranks a refactor, and inside each group the **excess ratio**
orders them. The ratio is used rather than the raw value because a cyclomatic 12 and an argument
count 12 are not the same distance past their own thresholds. The path breaks a tie, so two runs
over one tree rank the same files in the same order.
"""
function candidates(cm, jm, rulings, cfg::Config, baseline, jet_baseline)
    out = Candidate[]
    for path in cm.files
        numbers = cm.numbers[path]
        row = get(baseline, path, nothing)
        breaches, worst = Breach[], Pair{String, Vector{CodeHealth.Definition}}[]
        excess = 0.0
        for (key, (name, _)) in METRICS
            limit = cfg.limits[key]
            kept = surviving(numbers, path, rulings, key)
            value = isempty(kept) ? 0 : first(kept).value
            excess = max(excess, value / limit)
            if !(value > limit)
                continue
            end
            recorded = row === nothing ? numbers.max[key] : row[key]
            push!(breaches, Breach(name, value, recorded, limit))
            push!(worst, key => first(kept, WORST_SHOWN))
        end
        reports = jm === nothing ? CodeHealth.Reviewed[] : jm.reviewed[path]
        if length(reports) >= cfg.jet_reviewed
            recorded = sum(r -> get(get(jet_baseline, r, Dict{String, Any}()), path,
                                    Dict("reviewed" => 0))["reviewed"], keys(jet_baseline);
                           init = 0)
            push!(breaches,
                  Breach("JET reports, reviewed", length(reports), recorded,
                         cfg.jet_reviewed))
        end
        if isempty(breaches)
            continue
        end
        push!(out, Candidate(path, breaches, worst, reports, excess))
    end
    return sort!(out; by = c -> (has_jet(c) ? 0 : 1, -c.max_excess, c.path))
end

# --- deduplication ---------------------------------------------------------

"""
    verdict(candidate, existing) -> (action, reason)

ADR 0078's deduplication rule, both clauses. `action` is `:file`, `:refile` or `:skip`.

An **open** match always skips: the file already holds a slot in the queue. A **closed** match
skips as well, unless the file's recorded number now stands above the number it carried when that
issue closed. The most recent closed issue is the one consulted, because it is the last time a
person signed the file off.
"""
function verdict(c::Candidate, existing::Vector{Existing})
    matches = filter(e -> CodeHealth.names_path(e.title, c.path), existing)
    if isempty(matches)
        return :file, "never filed"
    end
    open_match = findfirst(e -> e.open, matches)
    if open_match !== nothing
        return :skip, "issue #$(matches[open_match].number) is open"
    end
    closed = filter(e -> !(isempty(e.closed_at)), matches)
    if isempty(closed)
        return :skip, "issue #$(first(matches).number) matched and records no close time"
    end
    last = argmax(e -> e.closed_at, closed)
    commit = commit_before(last.closed_at)
    if commit === nothing
        return :skip,
               "issue #$(last.number) closed at $(last.closed_at), and the checkout reaches no " *
               "commit at or before it"
    end
    before, now = recorded_numbers(commit, c.path), recorded_numbers("HEAD", c.path)
    if rose(before, now)
        return :refile,
               "issue #$(last.number) closed at $(last.closed_at), and the baseline has risen " *
               "since $(first(commit, 10))"
    end
    return :skip,
           "issue #$(last.number) closed at $(last.closed_at) and nothing has risen since"
end

# --- the issue a person reads ----------------------------------------------

title_of(c::Candidate) = "$LABEL: $(c.path)"

function metric_table(c::Candidate)
    io = IOBuffer()
    println(io, "| metric | value | baseline | threshold |")
    println(io, "| --- | ---: | ---: | ---: |")
    for b in c.breaches
        println(io, "| ", b.name, " | ", b.value, " | ", b.baseline, " | ", b.threshold,
                " |")
    end
    return String(take!(io))
end

function definition_tables(c::Candidate)
    io = IOBuffer()
    for (key, defs) in c.worst
        name = first(METRICS[findfirst(p -> p.first == key, METRICS)].second)
        println(io, "**", name, "**\n")
        println(io, "| definition | value | line |")
        println(io, "| --- | ---: | ---: |")
        for d in defs
            println(io, "| `", d.name, "` | ", d.value, " | ", d.line, " |")
        end
        println(io)
    end
    return String(take!(io))
end

function report_table(c::Candidate)
    io = IOBuffer()
    println(io, "| run | line | kind | message |")
    println(io, "| --- | ---: | --- | --- |")
    for r in first(c.reports, REPORTS_SHOWN)
        message = replace(strip(r.message), '\n' => " ", '|' => "\\|")
        println(io, "| ", r.run, " | ", r.line, " | `", r.kind, "` | ", message, " |")
    end
    if length(c.reports) > REPORTS_SHOWN
        println(io, "\n", length(c.reports) - REPORTS_SHOWN,
                " more reports are not listed. Run the check to see them all.")
    end
    return String(take!(io))
end

"""
    body_of(candidate, action, reason, commit) -> String

The issue body of ADR 0078: the path, each breached metric with its value, its baseline and its
threshold, the worst definitions by name and line, the JET reports by kind and site, and the
reproduce command.

**Nothing here is ever machine-read.** Every fact a later run needs comes from a committed file or
from the tracker's own metadata, so this text is free to change without breaking the job.
"""
function body_of(c::Candidate, action::Symbol, reason::AbstractString,
                 commit::AbstractString)
    io = IOBuffer()
    println(io, "`", c.path, "` stands above a code-health threshold.\n")
    if action === :refile
        println(io,
                "This file was filed and closed before, and its recorded number has risen since. ",
                "A rise recorded with `refresh --accept-rise` is a correct outcome, so this is a ",
                "standing reminder rather than a regression. See ADR 0078.\n")
    end
    println(io, "## Breached metrics\n")
    println(io, metric_table(c))
    # The two columns measure different things on purpose, and a reader who does not know that
    # reads a contradiction. An Exemption is dropped before the maximum for candidacy and never
    # moves the baseline, so the baseline is the higher of the two exactly when one applies here.
    if any(b -> b.baseline > b.value, c.breaches)
        println(io,
                "Where the baseline stands **above** the value, the difference is an Exemption. ",
                "An Exemption is dropped before the maximum is taken for candidacy, and it never ",
                "moves the baseline.\n")
    end
    if any(b -> b.value > b.baseline, c.breaches)
        println(io,
                "Where the value stands **above** the baseline, the ratchet is red on this ",
                "branch. Clear that first: this issue asks for a number to fall, not to be ",
                "recorded.\n")
    end
    if !(isempty(c.worst))
        println(io, "## The worst definitions\n")
        println(io,
                "The file's number is the **maximum** over its definitions, so only the worst ",
                "one moves it.\n")
        print(io, definition_tables(c))
    end
    if has_jet(c)
        println(io, "## JET reports\n")
        println(io,
                "Correctness outranks maintainability, so these are worked first. Each report is ",
                "either real, and is fixed, or a false positive, and is dismissed.\n")
        println(io, report_table(c))
    end
    println(io, "## Reproduce\n")
    println(io, "```bash")
    println(io, "julia --project=code_health code_health/complexity.jl check")
    if has_jet(c)
        println(io, "julia --project=code_health code_health/jet.jl check")
    end
    println(io, "```\n")
    println(io, "The procedure is on the [Code health](",
            "https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/src/contribute/",
            "3-code-health.md) page. **No `code-health` issue closes without a commit**: a number ",
            "that fell, a Dismissal, or an Exemption.\n")
    println(io, "---\n")
    println(io, "Filed by the code-health job against `", commit, "`. Reason: ", reason,
            ".")
    return String(take!(io))
end

# --- the plan --------------------------------------------------------------

"""
    write_plan(dir, chosen, commit, jet_ran) -> nothing

One title file and one body file per issue to open, plus a manifest. The workflow opens them with
`gh` and nothing more, so a dry run is the same plan left unposted rather than a second code path.

`READY` is written only when JET ran. The ranking's first key is whether a file carries a reviewed
report, so a plan measured without JET is ranked on half the evidence and must never reach the
tracker.
"""
function write_plan(dir::AbstractString, chosen, commit::AbstractString, jet_ran::Bool)
    mkpath(dir)
    # Only this job's own files are cleared. The workflow puts the tracker's dump in the same
    # directory, and a blanket `rm` of a path that arrives as an option would take it with it.
    for f in readdir(dir)
        if occursin(r"^\d\d\.(title|body)$", f) ||
           f in ("manifest.tsv", "summary.md", "READY")
            rm(joinpath(dir, f))
        end
    end
    manifest = IOBuffer()
    for (i, (c, action, reason)) in enumerate(chosen)
        stem = joinpath(dir, string(i; pad = 2))
        write(stem * ".title", title_of(c))
        write(stem * ".body", body_of(c, action, reason, commit))
        println(manifest, string(i; pad = 2), '\t', c.path, '\t', action)
    end
    write(joinpath(dir, "manifest.tsv"), String(take!(manifest)))
    if jet_ran
        write(joinpath(dir, "READY"), "")
    end
    return nothing
end

function summarise(io::IO, all, chosen, verdicts, cfg::Config, open_now::Int, jet_ran::Bool)
    println(io, "### The code-health job\n")
    println(io, "| field | value |")
    println(io, "| --- | ---: |")
    println(io, "| candidates | ", length(all), " |")
    println(io, "| open already | ", open_now, " |")
    println(io, "| room in the queue | ", max(0, cfg.open_queue - open_now), " |")
    println(io, "| to open | ", length(chosen), " |")
    println(io, "| JET measured | ", jet_ran ? "yes" : "**no**", " |")
    println(io)
    if !(isempty(chosen))
        println(io, "| file | action | reason |")
        println(io, "| --- | --- | --- |")
        for (c, action, reason) in chosen
            println(io, "| `", c.path, "` | ", action, " | ", reason, " |")
        end
        println(io)
    end
    skipped = [(c, r) for (c, a, r) in verdicts if a === :skip]
    if !(isempty(skipped))
        println(io, "<details><summary>", length(skipped),
                " candidate(s) skipped</summary>\n")
        for (c, r) in skipped
            println(io, "- `", c.path, "` — ", r)
        end
        println(io, "\n</details>")
    end
    return nothing
end

# --- the command line ------------------------------------------------------

struct Options
    existing::String
    out::String
end

"""
    parse_options(args) -> Options

`--no-jet` is read at load time, into `WITH_JET`, because it decides whether the JET module is
loaded at all. It is accepted here so that it is not an error, and it is not read twice.
"""
function parse_options(args)
    usage = "usage: triage.jl [--existing <tsv>] [--out <dir>] [--no-jet]"
    existing, out = "", PLAN_DIR
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--no-jet"
            i += 1
        elseif a in ("--existing", "--out") && i < length(args)
            a == "--existing" ? (existing = args[i + 1]) : (out = args[i + 1])
            i += 2
        else
            error(usage)
        end
    end
    return Options(existing, out)
end

function main(args)
    opts = parse_options(args)
    rulings = CodeHealth.read_rulings()
    cfg = read_config(rulings)
    existing = read_existing(opts.existing)
    open_now = count(e -> e.open, existing)
    room = max(0, cfg.open_queue - open_now)

    if !(WITH_JET)
        println(stderr,
                "WARNING: --no-jet measures complexity alone. The ranking's first key is a " *
                "reviewed\n         JET report, so this plan must not be filed. It is a " *
                "development aid.")
    end
    cm = Complexity.measure()
    jm = WITH_JET ? Main.Jet.measure() : nothing
    baseline = get(CodeHealth.read_toml(joinpath(CodeHealth.DIR,
                                                 "complexity_baseline.toml")), "file",
                   Dict{String, Any}())
    jet_baseline = Dict(run => get(block, "file", Dict{String, Any}())
                        for (run, block) in
                            get(CodeHealth.read_toml(joinpath(CodeHealth.DIR,
                                                              "jet_baseline.toml")), "run",
                                Dict{String, Any}()))

    all = candidates(cm, jm, rulings, cfg, baseline, jet_baseline)
    verdicts = [(c, verdict(c, existing)...) for c in all]
    chosen = first([v for v in verdicts if v[2] !== :skip], room)
    commit = CodeHealth.git_short_commit()
    write_plan(opts.out, chosen, commit, WITH_JET)

    println(length(all), " candidate(s), ", open_now, " open already, room for ", room,
            ", ", length(chosen), " to open.")
    for (c, action, reason) in chosen
        println("  ", action, " ", c.path, " — ", reason)
    end
    io = IOBuffer()
    summarise(io, all, chosen, verdicts, cfg, open_now, WITH_JET)
    text = String(take!(io))
    CodeHealth.step_summary(text)
    write(joinpath(opts.out, "summary.md"), text)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
