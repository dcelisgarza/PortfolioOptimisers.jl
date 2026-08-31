#!/usr/bin/env julia
#
# The conformance half of the sweep of ADR 0084, run BEFORE the commit rather than after it.
#
#     julia --project=code_health code_health/sweep_check.jl
#     julia --project=code_health code_health/sweep_check.jl --base origin/dev
#     julia --project=code_health code_health/sweep_check.jl --file src/26_New.jl --fetch
#     julia --project=code_health code_health/sweep_check.jl --all
#
# It takes the files a branch added or changed under `src/` and `ext/`, and reports every duty the
# sweep places on them: the manifest row, the unit count, the child map, the coverage entry, the
# `include` line, and — when the tracker is read — the state of the child map and whether a sub-issue
# already names the path. It prints the exact line to paste and the exact command to run for each
# failure, and it exits non-zero when one stands.
#
# **It measures. It writes nothing, and it touches no tracker.** `code_health/sweep_triage.jl`
# plans and `code_health/sweep_issues.sh` applies; this one only says what is owed.
#
# It is a convenience, never an Authority. Every rule it reports is owned elsewhere, and the report
# names the owner on every line:
#
#   the row, the unit count, the child map  `test/test_45_sweep_census.jl`
#   the coverage entry                      ADR 0082, `code_health/coverage.jl`
#   the `include` line                      `test/test_47_alias_and_module_census.jl`
#   the `algorithm` floor of a swept row    `test/test_26_docs.jl`
#   the sub-issue and the reopened map      ADR 0084, `code_health/sweep_triage.jl`
#
# So a green run here is not a green build. It is the four steps of `CLAUDE.md` § *Functionality you
# add*, checked while they can still be done in the commit that owes them.

include(joinpath(@__DIR__, "CodeHealth.jl"))

using .CodeHealth
using TOML

const UMBRELLA = 404
const LABEL = "sweep"
const PLAN_DIR = joinpath(CodeHealth.DIR, "_sweep")

const MANIFEST_PATH = joinpath(CodeHealth.REPO_ROOT, "sweep", "manifest.toml")
const COVERAGE_PATH = joinpath(CodeHealth.DIR, "coverage_baseline.toml")
const ENTRY_PATH = joinpath(CodeHealth.REPO_ROOT, "src", "PortfolioOptimisers.jl")

# --- what is in scope ------------------------------------------------------

"""
    in_scope(path) -> Bool

The sweep measures `src/` and `ext/` and nothing else, which is ADR 0072's scope and ADR 0074's row
set. A change to a test, to a document or to `code_health/` owes this check nothing.
"""
function in_scope(path::AbstractString)
    return endswith(path, ".jl") && any(r -> startswith(path, r), CodeHealth.MEASURED_ROOTS)
end

"""
    git_lines(args...) -> Vector{String}

Run `git` in the repository root and split its output on newlines. An empty answer is an empty
vector, never a vector holding one empty string.
"""
function git_lines(args::Cmd)
    out = read(Cmd(args; dir = CodeHealth.REPO_ROOT), String)
    return String.(filter!(!isempty, split(out, '\n')))
end

"""
    changed_files(base) -> Vector{String}, String

Every file in scope that this branch touches: the ones that differ from `base`, the ones staged or
edited in the working tree, and the untracked ones.

The untracked half is the reason this cannot be a `git diff` alone. **A brand-new file is exactly
the case the sweep is aimed at**, and until it is added it appears in no diff at all.

`base` falls back when the ref does not resolve, so a fresh clone or a detached checkout still
reports something rather than throwing. The ref actually used is returned, so the report can name
it and the reader can tell a fallback from a choice.
"""
function changed_files(base::AbstractString)
    ref = base
    ok = try
        success(Cmd(`git rev-parse --verify --quiet $base`; dir = CodeHealth.REPO_ROOT))
    catch
        false
    end
    if !(ok)
        ref = "HEAD"
    end
    files = String[]
    append!(files, git_lines(`git diff --name-only $ref --`))
    append!(files, git_lines(`git diff --name-only --cached $ref --`))
    append!(files, git_lines(`git ls-files --others --exclude-standard -- '*.jl'`))
    return sort!(unique!(filter!(in_scope, files))), ref
end

# --- the manifest row ------------------------------------------------------

"""
    row_line(f, m, u, s; algorithm) -> String

The manifest line a person pastes. It mirrors the printer of `test/test_45_sweep_census.jl`,
including its `algorithm` clause: a swept row carries that key, and a line pasted without it would
delete the floor `test/test_26_docs.jl` holds, where the deletion would read as a correction.
"""
function row_line(f, m, u, s; algorithm = nothing)
    a = isnothing(algorithm) ? "" : string(", algorithm = ", algorithm)
    return string("\"", f, "\" = { map = ", m, ", units = ", u, a, ", swept = ", s, " }")
end

"""
    candidate_maps(rows, map_names, f) -> Vector{Int}

The child maps a file's own directory already uses, which is what the census prints and for the
same reason: **`map` is not derivable from a path**. Each of the nine subdirectories of `src/` and
`ext/` uses exactly one map, and there the answer is an answer. The top level of `src/` holds files
across five maps, and the numeric prefix does not rescue the lookup. A file in a brand-new
directory has no sibling row, and then every map is a candidate.
"""
function candidate_maps(rows, map_names, f::AbstractString)
    d = dirname(f)
    ms = sort(unique(r["map"] for (g, r) in rows if dirname(g) == d))
    return isempty(ms) ? sort(parse.(Int, collect(keys(map_names)))) : ms
end

# --- the tracker, when it was read -----------------------------------------

"""
    Tracker

What the two dumps say, reduced to what this check reads: the child maps by number, whether the
umbrella is open, and the sweep issues as open-flag and title.

`nothing` for the whole struct means the tracker was not read. That is a normal outcome, not a
failure: the manifest half of the check needs no network and must still run when `gh` is absent or
unauthenticated.
"""
struct Tracker
    maps::Dict{Int, Tuple{Int, Bool, String}}
    umbrella_open::Union{Bool, Nothing}
    issues::Vector{Tuple{Bool, String}}
end

function read_tsv(path::AbstractString)
    out = Tuple{Int, Bool, String}[]
    if !(isfile(path))
        return out
    end
    for line in eachline(path)
        if isempty(strip(line))
            continue
        end
        parts = split(line, '\t')
        if !(length(parts) == 3)
            error("$path holds a line with $(length(parts)) fields, not 3: " * repr(line))
        end
        push!(out, (parse(Int, parts[1]), uppercase(parts[2]) == "OPEN", String(parts[3])))
    end
    return out
end

"""
    read_tracker(maps_path, existing_path) -> Union{Tracker, Nothing}

The two dumps `code_health/sweep_issues.sh dump` writes, read the way
`code_health/sweep_triage.jl` reads them: the child maps are the titles that read
`Child map <n>: <name>`, and the umbrella is read by number.

An absent maps dump returns `nothing`, so the caller reports the tracker as unread rather than
reporting every child map as missing.
"""
function read_tracker(maps_path::AbstractString, existing_path::AbstractString)
    if !(isfile(maps_path))
        return nothing
    end
    maps = Dict{Int, Tuple{Int, Bool, String}}()
    umbrella = nothing
    for (issue, isopen, title) in read_tsv(maps_path)
        if issue == UMBRELLA
            umbrella = isopen
            continue
        end
        m = match(r"^Child map (\d+): (.+)$", title)
        if m === nothing
            continue
        end
        maps[parse(Int, m.captures[1])] = (issue, isopen, String(m.captures[2]))
    end
    issues = [(isopen, title) for (_, isopen, title) in read_tsv(existing_path)]
    return Tracker(maps, umbrella, issues)
end

# --- the findings ----------------------------------------------------------

"""
    Finding

One line of the report: its level, its text, and the lines that follow it indented.

`:fail` is a duty the file owes and does not meet, and one of them makes the run exit non-zero.
`:note` is a duty the check cannot settle by itself — a coverage entry it has not measured, a
tracker it did not read — and it never fails the run. Reporting a duty the check cannot measure as
a failure would teach the reader to ignore the level.
"""
struct Finding
    level::Symbol
    text::String
    detail::Vector{String}
end

Finding(level, text) = Finding(level, text, String[])

const OWES = Dict(:fail => "[fail]", :note => "[note]", :ok => "[ ok ]")

function print_findings(io::IO, path, fs)
    println(io, path)
    for f in fs
        println(io, "  ", OWES[f.level], " ", f.text)
        for d in f.detail
            println(io, "         ", d)
        end
    end
    println(io)
    return nothing
end

"""
    check_file(path, rows, map_names, coverage, entry, tracker, exempted, survey) -> Vector{Finding}

Every duty the sweep places on one file, in the order a person meets them.

 1. The row exists. Without it `test/test_45_sweep_census.jl` reds the build, and the whole rest of
    the check has nothing to read.
 2. The row's `units` matches the file. This is the check that catches the case the rule is really
    aimed at: a type or a function added to an EXISTING file, which already has a row.
 3. The row's `map` is a child map that `[map]` lists, and the `swept` flag is a Bool.
 4. A swept row carries `algorithm`, which `test/test_26_docs.jl` holds as a floor.
 5. The file has a coverage row, or it is new and enters under ADR 0082's rule.
 6. A file under `src/` is `include`d by `src/PortfolioOptimisers.jl` exactly once.
 7. The child map is open, and a sub-issue names the path. Both need the tracker.
"""
function check_file(path, rows, map_names, coverage, entry, tracker, exempted, survey)
    fs = Finding[]
    measured = CodeHealth.documented_units(joinpath(CodeHealth.REPO_ROOT, path))
    row = get(rows, path, nothing)

    if row === nothing
        cands = candidate_maps(rows, map_names, path)
        detail = String[]
        if length(cands) == 1
            m = only(cands)
            push!(detail, row_line(path, m, measured, false))
            push!(detail,
                  "map $m: $(map_names[string(m)]) — the only map this directory uses")
        else
            push!(detail, row_line(path, "?", measured, false))
            push!(detail, "The directory offers no single map. Choose by subject:")
            for m in cands
                push!(detail, "  map $m: $(map_names[string(m)])")
            end
        end
        push!(fs,
              Finding(:fail,
                      "no row in `sweep/manifest.toml`. Paste this one, and pick `map`:",
                      detail))
        # Everything below reads the row, so there is nothing more to say about this file until it
        # has one. Saying it anyway would bury the one thing the reader must do first.
        return fs
    end

    if !(row["units"] == measured)
        push!(fs,
              Finding(:fail,
                      "the unit count moved: $(row["units"]) -> $measured. Record it:",
                      [row_line(path, row["map"], measured, row["swept"];
                                algorithm = get(row, "algorithm", nothing)),
                       "A documented unit joined this file, so it joins the file's child map too."]))
    else
        push!(fs, Finding(:ok, "the row is current: $measured unit(s), map $(row["map"])."))
    end

    if !(haskey(map_names, string(row["map"])))
        push!(fs,
              Finding(:fail,
                      "the row names map $(row["map"]), which `[map]` does not list.",
                      String[]))
    end
    if !(row["swept"] isa Bool)
        push!(fs,
              Finding(:fail, "`swept` is not a Bool, so a later gate reads it as true."))
    end
    if row["swept"] === true && !(haskey(row, "algorithm"))
        push!(fs,
              Finding(:fail, "the row reads `swept = true` and carries no `algorithm` key.",
                      ["`test/test_26_docs.jl` holds that count as a floor and demands the key."]))
    elseif row["swept"] === true
        push!(fs,
              Finding(:note,
                      "this file is SWEPT, so the addition meets the swept standard now.",
                      ["`# Algorithm` floor: $(row["algorithm"]). A new unit that carries the",
                       "section raises it, in this commit. No `# Details` section, a `Where:`",
                       "bullet interpolates `math_dict`, and a dispatch alias carries",
                       "`# Related`. `test/test_26_docs.jl` holds all four."]))
    end

    cov = get(coverage, path, nothing)
    if cov === nothing
        push!(fs,
              Finding(:note,
                      "no row in `code_health/coverage_baseline.toml`, so it is new to the gate.",
                      ["ADR 0082: an added file enters with EVERY line covered, or with a named",
                       "Coverage Exemption in `code_health/rulings.toml`."]))
    elseif cov["misses"] > 0 && !(path in exempted)
        push!(fs,
              Finding(:note,
                      "the baseline records $(cov["misses"]) miss(es) of $(cov["lines"]) line(s).",
                      ["The gate ratchets that number. An addition may not raise it."]))
    end

    # The entry file includes every OTHER file under `src/`, so it is the one path with no line of
    # its own to find. `test/test_47_alias_and_module_census.jl` draws the same exception.
    if startswith(path, "src/") && !(path == "src/PortfolioOptimisers.jl")
        n = count(l -> strip(l) == path_include(path), split(entry, '\n'))
        if !(n == 1)
            push!(fs,
                  Finding(:fail,
                          "`src/PortfolioOptimisers.jl` holds $n `include` line(s) for it, not 1.",
                          ["`test/test_47_alias_and_module_census.jl` demands exactly one.",
                           "Add `$(path_include(path))` in the load order the file needs."]))
        end
    end

    append!(fs, check_tracker(path, row, map_names, tracker, survey))
    return fs
end

"""
    path_include(path) -> String

The `include` call a file under `src/` is named by, as `src/PortfolioOptimisers.jl` writes it. The
entry file includes by a path relative to `src/`, so the `src/` prefix is dropped.
"""
function path_include(path::AbstractString)
    return string("include(\"", path[(length("src/") + 1):end], "\")")
end

"""
    check_tracker(path, row, map_names, tracker, survey) -> Vector{Finding}

Steps 3 and 4 of `CLAUDE.md` § *Functionality you add*, which no Julia test can reach: reopen the
child map and the umbrella, and open one sub-issue for the file.

An unread tracker is reported once as a `:note` naming the flag that reads it. A row that already
reads `swept = true` owes no sub-issue, so nothing is reported for it.

**A missing sub-issue is a failure for a file the branch touched, and a note under `--all`.** The
rule of ADR 0084 is about a late addition: a file that joined the library after its child map was
charted. The default scope and `--file` hold exactly such files, because the caller is asserting
that this branch added or changed them. `--all` is a survey of the whole manifest, and there most
unswept files are the standing backlog of an open child map, which the sweeper files as the map
progresses and which owes no ticket of its own. Reporting that backlog as a failure would teach the
reader to ignore the level.

A CLOSED child map is a failure in every scope. That is ADR 0084's own trigger, and it does not
depend on how the file entered the manifest.
"""
function check_tracker(path, row, map_names, tracker, survey)
    if row["swept"] === true
        return Finding[]
    end
    if tracker === nothing
        return [Finding(:note, "the tracker was not read, so steps 3 and 4 are unchecked.",
                        ["Re-run with `--fetch` to read the child map's state and the",
                         "sub-issues that name this path."])]
    end
    m = get(tracker.maps, row["map"], nothing)
    if m === nothing
        return [Finding(:fail,
                        "the tracker holds no issue titled " *
                        "\"Child map $(row["map"]): $(get(map_names, string(row["map"]), "?"))\".",
                        ["`code_health/sweep_triage.jl` throws on this, and files nothing."])]
    end
    issue, isopen, _ = m
    fs = Finding[]
    if !(isopen)
        push!(fs,
              Finding(:fail, "child map #$issue is CLOSED, and this file joins it.",
                      ["Reopen #$issue and #$UMBRELLA, then file the sub-issue."]))
    end
    filed = any(e -> e[1] && CodeHealth.names_path(e[2], path), tracker.issues)
    if !(filed)
        push!(fs,
              Finding(survey ? :note : :fail, "no open `$LABEL` issue names this path.",
                      [if survey
                           "An addition owes one. The standing backlog of an open map does not."
                       else
                           "File it:"
                       end, "  julia --project=code_health code_health/sweep_triage.jl \\",
                       "    --fetch --file $path",
                       "  code_health/sweep_issues.sh apply --dry-run",
                       "  code_health/sweep_issues.sh apply"]))
    else
        push!(fs, Finding(:ok, "an open `$LABEL` issue already names it."))
    end
    return fs
end

# --- the command line ------------------------------------------------------

struct Options
    base::String
    files::Vector{String}
    all::Bool
    fetch::Bool
    out::String
end

const USAGE = """
usage: sweep_check.jl [--base <ref>] [--file <path>]... [--all] [--fetch] [--out <dir>]

  --base <ref>  the ref the branch is measured against. Default: origin/dev.
  --file <path> check this path, whatever the diff says. Repeatable.
  --all         check every file under `src/` and `ext/`.
  --fetch       read the tracker too, through `code_health/sweep_issues.sh dump`.
  --out <dir>   where the dumps are read and written. Default: code_health/_sweep.
"""

function parse_options(args)
    base, all, fetch, out = "origin/dev", false, false, PLAN_DIR
    files = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--all"
            all = true
            i += 1
        elseif a == "--fetch"
            fetch = true
            i += 1
        elseif a in ("--base", "--file", "--out") && i < length(args)
            if a == "--base"
                (base = args[i + 1])
            elseif a == "--file"
                push!(files, args[i + 1])
            else
                (out = args[i + 1])
            end
            i += 2
        else
            error(USAGE)
        end
    end
    return Options(base, files, all, fetch, out)
end

"""
    coverage_exempted(rulings) -> Set{String}

The files that hold at least one Coverage Exemption. The check reports a standing miss count as a
note, and a file whose misses are all exempted has nothing to report at all.
"""
function coverage_exempted(rulings)
    out = Set{String}()
    for e in get(rulings, "coverage_exemption", Any[])
        if haskey(e, "path")
            push!(out, e["path"])
        end
    end
    return out
end

function main(args)
    opts = parse_options(args)

    manifest = CodeHealth.read_toml(MANIFEST_PATH)
    rows = get(manifest, "file", Dict{String, Any}())
    map_names = get(manifest, "map", Dict{String, Any}())
    coverage = get(CodeHealth.read_toml(COVERAGE_PATH), "file", Dict{String, Any}())
    exempted = coverage_exempted(CodeHealth.read_toml(CodeHealth.RULINGS_PATH))
    entry = read(ENTRY_PATH, String)

    # The tracker is read ONLY when this run fetched it. A dump left in the plan directory by an
    # earlier run answers every question with the state of that earlier day, and it answers it
    # silently. A stale answer about an open issue is worse than no answer.
    tracker = nothing
    if opts.fetch
        mkpath(opts.out)
        run(`bash $(joinpath(CodeHealth.DIR, "sweep_issues.sh")) dump --out $(opts.out)`)
        tracker = read_tracker(joinpath(opts.out, "maps.tsv"),
                               joinpath(opts.out, "existing.tsv"))
    end

    scope, ref = if opts.all
        sort(collect(keys(rows))), "the whole manifest"
    elseif !(isempty(opts.files))
        sort(opts.files), "the paths named"
    else
        changed_files(opts.base)
    end
    # A path named with `--file` is taken on trust, but a path that does not exist can only be a
    # typo, and measuring it would throw inside the unit counter with no useful message.
    for f in scope
        if !(isfile(joinpath(CodeHealth.REPO_ROOT, f)))
            error("`$f` is not a file in this checkout.")
        end
    end

    println("The sweep conformance check. ", length(scope), " file(s) in scope, against ",
            ref, ".")
    println("Rules: `CLAUDE.md` § Functionality you add, ADR 0084, ADR 0082.\n")

    if isempty(scope)
        println("No file under `src/` or `ext/` changed, so the sweep is owed nothing.")
        return 0
    end

    failures = 0
    for f in scope
        fs = check_file(f, rows, map_names, coverage, entry, tracker, exempted, opts.all)
        failures += count(x -> x.level === :fail, fs)
        print_findings(stdout, f, fs)
    end

    if failures == 0
        println("Every file in scope meets the four steps this check can measure.")
        if tracker === nothing
            println("The tracker was not read. `--fetch` checks steps 3 and 4.")
        end
        return 0
    end
    println(failures, " duty(ies) unmet. The four steps, from `CLAUDE.md`:\n")
    println("  1. Add or correct the file's row in `sweep/manifest.toml`, `swept = false`.")
    println("  2. Cover every line of a new file, or give it a Coverage Exemption (ADR 0082).")
    println("  3. Reopen the child map that owns the file, and reopen #", UMBRELLA, ".")
    println("  4. Open one sub-issue of that child map for the addition.")
    println("\nSteps 3 and 4:  julia --project=code_health code_health/sweep_triage.jl \\")
    println("                  --fetch --file <path>")
    println("                code_health/sweep_issues.sh apply")
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
