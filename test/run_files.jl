#!/usr/bin/env julia
#=
Run the named test files the way `test/runtests.jl` runs them, without the full suite.

    julia -t 1 --project=test test/run_files.jl test_04_owa.jl test_26_docs.jl
    julia -t 1 --project=test test/run_files.jl --done=<path> test_04_owa.jl
    julia -t 1 --project=test test/run_files.jl --coverage=<absolute dir> test_04_owa.jl
    julia -t 1 --project=test test/run_files.jl --testset='^OWA l-moments$' test_04_owa.jl
    julia -t 1 --project=test test/run_files.jl --time-depth=2 --timeout=900 test_04_owa.jl
    julia -t 1 --project=test test/run_files.jl --ci test_78
    julia -t 1 --project=test test/run_files.jl --case='^bl_' test_78
    julia -t 1 --project=test test/run_files.jl --resume=<done path> --done=<path> test_04 test_26

A file is named as `test_X.jl`, `test/test_X.jl`, a path, or a unique prefix such as `test_78`.
A name that matches no file, or several, is an error that lists the candidates.

Each file runs in a module of its own, after the `init_code` preamble that `test/runtests.jl`
declares, so a file sees what it sees in the suite. A red file does not stop the run. The driver
prints one line per file and one line for the run, and it exits 0 only when every file is green:

    RESULT test_82_test_file_name_census.jl pass=10 fail=0 error=0 broken=0 seconds=6.5
    DONE files=1 pass=10 fail=0 error=0 broken=0 status=green

`--done=<path>` writes the same lines to `<path>` when the run ends, red, green, crashed or timed
out. A background wait on that file wakes the moment the run ends, where a fixed timer guesses.

`--testset=<regex>` runs only the test sets whose name the regex matches (`occursin`), at any
depth, so a fix to one test set is rechecked without the rest of its file. The driver prunes the
syntax tree of each file before it runs it, so the body of a test set that is not selected is never
evaluated:

  - A selected test set runs whole, with every test set inside it.
  - A test set that is not selected, and holds no selected test set, is dropped.
  - A test set that holds a selected one is kept as the path to it: its statements run, so the
    `using` lines, fixtures and helpers that the selected one reads are still there. Its own
    assertions (`@test`, `@test_broken`, `@test_skip`, `@test_throws`) and its other test sets are
    dropped. A `for`, `while`, `let`, `begin`, `if` or `do` block (`with_logger(l) do ... end`)
    that holds a selected test set is kept the same way, and one that holds none is dropped.
  - Every other statement of the file runs: `using`, `include`, definitions, fixtures. A test set
    inside a function or another macro (`@time @testset`) is not reached by the pruning, and runs
    whenever its statement runs.
  - A name is matched as it is written in the source. An interpolated name is matched with its
    `$(...)` in place, so `"risk, $(c)"` is selected by `risk`, and one iteration of a `for` test
    set cannot be selected alone.
  - The pruning reaches the files that a test file `include`s, too.

The `RESULT` line then ends with `selected=<n>`, the number of test sets the regex matched. A file
in which the regex matches nothing is red, with one error, because a run that tested nothing proves
nothing.

`--time-depth=<N>` prints a line as each test set at depth `N` or less finishes, flushed at once, so
a run that is killed halfway still leaves the timings of what finished. The top-level test sets of a
file are at depth 1, and the path joins the names of the nested sets with ` > `. The seconds are the
last field:

    TIME test_04_owa.jl OWA > OWA l-moments 41.27

`--timeout=<seconds>` bounds each file. All the files run in one process, and a solver call cannot
be interrupted from inside it, so the bound is kept by a watchdog: a `bash` process that sleeps for
the bound while the file runs. If the file outlives it, the watchdog prints
`RESULT <file> timeout seconds=<bound>` and `DONE files=<n> status=timeout`, writes the lines of
the files that finished and these two lines to the `--done` file, and kills the driver with
`SIGKILL` (exit status 137). The files after it do not run. The option needs `bash`, and it is
refused from a REPL, whose process the watchdog would kill.

`test/known_red.toml` lists the reds that are known and open as issues. When a listed file is red,
its `RESULT` line names the issue and the reason, so a session sees at once that the red is not its
own:

    RESULT test_04_owa.jl pass=50 fail=1 error=0 broken=0 seconds=315.6 known red, #1709: ...

A failure that no row covers is named after the rows, as `red outside known_red.toml: <path>`. The
verdict does not change: a known red is still red. (CI is different: `test/runtests.jl` passes a run
whose every failure a row covers. `test/known_red.jl` holds the rows for both.) When a row's file
or test set runs and passes,
the driver prints a `STALE` line that asks for the row to be deleted. Under `--testset`, only the
rows whose test set ran are judged.

`--coverage=<absolute dir>` runs the files in a child process under `--code-coverage=@<dir>`,
because Julia writes the counters only when a process exits. The parent then prints one
`COVERAGE` line per source file under `<dir>`, with the lines that no test reached, and deletes the
counter files of the child. The other options are passed to the child. `<dir>` is the source
directory to measure, such as `<worktree>/src/16_RiskMeasures`, because Julia counts only the files
under it. A scratch directory holds no source file, so the run prints no `COVERAGE` line at all.

The run then prints one `PATCH` line for each `.jl` file under `src/` or `ext/` that the working
tree changes against its merge-base with `origin/dev`, with the added lines that CI reads as a miss.
CI also reads as a miss each line of a method that no test compiled, which the `COVERAGE` line
cannot show, so a `PATCH` line counts those too (#1912). When every file ran whole and green, every
changed file lies under `<dir>`, and no added line is a miss, the run records the `coverage` stamp
that the push guard asks of a change to the code. Use the narrowest directory that holds every
changed file, such as `<worktree>/src`.

`--ci` runs the files in a child process under the flags of `Pkg.test` on CI,
`--check-bounds=yes --code-coverage=user`. A test that pins solver output can pass locally and fail
on CI only, because the two flags change the compiled code. The coverage goes to one scratch
tracefile, which the parent deletes, so no `.cov` file lands next to a source file.

`--case=<regex>` sets `ENV["RUN_FILES_CASE"]` for the run. A file whose test sets come from one
`for` loop over named cases, whose one interpolated name `--testset` cannot select, filters its
cases by it (`test/test_78_brt_pinned_outputs.jl` does).

`--resume=<path>` skips each named file that already has a `RESULT` line in `<path>`, the `--done`
file or the log of an earlier run that stopped halfway, for example when its session ended.

Each file also records a gate stamp through `code_health/gate_stamp.sh`, which the push guard of
`.pre-commit-config.yaml` reads. A file run under `--testset` or `--case` records none: it ran only
a part.

A run from the command line holds a slot of `code_health/julia_slot.jl` while it runs, so it waits
for free memory rather than being killed for the lack of it. A `SLOT wait` line names the holders.

From a REPL, `include` this file and call `RunFiles.main(["test_04_owa.jl"])`. It returns the
exit code. Pass `--done` there too: a REPL that strips `println` still writes the file.

This file is not a test: `test/runtests.jl` discovers only `test_*.jl`.
=#
module RunFiles

using Test, LinearAlgebra

const TEST_DIR = @__DIR__
const REPO_ROOT = dirname(TEST_DIR)

include(joinpath(REPO_ROOT, "code_health", "julia_slot.jl"))
# `test/runtests.jl` reads the same rows, to filter the failures of CI (#1781).
include(joinpath(TEST_DIR, "known_red.jl"))
using .KnownRedRows: KNOWN_RED, KnownRed, known_red, covers, row_name

"""
Where [`Collect`](@ref) prints a failure and a `TIME` line. A test of the driver points it at
`devnull`, so the failures of its fixture do not read as failures of the suite.
"""
const OUT = Ref{IO}(stdout)

"""
The deepest test set whose time [`Collect`](@ref) prints, `0` for none. [`run_file`](@ref) sets it
for the length of one file.
"""
const TIME_DEPTH = Ref(0)

"""
Whether the driver runs as the program of its process, the one process `--timeout` may kill. The
command line sets it. A REPL, and a test that includes this file, leave it `false`.
"""
const OWNS_PROCESS = Ref(false)

"""
    init_code(path = joinpath(TEST_DIR, "runtests.jl")) -> Expr

The `init_code` preamble of `test/runtests.jl`, parsed out of its text. Including that file would
start the full suite, because it ends with the call that runs every test.
"""
function init_code(path::AbstractString = joinpath(TEST_DIR, "runtests.jl"))
    s = read(path, String)
    head = "const init_code = "
    i = findfirst(head * "quote", s)
    if i === nothing
        error("`$path` declares no `const init_code = quote ... end`.")
    end
    ex = Meta.parse(s, first(i) + length(head); greedy = true, raise = true)[1]
    # `.args[1]` unwraps the `quote`. The quote itself evaluates to the block and runs nothing.
    return ex.args[1]
end

"""
    Collect <: Test.AbstractTestSet

A test set that counts and keeps going. A nested test set inherits the type of its parent, so the
counts of a whole file reach the top. Only the top set, made with `top = true`, keeps its counts
to itself: inside another test set, such as a worker of the suite, the driver's verdict must not
leak into the parent.

A failure and an error are printed where they happen, as the default test set prints them. Each
set also keeps the paths of the sets that ran and of the sets that failed, for the rows of
`test/known_red.toml`, and prints its time when it is no deeper than [`TIME_DEPTH`](@ref).
"""
mutable struct Collect <: Test.AbstractTestSet
    description::String
    top::Bool
    pass::Int
    fail::Int
    error::Int
    broken::Int
    file::String
    path::String
    depth::Int
    start::UInt64
    ran::Vector{String}
    failed::Vector{String}
end

function Collect(description::AbstractString; top::Bool = false, kwargs...)
    parent = Test.get_testset_depth() > 0 ? Test.get_testset() : nothing
    file, path, depth = if !(top) && parent isa Collect
        p = isempty(parent.path) ? description : parent.path * " > " * description
        parent.file, p, parent.depth + 1
    else
        description, "", 0
    end
    return Collect(description, top, 0, 0, 0, 0, file, path, depth, time_ns(), String[],
                   String[])
end

# `@test_throws` returns what `record` returns, as `Test.DefaultTestSet` does, so a test can read
# the exception off it.
function Test.record(ts::Collect, r::Test.Pass)
    ts.pass += 1
    return r
end
function Test.record(ts::Collect, r::Test.Broken)
    ts.broken += 1
    return r
end
# A failed `@test_logs` records a `Test.LogTestFailure`, which is a failed test too.
function Test.record(ts::Collect, r::Union{Test.Fail, Test.Error, Test.LogTestFailure})
    if r isa Union{Test.Fail, Test.LogTestFailure}
        ts.fail += 1
    else
        ts.error += 1
    end
    push!(ts.failed, ts.path)
    println(OUT[], ts.description, ": ", r)
    return r
end
function Test.record(parent::Collect, child::Collect)
    parent.pass += child.pass
    parent.fail += child.fail
    parent.error += child.error
    parent.broken += child.broken
    append!(parent.ran, child.ran)
    append!(parent.failed, child.failed)
    return nothing
end
function Test.finish(ts::Collect)
    if ts.depth > 0
        push!(ts.ran, ts.path)
        if ts.depth <= TIME_DEPTH[]
            seconds = round((time_ns() - ts.start) / 1e9; digits = 2)
            println(OUT[], "TIME ", ts.file, " ", ts.path, " ", seconds)
            flush(OUT[])
        end
    end
    if !(ts.top) && Test.get_testset_depth() > 0
        Test.record(Test.get_testset(), ts)
    end
    return ts
end

green(ts::Collect) = ts.fail == 0 && ts.error == 0

function result_line(ts::Collect, seconds::Real;
                     selected::Union{Nothing, Integer} = nothing, note::AbstractString = "")
    line = "RESULT $(ts.description) pass=$(ts.pass) fail=$(ts.fail) error=$(ts.error) " *
           "broken=$(ts.broken) seconds=$(round(seconds; digits = 1))"
    if selected !== nothing
        line *= " selected=$selected"
    end
    return isempty(note) ? line : line * " " * note
end

function done_line(sets::Vector{Collect})
    total(f) = sum(f, sets; init = 0)
    status = all(green, sets) ? "green" : "red"
    return "DONE files=$(length(sets)) pass=$(total(t -> t.pass)) fail=$(total(t -> t.fail)) " *
           "error=$(total(t -> t.error)) broken=$(total(t -> t.broken)) status=$status"
end

"""
    resolve(name) -> String

The absolute path of a test file named as `test_X.jl`, `test/test_X.jl`, a path, or a prefix of
the name of exactly one test file, such as `test_78`. A miss lists the files that share the number
of the name.
"""
function resolve(name::AbstractString; dir::AbstractString = TEST_DIR)
    for p in (joinpath(dir, basename(name)), abspath(name))
        if isfile(p)
            return p
        end
    end
    files = filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(dir))
    hits = filter(f -> startswith(f, basename(name)), files)
    if length(hits) == 1
        return joinpath(dir, only(hits))
    end
    m = match(r"^(test_\d+)", basename(name))
    near = isempty(hits) && m !== nothing ? filter(f -> startswith(f, m[1]), files) : hits
    hint = isempty(near) ? "" : " Candidates: $(join(near, ", "))."
    return error("No single test file `$name` under `$dir`.$hint")
end

# ---------------------------------------------------------------------------------------------
# `--testset`: prune the syntax tree of a file down to the selected test sets.
# ---------------------------------------------------------------------------------------------

"""
    Select(re)

The test sets that `--testset` selects, and a count of the selected ones that a file reached.
"""
struct Select
    re::Regex
    matched::Base.RefValue{Int}
end
Select(re::Regex) = Select(re, Ref(0))

const TESTSET = Symbol("@testset")
const ASSERTIONS = Symbol.(["@test", "@test_broken", "@test_skip", "@test_throws"])

"""
    macro_name(ex) -> Union{Symbol, Nothing}

The name of the macro that `ex` calls, written bare or as `Test.@name`, or `nothing`.
"""
function macro_name(ex)
    if !(ex isa Expr && ex.head === :macrocall)
        return nothing
    end
    m = ex.args[1]
    if m isa Symbol
        return m
    elseif m isa Expr && m.head === :. && last(m.args) isa QuoteNode
        return last(m.args).value
    end
    return nothing
end

is_testset(ex) = macro_name(ex) === TESTSET
is_assertion(ex) = macro_name(ex) in ASSERTIONS

"""
    testset_label(ex) -> String

The name of a `@testset` call as its source writes it. An interpolation keeps its `\$(...)`, and a
set with no name is `"test set"`, the name `Test` gives it.
"""
function testset_label(ex::Expr)
    for a in ex.args[2:(end - 1)]
        if a isa String
            return a
        elseif a isa Expr && a.head === :string
            return join(p isa String ? p : "\$(" * string(p) * ")" for p in a.args)
        end
    end
    return "test set"
end

# The heads that the pruning descends into, with the body of a `do` block. A function body, an
# anonymous function and a macro other than `@testset` are statements: they run whole.
const DESCEND = (:block, :toplevel, :for, :while, :let, :if, :elseif)

"""
    reaches_testset(ex) -> Bool

Whether `ex` is a test set, or holds one along the heads the pruning descends into.
"""
function reaches_testset(ex)
    if is_testset(ex)
        return true
    elseif is_do_block(ex)
        return reaches_testset(do_body(ex))
    end
    return ex isa Expr && ex.head in DESCEND && any(reaches_testset, ex.args)
end

# A `do` block, `f(args) do x ... end`, is `Expr(:do, call, Expr(:->, params, body))`. Its body
# runs when the call runs, as the body of `with_logger(logger) do ... end` does.
function is_do_block(ex)
    return ex isa Expr &&
           ex.head === :do &&
           length(ex.args) == 2 &&
           ex.args[2] isa Expr &&
           ex.args[2].head === :->
end
do_body(ex::Expr) = ex.args[2].args[2]

"""
    prune(sel, ex) -> Union{Expr, Nothing}

`ex`, a statement that reaches a test set, cut down to the test sets that `sel` selects and the
statements on the path to them, or `nothing` when it holds no selected test set.
"""
function prune(sel::Select, ex::Expr)
    if is_testset(ex)
        if occursin(sel.re, testset_label(ex))
            sel.matched[] += 1
            return ex
        end
        body = prune(sel, ex.args[end])
        return body === nothing ? nothing : Expr(:macrocall, ex.args[1:(end - 1)]..., body)
    elseif ex.head in (:block, :toplevel)
        kept, any_selected = Any[], false
        for a in ex.args
            if reaches_testset(a)
                p = prune(sel, a)
                if p !== nothing
                    any_selected = true
                    push!(kept, p)
                end
            elseif !(is_assertion(a))
                push!(kept, a)
            end
        end
        return any_selected ? Expr(ex.head, kept...) : nothing
    elseif is_do_block(ex)
        body = prune(sel, do_body(ex))
        if body === nothing
            return nothing
        end
        return Expr(:do, ex.args[1], Expr(:->, ex.args[2].args[1], body))
    elseif ex.head in (:for, :while, :let)
        # The first argument is the iteration, the condition or the bindings, the second the body.
        body = ex.args[2] isa Expr ? prune(sel, ex.args[2]) : nothing
        return body === nothing ? nothing : Expr(ex.head, ex.args[1], body)
    elseif ex.head in (:if, :elseif)
        branches = map(ex.args[2:end]) do b
            return reaches_testset(b) ? prune(sel, b) : nothing
        end
        if all(isnothing, branches)
            return nothing
        end
        # A branch without a selected test set runs nothing.
        return Expr(ex.head, ex.args[1], (something(b, Expr(:block)) for b in branches)...)
    end
    return ex
end

"""
    prune_toplevel(sel, ex)

The `mapexpr` that `Base.include` applies to each top-level statement of a file under
`--testset`. A statement that reaches no test set runs as it is, unless it is an assertion.
"""
function prune_toplevel(sel::Select, ex)
    if reaches_testset(ex)
        return prune(sel, ex)
    end
    return is_assertion(ex) ? nothing : ex
end

"""
    run_file(path; code = init_code(), select = nothing, time_depth = 0) -> Collect, Float64

Run one test file in a module of its own, after `code`, from the test directory, as the suite
runs it. The second value is the wall-clock seconds. With a [`Select`](@ref), the file and the
files it includes run pruned to the selected test sets, and a file that selects none is red.
"""
function run_file(path::AbstractString; code::Expr = init_code(),
                  select::Union{Nothing, Select} = nothing, time_depth::Integer = 0)
    m = Module(Symbol(splitext(basename(path))[1]))
    mapexpr = select === nothing ? identity : (ex -> prune_toplevel(select, ex))
    Core.eval(m, :(include(x) = Base.include($mapexpr, $m, x)))
    local ts
    old_depth = TIME_DEPTH[]
    TIME_DEPTH[] = time_depth
    try
        seconds = @elapsed cd(TEST_DIR) do
            ts = @testset Collect top=true "$(basename(path))" begin
                Core.eval(m, code)
                Base.include(mapexpr, m, path)
            end
        end
        if select !== nothing && select.matched[] == 0
            ts.error += 1
            push!(ts.failed, "")
            println(OUT[], ts.description, ": no test set matches `", select.re.pattern,
                    "`.")
        end
        return ts, seconds
    finally
        TIME_DEPTH[] = old_depth
    end
end

"""
    record_gate(path)

Record a gate stamp for one test file. A failure to write it never fails the run.
"""
record_gate(path::AbstractString) = record_stamp("test/$(basename(path))")

# ---------------------------------------------------------------------------------------------
# `test/known_red.toml`: the reds that are known and open as issues.
# ---------------------------------------------------------------------------------------------

"""
    known_red_notes(ts, rows; filtered = false) -> (note, stale)

The note that the `RESULT` line of `ts` ends with, and one `STALE` line for each row of the file
whose scope ran and passed. A row names a test set by a substring of its path. Under `--testset`
(`filtered`), a row of a whole file is not judged, because the run did not run the whole file.
"""
function known_red_notes(ts::Collect, rows::Vector{KnownRed}; filtered::Bool = false)
    hits, stale = KnownRed[], String[]
    for r in rows
        if r.file != ts.description
            continue
        end
        ran = r.testset === nothing ? !(filtered) : any(p -> covers(r, p), ts.ran)
        if !(ran)
            continue
        end
        if any(p -> covers(r, p), ts.failed)
            push!(hits, r)
        else
            host = r.host === nothing ? "" : " The row's host note: $(r.host)."
            push!(stale,
                  "STALE test/known_red.toml: $(row_name(r)) (#$(r.issue)) passed. Delete the " *
                  "row if #$(r.issue) is fixed.$host")
        end
    end
    if isempty(hits)
        return "", stale
    end
    parts = map(hits) do r
        return "#$(r.issue): $(r.reason)" * (r.host === nothing ? "" : " ($(r.host))")
    end
    note = "known red, " * join(parts, "; ")
    other = unique(filter(p -> !any(r -> covers(r, p), hits), ts.failed))
    if !(isempty(other))
        note *= "; red outside known_red.toml: " *
                join((isempty(p) ? "(file level)" : p for p in other), ", ")
    end
    return note, stale
end

# ---------------------------------------------------------------------------------------------
# `--timeout`: a watchdog process per file.
# ---------------------------------------------------------------------------------------------

# Arguments: the bound, the RESULT line, the DONE line, the done path ("" for none), the file of
# the lines so far, the PID to kill. A TERM from the driver, when the file ends in time, stops the
# sleep and the watchdog without a trace.
const WATCHDOG = raw"""
trap 'kill "$s" 2>/dev/null; exit 0' TERM
sleep "$1" &
s=$!
wait "$s"
printf '%s\n%s\n' "$2" "$3"
if [ -n "$4" ]; then
    { cat "$5"; printf '%s\n%s\n' "$2" "$3"; } > "$4.tmp" && mv "$4.tmp" "$4"
    rm -f "$5"
fi
kill -9 "$6"
"""

"""
    start_watchdog(seconds, file, lines, done; pid = getpid(), out = stdout) -> Base.Process

A `bash` process that, unless it is killed first, prints the timeout lines of `file` to `out`
after `seconds`, writes them after `lines` to `done`, and kills `pid`.
"""
function start_watchdog(seconds::Real, file::AbstractString, lines::Vector{String},
                        done::Union{Nothing, AbstractString}; pid::Integer = getpid(),
                        out = stdout)
    partial = done === nothing ? "" : done * ".partial"
    if done !== nothing
        write(partial, join((l * '\n' for l in lines)))
    end
    nfiles = count(startswith("RESULT"), lines) + 1
    result = "RESULT $file timeout seconds=$seconds"
    final = "DONE files=$nfiles status=timeout"
    cmd = `bash -c $WATCHDOG watchdog $seconds $result $final $(something(done, "")) $partial $pid`
    return run(pipeline(cmd; stdout = out); wait = false)
end

function stop_watchdog(p::Base.Process, done::Union{Nothing, AbstractString})
    kill(p)
    wait(p)
    if done !== nothing
        rm(done * ".partial"; force = true)
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------
# `--coverage` and the command line.
# ---------------------------------------------------------------------------------------------

"""
    coverage_lines(cov_path) -> (counted, missed)

The `.cov` file of one source file. Each line opens with a count nine characters wide and a
space. `-` marks a line with no counter, `0` a line no test reached.
"""
function coverage_lines(cov_path::AbstractString)
    tags = coverage_tags(cov_path)
    return count(!isnothing, tags), findall(==(0), tags)
end

"""
    coverage_tags(cov_path) -> Vector{Union{Nothing, Int}}

The count of each line of a `.cov` file, or `nothing` for a line with no counter. A count wider
than its nine characters is read as a hit.
"""
function coverage_tags(cov_path::AbstractString)
    tags = Union{Nothing, Int}[]
    for line in eachline(cov_path)
        tag = strip(first(line, 9))
        push!(tags, tag == "-" || isempty(tag) ? nothing : something(tryparse(Int, tag), 1))
    end
    return tags
end

"""
    is_method_definition(e) -> Bool

Whether `e` defines a method: a `function` block, or a short form `f(x) = …`, also under `where`
or a return type.
"""
function is_method_definition(e::Expr)
    if e.head === :function
        return true
    elseif e.head === :(=)
        lhs = e.args[1]
        while lhs isa Expr && lhs.head in (:where, :(::))
            lhs = lhs.args[1]
        end
        return lhs isa Expr && lhs.head === :call
    end
    return false
end

function body_lines!(out::Set{Int}, e, inside::Bool)
    if e isa LineNumberNode
        inside && push!(out, e.line)
    elseif e isa Expr
        inside = inside || is_method_definition(e)
        for a in e.args
            body_lines!(out, a, inside)
        end
    end
    return out
end

"""
    function_body_lines(text) -> Set{Int}

The lines of `text` that start a statement inside a method body. CI processes the `.cov` files with
`CoverageTools.amend_coverage_from_src!`, which reads such a line as a miss when it has no counter.
A method that no test compiled has no counter on any line, so CI counts every line of it as a miss,
while the raw `.cov` file shows none (#1912).
"""
function function_body_lines(text::AbstractString)
    return body_lines!(Set{Int}(), Meta.parseall(text), false)
end

"""
    patch_misses(tags, body, added) -> Vector{Int}

The lines of `added` that CI reads as a miss: a counter at zero, or a body line with no counter.
`tags` comes from [`coverage_tags`](@ref), or is empty when no test loaded the file.
"""
function patch_misses(tags::AbstractVector, body::Set{Int}, added)
    out = Int[]
    for n in sort!(collect(added))
        t = n <= length(tags) ? tags[n] : nothing
        if t == 0 || (isnothing(t) && n in body)
            push!(out, n)
        end
    end
    return out
end

"""
    added_lines(diff) -> Set{Int}

The line numbers that a `git diff -U0` adds to the new side.
"""
function added_lines(diff::AbstractString)
    out = Set{Int}()
    for line in eachline(IOBuffer(diff))
        m = match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m === nothing
            continue
        end
        start = parse(Int, m[1])
        n = m[2] === nothing ? 1 : parse(Int, m[2])
        union!(out, start:(start + n - 1))
    end
    return out
end

git_text(args...) = readchomp(Cmd(`git $args`; dir = REPO_ROOT))

"""
    patch_report(dir, tags) -> (lines, clean)

One `PATCH` line for each `.jl` file under `src/` or `ext/` that the working tree changes against
its merge-base with `origin/dev` (`GATE_STAMP_REF`), with the added lines that CI reads as a miss.
`tags` maps a path relative to the repository to its [`coverage_tags`](@ref). `clean` is true when
every changed file lies under `dir` and no added line is a miss.
"""
function patch_report(dir::AbstractString, tags::AbstractDict)
    ref = get(ENV, "GATE_STAMP_REF", "origin/dev")
    base = try
        git_text("merge-base", ref, "HEAD")
    catch
        return ["PATCH skipped: no merge-base with $ref"], false
    end
    changed = vcat(split(git_text("diff", "--name-only", "--no-renames", base, "--", "src",
                                  "ext")),
                   split(git_text("ls-files", "--others", "--exclude-standard", "src",
                                  "ext")))
    untracked = Set(split(git_text("ls-files", "--others", "--exclude-standard", "src",
                                   "ext")))
    lines, clean = String[], true
    for rel in sort!(unique(filter(f -> endswith(f, ".jl"), changed)))
        full = joinpath(REPO_ROOT, rel)
        if !(isfile(full))
            continue
        end
        if !startswith(full, rstrip(dir, '/') * "/")
            push!(lines, "PATCH $rel not measured: outside $dir")
            clean = false
            continue
        end
        text = read(full, String)
        added = if rel in untracked
            Set(1:countlines(IOBuffer(text)))
        else
            added_lines(git_text("diff", "-U0", "--no-color", "--no-ext-diff", base, "--",
                                 rel))
        end
        missed = patch_misses(get(tags, rel, Union{Nothing, Int}[]),
                              function_body_lines(text), added)
        push!(lines, "PATCH $rel added=$(length(added)) missed=$(length(missed))")
        if !isempty(missed)
            push!(lines, "  missed lines: " * join(missed, ", "))
            clean = false
        end
    end
    return lines, clean
end

"""
    record_stamp(gate)

Record a stamp of `gate` through `code_health/gate_stamp.sh`. A failure never fails the run.
"""
function record_stamp(gate::AbstractString)
    script = joinpath(REPO_ROOT, "code_health", "gate_stamp.sh")
    try
        run(pipeline(Cmd(`bash $script record $gate`; dir = REPO_ROOT); stdout = devnull,
                     stderr = devnull))
    catch
    end
    return nothing
end

"""
    run_coverage(dir, rest) -> Int

Run the driver again in a child process under `--code-coverage=@<dir>`, then report the counter
files that child wrote and delete them. Then print the [`patch_report`](@ref) of the change, and
record a `coverage` stamp when every file ran whole and green and the report is clean.
"""
function run_coverage(dir::AbstractString, rest::Vector{String})
    if !isabspath(dir)
        error("`--coverage` needs an absolute directory. A relative one instruments nothing. Got $dir")
    elseif !isdir(dir)
        error("`--coverage` needs a source directory that exists, such as <worktree>/src. Got $dir")
    end
    cmd = `$(Base.julia_cmd()) -t 1 --project=$(Base.active_project()) --code-coverage=@$dir $(@__FILE__) $rest`
    p = run(ignorestatus(cmd); wait = false)
    # A process that has exited has no PID to read, so read it before the wait.
    pid = getpid(p)
    wait(p)
    tags = Dict{String, Vector{Union{Nothing, Int}}}()
    for (root, _, files) in walkdir(dir)
        for f in sort(files)
            m = match(r"^(.+\.jl)\.(\d+)\.cov$", f)
            if m === nothing || parse(Int, m[2]) != pid
                continue
            end
            cov = joinpath(root, f)
            counted, missed = coverage_lines(cov)
            rel = relpath(joinpath(root, m[1]), REPO_ROOT)
            println("COVERAGE $rel counted=$counted missed=$(length(missed))")
            if !(isempty(missed))
                println("  missed lines: ", join(missed, ", "))
            end
            tags[rel] = coverage_tags(cov)
            rm(cov)
        end
    end
    lines, clean = patch_report(dir, tags)
    foreach(println, lines)
    whole = !any(a -> startswith(a, "--testset=") || startswith(a, "--case="), rest)
    if success(p) && whole && clean
        record_stamp("coverage")
        println("PATCH clean: recorded the `coverage` stamp.")
    else
        println("PATCH not clean, or a file was red or ran in part: no `coverage` stamp.")
    end
    # A child killed by its watchdog has no exit code, only a signal.
    return success(p) ? 0 : max(p.exitcode, 1)
end

const USAGE = "usage: run_files.jl [--done=<path>] [--coverage=<absolute dir>] [--ci] " *
              "[--testset=<regex>] [--case=<regex>] [--resume=<path>] [--time-depth=<N>] " *
              "[--timeout=<seconds>] test_X.jl..."

"""
    parse_args(args) -> NamedTuple

The options and the file names of a command line. An unknown or malformed option is an error that
prints [`USAGE`](@ref).
"""
function parse_args(args::Vector{String})
    done, coverage, testset, time_depth, timeout = nothing, nothing, nothing, 0, nothing
    ci, case, resume = false, nothing, nothing
    names = String[]
    value(a, key) = a[(length(key) + 1):end]
    for a in args
        if startswith(a, "--done=")
            done = value(a, "--done=")
        elseif startswith(a, "--coverage=")
            coverage = value(a, "--coverage=")
        elseif startswith(a, "--testset=")
            testset = Regex(value(a, "--testset="))
        elseif startswith(a, "--case=")
            case = value(a, "--case=")
            # A malformed pattern fails here, before a file loads.
            Regex(case)
        elseif startswith(a, "--resume=")
            resume = value(a, "--resume=")
        elseif a == "--ci"
            ci = true
        elseif startswith(a, "--time-depth=")
            time_depth = something(tryparse(Int, value(a, "--time-depth=")), -1)
            if time_depth < 0
                error("`--time-depth` takes a depth of 0 or more.\n" * USAGE)
            end
        elseif startswith(a, "--timeout=")
            timeout = something(tryparse(Float64, value(a, "--timeout=")), -1.0)
            if !(timeout > 0)
                error("`--timeout` takes a number of seconds above 0.\n" * USAGE)
            end
        elseif startswith(a, "--")
            error(USAGE)
        else
            push!(names, a)
        end
    end
    if isempty(names)
        error(USAGE)
    end
    return (; done, coverage, testset, time_depth, timeout, names, ci, case, resume)
end

"""
    finished_files(path) -> Set{String}

The files that have a `RESULT` line in the log or `--done` file `path`.
"""
function finished_files(path::AbstractString)
    out = Set{String}()
    if !(isfile(path))
        return out
    end
    for line in eachline(path)
        m = match(r"^RESULT (\S+) ", line)
        m === nothing || push!(out, m[1])
    end
    return out
end

"""
    run_ci(rest) -> Int

Run the driver again in a child process under `--check-bounds=yes --code-coverage=user`, the
flags of `Pkg.test` on CI, with the coverage sent to a scratch tracefile that is deleted after.
"""
function run_ci(rest::Vector{String})
    dir = mktempdir()
    try
        cmd = `$(Base.julia_cmd()) -t 1 --project=$(Base.active_project()) --check-bounds=yes --code-coverage=user --code-coverage=$(joinpath(dir, "trace-%p.info")) $(@__FILE__) $rest`
        p = run(ignorestatus(cmd))
        return success(p) ? 0 : max(p.exitcode, 1)
    finally
        rm(dir; recursive = true, force = true)
    end
end

function main(args::Vector{String})
    (; done, coverage, testset, time_depth, timeout, names, ci, case, resume) = parse_args(args)
    if coverage !== nothing
        rest = filter(a -> !startswith(a, "--coverage="), args)
        return run_coverage(coverage, rest)
    end
    if ci
        return run_ci(filter(!=("--ci"), args))
    end
    if !(OWNS_PROCESS[])
        return run_files(; done, testset, time_depth, timeout, names, case, resume)
    end
    return JuliaSlot.with_slot("run_files.jl $(join(names, ' '))") do
        return run_files(; done, testset, time_depth, timeout, names, case, resume)
    end
end

function run_files(; done, testset, time_depth, timeout, names, case = nothing,
                   resume = nothing)
    if timeout !== nothing && !(OWNS_PROCESS[])
        error("`--timeout` kills the process that runs the driver, so it runs only from the " *
              "command line, never from a REPL.")
    end
    BLAS.set_num_threads(1)
    lines = String[]
    sets = Collect[]
    status = 1
    old_case = get(ENV, "RUN_FILES_CASE", nothing)
    if case !== nothing
        ENV["RUN_FILES_CASE"] = case
    end
    try
        paths = resolve.(names)
        if resume !== nothing
            skip = finished_files(resume)
            for p in filter(p -> basename(p) in skip, paths)
                println("SKIP $(basename(p)): it has a RESULT line in $resume")
            end
            filter!(p -> !(basename(p) in skip), paths)
        end
        rows = known_red()
        code = init_code()
        for p in paths
            select = testset === nothing ? nothing : Select(testset)
            dog = if timeout === nothing
                nothing
            else
                start_watchdog(timeout, basename(p), lines, done)
            end
            ts, seconds = try
                run_file(p; code, select, time_depth)
            finally
                if dog !== nothing
                    stop_watchdog(dog, done)
                end
            end
            if select === nothing && case === nothing
                record_gate(p)
            end
            note, stale = known_red_notes(ts, rows; filtered = select !== nothing)
            push!(sets, ts)
            selected = select === nothing ? nothing : select.matched[]
            push!(lines, result_line(ts, seconds; selected, note))
            println(last(lines))
            for s in stale
                push!(lines, s)
                println(s)
            end
            flush(stdout)
        end
        push!(lines, done_line(sets))
        println(last(lines))
        status = all(green, sets) ? 0 : 1
    finally
        if case !== nothing && old_case === nothing
            delete!(ENV, "RUN_FILES_CASE")
        elseif case !== nothing
            ENV["RUN_FILES_CASE"] = old_case
        end
        if done !== nothing
            if status == 1 && !(any(startswith("DONE"), lines))
                push!(lines, "DONE files=$(length(sets)) status=crashed")
            end
            write(done, join(lines, '\n') * '\n')
        end
    end
    return status
end

end # module

# `@__MODULE__` keeps a test that includes this file from running the driver again: when the driver
# runs that test, `PROGRAM_FILE` is still this file.
if @__MODULE__() === Main && abspath(PROGRAM_FILE) == @__FILE__
    RunFiles.OWNS_PROCESS[] = true
    exit(RunFiles.main(ARGS))
end
