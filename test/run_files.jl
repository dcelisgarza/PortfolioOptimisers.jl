#!/usr/bin/env julia
#=
Run the named test files the way `test/runtests.jl` runs them, without the full suite.

    julia -t 1 --project=test test/run_files.jl test_12k_parity.jl test_26_docs.jl
    julia -t 1 --project=test test/run_files.jl --done=<path> test_12k_parity.jl
    julia -t 1 --project=test test/run_files.jl --coverage=<absolute dir> test_12k_parity.jl

Each file runs in a module of its own, after the `init_code` preamble that `test/runtests.jl`
declares, so a file sees what it sees in the suite. A red file does not stop the run. The driver
prints one line per file and one line for the run, and it exits 0 only when every file is green:

    RESULT test_12k_parity.jl pass=80 fail=0 error=0 broken=0 seconds=128.4
    DONE files=1 pass=80 fail=0 error=0 broken=0 status=green

`--done=<path>` writes the same lines to `<path>` when the run ends, red, green or crashed. A
background wait on that file wakes the moment the run ends, where a fixed timer guesses.

`--coverage=<absolute dir>` runs the files in a child process under `--code-coverage=@<dir>`,
because Julia writes the counters only when a process exits. The parent then prints one
`COVERAGE` line per source file under `<dir>`, with the lines that no test reached, and deletes the
counter files of the child.

Each file also records a gate stamp through `code_health/gate_stamp.sh`, which the push guard of
`.pre-commit-config.yaml` reads.

From a REPL, `include` this file and call `RunFiles.main(["test_12k_parity.jl"])`. It returns the
exit code. Pass `--done` there too: a REPL that strips `println` still writes the file.

This file is not a test: `test/runtests.jl` discovers only `test_*.jl`.
=#
module RunFiles

using Test, LinearAlgebra

const TEST_DIR = @__DIR__
const REPO_ROOT = dirname(TEST_DIR)

"""
Where [`Collect`](@ref) prints a failure. A test of the driver points it at `devnull`, so the
failures of its fixture do not read as failures of the suite.
"""
const OUT = Ref{IO}(stdout)

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

A failure and an error are printed where they happen, as the default test set prints them.
"""
mutable struct Collect <: Test.AbstractTestSet
    description::String
    top::Bool
    pass::Int
    fail::Int
    error::Int
    broken::Int
end

function Collect(description::AbstractString; top::Bool = false, kwargs...)
    return Collect(description, top, 0, 0, 0, 0)
end

function Test.record(ts::Collect, ::Test.Pass)
    ts.pass += 1
    return nothing
end
function Test.record(ts::Collect, ::Test.Broken)
    ts.broken += 1
    return nothing
end
function Test.record(ts::Collect, r::Union{Test.Fail, Test.Error})
    if r isa Test.Fail
        ts.fail += 1
    else
        ts.error += 1
    end
    println(OUT[], ts.description, ": ", r)
    return nothing
end
function Test.record(parent::Collect, child::Collect)
    parent.pass += child.pass
    parent.fail += child.fail
    parent.error += child.error
    parent.broken += child.broken
    return nothing
end
function Test.finish(ts::Collect)
    if !(ts.top) && Test.get_testset_depth() > 0
        Test.record(Test.get_testset(), ts)
    end
    return ts
end

green(ts::Collect) = ts.fail == 0 && ts.error == 0

function result_line(ts::Collect, seconds::Real)
    return "RESULT $(ts.description) pass=$(ts.pass) fail=$(ts.fail) error=$(ts.error) " *
           "broken=$(ts.broken) seconds=$(round(seconds; digits = 1))"
end

function done_line(sets::Vector{Collect})
    total(f) = sum(f, sets; init = 0)
    status = all(green, sets) ? "green" : "red"
    return "DONE files=$(length(sets)) pass=$(total(t -> t.pass)) fail=$(total(t -> t.fail)) " *
           "error=$(total(t -> t.error)) broken=$(total(t -> t.broken)) status=$status"
end

"""
    resolve(name) -> String

The absolute path of a test file named as `test_X.jl`, `test/test_X.jl` or a path.
"""
function resolve(name::AbstractString)
    for p in (joinpath(TEST_DIR, basename(name)), abspath(name))
        if isfile(p)
            return p
        end
    end
    return error("No test file `$name` under `$TEST_DIR`.")
end

"""
    run_file(path; code = init_code()) -> Collect, Float64

Run one test file in a module of its own, after `code`, from the test directory, as the suite
runs it. The second value is the wall-clock seconds.
"""
function run_file(path::AbstractString; code::Expr = init_code())
    m = Module(Symbol(splitext(basename(path))[1]))
    Core.eval(m, :(include(x) = Base.include($m, x)))
    local ts
    seconds = @elapsed cd(TEST_DIR) do
        ts = @testset Collect top=true "$(basename(path))" begin
            Core.eval(m, code)
            Base.include(m, path)
        end
    end
    return ts, seconds
end

"""
    record_gate(path)

Record a gate stamp for one test file. A failure to write it never fails the run.
"""
function record_gate(path::AbstractString)
    script = joinpath(REPO_ROOT, "code_health", "gate_stamp.sh")
    try
        cmd = Cmd(`bash $script record test/$(basename(path))`; dir = REPO_ROOT)
        run(pipeline(cmd; stdout = devnull, stderr = devnull))
    catch
    end
    return nothing
end

"""
    coverage_lines(cov_path) -> (counted, missed)

The `.cov` file of one source file. Each line opens with a count nine characters wide and a
space. `-` marks a line with no counter, `0` a line no test reached.
"""
function coverage_lines(cov_path::AbstractString)
    counted, missed = 0, Int[]
    for (n, line) in enumerate(eachline(cov_path))
        tag = strip(first(line, 9))
        if tag == "-" || isempty(tag)
            continue
        end
        counted += 1
        if tag == "0"
            push!(missed, n)
        end
    end
    return counted, missed
end

"""
    run_coverage(dir, rest) -> Int

Run the driver again in a child process under `--code-coverage=@<dir>`, then report the counter
files that child wrote and delete them.
"""
function run_coverage(dir::AbstractString, rest::Vector{String})
    if !(isabspath(dir) && isdir(dir))
        error("`--coverage` needs an absolute directory. A relative one instruments nothing.")
    end
    cmd = `$(Base.julia_cmd()) -t 1 --project=$(Base.active_project()) --code-coverage=@$dir $(@__FILE__) $rest`
    p = run(ignorestatus(cmd); wait = false)
    wait(p)
    pid = getpid(p)
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
            rm(cov)
        end
    end
    return p.exitcode
end

const USAGE = "usage: run_files.jl [--done=<path>] [--coverage=<absolute dir>] test_X.jl..."

function main(args::Vector{String})
    done, coverage, names = nothing, nothing, String[]
    for a in args
        if startswith(a, "--done=")
            done = a[(length("--done=") + 1):end]
        elseif startswith(a, "--coverage=")
            coverage = a[(length("--coverage=") + 1):end]
        elseif startswith(a, "--")
            error(USAGE)
        else
            push!(names, a)
        end
    end
    if isempty(names)
        error(USAGE)
    end
    if coverage !== nothing
        rest = filter(a -> !startswith(a, "--coverage="), args)
        return run_coverage(coverage, rest)
    end
    BLAS.set_num_threads(1)
    lines = String[]
    sets = Collect[]
    status = 1
    try
        paths = resolve.(names)
        code = init_code()
        for p in paths
            ts, seconds = run_file(p; code)
            record_gate(p)
            push!(sets, ts)
            push!(lines, result_line(ts, seconds))
            println(last(lines))
        end
        push!(lines, done_line(sets))
        println(last(lines))
        status = all(green, sets) ? 0 : 1
    finally
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
    exit(RunFiles.main(ARGS))
end
