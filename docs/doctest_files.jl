#!/usr/bin/env julia
#=
Run the doctests of the docstrings that named source files hold, not of the whole module.

    julia -t 1 --project=docs docs/doctest_files.jl src/01_Base/02_Foo.jl src/09_Bar/
    julia -t 1 --project=docs docs/doctest_files.jl --against origin/dev
    julia -t 1 --project=docs docs/doctest_files.jl --fix src/01_Base/02_Foo.jl

A path names a file, or a directory and every file below it. `--against <ref>` names the `.jl`
files under `src/` and `ext/` that the branch changes against `ref`, from the merge-base on, as
`code_health/CodeHealth.jl` finds them. `--fix` writes the output of each failed block back into its
docstring, as `doctest(...; fix = true)` does.

The full run of the `run-doctests` skill checks every docstring of the module and the manual pages
under `docs/src`, which takes about nine minutes. This script checks the docstrings that the named
files hold, and nothing else. It is the check of a branch's own blocks between rebases. It does not
replace the full run before the push, because a change in one file can change what a block in
another file prints.

The script runs the setup of the doctest job of `.github/workflows/Docs.yml` from the text of that
file, so its `DocTestSetup` and its two show settings cannot drift from CI. It then copies each
selected docstring into the docs table of a scratch module, and gives that module to Documenter's
public `doctest(nothing, [module])`. A docstring keeps its `:path`, so `--fix` writes to the real
source file. Like the CI job, it reads the docstrings of `PortfolioOptimisers` alone, so a
docstring that an extension module holds is out of reach.

Run it in a fresh process, for the reason the `run-doctests` skill gives. A green run without
`--fix` records the gate stamp `doctest-files` through `code_health/gate_stamp.sh`.

This file loads no package at the top, so `test/test_76_gate_tooling.jl` includes it to test the
selection without Documenter.
=#
module DoctestFiles

include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))

const REPO_ROOT = CodeHealth.REPO_ROOT
const WORKFLOW = joinpath(REPO_ROOT, ".github", "workflows", "Docs.yml")
const USAGE = "usage: doctest_files.jl [--fix] (--against <ref> | <src file or directory>...)"

"""
    Command

What one run checks: the source `paths`, and whether it fixes a failed block.
"""
struct Command
    paths::Vector{String}
    fix::Bool
end

"""
    parse_command(args; root = REPO_ROOT) -> Command

Read the arguments. A relative path resolves against the working directory, and `--against <ref>`
adds the `.jl` files under `src/` and `ext/` that the branch changes against `ref`, as absolute
paths under `root`. A file the branch deleted is left out.
"""
function parse_command(args::AbstractVector{<:AbstractString};
                       root::AbstractString = REPO_ROOT)
    paths, fix, i = String[], false, 1
    while i <= length(args)
        a = args[i]
        if a == "--fix"
            fix = true
        elseif a == "--against"
            if i == length(args)
                error(USAGE)
            end
            i += 1
            for f in CodeHealth.branch_files(args[i]; root)
                if endswith(f, ".jl") &&
                   any(startswith(f, r) for r in CodeHealth.MEASURED_ROOTS)
                    p = joinpath(root, f)
                    if isfile(p)
                        push!(paths, p)
                    end
                end
            end
        elseif startswith(a, "--")
            error(USAGE)
        else
            p = abspath(a)
            if !(ispath(p))
                error("No file or directory `$a`.")
            end
            push!(paths, p)
        end
        i += 1
    end
    return Command(unique!(paths), fix)
end

"""
    in_scope(path, targets) -> Bool

Whether the source file `path` is one of `targets` or lies below a directory of `targets`. Both
sides compare as real paths, so a symbolic link or a `..` does not hide a match.
"""
function in_scope(path::AbstractString, targets::AbstractVector{<:AbstractString})
    if !(isfile(path))
        return false
    end
    p = realpath(path)
    for t in targets
        r = realpath(t)
        if p == r || (isdir(r) && startswith(p, joinpath(r, "")))
            return true
        end
    end
    return false
end

"""
    count_blocks(docstr) -> Int

The number of `jldoctest` blocks in the raw text of one docstring.
"""
function count_blocks(docstr::Base.Docs.DocStr)
    return sum(t -> t isa AbstractString ? count("```jldoctest", t) : 0, docstr.text;
               init = 0)
end

"""
    scope_module(mod, targets) -> (scope, docstrings, blocks, files)

A scratch module whose docs table holds the docstrings of `mod` that a file of `targets` holds.
A binding whose methods are documented in several files keeps the methods of the named files only.
`files` are the source files that gave at least one docstring.
"""
function scope_module(mod::Module, targets::AbstractVector{<:AbstractString})
    scope = Module(:DoctestScope)
    meta = Base.Docs.meta(scope)
    docstrings, blocks, files = 0, 0, Set{String}()
    for (binding, multidoc) in Base.Docs.meta(mod)
        kept = Base.Docs.MultiDoc()
        for sig in multidoc.order
            docstr = multidoc.docs[sig]
            path = get(docstr.data, :path, nothing)
            if path isa AbstractString && in_scope(path, targets)
                push!(kept.order, sig)
                kept.docs[sig] = docstr
                docstrings += 1
                blocks += count_blocks(docstr)
                push!(files, realpath(path))
            end
        end
        if !(isempty(kept.order))
            meta[binding] = kept
        end
    end
    return scope, docstrings, blocks, files
end

"""
    workflow_setup(text = read(WORKFLOW, String)) -> String

The Julia code of the `Run doctest` step of the doctest job, without its last call
`doctest(PortfolioOptimisers)`. It loads the packages, sets the two show settings, and sets the
`DocTestSetup` of `PortfolioOptimisers`.
"""
function workflow_setup(text::AbstractString = read(WORKFLOW, String))
    i = findfirst("- name: Run doctest", text)
    if i === nothing
        error("`$WORKFLOW` has no step `Run doctest`.")
    end
    m = match(r"--project=docs -e '([^']*)'", text, last(i))
    if m === nothing
        error("The step `Run doctest` of `$WORKFLOW` has no `julia --project=docs -e '...'` call.")
    end
    code = rstrip(m[1])
    tail = "doctest(PortfolioOptimisers)"
    if !(endswith(code, tail))
        error("The step `Run doctest` of `$WORKFLOW` no longer ends with `$tail`. Update `workflow_setup`.")
    end
    return code[1:(end - length(tail))]
end

"""
    record_gate()

Record the gate stamp `doctest-files`. A failure to write it never fails the run.
"""
function record_gate()
    script = joinpath(REPO_ROOT, "code_health", "gate_stamp.sh")
    try
        cmd = Cmd(`bash $script record doctest-files`; dir = REPO_ROOT)
        run(pipeline(cmd; stdout = devnull, stderr = devnull))
    catch
    end
    return nothing
end

"""
    run_doctests(cmd) -> Int

Run the doctests of `cmd` after the setup of the workflow has loaded the packages into `Main`. It
returns the exit code: `0` when every block passes, `1` otherwise.
"""
function run_doctests(cmd::Command)
    po = Main.PortfolioOptimisers
    scope, docstrings, blocks, files = scope_module(po, cmd.paths)
    for p in cmd.paths
        if isfile(p) && !(realpath(p) in files)
            println("NOTE $(relpath(p, REPO_ROOT)) holds no docstring of PortfolioOptimisers.")
        end
    end
    head = "DOCTEST files=$(length(files)) docstrings=$docstrings blocks=$blocks"
    if blocks == 0
        println(head, " status=green")
        return 0
    end
    Main.DocMeta.setdocmeta!(scope, :DocTestSetup,
                             Main.DocMeta.getdocmeta(po, :DocTestSetup))
    green = false
    seconds = @elapsed try
        Main.doctest(nothing, [scope]; testset = "Doctests of $(length(files)) files",
                     fix = cmd.fix)
        green = true
    catch err
        if !(err isa Main.Test.TestSetException)
            rethrow()
        end
    end
    println(head,
            " seconds=$(round(seconds; digits = 1)) status=$(green ? "green" : "red")")
    if green && !(cmd.fix)
        record_gate()
    end
    return green ? 0 : 1
end

function main(args::Vector{String})
    cmd = parse_command(args)
    if isempty(cmd.paths)
        println("DOCTEST files=0 docstrings=0 blocks=0 status=green")
        return 0
    end
    include_string(Main, "import Test\n" * workflow_setup(), WORKFLOW)
    return Base.invokelatest(run_doctests, cmd)
end

end # module

# `@__MODULE__` keeps a test that includes this file from running the script.
if @__MODULE__() === Main && abspath(PROGRAM_FILE) == @__FILE__
    exit(DoctestFiles.main(ARGS))
end
