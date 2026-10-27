#=
The tools a branch uses around the gates: the scope of a gate script, the gate stamps with their
push guard, and the test driver.

On a busy map, sibling commits land on `dev` while a branch runs its gates. Three costs followed.
A plain `check` reported rows that other commits raised, and a session could not tell them from its
own. A gate measured before a rebase was trusted after it, and a merged tree went out unmeasured.
Each session wrote its own driver for one test file, and one of those drivers left out the
`init_code` preamble. The three tools below answer the three costs, and this file is their fixture:

  - `code_health/CodeHealth.jl` takes `--against <ref>` and `--files <path>...`. A scoped run
    answers for the branch's rows, names every foreign row, and a scoped refresh keeps the
    foreign rows as the baseline has them.
  - `code_health/gate_stamp.sh` records the merge-base a gate measured, and its `check` is the
    pre-push guard of `.pre-commit-config.yaml`. The guard requires the stamps of the gates of a
    push that changes `src/`, leaves out the commits the worktree pushed itself, and stales a
    stamp on a shared ledger file only when its gate reads the file. The test holds the list of
    readers against the tests that name each ledger file.
  - `code_health/gate_reach.jl` names the class of a change, so the guard demands `jet` and
    `doctest` only of a change they can reach. The test checks each class on a fixture file, and
    the demand of the guard for each.
  - `code_health/julia_slot.jl` is the machine-wide slot that a run which loads the package takes,
    and `code_health/manifest_provenance.jl` refuses a commit that moves an analyser in the
    Manifest without the refresh of its baselines.
  - `test/run_files.jl` runs named test files after the preamble and counts each one. It prunes
    a file to the test sets `--testset` selects, times the sets, bounds a file with a watchdog,
    and names a red that `test/known_red.toml` lists. It takes a file by a unique prefix, a case
    regex, a resume log, and the flags of CI. `test/known_red.jl` holds the rows for it and for
    `test/runtests.jl`, whose filter passes a CI run when a row covers every failure (#1781).
  - `code_health/sweep_check.jl` names `code_health/census_preview.jl` when a branch adds a
    declaration that the two package-level censuses read. The test checks which diff lines name one.
  - `code_health/sweep_check.jl` and `code_health/sweep_triage.jl` refuse an open sweep issue whose
    title names a `.jl` file that has no row. The test checks the match, the report and the refusal.
  - `docs/doctest_files.jl` runs the doctests of the docstrings that named source files hold. The
    test checks its selection on a fixture module. The doctest run itself needs Documenter, which
    the test environment does not hold.

The git fixtures are small repositories in a temporary directory. CI runs on Ubuntu, where `git`
and `bash` are present. The loads sit OUTSIDE the `@testset`, for the reason
`test_49_coverage_attribution_census.jl` gives.
=#
module GateScopeSeam
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

module DriverSeam
include(joinpath(@__DIR__, "run_files.jl"))
end

module DocFilesSeam
include(joinpath(@__DIR__, "..", "docs", "doctest_files.jl"))
end

module SweepCheckSeam
include(joinpath(@__DIR__, "..", "code_health", "sweep_check.jl"))
end

module SweepTriageSeam
include(joinpath(@__DIR__, "..", "code_health", "sweep_triage.jl"))
end

module ProvenanceSeam
include(joinpath(@__DIR__, "..", "code_health", "manifest_provenance.jl"))
end

module GateReachSeam
include(joinpath(@__DIR__, "..", "code_health", "gate_reach.jl"))
end

using Test

const GATE_STAMP = joinpath(@__DIR__, "..", "code_health", "gate_stamp.sh")

function git(dir::AbstractString, args...)
    cmd = `git -c user.name=fixture -c user.email=fixture@example.com -c commit.gpgsign=false $args`
    return readchomp(Cmd(cmd; dir))
end

function put(dir::AbstractString, path::AbstractString, text::AbstractString)
    mkpath(dirname(joinpath(dir, path)))
    write(joinpath(dir, path), text)
    return nothing
end

function commit(dir::AbstractString, msg::AbstractString)
    git(dir, "add", "-A")
    git(dir, "commit", "-q", "-m", msg)
    return git(dir, "rev-parse", "HEAD")
end

"""
A repository with a `trunk` branch that plays `origin/dev`, and a `work` branch that changed
`src/a/one.jl` and holds the untracked `src/a/new.jl`.
"""
function fixture_repo()
    dir = mktempdir()
    git(dir, "init", "-q", "-b", "trunk")
    put(dir, "src/a/one.jl", "one = 1\n")
    put(dir, "src/b/two.jl", "two = 2\n")
    put(dir, "test/t.jl", "t = 0\n")
    c0 = commit(dir, "c0")
    git(dir, "checkout", "-q", "-b", "work")
    put(dir, "src/a/one.jl", "one = 11\n")
    c1 = commit(dir, "c1")
    git(dir, "checkout", "-q", "trunk")
    put(dir, "src/b/two.jl", "two = 22\n")
    c2 = commit(dir, "c2")
    git(dir, "checkout", "-q", "work")
    put(dir, "src/a/new.jl", "new = 0\n")
    return dir, c0, c1, c2
end

"""
Run `gate_stamp.sh` in `dir` and return the exit code and the output. `force` stands in for a
worktree under `.claude/worktrees/`. No gate is required unless `GATE_STAMP_REQUIRED` names one.
"""
function stamp(dir::AbstractString, args...; force::Bool = true, env...)
    vars = Pair{String, String}["GATE_STAMP_REF" => "trunk", "GATE_STAMP_REQUIRED" => ""]
    if force
        push!(vars, "GATE_STAMP_FORCE" => "1")
    end
    for (k, v) in env
        push!(vars, string(k) => string(v))
    end
    io = IOBuffer()
    cmd = addenv(Cmd(`bash $GATE_STAMP $args`; dir), vars...)
    p = run(pipeline(ignorestatus(cmd); stdout = io, stderr = io))
    return p.exitcode, String(take!(io))
end

const Q3 = "\"\"\""

"""
A source file with a docstring that holds a `jldoctest` block, a field docstring, and a function
that returns a string. Each keyword changes one of them.
"""
function reach_text(; doc = "Doc of f.", out = "2", body = "x + 1", field = "field doc",
                    msg = "message")
    return """
           $Q3
               f(x)

           $doc

           ```jldoctest
           julia> f(1)
           $out
           ```
           $Q3
           f(x) = $body

           struct A
               $Q3$field$Q3
               a::Int
           end

           g() = "$msg"
           """
end

"""
A repository whose `trunk` holds `src/a/one.jl` as `reach_text()`, and whose `work` branch
commits `text` there.
"""
function reach_repo(text::AbstractString)
    dir = mktempdir()
    git(dir, "init", "-q", "-b", "trunk")
    put(dir, "src/a/one.jl", reach_text())
    c0 = commit(dir, "c0")
    git(dir, "checkout", "-q", "-b", "work")
    put(dir, "src/a/one.jl", text)
    c1 = commit(dir, "c1")
    return dir, c0, c1
end

@testset "Gate tooling: scope, stamps and driver" begin
    CH = GateScopeSeam.CodeHealth

    @testset "parse_command reads a scope after the verb" begin
        @test CH.parse_command(["check"]) == CH.Command(:check, false)
        c = CH.parse_command(["refresh", "--accept-rise", "--against", "origin/dev"])
        @test (c.verb, c.accept_rise, c.against, c.files) ==
              (:refresh, true, "origin/dev", nothing)
        c = CH.parse_command(["check", "--files", "src/a.jl", "src/b.jl"])
        @test c.files == ["src/a.jl", "src/b.jl"]
        c = CH.parse_command(["refresh", "--files", "src/a.jl", "--accept-rise"])
        @test (c.files, c.accept_rise) == (["src/a.jl"], true)
        for bad in ([], ["measure"], ["check", "--accept-rise"], ["check", "--against"],
                    ["check", "--files"], ["check", "--against", "x", "--files", "y"],
                    ["refresh", "--accept-rise", "--accept-rise"])
            @test_throws ErrorException CH.parse_command(bad)
        end
    end

    @testset "a scoped run drops foreign rows and names them" begin
        recorded = Dict("src/a.jl" => Dict("cyc" => 1), "src/b.jl" => Dict("cyc" => 1),
                        "@macro" => Dict("cyc" => 1))
        measured = Dict("src/a.jl" => Dict("cyc" => 2), "src/b.jl" => Dict("cyc" => 2),
                        "@macro" => Dict("cyc" => 2))
        CH.set_scope!(nothing)
        @test length(CH.rises(recorded, measured, ["cyc"])) == 3
        CH.set_scope!(["src/a.jl"])
        rs = CH.rises(recorded, measured, ["cyc"])
        # A key that is not a path under `src/` or `ext/` is always in scope.
        @test sort([r.key for r in rs]) == ["@macro", "src/a.jl"]
        @test CH.SCOPE.foreign == Set(["src/b.jl"])
        missing_rows, dead_rows = CH.set_differences(["src/a.jl", "src/c.jl", "ext/x.jl"],
                                                     ["src/a.jl", "src/d.jl"])
        @test (missing_rows, dead_rows) == (String[], String[])
        @test CH.SCOPE.foreign == Set(["src/b.jl", "src/c.jl", "src/d.jl", "ext/x.jl"])
        CH.set_scope!(["src/c.jl"])
        @test CH.set_differences(["src/a.jl", "src/c.jl"], ["src/a.jl"]) ==
              (["src/c.jl"], String[])
        CH.set_scope!(nothing)
        @test isempty(CH.SCOPE.foreign)
    end

    @testset "a scoped refresh keeps every foreign row" begin
        old = """
        [run.main.file]
        "src/a.jl" = { reviewed = 1 }
        "src/b.jl" = { reviewed = 1 }
        "src/gone.jl" = { reviewed = 4 }

        [run.other.file]
        "src/a.jl" = { reviewed = 2 }
        "src/b.jl" = { reviewed = 2 }
        """
        new = """
        [run.main.file]
        "src/a.jl" = { reviewed = 5 }
        "src/b.jl" = { reviewed = 5 }
        "src/new.jl" = { reviewed = 0 }

        [run.other.file]
        "src/a.jl" = { reviewed = 6 }
        "src/b.jl" = { reviewed = 6 }
        """
        CH.set_scope!(["src/a.jl"])
        merged = CH.keep_foreign_rows(new, old)
        CH.set_scope!(nothing)
        @test merged == """
              [run.main.file]
              "src/a.jl" = { reviewed = 5 }
              "src/b.jl" = { reviewed = 1 }
              "src/new.jl" = { reviewed = 0 }

              [run.other.file]
              "src/a.jl" = { reviewed = 6 }
              "src/b.jl" = { reviewed = 2 }
              """
    end

    @testset "branch_files starts at the merge-base" begin
        dir, _, _, _ = fixture_repo()
        # `trunk` changed `src/b/two.jl` after `work` started. A diff against `trunk` itself
        # names it, and the merge-base does not.
        @test occursin("src/b/two.jl", git(dir, "diff", "--name-only", "trunk"))
        @test CH.branch_files("trunk"; root = dir) == ["src/a/new.jl", "src/a/one.jl"]
        @test CH.command_scope(CH.Command(:check, false, "trunk", nothing); root = dir) ==
              ["src/a/new.jl", "src/a/one.jl"]
        @test CH.command_scope(CH.Command(:check, false)) === nothing
    end

    @testset "the push guard refuses a gate the target moved under" begin
        dir, c0, c1, c2 = fixture_repo()
        stamps = joinpath(dir, ".git", "gate_stamps")

        # Outside a worktree of `.claude/worktrees/` nothing is recorded or checked.
        @test stamp(dir, "record", "jet"; force = false) == (0, "")
        @test !isfile(stamps)

        @test first(stamp(dir, "record", "jet")) == 0
        @test readchomp(stamps) == "jet\t$c0"
        push = (; PRE_COMMIT_FROM_REF = c2, PRE_COMMIT_TO_REF = c1,
                PRE_COMMIT_REMOTE_BRANCH = "refs/heads/dev")

        # `trunk` moved in `src/b/`, which the branch does not touch.
        @test first(stamp(dir, "check"; push...)) == 0
        # A push that changes no file under `src/` or `ext/` has no directory to compare.
        @test stamp(dir, "check"; push..., PRE_COMMIT_TO_REF = c0) == (0, "")

        # `trunk` adds a file in `src/a/`, the directory the branch changes.
        git(dir, "checkout", "-q", "trunk")
        put(dir, "src/a/three.jl", "three = 3\n")
        c3 = commit(dir, "c3")
        git(dir, "checkout", "-q", "work")
        code, out = stamp(dir, "check"; push..., PRE_COMMIT_FROM_REF = c3)
        @test code == 1
        @test occursin("jet, measured on $(c0[1:10])", out)
        @test occursin("src/a/three.jl", out)
        @test !occursin("src/b/two.jl", out)

        # The guard protects `dev` alone, and a plain run reads HEAD against the ref.
        @test first(stamp(dir, "check"; push..., PRE_COMMIT_FROM_REF = c3,
                          PRE_COMMIT_REMOTE_BRANCH = "refs/heads/other")) == 0
        @test first(stamp(dir, "check")) == 1

        # A gate run again on the rebased tree is current.
        git(dir, "rebase", "-q", "trunk")
        @test first(stamp(dir, "record", "jet")) == 0
        head = git(dir, "rev-parse", "HEAD")
        @test first(stamp(dir, "check"; push..., PRE_COMMIT_FROM_REF = c3,
                          PRE_COMMIT_TO_REF = head)) == 0

        # A file both sides change, outside `src/`, trips the guard too.
        put(dir, "test/t.jl", "t = 1\n")
        head = commit(dir, "c4 on work")
        @test first(stamp(dir, "record", "test/t.jl")) == 0
        git(dir, "checkout", "-q", "trunk")
        put(dir, "test/t.jl", "t = 2\n")
        c5 = commit(dir, "c5")
        git(dir, "checkout", "-q", "work")
        code, out = stamp(dir, "check"; push..., PRE_COMMIT_FROM_REF = c5,
                          PRE_COMMIT_TO_REF = head)
        # Both gates measured `c3`, and the branch changes the file `c5` changed.
        @test code == 1
        @test occursin("test/t.jl, measured on $(c3[1:10])", out)
        @test occursin("jet, measured on $(c3[1:10])", out)

        # Recording a gate again replaces its line, and `drop` forgets a stamp.
        @test first(stamp(dir, "record", "jet")) == 0
        @test count(l -> startswith(l, "jet\t"), readlines(stamps)) == 1
        @test first(stamp(dir, "drop", "test/t.jl")) == 0
        @test last(stamp(dir, "list")) == "jet\t$c3\n"
        @test first(stamp(dir, "drop", "jet")) == 0
        @test last(stamp(dir, "list")) == ""
        @test first(stamp(dir, "check"; push..., PRE_COMMIT_FROM_REF = c5,
                          PRE_COMMIT_TO_REF = head)) == 0
        @test first(stamp(dir, "bogus")) == 2
    end

    @testset "the push guard ends a refusal with the line that drops the stale stamps" begin
        dir, c0, c1, _ = fixture_repo()
        @test first(stamp(dir, "record", "jet", "size")) == 0
        git(dir, "checkout", "-q", "trunk")
        put(dir, "src/a/three.jl", "three = 3\n")
        c3 = commit(dir, "c3")
        git(dir, "checkout", "-q", "work")
        code, out = stamp(dir, "check"; PRE_COMMIT_FROM_REF = c3, PRE_COMMIT_TO_REF = c1,
                          PRE_COMMIT_REMOTE_BRANCH = "refs/heads/dev")
        @test code == 1
        @test occursin("code_health/gate_stamp.sh drop jet size", out)
    end

    @testset "a push that changes src/ needs the required gates or a waiver" begin
        dir, c0, c1, c2 = fixture_repo()
        push = (; PRE_COMMIT_FROM_REF = c2, PRE_COMMIT_TO_REF = c1,
                PRE_COMMIT_REMOTE_BRANCH = "refs/heads/dev",
                GATE_STAMP_REQUIRED = "jet doctest")
        # No stamp at all: both gates are absent.
        code, out = stamp(dir, "check"; push...)
        @test code == 1
        @test occursin("no stamp records these gates: jet doctest", out)
        @test first(stamp(dir, "record", "jet")) == 0
        code, out = stamp(dir, "check"; push...)
        @test code == 1
        @test occursin("these gates: doctest", out)
        # A waiver carries its reason, and `list` shows it.
        @test first(stamp(dir, "waive", "doctest", "no", "docstring", "changed")) == 0
        @test occursin("doctest\twaived: no docstring changed", last(stamp(dir, "list")))
        @test first(stamp(dir, "check"; push...)) == 0
        @test first(stamp(dir, "waive", "doctest")) == 2
        # A push that changes nothing under src/ or ext/ needs no gate.
        dir2, d0, _, d2 = fixture_repo()
        @test first(stamp(dir2, "check"; push..., PRE_COMMIT_FROM_REF = d2,
                          PRE_COMMIT_TO_REF = d0)) == 0
    end

    @testset "gate_reach names the class of a change" begin
        R = GateReachSeam.GateReach
        old = reach_text()
        @test R.file_class("src/a.jl", old, reach_text(; doc = "Other prose.")) == "prose"
        @test R.file_class("src/a.jl", old, reach_text(; field = "other doc")) == "prose"
        @test R.file_class("src/a.jl", old, reach_text(; out = "3")) == "block"
        @test R.file_class("src/a.jl", old, reach_text(; msg = "other")) == "literal"
        @test R.file_class("src/a.jl", old, reach_text(; body = "x + 2")) == "code"
        @test R.file_class("src/a.jl", old, nothing) == "code"
        @test R.file_class("src/a.jl", old, "f(x) = (") == "code"
        # A string that a function returns is not a docstring.
        h(s) = "function h()\n    \"$s\"\nend\n"
        @test R.file_class("src/a.jl", h("a"), h("b")) == "literal"
        # A key or a value of a docstring dictionary feeds only docstrings.
        d(k, v) = "unique_key_dict!(arg_dict, :arg_dict, :$k => \"$v\")\n"
        dicts = "src/01_Base/01_DocstringDictionaries/02_X.jl"
        @test R.file_class(dicts, d("w", "a"), d("v", "b")) == "prose"
        @test R.file_class("src/b.jl", d("w", "a"), d("w", "b")) == "literal"
        @test R.file_class(dicts, d("w", "a"), d("w", "a") * "x = 1\n") == "code"
    end

    @testset "the push guard demands only the gates a change reaches" begin
        guard(dir, from, to) = stamp(dir, "check"; PRE_COMMIT_FROM_REF = from,
                                     PRE_COMMIT_TO_REF = to,
                                     PRE_COMMIT_REMOTE_BRANCH = "refs/heads/dev",
                                     GATE_STAMP_REQUIRED = "jet doctest")
        demand(text) = guard(reach_repo(text)...)
        @test demand(reach_text(; doc = "Other prose.")) == (0, "")
        code, out = demand(reach_text(; msg = "other"))
        @test occursin("(literal), and no stamp records these gates: doctest\n", out)
        code, out = demand(reach_text(; body = "x + 2"))
        @test occursin("(code), and no stamp records these gates: jet doctest\n", out)
        # A changed block needs the scoped run, and the full run covers it.
        dir, c0, c1 = reach_repo(reach_text(; out = "3"))
        code, out = guard(dir, c0, c1)
        @test code == 1
        @test occursin("(block), and no stamp records these gates: doctest-files\n", out)
        @test first(stamp(dir, "record", "doctest")) == 0
        @test guard(dir, c0, c1) == (0, "")
        # A file under src/ that is not Julia reaches no gate.
        put(dir, "src/CLAUDE.md", "rules\n")
        c2 = commit(dir, "c2")
        @test guard(dir, c1, c2) == (0, "")
        # `reach` reads the working tree against the ref, and names the gates.
        put(dir, "src/a/one.jl", reach_text(; doc = "Other prose."))
        code, out = stamp(dir, "reach"; GATE_STAMP_REQUIRED = "jet size doctest")
        @test code == 0
        @test occursin("prose src/a/one.jl\nreach prose\ngates size\n", out)
    end

    @testset "a commit this worktree pushed is not a sibling's change" begin
        dir, c0, c1, c2 = fixture_repo()
        @test first(stamp(dir, "record", "jet")) == 0
        # The guard lets the push of `c1` through and remembers it. `trunk` then holds `c1`.
        @test first(stamp(dir, "check"; PRE_COMMIT_FROM_REF = c0, PRE_COMMIT_TO_REF = c1,
                          PRE_COMMIT_REMOTE_BRANCH = "refs/heads/dev")) == 0
        @test occursin(c1, read(joinpath(dir, ".git", "gate_pushed"), String))
        git(dir, "checkout", "-q", "trunk")
        git(dir, "reset", "-q", "--hard", c1)
        git(dir, "checkout", "-q", "work")
        put(dir, "src/a/one.jl", "one = 111\n")
        c6 = commit(dir, "c6 on work")
        push = (; PRE_COMMIT_FROM_REF = c1, PRE_COMMIT_TO_REF = c6,
                PRE_COMMIT_REMOTE_BRANCH = "refs/heads/dev")
        # `trunk` moved from `c0` to `c1` in `src/a/one.jl`, the file `c6` changes, but the move is
        # this worktree's own push.
        @test first(stamp(dir, "check"; push...)) == 0
        rm(joinpath(dir, ".git", "gate_pushed"))
        @test first(stamp(dir, "check"; push...)) == 1
    end

    @testset "a shared ledger file stales only the gates that read it" begin
        dir, c0, _, _ = fixture_repo()
        put(dir, "GLOSSARY.md", "a\n")
        put(dir, "code_health/sweep_manifest.toml", "a = 1\n")
        head = commit(dir, "ledgers on work")
        @test first(stamp(dir, "record", "jet", "prose", "test/test_45_sweep_census.jl")) ==
              0
        git(dir, "checkout", "-q", "trunk")
        put(dir, "GLOSSARY.md", "b\n")
        put(dir, "code_health/sweep_manifest.toml", "a = 2\n")
        c7 = commit(dir, "ledgers on trunk")
        git(dir, "checkout", "-q", "work")
        code, out = stamp(dir, "check"; PRE_COMMIT_FROM_REF = c7, PRE_COMMIT_TO_REF = head,
                          PRE_COMMIT_REMOTE_BRANCH = "refs/heads/dev")
        @test code == 1
        @test occursin("prose, measured on", out)
        @test occursin("test/test_45_sweep_census.jl, measured on", out)
        @test !occursin("jet, measured on", out)
        @test occursin("drop prose test/test_45_sweep_census.jl", out)
        @test strip(last(stamp(dir, "readers", "src/a/one.jl"))) == "*"
        @test strip(last(stamp(dir, "readers", "code_health/jet_baseline.toml"))) == "jet"
    end

    @testset "the ledger readers of the push guard cover every test that names the ledger" begin
        # A test file reads a ledger when its own text names it, or when it includes a
        # `code_health/` script whose text names it. `CodeHealth.jl` is left out: nearly every
        # census includes it, and it names the manifest only for `test_45`.
        root = joinpath(@__DIR__, "..")
        needles = ["code_health/sweep_manifest.toml" => ["sweep_manifest"],
                   "GLOSSARY.md" => ["GLOSSARY"],
                   "code_health/size_baseline.toml" => ["size_baseline"],
                   "code_health/prose_baseline.toml" => ["prose_baseline"],
                   "code_health/jet_baseline.toml" => ["jet_baseline"],
                   "code_health/complexity_baseline.toml" => ["complexity_baseline"],
                   "code_health/perf_baseline.toml" => ["perf_baseline"]]
        scripts = filter(f -> endswith(f, ".jl") && f != "CodeHealth.jl",
                         readdir(joinpath(root, "code_health")))
        script_text = Dict(f => read(joinpath(root, "code_health", f), String)
                           for f in scripts)
        for (ledger, words) in needles
            listed = split(strip(last(stamp(root, "readers", ledger; force = false))))
            for f in
                filter(f -> occursin(r"^test_.*\.jl$", f), readdir(joinpath(root, "test")))
                f == "test_76_gate_tooling.jl" && continue
                text = read(joinpath(root, "test", f), String)
                names_it = any(w -> occursin(w, text), words)
                via = any(s -> occursin("\"code_health\", \"$s\"", text) &&
                               any(w -> occursin(w, script_text[s]), words), scripts)
                if names_it || via
                    @test "test/$f" in listed
                end
            end
        end
    end

    @testset "run_files counts each file after the preamble" begin
        RF = DriverSeam.RunFiles
        code = RF.init_code()
        @test code isa Expr && code.head === :block
        # The preamble defines `find_tol`, which a hand-copied `using` list leaves out.
        @test any(e -> e isa Expr && e.head === :function, code.args)

        path = joinpath(mktempdir(), "test_fixture.jl")
        write(path, """
              @testset "outer" begin
                  @test find_tol(1.0, 1.0) === nothing
                  # `@test_throws` returns the result that `record` returns (#1475).
                  r = @test_throws ErrorException error("x")
                  @test r.value isa ErrorException
                  @testset "inner" begin
                      @test 1 == 1
                      @test 1 == 2
                      @test_broken 1 == 2
                      # A failed `@test_logs` is a fail, not an error of the driver.
                      @test_logs (:warn, "never logged") nothing
                  end
                  @test error("boom")
              end
              """)
        RF.OUT[] = devnull
        ts, seconds = redirect_stdout(devnull) do
            return RF.run_file(path; code)
        end
        RF.OUT[] = stdout
        @test (ts.pass, ts.fail, ts.error, ts.broken) == (4, 2, 1, 1)
        @test seconds >= 0
        @test RF.result_line(ts, 1.0) ==
              "RESULT test_fixture.jl pass=4 fail=2 error=1 broken=1 seconds=1.0"
        @test RF.done_line([ts]) == "DONE files=1 pass=4 fail=2 error=1 broken=1 status=red"
        @test RF.resolve("test_76_gate_tooling.jl") ==
              joinpath(RF.TEST_DIR, "test_76_gate_tooling.jl")
        @test_throws ErrorException RF.resolve("test_no_such_file.jl")

        cov = joinpath(mktempdir(), "x.jl.123.cov")
        write(cov,
              "        - function f(x)\n        3     y = x\n        0     z = 1\n" *
              "        - end\n")
        @test RF.coverage_lines(cov) == (2, [3])
        # Each of the two refusals of `--coverage` names its own cause.
        msg(f) =
            try
                f()
                ""
            catch e
                sprint(showerror, e)
            end
        @test occursin("absolute directory", msg(() -> RF.run_coverage("cov", String[])))
        missing_dir = joinpath(mktempdir(), "no_such_dir")
        @test occursin("a source directory that exists",
                       msg(() -> RF.run_coverage(missing_dir, String[])))
    end

    @testset "run_files --testset prunes to the selected sets, --time-depth times them" begin
        RF = DriverSeam.RunFiles
        code = RF.init_code()
        dir = mktempdir()
        # An unselected body that ran would error, so a count of zero errors proves the pruning.
        write(joinpath(dir, "helper.jl"), """
              const FROM_HELPER = 3
              @testset "helper set" begin
                  error("an unselected set of an included file ran")
              end
              """)
        path = joinpath(dir, "test_fixture.jl")
        write(path, """
              include(joinpath(@__DIR__, "helper.jl"))
              const SETUP = 1
              @test error("a top-level assertion ran")
              @testset "outer" begin
                  x = SETUP + 1
                  @test error("an assertion on the path ran")
                  @testset "skip me" begin
                      error("an unselected body ran")
                  end
                  for k in 1:2
                      y = k
                      @testset "pick \$(k)" begin
                          @test y == k
                          @testset "inner" begin
                              @test x == 2
                              @test FROM_HELPER == 3
                          end
                      end
                      @testset "other" begin
                          error("an unselected body ran")
                      end
                  end
                  if true
                      @testset "pick in an if" begin
                          @test true
                      end
                  else
                      @testset "never" begin
                          error("an unselected body ran")
                      end
                  end
                  foreach(1:1) do _
                      @testset "pick in a do block" begin
                          @test true
                      end
                      @testset "skip in a do block" begin
                          error("an unselected body ran")
                      end
                  end
              end
              @testset "not on the path" begin
                  error("an unselected body ran")
              end
              """)
        sel = RF.Select(r"^pick")
        buf = IOBuffer()
        RF.OUT[] = buf
        ts, _ = redirect_stdout(devnull) do
            return RF.run_file(path; code, select = sel, time_depth = 2)
        end
        RF.OUT[] = stdout
        @test (ts.pass, ts.fail, ts.error, ts.broken) == (8, 0, 0, 0)
        @test sel.matched[] == 3
        @test ts.ran ==
              ["outer > pick 1 > inner", "outer > pick 1", "outer > pick 2 > inner",
               "outer > pick 2", "outer > pick in an if", "outer > pick in a do block",
               "outer"]
        # Depth 2 and above are timed, depth 3 is not, and each line ends with its seconds.
        times = filter(startswith("TIME "), split(String(take!(buf)), '\n'))
        @test length(times) == 5
        @test all(l -> startswith(l, "TIME test_fixture.jl outer"), times)
        @test all(l -> tryparse(Float64, last(split(l))) !== nothing, times)
        @test RF.result_line(ts, 1.0; selected = 3) ==
              "RESULT test_fixture.jl pass=8 fail=0 error=0 broken=0 seconds=1.0 selected=3"

        # A regex that selects nothing makes the file red: a run that tested nothing proves
        # nothing.
        RF.OUT[] = devnull
        ts, _ = redirect_stdout(devnull) do
            return RF.run_file(path; code, select = RF.Select(r"no such set"))
        end
        RF.OUT[] = stdout
        @test (ts.pass, ts.error) == (0, 1)

        @test RF.testset_label(Meta.parse("@testset \"a \$(b) c\" for b in 1:2 end")) ==
              "a \$(b) c"
        @test RF.testset_label(Meta.parse("@testset begin end")) == "test set"
        @test RF.is_testset(Meta.parse("Test.@testset \"a\" begin end"))
    end

    @testset "run_files reads test/known_red.toml and names a known red" begin
        RF = DriverSeam.RunFiles
        # Every row of the real file names a test file that exists.
        for r in RF.known_red()
            @test isfile(joinpath(RF.TEST_DIR, r.file))
            @test r.issue > 0 && !isempty(r.reason)
        end
        bad = joinpath(mktempdir(), "known_red.toml")
        write(bad,
              "[[red]]\nfile = \"test_x.jl\"\nissue = 1\nreason = \"r\"\ntestest = \"typo\"\n")
        @test_throws ErrorException RF.known_red(bad)
        write(bad, "[[red]]\nfile = \"test_x.jl\"\nissue = 1\n")
        @test_throws ErrorException RF.known_red(bad)
        @test RF.known_red(joinpath(mktempdir(), "absent.toml")) == RF.KnownRed[]

        function fixture_set(; failed, ran)
            ts = RF.Collect("test_x.jl"; top = true)
            ts.failed, ts.ran = failed, ran
            ts.fail = length(failed)
            return ts
        end
        rows = [RF.KnownRed("test_x.jl", "slow solve", 7, "a tolerance", "Linux"),
                RF.KnownRed("test_y.jl", nothing, 8, "other file", nothing)]
        ran = ["a > slow solve", "a > fast", "a"]
        # A red inside the row's test set is named, with the host note.
        note, stale = RF.known_red_notes(fixture_set(; failed = ["a > slow solve"], ran),
                                         rows)
        @test note == "known red, #7: a tolerance (Linux)"
        @test isempty(stale)
        # A red outside every row is named after the rows.
        note, stale = RF.known_red_notes(fixture_set(;
                                                     failed = ["a > slow solve",
                                                               "a > fast"], ran), rows)
        @test note ==
              "known red, #7: a tolerance (Linux); red outside known_red.toml: a > fast"
        # A red that no row covers has no note, so it reads as the session's own.
        note, stale = RF.known_red_notes(fixture_set(; failed = ["a > fast"], ran), rows)
        @test note == ""
        @test length(stale) == 1 && startswith(only(stale), "STALE test/known_red.toml")
        # A row whose test set passed is stale. Under --testset, a row whose set did not run is
        # not judged, nor is a row of a whole file.
        note, stale = RF.known_red_notes(fixture_set(; failed = String[], ran), rows)
        @test note == "" && length(stale) == 1
        @test occursin("#7", only(stale)) && occursin("Linux", only(stale))
        note, stale = RF.known_red_notes(fixture_set(; failed = String[],
                                                     ran = ["a > fast"]), rows;
                                         filtered = true)
        @test note == "" && isempty(stale)
        whole = [RF.KnownRed("test_x.jl", nothing, 9, "all of it", nothing)]
        note, stale = RF.known_red_notes(fixture_set(; failed = [""], ran = String[]),
                                         whole)
        @test note == "known red, #9: all of it"
        note, stale = RF.known_red_notes(fixture_set(; failed = String[], ran = String[]),
                                         whole; filtered = true)
        @test isempty(stale)
        @test length(last(RF.known_red_notes(fixture_set(; failed = String[],
                                                         ran = String[]), whole))) == 1
    end

    @testset "runtests passes a run whose every failure a known_red row covers (#1781)" begin
        KR = DriverSeam.RunFiles.KnownRedRows
        rows = [KR.KnownRed("test_x.jl", "slow solve", 7, "a tolerance", "Linux"),
                KR.KnownRed("test_y.jl", nothing, 8, "other file", nothing)]
        # `Test` prints the head of a block in bold when the log has colour.
        bold(s) = "\e[1m" * s * "\e[22m"
        function block(name, head, site)
            return "Error in testset $name:\n$(bold(head)) at $(bold(site))\n  Expression: a == b\n"
        end
        summary = "Test Summary: | Pass  Fail  Total\n  Overall     |    9     1     10\n"
        # A block that a test file prints comes before the summary, and is not a site.
        early = block("slow solve", "Test Failed", "/r/test/test_x.jl:1")
        text(blocks...) = early * summary * join(blocks)
        known = block("slow solve", "Test Failed", "/r/test/test_x.jl:12")
        whole = block("any set", "Error During Test", "/r/test/test_y.jl:4")
        @test KR.failure_sites(text(known, whole)) ==
              [KR.FailureSite("slow solve", "/r/test/test_x.jl", 12),
               KR.FailureSite("any set", "/r/test/test_y.jl", 4)]
        @test isempty(KR.failure_sites(text())) && isempty(KR.failure_sites(early))
        # Every failure is covered: the run passes, and each failure is named.
        lines, green = KR.verdict(text(known, whole), rows)
        @test green
        @test lines[1] ==
              "KNOWN RED /r/test/test_x.jl:12 in \"slow solve\": #7: a tolerance (Linux)"
        @test lines[2] == "KNOWN RED /r/test/test_y.jl:4 in \"any set\": #8: other file"
        @test startswith(lines[3], "KNOWN RED only")
        # A new red behind a known one keeps the run red, and is named.
        lines, green = KR.verdict(text(known,
                                       block("fast", "Test Failed", "/r/test/test_x.jl:30")),
                                  rows)
        @test !(green)
        @test lines[2] ==
              "RED /r/test/test_x.jl:30 in \"fast\": no row of test/known_red.toml covers it"
        # An interrupted file, the `none:1` of a crashed worker, and a run with no block to read
        # are red.
        @test !(last(KR.verdict(text("Error in testset test_x:\nInterrupted\n"), rows)))
        @test !(last(KR.verdict(text(block("slow solve", "Error During Test", "none:1")),
                                rows)))
        lines, green = KR.verdict(text(), rows)
        @test !(green) && startswith(only(lines), "RED the run failed")
        # The tee writes every byte to the stream and to the buffer.
        out, buffer = IOBuffer(), IOBuffer()
        t = IOContext(KR.Tee(out, buffer), :color => true)
        print(t, "ab", 'c')
        write(t, 0x64)
        @test String(take!(out)) == "abcd" && String(take!(buffer)) == "abcd"
    end

    @testset "run_files parses its options and its watchdog kills on time" begin
        RF = DriverSeam.RunFiles
        o = RF.parse_args(["--testset=^a b\$", "--time-depth=2", "--timeout=30",
                           "--done=/x", "test_x.jl"])
        @test o.testset == r"^a b$" && o.time_depth == 2 && o.timeout == 30.0
        @test o.done == "/x" && o.names == ["test_x.jl"]
        o = RF.parse_args(["test_x.jl"])
        @test o.testset === nothing && o.time_depth == 0 && o.timeout === nothing
        @test_throws ErrorException RF.parse_args(["--time-depth=-1", "test_x.jl"])
        @test_throws ErrorException RF.parse_args(["--timeout=0", "test_x.jl"])
        @test_throws ErrorException RF.parse_args(["--timeout=soon", "test_x.jl"])
        @test_throws ErrorException RF.parse_args(["--bogus", "test_x.jl"])
        @test_throws ErrorException RF.parse_args(["--testset=a"])
        # `--timeout` kills the process that runs the driver, so a REPL may not pass it.
        @test_throws ErrorException RF.main(["--timeout=5", "test_x.jl"])

        # The watchdog kills a stand-in for the driver and writes the verdict after the lines
        # of the files that finished.
        dir = mktempdir()
        done = joinpath(dir, "done.txt")
        target = run(`sleep 60`; wait = false)
        dog = RF.start_watchdog(0.5, "test_x.jl", ["RESULT test_a.jl pass=1"], done;
                                pid = getpid(target), out = devnull)
        wait(target)
        wait(dog)
        @test target.termsignal == 9
        @test readlines(done) ==
              ["RESULT test_a.jl pass=1", "RESULT test_x.jl timeout seconds=0.5",
               "DONE files=2 status=timeout"]
        @test !isfile(done * ".partial")
        # A file that ends in time stops its watchdog, and nothing is killed or written.
        rm(done)
        target = run(`sleep 60`; wait = false)
        dog = RF.start_watchdog(30, "test_x.jl", String[], done; pid = getpid(target),
                                out = devnull)
        RF.stop_watchdog(dog, done)
        @test process_running(target)
        @test !isfile(done) && !isfile(done * ".partial")
        kill(target)
        wait(target)
    end

    @testset "run_files takes a prefix, a case, a resume log and the CI flags" begin
        RF = DriverSeam.RunFiles
        o = RF.parse_args(["--ci", "--case=^bl_", "--resume=/r", "test_x.jl"])
        @test o.ci && o.case == "^bl_" && o.resume == "/r"
        o = RF.parse_args(["test_x.jl"])
        @test !(o.ci) && o.case === nothing && o.resume === nothing
        @test_throws Exception RF.parse_args(["--case=(", "test_x.jl"])

        dir = mktempdir()
        for f in ("test_01_a.jl", "test_01b_b.jl", "test_02_c.jl")
            write(joinpath(dir, f), "")
        end
        @test RF.resolve("test_02"; dir) == joinpath(dir, "test_02_c.jl")
        @test RF.resolve("test_01b"; dir) == joinpath(dir, "test_01b_b.jl")
        # `test_01` names two files, and a miss lists the files of its number.
        err = try
            RF.resolve("test_01"; dir)
        catch e
            e
        end
        @test occursin("Candidates: test_01_a.jl, test_01b_b.jl", err.msg)
        err = try
            RF.resolve("test_02_wrong.jl"; dir)
        catch e
            e
        end
        @test occursin("Candidates: test_02_c.jl", err.msg)

        log = joinpath(dir, "log.txt")
        write(log, "noise\nRESULT test_01_a.jl pass=1 fail=0\nTIME test_02_c.jl x 1.0\n")
        @test RF.finished_files(log) == Set(["test_01_a.jl"])
        @test isempty(RF.finished_files(joinpath(dir, "absent.txt")))
    end

    @testset "a Manifest move without a baseline refresh is named" begin
        MP = ProvenanceSeam.ManifestProvenance
        dir = mktempdir()
        write(joinpath(dir, "Manifest.toml"), """
              julia_version = "1.13.2"
              [[deps.JET]]
              version = "0.12.3"
              [[deps.JuliaSyntax]]
              version = "1.0.2"
              """)
        write(joinpath(dir, "jet_baseline.toml"),
              "[provenance]\njulia = \"1.13.2\"\njet = \"0.12.1\"\nfilter = \"x\"\n")
        write(joinpath(dir, "size_baseline.toml"),
              "[provenance]\njulia = \"1.13.2\"\njulia_syntax = \"1.0.2\"\n")
        write(joinpath(dir, "expansion_bound.toml"),
              "[provenance]\njulia = \"1.13.1\"\ncode_complexity = \"0.2.0\"\n")
        @test MP.mismatches(dir) ==
              ["code_health/expansion_bound.toml: julia is \"1.13.1\", the Manifest holds \"1.13.2\"",
               "code_health/jet_baseline.toml: jet is \"0.12.1\", the Manifest holds \"0.12.3\""]
        # The live tree is consistent.
        @test isempty(MP.mismatches())
    end

    @testset "a Julia slot is taken once, waits for a holder, and frees a dead one" begin
        JS = GateScopeSeam.CodeHealth.JuliaSlot
        dir = mktempdir()
        path = joinpath(dir, "slot-1.pid")
        @test JS.try_take(path, "first")
        @test JS.holder(path) == (getpid(), "first")
        @test !(JS.try_take(path, "second"))
        # A slot whose holder has exited is free.
        p = run(`true`; wait = false)
        gone = getpid(p)
        wait(p)
        write(path, "$gone gone\n")
        @test JS.try_take(path, "third")
        JS.release(path)
        @test !isfile(path)
        # A slot of another live process is not released by this one.
        write(path, "1 init\n")
        JS.release(path)
        @test isfile(path)

        withenv("PO_JULIA_SLOT_DIR" => dir, "PO_JULIA_SLOTS" => "1",
                "PO_JULIA_MIN_FREE_GB" => "0", "CI" => nothing,
                "PO_JULIA_SLOT_HELD" => nothing) do
            rm(path)
            @test JS.active()
            @test JS.with_slot(() -> (ENV["PO_JULIA_SLOT_HELD"], isfile(path)), "probe") ==
                  (path, true)
            @test !isfile(path) && !haskey(ENV, "PO_JULIA_SLOT_HELD")
            # Under a holder, and on CI, the lock is off.
            withenv("PO_JULIA_SLOT_HELD" => path) do
                @test !(JS.active())
                @test JS.acquire("child") === nothing
            end
            withenv("CI" => "true") do
                @test !(JS.active())
            end
            # A busy slot makes the run wait and say so.
            write(path, "1 init\n")
            io = IOBuffer()
            t = @async JS.acquire("waiter"; poll = 0.05, io)
            sleep(0.3)
            @test !istaskdone(t)
            rm(path)
            @test fetch(t) == path
            @test occursin("SLOT wait: waiter", String(take!(io)))
            JS.release(path)
        end
        @test JS.mem_available_gb() > 0
    end

    @testset "doctest_files selects the docstrings of the named files" begin
        DF = DocFilesSeam.DoctestFiles
        dir = mktempdir()
        a = joinpath(dir, "a.jl")
        b = joinpath(dir, "sub", "b.jl")
        block = "```jldoctest\njulia> 1 + 1\n2\n```"
        put(dir, "a.jl", """
            "f of an Int.\n\n$block\n"
            f(x::Int) = x
            "S.\n\n$block\n\n$block\n"
            struct S end
            """)
        put(dir, "sub/b.jl", """
            "f of a Float64.\n\n$block\n"
            f(x::Float64) = x
            "g, with no block."
            g() = 0
            """)
        fixture = Module(:DocFixture)
        Base.include(fixture, a)
        Base.include(fixture, b)
        binding(name) = Base.Docs.Binding(fixture, name)

        # A binding documented in two files keeps the methods of the named file alone.
        scope, docstrings, blocks, files = DF.scope_module(fixture, [a])
        meta = Base.Docs.meta(scope)
        @test Set(keys(meta)) == Set([binding(:f), binding(:S)])
        @test meta[binding(:f)].order == [Tuple{Int}]
        @test (docstrings, blocks, files) == (2, 3, Set([realpath(a)]))
        # The meta of the fixture is left as it was.
        @test length(Base.Docs.meta(fixture)[binding(:f)].order) == 2

        # A directory names every file below it.
        scope, docstrings, blocks, files = DF.scope_module(fixture, [joinpath(dir, "sub")])
        @test Set(keys(Base.Docs.meta(scope))) == Set([binding(:f), binding(:g)])
        @test (docstrings, blocks, files) == (2, 1, Set([realpath(b)]))
        @test DF.in_scope(b, [joinpath(dir, "sub", "..", "sub")])
        @test !(DF.in_scope(a, [joinpath(dir, "sub")]))
        @test !(DF.in_scope(joinpath(dir, "gone.jl"), [dir]))
        # A directory whose name is a prefix of another does not reach into it.
        put(dir, "subway/c.jl", "c = 0\n")
        @test !(DF.in_scope(joinpath(dir, "subway", "c.jl"), [joinpath(dir, "sub")]))

        c = DF.parse_command(["--fix", a, a])
        @test (c.paths, c.fix, c.all) == ([a], true, false)
        c = DF.parse_command(["--all"])
        @test c.all && isempty(c.paths)
        @test_throws ErrorException DF.parse_command(["--all", a])
        for bad in (["--against"], ["--bogus"], [joinpath(dir, "gone.jl")])
            @test_throws ErrorException DF.parse_command(bad)
        end
        repo, _, _, _ = fixture_repo()
        c = DF.parse_command(["--against", "trunk"]; root = repo)
        @test c.paths == [joinpath(repo, "src/a/new.jl"), joinpath(repo, "src/a/one.jl")]

        # The setup is the text of the CI job, without its last call.
        setup = DF.workflow_setup()
        @test occursin("DocMeta.setdocmeta!(PortfolioOptimisers, :DocTestSetup", setup)
        @test occursin("set_show_nothing_fields!(true)", setup)
        @test !(occursin("doctest(PortfolioOptimisers", setup))
        @test occursin("const DOCTEST_FILTERS", setup)
        @test_throws ErrorException DF.workflow_setup("jobs: {}\n")
        @test_throws ErrorException DF.workflow_setup("- name: Run doctest\n  run: julia --project=docs -e 'x = 1'\n")
    end

    @testset "sweep_check names the declarations the package censuses read" begin
        #=
        `test/test_26_docs.jl` ("every choice is catalogued") and
        `test/test_61_interfaces_section_census.jl` need the package loaded, so an agent met them
        only after the targeted test files ran. `sweep_check.jl` reads the branch's diff, and an
        added `struct`, `abstract type`, `export` or `public` names the command that runs both.
        =#
        SC = SweepCheckSeam
        diff = ["+++ b/src/a.jl", "@@ -1,0 +1,9 @@",
                "+abstract type AbstractThing <: AbstractEstimator end",
                "+@kwdef struct Thing{T} <: AbstractThing", "+mutable struct State",
                "+    @concrete struct Inner <: AbstractThing", "+export Thing, State",
                "+public AbstractThing", "-struct Removed end", " struct Context end",
                "+    x::Int", "+export Thing"]
        @test SC.declarations_in(diff) ==
              ["`abstract type AbstractThing`", "`struct Thing`", "`struct State`",
               "`struct Inner`", "an `export` line", "a `public` line"]
        # A field, a call and a comment name nothing, and nor does a file with no addition.
        @test isempty(SC.declarations_in(["+    s::Int", "+structure(x) = x",
                                          "+# struct Foo", "+exported = 1"]))
        @test isempty(SC.declarations_in(String[]))
    end
    @testset "the body of a sweep sub-issue says the file is not swept yet" begin
        #=
        `code_health/sweep_triage.jl` files a file only when its row reads `swept = false`, and its
        body said "This file is swept." for every one of them. #1717 and the plans of #1752 carried
        the false sentence. The body states the state of the row, and only the arm the map takes.
        =#
        ST = SweepTriageSeam
        for open in (true, false)
            c = ST.Candidate("src/a.jl", ST.ChildMap(9, 423, open, "Risk measures"), 3, 10,
                             1)
            body = ST.body_of(c, "abc")
            @test occursin("This file is not swept yet.", body)
            @test !occursin("This file is swept.", body)
            @test occursin(open ? "after that map was charted" : "after that map closed",
                           body)
        end
    end
    @testset "a sweep title that names no current path is reported and refused" begin
        #=
        Both sweep jobs find a sweep issue by the path its title names. Forty-nine tickets named a
        file by its bare name or by a path that #1058 moved, so the check reported their files as
        unfiled and the triage filed duplicates (#1755). A colon now ends a path, and a title
        whose `.jl` token names no row fails the check and stops the triage.
        =#
        CH = GateScopeSeam.CodeHealth
        @test CH.names_path("Sweep src/a.jl: 3 units", "src/a.jl")
        @test !(CH.names_path("Sweep src/a.jl.orig", "src/a.jl"))
        paths = Set(["src/a.jl", "src/b/c.jl"])
        issues = [(1, true, "Sweep `src/a.jl` into child map 9: Risk measures"),
                  (2, true, "Sweep the pair: `src/old/c.jl` and 02_D.jl, 3 units"),
                  (3, false, "Sweep `src/gone.jl`"),
                  (4, true, "Sweep the views of src/13_Prior: 19 new units")]
        @test CH.stale_titles(issues, paths) ==
              [(2, issues[2][3], ["src/old/c.jl", "02_D.jl"])]

        SC = SweepCheckSeam
        tracker = SC.Tracker(Dict{Int, Tuple{Int, Bool, String}}(), true, issues)
        fs = SC.check_titles(tracker, paths)
        @test length(fs) == 1
        @test fs[1].level === :fail
        @test occursin("issue #2", fs[1].text)
        @test isempty(SC.check_titles(nothing, paths))
        @test isempty(SC.check_titles(tracker, union(paths, ["src/old/c.jl", "02_D.jl"])))

        ST = SweepTriageSeam
        dir = mktempdir()
        write(joinpath(dir, "existing.tsv"),
              join(("$n\t$(open ? "OPEN" : "CLOSED")\t$t" for (n, open, t) in issues),
                   "\n"))
        existing = ST.read_existing(joinpath(dir, "existing.tsv"))
        @test existing == issues
        @test ST.already_filed(existing, "src/a.jl")
        @test !(ST.already_filed(existing, "src/gone.jl"))
        rows = Dict("src/a.jl" => Dict{String, Any}("map" => 9, "swept" => false),
                    "src/b/c.jl" => Dict{String, Any}("map" => 9, "swept" => true))
        map_names = Dict("9" => "Risk measures")
        maps = Dict("9" => ST.ChildMap(9, 423, true, "Risk measures"))
        @test ST.reconcile(rows, map_names, maps, existing[[1, 3, 4]]) === nothing
        err = try
            ST.reconcile(rows, map_names, maps, existing)
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("issue #2 names `src/old/c.jl`, `02_D.jl`", err.msg)
    end
    @testset "retry prints the output of the run that succeeded alone" begin
        #=
        `sweep_issues.sh apply` reads the database id of a new issue through `retry`, because the
        read can answer 404 for a short time (#1713). A failed `gh api` prints the error body to
        stdout, and `retry` printed the output of every run, so after two 404s the id read
        `{…}{…}42` and the attach failed five times. The flaky command here fails twice with a
        404 body on stdout, and a command that always fails prints a body every time.
        =#
        script = """
                 source "\$1"
                 c="\$2/count"
                 echo 0 > "\$c"
                 flaky() {
                     local n=\$(( \$(cat "\$c") + 1 ))
                     echo "\$n" > "\$c"
                     if [ "\$n" -lt 3 ]; then
                         echo '{"message":"Not Found","status":"404"}'
                         return 1
                     fi
                     echo 42
                 }
                 broken() { echo '{"message":"Not Found","status":"404"}'; return 1; }
                 id=\$(retry flaky)
                 echo "id=[\$id] runs=\$(cat "\$c")"
                 if out=\$(retry broken); then echo "broken passed"; else echo "broken=[\$out]"; fi
                 """
        dir = mktempdir()
        sweep_issues = joinpath(@__DIR__, "..", "code_health", "sweep_issues.sh")
        out = read(`bash -c $script _ $sweep_issues $dir`, String)
        @test occursin("id=[42] runs=3", out)
        @test occursin("broken=[]", out)
    end
end
