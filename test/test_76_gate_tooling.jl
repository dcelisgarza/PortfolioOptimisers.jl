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
    pre-push guard of `.pre-commit-config.yaml`.
  - `test/run_files.jl` runs named test files after the preamble and counts each one.
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
worktree under `.claude/worktrees/`.
"""
function stamp(dir::AbstractString, args...; force::Bool = true, env...)
    vars = Pair{String, String}["GATE_STAMP_REF" => "trunk"]
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
                  @testset "inner" begin
                      @test 1 == 1
                      @test 1 == 2
                      @test_broken 1 == 2
                  end
                  @test error("boom")
              end
              """)
        RF.OUT[] = devnull
        ts, seconds = redirect_stdout(devnull) do
            return RF.run_file(path; code)
        end
        RF.OUT[] = stdout
        @test (ts.pass, ts.fail, ts.error, ts.broken) == (2, 1, 1, 1)
        @test seconds >= 0
        @test RF.result_line(ts, 1.0) ==
              "RESULT test_fixture.jl pass=2 fail=1 error=1 broken=1 seconds=1.0"
        @test RF.done_line([ts]) == "DONE files=1 pass=2 fail=1 error=1 broken=1 status=red"
        @test RF.resolve("test_76_gate_tooling.jl") ==
              joinpath(RF.TEST_DIR, "test_76_gate_tooling.jl")
        @test_throws ErrorException RF.resolve("test_no_such_file.jl")

        cov = joinpath(mktempdir(), "x.jl.123.cov")
        write(cov,
              "        - function f(x)\n        3     y = x\n        0     z = 1\n" *
              "        - end\n")
        @test RF.coverage_lines(cov) == (2, [3])
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
        @test (c.paths, c.fix) == ([a], true)
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
        @test !(occursin("doctest(PortfolioOptimisers)", setup))
        @test_throws ErrorException DF.workflow_setup("jobs: {}\n")
        @test_throws ErrorException DF.workflow_setup("- name: Run doctest\n  run: julia --project=docs -e 'x = 1'\n")
    end
end
