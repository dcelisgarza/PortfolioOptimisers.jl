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
  - `test/run_files.jl` runs named test files after the preamble and counts each one. It prunes
    a file to the test sets `--testset` selects, times the sets, bounds a file with a watchdog,
    and names a red that `test/known_red.toml` lists.
  - `code_health/sweep_check.jl` names `code_health/census_preview.jl` when a branch adds a
    declaration that the two package-level censuses read. The test checks which diff lines name one.
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
                  # `@test_throws` returns the result that `record` returns (#1475).
                  r = @test_throws ErrorException error("x")
                  @test r.value isa ErrorException
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
        @test (ts.pass, ts.fail, ts.error, ts.broken) == (4, 1, 1, 1)
        @test seconds >= 0
        @test RF.result_line(ts, 1.0) ==
              "RESULT test_fixture.jl pass=4 fail=1 error=1 broken=1 seconds=1.0"
        @test RF.done_line([ts]) == "DONE files=1 pass=4 fail=1 error=1 broken=1 status=red"
        @test RF.resolve("test_76_gate_tooling.jl") ==
              joinpath(RF.TEST_DIR, "test_76_gate_tooling.jl")
        @test_throws ErrorException RF.resolve("test_no_such_file.jl")

        cov = joinpath(mktempdir(), "x.jl.123.cov")
        write(cov,
              "        - function f(x)\n        3     y = x\n        0     z = 1\n" *
              "        - end\n")
        @test RF.coverage_lines(cov) == (2, [3])
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
end
