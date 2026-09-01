#=
The code-health gates read a tree through a `root`, and a fixture tree is a tree.

`code_health/CodeHealth.jl` and the four entry scripts reached `REPO_ROOT` directly, and every
`measure` was zero-arity. That interface admits exactly one adapter, the live checkout, so the
counting was checked only by CI running the real script against the real repository. Each reader
now takes `root` and a `files` list, with the live values as the defaults, and this file is the
second adapter: a small tree in a temporary directory, and a small `lcov.info` beside it.

`docs/generate_sitemap.jl` and `test/test_50_docs_sitemap.jl` are the same pair of adapters at the
same kind of seam, and the reason given there holds here. A gate exercised only by CI is a counting
defect found months late.

`test/Project.toml` carries `TOML` and nothing else the gate needs, so this file drives
`CodeHealth.jl` and `code_health/coverage.jl`. The other three entry scripts load `CodeComplexity`,
`JuliaSyntax` or `JET`, which live in `code_health/Project.toml` alone. They take the same `root`,
`files` and `declaring` parameters, and a fixture test for them needs the gate's own environment.

The load sits OUTSIDE the `@testset`, for the reason `test_49_coverage_attribution_census.jl`
gives: `include` defines methods, and a method defined inside one top-level statement is not
visible to a call in that same statement. The module wrapper keeps the script's own names --
`measure`, `verify`, `render`, `publish` -- out of the worker module.
=#
module CoverageSeam
include(joinpath(@__DIR__, "..", "code_health", "coverage.jl"))
end

using Test, TOML

#=
One line per element, so an element's index in this vector IS its line number in the written file.
The lcov fixture names three of those numbers, and each number is asserted against the text of the
line it stands for, so an edit here fails loudly rather than moving a miss in silence.
=#
const FIXTURE_LINES = ["\"\"\"", "    kept(x)", "", "A documented function.", "\"\"\"",
                       "function kept(x)", "    return x + 1", "end", "", "\"\"\"",
                       "    dropped(x)", "", "A short form that declares a return type.",
                       "\"\"\"", "dropped(x)::Bool = x > 0", "", "const SETTING = 3", "",
                       "precompile(kept, (Int,))"]

"""
The line inside `kept`. It is uncovered in both lcov sessions.
"""
const MISS_IN_KEPT = 7

"""
The line the two lcov sessions disagree about. The hits add up, so it is covered.
"""
const MERGED_LINE = 15

"""
The line inside no named definition. It is uncovered, and it belongs to `<toplevel>`.
"""
const MISS_AT_TOPLEVEL = 19

const DECLARES_SRC = """
macro fixture_declare(ex)
    return esc(ex)
end

macro fixture_unused(ex)
    return esc(ex)
end
"""

const USES_SRC = """
@fixture_declare struct Holder
    x::Int
end

struct Plain
    @fprop y::Int
end
"""

"""
The files of the fixture tree that lie under the measured roots, in the order `git ls-files` sorts
them.
"""
const FIXTURE_SCOPED = ["src/Declares.jl", "src/Fixture.jl", "src/Quiet.jl", "src/Uses.jl"]

"""
    write_code_health_fixture(dir) -> String

A stand-in for the repository: four files under `src/`, one outside the measured roots, and the
`lcov.info` the test job would have written.

The LCOV file holds two `SF:` records for the same source file, one relative and one absolute,
which is what a second upload session writes. `src/Foreign.jl` and `test/helper.jl` are named by no
`files` list, so they are foreign records rather than failures.
"""
function write_code_health_fixture(dir)
    mkpath(joinpath(dir, "src"))
    mkpath(joinpath(dir, "docs"))
    write(joinpath(dir, "src", "Fixture.jl"), join(FIXTURE_LINES, "\n") * "\n")
    write(joinpath(dir, "src", "Quiet.jl"), "quiet() = nothing\n")
    write(joinpath(dir, "src", "Declares.jl"), DECLARES_SRC)
    write(joinpath(dir, "src", "Uses.jl"), USES_SRC)
    write(joinpath(dir, "docs", "notes.jl"), "# Outside the measured roots.\n")
    lcov = """
    SF:src/Fixture.jl
    DA:$(MISS_IN_KEPT),0
    DA:8,1
    DA:$(MERGED_LINE),0
    DA:17,1
    end_of_record
    SF:$(joinpath(dir, "src", "Fixture.jl"))
    DA:$(MERGED_LINE),2
    DA:$(MISS_AT_TOPLEVEL),0
    end_of_record
    SF:src/Foreign.jl
    DA:1,0
    end_of_record
    SF:test/helper.jl
    DA:1,1
    end_of_record
    """
    write(joinpath(dir, "lcov.info"), lcov)
    return dir
end

#=
`verify` writes a `::error` annotation and a step summary when it runs on CI, and this fixture's
failures are not the repository's. Both channels are switched off for the duration of a call, so a
run of this file annotates nothing and appends nothing.
=#
quietly(f) = withenv(f, "GITHUB_ACTIONS" => nothing, "GITHUB_STEP_SUMMARY" => nothing)

@testset "Code-health root seam: a fixture tree drives the gate" begin
    using Test, TOML

    CH = CoverageSeam.CodeHealth
    CV = CoverageSeam

    @test length(FIXTURE_LINES) == 19
    @test FIXTURE_LINES[MISS_IN_KEPT] == "    return x + 1"
    @test FIXTURE_LINES[MERGED_LINE] == "dropped(x)::Bool = x > 0"
    @test FIXTURE_LINES[MISS_AT_TOPLEVEL] == "precompile(kept, (Int,))"

    @testset "CodeHealth reads the tree it is given" begin
        mktempdir() do dir
            write_code_health_fixture(dir)

            # A relative path is read from `root`. Nothing here touches the live checkout.
            @test CH.parse_file("src/Fixture.jl"; root = dir) isa Expr
            @test CH.documented_units("src/Fixture.jl"; root = dir) == 2
            @test CH.documented_units("src/Quiet.jl"; root = dir) == 0

            # `declaring` is a parameter for the same reason `root` is: a fixture tree holds no
            # `src/01_Base.jl`, so a hard-coded list would admit the live checkout alone.
            @test CH.declared_macros("src/Declares.jl"; root = dir) ==
                  ["@fixture_declare", "@fixture_unused"]

            called = CH.called_macros("src/Uses.jl"; root = dir)
            @test "@fixture_declare" in called
            # A field-prefix macro sits inside a struct body, and the walk prunes there.
            @test !("@fprop" in called)

            # The intersection of declared and called. `@fixture_unused` is declared and never
            # called, so it is not a Declaration Macro of this tree.
            @test CH.declaration_macros(["src/Uses.jl"]; root = dir,
                                        declaring = ["src/Declares.jl"]) ==
                  ["@fixture_declare"]

            # `git ls-files` runs in `root`, and `source_files` drops what is out of scope.
            run(pipeline(`git init -q $dir`; stdout = devnull, stderr = devnull))
            run(pipeline(`git -C $dir add -A`; stdout = devnull, stderr = devnull))
            @test CH.tracked_jl_files(; root = dir) ==
                  vcat(["docs/notes.jl"], FIXTURE_SCOPED)
            @test CH.source_files(; root = dir) == FIXTURE_SCOPED

            # The fixture has no commit, so the reader gives its own answer rather than the live
            # checkout's commit.
            @test CH.git_short_commit(; root = dir) == "unknown"
        end
    end

    @testset "The live checkout is still the default adapter" begin
        # The second adapter must not move the first. Every default is the live value, which is
        # what lets the four entry scripts go on calling `measure()` with no argument.
        live = CH.source_files()
        @test live == CH.source_files(; root = CH.REPO_ROOT)
        @test "src/PortfolioOptimisers.jl" in live
        @test all(f -> startswith(f, "src/") || startswith(f, "ext/"), live)
    end

    @testset "Coverage counts and attributes from a fixture" begin
        mktempdir() do dir
            write_code_health_fixture(dir)

            # `COVERAGE_LCOV` overrides the path, so it is cleared before the default is asserted.
            withenv("COVERAGE_LCOV" => nothing) do
                @test CV.lcov_path(; root = dir) == joinpath(dir, "lcov.info")
            end
            @test CV.relative(joinpath(dir, "src", "Fixture.jl"); root = dir) ==
                  "src/Fixture.jl"

            files = ["src/Fixture.jl", "src/Quiet.jl"]
            m = CV.measure(; root = dir, files = files, lcov = joinpath(dir, "lcov.info"))

            @test m.files == files
            # Five `DA` lines survive the merge of the two sessions.
            @test m.numbers["src/Fixture.jl"].lines == 5
            @test m.numbers["src/Fixture.jl"].misses == [MISS_IN_KEPT, MISS_AT_TOPLEVEL]
            # The two sessions add up, so the line one of them covered is covered.
            @test !(MERGED_LINE in m.numbers["src/Fixture.jl"].misses)
            @test m.numbers["src/Fixture.jl"].by_definition ==
                  Dict("kept" => 1, CV.TOPLEVEL => 1)

            # A file with no lcov record records zero of both, and is not a failure.
            @test m.numbers["src/Quiet.jl"].lines == 0
            @test isempty(m.numbers["src/Quiet.jl"].misses)

            # A record naming a file outside the list is foreign, and costs the gate nothing.
            @test m.foreign == ["src/Foreign.jl", "test/helper.jl"]

            @test CV.rows(m)["src/Fixture.jl"] == Dict("lines" => 5, "misses" => 2)
        end
    end

    @testset "Coverage renders and verifies against a fixture baseline" begin
        mktempdir() do dir
            write_code_health_fixture(dir)
            files = ["src/Fixture.jl", "src/Quiet.jl"]
            m = CV.measure(; root = dir, files = files, lcov = joinpath(dir, "lcov.info"))

            baseline_row(l, ms) = Dict("lines" => l, "misses" => ms)
            green = Dict("file" => Dict("src/Fixture.jl" => baseline_row(5, 2),
                                        "src/Quiet.jl" => baseline_row(0, 0)))
            tripped = Dict("file" => Dict("src/Fixture.jl" => baseline_row(5, 1),
                                          "src/Quiet.jl" => baseline_row(0, 0)))
            entering = Dict("file" => Dict("src/Quiet.jl" => baseline_row(0, 0)))
            no_rulings = Dict{String, Any}()

            # A baseline that records what the tree measures is green, and the rendered text is
            # readable TOML carrying those numbers.
            failures, provenance_ok = quietly() do
                return CV.verify(m, green; rulings = no_rulings)
            end
            @test isempty(failures)
            @test provenance_ok
            parsed = TOML.parse(CV.render(m, green, false; rulings = no_rulings))
            @test parsed["file"]["src/Fixture.jl"]["misses"] == 2
            @test parsed["file"]["src/Fixture.jl"]["lines"] == 5
            @test parsed["file"]["src/Quiet.jl"]["misses"] == 0
            @test haskey(parsed, "provenance")

            # A miss count above the recorded one trips the ratchet, and a bare refresh refuses to
            # record the rise.
            failures = first(quietly() do
                                 return CV.verify(m, tripped; rulings = no_rulings)
                             end)
            @test any(f -> occursin("coverage ratchet tripped on 1 file(s)", f), failures)
            @test_throws CH.RefreshRefused CV.render(m, tripped, false;
                                                     rulings = no_rulings)
            accepted = TOML.parse(CV.render(m, tripped, true; rulings = no_rulings))
            @test accepted["file"]["src/Fixture.jl"]["misses"] == 2

            # A file with no row enters terminal, so a file entering with an unaccounted miss is
            # refused by name.
            refusal = try
                CV.render(m, entering, true; rulings = no_rulings)
                nothing
            catch e
                e
            end
            @test refusal isa CH.RefreshRefused
            @test occursin("src/Fixture.jl enters with 2 uncovered line(s)",
                           refusal.message)

            # Two Coverage Exemptions account for both lines, so the same file enters.
            exempt(defn, n) = Dict("path" => "src/Fixture.jl", "definition" => defn,
                                   "misses" => n, "rationale" => "fixture")
            rulings = Dict("coverage_exemption" =>
                               [exempt("kept", 1), exempt(CV.TOPLEVEL, 1)],
                           "rationale" => Dict("fixture" => Dict("text" => "A fixture.")))
            @test CV.residue(m, rulings, "src/Fixture.jl") == 0
            @test CV.is_terminal(m, rulings, "src/Fixture.jl")
            entered = TOML.parse(CV.render(m, entering, true; rulings = rulings))
            @test entered["file"]["src/Fixture.jl"]["misses"] == 2

            # A claim above the truth is stale, and the gate holds a claim in both directions.
            stale = Dict("coverage_exemption" => [exempt("kept", 3)],
                         "rationale" => Dict("fixture" => Dict("text" => "A fixture.")))
            bad = CV.exemption_failures(m, stale)
            @test length(bad) == 1
            @test occursin("claims 3 uncovered line(s), and 1 are uncovered", only(bad))

            # A Coverage Exemption must cite a Rationale the file defines.
            uncited = Dict("coverage_exemption" =>
                               [Dict("path" => "src/Fixture.jl", "definition" => "kept",
                                     "misses" => 1)])
            failures = first(quietly() do
                                 return CV.verify(m, green; rulings = uncited)
                             end)
            @test any(f -> occursin("unknown rationale", f), failures)
        end
    end
end
