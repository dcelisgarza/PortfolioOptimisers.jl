#=
`code_health/size.jl` counts the code lines in a file, and that count IS the size ratchet. ADR 0101
puts the number in a pass rule, so a wrong reading does not report a wrong number: it holds the
wrong files and lets the wrong files grow.

The classification is the whole instrument, and it is not obvious. A line-by-line scan for `"""`
and `#` reads a LONG STRING LITERAL inside a function as a docstring: it opens on the `"""` and
takes every line to the closing one as prose. So `size.jl` reads the source text twice instead --
once as tokens, for the comment bytes, and once as a tree, for the docstring bytes.

A FIELD DOCSTRING is the shape that makes the choice of parser matter. `CodeHealth.isdocstring`
reads a tree from `Meta.parseall`, where a field docstring is a bare string literal, because a
struct body binds nothing for `Core.@doc` to attach to. `JuliaSyntax.parseall(SyntaxNode, ...)`
keeps the `doc` node in both places, so `mark_documentation!` reads one rule and gets both. The
field-docstring case below asserts that, so a parser that stops keeping the node is caught here
rather than in a silently wrong count.

The classifier was wrong on arrival by writing a SECOND rule for that shape and writing it too
wide. It marked a bare string literal as documentation wherever it stood in ANY block, and a
branch value is such a string:

    shape = if isa(X, AbstractArray)
        "a $(ndims(X))-dimensional input of size $(size(X))"
    else
        "the given input"
    end

`src/01_Base/09_ObservationWeights.jl` writes exactly that, and it measured 15 code lines where a
hand count of its source gives 17. The second rule is gone, and `a value string in a branch is
code` below is the regression test for it.

This file is the fixture adapter for the gate, in the sense `test_51_code_health_root_seam.jl`
gives: a gate exercised only by CI running the real script against the real repository is a
counting defect found months late. It writes small trees in a temporary directory, labels every
line of them by hand, and asks the classifier to agree.

`test/Project.toml` carries `JuliaSyntax`, which is what lets this file load the gate at all.
`size.jl` reads the analyser's version with `pkgversion` rather than with `Pkg.dependencies`, so it
loads under any environment that carries `JuliaSyntax` and `TOML`, and the test environment now
carries both.

The load sits OUTSIDE the `@testset`, for the reason `test_49_coverage_attribution_census.jl`
gives: `include` defines methods, and a method defined inside one top-level statement is not
visible to a call in that same statement. The module wrapper keeps the script's own names --
`measure`, `verify`, `render`, `publish` -- out of the worker module.
=#
module SizeClassification
include(joinpath(@__DIR__, "..", "code_health", "size.jl"))
end

@testset "Size classification census: a docstring is prose and a value string is code" begin
    using Test, TOML

    S = SizeClassification
    root = normpath(joinpath(@__DIR__, ".."))

    #=
    One fixture, written to a temporary file and read back. `expected` labels every line of
    `text` in order, so a wrong reading names the line it got wrong rather than a total.
    =#
    function agrees(name, text, expected)
        mktempdir() do dir
            path = joinpath(dir, "fixture.jl")
            write(path, text)
            counts = S.line_counts(path)
            tally = Dict("code" => 0, "doc" => 0, "comment" => 0, "blank" => 0)
            for k in expected
                tally[k] += 1
            end
            for k in ("code", "doc", "comment", "blank")
                if counts[k] != tally[k]
                    @info "$name: $k lines" measured = counts[k] expected = tally[k] text
                end
                @test counts[k] == tally[k]
            end
            @test counts["total"] == length(expected)
        end
    end

    # ------------------------------------------------------- the four kinds, one at a time

    @testset "a docstring on a binding is prose" begin
        text = """
        \"\"\"
            f(x)

        Prose.
        \"\"\"
        f(x) = x
        """
        agrees("binding docstring", text, ["doc", "doc", "blank", "doc", "doc", "code"])
    end

    @testset "a field docstring is prose" begin
        #=
        `Meta.parseall` drops the `Core.@doc` wrapper here and `JuliaSyntax.parseall` keeps
        it, so this passes on one rule. It is asserted rather than assumed: a parser that
        stopped keeping the node would make every field docstring in the library read as
        code, and this library writes most of its prose as field docstrings.
        =#
        text = """
        struct T
            \"\"\"
            The first field.
            \"\"\"
            a::Int
            b::Float64
        end
        """
        agrees("field docstring", text,
               ["code", "doc", "doc", "doc", "code", "code", "code"])
    end

    @testset "a value string in a branch is code" begin
        #=
        The regression test for the arrival defect. Both branch values are string literals
        standing in a block, and both are code.
        =#
        text = """
        function f(X)
            shape = if isa(X, AbstractArray)
                "an array"
            else
                "something else"
            end
            return shape
        end
        """
        agrees("branch value strings", text,
               ["code", "code", "code", "code", "code", "code", "code", "code"])
    end

    @testset "a long string literal in a function is code" begin
        #=
        The other shape a line scan gets wrong. Every line of the triple-quoted literal is
        code, because the literal is a value.
        =#
        text = """
        function message()
            return \"\"\"
                   a literal
                   over three lines
                   \"\"\"
        end
        """
        agrees("long value literal", text, ["code", "code", "code", "code", "code", "code"])
    end

    @testset "both comment forms are comment" begin
        text = """
        # a line comment
        #= a block
           comment over
           three lines =#
        f(x) = x  # a trailing comment
        """
        agrees("comments", text, ["comment", "comment", "comment", "comment", "code"])
    end

    @testset "a blank line inside a docstring is blank" begin
        #=
        The four kinds PARTITION the file, so a line that carries nothing is blank wherever it
        stands. Only `code` binds, so this affects no pass rule; it is stated because the
        partition is asserted over the whole tree below.
        =#
        text = """
        \"\"\"
        Prose.

        More prose.
        \"\"\"
        const A = 1
        """
        agrees("blank inside a docstring", text,
               ["doc", "doc", "blank", "doc", "doc", "code"])
    end

    # ------------------------------------------------------------------ two shared lines

    @testset "code wins a line it shares" begin
        #=
        A docstring and a definition on ONE line is one line, and it counts once. Code wins,
        because the reader must still read the code on it.
        =#
        text = """
        \"\"\"Prose.\"\"\" f(x) = x
        # a comment
        """
        agrees("shared line", text, ["code", "comment"])
    end

    @testset "a file with no trailing newline still ends a line" begin
        agrees("no trailing newline", "f(x) = x\ng(x) = x", ["code", "code"])
        agrees("empty file", "", String[])
        agrees("one newline", "\n", ["blank"])
    end

    # ------------------------------------------------- the partition holds over the tree

    @testset "the four kinds partition every file in scope" begin
        #=
        A property rather than a number, so an ordinary edit under `src/` cannot red it. It is
        the one assertion that reads the real tree, and it costs about five seconds.
        =#
        files = filter(S.CodeHealth.in_scope, S.CodeHealth.tracked_jl_files())
        @test !isempty(files)
        bad_partition = String[]
        bad_total = String[]
        for f in files
            path = joinpath(root, f)
            c = S.line_counts(path)
            if c["code"] + c["doc"] + c["comment"] + c["blank"] != c["total"]
                push!(bad_partition, f)
            end
            if c["total"] != countlines(path)
                push!(bad_total, f)
            end
        end
        @test isempty(bad_partition)
        @test isempty(bad_total)
    end

    # ------------------------------------------------------------------- the pass rule

    @testset "the ceiling is the greater of the threshold and the recorded number" begin
        #=
        ADR 0101's whole pass rule, in the five cases that separate it from a plain ratchet. A
        plain ratchet would red the first case, which is the noise issue #336 measured.
        =#
        row(n) = Dict("code" => n, "doc" => 0, "comment" => 0, "blank" => 0, "total" => n)
        limit = 500

        @test S.ceiling(nothing, limit) == limit
        @test S.ceiling(row(100), limit) == limit
        @test S.ceiling(row(800), limit) == 800

        rises(rec, meas) = S.size_rises(rec, meas, limit)

        # Under the threshold a file is free, however far it grows.
        @test isempty(rises(Dict("a.jl" => row(100)), Dict("a.jl" => row(499))))
        # Crossing the threshold trips, against the threshold.
        crossed = rises(Dict("a.jl" => row(100)), Dict("a.jl" => row(700)))
        @test length(crossed) == 1
        @test only(crossed).old == limit
        @test only(crossed).new == 700
        @test only(crossed).metric == "code"
        # Over the threshold a file is held where it stands, to the line.
        held = rises(Dict("a.jl" => row(799)), Dict("a.jl" => row(800)))
        @test length(held) == 1
        @test only(held).old == 799
        # A fall is green, and it is what a bare refresh records.
        @test isempty(rises(Dict("a.jl" => row(799)), Dict("a.jl" => row(600))))
        # An added file has no row, so it takes the threshold. ADR 0074's entry test.
        @test length(rises(Dict{String, Any}(), Dict("a.jl" => row(800)))) == 1
        @test isempty(rises(Dict{String, Any}(), Dict("a.jl" => row(500))))
    end

    # --------------------------------------------------------- the two committed files

    @testset "the threshold and the baseline are well formed" begin
        #=
        Structural facts about the two committed files, and no measurement of the tree: the
        gate itself compares the baseline against the tree, and asserting that here would red
        the suite for every source edit that has not been followed by a refresh.
        =#
        rulings = TOML.parsefile(joinpath(root, "code_health", "rulings.toml"))
        @test haskey(rulings, "size")
        @test rulings["size"]["code_lines"] isa Int
        @test rulings["size"]["code_lines"] > 0
        # The threshold lives OUTSIDE `[thresholds]`, whose numbers never reach a pass rule.
        @test !haskey(rulings["thresholds"], "code_lines")

        baseline = TOML.parsefile(joinpath(root, "code_health", "size_baseline.toml"))
        @test haskey(baseline["provenance"], "julia_syntax")
        rows = baseline["file"]
        @test !isempty(rows)
        bad_keys = [f
                    for (f, r) in rows
                    if !all(k -> haskey(r, k) && r[k] isa Int, S.ROW_NUMBERS)]
        @test isempty(bad_keys)
        bad_sum = [f
                   for (f, r) in rows
                   if r["code"] + r["doc"] + r["comment"] + r["blank"] != r["total"]]
        @test isempty(bad_sum)
        @test all(f -> S.CodeHealth.in_scope(f), keys(rows))
    end
end
