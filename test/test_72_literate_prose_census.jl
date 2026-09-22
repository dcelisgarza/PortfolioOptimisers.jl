#=
The prose of an example or a user-guide page is written for a reader, and
`.github/instructions/julia-prose.instructions.md` states what it must do. Part of that
rule is a list: a character, a word, a phrase, a heading shape. A list is read by a parser, and a
rule a parser reads is a rule a review does not have to hold in its head.

This file is the gate for the readable part. `code_health/literate_prose.jl` is the reader, and it
is included rather than copied, so a rewrite session's scan and this census cannot drift. A rewrite
session runs

    julia --project=code_health code_health/literate_prose.jl scan <its pages>

before and after its rewrite, and pastes the row the scan prints.

The baseline is PER PAGE and never a library total. Fourteen rewrite tickets run in parallel, and a
library total is one number that two sessions each lower to a different value, after which a merge
keeps one of the two writes. A per-page row is touched by one session alone.

The rules a parser cannot read hold by review: rules 10, 11, 27, 28 and 32 of the `unslop` skill,
and the self-audit that closes a rewrite. This census makes no claim about them.

Five words of the lists are absent from the reader, and the instruction file's *The Gate* section
says why: this library writes "the expected returns vector", "a Pareto surface", "a feature matrix"
and "a leveraged portfolio", and each of those is the concrete word the rule asks for. A census
over the lists as written could never reach zero, and a gate that can never be met is a gate
nobody pays.

The include sits OUTSIDE the `@testset`, for the reason `test_49_coverage_attribution_census.jl`
gives: `include` defines methods, and a method defined inside one top-level statement is not
visible to a call in that same statement. The module wrapper keeps the reader's names out of the
worker module.
=#
module LiterateProse
include(joinpath(@__DIR__, "..", "code_health", "literate_prose.jl"))
end

@testset "Literate prose census: every page is at or below its row" begin
    using Test

    P = LiterateProse
    root = normpath(joinpath(@__DIR__, ".."))

    #=
    One prose line, written into a Literate page of its own and read back through the whole reader.
    A fixture therefore exercises the comment stripping, the fence stripping and the counters in
    one call, rather than the counters alone.
    =#
    glossary = P.glossary_pattern(P.glossary_terms(; root = root))
    function row_of(body)
        mktempdir() do dir
            path = joinpath(dir, "page.jl")
            write(path, "#=\n" * body * "\n=#\n")
            return P.counts(path; glossary = glossary)
        end
    end
    count_of(body, column) = row_of(body)[column]

    @testset "the reader takes the lines Literate renders and nothing else" begin
        text = """
        #=
        ```@meta
        Description = "A page that is about one thing."
        ```

        # A heading

        A paragraph, and an em dash — inside a fenced block below.

        ```julia
        w = 1 # an em dash — in a code cell
        ```
        =#

        w = 1 #src
        ## A code comment, which renders inside a cell.
        # A rendered line with `an em dash — in a code span`.
        w = 2
        """
        mktempdir() do dir
            path = joinpath(dir, "page.jl")
            write(path, text)
            lines = P.prose(path)
            joined = join(lines, "\n")
            @test occursin("A page that is about one thing.", joined)
            @test occursin("# A heading", joined)
            @test occursin("A paragraph, and an em dash", joined)
            @test occursin("A rendered line with", joined)
            # The three lines that never reach a reader: the `#src` line, the `##` code comment,
            # and the body of the fenced block.
            @test !occursin("#src", joined)
            @test !occursin("code comment", joined)
            @test !occursin("code cell", joined)
            # The em dash inside a code span is code, and the one in the paragraph is prose.
            @test P.counts(path; glossary = glossary)["emdash"] == 1
        end
    end

    @testset "the characters of rule 13 and rule 19" begin
        @test count_of("The fold, the fit — and the read.", "emdash") == 1
        @test count_of("The Bayes–Stein estimator.", "endash") == 1
        @test count_of("A hyphenated Bayes-Stein estimator.", "endash") == 0
        @test count_of("She said “yes” to it.", "curly") == 2
        @test count_of("She said \"yes\" to it.", "curly") == 0
    end

    @testset "the word lists of rules 7, 8, 9, 23, 26 and 31" begin
        @test count_of("The split matters, not just the window.", "notjust") == 1
        @test count_of("The split matters, and the window matters.", "notjust") == 0
        @test count_of("A crucial step that showcases the interplay.", "aivocab") == 3
        @test count_of("The prior serves as the baseline.", "fancy_is") == 1
        # Rule 8 names "features" too, and this library writes "a feature matrix".
        @test count_of("The estimator features a feature matrix.", "fancy_is") == 0
        @test count_of("In order to fit it, we split first.", "filler") == 1
        @test count_of("To fit it, we split first.", "filler") == 0
        @test count_of("The substrate of the paradigm.", "metaphor") == 2
        # Rule 26 names "vector", "surface", "primitive", "harness" and "ratchet", and this
        # library writes the first two in their concrete sense.
        @test count_of("The expected returns vector traces a Pareto surface.",
                       "metaphor") == 0
        @test count_of("We utilise it to facilitate numerous fits.", "plainword") == 3
        # Rule 31 names "leverage", which is a position in this library.
        @test count_of("A leveraged portfolio and its leverage.", "plainword") == 0
    end

    @testset "the shapes of rules 16, 17 and 18" begin
        @test count_of("- **Speed:** the fit is faster.", "bold_label") == 1
        @test count_of("- **Speed**: the fit is faster.", "bold_label") == 1
        # Rule 16 permits a bold lead-in that ends in a period and carries new detail.
        @test count_of("- **Speed.** The fit takes half the time.", "bold_label") == 0
        @test count_of("## Hierarchical Risk Parity", "title_case") == 1
        @test count_of("## Hierarchical risk parity", "title_case") == 0
        # A proper noun in a sentence-case heading is not title case.
        @test count_of("## The Black-Litterman prior", "title_case") == 0
        @test count_of("## 2. The gain, measured over ten folds", "title_case") == 0
        @test count_of("✅ The run finished.", "emoji") == 1
        # A mathematical symbol is not a decorative emoji.
        @test count_of("The weights are a ≈ b.", "emoji") == 0
    end

    @testset "the words this repository owns" begin
        @test count_of("The Coverage Universe holds the rows.", "glossary") == 1
        # The rule asks for the plain phrase, in lower case, defined where it first appears.
        @test count_of("the assets with enough observations, the coverage universe",
                       "glossary") == 0
        @test count_of("The seam hands its carrier to the host at read-out.",
                       "mechanism") == 4
        @test count_of("See § 3 for the fold.", "mechanism") == 1
        @test count_of("The two runs agree to the bit, by construction.", "verdict") == 3
        # The identity matrix is a matrix, and a page that shrinks towards it says so.
        @test count_of("We shrink the covariance towards the identity matrix.",
                       "verdict") == 0
        @test count_of("The largest difference over every fold is zero.", "verdict") == 0
    end

    @testset "the glossary list is read off CONTEXT.md" begin
        terms = P.glossary_terms(; root = root)
        @test "Coverage Universe" in terms
        @test "Asset Panel" in terms
        # A term is multi-word and capitalised throughout. A type name carries no space, and a
        # proper noun with an ordinary word after it is not the glossary's name for a concept.
        @test !any(t -> !occursin(" ", t), terms)
        @test !("Black-Litterman family" in terms)
        @test !isempty(terms)
    end

    @testset "the ratchet binds, and a row prints itself" begin
        #=
        The gate below reds only on a page that rose, so a green suite says nothing about the
        comparison itself. These four cases drive it on rows written by hand.
        =#
        row = Dict{String, Int}(c => 0 for c in P.COLUMNS)
        row["emdash"] = 4
        row["verdict"] = 1
        row[P.WORDS] = 900

        # An absent column is a ceiling of zero, so a page with no row at all must be clean.
        @test P.rises(row, Dict{String, Int}()) ==
              ["emdash" => (0 => 4), "verdict" => (0 => 1)]
        # A count at its ceiling is not a rise, and a count under it is not either.
        @test P.rises(row, Dict("emdash" => 4, "verdict" => 9)) ==
              Pair{String, Pair{Int, Int}}[]
        @test P.rises(row, Dict("emdash" => 3, "verdict" => 1)) == ["emdash" => (3 => 4)]
        # `words` carries no limit, so it never rises.
        big = copy(row)
        big[P.WORDS] = 90_000
        @test P.rises(big, Dict("emdash" => 4, "verdict" => 1, "words" => 900)) ==
              Pair{String, Pair{Int, Int}}[]

        # The row a failing page prints holds its counts above zero, and `words` last.
        @test P.row_text("examples/x.jl", row) ==
              "\"examples/x.jl\" = { emdash = 4, verdict = 1, words = 900 }"
        # A page whose every counted rule is at zero has an empty binding, which is what tells the
        # gate to ask for its row to be deleted.
        clean = Dict{String, Int}(c => 0 for c in P.COLUMNS)
        clean[P.WORDS] = 500
        @test isempty(P.binding(clean))
        @test !isempty(P.binding(row))
    end

    #=
    The gate. Every page of the corpus is measured, and each count is held to its row. A count may
    fall and may not rise. A page whose every counted rule is at zero carries no row, so the
    baseline empties as the rewrites land, and a row that names no page is a rename left behind.
    =#
    @testset "every page is at or below its row" begin
        recorded = P.read_baseline(; root = root)
        measured = P.measure(; root = root)

        @test !isempty(measured)

        risen = String[]
        for page in sort!(collect(keys(measured)))
            row = measured[page]
            ceiling = get(recorded, page, Dict{String, Int}())
            rs = P.rises(row, ceiling)
            isempty(rs) && continue
            push!(risen, page)
            for (column, (old, now)) in rs
                println("  $page: $column rose to $now, over a ceiling of $old.")
            end
            println("  Paste this row into code_health/", P.NAME, ":")
            println("    ", P.row_text(page, row))
        end
        if !isempty(risen)
            println("A counted rule of .github/instructions/julia-prose.instructions.md ",
                    "rose on $(length(risen)) page(s). A count may fall and may not rise. Rewrite ",
                    "the prose, or paste the printed row when a count fell elsewhere on the page.")
        end
        @test risen == String[]

        dead = sort!([p for p in keys(recorded) if !haskey(measured, p)])
        if !isempty(dead)
            println("Rows in code_health/", P.NAME, " that name no page. Delete them:")
            for p in dead
                println("    ", p)
            end
        end
        @test dead == String[]

        spent = sort!([p
                       for p in keys(recorded)
                       if haskey(measured, p) && isempty(P.binding(measured[p]))])
        if !isempty(spent)
            println("Pages whose every counted rule is now zero. Delete their rows from ",
                    "code_health/", P.NAME,
                    ", so the baseline empties as the rewrites land:")
            for p in spent
                println("    ", p)
            end
        end
        @test spent == String[]
    end
end
