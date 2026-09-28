#=
A reader who meets a source in the documentation must be able to open it. The rule in
`.github/instructions/julia-docstrings.instructions.md` § *A docstring names a source by its
citation* and its twin in `.github/instructions/julia-prose.instructions.md` ban the words that
name a work and link to nothing: "the paper", "the authors", and an author list with a year in
place of a `[key](@cite)` or a `[key](@citet)`.

Issue #1360 wrote the rule. On the day it was written, this census named 158 texts in 34 files:
docstrings that said "the paper's parameters" to an implementer, "Györfi and Schäfer (2003)" in
place of a link, and "the paper" in the field text of a prior. Most of them sat in the online
portfolio selection family, whose docstrings a sweep had checked against the papers one by one. A
review reads a claim, and it does not see the words that name its source. So the rule needs a
parser, and the census costs a few seconds.

------------------------------------------------------------------ what is scanned

Three corpora, read without the package:

  - The docstrings of `src/**/*.jl` and `ext/**/*.jl`, by `Meta.parseall`. A docstring is the
    literal text of a `@doc` call, which `CodeHealth.docstring_text` reads, and the literal text
    of a string that stands alone in the body of a `struct`, which is the docstring of the field
    below it. A `#` comment never renders, so it is out of scope.
  - The values of the docstring dictionaries under `src/01_Base/01_DocstringDictionaries/`: each
    string literal of those files, except `13_References.jl`. `ref_dict` holds the bibliography
    text itself, and an author list with a year is what a bibliography entry is.
  - The pages that `code_health/prose.jl` reads: the Literate sources, the hand-written Markdown
    pages, `README.md` and the Capability Catalogue, through its one reader, `prose`.

Before the match, a docstring loses its code: a fenced block, a `jldoctest` block, an inline code
span and an inline LaTeX expression. A citation `[key](@cite)` and a link lose their target.

------------------------------------------------------------------ what is a match

  - WORDS: "paper", "papers", "article", "articles", "co-author", "co-authors" and "the authors",
    in any case. "author" alone is not a match: "an extension author" is the reader who subtypes
    an abstract type, and the library uses the word in that sense.
  - AUTHOR_YEAR: a word with a capital letter, or "et al.", and then a year in parentheses, such
    as "Singer (1997)" and "Zinkevich's (2003)", or "et al." and then a year, such as
    "Damian et al. 2023". A bare "In 2008" is not a match, because a date is not a source.

An author list with no year, and a private name for a value, hold by review. A match is a
defect, and the fix is the rule's own: cite the work, and repeat the citation where the text
said "the paper". There is no allow-list.
=#
module SourceCitationCensus
module CH
    include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end
module Prose
    include(joinpath(@__DIR__, "..", "code_health", "prose.jl"))
end
end

@testset "Source citation census: the documentation names a source by its citation" begin
    using Test

    CH = SourceCitationCensus.CH.CodeHealth
    P = SourceCitationCensus.Prose
    ROOT = normpath(joinpath(@__DIR__, ".."))
    DICTS = joinpath(ROOT, "src", "01_Base", "01_DocstringDictionaries")

    WORDS = r"\b(?:papers?|articles?|co-authors?)\b|\bthe authors\b"i
    AUTHOR_YEAR = r"(?:\b\p{Lu}[\p{L}'’-]+|et al\.?),? \((?:19|20)\d{2}\b|\bet al\.?,? (?:19|20)\d{2}\b"

    function files_under(dir)
        acc = String[]
        isdir(dir) || return acc
        for (root, _, files) in walkdir(dir), f in files
            endswith(f, ".jl") && push!(acc, joinpath(root, f))
        end
        return sort!(acc)
    end

    # The text of a docstring that the rule governs: no code, no LaTeX, no link target.
    function readable(text)
        s = replace(text, r"```.*?```"s => " ")
        s = replace(s, r"``.*?``"s => " ")
        s = replace(s, r"`[^`\n]*`" => " ")
        return replace(s, r"\]\([^)\n]*\)" => "]")
    end

    function matches(text)
        s = readable(text)
        return vcat([m.match for m in eachmatch(WORDS, s)],
                    [m.match for m in eachmatch(AUTHOR_YEAR, s)])
    end

    # The literal text of a string, with each interpolation read as a space, or `nothing`.
    function literal(x)
        x isa AbstractString && return String(x)
        Meta.isexpr(x, :string) || return nothing
        return join(p isa AbstractString ? p : " " for p in x.args)
    end

    # Every docstring of one source file, as `(line, text)`.
    function docstrings(file)
        acc, line = Tuple{Int, String}[], Ref(0)
        function walk(ex)
            ex isa LineNumberNode && (line[] = ex.line; return nothing)
            ex isa Expr || return nothing
            if CH.isdocstring(ex)
                push!(acc, (line[], CH.docstring_text(ex)))
            elseif Meta.isexpr(ex, :struct)
                for a in ex.args[3].args
                    a isa LineNumberNode && (line[] = a.line)
                    t = literal(a)
                    isnothing(t) || push!(acc, (line[], t))
                end
            end
            foreach(walk, ex.args)
            return nothing
        end
        walk(Meta.parseall(read(file, String); filename = file))
        return acc
    end

    # Every string literal of one dictionary file, as `(line, text)`.
    function dictionary_values(file)
        acc, line = Tuple{Int, String}[], Ref(0)
        function walk(ex)
            ex isa LineNumberNode && (line[] = ex.line; return nothing)
            t = literal(ex)
            isnothing(t) || (push!(acc, (line[], t)); return nothing)
            ex isa Expr && foreach(walk, ex.args)
            return nothing
        end
        walk(Meta.parseall(read(file, String); filename = file))
        return acc
    end

    # Each offender on a line of its own: `Test` cuts a long array short when it prints it.
    function report(offenders)
        isempty(offenders) ||
            @warn """$(length(offenders)) text(s) name a source by words that link to nothing.
                     Cite the work with `[key](@cite)` or `[key](@citet)`, and repeat the
                     citation where the text says "the paper":\n  $(join(offenders, "\n  "))"""
        return offenders
    end

    hits(file, entries) = ["$(relpath(file, ROOT)):$(l): $(join(matches(t), " | "))"
                           for (l, t) in entries if !isempty(matches(t))]

    @testset "the patterns read what they claim" begin
        @test matches("This is the rule of Györfi and Schäfer (2003).") == ["Schäfer (2003"]
        @test matches("Relativistic Value-at-Risk (Damian et al. 2023):") == ["et al. 2023"]
        @test matches("the comparator of Zinkevich's (2003) Definition 7") ==
              ["Zinkevich's (2003"]
        @test length(matches("The paper's defaults. The authors publish code.")) == 2
        @test isempty(matches("This is the rule of [gyorfischafer2003](@citet)."))
        @test isempty(matches("In 2008 the methods an extension author must define."))
        @test isempty(matches("See ``\\text{paper}`` and `paper_key` in code."))
        mktempdir() do dir
            probe = joinpath(dir, "probe.jl")
            write(probe,
                  "\"\"\"\nThe paper.\n\"\"\"\nstruct A\n    \"\"\"\n    " *
                  "The paper's L.\n    \"\"\"\n    L::Int\nend\n# the paper\n")
            @test length(docstrings(probe)) == 2
            @test length(hits(probe, docstrings(probe))) == 2
        end
    end

    @testset "src and ext docstrings" begin
        offenders = String[]
        for dir in ("src", "ext"), f in files_under(joinpath(ROOT, dir))
            startswith(f, DICTS) && continue
            append!(offenders, hits(f, docstrings(f)))
        end
        @test report(offenders) == String[]
    end

    @testset "docstring dictionary values" begin
        offenders = String[]
        for f in files_under(DICTS)
            basename(f) == "13_References.jl" && continue
            append!(offenders, hits(f, dictionary_values(f)))
        end
        @test report(offenders) == String[]
    end

    @testset "pages" begin
        offenders = String[]
        for p in P.pages(; root = ROOT)
            for (i, ln) in enumerate(P.prose(joinpath(ROOT, p)))
                m = matches(ln)
                isempty(m) || push!(offenders, "$(p) (prose line $(i)): $(join(m, " | "))")
            end
        end
        @test report(offenders) == String[]
    end
end
