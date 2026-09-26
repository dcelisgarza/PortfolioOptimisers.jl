@testset "Process citation census: user-facing prose names no issue, pull request or ADR" begin
    using Test

    #=
    A docstring documents the released unit, not the discussion that produced it. The rule
    in `.github/instructions/julia-docstrings.instructions.md` § *A docstring cites no
    process* bans a GitHub issue, a pull request, an ADR and an unpublished experiment from
    every page a library user reaches without opening `docs/adr/`: the docstrings of `src/`
    and `ext/`, the hand-written pages of `docs/src/`, the `Prose` text of the Capability
    Catalogue, and the rendered prose of the Literate sources that build the example and
    user-guide pages. A number that names a ticket means nothing to that reader, and sends
    them to a page that is not part of the package.

    `STANDARDS.md` carried the rule as *none -- unenforced* until this file. Four strips
    on one branch removed about thirty sites, and six survived them, because a strip is
    a grep run once and a rule with no gate holds by memory alone.

    ------------------------------------------------------------- what is scanned

    Four corpora are read as TEXT, so the census loads no package:

      - `src/**/*.jl` and `ext/**/*.jl`: every line inside a triple-quoted block. The
        toggle is the one `test_26_docs.jl` uses for section order, so a plain
        triple-quoted literal is scanned too and contributes nothing. A `#` comment is out
        of scope: it is written for a contributor and never renders.
      - `docs/src/**/*.md`, except `docs/src/contribute/`, which the rule exempts, and
        except the four generated paths, which the docs build writes from the Literate
        sources scanned below and which are not in the tree.
      - `docs/capability_catalogue.jl`: every line that is not a `#` comment. The file
        holds `Prose` text and `Cap` names and nothing else that renders, and a name is
        never a match.
      - `examples/**/*.jl` and `user_guide/*.jl`: the lines Literate renders as markdown,
        which are a line whose first non-blank character opens a `# ` comment at column
        zero and every line of a `#= ... =#` block, minus a line that carries the `#src`
        trailer, which Literate drops from every output.

    A fifth corpus is the error text of `src/` and `ext/`. An error message reaches the user
    through the REPL, so it is as user-facing as a page, and the rule names it in its scope.
    The docstring toggle cannot read it: an error literal sits indented inside its `throw`,
    and a column does not tell it from any other string. So this corpus is read by
    `Meta.parseall`, which loads no package either. It is every string literal, and every
    literal part of an interpolated string, inside one of:

      - a call to `throw`, `error` or `rethrow`, or to a name that ends in `Error` or
        `Exception`, such as `ArgumentError(…)` and `DomainError(…)`;
      - an `@argcheck` or an `@assert`, whose message is the error text;
      - the value of a `const` whose name ends in `_remedy` or `_message`, which is a
        remedy that more than one error interpolates.

    A message that is built in a local variable and thrown later is not read.

    ------------------------------------------------------------- what is a citation

    A citation is one of: `ADR` followed by four digits; a `#` followed by three or four
    digits at a word boundary, which is how this repository writes an issue or pull
    request number; the words `pull request`; `PR` followed by a number; or a GitHub
    `issues/` or `pull/` URL. Three digits is the floor because the tracker passed #100
    before the first docstring cited it, and a two-digit `#n` inside a docstring is a
    markdown heading level or a comment, never a ticket.

    A match is a defect, and the fix is the rule's own: state the fact, not its
    provenance. There is no allow-list, because the sites a strip left behind were exactly
    the ones a reader had judged harmless.
    =#

    ROOT = normpath(joinpath(@__DIR__, ".."))
    CITATION = r"\bADR\s?\d{4}\b|(?<![\w/])#\d{3,4}\b|pull request|\bPR\s?#?\d+\b|github\.com/[^\s)]*/(?:issues|pull)/\d+"

    function files_under(dir, ext; skip = String[])
        acc = String[]
        isdir(dir) || return acc
        for (root, dirs, files) in walkdir(dir)
            filter!(d -> !(joinpath(root, d) in skip), dirs)
            for f in files
                endswith(f, ext) && push!(acc, joinpath(root, f))
            end
        end
        return sort!(acc)
    end

    hit(file, i, ln) = "$(relpath(file, ROOT)):$(i): $(strip(ln))"

    # The docstrings of a source file: every line between a `"""` at column zero and the
    # next. A docstring opens and closes at column zero in this repository, and an error
    # literal written with triple quotes sits indented inside its `throw`, so the column is
    # what tells the two apart.
    function docstring_hits(file)
        acc, indoc = String[], false
        for (i, ln) in enumerate(readlines(file))
            opens = startswith(ln, "\"\"\"")
            if !indoc
                opens || continue
                if length(findall("\"\"\"", ln)) >= 2
                    # A one-line `"""text"""` opens and closes on the same line.
                    occursin(CITATION, ln) && push!(acc, hit(file, i, ln))
                else
                    indoc = true
                end
            else
                occursin(CITATION, ln) && push!(acc, hit(file, i, ln))
                opens && (indoc = false)
            end
        end
        return acc
    end

    # The error text of a source file: every string literal inside a call that raises, an
    # `@argcheck` or an `@assert`, or the value of a `const` named `*_remedy` or
    # `*_message`. A hit names the line of the statement that holds the literal.
    function callee(ex)
        f = ex.args[1]
        f isa Symbol && return f
        (Meta.isexpr(f, :.) && f.args[end] isa QuoteNode) && return f.args[end].value
        return nothing
    end
    function raises(ex)
        if Meta.isexpr(ex, :call)
            f = callee(ex)
            f isa Symbol || return false
            return f in (:throw, :error, :rethrow) ||
                   endswith(string(f), "Error") ||
                   endswith(string(f), "Exception")
        elseif Meta.isexpr(ex, :macrocall)
            return ex.args[1] in (Symbol("@argcheck"), Symbol("@assert"))
        elseif Meta.isexpr(ex, :const) && Meta.isexpr(ex.args[1], :(=))
            name = ex.args[1].args[1]
            return name isa Symbol && occursin(r"_(remedy|message)$", string(name))
        end
        return false
    end
    function error_text_hits(file)
        acc, line = String[], Ref(0)
        function walk(ex, inerr)
            if ex isa LineNumberNode
                line[] = ex.line
            elseif ex isa String
                m = match(CITATION, ex)
                (inerr && !isnothing(m)) &&
                    push!(acc, "$(relpath(file, ROOT)):$(line[]): $(m.match)")
            elseif ex isa Expr
                inerr = inerr || raises(ex)
                foreach(a -> walk(a, inerr), ex.args)
            end
            return nothing
        end
        walk(Meta.parseall(read(file, String); filename = file), false)
        return acc
    end

    # A whole file, line by line, minus a `#` comment line, which never renders.
    function page_hits(file; comments = true)
        acc = String[]
        for (i, ln) in enumerate(readlines(file))
            (!comments && startswith(lstrip(ln), "#")) && continue
            occursin(CITATION, ln) && push!(acc, hit(file, i, ln))
        end
        return acc
    end

    # The markdown lines of a Literate source: `# ` at column zero, and `#= ... =#` blocks,
    # minus a `#src` line.
    function literate_hits(file)
        acc, inblock = String[], false
        for (i, ln) in enumerate(readlines(file))
            endswith(rstrip(ln), "#src") && continue
            s = lstrip(ln)
            if inblock
                occursin(CITATION, ln) && push!(acc, hit(file, i, ln))
                startswith(s, "=#") && (inblock = false)
            elseif startswith(ln, "#=")
                inblock = true
                occursin(CITATION, ln) && push!(acc, hit(file, i, ln))
            elseif startswith(ln, "# ") || ln == "#"
                occursin(CITATION, ln) && push!(acc, hit(file, i, ln))
            end
        end
        return acc
    end

    @testset "src and ext docstrings" begin
        offenders = String[]
        for dir in ("src", "ext"), f in files_under(joinpath(ROOT, dir), ".jl")
            append!(offenders, docstring_hits(f))
        end
        @test offenders == String[]
    end

    @testset "src and ext error text" begin
        offenders = String[]
        for dir in ("src", "ext"), f in files_under(joinpath(ROOT, dir), ".jl")
            append!(offenders, error_text_hits(f))
        end
        @test offenders == String[]
        # The scan reads what it claims: a `throw`, an `@argcheck`, an interpolated literal
        # and a remedy `const` each yield a hit, and a string outside them yields none.
        probe = joinpath(mktempdir(), "probe.jl")
        write(probe, """
                     const x_remedy = "see ADR 0001"
                     f(a) = a > 0 || throw(ArgumentError("a \$(a) breaks ADR 0002"))
                     g(a) = @argcheck(a > 0, DomainError(a, "see #1234"))
                     h() = println("ADR 0003")
                     """)
        @test length(error_text_hits(probe)) == 3
    end

    @testset "hand-written docs pages" begin
        docs = joinpath(ROOT, "docs", "src")
        skip = [joinpath(docs, "contribute"), joinpath(docs, "examples"),
                joinpath(docs, "user_guide")]
        offenders = String[]
        for f in files_under(docs, ".md"; skip = skip)
            basename(f) in ("capability_catalogue.md", "TypeHierarchy.md") && continue
            append!(offenders, page_hits(f))
        end
        @test offenders == String[]
    end

    @testset "capability catalogue prose" begin
        catalogue = joinpath(ROOT, "docs", "capability_catalogue.jl")
        @test page_hits(catalogue; comments = false) == String[]
    end

    @testset "example and user-guide prose" begin
        offenders = String[]
        for dir in ("examples", "user_guide"), f in files_under(joinpath(ROOT, dir), ".jl")
            append!(offenders, literate_hits(f))
        end
        @test offenders == String[]
    end
end
