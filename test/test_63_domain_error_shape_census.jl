@testset "DomainError shape census: a raise carries the value and a message" begin
    using Test

    #=
    Issue #1039. A `DomainError` is raised in one shape, `DomainError(value, message)`, so
    that `.val` is the quantity out of range and `.msg` says what range it left. The
    Authority is `.github/instructions/julia-source-code.instructions.md` § *A
    `DomainError` carries the value and a message*.

    Before the commit that added this file, the raise was written in three shapes across
    the library. `DomainError(value, message)` stood at most sites. `DomainError("sentence")`
    stood at sixty-two: the one-argument constructor takes a VALUE, so the sentence landed
    in `.val`, `.msg` was left undefined, and `showerror` printed `DomainError with
    <sentence>:` and then nothing. A bare `DomainError` type stood at twenty-eight: `ArgCheck`
    then writes the message from the source expression, so the caller read `fraction <=
    one(fraction) must hold` where a field name and a range were owed. A maintainer writing
    the next guard had three precedents and no rule, and the idiom spread by copy.

    This census reads the source as text rather than through dispatch, because the breach is
    a spelling and a spelling is what a reader copies. It reads two spellings:

      - `DomainError(` followed by a string literal, on the same line or on the next. A
        two-argument raise opens with the value, which is never a literal string, so a
        string in the first slot is the one-argument form.
      - `DomainError` followed by a closing parenthesis, which is the bare type passed as
        the second argument of `@argcheck`.

    What it does not read is `DomainError(x)` with an identifier in the slot, because that
    spelling stands nowhere in `src/` or `ext/` today. Add the pattern with the site, not
    before it, so that the census is never a rule with no reader.
    =#

    ROOT = normpath(joinpath(@__DIR__, ".."))
    SCOPE = [joinpath(ROOT, "src"), joinpath(ROOT, "ext")]

    files = String[]
    for dir in SCOPE
        for (dp, _, fns) in walkdir(dir)
            for fn in fns
                if endswith(fn, ".jl")
                    push!(files, joinpath(dp, fn))
                end
            end
        end
    end
    @test !isempty(files)

    # A string literal in the value slot, on the same line as the constructor.
    SENTENCE = r"\bDomainError\(\s*\""
    # The constructor with its argument list continued on the next line, which the
    # formatter produces when the argument does not fit the margin.
    SENTENCE_OPEN = r"\bDomainError\($"
    SENTENCE_NEXT = r"^\s*\""
    # The bare type passed where an exception is owed.
    BARE = r"\bDomainError\s*\)"

    sentences = String[]
    bares = String[]
    for f in files
        rel = relpath(f, ROOT)
        lines = readlines(f)
        for (i, line) in enumerate(lines)
            if occursin(SENTENCE, line)
                push!(sentences, "$rel:$i: $(strip(line))")
            elseif occursin(SENTENCE_OPEN, line) &&
                   i < length(lines) &&
                   occursin(SENTENCE_NEXT, lines[i + 1])
                push!(sentences, "$rel:$i: $(strip(line)) $(strip(lines[i + 1]))")
            end
            if occursin(BARE, line)
                push!(bares, "$rel:$i: $(strip(line))")
            end
        end
    end

    if !isempty(sentences)
        @info "Sites that raise a `DomainError` from a sentence alone, so the sentence is `.val` and `.msg` is empty:\n" *
              join(sentences, "\n")
    end
    @test isempty(sentences)

    if !isempty(bares)
        @info "Sites that pass the bare `DomainError` type, so `ArgCheck` writes the message from the source expression:\n" *
              join(bares, "\n")
    end
    @test isempty(bares)

    # The two patterns read the spellings they are named for. A mutation of each is fed to
    # them here, so the census never passes because a pattern rotted rather than because the
    # tree is clean.
    @test occursin(SENTENCE, "    DomainError(\"0 < n must hold\"))")
    @test occursin(SENTENCE_OPEN, "              DomainError(")
    @test occursin(SENTENCE_NEXT, "                          \"0 < n must hold\"))")
    @test occursin(BARE, "    @argcheck(zero(n) < n, DomainError)")
    @test occursin(BARE, "                      DomainError)")
    @test !occursin(SENTENCE, "    DomainError(n, \"0 < n must hold\"))")
    @test !occursin(BARE, "    DomainError(n, \"0 < n must hold\"))")
    @test !occursin(BARE, "  - `DomainError`: if `b <= s`.")
end
