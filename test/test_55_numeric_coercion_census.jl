@testset "Numeric coercion census: a numeric type is derived, never coerced" begin
    using Test

    #=
    Issue #876. A numeric type must be read off the arguments with `eltype`, `typeof`,
    `real` and `promote_type`, and the arithmetic must widen it. The Authority is
    `.github/instructions/julia-source-code.instructions.md` § *Numeric types come from
    the data*.

    Before the commit that added this file, the library wrapped a derived type in `float`
    at sites spread over more than thirty files. The wrapper reads as a harmless
    defensive measure and is not one: `float(Int)` is `Float64`, so ONE integer argument
    inside a `promote_type` widened the whole computation to `Float64`, and a `Float32`
    caller lost its precision with no error and no warning. The same wrapper decided for
    every number type the library has never seen.

    Nothing enforced the rule, and the idiom spread by copy: a new file took the line from
    its neighbour. This census stops that. It reads the source as text rather than through
    dispatch, because the breach is a spelling and a spelling is what a reader copies.

    `nextfloat` and `prevfloat` step a value and coerce nothing, so the pattern below
    refuses a match that carries an identifier character before it.

    Issue #994. `float(` is not the only spelling of the breach: naming a concrete float
    type directly decides the same thing, and for the same reasons. `NAMED_FLOAT` reads
    that spelling. Two readings of a concrete type name are legitimate and it must let
    both past. A **type parameter or annotation** converts nothing, so the lookbehind
    refuses a `{` and the missing `(` refuses `::Float64` and `Float64[]`. A **literal**
    of that type states a constant rather than converting a derived one, so the pattern
    requires an identifier character after the parenthesis.

    What the census does not read is what carries no `(`: `zeros(Float64, n)` and its
    siblings name a concrete type for an allocation, and `convert(Float64, x)` names one
    for a conversion. Neither spelling stands in `src/` or `ext/` today. Add the pattern
    with the site, not before it, so that the census is never a rule with no reader.
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

    # A coercion of a type or a value into a float.
    COERCE = r"(?<![\w.])float\("
    # An index rounded in one type and converted in another, which `ceil(Int, x)` and its
    # siblings do in one step and in one type.
    ROUND_THEN_CONVERT = r"\b(?:Int|Int8|Int16|Int32|Int64|Int128|UInt|UInt8|UInt16|UInt32|UInt64|UInt128)\(\s*(?:ceil|floor|round|trunc)\("
    # A concrete float type used as a converter, which decides the width the arguments
    # were going to derive. The lookbehind lets a type parameter past, and the trailing
    # identifier character lets a literal past.
    NAMED_FLOAT = r"(?<![\w.{])(?:Float16|Float32|Float64|BigFloat|ComplexF16|ComplexF32|ComplexF64)\(\s*[A-Za-z_]"

    #=
    A site the rule owns and another ticket ships. Each entry maps a file to the fragment
    of the one line the census forgives there, and the census asserts that the fragment is
    still found, so the entry reds the build when its ticket lands and it is not removed.
    The dictionary is empty: the last entry, `unify_gaps` naming `Float64` as the
    unification target of a price panel, was shipped by issue #1002 under ADR 0135, which
    derives the target from the series.
    =#
    EXEMPT = Dict{String, String}()

    coerced = String[]
    rounded = String[]
    named = String[]
    forgiven = String[]
    for f in files
        rel = relpath(f, ROOT)
        exempt = get(EXEMPT, replace(rel, '\\' => '/'), nothing)
        for (i, line) in enumerate(eachline(f))
            if occursin(COERCE, line)
                push!(coerced, "$rel:$i: $(strip(line))")
            end
            if occursin(ROUND_THEN_CONVERT, line)
                push!(rounded, "$rel:$i: $(strip(line))")
            end
            if occursin(NAMED_FLOAT, line)
                if !isnothing(exempt) && occursin(exempt, line)
                    push!(forgiven, "$rel:$i: $(strip(line))")
                else
                    push!(named, "$rel:$i: $(strip(line))")
                end
            end
        end
    end

    if !isempty(coerced)
        @info "Sites that coerce a numeric type with `float`:\n" * join(coerced, "\n")
    end
    @test isempty(coerced)

    if !isempty(rounded)
        @info "Sites that round an index and then convert it:\n" * join(rounded, "\n")
    end
    @test isempty(rounded)

    if !isempty(named)
        @info "Sites that name a concrete float type as a converter:\n" * join(named, "\n")
    end
    @test isempty(named)

    # Every exemption is still a site. One that is not has been shipped, and the entry that
    # forgives it must go with it.
    @test length(forgiven) == length(EXEMPT)
end
