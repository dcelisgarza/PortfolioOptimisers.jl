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

    Issue #1061. Two more spellings of the same breach passed the two patterns above.
    `NAMED_FLOAT_BROADCAST` reads the broadcast form of a named type, `Float64.(x)`,
    which `NAMED_FLOAT` missed because the dot sits between the name and the
    parenthesis. `FILL_LITERAL` reads a `NaN` or `Inf` literal filled into an array,
    `fill(NaN, n)`: the literal is `Float64`, so the array is `Float64` whatever the data,
    and the eight sites that stood in `src/` were each a breach with the data in hand — a
    Welford state has `X`, a graph traversal has the connection matrix, a JuMP model has
    its value type, and the free weight bounds take the `datatype` every caller already
    passed. A constant filled in a derived type is spelled `fill(convert(T, NaN), n)`.

    2026-09-26. Sessions repaired an integer input over and over, first with `float(` and
    then, once this census refused that spelling, with a type derived from a division:
    `typeof(one(T) / one(T))`, `typeof(zero(T) / one(Int))`. Both decide for a type that
    needed no repair. `float` turns a `Rational` into a `Float64`, and a division lands in
    whatever type a number type defines for it. The maintainer ruled that a site takes the
    type of its data and repairs an integer only, through ONE guard:
    `float_if_integer` in `src/02_Tools/03_TypeUtilities.jl`. `GUARD` is the one line
    that may call `float`, and `DIVISION_TYPE` reads the division spelling. A type derived
    from an operation that leaves every type, such as `typeof(sqrt(one(Tf)))` or a division
    by a square root, is not a repair, so the pattern reads a division of a bare `one`,
    `zero` or `oneunit` by another of them or by a literal only.

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

    # A coercion of a type or a value into a float, spelled as a call or as a broadcast.
    COERCE = r"(?<![\w.])float\.?\("
    # An index rounded in one type and converted in another, which `ceil(Int, x)` and its
    # siblings do in one step and in one type.
    ROUND_THEN_CONVERT = r"\b(?:Int|Int8|Int16|Int32|Int64|Int128|UInt|UInt8|UInt16|UInt32|UInt64|UInt128)\(\s*(?:ceil|floor|round|trunc)\("
    # A concrete float type used as a converter, which decides the width the arguments
    # were going to derive. The lookbehind lets a type parameter past, and the trailing
    # identifier character lets a literal past.
    NAMED_FLOAT = r"(?<![\w.{])(?:Float16|Float32|Float64|BigFloat|ComplexF16|ComplexF32|ComplexF64)\(\s*[A-Za-z_]"
    # A concrete float type used as a broadcast converter, which decides the same thing
    # for every element.
    NAMED_FLOAT_BROADCAST = r"(?<![\w.{])(?:Float16|Float32|Float64|BigFloat|ComplexF16|ComplexF32|ComplexF64)\.\(\s*[A-Za-z_]"
    # A float literal filled into an array, whose element type is then the literal's.
    FILL_LITERAL = r"\bfill\(\s*-?(?:NaN|Inf)\b"
    # A working type read off a division of constants, which repairs an integer input by
    # deciding the type of every input.
    DIVISION_TYPE = r"\btypeof\(\s*(?:one|zero|oneunit)\(.*?\)\s*/\s*(?:(?:one|zero|oneunit)\(|\d)"
    # The one line that may call `float`: the guard that repairs an integer type and keeps
    # every other type.
    GUARD = ("src/02_Tools/03_TypeUtilities.jl",
             "float_if_integer(::Type{T}) where {T <: Integer} = float(T)")

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
    broadcast = String[]
    filled = String[]
    divided = String[]
    forgiven = String[]
    guarded = String[]
    for f in files
        rel = relpath(f, ROOT)
        exempt = get(EXEMPT, replace(rel, '\\' => '/'), nothing)
        for (i, line) in enumerate(eachline(f))
            if occursin(COERCE, line)
                if replace(rel, '\\' => '/') == GUARD[1] && strip(line) == GUARD[2]
                    push!(guarded, "$rel:$i: $(strip(line))")
                else
                    push!(coerced, "$rel:$i: $(strip(line))")
                end
            end
            if occursin(DIVISION_TYPE, line)
                push!(divided, "$rel:$i: $(strip(line))")
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
            if occursin(NAMED_FLOAT_BROADCAST, line)
                push!(broadcast, "$rel:$i: $(strip(line))")
            end
            if occursin(FILL_LITERAL, line)
                push!(filled, "$rel:$i: $(strip(line))")
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

    if !isempty(broadcast)
        @info "Sites that name a concrete float type as a broadcast converter:\n" *
              join(broadcast, "\n")
    end
    @test isempty(broadcast)

    if !isempty(filled)
        @info "Sites that fill an array with a float literal:\n" * join(filled, "\n")
    end
    @test isempty(filled)

    if !isempty(divided)
        @info "Sites that read a working type off a division; take the type of the data and pass it to `float_if_integer`:\n" *
              join(divided, "\n")
    end
    @test isempty(divided)

    # The guard is still where the census looks for it, so its one allowance is not stale.
    @test length(guarded) == 1

    # Every exemption is still a site. One that is not has been shipped, and the entry that
    # forgives it must go with it.
    @test length(forgiven) == length(EXEMPT)
end
