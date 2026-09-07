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

    coerced = String[]
    rounded = String[]
    for f in files
        rel = relpath(f, ROOT)
        for (i, line) in enumerate(eachline(f))
            if occursin(COERCE, line)
                push!(coerced, "$rel:$i: $(strip(line))")
            end
            if occursin(ROUND_THEN_CONVERT, line)
                push!(rounded, "$rel:$i: $(strip(line))")
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
end
