#=
`code_health/CodeHealth.jl` holds the one parser every census reads the source text with. The load
sits OUTSIDE the `@testset` on purpose: `include` defines methods, and a method defined inside one
top-level statement is not visible to a call in that same statement. The module wrapper keeps that
module's own names out of the worker module. `test_70_trivial_unit_census.jl` loads it the same way.
=#
module ConcreteStructCensusHealth
include(joinpath(@__DIR__, "..", "code_health", "CodeHealth.jl"))
end

@testset "Concrete struct census: a struct takes its type parameters from @concrete and its bounds from its constructors (#1359)" begin
    using PortfolioOptimisers, Test

    #=
    Issue #1359. `.github/instructions/julia-source-code.instructions.md` § *Type Definitions*
    states that a struct uses `@concrete`, and § *Constructor Pattern* states that the inner
    constructor holds the type bounds. Eighty structs did not follow the rule. Each declared its
    type parameters by hand, `struct Foo{T1 <: Real, T2 <: EntropicProjection}`, and most of
    them bound each parameter in the header. A bound in the header is a bound that every
    signature which names the type must satisfy, and inference can fail or take very long on
    it. The online portfolio selection family carried most of them.

    The census reads two shapes from the parsed source:

    1. a struct outside `@concrete` whose header declares type parameters, and
    2. a field line of a `@concrete` struct that uses the bounded form `field <: T`, which
       gives the generated type parameter a bound in the header.

    A flagged name is kept only by an entry below, with its reason. An entry whose name is no
    longer flagged is stale and fails, so each list can only shrink.
    =#
    CH = ConcreteStructCensusHealth.CodeHealth
    PO = PortfolioOptimisers
    root = normpath(joinpath(@__DIR__, ".."))

    #=
    A struct that cannot be written with `@concrete`. The macro gives each untyped field its
    own parameter, so it cannot pass a parameter to a parametric supertype, and it cannot let
    two fields share one parameter. None of these parameters carries a bound, and the check
    below the census holds that.
    =#
    hand_allowed = Dict(:ScopedConfig => "a `mutable` holder whose `@atomic` default and whose `ScopedValue{Union{Nothing, T}}` share one parameter",
                        :RepeatedLeading => "an `AbstractArray{T, N}`, whose supertype reads the element type and the rank",
                        :MinimumSquaredDistance => "the parameter is the one of its supertype `SquaredOrderedWeightsArrayAlgorithm{T}`",
                        :MinimumSumSquares => "the parameter is the one of its supertype `SquaredOrderedWeightsArrayAlgorithm{T}`")
    #=
    ADR 0049 bounds the similarity of these three fields in the header on purpose. `@concrete`
    also writes a generic positional constructor, and a bound in the header is what makes that
    constructor refuse a similarity that can go negative, so every construction route refuses
    one.
    =#
    bounded_allowed = Dict(:NetworkEstimator => "ADR 0049: `alg <: Tree_SimMat` refuses a similarity that can go negative on every construction route",
                           :LoGo => "ADR 0049: `sim <: AbstractNonNegativeSimilarityMatrixAlgorithm` refuses a similarity that can go negative on every construction route",
                           :DBHT => "ADR 0049: `sim <: AbstractNonNegativeSimilarityMatrixAlgorithm` refuses a similarity that can go negative on every construction route")

    files = String[]
    for d in ("src", "ext"), (r, _, fs) in walkdir(joinpath(root, d)), f in fs
        endswith(f, ".jl") && push!(files, joinpath(r, f))
    end
    sort!(files)

    function struct_head(s)
        h = s.args[2]
        if h isa Expr && h.head === :(<:)
            h = h.args[1]
        end
        return h isa Expr && h.head === :curly ? (h.args[1], true) : (h, false)
    end

    hand = Dict{Symbol, String}()
    bounded = Dict{Symbol, String}()
    n_concrete = 0
    for f in files
        rel = relpath(f, root)
        CH.walk_ast(CH.parse_file(f; root)) do ex
            if ex.head === :macrocall && ex.args[1] === Symbol("@concrete")
                s = ex.args[end]
                if s isa Expr && s.head === :struct
                    n_concrete += 1
                    name, _ = struct_head(s)
                    for line in s.args[3].args
                        if line isa Expr && line.head === :(<:)
                            bounded[name] = "$rel: $(line)"
                        end
                    end
                end
                return CH.PRUNE
            elseif ex.head === :struct
                name, has_params = struct_head(ex)
                if has_params
                    hand[name] = rel
                end
            end
        end
    end

    # A census that parses nothing passes on an empty set. 582 structs used `@concrete` when
    # the census was written.
    @test n_concrete >= 550

    # A new entry here is a struct that declares its own parameters. Write it with `@concrete`,
    # and bind its arguments in its inner constructor.
    @test sort!([string(n, " (", hand[n], ")")
                 for n in setdiff(keys(hand), keys(hand_allowed))]) == String[]
    @test sort!(collect(setdiff(keys(hand_allowed), keys(hand)))) == Symbol[]

    # A new entry here is a `@concrete` field bounded in the header. Drop the bound, and bind the
    # argument in the inner constructor.
    @test sort!([bounded[n] for n in setdiff(keys(bounded), keys(bounded_allowed))]) ==
          String[]
    @test sort!(collect(setdiff(keys(bounded_allowed), keys(bounded)))) == Symbol[]

    # A struct kept on the hand-written list takes its bounds from its constructors all the same.
    for n in sort!(collect(keys(hand_allowed)))
        U = Base.unwrap_unionall(getfield(PO, n))
        @test all(p -> !(p isa TypeVar) || p.ub === Any, U.parameters)
    end
end
