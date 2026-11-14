@testset "Struct layout census: the common instance of a struct wastes no padding (#1754)" begin
    using PortfolioOptimisers, Test

    #=
    Issue #1754. Julia lays out the fields of a struct in the order of their declaration, and
    it aligns each field to the alignment of its type. A narrow field before a wider one, a
    `Bool` before an `Int` or a pointer, gets padding up to the alignment of the wider field.
    `.github/instructions/julia-source-code.instructions.md` § *Type Definitions* states the
    rule: the common instance of a struct wastes no byte that another order of its fields saves.

    The common instance is the one that the keyword constructor builds from its defaults. The
    census builds it for every concrete struct of the library that has a constructor with no
    arguments, and it audits every value of the library nested in that instance. A struct
    passes when its `sizeof` equals the size of its fields sorted by alignment in descending
    order, which is the smallest size that an order of its fields can reach. A field that holds
    a singleton, such as `nothing`, takes no bytes, so it never breaks the alignment.

    Four structs failed, each by eight bytes. `HindsightSplit` put `prefix::Bool` before
    `start::Int`. `JuMPOptimiser` put `xbgt` among the budgets and `strict` at the end.
    `FactorBlackLittermanPrior` put `rsd` far from `ve`, whose common instance is one byte wide.
    `TargetReturnForecast` spread its three `Bool` flags over the struct.
    =#
    PO = PortfolioOptimisers

    #=
    The size and the alignment of one field slot of a concrete struct. Julia stores an isbits
    union, such as the `Union{Nothing, Int}` fields of `LogLimits`, inline: the widest member,
    then one selector byte. Each such field of `LogLimits` takes 16 bytes, and no order saves one.
    =#
    function slot(ft)
        if isconcretetype(ft) && Base.allocatedinline(ft)
            return (sizeof(ft), Base.datatype_alignment(ft))
        elseif Base.isbitsunion(ft)
            us = Base.uniontypes(ft)
            return (maximum(sizeof, us) + 1, maximum(Base.datatype_alignment, us))
        else
            return (sizeof(Ptr{Cvoid}), Base.datatype_alignment(Ptr{Cvoid}))
        end
    end
    function packed_size(slots)
        off = 0
        amax = 1
        for (s, a) in slots
            s == 0 && continue
            off = cld(off, a) * a + s
            amax = max(amax, a)
        end
        return cld(off, amax) * amax
    end
    function walk!(seen, x, depth = 0)
        depth > 8 && return nothing
        if x isa Array && !isbitstype(eltype(x))
            foreach(v -> walk!(seen, v, depth + 1), Iterators.take(x, 4))
            return nothing
        elseif x isa AbstractDict
            foreach(v -> walk!(seen, v, depth + 1), Iterators.take(values(x), 4))
            return nothing
        end
        S = typeof(x)
        (S in keys(seen) || !isstructtype(S)) && return nothing
        (parentmodule(S) === PO || S <: Tuple || S <: NamedTuple) || return nothing
        seen[S] = x
        for i in 1:fieldcount(S)
            isdefined(x, i) && walk!(seen, getfield(x, i), depth + 1)
        end
        return nothing
    end

    seen = Dict{DataType, Any}()
    for nm in names(PO; all = true)
        isdefined(PO, nm) || continue
        T = getfield(PO, nm)
        T isa Type || continue
        U = Base.unwrap_unionall(T)
        U isa DataType || continue
        (isabstracttype(U) || !isstructtype(U) || parentmodule(U) !== PO || U <: Function) &&
            continue
        x = try
            T()
        catch
            continue
        end
        walk!(seen, x)
    end
    #=
    A struct that needs an argument has no instance with no arguments, so the loop above
    cannot build it. Each of these is the common instance of a struct that #1754 fixed, as its
    docstring example builds it. The audit of #1754 replayed every `jldoctest` example of the
    library and found no other struct that wastes bytes.
    =#
    for x in (MeanRisk(; opt = JuMPOptimiser(; slv = Solver(; solver = nothing))),
              FactorBlackLittermanPrior(;
                                        sets = UniverseSets(;
                                                            dict = Dict("nx" => ["A", "B", "C"],
                                                                        "nf" => ["F1", "F2"])),
                                        views = LinearConstraintEstimator(;
                                                                          val = ["F1 == 0.03",
                                                                                 "F2 == 0.04"])),
              TargetReturnForecast(;
                                   scores = DescriptorScores(;
                                                             descriptors = [Passthrough(;
                                                                                        field = "a")]),
                                   target_outlier = nothing))
        walk!(seen, x)
    end
    audited = [S
               for S in keys(seen)
               if parentmodule(S) === PO && isconcretetype(S) && fieldcount(S) >= 2]
    # The census reaches the library: hundreds of structs, the fixed ones among them.
    @test length(audited) > 200
    @test typeof(HindsightSplit()) in audited
    @test sizeof(HindsightSplit()) == 16
    @test slot(Union{Nothing, Int}) == (9, 8)
    @test PO.LogLimits in audited
    for T in (JuMPOptimiser, FactorBlackLittermanPrior, TargetReturnForecast)
        @test any(S -> S <: T, audited)
    end

    slots(S) = [slot(fieldtype(S, i)) for i in 1:fieldcount(S)]
    # The model of the layout agrees with Julia's, so a pass below is a pass of the real layout.
    @test isempty([S for S in audited if packed_size(slots(S)) != sizeof(S)])
    wasteful = [(S.name.name, sizeof(S), packed_size(sort(slots(S); by = x -> -x[2])))
                for S in audited
                if sizeof(S) != packed_size(sort(slots(S); by = x -> -x[2]))]
    @test isempty(wasteful)
end
