@testset "Tools" begin
    using PortfolioOptimisers, Statistics, StatsBase, Test, LinearAlgebra
    A = [0, 1, 1, 2, 3, 4, 5, 8]
    B = [2, 3, 3, 5, 8, 10, 15, 21]
    C = 13
    W = fweights([3, 5, 9, 8, 12, 3, 5, 9])
    @test PortfolioOptimisers.:⊙(C, A) == C * A
    @test PortfolioOptimisers.:⊘(A, C) == A / C
    @test PortfolioOptimisers.:⊘(C, C) == C / C
    @test PortfolioOptimisers.:⊕(A, B) == A + B
    @test PortfolioOptimisers.:⊕(C, B) == C .+ B
    @test PortfolioOptimisers.dot_scalar(C, A) == C * sum(A)
    @test PortfolioOptimisers.dot_scalar(A, C) == C * sum(A)
    @test PortfolioOptimisers.vec_to_real_measure(StdValue(), A) == std(A)
    @test PortfolioOptimisers.vec_to_real_measure(VarValue(), A) == var(A)
    @test PortfolioOptimisers.vec_to_real_measure(SumValue(), A) == sum(A)
    @test PortfolioOptimisers.vec_to_real_measure(ProdValue(), A) == prod(A)
    @test PortfolioOptimisers.vec_to_real_measure(ModeValue(), A) == mode(A)
    @test PortfolioOptimisers.vec_to_real_measure(MedianValue(), A) == median(A)
    @test PortfolioOptimisers.vec_to_real_measure(StandardisedValue(), A) ==
          mean(A) / std(A)
    msv = StandardisedValue()
    @test factory(msv) === msv
    msv = factory(StandardisedValue(), W)
    @test msv.mv.w === W
    @test msv.sv.w === W
    vv = factory(VarValue(), W)
    @test vv.w === W
    vv = factory(VarValue(), W)
    @test vv.w === W
    mdv = factory(MedianValue(), W)
    @test mdv.w === W
end
@testset "@propagatable contract" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    # The shipped guarantee: every `@propagatable` declaration satisfies the contract behind
    # the generated `factory`/`port_opt_view` methods. This is what precompilation asserts.
    @test !isempty(PO.PROPAGATABLE_CONTRACTS)
    @test isnothing(PO.check_propagatable_contracts())
    # A type declared outside this package must join the factory propagation chain (ADR 0002,
    # decision 4 and its 2026-08-17 amendment). Every name the expansion emits is escaped into
    # the caller, so a bare one resolves in the *declaring* module. This probe imports the
    # macro and nothing else, which is the least a caller can write, and it catches a bare name
    # of either kind: a private one throws an `UndefVarError`, and an exported one is worse --
    # `using` binds it implicitly, so the method definition declares a function of the caller's
    # own and the declaration then compiles and registers while carrying no method at all.
    n_registered = length(PO.PROPAGATABLE_CONTRACTS)
    foreign = Module(:PropagatableForeignProbe)
    # The import lands in its own `eval`: a block is macro-expanded as a unit, so a
    # `@propagatable` written beside its own `using` is expanded before that `using` runs.
    Base.eval(foreign, :(using PortfolioOptimisers))
    Base.eval(foreign, :(using PortfolioOptimisers: @propagatable))
    Base.eval(foreign, quote
                  @propagatable struct ForeignProbe{T1}
                      inner::T1
                  end
                  ForeignProbe(; inner = nothing) = ForeignProbe(inner)
              end)
    @test length(PO.PROPAGATABLE_CONTRACTS) == n_registered + 1
    @test PO.PROPAGATABLE_CONTRACTS[end] == (foreign.ForeignProbe, ())
    # `factory` in the declaring module is still this package's, not a new one of the caller's.
    # This is the assertion a bare exported name fails, and it fails silently: the two lines
    # above pass either way, because the declaration compiles and registers regardless.
    @test getglobal(foreign, :factory) === PO.factory
    @test parentmodule(getglobal(foreign, :factory)) === PO
    # The generated method is owned by the declaring module and lives on this package's
    # function, so the new type dispatches through the shipped chain.
    @test count(m -> m.module === foreign, methods(PO.factory)) == 1
    foreign_probe = foreign.ForeignProbe(; inner = 7)
    @test PO.factory(foreign_probe) === foreign_probe
    # The declaration was throwaway, so drop it again and leave the checked registry as found.
    pop!(PO.PROPAGATABLE_CONTRACTS)
    @test length(PO.PROPAGATABLE_CONTRACTS) == n_registered
    pool = PO.prior_result_property_pool()
    # `HighOrderPrior` forwards the whole of its `pr`, so the low-order names are in the
    # pool without being its fields.
    @test issubset((:mu, :sigma, :w, :kt, :sk), pool)
    @test issubset(fieldnames(PO.LowOrderPrior), pool)
    @test issubset(fieldnames(PO.HighOrderPrior), pool)
    # Declared in a throwaway module so the checked registry stays clean.
    m = Module(:PropagatableContractProbe)
    Base.eval(m, quote
                  struct KwMismatch{T1, T2}
                      inner::T1
                      typo::T2
                  end
                  KwMismatch(; inner = nothing, typoo = 1) = KwMismatch(inner, typoo)
                  struct Slurped{T1}
                      inner::T1
                  end
                  Slurped(; kwargs...) = Slurped(get(kwargs, :inner, nothing))
                  struct Named{T1}
                      inner::T1
                  end
                  Named(; inner = nothing) = Named(inner)
              end)
    # A field the outer keyword constructor does not name, with its near-miss suggestion.
    kwmsgs = PO.propagatable_contract_violations(m.KwMismatch, (), pool)
    @test length(kwmsgs) == 1
    @test occursin("field `typo` is not a keyword", kwmsgs[1])
    @test occursin("did you mean `typoo`?", kwmsgs[1])
    # A `kwargs...` slurp accepts the keyword and discards it, so it must not satisfy the
    # contract.
    @test PO.propagatable_keywords(m.Slurped) == Symbol[]
    @test length(PO.propagatable_contract_violations(m.Slurped, (), pool)) == 1
    # A named keyword does satisfy it.
    @test PO.propagatable_keywords(m.Named) == [:inner]
    @test isempty(PO.propagatable_contract_violations(m.Named, (), pool))
    # An `@pprop` field absent from every prior result carrier.
    prmsgs = PO.propagatable_contract_violations(m.Named, (:sgima,), pool)
    @test length(prmsgs) == 1
    @test occursin("`@pprop` field `sgima` is not a property of a prior result", prmsgs[1])
    @test occursin("did you mean `sigma`?", prmsgs[1])
    # Every violation is reported, not just the first.
    allmsgs = PO.propagatable_contract_violations(m.KwMismatch, (:sgima, :muu), pool)
    @test length(allmsgs) == 3
end
@testset "propagation tag table" begin
    using PortfolioOptimisers, Test
    PO = PortfolioOptimisers
    # The shipped guarantee: every row of the tag table has a stub macro, a field transform
    # and a channel, so a tag cannot parse and then never propagate (ADR 0061).
    @test isnothing(PO.check_prop_tag_macros())
    @test PO.PROP_TAG_MACRO_NAMES == map(tag -> Symbol("@", tag), PO.PROP_TAG_NAMES)
    # A tag is looked up in the table, in both spellings a `:macrocall` head takes: a bare
    # `Symbol` in a hand-written struct body, and a `GlobalRef` once another macro expanded
    # around it.
    for tag in PO.PROP_TAG_NAMES
        @test PO.prop_tag(Symbol("@", tag)) === tag
        @test PO.prop_tag(GlobalRef(PO, Symbol("@", tag))) === tag
        @test PO.is_prop_tag_call(Expr(:macrocall, Symbol("@", tag), nothing, :a))
    end
    # A macro outside the table is not a tag. It used to fall through to `@wprop`, which
    # substituted observation weights into the field with no diagnostic.
    @test isnothing(PO.prop_tag(Symbol("@sixth")))
    @test isnothing(PO.prop_tag(:a))
    @test isnothing(PO.prop_tag(1))
    @test !PO.is_prop_tag_call(Expr(:macrocall, Symbol("@sixth"), nothing, :a))
    sixth = Expr(:macrocall, Symbol("@sixth"), nothing, :a)
    @test PO.peel_prop_tags(sixth) == (Set{Symbol}(), sixth)
    # Stacked tags peel in either order, and the whole stack is recorded.
    stacked = Expr(:macrocall, Symbol("@pprop"), nothing,
                   Expr(:macrocall, Symbol("@wprop"), nothing, :(w::T1)))
    @test PO.peel_prop_tags(stacked) == (Set([:pprop, :wprop]), :(w::T1))
    # The parser answers one entry per table row, and every declared field.
    tagged, all_fields, new_body = PO.propagatable_parse_body(quote
                                                                  @fprop @vprop a
                                                                  @pprop @wprop w
                                                                  @cprop slv
                                                                  plain
                                                              end)
    @test issetequal(keys(tagged), PO.PROP_TAG_NAMES)
    @test tagged[:fprop] == [:a]
    @test tagged[:vprop] == [:a]
    @test tagged[:pprop] == [:w]
    @test tagged[:wprop] == [:w]
    @test tagged[:cprop] == [:slv]
    @test all_fields == [:a, :w, :slv, :plain]
    @test !any(PO.is_prop_tag_call, new_body.args)
    # The gate decides whether a channel emits a method at all.
    @test PO.prop_channel_active(:factory, tagged)
    @test PO.prop_channel_active(:view, tagged)
    @test PO.prop_channel_active(:prior, tagged)
    untagged = Dict{Symbol, Vector{Symbol}}(tag => Symbol[] for tag in PO.PROP_TAG_NAMES)
    @test !PO.prop_channel_active(:factory, untagged)
    @test !PO.prop_channel_active(:view, untagged)
    @test !PO.prop_channel_active(:prior, untagged)
    @test PO.prop_channel_active(:obs, tagged)
    @test !PO.prop_channel_active(:obs, untagged)
    # `@wprop` alone opens the factory and observation channels, and neither of the others.
    wonly = merge(untagged, Dict(:wprop => [:w]))
    @test PO.prop_channel_active(:factory, wonly)
    @test PO.prop_channel_active(:obs, wonly)
    @test !PO.prop_channel_active(:view, wonly)
    @test !PO.prop_channel_active(:prior, wonly)
    # `@fprop` is in the observation channel's precedence but not its gate, so a struct with
    # composed children and no weights of its own emits no `obs_weights_view` method and
    # falls through to the identity.
    fonly = merge(untagged, Dict(:fprop => [:a]))
    @test PO.prop_channel_active(:factory, fonly)
    @test !PO.prop_channel_active(:obs, fonly)
    # The precedence is data, and the prior channel puts `@pprop` above `@fprop` (ADR 0012).
    pairs = PO.prop_channel_pairs(:prior, tagged, all_fields, :xr, PO, (:pr,))
    @test [p.args[1] for p in pairs] == all_fields
    @test pairs[2].args[2] == :($(PO).sel(xr.w, getproperty(pr, :w)))
    @test pairs[3].args[2] == :($(PO).sel(xr.slv, $(PO)._ctx(args...)))
    @test pairs[4].args[2] == :(xr.plain)
    both = merge(untagged, Dict(:pprop => [:a], :fprop => [:a]))
    prior_pair = only(PO.prop_channel_pairs(:prior, both, [:a], :xr, PO, (:pr,)))
    @test prior_pair.args[2] == :($(PO).sel(xr.a, getproperty(pr, :a)))
    # The factory channel does not know `@pprop`, so the same field recurses there instead.
    factory_pair = only(PO.prop_channel_pairs(:factory, both, [:a], :x, PO, ()))
    @test factory_pair.args[2] == :($(PO).factory_child(x.a, args...; kwargs...))
    #=
    A tag means what its CHANNEL says it means. `@wprop` names the same field in the factory
    channel and in the observation channel, and the two transforms differ: `factory`
    REPLACES the field with an incoming `ObsWeights`, while `obs_weights_view` INDEXES the
    value already there. This is why `prop_tag_expr` takes the channel as well as the tag,
    and it is what lets a weights field join the observation-axis view with no second tag.
    =#
    obs_pairs = PO.prop_channel_pairs(:obs, tagged, all_fields, :x, PO, (:i,))
    @test [p.args[1] for p in obs_pairs] == all_fields
    # `@fprop` recurses, `@wprop` indexes, everything else is carried through.
    @test obs_pairs[1].args[2] == :($(PO).obs_weights_view(x.a, i))
    @test obs_pairs[2].args[2] == :($(PO).nothing_scalar_array_getindex(x.w, i))
    @test obs_pairs[4].args[2] == :(x.plain)
    # The same field, the same tag, a different channel, a different transform.
    factory_w = PO.prop_channel_pairs(:factory, wonly, [:w], :x, PO, ())[1].args[2]
    obs_w = PO.prop_channel_pairs(:obs, wonly, [:w], :x, PO, (:i,))[1].args[2]
    @test factory_w == :($(PO)._wprop(x.w, args...; kwargs...))
    @test obs_w == :($(PO).nothing_scalar_array_getindex(x.w, i))
    @test factory_w != obs_w
    # Indexing, not viewing: a `view` of an `AbstractWeights` is a `SubArray`, which the
    # weighted `Statistics.std` methods do not accept.
    aw = StatsBase.AnalyticWeights(collect(1.0:10.0))
    @test isa(PO.nothing_scalar_array_getindex(aw, 3:5), StatsBase.AnalyticWeights)
    # A table row with no field transform errors rather than taking another tag's transform.
    @test_throws ErrorException PO.prop_tag_expr(:factory, :sixth, :a, :(x.a), PO, ())
    # `@vprop` has no transform in the observation channel, and asking for one errors rather
    # than silently taking the view channel's.
    @test_throws ErrorException PO.prop_tag_expr(:obs, :vprop, :a, :(x.a), PO, (:i,))
    # Every row is complete in every channel that names it.
    @test isnothing(PO.check_prop_tag_macros())
end
