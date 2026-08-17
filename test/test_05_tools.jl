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
