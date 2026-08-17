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
