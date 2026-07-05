@testset "Risk-measure ↔ optimiser compatibility (U5)" begin
    using PortfolioOptimisers, Test, InteractiveUtils

    @testset "supports_risk_measure predicate" begin
        # JuMP optimisers accept RiskMeasure, not hierarchical-only measures.
        @test supports_risk_measure(MeanRisk, Variance)
        @test supports_risk_measure(RiskBudgeting, ConditionalValueatRisk)
        @test supports_risk_measure(NearOptimalCentering, Variance)
        @test supports_risk_measure(FactorRiskContribution, Variance)
        @test !supports_risk_measure(MeanRisk, EqualRisk)
        # Clustering optimisers accept every OptimisationRiskMeasure.
        @test supports_risk_measure(HierarchicalRiskParity, Variance)
        @test supports_risk_measure(HierarchicalRiskParity, EqualRisk)
        @test supports_risk_measure(HierarchicalEqualRiskContribution, EqualRisk)
        # Naive / finite-allocation optimisers accept none.
        @test !supports_risk_measure(EqualWeighted, Variance)
        # Instance forms forward to the type methods.
        @test supports_risk_measure(EqualWeighted(), Variance()) ===
              supports_risk_measure(EqualWeighted, Variance)
    end

    @testset "supported_risk_measures category" begin
        @test supported_risk_measures(MeanRisk) === RiskMeasure
        @test supported_risk_measures(HierarchicalRiskParity) ===
              PortfolioOptimisers.OptimisationRiskMeasure
        @test supported_risk_measures(EqualWeighted) === Union{}
        @test supported_risk_measures(EqualWeighted()) === Union{}
    end

    @testset "predicate derives from category" begin
        for O in (MeanRisk, HierarchicalRiskParity, EqualWeighted),
            R in (Variance, EqualRisk, ConditionalValueatRisk)

            @test supports_risk_measure(O, R) === (R <: supported_risk_measures(O))
        end
    end

    @testset "meta-optimisers delegate (intersection of children)" begin
        jmp() = MeanRisk(; opt = JuMPOptimiser(; slv = Solver(; solver = nothing)))  # RiskMeasure
        none = EqualWeighted()                                                       # Union{}

        # NestedClustered: inner ∩ outer.
        nco_rm = NestedClustered(; opti = jmp(), opto = jmp())
        @test supported_risk_measures(nco_rm) === RiskMeasure
        @test supports_risk_measure(nco_rm, Variance())
        @test !supports_risk_measure(nco_rm, EqualRisk())
        # One child that accepts nothing collapses the whole meta to Union{}.
        nco_none = NestedClustered(; opti = jmp(), opto = none)
        @test supported_risk_measures(nco_none) === Union{}
        @test !supports_risk_measure(nco_none, Variance())

        # SubsetResampling: delegates to its single inner optimiser.
        @test supported_risk_measures(SubsetResampling(; opt = jmp())) === RiskMeasure
        @test supports_risk_measure(SubsetResampling(; opt = jmp()), Variance())
        @test supported_risk_measures(SubsetResampling(; opt = none)) === Union{}

        # Stacking: outer ∩ all inner, broadcasting over the (possibly heterogeneous) inner vector.
        stk_rm = Stacking(; opti = [jmp(), jmp()], opto = jmp())
        @test supported_risk_measures(stk_rm) === RiskMeasure
        @test supports_risk_measure(stk_rm, Variance())
        # An inner optimiser that accepts nothing ⇒ the meta accepts nothing.
        stk_none = Stacking(; opti = [jmp(), none], opto = jmp())
        @test supported_risk_measures(stk_none) === Union{}
        @test !supports_risk_measure(stk_none, Variance())
    end

    @testset "clustering acceptance ⊇ JuMP acceptance (table invariant)" begin
        function leaf_risk_measures(T, acc = Type[])
            subs = subtypes(T)
            isempty(subs) ? push!(acc, T) : foreach(S -> leaf_risk_measures(S, acc), subs)
            return acc
        end
        rms = unique(vcat(leaf_risk_measures(RiskMeasure),
                          leaf_risk_measures(PortfolioOptimisers.HierarchicalRiskMeasure)))
        jump = filter(M -> supports_risk_measure(MeanRisk, M), rms)
        clus = filter(M -> supports_risk_measure(HierarchicalRiskParity, M), rms)
        @test !isempty(jump)
        @test length(clus) == length(rms)          # clustering accepts all of them
        @test issubset(Set(jump), Set(clus))       # and a superset of the JuMP-accepted ones
    end
end
