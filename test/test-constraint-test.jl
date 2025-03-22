@safetestset "Constraints" begin
    using PortfolioOptimisers, DataFrames, Test
    @testset "Linear Constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        constr = [LinearConstraint(; A = A_LinearConstraint(; group = :Assets, name = 1),
                                   B = 0.35, comp = EQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(;
                                                          group = [:Assets, :Assets,
                                                                   :Assets],
                                                          name = [1, 1, 3],
                                                          coef = [1, -0, -0.3]), B = 0.25,
                                   comp = LEQ()),
                  LinearConstraint(; A = A_LinearConstraint(; group = :Assets, name = 5),
                                   B = 0.5, comp = EQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(; group = [:Clusters, :Clusters],
                                                          name = [3, 2], coef = [2, -3]),
                                   comp = GEQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(;
                                                          group = [:Clusters, :Clusters,
                                                                   :Clusters],
                                                          name = [1, 3, 2],
                                                          coef = [-1, 2, -3]), B = -0.1,
                                   comp = LEQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(; group = fill(:Assets, 20),
                                                          name = [assets; assets],
                                                          coef = [-loadings.MTUM;
                                                                  loadings.QUAL]), B = 0.9,
                                   comp = GEQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(; group = [:Clusters, :Assets],
                                                          name = [2, 7], coef = -[1, 1]),
                                   B = 0.7, comp = EQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(; group = :Assets, name = 1,
                                                          coef = 1), B = 4, comp = EQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(;
                                                          group = [:Assets, :Assets,
                                                                   :Assets],
                                                          name = [1, 1, 3],
                                                          coef = [1, -1, 1]), B = 5,
                                   comp = LEQ()),
                  LinearConstraint(; A = A_LinearConstraint(; group = :Assets, name = 5),
                                   B = 7, comp = EQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(; group = [:Clusters, :Clusters],
                                                          name = [3, 2], coef = [1, -1]),
                                   B = 3, comp = GEQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(;
                                                          group = [:Clusters, :Clusters,
                                                                   :Clusters],
                                                          name = [1, 3, 2],
                                                          coef = -[1, 1, -1]), B = 7,
                                   comp = LEQ()),
                  LinearConstraint(;
                                   A = A_LinearConstraint(; group = [:Clusters, :Assets],
                                                          name = [2, 7], coef = [-1, -1]),
                                   B = 8, comp = EQ())]

        (; ineq, eq) = linear_constraints(constr, sets)
        A_ineq, B_ineq = ineq.A, ineq.B
        A_eq, B_eq = eq.A, eq.B
        A_ineq_t = reshape([1.0, -0.0, -1.0, 2.0, 0.0, -0.0, -1.0, 0.0, -0.0, -1.0, 0.0,
                            0.0, -0.0, -1.0, -0.3, -2.0, 2.0, -2.0, 1.0, -1.0, -1.0, 0.0,
                            3.0, -3.0, 1.0, 0.0, 1.0, 1.0, 0.0, -2.0, 2.0, 1.0, 0.0, -1.0,
                            -1.0, 0.0, 3.0, -3.0, 1.0, 0.0, 1.0, 1.0, 0.0, 3.0, -3.0, -1.0,
                            0.0, 1.0, 1.0, 0.0, -0.0, -1.0, 1.0, 0.0, -0.0, -1.0, 0.0, -2.0,
                            2.0, 1.0, 0.0, -1.0, -1.0, 0.0, -2.0, 2.0, -1.0, 0.0, -1.0,
                            -1.0], 7, 10)
        B_ineq_t = [0.25, -0.0, -0.1, -0.9, 5.0, -3.0, 7.0]
        A_eq_t = reshape([1.0, 0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
                          1.0, -0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0,
                          0.0, -2.0, 0.0, 0.0, -2.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0], 6, 10)
        B_eq_t = [0.35, 0.5, 0.7, 4.0, 7.0, 8.0]
        @test isapprox(A_ineq, A_ineq_t)
        @test isapprox(B_ineq, B_ineq_t)
        @test isapprox(A_eq, A_eq_t)
        @test isapprox(B_eq, B_eq_t)

        (; ineq, eq) = linear_constraints(LinearConstraint(;), sets)
        A_ineq, B_ineq = ineq.A, ineq.B
        A_eq, B_eq = eq.A, eq.B
        @test isnothing(A_ineq)
        @test isnothing(B_ineq)
        @test isnothing(A_eq)
        @test isnothing(B_eq)

        (; ineq, eq) = linear_constraints(LinearConstraint(;
                                                           A = A_LinearConstraint(;
                                                                                  group = [:Foo],
                                                                                  name = [20],
                                                                                  coef = [5]),
                                                           B = 0.35), sets)
        A_ineq, B_ineq = ineq.A, ineq.B
        A_eq, B_eq = eq.A, eq.B
        @test isnothing(A_ineq)
        @test isnothing(B_ineq)
        @test isnothing(A_eq)
        @test isnothing(B_eq)

        @test_throws AssertionError A_LinearConstraint(; group = [nothing], name = ["a"],
                                                       coef = [2])
        lcs = A_LinearConstraint(; group = [nothing], name = [nothing], coef = [2])
        @test isnothing(lcs.group[1])
        @test isnothing(lcs.name[1])

        lhs_1 = A_LinearConstraint(; group = :Asset, name = 1)
        constr = LinearConstraint(; A = lhs_1, B = 0.35, comp = EQ())
        @test_throws ArgumentError linear_constraints(constr, sets, strict = true)

        lhs_1 = A_LinearConstraint(; group = [:Asset, :Foo], name = [1, :Bar],
                                   coef = [1, -1])
        constr = LinearConstraint(; A = lhs_1, B = 0.35, comp = EQ())
        @test_throws ArgumentError linear_constraints(constr, sets, strict = true)
    end
    @testset "Hierarchical constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        hcc_1 = HierarchicalConstraint(; group = :Assets, name = 1, lo = 0.7, hi = 0.8)
        (; w_min, w_max) = hc_constraints(hcc_1, sets)
        @test isapprox(w_min, [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(w_max, [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_2 = HierarchicalConstraint(; group = [:Assets, :Assets], name = [1, 5],
                                       lo = [0.7, 0.3], hi = [0.8, 0.6])
        (; w_min, w_max) = hc_constraints(hcc_2, sets)
        @test isapprox(w_min, [0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(w_max, [0.8, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_3 = HierarchicalConstraint(; group = :Clusters, name = 3, lo = 0.2, hi = 0.5)
        (; w_min, w_max) = hc_constraints(hcc_3, sets)
        @test isapprox(w_min, [0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.2])
        @test isapprox(w_max, [1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5])

        @test_throws AssertionError HierarchicalConstraint(group = :Asset)
        hcc = HierarchicalConstraint()
        @test isnothing(hcc.group)
        @test isnothing(hcc.name)

        @test_throws AssertionError HierarchicalConstraint(; group = [nothing],
                                                           name = ["a"], hi = [2], lo = [1])
        lcs = HierarchicalConstraint(; group = [nothing], name = [nothing], hi = [5],
                                     lo = [3])
        @test isnothing(lcs.group[1])
        @test isnothing(lcs.name[1])

        hcc_1 = HierarchicalConstraint(; group = :Foo, name = :Bar, lo = 0.7, hi = 0.8)
        @test_throws ArgumentError hc_constraints(hcc_1, sets, strict = true)

        hcc_1 = HierarchicalConstraint(; group = [:Foo], name = [:Bar], lo = [0.7],
                                       hi = [0.8])
        @test_throws ArgumentError hc_constraints(hcc_1, sets, strict = true)

        (; w_min, w_max) = hc_constraints(hcc_1, sets)
        @test all(iszero.(w_min))
        @test all(isone.(w_max))

        hcc_1 = HierarchicalConstraint(; group = :Foo, name = :Bar, lo = 0.7, hi = 0.8)
        (; w_min, w_max) = hc_constraints(hcc_1, sets)
        @test all(iszero.(w_min))
        @test all(isone.(w_max))
    end
    @testset "Risk budget constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        c1 = A_LinearConstraint(; group = :Assets, name = 1)
        c2 = A_LinearConstraint(; group = [:Assets, :Assets], name = [1, 3],
                                coef = [0.6, 0.3])
        c3 = A_LinearConstraint(; group = :Clusters, name = 3, coef = 0.25)
        c4 = A_LinearConstraint(; group = [:Clusters, :Clusters, :Clusters],
                                name = [1, 2, 3],
                                coef = [inv(length(sets[!, :Clusters][sets[!, :Clusters] .== i]))
                                        for i ∈ 1:3])

        rb = risk_budget_constraints(c1, sets)
        @test isapprox(rb[1] / rb[2], 10)

        rb = risk_budget_constraints(c2, sets)
        @test isapprox(rb[1] / rb[2], 6)
        @test isapprox(rb[3] / rb[2], 3)

        rb = risk_budget_constraints(c3, sets)
        idx = sets[!, :Clusters] .== 3
        @test isapprox(rb[idx][1] / rb[.!idx][1], 2.5)

        rb = risk_budget_constraints(c4, sets)
        idx1 = sets[!, :Clusters] .== 1
        idx2 = sets[!, :Clusters] .== 2
        idx3 = sets[!, :Clusters] .== 3
        @test isapprox(rb[idx1][1], 1 / 3 / 3)
        @test isapprox(rb[idx2][1], 1 / 3 / 3)
        @test isapprox(rb[idx3][1], 1 / 4 / 3)

        c1 = A_LinearConstraint(;)
        rb = risk_budget_constraints(c1, sets)
        @test isapprox(rb, fill(inv(10), 10))
        @test_throws ArgumentError risk_budget_constraints(c1, sets, strict = true)

        c1 = A_LinearConstraint(; group = [nothing], name = [nothing], coef = [1])
        rb = risk_budget_constraints(c1, sets)
        @test isapprox(rb, fill(inv(10), 10))
        @test_throws ArgumentError risk_budget_constraints(c1, sets, strict = true)

        c1 = A_LinearConstraint(; group = :Assets, name = -1)
        @test_throws ArgumentError risk_budget_constraints(c1, sets; strict = true)
        rb = risk_budget_constraints(c1, sets; strict = false)
        @test isapprox(rb, fill(1 / 10, 10))

        c1 = A_LinearConstraint(; group = [:Assets], name = [-1], coef = [1])
        @test_throws ArgumentError risk_budget_constraints(c1, sets; strict = true)

        rb = risk_budget_constraints(c1, sets; strict = false)
        @test isapprox(rb, fill(1 / 10, 10))

        rb = risk_budget_constraints(c1, sets; strict = false)
        @test_throws AssertionError risk_budget_constraints([A_LinearConstraint(;
                                                                                group = [nothing],
                                                                                name = ["a"],
                                                                                coef = [2])],
                                                            set)
    end
end
