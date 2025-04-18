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

        (; ineq, eq) = linear_constraints(LinearConstraint(;
                                                           A = A_LinearConstraint(;
                                                                                  group = nothing,
                                                                                  name = nothing)),
                                          sets)
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

        @test_throws UndefKeywordError A_LinearConstraint(; coef = [2])
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
    @testset "Weight Bounds constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        hcc_1 = WeightBoundsConstraints(; group = :Assets, name = 1, lb = 0.7, ub = 0.8)
        (; lb, ub) = weight_bounds_constraints(hcc_1, sets)
        @test isapprox(lb, [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(ub, [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_2 = WeightBoundsConstraints(; group = [:Assets, :Assets], name = [1, 5],
                                        lb = [0.7, 0.3], ub = [0.8, 0.6])
        (; lb, ub) = weight_bounds_constraints(hcc_2, sets)
        @test isapprox(lb, [0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(ub, [0.8, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_3 = WeightBoundsConstraints(; group = :Clusters, name = 3, lb = 0.2, ub = 0.5)
        (; lb, ub) = weight_bounds_constraints(hcc_3, sets)
        @test isapprox(lb, [0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.2])
        @test isapprox(ub, [1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5])

        @test_throws UndefKeywordError WeightBoundsConstraints(group = :Asset)
        hcc = WeightBoundsConstraints(; group = nothing, name = nothing)
        @test isnothing(hcc.group)
        @test isnothing(hcc.name)

        @test_throws UndefKeywordError WeightBoundsConstraints(; group = [nothing],
                                                               ub = [2], lb = [1])
        lcs = WeightBoundsConstraints(; group = [nothing], name = [nothing], ub = [5],
                                      lb = [3])
        @test isnothing(lcs.group[1])
        @test isnothing(lcs.name[1])

        hcc_1 = WeightBoundsConstraints(; group = :Foo, name = :Bar, lb = 0.7, ub = 0.8)
        @test_throws ArgumentError weight_bounds_constraints(hcc_1, sets; strict = true)

        hcc_1 = WeightBoundsConstraints(; group = [:Foo], name = [:Bar], lb = [0.7],
                                        ub = [0.8])
        @test_throws ArgumentError weight_bounds_constraints(hcc_1, sets; strict = true)

        (; lb, ub) = weight_bounds_constraints(hcc_1, sets)
        @test all(iszero.(lb))
        @test all(isone.(ub))

        hcc_1 = WeightBoundsConstraints(; group = :Foo, name = :Bar, lb = 0.7, ub = 0.8)
        (; lb, ub) = weight_bounds_constraints(hcc_1, sets)
        @test all(iszero.(lb))
        @test all(isone.(ub))
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

        c1 = A_LinearConstraint(; group = nothing, name = nothing)
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
        @test_throws UndefKeywordError risk_budget_constraints([A_LinearConstraint(;
                                                                                   group = [nothing],
                                                                                   coef = [2])],
                                                               sets)
        @test_throws AssertionError risk_budget_constraints(A_LinearConstraint[], sets)
    end
end
