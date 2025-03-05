@safetestset "Constraints" begin
    using PortfolioOptimisers, DataFrames, Test
    @testset "Linear Constraints" begin
        assets = 1:10
        asset_sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        lhs_1 = LinearConstraintAtom(; group = :Assets, name = 1)
        rhs_1 = LinearConstraintAtom(; cnst = 0.35)
        lhs_2 = LinearConstraintAtom(; group = :Assets, name = 1)
        rhs_2 = LinearConstraintAtom(; group = [:Assets, :Assets], name = [1, 3],
                                     coef = [0, 0.3], cnst = 0.25)
        lhs_3 = LinearConstraintAtom(; group = :Assets, name = 5, cnst = -0.5)
        rhs_3 = LinearConstraintAtom(;)
        lhs_4 = LinearConstraintAtom(; group = :Clusters, name = 3, coef = 2)
        rhs_4 = LinearConstraintAtom(; group = :Clusters, name = 2, coef = 3)
        lhs_5 = LinearConstraintAtom(; group = [:Clusters, :Clusters], name = [1, 3],
                                     coef = [-1, 2], cnst = 0.1)
        rhs_5 = LinearConstraintAtom(; group = :Clusters, name = 2, coef = 3)
        lhs_6 = LinearConstraintAtom(; group = fill(:Assets, 10), name = assets,
                                     coef = loadings.MTUM)
        rhs_6 = LinearConstraintAtom(; group = fill(:Assets, 10), name = assets,
                                     coef = loadings.QUAL, cnst = 0.9)
        lhs_7 = LinearConstraintAtom(;)
        rhs_7 = LinearConstraintAtom(; group = [:Clusters, :Assets], name = [2, 7],
                                     coef = [1, 1], cnst = 0.7)
        lhs_8 = LinearConstraintAtom(;)
        rhs_8 = LinearConstraintAtom(; group = :Assets, name = 1, coef = -1, cnst = 4)
        lhs_9 = LinearConstraintAtom(; group = :Assets, name = 1)
        rhs_9 = LinearConstraintAtom(; group = [:Assets, :Assets], name = [1, 3],
                                     coef = [1, -1], cnst = 5)
        lhs_10 = LinearConstraintAtom(; group = :Assets, name = 5, cnst = -7)
        rhs_10 = LinearConstraintAtom(;)
        lhs_11 = LinearConstraintAtom(; group = :Clusters, name = 3, cnst = 5)
        rhs_11 = LinearConstraintAtom(; group = :Clusters, name = 2, cnst = 8)
        lhs_12 = LinearConstraintAtom(; group = [:Clusters, :Clusters], name = [1, 3],
                                      coef = [1, 1], cnst = -3)
        rhs_12 = LinearConstraintAtom(; group = :Clusters, name = 2, cnst = 4)
        lhs_13 = LinearConstraintAtom(;)
        rhs_13 = LinearConstraintAtom(; group = [:Clusters, :Assets], name = [2, 7],
                                      coef = [1, 1], cnst = 8)

        constr = [LinearConstraint(; lhs = lhs_1, rhs = rhs_1, comp = EQ()),
                  LinearConstraint(; lhs = lhs_2, rhs = rhs_2, comp = LEQ()),
                  LinearConstraint(; lhs = lhs_3, rhs = rhs_3, comp = EQ()),
                  LinearConstraint(; lhs = lhs_4, rhs = rhs_4, comp = GEQ()),
                  LinearConstraint(; lhs = lhs_5, rhs = rhs_5, comp = LEQ()),
                  LinearConstraint(; kind = FactorLinearConstraint(), lhs = lhs_6,
                                   rhs = rhs_6, comp = GEQ()),
                  LinearConstraint(; kind = FactorLinearConstraint(), lhs = lhs_7,
                                   rhs = rhs_7, comp = EQ()),
                  LinearConstraint(; lhs = lhs_8, rhs = rhs_8, comp = EQ()),
                  LinearConstraint(; lhs = lhs_9, rhs = rhs_9, comp = LEQ()),
                  LinearConstraint(; lhs = lhs_10, rhs = rhs_10, comp = EQ()),
                  LinearConstraint(; lhs = lhs_11, rhs = rhs_11, comp = GEQ()),
                  LinearConstraint(; kind = FactorLinearConstraint(), lhs = lhs_12,
                                   rhs = rhs_12, comp = LEQ()),
                  LinearConstraint(; lhs = lhs_13, rhs = rhs_13, comp = EQ())]
        A_ineq, B_ineq, A_eq, B_eq = linear_constraints(constr, asset_sets)
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

        a, b, c, d = linear_constraints(LinearConstraint(; lhs = LinearConstraintAtom(),
                                                         rhs = LinearConstraintAtom()),
                                        asset_sets)
        @test isempty(a)
        @test isempty(b)
        @test isempty(c)
        @test isempty(d)

        lhs = LinearConstraintAtom(; group = [:Foo], name = [20], coef = [5])
        rhs = LinearConstraintAtom(; cnst = 0.35)
        a, b, c, d = linear_constraints(LinearConstraint(; lhs = lhs, rhs = rhs),
                                        asset_sets)
        @test isempty(a)
        @test isempty(b)
        @test isempty(c)
        @test isempty(d)

        @test_throws AssertionError LinearConstraintAtom(; group = [nothing], name = ["a"],
                                                         coef = [2], cnst = 0)
        lcs = LinearConstraintAtom(; group = [nothing], name = [nothing], coef = [2],
                                   cnst = 0)
        @test isnothing(lcs.group[1])
        @test isnothing(lcs.name[1])

        lhs_1 = LinearConstraintAtom(; group = :Assets, name = 1)
        rhs_1 = LinearConstraintAtom(; cnst = 0.35)
        constr = LinearConstraint(; lhs = lhs_1, rhs = rhs_1, comp = EQ())
        @test_throws ArgumentError linear_constraints(constr, asset_sets, strict = true)

        lhs_1 = LinearConstraintAtom(; group = :Assets, name = 1)
        rhs_1 = LinearConstraintAtom(; group = [:Foo], name = [:Bar], coef = [1],
                                     cnst = 0.35)
        constr = LinearConstraint(; lhs = lhs_1, rhs = rhs_1, comp = EQ())
        @test_throws ArgumentError linear_constraints(constr, asset_sets, strict = true)
    end
    @testset "Hierarchical constraints" begin
        assets = 1:10
        asset_sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        hcc_1 = HierarchicalConstraint(; group = :Assets, name = 1, lo = 0.7, hi = 0.8)
        w_min, w_max = hc_constraints(hcc_1, asset_sets)
        @test isapprox(w_min, [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(w_max, [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_2 = HierarchicalConstraint(; group = [:Assets, :Assets], name = [1, 5],
                                       lo = [0.7, 0.3], hi = [0.8, 0.6])
        w_min, w_max = hc_constraints(hcc_2, asset_sets)
        @test isapprox(w_min, [0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(w_max, [0.8, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_3 = HierarchicalConstraint(; group = :Clusters, name = 3, lo = 0.2, hi = 0.5)
        w_min, w_max = hc_constraints(hcc_3, asset_sets)
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
        @test_throws ArgumentError hc_constraints(hcc_1, asset_sets, strict = true)

        hcc_1 = HierarchicalConstraint(; group = [:Foo], name = [:Bar], lo = [0.7],
                                       hi = [0.8])
        @test_throws ArgumentError hc_constraints(hcc_1, asset_sets, strict = true)

        w_min, w_max = hc_constraints(hcc_1, asset_sets)
        @test all(iszero.(w_min))
        @test all(isone.(w_max))
    end
end
