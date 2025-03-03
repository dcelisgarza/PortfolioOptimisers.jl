@safetestset "Constraints" begin
    using PortfolioOptimisers, DataFrames, Test

    @testset "Linear Constraints" begin
        assets = 1:10
        asset_sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        lhs_1 = LinearConstraintSide(; group = :Assets, name = 1)
        rhs_1 = LinearConstraintSide(; group = :Assets, name = 1, coef = 0, cnst = 0.35)
        lhs_2 = LinearConstraintSide(; group = :Assets, name = 1)
        rhs_2 = LinearConstraintSide(; group = [:Assets, :Assets], name = [1, 3],
                                     coef = [0, 0.3], cnst = [0.25, 0])
        lhs_3 = LinearConstraintSide(; group = :Assets, name = 5, cnst = -0.5)
        rhs_3 = LinearConstraintSide(;)
        lhs_4 = LinearConstraintSide(; group = :Clusters, name = 3, coef = 2)
        rhs_4 = LinearConstraintSide(; group = :Clusters, name = 2, coef = 3)
        lhs_5 = LinearConstraintSide(; group = [:Clusters, :Clusters], name = [1, 3],
                                     coef = [-1, 2], cnst = [0.3, -0.2])
        rhs_5 = LinearConstraintSide(; group = :Clusters, name = 2, coef = 3)
        lhs_6 = LinearConstraintSide(; group = fill(:Assets, 10), name = assets,
                                     coef = loadings.MTUM,
                                     cnst = collect(Iterators.repeated(0, 10)))
        rhs_6 = LinearConstraintSide(; group = fill(:Assets, 10), name = assets,
                                     coef = loadings.QUAL,
                                     cnst = [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        lhs_7 = LinearConstraintSide(;)
        rhs_7 = LinearConstraintSide(; group = [:Clusters, :Assets], name = [2, 7],
                                     coef = [1, 1], cnst = [0.7, 0])
        lhs_8 = LinearConstraintSide(;)
        rhs_8 = LinearConstraintSide(; group = :Assets, name = 1, coef = -1, cnst = 4)
        lhs_9 = LinearConstraintSide(; group = :Assets, name = 1)
        rhs_9 = LinearConstraintSide(; group = [:Assets, :Assets], name = [1, 3],
                                     coef = [1, -1], cnst = [2, 3])
        lhs_10 = LinearConstraintSide(; group = :Assets, name = 5, cnst = -7)
        rhs_10 = LinearConstraintSide(;)
        lhs_11 = LinearConstraintSide(; group = :Clusters, name = 3, cnst = 5)
        rhs_11 = LinearConstraintSide(; group = :Clusters, name = 2, cnst = 8)
        lhs_12 = LinearConstraintSide(; group = [:Clusters, :Clusters], name = [1, 3],
                                      coef = [1, 1], cnst = [-1, -2])
        rhs_12 = LinearConstraintSide(; group = :Clusters, name = 2, cnst = 4)
        lhs_13 = LinearConstraintSide(;)
        rhs_13 = LinearConstraintSide(; group = [:Clusters, :Assets], name = [2, 7],
                                      coef = [1, 1], cnst = [8, 0])

        constr = [LinearConstraint(; lhs = lhs_1, rhs = rhs_1, comp = EQ()),
                  LinearConstraint(; lhs = lhs_2, rhs = rhs_2, comp = LEQ(),
                                   normalise = true),
                  LinearConstraint(; lhs = lhs_3, rhs = rhs_3, comp = EQ(),
                                   normalise = true),
                  LinearConstraint(; lhs = lhs_4, rhs = rhs_4, comp = GEQ()),
                  LinearConstraint(; lhs = lhs_5, rhs = rhs_5, comp = LEQ()),
                  LinearConstraint(; kind = FactorLinearConstraint(), lhs = lhs_6,
                                   rhs = rhs_6, comp = GEQ(), normalise = true),
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
        A_ineq_t = reshape([1.4285714285714286, 0.0, -1.0, 0.6666666666666666, 0.0, -0.0,
                            -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -0.0, -1.0, -0.4285714285714286,
                            -2.0, 2.0, -0.6666666666666666, 1.0, -1.0, -1.0, 0.0, 3.0, -3.0,
                            0.3333333333333333, 0.0, 1.0, 1.0, 0.0, -2.0, 2.0,
                            0.3333333333333333, 0.0, -1.0, -1.0, 0.0, 3.0, -3.0,
                            0.3333333333333333, 0.0, 1.0, 1.0, 0.0, 3.0, -3.0,
                            -0.3333333333333333, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0,
                            0.3333333333333333, 0.0, -0.0, -1.0, 0.0, -2.0, 2.0,
                            0.3333333333333333, 0.0, -1.0, -1.0, 0.0, -2.0, 2.0,
                            -0.3333333333333333, 0.0, -1.0, -1.0], 7, 10)
        B_ineq_t = [0.25, -0.0, -0.09999999999999998, -0.9, 5.0, -3.0, 7.0]
        A_eq_t = reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0,
                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
                          -2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 10)
        B_eq_t = [0.35, 0.5, 0.7, 4.0, 7.0, 8.0]
        @test isapprox(A_ineq, A_ineq_t)
        @test isapprox(B_ineq, B_ineq_t)
        @test isapprox(A_eq, A_eq_t)
        @test isapprox(B_eq, B_eq_t)
    end
end
