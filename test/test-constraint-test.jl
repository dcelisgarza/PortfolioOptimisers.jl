@safetestset "Constraints" begin
    using PortfolioOptimisers, DataFrames, Test, Random, StableRNGs, LinearAlgebra
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    @testset "Linear Constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        constr = [LinearConstraint(; A = LinearConstraintSide(; group = :Assets, name = 1),
                                   B = 0.35, comp = EQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(;
                                                            group = [:Assets, :Assets,
                                                                     :Assets],
                                                            name = [1, 1, 3],
                                                            coef = [1, -0, -0.3]), B = 0.25,
                                   comp = LEQ()),
                  LinearConstraint(; A = LinearConstraintSide(; group = :Assets, name = 5),
                                   B = 0.5, comp = EQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(;
                                                            group = [:Clusters, :Clusters],
                                                            name = [3, 2], coef = [2, -3]),
                                   comp = GEQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(;
                                                            group = [:Clusters, :Clusters,
                                                                     :Clusters],
                                                            name = [1, 3, 2],
                                                            coef = [-1, 2, -3]), B = -0.1,
                                   comp = LEQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(; group = fill(:Assets, 20),
                                                            name = [assets; assets],
                                                            coef = [-loadings.MTUM;
                                                                    loadings.QUAL]),
                                   B = 0.9, comp = GEQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(; group = [:Clusters, :Assets],
                                                            name = [2, 7], coef = -[1, 1]),
                                   B = 0.7, comp = EQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(; group = :Assets, name = 1,
                                                            coef = 1), B = 4, comp = EQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(;
                                                            group = [:Assets, :Assets,
                                                                     :Assets],
                                                            name = [1, 1, 3],
                                                            coef = [1, -1, 1]), B = 5,
                                   comp = LEQ()),
                  LinearConstraint(; A = LinearConstraintSide(; group = :Assets, name = 5),
                                   B = 7, comp = EQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(;
                                                            group = [:Clusters, :Clusters],
                                                            name = [3, 2], coef = [1, -1]),
                                   B = 3, comp = GEQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(;
                                                            group = [:Clusters, :Clusters,
                                                                     :Clusters],
                                                            name = [1, 3, 2],
                                                            coef = -[1, 1, -1]), B = 7,
                                   comp = LEQ()),
                  LinearConstraint(;
                                   A = LinearConstraintSide(; group = [:Clusters, :Assets],
                                                            name = [2, 7], coef = [-1, -1]),
                                   B = 8, comp = EQ())]
        constr_result = linear_constraints(constr, sets)
        constr_result2 = linear_constraints(constr_result, sets)
        @test constr_result === constr_result2
        (; ineq, eq) = constr_result
        A_ineq, B_ineq = ineq.A, ineq.B
        A_eq, B_eq = eq.A, eq.B
        @test A_ineq === constr_result.A_ineq
        @test B_ineq === constr_result.B_ineq
        @test A_eq === constr_result.A_eq
        @test B_eq === constr_result.B_eq
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

        @test isnothing(linear_constraints(LinearConstraint(;
                                                            A = LinearConstraintSide(;
                                                                                     group = nothing,
                                                                                     name = nothing)),
                                           sets))

        @test isnothing(linear_constraints(LinearConstraint(;
                                                            A = LinearConstraintSide(;
                                                                                     group = [:Foo],
                                                                                     name = [20],
                                                                                     coef = [5]),
                                                            B = 0.35), sets))

        @test_throws UndefKeywordError LinearConstraintSide(; coef = [2])
        lcs = LinearConstraintSide(; group = [nothing], name = [nothing], coef = [2])
        @test isnothing(lcs.group[1])
        @test isnothing(lcs.name[1])

        lhs_1 = LinearConstraintSide(; group = :Asset, name = 1)
        constr = LinearConstraint(; A = lhs_1, B = 0.35, comp = EQ())
        @test_throws ArgumentError linear_constraints(constr, sets, strict = true)

        lhs_1 = LinearConstraintSide(; group = [:Asset, :Foo], name = [1, :Bar],
                                     coef = [1, -1])
        constr = LinearConstraint(; A = lhs_1, B = 0.35, comp = EQ())
        @test_throws ArgumentError linear_constraints(constr, sets, strict = true)
        @test isnothing(linear_constraints(nothing))
    end
    @testset "Cardinality Constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        constr = [CardinalityConstraint(;
                                        A = CardinalityConstraintSide(; group = :Assets,
                                                                      name = 1),
                                        comp = EQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Assets,
                                                                               :Assets,
                                                                               :Assets],
                                                                      name = [1, 1, 3]),
                                        B = 5, comp = LEQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(; group = :Assets,
                                                                      name = 5), B = 1,
                                        comp = EQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Clusters,
                                                                               :Clusters],
                                                                      name = [3, 2]),
                                        comp = GEQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Clusters,
                                                                               :Clusters,
                                                                               :Clusters],
                                                                      name = [1, 3, 2]),
                                        B = 3, comp = LEQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Clusters,
                                                                               :Assets],
                                                                      name = [2, 7]), B = 7,
                                        comp = EQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(; group = :Assets,
                                                                      name = 1), B = 4,
                                        comp = EQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Assets,
                                                                               :Assets,
                                                                               :Assets],
                                                                      name = [1, 1, 3]),
                                        B = 5, comp = LEQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(; group = :Assets,
                                                                      name = 5), B = 3,
                                        comp = EQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Clusters,
                                                                               :Clusters],
                                                                      name = [3, 2]), B = 2,
                                        comp = GEQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Clusters,
                                                                               :Clusters,
                                                                               :Clusters],
                                                                      name = [1, 3, 2]),
                                        B = 8, comp = LEQ()),
                  CardinalityConstraint(;
                                        A = CardinalityConstraintSide(;
                                                                      group = [:Clusters,
                                                                               :Assets],
                                                                      name = [2, 7]), B = 9,
                                        comp = EQ())]

        constr_result = cardinality_constraints(constr, sets)
        constr_result2 = cardinality_constraints(constr_result, sets)
        @test constr_result === constr_result2
        (; ineq, eq) = constr_result
        A_ineq, B_ineq = ineq.A, ineq.B
        A_eq, B_eq = eq.A, eq.B
        A_ineq_t = reshape([2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                            -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                            -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                            -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                            -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0], 6,
                           10)
        B_ineq_t = [5.0, -1.0, 3.0, 5.0, -2.0, 8.0]
        A_eq_t = reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0,
                          0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6, 10)
        B_eq_t = [1.0, 1.0, 7.0, 4.0, 3.0, 9.0]
        @test isapprox(A_ineq, A_ineq_t)
        @test isapprox(B_ineq, B_ineq_t)
        @test isapprox(A_eq, A_eq_t)
        @test isapprox(B_eq, B_eq_t)

        @test isnothing(cardinality_constraints(CardinalityConstraint(;
                                                                      A = CardinalityConstraintSide(;
                                                                                                    group = nothing,
                                                                                                    name = nothing)),
                                                sets))

        @test isnothing(cardinality_constraints(CardinalityConstraint(;
                                                                      A = CardinalityConstraintSide(;
                                                                                                    group = [:Foo],
                                                                                                    name = [20]),
                                                                      B = 5), sets))

        @test_throws UndefKeywordError CardinalityConstraintSide(; name = [2])
        lcs = CardinalityConstraintSide(; group = [nothing], name = [nothing])
        @test isnothing(lcs.group[1])
        @test isnothing(lcs.name[1])

        lhs_1 = CardinalityConstraintSide(; group = :Asset, name = 1)
        constr = CardinalityConstraint(; A = lhs_1, B = 2, comp = EQ())
        @test_throws ArgumentError cardinality_constraints(constr, sets, strict = true)

        lhs_1 = CardinalityConstraintSide(; group = [:Asset, :Foo], name = [1, :Bar])
        constr = CardinalityConstraint(; A = lhs_1, B = 9, comp = EQ())
        @test_throws ArgumentError cardinality_constraints(constr, sets, strict = true)
        @test isnothing(cardinality_constraints(nothing))
    end
    @testset "Weight Bounds constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        loadings = DataFrame(; MTUM = [3, 1, 1, 3, 4, 3, 1, 2, 4, 2],
                             QUAL = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])

        hcc_1 = WeightBoundsConstraint(; group = :Assets, name = 1, lb = 0.7, ub = 0.8)
        wb_result = weight_bounds_constraints(hcc_1, sets)
        wb_result2 = weight_bounds_constraints(wb_result, sets)
        @test wb_result === wb_result2
        (; lb, ub) = wb_result
        @test isapprox(lb, [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(ub, [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_2 = WeightBoundsConstraint(; group = [:Assets, :Assets], name = [1, 5],
                                       lb = [0.7, 0.3], ub = [0.8, 0.6])
        (; lb, ub) = weight_bounds_constraints(hcc_2, sets)
        @test isapprox(lb, [0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(ub, [0.8, 1.0, 1.0, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 1.0])

        hcc_3 = WeightBoundsConstraint(; group = :Clusters, name = 3, lb = 0.2, ub = 0.5)
        (; lb, ub) = weight_bounds_constraints(hcc_3, sets)
        @test isapprox(lb, [0.0, 0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2, 0.2])
        @test isapprox(ub, [1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5])

        hcc_4 = WeightBoundsConstraint(; group = :Assets, name = 1, lb = nothing,
                                       ub = nothing)
        (; lb, ub) = weight_bounds_constraints(hcc_4, sets; strict = true)
        @test isapprox(lb, [-Inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        @test isapprox(ub, [Inf, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        @test_throws UndefKeywordError WeightBoundsConstraint(group = :Asset)
        hcc = WeightBoundsConstraint(; group = nothing, name = nothing)
        @test isnothing(hcc.group)
        @test isnothing(hcc.name)

        @test_throws UndefKeywordError WeightBoundsConstraint(; group = [nothing], ub = [2],
                                                              lb = [1])
        lcs = WeightBoundsConstraint(; group = [nothing], name = [nothing], ub = [5],
                                     lb = [3])
        @test isnothing(lcs.group[1])
        @test isnothing(lcs.name[1])

        hcc_1 = WeightBoundsConstraint(; group = :Foo, name = :Bar, lb = 0.7, ub = 0.8)
        @test_throws ArgumentError weight_bounds_constraints(hcc_1, sets; strict = true)

        hcc_1 = WeightBoundsConstraint(; group = [:Foo], name = [:Bar], lb = [0.7],
                                       ub = [0.8])
        @test_throws ArgumentError weight_bounds_constraints(hcc_1, sets; strict = true)

        (; lb, ub) = weight_bounds_constraints(hcc_1, sets)
        @test all(iszero, lb)
        @test all(isone, ub)

        hcc_1 = WeightBoundsConstraint(; group = :Foo, name = :Bar, lb = 0.7, ub = 0.8)
        (; lb, ub) = weight_bounds_constraints(hcc_1, sets)
        @test all(iszero, lb)
        @test all(isone, ub)

        (; lb, ub) = weight_bounds_constraints(nothing; N = 5)
        @test lb == fill(-Inf, 5)
        @test ub == fill(Inf, 5)

        wb = weight_bounds_constraints(nothing)
        @test WeightBoundsResult(-Inf, Inf) == wb

        wb = WeightBoundsResult(; lb = 0.5, ub = 0.7)
        (; lb, ub) = wb
        @test lb == 0.5
        @test ub == 0.7

        wb2 = weight_bounds_constraints(wb)
        @test wb === wb2

        (; lb, ub) = weight_bounds_constraints(wb; N = 5)
        @test lb == range(; start = 0.5, stop = 0.5, length = 5)
        @test ub == range(; start = 0.7, stop = 0.7, length = 5)

        wb = WeightBoundsResult(; lb = 0.5, ub = [0.7])
        @test wb.lb == 0.5
        @test wb.ub == [0.7]

        wb = WeightBoundsResult(; lb = [0.5], ub = 0.7)
        @test wb.lb == [0.5]
        @test wb.ub == 0.7

        wb = WeightBoundsResult(; lb = nothing, ub = [0.7])
        @test isnothing(wb.lb)
        @test wb.ub == [0.7]
        wb = weight_bounds_constraints(wb; N = 1)
        @test all(wb.lb .== -Inf)

        wb = WeightBoundsResult(; lb = [0.5], ub = nothing)
        @test wb.lb == [0.5]
        @test isnothing(wb.ub)
        wb = weight_bounds_constraints(wb; N = 1)
        @test all(wb.ub .== Inf)

        wb = WeightBoundsResult(; lb = nothing, ub = [0.7])
        @test isnothing(wb.lb)
        @test wb.ub == [0.7]

        wb = WeightBoundsResult(; lb = nothing, ub = 0.7)
        @test isnothing(wb.lb)
        @test wb.ub == 0.7

        wb = WeightBoundsResult(; lb = [0.5], ub = nothing)
        @test wb.lb == [0.5]
        @test isnothing(wb.ub)

        wb = WeightBoundsResult(; lb = 0.5, ub = nothing)
        @test wb.lb == 0.5
        @test isnothing(wb.ub)
    end
    @testset "Risk budget constraints" begin
        assets = 1:10
        sets = DataFrame(; Assets = assets, Clusters = [1, 1, 3, 2, 3, 2, 2, 1, 3, 3])
        c1 = LinearConstraintSide(; group = :Assets, name = 1)
        c2 = LinearConstraintSide(; group = [:Assets, :Assets], name = [1, 3],
                                  coef = [0.6, 0.3])
        c3 = LinearConstraintSide(; group = :Clusters, name = 3, coef = 0.25)
        c4 = LinearConstraintSide(; group = [:Clusters, :Clusters, :Clusters],
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

        c1 = LinearConstraintSide(; group = nothing, name = nothing)
        rb = risk_budget_constraints(c1, sets)
        @test isapprox(rb, fill(inv(10), 10))
        @test_throws ArgumentError risk_budget_constraints(c1, sets, strict = true)

        c1 = LinearConstraintSide(; group = [nothing], name = [nothing], coef = [1])
        rb = risk_budget_constraints(c1, sets)
        @test isapprox(rb, fill(inv(10), 10))
        @test_throws ArgumentError risk_budget_constraints(c1, sets, strict = true)

        c1 = LinearConstraintSide(; group = :Assets, name = -1)
        @test_throws ArgumentError risk_budget_constraints(c1, sets; strict = true)
        rb = risk_budget_constraints(c1, sets; strict = false)
        @test isapprox(rb, fill(1 / 10, 10))

        c1 = LinearConstraintSide(; group = [:Assets], name = [-1], coef = [1])
        @test_throws ArgumentError risk_budget_constraints(c1, sets; strict = true)

        rb = risk_budget_constraints(c1, sets; strict = false)
        @test isapprox(rb, fill(1 / 10, 10))

        rb = risk_budget_constraints(c1, sets; strict = false)
        @test_throws UndefKeywordError risk_budget_constraints([LinearConstraintSide(;
                                                                                     group = [nothing],
                                                                                     coef = [2])],
                                                               sets)
        @test_throws AssertionError risk_budget_constraints(LinearConstraintSide[], sets)
        w = rand(10)
        rb = risk_budget_constraints(w)
        @test rb === w
    end
    @testset "Philogeny constraints" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)

        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator(), p = 0.1)
        res = philogeny_constraints(plc, X)
        @test res === philogeny_constraints(res)
        @test isapprox(res.A, philogeny_matrix(NetworkEstimator(), X))
        @test res.p == 0.1

        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = ClusteringEstimator(),
                                                       p = 0.15)
        res = philogeny_constraints(plc, X)
        @test res === philogeny_constraints(res)
        @test isapprox(res.A, philogeny_matrix(ClusteringEstimator(), X))
        @test res.p == 0.15

        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator(), B = 2)
        res = philogeny_constraints(plc, X)
        @test res === philogeny_constraints(res)
        @test isapprox(res.A, philogeny_matrix(NetworkEstimator(), X) + I)
        @test res.B == 2
        @test isapprox(res.scale, 1e5)

        B = fill(1, 20)
        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator(), B = B)
        res = philogeny_constraints(plc, X)
        @test res === philogeny_constraints(res)
        @test isapprox(res.A, philogeny_matrix(NetworkEstimator(), X) + I)
        @test res.B === B

        plc = IntegerPhilogenyConstraintEstimator(; pe = ClusteringEstimator(), scale = 1e3)
        res = philogeny_constraints(plc, X)
        @test res === philogeny_constraints(res)
        @test isapprox(res.A,
                       unique(philogeny_matrix(ClusteringEstimator(), X) + I; dims = 1))
        @test res.B == 1
        @test res.scale == 1e3

        plc = IntegerPhilogenyConstraintEstimator(;
                                                  pe = ClusteringEstimator(;
                                                                           onc = OptimalNumberClusters(;
                                                                                                       max_k = 2)),
                                                  B = [3, 5])
        res = philogeny_constraints(plc, X)
        @test res === philogeny_constraints(res)
        @test isapprox(res.A,
                       unique(philogeny_matrix(ClusteringEstimator(), X) + I; dims = 1))
        @test res.B == [3, 5]

        pe = ClusteringEstimator(;
                                 onc = OptimalNumberClusters(;
                                                             alg = PredefinedNumberClusters(;
                                                                                            k = 3),
                                                             max_k = 4))
        plc = IntegerPhilogenyConstraintEstimator(; pe = pe, B = [3, 5, 4])
        res = philogeny_constraints(plc, X)
        @test res === philogeny_constraints(res)
        @test isapprox(res.A, unique(philogeny_matrix(pe, X) + I; dims = 1))
        @test res.B == [3, 5, 4]

        @test isnothing(philogeny_constraints(nothing))
    end
    @testset "Centrality constraints" begin
        rng = StableRNG(123456789)
        X = randn(rng, 1000, 20)
        ces = [CentralityEstimator(), CentralityEstimator(; cent = EigenvectorCentrality()),
               CentralityEstimator(; cent = ClosenessCentrality()),
               CentralityEstimator(; cent = StressCentrality()),
               CentralityEstimator(; cent = RadialityCentrality())]

        ce1 = CentralityConstraintEstimator(; A = ces[1], B = MinValue(), comp = GEQ())
        ce2 = CentralityConstraintEstimator(; A = ces[2], B = MeanValue(), comp = LEQ())
        ce3 = CentralityConstraintEstimator(; A = ces[3], B = MedianValue(), comp = EQ())
        ce4 = CentralityConstraintEstimator(; A = ces[4], B = 50, comp = EQ())
        ce5 = CentralityConstraintEstimator(; A = ces[5], B = MaxValue(), comp = LEQ())
        ccs = [ce1, ce2, ce3, ce4, ce5]
        lcs = centrality_constraints(ccs, X)
        lcs2 = centrality_constraints(lcs, X)
        @test lcs === lcs2

        eq = 1
        ineq = 1
        for (i, ce) ∈ enumerate(ccs)
            d, flag_ineq = PortfolioOptimisers.comparison_sign_ineq_flag(ce.comp)
            A = d * centrality_vector(ce.A, X)
            B = d * PortfolioOptimisers.vec_to_real_measure(ce.B, A)
            if flag_ineq
                res = isapprox(lcs.A_ineq[ineq, :], A)
                if !res
                    println("A_ineq fails on index $ineq")
                    find_tol(lcs.A_ineq[ineq, :], A; name1 = :A_ineq, name2 = :A)
                end
                @test res
                res = isapprox(lcs.B_ineq[ineq], B)
                if !res
                    println("B_ineq fails on index $ineq")
                    find_tol(lcs.B_ineq[ineq], B; name1 = :B_ineq, name2 = :B)
                end
                @test res
                ineq += 1
            else
                res = isapprox(lcs.A_eq[eq, :], A)
                if !res
                    println("A_eq fails on index $eq")
                    find_tol(lcs.A_eq[eq, :], A; name1 = :A_eq, name2 = :A)
                end
                @test res
                res = isapprox(lcs.B_eq[eq], B)
                if !res
                    println("B_eq fails on index $eq")
                    find_tol(lcs.B_eq[eq], B; name1 = :B_eq, name2 = :B)
                end
                @test res
                eq += 1
            end
        end

        @test isnothing(centrality_constraints(nothing))
    end
end
