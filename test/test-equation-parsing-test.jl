@safetestset "Equation tests" begin
    using PortfolioOptimisers, Test
    @testset "Equation parsing" begin
        eqn = "- 7*f*8/9 + 10/11*g*12 + a - 1e0*b -c*2 +   3*d/4 - 1/5*e*6  + - -   6 -  - --- 6 =  3*a +- 13/14*h*15/16-5"
        res = parse_constraint_equation(eqn)
        @test res.eqn ==
              "-2.0*a -b -2.0*c + 0.75*d -1.2*e -6.222222222222222*f + 10.909090909090908*g + 0.8705357142857143*h == -5.0"
        @test res.vars == ["a", "b", "c", "d", "e", "f", "g", "h"]
        @test res.comp == "=="
        @test res.cnst == -5.0
        @test res.coef ==
              [1 - 3, -1, -2, 3 / 4, -6 / 5, -7 * 8 / 9, 10 / 11 * 12, 13 / 14 * 15 / 16]

        eqn = "A_a.A - 1*B.b_B - C_c.C*2 + 3*D.d.D/4 - 1/5*E_e_E*6 + - 7*f*8/9 + 10/11*g*12 + 13/14*h*15/16 + --    3*A_a.A+--6 -- - 6 <= 2"
        res = parse_constraint_equation(eqn)
        @test res.eqn ==
              "4.0*A_a.A -B.b_B -2.0*C_c.C + 0.75*D.d.D -1.2*E_e_E -6.222222222222222*f + 10.909090909090908*g + 0.8705357142857143*h <= 2.0"
        @test res.vars == ["A_a.A", "B.b_B", "C_c.C", "D.d.D", "E_e_E", "f", "g", "h"]
        @test res.comp == "<="
        @test res.cnst == 2.0
        @test res.coef ==
              [1 + 3, -1, -2, 3 / 4, -6 / 5, -7 * 8 / 9, 10 / 11 * 12, 13 / 14 * 15 / 16]

        res = parse_constraint_equation("fa-  fe >= 3")
        @test res.eqn == "fa -fe >= 3.0"
        @test res.vars == ["fa", "fe"]
        @test res.comp == ">="
        @test res.cnst == 3.0
        @test res.coef == [1, -1]

        res = parse_constraint_equation("A+2*2b>=6", false)
        @test res.eqn == "A >= 6.0"
        @test res.vars == ["A"]
        @test res.coef == [1]
        @test res.comp == ">="
        @test res.cnst == 6.0

        @test_throws ArgumentError parse_constraint_equation("A+b")
        @test_throws ArgumentError parse_constraint_equation("A+2*2b>=6")
        @test_throws ArgumentError parse_constraint_equation("A+b=>2")
        @test_throws ArgumentError parse_constraint_equation("A+b=<2")
        @test_throws ArgumentError parse_constraint_equation("A+b===2")
        @test_throws ArgumentError parse_constraint_equation("A+b===2")

        res = parse_constraint_equation("1/2*a+3e-5/5e-6-b -d*1e-6/1e-4 +h/1<=3-- 1e-5*f/1e-8 + 1e-5/ 5e-6*f*1e-8 /2e-7 - 2/2*g/1")
        @test res.eqn ==
              "0.5*a -b -0.009999999999999998*d -1000.1000000000001*f + g + h <= -3.0"
        @test res.vars == ["a", "b", "d", "f", "g", "h"]
        @test res.coef ==
              [1 / 2, -1, -1e-6 / 1e-4, -1e-5 / 1e-8 - 1e-5 / 5e-6 * 1e-8 / 2e-7, 1, 1]
        @test res.comp == "<="
        @test res.cnst == -3.0
    end
end
