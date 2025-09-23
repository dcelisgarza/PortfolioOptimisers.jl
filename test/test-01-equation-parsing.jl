@safetestset "Equation tests" begin
    using PortfolioOptimisers, Test, Logging
    Logging.disable_logging(Logging.Warn)
    (; vars, coef, op, rhs, eqn) = parse_equation("2*sqrt(prior(a ,   1b1)) /2*  5 + cbrt(3)^3*f >= 5/5 + d-69/3*c")
    @test vars == ["d", "f", "c", "sqrt(prior(a, 1b1))"]
    @test coef == [-1.0, 3.0, 23.0, 5.0]
    @test op == ">="
    @test rhs == 1.0
    (; vars, coef, op, rhs, eqn) = parse_equation("2*sqrt(prior(a ,   b)) /2*  5 - - - -6 + cbrt(3)^3*f  -69/3*c <= - d")
    @test vars == ["sqrt(prior(a, b))", "f", "c", "d"]
    @test coef == [5.0, 3.0, -23.0, 1.0]
    @test op == "<="
    @test rhs == -6.0
    (; vars, coef, op, rhs, eqn) = parse_equation(" == 2*sqrt(prior(a ,   b)) /2*  5 - 6 + 7 + -  - cbrt(3)^3 *f  - - 69/ 3*c-d+(3*2)/(3*(1+1))*d")
    @test vars == ["sqrt(prior(a, b))", "f", "c", "d"]
    @test coef == [-5.0, -3.0, -23.0, 0.0]
    @test op == "=="
    @test rhs == 1.0
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 + cbrt(3)^3*f >= 5/")
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b))   5 + cbrt(3)^3*f + 1/>= 5/5 ")
    @test_throws ErrorException parse_equation("2*sqrt(prior(a ,   b))   5 + cbrt(3)^3*f + 1/ = 5/5 ")
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 ++ cbrt(3)^3*f >= 5/5 + d-69/3*c")
    res = parse_equation("1/3*x/((2^z)*y)+5z==1-5")
    @test res.vars == ["z", "(0.3333333333333333x) / (2 ^ z * y)"]
    @test res.coef == [5.0, 1.0]
    @test res.op == "=="
    @test res.rhs == -4.0
    @test res.eqn == "5.0*z + (0.3333333333333333x) / (2 ^ z * y) == -4.0"
    res = parse_equation("-x>=-1.0y+2")
    @test res.vars == ["x", "y"]
    @test res.coef == [-1.0, 1.0]
    @test res.op == ">="
    @test res.rhs == 2.0
    @test res.eqn == "-x + y >= 2.0"
end
