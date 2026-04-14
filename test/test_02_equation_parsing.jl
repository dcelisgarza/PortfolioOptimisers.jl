@safetestset "Equation tests" begin
    using PortfolioOptimisers, Test, Logging
    Logging.disable_logging(Logging.Warn)
    res = parse_equation("2*sqrt(prior(a ,   1b1)) /2*  5 + cbrt(3)^3*f >= 5/5 + d-69/3*c")
    idx = sortperm(res.vars)
    @test res.vars[idx] == ["d", "f", "c", "sqrt(prior(a, 1b1))"][idx]
    @test res.coef[idx] == [-1.0, 3.0, 23.0, 5.0][idx]
    @test res.op == ">="
    @test res.rhs == 1.0
    res = parse_equation("2*sqrt(prior(a ,   b)) /2*  5 - - - -6 + cbrt(3)^3*f  -69/3*c <= - d")
    idx = sortperm(res.vars)
    @test res.vars[idx] == ["sqrt(prior(a, b))", "f", "c", "d"][idx]
    @test res.coef[idx] == [5.0, 3.0, -23.0, 1.0][idx]
    @test res.op == "<="
    @test res.rhs == -6.0
    res = parse_equation(" == 2*sqrt(prior(a ,   b)) /2*  5 - 6 + 7 + -  - cbrt(3)^3 *f  - - 69/ 3*c-d+(3*2)/(3*(1+1))*d")
    idx = sortperm(res.vars)
    @test res.vars[idx] == ["sqrt(prior(a, b))", "f", "c", "d"][idx]
    @test res.coef[idx] == [-5.0, -3.0, -23.0, 0.0][idx]
    @test res.op == "=="
    @test res.rhs == 1.0
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 + cbrt(3)^3*f >= 5/")
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b))   5 + cbrt(3)^3*f + 1/>= 5/5 ")
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b))   5 + cbrt(3)^3*f + 1/ = 5/5 ")
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 ++ cbrt(3)^3*f >= 5/5 + d-69/3*c")
    @test_throws Meta.ParseError parse_equation(:(2 * sqrt(prior(a, b)) / 2 * 5 ++
                                                  cbrt(3)^3 * f >= 5 / 5 + d - 69 / 3 * c))
    @test_throws Meta.ParseError parse_equation("2*sqrt(prior(a ,   b)) /2*  5 + cbrt(3)^3*f >= >= 5/5 + d-69/3*c")
    @test_throws Meta.ParseError parse_equation(:(2 * sqrt(prior(a, b)) / 2 * 5 >=
                                                  cbrt(3)^3 * f >=
                                                  5 / 5 + d - 69 / 3 * c))
    @test_throws Meta.ParseError parse_equation(:(2 * sqrt(prior(a, b)) / 2 * 5 +
                                                  cbrt(3)^3 * f +
                                                  5 / 5 +
                                                  d - 69 / 3 * c))
    res = parse_equation("1/3*x/((2^z)*y)+5z==1-5")
    idx = sortperm(res.vars)
    @test res.vars[idx] == ["z", "(0.3333333333333333x) / (2 ^ z * y)"][idx]
    @test res.coef[idx] == [5.0, 1.0][idx]
    @test res.op == "=="
    @test res.rhs == -4.0
    # @test res.eqn == "5.0*z + (0.3333333333333333x) / (2 ^ z * y) == -4.0"
    res = parse_equation("-x>=-1.0y+2")
    idx = sortperm(res.vars)
    @test res.vars[idx] == ["x", "y"][idx]
    @test res.coef[idx] == [-1.0, 1.0][idx]
    @test res.op == ">="
    @test res.rhs == 2.0
    # @test res.eqn == "-x + y >= 2.0"
    res = parse_equation("-_1*_3<=-1.1y+2")
    idx = sortperm(res.vars)
    @test res.vars[idx] == ["-_1 * _3", "y"][idx]
    @test res.coef[idx] == [1.0, 1.1][idx]
    @test res.op == "<="
    @test res.rhs == 2.0
    # @test res.eqn == "-_1 * _3 + 1.1*y <= 2.0"
    res = parse_equation("Inf*a<=Inf")
    idx = sortperm(res.vars)
    @test res.vars[idx] == ["a"][idx]
    @test res.coef[idx] == [Inf][idx]
    @test res.op == "<="
    @test res.rhs == Inf
    # @test res.eqn == "Inf*a <= Inf"
end
