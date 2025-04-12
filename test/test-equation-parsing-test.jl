@safetestset "Equation parsing tests" begin
    using PortfolioOptimisers
    eqn = "a - 1*b - c*2 + 3*d/4 - 1/5*e*6 + - 7*f*8/9 + 10/11*g*12 + 13/14*h*15/16 - 3*a +--6 --- 6 = -5"
    res = parse_constraint_equation(eqn)
    eqn = "A_a.A - 1*B.b_B - C_c.C*2 + 3*D.d.D/4 - 1/5*E_e_E*6 + - 7*f*8/9 + 10/11*g*12 + 13/14*h*15/16 -- 3*A_a.A +--6 --- 6 <= -5"
    res = parse_constraint_equation(eqn)
end
