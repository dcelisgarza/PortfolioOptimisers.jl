# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
abstract type BaseGerberIQCovariance <: BaseGerberCovariance end
abstract type GerberIQCovarianceAlgorithm <: AbstractMomentAlgorithm end
function clamp_gerber_iq_n(kind::GerberIQCovarianceAlgorithm, args...)
    return kind
end
abstract type GerberIQTauEstimator <: AbstractEstimator end
const GerberIQTau = Union{<:Integer, Function, <:GerberIQTauEstimator}
function gerber_iq_tau(t::Integer, X::MatNum)
    return size(X, 1) - t
end
function gerber_iq_tau(t::Function, X::MatNum)
    return t(X)::Integer
end
function gerber_iq_tau(::Option{<:GerberIQTauEstimator}, X::MatNum)
    T, N = size(X)
    return round(Int, T - T / N)
end
abstract type GerberIQGammaEstimator <: AbstractEstimator end
const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}
function gerber_iq_gamma(y::Real, X::MatNum)
    return y
end
function gerber_iq_gamma(y::Function, X::MatNum)
    return y(X)::Number
end
function gerber_iq_gamma(::Option{<:GerberIQGammaEstimator}, X::MatNum)
    return log(2) / size(X, 2)
end
abstract type GerberIQScalerEstimator <: AbstractEstimator end
const GerberIQScaler = Union{Function, <:GerberIQScalerEstimator}
function gerber_iq_scaling(::Option{<:GerberIQScalerEstimator})
    return (x, y) -> (x + y) / 2
end
function gerber_iq_scaling(sca::Function)
    return sca
end
"""
```
            4 ┬─────┰───────────┬─────┬─────┬───────────┰─────┐
     ┌────    │  1  ┃    n^2    ╎     │     ╎    n^2    ┃  1  │
  d ─┤      3 ┾━━━━━╋━━━━━━━━━━━┿━━━━━┿━━━━━┿━━━━━━━━━━━╋━━━━━┥
     └────    │     ┃           ╎     │     ╎           ┃     │
            2 ┤ n^2 ┃     n     ╎     │     ╎     n     ┃ n^2 │
              │     ┃           ╎     │     ╎           ┃     │
     ┌────  1 ┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┴╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┤
     │        │     ┃           ╎           ╎           ┃     │
 2c ─┤ r_j  0 ┼─────╂───────────┤    r0     ├───────────╂─────┤
     │        │     ┃           ╎           ╎           ┃     │
     └──── -1 ┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┬╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┤
              │     ┃           ╎     │     ╎           ┃     │
           -2 ┤ n^2 ┃     n     ╎     │     ╎     n     ┃ n^2 │
     ┌────    │     ┃           ╎     │     ╎           ┃     │
  d ─┤     -3 ┾━━━━━╋━━━━━━━━━━━┿━━━━━┿━━━━━┿━━━━━━━━━━━╋━━━━━┥
     └────    │  1  ┃    n^2    ┊     │     ╎    n^2    ┃  1  │
           -4 ┼─────╀─────┬─────┼─────┼─────┼─────┬─────╀─────┤
             -4    -3    -2    -1     0     1     2     3     4
                                     r_i
                 │     │        │           │        │     │
                 └──┬──┘        └─────┬─────┘        └──┬──┘
                    d                2c                 d
```
"""
@concrete struct BasicGerberIQ <: GerberIQCovarianceAlgorithm
    d
    n
    function BasicGerberIQ(d::Number, n::Number)
        assert_nonempty_gt0_finite_val(d, :d)
        @argcheck(zero(n) <= n <= one(n))
        return new{typeof(d), typeof(n)}(d, n)
    end
end
function BasicGerberIQ(; d::Number = 2.0, n::Number = 0.5)
    return BasicGerberIQ(d, n)
end
function gerber_iq_assert_c_d(c::Number, kind::BasicGerberIQ)
    @argcheck(c < kind.d)
    return nothing
end
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sc::Number,
                          kind::BasicGerberIQ)
    (; d, n) = kind
    d *= sc
    return if axi < d && axj < d
        n
    elseif axi >= d && axj >= d
        one(n)
    else
        n^2
    end
end
"""
```
                         ddn                     dcp
                       ┌──┴──┐                 ┌──┴──┐
                       │     │                 │     │
            4 ┬───────────┰─────┬─────┬─────┬─────┰───────────┐
     ┌────    │    n6     ┃ n9  ╎     │     ╎     ┃           │
ddp ─┤      3 ┾━━━━━━━━━━━╋━━━━━┿━━━━━┥     ╎ n7  ┃    n4     │
     └────    │           ┃     ╎     │     ╎     ┃           │ ────┐
            2 ┤    n10    ┃ n3  ╎     ┝━━━━━┿━━━━━╋━━━━━━━━━━━┥     ├─ dcp
              │           ┃     ╎     │     ╎ n1  ┃    n7     │ ────┘
     ┌────  1 ┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┼╌╌╌╌╌┴╌╌╌╌╌┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┤
     │        │           ┃     ╎           ╎     ┃           │
 2c ─┤ r_j  0 ┼─────┰─────┸─────┤    r0     ├─────┸─────┰─────┤
     │        │     ┃           ╎           ╎           ┃     │
     └──── -1 ┼╌╌╌╌╌╂╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┬╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╂╌╌╌╌╌┤
              │     ┃           ╎     │     ╎    n3     ┃ n9  │ ────┐
           -2 ┤ n8  ┃    n2     ╎     ┝━━━━━┿━━━━━━━━━━━╋━━━━━┥     ├─ ddn
     ┌────    │     ┃           ╎     │     ╎           ┃     │ ────┘
dcn ─┤     -3 ┾━━━━━╋━━━━━━━━━━━┿━━━━━┥     ╎    n10    ┃ n6  │
     └────    │ n5  ┃    n8     ╎     │     ╎           ┃     │
           -4 ┼─────╀─────┬─────┼─────┼─────┼─────┬─────╀─────┤
             -4    -3    -2    -1     0     1     2     3     4
                                     r_i
                 │     │        │           │        │     │
                 └──┬──┘        └─────┬─────┘        └──┬──┘
                   dcn               2c                ddp
```
"""
@concrete struct PartialGerberIQ <: GerberIQCovarianceAlgorithm
    dcp
    dcn
    ddp
    ddn
    n1
    n2
    n3
    n4
    n5
    n6
    n7
    n8
    n9
    n10
    function PartialGerberIQ(dcp::Number, dcn::Number, ddp::Number, ddn::Number, n1::Number,
                             n2::Number, n3::Number, n4::Number, n5::Number, n6::Number,
                             n7::Number, n8::Number, n9::Number, n10::Number)
        assert_nonempty_gt0_finite_val(dcp, :dcp)
        assert_nonempty_gt0_finite_val(dcn, :dcn)
        assert_nonempty_gt0_finite_val(ddp, :ddp)
        assert_nonempty_gt0_finite_val(ddn, :ddn)
        @argcheck(zero(n1) <= n1 <= one(n1))
        @argcheck(zero(n2) <= n2 <= one(n2))
        @argcheck(zero(n3) <= n3 <= one(n3))
        @argcheck(zero(n4) <= n4 <= one(n4))
        @argcheck(zero(n5) <= n5 <= one(n5))
        @argcheck(zero(n6) <= n6 <= one(n6))
        @argcheck(zero(n7) <= n7 <= one(n7))
        @argcheck(zero(n8) <= n8 <= one(n8))
        @argcheck(zero(n9) <= n9 <= one(n9))
        @argcheck(zero(n10) <= n10 <= one(n10))
        return new{typeof(dcp), typeof(dcn), typeof(ddp), typeof(ddn), typeof(n1),
                   typeof(n2), typeof(n3), typeof(n4), typeof(n5), typeof(n6), typeof(n7),
                   typeof(n8), typeof(n9), typeof(n10)}(dcp, dcn, ddp, ddn, n1, n2, n3, n4,
                                                        n5, n6, n7, n8, n9, n10)
    end
end
function PartialGerberIQ(; dcp::Number = 2.0, dcn::Number = 2.0, ddp::Number = 2.0,
                         ddn::Number = 2.0, n1::Number = 0.5, n2::Number = 0.5,
                         n3::Number = sqrt(n1 * n2), n4::Number = 1.0, n5::Number = 1.0,
                         n6::Number = 1.0, n7::Number = sqrt(n1 * n4),
                         n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n3 * n6),
                         n10::Number = sqrt(n3 * n6))
    return PartialGerberIQ(dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10)
end
function clamp_gerber_iq_n(alg::PartialGerberIQ, ::Gerber2)
    (; n1, n2, n3, n4, n5, n7, n8) = alg
    n3 = min(n3, sqrt(n1 * n2))
    n7 = min(n7, sqrt(n1 * n4))
    n8 = min(n8, sqrt(n2 * n5))
    return PartialGerberIQ(; dcp = alg.dcp, dcn = alg.dcn, ddp = alg.ddp, ddn = alg.ddn,
                           n1 = n1, n2 = n2, n3 = n3, n4 = n4, n5 = n5, n6 = alg.n6,
                           n7 = n7, n8 = n8, n9 = alg.n9, n10 = alg.n10)
end
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sc::Number,
                          kind::PartialGerberIQ)
    (; dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10) = kind
    dcp *= sc
    dcn *= sc
    ddp *= sc
    ddn *= sc
    return if dcp <= xi && dcp <= xj
        n4
    elseif dcp <= xi && zero(xj) < xj < dcp || dcp <= xj && zero(xi) < xi < dcp
        n7
    elseif zero(xi) < xi < dcp && zero(xj) < xj < dcp
        n1
    elseif xi <= -dcn && xj <= -dcn
        n5
    elseif xi <= -dcn && -dcn < xj < zero(xj) || xj <= -dcn && -dcn < xi < zero(xi)
        n8
    elseif -dcn < xi < zero(xi) && -dcn < xj < zero(xj)
        n2
    elseif ddp <= xi && xj <= -ddn || ddp <= xj && xi <= -ddn
        n6
    elseif ddp <= xi && -ddn < xj < zero(xj) || ddp <= xj && -ddn < xi < zero(xj)
        n9
    elseif zero(xi) < xi < ddp && xj <= -ddn || zero(xj) < xj < ddp && xi <= -ddn
        n10
    elseif zero(xi) < xi < ddp && -ddn < xj < zero(xj) ||
           zero(xj) < xj < ddp && -ddn < xi < zero(xi)
        n3
    else
        zero(xi)
    end
end
"""
```
                         ddn                     dcp
                       ┌──┴──┐                 ┌──┴──┐
                       │     │                 │     │
            4 ┬─────┰─────┰─────┬─────┬─────┬─────┰─────┰─────┐
     ┌────    │ n13 ┃ n19 ┃ n18 ╎     │     ╎ n15 ┃ n14 ┃ n11 │
ddp ─┤      3 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥
     └────    │ n20 ┃ n6  ┃ n9  ╎     │     ╎ n7  ┃ n4  ┃ n14 │ ────┐
            2 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥     ├─ dcp
              │ n21 ┃ n10 ┃ n3  ╎     │     ╎ n1  ┃ n7  ┃ n15 │ ────┘
     ┌────  1 ┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┼╌╌╌╌╌┴╌╌╌╌╌┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┤
     │        │     ┃     ┃     ╎           ╎     ┃     ┃     │
 2c ─┤ r_j  0 ┼─────╂─────╂─────┤    r0     ├─────╂─────╂─────┤
     │        │     ┃     ┃     ╎           ╎     ┃     ┃     │
     └──── -1 ┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┼╌╌╌╌╌┬╌╌╌╌╌┼╌╌╌╌╌╂╌╌╌╌╌╂╌╌╌╌╌┤
              │ n16 ┃ n8  ┃ n2  ╎     │     ╎ n3  ┃ n9  ┃ n18 │ ────┐
           -2 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥     ├─ ddn
     ┌────    │ n17 ┃ n5  ┃ n8  ╎     │     ╎ n10 ┃ n6  ┃ n19 │ ────┘
dcn ─┤     -3 ┾━━━━━╋━━━━━╋━━━━━┿━━━━━┿━━━━━┿━━━━━╋━━━━━╋━━━━━┥
     └────    │ n12 ┃ n17 ┃ n16 ╎     │     ╎ n21 ┃ n20 ┃ n13 │
           -4 ┼─────╀─────╀─────┼─────┼─────┼─────╀─────╀─────┤
             -4    -3    -2    -1     0     1     2     3     4
                                     r_i
                 │     │        │           │        │     │
                 └──┬──┘        └─────┬─────┘        └──┬──┘
                   dcn               2c                ddp
```
"""
@concrete struct FullGerberIQ <: GerberIQCovarianceAlgorithm
    dcp
    dcn
    ddp
    ddn
    n1
    n2
    n3
    n4
    n5
    n6
    n7
    n8
    n9
    n10
    n11
    n12
    n13
    n14
    n15
    n16
    n17
    n18
    n19
    n20
    n21
    function FullGerberIQ(dcp::Number, dcn::Number, ddp::Number, ddn::Number, n1::Number,
                          n2::Number, n3::Number, n4::Number, n5::Number, n6::Number,
                          n7::Number, n8::Number, n9::Number, n10::Number, n11::Number,
                          n12::Number, n13::Number, n14::Number, n15::Number, n16::Number,
                          n17::Number, n18::Number, n19::Number, n20::Number, n21::Number)
        assert_nonempty_gt0_finite_val(dcp, :dcp)
        assert_nonempty_gt0_finite_val(dcn, :dcn)
        assert_nonempty_gt0_finite_val(ddp, :ddp)
        assert_nonempty_gt0_finite_val(ddn, :ddn)
        @argcheck(zero(n1) <= n1 <= one(n1))
        @argcheck(zero(n2) <= n2 <= one(n2))
        @argcheck(zero(n3) <= n3 <= one(n3))
        @argcheck(zero(n4) <= n4 <= one(n4))
        @argcheck(zero(n5) <= n5 <= one(n5))
        @argcheck(zero(n6) <= n6 <= one(n6))
        @argcheck(zero(n7) <= n7 <= one(n7))
        @argcheck(zero(n8) <= n8 <= one(n8))
        @argcheck(zero(n9) <= n9 <= one(n9))
        @argcheck(zero(n10) <= n10 <= one(n10))
        @argcheck(zero(n11) <= n11 <= one(n11))
        @argcheck(zero(n12) <= n12 <= one(n12))
        @argcheck(zero(n13) <= n13 <= one(n13))
        @argcheck(zero(n14) <= n14 <= one(n14))
        @argcheck(zero(n15) <= n15 <= one(n15))
        @argcheck(zero(n16) <= n16 <= one(n16))
        @argcheck(zero(n17) <= n17 <= one(n17))
        @argcheck(zero(n18) <= n18 <= one(n18))
        @argcheck(zero(n19) <= n19 <= one(n19))
        @argcheck(zero(n20) <= n20 <= one(n20))
        @argcheck(zero(n21) <= n21 <= one(n21))
        return new{typeof(dcp), typeof(dcn), typeof(ddp), typeof(ddn), typeof(n1),
                   typeof(n2), typeof(n3), typeof(n4), typeof(n5), typeof(n6), typeof(n7),
                   typeof(n8), typeof(n9), typeof(n10), typeof(n11), typeof(n12),
                   typeof(n13), typeof(n14), typeof(n15), typeof(n16), typeof(n17),
                   typeof(n18), typeof(n19), typeof(n20), typeof(n21)}(dcp, dcn, ddp, ddn,
                                                                       n1, n2, n3, n4, n5,
                                                                       n6, n7, n8, n9, n10,
                                                                       n11, n12, n13, n14,
                                                                       n15, n16, n17, n18,
                                                                       n19, n20, n21)
    end
end
function FullGerberIQ(; dcp::Number = 2.0, dcn::Number = 2.0, ddp::Number = 2.0,
                      ddn::Number = 2.0, n1::Number = 0.5, n2::Number = 0.5,
                      n3::Number = 0.5, n4::Number = 0.75, n5::Number = 0.75,
                      n6::Number = 0.75, n7::Number = sqrt(n1 * n4),
                      n8::Number = sqrt(n2 * n5), n9::Number = sqrt(n3 * n6),
                      n10::Number = sqrt(n3 * n6), n11::Number = 1.0, n12::Number = 1.0,
                      n13::Number = 1.0, n14::Number = sqrt(n4 * n11),
                      n15::Number = sqrt(n7 * n14), n17::Number = sqrt(n5 * n12),
                      n16::Number = sqrt(n8 * n17), n19::Number = sqrt(n13 * n6),
                      n18::Number = sqrt(n19 * n9), n20::Number = sqrt(n13 * n6),
                      n21::Number = sqrt(n10 * n20))
    return FullGerberIQ(dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11,
                        n12, n13, n14, n15, n16, n17, n18, n19, n20, n21)
end
function gerber_iq_assert_c_d(c::Number, kind::Union{<:PartialGerberIQ, <:FullGerberIQ})
    @argcheck(c < kind.dcp)
    @argcheck(c < kind.dcn)
    @argcheck(c < kind.ddp)
    @argcheck(c < kind.ddn)
    return nothing
end
function clamp_gerber_iq_n(alg::FullGerberIQ, ::Gerber2)
    (; n1, n2, n3, n4, n5, n7, n8, n11, n12, n14, n17) = alg
    n3 = min(n3, sqrt(n1 * n2))
    n7 = min(n7, sqrt(n1 * n4))
    n8 = min(n8, sqrt(n2 * n5))
    n14 = min(n14, sqrt(n4 * n11))
    n17 = min(n17, sqrt(n5 * n12))
    return FullGerberIQ(; dcp = alg.dcp, dcn = alg.dcn, ddp = alg.ddp, ddn = alg.ddn,
                        n1 = n1, n2 = n2, n3 = n3, n4 = n4, n5 = n5, n6 = alg.n6, n7 = n7,
                        n8 = n8, n9 = alg.n9, n10 = alg.n10, n11 = n11, n12 = n12,
                        n13 = alg.n13, n14 = n14, n15 = alg.n15, n16 = alg.n16, n17 = n17,
                        n18 = alg.n18, n19 = alg.n19, n20 = alg.n20, n21 = alg.n21)
end
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sc::Number,
                          kind::FullGerberIQ)
    (; dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21) = kind
    dp1 = max(dcp, ddp) * sc
    dp2 = min(dcp, ddp) * sc
    dn1 = min(dcn, ddn) * sc
    dn2 = max(dcn, ddn) * sc
    return if dp1 <= xi && dp1 <= xj
        n11
    elseif dp1 <= xi && dp2 <= xj < dp1 || dp1 <= xj && dp2 <= xi < dp1
        n14
    elseif dp1 <= xi && zero(xj) < xj < dp2 || dp1 <= xj && zero(xi) < xi < dp2
        n15
    elseif dp1 <= xi && -dn1 < xj < zero(xj) || dp1 <= xj && -dn1 < xi < zero(xi)
        n18
    elseif dp1 <= xi && -dn2 < xj <= -dn1 || dp1 <= xj && -dn2 < xi <= -dn1
        n19
    elseif dp1 <= xi && xj <= -dn2 || dp1 <= xj && xi <= -dn2
        n13
    elseif dp2 <= xi < dp1 && dp2 <= xj < dp1
        n4
    elseif dp2 <= xi < dp1 && zero(xj) < xj < dp2 || dp2 <= xj < dp1 && zero(xi) < xi < dp2
        n7
    elseif dp2 <= xi < dp1 && -dn1 < xj < zero(xj) ||
           dp2 <= xj < dp1 && -dn1 < xi < zero(xi)
        n9
    elseif dp2 <= xi < dp1 && -dn2 < xj <= -dn1 || dp2 <= xj < dp1 && -dn2 < xi <= -dn1
        n6
    elseif dp2 <= xi < dp1 && xj <= -dn2 || dp2 <= xj < dp1 && xi <= -dn2
        n20
    elseif zero(xi) < xi < dp2 && zero(xj) < xj < dp2
        n1
    elseif zero(xi) < xi < dp2 && -dn1 < xj < zero(xj) ||
           zero(xj) < xj < dp2 && -dn1 < xi < zero(xi)
        n3
    elseif zero(xi) < xi < dp2 && -dn2 < xj <= -dn1 ||
           zero(xj) < xj < dp2 && -dn2 < xi <= -dn1
        n10
    elseif zero(xi) < xi < dp2 && xj <= -dn2 || zero(xi) < xi < dp2 && xj <= -dn2
        n21
    elseif -dn1 < xi < zero(xi) && -dn1 < xj < zero(xj)
        n2
    elseif -dn1 < xi < zero(xi) && -dn2 < xj <= -dn1 ||
           -dn1 < xj < zero(xj) && -dn2 < xi <= -dn1
        n8
    elseif -dn1 < xi < zero(xi) && xj <= -dn2 || -dn1 < xj < zero(xj) && xi <= -dn2
        n16
    elseif -dn2 < xi <= -dn1 && -dn2 < xj <= -dn1
        n5
    elseif -dn2 < xi <= -dn1 && xj <= -dn2 || -dn2 < xj <= -dn1 && xj <= -dn2
        n17
    elseif xi <= -dn2 && xj <= -dn2
        n12
    else
        zero(xi)
    end
end
@concrete struct GerberIQCovariance <: BaseGerberIQCovariance
    ve
    me
    pdm
    c
    t
    e
    y
    sc
    kind
    alg
    ex
    function GerberIQCovariance(ve::StatsBase.CovarianceEstimator,
                                me::AbstractExpectedReturnsEstimator, pdm::Option{<:Posdef},
                                c::Number, t::Option{<:GerberIQTau}, e::Integer,
                                y::Option{<:GerberIQGamma}, sc::Option{<:GerberIQScaler},
                                kind::GerberIQCovarianceAlgorithm,
                                alg::GerberCovarianceAlgorithm,
                                ex::FLoops.Transducers.Executor)
        assert_nonempty_nonneg_finite_val(c, :c)
        gerber_iq_assert_c_d(c, kind)
        if isa(t, Integer)
            assert_nonempty_nonneg_finite_val(t, :t)
        end
        assert_nonempty_nonneg_finite_val(e, :e)
        if isa(y, Number)
            assert_nonempty_nonneg_finite_val(y, :y)
        end
        kind = clamp_gerber_iq_n(kind, alg)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(c), typeof(t), typeof(e),
                   typeof(y), typeof(sc), typeof(kind), typeof(alg), typeof(ex)}(ve, me,
                                                                                 pdm, c, t,
                                                                                 e, y, sc,
                                                                                 kind, alg,
                                                                                 ex)
    end
end
function GerberIQCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                            me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                            pdm::Option{<:Posdef} = Posdef(), c::Number = 0.5,
                            t::Option{<:GerberIQTau} = nothing, e::Integer = 0,
                            y::Option{<:GerberIQGamma} = nothing,
                            sc::Option{<:GerberIQScaler} = nothing,
                            kind::GerberIQCovarianceAlgorithm = BasicGerberIQ(),
                            alg::GerberCovarianceAlgorithm = Gerber1(),
                            ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx())
    return GerberIQCovariance(ve, me, pdm, c, t, e, y, sc, kind, alg, ex)
end
function factory(ce::GerberIQCovariance, w::ObsWeights)
    return GerberIQCovariance(; ve = factory(ce.ve, w), me = factory(ce.me, w),
                              pdm = ce.pdm, c = ce.c, t = ce.t, e = ce.e, y = ce.y,
                              sc = ce.sc, kind = ce.kind, alg = ce.alg, ex = ce.ex)
end
function gerber_iq_decay(T::Integer, t::Integer, k::Integer, e::Number, y::Number)
    m = T - (t + k)
    return exp(-y * max(0, m - e))
end
function gerber_IQ_delta(xi::Number, xj::Number, axi::Number, axj::Number, T::Integer,
                         t::Integer, k::Integer, e::Number, y::Number, sc::Number,
                         kind::GerberIQCovarianceAlgorithm)
    w = gerber_iq_weight(xi, xj, axi, axj, sc, kind)
    p = gerber_iq_decay(T, k, t, e, y)
    return w * p
end
function gerber_IQ(ce::GerberIQCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:Any, <:Gerber0}, X::MatNum, sd::VecNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, t, e, y, sc, kind, ex) = ce
    t = gerber_iq_tau(t, T)
    scaler = gerber_iq_scaling(sc)
    FLoops.@floop ex for j in axes(X, 2)
        sdj = sd[j]
        for i in 1:j
            sdi = sd[i]
            scij = scaler(sdi, sdj)
            cij = scij * c
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                axi = abs(xi)
                axj = abs(xj)
                if axi < cij && axj < cij
                    continue
                end
                if axi >= cij && axj >= cij && xi * xj > zero(xi)
                    pos += gerber_IQ_delta(xi, xj, axi, axj, T, t, k, e, y, scij, kind)
                elseif axi >= cij && axj >= cij && xi * xj < zero(xi)
                    neg += gerber_IQ_delta(xi, xj, axi, axj, T, t, k, e, y, scij, kind)
                end
                den = (pos + neg)
                rho[j, i] = rho[i, j] = if !iszero(den)
                    (pos - neg) / den
                else
                    zero(eltype(X))
                end
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function gerber_IQ(ce::GerberIQCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:Any, <:Gerber1}, X::MatNum, sd::VecNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, t, e, y, sc, kind, ex) = ce
    t = gerber_iq_tau(t, T)
    scaler = gerber_iq_scaling(sc)
    FLoops.@floop ex for j in axes(X, 2)
        sdj = sd[j]
        for i in 1:j
            sdi = sd[i]
            scij = scaler(sdi, sdj)
            cij = scij * c
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            nn = zero(eltype(X))
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                axi = abs(xi)
                axj = abs(xj)
                if axi < cij && axj < cij
                    continue
                end
                if axi >= cij && axj >= cij && xi * xj > zero(xi)
                    pos += gerber_IQ_delta(xi, xj, axi, axj, T, t, k, e, y, scij, kind)
                elseif axi >= cij && axj >= cij && xi * xj < zero(xi)
                    neg += gerber_IQ_delta(xi, xj, axi, axj, T, t, k, e, y, scij, kind)
                else
                    nn += gerber_IQ_delta(xi, xj, axi, axj, T, t, k, e, y, scij, kind)
                end
                den = (pos + neg + nn)
                rho[j, i] = rho[i, j] = if !iszero(den)
                    (pos - neg) / den
                else
                    zero(eltype(X))
                end
            end
        end
    end
    posdef!(ce.pdm, rho)
    return rho
end
function gerber_IQ(ce::GerberIQCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:Any, <:Gerber2}, X::MatNum, sd::VecNum)
    #! Needs to scale ce.kind
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, t, e, y, sc, kind, ex) = ce
    t = gerber_iq_tau(t, T)
    scaler = gerber_iq_scaling(sc)
    FLoops.@floop ex for j in axes(X, 2)
        sdj = sd[j]
        for i in 1:j
            sdi = sd[i]
            scij = scaler(sdi, sdj)
            cij = scij * c
            neg = zero(eltype(X))
            pos = zero(eltype(X))
            for k in 1:T
                xi = X[k, i]
                xj = X[k, j]
                axi = abs(xi)
                axj = abs(xj)
                if axi < cij && axj < cij
                    continue
                end
                if axi >= cij && axj >= cij && xi * xj > zero(xi)
                    pos += gerber_IQ_delta(xi, xj, axi, axj, T, t, k, e, y, scij, kind)
                elseif axi >= cij && axj >= cij && xi * xj < zero(xi)
                    neg += gerber_IQ_delta(xi, xj, axi, axj, T, t, k, e, y, scij, kind)
                end
                rho[j, i] = rho[i, j] = (pos - neg)
            end
        end
    end
    h = sqrt.(LinearAlgebra.diag(rho))
    rho .= LinearAlgebra.Symmetric(rho ⊘ (h * transpose(h)), :U)
    posdef!(ce.pdm, rho)
    return rho
end
function Statistics.cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    return gerber_IQ(ce, X, sd)
end
function Statistics.cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    X = demean_returns(X, ce.me; dims = 1, kwargs...)
    sigma = gerber_IQ(ce, X, sd)
    return StatsBase.cor2cov!(sigma, sd)
end

export BasicGerberIQ, PartialGerberIQ, FullGerberIQ, GerberIQCovariance
