# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
abstract type BaseGerberIQCovariance <: BaseGerberCovariance end
abstract type GerberIQCovarianceAlgorithm <: AbstractMomentAlgorithm end
function clamp_gerber_iq_n(kind::GerberIQCovarianceAlgorithm, args...)
    return kind
end
abstract type GerberIQTauEpsEstimator <: AbstractEstimator end
const GerberIQTauEps = Union{<:Integer, Function, <:GerberIQTauEpsEstimator}
function gerber_iq_tau_eps(t::Integer, ::MatNum)
    return t
end
function gerber_iq_tau_eps(t::Function, X::MatNum)
    return t(X)::Integer
end
function gerber_iq_tau_eps(::Option{<:GerberIQTauEpsEstimator}, X::MatNum)
    T, N = size(X)
    return round(Int, T - T / N)
end
abstract type GerberIQGammaEstimator <: AbstractEstimator end
const GerberIQGamma = Union{<:Number, Function, <:GerberIQGammaEstimator}
function gerber_iq_gamma(y::Real, ::MatNum)
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
struct AssetVolatilityGerberIQScaler <: GerberIQScalerEstimator end
function gerber_iq_scaling(::AssetVolatilityGerberIQScaler, sdi::Number, sdj::Number)
    return sdi, sdj
end
function gerber_iq_scaling(::Option{<:GerberIQScalerEstimator}, sdi::Number, sdj::Number)
    res = (sdi + sdj) / 2
    return res, res
end
function gerber_iq_scaling(sca::Function, sdi::Number, sdj::Number)
    return sca(sdi, sdj)
end
abstract type GerberIQDecayEstimator <: AbstractEstimator end
const GerberIQDecay = Union{Function, <:GerberIQDecayEstimator}
function gerber_iq_decay(::Option{<:GerberIQDecay}, T::Integer, t::Integer, k::Integer,
                         e::Integer, y::Number)
    m = T - (t + k)
    return exp(-y * max(0, m - e))
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
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sci::Number,
                          scj::Number, kind::BasicGerberIQ)
    (; d, n) = kind
    di = d * sci
    dj = d * scj
    return if di <= axi && dj <= axj
        one(n)
    elseif axi < di && axj < dj
        n
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
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sci::Number,
                          scj::Number, kind::PartialGerberIQ)
    (; dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10) = kind
    dcpi = dcp * sci
    dcni = dcn * sci
    ddpi = ddp * sci
    ddni = ddn * sci
    dcpj = dcp * scj
    dcnj = dcn * scj
    ddpj = ddp * scj
    ddnj = ddn * scj
    zro = zero(xi)
    return if dcpi <= xi && dcpj <= xj
        n4
    elseif dcpi <= xi && zro < xj < dcpj || dcpj <= xj && zro < xi < dcpi
        n7
    elseif zro < xi < dcpi && zro < xj < dcpj
        n1
    elseif xi <= -dcni && xj <= -dcnj
        n5
    elseif xi <= -dcni && -dcnj < xj < zro || xj <= -dcnj && -dcni < xi < zro
        n8
    elseif -dcni < xi < zro && -dcnj < xj < zro
        n2
    elseif ddpi <= xi && xj <= -ddnj || ddpj <= xj && xi <= -ddni
        n6
    elseif ddpi <= xi && -ddnj < xj < zro || ddpj <= xj && -ddni < xi < zro
        n9
    elseif zro < xi < ddpi && xj <= -ddnj || zro < xj < ddpj && xi <= -ddni
        n10
    elseif zro < xi < ddpi && -ddnj < xj < zro || zro < xj < ddpj && -ddni < xi < zro
        n3
    else
        zro
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
                      n16::Number = sqrt(n8 * n17), n19::Number = sqrt(n6 * n13),
                      n18::Number = sqrt(n9 * n19), n20::Number = sqrt(n6 * n13),
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
function gerber_iq_weight(xi::Number, xj::Number, axi::Number, axj::Number, sci::Number,
                          scj::Number, kind::FullGerberIQ)
    (; dcp, dcn, ddp, ddn, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21) = kind
    _dp1 = max(dcp, ddp)
    _dp2 = min(dcp, ddp)
    _dn1 = min(dcn, ddn)
    _dn2 = max(dcn, ddn)
    dp1i = _dp1 * sci
    dp2i = _dp2 * sci
    dn1i = _dn1 * sci
    dn2i = _dn2 * sci
    dp1j = _dp1 * scj
    dp2j = _dp2 * scj
    dn1j = _dn1 * scj
    dn2j = _dn2 * scj
    zro = zero(xi)
    return if dp1i <= xi && dp1j <= xj
        n11
    elseif dp1i <= xi && dp2j <= xj < dp1j || dp1j <= xj && dp2i <= xi < dp1i
        n14
    elseif dp1i <= xi && zro < xj < dp2j || dp1j <= xj && zro < xi < dp2i
        n15
    elseif dp1i <= xi && -dn1j < xj < zro || dp1j <= xj && -dn1i < xi < zro
        n18
    elseif dp1i <= xi && -dn2j < xj <= -dn1j || dp1j <= xj && -dn2i < xi <= -dn1i
        n19
    elseif dp1i <= xi && xj <= -dn2j || dp1j <= xj && xi <= -dn2i
        n13
    elseif dp2i <= xi < dp1i && dp2j <= xj < dp1j
        n4
    elseif dp2i <= xi < dp1i && zro < xj < dp2j || dp2j <= xj < dp1j && zro < xi < dp2i
        n7
    elseif dp2i <= xi < dp1i && -dn1j < xj < zro || dp2j <= xj < dp1j && -dn1i < xi < zro
        n9
    elseif dp2i <= xi < dp1i && -dn2j < xj <= -dn1j ||
           dp2j <= xj < dp1j && -dn2i < xi <= -dn1i
        n6
    elseif dp2i <= xi < dp1i && xj <= -dn2j || dp2j <= xj < dp1j && xi <= -dn2i
        n20
    elseif zro < xi < dp2i && zro < xj < dp2j
        n1
    elseif zro < xi < dp2i && -dn1j < xj < zro || zro < xj < dp2j && -dn1i < xi < zro
        n3
    elseif zro < xi < dp2i && -dn2j < xj <= -dn1j || zro < xj < dp2j && -dn2i < xi <= -dn1i
        n10
    elseif zro < xi < dp2i && xj <= -dn2j || zro < xi < dp2i && xj <= -dn2j
        n21
    elseif -dn1i < xi < zro && -dn1j < xj < zro
        n2
    elseif -dn1i < xi < zro && -dn2j < xj <= -dn1j ||
           -dn1j < xj < zro && -dn2i < xi <= -dn1i
        n8
    elseif -dn1i < xi < zro && xj <= -dn2j || -dn1j < xj < zro && xi <= -dn2i
        n16
    elseif -dn2i < xi <= -dn1i && -dn2j < xj <= -dn1j
        n5
    elseif -dn2i < xi <= -dn1i && xj <= -dn2j || -dn2j < xj <= -dn1j && xj <= -dn2j
        n17
    elseif xi <= -dn2i && xj <= -dn2j
        n12
    else
        zro
    end
end
@concrete struct GerberIQCovariance <: BaseGerberIQCovariance
    ve
    me
    pdm
    c
    decay
    t
    e
    y
    sc
    kind
    alg
    ex
    function GerberIQCovariance(ve::StatsBase.CovarianceEstimator,
                                me::AbstractExpectedReturnsEstimator, pdm::Option{<:Posdef},
                                c::Number, decay::Option{<:GerberIQDecay},
                                t::Option{<:GerberIQTauEps}, e::Option{<:GerberIQTauEps},
                                y::Option{<:GerberIQGamma}, sc::Option{<:GerberIQScaler},
                                kind::GerberIQCovarianceAlgorithm,
                                alg::GerberCovarianceAlgorithm,
                                ex::FLoops.Transducers.Executor)
        assert_nonempty_nonneg_finite_val(c, :c)
        gerber_iq_assert_c_d(c, kind)
        if isa(t, Integer)
            assert_nonempty_nonneg_finite_val(t, :t)
        end
        if isa(e, Integer)
            assert_nonempty_nonneg_finite_val(e, :e)
        end
        if isa(y, Number)
            assert_nonempty_nonneg_finite_val(y, :y)
        end
        kind = clamp_gerber_iq_n(kind, alg)
        return new{typeof(ve), typeof(me), typeof(pdm), typeof(c), typeof(decay), typeof(t),
                   typeof(e), typeof(y), typeof(sc), typeof(kind), typeof(alg), typeof(ex)}(ve,
                                                                                            me,
                                                                                            pdm,
                                                                                            c,
                                                                                            decay,
                                                                                            t,
                                                                                            e,
                                                                                            y,
                                                                                            sc,
                                                                                            kind,
                                                                                            alg,
                                                                                            ex)
    end
end
function GerberIQCovariance(; ve::StatsBase.CovarianceEstimator = SimpleVariance(),
                            me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                            pdm::Option{<:Posdef} = Posdef(), c::Number = 0.5,
                            decay::Option{<:GerberIQDecay} = nothing,
                            t::Option{<:GerberIQTauEps} = nothing,
                            e::Option{<:GerberIQTauEps} = nothing,
                            y::Option{<:GerberIQGamma} = nothing,
                            sc::Option{<:GerberIQScaler} = nothing,
                            kind::GerberIQCovarianceAlgorithm = BasicGerberIQ(),
                            alg::GerberCovarianceAlgorithm = Gerber1(),
                            ex::FLoops.Transducers.Executor = FLoops.Transducers.ThreadedEx())
    return GerberIQCovariance(ve, me, pdm, c, decay, t, e, y, sc, kind, alg, ex)
end
function factory(ce::GerberIQCovariance, w::ObsWeights)
    return GerberIQCovariance(; ve = factory(ce.ve, w), me = factory(ce.me, w),
                              pdm = ce.pdm, c = ce.c, decay = ce.decay, t = ce.t, e = ce.e,
                              y = ce.y, sc = ce.sc, kind = ce.kind, alg = ce.alg,
                              ex = ce.ex)
end
function gerber_IQ_delta(xi::Number, xj::Number, axi::Number, axj::Number,
                         decay::Option{<:GerberIQDecay}, T::Integer, t::Integer, k::Integer,
                         e::Number, y::Number, sci::Number, scj::Number,
                         kind::GerberIQCovarianceAlgorithm)
    w = gerber_iq_weight(xi, xj, axi, axj, sci, scj, kind)
    p = gerber_iq_decay(decay, T, t, k, e, y)
    return w * p
end
function gerber_IQ(ce::GerberIQCovariance{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:Any, <:Any, <:Any, <:Gerber0}, X::MatNum,
                   sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, decay, t, e, y, sc, kind, ex) = ce
    t = gerber_iq_tau_eps(t, X)
    e = gerber_iq_tau_eps(e, X)
    y = gerber_iq_gamma(y, X)
    let t = t, y = y
        FLoops.@floop ex for j in axes(X, 2)
            sdj = sd[j]
            for i in 1:j
                sdi = sd[i]
                sci, scj = gerber_iq_scaling(sc, sdi, sdj)
                ci, cj = sci * c, scj * c
                neg = zero(eltype(X))
                pos = zero(eltype(X))
                for k in 1:T
                    xi = X[k, i]
                    xj = X[k, j]
                    axi = abs(xi)
                    axj = abs(xj)
                    if axi < ci && axj < cj
                        continue
                    end
                    if axi >= ci && axj >= cj && xi * xj > zero(xi)
                        pos += gerber_IQ_delta(xi, xj, axi, axj, decay, T, t, k, e, y, sci,
                                               scj, kind)
                    elseif axi >= ci && axj >= cj && xi * xj < zero(xi)
                        neg += gerber_IQ_delta(xi, xj, axi, axj, decay, T, t, k, e, y, sci,
                                               scj, kind)
                    end
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
                                          <:Any, <:Any, <:Any, <:Gerber1}, X::MatNum,
                   sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, decay, t, e, y, sc, kind, ex) = ce
    t = gerber_iq_tau_eps(t, X)
    e = gerber_iq_tau_eps(e, X)
    y = gerber_iq_gamma(y, X)
    let t = t, y = y
        FLoops.@floop ex for j in axes(X, 2)
            sdj = sd[j]
            for i in 1:j
                sdi = sd[i]
                sci, scj = gerber_iq_scaling(sc, sdi, sdj)
                ci, cj = sci * c, scj * c
                neg = zero(eltype(X))
                pos = zero(eltype(X))
                nn = zero(eltype(X))
                for k in 1:T
                    xi = X[k, i]
                    xj = X[k, j]
                    axi = abs(xi)
                    axj = abs(xj)
                    if axi < ci && axj < cj
                        continue
                    end
                    if axi >= ci && axj >= cj && xi * xj > zero(xi)
                        pos += gerber_IQ_delta(xi, xj, axi, axj, decay, T, t, k, e, y, sci,
                                               scj, kind)
                    elseif axi >= ci && axj >= cj && xi * xj < zero(xi)
                        neg += gerber_IQ_delta(xi, xj, axi, axj, decay, T, t, k, e, y, sci,
                                               scj, kind)
                    else
                        nn += gerber_IQ_delta(xi, xj, axi, axj, decay, T, t, k, e, y, sci,
                                              scj, kind)
                    end
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
                                          <:Any, <:Any, <:Any, <:Gerber2}, X::MatNum,
                   sd::ArrNum)
    T, N = size(X)
    rho = Matrix{eltype(X)}(undef, N, N)
    (; c, decay, t, e, y, sc, kind, ex) = ce
    t = gerber_iq_tau_eps(t, X)
    e = gerber_iq_tau_eps(e, X)
    y = gerber_iq_gamma(y, X)
    let t = t, y = y, e = e
        FLoops.@floop ex for j in axes(X, 2)
            sdj = sd[j]
            for i in 1:j
                sdi = sd[i]
                sci, scj = gerber_iq_scaling(sc, sdi, sdj)
                ci, cj = sci * c, scj * c
                neg = zero(eltype(X))
                pos = zero(eltype(X))
                for k in 1:T
                    xi = X[k, i]
                    xj = X[k, j]
                    axi = abs(xi)
                    axj = abs(xj)
                    if axi < ci && axj < cj
                        continue
                    end
                    if axi >= ci && axj >= cj && xi * xj > zero(xi)
                        pos += gerber_IQ_delta(xi, xj, axi, axj, decay, T, t, k, e, y, sci,
                                               scj, kind)
                    elseif axi >= ci && axj >= cj && xi * xj < zero(xi)
                        neg += gerber_IQ_delta(xi, xj, axi, axj, decay, T, t, k, e, y, sci,
                                               scj, kind)
                    end
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

export AssetVolatilityGerberIQScaler, BasicGerberIQ, PartialGerberIQ, FullGerberIQ,
       GerberIQCovariance
