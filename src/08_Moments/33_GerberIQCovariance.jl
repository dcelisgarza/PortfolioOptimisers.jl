# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
abstract type BaseGerberIQCovariance <: BaseGerberCovariance end
abstract type GerberIQCovarianceAlgorithm <: AbstractMomentAlgorithm end
abstract type GerberIQCovarianceCTransformAlg <: AbstractMomentAlgorithm end
const GerberIQCovarianceCTransform = Union{Function, <:GerberIQCovarianceCTransformAlg}
"""
```
            4 ┬─────┰───────────┬───────────┬───────────┰─────┐
     ┌────    │  1  ┃    n^2    ╎           ╎    n^2    ┃  1  │
  d ─┤      3 ┾━━━━━╋━━━━━━━━━━━┥           ┝━━━━━━━━━━━╋━━━━━┥
     └────    │     ┃           ╎           ╎           ┃     │
            2 ┤ n^2 ┃     n     ╎           ╎     n     ┃ n^2 │
              │     ┃           ╎           ╎           ┃     │
     ┌────  1 ┼╌╌╌╌╌┸╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┸╌╌╌╌╌┤
     │        │                 ╎           ╎                 │
 2c ─┤ r_j  0 ┤                 ╎    r0     ╎                 │
     │        │                 ╎           ╎                 │
     └──── -1 ┼╌╌╌╌╌┰╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┰╌╌╌╌╌┤
              │     ┃           ╎           ╎           ┃     │
           -2 ┤ n^2 ┃     n     ╎           ╎     n     ┃ n^2 │
     ┌────    │     ┃           ╎           ╎           ┃     │
  d ─┤     -3 ┾━━━━━╋━━━━━━━━━━━┥           ┝━━━━━━━━━━━╋━━━━━┥
     └────    │  1  ┃    n^2    ╎           ╎    n^2    ┃  1  │
           -4 ┼─────╀─────┬─────┼─────┬─────┼─────┬─────╀─────┤
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
"""
```
                         ddn                     dcp
                       ┌──┴──┐                 ┌──┴──┐
                       │     │                 │     │
            4 ┬───────────┰─────┬───────────┬─────┰───────────┐
     ┌────    │    n6     ┃ n9  ╎           ╎     ┃           │
ddp ─┤      3 ┾━━━━━━━━━━━╋━━━━━┥           ╎ n7  ┃    n4     │
     └────    │           ┃     ╎           ╎     ┃           │ ────┐
            2 ┤    n10    ┃ n3  ╎           ┝━━━━━╋━━━━━━━━━━━┥     ├─ dcp
              │           ┃     ╎           ╎ n1  ┃    n7     │ ────┘
     ┌────  1 ┼╌╌╌╌╌╌╌╌╌╌╌┸╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┸╌╌╌╌╌╌╌╌╌╌╌┤
     │        │                 ╎           ╎                 │
 2c ─┤ r_j  0 ┤                 ╎    r0     ╎                 │
     │        │                 ╎           ╎                 │
     └──── -1 ┼╌╌╌╌╌┰╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┰╌╌╌╌╌┤
              │     ┃           ╎           ╎    n3     ┃ n9  │ ────┐
           -2 ┤ n8  ┃    n2     ╎           ┝━━━━━━━━━━━╋━━━━━┥     ├─ ddn
     ┌────    │     ┃           ╎           ╎           ┃     │ ────┘
dcn ─┤     -3 ┾━━━━━╋━━━━━━━━━━━┥           ╎    n10    ┃ n6  │
     └────    │ n5  ┃    n8     ╎           ╎           ┃     │
           -4 ┼─────╀─────┬─────┼─────┬─────┼─────┬─────╀─────┤
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
"""
```
                         ddn                     dcp
                       ┌──┴──┐                 ┌──┴──┐
                       │     │                 │     │
            4 ┬─────┰─────┰─────┬───────────┬─────┰─────┰─────┐
     ┌────    │ n13 ┃ n19 ┃ n18 ╎           ╎ n15 ┃ n14 ┃ n11 │
ddp ─┤      3 ┾━━━━━╋━━━━━╋━━━━━┥           ┝━━━━━╋━━━━━╋━━━━━┥
     └────    │ n20 ┃ n6  ┃ n9  ╎           ╎ n7  ┃ n4  ┃ n14 │ ────┐
            2 ┾━━━━━╋━━━━━╋━━━━━┥           ┝━━━━━╋━━━━━╋━━━━━┥     ├─ dcp
              │ n21 ┃ n10 ┃ n3  ╎           ╎ n1  ┃ n7  ┃ n15 │ ────┘
     ┌────  1 ┼╌╌╌╌╌┸╌╌╌╌╌┸╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┸╌╌╌╌╌┸╌╌╌╌╌┤
     │        │                 ╎           ╎                 │
 2c ─┤ r_j  0 ┤                 ╎    r0     ╎                 │
     │        │                 ╎           ╎                 │
     └──── -1 ┼╌╌╌╌╌┰╌╌╌╌╌┰╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┰╌╌╌╌╌┰╌╌╌╌╌┤
              │ n16 ┃ n8  ┃ n2  ╎           ╎ n3  ┃ n9  ┃ n18 │ ────┐
           -2 ┾━━━━━╋━━━━━╋━━━━━┥           ┝━━━━━╋━━━━━╋━━━━━┥     ├─ ddn
     ┌────    │ n17 ┃ n5  ┃ n8  ╎           ╎ n10 ┃ n6  ┃ n19 │ ────┘
dcn ─┤     -3 ┾━━━━━╋━━━━━╋━━━━━┥           ┝━━━━━╋━━━━━╋━━━━━┥
     └────    │ n12 ┃ n17 ┃ n16 ╎           ╎ n21 ┃ n20 ┃ n13 │
           -4 ┼─────╀─────╀─────┼─────┬─────┼─────╀─────╀─────┤
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
@concrete struct GerberIQCovariance <: BaseGerberIQCovariance
    r0
    c
    t
    e
    y
    alg
    function GerberIQCovariance(r0::Number, c::Number, t::Number, e::Number, y::Number,
                                alg::GerberIQCovarianceAlgorithm)
        assert_nonempty_nonneg_finite_val(r0, :r0)
        assert_nonempty_nonneg_finite_val(c, :c)
        assert_nonempty_nonneg_finite_val(t, :t)
        assert_nonempty_nonneg_finite_val(e, :e)
        assert_nonempty_nonneg_finite_val(y, :y)
        return new{typeof(r0), typeof(c), typeof(t), typeof(e), typeof(y), typeof(alg)}(r0,
                                                                                        c,
                                                                                        t,
                                                                                        e,
                                                                                        y,
                                                                                        alg)
    end
end
function get_M(c::Number, t::Number, X::MatNum)
    window = get_window(t, X)
    return -c .<= view(X, window, :) .<= c
end
#! Assume X is already transformed
function gerber_IQ(c::Number, t::Number, e::Number, y::Number,
                   alg::GerberIQCovarianceAlgorithm, std_vec::VecNum, X::MatNum,
                   f::Union{Function, <:GerberIQCovarianceCTransformAlg} = (x, y, z) -> x)
    window = get_window(t, X)
    X = view(X, window, :)
    T, N = size(X)
    Mj = falses(T)
    Mi = falses(T)
    Eu = falses(T)
    Ei = falses(T)
    rho = similar(X)
    for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            xi = view(X, :, i)
            # Transform c with std_vec if needed
            ct = f(c, std_vec[i], std_vec[j])
            Mj .= -ct .<= xj .<= ct
            Mi .= -ct .<= xi .<= ct
            if i != j
                Eu .= .!(Mj .| Mi)
                Ei .= .!(Mj .& Mi)
                rho[i, j] = rho[j, i] = val
            else
                Mi .= Mj
                Eu .= Ei .= .!Mj
                rho[i, j] = val
            end
        end
    end
    return nothing
end
