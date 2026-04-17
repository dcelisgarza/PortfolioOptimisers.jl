# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
abstract type BaseGerberIQCovariance <: BaseGerberCovariance end

"""
┌─┬─┐
│ │ │
├─┼─┤
│ │ │
└─┴─┘

```
                                  ddn                                     dcp
                                ┌──┴──┐                                 ┌──┴──┐
                                │     │                                 │     │
              ┌──────────┰─────────┰─────────┬───────────────────┬─────────┰─────────┰─────────┐
            4 ┤    e13   ┃   e19   ┃   e18   ╎                   ╎   e15   ┃   e14   ┃   e11   │
     ┌────    │          ┃         ┃         ╎                   ╎         ┃         ┃         │
ddp ─┤      3 ┾━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥                   ┝━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥
     └────    │    e20   ┃   e6    ┃   e9    ╎                   ╎   e7    ┃   e4    ┃   e14   │  ────┐
            2 ┾━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥                   ┝━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥      ├─ dcp
              │    e21   ┃   e10   ┃   e3    ╎                   ╎   e1    ┃   e7    ┃   e15   │  ────┘
     ┌────  1 ┼╌╌╌╌╌╌╌╌╌╌┸╌╌╌╌╌╌╌╌╌┸╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┸╌╌╌╌╌╌╌╌╌┸╌╌╌╌╌╌╌╌╌┤
     │        │                              ╎                   ╎                             │
 2c ─┤ r_j  0 ┤                              ╎        r0         ╎                             │
     │        │                              ╎                   ╎                             │
     └──── -1 ┼╌╌╌╌╌╌╌╌╌╌┰╌╌╌╌╌╌╌╌╌┰╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┰╌╌╌╌╌╌╌╌╌┰╌╌╌╌╌╌╌╌╌┤
              │    e16   ┃   e8    ┃   e2    ╎                   ╎   e3    ┃   e9    ┃   e18   │  ────┐
           -2 ┾━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥                   ┝━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥      ├─ ddn
      ┌────   │    e17   ┃   e5    ┃   e8    ╎                   ╎   e10   ┃   e6    ┃   e19   │  ────┘
 dcn ─┤    -3 ┾━━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥                   ┝━━━━━━━━━╋━━━━━━━━━╋━━━━━━━━━┥
      └────   │          ┃         ┃         ╎                   ╎         ┃         ┃         │
           -4 ┤    e12   ┃   e17   ┃   e16   ╎                   ╎   e21   ┃   e20   ┃   e13   │
              └┬─────────╀─────────╀─────────┼─────────┬─────────┼─────────╀─────────╀─────────┤
              -4        -3        -2        -1         0         1         2         3         4
                                                      r_i
                       │   │                 │                   │                 │   │
                       └─┬─┘                 └─────────┬─────────┘                 └─┬─┘
                        dcn                            2c                           ddp

```
"""
@concrete struct FullGerberIQCovariance <: BaseGerberIQCovariance
    c
    r0
    dcp
    dcn
    ddp
    ddn
    e1
    e2
    e3
    e4
    e5
    e6
    e7
    e8
    e9
    e10
    e11
    e12
    e13
    e14
    e15
    e16
    e17
    e18
    e19
    e20
    e21
    # function GerberIQCovariance(c::Number, r0::Num_VecToScaM, delta::AbstractVector,
    #                             eta::AbstractVector)
    #     assert_nonempty_nonneg_finite_val(c, :c)
    #     @argcheck(isfinite(r0))
    #     assert_nonempty_nonneg_finite_val(delta, :delta)
    #     assert_nonempty_nonneg_finite_val(eta, :eta)
    #     return new{}
    # end
end
