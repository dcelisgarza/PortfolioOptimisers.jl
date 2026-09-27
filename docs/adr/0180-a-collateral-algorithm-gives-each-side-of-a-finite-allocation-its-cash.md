---
status: accepted
---

# A collateral algorithm gives each side of a finite allocation its cash

## Context

A finite allocation solves the short side of the book first and the long side second. The
function `adjust_long_cash(bgt, lcash, scash)` gave the long side its cash after the short side
reported `scash`, its leftover cash after shares and fee. It had two branches, and each branch
used a different model of short cash (#1337):

- With a budget `bgt >= 1`, it returned `lcash - scash`. A short sale was a source of cash, and
  cash the short side did not raise was cash the long side did not get.
- With `bgt < 1`, it returned `lcash + scash`. The short side's cash was collateral, and the long
  side took the part the short side did not use.

The second branch let the long side spend more than the cash. With `cash = 100`, prices
`[1, 1000]` and weights `[1.2, -0.5]`, the short side sold nothing and the long side bought 170
shares. A market-neutral book `[0.5, -0.5]` became all long at twice its long target. The rule also
jumped by `2 * scash` at `bgt = 1`, so a change of `0.0001` in one weight moved the long side from
119 shares to 100.

The first branch had a smaller defect. `scash` is the leftover after the short fee, so
`lcash - scash = C b + m_S + F_S`, where `m_S` is the short money and `F_S` the short fee. A short
fee is money that leaves the account, so the long side must get `C b + m_S - F_S`. The rule gave
the long side `2 F_S` too much whenever the short side paid a fee.

The issue offered two consistent models: proceeds everywhere, or collateral capped at the cash.
The choice is a property of the account and its broker, not of the allocator, so the maintainer
asked for both, as members of one family of functors, with the collateral amount as a field of
the second.

## Decision

1. **An abstract `AbstractCollateralAlgorithm <: AbstractAlgorithm`, unexported.** A concrete
   subtype is a functor with two methods. `ca(w, prices, cash)` returns the cash of the short
   side before it trades. `ca(w, prices, cash, smoney, sfee)` returns the cash of the long side,
   from the money of the shares that the short side sold and the fee that it paid.
   `adjust_long_cash` is deleted, and `setup_alloc_optim(w)` no longer takes the cash: it returns
   the two side budgets and the two masks.
2. **`ProceedsCollateral`, the default.** The short cash is `C b_S`. The long cash is
   `max(0, C b + m_S - F_S)`. The net money of the book, long less short with both fees, is at
   most `C b` for every book, and the rule is continuous in the weights. A book with `b <= 0`
   whose short side sells nothing buys nothing long.
3. **`CashCollateral(; amount = nothing)`.** A short sale gives no cash to spend, and it ties up
   collateral equal to its money. With `K = something(amount, cash)`, the short cash is
   `min(C b_S, K)` and the long cash is `max(0, min(C (b_L + b_S), K) - m_S - F_S)`. The long
   money, the short money and both fees are together at most `K`. The long side takes the
   collateral that the short side did not use, which is the issue's capped option with `K = C`.
4. **The algorithm is the field `ca` of `FiniteAllocationInput`, not of an allocator.**
   `DiscreteAllocation` falls back to `GreedyAllocation` by default. A field on each allocator
   lets the head and its fallback count short cash two ways in one chain. On the input, every
   allocator of the chain reads the same rule, as it reads the same cash.
5. **A third model, collateral derived from the holdings, is its own build ticket, #1340.** A broker can
   rate each product on a scale and lend a percentage of the money held in it, by rating. That
   model needs a Universe Sets instance to name the rating of each asset, and its design has
   open questions: whether the collateral is read from the target weights or from the realised
   long shares, which fixes the order of the two sub-problems, and whether a short position is
   rated too. The two methods above admit it, because they take `w` and `prices`.

## Consequences

- The default moves the result of a long-short book with `b < 1` whose short side leaves cash,
  in both allocators. Such a book no longer spends more than the cash. A long-only book, and a
  book with `b >= 1` whose short side pays no fee, keep their results.
- A book whose short side pays a fee gets `2 F_S` less long cash than before, under the default.
- A caller who wants the old behaviour below a unit budget passes `ca = CashCollateral()`, which
  keeps it where it was sound and caps it at the cash where it was not.
- `FiniteAllocationInput` gains an eighth field and an eighth type parameter.

## Rejected

- **One model only.** Both are real accounts. A cash account whose broker holds the proceeds of a
  short sale is the collateral model, and a margin account that credits the proceeds is the
  proceeds model.
- **A fourth argument `C` to `adjust_long_cash`.** It fixes the collateral at the cash, and a
  function of four numbers has no place to hold a collateral amount or a rating table.
- **The algorithm as a field of each allocator.** See decision 4.
