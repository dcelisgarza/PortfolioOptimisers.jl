# =============================================================================
# Prototype 18 — Tax-aware rebalancing: lots, wash sales, and lot selection.
#
# Purpose
#   Reports 2 and 3 both flag this, and report 2 is right that **no open-source
#   portfolio library does it well**. It is the clearest differentiation
#   available, and it is also the least glamorous.
#
#   The core observation is that a position is not a number. It is a **stack of
#   lots**, each with its own cost basis and acquisition date, and selling
#   "ten per cent of Apple" is a choice of which lots to sell. That choice can
#   change the tax bill by more than a year of alpha, and it is invisible to
#   every optimiser that models a position as a scalar weight.
#
#   Three things are needed and all three are implemented here:
#
#     1. Lot accounting, with the four standard selection rules.
#     2. The short-term against long-term split, which is where the money is.
#     3. The wash-sale rule, which silently voids a harvested loss.
#
# Status
#   Standalone. Depends on `Dates` and `Statistics`.
#
#   **This models the mechanics of lot accounting. It is not tax advice, and
#   the rules encoded are a simplified reading of the United States federal
#   regime.** Other jurisdictions differ, sometimes fundamentally: several
#   mandate a single averaged basis and have no lot choice at all.
#
# Notation used throughout this file
#   lot       One purchase: a share count, a cost basis per share, a date.
#   basis     Cost per share.
#   proceeds  Shares sold times the sale price.
#   gain      Proceeds minus basis, per lot, summed.
#   st, lt    Short term and long term. The boundary is a holding period.
#
# Sources
#   Constantinides, G. M. (1983). Capital market equilibrium with personal tax.
#     Econometrica 51(3), 611-636. The value of the timing option in
#     realisation.
#   Dammon, R. M., Spatt, C. S. and Zhang, H. H. (2001). Optimal consumption
#     and investment with capital gains taxes. Review of Financial Studies
#     14(3), 583-616.
#   Stein, D. M. and Narasimhan, P. (1999). Of passive and active equity
#     portfolios in the presence of taxes. Journal of Wealth Management 2(2),
#     55-63. The practitioner framing of tax-aware indexing.
#   Berkin, A. L. and Ye, J. (2003). Tax management, loss harvesting, and HIFO
#     accounting. Financial Analysts Journal 59(4), 91-102. The empirical value
#     of highest-in-first-out accounting.
#   Chaudhuri, S. E., Burnham, T. C. and Lo, A. W. (2020). An empirical
#     evaluation of tax-loss-harvesting alpha. Financial Analysts Journal
#     76(3), 99-108.
#   Internal Revenue Service. Publication 550, Investment Income and Expenses.
#     The wash-sale rule and the holding-period boundary.
# =============================================================================
module TaxAware

using Dates, Statistics

export TaxLot, LotSelectionMethod, FIFO, LIFO, HIFO, LOFO, TaxRates, Position, select_lots,
       realise_sale, wash_sale_flags, harvest_candidates, after_tax_proceeds

"""
    TaxLot{T}

One purchase of one asset.

# Fields

  - `shares::T`: Share count remaining in the lot. Always positive.
  - `basis::T`: Cost per share.
  - `acquired::Date`: Acquisition date, which sets the holding period.
"""
struct TaxLot{T <: Real}
    shares::T
    basis::T
    acquired::Date
    function TaxLot(shares::T, basis::T, acquired::Date) where {T <: Real}
        if shares <= 0
            throw(DomainError(shares, "a lot must hold a positive share count"))
        end
        return new{T}(shares, basis, acquired)
    end
end
function TaxLot(; shares::Real, basis::Real, acquired::Date)
    return TaxLot(promote(float(shares), float(basis))..., acquired)
end

"""
    LotSelectionMethod

Which lots to sell first.

  - `FIFO`: oldest first. The statutory default in the United States when no
    election is made.
  - `LIFO`: newest first.
  - `HIFO`: highest cost basis first. **Minimises the realised gain** for a
    given share count.
  - `LOFO`: lowest cost basis first. Maximises the realised gain, which is
    occasionally what a caller wants, to use an expiring capital loss.

# Notes

  - **`HIFO` minimises the gain, not necessarily the tax.** A long-term gain is
    taxed more lightly than a short-term one, so selling a high-basis
    short-term lot can cost more than selling a lower-basis long-term one.
    [`select_lots`](@ref) with `HIFO` optimises the wrong objective whenever
    the two rates differ, and `:tax_optimal` is the rule that does not.
"""
@enum LotSelectionMethod FIFO LIFO HIFO LOFO

"""
    TaxRates

The rates applied to realised gains.

# Fields

  - `short_term::Float64`: Rate on gains from lots held at most
    `long_term_days`.
  - `long_term::Float64`: Rate on the rest.
  - `long_term_days::Int`: Holding-period boundary. `366` for the United
    States, where the requirement is *more than* one year.

# Notes

  - **The boundary is strict.** A lot held exactly 365 days is short term in
    the United States. Off-by-one here is a real and expensive bug, so the
    comparison in [`realise_sale`](@ref) is `>=` against `long_term_days` with
    the default set to 366.
"""
Base.@kwdef struct TaxRates
    short_term::Float64 = 0.37
    long_term::Float64 = 0.20
    long_term_days::Int = 366
end

"""
    Position{T}

All open lots of a single asset.

# Fields

  - `symbol::String`: Asset label.
  - `lots::Vector{TaxLot{T}}`: The open lots, in any order.
"""
struct Position{T <: Real}
    symbol::String
    lots::Vector{TaxLot{T}}
end
Position(symbol::AbstractString, lots::Vector{<:TaxLot}) = Position(String(symbol), lots)

"""
    total_shares(p::Position) -> Real

Return the sum of the share counts of every open lot.
"""
total_shares(p::Position) = isempty(p.lots) ? 0.0 : sum(l.shares for l in p.lots)

"""
    select_lots(p::Position, shares::Real, method::Union{LotSelectionMethod, Symbol};
                price::Real = 0.0, today::Date = Dates.today(),
                rates::TaxRates = TaxRates()) -> Vector{Tuple{Int, Real}}

Choose which lots to sell, and how much of each.

# Arguments

  - `p`: The position.
  - `shares`: Share count to sell. Must not exceed the position.
  - `method`: A [`LotSelectionMethod`](@ref), or `:tax_optimal`.
  - `price`, `today`, `rates`: Required only by `:tax_optimal`.

# Returns

  - A vector of `(lot index, shares taken)` pairs, in the order the lots are
    consumed. The share counts sum to `shares`.

# The `:tax_optimal` rule

Rank the lots by the **tax paid per share sold**,

    tax per share  =  rate(lot) * ( price - basis )

and take the cheapest first. Because the tax of a sale is a sum over lots of a
per-share amount that does not depend on which other lots are chosen, the
greedy order is **exactly optimal**, not merely a heuristic. That is the
matroid structure of the problem: minimising a linear objective over the
polytope of "sell `shares` in total" is solved by sorting.

A lot at a loss has a negative tax per share, so losses are harvested first,
which is the correct behaviour.

# Notes

  - **`:tax_optimal` dominates `HIFO` whenever the two rates differ**, and
    equals it when they are the same. The verification driver asserts both.
  - The greedy argument fails as soon as anything couples the lots, such as an
    annual loss-deduction cap or a wash-sale interaction. Then the problem
    becomes an integer programme. This implementation solves the uncoupled
    case, which is the one that occurs at a single rebalance.
"""
function select_lots(p::Position, shares::Real, method::Union{LotSelectionMethod, Symbol};
                     price::Real = 0.0, today::Date = Dates.today(),
                     rates::TaxRates = TaxRates())
    avail = total_shares(p)
    if shares < 0
        throw(DomainError(shares, "shares must be >= 0"))
    end
    if shares > avail + 1e-10
        throw(ArgumentError("cannot sell $(shares) shares of $(p.symbol); only $(avail) are held"))
    end
    n = length(p.lots)
    order = if method === FIFO
        sortperm([l.acquired for l in p.lots])
    elseif method === LIFO
        sortperm([l.acquired for l in p.lots]; rev = true)
    elseif method === HIFO
        sortperm([l.basis for l in p.lots]; rev = true)
    elseif method === LOFO
        sortperm([l.basis for l in p.lots])
    elseif method === :tax_optimal
        sortperm([_tax_per_share(p.lots[i], price, today, rates) for i in 1:n])
    else
        throw(ArgumentError("unknown method $(method)"))
    end
    out = Tuple{Int, Float64}[]
    remaining = float(shares)
    for i in order
        if remaining <= 1e-12
            break
        end
        take = min(p.lots[i].shares, remaining)
        push!(out, (i, take))
        remaining -= take
    end
    return out
end

"""
    _tax_per_share(lot::TaxLot, price::Real, today::Date, rates::TaxRates) -> Float64

Return the tax owed per share if this lot is sold at `price` on `today`.
"""
function _tax_per_share(lot::TaxLot, price::Real, today::Date, rates::TaxRates)
    held = Dates.value(today - lot.acquired)
    rate = held >= rates.long_term_days ? rates.long_term : rates.short_term
    return rate * (price - lot.basis)
end

"""
    realise_sale(p::Position, shares::Real, price::Real, today::Date;
                 method = HIFO, rates::TaxRates = TaxRates())
        -> NamedTuple

Sell `shares` and report the cash and tax consequences.

# Arguments

  - `p`: The position.
  - `shares`: Share count to sell.
  - `price`: Sale price per share.
  - `today`: Trade date.
  - `method`: Lot selection rule.
  - `rates`: Tax rates.

# Returns

A `NamedTuple`:

  - `proceeds`: `shares * price`.
  - `short_gain`, `long_gain`: Realised gains, split by holding period. Either
    may be negative.
  - `tax`: `short_gain * rate_st + long_gain * rate_lt`. Negative means the
    sale generates a deduction.
  - `net`: `proceeds - tax`.
  - `remaining`: A new [`Position`](@ref) with the sold lots reduced or
    removed. **The input is not mutated.**
  - `detail`: Per-lot records.

# Mathematical definition

    gain    =  sum over selected lots of  taken_k * ( price - basis_k )
    tax     =  rate_st * short_gain  +  rate_lt * long_gain

# Notes

  - **A negative tax is not a refund.** Realised losses offset gains, and in
    the United States only three thousand dollars of net loss offsets ordinary
    income in a year, with the excess carried forward. That cap is an
    annual-level constraint and is deliberately not modelled here, because it
    couples every trade in the year and would make a per-trade function lie.
    Handle it at the portfolio level.
"""
function realise_sale(p::Position, shares::Real, price::Real, today::Date;
                      method::Union{LotSelectionMethod, Symbol} = HIFO,
                      rates::TaxRates = TaxRates())
    picks = select_lots(p, shares, method; price = price, today = today, rates = rates)
    short_gain = 0.0
    long_gain = 0.0
    detail = NamedTuple[]
    newlots = collect(p.lots)
    for (i, take) in picks
        lot = p.lots[i]
        held = Dates.value(today - lot.acquired)
        is_long = held >= rates.long_term_days
        g = take * (price - lot.basis)
        is_long ? (long_gain += g) : (short_gain += g)
        push!(detail,
              (; lot = i, shares = take, basis = lot.basis, acquired = lot.acquired,
               days_held = held, long_term = is_long, gain = g))
    end
    # Rebuild the remaining lots.
    taken = Dict(i => t for (i, t) in picks)
    kept = TaxLot{Float64}[]
    for (i, lot) in enumerate(newlots)
        t = get(taken, i, 0.0)
        left = lot.shares - t
        left > 1e-12 && push!(kept, TaxLot(left, lot.basis, lot.acquired))
    end
    tax = short_gain * rates.short_term + long_gain * rates.long_term
    proceeds = shares * price
    return (; proceeds = proceeds, short_gain = short_gain, long_gain = long_gain,
            tax = tax, net = proceeds - tax, remaining = Position(p.symbol, kept),
            detail = detail)
end

"""
    wash_sale_flags(sales::AbstractVector{<:Tuple{Date, Real}},
                    purchases::AbstractVector{<:Tuple{Date, Real}};
                    window_days::Integer = 30) -> Vector{<:NamedTuple}

Flag realised losses that the wash-sale rule disallows.

# Arguments

  - `sales`: `(date, realised gain)` pairs. A negative gain is a loss.
  - `purchases`: `(date, shares bought)` pairs for the **same or a
    substantially identical** security.
  - `window_days`: Half-width of the window. Thirty in the United States.

# Returns

One `NamedTuple` per loss-making sale, with `date`, `loss`, `washed` and
`triggering_purchases`.

# The rule

A loss is disallowed if substantially identical securities are bought within
`window_days` **before or after** the sale. The disallowed loss is added to the
basis of the replacement shares rather than lost, so it is deferred and not
destroyed.

# Notes

  - **The window is symmetric, and the "before" half is the one people
    forget.** Buying on the first of the month and selling at a loss on the
    fifteenth triggers the rule just as buying back on the twentieth does.
  - "Substantially identical" is a legal test, not an arithmetic one. Two
    index funds tracking the same benchmark from different providers are
    generally treated as distinct, which is the entire mechanism of
    tax-loss harvesting. **This function cannot decide that question**; the
    caller supplies the purchase list, and what belongs in it is a judgement.
"""
function wash_sale_flags(sales::AbstractVector{<:Tuple{Date, <:Real}},
                         purchases::AbstractVector{<:Tuple{Date, <:Real}};
                         window_days::Integer = 30)
    out = NamedTuple[]
    for (sdate, gain) in sales
        if gain >= 0
            continue
        end
        trig = [(pd, ps)
                for (pd, ps) in purchases
                if ps > 0 && abs(Dates.value(pd - sdate)) <= window_days]
        push!(out,
              (; date = sdate, loss = gain, washed = !isempty(trig),
               triggering_purchases = trig))
    end
    return out
end

"""
    harvest_candidates(positions::AbstractVector{<:Position},
                       prices::AbstractDict{String, <:Real}, today::Date;
                       rates::TaxRates = TaxRates(), min_loss::Real = 0.0)
        -> Vector{<:NamedTuple}

Rank the lots whose sale would realise a loss, best first.

# Arguments

  - `positions`: The open positions.
  - `prices`: Current price per symbol.
  - `today`: Valuation date.
  - `rates`: Tax rates.
  - `min_loss`: Ignore candidates whose tax benefit is below this.

# Returns

One `NamedTuple` per candidate lot, sorted by `tax_benefit` descending, with
`symbol`, `lot`, `shares`, `basis`, `price`, `loss`, `long_term` and
`tax_benefit`.

# Mathematical definition

    loss         =  shares * ( price - basis )        (negative)
    tax_benefit  =  -loss * rate(lot)

# Notes

  - **Short-term losses are worth more than long-term ones**, because they
    offset income taxed at the higher rate. So the ranking prefers recent
    purchases that have fallen, which is the opposite of the intuition that
    older lots are the ones to clean up.
  - The output is a candidate list, not a trade list. Selling every candidate
    changes the portfolio's risk, and the wash-sale rule constrains the
    replacement. Feed this to the optimiser as a preference, not as an order.
"""
function harvest_candidates(positions::AbstractVector{<:Position},
                            prices::AbstractDict{String, <:Real}, today::Date;
                            rates::TaxRates = TaxRates(), min_loss::Real = 0.0)
    out = NamedTuple[]
    for p in positions
        if !(haskey(prices, p.symbol))
            continue
        end
        px = prices[p.symbol]
        for (i, lot) in enumerate(p.lots)
            loss = lot.shares * (px - lot.basis)
            if loss >= 0
                continue
            end
            held = Dates.value(today - lot.acquired)
            is_long = held >= rates.long_term_days
            benefit = -loss * (is_long ? rates.long_term : rates.short_term)
            if benefit < min_loss
                continue
            end
            push!(out,
                  (; symbol = p.symbol, lot = i, shares = lot.shares, basis = lot.basis,
                   price = px, loss = loss, long_term = is_long, tax_benefit = benefit))
        end
    end
    return sort(out; by = c -> c.tax_benefit, rev = true)
end

"""
    after_tax_proceeds(p::Position, shares::Real, price::Real, today::Date;
                       rates::TaxRates = TaxRates()) -> NamedTuple

Compare every lot-selection rule on the same sale.

# Returns

A `NamedTuple` mapping each rule name to its `(tax, net)` pair, plus `best`.

# Notes

  - This exists to make the size of the choice visible. **A caller who has
    never compared the rules on their own book usually does not believe the
    gap.**
"""
function after_tax_proceeds(p::Position, shares::Real, price::Real, today::Date;
                            rates::TaxRates = TaxRates())
    methods = [("FIFO", FIFO), ("LIFO", LIFO), ("HIFO", HIFO), ("LOFO", LOFO),
               ("tax_optimal", :tax_optimal)]
    res = map(methods) do (nm, m)
        r = realise_sale(p, shares, price, today; method = m, rates = rates)
        return (; method = nm, tax = r.tax, net = r.net, short_gain = r.short_gain,
                long_gain = r.long_gain)
    end
    best = res[argmin([r.tax for r in res])]
    return (; results = res, best = best)
end

end # module TaxAware
