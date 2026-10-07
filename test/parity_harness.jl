#=
The parity harness of map #1375 (#1376). Every parity ticket of the map reads its fixtures, its
exchange, its comparison and its storage from this file, so no ticket builds its own.

Not a test file (no `test_` prefix), so the runner does not discover it. A parity test includes
it with `include(joinpath(@__DIR__, "parity_harness.jl"))`. `test_12o_parity_harness.jl` covers it.

**The fixtures.** `parity_small_panel` (12 assets × 80 observations) and `parity_large_panel`
(40 × 250) are two draws of one generator, `parity_panel`. The stored cases of map #643 use fully
active panels, so each panel here holds the conditions they never see, at fixed assets:

  - asset 2 lists late, asset 3 delists, and asset 4 delists and lists again;
  - asset 5 misses one observation inside an active stretch, a holiday gap;
  - asset 6 is active but outside the estimation mask on the first quarter of the observations;
  - `book_equity`, the denominator of a return on equity, is zero on asset 7 at one observation
    and negative on asset 8 on ten;
  - `adj_volume` is negative on asset 9 at one observation, and `short_interest` on asset 10;
  - `style1` ties assets 9, 10 and 11 at one observation;
  - the industry level "Utilities" holds asset 3 alone, so it is empty on the active universe after
    the delisting, and on every sub-universe without asset 3;
  - the Exogenous Series holds three currencies, "EUR", "JPY" and "USD", and one macro series,
    "MACRO".

The returns follow a factor model: one industry factor, two lagged styles, the macro series with
a loading for each asset, and the currency of the asset, which the base-currency return adds.

`ccy_fixture` (40 × 120) is the fully active currency fixture of #1369, which the observed-factor
tests share, with `ccy_pass` for a passthrough factor of one of its Panel Fields.

**The exchange.** `parity_write` writes a `ReturnsResult` and its panel to a folder of CSV files,
one row a line, written with `join`. A numeric cell outside the active mask, or not observed, is
`NaN`, because the oracle refuses a finite value there. Under `filled = true` an active cell
that is not observed holds the value of the fill policy instead. A categorical cell is a zero-based code
over its levels, and `-1` outside the active mask or not observed. `parity_read` reads a CSV file
back, plain or gzipped. The scripts on the oracle side read the same folder; they live outside the
repository.

**The comparison.** `parity_compare` is the one comparison of every parity ticket. It needs the
same shape and the same pattern of non-finite values, and every finite pair within
`atol + rtol * max(|a|, |b|)`, `rtol = 1e-12` by default. It prints the largest relative and
absolute difference, so a test states the measured value next to its tolerance. A small entry
formed by a cancellation, such as an off-diagonal covariance of two unrelated assets, carries a
relative round-off far above that of the matrix: `scale = :array` measures each cell against the
largest entry instead, and a test that uses it says why.

**The storage.** A stored oracle output is `test/assets/Parity_<Unit>_<Case>_<Output>.csv.gz`, with
a header row `x1, x2, …`. `Unit` is the Julia name of the unit under test, `Case` names the
fixture and the configuration, and `Output` names the field of the result. The test of each stored
case states in one sentence how the case was made: the fixture, the configuration, and the output.
=#
using Statistics

# The default factor prior and idiosyncratic variance of `CrossSectionalFactorPrior`, with the
# oracle's raw regime statistic. The library's default divides the statistic by the bias of the
# estimate it reads (#1428, ADR 0190), and the oracle does not.
const PARITY_PE = EmpiricalPrior(; me = ExpWeightedExpectedReturns(),
                                 ce = RegimeAdjustedExpWeightedCovariance(;
                                                                          centring = PreCentred(),
                                                                          debias = RawStatistic()))
const PARITY_VE = RegimeAdjustedExpWeightedVariance(; centring = PreCentred(),
                                                    debias = RawStatistic())

"""
    parity_panel(; T, N, seed) -> NamedTuple

Draw the parity fixture of `T` observations and `N ≥ 12` assets. Return `rd`, the
`ReturnsResult` with its panel and its Exogenous Series, the two masks, and `at`, the cells
that hold each condition of the fixture.
"""
function parity_panel(; T::Integer = 80, N::Integer = 12, seed::Integer = 1376)
    @assert N >= 12 && T >= 40
    rng = StableRNG(seed)
    q = T ÷ 4
    h = T ÷ 2
    # The universe: a late listing, a delisting, and a delisting that lists again.
    amsk = trues(T, N)
    amsk[1:q, 2] .= false
    amsk[(3 * q + 1):T, 3] .= false
    amsk[(T ÷ 3 + 1):h, 4] .= false
    emsk = copy(amsk)
    emsk[1:q, 6] .= false
    at = (; late = (2, q + 1), delist = (3, 3 * q), relist = (4, T ÷ 3 + 1, h + 1),
          holiday = (h + 5, 5), outside_estimation = (1:q, 6), zero_denominator = (h, 7),
          negative_denominator = ((h + 1):(h + 10), 8), negative_volume = (h + 2, 9),
          negative_short_interest = (h + 3, 10), ties = (h - 3, [9, 10, 11]),
          empty_level = ("Utilities", 3))
    # The classifications. Asset 3 alone holds "Utilities".
    inds = ["Banks", "Energy", "Software"]
    ind = [i == 3 ? "Utilities" : inds[mod(i - 1, 3) + 1] for i in 1:N]
    ccys = ["EUR", "JPY", "USD"]
    code = [mod(i, 3) + 1 for i in 1:N]
    # The factors and the returns.
    z(A) = (A .- mean(A; dims = 2)) ./ std(A; dims = 2, corrected = false)
    s1 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    s2 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    s1[at.ties[1], at.ties[2]] .= s1[at.ties[1], at.ties[2][1]]
    fi = Dict(k => 0.01 .* randn(rng, T) for k in [inds; "Utilities"])
    fs = 0.004 .* randn(rng, T, 2)
    macro_ = 0.006 .* randn(rng, T)
    bm = randn(rng, N)
    R = 0.005 .* randn(rng, T, 3)
    loc = 0.01 .* randn(rng, T, N)
    for t in 2:T, i in 1:N
        loc[t, i] += fi[ind[i]][t] +
                     s1[t - 1, i] * fs[t, 1] +
                     s2[t - 1, i] * fs[t, 2] +
                     bm[i] * macro_[t]
    end
    X = loc .+ R[:, code]
    # The fields. The price moves with the base return of each active observation.
    close = exp.(log(40.0) .+ 0.3 .* randn(rng, 1, N) .+
                 cumsum(log1p.(ifelse.(amsk, X, 0.0)); dims = 1))
    shares = exp.(log(1e7) .+ 0.5 .* randn(rng, 1, N) .+
                  0.001 .* cumsum(randn(rng, T, N); dims = 1))
    mcap = close .* shares
    volume = shares .* exp.(log(0.008) .+ 0.4 .* randn(rng, T, N))
    short_interest = shares .* exp.(log(0.02) .+ 0.3 .* randn(rng, T, N))
    book = mcap .* exp.(log(0.4) .+ 0.2 .* randn(rng, 1, N))
    assets = mcap ./ exp.(log(0.5) .+ 0.2 .* randn(rng, 1, N))
    income = 0.05 .* book .+ 0.02 .* book .* randn(rng, T, N)
    sales = 0.6 .* assets .* exp.(0.1 .* randn(rng, T, N))
    book[at.zero_denominator...] = 0.0
    book[at.negative_denominator...] .= -abs.(book[at.negative_denominator...])
    volume[at.negative_volume...] = -volume[at.negative_volume...]
    short_interest[at.negative_short_interest...] = -short_interest[at.negative_short_interest...]
    # The holiday gap: no return, no price and no volume on one active observation.
    X[at.holiday...] = NaN
    close[at.holiday...] = NaN
    volume[at.holiday...] = NaN
    numeric = ["market_cap" => mcap, "adj_close" => close,
               "adj_shares_outstanding" => shares, "adj_volume" => volume,
               "short_interest" => short_interest, "book_equity" => book,
               "total_assets" => assets, "net_income_ttm" => income, "sales_ttm" => sales,
               "style1" => s1, "style2" => s2]
    for (_, A) in numeric
        A[.!amsk] .= NaN
    end
    X[.!amsk] .= NaN
    # Every numeric field blanks outside the active mask, so each one carries a fill policy.
    pf = [[NumericPanelInput(; name = k, vals = A, alg = ForwardPanelFill(; val = 0.0))
           for (k, A) in numeric];
          CategoricalPanelInput(; name = "industry", vals = repeat(permutedims(ind), T));
          CategoricalPanelInput(; name = "currency",
                                vals = repeat(permutedims(ccys[code]), T))]
    pnl = asset_panel(pf; amsk = amsk, emsk = emsk)
    rd = ReturnsResult(; nx = ["a$i" for i in 1:N], X = X, ne = [ccys; "MACRO"],
                       E = hcat(R, macro_), pnl = pnl)
    return (; rd, amsk, emsk, at, loc, R, code, ind)
end
parity_small_panel() = parity_panel(; T = 80, N = 12, seed = 1376)
parity_large_panel() = parity_panel(; T = 250, N = 40, seed = 1377)

"""
    ccy_fixture(; T = 120, N = 40, seed = 927_001) -> NamedTuple

Draw the currency fixture of #1369: a fully active panel of `N` assets over `T` observations,
three industries, two styles and three currencies. The local returns follow the industries and
the lagged styles, and the base-currency returns add the Currency Excess Return of the currency of
each asset. Return `rd`, the `ReturnsResult` with its panel and the currency returns as the
Exogenous Series, `R`, the currency returns, `codes`, the currency of each asset, and `ind`, its
industry.
"""
function ccy_fixture(; T = 120, N = 40, seed = 927_001)
    rng = StableRNG(seed)
    ind = [mod(i - 1, 3) + 1 for i in 1:N]
    codes = [mod(i, 3) + 1 for i in 1:N]
    I3 = zeros(T, N, 3)
    for i in 1:N
        I3[:, i, ind[i]] .= 1.0
    end
    z(A) = (A .- mean(A; dims = 2)) ./ std(A; dims = 2, corrected = false)
    s1 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    s2 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    mcap = exp.(randn(rng, 1, N) .* 0.8 .+ 0.05 .* cumsum(randn(rng, T, N); dims = 1))
    fi = 0.01 .* randn(rng, T, 3)
    fs = 0.004 .* randn(rng, T, 2)
    loc = zeros(T, N)
    for t in 2:T, i in 1:N
        loc[t, i] = fi[t, ind[i]] +
                    s1[t - 1, i] * fs[t, 1] +
                    s2[t - 1, i] * fs[t, 2] +
                    0.01 * randn(rng)
    end
    loc[1, :] .= 0.01 .* randn(rng, N)
    R = 0.005 .* randn(rng, T, 3)
    X = copy(loc)
    for t in 1:T, i in 1:N
        X[t, i] += R[t, codes[i]]
    end
    lv = ["EUR", "JPY", "USD"]
    pf = [NumericPanelInput(; name = "market_cap", vals = mcap),
          NumericPanelInput(; name = "style1", vals = s1),
          NumericPanelInput(; name = "style2", vals = s2),
          [NumericPanelInput(; name = "ind$k", vals = I3[:, :, k]) for k in 1:3]...,
          CategoricalPanelInput(; name = "currency",
                                vals = repeat(permutedims(lv[codes]), T))]
    pnl = asset_panel(pf; amsk = trues(T, N), emsk = trues(T, N))
    rd = ReturnsResult(; nx = ["a$i" for i in 0:(N - 1)], X = X, ne = lv, E = R, pnl = pnl)
    return (; rd, R, codes, ind)
end
"""
    ccy_pass(field, family) -> CompositeExposure

A factor that passes the Panel Field `field` through, unscored and untrimmed, in `family`.
"""
function ccy_pass(field, family)
    return CompositeExposure(; descriptors = [Passthrough(; field = field)],
                             outlier = nothing, scoring = nothing, family = family)
end

function parity_write_rows(path::AbstractString, A::AbstractMatrix)
    open(path, "w") do io
        for r in eachrow(A)
            println(io, join(r, ','))
        end
    end
    return path
end
function parity_write_rows(path::AbstractString, v::AbstractVector)
    return parity_write_rows(path, reshape(v, :, 1))
end

function parity_field_rows(f::NumericPanelField, keep::AbstractMatrix{Bool})
    return ifelse.(keep, f.vals, NaN)
end
function parity_field_rows(f::CategoricalPanelField, keep::AbstractMatrix{Bool})
    return ifelse.(keep, f.codes .- 1, -1)
end
parity_field_kind(::NumericPanelField) = "numeric"
parity_field_kind(::CategoricalPanelField) = "categorical"

"""
    parity_write(dir, rd::ReturnsResult; filled = false) -> dir

Write `rd` and its panel to CSV files in `dir`:

  - `returns.csv`: `X`, `NaN` outside the active mask;
  - `active_mask.csv`, `estimation_mask.csv`: `0` or `1`;
  - `assets.csv`: one asset name a line;
  - `fields.csv`: one line `<name>,numeric` or `<name>,categorical` for each Panel Field;
  - `field_<name>.csv`: the values, or the zero-based codes, of each Panel Field;
  - `levels_<name>.csv`: the levels of each categorical Panel Field, one a line;
  - `exogenous.csv`: the Exogenous Series, under a header row of its names.

A cell outside the active mask is `NaN`, or `-1` for a code. An active cell that is not observed
is too, unless `filled` is `true`: then it holds the value that the fill policy of the Panel Field
gave it, which is the value a Julia estimator reads.
"""
function parity_write(dir::AbstractString, rd::ReturnsResult; filled::Bool = false)
    mkpath(dir)
    pnl = rd.pnl
    amsk = pnl.amsk
    parity_write_rows(joinpath(dir, "returns.csv"), ifelse.(amsk, rd.X, NaN))
    parity_write_rows(joinpath(dir, "active_mask.csv"), Int.(amsk))
    parity_write_rows(joinpath(dir, "estimation_mask.csv"), Int.(pnl.emsk))
    parity_write_rows(joinpath(dir, "assets.csv"), rd.nx)
    open(joinpath(dir, "fields.csv"), "w") do io
        for f in pnl.pf
            keep = filled || isnothing(f.omsk) ? amsk : amsk .& f.omsk
            parity_write_rows(joinpath(dir, "field_$(f.name).csv"),
                              parity_field_rows(f, keep))
            println(io, f.name, ',', parity_field_kind(f))
            if f isa CategoricalPanelField
                parity_write_rows(joinpath(dir, "levels_$(f.name).csv"), f.levels)
            end
        end
    end
    if !isnothing(rd.E)
        open(joinpath(dir, "exogenous.csv"), "w") do io
            println(io, join(rd.ne, ','))
            for r in eachrow(rd.E)
                println(io, join(r, ','))
            end
        end
    end
    return dir
end

"""
    parity_read(path; header = false) -> Matrix{Float64}

Read a CSV file, plain or gzipped, into a matrix. The oracle side writes its outputs with no
header, and a stored file carries one.
"""
function parity_read(path::AbstractString; header::Bool = false)
    return Matrix{Float64}(CSV.read(path, DataFrame; header = header))
end

"""
    parity_compare(a, b; rtol = 1e-12, atol = 0.0, scale = :cell, name = "") -> NamedTuple

Compare `a` with `b` cell by cell. `ok` is true when the shapes agree, the non-finite cells agree
exactly, and every finite pair lies within `atol + rtol * m`. Under `scale = :cell`, `m` is
`max(|a|, |b|)` of the pair. Under `scale = :array`, `m` is the largest finite `|a|` or `|b|` of
the two arrays, for an output whose small entries come from a cancellation, such as an
off-diagonal covariance.

`pattern` reports the non-finite cells alone. `maxrel` is the largest difference of a finite pair
over its own size, `maxscaled` the largest difference over the largest entry, and `maxabs` the
largest difference. Print one line with all three, under `name`.
"""
function parity_compare(a::AbstractArray, b::AbstractArray; rtol::Real = 1e-12,
                        atol::Real = 0.0, scale::Symbol = :cell, name::AbstractString = "")
    @assert scale in (:cell, :array)
    if size(a) != size(b)
        println("parity $(name): size $(size(a)) against $(size(b))")
        return (; ok = false, pattern = false, maxrel = Inf, maxscaled = Inf, maxabs = Inf)
    end
    big = 0.0
    for x in Iterators.flatten((a, b))
        big = isfinite(x) ? max(big, abs(x)) : big
    end
    pattern = true
    within = true
    maxrel = 0.0
    maxabs = 0.0
    for (x, y) in zip(a, b)
        if isfinite(x) && isfinite(y)
            d = abs(x - y)
            m = max(abs(x), abs(y))
            maxabs = max(maxabs, d)
            maxrel = max(maxrel, iszero(d) ? zero(d) : d / m)
            within &= d <= atol + rtol * (scale == :cell ? m : big)
        else
            pattern &= isequal(x, y)
        end
    end
    maxscaled = iszero(maxabs) ? 0.0 : maxabs / big
    println("parity $(name): maxrel = $(maxrel), maxscaled = $(maxscaled), maxabs = $(maxabs), pattern = $(pattern)")
    return (; ok = pattern && within, pattern, maxrel, maxscaled, maxabs)
end

"""
    parity_asset(unit, case, output; dir) -> String

The path of the stored oracle output `Parity_<unit>_<case>_<output>.csv.gz` under `dir`,
`test/assets` by default.
"""
function parity_asset(unit::AbstractString, case::AbstractString, output::AbstractString;
                      dir::AbstractString = joinpath(@__DIR__, "assets"))
    return joinpath(dir, "Parity_$(unit)_$(case)_$(output).csv.gz")
end

"""
    parity_store(path, A) -> path

Store the oracle output `A`, a vector or a matrix, gzipped at `path` under a header row
`x1, x2, …`.
"""
function parity_store(path::AbstractString, A::AbstractVecOrMat)
    M = A isa AbstractVector ? reshape(A, :, 1) : A
    CSV.write(path, DataFrame(M, :auto); compress = true)
    return path
end

"""
    parity_load(unit, case, output; dir) -> Matrix{Float64}

Read the stored oracle output of [`parity_asset`](@ref).
"""
function parity_load(unit::AbstractString, case::AbstractString, output::AbstractString;
                     dir::AbstractString = joinpath(@__DIR__, "assets"))
    return parity_read(parity_asset(unit, case, output; dir = dir); header = true)
end
