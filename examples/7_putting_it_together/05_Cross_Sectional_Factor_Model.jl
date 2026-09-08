#=
# Cross-sectional factor model, end to end

Every other factor example on this site fits a factor model **through time**: it regresses each
asset's return series on a set of observed factor series, one regression per asset. This page
fits one **across the cross-section**. At each observation it regresses that day's returns of
every asset on the *lagged* traits of those assets — their size, their value, their industry —
and the coefficients it recovers are the factor returns of that day. Nobody supplies a factor
series; the fit produces one.

That change of direction is what a point-in-time **Asset Panel** buys. A panel is a stack of
**Panel Fields** — market capitalisation, book equity, an industry label — each indexed by
observation and asset, and it travels on the returns carrier as `rd.pnl`. The panel is where the
factor *exposures* come from, so the model can carry factors nobody publishes a series for.

The page runs the whole route on a synthetic panel drawn from a factor model we know, so every
claim below is checkable against an answer fixed before the estimator ran:

 1. Build the panel, and the traits it was drawn from.
 2. Fit a [`CrossSectionalFactorPrior`](@ref) on it.
 3. Check what the fit recovered, and the identities it makes exact.
 4. Hand the prior's own factor model to two **orthogonal uncertainty sets**, and watch the book
    leave the directions the factors do not span.
 5. Run the same optimiser through a [`WalkForward`](@ref) under a factor mandate.
 6. Read a predicted and a realised **factor attribution** off the answer.

!!! tip "When to reach for this"
    Reach for a cross-sectional factor model when your conviction lives in **asset traits** rather
    than in factor series — when you can say "cheap companies beat expensive ones" but have no
    published value-factor return to regress on. It is also the only route here that admits a
    universe whose membership *changes*: assets list and delist, and the panel says which pair is
    live at which observation. If you already hold factor series, [`FactorPrior`](@ref) is the
    simpler tool.
=#

using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, PrettyTables,
      DataFrames, Clarabel

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

#=
## 1. A synthetic Asset Panel, and the model it was drawn from

The generator below draws nine factors — a market factor, four industries and four styles — and
gives every asset a fixed trait vector: a market beta, a one-hot industry membership and four
style loadings. Returns are that trait vector through the factor returns, plus an idiosyncratic
shock and a small per-asset alpha.

The Panel Fields are then built as *noisy functions of the same traits*. Log market
capitalisation tracks the size trait, book-to-price tracks the value trait, and so on. That is
what makes the panel an acceptance test rather than a demonstration: a fit that recovers the
traits from the fields has recovered something we can name.

A fifth of the assets **list late**, so the panel's active mask is not all-`true` and the
universe is genuinely point-in-time.
=#

function synthetic_panel(; T = 500, N = 80, seed = 661_001)
    rng = StableRNG(seed)
    industries = ["Energy", "Financials", "Health Care", "Technology"]
    Kind = length(industries)
    ind = rand(rng, 1:Kind, N)
    ## The traits. `beta` is the market exposure, `Ltrue` the four style loadings.
    beta = 1.0 .+ 0.35 .* randn(rng, N)
    Ltrue = randn(rng, N, 4)
    alpha = 0.0004 .* randn(rng, N)
    ## The factor returns, and the idiosyncratic shocks.
    onehot = Float64[ind[i] == k for i in 1:N, k in 1:Kind]
    Btrue = hcat(beta, onehot, Ltrue)
    ftrue = hcat(0.009 .* randn(rng, T), 0.005 .* randn(rng, T, Kind),
                 0.004 .* randn(rng, T, 4))
    ivol = 0.008 .+ 0.012 .* rand(rng, N)
    X = ftrue * transpose(Btrue) + randn(rng, T, N) .* transpose(ivol) .+ transpose(alpha)
    ## The Panel Fields, each a noisy function of the traits above.
    logcap = 20.0 .+ 1.4 .* transpose(view(Ltrue, :, 1)) .+
             0.3 .* cumsum(randn(rng, T, N); dims = 1) ./ sqrt(T)
    mcap = exp.(logcap)
    shares = mcap ./ 50.0
    fields = ["market_cap" => mcap,
              "book_equity" =>
                  mcap .* exp.(-0.7 .+ 0.5 .* transpose(view(Ltrue, :, 2)) .+
                               0.05 .* randn(rng, T, N)),
              "net_income_ttm" =>
                  mcap .* (0.05 .+ 0.02 .* transpose(view(Ltrue, :, 3)) .+
                           0.004 .* randn(rng, T, N)), "adj_shares_outstanding" => shares,
              "adj_volume" =>
                  shares .* exp.(-4.5 .+ 0.6 .* transpose(view(Ltrue, :, 4)) .+
                                 0.15 .* randn(rng, T, N)),
              ## A field no Factor Exposure reads, so a forecast built on it carries a part
              ## the factors do not span.
              "signal" => transpose(alpha) .+ 0.0002 .* randn(rng, T, N)]
    ## A fifth of the assets list late. A cell before an asset lists is blank.
    listed = [i <= N ÷ 5 ? rand(rng, 20:120) : 1 for i in 1:N]
    amsk = [t >= listed[i] for t in 1:T, i in 1:N]
    for (_, a) in fields
        a[.!amsk] .= NaN
    end
    inputs = [[NumericPanelInput(; name = n, vals = a, alg = ForwardPanelFill(; val = 0.0))
               for (n, a) in fields]
              CategoricalPanelInput(; name = "industry",
                                    vals = repeat(reshape(industries[ind], 1, N), T, 1),
                                    levels = industries)]
    days = filter(d -> Dates.dayofweek(d) <= 5,
                  Date(2015, 1, 1):Day(1):(Date(2015, 1, 1) + Day(2 * T + 10)))[1:T]
    rd = ReturnsResult(; nx = ["A" * lpad(i, 3, '0') for i in 1:N],
                       X = ifelse.(amsk, X, NaN), ts = days,
                       pnl = asset_panel(inputs; amsk = amsk, emsk = amsk))
    nf = vcat("market", ["industry=" * l for l in industries],
              ["size", "value", "earnings_yield", "liquidity"])
    return (; rd = rd, B = Btrue, f = ftrue, ivar = ivol .^ 2, nf = nf)
end

syn = synthetic_panel()
rd = syn.rd
T, N = size(rd.X)

#=
The panel rides the carrier. `rd.X` is the returns, `rd.pnl` is the panel, and the active mask
says which pair is live.
=#

pretty_table(DataFrame("Assets" => N, "Observations" => T,
                       "Panel Fields" => length(rd.pnl.pf),
                       "Active cells" => count(rd.pnl.amsk) / (T * N));
             formatters = [numfmt], title = "The synthetic Asset Panel")

#=
## 2. The prior

A [`CrossSectionalFactorPrior`](@ref) is specified by its **Factor Exposures** — one per factor,
each naming the family it belongs to. Four kinds ship, and three of them appear here:

  - [`ConstantExposure`](@ref) is a column of ones, the market intercept. Here we use a
    [`CompositeExposure`](@ref) over [`EWMarketBeta`](@ref) instead, so the market exposure is the
    asset's own estimated beta rather than one.
  - [`OneHotExposure`](@ref) turns a categorical Panel Field into a block of indicator columns,
    one per level.
  - [`CompositeExposure`](@ref) scores one or more **Descriptors** across the cross-section. A
    Descriptor is the trait itself — [`LogMarketCap`](@ref), [`BookToPrice`](@ref) — and the
    exposure standardises it.

Three further pieces of the specification are worth naming, because each one changes the answer:

 1. `families = ["industry" => nothing]` puts the industry block under a **zero-sum re-basis**.
    An industry block sums to one for every asset, so it is collinear with a market factor whose
    exposure is a beta near one. The constraint is what identifies the members.
 2. `neutralise = ["style" => "industry"]` removes the benchmark-weighted overlap of the style
    family with the industry family, so a style factor return is not an industry bet in disguise.
 3. `wa = BlendedInverseVarianceWeights(...)` makes the regression a **two-pass** fit: the first
    pass estimates residual variances, the second re-weights by them.

Finally, `rfe` supplies an **alpha forecast**. It scores the `signal` field, which no exposure
reads, and `lambda`/`c` split the forecast into the part the factors span and the part they do
not. The second part lands in `rr.b`, and that is the whole reason the orthogonal sets of §4
have anything to bite on.
=#

style(d) = CompositeExposure(; descriptors = [d], family = "style")
factors = ["market" =>
               CompositeExposure(; descriptors = [EWMarketBeta()], outlier = nothing,
                                 scoring = nothing, family = "market"),
           "industry" => OneHotExposure(; field = "industry", family = "industry"),
           "size" => style(LogMarketCap()), "value" => style(BookToPrice()),
           "earnings_yield" => style(EarningsToPrice()),
           "liquidity" => style(EWShareTurnover())]

forecast = FixedWeightedReturnForecast(;
                                       scores = DescriptorScores(;
                                                                 descriptors = [Passthrough(;
                                                                                            field = "signal")],
                                                                 outlier = nothing,
                                                                 scoring = nothing),
                                       scale = 1.0)

pe = CrossSectionalFactorPrior(; factors = factors, families = ["industry" => nothing],
                               neutralise = ["style" => "industry"],
                               wa = BlendedInverseVarianceWeights(; lambda = 0.5),
                               rfe = forecast, lambda = 1.0, c = 1.0)

pr = prior(pe, rd)
rr = pr.rr

pretty_table(DataFrame("Factor" => rr.nf, "Family" => rr.fam);
             title = "The nine factors the fit produced, and their families")

#=
The result is an ordinary [`LowOrderPrior`](@ref) over the **full** asset universe, so every
consumer in the library takes it unchanged. What is new is the block on `rr`: a
[`CrossSectionalFactorModel`](@ref) carrying the exposure history `Ms`, the realised factor
returns, the idiosyncratic returns and variances, the regression and benchmark weights, and the
family basis.

The fit keeps the tail of the observation axis — the window left after the Descriptors' warm-up
and the exposure lag.
=#

pretty_table(DataFrame("Fit rows" => size(pr.X, 1), "Warm-up rows" => T - size(pr.X, 1),
                       "Factors" => size(rr.Ms, 3),
                       "Investable assets" => count(isfinite, pr.mu));
             formatters = [numfmt], title = "What the fit kept")

#=
## 3. What the fit recovered, and what it makes exact

Two different claims live here, and they need different tests.

The **recovery** is statistical. The Panel Fields are noisy functions of the traits, so a fitted
exposure correlates with the truth rather than equalling it. The one exception is the industry
block: a one-hot exposure of a known classification is the classification, so it is recovered
exactly.
=#

active = findall(view(rd.pnl.amsk, T, :))
loading_corr = [cor(view(rr.M, active, k), view(syn.B, active, k))
                for k in eachindex(rr.nf)]
industry_k = findall(isequal("industry"), rr.fam)

pretty_table(DataFrame("Factor" => rr.nf, "corr(fitted, true) loading" => loading_corr);
             formatters = [numfmt], title = "The fit recovers the traits it was drawn from")

pretty_table(DataFrame("One-hot industry loadings recovered exactly" =>
                           rr.M[active, industry_k] == syn.B[active, industry_k]))

#=
The **systematic return** of a pair is the quantity the model actually asserts about an asset. A
single factor return is identified only up to the basis the family constraint chose, but
`Ms[t - lag] · f_t` is basis-free, and it is comparable with the generator's own.
=#

## The systematic return of every eligible pair, fitted against the generator's own.
function systematic_pairs(pr, B, f)
    rr = pr.rr
    Tf = size(rr.csr.eps, 1)
    ftail = view(f, (size(f, 1) - Tf + 1):size(f, 1), :)
    fitted = Float64[]
    truth = Float64[]
    for t in 2:Tf, i in axes(rr.csr.eps, 2)
        u = dot(view(rr.Ms, t - 1, i, :), view(pr.fpr.X, t, :))
        v = dot(view(B, i, :), view(ftail, t, :))
        if isfinite(u) && isfinite(v)
            push!(fitted, u)
            push!(truth, v)
        end
    end
    return fitted, truth
end

sys_fit, sys_true = systematic_pairs(pr, syn.B, syn.f)

idio_ok = findall(isfinite, rr.esigma)
pretty_table(DataFrame("corr(systematic return)" => cor(sys_fit, sys_true),
                       "Pairs" => length(sys_fit),
                       "corr(idio variance)" => cor(rr.esigma[idio_ok], syn.ivar[idio_ok]),
                       "median ratio" => median(rr.esigma[idio_ok] ./ syn.ivar[idio_ok]));
             formatters = [numfmt],
             title = "The systematic return and the idiosyncratic level")

#=
The **identities**, by contrast, are constructions rather than estimates, and they hold at machine
precision. Three of them need only public fields:

 1. The lagged exposures through the factor returns, plus the idiosyncratic return, reproduce the
    asset's return exactly. This is the reconciliation the fit is built to satisfy.
 2. The covariance square root reproduces the covariance.
 3. `mu` is the loadings through the factor mean, plus the orthogonal part of the alpha forecast.
=#

## The reconciliation identity, over every pair whose arithmetic is finite.
function reconciliation(pr, X)
    rr = pr.rr
    Tf = size(rr.csr.eps, 1)
    Xtail = view(X, (size(X, 1) - Tf + 1):size(X, 1), :)
    worst = 0.0
    n = 0
    for t in 2:Tf, i in axes(rr.csr.eps, 2)
        v = dot(view(rr.Ms, t - 1, i, :), view(pr.fpr.X, t, :)) + rr.csr.eps[t, i]
        if isfinite(v) && isfinite(Xtail[t, i])
            worst = max(worst, abs(v - Xtail[t, i]))
            n += 1
        end
    end
    return worst, n
end

recon, recon_pairs = reconciliation(pr, rd.X)
inv_i = findall(isfinite, pr.mu)

pretty_table(DataFrame("max |Ms*f + eps - X|" => recon, "Pairs" => recon_pairs,
                       "max |chol'chol - S|" => maximum(abs,
                                                        transpose(pr.chol[:, inv_i]) * pr.chol[:, inv_i] -
                                                        pr.sigma[inv_i, inv_i]),
                       "max |mu - M*f - b|" => maximum(abs,
                                                       view(pr.mu, inv_i) - view(rr.M * pr.fpr.mu + rr.b, inv_i)));
             formatters = [numfmt], title = "Three identities the fit makes exact")

#=
The zero-sum re-basis is exact too. Under the constraint, the benchmark-weighted sum of the
industry family's factor returns is zero at every observation, relative to the size of its own
terms.
=#

## The benchmark-weighted sum of a family's factor returns, relative to the size of its
## own terms. The exposures are read `lag` observations back, because the fit's coefficients
## are coordinates in the basis of that observation.
function family_zero_sum(pr, family)
    rr = pr.rr
    worst = 0.0
    for u in axes(pr.fpr.X, 1)
        t = u - rr.lag
        if t < 1
            continue
        end
        total = 0.0
        magnitude = 0.0
        for k in family
            c = 0.0
            for a in axes(rr.bw, 2)
                e = rr.Ms[t, a, k]
                isfinite(e) && (c += rr.bw[t, a] * e)
            end
            total += c * pr.fpr.X[u, k]
            magnitude += abs(c * pr.fpr.X[u, k])
        end
        worst = max(worst, abs(total) / max(magnitude, eps()))
    end
    return worst
end

zero_sum = family_zero_sum(pr, industry_k)

pretty_table(DataFrame("max relative benchmark-weighted industry return" => zero_sum);
             formatters = [numfmt], title = "The zero-sum re-basis")

#=
## 4. Two orthogonal uncertainty sets

Here is the idea the rest of this page builds to. The factor model splits the asset space in two:
the directions the loadings **span**, where the model has something to say, and the
**Orthogonal Subspace**, where it has nothing. A portfolio that bets in the second half is
betting on estimation error.

[`OrthogonalUncertaintySet`](@ref) reads the factor model off the optimisation's **own** prior
result and confines the uncertainty to that second half. One estimator produces two sets:

  - a low-rank **norm ball** on the mean, whose radius is `sqrt(χ²_r)` at the rank `r` of the
    complement, and
  - a **compact covariance set**, `(κ, C, Q)`, whose worst-case variance is
    `w'Σw + κ·min_z ‖Cw − Qz‖²` — a quadratic term the variance consumer adds directly, with no
    lifted semidefinite block. `Q` is an orthonormal basis of the subspace the penalty **spares**,
    which is the weighted factor span.

Both are fitted from the prior we already have.
=#

mu_set = mu_ucs(OrthogonalUncertaintySet(), pr)
sigma_set = sigma_ucs(OrthogonalUncertaintySet(), pr)

pretty_table(DataFrame("Rank of the factor span" => size(sigma_set.Q, 2),
                       "Rank of the Orthogonal Subspace" => size(mu_set.L, 2),
                       "Mean-ball radius" => mu_set.kappa); formatters = [numfmt],
             title = "The geometry the two sets share")

#=
The natural reading of a book is then: how much of it lies **outside** the spared subspace? That
is `‖(I − QQᵀ)Cw‖ / ‖Cw‖`, and it is the number the two sweeps below move.
=#

universe = UniverseSets(; dict = Dict("nx" => rd.nx, "ncf" => rr.nf))

solver = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                check_sol = (; allow_local = true, allow_almost = true),
                settings = Dict("verbose" => false))

function orthogonal_share(w, set)
    v = set.C .* w
    return norm(v - set.Q * (transpose(set.Q) * v)) / norm(v)
end

function book(; radius = 0.0, kappa = 1.0, obj = MinimumRisk(), constraint = nothing,
              data = rd)
    u = OrthogonalUncertaintySet(; kappa = kappa, method = radius)
    return optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = u), obj = obj,
                             opt = JuMPOptimiser(; pe = pe, slv = solver, bgt = 1.0,
                                                 wb = WeightBounds(; lb = 0.0, ub = 0.1),
                                                 sets = universe, lcse = constraint,
                                                 ret = ArithmeticReturn(; ucs = u))), data)
end

#=
### The covariance radius moves the book smoothly

`κ` scales a **quadratic** penalty on the orthogonal component, so raising it squeezes the book
out of the Orthogonal Subspace gradually. The minimum-risk book below starts with 91% of its
metric-scaled weight outside the factor span, and ends with a tenth of a percent of it.
=#

kappa_grid = [0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0]
kappa_books = [book(; kappa = k) for k in kappa_grid]

pretty_table(DataFrame("kappa" => kappa_grid,
                       "Outside the span" =>
                           [orthogonal_share(r.w, sigma_set) for r in kappa_books],
                       "Volatility" =>
                           [sqrt(dot(r.w, pr.sigma * r.w)) for r in kappa_books],
                       "Expected return" => [dot(r.w, pr.mu) for r in kappa_books],
                       "Names" => [count(>(1e-6), r.w) for r in kappa_books]);
             formatters = [(v, i, j) -> if j == 2
                               "$(round(v * 100, digits = 2)) %"
                           elseif j == 3 || j == 4
                               "$(round(v * 10_000, digits = 2)) bp"
                           else
                               v
                           end], title = "The compact covariance radius")

#=
### Calibrating the covariance radius instead of stating it

The sweep above is the honest way to explore `κ`, and it is also an admission: the numbers in
`kappa_grid` were chosen by hand, and nothing in the data suggested them. `κ` can instead be
**sized from the sample**, by a rule of [`AbstractCompactRadiusAlgorithm`](@ref) placed in the
same field. The estimator resolves it inside the fit, where the metric, the loadings block and
the factor span are all in hand, and the set that comes out carries a plain number.

Two rules ship, and they answer two different questions.

  - [`ResidualInflation`](@ref) treats `κ` as a **confidence level**. The penalty lives exactly
    where the idiosyncratic variance lives, so the question is how far the *estimate* of that
    variance can sit from the truth, and a variance has a chi-squared bound. Under this model's
    default metric the answer is dimensionless — it is the relative inflation itself.
  - [`VarianceFraction`](@ref) treats `κ` as a **magnitude with a unit**. It sizes the penalty so
    that a reference portfolio pays a stated fraction of its nominal variance, which is a number a
    desk can argue about: *robustify by ten percent*.
=#

calibrated = ["Stated" => 100.0, "ResidualInflation()" => ResidualInflation(),
              "ResidualInflation(; q = 0.01)" => ResidualInflation(; q = 0.01),
              "VarianceFraction(; f = 0.1)" => VarianceFraction(; f = 0.1),
              "VarianceFraction(; f = 0.5)" => VarianceFraction(; f = 0.5)]
calibrated_sets = [sigma_ucs(OrthogonalUncertaintySet(; kappa = k), rd, pr)
                   for (_, k) in calibrated]
calibrated_books = [book(; kappa = k) for (_, k) in calibrated]

pretty_table(DataFrame("kappa" => first.(calibrated),
                       "Resolved" => [s.kappa for s in calibrated_sets],
                       "Outside the span" =>
                           [orthogonal_share(r.w, sigma_set) for r in calibrated_books],
                       "Volatility" =>
                           [sqrt(dot(r.w, pr.sigma * r.w)) for r in calibrated_books],
                       "Names" => [count(>(1e-6), r.w) for r in calibrated_books]);
             formatters = [(v, i, j) -> if j == 2
                               round(v; sigdigits = 4)
                           elseif j == 3
                               "$(round(v * 100, digits = 2)) %"
                           elseif j == 4
                               "$(round(v * 10_000, digits = 2)) bp"
                           else
                               v
                           end], title = "A radius the sample chose")

#=
The two rules land in different places, and the gap between them is the whole reading.
`ResidualInflation` returns about `0.13` here — three orders of magnitude below the `100.0` the
sweep above needed to move the book — and that is the point rather than a defect: a chi-squared
bound on a residual variance is a statement about *estimation error*, and over this sample that
error is small. A radius that size barely moves the book, and the table says so: 90% of the
metric-scaled weight still sits outside the factor span. `VarianceFraction` is the rule to reach
for when you want the book to *move*, because it is sized against the nominal variance rather
than against the sampling error, and it is linear in `f` — the `f = 0.5` row is exactly five
times the `f = 0.1` row.

Neither answer is more correct than the other. They price different things, and stating `100.0`
prices a third thing that nothing in the sample asked for. What the rules buy is that the number
now moves with the data instead of holding still across every fold.

`VarianceFraction` reads a reference portfolio, and `w0` admits a weight vector or any
non-finite-allocation optimiser — the optimiser carries its own solver, so nothing extra is
threaded into the fit. `nothing` reads the equal-weight book.

Two notes on where each rule applies. `ResidualInflation` reads the idiosyncratic variances off
`rr.esigma`, so it refuses a block fitted without a residual term; `VarianceFraction` reads none
and serves that block too. And `ResidualInflation`'s own `q` defaults to the estimator's, so one
confidence level governs both axes unless you state otherwise — the two are tail probabilities
over different errors, but they tighten in the same direction.
=#

vf_book = book(; kappa = VarianceFraction(; f = 0.1, w0 = InverseVolatility()))
pretty_table(DataFrame("Reference" => ["Equal weight (default)", "InverseVolatility()"],
                       "Outside the span" =>
                           [orthogonal_share(calibrated_books[4].w, sigma_set),
                            orthogonal_share(vf_book.w, sigma_set)]);
             formatters = [(v, i, j) -> j == 2 ? "$(round(v * 100, digits = 2)) %" : v],
             title = "The fraction is measured at a portfolio you choose")

#=
### The radius is also searchable

Nothing above had to be chosen in advance. `kappa` is a plain field, so its lens path
`"ucs.kappa"` is a key a search grid ranges over, and the grid may hold **rules beside numbers**:
each candidate is fitted per fold, and the walk-forward score decides. That is the third route,
after stating a size and calibrating one.

```julia
grid = ["r.ucs.kappa" => [0.0, 1.0, 100.0, ResidualInflation(), VarianceFraction(; f = 0.1)]]
search_cross_validation(mr, GridSearchCrossValidation(grid; cv = IndexWalkForward(252, 63)), rd)
```
=#

#=
### The mean radius is a threshold, not a dial

The mean set behaves differently, and the difference is worth understanding. Its penalty is a
**norm**, `−κ‖Lᵀw‖`, and a norm is not differentiable at zero. So any strictly positive radius
drives the orthogonal component to exactly zero, and raising it further changes nothing. The
maximum-return book below pays for that with 5 bp of expected return and goes from 10 names to 73.
=#

radius_grid = [0.0, 0.5, 1.0, 2.0, 4.0, mu_set.kappa]
radius_books = [book(; radius = rad, obj = MaximumReturn()) for rad in radius_grid]

pretty_table(DataFrame("Radius" => radius_grid,
                       "Outside the span" =>
                           [orthogonal_share(r.w, sigma_set) for r in radius_books],
                       "Volatility" =>
                           [sqrt(dot(r.w, pr.sigma * r.w)) for r in radius_books],
                       "Expected return" => [dot(r.w, pr.mu) for r in radius_books],
                       "Names" => [count(>(1e-6), r.w) for r in radius_books]);
             formatters = [(v, i, j) -> if j == 1
                               round(v; digits = 3)
                           elseif j == 2
                               "$(round(v * 100, digits = 4)) %"
                           elseif j == 3 || j == 4
                               "$(round(v * 10_000, digits = 2)) bp"
                           else
                               v
                           end], title = "The mean norm-ball radius")

#=
!!! note "The tangency objective and a mean uncertainty set"
    [`MaximumRatio`](@ref) solves a homogenised problem in a scaled variable `k`, and a mean
    uncertainty set wide enough that no feasible portfolio's worst case beats `rf` leaves nothing
    to pin that scale: the objective is then non-positive along every ray and its supremum sits at
    the origin. `MaximumRatio` writes a floor `k >= kmin` for exactly this, so the constraints
    stay meaningful and the recovered weights keep the mandate. A `k` that comes back **on** the
    floor is the signal that there was no tangency portfolio to find, and that the weights beside
    it maximise the return expression at that scale rather than the ratio. The books on this page
    use [`MinimumRisk`](@ref) and [`MaximumReturn`](@ref), whose scale is fixed at one, so the
    question does not arise for them.

## 5. A walk-forward, under a factor mandate

Nothing above is worth much if it only holds on the sample it was fitted on. A
[`WalkForward`](@ref) refits everything per fold: the prior refits on the fold's own rows, the two
uncertainty sets refit against **that fold's** factor model, and a factor-exposure constraint
written in a factor **name** is re-based through the loadings the fold actually fitted.

The mandate below is one line of the constraint grammar — `"size >= 0.10"` — wrapped in an
[`ExposureConstraintEstimator`](@ref) that declares the space the name lives in.
=#

mandate = ExposureConstraintEstimator(;
                                      lce = LinearConstraintEstimator(;
                                                                      val = "size >= 0.10"),
                                      space = FactorSpace())
ucs_wf = OrthogonalUncertaintySet(; kappa = 100.0)
strategy = MeanRisk(; r = UncertaintySetVariance(; ucs = ucs_wf), obj = MinimumRisk(),
                    opt = JuMPOptimiser(; pe = pe, slv = solver, bgt = 1.0,
                                        wb = WeightBounds(; lb = 0.0, ub = 0.1),
                                        sets = universe, lcse = mandate,
                                        ret = ArithmeticReturn(; ucs = ucs_wf)))

walk = IndexWalkForward(252, 63)
folds = cross_val_predict(strategy, rd, walk)

size_k = findfirst(isequal("size"), rr.nf)
fold_priors = [p.res.pa.pr for p in folds.pred]
fold_sets = [sigma_ucs(OrthogonalUncertaintySet(), prf) for prf in fold_priors]

pretty_table(DataFrame("Fold" => eachindex(folds.pred),
                       "Fit rows" => [size(prf.X, 1) for prf in fold_priors],
                       "Size exposure" =>
                           [(transpose(fold_priors[i].rr.M) * folds.pred[i].res.w)[size_k]
                            for i in eachindex(folds.pred)],
                       "norm(M_fold - M_full)" =>
                           [norm(prf.rr.M - rr.M) for prf in fold_priors],
                       "Outside the span" =>
                           [orthogonal_share(folds.pred[i].res.w, fold_sets[i])
                            for i in eachindex(folds.pred)],
                       "Names" => [count(>(1e-6), p.res.w) for p in folds.pred]);
             formatters = [numfmt],
             title = "Every fold refits the prior, both sets and the mandate")

#=
Three things to read off that table. The fold loadings differ from the full-sample loadings, so
the prior really did refit. The share outside the span differs per fold, so the sets really were
rebuilt against each fold's own factor model. And the size exposure is `0.1` in every fold: the
mandate binds exactly, in a basis that was refitted underneath it.

## 6. Factor attribution

The last question is where the book's risk and return actually came from.
[`factor_attribution`](@ref) answers it twice. The **predicted** decomposition reads the prior's
own moments; the **realised** one reads a return series and decomposes what happened.
=#

final = book(; kappa = 100.0, constraint = mandate)
predicted = factor_attribution(final.w, final.pa.pr; assets = true)

#=
The realised call needs returns. Our panel lists a fifth of its assets late, so a held asset can
carry a non-finite return at an observation before it listed. `strict = false` warns and zeroes
those pairs rather than refusing; `strict = true` would refuse. The warning is the point — it
names the assets and counts the pairs, so an understated total is never silent.
=#

realised = factor_attribution(final.w, final.pa.pr, rd.X; assets = true, strict = false)

#=
Both decompositions are exact: the systematic, idiosyncratic and unattributed parts sum to the
portfolio's own volatility and return, and the per-factor contributions sum to the systematic
part.
=#

function attribution_residuals(a)
    return (a.sys.vol_contrib + a.idio.vol_contrib + a.unattr.vol_contrib -
            a.total.vol_contrib,
            a.sys.mu_contrib + a.idio.mu_contrib + a.unattr.mu_contrib - a.total.mu_contrib,
            a.sys.pct_var + a.idio.pct_var + a.unattr.pct_var - 1,
            sum(a.fbd.vol_contrib) - a.sys.vol_contrib,
            sum(a.fmbd.vol_contrib) - a.sys.vol_contrib)
end

residuals = [attribution_residuals(predicted), attribution_residuals(realised)]

pretty_table(DataFrame("Decomposition" => ["Predicted", "Realised"],
                       "sigma parts - sigma" => [r[1] for r in residuals],
                       "mu parts - mu" => [r[2] for r in residuals],
                       "variance shares - 1" => [r[3] for r in residuals],
                       "factors - systematic" => [r[4] for r in residuals],
                       "families - systematic" => [r[5] for r in residuals]);
             formatters = [numfmt], title = "The decomposition closes")

pretty_table(DataFrame("Component" =>
                           ["Systematic", "Idiosyncratic", "Unattributed", "Total"],
                       "Predicted sigma contribution" =>
                           [predicted.sys.vol_contrib, predicted.idio.vol_contrib,
                            predicted.unattr.vol_contrib, predicted.total.vol_contrib],
                       "Predicted % variance" =>
                           [predicted.sys.pct_var, predicted.idio.pct_var,
                            predicted.unattr.pct_var, 1.0],
                       "Realised sigma contribution" =>
                           [realised.sys.vol_contrib, realised.idio.vol_contrib,
                            realised.unattr.vol_contrib, realised.total.vol_contrib],
                       "Realised mu contribution" =>
                           [realised.sys.mu_contrib, realised.idio.mu_contrib,
                            realised.unattr.mu_contrib, realised.total.mu_contrib]);
             formatters = [numfmt], title = "Where the book's risk and return came from")

pretty_table(DataFrame("Family" => rr.fam, "Factor" => rr.nf,
                       "Predicted sigma contribution" => predicted.fbd.vol_contrib,
                       "Predicted % variance" => predicted.fbd.pct_var,
                       "Realised sigma contribution" => realised.fbd.vol_contrib);
             formatters = [numfmt], title = "By factor")

#=
## Where to go next

  - [Cross-sectional factor model through a Pipeline](06_Cross_Sectional_Factor_Pipeline.md)
    reaches the very same weights with the panel entering as a Pipeline Data Slot.
  - [Factor priors](../2_moments_priors/04_Factor_Priors.md) is the time-series counterpart of §2.
  - [Uncertainty sets](../2_moments_priors/09_Uncertainty_Sets.md) covers the box, ellipsoidal and
    norm-ball shapes the orthogonal sets of §4 specialise.
  - [Factor exposure constraints](../4_constraints_costs/10_Factor_Exposure_Constraints.md) is the
    full grammar behind the one-line mandate of §5.
=#
