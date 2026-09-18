# 1149 — The ledger of online portfolio selection algorithms: state, update, geometry, forecast, solver, regret

Research ticket #1149 of wayfinder map #1148. Written 2026-09-18.

Sources are `research/prototypes/09_online_portfolio_selection.jl` (745 lines, the seven
strategies and two hindsight benchmarks it ships, read in full), and the papers named by author
and year in each row. Every paper marked **read** was fetched as a PDF and its text extracted;
a paper marked **survey** is paywalled or bot-blocked and its row rests on Li and Hoi (2014,
arXiv 1212.2129, read in full) or on a later paper that restates it, named in the row. Nothing
in this note names, links or describes any software implementing these algorithms. The library
census in section 4 comes from `grep -rn "<: AbstractExpectedReturnsEstimator" src/` and a read
of `src/05_Moments/` and `src/18_ExpectedReturns.jl` at the tip of `dev`, `4c070a9792`.

---

## Summary

- **Thirty-six rows** across the five families of Li and Hoi's taxonomy: 4 benchmarks, 11
  follow-the-winner, 9 follow-the-loser, 7 pattern-matching, 5 meta-learning. The prototype's
  nine are the first rows of their families and every one of them was checked line by line
  against its paper; the differences are section 6.
- **Thirteen rows read a forecast of the next price relative** (section 3.1). Three of them
  read the last price relative or its reciprocal (nothing to estimate), five read a statistic of
  a **window of price levels** (a simple moving average, an exponential moving average, an L1
  median, a peak), and five read a **conditional distribution over matched history** (the
  pattern-matching family). The library has **no estimator for any of the price-level statistics
  and no spatial median**: `WindowedExpectedReturns` and `ExpWeightedExpectedReturns` average
  *returns*, and `MedianExpectedReturns` is a coordinatewise median of returns, which is the
  Weiszfeld *initialiser* of Huang and co-authors (2013), not the median they revert to.
- **Four rows carry a matrix state that is not a return covariance** (section 3.2): the online
  Newton step and its meta twin carry a Gram matrix of gradients plus a gradient sum; the
  confidence-weighted reversion carries a diagonal covariance *of the portfolio weights*; the
  anti-correlation rule recomputes a lagged cross-correlation of log price relatives from a
  window each period, so it is a window state rather than a carried matrix.
- **Four rows need an inner programme under the simplex constraint** (section 3.3): the
  paper's online Newton step (a projection in the `A_t` norm is a quadratic programme), every
  pattern-matching rule (a log-optimal portfolio over the matched sample, a concave programme
  the papers solve iteratively), the short-term sparse portfolio (an ADMM loop), and the
  hindsight best constant rebalanced portfolio (Cover's fixed point). Everything else is a
  closed form followed by the Euclidean simplex projection of Duchi and co-authors (2008), or a
  wealth-weighted average of experts.
- **Where the prototype differs from the paper** (section 6): the online Newton step projects in
  the Euclidean norm where the paper projects in the `A_t` norm; the moving-average reversion
  admits `window = 2` and any `epsilon` where the paper requires `w >= 3` and `eps > 1`; the
  median reversion uses the plain Weiszfeld iteration seeded at the mean with an absolute
  tolerance where the paper uses the modified iteration of Vardi and Zhang (2000) seeded at the
  coordinatewise median with a relative one, and defaults `epsilon = 10` where the paper sets 5;
  the universal portfolio is a Monte Carlo average where the paper is an integral; and the
  `best_crp` docstring claims an assertion the code does not make.

---

## 1. Notation

The prototype's, kept throughout. `T` periods, `N` assets, `X` the `T x N` price-relative
matrix with `x_t` its row `t` (`x_t = 1 .+ r_t` for simple returns `r_t`). `w_t` is the portfolio
held during period `t`, decided before `x_t` is seen; `S_t = prod_{s<=t} <w_s, x_s>` is wealth.
`p_t = prod_{s<=t} x_s` (elementwise) is the relative price level, `p_0 = 1`. `Proj_Delta` is
the Euclidean projection onto the simplex (`project_simplex`). `xbar` is the mean of a vector
over assets, `1` the ones vector, `eta` a learning rate, `eps` a reversion threshold, `w` a
window length. The papers write `b_t` for `w_t` and `m` or `d` for `N`; the survey writes `n`
for `T`.

Every rule below is `(state, w_t, x_t) -> (state', w_{t+1})`, and every rule is *online*:
`w_{t+1}` reads rows `1:t` of `X` only. The hindsight benchmarks are not rules; they read all
of `X`.

## 2. The ledger

Column key. **state**: what is carried between periods beyond `w_t`. **geometry**: the
divergence the unconstrained step minimises against, and whether the paper's constrained step
is a projection or an inner programme. **forecast**: what prediction of `x_{t+1}` the step reads
and what the library has for it (section 4). **solver**: closed form, a one-dimensional root, a
fixed point, a quadratic programme, a Monte Carlo integral. **regret**: the bound and its
comparator. **cadence**: every paper updates once per observed price relative; the column
records what the rule can do when several periods pass unobserved, which no paper defines.
**Names** give the acronym the literature uses and a full-word spelling in the library's
convention (`HierarchicalRiskParity`, not `HRP`).

### 2.1 Benchmarks

| name, acronym, source | taxonomy | state | update | geometry | forecast + library stand-in | hyperparameters | solver | regret | cadence |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Uniform buy-and-hold, **UBAH**, `UniformBuyAndHold`; Li and Hoi (2014) §3.1.1, **read**. Absent from the prototype. | benchmark | none (the drift is implicit) | `w_{t+1} = (w_t .* x_t) / <w_t, x_t>`; `S_T = (1/N) sum_i prod_t x_{t,i}` | none; no step is taken | none | none | closed form | none; it *is* the market comparator | every period; a gap folds exactly because `prod_s x_s` composes elementwise |
| Best stock in hindsight, **Best**, `BestStockInHindsight`; Li and Hoi (2014) §3.1.2, **read**; prototype `best_stock` | benchmark (hindsight) | none | `i* = argmax_i prod_t x_{t,i}`, `w_t = e_{i*}` for all `t` | none | none | none | closed form (one product per asset) | none; a comparator | not a rule |
| Uniform constant rebalanced portfolio, **UCRP**, `UniformConstantRebalanced`; Cover (1991), Li and Hoi (2014) §3.1.3, **read**; prototype `uniform_crp` | benchmark | none | `w_{t+1} = 1/N` | rebalance to a fixed point of the simplex | none | none | closed form | `O(N log T)` gap to BCRP is *not* guaranteed; UCRP is a fixed comparator, and Cover proves BCRP beats it | every period; a gap is a missed rebalance, the drifted holding is `(1/N) x_gap / mean(x_gap)` |
| Best constant rebalanced portfolio, **BCRP**, `BestConstantRebalanced`; Cover (1984) for the fixed point, Cover (1991) as the regret comparator, **survey** (paywalled), Li and Hoi (2014) §3.1.3, **read**; prototype `best_crp` | benchmark (hindsight) | none; the solve reads all of `X` | `w* = argmax_{w in Delta} sum_t log <w, x_t>`; Cover's fixed point `w_i <- w_i * (1/T) sum_t x_{t,i} / <w, x_t>`, renormalised | concave programme on the simplex; the iteration is a minorise-maximise scheme | none | `iters = 20_000`, `tol = 1e-12` (prototype) | fixed point (prototype), or any concave solver | defines regret: `R_T = log S_T(BCRP) - log S_T(strategy)` | not a rule; the windowed form is the VRP row below |

### 2.2 Follow-the-winner

| name, acronym, source | taxonomy | state | update | geometry | forecast + library stand-in | hyperparameters | solver | regret | cadence |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Universal portfolio, **UP**, `UniversalPortfolio`; Cover (1991), **survey** (paywalled), Li and Hoi (2014) §3.2.1, **read**; prototype `universal_portfolio` | follow-the-winner (aggregate of CRP experts) | the wealth `S_t(b)` of every CRP `b` on the simplex; the prototype carries `logS` of `n_experts` sampled `b` | `w_{t+1} = int_Delta b S_t(b) dmu(b) / int_Delta S_t(b) dmu(b)`, `S_t(b) = prod_{s<=t} <b, x_s>`, `mu` uniform. Prototype: `logS .+= log.(B * x_t)`, `w = B' softmax(logS)` | none in the divergence sense; it is a Bayesian mixture (an exponentially weighted average forecaster under log loss) and never leaves the simplex | none | prior `mu` (uniform); prototype `n_experts = 2000`, `rng` | Monte Carlo integral (prototype); exact integral is `O(T^N)`; Kalai and Vempala (2002, **read**) sample it in `O(N^7 T^8)` by a rapidly mixing random walk with the guarantee kept | `log S_T(BCRP) - log S_T(UP) <= (N - 1) log(T + 1)`, every sequence; the Monte Carlo approximation weakens it to the sampled experts' best | every period; a gap **cannot** be folded: `prod_s <b, x_s> != <b, prod_s x_s>`, so every expert's wealth needs every row |
| Dirichlet(1/2) universal portfolio, `DirichletUniversalPortfolio`; Cover and Ordentlich (1996), **survey**, Li and Hoi (2014) §3.2.1 | follow-the-winner | as UP | as UP with `mu = Dirichlet(1/2, ..., 1/2)` | as UP | none | none | as UP | same order as UP, better constant (their Theorem 2): `<= (N - 1)/2 log(T + 1) + O(1)`, roughly | as UP |
| Exponentiated gradient, **EG**, `ExponentiatedGradient`; Helmbold, Schapire, Singer and Warmuth (1998), **read**; prototype `exponentiated_gradient` | follow-the-winner | none | `w_{t+1,i} ∝ w_{t,i} exp(eta x_{t,i} / <w_t, x_t>)`, normalised (their Eq. 3.3); the argmax of `eta log <w, x_t> - D_KL(w || w_t)` after linearising the log at `w_t` | relative entropy `D_KL(w || w_t)`; the multiplicative form stays on the simplex, no projection | last price relative `x_t` (the step assumes it repeats); nothing to estimate: `1 .+ X[end, :]` | `eta = 0.05` (paper §5.2, "learning rates around 0.05"; 0.01 to 0.15 all good; above 1 loses money on their two-stock case) | closed form, `O(N)` | Theorem 4.1: with `x_{t,i} >= r > 0` and `eta = r sqrt(2 log N / T)`, `log S_T(u) - log S_T(EG) <= sqrt(T log N / (2 r^2))` for every `u`, i.e. `O(sqrt(T log N))` against BCRP; Theorem 4.2's `EG(alpha, eta)` mixes with uniform and reaches `O(T^{3/4})` without the lower bound `r`; a doubling trick makes it universal (Cor. 4.3) | every period; a gap folded as one compounded relative is one EG step, which is *not* the sum of the per-period steps |
| Gradient projection and expectation-maximisation updates, **GP**, **EM**; Helmbold and co-authors (1997), **survey**, Li and Hoi (2014) §3.2.2 | follow-the-winner | none | GP: `w_{t+1,i} = w_{t,i} + eta (x_{t,i}/<w_t,x_t> - (1/N) sum_j x_{t,j}/<w_t,x_t>)`; EM: `w_{t+1,i} = w_{t,i} (eta (x_{t,i}/<w_t,x_t> - 1) + 1)` | GP: Euclidean `(1/2)||w - w_t||^2`; EM: chi-squared `(1/2) sum (w_i - w_{t,i})^2 / w_{t,i}` | last price relative, as EG | `eta` | closed form | GP `O(sqrt(N T))` (worse than EG); EM is the first-order approximation of EG | as EG |
| Follow the leader / successive constant rebalanced portfolios, **FTL**, **SCRP**, `FollowTheLeader`, `SuccessiveConstantRebalanced`; Gaivoronski and Stella (2000), **survey**, Li and Hoi (2014) §3.2.3 | follow-the-winner | all rows `1:t` (the BCRP-to-date needs them), or a warm start `w*_t` | `w_{t+1} = w*_t = argmax_{w in Delta} sum_{s<=t} log <w, x_s>`; Ordentlich (1996) mixes it with uniform, `w_{t+1} = t/(t+1) w*_t + 1/(t+1) 1/N` | none on the step; the inner programme is the BCRP programme on the prefix | none | none (Ordentlich's mix has none) | concave programme per period (Cover's fixed point, or stochastic optimisation as in their Algorithm 1) | `O(K^2 log T)` (their Theorem 1) with `K` a bound on the gradient of `log <w, x>`; same order as UP with a worse constant | every period; a gap is a missing row of the prefix, the programme is unchanged |
| Weighted successive constant rebalanced portfolio, **WSCRP**, `WeightedSuccessiveConstantRebalanced`; Gaivoronski and Stella (2000), **survey** | follow-the-winner | as SCRP | `w_{t+1} = (1 - gamma) w*_t + gamma w_t` | a convex mix of the leader and the last portfolio | none | `gamma in [0, 1]` | as SCRP | `O(K^2 log T)` (their Theorem 4) | as SCRP |
| Variable rebalanced portfolio, **VRP**, `WindowedBestConstantRebalanced`; Gaivoronski and Stella (2000), **survey** | follow-the-winner (non-stationary) | the last `W` rows | `w_{t+1} = argmax_{w in Delta} sum_{s = t-W+1}^{t} log <w, x_s>` | BCRP programme on a window | none | `W` | concave programme per period | none given | every period; a gap shortens the window |
| Online Newton step, **ONS**, `OnlineNewtonStep`; Agarwal, Hazan, Kale and Schapire (2006), **survey** (paywalled; the general form is Hazan, Agarwal and Kale (2007), **read**, Fig. 2 and Theorem 2); prototype `online_newton_step` | follow-the-winner (follow the regularised leader) | `A_t = I + sum_{s<=t} g_s g_s'` (`N x N`), `b_t = (1 + 1/beta) sum_{s<=t} g_s`, with `g_s = x_s / <w_s, x_s>` | `w_{t+1} = (1 - eta) Proj_Delta^{A_t}(delta A_t^{-1} b_t) + eta 1/N` | the `A_t`-weighted norm `(y - w)' A_t (y - w)`: the paper's constrained step is a **projection in that norm**, which is a quadratic programme (Hazan and co-authors 2007 §4); the prototype substitutes the Euclidean projection | last price relative through `g_t`; nothing to estimate | `beta = 1`, `delta = 1/8`, `eta = 0` (the paper's experiments, as restated by Lai and co-authors 2018 and Tsang, Sit and Wong 2022, both **read**); Hazan and co-authors (2007) set `beta = (1/2) min(1/(4GD), alpha)`, `epsilon = 1/(beta^2 D^2)` | quadratic programme (paper); a linear solve `A_t \ b_t` plus `Proj_Delta` (prototype), `O(N^2)` per step with a rank-one inverse update, `O(N^3)` naively | `O(N^{1.5} log(N T))` (their Theorem 1, as the survey states it); Hazan and co-authors (2007) Theorem 2: `Regret_T(ONS) <= 5 (1/alpha + G D) N log T` for `alpha`-exp-concave losses with gradient bound `G` on a set of diameter `D` | every period; a gap folded as one compounded relative adds one rank-one term where the paper would add several, so the Gram matrix undercounts |
| Exp-concave follow the leader, `ExpConcaveFollowTheLeader`; Hazan and Kale (2012), **survey**, Li and Hoi (2014) §3.2.4 | follow-the-winner | the prefix, or the FTAL state `(A_t, b_t)` of Hazan and co-authors (2007) Fig. 3 | `w_{t+1} = argmax_{w in Delta} sum_{s<=t} log <w, x_s> - (1/2)||w||^2` | an `L2` regulariser on `w` alone (not on the step) | none | none | concave programme, or the ONS-form projection `Proj^{A_t}(A_t^{-1} b_t)` | `O(N log(Q + N))` with `Q` the quadratic variability (`(T - 1)` times the sample variance of the price relatives); `Q << T` in practice | every period |
| Aggregating algorithm, **AA**, `AggregatingAlgorithm`; Vovk and Watkins (1998), **survey**, Li and Hoi (2014) §3.2.5 and §3.5.1 | follow-the-winner / meta | the weight `P_t` of every expert in a countable or measurable set | `w_{t+1} = int_Delta b prod_{s<=t} <b, x_s>^eta P_0(db) / int_Delta prod_{s<=t} <b, x_s>^eta P_0(db)`; in general `P_{t+1}(A) = int_A beta^{loss(x_t, gamma_t(theta))} P_t(dtheta)`, `beta = e^{-eta}` | Bayesian mixture under log loss; UP is `eta = 1`, uniform `P_0` | none | `eta`, prior `P_0` | integral over experts (Monte Carlo, or finite experts exactly) | as UP for `eta = 1`; competitive with the best fixed expert in general | as UP |
| Switching portfolios, **SP**, `SwitchingPortfolio`; Singer (1997), **survey**, Li and Hoi (2014) §3.2.5 | follow-the-winner (regime switching) | the posterior over basic strategies | fixed-`gamma` form (their Eq. 6): `w_{t+1} = (1 - gamma - gamma/(N - 1)) w_t + gamma/(N - 1)`; varying `gamma` has no closed form | a mixture that decays geometrically to uniform | none | `gamma` (switching probability) | closed form (fixed `gamma`) | lower bound of log wealth against any switching regime in hindsight (their Theorem 2) | every period |

### 2.3 Follow-the-loser

| name, acronym, source | taxonomy | state | update | geometry | forecast + library stand-in | hyperparameters | solver | regret | cadence |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Anti-correlation, **Anticor**, `AntiCorrelation`; Borodin, El-Yaniv and Gogan (2004, JAIR 21, arXiv 1107.0036), **read** | follow-the-loser | the last `2w` rows of `X` (two windows); nothing else carried | `LX1 = log X[t-2w+1:t-w, :]`, `LX2 = log X[t-w+1:t, :]`, `mu_k`, `sigma_k` their column means and standard deviations; `Mcov(i,j) = (1/(w-1)) (LX1[:,i] - mu_1(i))'(LX2[:,j] - mu_2(j))`, `Mcor = Mcov / (sigma_1(i) sigma_2(j))` or 0. `claim_{i->j} = Mcor(i,j) + A(i) + A(j)` iff `mu_2(i) >= mu_2(j)` and `Mcor(i,j) > 0`, `A(h) = |Mcor(h,h)|` if `Mcor(h,h) < 0` else 0. `transfer_{i->j} = w_{t,i} claim_{i->j} / sum_j claim_{i->j}`; `w_{t+1,i} = w_{t,i} + sum_{j != i} (transfer_{j->i} - transfer_{i->j})` | none; a heuristic wealth transfer that conserves the budget and never leaves the simplex | none explicitly; the lagged cross-correlation *is* the forecast (asset `j` will emulate asset `i`'s past growth); no library estimator computes a lagged cross-window correlation of log price relatives | `w` (their Figure 2 scans 2 to 30); the paper's headline is `BAH_W(Anticor)`, a uniform buy-and-hold over experts `w in 2:W` with `W = 30` (arbitrary, their §3), and `Anticor(Anticor)` which runs the rule over the experts' own wealth | closed form, `O(w N^2)` per step | none ("difficult to obtain a useful bound", survey §3.3.1) | every period; a gap breaks the two `w x N` windows unless it is folded as one log relative |
| Passive-aggressive mean reversion, **PAMR** (`PAMR-0/1/2`), `PassiveAggressiveMeanReversion`; Li, Zhao, Hoi and Gopalkrishnan (2012, Machine Learning 87), **read** (publisher page, full text); prototype `pamr` | follow-the-loser | none | `tau_t = max(0, (<w_t, x_t> - eps) / ||x_t - xbar_t 1||^2)` (PAMR), `max(0, min(C, ...))` (PAMR-1), `max(0, (<w_t,x_t> - eps) / (||x_t - xbar_t 1||^2 + 1/(2C)))` (PAMR-2); `w_{t+1} = Proj_Delta(w_t - tau_t (x_t - xbar_t 1))` | Euclidean `(1/2)||w - w_t||^2` s.t. `<w, x_t> <= eps` (PAMR-1 adds `+ C xi`, PAMR-2 `+ C xi^2` with the slack in the constraint); non-negativity dropped in the derivation, restored by `Proj_Delta` | last price relative `x_t`, read *contrarian*; nothing to estimate | `eps = 0.5` (`0 <= eps <= 1` in the paper), `C = 500`; the paper also runs a uniform buy-and-hold mixture of PAMR with ONS, Anticor and BNN | closed form, `O(N)` | none for log wealth; the paper notes the `eps`-insensitive loss "would achieve a potential regret of `O(sqrt(T))`" on that loss, not on log wealth | every period; a gap folded as one relative is one PAMR step |
| Confidence-weighted mean reversion, **CWMR** (`-Var`, `-Stdev`), `ConfidenceWeightedMeanReversion`; Li, Hoi, Zhao and Gopalkrishnan (2011, AISTATS, **read**; 2013, TKDD, **survey**) | follow-the-loser | `mu_t` (the portfolio mean, `= w_t`) and `Sigma_t`, a **diagonal `N x N` covariance of the portfolio weights** (the confidence in each weight), seeded `mu_1 = 1/N`, `Sigma_1 = I/N^2` | `M_t = <mu_t, x_t>`, `V_t = x_t' Sigma_t x_t`, `xbar_t = 1' Sigma_t x_t / 1' Sigma_t 1`. CWMR-Var: `mu_{t+1} = mu_t - lambda Sigma_t (x_t - xbar_t 1) / M_t`, `Sigma_{t+1}^{-1} = Sigma_t^{-1} + 2 lambda phi diag(x_t)^2`; CWMR-Stdev: same mean step, `Sigma_{t+1}^{-1} = Sigma_t^{-1} + lambda phi diag(x_t)^2 / sqrt(U_t)`, `U_t = ((-lambda V_t phi + sqrt(lambda^2 V_t^2 phi^2 + 4 V_t)) / 2)^2`. Then `mu_{t+1} = Proj_Delta(mu_{t+1})` and `Sigma_{t+1} <- Sigma_{t+1} / (N^2 tr(Sigma_{t+1}))`. `lambda` is the positive root of a quadratic `a lambda^2 + b lambda + c = 0` (their Eq. 5 and 7), `lambda = max(root_1, root_2, 0)` | KL divergence between Gaussians `D_KL(N(mu, Sigma) || N(mu_t, Sigma_t))` s.t. `Pr[<w, x_t> <= eps] >= theta`; the 2011 text writes the constraint on `log <mu, x_t>`, the 2013 text on `<mu, x_t>`; mean projected to the simplex, covariance rescaled | last price relative, contrarian | 2011: `phi = Phi^{-1}(theta) = 2` (`theta = 95%`), `eps = -0.5` on the log form; 2013 and later defaults: `eps = 0.5`, `phi = 2` on the linear form (Tsang and co-authors 2022, **read**) | a one-dimensional root (quadratic in `lambda`) plus closed forms, `O(N)` with the diagonal | none | every period |
| Online moving average reversion, **OLMAR** (`OLMAR-1`), `OnlineMovingAverageReversion`; Li and Hoi (2012, ICML, arXiv 1206.4626), **read**; Li, Hoi, Sahoo and Liu (2015, AIJ 222), **survey** (paywalled); prototype `olmar` | follow-the-loser | the last `w` price levels `p_{t-w+1:t}` (equivalently the last `w - 1` price relatives) | `xhat_{t+1} = MA_t(w) / p_t = (1/w)(1 + 1/x_t + 1/(x_t x_{t-1}) + ... + 1/prod_{i=0}^{w-2} x_{t-i})` (their Eq. 1); `lambda_{t+1} = max(0, (eps - <w_t, xhat>) / ||xhat - xbar 1||^2)`; `w_{t+1} = Proj_Delta(w_t + lambda_{t+1} (xhat - xbar 1))` | Euclidean `(1/2)||w - w_t||^2` s.t. `<w, xhat_{t+1}> >= eps`; non-negativity dropped, restored by `Proj_Delta` | a **simple moving average of price levels** divided by the last price; the library has none (section 4) | `eps = 10`, `w = 5` (their §5.3; `eps > 1`, `w >= 3` in Algorithm 1); sensitivity: wealth rises sharply as `eps` leaves 1 and flattens past a threshold, `w` peaks at a data-dependent value; `BAH_W(OLMAR)` mixes experts `w in 3:W`, `W = 30` | closed form, `O(N)` | none (their §5.5, "without theoretical guarantee") | every period; a gap folded as one relative shortens the average's memory by the gap |
| Exponential moving average reversion, **OLMAR-2**, `OnlineExponentialMovingAverageReversion`; Li, Hoi, Sahoo and Liu (2015), **survey** (paywalled; the form is as the later literature restates it and is not verified here against the paper) | follow-the-loser | the last forecast `xhat_t` | `xhat_{t+1} = alpha 1 + (1 - alpha) xhat_t ./ x_t` (an exponential moving average of price levels, divided by `p_t`); then the OLMAR step | as OLMAR | an **exponential moving average of price levels**; the library's `ExpWeightedExpectedReturns` is an EWMA of *returns* and cannot stand in | `alpha = 0.5`, `eps = 10` | closed form, `O(N)` | none | every period; the recursion has no gap rule |
| Robust median reversion, **RMR**, `RobustMedianReversion`; Huang, Zhou, Li, Hoi and Zhou (2013, IJCAI), **read**; (2016, TKDE 28), **survey** (bot-blocked); prototype `rmr` | follow-the-loser | the last `w` price levels | `mhat = argmin_y sum_{i=0}^{w-1} ||p_{t-i} - y||_2` (the L1 / spatial median, Weber's problem) by the **modified Weiszfeld iteration** of Vardi and Zhang (2000): `T(y) = (1 - eta(y)/gamma(y))^+ Ttilde(y) + min(1, eta(y)/gamma(y)) y`, with `Ttilde(y) = sum_{p != y} (p/||p - y||) / sum_{p != y} 1/||p - y||`, `eta(y) = 1` if `y` is a data point else 0, seeded at the **coordinatewise median** `median(p_t, ..., p_{t-w+1})`, stopped at `||y_{k-1} - y_k||_1 <= tau ||y_k||_1` or `m` iterations; `xhat_{t+1} = mhat / p_t`; then `alpha = min(0, (<w_t, xhat> - eps) / ||xhat - xbar 1||^2)`, `w_{t+1} = Proj_Delta(w_t - alpha (xhat - xbar 1))` | as OLMAR | the **spatial median of price levels** divided by the last price; the library has no spatial median, and `MedianExpectedReturns` (coordinatewise, over returns) is the paper's *initialiser*, not its estimate | `w = 5`, `eps = 5`, `m = 200`, `tau` (their §5; `eps > 1`, `w >= 2`) | fixed point (Weiszfeld, `O(m w N)`) plus a closed form | none | every period; a gap folded as one relative shortens the window |
| Transaction-cost optimisation, **TCO** (`TCO1`, `TCO2`), `TransactionCostOptimisation`; Li, Wang, Huang and Hoi (2018, Quantitative Finance 18), **survey** (paywalled and bot-blocked; the form is Moon's (2019, arXiv 1909.04327, **read**) restatement, Eq. 7 to 10) | follow-the-loser (cost-aware) | the **drifted** holding `what_t = (w_t .* x_t) / <w_t, x_t>` (the paper reads the price-adjusted portfolio, not `w_t`); TCO2 also the last `w` price levels | `v = eta (xhat / <what_t, xhat> - (1/N) 1' (xhat / <what_t, xhat>) 1)`; `w_{t+1,j} = what_{t,j} + sign(v_j) max(|v_j| - lambda, 0)`; then `Proj_Delta` | `argmax_w log <w, xhat> - lambda ||w - what_t||_1` on the simplex, the log linearised at `what_t`; the `L1` term is a soft threshold that leaves small trades at zero | TCO1: `xhat = 1 ./ x_t` (last relative, reciprocal); TCO2: `xhat = MA_t(5) / p_t` as OLMAR | `eta = 10`, `lambda = 10 eta gamma` with `gamma` the proportional cost rate, `w = 5` (TCO2) | closed form, `O(N)` | none | every period; a gap changes `what_t` by the compounded drift, which is exact |
| Peak price tracking, **PPT**, `PeakPriceTracking`; Lai, Dai, Ren and Huang (2018, IEEE TNNLS 29(7), online 2017), **survey** (paywalled; the abstract and Tsang and co-authors' (2022) one-line description are all that was read; the ticket's attribution to Lai, Yang, Fang and Wu is the SSPO paper's author list, corrected here) | follow-the-loser (trend / peak) | the last `w` price levels | `xhat_{t+1} = max_{0<=k<w} p_{t-k} / p_t` (the recent peak over the last price), then a linear "transform function" step towards `xhat` with rate `eta`, projected; the abstract states the objective "can be formulated as a fast backpropagation algorithm". The closed form is **not verified here** | Euclidean, by the abstract's description; unverified | the **peak of a window of price levels** over the last price; the library has none | `w = 5`, `eta` (100 in the later literature; unverified) | closed form (unverified) | none | every period |
| Short-term sparse portfolio optimisation, **SSPO**, `ShortTermSparsePortfolio`; Lai, Yang, Fang and Wu (2018, JMLR 19), **read** | follow-the-loser (peak, sparse) | the last `w` price levels; inside a step the ADMM iterates `(b, g, rho)` | `pmax_i = max_{0<=k<w} p_{t-k,i}`; `phi_t = -(1.1 log(pmax / p_t) + 1)`; ADMM on `min_b <b, phi_t> + lambda ||b||_1 s.t. 1'b = 1`: `b <- (lambda/gamma I + eta 1 1')^{-1} (lambda/gamma g + (eta - rho) 1 - phi_t)`, `g <- sign(b) .* max(|b| - gamma, 0)`, `rho <- rho + eta (1'b - 1)`, seeded `b = g = w_t`, `rho = 0`, until `|1'b - 1| < 1e-4` or `1e4` iterations; `w_{t+1} = Proj_Delta(zeta b)` | a linear objective with an `L1` penalty and a self-financing equality; the augmented Lagrangian is proved to have a saddle point (their §3.3.1); the final step is a Euclidean projection of a **scaled** iterate | the **peak of a window of price levels** through the generalised log return `Rt = 1.1 log(pmax / p_t) + 1`; the library has none | `w = 5`, `lambda = 0.5` (tuned over 0.4 to 0.65), `gamma = 0.01`, `eta = 0.005`, `zeta = 500` (their §4.1) | an inner programme (ADMM with one fixed `N x N` solve), then `Proj_Delta` | none | every period |

### 2.4 Pattern-matching

All rows follow Györfi, Lugosi and Udina's (2006) two-step frame as the survey states it (§3.4): a
**sample selection** `C_t(w, ·)` of past indices whose preceding `w`-window resembles the latest
one, then a **portfolio optimisation** `w_{t+1} = argmax_{w in Delta} U(w; C_t)`, uniform if `C_t`
is empty, and an **aggregation** of the experts indexed by `(w, l)` or `(w, rho)` by their past
wealth, `w_{t+1} = sum_k q_k S_t(h_k) h_k(t+1) / sum_k q_k S_t(h_k)` with a uniform prior `q`
(the buy-and-hold mixture of section 2.5). The state of every row is therefore **all of the
history** `X[1:t, :]` plus the running wealth of every expert; the cadence is every period, and a
gap corrupts the pattern windows unless it is folded as one relative. Györfi's papers are
paywalled (Mathematical Finance, Statistics and Decisions) and their rows rest on the survey.

| name, acronym, source | taxonomy | state | update | geometry | forecast + library stand-in | hyperparameters | solver | regret | cadence |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Histogram-based log-optimal, **BH**, `HistogramPatternMatching`; Györfi and Schäfer (2003), **survey** §3.4.1 | pattern-matching | history + expert wealths | `C_H = {w < i < t+1 : G_l(x_{t-w+1:t}) = G_l(x_{i-w:i-1})}` with `G_l` a discretisation of `R^N_+` into `d_l` cells; `U_L(w; C) = sum_{i in C} log <w, x_i>` | none on the step; a concave programme over the matched sample | the empirical conditional distribution of `x_{t+1}` given the last window (the matched `x_i`, uniform weights); no library estimator | `w` (window), `l` (partition fineness) | concave programme per expert per period | universally consistent: growth rate reaches the optimum for any stationary ergodic market | every period |
| Kernel-based log-optimal, **BK**, `KernelPatternMatching`; Györfi, Lugosi and Udina (2006, Mathematical Finance 16), **survey** §3.4.1, §3.4.3 | pattern-matching | history + expert wealths | `C_K = {w < i < t+1 : ||x_{t-w+1:t} - x_{i-w:i-1}|| <= c/l}` (a uniform kernel of radius `c/l` on the concatenated `w N`-vector); `U_L` as BH | as BH | as BH; a kernel regression over matched windows; no library estimator | `w in 1:K`, `l in 1:L` (the survey reports `K = 5`, `L = 10` grids in this family), `c` | concave programme per expert per period | universally consistent | every period |
| Nearest-neighbour log-optimal, **BNN**, `NearestNeighbourPatternMatching`; Györfi, Udina and Walk (2008, Statistics and Decisions 26), **survey** §3.4.1, §3.4.3 | pattern-matching | history + expert wealths | `C_N = {w < i < t+1 : x_{i-w:i-1} is among the l nearest neighbours of x_{t-w+1:t}}` (Euclidean); `U_L` as BH | as BH | as BH; a nearest-neighbour regression; no library estimator, and no neighbour search in `src/` | `w`, `l` (the number of neighbours, a fraction `p_l` of `t` in the paper) | concave programme per expert per period | universally consistent | every period |
| Kernel-based semi-log-optimal, **BS**, `SemiLogOptimalKernelPatternMatching`; Györfi, Urbán and Vajda (2007), Vajda (2006), **survey** §3.4.2 | pattern-matching | as BK | `C_K` with `U_S(w; C) = sum_{i in C} f(<w, x_i>)`, `f(z) = (z - 1) - (1/2)(z - 1)^2` (the second-order Taylor expansion of `log` at 1) | a quadratic programme over the matched sample | as BK | as BK | quadratic programme (closed-form-ish with the simplex) | universally consistent (Vajda 2006) | every period |
| Kernel-based Markowitz-type, **BM**, `MeanVarianceKernelPatternMatching`; Ottucsák and Vajda (2007), **survey** §3.4.2 | pattern-matching | as BK | `C_K` with `U_M(w; C) = E[<w, x> | C] - lambda Var[<w, x> | C]` | a quadratic programme; the semi-log-optimal is the case of one specific `lambda` | as BK, with the matched sample's mean and variance | as BK plus `lambda` | quadratic programme | none stated | every period |
| Kernel-based transaction-cost (GV-type), **BGV**, `TransactionCostKernelPatternMatching`; Györfi and Vajda (2008), **survey** §3.4.2 | pattern-matching | as BK plus `w_t` | `C_K` with `U_T(w; C) = sum_{i in C} (log <w, x_i> + log c(w_t, w, x_i))`, `c` the fraction of wealth left after proportional costs | a concave programme with the cost factor inside the log | as BK | as BK plus the cost rate | concave programme (the cost factor needs a fixed point per candidate) | growth optimal under a first-order Markov market with known distribution | every period |
| Correlation-driven nonparametric learning, **CORN** (`CORN-U`, `CORN-K`), `CorrelationDrivenNonparametric`; Li, Hoi and Gopalkrishnan (2011, ACM TIST 2), **survey** §3.4.1, §3.4.3; its online appendix (arXiv 1306.1378, **read**) carries the consistency proof only | pattern-matching | history + expert wealths | `C_C = {w < i < t+1 : corr(x_{i-w:i-1}, x_{t-w+1:t}) >= rho}` (Pearson correlation of the two concatenated windows); `U_L` as BH; CORN-U mixes the experts `w in 1:W` uniformly by wealth, CORN-K holds the top-`K` experts by past wealth | as BH | as BH; a correlation-matched sample; no library estimator | `W = 5`, `rho = 0.1` (as Lai and co-authors 2018 restate the defaults), `K = 5` for CORN-K (Tsang and co-authors 2022) | concave programme per expert per period | universally consistent (the appendix) | every period |

### 2.5 Meta-learning

| name, acronym, source | taxonomy | state | update | geometry | forecast + library stand-in | hyperparameters | solver | regret | cadence |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Buy-and-hold over experts, **BAH_W(·)**, `BuyAndHoldMixture`; Borodin and co-authors (2004) §3, **read**; used by OLMAR, PAMR, Anticor and every pattern-matching row | meta-learning | the wealth `S_t(h_k)` of every expert `k` | `w_{t+1} = sum_k S_t(h_k) h_k(t+1) / sum_k S_t(h_k)`, uniform initial split | a wealth-weighted average (UP over a finite expert set) | whatever the experts read | the expert grid (`W`) | closed form over `K` experts | as UP over a finite set: `log S_T(best expert) - log S_T(mix) <= log K` | every period; a gap cannot be folded (as UP) |
| Fast universalisation, **FU**, `FastUniversalisation`; Akcoglu, Drineas and Kannan (2002, 2004), **survey** §3.5.2 | meta-learning | expert wealths | as BAH over a parameterised class of strategies, sampled | as UP | as the experts | the class and the sample | Monte Carlo / sampling | asymptotically the wealth of the best fixed convex combination of experts; reduces to UP when the experts are CRPs | every period |
| Online gradient update, **OGU**, `OnlineGradientUpdate`; Das and Banerjee (2011, KDD), **survey** §3.5.3 (the paper is paywalled; its abstract was read) | meta-learning | the weight vector `p_t` over `K` experts | EG run on the expert-return vector `r_t = (<h_k(t), x_t>)_k`: `p_{t+1,k} ∝ p_{t,k} exp(eta r_{t,k} / <p_t, r_t>)`; `w_{t+1} = sum_k p_{t+1,k} h_k(t+1)` | relative entropy over the expert simplex | as the experts | `eta` | closed form | no worse than the best convex combination of experts, `O(sqrt(T log K))` (as EG); universal if any expert is | every period |
| Online Newton update, **ONU**, `OnlineNewtonUpdate`; Das and Banerjee (2011), **survey** §3.5.3 | meta-learning | `A_t`, `b_t` over the expert-return vectors (`K x K` Gram matrix and gradient sum) | ONS run on `r_t`: `p_{t+1} = Proj^{A_t}_{Delta_K}(delta A_t^{-1} b_t)`; `w_{t+1} = sum_k p_{t+1,k} h_k(t+1)` | the `A_t`-weighted norm on the expert simplex; a quadratic programme, or the Euclidean simplification | as the experts | `beta`, `delta`, `eta` | quadratic programme (or linear solve + projection) | `O(K log T)` against the best convex combination of experts | every period |
| Follow the leading history, **FLH**, `FollowTheLeadingHistory`; Hazan and Seshadhri (2009, ICML), **survey** §3.5.4 | meta-learning (changing environments) | a working set of experts started at different periods, each with its own state (an ONS instance in the paper), and a Herbster-Warmuth weight over them | add an expert each period, prune by performance, weight the survivors by a fixed-share rule, and play the weighted mix | as the base expert plus a fixed-share mixture | as the base experts | the pruning rule and the base expert's parameters | as the base expert | adaptive regret: universal when the base is; the survey reports FLH with ONS beats ONS empirically | every period |

## 3. The three facts the roster ticket needs

### 3.1 Which algorithms read a forecast

A step **reads a forecast** when it consumes a vector `xhat_{t+1}` that stands for the next
price relative and would change if a different predictor were plugged in. Thirteen rows do.
Everything else reads `x_t` only through a loss or a gradient (EG, GP, EM, ONS, PAMR, CWMR, the
meta rows), or reads nothing (the benchmarks, SP), or reads a whole history without a point
forecast (UP, AA, FTL, VRP, FLH).

| forecast | rows | what it is | library stand-in |
|:---|:---|:---|:---|
| last price relative, `x_t` or `1 ./ x_t` | TCO1; and PAMR, CWMR, EG read it through the loss, not as a slot | nothing to estimate | none needed; the last row of the returns matrix, `1 .+ X[end, :]` |
| simple moving average of **price levels** over the last price, `MA_t(w) / p_t` | OLMAR, TCO2 | `(1/w) sum_{i=0}^{w-1} prod_{j=0}^{i-1} 1/x_{t-j}` | **none.** `WindowedExpectedReturns` wraps a returns estimator over the last `window` rows of *returns*; its `mean` is a mean of `r_t`, not this reciprocal product. The quantity is a function of the last `w - 1` price relatives, so a windowed adapter could compute it, but no existing `mean` method does |
| exponential moving average of price levels | OLMAR-2 | `xhat_{t+1} = alpha 1 + (1 - alpha) xhat_t ./ x_t` | **none.** `ExpWeightedExpectedReturns` is an EWMA of *returns* with a holiday freeze and cold-start correction, and its state (`ExpWeightedExpectedReturnsState`) carries a mean of returns, not a price level |
| spatial (L1) median of price levels over the last price | RMR | Weber's point of `p_{t-w+1:t}` | **none.** `grep -rn "Weiszfeld\|spatial median\|geometric median\|L1 median" src/` returns nothing. `MedianExpectedReturns` is the coordinatewise median of *returns* per asset; the coordinatewise median of *prices* is what Huang and co-authors seed the iteration with, not what they revert to |
| peak of a window of price levels over the last price | PPT, SSPO | `max_{k<w} p_{t-k} / p_t` | **none** |
| conditional distribution over matched history | BH, BK, BNN, BS, BM, BGV, CORN | the matched sample `{x_i : i in C_t}`, uniform weights | **none**; no kernel regression, neighbour search or window correlation over an observation axis exists in `src/` (`grep -rn "nearest.neighbour\|knn" src/` hits prose only) |

The one estimator that could host a forecast today is `CustomValueExpectedReturns`, whose `val`
field takes a callable `(X; dims, kwargs...) -> Vector`; it would receive the returns window and
return `xhat - 1`. That is an adapter, not a stand-in: the forecast seam would still own the
price-level reconstruction and the window.

### 3.2 Which algorithms carry a matrix state that is not a return covariance

| row | matrix | what it is | shape | how it moves |
|:---|:---|:---|:---|:---|
| ONS | `A_t = I + sum_s g_s g_s'` | a Gram matrix of scaled gradients `g_s = x_s / <w_s, x_s>`, plus the vector `b_t = (1 + 1/beta) sum_s g_s` | `N x N`, symmetric positive definite | rank-one update per period; never reset; inverted (or its inverse updated by Sherman-Morrison) each step |
| ONU | the same over expert returns | `K x K` | as ONS | as ONS |
| CWMR | `Sigma_t` | a **diagonal covariance of the portfolio weights** (the confidence in each weight), seeded `I / N^2` | `N x N` diagonal, so `N` numbers | its inverse gains `2 lambda phi diag(x_t)^2` per period, then it is rescaled to trace `1/N^2`; a diagonal is all the paper keeps, so the state is a vector in practice |
| Anticor | `Mcor` | a lagged cross-correlation of log price relatives between two consecutive `w`-windows | `N x N`, not symmetric | **recomputed** from the last `2w` rows every period; nothing is carried between periods beyond those rows, so the honest state is the window |

Everything else carries a vector (CWMR's mean, OGU's expert weights, UP's expert log wealths), a
window of rows (OLMAR, RMR, PPT, SSPO, TCO2, VRP, the pattern-matching rows), or nothing.

### 3.3 Which algorithms need an inner programme under constraints

A step **needs an inner programme** when its constrained solution is not a closed form followed
by `Proj_Delta`.

| row | programme | class | what the papers do |
|:---|:---|:---|:---|
| ONS (paper form), ONU | `argmin_{w in Delta} (w - y)' A_t (w - y)` with `y = delta A_t^{-1} b_t` | quadratic programme on the simplex | Hazan, Agarwal and Kale (2007) §4 call it a generalised projection solvable to any accuracy in polynomial time; the prototype replaces it with the Euclidean projection |
| BCRP, FTL/SCRP, VRP, BH, BK, BNN, BGV, CORN | `argmax_{w in Delta} sum_{i in C} log <w, x_i>` (over a prefix, a window or a matched sample) | concave programme on the simplex | Cover's (1984) fixed point, or a general concave solver; BGV adds the cost factor |
| BS, BM | `argmax_{w in Delta} sum_{i in C} f(<w, x_i>)` with `f` quadratic | quadratic programme on the simplex | the semi-log-optimal is the second-order expansion of the row above, introduced to avoid it |
| SSPO | `min_b <b, phi_t> + lambda ||b||_1 s.t. 1'b = 1` | linear objective with an `L1` penalty and an equality | ADMM with one fixed `N x N` solve, a soft threshold and a dual ascent, `<= 1e4` iterations, then `Proj_Delta(zeta b)` |
| Exp-concave FTL | `argmax_{w in Delta} sum_{s<=t} log <w, x_s> - (1/2)||w||^2` | concave programme | Hazan and Kale (2012) solve it in the ONS form |

The **closed-form-plus-projection** rows are EG (no projection), GP, EM, PAMR, CWMR (a scalar
root), OLMAR, OLMAR-2, RMR (a Weiszfeld fixed point on the *forecast*, not on the portfolio),
TCO, PPT (as described), Anticor (no projection), SP, and every wealth-weighted mixture (UP, AA,
BAH, FU, OGU).

## 4. The library's forecast candidates

`grep -rn "<: AbstractExpectedReturnsEstimator" src/` at `4c070a9792` returns seven concrete
subtypes and one abstract one:

| estimator | file | what its `mean` returns | can it stand in a forecast slot |
|:---|:---|:---|:---|
| `SimpleExpectedReturns` | `src/05_Moments/02_SimpleExpectedReturns.jl` | the (weighted) sample mean of returns | as a momentum forecast only; no ledger row reads a mean of returns |
| `ShrunkExpectedReturns` (`JamesStein`, `BayesStein`, `BodnarOkhrinParolya` towards `GrandMean`, `VolatilityWeighted`, `MeanSquaredError`) | `src/05_Moments/15_ShrunkExpectedReturns.jl` | a shrunk sample mean | no row reads it |
| `EquilibriumExpectedReturns`, `ExcessExpectedReturns` | `16_`, `17_` | a covariance-implied or risk-free-adjusted mean | no row reads it |
| `StandardDeviationExpectedReturns`, `VarianceExpectedReturns` | `25_StandardDeviationExpectedReturns.jl` | a volatility proxy | no row reads it |
| `WindowedExpectedReturns` (via `@windowed_estimator`, `src/05_Moments/01_Base_Moments.jl:1965`) | `src/05_Moments/26_Windowed/01_WindowedExpectedReturns.jl` | the wrapped estimator's mean over the last `window` rows of **returns** | the *window* is the right idiom for OLMAR, RMR, PPT, SSPO, Anticor; the *statistic* is wrong for all of them, because none of them averages returns |
| `MedianExpectedReturns` | `src/05_Moments/27_MedianExpectedReturns.jl` | the per-asset (coordinatewise) median of returns, optionally weighted | not the spatial median RMR needs; it is the seed of the paper's iteration if applied to price levels |
| `CustomValueExpectedReturns` | `src/05_Moments/28_CustomValueExpectedReturns.jl` | a scalar, a vector, or a callable's output on `X` | an adapter for any forecast function; carries no window or state of its own |
| `ExpWeightedExpectedReturns` (state `ExpWeightedExpectedReturnsState`) | `src/05_Moments/29_ExpWeighted/01_ExpWeightedExpectedReturns.jl` | an exponentially weighted mean of returns with a holiday freeze, a cold-start correction and `min_obs` | the *recursion* is OLMAR-2's idiom, the *quantity* is not: OLMAR-2 recurses on price levels |

`src/18_ExpectedReturns.jl` holds `ExpectedReturn` and `ExpectedReturnRiskRatio` (risk measures
that read an estimator) and `PerformanceSummaryResult`; it defines no estimator.

**Absent from `src/`, by grep:** a moving average of price *levels* (every windowed and
exponentially weighted estimator averages returns), a spatial or L1 median (no `Weiszfeld`,
`geometric median`, `spatial median` or `L1 median` token anywhere), a windowed peak of price
levels, a lagged cross-window correlation, a kernel or nearest-neighbour regression over the
observation axis. Price levels themselves exist only upstream of `prices_to_returns`
(`src/03_InputData/11_PricesToReturns.jl`); every estimator receives returns, so a price-level
forecast must reconstruct `p_t = prod x_s` from the returns window, as the prototype does.

**Adjacent machinery that is relevant:** `PreviousWeights` and `EqualWeighted`
(`src/17_Optimisation/03_NaiveOptimisation.jl`) are UBAH's and UCRP's optimisers already;
`OnlineStep` and the partial-fit states (`src/01_Base/15_Online.jl`,
`src/17_Optimisation/09_OnlineOptimisation.jl`) supply the fold loop every row would run in, and
`SampleBufferState` is the returns window the price-level forecasts would read.
`docs/src/References.bib` cites none of the papers in this ledger.

## 5. Hyperparameter defaults at a glance

The values the papers themselves use, and the ranges they scan.

| row | defaults | scanned |
|:---|:---|:---|
| UP | uniform prior; prototype `n_experts = 2000` | — |
| EG | `eta = 0.05` | `0.01` to `0.15` all good; `> 1` loses on the two-stock case |
| ONS | `beta = 1`, `delta = 1/8`, `eta = 0` | — |
| Anticor | `w`; `BAH_W`, `W = 30` | `w in 2:30`, `W in 2:50` |
| PAMR | `eps = 0.5`, `C = 500` | `eps in [0, 1]` |
| CWMR | `phi = 2`, `eps = 0.5` (2013) / `-0.5` on the log form (2011) | `eps` scanned, `phi` "not decisive" |
| OLMAR | `eps = 10`, `w = 5`; `BAH_W`, `W = 30` | `eps` past 1, `w in 3:100` |
| OLMAR-2 | `alpha = 0.5`, `eps = 10` | — |
| RMR | `w = 5`, `eps = 5`, `m = 200` | `eps`, `w` scanned |
| TCO | `eta = 10`, `lambda = 10 eta gamma`, `w = 5` | — |
| PPT | `w = 5` | — |
| SSPO | `w = 5`, `lambda = 0.5`, `gamma = 0.01`, `eta = 0.005`, `zeta = 500` | `lambda in 0.4:0.65`, one at a time for the rest |
| BK, BNN | `w in 1:5`, `l in 1:10` | — |
| CORN | `W = 5`, `rho = 0.1`, `K = 5` | — |

## 6. Where the prototype differs from the paper

Checked line by line against the papers named in section 2. "Matches" means the update, the
defaults and the projection agree.

1. **`universal_portfolio`**: Cover (1991) is an integral over the whole simplex; the prototype
   is a Monte Carlo average over `n_experts = 2000` uniform Dirichlet draws, and its docstring
   says so. What it does not say: Kalai and Vempala (2002) keep the bound with a polynomial-time
   sampler, and Cover and Ordentlich (1996) get a better constant from a Dirichlet(1/2) prior.
   The prototype's bound is therefore the sampled experts' best, not `(N - 1) log(T + 1)`.
2. **`exponentiated_gradient`**: matches Helmbold and co-authors (1998) Eq. 3.3 and their
   `eta = 0.05`. The docstring's regret `O(sqrt(T log N))` is the paper's Theorem 4.1 with the
   lower bound `r` on the price relatives folded into the constant; the paper's universal
   variant `EG(alpha, eta)` (mix with uniform, Theorem 4.2) and the doubling trick are absent,
   which is fine for a rule and matters for a guarantee.
3. **`online_newton_step`**: `A_t`, `b_t`, `delta`, `beta` and the `eta` mix with uniform match
   Agarwal and co-authors (2006) as the survey states them, and the defaults `beta = 1`,
   `delta = 0.125`, `eta = 0` are the paper's. The projection differs: the paper projects in the
   norm induced by `A_t` (a quadratic programme, Hazan and co-authors 2007 §4), the prototype
   projects in the Euclidean norm and its docstring calls that "the standard simplification".
   The regret bound is proved for the `A_t` projection (the inequality in their Lemma 8 is the
   reason for it), so the prototype's `O(N log T)` claim is inherited, not proved.
4. **`pamr`**: matches Li and co-authors (2012) Propositions for PAMR, PAMR-1 and PAMR-2, and
   the defaults `eps = 0.5`, `C = 500`. The paper bounds `eps` to `[0, 1]`; the prototype accepts
   any real. The paper's headline mixture (uniform buy-and-hold of PAMR with ONS, Anticor and
   BNN) is absent.
5. **`olmar`**: matches Li and Hoi (2012) Eq. 1 and Algorithm 2, and the defaults `eps = 10`,
   `w = 5`. Two differences. The paper's Algorithm 1 requires `eps > 1` and `w >= 3`; the
   prototype checks `window >= 2` only and takes any `epsilon`. The paper's formula needs
   `t >= w - 1` and says nothing about earlier periods; the prototype truncates the window
   (`lo = max(1, t + 1 - window + 1)`) and averages over the price levels it has, including
   `p_0 = 1`, so its first `w - 2` forecasts are shorter-window averages the paper does not
   define. OLMAR-2 (the exponential moving average) and `BAH_W(OLMAR)` are absent.
6. **`rmr`**: the passive-aggressive step matches Huang and co-authors (2013) Proposition 2 up
   to the sign convention (`alpha = min(0, ·)` subtracted there, `lambda = max(0, ·)` added
   here; identical). The median differs in four ways. (a) The paper uses the modified Weiszfeld
   iteration of Vardi and Zhang (2000), which is defined when the iterate lands on a data point;
   the prototype uses the plain iteration with a `1e-12` floor on the distance. (b) The paper
   seeds at the coordinatewise median of the window; the prototype seeds at the mean. (c) The
   paper stops at `||y_{k-1} - y_k||_1 <= tau ||y_k||_1` (relative, `L1`) after at most
   `m = 200` iterations; the prototype stops at `||y_new - y||_2 < 1e-8` (absolute) after at
   most 100. (d) The paper's default is `eps = 5`; the prototype's is `eps = 10` (OLMAR's).
   The paper's `w >= 2` matches the prototype's check.
7. **`best_crp`**: Cover's (1984) fixed point, as documented. The docstring says "The routine
   asserts that monotonicity" of the log wealth; the code has no assertion (`grep -n assert`
   on the file returns nothing). Either the sentence or an `@assert lw >= prev` is missing.
8. **`best_stock`, `uniform_crp`**: match the survey's definitions. The market benchmark
   (uniform buy-and-hold) is absent from the prototype although every paper quotes it.
9. **`regret_against_bcrp`**: matches the survey's Eq. 2 in log wealth. The prototype's
   `run_online` enforces the no-look-ahead rule once for every strategy; no paper states it as
   a checked invariant, and it is the property the ledger's "cadence" column assumes.
10. **`project_simplex`**: Duchi and co-authors (2008), as documented, and it is the projection
    every reversion paper in section 2.3 cites.
11. **Sourcing correction for the ticket**: "Lai, Yang, Fang, Wu 2018 peak/price tracking and
    sparse portfolio" names two papers. The peak price tracking system is Lai, Dai, Ren and
    Huang (2018, IEEE TNNLS 29(7)); the sparse portfolio is Lai, Yang, Fang and Wu (2018, JMLR
    19). Only the second was readable.

## 7. What could not be sourced from a primary text

Cover (1991), Cover and Ordentlich (1996), Agarwal and co-authors (2006), Gaivoronski and Stella
(2000), Vovk and Watkins (1998), Singer (1997), Györfi's four papers, Ottucsák and Vajda (2007),
Li, Hoi, Sahoo and Liu (2015), Huang and co-authors (2016), Li, Wang, Huang and Hoi (2018), Lai,
Dai, Ren and Huang (2018), Das and Banerjee (2011), Hazan and Kale (2012), Hazan and Seshadhri
(2009) and Akcoglu and co-authors (2004) are paywalled or bot-blocked from this session. Their
rows rest on Li and Hoi (2014), on Hazan, Agarwal and Kale (2007) for the ONS form, on the 2013
IJCAI text of RMR, on the 2011 AISTATS text of CWMR, and on Moon (2019) and Tsang, Sit and Wong
(2022) for the TCO and PPT restatements and the defaults they list. The PPT closed form and the
OLMAR-2 recursion are the two rows whose *update* is unverified against its own paper; every
other update in the table is either read from the paper or is the survey's transcription of it.
