# PortfolioOptimisers.jl — comparison, recommendations, and verified prototypes

**Date:** 2026-08-16 · **Base:** branch `dev`, commit `0ddb6d671`, v0.28.0 · **Author:** Claude (Opus 5)

---

## 1. Summary

This report differs from reports 1 to 7 in one way. **Every claim about the library was checked
against the source, and every mathematical claim was checked by a program that ran.** The six
prototypes in `research/prototypes/` are complete, documented, and verified. One driver reproduces
every number quoted below:

```bash
julia -t 1 --project=. research/prototypes/run_all.jl
```

The recommendations, in order:

| # | Recommendation | Size | Prototype | Status of the gap |
| :-- | :--------------- | :----- | :---------- | :------------------ |
| 1 | A sample seam on the Prior | Medium | `01_scenario_generation.jl` | Real. Nothing in `src/` draws from a Prior. |
| 2 | A Wasserstein ambiguity set | Small | `02_wasserstein_ambiguity.jl` | Real, and **cheaper than it looks**. |
| 3 | A model population | Small | `03_model_population.jl` | The Result type exists. The generator does not. |
| 4 | Calibration against a known truth | Small | `04_simulated_truth_calibration.jl` | Real. Depends on 1. |
| 5 | The breakeven view | Small | `05_breakeven_view.jl` | Original. The solver for it already exists. |
| 6 | A performance summary Result | Very small | `06_performance_summary.jl` | A layering defect, with a line number. |

Two findings are worth stating before anything else.

**Finding A.** The distributionally robust mean-variance problem is **already expressible** in the
library. `L2Regularisation` with the `SOCRiskExpr` algorithm adds $\lambda \lVert w \rVert_2$ to the
objective. Blanchet, Chen and Zhou (2022) prove that this term *is* the exact price of a type-2
Wasserstein ball of radius $\lambda$. The gap is not a cone. The gap is a **name** and a
**calibrated radius**.

**Finding B.** `PopulationPredictionResult` already carries `sort_by_measure` and
`quantile_by_measure`. Its members are cross-validation paths. A population whose members are
*models* needs no new Result type at all. It needs a generator and six statistics.

Both findings shrink the work. Neither was in reports 1 to 7.

---

## 2. Method

I did four things.

1. I read `src/` directly. The library is 109,665 lines across 26 top-level units.
2. I read the seven reports in `research/` and checked each claimed gap against the code.
3. I read the published documentation and papers of the comparison libraries, as of August 2026. I
   did not run them, so section 3 reports what they document, not what they do.
4. I wrote six prototypes and ran them. Section 5 quotes their output.

Nothing in the library was changed. The prototypes are standalone modules. They load no part of
PortfolioOptimisers.jl, so a failure in one is a failure of the prototype and never of the library.

---

## 3. Comparison with other libraries

### 3.1 The comparison table

The table records what each library documents. A blank cell is not a criticism. Most of these
libraries are deliberately narrow.

| Capability | PortfolioOptimisers.jl | skfolio | Riskfolio-Lib | PyPortfolioOpt | cvxportfolio | PortfolioAnalytics (R) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Convex mean-risk optimisers | Yes | Yes | Yes | Yes | Yes | Yes |
| Risk measure count | ~30 families | ~20 | 37 for hierarchical | ~5 | Custom | ~10 |
| Hierarchical (HRP, HERC) | Yes, plus Schur | Yes | Yes | Yes | No | No |
| Nested clustered / stacking | Yes | Yes (NCO) | Yes | No | No | No |
| OWA risk measures | Yes | No | Yes | No | No | No |
| Higher moments (coskew, cokurt) | Yes | No | Yes | No | No | Partial |
| Entropy pooling | Yes | No | Partial | No | No | No |
| Black-Litterman family | Five variants | One | One | One | No | No |
| Uncertainty sets | Four families | Yes | Yes | No | No | No |
| Distributionally robust CVaR | Yes | No | No | No | No | No |
| Graph / network constraints | Yes | No | Partial | No | No | No |
| Time-dependent constraints | Yes | No | No | No | No | No |
| Cross-validation, walk-forward | Yes | Yes | No | No | No | No |
| Combinatorial purged CV | Yes | Yes | No | No | No | No |
| Hyperparameter search | Yes | Yes | No | No | No | No |
| Pipeline as a reified object | Yes | Yes (sklearn) | No | No | No | No |
| **Multi-period optimisation** | **No** | No | No | No | **Yes** | No |
| **Transaction-cost simulation** | Partial | No | No | No | **Yes** | Partial |
| **Scenario generation** | **No** | Partial | No | No | No | Yes |
| **Copulas** | **No** | Yes (vine) | No | No | No | No |
| **Numeric tearsheet** | **No** | Yes | Yes | No | Yes | Yes |
| **Sampling from a fitted model** | **No** | Yes | No | No | No | Yes |
| Discrete allocation | Yes | No | Yes | Yes | No | No |
| Mixed-integer constraints | Yes | No | Yes | Partial | No | Yes |

### 3.2 What the comparison says

PortfolioOptimisers.jl **leads on the estimation side and on the constraint side.** No comparison
library has all of: five Black-Litterman variants, entropy pooling, distributionally robust CVaR,
graph constraints, time-dependent constraints, and combinatorial purged cross-validation. The
capability catalogue is not a marketing document. It is accurate.

The library **trails on the generative side.** Four of the five bold rows in the table are one gap
wearing four hats: nothing in `src/` can produce a return that was not observed. A scenario
generator, a copula, a simulated stress test and a synthetic study are the same absent seam.

Two libraries deserve a specific word.

**cvxportfolio** is the reference for multi-period work, from Boyd and co-authors (2017). Its scope
is a loop over time with transaction and holding costs. It is weak where this library is strong,
because it carries almost no estimation machinery. **I do not recommend competing with it inside
the core**, for the reason in section 7.

**skfolio** is the closest peer in shape, because it also reifies a pipeline and also treats
estimators as configuration. Its paper is Nicolini and co-authors (2025). It ships vine copulas and
synthetic data, which is precisely recommendation 1 of this report. It does **not** ship entropy
pooling, distributionally robust CVaR, or graph constraints. The two libraries are converging from
opposite ends.

---

## 4. Corrections to reports 1 to 7

These gaps are closed already. The evidence is in the code.

| Claimed gap | Reports | Reality |
| :--- | :--- | :--- |
| "Add a backtest engine" | 2, 5, 6 | `WalkForward`, `PredictionCV`, `SearchCrossValidation`, `MultiPeriodPredictionResult`, `PopulationPredictionResult` all exist. |
| "Add a performance tearsheet" | 5, 6 | The **plot** exists. The **numbers** do not. See recommendation 6. |
| "Add attribution" | 2, 4 | `brinson_attribution` is at `src/21_ExpectedReturns.jl:556`. Factor risk contribution is at `src/20_Optimisation/12_FactorRiskContribution.jl`. |
| "Add regime-aware moments" | 2, 7 | `src/08_Moments/36_*` and `37_*` hold them. |
| "Add pre-selection" | 1, 3 | `src/22_Preselection.jl` and the Asset Selector family hold it. |
| "Add graph constraints" | 3, 6 | `src/12_ConstraintGeneration/04_PhylogenyConstraintGeneration.jl`. |
| "Add ESG constraints" | 5, 6 | A linear constraint on a score is already expressible. `UniverseSets` plus `LinearConstraintGeneration` covers it. An ESG type would be sugar. |

Report 4 is the only one of the seven that checked before it recommended. Its corrections are
right, and its central thesis is right. This report agrees with it and adds the two shrink findings
of section 1.

One report is right about a defect. Report 5, Issue 1, says that the DR-CVaR docstring does not
match the implementation. It does not. See section 5.2.

---

## 5. The recommendations

### 5.1 A sample seam on the Prior

#### The gap

A Prior asserts a distribution over returns. Nothing in `src/` draws from it. The only calls to
`StatsBase.sample` are inside cross-validation, at
`src/20_Optimisation/02_CrossValidation/05_MultipleRandomised.jl:347` and
`:11_RandomisedSearchCrossValidation.jl:29`. Both resample *indices*, not returns. The word `copula`
appears nowhere in `src/`.

`Distributions.jl` is already a direct dependency, so this costs no new dependency.

#### Why it is first

One seam unlocks four features that reports 1 to 7 list separately:

1. Scenario generation and stress tests.
2. Synthetic data for tests and documentation.
3. Monte-Carlo spread on the weights, which feeds recommendation 3.
4. Calibration against a known truth, which is recommendation 4.

#### The mathematics

Four generators cover the useful range. Let $X$ be the $T \times N$ historical returns, $\mu$ the
expected returns, and $\Sigma$ the covariance.

**Gaussian.** With $\Sigma = LL'$ and $Z$ an $S \times N$ matrix of standard normal variates,

$$
R = \mathbf{1}\mu' + Z L'
$$

**Student-t.** A multivariate $t$ with scale $S_c$ and $\nu$ degrees of freedom has covariance
$\frac{\nu}{\nu-2} S_c$. To match a target covariance $\Sigma$, set $S_c = \frac{\nu-2}{\nu}\Sigma$.
Then

$$
R_t = \mu + \sqrt{\frac{\nu}{W_t}}\, L' Z_t , \qquad W_t \sim \chi^2_\nu
$$

The shared radial shock $W_t$ is what produces tail dependence. A Gaussian draw has none.

**Gaussian copula with empirical marginals.** By Sklar's theorem (1959),
$F(x) = C(F_1(x_1),\dots,F_N(x_N))$. Draw $Z \sim N(0, P)$, map to uniforms with
$U_j = \Phi(Z_j)$, then map each column through the empirical quantile of that asset. The rank
transform distorts a Pearson correlation, so recover $P$ from the Spearman rank correlation
$\rho_s$ through the Gaussian-copula identity

$$
P_{ij} = 2 \sin\!\left(\frac{\pi \rho_{s,ij}}{6}\right)
$$

This keeps every marginal exactly as observed, skew and fat tails included.

**Stationary bootstrap.** Politis and Romano (1994). Take the next observation with probability
$1-p$, and jump to a fresh uniform start with probability $p = 1/\text{block size}$. Indices wrap.

#### The interface

```julia
simulate_returns(alg::GaussianScenarios,          mu, sigma; n_obs, rng) -> Matrix
simulate_returns(alg::StudentTScenarios,          mu, sigma; n_obs, rng) -> Matrix
simulate_returns(alg::GaussianCopulaScenarios,    X;         n_obs, rng) -> Matrix
simulate_returns(alg::StationaryBootstrapScenarios, X;       n_obs, rng) -> Matrix
```

In the library the entry point becomes `simulate(pr::AbstractPriorResult, alg, n; rng)` and returns
a `ReturnsResult`, so the output feeds straight back into `optimise`. **The generator dispatches on
the Prior Result, not on the Estimator**, because a draw needs the fitted answer and never the
configuration. That places it correctly under the rule in `CONTEXT.md`.

#### Verification

```
gaussian    : max |cov error| / max|sigma| = 4.23e-03
student-t   : max |cov error| / max|sigma| = 5.36e-03,  excess kurtosis = 2.58 (theory 3.00)
copula      : max rank-correlation error = 0.008, 1% marginal quantile error = 4.68e-04
bootstrap   : every simulated row is an observed row = true
```

Each generator reproduces the quantity it claims to reproduce.

#### The known limit

**The Gaussian copula has zero tail dependence.** Two assets simulated that way become independent
in the extreme, whatever their correlation. If the joint tail is the object of study, a t-copula or
a vine copula is the correct next step. See Aas, Czado, Frigessi and Bakken (2009). skfolio ships
vine copulas, and it is the one place where it is clearly ahead.

---

### 5.2 A Wasserstein ambiguity set

#### The gap

The uncertainty-set family has four members: Delta, Normal, Bootstrap and L1. None of them is an
ambiguity set. Wasserstein machinery exists only inside two risk measures, at
`src/19_RiskMeasures/07_ConditionalXatRisk.jl` and its constraint file.

#### The mathematics

Define the ball by the **distance**, not by its power:

$$
\mathcal{B}_\delta(\hat{P}) = \{ P : W_k(P, \hat{P}) \le \delta \}
$$

where $W_k$ uses the ground norm $\lVert \cdot \rVert_q$ on the return space. Duality replaces it
with $\lVert \cdot \rVert_p$ on the weight space, with $1/p + 1/q = 1$. Three robust counterparts
follow from one radius.

**Mean.** By Hölder's inequality and a coupling argument,

$$
\inf_{P \in \mathcal{B}_\delta} \mathbb{E}_P[w'\xi] = \hat{\mu}'w - \delta \lVert w \rVert_p
$$

**CVaR.** In the Rockafellar and Uryasev (2000) form,
$\mathrm{CVaR}_\alpha(\ell) = \min_\tau \{\tau + \alpha^{-1}\mathbb{E}[(\ell - \tau)_+]\}$. For
fixed $\tau$ the integrand is Lipschitz in $\xi$ with modulus $\lVert w \rVert_p / \alpha$. For a
type-1 ball the supremum of a Lipschitz expectation is the empirical value plus radius times
modulus, so

$$
\sup_{P \in \mathcal{B}_\delta} \mathrm{CVaR}^P_\alpha(-w'\xi)
= \mathrm{CVaR}^{\hat{P}}_\alpha(-w'\xi) + \frac{\delta}{\alpha} \lVert w \rVert_p
$$

The $1/\alpha$ factor is the whole content. **A tail functional pays a higher price than a mean for
the same ambiguity, and the multiplier is exactly the reciprocal of the tail probability.** At
$\alpha = 0.05$ the same radius costs twenty times as much.

**Standard deviation.** Blanchet, Chen and Zhou (2022), for a type-2 ball with Euclidean cost:

$$
\sup_{P \in \mathcal{B}_\delta} \sqrt{\mathrm{Var}_P(w'\xi)}
= \sqrt{w'\hat{\Sigma}w} + \delta \lVert w \rVert_2
$$

#### Finding A, stated precisely

The last identity says that a distributionally robust mean-variance problem is the plain problem
plus $\delta \lVert w \rVert_2$. The library builds that term today.
`src/20_Optimisation/09_JuMPConstraints/12_RegularisationConstraints.jl:347` documents it:

> `l2::L2Regularisation{<:Any, <:SOCRiskExpr}`: Introduces `t_l2_i`, constrains `[t_l2_i; w] in SecondOrderCone` so that `t_l2_i >= norm(w, 2)`, and penalises `val * t_l2_i`.

So `L2Reg(; val = delta)` beside a `StandardDeviation` measure **is** the distributionally robust
problem. Note that `SOCRiskExpr` is the **default** algorithm of `L2Regularisation`, at
`src/20_Optimisation/09_JuMPConstraints/12_RegularisationConstraints.jl:272`, so a caller who writes
`L2Reg(; val = 0.01)` already gets the robust form and not the squared one. Nothing tells them that.
Nothing calibrates `val`.

This also explains an empirical result. DeMiguel, Garlappi, Nogales and Uppal (2009) found that
norm-constrained portfolios do better out of sample and could not say why. **A norm penalty is not
an ad-hoc smoother. It is the exact price of distributional ambiguity.**

#### The radius

Two routes, and one of them is honest.

Blanchet, Kang and Murthy (2019) give the parametric rate:

$$
\delta(T) = c \sqrt{\frac{\chi^2_N(\text{confidence})}{T}}
$$

Mohajerin Esfahani and Kuhn (2018) give a distribution-free radius that shrinks at $T^{-1/N}$. That
rate is useless for a realistic universe. At $N = 50$ it barely shrinks at all.

**The honest route is cross-validation.** The library already splits a Pipeline on contiguous time
windows. Treat $\delta$ as one more hyperparameter for `GridSearchCrossValidation`. The closed form
supplies the grid's centre, never its answer.

#### The docstring defect

The DR-CVaR docstring at `src/19_RiskMeasures/07_ConditionalXatRisk.jl:126` states:

$$
\mathrm{DR\text{-}CVaR}_{\alpha,l,r}(x) = \mathrm{CVaR}_\alpha(x) + l \cdot r
$$

The implementation at `src/20_Optimisation/20_RiskMeasureConstraints/07_ConditionalXatRiskConstraints.jl:195`
does something else. It builds the Mohajerin Esfahani and Kuhn conic program for the two-piece
loss $\ell(\xi) = -w'\xi + l[\tau + \alpha^{-1}(-w'\xi-\tau)_+]$, with

```
a1 = -1,              b1 = l
a2 = -1 - l/alpha,    b2 = l(1 - 1/alpha)
```

an infinity-norm cone per scenario, non-negative multipliers `u, v` for the support constraint
$\xi \ge -1$, and the risk expression `radius * lb + mean(s)`. Two statements are therefore wrong in
the docstring:

1. It omits the mean-loss term, so it describes a pure CVaR when the code prices a mean plus $l$
   times a CVaR.
2. It treats $l$ as a constant, when the Lipschitz modulus that multiplies the radius **depends on
   $w$**. That is what the variable `lb` is.

The implementation is correct and it is tighter than the closed form, because the support
constraint is real. **Only the docstring is wrong.** Report 5 was right to raise it.

#### Verification

The prototype builds the measure that attains the worst case, and checks the closed form against it.

```
ground   dual        closed form       attained   abs diff
q=1.0    p=Inf       0.030448415    0.030448415    0.0e+00
q=2.0    p=2.0       0.037286706    0.037286706    0.0e+00
q=Inf    p=1.0       0.059046201    0.059046201    6.9e-18

transport cost of the worst case = 0.002000, radius = 0.002000
robust sd = 0.011152 = empirical sd 0.009328 + radius * ||w||_2 0.001824
```

The formula is attained exactly, for all three ground metrics, and the transport cost of the
attaining measure equals the radius.

---

### 5.3 A model population

#### The gap

`PopulationPredictionResult` is at
`src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl:625`, with `sort_by_measure` and
`quantile_by_measure` beside it. Its members are **cross-validation paths**. A population whose
members are **models** does not exist.

The two answer different questions. A path population asks "would this model have worked?". A model
population asks "does the answer depend on a choice I could not justify?".

`Stacking`, `SubsetResampling` and `NestedClustered` already run many subproblems. Every one of
them *combines* the answers into one weight vector. None reports the *spread*.

#### The mathematics

Let $W$ be the $N \times M$ weight matrix, $\pi$ the member probabilities, and
$\bar{w} = W\pi$ the consensus.

**The ambiguity decomposition.** For any covariance $\Sigma$ and any $\pi$,

$$
\sum_m \pi_m\, w_m' \Sigma w_m
= \bar{w}' \Sigma \bar{w} + \sum_m \pi_m (w_m - \bar{w})' \Sigma (w_m - \bar{w})
$$

The proof is one line. Expand $w_m = \bar{w} + (w_m - \bar{w})$ and note that the cross term
carries $\sum_m \pi_m (w_m - \bar{w}) = 0$. It is the decomposition of Krogh and Vedelsby (1995),
written for a covariance form instead of a squared error.

**This is a proof, not a heuristic, that the consensus is never riskier than the average member.**
The disagreement term is a quadratic form in a positive semi-definite matrix, so it cannot be
negative. The gap it measures is the risk a caller removes by a refusal to pick one model. It is
also the risk a caller accepts by picking one.

**Active share.** $\mathrm{AS}(w_1, w_2) = \tfrac{1}{2}\lVert w_1 - w_2\rVert_1$. The pairwise
matrix is a distance matrix, so the library's clustering machinery reads it directly. **Clustering
the models rather than the assets** answers a question nothing in the library asks today: which of
my twelve estimators are really the same estimator?

**Effective number of models.** The participation ratio of the eigenvalues of the Gram matrix of the
centred weight columns. The upper bound is $M-1$, not $M$, because centring makes the deviations sum
to zero. A population of four members that disagree in four orthogonal directions of equal size
scores three.

#### Verification

```
average member variance = 8.687083e-05
consensus variance      = 8.075928e-05
disagreement            = 6.111555e-06
identity residual       = 7.62e-21  (must be ~0)
consensus is never worse: true
mean pairwise active share = 0.415
effective models = 3.26 of a possible 5
mean Jaccard = 0.80, contested assets = [2, 3, 6, 7]
```

The identity holds to 7.6e-21. The contested list is the output that matters: those are the
decisions the data does not settle.

#### A warning

**A consensus of ten cardinality-constrained portfolios is not a cardinality-constrained
portfolio.** An average of points in a convex set stays in that set, so budget, box and linear
group constraints survive. Integer constraints do not. Report the consensus, and re-solve before
trading it.

---

### 5.4 Calibration against a known truth

#### The gap

Cross-validation asks whether a method would have worked on the history a caller has. It cannot ask
whether the method recovers the right answer when the right answer is known. That needs a simulated
world, and recommendation 1 supplies it.

#### The protocol

1. Fix a true $(\mu, \Sigma)$. This is the world.
2. Draw $T$ observations from it. This is the sample a caller would see.
3. Fit the moment estimator under test.
4. Optimise on the fitted moments.
5. Score the portfolio **under the true moments**, against the portfolio the truth implies.
6. Repeat, and report the distribution.

Step 5 is what history cannot give. It separates the error of the method from the noise of one
realised path.

#### The statistics

With the oracle $w^\star \propto \Sigma^{-1}\mu$ and the estimate $\hat{w}$:

$$
\text{Sharpe loss} = \mathrm{SR}(w^\star) - \mathbb{E}[\mathrm{SR}(\hat{w})], \qquad
\text{risk inflation} = \frac{\mathbb{E}\sqrt{\hat{w}'\Sigma\hat{w}}}{\sqrt{w^{\star\prime}\Sigma w^\star}}
$$

both evaluated under the **true** $\Sigma$ and $\mu$.

#### Verification, and a known result reproduced

```
T =   60 observations   oracle annual Sharpe = 0.975
  estimator                SR(ann)   SR loss  risk infl  AS(orcl)  leverage
  sample                     0.129     0.846      14.16     27.84     55.57
  ledoit-wolf cov            0.151     0.824      20.70     41.49     82.70
  bayes-stein mean           0.189     0.786       6.54     12.96     25.66
  1/N (ignores the fit)      0.686     0.289       0.78      1.48      1.00

T = 1000 observations   oracle annual Sharpe = 0.975
  estimator                SR(ann)   SR loss  risk infl  AS(orcl)  leverage
  sample                     0.332     0.643       9.09     17.65     35.57
  ledoit-wolf cov            0.313     0.662       8.15     15.79     31.80
  bayes-stein mean           0.458     0.517       3.38      6.47     12.87
  1/N (ignores the fit)      0.686     0.289       0.78      1.48      1.00
```

Three results, none of them put in by hand:

1. **1/N beats every optimised rule at every sample size.** That is the DeMiguel, Garlappi and Uppal
   (2009) finding, reproduced from first principles in a few hundred lines.
2. **The unconstrained rule runs 55 times gross leverage at $T = 60$.** That is the error
   maximisation Michaud (1989) named.
3. Mean shrinkage helps more than covariance shrinkage in this world, and the gap grows with $T$.

The protocol carries its own bug detector. **No estimator can beat the oracle.** One that appears to
has leaked the true moments into the fit. The driver asserts it.

---

### 5.5 The breakeven view

**This is the original contribution of this report.** It is not in reports 1 to 7.

#### The question

An optimiser answers "what is the best portfolio?". Nobody can act on that without knowing how
fragile it is. Ask instead:

> What is the smallest change to the market's probabilities that makes my second choice beat my
> first?

#### The construction

Let $c_t$ be the amount by which the incumbent beats the challenger in scenario $t$. For two
portfolios, $c = X(w_A - w_B)$. Define

$$
\delta^\star = \min_q \left\{ \sum_t q_t \log\frac{q_t}{p_t} \;:\; \sum_t q_t c_t \le 0,\;
\sum_t q_t = 1,\; q \ge 0 \right\}
$$

This is the information projection of the prior onto the half-space of measures that reverse the
decision. It is the construction of Csiszár (1975).

Three cases exhaust it:

1. $\mathbb{E}_p[c] \le 0$. The prior already prefers the challenger. $\delta^\star = 0$.
2. $\min_t c_t > 0$. The incumbent wins in **every** scenario, so no reweighting reverses it and
   $\delta^\star = \infty$. The decision is not fragile with respect to probabilities. It may still
   be fragile with respect to the scenario set, which is a Wasserstein question and not an entropy
   one.
3. Otherwise the constraint binds, so the inequality becomes the equality $\mathbb{E}_q[c] = 0$, and
   the problem is **exactly a one-view entropy pooling program**.

Case 3 is why this is cheap. The dual is one-dimensional:

$$
q_t(\lambda) = \frac{p_t e^{-\lambda c_t}}{Z(\lambda)}, \qquad
L(\lambda) = \log Z(\lambda), \qquad
\delta^\star = -L(\lambda^\star)
$$

**One scalar convex problem answers a question about the stability of the whole optimisation.** The
library already solves the forward version of it in `src/13_Prior/10_EntropyPoolingPrior.jl`.

#### Making a nat readable

Relative entropy is not a unit a committee understands. Convert it with the Hansen and Sargent
(2008) detection-error probability:

$$
p_{\text{detect}} \approx \Phi\!\left(-\sqrt{\tfrac{n D}{2}}\right)
$$

This is the probability that the optimal statistical test, given $n$ observations, **fails to tell
the two worlds apart**.

#### Verification

```
decision                         edge delta* (nats)     ENS%  p_detect
tangency vs 1/N              1.45e-03    2.109e-03     99.8     0.073
1/N vs min-variance          4.69e-05    1.217e-04    100.0     0.364
tangency vs min-variance     1.50e-03    2.147e-03     99.8     0.071
```

Read the middle row. The preference for 1/N over minimum variance would survive a probability
change that **a competent statistician with the same 2000 observations would fail to detect 36 per
cent of the time**. That ranking is not a conclusion. It is a coin toss with extra steps. The first
and third rows, at 7 per cent, are supported.

An independent scalar root-find agrees with the solver to 1.0e-11, and the posterior satisfies its
constraint to 4.1e-12.

#### A second use of the same solver

The same function ranks *views* by how bold they are.

```
entropy cost of a stated view on asset 1's mean:
  shift +0.00000 -> D = 0.000000 nats, ENS = 100.0%, p_detect = 0.500
  shift +0.00010 -> D = 0.000037 nats, ENS = 100.0%, p_detect = 0.424
  shift +0.00040 -> D = 0.000591 nats, ENS =  99.9%, p_detect = 0.221
  shift +0.00080 -> D = 0.002363 nats, ENS =  99.8%, p_detect = 0.062
```

A caller with five candidate views has no way today to say which is the bold claim. This says it in
one unit, and it never asks the caller to guess a confidence.

---

### 5.6 A performance summary Result

#### The defect

The seven headline statistics are computed at
`ext/PortfolioOptimisersPlotsExt.jl:1977-1992`, **inside the plot function**, and are never
returned. A caller without `StatsPlots` cannot get them. A caller with `StatsPlots` gets a bar
chart when they wanted a table.

That inverts the library's own layering. Everywhere else a plot renders a Result that another
function computed. This is the one place a plot computes one.

#### The fix

A `PerformanceSummaryResult` and a `performance_summary` function in `src/`. The plot extension then
keeps only the drawing. The prototype reproduces the extension's five existing formulas exactly, and
adds six fields, of which one matters most.

**Add the standard error of the Sharpe ratio.** Bailey and López de Prado (2012):

$$
\mathrm{se}(\widehat{\mathrm{SR}}) = \sqrt{\frac{1 - g_1 \mathrm{SR} + \frac{g_2}{4}\mathrm{SR}^2}{T-1}}
$$

with $g_1$ the skewness and $g_2$ the excess kurtosis. A non-normal series makes a Sharpe ratio less
precise than the naive expression suggests, and negative skew makes it worse. That is the case that
matters for a real portfolio.

#### Verification

```
Sharpe                 0.0080  +/- 0.3163  <- the plot cannot show this
```

The Sharpe ratio is 0.03 standard errors from zero. A bar chart of the point estimate invites a
reader to rank this strategy against another. The Result does not.

The prototype matches the extension on `ann_return`, `ann_vol`, `sharpe`, `sortino` and `max_dd`,
and the standard error reduces to the naive form on a normal sample, ratio 1.0000 over 200,000
draws.

---

## 6. Sequencing

| Phase | Work | Depends on |
| :-- | :-- | :-- |
| 1 | Recommendation 6. Move the statistics into `src/`. | Nothing |
| 1 | Fix the DR-CVaR docstring (section 5.2). | Nothing |
| 2 | Recommendation 2. The ambiguity set and the radius. | Nothing |
| 3 | Recommendation 1. The sample seam. | Nothing |
| 4 | Recommendation 3. The model population. | Nothing, but better after 1 |
| 5 | Recommendation 4. Calibration. | 1 |
| 6 | Recommendation 5. The breakeven view. | Nothing |

Phase 1 is a day. Phases 2 and 6 are small because the solvers exist. Phase 3 is the only medium
one.

---

## 7. What I would not add

I would refuse these, whatever reports 1 to 7 say.

| Item | Reason |
| :-- | :-- |
| Reinforcement learning environments | A research topic, not a library feature. It needs an environment, a reward, and a training loop, none of which belong beside an estimator. |
| LLM-generated Black-Litterman views | A network call inside a Prior breaks reproducibility, the property the whole Pipeline exists to protect. |
| Quantum and quantum-inspired solvers | No solver available today beats Clarabel on these problems. JuMP would carry it if one did. |
| Rough volatility and fractional models | A univariate research topic. It has no seam in a cross-sectional allocation library. |
| Broker and paper-trading connectivity | Belongs in the `AlgoTrader.jl` companion sketched in `research/NOTRACK_AlgoTrader.jl_README.md`. |
| GPU acceleration | The bottleneck is the conic solver, which is not the library's code. Optimise the estimators only after a profile says so. |
| **Multi-period optimisation** | See below. |

**Multi-period deserves its own paragraph, because it is the one genuinely valuable item on this
list.** cvxportfolio does it well, and Boyd and co-authors (2017) is the right reference. I would
still keep it out of the core. The reason is structural: `w` is one vector in every constraint file
in `src/20_Optimisation/09_JuMPConstraints/`. A horizon index would break all thirteen of them, plus
every risk-measure constraint file. That is not a feature. That is a rewrite. The design note in
`research/NOTRACK_AlgoTrader.jl_README.md` already puts the loop in the companion package, and
treats `optimise` as a pure function called inside it. **I agree with that split**, and I think it
is the right long-term shape.

---

## 8. One general rule

The library is 109,665 lines in `src/`, with 229 abstract types. The risk to it now is surface area,
not capability.

**Prefer an addition that is a seam over an addition that is a leaf.** A seam is one verb that many
types answer: `simulate`, `port_opt_view`, `factory`. A leaf is one more estimator beside the forty
that exist. Five of the six recommendations in this report are seams. That is deliberate.

---

## 9. Sources

### Libraries

Named in prose, per the repository convention. Versions are those documented in August 2026.

- **skfolio**, Python, built on scikit-learn. Nicolini, C. and co-authors (2025). *skfolio:
  Portfolio Optimization in Python*. arXiv:2507.04176.
- **Riskfolio-Lib**, Python, version 7.3. Documents 37 risk measures for hierarchical optimisation
  and an OWA portfolio module.
- **PyPortfolioOpt**, Python.
- **cvxportfolio**, Python, version 1.5. The reference implementation of Boyd and co-authors (2017).
- **PortfolioAnalytics**, R.

### Papers

1. Aas, K., Czado, C., Frigessi, A. and Bakken, H. (2009). Pair-copula constructions of multiple
   dependence. *Insurance: Mathematics and Economics* 44(2), 182–198.
2. Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of
   Risk* 15(2), 3–44.
3. Blanchet, J., Chen, L. and Zhou, X. Y. (2022). Distributionally robust mean-variance portfolio
   selection with Wasserstein distances. *Management Science* 68(9), 6382–6410. arXiv:1802.04885.
4. Blanchet, J., Kang, Y. and Murthy, K. (2019). Robust Wasserstein profile inference and
   applications to machine learning. *Journal of Applied Probability* 56(3), 830–857.
   arXiv:1610.05627.
5. Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P. and Speth, J. (2017).
   Multi-period trading via convex optimization. *Foundations and Trends in Optimization* 3(1),
   1–76. arXiv:1705.00109.
6. Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues.
   *Quantitative Finance* 1(2), 223–236.
7. Cremers, K. J. M. and Petajisto, A. (2009). How active is your fund manager? *Review of Financial
   Studies* 22(9), 3329–3365.
8. Csiszár, I. (1975). I-divergence geometry of probability distributions and minimization problems.
   *Annals of Probability* 3(1), 146–158.
9. DeMiguel, V., Garlappi, L., Nogales, F. J. and Uppal, R. (2009). A generalized approach to
   portfolio optimization: improving performance by constraining portfolio norms. *Management
   Science* 55(5), 798–812.
10. DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive diversification: how
    inefficient is the 1/N portfolio strategy? *Review of Financial Studies* 22(5), 1915–1953.
11. Gao, R. and Kleywegt, A. J. (2023). Distributionally robust stochastic optimization with
    Wasserstein distance. *Mathematics of Operations Research* 48(2), 603–655.
12. Hansen, L. P. and Sargent, T. J. (2008). *Robustness*. Princeton University Press.
13. Jorion, P. (1986). Bayes-Stein estimation for portfolio analysis. *Journal of Financial and
    Quantitative Analysis* 21(3), 279–292.
14. Kan, R. and Zhou, G. (2007). Optimal portfolio choice with parameter uncertainty. *Journal of
    Financial and Quantitative Analysis* 42(3), 621–656.
15. Krogh, A. and Vedelsby, J. (1995). Neural network ensembles, cross validation, and active
    learning. *Advances in Neural Information Processing Systems* 7, 231–238.
16. Kruskal, W. H. (1958). Ordinal measures of association. *Journal of the American Statistical
    Association* 53(284), 814–861.
17. Kullback, S. and Leibler, R. A. (1951). On information and sufficiency. *Annals of Mathematical
    Statistics* 22(1), 79–86.
18. Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance
    matrices. *Journal of Multivariate Analysis* 88(2), 365–411.
19. Martin, P. G. and McCann, B. B. (1989). *The Investor's Guide to Fidelity Funds*. Wiley.
20. Meucci, A. (2008). Fully flexible views: theory and practice. *Risk* 21(10), 97–102.
21. Meucci, A. (2010). Historical scenarios with fully flexible probabilities. *GARP Risk
    Professional*, December, 40–43.
22. Michaud, R. O. (1989). The Markowitz optimization enigma: is optimized optimal? *Financial
    Analysts Journal* 45(1), 31–42.
23. Michaud, R. O. (1998). *Efficient Asset Management*. Harvard Business School Press.
24. Mohajerin Esfahani, P. and Kuhn, D. (2018). Data-driven distributionally robust optimization
    using the Wasserstein metric. *Mathematical Programming* 171(1), 115–166.
25. Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American
    Statistical Association* 89(428), 1303–1313.
26. Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal
    of Risk* 2(3), 21–41.
27. Sharpe, W. F. (1994). The Sharpe ratio. *Journal of Portfolio Management* 21(1), 49–58.
28. Sklar, A. (1959). Fonctions de répartition à n dimensions et leurs marges. *Publications de
    l'Institut de Statistique de l'Université de Paris* 8, 229–231.
29. Sortino, F. A. and Price, L. N. (1994). Performance measurement in a downside risk framework.
    *Journal of Investing* 3(3), 59–64.
30. Young, T. W. (1991). Calmar ratio: a smoother tool. *Futures* 20(1), 40.

---

## 10. The prototypes

| File | Lines | Verified claim |
| :-- | --: | :-- |
| `research/prototypes/01_scenario_generation.jl` | 424 | Each generator reproduces its target moments, rank correlation and marginals. |
| `research/prototypes/02_wasserstein_ambiguity.jl` | 526 | The closed form is attained exactly by the worst-case measure, for three ground metrics. |
| `research/prototypes/03_model_population.jl` | 465 | The ambiguity decomposition holds to 7.6e-21. |
| `research/prototypes/04_simulated_truth_calibration.jl` | 354 | Reproduces DeMiguel–Garlappi–Uppal (2009). No estimator beats the oracle. |
| `research/prototypes/05_breakeven_view.jl` | 434 | $D = -L(\lambda^\star)$, and the solver agrees with an independent root-find to 1.0e-11. |
| `research/prototypes/06_performance_summary.jl` | 287 | Matches the plot extension's five formulas. The Sharpe standard error reduces to the naive form on a normal sample. |
| `research/prototypes/run_all.jl` | 241 | Reproduces every number in this report. |

Each file states its own notation, defines every variable, and documents every function with its
arguments, returns, mathematics and known limits. None of them loads PortfolioOptimisers.jl, so
each can be adapted into `src/` one at a time.
