# 957 — How the reference implementation ingests a gapped panel, and what its estimation universe is

Research ticket #957 of wayfinder map #955. Written 2026-09-09.

Every number below comes from running the reference implementation at its latest tag in a
dedicated Python environment, on the fixtures in section 7. Its files are cited by their path
relative to its `src/` directory. The reference is never named, per the map's Notes.

---

## Summary

- **There is an ingestion verb, and it emits no mask.** `preprocessing/_returns.py:15`,
  `prices_to_returns`, turns a price table into a returns table and returns a DataFrame alone.
  Its default settings fill every gap forward and truncate the panel to the latest inception,
  so a delisted asset becomes a flat zero-return column and nothing downstream can tell. The
  only setting that preserves gap positions is `fill_nan=False, drop_inceptions_nan=False`,
  and under it a price gap of `k` observations becomes a return gap of `k + 1`, because the
  first priced observation after the gap has no previous price. Section 1.
- **The active mask is never derived from prices.** The panel defaults it to all `True`
  (`containers/_asset_panel/_panel.py:298`), every estimator treats a missing mask as all
  active (`utils/tools.py:512`), and the one helper, `align_active_mask_to`
  (`_panel.py:1067`), trims **leading** active entries only. The reference's direction is
  mask to data: a caller supplies the mask, and the panel blanks every field outside it and
  refuses a finite value there (`containers/_asset_panel/_utils.py:116`). The span rule's
  trailing half, the delisting, has no counterpart. Section 2.
- **The estimation universe is `estimation_mask & isfinite(returns)`, formed at the point of
  use.** The stored `estimation_mask` is a caller-declared subset of the active mask, all
  active by default, and every consumer intersects it with finiteness before it reads a
  cross-section (`prior/_characteristics_factor_model.py:1156`, `utils/stats.py:1611`,
  `linear_model/_cross_sectional/_base.py:216`). An asset inside a suspension is `True` in
  the stored mask and out of that observation's regression: its regression and benchmark
  weights are zero, and the factor return is bit-identical to the fit that declares the cell
  out of the estimation mask, and to the fit that declares it inactive. **The map's decision
  3 agrees with the reference, and nothing reopens.** Section 3.
- **The factor axis has no gap vocabulary.** `prices_to_returns` forward-fills a factor gap
  and drops the row on a late-listed factor, and a `NaN` factor return that survives to the
  time-series factor prior is refused by the plain mean estimator. Section 4.
- **The policy surface is five knobs.** A row filter on the missing share, an unlimited
  forward fill of prices, an inception cut, an endpoint survivorship filter with no
  threshold, and a per-estimator warm-up. No policy reads the mask except the panel's own
  fills, which stop at the active boundary. Section 5.
- **One question surfaced for the map.** The return the first priced observation after a
  Held Gap carries is a decision the reference makes by omission: its no-fill route discards
  the catch-up return across the gap, and its fill route books it one observation late on top
  of `k` zeros. Section 6.

---

## 1. The ingestion verb

`prices_to_returns(X, y=None, log_returns=False, nan_threshold=1, join="outer",
drop_inceptions_nan=True, fill_nan=True)`, `preprocessing/_returns.py:15-135`. Its body is
five steps in this order:

| Step | Line | Rule |
| --- | --- | --- |
| row filter | `:107-114` | drop an observation whose count of missing **asset** prices strictly exceeds `n_assets * nan_threshold`; the default `1` never fires |
| fill | `:117-118` | forward-fill every column, unlimited, when `fill_nan` |
| inception cut | `:121` | drop a row where **any** value is missing when `drop_inceptions_nan`, else drop a row where **all** are missing |
| column cut | `:123` | drop a column whose every value is missing |
| returns | `:126-128` | `pct_change(fill_method=None)`, drop the first row, `log1p` when asked |

It returns the returns DataFrame, or the pair `(X, y)` when a factor table is joined. It
carries no mask out, and no other verb in the reference reads a price.

### 1.1 What each setting does with a partly absent column

The fixture is six price columns over eight business days from 2024-01-01: `clean`, `leading`
(missing on the first two days), `trailing` (missing on the last three), `interior` (missing
on day three), `interior2` (missing on days three and four), and `allnan`.

**Defaults.** The output is **empty**: shape `(0, 0)`. The all-missing column survives the
forward fill, the inception cut then drops every row because that column is missing on all
of them, and the column cut runs one step too late. Without that column the output is five
rows: the two rows before `leading` is priced are gone, the gap in `interior` is a `0.0`
return followed by a catch-up of `0.0198`, and `trailing` is `0.0` on every day after its
last price. The delisting is invisible.

| Date | clean | leading | trailing | interior | interior2 |
| --- | --- | --- | --- | --- | --- |
| 01-04 | 0.0098 | 0.0098 | 0.0098 | 0.0198 | 0.0000 |
| 01-05 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | 0.0297 |
| 01-08 | 0.0096 | 0.0096 | 0.0000 | 0.0096 | 0.0096 |
| 01-09 | 0.0095 | 0.0095 | 0.0000 | 0.0095 | 0.0095 |
| 01-10 | 0.0094 | 0.0094 | 0.0000 | 0.0094 | 0.0094 |

**`fill_nan=False, drop_inceptions_nan=False`.** The only setting that keeps every gap where
it is. Seven rows, five columns, and the `NaN` pattern is the price pattern shifted by one:

| Date | clean | leading | trailing | interior | interior2 |
| --- | --- | --- | --- | --- | --- |
| 01-02 | 0.0100 | NaN | 0.0100 | 0.0100 | 0.0100 |
| 01-03 | 0.0099 | NaN | 0.0099 | NaN | NaN |
| 01-04 | 0.0098 | 0.0098 | 0.0098 | NaN | NaN |
| 01-05 | 0.0097 | 0.0097 | 0.0097 | 0.0097 | NaN |
| 01-08 | 0.0096 | 0.0096 | NaN | 0.0096 | 0.0096 |
| 01-09 | 0.0095 | 0.0095 | NaN | 0.0095 | 0.0095 |
| 01-10 | 0.0094 | 0.0094 | NaN | 0.0094 | 0.0094 |

A one-day price gap in `interior` is a **two-day** return gap, on 01-03 and 01-04: the
second day has a price but no previous price, and `pct_change(fill_method=None)` gives
`NaN`. The catch-up return over the gap, `103 / 101 - 1`, appears nowhere. A leading gap of
`k` prices is `k` return gaps, and a trailing gap of `k` prices is `k` return gaps, because
each is bounded by the panel edge on one side.

**`fill_nan=True, drop_inceptions_nan=False`** keeps the leading `NaN`s and fills the rest:
`interior` is `0.0` then `0.0198`, `trailing` is `0.0` forever.

**`fill_nan=False, drop_inceptions_nan=True`** is empty on this fixture for the same reason
the defaults are, and on a fixture without the all-missing column it drops every row that
holds any gap, leading, interior or trailing.

**`nan_threshold=0.5`** drops no row here: the rule is a strict `>` on the count of missing
asset prices, and no row of the fixture exceeds half.

**A zero price.** The reference's own incomplete-dataset example encodes a default as a `0.0`
price followed by `NaN`s. The verb returns `-1.0` on the day of the zero and `NaN` after it,
under every setting, because a forward fill propagates the zero and a return from zero is
undefined. The `-1.0` is the reference's encoding of a total loss, and a delisting at a
non-zero last price gets no such return.

## 2. Where the active mask comes from

**Nowhere in the data path.** The sites, in the order a caller meets them:

| Site | Path | What it does with no mask |
| --- | --- | --- |
| the mask validator | `utils/tools.py:489-513` | returns `None` for `None` |
| the exponentially weighted estimators | `moments/expected_returns/_ew_mu.py:276-283` | treat every asset as active; a `NaN` return **freezes** the estimate, and a trailing gap is read as an endless holiday |
| the panel | `containers/_asset_panel/_panel.py:298-305` | both masks default to all `True` |
| the guide | `docs/user_guide/data_representation.rst:201-224` | states the two-mask convention and names no derivation |

Measured on `EWMu(half_life=2)` over `clean` and `trailing`: with no mask, `mu_` for
`trailing` is the last value before the gap, its observation count is `4`, and it is still
flagged active. With `active_mask = isfinite(returns)` it is `NaN`. `EWCovariance` behaves the
same way, entry for entry. That is #670's freeze-versus-reset, and the reference's docstring
at `_ew_mu.py:61` says in as many words that a trailing `NaN` is ambiguous without the mask.

**The one helper derives half the span rule.** `AssetPanel.align_active_mask_to(fields)`,
`_panel.py:1067-1118`, removes an asset's **leading** active entries until the named field
is finite, and removes all of them when the field is never finite. On the seven-row returns
panel it removed two entries, the two leading rows of `leading`, and left `trailing` active
through its three-row gap. Missing values after the first valid one are left alone by
design (`:1073-1075`). So the reference can infer a listing date from the data and cannot
infer a delisting; a caller holding only prices has no way to mark one.

**The direction is mask to data.** The synthetic panel draws listing and delisting windows,
forms the mask from them (`datasets/_synthetic.py:495-509`), and then blanks every field
outside it (`:604-611`). The panel enforces that direction as an invariant:

- a finite value where `active_mask` is `False` raises
  (`containers/_asset_panel/_utils.py:116-178`; measured: `ValueError`, "Field 'returns' has
  1 finite value(s) where active_mask is False");
- an observation with no active asset raises (`_utils.py:525-531`; measured);
- a view `panel[a:b]` slices both masks with the fields (measured on `panel[2:5]`).

The panel's own fills respect the mask: `ffill` and `bfill` (`_panel.py:999-1066`,
`_utils.py:326-349`) fill only active cells, with an optional `limit`. Measured: the interior
gaps fill, and so does the trailing gap, because nothing marked it inactive.

## 3. The estimation universe

### 3.1 The stored mask

`estimation_mask`, `_panel.py:106-112`, is a boolean of the panel's shape, `None` by default,
which the constructor sets to all `True` (`:302-305`). It is documented as "use this asset
in this estimator statistic" (`data_representation.rst:228-235`) and, in the factor-model
guide, as the liquidity-and-quality screen that fits cross-sectional statistics, factor-return
regressions, benchmark and regression weights, alpha estimators and regime statistics
(`docs/user_guide/factor_models.rst:618-623`). A `True` outside the active mask is **coerced**
to `False` without a warning (`_panel.py:1410-1416`, measured), and the invariant is
re-enforced every time the masks are edited (`:417-441`).

### 3.2 Where it is consumed, and with what

Every reader intersects it with finiteness before it forms a cross-section:

| Reader | Path | The effective mask |
| --- | --- | --- |
| the regression-eligible set | `prior/_characteristics_factor_model.py:1156-1160` | `isfinite(returns) & estimation_mask & all(isfinite(exposures))` |
| the all-missing-observation guard | `:1649-1655` | `isfinite(returns) & estimation_mask`, must hold for one asset per row |
| the benchmark weights | `:1997-2001` | `isfinite(returns) & estimation_mask` |
| the market return | `utils/stats.py:1611` | `estimation_mask & isfinite(returns) & isfinite(weights)` |
| the cross-sectional regressor | `linear_model/_cross_sectional/_base.py:205-223` | a positive weight on a non-finite pair **raises**; the caller must zero it first |
| the cross-sectional scalers | `preprocessing/_transformer/_cross_sectional/_utils.py:116-131` | `isfinite(X) & (weights > 0)`; a `NaN` is preserved and skipped |

So the reference never stores the finiteness half. The effective estimation universe at an
observation is `estimation_mask & isfinite(returns)`, where the stored mask is a declared
subset of the active mask. Set the declared subset to "all active" and this is exactly the
map's `emsk = amsk .& isfinite.(returns)`; declare a narrower one and it is the map's
"a caller declaration overrides it".

### 3.3 An asset inside a suspension, measured at the top-level verb

The fixture is fourteen observations of six assets with one passthrough exposure field,
`min_regression_assets=2`, a plain empirical factor prior, and short-half-life idiosyncratic
estimators. Asset 2's return at observation 8 is set to `NaN` with prices on both sides,
and both masks are left all `True`. Three fits:

| Fit | Factor return at the row | Regression weight of asset 2 at the row | Benchmark weight | Idio return at the cell |
| --- | --- | --- | --- | --- |
| dense | `-0.006550` | `1.0` | `1.0` | finite |
| `NaN` inside both masks | `-0.008378` | `0.0` | `0.0` | `NaN` |
| `estimation_mask[8, 2] = False` | `-0.008378` | `0.0` | `0.0` | `NaN` |
| `active_mask[8, 2] = False`, fields blanked | `-0.008378` | `0.0` | `0.0` | `NaN` |

The three gapped fits agree on the cross-section bit for bit. So the direct answer: **an asset
inside a suspension is in the stored estimation mask and out of that observation's
regression**, which is what the map's decision 3 produces. The two masks part only in the
state-carrying estimators: the asset's expected return is `0.000635` under the `NaN` and
`0.000670` under the one-row inactive, because the exponentially weighted idiosyncratic
estimators freeze on the first and reset on the second (#670). The cross-section does not
see the difference; the moments do.

The `active_mask=False` fit also had to blank the asset's `market_cap` and exposure at that
cell by hand, or the panel refused it (section 2). A one-row inactivity is therefore a
mutation of every field, not a flag, in the reference.

### 3.4 The one thing the map's rule does not name

The reference's `estimation_mask` is a **declared screen** independent of finiteness, and a
caller uses it to exclude assets that are active, priced and undesirable for estimation, for
example illiquid names. The map's rule reaches this through the override clause. What the
rule does not do is keep a *record* of a screen separate from the finiteness intersection:
the reference stores the screen and computes the intersection at use, the map stores the
intersection. A consumer that wants to know "was this asset screened, or merely unpriced" can
tell in the reference and cannot in a stored intersection. Whether the library needs that
distinction is the decomposition ticket's to weigh, not this one's; the moments and the
cross-section measured here never asked.

## 4. A gapped factor or benchmark series

`prices_to_returns` joins `y` to `X` with an outer join (`_returns.py:101`), counts missing
prices over `X`'s columns only (`:106-112`), and then treats the joined table as one:

| Setting | A factor missing on one day | A factor listed three days late |
| --- | --- | --- |
| defaults | filled forward: factor return `0.0` then a catch-up of `0.0392`; asset rows untouched | the first three asset rows **dropped** by the inception cut |
| `fill_nan=False, drop_inceptions_nan=False` | two `NaN` factor returns, as in section 1.1 | three `NaN` factor returns; asset rows kept |

A `NaN` factor return that reaches the time-series factor prior is refused:
`TimeSeriesFactorModel().fit(X, factors=y)` raises `ValueError: Input X contains NaN` from the
plain mean estimator (measured). No estimator on the factor axis takes a mask, and the
characteristics factor model's macro-sensitivity descriptors take their factor series through
the same finite-or-refuse path. The benchmark side of the portfolio layer was measured on
#671 and is unchanged. Recorded for a successor map; it changes nothing on this one.

## 5. Filtering and imputation

Everything the reference applies before it forms a moment, in the order a panel meets it:

| Policy | Path | Rule and default |
| --- | --- | --- |
| row filter | `preprocessing/_returns.py:107-114` | drop an observation whose count of missing asset prices `> n_assets * nan_threshold`; default `1.0`, never fires |
| forward fill of prices | `_returns.py:117-118` | unlimited, every column, **on by default** |
| inception cut | `_returns.py:121` | drop every row where any column is missing after the fill; **on by default**, so the panel starts at the latest listing |
| all-missing column cut | `_returns.py:123` | unconditional, and ordered after the row cut, which is what empties a panel with one such column |
| panel fill | `_panel.py:999-1066` | `ffill`/`bfill` with a `limit`, inside the active mask only, opt-in |
| listing alignment | `_panel.py:1067` | trim leading active entries to the first finite value, opt-in |
| survivorship filter | `pre_selection/_select_complete.py:101-109` | default keeps a column iff its **first and last** rows are finite, no threshold; `drop_assets_with_internal_nan=True` drops on any `NaN`. Measured again on four columns: default keeps `clean` and `interior`, strict keeps `clean` |
| warm-up, moments | `moments/expected_returns/_ew_mu.py:308-309` | `min_observations` defaults to `max(1, int(half_life))`; output `NaN` until reached. Measured: half-life 20, five observations, `mu_` is `NaN` |
| coverage floor, factor model | `prior/_characteristics_factor_model.py:543-548`, `:1797-1804` | `min_regression_assets` defaults to `max(2 * n_factors, 30)` eligible assets per observation, and a shortfall **raises** |

The plain moment estimators refuse a `NaN` outright (`EmpiricalMu` raised on the gapped
fixture; the full census is #672's). So the reference offers exactly two consistent routes:
fill everything at ingestion and run plain estimators, or fill nothing and run the
exponentially weighted family under a caller-supplied mask. The survivorship filter sits
between them and, as #671 found, guarantees neither route a finite input.

## 6. What this changes on the map

1. **Decisions 2 and 3 stand.** The span rule adds the delisting half the reference lacks and
   keeps the listing half it has; the estimation-mask rule is the reference's effective rule
   with the declared screen reachable by override. Nothing measured here disagrees with
   either.
2. **The decomposition ticket inherits one fact.** The reference's ingestion is two halves
   that never meet: a verb that emits returns and no mask, and a container that takes a mask
   and no prices. No verb goes from a price table to a masked panel. The destination's
   "a caller holding only a gapped price table" is a route the reference does not have.
3. **One decision surfaced.** The value the first priced observation after a Held Gap
   carries. The reference's no-fill route gives `NaN`, and so a `k`-observation price gap is a
   `k + 1` Held Gap that discards the catch-up return across it. Its fill route gives `k`
   zeros and books the catch-up on the first priced day. A third answer, the catch-up return
   on the first priced day and `NaN` inside the gap, is the one that conserves wealth across
   the suspension, and no route of the reference produces it. This is a value rule the build
   ticket must implement, so it is sharp enough to ticket now.

## 7. The fixtures

**Section 1 and 4.** Prices on the eight business days from 2024-01-01. `clean` is
`100, 101, …, 107`. `leading` is the same with the first two missing. `trailing` is the same
with the last three missing. `interior` is missing on the third day, `interior2` on the third
and fourth. `allnan` is missing throughout. The factor is `50, 51, …, 57` with the third day
missing, or the first three missing for the late listing.

**Section 2.** The seven-row returns table of section 1.1 without `allnan`, loaded into a
panel with no masks.

**Section 3.** Seed `7`. Exposures `beta` are normal with mean `1` and standard deviation
`0.3`, shape `(14, 6)`. The factor is normal with standard deviation `0.02`. Returns are
`beta * f + noise` with noise standard deviation `0.005`. Market capitalisation is all ones.
The model is one passthrough factor of family `market`, `min_regression_assets = 2`, an
empirical factor prior, and exponentially weighted idiosyncratic variance and correlation with
half-life `3` and `min_observations = 1`. The exposure lag is `1`, so observation 8's return
row is regression row 7.

The drivers import the reference's public modules and are not tracked, so that this file
names nothing.
