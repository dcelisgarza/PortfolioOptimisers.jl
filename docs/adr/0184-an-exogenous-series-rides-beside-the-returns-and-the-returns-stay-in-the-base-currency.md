---
status: accepted
---

# An Exogenous Series rides beside the returns, and the returns stay in the base currency

## Context

A universe held across currencies earns an exchange rate return that no style or category factor
explains. `CrossSectionalFactorPrior` gets Currency Factors in #927: a factor for each currency,
with a one-hot exposure on the assets that hold the currency. Its return is the observed Currency
Excess Return of the currency, and the fit does not estimate it. The Cross-Sectional Regression
runs on the other exposures alone. The currency factors are appended to the reduced factor
returns and to the loadings after the regression, and the Factor Family Basis passes them
through. #926 settles two questions before that build: where the
`observations × currencies` matrix of Currency Excess Returns rides, and what the asset returns
mean when Currency Factors are present.

Four facts constrain the answer.

1. **`rd.X` is the only asset return series.** The Asset Panel has no returns field. The
   regression reads `rd.X`, the prior result keeps it in `o_X`, and a cross-validation fold reads
   the realised returns of its test rows from it. A contract that reads `rd.X` as local-currency
   returns under Currency Factors reaches every one of these consumers. The realised return of a
   fold is then in a currency that the investor does not hold.
2. **The Asset Panel has only asset axes.** Its constructor refuses a Panel Field whose second
   axis is not the asset axis, so an `observations × currencies` matrix has no place in it.
3. **`rd.F` is read whole as time-series factors.** `FactorPrior`, `regression`, the uncertainty
   sets, the factor risk contribution, the online step and the pipeline's time-series factor axis
   all read every column of `F`. A currency column in `F` becomes a factor for each of them.
4. **Nothing routes a keyword to a prior inside a fold.** `optimise(opt, port_opt_view(rd, rows,
   cols))` passes `rd` alone. A keyword argument of `prior` reaches a direct call only.

`EWMacroSensitivity` has the same problem. It needs an exchange rate or rate-of-interest series
as its keyword `ref`. `composite_score` calls `descriptor(de, rd)` with no keywords, so inside
the prior the descriptor always throws.

The identity that splits a base-currency return is exact only in log returns. With simple
returns, `R_base = (1 + R_local)(1 + R_FX) - 1`, so the base-currency excess return is

```text
R_excess_base_i = R_excess_local_i + (R_FX_C + r_cash_C - r_cash_base) + R_local_i * R_FX_C
```

The cross term `R_local_i * R_FX_C` depends on the asset, so no series with one column for each
currency can hold it. With log returns, `ln(1 + R_base) = ln(1 + R_local) + ln(1 + R_FX)`, and
the split has no cross term.

## Decision

1. **The asset returns stay in the base currency, with or without Currency Factors.** The
   contract of `rd.X` does not change. By default the prior derives the local excess returns as
   `X - Z_ccy * r_ccy`. `Z_ccy` is the one-hot currency exposure of each asset at each
   observation. A field on the prior can name a `NumericPanelField` of local returns instead.
   The regression then reads that field, and the caller's own measure of the local return enters
   the fit unchanged. Two cases need the named field. The first is an asset with no currency label
   at an observation: the derived path gives it a `NaN` local return, and the asset leaves that
   regression. The second is a caller whose local returns are not `X - Z_ccy * r_ccy`, which
   includes every simple-return panel because of the cross term.
2. **A new block, the Exogenous Series, rides on `ReturnsResult` and on `PricesResult`.** The
   block is a pair of fields, `ne` for the names and `E` for the `observations × series` matrix,
   beside `nx/X`, `nf/F` and `nb/B`. It holds named series over the observation axis that belong
   to no asset. A consumer reads a column only by its name, so no model takes the block whole as
   its factors.
3. **Every observation slice cuts `E` by rows, and an asset view passes it through.** Both
   arities of `port_opt_view`, the validation of the constructor and every site that rebuilds a
   `ReturnsResult` carry the block. The asset-only view leaves it whole, as it leaves `F`.
4. **`prices_to_returns` carries `E` unconverted.** `E` is already returns-level data, so the
   ingestion aligns it by timestamp to the rows it keeps and does not difference it. A timestamp
   that `E` lacks, and a `NaN` in `E`, stay `NaN`. `E` never drops, pads or fills a row, because
   a series that one consumer reads must not remove observations from every asset. The consumer
   that names a series refuses a non-finite value only on the rows that it fits.
5. **The Currency Factors select their columns by the currency level.** The level of the
   categorical Panel Field, for example `"USD"`, is the key, not the one-hot factor name
   `"<field>=<level>"`. Columns that no currency names are ignored. A level with no column is
   refused, even when no asset holds that level, because its factor enters the factor covariance
   with zero loadings.
6. **The library builds a Currency Excess Return, but it converts no asset return.** #927 ships
   a helper that turns exchange rates, stated as base currency per unit of the currency, and cash
   rates for each period into Currency Excess Returns. The helper takes the `ret_method` of
   `prices_to_returns`. With `:log` it gives the exact split. With `:simple` it gives
   `R_FX + r_cash_C - r_cash_base`, and its docstring states where the cross term goes: into the
   derived local return, or out of the model when a named field supplies the local returns.
7. **The family label `currency` is reserved** for the Currency Factors. No other Exposure
   Estimator can claim it, and no Factor Family constraint can name it.

## Considered options

- **`rd.X` holds local returns under Currency Factors.** Rejected. It changes the meaning of the
  one asset return series under one configuration, and `o_X` and the realised returns of every
  fold then hold local-currency returns.
- **The derived path alone, or the named field alone.** The derived path alone loses the asset
  with no currency label and the caller's own local returns. The named field alone makes every
  caller build one more `observations × assets` matrix, even when `X - Z_ccy * r_ccy` is what
  they would supply. Both paths together cost one optional field.
- **The currency returns inside `F`, selected by name.** Rejected by fact 3. No slicing site
  needs an edit, but every time-series reader of `F` then models the currencies as factors.
- **A dedicated currency block.** It needs the same edits as the Exogenous Series, and it serves
  one consumer. `EWMacroSensitivity` needs the same carrier.
- **A keyword argument of `prior`.** Rejected by fact 4.
- **A panel-level series on the Asset Panel.** Rejected by fact 2. It also adds a fourth kind of
  Panel Field whose axis is not the asset, and about sixteen rebuild sites of the panel.
- **The Exogenous Series on `ReturnsResult` only, attached after `prices_to_returns`.** It keeps
  price containers free of returns-level data. It was not chosen, because a caller who starts
  from prices then aligns the block by hand, outside the path that already aligns every other
  series.
- **`prices_to_returns` refuses a gap in `E`.** Rejected. It refuses rows that the fit never
  reads: the warm-up of the descriptors and the exposure lag consume the first rows.

## Consequences

- `ReturnsResult` and `PricesResult` gain the fields `ne` and `E`. Each site that rebuilds a
  `ReturnsResult` must carry them. The census on #926 names these sites: the two
  `port_opt_view` arities, `prices_to_returns`, the cross-sectional prior's rebuilds, the partial
  fit concatenation, `ReturnsBufferState`, the meta-optimisation rebuilds and the cross-validation
  rebuild.
- `CrossSectionalFactorPrior` has no online path now. A future online path must buffer `E` as
  it buffers `F`, or a later batch loses its Currency Excess Returns.
- `EWMacroSensitivity` can name a column of the Exogenous Series in place of its keyword `ref`,
  so that it works inside the prior. #1365, blocked by #927, tracks that work.
- Factor attribution already treats the family label `currency` as a direct return with no
  standard error, so the Currency Factors reach it with no change.
