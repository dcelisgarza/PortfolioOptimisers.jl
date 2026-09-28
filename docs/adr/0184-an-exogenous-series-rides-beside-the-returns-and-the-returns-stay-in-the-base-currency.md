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
returns, `R_base = (1 + R_local)(1 + R_FX) - 1`, so the base-currency excess return holds a cross
term `R_local * R_FX`. The term depends on the asset, so no series with one column for each
currency can hold it. With log returns the split has no cross term.

## Decision

1. **The asset returns stay in the base currency, with or without Currency Factors.** The
   contract of `rd.X` does not change. By default the prior regresses on the returns net of its
   observed factors, `X_t - Z(t - lag) r_t`, where `Z` is the exposure of each asset to each
   observed factor at the observation the model loads for `t`. Under Currency Factors these are
   the local returns. A field `lx` on the prior can name a `NumericPanelField` of net returns
   instead, and the regression then reads that field unchanged. Two cases need the named field:
   an asset with no currency label at an observation, which the derived path gives a `NaN` net
   return, and a caller whose local returns are not `X - Z r`, which includes every simple-return
   panel because of the cross term.
2. **A new block, the Exogenous Series, rides on `PricesResult` and on `ReturnsResult`.** It holds
   named series over the observation axis that belong to no asset. `PricesResult.E` holds
   **levels**, a `TimeArray` whose column names are the names, and `ReturnsResult` holds their
   **returns** in the pair `ne`/`E`, beside `nx/X`, `nf/F` and `nb/B`. A consumer reads a column
   only by its name, so no model takes the block whole as its factors.
3. **`E` is data like every other series, and it never changes the universe of the assets.**
   `price_ingestion` aligns `E` by timestamp to the clock it emits, as it aligns the implied
   volatilities: it pads `NaN` where `E` is silent and names the padding, and `E` never adds or
   drops an observation of the assets. `prices_to_returns` converts `E` with the same
   `ret_method`, `padding` and Gap Return rule as `X`, so the two cannot use two different return
   methods. The online form keeps the last price row, so it covers `E` too, and a price-level fold
   and a `Pipeline` see `E` as one more price series. A gap stays a gap: the consumer that names
   a series refuses a non-finite value only on the rows that it fits.
4. **Every observation slice cuts `E` by rows, and an asset view passes it through.** Both
   arities of `port_opt_view`, the validation of the constructors and every site that rebuilds a
   `ReturnsResult` carry the block. The asset-only view leaves it whole, as it leaves `F`.
5. **An observed factor is a kind of Exposure Estimator.** A member of
   `AbstractObservedExposureEstimator` gives the exposures of its factors and names the column of
   `E` that holds the return of each. `CurrencyExposure` gives one Currency Factor per level of a
   categorical Panel Field, and selects its column by the level, for example `"USD"`.
   `ObservedExposure` wraps an estimated member and names one series, for a macro factor (#1365)
   or an observed market return. The prior takes observed members in its factor list and puts
   them after the estimated factors on every axis. A series with no column is refused, even when
   no asset loads on the factor, because its factor enters the factor covariance.
6. **No family label is reserved.** The family of an observed factor is a property of its member,
   `currency` by default for `CurrencyExposure`. A family holds estimated factors or observed ones,
   never both, and no constrained family can name an observed family. The factor-model block
   states its observed factors through `fx`, their returns on the rows of `csr.f`, and every
   consumer finds them as the trailing columns of both axes: attribution reports them as direct
   returns with no standard error, and the regression diagnostics read the design the regression
   ran on.
7. **The library builds a Currency Excess Index, and it converts no asset return.**
   `currency_excess_index` turns exchange rates, stated as base currency per unit of the currency,
   and cash total-return indices into one level series per currency:
   `I_C(t) = FX_C(t) * cash_C(t) / cash_base(t)`. Its simple return is
   `(1 + R_FX)(1 + r_C) / (1 + r_base) - 1`, the Currency Excess Return. The conversion applies
   `ret_method`: with `:log` the split is exact, and with `:simple` the cross term of each asset
   remains, in the derived net return or out of the model under a named field.

## Considered options

- **`rd.X` holds local returns under Currency Factors.** Rejected. It changes the meaning of the
  one asset return series under one configuration, and `o_X` and the realised returns of every
  fold then hold local-currency returns.
- **The derived path alone, or the named field alone.** The derived path alone loses the asset
  with no currency label and the caller's own local returns. The named field alone makes every
  caller build one more `observations × assets` matrix.
- **The currency returns inside `F`, selected by name.** Rejected by fact 3.
- **A dedicated currency block.** It needs the same edits as the Exogenous Series, and it serves
  one consumer. `EWMacroSensitivity` needs the same carrier.
- **A keyword argument of `prior`.** Rejected by fact 4.
- **A panel-level series on the Asset Panel.** Rejected by fact 2.
- **`E` as returns-level data at the price level, carried unconverted.** The first text of this
  ADR chose it. Rejected, because a `PricesResult` then held one returns-level series beside its
  prices: `X` and `E` could use two different return methods, and the online form, the folds and
  the Gap Return rule each needed a special case for `E`. Levels at the price level let the
  existing conversion do all of it.
- **A Currency Factor with a reserved family label `currency` and its own field on the prior.**
  The first text of this ADR chose it. Rejected, because the mechanism "a factor whose return is
  read from `E` by name" is not specific to currencies: #1365 needs it for macro series, and an
  observed market return needs it too. A kind of member states it once, and a magic string does
  not.
- **`prices_to_returns` refuses a gap in `E`.** Rejected. It refuses rows that the fit never
  reads: the warm-up of the descriptors and the exposure lag consume the first rows.

## Consequences

- `ReturnsResult`, `PricesResult`, `PredictionReturnsResult` and `ReturnsBufferState` gain the
  Exogenous Series, and every site that rebuilds one carries it (#1366).
- `CrossSectionalFactorModel` gains `fx`, `FactorFamilyBasis` gains a pass-through arm, and the
  regression diagnostics read the design of the regression (#1367). That change also makes them
  answer on a block that a prior fitted with a constrained family, which threw before.
- `CrossSectionalFactorPrior` takes observed members in its factor list (#1368).
- The outer problem of a meta-optimiser views its returns data onto the observations its Prior
  Result answers on, so a cross-sectional prior runs inside `NestedClustered` (#1369).
- `CrossSectionalFactorPrior` has no online path now. A future online path must buffer `E` as
  it buffers `F`.
- `EWMacroSensitivity` gains the field `series`, which names a column of the Exogenous Series,
  so that it works inside the prior and inside a fold, and an `ObservedExposure` can pair it with
  the observed series (#1365). The keyword `ref` stays for a direct call, because its removal
  breaks released code. A call that gives both is refused, because no rule can say which of the
  two the caller meant. The descriptor refuses a named series where it is not finite after the
  warm-up of its recursion, as point 3 states. The keyword names no series, so it keeps its
  released rule: the state of the recursion holds its value at a reference return that is not
  finite.
