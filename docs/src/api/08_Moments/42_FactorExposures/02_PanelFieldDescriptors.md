# [Panel Field Descriptors](@id api-panel-field-descriptors)

## Types

```@docs
PanelFieldRatio
PanelFieldLog
Passthrough
```

## Functions

```@docs
descriptor(de::PanelFieldRatio, rd::ReturnsResult)
BookToPrice
CashFlowToPrice
SalesToPrice
EarningsToPrice
ForwardEarningsToPrice
EbitdaToEnterpriseValue
DividendToPrice
ForwardDividendToPrice
ShareholderYield
BookLeverage
MarketLeverage
DebtToAssets
GrossProfitability
GrossMargin
ReturnOnAssets
ReturnOnEquity
AssetTurnover
CashFlowToAssets
SalesToEnterpriseValue
AccrualsCashFlow
AnalystDispersionToPrice
LogMarketCap
ShortInterest
assert_panel_terms
panel_term_names
assert_panel_guard_names
assert_nonneg_panel_fields
positive_panel_fields_fill!
```
