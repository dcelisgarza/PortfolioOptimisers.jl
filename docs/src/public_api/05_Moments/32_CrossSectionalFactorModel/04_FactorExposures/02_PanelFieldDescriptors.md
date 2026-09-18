```@meta
Description = "Panel Field Descriptors, public API of PortfolioOptimisers.jl: PanelFieldRatio, PanelFieldLog, Passthrough, descriptor, BookToPrice, CashFlowToPrice, …"
```

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
```
