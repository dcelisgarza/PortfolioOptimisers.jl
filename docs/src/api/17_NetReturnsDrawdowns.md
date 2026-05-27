# Net returns and drawdowns

Net returns and drawdowns are two of the performance metrics of a portfolio. Here we define functions used to compute portfolio returns and related quantities.

```@docs
calc_net_returns(w::VecNum, X::MatNum, args...)
calc_net_asset_returns
cumulative_returns
drawdowns
absolute_drawdown_arr
relative_drawdown_arr
_relative_cumulative_returns
_absolute_cumulative_returns
```
