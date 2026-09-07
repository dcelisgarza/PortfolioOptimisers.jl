# Net returns and drawdowns

Net returns and drawdowns are two of the performance metrics of a portfolio. Here we define functions used to compute portfolio returns and related quantities.

```@docs
AbstractWeightDrift
SelfFinancingDrift
drift_position_values
drift_wealth
drift_returns
non_positive_wealth_index
assert_positive_wealth
weight_path
held_weights
drifted_weight_path
drifted_held_weights
AbstractPreviousWeightsSource
DriftedWeights
HeldWeightsResult
assert_held_weights_shape
rebuild_weight_path
held_weights_drift
held_weights_result
drift_observations
investable_reduction(X::MatNum, w::Union{<:VecNum, <:VecVecNum, <:MatNum}, fees::Option{<:Fees}, strict::Bool)
investable_returns_view
held_gap_pairs
filter_held_gaps
held_gap_msg
expand_investable_columns
expand_held_weights
expand_held_member
calc_net_returns(w::VecNum, X::MatNum, args...)
calc_net_returns(w::MatNum, X::MatNum, args...)
calc_net_returns(w::VecVecNum, X::MatNum, fees, wd::AbstractWeightDrift, obs = nothing)
calc_net_asset_returns
cumulative_returns
drawdowns
absolute_drawdown_arr
relative_drawdown_arr
relative_cumulative_returns
absolute_cumulative_returns
```
