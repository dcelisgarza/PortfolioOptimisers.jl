```@meta
Description = "The online portfolio selection head, private API of PortfolioOptimisers.jl: online_selection_seed, fold_online_selection, online_selection_pin, …"
```

# The online portfolio selection head: private API

```@docs
PortfolioOptimisers.online_selection_seed
PortfolioOptimisers.fold_online_selection
PortfolioOptimisers.online_selection_pin
PortfolioOptimisers.online_selection_row!
PortfolioOptimisers.row_timestamp
PortfolioOptimisers.report_held_steps
PortfolioOptimisers.report_row_gaps
PortfolioOptimisers.rows_carrier
PortfolioOptimisers.buffer_panel
PortfolioOptimisers.online_selection_readout
PortfolioOptimisers.online_readout(::OnlinePortfolioSelection)
PortfolioOptimisers.held_timestamps(opt::OnlinePortfolioSelection)
PortfolioOptimisers.fees_carry_turnover
PortfolioOptimisers.assert_online_fee_source(opt::OnlinePortfolioSelection, pws)
PortfolioOptimisers.online_portfolio_selection_td_defaults
PortfolioOptimisers.static_allocation_set
PortfolioOptimisers.online_step_fold(opt::OnlinePortfolioSelection{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:PortfolioOptimisers.Option{<:PortfolioOptimisers.OnlinePortfolioSelectionState}}, ctx::PortfolioOptimisers.TimeDependentContext, rd::ReturnsResult)
```
