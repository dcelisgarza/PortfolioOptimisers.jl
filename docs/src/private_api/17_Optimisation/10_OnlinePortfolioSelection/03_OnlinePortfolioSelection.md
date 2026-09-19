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
PortfolioOptimisers.fill_row_gaps!
PortfolioOptimisers.online_selection_readout
PortfolioOptimisers.online_readout(::OnlinePortfolioSelection)
PortfolioOptimisers.held_timestamps(opt::OnlinePortfolioSelection)
PortfolioOptimisers.online_state_seed(::OnlinePortfolioSelection, ::PortfolioOptimisers.Option{<:Integer})
PortfolioOptimisers.assert_online_fee_source(opt::OnlinePortfolioSelection, pws)
```
