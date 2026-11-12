```@meta
Description = "Allocation risk ceiling, private API of PortfolioOptimisers.jl: assert_risk_ceiling, risk_reads_rows, allocation_set_prior, prior_investable_mask, …"
```

# Allocation risk ceiling: private API

```@docs
PortfolioOptimisers.assert_risk_ceiling
PortfolioOptimisers.risk_reads_rows
PortfolioOptimisers.allocation_set_prior
PortfolioOptimisers.prior_investable_mask
PortfolioOptimisers.assert_set_prior_priced
PortfolioOptimisers.allocation_set_priced_mask
PortfolioOptimisers.allocation_set_entry_mask
PortfolioOptimisers.fit_allocation_set_block
PortfolioOptimisers.resolve_allocation_set_block
PortfolioOptimisers.allocation_set_block_step
PortfolioOptimisers.set_allocation_risk_ceiling!
PortfolioOptimisers.set_allocation_risk_return_bounds!
PortfolioOptimisers.risk_contribution_constraints(r::Variance, set::ProgrammeAllocationSet, pr::AbstractPriorResult)
PortfolioOptimisers.clip_at_zero
PortfolioOptimisers.assert_return_floor
```
