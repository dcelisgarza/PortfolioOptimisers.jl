```@meta
Description = "Returns buffer state, private API of PortfolioOptimisers.jl: ReturnsBufferState, assert_pinned_context, pinned_agree, pinned_repr, assert_column_presence, …"
```

# Returns buffer state: private API

## Types

```@docs
PortfolioOptimisers.ReturnsBufferState
```

## Functions

```@docs
PortfolioOptimisers.assert_pinned_context
PortfolioOptimisers.pinned_agree
PortfolioOptimisers.pinned_repr
PortfolioOptimisers.assert_column_presence
PortfolioOptimisers.context_count
PortfolioOptimisers.column_count
PortfolioOptimisers.fold_column
PortfolioOptimisers.fold_column_masked
PortfolioOptimisers.returns_result(state::PortfolioOptimisers.ReturnsBufferState, rows::PortfolioOptimisers.SampleBufferState)
PortfolioOptimisers.column_matrix
PortfolioOptimisers.merge_column
Base.copy(x::PortfolioOptimisers.ReturnsBufferState)
PortfolioOptimisers.copy_column
```
