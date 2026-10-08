```@meta
Description = "Rolling Descriptors, private API of PortfolioOptimisers.jl: RollingLogReturnState, CarriedDescriptor, descriptor_returns, rolling_window_max, …"
```

# Rolling Descriptors: private API

## Types

```@docs
RollingLogReturnState
CarriedDescriptor
```

## Functions

```@docs
descriptor_returns
rolling_window_max
assert_rolling_sign
show_fields(::RollingLogReturn)
Base.copy(x::PortfolioOptimisers.RollingLogReturnState)
rolling_state_buffer
rolling_state_seed
rolling_state_fold!
rolling_state_value!
rolling_state_push!
descriptor_step
descriptor(de::PortfolioOptimisers.CarriedDescriptor, rd::ReturnsResult)
descriptor_carry
carry_lookback
```
