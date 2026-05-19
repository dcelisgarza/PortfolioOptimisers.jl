---
applyTo: "test/test-*.jl"
---

# Test Writing Guidelines for PortfolioOptimisers.jl

## Test File Structure

- All test files must be named `test-*.jl` and placed in the `test/` directory.
- Tests are auto-discovered and run by the test harness in `test/runtests.jl`.
- Each test file should focus on testing a specific module or feature.

## Test Patterns

- **Use `@safetestset`**: All top-level test sets must use `@safetestset` to ensure isolation.

    ```julia
    @safetestset "Feature name tests" begin
        using Test, PortfolioOptimisers
        @testset "Specific functionality" begin
            # test code here
        end
    end
    ```

- **Import required packages**: Import all necessary packages inside the `@safetestset` block.
- **Nested test sets**: Use `@testset` for grouping related tests within a `@safetestset`.

## Validation Testing

- **Test argument validation**: Always test that `@argcheck` validations work correctly.

  - Test for `IsEmptyError`, `IsNothingError`, `DimensionMismatch`, `DomainError`, etc.
  - Use `@test_throws ErrorType function_call` to verify error conditions.

- **Example validation tests**:

    ```julia
    @test_throws IsEmptyError Constructor(; v = [])
    @test_throws DimensionMismatch Constructor(; a = rand(3, 4), b = rand(2, 3))
    @test_throws DomainError Constructor(; x = Inf)
    ```

## Test Coverage

- Tests should cover:

  - Normal use cases with valid inputs.
  - Edge cases (empty arrays, boundary values, etc.).
  - Error conditions (invalid inputs, dimension mismatches, etc.).
  - Type stability where relevant.

- Aim for comprehensive coverage of new functionality and edge cases.

## Test Data

- Use simple, reproducible test data (e.g., `rand(3, 4)`, `1:10`, etc.).
- Use meaningful test data that exercises the feature being tested.
- For date/time tests, use `Dates` module: `Date(2020, 1, 1) .+ Day.(0:2)`.

## Test Organization

- Group related tests logically within `@testset` blocks.
- Use descriptive names for test sets and individual tests.
- Keep tests isolated - each test should be independent.

## Testing Multiple Dispatch Methods

When a function has several dispatch variants (e.g., different algorithm types), test each variant explicitly:

```julia
@testset "Full vs Semi algorithm dispatch" begin
    X = rand(50, 4)
    ce_full = Covariance(; alg = Full())
    ce_semi = Covariance(; alg = Semi())
    sigma_full = Statistics.cov(ce_full, X)
    sigma_semi = Statistics.cov(ce_semi, X)
    @test size(sigma_full) == (4, 4)
    @test size(sigma_semi) == (4, 4)
    @test sigma_full != sigma_semi  # different algorithms should differ
end
```

## Testing Composability

Test that nested/composed estimators produce correct results when combined:

```julia
@testset "Composed estimator" begin
    X = rand(50, 4)
    ce = PortfolioOptimisersCovariance(; ce = Covariance(; alg = Semi()))
    sigma = Statistics.cov(ce, X)
    @test size(sigma) == (4, 4)
    @test LinearAlgebra.isposdef(sigma)
end
```

## Testing Result Passthrough

For result types that pass through unchanged (i.e., `f(::AbstractResult, args...) = result`), test that the passthrough is a no-op:

```julia
@testset "Prior result passthrough" begin
    X = rand(50, 4)
    pr = prior(EmpiricalPrior(), X)
    @test prior(pr) === pr   # result passed to prior returns itself
end
```

## Testing `factory` and `moment_view`

When an estimator implements `factory` (for observation weights) or `moment_view` (for slicing), test both:

```julia
@testset "factory propagates weights" begin
    ce = Covariance()
    w = StatsBase.Weights([0.2, 0.3, 0.5])
    ce_w = factory(ce, w)
    @test ce_w.me.w == w
    @test ce_w.ce.w == w
end

@testset "moment_view slices correctly" begin
    ce = Covariance()
    ce_v = moment_view(ce, 1:3)
    @test ce_v isa Covariance
end
```

## Adding New Tests

When adding new functionality:

 1. Create a new test file `test-<feature>.jl` or add to existing test file.
 2. Use `@safetestset` for top-level organisation.
 3. Test both success and failure cases.
 4. Verify all validation logic works correctly.
 5. Test each dispatch variant for functions with multiple methods.
 6. Test composability with other estimators where applicable.
 7. Test result passthrough where applicable.
 8. Run tests locally before committing: `] activate .` then `] test`.
