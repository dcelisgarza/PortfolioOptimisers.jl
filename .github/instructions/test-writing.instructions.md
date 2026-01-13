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
    
      + Test for `IsEmptyError`, `IsNothingError`, `DimensionMismatch`, `DomainError`, etc.
      + Use `@test_throws ErrorType function_call` to verify error conditions.

  - **Example validation tests**:
    
    ```julia
    @test_throws IsEmptyError Constructor(; v = [])
    @test_throws DimensionMismatch Constructor(; a = rand(3, 4), b = rand(2, 3))
    @test_throws DomainError Constructor(; x = Inf)
    ```

## Test Coverage

  - Tests should cover:
    
      + Normal use cases with valid inputs.
      + Edge cases (empty arrays, boundary values, etc.).
      + Error conditions (invalid inputs, dimension mismatches, etc.).
      + Type stability where relevant.

  - Aim for comprehensive coverage of new functionality and edge cases.

## Test Data

  - Use simple, reproducible test data (e.g., `rand(3, 4)`, `1:10`, etc.).
  - Use meaningful test data that exercises the feature being tested.
  - For date/time tests, use `Dates` module: `Date(2020, 1, 1) .+ Day.(0:2)`.

## Test Organization

  - Group related tests logically within `@testset` blocks.
  - Use descriptive names for test sets and individual tests.
  - Keep tests isolated - each test should be independent.

## Adding New Tests

When adding new functionality:

 1. Create a new test file `test-<feature>.jl` or add to existing test file.
 2. Use `@safetestset` for top-level organization.
 3. Test both success and failure cases.
 4. Verify all validation logic works correctly.
 5. Run tests locally before committing: `] activate .` then `] test`.
