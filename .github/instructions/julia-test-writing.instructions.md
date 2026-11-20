---
applyTo: "test/test_*.jl"
---

# Test Writing Guidelines for PortfolioOptimisers.jl

## Test File Structure

- Every test file must have a name of the form `test_*.jl`, and it must be in the `test/` directory.
- The test runner in `test/runtests.jl` finds the test files and runs them.
- Each test file should test one module or one feature.
- A comment, a doc or a tool that names a test file names a file that exists, by its full name or by its `test_<number>` prefix. A commit that merges, renames or deletes a test file corrects every live name of it in the same change. A released ADR keeps the old name, because it is history. `test/test_82_test_file_name_census.jl` gates the rule.

## Test Patterns

- **Each file runs in its own module.** `test/runtests.jl` runs every `test_*.jl` file in an isolated module through ParallelTestRunner.jl (ADR 0003), after its `init_code` preamble loads the common packages. Group the tests of a file in top-level `@testset` blocks.

    ```julia
    @testset "Feature name tests" begin
        @testset "Specific functionality" begin
            # test code here
        end
    end
    ```

- **Rare packages.** Write `using` at the top of the file for a package that `init_code` does not load.
- **Nested test sets.** Use `@testset` to group related tests inside a top-level `@testset`.
- **A case named for a route asserts that the route ran.** A testset or a comment can name a branch, such as a choice that moves, a fallback, a refusal, or a factor that comes alive. Then assert a value that only that branch produces, before the values that the branch shares with the others. The outputs of two branches can be equal on the data of the case, and then every check passes on the branch that the case does not name. A case that names a choice that moves opens with the choice at each step:

    ```julia
    @test [dropped(x.pr) for x in sb] == [["style1"], ["style2"], ["style2"]]
    ```

    This is a judgement for review, not a mechanical rule, so no census gates it.

## Validation Testing

- **Argument validation.** Always test that each `@argcheck` validation throws when it must.

  - Test for `IsEmptyError`, `IsNothingError`, `DimensionMismatch`, `DomainError` and the other error types.
  - Use `@test_throws ErrorType function_call` to check an error condition.

- **Example validation tests.**

    ```julia
    @test_throws IsEmptyError Constructor(; v = [])
    @test_throws DimensionMismatch Constructor(; a = rand(3, 4), b = rand(2, 3))
    @test_throws DomainError Constructor(; x = Inf)
    ```

## Test Coverage

- Tests should cover these cases:

  - Normal use with valid inputs.
  - Edge cases, such as empty arrays and boundary values.
  - Error conditions, such as invalid inputs and dimension mismatches.
  - Type stability, where it matters.

- A new file enters with every line covered, or with a Coverage Exemption. ADR 0082 owns that rule.

## Test Data

- Use simple, reproducible test data, such as `rand(3, 4)` or `1:10`.
- Use test data that reaches the feature under test.
- For a date or time test, use the `Dates` module, for example `Date(2020, 1, 1) .+ Day.(0:2)`.

## Test Organization

- Group related tests in `@testset` blocks.
- Give each test set and each test a name that says what it tests.
- Keep each test independent of the others.

## Testing Multiple Dispatch Methods

When a function has several dispatch variants, such as one per algorithm type, test each variant by name:

```julia
@testset "FullMoment vs SemiMoment algorithm dispatch" begin
    X = rand(50, 4)
    ce_full = Covariance(; alg = FullMoment())
    ce_semi = Covariance(; alg = SemiMoment())
    sigma_full = Statistics.cov(ce_full, X)
    sigma_semi = Statistics.cov(ce_semi, X)
    @test size(sigma_full) == (4, 4)
    @test size(sigma_semi) == (4, 4)
    @test sigma_full != sigma_semi  # different algorithms should differ
end
```

## Testing Composability

Test that a nested or composed estimator gives correct results:

```julia
@testset "Composed estimator" begin
    X = rand(50, 4)
    ce = PortfolioOptimisersCovariance(; ce = Covariance(; alg = SemiMoment()))
    sigma = Statistics.cov(ce, X)
    @test size(sigma) == (4, 4)
    @test LinearAlgebra.isposdef(sigma)
end
```

## Testing Result Passthrough

A result type can pass through a function unchanged, as in `f(::AbstractResult, args...) = result`. Test that the passthrough is a no-op:

```julia
@testset "Prior result passthrough" begin
    X = rand(50, 4)
    pr = prior(EmpiricalPrior(), X)
    @test prior(pr) === pr   # result passed to prior returns itself
end
```

## Testing `factory` and `port_opt_view`

When an estimator implements `factory`, for observation weights, or `port_opt_view`, for slicing, test both:

```julia
@testset "factory propagates weights" begin
    ce = Covariance()
    w = StatsBase.Weights([0.2, 0.3, 0.5])
    ce_w = factory(ce, w)
    @test ce_w.me.w == w
    @test ce_w.ce.w == w
end

@testset "port_opt_view slices correctly" begin
    ce = Covariance()
    ce_v = port_opt_view(ce, 1:3)
    @test ce_v isa Covariance
end
```

## Adding New Tests

When you add new functionality:

 1. Create a new test file `test_<feature>.jl`, or add to an existing test file.
 2. Put the tests in a top-level `@testset`.
 3. Test both the success cases and the failure cases.
 4. Check that every validation throws when it must.
 5. Test each dispatch variant for functions with multiple methods.
 6. Test composability with other estimators where applicable.
 7. Test result passthrough where applicable.
 8. Before committing, run the pre-commit checks, the tests for the area you changed, and the doctests, as `.github/prompts/pre-commit-and-test.prompt.md` orders them. `] test` does not run the doctests.
