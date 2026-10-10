#=
The census of #1411: every consumer of an Asset Panel ignores what its inactive cells hold.

A numeric Panel Field keeps a finite value in every cell, the inactive ones included (ADR 0102),
and a consumer reads the universe masks to leave those cells out. A consumer that forgets the
masks reads the stored value and gives a plausible answer, where the usual convention of a missing
cell outside the universe would give `NaN`. This file finds such a consumer.

**The poison.** `census_poison` copies a `ReturnsResult` and changes every inactive cell of every
Panel Field: a numeric or a tensor value becomes `1e6`, a category code moves to the next level,
and the observed mask becomes `true`. The last change matters: the builder marks each inactive
cell of the fixture as unobserved, and a reader that honours the observed mask would then never
see the poison. Each case runs one consumer on the clean and on the poisoned copy, and needs the
two answers equal cell by cell, `NaN` pattern included.

**The census.** Two lists are built by reflection, not by hand:

  - every function of the package with a method whose signature names an `AssetPanel`, found by
    walking the signature types;
  - every concrete Descriptor Estimator, exposure estimator, forecast unit and forecast target,
    each of which reads the Asset Panel that a `ReturnsResult` carries.

Each name on the first list has a case below or an exemption that states why no cell can reach its
answer. Each type on the second list has a case. A new consumer fails the census until it gets
one, so it is checked the day it lands.

`FeatureDistance` reads each asset and each pair at its own active rows since #1454 (the
decision of #1450), and its cases cover each collapse and each rule for an empty pair. The panel
collapse of a meta-optimiser weighs the active members of each observation since #1456 (the
decision of #1451). Its cases run under each rule, on the fixture and on a second one that adds
a lifted field, a square field, the implied volatilities and a clock for the cross-validated path.
The poison also reaches `iv` at each inactive cell, and `ivpa` at each asset that is inactive at
the last row, the row at which the collapse reads it.
=#
include(joinpath(@__DIR__, "panel_census.jl"))

@testset "Every consumer of an Asset Panel ignores its inactive cells (#1411)" begin
    rd, rdp = census_fixture()
    cases = census_cases(rd)
    rdk, rdkp = census_collapse_fixture(rd)
    collapse_cases = census_collapse_cases(rdk)
    @testset "The census names each consumer once" begin
        census_names_test(cases, collapse_cases, CENSUS_EXEMPT)
    end
    @testset "$(join(names, ", ")): $(label)" for (names, label, f) in cases
        @test census_equal(f(rd), f(rdp))
    end
    @testset "$(join(names, ", ")): $(label)" for (names, label, f) in collapse_cases
        @test census_equal(f(rdk), f(rdkp))
    end
end
