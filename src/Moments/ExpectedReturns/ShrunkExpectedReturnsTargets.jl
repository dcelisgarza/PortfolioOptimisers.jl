struct SERT_GrandMean <: ShrunkExpectedReturnsTarget end
struct SERT_VolatilityWeighted <: ShrunkExpectedReturnsTarget end
struct SERT_MeanSquareError <: ShrunkExpectedReturnsTarget end

export SERT_GrandMean, SERT_VolatilityWeighted, SERT_MeanSquareError
