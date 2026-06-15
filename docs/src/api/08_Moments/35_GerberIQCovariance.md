# Gerber Information Quality Covariance

```@docs
BaseGerberIQCovariance
GerberIQCovarianceAlgorithm
clamp_gerber_iq_n
GerberIQEpsEstimator
GerberIQEps
gerber_iq_eps
GerberIQGammaEstimator
GerberIQGamma
gerber_iq_gamma
GerberIQScalerEstimator
GerberIQScaler
AssetVolatilityGerberIQScaler
gerber_iq_scaling
GerberIQDecayEstimator
ExpGerberIQDecay
regenerate_decay
BasicGerberIQ
PartialGerberIQ
FullGerberIQ
gerber_iq_assert_c_d
gerber_iq_weight
GerberIQCovariance
factory(ce::GerberIQCovariance, w::ObsWeights)
port_opt_view(ce::GerberIQCovariance, i, args...)
gerber_IQ_delta
gerber_IQ
cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1,
                        kwargs...)
cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1,
                        kwargs...)
```
