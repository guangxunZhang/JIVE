"""RF-Inversion (Rout et al.) for FLUX: inversion + controlled SDE sampler.

pipeline_rf_inversion_sde.RFInversionFluxPipelineSDE inverts an image once and
denoises from the inverted latent with the eta-controlled stochastic sampler;
scheduling_flow_match_euler_discrete_sde adds the SDE step to the flow-matching
Euler scheduler. Used by local_level/run_rf_inverse.py (the RF-Inversion arm)
and by run_flux_arm.py, whose Boomerang arm reuses the same SDE update.
"""
