"""Baseline algorithms JIVE is compared against, vendored here so a table row
can never be produced by code that has drifted from what the paper cites.

  oscar/         OSCAR (per-step CLIP volume perturbation), Oscar package from
                 the authors' release.
  cads.py        CADS (Sadat et al., ICLR 2024): annealed conditioning
                 corruption. Standalone FLUX.1-schnell generator, and the
                 source of the cads_gamma/cads_corrupt used by the JIVE+CADS
                 arm so the two cannot disagree on the corruption math.
  rf_inversion/  RF-Inversion (Rout et al.): FLUX inversion + SDE sampler,
                 the inversion-based arm of the local-level comparison.
"""
