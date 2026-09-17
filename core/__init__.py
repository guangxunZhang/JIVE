"""Model-agnostic machinery shared by every arm and both experiments:
diversity metrics (Vendi), distribution shift (KID), quality scorers,
per-run reporting, the flow-matching Brownian increment, the projected-noise
draw JIVE injects, and the OSCAR volume objective builder.

Nothing in here is specific to FLUX, SD3.5 or a particular experiment; the
arms in set_level/jive_flux and local_level import from it so a metric can
never drift between two arms of the same table.
"""
