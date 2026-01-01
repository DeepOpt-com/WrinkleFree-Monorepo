"""Patch for muon_fsdp2 missing muon_update function.

The PyPI muon-fsdp2==0.3.0 package is missing the `muon_update` function
that's called in `SingelDeviceWork.start()`. This module provides the
missing function and patches it into the muon_fsdp2 module.

Usage:
    import wrinklefree.training.muon_patch  # Call before using muon_fsdp2
"""

import logging

logger = logging.getLogger(__name__)


def _create_muon_update():
    """Create the missing muon_update function using existing helpers."""
    try:
        import muon_fsdp2

        # Check if already patched
        if hasattr(muon_fsdp2, 'muon_update'):
            return

        # Get the existing helper functions from muon_fsdp2
        zeropower_via_newtonschulz5 = muon_fsdp2.zeropower_via_newtonschulz5
        apply_momentum = muon_fsdp2.apply_momentum
        apply_scaling = muon_fsdp2.apply_scaling

        def muon_update(grad, momentum_buffer, momentum, nesterov, ns_steps, rms_scale):
            """Apply Muon update: momentum + Newton-Schulz orthogonalization + scaling.

            Args:
                grad: Parameter gradient
                momentum_buffer: EMA momentum buffer
                momentum: Momentum beta coefficient
                nesterov: Whether to use Nesterov momentum
                ns_steps: Number of Newton-Schulz iterations
                rms_scale: Whether to use RMS scaling (Moonlight paper)

            Returns:
                Scaled, orthogonalized update tensor
            """
            # Apply momentum (modifies momentum_buffer in-place, returns update)
            update = apply_momentum(grad, momentum_buffer, momentum, nesterov)

            # Apply Newton-Schulz orthogonalization
            update = zeropower_via_newtonschulz5(update, ns_steps)

            # Apply scaling
            update = apply_scaling(update, rms_scale)

            return update

        # Patch into module
        muon_fsdp2.muon_update = muon_update

        # Also patch into the global namespace where SingelDeviceWork looks for it
        # The class method accesses it as a bare name, so we need to inject it
        import sys
        module = sys.modules['muon_fsdp2']
        setattr(module, 'muon_update', muon_update)

        logger.info("Patched muon_fsdp2 with missing muon_update function")

    except ImportError:
        logger.debug("muon_fsdp2 not installed, skipping patch")
    except Exception as e:
        logger.warning(f"Failed to patch muon_fsdp2: {e}")


# Apply patch on import
_create_muon_update()
