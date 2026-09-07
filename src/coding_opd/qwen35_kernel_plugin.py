"""veRL external-lib plugin loaded inside every Qwen3.5 actor worker."""

from coding_opd.qwen35_kernels import install_qwen35_fla_kernels
from coding_opd.verl_optimizations import install_safe_fsdp2_deferred_gradient_sync

install_qwen35_fla_kernels()
install_safe_fsdp2_deferred_gradient_sync()
