"""Project NCCL checkpoint optimization loaded in actor and rollout workers."""

from coding_opd.verl_optimizations import install_persistent_nccl_sender_buffers

install_persistent_nccl_sender_buffers()
