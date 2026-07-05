# Paper replication applications. Each subdirectory reproduces one paper and owns
# its own executable target. These depend on local trace data that is not part of
# the repository, so the whole tree is gated behind LEAKFLOW_BUILD_REPLICATIONS
# (default OFF) and is never part of the normal build or CI.
add_subdirectory(rezaeezade_2025_blindfold)
