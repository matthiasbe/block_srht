if haskey(ENV, "SLURM_NTASKS") || haskey(ENV, "BRIDGE_MSUB_NPROC")
  using ClusterManagers
  using Distributed
  using LinearAlgebra

  LinearAlgebra.BLAS.set_num_threads(1)

  ncores = parse(Int, ENV["SLURM_NTASKS"])
  @info "Setting up for SLURM, $ncores tasks detected"; flush(stdout)
  addprocs_slurm(ncores)
end
