using Distributed

function setnworkers(n::Int)::Nothing
  if nprocs() < n + 1
    addprocs(n - nprocs() + 1)
  elseif nprocs() > n + 1
    rmprocs(workers()[(1 + n):nworkers()])
  end
  @info "Executing the tests with $(nworkers()) workers"
end
