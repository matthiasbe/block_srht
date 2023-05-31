"""
Perform the operation B^t A+ B where A is broadcast everywhere (small square matrix)
and B is a wide and flat matrix distributed on a row of processes (grid 1 × Pc).
"""
function multiply(A::Array{Float64,2}, B::DArray{Float64,2}; ps::Array{Int}=workers())::DArray{Float64,2}
  @assert size(A)[1] == size(A)[2]
  @assert size(A)[1] == size(B)[1]

  n::Int = size(A)[1] # A is n × n
  m::Int = size(B)[2] # B is n × m
  dims = DistributedArrays.defaultdist((m, m), ps)
  proc_grid = reshape(ps, (dims[1], dims[2]))
  @assert size(proc_grid)[2] == prod(size(procs(B)))

  res = [@spawnat p begin
           i,j = Tuple(findfirst(x -> x == p, proc_grid))
           B_left::Matrix{Float64} = @fetchfrom procs(B)[i] localpart(B)
           B_right::Matrix{Float64} = @fetchfrom procs(B)[j] localpart(B)

           (B_left' * A) * (A * B_right)
         end for p in ps]
  [fetch(i) for i in res]
  return DArray(reshape(res, size(proc_grid)))
end
