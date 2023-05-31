using Distributed
@everywhere using DistributedArrays

include("power_of_two.jl") # Introduces the is_pot() function
@everywhere include("darray.jl") # introduces the check_uniform_chunk_size() function

"""
Reduces the array `objs` to one using the binary operator `op`.
This uses a binary reduction tree.
"""
function binarytree_reduce(op::Function, objs::Vector{T})::T where {T}
  nobjs = length(objs)

  # For now objs must contain a power of two objects
  @assert is_pot(nobjs)

  res::Vector{Any} = copy(objs)

  # The level in the tree
  level = 0
  while 2^level < nobjs
    for i in 1:2^(level+1):nobjs
      res[i] = op(res[i], res[i+2^level])
    end
    level += 1
  end
  return res[1]
end

"""
Sum all the parts of a DistributedArray. They must be all of the same size.

The reduction is perform in-place, so the parameter `A` will be altered.
The result stand in the chunk of the worker which id is returned.
"""
function reduce_sum(A::DArray)::Int
  nps = prod(size(procs(A)))
  ps = reshape(procs(A), nps)

  # All chunk must have the same size
  check_uniform_chunk_size(A)

  # This dictionary stores the `Future` objects
  res = Dict()

  # Define the operation reduction
  agg = binarytree_reduce((p1,p2) -> begin
      # Wait for results of the previous reductions on these objects
      if p1 in keys(res)
        wait(pop!(res, p1))
      end
      if p2 in keys(res)
        wait(pop!(res, p2))
      end

      # Sum two chunks
      res[p1] = @spawnat p1 begin
              A[:l] += @fetchfrom p2 A[:l]
              return nothing
            end
      return p1
    end, ps)

  # Wait for last reduction
  wait(res[agg])

  # `agg` contains the proc id where the result of the reduction stands
  return agg
end
