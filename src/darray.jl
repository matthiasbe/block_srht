@everywhere using DistributedArrays
@everywhere using LinearAlgebra

"""
Select a submatrix corresponding to the column of processors at the given index
"""
function select_proc_column end
@everywhere function select_proc_column(Ap::DArray{Float64,2}, column_proc_index::Int)::DArray{Float64,2}
  nprocs = size(procs(Ap))[1]
  res = [@spawnat p Ap[:l] for p in procs(Ap)[:,column_proc_index]]
  return DArray(reshape(res, (nprocs, 1)))
end

"""
Select a submatrix corresponding to the column of processors at the given index
"""
function select_proc_row end
@everywhere function select_proc_column(Ap::DArray{Float64,2}, row_proc_index::Int)::DArray{Float64,2}
  nprocs = size(procs(Ap))[2]
  res = [@spawnat p Ap[:l] for p in procs(Ap)[row_proc_index, :]]
  return DArray(reshape(res, (1, nprocs)))
end

"""
Get the darray indices owned by a given process
"""
function get_proc_indices(A::DArray, proc::Int)::Tuple{UnitRange{Int}, UnitRange{Int}}
  proc_coord = findall(x -> x == proc, procs(A))
  return A.indices[proc_coord][1]
end

"""
Check all local matrices have the same size.
"""
function check_uniform_chunk_size(A::DArray)
  nps = prod(size(procs(A)))
  i = map(length, A.indices[1])
  for j in reshape(A.indices, nps)
    @assert i == map(length, j)
  end
end

"""
Redistribute data among the processes such that the processor grid is transposed
i.e. procs(B) = procs(A)'
"""
function darray_transpose_layout(A::DArray{Float64,2})::DArray{Float64,2}
  B = dzeros(size(A), procs(A), size(procs(A)'))
  @everywhere procs(B) $B[:l] = collect($A[localindices($B)...])
  return B
end

function norm_frobenius(A::DArray)
  sum = 0
  for p in procs(A)
    sum += @fetchfrom p LinearAlgebra.norm(A[:l])^2
  end
  return sqrt(sum)
end

function nuclear_norm(A::DArray)
  sum = 0
  @assert size(procs(A))[1] == size(procs(A))[2]
  for p in diag(procs(A))
    sum += @fetchfrom p LinearAlgebra.sum(diag(A[:l]))
  end
  return sum
end
