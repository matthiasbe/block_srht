@everywhere using DistributedArrays
@everywhere using LinearAlgebra
@everywhere include("io.jl")

using Dates


function generate_rbs_kernel(A::Array{Float64,2}, σ::Float64)::Array{Float64,2}
  return generate_rbs_kernel_subblock(A, A, σ)
end

@everywhere function generate_rbs_kernel_subblock(A1::Array{Float64,2}, A2::Array{Float64,2}, σ::Float64)::Array{Float64,2}
  @assert size(A1) == size(A2)
  B = [exp(-(norm(A1[i,:] - A2[j,:])^2)/(σ^2)) for j in 1:size(A2)[1] for i in 1:size(A1)[1]]
  return reshape(B, (size(A1)[1], size(A2)[1]))
end

"""
Generate a SPSD matrix from a given dataset, stored in CSV format. The number
of lines in the CSV will give the size of the matrix.

The matrix will be distributed as a DArray{Float64}.

Each coefficient of the matrix is computed with the following formula:

b_ij = exp(-|| a_i - a_j ||^2 / sigma)

Where a_i is the ith row of the input dataset, and sigma is a given parameter.
The larger is sigma, the more low-rank is the matrix.
"""
function generate_darray(
        filename::String, σ::Float64; procs::Array{Int}=workers(), sep::Char=' ',
        proc_grid_dims::Array{Int}=Array{Int}(undef,0),
        global_size::Array{Int}=Array{Int}(undef,0))::DArray{Float64,2}
  @assert length(proc_grid_dims) == 0 || length(proc_grid_dims) == 2
  @assert length(global_size) == 0 || length(global_size) == 2

  height, width = length(global_size) == 0 ? 
      (countlines(filename), countlines(filename)) :
      global_size

  dims = length(proc_grid_dims) == 0 ?
      DistributedArrays.defaultdist((height, width), procs) :
      proc_grid_dims

  indices = DistributedArrays.chunk_idxs((height, width), dims)[1]
  proc_grid = reshape(procs, (dims[1], dims[2]))

  println("$(now()) Generating RBS Kernel matrix of size $height x $width on a grid $(size(proc_grid))"); flush(stdout)
  
  res = [@spawnat p begin
           local_row_indices = indices[p_coord][1]
           local_col_indices = indices[p_coord][2]
           @assert size(local_row_indices)[1] == size(local_col_indices)[1]
           skiprow = collect(local_row_indices)[1] - 1
           A1 = read_dataset(filename, skiplines=skiprow, linecount=size(local_row_indices)[1], sep=sep)
           skiprow = collect(local_col_indices)[1] - 1
           A2 = read_dataset(filename, skiplines=skiprow, linecount=size(local_col_indices)[1], sep=sep)

           generate_rbs_kernel_subblock(A1, A2, σ)
         end for (p_coord, p) in enumerate(proc_grid)]

  [fetch(i) for i in res]

  return DArray(res)
end

