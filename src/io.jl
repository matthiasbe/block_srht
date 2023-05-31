@everywhere using DistributedArrays
using Dates

"""
Create a folder in the current location.

If the folder already exists, it will be appended a ".2", ".3" etc ... suffix.
"""
function create_folder(name::String)::String
  if !isdir(name)
    return mkdir(name)
  end

  i = 2
  while isdir(name * "." * repr(i))
    i = i + 1
  end
  return mkdir(name * "." * repr(i))
end

"""
Store a distributed array in a directory of name `filename`
"""
function write_darray(A::DArray{Float64,2}, filename::String)::String
  # Create folder
  foldername::String = create_folder(filename)

  # Each process p of the DArray create a file `filename.p` in the folder
  res = [@spawnat p open(foldername * "/part." * repr(p), "w") do f
           write(f, A[:l])
  end for p in procs(A)]
  [fetch(x) for x in res]

  # Add two more files containing partition information
  open(foldername * "/" * "proc_grid_dims", "w") do f
    write(f, collect(size(procs(A))))
  end
  open(foldername * "/" * "procs", "w") do f
    write(f, procs(A))
  end
  open(foldername * "/" * "indices", "w") do f
    write(f, A.indices)
  end
  return foldername
end

"""
Store an array to a file
"""
function write_array(A::Array{Float64,2}, filename)
  open(filename, "w") do f
    write(f, A)
  end
end

#"""
#Read a contiguous subarray `A[row_indices, column_indices]` from an array `A` stored in a file of name `filename`.
#`row_count` is the number of rows of `A`.
#"""
@everywhere function read_array(filename::String, row_indices::UnitRange{Int}, column_indices::UnitRange{Int}, row_count::Int)
  B = Array{Float64}(undef, length(row_indices), length(column_indices))
  open(filename) do f
    seek(f, ((column_indices[1] - 1) * row_count) * sizeof(Float64))
    for (local_j, global_j) in enumerate(column_indices)
      skip(f, (row_indices[1] - 1) * sizeof(Float64))
      read!(f, @view B[:,local_j])
      skip(f, (row_count - row_indices[length(row_indices)]) * sizeof(Float64))
    end
  end
  return B
end

@everywhere function read_metadata(filename)::Tuple{Array{Int,2},Array{Tuple{UnitRange{Int},UnitRange{Int}},2}}
  proc_grid_dims = Array{Int64}(undef, 2)

  read!(filename * "/proc_grid_dims", proc_grid_dims)
  Pr = proc_grid_dims[1]
  Pc = proc_grid_dims[2]

  procs = Array{Int64}(undef, Pr, Pc)
  indices = Array{Tuple{UnitRange{Int64},UnitRange{Int64}}}(undef, Pr, Pc)
  read!(filename * "/procs", procs)
  read!(filename * "/indices", indices)

  return (procs, indices)
end


#Read a contiguous subarray `A[row_indices,column_indices]` from a distributed array `A` store in a file of name `filename`
#using the function `write_darray`.

@everywhere function read_darray(filename::String, row_indices::UnitRange{Int}, column_indices::UnitRange{Int})::Array{Float64,2}
  (procs, indices) = read_metadata(filename)

  A = Array{Float64}(undef, length(row_indices), length(column_indices))

  for (p, p_coords) in enumerate(indices)
    local_rows = intersect(p_coords[1], row_indices)
    local_cols = intersect(p_coords[2], column_indices)
    if !isempty(local_rows) && !isempty(local_cols)
      #println(local_rows, " ", local_cols, " on ", procs[p], "(", p_coords, ")")
      A[.-(local_rows, row_indices[1] - 1), .-(local_cols, column_indices[1] - 1)] = read_array(filename * "/part." * repr(procs[p]), .-(local_rows, p_coords[1][1] - 1), .-(local_cols, p_coords[2][1] - 1), length(p_coords[1]))
    end
  end
  return A
end

function read_darray(filename::String; procs::Array{Int}=workers(), dims::Union{Array{Int},Nothing}=nothing)::DArray{Float64,2}
  (nb_files, indices) = read_metadata(filename)

  height::Int = reduce(max, map(x -> last(x[1]), indices))
  width::Int = reduce(max, map(x -> last(x[2]), indices))

  @info "$(now()) Loading matrix of size $height x $width from filesystem"; flush(stderr)

  if(dims == nothing)
    dims = DistributedArrays.defaultdist((height, width), procs)
  end
  indices = DistributedArrays.chunk_idxs((height, width), dims)[1]

  proc_grid = reshape(procs, (dims[1], dims[2]))

  res = [@spawnat p read_darray(filename, indices[p_coord][1], indices[p_coord][2]) for (p_coord, p) in enumerate(proc_grid)]

  [fetch(i) for i in res]

  return DArray(res)
end

"""
Distribute an array stored in one file on multiple processors.

lda: leading dimension, if we need the top-left part of a larger matrix
"""
function read_darray_onefile(filename::String, height::Int, width::Int; procs::Array{Int}=workers(), dims::Union{Array{Int},Nothing}=nothing, lda::Int=height)::DArray{Float64,2}
  @info "$(now()) Loading matrix of size $height x $width from filesystem"; flush(stderr)

  if(dims == nothing)
    dims = DistributedArrays.defaultdist((height, width), procs)
  end
  indices = DistributedArrays.chunk_idxs((height, width), dims)[1]

  proc_grid = reshape(procs, (dims[1], dims[2]))

  res = [@spawnat p read_array(filename, indices[p_coord][1], indices[p_coord][2], lda) for (p_coord, p) in enumerate(proc_grid)]

  [fetch(i) for i in res]

  return DArray(res)
end

function parse_line(line::String, feature_count::Int; sep::Char=' ')::Vector{Float64}
  record::Vector{Float64} = zeros(feature_count)
  splitted::Vector{String} = split(line, sep, keepempty=false)
  for (i,f) in enumerate(splitted)
    key_value = split(f,':')
    if length(key_value) == 2
      record[parse(Int, key_value[1])] = parse(Float64, key_value[2])
    elseif length(key_value) == 1
      record[i] = parse(Float64, key_value[1])
    else
      @warn "invalid key-value $key_value at line $i"
    end
  end
  return record
end

function read_dataset(filename::String; feature_count::Int64=0, skiplines::Int=0, linecount::Int=0, sep::Char=' ')
  if feature_count == 0
    @assert(length(split(filename, '_')) > 1, "please include feature count in filename")
    feature_count = parse(Int, split(filename, '_')[end])
  end
  file = open(filename)
  for i in 1:skiplines
    readuntil(file, '\n')
  end
  totlines = countlines(filename)
  if linecount == 0
    linecount = totlines - skiplines
  end
  records = Array{Float64,2}(undef, (linecount, feature_count))
  for i in 1:linecount
    line = readuntil(file, '\n')
    records[i,:] = parse_line(line, feature_count, sep=sep)
  end
  return records
end
