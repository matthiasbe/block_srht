include("tsqr.jl")
@everywhere include("reduce.jl")
@everywhere include("srht.jl")
@everywhere using DistributedArrays
@everywhere using TimerOutputs

using Dates

"""
Compute B = AΩ where A is a m×m column-distributed matrix

A = [A1 ... Ap]

and Ω is a m×l sketching matrix (see `srht()`)

We must have l ≤ m/p
"""
function sketch_right(Ap::DArray{Float64,2}, l::Int, to::TimerOutput; global_seed::UInt=rand(UInt), block_seeds::Vector{UInt}=rand(UInt,length(procs(Ap))))::Array{Float64,2}
  m::Int = size(Ap)[1]
  #@assert l <= m / size(procs(Ap))[2]
  @assert(size(procs(Ap))[1] == 1) # check Ap is column-distributed

  nps = prod(size(Ap.pids))
  ps = reshape(Ap.pids, nps)

  # Sketch each block: compute A_i Ω_i
  #@info "$(now()) starting sketching ..."; flush(stdout)
  @timeit to "local" begin
    res = [@spawnat p begin
      Atp = permutedims(Ap[:l])
      srht(Atp, l, to=to, global_seed=global_seed, block_seed=block_seeds[p_index])
    end for (p_index,p) in enumerate(ps)]
    [wait(r) for r in res]
    Atp_sketched::DArray{Float64,2} = DArray(reshape(res,(1,nps)))
  end

  # Sum reduction : sums all partitions of Atp_sketched
  # The result is on aggregator_worker
  @timeit to "reduce" aggregator_worker = reduce_sum(Atp_sketched)

  # Get the result on master process
  A_sketched = @fetchfrom aggregator_worker Atp_sketched[:l]' ./ sqrt(nps)

  close(Atp_sketched)
  return A_sketched
end

"""
Compute B = ΩA where A is a m×m row-distributed matrix

A = [A1;...;Ap]

and Ω is a l×m sketching matrix (see `srht()`)

We must have l ≤ m/p
"""
function sketch_left end
@everywhere function sketch_left(Ap::DArray{Float64,2}, l::Int, to::TimerOutput)::Array{Float64,2}
  m::Int = size(Ap)[2]
  @assert(size(procs(Ap))[2] == 1) # check Ap is row-distributed

  nps = prod(size(Ap.pids))
  ps = reshape(Ap.pids, nps)

  seed = rand(UInt)

  # Sketch each block: compute A_i Ω_i
  @info "$(now()) starting sketching ..."; flush(stdout)
  @timeit to "local" begin
    res = [@spawnat p srht(Ap[:l], l, to=to, global_seed=seed) for p in ps]
    [wait(r) for r in res]
    Ap_sketched::DArray{Float64,2} = DArray(reshape(res,(1,nps)))
  end

  # Sum reduction : sums all partitions of Atp_sketched
  # The result is on aggregator_worker
  aggregator_worker = reduce_sum(Ap_sketched)

  # Get the result on master process
  A_sketched = @fetchfrom aggregator_worker Ap_sketched[:l] ./ sqrt(nps)

  close(Ap_sketched)
  return A_sketched
end

"""
Sketch a 2D distributed matrix from right and left

Ap is distributed on a Pr×Pc processor grid

First reduce along each column of the processor grid,
then reduce along the final obtained row.

Return B = ΘAΩ
"""
function sketch_2D(Ap::DArray{Float64,2},l::Tuple{Int,Int}, to::TimerOutput)::Array{Float64,2}
  ncols::Int = size(Ap.pids)[2]
  left_sketches_futures::Array{Future,1} = Array{Future}(undef, ncols)

  # For each col in the processor grid
  # Compute the sketch of the block column
  for j in 1:ncols
    # Elect a new master on each column
    master = Ap.pids[1,j]
    # Each master node takes care of its column of processes
    left_sketches_futures[j] = @spawnat master begin
      sub = select_proc_column(Ap, j)
      sketch_left(sub, l[1], to)
    end
  end

  # Use Futures to build a new matrix containing the left sketch for each block column
  left_sketches = DArray(reshape(left_sketches_futures, (1,ncols)))
  return sketch_right(left_sketches, l[2], to)
end

"""
Sketch a 2D distributed matrix from right and left.

Each local block is sketched, then all results are sum-reduced.
Perform B_ij = Θ_i A_ij Ω_j
Sum all B_ij
"""
function sketch_2D_eachblock(Ap::DArray{Float64,2}, l::Tuple{Int,Int}, to::TimerOutput)::Array{Float64,2}

  (Pr,Pc) = size(Ap.pids)
  nps = prod(size(Ap.pids))
  ps = reshape(Ap.pids, nps)

  # Initialize global Random Number Generators, for left and right sketching
  left_seed = rand(Int)
  right_seed = rand(Int)
  left_seeds = rand(Int, Pr+1)
  right_seeds = rand(Int, Pc+1)

  # Sketch each block: compute Θ_i A_i Ω_i
  @info "$(now()) starting sketching ..."; flush(stdout)
  @timeit to "local" begin
    res = [@spawnat p begin
             proc_coord = findall(x -> x == p, procs(A))[1]
             temp = srht(Ap[:l], l[1], to=to, global_seed=left_seed, block_seed=left_seeds[proc_coord[1]])
             temp = permutedims(temp)
             srht(temp, l[2], to=to, global_seed=right_seed, block_seed=right_seeds[proc_coord[2]])
           end for p in ps]
    [wait(r) for r in res]
    Atp_sketched::DArray{Float64,2} = DArray(reshape(res,size(procs(Ap))))
  end

  # Sum reduction : sums all partitions of Atp_sketched
  # The result is on aggregator_worker
  @timeit to "reduce" aggregator_worker = reduce_sum(Atp_sketched)

  # Get the result on master process
  A_sketched = @fetchfrom aggregator_worker Atp_sketched[:l]'

  close(Atp_sketched)
  return A_sketched
end

"""
Sketch from right and left independently to compute projection matrices Ql and Qr.

Then project the DArray A as Ql^t A Qr.

Where Ql = qr(AΩ) and Qr = qr(A^t Θ^t)

Not completely implemented yet, needs TSQR algorithm (see tsqr.jl)
"""
function sketch__project_2D(Ap::DArray{Float64,2},l::Tuple{Int,Int})::Array{Float64,2}
  throw(ErrorException("Not implemented"))
  ncols::Int = size(Ap.pids)[2]
  left_sketches_futures::Array{Future,1} = Array{Future}(undef, ncols)

  # For each col in the processor grid
  # Compute the sketch of the block column
  for j in 1:ncols
    # Elect a new master on each column
    master = Ap.pids[1,j]
    # Each master node takes care of its column of processes
    left_sketches_futures[j] = @spawnat master begin
      sub = select_proc_column(Ap, j)
      sketch_left(sub, l[1])
    end
  end

  # Convert Array{Future{Future}} to Array{Future} by fetching results
  for j in 1:ncols
    left_sketches_futures[j] = fetch(left_sketches_futures[j])
  end

  # Use Futures to build a new matrix containing the left sketch for each block column
  left_sketches = DArray(reshape(left_sketches_futures, (1,ncols)))
  return sketch_right(left_sketches, l[2])
end

"""
Apply a left sketch B = Ω A where A is distributed on a 2D grid Pr × Pc
and Ω on 1×Pr
"""
function sketch_left_2D(Ap::DArray{Float64,2}, l::Int, to::TimerOutput; global_seed::UInt=rand(UInt), block_seeds::Vector{UInt}=rand(UInt,length(procs(Ap))))::DArray{Float64,2}
  (Pr,Pc) = size(procs(Ap))
  nps = prod(size(procs(Ap)))
  ps = reshape(procs(Ap), nps)

  # Sketch each block: compute Θ_i^t A_ij Θ_j, Θ_i^t A_ij and A_ij Θ_j
  #@info "$(now()) sketching locally and gathering UtAU"; flush(stdout)
  @timeit to "local" begin
    res = [@spawnat p begin
             proc_coord::Tuple{Int,Int} = findall(x -> x == p, procs(Ap))[1]
             # D_i = Θ_i^t A_ij
             srht(copy(localpart(Ap)), l, to=to,
                             global_seed=global_seed,
                             block_seed=block_seeds[proc_coord[1]])
           end for p in ps]
    [wait(r) for r in res]
    Dp_temp::DArray{Float64,2} = DArray(reshape(res,(Pr,Pc)))
  end

  # Sum all blocks of D_i in a column of procs
  @timeit to "reduce UtA" begin
    #@info "$(now()) gathering AU"; flush(stdout)
    resD = [@spawnat procs(Ap)[1,j] begin
              # Select the submatrix of the column
              res = [@spawnat procs(Ap)[i,j] begin Dp_temp[:l] end for i in 1:Pr]
              [wait(r) for r in res]
              sub::DArray{Float64, 2} = DArray(reshape(res, (Pr,1)))
              # Sum all blocks
              agg = reduce_sum(sub)
              @assert(agg == myid()) # Check we were elected by the reduce
              localpart(sub) ./ sqrt(Pr)
            end for j in 1:Pc]
    [wait(r) for r in resD]
    Dp::DArray{Float64,2} = DArray(reshape(resD,(1,Pc)))
  end
  return Dp
end

"""
Sketch a 2D distributed matrix from right and left with the same random sketch.

Each local block is sketched, then all results are sum-reduced.
Perform B_ij = Θ_i^t A_ij Θ_j
Sum all B_ij

Perform D_ij = A_ij Θ_j
Sum all D_ij for a given i

Then we obtain the low rank approximation

A ≈ D B^{-1} D^t
"""
function sketch_nystrom_pinv(Ap::DArray{Float64,2}, l::Int, to::TimerOutput)::Tuple{Matrix{Float64}, DArray{Float64,2}}
  (Pr,Pc) = size(procs(Ap))

  # Initialize global Random Number Generator
  seed = rand(UInt)
  seeds = rand(UInt, Pr)

  @timeit to "reduce UtA" Dp = sketch_left_2D(Ap, l, to, global_seed=seed, block_seeds=seeds)
  @timeit to "reduce UtAU" B = sketch_right(Dp, l, to, global_seed=seed, block_seeds=seeds)

  return B, Dp
end

function sketch_nystrom_cholesky(Ap::DArray{Float64,2}, l::Int, to::TimerOutput)::Tuple{Matrix{Float64}, DArray{Float64,2}}
  (Pr,Pc) = size(procs(Ap))

  # Initialize global Random Number Generator
  seed = rand(UInt)
  seeds = rand(UInt, Pr)

  @timeit to "reduce UtA" Dp = sketch_left_2D(Ap, l, to, global_seed=seed, block_seeds=seeds)
  @timeit to "reduce UtAU" B = sketch_right(Dp, l, to, global_seed=seed, block_seeds=seeds)

  @timeit to "cholesky" C = cholesky(Symmetric(B))
  @timeit to "compute B = Y C-1" begin
    res = [@spawnat p (C.L \ Dp[:l]) for p in procs(Dp)]
    [wait(r) for r in res]
    CDp = DArray(reshape(res, (1, length(procs(Dp)))))
  end

  return I(l), CDp
end

function sketch_nystrom(Ap::DArray{Float64,2}, l::Int, to::TimerOutput)::Tuple{Vector{Float64}, DArray{Float64,2}, Matrix{Float64}}
  (Pr,Pc) = size(procs(Ap))

  # Initialize global Random Number Generator
  seed = rand(UInt)
  seeds = rand(UInt, Pr)

  @timeit to "reduce UtA" Dp = sketch_left_2D(Ap, l, to, global_seed=seed, block_seeds=seeds)
  @timeit to "reduce UtAU" B = sketch_right(Dp, l, to, global_seed=seed, block_seeds=seeds)

  tol = 1e-14
  @timeit to "cholesky" U_B, S_B, _ = svd(B)
  k_tol = findfirst(x -> x < tol, S_B)
  if k_tol == nothing
    k_tol = length(S_B)
  end

  
  @timeit to "compute B = Y C-1" begin
    res = [@spawnat p (diagm(map(x -> 1 / sqrt(x), S_B[1:k_tol])) * U_B[:,1:k_tol]'* Dp[:l])' for p in procs(Dp)]
    [wait(r) for r in res]
    CDp = DArray(reshape(res, (length(procs(Dp)), 1)))
  end

  @timeit to "tsqr" R::Matrix{Float64} = tsqr(CDp)

  @timeit to "last step" begin 
    @timeit to "svd or R" _,S,V = svd(R)
  end
  close(Dp)

  return S,CDp,V
end

