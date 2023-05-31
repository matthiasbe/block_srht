include("sequential.jl")

using MPI

function generate_collective_orthogonal_matrix(m::Int, comm::MPI.Comm)
  world_rank::Int = MPI.Comm_rank(comm)
  world_size::Int = MPI.Comm_size(comm)

  ### Form orthogonal right matrices V_i .. V_p
  alt_comm::Array{MPI.Comm, 1} = Array{MPI.Comm}(undef, world_size)
  
  # Create communicators
  for i = 1:world_size
    alt_comm[i] = MPI.Comm_split(comm, world_rank >= i, world_rank)
  end
  
  # Init random gaussian matrix
  V::Array{Float64,2} = rand(Float64, (m, n_i))
  
  # Receive from 0 .. i-1 to orthogonalize against
  V2::Array{Float64,2} = zeros((m, n_max))
  for i = 1:world_rank
    MPI.Bcast!(V2, i-1, alt_comm[i])
    V = V - V2 * V2' * V
  end

  # Make orthogonal
  V = LinearAlgebra.qr(V).Q

  # send to processes i+1 .. p so that they orthogonalize against
  if world_rank < world_size - 1
    # TODO use tags to avoid interchange
    MPI.Bcast!(V, world_rank, alt_comm[world_rank+1])
  end

  V_check::Array{Float64,2} = Array{Float64}(undef,(m,m))
  MPI.Gather!(V, V_check, 0, comm)
  if world_rank == 0
    println("Norm 1 of V_check for $world_rank $(norm(I - V_check' * V_check, Inf)))")
    println("Norm 2 of V_check for $world_rank $(norm(I - V_check * V_check', Inf)))")
  else
    println("Norm 1 of V_check for $world_rank $(norm(I - hcat(V,V2)' * hcat(V,V2), Inf)))")
    println("Norm 2 of V_check for $world_rank $(norm(I - hcat(V,V2) * hcat(V,V2)', Inf)))")
  end

  return V
end

"""
Generate a random matrix with an exponential decay of the singular values

Collective version: the local matrix is of size m×n_i
The collective matrix is of size m×N and has roughly k meaningful singular values
where N = Σn_i
"""
function generate_exp_decay_parallel(m, n_i, n_max, k)::Tuple{Array{Float64,1},Array{Float64,2}}
  world_rank::Int = MPI.Comm_rank(MPI.COMM_WORLD)
  world_size::Int = MPI.Comm_size(MPI.COMM_WORLD)

  # Create m singular values on rank 0 and broadcast them

  S::Array{Float64,1} = zeros(m)
  if world_rank == 0
    α::Float64 = 5 / k
    `singular values`
    S = map(y -> exp(-α * y), 1:m)
  end
  MPI.Bcast!(S, 0, MPI.COMM_WORLD)

  ### Form common left matrix U : generate on rank 0 and broadcast
 
  U::Array{Float64,2} = zeros((m, m))

  if world_rank == 0
    U = LinearAlgebra.qr(rand(Float64, (m, m))).Q
  end

  MPI.Bcast!(U, 0, MPI.COMM_WORLD)

  println("Norm for U $world_rank $(norm(I - U' * U, Inf)))")
  println("Norm of V for $world_rank $(norm(I - V' * V, Inf)))")


  US::Array{Float64,2} = hcat(map(x -> S .* x, eachrow(U)))

  M = US * V

  (S, M)
end

function block_rand_parallel(;m::Int=2^10, l::Int=2^6, k::Int=10)
  MPI.Init()

  world_rank::Int = MPI.Comm_rank(MPI.COMM_WORLD)
  world_size::Int = MPI.Comm_size(MPI.COMM_WORLD)

  println("my rank: $world_rank")
  println("size $world_size")

  @assert l <= m
  @assert k <= m / world_size

  # Partitioning of the columns of A which is m×m
  # A = [A1 A2 ... Ap]
  # partitions[i] = (first column index of Ai, last column index of Ai)
  partition_start_indices = floor.(Int, collect(range(0,m,length=world_size+1)))
  "An array of tuples, each tuple defines a domain of A"
  partitions = [partition_start_indices[i]+1:partition_start_indices[i+1] for i in 1:length(partition_start_indices) - 1]
  current_partition = partitions[world_rank + 1]
  println("Partition of rank $world_rank: $current_partition")


  println("Generating local matrix...")
  n_i = length(current_partition)
  n_max = reduce(max, map(length, partitions))
  S,Atp::Array{Float64,2} = generate_exp_decay_parallel(m, n_i, n_max,k)
  println("Transposed matrix size $(size(Atp))")

  Atp_sketched::Array{Float64,2} = @time srht(Atp, l)

  MPI.Reduce!(Atp_sketched, MPI.SUM, 0, MPI.COMM_WORLD)

  if world_rank == 0
    #@time SVD_A = svd(At)
    #@time SVD_truncA::Tuple{Array{Float64,2}, Array{Float64,1}, Array{Float64,2}} = TSVD.tsvd(At, k)
    @time SVD_sketched::SVD = LinearAlgebra.svd(Atp_sketched)

    plot([S[1:k] SVD_sketched.S[1:k]], lab=["Original" "Sketched"])
    savefig("result.pdf")
  end

  MPI.Finalize()
end

#block_rand_parallel()
