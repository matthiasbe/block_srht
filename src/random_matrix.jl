using Distributed
@everywhere using DistributedArrays
@everywhere using LinearAlgebra
@everywhere using Dates

include("darray.jl")

function memusage()
  A = rand(1000,1000)
  ntasks = parse(Int, ENV["SLURM_NTASKS"])
  bits =  parse(Int, readchomp(`sstat -j $(ENV["SLURM_JOB_ID"]) -o AveRSS --noheader --noconvert`))
  return floor(Int, ntasks * bits/(8*1024^3))
end

"""
Generate a matrix and write the singular values to a file
"""
function generate(;m::Int=2^10, k::Int=10, n::Int=m)::Tuple{DArray{Float64,2},Array{Float64,1}}
  Ap::DArray{Float64,2},S::Array{Float64,1} = generate_exp_decay_distributed(k, m, n=n)
  @debug "$(now()) Matrix size $(size(Ap))"; flush(stdout)
  #writedlm("original_k$(k)_p$(length(workers()))_$(log2(m))x$(log2(n)).csv",  S[1:k], ",")
  return Ap, S
end

"""
Orthonormalize the columns of a column partitioned matrix

Requires m ≥ n otherwise columns cannot be all orthogonal

A = [A1 A2 ... Ap]

such that ∀i,j≤p Ai' Aj = {I if i == j, 0 else} where I is the identity matrix
and 0 is the zero matrix
"""
function orthonormalize(A::DArray)
  @assert size(A)[1] >= size(A)[2]
  for (p_i, p) in enumerate(workers())
    @info "$(now()) Orthonormalization on worker $p ($p_i / $(nworkers())) (matrix size $( @fetchfrom p size(A[:l])) mem $(memusage())GB"; flush(stderr)

    # Compute the local orthonormal partition
    remotecall_wait((p_i) -> begin
      fact = LinearAlgebra.qr!(A[:l])
      @info "$(now()) Local matrix is orthogonal, sending"; flush(stderr)

      # On processors p+1 .. P, orthogonalize against the basis of p
      w = workers()[p_i+1:nworkers()]
      @info "$(now()) broadcasting $(Base.summarysize(fact)/(2^20)) MBytes to $(length(w)) processes"; flush(stderr)
      if length(w) > 0
        block_size = size(A[:l])[2]

        begin
          f::Array{Future,1} = [@spawnat p2 begin
            Q = fact.Q
            A[:l] = A[:l] .- Q * (Q' * A[:l])[1:block_size,:]
            Q = nothing
            GC.gc()
          end for p2 in w]

          [wait(r) for r in f]
        end
      end
      A[:l] = Matrix(fact.Q)
      fact = nothing
      GC.gc()
    end, p, p_i)
    GC.gc()
  end
end

"""
Generate an m×n matrix M from a computed diagonal matrix S of size min(m,n) x min(m,n)
and two random orthogonal singular values U and V such that M = USV

S is formed of m coefficient, and S[i] = e^{-5/k}

This function returns the matrix M and its singular values S. 

The matrix M is distributed along the second dimension

M = [M1 ... Mp]
"""
function generate_exp_decay_distributed(k::Int, m;n = m)::Tuple{DArray{Float64,2},Array{Float64,1}}
  @info "$(now()) Generating a random matrix of size $m x $n"; flush(stderr)
  l = min(m,n)

  # Create m singular values on master
  α::Float64 = 5 / k
  `singular values`
  S = map(y -> exp(-α * y), 1:l) # exponential decay
  #S = map(y -> 100/y, 1:l) # Hyperbolic decay
  #S = [[n*k-n*i+1 for i in 1:k];[1-i/n for i in 1:n-k]] # Two segment decay, of slope -n and -1/n

  # Computing U and V orthogonal matrices, such M = U S V^t
  # We take U and V
  # * Column-partitioned for the orthogonalization
  #   such that U^t U = I and V^t V = I
  # * Row partitioned for the product computation U S V^t
  #   such that the result resides on multiple processors
  #
  # [U1                    [U1                    [ U1 S V1t   ..   U1 S VPt ]
  #  ..  S  [V1t .. VPt] =  ..  [SV1t .. SVPt] =  [   ..               ..    ]
  #  UP]                    UP]                   [ UP S V1t   ..   UP S VPt ]

  @info "$(now()) Generating left orthogonal matrix"; flush(stderr)
  U::DArray{Float64,2} = drand((m,l), workers(), [1,nworkers()])
  @info "$(now()) Orthonormalizing left orthogonal matrix"; flush(stderr)
  orthonormalize(U)
  @info "$(now()) Transposing left orthogonal matrix"; flush(stderr)
  U = darray_transpose_layout(U)

  GC.gc()
  [GC.gc() for p in workers()]

  @info "$(now()) Generating right orthogonal matrix"; flush(stderr)
  V::DArray{Float64,2} = drand((n,l), workers(), [1,nworkers()])
  @info "$(now()) Orthonormalizing right orthogonal matrix"; flush(stderr)
  orthonormalize(V)
  @info "$(now()) Transposing right orthogonal matrix"; flush(stderr)
  V = darray_transpose_layout(V)

  GC.gc()
  [GC.gc() for p in workers()]

  M::DArray{Float64,2} = dzeros((m,n), workers(), [1,nworkers()])
  res = [@spawnat p begin
           # V ← VS ((SVt)^t = VS^t = VS)
           rmul!(V[:l], Diagonal(S))
           # V ← UVt
           for p in workers()
             (U_on_p, indices) = @fetchfrom p (U[:l], localindices(U)[1])
             M[:l][indices,:] = U_on_p * V[:l]'
           end
           GC.gc()
           @info "$(now()) multiplication done"; flush(stderr)
  end for p in workers()]
  [wait(r) for r in res]

  close(U)
  close(V)

  GC.gc()
  [GC.gc() for p in workers()]

  (M, S)
end
