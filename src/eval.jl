
"""
Take a right sketched matrix as input
Evaluate the sketching : store the k first singular values to a file
"""
function sketch_right_svd_eval(Ap::DArray{Float64,2}, A_sketched::Array{Float64,2}; k::Int=10, inc::Int=0, method::String)
  m::Int = size(Ap)[1]
  n::Int = size(Ap)[2]
  l::Int = size(A_sketched)[2]
  
  SVD_sketched::SVD = LinearAlgebra.svd(A_sketched)
  writedlm("sketched_k$(k)_p$(length(workers()))_$(log2(m))x$(log2(n))_l$(log2(l))_svdeval_i$(inc)_$(method).csv",  SVD_sketched.S, ",")
  return SVD_sketched.S
end

"""
Take a right sketched matrix B as input
Evaluate the sketching : B = QR, then compute Q^t A
Store the k first singular values of Q^t A
"""
function sketch_right_project(Ap::DArray{Float64,2}, A_sketched::Array{Float64,2};k::Int=10, inc::Int=0, method::String)
  m::Int = size(A_sketched)[1]
  n::Int = size(Ap)[2]
  l::Int = size(A_sketched)[2]
  @assert l < m / prod(size(procs(A)))
  
  @info "$(now()) QR decomposition of the sketched matrix"; flush(stdout)
  fact = qr(A_sketched)

  @info "$(now()) Compute Qt * A"; flush(stdout)
  # Project A : A_k = Q^t A
  A_proj_future = [@spawnat p (fact.Q' * Ap[:l])[1:l,:] for p in workers()]
  A_proj = DArray(reshape(A_proj_future, (1,length(A_proj_future))))

  @info "$(now()) SVD of QtA (size $(size(A_proj))"; flush(stdout)
  # Compute the singular values of A_k
  SVD_sketched::SVD = LinearAlgebra.svd(Matrix(A_proj))
  writedlm("sketched_k$(k)_p$(length(workers()))_$(log2(m))x$(log2(n))_l$(log2(l))_rightproject_i$(inc)_$(method).csv",  SVD_sketched.S, ",")
  @info "$(now()) done"; flush(stdout)
  return SVD_sketched.S[1:k]
end


