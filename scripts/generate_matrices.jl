using DelimitedFiles

include("../src/init_cluster.jl")

include("../src/random_matrix.jl")
include("../src/io.jl")

generation_folder = "generated_matrices"
mkpath(generation_folder)

m = 5
n = 5
k = 4

flush(stdout); flush(stderr)
folder_name =  generation_folder * "/matrix_$(m)x$(n)_k$(k)"
A,S = generate(m=2^m, k=k, n=2^n)
folder_name = write_darray(A, folder_name)
writedlm(folder_name * "/original_singular_values.csv",  S, ",")
flush(stdout); flush(stderr)
