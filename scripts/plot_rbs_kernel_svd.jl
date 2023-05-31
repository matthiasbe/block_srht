using Plots

include("../src/rbs_kernel.jl")

function main()
  dataset_filename = "mnist/mnist_pot12_780"
  sep = ' '
  sigmas = [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
  plot(yticks=10)
  for σ in sigmas
    A = generate_darray("../datasets/" * dataset_filename, σ, sep=sep)
    _,S,_ = svd(Matrix(A))
    plot!(S, yscale=:log10, label="σ = $σ")
  end
  savefig("svd_$(basename(dataset_filename)).pdf")
end

main()
