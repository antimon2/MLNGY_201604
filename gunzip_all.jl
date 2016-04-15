# GUNZIP MNIST binary

import GZip

filenames = ("MNIST_data/train-labels-idx1-ubyte","MNIST_data/train-images-idx3-ubyte",
            "MNIST_data/t10k-labels-idx1-ubyte","MNIST_data/t10k-images-idx3-ubyte")

for filename in filenames
  gzf = GZip.open("$filename.gz", "rb")

  open(filename, "w") do f
    data = readall(gzf)
    write(f, data)
  end

  close(gzf)
end
