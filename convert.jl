# convert MNIST binary into HDF5 data
# original: https://github.com/pluskid/Mocha.jl/blob/master/examples/mnist/convert.jl

import GZip
using HDF5
# using Compat

# srand(12345678)

datasets = Dict("train" => ("../MNIST_data/train-labels-idx1-ubyte.gz","../MNIST_data/train-images-idx3-ubyte.gz"),
            "test" => ("../MNIST_data/t10k-labels-idx1-ubyte.gz","../MNIST_data/t10k-images-idx3-ubyte.gz"))

for key in keys(datasets)
  label_fn, data_fn = datasets[key]
  label_f = GZip.open(label_fn)
  data_f  = GZip.open(data_fn)

  # read(label_f, Int32, 2)
  label_header = zeros(Int32, 2)
  read(label_f, label_header)
  @assert ntoh(label_header[1]) == 2049
  n_label = Int(ntoh(label_header[2]))
  # read(data_f, Int32, 4)
  data_header = zeros(Int32, 4)
  read(data_f, data_header)
  @assert ntoh(data_header[1]) == 2051
  n_data = Int(ntoh(data_header[2]))
  @assert n_label == n_data
  h = Int(ntoh(data_header[3]))
  w = Int(ntoh(data_header[4]))

  n_batch = 1
  @assert n_data % n_batch == 0
  batch_size = Int(n_data / n_batch)

  println("Exporting $n_data digits of size $h x $w")

  h5open("../MNIST_data/$key.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, 1, n_data))
    dset_label = d_create(h5, "label", datatype(Float32), dataspace(1, n_data))

    for i = 1:n_batch
      idx = (i-1)*batch_size+1:i*batch_size
      println("  $idx...")

      idx = collect(idx)
      rp = randperm(length(idx))

      # img = readbytes(data_f, batch_size * h*w)
      img = zeros(UInt8, batch_size * h*w)
      read(data_f, img)
      img = convert(Array{Float32},img) / 256 # scale into [0,1)
      # class = readbytes(label_f, batch_size)
      class = zeros(UInt8, batch_size)
      read(label_f, class)
      class = convert(Array{Float32},class)

      for j = 1:length(idx)
        r_idx = rp[j]
        dset_data[:,:,1,idx[j]] = img[(r_idx-1)*h*w+1:r_idx*h*w]
        dset_label[1,idx[j]] = class[r_idx]
      end
    end
  end

  close(label_f)
  close(data_f)
end
