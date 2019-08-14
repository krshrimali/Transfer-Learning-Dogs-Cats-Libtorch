# Transfer-Learning-Dogs-Cats-Libtorch

Transfer Learning on Dogs vs Cats dataset using PyTorch C+ API

**Implementation**

1. `mkdir build`
2. `cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..`
3. `make`
4. `./example <path_to_scripting_model>`

TODOs:

1. Load dataset in the way suggested. Prevents OOM (lazily load a single image)
2. Test accuracy. And predictions samples.
