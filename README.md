# Custom Neural Network

**A custom neural network made by Jonah Taylor**

This custom neural network was for the purpose of demystifying machine learning. This project was built over Fall 2024. The project was largely inspired by 3Blue1Brown's general view youtube series on machine learning.
---

## Build and run the program

```
git submodule update --init --recursive

mkdir build
cd build

cmake ..
make

./pong
```

---

## Developer Notes

### Layers

This neural network includes dense layers and convolutional layers. Convolutional layers are clunky because each convo-layer only has one convolution.
