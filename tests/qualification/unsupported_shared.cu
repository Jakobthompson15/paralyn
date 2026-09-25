__global__ void unsupported_shared(float *output) {
  __shared__ float tile[8];
  output[0] = tile[0];
}
int main() { return 0; }
