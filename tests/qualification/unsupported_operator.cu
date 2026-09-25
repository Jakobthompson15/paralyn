__global__ void unsupported_operator(float *output) { output[0] = output[1] - output[2]; }
int main() { return 0; }
