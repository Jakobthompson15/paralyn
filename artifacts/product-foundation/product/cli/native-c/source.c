#ifdef __cplusplus
#error C source compiled as C++
#endif
#include <stdio.h>
#include <paralyn/native.h>
int main(void) { printf("C ABI %u\n",pr_abi_version()); return 0; }
