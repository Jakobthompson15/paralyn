#include <paralyn/native.h>
int main(void) { uint64_t n=0; pr_status s=pr_buffer_size(0,&n); return s==PR_INVALID_HANDLE ? 37 : 99; }
