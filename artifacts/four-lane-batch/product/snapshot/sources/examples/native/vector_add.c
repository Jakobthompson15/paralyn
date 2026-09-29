#include <paralyn/native.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void check(pr_status status) {
  if (status != PR_SUCCESS) {
    pr_error e;
    pr_last_error(&e);
    fprintf(stderr, "%s: %s (status %d)\n", e.operation, e.message, status);
    exit(1);
  }
}
int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: native_c MODULE.prk NEW_EVIDENCE_DIRECTORY\n");
    return 2;
  }
  enum { n = 1003, guard = 4, count = n + 2 * guard };
  float a[count], b[count], out[count];
  for (int i = 0; i < count; ++i) {
    a[i] = (float)((i % 97) - 48) * 0.25f;
    b[i] = (float)((i % 31) - 15) * 0.5f;
    out[i] = -7777.0f;
  }
  uint32_t devices = 0;
  check(pr_device_count(&devices));
  if (!devices)
    return 1;
  pr_context context = 0;
  check(pr_context_create("auto", &context));
  pr_device_info device;
  check(pr_context_device(context, &device));
  pr_module module = 0;
  check(pr_module_load_file(context, argv[1], &module));
  pr_kernel add = 0, affine = 0;
  check(pr_module_kernel(module, "vector_add", &add));
  check(pr_module_kernel(module, "affine", &affine));
  pr_queue queue = 0;
  check(pr_queue_get(context, &queue));
  pr_buffer ba = 0, bb = 0, bc = 0;
  check(pr_buffer_create(context, sizeof(a), &ba));
  check(pr_buffer_create(context, sizeof(b), &bb));
  check(pr_buffer_create(context, sizeof(out), &bc));
  check(pr_buffer_write(ba, 0, a, sizeof(a)));
  check(pr_buffer_write(bb, 0, b, sizeof(b)));
  check(pr_buffer_write(bc, 0, out, sizeof(out)));
  pr_view va = 0, vb = 0, vc = 0;
  check(pr_view_create(ba, guard * 4, n * 4, 4, PR_READ, &va));
  check(pr_view_create(bb, guard * 4, n * 4, 4, PR_READ, &vb));
  check(pr_view_create(bc, guard * 4, n * 4, 4, PR_READ_WRITE, &vc));
  pr_argument args[5] = {{0}};
  args[0].type = PR_BUFFER;
  args[0].view = va;
  args[1].type = PR_BUFFER;
  args[1].view = vb;
  args[2].type = PR_BUFFER;
  args[2].view = vc;
  args[3].type = PR_I32;
  args[3].i32 = n;
  pr_event event = 0;
  check(pr_launch(queue, add, (pr_dim3){8, 1, 1}, (pr_dim3){128, 1, 1}, args, 4, &event));
  pr_event_info timing;
  check(pr_event_wait(event, &timing));
  if (!timing.completed ||
      !(timing.gpu_start_seconds > 0 && timing.gpu_end_seconds > timing.gpu_start_seconds))
    return 1;
  check(pr_buffer_read(bc, 0, out, sizeof(out)));
  for (int i = 0; i < count; ++i)
    if (out[i] != (i >= guard && i < n + guard ? a[i] + b[i] : -7777.0f)) {
      fprintf(stderr, "C add mismatch %d\n", i);
      return 1;
    }
  check(pr_release(event));
  args[4].type = PR_F32;
  args[4].f32 = 2.0f;
  check(pr_launch(queue, affine, (pr_dim3){8, 1, 1}, (pr_dim3){128, 1, 1}, args, 5, &event));
  args[4].f32 = 999.0f; // Submission must already own its scalar bytes.
  check(pr_release(va));
  check(pr_release(vb));
  check(pr_release(ba));
  check(pr_release(bb));
  check(pr_release(module));
  check(pr_release(add));
  check(pr_release(affine));
  check(pr_event_wait(event, &timing));
  check(pr_buffer_read(bc, 0, out, sizeof(out)));
  for (int i = 0; i < count; ++i)
    if (out[i] != (i >= guard && i < n + guard ? a[i] * 2.0f + b[i] : -7777.0f)) {
      fprintf(stderr, "C affine/lifetime mismatch %d\n", i);
      return 1;
    }
  check(pr_release(event));
  check(pr_release(vc));
  check(pr_release(bc));
  check(pr_release(queue));
  check(pr_context_write_evidence(context, argv[2]));
  check(pr_release(context));
  printf("Device: %s (%s)\nVerification: PASS native C vector_add + affine, "
         "offsets/canaries/scalar/resource lifetime\n",
         device.name, device.backend);
  return 0;
}
