//__kernel void kernel_threshold(__global unsigned char *input,
//                               __global unsigned long *output_brightness,
//                               const int width, const int height) {
//  int i = get_global_id(0);
//  int j = get_global_id(1);
//
//
//  output_brightness[0] = output_brightness[0] + input[i * width + j];
//}
__kernel void kernel_threshold(__global unsigned int *input,
                               __global unsigned int *output_brightness,
                               const int width, const int height) {

  __local unsigned int local_brightness;
  int gid = get_global_id(0) * get_local_size(0);
  for (int i = 0; i < get_local_size(0); i++) {
    output_brightness[get_global_id(0)] =
        output_brightness[get_global_id(0)] + input[gid + i];
  }
}