__kernel void kernel_grey(__global unsigned int *input,
                          __global unsigned int *output, const int width,
                          const int height, const int depth) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  int index = 0;
  float l = 0;
  // if ((i < width) || (j < height)) {
  //  index = (i * width + j) * depth;
  //  printf("i: %d\n", i);
  //  printf("j: %d\n", j);
  //  printf("index: %d\n", index);

  l = 0.299 * input[(i * width * depth + j * depth)] +
      0.587 * input[(i * width * depth + j * depth) + 1] +
      0.114 * input[(i * width * depth + j * depth) + 2];
  output[(i * width + j)] = round(l);
}
