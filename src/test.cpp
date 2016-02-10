#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <tuple>
#include <emmintrin.h>
#include <smmintrin.h>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"
#include "gtest/gtest.h"
#include "task2.h"
class AbsTest : public ::testing::Test{};

TEST_F(AbsTest, MethodBarDoesAbc) {
  BMP* image = new BMP();
  image->ReadFromFile("test_data/Lenna.bmp");
  int h = image->TellHeight() - 2, w = image->TellWidth() - 2;
  Matrix<float> Direct(h, w), Abs_SSE(h, w), Abs(h, w);
  get_dir_abs(image, Direct, Abs);
  get_dir_abs_SSE(image, Direct, Abs_SSE);
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j)
        ASSERT_TRUE(abs(Abs_SSE(i, j) - Abs(i, j)) < 0.1);
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
