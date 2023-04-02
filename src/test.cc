#include <gtest/gtest.h>

TEST(Arithmetic,TestAdd) {
  int a=3;
  int b=7;
  int c=10;
  EXPECT_EQ(a+b, c);
}

TEST(Arithmetic,TestMultiply) {
  int a=3;
  int b=7;
  int c=21;
  EXPECT_EQ(a*b, c);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

