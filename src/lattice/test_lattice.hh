#include <gtest/gtest.h>
#include <memory>
#include <Eigen/Dense>
#include "lattice.hh"
#include "lattice1d.hh"
#include "lattice2d.hh"
#include "lattice3d.hh"

/** @brief test lattice classes */
class LatticeTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        int n = 6;
        lattice1d = std::make_shared<Lattice1d>(n);
        int nx = 4;
        int ny = 3;
        lattice2d = std::make_shared<Lattice2d>(nx, ny);
        int nz = 5;
        lattice3d = std::make_shared<Lattice3d>(nx, ny, nz);
    }
    /** Underlying lattice */
    std::shared_ptr<Lattice1d> lattice1d;
    std::shared_ptr<Lattice2d> lattice2d;
    std::shared_ptr<Lattice3d> lattice3d;
};

/** @brief check that index conversion works */
TEST_F(LatticeTest, TestLinear2Euclidean1d)
{
    Eigen::VectorXi idx = lattice1d->cellidx_linear2euclidean(5);
    Eigen::VectorXi cart_idx(1);
    cart_idx(0) = 5;
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that index conversion works */
TEST_F(LatticeTest, TestEuclidean2Linear1d)
{
    Eigen::VectorXi idx(1);
    idx(0) = 3;
    unsigned int ell = lattice1d->cellidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 3);
}

/** @brief check that shifting indices works */
TEST_F(LatticeTest, TestShift1d)
{
    Eigen::VectorXi shift_right(1);
    shift_right(0) = +1;
    Eigen::VectorXi shift_left(1);
    shift_left(0) = -1;
    EXPECT_EQ(lattice1d->shift_cellidx(3, shift_right), 4);
    EXPECT_EQ(lattice1d->shift_cellidx(3, shift_left), 2);
    EXPECT_EQ(lattice1d->shift_cellidx(5, shift_right), 0);
    EXPECT_EQ(lattice1d->shift_cellidx(0, shift_left), 5);
}

/** @brief check that index conversion works */
TEST_F(LatticeTest, TestLinear2Euclidean2d)
{
    Eigen::Vector2i idx = lattice2d->cellidx_linear2euclidean(6);
    Eigen::Vector2i cart_idx = {2, 1};
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that index conversion works */
TEST_F(LatticeTest, TestEuclidean2Linear2d)
{
    Eigen::Vector2i idx = {1, 2};
    unsigned int ell = lattice2d->cellidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 9);
}

/** @brief check that shifting indices works */
TEST_F(LatticeTest, TestShift2d)
{
    Eigen::Vector2i shift_north = {0, +1};
    EXPECT_EQ(lattice2d->shift_cellidx(3, shift_north), 7);
    Eigen::Vector2i shift_south = {0, -1};
    EXPECT_EQ(lattice2d->shift_cellidx(3, shift_south), 11);
    Eigen::Vector2i shift_east = {+1, 0};
    EXPECT_EQ(lattice2d->shift_cellidx(3, shift_east), 0);
    Eigen::Vector2i shift_west = {-1, 0};
    EXPECT_EQ(lattice2d->shift_cellidx(3, shift_west), 2);
}

/** @brief check that index conversion works */
TEST_F(LatticeTest, TestLinear2Euclidean3d)
{
    Eigen::Vector3i idx = lattice3d->cellidx_linear2euclidean(42);
    Eigen::Vector3i cart_idx = {2, 1, 3};
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that index conversion works */
TEST_F(LatticeTest, TestEuclidean2Linear3d)
{
    Eigen::Vector3i idx = {1, 2, 3};
    unsigned int ell = lattice3d->cellidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 45);
}

/** @brief check that shifting indices works */
TEST_F(LatticeTest, TestShift3d)
{
    Eigen::Vector3i shift_north = {0, +1, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(57, shift_north), 49);
    Eigen::Vector3i shift_south = {0, -1, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(57, shift_south), 53);
    Eigen::Vector3i shift_east = {+1, 0, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(57, shift_east), 58);
    Eigen::Vector3i shift_west = {-1, 0, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(57, shift_west), 56);
    Eigen::Vector3i shift_up = {0, 0, +1};
    EXPECT_EQ(lattice3d->shift_cellidx(57, shift_up), 9);
    Eigen::Vector3i shift_down = {0, 0, -1};
    EXPECT_EQ(lattice3d->shift_cellidx(57, shift_down), 45);
}