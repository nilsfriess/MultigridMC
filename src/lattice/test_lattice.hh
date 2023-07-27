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
        int ny = 5;
        lattice2d = std::make_shared<Lattice2d>(nx, ny);
        int nz = 6;
        lattice3d = std::make_shared<Lattice3d>(nx, ny, nz);
    }
    /** Underlying lattice */
    std::shared_ptr<Lattice1d> lattice1d;
    std::shared_ptr<Lattice2d> lattice2d;
    std::shared_ptr<Lattice3d> lattice3d;
};

/* ******************************************** *
 * **************** 1d Lattice **************** *
 * ******************************************** */

/** @brief check that cell index conversion works */
TEST_F(LatticeTest, TestCellLinear2Euclidean1d)
{
    Eigen::VectorXi idx = lattice1d->cellidx_linear2euclidean(5);
    Eigen::VectorXi cart_idx(1);
    cart_idx(0) = 5;
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that cell index conversion works */
TEST_F(LatticeTest, TestCellEuclidean2Linear1d)
{
    Eigen::VectorXi idx(1);
    idx(0) = 3;
    unsigned int ell = lattice1d->cellidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 3);
}

/** @brief check that shifting cell indices works */
TEST_F(LatticeTest, TestCellShift1d)
{
    Eigen::VectorXi shift_right(1);
    shift_right(0) = +1;
    Eigen::VectorXi shift_left(1);
    shift_left(0) = -1;
    EXPECT_EQ(lattice1d->shift_cellidx(3, shift_right), 4);
    EXPECT_EQ(lattice1d->shift_cellidx(3, shift_left), 2);
    EXPECT_EQ(lattice1d->shift_cellidx(4, shift_right), 5);
    EXPECT_EQ(lattice1d->shift_cellidx(4, shift_left), 3);
}

/** @brief check that vertex index conversion works */
TEST_F(LatticeTest, TestVertexLinear2Euclidean1d)
{
    Eigen::VectorXi idx = lattice1d->vertexidx_linear2euclidean(4);
    Eigen::VectorXi cart_idx(1);
    cart_idx(0) = 5;
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that vertex index conversion works */
TEST_F(LatticeTest, TestVertexEuclidean2Linear1d)
{
    Eigen::VectorXi idx(1);
    idx(0) = 3;
    unsigned int ell = lattice1d->vertexidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 2);
}

/** @brief check that shifting cell indices works */
TEST_F(LatticeTest, TestVertexShift1d)
{
    Eigen::VectorXi shift_right(1);
    shift_right(0) = +1;
    Eigen::VectorXi shift_left(1);
    shift_left(0) = -1;
    EXPECT_EQ(lattice1d->shift_vertexidx(3, shift_right), 4);
    EXPECT_EQ(lattice1d->shift_vertexidx(3, shift_left), 2);
    EXPECT_EQ(lattice1d->shift_vertexidx(4, shift_right), 5);
    EXPECT_EQ(lattice1d->shift_vertexidx(4, shift_left), 3);
}

/** @brief check that working out fine vertex index works */
TEST_F(LatticeTest, TestFineVertexIndex1d)
{
    EXPECT_EQ(lattice1d->fine_vertex_idx(3), 7);
    EXPECT_EQ(lattice1d->fine_vertex_idx(0), 1);
    EXPECT_EQ(lattice1d->fine_vertex_idx(2), 5);
}

/* ******************************************** *
 * **************** 2d Lattice **************** *
 * ******************************************** */

/** @brief check that cell index conversion works */
TEST_F(LatticeTest, TestCellLinear2Euclidean2d)
{
    Eigen::Vector2i idx = lattice2d->cellidx_linear2euclidean(6);
    Eigen::Vector2i cart_idx = {2, 1};
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that cell index conversion works */
TEST_F(LatticeTest, TestCellEuclidean2Linear2d)
{
    Eigen::Vector2i idx = {1, 2};
    unsigned int ell = lattice2d->cellidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 9);
}

/** @brief check that shifting cell indices works */
TEST_F(LatticeTest, TestCellShift2d)
{
    Eigen::Vector2i shift_north = {0, +1};
    EXPECT_EQ(lattice2d->shift_cellidx(5, shift_north), 9);
    Eigen::Vector2i shift_south = {0, -1};
    EXPECT_EQ(lattice2d->shift_cellidx(5, shift_south), 1);
    Eigen::Vector2i shift_east = {+1, 0};
    EXPECT_EQ(lattice2d->shift_cellidx(5, shift_east), 6);
    Eigen::Vector2i shift_west = {-1, 0};
    EXPECT_EQ(lattice2d->shift_cellidx(5, shift_west), 4);
}

/** @brief check that vertex index conversion works */
TEST_F(LatticeTest, TestVertexLinear2Euclidean2d)
{
    Eigen::Vector2i idx = lattice2d->vertexidx_linear2euclidean(5);
    Eigen::Vector2i cart_idx = {3, 2};
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that cell index conversion works */
TEST_F(LatticeTest, TestVertexEuclidean2Linear2d)
{
    Eigen::Vector2i idx = {3, 2};
    unsigned int ell = lattice2d->vertexidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 5);
}

/** @brief check that shifting cell indices works */
TEST_F(LatticeTest, TestVertexShift2d)
{
    Eigen::Vector2i shift_north = {0, +1};
    EXPECT_EQ(lattice2d->shift_vertexidx(7, shift_north), 10);
    Eigen::Vector2i shift_south = {0, -1};
    EXPECT_EQ(lattice2d->shift_vertexidx(7, shift_south), 4);
    Eigen::Vector2i shift_east = {+1, 0};
    EXPECT_EQ(lattice2d->shift_vertexidx(7, shift_east), 8);
    Eigen::Vector2i shift_west = {-1, 0};
    EXPECT_EQ(lattice2d->shift_vertexidx(7, shift_west), 6);
}

/** @brief check that working out fine vertex index works */
TEST_F(LatticeTest, TestFineVertexIndex2d)
{
    EXPECT_EQ(lattice2d->fine_vertex_idx(0), 8);
    EXPECT_EQ(lattice2d->fine_vertex_idx(7), 38);
    EXPECT_EQ(lattice2d->fine_vertex_idx(3), 22);
}

/* ******************************************** *
 * **************** 3d Lattice **************** *
 * ******************************************** */

/** @brief check that cell index conversion works */
TEST_F(LatticeTest, TestCellLinear2Euclidean3d)
{
    Eigen::Vector3i idx = lattice3d->cellidx_linear2euclidean(53);
    Eigen::Vector3i cart_idx = {1, 3, 2};
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that cell index conversion works */
TEST_F(LatticeTest, TestCellEuclidean2Linear3d)
{
    Eigen::Vector3i idx = {1, 3, 2};
    unsigned int ell = lattice3d->cellidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 53);
}

/** @brief check that shifting cell indices works */
TEST_F(LatticeTest, TestCellShift3d)
{
    Eigen::Vector3i shift_north = {0, +1, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(59, shift_north), 63);
    Eigen::Vector3i shift_south = {0, -1, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(59, shift_south), 55);
    Eigen::Vector3i shift_east = {+1, 0, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(59, shift_east), 60);
    Eigen::Vector3i shift_west = {-1, 0, 0};
    EXPECT_EQ(lattice3d->shift_cellidx(59, shift_west), 58);
    Eigen::Vector3i shift_up = {0, 0, +1};
    EXPECT_EQ(lattice3d->shift_cellidx(59, shift_up), 79);
    Eigen::Vector3i shift_down = {0, 0, -1};
    EXPECT_EQ(lattice3d->shift_cellidx(59, shift_down), 39);
}

/** @brief check that vertex index conversion works */
TEST_F(LatticeTest, TestVertexLinear2Euclidean3d)
{
    Eigen::Vector3i idx = lattice3d->vertexidx_linear2euclidean(23);
    Eigen::Vector3i cart_idx = {3, 4, 2};
    EXPECT_EQ(idx, cart_idx);
}
/** @brief check that vertex index conversion works */
TEST_F(LatticeTest, TestVertexEuclidean2Linear3d)
{
    Eigen::Vector3i idx = {3, 4, 2};
    unsigned int ell = lattice3d->vertexidx_euclidean2linear(idx);
    EXPECT_EQ(ell, 23);
}

/** @brief check that shifting vertex indices works */
TEST_F(LatticeTest, TestVertexShift3d)
{
    Eigen::Vector3i shift_north = {0, +1, 0};
    EXPECT_EQ(lattice3d->shift_vertexidx(23, shift_north), 26);
    Eigen::Vector3i shift_south = {0, -1, 0};
    EXPECT_EQ(lattice3d->shift_vertexidx(23, shift_south), 20);
    Eigen::Vector3i shift_east = {+1, 0, 0};
    EXPECT_EQ(lattice3d->shift_vertexidx(23, shift_east), 24);
    Eigen::Vector3i shift_west = {-1, 0, 0};
    EXPECT_EQ(lattice3d->shift_vertexidx(23, shift_west), 22);
    Eigen::Vector3i shift_up = {0, 0, +1};
    EXPECT_EQ(lattice3d->shift_vertexidx(23, shift_up), 35);
    Eigen::Vector3i shift_down = {0, 0, -1};
    EXPECT_EQ(lattice3d->shift_vertexidx(23, shift_down), 11);
}

/** @brief check that working out fine vertex index works */
TEST_F(LatticeTest, TestFineVertexIndex3d)
{
    EXPECT_EQ(lattice3d->fine_vertex_idx(23), 243);
}