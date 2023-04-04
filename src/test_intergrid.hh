#include <gtest/gtest.h>
#include <random>
#include "intergrid_operator.hh"
#include "diffusion_operator_2d.hh"

/** @brief fixture class for intergrid tests */
class IntergridTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        // Lattice sizes - use anisotropic lattice
        unsigned int nx = 16;
        unsigned int ny = 16;
        lattice = std::make_shared<Lattice2d>(nx, ny);
        coarse_lattice = std::make_shared<Lattice2d>(nx / 2, ny / 2);
        intergrid_operator_avg = std::make_shared<IntergridOperator2dAvg>(lattice);
    }

    /** @brief return a sample state
     *
     * @param[in] coarse Generate state on coarse level?
     * @param[in] random Initialise with random numbers?
     */
    std::shared_ptr<SampleState> get_state(const bool coarse = false, const bool random = false)
    {
        unsigned int ndof = coarse ? coarse_lattice->M : lattice->M;
        unsigned int seed = 1212417;
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        std::shared_ptr<SampleState> X = std::make_shared<SampleState>(ndof);
        if (random)
        {
            for (unsigned int ell = 0; ell < ndof; ++ell)
            {
                X->data[ell] = dist(rng);
            }
        }
        return X;
    }

    /** @brief underlying lattice */
    std::shared_ptr<Lattice2d> lattice;
    /** @brief underlying coarse lattice */
    std::shared_ptr<Lattice2d> coarse_lattice;
    /** @brief intergrid operator for averaging */
    std::shared_ptr<IntergridOperator2dAvg> intergrid_operator_avg;
};

/** @brief check that prolongating then restricting will return the same field up to a factor */
TEST_F(IntergridTest, TestProlongRestrict)
{
    // initial coarse level state
    std::shared_ptr<SampleState> X_coarse = get_state(true, true);
    // prolongated state
    std::shared_ptr<SampleState> X_prol = get_state(false, false);
    // prolongate and restricted state
    std::shared_ptr<SampleState> X_prol_restr = get_state(false, false);
    intergrid_operator_avg->prolongate_add(X_coarse, X_prol);
    intergrid_operator_avg->restrict(X_prol, X_prol_restr);
    double tolerance = 1.E-12;
    EXPECT_NEAR((X_prol_restr->data - 4. * X_coarse->data).norm(), 0.0, tolerance);
}

/** @brief check that coarsening the operator works */
TEST_F(IntergridTest, TestCoarsen)
{
    DiffusionOperator2d linear_operator(lattice, 1.0, 0.0, 1.0, 0.0);
    DiffusionOperator2d coarse_operator(coarse_lattice, 8.0, 0.0, 4.0, 0.0);
    LinearOperator coarsened_operator = intergrid_operator_avg->coarsen_operator(linear_operator);
    const double tolerance = 1.E-12;
    EXPECT_NEAR((coarse_operator.as_sparse() - coarsened_operator.as_sparse()).norm(), 0.0, tolerance);
}