#ifndef PRECONDITIONER_HH
#define PRECONDITIONER_HH PRECONDITIONER_HH
#include <memory>
#include "linear_operator.hh"

/** @file preconditioner.hh
 *
 * @brief Preconditioner base class
 */

/** @class Preconditioner
 *
 * @brief base class for preconditioner
 */
class Preconditioner
{
public:
    /** @brief Create a new instance
     *
     * @param[in] operator_ underlying linear operator
     */
    Preconditioner(std::shared_ptr<LinearOperator> linear_operator_) : linear_operator(linear_operator_) {}

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x) = 0;

protected:
    /** @brief Underlying linear operator */
    std::shared_ptr<LinearOperator> linear_operator;
};

#endif // PRECONDITIONER_HH