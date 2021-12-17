/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>

namespace gauss::gp {
    void set_matrix_portion(Eigen::MatrixXd& recipient, const Eigen::MatrixXd& portion,
        const MatrixIndices& portion_rows, const MatrixIndices& portion_cols) {
        recipient.block(portion_rows.start, portion_cols.start,
            portion_rows.end - portion_rows.start, portion_cols.end -  portion_cols.start) = portion;
    }
}
