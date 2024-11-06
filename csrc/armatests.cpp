// Test of armadillo library

#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;


SpMat<double> build_sparseT(int n);

int main() {
    
    cout << "Test of ARMADILLO library functionalities for the Kemeny problem" << endl;

    int n = 10; // Size of the random stochastic matrix
    mat P = randu<mat>(n,n); // Generate a random matrix
    P.each_row() /= sum(P,1).t(); // Normalize the matrix
    colvec one(n, fill::ones); // all ones vector

    std::cout << "The matrix P is: " << std::endl;
    P.print();

    // Compute the stationary distribution of P
    rowvec pi(n, fill::ones);
    rowvec pi_old(n, fill::zeros);
    while(norm(pi-pi_old,2) > 1e-13){
    pi_old = pi;
    pi = pi*P;
    pi = pi/sum(pi);
    }

    std::cout << "The stationary distribution is: " << std::endl;
    pi.print();
    std::cout << "Error on the stationary distribution is: " << norm(pi*P-pi,2) << std::endl;

    // Build the sparse matrix T mapping the vectorization of a matrix to the matrix transpose
    SpMat<double> T = build_sparseT(n);

    // Reshape P to a vector
    vec Pvec = vectorise(P);
    // Reshape P' to a vector
    vec PTvec = vectorise(P.t());
    // Check that T*Pvec = PTvec
    vec TPvec = T*Pvec;
    std::cout << "The norm of the difference between T*Pvec and PTvec is: " << norm(TPvec-PTvec,2) << std::endl;

    // Build the matrix of the linear constraints
    // C = [kron(one,I);kron(I,pi');kron(I,Dpi) - kron(Dpi,I)*T];
    // where I is the sparse identity matrix, Dpi is the sparse matrix with pi on the diagonal
    // and T is the sparse matrix mapping the vectorization of a matrix to the matrix transpose
    SpMat<double> I = speye(n,n); // sparse identity matrix
    SpMat<double> Dpi(n,n); // sparse matrix with pi on the diagonal
    Dpi.diag() = pi;     // set the diagonal of Dpi to pi
    SpMat<double> matone(1,n); // sparse matrix with all ones
    SpMat<double> matpi(n,1); // sparse matrix with pi as a row
    matone = one.t(); // set the matrix matone to all ones
    matpi = pi.t(); // set the matrix matpi to pi as a row
    SpMat<double> C = join_cols(join_cols(kron(matone.t(),I),kron(I,matpi)),kron(I,Dpi) - kron(Dpi,I)*T);

    return 0;

}

SpMat<double> build_sparseT(int n) {
    // Initialize arrays to store row, column indices and values
    Mat<uword> locations(2, n * n);
    vec values(n * n, fill::ones);

    // Loop over each element of the matrix P
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // Linear index for element (i, j) in matrix P(:)
            uword row = j * n + i;
            // Linear index for element (i, j) in matrix reshape(P',n^2,1)
            uword col = i * n + j;
            // Store indices in sparse matrix
            locations(0, row) = row;
            locations(1, row) = col;
        }
    }
    // Build the sparse matrix T using the template
    // sp_mat(locations, values, n_rows, n_cols, sort_locations = true, check_for_zeros = true)
    SpMat<double> T(locations, values, n * n, n * n, true, false);

    return T;
}