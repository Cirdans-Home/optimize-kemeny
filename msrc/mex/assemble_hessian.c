#include "mex.h"
#include <math.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check for proper number of arguments
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("MATLAB:assemble_hessian:invalidNumInputs", "Three inputs required.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("MATLAB:assemble_hessian:maxlhs", "Too many output arguments.");
    }

    // Get the inputs
    double *INV1 = mxGetPr(prhs[0]);
    double *GMAT = mxGetPr(prhs[1]);
    int n = (int) mxGetScalar(prhs[2]);
    double *sqp = mxGetPr(prhs[3]);

    // Create the output matrix
    mwSize dims[2] = {n*n, n*n};
    plhs[0] = mxCreateDoubleMatrix(dims[0], dims[1], mxREAL);
    double *H = mxGetPr(plhs[0]);

    // Declare variables
    int p, q, h, k;
    double term1, term2, term3, term4;

    // Nested loops to fill the H matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p = i + j * n;
    #pragma omp parallel for collapse(2) shared(H, INV1, GMAT,n) private(h, k, q, term1, term2, term3, term4)
            for (h = 0; h < n; h++) {
                for (k = 0; k < n; k++) {
                    q = h + k * n;
                    term1 = INV1[j + k * n]; // Compute e_j' * INV1 * e_h
                    term2 = GMAT[h + i * n]; // Compute e_k' * GMAT * e_i
                    term3 = GMAT[j + k * n]; // Compute e_j' * GMAT * e_h
                    term4 = INV1[h + i * n]; // Compute e_k' * INV1 * e_i
                    H[p + q * n * n] = (sqp[i]/sqp[j])*(sqp[h]/sqp[k])*(term1 * term2 + term3 * term4);
                }
            }
        }
    }

    // Add speye(n^2, n^2) to H
    int i;
    #pragma omp parallel for shared(H, n) private(i)
    for (int i = 0; i < n * n; i++) {
        H[i + i * n * n] += 1.0;
    }
}