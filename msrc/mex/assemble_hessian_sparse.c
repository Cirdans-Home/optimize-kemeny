#include "mex.h"
#include <omp.h> // OpenMP header

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Check number of inputs
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("OptimizeKemeny:compute_H:nrhs", "6 inputs required: INV1, GMAT, irow, jcol, sqp, nnzp.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("OptimizeKemeny:compute_H:nlhs", "1 output required: H.");
    }

    // Inputs
    double *INV1 = mxGetPr(prhs[0]);
    double *GMAT = mxGetPr(prhs[1]);
    double *irow = mxGetPr(prhs[2]);
    double *jcol = mxGetPr(prhs[3]);
    double *sqp  = mxGetPr(prhs[4]);
    mwSize nnzp  = (mwSize) mxGetScalar(prhs[5]);

    // Dimensions
    mwSize n = mxGetM(prhs[0]); // assuming INV1 is n x n

    // Safety checks
    if (mxGetM(prhs[0]) != mxGetN(prhs[0]) || mxGetM(prhs[1]) != mxGetN(prhs[1])) {
        mexErrMsgIdAndTxt("OptimizeKemeny:compute_H:input", "INV1 and GMAT must be square matrices.");
    }

    if (mxGetNumberOfElements(prhs[4]) < n) {
        mexErrMsgIdAndTxt("OptimizeKemeny:compute_H:input", "Length of sqp must be at least n.");
    }

    // Allocate output
    plhs[0] = mxCreateDoubleMatrix(nnzp, nnzp, mxREAL);
    double *H = mxGetPr(plhs[0]);

    // Declare indices and temporaries outside the loop
    mwSize indi, indj;
    mwSize i, j, h, k;
    double term1, term2, term3, term4, val;

    // Parallel loop over rows (indj) using OpenMP  
    for (indj = 0; indj < nnzp; ++indj) {
        h = (mwSize)irow[indj] - 1;
        k = (mwSize)jcol[indj] - 1;

        #pragma omp parallel for private(indi, i, j, h, k, term1, term2, term3, term4, val) shared(H,irow,jcol,INV1,GMAT,sqp)
        for (indi = 0; indi < nnzp; ++indi) {
            i = (mwSize)irow[indi] - 1;
            j = (mwSize)jcol[indi] - 1;

            term1 = INV1[j + k * n]; // INV1(j,k)
            term2 = GMAT[h + i * n]; // GMAT(h,i)
            term3 = GMAT[j + k * n]; // GMAT(j,k)
            term4 = INV1[h + i * n]; // INV1(h,i)

            val = (sqp[i] / sqp[j]) * (sqp[h] / sqp[k]) * (term1 * term2 + term3 * term4);

            H[indj * nnzp + indi] = val;
        }
    }

    // Add identity matrix to H (can also be parallelized)
    mwSize d;
    #pragma omp parallel for shared(H,n) private(d)
    for (d = 0; d < nnzp; ++d) {
        H[d * nnzp + d] += 1.0;
    }
}
