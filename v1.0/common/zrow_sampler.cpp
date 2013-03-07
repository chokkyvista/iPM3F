/* Sample each row (existing features) in the binary latent feature matrix Z
 * according to the semi-collapsed Gibbs sampler
 * 
 * Written by Minjie Xu (chokkyvista06@gmail.com)
 */

#include <assert.h>
#include <string.h>
#include <math.h>
#include <random>

#include "mex.h"

/* 
 * exp(-C*Wk_d'*(cv1_d+cv2_d.*(WZnt_d+(0.5-oZnk)*Wk_d)))
 * WZnt_d = WZnt_d + (Znk-OZnk)*Wk_d;
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Check inputs & outputs */
	if ((nrhs != 8 && nrhs != 12) || nlhs != 0)
		mexErrMsgTxt("Incorrect number of inputs/outputs!");
	for (int i = 0; i < nrhs; ++i)
		if (mxIsSparse(prhs[i]))
			mexErrMsgTxt("Inputs must be full matrices!");

	/* Input */
	double* const Zn_k = mxGetPr(prhs[0]);
	const double* const p1bp0_prior_k = mxGetPr(prhs[1]);
	
	const double* const W_dk = mxGetPr(prhs[2]);
	const double* const cv1_d = mxGetPr(prhs[3]);
	const double* const cv2_d = mxGetPr(prhs[4]);
	double* const WZnt_d = mxGetPr(prhs[5]);
	const double C = mxGetScalar(prhs[6]);
	const int algtype = (int)mxGetScalar(prhs[7]);
	assert(algtype==1 || algtype==2);

	const double* yneta_k = NULL;
	double cc, Znyneta, invlambda;
	if (nrhs > 8) {
		yneta_k = mxGetPr(prhs[8]);
		cc = mxGetScalar(prhs[9]);
		Znyneta = mxGetScalar(prhs[10]);
		invlambda = mxGetScalar(prhs[11]);
	}
	
	/* Compute Sizes */
	const int K = mxGetNumberOfElements(prhs[0]);
	const int D = mxGetNumberOfElements(prhs[3]);

	static std::mt19937 eng;
	static std::uniform_real_distribution<double> unif(0, 1);
	
	double oZnk, tsum, p1bp0, rval;
	const double* W_dk_ptr = W_dk;
	for (int k = 0; k < K; ++k) {
		oZnk = Zn_k[k];
		tsum = 0.0;
		for (int d = 0; d < D; ++d) {
			tsum += W_dk_ptr[d]*(cv1_d[d]+cv2_d[d]*(WZnt_d[d]+(0.5-oZnk)*W_dk_ptr[d]));
		}
		p1bp0 = p1bp0_prior_k[k]*exp(-C*tsum);
		if (nrhs == 11) {
			p1bp0 *= exp(-invlambda*yneta_k[k]*(cc+Znyneta+(0.5-oZnk)*yneta_k[k]));
		}
		if (algtype == 1) {
			rval = unif(eng);
			Zn_k[k] = rval*(1+p1bp0) > 1;
		} else {
			Zn_k[k] = p1bp0 > 1;
		}
		//printf("%.3f\t%.3f\t%d\n", p1bp0, rval, (int)Zn_k[k]);
		double dZnk = Zn_k[k] - oZnk;
		for (int d = 0; d < D; ++d) {
			WZnt_d[d] += dZnk*W_dk_ptr[d];
		}
		if (nrhs == 11) {
			Znyneta += dZnk*yneta_k[k];
		}
		W_dk_ptr += D;
	}
}