#include <assert.h>
#include <math.h>

#include "mex.h"

#define NSTEP 40
// precision up to a default 40-step golden-section search on [0,1]
#define EPSILON 0.000000004370130339181083

class myFunc {
private:
	const bool* const T_rj;
	const double* const theta_r;
	const double* const Lambda_jk;
	const double* const cumsum_elognuj_k;
	const double* const Lnu_k;
	const double* const Lambda_jk_psivit;
	const int Lm1, J, K;
	const double C, l;
	
public:
	int k;
	double psivik;

public:
	myFunc(const bool* const _T_rj, const double* const _theta_r, const double* const _Lambda_jk,
		const double* const _cumsum_elognuj_k, const double* const _Lnu_k, 
		const int _L, const int _J, const int _K, const double _C, const double _l,
		double* const _Lambda_jk_psivit)
		: T_rj(_T_rj), theta_r(_theta_r), Lambda_jk(_Lambda_jk),
		cumsum_elognuj_k(_cumsum_elognuj_k), Lnu_k(_Lnu_k),
		Lm1(_L-1), J(_J), K(_K), C(_C), l(_l),
		Lambda_jk_psivit(_Lambda_jk_psivit) {
	}

	double operator()(double x) const {
		double y = 0.0;
		const double* Lambda_jk_ptr = Lambda_jk + k*J;
		const bool* T_rj_ptr = T_rj;
		double Lambda_jk_psivit_j;
		for (int j = 0; j < J; ++j) {
			Lambda_jk_psivit_j = Lambda_jk_psivit[j] + (x-psivik)*Lambda_jk_ptr[j];
			for (int r = 0; r < Lm1; ++r, ++T_rj_ptr) {
				double tval = l - (*T_rj_ptr ? theta_r[r]-Lambda_jk_psivit_j : Lambda_jk_psivit_j-theta_r[r]);
				if (tval > 0) y += tval;
			}
		}
		return C*y + (log(x)-cumsum_elognuj_k[k])*x + (log(1-x)-Lnu_k[k])*(1-x);
	}
};

double golden(myFunc &fun, double a, double b, int N);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Check inputs & outputs */
	if (nrhs != 10 || nlhs > 0)
		mexErrMsgTxt("Incorrect number of inputs/outputs!");
	for (int i = 0; i < nrhs; ++i)
		if (mxIsSparse(prhs[i]))
			mexErrMsgTxt("Inputs must be full matrices!");

	/* Input */
	const bool* const T_rj = (bool*)mxGetPr(prhs[0]);
	const double* const theta_r = mxGetPr(prhs[1]);
	const double* const Lambda_jk = mxGetPr(prhs[2]);
	const double* const cumsum_elognuj_k = mxGetPr(prhs[3]);
	const double* const Lnu_k = mxGetPr(prhs[4]);
	const double C = mxGetScalar(prhs[5]);
	const double l = mxGetScalar(prhs[6]);
	double* const Lambda_jk_psivit = mxGetPr(prhs[7]);
	double* const psivi = mxGetPr(prhs[8]);
	const double R = mxGetScalar(prhs[9]);

	/* Compute Sizes */
	int L = mxGetN(prhs[1]) + 1;
	int J = mxGetN(prhs[0]);
	int K = mxGetN(prhs[2]);
	
	myFunc fun(T_rj, theta_r, Lambda_jk, cumsum_elognuj_k, Lnu_k, L, J, K, C, l, Lambda_jk_psivit);
	
	const double* Lambda_jk_ptr;
	bool convrg = false;
	for (int r = 0; r < R && !convrg; ++r) {
		convrg = true;
		Lambda_jk_ptr = Lambda_jk;
		for (fun.k = 0; fun.k < K; ++fun.k) {
			fun.psivik = psivi[fun.k];
			psivi[fun.k] = golden(fun, 0, 1, NSTEP);
			register double delta_psivik = psivi[fun.k] - fun.psivik;
			if (delta_psivik > EPSILON) convrg = false;
			for (int j = 0; j < J; ++j) {
				Lambda_jk_psivit[j] += delta_psivik*Lambda_jk_ptr[j];
			}
			Lambda_jk_ptr += J;
		}
	}
}

double golden(myFunc &fun, double a, double b, int N) {
	const double r = 0.381966011250105;
	double s = a + r*(b-a);
	double t = a + b - s;
	double f1 = fun(s);
	double f2 = fun(t);
	for (int i = 0; i < N-1; ++i) {
		if (f1 < f2) {
			b = t;
			t = s;
			f2 = f1;
			s = a + b - t;
			f1 = fun(s);
		} else {
			a = s;
			s = t;
			f1 = f2;
			t = a + b - s;
			f2 = fun(t);
		}
	}
	return (f1 < f2 ? s : t);
}