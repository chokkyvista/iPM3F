#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mex.h"

struct indzcr {
	int	   ind;
	double zcr;
};

static double optsols [2];
static const double inf = mxGetInf();

class regularizer_gradient {
protected:
	const bool paramed_by_slen;

public:
	regularizer_gradient() : paramed_by_slen(false) {}
	regularizer_gradient(bool paramed_by_slen_) : paramed_by_slen(paramed_by_slen_) {}

	virtual void update_params(int n, const double* const w, const double* const d, int K) = 0;
	virtual double rg(const double x) const = 0;
	virtual double invrg(const double y) const = 0;
};

/* 0 */
class rg_null : public regularizer_gradient {
public:
	rg_null() : regularizer_gradient() {}

	rg_null(bool paramed_by_slen_) : regularizer_gradient(paramed_by_slen_) {}

	void update_params(int n, const double* const w, const double* const d, int K) {}

	double rg(const double x) const {
		return 0.0;
	}

	double invrg(const double y) const {
		mexErrMsgTxt("\'invrg\' should never get called when \'lossonly\'!");
		return 0.0;
	}
};

/* 0.5\sum_{k=1}^K w_k^2 */
class rg_svm : public regularizer_gradient {
private:
	double a, b;

public:
	rg_svm() : regularizer_gradient() {}

	rg_svm(bool paramed_by_slen_) : regularizer_gradient(paramed_by_slen_) {}

	void update_params(int n, const double* const w, const double* const d, int K) {
		if (!paramed_by_slen)
			return;

		a = 0.0; b = 0.0;
		for (int k = 0; k < K; ++k) {
			a += d[k]*d[k];
			b += w[k]*d[k];
		}
		assert(a > 0.0);
	}

	double rg(const double x) const {
		return (paramed_by_slen ? a*x+b : x);
	}

	double invrg(const double y) const {
		return (paramed_by_slen ? (y-b)/a : y);
	}
};

/* \sum_{k=1}^K [w_k\log(w_k) + (1-w_k)\log(1-w_k) + a_kw_k + b_k(1-w_k)] */
class rg_psi : public regularizer_gradient {
private:
	const double* const a_k;
	const double* const b_k;
	const int K;

	double c;
	double* const w_k;
	double* const d_k;

	double log_xby1mx(double x) const {
		return (x <= 0.0 ? -inf : (x >= 1.0 ? inf : log(x/(1.0-x))));
	}

public:
	rg_psi(bool paramed_by_slen_, const double* const a_k_, const double* const b_k_, int K_)
		: regularizer_gradient(paramed_by_slen_),
		  a_k(a_k_), b_k(b_k_), K(K_), 
		  w_k((double*)malloc(sizeof(double)*K)), d_k((double*)malloc(sizeof(double)*K)) {
	}

	~rg_psi() {
		free(w_k);
		free(d_k);
	}

	void update_params(int n, const double* const w, const double* const d, int K) {
		if (paramed_by_slen) {
			memcpy(w_k, w, sizeof(double)*K);
			memcpy(d_k, d, sizeof(double)*K);
			c = 0.0;
			for (int k = 0; k < K; ++k) {
				c += d[k]*(a_k[k]-b_k[k]);
			}
		} else {
			c = a_k[n]-b_k[n];
		}
	}

	double rg(const double x) const {
		if (paramed_by_slen) {
			double y = c;
			for (int k = 0; k < K; ++k) {
				y += d_k[k]*log_xby1mx(w_k[k]+x*d_k[k]);
			}
			return y;
		} else {
			return log_xby1mx(x) + c;
		}
	}

	double invrg(const double y) const {
		if (paramed_by_slen) {
			mexErrMsgTxt("No closed form solution for \'invrg\' on \'slen\' in \'rg_psi\'!\n"
				"Please apply explicit rotation to the search directions to enable traditional coordinate descent.");
			return 0.0;
		} else {
			return 1/(1+exp(c-y));
		}
	}
};

void search_shpcr(const double* const tzcrs, const double* const slope, int nzcrs, regularizer_gradient* const rgobj, const double C);
int compare_indzcr(const void *a, const void *b);
int compare_objgl(const void *pkey, const void *pelem);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	/* Check inputs & outputs */
	if (nrhs < 10 || nlhs > 0)
		mexErrMsgTxt("Incorrect number of inputs/outputs!");
	for (int i = 0; i < nrhs; ++i)
		if (mxIsSparse(prhs[i]))
			mexErrMsgTxt("Inputs must be full matrices!");

	/* Input */
	const bool* const T_ri = (bool*)mxGetPr(prhs[0]);
	const double* const X_ki = mxGetPr(prhs[1]);
	const double* const b_ri = mxGetPr(prhs[2]);
	const double C = mxGetScalar(prhs[3]);
	const double l = mxGetScalar(prhs[4]);
	const int rg_type = (int)mxGetScalar(prhs[5]);
	const int iters = (int)mxGetScalar(prhs[6]);
	const int ds_alg = (int)mxGetScalar(prhs[7]);
	double* const d_kn = mxGetPr(prhs[8]);
	double* const w_k = mxGetPr(prhs[9]);
	
	/* Compute Sizes */
	int Lm1 = mxGetM(prhs[0]);
	int I = mxGetN(prhs[0]);
	int K = mxGetM(prhs[1]);
	int N = mxGetN(prhs[8]);
	int nzcrs = mxGetNumberOfElements(prhs[0]);

	bool paramed_by_slen = (mxGetM(prhs[8]) != 1);
	if (! paramed_by_slen) {
		if (N == 1)
			N = (int)mxGetScalar(prhs[8]);
		assert(N <= K);
	} else {
		assert(N <= K+1);
	}

	double *slen_ptr = NULL;
	bool powell, rosenbrock;
	switch (ds_alg) {
	case 0:
		powell = rosenbrock = false;
		break;
	case 1:
		powell = true; rosenbrock = false;
		assert(paramed_by_slen);
		if (N != K+1)
			mexErrMsgTxt("\'d_kn\' should be of size K*(K+1) for Powell method!\n"
			"(last column buffering the newly generated conjugate direction)");
		break;
	case 2:
		powell = false; rosenbrock = true;
		if (paramed_by_slen) {
			if (--N != K)
				mexErrMsgTxt("\'d_kn\' should be of size K*(K+1) for step-length-parameterized Rosenbrock method!\n"
					"(last column buffering the step lengths)");
			slen_ptr = d_kn + K*K;
		} else {
			if (N != K)
				mexErrMsgTxt("\'d_kn\' should be of size 1*K for standard-orthogonal-basised Rosenbrock method!\n"
				"(buffering the step lengths)");
			slen_ptr = d_kn;
		}
		memset(slen_ptr, 0, sizeof(double)*K);
		break;
	default:
		mexErrMsgTxt("Unknown direct search algorithm!");
	}
	assert(!powell || !rosenbrock);
	
	regularizer_gradient *rgobj = NULL;
	switch (rg_type) {
	case 0: // rg_null
		rgobj = new rg_null(paramed_by_slen);
		break;
	case 1: // rg_svm
		rgobj = new rg_svm(paramed_by_slen);
		break;
	case 2: // rg_psi
		assert(nrhs == 12);
		rgobj = new rg_psi(paramed_by_slen, mxGetPr(prhs[10]), mxGetPr(prhs[11]), K);
		break;
	default:
		mexErrMsgTxt("Unknown type of regularizer!");
	}

	double* const slope = (double*)malloc(sizeof(double)*nzcrs);
	double* const tzcrs = (double*)malloc(sizeof(double)*nzcrs);
	const double *X_ki_ptr = X_ki;
	double wXi;
	double* const w0_k = (double*)malloc(sizeof(double)*K);
	for (int i = 0, ri = 0; i < I; ++i) {
		wXi = 0.0;
		for (int k = 0; k < K; ++k) {
			wXi += w_k[k]*X_ki_ptr[k];
		}
		for (int r = 0; r < Lm1; ++r, ++ri) {
			tzcrs[ri] =  b_ri[ri] - (T_ri[ri] ? wXi : -wXi) - l;
		}
		X_ki_ptr += K;
	}

	for (int iter = 0; iter < iters; ++iter) {
		double *d_kn_ptr = d_kn;
		double dtnXi;
		double sol;
		memcpy(w0_k, w_k, sizeof(double)*K);
		for (int n = 0; n < N; ++n) {
			X_ki_ptr = X_ki;
			for (int i = 0, ri = 0; i < I; ++i) {
				if (paramed_by_slen) {
					dtnXi = 0.0;
					for (int k = 0; k < K; ++k) {
						dtnXi += d_kn_ptr[k]*X_ki_ptr[k];
					}
					for (int r = 0; r < Lm1; ++r, ++ri) {
						slope[ri] = (T_ri[ri] ? dtnXi : -dtnXi);
					}
				} else {
					for (int r = 0; r < Lm1; ++r, ++ri) {
						slope[ri] = (T_ri[ri] ? X_ki_ptr[n] : -X_ki_ptr[n]);
						tzcrs[ri] += w_k[n]*slope[ri];
					}
				}
				X_ki_ptr += K;
			}

			rgobj->update_params(n, w_k, d_kn_ptr, K);
			search_shpcr(tzcrs, slope, nzcrs, rgobj, C);

			sol = 0.5*(optsols[0] + optsols[1]);
			if (rg_type == 0) {
				if (optsols[1] == inf) sol = optsols[0];
				else if (optsols[0] = -inf) sol = optsols[1];
			}

			if (paramed_by_slen) {
				if (rosenbrock)
					slen_ptr[n] += sol;
				for (int k = 0; k < K; ++k) {
					w_k[k] += sol*d_kn_ptr[k];
				}
				for (int ri = 0; ri < nzcrs; ++ri) {
					tzcrs[ri] -= sol*slope[ri];
				}
				d_kn_ptr += K;
				if (powell && n==K-1) {
					double norm = 0.0;
					for (int k = 0; k < K; ++k) {
						d_kn_ptr[k] = w_k[k] - w0_k[k];
						norm += d_kn_ptr[k]*d_kn_ptr[k];
					}
					norm = sqrt(norm);
					for (int k = 0; k < K; ++k) {
						d_kn_ptr[k] /= norm;
					}
				}
			} else {
				w_k[n] = sol;
				if (rosenbrock)
					slen_ptr[n] += w_k[n] - w0_k[n];
				for (int ri = 0; ri < nzcrs; ++ri) {
					tzcrs[ri] -= w_k[n]*slope[ri];
				}
			}
		}
		if (powell) {
			memcpy(d_kn, d_kn+K, sizeof(double)*K*(K-1));
			memcpy(d_kn_ptr-(K<<1), d_kn_ptr-K, sizeof(double)*K);
		}
	}

	delete rgobj;
	free(slope);
	free(tzcrs);
	free(w0_k);
}

void search_shpcr(const double* const tzcrs, const double* const slope, int nzcrs, regularizer_gradient* const rgobj, const double C) {
	bool lossonly = (dynamic_cast<rg_null* const>(rgobj) != NULL);

	int ci = 0;
	double shg_lm = 0.0;
	for (int i = 0; i < nzcrs; ++i) {
		if (slope[i] != 0) {
			ci += 1;
			if (slope[i] < 0) {
				shg_lm += slope[i];
			}
		}
	}
	
	indzcr *indzcrs = (indzcr*)malloc(sizeof(indzcr)*ci);
	ci = 0;
	for (int i = 0; i < nzcrs; ++i) {
		if (slope[i] != 0) {
			indzcrs[ci].ind = i;
			indzcrs[ci].zcr = tzcrs[i]/slope[i];
			ci += 1;
		}
	}
	nzcrs = ci;

	qsort(indzcrs, nzcrs, sizeof(indzcr), compare_indzcr);

	double* const shg = (double*)malloc(sizeof(double)*(nzcrs+1));
	shg[0] = shg_lm;
	for (int i = 0; i < nzcrs; ++i) {
		shg[i+1] = shg[i] + fabs(slope[indzcrs[i].ind]);
	}

	double objgl = C*shg[0] + rgobj->rg(-inf);
	if (objgl > 0) {
		assert(!lossonly);
		optsols[0] = optsols[1] = -inf;
	} else if (objgl == 0) {
		optsols[0] = -inf;
		optsols[1] = (lossonly ? indzcrs[0].zcr : optsols[0]);
	} else if ((objgl=C*shg[nzcrs] + rgobj->rg(indzcrs[nzcrs-1].zcr)) == 0) {
		optsols[0] = indzcrs[nzcrs-1].zcr;
		optsols[1] = (lossonly ? inf : optsols[0]);
	} else if (objgl < 0) {
		assert(!lossonly);
		optsols[0] = optsols[1] = rgobj->invrg(-C*shg[nzcrs]);
	} else {
		/* search for the last position where objgl<=0 */
		/*
		int k = 1;
		do {
			objgl = C*shg[k] + rgobj->rg(indzcrs[k-1].zcr);
		} while (objgl <= 0 && ++k <= nzcrs);
		--k;
		//*/
		//* continually narrow search until just one element remains
		int imin = 0, imax = nzcrs, imid;
		while (imin < imax) {
			imid = (imin + imax + 1) >> 1;
			// reduce the search
			if ((objgl=C*shg[imid]+rgobj->rg(indzcrs[imid-1].zcr)) <= 0)
				imin = imid;
			else
				imax = imid - 1;
		}
		assert(imin == imax);
		int k = imin;
		//*/
		if (lossonly) {
			if (C*shg[k]+rgobj->rg(indzcrs[k-1].zcr) < 0) {
				optsols[0] = optsols[1] = indzcrs[k].zcr;
			} else {
				optsols[1] = indzcrs[k].zcr;
				if (k > 0) {
					optsols[0] = indzcrs[k-1].zcr;
				} else {
					optsols[0] = -inf;
				}
			}
		} else if (C*shg[k]+rgobj->rg(indzcrs[k].zcr) <= 0) {
			optsols[0] = optsols[1] = indzcrs[k].zcr;
		} else {
			optsols[0] = optsols[1] = rgobj->invrg(-C*shg[k]);
		}
	}

	free(indzcrs);
	free(shg);

	return;
}

int compare_indzcr(const void *a, const void *b) {
	double diff = ((indzcr*)a)->zcr - ((indzcr*)b)->zcr;
	return (diff < 0 ? -1 : (diff > 0 ? 1 : 0));
}

int compare_objgl(const void *pkey, const void *pelem) {
	return (*(double*)pelem > 0 ? 0 : 1);
}
