/* A specific mex wrapper that solves the conditional SVMs
 * in M3F problems via SVM_Multiclass
 * 
 * Written by Minjie Xu (chokkyvista06@gmail.com)
 */

#include <assert.h>

#include "mex.h"

#include "../SVM_Multiclass/svm_struct_api.h"
#include "../SVM_Multiclass/svm_struct_learn.h"
#include "../SVM_Multiclass/svm_struct_common.h"
#include "../SVMLight/svm_common.h"

double m3fviasvm(SAMPLE &sample, const double C, const double epsilon, const int K, const double margin, double *U);

void reset_hideo_globals();

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Check inputs & outputs */
	if (nrhs < 6 || nlhs > 0)
		mexErrMsgTxt("Incorrect number of inputs/outputs!");
	if (!mxIsClass(prhs[0], "logical") || mxIsSparse(prhs[0]))
		mexErrMsgTxt("T_rj must be a full logical array!");
	
	/* Input */
	const bool* const T_rj = (bool*)mxGetPr(prhs[0]);
	double *U = mxGetPr(prhs[1]);
	const double* const Vt_j = mxGetPr(prhs[2]);
	const double* const theta_rj = mxGetPr(prhs[3]);
	const double C = mxGetScalar(prhs[4]);
	const double l = mxGetScalar(prhs[5]);
	const double reps = (nrhs >= 7 ? mxGetScalar(prhs[6]) : 1.0);
	const double cfObj = (nrhs >= 8 ? mxGetScalar(prhs[7]) : INT_MAX);
	
	/* Compute Sizes */
	int Lm1, ndocs, K;
	K = mxGetN(prhs[1]);
	ndocs = mxGetN(prhs[2]);
	Lm1 = mxGetM(prhs[3]);
	assert(mxGetM(prhs[0]) == Lm1);
	assert(mxGetN(prhs[0]) == ndocs);
	assert(mxGetM(prhs[2]) == K);
	assert(mxGetN(prhs[3]) == ndocs);
	
	/* Format into Samples */
	SAMPLE sample;
	sample.n = mxGetNumberOfElements(prhs[0]);
	EXAMPLE *examples = (EXAMPLE *)my_malloc(sizeof(EXAMPLE)*sample.n);
	sample.examples = examples;

	long queryid = 0;
	long slackid = 0;
	double costfactor = 1;
	double minnzero = 1e-10;
	char *comment = NULL;
	SVECTOR *fvec = NULL;
	long i = 0;
	double *rawvec = create_nvector(1+K);
	for (int j = 0; j < ndocs; ++j) {
		memcpy(rawvec+2, Vt_j+j*K, K*sizeof(double));
		for(int r = 0; r < Lm1; ++r) {
			rawvec[1] = theta_rj[i];
			/* make sure 'theta' gets reserved in the svector 'fvec' */
			if (rawvec[1] == 0) rawvec[1] = minnzero;
			fvec = create_svector_n(rawvec, 1+K, comment, 1.0); // +1 for 'theta_ir'
			examples[i].x.doc = create_example(i, queryid, slackid, costfactor, fvec);
			examples[i].y.classlabel = (T_rj[i] ? 1 : 2);
			examples[i].y.scores = NULL;
			examples[i].y.num_classes = 2;
			++i;
		}
	}
	free(rawvec);

	struct_verbosity = -1; //4;
	verbosity = -1; //5;

	double *wptr = (double *)my_malloc(sizeof(double)*K);
	double epsilon = DEFAULT_EPS*reps;
	double nfObj = m3fviasvm(sample, C, epsilon, K, l, wptr);
	/*
	while (nfObj >= cfObj && epsilon >= 0.02) {
		epsilon = epsilon / 2;
		nfObj = m3fviasvm(sample, C, epsilon, K, wptr);
	}
	//*/
	free_struct_sample(sample);

//	if (nfObj < cfObj)
	memcpy(U, wptr, K*sizeof(double));
	free(wptr);
}

double m3fviasvm(SAMPLE &sample, const double C, const double epsilon, const int K, const double margin, 
                 double *U) {
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	STRUCT_LEARN_PARM struct_parm;
	STRUCTMODEL structmodel;

	/* set the parameters. */
	set_learning_defaults(&learn_parm, &kernel_parm);
	learn_parm.maxiter = 500;

	struct_parm.C = C;
	struct_parm.slack_norm = 1;
	struct_parm.epsilon = epsilon;
	struct_parm.custom_argc = 0;
	struct_parm.loss_function = 2; /* specific loss for bilinear SVM optimization */
	struct_parm.loss_type = DEFAULT_RESCALING;
	struct_parm.newconstretrain = 100;
	struct_parm.ccache_size = 5;
	struct_parm.batch_size = 100;
	struct_parm.delta_ell = 1;
	struct_parm.hinge_margin = margin;

	struct_parm.C = struct_parm.C * sample.n;
	//* reset global vars in SVMLight/svm_hideo.cpp
	reset_hideo_globals();
	//*/

	svm_learn_struct_joint(sample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, ONESLACK_PRIMAL_ALG);

	/*
	double wbuffer[100];
	memcpy(wbuffer, structmodel.w+2, sizeof(double)*(structmodel.sizePsi-1));
	//*/
	memset(U, 0, sizeof(double)*K);
	memcpy(U, structmodel.w+2, sizeof(double)*(structmodel.sizePsi-1));

	/*
	plhs[0] = mxCreateDoubleMatrix(1, K, mxREAL);
	double *newU = mxGetPr(plhs[0]);
	// memcpy(newU, structmodel.w+2, sizeof(double)*(structmodel.sizePsi-1));
	memcpy(newU, wbuffer, sizeof(wbuffer));
	//*/

	double nfObj = structmodel.primalobj;
	free_struct_model(structmodel);
	return nfObj;
}
