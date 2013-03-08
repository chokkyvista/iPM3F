// SVMStruct_CPlus.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

/***********************************************************************/
/*                                                                     */
/*   svm_struct_main.c                                                 */
/*                                                                     */
/*   Command line interface to the alignment learning module of the    */
/*   Support Vector Machine.                                           */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/


/* the following enables you to use svm-learn out of C++ */
//#ifdef __cplusplus
//extern "C" {
//#endif
#include "../SVMLight/svm_common.h"
#include "../SVMLight/svm_learn.h"
//#ifdef __cplusplus
//}
//#endif
# include "svm_struct_learn.h"
# include "svm_struct_classify.h"
# include "svm_struct_common.h"
# include "svm_struct_api.h"

#include <stdio.h>
#include <string>
#include <assert.h>
#include <windows.h>
#include <fstream>
#include <queue>
using namespace std;
/* } */

char trainfile[200];           /* file with training examples */
char modelfile[200];           /* file for resulting classifier */
bool g_bClassify;
bool g_bCrossValidation;
bool g_bInnerCV;
bool g_bWarmStart;
int g_nStopStep;			  /* the number of iterations to early stop. */
string g_strTrainDataDir;
int g_nDim;
int g_nStateNum;
int g_nFoldNum;

void eval_CV(vector<wstring>&vecFileName, const int &nFoldNum, 
			 STRUCTMODEL &structmodel,
			 STRUCT_LEARN_PARM &struct_parm,
			 LEARN_PARM &learn_parm, 
			 KERNEL_PARM &kernel_parm, 
			 int alg_type, ofstream &ofs);
void eval_Ratio(vector<wstring>&vecFileName, vector<int> &vecRatio, 
				STRUCTMODEL &structmodel,
				STRUCT_LEARN_PARM &struct_parm,
				LEARN_PARM &learn_parm, 
				KERNEL_PARM &kernel_parm, 
				int alg_type, ofstream &ofs);
double inner_CV(SAMPLE *sample, STRUCTMODEL &structmodel,
			 STRUCT_LEARN_PARM &struct_parm,
			 LEARN_PARM &learn_parm, 
			 KERNEL_PARM &kernel_parm, 
			 int alg_type, ofstream &ofs);
void   read_input_parameters(int, char **, char *, char *,long *, long *,
							 STRUCT_LEARN_PARM *, LEARN_PARM *, KERNEL_PARM *, int *);
void   wait_any_key();
void   print_help();

void partitionData(SAMPLE *sample, const int &nFoldNum, vector<int> &vecSampleNum)
{
	if ( sample->examples[0].m_nFoldIx == -1 ) {
		vecSampleNum.assign( nFoldNum, 0 );
		int nUnitNum = sample->n / nFoldNum;
		for ( int i=0; i<nUnitNum*nFoldNum; i++ ) {
			sample->examples[i].m_nFoldIx = i / nUnitNum;
			vecSampleNum[sample->examples[i].m_nFoldIx] ++;
		}
		for ( int i=nUnitNum*nFoldNum; i<sample->n; i++ ) {
			sample->examples[i].m_nFoldIx = nFoldNum - 1;
			vecSampleNum[nFoldNum-1] ++;
		}
	}
}

SAMPLE* get_traindata(SAMPLE* c, const int&nfold, const int &foldix)
{
	int nunit = c->n / nfold;

	SAMPLE *subc = (SAMPLE*)malloc(sizeof(SAMPLE));
	subc->examples = 0;
	subc->n = 0;

	int nd = 0;
	for ( int i=0; i<c->n; i++ )
	{
		if ( foldix < nfold ) {
			if ( (i >= (foldix-1)*nunit) && ( i < foldix*nunit ) ) continue;
		} else {
			if ( i >= (foldix-1) * nunit ) continue;
		}

		subc->examples = (EXAMPLE*) realloc(subc->examples, sizeof(EXAMPLE)*(nd+1));
		subc->examples[nd] = c->examples[i];

		nd++;
	}
	subc->n = nd;
	return subc;
}

SAMPLE* get_testdata(SAMPLE* c, const int&nfold, const int &foldix)
{
	int nunit = c->n / nfold;

	SAMPLE *subc = (SAMPLE*)malloc(sizeof(SAMPLE));
	subc->examples = 0;
	subc->n = 0;

	int nd = 0, nw = 0;
	for ( int i=0; i<c->n; i++ )
	{
		if ( foldix < nfold ) {
			if ( i < ((foldix-1)*nunit) || i >= foldix*nunit ) continue;
		} else {
			if ( i < (foldix-1) * nunit ) continue;
		}

		subc->examples = (EXAMPLE*) realloc(subc->examples, sizeof(EXAMPLE)*(nd+1));
		subc->examples[nd] = c->examples[i];
		
		nd++;
	}
	subc->n = nd;
	return subc;
}

void reorder(SAMPLE* sample, char *filename)
{
	int num, ix=0;
	int *order = (int*)malloc(sizeof(int)*sample->n);
	FILE *fileptr = fopen(filename, "r");
	while ( (fscanf(fileptr, "%10d", &num) != EOF ) ) {
		order[ix] = num;
		ix ++;
	}
	
	EXAMPLE *docs = (EXAMPLE*)malloc(sizeof(EXAMPLE) * sample->n);
	for ( int i=0; i<sample->n; i++ )
		docs[i] = sample->examples[i];
	for ( int i=0; i<c->num_docs; i++ )
		sample->examples[i] = docs[order[i]];
	free(docs);
	free(order);
}


double evaluation(SAMPLE *testsample, STRUCTMODEL &model, ofstream &ofs, bool bWrite)
{
	double avgloss = 0;
	for( int i=0; i<testsample->n; i++) 
	{
		LABEL y = classify_struct_example(testsample->examples[i].x, &model, NULL);
		double l = loss(testsample->examples[i].y, y, NULL);

		if ( bWrite ) {
			ofs << y.classlabel << "\t" << testsample->examples[i].y.classlabel << endl;
		}

		avgloss += l;
		/*if(l == 0) correct++;
		else incorrect++;*/
		//eval_prediction(i, testsample.examples[i], y, &model, &sparm, &teststats);

		if(empty_label(testsample->examples[i].y)) 
		//{ no_accuracy=1; } /* test data is not labeled */
		if(verbosity>=2) {
			if((i+1) % 100 == 0) {
				printf("%ld..",i+1); fflush(stdout);
			}
		}
		free_label(y);
	}

	avgloss /= testsample.n;
	return avgloss;
}
void loadTrainScheme(vector<int> &trainScheme, char *pFileName)
{
	int dTrainingDataRatio;
	char buff[256];
	int index = 0;

	ifstream ifs;
	ifs.open(pFileName, ios_base::in);
	if ( !ifs.is_open() ) { exit(0);}

	while ( !ifs.fail() ) {
		ifs.getline( buff, 256 );
		string str(buff);

		dTrainingDataRatio = atoi( str.c_str() );
		if ( dTrainingDataRatio <= 0 ) continue;

		trainScheme.push_back(dTrainingDataRatio);
	}
	ifs.close();
}
void loadConfig(const char* pFileName)
{
	g_bClassify = false;
	g_bCrossValidation = false;
	g_bInnerCV = false;
	g_bWarmStart = false;
	g_nStopStep = 200;
	ifstream ifs(pFileName, ios_base::in);
	if ( !ifs.is_open() ) return;
	char buff[512];
	while ( !ifs.eof() ) {
		ifs.getline(buff, 512);
		string str(buff);
		if ( str.empty() || str.find("%%") != str.npos ) continue;

		if ( str.find("classify") != str.npos) {
			g_bClassify = true;
		} else if ( str.compare("Cross-Validation") == 0 ) {
			g_bCrossValidation = true;
		} else if ( str.compare("Inner-CV") == 0 ) {
			g_bInnerCV = true;
		} else if ( str.compare("Warm-Start") == 0 ) {
			g_bWarmStart = true;
		} else;

		size_t stPos = str.find("=");
		if ( str.find("TrainDataDir") != str.npos ) {
			g_strTrainDataDir = str.substr(stPos+2, str.size()-stPos-2);
		} else if ( str.find("Total Dimension") != str.npos ) {
			g_nDim = atoi(str.substr(stPos+2, str.size()-stPos-2).c_str());
		} else if (str.find("State Number") != str.npos ) {
			g_nStateNum = atoi(str.substr(stPos+2, str.size()-stPos-2).c_str());
		} else if ( str.find("FoldNumber") != str.npos ) {
			g_nFoldNum = atoi(str.substr(stPos+2, str.size()-stPos-2).c_str());
		} else if ( str.find("EarlyStopStep") != str.npos ) {
			g_nStopStep = atoi(str.substr(stPos+2, str.size()-stPos-2).c_str());
		} else ;
	}
	ifs.close();
}
wstring UTF82WChar(const BYTE * pszSrc, int nLen)
{
	int nSize = MultiByteToWideChar(CP_UTF8,
		0, 
		(LPCSTR)pszSrc,
		nLen,
		0,
		0);
	if (nSize <= 0) 
	{
		return L"";
	}
	WCHAR * pwszDst	= new WCHAR[nSize + 1];
	if (NULL == pwszDst) 
	{
		return L"";
	}
	MultiByteToWideChar(CP_UTF8,
		0,
		(LPCSTR) pszSrc,
		nLen,
		pwszDst,
		nSize);
	pwszDst[nSize] = L'\0';
	// skip 0xfeff
	if (pwszDst[0] == 0xFEFF)			//skip UTF8 head if needed
	{
		for (int i = 0; i < nSize; i++)
		{
			pwszDst[i] = pwszDst[i + 1];
		}
	}

	wstring wstrTemp = pwszDst;
	delete[] pwszDst;

	return wstrTemp;
}

string WChar2Ansi(LPCWSTR pwszSrc)
{
	int nLen = WideCharToMultiByte(CP_ACP, 
		0,
		pwszSrc,
		-1,
		NULL, 
		0,
		NULL,
		NULL);
	if (nLen<= 0)
	{
		return NULL;
	}
	char* pszDst = new char[nLen];
	if (NULL == pszDst)
	{
		return NULL;
	}
	WideCharToMultiByte(CP_ACP, 
		0,
		pwszSrc,
		-1,
		pszDst,
		nLen,
		NULL,
		NULL);
	pszDst[nLen -1] = 0;

	string strTemp = pszDst;
	delete [] pszDst;

	return strTemp;
}
int TraverseDirectory(IN const wstring &Dir,IN const wstring &Suffix, OUT vector<wstring> &vecFileName)
{
	if ( !vecFileName.empty() ) vecFileName.resize(0);

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	queue<wstring> queDir;
	queDir.push(Dir);

	int iCount = 0;
	wchar_t szExt[MAX_PATH];
	//while ( queDir.empty() != true )
	//{
		wstring strCurrDir = queDir.front();
		wprintf(L"%s\n", strCurrDir.c_str());
		wstring strFileName = strCurrDir + wstring(L"\\*.*");;
		wprintf(L"%s\n\n", strFileName.c_str());

		wcscpy(FindFileData.cFileName, strFileName.c_str());

		hFind = FindFirstFile(strFileName.c_str(), &FindFileData);
		if ( hFind != INVALID_HANDLE_VALUE )
		{
			if ( ( FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FALSE 
				&& wcscmp(FindFileData.cFileName, L".") != 0
				&& wcscmp(FindFileData.cFileName, L"..") != 0
				)
			{
				//wprintf(L"Find a Directory: %s\n", FindFileData.cFileName);
				//queDir.push(strCurrDir + wstring(L"\\") + wstring(FindFileData.cFileName));
			}
			else
			{
				_wsplitpath(FindFileData.cFileName, NULL, NULL, NULL, szExt);
				strFileName = strCurrDir + wstring(L"\\") + wstring(FindFileData.cFileName);

				if ( wcscmp(szExt+1, Suffix.c_str()) == 0 /*
					&& wcscmp(FindFileData.cFileName, L".") != 0
					&& wcscmp(FindFileData.cFileName, L"..") != 0*/
					)
				{
					iCount ++;
					wstring strFullName = strCurrDir + wstring(L"\\") + wstring(FindFileData.cFileName);
					vecFileName.push_back(strFullName);
				}
			}
			BOOL bRet = FindNextFile(hFind,&FindFileData);
			while ( bRet == TRUE ){

				if ( ( bRet == TRUE 
					&& FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FALSE 
					&& wcscmp(FindFileData.cFileName, L".") != 0
					&& wcscmp(FindFileData.cFileName, L"..") != 0
					)
				{
					//wprintf(L"Find a Directory: %s\n", FindFileData.cFileName);
					//queDir.push(strCurrDir + wstring(L"\\") + wstring(FindFileData.cFileName));
				}
				else
				{
					_wsplitpath(FindFileData.cFileName,NULL,NULL,NULL,szExt);
					strFileName = strCurrDir + wstring(L"\\") + wstring(FindFileData.cFileName);

					if ( wcscmp(szExt+1, Suffix.c_str()) == 0 /*
						&& wcscmp(FindFileData.cFileName, L".") != 0
						&& wcscmp(FindFileData.cFileName, L"..") != 0*/
						)
					{
						iCount ++;
						wstring strFullName = strCurrDir + wstring(L"\\") + wstring(FindFileData.cFileName);
						vecFileName.push_back(strFullName);
					}
				}
				bRet = FindNextFile(hFind, &FindFileData);
			}
		}
		queDir.pop();
	//}

	return ERROR_SUCCESS;
}


int main (int argc, char* argv[])
{  
	SAMPLE sample;  /* training sample */
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;
	STRUCT_LEARN_PARM struct_parm;
	STRUCTMODEL structmodel;
	int alg_type;

	LABEL *pLabel = new LABEL();
	LABEL lb;
	lb.m_labels = NULL;
	EXAMPLE *pEx = new EXAMPLE();
	pEx->y.scores = NULL;


	svm_struct_learn_api_init(argc,argv);

	read_input_parameters(argc, argv, trainfile, modelfile, &verbosity,
		&struct_verbosity, &struct_parm, &learn_parm, &kernel_parm,&alg_type);

	loadConfig(learn_parm.addConfigFile);
	if ( g_bClassify ) { classify(argc, argv); return 0; }


	/* do evaluation */
	vector<wstring> vecFileName;
	TraverseDirectory(UTF82WChar((unsigned char *)g_strTrainDataDir.c_str(), g_strTrainDataDir.size()), L"data", vecFileName);

	ofstream ofs("evRes.txt", ios_base::out | ios_base::app);
	ofs.seekp(ios_base::end);
	if ( learn_parm.m_bQPSolver ) ofs << "L2-norm penalty" << endl;
	else ofs << "LP Solver for L1-norm penalty" << endl;
	ofs << "\t -- C: " << struct_parm.C;
	if ( g_bCrossValidation ) {
		ofs << "\tCV: " << g_nFoldNum << endl;
		eval_CV(vecFileName, g_nFoldNum, structmodel, struct_parm, learn_parm, kernel_parm, alg_type, ofs);
	} else {
		ofs << "\tSampling" << endl;
		vector<int> vecRatio;
		loadTrainScheme(vecRatio, "trainScheme.txt");

		eval_Ratio(vecFileName, vecRatio, structmodel, struct_parm, learn_parm, kernel_parm, alg_type, ofs);
	}

	ofs.close();

	svm_struct_learn_api_exit();

	return 0;
}

void eval_CV(vector<wstring> &vecFileName, const int &nFoldNum, 
			 STRUCTMODEL &structmodel,
			 STRUCT_LEARN_PARM &struct_parm,
			 LEARN_PARM &learn_parm, 
			 KERNEL_PARM &kernel_parm, 
			 int alg_type, ofstream &ofs)
{
	if ( vecFileName.empty() ) return;

	vector<double> totalAvg( vecFileName.size() );
	vector<double> totalVar( vecFileName.size() );
	vector<double> arryAvgLoss;
	for ( int fIx=0; fIx<vecFileName.size(); fIx++ ) 
	{
		arryAvgLoss.assign(nFoldNum, 0);
		strcpy(trainfile, WChar2Ansi(vecFileName[fIx].c_str()).c_str());
		ofs << "   ****** file: " << trainfile << " ********" << endl;
		if(struct_verbosity >=1 ) { printf("Reading training examples..."); fflush(stdout); }

		/* read the training examples */
		SAMPLE sample = read_struct_examples(trainfile, &struct_parm);
		if(struct_verbosity>=1) { printf("done\n"); fflush(stdout); }

		// 10 fold CV for the dataset
		vector<int> vecSampleNum;
		partitionData(&sample, nFoldNum, vecSampleNum);

		// for each partition of the data
		for ( int it=0; it<nFoldNum; it++ )
		{
			ofs << "\tfold: " << it;
			//char buff[512];
			sprintf(modelfile, "parm_f%d_cv%d.txt", fIx, it+1);

			// split the data into training & testing sets
			SAMPLE trSample, tsSample;
			trSample.n = vecSampleNum[it];
			trSample.examples = (EXAMPLE*)malloc( sizeof(EXAMPLE) * vecSampleNum[it] );
			tsSample.n = sample.n - vecSampleNum[it];
			tsSample.examples = (EXAMPLE*)malloc( sizeof(EXAMPLE) * tsSample.n );
			int trIx=0, tsIx=0;
			for ( int k=0; k<sample.n; k++ ) {
				if ( it == sample.examples[k].m_nFoldIx ) {
					trSample.examples[trIx].x.SetHead( sample.examples[k].x.m_pHeadNode );
					trSample.examples[trIx].x.m_pEndNode = sample.examples[k].x.m_pEndNode;
					trSample.examples[trIx].y.m_nSize = sample.examples[k].y.m_nSize;
					trSample.examples[trIx ++].y.m_labels = sample.examples[k].y.m_labels;
				} else {
					tsSample.examples[tsIx].x.SetHead( sample.examples[k].x.m_pHeadNode );
					tsSample.examples[tsIx].x.m_pEndNode = sample.examples[k].x.m_pEndNode;
					tsSample.examples[tsIx].y.m_nSize = sample.examples[k].y.m_nSize;
					tsSample.examples[tsIx ++].y.m_labels = sample.examples[k].y.m_labels;
				}
			}

			if ( learn_parm.m_bInnerCV ) {
				struct_parm.C = inner_CV( &trSample, structmodel, 
					struct_parm, learn_parm, kernel_parm, alg_type, ofs);
			}

			/* Do the learning and return structmodel. */
			long runtime_start, runtime_end;
			runtime_start = get_runtime();

			bool bSucc = true;
			if(alg_type == 1) {
				if ( learn_parm.m_bQPSolver )
					svm_learn_struct(trSample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
				else 
					bSucc = svm_learn_struct_lp(trSample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, g_nStopStep, true);
			} else exit(1);
			runtime_end = get_runtime();

			/* Warning: The model contains references to the original data 'docs'.
			If you want to free the original data, and only keep the model, you 
			have to make a deep copy of 'model'. */
			if(struct_verbosity>=1) { printf("Writing learned model...");fflush(stdout);}
			write_struct_model(modelfile, &structmodel, &struct_parm);
			if(struct_verbosity>=1) {printf("done\n");fflush(stdout); }

			// do classification on the testing data
			arryAvgLoss[it] = evaluation(&tsSample, structmodel, ofs, false);
			ofs << "\tAvgLoss: " << arryAvgLoss[it] << "  [" << bSucc << "]" 
				<< "\tCPU Seconds: " << (runtime_end-runtime_start)/100.0 << endl;

			// free memory
			for ( int k=0; k<trSample.n; k++ ) {
				trSample.examples[k].x.SetHead(NULL);
				trSample.examples[k].y.m_labels = NULL;
			}
			for ( int k=0; k<tsSample.n; k++ ) {
				tsSample.examples[k].x.SetHead(NULL);
				tsSample.examples[k].y.m_labels = NULL;
			}
			free_struct_sample(trSample);
			free_struct_sample(tsSample);
			free_struct_model(structmodel);
		}

		free_struct_sample(sample);


		ofs << "\tAverage Loss: ";
		totalAvg[fIx] = 0;
		for ( int i=0; i<nFoldNum; i++ ) totalAvg[fIx] += arryAvgLoss[i] / nFoldNum;
		totalVar[fIx] = 0;
		for ( int i=0; i<nFoldNum; i++ ) totalVar[fIx] += (arryAvgLoss[i] - totalAvg[fIx]) * (arryAvgLoss[i] - totalAvg[fIx]) / nFoldNum;
		totalVar[fIx] = sqrt( totalVar[fIx] );
		ofs << "\t  Avg: " << totalAvg[fIx] << "\tVar: " << totalVar[fIx] << endl;
	}

	ofs << "\tTotal Avg over " << vecFileName.size() << " files: " << endl;
	double dAvgVal = 0, dAvgVar = 0;
	for ( int i=0; i<vecFileName.size(); i++ ) {
		dAvgVal += totalAvg[i] / vecFileName.size();
		dAvgVar += totalVar[i] / vecFileName.size();
	}
	ofs << "\t  Avg: " << dAvgVal << "\tVar: " << dAvgVar << endl;
}

void eval_Ratio(vector<wstring>&vecFileName, vector<int> &vecRatio, 
				STRUCTMODEL &structmodel,
				STRUCT_LEARN_PARM &struct_parm,
				LEARN_PARM &learn_parm, 
				KERNEL_PARM &kernel_parm, 
				int alg_type, ofstream &ofs)
{
	vector<vector<double> > arryAvgLoss( vecFileName.size() );
	for ( int db=0; db<vecFileName.size(); db++ )
	{
		strcpy(trainfile, WChar2Ansi(vecFileName[db].c_str()).c_str());
		ofs << "   ****** file: " << trainfile << " ********" << endl;
		if(struct_verbosity >=1 ) { printf("Reading training examples..."); fflush(stdout); }

		/* read the training examples */
		SAMPLE sample = read_struct_examples(trainfile,&struct_parm);
		if(struct_verbosity>=1) { printf("done\n"); fflush(stdout); }

		arryAvgLoss[db].assign(vecRatio.size(), 0);
		// for each partition of the data
		for ( int it=0; it<vecRatio.size(); it++ )
		{
			ofs << "\tRatio: " << vecRatio[it] << endl;
			// split the data into training & testing sets
			int nTrainSampleNum = sample.n * vecRatio[it] / 100;
			SAMPLE trSample;
			trSample.n = nTrainSampleNum;
			trSample.examples = (EXAMPLE*)malloc( sizeof(EXAMPLE) * nTrainSampleNum );
			for ( int k=0; k<nTrainSampleNum; k++ ) {
				trSample.examples[k].x.SetHead( sample.examples[k].x.m_pHeadNode );
				trSample.examples[k].x.m_pEndNode = sample.examples[k].x.m_pEndNode;
				trSample.examples[k].y.m_nSize = sample.examples[k].y.m_nSize;
				trSample.examples[k].y.m_labels = sample.examples[k].y.m_labels;
			}
			SAMPLE tsSample;
			tsSample.n = sample.n - nTrainSampleNum;
			tsSample.examples = (EXAMPLE*)malloc( sizeof(EXAMPLE) * tsSample.n );
			for ( int k=0; k<tsSample.n; k++ ) {
				tsSample.examples[k].x.SetHead( sample.examples[nTrainSampleNum+k].x.m_pHeadNode );
				tsSample.examples[k].x.m_pEndNode = sample.examples[nTrainSampleNum+k].x.m_pEndNode;
				tsSample.examples[k].y.m_nSize = sample.examples[nTrainSampleNum+k].y.m_nSize;
				tsSample.examples[k].y.m_labels = sample.examples[nTrainSampleNum+k].y.m_labels;
			}

			/* perfrom inner-cv to select good param. */
			if ( learn_parm.m_bInnerCV ) {
				struct_parm.C = inner_CV( &trSample, structmodel, 
					struct_parm, learn_parm, kernel_parm, alg_type, ofs);
			}

			/* Do the learning and return structmodel. */
			long runtime_start, runtime_end;
			runtime_start = get_runtime();

			bool bSucc = true;
			if(alg_type == 1) {
				if ( learn_parm.m_bQPSolver )
					svm_learn_struct(trSample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
				else 
					bSucc = svm_learn_struct_lp(trSample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, g_nStopStep, true);
			} else exit(1);
			runtime_end = get_runtime();


			/* Warning: The model contains references to the original data 'docs'.
			If you want to free the original data, and only keep the model, you 
			have to make a deep copy of 'model'. */
			if(struct_verbosity>=1) { printf("Writing learned model...");fflush(stdout);}
			write_struct_model(modelfile, &structmodel, &struct_parm);
			if(struct_verbosity>=1) {printf("done\n");fflush(stdout); }

			// do classification on the testing data
			arryAvgLoss[db][it] = evaluation(&tsSample, structmodel, ofs, false);
			ofs << "\tAvgLoss: " << arryAvgLoss[db][it] << "  [" << bSucc << "]" 
				<< "\tCPU Seconds: " << (runtime_end-runtime_start)/100.0 << endl;

			// free memory
			for ( int k=0; k<trSample.n; k++ ) {
				trSample.examples[k].x.SetHead(NULL);
				trSample.examples[k].y.m_labels = NULL;
			}
			for ( int k=0; k<tsSample.n; k++ ) {
				tsSample.examples[k].x.SetHead(NULL);
				tsSample.examples[k].y.m_labels = NULL;
			}
			free_struct_sample(trSample);
			free_struct_sample(tsSample);
			free_struct_model(structmodel);
		}

		free_struct_sample(sample);
		//free_struct_model(structmodel);
	}
	ofs << "\tAverage Loss over the " << vecFileName.size() << " files" << endl;
	vector<double> vecDBAvgLoss(vecRatio.size(), 0);
	for ( int i=0; i<vecRatio.size(); i++ ) {
		for ( int db=0; db<vecFileName.size(); db++ )
			vecDBAvgLoss[i] += arryAvgLoss[db][i] / vecFileName.size();
		ofs << "\t" << vecRatio[i] << "\t: " << vecDBAvgLoss[i] << endl;
	}
}

/* inner cross-validation to select good parameters. */
double inner_CV(SAMPLE *sample, STRUCTMODEL &structmodel,
			 STRUCT_LEARN_PARM &struct_parm,
			 LEARN_PARM &learn_parm, 
			 KERNEL_PARM &kernel_parm, 
			 int alg_type, ofstream &ofs)
{
	/* the candidate lambda. */
	vector<double> vecLambda;
	vecLambda.push_back(0.01);
	vecLambda.push_back(0.1);
	vecLambda.push_back(1);
	vecLambda.push_back(4);
	vecLambda.push_back(9);
	vecLambda.push_back(16);

	vector<double> arryAvgLoss( vecLambda.size(), 0 );
	int nUnitNum = sample->n / 5;
	for ( int cv=0; cv<5; cv++ ) /* 5-fold inner cv. */
	{
		// split the data into training & testing sets
		SAMPLE trSample, tsSample;
		trSample.n = nUnitNum;
		if ( cv == 4 ) trSample.n = sample->n - nUnitNum * 4;
		trSample.examples = (EXAMPLE*)malloc( sizeof(EXAMPLE) * trSample.n );
		tsSample.n = sample->n - trSample.n;
		tsSample.examples = (EXAMPLE*)malloc( sizeof(EXAMPLE) * tsSample.n );
		for ( int k=0; k<trSample.n; k++ ) {
			trSample.examples[k].x.SetHead( sample->examples[k+nUnitNum*cv].x.m_pHeadNode );
			trSample.examples[k].x.m_pEndNode = sample->examples[k+nUnitNum*cv].x.m_pEndNode;
			trSample.examples[k].y.m_nSize = sample->examples[k+nUnitNum*cv].y.m_nSize;
			trSample.examples[k].y.m_labels = sample->examples[k+nUnitNum*cv].y.m_labels;
		}
		int tsIx=0;
		for ( int k=0; k<sample->n; k++ ) {
			if ( k >= nUnitNum*cv && k < nUnitNum*cv+trSample.n ) continue;
			tsSample.examples[tsIx].x.SetHead( sample->examples[k].x.m_pHeadNode );
			tsSample.examples[tsIx].x.m_pEndNode = sample->examples[k].x.m_pEndNode;
			tsSample.examples[tsIx].y.m_nSize = sample->examples[k].y.m_nSize;
			tsSample.examples[tsIx ++].y.m_labels = sample->examples[k].y.m_labels;
		}

		for ( int k=0; k<vecLambda.size(); k++ ) {
			struct_parm.C = vecLambda[k];
			/* Do the learning and return structmodel. */
			bool bSucc = true;
			if(alg_type == 1) {
				if ( learn_parm.m_bQPSolver )
					svm_learn_struct(trSample, &struct_parm, &learn_parm, &kernel_parm, &structmodel);
				else 
					bSucc = svm_learn_struct_lp(trSample, &struct_parm, &learn_parm, &kernel_parm, &structmodel, g_nStopStep, true);
			} else exit(1);

			/* Warning: The model contains references to the original data 'docs'.
			If you want to free the original data, and only keep the model, you 
			have to make a deep copy of 'model'. */
			write_struct_model(modelfile, &structmodel, &struct_parm);

			arryAvgLoss[k] += evaluation(&tsSample, structmodel, ofs, false) / 5;

			// free memory
			for ( int k=0; k<trSample.n; k++ ) {
				trSample.examples[k].x.SetHead(NULL);
				trSample.examples[k].y.m_labels = NULL;
			}
			for ( int k=0; k<tsSample.n; k++ ) {
				tsSample.examples[k].x.SetHead(NULL);
				tsSample.examples[k].y.m_labels = NULL;
			}
			free_struct_sample(trSample);
			free_struct_sample(tsSample);
			free_struct_model(structmodel);
		}
	}

	double dBestLambda;
	double dMinErrRate = 1;
	for ( int k=0; k<vecLambda.size(); k++ ) {
		if ( arryAvgLoss[k] < dMinErrRate )
		{
			dMinErrRate = arryAvgLoss[k];
			dBestLambda = vecLambda[k];
		}
	}

	return dBestLambda;
}


/*---------------------------------------------------------------------------*/

void read_input_parameters(int argc,char *argv[],char *trainfile,
						   char *modelfile,
						   long *verbosity,long *struct_verbosity, 
						   STRUCT_LEARN_PARM *struct_parm,
						   LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
						   int *alg_type)
{
	long i;
	char type[100];

	/* set default */
	(*alg_type)=DEFAULT_ALG_TYPE;
	struct_parm->C=-0.01;
	struct_parm->slack_norm=1;
	struct_parm->epsilon=DEFAULT_EPS;
	struct_parm->custom_argc=0;
	struct_parm->loss_function=DEFAULT_LOSS_FCT;
	struct_parm->loss_type=DEFAULT_RESCALING;
	struct_parm->newconstretrain=100;
	struct_parm->ccache_size=5;

	strcpy (modelfile, "svm_struct_model");
	strcpy (learn_parm->predfile, "trans_predictions");
	strcpy (learn_parm->alphafile, "");
	(*verbosity)=0;/*verbosity for svm_light*/
	(*struct_verbosity)=1; /*verbosity for struct learning portion*/
	learn_parm->biased_hyperplane=1;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=10;
	learn_parm->svm_newvarsinqp=0;
	learn_parm->svm_iter_to_shrink=-9999;
	learn_parm->maxiter=100000;
	learn_parm->kernel_cache_size=40;
	learn_parm->svm_c=99999999;  /* overridden by struct_parm->C */
	learn_parm->eps=0.001;       /* overridden by struct_parm->epsilon */
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio=1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-10;  /* changed from 1e-15 */
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;
	learn_parm->m_bQPSolver=true; /* default to be QP solver*/
	kernel_parm->kernel_type=0;
	kernel_parm->poly_degree=3;
	kernel_parm->rbf_gamma=1.0;
	kernel_parm->coef_lin=1;
	kernel_parm->coef_const=1;
	strcpy(kernel_parm->custom,"empty");
	strcpy(type,"c");

	for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
		switch ((argv[i])[1]) 
		{ 
		case '?': print_help(); exit(0);
		case 'a': i++; strcpy(learn_parm->alphafile,argv[i]); break;
		case 'c': i++; struct_parm->C=atof(argv[i]); break;
		case 'p': i++; struct_parm->slack_norm=atol(argv[i]); break;
		case 'e': i++; struct_parm->epsilon=atof(argv[i]); break;
		case 'k': i++; struct_parm->newconstretrain=atol(argv[i]); break;
		case 'h': i++; learn_parm->svm_iter_to_shrink=atol(argv[i]); break;
		case '#': i++; learn_parm->maxiter=atol(argv[i]); break;
		case 'm': i++; learn_parm->kernel_cache_size=atol(argv[i]); break;
		case 'w': i++; (*alg_type)=atol(argv[i]); break;
		case 'o': i++; struct_parm->loss_type=atol(argv[i]); break;
		case 'n': i++; learn_parm->svm_newvarsinqp=atol(argv[i]); break;
		case 'q': i++; learn_parm->svm_maxqpsize=atol(argv[i]); break;
		case 'l': i++; struct_parm->loss_function=atol(argv[i]); break;
		case 'f': i++; struct_parm->ccache_size=atol(argv[i]); break;
		case 't': i++; kernel_parm->kernel_type=atol(argv[i]); break;
		case 'd': i++; kernel_parm->poly_degree=atol(argv[i]); break;
		case 'g': i++; kernel_parm->rbf_gamma=atof(argv[i]); break;
		case 's': i++; kernel_parm->coef_lin=atof(argv[i]); break;
		case 'r': i++; kernel_parm->coef_const=atof(argv[i]); break;
		case 'u': i++; strcpy(kernel_parm->custom,argv[i]); break;
		case '-': strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);i++; strcpy(struct_parm->custom_argv[struct_parm->custom_argc++],argv[i]);break; 
		case 'v': i++; (*struct_verbosity)=atol(argv[i]); break;
		case 'y': i++; (*verbosity)=atol(argv[i]); break;
		case 'b': i++; strcpy(learn_parm->addConfigFile,argv[i]); break;
		case 'x': learn_parm->m_bQPSolver=false; break;
		default: printf("\nUnrecognized option %s!\n\n",argv[i]);
			print_help();
			exit(0);
		}
	}
	if(i>=argc) {
		printf("\nNot enough input parameters!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	strcpy (trainfile, argv[i]);
	if((i+1)<argc) {
		strcpy (modelfile, argv[i+1]);
	}
	if(learn_parm->svm_iter_to_shrink == -9999) {
		learn_parm->svm_iter_to_shrink=100;
	}

	if((learn_parm->skip_final_opt_check) 
		&& (kernel_parm->kernel_type == LINEAR)) {
			printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
			learn_parm->skip_final_opt_check=0;
	}    
	if((learn_parm->skip_final_opt_check) 
		&& (learn_parm->remove_inconsistent)) {
			printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
			wait_any_key();
			print_help();
			exit(0);
	}    
	if((learn_parm->svm_maxqpsize<2)) {
		printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
		wait_any_key();
		print_help();
		exit(0);
	}
	if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
		printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
		printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->svm_iter_to_shrink<1) {
		printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
		wait_any_key();
		print_help();
		exit(0);
	}
	if(struct_parm->C<0) {
		printf("\nYou have to specify a value for the parameter '-c' (C>0)!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(((*alg_type) < 1) || ((*alg_type) > 4)) {
		printf("\nAlgorithm type must be either '1', '2', '3', or '4'!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->transduction_posratio>1) {
		printf("\nThe fraction of unlabeled examples to classify as positives must\n");
		printf("be less than 1.0 !!!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(learn_parm->svm_costratio<=0) {
		printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if(struct_parm->epsilon<=0) {
		printf("\nThe epsilon parameter must be greater than zero!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if((struct_parm->slack_norm<1) || (struct_parm->slack_norm>2)) {
		printf("\nThe norm of the slacks must be either 1 (L1-norm) or 2 (L2-norm)!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if((struct_parm->loss_type != SLACK_RESCALING) 
		&& (struct_parm->loss_type != MARGIN_RESCALING)) {
			printf("\nThe loss type must be either 1 (slack rescaling) or 2 (margin rescaling)!\n\n");
			wait_any_key();
			print_help();
			exit(0);
	}
	if(learn_parm->rho<0) {
		printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
		printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
		printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
		wait_any_key();
		print_help();
		exit(0);
	}
	if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
		printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
		printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
		printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
		wait_any_key();
		print_help();
		exit(0);
	}

	parse_struct_parameters(struct_parm);
}

void wait_any_key()
{
	printf("\n(more)\n");
	(void)getc(stdin);
}

void print_help()
{
	printf("\nSVM-struct learning module: %s, %s, %s\n",INST_NAME,INST_VERSION,INST_VERSION_DATE);
	printf("   includes SVM-struct %s for learning complex outputs, %s\n",STRUCT_VERSION,STRUCT_VERSION_DATE);
	printf("   includes SVM-light %s quadratic optimizer, %s\n",VERSION,VERSION_DATE);
	copyright_notice();
	printf("   usage: svm_struct_learn [options] example_file model_file\n\n");
	printf("Arguments:\n");
	printf("         example_file-> file with training data\n");
	printf("         model_file  -> file to store learned decision rule in\n");

	printf("General options:\n");
	printf("         -?          -> this help\n");
	printf("         -v [0..3]   -> verbosity level (default 1)\n");
	printf("         -y [0..3]   -> verbosity level for svm_light (default 0)\n");
	printf("Learning options:\n");
	printf("         -c float    -> C: trade-off between training error\n");
	printf("                        and margin (default 0.01)\n");
	printf("         -p [1,2]    -> L-norm to use for slack variables. Use 1 for L1-norm,\n");
	printf("                        use 2 for squared slacks. (default 1)\n");
	printf("         -o [1,2]    -> Rescaling method to use for loss.\n");
	printf("                        1: slack rescaling\n");
	printf("                        2: margin rescaling\n");
	printf("                        (default %d)\n",DEFAULT_RESCALING);
	printf("         -l [0..]    -> Loss function to use.\n");
	printf("                        0: zero/one loss\n");
	printf("                        (default %d)\n",DEFAULT_LOSS_FCT);
	printf("Kernel options:\n");
	printf("         -t int      -> type of kernel function:\n");
	printf("                        0: linear (default)\n");
	printf("                        1: polynomial (s a*b+c)^d\n");
	printf("                        2: radial basis function exp(-gamma ||a-b||^2)\n");
	printf("                        3: sigmoid tanh(s a*b + c)\n");
	printf("                        4: user defined kernel from kernel.h\n");
	printf("         -d int      -> parameter d in polynomial kernel\n");
	printf("         -g float    -> parameter gamma in rbf kernel\n");
	printf("         -s float    -> parameter s in sigmoid/poly kernel\n");
	printf("         -r float    -> parameter c in sigmoid/poly kernel\n");
	printf("         -u string   -> parameter of user defined kernel\n");
	printf("Optimization options (see [2][3]):\n");
	printf("         -w [1,2,3,4]-> choice of structural learning algorithm (default %d):\n",(int)DEFAULT_ALG_TYPE);
	printf("                        1: algorithm described in [2]\n");
	printf("                        2: joint constraint algorithm (primal) [to be published]\n");
	printf("                        3: joint constraint algorithm (dual) [to be published]\n");
	printf("                        4: joint constraint algorithm (dual) with constr. cache\n");
	printf("         -q [2..]    -> maximum size of QP-subproblems (default 10)\n");
	printf("         -n [2..q]   -> number of new variables entering the working set\n");
	printf("                        in each iteration (default n = q). Set n<q to prevent\n");
	printf("                        zig-zagging.\n");
	printf("         -m [5..]    -> size of cache for kernel evaluations in MB (default 40)\n");
	printf("                        (used only for -w 1 with kernels)\n");
	printf("         -f [5..]    -> number of constraints to cache for each example\n");
	printf("                        (default 5) (used with -w 4)\n");
	printf("         -e float    -> eps: Allow that error for termination criterion\n");
	printf("                        (default %f)\n",DEFAULT_EPS);
	printf("         -h [5..]    -> number of iterations a variable needs to be\n"); 
	printf("                        optimal before considered for shrinking (default 100)\n");
	printf("         -k [1..]    -> number of new constraints to accumulate before\n"); 
	printf("                        recomputing the QP solution (default 100) (-w 1 only)\n");
	printf("         -# int      -> terminate QP subproblem optimization, if no progress\n");
	printf("                        after this number of iterations. (default 100000)\n");
	printf("Output options:\n");
	printf("         -a string   -> write all alphas to this file after learning\n");
	printf("                        (in the same order as in the training set)\n");
	printf("Structure learning options:\n");
	print_struct_help();
	wait_any_key();

	printf("\nMore details in:\n");
	printf("[1] T. Joachims, Learning to Align Sequences: A Maximum Margin Aproach.\n");
	printf("    Technical Report, September, 2003.\n");
	printf("[2] I. Tsochantaridis, T. Joachims, T. Hofmann, and Y. Altun, Large Margin\n");
	printf("    Methods for Structured and Interdependent Output Variables, Journal\n");
	printf("    of Machine Learning Research (JMLR), Vol. 6(Sep):1453-1484, 2005.\n");
	printf("[3] T. Joachims, Making Large-Scale SVM Learning Practical. Advances in\n");
	printf("    Kernel Methods - Support Vector Learning, B. Schölkopf and C. Burges and\n");
	printf("    A. Smola (ed.), MIT Press, 1999.\n");
	printf("[4] T. Joachims, Learning to Classify Text Using Support Vector\n");
	printf("    Machines: Methods, Theory, and Algorithms. Dissertation, Kluwer,\n");
	printf("    2002.\n\n");
}



