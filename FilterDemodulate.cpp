#include "stdafx.h"
#include <windows.h>
#include <iostream>
#include <fstream>
#include "FilterDemodulate.h"
#include <cuda_runtime.h>
#include <time.h>
#include "gpu_info.h"

#include "math.h"
#include "chebFilter.h"
#include "chebFilter_emxAPI.h"
#include "chebFilter_emxutil.h"
#include  "omp.h"
#include "FilterDemodulate_cu.h"
//#include "mkl_vsl_functions.h"

#define PI 3.1415926

using namespace std;

//int i1 = 0; 
//int i2 = 0;
//int i3 = 0;
void IQ(double *input, int inputLength, double *FIRfilter, int filterLength, double *Fc, double *Fs)
{
	int n = inputLength;
	int m = filterLength;
	int samplelen = 4;
	double *Itin = new double[n];
	double *Qtin = new double[n];
	double *Itout = new double[n];
	double *Qtout = new double[n];

	memset(Itout, 0, sizeof(double)*n);
	memset(Qtout, 0, sizeof(double)*n);
	
	for (int i = 0; i <= (n - m); i++)
	{
		double sumI = 0;
		double sumQ = 0;
		for (int j = 0; j < m; j++)
		{
			sumI += input[i + j] * Fc[i + j] * FIRfilter[j];
			sumQ += input[i + j] * Fs[i + j] * FIRfilter[j];
		}
			
		Itout[m / 2 + i - 1] = sumI;
		Qtout[m / 2 + i - 1] = sumQ;
	}
	
	
	int SampleManagerDataLen = n / samplelen + (n % samplelen == 0 ? 0 : 1);
	double *FilterimageDatachangeI = new double[SampleManagerDataLen];
	memset(FilterimageDatachangeI, 0.0, SampleManagerDataLen * sizeof(double));
	double *FilterimageDatachangeQ = new double[SampleManagerDataLen];
	memset(FilterimageDatachangeQ, 0.0, SampleManagerDataLen * sizeof(double));

	for (int j = 0; j < n;)
	{
		FilterimageDatachangeI[j / samplelen] = Itout[j];
		FilterimageDatachangeQ[j / samplelen] = Qtout[j];
		j += samplelen;
	}
	
	for (int i = 0; i <SampleManagerDataLen; i++){
		input[i] = sqrt(FilterimageDatachangeI[i] * FilterimageDatachangeI[i] + FilterimageDatachangeQ[i] * FilterimageDatachangeQ[i])*2;//0.005
		//printf("-------------------------------------input   %d is %lu\n", i,input[i]);
	}		
	//char* outputFilename11 = "F:\\包络后.txt";
	//ofstream fout11(outputFilename11);
	////fout.open(outputFilename + _itoa(i1 + 1, FileNum, 10) + ".txt");
	//if (fout11){
	//	for (int j = 0; j < SampleManagerDataLen; j++)
	//		fout11 << input[j] << endl;
	//	fout11.close();
	//}

	delete[]Itin;
	Itin = NULL;
	delete[]Qtin;
	Qtin = NULL;
	delete[]Itout;
	Itout = NULL;
	delete[]Qtout;
	Qtout = NULL;
	//delete[]midData;
	delete[]FilterimageDatachangeI;
	FilterimageDatachangeI = NULL;
	delete[]FilterimageDatachangeQ;
	FilterimageDatachangeQ = NULL;
}


void FilterDemodulate(/*UINT32*/double *inputData, int inputLength, /*UINT32*/double *outputData,
	double *HighFIRfilter, int HighfilterLength, double *IQFIRfilter, int IQfilterLength, double *ParaCos, double *ParaSin, double *Itout, double *Qtout, double *FilterimageDatachangeI, double *FilterimageDatachangeQ, int samplelen, int SampleManagerDataLen)
{

	double sumData = 0;
	double sumQ = 0;
	double *Itin = new double[inputLength];
	double *Qtin = new double[inputLength];

	int length = inputLength - IQfilterLength; //IQfilterLength:51    HighfilterLength:51
	int index = IQfilterLength / 2 - 1;

	for (int i = 0; i < inputLength; i++)//////////////////IQ解调 inputLength:2800
	{
		Itin[i] = inputData[i] * ParaCos[i];
		Qtin[i] = inputData[i] * ParaSin[i];

	}

	int temp = IQfilterLength / 2;	

	for (int j = IQfilterLength; j <= inputLength; j++)
	{
		sumData = 0;
		sumQ = 0;
		if (j > IQfilterLength - 1) {
			for (int k = 0; k < IQfilterLength; k++) {
				sumData += Itin[j - k] * IQFIRfilter[k];
				sumQ += Qtin[j - k] * IQFIRfilter[k];
			}
		}
		else {
			for (int k = 0; k < j + 1; k++) {
				sumData += Itin[j - k] * IQFIRfilter[k];
				sumQ += Qtin[j - k] * IQFIRfilter[k];
			}
		}
		Itout[j - IQfilterLength] = sumData;
		Qtout[j - IQfilterLength] = sumQ;
	}


	for (int i = 0; i <SampleManagerDataLen; ++i){////////////////////////////////包络检测   SampleManagerDataLen:560
		int j = i * samplelen;
		if (j < inputLength)
			outputData[i] = sqrtf(Itout[j] * Itout[j] + Qtout[j] * Qtout[j]) * 2;//0.005
		else
			outputData[i] = 0;//0.005
	}

	delete[]Itin;
	Itin = NULL;
	delete[]Qtin;
	Qtin = NULL;
}


void filter(double *input, double* output, int inputLength, double *FIRfilter, int filterLength)
{
	int n = inputLength;
	int m = filterLength;
	//_LARGE_INTEGER k;
	//_LARGE_INTEGER k1;
	//double dqFreq;                /*计时器频率*/
	//LARGE_INTEGER f;            /*计时器频率*/
	//QueryPerformanceFrequency(&f);
	//dqFreq = (double)f.QuadPart;
	//omp_set_num_threads(2);
	////QueryPerformanceCounter(&k);

 //   #pragma omp parallel for
	//{   
//#pragma omp for
	for (int i = 0; i <= (n - m); i++)
	{
		double sum = 0;
		
		for (int j = 0; j < m; j++)
			sum += input[i + j] * FIRfilter[j];
			
		output[m / 2 + i - 1] = sum;
	}
	//}
	//QueryPerformanceCounter(&k1);
	//double aa3 = (k1.QuadPart - k.QuadPart) / dqFreq;
	//	//if ((aa3 * 1000) >1)
	//	{
	//	printf("-------------------IQ  time  is %lf\n", aa3 * 1000);
	//	}
}

