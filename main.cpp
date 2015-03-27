#include "qclcontext.h"
#include "qclprogram.h"
#include "qclkernel.h"
#include <ctime>
#include "iostream"
#include <math.h>
using namespace std;

#define GPU 0
#define CPU 1
#define STRASSEN 2

const int leafsize = 512;
int mode = CPU;
QCLVector<int> inbuffer_A;
QCLVector<int> inbuffer_B;
QCLVector<int> outbuffer;

QCLKernel kernel;
QCLContext context;
QCLProgram program;

void ikjalgorithmCPU(vector< vector<int> > A,vector< vector<int> > B,vector< vector<int> > &C, int n);

void strassen(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, unsigned int tam);
int nextPuissance2(int n);
void strassenR(vector< vector<int> > &A,
               vector< vector<int> > &B,
               vector< vector<int> > &C,
               int tam);
void sum(vector< vector<int> > &A,
         vector< vector<int> > &B,
         vector< vector<int> > &C, int tam);
void subtract(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, int tam);
void printMatrix(vector< vector<int> > matrix);


vector<int> aPlat(vector< vector<int> > A){
    vector<int> result;
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[i].size(); ++j) {
            result.push_back(A[i][j]);
        }
    }
    return result;
}



void ikjalgorithm(vector< vector<int> > A,
                  vector< vector<int> > B,
                  vector< vector<int> > &C, int n) {

    vector<int> A_plane = aPlat(A);
    vector<int> B_plane = aPlat(B);
    vector<int> outdata(A_plane.size());

    inbuffer_A=context.createVector<int>(n*n,QCLMemoryObject::ReadOnly);
    inbuffer_B=context.createVector<int>(n*n,QCLMemoryObject::ReadOnly);
    outbuffer=context.createVector<int>(n*n,QCLMemoryObject::WriteOnly);
    program=context.buildProgramFromSourceFile("multiplication.cl");
    kernel=program.createKernel("multiplication");
    kernel.setGlobalWorkSize(n,n);
    kernel.setArg(0,outbuffer);
    kernel.setArg(1,inbuffer_A);
    kernel.setArg(2,inbuffer_B);
    kernel.setArg(3,n);

    inbuffer_A.write(&A_plane[0],A_plane.size());
    inbuffer_B.write(&B_plane[0],B_plane.size());
    kernel.run();
    outbuffer.read(&outdata[0],A_plane.size());

    for (int i = 0,pas = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j,pas++) {
            C[i][j] += outdata[pas];
        }
    }
}

void ikjalgorithmCPU(vector< vector<int> > A,
                     vector< vector<int> > B,
                     vector< vector<int> > &C, int n) {


    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void strassenR(vector< vector<int> > &A,
               vector< vector<int> > &B,
               vector< vector<int> > &C, int tam) {
    if (tam <= leafsize) {
        ikjalgorithm(A, B, C, tam);
        return;
    }
    // other cases are treated here:
    else {
        int newTam = tam/2;
        vector<int> inner (newTam,0);
        vector< vector<int> >
                a11(newTam,inner), a12(newTam,inner), a21(newTam,inner), a22(newTam,inner),
                b11(newTam,inner), b12(newTam,inner), b21(newTam,inner), b22(newTam,inner),
                c11(newTam,inner), c12(newTam,inner), c21(newTam,inner), c22(newTam,inner),
                p1(newTam,inner), p2(newTam,inner), p3(newTam,inner), p4(newTam,inner),
                p5(newTam,inner), p6(newTam,inner), p7(newTam,inner),
                aResult(newTam,inner), bResult(newTam,inner);
        int i, j;
        //dividing the matrices in 4 sub-matrices:
        for (i = 0; i < newTam; i++) {
            for (j = 0; j < newTam; j++) {
                a11[i][j] = A[i][j];
                a12[i][j] = A[i][j + newTam];
                a21[i][j] = A[i + newTam][j];
                a22[i][j] = A[i + newTam][j + newTam];
                b11[i][j] = B[i][j];
                b12[i][j] = B[i][j + newTam];
                b21[i][j] = B[i + newTam][j];
                b22[i][j] = B[i + newTam][j + newTam];
            }
        }

        // Calculating p1 to p7:
        sum(a11, a22, aResult, newTam); // a11 + a22
        sum(b11, b22, bResult, newTam); // b11 + b22
        strassenR(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)
        sum(a21, a22, aResult, newTam); // a21 + a22
        strassenR(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)
        subtract(b12, b22, bResult, newTam); // b12 - b22
        strassenR(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)
        subtract(b21, b11, bResult, newTam); // b21 - b11
        strassenR(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)
        sum(a11, a12, aResult, newTam); // a11 + a12
        strassenR(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)
        subtract(a21, a11, aResult, newTam); // a21 - a11
        sum(b11, b12, bResult, newTam); // b11 + b12
        strassenR(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)
        subtract(a12, a22, aResult, newTam); // a12 - a22
        sum(b21, b22, bResult, newTam); // b21 + b22
        strassenR(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)
        // calculating c21, c21, c11 e c22:
        sum(p3, p5, c12, newTam); // c12 = p3 + p5
        sum(p2, p4, c21, newTam); // c21 = p2 + p4
        sum(p1, p4, aResult, newTam); // p1 + p4
        sum(aResult, p7, bResult, newTam); // p1 + p4 + p7
        subtract(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7
        sum(p1, p3, aResult, newTam); // p1 + p3
        sum(aResult, p6, bResult, newTam); // p1 + p3 + p6
        subtract(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6
        // Grouping the results obtained in a single matrix:
        for (i = 0; i < newTam ; i++) {
            for (j = 0 ; j < newTam ; j++) {
                C[i][j] = c11[i][j];
                C[i][j + newTam] = c12[i][j];
                C[i + newTam][j] = c21[i][j];
                C[i + newTam][j + newTam] = c22[i][j];
            }
        }
    }
}


int nextPuissance2(int n) {
    return pow(2, int(ceil(log2(n))));
}

void sum(vector< vector<int> > &A,
         vector< vector<int> > &B,
         vector< vector<int> > &C, int tam) {
    int i, j;
    for (i = 0; i < tam; i++) {
        for (j = 0; j < tam; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}


void subtract(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, int tam) {
    int i, j;
    for (i = 0; i < tam; i++) {
        for (j = 0; j < tam; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}


void printMatrix(vector< vector<int> > matrix) {
    for (int i=0; i < matrix.size(); i++) {
        for (int j=0; j < matrix[i].size(); j++) {
            if (j != 0) {
                cout << " ";
            }
            cout << matrix[i][j];
        }
        cout << endl;
    }
}

void print(int* matrix, int N) {
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            if (j != 0) {
                cout << " ";
            }
            cout << matrix[i*N+j];
        }
        cout << endl;
    }
}

void strassenRCPU(vector< vector<int> > &A,
               vector< vector<int> > &B,
               vector< vector<int> > &C, int tam) {
    if (tam <= leafsize) {
        ikjalgorithmCPU(A, B, C, tam);
        return;
    }
    // other cases are treated here:
    else {
        int newTam = tam/2;
        vector<int> inner (newTam,0);
        vector< vector<int> >
                a11(newTam,inner), a12(newTam,inner), a21(newTam,inner), a22(newTam,inner),
                b11(newTam,inner), b12(newTam,inner), b21(newTam,inner), b22(newTam,inner),
                c11(newTam,inner), c12(newTam,inner), c21(newTam,inner), c22(newTam,inner),
                p1(newTam,inner), p2(newTam,inner), p3(newTam,inner), p4(newTam,inner),
                p5(newTam,inner), p6(newTam,inner), p7(newTam,inner),
                aResult(newTam,inner), bResult(newTam,inner);
        int i, j;
        //dividing the matrices in 4 sub-matrices:
        for (i = 0; i < newTam; i++) {
            for (j = 0; j < newTam; j++) {
                a11[i][j] = A[i][j];
                a12[i][j] = A[i][j + newTam];
                a21[i][j] = A[i + newTam][j];
                a22[i][j] = A[i + newTam][j + newTam];
                b11[i][j] = B[i][j];
                b12[i][j] = B[i][j + newTam];
                b21[i][j] = B[i + newTam][j];
                b22[i][j] = B[i + newTam][j + newTam];
            }
        }

        // Calculating p1 to p7:
        sum(a11, a22, aResult, newTam); // a11 + a22
        sum(b11, b22, bResult, newTam); // b11 + b22
        strassenRCPU(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)
        sum(a21, a22, aResult, newTam); // a21 + a22
        strassenRCPU(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)
        subtract(b12, b22, bResult, newTam); // b12 - b22
        strassenRCPU(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)
        subtract(b21, b11, bResult, newTam); // b21 - b11
        strassenRCPU(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)
        sum(a11, a12, aResult, newTam); // a11 + a12
        strassenRCPU(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)
        subtract(a21, a11, aResult, newTam); // a21 - a11
        sum(b11, b12, bResult, newTam); // b11 + b12
        strassenRCPU(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)
        subtract(a12, a22, aResult, newTam); // a12 - a22
        sum(b21, b22, bResult, newTam); // b21 + b22
        strassenRCPU(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)
        // calculating c21, c21, c11 e c22:
        sum(p3, p5, c12, newTam); // c12 = p3 + p5
        sum(p2, p4, c21, newTam); // c21 = p2 + p4
        sum(p1, p4, aResult, newTam); // p1 + p4
        sum(aResult, p7, bResult, newTam); // p1 + p4 + p7
        subtract(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7
        sum(p1, p3, aResult, newTam); // p1 + p3
        sum(aResult, p6, bResult, newTam); // p1 + p3 + p6
        subtract(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6
        // Grouping the results obtained in a single matrix:
        for (i = 0; i < newTam ; i++) {
            for (j = 0 ; j < newTam ; j++) {
                C[i][j] = c11[i][j];
                C[i][j + newTam] = c12[i][j];
                C[i + newTam][j] = c21[i][j];
                C[i + newTam][j + newTam] = c22[i][j];
            }
        }
    }
}


void strassen(vector< vector<int> > &A,
              vector< vector<int> > &B,
              vector< vector<int> > &C, unsigned int n) {
    //unsigned int n = tam;
    unsigned int m = nextPuissance2(n);
    vector<int> inner(m,0);
    vector< vector<int> > APrep(m, inner), BPrep(m, inner), CPrep(m, inner);
    for(unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            APrep[i][j] = A[i][j];
            BPrep[i][j] = B[i][j];
        }
    }

   if(mode==STRASSEN){
       strassenRCPU(APrep,BPrep,CPrep, m);
   }else{
    strassenR(APrep, BPrep, CPrep, m);
   }
    for(unsigned int i=0; i<n; i++) {
        for (unsigned int j=0; j<n; j++) {
            C[i][j] = CPrep[i][j];
        }
    }
}












int main(int argc, char *argv[])
{    
    //dimension des matrices
    int N=10;
    if(argv[2] != NULL)
        N=(atoi(argv[2])>0)?atoi(argv[2]):10;
    if(argv[1] != NULL){
        mode = ( strcmp(argv[1],"-cpu") == 0)?CPU:mode;
        mode = ( strcmp(argv[1],"-gpu") == 0)?GPU:mode;
        mode = ( strcmp(argv[1],"-strassen") == 0)?STRASSEN:mode;
    }


    int *indata1=new int[N*N];
    int *indata2=new int[N*N];
    int *outdata=new int[N*N];

    //initialisation des matrices
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            indata1[(i*N)+j]=1;
            indata2[(i*N)+j]=1;
        }
    }

    if (mode == CPU) {
        cout<<"Multiplication en mode CPU avec matrice en 1D"<<endl;

        clock_t    timer;
        timer = clock();

        //multiplication de AxB
        for (int i = 0; i< N; ++i) {
            for(int j=0;j<N; j++){
                outdata[(i*N)+j] = 0;
                for(int k=0; k<N; k++){
                    outdata[(i*N)+j] += indata1[(i*N)+k]*indata2[(k*N)+j];
                }
            }
        }
        cout << "Timer CPU: " << (clock() - timer) / (double)(CLOCKS_PER_SEC)<< " s" <<endl;

        if(N<=5){
            cout<<"Matrice A"<<endl;
            print(indata1,N);
            cout<<"Matrice B"<<endl;
            print(indata2,N);
            cout<<"Matrice C"<<endl;
            print(outdata,N);
        }

    }else if(mode== GPU){
        if(!context.create()){
            qFatal("Could not create OpenCL context for the GPU\n");
            exit(0);
        }

        int n = N;
        vector<int> inner (n,0);
        vector< vector<int> > A(n, inner), B(n, inner), C(n, inner);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = B[i][j] = 1;
            }
        }
        if(N != nextPuissance2(N)){
            N = nextPuissance2(N);
        }

        cout<<"Multiplication en mode GPU"<<endl;

        clock_t    timer;
        timer = clock();
        strassen(A,B,C,n);
        cout<<"C [0] "<<C[0][0]<<endl;
        cout << "Timer CPU: " << (clock() - timer) / (double)(CLOCKS_PER_SEC) << " s" <<endl;


        if(N<=5){
            cout<<"Matrice A"<<endl;
            printMatrix(A);
            cout<<"Matrice B"<<endl;
            printMatrix(B);
            cout<<"Matrice C"<<endl;
            printMatrix(C);
        }

    }else if(mode==STRASSEN){
        cout<<"Multiplication en mode CPU avec algorithme STRASSEN"<<endl;

        int n = N;
        vector<int> inner (n,0);
        vector< vector<int> > A(n, inner), B(n, inner), C(n, inner);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i][j] = B[i][j] = 1;
            }
        }
        if(N != nextPuissance2(N)){
            N = nextPuissance2(N);
        }

        clock_t    timer;
        timer = clock();
        strassen(A,B,C,n);
        cout<<"C [0] "<<C[0][0]<<endl;
        cout << "Timer CPU avec strassen: " << (clock() - timer) / (double)(CLOCKS_PER_SEC) << " s" <<endl;

        if(N<=5){
            cout<<"Matrice A"<<endl;
            printMatrix(A);
            cout<<"Matrice B"<<endl;
            printMatrix(B);
            cout<<"Matrice C"<<endl;
            printMatrix(C);
        }
    }



}
