#include "qclcontext.h"
#include "qclprogram.h"
#include "qclkernel.h"
#include <ctime>
#include "iostream"
using namespace std;

#define GPU 0
#define CPU 1

void affichMatrice(){

}

int main(int argc, char *argv[])
{    


    // Declarations
    QCLContext context;
    QCLProgram program;
    QCLKernel kernel;

    //dimension des matrices
    int N=10;
    int mode = CPU;
    if(argv[2] != NULL)
        N=(atoi(argv[2])>0)?atoi(argv[2]):10;
    if(argv[1] != NULL){
        mode = ( strcmp(argv[1],"-cpu") == 0)?CPU:mode;
        mode = ( strcmp(argv[1],"-gpu") == 0)?GPU:mode;
    }

    //matrice donnée
    int **A = new int*[N];
    //matrice donnée
    int **B = new int*[N];
    //matrice résultat
    int **C = new int*[N];

    //création des matrices
    for (int i = 0; i < N; ++i) {
        A[i] = new int[N];
        B[i] = new int[N];
        C[i] = new int[N];
        for (int j = 0; j < N; ++j) {
            A[i][j] = 1;
            B[i][j] = 1;
        }
    }

    if (mode == CPU) {
        cout<<"Multiplication en mode CPU"<<endl;
        clock_t    timer;
        timer = clock();
        //multiplication de AxB
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i][j]=0;
                for (int k = 0; k < N; ++k) {
                    C[i][j] += A[i][k]*B[k][j];
                }
            }
        }
        cout << "Timer CPU: " << (clock() - timer) / (double)(CLOCKS_PER_SEC / 1000) << " ms" <<endl;


    }else{
        if(!context.create()){
            qFatal("Could not create OpenCL context for the GPU\n");
            exit(0);
        }

        QCLVector<int> inbufferA=context.createVector<int>(N*N,QCLMemoryObject::ReadOnly);
        QCLVector<int> inbufferB=context.createVector<int>(N*N,QCLMemoryObject::ReadOnly);
        QCLVector<int> outbuffer=context.createVector<int>(N*N,QCLMemoryObject::WriteOnly);
        program=context.buildProgramFromSourceFile("multiplication.cl");
        kernel=program.createKernel("multiplication");
        kernel.setGlobalWorkSize(N,N);
        kernel.setArg(0,outbuffer);
        kernel.setArg(1,inbufferA);
        kernel.setArg(2,inbufferB);
        kernel.setArg(3,N);

        cout<<"Multiplication en mode GPU"<<endl;
        int *indataA = new int[N*N];
        int *indataB = new int[N*N];
        int *outdata = new int[N*N];

        //Mise a plats des matrices
        int pas =0;
        for (int i = 0; i <N; i++) {
            for (int j = 0; j < N; j++) {
                indataA[pas] = A[i][j];
                indataB[pas] = B[j][i];
                pas++;
            }
        }

        inbufferA.write(indataA,N*N);
        inbufferB.write(indataB,N*N);
        clock_t    timer;
        timer = clock();
        kernel.run();
        cout << "Timer GPU: " << (clock() - timer) / (double)(CLOCKS_PER_SEC / 1000) << " ms" <<endl;

        outbuffer.read(outdata,N*N);

        for (int i = 0, pas=0; i < N; ++i) {
            for(int j=0; j < N; j++,pas++){
                C[i][j] = outdata[pas];
            }
        }

        delete[] indataA;
        delete[] indataB;
        delete[] outdata;
    }

    if(N<=5){
        cout<<"Matrice A"<<endl;
        for (int i=0; i<N; i++){
            for (int j = 0; j < N; j++){
                cout<<A[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<"Matrice B"<<endl;
        for (int i=0; i<N; i++){
            for (int j = 0; j < N; j++){
                cout<<B[i][j]<<" ";
            }
            cout<<endl;
        }
        cout<<"Matrice C"<<endl;
        for (int i=0; i<N; i++){
            for (int j = 0; j < N; j++){
                cout<<C[i][j]<<" ";
            }
            cout<<endl;
        }
    }

    for (int i = 0; i < N; i++){
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] B;
    delete[] C;
}
