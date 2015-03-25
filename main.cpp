#include "qclcontext.h"
#include "qclprogram.h"
#include "qclkernel.h"
#include <ctime>
#include "iostream"
#include <math.h>
using namespace std;

#define GPU 0
#define CPU 1


int main(int argc, char *argv[])
{    

    // Declarations+-
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
        cout<<"Multiplication en mode CPU"<<endl;

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

        inbufferA.write(indata1,N*N);
        inbufferB.write(indata2,N*N);
        clock_t    timer;
        timer = clock();
        kernel.run();
        outbuffer.read(outdata,N*N);
        cout << "Timer GPU: " << (clock() - timer) / (double)(CLOCKS_PER_SEC / 1000) << " ms" <<endl;


    }
    if(N<11){
            cout<<endl<<"Matrice A"<<endl;
            for(int i=0; i<N; i++){
                for(int j=0; j<N; j++)
                    cout<<indata1[(i*N)+j]<<" ";
                cout<<endl;
            }
            cout<<endl<<"Matrice B"<<endl;
            for(int i=0; i<N; i++){
                for(int j=0; j<N; j++)
                    cout<<indata2[(i*N)+j]<<" ";
                cout<<endl;
            }
            cout<<endl<<"Matrice C"<<endl;
            for(int i=0; i<N; i++){
                for(int j=0; j<N; j++)
                    cout<<outdata[(i*N)+j]<<" ";
                cout<<endl;
            }
        }
}
