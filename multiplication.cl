__kernel void multiplication(__global __write_only int* C,__global __read_only int* A,__global __read_only int* B,int size)
{
   int rang = get_global_id(0); 
   int rang1 = get_global_id(1);

   int value = 0;
   for (int k = 0; k < size; ++k)
   {
      int elementA = A[rang1 * size + k];
      int elementB = B[k * size + rang];
      value += elementA * elementB;
   }

   C[rang1 * size + rang] = value;

}
