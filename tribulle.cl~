__kernel void tribulle(__global __write_only int* output,__global __read_only int* input,const unsigned int size)
{
	unsigned int index=get_global_id(0);

	for(int i=0; i<size; i++){
	int pair = i%2;
		if(index%2 != pair){
			if(index+1==size)
			{
				output[index]=input[index];
			}else{
				output[index]=min(input[index],input[index+1]);
			} 	
		}else{
			if(index>0)
			{
				output[index]=max(input[index],input[index-1]);
			}else{
				output[index]=input[index];
			}
		}
	input[index] = output[index];
	}
}
