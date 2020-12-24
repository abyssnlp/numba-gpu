#include<stdio.h>
#include"cmult.h"

float cmult(int param1, float param2){
    float return_param=param1*param2;
    printf("Executing cmult: %d with %.1f returns %.1f\n",param1,param2,return_param);
    return return_param;
}