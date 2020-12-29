#include<iostream>
#include "cppmult.h"

using namespace std;

float cppmult(int param1,float param2){
    float return_param=param1*param2;
    cout<<"Executing in cppmult with param1: "<<param1<<"param2: "<<param2<<"returns "<<return_param;
    return return_param;
}