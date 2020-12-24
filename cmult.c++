#include<iostream>

using namespace std;
float cmult(int param1,float param2){
    float return_param=param1*param2;
    printf("Executing cmult: param1: %d, param2: %.1f ; Result: %.1f\n",param1,param2,return_param);
    return return_param;
    
}

int main(){
    int a;
    float b;
    cin>>a;
    cin>>b;
    cmult(a,b);
    return 1;
}

