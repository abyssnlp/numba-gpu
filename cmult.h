#ifdef _MSC_VER
    #define EXPORT_SYMBOL _declspec(dllexport)
#else
    #define EXPORT_SYMBOL
#endif

EXPORT_SYMBOL float cmult(int param1,float param2);


// gcc -c -DBUILD_DLL dll.c
// gcc -shared -o mydll.dll dll.o -Wl,--add-stdcall-alias