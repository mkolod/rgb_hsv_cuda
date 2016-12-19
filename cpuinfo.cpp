#include <iostream>
#include <string>

using std::cout;

enum cpuid_requests {
  CPUID_GETVENDORSTRING,
  CPUID_GETFEATURES,
  CPUID_GETTLB,
  CPUID_GETSERIAL,

  CPUID_INTELEXTENDED=0x80000000,
  CPUID_INTELFEATURES,
  CPUID_INTELBRANDSTRING,
  CPUID_INTELBRANDSTRINGMORE,
  CPUID_INTELBRANDSTRINGEND,
};

static inline void cpuid(int code, int *a, int *b, int *c, int *d) {
  __asm__ __volatile__("cpuid":"=a"(*a),"=b"(*b),
                        "=c"(*c),"=d"(*d):"a"(code));
}

void print_cpu_id() {

	union{
     struct reg{
         int eax;
         int ebx;
         int ecx;
         int edx;
     }cpu;
   char string[16];
  }info;

  cout << "Processor Brand:  ";

  cpuid(CPUID_INTELBRANDSTRING, &info.cpu.eax, &info.cpu.ebx, &info.cpu.ecx, &info.cpu.edx);
  cout << std::string(info.string, 16);

  cpuid(CPUID_INTELBRANDSTRINGMORE, &info.cpu.eax, &info.cpu.ebx, &info.cpu.ecx, &info.cpu.edx);
  cout << std::string(info.string, 16);

  cpuid(CPUID_INTELBRANDSTRINGEND, &info.cpu.eax, &info.cpu.ebx, &info.cpu.ecx, &info.cpu.edx);
  cout << std::string(info.string, 16)  << "\n";
}

int main(int argc, char **argv) {

  print_cpu_id();
  return 0;
}
