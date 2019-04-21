#ifndef _REPEATADD_DETECTION_
#define _REPEATADD_DETECTION_
#include <vector>
#include <fstream>
#include <assert.h>
#include "instr_info.h"
#include "omp.h"
//#include "addr2regs.h"
#include <map>
using namespace std;
//#define BATCH_SIZE 5000

class repeatadd_detect
{
public:
  //int num;
  //vector<addr2regs> addr_regs;
  //map<string, vector<dynid_mem>> addr2regs; 
  map<string, long long> addr2regs; 
  vector<instr_info> add_list;
  vector<instr_info> sub_list;
  
  repeatadd_detect();
  void detector(vector<instr_info> &instr_info_set, ofstream &ou);
  virtual ~repeatadd_detect(void);

};

#endif 
