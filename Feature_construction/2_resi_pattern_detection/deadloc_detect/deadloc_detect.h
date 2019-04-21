#ifndef _CONDITION_DETECTION_
#define _CONDITION_DETECTION_
#include <vector>
#include <fstream>
#include "instr_info.h"
#include "loc_arr.h"
#include "omp.h"

#define BATCH_SIZE 5000

class res_batch
{
public:
  int dead_reg;
  int all_reg;
  int dead_addr;
  int all_addr;
  double ratio;
  res_batch(int a, int b, int c, int d, double e){
    dead_reg = a;
    all_reg = b;
    dead_addr = c;
    all_addr = d;
    ratio = e;
  }
  ~res_batch();
};

class deadloc_detect
{
public:
  int num;
  vector<res_batch> dead_loc_ratio; // the dead loc ratio for each batch  
  deadloc_detect();
  void detector(vector<instr_info> &instr_info_set, ofstream &ou, string str1, string str2, bool ifend);
  virtual ~deadloc_detect(void);

};

#endif 
