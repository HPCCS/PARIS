#ifndef _4INSTYPE_DETECTION_
#define _4INSTYPE_DETECTION_
#include <vector>
#include <fstream>
#include <assert.h>
#include "instr_info.h"
#include "omp.h"

#define BATCH_SIZE 5000

class _4instype_detect
{
public:
  double cf_num; // control flow instrucitons
  double fp_num; // floating point instructions
  double in_num; // integer instructions
  double mem_num;  // memory-related instructions
  double sh_num; // shift instructions
  double con_num; // conditional instructions
  double tr_num; // truncation instructions
  double ow_num; // overwriting instructions
 
  _4instype_detect();
  void detector(vector<instr_info> &instr_info_set, ofstream &ou,bool signal);
  virtual ~_4instype_detect(void);

};

#endif 
