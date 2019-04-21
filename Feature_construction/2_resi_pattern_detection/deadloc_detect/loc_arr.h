#ifndef _LOC_ARR_H_
#define _LOC_ARR_H_

#include <vector>
#include <string>
#include "assert.h"

#include "instr_info.h"

class loc_arr
{
public:
  vector<string> reg_arr;
  vector<string> addr_arr;

  loc_arr(vector<instr_info> &instr_batch);
  virtual ~loc_arr(void);
};

#endif
