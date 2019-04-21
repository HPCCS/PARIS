#ifndef _SELF_ADD_TAB_H_
#define _SELF_ADD_TAB_H_

#include <string>
#include <vector>
#include <fstream>
#include "instr_info.h"
#include "assert.h"
#include "b_tree.h"
//#include "addr2regs.h" 

using namespace std;

struct self_add
{
  string head;
  vector<string> tail;
};

class self_add_tab
{
public:
  vector<self_add> tab; // the self-add list

  self_add_tab();
  virtual ~self_add_tab(void);
  void tab_builder(vector<instr_info> &add_sub_list, ofstream &ou);
};

#endif
