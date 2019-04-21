/* resilience patter detection
 * repeatadd_detection
 */
#include "repeatadd_detect.h"
#include "omp.h"

/*/ if elem in vec
template <class T>
bool isInOrNot(vector<T> &vec, T elem){
  int sz = vec.size();
  for(int i=0;i<sz;i++){
    if(vec[i]==elem){
      return true;
      break;
    }else{
      continue;
    }
  }
  return false;
}
*/

// if elem in vec
template <class T>
bool isInOrNot(vector<T> &vec, T elem){
  int sz = vec.size();
  bool bl[sz];
  int i;
  bool rt=false;
  #pragma omp parallel
  {
  #pragma omp parallel for shared(vec,bl) private(i)
  for(i=0;i<sz;i++){
    if(vec[i]==elem){
      bl[i]=true;
    }else{
      bl[i]=false;
    }
  }
  #pragma omp parallel for shared(bl) private(i)
  for(i=0;i<sz;i++){
    rt=rt||bl[i];
  }
  }
  return rt;
}


repeatadd_detect::repeatadd_detect(){
  //num = 0;
}

repeatadd_detect::~repeatadd_detect(void){

}

void repeatadd_detect::detector(vector<instr_info> &instr_info_set, ofstream &ou){
  int sz = instr_info_set.size();
  int opid;
  for(int i=0;i<sz;i++){
    //printf("the curr dyn instr id is: %d\n", instr_info_set[i].dynInstId);
    opid =  instr_info_set[i].opcodeId;

    if(opid==27||opid==28){
      if(opid==27){
	assert(instr_info_set[i].oprd_line_set[0].arguId=="1" && "BUG: bug in the format of load operation!\n");
	long long addr = (long long)instr_info_set[i].oprd_line_set[0].dynValue;
	string reg = instr_info_set[i].oprd_line_set[1].regName;
	addr2regs[reg] = addr;
/*
	if(addr2regs.find(reg)==addr2regs.end()){
  	  vector<dynid_mem> dynid_mem_tab;
	  dynid_mem a_dynid_mem;
	  a_dynid_mem.dynid = instr_info_set[i].dynInstId;
	  a_dynid_mem.mem = addr;
	  dynid_mem_tab.push_back(a_dynid_mem);
	  addr2regs[reg] = dynid_mem_tab;
	}else{
	  dynid_mem a_dynid_mem;
	  a_dynid_mem.dynid = instr_info_set[i].dynInstId;
	  a_dynid_mem.mem = addr;
	  addr2regs[reg].push_back(a_dynid_mem); 
 	}

	if(addr_regs.empty()){
	  addr2regs an_addr2regs;
	  an_addr2regs.addr = addr;
	  an_addr2regs.regs.push_back(reg);
	  addr_regs.push_back(an_addr2regs);
	}else{
	  int adsz = addr_regs.size();
	  bool is1 = false;
	  for(int j=0;j<adsz;j++){
	    if(addr_regs[j].addr==addr){
	      bool is2=isInOrNot(addr_regs[j].regs, reg);
	      if(is2){
		break;
	      }else{
		addr_regs[j].regs.push_back(reg);	
	      }
	      is1 = true;
	    }
	  }
	  if(!is1){
	    addr2regs an_addr2regs;
	    an_addr2regs.addr = addr;
	    an_addr2regs.regs.push_back(reg);
	    addr_regs.push_back(an_addr2regs);
	  }
	}
*/	
      }
      if(opid==28){
	assert(instr_info_set[i].oprd_line_set[0].arguId=="2" && "BUG: bug in the format of load operation!\n");
	long long addr = (long long)instr_info_set[i].oprd_line_set[0].dynValue;
	string reg = instr_info_set[i].oprd_line_set[1].regName;
	addr2regs[reg] = addr;
/*
  	if(addr2regs.find(reg)==addr2regs.end()){
  	  vector<dynid_mem> dynid_mem_tab;
	  dynid_mem a_dynid_mem;
	  a_dynid_mem.dynid = instr_info_set[i].dynInstId;
	  a_dynid_mem.mem = addr;
	  dynid_mem_tab.push_back(a_dynid_mem);
	  addr2regs[reg] = dynid_mem_tab;
	}else{
	  dynid_mem a_dynid_mem;
	  a_dynid_mem.dynid = instr_info_set[i].dynInstId;
	  a_dynid_mem.mem = addr;
	  addr2regs[reg].push_back(a_dynid_mem); 
 	}

	if(addr_regs.empty()){
	  addr2regs an_addr2regs;
	  an_addr2regs.addr = addr;
	  an_addr2regs.regs.push_back(reg);
	  addr_regs.push_back(an_addr2regs);
	}else{
	  int adsz = addr_regs.size();
	  bool is1 = false;
	  for(int j=0;j<adsz;j++){
	    if(addr_regs[j].addr==addr){
	      bool is2=isInOrNot(addr_regs[j].regs, reg);
	      if(is2){
		break;
	      }else{
		addr_regs[j].regs.push_back(reg);	
	      }
	      is1 = true;
	    }
	  }
	  if(!is1){
	    addr2regs an_addr2regs;
	    an_addr2regs.addr = addr;
	    an_addr2regs.regs.push_back(reg);
	    addr_regs.push_back(an_addr2regs);
	  }	
      }
*/
    }
    }

    // attaching memory address to registers in 
    // the following operations
    if((opid>=12&&opid<=16)||opid==8||opid==9||opid==10||opid==11){
      int sz = instr_info_set[i].oprd_line_set.size();
      
      for(int j=0;j<sz;j++){
	string tmpreg = instr_info_set[i].oprd_line_set[j].regName;
	if(tmpreg!=""){
	  if(addr2regs.find(tmpreg)==addr2regs.end()){
	    continue;
	  }else{
	    instr_info_set[i].oprd_line_set[j].addr = addr2regs[tmpreg];
	  }
	}
      }
    }

    // Building operation list for the next
    // pattern analyses
    if(opid>=12&&opid<=16){
      add_list.push_back(instr_info_set[i]);
      sub_list.push_back(instr_info_set[i]);
    }else if(opid==8||opid==9){
      add_list.push_back(instr_info_set[i]);
    }else if(opid==10||opid==11){
      sub_list.push_back(instr_info_set[i]);
    }else{
      continue;
    }
  }
}


