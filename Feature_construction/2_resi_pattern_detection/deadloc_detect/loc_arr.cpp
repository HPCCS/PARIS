/* split dynamic trace into batches.
 * For each batch, there is a location
 * array, in which there is no copied
 * locations
 */
#include "loc_arr.h"
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

// only handle a single batch here (loc_arr_size)
loc_arr::loc_arr(vector<instr_info> &instr_batch){
  // for the single batch
  // collect all its reg and addr arrays
  int bsz = instr_batch.size();
  for(int i=0;i<bsz;i++){
    
    int oprd_sz = instr_batch[i].oprd_line_set.size(); 
    for(int j=0;j<oprd_sz;j++){
      string str_tmp = instr_batch[i].oprd_line_set[j].regName;
      //bool signal = false; // if there is repeat in reg_arr
      if(this->reg_arr.empty()){
        this->reg_arr.push_back(str_tmp);
      }else{
	//int reg_sz = reg_arr.size();
	//for(int k=0;k<reg_sz;k++){
	  //if(reg_arr[k]==str_tmp){
	    //signal = true;
	    //break;
	  //}else{
	    //continue;
	  //}
	//}
	if(isInOrNot(reg_arr,str_tmp) == false){
	  reg_arr.push_back(str_tmp);
	}else{
	  continue;
	}
      }
      // handling these operations that have mem addr inside
      // alloca, load, store, getelementptr
      // fence, cmpxchg, atomicrmw
      int tmp_opc = instr_batch[i].opcodeId;
      if(tmp_opc>=26&&tmp_opc<=32&&instr_batch[i].lineId>0){
	switch(tmp_opc){
	  case 26:{
	    // get the mem addr of the alloca from the only oprand
	    // assert(oprd_sz==1 && "\nBUG: alloc has more than 1 operand!\n");
	    string addr = to_string((int)instr_batch[i].oprd_line_set[0].dynValue);
	    if(addr_arr.empty()){
	      addr_arr.push_back(addr);
	    }else{
	      if(isInOrNot(addr_arr,addr)==false){
		addr_arr.push_back(addr);
	      }
	    }
	    break;
	  }
	  case 27:{
	    // get the mem addr in load 
	    // the oprand id is 1
	    for(int k=0;k<oprd_sz;k++){
	      if(instr_batch[i].oprd_line_set[k].arguId == "1"){
		string addr = to_string((int)instr_batch[i].oprd_line_set[k].dynValue);
		if(addr_arr.empty()){
		  addr_arr.push_back(addr);
		  break;
		}else{
		  if(isInOrNot(addr_arr,addr)==false){
		    addr_arr.push_back(addr);
		  }else{
		    break;
		  }
		}
	      }
	    }
	    break;
	  }
	  case 28:{
	    // get the mem addr in store
	    // the oprand id is 2
	    for(int k=0;k<oprd_sz;k++){
	      if(instr_batch[i].oprd_line_set[k].arguId == "2"){
		string addr = to_string((int)instr_batch[i].oprd_line_set[k].dynValue);
		if(addr_arr.empty()){
		  addr_arr.push_back(addr);
		  break;
		}else{
		  if(isInOrNot(addr_arr,addr)==false){
		    addr_arr.push_back(addr);
		  }else{
		    break;
		  }
		}
	      }
	    }
	    break;
	  }
	  case 29:{
	    // get the mem addr in getelementptr
	    // the oprand id is r
	    // becase the base mem id is added
	    // by load operation already
	    for(int k=0;k<oprd_sz;k++){
	      if(instr_batch[i].oprd_line_set[k].arguId == "r"){
		string addr = to_string((int)instr_batch[i].oprd_line_set[k].dynValue);
		if(addr_arr.empty()){
		  addr_arr.push_back(addr);
		  break;
		}else{
		  if(isInOrNot(addr_arr,addr)==false){
		    addr_arr.push_back(addr);
		  }else{
		    break;
		  }
		}
	      }
	    }
	    break;
	  }
	  case 30:
	    printf("******************************\n \
		   the Fence operation is found!!\n \
		   ******************************\n");
	    break;

	  case 31:
	    printf("********************************\n \
		   the CmpXchg operation is found!!\n \
		    ********************************\n");
	    break;

	  case 32:
	    printf("*********************************\n \
		   the atomicrmw operation is found!!\n \
		    *********************************\n");
	    break;
	}
      }	
    }
  }
}

loc_arr::~loc_arr(){

}
