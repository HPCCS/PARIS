/* resilience patter detection
 * deadloc_detection
 */
#include "deadloc_detect.h"
#include "omp.h"

res_batch::~res_batch(){

}

deadloc_detect::deadloc_detect(){
  num = 0;
}

deadloc_detect::~deadloc_detect(void){

}


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


void deadloc_detect::detector(vector<instr_info> &instr_info_set, ofstream &ou, string str1, string str2, bool ifend){
  //int no_batch = sz/lsz; // number of batches
  // to control the size of batch, due to the varying size of chunks
  // a threshold is set up to automatically select different ways
  // to do it.
  int sz = instr_info_set.size();
  int lsz;
  int no_batch;
  if(sz<=6000){
    no_batch = 10; // number of batches
    lsz = sz/10; // number of instr in a batch
  }else{
    lsz = 1000;
    no_batch=(sz/1000)+1;
  }

  vector<loc_arr> loc_arr_set; // the set of location arrays

  for(int i=0;i<no_batch;i++){
    if((lsz+i*lsz)<sz){
      vector<instr_info> instr_batch_tmp(instr_info_set.begin()+i*lsz, instr_info_set.begin()+lsz+i*lsz);
      loc_arr a_loc_arr(instr_batch_tmp);
      loc_arr_set.push_back(a_loc_arr);    
      instr_batch_tmp.clear();
    }else{
      vector<instr_info> instr_batch_tmp(instr_info_set.begin()+i*lsz, instr_info_set.end());
      loc_arr a_loc_arr(instr_batch_tmp);
      loc_arr_set.push_back(a_loc_arr);    
      instr_batch_tmp.clear();
    }
    
    // output the set of loc arrays into ou 
    /*
    int rasz = a_loc_arr.reg_arr.size();
    for(int j=0;j<rasz;j++){
      ou << a_loc_arr.reg_arr[j] << " ";
    }
    ou << endl;

    int aasz = a_loc_arr.addr_arr.size();
    for(int j=0;j<aasz;j++){
      ou << a_loc_arr.addr_arr[j] << " ";
    }
    ou << endl;
    */
  }
  //ou << endl;
 

  // compare between loc arrays, e.g., 1 with 2, 3, 4, ...
  int loc_set_sz = loc_arr_set.size();
  for(int j=0;j<loc_set_sz-1;j++){
    // # dead locs: no_reg_dead_loc
    // and no_addr_dead_loc
    // and no_dead_loc = no_reg_dead_loc+no_addr_dead_loc
    int no_reg_dead_loc = 0;
    int no_addr_dead_loc = 0;
    //int no_dead_loc = 0;

    // comparing reg_arr
    int reg_sz = loc_arr_set[j].reg_arr.size();
    int m, k;
    string curr_reg;
    bool iscopy=false;

    //#pragma omp parallel for schedule(dynamic,1) collapse(2)
    for(m=0;m<reg_sz;m++){
      curr_reg = loc_arr_set[j].reg_arr[m];
      iscopy = false;	
      
      // comparing the current reg with the next locs
      //#pragma omp parallel
      {
      //#pragma omp parallel for  private(k)
      for(k=j+1;k<j+5;k++){
	if(k>=loc_set_sz){
	  break;
	}
	iscopy = iscopy||isInOrNot(loc_arr_set[k].reg_arr, curr_reg);
	//if(iscopy == true){
	  // find a use of curr_reg in next baches
	  // therefore curr_reg is not dead loc
	  //break;
	//}
      }
      }
      if(iscopy==false){
	// a dead loc is found
	// reg dead counter++
	no_reg_dead_loc++;
      }else{
	continue;
      }
    }

    // comparing addr_arr
    int addr_sz = loc_arr_set[j].addr_arr.size();
    for(int m=0;m<addr_sz-1;m++){
      string curr_addr = loc_arr_set[j].addr_arr[m];
      bool iscopy1 = false;	

      // comparing the current addr with the next locs
      //#pragma omp parallel
      {
      //#pragma omp parallel for  private(k)
      for(int k=j+1;k<j+5;k++){
	if(k>=loc_set_sz){
	  break;
	}
	iscopy1 = iscopy1||isInOrNot(loc_arr_set[k].addr_arr, curr_addr);
	//if(iscopy1 == true){
	  // find a use of curr_addr in next baches
	  // therefore curr_addr is not dead loc
	  //break;
	//}
      }
      }
      if(iscopy1==false){
	// a dead loc is found
	// addr dead counter++
	no_addr_dead_loc++;
      }else{
	continue;
      }
   
   }
   double tmp1=no_reg_dead_loc+no_addr_dead_loc;
   double tmp2=reg_sz+addr_sz;	   
   double deadloc_ratio_batch = tmp1/tmp2;
   res_batch a_batch_res(no_reg_dead_loc, reg_sz, no_addr_dead_loc, addr_sz, deadloc_ratio_batch);
   dead_loc_ratio.push_back(a_batch_res);

  }
  	
  // output the final resuts to the ou stream
  if(str1=="az"||str2=="az"||ifend){
  //ou << endl;
  ou << "****** the dead loc details for batches ******" << endl;
  int fsz = dead_loc_ratio.size();
  double dead=0;
  double all=0;
  for(int i=0;i<fsz;i++){
    ou << dead_loc_ratio[i].dead_reg << " " << dead_loc_ratio[i].all_reg << " "\
	<< dead_loc_ratio[i].dead_addr << " " << dead_loc_ratio[i].all_addr << " "\ 
	 << dead_loc_ratio[i].ratio << endl;
    dead+=(dead_loc_ratio[i].dead_reg+dead_loc_ratio[i].dead_addr);
    all+=(dead_loc_ratio[i].all_reg+dead_loc_ratio[i].all_addr+0.0001);
  }
  double tmp3=dead/all;
  ou << "overall ratio: "<< tmp3 << endl;
  dead_loc_ratio.clear();
  
  int locsz=loc_arr_set.size();
  for(int i=0;i<locsz;i++){
    loc_arr_set[i].reg_arr.clear();
    loc_arr_set[i].addr_arr.clear();
  }
  loc_arr_set.clear();

  }
}



