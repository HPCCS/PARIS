/** _4instype_detection
 * there are four types of instructions
 * 1) Control fpow changing instructions
 * 2) Floating point instructions
 * 3) Integer instructions
 * 4) Memory-related instructions
 */

#include "4instype_detect.h"

_4instype_detect::_4instype_detect(){
  cf_num = 0;
  fp_num = 0;
  in_num = 0;
  mem_num = 0;
  sh_num =0;
  con_num=0;
  tr_num=0;
  ow_num=0;
}

_4instype_detect::~_4instype_detect(void){

}

void _4instype_detect::detector(vector<instr_info> &instr_info_set, ofstream &ou,bool signal){
  int sz = instr_info_set.size();
  int opid;
  //printf("batch size is %d\n", sz);
/*
  #pragma omp parallel
  {   
    int tid = omp_get_thread_num();
    if(tid==0){ 
        nthread = omp_get_num_threads();
        printf("the number of threads %d ...\n", nthread);
    }
  }
*/
//  #pragma omp parallel shared(instr_info_set)
  {
  //  #pragma omp for
    for(int i=0;i<sz;i++){
      opid =  instr_info_set[i].opcodeId;
      ow_num++;
      switch(opid){
	case 20:
		{int tmp = instr_info_set[i].oprd_line_set[0].dynValue;
    		assert(instr_info_set[i].oprd_line_set[0].arguId == "2" && "ERROR: choosing the wrong operand!");
		sh_num+=tmp;
		break;}
	case 21:
		{int tmp = instr_info_set[i].oprd_line_set[0].dynValue;
    		assert(instr_info_set[i].oprd_line_set[0].arguId == "2" && "ERROR: choosing the wrong operand!");
		sh_num+=tmp;
		break;}
	case 22:		
		{int tmp = instr_info_set[i].oprd_line_set[0].dynValue;
    		assert(instr_info_set[i].oprd_line_set[0].arguId == "2" && "ERROR: choosing the wrong operand!");
		sh_num+=tmp;
		break;}
	case 2: 
		cf_num+=0.333;
		break;
	case 3:
		con_num+=0.25;
		break;
 	case 46:
		con_num+=0.75;
		break;
	case 47:
		con_num+=0.75;
		break;
	case 23:
		con_num+=0.833;
		break;
	case 24:
		con_num+=0.833;
		break;
	case 25:
		con_num+=0.333;
		break;
	case 4: 
		cf_num+=0.333;
		break;
	case 33:
		tr_num+=0.75;
		break;
	case 40:
		tr_num+=0.75;
		break;
	case 50: 
		cf_num+=0.25;
		break;
	case 48: 
		cf_num+=0.25;
		break;
	case 30: 
		cf_num+=0.25;
		break;
	case 49: 
		{int tmp=instr_info_set[i].oprd_line_set.size()/2;
		cf_num=tmp/(tmp+1);
		break;}
	case 9: 
		fp_num+=0.333;
		break;
	case 11: 
		fp_num+=0.333;
		break;
	case 13: 
		fp_num+=0.333;
		break;
	case 16: 
		fp_num+=0.333;
		break;
	case 19: 
		fp_num+=0.333;
		break;
	case 102: 
		fp_num+=0.25;
		break;
	case 103: 
		fp_num+=0.25;
		break;
	case 8: 
		in_num+=0.333;
		break;
	case 10: 
		in_num+=0.333;
		break;
	case 12: 
		in_num+=0.333;
		break;
	case 14: 
		in_num+=0.333;
		break;
	case 15: 
		in_num+=0.333;
		break;
	case 17: 
		in_num+=0.333;
		break;
	case 18: 
		in_num+=0.333;
		break;
	case 27: 
		mem_num+=0.25;
		break;
	case 28: 
		mem_num+=0.25;
		break;
	case 98: 
		mem_num+=0.25;
		break;
	case 99: 
		mem_num+=0.25;
		break;
   	case 29: 
		mem_num+=0.333;
		break;
   	case 54: 
		mem_num+=0.333;
		break;
   	case 55: 
		mem_num+=0.333;
		break;
   	case 57: 
		mem_num+=0.333;
		break;
   	case 58: 
		mem_num+=0.333;
		break;	
	case 97: 
		cf_num+=0.25;
		break;
      }
      if(((opid==28)&&(instr_info_set[i].oprd_line_set[0].regName=="az"||instr_info_set[i].oprd_line_set[1].regName=="az"))){
    	//assert(instr_info_set[i].oprd_line_set[0].arguId == "2" && "ERROR: choosing the wrong operand!");
        ou<<sh_num<<" "<<con_num<<" "<<tr_num<<" "<<ow_num<<" "<<cf_num<<" "<<fp_num<<" "<<in_num<<" "<<mem_num<<endl;
	cf_num=0;
	sh_num=0;
	ow_num=0;
	tr_num=0;
	fp_num=0;
	con_num=0;
	in_num=0;
	mem_num=0;
      }
      if(signal){
        if(i==(sz-1)){
    	  //assert(instr_info_set[i].oprd_line_set[0].arguId == "2" && "ERROR: choosing the wrong operand!");
          ou<<sh_num<<" "<<con_num<<" "<<tr_num<<" "<<ow_num<<" "<<cf_num<<" "<<fp_num<<" "<<in_num<<" "<<mem_num<<endl;
	  cf_num=0;
	  sh_num=0;
	  ow_num=0;
	  tr_num=0;
	  fp_num=0;
	  con_num=0;
	  in_num=0;
	  mem_num=0;
	}
      }
    }
  }
}


