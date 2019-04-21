/* repeatadding pattern detection
 * the main function
 */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <time.h>
#include <string>
#include <stdlib.h>
#include <fstream>
//#define NUM_SELF_ADD 10
//#define BATCH_SIZE 5000

#include "omp.h"
#include "repeatadd_detect.h"
#include "instr_info.h"
#include "self_add_tab.h"
using namespace std;

int num_self_add;
int batch_sz;
map<string, int> loop_wise_counter;

//int tab_processing(repeatadd_detect a_repeatadd_detect, ofstream &ou, vector<int> &radd, vector<int> &rsub){
void tab_processing(repeatadd_detect a_repeatadd_detect, ofstream &ou, string str1, string str2, bool ifend){
//int no_pattern=0;
self_add_tab a_self_add_tab;
a_self_add_tab.tab_builder(a_repeatadd_detect.add_list, ou);

self_add_tab a_self_sub_tab;
a_self_sub_tab.tab_builder(a_repeatadd_detect.sub_list, ou);

a_repeatadd_detect.add_list.clear();
a_repeatadd_detect.sub_list.clear();

int counter = 0, i;
// a_self_add_tab.tab & a_self_sub_tab.tab
int a_sz = a_self_add_tab.tab.size();
if(a_sz>0)
{
// a temp data obj to store tmp status
//self_add tmp_self_add;
//tmp_self_add.head = a_self_add_tab.tab[0].head;
//tmp_self_add.tail.insert(tmp_self_add.tail.begin(),a_self_add_tab.tab[0].tail.begin(), a_self_add_tab.tab[0].tail.end());
//#pragma omp parallel for
for(i=0;i<a_sz;i++){
  //int sz1=a_self_add_tab.tab[i].tail.size();
  //int sz2=tmp_self_add.tail.size();
  //if(tmp_self_add.head==a_self_add_tab.tab[i].head&&sz1==sz2i){
  //if(tmp_self_add.head==a_self_add_tab.tab[i].head){
    //int tab_sz = a_self_add_tab.tab[i].tail.size();
    //if(a_self_add_tab.tab[i].tail[1]==tmp_self_add.tail[1]&& 
	//a_self_add_tab.tab[i].tail[tab_sz-1]==tmp_self_add.tail[tab_sz-1]){
     // counter++;
    //}else{
      // how many times each repeatadd pattern appears
      // #of elem in radd: #of kinds for patterns
      // the value of elem in radd: #of times of self_add patterns
      //loop_wise_counter[tmp_self_add.head]+=counter;
      //radd.push_back(counter);
      //counter=0;
    //}
 // }else{
    loop_wise_counter[a_self_add_tab.tab[i].head]+=1;
    //radd.push_back(counter);
    //counter=0;
  //}
  //tmp_self_add.tail.clear();
  //tmp_self_add.head = a_self_add_tab.tab[i].head;
  //tmp_self_add.tail.insert(tmp_self_add.tail.begin(),a_self_add_tab.tab[i].tail.begin(), a_self_add_tab.tab[i].tail.end());
}
// when there is only one repeated add pattern
// or for the last repeated add pattern
//if(counter>0)
  //radd.push_back(counter);
//}else{
  //printf("WARMING: there is no elem in SELF_ADD_TAB!\n");
//}
}
int s_sz = a_self_sub_tab.tab.size();
// a temp data obj to store tmp status
if(s_sz>0)
{
//self_add tmp_self_sub;
//tmp_self_sub.head = a_self_sub_tab.tab[0].head;
//tmp_self_sub.tail.insert(tmp_self_sub.tail.begin(),a_self_sub_tab.tab[0].tail.begin(), a_self_sub_tab.tab[0].tail.end());
//int counter = 0, i;
//pragma omp parallel for
for(i=0;i<s_sz;i++){
  //int sz1=a_self_sub_tab.tab[i].tail.size();
  //int sz2=tmp_self_sub.tail.size();
  //if(tmp_self_sub.head==a_self_sub_tab.tab[i].head&&sz1==sz2){
    //int tab_sz = a_self_sub_tab.tab[i].tail.size();
    //if(a_self_sub_tab.tab[i].tail[1]==tmp_self_sub.tail[1]&&
	//a_self_sub_tab.tab[i].tail[tab_sz-1]==tmp_self_sub.tail[tab_sz-1]){
      //counter++;
    //}else{
      // how many times each repeatadd pattern appears
      // #of elem in radd: #of kinds for patterns
      // the value of elem in radd: #of times of self_add patterns
      //rsub.push_back(counter);
      //counter=0;
    //}
  //}else{
    loop_wise_counter[a_self_sub_tab.tab[i].head]-=1;
    //rsub.push_back(counter);
    //counter=0;
  //}
  //tmp_self_sub.tail.clear();
  //tmp_self_sub.head = a_self_sub_tab.tab[i].head;
  //tmp_self_sub.tail.insert(tmp_self_sub.tail.begin(),a_self_sub_tab.tab[i].tail.begin(), a_self_sub_tab.tab[i].tail.end());
}
// when there is only one repeated add pattern
// or for the last repeated add pattern
//if(counter>0)
  //rsub.push_back(counter);
//}else{
  //printf("WARMING: there is no elem in SELF_ADD_TAB!\n");
//}
}
// apply the threshold, 10 in default
// that how many self_add make a repeated add 
// no_pattern: #total repeat add patterns
//int rasz = radd.size();
//int rssz = rsub.size();
//ou<<endl;
//ou<<"Num of each self addition pattern:"<<endl;
//ou<<"for add-list"<<endl;
//vector<int> num_self;
int tmp;
if(str1=="az"||str2=="az"||ifend){
  for(map<string,int>::iterator it = loop_wise_counter.begin(); it != loop_wise_counter.end(); ++it) {
    tmp=abs(it->second);
    if(tmp>=num_self_add){
      counter++;
    }
  }
  printf("DONE: the num of repeat add pattern is: %d\n", counter);
  ou<<counter<<endl;
  counter=0;
  loop_wise_counter.clear();
}

//int i;

//#pragma omp parallel
//{
//#pragma omp parallel for private(i)
//for(i=0;i<rasz;i++){
  //ou<<radd[i]<<" ";
  //if(radd[i]>=num_self_add){
    //no_pattern++;
  //}
//}
//ou<<endl;
//ou<<"for sub-list"<<endl;
//#pragma omp parallel for private(i)
//for(i=0;i<rssz;i++){
  //ou<<rsub[i]<<" ";
  //if(rsub[i]>num_self_add){
    //no_pattern++;
  //}
//}
//}

//ou<<endl;
//ou<<"DONE: the num of repeat add pattern totally is:"<< no_pattern <<endl;
a_self_add_tab.tab.clear();
a_self_sub_tab.tab.clear();
//radd.clear();
//rsub.clear();
//return no_pattern;
}


int main(){
  vector<instr_info> instr_info_set;
  repeatadd_detect a_repeatadd_detect;
  
  // results for repeatadd pattern
  //vector<int> radd;
  //vector<int> rsub;

  // get env vars
  char* cnum_self_add = getenv("NUM_SELF_ADD");
  num_self_add = atoi(cnum_self_add);
  char* c_batch_sz = getenv("BATCH_SIZE");
  batch_sz = atoi(c_batch_sz);

  // the file stream for dynamic trace
  ifstream in;
  string file(getenv("TRACE_FILE_NAME"));
  string infile="../NEW_TRACES/"+file;
  //string infile="./trace_files/"+file;
  in.open(infile.c_str(), ifstream::in);
  ofstream ou;
  string outfile="../NEW_results/"+file+"_repeatadd.txt";
  //string outfile=file+"_repeatadd.txt";
  ou.open(outfile.c_str(), ofstream::out);
/*
  ifstream in;
  string infile="./trace_files/triad_dynamic_trace";
  in.open(infile.c_str(), ifstream::in);
  ofstream ou;
  string outfile="./triad_result.txt";
  ou.open(outfile.c_str(), ofstream::out);
*/
  // 
  if(!in){
    cout<<"ERROR IN OPENNING THE TRACEFILE!"<<endl;
    return 1;
  }

  // reading the dynamic trace into memory
  //
  bool ifend;
  int final_pattern=0;
  //bool signal=false;
  string line, st1, st2;
  int i=0, n=0, len;
  printf("Begin reading the instruction trace...\n");
  while(true){
    //read info from the trace file
    getline(in, line);
    size_t yes_no, NumOfComma1, NumOfComma2;
    yes_no = line.find(",",0);
    if (yes_no == string::npos){
      if(i>0){
	++n;
      }

      // dynamic trace processing batch by batch
      //if ((n>=an_arguConfig->NUMBER_OF_BLOCKS_PROCESS_PER_BATCH)||in.eof()){
      int zs=instr_info_set.size();
      if(zs>1){
	if(instr_info_set[zs-1].opcodeId==28){
          st1=instr_info_set[zs-1].oprd_line_set[0].regName;
	  st2=instr_info_set[zs-1].oprd_line_set[1].regName;
	}
      }
      //if (n>=batch_sz||in.eof()){
      if (st1=="az"||st2=="az"||n>=batch_sz||in.eof()){
	// int k;
	// Stopping read of the trace, and starting trace analysis
	//len = instr_info_set.size();
        if(st1!="az"&&st2!="az"&&n<batch_sz){
	  ifend=true;
	}else{
	  ifend=false;
	}
	a_repeatadd_detect.detector(instr_info_set, ou);
  	tab_processing(a_repeatadd_detect, ou, st1, st2, ifend); 
	a_repeatadd_detect.addr2regs.clear();

	cout<<"the current inst id after the loop is: " << instr_info_set[zs-1].dynInstId  << endl;
/*
	if(st1!="az"&&st2!="az"&&n>=batch_sz){
	  final_pattern+=no_pattern;
	  //signal=true;
	}else if((st1=="az"||st2=="az")&&n<=batch_sz){
	  final_pattern+=no_pattern;
	  ou<< final_pattern <<endl;
	  printf("DONE: the num of repeat add pattern is: %d\n", final_pattern);
	  final_pattern=0;
	  //signal=false;
	}else{
	  final_pattern+=no_pattern;
	  ou<< final_pattern <<endl;
	  printf("DONE: the num of repeat add pattern is: %d\n", final_pattern);
 	  final_pattern=0;
	}
*/
	st1="";
	st2="";
	//no_pattern=0;
 	instr_info_set.clear();
	n = 0;
	instr_info an_instr_info;
	instr_info_set.push_back(an_instr_info);
/*
	for(k=0;k<len;k++){
	  int opcodeId = instr_info_set[k].opcodeId;
	  if (opcodeId < 0){
	    printf("Exception: disaster found %d!", instr_info_set[k].dynInstId);
	  }

	  /-* resilience pattern analysis
 	   * TODO: repeatadding
 	   *-/
	  // the size of trace block: sz
	  //int sz = instr_info_set.size();
	  //printf("the size of trace block is: %d \n", sz);
	  
	  // repeatadding pattern detection
	  a_repeatadd_detect.detector(instr_info_set, ou);

	  /-* resilience pattern detection DONE!
 	   *-/
           
	  // print out the execution states
	  if(instr_info_set[k].dynInstId%4998 == 0){
	    printf("The Tool has completed the %d instructions...\n", instr_info_set[k].dynInstId);
	  }

	  // start to free the memory space when 80% of the batch is hit
	  //if (k>=0.8*an_arguConfig->NUMBER_OF_BLOCKS_PROCESS_PER_BATCH){

	  if (k>=0.8*batch_sz){
	    instr_info_set.erase(instr_info_set.begin(), (instr_info_set.begin()+k));
	    n = instr_info_set.size();
	    instr_info an_instr_info;
	    instr_info_set.push_back(an_instr_info);
	    break;
	  }
	}
*/
	// Jump out of the loop while hitting the bottom of the file
	if (in.eof()){
	  //int n0 = instr_info_set.size();
	  //if (k >= n0-1){
	    break;
	  //}
	}
	continue;
      }else{
	// While 80% trace finished its analysis, stopping the analysis, and starting free the completed trace
	i = 0;
	instr_info an_instr_info;
	instr_info_set.push_back(an_instr_info);
	continue;
      }
      
    }else{
      NumOfComma1 = 0;
      NumOfComma2 = line.find(",", NumOfComma1);
      string blockId_abnom = line.substr(NumOfComma1, NumOfComma2).c_str();
      // assign to line-0/1/2/r
      if(blockId_abnom == "0") // assign to line 0
      {
	instr_info_set[n].blockId = line.substr(NumOfComma1, NumOfComma2).c_str();
	// the first field
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	instr_info_set[n].lineId = atoi(line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str());
	// the second field
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	instr_info_set[n].funcName = line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str();
	// the third field
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	instr_info_set[n].basicBlockId = line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str();
	// the fourth field
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	instr_info_set[n].staticInstId = line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str();
	// the fifth field
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	instr_info_set[n].opcodeId = atoi(line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str());
	// the sixth field
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	instr_info_set[n].dynInstId = atoi(line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str());
	//
	i++;
	continue;
      }else{
	oprd_line an_oprd_line;
	// the argueId
	an_oprd_line.arguId = line.substr(NumOfComma1, NumOfComma2).c_str();
	// the size of argue
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	an_oprd_line.sizeOfArgu = atoi(line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str());
	// the dynamic value
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	an_oprd_line.dynValue = atof(line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str());
	// is reg or not
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	an_oprd_line.regORnot = (atoi(line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str()) ==1);
	// the reg name
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	an_oprd_line.regName = line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str();
	// the pre-block id
	NumOfComma1 = NumOfComma2;
	NumOfComma2 = line.find(",", NumOfComma1 + 1);
	an_oprd_line.pre_block_id = line.substr(NumOfComma1+1, NumOfComma2-NumOfComma1-1).c_str();
 	// push back
 	instr_info_set[n].oprd_line_set.push_back(an_oprd_line);
	//
	i++;
	continue;
      }
    }
  }

// finishing reading, close the file stream
in.close();
ou.close();
return 0;

}

