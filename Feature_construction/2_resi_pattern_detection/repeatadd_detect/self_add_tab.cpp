#include "self_add_tab.h"
#include "omp.h"

self_add_tab::self_add_tab(){
}

self_add_tab::~self_add_tab(){
}

void self_add_tab::tab_builder(vector<instr_info> &add_sub_list, ofstream &ou){
  // reading the back tracing
  // threshold from env var
  char* bk_thred = getenv("TRACE_BACK_THRED");
  int bkth = atoi(bk_thred);

  // tracing back through the 
  // add list or sub list
  int sz = add_sub_list.size();

  for(int i=sz-1;i>=0;i--){
    assert(add_sub_list[i].oprd_line_set[2].arguId=="r" && "BUG: mistake in result operand of Addition!");
    btree a_btree(add_sub_list[i].oprd_line_set[2].dynValue, add_sub_list[i].oprd_line_set[2].regName,add_sub_list[i].oprd_line_set[2].addr);
    string tmp_reg1 = add_sub_list[i].oprd_line_set[0].regName;
    string tmp_reg2 = add_sub_list[i].oprd_line_set[1].regName;
    double tmp_val1 = add_sub_list[i].oprd_line_set[0].dynValue;
    double tmp_val2 = add_sub_list[i].oprd_line_set[1].dynValue;
    long long tmp_addr1 = add_sub_list[i].oprd_line_set[0].addr;
    long long tmp_addr2 = add_sub_list[i].oprd_line_set[1].addr;
    a_btree.insert(tmp_val1, tmp_val2, tmp_reg1, tmp_reg2, a_btree.root,tmp_addr1,tmp_addr2);
    if(add_sub_list[i].opcodeId>=8&&add_sub_list[i].opcodeId<=11){
      a_btree.add_counter++;
    }
    if((a_btree.root->reg==tmp_reg1||a_btree.root->reg==tmp_reg2)&&a_btree.root->reg!=""&&a_btree.add_counter>0){
      self_add a_self_add;
      a_self_add.head=a_btree.root->reg;
      a_btree.deposit_tree(a_self_add.tail);
      tab.push_back(a_self_add);
      //a_btree.destroy_tree();
      //ou<<a_self_add.head<<" ";
      //int siz = a_self_add.tail.size();
      //for(int x=0;x<siz;x++){
	//ou<< a_self_add.tail[x]<< " ";
      //}
      //ou << endl;
      continue;
    }else{
      for(int j=i-1;j>=i-1-bkth;j--){
	if(j<0)
	  break; 
	assert(add_sub_list[j].oprd_line_set[2].arguId=="r" && "BUG:mistake in result operand of Addition!");
	a_btree.collect_leafs();
	int leaf_sz = a_btree.leaf_nodes.size();
        string pre_res_reg = add_sub_list[j].oprd_line_set[2].regName;
	long long pre_addr = add_sub_list[j].oprd_line_set[2].addr;
	for(int k=0;k<leaf_sz;k++){
	  // TODO: compare by mem addr
	  // find mem addr
	  string leaf_reg = a_btree.leaf_nodes[k]->reg;
	  long long leaf_addr = a_btree.leaf_nodes[k]->addr;
	  //bool is = isInOrNot(addr_regs[addr_idx].regs,pre_res_reg);
	  if((leaf_reg==pre_res_reg||leaf_addr==pre_addr)&&leaf_reg!=""){
	    node *inse = a_btree.search(leaf_reg, leaf_addr);
    	    string tmp_reg3 = add_sub_list[j].oprd_line_set[0].regName;
    	    string tmp_reg4 = add_sub_list[j].oprd_line_set[1].regName;
    	    double tmp_val3 = add_sub_list[j].oprd_line_set[0].dynValue;
    	    double tmp_val4 = add_sub_list[j].oprd_line_set[1].dynValue;
    	    long long tmp_addr3 = add_sub_list[j].oprd_line_set[0].addr;
    	    long long tmp_addr4 = add_sub_list[j].oprd_line_set[1].addr;
	    a_btree.insert(tmp_val3,tmp_val4,tmp_reg3,tmp_reg4,inse,tmp_addr3,tmp_addr4);
    	    if(add_sub_list[j].opcodeId>=8&&add_sub_list[j].opcodeId<=11){
      	      a_btree.add_counter++;
    	    }
	    if(((a_btree.root->reg==tmp_reg3||a_btree.root->reg==tmp_reg4)&&a_btree.add_counter>0)||((a_btree.root->addr==tmp_addr3||a_btree.root->addr==tmp_addr4)&&a_btree.add_counter>0)){
	      self_add a_self_add;
	      a_self_add.head=a_btree.root->reg;
	      a_btree.deposit_tree(a_self_add.tail);
	      tab.push_back(a_self_add);
	      //a_btree.destroy_tree();
      	      //ou<<a_self_add.head<<" ";
      	      //int siz = a_self_add.tail.size();
     	      //for(int x=0;x<siz;x++){
		//ou<< a_self_add.tail[x]<< " ";
      	      //}
      	      //ou << endl;
	      break;
	    }else{
	      continue;
	    }
	  }else{
	    continue;
	  }
	}	  
      }
    }
    a_btree.destroy_tree();
  }
  
}
