#include "b_tree.h"
#include "omp.h"

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

void btree::collect_leafs(){
  this->leaf_nodes.clear();
  collect_leafs(leaf_nodes,this->root);
/* 
 int xx = addr_regs.size();
  int zz = leaf_nodes.size();
  for(int i=0;i<xx;i++){
    int yy = addr_regs[i].regs.size();
    for(int j=0;j<yy;j++){
      for(int k=0;k<zz;k++){
	if(leaf_nodes[k]->reg==addr_regs[i].regs[j]){
	  leaf_nodes[k]->addr = i;//addr index is assigned
	}
      }
    }
  }
*/
}

void btree::collect_leafs(vector<node *> &leaf_nodes, node *leaf){
  if(leaf!=NULL){
    if(leaf->left==NULL&&leaf->right==NULL){
      leaf_nodes.push_back(leaf);
    }else{
      collect_leafs(leaf_nodes,leaf->left);
      collect_leafs(leaf_nodes,leaf->right);
    }
  }else{
    printf("BUG: the current node is NULL!");
  }
}

btree::btree(double a, string b, long long c)
{
  root = new node();
  root->key_value = a;
  root->reg = b;
  root->addr = c;
  root->left = NULL;
  root->right = NULL;
  
  add_counter = 0;
}

void btree::destroy_tree(node *leaf)
{
  if(leaf!=NULL)
  {
    destroy_tree(leaf->left);
    destroy_tree(leaf->right);
    delete leaf;
    leaf = NULL;
  }
}

void btree::deposit_tree(vector<string> &tail, node *leaf){
  if(leaf->left!=NULL){
    if(tail.empty())
      tail.push_back(leaf->left->reg);
    else{
      bool is = isInOrNot(tail,leaf->left->reg);
      if(!is){
	tail.push_back(leaf->left->reg);
      }
    }
    deposit_tree(tail, leaf->left);
  }
  if(leaf->right!=NULL){
    if(tail.empty())
      tail.push_back(leaf->right->reg);
    else{
      bool is = isInOrNot(tail,leaf->right->reg);
      if(!is){
	tail.push_back(leaf->right->reg);
      }
    }
    deposit_tree(tail, leaf->right);
  }
}

void btree::insert(double key1, double key2, string reg1, string reg2,  node *leaf, long long addr1, long long addr2)
{
  if(key1<=key2){
    //assert(leaf->left==NULL && "BUG: left leaf of current node is not NULL");
    
    if(leaf->left==NULL){
        leaf->left = NULL;
	leaf->left = new node;
        leaf->left->key_value=key1;
        leaf->left->reg=reg1;
        leaf->left->addr = addr1;
        leaf->left->left=nullptr;
        leaf->left->right=nullptr;
    }else{
	printf("BUG: left leaf of current node is not NULL");
    }
    //node n1;
    //n1.key_value=key1;
    //n1.reg=reg1;
    //n1.left=nullptr;
    //n1.right=nullptr;
    //leaf->left = &n1;
    assert(leaf->right==NULL && "BUG: right leaf of current node is not NULL");
    leaf->right = new node();
    leaf->right->key_value=key2;
    leaf->right->reg=reg2;
    leaf->right->addr = addr2;
    leaf->right->left=nullptr;
    leaf->right->right=nullptr;
    
    //node n2;
    //n2.key_value=key2;
    //n2.reg=reg2;
    //n2.left=nullptr;
    //n2.right=nullptr;
    //leaf->right = &n2;
  }else{
    assert(leaf->left==NULL && "BUG: left leaf of current node is not NULL");
    leaf->left = new node();
    leaf->left->key_value=key2;
    leaf->left->reg=reg2;
    leaf->left->addr = addr2;
    leaf->left->left=nullptr;
    leaf->left->right=nullptr;
    //node n2;
    //n2.key_value=key2;
    //n2.reg=reg2;
    //n2.left=nullptr;
    //n2.right=nullptr;
    //leaf->left = &n2;
    assert(leaf->right==NULL && "BUG: right leaf of current node is not NULL");
    leaf->right = new node();
    leaf->right->key_value=key1;
    leaf->right->reg=reg1;
    leaf->right->addr = addr1;
    leaf->right->left=nullptr;
    leaf->right->right=nullptr;
    //node n1;
    //n1.key_value=key1;
    //n1.reg=reg1;
    //n1.left=nullptr;
    //n1.right=nullptr;
    //leaf->right = &n1;
  }
}

node *btree::search(string reg, long long addr, node *leaf)
{
  if(leaf!=NULL)
  {
    if(reg==leaf->reg&&addr==leaf->addr&&leaf->right==NULL&&leaf->left==NULL)
      return leaf;
    else{
      node *tmp = search(reg,addr, leaf->left);
      if(tmp==NULL){
	return search(reg,addr, leaf->right);
      }else{
	return tmp;
      }
    }
  }
  else{
    return NULL;
  }
}

//void btree::insert(int key)
//{
  //if(root!=NULL)
    //insert(key, root);
 // else
  //{
    //root=new node;
   // root->key_value=key;
   // root->left=NULL;
   // root->right=NULL;
 // }
//}

node *btree::search(string reg, long long addr)
{
  return search(reg,addr, root);
}

void btree::deposit_tree(vector<string> &tail){
  return deposit_tree(tail, root);
}

void btree::destroy_tree()
{
  destroy_tree(root);
  this->leaf_nodes.clear();
}

btree::~btree(){
}
