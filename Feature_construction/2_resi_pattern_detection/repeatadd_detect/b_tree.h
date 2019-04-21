#ifndef _B_TREE_H_
#define _B_TREE_H_
#include <string>
#include <vector>
#include "assert.h"
#include "repeatadd_detect.h"

using namespace std;

template <class T>
bool isInOrNot(vector<T> &vec, T elem);

struct node
{
  double key_value;
  string reg;
  long long addr = 0;
  node *left=NULL;
  node *right=NULL;
};

class btree
{
    public:
        btree(double a, string b, long long c);
        virtual ~btree(void);
	vector<node*> leaf_nodes;
	int add_counter;

        //void insert(double key);
        node *search(string reg, long long addr);
        void destroy_tree();
	void deposit_tree(vector<string> &tail);
	void collect_leafs();

    //private:
        void destroy_tree(node *leaf);
        void insert(double key1, double key2,string reg1, string reg2, node *leaf, long long addr1, long long addr2);
        node *search(string reg, long long addr, node *leaf);
	void deposit_tree(vector<string> &tail, node *leaf);
 	void collect_leafs(vector<node*> &leaf_nodes, node *leaf);
        
        node *root;
};



#endif
