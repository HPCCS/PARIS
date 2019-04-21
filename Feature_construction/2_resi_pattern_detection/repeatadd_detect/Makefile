#export OMP_NUM_THREADS=16
#export TRACE_BACK_THRED=100
#export NUM_SELF_ADD=10
#export BATCH_SIZE=5000

EXE=repeatadd
CC = g++ -std=c++11 -I/usr/apps/gnu/4.9.3
C_INC = 
CFLAGS = -fopenmp  -std=c++11 -Wall -O0 -g
#CFLAGS = -std=c++11 -fopenmp -O3 -Wall
#LDFLAGS+=-L/g/g90/guo7/graphviz/lib -lgvc -lcgraph  -lcdt
CLINK = ${CC} 
#CLINKFLAGS = -g 

SRCS = main.cpp  repeatadd_detect.cpp b_tree.cpp self_add_tab.cpp

OBJS=$(patsubst %.cpp, %.o, $(SRCS))

$(EXE): main.o repeatadd_detect.o b_tree.o self_add_tab.o
	$(CC) main.o repeatadd_detect.o b_tree.o self_add_tab.o -fopenmp -std=c++11 -O0 -o $@ #$(LDFLAGS)

main.o: main.cpp instr_info.h repeatadd_detect.h
	$(CC) -c  main.cpp -o $@ $(CFLAGS)

repeatadd_detect.o: repeatadd_detect.cpp
	$(CC) -c repeatadd_detect.cpp -fopenmp -std=c++11 -o $@ $(CFLAGS)

self_add_tab.o: self_add_tab.cpp self_add_tab.h
	$(CC) -c self_add_tab.cpp -fopenmp -std=c++11 -o $@ $(CFLAGS)

#addr2regs.o: addr2regs.cpp addr2regs.h
#	$(CC) -c addr2regs.cpp -fopenmp -std=c++11 -o $@ $(CFLAGS)

b_tree.o: b_tree.cpp b_tree.h
	$(CC) -c b_tree.cpp -o $@ $(CFLAGS)

clean:
	rm -f *.o ${EXE}
