# please export BATCH_SIZE 
# and LOC_ARR_SIZE
# before execution. 
# LOC_ARR_SIZE should divide
# BATCH_SIZE*0.8 without 
# remaider 

EXE=deadloc
CC = g++ -std=c++11 -I/usr/apps/gnu/4.9.3
C_INC = 
#CFLAGS =   -std=c++11 -Wall -O0 -g
CFLAGS = -std=c++11 -fopenmp -O3
#LDFLAGS+=-L/g/g90/guo7/graphviz/lib -lgvc -lcgraph  -lcdt
CLINK = ${CC} 
#CLINKFLAGS = -g 


SRCS = main.cpp  deadloc_detect.cpp loc_arr.cpp

OBJS=$(patsubst %.cpp, %.o, $(SRCS))

$(EXE): main.o deadloc_detect.o loc_arr.o
	$(CC) main.o deadloc_detect.o loc_arr.o -fopenmp -std=c++11 -o $@ #$(LDFLAGS)

main.o: main.cpp instr_info.h deadloc_detect.h
	$(CC) -c  main.cpp -o $@ $(CFLAGS)

deadloc_detect.o: deadloc_detect.cpp
	$(CC) -c deadloc_detect.cpp -fopenmp -std=c++11 -o $@ $(CFLAGS)

loc_arr.o: loc_arr.cpp loc_arr.h
	$(CC) -c loc_arr.cpp -fopenmp -std=c++11 -o $@ $(CFLAGS)

clean:
	rm -f *.o ${EXE}
