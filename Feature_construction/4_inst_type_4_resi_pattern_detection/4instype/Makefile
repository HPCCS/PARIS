#export OMP_NUM_THREADS=16

EXE=4instype
CC = g++ -std=c++11 -I/usr/apps/gnu/4.9.3
C_INC = 
#CFLAGS =   -std=c++11 -Wall -O0 -g
CFLAGS = -std=c++11 -O3
#LDFLAGS+=-L/g/g90/guo7/graphviz/lib -lgvc -lcgraph  -lcdt
CLINK = ${CC} 
#CLINKFLAGS = -g 
#MY_LIB=/opt/pasa_lib/graphviz-2.40/lib/pkgconfig

SRCS = main.cpp  4instype_detect.cpp

OBJS=$(patsubst %.cpp, %.o, $(SRCS))

$(EXE): main.o 4instype_detect.o
	$(CC) main.o 4instype_detect.o -fopenmp -std=c++11 -o $@ #$(LDFLAGS)

main.o: main.cpp instr_info.h 4instype_detect.h
	$(CC) -c  main.cpp -o $@ $(CFLAGS)

4instype_detect.o: 4instype_detect.cpp
	$(CC) -c 4instype_detect.cpp -fopenmp -std=c++11 -o $@ $(CFLAGS)

clean:
	rm -f *.o ${EXE}
