CCPP=g++ -std=c++11
CC=gcc

LAPACKCFLAGS=#-DTHREADEDLAPACK
LAPACKLDFLAGS=-L/usr/lib64/atlas/ -llapack

CFLAGS= -fPIC -Wall -O3 -mavx -fsigned-char $(LAPACKCFLAGS) # -ggdb3 -fopenmp -DUSE_OPENMP
#CFLAGS= -fPIC -Wall -O3 -march=native -fsigned-char $(LAPACKCFLAGS) # -ggdb3 -fopenmp -DUSE_OPENMP
LDFLAGS=-Wall -O3 -ljpeg -lpng $(LAPACKLDFLAGS)  # ggdb3  -fopenmp 



SOURCES_CPP := $(shell find . -name '*.cpp')
SOURCES_C := $(shell find . -name '*.c')
OBJ := $(SOURCES_CPP:%.cpp=%.o) $(SOURCES_C:%.c=%.o) 
HEADERS := $(shell find . -name '*.h')

all: epicflow

.cpp.o:  %.cpp %.h
	$(CCPP) -o $@ $(CFLAGS) -c $+

.c.o:  %.c %.h
	$(CC) -o $@ $(CFLAGS) -c $+

epicflow: $(HEADERS) $(OBJ)
	$(CCPP) -o $@ $^ $(LDFLAGS)

epicflow-static: $(HEADERS) $(OBJ)
	$(CCPP) -o $@ $^ -static  /usr/lib64/libjpeg.a /usr/lib64/libpng.a /usr/lib64/libz.a /usr/lib64/libm.a /usr/lib64/liblapack.a /usr/lib/gcc/x86_64-redhat-linux/4.7.2/libgfortran.a /usr/lib64/libblas.a


clean:
	rm -f $(OBJ) epicflow

