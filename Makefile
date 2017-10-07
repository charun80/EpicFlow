CCPP=g++ -std=c++11
CC=gcc

LAPACKCFLAGS=#-DTHREADEDLAPACK
LAPACKLDFLAGS=-L/usr/lib64/atlas/ -llapack

CFLAGS=-fPIC -Wall -Wstrict-aliasing=3 -fno-strict-aliasing -O3 -march=native -fsigned-char $(LAPACKCFLAGS) -fvisibility=hidden -fstack-protector-strong -Wformat -Werror=format-security # -fopenmp -DUSE_OPENMP
LDFLAGS=-Wall -O3 -fno-strict-aliasing -ljpeg -lpng  $(LAPACKLDFLAGS) # -fopenmp 

TARGET_LIB = libctypesEpicFlow.so # target lib

#SOURCES_CPP := $(shell find . -name '*.cpp')
#SOURCES_C := $(shell find . -name '*.c')

COMMONOBJ=epic_aux.o epic.o image.o io.o solver.o variational_aux.o variational.o
EXECOBJ=io.o epicflow.o tictoc.o
PYLIBOBJ=ctypesEpicFlow.o


OBJ := $(SOURCES_CPP:%.cpp=%.o) $(SOURCES_C:%.c=%.o) 
HEADERS := $(shell find . -name '*.h')

all: epicflow $(TARGET_LIB)

.cpp.o:  %.cpp %.h
	$(CCPP) -o $@ $(CFLAGS) -c $+

.c.o:  %.c %.h
	$(CC) -o $@ $(CFLAGS) -c $+

epicflow: $(HEADERS) $(COMMONOBJ) $(EXECOBJ)
	$(CCPP) -o $@ $^ $(LDFLAGS)

epicflow-static: $(HEADERS) $(OBJ)
	$(CCPP) -o $@ $^ -static  /usr/lib64/libjpeg.a /usr/lib64/libpng.a /usr/lib64/libz.a /usr/lib64/libm.a /usr/lib64/liblapack.a /usr/lib/gcc/x86_64-redhat-linux/4.7.2/libgfortran.a /usr/lib64/libblas.a


$(TARGET_LIB): $(HEADERS) $(COMMONOBJ) $(PYLIBOBJ)
	$(CCPP) $(CFLAGS) -fvisibility=hidden -shared -Wl,-soname,$(TARGET_LIB) -o $@ $^  ${LDFLAGS}


clean:
	rm -f $(COMMONOBJ) $(EXECOBJ) $(PYLIBOBJ) epicflow epicflow-static

