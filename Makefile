CC = gcc
CFLAGS = -O3 --fast-math -fopenmp -Wno-unused-result -mavx
OBJ = tdc_omp_sse3.o
OBJ2 = tdc_omp_avx.o

tdc : $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o tdc -lm
tdc-avx : $(OBJ2)
	$(CC) $(CFLAGS) $(OBJ2) -o tdc-avx -lm 

clean: 
	rm -f tdc tdc-avx *.o
