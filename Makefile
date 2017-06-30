CC = gcc
CFLAGS = -O3 --fast-math -fopenmp -Wno-unused-result
OBJ = tdc-omp-03-2013.o

tdc : $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o tdc -lm
clean: 
	rm -f tdc tdc-omp-03-2013.o
