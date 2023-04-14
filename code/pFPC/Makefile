CC = gcc
OPTFLAGS = -O3 
CFLAGS = -pthread -std=c99 $(OPTFLAGS) #-DUSE_MPI_COLLECTIVES #-DUSE_32_BIT_GRAPH  #-DDEBUG_PRINTF

OBJ = pfpc.o 

TARGET = pfpc

all: $(TARGET)

$(TARGET):  pFPC.c
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -rf *~ $(OBJFILES) $(TARGET) 
