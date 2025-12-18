schedule: main.o utils.o fcfs.o priority.o rr.o
	if not exist build mkdir build
	gcc -o build/schedule main.o utils.o fcfs.o priority.o rr.o

main.o: main.c pcb.h
	gcc -c main.c
utils.o: utils.c pcb.h
	gcc -c utils.c
fcfs.o: fcfs.c pcb.h
	gcc -c fcfs.c
priority.o: priority.c pcb.h
	gcc -c priority.c
rr.o: rr.c pcb.h
	gcc -c rr.c

clean:
	rm *.o build/schedule