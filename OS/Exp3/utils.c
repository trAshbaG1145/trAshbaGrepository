#include "pcb.h"

void printTable(PCB p[], int n) {
    printf("\n[Final Result Table]\n");
    printf("----------------------------------------------------------------------------------\n");
    printf("Name\tArrival\tBurst\tPrio\tFinish\tTurnAround\tWaiting\n");
    printf("----------------------------------------------------------------------------------\n");
    double avgTurn = 0, avgWait = 0;
    for(int i=0; i<n; i++) {
        printf("%s\t%d\t%d\t%d\t%d\t%d\t\t%d\n", 
            p[i].name, p[i].arrivalTime, p[i].burstTime, p[i].priority, 
            p[i].finishTime, p[i].turnAroundTime, p[i].waitingTime);
        avgTurn += p[i].turnAroundTime;
        avgWait += p[i].waitingTime;
    }
    printf("----------------------------------------------------------------------------------\n");
    printf("Average Turnaround: %.2f\n", avgTurn / n);
    printf("Average Waiting: %.2f\n", avgWait / n);
}

void sortByArrival(PCB p[], int n) {
    PCB temp;
    for(int i=0; i<n-1; i++) {
        for(int j=0; j<n-1-i; j++) {
            if(p[j].arrivalTime > p[j+1].arrivalTime) {
                temp = p[j];
                p[j] = p[j+1];
                p[j+1] = temp;
            }
        }
    }
}

void inputProcesses(PCB p[], int *n) {
    printf("Enter number of processes: ");
    scanf("%d", n);
    for(int i=0; i<*n; i++) {
        printf("Process %d [Name Arrival Burst Priority]: ", i+1);
        scanf("%s %d %d %d", p[i].name, &p[i].arrivalTime, &p[i].burstTime, &p[i].priority);
        p[i].remainingTime = p[i].burstTime;
    }
}