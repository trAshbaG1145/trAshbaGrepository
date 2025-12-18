#include "pcb.h"

void FCFS(PCB p[], int n) {
    printf("\n=== FCFS Simulation Start ===\n");
    sortByArrival(p, n);
    int currentTime = 0;
    
    for(int i=0; i<n; i++) {
        // 如果当前时间还未到进程到达时间，CPU空闲
        if(currentTime < p[i].arrivalTime) {
            printf("Time %d -> %d: CPU Idle\n", currentTime, p[i].arrivalTime);
            currentTime = p[i].arrivalTime;
        }
        
        // 打印正在运行
        printf("Time %d -> %d: Process %s is running (Arrival: %d)\n", 
               currentTime, currentTime + p[i].burstTime, p[i].name, p[i].arrivalTime);

        currentTime += p[i].burstTime;
        p[i].finishTime = currentTime;
        p[i].turnAroundTime = p[i].finishTime - p[i].arrivalTime;
        p[i].waitingTime = p[i].turnAroundTime - p[i].burstTime;
    }
    printf("=== Simulation Finished ===\n");
    printTable(p, n);
}