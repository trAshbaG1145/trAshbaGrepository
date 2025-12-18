#include "pcb.h"

void PriorityScheduling(PCB p[], int n) {
    printf("\n=== Priority Scheduling Simulation Start ===\n");
    int currentTime = 0;
    int completed = 0;
    int isCompleted[MAX_PROCESS] = {0};

    while(completed < n) {
        int idx = -1;
        int maxPrio = -10000; 

        // 寻找最高优先级
        for(int i=0; i<n; i++) {
            if(p[i].arrivalTime <= currentTime && isCompleted[i] == 0) {
                if(p[i].priority > maxPrio) {
                    maxPrio = p[i].priority;
                    idx = i;
                } else if(p[i].priority == maxPrio) {
                    if(idx == -1 || p[i].arrivalTime < p[idx].arrivalTime)
                        idx = i;
                }
            }
        }

        if(idx != -1) {
            // 打印过程
            printf("Time %d: Process %s starts (Priority: %d)\n", currentTime, p[idx].name, p[idx].priority);
            
            currentTime += p[idx].burstTime;
            
            p[idx].finishTime = currentTime;
            p[idx].turnAroundTime = p[idx].finishTime - p[idx].arrivalTime;
            p[idx].waitingTime = p[idx].turnAroundTime - p[idx].burstTime;
            isCompleted[idx] = 1;
            completed++;
            
            printf("        -> Finished at %d\n", currentTime);
        } else {
            printf("Time %d: CPU Idle\n", currentTime);
            currentTime++; 
        }
    }
    printf("=== Simulation Finished ===\n");
    printTable(p, n);
}