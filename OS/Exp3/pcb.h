#ifndef PCB_H
#define PCB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PROCESS 20

typedef struct {
    char name[10];
    int arrivalTime;
    int burstTime;
    int priority;       
    int remainingTime;  
    int finishTime;
    int waitingTime;
    int turnAroundTime;
    int startTime;      // 新增：记录首次开始时间（可选）
} PCB;

void inputProcesses(PCB p[], int *n);
void printTable(PCB p[], int n);
void sortByArrival(PCB p[], int n);
void resetProcesses(PCB p[], int n);

void FCFS(PCB p[], int n);
void PriorityScheduling(PCB p[], int n);
void RoundRobin(PCB p[], int n, int quantum);

#endif