#include "pcb.h"

void RoundRobin(PCB p[], int n, int quantum) {
    printf("\n=== Round Robin (q=%d) Simulation Start ===\n", quantum);
    sortByArrival(p, n);
    
    int queue[1000]; 
    int front = 0, rear = 0;
    int visited[MAX_PROCESS] = {0};
    int currentTime = 0;
    int completed = 0;

    // 初始化：第一个进程入队
    if(n > 0) {
        queue[rear++] = 0;
        visited[0] = 1;
        currentTime = p[0].arrivalTime;
    }

    while(completed < n) {
        // 队列为空但仍有未完成进程（处理CPU空闲期）
        if(front == rear) {
             for(int i=0; i<n; i++) {
                if(p[i].remainingTime > 0 && !visited[i]) {
                    printf("Time %d -> %d: CPU Idle\n", currentTime, p[i].arrivalTime);
                    currentTime = p[i].arrivalTime;
                    queue[rear++] = i;
                    visited[i] = 1;
                    break;
                }
             }
        }

        int idx = queue[front++]; // 出队
        
        // 计算本次运行时间
        int exec = (p[idx].remainingTime > quantum) ? quantum : p[idx].remainingTime;
        
        // 打印过程！
        printf("Time %d: Run %s (Rem: %d) -> ", currentTime, p[idx].name, p[idx].remainingTime);
        
        p[idx].remainingTime -= exec;
        currentTime += exec;
        
        printf("Time %d (Rem: %d)\n", currentTime, p[idx].remainingTime);

        // 新到达进程入队
        for(int i=0; i<n; i++) {
            if(p[i].remainingTime > 0 && p[i].arrivalTime <= currentTime && !visited[i]) {
                queue[rear++] = i;
                visited[i] = 1;
            }
        }

        // 进程未完成，重新入队
        if(p[idx].remainingTime > 0) {
            queue[rear++] = idx;
        } else {
            // 进程完成
            printf("       *** %s Finished ***\n", p[idx].name);
            p[idx].finishTime = currentTime;
            p[idx].turnAroundTime = p[idx].finishTime - p[idx].arrivalTime;
            p[idx].waitingTime = p[idx].turnAroundTime - p[idx].burstTime;
            completed++;
        }
    }
    printf("=== Simulation Finished ===\n");
    printTable(p, n);
}