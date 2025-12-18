#include "pcb.h"

int main() {
    PCB processes[MAX_PROCESS];
    PCB temp[MAX_PROCESS];
    int n, choice, q;

    inputProcesses(processes, &n);

    while(1) {
        printf("\n=== OS Process Scheduler ===\n");
        printf("1. FCFS\n");
        printf("2. Priority Scheduling (Non-preemptive)\n");
        printf("3. Round Robin\n");
        printf("0. Exit\n");
        printf("Select Algorithm: ");
        scanf("%d", &choice);

        if(choice == 0) break;

        // 复制数据，保证每次算法运行时数据是干净的
        for(int i=0; i<n; i++) temp[i] = processes[i];

        switch(choice) {
            case 1: FCFS(temp, n); break;
            case 2: PriorityScheduling(temp, n); break;
            case 3: 
                printf("Enter Time Quantum: "); 
                scanf("%d", &q);
                RoundRobin(temp, n, q); 
                break;
            default: printf("Invalid input.\n");
        }
    }
    return 0;
}