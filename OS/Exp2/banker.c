#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_PROCESS 50  // 最大进程数 n
#define MAX_RESOURCE 50 // 最大资源种类 m

// --- 4. 主要数据结构 ---
int m, n;                       // m:资源种类数, n:进程数
int Available[MAX_RESOURCE];    // 可利用资源向量
int Max[MAX_PROCESS][MAX_RESOURCE];        // 最大需求矩阵
int Allocation[MAX_PROCESS][MAX_RESOURCE]; // 分配矩阵
int Need[MAX_PROCESS][MAX_RESOURCE];       // 还需资源矩阵
int Request[MAX_RESOURCE];      // 申请资源数量
int SafeSequence[MAX_PROCESS];  // 记录安全序列

// --- 函数声明 ---
void initialize();
void show();
bool safe();
void bank();

// --- 5. 主程序 main() ---
int main() {
    initialize(); // (1) 初始化
    show();       // (4) 显示初始状态

    if (!safe()) { // (2) 初始安全性检查
        printf("\n【警告】初始状态是不安全的！死锁可能已经发生。\n");
    } else {
        printf("\n【提示】初始状态是安全的。\n");
    }

    // 循环让用户输入请求
    while (true) {
        printf("\n--------------------------------------------------\n");
        printf("是否要申请资源？(y/n): ");
        char ch;
        scanf(" %c", &ch); // 注意%c前的空格，过滤回车符
        if (ch == 'n' || ch == 'N') break;
        
        bank(); // (3) 调用银行家算法
    }

    return 0;
}

// --- (1) 初始化 initialize() ---
void initialize() {
    printf("请输入进程的数量(n): ");
    scanf("%d", &n);
    printf("请输入资源的种类数量(m): ");
    scanf("%d", &m);

    printf("\n请输入 Available 向量 (例如: 3 3 2):\n");
    for (int i = 0; i < m; i++) {
        scanf("%d", &Available[i]);
    }

    printf("\n请输入 Max 矩阵 (%d 行 %d 列):\n", n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            scanf("%d", &Max[i][j]);
        }
    }

    printf("\n请输入 Allocation 矩阵 (%d 行 %d 列):\n", n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            scanf("%d", &Allocation[i][j]);
            // 计算 Need 矩阵: Need[i][j] = Max[i][j] - Allocation[i][j]
            Need[i][j] = Max[i][j] - Allocation[i][j];
            if (Need[i][j] < 0) {
                printf("错误：进程 P%d 的 Allocation 大于 Max！\n", i);
                exit(1);
            }
        }
    }
}

// --- (4) 显示当前状态 show() ---
void show() {
    printf("\n当前资源分配情况：\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("进程\tMax\t\tAllocation\tNeed\t\tAvailable\n");
    printf("\t");
    // 打印资源表头 (A B C ...)
    for(int k=0; k<3; k++) { // 为了排版整齐，每个矩阵列打印表头
        for (int j = 0; j < m; j++) printf("R%d ", j);
        printf("\t\t");
    }
    printf("\n");

    for (int i = 0; i < n; i++) {
        printf("P%d\t", i);
        
        for (int j = 0; j < m; j++) printf("%d  ", Max[i][j]);
        printf("\t\t");
        
        for (int j = 0; j < m; j++) printf("%d  ", Allocation[i][j]);
        printf("\t\t");
        
        for (int j = 0; j < m; j++) printf("%d  ", Need[i][j]);
        
        // Available 只在第一行打印
        if (i == 0) {
            printf("\t\t");
            for (int j = 0; j < m; j++) printf("%d  ", Available[j]);
        }
        printf("\n");
    }
    printf("--------------------------------------------------------------------------------\n");
}

// --- 2. 安全性算法 safe() ---
bool safe() {
    int Work[MAX_RESOURCE];
    bool Finish[MAX_PROCESS];
    int count = 0; // 记录完成的进程数

    // (1) 初始化
    for (int j = 0; j < m; j++) Work[j] = Available[j];
    for (int i = 0; i < n; i++) Finish[i] = false;

    // (2) 寻找满足条件的进程
    while (count < n) {
        bool found = false;
        for (int i = 0; i < n; i++) {
            if (Finish[i] == false) {
                // 检查 Need[i] <= Work
                bool enough = true;
                for (int j = 0; j < m; j++) {
                    if (Need[i][j] > Work[j]) {
                        enough = false;
                        break;
                    }
                }

                // (3) 若找到，回收资源
                if (enough) {
                    for (int j = 0; j < m; j++) {
                        Work[j] += Allocation[i][j];
                    }
                    Finish[i] = true;
                    SafeSequence[count++] = i; // 记录安全序列
                    found = true;
                }
            }
        }
        // 如果一轮遍历下来没有找到可执行的进程，且还有进程未完成 -> 死锁/不安全
        if (!found) {
            return false;
        }
    }

    // (4) 所有进程 Finish = true
    printf("系统处于安全状态。安全序列为: ");
    for (int i = 0; i < n; i++) {
        printf("P%d", SafeSequence[i]);
        if (i < n - 1) printf(" -> ");
    }
    printf("\n");
    return true;
}

// --- 1. 银行家算法 bank() ---
void bank() {
    int pid;
    printf("请输入申请资源的进程ID (0-%d): ", n - 1);
    scanf("%d", &pid);
    if (pid < 0 || pid >= n) {
        printf("输入的进程ID不存在！\n");
        return;
    }

    printf("请输入进程 P%d 的请求向量 (%d 个数值): ", pid, m);
    for (int j = 0; j < m; j++) {
        scanf("%d", &Request[j]);
    }

    // 步骤 (1): 检查 Request <= Need
    for (int j = 0; j < m; j++) {
        if (Request[j] > Need[pid][j]) {
            printf("【错误】请求资源大于该进程还需要的资源(Need)！\n");
            return;
        }
    }

    // 步骤 (2): 检查 Request <= Available
    for (int j = 0; j < m; j++) {
        if (Request[j] > Available[j]) {
            printf("【等待】请求资源大于系统当前可用资源(Available)，进程需等待。\n");
            return;
        }
    }

    // 步骤 (3): 试探性分配
    for (int j = 0; j < m; j++) {
        Available[j] -= Request[j];
        Allocation[pid][j] += Request[j];
        Need[pid][j] -= Request[j];
    }

    // 步骤 (4): 安全性检查
    if (safe()) {
        printf("【成功】资源分配成功！\n");
        show(); // 显示分配后的状态
    } else {
        printf("【失败】尝试分配后系统将处于不安全状态！本次分配作废，恢复原状。\n");
        // 恢复原来的状态 (Rollback)
        for (int j = 0; j < m; j++) {
            Available[j] += Request[j];
            Allocation[pid][j] -= Request[j];
            Need[pid][j] += Request[j];
        }
    }
}