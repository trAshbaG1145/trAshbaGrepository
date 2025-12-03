#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <sys/types.h>

int wait_mark = 1; // 全局变量，控制循环等待

void stop(int signum) {
    wait_mark = 0; // 收到信号后，修改标志位跳出循环
}

void waiting() {
    while (wait_mark != 0); // 忙等待，直到 wait_mark 变为 0
}

int main() {
    int p1, p2;

    // 注册 SIGINT (Ctrl+C) 的处理函数
    // 注意：这里放在 fork 之前，意味着子进程也会继承这个处理函数
    signal(SIGINT, stop); 

    while ((p1 = fork()) == -1); // 创建子进程 1

    if (p1 > 0) {
        // 父进程代码块
        // ① 如果 signal 放在这里
        while ((p2 = fork()) == -1); // 创建子进程 2

        if (p2 > 0) {
            // 父进程代码块
            // ② 如果 signal 放在这里
            wait_mark = 1;
            waiting(); // 父进程在这里空转，等待 Ctrl+C 触发 stop()

            // 收到 Ctrl+C 后，wait_mark 变 0，继续执行
            kill(p1, 10); // 向 p1 发送信号 10 (SIGUSR1)
            kill(p2, 12); // 向 p2 发送信号 12 (SIGUSR2)
            
            wait(0); // 等待第一个子进程结束
            wait(0); // 等待第二个子进程结束
            
            printf("Parent process is killed!\n");
            exit(0);
        } else {
            // 子进程 2 代码块
            wait_mark = 1;
            signal(12, stop); // 注册信号 12 的处理函数
            waiting(); // 等待父进程发信号
            
            lockf(1, 1, 0); // 锁定 stdout 防止输出混乱
            printf("Child process 2 is killed by parent!\n");
            lockf(1, 0, 0);
            exit(0);
        }
    } else {
        // 子进程 1 代码块
        wait_mark = 1;
        signal(10, stop); // 注册信号 10 的处理函数
        waiting(); // 等待父进程发信号
        
        lockf(1, 1, 0);
        printf("Child process 1 is killed by parent!\n");
        lockf(1, 0, 0);
        exit(0);
    }
}