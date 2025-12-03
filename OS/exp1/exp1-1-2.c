#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>

int pid1, pid2;

// 子进程收到信号后的处理函数
void Int1(int signum) {
    printf("Child process 1 is killed by parent!\n");
    exit(0);
}

void Int2(int signum) {
    printf("Child process 2 is killed by parent!\n");
    exit(0);
}

// 父进程收到 SIGINT 后的处理函数
void IntDelete(int signum) {
    kill(pid1, 10); // 发送 SIGUSR1 给子进程1
    kill(pid2, 12); // 发送 SIGUSR2 给子进程2
}

int main() {
    // 1. 先让所有进程默认忽略 Ctrl+C 和 Ctrl+\
    // 这样稍后 fork 出来的子进程也会继承“忽略”属性
    signal(SIGINT, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);

    while ((pid1 = fork()) == -1);

    if (pid1 == 0) {
        // --- 子进程 1 ---
        // 恢复对自定义信号的处理
        signal(SIGUSR1, Int1); 
        // 依然保持忽略 SIGQUIT (其实继承下来就是忽略，这句可省略，但为了明确)
        signal(SIGQUIT, SIG_IGN); 
        
        pause(); // 挂起，等待信号
        exit(0);
    } else {
        while ((pid2 = fork()) == -1);

        if (pid2 == 0) {
            // --- 子进程 2 ---
            signal(SIGUSR2, Int2);
            signal(SIGQUIT, SIG_IGN);
            
            pause(); // 挂起
            exit(0);
        } else {
            // --- 父进程 ---
            // 父进程重新开启对 SIGINT 的捕获，以便响应用户的 Ctrl+C
            signal(SIGINT, IntDelete); 
            
            // 父进程在这里阻塞等待。
            // 此时按 Ctrl+C，触发 IntDelete -> kill children
            // 此时按 Ctrl+\ (SIGQUIT)，因为上面设了 IGN，所以没反应
            
            wait(NULL); // 等待子进程1
            wait(NULL); // 等待子进程2
            
            printf("Parent process is killed\n");
            exit(0);
        }
    }
}