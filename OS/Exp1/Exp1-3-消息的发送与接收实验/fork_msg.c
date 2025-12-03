#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <sys/wait.h>

#define MSGKEY 88 // 使用一个新的 KEY
#define MSG_SIZE 256

struct msgform {
    long mtype;
    char mtext[MSG_SIZE];
};

int main() {
    int msgqid;
    int pid;
    struct msgform msg_send, msg_recv;

    // 1. 父进程创建消息队列
    msgqid = msgget(MSGKEY, 0777 | IPC_CREAT);
    if (msgqid == -1) {
        perror("msgget failed");
        exit(1);
    }

    // 2. 创建子进程
    pid = fork();

    if (pid < 0) {
        perror("fork failed");
        exit(1);
    } else if (pid == 0) {
        // --- 子进程 (发送方) ---
        msg_send.mtype = 100; // 消息类型
        strcpy(msg_send.mtext, "Hello, Daddy! This is a message from Child.");
        
        printf("[Child] Sending message...\n");
        if (msgsnd(msgqid, &msg_send, MSG_SIZE, 0) == -1) {
            perror("msgsnd failed");
            exit(1);
        }
        printf("[Child] Message sent. Exiting.\n");
        exit(0);
    } else {
        // --- 父进程 (接收方) ---
        printf("[Parent] Waiting for message...\n");
        
        // 接收消息
        if (msgrcv(msgqid, &msg_recv, MSG_SIZE, 100, 0) == -1) {
            perror("msgrcv failed");
            exit(1);
        }

        printf("[Parent] Received: %s\n", msg_recv.mtext);

        // 等待子进程结束
        wait(NULL);

        // 删除消息队列
        msgctl(msgqid, IPC_RMID, 0);
        printf("[Parent] Queue deleted.\n");
        exit(0);
    }
}