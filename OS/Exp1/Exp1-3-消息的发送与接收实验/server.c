#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>

#define MSGKEY 75 

struct msgform {
    long mtype; 
    char mtext[1000]; 
} msg; 

int main() {
    int msgqid;
    
    // 创建消息队列 (0777 表示读写执行权限)
    msgqid = msgget(MSGKEY, 0777 | IPC_CREAT); 
    if (msgqid == -1) {
        perror("server create queue failed");
        exit(1);
    }
    
    printf("Server: Queue created (ID: %d). Waiting for messages...\n", msgqid);

    do {
        // 接收消息
        // 参数：ID, 消息结构体指针, 消息内容大小, 消息类型(0表示任意), 标志位
        msgrcv(msgqid, &msg, sizeof(msg.mtext), 0, 0); 
        
        printf("(server) received message type: %ld\n", msg.mtype);
        
    } while (msg.mtype != 1); // 当收到类型为 1 的消息时停止

    // 删除消息队列
    msgctl(msgqid, IPC_RMID, 0); 
    printf("Server: Queue deleted. Exiting.\n");
    
    exit(0);
}