#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <unistd.h>

#define MSGKEY 75 

struct msgform {
    long mtype; 
    char mtext[1000]; 
} msg; 

int main() {
    int msgqid, i;
    
    // 获取消息队列 ID (不需要 IPC_CREAT，因为 Server 已经创建了)
    msgqid = msgget(MSGKEY, 0777);
    if (msgqid == -1) {
        perror("client get queue failed (Did you run server first?)");
        exit(1);
    }

    // 倒序发送 10 到 1
    for (i = 10; i >= 1; i--) {
        msg.mtype = i; 
        // 这里的文本内容不是重点，重点是 mtype
        printf("(client) sent type: %d\n", i);
        
        // 发送消息
        msgsnd(msgqid, &msg, sizeof(msg.mtext), 0); 
        
        sleep(1); // 稍微延时一下，方便观察
    } 
    
    exit(0);
}