// Linux下基于多进程和信号量的生产者-消费者问题实现


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <semaphore.h>
#include <fcntl.h>

#define BUFFER_SIZE 5   // 缓冲区大小
#define PRODUCE_COUNT 10 // 演示生成的总产品数

// 定义共享内存结构体
typedef struct {
    int buffer[BUFFER_SIZE]; // 循环缓冲区
    int in;  // 写入指针
    int out; // 读取指针
    sem_t mutex; // 互斥信号量
    sem_t empty; // 空闲槽位信号量
    sem_t full;  // 已用槽位信号量
} SharedData;

// 生产者进程逻辑
void producer_process(SharedData *shared) {
    for (int i = 0; i < PRODUCE_COUNT; i++) {
        int item = rand() % 100; // 生成一个随机数作为产品
        
        // 模拟生产耗时
        sleep(rand() % 2); 

        // P操作：申请空槽位
        sem_wait(&shared->empty);
        // P操作：申请互斥锁
        sem_wait(&shared->mutex);

        // --- 临界区开始 ---
        shared->buffer[shared->in] = item;
        printf("[生产者 PID:%d] 生产数据: %d, 放入位置: %d\n", getpid(), item, shared->in);
        shared->in = (shared->in + 1) % BUFFER_SIZE;
        // --- 临界区结束 ---

        // V操作：释放互斥锁
        sem_post(&shared->mutex);
        // V操作：增加满槽位
        sem_post(&shared->full);
    }
    printf("生产者进程结束。\n");
    exit(0);
}

// 消费者进程逻辑
void consumer_process(SharedData *shared) {
    for (int i = 0; i < PRODUCE_COUNT; i++) {
        
        // 模拟消费前的准备
        sleep(rand() % 3); 

        // P操作：申请满槽位（检查是否有数据）
        sem_wait(&shared->full);
        // P操作：申请互斥锁
        sem_wait(&shared->mutex);

        // --- 临界区开始 ---
        int item = shared->buffer[shared->out];
        printf("    [消费者 PID:%d] 消费数据: %d, 取出位置: %d\n", getpid(), item, shared->out);
        shared->out = (shared->out + 1) % BUFFER_SIZE;
        // --- 临界区结束 ---

        // V操作：释放互斥锁
        sem_post(&shared->mutex);
        // V操作：增加空槽位
        sem_post(&shared->empty);
    }
    printf("消费者进程结束。\n");
    exit(0);
}

int main() {
    // 1. 创建共享内存
    // PROT_READ | PROT_WRITE: 可读写
    // MAP_SHARED: 进程间共享
    // MAP_ANONYMOUS: 匿名映射（不依赖文件）
    SharedData *shared = mmap(NULL, sizeof(SharedData), 
                              PROT_READ | PROT_WRITE, 
                              MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    if (shared == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }

    // 2. 初始化共享变量
    shared->in = 0;
    shared->out = 0;

    // 3. 初始化信号量
    // 参数2为1表示信号量在进程间共享 (pshared=1)
    sem_init(&shared->mutex, 1, 1);           // 互斥锁初值为1
    sem_init(&shared->empty, 1, BUFFER_SIZE); // 空槽位初值为5
    sem_init(&shared->full, 1, 0);            // 满槽位初值为0

    printf("------ 模拟开始 (缓冲区大小: %d) ------\n", BUFFER_SIZE);

    pid_t pid_p, pid_c;

    // 4. 创建生产者进程
    pid_p = fork();
    if (pid_p == 0) {
        srand(time(NULL) ^ getpid()); // 重置随机种子
        producer_process(shared);
    } else if (pid_p < 0) {
        perror("fork producer failed");
        exit(1);
    }

    // 5. 创建消费者进程
    pid_c = fork();
    if (pid_c == 0) {
        srand(time(NULL) ^ getpid());
        consumer_process(shared);
    } else if (pid_c < 0) {
        perror("fork consumer failed");
        exit(1);
    }

    // 6. 父进程等待子进程结束
    waitpid(pid_p, NULL, 0);
    waitpid(pid_c, NULL, 0);

    // 7. 销毁信号量与释放内存
    sem_destroy(&shared->mutex);
    sem_destroy(&shared->empty);
    sem_destroy(&shared->full);
    munmap(shared, sizeof(SharedData));

    printf("------ 模拟结束 ------\n");
    return 0;
}