#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <map>
#include <fstream>

using namespace std;

const int TOTAL_INSTRUCTIONS = 320;
const int INSTRUCTIONS_PER_PAGE = 10;
const int TOTAL_PAGES = 32;

// 全局指令序列
vector<int> instruction_stream;
vector<int> page_stream;

// 1. 生成指令序列 (遵循实验要求的 A-E 步骤)
void generate_instructions() {
    instruction_stream.clear();
    int m = 0;
    srand((unsigned)time(NULL));

    // 需要生成 320 条，按照逻辑循环生成
    while (instruction_stream.size() < TOTAL_INSTRUCTIONS) {
        // A: 在 [0, 319] 随机选取 m
        m = rand() % 320;
        instruction_stream.push_back(m);
        if (instruction_stream.size() >= TOTAL_INSTRUCTIONS) break;

        // B: 顺序执行 m+1
        if (m + 1 < 320) instruction_stream.push_back(m + 1);
        if (instruction_stream.size() >= TOTAL_INSTRUCTIONS) break;

        // C: 在前地址 [0, m+1] 中随机选取 m'
        int m_prime = rand() % (m + 2); // m+1 is inclusive index, so mod m+2
        if (m_prime > 319) m_prime = 319; 
        instruction_stream.push_back(m_prime);
        if (instruction_stream.size() >= TOTAL_INSTRUCTIONS) break;

        // D: 顺序执行 m'+1
        if (m_prime + 1 < 320) instruction_stream.push_back(m_prime + 1);
        if (instruction_stream.size() >= TOTAL_INSTRUCTIONS) break;

        // E: 在后地址 [m'+2, 319] 中随机选取
        if (m_prime + 2 < 320) {
            int range = 319 - (m_prime + 2) + 1;
            int m_double_prime = (m_prime + 2) + rand() % range;
            instruction_stream.push_back(m_double_prime);
        }
    }
    
    // 转换为页地址流
    page_stream.clear();
    for (int instr : instruction_stream) {
        page_stream.push_back(instr / INSTRUCTIONS_PER_PAGE);
    }
}

// 检查页面是否在内存中
bool is_in_memory(const vector<int>& memory, int page) {
    for (int p : memory) {
        if (p == page) return true;
    }
    return false;
}

// 2. OPT 算法
double simulate_OPT(int capacity) {
    vector<int> memory;
    int page_faults = 0;

    for (int i = 0; i < TOTAL_INSTRUCTIONS; i++) {
        int page = page_stream[i];
        if (!is_in_memory(memory, page)) {
            page_faults++;
            if (memory.size() < capacity) {
                memory.push_back(page);
            } else {
                // 寻找最远未来使用的页面进行替换
                int furthest_idx = -1;
                int victim_idx = -1;
                
                for (int m_idx = 0; m_idx < memory.size(); m_idx++) {
                    int next_use = 99999;
                    for (int j = i + 1; j < TOTAL_INSTRUCTIONS; j++) {
                        if (page_stream[j] == memory[m_idx]) {
                            next_use = j;
                            break;
                        }
                    }
                    if (next_use > furthest_idx) {
                        furthest_idx = next_use;
                        victim_idx = m_idx;
                    }
                }
                memory[victim_idx] = page;
            }
        }
    }
    return 1.0 - (double)page_faults / TOTAL_INSTRUCTIONS;
}

// 3. FIFO 算法
double simulate_FIFO(int capacity) {
    vector<int> memory;
    int page_faults = 0;
    
    for (int page : page_stream) {
        if (!is_in_memory(memory, page)) {
            page_faults++;
            if (memory.size() < capacity) {
                memory.push_back(page);
            } else {
                memory.erase(memory.begin()); // 移除头部（最早进入的）
                memory.push_back(page);
            }
        }
    }
    return 1.0 - (double)page_faults / TOTAL_INSTRUCTIONS;
}

// 4. LRU 算法
double simulate_LRU(int capacity) {
    vector<int> memory; // 这里用 vector 模拟栈，尾部是最近使用的
    int page_faults = 0;

    for (int page : page_stream) {
        auto it = find(memory.begin(), memory.end(), page);
        if (it != memory.end()) {
            // 命中：将其移到末尾（最新）
            memory.erase(it);
            memory.push_back(page);
        } else {
            page_faults++;
            if (memory.size() < capacity) {
                memory.push_back(page);
            } else {
                memory.erase(memory.begin()); // 移除头部（最久未使用的）
                memory.push_back(page);
            }
        }
    }
    return 1.0 - (double)page_faults / TOTAL_INSTRUCTIONS;
}

// 5. LFU 算法
double simulate_LFU(int capacity) {
    vector<int> memory;
    map<int, int> freq_counter; // 记录频率
    int page_faults = 0;

    for (int page : page_stream) {
        freq_counter[page]++; // 访问计数 +1
        
        if (!is_in_memory(memory, page)) {
            page_faults++;
            if (memory.size() < capacity) {
                memory.push_back(page);
            } else {
                // 找频率最小的，若相同则找最早进来的（这里简化，直接按顺序找）
                int min_freq = 99999;
                int victim_idx = -1;
                for (int i = 0; i < memory.size(); i++) {
                    if (freq_counter[memory[i]] < min_freq) {
                        min_freq = freq_counter[memory[i]];
                        victim_idx = i;
                    }
                }
                // 淘汰并重置计数(或不重置，视具体LFU变种而定，这里不重置历史计数)
                // freq_counter.erase(memory[victim_idx]); 
                memory[victim_idx] = page;
            }
        }
    }
    return 1.0 - (double)page_faults / TOTAL_INSTRUCTIONS;
}

// 6. NUR 算法 (简单 Clock 实现)
double simulate_NUR(int capacity) {
    struct PageFrame { int id; int visited; };
    vector<PageFrame> memory;
    int page_faults = 0;
    int pointer = 0; // 模拟时钟指针

    for (int page : page_stream) {
        bool hit = false;
        for (auto &frame : memory) {
            if (frame.id == page) {
                frame.visited = 1; // 访问位设为1
                hit = true;
                break;
            }
        }

        if (!hit) {
            page_faults++;
            if (memory.size() < capacity) {
                memory.push_back({page, 1});
            } else {
                // 寻找 visited = 0
                while (true) {
                    if (memory[pointer].visited == 0) {
                        memory[pointer] = {page, 1}; // 替换
                        pointer = (pointer + 1) % capacity;
                        break;
                    } else {
                        memory[pointer].visited = 0; // 给一次机会，置0
                        pointer = (pointer + 1) % capacity;
                    }
                }
            }
        }
    }
    return 1.0 - (double)page_faults / TOTAL_INSTRUCTIONS;
}

// 6. 绘图函数，生成 Python 脚本并调用
void plot_results(const vector<int>& x, const vector<double>& opt, 
                  const vector<double>& fifo, const vector<double>& lru,
                  const vector<double>& lfu, const vector<double>& nur) {
    ofstream py("plot_result.py");
    if (!py.is_open()) return;

    // 写入 Python 代码
    py << "import matplotlib.pyplot as plt\n\n";
    
    // 将 C++ vector 转换为 Python list
    auto write_vec = [&](string name, const vector<double>& v) {
        py << name << " = [";
        for(size_t i=0; i<v.size(); ++i) py << v[i] << (i==v.size()-1?"":",");
        py << "]\n";
    };
    
    py << "x = ["; for(size_t i=0; i<x.size(); ++i) py << x[i] << (i==x.size()-1?"":","); py << "]\n";
    write_vec("opt", opt);
    write_vec("fifo", fifo);
    write_vec("lru", lru);
    write_vec("lfu", lfu);
    write_vec("nur", nur);

    // 绘图命令
    py << "plt.figure(figsize=(10, 6))\n"
       << "plt.plot(x, opt, 'o-', label='OPT')\n"
       << "plt.plot(x, fifo, 's-', label='FIFO')\n"
       << "plt.plot(x, lru, '^-', label='LRU')\n"
       << "plt.plot(x, lfu, 'x-', label='LFU')\n"
       << "plt.plot(x, nur, 'd-', label='NUR')\n"
       << "plt.title('Page Replacement Algorithms Hit Rate')\n"
       << "plt.xlabel('Memory Size')\n"
       << "plt.ylabel('Hit Rate')\n"
       << "plt.legend()\n"
       << "plt.grid(True)\n"
       << "plt.xticks(x)\n"
       << "plt.savefig('result_chart.png')\n"; // 保存为图片
    py.close();

    // 调用系统 Python 运行
    system("python3 plot_result.py"); 
}

// 主函数，运行模拟并输出结果
int main() {
    generate_instructions();
    
    cout << "Total Instructions: " << TOTAL_INSTRUCTIONS << endl;
    cout << "--------------------------------------------------------" << endl;
    cout << "MemSize\tOPT\tFIFO\tLRU\tLFU\tNUR" << endl;
    cout << "--------------------------------------------------------" << endl;

    // 新增：定义向量用于存储画图数据
    vector<int> x_axis;
    vector<double> y_opt, y_fifo, y_lru, y_lfu, y_nur;

    // 模拟内存从 4 页到 32 页的变化
    for (int cap = 4; cap <= 32; cap += 2) { // 步长为2，节省输出篇幅
        double opt = simulate_OPT(cap);
        double fifo = simulate_FIFO(cap);
        double lru = simulate_LRU(cap);
        double lfu = simulate_LFU(cap);
        double nur = simulate_NUR(cap);

        cout << cap << "\t" 
             << fixed << setprecision(3) << opt << "\t"
             << fifo << "\t"
             << lru << "\t"
             << lfu << "\t"
             << nur << endl;

        // 存储数据用于绘图
        x_axis.push_back(cap);
        y_opt.push_back(opt);
        y_fifo.push_back(fifo);
        y_lru.push_back(lru);
        y_lfu.push_back(lfu);
        y_nur.push_back(nur);
    }

    // 新增：循环结束后调用画图函数
    cout << "[Info] Generating chart..." << endl;
    plot_results(x_axis, y_opt, y_fifo, y_lru, y_lfu, y_nur);
    cout << "[Done] Chart saved as 'result_chart.png'" << endl;

    return 0;
}
