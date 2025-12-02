#include <stdio.h>
int min_ans = 99999999;
void dfs(int t_num, int m_num, int task[], int machine[], int idx)
{
    if (idx == t_num)
    {
        int max = machine[0];
        for (int i = 1; i < m_num; i++)
            if (machine[i] > max)
                max = machine[i];
        if (max < min_ans)
            min_ans = max;
        return;
    }
    for (int i = 0; i < m_num; i++)
    {
        machine[i] += task[idx];
        dfs(t_num, m_num, task, machine, idx + 1);
        machine[i] -= task[idx];
    }
}
int main()
{
    int t_num, m_num;
    scanf("%d", &t_num);
    scanf("%d", &m_num);
    int task[t_num], machine[m_num];
    for (int i = 0; i < t_num; i++)
        scanf("%d", &task[i]);
    for (int i = 0; i < m_num; i++)
        machine[i] = 0;
    min_ans = 999;
    dfs(t_num, m_num, task, machine, 0);
    printf("%d\n", min_ans);
    return 0;
}