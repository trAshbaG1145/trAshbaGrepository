#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
typedef struct node
{
    int value;                // 当前总价值
    int weight;               // 当前总重量
    bool check;               // 是否超重（超重=true，停止建树）
    struct node *in_left;     // 左子树：放入当前物品
    struct node *notin_right; // 右子树：不放入当前物品
} node;
node *createNode(int weight, int value, bool check)
{
    node *newNode = (node *)malloc(sizeof(node));
    newNode->weight = weight;
    newNode->value = value;
    newNode->check = check;
    newNode->in_left = NULL;
    newNode->notin_right = NULL;
    return newNode;
}
node *buildTree(int i, int current_weight, int current_value, int max_weight, int item[4][3])
{
    if (current_weight > max_weight)
    {
        return createNode(current_weight, current_value, true);
    }
    if (i >= 4)
    {
        return createNode(current_weight, current_value, false);
    }
    node *root = createNode(current_weight, current_value, false);
    root->in_left = buildTree(i + 1, current_weight + item[i][0], current_value + item[i][1], max_weight, item);
    root->notin_right = buildTree(i + 1, current_weight, current_value, max_weight, item);
    return root;
}
int preOrderTraversal(node *root, int depth)
{
    if (root == NULL)
    {
        return INT_MIN;
    }
    int current_max = INT_MIN;
    if (root->check == false)
    {
        current_max = root->value;
    }
    int left_max = preOrderTraversal(root->in_left, depth + 1);
    int right_max = preOrderTraversal(root->notin_right, depth + 1);
    int temp_max = current_max;
    if (left_max > temp_max)
        temp_max = left_max;
    if (right_max > temp_max)
        temp_max = right_max;
    return temp_max;
}
int main()
{
    int max_weight;
    scanf("%d", &max_weight);
    int item[4][3];
    for (int i = 0; i < 4; i++)
    {
        scanf("%d %d", &item[i][0], &item[i][1]);
        item[i][2] = item[i][1] / item[i][0];
    }
    for (int i = 0; i < 4 - 1; i++)
    {
        for (int j = 0; j < 4 - i - 1; j++)
        {
            if (item[j][2] < item[j + 1][2])
            {
                int temp0 = item[j][0];
                int temp1 = item[j][1];
                int temp2 = item[j][2];
                item[j][0] = item[j + 1][0];
                item[j][1] = item[j + 1][1];
                item[j][2] = item[j + 1][2];
                item[j + 1][0] = temp0;
                item[j + 1][1] = temp1;
                item[j + 1][2] = temp2;
            }
        }
    }
    node *root = buildTree(0, 0, 0, max_weight, item);
    int max = preOrderTraversal(root, 0);
    printf("%d", max);
}