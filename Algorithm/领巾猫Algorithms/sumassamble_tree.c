#include <stdio.h>
#include <stdlib.h>
typedef struct Node
{
    int index;
    int current_sum;
    struct Node *left;
    struct Node *right;
} Node;
int compare(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}
void buildTree(int nums[], int n, int target, Node *parent, int *count)
{
    int idx = parent->index;
    int sum = parent->current_sum;
    if (idx == n)
    {
        if (sum == target)
        {
            (*count)++;
        }
        return;
    }
    if (sum > target)
    {
        return;
    }
    if (idx > 0 && nums[idx] == nums[idx - 1])
    {
        Node *grandparent = NULL;
        if (parent->index == idx - 1)
        {
            buildTree(nums, n, target, parent, count);
            return;
        }
    }
    Node *leftNode = (Node *)malloc(sizeof(Node));
    leftNode->index = idx + 1;
    leftNode->current_sum = sum + nums[idx];
    leftNode->left = NULL;
    leftNode->right = NULL;
    parent->left = leftNode;
    buildTree(nums, n, target, leftNode, count);
    Node *rightNode = (Node *)malloc(sizeof(Node));
    rightNode->index = idx + 1;
    rightNode->current_sum = sum;
    rightNode->left = NULL;
    rightNode->right = NULL;
    parent->right = rightNode;
    buildTree(nums, n, target, rightNode, count);
}
int main()
{
    int n, target;
    scanf("%d %d", &n, &target);
    int *nums = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &nums[i]);
    }
    qsort(nums, n, sizeof(int), compare);
    Node *root = (Node *)malloc(sizeof(Node));
    root->index = 0;
    root->current_sum = 0;
    root->left = NULL;
    root->right = NULL;
    int count = 0;
    buildTree(nums, n, target, root, &count);
    printf("%d\n", count);
    return 0;
}