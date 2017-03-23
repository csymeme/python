# -*- coding:utf-8 -*-
 
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class TreePrinter:
    def printTree(self, root):
        # write code here
        if root == None:
            return []
        stack = [root, '#']
        res = []
        temp = []
        while stack:
            cur = stack.pop(0)
            if cur == '#':
                res.append(temp)
                temp = []
            else:
                temp.append(cur.val)
                if cur.left:
                    stack.append(cur.left)
                if cur.right:
                    stack.append(cur.right)
                if stack[0] == '#':
                    stack.append('#')
        return res[:-1]