#_*_ coding:utf-8_*
#背包问题
def knapsack(t,w):
	'''
	param t:背包总容量
	param w:物品重量列表
	'''

	n=len(w)
	stack=[]
	k=0
	while stack or k<n:
		while t>0 or k<n:
			if t>=w[k]:
				stack.append(k)
				t-=w[k]
			k+=1
		if t==0:
			print stack
		k=stack.pop()
		t+=w[k]
		k+=1
knapsack(10,[1,8,4,3,5,2])