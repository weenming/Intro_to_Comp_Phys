solutions = set()

def point24(numbers):
    global solutions
    if len(numbers) == 1:
        if abs(eval(numbers[0]) - 24) < 0.00001:
            solutions.add(numbers[0])
    else:
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                rest_numbers = [x for p, x in enumerate(numbers) if p != i and p != j]
                for op in "+-*/":
                    if op in "+-*" or eval(str(numbers[j])) != 0:
                        point24(["("+ str(numbers[i]) + op + str(numbers[j]) + ")"] + rest_numbers)
                    if op == "-" or (op == "/" and eval(str(numbers[i])) != 0):
                        point24(["("+ str(numbers[j]) + op + str(numbers[i]) + ")"] + rest_numbers)


point24([3, 7, 9, 13]) # 测试用例
print("Found %d solutions." %len(solutions))
for i, s in enumerate(solutions):
    print("%d: %s = 24" %(i+1, s))