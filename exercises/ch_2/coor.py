a = 0
b = 10
n = 40
h = (b - a) / n

# res = []
# for i in range(n):
#     res.append(a + i * h)

# print(res)
def gen_spaced_points(a, b, n):
    return [(a + i * h) for i in range(n + 1)]


if __name__ == "__main__":
    print(gen_spaced_points(a, b, n))
