import numpy as np


# sumtree按照元素值在td_error总和占地比例来决定被采样的概率
# 保证值更大的元素被选到概率更大

class SumTree:
    def __init__(self, td_error):
        self.td_error = td_error
        self.sumtree = self.build_tree()

    def build_tree(self):
        ans = []
        t_sum = self.td_error
        ans.append(t_sum)
        # 逐层求和，利用numpy的reshape,sum来求相邻两项的和
        while len(t_sum) > 1:
            # print(t_sum)
            if len(t_sum) % 2 == 1:
                t_sum = np.append(t_sum, [0])
            # 每两个求和
            t_sum = t_sum.reshape(len(t_sum) // 2, 2).sum(axis=1)
            ans.append(t_sum)

        return ans[::-1]

    def sample(self, s):
        if s > self.td_error.sum():
            return len(self.td_error)
        if s < 0:
            return 0

        level = 0
        idx = 0
        while level < len(self.sumtree):
            # print(self.sumtree[level][idx])
            # print(idx)
            left = idx * 2
            right = left + 1
            # s肯定小于顶点，第一次肯定走if
            if s <= self.sumtree[level][left]:
                idx = left
            else:
                idx = right
                s -= self.sumtree[level][left]
            level += 1
        return idx

    def gen_batch_index(self, batch_num):
        td_error_sum = self.td_error.sum()
        # 这里增加一个采样基准值，防止采样过于集中
        sample_base_value = np.linspace(0, (batch_num - 1) / batch_num * td_error_sum, batch_num)
        # np.random.rand 返回0-1值，-0.5使其均值为0
        sample_random_value = (np.random.randn(batch_num) - 0.5) * td_error_sum / batch_num
        sample_value = sample_base_value + sample_random_value
        # sample_value = np.random.uniform(0, td_error_sum, batch_num)
        # print(sample_value)
        return [self.sample(v) for v in sample_value]


if __name__ == '__main__':
    td_error = [3, 10, 12, 4, 1, 2, 8, 2]
    # t = sum_tree(np.array(range(100))[np.random.permutation(range(100))])
    sum_tree = SumTree(np.array(td_error))
    # [level[0:9] for level in t.sumtree]
    # [sum(level) for level in t.sumtree]
    # 貌似固定的td_error顺序，输出十固定的，输入之前需要先
    t = np.array(sum_tree.gen_batch_index(1000))
    hist = [len(t[t == i]) for i in range(len(td_error))]
    hist = np.array(hist)
    hist /= hist.min()