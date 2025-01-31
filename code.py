import six
import functools


class Solution:
    def minNumber(self, nums) -> str:
        def sort_rule(x, y):
            a, b = x + y, y + x
            if a > b:
                return 1
            elif a < b:
                return -1
            else:
                return 0

        strs = [str(num) for num in nums]
        # strs.sort(key=functools.cmp_to_key(sort_rule))
        strs = sorted(strs,key=functools.cmp_to_key(sort_rule))
        return ''.join(strs)

s = Solution()
print(s.minNumber([1,3,4,0,2]))