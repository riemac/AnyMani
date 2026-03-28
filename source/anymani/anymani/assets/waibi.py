# ❌ 错误示范（普通类）
class Foo:
    def __init__(self, items=[]):  # 所有实例共享同一个列表！
        self.items = items

a = Foo()
b = Foo()
a.items.append(1)
print(b.items)  # [1]  ← b 也被污染了！
