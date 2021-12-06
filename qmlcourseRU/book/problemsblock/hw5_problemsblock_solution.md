---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(hw5_solution)=

# Задание 5. Модель Изинга, оптимизация. Решение

Автор – Юрий Кашницкий. [Ссылка на тест](https://ods.ai/tracks/qmlcourse/blocks/8d9e112a-a991-4d7c-ab46-73a2edc21fd3)

_Замечания:_
 - _Правильные варианты ответа помечены значком **[x]**_
 - _Изначальная формулировка 2-го варианта ответа в 3-м вопросе отличалась_

**1. Выберите все верные утверждения:**

- **[x]** Теория Бора возникла как попытка объяснить расхождение теории и эксперимента, в котором наблюдался дискретный спектр излучения атома водорода в видимом диапазоне
- Один из постулатов Бора заключается в том, что импульс электрона пропорционален его релятивистской массе
- **[x]** Предсказанные теорией Бора значения энергии электрона в атоме водорода пропорциональны $\frac{1}{n^2}, n \in \mathbb{Z}$
- **[x]** Одна из слабостей теории Бора – в том, что уже для атома гелия наблюдаемый спектр не согласуется с предсказанным теоретически

**Комментарий:** См. [лекцию курса](https://semyonsinchenko.github.io/qmlcourse/_build/html/book/problemsblock/quantchembasic.html) c введением в квантовую химию и теорию Бора.

**2. Выберите все верные утверждения:**

- **[x]** NP-полные задачи -- это подмножество NP-задач
- **[x]** Упрощенная задача коммивояжера, в которой надо найти путь не длиннее некоторой константы, – это NP-полная задача
- Задача о максимальном разрезе в графе решается точно эффективным алгоритмом с полиномиальной сложностью, в частности, реализованным в `NetworkX`
- Задача о выделении сообществ в графах -- NP-полная, но не NP-трудная

**Комментарий:** См. [лекцию курса](https://semyonsinchenko.github.io/qmlcourse/_build/html/book/qcalgo/quantum_algorithms_overview.html#id2) c обзором квантовых алгоритмов, раздел "Классификация задач по временной сложности".


**3. Рассмотрим [пример из лекции](https://semyonsinchenko.github.io/qmlcourse/_build/html/book/problemsblock/copt.html#id12), где реализуется жадный алгоритм для решения задачи о рюкзаке. Пусть теперь вместимость рюкзака -- 22, а вес и стоимости предметов задаются массивом:**

	items = [(4, 2500), (9, 1950), (10, 3500), (21, 6700), (17, 6100), (3, 1800), (27, 8300)]

**Реализуйте жадный алгоритм, как описано в лекции (в-общем, ровно тот же код сойдет), а также найдите точное решение задачи о рюкзаке с помощью полного перебора. Отличаются ли два решения, и какое наполнение рюкзака будет оптимальным? Выберите все верные варианты:**

- **[x]** Некоторых предметов вообще не может быть в решении -- из-за их веса, превышающего вместимость рюкзака
- **[x]** Жадный алгоритм нашел решение: 5 предметов со стоимостью 2500. Оно не соответствует оптимальному заполнению рюкзака
- Жадный алгоритм нашел неоптимальное решение: 7 предметов со стоимостью 1800
- **[x]** Решение, найденное жадным алгоритмом, "отстает" от полученного полным перебором более чем на 1000 (по значению целевой функции, т.е. суммарной стоимости предметов в рюкзаке)
- **[x]** Оптимальное заполнение рюкзака -- это комбинация предметов с весом 3 и предметов с весом 4

**Решение**

Решение с помощью жадного алгоритма получаем ровно так же, как в [примере из лекции](https://semyonsinchenko.github.io/qmlcourse/_build/html/book/problemsblock/copt.html#id12).

```{code-cell} ipython3
from itertools import combinations
```

```{code-cell} ipython3
capacity = 22
items = [(4, 2500), (9, 1950), (10, 3500), (21, 6700), (17, 6100), (3, 1800), (27, 8300)]
```

```{code-cell} ipython3
items_and_score = sorted(
    [(it[0], it[1], it[1] / it[0]) for it in items],
    key=lambda x: x[2],
    reverse=True
)

print("Items, sorted by relative cost:")
for it in items_and_score:
    print(f"Weight: {it[0]}\tCost: {it[1]}\tRelative Cost: {it[2]}")
```

```{code-cell} ipython3
capacity = 22
items = [(4, 2500), (9, 1950), (10, 3500), (21, 6700), (17, 6100), (3, 1800), (27, 8300)]
```

```{code-cell} ipython3
solution = []
w = capacity
min_weight = min([it[0] for it in items_and_score])

while True:
    if w < min_weight:
        break
    else:
        cand = [it for it in items_and_score if it[0] <= w][0]
        solution.append(cand)
        w -= cand[0]

final_score = sum([it[1] for it in solution])
final_weight = sum([it[0] for it in solution])

print(f"Final score: {final_score}")
print(f"Total weight of items: {final_weight}")
print(f"Solution: {solution}")
```

Теперь brute force решение, то есть полный перебор.

Модифицируем список предметов, дублируя эти предметы до тех пор, пока их вес -- меньше вместимости рюкзака. Например, первый предмет с весом 4 повторили еще 4 раза, теперь в списке предметов -- 5 кортежей вида `(4, 2500)`, потому что в рюкзак вместимостью 22 можно вместить не более 5 предметов веса 4. _Конечно, можно было написать кусок кода, чтоб модифицировать список предметов, но для такого простого примера мы просто сделали это вручную._

```{code-cell} ipython3
items = [(4, 2500), (4, 2500), (4, 2500), (4, 2500), (4, 2500),
         (9, 1950), (9, 1950),
         (10, 3500), (10, 3500),
         (17, 6100),
         (3, 1800), (3, 1800), (3, 1800), (3, 1800), (3, 1800), (3, 1800), (3, 1800)]
```

Перебираем все варианты, отбираем только удовлетворяющие ограничению по весу и сохраняем максимальное значение суммарной стоимости предметов в рюкзаке.

```{code-cell} ipython3
best_score  = 0
for i in range(1, 8):
    for comb in combinations(items, i):
        weight = sum(el[0] for el in comb)
        if  weight <= capacity:
            score = sum(el[1] for el in comb)
            if score > best_score:
                best_score = score
                print(f"Weight: {weight}, score: {score}")
                print(comb)
```

Жадное решение: 5 предметов с весом 4. Оно неоптимально, отстает от лучшего более чем на 1000 (12500 vs. 13600). Оптимальное заполнение рюкзака -- это комбинация предметов с весом 3 и предметов с весом 4.


**4. Рассмотрим [пример из лекции](https://semyonsinchenko.github.io/qmlcourse/_build/html/book/problemsblock/ising.html#numpy) с реализацией модели Изинга на чистом `Numpy`. При каких значениях соотношения внешнего поля к константе обменного взаимодействия $\frac{h}{J}$ ориентация всех спинов будет одинаковой? Выберите все подходящие варианты:**

- 1.5
- 2
- **[x]** 2.25
- **[x]** 2.5
- **[x]** 3
- **[x]** 50

**Решение**

Переиспользуем код [примера из лекции](https://semyonsinchenko.github.io/qmlcourse/_build/html/book/problemsblock/ising.html#numpy).

```{code-cell} ipython3
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sl

np.set_printoptions(precision=2)
```

```{code-cell} ipython3
def sigmaz_k(k: int, n: int) -> (sparse.csr_matrix):
    left_part = sparse.eye(2 ** k)
    right_part = sparse.eye(2 ** (n - 1 - k))

    return sparse.kron(
        sparse.kron(
            left_part,
            sparse.csr_matrix(np.array([[1, 0,], [0, -1,],]))
        ),
        right_part
    )

def ising(j: float, h: float, n: int) -> (sparse.csr_matrix):
    res = sparse.csr_matrix((2 ** n, 2 ** n), dtype=np.complex64)

    for i in range(n - 1):
        res += j * sigmaz_k(i, n) * sigmaz_k(i + 1, n)
        res -= h * sigmaz_k(i, n)

    res -= h * sigmaz_k(n - 1, n)

    return res

def probs2bit_str(probs: np.array) -> (str):
    size = int(np.log2(probs.shape[0]))
    bit_s_num = np.where(probs == probs.max())[0][0]

    s = f"{bit_s_num:b}"
    s = "0" * (size - len(s)) + s

    return s

def external_field(j: float, h: float, n: int) -> (None):
    op = ising(j, h, n)
    solution = sl.eigs(op, which="SR", k=1, return_eigenvectors=True)

    probs = solution[1] * solution[1].conj()
    print(f"Energy: {np.real(solution[0][0]).round(1)}. Spin order: {probs2bit_str(probs)}")

```

Значение константы обменного взаимодействия $J$ не меняем (в функции `external_field` -- аргумент $j=1$), а коэффициент внешнего поля (в функции `external_field` -- аргумент $h$) варьируем:

```{code-cell} ipython3
for h in [1.5, 2, 2.25, 2.5, 3, 50]:
    external_field(j=1, h=h, n=10)
```

Начиная с $h=2.25$, ориентация всех спинов одинаковая.
