# kaggle_made
*Идея решения*:
1. Берем предобученную сеть
2. Выделяем признаки
3. Обучаем логистическую регрессию на выделенных признаках

P.S.
Большую часть времени контеста потратил, чтобы разобраться в pytorch lightning, однако результат получился неудовлетворительный (видимо я вызывал Trainer.fit(),
который по-умолчанию обучал всю сеть, не замораживая backbone - с этим еще не разобрался, буду признателен, если поделитесь ресурсами, где можно почитать про реальные случаи finetuning-а в lightning).

Времени оставалось мало, решение было воспроизводимым, но качество не било baseline.

К финальному решению пришел благодаря видеороликам Aladin Person на youtube.

*Pipeline*:

0. В корне создаем папку `data`, куда разархивируем тренировочный набор данных.
Итоговая структура должна иметь следующий вид:

```
data:
├── train
│   ├── ***.jpg
│   ├── ...
│   └── ***.jpg
├── test
│   ├── ***.jpg
│   ├── ...
│   └── ***.jpg
├── train.csv
└── test.csv
```
1. Убираем неподходящие данные (количество слоев != 3)

`prepare_data.ipynb`

2. Обучение (нескольких эпох) предобученной сети (брал EF-0, EF-7)

В корне предварительно создаем папку `data_features`, куда будут сохранятся feature-эмбединги.

`train.py`

3. Запуск логистической регрессии

`logreg_on_features.ipynb`
