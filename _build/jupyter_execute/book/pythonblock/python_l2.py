#!/usr/bin/env python
# coding: utf-8

# (python_l2)=
# 
# # Установка необходимых инструментов и настройка среды
# 
# Автор(ы):
# 
# - [Котенков Игорь](https://github.com/stalkermustang)
# - [Дмитрий Коржов](https://github.com/dkorzhov)
# 
# 
# ## Описание лекции
# 
# Эта лекция расскажет:
# 
# - как скачать курс себе;
# - как собрать свой экземпляр книги;
# - как скачать сотню библиотек за раз;
# - как проверить, корректно ли прошла установка.
# 
# ## Git и получение файлов курса
# 
# [`Git`](https://github.com/git-guides) - это система контроля версий. Этот инструмент часто используется для совместной разработки программ (и не только!) группой людей, каждый из которых работает над своей отдельной проблемой. Так, например, этот курс был создан с помощью `Git`, и каждый автор работал с отдельной копией, создавая набор лекций. А затем все копии собрались в одну книгу, которую вы и читаете. Для вашего удобства, а также возможности получить ("собрать" на сленге) свой экземпляр книги (если он вам зачем-то нужен -- ведь все доступно онлайн в полном объеме). К сожалению, знакомство с этим инструментов не входит в программу настоящего курса, и единственное, что вам потребуется - это скачать файлы себе для дальнейшей работы. Хороший туториал на английском можно найти по [этой ссылке](https://git-scm.com/docs/gittutorial).
# 
# Однако сейчас ограничимся лишь простой операцией: для этого надо зайти на страницу с репозиторием (папкой) [по ссылке на github](https://github.com/quantum-ods/qmlcourse) (одну из разновидностей `Git`). Затем найдите зеленую кнопку с текстом `<> Code` и кликните по ней. В открывшемся окне убедитесь, что выбрана вкладка `HTTPS`, а не `SSH` или `GitHub CLI`. В самом низу окошка будет кнопка `Download ZIP` - она нам и нужна, жмите!
# 
# ```{tip}
# Если вы проследуете инструкциям из документации по установке `Git`, то действия выше можно проделать одной командой из консоли:
# 
#     git clone https://github.com/quantum-ods/qmlcoursesdfd
# 
# Вообще `Git` - очень полезный инструмент, и рекомендуется уделить время прочтению инструкций по его установке и настройке, это пригодится в будущем.
# ```
# 
# ```{figure} /_static/pythonblock/install_l2/git_download.png
# :name: git_download
# :width: 400px
# 
# Не пугайтесь темного цвета - это просто цветовая схема в браузере. На светлой странице все, что написано выше, применимо в полном объеме.
# ```
# 
# Затем скачанный файл необходимо разархивировать любым удобным способом. В современных версиях Windows можно выбрать соответствующий пункт в выпадающем меню при клике на скачанный архив правой кнопкой мыши, на `MacOS` достаточно просто кликнуть дважды, а на `Ubuntu` ... что вы делаете на этой странице? Нужно поставить программу, если ее еще нет, а затем разархивировать по любому удобному пути:
# 
# ```
# sudo apt install unzip && unzip qmlcourse-master.zip -d <путь/к/вашей/папке/>
# ```
# 
# Теперь в папке qmlcourse-master вы увидите содержимое курса, над которым трудилась наша команда: это и картинки, и текст, и задания.
# 
# ## Что делать дальше
# 
# В этом блоке мы сделаем две вещи: соберем книгу и установим на ваш компьютер все необходимое для прохождения курса, от `Python` и его библиотек до специальных программ, которые занимаются отслеживанием этих самых библиотек. Буквально после пары строк на вашем компьютере появится сотня библиотек, которыми мы хвастались в первой лекции -- они позволят как проводить математические расчеты, так и писать программы для квантовых компьютеров. Итак, дальнейшие шаги зависят от вашей операционной системы:
# 
# ### Инструкция для Windows
# 
# Для дальнейшей работы необходимо установить менеджер пакетов для Windows - `chocolatey`. Официальная инструкция находится [по этой ссылке](https://docs.chocolatey.org/en-us/choco/setup#non-administrative-install). В данной инструкции рассмотрена установка без администраторских прав, то есть вы можете установить все программы даже на рабочий ноутбук или компьютер - дополнительных соглашений или паролей для установки не требуется.
# 
# Итак, необходимо перейти по ссылке выше и скопировать код файла `ChocolateyInstallNonAdmin.ps1`. Создайте где-нибудь (хоть на рабочем столе) пустой файл с указанным именем, вставьте в него скопированное содержимое.
# 
# 
# ```{tip}
# Важно, чтобы расширение файла было именно `ps1`. Если вы не знаете, как создавать файлы произвольных расширений, то картинка ниже поможет.
# ```
# 
# ```{figure} /_static/pythonblock/install_l2/install_windows.png
# :name: install_windows
# :width: 600px
# 
# Необходимо зайти в "Проводник" (любая папка с документами), затем перейти во вкладку "Вид", после чего поставить галочку у соответствующего пункта. Теперь при создании файла, к примеру, `my_file.abc` расширение будет `abc`, а не `txt` (хотя файл может оставаться текстовым). Такой файл можно открыть через блокнот (нажав правой кнопкой мыши и выбрав программу для запуска).
# ```
# 
# Если вы все сделали правильно, то иконка файла изменится:
# 
# ```{figure} /_static/pythonblock/install_l2/open_windows.png
# :name: open_windows
# :width: 400px
# 
# Иконка изменилась (она соответствует `PowerShell`, см. ниже). Файл можно редактировать через блокнот, как показано выше.
# ```
# 
# Поздравляем, вы создали исполняемый файл! В своем роде это скрипт, который и запустит установку. Для этого нужно открыть `PowerShell` (терминал в рамках операционной системы  Windows). Его вы можете найти очень просто - достаточно набрать название в меню "Пуск". После этого выполните команду (скопируйте, вставьте и нажмите `Enter`):
# 
# ```
# Set-ExecutionPolicy Bypass -Scope Process -Force;
# ```
# 
# После этого перетащите в терминал файл (прямо мышкой с рабочего стола) и отпустите левую кнопку: в терминале должен появиться путь до файла. Снова выполните команду (`Enter`). Дождитесь окончания установки.
# 
# И, наконец, закончим установкой вспомогательных инструментов. Запустите команду
# 
# ```
# choco install make curl miniconda3
# ```
# 
# Все сопроводительное обеспечение установлено, осталось непосредственно собрать окружение и собственный экземпляр книги - в этом нам поможет запуск двух команд (**но перед этим нужно перейти в директорию со скачанным курсом** . Для этого нужно воспользоваться командой `cd` (change directory, сменить директорию) и указать путь к директории. Самый простой способ это сделать - перетащить папку из окна проводника с файлами (или рабочего стола) в терминал: путь подставится автоматически. Пример правильной команды: `cd /home/ivan.petrov/Downloads/course/qmlcourse-master/`):
# 
# ```
# make install-windows
# make build-windows
# ```
# 
# ### Инструкция для MacOS
# 
# Для дальнейшей работы необходимо установить менеджер пакетов для MacOS - `brew`. [По ссылке](https://brew.sh/index_ru) вы найдете инструкцию и более детальное описание, чем является `brew`. Вам необходимо выполнить в терминале (который доступен как приложение на экране со всеми приложениями -- просто начните вводить `term` и поиск подскажет верный вариант) команду:
# 
# ```bash
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# ```
# 
# ```{attention}
# Скорее всего, для выполнения этой команды, нужно задействовать команду `sudo`, это вполне нормальная история.
# ```
# 
# Затем выполнить простую команду (тоже в терминале), предварительно перейдя в папку с курсом, которую вы извлекли из архива. Для этого нужно воспользоваться командой `cd` (change directory, сменить директорию) и указать путь до папки. Самый простой способ это сделать - перетащить папку из окна Finder в терминал: путь подставится автоматически. Например:
# 
# ```bash
# cd /home/ivan.petrov/Downloads/course/qmlcourse-master/
# ```
# 
# И из этой папки вызовите последовательность из двух команд:
# 
# ```bash
# make install-macos-latest
# make build-linux-macos
# ```
# 
# ### Инструкция для Ubuntu
# 
# Эта часть подойдет также для всех "родственных" систем: Debian, Linux Mint, Deepin Linux (который можно встретить на ноутбуках компании Huawei, к примеру) и многие другие системы, использующие менеджер пакетов `apt`.
# Для начала необходимо установить дополнительные программы. Выполните код ниже в терминале:
# 
# ```bash
# sudo apt-get update
# sudo apt-get install build-essential -y
# sudo apt-get install git wget -y
# ```
# 
# ```{attention}
# При использовании sudo терминал может попросить вас ввести пароль для установки и обновления пакетов программ. Это абсолютно нормально.
# ```
# 
# Затем выполнить простую команду (тоже в терминале), предварительно перейдя в папку с курсом, которую вы извлекли из архива. Для этого нужно воспользоваться командой `cd` (change directory, сменить директорию) и указать путь до папки. Самый простой способ это сделать - перетащить папку из окна Finder в терминал: путь подставится автоматически. Например:
# 
# ```bash
# cd /home/ivan.petrov/Downloads/course/qmlcourse-master/
# ```
# 
# И из этой папки вызовите последовательность из двух команд:
# 
# ```bash
# make install-ubuntu-latest
# make build-linux-macos
# ```
# 
# ```{hint}
# Первая команда установит `Python`, [`poetry`](https://python-poetry.org) - это еще один менеджер, который отвечает за управление библиотеками `Python` и их версиями, а также сами библиотеки, необходимые для прохождения курса; вторая – соберет [jupyter book](https://jupyterbook.org/intro.html), то что по ссылке [https://quantum-ods.github.io/qmlcourse/book/index.html](https://quantum-ods.github.io/qmlcourse/book/index.html) (так что можете не выполнять если не хотите, трава и так будет расти)
# ```
# 
# ## Проверка корректности установки
# 
# Чтобы убедиться в том, что все отработало штатно, и у вас полный комплект, запустим очень мощный инструмент для работы в машинном обучении - `Jupyter Notebooks`. С ним вы познакомитесь ближе в следующей лекции, а пока введите в терминале команду
# 
# ```bash
# poetry run jupyter notebook
# ```
# 
# ```{hint}
# Для того, чтобы запустить что-либо в текущем окружении, вам нужно начинать писать команду с `poetry run` (как в примере выше), ибо все ваши установленные библиотеки (которые в свою очередь устанавливаются как `poetry add <name_lib>`) не будут подгружаться
# ```
# 
# В терминале должна появиться надпись о том, что запущен сервер, а также будет предоставлено две или три ссылки.
# 
# ```{figure} /_static/pythonblock/install_l2/jupyter_console.png
# :name: jupyter_console
# :width: 400px
# 
# Примерно так будет выглядеть терминал со ссылками.
# ```
# 
# Перейдите по одной из ссылок, в которых есть слово `token` -- они начинаются с `http://localhost` или `http://127.0.0.1`. В вашем браузере откроется страница с обзором папки, из которой был запущен `Jupyter Notebook`. Если вы в дальнейшем хотите хранить все свои результаты в другой папке, то перед запуском команды с помощью уже указанной инструкции в терминале `cd <путь/до/директории>` перейдите к ней.
# 
# Для создания нового файла -- блокнота, как его еще называют (почему? узнаете в следующей лекции) -- кликните по кнопке `new` в правом верхнем углу, а затем -- по `Python 3` (возможно будет какая-нить приставка, это штатно).
# 
# ```{figure} /_static/pythonblock/install_l2/jupyter_create.png
# :name: jupyter_create
# :width: 600px
# 
# Таким образом можно создать новый файл с кодом.
# ```
# 
# Для завершения проверки скопируйте код ниже в тетрадку в браузере, а затем нажмите `CTRL + Enter` или `command + Enter` (это заставит код выполниться, подробнее дальше в курсе). Если увидите график - то все в полном порядке!
# 
# ```python3
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
# 
# # Fixing random state for reproducibility
# np.random.seed(19680801)
# 
# # Compute pie slices
# N = 20
# θ = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = 10 * np.random.rand(N)
# width = np.pi / 4 * np.random.rand(N)
# colors = plt.cm.viridis(radii / 10.)
# 
# ax = plt.subplot(111, projection='polar')
# ax.bar(θ, radii, width=width, bottom=0.0, color=colors, alpha=0.5)
# 
# plt.show()
# ```
# 
# Не переживайте, этот код не нужно разбирать сейчас -- мы просто убеждаемся, что все работает согласно задумке. Если что-то не так, пересмотрите все ли сделали согласно инструкции; если да и не воспроизводится, то задавайте вопрос в [канале курса](https://opendatascience.slack.com/archives/CEH3VJCRJ)
# 
# ## Что мы узнали в лекции
# 
# - что такое `Git` и как скачать проект на локальный компьютер;
# - как становить необходимые библиотеки, `Python` и прочие прелести;
# - как запустить `Jupyter Notebook` для набора кода;
